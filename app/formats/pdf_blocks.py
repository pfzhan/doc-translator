"""Text blocks and the helpers that both reading and writing pages use: line joining,
sentence ends, glyph ink and math-font checks."""

import contextvars
import functools
import hashlib
import math
import os
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import pymupdf

from .pdf_flow import (
    STYLE_MARK_RE,
    emphasis_of,
)
from .pdf_layout import (
    PageGeometry,
)
from .pdf_runs import (
    Run,
)

MATH_FONT_RE = re.compile(r"CMMI|CMB?SY|CMEX|MSBM|MSAM|Math|Symbol|StandardSym|STIX|Cambria Math", re.I)
# Allowlist of exact math font names (a match means formula directly)
# Math fonts that accompany a text font have Math in their names (XCharterMath, CambriaMath); the math fonts of newtx
# and pxfonts have short names such as txmi and txsy.
# The math italic of newtxmath is called NewTXMI or NewTXMI7, cochMI with Cochineal, and the math roman is cochMRM.
# Euler script and Fraktur (EUSM, EUFM) and blackboard bold (bbold, dsfont, bbm) are used only in formulas
MATH_FONT_PRECISE_RE = re.compile(
    r"Asana|FiraMath|STIX|TeXGyre.*Math|XITSMath|LibertinusMath|MathJax|Cambria Math|LatinModern.*Math"
    r"|(?-i:Math(?![a-z]))|(?:^|\+)n?[tp]x(?:mi|sy|ex)|(?:^|\+)(?:New[TP]X|coch)B?(?:MI|MRM)\d*$"
    r"|(?:^|\+)(?:EU[SF]M|EURM|EUEX|BBOLD|bbm|dsrom|dsss|DSSerif)",
    re.I,
)
# Blocklist of common text fonts (takes precedence over the heuristic; Greek letters and symbols set in them are
# recognized by character-level rules)
TEXT_FONT_RE = re.compile(
    r"Times|Arial|Calibri|Minion|Palatino|\bCMR|Charter|Georgia|Helvetica|Verdana|Roboto|Lato|"
    r"Open ?Sans|Source ?Sans|Noto(?!.*Math)|Libertine(?!.*Math)|Garamond|Baskerville|Bookman|"
    r"Courier|Consolas|Menlo|Monaco|Ubuntu|DejaVu|Liberation|FreeSerif|FreeSans|Nimbus|"
    r"Trebuchet|Candara|Constantia|Franklin|Gill|Lucida|Segoe|Optima|Futura|Avenir|Univers|"
    r"Myriad|Frutiger|\bDIN\b|Proxima|Museo|\bPT S|Merriweather|Lora|Crimson|Playfair|Cormorant|"
    r"Spectral|Inter|Work Sans|IBM Plex|Charis|Gentium|Doulos|Andika|"
    r"PingFang|SimSun|SimHei|Songti|Heiti|Kaiti|FangSong|SourceHan|WenQuanYi|YaHei|YouYuan|LiSu|"
    r"STFangsong|STHeiti|STKaiti|STSong|STXihei|Yuanti|Hiragino|Ryumin|GothicBBB|Kozuka|IPA|"
    r"Takao|Sazanami|Motoya|Yomogi|BIZ |M PLUS|GenShin|GenRyu|NotoSansCJK|NotoSerifCJK",
    re.I,
)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_PAREN_ITEM_RE = re.compile(r"^\([a-zA-Z]\)\s+")
# (ii), (iii) followed by a capital letter start another item of a proof. Followed by lowercase, they are an
# enumeration inside a sentence ((iii) the arm must ...)
_ROMAN_ITEM_RE = re.compile(r"^\((?:[ivx]{2,4}|[IVX]{2,4})\)\s+[A-Z]")
# List marker at the start of a line: a bullet symbol (space optional), a dash (space required, so that hyphenated
# words are not matched), or a number or letter
BULLET_RE = re.compile(r"^\s*(?:[•◦▪‣·○■□►»]\s*|[-–—*]\s+|\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+)")
_MARK_ITEM_RE = re.compile(r"^\s*(?:[•◦▪‣·○■□►»]|[-–—*]\s)")
# Semicolons are excluded: they are among the joining punctuation below, so a hard line break after one does not
# split the paragraph
SENT_ENDS = ("。", "！", "？", "：", "…", ".", "!", "?", ":")
# Closing quotes and brackets after sentence-final punctuation (found it.), he said."). When they do not follow
# sentence-final punctuation, as in a citation year (Vilain et al., 1995) or code r"cat|dog", the sentence has not ended
_SENT_CLOSERS = "\"'”’」』)）"


# An end-of-proof mark at the end of a paragraph (□, ∎; extracted from a Symbol font as \x00 or a private use
# character) is not body text, and the period comes before it.
# \x01 and \x02 are formula sentinel delimiters and must not be stripped
_END_MARK_RE = re.compile(r"[\s\x00\ue000-\uf8ff□■∎◻◼]+$")


def _ends_sentence(text: str) -> bool:
    """Ends with sentence-final punctuation or with a formula. An end-of-proof mark at the end of a paragraph is
    ignored."""
    text = _END_MARK_RE.sub("", text)
    return text.rstrip(_SENT_CLOSERS).endswith(SENT_ENDS) or _ends_in_call(text)


def _ends_in_call(text: str) -> bool:
    """The opening bracket that matches the final closing bracket directly follows a letter or digit: softmax(hW),
    P(B|A), count(... doc). The text ends in a formula, and a where or with below a displayed formula is not joined to
    it. A bracket in body text has a space before it."""
    if not text.endswith(")"):
        return False
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                return index > 0 and text[index - 1].isascii() and text[index - 1].isalnum()
    return False


@dataclass(frozen=True)
class Leader:
    """Leader and page number of a table of contents entry. The leader is erased together with the title and the page
    number stays in place; when writing, dots are added leftward from dot_x (the rightmost dot before the page number)
    at spacing pitch until they reach the end of the translation. dots are the dots to erase (including the span set
    together with the title). With a page number and no leader, pitch is 0 and no dots are added."""

    number: pymupdf.Rect
    dot_x: float
    pitch: float
    baseline: float
    size: float
    color: int
    dots: tuple[dict, ...] = ()


# Gap from the bottom of one line box to the top of the next at normal line spacing, in units of font size. Used
# when there is too little body text to measure it
_LINE_GAP = 0.1


@dataclass
class TextBlock:
    page: int
    rect: pymupdf.Rect
    line_rects: list[pymupdf.Rect]
    text: str
    size: float
    color: str
    bold: bool
    # Original spans of each line (including whitespace spans): the text sent for translation rebuilt from them
    # matches the original character for character
    span_lines: list[list[dict]] = None
    # Ordered runs after placeholder rebuilding. Empty during extraction, filled in before translation.
    runs: list[Run] = field(default_factory=list)
    # Cell of a ruled table. Writing must not go outside it; blocks in different cells must not be merged into one
    # paragraph.
    clip: tuple[float, float, float, float] | None = None
    # Marginal note outside the text area. Translated on its own, never merged with body text
    side_note: bool = False
    # Leader and page number at the end of a table of contents entry (see _cut_leaders). The block holds only the
    # title; the leader and page number stay on the original page
    leader: Leader | None = None
    # Gap between two adjacent body lines in this document (see _line_gap). When deciding whether the next line
    # continues the block, only the spacing beyond this gap counts
    line_gap: float = _LINE_GAP
    # Body font size of this document (see _body_size). 0 means not measured
    body_size: float = 0.0
    # A standalone entry: an entry on an index page ending with a page number (see _index_page), or a cell of a word
    # grid (see _grid_cells). Never merged with the entries above or below
    entry: bool = False
    # A cell of a word grid. Like a figure, a word grid is skipped when looking for body text at the top and bottom of
    # a column; index entries still occupy the top and bottom of a column
    grid: bool = False
    # On an index page at the back of the book (see _index_page)
    index: bool = False


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        out, line = _carry_marks(out, line)
        if out.endswith("-") and re.match(r"[a-zA-ZÀ-ÿ]", line):
            # Rejoin a word hyphenated at the line end. A hyphen after a digit belongs to the name (Claude-3.7-|Sonnet)
            # and is kept. Lowercase followed by uppercase (Distill-|Qwen, Mc-|Callum) depends on the whole text: it is
            # a hyphenation break only if the joined word occurs elsewhere
            kept = out[-2:-1].isdigit() or (
                out[-2:-1].islower() and line[0].isupper() and _camel_join(out, line) not in _CAMEL_WORDS.get()
            )
            out = out + line if kept else out[:-1] + line
        elif out.endswith("-") and out[-2:-1].isalpha() and line[:1].isdigit():
            out += line  # Model name broken at a hyphen (VE-|8, GPT-|4): the hyphen stays, no space between
        elif CJK_RE.search(out[-1]) or CJK_RE.search(line[0]):
            out += line
        elif line[:1] in ")]}>，。；：、）】」.,":
            out += line  # A closing bracket or period after a line break attaches to the formula, no space before it
        elif _URL_TAIL_RE.search(out) and _URL_PART_RE.match(line):
            out += line  # URL broken at / (https://github.com/ | tensorflow/tensor2tensor), no space between
        else:
            out += " " + line
    return out


# Words in the whole text with a capital letter inside (McCallum, PropBank). When TeX hyphenates before a capital
# letter, the hyphen is removed when rejoining
_CAMEL_WORDS: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar("camel_words", default=frozenset())


def _camel_join(out: str, line: str) -> str:
    left = re.search(r"[A-Za-z]+(?=-$)", out)
    right = re.match(r"[A-Za-z]+", line)
    return (left.group() if left else "") + (right.group() if right else "")


def _carry_marks(out: str, line: str) -> tuple[str, str]:
    """Marks closed at the end of the previous line and reopened unchanged at the start of the next are removed on both
    sides, so the marks continue from the previous line: {i}confer-{/i} | {i}ence{/i} becomes {i}conference{/i}. Only
    then can the hyphenation hyphen be recognized."""
    closers = re.search(r"(?:\{/[a-z]\})+$", out)
    openers = re.match(rf"(?:{STYLE_MARK_RE.pattern})+", line)
    if not closers or not openers or closers.start() == 0 or openers.end() == len(line):
        return out, line
    stack: list[str] = []
    for tag in STYLE_MARK_RE.findall(out[: closers.start()]):
        if tag.startswith("{/"):
            if stack and stack[-1][1] == tag[2]:
                stack.pop()
        else:
            stack.append(tag)
    closed = re.findall(r"\{/([a-z])\}", closers.group())
    if len(closed) > len(stack):
        return out, line
    reopened = stack[len(stack) - len(closed) :]
    if [tag[1] for tag in reversed(reopened)] != closed or "".join(reopened) != openers.group():
        return out, line
    return out[: closers.start()], line[openers.end() :]


# The line ends at a URL separator, and the first chunk of the next line also contains a URL separator: that chunk is
# the second half of the URL
# The break can also fall after the scheme (http: | //www.ling.ohio-state.edu/)
_URL_TAIL_RE = re.compile(r"(?:(?:https?://|www\.)\S*[/._\-#?=&~]|\bhttps?:)$")
_URL_PART_RE = re.compile(r"//\w|[\w%~-]+(?:/|[/._#?=&][\w%~/._#?=&-]*[\w/])")


def _list_item(text: str) -> bool:
    """a., (a) and (ii) followed by a capital letter are list items. A Q. after a line break is excluded by the
    caller."""
    stripped = text.lstrip()
    return bool(BULLET_RE.match(text) or _PAREN_ITEM_RE.match(stripped) or _ROMAN_ITEM_RE.match(stripped))


_INK_PAD = 0.6


def _attach_ink(page: pymupdf.Page, data: dict) -> None:
    """Record for each span the ink box measured from the glyph outlines (span["ink"]), matched by text and origin.

    The default get_text box comes from the font's ascender and descender, independent of the glyphs. CMSY and CMEX
    reserve their ascender and descender for large brackets and radicals, so the box of a single arrow can cover the
    lines above and below."""
    accurate = page.get_text("dict", flags=_TEXT_FLAGS | pymupdf.TEXT_ACCURATE_BBOXES)
    found: dict[tuple[str, float, float], tuple[float, float, float, float]] = {}
    for block in accurate["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                key = (span["text"], round(span["origin"][0], 1), round(span["origin"][1], 1))
                found[key] = tuple(span["bbox"])
    fonts: dict[str, list[tuple[bytes, bytes]]] | None = None
    glyphs: dict[tuple[str, float, float], list[tuple[int, float, float]]] | None = None
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line["spans"]:
                ink = found.get((span["text"], round(span["origin"][0], 1), round(span["origin"][1], 1)))
                if ink is None:
                    continue
                # For glyphs that cannot be measured (the radical of txsys) the accurate box equals the default box;
                # draw them alone with the embedded font and measure. Only ink outside the font box means the font box
                # is wrong, and then the ink replaces it; ink inside the box leaves it unchanged
                if ink == tuple(span["bbox"]) and span["text"].strip() and _font_is_math(str(span.get("font") or "")):
                    if fonts is None:
                        fonts = _embedded_fonts(page)
                        glyphs = _char_origins(page)
                    key = (span["text"], round(span["origin"][0], 1), round(span["origin"][1], 1))
                    drawn = _drawn_ink(span, fonts.get(str(span.get("font") or ""), []), (glyphs or {}).get(key, []))
                    if drawn is not None and (drawn[1] < ink[1] - _INK_PAD or drawn[3] > ink[3] + _INK_PAD):
                        ink = drawn
                        span["ink_drawn"] = True
                span["ink"] = ink


def _embedded_fonts(page: pymupdf.Page) -> dict[str, list[tuple[bytes, bytes]]]:
    """Font files embedded in the page with their hashes, grouped by name without the subset prefix. Fonts whose file
    cannot be extracted, such as Type 3, are left out."""
    out: dict[str, list[tuple[bytes, bytes]]] = {}
    for xref, _ext, _kind, basefont, *_rest in page.get_fonts():
        try:
            buffer = page.parent.extract_font(xref)[3]
        except (RuntimeError, ValueError):
            continue
        if buffer:
            out.setdefault(basefont.split("+", 1)[-1], []).append((hashlib.sha1(buffer).digest(), buffer))
    return out


def _char_origins(page: pymupdf.Page) -> dict[tuple[str, float, float], list[tuple[int, float, float]]]:
    """Glyph ID, baseline and character width of each character of each span, keyed by the span's text and origin.
    Whitespace and characters without a matching glyph ID are left out. The origin of a span is that of its first
    character: the space and the radical in " √" are not on the same baseline."""
    gids = {
        (round(char[2][0], 1), round(char[2][1], 1)): char[1]
        for trace in page.get_texttrace()
        for char in trace["chars"]
        if char[1] >= 0
    }
    found: dict[tuple[str, float, float], list[tuple[int, float, float]]] = {}
    for block in page.get_text("rawdict", flags=_TEXT_FLAGS).get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars") or []
                text = "".join(char["c"] for char in chars)
                key = (text, round(span["origin"][0], 1), round(span["origin"][1], 1))
                found[key] = [
                    (gid, char["origin"][1], char["bbox"][2] - char["bbox"][0])
                    for char in chars
                    if not char["c"].isspace()
                    and (gid := gids.get((round(char["origin"][0], 1), round(char["origin"][1], 1)))) is not None
                ]
    return found


def _drawn_ink(
    span: dict,
    fonts: list[tuple[bytes, bytes]],
    glyphs: list[tuple[int, float, float]],
) -> tuple[float, float, float, float] | None:
    """Top and bottom of the span from the ink of glyphs drawn alone; left and right come from the original box. Each
    character is placed on its own baseline. Glyphs are drawn by the glyph IDs used on the page: large delimiters come
    in several sizes, and the wide tilde of CMEX cannot be found in the font by Unicode. A font name can have several
    subsets; one whose character widths do not match is the wrong one."""
    size = float(span.get("size") or 0)
    if size <= 0 or not glyphs:
        return None
    for digest, buffer in fonts:
        tops: list[float] = []
        bottoms: list[float] = []
        for gid, base, width in glyphs:
            extent = _glyph_extent(digest, buffer, gid)
            if extent is None or abs(extent[2] * size - width) > max(0.3, width * 0.08):
                break
            tops.append(base + extent[0] * size)
            bottoms.append(base + extent[1] * size)
        else:
            x0, _y0, x1, _y1 = span["bbox"]
            return (x0, min(tops), x1, max(bottoms))
    return None


# Top and bottom of glyph ink, keyed by the hash of the font file and the glyph ID
_GLYPH_EXTENTS: dict[tuple[bytes, int], tuple[float, float, float, float, float] | None] = {}


def _glyph_extent(digest: bytes, buffer: bytes, gid: int) -> tuple[float, float, float, float, float] | None:
    """Top and bottom of the glyph ink (positive below the baseline), character width, and left and right of the ink,
    in units of font size. None when the font has no such glyph or the glyph draws no ink."""
    key = (digest, gid)
    if key not in _GLYPH_EXTENTS:
        _GLYPH_EXTENTS[key] = _measure_glyph(buffer, gid)
    return _GLYPH_EXTENTS[key]


def _measure_glyph(buffer: bytes, gid: int) -> tuple[float, float, float, float, float] | None:
    """Draw the glyph alone by glyph ID at size 100 and measure the top and bottom of the ink, the character width, and
    the left and right of the ink (distance from the origin)."""
    mupdf = pymupdf.mupdf
    try:
        font = mupdf.fz_new_font_from_buffer(None, mupdf.fz_new_buffer_from_copied_data(buffer), 0, 0)
        if not 0 < gid < font.m_internal.glyph_count:
            return None
        advance = mupdf.fz_advance_glyph(font, gid, 0)
        text = mupdf.fz_new_text()
        mupdf.fz_show_glyph(
            text,
            font,
            mupdf.FzMatrix(100, 0, 0, -100, 100, 150),
            gid,
            0,
            0,
            0,
            mupdf.FZ_BIDI_LTR,
            mupdf.FZ_LANG_UNSET,
        )
        canvas = mupdf.fz_new_pixmap(mupdf.fz_device_gray(), 300, 300, mupdf.FzSeparations(), 0)
        mupdf.fz_clear_pixmap_with_value(canvas, 255)
        device = mupdf.fz_new_draw_device(mupdf.FzMatrix(), canvas)
        mupdf.fz_fill_text(device, text, mupdf.FzMatrix(), mupdf.fz_device_gray(), [0.0], 1.0, mupdf.FzColorParams())
        mupdf.fz_close_device(device)
        pix = pymupdf.Pixmap(canvas)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
    except (RuntimeError, ValueError):
        return None
    width = pix.width
    samples = pix.samples
    rows = [y for y in range(pix.height) if min(samples[y * width : (y + 1) * width]) < 128]
    if not rows:
        return None
    columns = [x for x in range(width) if min(samples[x::width]) < 128]
    return (
        (rows[0] - 150) / 100,
        (rows[-1] + 1 - 150) / 100,
        advance,
        (columns[0] - 100) / 100,
        (columns[-1] + 1 - 100) / 100,
    )


def _bare_font_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", name)


class _GlyphInk:
    """Left and right ink edges of each character on the page, measured from the glyphs. The get_text box only spans
    the character width; the hook and tail of an italic f extend beyond it. Glyph IDs and embedded fonts must be read
    before the original text is erased, since erased characters can no longer be read."""

    def __init__(self, page: pymupdf.Page):
        # Names in the font list may be spelled differently from those texttrace reports (Nimbus Roman Italic,
        # NimbusRoman-Italic), so they are compared with spaces and hyphens removed
        self._fonts: dict[str, list[tuple[bytes, bytes]]] = {}
        for name, found in _embedded_fonts(page).items():
            self._fonts.setdefault(_bare_font_name(name), []).extend(found)
        self._chars = [
            (_bare_font_name(str(trace["font"])), float(trace["size"]), char)
            for trace in page.get_texttrace()
            if tuple(trace["dir"]) == (1.0, 0.0)
            for char in trace["chars"]
            if char[1] >= 0 and char[0] != 32
        ]

    def x_bounds(self, span: dict) -> tuple[float, float] | None:
        """Leftmost and rightmost ink of the characters in span. None if any character cannot be measured."""
        origin = span.get("origin")
        if not origin:
            return None
        x0, _y0, x1, _y1 = span["bbox"]
        lefts: list[float] = []
        rights: list[float] = []
        for font, size, (_code, gid, (ox, oy), box) in self._chars:
            if not x0 - 0.05 <= ox < x1 - 0.05 or abs(oy - origin[1]) > 0.1:
                continue
            for digest, buffer in self._fonts.get(font, []):
                extent = _glyph_extent(digest, buffer, gid)
                if extent is not None and abs(extent[2] * size - (box[2] - box[0])) <= max(
                    0.3, (box[2] - box[0]) * 0.08
                ):
                    lefts.append(ox + extent[3] * size)
                    rights.append(ox + extent[4] * size)
                    break
            else:
                return None
        return (min(lefts), max(rights)) if lefts else None


@functools.lru_cache(maxsize=1024)
def _font_is_math(font: str) -> bool:
    """Three-level font check (the BabelDOC approach): exact allowlist → formula; text font blocklist → not formula;
    broad heuristic → formula. The blocklist comes first so that math passages set in Times/Arial are not misjudged.

    It is asked for every span, while a book has only a few dozen fonts, so results are cached by font name."""
    if MATH_FONT_PRECISE_RE.search(font):
        return True
    if TEXT_FONT_RE.search(font):
        return False
    return bool(MATH_FONT_RE.search(font))


def _span_is_math(span: dict) -> bool:
    """Formula check by font. math is set during extraction from the fonts of the whole text (see _mark_math_roman)."""
    return bool(span.get("math")) or _font_is_math(str(span.get("font") or ""))


# '1 Introduction' and '3.1 Encoder' are section headings and must not be taken as the tail of a paragraph carried
# over to the next page
_SRC_HEAD_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+\S")


# Formula sentinel. A hard cut must not fall inside a sentinel, or neither page matches.
_SENTINEL_NUM_RE = re.compile(r"\x01i\x02(\d+)\x01/i\x02")


def _color_value(color: str) -> int | None:
    """Convert #rrggbb to an integer, the same form as the colors in spans."""
    return int(color[1:], 16) if re.fullmatch(r"#[0-9a-fA-F]{6}", color or "") else None


def _small_caps_sizes(span_lines: list[list[dict]]) -> dict[int, float]:
    """The smaller part of small caps (INTRODUCTION with a larger first letter and smaller remaining letters), mapped to
    the size of the capitals.

    Within one word, a capital followed by smaller capitals with no space between is one typeface at two sizes, not
    emphasis. Writing the size marks as they are would cut the word in the middle: {z125}I{/z}NTRODUCTION.
    """
    out: dict[int, float] = {}
    for line in span_lines:
        pairs: dict[float, float] = {}
        for prev, cur in zip(line, line[1:]):
            left, right = str(prev.get("text") or ""), str(cur.get("text") or "")
            if not left or not right or left[-1].isspace() or right[0].isspace():
                continue
            if not (left[-1].isupper() or left[-1] in "’'") or not _all_caps(right):
                continue
            big = out.get(id(prev), float(prev.get("size") or 0))
            small = float(cur.get("size") or 0)
            if big > 0 and 0.6 <= small / big < 0.92:
                out[id(cur)] = big
                pairs[round(small, 1)] = big
        # Words that are lowercase in the source are set entirely in smaller capitals (UPDATE RULE); on the same line
        # they use the same size mapping
        for span in line:
            big = pairs.get(round(float(span.get("size") or 0), 1))
            if big and id(span) not in out and _all_caps(str(span.get("text") or "")):
                out[id(span)] = big
    # Whole words in body text set in small caps (AGENT, GOVERN) with no larger first letter:
    # all-caps words on the body baseline and one size below the body are sized as body text. Chinese has no case, so
    # copying the small size would cramp it
    lower = [s for line in span_lines for s in line if any(c.islower() for c in str(s.get("text") or ""))]
    if not lower:
        return out
    body = float(_by_chars(lower, lambda s: round(float(s.get("size") or 0), 1)))
    lower_ids = {id(s) for s in lower}
    for line in span_lines:
        bases = [float(s["origin"][1]) for s in line if id(s) in lower_ids and s.get("origin") is not None]
        # A run of the same size that contains lowercase ({z90}mouse (N)) is body text one size smaller, not small caps
        lowered = [False] * len(line)
        start = 0
        for i in range(1, len(line) + 1):
            if i == len(line) or round(float(line[i].get("size") or 0), 1) != round(
                float(line[start].get("size") or 0), 1
            ):
                low = any(c.islower() for s in line[start:i] for c in str(s.get("text") or ""))
                lowered[start:i] = [low] * (i - start)
                start = i
        caps_at = []
        for i, span in enumerate(line):
            text = str(span.get("text") or "")
            size = float(span.get("size") or 0)
            on_base = (
                not bases
                or span.get("origin") is None
                or min(abs(float(span["origin"][1]) - b) for b in bases) <= body * 0.15
            )
            caps_at.append(
                id(span) not in out
                and body > 0
                and 0.6 <= size / body < 0.92
                and on_base
                and not lowered[i]
                and (_all_caps(text) or not text.strip())
            )
        # Spaces follow their neighbors: a space between small caps words is included, a space between body text and
        # small caps is left alone
        for i, span in enumerate(line):
            if not caps_at[i]:
                continue
            if str(span.get("text") or "").strip():
                out[id(span)] = body
                continue
            sides = [j for j in (i - 1, i + 1) if 0 <= j < len(line)]
            if sides and all(id(line[j]) in out or caps_at[j] for j in sides):
                out[id(span)] = body
    return out


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


_TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES


class _CharIndex(list):
    """Span boxes and character boxes of a page, bucketed by the top of the span box so that looking up a box (see
    find) does not compare against every entry."""

    def __init__(self, entries: Iterable[tuple[pymupdf.Rect, list[tuple[str, pymupdf.Rect]]]] = ()) -> None:
        super().__init__(entries)
        self._rows: dict[int, list[int]] = {}
        for order, (span_box, _chars) in enumerate(self):
            self._rows.setdefault(math.floor(span_box.y0), []).append(order)

    def find(self, box: pymupdf.Rect) -> list[tuple[str, pymupdf.Rect]] | None:
        """Character boxes of the first span whose box matches box (see _boxes_match), or None if there is none.
        The tops differ by less than half a point, so their buckets differ by at most one."""
        row = math.floor(box.y0)
        found = [
            order
            for key in (row - 1, row, row + 1)
            for order in self._rows.get(key, ())
            if _boxes_match(box, self[order][0])
        ]
        return self[min(found)][1] if found else None


def _page_char_index(page: pymupdf.Page) -> _CharIndex:
    data = page.get_text("rawdict", flags=_TEXT_FLAGS)
    index: list[tuple[pymupdf.Rect, list[tuple[str, pymupdf.Rect]]]] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                chars = [
                    (str(char.get("c") or ""), pymupdf.Rect(char["bbox"]))
                    for char in span.get("chars") or []
                    if char.get("bbox") is not None
                ]
                index.append((pymupdf.Rect(span["bbox"]), chars))
    return _CharIndex(index)


def _boxes_match(left: pymupdf.Rect, right: pymupdf.Rect) -> bool:
    return (
        abs(left.x0 - right.x0) < 0.5
        and abs(left.y0 - right.y0) < 0.5
        and abs(left.x1 - right.x1) < 0.5
        and abs(left.y1 - right.y1) < 0.5
    )


def _outside_formula(
    page_spans: list[dict],
    run_spans: Sequence[dict],
    char_index: _CharIndex | None = None,
) -> list[dict]:
    """Spans on the page, excluding the formula's own characters.

    A formula's characters may have been cut out of a body text span (σ or y = 1 boxed by the formula detection model,
    the number under a radical). The page still has the span from before the cut, whose box does not match the
    formula's spans, so the whole span would be treated as someone else's characters: when the formula is tall and
    wide (with fractions or superscripts) and a hole is cut for it, the formula's own characters are cut away. Such a
    span keeps only the non-whitespace characters left after the cut, as contiguous pieces. Character boxes come from
    the span's chars (see _attach_chars), or from char_index when it has none."""
    sources = [pymupdf.Rect(span["source"]) for span in run_spans if span.get("source") is not None]
    if not sources:
        return page_spans
    pieces = [pymupdf.Rect(span["bbox"]) for span in run_spans if span.get("bbox") is not None]
    out: list[dict] = []
    for span in page_spans:
        box = pymupdf.Rect(span["bbox"])
        if not any(_boxes_match(box, source) for source in sources):
            out.append(span)
            continue
        chars = span.get("chars")
        if chars is None and char_index is not None:
            chars = char_index.find(box)
        if chars is None:
            out.append(span)
            continue
        group: list[tuple[str, pymupdf.Rect]] = []
        for char, rect in [*chars, ("", None)]:
            middle = (rect.x0 + rect.x1) / 2 if rect is not None else 0.0
            ours = rect is not None and any(
                piece.x0 - 0.1 <= middle <= piece.x1 + 0.1 and min(piece.y1, box.y1) > max(piece.y0, box.y0)
                for piece in pieces
            )
            if rect is not None and char.strip() and not ours:
                group.append((char, rect))
                continue
            if group:
                x0, x1 = min(r.x0 for _c, r in group), max(r.x1 for _c, r in group)
                rest = {**span, "text": "".join(c for c, _r in group), "bbox": (x0, box.y0, x1, box.y1)}
                if span.get("ink") is not None:
                    rest["ink"] = (x0, span["ink"][1], x1, span["ink"][3])
                out.append(rest)
            group = []
    return out


# Gap (points) between the pieces of a formula split by a line break when they are placed side by side
_PART_GAP = 1.5


def _formula_parts(formula: dict) -> list[dict]:
    return formula.get("parts") or [formula]


def _by_chars(spans: list[dict], key: Callable[[dict], object]) -> object:
    """The value that covers the most characters."""
    counts: dict[object, int] = {}
    for span in spans:
        value = key(span)
        counts[value] = counts.get(value, 0) + len(span["text"].strip())
    return max(counts, key=counts.__getitem__)


_AFFIL_MARK_RE = re.compile(r"^[∗*†‡§¶]")


def _garbled_label(text: str) -> bool:
    """In figure labels set in a symbol font, $ replaces the space. Reflowing would translate the garbled text, so the
    original is kept. Only a $ touching a letter counts: the $ in "of $2 per hour" already has a space before it and
    would be misjudged if whitespace were removed first."""
    compact = re.sub(r"\s+", "", text)
    if not compact or len(compact) > 80:
        return False
    return bool(re.search(r"[A-Za-z]\$|\$[A-Za-z]", text))


# A label of a word and a number at the start of a line: EXAMPLE 2.1, Lemma 3., Table 2:
_LABEL_START_RE = re.compile(r"[A-Z][A-Za-z]*\s+\d+(?:\.\d+)*[.:]?\s")
# A section number at the start of a line: 2.1. Bosonic Fourier transform, 3 Method
_NUMBERED_HEAD_RE = re.compile(r"\d+(?:\.\d+)*\.?\s+[A-Z]")


def _styled_edge(block: TextBlock, index: int) -> bool:
    """The first span (index=0) or the last (-1) is bold or italic, or differs from the block in font size.
    Small caps words (MUC (Vilain et al., 1995) and ...) count at body size: they are abbreviations within a sentence,
    not run-in headings"""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    if not spans:
        return False
    span = spans[index]
    mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))
    size = _small_caps_sizes(block.span_lines or []).get(id(span), float(span.get("size") or block.size))
    return mark.bold or mark.italic or abs(size - block.size) > block.size * 0.05


def _first_line_indented(block: TextBlock) -> bool:
    """The first line of the block is indented by half an em to three em relative to the following lines, the following
    lines share a left edge, and the first line reaches the same right edge as the rest. Centered lines, whose right
    edges move in together with their left edges, do not count"""
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    if len(rows) < 2:
        return False
    lefts = [min(rect.x0 for rect in row) for row in rows]
    rights = [max(rect.x1 for rect in row) for row in rows]
    rest = lefts[1:]
    return (
        max(rest) - min(rest) <= 0.5 * block.size
        and 0.5 * block.size < lefts[0] - min(rest) <= _MAX_INDENT_EM * block.size
        and rights[0] >= max(rights[1:]) - 0.5 * block.size
    )


def _indents_paragraphs(blocks: list[TextBlock]) -> bool:
    """Paragraphs in this document indent their first line. Looks at the block right below the last line of a paragraph
    (ending a sentence, short of the right edge): blocks whose first line is indented by one or two em outnumber those
    flush with that last line. Documents that separate paragraphs with space start new paragraphs flush and do not
    count. Labels of theorems and lemmas (Proposition 2.4., Proof.) and numbered headings start flush anyway and are
    not counted. The normal line gap (see _line_gap) is subtracted from the gap between the two lines first: with
    fonts whose line boxes are taller than the line spacing, the boxes of two adjacent lines overlap slightly"""
    indented = flush = 0
    for p, b in zip(blocks, blocks[1:]):
        if p.page != b.page or not p.line_rects or not b.line_rects or abs(p.size - b.size) > p.size * 0.05:
            continue
        if _styled_edge(b, 0) or _LABEL_START_RE.match(b.text) or _NUMBERED_HEAD_RE.match(b.text):
            continue
        tail, head = p.line_rects[-1], b.line_rects[0]
        if (
            not _ends_sentence(p.text)
            or _list_item(b.text)
            or tail.x1 > max(p.rect.x1, b.rect.x1) - 2 * p.size
            or not -0.3 * p.size <= head.y0 - tail.y1 - p.line_gap * p.size <= 1.5 * p.size
        ):
            continue
        shift = head.x0 - tail.x0
        if p.size * 0.5 < shift <= p.size * _MAX_INDENT_EM:
            indented += 1
        elif abs(shift) <= p.size * 0.3:
            flush += 1
    return indented >= 3 and indented > flush


# Maximum first-line indent of a paragraph, in em
_MAX_INDENT_EM = 3


def _numbered_line(text: str) -> bool:
    """The line starts with a number followed by a capital letter, a Chinese character or nothing: the number of an
    example, a formula or an entry ((21.15) Ok, we fly ..., (1) For a fixed ...).
    A number followed by lowercase or punctuation is usually a formula reference in body text that wrapped to the line
    start ((2) for improved ..., (5.9), we have)"""
    lead = text.lstrip()
    tag = _EQUATION_TAG_RE.match(lead)
    if tag is None:
        return False
    rest = lead[tag.end() :].lstrip()
    return not rest or not (rest[0].islower() or rest[0] in ",.;:()[]")


# A number at the start of a line: (2.11), (3). A four-digit (2007) is a year
_EQUATION_TAG_RE = re.compile(r"\((?:\d+\.)+\d+\)|\(\d{1,3}\)")


def _same_page_line_pair(host: TextBlock, tail: TextBlock) -> bool:
    """Half a sentence at a line end continues on a short next line on the same page. Column and page breaks are not
    this case, and the boundary stays."""
    if host.page != tail.page or len(host.line_rects) > 1:
        return False
    if tail.rect.x0 > host.rect.x0 - max(host.size * 3.5, 28):
        return False
    gap = tail.rect.y0 - host.rect.y1 - host.line_gap * host.size
    return -0.5 * host.size <= gap < 1.5 * host.size


def _is_side_label(block: TextBlock) -> bool:
    """Short label on the left, beside the first line of body text. Memo., Property Enforcement., (2) Statistics
    Derivation."""
    text = block.text.strip()
    if not text.endswith(".") or len(block.line_rects) > 1:
        return False
    if len(text.split()) > 6 or block.rect.width > 150:
        return False
    if block.bold:
        return True
    return bool(re.match(r"^\(?\d+\)", text))


def _diagramish(block: TextBlock) -> bool:
    """Figure label that uses symbols as spaces. Not merged with body text into one paragraph."""
    text = block.text.strip()
    if not text or _garbled_label(text) or _AFFIL_MARK_RE.match(text):
        return True
    words = text.split()
    junk = sum(text.count(ch) for ch in "%!#$")
    if junk and (len(words) <= 4 or block.rect.width < 90):
        return True
    if "'" in text and block.rect.width < 80 and len(words) <= 3:
        return True
    return sum(ch.isalpha() for ch in text) < 2


class _FormulaImage:
    """Bitmap of a formula on its original page. Used only when the formula is on another page or pasting it back as
    vectors fails (see _paint_formula). Cropping only records the page and the area to crop; rendering happens on first
    use, once.

    When writing runs in a forked child process (see render_in_parts), it cannot render with the source document
    opened by the parent: parent and children share the file's read position, so several children reading at once
    get bytes from elsewhere, bitmap font glyphs lose strokes or are drawn as black blocks, and the page tree is read
    corrupted. A child opens its own copy by file name (see _own_source)."""

    def __init__(self, doc: pymupdf.Document, page: int, clip: pymupdf.Rect, spans: Sequence[dict] = ()):
        self.doc, self.page, self.clip = doc, page, pymupdf.Rect(clip)
        self.spans = list(spans)
        self._pid = os.getpid()
        self._area: pymupdf.Rect | None = None
        self._png: bytes | None = None

    def _source(self) -> pymupdf.Document:
        return self.doc if os.getpid() == self._pid else _own_source(self.doc)

    def area(self) -> pymupdf.Rect:
        """The area actually cropped. The hook and tail of an italic f extend beyond the character width, so left and
        right are widened to the formula's own glyph ink, by at most 1.5pt per side."""
        if self._area is None:
            area = pymupdf.Rect(self.clip)
            if self.spans:
                ink = _page_glyph_ink(self._source(), self.page)
                bounds = [ink.x_bounds(span) for span in self.spans]
                if bounds and None not in bounds:
                    area.x0 = max(min(area.x0, *(b[0] for b in bounds)), self.clip.x0 - 1.5)
                    area.x1 = min(max(area.x1, *(b[1] for b in bounds)), self.clip.x1 + 1.5)
            self._area = area
        return self._area

    def png(self) -> bytes:
        if self._png is None:
            # Transparent background: a white one would cover frame lines and rules under the pasted image
            pixmap = self._source()[self.page].get_pixmap(dpi=300, clip=self.area(), alpha=True)
            self._png = pixmap.tobytes("png")
        return self._png


# Measuring ink requires reading glyph IDs character by character and extracting embedded fonts. Formulas of one
# page are cropped one after another, so only the most recent page is kept
_GLYPH_INK_PAGE: list[tuple[tuple[int, int], "_GlyphInk"]] = []


def _page_glyph_ink(doc: pymupdf.Document, page: int) -> "_GlyphInk":
    key = (id(doc), page)
    if not _GLYPH_INK_PAGE or _GLYPH_INK_PAGE[0][0] != key:
        _GLYPH_INK_PAGE[:] = [(key, _GlyphInk(doc[page]))]
    return _GLYPH_INK_PAGE[0][1]


# Source documents a child process opened itself, keyed by file name. Stored only in child processes, which exit
# after writing
_OWN_SOURCES: dict[str, pymupdf.Document] = {}


def _own_source(doc: pymupdf.Document) -> pymupdf.Document:
    """The copy of the source document opened by this process. A document opened from memory has no file to share, so
    the original is used."""
    if not doc.name or not os.path.isfile(doc.name):
        return doc
    if doc.name not in _OWN_SOURCES:
        _OWN_SOURCES[doc.name] = pymupdf.open(doc.name)
    return _OWN_SOURCES[doc.name]


def _source_bold_lead(block: TextBlock) -> str:
    """Short bold phrase at the start of the first line, ending with a period and followed by non-bold body text."""
    if not block.span_lines:
        return ""
    lead: list[str] = []
    saw_plain = False
    for span in block.span_lines[0]:
        text = str(span.get("text") or "")
        if not text.strip():
            if lead:
                lead.append(text)
            continue
        mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))
        if mark.bold and not saw_plain:
            lead.append(text)
            continue
        saw_plain = True
        break
    phrase = "".join(lead).strip()
    if not saw_plain or not phrase.endswith(".") or len(phrase.split()) > 6 or len(phrase) > 40:
        return ""
    return phrase


def _uniform_bold(block: TextBlock) -> bool:
    """All body text characters are bold. Marks in a math font (the asterisk after a name) are pasted back unchanged
    and do not count."""
    spans = [
        span
        for line in (block.span_lines or [])
        for span in line
        if str(span.get("text") or "").strip() and not _span_is_math(span)
    ]
    if not spans:
        return False
    return all(emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)).bold for span in spans)


def _wrap_lead_bold(text: str) -> str:
    """When the translation dropped the bold at the start of the paragraph, restore it up to the first period. Run-in
    leads are short, so the whole paragraph is never wrapped."""
    stripped = text.lstrip()
    pad = text[: len(text) - len(stripped)]
    cut = 0
    for index, char in enumerate(stripped):
        if index > 16:
            return text
        if char in "。." and index >= 1:
            cut = index + 1
            break
    if cut <= 1:
        return text
    return f"{pad}{{b}}{stripped[:cut]}{{/b}}{stripped[cut:]}"


def _with_source_bold(block: TextBlock, text: str) -> str:
    """Restore the source's bold when writing. The model often drops {b}, and Chinese glyphs have no bold face of
    their own."""
    if not text or text == SPILL_TAIL or "{b}" in text:
        return text
    if _source_bold_lead(block):
        return _wrap_lead_bold(text)
    if _uniform_bold(block):
        return "{b}" + text + "{/b}"
    return text


def _visual_lines(block: TextBlock) -> list[list[pymupdf.Rect]]:
    """Merge fragments on the same baseline into one line. When line boxes are split per word, the next fragment is not
    the next line."""
    grouped: list[list[pymupdf.Rect]] = []
    gap = max(block.size * 0.4, 2.0)
    for rect in block.line_rects:
        placed = False
        for line in grouped:
            if abs(rect.y0 - line[0].y0) <= gap:
                line.append(rect)
                placed = True
                break
        if not placed:
            grouped.append([rect])
    return grouped


def _column_left(rect: pymupdf.Rect, geo: PageGeometry | None) -> float | None:
    """Left edge of the column containing rect (the text area for a single column). None when rect lies left of that
    edge"""
    if geo is None:
        return None
    edges = [geo.column_of(rect).x0] if len(geo.columns) >= 2 else []
    if geo.frame is not None:
        edges.append(geo.frame.x0)
    return max((edge for edge in edges if edge <= rect.x0 + 2), default=None)


def _paragraph_indent(blocks: list[TextBlock]) -> float | None:
    """Paragraph indent (points) in a document that indents first lines: the median indent of the first line of a
    multi-line paragraph relative to the following lines. None for documents without indents"""
    if not _indents_paragraphs(blocks):
        return None
    shifts = []
    for block in blocks:
        if _first_line_indented(block):
            lefts = [min(rect.x0 for rect in row) for row in sorted(_visual_lines(block), key=lambda row: row[0].y0)]
            shifts.append(lefts[0] - min(lefts[1:]))
    return statistics.median(shifts) if shifts else None


def _pages_to_write(blocks: list[TextBlock], translations: list[str]) -> dict[int, list[tuple[TextBlock, str]]]:
    """Blocks to write back on each page and the text to write for each. Blocks whose translation equals the original
    are left alone."""
    by_page: dict[int, list[tuple[TextBlock, str]]] = {}
    for b, t in zip(blocks, translations):
        if t == SPILL_TAIL or (t and t.strip() and t.strip() != b.text.strip()):
            shown = t if t == SPILL_TAIL else _with_source_bold(b, t)
            by_page.setdefault(b.page, []).append((b, shown))
    return by_page


# A tail block only has its original text erased. The translation is written in full lines into the host block and
# continues into the tail once the host is full, without running past the page edge.
SPILL_TAIL = "\x02"


# Section and item numbers: 4, 4.1, 3.2.1., B.3.2., 1., (a), a), (1). The number may be wrapped in size or bold
# marks
_NUMBER_LABEL_RE = re.compile(
    r"^((?:\{[bizu]\d*\})*)"
    r"(\d{1,3}(?:\.\d{1,3})*\.?|[A-Z](?:\.\d{1,3})+\.?|[A-Z]\.|[IVX]{1,4}\.|\(?[a-z]\)|\(\d{1,2}\))"
    r"((?:\{/[bizu]\})*)(?=\s|\{[bizu]\d*\}\s)"
)


_CAPTION_NUM_RE = re.compile(r"\s*(?:Figure|Fig\.?|Table|Tab\.?)\s*(\d+)\b", re.I)


# Marks in the text sent for translation: style, size, color, formula placeholder, column boundary
_MARKUP_RE = re.compile(r"\{(?:/?[biu]|z\d+|/z|c[0-9a-f]{6}|/c|v\d+|\|)\}")
