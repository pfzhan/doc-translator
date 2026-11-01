"""Recognize the letters of Type3 bitmap fonts without Unicode from their glyph shapes.

Old versions of Ghostscript draw every character as a bitmap with encoding names /a0, /a1 and no ToUnicode, so
text extraction yields character codes. A character code draws the same bitmap throughout the document. The whole
document is first recognized with tesseract, and the recognized letters vote by position on the bitmaps, which
tells which letter each bitmap is. The text layer is then rebuilt from the bitmaps: position and size come from the
bitmap box given by d1, word spacing from the gaps between the ink. Drift of OCR word boxes, missed words and
occasional misreadings no longer reach the translation.

Math italic a, v, S only occur on their own and never inside words; math symbols are recognized differently each
time. These two kinds are written in math fonts, and superscripts and subscripts are written with their actual size
and position. Formula placeholders and paragraph splitting downstream follow the same logic as ordinary TeX PDFs.
"""

from __future__ import annotations

import bisect
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

_CUSTOM_NAME = re.compile(r"^[A-Za-z]\d+$")
# Share of the page's text in broken fonts. A few characters in a symbol font do not rebuild the whole page.
_BROKEN_SHARE = 0.4
_MIN_BROKEN_CHARS = 20
_OCR_SCALE = 3.0
_OCR_TIMEOUT = 90
# Agreement ratio of the recognition. Glyphs below it are recognized differently each time: math symbols or
# math italic.
_SURE = 0.8
# A page is rebuilt only when this share of its glyphs has a recognized letter. Otherwise the original glyphs stay.
_READABLE = 0.8
# A glyph with more than a quarter of its occurrences inside words is a body-text letter
_IN_WORD = 0.25
# Gaps in x-heights: up to 0.4 is letter spacing, from 0.6 on is a space. The band in between has both; it is
# split by the letter and word spacing of the line, and when the line has no clear word spacing OCR words decide
_TIGHT = 0.4
_WIDE = 0.6
# Largest horizontal gap within one line, in font sizes. Even stretched word spaces stay under half a font size;
# wider gaps separate side-by-side author names, another column, or another table column
_RUN_GAP = 1.6
# Glyphs this much smaller than the line's font size are superscripts or subscripts. So are glyphs of full size
# whose baseline is shifted by this much (in font sizes)
_SCRIPT = 0.8
_SHIFT = 0.18
# Two bitmaps of the same letter with this similarity are the same font
_SAME_SHAPE = 0.85
_SAME_MARK = 0.5
# The ink of a filled dot covers more than 70% of its box, a letter less than half
_FILLED = 0.65
_SHAPE_DPI = 600
_SHAPE_GRID = 16
# The next line joins the same block when its baseline is within this many font sizes. TeX leading is 1.2 font
# sizes, a bit more where an inline formula makes the line taller
_LEADING = 1.6
_MATH_SYMBOLS = set("<>=+|{}^~*∑∏√∞≤≥≠≈∈∉⊂⊃±×÷→←")
_SENTENCE_MARKS = set(".,;:")
_OPERATORS = set("=<>+-−–—≤≥≠×")
_RELATIONS = set("=<>≤≥≠≺⪯≈∈")
_TRAILING_MARKS = set(",.;:)")
# Numbers such as section numbers, item numbers and years
_NUMBER = re.compile(r"\(?\d+(?:\.\d+)*[.:,)]?")
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
# Lower bound of the monospace cell width, in font sizes. Monospace is 0.525; the most evenly spaced body-text
# words (each, latter) stay below 0.48
_MONO_PITCH = 0.5
# Capital letters that form a word on their own in English
_LETTER_WORDS = {"A", "I"}
# Characters that are very narrow in body-text fonts (around 0.28 font sizes). In monospace fonts they also fill a
# whole cell
_NARROW = set("iljI.,;:’'!|")
# Ink height of each letter class as a share of the font size (about the same in Computer Modern and Times)
_X_LETTERS = set("acemnorsuvwxz")
_TALL_LETTERS = set("bdhkl") | set("ABCDEFGHIKLMNOPRSTUVWXYZ") | set("0123456789")
_DESCENDERS = set("gpqy")
_UNSTEADY = set("fjJQ")
_HEIGHT_RATIO = {"x": 0.45, "tall": 0.69, "desc": 0.64, "i": 0.67, "t": 0.62}
# Height of the center of each letter class above the baseline
_CENTER_RATIO = {"x": 0.225, "tall": 0.345, "desc": 0.13, "i": 0.335, "t": 0.31}
_QUOTES = set("‘’“”'\"`")
_DASHES = set("-–—−~")
# Ink width of the minus sign (in font sizes). En dash 0.5, em dash 1.0, hyphen 0.3
_MINUS_WIDTH = (0.55, 0.8)
_TEXT_FONT = "Times-Roman"
_MONO_FONT = "Courier"
_MATH_FONT = "CMMI10"
_SYMBOL_FONT = "CMSY10"
# Structure of PyMuPDF get_text("dict"). Its fields change between versions, so it only passes through this layer.
PageData = dict[str, Any]
Key = tuple[int, int]
# Thin rules: the underline of \_ and inline fraction bars. Rules longer than 30pt are table rules and footnote rules
_RULE: Key = (-2, 0)
_RULE_HEIGHT = 1.2
_RULE_LENGTH = 30.0
_PAD = 0.6


@dataclass(frozen=True)
class OcrChar:
    text: str
    rect: pymupdf.Rect
    conf: float


@dataclass(frozen=True)
class OcrWord:
    text: str
    rect: pymupdf.Rect
    conf: float
    chars: tuple[OcrChar, ...] = ()


@dataclass(frozen=True)
class Glyph:
    """One glyph drawn on the page. key is (font xref, character code), rect is the ink box of the bitmap, origin is
    the glyph origin."""

    key: Key
    rect: pymupdf.Rect
    color: int = 0
    origin: tuple[float, float] | None = None


@dataclass(frozen=True)
class Label:
    text: str
    agreement: float


@dataclass
class _Line:
    glyphs: list[Glyph]
    tokens: list[list[Glyph]] = field(default_factory=list)
    baseline: float = 0.0
    xheight: float = 0.0
    size: float = 0.0


def broken_type3_names(doc: pymupdf.Document, page: pymupdf.Page) -> set[str]:
    """Type3 fonts without ToUnicode whose glyph names are made-up names such as /a0."""
    return set(_broken_fonts(doc, page).values())


def _broken_fonts(doc: pymupdf.Document, page: pymupdf.Page) -> dict[int, str]:
    found: dict[int, str] = {}
    for xref, _ext, subtype, name, _ref, _encoding in page.get_fonts():
        if subtype == "Type3" and name and _custom_encoding(doc, xref):
            found[xref] = name
    return found


class Type3Reader:
    """The bitmap fonts of one document. The whole document is recognized on first use; after that the rebuilt lines
    are returned page by page."""

    def __init__(self, doc: pymupdf.Document):
        self.doc = doc
        self._lines: dict[int, list[_Line]] | None = None
        self.labels: dict[Key, Label] = {}
        self.roles: dict[Key, str] = {}
        self.mono: dict[Key, float] = {}
        self.proportional: set[Key] = set()
        self.doubtful: set[Key] = set()

    def lines(self, pno: int) -> list[_Line] | None:
        if self._lines is None:
            self._lines = self._read()
        return self._lines.get(pno)

    def _read(self) -> dict[int, list[_Line]]:
        shapes: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        pages: dict[int, list[tuple[Key, float, float, float, int]]] = {}
        for page in self.doc:
            fonts = _broken_fonts(self.doc, page)
            if not fonts:
                continue
            for xref in fonts:
                if xref not in shapes:
                    shapes[xref] = _font_shapes(self.doc, xref)
            placed = _placed_codes(page, {name: xref for xref, name in fonts.items()}, shapes)
            if len(placed) >= _MIN_BROKEN_CHARS:
                pages[page.number] = placed
        if not pages:
            return {}
        flip = _y_flip(self.doc, pages, shapes)
        glyphs = {
            pno: [_glyph(key, x, y, size, color, shapes, flip) for key, x, y, size, color in placed]
            for pno, placed in pages.items()
        }
        words = _ocr_pages(self.doc, list(glyphs))
        self.labels = _vote_labels(glyphs, words)
        # \_ and inline fraction bars are thin rules that old Ghostscript draws as small images. They belong to the line
        self.labels[_RULE] = Label("_", 1.0)
        for pno, items in glyphs.items():
            items.extend(Glyph(_RULE, rect) for rect in _rules(self.doc[pno]))
        owners = {pno: _word_owners(items, words.get(pno, [])) for pno, items in glyphs.items()}
        lines = {pno: _build_lines(items, self.labels, owners[pno]) for pno, items in glyphs.items()}
        self.roles, self.doubtful = _learn_roles(lines, self.labels, lambda keys: _glyph_shapes(self.doc, glyphs, keys))
        self.mono = _mono_kinds(lines, self.labels)
        self.proportional = _proportional_kinds(lines, self.mono)
        for key in _bullets(lines, self.labels, lambda keys: _ink_share(self.doc, glyphs, keys)):
            self.labels[key] = Label("•", 1.0)
            self.roles[key] = "text"
        bearings = _bearings(lines, self.labels)
        # A monospace space is a whole blank cell and the letter spacing itself is wide; the hook of f reaches into the
        # blank on its right. Words are split again after monospace glyphs are found and side bearings measured
        for pno, page_lines in lines.items():
            for line in page_lines:
                line.tokens = _tokens(line, self.labels, owners[pno], self.mono, bearings)
        return {pno: page_lines for pno, page_lines in lines.items() if _readable(glyphs[pno], self.labels)}


def recover_type3_page(page: pymupdf.Page, data: PageData, reader: Type3Reader | None = None) -> bool:
    """When most of this page is bitmap font text with a made-up encoding, replace the character codes with the
    recognized text layer.

    Pages that cannot be recognized (OCR timeout, most glyphs without a recognized letter) only lose the character
    codes; the bitmap glyphs stay on the original page, so the whole book does not fail.
    """
    doc = page.parent
    if doc is None or "blocks" not in data:
        return False
    names = broken_type3_names(doc, page)
    if not names or _broken_share(data, names) < _BROKEN_SHARE:
        return False
    reader = reader or Type3Reader(doc)
    lines = reader.lines(page.number)
    _drop_broken_lines(data, names)
    if not lines:
        return False
    built = [line for line in (_line_dict(line, reader) for line in lines) if line is not None]
    if not built:
        return False
    for group in _group_lines(built):
        data["blocks"].append({"type": 0, "bbox": _union_bbox(group), "lines": group})
    return True


# ---- Fonts and glyph positions ----


def _font_shapes(doc: pymupdf.Document, xref: int) -> dict[int, tuple[float, float, float, float]]:
    """Boxes in glyph space (FontMatrix applied) of the glyphs that draw something. Empty glyphs that only advance
    the pen are left out."""
    matrix = _numbers(doc.xref_get_key(xref, "FontMatrix")[1]) or [0.001, 0, 0, 0.001, 0, 0]
    font_box = _numbers(doc.xref_get_key(xref, "FontBBox")[1]) or [0, 0, 1000, 1000]
    procs = dict(re.findall(r"/([^\s/<>\[\]()]+)\s+(\d+)\s+0\s+R", doc.xref_get_key(xref, "CharProcs")[1]))
    shapes: dict[int, tuple[float, float, float, float]] = {}
    for code, name in _encoding_names(doc, xref).items():
        proc = procs.get(name)
        if proc is None:
            continue
        stream = doc.xref_stream(int(proc)) or b""
        found = re.search(rb"\bd([01])\b", stream)
        if found is None or not stream[found.end() :].strip():
            continue
        head = _numbers(stream[: found.start()].decode("latin-1"))
        box = head[2:6] if found.group(1) == b"1" and len(head) >= 6 else font_box[:4]
        scale_x, scale_y = matrix[0], matrix[3]
        x0, x1 = sorted((box[0] * scale_x, box[2] * scale_x))
        y0, y1 = sorted((box[1] * scale_y, box[3] * scale_y))
        shapes[code] = (x0, y0, x1, y1)
    return shapes


def _encoding_names(doc: pymupdf.Document, xref: int) -> dict[int, str]:
    kind, value = doc.xref_get_key(xref, "Encoding")
    source = doc.xref_object(int(value.split()[0])) if kind == "xref" else value
    found = re.search(r"/Differences\s*\[(.*?)\]", source, re.S)
    names: dict[int, str] = {}
    if found is None:
        return names
    code = 0
    # Names need not be separated by spaces: [65/a0/a1]
    for token in re.findall(r"/[^\s/\[\]<>()]+|\d+", found.group(1)):
        if token.isdigit():
            code = int(token)
        else:
            names[code] = token[1:]
            code += 1
    return names


def _numbers(raw: str) -> list[float]:
    """PDF numbers may omit the integer part: .001."""
    return [float(item) for item in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", raw)]


def _placed_codes(
    page: pymupdf.Page,
    fonts: dict[str, int],
    shapes: dict[int, dict[int, tuple[float, float, float, float]]],
) -> list[tuple[Key, float, float, float, int]]:
    """List the glyphs that draw something in content stream order: (key, origin x, origin y, font size, color).

    get_text drops zero-width glyphs at the same position as the previous character; texttrace keeps them.
    """
    placed: list[tuple[Key, float, float, float, int]] = []
    for span in page.get_texttrace():
        xref = fonts.get(str(span.get("font")))
        if xref is None or abs(span["dir"][1]) > 0.01:
            continue
        drawn = shapes.get(xref, {})
        color = _rgb(span.get("color"))
        for char in span["chars"]:
            if char[1] in drawn:
                placed.append(((xref, char[1]), char[2][0], char[2][1], float(span["size"]), color))
    return placed


def _glyph(
    key: Key,
    x: float,
    y: float,
    size: float,
    color: int,
    shapes: dict[int, dict[int, tuple[float, float, float, float]]],
    flip: bool,
) -> Glyph:
    x0, y0, x1, y1 = shapes[key[0]][key[1]]
    if flip:
        rect = pymupdf.Rect(x + x0 * size, y + y0 * size, x + x1 * size, y + y1 * size)
    else:
        rect = pymupdf.Rect(x + x0 * size, y - y1 * size, x + x1 * size, y - y0 * size)
    return Glyph(key, rect, color, (x, y))


def _y_flip(
    doc: pymupdf.Document,
    pages: dict[int, list[tuple[Key, float, float, float, int]]],
    shapes: dict[int, dict[int, tuple[float, float, float, float]]],
) -> bool:
    """Whether glyph space points up or down. Ghostscript bitmap fonts use an inverted Tm and draw glyphs below the
    origin.

    PyMuPDF does not expose the text matrix, so glyphs are placed both ways to see which one covers the ink on the
    page.
    """
    pno = max(pages, key=lambda number: len(pages[number]))
    pix = doc[pno].get_pixmap(dpi=36, colorspace=pymupdf.csGRAY, alpha=False)
    scale = 36 / 72

    def covered(flip: bool) -> int:
        mask = bytearray(pix.width * pix.height)
        for key, x, y, size, color in pages[pno]:
            rect = _glyph(key, x, y, size, color, shapes, flip).rect
            left, right = max(int(rect.x0 * scale), 0), min(int(rect.x1 * scale) + 1, pix.width)
            for row in range(max(int(rect.y0 * scale), 0), min(int(rect.y1 * scale) + 1, pix.height)):
                mask[row * pix.width + left : row * pix.width + right] = b"\x01" * max(right - left, 0)
        return sum(1 for hit, value in zip(mask, pix.samples) if hit and value < 128)

    return covered(True) > covered(False)


def _rules(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Short horizontal rules within lines, whether drawn as images or as vectors."""
    rects = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    try:
        rects.extend(pymupdf.Rect(drawing["rect"]) for drawing in page.get_drawings())
    except Exception:  # noqa: BLE001 - vector graphics of some pages cannot be read; treat as none
        pass
    return [rect for rect in rects if rect.height <= _RULE_HEIGHT and 1.5 <= rect.width <= _RULE_LENGTH]


def _rgb(color: object) -> int:
    if not isinstance(color, (tuple, list)) or len(color) < 3:
        return 0
    red, green, blue = (min(max(int(float(part) * 255 + 0.5), 0), 255) for part in color[:3])
    return (red << 16) | (green << 8) | blue


# ---- Recognition ----


def _ocr_pages(doc: pymupdf.Document, numbers: list[int]) -> dict[int, list[OcrWord]]:
    """Recognize the whole document together. Rendering runs on the main thread and tesseract in parallel; each page
    is recognized as soon as it is rendered. A page that times out casts no votes."""
    if shutil.which("tesseract") is None:
        raise ValueError("此 PDF 的文字为不含 Unicode 编码的点阵字体，需要安装 tesseract 才能识别原文")

    def read(path: Path) -> list[OcrWord]:
        try:
            return ocr_image(path)
        except subprocess.TimeoutExpired:
            return []

    workers = max(1, min(4, os.cpu_count() or 1, len(numbers)))
    with tempfile.TemporaryDirectory(prefix="type3-") as tmp, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for pno in numbers:
            path = Path(tmp) / f"p{pno}.png"
            doc[pno].get_pixmap(matrix=pymupdf.Matrix(_OCR_SCALE, _OCR_SCALE), alpha=False).save(path)
            pending[pno] = pool.submit(read, path)
        return {pno: future.result() for pno, future in pending.items()}


def ocr_image(path: Path, scale: float = _OCR_SCALE) -> list[OcrWord]:
    exe = shutil.which("tesseract")
    if exe is None:
        return []
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    proc = subprocess.run(
        [exe, str(path), "stdout", "--psm", "3", "-l", "eng", "-c", "hocr_char_boxes=1", "hocr"],
        capture_output=True,
        text=True,
        timeout=_OCR_TIMEOUT,
        check=False,
        env=env,
    )
    if proc.returncode != 0 and "traineddata" in proc.stderr:
        raise ValueError("点阵字体页面未识别出文字，请确认已安装 tesseract 英文语言包")
    parser = _HocrWords(scale)
    parser.feed(proc.stdout)
    return parser.words


class _HocrWords(HTMLParser):
    """Words and per-character boxes in hOCR. Character boxes and confidences are on ocrx_cinfo."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.words: list[OcrWord] = []
        self._stack: list[str] = []
        self._word: tuple[pymupdf.Rect, float] | None = None
        self._chars: list[OcrChar] = []
        self._char: tuple[pymupdf.Rect, float] | None = None
        self._text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            return
        info = dict(attrs)
        kind = info.get("class") or ""
        title = info.get("title") or ""
        self._stack.append(kind)
        if kind == "ocrx_word":
            box = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
            conf = re.search(r"x_wconf ([\d.]+)", title)
            self._word = (self._rect(box), float(conf.group(1)) if conf else 0.0) if box else None
            self._chars = []
        elif kind == "ocrx_cinfo":
            box = re.search(r"x_bboxes (\d+) (\d+) (\d+) (\d+)", title)
            conf = re.search(r"x_conf ([\d.]+)", title)
            self._char = (self._rect(box), float(conf.group(1)) if conf else 0.0) if box else None
            self._text = ""

    def handle_data(self, data: str) -> None:
        if self._char is not None:
            self._text += data

    def handle_endtag(self, tag: str) -> None:
        if tag != "span" or not self._stack:
            return
        kind = self._stack.pop()
        if kind == "ocrx_cinfo" and self._char is not None:
            text = _LIGATURES.get(self._text.strip(), self._text.strip())
            if text:
                self._chars.append(OcrChar(text, *self._char))
            self._char = None
        elif kind == "ocrx_word" and self._word is not None:
            text = "".join(char.text for char in self._chars)
            if text:
                self.words.append(OcrWord(text, self._word[0], self._word[1], tuple(self._chars)))
            self._word = None

    def _rect(self, box: re.Match[str]) -> pymupdf.Rect:
        return pymupdf.Rect(*(int(box.group(index)) / self.scale for index in range(1, 5)))


def _vote_labels(glyphs: dict[int, list[Glyph]], words: dict[int, list[OcrWord]]) -> dict[Key, Label]:
    """Recognized characters vote on the glyphs they fall on by position.

    Words whose character count matches the glyph count are paired one by one, which settles most glyphs. Words that
    do not match (the ligature ffi is one glyph, OCR missed or added a character) are then aligned by the settled
    glyphs and by position: a reliably recognized glyph takes its own character, and the remaining characters go to
    the unsettled glyphs.
    """
    votes: dict[Key, Counter[str]] = defaultdict(Counter)
    uneven: list[tuple[list[Glyph], tuple[OcrChar, ...]]] = []
    for pno, page_words in words.items():
        centers = _GlyphCenters(glyphs.get(pno, []))
        for word in page_words:
            box = pymupdf.Rect(word.rect.x0 - 0.7, word.rect.y0 - 0.7, word.rect.x1 + 0.7, word.rect.y1 + 0.7)
            inside = sorted(centers.inside(box), key=lambda glyph: glyph.rect.x0)
            if not inside or not word.chars:
                continue
            # When the counts match, pair in order: the OCR character order is reliable, while per-character boxes
            # are often two or three points off
            if len(inside) == len(word.chars):
                for glyph, char in zip(inside, word.chars):
                    votes[glyph.key][char.text] += char.conf
            else:
                uneven.append((inside, word.chars))
    first = _best_labels(votes)
    for inside, chars in uneven:
        for glyph, taken in zip(inside, _align(inside, chars, first)):
            if taken and not _sure(first.get(glyph.key)):
                votes[glyph.key]["".join(char.text for char in taken)] += min(char.conf for char in taken) / 2
    return _best_labels(votes)


def _best_labels(votes: dict[Key, Counter[str]]) -> dict[Key, Label]:
    labels: dict[Key, Label] = {}
    for key, counter in votes.items():
        total = sum(counter.values())
        if total > 0:
            text, weight = counter.most_common(1)[0]
            labels[key] = Label(text, weight / total)
    return labels


def _sure(label: Label | None) -> bool:
    return label is not None and label.agreement >= _SURE


def _align(
    glyphs: list[Glyph],
    chars: tuple[OcrChar, ...],
    labels: dict[Key, Label],
) -> list[tuple[OcrChar, ...]]:
    """Assign the recognized characters to the glyphs in order, 0 to 3 characters per glyph, with the lowest cost.

    A settled glyph should get its own character; a character that does not overlap its glyph horizontally adds cost,
    so an extra dot recognized by OCR does not push all following characters out of place.
    """
    count, total = len(glyphs), len(chars)
    inf = float("inf")
    cost = [[inf] * (total + 1) for _ in range(count + 1)]
    back = [[0] * (total + 1) for _ in range(count + 1)]
    cost[0][0] = 0.0
    for index in range(1, count + 1):
        label = labels.get(glyphs[index - 1].key)
        for used in range(total + 1):
            for take in range(0, min(3, used) + 1):
                prev = cost[index - 1][used - take]
                if prev == inf:
                    continue
                taken = chars[used - take : used]
                text = "".join(char.text for char in taken)
                if _sure(label):
                    step = 0.0 if text == label.text else 2.0
                else:
                    step = 1.5 if take == 0 else 0.3 * (take - 1)
                # Characters assigned to a glyph must overlap it horizontally
                step += sum(1.0 for char in taken if _overlap(glyphs[index - 1].rect, char.rect) <= 0)
                if prev + step < cost[index][used]:
                    cost[index][used] = prev + step
                    back[index][used] = take
    if cost[count][total] == inf:
        return [() for _ in glyphs]
    taken: list[tuple[OcrChar, ...]] = []
    used = total
    for index in range(count, 0, -1):
        take = back[index][used]
        taken.append(tuple(chars[used - take : used]))
        used -= take
    return taken[::-1]


def _readable(glyphs: list[Glyph], labels: dict[Key, Label]) -> bool:
    known = sum(1 for glyph in glyphs if glyph.key in labels)
    return bool(glyphs) and known >= len(glyphs) * _READABLE


# ---- Lines, words and glyph roles ----


def _build_lines(glyphs: list[Glyph], labels: dict[Key, Label], owners: dict[int, int]) -> list[_Line]:
    """Arrange glyphs into lines by baseline, ordered by position like the text of ordinary PDFs, and not by content
    stream order.

    Each glyph with a recognized letter class has its own baseline, and glyphs with the same baseline form a line.
    Superscripts, subscripts, numerators and denominators are smaller and off the baseline, so they first form small
    lines of their own and are then merged back into the line they belong to; symbols and punctuation without a
    recognized class are merged by height. After merging, lines are cut at large horizontal gaps: side-by-side author
    names, two columns. In the content stream a denominator moves the pen back to the left of the fraction, and joining
    lines in stream order would attach the rest of the sentence after the fraction to the denominator.
    """
    placed: list[tuple[Glyph, float, float]] = []
    loose: list[Glyph] = []
    for glyph in glyphs:
        size = _glyph_size(glyph, labels)
        if size is None:
            loose.append(glyph)
        else:
            placed.append((glyph, size, _own_baseline(glyph, labels, size)))
    bands: list[list[tuple[Glyph, float, float]]] = []
    for item in sorted(placed, key=lambda item: item[2]):
        band = next((band for band in reversed(bands[-6:]) if _same_band(band, item)), None)
        if band is None:
            bands.append([item])
        else:
            band.append(item)
    ordered = [[item[0] for item in band] for band in sorted(bands, key=len, reverse=True)]
    rows: list[_Line] = []
    for piece in (piece for piece in ordered if len(piece) >= 3):
        # Body-text glyphs lowered as a whole (supplier below N) can be many, and are still a subscript of an earlier
        # piece
        host = _host_line(rows, piece, labels)
        if host is None:
            rows.append(_row(piece, labels))
        else:
            host.glyphs.extend(piece)
    pending = [*(piece for piece in ordered if len(piece) < 3), *([glyph] for glyph in loose)]
    while pending:
        pending = _attach(rows, pending, labels)
        if pending:
            # The remaining small pieces form lines of their own: the largest piece becomes a line first, then the
            # others are checked again for merging into it
            rows.append(_row(pending.pop(0), labels))
    lines: list[_Line] = []
    for row in rows:
        row.glyphs.sort(key=lambda glyph: glyph.rect.x0)
        _measure(row, labels)
        current = [row.glyphs[0]]
        right = row.glyphs[0].rect.x1
        for glyph in row.glyphs[1:]:
            if glyph.rect.x0 - right > row.size * _RUN_GAP:
                lines.append(_Line(current))
                current = []
            current.append(glyph)
            right = max(right, glyph.rect.x1)
        lines.append(_Line(current))
    # Thin rules that fall into no line are lines of a figure, not characters
    lines = [line for line in lines if any(glyph.key != _RULE for glyph in line.glyphs)]
    for line in lines:
        _measure(line, labels)
        line.tokens = _tokens(line, labels, owners)
    return sorted(lines, key=lambda line: (line.baseline, line.glyphs[0].rect.x0))


def _row(glyphs: list[Glyph], labels: dict[Key, Label]) -> _Line:
    row = _Line(list(glyphs))
    _measure(row, labels)
    return row


def _attach(rows: list[_Line], pending: list[list[Glyph]], labels: dict[Key, Label]) -> list[list[Glyph]]:
    """Merge small pieces into lines round by round until none can be merged: "1", "." and "1" reach the heading
    only one after another."""
    extents = {id(row): _extent(row) for row in rows}
    while pending:
        still: list[list[Glyph]] = []
        for piece in pending:
            host = _host_line(rows, piece, labels, extents)
            if host is None:
                still.append(piece)
                continue
            host.glyphs.extend(piece)
            left, right = extents[id(host)]
            extents[id(host)] = (
                min(left, *(glyph.rect.x0 for glyph in piece)),
                max(right, *(glyph.rect.x1 for glyph in piece)),
            )
        if len(still) == len(pending):
            return still
        pending = still
    return []


def _extent(row: _Line) -> tuple[float, float]:
    return min(glyph.rect.x0 for glyph in row.glyphs), max(glyph.rect.x1 for glyph in row.glyphs)


def _own_baseline(glyph: Glyph, labels: dict[Key, Label], size: float) -> float:
    kind = _letter_class(labels[glyph.key].text)
    return glyph.rect.y1 - (size * 0.19 if kind == "desc" else 0.0)


def _same_band(band: list[tuple[Glyph, float, float]], item: tuple[Glyph, float, float]) -> bool:
    """Glyphs whose baselines differ by less than 12% of the font size and whose font sizes are similar sit on the
    same baseline."""
    _glyph, size, baseline = item
    sizes = [entry[1] for entry in band[-8:]]
    base = statistics.median(entry[2] for entry in band[-8:])
    ref = statistics.median(sizes)
    return abs(baseline - base) <= min(size, ref) * 0.12 and max(size, ref) <= min(size, ref) * 1.25


def _host_line(
    lines: list[_Line],
    piece: list[Glyph],
    labels: dict[Key, Label],
    extents: dict[int, tuple[float, float]] | None = None,
) -> _Line | None:
    """The line that a small piece such as a superscript, subscript, fraction or punctuation joins: the piece's center
    lies within 0.75 font sizes of the middle of the line's x-height, and the piece is horizontally next to the line.
    Pieces in a larger size do not join lines in a smaller size."""
    left = min(glyph.rect.x0 for glyph in piece)
    right = max(glyph.rect.x1 for glyph in piece)
    center = statistics.median((glyph.rect.y0 + glyph.rect.y1) / 2 for glyph in piece)
    sizes = [size for glyph in piece if (size := _glyph_size(glyph, labels))]
    best: tuple[float, _Line] | None = None
    for line in lines:
        if sizes and statistics.median(sizes) > line.size * 1.1:
            continue
        offset = abs(center - (line.baseline - line.xheight / 2))
        if offset > line.size * 0.75 or (best is not None and offset >= best[0]):
            continue
        line_left, line_right = extents[id(line)] if extents else _extent(line)
        reach = line.size * _RUN_GAP
        if line_left - reach <= right and left <= line_right + reach:
            best = (offset, line)
    return best[1] if best is not None else None


def _letter_class(text: str) -> str | None:
    """Letter class by ink height. f, j, J and Q reach below the baseline in italic and have no fixed height, so they
    get no class."""
    if len(text) != 1 or text in _UNSTEADY:
        return None
    if text in _X_LETTERS:
        return "x"
    if text in _DESCENDERS:
        return "desc"
    if text in ("i", "t"):
        return text
    if text in _TALL_LETTERS:
        return "tall"
    return None


def _glyph_size(glyph: Glyph, labels: dict[Key, Label], sure: bool = True) -> float | None:
    """Convert the ink height back to a font size by letter class. Glyphs not recognized reliably have no font
    size."""
    label = labels.get(glyph.key)
    if label is None or (sure and label.agreement < _SURE):
        return None
    kind = _letter_class(label.text)
    if kind is None:
        return None
    return glyph.rect.height / _HEIGHT_RATIO[kind]


def _measure(line: _Line, labels: dict[Key, Label]) -> None:
    sizes = [size for glyph in line.glyphs if (size := _glyph_size(glyph, labels))]
    line.size = statistics.median(sizes) if sizes else statistics.median(g.rect.height for g in line.glyphs) / 0.7
    sitting = [
        glyph
        for glyph in line.glyphs
        if (size := _glyph_size(glyph, labels))
        and size >= line.size * _SCRIPT
        and _letter_class(labels[glyph.key].text) in ("x", "tall", "i", "t")
    ]
    line.baseline = statistics.median(glyph.rect.y1 for glyph in (sitting or line.glyphs))
    line.xheight = line.size * _HEIGHT_RATIO["x"]


def _tokens(
    line: _Line,
    labels: dict[Key, Label],
    owners: dict[int, int],
    mono: dict[Key, float] | None = None,
    bearings: dict[Key, tuple[float, float]] | None = None,
) -> list[list[Glyph]]:
    tokens: list[list[Glyph]] = []
    mono = mono or {}
    bearings = bearings or {}
    gaps = [_gap(left, right, line, bearings) for left, right in zip(line.glyphs, line.glyphs[1:])]
    split = _space_split(gaps)
    for index, glyph in enumerate(line.glyphs):
        prev = tokens[-1][-1] if tokens else None
        pitch = (mono.get(prev.key) or mono.get(glyph.key)) if prev is not None else None
        same_word = (
            prev is not None and owners.get(id(prev)) is not None and owners.get(id(prev)) == owners.get(id(glyph))
        )
        if prev is not None and pitch and (prev.key in mono and glyph.key in mono or same_word):
            # One side is monospace and OCR also reads both as one word (the s in @db.stanford, which occurs only
            # once): decide by the cell width
            joined = _middle(glyph) - _middle(prev) <= pitch * 1.5
        else:
            joined = prev is not None and not _space_before(gaps[index - 1], prev, glyph, owners, split)
        if joined:
            tokens[-1].append(glyph)
        else:
            tokens.append([glyph])
    # No space precedes a comma or period. In monospace fonts the comma sits in the middle of its cell, with its ink
    # far from the previous character
    merged: list[list[Glyph]] = []
    for token in tokens:
        marks = all((labels.get(glyph.key) or Label("?", 0)).text in _TRAILING_MARKS for glyph in token)
        if merged and marks and token[0].rect.x0 - max(glyph.rect.x1 for glyph in merged[-1]) <= line.xheight:
            merged[-1].extend(token)
        else:
            merged.append(token)
    return merged


def _gap(left: Glyph, right: Glyph, line: _Line, bearings: dict[Key, tuple[float, float]]) -> float:
    """Blank space between the ink of two glyphs, in x-heights, minus the side bearings of both glyphs."""
    raw = (right.rect.x0 - left.rect.x1) / line.xheight
    return raw - bearings.get(left.key, (0.0, 0.0))[1] - bearings.get(right.key, (0.0, 0.0))[0]


def _space_before(gap: float, prev: Glyph, glyph: Glyph, owners: dict[int, int], split: float | None) -> bool:
    if gap <= _TIGHT:
        return False
    if gap >= _WIDE:
        return True
    if split is not None:
        return gap >= split
    left, right = owners.get(id(prev)), owners.get(id(glyph))
    if left is None or right is None:
        return gap > (_TIGHT + _WIDE) / 2
    return left != right


def _space_split(gaps: list[float]) -> float | None:
    """In a justified line all word spaces have the same width. The midpoint between the line's letter spacing and
    word spacing is the threshold; when the line has no clear word spacing, return None and leave it to the OCR
    words."""
    letters = [gap for gap in gaps if gap <= _TIGHT]
    spaces = [gap for gap in gaps if _WIDE <= gap <= 3.0]
    if len(letters) < 3 or len(spaces) < 2:
        return None
    return (statistics.median(letters) + statistics.median(spaces)) / 2


def _bearings(pages: dict[int, list[_Line]], labels: dict[Key, Label]) -> dict[Key, tuple[float, float]]:
    """Left and right side bearings of each glyph: how much wider or narrower than usual the letter spacing before
    and after it is in reliably recognized words.

    The hook of upright f reaches to the right, so the spacing after it is always narrower than after other letters,
    and the space in "of the" looks only half as wide.
    """
    after: dict[Key, list[float]] = defaultdict(list)
    before: dict[Key, list[float]] = defaultdict(list)
    every: list[float] = []
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            if not _sure_word(token, labels):
                continue
            for left, right in zip(token, token[1:]):
                gap = (right.rect.x0 - left.rect.x1) / line.xheight
                after[left.key].append(gap)
                before[right.key].append(gap)
                every.append(gap)
    if not every:
        return {}
    usual = statistics.median(every)

    def shift(values: list[float]) -> float:
        return max(-0.3, min(0.3, statistics.median(values) - usual)) if len(values) >= 3 else 0.0

    return {key: (shift(before.get(key, [])), shift(after.get(key, []))) for key in {*after, *before}}


def _word_owners(glyphs: list[Glyph], words: list[OcrWord]) -> dict[int, int]:
    """Index of the OCR word each glyph falls in. Words are split by it when the word spacing is ambiguous."""
    owners: dict[int, int] = {}
    centers = _GlyphCenters(glyphs)
    for index, word in enumerate(words):
        box = pymupdf.Rect(word.rect.x0 - 0.7, word.rect.y0 - 0.7, word.rect.x1 + 0.7, word.rect.y1 + 0.7)
        for glyph in centers.inside(box):
            owners[id(glyph)] = index
    return owners


class _GlyphCenters:
    """The glyphs of one page sorted by the vertical coordinate of their center. A page has a thousand glyphs and
    hundreds of words, and comparing every word with every glyph is too slow, so a box lookup only checks the range
    whose centers lie between the top and bottom of the box."""

    def __init__(self, glyphs: list[Glyph]) -> None:
        self._glyphs = glyphs
        self._order = sorted(range(len(glyphs)), key=lambda index: (glyphs[index].rect.y0 + glyphs[index].rect.y1) / 2)
        self._middles = [(glyphs[index].rect.y0 + glyphs[index].rect.y1) / 2 for index in self._order]

    def inside(self, box: pymupdf.Rect) -> list[Glyph]:
        """Glyphs whose center lies in box, in their original order."""
        low = bisect.bisect_left(self._middles, box.y0)
        high = bisect.bisect_right(self._middles, box.y1)
        found = sorted(index for index in self._order[low:high] if _center_in(self._glyphs[index].rect, box))
        return [self._glyphs[index] for index in found]


def _learn_roles(
    pages: dict[int, list[_Line]],
    labels: dict[Key, Label],
    shapes: Callable[[set[Key]], dict[Key, np.ndarray]],
) -> tuple[dict[Key, str], set[Key]]:
    """Count across the document how often each glyph occurs inside words, and decide whether it is body text, math,
    or a digit or punctuation mark.

    First, words of four or more letters with every character recognized reliably settle a set of body-text glyphs
    (view names such as psc and cp are math italic and do not count). All letters of such words count as inside a
    word. Short words come next: a letter counts as inside a word when another letter of the same word is a body-text
    glyph and none of them is a variable letter that often stands alone (monospace S is also used alone as a variable
    in pseudocode). This fills in the ligature ff, the W of We, and T and h that occur in headings only in The.
    Math italic a and v almost never occur in words (v’s counts once in a while).
    Letters that never occur in a word are variables. In English only A and I form words on their own: their bitmaps
    are compared with A and I in body text, and a similar one is body text (the A one size smaller in the abstract).
    Italic body text and math italic look the same, so other letters cannot be told apart by comparison.
    """
    tokens = [
        _without_plural(_main_glyphs(line, token, labels), labels)
        for lines in pages.values()
        for line in lines
        for token in line.tokens
    ]
    seen = Counter(glyph.key for token in tokens for glyph in token)
    seeds = {
        glyph.key for token in tokens if _sure_word(token, labels) for glyph in token if _letter(labels.get(glyph.key))
    }
    alone = Counter(token[0].key for token in tokens if len(token) == 1 and _letter(labels.get(token[0].key)))
    loner = {key for key, count in alone.items() if count * 3 >= seen[key]}
    in_word: Counter[Key] = Counter()
    for token in tokens:
        if not _alpha_core(token, labels, 2):
            continue
        letters = [glyph for glyph in token if _letter(labels.get(glyph.key))]
        sure = _sure_word(token, labels)
        for glyph in letters:
            others = [item for item in letters if item is not glyph]
            if sure or (any(item.key in seeds for item in others) and not any(item.key in loner for item in others)):
                in_word[glyph.key] += 1
    numeral: Counter[Key] = Counter()
    for token in tokens:
        text = "".join(_text_of(glyph, labels) for glyph in token)
        # At least two digits: math italic b is often read as 6, and followed by a comma it also looks like a number
        if sum(ch.isdigit() for ch in text) >= 2 and _NUMBER.fullmatch(text):
            for glyph in token:
                numeral[glyph.key] += 1
    roles = {key: _role(labels.get(key), in_word[key], count, numeral[key] * 2 >= count) for key, count in seen.items()}
    # Dashes are told apart by width: the minus sign is 0.6 font sizes, and hyphen, en dash and em dash each have
    # their own width; marks taller than a dash are arrows
    dashes: dict[Key, list[tuple[float, float]]] = defaultdict(list)
    for line in (line for lines in pages.values() for line in lines):
        for glyph in line.glyphs:
            if _text_of(glyph, labels) in _DASHES:
                dashes[glyph.key].append((glyph.rect.width / line.size, glyph.rect.height / line.size))
    for key, shapes_seen in dashes.items():
        width = statistics.median(shape[0] for shape in shapes_seen)
        height = statistics.median(shape[1] for shape in shapes_seen)
        if height > 0.15 or _MINUS_WIDTH[0] <= width <= _MINUS_WIDTH[1]:
            roles[key] = "symbol"
    # The article A starts a sentence, the variable A sits inside one. It is the article only when it mostly starts
    # sentences and its bitmap looks like the body-text A.
    # Punctuation that occurs only once or twice is also compared by bitmap with common glyphs of the same name
    opening = Counter()
    for lines in pages.values():
        previous = "."
        for line in lines:
            for token in line.tokens:
                if len(token) == 1:
                    if previous[-1:] in ".!?:":
                        opening[token[0].key] += 1
                previous = "".join(_text_of(glyph, labels) for glyph in token) or previous
    words = {
        key
        for key, role in roles.items()
        if role == "letter" and labels[key].text in _LETTER_WORDS and opening[key] * 2 >= seen[key]
    }
    odd = {
        key for key, role in roles.items() if role == "neutral" and seen[key] <= 2 and not labels[key].text.isalnum()
    }
    names = {labels[key].text for key in words | odd}
    refs = {
        key
        for key, role in roles.items()
        if key not in odd
        and labels.get(key) is not None
        and labels[key].text in names
        and (role == "text" or (role == "neutral" and seen[key] >= 4))
    }
    doubtful: set[Key] = set()
    if words or odd:
        images = shapes(words | odd | refs)
        for key in words:
            if _looks_like(key, refs, labels, images, _SAME_SHAPE):
                roles[key] = "text"
        # A footnote asterisk read as a quote mark does not look like real quote marks, so its recognition is not
        # trusted
        doubtful = {key for key in odd if not _looks_like(key, refs, labels, images, _SAME_MARK)}
    return {key: "math" if role == "letter" else role for key, role in roles.items()}, doubtful


def _without_plural(token: list[Glyph], labels: dict[Key, Label]) -> list[Glyph]:
    """Plurals of variables in math: the a’s, v’s. After removing ’s, the single letter left is not a word."""
    texts = [_text_of(glyph, labels) for glyph in token]
    while texts and not texts[-1].isalnum():
        texts.pop()
    if len(texts) >= 3 and texts[-1] == "s" and texts[-2] in ("’", "'"):
        return token[: len(texts) - 2]
    return token


def _letter(label: Label | None) -> bool:
    return label is not None and label.text.isalpha()


def _core(token: list[Glyph], labels: dict[Key, Label]) -> str:
    """Strip punctuation, quote marks and footnote marks (the asterisk of Efficiently*) from both ends and return the
    characters in between."""
    texts = [(labels.get(glyph.key) or Label("?", 0)).text for glyph in token]
    start, end = 0, len(texts)
    while start < end and not texts[start].isalpha():
        start += 1
    while end > start and not texts[end - 1].isalpha():
        end -= 1
    return "".join(texts[start:end])


def _alpha_core(token: list[Glyph], labels: dict[Key, Label], length: int) -> bool:
    core = _core(token, labels)
    return sum(ch.isalpha() for ch in core) >= length and core.replace("-", "").replace("’", "").isalpha()


def _sure_word(token: list[Glyph], labels: dict[Key, Label]) -> bool:
    if any((label := labels.get(glyph.key)) is None or label.agreement < _SURE for glyph in token):
        return False
    return _alpha_core(token, labels, 4)


def _role(label: Label | None, in_word: int, count: int, numeral: bool = False) -> str:
    """Digits are recognized unsteadily (bold 1 is sometimes read as l), but they mostly occur in numbers such as 1.1
    and 12, so they are still digits."""
    if label is None:
        return "symbol"
    if count and in_word / count >= _IN_WORD:
        return "text"
    if any(ch in _MATH_SYMBOLS for ch in label.text):
        return "symbol"
    if label.agreement < _SURE and not (numeral and label.text.isdigit()):
        return "math" if label.text.isalpha() else "symbol"
    if label.text.isalpha():
        return "letter"
    return "neutral"


def _text_of(glyph: Glyph, labels: dict[Key, Label]) -> str:
    label = labels.get(glyph.key)
    return label.text if label is not None else "?"


def _looks_like(
    key: Key,
    refs: set[Key],
    labels: dict[Key, Label],
    images: dict[Key, np.ndarray],
    threshold: float,
) -> bool:
    image = images.get(key)
    label = labels.get(key)
    if image is None or label is None:
        return False
    same = [images[other] for other in refs if other in images and labels[other].text == label.text]
    return any(float(np.dot(image, other)) >= threshold for other in same)


def _bullets(
    pages: dict[int, list[_Line]],
    labels: dict[Key, Label],
    density: Callable[[set[Key]], dict[Key, float]],
) -> set[Key]:
    """Bullets: always alone at the start of a line, square, and filled with ink. OCR often reads a filled dot as e."""
    lines = [line for page_lines in pages.values() for line in page_lines]
    seen = Counter(glyph.key for line in lines for glyph in line.glyphs)
    starts = Counter(line.tokens[0][0].key for line in lines if line.tokens and len(line.tokens[0]) == 1)
    shapes: dict[Key, pymupdf.Rect] = {}
    for line in lines:
        for glyph in line.glyphs:
            shapes.setdefault(glyph.key, glyph.rect)
    keys = {
        key
        for key, count in starts.items()
        if count >= 2
        and count * 2 >= seen[key]
        and key != _RULE
        and 0.8 <= shapes[key].width / max(shapes[key].height, 0.1) <= 1.25
    }
    filled = density(keys) if keys else {}
    return {key for key in keys if filled.get(key, 0.0) >= _FILLED}


def _ink_share(doc: pymupdf.Document, glyphs: dict[int, list[Glyph]], keys: set[Key]) -> dict[Key, float]:
    """Share of the glyph box covered by ink."""
    found: dict[Key, float] = {}
    for pno, items in glyphs.items():
        for glyph in items:
            if glyph.key not in keys or glyph.key in found:
                continue
            pix = doc[pno].get_pixmap(dpi=_SHAPE_DPI, clip=glyph.rect, colorspace=pymupdf.csGRAY, alpha=False)
            ink = 1.0 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width] / 255
            found[glyph.key] = float(ink.mean()) if ink.size else 0.0
    return found


def _glyph_shapes(
    doc: pymupdf.Document,
    glyphs: dict[int, list[Glyph]],
    keys: set[Key],
) -> dict[Key, np.ndarray]:
    """For each glyph kind, take the bitmap of its first occurrence, place it in a square keeping its aspect ratio,
    block-average it to 16×16, subtract the mean and normalize; the dot product is the similarity."""
    first: dict[Key, tuple[int, Glyph]] = {}
    for pno, items in glyphs.items():
        for glyph in items:
            if glyph.key in keys and glyph.key not in first:
                first[glyph.key] = (pno, glyph)
    found: dict[Key, np.ndarray] = {}
    for key, (pno, glyph) in first.items():
        pix = doc[pno].get_pixmap(dpi=_SHAPE_DPI, clip=glyph.rect, colorspace=pymupdf.csGRAY, alpha=False)
        if pix.width < 2 or pix.height < 2:
            continue
        ink = 1.0 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width] / 255
        side = max(ink.shape)
        canvas = np.zeros((side, side))
        top, left = (side - ink.shape[0]) // 2, (side - ink.shape[1]) // 2
        canvas[top : top + ink.shape[0], left : left + ink.shape[1]] = ink
        edges = np.linspace(0, side, _SHAPE_GRID + 1).astype(int)[:-1]
        grid = np.add.reduceat(np.add.reduceat(canvas, edges, axis=0), edges, axis=1).flatten()
        grid -= grid.mean()
        norm = float(np.linalg.norm(grid))
        if norm > 0:
            found[key] = grid / norm
    return found


# ---- Writing PyMuPDF lines ----


def _line_dict(line: _Line, reader: Type3Reader) -> dict[str, Any] | None:
    styles = _glyph_styles(line, reader)
    items: list[tuple[str, Glyph]] = []
    for index, token in enumerate(line.tokens):
        if index:
            items.append(("", _SPACE))
        items.extend((styles[id(glyph)], glyph) for glyph in token)
    # Spaces between words go with the body text: a formula span is replaced as a whole by a placeholder, and a space
    # inside it would be lost with it
    for index, (_style, glyph) in enumerate(items):
        if glyph is _SPACE:
            before, after = items[index - 1][0], items[index + 1][0]
            items[index] = (before if before in _PROSE or after not in _PROSE else after, glyph)
    spans: list[dict[str, Any]] = []
    run: list[Glyph] = []
    current = ""
    for style, glyph in items:
        if run and style != current:
            spans.append(_span(run, current, line, reader))
            run = []
        run.append(glyph)
        current = style
    if run:
        spans.append(_span(run, current, line, reader))
    spans = [span for span in spans if span["text"].strip()]
    if not spans:
        return None
    box = pymupdf.Rect()
    for span in spans:
        box |= pymupdf.Rect(span["bbox"])
    return {"spans": spans, "wmode": 0, "dir": (1.0, 0.0), "bbox": (box.x0, box.y0, box.x1, box.y1)}


# Space between words
_SPACE = Glyph((-1, -1), pymupdf.Rect())
# Spaces only go with plain body text: monospace words are replaced by placeholders like formulas
_PROSE = ("text",)


def _glyph_styles(line: _Line, reader: Type3Reader) -> dict[int, str]:
    """What each glyph is written as: text, mono, math, symbol, script.

    A word containing math glyphs is a formula as a whole; digits and brackets between two formula words join it, so
    (p, s, c) is one formula. A period or comma at the end of a run of formula words is sentence punctuation and stays
    in the body text.
    """
    mono = _mono_tokens(line, reader)
    kinds: list[str] = []
    for index, token in enumerate(line.tokens):
        roles = [_role_of(glyph, line, reader) for glyph in token if not _is_script(glyph, line, reader)]
        if index in mono:
            # Monospace fonts have no math: the braces and @ in {venky, anand}@db are literal characters
            kinds.append("text")
        elif any(role in ("math", "symbol") for role in roles):
            kinds.append("math")
        elif "text" in roles:
            kinds.append("text")
        else:
            kinds.append("neutral")
    for index, kind in enumerate(kinds):
        if kind != "neutral":
            continue
        before = next((item for item in reversed(kinds[:index]) if item != "neutral"), "")
        after = next((item for item in kinds[index + 1 :] if item != "neutral"), "")
        # k = 2, 1 − (…): a number right next to an operator is also part of the formula
        # A list number 1. followed by ≺ is not a minuend either: numbers with a period do not count
        operand = (
            index > 0 and kinds[index - 1] == "math" and _ends_with_operator(line.tokens[index - 1], reader)
        ) or (
            index + 1 < len(kinds)
            and kinds[index + 1] == "math"
            and _starts_with_operator(line.tokens[index + 1], reader)
            and not _sentence_mark(line.tokens[index][-1], reader)
        )
        kinds[index] = "math" if (before == "math" and after == "math") or operand else "text"
    absorbed = _close_brackets(line.tokens, kinds, reader)
    prose_tails = _hyphen_tails(line, kinds, reader)
    splits = _clause_splits(line.tokens, kinds, reader)
    styles: dict[int, str] = {}
    for index, (token, kind) in enumerate(zip(line.tokens, kinds)):
        tail = len(token)
        phrase_end = (
            index + 1 == len(kinds) or kinds[index + 1] != "math" or _ends_sentence(token, reader) or index in splits
        )
        if index in mono and index + 1 not in mono:
            # A period or comma at the end of a run of monospace words belongs to the sentence. The comma in C1, C2
            # is still code
            while tail > 1 and _sentence_mark(token[tail - 1], reader):
                tail -= 1
        if kind == "math" and phrase_end:
            opened = _phrase_depth(line.tokens, kinds, index, reader)
            while tail > 1:
                last = token[tail - 1]
                if _sentence_mark(last, reader):
                    tail -= 1
                elif _bracket(last, reader) < 0 and opened < 0:
                    # The closing parenthesis of (using d) was opened in the body text and stays there
                    opened += 1
                    tail -= 1
                else:
                    break
        for position, glyph in enumerate(token):
            if _is_script(glyph, line, reader):
                styles[id(glyph)] = "script"
            elif id(glyph) in prose_tails:
                styles[id(glyph)] = "text"
            elif (kind == "math" and position < tail) or id(glyph) in absorbed:
                styles[id(glyph)] = "symbol" if _role_of(glyph, line, reader) == "symbol" else "math"
            else:
                styles[id(glyph)] = "mono" if index in mono and position < tail else "text"
    return styles


def _hyphen_tails(line: _Line, kinds: list[str], reader: Type3Reader) -> set[int]:
    """In n-dimensional, the body-text word after the hyphen stays in the body text and only n is a formula.
    Otherwise the model does not see the word and translates it again."""
    tails: set[int] = set()
    for token, kind in zip(line.tokens, kinds):
        if kind != "math":
            continue
        for position, glyph in enumerate(token):
            if _label_text(glyph, reader) not in "-‐" or position == 0:
                continue
            rest = [item for item in token[position + 1 :] if _label_text(item, reader).isalpha()]
            if len(rest) >= 3 and all(reader.roles.get(item.key) == "text" for item in rest):
                tails.update(id(item) for item in token[position:])
            break
    return tails


def _close_brackets(tokens: list[list[Glyph]], kinds: list[str], reader: Type3Reader) -> set[int]:
    """A bracket opened in a formula must close in the formula: in (i + 1)st, 1) goes to the formula and st stays in
    the body text."""
    absorbed: set[int] = set()
    depth = 0
    for token, kind in zip(tokens, kinds):
        if kind == "math":
            for glyph in token:
                depth = max(depth + _bracket(glyph, reader), 0)
            continue
        if depth <= 0:
            continue
        for glyph in token:
            absorbed.add(id(glyph))
            depth += _bracket(glyph, reader)
            if depth <= 0:
                break
        depth = max(depth, 0)
    return absorbed


def _phrase_depth(tokens: list[list[Glyph]], kinds: list[str], end: int, reader: Type3Reader) -> int:
    """How many more opening than closing brackets the run of formula words ending at end has. A negative number
    means extra closing brackets."""
    start = end
    while start > 0 and kinds[start - 1] == "math":
        start -= 1
    return sum(_bracket(glyph, reader) for token in tokens[start : end + 1] for glyph in token)


def _clause_splits(tokens: list[list[Glyph]], kinds: list[str], reader: Type3Reader) -> set[int]:
    """The words in a run of formula words whose trailing comma splits the formula.

    A comma inside brackets does not split: (p, s, c). When the part left of the comma has no relation symbol yet and
    is a single variable or one bracketed group, it splits: d, e, g and h are one formula each, ⟨L, ⪯⟩, a ≺ b is two.
    When the left part already has a relation symbol (i = 1, 2, …, k), what follows is only an enumeration, unless
    another relation follows (a = b, c = d).
    """
    splits: set[int] = set()
    depth = 0
    related = False
    for index, (token, kind) in enumerate(zip(tokens, kinds)):
        if kind != "math":
            depth, related = 0, False
            continue
        depth = max(depth + sum(_bracket(glyph, reader) for glyph in token), 0)
        related = related or _has_relation(token, reader)
        last = token[-1]
        if depth > 0 or not _sentence_mark(last, reader) or _label_text(last, reader) not in ",;":
            continue
        if not related or _relation_follows(tokens, kinds, index, reader):
            splits.add(index)
            depth, related = 0, False
    return splits


def _has_relation(token: list[Glyph], reader: Type3Reader) -> bool:
    return any(reader.roles.get(glyph.key) == "symbol" and _label_text(glyph, reader) in _RELATIONS for glyph in token)


def _relation_follows(tokens: list[list[Glyph]], kinds: list[str], index: int, reader: Type3Reader) -> bool:
    for after in range(index + 1, len(tokens)):
        if kinds[after] != "math":
            return False
        if _has_relation(tokens[after], reader):
            return True
        if _sentence_mark(tokens[after][-1], reader) and _label_text(tokens[after][-1], reader) in ",;":
            return False
    return False


def _ends_sentence(token: list[Glyph], reader: Type3Reader) -> bool:
    """size S. c gives: the period after S separates two formulas. An ellipsis ... does not count."""
    if len(token) < 2 or not _sentence_mark(token[-1], reader) or _label_text(token[-1], reader) != ".":
        return False
    return _label_text(token[-2], reader) != "."


def _starts_with_operator(token: list[Glyph], reader: Type3Reader) -> bool:
    first = token[0]
    return reader.roles.get(first.key) == "symbol" and _label_text(first, reader) in _OPERATORS


def _ends_with_operator(token: list[Glyph], reader: Type3Reader) -> bool:
    last = token[-1]
    return reader.roles.get(last.key) == "symbol" and _label_text(last, reader) in _OPERATORS


def _bracket(glyph: Glyph, reader: Type3Reader) -> int:
    text = _label_text(glyph, reader)
    return text.count("(") + text.count("[") - text.count(")") - text.count("]")


def _sentence_mark(glyph: Glyph, reader: Type3Reader) -> bool:
    """Sentence punctuation recognized reliably. Subscript i is often read as a semicolon, but unsteadily, so it does
    not count."""
    label = reader.labels.get(glyph.key)
    return _sure(label) and label is not None and label.text in _SENTENCE_MARKS


def _role_of(glyph: Glyph, line: _Line, reader: Type3Reader) -> str:
    """A thin rule at the baseline is an underline and counts as a character; above the baseline it is a fraction
    bar and belongs to the formula."""
    if glyph.key == _RULE:
        center = (glyph.rect.y0 + glyph.rect.y1) / 2
        return "neutral" if abs(center - line.baseline) <= line.size * 0.12 else "symbol"
    return reader.roles.get(glyph.key, "symbol")


def _is_script(glyph: Glyph, line: _Line, reader: Type3Reader) -> bool:
    """Superscript or subscript: a clearly smaller font size, or the same size with the whole glyph shifted up or
    down (the upright supplier below N).

    The shift is measured at the glyph center, not the bottom edge: italic f reaches below the baseline anyway.
    Symbols without a recognized letter have no font size; those whose bottom edge hangs above three quarters of the
    x-height (footnote marks) count as superscripts. Quote marks hang anyway and hyphens sit around half the
    x-height; neither counts.
    """
    # For letters recognized unsteadily (subscript N is sometimes read as M) the class is usually right, so the font
    # size can still be estimated
    size = _glyph_size(glyph, reader.labels, sure=False)
    if size is None:
        text = _label_text(glyph, reader)
        role = reader.roles.get(glyph.key, "symbol")
        trusted = role != "symbol" and glyph.key not in reader.doubtful
        if role == "text" or (trusted and (text in _QUOTES or text in _DASHES)):
            return False
        return glyph.rect.y1 < line.baseline - line.xheight * 0.75
    return _off_size(glyph, line, reader.labels, size)


def _off_size(glyph: Glyph, line: _Line, labels: dict[Key, Label], size: float) -> bool:
    """The font size is clearly smaller than the line's, or the center is off the height expected for its letter
    class."""
    if size < line.size * _SCRIPT:
        return True
    kind = _letter_class(labels[glyph.key].text)
    center = (glyph.rect.y0 + glyph.rect.y1) / 2
    expected = line.baseline - _CENTER_RATIO.get(kind or "", 0.3) * line.size
    return abs(center - expected) > line.size * _SHIFT


def _main_glyphs(line: _Line, token: list[Glyph], labels: dict[Key, Label]) -> list[Glyph]:
    """Remove superscripts and subscripts. The supplier below N sits in one word with N and must not make N look
    like part of a word."""
    kept = []
    for glyph in token:
        size = _glyph_size(glyph, labels, sure=False)
        if size is None or not _off_size(glyph, line, labels, size):
            kept.append(glyph)
    return kept or token


def _mono_tokens(line: _Line, reader: Type3Reader) -> set[int]:
    """Words in which at least half of the letters are monospace glyphs. Words next to a monospace word whose letter
    spacing is also the cell width (SELECT, AS, FROM) count as well."""
    found: dict[int, float] = {}
    for index, token in enumerate(line.tokens):
        # Only letters count: the parentheses of (year) are upright. Subscripts do not count either: the monospace
        # supplier below N does not make N monospace
        letters = [
            glyph for glyph in token if _label_text(glyph, reader).isalpha() and not _is_script(glyph, line, reader)
        ]
        known = [reader.mono[glyph.key] for glyph in letters if glyph.key in reader.mono]
        # Monospace and body text use two sets of bitmaps; a word with half of its letters in monospace glyphs is a
        # monospace word (the y of (day) occurs only once)
        if letters and len(known) * 2 >= len(letters):
            found[index] = statistics.median(known)
    # Letter spacing exactly at a monospace cell width, with glyphs that never occur in proportional-font words (the
    # FROM line has no other monospace word)
    pitches = set(reader.mono.values())
    for index, token in enumerate(line.tokens):
        gaps = _steps(token)
        if index in found or not gaps or min(gaps) <= 0:
            continue
        if any(
            reader.roles.get(glyph.key) in ("math", "symbol") or glyph.key in reader.proportional for glyph in token
        ):
            continue
        pitch = next((value for value in pitches if all(abs(gap - value) <= value * 0.12 for gap in gaps)), None)
        if pitch is not None:
            found[index] = pitch
    grown = bool(found)
    while grown:
        grown = False
        for index, token in enumerate(line.tokens):
            near = [found[other] for other in (index - 1, index + 1) if other in found]
            if index in found or not near:
                continue
            # part p: a single variable next to a monospace word is still a formula. Glyphs seen in body text, as in
            # So and the, do not follow either
            if any(
                reader.roles.get(glyph.key) in ("math", "symbol") or glyph.key in reader.proportional for glyph in token
            ):
                continue
            if all(abs(gap - near[0]) <= near[0] * 0.12 for gap in _steps(token)):
                found[index] = near[0]
                grown = True
    return set(found)


def _mono_kinds(pages: dict[int, list[_Line]], labels: dict[Key, Label]) -> dict[Key, float]:
    """Monospace glyphs and their cell widths.

    They are found in long words: five or more characters, the same distance between centers throughout, and a narrow
    character such as i, l or a comma. In body-text fonts the center distance between a narrow character and its
    neighbors is shorter, so the distances do not come out equal; digits have equal widths in every font and do not
    count. The monospace cell width is 0.525 font sizes; evenly spaced body-text words such as each and latter stay
    below 0.48. When at least three such words fall in the same cell-width band (one band for body text, one for
    footnotes), their glyphs are monospace glyphs.
    """
    candidates: list[tuple[float, list[Glyph]]] = []
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            gaps = _steps(token)
            if len(token) < 5 or min(gaps) <= 0 or max(gaps) > min(gaps) * 1.15:
                continue
            if statistics.median(gaps) < line.size * _MONO_PITCH:
                continue
            if any(_text_of(glyph, labels) in _NARROW for glyph in token):
                candidates.append((statistics.median(gaps), token))
    clusters: list[list[tuple[float, list[Glyph]]]] = []
    for value, token in sorted(candidates, key=lambda item: item[0]):
        if clusters and value <= clusters[-1][0][0] * 1.04:
            clusters[-1].append((value, token))
        else:
            clusters.append([(value, token)])
    mono: dict[Key, float] = {}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        pitch = statistics.median(value for value, _token in cluster)
        for _value, token in cluster:
            mono.update((glyph.key, pitch) for glyph in token if not _text_of(glyph, labels).isdigit())
    return mono


def _proportional_kinds(pages: dict[int, list[_Line]], mono: dict[Key, float]) -> set[Key]:
    """Glyphs that occur in words with uneven letter spacing: proportional fonts. Words in a monospace font have the
    same spacing throughout.

    Words such as (year), with upright parentheses around monospace characters, also have uneven spacing, so words
    containing monospace glyphs do not count.
    """
    found: set[Key] = set()
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            gaps = _steps(token)
            if len(token) < 3 or min(gaps) <= 0 or max(gaps) <= min(gaps) * 1.3:
                continue
            if not any(glyph.key in mono for glyph in token):
                found.update(glyph.key for glyph in token)
    return found


def _steps(token: list[Glyph]) -> list[float]:
    return [_middle(right) - _middle(left) for left, right in zip(token, token[1:])]


def _middle(glyph: Glyph) -> float:
    return (glyph.rect.x0 + glyph.rect.x1) / 2


def _label_text(glyph: Glyph, reader: Type3Reader) -> str:
    return _text_of(glyph, reader.labels)


def _span(run: list[Glyph], style: str, line: _Line, reader: Type3Reader) -> dict[str, Any]:
    glyphs = [glyph for glyph in run if glyph is not _SPACE]
    text = "".join(" " if glyph is _SPACE else _label_text(glyph, reader) for glyph in run)
    ink = pymupdf.Rect()
    for glyph in glyphs:
        ink |= glyph.rect
    if style == "script":
        sizes = [size for glyph in glyphs if (size := _glyph_size(glyph, reader.labels))]
        size = round(statistics.median(sizes), 1) if sizes else round(line.size * 0.7, 1)
        baseline = ink.y1
        box = pymupdf.Rect(ink)
    else:
        size = round(line.size, 1)
        baseline = line.baseline
        box = ink | pymupdf.Rect(ink.x0, baseline - size * 0.72, ink.x1, baseline + size * 0.22)
    # Formulas are cropped from the original page with this box; a small margin on all sides keeps the pixels at the
    # bitmap edges
    box = pymupdf.Rect(box.x0 - _PAD, box.y0 - _PAD, box.x1 + _PAD, box.y1 + _PAD)
    font = {
        "text": _TEXT_FONT,
        "mono": _MONO_FONT,
        "math": _MATH_FONT,
        "symbol": _SYMBOL_FONT,
        "script": _MATH_FONT,
    }[style]
    return {
        "text": text,
        "bbox": (box.x0, box.y0, box.x1, box.y1),
        "size": size,
        "font": font,
        "flags": 0,
        "color": glyphs[0].color if glyphs else 0,
        "origin": (ink.x0, baseline),
        # Erasing the original text finds characters by these origins. The font box of a bitmap font is half a line
        # taller than the ink, so finding them by box would catch other characters
        "origins": [glyph.origin for glyph in glyphs if glyph.origin is not None],
        "ascender": 0.75,
        "descender": -0.25,
    }


def _group_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Lines within the leading that overlap horizontally and have similar font sizes are grouped into one block, as
    PyMuPDF does for ordinary fonts. Paragraph splitting is left to later steps.

    Leading is measured between baselines. Line boxes are stretched by tall brackets and subscripts, so overlapping
    boxes of two lines mean nothing.
    """
    ordered = sorted(lines, key=lambda line: (_baseline(line), float(line["bbox"][0])))
    groups: list[list[dict[str, Any]]] = []
    for line in ordered:
        box = pymupdf.Rect(line["bbox"])
        size = _line_size(line)
        best: list[dict[str, Any]] | None = None
        for group in groups:
            prev = pymupdf.Rect(group[-1]["bbox"])
            prev_size = _line_size(group[-1])
            step = _baseline(line) - _baseline(group[-1])
            if not (0 < step <= max(size, prev_size) * _LEADING):
                continue
            if min(box.x1, prev.x1) - max(box.x0, prev.x0) <= 0:
                continue
            if max(size, prev_size) > min(size, prev_size) * 1.15:
                continue
            best = group
        if best is None:
            groups.append([line])
        else:
            best.append(line)
    return groups


def _baseline(line: dict[str, Any]) -> float:
    spans = [span for span in line["spans"] if span["font"] in (_TEXT_FONT, _MONO_FONT)] or line["spans"]
    return float(max(spans, key=lambda span: float(span["size"]))["origin"][1])


def _line_size(line: dict[str, Any]) -> float:
    sizes = [float(span["size"]) for span in line["spans"] if span["font"] in (_TEXT_FONT, _MONO_FONT)]
    return max(sizes) if sizes else max(float(span["size"]) for span in line["spans"])


# ---- Shared helpers ----


def _custom_encoding(doc: pymupdf.Document, xref: int) -> bool:
    if doc.xref_get_key(xref, "ToUnicode")[0] != "null":
        return False
    kind, value = doc.xref_get_key(xref, "Encoding")
    if kind == "name":
        return False
    if kind != "xref":
        return True
    enc_xref = int(value.split()[0])
    names = re.findall(r"/([A-Za-z][A-Za-z0-9.]*)", doc.xref_object(enc_xref))
    custom = [name for name in names if _CUSTOM_NAME.fullmatch(name)]
    return len(custom) >= 8


def _broken_share(data: PageData, names: set[str]) -> float:
    total = 0
    broken = 0
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                count = len(str(span.get("text") or ""))
                total += count
                if span.get("font") in names:
                    broken += count
    if broken < _MIN_BROKEN_CHARS or total == 0:
        return 0.0
    return broken / total


def _drop_broken_lines(data: PageData, names: set[str]) -> None:
    """Remove the character codes. Images and normal fonts stay, and so do normal characters mixed into the same
    line."""
    kept: list[dict[str, Any]] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            kept.append(block)
            continue
        lines: list[dict[str, Any]] = []
        for line in block["lines"]:
            spans = [span for span in line["spans"] if span.get("font") not in names]
            if not spans:
                continue
            if len(spans) != len(line["spans"]):
                line = dict(line)
                line["spans"] = spans
                line["bbox"] = _union_bbox([{"bbox": span["bbox"]} for span in spans if span.get("bbox")])
            lines.append(line)
        if not lines:
            continue
        block["lines"] = lines
        kept.append(block)
    data["blocks"] = kept


def _union_bbox(lines: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    box = pymupdf.Rect()
    for line in lines:
        raw = line.get("bbox")
        if raw:
            box |= pymupdf.Rect(raw)
    return (box.x0, box.y0, box.x1, box.y1)


def _center_in(rect: pymupdf.Rect, box: pymupdf.Rect) -> bool:
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    return box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1


def _overlap(left: pymupdf.Rect, right: pymupdf.Rect) -> float:
    return min(left.x1, right.x1) - max(left.x0, right.x0)
