"""Line breaking of translations, and markers for bold, italic, size and color differences within a block.

Formula sentinels are atoms that cannot be split. Short bracket groups are not split either when they fit.
Style markers take no width and are written only where they differ from the block's base style.
A block in a single style is still handled by the block-level size and weight, so the text sent for translation
matches the original character for character. Size markers are percentages of the base size; writing scales by
that ratio and follows the shrink steps.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

_SENTINEL_RE = re.compile(r"\x01i\x02(\d+)\x01/i\x02")
STYLE_MARK_RE = re.compile(r"\{z\d+\}|\{/z\}|\{c[0-9a-f]{6}\}|\{/c\}|\{/?[biu]\}")
_ATOM_RE = re.compile(rf"\x01i\x02\d+\x01/i\x02|{STYLE_MARK_RE.pattern}")
# URLs and email addresses are never broken. A trailing period, bracket or comma belongs to the sentence and is
# not part of them
_LINK_RE = re.compile(r"(?:https?://|www\.)[!-~]*[A-Za-z0-9/#=_-]|[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", re.A)
_SIZE_OPEN_RE = re.compile(r"\{z(\d+)\}")
COLOR_MARK_RE = re.compile(r"\{c[0-9a-f]{6}\}|\{/c\}")
_COLOR_OPEN_RE = re.compile(r"\{c([0-9a-f]{6})\}")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

_BOLD_FLAG = 16
_ITALIC_FLAG = 2
_PAREN_CLOSE = {"（": "）", "(": ")"}
_PAREN_LIMIT = 10
# These punctuation marks must not start a line. When one does not fit, the character before it moves to the next
# line and the mark follows it.
_NO_LINE_START = frozenset("。，、；：？！）】」』〉》”’％%.,;:?!)]}>")
# Opening brackets and quotes must not end a line; they move to the next line with the character after them.
_NO_LINE_END = frozenset("（【「『《〈“‘([{")
# For half-width punctuation at the end of a line, the space after it is the mark's own spacing and is not kept on
# the line.
_HALF_PUNCT = frozenset(".,;:?!)]}%")


@dataclass(frozen=True)
class Emphasis:
    bold: bool
    italic: bool
    # The source text is underlined. The line is not part of the font; extraction finds it as a thin horizontal
    # line below the text (span["underline"])
    underline: bool = False


class EmphasisWriter:
    """Write {b} {i} {u} {zN} {cRRGGBB} relative to the block's base style. Formula placeholders are atoms and are
    never wrapped in emphasis, size or color. When writing, the block level carries only size and weight, so
    underline is always written as a marker, even when the whole block is underlined."""

    def __init__(self, base: Emphasis, mixed: bool, base_size: float = 0, base_color: int | None = None) -> None:
        self.base = base
        self.mixed = mixed
        self.base_size = base_size
        self.base_color = base_color
        self._bold = False
        self._italic = False
        self._underline = False
        self._size = 100
        # The color currently written. None is the block's color
        self._color: int | None = None

    def text(self, text: str, emphasis: Emphasis, size: float = 0, color: int | None = None) -> str:
        if not text or not self.mixed:
            return text
        percent = size_percent(size, self.base_size) if size and self.base_size else 100
        stripped = text.lstrip()
        if not stripped:
            # Whitespace alone does not change the color, so two words in the same color are not split apart
            return self._space(text, emphasis, percent, self._color)
        # A span of punctuation only does not change the size: with a period set at superscript size, the model
        # would wrap {z65}.{/z} around half a sentence
        if not re.search(r"[^\W_]", stripped):
            percent = self._size
        shade = marked_color(color, self.base_color)
        lead = ""
        if stripped != text:
            # A leading space is written as whitespace and the marker opens after it: • {b}Math:, not •{b} Math:.
            # When the following text changes color, the marker closes before the space
            lead = self._space(
                text[: len(text) - len(stripped)], emphasis, percent, shade if shade == self._color else None
            )
        return lead + self._marks(emphasis, percent, shade) + stripped

    def _space(self, text: str, emphasis: Emphasis, percent: int, shade: int | None) -> str:
        """Whitespace keeps only the size and emphasis it shares with what is being written: a space opens no
        marker, and markers that the following text does not use close before the space
        ({z110}Figure 1:{/z} Gen...)."""
        kept = Emphasis(
            emphasis.bold and (self._bold or self.base.bold),
            emphasis.italic and (self._italic or self.base.italic),
            emphasis.underline and self._underline,
        )
        return self._marks(kept, percent if percent == self._size else 100, shade) + text

    def atom(self, text: str) -> str:
        if not self.mixed:
            return text
        return self._marks(self.base, 100) + text

    def close(self) -> str:
        if not self.mixed:
            return ""
        return self._marks(self.base, 100)

    def _marks(self, emphasis: Emphasis, size_pct: int, color: int | None = None) -> str:
        """From outside in: size, color, emphasis. Changing an outer marker first closes the inner ones and reopens
        them afterwards, so markers always nest in pairs."""
        if size_pct == self._size and color == self._color:
            return self._emphasis_marks(emphasis)
        parts = [self._emphasis_marks(self.base)]
        if size_pct != self._size:
            parts.append(self._switch_color(None))
            parts.append(self._switch_size(size_pct))
        parts.append(self._switch_color(color))
        parts.append(self._emphasis_marks(emphasis))
        return "".join(parts)

    def _switch_color(self, color: int | None) -> str:
        if color == self._color:
            return ""
        parts: list[str] = []
        if self._color is not None:
            parts.append("{/c}")
        if color is not None:
            parts.append(f"{{c{color:06x}}}")
        self._color = color
        return "".join(parts)

    def _switch_size(self, size_pct: int) -> str:
        parts: list[str] = []
        if self._size != 100:
            parts.append("{/z}")
        if size_pct != 100:
            parts.append(f"{{z{size_pct}}}")
        self._size = size_pct
        return "".join(parts)

    def _emphasis_marks(self, emphasis: Emphasis) -> str:
        want_bold = emphasis.bold and not self.base.bold
        want_italic = emphasis.italic and not self.base.italic
        want_underline = emphasis.underline
        parts: list[str] = []
        if self._underline and not want_underline:
            parts.append("{/u}")
            self._underline = False
        if self._italic and not want_italic:
            parts.append("{/i}")
            self._italic = False
        if self._bold and not want_bold:
            parts.append("{/b}")
            self._bold = False
        if want_bold and not self._bold:
            parts.append("{b}")
            self._bold = True
        if want_italic and not self._italic:
            parts.append("{i}")
            self._italic = True
        if want_underline and not self._underline:
            parts.append("{u}")
            self._underline = True
        return "".join(parts)


def size_percent(size: float, base: float) -> int:
    """Size relative to the block's base size, rounded to 5%. A difference under 8% counts as equal and gets no
    marker."""
    if base <= 0 or size <= 0:
        return 100
    ratio = size / base
    if 0.92 <= ratio <= 1.08:
        return 100
    percent = int(round(ratio * 20.0)) * 5
    return min(200, max(50, percent))


def emphasis_of(font: str, flags: int) -> Emphasis:
    bold = bool(flags & _BOLD_FLAG) or "Bold" in font
    italic = bool(flags & _ITALIC_FLAG) or "Italic" in font or "Oblique" in font
    return Emphasis(bold, italic)


def span_emphasis(span: dict[str, object]) -> Emphasis:
    """Weight and italic of the span, plus the underline found during extraction."""
    mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))  # type: ignore[arg-type]
    return Emphasis(mark.bold, mark.italic, bool(span.get("underline")))


def writer_for(
    spans: list[dict[str, object]],
    base_size: float = 0,
    base_color: int | None = None,
) -> EmphasisWriter:
    """Find the base emphasis; markers are written only for spans whose emphasis, size or color differs from it.
    Bold: the base is bold only when the whole block is bold. Chinese writing recognizes bold only through {b}, and
    a fully bold block gets its bold from the writing step; with a bold base, bold inside the block gets no marker,
    and a mixed block (• a bold label followed by body text) would lose its bold.
    Italic: the style covering most characters wins, upright on a tie. Chinese has no italic, and marking the
    italic body of a theorem with {i} paragraph by paragraph only adds noise.
    Spans without letters (punctuation, digits) take no part.
    base_color is the block's color, used for the whole block when writing; only text in other colors gets color
    markers."""
    weighted: list[tuple[Emphasis, int]] = []
    other_mixed = False
    for span in spans:
        text = str(span.get("text") or "")
        if not text.strip():
            continue
        mark = span_emphasis(span)
        weighted.append((mark, len(text) if re.search(r"[^\W\d_]", text) else 0))
        if base_size and size_percent(float(span.get("size") or 0), base_size) != 100:
            other_mixed = True
        if marked_color(span.get("color"), base_color) is not None:
            other_mixed = True
    if not weighted:
        return EmphasisWriter(Emphasis(False, False), False, base_size, base_color)
    worded = [item for item in weighted if item[1]] or [(mark, 1) for mark, _length in weighted]
    italic = sum(length for mark, length in worded if mark.italic)
    base = Emphasis(all(mark.bold for mark, _length in worded), italic * 2 > sum(length for _mark, length in worded))
    mixed = other_mixed or any(mark != base for mark, _length in weighted)
    return EmphasisWriter(base, mixed, base_size, base_color)


def marked_color(color: object, base_color: int | None) -> int | None:
    """The color to write as a marker. None when it matches the block's color, or when it is very light: a white
    caption label on black is pasted back whole, and white translated text would land invisibly on a white
    background. None also when the block itself is in light text."""
    if base_color is None or not isinstance(color, int) or color == base_color:
        return None
    if _light(color) or _light(base_color):
        return None
    return color


def _light(color: int) -> bool:
    red, green, blue = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return 0.299 * red + 0.587 * green + 0.114 * blue > 0.8 * 255


def restore_emphasis(escaped_html: str) -> str:
    """After html.escape, replace style markers with tags. The markers contain no characters that need escaping.
    Italic stays <i>: the fallback font has no bold Chinese, so <b> would neither emphasize the Chinese nor keep
    the Latin italic."""
    tagged = (
        escaped_html.replace("{b}", "<b>")
        .replace("{/b}", "</b>")
        .replace("{i}", "<i>")
        .replace("{/i}", "</i>")
        .replace("{u}", "<u>")
        .replace("{/u}", "</u>")
        .replace("{/z}", "</span>")
        .replace("{/c}", "</span>")
    )
    tagged = _COLOR_OPEN_RE.sub(r'<span style="color:#\1">', tagged)
    return _SIZE_OPEN_RE.sub(r'<span style="font-size:\1%">', tagged)


def strip_style_marks(text: str) -> str:
    """Remove style markers for the preview. Writing the PDF still needs them."""
    return STYLE_MARK_RE.sub("", text)


def break_lines(
    text: str,
    em: float,
    max_width: float,
    formula_widths: dict[int, float],
    first_widths: Sequence[float] = (),
) -> list[str]:
    """Break lines by estimated width. Sentinels, overwide formulas and short bracket groups that fit are never
    split. Each line carries its own style markers: bold or color spanning lines is reopened at the start of the
    next line and closed at the end of this one. Lines are written separately (one non-wrapping box per line in
    HTML, Chinese drawn piece by piece), so without this the next line would lose its style.
    first_widths are the widths of the first few lines (those making room for a figure); the other lines use
    max_width."""
    return balance_lines(_break_lines(text, em, max_width, formula_widths, first_widths))


def balance_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    opened: list[str] = []
    for line in lines:
        if not STYLE_MARK_RE.sub("", line):
            out.append(line)
            opened = open_marks(line, opened)
            continue
        text = "".join(opened) + line
        opened = open_marks(text)
        out.append(text + _closing(opened))
    return out


def open_marks(text: str, opened: list[str] | None = None) -> list[str]:
    """Style markers still open after text, from outside in. opened are those already open before text."""
    stack = list(opened or ())
    for matched in STYLE_MARK_RE.finditer(text):
        token = matched.group(0)
        if not token.startswith("{/"):
            stack.append(token)
            continue
        for index in range(len(stack) - 1, -1, -1):
            if stack[index][1] == token[2]:
                del stack[index]
                break
    return stack


def _closing(opened: list[str]) -> str:
    return "".join("{/" + token[1] + "}" for token in reversed(opened))


def balance_marks(left: str, right: str) -> tuple[str, str]:
    """Reopen the style markers still open at the cut. Size and color do not nest, so the one still open must be
    reopened, not an earlier one that has already closed."""
    opened = open_marks(left)
    if not opened:
        return left, right
    return left + _closing(opened), "".join(opened) + right


def _break_lines(
    text: str,
    em: float,
    max_width: float,
    formula_widths: dict[int, float],
    first_widths: Sequence[float] = (),
) -> list[str]:
    if "\n" in text:
        lines: list[str] = []
        for part in text.split("\n"):
            lines.extend(_break_lines(part, em, max_width, formula_widths, first_widths[len(lines) :]))
        return lines
    if not text or max_width <= 0:
        return [text]
    tokens = _tokenize(text)
    # Bracket groups are locked against the narrowest line; otherwise a group would not fit on a narrow line and
    # rolling back would return to the same place
    group_of = _paren_groups(tokens, em, formula_widths, min((max_width, *first_widths)))
    for index, start in _marked_words(tokens, em, formula_widths, min((max_width, *first_widths))).items():
        group_of.setdefault(index, start)
    lines: list[str] = []
    # Words of each line, one entry per line in lines. Hard cuts and overwide formulas on a line of their own are
    # None and are not justified
    rows: list[list[tuple[int, str]] | None] = []
    buf: list[tuple[int, str]] = []
    width = 0.0

    def room() -> float:
        return first_widths[len(lines)] if len(lines) < len(first_widths) else max_width

    def flush() -> None:
        nonlocal width
        if buf:
            lines.append(_trim_line_end("".join(token for _, token in buf)))
            rows.append(list(buf))
            buf.clear()
        width = 0.0

    def put(line: str) -> None:
        lines.append(line)
        rows.append(None)

    index = 0
    while index < len(tokens):
        token = tokens[index]
        token_width = _token_width(token, em, formula_widths)
        if not token.isspace() and token_width > room():
            flush()
            if _SENTINEL_RE.fullmatch(token):
                put(token)  # overwide formula: a line of its own (the caller scales it down)
            else:
                # Overwide ordinary words (long URLs and the like) are cut by character instead of running past
                # the right edge with nowrap
                chunk, cw = "", 0.0
                for ch in token:
                    chw = _token_width(ch, em, formula_widths)
                    if chunk and cw + chw > room():
                        moved = _detach_char(chunk) if ch in _NO_LINE_START else None
                        if moved is not None:
                            head, tail = moved
                            if head:
                                put(_trim_line_end(head))
                            chunk, cw = tail, _token_width(tail, em, formula_widths)
                        else:
                            put(_trim_line_end(chunk))
                            chunk, cw = "", 0.0
                    chunk += ch
                    cw += chw
                if chunk:
                    buf.append((index, chunk))
                    width += cw
            index += 1
            continue
        if buf and width + token_width > room():
            start = group_of.get(index)
            if start is not None and any(token_index >= start for token_index, _ in buf):
                buf[:] = [(token_index, item) for token_index, item in buf if token_index < start]
                # The opening bracket before the group also moves to the next line with it
                carried = _detach_opener(buf)
                width = sum(_token_width(item, em, formula_widths) for _, item in buf)
                flush()
                index = carried[0][0] if carried else start
                continue
            if token in _NO_LINE_START:
                detached = _detach_for_punct(buf)
                if detached is not None:
                    detached_width = sum(_token_width(item, em, formula_widths) for _, item in detached)
                    if detached_width + token_width <= room():
                        flush()
                        buf.extend(detached)
                        width = detached_width
                    else:
                        buf.extend(detached)
                        flush()
                        if token.isspace():
                            index += 1
                            continue
                else:
                    flush()
                    if token.isspace():
                        index += 1
                        continue
            else:
                carried = _detach_opener(buf)
                flush()
                if carried:
                    buf.extend(carried)
                    width = sum(_token_width(item, em, formula_widths) for _, item in carried)
                if token.isspace():
                    index += 1
                    continue
        buf.append((index, token))
        width += token_width
        index += 1
    flush()
    if not lines:
        return [text]
    widths = [first_widths[k] if k < len(first_widths) else max_width for k in range(len(lines))]
    for k in _even_out(rows, widths, em, formula_widths, group_of):
        lines[k] = _trim_line_end("".join(token for _, token in rows[k]))
    return lines


def _even_out(
    rows: list[list[tuple[int, str]] | None],
    widths: Sequence[float],
    em: float,
    formula_widths: dict[int, float],
    group_of: dict[int, int],
) -> set[int]:
    """When a long word or formula at the start of the next line did not fit, this line is short by several
    character widths and justification spreads it widely. Move a few Chinese characters from the end of the line
    above down to share the space between the two lines, until the widest gap in the two lines stops getting
    smaller; the line above then takes from the one above it, and so on upward. Only single Chinese characters
    and the style markers attached to them move, never Latin words, formulas or bracket groups, and punctuation
    at line starts and ends follows the line-breaking rules. The last line is not justified, however short it
    is. Returns the indexes of changed lines."""
    # pdf_align distributes justification space; it imports this module, so it can only be imported here
    from .pdf_align import line_extras

    changed: set[int] = set()

    def used(row: list[tuple[int, str]]) -> float:
        return sum(_token_width(token, em, formula_widths) for _, token in row)

    def stretch(row: list[tuple[int, str]], slack: float) -> float:
        # Whether a line is stretched follows how writing distributes space: lines with too few gaps or too little
        # slack are not stretched and count as 0. A stretched line counts its slack spread over every gap,
        # uncapped: a line stretched to the limit keeps the rest of its slack at the end, and moving a few
        # characters down still brings it back within the limit
        texts: list[str | None] = [""]
        for _, token in row:
            if _SENTINEL_RE.fullmatch(token):
                texts.extend([None, ""])
            elif not STYLE_MARK_RE.fullmatch(token):
                texts[-1] += token
        slots = sum(1 for extra in line_extras(texts, slack, em) if extra)
        return slack / slots if slots else 0.0

    for k in range(len(rows) - 2, 0, -1):
        upper, lower = rows[k - 1], rows[k]
        if upper is None or lower is None:
            continue
        while True:
            end = len(upper)
            while end > 0 and STYLE_MARK_RE.fullmatch(upper[end - 1][1]):
                end -= 1
            if end < 2:
                break
            index, char = upper[end - 1]
            if not _CJK_RE.fullmatch(char) or index in group_of:
                break
            # A style marker opened right before the character moves with it, so no empty {b} is left on the line
            # above
            start = end - 1
            while (
                start > 0 and STYLE_MARK_RE.fullmatch(upper[start - 1][1]) and not upper[start - 1][1].startswith("{/")
            ):
                start -= 1
            if start == 0:
                break
            rest = upper[start - 1][1]
            if rest.isspace() or rest in _NO_LINE_END or STYLE_MARK_RE.fullmatch(rest):
                break
            char_width = _token_width(char, em, formula_widths)
            upper_slack = widths[k - 1] - used(upper)
            lower_slack = widths[k] - used(lower)
            moved, kept = upper[start:] + lower, upper[:start]
            before = max(stretch(upper, upper_slack), stretch(lower, lower_slack))
            after = max(stretch(kept, upper_slack + char_width), stretch(moved, lower_slack - char_width))
            if lower_slack < char_width or after >= before:
                break
            rows[k - 1], rows[k] = upper, lower = kept, moved
            changed.update((k - 1, k))
    return changed


def _trim_line_end(line: str) -> str:
    """Remove the space after half-width punctuation at the end of a line. Style markers are not characters."""
    body = line
    marks = ""
    while True:
        matched = re.search(r"(\{/?[biu]\}|\{/z\}|\{z\d+\})$", body)
        if matched is None:
            break
        marks = matched.group(0) + marks
        body = body[: matched.start()]
    stripped = body.rstrip(" \t")
    if stripped != body and stripped and stripped[-1] in _HALF_PUNCT:
        return stripped + marks
    return line


def _detach_char(chunk: str) -> tuple[str, str] | None:
    """When hard-cutting an overwide word, leave the last character for the punctuation that starts the line."""
    if len(chunk) < 2 or chunk[-1] in _NO_LINE_START:
        return None
    return chunk[:-1], chunk[-1]


def _detach_opener(buf: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Move an opening bracket at the end of a line (with style markers attached to it) to the start of the next
    line. Nothing moves when the line holds only the bracket."""
    start = len(buf)
    while start > 0 and STYLE_MARK_RE.fullmatch(buf[start - 1][1]):
        start -= 1
    if start == 0 or buf[start - 1][1] not in _NO_LINE_END:
        return []
    start -= 1
    if not any(not item.isspace() and not STYLE_MARK_RE.fullmatch(item) for _, item in buf[:start]):
        return []
    detached = buf[start:]
    del buf[start:]
    return detached


def _detach_for_punct(buf: list[tuple[int, str]]) -> list[tuple[int, str]] | None:
    """When punctuation at the end of a line does not fit, take the character before it along. Nothing moves when
    the line has no character to take."""
    end = len(buf)
    while end > 0 and (
        STYLE_MARK_RE.fullmatch(buf[end - 1][1]) or buf[end - 1][1].isspace() or buf[end - 1][1] in _NO_LINE_START
    ):
        end -= 1
    if end == 0:
        return None
    start = end - 1
    while start > 0 and STYLE_MARK_RE.fullmatch(buf[start - 1][1]):
        start -= 1
    detached = buf[start:]
    del buf[start:]
    return detached


def _visible_len(token: str) -> int:
    """Length of a bracket group without style markers. A formula sentinel counts as one character regardless of
    the length of its text."""
    if STYLE_MARK_RE.fullmatch(token):
        return 0
    if _SENTINEL_RE.fullmatch(token):
        return 1
    return len(token)


_WORD_PIECE_RE = re.compile(r"[A-Za-z0-9'’_-]+")


def _marked_words(
    tokens: list[str],
    em: float,
    formula_widths: dict[int, float],
    max_width: float,
) -> dict[int, int]:
    """A word split into several tokens by a style marker inside it ({b}R{/b}epresentations) is locked as one
    group and never broken at the marker. Markers opened before it move with the word. Words overlapping a
    bracket group or wider than the line are not locked."""
    group_of: dict[int, int] = {}
    count = len(tokens)
    index = 0
    while index < count:
        start = index
        while index < count and STYLE_MARK_RE.fullmatch(tokens[index]):
            index += 1
        if index >= count or not _WORD_PIECE_RE.fullmatch(tokens[index]):
            index = max(index, start + 1)
            continue
        end = index
        cursor = index + 1
        while cursor < count:
            probe = cursor
            while probe < count and STYLE_MARK_RE.fullmatch(tokens[probe]):
                probe += 1
            if probe == cursor or probe >= count or not _WORD_PIECE_RE.fullmatch(tokens[probe]):
                break
            end = probe
            cursor = probe + 1
        if end > index:
            members = range(start, end + 1)
            width = sum(_token_width(tokens[m], em, formula_widths) for m in members)
            if width <= max_width:
                for m in members:
                    group_of[m] = start
        index = end + 1
    return group_of


def _paren_groups(
    tokens: list[str],
    em: float,
    formula_widths: dict[int, float],
    max_width: float,
) -> dict[int, int]:
    """Map each token of a short bracket group to the index of its opening bracket. A group wider than the line is
    not locked and may be split."""
    group_of: dict[int, int] = {}
    index = 0
    count = len(tokens)
    while index < count:
        close = _PAREN_CLOSE.get(tokens[index])
        if close is None:
            index += 1
            continue
        end: int | None = None
        chars = 0
        width = 0.0
        for cursor in range(index, count):
            chars += _visible_len(tokens[cursor])
            width += _token_width(tokens[cursor], em, formula_widths)
            if chars > _PAREN_LIMIT:
                break
            if cursor > index and tokens[cursor] in _PAREN_CLOSE:
                break
            if tokens[cursor] == close and cursor > index:
                end = cursor
                break
        # The locking condition must match the overflow test used when filling: relaxing it to +0.5 lets a group
        # overflow while filling, and the rollback is followed by another rollback because the group is locked,
        # an infinite loop
        if end is not None and width <= max_width:
            for cursor in range(index, end + 1):
                group_of[cursor] = index
            index = end + 1
        else:
            index += 1
    return group_of


def nowrap_lines(lines: list[str]) -> str:
    """One nowrap span per line, written like bracket groups, so MuPDF cannot break a formula in the middle."""
    return "<br>".join(f'<span style="white-space:nowrap">{line}</span>' for line in lines)


def tokenize(text: str) -> list[str]:
    return _tokenize(text)


def measure_token(token: str, em: float, formula_widths: dict[int, float]) -> float:
    return _token_width(token, em, formula_widths)


def formula_index(token: str) -> int | None:
    matched = _SENTINEL_RE.fullmatch(token)
    return int(matched.group(1)) if matched else None


# Text right after a formula needs a space. None is added before punctuation or existing whitespace, and a second
# pass adds no more.
_NO_FORMULA_SPACE = frozenset("。，、；：？！）】」』〉》％%.,;:?!)]}>（【「『《〈([{")


_MARK_RUN = r"(?:\{/?[biu]\}|\{z\d+\}|\{/z\})*"
_CJK_CHAR = r"[぀-ヿ㐀-鿿가-힯]"
_LATIN_CHAR = r"[A-Za-z]"


def space_around_latin(text: str) -> str:
    """Put one space on each side of Latin text between Chinese characters.

    When an ordinal is written as {i} i-th{/i}, the space is inside the italic and the Chinese after it touches
    the letters. Spaces inside markers are moved out, a missing space at a boundary is added, and several are
    collapsed into one.
    """
    if not text or not _CJK_RE.search(text):
        return text
    text = re.sub(r"(\{(?:[biu]|z\d+)\})\s+", r" \1", text)
    text = re.sub(r"\s+(\{/(?:[biu]|z)\})", r"\1 ", text)
    text = re.sub(
        rf"({_CJK_CHAR})[ \t]*({_MARK_RUN})[ \t]*({_LATIN_CHAR})",
        r"\1 \2\3",
        text,
    )
    return re.sub(
        rf"({_LATIN_CHAR})[ \t]*({_MARK_RUN})[ \t]*({_CJK_CHAR})",
        r"\1\2 \3",
        text,
    )


def separate_after_formula(text: str) -> str:
    """Add a space when a formula sentinel is directly followed by a letter, digit or Chinese character."""
    if "\x01i\x02" not in text:
        return text
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        matched = _SENTINEL_RE.match(text, index)
        if matched is None:
            out.append(text[index])
            index += 1
            continue
        out.append(matched.group(0))
        index = matched.end()
        while True:
            mark = STYLE_MARK_RE.match(text, index)
            if mark is None:
                break
            out.append(mark.group(0))
            index = mark.end()
        nxt = text[index : index + 1]
        if nxt and not nxt.isspace() and nxt not in _NO_FORMULA_SPACE and _wants_formula_space(nxt):
            out.append(" ")
    return "".join(out)


def peel_leading_space(text: str) -> tuple[str, bool]:
    """The first space after removing style markers. HTML boxes swallow an ordinary leading space."""
    index = 0
    while True:
        mark = STYLE_MARK_RE.match(text, index)
        if mark is None:
            break
        index = mark.end()
    if index < len(text) and text[index] == " ":
        return text[:index] + text[index + 1 :], True
    return text, False


def _wants_formula_space(char: str) -> bool:
    if char.isalpha() or char.isdigit():
        return True
    if _CJK_RE.match(char):
        return True
    return unicodedata.east_asian_width(char) in ("W", "F")


def _inside_number(text: str, index: int) -> bool:
    return text[index] in ".," and 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        atom = _ATOM_RE.match(text, index) or _LINK_RE.match(text, index)
        if atom:
            tokens.append(atom.group(0))
            index = atom.end()
            continue
        char = text[index]
        if char.isspace() and char != "\n":
            end = index + 1
            while end < length and text[end].isspace() and text[end] != "\n":
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if _is_word_char(char):
            end = index + 1
            # A period or comma between digits, as in 2.07 or 10,000, belongs to the number and is no break point
            while end < length and (_is_word_char(text[end]) or _inside_number(text, end)):
                end += 1
                if _hyphen_break(text, end):
                    break
            tokens.append(text[index:end])
            index = end
            continue
        tokens.append(char)
        index += 1
    return tokens


# Minimum letters on both sides of a hyphen for a break after it: Super-|NaturalInstructions, pre-|training.
# GPT-4 and x-axis do not break
_HYPHEN_SIDE = 3
_HYPHEN_LEFT_RE = re.compile(rf"[A-Za-z]{{{_HYPHEN_SIDE}}}-$")
_HYPHEN_RIGHT_RE = re.compile(rf"[A-Za-z]{{{_HYPHEN_SIDE}}}")


def _hyphen_break(text: str, end: int) -> bool:
    """text[:end] ends with a hyphen at which the line may break."""
    return bool(_HYPHEN_LEFT_RE.search(text, max(0, end - _HYPHEN_SIDE - 1), end)) and bool(
        _HYPHEN_RIGHT_RE.match(text, end)
    )


def _is_word_char(char: str) -> bool:
    if _CJK_RE.match(char):
        return False
    # Identifiers such as l_partkey must not break at the underscore
    return char.isalnum() or char in "-'_"


_drawing_fonts: tuple[pymupdf.Font, ...] | None = None


def _drawing_fonts_loaded() -> tuple[pymupdf.Font, ...]:
    """Line breaking needs the widths actually written. Estimating dashes and M at 0.5em would squeeze an extra
    character onto a line."""
    global _drawing_fonts
    if _drawing_fonts is None:
        _drawing_fonts = (
            pymupdf.Font("china-ss"),
            pymupdf.Font("heit"),
            pymupdf.Font("hebo"),
        )
    return _drawing_fonts


def measure_width(text: str, em: float, formula_widths: dict[int, float] | None = None) -> float:
    """Width of a piece of translation. Style markers are 0, formula sentinels take their slot width, or one em
    without one."""
    widths = formula_widths or {}
    cleaned = strip_style_marks(text)
    total = 0.0
    pos = 0
    for match in _SENTINEL_RE.finditer(cleaned):
        total += sum(_char_width(ch, em) for ch in cleaned[pos : match.start()])
        total += widths.get(int(match.group(1)), em)
        pos = match.end()
    total += sum(_char_width(ch, em) for ch in cleaned[pos:])
    return total


# Line breaking measures every character, millions of times for a book; the width depends only on the character
# and the size, so measured widths are cached
@functools.lru_cache(maxsize=65536)
def _char_width(char: str, em: float) -> float:
    if _CJK_RE.match(char) or unicodedata.east_asian_width(char) in ("W", "F"):
        estimate = em
    elif char.isspace():
        estimate = em * 0.33
    else:
        estimate = em * 0.5
    # When the font has the character, measure the written width. The estimate is only for characters no font
    # has: with 0.5em as the default, i, l and commas come out twice too wide, mixed Chinese and English lines
    # break before they are full, and justification cannot fill them
    drawn = 0.0
    code = ord(char)
    for font in _drawing_fonts_loaded():
        if font.has_glyph(code):
            drawn = max(drawn, font.text_length(char, em))
    return drawn or estimate


def _token_width(token: str, em: float, formula_widths: dict[int, float]) -> float:
    sentinel = _SENTINEL_RE.fullmatch(token)
    if sentinel:
        return formula_widths.get(int(sentinel.group(1)), em)
    if STYLE_MARK_RE.fullmatch(token):
        return 0.0
    return sum(_char_width(char, em) for char in token)
