"""Move the halves of lines that PyMuPDF cut in the middle back into the block of the first half."""

import re
from collections import Counter
from collections.abc import Callable, Sequence

import pymupdf

from .pdf_blocks import (
    _by_chars,
    _span_is_math,
)
from .pdf_math import (
    _line_box,
    _plain_units,
)
from .pdf_rows import _text_box
from .pdf_split_lines import _math_seam
from .pdf_types import Block, Line, PageDict, Span

# Column edge: at least this many lines, each at least this many ems long, start at the same x
_COLUMN_LINES = 3


_COLUMN_LINE_EMS = 15

# The box of a line used for band comparisons: _line_box, or a cached wrapper of it
LineBox = Callable[[Line], pymupdf.Rect]


def _rejoin_split_lines(data: PageDict) -> None:
    """PyMuPDF sometimes cuts a line in the middle and puts the second half into another block (often at
    superscripts and subscripts). Move the second half back into the block of the first half, right after that
    line.

    Left in place, a half such as "and β₂ᵗ" at the end of a line looks like a formula line on its own and is
    skipped as a formula: the sentence loses a few words and the original text stays on the page. The two
    halves must meet end to start on the same line. When every line of the paragraph stops before the column
    gap, what lies to the right is the next column and is not joined, even across a narrow gap; a piece with a
    clearly larger font size is not joined either.
    """
    blocks = [block for block in data.get("blocks", []) if block.get("type") == 0]
    if len(blocks) < 2:
        return
    # Every line is compared with all lines on the page, and computing line boxes each time takes tens of
    # seconds. Lines only change blocks and their content stays the same, so each box is computed once
    boxes: dict[int, pymupdf.Rect] = {}

    def line_box(line: Line) -> pymupdf.Rect:
        box = boxes.get(id(line))
        if box is None:
            box = boxes[id(line)] = _line_box(line)
        return box

    starts = _column_starts(blocks)
    # Only after one piece has moved does a numerator left in the original block lose its companion, so it
    # finds its destination in the next pass
    for _ in range(3):
        if not _rejoin_pass(blocks, line_box, starts):
            break
    _embed_math_blocks(blocks, line_box)


def _column_starts(blocks: list[Block]) -> list[float]:
    """Left edges where several long lines start together: the edges of the columns."""
    counter: Counter[int] = Counter()
    for block in blocks:
        for line in block["lines"]:
            x0, _, x1, _ = line["bbox"]
            size = max((span["size"] for span in line["spans"]), default=10.0)
            if abs(line["dir"][1]) <= 0.01 and x1 - x0 >= size * _COLUMN_LINE_EMS:
                counter[round(x0)] += 1
    return [x for x in counter if sum(n for other, n in counter.items() if abs(other - x) <= 1) >= _COLUMN_LINES]


def _rejoin_pass(blocks: list[Block], line_box: LineBox | None = None, starts: Sequence[float] = ()) -> bool:
    line_box = line_box or _line_box
    moved = False
    for block in blocks:
        for line in list(block["lines"]):
            if abs(line["dir"][1]) > 0.01 or not "".join(s["text"] for s in line["spans"]).strip():
                continue
            host = _line_host(line, block, blocks, line_box, starts) or _stack_host(line, block, blocks)
            if host is None:
                continue
            host_block, host_line = host
            block["lines"].remove(line)
            index = host_block["lines"].index(host_line) + 1
            # When pieces of the same line have already moved here, go after them. Compared by font box, so
            # numerators, denominators and punctuation all belong to this line
            while index < len(host_block["lines"]) and _same_band(host_block["lines"][index], host_line, _font_box):
                index += 1
            host_block["lines"].insert(index, line)
            host_block["bbox"] = tuple(pymupdf.Rect(host_block["bbox"]) | pymupdf.Rect(line["bbox"]))
            moved = True
    return moved


def _embed_math_blocks(blocks: list[Block], line_box: LineBox) -> None:
    """The pieces at one line height in a block are all math (a numerator, or half an expression joined to a
    numerator), they overlap the math on a line of another block, and they share the height of a body line in
    that block: PyMuPDF has split the math out of that line, so move them back together.
    Moving line by line does not work: the pieces are superscripts and subscripts of one another, so each one
    on its own has a line of the same block next to it. Pieces left in the original block are treated as a
    formula line and keep the original text; after the rest of the line is translated and moved, they remain
    alone in the old place and the formula is missing a piece."""
    for block in blocks:
        shown = [
            line
            for line in block["lines"]
            if abs(line["dir"][1]) <= 0.01 and "".join(span["text"] for span in line["spans"]).strip()
        ]
        maths = [line for line in shown if all(_span_is_math(span) for span in line["spans"] if span["text"].strip())]
        while maths:
            group = [line for line in maths if _same_band(line, maths[0], line_box)]
            maths = [line for line in maths if not any(line is member for member in group)]
            box = pymupdf.Rect()
            for line in group:
                box |= line_box(line)
            # The same line also has body text of this block, so the math belongs to this block
            if any(_band_overlap(line_box(line), box) for line in shown if not any(line is member for member in group)):
                continue
            spans = [span for line in group for span in line["spans"] if span["text"].strip()]
            under = _math_host(blocks, block, spans, box, line_box)
            if under is None:
                continue
            host, anchor = under
            index = host["lines"].index(anchor) + 1
            host["lines"][index:index] = group
            block["lines"] = [line for line in block["lines"] if not any(line is member for member in group)]
            host["bbox"] = tuple(pymupdf.Rect(host["bbox"]) | box)


def _math_host(
    blocks: list[Block],
    own: Block,
    spans: list[Span],
    box: pymupdf.Rect,
    line_box: LineBox,
) -> tuple[Block, Line] | None:
    """The block with a math line that box lies on, where that block also has a body line (three or more body
    words) at the same line height. A box level with formula lines only is not moved: moving a split brace of a
    display formula into another piece of the formula is of no use."""
    size = max(float(span["size"]) for span in spans)
    for host in blocks:
        if host is own or not any(
            _band_overlap(line_box(line), box)
            and _plain_units([span for span in line["spans"] if span["text"].strip()]) >= 3
            for line in host["lines"]
        ):
            continue
        under = _stacked_on(host, {"spans": spans}, box, size, False)
        if under is not None:
            return host, under
    return None


def _line_host(
    line: Line,
    own: Block,
    blocks: list[Block],
    line_box: LineBox | None = None,
    starts: Sequence[float] = (),
) -> tuple[Block, Line] | None:
    """The piece right next to the left of line. Nothing moves when it is already in line's block:
    taking the first one found, when two blocks each have a piece that could join, line would move to the
    other block on every pass.
    When line starts at the edge of a column and is more than half an em from the piece on its left, it is
    text of the neighbouring column (a section number 4.2.3); the second half of a line cut in the middle does
    not fall exactly on a column edge."""
    line_box = line_box or _line_box
    box = pymupdf.Rect(line["bbox"])
    size = max((span["size"] for span in line["spans"]), default=10.0)
    column_edge = any(abs(box.x0 - x) <= 1 for x in starts)
    column_font = _column_font(line, own, size)
    pieces = [span for span in line["spans"] if span["text"].strip()]
    all_math = bool(pieces) and all(_span_is_math(span) for span in pieces)
    best: tuple[float, Block, Line] | None = None
    for block in blocks:
        for other in block["lines"]:
            if other is line or abs(other["dir"][1]) > 0.01:
                continue
            # Filter by horizontal position first, since vertical overlap needs line boxes and is slow. Every
            # line is compared with all lines on the page, so read coordinates directly without building a Rect
            left_x0, _, left_x1, _ = other["bbox"]
            gap = box.x0 - left_x1
            # Small fractions and limits that make up a whole piece of math are one size smaller than body
            # text; the gap to the math on the left is measured with the larger size of the two
            reach = max(size, max((span["size"] for span in other["spans"]), default=size)) if all_math else size
            if left_x0 >= box.x0 or gap < -size or gap > reach:
                continue
            if not _same_band(other, line, line_box):
                continue
            # Superscripts and subscripts can make the two pieces overlap by one em where they meet; a gap
            # wider than word spacing is a column gap.
            # Gaps around equals signs and fractions are wider than word spacing, so when one side is math the
            # limit is relaxed to one em
            if _past_column(block, other, box.x0, size, line_box) or _larger_text(line, other):
                continue
            if column_edge and gap > size * 0.5:
                continue
            if column_font and _main_font(other) != column_font:
                continue
            left = [span for span in other["spans"] if span["text"].strip()]
            if (
                gap <= size * 0.6
                or (gap <= size and _math_seam(other["spans"], line["spans"]))
                or (gap <= reach and bool(left) and _span_is_math(left[-1]))
            ):
                if best is None or abs(gap) < abs(best[0]):
                    best = (gap, block, other)
    if best is None or best[1] is own:
        return None
    return best[1], best[2]


def _column_font(line: Line, own: Block, size: float) -> str:
    """The font of this column, when the line above or below in the block aligns with it on the left, sits
    right against it and uses the same font.
    A line on the left in a different font is not its first half: an example set in a monospaced font in the
    right column of a slide sits at the same height as the body text on the left.
    An interlinear glossed example has different fonts on its two lines and does not count."""
    x0, y0, _, y1 = line["bbox"]
    font = _main_font(line)
    for other in own["lines"]:
        if other is line or abs(other["dir"][1]) > 0.01:
            continue
        ox0, oy0, _, oy1 = other["bbox"]
        if (
            abs(ox0 - x0) <= 1
            and (0 <= y0 - oy1 < size * 0.6 or 0 <= oy0 - y1 < size * 0.6)
            and _main_font(other) == font
        ):
            return font
    return ""


def _main_font(line: Line) -> str:
    spans = [span for span in line["spans"] if span["text"].strip()]
    return str(_by_chars(spans, lambda span: span.get("font") or "")) if spans else ""


def _font_box(line: Line) -> pymupdf.Rect:
    return pymupdf.Rect(line["bbox"])


def _past_column(block: Block, line: Line, x: float, size: float, line_box: LineBox | None = None) -> bool:
    """All other lines of line's paragraph stop before x, and one of them stops at the same right edge as line
    (a justified column): x is outside the column and the text there belongs to the neighbouring column. A
    line cut in the middle is cut inside the column, and the lines above and below extend past that point.
    Only full lines wide enough and starting at the same left edge as line are considered: small lines such
    as numerators and denominators do not count; the short last line of a paragraph does not reach the right
    edge and is not enough on its own."""
    box = pymupdf.Rect(line["bbox"])
    rows = [
        pymupdf.Rect(other["bbox"])
        for other in block["lines"]
        if other is not line
        and not _same_band(other, line, line_box)
        and abs(pymupdf.Rect(other["bbox"]).x0 - box.x0) <= size
        and pymupdf.Rect(other["bbox"]).width >= size * 8
    ]
    return bool(rows) and all(row.x1 < x for row in rows) and any(abs(row.x1 - box.x1) <= size * 0.5 for row in rows)


def _larger_text(line: Line, host: Line) -> bool:
    """The body text in line is clearly larger than in host: it is a heading in the neighbouring column on the
    same baseline, since the second half of a cut line keeps its size.
    A piece with only punctuation does not count: the period after an expression is at body size and follows
    a denominator one size smaller (Σ_j e^{S·T_j} .)."""
    sizes = [
        float(span["size"])
        for span in line["spans"]
        if any(ch.isalnum() for ch in span["text"]) and not _span_is_math(span)
    ]
    host_size = max((float(span["size"]) for span in host["spans"] if span["text"].strip()), default=0.0)
    return bool(sizes) and host_size > 0 and max(sizes) > host_size * 1.2


def _stack_host(line: Line, own: Block, blocks: list[Block]) -> tuple[Block, Line] | None:
    """Small blocks stacked above or below math, such as numerators, denominators and summation limits: the
    whole block is formula, it falls horizontally within the extent of a math line, and it sits right above
    or below that line. Body lines above and below also cover it horizontally, but they are not math.

    A small block left alone in a block is collected this way. One that shares its block with other lines is
    a superscript or subscript of that block when math in the block sits against it, and is left there; when
    the block has none and it is one size smaller than the host line, it is a denominator that was put into
    the block of the next line (the N of count(w)/N).
    In books set in Times, variables are body italics and cannot be recognized as math by font. Smaller
    italics stacked on smaller italics in a line of body size are also numerators and denominators. Numeric
    fractions in the body font (24000/75) are likewise recognized by the smaller size and centering."""
    spans = [span for span in line["spans"] if span["text"].strip()]
    if not spans or not all(_span_is_math(span) or _italic_span(span) or _numeral_span(span) for span in spans):
        return None
    italic = not all(_span_is_math(span) for span in spans)
    box = _line_box(line)
    size = max(float(span["size"]) for span in spans)
    shared = len(own["lines"]) > 1
    if shared and _stacked_on(own, line, box, size, italic) is not None:
        return None
    for block in blocks:
        if block is own:
            continue
        other = _stacked_on(block, line, box, size, italic)
        if other is None:
            continue
        if (shared or italic) and size >= _line_size(other) * 0.79:
            return None
        return block, other
    return None


def _italic_span(span: Span) -> bool:
    return bool(_ITALIC_FONT_RE.search(str(span.get("font") or "")) or int(span.get("flags") or 0) & 2)


def _numeral_span(span: Span) -> bool:
    return bool(_NUMERAL_RE.fullmatch(str(span.get("text") or "").strip()))


def _line_size(line: Line) -> float:
    return max((float(span["size"]) for span in line["spans"] if span["text"].strip()), default=0.0)


def _stacked_on(block: Block, line: Line, box: pymupdf.Rect, size: float, italic: bool) -> Line | None:
    """The math line in block directly above or below line and touching it. It must have math characters over
    line; covering it with the line's extent is not enough."""
    lines: list[Line] = block["lines"]
    for other in lines:
        if other is line or abs(other["dir"][1]) > 0.01:
            continue
        small = _line_size(other) * 0.79
        for span in other["spans"]:
            if not span["text"].strip() or "bbox" not in span:
                continue
            # For an italic line only smaller numerators and denominators count; the K and n above and below a
            # large operator such as a summation sign are not moved
            if italic and not (
                float(span["size"]) < small and (_span_is_math(span) or _italic_span(span) or _numeral_span(span))
            ):
                continue
            if not italic and not _span_is_math(span):
                continue
            under = _text_box(span)
            if min(under.x1, box.x1) - max(under.x0, box.x0) <= 0:
                continue
            if max(under.y0 - box.y1, box.y0 - under.y1) > size * 0.5:
                continue
            # Italic numerators and denominators are centered on each other. The superscript and subscript of
            # w^i_{i-k+1} are aligned on the left and are not a fraction
            if italic:
                band = _small_band(other["spans"], box, small)
                if band is None or abs((box.x0 + box.x1) - (band.x0 + band.x1)) / 2 > max(
                    1.0, 0.15 * min(box.width, band.width)
                ):
                    continue
            return other
    return None


def _same_band(a: Line, b: Line, box: LineBox | None = None) -> bool:
    """Enough vertical overlap. Denominators of fractions and large braces pull the box down, so two pieces of
    the same line overlap by less than half.
    By default the comparison uses glyph height: the font box of a math font reaches into the next line and
    overlaps it, and a piece would move back and forth between the two lines."""
    box = box or _line_box
    return _band_overlap(box(a), box(b))


def _band_overlap(ra: pymupdf.Rect, rb: pymupdf.Rect) -> bool:
    overlap = min(ra.y1, rb.y1) - max(ra.y0, rb.y0)
    return bool(overlap >= min(ra.height, rb.height) * 0.35)


# Expressions not set in a math font (older papers use Times-Italic for variables and Symbol for operators):
# variables in body italics, brackets, operators.
# A variable is a single letter or two capital letters (PE). Two-letter words such as If and op count only
# together with brackets or operators (op c)); theorem bodies are italic throughout, and If or In next to a
# formula is still body text
_ITALIC_FONT_RE = re.compile(r"Italic|Oblique", re.I)


_NUMERAL_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _small_band(run_spans: list[Span], over: pymupdf.Rect, small: float) -> pymupdf.Rect | None:
    """The whole band of small text in run_spans that falls horizontally below or above over (a denominator or a
    numerator)."""
    spans = [span for span in run_spans if str(span.get("text") or "").strip() and float(span["size"]) < small]
    seeds = [
        pymupdf.Rect(span["bbox"])
        for span in spans
        if min(over.x1, span["bbox"][2]) - max(over.x0, span["bbox"][0]) > 0
    ]
    if not seeds:
        return None
    top, bottom = min(seed.y0 for seed in seeds), max(seed.y1 for seed in seeds)
    band = pymupdf.Rect()
    for span in spans:
        box = pymupdf.Rect(span["bbox"])
        if top <= (box.y0 + box.y1) / 2 <= bottom:
            band |= box
    return band
