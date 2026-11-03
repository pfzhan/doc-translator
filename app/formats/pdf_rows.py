"""Split text lines into rows and cells: rejoin cut lines, split table cells, contents leaders, list items and
stacked or side-by-side pieces."""

import re
from collections import Counter
from collections.abc import Callable, Collection, Sequence

import pymupdf

from .pdf_blocks import (
    _MARK_ITEM_RE,
    _ROMAN_ITEM_RE,
    CJK_RE,
    Leader,
    _by_chars,
    _ends_sentence,
    _list_item,
    _span_is_math,
)
from .pdf_flow import (
    emphasis_of,
)
from .pdf_math import (
    _line_box,
    _plain_units,
    _prose_units,
    _text_bounds,
)

MONO_FONT_RE = re.compile(
    r"Courier|NimbusMon|Mono|Consolas|Menlo|Monaco|CMTT|Typewriter|Inconsolata|Fira ?Code|Lucida ?Console|Source ?Code",
    re.I,
)


NO_TEXT_RE = re.compile(r"^[\W\d_]*$")


# Operators and relations left at the end of a line when an expression wraps. Hyphens used as empty values
# in tables, and digits, do not count
_OPERATOR_RE = re.compile(r"[=+−±×·<>≤≥≈≃≅∼≡≠→←⇒⇔∈∉⊂⊆∝]+")


# A single letter and a period, as in "a. ", which also matches "Q. The cost" after a line break. The latter
# is not a list item.
_LETTER_BULLET_RE = re.compile(r"^[a-zA-Z][.)]\s+")


_DASH_BULLET_RE = re.compile(r"^[-–—]\s+")


# Example sentences numbered by chapter in a book: (1.2) It was difficult ...
_EXAMPLE_NUMBER_RE = re.compile(r"^\(\d+\.\d+[a-z]?\)\s+\S")


# A number standing alone as one piece: (25.4) or (3) for an example or an expression, (a) for a list
_PAREN_LABEL_RE = re.compile(r"\((?:[a-zA-Z]|\d+(?:\.\d+)*)\)")


_LIST_INTRO_END = (".", "?", "!", "。", "！", "？", ":", ";", "：", "；")


# Start of a new paragraph: capital letter, digit, quote or bracket (a lowercase start is usually a line that
# continues after automatic wrapping).
# CJK is not included: at the start of a CJK line a new sentence cannot be told from a continuation line, so
# it counts as a new paragraph only when the line above ends with sentence-final punctuation (see below)
SEG_START_RE = re.compile(r"^[A-Z0-9À-Þ(\"'“‘\[（【]")


CJK_START_RE = re.compile(r"^[぀-ヿ㐀-鿿가-힯]")


def _same_line_word(
    prev: str,
    nxt: str,
    prev_rect: pymupdf.Rect,
    rect: pymupdf.Rect,
    size: float,
) -> bool:
    """The next word on the same baseline, within word spacing. When a line is split by word, a capitalized
    word inside the line still belongs to this sentence."""
    height = min(prev_rect.height, rect.height)
    overlap = min(prev_rect.y1, rect.y1) - max(prev_rect.y0, rect.y0)
    if height <= 0 or overlap <= height * 0.5:
        return False
    gap = rect.x0 - prev_rect.x1
    if not (-1 <= gap < max(size * 1.4, 12)):
        return False
    # A number is a label and is not joined to the sentence on its right: once joined, the left edge of the
    # block moves back to the number and the hanging indent of the example no longer lines up
    if _ends_sentence(prev) or _PAREN_LABEL_RE.fullmatch(prev.strip()):
        return False
    head = nxt.lstrip()[:1]
    if not head.isalpha():
        return False
    if prev.lstrip()[:1] == "<" or nxt.lstrip()[:1] == "<":
        return False
    if "%" in prev or prev.rstrip()[-1:] in "!%":
        return False
    return True


# Numbered labels of theorems, lemmas, definitions and the like: Lemma 10.3., Theorem 4.1., Definition 2.
_NUMBERED_LABEL_RE = re.compile(r"[A-Z][a-z]+\s+\d+(?:\.\d+)*\.\s")


def _label_lead(spans: list[dict]) -> bool:
    """The line starts with a bold label ending in a colon or period, followed by body text that is not bold
    (Speaker demographics: What..., On the Optimality of the Level Schedule. Recall...). One bold word followed
    by a number is a label of the theorem kind (Proposition 5 (Efficiency ...). Under...); the number is often
    set in another font. A bold Q2 or RQ1 is a numbered item"""
    # Spans split by word have no space span between them; where they are apart, add a space
    texts = [str(span.get("text") or "") for span in spans]
    for k in range(1, len(spans)):
        before, after = spans[k - 1].get("bbox"), spans[k].get("bbox")
        size = float(spans[k].get("size") or 10)
        if before and after and after[0] - before[2] > 0.1 * size and not texts[k - 1].endswith(" "):
            texts[k] = " " + texts[k]
    label = ""
    for index, span in enumerate(spans):
        text = texts[index]
        if emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)).bold:
            label += text
            continue
        if not label.strip():
            return False
        rest = "".join(texts[index:]).strip()
        phrase = label.strip()
        # Numbered items such as research questions and hypotheses (Q2 Which..., RQ1 How..., H3 ...) also count
        numbered = re.fullmatch(r"[A-Z]{1,3}\d+", phrase) is not None or (
            re.fullmatch(r"[A-Z][a-z]+(?:\s+\d+(?:\.\d+)*)?", phrase) is not None
            and (any(ch.isdigit() for ch in phrase) or re.match(r"\d+(?:\.\d+)*\b", rest) is not None)
        )
        if numbered:
            return _prose_units(rest) >= 2
        if rest[:1] in ":.":
            label, rest = label + rest[0], rest[1:].strip()
        phrase = label.strip()
        # The line after the label may wrap with only one word left (... Batch Instances. A | batch consists of ...)
        return (
            phrase.endswith((":", "."))
            and len(phrase.split()) <= _LABEL_WORDS
            and re.search(r"[^\W\d_]", rest) is not None
        )
    return False


# Maximum number of words in a bold label at the start of a line. Anything longer is a whole sentence set in
# bold for emphasis
_LABEL_WORDS = 10


def _justified_joins(items: list[tuple], max_x1: float, whole_rows: bool = False) -> set[int]:
    """Indices of the later words when PyMuPDF splits a justified line by word.

    A justified heading in a narrow column can have word spacing stretched to two ems; splitting by word
    spacing then gives one word per piece.
    How it is recognized: same baseline, same font and size, the last word reaches the right edge of the
    block, and the word gaps are about equal.
    A line split into only two pieces is checked by _full_row.
    Cells side by side and side labels differ: the gaps between cells vary in size and do not fill a line
    exactly.
    When whole_rows is true (an index page, where one block is one column with no cells or side labels), the
    pieces on the same row are all one line.
    """
    joins: set[int] = set()
    start = 0
    while start < len(items):
        end = start + 1
        while end < len(items) and _same_row(items[end - 1], items[end]):
            end += 1
        row = items[start:end]
        if (
            whole_rows
            and len(row) >= 2
            or len(row) >= 3
            and _spread_evenly(row, max_x1)
            or len(row) == 2
            and _full_row(row, items, max_x1)
        ):
            joins.update(range(start + 1, end))
        start = end
    return joins


def _full_row(row: list[tuple], items: list[tuple], max_x1: float) -> bool:
    """The two pieces together are one full line of a paragraph: the first starts at the left edge of the
    paragraph, the second reaches the right edge, only one wide space lies between them, and the block has
    other full lines from the same left edge to the same right edge. In justified text the space after a
    period is stretched wide and PyMuPDF splits there.
    Two pieces on a line of their own may be a side label and body text, and do not count."""
    left, right = row[0][1], row[1][1]
    size = max((float(s.get("size") or 0) for s in row[0][2]), default=10.0)
    if right.x1 < max_x1 - 1 or not 0 < right.x0 - left.x1 <= size * 1.5:
        return False
    return any(
        abs(item[1].x0 - left.x0) <= 1 and item[1].x1 >= max_x1 - 1
        for item in items
        if item is not row[0] and item is not row[1]
    )


def _same_row(left: tuple, right: tuple) -> bool:
    a, b = left[1], right[1]
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap < min(a.height, b.height) * 0.7 or b.x0 <= a.x1:
        return False

    # Same size, sharing a font. A piece with an italic variable inside it (AVG(E)) also counts; cells side by
    # side often differ entirely in font and size
    def styles(item: tuple) -> set[tuple]:
        return {
            (s.get("font"), round(float(s.get("size") or 0), 1)) for s in item[2] if str(s.get("text") or "").strip()
        }

    ours, theirs = styles(left), styles(right)
    return len({size for _, size in ours | theirs}) == 1 and bool(ours & theirs)


def _spread_evenly(row: list[tuple], max_x1: float) -> bool:
    size = max((float(s.get("size") or 0) for s in row[0][2]), default=10.0)
    if row[-1][1].x1 < max_x1 - 1:
        return False
    gaps = [b[1].x0 - a[1].x1 for a, b in zip(row, row[1:])]
    # The gap after a section number is left out: the distance in "2  Towards" is set by the typesetting and
    # is not stretched by justification
    if re.fullmatch(r"\d+(?:\.\d+)*\.?", row[0][0].strip()):
        gaps = gaps[1:]
    if not gaps or max(gaps) > size * 3:
        return False
    return max(gaps) <= min(gaps) * 1.5 + size * 0.2


def _text_box(span: dict) -> pymupdf.Rect:
    """The height of the line the span occupies. Math fonts are measured by glyph ink, the rest by PyMuPDF's
    box.

    Radical signs and large braces have their origin at the top of the glyph instead of on the baseline, so
    only the ink can be used."""
    return pymupdf.Rect(_text_bounds(span))


# Column edge: at least this many lines, each at least this many ems long, start at the same x
_COLUMN_LINES = 3


_COLUMN_LINE_EMS = 15


def _rejoin_split_lines(data: dict) -> None:
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

    def line_box(line: dict) -> pymupdf.Rect:
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


def _column_starts(blocks: list[dict]) -> list[float]:
    """Left edges where several long lines start together: the edges of the columns."""
    counter: Counter[int] = Counter()
    for block in blocks:
        for line in block["lines"]:
            x0, _, x1, _ = line["bbox"]
            size = max((span["size"] for span in line["spans"]), default=10.0)
            if abs(line["dir"][1]) <= 0.01 and x1 - x0 >= size * _COLUMN_LINE_EMS:
                counter[round(x0)] += 1
    return [x for x in counter if sum(n for other, n in counter.items() if abs(other - x) <= 1) >= _COLUMN_LINES]


def _rejoin_pass(blocks: list[dict], line_box=None, starts: Sequence[float] = ()) -> bool:
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


def _embed_math_blocks(blocks: list[dict], line_box) -> None:
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
    blocks: list[dict],
    own: dict,
    spans: list[dict],
    box: pymupdf.Rect,
    line_box,
) -> tuple[dict, dict] | None:
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


def _stacked_rows(upper: list[dict], lower: list[dict]) -> bool:
    """Both pieces are body text of the same size and their baselines differ by more than about a third of the
    font size: they are two lines close above each other, not one line (an action line in a figure pressed onto
    the caption, with ink boxes overlapping by more than half). Superscripts and subscripts are smaller, and
    pieces split from a formula use a math font; neither counts."""
    pieces = [[span for span in spans if str(span.get("text") or "").strip()] for spans in (upper, lower)]
    if not all(pieces) or any("origin" not in span or _span_is_math(span) for part in pieces for span in part):
        return False
    size = max(span["size"] for span in pieces[0])
    if abs(max(span["size"] for span in pieces[1]) - size) > size * 0.1:
        return False
    # The baseline is taken from characters at the main size of the piece; the small numerators and
    # denominators of inline fractions sit on other baselines
    baselines = [
        float(_by_chars([span for span in part if span["size"] >= size * 0.9], lambda span: span["origin"][1]))
        for part in pieces
    ]
    return abs(baselines[0] - baselines[1]) > size * 0.3


def _left_to_right(items: list[tuple]) -> list[tuple]:
    """Sort the pieces of one visual line from left to right. The numerator of a fraction often comes before
    the denominator, which then no longer follows the equals sign before it.
    Line boxes are ink boxes, so two lines above each other overlap little and are not taken as one line."""
    out: list[tuple] = []
    run: list[tuple] = []
    for item in items:
        if run:
            first, rect = run[0][1], item[1]
            overlap = min(first.y1, rect.y1) - max(first.y0, rect.y0)
            if overlap >= min(first.height, rect.height) * 0.5 and not _stacked_rows(run[0][2], item[2]):
                run.append(item)
                continue
        out.extend(sorted(run, key=lambda piece: piece[1].x0))
        run = [item]
    out.extend(sorted(run, key=lambda piece: piece[1].x0))
    return out


# Leader and page number at the end of a line: at least four dots (spaces allowed between them), followed by
# an Arabic or Roman page number
_LEADER_RE = re.compile(r"(?:[.·…]\s*){4,}([0-9]{1,4}|[ivxlcdm]{1,7})\s*$", re.I)


# When a title on a contents page is long and runs up to the page number, the leader has only two or three
# dots left, with spaces between them
_SHORT_LEADER_RE = re.compile(r"\s(?:[.·…]\s+){2,3}([0-9]{1,4}|[ivxlcdm]{1,7})\s*$", re.I)


_PAGE_NUMBER_RE = re.compile(r"\s*([0-9]{1,4}|[ivxlcdm]{1,7})\s*", re.I)


_LEADER_DOTS_RE = re.compile(r"[\s.·…]*[.·…][\s.·…]*")


# Index entries end with page numbers: "fully-connected, 157", "linear chain CRF, 414, 415", "Kleene *, 50–52"
_INDEX_REF_RE = re.compile(r",\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*$")


# When an entry is too long, its page numbers wrap to the next line, which has only page numbers:
# "BNF (Backus-Naur form)," with "426" indented on the next line
_INDEX_REFS_ONLY_RE = re.compile(r"^\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*,?$")


# Word grid: at least this many lines, and several columns each with this many lines starting at the same x
_GRID_LINES = 9


_GRID_COLUMNS = 3


_GRID_ROWS = 3


def _grid_cells(items: list[tuple]) -> set[int]:
    """Cells of a word grid inside one block, one word per cell (FrameNet's VERBS: dwindle move soar ...):
    short lines of one or two words whose left edges fall on three or more columns of several lines each, and
    which lie vertically within the range of those columns. Merging lines into paragraphs would join the cells
    of one row into one sentence and stack the translated columns on top of each other. Captions and labels
    above or below the grid in the same block (Figure 2.4) are not cells."""
    if len(items) < _GRID_LINES:
        return set()
    short = [index for index, item in enumerate(items) if _grid_cell(item[0])]
    if len(short) < len(items) * 0.8:
        return set()
    starts = Counter(round(items[index][1].x0 / 2) for index in short)
    columns = {start for start, count in starts.items() if count >= _GRID_ROWS}
    members = [index for index in short if round(items[index][1].x0 / 2) in columns]
    if len(columns) < _GRID_COLUMNS or len({round(items[index][1].y0 / 2) for index in members}) < _GRID_ROWS:
        return set()
    top = min(items[index][1].y0 for index in members)
    bottom = max(items[index][1].y1 for index in members)
    return {index for index in short if top - 1 <= items[index][1].y0 and items[index][1].y1 <= bottom + 1}


def _grid_cell(text: str) -> bool:
    """One cell of a word grid: one or two words. Captions and row labels in the same block are split into
    paragraphs as usual."""
    return len(text.split()) <= 2 and len(text.strip()) <= 25


def _index_entry_ends(items: list[tuple], ref_rows: Sequence[pymupdf.Rect]) -> set[int]:
    """The last line (index) of each entry in a block of an index page. When an entry wraps, its continuation
    lines are indented deeper than its first line, all by the same amount: when the next line starts no
    further right than the first line of this entry, or steps back from the continuation indent (to a
    subentry), this line closes an entry ("chain rule" after "CFG, see context-free gram-" "mar").
    A line ending with page numbers, or holding only page numbers, also closes an entry. When page numbers
    wrap to the next line, a page number line in the same block goes with the entry; one in another block
    cannot be taken in, and the entry closes at the line that wraps.
    A line that justification splits into several pieces (collaborative | completion,) is treated as a whole
    line."""
    rows: list[list[int]] = []
    for index, item in enumerate(items):
        box = item[1]
        if rows and min(box.y1, items[rows[-1][0]][1].y1) - max(box.y0, items[rows[-1][0]][1].y0) > box.height * 0.5:
            rows[-1].append(index)
        else:
            rows.append([index])
    ends: set[int] = set()
    first_x0: float | None = None
    for at, row in enumerate(rows):
        rect = pymupdf.Rect()
        for index in row:
            rect |= items[index][1]
        continued = first_x0 is not None
        if first_x0 is None:
            first_x0 = rect.x0
        text = " ".join(items[index][0].strip() for index in row)
        size = max(span["size"] for index in row for span in items[index][2])
        nxt = rows[at + 1] if at + 1 < len(rows) else None
        below = None
        if nxt is not None:
            below = pymupdf.Rect()
            for index in nxt:
                below |= items[index][1]
        # The page numbers wrapped to the next line, which is in the same block: the entry is sent for
        # translation together with its page numbers and closes only at the page number line.
        # If the line with only page numbers were left on its own, this entry would have no page numbers in the
        # translation
        refs_next = (
            text.endswith(",")
            and below is not None
            and below.y0 - rect.y1 <= size * 0.5
            and _INDEX_REFS_ONLY_RE.match(" ".join(items[index][0].strip() for index in nxt)) is not None
        )
        # The line stops at half a word broken by a hyphen and the next line starts in lowercase: it is the
        # second half of the same word (resolu- | tion, 566).
        # When the block starts with a continuation line of an entry, this line is not the first line of the
        # entry, and the entry does not close even when the next line aligns with it
        broken = (
            re.search(r"[A-Za-z]-$", text) is not None
            and below is not None
            and below.y0 - rect.y1 <= size * 0.5
            and items[nxt[0]][0].lstrip()[:1].islower()
        )
        if refs_next or broken:
            continue
        if (
            _INDEX_REF_RE.search(text)
            or _INDEX_REFS_ONLY_RE.match(text)
            or (text.endswith(",") and _refs_below(rect, ref_rows, size))
            or (
                below is not None
                and below.y0 - rect.y1 <= size * 0.5
                and (below.x0 <= first_x0 + 1 or (continued and below.x0 < rect.x0 - 1))
            )
        ):
            ends.add(row[-1])
            first_x0 = None
    return ends


def _refs_below(rect: pymupdf.Rect, ref_rows: Sequence[pymupdf.Rect], size: float) -> bool:
    """The page numbers of the entry wrapped to the next line: a line with only page numbers sits right below
    and starts between the left and right edges of this line.
    When this line is itself a continuation line, the page numbers have the same indent (branching factor, |
    77)."""
    return any(
        -size * 0.3 <= box.y0 - rect.y1 <= size * 0.5 and rect.x0 - size * 0.5 <= box.x0 < rect.x1 for box in ref_rows
    )


# One number cell in a table: 1.5, -, 86.2%, 12.0/15, 93.1†
_NUMBER_CELL_RE = re.compile(r"(?:[-–—−]|[±+−-]?\d[\d.,/]*%?[*†‡]*)(?:\s+(?:[-–—−]|[±+−-]?\d[\d.,/]*%?[*†‡]*))*")


def _number_cell(text: str) -> bool:
    return _NUMBER_CELL_RE.fullmatch(text.strip()) is not None


def _number_cells(items: list[tuple]) -> set[int]:
    """Pieces of a table that hold only numbers: the left or right edge aligns with numbers on two other lines
    (the same column), or the piece sits on one row beside such a number.
    A number ending with a period (a year in references, 2015.) does not count."""
    numbers = [
        index for index, item in enumerate(items) if _number_cell(item[0]) and not item[0].rstrip().endswith(".")
    ]
    found: set[int] = set()
    for index in numbers:
        rect = items[index][1]
        for edge in ("x0", "x1"):
            mates = [
                other
                for other in numbers
                if other != index
                and min(rect.y1, items[other][1].y1) - max(rect.y0, items[other][1].y0) <= 0
                and abs(getattr(items[other][1], edge) - getattr(rect, edge)) <= 1
            ]
            if len(mates) >= 2:
                found.add(index)
                break
    # Numbers on the same row beside these (a - that appears on only two lines) are also cells of that row
    for index in numbers:
        rect = items[index][1]
        if index not in found and any(
            min(rect.y1, items[cell][1].y1) - max(rect.y0, items[cell][1].y0)
            > min(rect.height, items[cell][1].height) * 0.5
            for cell in found
        ):
            found.add(index)
    return found


def _row_labels(items: list[tuple], cells: Collection[int]) -> set[int]:
    """Row labels of a table: a number cell lies more than one em to the right on the same row."""
    found: set[int] = set()
    for index, (text, rect, spans, *_rest) in enumerate(items):
        if index in cells or not re.search(r"[^\W\d_]", text):
            continue
        size = max((float(span["size"]) for span in spans), default=10.0)
        for cell in cells:
            mate = items[cell][1]
            if (
                mate.x0 - rect.x1 >= size
                and min(rect.y1, mate.y1) - max(rect.y0, mate.y0) > min(rect.height, mate.height) * 0.5
            ):
                found.add(index)
                break
    return found


# Maximum number of words in one header cell. A justified body line with stretched word spacing can also be
# split into several pieces, each half a sentence
_HEADER_CELL_WORDS = 3


def _cut_header_cells(items: list[tuple], columns: Sequence[float]) -> tuple[list[tuple], set[int]]:
    """The cells of a header row (Fine-tune Fine-tune Feature-based) are close together and PyMuPDF merges
    them into one line, so the translation is squeezed to the left and does not line up with the columns
    below. Split the line at gaps of half an em or more; when each piece sits directly over a different
    number column below, each piece becomes a line of its own.
    The first piece may have no number below it (the header of the row label column).
    Returns the new lines and the indices of these cells. Body lines rarely split into several pieces, and
    pieces that each sit over a number column are rarer still."""
    if len(columns) < 2:
        return items, set()
    rows: list[list[int]] = []
    for index, item in enumerate(items):
        if rows:
            first = items[rows[-1][0]][1]
            if min(first.y1, item[1].y1) - max(first.y0, item[1].y0) >= min(first.height, item[1].height) * 0.5:
                rows[-1].append(index)
                continue
        rows.append([index])
    out: list[tuple] = []
    found: set[int] = set()
    for row in rows:
        spans = sorted((span for index in row for span in items[index][2]), key=lambda span: span["bbox"][0])
        pieces: list[list[dict]] = []
        for span in spans:
            if pieces and float(span["bbox"][0]) - float(pieces[-1][-1]["bbox"][2]) < float(span["size"]) * 0.5:
                pieces[-1].append(span)
            else:
                pieces.append([span])
        hits = []
        for piece in pieces:
            center = (float(piece[0]["bbox"][0]) + float(piece[-1]["bbox"][2])) / 2
            near = [
                k
                for k, column in enumerate(columns)
                if abs(column - center) <= max(2.0, float(piece[0]["size"]) * 0.25)
            ]
            hits.append(near[0] if near else None)
        # The first piece is the header of the row label column (System), with no numbers below it
        aligned = hits[1:] if hits and hits[0] is None else hits
        texts = [" ".join(str(span["text"]).strip() for span in piece) for piece in pieces]
        if (
            len(aligned) < 2
            or None in aligned
            or len(set(aligned)) < len(aligned)
            or not all(re.search(r"[^\W\d_]", text) and len(text.split()) <= _HEADER_CELL_WORDS for text in texts)
        ):
            out.extend(items[index] for index in row)
            continue
        raw = sorted((span for index in row for span in items[index][3]), key=lambda span: span["bbox"][0])
        top = min(items[index][1].y0 for index in row)
        bottom = max(items[index][1].y1 for index in row)
        for piece in pieces:
            x0, x1 = float(piece[0]["bbox"][0]), float(piece[-1]["bbox"][2])
            kept = [span for span in raw if x0 - 0.5 <= float(span["bbox"][0]) and float(span["bbox"][2]) <= x1 + 0.5]
            box = pymupdf.Rect()
            for span in piece:
                box |= _text_box(span)
            found.add(len(out))
            out.append(
                ("".join(str(span["text"]) for span in kept), pymupdf.Rect(box.x0, top, box.x1, bottom), piece, kept)
            )
    return out, found


def _drop_number_tails(items: list[tuple]) -> list[tuple]:
    """The row label is too long and PyMuPDF has merged the following number cells into the same line
    (BERT (Ens.+TriviaQA) 86.2 92.2 ...). When the rest of the line is all numbers and the left edge of its
    first cell aligns with a standalone number cell on another line, remove it from the line and leave it on
    the original page, not sent for translation, like the numbers on other lines. Only gaps of nearly half an em
    or more count: word spacing in a sentence is never that wide."""
    starts = [items[index][1].x0 for index in _number_cells(items)]
    if not starts:
        return items
    out: list[tuple] = []
    for item in items:
        text, rect, spans, raw = item[:4]
        cut = None
        for k in range(1, len(spans)):
            x = float(spans[k]["bbox"][0])
            size = float(spans[k]["size"])
            if (
                any(abs(x - start) <= 1 for start in starts)
                and x - float(spans[k - 1]["bbox"][2]) >= size * 0.5
                and _number_cell("".join(str(span["text"]) for span in spans[k:]))
                and re.search(r"[^\W\d_]", "".join(str(span["text"]) for span in spans[:k]))
            ):
                cut = x
                break
        if cut is None:
            out.append(item)
            continue
        kept = [span for span in raw if float(span["bbox"][0]) < cut - 0.5]
        while kept and not str(kept[-1]["text"]).strip():
            kept.pop()
        box = pymupdf.Rect()
        for span in kept:
            if str(span["text"]).strip():
                box |= _text_box(span)
        head = "".join(str(span["text"]) for span in kept)
        out.append(
            (
                head,
                pymupdf.Rect(box.x0, rect.y0, box.x1, rect.y1),
                [span for span in spans if float(span["bbox"][0]) < cut - 0.5],
                kept,
                *item[4:],
            )
        )
    return out


def _cut_leaders(items: list[tuple], toc_page: bool = False) -> tuple[list[tuple], dict[int, Leader]]:
    """Remove the leaders and page numbers of contents entries from their lines so that only the titles are
    sent for translation. Returns the new lines and {line index: leader}.
    With toc_page (the page has entries with leaders), a line without a leader whose page number sits to the
    right more than two ems away (a part title line) also counts, only no dots are added. On other pages such
    lines are values in tables and figures and are left alone.

    When the leader and page number are sent for translation with the title, the model rearranges the dots
    and page number along with the text and the page number is no longer right-aligned; an entry with a short
    title and many dots has too low a share of letters, the whole entry is taken as a symbol line and keeps
    the original text, and the same page ends up partly translated. Once removed, they stay on the original
    page unchanged.
    One visual line may be split into several pieces (the title in one, the leader and page number in
    another), so the search runs over whole lines."""
    rows: list[list[int]] = []
    for index, item in enumerate(items):
        if rows:
            first = items[rows[-1][0]][1]
            overlap = min(first.y1, item[1].y1) - max(first.y0, item[1].y0)
            if overlap >= min(first.height, item[1].height) * 0.5:
                rows[-1].append(index)
                continue
        rows.append([index])
    replaced: dict[int, tuple | None] = {}
    marked: dict[int, Leader] = {}
    for row in rows:
        pieces = [(index, span) for index in row for span in items[index][3]]
        text = "".join(str(span["text"]) for _, span in pieces)
        cut = _leader_start(pieces, text, toc_page)
        if cut is None or not re.search(r"[^\W\d_]", text[:cut]):
            continue
        kept: dict[int, list[dict]] = {index: [] for index in row}
        dropped: list[dict] = []
        shared: list[dict] = []
        offset = 0
        for index, span in pieces:
            body = str(span["text"])
            if offset + len(body) <= cut:
                kept[index].append(span)
            elif offset >= cut:
                dropped.append(span)
            else:
                left, right = _split_span(span, cut - offset)
                kept[index].append(left)
                dropped.append(right)
                # The boxes of the cut halves are estimated from character counts; erasing identifies the
                # characters by the whole original span
                shared.append(span)
            offset += len(body)
        leader = _leader_of(dropped, shared)
        if leader is None:
            continue
        last = None
        for index in row:
            spans = kept[index]
            if not any(str(span["text"]).strip() for span in spans):
                replaced[index] = None
                continue
            line = {"spans": spans, "bbox": items[index][1]}
            replaced[index] = (
                "".join(str(span["text"]) for span in spans),
                _line_box(line),
                [span for span in spans if str(span["text"]).strip()],
                spans,
            )
            last = index
        if last is not None:
            marked[last] = leader
    out: list[tuple] = []
    leaders: dict[int, Leader] = {}
    for index, item in enumerate(items):
        item = replaced.get(index, item)
        if item is None:
            continue
        if index in marked:
            leaders[len(out)] = marked[index]
        out.append(item)
    return out, leaders


def _leader_start(pieces: list[tuple[int, dict]], text: str, toc_page: bool) -> int | None:
    """Character position where the leader starts. On a contents page a leader may have only two or three
    dots; without a leader, check whether the last piece is a page number far from the rest."""
    found = _LEADER_RE.search(text)
    if found is not None:
        return found.start()
    if not toc_page:
        return None
    found = _SHORT_LEADER_RE.search(text)
    if found is not None:
        return found.start()
    visible = [(offset, span) for offset, span in _span_offsets(pieces) if str(span["text"]).strip()]
    if len(visible) < 2:
        return None
    (offset, last), (_, before) = visible[-1], visible[-2]
    size = float(last.get("size") or 10.0)
    gap = pymupdf.Rect(last["bbox"]).x0 - pymupdf.Rect(before["bbox"]).x1
    if _PAGE_NUMBER_RE.fullmatch(str(last["text"])) and gap > size * 2:
        return offset
    return None


def _span_offsets(pieces: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    offset = 0
    for _, span in pieces:
        out.append((offset, span))
        offset += len(str(span["text"]))
    return out


def _split_span(span: dict, at: int) -> tuple[dict, dict]:
    """Cut span in two at character at, dividing the box in proportion to character counts. Used when the
    leader and the title are in the same span."""
    body = str(span["text"])
    box = pymupdf.Rect(span["bbox"])
    middle = box.x0 + box.width * at / max(len(body), 1)
    left = {**span, "text": body[:at], "bbox": (box.x0, box.y0, middle, box.y1)}
    right = {**span, "text": body[at:], "bbox": (middle, box.y0, box.x1, box.y1)}
    return left, right


def _leader_of(spans: list[dict], shared: list[dict]) -> Leader | None:
    """spans are the leader and page number cut off; shared are the original spans that held the title as
    well and were cut."""
    visible = [span for span in spans if str(span["text"]).strip()]
    if not visible:
        return None
    is_dots = [bool(_LEADER_DOTS_RE.fullmatch(str(span["text"]))) for span in visible]
    numbers = [span for span, dot in zip(visible, is_dots) if not dot]
    dotted = [span for span, dot in zip(visible, is_dots) if dot]
    if not numbers:
        return None
    number = pymupdf.Rect()
    for span in numbers:
        number |= pymupdf.Rect(span["bbox"])
    first = numbers[0]
    size = float(first.get("size") or 10.0)
    xs = sorted(pymupdf.Rect(span["bbox"]).x0 for span in dotted if len(str(span["text"]).strip()) == 1)
    gaps = sorted(right - left for left, right in zip(xs, xs[1:]) if right - left > 0.5)
    if gaps:
        pitch, dot_x = gaps[len(gaps) // 2], xs[-1]
    elif dotted:
        # The dots are in one span: compute the dot pitch from the box width and the dot count; the rightmost
        # dot is at the end of the box
        run = max(dotted, key=lambda span: len(str(span["text"])))
        box = pymupdf.Rect(run["bbox"])
        count = max(1, sum(str(run["text"]).count(mark) for mark in ".·…"))
        pitch = box.width / count
        dot_x = box.x1 - pitch
    else:
        pitch, dot_x = 0.0, number.x0
    origin = first.get("origin") or (number.x0, number.y1)
    # The page number is often a blue link; the dots use the leader's own color
    color = int((dotted[0] if dotted else first).get("color") or 0)
    return Leader(
        number,
        dot_x,
        pitch,
        float(origin[1]),
        size,
        color,
        tuple(span for span in [*dotted, *shared] if "origin" in span),
    )


def _cut_after(group: list[int], ends: Collection[int]) -> list[list[int]]:
    parts: list[list[int]] = [[]]
    for index in group:
        parts[-1].append(index)
        if index in ends:
            parts.append([])
    return [part for part in parts if part]


def _line_host(
    line: dict,
    own: dict,
    blocks: list[dict],
    line_box=None,
    starts: Sequence[float] = (),
) -> tuple[dict, dict] | None:
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
    best: tuple[float, dict, dict] | None = None
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


def _column_font(line: dict, own: dict, size: float) -> str:
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


def _main_font(line: dict) -> str:
    spans = [span for span in line["spans"] if span["text"].strip()]
    return str(_by_chars(spans, lambda span: span.get("font") or "")) if spans else ""


def _font_box(line: dict) -> pymupdf.Rect:
    return pymupdf.Rect(line["bbox"])


def _past_column(block: dict, line: dict, x: float, size: float, line_box=None) -> bool:
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


def _larger_text(line: dict, host: dict) -> bool:
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


def _stack_host(line: dict, own: dict, blocks: list[dict]) -> tuple[dict, dict] | None:
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


def _italic_span(span: dict) -> bool:
    return bool(_ITALIC_FONT_RE.search(str(span.get("font") or "")) or int(span.get("flags") or 0) & 2)


def _numeral_span(span: dict) -> bool:
    return bool(_NUMERAL_RE.fullmatch(str(span.get("text") or "").strip()))


def _line_size(line: dict) -> float:
    return max((float(span["size"]) for span in line["spans"] if span["text"].strip()), default=0.0)


def _stacked_on(block: dict, line: dict, box: pymupdf.Rect, size: float, italic: bool) -> dict | None:
    """The math line in block directly above or below line and touching it. It must have math characters over
    line; covering it with the line's extent is not enough."""
    for other in block["lines"]:
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
                band = _small_band({"spans": other["spans"]}, box, small)
                if band is None or abs((box.x0 + box.x1) - (band.x0 + band.x1)) / 2 > max(
                    1.0, 0.15 * min(box.width, band.width)
                ):
                    continue
            return other
    return None


def _math_seam(left: list[dict], right: list[dict]) -> bool:
    """The seam is math: the left side ends with a formula, the right side starts with one, or one side is all
    symbols.
    Only the seam is checked: a line in the left column that starts with a formula and ends with body text
    does not join the right column."""
    left = [span for span in left if str(span.get("text") or "").strip()]
    right = [span for span in right if str(span.get("text") or "").strip()]
    if not left or not right:
        return False
    if any(NO_TEXT_RE.match("".join(str(span["text"]) for span in side)) for side in (left, right)):
        return True
    return _span_is_math(left[-1]) or _span_is_math(right[0])


def _same_band(a: dict, b: dict, box=None) -> bool:
    """Enough vertical overlap. Denominators of fractions and large braces pull the box down, so two pieces of
    the same line overlap by less than half.
    By default the comparison uses glyph height: the font box of a math font reaches into the next line and
    overlaps it, and a piece would move back and forth between the two lines."""
    box = box or _line_box
    return _band_overlap(box(a), box(b))


def _band_overlap(ra: pymupdf.Rect, rb: pymupdf.Rect) -> bool:
    overlap = min(ra.y1, rb.y1) - max(ra.y0, rb.y0)
    return overlap >= min(ra.height, rb.height) * 0.35


def _typewriter_line(spans: list[dict]) -> bool:
    """The whole line is monospaced Latin text. Chinese and Japanese fonts are also often marked monospaced and
    do not count."""
    visible = [span for span in spans if str(span.get("text") or "").strip()]
    if not visible or any(CJK_RE.search(str(span["text"])) for span in visible):
        return False
    return all(_mono_span(span) for span in visible)


def _row_style(spans: list[dict]) -> tuple[bool, bool]:
    """Style of a line: whether the whole line is bold, and whether it is all monospaced. Only body spans with
    letters are considered; superscript asterisks after names and braces in a math font around email
    addresses do not count."""
    words = [
        span for span in spans if any(ch.isalpha() for ch in str(span.get("text") or "")) and not _span_is_math(span)
    ]
    bold = bool(words) and all(
        emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)).bold for span in words
    )
    mono = (
        bool(words)
        and all(_mono_span(span) for span in words)
        and not any(CJK_RE.search(str(span.get("text") or "")) for span in words)
    )
    return bold, mono


def _row_boxes(rects: Sequence[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Merge pieces into visual lines by vertical overlap."""
    rows: list[pymupdf.Rect] = []
    for rect in rects:
        if rows:
            last = rows[-1]
            overlap = min(last.y1, rect.y1) - max(last.y0, rect.y0)
            if overlap > min(last.height, rect.height) * 0.5:
                rows[-1] = last | rect
                continue
        rows.append(pymupdf.Rect(rect))
    return rows


def _centered_rows(rows: Sequence[pymupdf.Rect], size: float) -> bool:
    """Several lines share one center line and differ in width: a centered stack (author names, affiliations,
    email addresses).
    Justified body lines share both left and right edges; with only two lines the widths may match by chance,
    so the whole stack is checked.
    Full body lines can also differ by a point or two on the left (a quote or hyphen at the start of a line
    sticks out), but their widths differ by less than a fifth of a character."""
    if len(rows) < 2:
        return False
    mids = [(row.x0 + row.x1) / 2 for row in rows]
    if max(mids) - min(mids) > size * 0.5:
        return False
    widths = [row.width for row in rows]
    return max(row.x0 for row in rows) - min(row.x0 for row in rows) >= 1 and max(widths) - min(widths) >= max(
        1.0, size * 0.2
    )


# A line that stops at one of these symbols has an unfinished expression
_OPERATOR_ENDS = ("(", "+", "*", "/", "=", "<", ">", "≤", "≥")


def _open_operator_end(text: str) -> bool:
    """The line stops at an operator or an opening parenthesis. An angle bracket needs a space before it, so a
    tag such as <Workflow> does not count; a run written together such as ===Instructions=== is a separator,
    and a lone asterisk after a period is a footnote mark; neither counts."""
    text = text.rstrip()
    if not text.endswith(_OPERATOR_ENDS) or text[-2:-1] == text[-1]:
        return False
    if text[:-1].rstrip()[-1:] in tuple(".!?:;,"):
        return False
    return text[-1] not in "<>" or text[-2:-1].isspace()


_PAREN_ENUM_RE = re.compile(r"\(([a-z]|\d{1,2})\)\s")


def _follows_inline_enum(text: str, before: list[str]) -> bool:
    """text starts with (b), and (a) has appeared in the middle of one of the lines before."""
    match = _PAREN_ENUM_RE.match(text)
    if not match:
        return False
    label = match.group(1)
    if label in ("a", "1", "0"):
        return False
    prev = chr(ord(label) - 1) if label.isalpha() else str(int(label) - 1)
    return any(f"({prev})" in line.strip()[1:] for line in before)


# A next line whose size is this many times that of the paragraph starts another paragraph. A line needs at
# least this many characters for its size to count as the paragraph's size
_SIZE_RISE = 1.15


_SIZE_ROW_CHARS = 6


def _row_left(items: list[tuple], index: int) -> float:
    """Left edge of the line that piece index is on: the leftmost of the pieces overlapping it vertically by
    more than half."""
    box = items[index][1]
    return min(
        other[1].x0
        for other in items
        if min(other[1].y1, box.y1) - max(other[1].y0, box.y0) > min(other[1].height, box.height) * 0.5
    )


def _split_lines(items: list[tuple], whole_rows: bool = False) -> list[list[int]]:
    """Split the lines of one text block into separate paragraphs (returns the line indices of each group).

    On slides PyMuPDF often puts several list items, or even unrelated caption labels, into one block;
    translating the block as one paragraph turns bullets into inline text and joins unrelated content.
    Three places to split:
    - a line that starts with a list marker;
    - a pure symbol line ("=", "→" and the like) becomes its own paragraph, which the filter later skips and
      keeps as it is;
    - a jump to the left of more than about 3 font sizes, where the two lines do not both reach the right
      margin (labels side by side). 1-2 em is a first-line indent and is never split, whether or not the line
      above is full.
    - the line above clearly stops short of the right margin (a hard line break, not automatic wrapping) and
      the next line looks like the start of a new sentence; it is not a hard break when the line above ends
      with a comma or other joining punctuation, when the jump to the left is only an indent, or when the line
      continues the same list item.
    Continuation lines of automatic wrapping (lowercase start, full line above) still join the paragraph
    above.
    Body text under a heading line that is clearly smaller and is a complete sentence becomes its own
    paragraph; a line that is clearly larger also becomes its own paragraph.
    Short lines split off by superscripts have too few characters and are not split. For whole_rows see
    _justified_joins.
    The jump to the left and the hard break are measured against "the last line at body size" (the reference
    line): small line dicts split off by superscripts and subscripts would throw off the jump and chop the
    paragraph up at formulas.
    Elements of items are tuples (line text, line rect, line spans, ...); only the first three are used.
    """
    if not items:
        return []
    max_x1 = max(it[1].x1 for it in items)
    justified = _justified_joins(items, max_x1, whole_rows)
    # A split line is compared as a whole line for the jump to the left and the right edge, not by its last
    # word
    row_box: dict[int, pymupdf.Rect] = {}
    for index in sorted(justified):
        head = index - 1
        while head in justified:
            head -= 1
        box = row_box.get(head, pymupdf.Rect(items[head][1])) | items[index][1]
        for member in range(head, index + 1):
            row_box[member] = box
    # Number cells and row labels of a table each become their own paragraph: with short row labels and
    # numbers close together, they would join the row labels above and below and the numbers on the same row
    # into one sentence
    cells = _number_cells(items)
    row_labels = _row_labels(items, cells)
    # When the whole block is a centered stack of short lines, a next line in another style is another item
    first_size = max((s["size"] for s in items[0][2]), default=10)
    stacked = _centered_rows(_row_boxes([item[1] for item in items]), first_size)
    groups, cur = [], [0]
    row_rect = pymupdf.Rect(items[0][1])  # The part of the current visual line joined so far
    row_spans = list(items[0][2])
    # Left edge of this paragraph's first line, and the number of lines so far. A line is often split into
    # several pieces, so the comparison is by line, not by piece
    group_x0, group_rows = row_rect.x0, 1
    ref = 0  # Reference line: the last line at body size. Small line dicts split off by scripts cannot serve,
    # or the jump to the left is measured against a small line and the paragraph is chopped up at formulas
    # ("(like a | sequence)")
    ref_size = max((s["size"] for s in items[0][2]), default=10)
    group_size = ref_size if len("".join(items[0][0].split())) >= _SIZE_ROW_CHARS else 0
    # When a full line leaves its sentence unfinished, a number wrapped to the start of the next line
    # (range from 0.01 to | 0.1.) is the end of the sentence, not a symbol line.
    # Aligning with the left edge of the line above also counts: with a first-line indent, the left edge only
    # starts from the second line (13 tokens: 11865, ... | 885, 220, ...).
    # A line above that stops at an abbreviation inside a sentence (Tanikawa et al. | 2012).) is also
    # unfinished
    number_tails: set[int] = set()
    for i in range(1, len(items)):
        text, rect = items[i][0], items[i][1]
        ptext, prect, pspans = items[ref][0], row_box.get(ref, items[ref][1]), items[ref][2]
        psize = max((s["size"] for s in pspans), default=10)
        stripped = text.lstrip()
        # A pure symbol line is its own paragraph; what follows is not joined to it
        prev_symbol = bool(NO_TEXT_RE.match(ptext.strip())) and ref not in number_tails
        if (
            NO_TEXT_RE.match(stripped)
            and re.search(r"\d", stripped)
            and not prev_symbol
            and max_x1 - prect.x1 <= 0.5 * psize
            and (not _ends_sentence(ptext) or _MID_ABBREV_RE.search(ptext.rstrip()) is not None)
            and min(abs(rect.x0 - group_x0), abs(rect.x0 - prect.x0)) <= 0.5 * psize
            and rect.y0 >= prect.y1 - 0.5 * psize
            and (i + 1 == len(items) or items[i + 1][1].y0 >= rect.y1 - 0.5 * psize)
        ):
            # The number has a line to itself. One followed by more text on the same line is a section number
            # at the start of the line (9.6 Summary)
            number_tails.add(i)
        # An index entry stops at a comma and its page numbers wrap to the next line: the page numbers go with
        # the entry (see _index_entry_ends) and are not split off however deep the indent
        index_tail = (
            whole_rows
            and ptext.rstrip().endswith(",")
            and bool(_INDEX_REFS_ONLY_RE.match(stripped))
            and -0.3 * psize <= rect.y0 - prect.y1 <= 0.5 * psize
        )
        if index_tail:
            number_tails.add(i)
        jump = prect.x0 - rect.x0
        # A 1-2 em jump to the left is a first-line indent, not a new paragraph. A first-line indent covers
        # only one line: when the paragraph already has two or more lines and the line above is not the first,
        # starting further left means the hanging continuation lines of a list item or example have ended.
        # When a block starts with a continuation line, its first line is also indented (datasheet entries).
        # Only lines of sentence text count; pieces split from formulas are not lines
        hanging_end = (
            group_rows >= 2
            and rect.x0 < row_rect.x0 - 0.5 * psize
            and _prose_units(stripped) >= 3
            and _prose_units(items[i - 1][0]) >= 2
        )
        # The first line of a first-line indent runs to the right edge. A first line in the block that is not
        # full, ends with a period and is followed by the start of a new sentence is the last line of the
        # paragraph before (the hanging continuation of an example)
        closed = (
            group_rows == 1
            and max_x1 - prect.x1 > 3 * psize
            and ptext.rstrip().endswith((".", "?", "!"))
            and _ends_sentence(ptext)
            and bool(SEG_START_RE.match(stripped))
        )
        # A first line that starts indented and does not reach the right edge, with the next line back on the
        # left: a line set off on its own (the example The green witch arrived), with body text continuing on
        # the next line. The indented first line of a paragraph runs to the right edge; in a paragraph set
        # around a figure all lines end at the same x, and the first line is no shorter than the next
        displayed = (
            group_rows == 1
            and row_rect.x0 - rect.x0 > 0.5 * psize
            and max_x1 - row_rect.x1 > 3 * psize
            and row_rect.x1 < rect.x1 - psize
            and _prose_units(items[i - 1][0]) >= 2
        )
        indent = 0 < jump <= 3 * psize and not hanging_end and not closed and not displayed
        # Continuation line of a list item: hanging alignment indented past the bullet. A line at the same x0
        # or jumping left is no continuation and may be the next paragraph
        continuing_list = _list_item(ptext) and not _list_item(text) and rect.x0 > prect.x0
        # At the start of a CJK line a new sentence cannot be told from a continuation, so a new paragraph
        # needs the line above to end with sentence-final punctuation
        seg_start = bool(SEG_START_RE.match(stripped)) or (bool(CJK_START_RE.match(stripped)) and _ends_sentence(ptext))
        # Text in a typewriter font is verbatim text line by line (dialogue, examples) with a ragged right
        # edge, so stopping short of the right margin does not mean a hard break.
        # A next line that looks like the start of a new sentence is a new line
        verbatim = _typewriter_line(pspans) and _typewriter_line(items[i][2])
        # A paragraph set around a figure: the lines beside the figure end at the same x, far from the right
        # edge of the block. When the line above ends at the same x as the line right above or below it, the
        # other line reaches at least that x, and the next line is not indented, the line fills the narrow
        # column beside the figure and is not a hard break.
        # Above the first line beside the figure is a full-width line, and below the last line the text
        # returns to full width. Short lines several lines apart that end at the same x by chance (answers to
        # exercises) do not count
        head = ref
        while head in justified:
            head -= 1
        above = row_box.get(head - 1, items[head - 1][1]) if head > 0 else None
        below = row_box.get(i, rect)
        lane_full = (
            above is not None
            and above.y0 < prect.y0 - 0.5 * psize
            and rect.x0 <= prect.x0 + 0.5 * psize
            and (
                abs(above.x1 - prect.x1) <= 0.5
                and below.x1 >= prect.x1 - 0.5
                or abs(below.x1 - prect.x1) <= 0.5
                and above.x1 >= prect.x1 - 0.5
            )
        )
        hard_break = (
            (max_x1 - prect.x1 > 3 * psize and not lane_full or verbatim)
            and seg_start
            and not prev_symbol
            and not indent
            and not continuing_list
            and not ptext.rstrip().endswith((",", "，", "、", ";", "；", "(", "+", "*", "/", "=", "<", ">"))
        )
        both_full = max_x1 - prect.x1 <= 3 * psize and max_x1 - rect.x1 <= 3 * psize
        # A wrapped line of a list item is indented right of the item number and lies left of the first line's
        # body text: when a wide gap follows the item number, the body text is a separate piece
        hanging_wrap = _list_item(items[cur[0]][0]) and group_x0 < rect.x0 <= group_x0 + 3 * psize
        # The caption label (Figure 4.12) is a piece of its own, separated from the following text by a gap.
        # When the first line is full and wraps to the next line back at the label's left edge, it is a
        # wrapped caption line, not a jump to the left. Measured from the body piece, the jump is several ems.
        # From the lines of the block alone, a column of full short lines (whose right edge is the block's
        # right edge) cannot be told apart from it, so the first piece must be a caption label
        wrap_back = (
            group_rows == 1
            and bool(_CAPTION_SPAN_RE.match(items[cur[0]][0].strip()))
            and abs(rect.x0 - group_x0) <= 0.5 * psize
            and max_x1 - prect.x1 <= 0.5 * psize
            and rect.y0 >= prect.y1 - 0.5 * psize
        )
        left_jump = jump > 3 * psize and not both_full and not hanging_wrap and not wrap_back
        # The two lines overlap vertically by more than half and their x ranges intersect: one visual line
        # split by superscripts, subscripts or a formula (the 1 and T of y_1^T each count as a line, with
        # interlocking x ranges), so no paragraph split here.
        # Labels side by side are apart horizontally (no horizontal overlap), and left_jump splits them as usual.
        qrect = items[i - 1][1]
        overlap = min(qrect.y1, rect.y1) - max(qrect.y0, rect.y0)
        h_overlap = min(qrect.x1, rect.x1) - max(qrect.x0, rect.x0)
        # The next word on the same baseline, within word spacing. When a line is split by word, a capitalized
        # word inside the line (DXL) is not a new paragraph.
        # Sentence ends, markup and labels inside figures are still split, so that side labels and cells are
        # not merged into body text.
        # Word spacing and the seam gap are measured with the size of the previous piece. When the previous
        # piece is more than twice as large as this one (a large chapter number one large em from the title,
        # with a vertical rule between), this piece's size is used. Small scripts in fractions still use the
        # body size
        piece_size = max((s["size"] for s in items[i][2]), default=psize)
        seam_gap = piece_size if psize > piece_size * 1.6 else psize
        same_word = _same_line_word(items[i - 1][0], text, qrect, rect, seam_gap) or i in justified
        # When an expression is split into several pieces (fraction, large brace, radical), pieces meeting end
        # to start are one visual line, and a pure symbol piece must not cut the line. Vertical overlap is
        # measured against the part of the line joined so far:
        # the boxes of a brace and of a denominator on their own barely touch. When both pieces are body
        # text, the sentence-end rule above still splits them.
        row_overlap = min(row_rect.y1, rect.y1) - max(row_rect.y0, rect.y0)
        # A line with an operator reaching the right edge is full and stretched by justification; the gaps
        # around the equals sign are stretched more than word spacing
        operator = bool(_OPERATOR_RE.fullmatch(stripped.rstrip()))
        seam_reach = seam_gap * 1.5 if operator and rect.x1 >= max_x1 - 1 else seam_gap
        touching = (
            _math_seam(items[i - 1][2], items[i][2])
            and row_overlap > min(row_rect.height, rect.height) * 0.35
            and -seam_gap <= rect.x0 - row_rect.x1 <= seam_reach
        )
        same_visual_line = (
            (
                h_overlap > 0
                and overlap > min(qrect.y1 - qrect.y0, rect.y1 - rect.y0) * 0.5
                and not _stacked_rows(items[i - 1][2], items[i][2])
            )
            or same_word
            or touching
        )
        # More than three ems apart on the same row, and not word spacing stretched by justification: labels
        # side by side (a caption and nodes in a figure).
        # A math seam does not count; pieces of a fraction may sit in another block and leave a gap.
        side_by_side = (
            not same_visual_line
            and i not in justified
            and not _math_seam(items[i - 1][2], items[i][2])
            and overlap > min(qrect.y1 - qrect.y0, rect.y1 - rect.y0) * 0.5
            and rect.x0 - qrect.x1 > 3 * psize
        )
        # A section number and its title form one line ('2.2 Training'): a line above that is only digits is
        # not a symbol line and is not split off as a hard break, or the number is filtered out and not masked,
        # the title is translated alone, and the baselines do not line up
        # A chapter number more than twice the size of the title is separate decoration; the title is written
        # in its original place and does not move with the number
        line_size = max((s["size"] for s in items[i][2]), default=10)
        sec_label = (
            bool(re.fullmatch(r"\d+(?:\.\d+)*\.?", ptext.strip()))
            and overlap > min(prect.y1 - prect.y0, rect.y1 - rect.y0) * 0.5
            and psize <= line_size * 1.6
        )
        # A 9pt lead-in under a 12pt heading differs on the left only by the section number and looks like an
        # indent. The drop in size is what makes it a new paragraph.
        # Short lines split off by scripts overlap the line above vertically; two words on a line of their own
        # also count (the affiliation under author names),
        # compared with the size of most characters on the line above, so a large asterisk at its start does
        # not count
        own_row = row_overlap <= min(row_rect.height, rect.height) * 0.2
        size_drop = not same_visual_line and (
            (line_size < ref_size * 0.85 and _prose_units(stripped) >= 4)
            or (own_row and line_size < _text_size(items[i - 1][2]) * 0.85 and _prose_units(stripped) >= 2)
        )
        # A line clearly larger than every line of the paragraph so far is its own paragraph: a subheading
        # under an expression on a slide sits slightly left of the indented expression and would be merged as
        # a first-line indent. Small-caps words (KNOWREF, GOVERN 1:) followed by body size do not count;
        # small lines split off by scripts and fragments of a few characters (a block starting with a
        # superscript u) do not count toward the paragraph's size. A lowercase start is a sentence continuing
        size_rise = (
            not same_visual_line
            and own_row
            and seg_start
            and 0 < group_size * _SIZE_RISE < line_size
            and _prose_units(stripped) >= 2
        )
        body_line = line_size >= ref_size * 0.79 and len("".join(stripped.split())) >= _SIZE_ROW_CHARS
        # A line after verbatim monospaced lines that switches back to the body font and looks like the start
        # of a new sentence: the verbatim text ends here and the body text starts a new paragraph.
        # The first-line indent of the body differs only slightly from the left edge of the monospaced lines,
        # and would be merged as an indent.
        # A line still starting in monospace is the wrapped tail of a URL or code from the line above
        # (...BaBao | 1). This ...) and has not switched back to body text
        leaves_verbatim = (
            own_row
            and seg_start
            and _typewriter_line(pspans)
            and not _typewriter_line(items[i][2])
            and not _typewriter_line(items[i][2][:1])
            and _prose_units(stripped) >= 4
        )
        # The author block has one item per line: names, affiliations and email addresses are centered, and a
        # change of font starts the next item.
        # A line may be split into several pieces (several author names side by side), so whole lines are
        # compared
        restyled = stacked and not same_visual_line and _row_style(row_spans) != _row_style(items[i][2])
        # When the full line above has no sentence end, "Q. " on the next line is a wrapped line, not a list
        # item like "a. ".
        # A dash at the start of a line can also be inside a sentence (... 2024 (URL) | - a performance that
        # ...): same size as the first line of the paragraph (the line above may be a monospaced URL one size
        # larger), not indented, and the paragraph is not itself an item starting with a dash.
        # An indented dash in another size is a lower-level bullet; one following an item is the next item at
        # the same level
        dash_wrap = (
            bool(_DASH_BULLET_RE.match(stripped))
            and abs(_text_size(items[i][2]) - _text_size(items[cur[0]][2])) <= psize * 0.05
            and rect.x0 <= prect.x0 + 0.5 * psize
            and not _DASH_BULLET_RE.match(items[cur[0]][0].lstrip())
        )
        wrapped_letter = (
            (bool(_LETTER_BULLET_RE.match(stripped)) or dash_wrap)
            and max_x1 - prect.x1 <= 3 * psize
            and not ptext.rstrip().endswith(_LIST_INTRO_END)
        )
        # An inline enumeration (a) ... (b) ...: the previous item is mid-line, this one happens to fall at the
        # start of a line, and the line above is full: a wrapped line, not a list
        inline_enum = max_x1 - prect.x1 <= 3 * psize and _follows_inline_enum(stripped, [items[j][0] for j in cur])
        # Before a proof case (iii) Let ..., the line above stops at an expression or a sentence end and is not
        # full. A full line above means an inline enumeration happened to wrap to the start of the line
        roman_wrap = bool(_ROMAN_ITEM_RE.match(stripped)) and max_x1 - row_rect.x1 <= 3 * psize
        bullet = _list_item(text) and not wrapped_letter and not inline_enum and not roman_wrap
        # A line starting with a closing bracket is bracketed content wrapped from the line above
        # ((* + ? . [ | ]) need to...), and the [ at the end of the line is not a symbol line
        closing = stripped[:1] in ")]}" and _prose_units(stripped) >= 2
        # The line above stops at an operator or opening parenthesis with the expression unfinished, wrapping
        # to the start of the next line: the 100) on the next line is the second half of the expression, not a
        # number label and not a symbol line. In a display expression the numerator starts right of the equals
        # sign, which is not a wrap
        open_end = (
            not prev_symbol and _open_operator_end(ptext) and not same_visual_line and rect.x0 <= row_rect.x0 + psize
        )
        # Label and value set in two columns (View ranges:   {l_partkey, ...} ∈ (150, +∞), | {o_custkey} ...):
        # the value is more than one em from the label, and the value's next line starts at the value's left
        # edge instead of returning under the label. The value is another column and its own paragraph
        next_row = items[i + 1][1] if i + 1 < len(items) else None
        # A hanging indent after an item number (2.1, a.) is a list item and is not split here; the label must
        # be a word
        value_column = (
            overlap > min(qrect.height, rect.height) * 0.5
            and not touching
            and bool(re.search(r"[^\W\d_]{2,}", items[i - 1][0]))
            and not _list_item(items[i - 1][0])
            and rect.x0 - qrect.x1 >= psize
            and rect.x0 > group_x0 + psize
            and next_row is not None
            and next_row.y0 >= rect.y0 + rect.height * 0.5
            and abs(next_row.x0 - rect.x0) <= 1.0
        )
        # The previous sentence has ended and the next line starts with an example number: one example per line
        example = bool(_EXAMPLE_NUMBER_RE.match(stripped)) and _ends_sentence(ptext)
        # The previous sentence (or a bracketed example word at its end) has ended and the next line starts
        # with a bold label: the next entry of a definition list.
        # Single-line entries are nearly full, so the hard break rule does not catch them
        # A line of its own, indented more than three ems on both sides, only in math fonts and with a
        # relation symbol, is a display expression, and is not joined even when the line above stops at a
        # comma. Summation signs and denominators split from a fraction have no relation symbol and do not
        # count
        display_math = (
            own_row
            and rect.x0 - group_x0 > 3 * psize
            and max_x1 - rect.x1 > 3 * psize
            and all(_span_is_math(span) for span in items[i][2] if str(span["text"]).strip())
            and bool(_RELATION_RE.search(stripped))
        )
        # A new paragraph below an indented line set off on its own (the translation of an example): this line
        # is left of the line above and the next line is one or two ems further left, so this line is the
        # first-line indent of the next paragraph, not a continuation of the line above. A justified line split
        # into several pieces is compared by the left edge of the whole line.
        # Three lines on the same center line are centered (author affiliations), wider further down, and not
        # indented
        above = _row_left(items, i - 1)
        below = _row_left(items, i + 1) if i + 1 < len(items) and items[i + 1][1].y0 >= rect.y1 - 0.5 * psize else None
        centered_rows = below is not None and all(
            abs((other.x0 + other.x1) - (rect.x0 + rect.x1)) / 2 <= 0.15 * psize for other in (prect, items[i + 1][1])
        )
        opens_indent = (
            own_row
            and seg_start
            and below is not None
            and not centered_rows
            and above - rect.x0 > 0.5 * psize
            and 0.5 * psize < rect.x0 - below <= 3 * psize
            and not ptext.rstrip().endswith((",", "，", "、", ";", "；", ":", "："))
        )
        # The line above ends a sentence (even when full), this line is indented one or two ems and runs to the
        # right edge, and the next line returns to the left edge of the line above: the first-line indent of a
        # new paragraph. Between numbered entries it is a hanging continuation line
        # (46. Fuller ... | Monthly Notices ... | 47. ...)
        indent_after_end = (
            own_row
            and seg_start
            and below is not None
            and not centered_rows
            and _ends_sentence(ptext)
            and not _list_item(items[i - 1][0])
            and not _list_item(items[i + 1][0])
            and 0.5 * psize < rect.x0 - above <= 3 * psize
            and abs(below - above) <= 0.5 * psize
            and rect.x1 >= max_x1 - 2 * psize
        )
        # A numbered theorem or lemma label (Lemma 10.3.) always starts a new paragraph, even when the line
        # above stops at an expression.
        # When this paragraph also starts with a bold label and the line above stops short of the right edge,
        # it is the next entry of a definition list whose entries are partial example sentences without
        # sentence-final punctuation (mouse² : ... a quiet animal like a mouse | bank¹ : ...)
        labeled = (
            not same_visual_line
            and _label_lead(items[i][2])
            and (
                _ends_sentence(ptext)
                or ptext.rstrip().endswith(")")
                or bool(_NUMBERED_LABEL_RE.match(stripped))
                or (max_x1 - prect.x1 > 3 * psize and _label_lead(items[cur[0]][2]))
            )
        )
        if not index_tail and (
            value_column
            or i in cells
            or i in row_labels
            or (
                not same_visual_line
                and not sec_label
                and (
                    (not open_end and (bullet or example or (NO_TEXT_RE.match(stripped) and i not in number_tails)))
                    or (prev_symbol and not closing)
                    or hard_break
                    or left_jump
                    or size_drop
                    or size_rise
                    or side_by_side
                    or leaves_verbatim
                    or restyled
                    or labeled
                    or display_math
                    or opens_indent
                    or indent_after_end
                    or displayed
                    or _caption_opens(items[i][2], pspans)
                )
            )
        ):
            groups.append(cur)
            cur = [i]
            group_x0, group_rows, group_size = rect.x0, 1, line_size if body_line else 0
        else:
            cur.append(i)
            if body_line:
                group_size = max(group_size, line_size)
            size_drop = False
            if not same_visual_line:
                group_rows += 1
            elif group_rows == 1:
                group_x0 = min(group_x0, rect.x0)
        row_rect = row_rect | rect if same_visual_line else pymupdf.Rect(rect)
        row_spans = row_spans + items[i][2] if same_visual_line else list(items[i][2])
        isize = max((s["size"] for s in items[i][2]), default=10)
        # For a new paragraph split off by a drop in size, this line becomes the reference. Otherwise the
        # following continuation lines of the same size are still split against the heading.
        # Short superscript lines do not update it here, so the jump to the left is not thrown off by a small
        # line. An equals sign joined at the end of a line is not used as the reference line either
        if size_drop or isize >= ref_size * 0.79 and not (same_visual_line and operator):
            ref, ref_size = i, isize
    groups.append(cur)
    return groups


def _caption_opens(spans: list[dict], above: list[dict]) -> bool:
    """The line starts with a caption label set apart (Figure 21.17 in bold or in another font): the first line
    of a caption.
    The caption of a wide figure sits further out on the left than the text in the figure, which looks like
    an indent, so the jump-left rule cannot split it. Figure 3.8 shows in body text does not count."""
    shown = [span for span in spans if str(span.get("text") or "").strip()]
    prior = [span for span in above if str(span.get("text") or "").strip()]
    if not shown or not prior or not _CAPTION_SPAN_RE.match(shown[0]["text"].strip()):
        return False
    first, main = shown[0], max(prior, key=lambda span: len(span["text"]))
    return first.get("font") != main.get("font") or bool(first.get("flags", 0) & 16) != bool(main.get("flags", 0) & 16)


# A caption label that is a span of its own, with a section-style number
_CAPTION_SPAN_RE = re.compile(r"^(?:Figure|Fig\.?|Table|Tab\.?)\s*\d+(?:[.\-]\d+)*\s*[:.]?$", re.I)


# Expressions not set in a math font (older papers use Times-Italic for variables and Symbol for operators):
# variables in body italics, brackets, operators.
# A variable is a single letter or two capital letters (PE). Two-letter words such as If and op count only
# together with brackets or operators (op c)); theorem bodies are italic throughout, and If or In next to a
# formula is still body text
_ITALIC_FONT_RE = re.compile(r"Italic|Oblique", re.I)


_NUMERAL_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _small_band(run: dict, over: pymupdf.Rect, small: float) -> pymupdf.Rect | None:
    """The whole band of small text in run that falls horizontally below or above over (a denominator or a
    numerator)."""
    spans = [span for span in run["spans"] if str(span.get("text") or "").strip() and float(span["size"]) < small]
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


def _text_size(spans: list[dict]) -> float:
    """The font size with the most characters in a line."""
    count: dict[float, int] = {}
    for span in spans:
        count[span["size"]] = count.get(span["size"], 0) + len(span.get("text", "").strip())
    return max(count, key=count.get, default=10)


# Relation symbols of display formulas. Plus, minus and times signs must be in a math font: in a body font
# they are often in headings such as Phoenix + Calcite.
# Arrows do not count; slides often use arrows to link a few words
_RELATION_RE = re.compile(r"[=≤≥≈≡≠∝∈∉⊂⊆⊇≜]|(?:^|\s)[<>](?=\s|$)")


def _cut_at_posts(group: list[int], items: list[tuple], posts: Sequence[pymupdf.Rect]) -> list[list[int]]:
    """Two pieces on the same row with a vertical rule between them are not joined into one sentence: a large
    chapter number at the start of a chapter sits close to the title, with a vertical rule between."""
    parts = [group[:1]]
    for prev, cur in zip(group, group[1:]):
        left, right = items[prev][1], items[cur][1]
        rows = min(left.height, right.height) * 0.5
        between = any(
            left.x1 - 1 <= post.x0
            and post.x1 <= right.x0 + 1
            and min(left.y1, right.y1, post.y1) - max(left.y0, right.y0, post.y0) > rows
            for post in posts
        )
        if between:
            parts.append([cur])
        else:
            parts[-1].append(cur)
    return [part for part in parts if part]


def _cut_runs(group: list[int], marked: set[int]) -> list[list[int]]:
    """Cut into consecutive runs by whether each index is in marked."""
    runs: list[list[int]] = []
    for index in group:
        if runs and (runs[-1][-1] in marked) == (index in marked):
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _voted_right(x0: float, size: float, rows: Sequence[pymupdf.Rect]) -> float | None:
    """When 40% or more of the lines on this page starting at x0 end at the same x, that x is the right edge.
    Returns None when there is none"""
    ends = Counter(round(row.x1) for row in rows if abs(row.x0 - x0) <= size)
    total = sum(ends.values())
    if ends:
        edge = max(ends, key=lambda value: (_near_votes(ends, value), value))
        votes = _near_votes(ends, edge)
        if votes >= 3 and votes >= total * 0.4:
            return float(edge)
    return None


def _near_votes(counter: Counter[int], value: int, near: int = 1) -> int:
    return sum(count for other, count in counter.items() if abs(other - value) <= near)


def _vivid(color: int) -> bool:
    """A vivid color: the three RGB components are far apart. Black and grey do not count"""
    rgb = [(color >> shift) & 0xFF for shift in (16, 8, 0)]
    return max(rgb) - min(rgb) >= 0x60


def _vivid_share(spans: Sequence[dict]) -> float:
    """Share of letters in vivid colors, 0 when there are no letters"""
    letters = sum(char.isalpha() for span in spans for char in str(span.get("text") or ""))
    vivid = sum(
        char.isalpha() for span in spans if _vivid(int(span.get("color") or 0)) for char in str(span.get("text") or "")
    )
    return vivid / letters if letters else 0.0


def _mono_span(span: dict) -> bool:
    """Monospaced font: recognized by font name, and also by the monospace flag in the PDF (fonts such as txtt
    whose names lack Mono)."""
    return bool(MONO_FONT_RE.search(str(span.get("font") or "")) or int(span.get("flags") or 0) & 8)


# A list with one sentence per line: at least this many lines
_LIST_RUN = 3


def _breaks_early(
    rows: Sequence[tuple[pymupdf.Rect, str, float]],
    right_of: Callable[[pymupdf.Rect, float], float],
    at_least: int = 1,
) -> bool:
    """Over rows of (box, text, size), at more than half of the line breaks, and at least at_least of them, the
    first word of the next line would still fit after the line above.
    right_of gives the right edge a line can reach."""
    early = 0
    for (upper, _text, size), (_rect, text, _size) in zip(rows, rows[1:]):
        right = right_of(upper, size)
        word = text.split()[0] if text.split() else ""
        if upper.x1 + (len(word) * 0.5 + 1.0) * size <= right:
            early += 1
    return early >= at_least and early * 2 >= len(rows) - 1


def _cut_line_list(group: list[int], items: list[tuple], page_rows: Sequence[pymupdf.Rect]) -> list[list[int]]:
    """PyMuPDF often puts a list with one sentence per line (examples, queries) into one block. The test is
    the same as for lists across blocks (see _line_lists): several lines, one per row, aligned on the left,
    same size, equal line spacing, and at more than half of the line breaks the first word of the next line
    would still fit after the line above; such a block is split by line.
    The right edge is taken from the longest line of this block: a column width estimated from other lines
    comes out too wide for paragraphs at figure edges, inside tables and in two-column layouts;
    measured against the block's longest line, an automatically wrapped paragraph never breaks early. Lines set
    around a figure end at the figure's edge: when more than one line on the page starts at the same left
    edge and ends at the same x, the line counts as full.
    A row split into several pieces (a table's row label and the cells after it) is merged into one line
    before comparing. When every line starts another column at the same x (hen | +female), or every line
    starts with the same word (x_i contains ..., x_i's word shape), it is a table or list and is also split by
    line."""
    if len(group) < _LIST_RUN:
        return [group]
    rows: list[list[int]] = []
    for i in group:
        rect = items[i][1]
        if rows:
            top = pymupdf.Rect(items[rows[-1][0]][1])
            for j in rows[-1][1:]:
                top |= items[j][1]
            if top.y0 < (rect.y0 + rect.y1) / 2 < top.y1:
                rows[-1].append(i)
                continue
        rows.append([i])
    if len(rows) < _LIST_RUN:
        return [group]
    rects = []
    for row in rows:
        rect = pymupdf.Rect(items[row[0]][1])
        for j in row[1:]:
            rect |= items[j][1]
        rects.append(rect)
    sizes = [max(float(span["size"]) for i in row for span in items[i][2]) for row in rows]
    size = sizes[0]
    pitches = [lower.y0 - upper.y0 for upper, lower in zip(rects, rects[1:])]
    aligned = (
        all(abs(rect.x0 - rects[0].x0) <= size * 0.5 for rect in rects)
        and all(abs(other - size) <= size * 0.05 for other in sizes)
        and all(upper.y1 - lower.y0 <= size * 0.3 for upper, lower in zip(rects, rects[1:]))
        and all(0 < pitch <= size * 2.0 and abs(pitch - pitches[0]) <= size * 0.2 for pitch in pitches)
    )
    if not aligned:
        return [group]
    widest = max(rect.x1 for rect in rects)

    def full(rect: pymupdf.Rect) -> bool:
        return sum(1 for row in page_rows if abs(row.x1 - rect.x1) <= 1 and abs(row.x0 - rect.x0) <= size) >= 2

    texts = [" ".join(items[i][0].strip() for i in row) for row in rows]
    # A block with only one early line break is usually the short last line of a paragraph followed by the next
    # paragraph or the next subheading
    lines = [(rect, text, size) for rect, text in zip(rects, texts)]
    early = _breaks_early(lines, lambda rect, _size: rect.x1 if full(rect) else widest, at_least=2)
    if not (early or _column_rows(rows, items, size) or _same_opening(texts)):
        return [group]
    # After a full line comes a wrap inside a sentence, so the lines stay together; every other line break is
    # split
    parts = [list(rows[0])]
    for rect, row in zip(rects, rows[1:]):
        if full(rect):
            parts[-1].extend(row)
        else:
            parts.append(list(row))
    return parts


def _cut_item_breaks(group: list[int], items: list[tuple], page_rows: Sequence[pymupdf.Rect]) -> list[list[int]]:
    """A list item breaks the line before reaching the right edge of the column although the first word of the
    next line would fit: the next line is a new line of its own (examples listed under an entry, American,
    Airlines, Tim, Wagner ...), so split it. Automatic wrapping only moves a word down when the line above
    cannot hold it.
    The right edge of the column is voted from the starts of paragraphs within three ems left of the list item
    (see _voted_right). When the entry's lines are ragged and no edge comes out of the vote, nothing is split:
    on a two-column page the right edge of the text frame is the right edge of the other column, and measured
    against it every line looks short.
    When an entry fills a whole line, an early break cannot be seen. When the lines above have not a single
    colored character and most letters on this line are in a vivid color, it is also an example listed on a
    line of its own (M1: NAME [it or he would be PRONOUN]), so split it.
    Only entries starting with a bullet are considered: one starting with A. or 12. may be a reference, where
    name initials look like letter labels, titles are often colored links, and line ends are ragged too."""
    if len(group) < 2 or not _MARK_ITEM_RE.match(items[group[0]][0]):
        return [group]
    rows: list[list[int]] = []
    for i in group:
        rect = items[i][1]
        if rows:
            top = pymupdf.Rect(items[rows[-1][0]][1])
            for j in rows[-1][1:]:
                top |= items[j][1]
            if top.y0 < (rect.y0 + rect.y1) / 2 < top.y1:
                rows[-1].append(i)
                continue
        rows.append([i])
    if len(rows) < 2:
        return [group]
    first = items[group[0]][1]
    size = max(float(span["size"]) for span in items[group[0]][2])
    lefts = sorted({round(row.x0) for row in page_rows if first.x0 - 3 * size <= row.x0 <= first.x0 + 1})
    right = next((edge for left in lefts if (edge := _voted_right(left, size, page_rows)) is not None), None)

    def spans(row: list[int]) -> list[dict]:
        return [span for i in row for span in items[i][2]]

    parts = [list(rows[0])]
    for upper, row in zip(rows, rows[1:]):
        rect = pymupdf.Rect(items[upper[0]][1])
        for j in upper[1:]:
            rect |= items[j][1]
        lower = (items[row[0]][1], items[row[0]][0], size)
        early = right is not None and _breaks_early([(rect, "", size), lower], lambda _rect, _size: right)
        colored = _vivid_share(spans(parts[-1])) == 0 and _vivid_share(spans(row)) > 0.5
        if early or colored:
            parts.append(list(row))
        else:
            parts[-1].extend(row)
    return parts


def _column_rows(rows: list[list[int]], items: list[tuple], size: float) -> bool:
    """Every line starts a new run of text at the same x (other than the line start), and at least one line has
    a gap of more than one em before it: a table column.
    In lines wrapped from a paragraph, word starts do not fall at the same x on every line, and the space
    between words is only one space wide."""
    starts: list[list[tuple[float, float]]] = []
    for row in rows:
        spans = sorted((span for i in row for span in items[i][2]), key=lambda span: span["bbox"][0])
        found = []
        for prev, span in zip(spans, spans[1:]):
            if span["text"][:1].isspace():
                continue
            found.append((span["bbox"][0], span["bbox"][0] - prev["bbox"][2]))
        starts.append(found)
    for x, _gap in starts[0]:
        gaps = []
        for found in starts:
            near = [gap for other, gap in found if abs(other - x) <= size * 0.5]
            if not near:
                break
            gaps.append(max(near))
        else:
            if max(gaps) >= size:
                return True
    return False


def _same_opening(texts: Sequence[str]) -> bool:
    """Every line starts with the same word (comparing only letters and digits, so x_i and x_i's are the
    same): a list set line by line."""
    heads = []
    for text in texts:
        match = re.match(r"\w+", text.strip())
        if match is None:
            return False
        heads.append(match.group(0))
    return len(set(heads)) == 1


# Abbreviations that appear almost only inside sentences. etc. often ends a sentence and is not included
_MID_ABBREV_RE = re.compile(r"(?:\bet al|\be\.g|\bi\.e|\bcf|\bvs)\.$")
