"""Read PDF pages into text blocks: split and rejoin lines, cut table cells, contents leaders
and list items, and merge lines that continue a paragraph."""

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import replace

import pymupdf

from .pdf_blocks import (
    _AFFIL_MARK_RE,
    _EQUATION_TAG_RE,
    _LABEL_START_RE,
    _LINE_GAP,
    _MARK_ITEM_RE,
    _MAX_INDENT_EM,
    _NUMBERED_HEAD_RE,
    _ROMAN_ITEM_RE,
    CJK_RE,
    Leader,
    TextBlock,
    _attach_ink,
    _by_chars,
    _diagramish,
    _ends_in_call,
    _ends_sentence,
    _first_line_indented,
    _indents_paragraphs,
    _is_side_label,
    _join_lines,
    _list_item,
    _numbered_line,
    _small_caps_sizes,
    _span_is_math,
    _styled_edge,
    _visual_lines,
)
from .pdf_flow import (
    emphasis_of,
)
from .pdf_frame import FrameLine, TextFrame, margin_notes, note_groups, note_pitch, text_frames
from .pdf_layout import (
    PageGeometry,
    PageScan,
    RuledTable,
    cell_containing,
    margin_skips,
    reading_order,
    scan_page,
)
from .pdf_math import (
    _fraction_parts,
    _line_box,
    _mark_detected_formulas,
    _mark_formula_only_fonts,
    _mark_math_alphabet,
    _mark_math_roman,
    _mark_overlays,
    _mark_radicands,
    _mark_underlines,
    _mark_visible_spaces,
    _PageChars,
    _plain_units,
    _prose_units,
    _script_parts,
    _script_seam,
    _split_accents,
    _text_bounds,
)
from .pdf_type3 import Type3Reader, recover_type3_page
from .pdf_vision import FormulaDetector

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


def _latin_page(data: dict) -> bool:
    """More than 90% of the characters on the page are not CJK"""
    text = "".join(
        span["text"]
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    )
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and len(CJK_RE.findall(text)) < letters * 0.1


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


def _has_leader_line(data: dict) -> bool:
    return any(
        _LEADER_RE.search("".join(span["text"] for span in line["spans"]))
        for block in data.get("blocks", [])
        for line in block.get("lines", [])
    )


# Title of a contents page
_TOC_TITLE_RE = re.compile(r"(?:(?:table|summary) of )?contents|list of (?:figures|tables)|目\s*录", re.I)
# A page needs at least this many entries in its page number column to count as contents
_TOC_MIN_ENTRIES = 5


# Index entries end with page numbers: "fully-connected, 157", "linear chain CRF, 414, 415", "Kleene *, 50–52"
_INDEX_REF_RE = re.compile(r",\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*$")
# When an entry is too long, its page numbers wrap to the next line, which has only page numbers:
# "BNF (Backus-Naur form)," with "426" indented on the next line
_INDEX_REFS_ONLY_RE = re.compile(r"^\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*,?$")
# A page is an index page only when it has at least this many lines and this share of them end with page numbers
_INDEX_MIN_LINES = 15
_INDEX_SHARE = 0.4


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


def _index_page(data: dict) -> bool:
    """An index page at the back of a book: 40% or more of the lines end with ", page number". Body lines
    rarely stop at a comma followed by a number."""
    texts = [
        "".join(span["text"] for span in line["spans"]).strip()
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
    ]
    texts = [text for text in texts if text]
    if len(texts) < _INDEX_MIN_LINES:
        return False
    return sum(bool(_INDEX_REF_RE.search(text)) for text in texts) >= len(texts) * _INDEX_SHARE


def _index_ref_rows(data: dict) -> list[pymupdf.Rect]:
    """Lines on an index page that hold only page numbers. PyMuPDF often puts such a line and the entry on the
    line above into two blocks, so they are collected across the whole page."""
    return [
        _line_box(line)
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        if _INDEX_REFS_ONLY_RE.match("".join(span["text"] for span in line["spans"]).strip())
    ]


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


def _toc_pages(pages: list[dict]) -> set[int]:
    """Contents pages without leaders: a title line, and page numbers far from the rest of each line,
    right-aligned and increasing down the page.
    When the contents span several pages, the later pages have no title; a page that follows a contents page
    and has the same page number column also counts.
    A page number column alone easily mistakes a table with a column of integers for contents, so a title or a
    preceding contents page is required."""
    found: set[int] = set()
    previous = False
    for pno, data in enumerate(pages):
        lines = [
            line
            for block in data.get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            if abs(line["dir"][1]) <= 0.01
        ]
        titled = any(_TOC_TITLE_RE.fullmatch("".join(span["text"] for span in line["spans"]).strip()) for line in lines)
        if _page_number_column(lines) and (titled or previous):
            found.add(pno)
            previous = True
        else:
            previous = _has_leader_line(data)
    return found


def _page_number_column(lines: list[dict]) -> bool:
    """Page numbers at the end of lines: a line holding only one number, with other text more than two ems to
    its left on the same row.
    Such page numbers form a right-aligned column that mostly does not decrease from top to bottom. When a
    list of tables follows the contents, the page numbers start over, so a few drops are allowed."""
    found: list[tuple[float, float, int]] = []
    for line in lines:
        text = "".join(span["text"] for span in line["spans"]).strip()
        if not re.fullmatch(r"\d{1,4}", text):
            continue
        box = pymupdf.Rect(line["bbox"])
        size = max((float(span["size"]) for span in line["spans"]), default=10.0)
        for other in lines:
            if other is line:
                continue
            left = pymupdf.Rect(other["bbox"])
            overlap = min(left.y1, box.y1) - max(left.y0, box.y0)
            if overlap >= min(left.height, box.height) * 0.5 and box.x0 - left.x1 > size * 2:
                found.append((box.x1, box.y0, int(text)))
                break
    best = max(
        ([(y0, number) for x1, y0, number in found if abs(x1 - edge) <= 1.5] for edge in {x for x, _, _ in found}),
        key=len,
        default=[],
    )
    numbers = [number for _, number in sorted(best)]
    drops = sum(1 for a, b in zip(numbers, numbers[1:]) if b < a)
    return len(numbers) >= _TOC_MIN_ENTRIES and drops <= len(numbers) // 5


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


def _number_columns(data: dict) -> list[float]:
    """Center lines of the number columns in tables on this page: three or more number cells whose centers
    fall at the same x (within one point)."""
    centers = sorted(
        (float(span["bbox"][0]) + float(span["bbox"][2])) / 2
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if str(span["text"]).strip()
        and _number_cell(str(span["text"]))
        and not str(span["text"]).rstrip().endswith(".")
    )
    columns: list[float] = []
    run: list[float] = []
    for center in centers:
        if run and center - run[0] > 1:
            if len(run) >= 3:
                columns.append(sum(run) / len(run))
            run = []
        run.append(center)
    if len(run) >= 3:
        columns.append(sum(run) / len(run))
    return columns


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


def _verbatim_block(block: TextBlock) -> bool:
    """The whole block is verbatim text in a monospaced font (example dialogue, code), set as separate
    lines."""
    return bool(block.span_lines) and all(_typewriter_line(spans) for spans in block.span_lines)


def _verbatim_seam(host: TextBlock, guest: TextBlock) -> bool:
    """One side is verbatim monospaced text set on its own, and the line touching it on the other side is body
    text. A verbatim line has no sentence-final punctuation, yet the lowercase body text below it is not its
    continuation. Code inside body text that happens to fall at the start or end of a line is joined as
    usual."""
    if not host.span_lines or not guest.span_lines:
        return False
    return (
        _verbatim_block(host)
        and not _typewriter_line(guest.span_lines[0])
        or _verbatim_block(guest)
        and not _typewriter_line(host.span_lines[-1])
    )


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


def _new_voice(host: TextBlock, guest: TextBlock) -> bool:
    """The next block differs in size by more than 10% and also changes color: an example under a slide title,
    or a caption under body text, and not a wrapped line of the same paragraph.
    Within one paragraph the body size and color stay the same, and emphasized words and formulas inside it
    never make up the majority."""
    return abs(guest.size - host.size) > host.size * 0.12 and guest.color != host.color


def _restyled_seam(host: TextBlock, guest: TextBlock) -> bool:
    """The two blocks are part of one centered stack of lines, and the lines on either side of the seam change
    style: bold author names followed by an affiliation in a regular font, an affiliation followed by an email
    address in a monospaced font. Each line is one item, not a wrapped line of a paragraph. A line split into
    several pieces is compared as a whole line."""
    if not host.span_lines or not guest.span_lines or not host.line_rects or not guest.line_rects:
        return False
    if len(host.span_lines) != len(host.line_rects) or len(guest.span_lines) != len(guest.line_rects):
        return False
    if not _centered_rows(_row_boxes(host.line_rects + guest.line_rects), host.size):
        return False
    gap = max(host.size * 0.4, 2.0)

    def row_spans(block: TextBlock, at: pymupdf.Rect) -> list[dict]:
        return [
            span
            for rect, spans in zip(block.line_rects, block.span_lines)
            if abs(rect.y0 - at.y0) <= gap
            for span in spans
        ]

    upper = row_spans(host, host.line_rects[-1])
    lower = row_spans(guest, guest.line_rects[0])
    return _row_style(upper) != _row_style(lower)


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


def _is_formula_char(c: str) -> bool:
    """Per-character formula test: Greek letters, math symbols and modifiers, Private Use Area."""
    if 0x370 <= ord(c) <= 0x3FF:
        return True
    return unicodedata.category(c) in ("Sm", "Sk", "Mn", "Co")


def _math_chars(spans: list[dict]) -> int:
    """Number of formula characters in a block: all characters in math fonts plus formula characters in body
    fonts."""
    n = 0
    for s in spans:
        if _span_is_math(s):
            n += len(s["text"])
        else:
            n += sum(1 for c in s["text"] if _is_formula_char(c))
    return n


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


def _caption_label(text: str) -> bool:
    return bool(_CAPTION_LABEL_RE.match(text.strip()))


# A footnote is one size smaller than body text and starts with a number (1 Although / 2Note / 4https://).
# Section numbers 1. / 2.2 are not footnotes. A footnote at the bottom of the page must not take the place of
# the page end of a paragraph carried over to the next page, or a paragraph broken after "quite" would not join
# "well" on the next page.
_FOOTNOTE_LEAD_RE = re.compile(r"^\d{1,2}(?!\.\d)(?:https?://|\s*[A-Za-z])", re.I)


def _unclosed_bracket(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth:
            depth -= 1
    return depth > 0


def _closes_first(text: str) -> bool:
    """The first sentence of the block contains an unmatched closing bracket."""
    depth = 0
    for match in re.finditer(r"[()\[\]]|[.!?](?=\s+[A-Z])", text[:300]):
        mark = match.group()
        if mark in "([":
            depth += 1
        elif mark in ")]":
            if depth == 0:
                return True
            depth -= 1
        elif depth == 0 and not _MID_ABBREV_RE.search(text[: match.end()]):
            return False
    return False


# A caption or listing at the top of a column does not take the place of the continued sentence. Figure 3 shows
# and Figure 8(a) shows in body text are not captions: a lowercase word follows the number.
# The number may also be Roman (IEEE's TABLE IV) or carry an appendix letter (Table A1, Figure S3, Figure B.6)
_FLOAT_HEAD_RE = re.compile(
    r"^(?:Figure|Fig\.?|Table|Tab\.?|Listing|List\.?|Scheme|Plate|Algorithm|图|表|清单)"
    r"\s*(?:(?-i:[A-Z])?\.?\d++(?:\.\d+)*+|(?-i:[IVXLC]++))\b"
    r"(?!(?:\s*\((?-i:[a-z])\))?\s+(?-i:[a-z]))",
    re.I,
)


def _is_float_text(text: str) -> bool:
    """A caption or listing title at the top of a column. The body text after it is what continues the
    paragraph before."""
    return _FLOAT_HEAD_RE.match(text.strip()) is not None


def _overlaps_line(block: TextBlock, other: TextBlock) -> bool:
    lines = other.line_rects or [other.rect]
    height = block.rect.height or block.size
    for line in lines:
        overlap = min(block.rect.y1, line.y1) - max(block.rect.y0, line.y0)
        if overlap > min(height, line.height) * 0.5:
            return True
    return False


def _blocks_share_column(
    left: TextBlock,
    right: TextBlock,
    geometries: dict[int, PageGeometry] | None,
) -> bool:
    """Even with a gap narrower than the column gap, confirm that the two blocks are in the same column.
    Without column information, rely on the gap alone."""
    if not geometries or left.page not in geometries:
        return True
    geo = geometries[left.page]
    if len(geo.columns) < 2:
        return True
    return geo.index_of(left.rect) == geo.index_of(right.rect) and not geo.spans_columns(left.rect | right.rect)


def _line_spans(block: TextBlock) -> list[list[dict]]:
    """Spans per line, one entry for each of line_rects. When they do not match, each line gets an empty list
    and no decision is made by span."""
    rows = len(block.line_rects) or 1
    if block.span_lines and len(block.span_lines) == rows:
        return [list(line) for line in block.span_lines]
    return [[] for _ in range(rows)]


def _shares_visual_row(
    left: TextBlock,
    right: TextBlock,
    geometries: dict[int, PageGeometry] | None,
) -> bool:
    """Whether the two blocks sit on the same baseline and are adjacent horizontally.

    Line boxes are used, not the union of the blocks. A short heading on the left sits beside only the first
    body line; the union would cover it and the paragraph would swallow the heading. A column gap is at least
    12pt, so a 6pt gap cannot reach another column.
    """
    left_lines = list(zip(left.line_rects or [left.rect], _line_spans(left)))
    right_lines = list(zip(right.line_rects or [right.rect], _line_spans(right)))
    for a, a_spans in left_lines:
        for b, b_spans in right_lines:
            overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
            if overlap <= min(a.height, b.height) * 0.5 or _stacked_rows(a_spans, b_spans):
                continue
            horizontal = min(a.x1, b.x1) - max(a.x0, b.x0)
            gap = b.x0 - a.x1 if b.x0 >= a.x0 else a.x0 - b.x1
            if horizontal > 0 or (
                0 <= gap < 6
                and _blocks_share_column(left, right, geometries)
                and not (b.x0 >= a.x0 and _script_seam(a_spans, b_spans))
            ):
                return True
    return False


def _merge_visual_lines(
    blocks: list[TextBlock],
    geometries: dict[int, PageGeometry] | None = None,
) -> list[TextBlock]:
    """Merge adjacent blocks that are really one visual line (PyMuPDF splits a line into two blocks where the
    font switches at a formula).

    Unmerged, the two translations would be written into overlapping rectangles on top of each other.
    """
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            if (
                p.page == b.page
                and p.clip == b.clip
                and not (p.side_note or b.side_note)
                and _shares_visual_row(p, b, geometries)
            ):
                p.line_rects.extend(b.line_rects)
                if p.span_lines and b.span_lines:
                    p.span_lines.extend(b.span_lines)
                else:
                    p.span_lines = None
                p.text = _join_lines([p.text, b.text])
                p.rect |= b.rect
                continue
        out.append(b)
    return out


# A caption label that is a span of its own, with a section-style number
_CAPTION_SPAN_RE = re.compile(r"^(?:Figure|Fig\.?|Table|Tab\.?)\s*\d+(?:[.\-]\d+)*\s*[:.]?$", re.I)
# A caption label may have no colon: "图 2", "Fig. 1."
_CAPTION_LABEL_RE = re.compile(
    r"^(?:(?:Figure|Fig\.?|Table|Tab\.?|Scheme|Plate)\s*\d+|(?:图|表)\s*\d+)\s*[:.。：]?\s*$",
    re.I,
)
_OPERATOR_NAMES = (
    "arg|max|min|log|ln|sin|cos|tan|cot|sec|csc|exp|lim|sup|inf|det|dim|ker|gcd|mod|sgn|deg|hom|Pr|arcsin|sinh|cosh"
)


# Expressions not set in a math font (older papers use Times-Italic for variables and Symbol for operators):
# variables in body italics, brackets, operators.
# A variable is a single letter or two capital letters (PE). Two-letter words such as If and op count only
# together with brackets or operators (op c)); theorem bodies are italic throughout, and If or In next to a
# formula is still body text
_ITALIC_FONT_RE = re.compile(r"Italic|Oblique", re.I)
_NUMERAL_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _block_size(main: dict, span_lines: list[list[dict]]) -> float:
    """Font size of the block: the size of its longest piece. In a small-caps heading (INITIALIZATION, with a
    larger initial) the longest piece is the small part, so the size of the capitals is used. The capitals
    count only on the same baseline; one offset downward is a subscript (BERT_LARGE)."""
    size = float(main["size"])
    big = _small_caps_sizes(span_lines).get(id(main))
    if big is None or main.get("origin") is None:
        return size
    base = float(main["origin"][1])
    for line in span_lines:
        for span in line:
            if (
                span is not main
                and str(span.get("text") or "").strip()
                and span.get("origin") is not None
                and abs(float(span.get("size") or 0) - big) <= 0.1
                and abs(float(span["origin"][1]) - base) <= big * 0.1
            ):
                return big
    return size


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


def _donate_punct_rects(prev: TextBlock, line_rects: list[pymupdf.Rect]) -> None:
    """When a punctuation fragment sits at the end of a line of the previous block, add its rectangles to that
    line for masking.

    The gap is measured from that line's x1, not from the union of the block: the union would erase the
    formula on the right, and formula lines are not redrawn.
    The fragment must be near the end of the line: tick numbers on a figure in the neighbouring column at the
    same height do not belong to this line, and merging them would stretch the write box into that column.
    """
    if not line_rects or not prev.line_rects:
        return
    fragment = line_rects[0]
    height = fragment.y1 - fragment.y0
    if height <= 0:
        return
    for tail in prev.line_rects:
        overlap = min(tail.y1, fragment.y1) - max(tail.y0, fragment.y0)
        if overlap > 0.5 * height and -height < fragment.x0 - tail.x1 < 2:
            prev.line_rects.extend(line_rects)
            return


def _clip_key(cell: pymupdf.Rect | None) -> tuple[float, float, float, float] | None:
    if cell is None:
        return None
    return (cell.x0, cell.y0, cell.x1, cell.y1)


def _split_table_blocks(blocks: list[TextBlock], tables: tuple[RuledTable, ...]) -> list[TextBlock]:
    """Split cells side by side in one line into separate blocks. A neighbouring cell that starts in lowercase
    would not be cut by the hard break rule."""
    if not tables:
        return blocks
    out: list[TextBlock] = []
    for block in blocks:
        out.extend(_split_one_table_block(block, tables))
    return out


def _split_one_table_block(block: TextBlock, tables: tuple[RuledTable, ...]) -> list[TextBlock]:
    lines = block.span_lines or []
    if len(lines) != len(block.line_rects):
        block.clip = _clip_key(cell_containing(block.rect, tables))
        return [block]
    grouped: dict[tuple[float, float, float, float] | None, list[tuple[pymupdf.Rect, list[dict]]]] = {}
    order: list[tuple[float, float, float, float] | None] = []
    for line_rect, spans in zip(block.line_rects, lines):
        by_cell: dict[tuple[float, float, float, float] | None, list[dict]] = {}
        cell_order: list[tuple[float, float, float, float] | None] = []
        for span in spans:
            if not str(span.get("text", "")).strip():
                # In justified lines the space between words is often a span of its own; it goes into the same
                # cell as the piece before it, and without it the words run together
                if cell_order:
                    by_cell[cell_order[-1]].append(span)
                continue
            bbox = span.get("bbox")
            key = _clip_key(cell_containing(_text_box(span), tables) if bbox else None)
            if key not in by_cell:
                cell_order.append(key)
                by_cell[key] = []
            by_cell[key].append(span)
        if not cell_order:
            key = _clip_key(cell_containing(line_rect, tables))
            _remember_cell(grouped, order, key, line_rect, [])
            continue
        if len(cell_order) == 1:
            _remember_cell(grouped, order, cell_order[0], line_rect, by_cell[cell_order[0]])
            continue
        for key in cell_order:
            box = pymupdf.Rect()
            for span in by_cell[key]:
                if str(span.get("text", "")).strip():
                    box |= pymupdf.Rect(span["bbox"])
            _remember_cell(grouped, order, key, box, by_cell[key])
    if len(order) <= 1:
        block.clip = order[0] if order else _clip_key(cell_containing(block.rect, tables))
        return [block]
    parts: list[TextBlock] = []
    for key in order:
        pieces = grouped[key]
        rect = pymupdf.Rect()
        for line_rect, _spans in pieces:
            rect |= line_rect
        text = _join_lines(["".join(str(span.get("text", "")) for span in spans) for _rect, spans in pieces])
        if not text.strip():
            continue
        parts.append(
            replace(
                block,
                rect=rect,
                line_rects=[line_rect for line_rect, _spans in pieces],
                text=text,
                span_lines=[spans for _rect, spans in pieces],
                clip=key,
                runs=[],
            )
        )
    return parts or [block]


def _remember_cell(
    grouped: dict[tuple[float, float, float, float] | None, list[tuple[pymupdf.Rect, list[dict]]]],
    order: list[tuple[float, float, float, float] | None],
    key: tuple[float, float, float, float] | None,
    line_rect: pymupdf.Rect,
    spans: list[dict],
) -> None:
    if key not in grouped:
        order.append(key)
        grouped[key] = []
    grouped[key].append((line_rect, spans))


# Tolerance for aligning lines in a wide block with the position of margin notes. The first column of a table
# is often indented slightly more than the caption label (99 against 96)
_WIDE_ALIGN = 4.0
_LIST_ITEM_RE = re.compile(r"^\d+[\.,]\s+[A-Za-z]")


def _numbered_list_item(text: str) -> bool:
    """'5. part (0.2M)' has a low share of letters but is still a list item, not a formula line."""
    stripped = text.strip()
    if not _LIST_ITEM_RE.match(stripped) or len(stripped) > 80:
        return False
    return any(len(word) >= 3 and word.isalpha() for word in re.findall(r"[A-Za-z]+", stripped))


def _frame_lines(data: dict) -> tuple[list[FrameLine], list[tuple[dict, dict]]]:
    """Lines used to vote on the text frame, and the block and line dict of each line."""
    lines: list[FrameLine] = []
    refs: list[tuple[dict, dict]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            if abs(line["dir"][1]) > 0.01:
                continue
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = "".join(span["text"] for span in line["spans"])
            lines.append(FrameLine(pymupdf.Rect(line["bbox"]), text, max(span["size"] for span in spans)))
            refs.append((block, line))
    return lines, refs


def _detach_margin_notes(
    data: dict,
    lines: list[FrameLine],
    refs: list[tuple[dict, dict]],
    notes: set[int],
    pitches: dict[float, float] | None = None,
) -> None:
    """Move margin note lines out of their original blocks; each margin note becomes its own block, placed at
    the end of the page.

    Left in place, a note would separate the two pieces of body text it sits between, and the two pieces could
    not be joined back into one paragraph.
    """
    if not notes:
        return
    picked = sorted(notes)
    touched: set[int] = set()
    for group in note_groups([lines[index] for index in picked], pitches):
        members = [refs[picked[index]] for index in group]
        rect = pymupdf.Rect()
        for block, line in members:
            block["lines"].remove(line)
            touched.add(id(block))
            rect |= pymupdf.Rect(line["bbox"])
        data["blocks"].append(
            {
                "type": 0,
                "bbox": tuple(rect),
                "lines": [line for _, line in members],
                "side_note": True,
            }
        )
    _rejoin_note_cuts(data, touched)
    kept = []
    for block in data["blocks"]:
        if block.get("type") == 0:
            if not block["lines"]:
                continue
            box = pymupdf.Rect()
            for line in block["lines"]:
                box |= pymupdf.Rect(line["bbox"])
            block["bbox"] = tuple(box)
        kept.append(block)
    data["blocks"] = kept


# A line this long counts as a body line; at least this many of them starting at the same x (within
# _EDGE_NEAR) make an edge of the body text on this page
_WIDE_TEXT = 40
_WIDE_ROWS = 3
_EDGE_NEAR = 2


def _wide_block_lines(
    lines: list[FrameLine],
    refs: list[tuple[dict, dict]],
    notes: set[int],
    frame: TextFrame | None,
) -> set[int]:
    """Lines of wide tables and wide captions reaching into the margin that fall where margin notes would be.
    When another line in the block starts at the same edge and runs into the text frame, this line is part of
    the block (the label of Figure 3.7, with the next caption line starting at the same x), not a margin note.
    The same holds when another line starts deep in the margin and runs into the text frame: body text does
    not start in the margin, so the whole block reaches into the margin (the label of Figure 6.17 followed by
    the caption, with the tion of on the next line also in the margin)."""
    if frame is None:
        return set()
    note_lines = {id(refs[index][1]) for index in notes}
    # The outermost x where body text on this page starts and ends (several lines are there). The text frame
    # is voted across pages on the same side, and the body text of some pages extends beyond it
    long_lines = [line.rect for line in lines if len(line.text.strip()) >= _WIDE_TEXT]
    starts = Counter(round(rect.x0) for rect in long_lines)
    ends = Counter(round(rect.x1) for rect in long_lines)
    body_x0 = min([frame.x0, *(x for x in starts if _near_votes(starts, x, _EDGE_NEAR) >= _WIDE_ROWS)])
    body_x1 = max([frame.x1, *(x for x in ends if _near_votes(ends, x, _EDGE_NEAR) >= _WIDE_ROWS)])
    kept: set[int] = set()
    for index in notes:
        block, _line = refs[index]
        rect = lines[index].rect
        left = rect.x1 <= frame.x0
        for other in block["lines"]:
            if id(other) in note_lines:
                continue
            box = pymupdf.Rect(other["bbox"])
            deep = lines[index].size * 2
            if left and (abs(box.x0 - rect.x0) <= _WIDE_ALIGN or box.x0 <= body_x0 - deep) and box.x1 > frame.x0:
                kept.add(index)
                break
            if not left and (abs(box.x1 - rect.x1) <= _WIDE_ALIGN or box.x1 >= body_x1 + deep) and box.x0 < frame.x1:
                kept.add(index)
                break
        if index in kept:
            continue
        # When an inline expression cuts a caption line into several blocks, the piece left in the margin
        # (that w[1]) is a block of its own.
        # When the line right above or below starts at the same x and runs into the text frame, this piece is
        # also a caption line
        pitch = lines[index].size * 0.6
        for other_index, other in enumerate(lines):
            if other_index in notes:
                continue
            box = other.rect
            if min(abs(box.y0 - rect.y1), abs(rect.y0 - box.y1)) > pitch:
                continue
            if left and abs(box.x0 - rect.x0) <= _WIDE_ALIGN and box.x1 > frame.x0:
                kept.add(index)
                break
            if not left and abs(box.x1 - rect.x1) <= _WIDE_ALIGN and box.x0 < frame.x1:
                kept.add(index)
                break
    return kept


def _rejoin_note_cuts(data: dict, touched: set[int]) -> None:
    """PyMuPDF often cuts a paragraph into two blocks at the line where a margin note sits. Once the notes are
    removed, two blocks right above each other and cut mid-sentence are joined back into one: a line break
    before a capital (Standard / Code for ...) is not recognized by block merging. When the upper block ends
    with a sentence end, or the lower block starts indented, they are two paragraphs."""
    blocks = [block for block in data["blocks"] if block.get("type") == 0 and not block.get("side_note")]
    index = 0
    while index < len(blocks) - 1:
        upper, lower = blocks[index], blocks[index + 1]
        if (id(upper) in touched or id(lower) in touched) and _note_cut(upper, lower):
            upper["lines"].extend(lower["lines"])
            lower["lines"] = []
            touched.add(id(upper))
            del blocks[index + 1]
            continue
        index += 1


def _note_cut(upper: dict, lower: dict) -> bool:
    above = [line for line in upper["lines"] if abs(line["dir"][1]) <= 0.01]
    below = [line for line in lower["lines"] if abs(line["dir"][1]) <= 0.01]
    if not above or not below:
        return False
    last = pymupdf.Rect(max(above, key=lambda line: line["bbox"][3])["bbox"])
    first = pymupdf.Rect(min(below, key=lambda line: line["bbox"][1])["bbox"])
    size = max((span["size"] for line in above for span in line["spans"] if span["text"].strip()), default=0)
    lower_size = max((span["size"] for line in below for span in line["spans"] if span["text"].strip()), default=0)
    if size <= 0 or abs(lower_size - size) > size * 0.15:
        return False
    if not -size * 0.3 <= first.y0 - last.y1 <= size * 0.6:
        return False
    # A lower block indented to the right is a new paragraph; one back to the left at the column edge can be
    # the continuation of a line with a first-line indent
    if not last.x0 - size * 3 <= first.x0 <= last.x0 + 2:
        return False
    # Continuation lines of a list item align with the item text; a lower block starting further left is body
    # text after the list has ended
    head = min(above, key=lambda line: line["bbox"][1])
    if _list_item("".join(span["text"] for span in head["spans"])) and first.x0 < last.x0 - size * 0.5:
        return False
    # A lower block with its own first-line indent (the first line right of the lines below) is a new
    # paragraph: a new paragraph following the translation of an example
    rest = [line["bbox"][0] for line in below if line["bbox"][1] >= first.y1 - 1 and line["bbox"][2] > first.x0]
    if rest and first.x0 > min(rest) + size * 0.5:
        return False
    tail = "".join(span["text"] for line in above if pymupdf.Rect(line["bbox"]) == last for span in line["spans"])
    return not _ends_sentence(tail)


def extract_blocks(doc: pymupdf.Document) -> list[TextBlock]:
    return _extract(doc)[0]


def _extract(
    doc: pymupdf.Document,
    on_page: Callable[[int, int], None] | None = None,
    formulas: FormulaDetector | None = None,
) -> tuple[list[TextBlock], list[TextFrame | None], list[PageScan]]:
    """Text blocks, and for each page the text frame and the tables, figures and words. on_page(pages read,
    total pages) is called after each page is read.
    formulas is the formula detection model; when given, the inline formulas it boxes are marked on the
    characters. The model runs ahead on the next page in a background thread (see FormulaDetector.pages)."""
    blocks = []
    scans: list[PageScan] = []
    tables_by_page: list[tuple[RuledTable, ...]] = []
    type3 = Type3Reader(doc)
    pages: list[dict] = []
    pending_cells: set[int] = set()
    detected = formulas.pages() if formulas is not None else None
    try:
        for page in doc:
            scans.append(scan_page(page))
            tables_by_page.append(scans[-1].tables)
            data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
            _attach_ink(page, data)
            page_chars = _PageChars(page)
            _mark_underlines(page_chars, data, scans[-1].strokes)
            _mark_radicands(page_chars, data, scans[-1].strokes)
            _mark_overlays(page_chars, data)
            _split_accents(page_chars, data)
            if detected is not None:
                _mark_detected_formulas(page_chars, data, next(detected))
            _mark_visible_spaces(page_chars, data, scans[-1].visible_spaces)
            # Bitmap glyphs with a custom encoding extract as character codes. They are recognized as letters
            # before the paragraph splitting below.
            recover_type3_page(page, data, type3)
            pages.append(data)
            if on_page is not None:
                on_page(len(pages), doc.page_count)
    finally:
        if detected is not None:
            # Stop background inference. When stopped midway (see background.check_stop), this thread also
            # waits for it to finish the page in hand
            detected.close()
    _mark_math_roman(pages)
    _mark_math_alphabet(pages)
    _mark_formula_only_fonts(pages)
    # No table search on contents pages: the page number column and the section number column align, short
    # entries would be taken as a cell table, and cells are not translated.
    # The same for index pages: several columns of short entries align at their starts, and a few rows of
    # them look like a table without rules
    toc_pages = _toc_pages(pages)
    index_pages = {pno for pno, data in enumerate(pages) if _index_page(data)}
    for pno in toc_pages | index_pages:
        tables_by_page[pno] = ()
        scans[pno] = replace(scans[pno], tables=())
    for data in pages:
        _rejoin_split_lines(data)
    line_gap = _line_gap(pages)
    body_size = _body_size(_sized_lines(pages))
    page_lines = [_frame_lines(data) for data in pages]
    frames = text_frames([lines for lines, _refs in page_lines])
    notes = margin_notes([lines for lines, _refs in page_lines], frames)
    pitches = note_pitch([lines for lines, _refs in page_lines], notes)
    for pno, (data, (lines, refs), found, frame) in enumerate(zip(pages, page_lines, notes, frames)):
        # An index page has several columns, and the leftmost lies outside the body text frame, looking like
        # margin notes. Index pages have no margin notes
        if pno in index_pages:
            continue
        _detach_margin_notes(data, lines, refs, found - _wide_block_lines(lines, refs, found, frame), pitches)
    absorbed: set[int] = set()
    for pno, data in enumerate(pages):
        margins = _left_margins(data)
        # Start and end of each line on this page, used to find the right edge of the column a line is in
        # Math lines on this page that keep the original text; fragments inside them (the denominator dx) are
        # math too
        kept_rows: list[pymupdf.Rect] = []
        page_start = len(blocks)
        row_ends = [
            _line_box(line)
            for block in data["blocks"]
            if block.get("type") == 0
            for line in block["lines"]
            if "".join(span["text"] for span in line["spans"]).strip()
        ]
        fractions = _fraction_parts(data, scans[pno].rules)
        toc_page = _has_leader_line(data) or pno in toc_pages
        index_page = _index_page(data)
        ref_rows = _index_ref_rows(data) if index_page else []
        # Page numbers on an index page form several columns, and words of entries happen to line up with them
        number_columns = [] if index_page else _number_columns(data)
        latin_page = _latin_page(data)
        for b in data["blocks"]:
            if b.get("type") != 0:
                continue
            items: list[tuple[str, pymupdf.Rect, list[dict]]] = []
            # Chinese source sentences split off (see _script_parts)
            cjk_heads: set[int] = set()
            for ln in b["lines"]:
                if abs(ln["dir"][1]) > 0.01:  # Rotated or vertical text is not handled
                    continue
                text = "".join(s["text"] for s in ln["spans"])
                if not text.strip():
                    continue
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                parts = _script_parts(ln, spans) if latin_page else []
                if parts:
                    cjk_heads.add(id(parts[0][2]))
                items.extend(parts or [(text, _line_box(ln), spans, ln["spans"])])
            items = _left_to_right(items)
            items, leaders = _cut_leaders(items, toc_page)
            items = _drop_number_tails(items)
            items, header_cells = _cut_header_cells(items, number_columns)
            side_note = bool(b.get("side_note"))
            # One margin note is one block and is not split by the body text rules
            groups = [list(range(len(items)))] if side_note and items else _split_lines(items, index_page)
            # A line with a leader is the last line of a contents entry, and the next line is another entry. The
            # same goes for a line ending with page numbers on an index page, an entry whose page numbers wrap
            # to the next line, and one cell of a word grid
            cells = _grid_cells(items)
            entry_ends = cells | header_cells | (_index_entry_ends(items, ref_rows) if index_page else set())
            # A Chinese source sentence is cut off on both sides and forms its own group
            entry_ends |= {j for i, item in enumerate(items) if id(item[2]) in cjk_heads for j in (i - 1, i) if j >= 0}
            groups = [part for group in groups for part in _cut_after(group, {*leaders, *entry_ends})]
            groups = [part for group in groups for part in _cut_at_posts(group, items, scans[pno].posts)]
            # A line holding a text fraction forms its own group, keeps the original text, and is not joined
            # with the body text above and below
            formula_rows = {i for i, item in enumerate(items) if any(id(span) in fractions for span in item[2])}
            groups = [part for group in groups for part in _cut_runs(group, formula_rows)]
            groups = [part for group in groups for part in _cut_line_list(group, items, row_ends)]
            groups = [part for group in groups for part in _cut_item_breaks(group, items, row_ends)]
            for group in groups:
                texts = [items[i][0] for i in group]
                line_rects = [items[i][1] for i in group]
                spans = [s for i in group for s in items[i][2]]
                span_lines = [items[i][3] for i in group]
                text = _join_lines(texts)
                rect = pymupdf.Rect()
                for r in line_rects:
                    rect |= r
                # Fragments inside a cell are kept for now: whether they are a formula is decided after the
                # pieces of the cell are merged
                in_cell = cell_containing(rect, tables_by_page[pno]) is not None
                # The few connecting words in a centered single-line expression (a definition, if and only if)
                # do not form a sentence, so the original text is kept:
                # the model often changes the mathematical meaning when it rearranges such lines
                displayed = len(group) == 1 and _displayed(rect, spans, margins)
                fraction = group[0] in formula_rows
                # The page numbers after an index entry do not count toward the share of letters:
                # perplexity, 27, 76, 88, 203 is one entry, not a row of numbers
                judged = _INDEX_REF_RE.sub("", text) if index_page else text
                # An index entry ending with page numbers is still an entry however few letters it has (F0, 340);
                # it stays a block with the original text and moves along when the index is rearranged
                entry = index_page and judged != text and bool(judged.strip()) and not fraction
                if (
                    not in_cell
                    and not entry
                    and (fraction or _untranslatable(spans, judged, _DISPLAY_PROSE if displayed else 3, displayed))
                    and not _wraps_from_above(blocks, pno, rect, text, row_ends, frames[pno], index_page)
                ):
                    # Only punctuation fragments are donated. Formula lines and lines with few letters are not
                    # sent for translation, and their rectangles must not be donated either:
                    # donated rectangles are erased, and these lines have no placeholder and are not redrawn.
                    if NO_TEXT_RE.match(text) and blocks and blocks[-1].page == pno:
                        _donate_punct_rects(blocks[-1], line_rects)
                    else:
                        kept_rows.append(rect)
                    continue
                # The size comes from the longest span. Color and weight come from the majority by character
                # count: in books with one span per word, the longest piece is often the caption label Figure
                # 10.3 in white on black, and the whole caption would be written in white
                main = _main_span([items[i][2] for i in group])
                main_size = _block_size(main, span_lines)
                color = _by_chars(spans, lambda s: s["color"])
                bold = _by_chars(spans, lambda s: bool(s["flags"] & 16) or "Bold" in s["font"])
                if in_cell:
                    pending_cells.add(len(blocks))
                blocks.append(
                    TextBlock(
                        page=pno,
                        rect=rect,
                        line_rects=line_rects,
                        text=text,
                        size=round(main_size, 1),
                        color=f"#{color:06x}",
                        bold=bold,
                        span_lines=span_lines,
                        side_note=side_note,
                        leader=leaders.get(group[-1]),
                        line_gap=line_gap,
                        body_size=body_size,
                        entry=group[-1] in entry_ends,
                        grid=group[-1] in cells,
                        index=index_page,
                    )
                )
        absorbed.update(id(block) for block in blocks[page_start:] if _inside_kept_formula(block, kept_rows))
    split: list[TextBlock] = []
    unchecked: set[int] = set()
    for index, block in enumerate(blocks):
        if id(block) in absorbed:
            continue
        tables = tables_by_page[block.page] if block.page < len(tables_by_page) else ()
        parts = _split_table_blocks([block], tables)
        if index in pending_cells:
            unchecked.update(id(part) for part in parts)
        split.extend(parts)
    split = _merge_cells(split, unchecked)
    # When the column gaps are clear, sort into reading order first. Continuation within a page looks only at
    # adjacent blocks, and another column in between would keep a block from joining the next line of its
    # own column.
    ordered: list[TextBlock] = []
    by_page: dict[int, list[TextBlock]] = {}
    geometries: dict[int, PageGeometry] = {}
    for block in split:
        by_page.setdefault(block.page, []).append(block)
    for pno in sorted(by_page):
        page_blocks = by_page[pno]
        geo = PageGeometry.from_page(doc[pno], [block.rect for block in page_blocks], scan=scans[pno])
        geometries[pno] = geo
        ordered.extend(reading_order(page_blocks, geo))
    lines_first = _merge_caption_fragments(_merge_visual_lines(ordered, geometries))
    # Running headers and footers are not joined to half sentences of body text: a book's running header is
    # set on the right without a period and looks like the second half of a line;
    # a paper's running header stops at et al. and looks like an unfinished sentence.
    # A running header often lies below the narrow strip at the top of the page and above the text frame, so it
    # is only recognized with the text frame
    margin_geometries = [
        PageGeometry.from_page(doc[pno], [block.rect for block in by_page.get(pno, [])], frames[pno], scans[pno])
        for pno in range(doc.page_count)
    ]
    heads = {id(block) for block, edge in zip(lines_first, margin_skips(lines_first, margin_geometries)) if edge}
    merged = _merge_continuations(lines_first, _line_lists(lines_first, frames), heads)
    edges = {id(block) for block, edge in zip(merged, margin_skips(merged, margin_geometries)) if edge}
    merged = _merge_same_line_sentences(_merge_line_tails(merged, edges), edges)
    return _split_affiliation_tails(merged), frames, scans


_DISPLAY_PROSE = 8


# Measuring the line spacing needs at least this many pairs of adjacent body lines
_LINE_GAP_PAIRS = 20


def _sized_lines(pages: list[dict]) -> list[list[tuple[pymupdf.Rect, float, int]]]:
    """The horizontal text lines of each page, with the size holding most characters and the character count,
    from top to bottom."""
    rows: list[list[tuple[pymupdf.Rect, float, int]]] = []
    for data in pages:
        row: list[tuple[pymupdf.Rect, float, int]] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [span for span in line["spans"] if span["text"].strip()]
                if abs(line["dir"][1]) > 0.01 or not spans:
                    continue
                size = round(float(_by_chars(spans, lambda span: span["size"])), 1)
                row.append((pymupdf.Rect(line["bbox"]), size, sum(len(span["text"].strip()) for span in spans)))
        rows.append(sorted(row, key=lambda item: (item[0].y0, item[0].x0)))
    return rows


def _body_size(rows: list[list[tuple[pymupdf.Rect, float, int]]]) -> float:
    """Body font size: the size with the most characters. 0 when there are no characters."""
    chars: dict[float, int] = {}
    for row in rows:
        for _box, size, count in row:
            chars[size] = chars.get(size, 0) + count
    return max(chars, key=chars.__getitem__, default=0.0)


def _line_gap(pages: list[dict]) -> float:
    """The gap between adjacent body lines across the document, in units of font size, taking the mode. Line
    box height varies with the font's ascender and descender: for two lines set equally tight, some fonts
    measure a gap of 0.2 em and others an overlap of 0.2 em.
    This amount is subtracted first when deciding whether the next line follows directly."""
    rows = _sized_lines(pages)
    body = _body_size(rows)
    if body <= 0:
        return _LINE_GAP
    counts: dict[float, int] = {}
    for row in rows:
        lines = [box for box, size, _count in row if abs(size - body) <= 0.3]
        for index, box in enumerate(lines):
            # The next line right below in the same column: overlapping horizontally by more than half, not on
            # the same row as this line
            for below in lines[index + 1 :]:
                if below.y0 > box.y1 + body * 3:
                    break
                if below.y0 < box.y1 - body * 0.3:
                    continue
                if min(box.x1, below.x1) - max(box.x0, below.x0) < min(box.width, below.width) * 0.5:
                    continue
                gap = round((below.y0 - box.y1) / body * 20) / 20
                counts[gap] = counts.get(gap, 0) + 1
                break
    if sum(counts.values()) < _LINE_GAP_PAIRS:
        return _LINE_GAP
    return max(counts, key=counts.__getitem__)


def _left_margins(data: dict) -> list[float]:
    """Left edges of body lines on this page (one per column): at least three long lines start there."""
    starts: list[float] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            if len("".join(span["text"] for span in line["spans"]).strip()) >= 40:
                starts.append(round(line["bbox"][0]))
    margins: list[float] = []
    for x in sorted(set(starts)):
        if sum(1 for start in starts if abs(start - x) <= 2) >= 3 and all(abs(x - m) > 2 for m in margins):
            margins.append(x)
    return margins


def _displayed(rect: pymupdf.Rect, spans: list[dict], margins: list[float]) -> bool:
    """An expression on a line of its own, more than two ems from the left edge of its column. An expression
    continuing the line above within a paragraph starts at the left edge."""
    left = [m for m in margins if m <= rect.x0 + 2]
    if not left:
        return False
    size = max((float(span.get("size") or 0) for span in spans), default=10.0)
    return rect.x0 - max(left) > size * 2.5


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


def _main_span(lines: list[list[dict]]) -> dict:
    """The piece that sets the block's size and color: the longest body-font piece on the baseline. In lines
    with much math the longest piece is often a subscript (the model of d_model, set small in the body font);
    a size taken from it is half the body size and does not join the line above."""
    kept: list[dict] = []
    for spans in lines:
        weights: Counter[int] = Counter()
        for span in spans:
            weights[round(float(span["origin"][1]))] += len(span["text"].strip())
        base = weights.most_common(1)[0][0] if weights else None
        kept += [
            span for span in spans if base is None or abs(float(span["origin"][1]) - base) <= float(span["size"]) * 0.2
        ]
    words = [span for span in kept if not _span_is_math(span) and span["text"].strip()]
    pool = words or kept or [span for spans in lines for span in spans]
    return max(pool, key=lambda span: len(span["text"]))


def _inside_kept_formula(block: TextBlock, kept: Sequence[pymupdf.Rect]) -> bool:
    """A one-line fragment with no word of three or more letters, mostly inside the box of a math line that
    keeps the original text: it is a piece of the expression (the denominator dx of a fraction) and is not
    sent for translation on its own. An annotation on a brace below an expression (prior) is a word and is
    translated."""
    if len(block.line_rects) != 1 or re.search(r"[A-Za-z]{3,}", block.text) or CJK_RE.search(block.text):
        return False
    area = block.rect.get_area()
    return area > 0 and any((block.rect & row).get_area() >= area * 0.6 for row in kept)


def _index_refs_tail(block: TextBlock, rect: pymupdf.Rect) -> bool:
    """rect is a line of page numbers wrapped from the last line of block: right below the last line, starting
    between its left and right edges."""
    last = block.line_rects[-1]
    return (
        -block.size * 0.3 <= rect.y0 - last.y1 <= block.size * 0.5 and last.x0 - block.size * 0.5 <= rect.x0 < last.x1
    )


def _wraps_from_above(
    blocks: list[TextBlock],
    pno: int,
    rect: pymupdf.Rect,
    text: str,
    rows: Sequence[pymupdf.Rect],
    frame: TextFrame | None,
    index_page: bool = False,
) -> bool:
    """A math line that starts with a lowercase body word and continues from the left edge of the body line
    above (and W^O ∈ ℝ^{hd_v×d_model}.) is wrapped from the sentence above, not a display expression: the line
    above runs to the right edge of the column without finishing a sentence, and this line sits right below
    it.
    When the line above ends with half a word broken by a hyphen, this line is the second half of that word. On
    an index page, a line indented deeper than the line above is a continuation of the previous entry (pipe in
    regular expressions | as, 51).
    The last line of a centered caption (significant (t-test with p ≤ 0.01).) shares the center line of the
    line above.
    Stacked expression lines (e_t = Ex_t, h_t = g(...)) do not reach the right edge.
    An index entry stops at a comma and its page numbers wrap to the next line, in the next block: a line of
    only page numbers right below is joined as well."""
    page_blocks = [block for block in blocks if block.page == pno and block.line_rects]
    if index_page and _INDEX_REFS_ONLY_RE.match(text.strip()):
        return any(block.text.rstrip().endswith(",") and _index_refs_tail(block, rect) for block in page_blocks[-1:])
    worded = re.match(r"[a-z]{2,}\b", text.lstrip()) is not None
    for block in reversed(page_blocks):
        last = block.line_rects[-1]
        if rect.y0 - last.y1 > 0.6 * block.size:
            # Blocks are in reading order; earlier ones are all higher up or in another column
            continue
        # What wraps is one line: with more lines stacked below, it is a separately set expression. The
        # stacked numerator and denominator of an inline fraction are taller than one line
        if (
            rect.height > block.size * (1.6 if worded else 2.5)
            or rect.y0 - last.y1 < -0.5 * block.size
            or _ends_sentence(block.text)
        ):
            continue
        # The left-aligned line above runs to the right edge of the column. In a centered paragraph each line
        # is a block, and a line above as wide as the line above it is a full line
        left = (
            abs(block.rect.x0 - rect.x0) <= block.size * 0.5
            and last.x1 >= _column_right(block.rect.x0, block.size, rows, frame) - block.size
        )
        if not worded:
            # A line not starting with a word is accepted only as an inline fraction with stacked numerator
            # and denominator, left-aligned and right below the line above (the odds ratio | p/(1−p):).
            # A display expression has space above it. Numbers and symbols one line tall belong to tables and
            # figures
            if left and rect.height > block.size * 1.6 and rect.y0 - last.y1 <= 0.3 * block.size:
                return True
            continue
        centered = abs((last.x0 + last.x1) / 2 - (rect.x0 + rect.x1) / 2) <= block.size and any(
            0 <= last.y0 - line.y1 <= block.size
            and abs(line.x0 - last.x0) <= block.size
            and abs(line.x1 - last.x1) <= block.size
            for other in page_blocks
            for line in other.line_rects
            if line is not last
        )
        # The line above stops at half a word broken by a hyphen (expres- | sions, 50, 65): it continues
        # whatever the indent.
        # A continuation line of an index entry has a hanging indent, and the remaining half word with its page
        # numbers has few letters and looks like a line of page numbers
        broken = (
            re.search(r"[A-Za-z]-$", block.text.rstrip()) is not None
            and block.rect.x0 - block.size * 0.5 <= rect.x0 < last.x1
        )
        hanging = index_page and last.x0 + block.size * 0.5 < rect.x0 < last.x1
        if left or centered or broken or hanging:
            return True
    return False


def _column_right(x0: float, size: float, rows: Sequence[pymupdf.Rect], frame: TextFrame | None) -> float:
    """Right edge of the column of a block starting at x0: 40% or more of the lines on this page starting at
    the same left edge end at the same x (justified).
    Ragged lines such as indexes and code give no such edge, and the right edge of the text frame is used."""
    edge = _voted_right(x0, size, rows)
    if edge is not None:
        return edge
    return frame.x1 if frame is not None else float("inf")


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


def _untranslatable(spans: list[dict], text: str, min_prose: int = 3, displayed: bool = False) -> bool:
    """Lines that are pure symbols, pure formula, or have almost no letters are not sent for translation and
    keep the original text.

    Only "pure formula lines" are skipped: many math characters and almost no body words. Three body words
    outside the formula make a sentence (If ... then ... Otherwise ...), which is translated, with the formula
    going through a placeholder. Letters inside a formula are not words:
    letters recognized from math symbols in bitmap fonts are garbage.

    A line set on its own with an operator and not a single body word is also an expression. In books that set
    variables in body italics (Q = H_dec W^Q, ŷ = argmax_{w∈V} P(w|w_<t)), formula characters cannot be
    recognized by font.
    """
    if NO_TEXT_RE.match(text):
        return True
    if displayed and _OPERATOR_RE.search(text) and not _prose_words(spans, text):
        return True
    total = sum(len(span["text"]) for span in spans)
    # Blank spans have been filtered out, so join with spaces, or It and follows would become one word
    prose = _prose_units(" ".join(span["text"] for span in spans if not _span_is_math(span)))
    if _math_chars(spans) > total * 0.3 and prose < min_prose:
        return True
    # Enough body words make a sentence (Therefore, ... It follows), translated however much formula it has and
    # however low its share of letters.
    # The few-letters rule is meant for labels and lines of numbers
    if prose >= max(min_prose, 3):
        return False
    letters = sum(c.isalpha() for c in text)
    return (letters < 2 or letters < len(text) * 0.4) and not _numbered_list_item(text)


_MATH_NAMES = frozenset(_OPERATOR_NAMES.split("|")) | {"argmax", "argmin", "softmax", "tanh", "sigmoid"}


def _prose_words(spans: list[dict], text: str) -> list[str]:
    """Words in the line separated by spaces on both sides and written in a body font (with lowercase letters,
    two or more letters).
    Words written together with symbols (H_dec, argmax_{char∈Alphabet}), all-capital matrix names (HU) and
    operator names (argmax, log) do not count."""
    upright = set()
    for span in spans:
        if not _span_is_math(span):
            upright.update(re.findall(r"[A-Za-z]{2,}", span["text"]))
    words = []
    for token in text.split():
        word = token.strip(",.;:!?\"'“”‘’()[]{}")
        if (
            len(word) >= 2
            and word.isascii()
            and word.isalpha()
            and word in upright
            and any(c.islower() for c in word)
            and word not in _MATH_NAMES
        ):
            words.append(word)
    return words


def _merge_cells(blocks: list[TextBlock], unchecked: set[int]) -> list[TextBlock]:
    """Merge the blocks in one cell into one block, ordered top to bottom and left to right. A cell is one
    translation unit.

    PyMuPDF splits the text of a cell into several blocks (line numbers one size smaller, formulas in a block
    of their own). Sent for translation separately, each half sentence has too few body words and is taken as
    a formula, leaving the English. Blocks in unchecked have not been tested for formulas yet; the test runs
    after merging.
    A cell is one complete sentence and is translated when it has a single body word (for k = 1 ... K do);
    cells of pure formula or pure numbers keep the original text.
    """
    cells: dict[tuple, list[TextBlock]] = {}
    for block in blocks:
        if block.clip is not None:
            cells.setdefault((block.page, block.clip), []).append(block)
    out: list[TextBlock] = []
    done: set[tuple] = set()
    for block in blocks:
        key = (block.page, block.clip)
        if block.clip is None:
            if id(block) in unchecked and _untranslatable(_block_spans(block), block.text):
                continue
            out.append(block)
            continue
        if key in done:
            continue
        done.add(key)
        parts = [
            part
            for part in _reading_sorted(cells[key])
            if not (id(part) in unchecked and _displayed_in_cell(part, cells[key]))
        ]
        for group in _cell_groups(parts):
            merged = group[0] if len(group) == 1 else _joined_cell(group)
            if any(id(part) in unchecked for part in group) and _untranslatable(_block_spans(merged), merged.text, 1):
                continue
            out.append(merged)
    return out


def _cell_groups(parts: list[TextBlock]) -> list[list[TextBlock]]:
    """Adjacent pieces in a cell are one sentence. A piece two ems away, or a blank line apart, is another
    sentence: when the boxes of a flowchart are taken as cells, the node labels in a box must not be merged
    into one sentence."""
    groups: list[list[TextBlock]] = []
    for part in parts:
        near = [group for group in groups if any(_close_in_cell(part, other) for other in group)]
        if not near:
            groups.append([part])
            continue
        near[0].append(part)
        for group in near[1:]:
            near[0].extend(group)
            groups.remove(group)
    return [_reading_sorted(group) if len(group) > 1 else group for group in groups]


def _close_in_cell(a: TextBlock, b: TextBlock) -> bool:
    size = max(a.size, b.size)
    dx = max(a.rect.x0 - b.rect.x1, b.rect.x0 - a.rect.x1, 0)
    dy = max(a.rect.y0 - b.rect.y1, b.rect.y0 - a.rect.y1, 0)
    return dx <= size * 2 and dy <= size


def _displayed_in_cell(part: TextBlock, parts: list[TextBlock]) -> bool:
    """A self-contained expression in a cell: on a line of its own and indented more than two ems from the
    text in the cell (a centered formula under an algorithm step), or more than two ems from the text on the
    same row (the expression of an algorithm step and the comment to its right).
    It keeps the original text and is not merged into the sentence: once merged, letters such as E and Var are
    translated as words, and the display formula is squeezed to one line height."""
    others = [other for other in parts if other is not part]
    if not others or not _untranslatable(_block_spans(part), part.text):
        return False
    beside = [
        other
        for other in others
        if min(other.rect.y1, part.rect.y1) - max(other.rect.y0, part.rect.y0)
        > min(other.rect.height, part.rect.height) * 0.35
    ]
    if beside:
        # A line number is not a sentence either, but it is not an expression, so it stays in this step
        if not any(_span_is_math(span) for span in _block_spans(part)):
            return False
        texts = [other for other in beside if not _untranslatable(_block_spans(other), other.text)]
        return bool(texts) and all(
            max(other.rect.x0 - part.rect.x1, part.rect.x0 - other.rect.x1) > part.size * 2 for other in texts
        )
    left = min(other.rect.x0 for other in others)
    return part.rect.x0 - left > part.size * 2.5


def _reading_sorted(parts: list[TextBlock]) -> list[TextBlock]:
    """Pieces on the same row are ordered left to right, different rows top to bottom. The box of a smaller
    line number has a slightly lower top, so box tops cannot be used for ordering."""
    rows: list[list[TextBlock]] = []
    for part in sorted(parts, key=lambda item: (item.rect.y0 + item.rect.y1) / 2):
        center = (part.rect.y0 + part.rect.y1) / 2
        if rows:
            last = rows[-1][-1]
            if abs(center - (last.rect.y0 + last.rect.y1) / 2) <= max(part.size, last.size) * 0.5:
                rows[-1].append(part)
                continue
        rows.append([part])
    return [part for row in rows for part in sorted(row, key=lambda item: item.rect.x0)]


def _joined_cell(parts: list[TextBlock]) -> TextBlock:
    main = max(parts, key=lambda part: len(part.text))
    rect = pymupdf.Rect()
    for part in parts:
        rect |= part.rect
    has_spans = all(part.span_lines for part in parts)
    return replace(
        main,
        rect=rect,
        line_rects=[line for part in parts for line in part.line_rects],
        text=_join_lines([part.text for part in parts]),
        span_lines=[line for part in parts for line in part.span_lines] if has_spans else None,
        runs=[],
    )


def _block_spans(block: TextBlock) -> list[dict]:
    return [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]


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


def _plain_words(block: TextBlock) -> int:
    """Number of body words outside formulas."""
    return _prose_units(
        " ".join(
            str(span.get("text") or "") for line in block.span_lines or [] for span in line if not _span_is_math(span)
        )
    )


def _line_text(span_lines: list[list[dict]], index: int) -> str:
    return "".join(str(span.get("text") or "") for span in span_lines[index])


def _block_slice(block: TextBlock, start: int, end: int) -> TextBlock:
    lines = block.line_rects[start:end]
    spans = block.span_lines[start:end] if block.span_lines else None
    texts = [_line_text(spans, i) for i in range(len(spans or []))]
    rect = pymupdf.Rect()
    for line in lines:
        rect |= line
    return TextBlock(
        page=block.page,
        rect=rect,
        line_rects=list(lines),
        text=_join_lines(texts) if texts else block.text,
        size=block.size,
        color=block.color,
        bold=block.bold,
        span_lines=list(spans) if spans is not None else None,
        clip=block.clip,
        line_gap=block.line_gap,
        body_size=block.body_size,
    )


def _affiliation_neighbor(blocks: list[TextBlock], block: TextBlock, line: pymupdf.Rect) -> bool:
    """Whether a short affiliation line sits beside this line on its right."""
    for other in blocks:
        if other is block or other.page != block.page or other.clip is not None:
            continue
        if other.rect.height > 32 or len(other.line_rects) > 2:
            continue
        overlap = min(line.y1, other.rect.y1) - max(line.y0, other.rect.y0)
        if overlap < min(line.height, other.rect.height) * 0.5:
            continue
        gap = other.rect.x0 - line.x1
        if 0 <= gap < 40 and _AFFIL_MARK_RE.match(other.text.lstrip()):
            return True
    return False


def _split_affiliation_tails(blocks: list[TextBlock]) -> list[TextBlock]:
    """When the affiliation at the end of an author block sits beside other affiliations, split it into a
    separate block. Otherwise the translation would cover them."""
    out: list[TextBlock] = []
    for block in blocks:
        spans = block.span_lines
        if spans is None or block.clip is not None or len(block.line_rects) < 3:
            out.append(block)
            continue
        if len(spans) != len(block.line_rects):
            out.append(block)
            continue
        cut: int | None = None
        for index, line in enumerate(block.line_rects):
            if index == 0 or line.width > block.rect.width * 0.5:
                continue
            if not _AFFIL_MARK_RE.match(_line_text(spans, index).strip()):
                continue
            if _affiliation_neighbor(blocks, block, line):
                cut = index
                break
        if cut is None:
            out.append(block)
            continue
        out.append(_block_slice(block, 0, cut))
        out.append(_block_slice(block, cut, len(block.line_rects)))
    return out


def _absorb_block(p: TextBlock, b: TextBlock):
    p.text = _join_lines([p.text, b.text])
    p.rect |= b.rect
    p.line_rects.extend(b.line_rects)
    if p.span_lines and b.span_lines:
        p.span_lines.extend(b.span_lines)
    else:
        p.span_lines = None
    if len(b.text) > len(p.text) - len(b.text):  # Style follows the larger fragment
        p.size, p.color, p.bold = b.size, b.color, b.bold
    # The merged block ends at b's last line: when b is the last line of an index entry, so is the merged block
    p.entry = b.entry


def _cjk_wrap(prev: TextBlock, nxt: TextBlock) -> bool:
    """Chinese line breaks have no letter case and are not counted in words. A new paragraph has an indent of
    about 2 em, and a short label is not wide enough."""
    head = nxt.text.lstrip()[:1]
    if not head or CJK_RE.search(head) is None:
        return False
    if _ends_sentence(prev.text):
        return False
    if abs(nxt.size - prev.size) > prev.size * 0.2:
        return False
    if abs(nxt.rect.x0 - prev.rect.x0) > prev.size * 0.6:
        return False
    return len(CJK_RE.findall(prev.text)) >= 8 or prev.rect.width >= prev.size * 12


# A list with one sentence per line: at least this many lines
_LIST_RUN = 3


def _line_lists(blocks: list[TextBlock], frames: Sequence[TextFrame | None]) -> set[int]:
    """Blocks in a list with one sentence per line (examples on a slide). Several blocks, one line each,
    aligned on the left with equal line spacing, where at more than half of the line breaks the first word of
    the next line would still fit after the line above, or every line starts with the same word: set line by
    line, not wrapped from one paragraph.
    A paragraph wraps line by line, and a word that fits on the line above is not moved to the next. Lines set
    around a figure end at the figure's edge; enough lines end at the same x that the column's right edge is
    taken at the figure's edge (see _column_right)."""
    rows_of: dict[int, list[pymupdf.Rect]] = {}
    for block in blocks:
        rows_of.setdefault(block.page, []).extend(block.line_rects or [block.rect])
    listed: set[int] = set()
    run: list[TextBlock] = []

    def close() -> None:
        if len(run) < _LIST_RUN:
            return
        frame = frames[run[0].page] if run[0].page < len(frames) else None
        if frame is None:
            return
        rows = rows_of[run[0].page]
        if _same_opening([block.text for block in run]) or _breaks_early(
            [(block.rect, block.text, block.size) for block in run],
            lambda rect, size: _column_right(rect.x0, size, rows, frame),
        ):
            listed.update(id(block) for block in run)

    for block in blocks:
        single = bool(block.line_rects) and len(_visual_lines(block)) == 1
        if run and single:
            last = run[-1]
            pitch = block.rect.y0 - last.rect.y0
            first_pitch = run[1].rect.y0 - run[0].rect.y0 if len(run) >= 2 else pitch
            if (
                block.page == last.page
                and abs(block.rect.x0 - last.rect.x0) <= last.size * 0.5
                and abs(block.size - last.size) <= last.size * 0.05
                and 0 < pitch <= last.size * 2.0
                and abs(pitch - first_pitch) <= last.size * 0.2
            ):
                run.append(block)
                continue
        close()
        run = [block] if single else []
    close()
    return listed


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


def _merge_continuations(
    blocks: list[TextBlock],
    listed: Collection[int] = frozenset(),
    margins: Collection[int] = frozenset(),
) -> list[TextBlock]:
    """Merge continuation lines of a paragraph that formulas or fragments broke apart within a page: when the
    line above has no sentence-final punctuation, the line below starts in lowercase (or with a closing
    bracket), the left margin is the same and the line spacing is normal, they are two lines of one paragraph
    split apart. Translated separately they produce half translations, and rendering forces a line break at
    the cut ("（如 | 序列）"). Chinese line breaks are joined the same way when the left edges align and the
    line above is long enough.
    English line breaks before a capital stay separate, since they cannot be told from the next sentence;
    the exception is a line above that stops at a word such as and, of or the.
    When a paragraph is set around a figure, the caption falls between the two halves in reading order. When
    that block lies below the first half and beside the second, it is skipped and the block before it is
    compared. margins are running headers and footers, which are not joined to the blocks above and below."""
    out: list[TextBlock] = []
    indents = _indents_paragraphs(blocks)
    # In documents with paragraph indents, a line starting flush below a sentence end is a caption, an example
    # or the half sentence after an expression, so no paragraph spacing can be measured
    space = 0.0 if indents else _paragraph_space(blocks)
    measure = _body_measure(blocks) if indents or space else 0.0
    for index, b in enumerate(blocks):
        hosts = out[-1:]
        if len(out) >= 2 and _beside_wrap(out[-1], out[-2], b):
            hosts.append(out[-2])
        for p in hosts:
            first = b.text[:1]
            # Fragments of a large heading ('...with' + 'Recurrent Neural Networks'): the second line of the
            # wrap often starts with a capital and has no punctuation. Merge only when the left edges of the two
            # lines clearly shift (centered heading lines): left-aligned large blocks (subheadings, byline
            # lines) stay separate.
            # A byline sits almost a line below the title and its left edge also shifts, but it is not the
            # title's next line.
            # A line with a list marker is the next item of a slide, whose left edge changes with the level; it
            # is not the title's next line either.
            # The two lines of a wrapped heading always overlap horizontally; one offset sideways is a label
            # beside it.
            # Large is measured against this document's body size: slide body text is around thirty points,
            # and indented subitems are not a wrapped heading.
            # Both lines of a wrapped heading are the same size. One size larger right after is the next
            # heading level (8.5 Summary under the 8.4.4 subsection name)
            title_frag = (
                p.size >= max(12, p.body_size * 1.1)
                and p.size * 0.7 <= b.size <= p.size * 1.1
                and 0 <= b.rect.y0 - p.rect.y1 < 0.75 * p.size
                and abs(b.rect.x0 - p.rect.x0) > p.size * 0.5
                and min(p.rect.x1, b.rect.x1) > max(p.rect.x0, b.rect.x0)
                and not _list_item(b.text)
                and not _ends_sentence(p.text)
                and not _ends_sentence(b.text)
            )
            y_close = -0.9 * p.size <= b.rect.y0 - p.rect.y1 - p.line_gap * p.size < 1.4 * p.size
            # A hanging indent is about 2-3 em. With a tight 2 em limit, continuation lines of a feature list are
            # split into two paragraphs and the short line is then shrunk on its own. A jump to the left still
            # uses 2 em, so that another column is not joined.
            x_jump = b.rect.x0 - p.rect.x0
            x_close = abs(x_jump) < 2 * p.size or 0 < x_jump <= max(4 * p.size, 36)
            # The last line of a centered paragraph (a table note) is short, indented on both sides, shares the
            # center line of the line above, and sits right below it at normal line spacing.
            # A centered formula line below a paragraph has more space around it, and spans more than one line
            # or is no sentence. A last line of only one or two words (the learning. of a caption) that is all
            # body words and ends with a period is not a formula either
            last_y1 = p.line_rects[-1].y1 if p.line_rects else p.rect.y1
            words = b.text.split()
            centered_tail = (
                len(_visual_lines(b)) == 1
                and b.rect.y0 - last_y1 - p.line_gap * p.size <= 0.3 * p.size
                and b.rect.x0 > p.rect.x0 + p.size
                and b.rect.x1 < p.rect.x1 - p.size
                and abs((b.rect.x0 + b.rect.x1) - (p.rect.x0 + p.rect.x1)) / 2 <= p.size
                and not _list_item(b.text)
                and (_prose_units(b.text) >= 3 or (_ends_sentence(b.text) and _prose_units(b.text) == len(words)))
            )
            # When the lines on both sides of a colon at a line end reach the right edge, it is a wrap inside a
            # sentence ("is called interpolation: | computing ...").
            # A next line that is not full is an example or list introduced by the colon
            last_x1 = p.line_rects[-1].x1 if p.line_rects else p.rect.x1
            next_x1 = b.line_rects[0].x1 if b.line_rects else b.rect.x1
            colon_wrap = (
                p.text.rstrip().endswith(":")
                and abs(last_x1 - next_x1) <= p.size
                and last_x1 >= max(p.rect.x1, b.rect.x1) - p.size
            )
            # Ending with f(x) counts as a sentence end so that a where below a display expression is not joined.
            # When the last line runs from the next line's left edge to the right edge and the next line starts
            # with a body word, it is a wrap inside a sentence and the expression just falls at the line end
            call_wrap = _ends_in_call(p.text.rstrip()) and _opens_with_word(b) and _wraps_into(p, b)
            # A line that starts with an inline expression followed by body text (Adam with | 1/√t decay on its
            # stepsize) is a wrap inside a sentence: it starts at the column edge, or the expression fragment at
            # its start became a block of its own while the rest runs to the right edge and the next line
            # returns to the column edge.
            # A display expression is indented, does not reach the right edge and has few body words; a footnote
            # starting with a superscript number is one size smaller
            first_x0 = b.line_rects[0].x0 if b.line_rects else b.rect.x0
            rows = sorted(_visual_lines(b), key=lambda row: row[0].y0) if b.line_rects else []
            math_wrap = (
                _opens_with_math(b)
                and _plain_words(b) >= _MATH_WRAP_WORDS
                and abs(b.size - p.size) <= p.size * 0.1
                and (
                    abs(first_x0 - p.rect.x0) <= p.size * 0.5
                    or (
                        len(rows) >= 2
                        and abs(max(rect.x1 for rect in rows[0]) - p.rect.x1) <= p.size * 0.5
                        and abs(min(rect.x0 for rect in rows[1]) - p.rect.x0) <= p.size * 0.5
                    )
                )
            )
            # An inline fraction at the end of a sentence wraps to the start of the next line with no body words
            # after it (the odds ratio | p/(1−p):): the stacked numerator and denominator are taller than one
            # line, start at the column edge, and sit right below a line above that runs to the right edge. A
            # display expression is indented and has space above it
            formula_tail = (
                bool(b.line_rects)
                and _plain_words(b) == 0
                and b.rect.height > p.size * 1.6
                and abs(first_x0 - p.rect.x0) <= p.size * 0.5
                and last_x1 >= p.rect.x1 - p.size
                and b.rect.y0 - last_y1 <= 0.3 * p.size
            )
            # An abbreviation inside a sentence (Brants et al. | (2007)) is not a sentence end. A continuation
            # line starting with an opening parenthesis is a citation year or a parenthetical when it is no
            # further right than the leftmost line of the paragraph above (a paragraph with a first-line
            # indent), shares a left edge with one of its lines (continuation of an entry with a hanging indent),
            # or is centered below a centered paragraph.
            # A label such as (a) is the next item, and (2.11) is the number of an example or formula;
            # an indented one is a display expression (a LUNAR logical form), and one offset to the right is a
            # column header such as (Acc).
            # When an entry starting with a bold label has only one line, there is no continuation line to
            # compare with, and a line with a hanging indent of one or two ems is also its continuation
            label_hang = (
                bool(p.span_lines)
                and 0 < b.rect.x0 - p.rect.x0 <= 3 * p.size
                and _label_lead([span for span in p.span_lines[0] if str(span.get("text") or "").strip()])
            )
            # The bracket is still open and the next line closes it in its first sentence
            # ((see e.g. | Ghosal et al., 2000), (Chérief- | Abdellatif ...))
            bracketed = _unclosed_bracket(p.text) and _closes_first(b.text)
            open_paren = (
                first == "("
                and not _list_item(b.text)
                and not _EQUATION_TAG_RE.match(b.text)
                and (
                    centered_tail
                    or label_hang
                    or b.rect.x0 <= p.rect.x0 + p.size * 0.5
                    or any(abs(b.rect.x0 - rect.x0) <= p.size * 0.5 for rect in p.line_rects)
                )
            )
            # When the line above stops at and or of, the next line continues even when it starts with a capital.
            # Not when the next line is another list item (...; and | • ...) or a footnote one size smaller
            same_size = abs(b.size - p.size) <= p.size * 0.05
            dangling = _dangling(p.text) and not _list_item(b.text) and not _opens_with_math(b) and same_size
            # The last line of a centered paragraph is joined even when it starts with a capital (Rates., MATH
            # dataset.); the smaller attribution under an epigraph is not
            centered_end = centered_tail and same_size
            # A centered paragraph (a list of authors) that wraps at a comma continues even when the next line
            # starts with a capital. The whole stack shares one center line and the font stays the same
            centered_comma = (
                same_size
                and p.text.rstrip().endswith((",", "，"))
                and not _list_item(b.text)
                and bool(p.line_rects)
                and bool(b.line_rects)
                and _centered_rows(_row_boxes(p.line_rects + b.line_rects), p.size)
            )
            # The indented first line of a paragraph shared a block with a margin note and was split off on its
            # own: when this line runs to the right edge without sentence-final punctuation and the next line is
            # one indent further left, it is a wrap inside a sentence, joined even when the next line starts
            # with a capital (a proper noun)
            full_wrap = (
                same_size and not _list_item(b.text) and len(_visual_lines(p)) == 1 and _wraps_into(p, b, indented=True)
            )
            # A first line after a list item that starts even further left is body text after the list has
            # ended. Continuation lines align with the item text or have a hanging indent and never move left
            list_end = _list_item(p.text) and first_x0 < p.rect.x0 - p.size * 0.5
            # The block above stops at half a broken word (retrieval-augmented genera- | tion, 267), and the next
            # line finishes the word in lowercase.
            # A digit after the hyphen is a model name broken across lines (VE- | 8. Both ..., GPT- | 4), not a
            # number label on the next line.
            # Joined even with only two words: headings and labels do not stop at a hyphen
            # A half line after an inline label or inline expression (Backbones. | Models are ... at temper-,
            # ... This no longer agrees with the MLE under) starts to their right, and the next line returns to
            # the column edge. When this line and the next block share a right edge, they are in the same column
            row_tail = len(_visual_lines(p)) == 1 and abs(p.rect.x1 - b.rect.x1) <= p.size and first_x0 < p.rect.x0
            word_cut = (
                same_size
                and y_close
                and re.search(r"[A-Za-z]{2}-$", p.text.rstrip()) is not None
                and (first.islower() or first.isdigit())
                and first_x0 < p.rect.x1
                and (p.rect.x0 - p.size <= first_x0 or row_tail)
            )
            row_wrap = same_size and row_tail and first.islower() and not _ends_sentence(p.text)
            # A line set off on its own (the indented example The green witch arrived): the next line returns to
            # the left to continue the body text and is not its continuation.
            # An indented first line cut into a block by a margin note reaches the right edge and is as wide as
            # the next line
            displayed_line = (
                len(_visual_lines(p)) == 1 and p.rect.x0 > first_x0 + p.size and p.rect.x1 < b.rect.x1 - p.size
            )
            # In documents with first-line paragraph indents, a line starting flush below a line that runs to the
            # right edge continues the same paragraph,
            # even when the line above happens to break at a sentence end and the next line starts with a
            # capital (in double-spaced manuscripts each line is a block of its own).
            # When the block above holds only the indented first line of a paragraph (cut off by a margin note),
            # a next line one indent further left is also a continuation.
            # A line starting with a change to bold, italic or another size is an unindented subheading
            # (Execution Accuracy (EX) We..., Lemma 2.3.)
            # "Runs to the right edge" is compared with the width of a full body line; a table cell or a short
            # heading would be full against its own narrow box.
            # Numbered labels (EXAMPLE 2.1, Table 3:) also start anew.
            # The next line must sit right below at normal line spacing: one a line further down is a reference
            # with a hanging indent whose continuation line between became a block of its own.
            # A next block with its own hanging indent below is the next reference; a line above whose pieces
            # are several ems apart is a table row.
            # In documents that separate paragraphs by spacing, a new paragraph has extra space above it: only a
            # line right below at normal line spacing is in the same paragraph.
            # A line starting with a number ((21.15) Ok, we fly ...) is an example or an expression, set flush
            # below the line that introduces it as a block of its own, and is not joined with the body text
            # below either.
            # A number followed by a lowercase start is an expression cited in body text wrapped to the line
            # start ((4.5) and (4.6), we have), joined when it sits right below at normal line spacing;
            # an example starting in lowercase ((24.25) her pay jumped ...) has extra space above it
            row_gap = b.rect.y0 - last_y1 - p.line_gap * p.size
            # The line above runs to the right edge and stops at an equals or plus sign (... and λ2 = | 1/2, what
            # is ...), and the next line starts at the same left edge right below at normal line spacing:
            # it is the other half of the expression, joined even when it starts with a digit. A display
            # expression line does not reach the right edge and has more space below; a line starting with a
            # capitalized word or a footnote number is the next sentence or the next entry
            open_end = (
                same_size
                and _open_operator_end(p.text)
                and row_gap <= 0.3 * p.size
                and abs(first_x0 - p.rect.x0) <= 0.5 * p.size
                and last_x1 >= max(p.rect.x1, b.rect.x1) - p.size
                and bool(p.line_rects)
                and _row_width(_visual_lines(p)[-1]) >= b.rect.width * 0.8
                and not _FOOTNOTE_LEAD_RE.match(b.text.lstrip())
                and not re.match(r"[A-Z][a-z]", b.text.lstrip())
            )
            flush_wrap = (
                same_size
                and ((indents and row_gap <= 0.5 * p.size) or (space and row_gap <= space / 2))
                and not _list_item(b.text)
                and not _LABEL_START_RE.match(b.text)
                and not _numbered_line(b.text)
                and not _numbered_line(p.text)
                and not (_EQUATION_TAG_RE.match(b.text.lstrip()) and row_gap > 0.15 * p.size)
                and not _is_float_text(b.text)
                and not _styled_edge(b, 0)
                and not _styled_edge(p, -1)
                and (_wraps_into(p, b) or full_wrap)
                and bool(p.line_rects)
                and _row_width(_visual_lines(p)[-1]) >= measure * 0.9
                and not _hangs(b, blocks[index + 1] if index + 1 < len(blocks) else None)
                and _row_gap(_visual_lines(p)[-1]) <= 1.5 * p.size
            )
            # The block above stops inside an unclosed code fence (```sql opened and not closed), and the next
            # line continues the code, joined even when it starts with a capital
            in_fence = len(_FENCE_RE.findall(p.text)) % 2 == 1 and same_size
            latin = (
                y_close
                and (x_close or centered_tail or math_wrap or centered_comma or word_cut or row_wrap)
                and not list_end
                and not displayed_line
                and (len(p.text.split()) >= 3 or word_cut)  # 1-2 unpunctuated words: heading or label, not a paragraph
                # A continuation has the size of the line above; small text under a large heading is in a figure
                and p.size * 0.7 <= b.size <= p.size / 0.7
                and (
                    colon_wrap
                    or call_wrap
                    or flush_wrap
                    or _MID_ABBREV_RE.search(p.text.rstrip())
                    or not _ends_sentence(p.text)
                )
                and (
                    first.islower()
                    or first in ".),;%,；，"
                    or open_paren
                    or math_wrap
                    or formula_tail
                    or centered_end
                    or dangling
                    or open_end
                    or centered_comma
                    or in_fence
                    or full_wrap
                    or flush_wrap
                    or bracketed
                    or word_cut
                )
                and not _verbatim_seam(p, b)
                # In documents with first-line paragraph indents, a block whose first line is indented one or
                # two ems more than the lines below and starts with a capitalized word is a new paragraph, not
                # joined even when the block above stops at and (half a sentence before an expression). A line
                # starting with an expression (W 1,∞ ...) does not count
                and not (
                    indents
                    and re.match(r"[A-Z][a-z]", b.text.lstrip()) is not None
                    and not _opens_with_math(b)
                    and not word_cut
                    and _first_line_indented(b)
                )
            )
            # An index entry stops at a comma and its page numbers wrap into the next block (see
            # _wraps_from_above): the entry ends at the page numbers
            if (
                p.index
                and p.page == b.page
                and p.text.rstrip().endswith(",")
                and p.line_rects
                and _INDEX_REFS_ONLY_RE.match(b.text.strip())
                and _index_refs_tail(p, b.rect)
            ):
                _absorb_block(p, b)
                break
            # A contents entry is complete at its leader and page number and is not merged with the entries or
            # part titles above and below
            if (
                p.page == b.page
                and p.clip == b.clip
                and not (p.side_note or b.side_note)
                and id(p) not in margins
                and id(b) not in margins
                and p.leader is None
                and b.leader is None
                and not p.entry
                and not (id(p) in listed and id(b) in listed)
                and (title_frag or latin or (y_close and _cjk_wrap(p, b)))
                and not _restyled_seam(p, b)
                and not _new_voice(p, b)
            ):
                _absorb_block(p, b)
                break
        else:
            out.append(b)
    return out


def _body_measure(blocks: list[TextBlock]) -> float:
    """Width of a full body line: the most common width (rounded to points) of the lines in multi-line blocks
    other than their last lines.
    Inline expressions cut a line into several pieces, so whole lines are measured"""
    widths: Counter[int] = Counter()
    for block in blocks:
        rows = _visual_lines(block)
        if len(rows) >= 3:
            widths.update(round(_row_width(row)) for row in rows[:-1])
    return float(widths.most_common(1)[0][0]) if widths else 0.0


def _row_width(row: list[pymupdf.Rect]) -> float:
    return max(rect.x1 for rect in row) - min(rect.x0 for rect in row)


def _row_gap(row: list[pymupdf.Rect]) -> float:
    """The widest gap between two adjacent pieces of a line."""
    pieces = sorted(row, key=lambda rect: rect.x0)
    return max((b.x0 - a.x1 for a, b in zip(pieces, pieces[1:])), default=0.0)


def _hangs(block: TextBlock, nxt: TextBlock | None = None) -> bool:
    """A hanging indent with the first line flush and the lines below indented to the right (one reference
    entry).
    With only one line, look at nxt right below: the continuation lines became a block of their own. A line
    ending at a sentence end is the last line of a paragraph, with an indented new paragraph below"""
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    if not rows:
        return False
    left = min(rect.x0 for rect in rows[0])
    if len(rows) == 1:
        if nxt is None or not nxt.line_rects or nxt.page != block.page or _ends_sentence(block.text):
            return False
        head = nxt.line_rects[0]
        bottom = max(rect.y1 for rect in rows[0])
        return (
            0 <= head.y0 - bottom <= block.size
            and abs(nxt.size - block.size) <= block.size * 0.05
            and block.size * 0.5 < head.x0 - left <= block.size * _MAX_INDENT_EM
        )
    return all(min(rect.x0 for rect in row) - left > block.size * 0.5 for row in rows[1:])


def _paragraph_space(blocks: list[TextBlock]) -> float:
    """In documents that separate paragraphs by spacing, the extra space above a new paragraph beyond normal
    line spacing (in points). Returns 0 for documents not set this way.
    Looks at blocks starting flush below the last line of a paragraph (ending a sentence, short of the right
    edge); it counts only when most of them have extra space of about half a character height or more"""
    gaps: list[tuple[float, float]] = []
    for p, b in zip(blocks, blocks[1:]):
        if p.page != b.page or not p.line_rects or not b.line_rects or abs(p.size - b.size) > p.size * 0.05:
            continue
        if _styled_edge(b, 0) or _LABEL_START_RE.match(b.text) or _NUMBERED_HEAD_RE.match(b.text) or _list_item(b.text):
            continue
        tail, head = p.line_rects[-1], b.line_rects[0]
        if (
            not _ends_sentence(p.text)
            or tail.x1 > max(p.rect.x1, b.rect.x1) - 2 * p.size
            or abs(head.x0 - tail.x0) > p.size * 0.3
        ):
            continue
        gap = head.y0 - tail.y1 - p.line_gap * p.size
        if 0 <= gap <= 1.5 * p.size:
            gaps.append((gap, p.size))
    if len(gaps) < _SPACED_PARAGRAPHS:
        return 0.0
    space, size = sorted(gaps)[len(gaps) // 2]
    return space if space >= 0.4 * size else 0.0


# Paragraph spacing must be measurable below at least this many last lines of paragraphs
_SPACED_PARAGRAPHS = 3
# Minimum number of body words in a continuation line that starts with an expression
_MATH_WRAP_WORDS = 3
# Markdown code fence. LaTeX sets ``` as curly quotes “‘
_FENCE_RE = re.compile(r"```|“‘|‘‘‘")
# Sentences do not stop at these words: when a line ends with one, the next line continues even when it starts
# with a capital (a proper noun)
_DANGLING_WORDS = frozenset(
    "a an the and or of to in for with by from on at as that which whose than its their our his her".split()
)


def _dangling(text: str) -> bool:
    words = text.split()
    return bool(words) and words[-1] in _DANGLING_WORDS


def _opens_with_math(block: TextBlock) -> bool:
    """The first piece is in a math font."""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    return bool(spans) and _span_is_math(spans[0])


def _opens_with_word(block: TextBlock) -> bool:
    """The first piece is not in a math font and starts with a lowercase word. An expression on the next line
    (t3 = t2 + x, hi = t5 + t3) does not count."""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    return (
        bool(spans) and not _span_is_math(spans[0]) and bool(re.match(r"[a-z]{2,}\b(?!\s*[\[(←=∈<>≤≥+])", block.text))
    )


def _wraps_into(block: TextBlock, nxt: TextBlock, indented: bool = False) -> bool:
    """The last line of block (with the numerators, denominators and limits on the same row) shares a left
    edge with the first line of nxt and runs to the right edge of both blocks.
    With indented, the last line of block is the indented first line of a paragraph, and the first line of
    nxt must be one indent further left."""
    lines = block.line_rects or [block.rect]
    tail = lines[-1]
    row = [
        rect for rect in lines if min(rect.y1, tail.y1) - max(rect.y0, tail.y0) >= min(rect.height, tail.height) * 0.5
    ]
    first = nxt.line_rects[0] if nxt.line_rects else nxt.rect
    shift = min(rect.x0 for rect in row) - first.x0
    if indented:
        lead = block.size * 0.5 < shift <= block.size * _MAX_INDENT_EM
    else:
        lead = abs(shift) <= block.size * 0.5
    return lead and max(rect.x1 for rect in row) >= max(block.rect.x1, nxt.rect.x1) - block.size * 0.5


# Abbreviations that appear almost only inside sentences. etc. often ends a sentence and is not included
_MID_ABBREV_RE = re.compile(r"(?:\bet al|\be\.g|\bi\.e|\bcf|\bvs)\.$")


def _beside_wrap(other: TextBlock, head: TextBlock, tail: TextBlock) -> bool:
    """other lies below head, the first half of the paragraph, and beside tail, the second half, horizontally
    within the width of head: the space the paragraph makes room for (a caption). A cell to the right on the
    same table row sits beside head and does not count."""
    if other.page != tail.page or other.rect.y0 < head.rect.y1 - 1:
        return False
    if other.rect.x0 >= tail.rect.x1:
        return head.rect.x1 > other.rect.x0
    if other.rect.x1 <= tail.rect.x0:
        return head.rect.x0 < other.rect.x1
    return False


def _is_line_tail(block: TextBlock, earlier: list[TextBlock]) -> bool:
    """This line sits right after the end of a line of a block to its left: half a sentence, not a new
    paragraph."""
    if len(block.line_rects) > 1 or _ends_sentence(block.text) or block.side_note:
        return False
    for other in earlier:
        if other.page != block.page or other.clip != block.clip or other.side_note:
            continue
        lines = other.line_rects or [other.rect]
        height = block.rect.height or block.size
        for line in lines:
            overlap = min(block.rect.y1, line.y1) - max(block.rect.y0, line.y0)
            if overlap <= min(height, line.height) * 0.5:
                continue
            gap = block.rect.x0 - line.x1
            if -2 <= gap < 12 and block.rect.x0 > line.x0 + max(block.size * 3, 24):
                return True
    return False


def _earlier_on_page(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """For each block, the blocks before it on the same page. Half sentences are compared only with lines on
    their own page; slicing the whole book at once takes a long time for books of several hundred pages."""
    seen: dict[int, list[TextBlock]] = {}
    out: list[list[TextBlock]] = []
    for block in blocks:
        same = seen.setdefault(block.page, [])
        out.append(list(same))
        same.append(block)
    return out


def _line_tail_blocked(nxt: TextBlock, fragment: TextBlock, blocks: list[TextBlock]) -> bool:
    """Do not merge when the continuing line also touches another block. Merged, the writing would cover that
    block."""
    for other in blocks:
        if other is nxt or other is fragment or other.page != nxt.page:
            continue
        if not _overlaps_line(nxt, other):
            continue
        if other.rect.x0 < nxt.rect.x1 and other.rect.x1 > nxt.rect.x0:
            return True
    return False


def _line_tail_match(block: TextBlock, nxt: TextBlock) -> bool:
    """Only a next line that returns to the column edge and starts in lowercase continues this half
    sentence."""
    if nxt.page != block.page or nxt.clip != block.clip or nxt.side_note:
        return False
    if abs(nxt.size - block.size) > block.size * 0.2:
        return False
    if nxt.rect.x0 > block.rect.x0 - max(block.size * 3.5, 28):
        return False
    gap = nxt.rect.y0 - block.rect.y1 - block.line_gap * block.size
    if not (-0.5 * block.size <= gap < 1.5 * block.size):
        return False
    return bool(nxt.text.lstrip()[:1].islower()) and not _verbatim_seam(block, nxt)


def _line_tail_continuation(
    block: TextBlock,
    later: list[TextBlock],
    blocks: list[TextBlock],
) -> TextBlock | None:
    for nxt in later:
        if nxt.page != block.page or nxt.rect.y0 > block.rect.y1 + block.size * 2:
            return None
        if not _line_tail_match(block, nxt) or _line_tail_blocked(nxt, block, blocks):
            continue
        return nxt
    return None


def _merge_line_tails(blocks: list[TextBlock], edges: Collection[int] = ()) -> list[TextBlock]:
    """Join a half sentence on the right of a line with its continuation on the next line back at the column
    edge into one paragraph. It is not merged into the sentence on its left.
    edges are running headers and footers, used neither as half sentences nor as continuations."""
    used: set[int] = set()
    out: list[TextBlock] = []
    earlier = _earlier_on_page(blocks)
    for index, block in enumerate(blocks):
        if id(block) in used:
            continue
        if id(block) not in edges and not block.entry and _is_line_tail(block, earlier[index]):
            later = [other for other in blocks[index + 1 :] if id(other) not in edges]
            nxt = _line_tail_continuation(block, later, blocks)
            if nxt is not None and id(nxt) not in used and not nxt.entry:
                _absorb_block(block, nxt)
                used.add(id(nxt))
        out.append(block)
    return out


def _returns_to_margin(host: TextBlock, guest: TextBlock) -> bool:
    """Only when the later lines return to the host's column left edge is it the same paragraph. Affiliation
    lines side by side do not return."""
    rest = guest.line_rects[1:]
    if not rest:
        return True
    host_left = min(line.x0 for line in host.line_rects)
    rest_left = min(line.x0 for line in rest)
    return abs(rest_left - host_left) <= max(host.size * 4, 36)


def _sits_on_last_line(host: TextBlock, guest: TextBlock) -> bool:
    """The next sentence starts to the right of the last line of the previous sentence. In two-column papers
    this is still the same paragraph."""
    if host.page != guest.page or host.clip != guest.clip:
        return False
    if not host.line_rects or not guest.line_rects or host.size <= 0:
        return False
    if abs(host.size - guest.size) > host.size * 0.2:
        return False
    # Compare positions before text: blocks on a page are compared pairwise, and most pairs are ruled out here
    last = host.line_rects[-1]
    first = guest.line_rects[0]
    height = min(last.height, first.height) or 1.0
    overlap = min(last.y1, first.y1) - max(last.y0, first.y0)
    if overlap <= height * 0.5:
        return False
    gap = first.x0 - last.x1
    if not (-2 <= gap < 16):
        return False
    if first.x0 <= last.x0 + max(host.size * 3, 24):
        return False
    if _is_side_label(host) or _diagramish(host) or _diagramish(guest):
        return False
    return _returns_to_margin(host, guest)


def _continues_fragment(host: TextBlock, guest: TextBlock) -> bool:
    """The line after a half sentence at a line end returns to the column edge and starts in lowercase. It is
    joined first, even when the next sentence sits to the right on this line.

    A right half line that already reaches a sentence end is no half sentence. The otherwise. of a formula
    condition line must not pull in the body text below.
    """
    if host.page != guest.page or host.clip != guest.clip:
        return False
    if not host.line_rects or not guest.line_rects or host.size <= 0:
        return False
    if _ends_sentence(host.text):
        return False
    if abs(host.size - guest.size) > host.size * 0.2:
        return False
    last = host.line_rects[-1]
    left = min(line.x0 for line in host.line_rects)
    if last.x0 <= left + max(host.size * 3, 24):
        return False
    first = guest.line_rects[0]
    if first.x0 > last.x0 - max(host.size * 3.5, 28):
        return False
    # A half sentence at a line end wraps to the next line, and this line runs to the right edge of the column.
    # A display expression has space on both sides, and the body text below starts a new paragraph;
    # limits under an expression are short pieces of math and are still joined
    if last.x1 < first.x1 - host.size * 3 and _prose_units(guest.text) >= 4:
        return False
    gap = first.y0 - last.y1 - host.line_gap * host.size
    if not (-0.5 * host.size <= gap < 1.5 * host.size):
        return False
    head = guest.text.lstrip()[:1]
    if not (head and (head.islower() or head in "),;%,；，")):
        return False
    return not (_diagramish(guest) or _is_side_label(guest) or _verbatim_seam(host, guest))


def _merge_same_line_sentences(blocks: list[TextBlock], edges: Collection[int] = ()) -> list[TextBlock]:
    """When a sentence ends mid-line and the next sentence starts to its right on the same line, merge them
    into one paragraph laid out at the column width.

    Short bold side labels and labels inside figures are not merged. A continuation line is joined first even
    when it shares a baseline with the following sentence, which is merged after it.
    Reading order can put the left half line after the right half, so adjacent blocks alone are not enough.
    The search stays within one page. Comparing the whole book pairwise takes a long time for books of several
    hundred pages.
    edges are running headers and footers and take no part.
    """
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if id(block) not in edges:
            by_page.setdefault(block.page, []).append(block)
    absorbed: set[int] = set()
    for page_blocks in by_page.values():
        absorbed |= _merge_page_line_sentences(page_blocks)
    return [block for block in blocks if id(block) not in absorbed]


def _same_line_sentence(host: TextBlock, guest: TextBlock) -> bool:
    # Margin notes, index entries and word grid cells each stand on their own
    if host.side_note or guest.side_note or host.entry or guest.entry:
        return False
    return _sits_on_last_line(host, guest) or _continues_fragment(host, guest)


def _merge_page_line_sentences(blocks: list[TextBlock]) -> set[int]:
    """Go through the hosts on a page in order, each taking the first block that can be merged into it.
    Returns the blocks merged away.

    After a merge the search does not restart from the beginning. The earlier hosts had no block to merge,
    and only the block that just changed may join after one of them.
    """
    out = list(blocks)
    absorbed: set[int] = set()
    index = 0
    while index < len(out):
        host = out[index]
        guest_index = next(
            (other for other, guest in enumerate(out) if other != index and _same_line_sentence(host, guest)),
            None,
        )
        if guest_index is None:
            index += 1
            continue
        absorbed.add(id(out[guest_index]))
        _absorb_block(host, out[guest_index])
        del out[guest_index]
        if guest_index < index:
            index -= 1
        while True:
            earlier = next((other for other in range(index) if _same_line_sentence(out[other], out[index])), None)
            if earlier is None:
                break
            absorbed.add(id(out[index]))
            _absorb_block(out[earlier], out[index])
            del out[index]
            index = earlier
    return absorbed


def _merge_caption_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    """Merge figure and table captions broken into fragments: 'Figure 2:', 'Examples of decay' and
    'schedules.' translated separately and put back into their own small boxes wrap badly. Joined back into
    one block, the sentence is translated whole and laid out on one line.
    The rules are conservative and join only three kinds: a) a fragment right after a short label on the same
    visual line, where the label may end with a colon or be "图 2" or "Fig. 1."; b) a one-line note right
    below such a label; c) within a caption group with a colon, a short continuation line that is
    left-aligned, starts in lowercase, and follows a line without sentence-final punctuation. Table rows and
    headings are not merged."""
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            # An index entry ends at its page numbers (perplexity:coined by, 89 has a colon but is no caption
            # label)
            if p.page == b.page and p.clip == b.clip and not (p.side_note or b.side_note or p.entry):
                y_ov = min(p.rect.y1, b.rect.y1) - max(p.rect.y0, b.rect.y0)
                same_line = y_ov > 0.5 * min(p.rect.y1 - p.rect.y0, b.rect.y1 - b.rect.y0)
                gap = b.rect.x0 - p.rect.x1
                # Notes and continuation lines have the caption's size. Small text under a large heading is a
                # box in a figure
                below = (
                    -0.1 * p.size <= b.rect.y0 - p.rect.y1 - p.line_gap * p.size < 1.1 * p.size
                    and abs(b.rect.x0 - p.rect.x0) < p.size
                    and abs(b.size - p.size) <= p.size * 0.2
                )
                label = (p.text.rstrip().endswith((":", "：")) and len(p.text) < 30) or _caption_label(p.text)
                head = b.text.lstrip()[:1]
                note = bool(head and (head.islower() or CJK_RE.search(head)) and not _caption_label(b.text))
                # A line that itself starts with a bold label is the next entry of a definition list
                # (mouse² : ... like a mouse | bank¹ : ...), not a continuation line
                continuation = (
                    len(p.text) < 60
                    and b.text[:1].islower()
                    and (":" in p.text or "：" in p.text)
                    and not _ends_sentence(p.text)
                    and not (
                        b.span_lines
                        and _label_lead([span for span in b.span_lines[0] if str(span.get("text") or "").strip()])
                    )
                )
                # A definition-style label is followed by a column of values: several lines, each starting at
                # the values' left edge (View ranges:  {...}, | {...}).
                # Merged into the label, the translation would wrap back under the label and the column would
                # be lost. Caption labels are merged as before
                rows = _visual_lines(b) if b.line_rects else []
                column = (
                    not _caption_label(p.text)
                    and len(rows) >= 2
                    and b.rect.x0 > p.rect.x1
                    and all(abs(min(rect.x0 for rect in row) - b.rect.x0) <= 1.0 for row in rows)
                )
                if (same_line and -1 <= gap < 3 * p.size and label and not column) or (
                    below and (continuation or (label and note))
                ):
                    _absorb_block(p, b)
                    continue
        out.append(b)
    return out
