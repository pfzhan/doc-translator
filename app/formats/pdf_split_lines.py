"""Split the text lines of a page into rows, cells and paragraph pieces by their spacing, fonts and labels."""

import re
from collections.abc import Collection, Sequence

import pymupdf

from .pdf_blocks import (
    _ROMAN_ITEM_RE,
    CJK_RE,
    _ends_sentence,
    _list_item,
    _span_is_math,
)
from .pdf_flow import (
    emphasis_of,
)
from .pdf_math import _prose_units
from .pdf_rows import _INDEX_REFS_ONLY_RE, _number_cells, _stacked_rows
from .pdf_types import LineItem, Span

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


def _label_lead(spans: list[Span]) -> bool:
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


def _justified_joins(items: list[LineItem], max_x1: float, whole_rows: bool = False) -> set[int]:
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


def _full_row(row: list[LineItem], items: list[LineItem], max_x1: float) -> bool:
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


def _same_row(left: LineItem, right: LineItem) -> bool:
    a, b = left[1], right[1]
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap < min(a.height, b.height) * 0.7 or b.x0 <= a.x1:
        return False

    # Same size, sharing a font. A piece with an italic variable inside it (AVG(E)) also counts; cells side by
    # side often differ entirely in font and size
    def styles(item: LineItem) -> set[tuple[str | None, float]]:
        return {
            (s.get("font"), round(float(s.get("size") or 0), 1)) for s in item[2] if str(s.get("text") or "").strip()
        }

    ours, theirs = styles(left), styles(right)
    return len({size for _, size in ours | theirs}) == 1 and bool(ours & theirs)


def _spread_evenly(row: list[LineItem], max_x1: float) -> bool:
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


def _row_labels(items: list[LineItem], cells: Collection[int]) -> set[int]:
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


def _math_seam(left: list[Span], right: list[Span]) -> bool:
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


def _typewriter_line(spans: list[Span]) -> bool:
    """The whole line is monospaced Latin text. Chinese and Japanese fonts are also often marked monospaced and
    do not count."""
    visible = [span for span in spans if str(span.get("text") or "").strip()]
    if not visible or any(CJK_RE.search(str(span["text"])) for span in visible):
        return False
    return all(_mono_span(span) for span in visible)


def _row_style(spans: list[Span]) -> tuple[bool, bool]:
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


def _row_left(items: list[LineItem], index: int) -> float:
    """Left edge of the line that piece index is on: the leftmost of the pieces overlapping it vertically by
    more than half."""
    box = items[index][1]
    left: float = min(
        other[1].x0
        for other in items
        if min(other[1].y1, box.y1) - max(other[1].y0, box.y0) > min(other[1].height, box.height) * 0.5
    )
    return left


def _split_lines(items: list[LineItem], whole_rows: bool = False) -> list[list[int]]:
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
        above_left = _row_left(items, i - 1)
        below_left = (
            _row_left(items, i + 1) if i + 1 < len(items) and items[i + 1][1].y0 >= rect.y1 - 0.5 * psize else None
        )
        centered_rows = below_left is not None and all(
            abs((other.x0 + other.x1) - (rect.x0 + rect.x1)) / 2 <= 0.15 * psize for other in (prect, items[i + 1][1])
        )
        opens_indent = (
            own_row
            and seg_start
            and below_left is not None
            and not centered_rows
            and above_left - rect.x0 > 0.5 * psize
            and 0.5 * psize < rect.x0 - below_left <= 3 * psize
            and not ptext.rstrip().endswith((",", "，", "、", ";", "；", ":", "："))
        )
        # The line above ends a sentence (even when full), this line is indented one or two ems and runs to the
        # right edge, and the next line returns to the left edge of the line above: the first-line indent of a
        # new paragraph. Between numbered entries it is a hanging continuation line
        # (46. Fuller ... | Monthly Notices ... | 47. ...)
        indent_after_end = (
            own_row
            and seg_start
            and below_left is not None
            and not centered_rows
            and _ends_sentence(ptext)
            and not _list_item(items[i - 1][0])
            and not _list_item(items[i + 1][0])
            and 0.5 * psize < rect.x0 - above_left <= 3 * psize
            and abs(below_left - above_left) <= 0.5 * psize
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


def _caption_opens(spans: list[Span], above: list[Span]) -> bool:
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


def _text_size(spans: list[Span]) -> float:
    """The font size with the most characters in a line."""
    count: dict[float, int] = {}
    for span in spans:
        count[span["size"]] = count.get(span["size"], 0) + len(span.get("text", "").strip())
    return max(count, key=count.__getitem__, default=10)


# Relation symbols of display formulas. Plus, minus and times signs must be in a math font: in a body font
# they are often in headings such as Phoenix + Calcite.
# Arrows do not count; slides often use arrows to link a few words
_RELATION_RE = re.compile(r"[=≤≥≈≡≠∝∈∉⊂⊆⊇≜]|(?:^|\s)[<>](?=\s|$)")


def _mono_span(span: Span) -> bool:
    """Monospaced font: recognized by font name, and also by the monospace flag in the PDF (fonts such as txtt
    whose names lack Mono)."""
    return bool(MONO_FONT_RE.search(str(span.get("font") or "")) or int(span.get("flags") or 0) & 8)


# Abbreviations that appear almost only inside sentences. etc. often ends a sentence and is not included
_MID_ABBREV_RE = re.compile(r"(?:\bet al|\be\.g|\bi\.e|\bcf|\bvs)\.$")
