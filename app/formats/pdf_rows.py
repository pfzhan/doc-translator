"""Split text lines into rows and cells: rejoin cut lines, split table cells, contents leaders, list items and
stacked or side-by-side pieces."""

import re
from collections import Counter
from collections.abc import Callable, Collection, Sequence

import pymupdf

from .pdf_blocks import (
    _MARK_ITEM_RE,
    Leader,
    _by_chars,
    _span_is_math,
)
from .pdf_math import (
    _line_box,
    _text_bounds,
)
from .pdf_types import Line, LineItem, Span


def _text_box(span: Span) -> pymupdf.Rect:
    """The height of the line the span occupies. Math fonts are measured by glyph ink, the rest by PyMuPDF's
    box.

    Radical signs and large braces have their origin at the top of the glyph instead of on the baseline, so
    only the ink can be used."""
    return pymupdf.Rect(_text_bounds(span))


def _stacked_rows(upper: list[Span], lower: list[Span]) -> bool:
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
    return bool(abs(baselines[0] - baselines[1]) > size * 0.3)


def _left_to_right(items: list[LineItem]) -> list[LineItem]:
    """Sort the pieces of one visual line from left to right. The numerator of a fraction often comes before
    the denominator, which then no longer follows the equals sign before it.
    Line boxes are ink boxes, so two lines above each other overlap little and are not taken as one line."""
    out: list[LineItem] = []
    run: list[LineItem] = []
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


def _grid_cells(items: list[LineItem]) -> set[int]:
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


def _index_entry_ends(items: list[LineItem], ref_rows: Sequence[pymupdf.Rect]) -> set[int]:
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
            # below is set exactly when nxt is; both checks are for the type checker
            and below is not None
            and nxt is not None
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
            and nxt is not None
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


def _number_cells(items: list[LineItem]) -> set[int]:
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


# Maximum number of words in one header cell. A justified body line with stretched word spacing can also be
# split into several pieces, each half a sentence
_HEADER_CELL_WORDS = 3


def _cut_header_cells(items: list[LineItem], columns: Sequence[float]) -> tuple[list[LineItem], set[int]]:
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
    out: list[LineItem] = []
    found: set[int] = set()
    for row in rows:
        spans = sorted((span for index in row for span in items[index][2]), key=lambda span: span["bbox"][0])
        pieces: list[list[Span]] = []
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


def _drop_number_tails(items: list[LineItem]) -> list[LineItem]:
    """The row label is too long and PyMuPDF has merged the following number cells into the same line
    (BERT (Ens.+TriviaQA) 86.2 92.2 ...). When the rest of the line is all numbers and the left edge of its
    first cell aligns with a standalone number cell on another line, remove it from the line and leave it on
    the original page, not sent for translation, like the numbers on other lines. Only gaps of nearly half an em
    or more count: word spacing in a sentence is never that wide."""
    starts = [items[index][1].x0 for index in _number_cells(items)]
    if not starts:
        return items
    out: list[LineItem] = []
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


def _cut_leaders(items: list[LineItem], toc_page: bool = False) -> tuple[list[LineItem], dict[int, Leader]]:
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
    replaced: dict[int, LineItem | None] = {}
    marked: dict[int, Leader] = {}
    for row in rows:
        pieces = [(index, span) for index in row for span in items[index][3]]
        text = "".join(str(span["text"]) for _, span in pieces)
        cut = _leader_start(pieces, text, toc_page)
        if cut is None or not re.search(r"[^\W\d_]", text[:cut]):
            continue
        kept: dict[int, list[Span]] = {index: [] for index in row}
        dropped: list[Span] = []
        shared: list[Span] = []
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
            row_box = items[index][1]
            line: Line = {"spans": spans, "bbox": (row_box.x0, row_box.y0, row_box.x1, row_box.y1)}
            replaced[index] = (
                "".join(str(span["text"]) for span in spans),
                _line_box(line),
                [span for span in spans if str(span["text"]).strip()],
                spans,
            )
            last = index
        if last is not None:
            marked[last] = leader
    out: list[LineItem] = []
    leaders: dict[int, Leader] = {}
    for index, item in enumerate(items):
        new_item = replaced.get(index, item)
        if new_item is None:
            continue
        if index in marked:
            leaders[len(out)] = marked[index]
        out.append(new_item)
    return out, leaders


def _leader_start(pieces: list[tuple[int, Span]], text: str, toc_page: bool) -> int | None:
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


def _span_offsets(pieces: list[tuple[int, Span]]) -> list[tuple[int, Span]]:
    out: list[tuple[int, Span]] = []
    offset = 0
    for _, span in pieces:
        out.append((offset, span))
        offset += len(str(span["text"]))
    return out


def _split_span(span: Span, at: int) -> tuple[Span, Span]:
    """Cut span in two at character at, dividing the box in proportion to character counts. Used when the
    leader and the title are in the same span."""
    body = str(span["text"])
    box = pymupdf.Rect(span["bbox"])
    middle = box.x0 + box.width * at / max(len(body), 1)
    left: Span = {**span, "text": body[:at], "bbox": (box.x0, box.y0, middle, box.y1)}
    right: Span = {**span, "text": body[at:], "bbox": (middle, box.y0, box.x1, box.y1)}
    return left, right


def _leader_of(spans: list[Span], shared: list[Span]) -> Leader | None:
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


def _cut_at_posts(group: list[int], items: list[LineItem], posts: Sequence[pymupdf.Rect]) -> list[list[int]]:
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


def _vivid_share(spans: Sequence[Span]) -> float:
    """Share of letters in vivid colors, 0 when there are no letters"""
    letters = sum(char.isalpha() for span in spans for char in str(span.get("text") or ""))
    vivid = sum(
        char.isalpha() for span in spans if _vivid(int(span.get("color") or 0)) for char in str(span.get("text") or "")
    )
    return vivid / letters if letters else 0.0


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


def _cut_line_list(group: list[int], items: list[LineItem], page_rows: Sequence[pymupdf.Rect]) -> list[list[int]]:
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


def _cut_item_breaks(group: list[int], items: list[LineItem], page_rows: Sequence[pymupdf.Rect]) -> list[list[int]]:
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

    def spans(row: list[int]) -> list[Span]:
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


def _column_rows(rows: list[list[int]], items: list[LineItem], size: float) -> bool:
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
