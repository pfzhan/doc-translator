"""Find tables on a page: ruled, banded, filled, unruled and algorithm boxes, with interlinear glosses and
numbered examples."""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

from .pdf_drawings import (
    _NEAR,
    _PAGE_FILL,
    _SPAN_MAX,
    RuledTable,
    _find_gutters,
    _mostly_inside,
    _page_drawings,
    _rule_segments,
    _TextWord,
    light_fill,
    page_words,
    text_page_rect,
)

_CELL_MIN_W = 18.0


_CELL_MIN_H = 8.0


_CELL_NARROW = 16.0


_MAX_TABLE_COLS = 12


# Snap distance of find_tables: filled shapes thinner than this also count as rules
_SNAP = 3.0


# Boundary between two adjacent rows of an algorithm box, as a share of the line spacing below the upper baseline
_ROW_CUT = 0.35


_TEXT_JOIN = 8.0


_COL_ALIGN = 5.0


_MIN_TEXT_ROWS = 3


_MIN_PITCH = 8.0


# Interlinear gloss: tolerance for aligned left edges, minimum number of words, and share of the words in each of the
# two lines that must align
_GLOSS_ALIGN = 1.0


_GLOSS_MIN = 4


_GLOSS_SHARE = 0.75


# Role labels below the words of an example sentence; small capitals extract as all capitals: AGENT, ARG0, ARGM-TMP
# Example numbers: (22.27), (3), (4a)
_NUMBER_LABEL_RE = re.compile(r"\(\d+(?:\.\d+)*[a-z]?\)")


# Example sentences numbered by chapter: (22.9), (3.4a)
_CHAPTER_NUMBER_RE = re.compile(r"\(\d+\.\d+[a-z]?\)")


_ROLE_LABEL_RE = re.compile(r"[A-Z][A-Z0-9-]{2,}")


# Grammatical abbreviations common in the Leipzig Glossing Rules. Single letters (S, A, P) are too common elsewhere
# and do not count
_LEIPZIG = frozenset(
    "SG PL DU NOM ACC GEN DAT ERG ABS LOC ABL ALL INS INSTR COM VOC PST PRS FUT PFV IPFV PRF IMPF "
    "PROG NEG DEF INDF CLF COP AUX POSS REFL RECP CAUS PASS APPL TOP FOC DECL IMP SBJ OBJ OBL DEM "
    "PTCP INF NMLZ COMP REL".split()
)


_PERSON_RE = re.compile(r"[123](?:SG|PL|DU)")


_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


# Syllables in a transcription line are separated by spaces and punctuation. Extracted tone marks are separate
# characters (z`ai, zh¯u) and do not separate syllables
_SYLLABLE_SPLIT_RE = re.compile(r"[\s,.;:!?，。；：！？、()（）\[\]“”\"'‘’]+")


_MORPHEME_SPLIT_RE = re.compile(r"[-.=\[\]<>:~()\s]+")


_MAX_PITCH = 36.0


_SHORT_WORDS = 3


_SHORT_WIDTH = 80.0


_SHORT_SHARE = 0.8


# Maximum number of text line heights between the top rule and the middle rule of a three-line table
_HEAD_ROWS = 4.0


# Minimum length of the rules of a three-line table. Underlines of words and fraction bars in body text are much
# shorter
_BAND_MIN_W = 120.0


_BAND_SHORT_SHARE = 0.6


_LABEL_RE = re.compile(r"(?:Figure|Fig\.|Table|Tab\.)\s*\d+(?:\.\d+)*\b")


def ruled_tables(page: pymupdf.Page, drawings: list[dict] | None = None) -> tuple[RuledTable, ...]:
    """Only cells enclosed by horizontal and vertical rules count. False tables formed by aligned text and grids of
    formula strokes are left out. drawings holds the vector graphics of this page (including rules drawn as images,
    see _page_drawings); read now when not given.

    find_tables first converts every character of the page into a dict, which takes tens of milliseconds per page.
    When the page does not have two horizontal and two vertical rules, no cell can be enclosed and the call is
    skipped; otherwise only the text in the band around the rules is taken (see _table_clip)."""
    if drawings is None:
        drawings = _page_drawings(page)
    if not _grid_possible(drawings):
        return ()
    pymupdf.no_recommend_layout()
    try:
        found = page.find_tables(
            clip=_table_clip(page, drawings),
            vertical_strategy="lines_strict",
            horizontal_strategy="lines_strict",
            paths=drawings,
        )
    except Exception:  # noqa: BLE001 - the rules of some pages cannot be read; treat as no tables
        return ()
    tables: list[RuledTable] = []
    rules: tuple[list[pymupdf.Rect], list[pymupdf.Rect]] | None = None
    for table in found.tables:
        cells = tuple(pymupdf.Rect(cell) for cell in table.cells if cell)
        if not _usable_table(cells):
            continue
        if rules is None:
            rules = _rule_segments(page, drawings)
        tables.append(_with_open_edges(RuledTable(pymupdf.Rect(table.bbox), cells), *rules))
    return tuple(tables)


def _table_clip(page: pymupdf.Page, drawings: Sequence[dict]) -> pymupdf.Rect | None:
    """The area from which find_tables takes text. All rules lie inside it, so the tables found are the same as
    without a limit; only the text outside the area is not converted.

    find_tables uses the text for two decisions: a table area must contain text to count as a table, and a patch of
    connected rules must contain text before an outer frame is added to it. The second compares the top and bottom
    edges of character boxes after conversion by the page transformation matrix, which amounts to comparing
    vertically flipped positions, so besides the band around the rules the area also includes that band flipped
    vertically. The bounding box of a rule may have zero width or height, and Rect union would skip it as empty, so
    the extremes are taken by coordinate. Rotated pages get no limit."""
    if page.rotation or not drawings:
        return None
    rects = [pymupdf.Rect(drawing["rect"]) for drawing in drawings]
    area = pymupdf.Rect(
        min(r.x0 for r in rects),
        min(r.y0 for r in rects),
        max(r.x1 for r in rects),
        max(r.y1 for r in rects),
    )
    flipped = area * ~page.transformation_matrix
    top, bottom = min(area.y0, flipped.y0), max(area.y1, flipped.y1)
    return (
        pymupdf.Rect(
            area.x0 - _TABLE_CLIP_PAD, top - _TABLE_CLIP_PAD, area.x1 + _TABLE_CLIP_PAD, bottom + _TABLE_CLIP_PAD
        )
        & page.rect
    )


# Margin of the text area around the rules: MuPDF decides whether to take a character by its own measured box,
# which differs slightly from the reported box
_TABLE_CLIP_PAD = 5.0


def _grid_possible(drawings: Sequence[dict]) -> bool:
    """At least two horizontal and two vertical rules.

    Counted the way lines_strict takes edges: line segments and the edges of rectangles and quads all count,
    regardless of length (short lines are joined into long ones); filled shapes without a stroke that are wider than
    the snap distance in both directions are not rules."""
    horizontal = vertical = 0
    for drawing in drawings:
        rect = pymupdf.Rect(drawing.get("rect"))
        if drawing.get("type") == "f" and rect.width > _SNAP and rect.height > _SNAP:
            continue
        for item in drawing.get("items", ()):
            if item[0] == "l":
                box = pymupdf.Rect(item[1], item[2]).normalize()
            elif item[0] == "re":
                box = pymupdf.Rect(item[1]).normalize()
            elif item[0] == "qu":
                box = pymupdf.Quad(item[1]).rect
            else:
                continue
            thin_h, thin_v = box.height <= _SNAP, box.width <= _SNAP
            if item[0] != "l" and not (thin_h or thin_v):
                horizontal, vertical = horizontal + 2, vertical + 2
            else:
                horizontal += thin_h
                vertical += thin_v
            if horizontal >= 2 and vertical >= 2:
                return True
    return False


def _with_open_edges(
    table: RuledTable,
    horizontal: list[pymupdf.Rect],
    vertical: list[pymupdf.Rect],
) -> RuledTable:
    """Add the outer rows and columns that have no outer frame.

    Many papers draw only the rules between cells: horizontal rules reach past the outermost vertical rule, and
    vertical rules reach past the top and bottom horizontal rules. find_tables only detects cells with rules on all
    four sides, so the header, the first and last columns and the last row are missed. The grid is completed up to
    where the rules reach; adjacent cells without a rule between them (headers spanning columns) are merged into one.
    """
    box = pymupdf.Rect(table.rect)
    for _ in range(3):
        ys = [
            v
            for v in vertical
            if box.x0 - _NEAR <= v.x0 <= box.x1 + _NEAR and v.y1 >= box.y0 - _NEAR and v.y0 <= box.y1 + _NEAR
        ]
        xs = [
            h
            for h in horizontal
            if box.y0 - _NEAR <= h.y0 <= box.y1 + _NEAR and h.x1 >= box.x0 - _NEAR and h.x0 <= box.x1 + _NEAR
        ]
        grown = pymupdf.Rect(
            min([box.x0, *(h.x0 for h in xs)]),
            min([box.y0, *(v.y0 for v in ys)]),
            max([box.x1, *(h.x1 for h in xs)]),
            max([box.y1, *(v.y1 for v in ys)]),
        )
        if all(abs(a - b) < 0.5 for a, b in zip(grown, box)):
            break
        box = grown
    if all(abs(a - b) < 0.5 for a, b in zip(box, table.rect)):
        return table
    col_edges = _edges([box.x0, box.x1, *(v.x0 for v in ys)])
    row_edges = _edges([box.y0, box.y1, *(h.y0 for h in xs)])
    found: list[pymupdf.Rect] = []
    for top, bottom in zip(row_edges, row_edges[1:]):
        row: list[pymupdf.Rect] = []
        for left, right in zip(col_edges, col_edges[1:]):
            center = pymupdf.Point((left + right) / 2, (top + bottom) / 2)
            if any(old.contains(center) for old in table.cells):
                found.extend(row)
                row = []
                continue
            cell = pymupdf.Rect(left, top, right, bottom)
            if row and abs(row[-1].x1 - left) < _NEAR and not _ruled(vertical, left, top, bottom, upright=True):
                row[-1] = row[-1] | cell
            else:
                row.append(cell)
        found.extend(row)
    added = _merge_rows(found, horizontal)
    return RuledTable(box, (*table.cells, *added))


def _edges(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if not out or value - out[-1] > _NEAR:
            out.append(value)
    return out


def _ruled(segments: list[pymupdf.Rect], at: float, start: float, end: float, *, upright: bool) -> bool:
    """Rules at at cover most of the range from start to end. With upright true, look for vertical rules, otherwise
    horizontal ones."""
    covered = 0.0
    for seg in segments:
        pos = seg.x0 if upright else seg.y0
        if abs(pos - at) > _NEAR:
            continue
        lo, hi = (seg.y0, seg.y1) if upright else (seg.x0, seg.x1)
        covered += max(0.0, min(hi, end) - max(lo, start))
    return covered >= (end - start) * 0.6


def _merge_rows(cells: list[pymupdf.Rect], horizontal: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Merge outer cells that are vertically adjacent, aligned left and right, and not separated by a horizontal rule
    (row names spanning rows)."""
    out: list[pymupdf.Rect] = []
    for cell in sorted(cells, key=lambda c: (c.x0, c.y0)):
        prev = out[-1] if out else None
        if (
            prev is not None
            and abs(prev.x0 - cell.x0) < _NEAR
            and abs(prev.x1 - cell.x1) < _NEAR
            and abs(prev.y1 - cell.y0) < _NEAR
            and not _ruled(horizontal, cell.y0, cell.x0, cell.x1, upright=False)
        ):
            out[-1] = prev | cell
            continue
        out.append(cell)
    return out


def page_tables(
    page: pymupdf.Page,
    drawings: list[dict] | None = None,
    words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """Ruled tables and tables made of shaded cells, plus short-cell tables with aligned column edges and algorithm
    boxes. Two-column body text and whole paragraphs are not included. drawings and words hold the vector graphics
    and words already read for this page; read now when not given."""
    if drawings is None:
        drawings = _page_drawings(page)
    if words is None:
        words = page_words(page)
    ruled = tuple(_with_text_rows(table, words) for table in ruled_tables(page, drawings))
    ruled = (*ruled, *filled_tables(page, drawings, words, ruled))
    ruled = (*ruled, *column_ruled_tables(_rule_segments(page, drawings)[1], words, ruled))
    algorithms = algorithm_tables(page, drawings, words)
    banded = banded_tables(_rule_segments(page, drawings)[0], words, (*ruled, *algorithms))
    ruled = (*(table for table in ruled if not _mostly_inside(table.rect, [band.rect for band in banded])), *banded)
    # Unruled cell tables are found by aligned rows in the first place, so an aligned header is already included and
    # there is no search upward
    ruled = _with_header_rows(ruled, words)
    unruled = unruled_tables(page, (*ruled, *algorithms), words)
    if unruled:
        unruled = _with_ruled_heads(unruled, words, _rule_segments(page, drawings)[0])
    return (*ruled, *unruled, *algorithms)


def _with_header_rows(tables: Sequence[RuledTable], words: Sequence[_TextWord]) -> tuple[RuledTable, ...]:
    """In a three-line table a rule separates the header from the body, and the rules only yield the body. Rows right
    above the table, horizontally within its extent, with every word inside one column of the table (not crossing a
    column line), covering three or more columns and with at most three words per column are header rows, and each
    row joins the table as one cell. Descriptions and body text above the table are continuous sentences that cross
    the column lines, and the search stops there."""
    zones = [table.rect for table in tables]
    out: list[RuledTable] = []
    for table in tables:
        rect = table.rect
        lines = sorted({x for cell in table.cells for x in (cell.x0, cell.x1) if rect.x0 + 2 < x < rect.x1 - 2})
        above = [
            word
            for word in words
            if word.x0 >= rect.x0 - 2
            and word.x1 <= rect.x1 + 2
            and word.y1 <= rect.y0 + 1
            and not _mostly_inside(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), zones)
        ]
        top = rect.y0
        headers: list[pymupdf.Rect] = []
        for row in reversed(_word_rows(above)):
            box = pymupdf.Rect(
                min(w.x0 for w in row), min(w.y0 for w in row), max(w.x1 for w in row), max(w.y1 for w in row)
            )
            if top - box.y1 > box.height or not _header_row(row, lines):
                break
            headers.append(pymupdf.Rect(rect.x0, box.y0 - 1, rect.x1, min(box.y1 + 1, top)))
            top = box.y0
        if headers:
            grown = pymupdf.Rect(rect.x0, top - 1, rect.x1, rect.y1)
            out.append(RuledTable(grown, (*reversed(headers), *table.cells), table.algorithm))
        else:
            out.append(table)
    return tuple(out)


def column_ruled_tables(
    vertical: Sequence[pymupdf.Rect],
    words: Sequence[_TextWord],
    known: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """Tables with only column lines: no outer frame and only one horizontal rule below the header, so find_tables
    cannot enclose cells; numbers are centered, so alignment of left and right edges does not detect them either.
    When several vertical lines have the same vertical extent, each text row within that extent has text on both sides
    of every line, and most cells are short, columns are split by the lines and rows by the text. A separator between
    two columns of body text has long sentences on both sides and does not count."""
    lines = _joined_verticals(vertical)
    tables: list[RuledTable] = []
    used: set[int] = set()
    for index, line in enumerate(lines):
        if index in used:
            continue
        group = [
            other
            for other in lines
            if other.height > 0
            and min(line.y1, other.y1) - max(line.y0, other.y0) >= 0.8 * min(line.height, other.height)
        ]
        used.update(lines.index(other) for other in group)
        xs = sorted(other.x0 for other in group)
        top, bottom = min(other.y0 for other in group), max(other.y1 for other in group)
        inside = [word for word in words if top < (word.y0 + word.y1) / 2 < bottom]
        rows = _word_rows(inside)
        if len(rows) < _MIN_TEXT_ROWS:
            continue
        sided = [row for row in rows if all(any(w.x1 <= x for w in row) and any(w.x0 >= x for w in row) for x in xs)]
        if len(sided) < len(rows) * 0.8:
            continue
        pieces = [cell for row in rows for cell in _row_cells(row)]
        if (
            sum(1 for cell in pieces if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
            < len(pieces) * _SHORT_SHARE
        ):
            continue
        left = min(w.x0 for w in inside) - 2
        right = max(w.x1 for w in inside) + 2
        area = pymupdf.Rect(left, top, right, bottom)
        if any(_mostly_inside(area, [table.rect]) or _mostly_inside(table.rect, [area]) for table in known):
            continue
        tops = [min(w.y0 for w in row) for row in rows]
        edges = [top, *((max(w.y1 for w in upper) + lower_top) / 2 for upper, lower_top in zip(rows, tops[1:])), bottom]
        cols = [left, *xs, right]
        cells = tuple(
            pymupdf.Rect(x0, y0, x1, y1) for y0, y1 in zip(edges, edges[1:]) for x0, x1 in zip(cols, cols[1:])
        )
        tables.append(RuledTable(area, cells))
    return tuple(tables)


def banded_tables(
    horizontal: Sequence[pymupdf.Rect],
    words: Sequence[_TextWord],
    known: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """Three-line tables: the top rule, the middle rule below the header and the bottom rule have the same length, with
    no vertical rules. The header often wraps and is centered, groups of data are separated by further rules, and the
    row spacing is uneven, so a search by alignment finds only one section. With three or more horizontal rules whose
    ends align and mostly short cells between them, the table is split into sections by the rules, and each section
    into rows and cells by the text. A table already found inside this area is replaced by this one; an area inside an
    existing table is not counted again."""
    rules = sorted((rule for rule in horizontal if rule.width >= _BAND_MIN_W), key=lambda rule: rule.y0)
    tables: list[RuledTable] = []
    used: set[int] = set()
    for index, rule in enumerate(rules):
        if index in used:
            continue
        group: list[pymupdf.Rect] = []
        for other_index, other in enumerate(rules):
            if abs(other.x0 - rule.x0) <= _NEAR and abs(other.x1 - rule.x1) <= _NEAR:
                used.add(other_index)
                if not group or other.y0 - group[-1].y0 > _NEAR:
                    group.append(other)
        if len(group) < 3:
            continue
        left, right = rule.x0, rule.x1
        bands: list[list[list[_TextWord]]] = []
        for upper, lower in zip(group, group[1:]):
            inside = [
                word
                for word in words
                if upper.y1 < (word.y0 + word.y1) / 2 < lower.y0
                and word.x0 >= left - _NEAR
                and word.x1 <= right + _NEAR
            ]
            bands.append(_word_rows(inside))
        rows = [row for band in bands for row in band]
        if len(rows) < _MIN_TEXT_ROWS or any(not band for band in bands):
            continue
        # Rules with a caption between them enclose a figure (grammar rules, example boxes), and the caption is
        # translated as usual
        if any(_LABEL_RE.match(" ".join(word.text for word in row)) for row in rows):
            continue
        # Backed by three rules, a lower share of short cells is enough: column names in the header and expressions
        # such as O(k·n·d) are longer than data cells.
        # A line of body text or pseudocode forms one cell spanning more than half the table
        pieces = [cell for row in rows for cell in _row_cells(row)]
        short = sum(1 for cell in pieces if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
        # A first row below a rule with a single short cell is a group name (Conditional, μ prediction) and is not
        # counted as a row without cells
        split = sum(
            1
            for band in bands
            for index, row in enumerate(band)
            if len(cells := _row_cells(row)) >= 2
            or (index == 0 and cells[0].words <= _SHORT_WORDS and cells[0].width <= _SHORT_WIDTH)
        )
        # A cell spanning more than half the table indicates body text only when it is the only cell in its row; row
        # names of a two-column table take up most of the width anyway
        if (
            short < len(pieces) * _BAND_SHORT_SHARE
            or split < len(rows) * 0.8
            or any(len(cells := _row_cells(row)) == 1 and cells[0].width > (right - left) / 2 for row in rows)
        ):
            continue
        area = pymupdf.Rect(left, group[0].y0, right, group[-1].y1)
        if any(_mostly_inside(area, [table.rect]) for table in known):
            continue
        cells: list[pymupdf.Rect] = []
        for (upper, lower), band in zip(zip(group, group[1:]), bands):
            tops = [min(w.y0 for w in row) for row in band]
            edges = [upper.y1, *((max(w.y1 for w in row) + top) / 2 for row, top in zip(band, tops[1:])), lower.y0]
            for row, y0, y1 in zip(band, edges, edges[1:]):
                boxes = _row_cells(row)
                cuts = [left, *((a.x1 + b.x0) / 2 for a, b in zip(boxes, boxes[1:])), right]
                cells.extend(pymupdf.Rect(x0, y0, x1, y1) for x0, x1 in zip(cuts, cuts[1:]))
        tables.append(RuledTable(area, tuple(cells)))
    return tuple(tables)


def _joined_verticals(vertical: Sequence[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Join vertical segments that meet end to end at the same x coordinate. Table rules drawn row by row have one
    segment per cell."""
    out: list[pymupdf.Rect] = []
    for seg in sorted(vertical, key=lambda r: (round(r.x0), r.y0)):
        last = out[-1] if out else None
        if last is not None and abs(last.x0 - seg.x0) < 1 and seg.y0 - last.y1 <= _SNAP:
            out[-1] = pymupdf.Rect(last.x0, last.y0, last.x1, max(last.y1, seg.y1))
        else:
            out.append(pymupdf.Rect(seg))
    return out


def _with_text_rows(table: RuledTable, words: Sequence[_TextWord]) -> RuledTable:
    """In a table with only column lines and no rules between rows, one cell holds several stacked lines. When the
    cells of one band have text in lines aligned across them, each line spanning two or more cells, a row name on every
    line and mostly short cells, the lines are unruled rows and are cut at the gaps between them. A wrapped header
    often has cells with only one line, a wrapped sentence stays within one cell, and long sentences do not count as
    short cells."""
    bands: dict[tuple[int, int], list[pymupdf.Rect]] = {}
    for cell in table.cells:
        bands.setdefault((round(cell.y0), round(cell.y1)), []).append(cell)
    cells: list[pymupdf.Rect] = []
    for band in bands.values():
        cells.extend(_split_band(band, words) if len(band) >= 2 else band)
    if len(cells) == len(table.cells):
        return table
    return RuledTable(table.rect, tuple(cells), table.algorithm)


def _split_band(band: Sequence[pymupdf.Rect], words: Sequence[_TextWord]) -> list[pymupdf.Rect]:
    def holder(word: _TextWord) -> int | None:
        center = pymupdf.Point((word.x0 + word.x1) / 2, (word.y0 + word.y1) / 2)
        return next((i for i, cell in enumerate(band) if cell.contains(center)), None)

    inside = [word for word in words if holder(word) is not None]
    rows = _word_rows(inside)
    if len(rows) < 2:
        return list(band)
    # The leftmost column with text holds the row names
    names = min({i for i in map(holder, inside) if i is not None}, key=lambda i: band[i].x0)
    pieces: list[_TextCell] = []
    for row in rows:
        by_cell: dict[int | None, list[_TextWord]] = {}
        for word in row:
            by_cell.setdefault(holder(word), []).append(word)
        if len(by_cell) < 2 or names not in by_cell:
            return list(band)
        for part in by_cell.values():
            pieces.extend(_row_cells(part))
    short = sum(1 for piece in pieces if piece.words <= _SHORT_WORDS and piece.width <= _SHORT_WIDTH)
    if short < len(pieces) * _SHORT_SHARE:
        return list(band)
    cuts = [
        (max(word.y1 for word in upper) + min(word.y0 for word in lower)) / 2 for upper, lower in zip(rows, rows[1:])
    ]
    out: list[pymupdf.Rect] = []
    for cell in band:
        edges = [cell.y0, *(y for y in cuts if cell.y0 < y < cell.y1), cell.y1]
        out.extend(pymupdf.Rect(cell.x0, top, cell.x1, bottom) for top, bottom in zip(edges, edges[1:]))
    return out


def _with_ruled_heads(
    tables: Sequence[RuledTable],
    words: Sequence[_TextWord],
    horizontal: Sequence[pymupdf.Rect],
) -> tuple[RuledTable, ...]:
    """The header of a three-line table sits between the top rule and the middle rule. Header text is often wider than
    the data columns and centered, so its left and right edges do not match the body, and a table found by alignment
    holds only the body. When a middle rule across the whole table lies right above the body and a top rule of the
    same length lies further up, the text between the two rules is the header; it is split into cells at the gaps
    between columns and added to the table."""
    out: list[RuledTable] = []
    for table in tables:
        rect = table.rect
        row_height = min(cell.height for cell in table.cells)
        spans = [h for h in horizontal if h.x0 <= rect.x0 + _NEAR and h.x1 >= rect.x1 - _NEAR]
        mids = [h for h in spans if h.y1 <= rect.y0 + _NEAR and rect.y0 - h.y0 <= row_height]
        if not mids:
            out.append(table)
            continue
        mid = max(mids, key=lambda h: h.y0)
        tops = [
            h
            for h in spans
            if abs(h.x0 - mid.x0) <= _NEAR
            and abs(h.x1 - mid.x1) <= _NEAR
            and row_height * 0.5 < mid.y0 - h.y0 <= row_height * _HEAD_ROWS
        ]
        if not tops:
            out.append(table)
            continue
        top = max(tops, key=lambda h: h.y0)
        band = [
            word
            for word in words
            if top.y1 < (word.y0 + word.y1) / 2 < mid.y0 and word.x0 >= mid.x0 - _NEAR and word.x1 <= mid.x1 + _NEAR
        ]
        heads = _head_cells([_row_cells(row) for row in _word_rows(band)], mid.x0, mid.x1, top.y1, mid.y0)
        if len(heads) < 2:
            out.append(table)
            continue
        grown = pymupdf.Rect(min(rect.x0, mid.x0), top.y0, max(rect.x1, mid.x1), rect.y1)
        out.append(RuledTable(grown, (*heads, *table.cells), table.algorithm))
    return tuple(out)


def _head_cells(
    rows: Sequence[Sequence[_TextCell]],
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> list[pymupdf.Rect]:
    """Split header text into cells by column. Text right below the nearest cell above it, when that cell overlaps
    only this one piece of text, is a wrapped line of the same cell; a group name spanning two columns is a cell of its
    own. Cells extend left and right to the middle of the gap to the neighboring cell, and up and down to the bounds
    of their header section."""
    groups: list[pymupdf.Rect] = []
    for row in rows:
        boxes = [pymupdf.Rect(cell.x0, cell.y0, cell.x1, cell.y1) for cell in row]
        start = len(groups)
        for box in boxes:
            hits = [g for g in groups[:start] if g.x0 < box.x1 and box.x0 < g.x1]
            near = max(hits, key=lambda g: g.y1, default=None)
            if near is not None and sum(near.x0 < b.x1 and b.x0 < near.x1 for b in boxes) == 1:
                near |= box
            else:
                groups.append(box)
    cells: list[pymupdf.Rect] = []
    for box in groups:
        side = [g for g in groups if g is not box and g.y0 < box.y1 and box.y0 < g.y1]
        x0 = max([left, *((g.x1 + box.x0) / 2 for g in side if g.x1 <= box.x0)])
        x1 = min([right, *((box.x1 + g.x0) / 2 for g in side if g.x0 >= box.x1)])
        y0 = max([top, *((g.y1 + box.y0) / 2 for g in groups if g.y1 <= box.y0 and g.x0 < box.x1 and box.x0 < g.x1)])
        y1 = min([bottom, *((box.y1 + g.y0) / 2 for g in groups if g.y0 >= box.y1 and g.x0 < box.x1 and box.x0 < g.x1)])
        cells.append(pymupdf.Rect(x0, y0, x1, y1))
    return cells


def _header_row(row: Sequence[_TextWord], lines: Sequence[float]) -> bool:
    """Text joined by normal word spacing is a sentence (the last line of a paragraph, Levenshtein in Eq. 2.20.);
    header columns are separated by gaps."""
    if len(_row_cells(row)) < 2:
        return False
    if any(word.x0 < x - 1 and word.x1 > x + 1 for word in row for x in lines):
        return False
    columns: dict[int, int] = {}
    for word in row:
        column = bisect.bisect(lines, (word.x0 + word.x1) / 2)
        columns[column] = columns.get(column, 0) + 1
    return len(columns) >= 3 and max(columns.values()) <= _SHORT_WORDS


def filled_tables(
    page: pymupdf.Page,
    drawings: list[dict],
    words: Sequence[_TextWord],
    ruled: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """Tables with one shaded shape per cell, the cells tiling an area (LaTeX \\rowcolor, \\cellcolor). Such tables
    often have only a few horizontal rules and no vertical ones, so the rules yield no cells and the text of all cells
    would be merged into one paragraph for translation.

    The shaded shapes must form at least two rows and two columns and tile the area, at least two cells must contain
    text, and all text in the area must fall into some cell. One shaded shape for the whole area is a text box, and
    one shape per line is the shading of a code box; neither counts. White cells laid out in a figure with labels
    spanning several cells do not count either. Tables already found by rules are not counted again.
    """
    bounds = text_page_rect(page)
    fills = [
        pymupdf.Rect(drawing["rect"])
        for drawing in drawings
        if light_fill(drawing)
        and min(drawing["fill"]) < 0.97
        and not (
            drawing["rect"].width > bounds.width * _PAGE_FILL and drawing["rect"].height > bounds.height * _PAGE_FILL
        )
    ]
    groups: list[list[pymupdf.Rect]] = []
    for fill in fills:
        near = [group for group in groups if any((fill + (-_NEAR, -_NEAR, _NEAR, _NEAR)).intersects(r) for r in group)]
        merged = [fill]
        for group in near:
            groups.remove(group)
            merged.extend(group)
        groups.append(merged)
    tables: list[RuledTable] = []
    for cells in groups:
        area = pymupdf.Rect(cells[0])
        for cell in cells[1:]:
            area |= cell
        if any(_mostly_inside(area, [table.rect]) for table in ruled):
            continue
        rows = _edges([cell.y0 for cell in cells])
        cols = _edges([cell.x0 for cell in cells])
        if len(cols) == 1 and _spaced_rows(cells, words):
            tables.append(RuledTable(area, tuple(sorted(cells, key=lambda c: c.y0))))
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        # Cells must not overlap and together must tile the area: nested shading (a shape inside a box) adds extra
        # area
        covered = _tiled_area(cells)
        if covered is None or abs(covered - area.get_area()) > area.get_area() * 0.05:
            continue
        boxes = [pymupdf.Rect(w.x0, w.y0, w.x1, w.y1) for w in words]
        boxes = [box for box in boxes if _mostly_inside(box, [area])]
        holders = [[cell for cell in cells if _holds(cell, box)] for box in boxes]
        if not all(holders):
            continue
        if len({id(found[0]) for found in holders}) >= 2:
            tables.append(RuledTable(area, tuple(sorted(cells, key=lambda c: (c.y0, c.x0)))))
    return tuple(tables)


def _tiled_area(cells: Sequence[pymupdf.Rect]) -> float | None:
    """Total area of the shaded shapes. Adjacent cells overlap in a strip along their shared edge (LaTeX paints a
    little beyond each cell so that no white seam shows between cells), and that strip is subtracted. When most of
    one shape lies on another, it is a shape inside a box, and None is returned."""
    total = sum(cell.get_area() for cell in cells)
    for index, cell in enumerate(cells):
        for other in cells[index + 1 :]:
            hit = cell & other
            if hit.is_empty:
                continue
            if hit.get_area() > min(cell.get_area(), other.get_area()) * 0.5:
                return None
            total -= hit.get_area()
    return total


def _spaced_rows(cells: Sequence[pymupdf.Rect], words: Sequence[_TextWord]) -> bool:
    """A one-column table shaded row by row (\\rowcolor): three or more rows, one line of text per row, each split into
    parts by large gaps (example sentence, blank line to fill in, answer). The shading of a code box also has one
    shape per line, but each line is one continuous sentence."""
    if len(cells) < _MIN_TEXT_ROWS:
        return False
    for cell in cells:
        inside = [word for word in words if _holds(cell, pymupdf.Rect(word.x0, word.y0, word.x1, word.y1))]
        rows = _word_rows(inside)
        if len(rows) != 1 or len(_row_cells(rows[0])) < 2:
            return False
    return True


def _holds(cell: pymupdf.Rect, box: pymupdf.Rect) -> bool:
    """The word lies entirely in the cell, sticking out only slightly left or right. A word spanning two cells shows
    that the shading is not laid out by cells."""
    middle = (box.y0 + box.y1) / 2
    return cell.x0 - _NEAR <= box.x0 and box.x1 <= cell.x1 + _NEAR and cell.y0 <= middle <= cell.y1


_SYMBOL_FONT_RE = re.compile(r"(?:^|\+)(?:CMSY|CMEX|MSAM|MSBM)", re.I)


_ALGORITHM_RE = re.compile(r"^\s*(?:Algorithm|Procedure|算法)\s*\d+", re.I)


_ALGORITHM_WORD_RE = re.compile(r"Algorithm|Procedure|算法", re.I)


def algorithm_tables(
    page: pymupdf.Page,
    drawings: list[dict] | None = None,
    words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """Treat pseudocode in an algorithm box as a one-column table with one cell per line.

    Pseudocode has one statement per line and indentation is structure; the lines do not form a paragraph. PyMuPDF
    often puts several lines into one block, and later merging would join them into a reflowed paragraph. As cells,
    each line is translated on its own, written back to its original line and not merged with other lines. The extent
    of an algorithm box is set by its frame rules: from the rule below the caption to the next rule below it.
    words holds the words of this page. A page where no word contains Algorithm, Procedure or "算法" has no algorithm
    caption, so the text of the whole page need not be read again.
    """
    if words is not None and not any(_ALGORITHM_WORD_RE.search(word.text) for word in words):
        return ()
    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return ()
    lines = [
        line
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        if abs(line["dir"][1]) <= 0.01
    ]
    captions = [caption for line in lines for caption in _algorithm_captions(line)]
    if not captions:
        return ()
    horizontal, _vertical = _rule_segments(page, drawings)
    tables: list[RuledTable] = []
    for box, size in captions:
        body = _box_below(box, size, horizontal) or _box_above(box, size, horizontal)
        if body is None:
            continue
        rows = _code_rows(lines, body, horizontal)
        if len(rows) >= 2:
            tables.append(RuledTable(body, rows, algorithm=True))
    return tuple(tables)


def _algorithm_captions(line: dict) -> list[tuple[pymupdf.Rect, float]]:
    """Algorithm captions in one line (box, font size). Two algorithm boxes side by side often have their captions in
    the same line, separated by a column gap."""
    spans = [span for span in line["spans"] if span["text"].strip()]
    starts: list[int] = []
    for index, span in enumerate(spans):
        if index and pymupdf.Rect(span["bbox"]).x0 - pymupdf.Rect(spans[index - 1]["bbox"]).x1 < span["size"] * 2:
            continue
        if _ALGORITHM_RE.match("".join(item["text"] for item in spans[index:])):
            starts.append(index)
    found: list[tuple[pymupdf.Rect, float]] = []
    for begin, end in zip(starts, [*starts[1:], len(spans)]):
        box = pymupdf.Rect()
        for span in spans[begin:end]:
            box |= pymupdf.Rect(span["bbox"])
        found.append((box, max(span["size"] for span in spans[begin:end])))
    return found


def _box_below(caption: pymupdf.Rect, size: float, horizontal: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """Caption on top: the first rule below the caption is the top of the box, the next one below is the bottom. The
    caption may have several lines."""
    rules = sorted(
        (rule for rule in horizontal if rule.y0 >= caption.y1 - _NEAR and _under(rule, caption, size)),
        key=lambda rule: rule.y0,
    )
    rules = _distinct_rules(rules)
    if len(rules) < 2 or rules[0].y0 - caption.y1 > size * 8:
        return None
    top, bottom = rules[0], rules[1]
    return pymupdf.Rect(top.x0, top.y0, top.x1, bottom.y0)


def _box_above(caption: pymupdf.Rect, size: float, horizontal: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """Caption at the bottom: the rule right above the caption is the bottom of the box, the next rule of the same
    width further up is the top."""
    rules = sorted(
        (rule for rule in horizontal if rule.y1 <= caption.y0 + _NEAR and _under(rule, caption, size)),
        key=lambda rule: -rule.y0,
    )
    rules = _distinct_rules(rules)
    if len(rules) < 2 or caption.y0 - rules[0].y1 > size * 2:
        return None
    bottom = rules[0]
    top = next(
        (rule for rule in rules[1:] if abs(rule.x0 - bottom.x0) < _NEAR and abs(rule.x1 - bottom.x1) < _NEAR), None
    )
    if top is None:
        return None
    return pymupdf.Rect(bottom.x0, top.y1, bottom.x1, bottom.y0)


_LINE_LABEL_RE = re.compile(r"^\s*\d+\s*[:.]")


# Line ends of an unfinished sentence: a lowercase letter, a comma, a hyphen of a broken word
_OPEN_END_RE = re.compile(r"[a-z,]$|[a-z]-$")


def _join_wrapped_rows(
    rows: list[pymupdf.Rect],
    baselines: list[float],
    spans: list[dict],
    main: float,
) -> list[pymupdf.Rect]:
    """In an algorithm with line numbers at the line starts (1: 2: ...), a line without a number after the numbered
    lines is a wrapped part of the previous statement and joins the previous cell. In an algorithm without line
    numbers, when the previous line reaches the right edge of the box and its sentence is unfinished, the next line is
    its wrapped part.

    If the two halves were sent for translation separately, a half sentence would have too few body-text words and
    would be kept in English as a formula. Input: and Output: before the first line number are one cell each."""
    labeled: list[bool] = []
    full: list[bool] = []
    for baseline in baselines:
        row = sorted(
            (span for span in spans if abs(span["origin"][1] - baseline) <= main * 0.4),
            key=lambda span: span["origin"][0],
        )
        text = "".join(span["text"] for span in row).rstrip()
        labeled.append(bool(_LINE_LABEL_RE.match(text)))
        # A full line has even spacing; a right-aligned comment (// mean) also reaches the right edge, but with a wide
        # gap after the statement
        boxes = [span["bbox"] for span in row if "bbox" in span and span["text"].strip()]
        right = max((box[2] for box in boxes), default=0.0)
        gaps = [b[0] - a[2] for a, b in zip(boxes, boxes[1:])]
        full.append(
            bool(rows)
            and right >= rows[0].x1 - main * 0.5
            and max(gaps, default=0.0) < main * 1.5
            and bool(_OPEN_END_RE.search(text))
        )
    if sum(labeled) < 2:
        out = []
        for index, row in enumerate(rows):
            if index and full[index - 1]:
                out[-1] = out[-1] | row
            else:
                out.append(row)
        return out
    first = labeled.index(True)
    out: list[pymupdf.Rect] = []
    for index, (row, has_label) in enumerate(zip(rows, labeled)):
        if index > first and not has_label:
            out[-1] = out[-1] | row
        else:
            out.append(row)
    return out


def _under(rule: pymupdf.Rect, caption: pymupdf.Rect, size: float) -> bool:
    """The rule starts at the left edge of the caption and covers the start of the caption. Rules of another
    algorithm box beside it do not count."""
    return rule.x0 <= caption.x0 + _NEAR and rule.x1 >= min(caption.x1, caption.x0 + size * 4)


def _distinct_rules(rules: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """One rule may be both a line segment and an edge of a box; of rules lying on top of each other only one is
    kept."""
    out: list[pymupdf.Rect] = []
    for rule in rules:
        if out and abs(rule.y0 - out[-1].y0) < _NEAR and abs(rule.x0 - out[-1].x0) < _NEAR:
            continue
        out.append(rule)
    return out


def _code_rows(
    lines: list[dict],
    body: pymupdf.Rect,
    rules: Sequence[pymupdf.Rect] = (),
) -> tuple[pymupdf.Rect, ...]:
    """One cell per line in the box. Lines are split by baselines at body-text size, and superscripts and subscripts
    fall into the line they belong to. rules holds the horizontal rules in the box: fraction bars and radical bars are
    short, frame lines and separators run across the whole box."""
    spans = [
        span
        for line in lines
        for span in line["spans"]
        if span["text"].strip() and body.contains(pymupdf.Point(span["origin"]))
    ]
    if not spans:
        return ()
    main = max(span["size"] for span in spans)
    bars = [rule for rule in rules if rule.width < body.width * 0.5 and body.intersects(rule)]
    baselines: list[float] = []
    # Symbol-font glyphs such as radicals and large braces do not have their origin on the baseline, so they are not
    # used to split lines; numerators and denominators of display formulas are as large as body text but sit against
    # a fraction bar, so they are not used either
    steady = [
        span
        for span in spans
        if span["size"] >= main * 0.85 and not _SYMBOL_FONT_RE.search(span["font"]) and not _on_bar(span, bars, main)
    ]
    for y in sorted(span["origin"][1] for span in steady):
        if not baselines or y - baselines[-1] > main * 0.4:
            baselines.append(y)
    if len(baselines) < 2:
        return ()
    # The boundary sits close to the upper baseline: subscripts and descenders are only slightly below the baseline,
    # while superscripts (the st of 1st) are raised high and would fall into the upper line with a boundary midway
    cuts = [body.y0, *(a + (b - a) * _ROW_CUT for a, b in zip(baselines, baselines[1:])), body.y1]
    rows = [pymupdf.Rect(body.x0, top, body.x1, bottom) for top, bottom in zip(cuts, cuts[1:])]
    return tuple(_join_wrapped_rows(rows, baselines, spans, main))


def _on_bar(span: dict, bars: Sequence[pymupdf.Rect], size: float) -> bool:
    """The span sits right above or below a short horizontal rule and within its horizontal extent: a numerator, a
    denominator, or text under a radical."""
    if not bars:
        return False
    box = pymupdf.Rect(span["bbox"])
    for bar in bars:
        if bar.x0 - 1 > box.x0 or box.x1 > bar.x1 + 1:
            continue
        if -1 <= bar.y0 - box.y1 <= size * 0.3 or -1 <= box.y0 - bar.y1 <= size * 0.3:
            return True
    return False


@dataclass(frozen=True)
class _TextCell:
    x0: float
    y0: float
    x1: float
    y1: float
    words: int
    text: str = ""

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def unruled_tables(
    page: pymupdf.Page,
    ruled: Sequence[RuledTable] = (),
    words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """Without rules, only grids of short cells count: column left or right edges align and row heights repeat. words
    holds the words of this page; read now when not given.

    Words at word spacing are merged into one cell. Long lines across a wide column gap are body text, not tables.
    Tables with empty cells, or with two adjacent columns so close that they merge into one cell, have a different
    number of cells per row and are not detected by row alignment. The remaining rows are checked again by the columns
    that the left edges of their cells fall on (see _sparse_grid).
    """
    zones = [table.rect for table in ruled]
    words = [
        word
        for word in (page_words(page) if words is None else words)
        if not _mostly_inside(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), zones)
    ]
    rows = [_row_cells(row) for row in _word_rows(words)]
    tables: list[RuledTable] = []
    index = 0
    while index < len(rows):
        run = [rows[index]]
        cursor = index + 1
        while cursor < len(rows) and _continues_grid(run, rows[cursor]):
            run.append(rows[cursor])
            cursor += 1
        table = _grid_table(run, text_page_rect(page).width)
        if table is not None:
            tables.append(table)
            index = cursor
        else:
            index += 1
    taken = [table.rect for table in tables]
    free = [
        not any(_mostly_inside(pymupdf.Rect(cell.x0, cell.y0, cell.x1, cell.y1), taken) for cell in row) for row in rows
    ]
    # On a two-column page a table in the left column sits next to body text in the right column, and column
    # alignment would join them into one table, so the two sides of the column gap are detected separately
    width = text_page_rect(page).width
    gutters = _find_gutters(
        [
            pymupdf.Rect(cell.x0, cell.y0, cell.x1, cell.y1)
            for row in rows
            for cell in row
            if cell.x1 - cell.x0 <= width * _SPAN_MAX
        ]
    )
    index = 0
    while index < len(rows):
        cursor = index
        while (
            cursor < len(rows)
            and free[cursor]
            and (cursor == index or _continues_rows(rows[index:cursor], rows[cursor]))
            and len(rows[cursor]) >= 2
        ):
            cursor += 1
        table = _sparse_grid(rows[index:cursor], gutters) if cursor - index >= _MIN_TEXT_ROWS else None
        if table is not None:
            tables.append(table)
            index = cursor
        else:
            index += 1
    return tuple(tables)


def _continues_rows(run: Sequence[Sequence[_TextCell]], row: Sequence[_TextCell]) -> bool:
    """The row spacing matches the rows before. The number of cells does not matter"""
    pitch = row[0].y0 - run[-1][0].y0
    first = run[1][0].y0 - run[0][0].y0 if len(run) > 1 else pitch
    return _MIN_PITCH <= pitch <= _MAX_PITCH and abs(pitch - first) <= max(4.0, first * 0.3)


# Page numbers at the end of an index entry: hallucinate, 31, 253
_PAGE_REFS_RE = re.compile(r",\s*\d+$")


def _sparse_grid(
    rows: Sequence[Sequence[_TextCell]],
    gutters: Sequence[tuple[float, float]] = (),
) -> RuledTable | None:
    """Tables with a different number of cells per row (three columns such as TMP when? yesterday evening, with an
    empty cell in some rows): the left edge of every cell falls on some column, and each column is used by at least
    two rows; three or more columns are each used by half of the rows, and adjacent columns are well apart.
    The last column may hold longer example sentences or explanations, the columns before it short cells of one or two
    words. Index subentries are indented only slightly more than entries, so the columns are too close and do not
    count; index entries in several columns ending with page numbers (hallucinate, 31, 253) do not count either.
    A first or last row with long sentences in the earlier columns (a caption below the table) is not part of the
    table. Two adjacent columns separated by the page's column gap hold the text of two separate columns and do not
    count."""
    flat = [cell for row in rows for cell in row]
    if sum(1 for cell in flat if _PAGE_REFS_RE.search(cell.text)) >= len(flat) * 0.3:
        return None
    heads = [cell for row in rows for cell in row[:-1]]
    short = sum(1 for cell in heads if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
    if not heads or short < len(heads) * _SHORT_SHARE:
        return None
    columns: list[list[tuple[int, _TextCell]]] = []
    for row_index, cell in sorted(
        ((index, cell) for index, row in enumerate(rows) for cell in row),
        key=lambda item: item[1].x0,
    ):
        if columns and cell.x0 - columns[-1][-1][1].x0 <= _COL_ALIGN:
            columns[-1].append((row_index, cell))
        else:
            columns.append([(row_index, cell)])
    used = [len({row_index for row_index, _ in column}) for column in columns]
    if min(used) < 2 or sum(1 for count in used if count * 2 >= len(rows)) < 3:
        return None
    anchors = [min(cell.x0 for _, cell in column) for column in columns]
    if any(right - left < _COL_ALIGN * 3 for left, right in zip(anchors, anchors[1:])):
        return None
    if any(
        left < start and end <= right + _COL_ALIGN
        for left, right in zip(anchors, anchors[1:])
        for start, end in gutters
    ):
        return None

    # A caption below the table (Figure 18.12 The A transition ...) has the same row spacing and also falls on the
    # columns, but its long sentence is not in the last column
    def long_head(row: Sequence[_TextCell]) -> bool:
        return any(cell.x0 < anchors[-1] - _COL_ALIGN and cell.words > _SHORT_WORDS for cell in row)

    if long_head(rows[0]) or long_head(rows[-1]):
        inner = rows[1 if long_head(rows[0]) else 0 : -1 if long_head(rows[-1]) else None]
        return _sparse_grid(inner, gutters) if len(inner) >= _MIN_TEXT_ROWS else None
    rights = [anchor - 3.0 for anchor in anchors[1:]]
    rights.append(max(max(cell.x1 for _, cell in columns[-1]) + 8.0, anchors[-1] + 28.0))
    tops = [min(cell.y0 for cell in row) for row in rows]
    bottoms = [top - 1.0 for top in tops[1:]] + [max(cell.y1 for cell in rows[-1]) + 2.0]
    cells = tuple(
        pymupdf.Rect(anchor - 1.0, top - 1.0, max(right, anchor + 18.0), bottom)
        for top, bottom in zip(tops, bottoms)
        for anchor, right in zip(anchors, rights)
    )
    if not _usable_table(cells):
        return None
    rect = pymupdf.Rect()
    for cell in cells:
        rect |= cell
    return RuledTable(rect, cells)


def interlinear_glosses(
    page: pymupdf.Page,
    words: Sequence[_TextWord] | None = None,
) -> tuple[pymupdf.Rect, ...]:
    """Interlinear glosses of linguistic examples: one line of original text, one line of gloss. Translating the gloss
    line word by word makes no sense, and any change of word order breaks the alignment, so both lines keep the
    original text. Two arrangements are detected:

    - Word-by-word alignment: every word of the lower line shares its left edge with some word of the upper line.
      Adjacent lines of body text have only a few words aligned by chance, so at least 75% of the words of each line
      must align. With only two or three words, the upper line must also be in a non-Latin script such as Chinese
      characters or kana, and the lower line in Latin letters.
    - Morpheme-by-morpheme gloss: both lines start at the same x, both consist of morphemes joined by hyphens, and
      the gloss line contains grammatical abbreviations such as 1SG and PFV (Leipzig Glossing Rules).
    - Character-by-character transcription: the upper line is Chinese characters and the lower line starts at the
      same x, consists only of Latin letters and has as many syllables as there are characters. Chinese characters
      have no spaces between them, so word-by-word alignment is impossible.
    - Role labels: the lower line consists of all-capital labels one size smaller (AGENT, ARG0, ARGM-TMP), each
      starting below some word of the upper line. A label refers to a whole phrase and aligns only with its first
      word.

    Tables align like this over many consecutive rows, while a gloss is an isolated pair of lines. words holds the
    words of this page; read now when not given.
    """
    words = list(page_words(page) if words is None else words)
    rows = _with_centered_numbers(_word_rows(words))
    pairs = [_glossed(upper, lower) for upper, lower in zip(rows, rows[1:])]
    found: list[pymupdf.Rect] = []
    for index, paired in enumerate(pairs):
        if not paired or (index > 0 and pairs[index - 1]) or (index + 1 < len(pairs) and pairs[index + 1]):
            continue
        rect = pymupdf.Rect()
        for word in (*rows[index], *rows[index + 1]):
            rect |= pymupdf.Rect(word.x0, word.y0, word.x1, word.y1)
        found.append(rect)
    return tuple(found)


def _with_centered_numbers(rows: list[list[_TextWord]]) -> list[list[_TextWord]]:
    """An example number (22.27) set level with the middle between the example and the label line below it forms a
    line of its own between the two. Merging it into the upper line puts the example and its labels next to each
    other."""
    out: list[list[_TextWord]] = []
    for index, row in enumerate(rows):
        below = rows[index + 1] if index + 1 < len(rows) else None
        if (
            out
            and below is not None
            and len(row) == 1
            and _NUMBER_LABEL_RE.fullmatch(row[0].text)
            and row[0].x1 < min(word.x0 for word in (*out[-1], *below))
            and row[0].y0 < max(word.y1 for word in out[-1])
            and row[0].y1 > min(word.y0 for word in below)
        ):
            out[-1] = [*out[-1], *row]
            continue
        out.append(row)
    return out


@dataclass(frozen=True)
class ExampleNumber:
    """An example number at the start of a line. rect is the number, row is the line of the number, text_x0 is
    where the text after the number starts"""

    rect: pymupdf.Rect
    row: pymupdf.Rect
    text_x0: float


def example_numbers(words: Sequence[_TextWord]) -> tuple[ExampleNumber, ...]:
    """Example numbers at the start of a line: (22.9), (22.11). A number has few letters, so extraction often leaves
    it outside any block while the sentence after it forms a block of its own, or the whole line ((22.12) fall.01)
    forms no block. Numbers with other text right to their left do not count (numbers cited in body text)."""
    found: list[ExampleNumber] = []
    for row in _word_rows(words):
        for index, word in enumerate(row):
            if not _CHAPTER_NUMBER_RE.fullmatch(word.text):
                continue
            height = word.y1 - word.y0
            if index and word.x0 - row[index - 1].x1 < height * 2:
                continue
            rect = pymupdf.Rect(word.x0, word.y0, word.x1, word.y1)
            for other in row[index + 1 :]:
                if other.x0 - rect.x1 > height * 3:
                    break  # another column
                rect |= pymupdf.Rect(other.x0, other.y0, other.x1, other.y1)
            text_x0 = row[index + 1].x0 if index + 1 < len(row) else word.x1
            found.append(ExampleNumber(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), rect, text_x0))
    return tuple(found)


def _glossed(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    height = max(word.y1 - word.y0 for word in upper)
    if lower[0].y0 - upper[0].y0 > height * 1.6:
        return False
    starts = [word.x0 for word in upper]
    # A gloss line consists of words, not a row of a table of numbers
    aligned = [
        word
        for word in lower
        if any(abs(word.x0 - x) <= _GLOSS_ALIGN for x in starts) and any(ch.isalpha() for ch in word.text)
    ]
    hit = [x for x in starts if any(abs(word.x0 - x) <= _GLOSS_ALIGN for word in lower)]
    # When most aligned words equal the word at the same place in the upper line, these are two list items with the
    # same start (Else search from right to left for ...); a gloss line is in another language, so the word at the
    # same place differs
    same = [
        word
        for word in aligned
        if any(abs(word.x0 - other.x0) <= _GLOSS_ALIGN and word.text == other.text for other in upper)
    ]
    if aligned and len(same) >= len(aligned) * 0.5:
        return False
    if len(aligned) >= len(lower) * _GLOSS_SHARE and len(hit) >= len(upper) * _GLOSS_SHARE:
        if len(aligned) >= _GLOSS_MIN:
            return True
        if len(aligned) >= 2 and _script_share(upper, latin=False) >= 0.5 and _script_share(lower, latin=True) == 1:
            return True
    return _morpheme_gloss(upper, lower) or _romanized(upper, lower) or _role_labels(upper, lower)


def _role_labels(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    """Above a label is a word one size larger, and the label lies entirely right below that word. These do not
    count: abbreviations in side notes (DFT, RNN), with side-note words of the same small size above them; subscripts
    (CLS below h), which overlap the line above; capitals in figures, with numbers above them. The size is compared
    with the shortest label in the row, since the subscript of ARG0 makes its box taller."""
    height = min(word.y1 - word.y0 for word in lower)
    return all(
        _ROLE_LABEL_RE.fullmatch(word.text) and any(_labels(word, other, height) for other in upper) for word in lower
    )


def _labels(label: _TextWord, word: _TextWord, height: float) -> bool:
    return (
        word.x0 - _GLOSS_ALIGN * 2 <= label.x0 <= word.x1
        and height < (word.y1 - word.y0) * 0.9
        and word.y1 - 1 <= label.y0 <= word.y1 + height * 0.35
        and any(ch.isalpha() for ch in word.text)
    )


def _romanized(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    if abs(lower[0].x0 - upper[0].x0) > _GLOSS_ALIGN or _script_share(lower, latin=True) < 1:
        return False
    text = "".join(word.text for word in upper)
    hanzi = len(_HANZI_RE.findall(text))
    if hanzi < _GLOSS_MIN or hanzi < len(re.sub(r"[\s\W\d_]", "", text)) * 0.9:
        return False
    syllables = [token for token in _SYLLABLE_SPLIT_RE.split(" ".join(word.text for word in lower)) if token]
    return len(syllables) == hanzi


def _script_share(row: Sequence[_TextWord], latin: bool) -> float:
    """Share of Latin-letter words (latin=True) or of non-Latin-script words among the words with letters in this row.
    Pinyin tone marks in PDFs are often extracted as separate modifier letters (wˇo); characters in that range also
    count as Latin letters."""
    lettered = [word for word in row if any(ch.isalpha() for ch in word.text)]
    if not lettered:
        return 0.0
    matched = [
        word
        for word in lettered
        if all(ord(ch) <= 0x24F or 0x2B0 <= ord(ch) <= 0x2FF for ch in word.text if ch.isalpha()) == latin
    ]
    return len(matched) / len(lettered)


def _morpheme_gloss(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    chains = [word for word in lower if word.text.count("-") >= 2]
    if not chains or sum(len(word.text) for word in chains) < sum(len(word.text) for word in lower) * 0.6:
        return False
    first = min(chains, key=lambda word: word.x0)
    if not any(abs(word.x0 - first.x0) <= _GLOSS_ALIGN and word.text.count("-") >= 2 for word in upper):
        return False
    tokens = _MORPHEME_SPLIT_RE.split(" ".join(word.text for word in chains))
    return any(token in _LEIPZIG or _PERSON_RE.fullmatch(token) for token in tokens)


def _word_rows(words: Sequence[_TextWord]) -> list[list[_TextWord]]:
    rows: list[list[_TextWord]] = []
    for word in sorted(words, key=lambda item: (item.y0, item.x0)):
        if rows and abs(word.y0 - rows[-1][0].y0) <= 4.0:
            rows[-1].append(word)
        else:
            rows.append([word])
    for row in rows:
        row.sort(key=lambda item: item.x0)
    return rows


def _row_cells(row: Sequence[_TextWord]) -> list[_TextCell]:
    if not row:
        return []
    groups: list[list[_TextWord]] = [[row[0]]]
    for word in row[1:]:
        if word.x0 - groups[-1][-1].x1 <= _TEXT_JOIN:
            groups[-1].append(word)
        else:
            groups.append([word])
    cells: list[_TextCell] = []
    for group in groups:
        cells.append(
            _TextCell(
                min(word.x0 for word in group),
                min(word.y0 for word in group),
                max(word.x1 for word in group),
                max(word.y1 for word in group),
                len(group),
                " ".join(word.text for word in group),
            )
        )
    return cells


def _continues_grid(run: Sequence[Sequence[_TextCell]], row: Sequence[_TextCell]) -> bool:
    if len(row) != len(run[0]) or len(row) < 2:
        return False
    pitch = row[0].y0 - run[-1][0].y0
    if pitch < _MIN_PITCH or pitch > _MAX_PITCH:
        return False
    pitches = [run[i + 1][0].y0 - run[i][0].y0 for i in range(len(run) - 1)]
    pitches.append(pitch)
    median = sorted(pitches)[len(pitches) // 2]
    if abs(pitch - median) > max(4.0, median * 0.3):
        return False
    return all(_edge_aligned(anchor, cell) for anchor, cell in zip(run[0], row))


def _edge_aligned(anchor: _TextCell, cell: _TextCell) -> bool:
    return abs(cell.x0 - anchor.x0) <= _COL_ALIGN or abs(cell.x1 - anchor.x1) <= _COL_ALIGN


def _grid_table(rows: Sequence[Sequence[_TextCell]], page_width: float) -> RuledTable | None:
    if len(rows) < _MIN_TEXT_ROWS or len(rows[0]) < 2:
        return None
    flat = [cell for row in rows for cell in row]
    short = sum(1 for cell in flat if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
    if short < len(flat) * _SHORT_SHARE:
        return None
    widths = sorted(cell.width for cell in flat)
    if widths[len(widths) // 2] > min(88.0, max(page_width, 1.0) * 0.28):
        return None
    cells = _grid_cells(rows)
    if not _usable_table(cells):
        return None
    rect = pymupdf.Rect()
    for cell in cells:
        rect |= cell
    return RuledTable(rect, cells)


def _grid_cells(rows: Sequence[Sequence[_TextCell]]) -> tuple[pymupdf.Rect, ...]:
    columns = len(rows[0])
    anchors = [min(row[col].x0 for row in rows) for col in range(columns)]
    rights: list[float] = []
    for col in range(columns - 1):
        rights.append(anchors[col + 1] - 3.0)
    last_text = max(row[-1].x1 for row in rows)
    rights.append(max(last_text + 8.0, anchors[-1] + 28.0))
    tops = [min(cell.y0 for cell in row) for row in rows]
    bottoms: list[float] = []
    for index in range(len(rows) - 1):
        bottoms.append(tops[index + 1] - 1.0)
    bottoms.append(max(cell.y1 for cell in rows[-1]) + 2.0)
    cells: list[pymupdf.Rect] = []
    for top, bottom, _row in zip(tops, bottoms, rows):
        for anchor, right in zip(anchors, rights):
            cells.append(pymupdf.Rect(anchor - 1.0, top - 1.0, max(right, anchor + 18.0), bottom))
    return tuple(cells)


def _usable_table(cells: tuple[pymupdf.Rect, ...]) -> bool:
    if len(cells) < 2:
        return False
    # Table cells do not overlap. Boxes inside boxes form a flowchart; treated as a table, all text in a box would be
    # merged into one cell
    for index, cell in enumerate(cells):
        for other in cells[index + 1 :]:
            hit = cell & other
            smaller = min(cell.width * cell.height, other.width * other.height)
            if not hit.is_empty and hit.width * hit.height > smaller * 0.25:
                return False
    widths = sorted(cell.width for cell in cells)
    heights = sorted(cell.height for cell in cells)
    if widths[len(widths) // 2] < _CELL_MIN_W or heights[len(heights) // 2] < _CELL_MIN_H:
        return False
    if sum(1 for cell in cells if cell.width < _CELL_NARROW) > len(cells) * 0.4:
        return False
    columns = len({round(cell.x0) for cell in cells})
    return columns <= _MAX_TABLE_COLS
