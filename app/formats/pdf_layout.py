"""Page geometry: find columns from the horizontal gaps between text blocks, and collect figures, vector graphics
and ruled tables as boundaries.

Does not depend on the layout model. Headings spanning columns (wider than 60% of the page width) take no part in
column detection. A column gap is at least 12pt wide, with three or more blocks on each side that overlap deeply
enough vertically. The gap need not sit in the middle of the page, so three columns can be separated too. When a
figure takes up the text on one side, the figure edge still acts as a column wall and writing does not cross it.
Tables with horizontal and vertical rules, and short-cell tables whose column edges align, limit writing to their
cells. Whole paragraphs of body text are not tables.
Running headers, footers and page numbers are skipped separately and take no part in column detection.
When the text column gaps are clear, the reading order runs column by column from top to bottom; pages split only by
figure edges are not reordered."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import pymupdf

from .pdf_drawings import (
    _GUTTER_MIN,
    _MIN_Y_OVERLAP,
    _NEAR,
    _PAGE_FILL,
    _RULE_THICK,
    _SPAN_MAX,
    RuledTable,
    _drawing_paths,
    _find_gutters,
    _group_y_overlap,
    _image_infos,
    _image_rules,
    _mostly_inside,
    _rule_segments,
    _TextWord,
    light_fill,
    page_words,
    text_page_rect,
    visible_drawings,
)
from .pdf_frame import TextFrame
from .pdf_tables import page_tables
from .pdf_types import Drawing, ImageInfo

_USABLE_W = 8.0


_USABLE_H = 3.0


_FIGURE_MIN = 8.0


_DRAW_MIN = 24.0


_DRAW_AREA = 800.0


_GROUND_SHARE = 2.0  # Maximum ratio of a figure's white ground to the area of the curves and images inside it


_GROUND_REACH = 24.0  # Maximum distance of titles and axis labels from the curves


_GROUND_MARKS = 3  # Minimum number of small marks on a white ground that holds no complete figure


_FRAME_REACH = 12.0  # Maximum distance of a frame line from the top or bottom edge of the white ground


_WALL_MIN_H = 48.0


_WALL_MIN_W = 36.0


_MARGIN_BAND = 0.07


_MARGIN_MIN = 42.0


_MARGIN_HEIGHT = 22.0


_MARGIN_LEN = 80


# A short line at the same height (within tolerance) as this many repeated running headers is also a running header
_MARGIN_ROW = 1.5


_MARGIN_ROWS = 3


_PAGE_NO_RE = re.compile(
    r"^(?:[-–—]\s*)?(?:(?:page|p\.)\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*页)?(?:\s*[-–—])?$"
    r"|^第\s*\d{1,4}\s*页$",
    re.I,
)


_ROMAN_RE = re.compile(
    r"^m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$",
    re.I,
)


@dataclass(frozen=True)
class Column:
    x0: float
    x1: float


@dataclass(frozen=True)
class PageGeometry:
    width: float
    columns: tuple[Column, ...]
    figures: tuple[pymupdf.Rect, ...]
    height: float = 0.0
    tables: tuple[RuledTable, ...] = ()
    # Figure edges can block writing, but with fewer than three text blocks on one side they cannot drive reordering.
    reads_in_columns: bool = False
    # Text frame voted across the whole document. The column's left edge is the leftmost block on this page, the
    # frame's left edge is where body text starts
    frame: TextFrame | None = None

    def index_of(self, rect: pymupdf.Rect) -> int:
        """The column the block's center falls in. A center in the gap between columns takes the nearest column."""
        center = (rect.x0 + rect.x1) / 2
        for index, column in enumerate(self.columns):
            if column.x0 - 1 <= center <= column.x1 + 1:
                return index
        return min(
            range(len(self.columns)),
            key=lambda index: _interval_distance(center, self.columns[index]),
        )

    def column_of(self, rect: pymupdf.Rect) -> Column:
        return self.columns[self.index_of(rect)]

    def spans_columns(self, rect: pymupdf.Rect) -> bool:
        """Headings and whole tables that span columns. Their center falls into some column, but they cannot serve as
        the start or end of a column."""
        return _spans_columns(rect, self)

    def cell_of(self, rect: pymupdf.Rect) -> pymupdf.Rect | None:
        return cell_containing(rect, self.tables)

    def right_limit(self, rect: pymupdf.Rect, obstacles: list[pymupdf.Rect]) -> float:
        """Right edge of the column, stopping before obstacles on the right that overlap vertically. It can be less
        than rect.x1; callers must not shrink the box because of that."""
        limit = self.column_of(rect).x1
        cell = self.cell_of(rect)
        if cell is not None:
            limit = min(limit, cell.x1 - 0.8)
        for obstacle in obstacles:
            if obstacle.x0 <= rect.x0 + 1:
                continue
            if _overlaps_y(rect, obstacle):
                limit = min(limit, obstacle.x0 - 2)
        return limit

    @staticmethod
    def from_rects(
        page_width: float,
        rects: list[pymupdf.Rect],
        figures: list[pymupdf.Rect] | None = None,
        height: float = 0.0,
        tables: Sequence[RuledTable] | None = None,
        frame: TextFrame | None = None,
    ) -> PageGeometry:
        table_tuple = tuple(tables or ())
        table_zones = [table.rect for table in table_tuple]
        # A table's outer frame is also a large vector shape and must not count as a figure, or the text in its cells
        # would be skipped as a whole.
        # A shape clearly larger than the table is a figure: a chart with a grid, where the grid was detected as a
        # table and the title and axis labels lie outside the grid
        figure_tuple = tuple(
            figure
            for figure in (figures or ())
            if not any(
                _mostly_inside(figure, [zone]) and figure.width * figure.height <= zone.width * zone.height * 1.2
                for zone in table_zones
            )
        )
        voting = _voting_rects(rects, figure_tuple, table_tuple, page_width, height)
        usable = [rect for rect in voting if rect.width >= _USABLE_W and rect.height >= _USABLE_H]
        body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
        columns = _columns_of(page_width, usable, _cutting_figures(figure_tuple, page_width, height))
        return PageGeometry(
            page_width,
            columns,
            figure_tuple,
            height,
            table_tuple,
            reads_in_columns=bool(_find_gutters(body)),
            frame=frame,
        )

    @staticmethod
    def from_page(
        page: pymupdf.Page,
        text_rects: list[pymupdf.Rect],
        frame: TextFrame | None = None,
        scan: PageScan | None = None,
    ) -> PageGeometry:
        """scan holds the tables and figures already read for this page. Read now when not given."""
        scan = scan or scan_page(page)
        bounds = text_page_rect(page)
        return PageGeometry.from_rects(
            bounds.width,
            text_rects,
            list(scan.figures),
            bounds.height,
            scan.tables,
            frame,
        )


@dataclass(frozen=True)
class PageScan:
    """Tables, figures and words of one page. Extraction, reading order, role detection and interlinear glosses all
    use them; finding tables parses the rules and text of each page, and every read of rules, image boxes or words
    takes from a few to tens of milliseconds, so each page is read only once."""

    tables: tuple[RuledTable, ...]
    figures: tuple[pymupdf.Rect, ...]
    words: tuple[_TextWord, ...]
    # Areas of connected light shading merged into one: the background of a table or a text box
    shades: tuple[pymupdf.Rect, ...] = ()
    # Thin horizontal rules: table rules, fraction bars, underlines
    rules: tuple[pymupdf.Rect, ...] = ()
    # All thin horizontal strokes, down to short ones one letter wide (the underline of a single letter is shorter
    # than the lower bound of rules)
    strokes: tuple[pymupdf.Rect, ...] = ()
    # Thin vertical rules: vertical table rules, separators between the chapter number and title at a chapter start
    posts: tuple[pymupdf.Rect, ...] = ()
    # Stroked-only arcs (dependency arcs, attention arcs) and filled rounded boxes (labels on the arcs)
    arcs: tuple[pymupdf.Rect, ...] = ()
    round_boxes: tuple[pymupdf.Rect, ...] = ()
    # Visible spaces ␣ drawn with three short lines (LaTeX \verb* and \textvisiblespace); the font has no such
    # character
    visible_spaces: tuple[pymupdf.Rect, ...] = ()


def scan_page(page: pymupdf.Page) -> PageScan:
    images = _image_infos(page)
    visible, raw = _drawing_paths(page)
    rules = _image_rules(images)
    drawings = [*visible, *rules]
    words = page_words(page)
    horizontal, vertical = _rule_segments(page, drawings)
    # Figures are still found with the unclipped path boxes: after clipping, shapes that covered the whole page and
    # were ignored as background would turn up and change which text inside figures is kept
    return PageScan(
        page_tables(page, drawings, words),
        tuple(_figure_rects(page, [*raw, *rules], images)),
        words,
        _shaded_areas(drawings, text_page_rect(page)),
        tuple(horizontal),
        tuple(_thin_strokes(drawings)),
        tuple(vertical),
        *_curved_paths(drawings),
        visible_spaces=tuple(_visible_spaces(drawings)),
    )


# All four corners of a rounded box are curves; an arc is at least this wide
_ROUND_CORNERS = 4


_ARC_MIN_WIDTH = 6.0


# Maximum distance of the two feet of an arc from the bottom edge of its box
_ARC_FOOT = 3.0


def _curved_paths(drawings: list[Drawing]) -> tuple[tuple[pymupdf.Rect, ...], tuple[pymupdf.Rect, ...]]:
    """A stroked-only arch with curves is an arc: start and end points apart, both on the bottom edge of the box, with
    the middle raised. Lines in a plot, with one end high and one low, and circles and rounded boxes, whose ends
    meet, do not count. A filled small box with curves at all four corners is a rounded box."""
    arcs: list[pymupdf.Rect] = []
    boxes: list[pymupdf.Rect] = []
    for drawing in drawings:
        items = drawing.get("items") or []
        curves = sum(1 for item in items if item[0] == "c")
        if not curves:
            continue
        rect = pymupdf.Rect(drawing["rect"])
        start, end = items[0][1], items[-1][-1]
        arched = (
            isinstance(start, pymupdf.Point)
            and isinstance(end, pymupdf.Point)
            and abs(start.x - end.x) >= _ARC_MIN_WIDTH * 0.5
            and abs(start.y - rect.y1) <= _ARC_FOOT
            and abs(end.y - rect.y1) <= _ARC_FOOT
            and rect.height >= _ARC_FOOT * 2
        )
        if drawing.get("type") == "s" and rect.width >= _ARC_MIN_WIDTH and arched:
            arcs.append(rect)
        elif drawing.get("fill") is not None and curves >= _ROUND_CORNERS:
            boxes.append(rect)
    return tuple(arcs), tuple(boxes)


# Length of the bottom line of a visible space (points). Drawn 0.3 em wide, about 3 points at body-text size
_VISIBLE_SPACE_WIDTH = (1.0, 6.0)


def _visible_spaces(drawings: list[Drawing]) -> list[pymupdf.Rect]:
    """A short horizontal line with a shorter vertical line standing at each end, whose lower ends sit on the
    horizontal line: the shape of ␣. Only the shape is detected; extraction decides whether it falls in a gap within a
    line of text."""
    flats: list[pymupdf.Rect] = []
    stems: list[pymupdf.Rect] = []
    low, high = _VISIBLE_SPACE_WIDTH
    for drawing in drawings:
        for item in drawing.get("items", ()):
            if item[0] != "l":
                continue
            rect = pymupdf.Rect(item[1], item[2]).normalize()
            if rect.height < 0.05 and low <= rect.width <= high:
                flats.append(rect)
            elif rect.width < 0.05 and 0.3 <= rect.height <= high:
                stems.append(rect)
    found: list[pymupdf.Rect] = []
    for flat in flats:
        ends = [
            [
                stem
                for stem in stems
                if abs(stem.x0 - x) <= 0.5 and abs(stem.y1 - flat.y1) <= 0.5 and stem.height < flat.width
            ]
            for x in (flat.x0, flat.x1)
        ]
        if all(ends):
            found.append(pymupdf.Rect(flat.x0, min(stem.y0 for stem in ends[0] + ends[1]), flat.x1, flat.y1))
    return found


# Minimum width of an underline (points). Shorter ones are strokes of characters or marks
_STROKE_MIN = 1.0


# An underline drawn as one flat polygon (links in slides) has an outline at most this tall
_STROKE_SHAPE_THICK = 3.0


def _thin_strokes(drawings: list[Drawing]) -> list[pymupdf.Rect]:
    """Thin horizontal strokes of any length: each line segment and each flat filled rectangle counts as one. Another
    path whose whole outline is a flat band (a filled long thin polygon) counts as one; taller outlines are edges of
    shapes (trapezoids, rounded boxes, dashed boxes) and do not count."""
    found: list[pymupdf.Rect] = []
    for drawing in drawings:
        pieces: list[pymupdf.Rect] = []
        flat = True
        for item in drawing.get("items", ()):
            if item[0] == "l":
                rect = pymupdf.Rect(item[1], item[2]).normalize()
            elif item[0] == "re":
                rect = pymupdf.Rect(item[1]).normalize()
            else:
                flat = False
                break
            if rect.height > _RULE_THICK or rect.width < rect.height:
                flat = False
                break
            pieces.append(rect)
        if not flat:
            outline = pymupdf.Rect(drawing.get("rect") or ())
            pieces = [outline] if 0 < outline.height <= _STROKE_SHAPE_THICK or outline.height == 0 else []
        found.extend(rect for rect in pieces if rect.width >= _STROKE_MIN)
    return found


def _shaded_areas(drawings: list[Drawing], bounds: pymupdf.Rect) -> tuple[pymupdf.Rect, ...]:
    """Merge adjacent light shapes into areas. A table often has one shaded shape per cell, with a narrow gap between
    cells. A shape covering the whole page is the paper color and does not count."""
    areas: list[pymupdf.Rect] = []
    for drawing in drawings:
        area = pymupdf.Rect(drawing["rect"])
        if not light_fill(drawing) or (
            area.width > bounds.width * _PAGE_FILL and area.height > bounds.height * _PAGE_FILL
        ):
            continue
        while True:
            near = [other for other in areas if (area + (-_NEAR, -_NEAR, _NEAR, _NEAR)).intersects(other)]
            if not near:
                break
            for other in near:
                areas.remove(other)
                area |= other
        areas.append(area)
    return tuple(areas)


class _MarginBlock(Protocol):
    page: int
    rect: pymupdf.Rect
    text: str


def cell_containing(rect: pymupdf.Rect, tables: Sequence[RuledTable]) -> pymupdf.Rect | None:
    """The cell that holds most of the block. Body text outside the rules returns None."""
    area = rect.width * rect.height
    if area < 1:
        return None
    best: pymupdf.Rect | None = None
    best_area = 0.0
    for table in tables:
        if (rect & table.rect).is_empty:
            continue
        for cell in table.cells:
            hit = rect & cell
            if hit.is_empty or hit.width < 0.4 or hit.height < 0.3:
                continue
            overlap = hit.width * hit.height
            if overlap > best_area:
                best, best_area = cell, overlap
    if best is None or best_area < area * 0.5:
        return None
    return best


class _OrderedBlock(Protocol):
    rect: pymupdf.Rect
    clip: tuple[float, float, float, float] | None


_BlockT = TypeVar("_BlockT", bound=_OrderedBlock)


@dataclass(frozen=True)
class _ReadUnit(Generic[_BlockT]):
    y0: float
    x0: float
    column: int
    order: int
    spanning: bool
    blocks: tuple[_BlockT, ...]


def reading_order(blocks: Sequence[_BlockT], geo: PageGeometry) -> list[_BlockT]:
    """With clear column gaps, order column by column from top to bottom. Headings and whole tables spanning columns
    stay at their vertical position and are not put into any column.

    Single-column pages and pages split only by figure edges keep the given order. Cells inside a table already have
    a row order, and tables spanning columns are not split up.
    """
    if not geo.reads_in_columns or len(geo.columns) < 2 or len(blocks) < 2:
        return list(blocks)
    units = _read_units(blocks, geo)
    separators = sorted(
        (unit for unit in units if unit.spanning),
        key=lambda unit: (unit.y0, unit.order),
    )
    flow = [unit for unit in units if not unit.spanning]
    ordered: list[_BlockT] = []
    cursor = -1e9
    for separator in separators:
        band = [unit for unit in flow if cursor <= unit.y0 < separator.y0]
        ordered.extend(_emit_band(band))
        ordered.extend(separator.blocks)
        taken = {id(unit) for unit in band}
        flow = [unit for unit in flow if id(unit) not in taken]
        cursor = separator.y0
    ordered.extend(_emit_band(flow))
    return ordered


def _read_units(blocks: Sequence[_BlockT], geo: PageGeometry) -> list[_ReadUnit[_BlockT]]:
    spanning_tables = tuple(table for table in geo.tables if _spans_columns(table.rect, geo))
    grouped: dict[int, list[tuple[int, _BlockT]]] = {}
    loose: list[tuple[int, _BlockT]] = []
    for order, block in enumerate(blocks):
        table_index = _spanning_table_index(block, spanning_tables)
        if table_index is None:
            loose.append((order, block))
        else:
            grouped.setdefault(table_index, []).append((order, block))
    units: list[_ReadUnit[_BlockT]] = []
    for index, pairs in grouped.items():
        table = spanning_tables[index]
        cells = tuple(block for _order, block in pairs)
        units.append(
            _ReadUnit(
                table.rect.y0,
                table.rect.x0,
                geo.index_of(table.rect),
                pairs[0][0],
                True,
                cells,
            )
        )
    for order, block in loose:
        units.append(
            _ReadUnit(
                block.rect.y0,
                block.rect.x0,
                geo.index_of(block.rect),
                order,
                _spans_columns(block.rect, geo),
                (block,),
            )
        )
    return units


def _emit_band(units: Sequence[_ReadUnit[_BlockT]]) -> list[_BlockT]:
    ordered: list[_BlockT] = []
    for unit in sorted(units, key=lambda unit: (unit.column, unit.y0, unit.x0, unit.order)):
        ordered.extend(unit.blocks)
    return ordered


def _spans_columns(rect: pymupdf.Rect, geo: PageGeometry) -> bool:
    if geo.width > 0 and rect.width > geo.width * _SPAN_MAX:
        return True
    hits = 0
    for column in geo.columns:
        overlap = min(rect.x1, column.x1) - max(rect.x0, column.x0)
        if overlap > _GUTTER_MIN:
            hits += 1
    return hits >= 2


def _spanning_table_index(block: _OrderedBlock, tables: Sequence[RuledTable]) -> int | None:
    if block.clip is None:
        return None
    rect = pymupdf.Rect(block.clip)
    for index, table in enumerate(tables):
        if cell_containing(rect, (table,)) is not None:
            return index
    return None


def margin_skips(blocks: Sequence[_MarginBlock], geometries: Sequence[PageGeometry]) -> list[bool]:
    """Short lines repeated in the page margins, and page numbers, keep the original text.

    Only the top and bottom of the page are checked: a fixed narrow band, or the area outside the text frame. Text
    counts as a running header or footer only when the same text appears on the same side on two or more pages,
    ignoring digits (running headers often contain page and chapter numbers). Running headers of a book change with
    the section, so one that appears only once also counts when it sits at the same height as the running headers of
    other pages. Page numbers differ on every page and are detected separately. Numbers inside cells are not page
    numbers.
    """
    zones: list[str | None] = []
    for block in blocks:
        zones.append(_margin_zone(block, geometries))
    seen: dict[tuple[str, str], set[int]] = {}
    for block, zone in zip(blocks, zones):
        if zone is None:
            continue
        key = (zone, _norm_margin(block.text))
        if not key[1] or len(key[1]) > _MARGIN_LEN:
            continue
        seen.setdefault(key, set()).add(block.page)
    repeated = {key for key, pages in seen.items() if len(pages) >= 2}
    rows: dict[str, list[float]] = {}
    for block, zone in zip(blocks, zones):
        if zone is not None and (zone, _norm_margin(block.text)) in repeated:
            rows.setdefault(zone, []).append(block.rect.y0)
    skips: list[bool] = []
    for block, zone in zip(blocks, zones):
        if zone is None:
            skips.append(False)
            continue
        key = (zone, _norm_margin(block.text))
        same_row = (
            0 < len(key[1]) <= _MARGIN_LEN
            and sum(1 for y in rows.get(zone, ()) if abs(y - block.rect.y0) <= _MARGIN_ROW) >= _MARGIN_ROWS
        )
        skips.append(key in repeated or same_row or is_page_number(block.text))
    return skips


def _interval_distance(x: float, column: Column) -> float:
    if x < column.x0:
        return column.x0 - x
    if x > column.x1:
        return x - column.x1
    return 0.0


def _overlaps_y(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.y0 < b.y1 and a.y1 > b.y0


def _page_filling(rect: pymupdf.Rect, page_width: float, page_height: float) -> bool:
    if page_height <= 0:
        return False
    return bool(rect.width > page_width * _PAGE_FILL and rect.height > page_height * _PAGE_FILL)


def _voting_rects(
    rects: Sequence[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
    tables: Sequence[RuledTable],
    page_width: float,
    page_height: float,
) -> list[pymupdf.Rect]:
    """Text inside tables and labels inside figures take no part in column detection, so that gaps in cells or
    figures are not taken as column gaps."""
    zones = [table.rect for table in tables]
    zones.extend(figure for figure in figures if not _page_filling(figure, page_width, page_height))
    return [rect for rect in rects if not _mostly_inside(rect, zones)]


def _cutting_figures(
    figures: Sequence[pymupdf.Rect],
    page_width: float,
    page_height: float,
) -> list[pymupdf.Rect]:
    return [
        figure
        for figure in figures
        if figure.width >= _WALL_MIN_W
        and figure.height >= _WALL_MIN_H
        and not _page_filling(figure, page_width, page_height)
    ]


def _columns_of(
    page_width: float,
    usable: list[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
) -> tuple[Column, ...]:
    if not usable:
        return (Column(0.0, page_width),)
    body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
    gutters = _merge_gaps([*_find_gutters(body), *_figure_walls(body, figures)])
    if not gutters:
        return (Column(min(rect.x0 for rect in usable), max(rect.x1 for rect in usable)),)
    return _columns_from_gutters(body, gutters, usable)


def _figure_walls(
    body: list[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
) -> list[tuple[float, float]]:
    """With fewer than three text blocks beside a figure, the figure edge still makes that side one column."""
    walls: list[tuple[float, float]] = []
    for figure in figures:
        beside = [rect for rect in body if _overlaps_y(rect, figure) and _x_gap(rect, figure) >= _GUTTER_MIN]
        left = [rect for rect in beside if rect.x1 <= figure.x0 + 1]
        right = [rect for rect in beside if rect.x0 >= figure.x1 - 1]
        if len(left) >= 2 and _group_y_overlap(left, [figure]) >= _MIN_Y_OVERLAP:
            edge = max(rect.x1 for rect in left)
            if figure.x0 - edge >= _GUTTER_MIN:
                walls.append((edge, figure.x0))
        if len(right) >= 2 and _group_y_overlap(right, [figure]) >= _MIN_Y_OVERLAP:
            edge = min(rect.x0 for rect in right)
            if edge - figure.x1 >= _GUTTER_MIN:
                walls.append((figure.x1, edge))
    return walls


def _x_gap(rect: pymupdf.Rect, other: pymupdf.Rect) -> float:
    if rect.x1 <= other.x0:
        return other.x0 - rect.x1
    if rect.x0 >= other.x1:
        return rect.x0 - other.x1
    return 0.0


def _columns_from_gutters(
    body: list[pymupdf.Rect],
    gutters: list[tuple[float, float]],
    usable: list[pymupdf.Rect],
) -> tuple[Column, ...]:
    mids = [(left + right) / 2 for left, right in gutters]
    bounds = [-1e9, *mids, 1e9]
    edges: list[tuple[float | None, float | None]] = [(None, gutters[0][0])]
    for index in range(len(gutters) - 1):
        edges.append((gutters[index][1], gutters[index + 1][0]))
    edges.append((gutters[-1][1], None))
    columns: list[Column] = []
    voters = body or usable
    for (x_start, x_end), (lo, hi) in zip(edges, zip(bounds, bounds[1:])):
        group = [rect for rect in voters if lo < (rect.x0 + rect.x1) / 2 <= hi]
        if not group:
            continue
        columns.append(
            Column(
                min(rect.x0 for rect in group) if x_start is None else x_start,
                max(rect.x1 for rect in group) if x_end is None else x_end,
            )
        )
    if columns:
        return tuple(columns)
    return (Column(min(rect.x0 for rect in usable), max(rect.x1 for rect in usable)),)


def _merge_gaps(gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for x0, x1 in sorted(gaps):
        if x1 - x0 < 1:
            continue
        if merged and x0 <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    return merged


def content_floor(height: float) -> float:
    """Lowest edge that body text can be written to. Below it are page numbers and the footer; text continued
    across pages stops here."""
    if height <= 0:
        return 0.0
    return height - max(_MARGIN_MIN, height * _MARGIN_BAND)


def _margin_zone(block: _MarginBlock, geometries: Sequence[PageGeometry]) -> str | None:
    if block.page < 0 or block.page >= len(geometries):
        return None
    geo = geometries[block.page]
    if geo.height <= 0 or block.rect.height > _MARGIN_HEIGHT:
        return None
    if geo.cell_of(block.rect) is not None:
        return None
    band = max(_MARGIN_MIN, geo.height * _MARGIN_BAND)
    frame = geo.frame
    if block.rect.y1 <= band or (frame is not None and frame.y0 is not None and block.rect.y1 <= frame.y0 - 1):
        return "header"
    floor = content_floor(geo.height)
    if block.rect.y0 >= floor or (frame is not None and frame.y1 is not None and block.rect.y0 >= frame.y1 + 1):
        return "footer"
    return None


def _norm_margin(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text).strip().casefold())


def is_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 16:
        return False
    if _PAGE_NO_RE.match(stripped):
        return True
    return len(stripped) >= 1 and _ROMAN_RE.match(stripped) is not None


def _is_blank_fill(drawing: Drawing) -> bool:
    """Near-white, filled rectangles only, no stroke. This is background shading, not a figure. Stroked boxes and
    images still count as figures."""
    if drawing.get("type") != "f":
        return False
    fill = drawing.get("fill")
    if not isinstance(fill, (tuple, list)) or len(fill) < 3:
        return False
    if any(float(channel) < 0.96 for channel in fill[:3]):
        return False
    items = drawing.get("items") or []
    return bool(items) and all(isinstance(item, (tuple, list)) and bool(item) and item[0] == "re" for item in items)


def _figure_rects(
    page: pymupdf.Page,
    drawings: list[Drawing] | None = None,
    images: list[ImageInfo] | None = None,
) -> list[pymupdf.Rect]:
    bounds = text_page_rect(page)
    width, height = bounds.width, bounds.height
    found: list[pymupdf.Rect] = []
    for info in _image_infos(page) if images is None else images:
        rect = pymupdf.Rect(info.get("bbox"))
        if rect.width > _FIGURE_MIN and rect.height > _FIGURE_MIN:
            found.append(rect)
    if drawings is None:
        drawings = visible_drawings(page)
    grounds: list[pymupdf.Rect] = []
    marks: list[pymupdf.Rect] = []
    ruled: list[pymupdf.Rect] = []
    for drawing in drawings:
        rect = pymupdf.Rect(drawing.get("rect"))
        if _is_mark(drawing):
            marks.append(rect)
        elif drawing.get("type") == "s":
            ruled.extend(_flat_strokes(drawing))
        if rect.width < _DRAW_MIN or rect.height < _DRAW_MIN or rect.width * rect.height < _DRAW_AREA:
            continue
        if rect.width > width * _PAGE_FILL and rect.height > height * _PAGE_FILL:
            continue
        if _is_blank_fill(drawing):
            grounds.append(rect)
            continue
        found.append(rect)
    framed = [ground for ground in grounds if _framed(ground, ruled)]
    return [*found, *_figure_grounds(grounds, found, marks, framed)]


def _is_mark(drawing: Drawing) -> bool:
    """Small marks in a figure: colored ellipses circling words, dashed arrows. Solid thin lines are frame lines or
    underlines and do not count"""
    items = drawing.get("items") or []
    if any(item[0] == "c" for item in items):
        return True
    if drawing.get("dashes") and str(drawing.get("dashes")).strip("[] 0"):
        return True
    fill = drawing.get("fill")
    return isinstance(fill, (tuple, list)) and len(fill) >= 3 and any(float(channel) < 0.96 for channel in fill[:3])


def _flat_strokes(drawing: Drawing) -> list[pymupdf.Rect]:
    """Horizontal lines in a stroked path: line segments, and the top and bottom edges of rectangles."""
    out: list[pymupdf.Rect] = []
    for item in drawing.get("items") or []:
        if item[0] == "l":
            rect = pymupdf.Rect(item[1], item[2]).normalize()
            if rect.height < 1.5:
                out.append(rect)
        elif item[0] == "re":
            rect = pymupdf.Rect(item[1])
            out += [pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0), pymupdf.Rect(rect.x0, rect.y1, rect.x1, rect.y1)]
    return out


def _framed(ground: pymupdf.Rect, ruled: Sequence[pymupdf.Rect]) -> bool:
    """A horizontal line above and below the white ground, each spanning the whole ground and close to its top or
    bottom edge: the white ground is a framed figure."""

    def edge(y: float) -> bool:
        return any(
            abs(line.y0 - y) <= _FRAME_REACH and line.x0 <= ground.x0 + 2 and line.x1 >= ground.x1 - 2 for line in ruled
        )

    return edge(ground.y0) and edge(ground.y1)


def _figure_grounds(
    grounds: Sequence[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
    marks: Sequence[pymupdf.Rect] = (),
    framed: Sequence[pymupdf.Rect] = (),
) -> list[pymupdf.Rect]:
    """Plotting programs (matplotlib) first lay down a white ground and write the title, axis labels and legend on it,
    with the curves only in the middle part. A white ground that holds curves or images and is at most twice their
    size is a figure. White grounds laid under body text and figures are much larger than the figures.
    Titles and axis labels sit right next to the curves, so the white ground is only taken up to a margin around the
    curves: some white grounds extend down into the body text.
    A framed white ground with a passage of sample text, colored ellipses circling words and dashed lines connecting
    them (ed3book Figure 4.2) is also a figure: when there are enough of these small marks and they cover most of the
    white ground, they count like curves. An unframed white ground lies under body text, and scattered curves and
    dashed lines on it do not count"""
    kept: list[pymupdf.Rect] = []
    for ground in grounds:
        inside = [figure for figure in figures if _mostly_inside(figure, [ground])]
        dots = [mark for mark in marks if ground in framed and _mostly_inside(mark, [ground])]
        if inside or len(dots) >= _GROUND_MARKS:
            inside += dots
        if not inside:
            continue
        drawn = pymupdf.Rect(inside[0])
        for figure in inside[1:]:
            drawn |= figure
        if ground.width * ground.height <= drawn.width * drawn.height * _GROUND_SHARE:
            kept.append(ground & (drawn + (-_GROUND_REACH, -_GROUND_REACH, _GROUND_REACH, _GROUND_REACH)))
    return kept
