"""页面几何：从文本块的横向间隔认出栏，把图、矢量图形和有线表格收成边界。

不依赖版面模型。跨栏标题（宽过页宽六成）不参与分栏。
栏间空档至少 12pt，左右各有三块以上、纵向交叠够深。空档不必落在页宽正中，
所以三栏也能分开。图把一侧的文字吃掉时，图缘仍是栏墙，写入不会跨过去。
有横竖线的表格用格子限制写入；页眉、页脚和页码另作跳过，不参与分栏。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import pymupdf

_GUTTER_MIN = 12.0
_SPAN_MAX = 0.60
_MIN_SIDE = 3
_MIN_Y_OVERLAP = 40.0
_USABLE_W = 8.0
_USABLE_H = 3.0
_FIGURE_MIN = 8.0
_DRAW_MIN = 24.0
_DRAW_AREA = 800.0
_PAGE_FILL = 0.85
_WALL_MIN_H = 48.0
_WALL_MIN_W = 36.0
_CELL_MIN_W = 18.0
_CELL_MIN_H = 8.0
_CELL_NARROW = 16.0
_MAX_TABLE_COLS = 12
_INSIDE = 0.6
_MARGIN_BAND = 0.07
_MARGIN_MIN = 42.0
_MARGIN_HEIGHT = 22.0
_MARGIN_LEN = 80
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
class RuledTable:
    rect: pymupdf.Rect
    cells: tuple[pymupdf.Rect, ...]


@dataclass(frozen=True)
class PageGeometry:
    width: float
    columns: tuple[Column, ...]
    figures: tuple[pymupdf.Rect, ...]
    height: float = 0.0
    tables: tuple[RuledTable, ...] = ()

    def index_of(self, rect: pymupdf.Rect) -> int:
        """块中心落在哪一栏。落在栏间空白时取最近的一栏。"""
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

    def cell_of(self, rect: pymupdf.Rect) -> pymupdf.Rect | None:
        return cell_containing(rect, self.tables)

    def right_limit(self, rect: pymupdf.Rect, obstacles: list[pymupdf.Rect]) -> float:
        """本栏右缘，并在纵向交叠的右侧障碍前停下。可以小于 rect.x1，调用方不得因此把框缩小。"""
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
    ) -> PageGeometry:
        table_tuple = tuple(tables or ())
        table_zones = [table.rect for table in table_tuple]
        # 表格外框也是一块大矢量，不能再当成图，否则格子里的字会被整段跳过
        figure_tuple = tuple(
            figure for figure in (figures or ()) if not _mostly_inside(figure, table_zones)
        )
        voting = _voting_rects(rects, figure_tuple, table_tuple, page_width, height)
        usable = [rect for rect in voting if rect.width >= _USABLE_W and rect.height >= _USABLE_H]
        columns = _columns_of(page_width, usable, _cutting_figures(figure_tuple, page_width, height))
        return PageGeometry(page_width, columns, figure_tuple, height, table_tuple)

    @staticmethod
    def from_page(page: pymupdf.Page, text_rects: list[pymupdf.Rect]) -> PageGeometry:
        return PageGeometry.from_rects(
            page.rect.width,
            text_rects,
            _figure_rects(page),
            page.rect.height,
            ruled_tables(page),
        )


class _MarginBlock(Protocol):
    page: int
    rect: pymupdf.Rect
    text: str


def cell_containing(rect: pymupdf.Rect, tables: Sequence[RuledTable]) -> pymupdf.Rect | None:
    """块的大部分落在哪个格子里。落在线外的正文返回 None。"""
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


def ruled_tables(page: pymupdf.Page) -> tuple[RuledTable, ...]:
    """只认横竖线围出的格子。文字对齐出来的假表、公式笔画网格不要。"""
    pymupdf.no_recommend_layout()
    try:
        found = page.find_tables(vertical_strategy="lines_strict", horizontal_strategy="lines_strict")
    except Exception:  # noqa: BLE001 - 个别页面的线读不出来，当没有表
        return ()
    tables: list[RuledTable] = []
    for table in found.tables:
        cells = tuple(pymupdf.Rect(cell) for cell in table.cells if cell)
        if _usable_table(cells):
            tables.append(RuledTable(pymupdf.Rect(table.bbox), cells))
    return tuple(tables)


def margin_skips(blocks: Sequence[_MarginBlock], geometries: Sequence[PageGeometry]) -> list[bool]:
    """页边重复的短行，以及页码，保留原文。

    只看页顶和页底一条窄带。同一侧、同一文本出现在两页以上才算页眉页脚。
    页码每页不同，单独认。格子里的数字不是页码。
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
    skips: list[bool] = []
    for block, zone in zip(blocks, zones):
        if zone is None:
            skips.append(False)
            continue
        key = (zone, _norm_margin(block.text))
        skips.append(key in repeated or _is_page_number(block.text))
    return skips


def _interval_distance(x: float, column: Column) -> float:
    if x < column.x0:
        return column.x0 - x
    if x > column.x1:
        return x - column.x1
    return 0.0


def _overlaps_y(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.y0 < b.y1 and a.y1 > b.y0


def _mostly_inside(rect: pymupdf.Rect, zones: Sequence[pymupdf.Rect]) -> bool:
    area = rect.width * rect.height
    if area < 1:
        return False
    for zone in zones:
        hit = rect & zone
        if hit.is_empty or hit.width < 0.4 or hit.height < 0.3:
            continue
        if hit.width * hit.height >= area * _INSIDE:
            return True
    return False


def _page_filling(rect: pymupdf.Rect, page_width: float, page_height: float) -> bool:
    if page_height <= 0:
        return False
    return rect.width > page_width * _PAGE_FILL and rect.height > page_height * _PAGE_FILL


def _voting_rects(
    rects: Sequence[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
    tables: Sequence[RuledTable],
    page_width: float,
    page_height: float,
) -> list[pymupdf.Rect]:
    """表内文字和图内标签不参与分栏，避免把格子或图中的缝当成栏沟。"""
    zones = [table.rect for table in tables]
    zones.extend(
        figure for figure in figures if not _page_filling(figure, page_width, page_height)
    )
    return [rect for rect in rects if not _mostly_inside(rect, zones)]


def _cutting_figures(
    figures: Sequence[pymupdf.Rect], page_width: float, page_height: float,
) -> list[pymupdf.Rect]:
    return [
        figure for figure in figures
        if figure.width >= _WALL_MIN_W and figure.height >= _WALL_MIN_H
        and not _page_filling(figure, page_width, page_height)
    ]


def _columns_of(
    page_width: float, usable: list[pymupdf.Rect], figures: Sequence[pymupdf.Rect],
) -> tuple[Column, ...]:
    if not usable:
        return (Column(0.0, page_width),)
    body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
    gutters = _merge_gaps([*_find_gutters(body), *_figure_walls(body, figures)])
    if not gutters:
        return (Column(min(rect.x0 for rect in usable), max(rect.x1 for rect in usable)),)
    return _columns_from_gutters(body, gutters, usable)


def _find_gutters(body: list[pymupdf.Rect]) -> list[tuple[float, float]]:
    if len(body) < _MIN_SIDE * 2:
        return []
    found: list[tuple[float, float]] = []
    for left_edge, right_edge in _gaps(_merged_x(body)):
        if right_edge - left_edge < _GUTTER_MIN:
            continue
        mid = (left_edge + right_edge) / 2
        left = [rect for rect in body if (rect.x0 + rect.x1) / 2 < mid]
        right = [rect for rect in body if (rect.x0 + rect.x1) / 2 >= mid]
        if len(left) < _MIN_SIDE or len(right) < _MIN_SIDE:
            continue
        if _group_y_overlap(left, right) < _MIN_Y_OVERLAP:
            continue
        found.append((left_edge, right_edge))
    return found


def _figure_walls(
    body: list[pymupdf.Rect], figures: Sequence[pymupdf.Rect],
) -> list[tuple[float, float]]:
    """图旁文字不够三块时，图缘仍把这一侧收成一栏。"""
    walls: list[tuple[float, float]] = []
    for figure in figures:
        beside = [
            rect for rect in body
            if _overlaps_y(rect, figure) and _x_gap(rect, figure) >= _GUTTER_MIN
        ]
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
        columns.append(Column(
            min(rect.x0 for rect in group) if x_start is None else x_start,
            max(rect.x1 for rect in group) if x_end is None else x_end,
        ))
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


def _merged_x(rects: list[pymupdf.Rect]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for x0, x1 in sorted((rect.x0, rect.x1) for rect in rects):
        if merged and x0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    return merged


def _gaps(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [
        (prev[1], nxt[0])
        for prev, nxt in zip(intervals, intervals[1:])
        if nxt[0] > prev[1]
    ]


def _group_y_overlap(left: Sequence[pymupdf.Rect], right: Sequence[pymupdf.Rect]) -> float:
    if not left or not right:
        return 0.0
    overlap = min(max(rect.y1 for rect in left), max(rect.y1 for rect in right))
    overlap -= max(min(rect.y0 for rect in left), min(rect.y0 for rect in right))
    return overlap


def _usable_table(cells: tuple[pymupdf.Rect, ...]) -> bool:
    if len(cells) < 2:
        return False
    widths = sorted(cell.width for cell in cells)
    heights = sorted(cell.height for cell in cells)
    if widths[len(widths) // 2] < _CELL_MIN_W or heights[len(heights) // 2] < _CELL_MIN_H:
        return False
    if sum(1 for cell in cells if cell.width < _CELL_NARROW) > len(cells) * 0.4:
        return False
    columns = len({round(cell.x0) for cell in cells})
    return columns <= _MAX_TABLE_COLS


def _margin_zone(block: _MarginBlock, geometries: Sequence[PageGeometry]) -> str | None:
    if block.page < 0 or block.page >= len(geometries):
        return None
    geo = geometries[block.page]
    if geo.height <= 0 or block.rect.height > _MARGIN_HEIGHT:
        return None
    if geo.cell_of(block.rect) is not None:
        return None
    band = max(_MARGIN_MIN, geo.height * _MARGIN_BAND)
    if block.rect.y1 <= band:
        return "header"
    if block.rect.y0 >= geo.height - band:
        return "footer"
    return None


def _norm_margin(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _is_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 16:
        return False
    if _PAGE_NO_RE.match(stripped):
        return True
    return len(stripped) >= 1 and _ROMAN_RE.match(stripped) is not None


def _figure_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    width, height = page.rect.width, page.rect.height
    found: list[pymupdf.Rect] = []
    for info in page.get_image_info():
        rect = pymupdf.Rect(info.get("bbox"))
        if rect.width > _FIGURE_MIN and rect.height > _FIGURE_MIN:
            found.append(rect)
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - 个别页面的矢量表读不出来，当没有图
        drawings = []
    for drawing in drawings:
        rect = pymupdf.Rect(drawing.get("rect"))
        if rect.width < _DRAW_MIN or rect.height < _DRAW_MIN:
            continue
        if rect.width * rect.height < _DRAW_AREA:
            continue
        if rect.width > width * _PAGE_FILL and rect.height > height * _PAGE_FILL:
            continue
        found.append(rect)
    return found
