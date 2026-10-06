"""页面几何：从文本块的横向间隔认出栏，把图和矢量图形收成障碍。

不依赖版面模型。跨栏标题（宽过页宽六成）不参与分栏，避免把整页收成一栏。
只认页宽 35%–65% 之间、至少 12pt 的空档，且左右各有三块以上、纵向交叠够深。
三栏或阅读顺序混乱的页面仍是一栏，写入时靠障碍躲开旁边的文字。
"""
from __future__ import annotations

from dataclasses import dataclass

import pymupdf

_GUTTER_MIN = 12.0
_SPLIT_LO = 0.35
_SPLIT_HI = 0.65
_SPAN_MAX = 0.60
_MIN_SIDE = 3
_MIN_Y_OVERLAP = 40.0
_USABLE_W = 8.0
_USABLE_H = 3.0
_FIGURE_MIN = 8.0
_DRAW_MIN = 24.0
_DRAW_AREA = 800.0
_PAGE_FILL = 0.85


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

    def right_limit(self, rect: pymupdf.Rect, obstacles: list[pymupdf.Rect]) -> float:
        """本栏右缘，并在纵向交叠的右侧障碍前停下。可以小于 rect.x1，调用方不得因此把框缩小。"""
        limit = self.column_of(rect).x1
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
    ) -> PageGeometry:
        usable = [rect for rect in rects if rect.width >= _USABLE_W and rect.height >= _USABLE_H]
        columns = _columns_of(page_width, usable)
        return PageGeometry(page_width, columns, tuple(figures or ()), height)

    @staticmethod
    def from_page(page: pymupdf.Page, text_rects: list[pymupdf.Rect]) -> PageGeometry:
        return PageGeometry.from_rects(
            page.rect.width, text_rects, _figure_rects(page), page.rect.height,
        )


def _interval_distance(x: float, column: Column) -> float:
    if x < column.x0:
        return column.x0 - x
    if x > column.x1:
        return x - column.x1
    return 0.0


def _overlaps_y(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.y0 < b.y1 and a.y1 > b.y0


def _columns_of(page_width: float, usable: list[pymupdf.Rect]) -> tuple[Column, ...]:
    if not usable:
        return (Column(0.0, page_width),)
    gutter = _find_gutter(page_width, usable)
    if gutter is None:
        return (Column(min(rect.x0 for rect in usable), max(rect.x1 for rect in usable)),)
    left_edge, right_edge = gutter
    mid = (left_edge + right_edge) / 2
    # 跨栏标题不参与定界，否则一栏会被拉到标题的右缘
    body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
    left = [rect for rect in body if (rect.x0 + rect.x1) / 2 < mid]
    right = [rect for rect in body if (rect.x0 + rect.x1) / 2 >= mid]
    if not left or not right:
        return (Column(min(rect.x0 for rect in usable), max(rect.x1 for rect in usable)),)
    return (
        Column(min(rect.x0 for rect in left), left_edge),
        Column(right_edge, max(rect.x1 for rect in right)),
    )


def _find_gutter(page_width: float, usable: list[pymupdf.Rect]) -> tuple[float, float] | None:
    body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
    if len(body) < _MIN_SIDE * 2:
        return None
    band_lo = page_width * _SPLIT_LO
    band_hi = page_width * _SPLIT_HI
    best: tuple[float, float] | None = None
    best_width = 0.0
    for left_edge, right_edge in _gaps(_merged_x(body)):
        width = right_edge - left_edge
        mid = (left_edge + right_edge) / 2
        if width < _GUTTER_MIN or width <= best_width or not band_lo <= mid <= band_hi:
            continue
        left = [rect for rect in body if (rect.x0 + rect.x1) / 2 < mid]
        right = [rect for rect in body if (rect.x0 + rect.x1) / 2 >= mid]
        if len(left) < _MIN_SIDE or len(right) < _MIN_SIDE:
            continue
        if _group_y_overlap(left, right) < _MIN_Y_OVERLAP:
            continue
        best = (left_edge, right_edge)
        best_width = width
    return best


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


def _group_y_overlap(left: list[pymupdf.Rect], right: list[pymupdf.Rect]) -> float:
    overlap = min(max(rect.y1 for rect in left), max(rect.y1 for rect in right))
    overlap -= max(min(rect.y0 for rect in left), min(rect.y0 for rect in right))
    return overlap


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
