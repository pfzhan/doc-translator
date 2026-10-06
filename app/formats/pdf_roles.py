"""可替换的版面角色：正文、标题、图注、图。

默认实现只看几何和字号，不下载模型。换检测器时实现 LayoutDetector.classify。
整页大图不当图，避免把带文本层的扫描页全部跳过。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import pymupdf

from .pdf_layout import PageGeometry

_HEADING_LEN = 100
_HEADING_SIZE = 1.3
_HEADING_BOLD = 1.15
_CAPTION_LEN = 400
_PAGE_FILL = 0.7
_INSIDE = 0.7
_CAPTION_RE = re.compile(
    r"^(?:(?:Figure|Fig\.?|Table|Tab\.?|Scheme|Plate)\s*\d+\s*[:.]|(?:图|表)\s*\d+\s*[:.。])",
    re.I,
)
_NUMBERED_TITLE_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_SENT_END = tuple("。！？.!?")


class Role(Enum):
    BODY = "body"
    HEADING = "heading"
    CAPTION = "caption"
    FIGURE = "figure"


@dataclass(frozen=True)
class LayoutItem:
    page: int
    rect: pymupdf.Rect
    text: str
    size: float
    bold: bool


class LayoutDetector(Protocol):
    def classify(
        self,
        items: Sequence[LayoutItem],
        geometries: Sequence[PageGeometry],
        median: float,
    ) -> list[Role]:
        """和 items 等长。换模型时只替换这一方法。"""


class HeuristicLayout:
    """图内文字、贴着图的图注、以及大于正文的短标题。其余是正文。"""

    def classify(
        self,
        items: Sequence[LayoutItem],
        geometries: Sequence[PageGeometry],
        median: float,
    ) -> list[Role]:
        roles: list[Role] = []
        for item in items:
            figures = _figures_of(item.page, geometries)
            if _inside_figure(item.rect, figures):
                roles.append(Role.FIGURE)
            elif _is_caption(item, figures):
                roles.append(Role.CAPTION)
            elif is_size_heading(item.text, item.size, item.bold, median) or _is_numbered_title(item.text):
                roles.append(Role.HEADING)
            else:
                roles.append(Role.BODY)
        return roles


def is_size_heading(text: str, size: float, bold: bool, median: float) -> bool:
    """短块里，明显大于正文的、以及略大于正文的粗体。"""
    if len(text) >= _HEADING_LEN or median <= 0:
        return False
    return size >= median * _HEADING_SIZE or (bold and size >= median * _HEADING_BOLD)


def _is_numbered_title(text: str) -> bool:
    """'3.1 Encoder'、'1 引言'。'2 samples' 和带句号的句子不是标题。"""
    stripped = text.strip()
    if not stripped or len(stripped) >= 80 or stripped.endswith(_SENT_END):
        return False
    if len(stripped.split()) > 8:
        return False
    matched = _NUMBERED_TITLE_RE.match(stripped)
    if matched is None:
        return False
    head = stripped[matched.end() - 1]
    return head.isupper() or "\u4e00" <= head <= "\u9fff"


def _figures_of(page: int, geometries: Sequence[PageGeometry]) -> tuple[pymupdf.Rect, ...]:
    if page < 0 or page >= len(geometries):
        return ()
    geo = geometries[page]
    if geo.height <= 0:
        return geo.figures
    kept: list[pymupdf.Rect] = []
    for figure in geo.figures:
        if figure.width > geo.width * _PAGE_FILL and figure.height > geo.height * _PAGE_FILL:
            continue
        kept.append(figure)
    return tuple(kept)


def _inside_figure(rect: pymupdf.Rect, figures: Sequence[pymupdf.Rect]) -> bool:
    area = rect.width * rect.height
    if area < 1:
        return False
    for figure in figures:
        hit = rect & figure
        if hit.is_empty or hit.width < 0.4 or hit.height < 0.3:
            continue
        if hit.width * hit.height >= area * _INSIDE:
            return True
    return False


def _is_caption(item: LayoutItem, figures: Sequence[pymupdf.Rect]) -> bool:
    text = item.text.strip()
    if len(text) >= _CAPTION_LEN or _CAPTION_RE.match(text) is None:
        return False
    return any(_attached(item.rect, figure, item.size) for figure in figures)


def _attached(block: pymupdf.Rect, figure: pymupdf.Rect, size: float) -> bool:
    slack = max(8.0, size * 1.6)
    overlap_x = min(block.x1, figure.x1) - max(block.x0, figure.x0)
    below = -2 <= block.y0 - figure.y1 <= slack and overlap_x > min(block.width, figure.width) * 0.3
    if below:
        return True
    gap = block.x0 - figure.x1 if block.x0 >= figure.x0 else figure.x0 - block.x1
    overlap_y = min(block.y1, figure.y1) - max(block.y0, figure.y0)
    return -2 <= gap <= slack and overlap_y > block.height * 0.4
