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
# 编号可以带节号（Figure 2.13）。标签后面跟冒号、句点或竖线，才一定是图注
_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.?|Table|Tab\.?|Scheme|Plate|图|表)\s*\d+(?:[.\-]\d+)*(\s*[:.。：|])?",
    re.I,
)
# 图注写在图下面。表注常在表上面，不在这里
_FIGURE_CAPTION_RE = re.compile(r"^(?:Figure|Fig\.?|图)\s*\d+(?:[.\-]\d+)*(\s*[:.。：|])?", re.I)
# 没有冒号的图注比正文小一号，或者标签另用粗体
_CAPTION_SMALL = 0.95
# 正文段落：从左缘起笔、有一句话长
_BODY_LEN = 20
_NUMBERED_TITLE_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_SENT_END = tuple("。！？.!?")
_BODY_END = tuple("。！？：.!?:")


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
    # 开头的 Figure、Table 标签用了和后文不同的字体（粗体）
    label_styled: bool = False


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
        floats = _float_zones(items, geometries, median)
        roles: list[Role] = []
        for item in items:
            figures = (*_figures_of(item.page, geometries), *floats.get(item.page, ()))
            # 画图程序铺的白底常连图注一起包住，图注标签起头的仍是图注
            labeled = len(item.text.strip()) < _CAPTION_LEN and _caption_start(item, _CAPTION_RE, median)
            if _inside_figure(item.rect, figures) and not labeled and not _framed_prose(item):
                roles.append(Role.FIGURE)
            elif labeled and _inside_figure(item.rect, figures) or _is_caption(item, figures, median):
                roles.append(Role.CAPTION)
            elif is_size_heading(item.text, item.size, item.bold, median) or _is_numbered_title(item.text):
                roles.append(Role.HEADING)
            else:
                roles.append(Role.BODY)
        return roles


# 框里成段的正文至少这么多词、这么多句
_PROSE_WORDS = 40
_PROSE_SENTENCES = 2


def _framed_prose(item: LayoutItem) -> bool:
    """框起来的一段正文（带底色、圆角框的提示段）。矢量框按图算，框里的字不能都当图里的标签：
    图里的标签是几个词，不会是几句话的整段。"""
    text = item.text.strip()
    if len(text.split()) < _PROSE_WORDS or not text.endswith(_SENT_END):
        return False
    return len(re.findall(r"[.!?。！？](?:\s|$)", text)) >= _PROSE_SENTENCES


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


def _float_zones(
    items: Sequence[LayoutItem], geometries: Sequence[PageGeometry], median: float,
) -> dict[int, list[pymupdf.Rect]]:
    """图注上方、到上一段正文为止是图。

    节点标签、表格、伪代码不一定画成矢量（老 TeX 的 picture 环境用字形画线），
    矢量框圈不住。图注写在图下面，正文从栏左缘或段首缩进起笔，图里的东西缩进或居中。
    上一张图的图注也是边界：两张图叠在一起时，中间没有正文。
    """
    zones: dict[int, list[pymupdf.Rect]] = {}
    for caption in items:
        if not _caption_start(caption, _FIGURE_CAPTION_RE, median):
            continue
        if caption.page < 0 or caption.page >= len(geometries):
            continue
        geo = geometries[caption.page]
        column = geo.column_of(caption.rect) if geo.columns else None
        left = column.x0 if column is not None else 0.0
        right = column.x1 if column is not None else geo.width
        # 单栏时正文从版心左缘起笔。栏左缘取的是本页最左的块，可能是页眉或外凸的节号
        framed = geo.frame is not None and len(geo.columns) < 2
        start = geo.frame.x0 if framed else left
        end = geo.frame.x1 if framed else right
        indent = max(median, 8.0) * 2
        top = 0.0
        for item in items:
            if item.page != caption.page or item is caption or item.rect.y1 > caption.rect.y0 + 1:
                continue
            if min(item.rect.x1, right) - max(item.rect.x0, left) <= 0:
                continue
            if _float_bound(item, start, end, indent, median):
                top = max(top, item.rect.y1)
        zone = pymupdf.Rect(min(start, caption.rect.x0), top + 0.5, right, caption.rect.y0 - 0.5)
        if zone.height > 2:
            zones.setdefault(caption.page, []).append(zone)
    return zones


def _float_bound(item: LayoutItem, start: float, end: float, indent: float, median: float) -> bool:
    """图区的上界：正文段落、小节标题，或上一张图的图注。

    正文从左缘起笔、有一句话长，字号不比正文小：框里的伪代码、图里的小字缩进不多，但小一号。
    还要像一段话：两端对齐写到右缘，或者以句末标点收尾。表格首列的格子写不到右缘，
    也不收句；格子里折成两行的也一样。
    """
    if _caption_start(item, _FIGURE_CAPTION_RE, median):
        return True
    if item.rect.x0 > start + indent or item.size < median * _CAPTION_SMALL:
        return False
    if is_size_heading(item.text, item.size, item.bold, median):
        return True
    text = item.text.strip()
    if len(text) < _BODY_LEN:
        return False
    return item.rect.x1 >= end - item.size * 0.5 or text.endswith(_BODY_END)


def _caption_start(item: LayoutItem, pattern: re.Pattern[str], median: float) -> bool:
    """以图表标签起头的图注。正文里的 Figure 2.13 shows ... 和正文一样大，标签也不另用字体。"""
    matched = pattern.match(item.text.strip())
    if matched is None:
        return False
    return bool(matched.group(1)) or item.label_styled or item.size < median * _CAPTION_SMALL


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


def _is_caption(item: LayoutItem, figures: Sequence[pymupdf.Rect], median: float) -> bool:
    text = item.text.strip()
    if len(text) >= _CAPTION_LEN or not _caption_start(item, _CAPTION_RE, median):
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
