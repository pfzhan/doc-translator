"""段落的有序 run：文字带样式和基线，公式带框、基线和相对所在行的偏移。

检测仍在 pdf.py。这里只放模型和两件纯操作：按 run 分界切开译文，以及挑出
落在公式框里的曲线。含公式的段落由排版器按行落位。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from .pdf_flow import Emphasis

# 跨页单元的 run 分界。送翻文本里保留它，写回时收成一段，不再按它切回两页。
BOUNDARY = "{|}"
_BOUNDARY_RE = re.compile(r"\s*\{\s*\|\s*\}\s*")


@dataclass(frozen=True)
class TextRun:
    text: str
    font: str
    size: float
    emphasis: Emphasis
    baseline: float
    bbox: pymupdf.Rect
    line: int
    owner: int = 0


@dataclass(frozen=True)
class FormulaRun:
    """公式是段落里的一个单元。dx、dy 相对它所在行的左缘和正文基线。"""

    index: int
    bbox: pymupdf.Rect
    baseline: float
    dx: float
    dy: float
    descent: float
    text: str
    page: int
    name: str
    line: int
    owner: int = 0
    curves: tuple[pymupdf.Rect, ...] = ()


Run = TextRun | FormulaRun


def split_page_boundary(text: str) -> tuple[str, str] | None:
    """按 {| } 切开。没有分界时返回 None。"""
    if _BOUNDARY_RE.search(text) is None:
        return None
    left, right = _BOUNDARY_RE.split(text, maxsplit=1)
    return left, right


def strip_boundary(text: str) -> str:
    """预览里去掉分界。分界两侧的空白收成一个空格。"""
    cleaned = _BOUNDARY_RE.sub(" ", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


# 分式线、根号经常贴在字形框外一两磅，相交判定收不到。再远就是下划线或通栏线。
_STROKE_GAP = 2.5
_STROKE_THIN = 2.4


def _edge_gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return max(dx, dy)


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _near_formula_stroke(box: pymupdf.Rect, rect: pymupdf.Rect) -> bool:
    """贴着框、但没有相交的细线或根号竖笔。长出一截的通栏线不收。"""
    if _edge_gap(box, rect) > _STROKE_GAP:
        return False
    if min(rect.width, rect.height) <= _STROKE_THIN and rect.height <= rect.width:
        overlap = _axis_overlap(rect.x0, rect.x1, box.x0, box.x1)
        cap = max(box.width * 2.5, box.width + 16)
        return overlap >= 0.55 * min(rect.width, box.width) and rect.width <= cap and rect.width < 200
    if min(rect.width, rect.height) <= _STROKE_THIN:
        overlap = _axis_overlap(rect.y0, rect.y1, box.y0, box.y1)
        cap = max(box.height * 2.5, box.height + 12)
        return overlap >= 0.55 * min(rect.height, box.height) and rect.height <= cap
    beside = rect.x1 <= box.x0 + 1 and _axis_overlap(
        rect.y0, rect.y1, box.y0, box.y1,
    ) >= 0.6 * min(rect.height, box.height)
    return bool(
        beside
        and rect.width <= max(8.0, box.width * 0.45)
        and rect.height <= max(box.height * 2.2, box.height + 10)
    )


def intersecting_curves(box: pymupdf.Rect, drawings: list[pymupdf.Rect]) -> tuple[pymupdf.Rect, ...]:
    """和公式框相交、或紧贴框外的分式线与根号。整页通栏线不并进来。"""
    found: list[pymupdf.Rect] = []
    for rect in drawings:
        if rect.width < 0.4 or rect.height < 0.3:
            continue
        hit = rect & box
        intersects = not hit.is_empty and hit.width >= 0.4 and hit.height >= 0.3
        if intersects:
            if rect.width > max(box.width * 3, box.width + 8):
                continue
            if rect.height > max(box.height * 3, box.height + 8):
                continue
        elif not _near_formula_stroke(box, rect):
            continue
        found.append(pymupdf.Rect(rect))
    return tuple(found)


def expand_box(box: pymupdf.Rect, curves: tuple[pymupdf.Rect, ...]) -> pymupdf.Rect:
    out = pymupdf.Rect(box)
    for curve in curves:
        out |= curve
    return out
