"""段落的有序 run：文字带样式和基线，公式带框、基线和相对所在行的偏移。

检测仍在 pdf.py。这里只放模型和两件纯操作：按 run 分界切开译文，以及挑出
落在公式框里的曲线。写入仍走原来的 HTML 盒子。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from .pdf_flow import Emphasis

# 跨页单元的 run 分界。送翻文本里保留它，译完按它切开，不再按字数比例猜。
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


def intersecting_curves(box: pymupdf.Rect, drawings: list[pymupdf.Rect]) -> tuple[pymupdf.Rect, ...]:
    """和公式框相交、且自身不大过公式太多的曲线。整页通栏线不并进来。"""
    found: list[pymupdf.Rect] = []
    for rect in drawings:
        if rect.width < 0.4 or rect.height < 0.3:
            continue
        hit = rect & box
        if hit.is_empty or hit.width < 0.4 or hit.height < 0.3:
            continue
        if rect.width > max(box.width * 3, box.width + 8):
            continue
        if rect.height > max(box.height * 3, box.height + 8):
            continue
        found.append(pymupdf.Rect(rect))
    return tuple(found)


def expand_box(box: pymupdf.Rect, curves: tuple[pymupdf.Rect, ...]) -> pymupdf.Rect:
    out = pymupdf.Rect(box)
    for curve in curves:
        out |= curve
    return out
