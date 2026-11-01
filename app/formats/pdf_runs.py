"""Ordered runs of a paragraph: text runs carry style and baseline, formula runs carry box, baseline and offset
relative to their line.

Formula spans are marked in pdf_math.py; runs are cut and placeholders made in pdf_units.py. This module holds
only the model and two pure operations: splitting a translation at run boundaries, and picking the curves that
fall in a formula box. Paragraphs with formulas are placed line by line by the typesetter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from .pdf_flow import Emphasis

# Run boundary of a unit that spans a page break. It is kept in the text sent for translation; when writing,
# the unit is gathered into one paragraph and is not split back into the two pages at the boundary.
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
    """A formula is one unit of the paragraph. dx and dy are relative to the left edge of its line and the body
    text baseline."""

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
    """Split at {|}. Return None when there is no boundary."""
    if _BOUNDARY_RE.search(text) is None:
        return None
    left, right = _BOUNDARY_RE.split(text, maxsplit=1)
    return left, right


def strip_boundary(text: str) -> str:
    """Remove boundaries for the preview. Whitespace around a boundary collapses into one space."""
    cleaned = _BOUNDARY_RE.sub(" ", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


# Fraction bars and radical signs often sit one or two points outside the glyph box, where the intersection
# test misses them. Anything farther away is an underline or a full-width rule.
_STROKE_GAP = 2.5
_STROKE_THIN = 2.4


def _edge_gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return max(dx, dy)


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _near_formula_stroke(box: pymupdf.Rect, rect: pymupdf.Rect) -> bool:
    """A thin line or vertical radical stroke right next to the box without intersecting it. A full-width rule
    that extends well beyond the box is not taken."""
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
        rect.y0,
        rect.y1,
        box.y0,
        box.y1,
    ) >= 0.6 * min(rect.height, box.height)
    return bool(
        beside and rect.width <= max(8.0, box.width * 0.45) and rect.height <= max(box.height * 2.2, box.height + 10)
    )


def intersecting_curves(box: pymupdf.Rect, drawings: list[pymupdf.Rect]) -> tuple[pymupdf.Rect, ...]:
    """Fraction bars and radical signs that intersect the formula box or sit right outside it. Rules across the
    full page width are not included."""
    found: list[pymupdf.Rect] = []
    x0, y0, x1, y1 = box
    for rect in drawings:
        if rect.width < 0.4 or rect.height < 0.3:
            continue
        # A page has thousands of lines; compare coordinates one by one instead of building an intersection Rect
        if (
            rect.x0 > x1 + _STROKE_GAP
            or rect.x1 < x0 - _STROKE_GAP
            or rect.y0 > y1 + _STROKE_GAP
            or rect.y1 < y0 - _STROKE_GAP
        ):
            continue
        intersects = min(rect.x1, x1) - max(rect.x0, x0) >= 0.4 and min(rect.y1, y1) - max(rect.y0, y0) >= 0.3
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
