"""Read what is drawn on a page: vector drawings, images, words and ruling segments, with the thresholds
shared by table and column detection."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

from .pdf_types import Drawing, ImageInfo, image_infos, page_drawings

_GUTTER_MIN = 12.0


_SPAN_MAX = 0.60


_MIN_SIDE = 3


_MIN_Y_OVERLAP = 40.0


_PAGE_FILL = 0.85


# Table rules: at most this thick and at least this long
_RULE_THICK = 1.5


_RULE_MIN = 5.0


# Distance within which a rule and a cell edge count as aligned
_NEAR = 2.0


_INSIDE = 0.6


@dataclass(frozen=True)
class RuledTable:
    rect: pymupdf.Rect
    cells: tuple[pymupdf.Rect, ...]
    # Pseudocode in an algorithm box is also cut into cells as a one-column table, but is translated line by line.
    # Other tables keep the original text
    algorithm: bool = False


def _image_infos(page: pymupdf.Page) -> list[ImageInfo]:
    """Images on this page, with their boxes reduced to the clipping paths."""
    try:
        infos = image_infos(page)
    except Exception:  # noqa: BLE001 - if they cannot be read, treat the page as having no images
        return []
    return _clip_images(page, infos) if infos else infos


# Description of an image block in the text layer: <image: DeviceRGB, width: 678, height: 678, bpc: 8>
_IMAGE_SIZE_RE = re.compile(r"width: (\d+), height: (\d+)")


def _clip_images(page: pymupdf.Page, infos: list[ImageInfo]) -> list[ImageInfo]:
    """Replace image boxes with the part visible after clipping. get_image_info ignores clipping paths: a sample image
    of 20 lines shows only its top three lines, yet its box reaches the bottom of the page, and the two paragraphs it
    covers are taken as text inside the image. Reading the text layer with clipping gives image blocks whose boxes are
    the visible parts; they are matched back to the images by pixel size and position. Images without a match keep
    their original box."""
    try:
        blocks = page.get_text("blocks", flags=pymupdf.TEXT_PRESERVE_IMAGES | pymupdf.TEXT_CLIP)
    except Exception:  # noqa: BLE001 - if it cannot be read, keep the original boxes
        return infos
    shown: list[tuple[pymupdf.Rect, tuple[int, int]]] = []
    for block in blocks:
        size = _IMAGE_SIZE_RE.search(str(block[4])) if block[6] == 1 else None
        if size is not None:
            shown.append((pymupdf.Rect(block[:4]), (int(size.group(1)), int(size.group(2)))))
    out: list[ImageInfo] = []
    for info in infos:
        whole = pymupdf.Rect(info["bbox"])
        found = next(
            (
                n
                for n, (rect, size) in enumerate(shown)
                if size == (info.get("width"), info.get("height")) and rect in whole + (-1, -1, 1, 1)
            ),
            None,
        )
        if found is None:
            out.append(info)
            continue
        rect, _size = shown.pop(found)
        out.append({**info, "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)} if rect != whole else info)
    return out


def visible_drawings(page: pymupdf.Page) -> list[Drawing]:
    """Paths from get_drawings, with boxes clipped to their clipping region. The box of a curve path clipped into a
    figure can reach tens of thousands of points off the page, and the original box would cover the whole column of
    body text below. Paths entirely outside their clipping region are invisible and are dropped"""
    return _drawing_paths(page)[0]


def _drawing_paths(page: pymupdf.Page) -> tuple[list[Drawing], list[Drawing]]:
    """Read the paths of this page once and return two lists: clipped (see visible_drawings) and unclipped."""
    try:
        paths = page_drawings(page)
    except Exception:  # noqa: BLE001 - the vector graphics of some pages cannot be read; treat as no rules
        return [], []
    out: list[Drawing] = []
    raw: list[Drawing] = []
    clips: list[tuple[int, pymupdf.Rect]] = []
    for path in paths:
        level = int(path.get("level") or 0)
        # A clipping region applies to paths nested below its level and lapses once a path is back at its level
        clips = [clip for clip in clips if clip[0] < level]
        kind = path.get("type")
        if kind == "clip":
            if path.get("scissor") is not None:
                clips.append((level, pymupdf.Rect(path["scissor"])))
            continue
        if kind == "group" or path.get("rect") is None:
            continue
        raw.append(path)
        rect = pymupdf.Rect(path["rect"])
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        for _, scissor in clips:
            x0, y0, x1, y1 = max(x0, scissor.x0), max(y0, scissor.y0), min(x1, scissor.x1), min(y1, scissor.y1)
        if x0 > x1 or y0 > y1:
            continue
        out.append({**path, "rect": pymupdf.Rect(x0, y0, x1, y1)} if clips else path)
    return out, raw


def _page_drawings(page: pymupdf.Page, images: list[ImageInfo] | None = None) -> list[Drawing]:
    """Vector graphics of this page plus thin rules drawn as images. images holds the image boxes of this page; read
    now when not given."""
    return [*visible_drawings(page), *_image_rules(_image_infos(page) if images is None else images)]


def light_fill(drawing: Drawing) -> bool:
    """Light filled-only shapes made of rectangles and thicker than a rule in both directions: the shading of table
    cells and text boxes. Dark shapes do not count: caption labels are white text on black."""
    rect = pymupdf.Rect(drawing["rect"])
    items = drawing.get("items") or []
    fill = drawing.get("fill")
    return (
        drawing.get("type") == "f"
        and bool(items)
        and all(item[0] in ("re", "qu") for item in items)
        and min(rect.width, rect.height) > _RULE_THICK
        and fill is not None
        and len(fill) == 3
        and 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2] > 0.6
    )


def page_words(page: pymupdf.Page) -> tuple[_TextWord, ...]:
    return tuple(
        _TextWord(float(item[0]), float(item[1]), float(item[2]), float(item[3]), str(item[4]))
        for item in page.get_text("words")
        if str(item[4]).strip()
    )


def text_page_rect(page: pymupdf.Page) -> pymupdf.Rect:
    """The page box that text coordinates refer to. On a rotated page page.rect is rotated, while extracted
    coordinates are not."""
    return pymupdf.Rect(0, 0, page.cropbox.width, page.cropbox.height)


def _image_rules(images: Sequence[ImageInfo]) -> list[Drawing]:
    """Thin strips made by stretching one-pixel images, written as filled rectangles in the get_drawings format.

    Old TeX and some converters draw frame lines with images, which get_drawings does not read, so find_tables does not
    see them either."""
    out: list[Drawing] = []
    for info in images:
        rect = pymupdf.Rect(info["bbox"])
        if min(rect.width, rect.height) <= _RULE_THICK and max(rect.width, rect.height) >= _RULE_MIN:
            out.append(
                {
                    "type": "f",
                    "rect": rect,
                    "items": [("re", rect, 1)],
                    "closePath": False,
                    "fill": (0.0, 0.0, 0.0),
                    "color": None,
                    "width": 0.0,
                }
            )
    return out


def _rule_segments(
    page: pymupdf.Page,
    drawings: list[Drawing] | None = None,
) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """Thin horizontal and vertical rules on the page. drawings holds the vector graphics of this page (including
    rules drawn as images); read now when not given.

    Line segments, long thin rectangles and the four edges of stroked rectangles all count."""
    rects: list[pymupdf.Rect] = []
    if drawings is None:
        drawings = _page_drawings(page)
    for drawing in drawings:
        stroked = "s" in str(drawing.get("type") or "")
        for item in drawing.get("items", ()):
            if item[0] == "l":
                rects.append(pymupdf.Rect(item[1], item[2]).normalize())
            elif item[0] == "re":
                rect = pymupdf.Rect(item[1]).normalize()
                rects.append(rect)
                if stroked:
                    rects.extend(_edges_of(rect))
    horizontal = [rect for rect in rects if rect.height <= _RULE_THICK and rect.width >= _RULE_MIN]
    vertical = [rect for rect in rects if rect.width <= _RULE_THICK and rect.height >= _RULE_MIN]
    return horizontal, vertical


def _edges_of(rect: pymupdf.Rect) -> list[pymupdf.Rect]:
    return [
        pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0),
        pymupdf.Rect(rect.x0, rect.y1, rect.x1, rect.y1),
        pymupdf.Rect(rect.x0, rect.y0, rect.x0, rect.y1),
        pymupdf.Rect(rect.x1, rect.y0, rect.x1, rect.y1),
    ]


@dataclass(frozen=True)
class _TextWord:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def width(self) -> float:
        return self.x1 - self.x0


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


def _merged_x(rects: list[pymupdf.Rect]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for x0, x1 in sorted((rect.x0, rect.x1) for rect in rects):
        if merged and x0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    return merged


def _gaps(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(prev[1], nxt[0]) for prev, nxt in zip(intervals, intervals[1:]) if nxt[0] > prev[1]]


def _group_y_overlap(left: Sequence[pymupdf.Rect], right: Sequence[pymupdf.Rect]) -> float:
    if not left or not right:
        return 0.0
    overlap = min(max(rect.y1 for rect in left), max(rect.y1 for rect in right))
    overlap -= max(min(rect.y0 for rect in left), min(rect.y0 for rect in right))
    return overlap
