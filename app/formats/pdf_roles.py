"""Replaceable layout roles: body text, heading, caption, figure.

The default implementation looks only at geometry and font size and downloads no model. Another detector
implements LayoutDetector.classify. A full-page image is not treated as a figure, so scanned pages with a text
layer are not skipped entirely.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pymupdf

from .pdf_layout import PageGeometry

_HEADING_LEN = 100
_HEADING_SIZE = 1.3
_HEADING_BOLD = 1.15
_CAPTION_LEN = 400
_PAGE_FILL = 0.7
_INSIDE = 0.7
# The number may include a section number (Figure 2.13). Only a label followed by a colon, period or vertical bar
# is certainly a caption
_CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.?|Table|Tab\.?|Scheme|Plate|图|表)\s*\d+(?:[.\-]\d+)*(\s*[:.。：|])?",
    re.I,
)
# Figure captions are below the figure. Table captions are often above the table and are not covered here
_FIGURE_CAPTION_RE = re.compile(r"^(?:Figure|Fig\.?|图)\s*\d+(?:[.\-]\d+)*(\s*[:.。：|])?", re.I)
# A caption without a colon is one size smaller than the body text, or has its label in bold
_CAPTION_SMALL = 0.95
# A body paragraph: starts at the left edge and is a sentence long
_BODY_LEN = 20
_NUMBERED_TITLE_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_ROMAN_TITLE_RE = re.compile(r"^[IVX]{1,6}\.\s+(?=\S)")
_LETTER_TITLE_RE = re.compile(r"^[A-Z]\.\s+[A-Z][a-z]")
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
    # The leading Figure or Table label uses a font different from the rest (bold)
    label_styled: bool = False
    # Every span of the block is bold or italic
    styled: bool = False


class LayoutDetector(Protocol):
    def classify(
        self,
        items: Sequence[LayoutItem],
        geometries: Sequence[PageGeometry],
        median: float,
    ) -> list[Role]:
        """Same length as items. Another model replaces only this method."""

    def text_regions(self, pno: int) -> list[pymupdf.Rect]:
        """Body-text regions boxed on a page. Empty without a model."""

    def prepare(self, pages: Sequence[int], on_page: Callable[[int, int], None]) -> None:
        """Examine these pages before roles are decided, reporting progress through on_page(pages done, total
        pages). Does nothing without a model."""


class HeuristicLayout:
    """Text in figures, captions attached to figures, and short headings larger than the body text. The rest is
    body text."""

    def classify(
        self,
        items: Sequence[LayoutItem],
        geometries: Sequence[PageGeometry],
        median: float,
    ) -> list[Role]:
        floats = _float_zones(items, geometries, median)
        captions = [
            item
            for item in items
            if len(item.text.strip()) < _CAPTION_LEN and _caption_start(item, _CAPTION_RE, median)
        ]
        boxed = _boxed_items(items, geometries, floats, captions)
        roles: list[Role] = []
        for item in items:
            figures = (*_figures_of(item.page, geometries), *floats.get(item.page, ()))
            # The white background laid down by drawing programs often encloses the caption too; a block starting
            # with a caption label is still a caption
            labeled = len(item.text.strip()) < _CAPTION_LEN and _caption_start(item, _CAPTION_RE, median)
            if _inside_figure(item.rect, figures) and not labeled and id(item) not in boxed:
                roles.append(Role.FIGURE)
            elif labeled and _inside_figure(item.rect, figures) or _is_caption(item, figures, median):
                roles.append(Role.CAPTION)
            elif (
                is_size_heading(item.text, item.size, item.bold, median)
                or _is_numbered_title(item.text)
                or _is_lettered_title(item.text, item.styled)
            ):
                roles.append(Role.HEADING)
            else:
                roles.append(Role.BODY)
        return roles

    def text_regions(self, pno: int) -> list[pymupdf.Rect]:
        return []

    def prepare(self, pages: Sequence[int], on_page: Callable[[int, int], None]) -> None:
        pass


# Minimum words and sentences of paragraph text in a box
_PROSE_WORDS = 40
_PROSE_SENTENCES = 2
# When paragraph text holds more than half of a box's words, the box is a text box. Other sentence lines in it
# (list items, examples) need at least this many words
_BOX_PROSE_SHARE = 0.5
_BOX_LINE_WORDS = 4


def _boxed_items(
    items: Sequence[LayoutItem],
    geometries: Sequence[PageGeometry],
    floats: Mapping[int, Sequence[pymupdf.Rect]],
    captions: Sequence[LayoutItem],
) -> set[int]:
    """Blocks (ids) in a box that count as body text. A box is decided as a whole: when paragraph text holds most
    of its words, it is a shaded note or a prompt box, and sentence-like list items and examples in it count as
    body text along with the paragraphs, so half of the box is not left in the original. Short labels in figures
    still count as figure."""
    prose = {
        id(item)
        for item in items
        if _boxed_text(item, (*_figures_of(item.page, geometries), *floats.get(item.page, ())), captions)
    }
    boxed = set(prose)
    by_page: dict[int, list[LayoutItem]] = {}
    for item in items:
        by_page.setdefault(item.page, []).append(item)
    for page, members in by_page.items():
        for figure in (*_figures_of(page, geometries), *floats.get(page, ())):
            inside = [item for item in members if _inside_figure(item.rect, [figure])]
            words = sum(len(item.text.split()) for item in inside)
            framed = sum(len(item.text.split()) for item in inside if id(item) in prose)
            if not framed or framed < words * _BOX_PROSE_SHARE:
                continue
            boxed.update(id(item) for item in inside if len(item.text.split()) >= _BOX_LINE_WORDS)
    return boxed


def _boxed_text(item: LayoutItem, figures: Sequence[pymupdf.Rect], captions: Sequence[LayoutItem]) -> bool:
    """A boxed passage of body text (a shaded or rounded-box note) rather than a figure. Vector boxes count as
    figures, but their text must not all be taken for figure labels. A box with a caption is a figure: English
    examples and prompts set in textbook figures are the subject of discussion and keep the original."""
    if not _framed_prose(item):
        return False
    holders = [figure for figure in figures if _inside_figure(item.rect, [figure])]
    return not any(
        caption.page == item.page and _captions(caption.rect, figure, caption.size)
        for figure in holders
        for caption in captions
    )


def _captions(caption: pymupdf.Rect, figure: pymupdf.Rect, size: float) -> bool:
    """A caption is next to this box: below or beside it (_attached), or above it (a table caption)."""
    if _attached(caption, figure, size):
        return True
    overlap_x = min(caption.x1, figure.x1) - max(caption.x0, figure.x0)
    return -2 <= figure.y0 - caption.y1 <= max(8.0, size * 1.6) and overlap_x > min(caption.width, figure.width) * 0.3


def _framed_prose(item: LayoutItem) -> bool:
    """A paragraph of several sentences. A label in a figure is a few words, never a paragraph of sentences."""
    text = item.text.strip()
    # A paragraph introducing what follows ends with a colon (...as your final verdict with a label:)
    if len(text.split()) < _PROSE_WORDS or not text.endswith(_BODY_END):
        return False
    return len(re.findall(r"[.!?。！？](?:\s|$)", text)) >= _PROSE_SENTENCES


def is_size_heading(text: str, size: float, bold: bool, median: float) -> bool:
    """Short blocks clearly larger than the body text, and bold ones slightly larger."""
    if len(text) >= _HEADING_LEN or median <= 0:
        return False
    return size >= median * _HEADING_SIZE or (bold and size >= median * _HEADING_BOLD)


def _is_numbered_title(text: str) -> bool:
    """'3.1 Encoder', '1 引言'. '2 samples' and sentences with a period are not headings."""
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


def _is_lettered_title(text: str, styled: bool) -> bool:
    """IEEE section headings are the same size as the body text: II. METHODS (small capitals), A. Materials
    (all italic). A Roman numeral followed by all capitals counts; a letter number needs the whole line in bold
    or italic, since B. Smith or V. Vuletić in body text are names."""
    stripped = text.strip()
    if not stripped or len(stripped) >= 80 or stripped.endswith(_SENT_END) or "," in stripped:
        return False
    if len(stripped.split()) > 8:
        return False
    roman = _ROMAN_TITLE_RE.match(stripped)
    if roman is not None:
        rest = stripped[roman.end() :]
        if styled or (rest.upper() == rest and sum(ch.isalpha() for ch in rest) >= 2):
            return True
    return styled and _LETTER_TITLE_RE.match(stripped) is not None


def _float_zones(
    items: Sequence[LayoutItem],
    geometries: Sequence[PageGeometry],
    median: float,
) -> dict[int, list[pymupdf.Rect]]:
    """The area above a caption, up to the previous body paragraph, is the figure.

    Node labels, tables and pseudocode are not always drawn as vectors (the picture environment of old TeX
    draws lines with glyphs), so vector boxes do not enclose them. The caption is below the figure, body text
    starts at the column's left edge or at the paragraph indent, and the figure's contents are indented or
    centered. The caption of the previous figure is also a boundary: two stacked figures have no body text in
    between.
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
        # In a single column body text starts at the left edge of the text frame. The column's left edge is taken
        # from the leftmost block on the page, which may be a running header or a protruding section number
        frame = geo.frame if len(geo.columns) < 2 else None
        start = frame.x0 if frame is not None else left
        end = frame.x1 if frame is not None else right
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
    """The upper bound of a figure area: a body paragraph, a subheading, or the caption of the previous figure.

    Body text starts at the left edge, is a sentence long and is no smaller than the body size: pseudocode in a
    box or small text in a figure is only slightly indented but one size smaller. It must also read as a
    paragraph: justified to the right edge, or ending with sentence punctuation. Cells in a table's first column
    neither reach the right edge nor end a sentence, also when they wrap onto two lines.
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
    """A caption starting with a figure or table label. Figure 2.13 shows ... in body text is at body size and its
    label uses no different font."""
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
