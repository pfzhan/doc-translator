"""Pasting the images of inline formulas over their marker pixels (app/formats/pdf_formula_images.py)."""

import pymupdf
import pytest

from app.formats.pdf_formula_images import _place_formula_images
from app.formats.pdf_types import PdfDict

BASELINE = 100.0  # of the text line the markers sit on


def _pixmap(width: int, height: int, color: tuple[int, int, int]) -> pymupdf.Pixmap:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pix.set_rect(pix.irect, color)
    return pix


def _marker(page: pymupdf.Page, rect: pymupdf.Rect, mid: int, blue: int = 200) -> int:
    """The 1x1 pixel the writer puts where formula mid goes: mid + 1 in red and green, 200 in blue."""
    code = mid + 1
    xref: int = page.insert_image(rect, pixmap=_pixmap(1, 1, (code & 0xFF, code >> 8, blue)))
    return xref


def _formula(**extra: float) -> PdfDict:
    """A formula 20 wide and 10 high whose baseline is 2 above its bottom, with a red bitmap."""
    return {"w": 20.0, "h": 10.0, "d": 2.0, "png": _pixmap(20, 10, (255, 0, 0)).tobytes("png"), **extra}


def _page() -> tuple[pymupdf.Document, pymupdf.Page]:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((40, BASELINE), "The area of the circle is", fontsize=12)
    return doc, page


def _images(page: pymupdf.Page, before: set[int]) -> list[pymupdf.Rect]:
    """Where the images added since before are drawn, other than the white pixels that replace markers."""
    added = [img[0] for img in page.get_images(full=True) if img[0] not in before and img[2] * img[3] > 1]
    return [rect for xref in added for rect in page.get_image_rects(xref)]


def _close(rect: pymupdf.Rect, expected: tuple[float, float, float, float]) -> bool:
    return all(abs(a - b) < 0.1 for a, b in zip((rect.x0, rect.y0, rect.x1, rect.y1), expected, strict=True))


def test_a_formula_sits_on_the_baseline_of_its_line() -> None:
    doc, page = _page()
    before: set[int] = set()
    # htmlbox aligns the marker by its bottom, which lies a little off the baseline of the text
    marker = _marker(page, pymupdf.Rect(200, 91, 220, 101.5), 0)
    images = {img[0] for img in page.get_images(full=True)}

    _place_formula_images(page, doc, before, {0: _formula()})

    # Painted white, so the marker itself no longer shows
    assert pymupdf.Pixmap(doc, marker).pixel(0, 0) == (255, 255, 255)
    # The formula's own baseline (2 above its bottom) on the line's baseline, 2pt after the marker's left edge
    [placed] = _images(page, images)
    assert _close(placed, (202, BASELINE + 2 - 10, 222, BASELINE + 2))


def test_the_formula_box_scales_and_a_raised_formula_moves_up() -> None:
    doc, page = _page()
    _marker(page, pymupdf.Rect(200, 91, 220, 101.5), 0)
    images = {img[0] for img in page.get_images(full=True)}

    # rs scales the box; raise is how far the formula's baseline lies above the body's (a lone superscript)
    _place_formula_images(page, doc, set(), {0: _formula(rs=1.5, **{"raise": 1.0})})

    [placed] = _images(page, images)
    depth = (2.0 - 1.0) * 1.5
    assert _close(placed, (202, BASELINE + depth - 15, 202 + 30, BASELINE + depth))


def test_without_text_on_its_line_the_marker_box_decides() -> None:
    doc, page = _page()
    _marker(page, pymupdf.Rect(200, 200, 220, 210), 0)  # far below the only text
    images = {img[0] for img in page.get_images(full=True)}

    _place_formula_images(page, doc, set(), {0: _formula()})

    [placed] = _images(page, images)
    assert _close(placed, (202, 202, 222, 212))


@pytest.mark.parametrize(
    "mid,blue",
    [
        (5, 200),  # a marker of a formula this page does not place
        (0, 199),  # not a marker: any other image of the page
    ],
)
def test_other_images_are_left_alone(mid: int, blue: int) -> None:
    doc, page = _page()
    image = _marker(page, pymupdf.Rect(200, 91, 220, 101.5), mid, blue)
    images = {img[0] for img in page.get_images(full=True)}

    _place_formula_images(page, doc, set(), {0: _formula()})

    assert pymupdf.Pixmap(doc, image).pixel(0, 0) == ((mid + 1) & 0xFF, (mid + 1) >> 8, blue)
    assert _images(page, images) == []


def test_images_from_before_the_page_was_written_are_not_markers() -> None:
    doc, page = _page()
    image = _marker(page, pymupdf.Rect(200, 91, 220, 101.5), 0)
    before = {img[0] for img in page.get_images(full=True)}

    _place_formula_images(page, doc, before, {0: _formula()})

    assert pymupdf.Pixmap(doc, image).pixel(0, 0) == (1, 0, 200)
    assert _images(page, before) == []
