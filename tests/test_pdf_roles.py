from pathlib import Path

import pymupdf

from app.formats.pdf import TextBlock, _cross_page_units
from app.formats.pdf_layout import PageGeometry
from app.formats.pdf_roles import HeuristicLayout, LayoutItem, Role


def _item(text: str, rect: pymupdf.Rect, size: float = 10, bold: bool = False, page: int = 0) -> LayoutItem:
    return LayoutItem(page, rect, text, size, bold)


def _geo(figures: list[pymupdf.Rect], width: float = 400, height: float = 500) -> PageGeometry:
    return PageGeometry.from_rects(width, [pymupdf.Rect(40, 40, 180, 54)], figures, height)


def test_size_heading_stays_a_heading_and_author_stays_body():
    items = [
        _item("Introduction", pymupdf.Rect(40, 40, 160, 54), size=12, bold=True),
        _item("Jakob Uszkoreit", pymupdf.Rect(40, 70, 160, 82), size=10, bold=True),
        _item("A body sentence long enough to stay a paragraph.", pymupdf.Rect(40, 100, 300, 114)),
    ]
    roles = HeuristicLayout().classify(items, [_geo([])], median=10)
    assert roles == [Role.HEADING, Role.BODY, Role.BODY]


def test_numbered_title_is_a_heading_but_a_quantity_is_not():
    items = [
        _item("3.1 Encoder", pymupdf.Rect(40, 40, 140, 54)),
        _item("1 引言", pymupdf.Rect(40, 60, 120, 74)),
        _item("2 samples", pymupdf.Rect(40, 80, 140, 94)),
        _item("2 Samples were collected today.", pymupdf.Rect(40, 100, 280, 114)),
    ]
    roles = HeuristicLayout().classify(items, [_geo([])], median=10)
    assert roles == [Role.HEADING, Role.HEADING, Role.BODY, Role.BODY]


def test_text_inside_a_figure_is_a_figure_and_a_caption_is_not():
    figure = pymupdf.Rect(80, 80, 220, 180)
    items = [
        _item("axis", pymupdf.Rect(100, 110, 130, 122), size=8),
        _item("Figure 1: a plot.", pymupdf.Rect(80, 184, 200, 198)),
        _item("Figure 1 shows the plot in detail.", pymupdf.Rect(80, 184, 280, 198)),
        _item("Body paragraph under the caption stays body text.", pymupdf.Rect(40, 220, 300, 234)),
    ]
    roles = HeuristicLayout().classify(items, [_geo([figure])], median=10)
    assert roles == [Role.FIGURE, Role.CAPTION, Role.BODY, Role.BODY]


def test_page_filling_image_does_not_swallow_the_text_layer():
    page = pymupdf.Rect(0, 0, 400, 500)
    items = [_item("Recognized line of the scanned page.", pymupdf.Rect(40, 80, 300, 94))]
    roles = HeuristicLayout().classify(items, [_geo([page], height=500)], median=10)
    assert roles == [Role.BODY]


def test_cross_page_does_not_merge_a_caption():
    left = TextBlock(
        page=0, rect=pymupdf.Rect(40, 700, 200, 714), line_rects=[],
        text="the model performs quite", size=10, color="#000", bold=False,
    )
    caption = TextBlock(
        page=1, rect=pymupdf.Rect(40, 40, 180, 54), line_rects=[],
        text="examples of the plot", size=10, color="#000", bold=False,
    )
    units = _cross_page_units([left, caption], 10.0, halt_ids={id(caption)})
    assert all(len(unit) == 1 for unit in units)


def test_figure_label_stays_original_and_caption_is_translated(tmp_path: Path):
    from app.formats import translate_file
    from tests.test_formats import run, runner

    src = tmp_path / "fig.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 16))
    pix.set_rect(pix.irect, (180, 180, 180))
    page.insert_image(pymupdf.Rect(80, 70, 220, 160), pixmap=pix)
    page.insert_text((100, 120), "axis", fontsize=9)
    page.insert_text((80, 176), "Figure 1: a plot.", fontsize=10)
    page.insert_text((40, 40), "Body paragraph with enough letters here.", fontsize=11)
    doc.save(src)
    doc.close()

    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    text = pymupdf.open(out)[0].get_text()
    assert "axis" in text and "[zh-CN] axis" not in text
    assert "[zh-CN] Figure 1: a plot." in text
    assert "[zh-CN] Body paragraph with enough letters here." in text
