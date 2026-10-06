import pymupdf

from app.formats.pdf import TextBlock, _cross_page_units, _expand_right, _render_translated
from app.formats.pdf_layout import PageGeometry


def _stack(x0: float, x1: float, y0: float, n: int = 4, step: float = 24) -> list[pymupdf.Rect]:
    return [pymupdf.Rect(x0, y0 + i * step, x1, y0 + 14 + i * step) for i in range(n)]


def _two_columns() -> PageGeometry:
    left = _stack(40, 150, 40)
    right = _stack(250, 360, 40)
    return PageGeometry.from_rects(400, left + right)


def test_one_mass_is_one_column():
    geo = PageGeometry.from_rects(400, _stack(40, 200, 40, n=5))
    assert len(geo.columns) == 1
    assert geo.columns[0].x0 == 40
    assert geo.columns[0].x1 == 200


def test_gutter_splits_two_columns():
    geo = _two_columns()
    assert len(geo.columns) == 2
    assert geo.columns[0].x1 == 150
    assert geo.columns[1].x0 == 250
    assert geo.index_of(pymupdf.Rect(40, 40, 90, 54)) == 0
    assert geo.index_of(pymupdf.Rect(250, 40, 300, 54)) == 1


def test_spanning_title_does_not_collapse_gutter():
    title = pymupdf.Rect(40, 10, 360, 28)
    geo = PageGeometry.from_rects(400, [title, *_stack(40, 150, 40), *_stack(250, 360, 40)])
    assert len(geo.columns) == 2
    assert geo.columns[0].x1 == 150


def test_stacked_groups_stay_one_column():
    top = _stack(40, 150, 40)
    bottom = _stack(250, 360, 200)
    geo = PageGeometry.from_rects(400, top + bottom)
    assert len(geo.columns) == 1


def test_right_limit_stops_at_column_and_figure():
    geo = _two_columns()
    narrow = pymupdf.Rect(40, 40, 90, 54)
    assert geo.right_limit(narrow, []) == 150
    figure = pymupdf.Rect(120, 40, 145, 80)
    assert geo.right_limit(narrow, [figure]) == 118


def test_expand_right_honors_column_limit():
    rect = pymupdf.Rect(40, 40, 90, 54)
    widened = _expand_right(rect, 400, [], right_limit=150)
    assert widened.x1 == 150
    # 栏缘比原框还左时不缩小
    kept = _expand_right(pymupdf.Rect(40, 40, 200, 54), 400, [], right_limit=150)
    assert kept.x1 == 200


def test_from_page_collects_image_not_page_border():
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(pymupdf.Rect(10, 10, 390, 390), width=0.4)
    page.draw_rect(pymupdf.Rect(220, 180, 320, 280), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), 1)
    page.insert_image(pymupdf.Rect(230, 80, 310, 150), pixmap=pixmap)
    geo = PageGeometry.from_page(page, _stack(40, 150, 40))
    assert any(rect.x0 >= 220 and rect.width < 120 for rect in geo.figures)
    assert all(rect.width < 350 for rect in geo.figures)


def test_cross_page_refuses_a_different_column():
    geo = _two_columns()
    right = TextBlock(
        page=0, rect=pymupdf.Rect(250, 700, 360, 712), line_rects=[],
        text="the model performs quite", size=10, color="#000", bold=False,
    )
    left = TextBlock(
        page=1, rect=pymupdf.Rect(40, 60, 150, 72), line_rects=[],
        text="well on this split.", size=10, color="#000", bold=False,
    )
    split = _cross_page_units([right, left], 10.0, geometries=[geo, geo])
    assert all(len(unit) == 1 for unit in split)
    # 不传几何时仍按原来的页末/页首规则合并
    assert any(len(unit) == 2 for unit in _cross_page_units([right, left], 10.0))

    same = TextBlock(
        page=0, rect=pymupdf.Rect(40, 700, 150, 712), line_rects=[],
        text="the model performs quite", size=10, color="#000", bold=False,
    )
    merged = _cross_page_units([same, left], 10.0, geometries=[geo, geo])
    assert any(len(unit) == 2 for unit in merged)


def test_two_column_translation_stops_at_the_gutter(tmp_path):
    """左栏短行的长译文可以右扩，但旁边的右栏即使纵向错开也不能被盖住。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((40, 52), "Short left", fontsize=12)
    src = tmp_path / "cols.pdf"
    doc.save(src)

    left = [pymupdf.Rect(40, 40, 90, 54), *_stack(40, 150, 80, n=3)]
    right = _stack(250, 360, 80, n=3)
    geo = PageGeometry.from_rects(400, left + right)
    block = TextBlock(
        page=0, rect=pymupdf.Rect(40, 40, 90, 54), line_rects=[pymupdf.Rect(40, 40, 90, 54)],
        text="Short left", size=12, color="#000000", bold=False,
    )
    translation = "这是一段很长的译文一定会向右铺开直到越过栏间空白才停下来"
    out = _render_translated(src, [block], [translation], "zh-CN", geometries=[geo])
    lines = [ln for blk in out[0].get_text("dict")["blocks"] if blk.get("type") == 0 for ln in blk["lines"]]
    zh = [ln for ln in lines if any("\u4e00" <= ch <= "\u9fff" for ch in "".join(s["text"] for s in ln["spans"]))]
    assert zh
    right_edge = max(ln["bbox"][2] for ln in zh)
    assert right_edge > block.rect.x1 + 1
    assert right_edge <= geo.columns[0].x1 + 4
    assert right_edge < geo.columns[1].x0
