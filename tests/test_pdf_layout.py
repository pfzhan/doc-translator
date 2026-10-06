import pymupdf

from app.formats.pdf import TextBlock, _cross_page_units, _expand_right, _render_translated, extract_blocks
from app.formats.pdf_layout import PageGeometry, RuledTable, margin_skips


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


def _block(text: str, page: int, rect: pymupdf.Rect, size: float = 10) -> TextBlock:
    return TextBlock(
        page=page, rect=rect, line_rects=[], text=text, size=size, color="#000", bold=False,
    )


def test_cross_page_joins_the_right_column_end_to_the_next_left_column():
    geo = _two_columns()
    right = _block("the model performs quite", 0, pymupdf.Rect(250, 700, 360, 712))
    left = _block("well on this split.", 1, pymupdf.Rect(40, 60, 150, 72))
    merged = _cross_page_units([right, left], 10.0, geometries=[geo, geo])
    assert any(len(unit) == 2 for unit in merged)
    # 不传几何时仍按原来的页末/页首规则合并
    assert any(len(unit) == 2 for unit in _cross_page_units([right, left], 10.0))

    same = _block("the model performs quite", 0, pymupdf.Rect(40, 700, 150, 712))
    same_column = _cross_page_units([same, left], 10.0, geometries=[geo, geo])
    assert any(len(unit) == 2 for unit in same_column)

    earlier = _block("the model performs quite", 0, pymupdf.Rect(250, 400, 360, 412))
    tail = _block("and then it ends.", 0, pymupdf.Rect(250, 700, 360, 712))
    stopped = _cross_page_units([earlier, tail, left], 10.0, geometries=[geo, geo])
    assert all(len(unit) == 1 for unit in stopped)


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


def test_three_gutters_make_three_columns():
    left = _stack(30, 110, 40)
    middle = _stack(140, 220, 40)
    right = _stack(250, 340, 40)
    geo = PageGeometry.from_rects(400, left + middle + right)
    assert len(geo.columns) == 3
    assert geo.columns[0].x1 == 110
    assert geo.columns[1].x0 == 140 and geo.columns[1].x1 == 220
    assert geo.columns[2].x0 == 250
    assert geo.index_of(middle[0]) == 1


def test_figure_beside_a_short_column_still_stops_the_other_side():
    left = _stack(40, 150, 40, n=4)
    figure = pymupdf.Rect(220, 36, 360, 150)
    below = _stack(230, 350, 180, n=2)
    geo = PageGeometry.from_rects(400, left + below, [figure], height=400)
    assert geo.column_of(left[0]).x1 <= 150.1
    assert geo.right_limit(left[0], []) < figure.x0


def test_ruled_cells_are_separate_blocks_and_keep_the_translation_inside(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.draw_rect(pymupdf.Rect(40, 40, 180, 90), color=(0, 0, 0), width=0.8)
    page.draw_rect(pymupdf.Rect(180, 40, 340, 90), color=(0, 0, 0), width=0.8)
    page.insert_text((48, 68), "Alpha cell", fontsize=11)
    page.insert_text((190, 68), "beta cell", fontsize=11)
    src = tmp_path / "table.pdf"
    doc.save(src)

    blocks = extract_blocks(pymupdf.open(src))
    assert [block.text for block in blocks] == ["Alpha cell", "beta cell"]
    assert blocks[1].clip is not None and blocks[1].clip[0] >= 175
    translation = "这是一段很长的译文一定会向右铺开直到越过格子才停下来"
    out = _render_translated(src, blocks, [blocks[0].text, translation], "zh-CN")
    lines = [ln for blk in out[0].get_text("dict")["blocks"] if blk.get("type") == 0 for ln in blk["lines"]]
    zh = [ln for ln in lines if any("\u4e00" <= ch <= "\u9fff" for ch in "".join(s["text"] for s in ln["spans"]))]
    assert zh
    assert max(ln["bbox"][2] for ln in zh) <= blocks[1].clip[2] + 2
    assert "Alpha cell" in out[0].get_text()
    assert "beta cell" not in out[0].get_text()


def test_text_alignment_is_not_a_table():
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for i in range(5):
        page.insert_text((40, 50 + i * 16), "Left column line of body", fontsize=10)
        page.insert_text((230, 50 + i * 16), "Right column line of body", fontsize=10)
    geo = PageGeometry.from_page(page, _stack(40, 160, 40) + _stack(230, 360, 40))
    assert geo.tables == ()
    assert len(geo.columns) == 2


def test_repeated_header_and_page_number_stay_untranslated():
    def block(text: str, page: int, y0: float, y1: float) -> TextBlock:
        return _block(text, page, pymupdf.Rect(40, y0, 220, y1), size=9)

    blocks = [
        block("Proceedings of the Conference", 0, 20, 32),
        block("Body paragraph on the first page stays.", 0, 80, 96),
        block("Proceedings of the Conference", 1, 20, 32),
        block("Body paragraph on the second page stays.", 1, 80, 96),
        block("4", 1, 760, 772),
        block("12 samples were collected near the top.", 2, 20, 34),
    ]
    geos = [PageGeometry.from_rects(400, [pymupdf.Rect(40, 80, 300, 96)], height=800) for _ in range(3)]
    assert margin_skips(blocks, geos) == [True, False, True, False, True, False]

    cell = pymupdf.Rect(40, 740, 140, 785)
    table = RuledTable(pymupdf.Rect(40, 700, 300, 790), (cell,))
    boxed = PageGeometry.from_rects(400, [pymupdf.Rect(40, 80, 200, 94)], height=800, tables=[table])
    number = _block("4", 0, pymupdf.Rect(48, 752, 70, 766), size=9)
    assert margin_skips([number], [boxed]) == [False]
