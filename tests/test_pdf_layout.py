import pymupdf

from app.formats.pdf import (
    TextBlock, _cross_page_units, _expand_right, _merge_continuations, _render_translated, extract_blocks,
)
from app.formats.pdf_layout import PageGeometry, RuledTable, margin_skips, reading_order


def _stack(x0: float, x1: float, y0: float, n: int = 4, step: float = 24) -> list[pymupdf.Rect]:
    return [pymupdf.Rect(x0, y0 + i * step, x1, y0 + 14 + i * step) for i in range(n)]


def _two_columns() -> PageGeometry:
    left = _stack(40, 150, 40)
    right = _stack(250, 360, 40)
    return PageGeometry.from_rects(400, left + right)


def test_one_mass_is_one_column():
    geo = PageGeometry.from_rects(400, _stack(40, 200, 40, n=5))
    assert not geo.reads_in_columns
    assert len(geo.columns) == 1
    assert geo.columns[0].x0 == 40
    assert geo.columns[0].x1 == 200


def test_gutter_splits_two_columns():
    geo = _two_columns()
    assert geo.reads_in_columns
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


def test_spanning_footer_does_not_steal_the_next_page_join():
    """跨栏页脚排在页末时，右栏末行仍接到下一页左栏，不被页脚抢走。"""
    geo = _two_columns()
    right = _block("the model performs quite", 0, pymupdf.Rect(250, 500, 360, 512))
    footer = _block("the notes continue", 0, pymupdf.Rect(40, 700, 360, 712))
    left = _block("well on this split.", 1, pymupdf.Rect(40, 60, 150, 72))
    units = _cross_page_units([right, footer, left], 10.0, geometries=[geo, geo])
    joined = [unit for unit in units if len(unit) == 2]
    assert joined == [[right, left]]


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
    assert not geo.reads_in_columns
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


def test_reading_order_joins_a_column_before_the_other_column():
    """另一栏插在中间时，本栏被拆开的两行仍要按栏接上。没有文字栏沟则保持原顺序。"""
    geo = _two_columns()
    left_a = _block("the training set is given like a", 0, pymupdf.Rect(40, 40, 150, 52))
    right_a = _block("The other column starts here.", 0, pymupdf.Rect(250, 40, 360, 52))
    left_b = _block("sequence while the target stays put.", 0, pymupdf.Rect(40, 64, 150, 76))
    right_b = _block("It has its own ending.", 0, pymupdf.Rect(250, 64, 360, 76))
    streamed = [left_a, right_a, left_b, right_b]
    streamed_merged = _merge_continuations(streamed)
    assert all(
        "sequence" not in block.text or not block.text.startswith("the training")
        for block in streamed_merged
    )
    ordered = reading_order(streamed, geo)
    assert [block.text for block in ordered] == [left_a.text, left_b.text, right_a.text, right_b.text]
    merged = _merge_continuations(ordered)
    assert merged[0].text.startswith("the training") and "sequence while" in merged[0].text
    assert all("other column" not in block.text or "sequence" not in block.text for block in merged)

    one = PageGeometry.from_rects(400, _stack(40, 200, 40, n=5))
    assert [block.text for block in reading_order(list(reversed(streamed)), one)] == [
        block.text for block in reversed(streamed)
    ]
    wall = PageGeometry.from_rects(
        400, _stack(40, 150, 40, n=4) + _stack(230, 350, 180, n=2),
        [pymupdf.Rect(220, 36, 360, 150)], height=400,
    )
    assert [block.text for block in reading_order(streamed, wall)] == [block.text for block in streamed]


def test_reading_order_keeps_spanning_title_and_table_between_bands():
    """跨栏标题和整表留在纵向位置。表内格子保持原顺序，不按页面分栏拆开。"""
    title = _block("3 Methods", 0, pymupdf.Rect(40, 120, 360, 136), size=14)
    left_above = _block("Left above.", 0, pymupdf.Rect(40, 40, 150, 52))
    right_above = _block("Right above.", 0, pymupdf.Rect(250, 40, 360, 52))
    left_below = _block("Left below.", 0, pymupdf.Rect(40, 170, 150, 182))
    right_below = _block("Right below.", 0, pymupdf.Rect(250, 170, 360, 182))
    geo = PageGeometry.from_rects(400, [title.rect, *_stack(40, 150, 40), *_stack(250, 360, 40)])
    ordered = reading_order(
        [left_below, right_below, title, right_above, left_above], geo,
    )
    assert [block.text for block in ordered] == [
        "Left above.", "Right above.", "3 Methods", "Left below.", "Right below.",
    ]

    table = RuledTable(
        pymupdf.Rect(40, 200, 360, 248),
        (pymupdf.Rect(40, 200, 180, 248), pymupdf.Rect(180, 200, 360, 248)),
    )
    table_geo = PageGeometry.from_rects(
        400, [*_stack(40, 150, 40), *_stack(250, 360, 40)], tables=(table,),
    )
    right_cell = TextBlock(
        page=0, rect=pymupdf.Rect(190, 210, 340, 230), line_rects=[], text="beta",
        size=10, color="#000", bold=False, clip=(180, 200, 360, 248),
    )
    left_cell = TextBlock(
        page=0, rect=pymupdf.Rect(48, 210, 160, 230), line_rects=[], text="alpha",
        size=10, color="#000", bold=False, clip=(40, 200, 180, 248),
    )
    below = _block("After the table.", 0, pymupdf.Rect(40, 270, 150, 282))
    ordered = reading_order([right_cell, left_above, left_cell, right_above, below], table_geo)
    assert [block.text for block in ordered] == [
        "Left above.", "Right above.", "beta", "alpha", "After the table.",
    ]


def test_extract_reads_two_columns_down_each_column(tmp_path):
    """提取顺序左右交错时，写回前的块序仍是先左栏后右栏，拆开的段落能接上。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=320)
    lines = [
        (40, 50, "the training set is given like a"),
        (250, 50, "The other column starts here."),
        (40, 64, "sequence while the target stays put."),
        (250, 64, "It has its own ending."),
    ]
    for index in range(3):
        lines.append((40, 100 + index * 16, f"Left filler line number {index} here."))
        lines.append((250, 100 + index * 16, f"Right filler line number {index} here."))
    for x, y, text in lines:
        page.insert_text((x, y), text, fontsize=10)
    src = tmp_path / "cols.pdf"
    doc.save(src)
    doc.close()

    blocks = extract_blocks(pymupdf.open(src))
    texts = [block.text for block in blocks]
    joined = next(text for text in texts if "sequence while" in text)
    assert joined.startswith("the training")
    assert all("other column" not in text or "sequence" not in text for text in texts)
    left_at = next(index for index, text in enumerate(texts) if text.startswith("Left filler"))
    right_at = next(index for index, text in enumerate(texts) if text.startswith("Right filler") or text.startswith("The other"))
    assert left_at < right_at


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
