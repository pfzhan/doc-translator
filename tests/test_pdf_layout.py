from dataclasses import replace

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


def test_white_rectangle_fill_is_not_a_figure():
    """近白矩形是底色。盖住它的正文不能被当成图内文字跳过。描边框和图片仍是图。"""
    from app.formats.pdf_layout import _figure_rects
    from app.formats.pdf_roles import HeuristicLayout, LayoutItem, Role

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(pymupdf.Rect(40, 30, 300, 340), fill=(1, 1, 1), color=None, width=0)
    page.draw_rect(pymupdf.Rect(40, 30, 180, 120), fill=(1, 1, 1), color=(0, 0, 0), width=1)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8))
    pix.set_rect(pix.irect, (80, 80, 80))
    page.insert_image(pymupdf.Rect(50, 40, 170, 110), pixmap=pix)
    page.insert_text((50, 200), "The remainder of this paper is organized as follows.", fontsize=10)
    rects = _figure_rects(page)
    assert any(rect.y1 <= 120 for rect in rects)
    assert all(rect.y1 < 160 for rect in rects)

    blocks = extract_blocks(doc)
    prose = next(block for block in blocks if "remainder" in block.text)
    geo = PageGeometry.from_page(page, [block.rect for block in blocks])
    roles = HeuristicLayout().classify(
        [LayoutItem(prose.page, pymupdf.Rect(prose.rect), prose.text, prose.size, prose.bold)],
        [geo],
        10,
    )
    assert roles == [Role.BODY]


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


def _cjk_line(page: pymupdf.Page) -> dict | None:
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"])
            if any("\u4e00" <= ch <= "\u9fff" for ch in text):
                return line
    return None


def test_centered_table_cell_keeps_the_translation_centered(tmp_path):
    """原文在格子中线上时，更短的译文也要落在格子中线上，不能从原文字左缘起排。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=160)
    page.insert_text((80, 60), "Baseline", fontsize=10)
    src = tmp_path / "cell.pdf"
    doc.save(src)
    doc.close()

    clip = (40.0, 40.0, 200.0, 80.0)
    block = TextBlock(
        page=0, rect=pymupdf.Rect(97.5, 50, 142.5, 62),
        line_rects=[pymupdf.Rect(97.5, 50, 142.5, 62)],
        text="Baseline", size=10, color="#000000", bold=False, clip=clip,
    )
    out = _render_translated(src, [block], ["基线"], "zh-CN")
    line = _cjk_line(out[0])
    assert line is not None
    mid = (line["bbox"][0] + line["bbox"][2]) / 2
    assert abs(mid - 120.0) < 1.5
    assert line["bbox"][2] <= clip[2] + 2


def test_left_aligned_table_cell_stays_left(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=160)
    page.insert_text((48, 60), "Name", fontsize=10)
    src = tmp_path / "cell.pdf"
    doc.save(src)
    doc.close()

    clip = (40.0, 40.0, 200.0, 80.0)
    block = TextBlock(
        page=0, rect=pymupdf.Rect(46, 50, 80, 62),
        line_rects=[pymupdf.Rect(46, 50, 80, 62)],
        text="Name", size=10, color="#000000", bold=False, clip=clip,
    )
    out = _render_translated(src, [block], ["姓名"], "zh-CN")
    line = _cjk_line(out[0])
    assert line is not None
    assert line["bbox"][0] < 70
    assert (line["bbox"][0] + line["bbox"][2]) / 2 < 90


def test_cell_bottom_line_keeps_the_table_font_size(tmp_path):
    """贴着格子下沿的一行，字号要和同格上一行一样，不能被裁矮后单独缩小。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=160)
    page.insert_text((50, 48), "Baseline", fontsize=10)
    page.insert_text((50, 92), "Scheduled Sampling ensemble of 5", fontsize=10)
    src = tmp_path / "cell.pdf"
    doc.save(src)
    doc.close()

    # 下沿几乎贴着第二行字形。旧逻辑会把这一行的下沿余量裁掉，行框矮过一行。
    clip = (40.0, 30.0, 220.0, 93.0)
    top = TextBlock(
        page=0, rect=pymupdf.Rect(50, 38, 110, 50),
        line_rects=[pymupdf.Rect(50, 38, 110, 50)],
        text="Baseline", size=10, color="#000000", bold=False, clip=clip,
    )
    bottom = TextBlock(
        page=0, rect=pymupdf.Rect(48, 82, 210, 94),
        line_rects=[pymupdf.Rect(48, 82, 210, 94)],
        text="Scheduled Sampling ensemble of 5", size=10, color="#000000", bold=False, clip=clip,
    )
    out = _render_translated(src, [top, bottom], ["基线", "五个模型的计划采样集成"], "zh-CN")
    sizes = [
        max(span["size"] for span in line["spans"])
        for block in out[0].get_text("dict")["blocks"] if block.get("type") == 0
        for line in block["lines"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in "".join(span["text"] for span in line["spans"]))
    ]
    assert len(sizes) == 2
    assert max(sizes) - min(sizes) < 0.6
    assert min(sizes) > 8.0


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
    assert all(block.clip is None for block in extract_blocks(doc))


def test_unruled_short_cells_stay_inside_and_do_not_vote_as_columns(tmp_path):
    """没有线的短格子按列缘对齐认成表，译文不越出格子，也不把页面拆成分栏。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=240)
    rows = [("Name", "Alice"), ("Age", "30"), ("City", "Beijing"), ("Role", "Editor")]
    for index, (left, right) in enumerate(rows):
        page.insert_text((48, 50 + index * 18), left, fontsize=11)
        page.insert_text((140, 50 + index * 18), right, fontsize=11)
    src = tmp_path / "plain-table.pdf"
    doc.save(src)
    doc.close()

    opened = pymupdf.open(src)
    blocks = extract_blocks(opened)
    alice = next(block for block in blocks if block.text == "Alice")
    assert alice.clip is not None and alice.clip[0] >= 130
    assert all(block.clip is not None for block in blocks)
    word_rects = [pymupdf.Rect(word[:4]) for word in opened[0].get_text("words")]
    geo = PageGeometry.from_page(opened[0], word_rects)
    assert len(geo.tables) == 1
    assert len(geo.columns) == 1

    translation = "这是一段很长的译文一定会向右铺开直到越过格子才停下来"
    out = _render_translated(src, blocks, [block.text if block is not alice else translation for block in blocks], "zh-CN")
    lines = [ln for blk in out[0].get_text("dict")["blocks"] if blk.get("type") == 0 for ln in blk["lines"]]
    zh = [ln for ln in lines if any("\u4e00" <= ch <= "\u9fff" for ch in "".join(s["text"] for s in ln["spans"]))]
    assert zh
    assert max(ln["bbox"][2] for ln in zh) <= alice.clip[2] + 2


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


def test_rotated_page_geometry_uses_the_unrotated_size():
    """旋转页的文字坐标没转。页宽页高也按没转的算，和不旋转的同一页一样。"""
    geos = []
    for rotation in (0, 90):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=800)
        page.draw_rect(pymupdf.Rect(60, 560, 300, 760), color=(0, 0, 0))
        page.set_rotation(rotation)
        geos.append(PageGeometry.from_page(page, [*_stack(40, 250, 40), *_stack(330, 560, 40)]))
    plain, rotated = geos
    assert (rotated.width, rotated.height) == (600, 800)
    assert rotated == plain


def _rule_page() -> pymupdf.Page:
    """格子之间有线、没有外框的表：横线伸出竖线，竖线伸过上下两条横线。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for y in (100, 130, 160):
        page.draw_line((50, y), (370, y), width=0.4)
    for x in (130, 210, 290):
        page.draw_line((x, 80), (x, 180), width=0.4)
    for row, y in enumerate((94, 118, 148, 174)):
        for col, x in enumerate((55, 135, 215, 295)):
            page.insert_text((x, y), f"cell {row}{col}", fontsize=9)
    return page


def test_open_edged_table_gets_its_header_row_and_outer_columns():
    from app.formats.pdf_layout import ruled_tables

    [table] = ruled_tables(_rule_page())
    assert table.rect == pymupdf.Rect(50, 80, 370, 180)
    rows = sorted({(round(cell.y0), round(cell.y1)) for cell in table.cells})
    cols = sorted({(round(cell.x0), round(cell.x1)) for cell in table.cells})
    assert rows == [(80, 100), (100, 130), (130, 160), (160, 180)]
    assert cols == [(50, 130), (130, 210), (210, 290), (290, 370)]
    assert len(table.cells) == 16


def test_table_cells_never_continue_into_each_other():
    """格子的列看起来像栏，上一列的末格不能接到下一列的表头上。"""
    from app.formats.pdf import _can_continue

    def cell(text, x0, y0, clip):
        rect = pymupdf.Rect(x0, y0, x0 + 80, y0 + 12)
        return TextBlock(0, rect, [rect], text, 10, "#000", False, clip=clip)

    left = cell("Nearly all models incur extreme difficulty", 50, 160, (50, 130, 150, 180))
    head = cell("deep undirected graphical models", 150, 80, (150, 80, 250, 100))
    assert not _can_continue(left, head, 10, set(), allow_capital=True)
    assert _can_continue(replace(left, clip=None), replace(head, clip=None), 10, set(), allow_capital=True)


def _algorithm_page(caption_below: bool) -> pymupdf.Page:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    code = ["Require: a step size", "while not converged do", "update the parameters", "end while"]
    if caption_below:
        page.draw_rect(pymupdf.Rect(50, 60, 350, 130), width=0.4)
        caption_y = 145
        top = 75
    else:
        page.draw_line((50, 50), (350, 50), width=0.8)
        page.draw_line((50, 66), (350, 66), width=0.4)
        page.draw_line((50, 130), (350, 130), width=0.8)
        caption_y = 62
        top = 80
    page.insert_text((52, caption_y), "Algorithm 1 Gradient descent", fontsize=10)
    for i, line in enumerate(code):
        page.insert_text((60 + 10 * (i in (1, 2)), top + i * 13), line, fontsize=10)
    return page


def test_algorithm_box_lines_become_cells():
    """伪代码一行一格，图注在框上方、在框下方都认。"""
    from app.formats.pdf_layout import algorithm_tables

    for below in (False, True):
        [table] = algorithm_tables(_algorithm_page(below))
        assert len(table.cells) == 4
        assert table.rect.x0 == 50 and table.rect.x1 == 350
        bands = sorted(table.cells, key=lambda cell: cell.y0)
        assert all(a.y1 == b.y0 for a, b in zip(bands, bands[1:]))


def test_algorithm_lines_are_translated_one_by_one():
    page = _algorithm_page(False)
    blocks = [block for block in extract_blocks(page.parent) if not block.text.startswith("Algorithm")]
    assert [block.text for block in blocks] == [
        "Require: a step size", "while not converged do", "update the parameters", "end while",
    ]
    assert all(block.clip is not None for block in blocks)


def test_side_by_side_algorithm_boxes_are_both_found():
    """两个算法框并排时，图注在 PyMuPDF 的同一行里；另一个框的线在旁边，不算。"""
    from app.formats.pdf_layout import algorithm_tables

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=300)
    for x0, x1, name in ((50, 290, "Algorithm 1 Training"), (310, 550, "Algorithm 2 Sampling")):
        for y in (50, 66, 130):
            page.draw_line((x0, y), (x1, y), width=0.5)
        page.insert_text((x0 + 2, 62), name, fontsize=10)
        for i in range(3):
            page.insert_text((x0 + 10, 80 + i * 13), f"{i + 1}: step number {i + 1}", fontsize=10)
    tables = algorithm_tables(page)
    assert sorted(round(table.rect.x0) for table in tables) == [50, 310]
    assert all(len(table.cells) == 3 for table in tables)


def test_algorithm_rows_ignore_radical_origins():
    """根号的起笔点在字形顶上，不能当成一行的基线另切出一格。"""
    from app.formats.pdf_layout import _code_rows

    def line(y, *spans):
        return {"spans": [{"text": text, "font": font, "size": 10.0, "origin": (120, y + dy)} for text, font, dy in spans]}

    lines = [line(300, ("x ← y", "CMMI10", 0)), line(311, ("z ←", "CMMI10", 0), ("√", "CMSY10", -7))]
    rows = _code_rows(lines, pymupdf.Rect(100, 290, 400, 320))
    assert len(rows) == 2


def test_algorithm_rows_ignore_fraction_numerators_and_denominators():
    """行间公式的分子分母和正文一样大，起笔点在基线上下大半个字高。贴着分数线的不另切一格。"""
    from app.formats.pdf_layout import _code_rows

    def span(text, font, x0, x1, y):
        return {"text": text, "font": font, "size": 10.0, "origin": (x0, y), "bbox": (x0, y - 7.5, x1, y + 2.5)}

    def row(y, numerator, denominator, comment):
        return [{"spans": [span("µ ←", "CMMI10", 321, 345, y), span(comment, "Times-Roman", 460, 540, y)]},
                {"spans": [span(numerator, "CMR10", 351, 356, y - 6.7)]},
                {"spans": [span(denominator, "CMMI10", 350, 358, y + 6.8)]}]

    lines = [*row(562.7, "1", "m", "// mini-batch mean"), *row(595.9, "1", "m", "// mini-batch variance")]
    bars = [pymupdf.Rect(349.5, 559.9, 358.3, 560.4), pymupdf.Rect(349.5, 593.1, 358.3, 593.5)]
    body = pymupdf.Rect(307, 541, 543, 610)
    assert len(_code_rows(lines, body, bars)) == 2
    assert len(_code_rows(lines, body)) == 6


def test_a_numbered_algorithm_statement_is_one_unit():
    """行号小一号、句子折成两行时，一格还是一句话；两行各自送翻会剩下半句英文。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for y in (50, 66, 150):
        page.draw_line((50, y), (350, y), width=0.5)
    page.insert_text((52, 62), "Algorithm 2 Training a network", fontsize=10)
    rows = [("1:", "Initialize the network weights", 80), ("2:", "Add the transformation y = f(x) to the", 93),
            ("", "network and train it", 106), ("3:", "Return the trained network", 119)]
    for label, text, y in rows:
        if label:
            page.insert_text((54, y), label, fontsize=8)
        page.insert_text((70, y), text, fontsize=10)
    texts = [block.text for block in extract_blocks(doc) if block.clip is not None]
    assert texts == [
        "1: Initialize the network weights",
        "2: Add the transformation y = f(x) to the network and train it",
        "3: Return the trained network",
    ]


def test_a_displayed_formula_inside_an_algorithm_step_keeps_its_glyphs():
    """一步下面单独居中一行式子：句子照翻，式子留原字，不并进句子里当词翻。
    紧接句子写、不缩进的式子行仍是句子的一部分。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for y in (50, 66, 170):
        page.draw_line((50, y), (350, y), width=0.5)
    page.insert_text((52, 62), "Algorithm 2 Training a network", fontsize=10)
    rows = [("1:", "Average the statistics over them:", 70, 80), ("", "u = v + 1", 160, 93),
            ("2:", "Optimize the parameters of the set", 70, 110), ("", "{a, b} = c", 70, 123),
            ("3:", "Return the trained network", 70, 140)]
    for label, text, x, y in rows:
        if label:
            page.insert_text((54, y), label, fontsize=8)
        page.insert_text((x, y), text, fontsize=10)
    texts = [block.text for block in extract_blocks(doc) if block.clip is not None]
    assert texts == [
        "1: Average the statistics over them:",
        "2: Optimize the parameters of the set {a, b} = c",
        "3: Return the trained network",
    ]
