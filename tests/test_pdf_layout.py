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


def test_a_plot_on_a_white_ground_covers_its_title_and_axis_labels():
    """matplotlib 的图先铺白底，标题、坐标轴名写在曲线外面的白底上。白底装着曲线、又不比曲线大出一倍的是图，
    只算到曲线外面一圈。比里面的图大得多的白底不是图；网格认成表时，比表大一圈的白底仍是图。"""
    from app.formats.pdf_layout import PageGeometry, RuledTable, _figure_rects

    page = pymupdf.open().new_page(width=400, height=500)
    page.draw_rect(pymupdf.Rect(40, 30, 260, 200), fill=(1, 1, 1), color=None, width=0)
    page.draw_polyline([(70, 170), (120, 120), (170, 140), (240, 60)], color=(0, 0, 1), width=1)
    rects = _figure_rects(page)
    assert pymupdf.Rect(46, 36, 260, 194) in rects

    # 白底一直铺到下面的正文里，只算到曲线外面 24pt
    tall = pymupdf.open().new_page(width=400, height=500)
    tall.draw_rect(pymupdf.Rect(40, 30, 260, 230), fill=(1, 1, 1), color=None, width=0)
    tall.draw_polyline([(70, 170), (120, 120), (170, 140), (240, 60)], color=(0, 0, 1), width=1)
    assert all(rect.y1 <= 195 for rect in _figure_rects(tall))

    wide = pymupdf.open().new_page(width=400, height=500)
    wide.draw_rect(pymupdf.Rect(20, 20, 380, 400), fill=(1, 1, 1), color=None, width=0)
    wide.draw_polyline([(70, 170), (120, 120), (170, 140), (240, 60)], color=(0, 0, 1), width=1)
    assert all(rect.width < 200 for rect in _figure_rects(wide))

    grid = RuledTable(pymupdf.Rect(70, 60, 240, 170), ())
    geo = PageGeometry.from_rects(400, [], [pymupdf.Rect(40, 30, 260, 200), pymupdf.Rect(70, 60, 240, 170)], 500, [grid])
    assert geo.figures == (pymupdf.Rect(40, 30, 260, 200),)


def test_a_framed_panel_with_circled_words_is_a_figure():
    """框起来的白底上一段示例文字，词上圈着彩色椭圆、连着虚线（ed3book 的 Figure 4.2）：
    白底是一幅图，里面的字留原文。同样的记号铺在没框的白底上不算。"""
    from app.formats.pdf_layout import _figure_rects

    def panel(framed):
        page = pymupdf.open().new_page(width=612, height=792)
        if framed:  # 框是四条单独的线
            for start, end in (((160, 510), (493, 510)), ((160, 623), (493, 623)),
                               ((160, 510), (160, 623)), ((493, 510), (493, 623))):
                page.draw_line(start, end, color=(0, 0, 0), width=0.4)
        page.draw_rect(pymupdf.Rect(168, 515, 485, 618), fill=(1, 1, 1), color=None, width=0)
        for x0, y0, x1, color in ((183, 544, 211, (1, 0.4, 0)), (417, 544, 465, (1, 0.4, 0)),
                                  (236, 556, 279, (0.8, 0.9, 0.1)), (326, 579, 339, (0.3, 0.2, 0.9))):
            page.draw_oval(pymupdf.Rect(x0, y0, x1, y0 + 10), fill=color, color=(0, 0, 0), width=0.4)
        for start, end in (((208, 545), (261, 527)), ((283, 524), (420, 546)), ((187, 577), (225, 602)),
                           ((404, 600), (448, 586))):
            page.draw_line(start, end, color=(0, 0, 0), width=0.4, dashes="[3 3] 0")
        return _figure_rects(page)

    assert any(rect.contains(pymupdf.Rect(170, 520, 480, 610)) for rect in panel(framed=True))
    assert not any(rect.contains(pymupdf.Rect(170, 520, 480, 610)) for rect in panel(framed=False))


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


def test_the_indented_next_line_does_not_stop_widening():
    """索引条目下一行是缩进的子条目，写入框比原文行多出的留白探进下一行，不是并排的障碍。
    算法框和段落首行并排十几点，就算比段落高很多也要让开。"""
    from app.formats.pdf import _beside_rows

    entry_row = pymupdf.Rect(197, 130.4, 234, 140.8)
    entry_box = pymupdf.Rect(197, 131.2, 236, 144.8)
    sub = pymupdf.Rect(205, 139.9, 254, 150.3)
    beside = pymupdf.Rect(300, 132, 340, 141)
    assert _beside_rows([sub, beside], entry_box, [entry_row]) == [beside]
    rows = [pymupdf.Rect(72, 651 + 12 * i, 303, 662 + 12 * i) for i in range(6)]
    algorithm = pymupdf.Rect(308, 497, 543, 665)
    assert _beside_rows([algorithm], pymupdf.Rect(72, 650.7, 303, 723), rows) == [algorithm]
    # 框往下多出两行，那里的障碍也算
    low = pymupdf.Rect(310, 735, 400, 746)
    assert _beside_rows([low], pymupdf.Rect(72, 650.7, 303, 750), rows) == [low]



def test_an_equation_number_between_two_rows_is_beside_them():
    """分情况的行间公式，右边的式号夹在「if …」「otherwise」两行中间，和每一行都只交叠一小半，
    也在这两行旁边，右扩要让开。在最后一行下面的不算。"""
    from app.formats.pdf import _beside_rows, _on_rows

    rows = [pymupdf.Rect(351.4, 215.1, 422.0, 227.1), pymupdf.Rect(351.4, 229.4, 390.1, 241.4)]
    number = pymupdf.Rect(471.5, 222.8, 493.2, 233.6)
    below = pymupdf.Rect(471.5, 238.0, 493.2, 248.8)
    assert _on_rows(number, rows)
    assert not _on_rows(below, rows)
    assert _beside_rows([number], pymupdf.Rect(351.4, 214, 470, 243), rows) == [number]

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


def test_a_table_column_with_body_text_below_does_not_continue_on_the_next_page():
    """图里的表格右边一列格子左缘对齐，分栏时认成了一栏；这一列底下接着通栏的正文，末一格不是页末，
    不接到下一页页首的段落上。"""
    def blk(page, text, rows, size=10.0):
        rects = [pymupdf.Rect(row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=page, rect=rect, line_rects=rects, text=text, size=size, color="#000", bold=False)

    body = [(142.2, y, 475.2, y + 12) for y in (110, 122, 134, 146)]
    blocks = [
        blk(0, "annual challenges included a slot filling task, which is the task of filling in values.",
            [*body, (142.2, 158, 229.1, 170)]),
        blk(0, "Example", [(80.4, 180.1, 118.0, 193.1)]),
        blk(0, "Entity Types & Label", [(371.4, 180.1, 462.8, 193.1)]),
        blk(0, "Carey will succeed Cathleen P. Black, who held the position at IBM",
            [(80.4, 192.8, 354.0, 204.8), (80.4, 204.8, 354.0, 216.8), (80.4, 216.7, 177.9, 228.7)]),
        blk(0, "PERSON/TITLE Relation: per:title", [(371.7, 192.8, 432.5, 204.8), (371.4, 204.8, 443.1, 216.8)]),
        blk(0, "Irene Morgan Kirkaldy, who was born and reared in Baltimore, Maryland",
            [(80.4, 228.7, 354.0, 240.7), (80.4, 240.6, 354.0, 252.6), (80.4, 252.6, 188.1, 264.6)]),
        blk(0, "PERSON/CITY Relation: per:city of birth", [(371.7, 228.7, 428.5, 240.7), (371.4, 240.6, 476.3, 252.6)]),
        blk(0, "Baldwin declined further comment, and said JetBlue chief executive Dave Barger",
            [(80.4, 264.6, 354.0, 276.6), (80.4, 276.5, 198.8, 288.5)]),
        blk(0, "Types: PERSON/TITLE Relation: no relation", [(371.4, 264.6, 461.9, 276.6), (371.4, 276.5, 455.8, 288.5)]),
        blk(0, "A standard dataset was also produced for the SemEval 2010 task on relations between nominals.",
            [(157.1, 313.1, 475.2, 325.1), (142.2, 325.0, 475.2, 337.1), (142.2, 337.0, 475.2, 349.0),
             (142.2, 349.0, 475.2, 361.0), (142.2, 360.9, 283.5, 372.9)]),
        blk(0, "Hearst points out that most human readers will not know what Gelidium is.",
            [(142.2, 582.4, 475.2, 594.4), (142.2, 594.0, 475.2, 607.0), (142.2, 605.9, 368.2, 618.9)]),
        blk(0, "allowing us to infer", [(142.2, 693.0, 219.7, 705.0)]),
        blk(1, "Figure 21.4 shows five patterns Hearst (1992a, 1998) suggested for inferring the hyponym relation.",
            [(175.1, 203.0, 493.2, 215.0), (160.2, 215.0, 493.2, 230.2), (160.2, 226.9, 493.2, 238.9),
             (160.2, 238.9, 493.2, 250.9), (160.2, 250.8, 377.1, 262.8)]),
        blk(1, "Hand-built patterns have the advantage of high-precision and they can be tailored to specific domains.",
            [(175.1, 368.5, 493.2, 380.5), (160.2, 380.4, 493.2, 392.4), (160.2, 392.4, 337.7, 404.4)]),
        blk(1, "These include word features (as embeddings, or 1-hot, stemmed or not):", [(160.2, 679.2, 449.0, 692.1)]),
    ]
    cell, head = blocks[8], blocks[-3]
    geos = [PageGeometry.from_rects(612, [b.rect for b in blocks if b.page == page], height=792) for page in (0, 1)]
    units = _cross_page_units(blocks, 10.0, geometries=geos)
    assert [cell] in units and [head] in units


def test_a_paragraph_continues_below_a_figure_at_the_top_of_the_next_page():
    """下一页先排了一张图和图注，被页边界切断的那段接在图注下面。"""
    end = _block("you can see the ex-", 0, pymupdf.Rect(40, 700, 300, 712))
    caption = _block("Windowing, showing a 25 ms window.", 1, pymupdf.Rect(60, 260, 300, 272))
    rest = _block("tracted windowed signal looks just like the original.", 1, pymupdf.Rect(40, 300, 300, 312))
    blocks = [end, caption, rest]
    assert all(len(unit) == 1 for unit in _cross_page_units(blocks, 10.0))
    units = _cross_page_units(blocks, 10.0, aside_ids={id(caption)})
    assert [end, rest] in units and [caption] in units


def test_the_left_column_end_skips_the_author_name_over_the_right_column():
    """右栏顶上是比正文大的署名，左栏末句接到署名下面的右栏正文，不接署名。"""
    geo = _two_columns()
    left = [
        _block("Training deep networks is hard because the inputs change.", 0, pymupdf.Rect(40, 200, 150, 240)),
        _block("The mini-batch approximates the gradient by computing", 0, pymupdf.Rect(40, 660, 150, 712)),
    ]
    name = _block("Christian Szegedy", 0, pymupdf.Rect(270, 60, 340, 74), size=12)
    right = [
        _block("Using mini-batches of examples is helpful in several ways.", 0, pymupdf.Rect(250, 150, 360, 190)),
        _block("Second, computation over a batch can be much more efficient.", 0, pymupdf.Rect(250, 200, 360, 240)),
    ]
    units = _cross_page_units([*left, name, *right], 10.0, geometries=[geo])
    assert [unit for unit in units if len(unit) == 2] == [[left[1], right[0]]]


def test_index_entries_hold_their_place_at_the_column_top():
    """索引条目不接上下文，但照样占着栏顶：右栏底一个没页码的词头（BERT）不能越过下一页
    栏顶的一串条目，接到栏中间另一个没页码的词头（broadcast news）上。"""
    geo = _two_columns()
    above = _block("Bayes", 0, pymupdf.Rect(250, 680, 280, 690), size=8)
    head = _block("BERT", 0, pymupdf.Rect(250, 700, 280, 710), size=8)
    entries = [_block(f"bigram term, {70 + i}", 1, pymupdf.Rect(40, 60 + 10 * i, 120, 70 + 10 * i), size=8)
               for i in range(5)]
    for entry in entries:
        entry.entry = True
    later = [_block("broadcast news", 1, pymupdf.Rect(40, 110, 100, 120), size=8),
             _block("cascade", 1, pymupdf.Rect(40, 140, 100, 150), size=8)]
    blocks = [above, head, *entries, *later]
    halt = {id(entry) for entry in entries}
    units = _cross_page_units(blocks, 8.0, geometries=[geo, geo], halt_ids=halt)
    assert all(len(unit) == 1 for unit in units)


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


def test_translation_stops_before_a_ruled_box_on_the_right(tmp_path):
    """栏间空白太窄没分出栏时，右边算法框的外框也挡住右扩。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((40, 52), "Short left", fontsize=12)
    src = tmp_path / "box.pdf"
    doc.save(src)

    box = pymupdf.Rect(152, 30, 362, 120)
    geo = PageGeometry.from_rects(
        400, [pymupdf.Rect(40, 40, 360, 54)], height=400, tables=[RuledTable(box, (), True)],
    )
    assert len(geo.columns) == 1
    block = TextBlock(
        page=0, rect=pymupdf.Rect(40, 40, 90, 54), line_rects=[pymupdf.Rect(40, 40, 90, 54)],
        text="Short left", size=12, color="#000000", bold=False,
    )
    translation = "这是一段很长的译文一定会向右铺开直到碰上右边的算法框才停下来"
    out = _render_translated(src, [block], [translation], "zh-CN", geometries=[geo])
    lines = [ln for blk in out[0].get_text("dict")["blocks"] if blk.get("type") == 0 for ln in blk["lines"]]
    zh = [ln for ln in lines if any("\u4e00" <= ch <= "\u9fff" for ch in "".join(s["text"] for s in ln["spans"]))]
    assert zh
    right_edge = max(ln["bbox"][2] for ln in zh)
    assert right_edge > block.rect.x1 + 1
    assert right_edge < box.x0


def test_translation_stops_before_text_left_in_place_on_the_right(tmp_path):
    """不属于任何块、原处留着的字（式号）也挡住右扩。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((40, 52), "Short left", fontsize=12)
    page.insert_text((200, 52), "(17)", fontsize=12)
    src = tmp_path / "kept.pdf"
    doc.save(src)

    geo = PageGeometry.from_rects(400, [pymupdf.Rect(40, 40, 360, 54)], height=400)
    block = TextBlock(
        page=0, rect=pymupdf.Rect(40, 40, 90, 54), line_rects=[pymupdf.Rect(40, 40, 90, 54)],
        text="Short left", size=12, color="#000000", bold=False,
    )
    translation = "这是一段很长的译文一定会向右铺开直到碰上右边留着的式号才停下来"
    out = _render_translated(src, [block], [translation], "zh-CN", geometries=[geo])
    lines = [ln for blk in out[0].get_text("dict")["blocks"] if blk.get("type") == 0 for ln in blk["lines"]]
    zh = [ln for ln in lines if any("\u4e00" <= ch <= "\u9fff" for ch in "".join(s["text"] for s in ln["spans"]))]
    assert zh
    assert "(17)" in out[0].get_text()
    assert max(ln["bbox"][2] for ln in zh) < 200


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


def test_nested_boxes_of_a_diagram_are_not_a_table():
    """流程图的框套框：find_tables 认出的格子互相包含，这不是表。当表的话框里的字全并成一格。"""
    from app.formats.pdf_layout import ruled_tables

    doc = pymupdf.open()
    page = doc.new_page(width=720, height=405)
    page.draw_rect(pymupdf.Rect(192, 123, 687, 349), color=(0.5, 0.5, 0.5), fill=(0.9, 0.9, 0.9))
    page.draw_rect(pymupdf.Rect(541, 74, 715, 159), color=(0.3, 0.4, 0.8), fill=(1, 1, 1))
    page.insert_text((210, 200), "Node for each node in Input", fontsize=10)
    page.insert_text((360, 200), "Rule: specifies an Operator", fontsize=10)
    page.insert_text((550, 90), "Add Rule matches to Queue", fontsize=10)
    assert ruled_tables(page) == ()
    texts = [block.text for block in extract_blocks(doc)]
    assert "Add Rule matches to Queue" in texts
    assert "Node for each node in Input" in texts


def test_ruled_tables_read_only_the_text_near_the_lines_and_find_the_same_tables():
    """find_tables 只取线所在一带的字时，找出的表和取整页的字一样。
    几条线连成的一片缺右边和下边，find_tables 看到片里有字才补上外框，比的是上下翻过来的位置，
    所以翻过来那一带的字也要取。只取线附近的字就少了这张表。"""
    from app.formats.pdf_layout import _page_drawings, _table_clip, ruled_tables

    def found(page, clip):
        pymupdf.no_recommend_layout()
        tables = page.find_tables(
            clip=clip, vertical_strategy="lines_strict", horizontal_strategy="lines_strict", paths=_page_drawings(page),
        ).tables
        return [pymupdf.Rect(table.bbox) for table in tables]

    for flipped_text in (False, True):
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=600)
        page.draw_line((40, 70), (260, 70))
        page.draw_line((40, 70), (40, 160))
        page.draw_line((40, 115), (260, 115))
        page.draw_line((150, 70), (150, 160))
        for x, y in ((60, 95), (170, 95), (60, 140), (170, 140)):
            page.insert_text((x, y), "ab", fontsize=9)
        if flipped_text:
            page.insert_text((60, 485), "Text at the flipped height", fontsize=9)
        whole = found(page, None)
        assert whole == ([pymupdf.Rect(40, 70, 260, 160)] if flipped_text else [])
        assert found(page, _table_clip(page, _page_drawings(page))) == whole
        assert [table.rect for table in ruled_tables(page)] == whole
        if flipped_text:
            assert found(page, pymupdf.Rect(35, 65, 265, 165)) == []


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


def test_algorithm_boxes_are_looked_for_only_on_pages_naming_one():
    """给了本页的词时，词里有 Algorithm 才逐行找图注；找出的框和不给词时一样。"""
    from app.formats.pdf_layout import algorithm_tables, page_words

    page = _algorithm_page(False)
    assert algorithm_tables(page, words=page_words(page)) == algorithm_tables(page) != ()
    plain = pymupdf.open().new_page(width=400, height=300)
    plain.draw_line((50, 50), (350, 50), width=0.8)
    plain.draw_line((50, 130), (350, 130), width=0.8)
    plain.insert_text((52, 62), "Figure 1 Gradient descent", fontsize=10)
    assert algorithm_tables(plain, words=page_words(plain)) == ()


def test_algorithm_lines_are_translated_one_by_one():
    page = _algorithm_page(False)
    blocks = [block for block in extract_blocks(page.parent) if not block.text.startswith("Algorithm")]
    assert [block.text for block in blocks] == [
        "Require: a step size", "while not converged do", "update the parameters", "end while",
    ]
    assert all(block.clip is not None for block in blocks)


def test_table_cells_stay_original_but_algorithm_lines_are_translated(tmp_path):
    """表格里的字一律留原文，图注照翻。算法框的伪代码也切成格子，仍逐行翻译。"""
    from app.formats.pdf import _plan_pdf

    class Runner:
        def report_stage(self, *args, **kwargs):
            pass

        def set_title(self, title):
            pass

    page = _algorithm_page(False)
    for x0, x1, word in ((50, 200, "Shakespeare corpus"), (200, 350, "Brown corpus")):
        page.draw_rect(pymupdf.Rect(x0, 170, x1, 200), color=(0, 0, 0), width=0.8)
        page.insert_text((x0 + 8, 189), word, fontsize=10)
    page.insert_text((50, 220), "Table 1 Rough numbers of types for some English corpora.", fontsize=10)
    src = tmp_path / "tables.pdf"
    page.parent.save(src)
    plan = _plan_pdf(src, Runner(), "")
    skipped = {text: skip for text, skip in zip(plan.texts, plan.skips)}
    assert skipped["Shakespeare corpus"] and skipped["Brown corpus"]
    assert not skipped["while not converged do"] and not skipped["update the parameters"]
    assert not any(skip for text, skip in skipped.items() if "Rough numbers" in text)


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


def test_unnumbered_algorithm_rows_join_full_width_wraps():
    """没有行号的算法里，排到框右缘、句子没完的行折到下一行，两行是一格。"""
    from app.formats.pdf_layout import _code_rows

    def line(y, text, x1):
        return {"spans": [{"text": text, "font": "Times-Roman", "size": 10.0,
                           "origin": (140, y), "bbox": (140, y - 8, x1, y + 2)}]}

    body = pymupdf.Rect(108, 115, 504, 200)
    lines = [
        line(125, "• Sample noise samples from noise prior.", 400),
        line(137, "• Sample minibatch of examples from data generating distribution", 504),
        line(149, "pdata(x).", 172),
        line(161, "The updates can use any learning rule. We used momen-", 504),
        line(173, "tum in our experiments.", 213),
        line(185, "end for", 160),
    ]
    rows = _code_rows(lines, body)
    assert len(rows) == 4
    assert rows[1].y0 < 137 < 149 < rows[1].y1
    assert rows[2].y0 < 161 < 173 < rows[2].y1


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


def test_an_algorithm_formula_beside_a_distant_comment_keeps_its_glyphs():
    """算法一步的行间公式和右边远处的注释在同一格：注释照翻，公式留原字，不压成一行高塞进注释里。
    紧挨着正文的式子仍是句子的一部分。"""
    from app.formats.pdf import _merge_cells

    clip = (307.7, 576.4, 543.5, 608.1)

    def part(text, font, x0, x1):
        rect = pymupdf.Rect(x0, 585.0, x1, 607.0)
        span = {"text": text, "font": font, "size": 10.0, "bbox": tuple(rect), "origin": (x0, 597.0)}
        return TextBlock(2, rect, [rect], text, 10.0, "#000", False, span_lines=[[span]], clip=clip)

    formula = part("σ ← (x − µ)", "CMMI10", 321.2, 420.0)
    comment = part("// mini-batch variance", "Times-Roman", 452.8, 540.2)
    merged = _merge_cells([formula, comment], {id(formula), id(comment)})
    assert [block.text for block in merged] == ["// mini-batch variance"]

    near = part("Set the parameters", "Times-Roman", 425.0, 540.2)
    merged = _merge_cells([formula, near], {id(formula), id(near)})
    assert [block.text for block in merged] == ["σ ← (x − µ) Set the parameters"]


def test_far_apart_labels_in_one_box_stay_separate():
    """框里隔开的几个节点标签不是一句话，各翻各的、各写回各的位置。挨着的换行仍并成一句。"""
    from app.formats.pdf import _merge_cells

    clip = (388.0, 163.0, 687.0, 373.0)

    def part(text, x0, y0, x1):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 13.4)
        return TextBlock(23, rect, [rect], text, 12.0, "#000", False, clip=clip)

    labels = [part("Scan [Emps]", 430, 287.1, 498), part("Aggregate [manager]", 554, 287.1, 666.7),
              part("Scan [Emps]", 579, 341.1, 647)]
    assert [block.text for block in _merge_cells(labels, set())] == ["Scan [Emps]", "Aggregate [manager]", "Scan [Emps]"]

    wrapped = [part("A long sentence in the box that", 400, 200, 660), part("wraps onto the next line", 400, 214, 600)]
    assert [block.text for block in _merge_cells(wrapped, set())] == ["A long sentence in the box that wraps onto the next line"]


def test_book_headers_above_the_text_frame_are_margins():
    """页眉在 y=84，比固定的页顶窄带低。页码和章号每页不同；只出现一次的小节页眉
    和其他页眉落在同一高度。版心里的标题不算。"""
    from app.formats.pdf_frame import TextFrame

    frame = TextFrame(142, 475, 111, 719, 10)
    geos = [PageGeometry.from_rects(612, [pymupdf.Rect(142, 112, 475, 700)], height=792, frame=frame)
            for _ in range(5)]
    blocks = [
        _block("2 CHAPTER 2", 0, pymupdf.Rect(77, 84, 143, 94), size=8),
        _block("4 CHAPTER 2", 1, pymupdf.Rect(77, 84, 143, 94), size=8),
        _block("6 CHAPTER 2", 2, pymupdf.Rect(77, 84, 143, 94), size=8),
        _block("2.3 • UNICODE", 3, pymupdf.Rect(398, 84, 476, 94), size=8),
        _block("2.3 Unicode", 4, pymupdf.Rect(95, 118, 192, 135), size=17),
    ]
    assert margin_skips(blocks, geos) == [True, True, True, True, False]


def test_word_by_word_gloss_is_found_but_a_table_is_not():
    from app.formats.pdf_layout import interlinear_glosses

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    source = ["keoi5", "waa6", "cyun4", "gwok3", "zeoi3", "daai6"]
    gloss = ["he", "say", "entire", "country", "most", "big"]
    for index, (word, under) in enumerate(zip(source, gloss)):
        x = 174 + index * 40
        page.insert_text((x, 290), word, fontsize=10)
        page.insert_text((x, 302), under, fontsize=10)
    page.insert_text((174, 316), "He said the biggest house in the country was this one", fontsize=10)
    for row in range(4):
        for col, word in enumerate(("alpha", "beta", "gamma", "delta")):
            page.insert_text((174 + col * 60, 420 + row * 12), f"{word}{row}", fontsize=10)
    [gloss_rect] = interlinear_glosses(page)
    assert 275 < gloss_rect.y0 and gloss_rect.y1 < 306


def test_indented_examples_are_not_a_column_that_ends_the_page():
    """缩进的例句落在正文栏中间，不是另一栏。它没有句末标点，也不能接下一页的段首。"""
    def body(text: str, page: int, y0: float, x0: float = 160, x1: float = 493) -> TextBlock:
        return _block(text, page, pymupdf.Rect(x0, y0, x1, y0 + 30))

    page0 = [
        body("The distinctions get even harder to make once we think about other languages.", 0, 100),
        body("YaoMing reaches finals", 0, 150, 192, 286),
        body("But the same sentence could be treated as five words in another standard.", 0, 200),
        body("Yao Ming reaches overall finals", 0, 250, 192, 318),
        body("How many words are there in English? We call them word types.", 0, 600),
    ]
    page1 = [
        body("You will notice that the larger the corpora we look at, the more types we find!", 1, 112, 142, 475),
        body("The value of beta depends on the corpus size and the genre of the text.", 1, 231, 142, 475),
    ]
    geo = PageGeometry.from_rects(612, [block.rect for block in page0], height=792)
    units = _cross_page_units([*page0, *page1], 10.0, geometries=[geo, geo])
    assert all(len(unit) == 1 for unit in units)


def test_short_and_morpheme_glosses_are_found():
    """三个词的汉字对译，和逐语素对译（带 1SG、PFV 这样的缩写）。两行英文短句碰巧对齐不算。"""
    from app.formats.pdf_layout import interlinear_glosses

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for x, word, under in ((192, "姚明", "YaoMing"), (232, "进入", "reaches"), (264, "总决赛", "finals")):
        page.insert_text((x, 455), word, fontsize=10, fontname="china-s")
        page.insert_text((x, 468), under, fontsize=10)
    page.insert_text((174, 397), "t-@-nk'e-mejN-@-jetem@-nni-k", fontsize=10)
    page.insert_text((174, 409), "1SG.S-E-midnight-big-E-yurt.cover-E-sew-1SG.S[PFV]", fontsize=10)
    for x, word, under in ((192, "Alice", "Smith"), (260, "Bob", "Jones"), (330, "Carol", "White")):
        page.insert_text((x, 600), word, fontsize=10)
        page.insert_text((x, 613), under, fontsize=10)
    found = sorted(interlinear_glosses(page), key=lambda rect: rect.y0)
    assert len(found) == 2
    assert found[0].y0 < 400 < found[0].y1 and found[1].y0 < 460 < found[1].y1


def test_pinyin_under_a_line_of_hanzi_is_a_gloss():
    """汉字一行、拼音一行，一字一音节，声调抽成单独的修饰字母。下面的英文意译照翻。
    汉字下面接一句音节数对不上的英文不算。"""
    from app.formats.pdf_layout import _TextWord, interlinear_glosses

    def row(y: float, *texts: str) -> list[_TextWord]:
        words, x = [], 164.0
        for text in texts:
            words.append(_TextWord(x, y, x + 6 * len(text), y + 10, text))
            x += 6 * len(text) + 4
        return words

    words = [
        *row(412, "在我的后园，可以看见墙外"),
        *row(424, "z`ai", "wˇo", "de", "h`ou", "yu´an，kˇe", "yˇı", "k`an", "ji`an", "qi´ang", "w`ai"),
        *row(436, "In", "my", "backyard,", "you", "can", "see", "beyond", "the", "fence"),
        *row(590, "我们在后园里看见了两株树"),
        *row(602, "We", "saw", "two", "trees", "in", "the", "garden", "behind", "the", "house"),
    ]
    page = pymupdf.open().new_page(width=612, height=792)
    [gloss] = interlinear_glosses(page, words)
    assert (gloss.y0, gloss.y1) == (412, 434)


def test_page_without_two_ruled_lines_each_way_skips_the_table_search(monkeypatch):
    """凑不出两横两竖的线就围不出格子：脚注线、色块背景不值得逐字解析一遍整页。"""
    from app.formats.pdf_layout import ruled_tables

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.draw_line((50, 250), (150, 250), width=0.4)
    page.draw_rect(pymupdf.Rect(50, 60, 350, 120), color=None, fill=(0.9, 0.9, 0.9))
    page.insert_text((60, 90), "A shaded box of body text", fontsize=10)

    searched: list[int] = []
    original = pymupdf.Page.find_tables

    def counted(self, *args, **kwargs):
        searched.append(self.number)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "find_tables", counted)
    assert ruled_tables(page) == ()
    assert searched == []


def test_table_drawn_with_dashed_rules_is_still_found():
    from app.formats.pdf_layout import ruled_tables

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for y in (80, 100, 130, 160, 180):
        for x in range(50, 370, 4):
            page.draw_line((x, y), (min(x + 3, 370), y), width=0.4)
    for x in (50, 130, 210, 290, 370):
        for y in range(80, 180, 4):
            page.draw_line((x, y), (x, min(y + 3, 180)), width=0.4)
    for row, y in enumerate((94, 118, 148, 174)):
        for col, x in enumerate((55, 135, 215, 295)):
            page.insert_text((x, y), f"cell {row}{col}", fontsize=9)
    [table] = ruled_tables(page)
    assert len(table.cells) == 16


def test_table_ruled_with_one_pixel_images_is_found():
    """老 TeX 用拉伸的一像素图片画表格线，页面上没有矢量线。"""
    from app.formats.pdf_layout import ruled_tables

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    dot = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 1, 1), False)
    dot.clear_with(0)
    for y in (80, 100, 120, 140):
        page.insert_image(pymupdf.Rect(50, y, 290, y + 0.5), pixmap=dot, keep_proportion=False)
    for x in (50, 130, 210, 290):
        page.insert_image(pymupdf.Rect(x, 80, x + 0.5, 140), pixmap=dot, keep_proportion=False)
    for row, y in enumerate((94, 114, 134)):
        for col, x in enumerate((55, 135, 215)):
            page.insert_text((x, y), f"cell {row}{col}", fontsize=9)
    assert page.get_drawings() == []
    [table] = ruled_tables(page)
    assert len(table.cells) == 9


def test_rows_without_rules_between_them_are_split_into_cells():
    """只画了列线的表，表身几行叠在一格里，按对齐的行切开。折行的表头有的格只有一行，不切。"""
    from app.formats.pdf_layout import page_tables

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for y in (40, 70, 140):
        page.draw_line((50, y), (350, y), width=0.4)
    for x in (50, 150, 250, 350):
        page.draw_line((x, 40), (x, 140), width=0.4)
    page.insert_text((55, 58), "Model", fontsize=9)
    page.insert_text((155, 52), "Steps to", fontsize=9)
    page.insert_text((155, 64), "72.2%", fontsize=9)
    page.insert_text((255, 58), "Accuracy", fontsize=9)
    for row, (name, steps, top) in enumerate((("Inception", "31.0e6", "72.2%"), ("BN-x5", "2.1e6", "73.0%"),
                                              ("BN-x30", "2.7e6", "74.8%"))):
        y = 85 + row * 15
        page.insert_text((55, y), name, fontsize=9)
        page.insert_text((155, y), steps, fontsize=9)
        page.insert_text((255, y), top, fontsize=9)
    [table] = page_tables(page)
    heads = [cell for cell in table.cells if cell.y1 <= 71]
    body = [cell for cell in table.cells if cell.y0 >= 69]
    assert len(heads) == 3
    assert len(body) == 9


def test_header_between_top_and_mid_rules_joins_the_unruled_table():
    """三线表的表头比数据列宽，左右缘对不上。顶线和中线之间的字按列分格并进表里。"""
    from app.formats.pdf_layout import page_tables

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    for y in (100, 116, 200):
        page.draw_line((80, y), (500, y), width=0.4)
    page.insert_text((85, 111), "Reverse process time (T - t + 1)", fontsize=10)
    page.insert_text((255, 111), "Rate (bits/dim)", fontsize=10)
    page.insert_text((370, 111), "Distortion (RMSE)", fontsize=10)
    for row, y in enumerate((130, 142, 154, 166, 178, 190)):
        page.insert_text((150, y), str(1000 - row * 100), fontsize=10)
        page.insert_text((270, y), f"0.{row}1507", fontsize=10)
        page.insert_text((390, y), f"{row}0.80948", fontsize=10)
    [table] = page_tables(page)
    assert round(table.rect.y0) == 100
    heads = [cell for cell in table.cells if cell.y1 <= 117]
    assert len(heads) == 3
    assert (round(heads[0].x0), round(heads[-1].x1)) == (80, 500)
    assert 225 < heads[0].x1 < 255 and 330 < heads[1].x1 < 370


def test_rows_of_one_shaded_column_split_by_blanks_are_a_table():
    """逐行底色的一列表：每行是例句、填空横线和答案，隔着大空档。每行一句连着排的底纹不是表。"""
    from app.formats.pdf_layout import filled_tables, page_words

    blue = (0.77, 0.86, 0.99)

    def page_of(rows):
        doc = pymupdf.open()
        page = doc.new_page(width=600, height=400)
        for row, (left, right) in enumerate(rows):
            y0 = 60 + row * 12
            page.draw_rect(pymupdf.Rect(170, y0, 470, y0 + 12), color=None, fill=blue)
            page.insert_text((175, y0 + 9), left, fontsize=9)
            if right:
                page.draw_line((330, y0 + 8), (360, y0 + 8), width=0.4)
                page.insert_text((370, y0 + 9), right, fontsize=9)
        return page

    blanks = page_of([("With roses, dahlias, and peonies, I was", "[flowers]"), ("The room was not just big it was", "[enormous]"),
                      ("The square root of 4 is", "[2]"), ("The professor said that", "[he]")])
    [table] = filled_tables(blanks, blanks.get_drawings(), page_words(blanks))
    assert len(table.cells) == 4
    code = page_of([("for word in words:", ""), ("    counts[word] += 1", ""), ("print(counts)", "")])
    assert filled_tables(code, code.get_drawings(), page_words(code)) == ()


def test_a_table_ruled_only_between_columns_is_found():
    """没有外框、只有列间竖线的表，数字居中排。两栏正文中间的竖线两边是长句，不是表。"""
    from app.formats.pdf_layout import page_tables

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    for x in (290, 343):
        page.draw_line((x, 82), (x, 137), width=0.4)
    page.draw_line((210, 93), (402, 93), width=0.4)
    rows = [("Model", "MNIST", "TFD"), ("DBN [3]", "138 ± 2", "1909 ± 66"), ("Stacked CAE [3]", "121 ± 1.6", "2110 ± 50"),
            ("Deep GSN [6]", "214 ± 1.1", "1890 ± 29"), ("Adversarial nets", "225 ± 2", "2057 ± 26")]
    for row, cells in enumerate(rows):
        y = 90 + row * 11
        for x, text in zip((220, 300, 352), cells):
            page.insert_text((x, y), text, fontsize=8)
    [table] = page_tables(page)
    assert len(table.cells) == 15

    prose = pymupdf.open().new_page(width=600, height=400)
    prose.draw_line((300, 60), (300, 200), width=0.4)
    for row in range(10):
        y = 70 + row * 12
        prose.insert_text((60, y), "a long line of running text in the left column here", fontsize=9)
        prose.insert_text((310, y), "and another long line of text in the right column", fontsize=9)
    assert page_tables(prose) == ()


def test_a_three_rule_table_is_found_whole():
    """顶线、中线、底线一样长的三线表：表头折行，几组数据之间还隔着横线，整张表都认出来。
    三条线之间排的是正文、伪代码的，不是表。"""
    from app.formats.pdf_layout import page_tables

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    for y in (100, 126, 170, 196):
        page.draw_line((120, y), (490, y), width=0.4)
    page.insert_text((130, 110), "Layer Type", fontsize=8)
    page.insert_text((230, 110), "Complexity per Layer", fontsize=8)
    page.insert_text((340, 110), "Sequential", fontsize=8)
    page.insert_text((340, 120), "Operations", fontsize=8)
    page.insert_text((410, 110), "Maximum Path Length", fontsize=8)
    rows = [("Self-Attention", "O(n2 d)", "O(1)", "O(1)"), ("Recurrent", "O(n d2)", "O(n)", "O(n)"),
            ("Convolutional", "O(k n d2)", "O(1)", "O(logk(n))"), ("Restricted", "O(r n d)", "O(1)", "O(n/r)"),
            ("Big", "O(n)", "O(1)", "O(1)")]
    for row, cells in enumerate(rows):
        y = 136 + row * 10 + (6 if row == 4 else 0)
        for x, text in zip((130, 240, 345, 420), cells):
            page.insert_text((x, y), text, fontsize=8)
    [table] = page_tables(page)
    assert table.rect.y0 <= 100 and table.rect.y1 >= 196
    assert len(table.cells) == 4 + 1 + 5 * 4

    prose = pymupdf.open().new_page(width=600, height=400)
    for y in (60, 76, 160):
        prose.draw_line((100, y), (500, y), width=0.4)
    prose.insert_text((110, 70), "Notes on the training run", fontsize=9)
    for row in range(6):
        prose.insert_text((110, 88 + row * 12), "for each step compute the gradient and update the weights", fontsize=9)
    assert page_tables(prose) == ()


def test_a_three_rule_table_with_group_names_is_found():
    """几组数据各有一行组名、组名下面是横线隔开的数据，组名只占一格，整张仍是表。"""
    from app.formats.pdf_layout import page_tables

    page = pymupdf.open().new_page(width=600, height=400)
    for y in (100, 113, 126, 152, 165, 191):
        page.draw_line((340, y), (500, y), width=0.4)
    for x, text in zip((345, 440, 480), ("Objective", "IS", "FID")):
        page.insert_text((x, 110), text, fontsize=7)
    for top, name in ((123, "mu prediction (baseline)"), (162, "eps prediction (ours)")):
        page.insert_text((345, top), name, fontsize=7)
        for row, cells in enumerate((("L, learned diagonal", "7.28", "23.69"), ("L, fixed isotropic", "8.06", "13.22"))):
            for x, text in zip((345, 435, 478), cells):
                page.insert_text((x, top + 13 + row * 9), text, fontsize=7)
    [table] = page_tables(page)
    assert table.rect.y0 <= 100 and table.rect.y1 >= 191
    assert len(table.cells) == 3 + 2 * (1 + 2 * 3)


def test_a_two_column_three_rule_table_with_long_row_names_is_found():
    """两列的三线表，行名占了大半宽也是表。线间排着图注的不是表，图注照常翻。"""
    from app.formats.pdf_layout import page_tables

    page = pymupdf.open().new_page(width=600, height=400)
    for y in (100, 114, 160):
        page.draw_line((350, y), (500, y), width=0.4)
    page.insert_text((355, 110), "Guardrails", fontsize=8)
    page.insert_text((450, 110), "MT Bench", fontsize=8)
    for row, (name, score) in enumerate((("No system prompt", "6.84 ± 0.07"), ("Llama 2 system prompt", "6.38 ± 0.07"),
                                         ("Mistral system prompt", "6.58 ± 0.05"))):
        page.insert_text((355, 126 + row * 12), name, fontsize=8)
        page.insert_text((450, 126 + row * 12), score, fontsize=8)
    [table] = page_tables(page)
    assert table.rect.y0 <= 100 and table.rect.y1 >= 160

    grammar = pymupdf.open().new_page(width=600, height=400)
    for y in (100, 150, 200):
        grammar.draw_line((100, y), (500, y), width=0.4)
    for row, (left, right) in enumerate((("S → NP VP", "I + want a morning flight"), ("NP → Pronoun", "I"),
                                         ("| Proper-Noun", "Los Angeles"))):
        grammar.insert_text((110, 115 + row * 12), left, fontsize=9)
        grammar.insert_text((300, 115 + row * 12), right, fontsize=9)
    grammar.insert_text((110, 170), "Figure 19.3", fontsize=9)
    grammar.insert_text((180, 170), "The grammar for L0, with example phrases for each rule.", fontsize=9)
    assert all(table.rect.y1 < 165 for table in page_tables(grammar))


def test_table_of_abutting_filled_cells_is_found():
    """每格一块底色、只有上下横线的表。一整块底色的文本框、白底的图、压着跨格图注的格子都不是表。"""
    from app.formats.pdf_layout import filled_tables, page_words

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=400)
    blue = (0.77, 0.86, 0.99)
    for row, y0 in enumerate((60, 72)):
        for col, (x0, x1) in enumerate(((240, 289), (289, 334), (334, 372), (372, 413))):
            page.draw_rect(pymupdf.Rect(x0, y0, x1, y0 + 12), color=None, fill=blue)
            page.insert_text((x0 + 3, y0 + 9), ("Unigram", "962", "170", "109")[col] if row else "N", fontsize=8)
    page.draw_rect(pymupdf.Rect(60, 150, 540, 200), color=None, fill=blue)
    page.insert_text((70, 170), "A shaded box of running text is a frame, not a table.", fontsize=10)
    for col in range(4):
        page.draw_rect(pymupdf.Rect(60 + col * 30, 260, 90 + col * 30, 290), color=None, fill=(1, 1, 1))
        page.draw_rect(pymupdf.Rect(60 + col * 30, 290, 90 + col * 30, 320), color=None, fill=(1, 1, 1))
    for col in range(4):
        page.draw_rect(pymupdf.Rect(300 + col * 20, 260, 320 + col * 20, 280), color=None, fill=blue)
        page.draw_rect(pymupdf.Rect(300 + col * 20, 280, 320 + col * 20, 300), color=None, fill=blue)
    page.insert_text((302, 272), "a", fontsize=8)
    page.insert_text((322, 272), "b", fontsize=8)
    page.insert_text((302, 292), "Figure 4: attention across the heads", fontsize=8)
    [table] = filled_tables(page, page.get_drawings(), page_words(page))
    assert (round(table.rect.x0), round(table.rect.y0), round(table.rect.x1), round(table.rect.y1)) == (240, 60, 413, 84)
    assert len(table.cells) == 8


def test_short_margin_terms_stay_original():
    """页边的术语留原文；成句的旁注照常翻译。"""
    from app.formats.pdf import _margin_term

    def note(text):
        return replace(_block(text, 0, pymupdf.Rect(80, 100, 128, 110), 7.2), side_note=True)

    assert _margin_term(note("perplexity"))
    assert _margin_term(note("log probability"))
    assert not _margin_term(note("This note explains why the counts are smoothed."))
    assert not _margin_term(_block("perplexity", 0, pymupdf.Rect(80, 100, 128, 110)))


def test_an_epigraph_in_another_language_stays_original():
    """英文书里的西班牙文题记分两列排，一项只有一两个词。合起来认出是外文，整组留原文。"""
    from app.formats.pdf import _foreign_ids

    body = [
        _block("Classification lies at the heart of language processing and intelligence. Recognizing a "
               "letter, a word, or a face, sorting mail, assigning grades to homeworks are all examples.",
               0, pymupdf.Rect(142, 342, 475, 402)),
        _block("Luckily, the classes we use for language processing are easier to define than those of "
               "Borges. In this chapter we introduce the logistic regression algorithm for classification.",
               0, pymupdf.Rect(142, 480, 475, 540)),
    ]
    epigraph = [_block("En sus remotas páginas está escrito que los animales se dividen en:", 0,
                       pymupdf.Rect(152, 220, 392, 230), 9)]
    rows = [("c. amaestrados", "j. innumerables"), ("d. lechones", "k. dibujados con un pincel finísimo"),
            ("e. sirenas", "l. etcétera"), ("f. fabulosos", "m. que acaban de romper el jarrón"),
            ("g. perros sueltos", "n. que de lejos parecen moscas")]
    for i, (left, right) in enumerate(rows):
        y0 = 231 + 11 * i
        epigraph.append(_block(left, 0, pymupdf.Rect(170, y0, 225, y0 + 10), 9))
        epigraph.append(_block(right, 0, pymupdf.Rect(291, y0, 410, y0 + 10), 9))
    found = _foreign_ids([*body, *epigraph])
    assert all(id(block) in found for block in epigraph)
    assert not any(id(block) in found for block in body)


def test_header_rows_above_a_three_line_table_join_it():
    """三线表的表头和表身隔着横线，按线只找得到表身。紧贴在上面、词都落在表的列里的行是表头；
    再往上的说明是连着排的句子，会跨过列线，不算。"""
    from app.formats.pdf_layout import RuledTable, _TextWord, _with_header_rows

    columns = [(115, 213), (213, 266), (266, 320), (320, 373), (373, 426), (426, 480)]
    body = RuledTable(pymupdf.Rect(115, 227, 480, 273), tuple(pymupdf.Rect(x0, 227, x1, 273) for x0, x1 in columns))

    def row(y, *items):
        return [_TextWord(x0, y, x1, y + 9, text) for text, x0, x1 in items]

    words = [
        *row(154, ("DeepSeek-R1,", 115, 170), ("we", 173, 184), ("report", 187, 214), ("results", 217, 248),
             ("under", 251, 276), ("two", 279, 294), ("configurations:", 297, 360)),
        *row(195, ("Discrimi.", 221, 259), ("Illegal", 281, 306), ("Harmful", 328, 365), ("Ethical", 382, 413),
             ("Overall", 437, 469)),
        *row(211, ("Ratio(%)", 117, 153), ("Unsafe", 215, 245), ("Rej.", 249, 264), ("Unsafe", 268, 298),
             ("Rej.", 302, 318), ("Unsafe", 322, 352), ("Rej.", 356, 371)),
    ]
    (table,) = _with_header_rows([body], words)
    assert table.rect.y0 < 195 and table.rect.y0 > 165
    assert len(table.cells) == len(body.cells) + 2


def test_a_combining_accent_is_drawn_over_its_letter(tmp_path):
    """没有合成字的 P̂：字体里没有组合字形，画成独立的 ˆ，中线对着 P，不落回 HTML 排版。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 52), "The estimated precision P is the fraction", fontsize=11)
    src = tmp_path / "hat.pdf"
    doc.save(src)
    rect = pymupdf.Rect(40, 42, 300, 56)
    block = TextBlock(page=0, rect=rect, line_rects=[rect], text="The estimated precision P is the fraction",
                      size=11, color="#000000", bold=False)
    out = _render_translated(src, [block], ["估计精度 {i}P̂{/i} 是比例"], "zh-CN")
    chars = [(chr(c[0]), c[3]) for trace in out[0].get_texttrace() for c in trace["chars"]]
    letter = next(box for char, box in chars if char == "P")
    hat = next(box for char, box in chars if char == "ˆ")
    assert abs((hat[0] + hat[2]) / 2 - (letter[0] + letter[2]) / 2) < 1.5
    assert hat[1] < letter[1] + 1


def test_image_box_is_clipped_to_what_shows():
    """图片只露出剪裁路径里的一块，框按看得见的部分算，压在被剪掉部分上的正文不算图里的字。"""
    import pymupdf

    from app.formats.pdf_layout import _image_infos, scan_page

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(100, 100, 400, 400), pixmap=pix)
    xref = page.get_contents()[0]
    # 只露出上面 60 点：PDF 坐标从页底往上量
    doc.update_stream(xref, b"q 100 640 300 60 re W n\n" + doc.xref_stream(xref) + b"\nQ")
    page.insert_text((110, 300), "Body text written below the part that shows.", fontsize=10)
    [info] = _image_infos(page)
    assert pymupdf.Rect(info["bbox"]) == pymupdf.Rect(100, 100, 400, 160)
    assert all(rect.y1 <= 160.5 for rect in scan_page(page).figures)


def test_two_list_items_with_the_same_opening_are_not_a_gloss():
    """两条列表项开头一样（Else search from right to left for ...），逐词左缘对得上，但同一处是同一个词。
    对译行写的是另一种语言，不会这样，两条照常送翻。"""
    from app.formats.pdf_layout import _TextWord, interlinear_glosses

    def row(y: float, *texts: str) -> list[_TextWord]:
        words, x = [], 177.0
        for text in texts:
            words.append(_TextWord(x, y, x + 5 * len(text), y + 10, text))
            x += 5 * len(text) + 3
        return words

    head = ("•", "Else", "search", "from", "right", "to", "left", "for", "the", "first", "child", "which", "is", "a")
    words = [
        *row(100, "•", "Else", "search", "from", "left", "to", "right", "for", "the", "first", "child", "which", "is", "an", "NP."),
        *row(111, *head, "$,", "ADJP,", "or", "PRN."),
        *row(122, *head, "CD."),
        *row(133, "•", "Else", "return", "the", "last", "word"),
    ]
    page = pymupdf.open().new_page(width=612, height=792)
    assert interlinear_glosses(page, words) == ()
