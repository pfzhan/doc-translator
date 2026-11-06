"""Writing translations onto the page (pdf_write, pdf_place, pdf_crop and the modules split from them)."""

import re
from pathlib import Path

import pymupdf
import pytest

from app.formats.pdf_blocks import TextBlock


def test_a_paragraph_starts_below_a_filled_paragraph_above(tmp_path):
    """The translation of the previous paragraph fills its box and its last line touches the start of the next
    paragraph. When the next paragraph does not fill its box, its start moves down to leave half a line between the
    two"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated
    from app.formats.pdf_write_geometry import _row_neighbor

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    for y in range(40, 148, 12):
        page.insert_text((40, y + 9), "x " * 60, fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)

    def block(y0: float) -> TextBlock:
        rows = [pymupdf.Rect(40, y, 360, y + 12) for y in (y0, y0 + 12, y0 + 24)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(40, y0, 360, y0 + 36),
            line_rects=rows,
            text="x " * 60 + "end.",
            size=10,
            color="#000000",
            bold=False,
        )

    blocks = [block(40), block(76), block(112)]
    full = "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 4
    out = _render_translated(src_path, blocks, [full, "第二段只有一行。", full[:60]], "zh-CN")
    # The source blocks have no spans and cannot be erased, so only the Chinese lines are checked
    rows = sorted(
        (line["spans"][0]["origin"][1], line["spans"][0]["size"], text)
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        for text in ["".join(span["text"] for span in line["spans"])]
        if re.search(r"[一-鿿]", text)
    )
    assert len(rows) == 6 and all(abs(size - 8.8) < 0.1 for _, size, _ in rows)
    baselines = [baseline for baseline, _, _ in rows]
    pitch = baselines[1] - baselines[0]
    assert rows[3][2].startswith("第二段")
    assert baselines[3] - baselines[2] >= pitch * 1.4
    # The second paragraph is short and already leaves space before the third, which is written at its original
    # position
    assert baselines[4] < 112 + 12

    # One-line items spaced as tightly as the original: moving an item down would hit the one below it, so nothing
    # moves
    for y in (200, 212, 224):
        page.insert_text((40, y + 9), "x " * 10, fontsize=10)
    src.save(tmp_path / "list.pdf")
    # The middle item has a subscript, so its box is taller than the line pitch and reaches into the next item
    rows = [
        TextBlock(
            page=0,
            rect=pymupdf.Rect(40, y, 160, y + h),
            line_rects=[pymupdf.Rect(40, y, 160, y + h)],
            text="x " * 10 + "end.",
            size=10,
            color="#000000",
            bold=False,
        )
        for y, h in ((200, 12), (212, 16), (224, 12))
    ]
    out = _render_translated(
        tmp_path / "list.pdf", rows, ["施事/主语, 主题/宾语", "工具/主语, 主题/宾语", "主题/主语"], "zh-CN"
    )
    listed = sorted(
        line["spans"][0]["origin"][1]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
    )
    assert len(listed) == 3
    assert all(lower - upper > 11 for upper, lower in zip(listed, listed[1:]))

    # Vertically adjacent table cells have other cells beside them, each aligned with its original row. A side note
    # in the page margin does not count
    cell = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 40, 90, 52),
        line_rects=[pymupdf.Rect(40, 40, 90, 52)],
        text="rise",
        size=10,
        color="#000000",
        bold=False,
    )
    other = TextBlock(
        page=0,
        rect=pymupdf.Rect(110, 40, 160, 52),
        line_rects=[pymupdf.Rect(110, 40, 160, 52)],
        text="fall",
        size=10,
        color="#000000",
        bold=False,
    )
    note = TextBlock(
        page=0,
        rect=pymupdf.Rect(0, 40, 30, 52),
        line_rects=[pymupdf.Rect(0, 40, 30, 52)],
        text="term",
        size=7,
        color="#000000",
        bold=False,
        side_note=True,
    )
    assert _row_neighbor(cell, [cell, other], None)
    assert not _row_neighbor(blocks[0], [*blocks, note], None)


def test_a_paragraph_borrows_the_spare_height_of_the_next_one(tmp_path):
    """The translation is one line longer than the original: the next paragraph is short and has spare room, so this
    paragraph extends into it at normal line spacing and the next paragraph moves down half a line. Only when the
    next paragraph fills its own box and has nothing to lend does this paragraph tighten its line spacing. A next
    block that is not at body size (a label in a figure) does not move either"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    for y in range(40, 124, 12):
        page.insert_text((40, y + 9), "x " * 60, fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)

    def block(y0: float, count: int, size: float = 10) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(40, y0, 360, y0 + 12 * count),
            line_rects=rows,
            text="x " * 60 + "end.",
            size=size,
            color="#000000",
            bold=False,
            body_size=10,
        )

    def baselines(doc) -> list[float]:
        return sorted(
            line["spans"][0]["origin"][1]
            for item in doc[0].get_text("dict")["blocks"]
            if item.get("type") == 0
            for line in item["lines"]
            if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
        )

    long = "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 5
    blocks = [block(40, 3), block(76, 4)]
    rows = baselines(_render_translated(src_path, blocks, [long, "第二段原文四行，译文一行。"], "zh-CN"))
    assert len(rows) == 5
    pitches = [lower - upper for upper, lower in zip(rows[:4], rows[1:4])]
    assert all(abs(pitch - 8.8 * 1.45) < 0.2 for pitch in pitches)
    assert rows[4] - rows[3] >= 8.8 * 1.45 * 1.4

    rows = baselines(_render_translated(src_path, blocks, [long, long[:130]], "zh-CN"))
    assert rows[1] - rows[0] < 8.8 * 1.4
    assert all(lower - upper > 8.8 for upper, lower in zip(rows, rows[1:]))

    rows = baselines(_render_translated(src_path, [blocks[0], block(76, 4, 8)], [long, "标签"], "zh-CN"))
    assert rows[1] - rows[0] < 8.8 * 1.4


def test_a_chain_of_paragraphs_moves_down_together(tmp_path):
    """The next paragraph fills its own box and the one after it has spare room: both move down together, and the two
    upper paragraphs are written at normal line spacing"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    for y in range(40, 172, 12):
        page.insert_text((40, y + 9), "x " * 60, fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)

    def block(y0: float, count: int) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(40, y0, 360, y0 + 12 * count),
            line_rects=rows,
            text="x " * 60 + "end.",
            size=10,
            color="#000000",
            bold=False,
            body_size=10,
        )

    def baselines(doc) -> list[float]:
        return sorted(
            line["spans"][0]["origin"][1]
            for item in doc[0].get_text("dict")["blocks"]
            if item.get("type") == 0
            for line in item["lines"]
            if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
        )

    long = "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 5
    full = "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 3
    blocks = [block(40, 3), block(76, 3), block(112, 5)]
    rows = baselines(_render_translated(src_path, blocks, [long, full, "第三段原文五行，译文一行。"], "zh-CN"))
    assert len(rows) == 8
    pitches = [lower - upper for upper, lower in zip(rows[:7], rows[1:7]) if lower - upper < 8.8 * 1.6]
    assert len(pitches) == 5 and all(abs(pitch - 8.8 * 1.45) < 0.2 for pitch in pitches)
    assert rows[7] - rows[6] >= 8.8 * 1.45 * 1.4

    # One-line list items: the first item gains a line and the moved start of the second falls below the top of the
    # third; the third still moves along so the items do not overlap
    items = [block(40, 1), block(52, 1), block(64, 1), block(76, 8)]
    rows = baselines(
        _render_translated(
            src_path,
            items,
            [full[:60], "第二项。", full[:60], "第四段原文八行，译文一行。"],
            "zh-CN",
        )
    )
    assert len(rows) == 6
    assert all(lower - upper > 8.8 * 1.4 for upper, lower in zip(rows, rows[1:]))


def test_a_paragraph_moves_up_into_the_space_left_above(tmp_path):
    """Nothing below to borrow from, and the translation of the previous paragraph is short and leaves space: this
    paragraph moves up, is written at normal line spacing and stays half a line below the previous one"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    for y in range(40, 124, 12):
        page.insert_text((40, y + 9), "x " * 40, fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)

    def block(y0: float, count: int) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(40, y0, 360, y0 + 12 * count),
            line_rects=rows,
            text="x " * 40 + "end.",
            size=10,
            color="#000000",
            bold=False,
            body_size=10,
        )

    def baselines(doc) -> list[float]:
        return sorted(
            line["spans"][0]["origin"][1]
            for item in doc[0].get_text("dict")["blocks"]
            if item.get("type") == 0
            for line in item["lines"]
            if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
        )

    long = "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 5
    rows = baselines(
        _render_translated(src_path, [block(40, 4), block(88, 3)], ["第一段原文四行，译文一行。", long], "zh-CN")
    )
    assert len(rows) == 5
    pitches = [lower - upper for upper, lower in zip(rows[1:], rows[2:])]
    assert all(abs(pitch - 8.8 * 1.45) < 0.2 for pitch in pitches)
    assert rows[1] - rows[0] >= 8.8 * 1.45 * 1.4
    assert rows[1] < 88


def test_the_first_paragraph_under_a_heading_moves_up(tmp_path):
    """The first paragraph under a heading does not fit and there is nothing below to borrow from: it moves up, at
    most to half a line below the last line of the heading"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    page.insert_text((40, 52), "Heading", fontsize=14)
    for y in range(88, 124, 12):
        page.insert_text((40, y + 9), "x " * 40, fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    heading = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 40, 120, 56),
        line_rects=[pymupdf.Rect(40, 40, 120, 56)],
        text="Heading",
        size=14,
        color="#000000",
        bold=True,
        body_size=10,
    )
    rows = [pymupdf.Rect(40, 88 + 12 * index, 360, 100 + 12 * index) for index in range(3)]
    body = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 88, 360, 124),
        line_rects=rows,
        text="x " * 40 + "end.",
        size=10,
        color="#000000",
        bold=False,
        body_size=10,
    )
    long = "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 5
    out = _render_translated(src_path, [heading, body], ["标题", long], "zh-CN")
    lines = sorted(
        (line["spans"][0]["origin"][1], line["spans"][0]["size"])
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
    )
    rows = [y for y, size in lines if abs(size - 8.8) < 0.1]
    assert len(rows) == 4
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows, rows[1:]))
    assert rows[0] < 85


def test_a_paragraph_does_not_move_up_past_a_heading_that_touches_it(tmp_path):
    """The line box of the heading overlaps the top edge of the paragraph below a little. When moving up, the heading
    still counts as content above, so the paragraph does not move onto it"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    page.insert_text((40, 29), "x " * 40, fontsize=10)
    page.insert_text((40, 70), "Heading", fontsize=14)
    for y in range(72, 108, 12):
        page.insert_text((40, y + 9), "x " * 40, fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    above = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 20, 360, 32),
        line_rects=[pymupdf.Rect(40, 20, 360, 32)],
        text="x " * 40 + "end.",
        size=10,
        color="#000000",
        bold=False,
        body_size=10,
    )
    heading = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 58, 120, 75),
        line_rects=[pymupdf.Rect(40, 58, 120, 75)],
        text="Heading",
        size=14,
        color="#000000",
        bold=True,
        body_size=10,
    )
    rows = [pymupdf.Rect(40, 72 + 12 * index, 360, 84 + 12 * index) for index in range(3)]
    body = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 72, 360, 108),
        line_rects=rows,
        text="x " * 40 + "end.",
        size=10,
        color="#000000",
        bold=False,
        body_size=10,
    )
    long = "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 5
    out = _render_translated(src_path, [above, heading, body], ["短。", "标题", long], "zh-CN")
    tops = [
        line["bbox"][1]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if "这一卷" in "".join(span["text"] for span in line["spans"])
    ]
    assert tops and min(tops) > 72


def test_an_empty_tail_above_does_not_stop_a_paragraph_moving_up(tmp_path):
    """The translation of the tail block at the column top was written entirely into the host, so nothing is written
    when its turn comes. It does not count as an unwritten block, and the paragraph below still moves up"""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)

    def block(x0: float, y0: float, count: int, size: float = 10, text: str = "x " * 15 + "end.") -> TextBlock:
        rows = [pymupdf.Rect(x0, y0 + 12 * index, x0 + 160, y0 + 12 * index + 12) for index in range(count)]
        for row in rows:
            page.insert_text((row.x0, row.y0 + 9), "x " * 15, fontsize=size)
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(x0, y0, x0 + 160, y0 + 12 * count),
            line_rects=rows,
            text=text,
            size=size,
            color="#000000",
            bold=False,
            body_size=10,
        )

    host = block(20, 200, 5, text="x " * 30 + "and")
    tail = block(220, 20, 1)
    short = block(220, 40, 3)
    body = block(220, 80, 3)
    label = block(220, 120, 2, size=8)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    long = "这一卷讲的是这个激动人心的构想如何实现。" * 3
    out = _render_translated(
        src_path,
        [host, tail, short, body, label],
        ["宿主短。", SPILL_TAIL, "短。", long, "标注"],
        "zh-CN",
        continuations={id(host): tail},
    )
    rows = sorted(
        line["spans"][0]["origin"][1]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if line["bbox"][0] > 200
        and "标注" not in line["spans"][0]["text"]
        and abs(line["spans"][0]["size"] - 8.8) < 0.1
    )
    assert len(rows) == 5
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows[1:], rows[2:]))


_LONG_ZH = "在实验研究中我们首先比较旧优化器与新优化器的计划质量，并说明新优化器在大规模数据上的收益。"


def test_a_paragraph_moves_up_when_the_next_box_overlaps_its_bottom(tmp_path):
    """The box of the next paragraph overlaps the bottom of this one. It moves along and does not count as fixed
    content inside this box, so this paragraph still moves up"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)

    def block(y0: float, count: int, size: float = 10, y1: float | None = None) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        for row in rows:
            page.insert_text((40, row.y0 + 9), "x " * 40, fontsize=size)
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(40, y0, 360, y1 or y0 + 12 * count),
            line_rects=rows,
            text="x " * 40 + "end.",
            size=size,
            color="#000000",
            bold=False,
            body_size=10,
        )

    short, body, below, label = block(20, 1), block(64, 3, y1=106), block(101, 3), block(139, 1, size=8)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    out = _render_translated(src_path, [short, body, below, label], ["短。", _LONG_ZH * 3, _LONG_ZH, "标注"], "zh-CN")
    rows = sorted(
        line["spans"][0]["origin"][1]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if abs(line["spans"][0]["size"] - 8.8) < 0.1 and "短" not in line["spans"][0]["text"]
    )
    assert len(rows) == 6
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows[:3], rows[1:4]))
    assert rows[4] - rows[3] > 8.8


def test_a_cross_page_paragraph_keeps_its_lines_when_lifting_more_does_not_help(tmp_path):
    """A paragraph carried over to the next page does not fit at the page bottom at normal line spacing, and its tail
    block has other blocks beside it and cannot take more. Moving up to hold extra lines would still be cramped,
    so the paragraph keeps its original number of lines and moves up to where those lines fit at normal line
    spacing"""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()

    def block(page: pymupdf.Page, y0: float, count: int, x1: float = 360) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, x1, y0 + 12 * index + 12) for index in range(count)]
        for row in rows:
            page.insert_text((40, row.y0 + 9), "x " * 40, fontsize=10)
        return TextBlock(
            page=page.number,
            rect=pymupdf.Rect(40, y0, x1, y0 + 12 * count),
            line_rects=rows,
            text="x " * 40 + "end.",
            size=10,
            color="#000000",
            bold=False,
            body_size=10,
        )

    first = doc.new_page(width=400, height=300)
    above, host = block(first, 40, 3), block(first, 124, 3)
    second = doc.new_page(width=400, height=300)
    tail = block(second, 40, 1, x1=200)
    second.insert_text((250, 49), "side " * 5, fontsize=10)
    side = TextBlock(
        page=1,
        rect=pymupdf.Rect(250, 40, 360, 52),
        line_rects=[pymupdf.Rect(250, 40, 360, 52)],
        text="side " * 5,
        size=10,
        color="#000000",
        bold=False,
        body_size=10,
    )
    doc.save(src)
    doc.close()
    out = _render_translated(
        src,
        [above, host, tail, side],
        ["短。", _LONG_ZH * 6, SPILL_TAIL, "旁边"],
        "zh-CN",
        continuations={id(host): tail},
    )
    rows = [
        line["spans"][0]["origin"][1]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if "短" not in line["spans"][0]["text"]
    ]
    assert len(rows) >= 2
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows, rows[1:]))


def test_a_cross_page_paragraph_below_moves_down_to_make_room(tmp_path):
    """A paragraph does not fit, and the paragraph below is carried over to the next page with room left at its tail
    block: the paragraph below moves down, fewer of its lines stay on this page, and the rest go to the tail block"""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()

    def block(page: pymupdf.Page, y0: float, count: int) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        for row in rows:
            page.insert_text((40, row.y0 + 9), "x " * 40, fontsize=10)
        return TextBlock(
            page=page.number,
            rect=pymupdf.Rect(40, y0, 360, y0 + 12 * count),
            line_rects=rows,
            text="x " * 40 + "end.",
            size=10,
            color="#000000",
            bold=False,
            body_size=10,
        )

    first = doc.new_page(width=400, height=300)
    body, host = block(first, 104, 3), block(first, 140, 3)
    second = doc.new_page(width=400, height=300)
    tail, after = block(second, 40, 2), block(second, 66, 6)
    doc.save(src)
    doc.close()
    out = _render_translated(
        src,
        [body, host, tail, after],
        [_LONG_ZH * 3, _LONG_ZH * 2, SPILL_TAIL, "短。"],
        "zh-CN",
        continuations={id(host): tail},
    )
    rows = [
        line["spans"][0]["origin"][1]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
    ]
    assert len(rows) == 5
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows[:4], rows[1:4]))
    assert rows[4] - rows[3] > 8.8 * 1.45


def test_the_last_paragraph_continues_on_the_next_page(tmp_path):
    """The last paragraph at the page bottom does not fit at normal line spacing and there is no room above or below:
    its last lines continue in front of the first paragraph on the next page, which moves down to make room"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()

    def block(page: pymupdf.Page, y0: float, count: int) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        for row in rows:
            page.insert_text((40, row.y0 + 9), "x " * 40, fontsize=10)
        return TextBlock(
            page=page.number,
            rect=pymupdf.Rect(40, y0, 360, y0 + 12 * count),
            line_rects=rows,
            text="x " * 40 + "end.",
            size=10,
            color="#000000",
            bold=False,
            body_size=10,
        )

    first = doc.new_page(width=400, height=300)
    above, last = block(first, 40, 5), block(first, 104, 3)
    second = doc.new_page(width=400, height=300)
    head = block(second, 40, 3)
    doc.save(src)
    doc.close()
    out = _render_translated(src, [above, last, head], [_LONG_ZH * 3, _LONG_ZH * 3, "短。"], "zh-CN")

    def rows(page: pymupdf.Page) -> list[tuple[float, str]]:
        return [
            (line["spans"][0]["origin"][1], "".join(span["text"] for span in line["spans"]))
            for item in page.get_text("dict")["blocks"]
            if item.get("type") == 0
            for line in item["lines"]
        ]

    first_rows, second_rows = rows(out[0]), rows(out[1])
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for (upper, _), (lower, _) in zip(first_rows[4:], first_rows[5:]))
    assert len(first_rows) == 7 and len(second_rows) == 2
    assert second_rows[1][1].startswith("短") and second_rows[1][0] - second_rows[0][0] > 8.8 * 1.45


def _zh_rows(page: pymupdf.Page, size: float = 8.8) -> list[float]:
    """Baselines of the translated lines, top to bottom"""
    return sorted(
        line["spans"][0]["origin"][1]
        for item in page.get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if abs(line["spans"][0]["size"] - size) < 0.1
    )


def _x_block(page: pymupdf.Page, y0: float, count: int, size: float = 10, row: float = 12) -> TextBlock:
    rows = [pymupdf.Rect(40, y0 + row * index, 360, y0 + row * index + row) for index in range(count)]
    for rect in rows:
        page.insert_text((40, rect.y0 + 9), "x " * 40, fontsize=size)
    return TextBlock(
        page=page.number,
        rect=pymupdf.Rect(40, y0, 360, y0 + row * count),
        line_rects=rows,
        text="x " * 40 + "end.",
        size=size,
        color="#000000",
        bold=False,
        body_size=10,
    )


def test_the_blank_under_the_last_line_may_pass_the_box_bottom(tmp_path):
    """Measured by full line pitch the text falls just short of fitting, but the glyphs of the last line are inside
    the box and half a line pitch away from the label below: it is written at normal line spacing without
    tightening"""
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    body, label = _x_block(page, 40, 3, row=11.5), _x_block(page, 82, 1, size=8)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    out = _render_translated(src_path, [body, label], ["测" * 100, "标注"], "zh-CN")
    rows = _zh_rows(out[0])
    assert len(rows) == 3
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows, rows[1:]))


def test_a_single_line_still_tightens_inside_its_box():
    """A single line has no line spacing to keep: it is measured against the box bottom and fitted into the box with
    tighter spacing, so the bottom of the last line box is higher and the paragraph below can move up"""
    from app.formats.pdf_place import _fit_typeset

    page = pymupdf.open().new_page(width=400, height=300)
    em = 8.8
    for text, rows, gap in (("测" * 10, 1, 1.2), ("测" * 50, 2, 1.45)):
        height = rows * em * 1.45 + em * 0.3 - 2
        rect = pymupdf.Rect(40, 40, 360, 40 + height)
        lines, size, used, _, _ = _fit_typeset(text, [], rect, em, 10, True, page, [], None, room=rect.y1 + 1)
        assert (len(lines), size, used) == (rows, em, gap)


def test_a_display_formula_left_at_the_page_bottom_is_part_of_the_body(tmp_path):
    """A display formula left in place at the page bottom is body content too: the paragraph above extends down to
    just above the formula and does not stop at its own original bottom edge"""
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    body = _x_block(page, 40, 3)
    page.insert_text((150, 125), "E = mc2", fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    out = _render_translated(src_path, [body], ["测" * 140], "zh-CN")
    rows = _zh_rows(out[0])
    assert len(rows) == 4
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows, rows[1:]))


def test_a_page_number_left_below_the_text_is_not_part_of_the_body(tmp_path):
    """A page number dropped during extraction stays in place and is not body content: the paragraph above does not
    extend down to it"""
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    body = _x_block(page, 40, 3)
    page.insert_text((195, 125), "12", fontsize=10)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    out = _render_translated(src_path, [body], ["测" * 140], "zh-CN")
    bottoms = [
        line["bbox"][3]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if "测" in line["spans"][0]["text"]
    ]
    assert bottoms and max(bottoms) < 85


def test_a_flush_right_attribution_widens_to_the_left(tmp_path):
    """A right-aligned byline whose translation is slightly wider than the original widens to the left as one line,
    still ending at the original right edge, without wrapping or a smaller font size"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write import _render_translated

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    right = 475.0
    page.insert_text(
        (right - pymupdf.get_text_length("Samuel Johnson", "tiro", 9), 240),
        "Samuel Johnson",
        fontsize=9,
        fontname="tiro",
    )
    body_line = "x" * 55
    for index in range(3):
        page.insert_text(
            (right - pymupdf.get_text_length(body_line, "cour", 10), 300 + index * 12),
            body_line,
            fontsize=10,
            fontname="cour",
        )
    path = tmp_path / "src.pdf"
    doc.save(path)
    page = pymupdf.open(path)[0]
    row = page.search_for("Samuel Johnson")[0]
    author = TextBlock(page=0, rect=row, line_rects=[row], text="Samuel Johnson", size=9.0, color="#000000", bold=False)
    body = TextBlock(
        page=0,
        rect=pymupdf.Rect(page.search_for(body_line)[0]) | page.search_for(body_line)[-1],
        line_rects=page.search_for(body_line),
        text=body_line * 3,
        size=10.0,
        color="#000000",
        bold=False,
    )
    frame = TextFrame(body.rect.x0, body.rect.x1, 100, 720, 10)
    geo = PageGeometry.from_rects(612, [author.rect, body.rect], height=792, frame=frame)
    out = _render_translated(path, [author, body], ["塞缪尔·约翰逊，《闲人》", "正文" * 40], "zh-CN", geometries=[geo])
    lines = [
        line
        for block in out[0].get_text("dict", clip=pymupdf.Rect(200, 225, 500, 255))["blocks"]
        for line in block.get("lines", [])
    ]
    assert len(lines) == 1
    assert abs(lines[0]["bbox"][2] - author.rect.x1) < 0.5
    assert abs(lines[0]["spans"][0]["size"] - 9.0 * 0.88) < 0.05


def test_a_last_line_of_one_character_is_pulled_back_by_tightening_the_letter_spacing(tmp_path):
    """The last line holds a single character and there is no room above or below: the character spacing tightens a
    little to pull it back into the previous line, the text is written at normal line spacing, and no line goes
    past the right edge"""
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=300)
    body, label = _x_block(page, 40, 3, row=11.5), _x_block(page, 82, 1, size=8)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)
    out = _render_translated(src_path, [body, label], ["测" * 109, "标注"], "zh-CN")
    rows = _zh_rows(out[0])
    assert len(rows) == 3
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows, rows[1:]))
    right = max(
        line["bbox"][2]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if abs(line["spans"][0]["size"] - 8.8) < 0.1
    )
    assert right < 362.5


def test_a_cross_page_paragraph_moves_up_to_keep_its_line_spacing(tmp_path):
    """A paragraph carried over to the next page does not fit on this page at normal line spacing and the next page
    cannot take more: the translation of the paragraph above is short and leaves space, so the whole paragraph
    moves up"""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    for bottom in (256, 64):
        page = doc.new_page(width=400, height=300)
        for y in range(40, bottom, 12):
            page.insert_text((40, y + 9), "x " * 40, fontsize=10)
    doc.save(src)
    doc.close()

    def block(page: int, y0: float, count: int) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        return TextBlock(
            page=page,
            rect=pymupdf.Rect(40, y0, 360, y0 + 12 * count),
            line_rects=rows,
            text="x " * 40 + "end.",
            size=10,
            color="#000000",
            bold=False,
            body_size=10,
        )

    above, host, tail = block(0, 40, 15), block(0, 220, 3), block(1, 40, 2)
    paragraph = "在实验研究中我们首先比较旧优化器与新优化器的计划质量，并说明新优化器在大规模数据上的收益。" * 4
    out = _render_translated(
        src,
        [above, host, tail],
        ["上面一段原文十五行，译文一行。", paragraph, SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    rows = sorted(
        line["spans"][0]["origin"][1]
        for item in out[0].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
    )[1:]
    assert len(rows) >= 3
    assert all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(rows, rows[1:]))
    assert rows[0] < 220


def test_pdf_cjk_lines_do_not_overlap(tmp_path):
    """CJK translations are shrunk and moved down so the two lines do not overlap; non-CJK text keeps the original
    font size."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=200)
    page.insert_text((40, 40), "AAAA BBBB CCCC", fontsize=12)
    page.insert_text((40, 54), "DDDD EEEE FFFF", fontsize=12)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)

    def block(y0: float, text: str) -> TextBlock:
        rect = pymupdf.Rect(40, y0, 200, y0 + 14)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=12, color="#000000", bold=False)

    blocks = [block(28, "AAAA BBBB CCCC"), block(42, "DDDD EEEE FFFF")]

    def line_boxes(lang: str, translations: list[str]) -> list[tuple[pymupdf.Rect, float]]:
        out = _render_translated(src_path, blocks, translations, lang)
        boxes = []
        for b in out[0].get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                boxes.append((pymupdf.Rect(ln["bbox"]), ln["spans"][0]["size"]))
        return boxes

    cjk = line_boxes("zh-CN", ["第一行中文译文加长", "第二行中文译文加长"])
    en = line_boxes("en", ["First line translation", "Second line translation"])
    assert len(cjk) == 2 and len(en) == 2
    assert cjk[0][0].y1 <= cjk[1][0].y0 + 0.05
    assert abs(cjk[0][1] - 12 * 0.88) < 0.2
    assert abs(en[0][1] - 12) < 0.2
    for lang in ("yue", "wyw"):
        sized = line_boxes(lang, ["第一行", "第二行"])
        assert sized and abs(sized[0][1] - 12 * 0.88) < 0.2
    # After moving down, the glyphs still fall inside the box extended for overflow (1pt of glyph overflow allowed)
    expanded_y0 = 28 + 12 * 0.1
    expanded_y1 = 42 + 12 * 0.3
    assert cjk[0][0].y0 >= expanded_y0 - 1
    assert cjk[0][0].y1 <= expanded_y1 + 1


def test_typesetting_ladder_expands_before_shrinking():
    """When the text does not fit, the box first widens to the right (avoiding the text block on the right): the font
    size after widening is larger than after shrinking directly, and the right block is not covered."""
    from app.formats.pdf_extract import extract_blocks
    from app.formats.pdf_write import _render_translated

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=100)
    page.insert_textbox(pymupdf.Rect(20, 17, 100, 40), "Short left", fontsize=12)
    page.insert_textbox(pymupdf.Rect(300, 17, 390, 40), "Right block", fontsize=12)
    src = doc.tobytes()

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(src)
        src_path = Path(f.name)
    blocks = extract_blocks(pymupdf.open(src_path))
    left = next(b for b in blocks if b.text.startswith("Short"))
    right = next(b for b in blocks if b.text.startswith("Right"))
    translations = ["这是一段比原文长很多的译文，原来肯定放不下" if b is left else b.text for b in blocks]
    out = _render_translated(src_path, blocks, translations, "zh-CN")
    spans = [ln for blk in out[0].get_text("dict")["blocks"] for ln in blk["lines"]]
    zh = [ln for ln in spans if "译文" in "".join(s["text"] for s in ln["spans"])]
    assert zh
    zh_rect = zh[0]["bbox"]
    # The translation goes past the right edge of the original block (it widened to the right) without covering
    # the right block
    assert zh_rect[2] > left.rect.x1 + 1 and zh_rect[2] <= right.rect.x0
    # The original text of the right block is still there
    assert "Right block" in out[0].get_text()


def test_marker_png_unique_per_formula():
    """Marker pixels encode a global index: markers of the same color would be deduplicated by MuPDF into one xref and
    every formula would land in the wrong slot."""
    from app.formats.pdf_crop import _marker_png

    def decode(png):
        pix = pymupdf.Pixmap(png)
        p = pix.pixel(0, 0)
        assert p[2] == 200  # Marker signature
        return p[0] + (p[1] << 8) - 1

    assert decode(_marker_png(0)) == 0
    assert decode(_marker_png(255)) == 255
    assert decode(_marker_png(300)) == 300  # Values beyond one byte are still distinguished
    assert len({_marker_png(i) for i in range(300)}) == 300


def test_rendered_formulas_placed_once_in_own_slots(tmp_path):
    """End-to-end vector passthrough: the formulas of two blocks each land in their own slot, are drawn once, and sit
    between the text before and after them in the content stream, so reading in content stream order (drag
    selection, copying) finds each formula in its place.

    Regression: marker images named by per-block index were deduplicated against each other, so formula #1 of
    every block piled up in the slot of the first block; all formulas were drawn in the last content stream of the
    page, so selecting body text by dragging missed the formulas.
    """
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_crop import _marker_png
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((50, 50), "alpha BETA gamma", fontsize=10)
    page.insert_text((50, 100), "delta EPSILON zeta", fontsize=10)
    doc.save(src)
    doc.close()

    d = pymupdf.open(src)
    r1 = d[0].search_for("BETA")[0]
    r2 = d[0].search_for("EPSILON")[0]

    def record(name, mid, rect):
        return {
            "bbox": rect,
            "name": name,
            "mid": mid,
            "w": round(rect.width, 1),
            "h": round(rect.height, 1),
            "d": 2.0,
            "text": "x",
            "has_img": True,
            "png": d[0].get_pixmap(dpi=150, clip=rect).tobytes("png"),
            "spans": [{"bbox": tuple(rect), "text": "x", "size": 10, "origin": (rect.x0, rect.y1 - 2)}],
        }

    f1, f2 = record("a_1.png", 0, r1), record("b_1.png", 1, r2)
    d.close()
    archive = pymupdf.Archive()
    archive.add(_marker_png(0), "ph_a_1.png")
    archive.add(_marker_png(1), "ph_b_1.png")

    def blk(text, y):
        rect = pymupdf.Rect(40, y, 200, y + 15)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000", bold=False)

    b1, b2 = blk("alpha BETA gamma", 40), blk("delta EPSILON zeta", 90)
    t1, t2 = "译文 \x01i\x021\x01/i\x02 甲", "译文 \x01i\x021\x01/i\x02 乙"
    out = _render_translated(src, [b1, b2], [t1, t2], "zh-CN", {id(b1): [f1], id(b2): [f2]}, archive)
    page = out[0]
    streams = [out.xref_stream(x) or b"" for x in page.get_contents()]
    assert sum(s.count(b"/FmF_a_1_png Do") for s in streams) == 1
    assert sum(s.count(b"/FmF_b_1_png Do") for s in streams) == 1
    words = [w[4] for w in page.get_text("words", sort=False)]
    assert [w for w in words if w in ("BETA", "EPSILON", "甲", "乙")] == ["BETA", "甲", "EPSILON", "乙"]

    # When translating into English each text piece is one HTML box, and the formula stream goes after the stream of
    # the preceding text piece
    out = _render_translated(
        src,
        [b1, b2],
        ["start \x01i\x021\x01/i\x02 one", "second \x01i\x021\x01/i\x02 two"],
        "en",
        {id(b1): [f1], id(b2): [f2]},
        archive,
    )
    words = [w[4] for w in out[0].get_text("words", sort=False)]
    assert [w for w in words if w in ("start", "BETA", "one", "second", "EPSILON", "two")] == [
        "start",
        "BETA",
        "one",
        "second",
        "EPSILON",
        "two",
    ]
    # After the original is redacted, passthrough restores the formula glyphs: each appears once, in different slots
    assert len(page.search_for("BETA")) == 1
    assert len(page.search_for("EPSILON")) == 1
    hit1, hit2 = page.search_for("BETA")[0], page.search_for("EPSILON")[0]
    # Each lands in its own slot (near the block rect): a wrong translation in a scaled cm would make the formulas
    # drift down as a whole
    assert pymupdf.Rect(30, 30, 210, 75).contains(hit1)
    assert pymupdf.Rect(30, 80, 210, 125).contains(hit2)
    # Passthrough carries only the glyphs of the formula itself. With the whole original page stacked under each
    # formula (even outside the page), drag selection in a browser selects a large area
    words = [w[4] for w in page.get_text("words", flags=pymupdf.TEXTFLAGS_WORDS & ~pymupdf.TEXT_MEDIABOX_CLIP)]
    assert not set(words) & {"alpha", "gamma", "delta", "zeta"}
    assert words.count("BETA") == 1 and words.count("EPSILON") == 1


def test_formulas_on_a_page_filter_their_own_lines():
    """The formulas of a page are halved top and bottom repeatedly, each part keeping only its own lines, and each
    formula is filtered on the smallest part; the glyphs kept are the same as when filtering on the whole page. A
    formula that was not registered still finds the part that covers it."""
    from app.formats.pdf_place_formulas import _FormulaSources, _keep_marks, _pdf_rect
    from app.formats.pdf_write_geometry import _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    for row in range(16):
        page.insert_text((40, 30 + 22 * row), f"line {row} has v{row} inside", fontsize=10)
    whole = _page_form(doc, page)
    formulas = []
    for row in range(16):
        box = page.search_for(f"v{row} ")[0] + (0, 0, -2, 0)
        formulas.append({"bbox": box, "spans": [{"bbox": tuple(box)}]})
    sources = _FormulaSources(doc, page, whole, formulas[:12])

    def shown(form):
        view = doc.new_page(width=400, height=400)
        draw = doc.get_new_xref()
        doc.update_object(draw, "<<>>")
        doc.update_stream(draw, b"q /Fm Do Q")
        doc.xref_set_key(view.xref, "Contents", f"{draw} 0 R")
        doc.xref_set_key(view.xref, "Resources", f"<< /XObject << /Fm {form} 0 R >> >>")
        return sorted(w[4] for w in doc[-1].get_text("words"))

    lines = shown(sources.source(formulas[3]))
    assert "v3" in lines and len(lines) <= 4 * 4 and "v12" not in lines
    for row in (0, 5, 11, 14):
        keep = [_pdf_rect(doc[0], formulas[row]["bbox"] + (-1, -1, 1, 1))]
        source = sources.source(formulas[row])
        assert shown(_keep_marks(doc, source, keep)) == shown(_keep_marks(doc, whole, keep)) == [f"v{row}"]
    assert sources.source(formulas[14]) == whole


def test_glyph_places_match_when_a_column_jitters_in_the_last_digits():
    """The content stream rewritten after filtering has numbers with fewer digits, so the x of the first glyph on
    lines in the same column differs by ten-thousandths and the order after sorting by position changes. Each
    glyph is still in place, so the position check passes; a glyph that really moved fails it."""
    from app.formats.pdf_place_formulas import _same_places

    before = [(72.0001, 500.9), (71.9999, 169.0), (72.0, 181.0), (80.0, 169.0)]
    after = [(71.9999, 500.9), (72.0001, 169.0), (71.9998, 181.0), (80.05, 169.0)]
    assert sorted(before) != sorted(after)
    assert _same_places(before, after)
    assert not _same_places(before, after[:3] + [(80.3, 169.0)])
    assert not _same_places(before, after[:3])


def test_a_formula_bitmap_is_rendered_only_when_used(monkeypatch):
    """Cropping a formula image only records where to crop; it is rendered only when vector placement fails and a
    bitmap is needed, and only once."""
    from app.formats.pdf_blocks import _FormulaImage
    from app.formats.pdf_place_formulas import _formula_png

    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "x + y", fontsize=12)
    calls = []
    original = pymupdf.Page.get_pixmap
    monkeypatch.setattr(pymupdf.Page, "get_pixmap", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    formula = {"png": _FormulaImage(doc, 0, pymupdf.Rect(15, 35, 60, 55))}
    assert calls == []
    png = _formula_png(formula)
    assert png.startswith(b"\x89PNG") and _formula_png(formula) is png and len(calls) == 1
    assert _formula_png({"png": b"given"}) == b"given" and _formula_png({}) is None


def test_formula_glyphs_stay_in_place_after_a_size_change_on_their_line():
    """When a line changes font size midway (small caps), MuPDF misplaces the formula after deleting the glyphs before
    it. If it is misplaced the whole line is kept, and the other lines are still deleted."""
    from app.formats.pdf_place_formulas import _keep_marks, _pdf_rect
    from app.formats.pdf_write_geometry import _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /F1 {font} 0 R >> >>")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(
        stream,
        (
            b"BT /F1 10 Tf 40 150 Td [(see the)]TJ /F1 6 Tf [(SMALL CAPITALS)]TJ /F1 10 Tf [( here )]TJ [(X)]TJ ET "
            b"BT /F1 10 Tf 40 130 Td [(next line)]TJ ET"
        ),
    )
    doc.xref_set_key(page.xref, "Contents", f"{stream} 0 R")
    target = page.search_for("X")[0]
    form = _keep_marks(doc, _page_form(doc, page), [_pdf_rect(page, target + (-1, -1, 1, 1))])

    shown = doc.new_page(width=400, height=200)
    draw = doc.get_new_xref()
    doc.update_object(draw, "<<>>")
    doc.update_stream(draw, b"q /Fm Do Q")
    doc.xref_set_key(shown.xref, "Contents", f"{draw} 0 R")
    doc.xref_set_key(shown.xref, "Resources", f"<< /XObject << /Fm {form} 0 R >> >>")
    words = doc[-1].get_text("words")
    assert [w[4] for w in words if w[4] == "X"] == ["X"]
    assert abs(next(w[0] for w in words if w[4] == "X") - target.x0) < 0.2
    assert not any(w[4] in ("next", "line") for w in words)


@pytest.mark.parametrize("box", ["CropBox", "MediaBox"])
def test_cjk_lines_follow_a_shifted_page_box(tmp_path, box):
    """When the crop box or media box does not start at the origin, Chinese written glyph by glyph still lands at the
    position of the original text."""
    from app.formats.pdf_extract import extract_blocks
    from app.formats.pdf_write import _render_translated

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for i in range(4):
        page.insert_text((120, 150 + i * 14), "The model computes the value for each input token here.", fontsize=10)
    if box == "CropBox":
        page.set_cropbox(pymupdf.Rect(80, 80, 520, 720))
    else:
        doc.xref_set_key(page.xref, "MediaBox", "[50 50 650 850]")
    src = tmp_path / "shifted.pdf"
    doc.save(src)
    source = pymupdf.open(src)[0].search_for("The model")[0]
    blocks = extract_blocks(pymupdf.open(src))
    assert len(blocks) == 1

    out = _render_translated(src, blocks, ["模型为每个输入词元计算这个值。"], "zh-CN")
    chars = _cjk_chars(out[0])
    assert chars
    assert abs(min(c[1] for c in chars) - source.x0) < 2
    assert abs(min(c[3] for c in chars) - (source.y1 - 2)) < 4


def test_formula_passthrough_follows_a_shifted_crop_box(tmp_path):
    """With an offset crop box, formulas are still cropped from their original position and placed back in their own
    slots."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_crop import _marker_png
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=400)
    page.insert_text((130, 150), "alpha BETA gamma", fontsize=10)
    page.set_cropbox(pymupdf.Rect(80, 100, 420, 360))
    doc.save(src)
    doc.close()

    d = pymupdf.open(src)
    rect = d[0].search_for("BETA")[0]
    formula = {
        "bbox": rect,
        "name": "a_1.png",
        "mid": 0,
        "w": round(rect.width, 1),
        "h": round(rect.height, 1),
        "d": 2.0,
        "text": "x",
        "has_img": True,
        "png": d[0].get_pixmap(dpi=150, clip=rect).tobytes("png"),
        "spans": [{"bbox": tuple(rect), "text": "x", "size": 10, "origin": (rect.x0, rect.y1 - 2)}],
    }
    line = pymupdf.Rect(rect.x0 - 45, rect.y0 - 2, rect.x1 + 60, rect.y1 + 3)
    d.close()
    archive = pymupdf.Archive()
    archive.add(_marker_png(0), "ph_a_1.png")
    block = TextBlock(page=0, rect=line, line_rects=[line], text="alpha BETA gamma", size=10, color="#000", bold=False)
    out = _render_translated(src, [block], ["译文 \x01i\x021\x01/i\x02 甲"], "zh-CN", {id(block): [formula]}, archive)
    hits = out[0].search_for("BETA")
    assert len(hits) == 1
    assert pymupdf.Rect(line.x0 - 10, line.y0 - 10, line.x1 + 10, line.y1 + 10).contains(hits[0])


def test_formula_clip_ignores_tall_bitmap_glyph_boxes(tmp_path):
    """When the next line is a bitmap font without Unicode, its reported font box is taller than its ink and reaches
    into the lower half of the formula. Such a box must not be used to cut holes, otherwise the lower half of the
    formula is cropped off."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_crop import _marker_png
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    procs = []
    for _ in range(10):
        xref = doc.get_new_xref()
        doc.update_object(xref, "<<>>")
        doc.update_stream(xref, b"500 0 0 0 400 600 d1 0 0 400 600 re f")
        procs.append(xref)
    glyphs = " ".join(f"/a{i}" for i in range(10))
    charprocs = " ".join(f"/a{i} {xref} 0 R" for i, xref in enumerate(procs))
    font = doc.get_new_xref()
    doc.update_object(
        font,
        (
            "<< /Type /Font /Subtype /Type3 /Name /BitmapF /FontBBox [0 -200 500 1800] "
            f"/FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << {charprocs} >> "
            f"/Encoding << /Type /Encoding /Differences [65 {glyphs}] >> "
            f"/FirstChar 65 /LastChar 74 /Widths [{' '.join(['500'] * 10)}] /Resources << >> >>"
        ),
    )
    doc.xref_set_key(
        page.xref,
        "Resources",
        f"<< /Font << /T3 {font} 0 R /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>",
    )
    codes = "".join(f"{65 + i % 10:02X}" for i in range(20))
    content = doc.get_new_xref()
    doc.update_object(content, "<<>>")
    doc.update_stream(
        content, (f"BT /F1 10 Tf 50 200 Td (alpha BETA gamma) Tj ET\nBT /T3 12 Tf 50 182 Td <{codes}> Tj ET").encode()
    )
    doc.xref_set_key(page.xref, "Contents", f"{content} 0 R")
    doc.save(src)
    doc.close()

    d = pymupdf.open(src)
    rect = d[0].search_for("BETA")[0]
    bitmap = next(
        span
        for block in d[0].get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["font"] == "BitmapF"
    )
    assert bitmap["bbox"][1] < rect.y1  # The bitmap glyph box does reach into the lower half of BETA

    def ink_share(pix):
        return sum(1 for v in pix.samples if v < 128) / len(pix.samples)

    ink = ink_share(d[0].get_pixmap(dpi=150, clip=rect, colorspace=pymupdf.csGRAY))
    formula = {
        "bbox": rect,
        "name": "a_1.png",
        "mid": 0,
        "w": round(rect.width, 1),
        "h": round(rect.height, 1),
        "d": 2.0,
        "text": "x",
        "has_img": True,
        "png": d[0].get_pixmap(dpi=150, clip=rect).tobytes("png"),
        "spans": [{"bbox": tuple(rect), "text": "x", "size": 10, "origin": (rect.x0, rect.y1 - 2)}],
    }
    line = pymupdf.Rect(45, rect.y0 - 2, 200, rect.y1 + 3)
    d.close()
    archive = pymupdf.Archive()
    archive.add(_marker_png(0), "ph_a_1.png")
    block = TextBlock(page=0, rect=line, line_rects=[line], text="alpha BETA gamma", size=10, color="#000", bold=False)
    out = _render_translated(src, [block], ["译文 \x01i\x021\x01/i\x02 甲"], "zh-CN", {id(block): [formula]}, archive)
    slot = out[0].search_for("BETA")[0]
    # Translated lines are set at 0.88 times the font size and the formula shrinks with them, so ink shares are
    # compared. With the lower half cropped off only about half remains
    drawn = ink_share(out[0].get_pixmap(dpi=150, clip=slot, colorspace=pymupdf.csGRAY))
    assert drawn >= ink * 0.65


def test_formula_crop_trims_side_whitespace_to_the_ink():
    """Leading spaces inside glyph boxes no longer remain on either side of the formula image. Spaces inside the
    formula stay. A neighboring glyph overlapping from the left does not count as ink of this formula."""
    from app.formats.pdf_crop import _tighten_formula_sides

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 80), "word", fontsize=12, fontname="tiro")
    page.insert_text((100, 80), " y", fontsize=12, fontname="tiro")
    page.insert_text((40, 130), "a = b", fontsize=12, fontname="times-roman")
    spans = [
        span
        for block in page.get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    ]
    spaced = next(span for span in spans if span["text"] == " y")
    equation = next(span for span in spans if "a = b" in span["text"])
    word = next(span for span in spans if span["text"] == "word")

    wide = pymupdf.Rect(spaced["bbox"])
    part = {
        "bbox": wide,
        "w": round(wide.width, 1),
        "dx": 0.0,
        "spans": [spaced],
        "curves": [],
    }
    _tighten_formula_sides(part, page)
    tight = pymupdf.Rect(part["bbox"])
    assert tight.x0 >= wide.x0 + 1.5
    assert tight.x1 <= wide.x1 + 0.2
    assert abs(tight.width - part["w"]) < 0.15
    assert part["dx"] == round(tight.x0 - wide.x0, 1)
    # The letters are still in the box, and the blank space on both sides shrinks to a narrow margin
    letter = next(
        pymupdf.Rect(char["bbox"])
        for span in page.get_text("rawdict")["blocks"]
        if span.get("type") == 0
        for line in span["lines"]
        for raw in line["spans"]
        for char in raw["chars"]
        if char["c"] == "y" and abs(char["bbox"][0] - 100) < 30
    )
    assert tight.x0 <= letter.x0 + 0.2
    assert tight.x1 >= letter.x1 - 0.2
    assert tight.x0 >= letter.x0 - 1.0
    assert tight.x1 <= letter.x1 + 1.0

    inner = pymupdf.Rect(equation["bbox"])
    kept = {"bbox": inner, "w": round(inner.width, 1), "spans": [equation], "curves": []}
    _tighten_formula_sides(kept, page)
    kept_box = pymupdf.Rect(kept["bbox"])
    assert kept_box.width > inner.width * 0.7
    assert kept_box.x0 < inner.x0 + 2
    assert kept_box.x1 > inner.x1 - 2

    word_box = pymupdf.Rect(word["bbox"])
    overlapped = pymupdf.Rect(word_box.x1 - 4, wide.y0, wide.x1, wide.y1)
    neighbor = {
        "bbox": overlapped,
        "w": round(overlapped.width, 1),
        "spans": [spaced],
        "curves": [],
    }
    _tighten_formula_sides(neighbor, page)
    trimmed = pymupdf.Rect(neighbor["bbox"])
    assert trimmed.x0 > word_box.x1 - 0.5

    page.draw_line((70, 78), (96, 78), width=0.8)
    bar = pymupdf.Rect(70, 77.2, 96, 78.8)
    with_bar = {
        "bbox": pymupdf.Rect(bar.x0, wide.y0, wide.x1, wide.y1),
        "w": 40.0,
        "spans": [spaced],
        "curves": [bar],
    }
    _tighten_formula_sides(with_bar, page)
    assert pymupdf.Rect(with_bar["bbox"]).x0 <= bar.x0 + 0.6
    doc.close()


def test_wrapped_formula_parts_stay_in_order(tmp_path):
    """The two pieces sit side by side in reading order, and the second half no longer stays at the source x at the
    start of the next line."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 80), "y= left", fontsize=11)
    page.insert_text((40, 100), "sh right", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    y_box = d[0].search_for("y=")[0]
    sh_box = d[0].search_for("sh")[0]
    d.close()

    def part(name, rect, text):
        return {
            "bbox": rect,
            "name": name,
            "w": round(rect.width, 1),
            "h": round(rect.height, 1),
            "d": 2.0,
            "raise": 0.0,
            "text": text,
            "has_img": True,
            "page": 0,
            "png": pymupdf.open(src)[0].get_pixmap(clip=rect, dpi=72).tobytes("png"),
            "spans": [{"bbox": tuple(rect), "text": text, "size": 10, "origin": (rect.x0, rect.y1 - 2)}],
        }

    first = part("f0_1.png", y_box, "y=")
    second = part("f0_1_p1.png", sh_box, "sh")
    first["parts"] = [first, second]
    first["has_img"] = True
    first["mid"] = 0
    rect = pymupdf.Rect(40, 40, 360, 80)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect, pymupdf.Rect(y_box), pymupdf.Rect(sh_box)],
        text="take arg max",
        size=10,
        color="#000",
        bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    out = _render_translated(src, [block], [f"也可以取 arg max({sentinel})。"], "zh-CN", {id(block): [first]})
    page = out[0]
    y_hit = page.search_for("y=")
    sh_hit = page.search_for("sh")
    assert y_hit and sh_hit
    assert y_hit[0].x0 < sh_hit[0].x0
    assert sh_hit[0].x0 - y_hit[0].x1 < 20
    assert abs(y_hit[0].y0 - sh_hit[0].y0) < 8


def test_translation_wraps_around_the_figure_its_paragraph_wraps_around():
    """The first lines of the original paragraph make room left of a figure and return to full width below it.
    Translated lines at the height of the figure make room the same way, and lines below the figure are written
    full width. A single line beside a figure is not wrapping around it (the source beside the last line of a
    quotation)."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_place import _place_plain
    from app.formats.pdf_write_geometry import _wrap_holes

    image = pymupdf.Rect(380, 268, 493, 428)
    caption = pymupdf.Rect(380, 432, 486, 443)
    narrow = [pymupdf.Rect(160, y, 370, y + 12) for y in range(385, 445, 12)]
    wide = [pymupdf.Rect(160, y, 493, y + 12) for y in range(445, 517, 12)]
    lines = narrow + wide
    block = TextBlock(
        page=0, rect=pymupdf.Rect(160, 385, 493, 517), line_rects=lines, text="x", size=10, color="#000000", bold=False
    )
    holes = _wrap_holes(block, [image, caption, pymupdf.Rect(131, 471, 146, 480)])
    assert sorted(map(tuple, holes)) == sorted([tuple(image), tuple(caption)])

    quote = [pymupdf.Rect(167, y, 475, y + 12) for y in (210, 222, 234)] + [pymupdf.Rect(167, 246, 203, 258)]
    quoted = TextBlock(
        page=0, rect=pymupdf.Rect(167, 210, 475, 258), line_rects=quote, text="x", size=10, color="#000000", bold=False
    )
    assert _wrap_holes(quoted, [pymupdf.Rect(411, 246, 475, 258)]) == []

    # The half paragraph carried over from the previous page is written at the column width of that page. A figure
    # outside the block box but inside the writing area gets room too
    figure = pymupdf.Rect(347, 103, 493, 253)
    tail_lines = [pymupdf.Rect(160, y, 337, y + 12) for y in range(110, 170, 12)]
    tail = TextBlock(
        page=0,
        rect=pymupdf.Rect(160, 110, 337, 170),
        line_rects=tail_lines,
        text="x",
        size=10,
        color="#000000",
        bold=False,
    )
    assert _wrap_holes(tail, [figure]) == []
    assert _wrap_holes(tail, [figure], pymupdf.Rect(160, 110, 493, 168)) == [figure]

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _place_plain(
        page,
        "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 8,
        [],
        pymupdf.Rect(160, 386, 495, 521),
        8.8,
        10,
        True,
        "left",
        "",
        "#000000",
        "normal",
        [],
        None,
        None,
        False,
        holes,
    )
    spans = [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip()
    ]
    assert spans
    for span in spans:
        box = pymupdf.Rect(span["bbox"])
        assert not box.intersects(image) and not box.intersects(caption), span["text"]
    assert any(pymupdf.Rect(span["bbox"]).x1 > 450 and span["bbox"][1] > 443 for span in spans)


def test_cjk_bold_strokes_once(tmp_path):
    """A Chinese character made bold by stroking is still one character after extraction."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=400, height=200)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 280, 80)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="Modularity. Using a highly extensible abstraction of metadata.",
        size=9,
        color="#000000",
        bold=False,
        span_lines=[
            [
                {"text": "Modularity.", "font": "CMBX9", "flags": 16},
                {"text": " Using a highly extensible abstraction of metadata.", "font": "CMR9", "flags": 4},
            ]
        ],
    )
    out = _render_translated(src, [block], ["模块化。借助高度可扩展的抽象来描述元数据。"], "zh-CN")
    text = out[0].get_text()
    assert text.count("模") == 1
    assert "2 Tr" in out[0].read_contents().decode("latin1")


def test_a_bitmap_glyph_without_its_own_box_is_kept_by_its_extent():
    """Bitmap glyphs report no box of their own (width 0), only the vertical bounds of the whole font, so the computed
    center is half a line away from the ink. Horizontally the origin is checked and vertically the overlap of the
    bounds with the keep box, so superscripts still stay; a glyph whose origin is outside the box is deleted as
    before."""
    from types import SimpleNamespace as NS

    from app.formats.pdf_place_formulas import _GlyphKeeper

    keeper = _GlyphKeeper([(484.9, 240.1, 490.5, 244.7)])
    trm = NS(a=0.1, b=0.0, c=0.0, d=-0.1, e=485.5, f=249.8)
    ctm = NS(a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0)
    font_box = NS(x0=0.0, y0=-65.0, x1=0.0, y1=86.0)
    assert keeper.text_filter(None, None, 0, trm, ctm, font_box) == 0
    beside = NS(a=0.1, b=0.0, c=0.0, d=-0.1, e=495.3, f=249.8)
    assert keeper.text_filter(None, None, 0, beside, ctm, font_box) == 1
    # Glyphs that report their own box are still judged by their center
    own_box = NS(x0=0.0, y0=-65.0, x1=60.0, y1=126.0)
    assert keeper.text_filter(None, None, 0, trm, ctm, own_box) == 1


def test_cross_page_paragraph_continues_on_the_next_page(tmp_path):
    """A translation carried over to the next page is set in full lines. Once this page is full it continues on the
    next page, lines are not drawn past the page margin, and footnotes stay in place."""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    first = doc.new_page(width=420, height=220)
    first.insert_text((40, 168), "performs quite", fontsize=10)
    first.insert_text((40, 200), "1 Note here.", fontsize=8)
    second = doc.new_page(width=420, height=220)
    second.insert_text((40, 48), "well. We hypothesize.", fontsize=10)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 140, 300, 174),
        line_rects=[pymupdf.Rect(40, 156, 180, 174)],
        text="performs quite",
        size=10,
        color="#000000",
        bold=False,
    )
    note = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 188, 180, 206),
        line_rects=[pymupdf.Rect(40, 188, 140, 206)],
        text="1 Note here.",
        size=8,
        color="#000000",
        bold=False,
    )
    tail = TextBlock(
        page=1,
        rect=pymupdf.Rect(40, 36, 300, 80),
        line_rects=[pymupdf.Rect(40, 36, 220, 54)],
        text="well. We hypothesize.",
        size=10,
        color="#000000",
        bold=False,
    )
    paragraph = (
        "对于这个问题，模型的表现相当不错。我们推测这与数据集的性质有关，"
        "因此这一段按满行往下排，这一页排满再从下一页接着写，不要把行画出页边。"
        "后面这几行放不进页末的空隙，就从下一页的栏首继续，脚注仍留在原来的位置。"
    )
    out = _render_translated(
        src,
        [host, note, tail],
        [paragraph, "脚注仍在原处。", SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    page0 = out[0].get_text()
    page1 = out[1].get_text()
    assert "相当不错" in page0 + page1
    assert "well" not in page1
    assert out[0].rect.height == 220
    assert all(
        span["bbox"][3] <= 221
        for block in out[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    )
    note_top = None
    for block in out[0].get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
        if "脚注" in text:
            note_top = block["bbox"][1]
    assert note_top is not None and note_top > 180
    assert page1.strip()
    assert "相当不错" in page0
    assert "相当" not in page1 or "不错" in page1


def test_split_filled_lines_keeps_breaks_when_the_page_has_room():
    """When the text fits on this page, the lines already broken are returned. Returning the original text would make
    the caller shrink the font size to fit one line."""
    from app.formats.pdf_write_geometry import _split_filled_lines

    text = "在实验研究中我们首先比较旧优化器与新优化器的计划质量并说明新优化器的收益"
    head, rest = _split_filled_lines(text, [], em=8.8, block_size=10, width=120, cjk=True, y0=100, limit=400)
    assert "\n" in head
    assert rest == ""


def test_split_filled_lines_fills_the_last_lines_of_the_page():
    """The page bottom has room for one or two original lines. The translation does not fit at the original line
    spacing, so the spacing tightens and the line spacing of the last line extends past the box, keeping the text
    on this page."""
    from app.formats.pdf_write_geometry import _filled_bottom, _split_filled_lines

    text = "人工智能领域的一项名为发展的研究计划的成员，并发表了一篇论文。" * 3
    head, rest = _split_filled_lines(text, [], em=8.8, block_size=10, width=300, cjk=True, y0=700, limit=712)
    assert head and "\n" not in head
    assert rest
    head, rest = _split_filled_lines(text, [], em=8.8, block_size=10, width=300, cjk=True, y0=696.5, limit=719.5)
    assert head.count("\n") == 1
    assert rest
    bottom = _filled_bottom(700, 1, 8.8, True, 712, 792)
    assert bottom - 700 >= 8.8 * 1.45 + 8.8 * 0.3 - 0.1
    assert bottom <= 712 + 0.45 * 8.8 + 0.01
    # When the writing box is clipped at the bottom of the original body text and limit is that line, the line
    # spacing of the last line cannot extend: one line fewer stays here and the paragraph is not shrunk
    text = "上界不匹配，因此必须强制执行谓词，这可以简化为一个等值谓词。" * 8
    roomy, _ = _split_filled_lines(text, [], em=7.92, block_size=9, width=240, cjk=True, y0=664.1, limit=722.5)
    capped, _ = _split_filled_lines(
        text,
        [],
        em=7.92,
        block_size=9,
        width=240,
        cjk=True,
        y0=664.1,
        limit=722.5,
        page_bottom=722.5,
    )
    assert roomy.count("\n") == 5
    assert capped.count("\n") == 4


def test_split_filled_lines_keeps_the_line_gap_when_the_continuation_has_room():
    """The page bottom holds one line at normal line spacing or two when tightened. When the tail block can hold the
    remaining lines this page does not tighten; it tightens only when they do not fit."""
    from app.formats.pdf_write_geometry import _split_filled_lines

    text = "人工智能领域的一项名为发展的研究计划的成员，并发表了一篇论文。" * 3
    args = dict(em=8.8, block_size=10, width=300, cjk=True, y0=696.5, limit=719.5)
    tight, _ = _split_filled_lines(text, [], **args)
    assert tight.count("\n") == 1
    roomy, rest = _split_filled_lines(text, [], **args, next_room=200)
    assert roomy and "\n" not in roomy and rest
    short, _ = _split_filled_lines(text, [], **args, next_room=12)
    assert short == tight
    # When the whole paragraph fits after tightening, it still carries over if the tail block has room
    two = "人工智能领域的一项名为发展的研究计划的成员，并发表了一篇论文。" * 2
    whole, rest = _split_filled_lines(two, [], **args)
    assert whole.count("\n") == 1 and rest == ""
    head, rest = _split_filled_lines(two, [], **args, next_room=40)
    assert "\n" not in head and rest


def test_fitted_cross_page_paragraph_keeps_body_size(tmp_path):
    """A translation carried over to the next page keeps the original font size when it fits on this page and is no
    longer squeezed into a box one line high."""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    first = doc.new_page(width=420, height=400)
    first.insert_text((40, 220), "In our experimental study we", fontsize=10)
    second = doc.new_page(width=420, height=400)
    second.insert_text((40, 48), "first compare the systems.", fontsize=10)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 200, 280, 230),
        line_rects=[pymupdf.Rect(40, 208, 220, 222), pymupdf.Rect(40, 222, 180, 234)],
        text="In our experimental study we",
        size=10,
        color="#000000",
        bold=False,
    )
    tail = TextBlock(
        page=1,
        rect=pymupdf.Rect(40, 36, 280, 70),
        line_rects=[pymupdf.Rect(40, 36, 220, 50)],
        text="first compare the systems.",
        size=10,
        color="#000000",
        bold=False,
    )
    paragraph = (
        "在实验研究中我们首先比较旧优化器与新优化器的计划质量，"
        "并说明新优化器在大规模数据上的收益，这一段应该保持正文字号。"
    )
    out = _render_translated(
        src,
        [host, tail],
        [paragraph, SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    sizes = [
        span["size"]
        for block in out[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in span["text"])
    ]
    assert sizes
    assert min(sizes) >= 7


def test_a_cross_page_paragraph_borrows_on_the_next_page(tmp_path):
    """The tail block of a paragraph carried over to the next page has a paragraph right below it with spare room:
    this page holds one line fewer at normal line spacing, the extra goes to the next page, and the tail block
    extends into the paragraph below, which moves down"""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    for bottom in (256, 124):
        page = doc.new_page(width=400, height=300)
        for y in range(40, bottom, 12):
            page.insert_text((40, y + 9), "x " * 40, fontsize=10)
    doc.save(src)
    doc.close()

    def block(page: int, y0: float, count: int, text: str) -> TextBlock:
        rows = [pymupdf.Rect(40, y0 + 12 * index, 360, y0 + 12 * index + 12) for index in range(count)]
        return TextBlock(
            page=page,
            rect=pymupdf.Rect(40, y0, 360, y0 + 12 * count),
            line_rects=rows,
            text=text,
            size=10,
            color="#000000",
            bold=False,
            body_size=10,
        )

    above = block(0, 40, 15, "x " * 60 + "end.")
    host = block(0, 220, 3, "x " * 60)
    tail = block(1, 40, 2, "x " * 30 + "end.")
    below = block(1, 64, 5, "x " * 60 + "end.")
    paragraph = "在实验研究中我们首先比较旧优化器与新优化器的计划质量，并说明新优化器在大规模数据上的收益。" * 4
    out = _render_translated(
        src,
        [above, host, tail, below],
        ["上面一段。", paragraph, SPILL_TAIL, "下面一段原文五行，译文一行。"],
        "zh-CN",
        continuations={id(host): tail},
    )

    def baselines(page) -> list[float]:
        return sorted(
            line["spans"][0]["origin"][1]
            for item in page.get_text("dict")["blocks"]
            if item.get("type") == 0
            for line in item["lines"]
            if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
            and line["spans"][0]["origin"][1] > 200
        )

    first = baselines(out[0])
    assert first and all(abs(lower - upper - 8.8 * 1.45) < 0.2 for upper, lower in zip(first, first[1:]))
    second = sorted(
        line["spans"][0]["origin"][1]
        for item in out[1].get_text("dict")["blocks"]
        if item.get("type") == 0
        for line in item["lines"]
        if re.search(r"[一-鿿]", "".join(span["text"] for span in line["spans"]))
    )
    pitches = [lower - upper for upper, lower in zip(second, second[1:])]
    assert len(second) >= 3
    assert all(abs(pitch - 8.8 * 1.45) < 0.2 for pitch in pitches[:-1])
    assert pitches[-1] >= 8.8 * 1.45 * 1.4


def test_carried_lines_keep_the_host_size(tmp_path):
    """The block on the continuation page is extracted with a slightly larger font size. Lines carried over are broken
    at the size of the previous block and written at that size too: written at the size of the tail block, one
    character would be pushed out at the line end, and the text would not fit and shrink."""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=420, height=300)
    doc.new_page(width=420, height=300)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 230, 375, 262),
        line_rects=[pymupdf.Rect(40, 230, 375, 246), pymupdf.Rect(40, 246, 375, 262)],
        text="In our experimental study we",
        size=10.8,
        color="#000000",
        bold=False,
    )
    tail = TextBlock(
        page=1,
        rect=pymupdf.Rect(40, 40, 375, 120),
        line_rects=[pymupdf.Rect(40, 40 + 16 * k, 375, 56 + 16 * k) for k in range(5)],
        text="first compare the systems.",
        size=11.0,
        color="#000000",
        bold=False,
    )
    paragraph = "在实验研究中我们首先比较旧优化器与新优化器的计划质量，并说明新优化器在大规模数据上的收益。" * 4
    out = _render_translated(
        src,
        [host, tail],
        [paragraph, SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    rows: dict[float, str] = {}
    for block in out[1].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if any("一" <= ch <= "鿿" for ch in span["text"]):
                    assert abs(span["size"] - 10.8 * 0.88) < 0.05
                    key = round(span["origin"][1])
                    rows[key] = rows.get(key, "") + span["text"]
    assert len(rows) >= 3
    assert all(len(text.strip()) > 3 for text in list(rows.values())[:-1])


def test_narrow_fragment_is_written_at_the_continuation_width(tmp_path):
    """A half sentence at the line end is not laid out in its own narrow box. The translation continues on the
    continuation page at the body font size."""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=420, height=300)
    doc.new_page(width=420, height=300)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0,
        rect=pymupdf.Rect(250, 250, 290, 262),
        line_rects=[pymupdf.Rect(250, 250, 290, 262)],
        text="An opti-",
        size=10,
        color="#000000",
        bold=False,
    )
    tail = TextBlock(
        page=1,
        rect=pymupdf.Rect(40, 40, 280, 90),
        line_rects=[pymupdf.Rect(40, 40, 280, 54)],
        text="mization stage continues here.",
        size=10,
        color="#000000",
        bold=False,
    )
    paragraph = "优化阶段在满足下列任一条件时结束，资源受限的系统也可以在这里停下来。"
    out = _render_translated(
        src,
        [host, tail],
        [paragraph, SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    sizes = [
        span["size"]
        for page in out
        for block in page.get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in span["text"])
    ]
    assert sizes
    assert min(sizes) >= 7
    assert out[1].get_text().strip()


def test_shared_baseline_continuation_stays_in_its_lines(tmp_path):
    """A continuation line sharing a baseline is written only in its own two lines, without covering the next paragraph
    on the right, at the body font size."""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((140, 126), "For example the next sentence.", fontsize=9)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0,
        rect=pymupdf.Rect(160, 100, 300, 112),
        line_rects=[pymupdf.Rect(160, 100, 300, 112)],
        text="Statistics promise computation is",
        size=9,
        color="#000000",
        bold=False,
    )
    tail = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 114, 130, 126),
        line_rects=[pymupdf.Rect(40, 114, 130, 126)],
        text="expression-specific.",
        size=9,
        color="#000000",
        bold=False,
    )
    neighbor_lines = [pymupdf.Rect(140, 114, 300, 126), pymupdf.Rect(40, 128, 300, 140)]
    neighbor = TextBlock(
        page=0,
        rect=neighbor_lines[0] | neighbor_lines[1],
        line_rects=neighbor_lines,
        text="For example the next sentence starts here.",
        size=9,
        color="#000000",
        bold=False,
    )
    paragraph = "统计承诺所描述的计算过程只对特定表达式成立。"
    out = _render_translated(
        src,
        [host, neighbor, tail],
        [paragraph, neighbor.text, SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    spans = [
        span
        for block in out[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in span["text"])
    ]
    assert spans
    assert min(span["size"] for span in spans) >= 7
    assert any(span["bbox"][1] < 113 for span in spans)
    assert any(span["bbox"][1] >= 113 for span in spans)
    for span in spans:
        x0, y0, x1, _y1 = span["bbox"]
        if y0 >= 113:
            assert x1 <= 136
        else:
            assert x0 >= 155
    assert "For example" in out[0].get_text()


def test_tighten_line_height_for_cjk():
    from app.formats.pdf_place import _tighten_line_height

    cjk = "* {line-height: 1.45; margin: 0;}"
    latin = "* {line-height: 1.2; margin: 0;}"
    assert "line-height: 1.2" in _tighten_line_height(cjk)
    assert "line-height: 1.1" in _tighten_line_height(latin)
    assert _tighten_line_height(cjk) != cjk


def test_save_after_redact_keeps_resources_indirect(tmp_path):
    """redact turns /Resources into an inline dictionary. Before writing it must become an indirect reference again,
    otherwise a large dictionary gets corrupted when saving with garbage collection."""
    import pymupdf as pm

    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "t.pdf"
    doc = pm.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((40, 60), "First block here")
    for i in range(8):
        page.insert_text((40, 80 + i * 12), f"Body line number {i} with enough text to matter")
    doc.save(src)
    doc.close()

    lines = ["First block here"] + [f"Body line number {n} with enough text to matter" for n in range(8)]
    blocks = []
    for index, line in enumerate(lines):
        y = 48 + index * 12
        rect = pm.Rect(40, y, 360, y + 12)
        blocks.append(
            TextBlock(
                page=0,
                rect=rect,
                line_rects=[rect],
                text=line,
                size=11,
                color="#000000",
                bold=False,
            )
        )
    trans = [f"译文第{i}段的内容写在这里测试保存" for i in range(len(blocks))]
    out = _render_translated(src, blocks, trans, "zh-CN")
    kind, value = out.xref_get_key(out[0].xref, "Resources")
    assert kind == "xref" and value.endswith(" 0 R")
    data = out.tobytes(garbage=4, deflate=True)
    check = pm.open("pdf", data)
    text = check[0].get_text()
    assert "译文第0段" in text and f"译文第{len(blocks) - 1}段" in text


def test_a_centered_title_breaks_after_a_clause_mark():
    """A centered subtitle that does not fit on one line breaks after a comma, so "识别" is not left alone on a line;
    when it fits on one line it is not broken."""
    from app.formats.pdf_place import _clause_lines
    from app.formats.pdf_typeset import typeset_lines

    text = "借助语言模型介绍自然语言处理, 计算语言学和语音识别"

    def broken(width):
        lines = _clause_lines(text, [], 10.0, 10.0, width, typeset_lines(text, 10.0, width, {}))
        return ["".join(piece.text for piece in line.pieces) for line in lines]

    assert broken(230.0) == ["借助语言模型介绍自然语言处理,", "计算语言学和语音识别"]
    assert broken(260.0) == [text]
    # An author list that is bold throughout breaks after a comma, and the second line is still bold
    names = "{b}Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed{/b}"
    lines = _clause_lines(names, [], 10.0, 10.0, 300.0, typeset_lines(names, 10.0, 300.0, {}))
    texts = ["".join(piece.text for piece in line.pieces) for line in lines]
    assert len(texts) == 2 and all(line.startswith("{b}") and line.endswith("{/b}") for line in texts), texts


def test_right_fragment_stays_on_its_own_line():
    """The first line that makes room for a side label is still written from that line, using its own left edge, and
    the paragraph does not move down."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write_geometry import _flow_y0, _inset_write_boxes

    fragment = pymupdf.Rect(190, 40, 360, 52)
    nxt = pymupdf.Rect(40, 56, 360, 68)
    block = TextBlock(
        page=0,
        rect=fragment | nxt,
        line_rects=[fragment, nxt],
        text="At the end of exploration.",
        size=9,
        color="#000",
        bold=False,
    )
    assert _flow_y0(block, True) == pytest.approx(40 + 0.9, abs=0.01)
    first, rest = _inset_write_boxes(block, True)
    assert first.x0 == pytest.approx(190)
    assert rest.x0 == pytest.approx(40)
    assert rest.y0 > first.y0
    indent = pymupdf.Rect(52, 40, 360, 52)
    body = pymupdf.Rect(40, 56, 360, 68)
    indented = TextBlock(
        page=0,
        rect=indent | body,
        line_rects=[indent, body],
        text="A normal indented paragraph.",
        size=9,
        color="#000",
        bold=False,
    )
    assert _flow_y0(indented, False) == 40
    assert _inset_write_boxes(indented, False) is None
    # Fragments on the same baseline are not the next line. Continuation lines start at the real next line and are
    # not pushed back onto the first line.
    fragments = [
        pymupdf.Rect(127.9, 595.2, 134.1, 604.2),
        pymupdf.Rect(141.4, 595.2, 198.1, 604.2),
        pymupdf.Rect(205.5, 595.2, 292.9, 604.2),
        pymupdf.Rect(53.8, 605.7, 292.9, 614.7),
    ]
    torn = TextBlock(
        page=0,
        rect=fragments[0] | fragments[-1],
        line_rects=fragments,
        text="is an extensible optimizer framework.",
        size=9,
        color="#000",
        bold=False,
    )
    torn_first, torn_rest = _inset_write_boxes(torn, True)
    assert torn_first.x0 == pytest.approx(127.9)
    assert torn_rest.x0 == pytest.approx(53.8)
    assert torn_rest.y0 > torn_first.y0 + 4


def test_write_helpers_only_need_the_blocks_on_their_page():
    """Writing passes only the blocks of the same page. Each layout check gives the same result with the blocks of the
    whole book as with the blocks of this page only."""
    import random

    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write_geometry import (
        _affiliation_write_rects,
        _beside_body,
        _body_first_starts,
        _fit_page_margins,
        _room_below,
    )

    rng = random.Random(5)
    texts = [
        "∗University of Somewhere",
        "†Institute of Elsewhere",
        "Memo.",
        "(2) Statistics Derivation.",
        "The body line runs on and on",
        "and the model keeps going.",
        "Results are shown here.",
    ]
    blocks = []
    for _ in range(90):
        x0 = rng.choice([40.0, 60.0, 120.0, 200.0, 320.0])
        y0 = rng.choice([30.0, 60.0, 100.0, 300.0, 700.0, 760.0]) + rng.randint(0, 3) * 12
        lines = [
            pymupdf.Rect(x0, y0 + i * 12, x0 + rng.choice([60, 140, 240]), y0 + i * 12 + 10)
            for i in range(rng.choice([1, 1, 2]))
        ]
        union = pymupdf.Rect(lines[0])
        for line in lines[1:]:
            union |= line
        blocks.append(
            TextBlock(
                page=rng.choice([0, 1, 2]),
                rect=union,
                line_rects=lines,
                text=rng.choice(texts),
                size=rng.choice([8.0, 10.0]),
                color="#000",
                bold=rng.random() < 0.4,
            )
        )

    def fixed(pno, text, lines, bold=False):
        rects = [pymupdf.Rect(*line) for line in lines]
        union = pymupdf.Rect(rects[0])
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=pno, rect=union, line_rects=rects, text=text, size=10, color="#000", bold=bold)

    # Each page has side-by-side affiliations, a side label and the body text beside it at the same position. A
    # check missing the same-page condition would mix in blocks from other pages.
    for pno in range(3):
        blocks += [
            fixed(pno, "∗University of Somewhere", [(60, 40, 180, 50)]),
            fixed(pno, "†Institute of Elsewhere", [(190, 40, 300, 50)]),
            fixed(pno, "Memo.", [(40, 100, 70, 110)], bold=True),
            fixed(pno, "This body line sits beside the label.", [(72, 100, 300, 110), (40, 112, 300, 122)]),
        ]
    page = pymupdf.open().new_page(width=600, height=800)
    found = {"stacked": 0, "starts": 0, "beside": 0}
    for pno in range(3):
        own = [block for block in blocks if block.page == pno]
        ids = {id(block) for block in own}
        whole = _affiliation_write_rects(blocks)
        stacked = _affiliation_write_rects(own)
        assert stacked == {key: rect for key, rect in whole.items() if key in ids}
        found["stacked"] += len(stacked)
        items = [(block, "译文" + block.text) for block in own]
        for cjk in (True, False):
            starts = _body_first_starts(own, items, cjk, None)
            assert starts == _body_first_starts(blocks, items, cjk, None)
            found["starts"] += len(starts)
            for block in own:
                rect = pymupdf.Rect(block.rect.x0, block.rect.y0, block.rect.x1 + 2, block.rect.y1 + 4)
                assert _fit_page_margins(block, rect, own, 800, cjk) == _fit_page_margins(block, rect, blocks, 800, cjk)
        for block in own:
            beside = _beside_body(block, own)
            assert beside is _beside_body(block, blocks)
            found["beside"] += beside is not None
            args = (page, block, block.rect.y0, block.rect.x0, block.rect.x1)
            assert _room_below(*args, own) == _room_below(*args, blocks)
    assert all(found.values())


def test_side_label_stays_beside_the_first_line(tmp_path):
    """A short heading stays left of the first body line. The body text starts right of the heading, and the next line
    returns to the left edge of the column."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=300).insert_text((40, 80), "heading and sentence", fontsize=11)
    doc.save(src)
    doc.close()
    heading_line = pymupdf.Rect(40, 40, 180, 52)
    fragment = pymupdf.Rect(190, 40, 360, 52)
    nxt = pymupdf.Rect(40, 56, 360, 68)
    heading = TextBlock(
        page=0,
        rect=heading_line,
        line_rects=[heading_line],
        text="(2) Statistics Derivation.",
        size=9,
        color="#000000",
        bold=True,
        span_lines=[[{"text": "(2) Statistics Derivation.", "font": "CMBX9", "flags": 16}]],
    )
    body = TextBlock(
        page=0,
        rect=fragment | nxt,
        line_rects=[fragment, nxt],
        text="At the end of exploration the memo keeps the space.",
        size=9,
        color="#000000",
        bold=False,
    )
    out = _render_translated(
        src,
        [body, heading],
        ["探索结束时备忘录保存完整的逻辑空间并且继续写下一段说明，后面这几个字回到栏左缘。", "(2) 统计信息推导。"],
        "zh-CN",
    )
    chars = _cjk_chars(out[0])
    heading_chars = [item for item in chars if item[3] < 54 and item[0] in "统计信息推导"]
    first_body = [item for item in chars if item[0] == "探"]
    assert heading_chars and first_body
    heading_right = max(item[2] for item in heading_chars)
    body_left = min(item[1] for item in first_body)
    assert heading_right <= 188
    # The English body text starts at 190. The Chinese label is shorter, so the first line fills the space freed and
    # does not overlap the label.
    assert body_left >= heading_right - 0.6
    assert body_left < 170
    assert abs(first_body[0][3] - heading_chars[0][3]) < 4
    later_body = [item for item in chars if item[3] > first_body[0][3] + 4]
    assert later_body
    assert min(item[1] for item in later_body) < 80


def test_inset_rest_reflows_at_the_column_width():
    """After the first line making room for a side label is broken at the narrow width, the rest carries no line
    breaks and is laid out again in the boxes of the continuation lines when writing. Line breaks fixed in advance
    would leave short half lines where the continuation lines make room for a figure."""
    from app.formats.pdf_write_geometry import _split_own_lines

    text = "Orca 包含一个可扩展框架, 用于基于形式化的属性规范描述查询需求和计划特征。属性有不同类型, 包括逻辑属性。"
    head, rest = _split_own_lines(text, [], 7.92, 9.0, 120.0)
    assert head and rest and "\n" not in rest
    assert "框架用于" not in head + rest
    glued = (head + rest.replace("\n", "")).replace(" ", "")
    assert glued == text.replace(" ", "")
    # Italic spans the end of the first line: the first line gets a closing mark at its end, the rest continues
    # with the second half, and the italic reopens at its start
    styled = "前面普通{i}斜体跨过第一行的末尾一直延续到下一行{/i}后面普通"
    head, rest = _split_own_lines(styled, [], 10.0, 10.0, 80.0)
    assert head == "前面普通{i}斜体跨过{/i}"
    assert rest == "{i}第一行的末尾一直延续到下一行{/i}后面普通"


def test_a_word_too_wide_for_the_shared_line_moves_whole_to_the_next_line():
    """When the rest of a line is only one short word wide and BERT does not fit, the whole word goes to the
    continuation line and is not cut into BER / T."""
    from app.formats.pdf_write_geometry import _split_own_lines

    assert _split_own_lines("BERT 和 GPT 在所有任务上的表现", [], 8.8, 10.0, 20.0) == (
        "",
        "BERT 和 GPT 在所有任务上的表现",
    )
    assert _split_own_lines("{b}BERT{/b} 在所有任务上", [], 8.8, 10.0, 20.0)[0] == ""
    head, rest = _split_own_lines("在所有任务上 BERT 都更好", [], 8.8, 10.0, 20.0)
    assert head == "在所" and rest.startswith("有任务")


def test_inset_continuation_fills_the_column(tmp_path):
    """For body text beside a side label, the continuation lines return to the left edge of the column and run to its
    right edge, so the font size is no longer reduced by too many narrow lines."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=400).insert_text((40, 80), "heading and sentence", fontsize=11)
    doc.save(src)
    doc.close()
    heading_line = pymupdf.Rect(40, 40, 180, 52)
    first = pymupdf.Rect(190, 40, 360, 52)
    later = [pymupdf.Rect(40, 56 + i * 12, 360, 68 + i * 12) for i in range(8)]
    heading = TextBlock(
        page=0,
        rect=heading_line,
        line_rects=[heading_line],
        text="(2) Statistics Derivation.",
        size=9,
        color="#000000",
        bold=True,
        span_lines=[[{"text": "(2) Statistics Derivation.", "font": "CMBX9", "flags": 16}]],
    )
    body = TextBlock(
        page=0,
        rect=first | later[-1],
        line_rects=[first, *later],
        text="At the end of exploration the memo keeps the space and continues.",
        size=9,
        color="#000000",
        bold=False,
    )
    text = (
        "探索结束时备忘录保存完整的逻辑空间并且继续写下一段很长的说明"
        "用来占满这一栏的宽度然后再换行继续写到栏的右缘这里应该用整栏宽度"
        "而不是停在第一行那么窄的位置上。"
    )
    out = _render_translated(src, [body, heading], [text, "(2) 统计信息推导。"], "zh-CN")
    chars = _cjk_chars(out[0])
    grouped = _group_chars(chars)
    body_lines = [line for line in grouped if line[0][1] >= 100 or line[0][3] > 58]
    assert len(body_lines) >= 3
    assert 80 < body_lines[0][0][1] < 170
    full = [line for line in body_lines[1:-1] if line[-1][2] > 300]
    assert full, [(line[0][1], line[-1][2], "".join(item[0] for item in line)) for line in body_lines]
    sizes = [
        span["size"]
        for block in out[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if span["bbox"][1] > 58 and span["bbox"][0] < 80 and span["text"].strip()
    ]
    assert sizes
    assert min(sizes) > 7.5


def test_same_baseline_fragments_do_not_cover_the_lead(tmp_path):
    """Fragments on the same baseline are not continuation lines. The continuation starts on the next line and does not
    overlap the untranslated Cascades on the left."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((62, 50), "Cascades", fontsize=9)
    doc.save(src)
    doc.close()
    lead = pymupdf.Rect(62.8, 40, 99, 52)
    fragments = [
        pymupdf.Rect(127.9, 40, 160, 52),
        pymupdf.Rect(168, 40, 220, 52),
        pymupdf.Rect(40, 56, 280, 68),
        pymupdf.Rect(40, 70, 280, 82),
    ]
    cascades = TextBlock(
        page=0,
        rect=lead,
        line_rects=[lead],
        text="Cascades",
        size=9,
        color="#000000",
        bold=False,
    )
    body = TextBlock(
        page=0,
        rect=fragments[0] | fragments[-1],
        line_rects=fragments,
        text="is an extensible optimizer framework whose principles have been used.",
        size=9,
        color="#000000",
        bold=False,
    )
    text = "是一个可扩展的优化器框架，其原则已被用于构建查询优化器，并且继续写到下一行。"
    out = _render_translated(src, [cascades, body], ["Cascades", text], "zh-CN")
    chars = _cjk_chars(out[0])
    first = [item for item in chars if item[0] == "是"]
    assert first
    same_line = [item for item in chars if abs(item[3] - first[0][3]) < 2]
    below = [item for item in chars if item[3] > first[0][3] + 4]
    assert same_line and below
    assert min(item[1] for item in same_line) >= 120
    assert "Cascades" in out[0].get_text()


def test_cross_page_stops_above_the_page_number(tmp_path):
    """A translation carried over to the next page stops above the page number band. The page number is digits only and
    not in the block list, and the text must not run over it."""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_layout import content_floor
    from app.formats.pdf_write import _render_translated
    from app.formats.pdf_write_geometry import _room_below

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=612, height=792)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0,
        rect=pymupdf.Rect(316.8, 699.8, 556, 719.3),
        line_rects=[
            pymupdf.Rect(325.8, 699.8, 556, 708.8),
            pymupdf.Rect(316.8, 710.3, 556, 719.3),
        ],
        text=(
            "Several efforts have addressed interactive processing on Hadoop by creating specialized query engines "
            "that allow"
        ),
        size=9,
        color="#000000",
        bold=False,
    )
    tail = TextBlock(
        page=1,
        rect=pymupdf.Rect(53.8, 56.8, 293, 180),
        line_rects=[pymupdf.Rect(53.8, 56.8, 293, 66)],
        text="SQL-based processing of data in HDFS.",
        size=9,
        color="#000000",
        bold=False,
    )
    paragraph = (
        "一些研究通过创建专用查询引擎来解决交互式处理问题，使得无需使用 MapReduce 即可基于 SQL 处理数据。"
        "这些方法在查询优化器和执行引擎的设计上各不相同。数据库与大数据平台的共置使数据能够以原生方式处理。"
        "后面这几行放不进页末，就从下一页接着写，不能压到页码上。"
    )
    out = _render_translated(
        src,
        [host, tail],
        [paragraph, SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    floor = content_floor(792)
    bottoms = [
        ch["bbox"][3]
        for block in out[0].get_text("rawdict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    assert bottoms
    assert max(bottoms) < floor + 4
    assert max(bottoms) < 745
    assert out[1].get_text().strip()
    note = TextBlock(
        page=0,
        rect=pymupdf.Rect(53, 750, 200, 762),
        line_rects=[pymupdf.Rect(53, 750, 200, 762)],
        text="1 Note.",
        size=8,
        color="#000",
        bold=False,
    )
    kept = _room_below(out[0], note, 750, 53, 200, [note])
    assert kept > 780


def test_cross_column_host_stops_above_kept_formula(tmp_path):
    """When a paragraph at the column bottom continues in another column, the translation stops above the display
    formula below it that keeps its original glyphs, and the rest goes to the tail block.

    Display formulas are dropped during extraction and are not in the block list; computing the room from blocks
    alone would write down to the bottom of the body text on this page, over the formula.
    """
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((150, 186), "dl/dTheta", fontsize=10)
    doc.save(src)
    doc.close()
    formula_top = pymupdf.open(src)[0].get_text("words")[0][1]

    def block(rect, text):
        rect = pymupdf.Rect(rect)
        lines = [pymupdf.Rect(rect.x0, y, rect.x1, y + 10) for y in range(int(rect.y0), int(rect.y1) - 9, 12)]
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10, color="#000000", bold=False)

    host = block((72, 100, 300, 160), "where x is the training data set and the mini-batch is used to approximate it")
    tail = block((311, 60, 540, 90), "Using mini-batches of examples is helpful in several ways.")
    lower = block((311, 300, 540, 400), "While stochastic gradient is simple and effective, it needs tuning.")
    paragraph = "其中训练数据集按小批量逐步处理，每一步都用小批量来近似损失函数的梯度。" * 6
    out = _render_translated(
        src,
        [host, tail, lower],
        [paragraph, SPILL_TAIL, "随机梯度下降简单而有效，但需要仔细调节。"],
        "zh-CN",
        continuations={id(host): tail},
    )
    chars = [
        pymupdf.Rect(ch["bbox"])
        for b in out[0].get_text("rawdict")["blocks"]
        if b.get("type") == 0
        for line in b["lines"]
        for span in line["spans"]
        for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    left = [rect for rect in chars if rect.x1 < 306]
    assert left and max(rect.y1 for rect in left) < formula_top
    assert any(rect.x0 > 306 and rect.y1 < 100 for rect in chars)
    assert "dl/dTheta" in out[0].get_text()


def test_page_margins_stay_with_the_source(tmp_path):
    """The page top aligns with the top edge of the original, and body text is not written below the bottom edge of
    the original. Footnotes in the footer band still get no extra cut."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated
    from app.formats.pdf_write_geometry import _room_below

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.save(src)
    doc.close()
    top = pymupdf.Rect(53.8, 56.8, 293.0, 90.0)
    bottom = pymupdf.Rect(53.8, 690.0, 293.0, 719.3)
    blocks = [
        TextBlock(
            page=0,
            rect=top,
            line_rects=[top],
            text="During query execution, data can be distributed.",
            size=9,
            color="#000000",
            bold=False,
        ),
        TextBlock(
            page=0,
            rect=bottom,
            line_rects=[bottom],
            text="DXL. Decoupling the optimizer from the database system requires a communication mechanism.",
            size=9,
            color="#000000",
            bold=False,
        ),
    ]
    long = "在查询执行期间，数据可以通过多种方式分布到各个分段节点，包括广播、哈希和复制。" * 6
    out = _render_translated(src, blocks, ["查询执行期间数据会分布到分段节点。", long], "zh-CN")
    chars = [
        ch
        for block in out[0].get_text("rawdict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    assert chars
    assert min(ch["bbox"][1] for ch in chars) <= 57.6
    assert max(ch["bbox"][3] for ch in chars) <= 719.5
    note = TextBlock(
        page=0,
        rect=pymupdf.Rect(53, 750, 200, 762),
        line_rects=[pymupdf.Rect(53, 750, 200, 762)],
        text="1 Note.",
        size=8,
        color="#000",
        bold=False,
    )
    assert _room_below(out[0], note, 750, 53, 200, [note]) > 780


def test_last_short_line_keeps_body_size(tmp_path):
    """The line box slack of the last one or two lines on a page is kept. Without it a Chinese line would not fit, and
    the font size must not become smaller than the body text above because of that."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=612, height=792)
    doc.save(src)
    doc.close()
    above = pymupdf.Rect(72, 640.5, 540, 651.4)
    last = pymupdf.Rect(90, 667.6, 276.9, 678.5)
    heading = pymupdf.Rect(72, 640.0, 280, 652.0)
    pair = pymupdf.Rect(72, 685.5, 540, 710.0)
    blocks = [
        TextBlock(
            page=0,
            rect=above,
            line_rects=[above],
            text="Copying data on every read and write access slows the DBMS.",
            size=10.9,
            color="#000000",
            bold=False,
        ),
        TextBlock(
            page=0,
            rect=last,
            line_rects=[last],
            text="Default libc malloc is slow. Never use it.",
            size=10.9,
            color="#000000",
            bold=False,
        ),
        TextBlock(
            page=1,
            rect=heading,
            line_rects=[heading],
            text="Optimistic Concurrency Control.",
            size=12,
            color="#000000",
            bold=True,
        ),
        TextBlock(
            page=1,
            rect=pair,
            line_rects=[
                pymupdf.Rect(72, 685.5, 540, 696.4),
                pymupdf.Rect(72, 699.1, 265, 710.0),
            ],
            text="Store all changes in a private workspace and merge them at commit time.",
            size=10.9,
            color="#000000",
            bold=False,
        ),
    ]
    out = _render_translated(
        src,
        blocks,
        [
            "每次读和写都会复制数据，因此数据库会变慢。",
            "默认的 libc malloc 速度很慢。永远不要使用它。",
            "乐观并发控制。",
            "将所有更改先放进私有工作区，提交时再检查冲突并合并。这个做法最早在一九八一年提出，后来被广泛使用。",
        ],
        "zh-CN",
    )

    def sizes(page: pymupdf.Page, y0: float) -> list[float]:
        found = [
            span["size"]
            for block in page.get_text("dict")["blocks"]
            if block.get("type") == 0
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip() and span["bbox"][1] >= y0
        ]
        assert found
        return found

    body = 10.9 * 0.88
    assert min(sizes(out[0], 660)) >= body - 0.05
    assert min(sizes(out[1], 670)) >= body - 0.05
    bottoms = [
        ch["bbox"][3]
        for page in out
        for block in page.get_text("rawdict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    assert bottoms
    assert max(bottoms) < 730


def test_cjk_lines_share_the_right_edge(tmp_path):
    """Lines other than the last are right-aligned to the writing box. Punctuation sits against the preceding
    character. The last line stays left-aligned."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=240).insert_text((40, 60), "source paragraph", fontsize=11)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 360, 140)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="source paragraph that differs",
        size=12,
        color="#111111",
        bold=False,
    )
    text = "我们在此提出一种采样机制，在训练过程中随机决定使用真实的前一个词元，还是来自模型本身的估计值。"
    out = _render_translated(src, [block], [text], "zh-CN")
    chars = _cjk_chars(out[0])
    grouped = _group_chars(chars)
    assert len(grouped) >= 2
    right = rect.x1 + 2
    assert abs(grouped[0][-1][2] - right) < 1.5
    assert grouped[-1][-1][2] < right - 8
    for line in grouped[:-1]:
        for prev, nxt in zip(line, line[1:]):
            if nxt[0] == "，":
                assert nxt[1] - prev[2] < 0.8
    assert rect.y0 - 2 < grouped[0][0][3] < rect.y1


def test_em_dash_line_stays_inside_the_box(tmp_path):
    """Dashes and HMM are broken by their real character width, so no extra character ends up past the right edge."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=240).insert_text((40, 60), "source paragraph", fontsize=11)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 280, 160)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="source paragraph that differs",
        size=12,
        color="#111111",
        bold=False,
    )
    text = "标准配置——每一帧都经过 HMM 对齐，破折号不能再多挤出一个字到栏外去。"
    out = _render_translated(src, [block], [text], "zh-CN")
    right = rect.x1 + 2
    for item in out[0].get_text("dict")["blocks"]:
        if item.get("type") != 0:
            continue
        for line in item["lines"]:
            for span in line["spans"]:
                if span["bbox"][0] < 20:
                    continue
                assert span["bbox"][2] <= right + 1.0, span["text"]


def test_italic_marks_still_justify_the_line(tmp_path):
    """Italic marks no longer switch to HTML. Chinese characters have no italic glyphs, and the line stays justified."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=240).insert_text((40, 60), "source paragraph", fontsize=11)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 360, 140)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="source paragraph that differs",
        size=12,
        color="#111111",
        bold=False,
    )
    text = (
        "我们在此提出一种{i}均匀计划采样{/i}机制，在训练过程中随机决定使用真实的前一个词元，还是来自模型本身的估计值。"
    )
    out = _render_translated(src, [block], [text], "zh-CN")
    grouped = _group_chars(_cjk_chars(out[0]))
    assert len(grouped) >= 2
    right = rect.x1 + 2
    assert abs(grouped[0][-1][2] - right) < 1.5
    assert grouped[-1][-1][2] < right - 8


def test_formula_does_not_jump_to_the_line_above(tmp_path):
    """When the previous line reaches the column edge, the formula on the next line still sits against the text of
    its own line. The glyph boxes of the previous line reach into the writing box of this line, so their right edge
    must not be used as the cursor."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=240)
    page.insert_text((80, 80), "t", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    box = d[0].search_for("t")[0]
    d.close()
    formula = {
        "bbox": box,
        "name": "f0_1.png",
        "w": round(box.width, 1),
        "h": round(box.height, 1),
        "d": 2.0,
        "raise": 0.0,
        "text": "t",
        "has_img": True,
        "page": 0,
        "mid": 0,
        "png": pymupdf.open(src)[0].get_pixmap(clip=box, dpi=72).tobytes("png"),
        "spans": [{"bbox": tuple(box), "text": "t", "size": 10, "origin": (box.x0, box.y1 - 2)}],
    }
    rect = pymupdf.Rect(40, 40, 360, 160)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="source paragraph",
        size=11,
        color="#000",
        bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    # A zero-width space makes glyph writing fail, so the HTML box that measures its right edge is used instead.
    text = (
        "我们在此提出一种采样机制，在训练过程中随机决定使用真实的前一个词元还是估计值。"
        f"在时间{sentinel}, 模型需要上一个标记。\u200b"
    )
    out = _render_translated(src, [block], [text], "zh-CN", {id(block): [formula]})
    page = out[0]
    time = page.search_for("在时间")
    painted = page.search_for("t")
    assert time and painted
    assert painted[0].x0 - time[0].x1 < 8
    assert painted[0].x0 < rect.x1 - 40


def test_italic_run_does_not_open_a_wide_gap_before_the_formula(tmp_path):
    """i-th has one space on each side. Estimating italic width at 0.5em would push the following formula away; it
    belongs right after the written text."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((80, 80), "y=", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    y_box = d[0].search_for("y=")[0]
    d.close()
    formula = {
        "bbox": y_box,
        "name": "f0_1.png",
        "w": round(y_box.width, 1),
        "h": round(y_box.height, 1),
        "d": 2.0,
        "raise": 0.0,
        "text": "y=",
        "has_img": True,
        "page": 0,
        "mid": 0,
        "png": pymupdf.open(src)[0].get_pixmap(clip=y_box, dpi=72).tobytes("png"),
        "spans": [{"bbox": tuple(y_box), "text": "y=", "size": 10, "origin": (y_box.x0, y_box.y1 - 2)}],
    }
    rect = pymupdf.Rect(40, 40, 460, 120)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="the i-th token",
        size=11,
        color="#000",
        bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    out = _render_translated(
        src,
        [block],
        [f"第{{i}} i-th{{/i}}个词元{sentinel}。"],
        "zh-CN",
        {id(block): [formula]},
    )
    page = out[0]
    spans = [
        span
        for blk in page.get_text("dict")["blocks"]
        if blk.get("type") == 0
        for ln in blk["lines"]
        for span in ln["spans"]
    ]
    chars = [
        ch
        for blk in page.get_text("rawdict")["blocks"]
        if blk.get("type") == 0
        for ln in blk["lines"]
        for span in ln["spans"]
        for ch in span["chars"]
    ]
    di = next(ch for ch in chars if ch["c"] == "第")
    ith = next(ch for ch in chars if ch["c"] == "i" and ch["bbox"][0] > di["bbox"][2])
    aitch = next(ch for ch in chars if ch["c"] == "h" and ch["bbox"][0] > ith["bbox"][0])
    ge = next(ch for ch in chars if ch["c"] == "个" and ch["bbox"][0] > aitch["bbox"][0])
    assert 1.2 <= ith["bbox"][0] - di["bbox"][2] <= 5
    assert 1.2 <= ge["bbox"][0] - aitch["bbox"][2] <= 5
    italic = next(span for span in spans if "i-th" in span["text"])
    assert "Italic" in italic["font"] or "Oblique" in italic["font"] or int(italic.get("flags") or 0) & 2
    yuan = next(span for span in spans if "元" in span["text"])
    painted = page.search_for("y=")
    assert painted
    assert painted[0].x0 - yuan["bbox"][2] < 5


def test_formula_keeps_a_space_before_following_text(tmp_path):
    """When text follows a formula, a gap one space wide separates them and they do not touch."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write import _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((80, 80), "y=", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    y_box = d[0].search_for("y=")[0]
    d.close()
    formula = {
        "bbox": y_box,
        "name": "f0_1.png",
        "w": round(y_box.width, 1),
        "h": round(y_box.height, 1),
        "d": 2.0,
        "raise": 0.0,
        "text": "y=",
        "has_img": True,
        "page": 0,
        "mid": 0,
        "png": pymupdf.open(src)[0].get_pixmap(clip=y_box, dpi=72).tobytes("png"),
        "spans": [{"bbox": tuple(y_box), "text": "y=", "size": 10, "origin": (y_box.x0, y_box.y1 - 2)}],
    }
    rect = pymupdf.Rect(40, 50, 420, 110)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect, pymupdf.Rect(y_box)],
        text="take the value",
        size=11,
        color="#000",
        bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    out = _render_translated(
        src,
        [block],
        [f"前面{sentinel}后面还有一段说明文字。"],
        "zh-CN",
        {id(block): [formula]},
    )
    page = out[0]
    y_hit = page.search_for("y=")
    after = page.search_for("后")
    assert y_hit and after
    gap = after[0].x0 - y_hit[0].x1
    assert gap >= 2.5
    # Punctuation still touches the formula, with no space added before )
    close = _render_translated(
        src,
        [block],
        [f"取{sentinel})。"],
        "zh-CN",
        {id(block): [formula]},
    )
    paren = close[0].search_for(")")
    painted = close[0].search_for("y=")
    if paren and painted:
        assert paren[0].x0 - painted[0].x1 < 4


def _cjk_chars(page: pymupdf.Page) -> list[tuple[str, float, float, float]]:
    chars: list[tuple[str, float, float, float]] = []
    for blk in page.get_text("rawdict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for span in ln["spans"]:
                for ch in span["chars"]:
                    if "\u4e00" <= ch["c"] <= "\u9fff" or ch["c"] in "，。":
                        chars.append((ch["c"], ch["bbox"][0], ch["bbox"][2], ch["origin"][1]))
    return chars


def _group_chars(
    chars: list[tuple[str, float, float, float]],
) -> list[list[tuple[str, float, float, float]]]:
    grouped: list[list[tuple[str, float, float, float]]] = []
    for item in chars:
        if not grouped or abs(item[3] - grouped[-1][0][3]) > 2:
            grouped.append([item])
        else:
            grouped[-1].append(item)
    return grouped


def test_the_hole_for_a_neighbour_stays_inside_its_font_box():
    """The exact box of a URL on the previous line extends half a line down, and cutting a hole by it would clip the
    tops of the glyphs of the URL on the next line. The hole is clamped to the font box; ink measured by drawing
    single glyphs (when the font box is wrong) is used as before."""
    from app.formats.pdf_place_formulas import _hole_ink

    def box(span):
        return tuple(round(v, 1) for v in _hole_ink(span))

    url = {
        "text": " https://artofproblemsolving.com",
        "bbox": (121.2, 420.0, 525.0, 432.0),
        "ink": (121.2, 420.0, 525.0, 437.5),
    }
    assert box(url) == (121.2, 420.0, 525.0, 432.0)
    arrow = {"text": "←", "bbox": (130.2, 214.0, 139.9, 232.0), "ink": (130.2, 221.3, 139.9, 226.4)}
    assert box(arrow) == (130.2, 221.3, 139.9, 226.4)
    root = {"text": "√", "bbox": (330.5, 187.2, 337.0, 195.0), "ink": (330.5, 187.2, 337.0, 200.0), "ink_drawn": True}
    assert box(root) == (330.5, 187.2, 337.0, 200.0)


def test_formula_box_height_follows_the_glyph_ink():
    """The CMSY font extent is almost two lines high. A box computed from the font would shrink the formula and crop
    glyphs of the lines above and below."""
    from app.formats.pdf_crop import _tighten_formula_height

    arrow = {"text": "←", "ink": (130.2, 221.3, 139.9, 226.4)}
    zero = {"text": "0", "ink": (142.6, 219.8, 147.4, 226.5)}
    part = {"bbox": pymupdf.Rect(114.7, 218.8, 147.4, 237.1), "h": 18.3, "d": 10.8, "spans": [arrow, zero]}
    _tighten_formula_height(part)
    assert [round(v, 1) for v in part["bbox"]] == [114.7, 219.2, 147.4, 227.1]
    assert part["h"] == 7.9 and part["d"] == 0.8
    # The minus sign lies entirely above the baseline (math axis); after tightening its bottom edge is still above
    # the baseline and must not snap to it
    minus = {"text": "−", "ink": (197.2, 713.4, 205.0, 715.0)}
    part = {"bbox": pymupdf.Rect(197.2, 708.9, 205.0, 726.2), "h": 17.3, "d": 9.6, "spans": [minus]}
    _tighten_formula_height(part)
    baseline = 726.2 - 9.6
    assert part["d"] < 0 and abs(part["bbox"].y1 - part["d"] - baseline) < 0.1
    # Bitmap glyphs have no ink box, so the original box is used as before
    unknown = {"bbox": pymupdf.Rect(0, 0, 10, 20), "h": 20, "d": 5, "spans": [{"text": "x"}]}
    _tighten_formula_height(unknown)
    assert unknown["h"] == 20


def test_fraction_bars_drawn_as_one_pixel_images_are_formula_curves():
    """Old TeX draws fraction bars and radical bars as stretched one-pixel images. They count as formula curves like
    vector lines, so the line at the original position is erased when the formula moves; illustrations do not
    count."""
    from app.formats.pdf_crop import _page_curve_rects

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    dot = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 1, 1), False)
    dot.clear_with(0)
    bar = pymupdf.Rect(357.6, 598.8, 394.1, 599.3)
    page.insert_image(bar, pixmap=dot, keep_proportion=False)
    photo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    photo.clear_with(128)
    page.insert_image(pymupdf.Rect(100, 100, 300, 300), pixmap=photo)
    # A stroked horizontal line: the path box has zero height, but the drawn line is as thick as the line width
    page.draw_line((234.7, 263.1), (259.9, 263.1), width=0.4)
    rects = _page_curve_rects(page)
    assert any(abs(r.x0 - bar.x0) < 0.1 and abs(r.y1 - bar.y1) < 0.1 for r in rects)
    assert not any(r.width > 150 and r.height > 150 for r in rects)
    stroke = next(r for r in rects if abs(r.x1 - 259.9) < 0.5)
    assert stroke.y0 < 263.1 < stroke.y1 and stroke.height >= 0.39

    from app.formats.pdf_runs import intersecting_curves

    assert intersecting_curves(pymupdf.Rect(234.7, 255.9, 295.8, 276.6), rects)


def test_a_formula_only_takes_the_rules_its_glyphs_touch():
    """The font box of |m/√v| < 1 reaches into the next line and covers the radical bar of √E[g²] there. Judged by ink,
    the bar belongs to the formula on the next line."""
    from app.formats.pdf_crop import _absorb_one_box

    def formula(bbox, inks):
        spans = [{"text": "x", "ink": ink} for ink in inks]
        return {"bbox": pymupdf.Rect(bbox), "d": 3.0, "spans": spans}

    above = formula(
        (334.3, 131.1, 395.7, 175.0),
        [(334.3, 133.6, 364.0, 145.2), (364.7, 133.9, 373.2, 145.6), (380.0, 135.9, 395.7, 141.2)],
    )
    below = formula(
        (355.1, 143.0, 433.7, 179.1),
        [(355.1, 145.6, 390.0, 157.7), (391.5, 150.6, 413.0, 158.4), (414.0, 148.9, 433.7, 155.1)],
    )
    own_radical = pymupdf.Rect(364.7, 138.2, 373.2, 138.6)
    next_radical = pymupdf.Rect(391.5, 150.0, 413.0, 150.4)
    _absorb_one_box(above, [own_radical, next_radical])
    _absorb_one_box(below, [own_radical, next_radical])
    assert above["curves"] == [own_radical]
    assert below["curves"] == [next_radical]
    assert abs(above["bbox"].x1 - 395.7) < 0.01


def test_a_fraction_bar_stays_when_its_numerator_stays():
    """When the numerator is in another block that stays in place, the fraction bar stays too: moved with the
    denominator, it would leave the numerator hanging in the air."""
    from app.formats.pdf_crop import _absorb_one_box

    denominator = {"text": "1−α", "origin": (164.2, 500.0), "ink": (164.2, 496.4, 187.5, 502.0)}
    numerator = {"text": "1−α", "origin": (159.7, 492.5), "ink": (159.7, 487.4, 187.5, 494.2)}
    bar = pymupdf.Rect(159.7, 494.9, 187.5, 495.3)
    alone = {"bbox": pymupdf.Rect(164.2, 490.1, 197.2, 507.6), "d": 3.0, "spans": [denominator]}
    _absorb_one_box(alone, [bar], [numerator, denominator])
    assert alone["curves"] == []
    whole = {"bbox": pymupdf.Rect(159.7, 487.0, 197.2, 507.6), "d": 3.0, "spans": [numerator, denominator]}
    _absorb_one_box(whole, [bar], [numerator, denominator])
    assert whole["curves"] == [bar]


def test_a_formula_leaves_the_edge_of_a_proof_box():
    """An end-of-proof box sits right below the formula at the line end. The top edge of the box meets its two vertical
    sides, so it is a side of the box and does not move with the formula."""
    from app.formats.pdf_crop import _absorb_one_box

    g = {"text": "g", "origin": (493.0, 571.0), "ink": (493.0, 566.5, 496.9, 574.6)}
    paren = {"text": ")", "origin": (497.6, 569.5), "ink": (497.6, 562.5, 501.5, 572.0)}
    top = pymupdf.Rect(496.9, 575.2, 503.3, 575.6)
    sides = [
        pymupdf.Rect(496.7, 575.0, 497.1, 582.1),
        pymupdf.Rect(503.1, 575.0, 503.5, 582.1),
        pymupdf.Rect(496.9, 581.5, 503.3, 581.9),
    ]
    formula = {"bbox": pymupdf.Rect(461.7, 561.9, 501.5, 573.0), "d": 3.0, "spans": [g, paren]}
    _absorb_one_box(formula, [top, *sides], [g, paren])
    assert formula["curves"] == []
    bar = pymupdf.Rect(493.0, 575.2, 501.5, 575.6)
    lone = {"bbox": pymupdf.Rect(461.7, 561.9, 501.5, 573.0), "d": 3.0, "spans": [g, paren]}
    _absorb_one_box(lone, [bar], [g, paren])
    assert lone["curves"] == [bar]


def test_formula_holes_follow_the_neighbours_ink_not_their_font_box():
    """The font box of a large brace on the previous line reaches the N on the next line, but its ink does not. Cutting
    holes by ink leaves the corner of N intact."""
    from app.formats.pdf_place_formulas import _formula_form
    from app.formats.pdf_write_geometry import _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    fms = _page_form(doc, page)
    base = {"text": "N", "bbox": (327.5, 456.4, 336.9, 469.7)}
    formula = {"bbox": pymupdf.Rect(base["bbox"]), "spans": [base]}
    brace = {"text": "{", "bbox": (328.0, 445.3, 332.9, 462.3)}

    def hole_bottom(neighbour):
        """Bottom edge of the hole (page coordinates). The first re is the clip box of the formula itself, the following
        ones are holes."""
        xref = _formula_form(doc, page, fms, formula, "/Fx", [dict(base), neighbour])
        rects = re.findall(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", doc.xref_stream(xref).decode())
        [(_, y, _, _)] = rects[1:]
        return page.rect.height - float(y)

    # The ink of N starts at 460.3
    assert hole_bottom(brace) > 460.3
    assert hole_bottom({**brace, "ink": (328.0, 445.6, 332.9, 457.4)}) <= 457.5


def test_a_formula_cut_from_a_text_span_is_not_holed_by_that_span():
    """The formula detection model cuts the σ out of " σ" as a formula. The page still has the original " σ" span,
    whose box does not match the σ of the formula; when the formula has a superscript and is taller than a line,
    cutting a hole by that box would remove the σ (leaving only the superscript in the translation). The original
    span keeps only the glyphs left after the cut, and the space has no ink, so no hole is cut. Glyphs reaching
    down from the line above still get holes"""
    from app.formats.pdf_place_formulas import _formula_form
    from app.formats.pdf_write_geometry import _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    fms = _page_form(doc, page)
    whole = (269.5, 213.2, 277.4, 222.2)
    sigma = {"text": "σ", "bbox": (272.0, 213.2, 277.4, 222.2), "source": whole}
    square = {"text": "2", "bbox": (277.6, 206.0, 280.0, 214.0)}
    formula = {"bbox": pymupdf.Rect(272.0, 206.0, 280.0, 222.2), "spans": [sigma, square]}
    source = {
        "text": " σ",
        "bbox": whole,
        "chars": [(" ", pymupdf.Rect(269.5, 213.2, 272.0, 222.2)), ("σ", pymupdf.Rect(272.0, 213.2, 277.4, 222.2))],
    }
    above = {"text": "p", "bbox": (273.0, 196.0, 276.0, 205.5)}

    def clip_rects(page_spans):
        xref = _formula_form(doc, page, fms, formula, "/Fx", page_spans)
        return re.findall(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", doc.xref_stream(xref).decode())

    assert len(clip_rects([dict(source), dict(square)])) == 1
    assert len(clip_rects([dict(source), dict(square), above])) == 2


def test_list_item_near_the_page_middle_stays_left_aligned():
    """A list item whose center line happens to match the center line of the widest block is still left-aligned and
    not written back centered. The (a) subcaption below a subfigure stays centered."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, x0, y0, x1):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 23.7)
        return TextBlock(page=17, rect=rect, line_rects=[rect], text=text, size=20.4, color="#000", bold=False)

    widest = blk("• Apache top-level project since October, 2015", 35.2, 101.8, 477.8)
    sibling = blk("– Optional SQL parser, JDBC server", 71.2, 317.8, 400.0)
    item = blk("– Community-authored rules, adapters", 71.2, 341.8, 431.0)
    rect, align = _write_rect(item, [(widest, ""), (sibling, ""), (item, "")], True, False, None)
    assert align == "left"
    assert rect.x0 == item.rect.x0

    caption = blk("(a) Pixel space nearest neighbors", 197.0, 367.0, 317.0)
    _, align = _write_rect(caption, [(widest, ""), (sibling, ""), (caption, "")], True, False, None)
    assert align == "center"


def test_heading_at_the_frame_edge_stays_left_aligned():
    """The 2.6.1 subheading starts at the left edge of the text area. The widest block is a paragraph merged with a
    side note and reaching further left; their center lines happen to match, but it is still written back
    left-aligned."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, x0, y0, x1, y1, size=10.0):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=16, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    heading = blk("2.6.1 Character Disjunction: The Square Bracket", 160, 173, 421, 185, 12)
    widest = blk("Figure 2.10 The caret for negation or just to mean the caret.", 96, 695, 490, 705, 9)
    title = blk("Exercises", 280, 300, 373, 312, 12)
    frame = TextFrame(160, 493, 111, 719, 10)
    geo = PageGeometry.from_rects(612, [heading.rect, widest.rect], height=792, frame=frame)
    items = [(heading, ""), (widest, ""), (title, "")]
    rect, align = _write_rect(heading, items, True, False, geo)
    assert align == "left" and rect.x0 == heading.rect.x0
    _, align = _write_rect(title, items, True, False, geo)
    assert align == "center"


def test_epigraph_and_attribution_at_the_frame_right_edge_are_written_flush_right():
    """Every line of an epigraph and its attribution at a chapter start ends at the right edge of the text area, far
    from the left edge: they are written back right-aligned, and the writing box does not extend to the right.
    When another block lies to the left on the same line, the right block is only the second half of a line and is
    left-aligned as usual."""
    from app.formats.pdf_align import _place_line
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_typeset import typeset_lines
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, rows, size=10.0):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=128, rect=rect, line_rects=rects, text=text, size=size, color="#000", bold=False)

    quote = blk(
        "Numquam ponenda est pluralitas sine necessitate ‘Plurality should never be proposed unless needed’",
        [(296.0, 545.3, 493.2, 557.2), (290.0, 557.2, 493.2, 569.2)],
    )
    author = blk("William of Occam", [(426.8, 570.1, 493.2, 580.9)], 9.0)
    body = blk(
        "There is a problem with learning weights that make the model perfectly match the training data.",
        [(160.2, 611.8, 493.2, 623.8)],
    )
    frame = TextFrame(160.2, 493.2, 100, 720, 10)
    geo = PageGeometry.from_rects(612, [quote.rect, author.rect, body.rect], height=792, frame=frame)
    items = [(quote, ""), (author, ""), (body, "")]
    for block in (quote, author):
        rect, align = _write_rect(block, items, True, False, geo)
        assert align == "right" and rect.x1 == block.rect.x1
    assert _write_rect(body, items, True, False, geo)[1] == "left"
    # Another block lies to the left on the same line: this block is the second half of a line
    left = blk("The first half of the row", [(160.2, 700, 280, 712)])
    tail = blk("and the second half", [(300, 700, 493.2, 712)])
    assert _write_rect(tail, [*items, (left, ""), (tail, "")], True, False, geo)[1] == "left"
    # The figure is on the previous page; the body text around it leaves the left side empty, and its lines start at
    # the same x
    wrap = blk(
        "Probably the most common visualization method, however, is to project the 100 dimensions",
        [(270, 218, 493.2, 230), (255, 230, 493.2, 242), (255, 242, 493.2, 254)],
    )
    assert _write_rect(wrap, [*items, (wrap, "")], True, False, geo)[1] == "left"
    # Only this block reaches the right edge of the column, so the block itself defines that edge
    lone = blk("1 request", [(300, 300, 493.2, 312)])
    short = blk("Count(w | denied the)", [(160.2, 280, 400, 292)])
    lone_geo = PageGeometry.from_rects(612, [lone.rect, short.rect], height=792, frame=frame)
    assert _write_rect(lone, [(short, ""), (lone, "")], True, False, lone_geo)[1] == "left"

    def start(text):
        line = typeset_lines(text, 9.0, 200.0, {})[0]
        ops = _place_line(line, 100.0, 200.0, 9.0, False, "right", frozenset(), {})[0]
        return float(next(op for op in ops if op.endswith("Tm")).split()[4]), line.width

    x, width = start("奥卡姆的威廉")
    assert abs(x + width - 300.0) < 0.5
    # A space left at the line end by wrapping is invisible and does not count toward the right-aligned position
    assert start("William of Occam ")[0] == start("William of Occam")[0]


def test_attribution_short_of_the_frame_edge_by_a_trailing_space_is_written_flush_right():
    """An attribution with a trailing word space ends two or three points short of the right edge of the text area and
    is still written right-aligned. So is an attribution pushed to the end of the last epigraph line with \\hfill.
    The last word split off a line in a justified narrow column, and a table header aligned with the start of the
    cells below, are left-aligned as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, rows, size=10.0):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=75, rect=rect, line_rects=rects, text=text, size=size, color="#000", bold=False)

    frame = TextFrame(142.0, 475.0, 100, 720, 10)

    def align(block, others):
        items = [(other, "") for other in [*others, block]]
        geo = PageGeometry.from_rects(612, [other.rect for other, _ in items], height=792, frame=frame)
        return _write_rect(block, items, True, False, geo)[1]

    quote = blk(
        "You are uniformly charming! cried he, with a smile of associating and now and then I bowed",
        [(167.1, 218, 475.2, 230), (167.1, 230, 431.4, 242)],
    )
    author = blk("Random sentence generated from a Jane Austen trigram model", [(246.8, 243, 472.4, 252)], 9.0)
    body = blk(
        "Predicting is difficult, especially about the future, as the old quip goes.",
        [(142.2, 280, 475.2, 292), (142.2, 292, 475.2, 304)],
    )
    assert align(author, [quote, body]) == "right"
    # The attribution shares a line with the last epigraph line
    babbage = blk(
        "On two occasions I have been asked, Pray, Mr. Babbage, if you put into the machine wrong figures",
        [(167.1, 210, 475.2, 222), (167.1, 222, 203.4, 234)],
    )
    signed = blk("Babbage (1864)", [(411.3, 222, 475.2, 234)])
    assert align(signed, [babbage, body]) == "right"
    # The last word split off a justified line, with a stretched word space to its left
    para = blk("states yields a finite-transition increment", [(142.2, 300, 475.2, 312), (142.2, 312, 300.0, 324)])
    word = blk("comprising", [(316.5, 312, 472.6, 324)])
    assert align(word, [para, body]) == "left"
    # The left edge of the table header aligns with the start of the cells below
    header = blk("Emotion Label Emotion Explanation", [(300.0, 330, 472.6, 340)])
    cells = blk("A V AV Semantic", [(300.8, 344, 320.0, 354), (340.0, 344, 400.0, 354)])
    assert align(header, [cells, body]) == "left"


def test_two_line_attribution_is_split_like_the_source_and_written_flush_right(tmp_path):
    """An epigraph and attribution on two lines, with a short first line starting toward the right: the translation
    breaks at the width of the first line, and both lines end at the original right edge."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write import _render_translated

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    right = 475.0
    # The start of each line is measured back from the right edge
    for text, y in (("Time will explain.", 230), ("Jane Austen, Persuasion", 240)):
        page.insert_text((right - pymupdf.get_text_length(text, "tiro", 9), y), text, fontsize=9, fontname="tiro")
    body_line = "x" * 55
    for index in range(3):
        page.insert_text(
            (right - pymupdf.get_text_length(body_line, "cour", 10), 300 + index * 12),
            body_line,
            fontsize=10,
            fontname="cour",
        )
    path = tmp_path / "src.pdf"
    doc.save(path)
    page = pymupdf.open(path)[0]

    def blk(text, rows, size):
        rects = [page.search_for(row)[0] for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=0, rect=rect, line_rects=rects, text=text, size=size, color="#000000", bold=False)

    author = blk("Time will explain. Jane Austen, Persuasion", ["Time will explain.", "Jane Austen, Persuasion"], 9.0)
    body = TextBlock(
        page=0,
        rect=pymupdf.Rect(page.search_for(body_line)[0]) | page.search_for(body_line)[-1],
        line_rects=page.search_for(body_line),
        text=body_line * 3,
        size=10.0,
        color="#000000",
        bold=False,
    )
    frame = TextFrame(body.rect.x0, body.rect.x1, 100, 720, 10)
    geo = PageGeometry.from_rects(612, [author.rect, body.rect], height=792, frame=frame)
    out = _render_translated(
        path, [author, body], ["时间会说明一切。简·奥斯汀，《劝导》", "正文" * 40], "zh-CN", geometries=[geo]
    )
    lines = [
        (line["bbox"], "".join(span["text"] for span in line["spans"]))
        for block in out[0].get_text("dict", clip=pymupdf.Rect(300, 215, 500, 250))["blocks"]
        for line in block.get("lines", [])
    ]
    assert [text for _, text in lines] == ["时间会说明一切。", "简·奥斯汀，《劝导》"]
    assert all(abs(bbox[2] - author.rect.x1) < 0.5 for bbox, _ in lines)


def test_header_cells_near_the_column_middle_are_not_all_centred_on_it():
    """The cells of a header row all have center lines that happen to lie near the column center line, but the row
    is not centered in the column: each is written at its own position and they are not all centered on the
    column center line on top of each other. In a heading split into a number block and a name block, the whole
    line is on the column center line, so the name is still centered."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, x0, x1, y0=676.0, size=10.0):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 11.6)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=True)

    body = TextBlock(
        page=0,
        rect=pymupdf.Rect(160.2, 600, 493.2, 660),
        line_rects=[pymupdf.Rect(160.2, 600, 493.2, 612)],
        text="Let us see how the cosine computes which of the words is closer",
        size=10.0,
        color="#000",
        bold=False,
    )
    cells = [blk("pie", 313.4, 324.9), blk("data", 332.3, 349.3), blk("computer", 355.7, 392.6)]
    frame = TextFrame(160.0, 493.0, 100, 720, 10)
    items = [(body, ""), *((cell, "") for cell in cells)]
    geo = PageGeometry.from_rects(612, [block.rect for block, _ in items], height=792, frame=frame)
    assert [_write_rect(cell, items, True, False, geo)[1] for cell in cells] == ["left", "left", "left"]
    # The section number is a separate block, and the whole line of number and name is on the column center line
    number, title = blk("V.", 290.0, 302.0, y0=400), blk("Conclusions", 306.0, 362.0, y0=400)
    items = [(body, ""), (number, ""), (title, "")]
    assert _write_rect(title, items, True, False, geo)[1] == "center"


def test_a_paragraph_with_an_overfull_line_is_written_to_the_frame_edge():
    """For a paragraph with an unbreakable line reaching into the page margin, the translation is written to the right
    edge of the text area, not to that protruding line. A block mostly outside the text area is still written at
    its own width."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_write_geometry import _write_limit
    from app.formats.pdf_write_rect import _write_rect

    def blk(rows):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(
            page=75,
            rect=rect,
            line_rects=rects,
            text="This probabilistic knowledge " * 8,
            size=10.0,
            color="#000",
            bold=False,
        )

    para = blk(
        [
            (157.1, 600, 475.2, 612),
            (142.2, 612, 475.2, 624),
            (142.2, 624, 489.5, 636),
            (142.2, 636, 475.2, 648),
            (142.2, 648, 262.0, 660),
        ]
    )
    wide = blk([(142.2, 400, 520.0, 412), (142.2, 412, 520.0, 424), (142.2, 424, 300.0, 436)])
    frame = TextFrame(142.0, 475.0, 100, 720, 10)
    geo = PageGeometry.from_rects(612, [para.rect, wide.rect], height=792, frame=frame)
    items = [(para, ""), (wide, "")]
    rect, align = _write_rect(para, items, True, False, geo)
    assert align == "left" and rect.x1 == 477.0
    assert _write_rect(wide, items, True, False, geo)[0].x1 == 522.0
    # When the translation does not fit and widens to the right, it also widens only to the right edge of the text
    # area; the right edge of the column was pushed out by the protruding line
    assert geo.columns[-1].x1 > 500
    assert _write_limit(para, rect, geo, []) == 477.0
    assert _write_limit(wide, rect, geo, []) > 500
    # A two-line paragraph with one protruding line. When side notes form a column of their own, this line pushes
    # out the right edge of the body column, and the text is still written to the right edge of the text area
    pair = blk([(157.1, 300, 489.5, 312), (142.2, 312, 475.2, 324)])
    geo = PageGeometry(612, (Column(79.4, 127.8), Column(142.2, 489.5)), (), 792, frame=frame)
    assert _write_rect(pair, [(pair, ""), (para, "")], True, False, geo)[0].x1 == 477.0


def test_two_line_title_centered_on_the_page_is_written_centered():
    """A slide title is in the middle of the page. The widest block is a list item toward the left whose center line is
    off to the right; the title is still written back centered on the page."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, x0, y0, x1, y1, size):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=29, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    title = blk("Beyond Phoenix 4.8 with Apache Calcite", 202, 18, 518, 100, 35.5)
    widest = blk("– To allow other forms of indices (maybe defined as external), e.g., a", 68, 210, 709, 256, 20)
    item = blk("• Get the missing SQL support", 31, 119, 367, 147, 24)
    geo = PageGeometry.from_rects(720, [title.rect, widest.rect, item.rect], height=405)
    rect, align = _write_rect(title, [(title, ""), (widest, ""), (item, "")], True, False, geo)
    assert align == "center"
    assert abs((rect.x0 + rect.x1) / 2 - 360) < 1

    widest_title = blk("Calcite Planning Process", 122, 25, 599, 76, 44)
    labels = [
        blk("Planner", 198, 104, 235, 115, 10),
        blk("Logical Plan", 97, 213, 151, 224, 10),
        blk("Best RelNode Graph", 451, 356, 543, 367, 10),
    ]
    geo = PageGeometry.from_rects(720, [widest_title.rect, *(label.rect for label in labels)], height=405)
    items = [(widest_title, ""), *((label, "") for label in labels)]
    _, align = _write_rect(widest_title, items, True, False, geo)
    assert align == "center"


def test_title_page_centered_off_the_page_middle_is_written_centered():
    """The lines of a title page share one center line, which is neither the middle of the page (asymmetric left and
    right margins) nor the center line of the text area of the book. Every block is written back centered on this
    line, the widest title too."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, x0, y0, x1, y1, size):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("Speech and Language Processing", 108.2, 112.9, 494.1, 147.7, 28.9),
        blk("with Language Models", 227.7, 198.4, 374.6, 217.6, 15.9),
        blk("Third Edition draft", 236.1, 242.2, 366.2, 263.0, 17.2),
        blk("James H. Martin", 244.0, 395.4, 358.3, 416.1, 17.2),
        blk("University of Colorado at Boulder", 202.5, 416.0, 399.8, 433.1, 14.3),
        blk("Draft of August 19, 2026. Comments and typos welcome!", 100.2, 644.4, 501.0, 665.2, 17.2),
    ]
    geo = PageGeometry.from_rects(612, [b.rect for b in blocks], height=792, frame=TextFrame(203, 493, 110, 719, 10))
    items = [(b, "") for b in blocks]
    for block in blocks:
        rect, align = _write_rect(block, items, True, False, geo)
        assert align == "center", block.text
        assert abs((rect.x0 + rect.x1) / 2 - 301) < 1, block.text


def test_body_lines_beside_margin_notes_do_not_make_a_centered_page():
    """Side notes in the margin widen the horizontal extent of the text on the page, but full body lines must still not
    count as short lines. The body lines are all on the center line of the text area, and an indented example whose
    center line happens to lie near it does not make the page centered."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_write_rect import _centered_axis

    def blk(text, x0, y0, x1, size=10.0):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    body = [
        blk("Words don't appear out of nowhere. Any particular piece of text", 160, 253 + 12 * i, 493) for i in range(6)
    ]
    note = blk("code switching", 101, 530, 146, size=8)
    example = blk("[he was and will remain a friend ... don't worry ... but have faith]", 192, 619, 465)
    title = blk("Corpora", 134, 213, 191, size=14)
    blocks = [*body, note, example, title]
    assert _centered_axis(example, blocks) is None


def test_left_aligned_dialogue_lines_are_not_centered():
    """The lines of a dialogue share a left edge. Lines whose center line happens to fall near the center line of the
    text area are also written back left-aligned."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_write_rect import _write_rect

    def blk(text, y0, x1):
        rect = pymupdf.Rect(221, y0, x1, y0 + 9)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=9, color="#000", bold=False)

    first = blk("User: I need some help, that much seems certain.", 189, 453)
    second = blk("ELIZA: WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP", 200, 471)
    geo = PageGeometry.from_rects(612, [first.rect, second.rect], height=792, frame=TextFrame(160, 493, 111, 719, 10))
    rect, align = _write_rect(first, [(first, ""), (second, "")], True, False, geo)
    assert align == "left" and rect.x0 == first.rect.x0


def test_light_cell_shading_is_not_a_formula_stroke():
    """A light cell background under the text is not merged into the formula box; the black background of a caption
    label and fraction bars are still taken in."""
    from app.formats.pdf_crop import _page_curve_rects

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 35), color=None, fill=(0.77, 0.86, 0.99))
    page.draw_rect(pymupdf.Rect(80, 20, 120, 35), color=None, fill=(0, 0, 0))
    page.draw_rect(pymupdf.Rect(140, 30, 180, 30.6), color=None, fill=(0, 0, 0))
    rects = _page_curve_rects(page)
    assert [round(r.x0) for r in rects] == [80, 140]


def test_a_radical_bar_moves_with_its_formula_under_the_line_above():
    """A descender of the line above touches the top of a radical bar; the left end of the bar meets the upper right
    corner of the radical of this formula, so the bar still goes with the formula."""
    from app.formats.pdf_crop import _absorb_one_box

    radical = {"text": "√", "origin": (339.8, 625.9), "ink": (339.8, 618.0, 348.1, 628.0)}
    alpha = {"text": "α", "origin": (348.1, 625.5), "ink": (348.1, 619.6, 354.5, 625.7)}
    bar = pymupdf.Rect(347.9, 618.0, 358.2, 618.4)
    above = {"text": "p", "origin": (349.0, 613.5), "ink": (349.0, 608.8, 354.0, 616.6)}
    formula = {"bbox": pymupdf.Rect(339.8, 617.0, 358.2, 628.6), "d": 3.0, "spans": [radical, alpha]}
    _absorb_one_box(formula, [bar], [radical, alpha, above])
    assert formula["curves"] == [bar]


def test_a_formula_clip_reaches_into_its_own_gaps_beside_a_comma():
    """The strokes of 𝒩 reach into the gap between it and (. When a comma follows the formula directly, only the
    rightmost rectangle must make room for the comma; the middle rectangles widen as usual, so the gap is not
    cut off."""
    from app.formats.pdf_place_formulas import _formula_form
    from app.formats.pdf_write_geometry import _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    fms = _page_form(doc, page)
    spans = [
        {"text": "x", "bbox": (407.1, 490.1, 413.0, 500.1)},
        {"text": "∼N", "bbox": (424.5, 489.9, 444.3, 498.1)},
        {"text": "(", "bbox": (445.8, 490.1, 449.7, 500.1)},
        {"text": ")", "bbox": (464.2, 490.1, 468.0, 500.1)},
    ]
    formula = {"bbox": pymupdf.Rect(407.1, 489.9, 468.0, 500.1), "spans": spans}
    # The for before and the comma after both sit right against the formula
    word = {"text": "for", "bbox": (395.2, 489.9, 407.0, 500.0)}
    comma = {"text": ",", "bbox": (468.0, 489.9, 470.6, 500.0)}
    xref = _formula_form(doc, page, fms, formula, "/Fx", [dict(span) for span in spans] + [word, comma])
    rects = [
        tuple(map(float, found))
        for found in re.findall(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", doc.xref_stream(xref).decode())
    ]

    def inside(x, y):
        y = page.rect.height - y
        return sum(x0 <= x <= x0 + w and y0 <= y <= y0 + h for x0, y0, w, h in rects) % 2 == 1

    assert inside(445.0, 491.0)
    assert not inside(469.0, 495.0)


def test_a_long_index_entry_takes_another_line_and_pushes_the_column_down():
    """An index column is full to the bottom edge. An entry whose translation needs two lines takes one extra line,
    the entries below it (including those keeping the original text) move down, the line spacing of the column
    tightens a little, and the last entry stays within the bottom edge; other columns do not move. With too many
    extra lines nothing moves, as before."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_index_reflow import _index_reflow
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, row):
        rect = pymupdf.Rect(x0, 100 + row * 9.6, x0 + 60, 110 + row * 9.6)
        span = {"text": text, "size": 8.0, "font": "Times-Roman", "flags": 0, "color": 0, "bbox": tuple(rect)}
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=8.0,
            color="#000",
            bold=False,
            span_lines=[[span]],
            entry=True,
            index=True,
        )

    left = [blk(f"entry {n}, {n}", 80, n) for n in range(10)]
    right = [blk(f"other {n}, {n}", 180, n) for n in range(10)]
    geo = PageGeometry(612, (Column(80, 280),), ())
    items = [(b, f"条目{n}, {n}") for n, b in enumerate(left) if n != 6] + [(b, "其他") for b in right]
    items[2] = (left[2], "在命名实体识别(NER)中, 241, 419")
    bottom = left[-1].rect.y1
    moved = _index_reflow(items, [*left, *right], geo, True, {}, set(), bottom)
    assert moved[id(left[2])].rect.height > left[2].rect.height + 8
    assert id(left[6]) in moved and moved[id(left[6])].rect.y0 > left[6].rect.y0
    assert moved[id(left[9])].rect.y1 <= bottom + 1
    assert not any(id(b) in moved for b in right)
    crowded = [(b, "在命名实体识别(NER)中, 241, 419") for b in left]
    assert _index_reflow(crowded, left, geo, True, {}, set(), bottom) == {}
    # Entry 5 was lost during extraction and its original glyphs stay in place. The four entries above cannot fit
    # the extra line, so they do not move and do not cover the original glyphs
    dropped = left[4].rect
    rest = [b for n, b in enumerate(left) if n != 4]
    moved = _index_reflow(
        [item for item in items if item[0] is not left[4]], rest, geo, True, {}, set(), bottom, [dropped]
    )
    assert not any(id(b) in moved for b in rest)


def test_an_index_entry_without_spans_that_keeps_its_text_still_moves():
    """An entry merged from fragments of which one had no spans has no span lines. When it keeps the original
    text and the entries above it push the column down, it moves with them, rewritten from its plain text."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_index_reflow import _index_reflow
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, row, spans=True):
        rect = pymupdf.Rect(x0, 100 + row * 9.6, x0 + 60, 110 + row * 9.6)
        span = {"text": text, "size": 8.0, "font": "Times-Roman", "flags": 0, "color": 0, "bbox": tuple(rect)}
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=8.0,
            color="#000",
            bold=False,
            span_lines=[[span]] if spans else None,
            entry=True,
            index=True,
        )

    left = [blk(f"entry {n}, {n}", 80, n, spans=n != 6) for n in range(10)]
    right = [blk(f"other {n}, {n}", 180, n) for n in range(10)]
    geo = PageGeometry(612, (Column(80, 280),), ())
    items = [(b, f"条目{n}, {n}") for n, b in enumerate(left) if n != 6] + [(b, "其他") for b in right]
    items[2] = (left[2], "在命名实体识别(NER)中, 241, 419")
    moved = _index_reflow(items, [*left, *right], geo, True, {}, set(), left[-1].rect.y1)
    assert moved[id(left[6])].rect.y0 > left[6].rect.y0


def test_an_index_entry_is_never_centered():
    """An indented subentry that falls exactly in the middle of the column is still left-aligned."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_write_rect import _write_rect

    geo = PageGeometry(612, (Column(80, 280), Column(300, 500)), ())
    rect = pymupdf.Rect(150, 100, 210, 110)
    plain = TextBlock(page=0, rect=rect, line_rects=[rect], text="in NER, 241, 419", size=8.0, color="#000", bold=False)
    entry = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="in NER, 241, 419",
        size=8.0,
        color="#000",
        bold=False,
        entry=True,
        index=True,
    )
    assert _write_rect(plain, [(plain, "")], True, False, geo)[1] == "center"
    assert _write_rect(entry, [(entry, "")], True, False, geo)[1] == "left"


def test_inline_formulas_keep_their_size_next_to_chinese():
    """The Chinese font size is 10 percent smaller than the original and looks the same size. Formulas are scaled
    relative to the Chinese size, so without shrinking they keep their original size instead of becoming 10 percent
    smaller; when the whole paragraph shrinks, formulas shrink with it."""
    from app.formats.pdf_place import _formula_slot

    formula = {"has_img": True, "w": 40.0, "h": 8.0, "d": 2.0, "text": "f(x)"}
    scale, _, _ = _formula_slot(formula, 10.0, 8.8, 8.8)
    assert abs(scale - 1.0) < 1e-6
    scale, _, _ = _formula_slot(formula, 10.0, 8.8 * 0.9, 8.8)
    assert abs(scale - 0.9) < 1e-6
    # Without reference, the ratio uses the original font size as before
    scale, _, _ = _formula_slot(formula, 10.0, 8.8)
    assert abs(scale - 0.88) < 1e-6


def test_a_formula_form_box_keeps_the_overhang_of_an_italic_letter():
    """The hook and tail of an italic f reach past its character width. The Form BBox clips too, so horizontally it
    widens to the ink measured from the glyphs and vertically it follows the formula box. It widens only to its
    own ink: a does not reach past its width, and the hook of the f of the preceding If reaching into the gap is
    not included."""
    from app.formats.pdf_blocks import _GlyphInk
    from app.formats.pdf_place_formulas import _formula_form
    from app.formats.pdf_write_geometry import _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_font(fontname="ti", fontbuffer=pymupdf.Font("tiit").buffer)
    page.insert_text((350, 526), "If f is a", fontname="ti", fontsize=10)
    spans = {}
    for char in page.get_text("rawdict")["blocks"][0]["lines"][0]["spans"][0]["chars"]:
        if char["c"] in "fa":
            spans.setdefault(char["c"], []).append({"text": char["c"], "bbox": char["bbox"], "origin": char["origin"]})
    fms = _page_form(doc, page)
    glyphs = _GlyphInk(page)

    def box(letter):
        formula = {"bbox": pymupdf.Rect(letter["bbox"]), "spans": [letter]}
        xref = _formula_form(doc, page, fms, formula, "/Fx", [dict(letter)], glyphs)
        x0, y0, x1, y1 = map(float, doc.xref_get_key(xref, "BBox")[1].strip("[]").split())
        return x0, 792 - y1, x1, 792 - y0

    letter = spans["f"][1]
    x0, y0, x1, y1 = box(letter)
    assert x0 < letter["bbox"][0] - 0.5 and x1 > letter["bbox"][2] + 0.5
    assert (round(y0, 1), round(y1, 1)) == (round(letter["bbox"][1], 1), round(letter["bbox"][3], 1))
    a = spans["a"][0]
    x0, _y0, x1, _y1 = box(a)
    assert abs(x0 - round(a["bbox"][0], 2)) < 0.02 and abs(x1 - round(a["bbox"][2], 2)) < 0.02


def test_a_formula_bitmap_keeps_the_overhang_of_an_italic_letter_on_a_clear_background():
    """Formulas carried over to the next page are placed as bitmaps. Cropping the bitmap to the formula box would cut
    off the hook and tail of an italic f reaching past its character width, so horizontally it widens to the ink
    measured from the glyphs, and the placement widens on both sides by the scale. The background is transparent,
    so placing it on a frame line does not hide the line."""
    from app.formats.pdf_blocks import _FormulaImage
    from app.formats.pdf_place_formulas import _formula_png_rect

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_font(fontname="ti", fontbuffer=pymupdf.Font("tiit").buffer)
    page.insert_text((350, 526), "If f is a", fontname="ti", fontsize=10)
    chars = page.get_text("rawdict")["blocks"][0]["lines"][0]["spans"][0]["chars"]
    letter = next(char for char in chars[1:] if char["c"] == "f")
    span = {"text": "f", "bbox": letter["bbox"], "origin": letter["origin"]}
    bbox = pymupdf.Rect(letter["bbox"])
    image = _FormulaImage(doc, 0, bbox, [span])
    area = image.area()
    assert area.x0 < bbox.x0 - 0.5 and area.x1 > bbox.x1 + 0.5
    assert (area.y0, area.y1) == (bbox.y0, bbox.y1)
    assert area.x0 >= bbox.x0 - 1.5 and area.x1 <= bbox.x1 + 1.5
    target = pymupdf.Rect(100, 100, 100 + bbox.width * 2, 100 + bbox.height * 2)
    placed = _formula_png_rect({"png": image, "bbox": bbox}, target)
    assert placed.x0 == pytest.approx(100 - (bbox.x0 - area.x0) * 2)
    assert placed.x1 == pytest.approx(target.x1 + (area.x1 - bbox.x1) * 2)
    assert (placed.y0, placed.y1) == (target.y0, target.y1)
    pixmap = pymupdf.Pixmap(image.png())
    assert pixmap.alpha and pixmap.pixel(0, 0)[3] == 0
    a = next(char for char in chars if char["c"] == "a")
    plain = _FormulaImage(doc, 0, pymupdf.Rect(a["bbox"]), [{"text": "a", "bbox": a["bbox"], "origin": a["origin"]}])
    assert plain.area() == pymupdf.Rect(a["bbox"])


def test_a_one_line_paragraph_is_written_from_the_column_edge():
    """In a book with first-line indents, a one-line paragraph is written from the left edge of the column like a
    multi-line paragraph. Indented blocks whose indent width does not match, and list items, stay as before"""
    from app.formats.pdf_blocks import TextBlock, _paragraph_indent
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_write_geometry import _indent_edge

    def blk(rows, text):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=0, rect=rect, line_rects=rects, text=text, size=10.0, color="#000000", bold=False)

    paragraphs = [
        blk(
            [(157, 110 + 24 * k, 475, 122 + 24 * k), (142, 122 + 24 * k, 300, 134 + 24 * k)],
            "The softmax function takes a vector of arbitrary values. It is an exponential function.",
        )
        for k in range(5)
    ]
    geo = PageGeometry(612, (Column(142, 475),), (), 792, frame=TextFrame(142, 475, 109, 718, 10))
    indent = _paragraph_indent(paragraphs)
    assert indent is not None and abs(indent - 15) < 0.5
    one = blk([(157, 300, 400, 312)], "For a vector z of dimensionality K, the softmax is defined as:")
    assert _indent_edge(one, geo, indent) == 142
    quote = blk([(172, 330, 440, 342)], "Ok, we fly from San Francisco to Boston.")
    assert _indent_edge(quote, geo, indent) is None
    item = blk([(157, 360, 400, 372)], "• Annotators rank N samples of the output.")
    assert _indent_edge(item, geo, indent) is None
    row = blk([(157, 390, 260, 402)], "BiDAF+ELMo (Single)")
    assert _indent_edge(row, geo, indent) is None
    flush = [
        blk(
            [(142, 110 + 30 * k, 475, 122 + 30 * k), (142, 122 + 30 * k, 300, 134 + 30 * k)],
            "Paragraphs here are set apart by space. They start flush.",
        )
        for k in range(5)
    ]
    assert _paragraph_indent(flush) is None


def test_a_paragraph_opening_above_a_figure_is_written_from_the_column_edge(tmp_path):
    """Only the indented first line of a paragraph is left above a figure at the page end, and the paragraph continues
    on the next page. The translation is written from the left edge of the column like other paragraphs, and what
    does not fit continues on the next page without running down onto the frame line of the figure. An indented
    tail block is a list item and is still written from the left edge of its first line."""
    from app.formats.pdf_blocks import SPILL_TAIL, TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_write import _render_translated
    from app.formats.pdf_write_geometry import _indented_start

    def blk(page, rows, text):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=page, rect=rect, line_rects=rects, text=text, size=10.0, color="#000000", bold=False)

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    first = doc.new_page(width=612, height=792)
    first.draw_line((142.2, 541.5), (475.2, 541.5), width=0.4)
    doc.new_page(width=612, height=792)
    doc.save(src)
    doc.close()
    host = blk(
        0,
        [(157.1, 517.0, 475.2, 529.0)],
        "Formally, then, the gradient of a multi-variable function f is a vector in which",
    )
    tail = blk(
        1,
        [(160.2, 109.9, 493.2, 121.9), (160.2, 121.9, 493.2, 134.2)],
        "each component expresses the partial derivative of f with respect to one of the variables.",
    )
    even = PageGeometry(
        612, (Column(77.4, 261.0), Column(278.6, 475.2)), (), 792, frame=TextFrame(142.0, 475.0, 109, 718, 10)
    )
    odd = PageGeometry(612, (Column(160.2, 493.2),), (), 792, frame=TextFrame(160.0, 493.0, 109, 718, 10))
    assert _indented_start(host, tail, [even, odd]) == 142.0
    item = blk(1, [(175.2, 109.9, 493.2, 121.9)], "expresses the partial derivative of f.")
    assert _indented_start(host, item, [even, odd]) is None

    paragraph = (
        "形式化地说，多变量函数 f 的梯度是一个向量，其中每个分量表示 f 相对于其中一个变量的偏导数。"
        "我们将使用倒置的希腊字母 delta 符号来表示梯度，并以此更明显地体现它对参数的依赖关系。"
    )
    out = _render_translated(
        src,
        [host, tail],
        [paragraph, SPILL_TAIL],
        "zh-CN",
        geometries=[even, odd],
        continuations={id(host): tail},
    )
    chars = [
        ch["bbox"]
        for block in out[0].get_text("rawdict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        for ch in span["chars"]
        if "一" <= ch["c"] <= "鿿"
    ]
    assert chars
    assert min(box[0] for box in chars) < 150
    # Kept clear of the frame line instead of touching it
    assert max(box[3] for box in chars) < 541.5 - 5
    assert "偏导数" in out[1].get_text()


def test_a_numbered_item_with_hanging_lines_keeps_the_hang():
    """For an item with its number flush left and the following lines aligned with the text after the number, the
    continuation lines keep the original indent; paragraphs with a first-line indent and blocks without a number do
    not count."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_place import _hang_lanes
    from app.formats.pdf_write_geometry import _hang_indent

    def blk(text, lefts):
        rects = [pymupdf.Rect(x0, 170 + 12 * i, 475, 180 + 12 * i) for i, x0 in enumerate(lefts)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(min(lefts), 170, 475, 180 + 12 * (len(lefts) - 1)),
            line_rects=rects,
            text=text,
            size=10,
            color="#000",
            bold=False,
        )

    assert _hang_indent(blk("3.1 Write out the equation for trigram probability estimation", [178, 209, 209])) == 31
    assert _hang_indent(blk("The asphalt that Los Angeles is famous for occurs mainly", [175, 160, 160])) == 0
    assert _hang_indent(blk("tf-idf weighting of the value for word t in document d", [178, 209])) == 0

    # The text aligns 31pt right of the number; with a 20pt wide translated number the first line moves 11pt right,
    # and with one wider than 31pt the whole column moves right with it
    box = pymupdf.Rect(178, 170, 475, 220)
    assert _hang_lanes(box, [], 31, 0, 8.8, 1.45, 3) == [(0.0, 297), (31, 266), (31, 266)]
    assert _hang_lanes(box, [], 31, 20, 8.8, 1.45, 3) == [(11, 286), (31, 266), (31, 266)]
    assert _hang_lanes(box, [], 31, 40, 8.8, 1.45, 3) == [(0.0, 297), (40, 257), (40, 257)]


def test_the_label_width_is_measured_on_the_translation():
    """Width of the number or bullet at the start of the translation, including the space after it (with a hanging
    indent the first line moves right by this amount)."""
    from app.formats.pdf_flow import measure_width
    from app.formats.pdf_place import _label_width

    assert _label_width("{b}3.1{/b} 写出三元语法概率估计的方程", 8.8) == pytest.approx(measure_width("3.1 ", 8.8))
    assert _label_width("• 由训练有素的标注者直接标注", 8.8) == pytest.approx(measure_width("• ", 8.8))
    assert _label_width("写出三元语法概率估计的方程", 8.8) == 0
