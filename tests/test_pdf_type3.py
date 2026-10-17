import shutil
from pathlib import Path

import pymupdf
import pytest

from app.formats.pdf import TextBlock, extract_blocks
from app.formats.pdf import _numbered_list_item
from app.formats.pdf_type3 import (
    GlyphBox,
    OcrWord,
    _bar_zones,
    _cluster_drawings,
    _compose_lines,
    _drop_broken_lines,
    _group_lines,
    _hidden,
    _ink_size,
    _label_zones,
    _line_from_words,
    _pipe_zones,
    _snap_page_sizes,
    _without_noise,
    broken_type3_names,
    recover_type3_page,
    segment_ocr_words,
    visual_em,
)

def _paper() -> Path | None:
    for path in (
        Path("data/jobs/54435e9fa9fd/implementing_data_cube.pdf"),
        Path("data/jobs/24ac3975db56/implementing_data_cube.pdf"),
    ):
        if path.exists():
            return path
    return None


PAPER = _paper()


def _word(text: str, x0: float, x1: float, y0: float, y1: float) -> OcrWord:
    return OcrWord(text, pymupdf.Rect(x0, y0, x1, y1), 90.0)


def test_visual_em_reads_bitmap_box_not_scale():
    """点阵字的 size 是 0.1 的缩放，不是磅值。字框高度才是字号。"""
    assert visual_em({"size": 0.1, "ascender": 126.0, "descender": -65.0}) == 13.8
    assert visual_em({"size": 11.0, "ascender": 0.8, "descender": -0.2}) == 11.0


def test_title_stays_one_line_and_authors_split():
    """标题词距大约 8pt，署名之间大约 30pt。词心落在旧行框的空隙里也不能把标题拆丢。"""
    words = [
        _word("Implementing", 145.0, 264.3, 107.3, 123.3),
        _word("Data", 271.7, 313.0, 108.0, 119.7),
        _word("Cubes", 321.0, 373.0, 107.7, 119.7),
        _word("Efficiently*", 380.7, 472.7, 107.3, 123.0),
        _word("Venky", 112.3, 149.0, 141.7, 152.3),
        _word("Harinarayan", 154.0, 225.3, 141.3, 152.3),
        _word("Anand", 256.3, 294.0, 141.7, 150.0),
        _word("Rajaraman", 299.7, 364.0, 141.0, 152.3),
        _word("Jeffrey", 394.0, 433.7, 141.7, 152.3),
    ]
    texts = [" ".join(word.text for word in segment) for segment in segment_ocr_words(words)]
    assert "Implementing Data Cubes Efficiently*" in texts
    assert "Venky Harinarayan" in texts
    assert "Anand Rajaraman" in texts
    assert "Jeffrey" in texts
    assert all("Harinarayan Anand" not in text for text in texts)
    numbered = segment_ocr_words([
        _word("1", 73.0, 78.7, 428.3, 437.7),
        _word("Introduction", 97.0, 187.3, 427.7, 438.0),
        _word("1.1", 73.0, 88.3, 192.3, 200.0),
        _word("The", 103.0, 125.0, 191.3, 200.0),
        _word("Data", 130.7, 158.3, 191.3, 200.0),
    ])
    numbered_text = [" ".join(word.text for word in segment) for segment in numbered]
    assert "1 Introduction" in numbered_text
    assert "1.1 The Data" in numbered_text


def test_rebuilt_line_covers_the_word_not_the_fragment_gap():
    """旧行框在 Implementing 中间断开。新行框要盖住认出来的词，字号不能留在 0.1。"""
    words = [_word("Implementing", 145.0, 264.3, 107.3, 123.3)]
    glyphs = [
        GlyphBox(pymupdf.Rect(145.1, 101.2, 152.9, 120.3), 13.8, 0, 107.7, 0),
        GlyphBox(pymupdf.Rect(211.4, 101.2, 255.3, 120.3), 13.8, 0, 107.7, 0),
    ]
    line = _line_from_words(words, glyphs)
    assert line is not None
    span = line["spans"][0]
    assert span["text"] == "Implementing"
    assert span["size"] > 6
    assert span["font"] == "Times-Roman"
    assert line["bbox"][0] <= 145.1
    assert line["bbox"][2] >= 264.0
    assert abs(line["dir"][1]) < 0.01


def test_rebuilt_line_baseline_comes_from_the_words():
    """点阵字报的基线在字框上部（95.6），按它算公式下沉会掉一行。基线改取不带降部的词的下沿。"""
    from app.formats.pdf_type3 import _line_from_pieces, _Piece

    pieces = [
        _Piece("formula", "", pymupdf.Rect(100.1, 94.1, 251.9, 111.6)),
        _Piece("prose", "where", pymupdf.Rect(256.0, 96.0, 285.0, 105.0)),
        _Piece("prose", "the", pymupdf.Rect(300.0, 96.2, 315.0, 105.1)),
        _Piece("prose", "suppliers.", pymupdf.Rect(320.0, 96.0, 370.0, 108.0)),
    ]
    glyphs = [GlyphBox(pymupdf.Rect(100.5, 89.1, 535.4, 108.2), 10.0, 0, 95.6, 0)]
    line = _line_from_pieces(pieces, glyphs)
    assert line is not None
    assert all(abs(span["origin"][1] - 105.0) <= 0.2 for span in line["spans"])

    words = [_word("Implementing", 145.0, 264.3, 107.3, 119.0), _word("Data", 270.0, 300.0, 107.3, 119.2)]
    rebuilt = _line_from_words(words, [GlyphBox(pymupdf.Rect(145.1, 101.2, 300.0, 120.3), 13.8, 0, 104.0, 0)])
    assert round(rebuilt["spans"][0]["origin"][1], 1) == 119.2

def test_ocr_word_off_the_glyphs_is_not_added():
    words = [_word("Header", 70, 120, 40, 50)]
    glyphs = [GlyphBox(pymupdf.Rect(100, 240, 500, 252), 11.0, 0, 248.0, 0)]
    assert _line_from_words(words, glyphs) is None


def test_columns_stay_apart_and_wrapped_lines_join():
    left = {"bbox": (72.0, 100.0, 280.0, 112.0), "spans": []}
    right = {"bbox": (320.0, 100.0, 530.0, 112.0), "spans": []}
    left_next = {"bbox": (72.0, 114.0, 280.0, 126.0), "spans": []}
    groups = _group_lines([left, right, left_next])
    assert len(groups) == 2
    assert groups[0] == [left, left_next]
    assert groups[1] == [right]


def test_drop_broken_lines_keeps_images_and_normal_spans():
    data = {
        "blocks": [
            {"type": 1, "bbox": (0, 0, 10, 10)},
            {
                "type": 0,
                "lines": [
                    {"spans": [{"font": "A", "text": "junk", "bbox": (0, 0, 10, 10)}], "bbox": (0, 0, 10, 10)},
                    {"spans": [{"font": "Times-Roman", "text": "keep", "bbox": (0, 20, 30, 30)}], "bbox": (0, 20, 30, 30)},
                ],
            },
        ]
    }
    _drop_broken_lines(data, {"A"})
    assert data["blocks"][0]["type"] == 1
    assert data["blocks"][1]["lines"][0]["spans"][0]["text"] == "keep"


def test_centered_author_is_not_a_title_fragment():
    """标题下面隔开一行的居中署名，左缘也会漂移。不能并进标题。贴着的下一行仍并。"""
    from app.formats.pdf import _merge_continuations

    def blk(text: str, y: float, x0: float, x1: float, size: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + size)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    split = _merge_continuations([
        blk("Implementing Data Cubes Efficiently*", 106.7, 144.4, 473.3, 16.8),
        blk("Anand Rajaraman", 140.4, 255.7, 364.6, 11.9),
    ])
    assert len(split) == 2
    joined = _merge_continuations([
        blk("Learning to Execute with", 80, 120, 400, 16),
        blk("Recurrent Neural Networks", 98, 150, 430, 16),
    ])
    assert len(joined) == 1
    assert "Recurrent" in joined[0].text


def test_inserted_text_is_not_treated_as_bitmap_font():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a normal paragraph with enough letters to translate.", fontsize=11)
    data = page.get_text("dict")
    assert broken_type3_names(doc, page) == set()
    assert recover_type3_page(page, data) is False
    text = "".join(
        span["text"]
        for block in data["blocks"] if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    )
    assert "normal paragraph" in text


def _line(text: str, x0: float, y0: float, x1: float, y1: float, size: float = 11.0) -> dict:
    return {
        "bbox": (x0, y0, x1, y1),
        "spans": [{"text": text, "size": size, "bbox": (x0, y0, x1, y1)}],
    }


def test_formula_noise_drops_the_bitmap_and_keeps_the_clause():
    """公式位图认成 ®Ngupplier。乱码和贴着的短标点丢掉，后面的句子留下。"""
    words = [
        _word("(p,", 100.7, 111.3, 94.7, 106.0),
        _word("$1,", 114.7, 125.7, 98.0, 105.0),
        _word("®Ngupplier’", 177.3, 237.3, 98.0, 111.0),
        _word("c),", 240.3, 251.3, 95.0, 106.0),
        _word("where", 255.0, 282.3, 95.0, 103.0),
        _word("1.2.", 73.0, 88.3, 662.0, 669.7),
        _word("Ngyppiier", 286.7, 340.0, 95.3, 108.3),
    ]
    words[1] = OcrWord("$1,", words[1].rect, 39.9)
    words[2] = OcrWord(words[2].text, words[2].rect, 32.7)
    words[3] = OcrWord("c),", words[3].rect, 87.9)
    words[0] = OcrWord("(p,", words[0].rect, 89.4)
    words[5] = OcrWord("1.2.", words[5].rect, 52.2)
    words[6] = OcrWord("Ngyppiier", words[6].rect, 17.8)
    kept = [word.text for word in _without_noise(words)]
    assert kept == ["where", "1.2."]


def test_words_from_a_math_font_stay_glyphs():
    """数学斜体和符号字体认出来的「6}」「(L,」「a」留原字形；正文字体、标题里的 of、
    斜体的 i.e.、节号照常是正文。"""
    from app.formats.pdf_type3 import _math_font_words

    def word(text, x0, x1, conf):
        return OcrWord(text, pymupdf.Rect(x0, 100, x1, 110), conf)

    body = [word("lattice", 10, 50, 96), word("elements", 55, 100, 97), word("ordered", 105, 150, 95)]
    math = [word("6}", 160, 170, 50), word("(L,", 175, 190, 62), word("a", 195, 200, 93)]
    heading = word("of", 210, 220, 93)
    italic = word("i.e.,", 225, 245, 87)
    section = word("4.3", 250, 265, 96)
    glyphs = [
        GlyphBox(pymupdf.Rect(10, 96, 150, 114), 10.0, 0, 108.0, 0, "A"),
        GlyphBox(pymupdf.Rect(160, 96, 200, 114), 10.0, 0, 108.0, 0, "H"),
        GlyphBox(pymupdf.Rect(210, 96, 220, 114), 10.0, 0, 108.0, 0, "L"),
        GlyphBox(pymupdf.Rect(225, 96, 245, 114), 10.0, 0, 108.0, 0, "E"),
        GlyphBox(pymupdf.Rect(250, 96, 265, 114), 10.0, 0, 108.0, 0, "L"),
    ]
    words = [*body, *math, heading, italic, section]
    found = _math_font_words(words, glyphs)
    assert {w.text for w in words if id(w) in found} == {"6}", "(L,", "a"}
    # 组行后这三个词并成一个公式槽，其余照常是正文
    lines = _compose_lines(words, [], glyphs, {tuple(w.rect) for w in words if id(w) in found})
    texts = " ".join(span["text"] for line in lines for span in line["spans"])
    assert "6}" not in texts and "(L," not in texts
    assert "of" in texts and "i.e.," in texts and "4.3" in texts
    assert sum(span["text"] == "×" for line in lines for span in line["spans"]) == 1

def test_low_confidence_words_from_formulas_are_noise():
    """公式字形认成的 rains、Tag 置信度在 25 上下，当乱码；40 上下的正文词 cell 留下。"""
    words = [
        OcrWord("rains.", pymupdf.Rect(10, 100, 40, 110), 26.0),
        OcrWord("Tag.", pymupdf.Rect(45, 100, 70, 110), 25.0),
        OcrWord("cell", pymupdf.Rect(75, 100, 95, 110), 37.0),
    ]
    assert [word.text for word in _without_noise(words)] == ["cell"]

def test_centered_line_of_formula_slots_is_a_display_formula():
    """next(a) = {b | a < b, ...}：= 和 < 都成了公式槽，剩下的 nezt(a)、Ac,a 仍是一行公式，不能当正文去接下一页。"""
    from app.formats.pdf_type3 import _Piece, _display_formula

    line = [
        _Piece("prose", "nezt(a)", pymupdf.Rect(230, 340, 262, 352)),
        _Piece("formula", "×", pymupdf.Rect(264, 340, 330, 352)),
        _Piece("prose", "Ac,a", pymupdf.Rect(332, 340, 352, 352)),
        _Piece("formula", "×", pymupdf.Rect(354, 340, 390, 352)),
    ]
    assert _display_formula(line)
    sentence = [
        _Piece("prose", "where the views", pymupdf.Rect(230, 340, 300, 352)),
        _Piece("formula", "×", pymupdf.Rect(302, 340, 320, 352)),
        _Piece("prose", "are materialized first", pymupdf.Rect(322, 340, 420, 352)),
    ]
    assert not _display_formula(sentence)

def test_short_token_follows_the_glyph_origins():
    """数学斜体的 b 被认成 6。字框漂到了隔壁的正文上，按字框它属于正文字体；
    起笔点上一半是数学斜体，算公式。英文冠词 a 底下全是正文字体，仍是正文。"""
    from app.formats.pdf_type3 import _math_font_words

    def word(text, x0, x1, conf):
        return OcrWord(text, pymupdf.Rect(x0, 100, x1, 110), conf)

    body = [word("either", 10, 40, 96), word("benefit", 45, 80, 97), word("choice", 85, 115, 95)]
    misread = word("6", 120, 126, 93)
    article = word("a", 140, 146, 96)
    section = word("4.3", 160, 175, 96)
    words = [*body, misread, article, section]
    # 字框只有正文字体一条，横跨整行
    glyphs = [GlyphBox(pymupdf.Rect(10, 96, 180, 114), 10.0, 0, 101.0, 0, "A")]
    origins = [("A", x, 100.5) for x in (10, 20, 45, 60, 85, 100, 141, 143, 161, 166)]
    origins += [("H", 120.5, 100.5), ("A", 120.5, 100.5)]
    found = _math_font_words(words, glyphs, origins)
    assert {w.text for w in words if id(w) in found} == {"6"}
    # 带逗号时间隔字和逗号压过了一半，单个字符有一个数学字体的字仍算
    comma = word("6,", 120, 128, 95)
    origins += [("A", 126.0, 100.5), ("A", 127.0, 100.5)]
    assert id(comma) in _math_font_words([*body, comma], glyphs, origins)

def test_weak_tokens_next_to_formula_slots_join_the_slot():
    """For all i: b − x − … ≤ aj. 里认成 vig、aj 的碎片挨着公式槽，并进槽里；
    认得没把握的常见短词 is 和远离公式的词留下。"""
    words = [
        _word("For", 100, 115, 300, 310), _word("all", 119, 130, 300, 310),
        _word("vig", 160, 175, 300, 310), _word("<", 180, 186, 300, 310),
        _word("aj.", 190, 202, 300, 312), _word("is", 210, 218, 300, 310),
        _word("then", 224, 245, 300, 310),
    ]
    words = [OcrWord(w.text, w.rect, conf) for w, conf in zip(words, (96, 96, 63, 90, 69, 78, 97))]
    math_rects = {tuple(pymupdf.Rect(135, 300, 155, 310))}
    slot = _word("6;", 135, 155, 300, 310)
    words.insert(2, slot)
    from app.formats.pdf_type3 import _weak_token

    weak = {tuple(w.rect) for w in words if _weak_token(w)}
    assert {w.text for w in words if _weak_token(w)} == {"vig", "aj."}
    lines = _compose_lines(words, [], _glyphs_for(words), math_rects, weak)
    spans = [span["text"].strip() for line in lines for span in line["spans"]]
    assert spans.count("×") == 1
    text = " ".join(spans)
    assert "vig" not in text and "aj." not in text and "is" in text and "For all" in text


def test_subscripted_variable_fragment_is_a_formula_token():
    """a;41、v2, 这种字母夹数字的碎片是带下标的变量；1st、Q4 认得有把握的不算。"""
    from app.formats.pdf_type3 import _formula_token

    assert _formula_token(OcrWord("a;41,", pymupdf.Rect(0, 0, 10, 10), 49))
    assert _formula_token(OcrWord("v2,", pymupdf.Rect(0, 0, 10, 10), 66))
    assert not _formula_token(OcrWord("1st", pymupdf.Rect(0, 0, 10, 10), 60))
    assert not _formula_token(OcrWord("Q4", pymupdf.Rect(0, 0, 10, 10), 95))

def test_listing_with_math_font_symbols_stays_a_listing():
    """伪代码里 S、= 来自符号字体。先按原词认出伪代码，不能因为它们换成公式槽就当正文翻译。"""
    listing = [_word("S", 101, 108, 279, 290), _word("=", 113, 120, 282, 288),
               _word("{top", 124, 148, 278, 290), _word("view};", 152, 185, 278, 290)]
    math_rects = {tuple(listing[0].rect), tuple(listing[1].rect)}
    assert _compose_lines(listing, [], _glyphs_for(listing), math_rects) == []


def test_formula_slots_and_glue_between_them_merge_into_one():
    """r = j/(n−j)：r=、j 换成槽，夹在中间的 /(n— 也并进去，整段是一个公式槽。"""
    words = [
        _word("where", 72, 100, 300, 310),
        _word("r=", 105, 118, 300, 310),
        _word("j/(n—", 120, 148, 300, 312),
        _word("j),", 150, 162, 300, 312),
        _word("or", 168, 180, 302, 310),
        _word("both", 185, 210, 300, 310),
    ]
    math_rects = {tuple(words[1].rect), tuple(words[3].rect)}
    lines = _compose_lines(words, [], _glyphs_for(words), math_rects)
    spans = [span["text"].strip() for line in lines for span in line["spans"]]
    assert spans.count("×") == 1
    assert "j/(n—" not in " ".join(spans) and "where" in " ".join(spans) and "both" in " ".join(spans)

def test_numbered_heading_list_and_sql_stay_apart():
    """节标题不吞正文，相邻列表项不并成一行，SQL 子句各留各的行。"""
    heading = _line("1.2. Motivating Example", 72, 661, 223, 672, 12)
    body = _line("The TPC-D database we considered", 72, 674, 540, 684, 10)
    item3 = _line("3. part, supplier (0.8M)", 86, 152, 220, 163)
    item4 = _line("4, supplier, customer (6M)", 86, 165, 250, 176)
    select = _line("SELECT Part, Customer, SUM(SP) AS TotalSales", 100, 471, 352, 480)
    from_r = _line("FROM R", 100, 482, 135, 493)
    group = _line("GROUP BY Part, Customer;", 100, 495, 236, 506)
    groups = _group_lines([heading, body, item3, item4, select, from_r, group])
    texts = [" ".join(span["text"] for line in group for span in line["spans"]) for group in groups]
    assert [len(group) for group in groups] == [1, 1, 1, 1, 1, 1, 1]
    assert set(texts) == {
        "1.2. Motivating Example",
        "The TPC-D database we considered",
        "3. part, supplier (0.8M)",
        "4, supplier, customer (6M)",
        "SELECT Part, Customer, SUM(SP) AS TotalSales",
        "FROM R",
        "GROUP BY Part, Customer;",
    }


def test_clause_after_a_formula_hole_is_not_pulled_over_the_bitmap():
    """公式右边的 where 不并进上一行，下一行回到左缘也不并上来。"""
    before = _line("sum of the values of cells of", 100, 81, 540, 92)
    clause = _line("where is the number of suppliers. The more cells", 255, 95, 540, 106)
    after = _line("we materialize, the better query performance is.", 100, 112, 540, 122)
    groups = _group_lines([before, clause, after])
    assert len(groups) == 3
    assert groups[1][0]["spans"][0]["text"].startswith("where")


def test_drawing_cluster_covers_labels_beside_thin_strokes():
    """细圆和连线收成一块，贴在圆外侧的标签落在框里。下面的图注不在框里。"""
    circles = [
        pymupdf.Rect(200, 150, 218, 168),
        pymupdf.Rect(200, 180, 218, 198),
        pymupdf.Rect(200, 210, 218, 228),
        pymupdf.Rect(272, 150, 290, 168),
        pymupdf.Rect(272, 210, 290, 228),
    ]
    edges = [pymupdf.Rect(218, 158, 272, 176), pymupdf.Rect(208, 168, 226, 182)]
    zones = _cluster_drawings([*circles, *edges], 612, 792)
    assert len(zones) == 1
    zone = zones[0].rect
    assert zone.x0 <= 188  # 圆左缘 200，再留出节点旁的 20
    assert zone.y1 >= 228
    assert zone.y1 < 260  # 图注在更下面


def test_pipe_column_covers_the_glyph_table_not_the_caption():
    """竖线不是矢量。表头和格子落在竖线带里，表注在带外。"""
    pipes = [
        _word("|", 209, 214, y, y + 12) for y in (452, 468, 496, 524, 552, 580, 608)
    ]
    pipes.extend([
        _word("|", 266, 271, 452, 465),
        _word("|", 313, 318, 452, 465),
        _word("|", 377, 382, 452, 465),
    ])
    zones = _pipe_zones(pipes)
    assert len(zones) == 1
    zone = zones[0].rect
    assert zone.x0 <= 170 and zone.x1 >= 430
    assert zone.y0 <= 452 and zone.y1 >= 620
    assert zone.y1 < 650


def test_same_line_clause_after_a_dash_stays_in_the_sentence():
    """破折号后的列举和左半句在同一基线，不能当成公式右边的半句拆开。"""
    left = _line("pick the first five views", 72, 511, 182, 521)
    names = _line("cp, ns, nt, c, and p", 201, 512, 290, 521)
    groups = _group_lines([left, names])
    assert len(groups) == 1


def test_next_line_does_not_cover_the_other_half_of_a_formula():
    """公式把一行拆成左右两半时，下一整行不能并进左半，否则写入框盖住右半。"""
    left = _line("Let w1, w2", 90, 684, 160, 693)
    right = _line("be the optimal set of k views", 178, 683, 539, 693)
    nxt = _line("The order in which these views appear is arbitrary", 72, 696, 540, 706)
    groups = _group_lines([left, right, nxt])
    texts = [" ".join(span["text"] for line in group for span in line["spans"]) for group in groups]
    assert "The order" not in texts[0]
    assert any(text.startswith("The order") for text in texts)


def test_glyph_rule_drops_a_short_label_not_the_caption():
    pipes = [_word("|", 235, 237, 78, 96), _word("|", 235, 237, 114, 132)]
    none = _word("none", 223, 245, 141, 146)
    caption = _word("Customer", 227, 273, 166, 174)
    zones = _label_zones(pipes, [])
    assert _hidden(none, zones)
    assert not _hidden(caption, zones)


def test_short_numbered_list_item_is_not_a_formula_line():
    assert _numbered_list_item("5. part (0.2M)")
    assert _numbered_list_item("8. none (1)")
    assert not _numbered_list_item("5.")
    assert not _numbered_list_item("(p, ALL, c)")


def _glyphs_for(words: list[OcrWord]) -> list[GlyphBox]:
    return [
        GlyphBox(word.rect, 10.0, 0, word.rect.y1, 0)
        for word in words
    ]


def test_formula_hole_stays_in_the_sentence():
    """公式位图拆开的 where 和 is 仍是一句，空洞是公式槽，乱码不进正文。"""
    prose = [
        _word("where", 255, 282, 95, 103),
        _word("is", 343, 355, 95, 103),
        _word("the", 360, 380, 95, 103),
    ]
    debris = [
        OcrWord("(p,", pymupdf.Rect(100, 94, 112, 106), 20.0),
        OcrWord("c),", pymupdf.Rect(240, 95, 251, 106), 30.0),
        OcrWord("Ngyppiier", pymupdf.Rect(286, 95, 340, 108), 17.0),
    ]
    lines = _compose_lines(prose, debris, _glyphs_for([*prose, *debris]))
    assert len(lines) == 1
    fonts = [span["font"] for span in lines[0]["spans"]]
    texts = " ".join(span["text"] for span in lines[0]["spans"])
    assert "CMMI10" in fonts
    assert "where" in texts and "is" in texts
    assert "Ngyppiier" not in texts
    assert lines[0]["bbox"][0] < 120


def test_formula_between_words_spans_the_whole_gap():
    """点阵字框比墨迹偏右几个点：where 和 is 之间的 N_supplier 只有 supplier 的字框落在空洞里。
    公式槽取两词之间的整段，N 不能被切掉。"""
    from app.formats.pdf_type3 import _formula_between

    where = _word("where", 255.0, 282.3, 95.0, 103.0)
    is_ = _word("is", 343.7, 350.3, 95.3, 103.0)
    glyphs = [GlyphBox(pymupdf.Rect(295.2, 92.2, 335.2, 107.5), 10.0, 0, 101.0, 0, "H")]
    piece = _formula_between(where, is_, [], glyphs)
    assert piece is not None and piece.kind == "formula"
    assert piece.rect.x0 < 286 and piece.rect.x1 > 340

def test_centered_formula_line_is_left_as_glyphs():
    """居中的 (year) < (month) 是公式行，不送去翻译。"""
    words = [
        _word("(year)", 243, 272, 625, 637),
        _word("<", 277, 284, 626, 635),
        _word("(month)", 290, 325, 625, 637),
        _word("=", 330, 337, 626, 635),
        _word("(day).", 343, 370, 625, 637),
    ]
    assert _compose_lines(words, [], _glyphs_for(words)) == []


def test_formula_line_starting_with_a_misread_digit_is_left_as_glyphs():
    """居中公式行开头的 ≥ 被认成 2，后面是 RAR、(ERY) 这种碎片：不是编号标题，留原字。
    居中的「5 Conclusions」仍是编号标题。"""
    words = [
        _word("2", 225, 232, 700, 712),
        _word("RAR", 245, 290, 700, 712),
        _word("+", 297, 304, 700, 712),
        _word("AR)", 310, 350, 700, 712),
        _word("(ERY)", 377, 420, 700, 712),
    ]
    debris = [OcrWord("~~", pymupdf.Rect(356, 699, 372, 713), 10.0)]
    assert _compose_lines(words, debris, _glyphs_for([*words, *debris])) == []
    heading = [_word("5", 250, 257, 300, 312), _word("Conclusions", 262, 330, 300, 312)]
    assert len(_compose_lines(heading, [], _glyphs_for(heading))) == 1

def test_inline_operator_and_paren_digit_become_formula_slots():
    """行内的 < 和括号之间的 4 是认错的比较符，收成公式槽，句子留下。"""
    words = [
        _word("(part)", 73, 102, 591, 603),
        _word("4", 107, 113, 592, 603),
        _word("(customer)", 119, 171, 592, 603),
        _word("and", 176, 193, 592, 603),
        _word("(part).", 200, 240, 592, 603),
    ]
    lines = _compose_lines(words, [], _glyphs_for(words))
    assert len(lines) == 1
    slots = [span for span in lines[0]["spans"] if span["font"] == "CMMI10"]
    assert len(slots) == 1
    assert "4" not in " ".join(span["text"] for span in lines[0]["spans"] if span["font"] != "CMMI10")


def test_bullet_letter_is_not_the_first_word():
    """项目符号被认成 e。句子从后面的大写词起翻，e 不送出去。"""
    words = [
        _word("e", 88, 96, 464, 474),
        _word("The", 100, 124, 464, 474),
        _word("cost", 128, 150, 464, 474),
    ]
    lines = _compose_lines(words, [], _glyphs_for(words))
    text = " ".join(span["text"] for span in lines[0]["spans"])
    assert text.strip().startswith("The")
    assert not text.strip().startswith("e ")


def test_set_builder_bar_is_not_a_table():
    """集合记号里单独的竖线不是表。同一基线上有大括号就不成区。"""
    words = [
        _word("descendant(a)", 239, 306, 215, 226),
        _word("=", 311, 318, 219, 222),
        _word("{b", 323, 332, 215, 226),
        _word("|", 337, 338, 215, 226),
        _word("a}", 362, 372, 215, 226),
        _word("nezt(a)", 217, 251, 272, 283),
        _word("=", 257, 263, 276, 279),
        _word("{b", 268, 277, 272, 283),
        _word("|", 282, 283, 272, 283),
        _word("6}", 385, 394, 272, 283),
        _word("Note", 72, 94, 236, 243),
    ]
    assert _bar_zones(words) == []


def test_bar_zone_covers_the_left_column():
    """竖线在格子中间时，左列和没认出竖线的中间行也留下原字。表注在带外。"""
    words = [
        _word("|", 360, 366, 559, 573),
        _word("Source", 110, 141, 559, 572),
        _word("Size", 338, 356, 561, 569),
        _word("|", 427, 431, 576, 588),
        _word("From", 110, 134, 578, 585),
        _word("10,000", 327, 356, 591, 601),
        _word("|", 304, 308, 616, 629),
        _word("From", 110, 134, 618, 626),
        _word("Figure 5: Growth", 166, 445, 652, 663),
    ]
    zones = _bar_zones(words)
    assert len(zones) == 1
    assert _hidden(words[1], zones)
    assert _hidden(words[5], zones)
    assert not _hidden(words[-1], zones)


def test_bar_glyphs_cover_the_table_not_the_caption():
    """竖线粘在格子文字里时，整表留下，表注不在带里。"""
    words = [
        _word("Size | Time", 300, 420, 559, 570),
        _word("2.07 | not", 390, 470, 576, 587),
        _word("customer) | 6,000,000", 110, 357, 615, 626),
        _word("Source", 110, 150, 559, 570),
        _word("Figure 5: Growth", 166, 445, 652, 663),
    ]
    zones = _bar_zones(words)
    assert len(zones) == 1
    assert _hidden(words[3], zones)
    assert not _hidden(words[4], zones)


def test_example_and_footnote_do_not_join_the_paragraph():
    """EXAMPLE 和脚注各自起段。脚注的下一行仍接在脚注上。"""
    body = _line("we only need the assumptions that", 72, 646, 343, 657)
    example = _line("EXAMPLE 2.1 The hypercube of Fig. 1", 72, 374, 540, 386)
    rest = _line("Section 1.2.", 72, 388, 127, 397)
    note = _line('"Here we disregard the number of index nodes accessed.', 85, 702, 310, 712)
    follow = _line("cubes are sparse: only a small fraction", 71, 714, 540, 724)
    groups = _group_lines([body, example, rest, note, follow])
    texts = [" ".join(span["text"] for line in group for span in line["spans"]) for group in groups]
    assert any(text.startswith("EXAMPLE 2.1") and "Section 1.2." in text for text in texts)
    assert all("assumptions" not in text or "EXAMPLE" not in text for text in texts)
    assert any(text.startswith('"Here') and "sparse" in text for text in texts)
    assert all("assumptions" not in text or "sparse" not in text for text in texts)


def test_example_reference_inside_a_sentence_stays_in_the_paragraph():
    """句中换行碰到「Example 1.1, we」不是例题标题，要和上一行连成一段；Example 2.1 Consider 仍单独起段。"""
    first = _line("function of the values of other cells. For instance, in", 98, 67, 540, 79)
    second = _line("Example 1.1, we can compute the value of cell", 99, 80, 540, 93)
    third = _line("where N is the number of suppliers.", 99, 94, 400, 106)
    heading = _line("Example 2.1 Consider the lattice of Fig. 7.", 99, 107, 540, 119)
    groups = _group_lines([first, second, third, heading])
    texts = [" ".join(span["text"] for line in group for span in line["spans"]) for group in groups]
    assert any("For instance, in Example 1.1, we can" in text and "suppliers." in text for text in texts)
    assert any(text.startswith("Example 2.1 Consider") for text in texts)

def test_label_just_above_a_figure_stays_a_glyph():
    """节点名贴在笔画上方几个点，仍留下原字。下面的图注不挡。"""
    from app.formats.pdf_type3 import _Zone

    zone = _Zone(pymupdf.Rect(263, 154, 362, 239))
    day = _word("Day", 301, 321, 148, 159)
    caption = _word("Hierarchy", 263, 309, 264, 275)
    assert _hidden(day, [zone])
    assert not _hidden(caption, [zone])


def test_descender_does_not_inflate_the_line():
    """有没有 g、p 不能把同样高的墨迹估成两档字号。"""
    assert _ink_size("cost", 11.5) == 11.5
    assert _ink_size("gap", 11.5) == 11.5


def test_page_body_snaps_to_one_size_and_taller_ink_stays():
    """同一页正文收成墨迹中位数。明显更高的标题留下。公式槽的高框不算。"""
    body = _line("we compute the benefit of each view", 72, 100, 400, 111.5, 16.9)
    other = _line("picking a view reduces the query cost", 72, 114, 400, 125.5, 12.1)
    title = _line("Theorem", 72, 40, 160, 58, 18.0)
    bloated = _line("This bound comes most easily from the first inequality", 88, 200, 400, 217.5, 17.5)
    mixed = {
        "bbox": (72, 130, 400, 154),
        "spans": [
            {"text": "the benefit of selecting this view", "size": 16.9, "font": "Times-Roman", "bbox": (72, 130, 300, 141.5)},
            {"text": "×", "size": 16.9, "font": "CMMI10", "bbox": (304, 126, 360, 150)},
        ],
    }
    _snap_page_sizes([body, other, title, mixed, bloated])
    assert body["spans"][0]["size"] == other["spans"][0]["size"] == mixed["spans"][0]["size"]
    assert bloated["spans"][0]["size"] == body["spans"][0]["size"]
    assert title["spans"][0]["size"] == 18.0


def test_letter_item_does_not_join_the_paragraph():
    """(a) 不并进上一段，(b) 不并进 (a)。(a) 的续行仍留在这一条。"""
    prev = _line("there must be at least one such view in S.", 72, 67, 500, 79)
    item = _line("(a) Let u be the view of least cost in S.", 105, 82, 500, 94)
    cont = _line("S, there must be at least one such view.", 124, 96, 500, 108)
    item_b = _line("(b) If C(v) is smaller then stop here.", 104, 111, 500, 123)
    groups = _group_lines([prev, item, cont, item_b])
    texts = [" ".join(span["text"] for line in group for span in line["spans"]) for group in groups]
    assert any(text.startswith("(a)") and "S, there must" in text for text in texts)
    assert all("least one such view in S." not in text or "(a)" not in text for text in texts)
    assert any(text.startswith("(b)") and "(a)" not in text for text in texts)


def test_listing_line_is_left_as_glyphs():
    """缩进的伪代码不送去翻译。栏左缘的正文留下。"""
    listing = [
        _word("S = {top view};", 101, 220, 278, 290),
        _word("for i=1 to k do begin", 101, 250, 293, 305),
        _word("select that view v not in S such that B(v,S) is maximized;", 124, 360, 306, 318),
        _word("end;", 101, 130, 334, 346),
        _word("resulting S is the greedy selection;", 101, 280, 347, 359),
    ]
    body = [
        _word("We", 72, 90, 320, 332),
        _word("compute", 94, 150, 320, 332),
        _word("the", 154, 172, 320, 332),
        _word("benefit", 176, 220, 320, 332),
    ]
    lines = _compose_lines([*listing, *body], [], _glyphs_for([*listing, *body]))
    text = " ".join(span["text"] for line in lines for span in line["spans"])
    assert "top view" not in text
    assert "do begin" not in text
    assert "select that view" not in text
    assert "greedy selection" not in text
    assert "benefit" in text


def test_equals_in_a_cell_still_zones_the_table():
    """格子里的等号不是公式。竖线围出的表留下，表注在带外。"""
    words = [
        _word("First Choice | Second", 197, 420, 169, 181),
        _word("b |", 179, 198, 186, 198),
        _word("50 x 5 = 250", 204, 320, 186, 198),
        _word("f |", 178, 196, 240, 252),
        _word("60 x 2 = 120", 202, 320, 240, 252),
        _word("Figure 8: Benefits of each round", 184, 420, 303, 315),
    ]
    zones = _bar_zones(words)
    assert len(zones) == 1
    assert _hidden(words[0], zones)
    assert _hidden(words[1], zones)
    assert _hidden(words[3], zones)
    assert not _hidden(words[5], zones)


def test_broken_subscript_becomes_a_formula_slot():
    """B,, 和 6; 是认错的下标，收成公式槽。an 和 Bu- 仍是正文。"""
    words = [
        _word("define", 72, 110, 70, 82),
        _word("B,,", 114, 132, 72, 84),
        _word("to", 136, 150, 70, 82),
        _word("be", 154, 170, 70, 82),
        _word("an", 174, 190, 70, 82),
        _word("Bu-", 194, 214, 70, 82),
        _word("S,", 218, 234, 70, 82),
    ]
    lines = _compose_lines(words, [], _glyphs_for(words))
    assert len(lines) == 1
    prose = " ".join(span["text"] for span in lines[0]["spans"] if span["font"] != "CMMI10")
    assert "B,," not in prose
    assert "an" in prose and "Bu-" in prose and "S," in prose
    assert any(span["font"] == "CMMI10" for span in lines[0]["spans"])
    kept = _compose_lines(
        [_word("Q;", 72, 90, 100, 112), _word("The", 94, 120, 100, 112), _word("cost", 124, 150, 100, 112)],
        [],
        _glyphs_for([_word("Q;", 72, 90, 100, 112), _word("The", 94, 120, 100, 112)]),
    )
    kept_text = " ".join(span["text"] for line in kept for span in line["spans"] if span["font"] != "CMMI10")
    assert "Q;" in kept_text


def test_line_bbox_covers_glyph_width_not_its_height():
    """擦除加宽到点阵字的左右。19pt 字框的高度不并进行框，以免压到下一行。"""
    word = _word("cost", 72, 100, 100, 111)
    glyphs = [GlyphBox(pymupdf.Rect(70, 90, 108, 112), 13.8, 0, 110, 0)]
    lines = _compose_lines([word], [], glyphs)
    box = lines[0]["bbox"]
    assert box[0] <= 70
    assert box[2] >= 108
    assert box[3] - box[1] < 16


def test_close_ink_sizes_in_one_paragraph_snap():
    """同一段墨迹差一号，不能因此在送翻文本里插入字号标记。"""
    groups = _group_lines([
        _line("Decision support applications", 72, 240, 400, 251, 9.5),
        _line("often need to aggregate data", 72, 254, 400, 266, 11.2),
    ])
    sizes = [span["size"] for line in groups[0] for span in line["spans"]]
    assert len(sizes) == 2
    assert sizes[0] == sizes[1]


@pytest.mark.skipif(PAPER is None, reason="样例论文不在仓库里")
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="本机没有 tesseract")
def test_custom_type3_page_reads_as_english():
    assert PAPER is not None
    src = pymupdf.open(PAPER)
    doc = pymupdf.open()
    doc.insert_pdf(src, from_page=0, to_page=0)
    blocks = extract_blocks(doc)
    texts = [block.text for block in blocks]
    title = next(text for text in texts if "Implementing" in text and "Data Cubes" in text)
    assert "Anand" not in title
    assert any("Venky Harinarayan" in text for text in texts)
    assert any("Abstract" in text for text in texts)
    assert any("1 Introduction" in text for text in texts)
    assert any("Decision support" in text for text in texts)
    assert all("n7oqp" not in text for text in texts)
    assert all(block.size > 6 for block in blocks if len(block.text) > 12)


@pytest.mark.skipif(PAPER is None, reason="样例论文不在仓库里")
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="本机没有 tesseract")
def test_type3_leaves_figures_tables_and_formula_bitmaps():
    """格子标签、字形表格和公式位图不重排。标题、列表和图注仍是正文。"""
    assert PAPER is not None
    src = pymupdf.open(PAPER)
    doc = pymupdf.open()
    for pno in (2, 3, 14):
        doc.insert_pdf(src, from_page=pno, to_page=pno)
    texts = [block.text for block in extract_blocks(doc)]
    assert any("number of suppliers" in text for text in texts)
    assert all("Ngupplier" not in text and "Ngyppiier" not in text for text in texts)
    assert any(text.strip().startswith("1.2.") for text in texts)
    assert all("TPC-D" not in text or not text.strip().startswith("1.2.") for text in texts)
    assert not any("SELECT" in text and "GROUP BY" in text for text in texts)
    assert any(text.strip().startswith("3.") for text in texts)
    assert any(text.strip().startswith("4,") or text.strip().startswith("4.") for text in texts)
    assert any("5. part" in text for text in texts)
    assert all("0.8M) 4" not in text for text in texts)
    assert any(text.strip().startswith("Figure 1") for text in texts)
    assert all("none |" not in text for text in texts)
    assert any(text.strip().startswith("Figure 9") for text in texts)
    assert any(text.strip().startswith("Figure 10") for text in texts)
    assert any("greedy" in text for text in texts)
    assert all("100 KY" not in text for text in texts)
    assert not any("Selection" in text and "Benefit" in text for text in texts)


@pytest.mark.skipif(PAPER is None, reason="样例论文不在仓库里")
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="本机没有 tesseract")
def test_type3_keeps_formula_slots_and_glyph_tables():
    """行内公式空洞留在原句里，居中公式、节点名和字形表不重排。"""
    assert PAPER is not None
    src = pymupdf.open(PAPER)
    doc = pymupdf.open()
    for pno in (2, 4, 5, 6, 7, 9, 10, 11):
        doc.insert_pdf(src, from_page=pno, to_page=pno)
    blocks = extract_blocks(doc)
    texts = [block.text for block in blocks]
    suppliers = next(text for text in texts if "number of suppliers" in text)
    assert "where" in suppliers
    assert "Ngupplier" not in suppliers
    assert all("(year) <" not in text and "(month) =" not in text for text in texts)
    assert all(text.strip() != "Day" for text in texts)
    assert any(text.startswith("Figure 2") for text in texts)
    assert any("Figure 5" in text for text in texts)
    assert all("From cell itself" not in text for text in texts)
    assert all(not text.strip().startswith("e ") for text in texts)
    greedy = next(text for text in texts if "space cost is the number of rows" in text)
    assert "be the cost of view" in greedy
    assert any(text.strip().startswith("EXAMPLE") for text in texts)
    assert any(text.strip().startswith("1.3") for text in texts)


def _bitmap_font_pdf(path: Path) -> None:
    """第一页全是自造编码的点阵字（没有 ToUnicode，字形名是 /a0 这种），第二页是普通文字。"""
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
    doc.update_object(font, (
        "<< /Type /Font /Subtype /Type3 /Name /BitmapF /FontBBox [0 0 500 700] "
        f"/FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << {charprocs} >> "
        f"/Encoding << /Type /Encoding /Differences [65 {glyphs}] >> "
        f"/FirstChar 65 /LastChar 74 /Widths [{' '.join(['500'] * 10)}] /Resources << >> >>"
    ))
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /T3 {font} 0 R >> >>")
    codes = "".join(f"{65 + i % 10:02X}" for i in range(30))
    content = doc.get_new_xref()
    doc.update_object(content, "<<>>")
    doc.update_stream(content, "\n".join(
        f"BT /T3 12 Tf 40 {250 - row * 16} Td <{codes}> Tj ET" for row in range(6)
    ).encode())
    doc.xref_set_key(page.xref, "Contents", f"{content} 0 R")
    doc.new_page(width=400, height=300).insert_text(
        (40, 60), "A normal paragraph on the next page is still translated.", fontsize=11,
    )
    doc.save(path)


@pytest.mark.parametrize("failure", ["no_lines", "timeout"])
def test_unreadable_bitmap_page_keeps_its_glyphs(tmp_path, monkeypatch, failure):
    """一页点阵字识别超时或认不出正文时，这一页留原样，其他页照常翻译，不让整本书失败。"""
    import subprocess

    from app.formats import pdf_type3

    def fake_ocr(page, scale=3.0):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("tesseract", 90)
        return []

    monkeypatch.setattr(pdf_type3, "ocr_page", fake_ocr)
    src = tmp_path / "bitmap.pdf"
    _bitmap_font_pdf(src)
    doc = pymupdf.open(src)
    assert broken_type3_names(doc, doc[0]) == {"BitmapF"}
    blocks = extract_blocks(doc)
    assert [block.page for block in blocks] == [1]
    assert "normal paragraph" in blocks[0].text
