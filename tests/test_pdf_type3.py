import shutil
from pathlib import Path

import pymupdf
import pytest

from app.formats.pdf import TextBlock, extract_blocks
from app.formats.pdf import _numbered_list_item
from app.formats.pdf_type3 import (
    GlyphBox,
    OcrWord,
    _cluster_drawings,
    _drop_broken_lines,
    _group_lines,
    _hidden,
    _label_zones,
    _line_from_words,
    _pipe_zones,
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
