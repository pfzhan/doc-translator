import shutil
from pathlib import Path

import pymupdf
import pytest

from app.formats.pdf import TextBlock, extract_blocks
from app.formats.pdf_type3 import (
    GlyphBox,
    OcrWord,
    _drop_broken_lines,
    _group_lines,
    _line_from_words,
    broken_type3_names,
    recover_type3_page,
    segment_ocr_words,
    visual_em,
)

PAPER = Path("data/jobs/24ac3975db56/implementing_data_cube.pdf")


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


@pytest.mark.skipif(not PAPER.exists(), reason="样例论文不在仓库里")
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="本机没有 tesseract")
def test_custom_type3_page_reads_as_english():
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
