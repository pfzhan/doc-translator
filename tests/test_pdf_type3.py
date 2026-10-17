import subprocess
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from app.formats import pdf_type3
from app.formats.pdf import TextBlock, _code_block, _numbered_list_item, extract_blocks
from app.formats.pdf_type3 import (
    Glyph,
    Label,
    OcrChar,
    OcrWord,
    Type3Reader,
    _build_lines,
    _bullets,
    _drop_broken_lines,
    _glyph_styles,
    _group_lines,
    _HocrWords,
    _is_script,
    _learn_roles,
    _Line,
    _mono_kinds,
    _vote_labels,
    broken_type3_names,
    recover_type3_page,
)

PAPER = next(iter(sorted(Path("data/jobs").glob("*/implementing_data_cube.pdf"))), None)
needs_paper = pytest.mark.skipif(PAPER is None, reason="样例论文不在本机")
needs_tesseract = pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None, reason="本机没有 tesseract",
)

ROMAN, MATH, TYPE = 1, 2, 3
SIZE = 10.0


def _box(ch: str, x: float, baseline: float, size: float) -> tuple[pymupdf.Rect, float]:
    """按 Computer Modern 的比例摆一个字的墨迹框，返回框和字宽。"""
    if ch in "acemnorsuvwxz":
        top, bottom = 0.45, 0.0
    elif ch in "gpqy":
        top, bottom = 0.45, -0.19
    elif ch == ",":
        top, bottom = 0.1, -0.15
    elif ch == ".":
        top, bottom = 0.1, 0.0
    elif ch in "()":
        top, bottom = 0.75, -0.25
    elif ch in "=+<":
        top, bottom = 0.5, 0.1
    elif ch in "’'":
        top, bottom = 0.69, 0.45
    else:
        top, bottom = 0.69, 0.0
    width = 0.25 * size if ch in ",.’'()il" else 0.5 * size
    return pymupdf.Rect(x, baseline - top * size, x + width, baseline - bottom * size), width


def _set(
    words: list[tuple[str, int]], labels: dict, x: float = 72.0, baseline: float = 100.0,
    size: float = SIZE, shift: float = 0.0,
) -> list[Glyph]:
    """一行字。family 是位图那一套：正文、数学斜体、等宽。字距 0.05 个字号，词距 0.4 个字号。"""
    glyphs: list[Glyph] = []
    for text, family in words:
        for ch in text:
            rect, width = _box(ch, x, baseline + shift, size)
            if family == TYPE:
                pitch = 0.525 * size
                rect = pymupdf.Rect(x + (pitch - width) / 2, rect.y0, x + (pitch + width) / 2, rect.y1)
                width = pitch - 0.05 * size
            key = (family, ord(ch))
            labels.setdefault(key, Label(ch, 1.0))
            glyphs.append(Glyph(key, rect))
            x += width + 0.05 * size
        x += 0.4 * size
    return glyphs


def _text(line: _Line, labels: dict) -> str:
    return " ".join("".join(labels[glyph.key].text for glyph in token) for token in line.tokens)


def _reader(labels: dict, roles: dict, mono: dict | None = None) -> Type3Reader:
    reader = Type3Reader(pymupdf.open())
    reader.labels = labels
    reader.roles = roles
    reader.mono = mono or {}
    return reader


def _styled(line: _Line, reader: Type3Reader) -> str:
    """公式字形加方括号，看哪些进了公式。"""
    styles = _glyph_styles(line, reader)
    out = []
    for token in line.tokens:
        out.append("".join(
            reader.labels[glyph.key].text if styles[id(glyph)] in ("text", "mono")
            else f"[{reader.labels[glyph.key].text}]"
            for glyph in token
        ))
    return " ".join(out)


def test_hocr_gives_words_with_letter_boxes():
    raw = (
        "<span class='ocrx_word' title='bbox 30 60 90 90; x_wconf 96'>"
        "<span class='ocrx_cinfo' title='x_bboxes 30 60 45 90; x_conf 99.1'>o</span>"
        "<span class='ocrx_cinfo' title='x_bboxes 45 60 90 90; x_conf 90.5'>ﬃ</span></span>"
    )
    parser = _HocrWords(3.0)
    parser.feed(raw)
    word = parser.words[0]
    assert word.text == "offi"
    assert word.rect == pymupdf.Rect(10, 20, 30, 30)
    assert [char.text for char in word.chars] == ["o", "ffi"]
    assert word.chars[1].conf == 90.5


def test_votes_pair_letters_and_give_a_ligature_its_letters():
    """字数和字形数对上的词逐个配对。office 少两个字形：o、c、e 已认定，剩下的 ffi 归连字。"""
    def glyph(key: int, x0: float) -> Glyph:
        return Glyph((1, key), pymupdf.Rect(x0, 90, x0 + 5, 100))

    def word(text: str, glyphs: list[Glyph], letters: list[str]) -> OcrWord:
        box = pymupdf.Rect(glyphs[0].rect.x0, 90, glyphs[-1].rect.x1, 100)
        step = box.width / len(letters)
        chars = tuple(
            OcrChar(letter, pymupdf.Rect(box.x0 + i * step, 90, box.x0 + (i + 1) * step, 100), 95.0)
            for i, letter in enumerate(letters)
        )
        return OcrWord(text, box, 95.0, chars)

    c, a, t, o, e, ffi = 1, 2, 3, 4, 5, 6
    cat = [glyph(c, 10), glyph(a, 16), glyph(t, 22)]
    toe = [glyph(t, 40), glyph(o, 46), glyph(e, 52)]
    office = [glyph(o, 70), glyph(ffi, 76), glyph(c, 82), glyph(e, 88)]
    labels = _vote_labels(
        {0: [*cat, *toe, *office]},
        {0: [word("cat", cat, list("cat")), word("toe", toe, list("toe")),
             word("office", office, list("office"))]},
    )
    assert labels[(1, o)].text == "o"
    assert labels[(1, ffi)].text == "ffi"


def _roles(lines: list[list[tuple[str, int]]], labels: dict, shapes=None) -> dict:
    page = [_line for glyphs in (_set(words, labels, baseline=100 + 14 * row) for row, words in enumerate(lines))
            for _line in _build_lines(glyphs, labels, {})]
    roles, _doubtful = _learn_roles({0: page}, labels, shapes or (lambda keys: {}))
    return roles


def test_letters_that_never_join_a_word_are_variables():
    """数学斜体的 p、s 只单独出现；正文的 p、s 在 price、sales 里。冠词 a 和词里的 a 是同一张位图。"""
    labels: dict = {}
    sentence = [("each", ROMAN), ("price", ROMAN), ("of", ROMAN), ("part", ROMAN), ("p", MATH),
                ("and", ROMAN), ("sales", ROMAN), ("s", MATH), ("is", ROMAN), ("a", ROMAN), ("cost", ROMAN)]
    roles = _roles([sentence] * 4, labels)
    assert roles[(MATH, ord("p"))] == "math"
    assert roles[(MATH, ord("s"))] == "math"
    assert roles[(ROMAN, ord("p"))] == "text"
    assert roles[(ROMAN, ord("a"))] == "text"


def test_ligature_and_short_word_capitals_count_as_text():
    """ff 认得不稳，W 只出现在 We 里。同词的其他字母是正文，它们也是。"""
    labels: dict = {(ROMAN, ord("F")): Label("ff", 0.6)}
    labels[(ROMAN, ord("W"))] = Label("W", 1.0)
    sentence = [("We", ROMAN), ("diFer", ROMAN), ("from", ROMAN), ("others", ROMAN)]
    roles = _roles([sentence] * 3, labels)
    assert roles[(ROMAN, ord("F"))] == "text"
    assert roles[(ROMAN, ord("W"))] == "text"


def test_plural_of_a_variable_is_not_a_word():
    """the v’s：v 后面跟着正文的 s，仍是变量。"""
    labels: dict = {}
    roles = _roles([[("the", ROMAN), ("v’s", MATH), ("and", ROMAN), ("v", MATH), ("only", ROMAN)]] * 3, labels)
    assert roles[(MATH, ord("v"))] == "math"


def test_unsteady_digit_is_a_number_only_inside_numbers():
    """粗体 1 认得不稳，但在 1.1、1.2 里是节号。b 被认成 6，后面跟逗号也不算数。"""
    labels: dict = {(ROMAN, ord("1")): Label("1", 0.75), (MATH, ord("6")): Label("6", 0.6)}
    roles = _roles([[("1.1", ROMAN), ("The", ROMAN), ("cube", ROMAN), ("6,", MATH), ("then", ROMAN)]] * 3, labels)
    assert roles[(ROMAN, ord("1"))] == "neutral"
    assert roles[(MATH, ord("6"))] == "symbol"


def test_article_a_starts_a_sentence_and_variable_a_does_not():
    """长得像正文 A 的单字母：句首是冠词，句中是变量。斜体正文和数学斜体光看位图分不出来。"""
    labels: dict = {}
    same = lambda keys: {key: np.ones(4) / 2 for key in keys}  # noqa: E731
    lines = [
        [("data.", ROMAN), ("A", 4), ("common", ROMAN), ("Also", ROMAN)],
        [("change", ROMAN), ("to", ROMAN), ("A", MATH), ("but", ROMAN), ("Also", ROMAN)],
    ] * 2
    roles = _roles(lines, labels, same)
    assert roles[(4, ord("A"))] == "text"
    assert roles[(MATH, ord("A"))] == "math"


def test_minus_is_math_and_dashes_in_prose_are_not():
    """OCR 把减号、破折号都认成 —。减号宽 0.6 个字号，长破折号 1 个字号。"""
    labels: dict = {(5, 1): Label("—", 1.0), (5, 2): Label("—", 1.0)}
    glyphs = _set([("cost", ROMAN), ("of", ROMAN), ("views", ROMAN)], labels)
    x = glyphs[-1].rect.x1 + 4
    em = Glyph((5, 2), pymupdf.Rect(x, 96.5, x + 10, 97))
    glyphs += [em, *_set([("whole", ROMAN), ("view", ROMAN), ("C", MATH)], labels, x=x + 14)]
    x = glyphs[-1].rect.x1 + 4
    minus = Glyph((5, 1), pymupdf.Rect(x, 96.5, x + 6, 97))
    glyphs += [minus, *_set([("C", MATH), ("then", ROMAN)], labels, x=x + 10)]
    page = [line for line in _build_lines(glyphs, labels, {})]
    roles, _doubtful = _learn_roles({0: page * 3}, labels, lambda keys: {})
    assert roles[(5, 1)] == "symbol"
    assert roles[(5, 2)] == "neutral"


def test_filled_square_at_line_starts_is_a_bullet():
    """项目符号总在行首单独一个，墨迹填满方框。OCR 认成 e 也不信。"""
    labels: dict = {(6, 1): Label("e", 1.0)}
    lines = []
    for row in range(3):
        dot = Glyph((6, 1), pymupdf.Rect(72, 94 + 14 * row, 76, 98 + 14 * row))
        words = _set([("For", ROMAN), ("each", ROMAN), ("view", ROMAN)], labels, x=82, baseline=100 + 14 * row)
        lines.extend(_build_lines([dot, *words], labels, {}))
    found = _bullets({0: lines}, labels, lambda keys: {key: 0.78 if key == (6, 1) else 0.3 for key in keys})
    assert found == {(6, 1)}


def test_line_follows_baselines_not_the_fraction_order():
    """内容流先画分母（笔头退回分数左边），再接着写正文。正文仍在同一行、分数之后。"""
    labels: dict = {}
    head = _set([("we", ROMAN), ("conclude", ROMAN), ("that", ROMAN), ("a", MATH), ("=", MATH)], labels)
    x = head[-1].rect.x1 + 2
    numerator = _set([("k", MATH)], labels, x=x, baseline=96, size=7)
    denominator = _set([("k", MATH), ("1", ROMAN)], labels, x=x - 1, baseline=104, size=7)
    tail = _set([("For", ROMAN), ("these", ROMAN), ("values", ROMAN)], labels, x=x + 14)
    labels[(MATH, ord("="))] = Label("=", 1.0)
    lines = _build_lines([*head, *numerator, *denominator, *tail], labels, {})
    assert len(lines) == 1
    text = _text(lines[0], labels)
    assert text.startswith("we conclude that a")
    assert text.endswith("For these values")


def test_lowered_full_size_letters_are_a_subscript():
    """N 下面的 supplier 和正文一样大，只是整体往下沉。"""
    labels: dict = {}
    line = _build_lines([
        *_set([("where", ROMAN), ("N", MATH)], labels),
        *_set([("supplier", 4)], labels, x=110, shift=2.6),
        *_set([("is", ROMAN), ("the", ROMAN)], labels, x=150),
    ], labels, {})[0]
    reader = _reader(labels, {key: "text" for key in labels})
    sub = [glyph for glyph in line.glyphs if glyph.key[0] == 4]
    plain = [glyph for glyph in line.glyphs if glyph.key[0] == ROMAN]
    assert all(_is_script(glyph, line, reader) for glyph in sub)
    assert not any(_is_script(glyph, line, reader) for glyph in plain)


def test_variable_with_a_typewriter_subscript_stays_a_formula():
    """N 的下标 supplier 是等宽字。下标不让 N 变成等宽字，也不让 N 像单词。"""
    labels: dict = {}
    glyphs = [
        *_set([("where", ROMAN), ("N", MATH)], labels),
        *_set([("supplier", TYPE)], labels, x=110, size=7.5, shift=2.0),
        *_set([("is", ROMAN), ("the", ROMAN)], labels, x=150),
    ]
    line = _build_lines(glyphs, labels, {})[0]
    roles = {key: "text" for key in labels}
    roles[(MATH, ord("N"))] = "math"
    mono = {key: 0.525 * 7.5 for key in labels if key[0] == TYPE}
    styled = _styled(line, _reader(labels, roles, mono))
    assert styled.startswith("where [N][s][u][p][p][l][i][e][r] is")


def test_formula_keeps_inner_commas_and_brackets_and_leaves_the_period():
    labels: dict = {}
    glyphs = _set([("cell", ROMAN), ("(p,", MATH), ("s,", MATH), ("c).", MATH), ("The", ROMAN),
                   ("(i", MATH), ("+", MATH), ("1)st", ROMAN), ("side", ROMAN),
                   ("(using", ROMAN), ("d)", MATH), ("and", ROMAN), ("k", MATH), ("=", MATH), ("2.", ROMAN),
                   ("size", ROMAN), ("S.", MATH), ("c", MATH), ("gives", ROMAN)], labels)
    roles = {key: "text" for key in labels}
    roles.update({key: "neutral" for key in labels if not labels[key].text.isalpha()})
    roles.update({(MATH, ord(ch)): "math" for ch in "pscidkS"})
    roles[(MATH, ord("+"))] = roles[(MATH, ord("="))] = "symbol"
    line = _build_lines(glyphs, labels, {})[0]
    styled = _styled(line, _reader(labels, roles))
    assert "[(][p][,] [s][,] [c][)]." in styled
    assert "[(][i] [+] [1][)]st" in styled
    assert "(using [d])" in styled
    assert "[k] [=] [2]." in styled
    assert "size [S]. [c] gives" in styled
    item = _build_lines(_set([("1.", ROMAN), ("<", MATH), ("is", ROMAN), ("a", ROMAN), ("partial", ROMAN)], labels,
                             baseline=130), labels, {})[0]
    assert _styled(item, _reader(labels, roles)).startswith("1. [<] is")


def test_comma_outside_brackets_separates_two_formulas():
    """⟨L, ⪯⟩, a ≺ b：括号外的逗号是句子的，两边是两个公式。"""
    labels: dict = {}
    glyphs = _set([("lattice", ROMAN), ("(L,", MATH), ("<),", MATH), ("a", MATH), ("<", MATH), ("b", MATH),
                   ("means", ROMAN)], labels)
    labels[(MATH, ord("<"))] = Label("<", 1.0)
    roles = {key: "text" for key in labels}
    roles.update({key: "neutral" for key in labels if not labels[key].text.isalpha()})
    roles.update({(MATH, ord(ch)): "math" for ch in "Lab"})
    roles[(MATH, ord("<"))] = "symbol"
    line = _build_lines(glyphs, labels, {})[0]
    styled = _styled(line, _reader(labels, roles))
    assert "[(][L][,] [<][)], [a] [<] [b] means" in styled
    enumeration = _build_lines(_set([("for", ROMAN), ("a", MATH), ("=", MATH), ("1,", ROMAN), ("2,", ROMAN), ("b.", MATH),
                                     ("The", ROMAN), ("an", ROMAN), ("b-dimensional", ROMAN)], labels,
                                    baseline=130), labels, {})[0]
    roles.update({key: "text" for key in labels if key[0] == ROMAN and labels[key].text.isalpha()})
    roles.update({key: "neutral" for key in labels if key[0] == ROMAN and not labels[key].text.isalpha()})
    roles[(MATH, ord("="))] = "symbol"
    roles[(MATH, ord("b"))] = "math"
    styled = _styled(enumeration, _reader(labels, roles))
    assert "for [a] [=] [1][,] [2][,] [b]. The" in styled
    views = _build_lines(_set([("views", ROMAN), ("d,", MATH), ("e,", MATH), ("and", ROMAN), ("b", MATH)], labels,
                              baseline=145), labels, {})[0]
    roles.update({key: "text" for key in labels if key[0] == ROMAN and labels[key].text.isalpha()})
    roles.update({(MATH, ord(ch)): "math" for ch in "de"})
    assert _styled(views, _reader(labels, roles)) == "views [d], [e], and [b]"
    head = _set([("an", ROMAN), ("n", MATH)], labels, baseline=160)
    tail = _set([("-dimensional", ROMAN)], labels, x=head[-1].rect.x1 + 0.5, baseline=160)
    cube = _set([("cube", ROMAN)], labels, x=tail[-1].rect.x1 + 4, baseline=160)
    roles.update({key: "text" for key in labels if key[0] == ROMAN and labels[key].text.isalpha()})
    roles[(ROMAN, ord("-"))] = "neutral"
    roles[(MATH, ord("n"))] = "math"
    line = _build_lines([*head, *tail, *cube], labels, {})[0]
    assert _styled(line, _reader(labels, roles)) == "an [n]-dimensional cube"


def test_typewriter_letters_are_learned_from_even_words_with_narrow_letters():
    """等宽词里 l、逗号也占满一格。latter、each 字距也匀，但只有 0.4 个字号，不算。"""
    labels: dict = {}
    words = [("TotalSales", TYPE), ("supplier", TYPE), ("Customer,", TYPE), ("latter", ROMAN), ("each,", ROMAN)]
    page = [_line for row in range(2) for _line in _build_lines(_set(words, labels, baseline=100 + 14 * row), labels, {})]
    mono = _mono_kinds({0: page}, labels)
    assert (TYPE, ord("l")) in mono and (TYPE, ord(",")) in mono
    assert (ROMAN, ord("l")) not in mono and (ROMAN, ord("e")) not in mono


def test_lines_join_by_baseline_even_when_their_boxes_overlap():
    """行内的高括号把行框撑到上一行里，按基线距仍是同一段。"""
    def line(y: float, top: float) -> dict:
        span = {"text": "word word", "font": "Times-Roman", "size": 10.0, "origin": (72, y),
                "bbox": (72, top, 400, y + 2.5)}
        return {"spans": [span], "bbox": (72, top, 400, y + 2.5)}

    groups = _group_lines([line(100, 92), line(113.5, 99), line(160, 152)])
    assert [len(group) for group in groups] == [2, 1]


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


def test_inserted_text_is_not_treated_as_bitmap_font():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a normal paragraph with enough letters to translate.", fontsize=11)
    data = page.get_text("dict")
    assert broken_type3_names(doc, page) == set()
    assert recover_type3_page(page, data) is False


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


def test_short_numbered_list_item_is_not_a_formula_line():
    assert _numbered_list_item("5. part (0.2M)")
    assert _numbered_list_item("8. none (1)")
    assert not _numbered_list_item("5.")
    assert not _numbered_list_item("(p, ALL, c)")


_LETTERS = "bdhklbdhkl"


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


def _fake_ocr(path: Path, scale: float = 3.0) -> list[OcrWord]:
    """每行一个词：字形宽 4.8、间距 6，基线在 50 + 16 × 行号。"""
    words = []
    for row in range(6):
        baseline = 50 + 16 * row
        chars = tuple(
            OcrChar(_LETTERS[i % 10], pymupdf.Rect(40 + 6 * i, baseline - 7.2, 44.8 + 6 * i, baseline), 96.0)
            for i in range(30)
        )
        words.append(OcrWord(_LETTERS * 3, pymupdf.Rect(40, baseline - 7.2, 220, baseline), 96.0, chars))
    return words


def test_bitmap_page_reads_back_its_letters(tmp_path, monkeypatch):
    """OCR 只用来给字形定名；字的位置、行、字号都来自点阵字本身。"""
    monkeypatch.setattr(pdf_type3, "ocr_image", _fake_ocr)
    monkeypatch.setattr(pdf_type3.shutil, "which", lambda name: "/usr/bin/" + name)
    src = tmp_path / "bitmap.pdf"
    _bitmap_font_pdf(src)
    blocks = extract_blocks(pymupdf.open(src))
    first = [block for block in blocks if block.page == 0]
    assert first and first[0].text.startswith(_LETTERS * 3)
    assert first[0].rect.y0 == pytest.approx(42.8, abs=1.0)
    assert any("normal paragraph" in block.text for block in blocks if block.page == 1)


@pytest.mark.parametrize("failure", ["no_words", "timeout"])
def test_unreadable_bitmap_page_keeps_its_glyphs(tmp_path, monkeypatch, failure):
    """一页点阵字识别超时或认不出字时，这一页留原样，其他页照常翻译，不让整本书失败。"""
    def broken(path, scale=3.0):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("tesseract", 90)
        return []

    monkeypatch.setattr(pdf_type3, "ocr_image", broken)
    monkeypatch.setattr(pdf_type3.shutil, "which", lambda name: "/usr/bin/" + name)
    src = tmp_path / "bitmap.pdf"
    _bitmap_font_pdf(src)
    doc = pymupdf.open(src)
    assert broken_type3_names(doc, doc[0]) == {"BitmapF"}
    blocks = extract_blocks(doc)
    assert [block.page for block in blocks] == [1]
    assert "normal paragraph" in blocks[0].text


def _paper_pages(*numbers: int) -> pymupdf.Document:
    assert PAPER is not None
    src = pymupdf.open(PAPER)
    doc = pymupdf.open()
    for number in numbers:
        doc.insert_pdf(src, from_page=number - 1, to_page=number - 1)
    return doc


@needs_paper
@needs_tesseract
def test_paper_reads_as_english_with_exact_words():
    blocks = extract_blocks(_paper_pages(1, 2, 3))
    texts = [block.text for block in blocks]
    joined = " ".join(texts)
    assert any(text.startswith("Implementing Data Cubes Efficiently") for text in texts)
    assert "Venky Harinarayan" in texts and "Anand Rajaraman" in texts
    assert "1 Introduction" in texts and "1.1 The Data Cube" in texts
    assert "BusinessObjects [X94]" in joined
    assert "differ only in the GROUP-BY clause" in joined
    # 旧做法把数学斜体的 R 认成 FR 送去翻译。现在它是公式，贴回原字形
    here = next(block for block in blocks if "refers to the raw-data relation" in block.text)
    spans = [span for line in here.span_lines for span in line]
    index = next(i for i, span in enumerate(spans) if span["text"].endswith("Here, "))
    assert spans[index + 1]["font"] == "CMMI10"


@needs_paper
@needs_tesseract
def test_paper_variables_are_formulas_and_sql_is_code():
    blocks = extract_blocks(_paper_pages(2, 3, 12, 18))
    sales = next(block for block in blocks if "total sales of part" in block.text)
    math = {span["text"].strip() for line in sales.span_lines for span in line if span["font"] == "CMMI10"}
    assert {"p", "s", "c"} <= math
    sql = next(block for block in blocks if block.text.startswith("SELECT"))
    assert _code_block(sql)
    conclude = next(block for block in blocks if "we conclude that" in block.text)
    assert conclude.text.index("we conclude that") < conclude.text.index("For these values")
