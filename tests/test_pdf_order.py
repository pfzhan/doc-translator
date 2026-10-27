import pymupdf

from app.formats import pdf, pdf_order
from app.formats.pdf import TextBlock, _render_translated


def _source(tmp_path, body: bytes, extra: bytes = b""):
    """一页：页眉、两段要翻的字、中间一行表格，全在一个文本对象里，用 Td 换行。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /F1 {font} 0 R >> >>")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(stream, body + extra)
    doc.xref_set_key(page.xref, "Contents", f"{stream} 0 R")
    path = tmp_path / "src.pdf"
    doc.save(path)
    return path


_BODY = (
    b"BT /F1 10 Tf 40 360 Td (Header line) Tj 0 -40 Td (alpha beta gamma) Tj "
    b"0 -40 Td (cell one two) Tj 0 -40 Td (delta epsilon zeta) Tj ET\n"
)


def _blocks(path):
    page = pymupdf.open(path)[0]
    out = []
    for text in ("alpha beta gamma", "delta epsilon zeta"):
        rect = page.search_for(text)[0]
        out.append(TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000000", bold=False))
    return out


def _words(doc):
    return [w[4] for w in doc[0].get_text("words", sort=False)]


def test_translations_go_back_where_the_source_text_was(tmp_path):
    """要翻的两段和页眉、表格在同一个文本对象里：在换行处切开，译文插回原处，按内容流的顺序读是阅读顺序。"""
    path = _source(tmp_path, _BODY)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    words = _words(out)
    assert words == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_moved_translations_keep_their_place_on_the_page(tmp_path, monkeypatch):
    """原页带坐标变换时，插回中间的译文和画在最后时位置一样，原页没翻的字也不动。"""
    path = _source(tmp_path, b"q 1 0 0 1 20 -10 cm 0.5 0 0 0.5 0 0 cm 2 0 0 2 0 0 cm\n" + _BODY + b"Q\n")

    def boxes():
        out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
        return {w[4]: tuple(round(v, 1) for v in w[:4]) for w in out[0].get_text("words")}, _words(out)

    moved, order = boxes()
    monkeypatch.setattr(pdf, "ReadingOrder", lambda page, blocks: pdf_order.ReadingOrder(page, blocks, enabled=False))
    at_end, _ = boxes()
    assert moved == at_end
    assert order == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_translation_covered_by_later_drawing_stays_on_top(tmp_path):
    """原页在后面画了一块不透明的底色压在第一段上：这一段挪进去会被盖住，照旧画在最后；第二段照常插回原处。"""
    path = _source(tmp_path, _BODY, b"0.9 0.9 0.9 rg 30 310 300 30 re f\n")
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "cell", "one", "two", "丁戊己", "甲乙丙"]
    page = out[0]
    # 字是黑的，画在底色上面
    pix = page.get_pixmap(dpi=144, clip=page.search_for("甲乙丙")[0])
    assert min(pix.samples) < 100


_SPLIT = (
    b"BT /F1 10 Tf 40 320 Td (alpha beta gamma) Tj ET\n"
    b"0.9 0.9 0.9 rg 30 230 300 30 re f 0 g\n"
    b"BT /F1 10 Tf 40 240 Td (delta epsilon zeta) Tj ET\n"
    b"BT /F1 10 Tf 40 200 Td (cell one two) Tj ET\n"
)


def test_text_drawn_after_a_fill_goes_back_after_the_fill(tmp_path):
    """两段之间画了一块底色，后一段的字原本画在底色上面：两段分开插，后一段插在底色之后，顺序和原文一样。"""
    path = _source(tmp_path, _SPLIT)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["甲乙丙", "丁戊己", "cell", "one", "two"]


def test_a_clip_drawn_as_four_lines_counts_as_a_rectangle(tmp_path):
    """整页套着一个用四条线画的矩形裁剪（Keynote 导出的常这样），盖住了写出的位置，照样插回原处。"""
    clip = b"q 0 0 m 400 0 l 400 400 l 0 400 l h W n\n"
    path = _source(tmp_path, clip + _BODY + b"Q\n")
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]
