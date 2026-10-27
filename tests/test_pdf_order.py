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


def test_a_label_inside_a_figure_goes_right_after_the_figure(tmp_path):
    """插图是一个 Form，标签的字在 Form 里，顶层流里放不了标记：译文插在画插图的 Do 后面，紧跟着插图读。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    form = doc.get_new_xref()
    doc.update_object(form, f"<< /Type /XObject /Subtype /Form /BBox [30 230 300 300] "
                            f"/Resources << /Font << /F1 {font} 0 R >> >> >>")
    doc.update_stream(form, b"BT /F1 10 Tf 40 280 Td (Figure) Tj 0 -40 Td (delta epsilon zeta) Tj ET\n")
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /F1 {font} 0 R >> /XObject << /Fm0 {form} 0 R >> >>")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(stream, b"BT /F1 10 Tf 40 320 Td (alpha beta gamma) Tj ET\n"
                              b"q /Fm0 Do Q\n"
                              b"BT /F1 10 Tf 40 200 Td (cell one two) Tj ET\n")
    doc.xref_set_key(page.xref, "Contents", f"{stream} 0 R")
    path = tmp_path / "src.pdf"
    doc.save(path)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["甲乙丙", "Figure", "丁戊己", "cell", "one", "two"]


def test_a_fill_counts_only_inside_its_clip(tmp_path):
    """后面画的整页填充套着一个不是矩形的小裁剪，只画在裁剪里，压不到译文，照样插回原处。"""
    clipped = b"q 300 20 m 350 20 l 330 60 l h W n 0.9 g 0 0 400 400 re f Q\n"
    path = _source(tmp_path, _BODY, clipped)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_thin_image_over_the_text_does_not_cover_it(tmp_path):
    """后面用 0.5 点高的内联图片画了一条线压在第一段上（分数线常这样画），压不住字，照样插回原处。"""
    line = b"q 200 0 0 0.5 40 324 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x00 EI Q\n"
    path = _source(tmp_path, _BODY, line)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_glyphs_cut_off_by_a_clip_do_not_belong_to_the_paragraph(tmp_path):
    """前面有一组字落在第一段的位置上，但被裁剪挡在外面看不见：它们不算第一段的字，
    标记不放进那组字的裁剪里，译文照样插回原处。"""
    hidden = b"q 300 0 100 400 re W n BT /F1 10 Tf 40 320 Td (hidden words) Tj ET Q\n"
    path = _source(tmp_path, hidden + _BODY)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_form_is_measured_by_its_own_box_not_a_nested_one_of_the_same_name(tmp_path):
    """后面画的小插图 Fm0 不压字；它里面引用的另一个 Form 也叫 Fm0，范围是整页。
    按页面这一层的 Fm0 算范围，译文照样插回原处。"""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    inner = doc.get_new_xref()
    doc.update_object(inner, "<< /Type /XObject /Subtype /Form /BBox [-400 -400 800 800] >>")
    doc.update_stream(inner, b"0 0 1 1 re f\n")
    outer = doc.get_new_xref()
    doc.update_object(outer, f"<< /Type /XObject /Subtype /Form /BBox [300 20 380 60] "
                             f"/Resources << /XObject << /Fm0 {inner} 0 R >> >> >>")
    doc.update_stream(outer, b"q 1 0 0 1 320 30 cm /Fm0 Do Q\n")
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /F1 {font} 0 R >> /XObject << /Fm0 {outer} 0 R >> >>")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(stream, _BODY + b"q /Fm0 Do Q\n")
    doc.xref_set_key(page.xref, "Contents", f"{stream} 0 R")
    path = tmp_path / "src.pdf"
    doc.save(path)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_shifted_index_entry_still_goes_back_where_it_was(tmp_path, monkeypatch):
    """索引条目往下挪过时按副本写，标记放在原块上，译文照样插回原处。"""
    from dataclasses import replace

    path = _source(tmp_path, _BODY)
    blocks = _blocks(path)
    first = blocks[0]
    monkeypatch.setattr(pdf, "_index_reflow", lambda *_args, **_kwargs: {id(first): replace(first)})
    out = _render_translated(path, blocks, ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]
