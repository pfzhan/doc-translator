from pathlib import Path

import pymupdf
import pytest

from app.formats import pdf_order, pdf_write
from app.formats.pdf_blocks import TextBlock
from app.formats.pdf_write import _render_translated


def _source(tmp_path: Path, body: bytes, extra: bytes = b"") -> Path:
    """One page: a running header, two paragraphs to translate and a table row between them, all in one text object
    with Td line breaks."""
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


def _blocks(path: Path) -> list[TextBlock]:
    page = pymupdf.open(path)[0]
    out: list[TextBlock] = []
    for text in ("alpha beta gamma", "delta epsilon zeta"):
        rect = page.search_for(text)[0]
        out.append(TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000000", bold=False))
    return out


def _words(doc: pymupdf.Document) -> list[str]:
    return [w[4] for w in doc[0].get_text("words", sort=False)]


def test_translations_go_back_where_the_source_text_was(tmp_path: Path) -> None:
    """The two paragraphs share a text object with the header and the table: the object is cut at line breaks and the
    translations are inserted in place, so content stream order is reading order."""
    path = _source(tmp_path, _BODY)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    words = _words(out)
    assert words == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_moved_translations_keep_their_place_on_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a coordinate transform on the original page, translations inserted in the middle land where they would
    when drawn at the end, and untranslated text does not move."""
    path = _source(tmp_path, b"q 1 0 0 1 20 -10 cm 0.5 0 0 0.5 0 0 cm 2 0 0 2 0 0 cm\n" + _BODY + b"Q\n")

    def boxes() -> tuple[dict[str, tuple[float, ...]], list[str]]:
        out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
        return {w[4]: tuple(round(v, 1) for v in w[:4]) for w in out[0].get_text("words")}, _words(out)

    moved, order = boxes()
    monkeypatch.setattr(
        pdf_write, "ReadingOrder", lambda page, blocks: pdf_order.ReadingOrder(page, blocks, enabled=False)
    )
    at_end, _ = boxes()
    assert moved == at_end
    assert order == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_translation_covered_by_later_drawing_stays_on_top(tmp_path: Path) -> None:
    """The original page later draws an opaque background over the first paragraph: moved in, that paragraph would be
    covered, so it is drawn at the end as before; the second paragraph is inserted in place as usual."""
    path = _source(tmp_path, _BODY, b"0.9 0.9 0.9 rg 30 310 300 30 re f\n")
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "cell", "one", "two", "丁戊己", "甲乙丙"]
    page = out[0]
    # The text is black and drawn above the background
    pix = page.get_pixmap(dpi=144, clip=page.search_for("甲乙丙")[0])
    assert min(pix.samples) < 100


_SPLIT = (
    b"BT /F1 10 Tf 40 320 Td (alpha beta gamma) Tj ET\n"
    b"0.9 0.9 0.9 rg 30 230 300 30 re f 0 g\n"
    b"BT /F1 10 Tf 40 240 Td (delta epsilon zeta) Tj ET\n"
    b"BT /F1 10 Tf 40 200 Td (cell one two) Tj ET\n"
)


def test_text_drawn_after_a_fill_goes_back_after_the_fill(tmp_path: Path) -> None:
    """A background is drawn between the two paragraphs, and the second one was drawn above it: the paragraphs are
    inserted separately, the second after the background, in the original order."""
    path = _source(tmp_path, _SPLIT)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["甲乙丙", "丁戊己", "cell", "one", "two"]


def test_a_clip_drawn_as_four_lines_counts_as_a_rectangle(tmp_path: Path) -> None:
    """The whole page is wrapped in a rectangular clip drawn with four lines (common in Keynote exports) that covers
    the written area; the translation is still inserted in place."""
    clip = b"q 0 0 m 400 0 l 400 400 l 0 400 l h W n\n"
    path = _source(tmp_path, clip + _BODY + b"Q\n")
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_label_inside_a_figure_goes_right_after_the_figure(tmp_path: Path) -> None:
    """The figure is a Form with its label text inside, so no marker can go in the top-level stream: the translation
    is inserted after the Do that draws the figure and reads right after it."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    form = doc.get_new_xref()
    doc.update_object(
        form, f"<< /Type /XObject /Subtype /Form /BBox [30 230 300 300] /Resources << /Font << /F1 {font} 0 R >> >> >>"
    )
    doc.update_stream(form, b"BT /F1 10 Tf 40 280 Td (Figure) Tj 0 -40 Td (delta epsilon zeta) Tj ET\n")
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /F1 {font} 0 R >> /XObject << /Fm0 {form} 0 R >> >>")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(
        stream,
        b"BT /F1 10 Tf 40 320 Td (alpha beta gamma) Tj ET\nq /Fm0 Do Q\nBT /F1 10 Tf 40 200 Td (cell one two) Tj ET\n",
    )
    doc.xref_set_key(page.xref, "Contents", f"{stream} 0 R")
    path = tmp_path / "src.pdf"
    doc.save(path)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["甲乙丙", "Figure", "丁戊己", "cell", "one", "two"]


def test_a_fill_counts_only_inside_its_clip(tmp_path: Path) -> None:
    """A later full-page fill sits inside a small non-rectangular clip and paints only there, not over the
    translation, which is still inserted in place."""
    clipped = b"q 300 20 m 350 20 l 330 60 l h W n 0.9 g 0 0 400 400 re f Q\n"
    path = _source(tmp_path, _BODY, clipped)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_thin_image_over_the_text_does_not_cover_it(tmp_path: Path) -> None:
    """A later inline image 0.5pt high draws a line over the first paragraph (fraction bars are often drawn this
    way); it does not cover text, and the translation is still inserted in place."""
    line = b"q 200 0 0 0.5 40 324 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x00 EI Q\n"
    path = _source(tmp_path, _BODY, line)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_glyphs_cut_off_by_a_clip_do_not_belong_to_the_paragraph(tmp_path: Path) -> None:
    """An earlier group of glyphs lies where the first paragraph is but is clipped out of view: they are not glyphs
    of the first paragraph, the marker is not placed inside their clip, and the translation is still inserted in
    place."""
    hidden = b"q 300 0 100 400 re W n BT /F1 10 Tf 40 320 Td (hidden words) Tj ET Q\n"
    path = _source(tmp_path, hidden + _BODY)
    out = _render_translated(path, _blocks(path), ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]


def test_a_form_is_measured_by_its_own_box_not_a_nested_one_of_the_same_name(tmp_path: Path) -> None:
    """A small figure Fm0 drawn later does not cover text; another Form it references is also named Fm0 and spans
    the whole page. The extent is taken from the page-level Fm0, and the translation is still inserted in place."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    inner = doc.get_new_xref()
    doc.update_object(inner, "<< /Type /XObject /Subtype /Form /BBox [-400 -400 800 800] >>")
    doc.update_stream(inner, b"0 0 1 1 re f\n")
    outer = doc.get_new_xref()
    doc.update_object(
        outer,
        f"<< /Type /XObject /Subtype /Form /BBox [300 20 380 60] /Resources << /XObject << /Fm0 {inner} 0 R >> >> >>",
    )
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


def test_a_shifted_index_entry_still_goes_back_where_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A moved-down index entry is written from a copy, with the marker on the original block, and its translation is
    still inserted in place."""
    from dataclasses import replace

    path = _source(tmp_path, _BODY)
    blocks = _blocks(path)
    first = blocks[0]
    monkeypatch.setattr(pdf_write, "_index_reflow", lambda *_args, **_kwargs: {id(first): replace(first)})
    out = _render_translated(path, blocks, ["甲乙丙", "丁戊己"], "zh-CN")
    assert _words(out) == ["Header", "line", "甲乙丙", "cell", "one", "two", "丁戊己"]
