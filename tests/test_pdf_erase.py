import pymupdf

from app.formats.pdf_erase import Erasure, erase
from app.formats.pdf_types import PdfDict


def _spans(page: pymupdf.Page) -> list[PdfDict]:
    return [
        span
        for block in page.get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    ]


def _words(page: pymupdf.Page) -> list[str]:
    return [word[4] for word in page.get_text("words")]


def _ink(page: pymupdf.Page, rect: pymupdf.Rect) -> int:
    pix = page.get_pixmap(dpi=300, clip=rect, colorspace=pymupdf.csGRAY)
    return sum(1 for value in pix.samples if value < 128)


def _erase_spans(page: pymupdf.Page, spans: list[PdfDict], line_rects: list[pymupdf.Rect] | None = None) -> int:
    erasure = Erasure()
    erasure.add_block([spans], line_rects or [pymupdf.Rect(spans[0]["bbox"])])
    return erase(page, erasure)


def test_a_neighbor_reaching_into_the_line_box_is_kept() -> None:
    """Descenders of the line above and an adjacent label reaching into the line box are not glyphs of this line.
    Erasing by rectangle would remove them too."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "translate me", fontsize=12)
    page.insert_text((140, 74), "x2", fontsize=7)
    page.insert_text((50, 69), "gyp", fontsize=12)
    spans = _spans(page)
    target = next(span for span in spans if span["text"] == "translate me")
    box = pymupdf.Rect(target["bbox"])
    # The line box is wider than the text and covers the label and the descenders of the line above
    _erase_spans(page, [target], [pymupdf.Rect(box.x0, box.y0 - 4, box.x1 + 20, box.y1)])
    assert _words(page) == ["x2", "gyp"]


def test_bold_drawn_twice_is_erased_with_its_span() -> None:
    """Simulated bold draws the same glyph again slightly offset. PyMuPDF reports only one, and both must be erased."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "Bold", fontsize=12)
    page.insert_text((50.3, 80), "Bold", fontsize=12)
    spans = _spans(page)
    assert _erase_spans(page, spans[:1]) == 8
    assert _ink(page, pymupdf.Rect(40, 60, 120, 90)) == 0


def test_glyphs_are_found_through_the_content_transform() -> None:
    """The content stream scales with cm before drawing text, so origins must be multiplied by the CTM to match
    PyMuPDF's coordinates."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 60), "Hi there", fontsize=12)
    xref = page.get_contents()[0]
    doc.update_stream(xref, b"q 1 0 0 1 20 -30 cm " + doc.xref_stream(xref) + b" Q")
    page = doc.reload_page(page)
    assert _erase_spans(page, _spans(page)) == 8
    assert _words(page) == []


def test_bitmap_glyphs_are_erased_by_their_origins() -> None:
    """Font boxes of bitmap glyphs are unreliable, so the span carries origins and is matched by point. Glyphs outside
    the box stay."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "ab", fontsize=12)
    page.insert_text((150, 80), "cd", fontsize=12)
    origins = [char[2] for span in page.get_texttrace() for char in span["chars"]]
    erasure = Erasure()
    span = {"text": "c", "bbox": (0, 0, 1, 1), "origin": (0, 0), "origins": [origins[2]]}
    erasure.add_block([[span]], [pymupdf.Rect(0, 0, 1, 1)])
    assert erase(page, erasure) == 1
    assert _words(page) == ["ab", "d"]


def test_rules_inside_a_rewritten_line_are_erased() -> None:
    """Old Ghostscript draws \\_ as a small one-pixel-high image. Those in lines being reflowed are erased, others
    stay."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    bar = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 8, 1), False)
    bar.clear_with(0)
    inside = pymupdf.Rect(100, 60, 104, 60.5)
    outside = pymupdf.Rect(100, 150, 104, 150.5)
    page.insert_image(inside, pixmap=bar)
    page.insert_image(outside, pixmap=bar)
    assert _ink(page, inside) > 0 and _ink(page, outside) > 0
    erasure = Erasure()
    erasure.add_block(None, [pymupdf.Rect(80, 50, 200, 62)])
    erase(page, erasure)
    assert _ink(page, inside) == 0
    assert _ink(page, outside) > 0


def test_only_lines_wholly_inside_the_line_box_are_erased() -> None:
    """An inline fraction bar lies entirely within the line and is erased with it. Table rules and figure borders
    crossing the line stay."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.draw_line((100, 58), (120, 58), width=0.4)
    page.draw_line((60, 40), (60, 120), width=0.4)
    erasure = Erasure()
    erasure.add_block(None, [pymupdf.Rect(50, 50, 200, 62)])
    erase(page, erasure)
    drawn = [drawing["rect"] for drawing in page.get_drawings()]
    assert len(drawn) == 1 and drawn[0].x0 == 60


def test_an_underline_below_the_line_box_goes_with_its_words() -> None:
    """A link underline is below the baseline, slightly lower than the bottom of the font box, so line boxes miss
    it. When the text is erased the line must go too; a table rule further down stays."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "Long Papers", fontsize=10)
    page.insert_text((150, 80), "kept", fontsize=10)
    page.insert_text((200, 80), "dNorm", fontsize=10)
    spans = _spans(page)
    box = pymupdf.Rect(spans[0]["bbox"])
    under = box.y1 + 0.3
    page.draw_line((50, under), (75, under), width=0.4)
    page.draw_line((77, under), (105, under), width=0.4)
    page.draw_line((150, under), (170, under), width=0.4)
    page.draw_line((200, under), (228, under), width=0.4)
    page.draw_line((50, 90), (110, 90), width=0.4)
    page = doc.reload_page(page)
    spans = _spans(page)
    # The numerator is formula text and the line below it a fraction bar; the denominator stays, so the line is kept
    erasure = Erasure(math=lambda span: span["text"] == "dNorm")
    numerator = [span for span in spans if span["text"] == "dNorm"]
    erasure.add_block([spans[:1], numerator], [pymupdf.Rect(spans[0]["bbox"]), pymupdf.Rect(numerator[0]["bbox"])])
    erase(page, erasure)
    drawn = sorted((round(d["rect"].x0), round(d["rect"].y0)) for d in page.get_drawings())
    assert drawn == [(50, 90), (150, round(under)), (200, round(under))]


def test_a_form_shared_with_another_page_is_left_alone() -> None:
    """A Form shared by two pages (running header, template) is erased on this page and unchanged on the other."""
    src = pymupdf.open()
    src.new_page(width=300, height=200).insert_text((50, 80), "shared", fontsize=12)
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page(width=300, height=200).show_pdf_page(pymupdf.Rect(0, 0, 300, 200), src, 0)
    erase(doc[0], _erasure_for(doc[0]))
    assert _words(doc[0]) == []
    assert _words(doc[1]) == ["shared"]


def _erasure_for(page: pymupdf.Page) -> Erasure:
    erasure = Erasure()
    erasure.add_block([_spans(page)], [pymupdf.Rect(0, 0, 300, 200)])
    return erasure


def test_a_span_starting_with_a_space_is_found_by_its_glyphs() -> None:
    """A span's origin is that of its first character. When a leading space sits on the baseline of a preceding
    subscript, a little off from the line, using it as the baseline misses the whole paragraph. Per-character
    origins are used instead."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((100, 80), "not converged", fontsize=10)
    [span] = _spans(page)
    span = {**span, "origin": (span["origin"][0], span["origin"][1] + 1.6)}
    assert _erase_spans(page, [span]) == 13
    assert _words(page) == []


def test_shared_type3_glyphs_are_left_alone() -> None:
    """When filtering recursively, MuPDF also enters the glyphs of Type3 fonts. Glyphs are shared by the whole
    document, and deleting from them would remove the same character elsewhere."""
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    glyph = doc.get_new_xref()
    doc.update_object(glyph, "<<>>")
    doc.update_stream(glyph, b"100 0 0 0 100 100 d1 0 0 100 100 re f")
    font = doc.get_new_xref()
    doc.update_object(
        font,
        (
            f"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 100 100] /FontMatrix [0.01 0 0 0.01 0 0] "
            f"/CharProcs << /a {glyph} 0 R >> /Encoding << /Type /Encoding /Differences [97 /a] >> "
            f"/FirstChar 97 /LastChar 97 /Widths [100] /Resources << >> >>"
        ),
    )
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /T3 {font} 0 R >> >>")
    content = doc.get_new_xref()
    doc.update_object(content, "<<>>")
    doc.update_stream(content, b"BT /T3 10 Tf 50 100 Td (a) Tj ET")
    doc.xref_set_key(page.xref, "Contents", f"{content} 0 R")
    before = doc.xref_stream(glyph)
    erasure = Erasure()
    erasure.lines.append(pymupdf.Rect(-1000, -1000, 1000, 1000))
    erase(page, erasure)
    assert doc.xref_stream(glyph) == before


def test_a_color_repeated_in_a_new_color_space_survives() -> None:
    """When the color space changes and the same color value follows, MuPDF omits the color when rewriting, while
    the space change resets the color to black, turning a white background in a figure into a black block. An
    original stream that draws without setting a color still draws the initial color."""
    lab = "[/Lab <</WhitePoint [0.9505 1 1.089]>>]"
    shades: dict[str, set[int]] = {}
    for name, stream in {
        "repeated": b"0 g 0 0 300 200 re f /L1 cs 100 0 0 sc 10 10 50 50 re f /L2 cs 100 0 0 sc 100 10 50 50 re f",
        "bare": b"1 g 0 0 300 200 re f /L1 cs 100 0 0 sc 10 10 50 50 re f /L2 cs 100 10 50 50 re f",
    }.items():
        doc = pymupdf.open()
        page = doc.new_page(width=300, height=200)
        page.insert_text((200, 150), "x")
        doc.update_stream(page.get_contents()[0], stream)
        doc.xref_set_key(page.xref, "Resources", f"<</ColorSpace<</L1 {lab} /L2 {lab}>>>>")
        erasure = Erasure()
        erasure.lines.append(pymupdf.Rect(0, 0, 1, 1))
        erase(page, erasure)
        pix = page.get_pixmap(clip=pymupdf.Rect(120, 160, 130, 170), colorspace=pymupdf.csGRAY)
        shades[name] = set(pix.samples)
    assert min(shades["repeated"]) > 200
    assert max(shades["bare"]) < 50
