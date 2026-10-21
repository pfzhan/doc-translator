import pymupdf

from app.formats.pdf_dedupe import merge_duplicates, replace_refs


def test_references_inside_strings_are_left_alone():
    text = "<</A 12 0 R/B (see 12 0 R \\) 12 0 R)/C <3132>/D [12 0 R 120 0 R]/E<</F 12 0 R>>>>"
    assert replace_refs(text, {12: 5}) == (
        "<</A 5 0 R/B (see 12 0 R \\) 12 0 R)/C <3132>/D [5 0 R 120 0 R]/E<</F 5 0 R>>>>"
    )


def _book_with_a_picture() -> bytes:
    doc = pymupdf.open()
    picture = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), True)
    picture.clear_with(180)
    for _ in range(2):
        page = doc.new_page(width=300, height=200)
        page.insert_image(pymupdf.Rect(20, 20, 120, 120), pixmap=picture)
        page.insert_text((140, 60), "Body text beside the picture.", fontsize=10)
    return doc.tobytes()


def _side_by_side(data: bytes) -> pymupdf.Document:
    """双语版的拼法：左边原书，右边原书的副本。"""
    left, right = pymupdf.open("pdf", data), pymupdf.open("pdf", data)
    out = pymupdf.open()
    for index in range(len(left)):
        page = out.new_page(width=600, height=200)
        page.show_pdf_page(pymupdf.Rect(0, 0, 300, 200), left, index)
        page.show_pdf_page(pymupdf.Rect(300, 0, 600, 200), right, index)
    return pymupdf.open("pdf", out.tobytes(garbage=2, deflate=True))


def _images(doc: pymupdf.Document) -> int:
    return sum(1 for xref in range(1, doc.xref_length()) if doc.xref_get_key(xref, "Subtype")[1] == "/Image")


def test_copies_of_the_same_picture_and_font_become_one():
    doc = _side_by_side(_book_with_a_picture())
    before = [page.get_pixmap(dpi=72).samples for page in doc]
    # 每份是一张图加它的透明蒙版。蒙版先合并，两张图的字典才一样
    assert _images(doc) == 4
    assert merge_duplicates(doc) > 0
    merged = pymupdf.open("pdf", doc.tobytes(garbage=1, deflate=True))
    assert _images(merged) == 2
    # 合并只改引用。流对象的字典改过以后流还在，页面画出来和原来一样
    assert [page.get_pixmap(dpi=72).samples for page in merged] == before


def test_pages_with_the_same_content_stay_two_pages():
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page(width=300, height=200).insert_text((40, 60), "Same text.", fontsize=10)
    doc = pymupdf.open("pdf", doc.tobytes())
    merge_duplicates(doc)
    merged = pymupdf.open("pdf", doc.tobytes(garbage=1))
    assert merged.page_count == 2 and merged[0].xref != merged[1].xref
    assert [page.get_text().strip() for page in merged] == ["Same text.", "Same text."]
