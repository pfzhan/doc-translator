"""Build the output files: side-by-side pages, the translated-only version cut from the
bilingual one, translated bookmarks, and font subsetting."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

from .pdf_blocks import _NUMBER_LABEL_RE, TextBlock, _pages_to_write
from .pdf_dedupe import merge_duplicates
from .pdf_flow import (
    strip_style_marks,
)
from .pdf_parts import Render, plan_parts, render_in_parts, worker_count
from .pdf_types import PdfDict

if TYPE_CHECKING:
    from multiprocessing.queues import Queue


def _render_by_parts(
    src_doc: pymupdf.Document,
    blocks: list[TextBlock],
    translations: list[str],
    continuations: dict[int, TextBlock],
    render: Render,
    on_page: Callable[[int, int], None],
) -> pymupdf.Document:
    """Write the translations back in several parts in parallel. A paragraph carried over to the next page
    keeps its host block and its tail in the same part. With rotated pages or few pages, the book is one part."""
    weights = {pno: len(items) for pno, items in _pages_to_write(blocks, translations).items()}
    page_of = {id(b): b.page for b in blocks}
    links = [(page_of[host], tail.page) for host, tail in continuations.items() if host in page_of]
    parts = (
        [list(weights)]
        if any(page.rotation for page in src_doc.pages())
        else plan_parts(weights, links, worker_count())
    )
    return render_in_parts(render, parts, on_page) or render(None, on_page)


def _render_side_by_side(src: pymupdf.Document, translated: pymupdf.Document) -> pymupdf.Document:
    out = pymupdf.open()
    for i in range(len(src)):
        w, h = src[i].rect.width, src[i].rect.height
        # A translated page may be lengthened to hold a paragraph carried over from the page before.
        # The side-by-side page grows with it so that the extra lines stay visible.
        th = translated[i].rect.height
        page = out.new_page(width=w * 2, height=max(h, th))
        page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, i)
        page.show_pdf_page(pymupdf.Rect(w, 0, w * 2, th), translated, i)
    return out


def translated_from_bilingual(src: Path, bilingual: Path, preview: PdfDict | None) -> bytes | None:
    """Build the translated-only PDF from the right halves of the bilingual one, in a few seconds.

    Rendering a book again takes minutes. The translated pages go into a copy of the source, which keeps
    the links, named destinations, page labels, bookmarks and metadata, so the result matches a fresh
    render. Bookmarks are translated from preview, the job's preview snapshot (see translate_bookmarks).
    Returns None, leaving the job to a full render, when a page is rotated, a media box does not start at
    the origin, a page has a crop box, a translated page differs in size from its source page, or the
    source has bookmarks and there is no preview snapshot."""
    src_doc, both = pymupdf.open(src), pymupdf.open(bilingual)
    if len(src_doc) != len(both) or (preview is None and src_doc.get_toc()):
        return None
    forms = []
    for page, side in zip(src_doc.pages(), both.pages()):
        box = page.mediabox
        if (
            page.rotation
            or box.x0
            or box.y0
            or page.cropbox != box
            or abs(side.rect.width - 2 * box.width) > 0.01
            or abs(side.rect.height - box.height) > 0.01
        ):
            return None
        form = _translated_form(both, side, box)
        if form is None:
            return None
        forms.append(form)
    # Keep the translated half of each side-by-side page: move the media box to the right half and draw only
    # the translated page. Merging may have left resources and content streams shared between pages, so each
    # page gets new ones.
    shown = both.get_new_xref()
    both.update_object(shown, "<<>>")
    both.update_stream(shown, b"q /T Do Q")
    for page, side, form in zip(src_doc.pages(), both.pages(), forms):
        both.xref_set_key(side.xref, "Resources", f"<</XObject<</T {form} 0 R>>>>")
        both.xref_set_key(side.xref, "Contents", f"{shown} 0 R")
        side.set_mediabox(pymupdf.Rect(page.rect.width, 0, 2 * page.rect.width, page.rect.height))
    blank = src_doc.get_new_xref()
    src_doc.update_object(blank, "<<>>")
    src_doc.update_stream(blank, b"")
    for i, page in enumerate(src_doc.pages()):
        src_doc.xref_set_key(page.xref, "Resources", "<<>>")
        src_doc.xref_set_key(page.xref, "Contents", f"{blank} 0 R")
        page.show_pdf_page(page.rect, both, i)
    translate_bookmarks(src_doc, preview)
    # The bilingual file is already merged and subset, so only drop objects the source no longer uses.
    # garbage=3 would compare duplicates again, which takes a minute on a book.
    out: bytes = src_doc.tobytes(garbage=2, deflate=True)
    return out


def translate_bookmarks(doc: pymupdf.Document, preview: PdfDict | None) -> None:
    """Replace the bookmark titles of the translated-only PDF with the translations of the matching headings.
    preview is the result of Runner.preview(-1).

    A bookmark takes the translation of a segment whose source text equals its title, ignoring section
    numbers and case. When a title was translated more than once (in the table of contents and in the body),
    the translation that appears on the target page or the page after it is used: hyperref anchors often sit
    at the bottom of the page before the heading. Titles with no match, or whose translation holds a formula,
    are left as they are. A section number at the start of the translation is dropped when the source
    bookmark has none."""
    toc = doc.get_toc()
    if not toc or not preview:
        return
    latest: dict[int, str | None] = {}
    for i, text, kept in preview.get("updates") or []:
        latest[i] = None if kept else text
    choices: dict[str, list[str]] = {}
    for i, segment in enumerate(preview.get("segments") or []):
        text = latest.get(i)
        if text is None:
            continue
        text = strip_style_marks(text).strip()
        if text and text != segment["s"].strip() and not re.search(r"[\x00-\x1f]|\{[a-z]\d+\}", text):
            choices.setdefault(_bookmark_key(segment["s"]), []).append(text)
    page_texts: dict[int, str] = {}

    def on_page(pno: int, text: str) -> bool:
        if not 1 <= pno <= len(doc):
            return False
        if pno not in page_texts:
            page_texts[pno] = re.sub(r"\s+", "", doc[pno - 1].get_text())
        return re.sub(r"\s+", "", text) in page_texts[pno]

    for xref, (_, title, pno) in zip(doc.get_outline_xrefs(), toc):
        found = list(dict.fromkeys(choices.get(_bookmark_key(title), ())))
        if len(found) > 1:
            found = [text for q in (pno, pno + 1) for text in found if on_page(q, text)]
        if not found:
            continue
        text = found[0]
        if not _NUMBER_LABEL_RE.match(title.strip()) and (number := _NUMBER_LABEL_RE.match(text)):
            text = text[number.end() :].strip()
        doc.xref_set_key(xref, "Title", pymupdf.get_pdf_str(text))


def _bookmark_key(text: str) -> str:
    text = strip_style_marks(text).strip().replace("’", "'")
    if number := _NUMBER_LABEL_RE.match(text):
        text = text[number.end() :]
    return " ".join(text.split()).casefold()


def _translated_form(doc: pymupdf.Document, page: pymupdf.Page, box: pymupdf.Rect) -> int | None:
    """The Form XObject that draws the translated page on a side-by-side page: the one show_pdf_page placed
    on the right half, with the size of the source page."""
    kind, value = doc.xref_get_key(page.xref, "Resources/XObject")
    if kind != "dict":
        return None
    found = []
    for xref in map(int, re.findall(r"(\d+) 0 R", value)):
        matrix = [float(v) for v in doc.xref_get_key(xref, "Matrix")[1].strip("[]").split()]
        bbox = [float(v) for v in doc.xref_get_key(xref, "BBox")[1].strip("[]").split()]
        if (
            len(matrix) == 6
            and len(bbox) == 4
            and matrix[:4] == [1, 0, 0, 1]
            and abs(matrix[4] - box.width) < 0.01
            and matrix[5] == 0
            and all(abs(a - b) < 0.01 for a, b in zip(bbox, (box.x0, box.y0, box.x1, box.y1)))
        ):
            found.append(xref)
    return found[0] if len(found) == 1 else None


def _subset_worker(data: bytes, queue: "Queue[bytes]") -> None:
    doc = pymupdf.open("pdf", data)
    doc.subset_fonts()
    queue.put(doc.tobytes(garbage=1, deflate=True))


def _subset_fonts_safe(data: bytes) -> bytes:
    """Subset the fonts in a child process, as BabelDOC does: PyMuPDF subsetting sometimes crashes or hangs.
    With no result after 60 seconds the file is used without subsetting; it is larger but displays the same.
    The fork context is used because spawn imports the main module again, which fails under stdin or a REPL.
    Platforms without fork (Windows) subset in the current process."""
    import multiprocessing

    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        try:
            doc = pymupdf.open("pdf", data)
            doc.subset_fonts()
            subset: bytes = doc.tobytes(garbage=1, deflate=True)
            return subset
        except Exception:  # noqa: BLE001
            return data
    queue: Queue[bytes] = ctx.Queue()
    proc = ctx.Process(target=_subset_worker, args=(data, queue))
    proc.start()
    try:
        out: bytes = queue.get(timeout=60)
        proc.join(timeout=5)
        return out
    except Exception:  # noqa: BLE001 - fall back on a timeout or a crashed child process
        proc.kill()
        return data


def _drop_mupdf_cache() -> None:
    """Empty the MuPDF resource store. It holds fonts and images loaded by earlier steps that later steps do
    not use, and once it fills up every later step slows down: on ed3book, merging duplicate objects takes
    more than twice as long and subsetting three times as long. Writer processes forked from the main process
    inherit the store as well. It is emptied before writing, before building side-by-side pages and before
    compacting."""
    pymupdf.TOOLS.store_shrink(100)


def _compact(doc: pymupdf.Document) -> bytes:
    """Merge objects with identical content, subset the fonts and return the bytes to write.

    insert_htmlbox embeds a full font every time, and the bilingual file holds the images and fonts of the
    source twice, once in each half. MuPDF's garbage=4 also merges, but it compares every pair of objects
    and takes over ten minutes on a book of a few hundred pages. Nothing is compressed before merging:
    the duplicate fonts add up to hundreds of MB, and most of them would be merged away after being
    compressed. Compressing afterwards saves 20 seconds on a 25-page paper.
    """
    doc = pymupdf.open("pdf", doc.tobytes(garbage=2))
    merge_duplicates(doc)
    return _subset_fonts_safe(doc.tobytes(garbage=1, deflate=True))
