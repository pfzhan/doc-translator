"""Saving the output files (app/formats/pdf_output.py): nothing drawn may go missing on the way."""

from pathlib import Path

import pymupdf

from app.formats.pdf_output import _compact, _render_side_by_side, translated_from_bilingual

# MuPDF renumbering objects (garbage=2 and up) loses one reference of a dictionary of exactly this many entries
FORMS = 101


def _page_of_forms(path: Path) -> None:
    """A page drawing FORMS small Form XObjects (formulas, as the writer pastes them), each different so that
    merging duplicates keeps them apart, listed in reverse order in its resources, plus an object nothing refers
    to, so that saving has something to drop."""
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    names = [f"FmF_{i}" for i in reversed(range(FORMS))]
    entries = []
    for i, name in enumerate(names):
        form = doc.get_new_xref()
        doc.update_object(form, "<< /Type /XObject /Subtype /Form /BBox [0 0 200 200] >>")
        doc.update_stream(form, f"{i} {i} 1 1 re f".encode())
        entries.append(f"/{name} {form} 0 R")
    resources = doc.get_new_xref()
    doc.update_object(resources, f"<< /XObject << {' '.join(entries)} >> >>")
    doc.xref_set_key(page.xref, "Resources", f"{resources} 0 R")
    contents = doc.get_new_xref()
    doc.update_object(contents, "<<>>")
    doc.update_stream(contents, " ".join(f"/{name} Do" for name in names).encode())
    doc.xref_set_key(page.xref, "Contents", f"{contents} 0 R")
    unused = doc.get_new_xref()
    doc.update_object(unused, "<<>>")
    doc.save(path)


def _complaints(data: bytes) -> str:
    """What MuPDF reports while drawing every page, such as an XObject resource it cannot find."""
    doc = pymupdf.open("pdf", data)
    pymupdf.TOOLS.mupdf_warnings(reset=True)
    for page in doc.pages():
        page.get_pixmap(dpi=10)
    warnings: str = pymupdf.TOOLS.mupdf_warnings(reset=True)
    return warnings


def test_compacting_keeps_every_formula_of_a_page(tmp_path: Path) -> None:
    src = tmp_path / "paper.pdf"
    _page_of_forms(src)
    assert _complaints(src.read_bytes()) == ""

    assert _complaints(_compact(pymupdf.open(src))) == ""


def test_the_translated_version_cut_from_the_bilingual_one_keeps_every_formula(tmp_path: Path) -> None:
    src = tmp_path / "paper.pdf"
    _page_of_forms(src)
    both = tmp_path / "paper.bilingual.pdf"
    # Not compacted, so that only cutting it is tested
    both.write_bytes(_render_side_by_side(pymupdf.open(src), pymupdf.open(src)).tobytes())

    data = translated_from_bilingual(src, both, None)

    assert data is not None and _complaints(data) == ""
