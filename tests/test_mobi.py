"""MOBI and AZW3 books (app/formats/mobi.py) without sample books: the unpacking is replaced by its result."""

import asyncio
import shutil
import stat
import zipfile
from pathlib import Path

import pytest

from app.formats import mobi
from tests.helpers import run, runner

FIRST = "The first paragraph of the old book."
SECOND = "The second paragraph comes after a page break."

OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Old Book</dc:title>
    <dc:language>en</dc:language><dc:identifier id="uid">old-book</dc:identifier></metadata>
  <manifest>
    <item id="text" href="book.html" media-type="text/html"/>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="cover" href="images/cover.jpg" media-type="image/jpeg"/>
  </manifest>
  <spine toc="ncx"><itemref idref="text"/></spine>
  <guide><reference type="toc" title="Contents" href="book.html#toc"/></guide>
</package>"""

NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head/><docTitle><text>Old Book</text></docTitle>
  <navMap><navPoint id="p1" playOrder="1"><navLabel><text>Start</text></navLabel>
    <content src="book.html#toc"/></navPoint></navMap>
</ncx>"""

# What kindleunpack writes for an old MOBI7 book: HTML with mbp: tags and a guide, not XHTML
BOOK = f"""<html><head><guide><reference type="toc" href="book.html#toc"/></guide></head>
<body><a id="toc"></a><p height="1em">{FIRST}</p><mbp:pagebreak/><p>{SECOND}</p></body></html>"""


def unpacked_mobi7(tmp_path: Path) -> Path:
    """The mobi7 directory kindleunpack leaves for an old book; returns its book.html."""
    base = tmp_path / "unpacked" / "mobi7"
    (base / "images").mkdir(parents=True)
    (base / "book.html").write_text(BOOK, encoding="utf-8")
    (base / "content.opf").write_text(OPF, encoding="utf-8")
    (base / "toc.ncx").write_text(NCX, encoding="utf-8")
    (base / "images" / "cover.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    return base / "book.html"


@pytest.fixture
def no_calibre(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(src: Path, dst: Path) -> bool:
        return False

    monkeypatch.setattr(mobi, "_calibre_convert", missing)


@pytest.mark.usefixtures("no_calibre")
def test_an_old_mobi_is_translated_as_an_epub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    book = unpacked_mobi7(tmp_path)
    monkeypatch.setattr(mobi, "_extract", lambda src, tmpdir: book)
    src = tmp_path / "old.mobi"
    src.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()

    outputs = run(mobi.translate_mobi(src, out, runner(), bilingual=False))

    assert outputs == [out / "old.translated.epub"]
    with zipfile.ZipFile(outputs[0]) as z:
        names = z.namelist()
        assert names[0] == "mimetype" and z.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        # book.html became XHTML under a new name; everything else of the book came along
        assert "OEBPS/book.html" not in names and "OEBPS/images/cover.jpg" in names
        xhtml = z.read("OEBPS/book.xhtml").decode()
        opf = z.read("OEBPS/content.opf").decode()
        ncx = z.read("OEBPS/toc.ncx").decode()
    assert f"[zh-CN] {FIRST}" in xhtml and f"[zh-CN] {SECOND}" in xhtml
    assert "mbp:" not in xhtml and "page-break-before: always" in xhtml
    assert 'href="book.xhtml"' in opf and 'media-type="application/xhtml+xml"' in opf
    assert 'href="book.xhtml#toc"' in opf and 'src="book.xhtml#toc"' in ncx


def test_a_mobi_is_also_written_back_when_calibre_is_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    book = unpacked_mobi7(tmp_path)
    monkeypatch.setattr(mobi, "_extract", lambda src, tmpdir: book)

    async def convert(src: Path, dst: Path) -> bool:
        dst.write_bytes(src.read_bytes())
        return True

    monkeypatch.setattr(mobi, "_calibre_convert", convert)
    src = tmp_path / "old.MOBI"
    src.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()

    outputs = run(mobi.translate_mobi(src, out, runner(), bilingual=True))

    assert outputs == [out / "old.bilingual.epub", out / "old.bilingual.mobi"]


@pytest.mark.usefixtures("no_calibre")
def test_a_mobi_holding_only_a_pdf_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "old.001.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(mobi, "_extract", lambda src, tmpdir: pdf)

    with pytest.raises(ValueError, match="仅包含 PDF"):
        run(mobi.translate_mobi(tmp_path / "old.mobi", tmp_path, runner(), bilingual=False))


def fake_calibre(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    """Put an ebook-convert that runs script on the PATH lookup."""
    exe = tmp_path / "ebook-convert"
    exe.write_text(f"#!/bin/sh\n{script}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(shutil, "which", lambda name: str(exe))


def test_calibre_writes_the_native_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_calibre(tmp_path, monkeypatch, 'cp "$1" "$2"')
    src, dst = tmp_path / "book.epub", tmp_path / "book.azw3"
    src.write_bytes(b"epub")
    assert run(mobi._calibre_convert(src, dst)) is True
    assert dst.read_bytes() == b"epub"


def test_a_failed_or_missing_calibre_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst = tmp_path / "book.epub", tmp_path / "book.azw3"
    src.write_bytes(b"epub")
    fake_calibre(tmp_path, monkeypatch, "exit 1")
    assert run(mobi._calibre_convert(src, dst)) is False

    monkeypatch.setattr(shutil, "which", lambda name: str(tmp_path / "not-installed"))
    assert run(mobi._calibre_convert(src, dst)) is False


def test_calibre_stops_when_the_job_is_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "still-running"
    # Left running, it would write the output file after the job was paused or deleted
    fake_calibre(tmp_path, monkeypatch, f'sleep 2; touch "{marker}"')
    src, dst = tmp_path / "book.epub", tmp_path / "book.azw3"

    async def cancel() -> None:
        task = asyncio.ensure_future(mobi._calibre_convert(src, dst))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(2.5)

    run(cancel())
    assert not marker.exists()
