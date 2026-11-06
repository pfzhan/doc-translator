"""MOBI / AZW3 translation: unpack to EPUB first, then run the EPUB pipeline.

- KF8 (AZW3, newer MOBI) contains an EPUB structure, which is used directly.
- Old MOBI7 has only a book.html, which is cleaned up into valid XHTML and packed into an EPUB.
Writing MOBI requires Calibre: if ebook-convert is installed locally, an extra .mobi / .azw3 is written,
otherwise only the EPUB.
"""

import asyncio
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from bs4 import BeautifulSoup
from lxml import etree
from lxml import html as lxml_html
from mobi.kindleunpack import unpackBook

from . import html_blocks as hb
from .epub import translate_epub

if TYPE_CHECKING:
    from ..runner import Runner


def _xhtml_from_mobi7(src_html: str) -> bytes:
    s = re.sub(r"<mbp:pagebreak\s*/?>(\s*</mbp:pagebreak>)?", '<div style="page-break-before: always"></div>', src_html)
    s = re.sub(r"</?mbp:[^>]*>", "", s)
    s = re.sub(r"<guide>.*?</guide>", "", s, flags=re.S)
    # Parse leniently with the HTML parser, then serialize as XML to get valid XHTML.
    # Non-standard MOBI7 attributes such as height/width do not affect reading and are kept
    body = lxml_html.document_fromstring(s).find("body")
    inner = ""
    if body is not None:
        inner = escape(body.text or "") + "".join(etree.tostring(c, encoding="unicode", method="xml") for c in body)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/><title>book</title>'
        f"</head><body>{inner}</body></html>"
    ).encode()


def _read_mobi7_html(html_path: Path) -> str:
    """kindleunpack writes the raw bytes into book.html according to the codec in the MOBI header: try UTF-8
    first, and on failure decode as cp1252 (common in old books)."""
    raw = html_path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _epub_from_mobi7(html_path: Path, epub_path: Path) -> None:
    base = html_path.parent
    opf = BeautifulSoup((base / "content.opf").read_text(encoding="utf-8"), "lxml-xml")
    for item in opf.find_all("item"):
        if item.get("href") == "book.html":
            item["href"] = "book.xhtml"
            item["media-type"] = "application/xhtml+xml"
    for ref in opf.find_all("reference"):
        href = hb.attr_text(ref, "href")
        if href.startswith("book.html"):
            ref["href"] = href.replace("book.html", "book.xhtml", 1)
    ncx = (base / "toc.ncx").read_text(encoding="utf-8").replace('src="book.html', 'src="book.xhtml')

    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        z.writestr("OEBPS/content.opf", str(opf))
        z.writestr("OEBPS/toc.ncx", ncx)
        z.writestr("OEBPS/book.xhtml", _xhtml_from_mobi7(_read_mobi7_html(html_path)))
        for f in base.rglob("*"):
            rel = f.relative_to(base).as_posix()
            if f.is_file() and rel not in ("book.html", "content.opf", "toc.ncx"):
                z.write(f, "OEBPS/" + rel)


async def _calibre_convert(src: Path, dst: Path) -> bool:
    exe = shutil.which("ebook-convert") or "/Applications/calibre.app/Contents/MacOS/ebook-convert"
    if not Path(exe).exists():
        return False
    proc = await asyncio.create_subprocess_exec(
        exe, str(src), str(dst), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        return await proc.wait() == 0 and dst.exists()
    except asyncio.CancelledError:
        # The subprocess does not stop when the job is paused or deleted; left running, it would keep writing the
        # output file in the background
        proc.kill()
        raise


def _extract(src: Path, tmpdir: Path) -> Path:
    """Unpack into tmpdir and return the EPUB, book.html or PDF in it. Same as mobi.extract, except that the
    caller creates and deletes the temporary directory: mobi.extract creates its own directory, which nobody
    deletes when unpacking fails."""
    unpackBook(str(src), str(tmpdir), epubver="A")
    for path in (tmpdir / "mobi8" / f"{src.stem}.epub", tmpdir / "mobi7" / "book.html", tmpdir / f"{src.stem}.001.pdf"):
        if path.exists():
            return path
    raise ValueError("无法解包此电子书，可能受 DRM 保护或文件已损坏")


async def translate_mobi(
    src: Path, out_dir: Path, runner: "Runner", bilingual: bool, target_lang: str = ""
) -> list[Path]:
    tmpdir = Path(tempfile.mkdtemp(prefix="mobiex"))
    # On pause or delete the coroutine is cancelled while the unpacking thread is still writing to the directory.
    # shield lets it finish, and the directory is deleted after it ends
    extract = asyncio.ensure_future(asyncio.to_thread(_extract, src, tmpdir))
    try:
        extracted = await asyncio.shield(extract)
        if extracted.suffix == ".epub":
            epub_src = extracted
        elif extracted.suffix == ".html":
            epub_src = tmpdir / "mobi7.epub"
            await asyncio.to_thread(_epub_from_mobi7, extracted, epub_src)
        else:
            raise ValueError("此 MOBI 文件仅包含 PDF 内容，请直接上传 PDF")

        suffix = "bilingual" if bilingual else "translated"
        epub_dst = out_dir / f"{src.stem}.{suffix}.epub"
        outputs = await translate_epub(epub_src, out_dir, runner, bilingual, target_lang, dst=epub_dst)

        native = out_dir / f"{src.stem}.{suffix}{src.suffix.lower()}"
        if await _calibre_convert(epub_dst, native):
            outputs.append(native)
        return outputs
    finally:
        if extract.done():
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            extract.add_done_callback(lambda task: _remove_after(task, tmpdir))


def _remove_after(task: asyncio.Future[Path], tmpdir: Path) -> None:
    if not task.cancelled():
        task.exception()  # Retrieve the exception, otherwise the event loop reports exception was never retrieved
    shutil.rmtree(tmpdir, ignore_errors=True)
