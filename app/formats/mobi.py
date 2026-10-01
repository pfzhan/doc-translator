"""MOBI / AZW3 翻译：先解包成 EPUB，再走 EPUB 流程。

- KF8（AZW3、新版 MOBI）里自带一份 EPUB 结构，直接用。
- 老的 MOBI7 只有一个 book.html，把它整理成合法 XHTML 后拼成 EPUB。
写 MOBI 需要 Calibre：如果本机装了 ebook-convert，会额外输出一份 .mobi / .azw3，否则只输出 EPUB。
"""
import asyncio
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import mobi
from bs4 import BeautifulSoup
from lxml import etree, html as lxml_html

from .epub import translate_epub


def _xhtml_from_mobi7(src_html: str) -> bytes:
    s = re.sub(r"<mbp:pagebreak\s*/?>(\s*</mbp:pagebreak>)?", '<div style="page-break-before: always"></div>', src_html)
    s = re.sub(r"</?mbp:[^>]*>", "", s)
    s = re.sub(r"<guide>.*?</guide>", "", s, flags=re.S)
    # 用 HTML 解析器容错解析，再按 XML 序列化，得到合法的 XHTML。
    # MOBI7 的 height/width 等非标准属性不影响阅读，保留即可
    body = lxml_html.document_fromstring(s).find("body")
    inner = ""
    if body is not None:
        inner = escape(body.text or "") + "".join(
            etree.tostring(c, encoding="unicode", method="xml") for c in body
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/><title>book</title>'
        f"</head><body>{inner}</body></html>"
    ).encode("utf-8")


def _read_mobi7_html(html_path: Path) -> str:
    """kindleunpack 按 MOBI 头 codec 把原始字节写进 book.html：先试 UTF-8，失败按 cp1252（老书常见）解码。"""
    raw = html_path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _epub_from_mobi7(html_path: Path, epub_path: Path):
    base = html_path.parent
    opf = BeautifulSoup((base / "content.opf").read_text(encoding="utf-8"), "lxml-xml")
    for item in opf.find_all("item"):
        if item.get("href") == "book.html":
            item["href"] = "book.xhtml"
            item["media-type"] = "application/xhtml+xml"
    for ref in opf.find_all("reference"):
        if ref.get("href", "").startswith("book.html"):
            ref["href"] = ref["href"].replace("book.html", "book.xhtml", 1)
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
    return await proc.wait() == 0 and dst.exists()


async def translate_mobi(src: Path, out_dir: Path, runner, bilingual: bool, target_lang: str = "") -> list[Path]:
    tmpdir, extracted = await asyncio.to_thread(mobi.extract, str(src))
    tmpdir = Path(tmpdir)
    try:
        extracted = Path(extracted)
        if extracted.suffix == ".epub":
            epub_src = extracted
        elif extracted.suffix == ".html":
            epub_src = tmpdir / "mobi7.epub"
            await asyncio.to_thread(_epub_from_mobi7, extracted, epub_src)
        else:
            raise ValueError("这本 MOBI 里只有 PDF 内容，请直接上传 PDF")

        suffix = "bilingual" if bilingual else "translated"
        epub_dst = out_dir / f"{src.stem}.{suffix}.epub"
        outputs = await translate_epub(epub_src, out_dir, runner, bilingual, target_lang, dst=epub_dst)

        native = out_dir / f"{src.stem}.{suffix}{src.suffix.lower()}"
        if await _calibre_convert(epub_dst, native):
            outputs.append(native)
        return outputs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
