"""EPUB 翻译：解压 → 按 OPF spine 找出正文 XHTML → 段落翻译 → 重新打包。

不依赖 ebooklib，直接按 zip 处理，原书里的 CSS、图片、字体、元数据全部原样保留。
"""
import posixpath
import zipfile
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

from . import html_blocks as hb

HTML_TYPES = {"application/xhtml+xml", "text/html"}


def _opf_path(zf: zipfile.ZipFile) -> str:
    container = BeautifulSoup(zf.read("META-INF/container.xml"), "lxml-xml")
    rootfile = container.find("rootfile")
    if rootfile is None:
        raise ValueError("EPUB 缺少 rootfile")
    return rootfile["full-path"]


def _content_docs(zf: zipfile.ZipFile, opf_path: str):
    """返回 (正文 XHTML 列表, nav 文档, ncx 文档, opf soup)。"""
    opf = BeautifulSoup(zf.read(opf_path), "lxml-xml")
    base = posixpath.dirname(opf_path)
    items = {}
    nav = ncx = None
    for it in opf.find_all("item"):
        href = posixpath.normpath(posixpath.join(base, unquote(it.get("href", ""))))
        items[it.get("id")] = (href, it.get("media-type", ""))
        if "nav" in (it.get("properties") or "").split():
            nav = href
        if it.get("media-type") == "application/x-dtbncx+xml":
            ncx = href
    docs = []
    for ref in opf.find_all("itemref"):
        href, mt = items.get(ref.get("idref"), (None, None))
        if href and mt in HTML_TYPES and href not in docs:
            docs.append(href)
    # spine 之外的 HTML（少见）也一起翻译
    for href, mt in items.values():
        if mt in HTML_TYPES and href not in docs and href != nav:
            docs.append(href)
    return docs, nav, ncx, opf


def _set_language(opf: BeautifulSoup, lang: str, bilingual: bool):
    if bilingual:
        return
    el = opf.find("dc:language") or opf.find("language")
    if el is not None:
        el.string = lang


async def translate_epub(src: Path, out_dir: Path, runner, bilingual: bool, target_lang: str = "",
                         dst: Path | None = None) -> list[Path]:
    zin = zipfile.ZipFile(src)
    opf_path = _opf_path(zin)
    docs, nav, ncx, opf = _content_docs(zin, opf_path)
    names = set(zin.namelist())

    # 1. 解析所有文档，收集段落
    soups: dict[str, BeautifulSoup] = {}
    tasks: list[tuple[str, object, str, str]] = []  # (文档, 元素, 原文, 类型)
    for name in docs:
        if name not in names:
            continue
        soup = hb.parse(zin.read(name), xml=True)
        soups[name] = soup
        tasks.extend((name, el, text, "block") for el, text in hb.find_blocks(soup))
    if nav and nav in names:
        soup = hb.parse(zin.read(nav), xml=True)
        soups[nav] = soup
        tasks.extend((nav, el, text, "label") for el, text in hb.translate_nav_links(soup))
    if ncx and ncx in names:
        soup = BeautifulSoup(zin.read(ncx), "lxml-xml")
        soups[ncx] = soup
        for el in soup.select("navLabel > text"):
            if hb.translatable(el.get_text(strip=True)):
                tasks.append((ncx, el, el.get_text(strip=True), "label"))

    # 2. 翻译（书名作为上下文）
    title_el = opf.find("dc:title") or opf.find("title")
    runner.set_title(title_el.get_text(strip=True) if title_el else src.stem)
    results = await runner.translate_all([t[2] for t in tasks])

    # 3. 回写
    for (name, el, text, kind), translated in zip(tasks, results):
        if kind == "block":
            hb.apply_translation(soups[name], el, text, translated, bilingual)
        else:
            hb.set_label(el, text, translated, bilingual)
    if bilingual:
        for name in docs:
            if name in soups:
                hb.add_style(soups[name])
    if target_lang:
        _set_language(opf, target_lang, bilingual)

    # 4. 打包：mimetype 必须是第一个文件且不压缩
    suffix = "bilingual" if bilingual else "translated"
    dst = dst or out_dir / f"{src.stem}.{suffix}.epub"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        zout.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for info in zin.infolist():
            if info.filename == "mimetype":
                continue
            if info.filename in soups:
                data = str(soups[info.filename]).encode("utf-8")
            elif info.filename == opf_path:
                data = str(opf).encode("utf-8")
            else:
                data = zin.read(info.filename)
            zout.writestr(info.filename, data)
    zin.close()
    return [dst]
