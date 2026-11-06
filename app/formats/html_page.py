"""HTML translation: a saved web page or an HTML document is translated block by block like an EPUB chapter, keeping
inline formatting, links and the page's styles and scripts."""

from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

from . import html_blocks as hb

if TYPE_CHECKING:
    from ..runner import Runner

SUFFIXES = {".html", ".htm", ".xhtml"}


def _set_language(soup: BeautifulSoup, lang: str) -> None:
    root = soup.find(lambda t: hb._local(t.name) == "html")
    if root is None:
        return
    for key, value in hb.lang_attrs(lang).items():
        if key != "xml:lang" or root.has_attr("xml:lang"):
            root[key] = value


async def translate_html(
    src: Path, out_dir: Path, runner: "Runner", bilingual: bool, target_lang: str = ""
) -> list[Path]:
    xml = src.suffix.lower() == ".xhtml"
    soup = hb.parse(src.read_bytes(), xml=xml)
    if soup.find(True) is None:
        raise ValueError(f"无法解析 HTML 文件: {src.name}")
    blocks = hb.find_blocks(soup)
    flags, _ = hb.bibliography_skips([el for el, _, _ in blocks])
    # The page title is translated like a table of contents entry: the bilingual version shows both
    title_el = soup.find(lambda t: hb._local(t.name) == "title")
    title = title_el.get_text(strip=True) if title_el else ""
    labels = [(title_el, title)] if title_el is not None and title and hb.translatable(title) else []
    runner.set_title(title or src.stem)
    texts = [html or text for _, text, html in blocks] + [text for _, text in labels]
    results = await runner.translate_all(
        texts,
        kinds=[hb.preview_kind(el) for el, _, _ in blocks] + ["toc"] * len(labels),
        preview=True,
        html=[html is not None for _, _, html in blocks] + [False] * len(labels),
        skip=list(flags) + [False] * len(labels),
    )
    for (el, text, html), translated in zip(blocks, results):
        if html is not None:
            hb.apply_translation_html(soup, el, html, translated, bilingual, target_lang)
        else:
            hb.apply_translation(soup, el, text, translated, bilingual, target_lang)
    for (el, text), translated in zip(labels, results[len(blocks) :]):
        hb.set_label(el, text, translated, bilingual)
    if bilingual:
        hb.add_style(soup)
    elif target_lang:
        _set_language(soup, target_lang)
    suffix = "bilingual" if bilingual else "translated"
    dst = out_dir / f"{src.stem}.{suffix}{src.suffix}"
    # encode() also rewrites <meta charset> to UTF-8, the encoding written
    dst.write_bytes(soup.encode("utf-8"))
    return [dst]
