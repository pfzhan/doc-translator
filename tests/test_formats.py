import asyncio
import zipfile
from pathlib import Path

import pymupdf
import pytest
from bs4 import BeautifulSoup

from app.formats import translate_file
from app.formats.epub import _content_docs, translate_epub
from app.formats.markdown import collect_segments, render, split_blocks
from app.formats.mobi import _read_mobi7_html, _xhtml_from_mobi7
from app.formats.pdf import _join_lines
from app.runner import Runner
from app.translators import MockTranslator, OpenAITranslator

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class MemCache:
    def __init__(self):
        self.d = {}

    def get_many(self, prefix, texts):
        return {t: self.d[(prefix, t)] for t in texts if (prefix, t) in self.d}

    def put_many(self, prefix, pairs):
        for s, t in pairs.items():
            self.d[(prefix, s)] = t


def runner():
    return Runner(MockTranslator("zh-CN"), cache=MemCache())


def run(coro):
    return asyncio.run(coro)


MD = """---
title: Demo
---

# Hello World

This is a paragraph
spanning two lines.

```python
print("keep me")
```

- item one
- item two
  continued

> quoted text

| Name | Desc |
| --- | --- |
| foo | bar baz |

![img](a.png)
"""


def test_markdown_translated_only():
    blocks = split_blocks(MD)
    texts = collect_segments(blocks)
    assert "Hello World" in texts
    assert "This is a paragraph spanning two lines." in texts
    assert "item two continued" in texts
    assert not any("keep me" in t for t in texts)
    assert not any("title" in t for t in texts)
    out = render(blocks, {t: "T:" + t for t in texts}, bilingual=False)
    assert "# T:Hello World" in out
    assert 'print("keep me")' in out
    assert "- T:item two continued" in out
    assert "> T:quoted text" in out
    assert "| T:foo | T:bar baz |" in out
    assert "![img](a.png)" in out
    assert out.startswith("---\ntitle: Demo\n---")


def test_markdown_bilingual_keeps_original():
    blocks = split_blocks(MD)
    texts = collect_segments(blocks)
    out = render(blocks, {t: "T:" + t for t in texts}, bilingual=True)
    assert "# Hello World" in out and "# T:Hello World" in out
    assert "This is a paragraph\nspanning two lines.\n\nT:This is a paragraph" in out
    assert "foo<br>T:foo" in out


def _epub_texts(path):
    z = zipfile.ZipFile(path)
    assert z.infolist()[0].filename == "mimetype"
    assert z.infolist()[0].compress_type == zipfile.ZIP_STORED
    html = "".join(z.read(n).decode() for n in z.namelist() if n.endswith((".xhtml", ".html", ".htm")))
    return BeautifulSoup(html, "lxml").get_text()


@pytest.mark.parametrize("name", ["alice.epub", "alice.mobi", "alice.azw3"])
@pytest.mark.parametrize("bilingual", [True, False])
def test_ebooks(tmp_path, name, bilingual):
    src = SAMPLES / name
    if not src.exists():
        pytest.skip("sample missing")
    outputs = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
    epub = next(p for p in outputs if p.suffix == ".epub")
    text = _epub_texts(epub)
    assert "[zh-CN] Alice was beginning to get very tired" in text
    # 双语保留原文；仅译文模式下原文只会以“[zh-CN] 原文”形式出现
    plain = text.replace("[zh-CN] Alice was beginning", "")
    assert ("Alice was beginning to get very tired" in plain) == bilingual


def test_pdf(tmp_path):
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    for bilingual in (False, True):
        [out] = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
        doc = pymupdf.open(out)
        assert len(doc) == len(pymupdf.open(src))
        text = doc[0].get_text()
        assert "[zh-CN]" in text
        if bilingual:
            assert doc[0].rect.width == pytest.approx(pymupdf.open(src)[0].rect.width * 2)


def test_llm_unparseable_batch_falls_back_to_single():
    tr = OpenAITranslator("zh-CN", api_key="x", base_url="http://localhost", model="m")
    calls = []

    async def fake_complete(system, content):
        calls.append(content)
        return "garbage" if len(calls) == 1 else "你好"

    tr._complete = fake_complete
    assert run(tr.translate_batch(["a", "b"])) == ["你好", "你好"]  # 批量结果无法解析，逐条补翻
    assert len(calls) == 3


# --- 第二批：格式修复的回归测试（EPUB/MOBI/Markdown/PDF 样本全部内联构造） ---

CONTAINER_XML = (
    '<?xml version="1.0"?>'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def _write_epub(path, manifest: str, spine: str, docs: dict[str, str]):
    """构造最小 EPUB：mimetype 第一个且不压缩，OEBPS/content.opf + 给定文档。"""
    opf = (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>Test Book</dc:title><dc:language>en</dc:language>"
        f"</metadata><manifest>{manifest}</manifest><spine>{spine}</spine></package>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONTAINER_XML)
        z.writestr("OEBPS/content.opf", opf)
        for name, data in docs.items():
            z.writestr(f"OEBPS/{name}", data)


def test_epub_chapter_with_internal_dtd_not_dropped(tmp_path):
    """带内部 DTD 子集的 XHTML 章节：lxml-xml 解析会静默产出空文档，回退后内容必须保留。"""
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html [<!ENTITY nbsp "&#160;">]>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>c1</title></head>'
        "<body><p>Chapter one text&nbsp;here</p></body></html>"
    )
    src = tmp_path / "book.epub"
    _write_epub(
        src,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>',
        '<itemref idref="ch1"/>',
        {"ch1.xhtml": chapter},
    )
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    text = _epub_texts(out)
    # 章节没有被清空，原文段落照常翻译（&nbsp; 归并成普通空格）
    assert "[zh-CN] Chapter one text here" in text


def test_mobi7_cp1252_decoding(tmp_path):
    """老 MOBI7 的 book.html 按 cp1252 解码：é 等字符不能变成 U+FFFD。"""
    html_file = tmp_path / "book.html"
    html_file.write_bytes("<html><body><p>café et naïve</p></body></html>".encode("cp1252"))
    text = _read_mobi7_html(html_file)
    assert "café et naïve" in text
    assert "\ufffd" not in text
    # 转成 XHTML 后字符也保留
    assert "café et naïve" in _xhtml_from_mobi7(text).decode("utf-8")


def test_markdown_list_does_not_swallow_following_blocks():
    """列表后没有空行时，紧跟的标题 / 引用是独立的块，不能并进最后一个列表项。"""
    blocks = split_blocks("- item\n# Heading\n> quote")
    assert [b.kind for b in blocks] == ["list", "heading", "quote"]
    assert collect_segments(blocks) == ["item", "Heading", "quote"]


def test_epub_nav_in_spine_parsed_once(tmp_path):
    """EPUB3 的 nav 文档同时在 spine 里：只走 label 翻译路径一次，不重复解析、译文不丢。"""
    nav = (
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        '<head><title>Nav</title></head><body>'
        '<nav epub:type="toc"><ol><li><a href="ch1.xhtml">Chapter One</a></li></ol></nav>'
        "</body></html>"
    )
    chapter = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>c1</title></head>'
        "<body><p>Body paragraph</p></body></html>"
    )
    src = tmp_path / "book.epub"
    _write_epub(
        src,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<itemref idref="ch1"/><itemref idref="nav"/>',
        {"ch1.xhtml": chapter, "nav.xhtml": nav},
    )
    with zipfile.ZipFile(src) as z:
        docs, nav_href, _, _ = _content_docs(z, "OEBPS/content.opf")
    assert nav_href == "OEBPS/nav.xhtml"
    assert "OEBPS/nav.xhtml" not in docs  # nav 不作为正文再解析一遍

    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    with zipfile.ZipFile(out) as z:
        nav_html = z.read("OEBPS/nav.xhtml").decode()
        ch_html = z.read("OEBPS/ch1.xhtml").decode()
    # 目录项恰好翻译一次，没有重复也没有丢回原文
    assert nav_html.count("[zh-CN] Chapter One") == 1
    assert "<a href=\"ch1.xhtml\">" in nav_html  # 链接结构不变
    assert "[zh-CN] Body paragraph" in ch_html


def test_pdf_hyphen_joins_uppercase_word():
    """行尾连字符断词：德语断词后首字母大写也要合并掉连字符。"""
    assert _join_lines(["Donau-", "Dampfschiff"]) == "DonauDampfschiff"
    assert _join_lines(["word-", "wrap"]) == "wordwrap"
    assert _join_lines(["hello", "world"]) == "hello world"  # 非断词只是换行


def _pdf_with_text(text: str, fontname: str = "helv") -> "pymupdf.Document":
    doc = pymupdf.open()
    doc.new_page(width=400, height=300).insert_textbox(
        pymupdf.Rect(30, 30, 370, 250), text, fontsize=12, fontname=fontname)
    return doc


def test_pdf_list_items_split_into_segments():
    """同一块里的列表项各自成为独立翻译单元（幻灯片常见）；换行续行仍并入上一项。"""
    from app.formats.pdf import extract_blocks

    doc = _pdf_with_text("- First item here\n- Second item continues\nonto next line\n- Third item")
    texts = [b.text for b in extract_blocks(doc)]
    assert texts == ["- First item here", "- Second item continues onto next line", "- Third item"]


def test_pdf_hard_break_splits_lines():
    """无项目符号的硬换行（上行远短于块最宽行、下行大写开头、左对齐）也拆成两段。"""
    from app.formats.pdf import extract_blocks

    doc = _pdf_with_text("Short Name\nA Much Longer Affiliation Line Here")
    texts = [b.text for b in extract_blocks(doc)]
    assert texts == ["Short Name", "A Much Longer Affiliation Line Here"]


def _line(text: str, x0: float, x1: float, y0: float, size: float = 10) -> tuple[str, pymupdf.Rect, list[dict[str, float]]]:
    return (text, pymupdf.Rect(x0, y0, x1, y0 + size + 2), [{"size": size}])


def test_pdf_left_jump_keeps_full_width_wrap():
    """首行缩进、两行都撑到右边距，是自动换行，不能拆成两段。"""
    from app.formats.pdf import _split_lines

    items = [
        _line("started", 120, 400, 10),
        _line("the effort continues here", 108, 400, 24),
    ]
    assert _split_lines(items) == [[0, 1]]


def test_pdf_cjk_indent_wrap_stays_one_segment():
    """2em 首行缩进、续行没撑满，仍是同一段。"""
    from app.formats.pdf import _split_lines

    items = [
        _line("第一行撑满右边距", 64, 400, 10, size=12),
        _line("续行", 40, 80, 26, size=12),
    ]
    assert _split_lines(items) == [[0, 1]]


def test_pdf_left_jump_splits_stacked_labels():
    """上行没撑满、下行明显左跳，是并列短标签，要拆开。"""
    from app.formats.pdf import _split_lines

    items = [
        _line("Fig", 180, 210, 10),
        _line("A longer caption here", 40, 300, 24),
    ]
    assert _split_lines(items) == [[0], [1]]


def test_pdf_wrapped_body_on_attention_sample():
    """论文正文的首行缩进不能把句子从中间切开。"""
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    from app.formats.pdf import extract_blocks

    texts = [b.text for b in extract_blocks(pymupdf.open(src))]
    for sentence in (
        "started the effort",
        "Each layer has two sub-layers",
        "independent random variables",
        "added to the sub-layer input",
    ):
        assert any(sentence in t for t in texts), sentence


def test_pdf_cjk_lines_do_not_overlap(tmp_path):
    """中日韩译文缩小并下移，两行不重叠；非中日韩仍用原字号。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = pymupdf.open()
    page = src.new_page(width=400, height=200)
    page.insert_text((40, 40), "AAAA BBBB CCCC", fontsize=12)
    page.insert_text((40, 54), "DDDD EEEE FFFF", fontsize=12)
    src_path = tmp_path / "src.pdf"
    src.save(src_path)

    def block(y0: float, text: str) -> TextBlock:
        rect = pymupdf.Rect(40, y0, 200, y0 + 14)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=12, color="#000000", bold=False)

    blocks = [block(28, "AAAA BBBB CCCC"), block(42, "DDDD EEEE FFFF")]

    def line_boxes(lang: str, translations: list[str]) -> list[tuple[pymupdf.Rect, float]]:
        out = _render_translated(src_path, blocks, translations, lang)
        boxes = []
        for b in out[0].get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                boxes.append((pymupdf.Rect(ln["bbox"]), ln["spans"][0]["size"]))
        return boxes

    cjk = line_boxes("zh-CN", ["第一行中文译文加长", "第二行中文译文加长"])
    en = line_boxes("en", ["First line translation", "Second line translation"])
    assert len(cjk) == 2 and len(en) == 2
    assert cjk[0][0].y1 <= cjk[1][0].y0 + 0.05
    assert abs(cjk[0][1] - 12 * 0.88) < 0.2
    assert abs(en[0][1] - 12) < 0.2
    for lang in ("yue", "wyw"):
        sized = line_boxes(lang, ["第一行", "第二行"])
        assert sized and abs(sized[0][1] - 12 * 0.88) < 0.2
    # 下移之后，字形仍落在为溢出留出的扩展框里（允许 1pt 字面溢出）
    expanded_y0 = 28 + 12 * 0.1
    expanded_y1 = 42 + 12 * 0.3
    assert cjk[0][0].y0 >= expanded_y0 - 1
    assert cjk[0][0].y1 <= expanded_y1 + 1


# --- 行内格式（富文本段落）保留 ---

RICH_CHAPTER = (
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>c1</title></head><body>'
    "<p>Plain paragraph without formatting.</p>"
    '<p>With <b>bold</b> and <a href="https://example.com">a link</a> inside.</p>'
    "</body></html>"
)


def _rich_epub(path):
    _write_epub(
        path,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>',
        '<itemref idref="ch1"/>',
        {"ch1.xhtml": RICH_CHAPTER},
    )


def test_find_blocks_marks_rich_paragraphs():
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    blocks = hb.find_blocks(soup)
    assert [html is not None for _, _, html in blocks] == [False, True]
    assert "<b>bold</b>" in blocks[1][2] and 'href="https://example.com"' in blocks[1][2]


def test_apply_translation_html_keeps_tags_and_strips_script():
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    el, _, html = hb.find_blocks(soup)[1]
    hb.apply_translation_html(
        soup, el, html,
        '带有<b>粗体</b>和<a href="https://example.com">链接</a><script>alert(1)</script>。',
        False, "zh-CN")
    p = soup.find_all("p")[1]
    assert p.find("b").get_text() == "粗体"
    assert p.find("a")["href"] == "https://example.com"
    assert p.find("script") is None


def test_apply_translation_html_falls_back_when_tags_lost():
    """译文把行内标签全丢了（模型不守规矩时）：回退纯文本，不写入空壳。"""
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    el, _, html = hb.find_blocks(soup)[1]
    hb.apply_translation_html(soup, el, html, "第一行<br>第二行，完全没有标签。", False, "zh-CN")
    p = soup.find_all("p")[1]
    assert p.find("b") is None
    assert p.get_text(separator="\n") == "第一行\n第二行，完全没有标签。"
    assert p.find("br") is not None


def test_pagebreak_anchor_kept_without_rich_tags():
    """只有分页锚点的段落也走富文本路径；译文丢掉锚点时补回去。"""
    from app.formats import html_blocks as hb

    chapter = (
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        "<body><p>"
        '<span epub:type="pagebreak" id="page7"></span>'
        "The chapter continues here."
        "</p></body></html>"
    )
    soup = hb.parse(chapter, xml=True)
    blocks = hb.find_blocks(soup)
    assert len(blocks) == 1 and blocks[0][2] is not None
    assert 'id="page7"' in blocks[0][2]
    el, _, html = blocks[0]
    hb.apply_translation_html(soup, el, html, "章节从这里继续。", False, "zh-CN")
    anchor = soup.find(id="page7")
    assert anchor is not None and "pagebreak" in str(anchor.get("epub:type") or anchor.get("type") or "")


def test_sanitize_drops_event_handlers_and_script_urls():
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    el, _, html = hb.find_blocks(soup)[1]
    hb.apply_translation_html(
        soup, el, html,
        '<a href="javascript:alert(1)" onclick="alert(1)">链接</a><b onerror="x">粗体</b>',
        False, "zh-CN")
    p = soup.find_all("p")[1]
    link = p.find("a")
    assert link is not None and "href" not in link.attrs and "onclick" not in link.attrs
    assert p.find("b") is not None and "onerror" not in p.find("b").attrs


def test_epub_rich_paragraphs_keep_inline_formatting(tmp_path):
    src = tmp_path / "book.epub"
    _rich_epub(src)
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    z = zipfile.ZipFile(out)
    html = "".join(z.read(n).decode() for n in z.namelist() if n.endswith((".xhtml", ".html", ".htm")))
    assert "<b>" in html and 'href="https://example.com"' in html
    assert "[zh-CN]" in html
