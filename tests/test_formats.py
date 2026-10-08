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
    assert _join_lines(["the {v1}", ". This"]) == "the {v1}. This"
    assert _join_lines(["end", ", next"]) == "end, next"


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


def test_mid_line_capital_stays_in_the_sentence():
    """按词拆行时，行内大写词仍是这一句。句末之后的下一句、标记语言仍分开。"""
    from app.formats.pdf import _split_lines

    items = [
        _line("The", 325.8, 341.6, 590.3, size=9),
        _line("extracted", 350.8, 388.4, 590.3, size=9),
        _line("plan", 397.6, 414.9, 590.3, size=9),
        _line("is", 424.1, 430.3, 590.3, size=9),
        _line("serialized", 439.4, 476.3, 590.3, size=9),
        _line("in", 485.5, 493.2, 590.3, size=9),
        _line("DXL", 502.3, 519.9, 590.3, size=9),
        _line("format", 529.0, 555.9, 590.3, size=9),
        _line("and", 316.8, 331.6, 600.7, size=9),
        _line("shipped.", 339.9, 370.9, 600.7, size=9),
    ]
    groups = _split_lines(items)
    assert groups[0] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert 8 in groups[1]

    ended = [
        _line("workloads.", 316.8, 448.3, 338.4, size=9),
        _line("It is distinguished from", 457.3, 555.9, 338.4, size=9),
    ]
    assert _split_lines(ended) == [[0], [1]]
    markup = [
        _line("<dxl:Ident", 53.8, 110.0, 80, size=8),
        _line('ColId="0" Name="a"/>', 116.0, 220.0, 80, size=8),
    ]
    assert _split_lines(markup) == [[0], [1]]


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


def test_pdf_short_indent_stays_one_segment():
    """第一行又短又缩进、续行更长，仍是自动换行，不能拆。中文续行也不因为像新句子就拆。"""
    from app.formats.pdf import _split_lines

    latin = [
        _line("Hello", 64, 140, 10, size=12),
        _line("And the sentence continues much further", 40, 400, 26, size=12),
    ]
    cjk = [
        _line("开始", 64, 120, 10, size=12),
        _line("续行比第一行长很多", 40, 360, 26, size=12),
    ]
    assert _split_lines(latin) == [[0, 1]]
    assert _split_lines(cjk) == [[0, 1]]


def test_pdf_list_continuation_keeps_capital():
    """列表项的续行即使大写开头，也还是这一项，不是新段落。"""
    from app.formats.pdf import _split_lines

    items = [
        _line("- First item", 40, 120, 10),
        _line("Continues Here", 55, 220, 24),
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


def test_sanitize_rejects_anchor_only_translation():
    """原文有文字、译文只剩锚点（模型抽风）：判不合格回退，不能清空原段落。"""
    from app.formats import html_blocks as hb

    soup = hb.parse('<html><body><p><a id="page42"/>Anchor and plain text here.</p></body></html>', xml=False)
    el, _, html = hb.find_blocks(soup)[0]
    assert html is not None  # 锚点段落走富文本路径
    hb.apply_translation_html(soup, el, html, '<a id="page42"/>', False, "zh-CN")
    p = soup.find("p")
    assert p.get_text()  # 回退后仍有内容（译文不合格时原文或纯文本译文，不能是空壳）


def test_sanitize_rejects_numbered_pagebreak_as_only_text():
    """带页码的 pagebreak 不是幸存译文。只回锚点、<br>、或空标签加页码，都不能把正文清成页码。"""
    from app.formats import html_blocks as hb

    def chapter(body: str) -> str:
        return (
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
            f"<body><p>{body}</p></body></html>"
        )

    anchor = '<span epub:type="pagebreak" id="page7" role="doc-pagebreak">7</span>'
    prose = "The chapter continues here."

    def apply(body: str, translated: str) -> str:
        soup = hb.parse(chapter(body), xml=True)
        el, _, html = hb.find_blocks(soup)[0]
        assert html is not None
        hb.apply_translation_html(soup, el, html, translated, False, "zh-CN")
        text = soup.find("p").get_text()
        assert soup.find(id="page7") is not None
        return text

    assert prose in apply(anchor + prose, anchor)
    assert prose in apply(anchor + prose, "<br/>")
    assert "Important" in apply(anchor + "<b>Important</b> words.", anchor + "<b></b>")
    kept = apply(anchor + prose, anchor + "章节从这里继续。")
    assert "章节从这里继续。" in kept


def test_sanitize_keeps_raster_data_image_but_strips_script_url():
    from app.formats import html_blocks as hb

    src = '<p>Text <img src="data:image/png;base64,AAAA"/> with <a href="https://x">link</a></p>'
    soup = hb.parse(f"<html><body>{src}</body></html>", xml=False)
    el, _, html = hb.find_blocks(soup)[0]
    hb.apply_translation_html(
        soup, el, html,
        '文字 <img src="data:image/png;base64,AAAA" onerror="alert(1)"/>'
        '<img src="data:image/svg+xml;base64,PHN2Zw=="/> 和 '
        '<a href="javascript:alert(1)">坏链接</a><a href="https://x">链接</a>',
        False, "zh-CN")
    raster, svg = soup.find_all("img")
    assert raster["src"].startswith("data:image/png") and not raster.has_attr("onerror")
    assert not svg.has_attr("src")
    bad, good = soup.find_all("a")
    assert not bad.has_attr("href") and good["href"] == "https://x"
    assert hb._unsafe_url("data:image/jpeg;base64,AAAA", "image") is False
    assert hb._unsafe_url("data:image/svg+xml,<svg></svg>", "image") is True


def test_pdf_list_continuation_must_be_indented():
    """列表项后同 x0 的新行不是续行（可能是下一段）；缩进的悬挂对齐行才是续行。"""
    from app.formats.pdf import extract_blocks

    doc = _pdf_with_text("- Short item\nNew paragraph at same indent")
    assert [b.text for b in extract_blocks(doc)] == ["- Short item", "New paragraph at same indent"]


def test_pdf_cjk_continuation_needs_sentence_end():
    """CJK 行首：上行不以句末标点结尾视为续行；以句末标点结尾才拆段。"""
    from app.formats.pdf import extract_blocks

    # 第一行明显短于第二行（硬换行特征），但上行不是句末 → 不拆
    doc = _pdf_with_text("这是一个短行\n而这是一行明显更长的内容用来撑满整行的宽度", fontname="china-s")
    assert len(extract_blocks(doc)) == 1
    doc = _pdf_with_text("这是完整的一句。\n而这是一行明显更长的内容用来撑满整行的宽度", fontname="china-s")
    assert len(extract_blocks(doc)) == 2


def test_pdf_cjk_straight_quote_ends_sentence():
    """句末 ASCII 直引号和弯引号一样算句末；分号是连接标点，不拆段。"""
    from app.formats.pdf import _split_lines

    def pair(end: str) -> list[tuple[str, pymupdf.Rect, list[dict[str, float]]]]:
        return [
            _line(f"完整一句。{end}", 40, 140, 10, size=12),
            _line("而这是一行明显更长的内容用来撑满整行的宽度", 40, 400, 26, size=12),
        ]

    assert _split_lines(pair('"')) == [[0], [1]]
    assert _split_lines(pair("'")) == [[0], [1]]
    assert _split_lines(pair("”")) == [[0], [1]]
    assert _split_lines(pair("；")) == [[0, 1]]


# --- 参考文献章节保留原文 ---

def _biblio_epub(path):
    ch1 = ('<html xmlns="http://www.w3.org/1999/xhtml"><body>'
           "<h1>Chapter One</h1><p>Normal paragraph here.</p>"
           "<h2>Bibliography</h2><p>Smith, J. (2020). Some book. Press.</p>"
           "<h3>Primary sources</h3><p>Jones, A. (2019). Another book.</p>"
           "<h2>Appendix</h2><p>Appendix paragraph here.</p></body></html>")
    biblio = ('<html xmlns="http://www.w3.org/1999/xhtml"><body>'
              "<h1>References</h1><p>Doe, B. (2018). Cited work.</p></body></html>")
    _write_epub(
        path,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="bib" href="bibliography.xhtml" media-type="application/xhtml+xml"/>',
        '<itemref idref="ch1"/><itemref idref="bib"/>',
        {"ch1.xhtml": ch1, "bibliography.xhtml": biblio},
    )


def test_epub_bibliography_kept_original(tmp_path):
    src = tmp_path / "book.epub"
    _biblio_epub(src)
    r = runner()
    [out] = run(translate_epub(src, tmp_path, r, False, "zh-CN"))
    text = _epub_texts(out)
    # 正文和附录照翻；参考文献标题照翻；引文条目保留原文
    assert "[zh-CN] Normal paragraph here." in text
    assert "[zh-CN] Appendix paragraph here." in text
    assert "Smith, J. (2020). Some book. Press." in text and "[zh-CN] Smith" not in text
    assert "Jones, A. (2019). Another book." in text
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    # 预览里标成跳过
    updates = {u[0]: u for u in r.preview()["updates"]}
    segs = r.preview()["segments"]
    skipped_srcs = {segs[u[0]]["s"] for u in updates.values() if u[2] == 1}
    assert any("Smith, J." in s for s in skipped_srcs)
    assert any("Doe, B." in s for s in skipped_srcs)


def test_pdf_biblio_skips_follows_font_size():
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, page=0, bold=False):
        return TextBlock(page=page, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("Chapter body text.", 10),
        blk("References", 14),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Jones, A. (2019). Another.", 10),
        blk("Index", 14),
        blk("Index entry text.", 10),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, False, False]
    # 页眉不能把阈值改小；同等字号的引文不是退出标题；略小的下一章标题要退出
    mixed = [
        blk("Chapter body text.", 11),
        blk("References", 16),
        blk("Smith, J. (2020). Some book.", 12),
        blk("References", 9),
        blk("Jones, A. (2019). Another.", 12),
        blk("Appendix: supplementary notes for the study of this volume", 14),
        blk("Appendix paragraph here.", 11),
    ]
    assert biblio_skips(mixed) == [False, False, True, True, True, False, False]
    # 目录里的 References 后面没有引文，不进入
    toc = [
        blk("References", 14),
        blk("A normal chapter paragraph without a citation year.", 11),
        blk("More body text here.", 11),
    ]
    assert biblio_skips(toc) == [False, False, False]
    # 标题和引文同字号时，引文仍保留原文，下一标题退出
    same = [
        blk("Intro paragraph.", 11),
        blk("参考文献：", 12),
        blk("Smith, J. (2020). Some book.", 12),
        blk("Index", 12),
        blk("Index entry text.", 11),
    ]
    assert biblio_skips(same) == [False, False, True, False, False]


def test_pdf_biblio_entry_needs_a_citation_not_a_year_in_prose():
    """正文里的年份、arXiv 字样，以及更小的下一标题，都不能确认进入。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    prose = [
        blk("Chapter body text.", 11),
        blk("References", 14),
        blk("In 2017 the authors published a follow-up.", 11),
        blk("WMT 2014 is a common benchmark.", 11),
        blk("See the arXiv preprint for details.", 11),
        blk("More body text here.", 11),
    ]
    assert biblio_skips(prose) == [False] * 6
    zh = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14),
        blk("这项工作发表于2016年。", 11),
        blk("后续正文还在这里。", 11),
    ]
    assert biblio_skips(zh) == [False, False, False, False]
    # 14pt 目录标题后面是更小的加粗章节，前瞻必须停，不能把这一章吞掉
    toc = [
        blk("Contents line.", 11),
        blk("References", 14),
        blk("Introduction", 11, bold=True),
        blk("In 2017 the authors wrote this chapter.", 11),
        blk("The chapter continues here.", 11),
    ]
    assert biblio_skips(toc) == [False] * 5


def test_pdf_biblio_same_size_citations_do_not_exit():
    """行尾裸年份、CoRR、编号条目和换行续行都不是下一章；正文字号的加粗标题要退出。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    same = [
        blk("Intro paragraph.", 11),
        blk("References", 12, bold=True),
        blk("Vaswani, A. Attention is all you need. CoRR, abs/1706.03762, 2017.", 12),
        blk("1. Zhang, S. A paper title goes here. 2017.", 12),
        blk("[2] Dzmitry Bahdanau and Yoshua Bengio. Neural machine translation by jointly", 12),
        blk("learning to align and translate. CoRR, abs/1409.0473, 2014.", 12),
        blk("张三. 书名. 2016.", 12),
        blk("Index", 12, bold=True),
        blk("Index entry text.", 11),
    ]
    assert biblio_skips(same) == [False, False, True, True, True, True, True, False, False]
    bold_next = [
        blk("Chapter body text.", 10),
        blk("Another body line.", 10),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Acknowledgements", 10.1, bold=True),
        blk("Thanks to the reviewers.", 10),
    ]
    assert biblio_skips(bold_next) == [False, False, False, True, False, False]
    numbered_heading = [
        blk("Chapter body text.", 10),
        blk("References", 12, bold=True),
        blk("[1] Smith, J. A paper. CoRR, abs/1706.03762, 2017.", 10),
        blk("[1] Appendix", 14, bold=True),
        blk("Appendix paragraph here.", 10),
    ]
    assert biblio_skips(numbered_heading) == [False, False, True, False, False]
    appendix = [
        blk("Chapter body text.", 10),
        blk("Another body line.", 10),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("1. Appendix", 10.1, bold=True),
        blk("Appendix paragraph here.", 10),
    ]
    assert biblio_skips(appendix) == [False, False, False, True, False, False]


def _one_epub(path, docs: dict[str, str]):
    items = "".join(
        f'<item id="d{i}" href="{name}" media-type="application/xhtml+xml"/>'
        for i, name in enumerate(docs)
    )
    spine = "".join(f'<itemref idref="d{i}"/>' for i in range(len(docs)))
    _write_epub(path, items, spine, docs)


def _xhtml(body: str) -> str:
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        f"<body>{body}</body></html>"
    )


def test_epub_chapter_titled_papers_ends_bibliography(tmp_path):
    """平铺文档里 h2 Papers 是下一章，不能因为单词像小标题就留成原文。"""
    src = tmp_path / "book.epub"
    _one_epub(src, {"book.xhtml": _xhtml(
        "<h1>References</h1><p>Doe, B. (2018). Cited work.</p>"
        "<h2>Papers</h2><p>Papers chapter paragraph.</p>"
        "<h2>Appendix</h2><p>Appendix paragraph here.</p>"
    )})
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    assert "[zh-CN] Papers chapter paragraph." in text
    assert "[zh-CN] Appendix paragraph here." in text


def test_epub_deeper_heading_ends_flat_bibliography(tmp_path):
    """单文件里 h1 参考文献后的 h2 附录是下一章，不能把后面整篇留成原文。

    包在后一个 section 里的更深标题同样结束，不靠标题数字单独判断。
    """
    src = tmp_path / "book.epub"
    _one_epub(src, {"book.xhtml": _xhtml(
        "<h1>References</h1><p>Doe, B. (2018). Cited work.</p>"
        "<h2>Appendix</h2><p>Appendix paragraph here.</p>"
        "<section><h1>References</h1><p>Lee, C. (2017). Third work.</p></section>"
        "<section><h2>Appendix</h2><p>Second appendix paragraph.</p></section>"
    )})
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    assert "[zh-CN] Appendix paragraph here." in text
    assert "Lee, C. (2017). Third work." in text and "[zh-CN] Lee" not in text
    assert "[zh-CN] Second appendix paragraph." in text


def test_epub_filename_substring_does_not_skip_chapter(tmp_path):
    src = tmp_path / "book.epub"
    _one_epub(src, {
        "chapter-references.xhtml": _xhtml("<h1>Cross References in Plato</h1><p>Normal paragraph here.</p>"),
        "bibliographical-notes.xhtml": _xhtml("<h1>Notes</h1><p>Another normal paragraph.</p>"),
    })
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "[zh-CN] Cross References in Plato" in text
    assert "[zh-CN] Normal paragraph here." in text
    assert "[zh-CN] Another normal paragraph." in text


def test_epub_bibliography_file_still_exits_on_next_heading(tmp_path):
    src = tmp_path / "book.epub"
    _one_epub(src, {"bibliography.xhtml": _xhtml(
        "<h1>References</h1><p>Doe, B. (2018). Cited work.</p>"
        "<h1>Index</h1><p>Index entry text here.</p>"
    )})
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    assert "[zh-CN] Index entry text here." in text


def test_epub_bibliography_continues_across_spine(tmp_path):
    src = tmp_path / "book.epub"
    _one_epub(src, {
        "ch1.xhtml": _xhtml("<h2>Bibliography</h2><p>Smith, J. (2020). Some book. Press.</p>"),
        "index_split_002.xhtml": _xhtml(
            "<p>Jones, A. (2019). Another book.</p><h1>Appendix</h1><p>Appendix paragraph here.</p>"
        ),
    })
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Smith, J. (2020). Some book. Press." in text and "[zh-CN] Smith" not in text
    assert "Jones, A. (2019). Another book." in text and "[zh-CN] Jones" not in text
    assert "[zh-CN] Appendix paragraph here." in text


def test_epub_repeated_citation_in_body_is_translated(tmp_path):
    src = tmp_path / "book.epub"
    cite = "Smith, J. (2020). Some book. Press."
    _one_epub(src, {"ch1.xhtml": _xhtml(
        f"<p>{cite}</p><h2>Bibliography</h2><p>{cite}</p>"
        "<h2>Appendix</h2><p>Appendix paragraph here.</p>"
    )})
    r = runner()
    text = _epub_texts(run(translate_epub(src, tmp_path, r, False, "zh-CN"))[0])
    assert f"[zh-CN] {cite}" in text
    assert text.count(cite) >= 1
    updates = {u[0]: u for u in r.preview()["updates"]}
    segs = r.preview()["segments"]
    cite_idxs = [i for i, seg in enumerate(segs) if cite in seg["s"]]
    assert len(cite_idxs) == 2
    assert updates[cite_idxs[0]][2] == 0 and updates[cite_idxs[0]][1].startswith("[zh-CN]")
    assert updates[cite_idxs[1]][2] == 1


def test_epub_styled_heading_colon_and_epub_type(tmp_path):
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(
        '<p class="heading">Bibliography</p><p>Smith, J. (2020). Some book. Press.</p>'
        "<h2>Appendix</h2><p>Appendix paragraph here.</p>"
        '<h2>参考文献：</h2><p>Doe, B. (2018). Cited work.</p>'
        "<h2>Index</h2><p>Index entry text here.</p>"
        '<section epub:type="bibliography"><h2>Notes</h2>'
        "<p>Lee, C. (2017). Third work.</p></section>"
        "<h2>Afterword</h2><p>Afterword paragraph here.</p>"
        "<p>Bibliography</p><p>This sentence mentions Bibliography but is body text.</p>"
        "<h2>12. References</h2><p>Ng, D. (2016). Fourth work.</p>"
        "<h2>Glossary</h2><p>Glossary paragraph here.</p>"
    )})
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Smith, J. (2020). Some book. Press." in text and "[zh-CN] Smith" not in text
    assert "[zh-CN] Appendix paragraph here." in text
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    assert "[zh-CN] Index entry text here." in text
    assert "Lee, C. (2017). Third work." in text and "[zh-CN] Lee" not in text
    assert "[zh-CN] Afterword paragraph here." in text
    assert "[zh-CN] This sentence mentions Bibliography but is body text." in text
    assert "Ng, D. (2016). Fourth work." in text and "[zh-CN] Ng" not in text
    assert "[zh-CN] Glossary paragraph here." in text


def test_pdf_biblio_skips_on_attention_paper():
    """arXiv 风格引文（年份在行尾不带括号）也要识别：attention.pdf 的 References 整节跳过。"""
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    from app.formats.pdf import extract_blocks, biblio_skips

    blocks = extract_blocks(pymupdf.open(src))
    skips = biblio_skips(blocks)
    refs = next(i for i, b in enumerate(blocks) if b.text.strip() == "References")
    viz = next(i for i, b in enumerate(blocks) if b.text.strip() == "Attention Visualizations")
    assert skips.index(True) == refs + 1
    assert blocks[refs + 1].text.strip().startswith("[")
    assert all(skips[refs + 1:viz])
    assert not skips[viz]
    assert not any(skips[viz + 1:])


def test_skipped_bibliography_keeps_hanging_indent(tmp_path):
    """跳过的引文留在原页。刊名斜体不能触发擦除重排，否则悬挂缩进和行尾断词会丢。"""
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    page = doc.new_page(width=420, height=240)
    roman = pymupdf.Font("tiro")
    italic = pymupdf.Font("tiit")
    bold = pymupdf.Font("tibo")
    writer = pymupdf.TextWriter(page.rect)
    writer.append((40, 36), "A body paragraph long enough to set the median size.", font=roman, fontsize=10)
    writer.append((40, 70), "References", font=bold, fontsize=14)
    _, cursor = writer.append((40, 100), "[1] Smith, J. (2020). A paper. In ", font=roman, fontsize=9)
    writer.append(cursor, "Proceedings of the Inter-", font=italic, fontsize=9)
    _, cursor = writer.append((54, 112), "national Conference, ", font=italic, fontsize=9)
    writer.append(cursor, "ICML, 2009.", font=roman, fontsize=9)
    writer.write_text(page)
    src = tmp_path / "refs.pdf"
    doc.save(src)

    before = pymupdf.open(src)[0]
    national = before.search_for("national")
    proceedings = before.search_for("Proceedings")
    assert national and proceedings

    [out] = run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))
    page = pymupdf.open(out)[0]
    assert "[zh-CN]" in page.get_text()
    assert "Inter- national" not in page.get_text()
    kept = page.search_for("national")
    kept_proc = page.search_for("Proceedings")
    assert kept and kept_proc
    assert abs(kept[0].x0 - national[0].x0) < 0.8
    assert abs(kept_proc[0].x0 - proceedings[0].x0) < 0.8
    fonts = {
        span["font"]
        for block in page.get_text("dict")["blocks"] if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if "Proceedings" in span["text"] or "national" in span["text"]
    }
    assert fonts
    assert all("Times" in name or "NimbusRom" in name for name in fonts)


def test_pdf_biblio_entry_with_bold_subheadings():
    """References 下先出现加粗小标题（Books/Articles）再出现引文，也要进入参考文献。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000", bold=bold)

    blocks = [
        blk("Body paragraph.", 10),
        blk("References", 14, bold=True),
        blk("Books", 11, bold=True),
        blk("Smith, J. (2020). Some book title here. Press.", 10),
        blk("Articles", 11, bold=True),
        blk("Doe, A. (2019). Some article title. Journal.", 10),
        blk("Index", 14, bold=True),
        blk("Index entry.", 10),
    ]
    assert biblio_skips(blocks) == [False, False, False, True, False, True, False, False]


def test_pdf_biblio_section_title_stops_lookahead():
    """更大的 References 不能穿过 Endnotes 进入，否则正文字号的下一标题退不出去。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("Chapter body text.", 10),
        blk("Another body line.", 10),
        blk("References", 16, bold=True),
        blk("This paragraph is still the chapter.", 10),
        blk("Endnotes", 12, bold=True),
        blk("See Smith et al. (2018). A cited work.", 10),
        blk("Acknowledgements", 10.1, bold=True),
        blk("Thanks to the reviewers.", 10),
    ]
    assert biblio_skips(blocks) == [False, False, False, False, False, True, False, False]


def test_pdf_biblio_journal_line_does_not_end_numbered_entry():
    """引文里单独一行 Journal 不是小标题，后面的小写续行仍留在参考文献里。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("Intro paragraph.", 11),
        blk("References", 12, bold=True),
        blk("[2] Dzmitry Bahdanau and Yoshua Bengio. Neural machine translation by jointly", 12),
        blk("Journal", 12),
        blk("of medicine and was widely cited afterward.", 12),
        blk("Smith, J. (2019). Another paper.", 12),
        blk("Index", 12, bold=True),
        blk("Index entry text.", 11),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True, False, False]


def test_pdf_biblio_chapter_titled_books_or_paper_does_not_enter():
    """同字号或更大的 Books、论文是下一章。更小的 Paper 仍是内部小标题。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    books = [
        blk("Body paragraph.", 10),
        blk("References", 14, bold=True),
        blk("Books", 16, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Chapter continues here.", 10),
    ]
    assert biblio_skips(books) == [False, False, False, False, False]
    thesis = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("论文", 12, bold=True),
        blk("这项研究发表于某出版社。", 11),
        blk("后续正文还在这里。", 11),
    ]
    assert biblio_skips(thesis) == [False, False, False, False, False]
    paper = [
        blk("Body paragraph.", 10),
        blk("References", 14, bold=True),
        blk("Paper", 11, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Index", 14, bold=True),
        blk("Index entry.", 10),
    ]
    assert biblio_skips(paper) == [False, False, False, True, False, False]


def test_pdf_biblio_chinese_gbt_citations():
    """中文 GB/T 引文（[J][M] 类型标记）：小一号“论文”是分组，引文跳过，正文提出版社不进入。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("论文", 12, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("李四. 某书[M]. 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(blocks) == [False, False, False, True, True, False, False]


def test_pdf_biblio_type_mark_does_not_swallow_headings():
    """行内 [J]/[M] 不是引文。含标记的下一章标题要退出，标题本身不跳过。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    prose = [
        blk("Chapter body text.", 10),
        blk("References", 16, bold=True),
        blk("期刊论文的类型标识为[J]，专著为[M]。", 10),
        blk("See note [J]", 10),
        blk("The concentration [M] was measured carefully.", 10),
        blk("Endnotes", 12, bold=True),
        blk("See Smith et al. (2018). A cited work.", 10),
        blk("Acknowledgements", 10.1, bold=True),
        blk("Thanks to the reviewers.", 10),
    ]
    assert biblio_skips(prose) == [False, False, False, False, False, False, True, False, False]
    appendix = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("Appendix [C]", 14, bold=True),
        blk("Appendix paragraph.", 11),
    ]
    assert biblio_skips(appendix) == [False, False, True, False, False]
    cn = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("附录[J]", 14, bold=True),
        blk("附录正文。", 11),
    ]
    assert biblio_skips(cn) == [False, False, True, False, False]
    larger = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("第2章 [M]", 16, bold=True),
        blk("章正文。", 11),
    ]
    assert biblio_skips(larger) == [False, False, True, False, False]


def test_pdf_biblio_numbered_publisher_entry_without_space():
    """[1] 后没有空格，或出版社在下一块，仍进入。无编号正文里的出版社不进入。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    tight = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1]张三. 书名. 北京: 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(tight) == [False, False, True, False, False]
    split = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1] 张三. 书名.", 11),
        blk("北京: 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(split) == [False, False, True, True, False, False]
    prose = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("这项研究发表于某出版社。", 11),
        blk("后续正文还在这里。", 11),
    ]
    assert biblio_skips(prose) == [False, False, False, False]


def test_pdf_biblio_edition_heading_exits():
    """以年版结尾的编号短标题是下一章，标题本身不跳过。真正的著录行仍跳过。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    year = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("1. 2019年版", 14, bold=True),
        blk("后续正文。", 11),
    ]
    assert biblio_skips(year) == [False, False, True, False, False]
    press = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1]张三. 书名. 人民出版社, 2019年版.", 11),
        blk("1. 人民出版社2019年版", 14, bold=True),
        blk("后续正文。", 11),
    ]
    assert biblio_skips(press) == [False, False, True, False, False]


def test_pdf_biblio_type_mark_in_prose_does_not_enter():
    """正文句子中间的 [M]。 不是引文（没有年份），不能把后面的正文留成原文。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("Body text.", 11),
        blk("References", 14, bold=True),
        blk("这个概念的出处见某书[M]。后续还有讨论。", 11),
        blk("正文继续。", 11),
    ]
    assert biblio_skips(blocks) == [False, False, False, False]
    later = [
        blk("Body text.", 11),
        blk("References", 14, bold=True),
        blk("这个概念的出处见某书[M]。后续还有讨论。", 11),
        blk("2020年又有新的讨论。", 11),
        blk("正文继续。", 11),
    ]
    assert biblio_skips(later) == [False, False, False, False, False]


def test_pdf_biblio_type_mark_year_on_next_block_enters():
    """类型标记和年份被拆开时仍进入。下一块要是卷期或出版社，不能是任意后文。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    volume = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报", 11),
        blk("2020, 3(2): 1-10.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(volume) == [False, False, True, True, False, False]
    numbered = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1]张三. 某某研究[J]. 某某学报", 11),
        blk("2020, 3(2): 1-10.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(numbered) == [False, False, True, True, False, False]
    press = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某书[M]. 书名", 11),
        blk("北京: 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(press) == [False, False, True, True, False, False]


def test_pdf_biblio_year_continuation_tightened():
    """拆行的年份确认不能松到正文：年份开头的英文句子、提到出版社和年份的正文都不算续行。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    def scenario(second: str):
        return biblio_skips([
            blk("参考文献", 14, bold=True),
            blk("见某书[M]。如下。", 11),
            blk(second, 11),
        ])

    assert scenario("2020, we showed more.") == [False, False, False]
    assert scenario("2020, volume were higher than expected.") == [False, False, False]
    assert scenario("2020, Volume of the series was larger.") == [False, False, False]
    assert scenario("这项研究2020年发表于某出版社。") == [False, False, False]
    assert scenario("人民出版社, 2019年出版.") == [False, False, False]
    # 卷期页码、第N期，以及以「年」或中文句号收尾的出版社行仍算
    assert scenario("2020, 3(2): 1-10.") == [False, True, True]
    assert scenario("2020, Vol. 3") == [False, True, True]
    assert scenario("2020, Volume 3") == [False, True, True]
    assert scenario("2020, volume. 12, no. 2") == [False, True, True]
    assert scenario("2020, pp. 1-10.") == [False, True, True]
    assert scenario("2020, 第3期") == [False, True, True]
    assert scenario("北京: 人民出版社, 2019.") == [False, True, True]
    assert scenario("北京: 人民出版社, 2019年.") == [False, True, True]
    assert scenario("北京: 人民出版社, 2019。") == [False, True, True]
    assert scenario("北京: 人民出版社2019年.") == [False, True, True]


def test_pdf_biblio_vol_abbreviation_also_needs_digit():
    """Vol 缩写和 Volume 全拼一样：后面必须是数字，“2020, Vol were…” 是正文。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    def scenario(second: str):
        return biblio_skips([
            blk("参考文献", 14, bold=True),
            blk("见某书[M]。如下。", 11),
            blk(second, 11),
        ])

    assert scenario("2020, Vol were higher than expected.") == [False, False, False]
    assert scenario("2020, Vol of the series was larger.") == [False, False, False]
    assert scenario("2020, Vol. mix") == [False, False, False]
    assert scenario("2020, Vol. mill") == [False, False, False]
    assert scenario("2020, Vol. 3") == [False, True, True]
    assert scenario("2020, Vol 3") == [False, True, True]
    assert scenario("2020, Vol. III") == [False, True, True]
    assert scenario("2020, Vol. IV, pp. 1-10.") == [False, True, True]
    assert scenario("2020, Vol. xii, pp. 1-10.") == [False, True, True]


def test_pdf_biblio_volume_lines_protected_at_same_size():
    """参考文献标题和正文同字号时，卷期/出版社续行也不能被误判为下一章。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("Body text.", 12),
        blk("参考文献", 12, bold=True),
        blk("张三. 某某研究[J]. 某某学报", 12),
        blk("2020, 3(2): 1-10.", 12),
        blk("李四. 某书[M]. 书名", 12),
        blk("2020, Vol. III", 12),
        blk("王五. 另一本书[M]. 书名", 12),
        blk("北京: 人民出版社, 2019年。", 12),
        blk("Index", 12, bold=True),
        blk("索引条目。", 12),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True, True, True, False, False]


def test_pdf_biblio_year_prefixed_chapter_still_exits():
    """行首是年份的下一章不是卷期续行。加粗的长标题同样退出。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    def scenario(line: str, *, bold: bool = False) -> list[bool]:
        body = (
            "这是后续正文，需要被翻译，不能因为上一行像年份就留在参考文献里。"
            "这里再补上一整句，让整段明显超过六十个字，避免它本身被当成同字号短标题。"
        )
        return biblio_skips([
            blk("Body text.", 12),
            blk("References", 12, bold=True),
            blk("Smith, J. (2020). Some book.", 12),
            blk(line, 12, bold=bold),
            blk(body, 12),
        ])

    assert scenario("2020, 1. Introduction") == [False, False, True, False, False]
    assert scenario("2020, 3. Results") == [False, False, True, False, False]
    assert scenario("2020, 第1章 结论") == [False, False, True, False, False]
    assert scenario("2020, No further reading") == [False, False, True, False, False]
    assert scenario("2020, p < 0.05.") == [False, False, True, False, False]
    assert scenario("2020, 3 days later.") == [False, False, True, False, False]
    assert scenario("2020, 1. Introduction to machine translation systems", bold=True) == [
        False, False, True, False, False,
    ]
    assert scenario("2020, 3(2): 1-10.") == [False, False, True, True, True]
    assert scenario("2020, Vol. III") == [False, False, True, True, True]


def test_pdf_biblio_long_bold_chapter_title_exits():
    """超过 80 字的加粗长章节标题也要退出（heading_like 的加粗分支有 80 字上限，不能靠它兜底）。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    body = "这是后续正文，需要被翻译，这里再补上一整句，让整段明显超过六十个字，避免被当成短标题。"
    blocks = [
        blk("Body text.", 12),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 12),
        blk("Chapter 12: On the relationship between government forms and the laws that govern them", 12, bold=True),
        blk(body, 12),
    ]
    assert biblio_skips(blocks) == [False, False, True, False, False]


def test_pdf_biblio_long_bold_citation_stays():
    """超过 80 字的加粗著录行不是下一章。类型标记或行尾年份仍要跳过。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    gbt = (
        "张三. 某某研究的题目在这里继续写长，把期刊论文的背景、方法和结论都写进题名，"
        "直到这一行确实超过八十个字。[J]. 某某学报, 2020, 3(2): 1-10."
    )
    smith = "Smith, J. A long paper title goes here and continues past eighty characters. 2019."
    assert len(gbt) > 80 and len(smith) > 80
    body = "Jones, K. (2018). Another citation that must stay untranslated."
    blocks = [
        blk("Body text.", 12),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 12),
        blk(gbt, 12, bold=True),
        blk(smith, 12, bold=True),
        blk(body, 12),
        blk("Index", 12, bold=True),
        blk("Index entry text.", 12),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True, False, False]


def test_scan_status_detection():
    from app.formats.pdf import _scan_status

    # 正常 PDF
    doc = _pdf_with_text("Visible paragraph text here.")
    assert _scan_status(doc) == "ok"
    # 带隐藏 OCR 文本层（render mode 3）
    doc = pymupdf.open()
    for _ in range(4):
        p = doc.new_page()
        p.insert_text((50, 50), "hidden layer text here", fontsize=12, render_mode=3)
    assert _scan_status(doc) == "hidden_text"
    # 纯图片扫描件
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page().insert_image(pymupdf.Rect(10, 10, 60, 60),
                                    pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 5, 5)))
    assert _scan_status(doc) == "scanned"


def test_pdf_scanned_error_guides_ocr(tmp_path):
    from app.formats.pdf import translate_pdf

    src = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(10, 10, 60, 60),
                                pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 5, 5)))
    doc.save(src)
    with pytest.raises(ValueError, match="OCR"):
        run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))


def test_pdf_bookmarks_migrated(tmp_path):
    doc = _pdf_with_text("Chapter one content here for testing.")
    doc.set_toc([(1, "Chapter One", 1)])
    src = tmp_path / "book.pdf"
    doc.save(src)
    for bilingual in (False, True):
        [out] = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
        assert pymupdf.open(out).get_toc() == [[1, "Chapter One", 1]]


def test_subset_fonts_safe_returns_valid_doc():
    from app.formats.pdf import _subset_fonts_safe

    doc = _pdf_with_text("Some text for subsetting.")
    out = _subset_fonts_safe(doc)
    assert out.page_count == doc.page_count and "Some text" in out[0].get_text()


def test_typesetting_ladder_expands_before_shrinking():
    """放不下时先向右扩（避开右侧文本块）：扩出去后字号比直接缩更大，且不盖住右侧块。"""
    from app.formats.pdf import _render_translated, extract_blocks

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=100)
    page.insert_textbox(pymupdf.Rect(20, 17, 100, 40), "Short left", fontsize=12)
    page.insert_textbox(pymupdf.Rect(300, 17, 390, 40), "Right block", fontsize=12)
    src = doc.tobytes()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(src)
        src_path = Path(f.name)
    blocks = extract_blocks(pymupdf.open(src_path))
    left = next(b for b in blocks if b.text.startswith("Short"))
    right = next(b for b in blocks if b.text.startswith("Right"))
    translations = ["这是一段比原文长很多的译文，原来肯定放不下" if b is left else b.text for b in blocks]
    out = _render_translated(src_path, blocks, translations, "zh-CN")
    spans = [ln for blk in out[0].get_text("dict")["blocks"] for ln in blk["lines"]]
    zh = [ln for ln in spans if "译文" in "".join(s["text"] for s in ln["spans"])]
    assert zh
    zh_rect = zh[0]["bbox"]
    # 译文越过了原块的右边界（发生了右扩），且没有盖住右侧块
    assert zh_rect[2] > left.rect.x1 + 1 and zh_rect[2] <= right.rect.x0
    # 右侧块原文还在
    assert "Right block" in out[0].get_text()


def test_formula_font_and_char_rules():
    from app.formats.pdf import _font_is_math, _is_formula_char, _math_chars

    assert _font_is_math("CMMI10") and _font_is_math("STIXTwoMath")
    assert _font_is_math("LibertinusMath-Regular")
    assert not _font_is_math("TimesNewRomanPSMT") and not _font_is_math("ArialMT")
    assert not _font_is_math("NotoSansCJK-Regular")
    # 黑名单优先：Noto 系正文字体不匹配宽泛 *Sans* 启发式
    assert _is_formula_char("α") and _is_formula_char("∑") and _is_formula_char("∂")
    assert not _is_formula_char("a") and not _is_formula_char("中")
    spans = [
        {"text": "E = mc", "font": "TimesNewRomanPSMT"},
        {"text": "αβγ", "font": "TimesNewRomanPSMT"},
        {"text": " + more text here", "font": "ArialMT"},
    ]
    assert _math_chars(spans) == 5  # αβγ 三个，加上 = 和 +（Sm 类数学符号）


def _block_with_spans(span_lines, size=12.0):
    from app.formats.pdf import TextBlock

    return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                     text="", size=size, color="#000", bold=False, span_lines=span_lines)


def test_placeholderize_and_restore_roundtrip():
    from app.formats.pdf import _placeholderize, _restore_placeholders

    def span(text, font, size, x0=0, y0=0):
        return {"text": text, "font": font, "size": size, "bbox": (x0, y0, x0 + 10, y0 + 10),
                "origin": (x0, y0 + 8)}

    spans = [
        span("The formula ", "TimesNewRomanPSMT", 12.0),
        span("E=mc", "CMMI10", 12.0, x0=60),
        span("2", "CMR10", 8.5, x0=80, y0=-2),  # 上标并入公式 run
        span(" shows energy", "TimesNewRomanPSMT", 12.0, x0=90),
    ]
    b = _block_with_spans([spans])
    sent, formulas = _placeholderize(b, 0)
    assert sent == "The formula {v1} shows energy"
    assert len(formulas) == 1
    f = formulas[0]
    assert f["name"] == "f0_1.png" and f["h"] > 0 and f["bbox"].width > 0
    out = _restore_placeholders("公式 { v 1 } 说明了能量", formulas)
    assert out == "公式 \x01i\x021\x01/i\x02 说明了能量"
    # 占位符丢失 → None（回退原文）；幻觉占位符被删掉
    assert _restore_placeholders("没有占位符", formulas) is None
    assert _restore_placeholders("公式 {v1} {v9}", formulas) == "公式 \x01i\x021\x01/i\x02 "


def test_placeholderize_marks_mixed_emphasis_only():
    from app.formats.pdf import _placeholderize

    def span(text, font="TimesNewRomanPSMT", flags=0, x0=0):
        return {"text": text, "font": font, "size": 12.0, "flags": flags,
                "bbox": (x0, 0, x0 + 20, 12), "origin": (x0, 10)}

    mixed = _block_with_spans([[
        span("See "),
        span("this", flags=16, x0=20),
        span(" word", x0=40),
    ]])
    sent, formulas = _placeholderize(mixed)
    assert sent == "See {b}this{/b} word" and formulas == []

    italic = _block_with_spans([[
        span("A "),
        span("term", font="TimesNewRomanPS-ItalicMT", x0=12),
        span(" here", x0=30),
    ]])
    assert _placeholderize(italic)[0] == "A {i}term{/i} here"

    uniform = _block_with_spans([[span("All bold words here", flags=16)]])
    assert "{b}" not in _placeholderize(uniform)[0]


def test_placeholderize_no_formula_keeps_text_identical():
    from app.formats.pdf import _placeholderize

    spans = [{"text": "Plain sentence ", "font": "TimesNewRomanPSMT", "size": 12.0},
             {"text": "without any math.", "font": "ArialMT", "size": 12.0}]
    b = _block_with_spans([spans])
    b.text = "Plain sentence without any math."
    sent, formulas = _placeholderize(b)
    assert sent == b.text and formulas == []


def test_attention_formula_placeholders_end_to_end(tmp_path):
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    from app.formats.pdf import _placeholderize, extract_blocks

    blocks = extract_blocks(pymupdf.open(src))
    sent_meta = [_placeholderize(b) for b in blocks]
    n_ph = sum(len(f) for _, f in sent_meta)
    assert n_ph > 50  # 论文里有大量行内公式和角标
    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    text = pymupdf.open(out)[0].get_text() + pymupdf.open(out)[1].get_text()
    assert "{v" not in text  # 全部恢复成功


def test_cross_page_units_merge_continuation():
    """跨页续段合并：上页末块不以句末标点结尾、下页首块小写开头 → 合成一个翻译单元。"""
    from app.formats.pdf import TextBlock, _cross_page_units

    def blk(text, page, size=10.0, y=0):
        return TextBlock(page=page, rect=pymupdf.Rect(0, y, 100, y + 10), line_rects=[],
                         text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("It can be seen that the model performs quite", 0, y=700),
        blk("1 Although one could also use another projection.", 0, size=9.0, y=740),
        blk("4https://example.com/footnote", 0, size=9.0, y=760),  # 小一号脚注，不占页尾
        blk("well. We hypothesize that this has to do with it.", 1, y=60),
        blk("A new section starts here.", 1, y=90),
    ]
    units = _cross_page_units(blocks, 10.0)
    merged = [u for u in units if len(u) == 2]
    assert len(merged) == 1
    assert merged[0][0].text.endswith("quite") and merged[0][1].text.startswith("well")
    # 完整句子不合并
    blocks2 = [
        blk("This sentence is complete.", 0, y=700),
        blk("another paragraph starts here.", 1, y=60),
    ]
    assert all(len(u) == 1 for u in _cross_page_units(blocks2, 10.0))


def test_split_translation_at_clause_boundary():
    from app.formats.pdf import _split_translation

    t = "可以看出，基线模型的性能优于采样模型。这是意料之中的。然而还可以看出模型的表现相当好。我们推测原因在此。"
    a, b = _split_translation(t, 0.6)
    assert a.endswith(("。", "，", "；")) and len(a) > len(b) * 0.8
    assert a + b == t
    # 拉丁文本按空格切
    a2, b2 = _split_translation("the quick brown fox jumps over the lazy dog", 0.5)
    assert a2.endswith("fox") and a2 + b2 == "the quick brown fox jumps over the lazy dog"


def test_equation_vs_prose_with_formula():
    """公式行（数学字符多、正文词少）跳过；含公式的正常句子照翻。"""
    from app.formats.pdf import extract_blocks

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)
    # 含公式的句子：正文词多，必须翻
    page.insert_textbox(pymupdf.Rect(30, 20, 480, 45),
                        "The sequence y_1, y_2, ..., y_T has a variable number of tokens here", fontsize=11)
    # 公式行：几乎没正文词，跳过
    page.insert_textbox(pymupdf.Rect(30, 60, 480, 85), "x + y = z", fontsize=11)
    blocks = extract_blocks(doc)
    texts = [b.text for b in blocks]
    assert any("sequence" in t for t in texts)
    assert not any("x + y = z" in t for t in texts)


def test_formula_runs_merge_requires_deep_x_overlap():
    """跨 dict 行的 run 归并：上下标（x 深度交叠）并成一个公式；
    相邻两行各自的公式（x 只沾 1pt 边）不能并。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None):
        return {"text": text, "font": font, "size": size,
                "bbox": (x0, y0, x1 if x1 is not None else x0 + 10,
                         y1 if y1 is not None else y0 + 10),
                "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2)}

    # y 和上标 T 拆在两行，x 几乎全叠 → 并成一个
    b = _block_with_spans([
        [span("see ", "TimesNewRomanPSMT", 10, 0, 50), span("y", "CMMI10", 10, 50, 50)],
        [span("T", "CMR7", 7, 55, 44, x1=59, y1=50)],
    ], size=10.0)
    sent, formulas = _placeholderize(b, 0)
    assert len(formulas) == 1 and sent.count("{v1}") == 1

    # θ⋆ 在第一行末、X^i 在第二行，x 只沾 1pt → 两个独立公式
    b2 = _block_with_spans([
        [span("use ", "TimesNewRomanPSMT", 10, 0, 50),
         span("θ", "CMMI10", 10, 50, 50, x1=58),
         span(" then", "TimesNewRomanPSMT", 10, 62, 50)],
        [span("and ", "TimesNewRomanPSMT", 10, 0, 65),
         span("X", "CMMI10", 10, 57, 65, x1=65, y1=75),
         span("i", "CMR7", 7, 60, 62, x1=63, y1=68)],
    ], size=10.0)
    sent2, formulas2 = _placeholderize(b2, 1)
    assert len(formulas2) == 2
    assert "{v1}" in sent2 and "{v2}" in sent2


def test_marker_png_unique_per_formula():
    """标记像素按全局序号编码：同色标记会被 MuPDF 去重成同一 xref，落位全串。"""
    from app.formats.pdf import _marker_png

    def decode(png):
        pix = pymupdf.Pixmap(png)
        p = pix.pixel(0, 0)
        assert p[2] == 200  # 标记签名
        return p[0] + (p[1] << 8) - 1

    assert decode(_marker_png(0)) == 0
    assert decode(_marker_png(255)) == 255
    assert decode(_marker_png(300)) == 300  # 超过单字节上限也能区分
    assert len({_marker_png(i) for i in range(300)}) == 300


def test_rendered_formulas_placed_once_in_own_slots(tmp_path):
    """矢量透传端到端：两个块的公式各落各的槽位、只画一次、画在最后一条内容流。

    回归场景：块内序号命名的标记图互相去重，所有块的公式 #1 全堆到第一个块的
    槽位；公式绘制写进 stream[0] 又被后续块的涂白标记盖住上半截。
    """
    from app.formats.pdf import TextBlock, _marker_png, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((50, 50), "alpha BETA gamma", fontsize=10)
    page.insert_text((50, 100), "delta EPSILON zeta", fontsize=10)
    doc.save(src)
    doc.close()

    d = pymupdf.open(src)
    r1 = d[0].search_for("BETA")[0]
    r2 = d[0].search_for("EPSILON")[0]

    def record(name, mid, rect):
        return {"bbox": rect, "name": name, "mid": mid,
                "w": round(rect.width, 1), "h": round(rect.height, 1), "d": 2.0,
                "text": "x", "has_img": True,
                "png": d[0].get_pixmap(dpi=150, clip=rect).tobytes("png"),
                "spans": [{"bbox": tuple(rect), "text": "x", "size": 10,
                           "origin": (rect.x0, rect.y1 - 2)}]}

    f1, f2 = record("a_1.png", 0, r1), record("b_1.png", 1, r2)
    d.close()
    archive = pymupdf.Archive()
    archive.add(_marker_png(0), "ph_a_1.png")
    archive.add(_marker_png(1), "ph_b_1.png")

    def blk(text, y):
        rect = pymupdf.Rect(40, y, 200, y + 15)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text,
                         size=10, color="#000", bold=False)

    b1, b2 = blk("alpha BETA gamma", 40), blk("delta EPSILON zeta", 90)
    t1, t2 = "译文 \x01i\x021\x01/i\x02 甲", "译文 \x01i\x021\x01/i\x02 乙"
    out = _render_translated(src, [b1, b2], [t1, t2], "zh-CN",
                             {id(b1): [f1], id(b2): [f2]}, archive)
    page = out[0]
    streams = [out.xref_stream(x) or b"" for x in page.get_contents()]
    # 每个公式只画一次，且集中画在最后一条内容流（不被后续块的白块盖住）
    assert sum(s.count(b"/FmF_a_1_png Do") for s in streams) == 1
    assert sum(s.count(b"/FmF_b_1_png Do") for s in streams) == 1
    assert b"/FmF_a_1_png Do" in streams[-1] and b"/FmF_b_1_png Do" in streams[-1]
    # 原文被 redact 后公式 glyph 由透传补回：各出现一次，槽位不同
    assert len(page.search_for("BETA")) == 1
    assert len(page.search_for("EPSILON")) == 1
    hit1, hit2 = page.search_for("BETA")[0], page.search_for("EPSILON")[0]
    # 各落各的槽位（块 rect 附近）：带缩放的 cm 平移量算错会让公式整体往下漂
    assert pymupdf.Rect(30, 30, 210, 75).contains(hit1)
    assert pymupdf.Rect(30, 80, 210, 125).contains(hit2)


def test_body_font_symbols_and_script_bases_become_formulas():
    """Times 里的希腊字母是公式。贴在角标上的单字母并进角标。
    斜体单词、孤立等号、引用上标前的 a 仍留在正文。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None, flags=0):
        return {"text": text, "font": font, "size": size, "flags": flags,
                "bbox": (x0, y0, x1 if x1 is not None else x0 + 8,
                         y1 if y1 is not None else y0 + 10),
                "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2)}

    greek = _block_with_spans([[
        span("see ", "TimesNewRomanPSMT", 10, 0, 50),
        span("α", "TimesNewRomanPSMT", 10, 28, 50, x1=36),
        span(" here", "TimesNewRomanPSMT", 10, 38, 50),
    ]], size=10.0)
    sent, formulas = _placeholderize(greek)
    assert sent == "see {v1} here" and len(formulas) == 1

    base = _block_with_spans([
        [span("let ", "TimesNewRomanPSMT", 10, 0, 50),
         span("x", "TimesNewRomanPSMT", 10, 30, 50, x1=36)],
        [span("i", "TimesNewRomanPSMT", 7, 34, 44, x1=38, y1=50)],
    ], size=10.0)
    sent_b, formulas_b = _placeholderize(base)
    assert sent_b == "let {v1}" and len(formulas_b) == 1
    assert "x" not in sent_b

    cite = _block_with_spans([[
        span("a", "TimesNewRomanPSMT", 10, 0, 50, x1=6),
        span("1", "TimesNewRomanPSMT", 7, 8, 46, x1=12, y1=52),
        span(" word", "TimesNewRomanPSMT", 10, 14, 50),
    ]], size=10.0)
    sent_c, _ = _placeholderize(cite)
    assert sent_c.startswith("a") and "{v1}" in sent_c

    equals = _block_with_spans([[
        span("E ", "TimesNewRomanPSMT", 10, 0, 50),
        span("=", "TimesNewRomanPSMT", 10, 14, 50, x1=22),
        span(" mc", "TimesNewRomanPSMT", 10, 24, 50),
    ]], size=10.0)
    assert "=" in _placeholderize(equals)[0]


def test_math_roman_pieces_join_the_formula_and_ordinals_stay_text():
    """贴着公式的 CMR 数字、括号、算子名并进同一占位符。
    i^{th} 留在正文，送翻成 i-th。远处的数字、Times 里的数字不收。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, y0, x1, y1=None):
        return {"text": text, "font": font, "size": size,
                "bbox": (x0, y0, x1, y1 if y1 is not None else y0 + size),
                "origin": (x0, (y1 if y1 is not None else y0 + size) - 2)}

    equation = _block_with_spans([[
        span("When", "NimbusRomNo9L-Regu", 10, 0, 50, 28),
        span("ε", "CMMI10", 10, 30, 50, 36),
        span("i", "CMMI7", 7, 36, 53, 40, 60),
        span("=", "CMR10", 10, 44, 50, 52),
        span(" ", "CMR10", 10, 52, 50, 56),
        span("1", "CMR10", 10, 56, 50, 62),
        span(",", "NimbusRomNo9L-Regu", 10, 62, 50, 66),
    ]], size=10.0)
    sent, formulas = _placeholderize(equation)
    assert len(formulas) == 1
    assert sent == "When{v1},"
    assert "1" in formulas[0]["text"]

    wrapped = _block_with_spans([
        [span("taken as the ", "NimbusRomNo9L-Regu", 10, 40, 80, 130),
         span("arg max", "CMR10", 10, 130, 80, 180),
         span("s", "CMMI7", 7, 180, 84, 186, 90),
         span("P", "CMMI10", 10, 188, 80, 196),
         span("(", "CMR10", 10, 197, 80, 203),
         span("y", "CMMI10", 10, 203, 80, 210),
         span("=", "CMR10", 10, 220, 80, 228)],
        [span("s", "CMMI10", 10, 40, 94, 46),
         span(")", "CMR10", 10, 48, 94, 54),
         span(". This", "NimbusRomNo9L-Regu", 10, 56, 94, 90)],
    ], size=10.0)
    sent_w, formulas_w = _placeholderize(wrapped)
    assert sent_w.count("{v") == 1
    assert "arg" not in sent_w and "P" not in sent_w
    assert "{v1}." in sent_w and " ." not in sent_w
    assert "arg max" in formulas_w[0]["text"]
    assert len(formulas_w[0]["parts"]) == 2
    assert formulas_w[0]["parts"][0]["bbox"].x0 >= 130
    assert formulas_w[0]["parts"][1]["bbox"].x0 <= 40

    ordinal = _block_with_spans([[
        span("of the ", "NimbusRomNo9L-Regu", 10, 0, 50, 40),
        span("i", "CMMI10", 10, 40, 50, 44),
        span("th", "CMMI7", 7, 44, 44, 54, 51),
        span(" mini-batch", "NimbusRomNo9L-Regu", 10, 56, 50, 120),
    ]], size=10.0)
    sent_o, formulas_o = _placeholderize(ordinal)
    assert formulas_o == []
    assert sent_o == "of the i-th mini-batch"

    distant = _block_with_spans([[
        span("x", "CMMI10", 10, 0, 50, 8),
        span(" see page ", "NimbusRomNo9L-Regu", 10, 10, 50, 70),
        span("1", "CMR10", 10, 80, 50, 86),
    ]], size=10.0)
    sent_d, formulas_d = _placeholderize(distant)
    assert len(formulas_d) == 1 and "1" in sent_d

    times_digit = _block_with_spans([[
        span("x", "CMMI10", 10, 0, 50, 8),
        span("1", "TimesNewRomanPSMT", 10, 10, 50, 16),
    ]], size=10.0)
    sent_t, _ = _placeholderize(times_digit)
    assert "1" in sent_t


def test_formula_crop_trims_side_whitespace_to_the_ink():
    """字框里的前导空格不再留在公式图两侧。式子内部的空格还在。
    左边叠进来的邻字不算本公式的墨迹。"""
    from app.formats.pdf import _tighten_formula_sides

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 80), "word", fontsize=12, fontname="tiro")
    page.insert_text((100, 80), " y", fontsize=12, fontname="tiro")
    page.insert_text((40, 130), "a = b", fontsize=12, fontname="times-roman")
    spans = [
        span
        for block in page.get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    ]
    spaced = next(span for span in spans if span["text"] == " y")
    equation = next(span for span in spans if "a = b" in span["text"])
    word = next(span for span in spans if span["text"] == "word")

    wide = pymupdf.Rect(spaced["bbox"])
    part = {
        "bbox": wide,
        "w": round(wide.width, 1),
        "dx": 0.0,
        "spans": [spaced],
        "curves": [],
    }
    _tighten_formula_sides(part, page)
    tight = pymupdf.Rect(part["bbox"])
    assert tight.x0 >= wide.x0 + 1.5
    assert tight.x1 <= wide.x1 + 0.2
    assert abs(tight.width - part["w"]) < 0.15
    assert part["dx"] == round(tight.x0 - wide.x0, 1)
    # 字母还在框里，左右空白都收到窄边
    letter = next(
        pymupdf.Rect(char["bbox"])
        for span in page.get_text("rawdict")["blocks"]
        if span.get("type") == 0
        for line in span["lines"]
        for raw in line["spans"]
        for char in raw["chars"]
        if char["c"] == "y" and abs(char["bbox"][0] - 100) < 30
    )
    assert tight.x0 <= letter.x0 + 0.2
    assert tight.x1 >= letter.x1 - 0.2
    assert tight.x0 >= letter.x0 - 1.0
    assert tight.x1 <= letter.x1 + 1.0

    inner = pymupdf.Rect(equation["bbox"])
    kept = {"bbox": inner, "w": round(inner.width, 1), "spans": [equation], "curves": []}
    _tighten_formula_sides(kept, page)
    kept_box = pymupdf.Rect(kept["bbox"])
    assert kept_box.width > inner.width * 0.7
    assert kept_box.x0 < inner.x0 + 2
    assert kept_box.x1 > inner.x1 - 2

    word_box = pymupdf.Rect(word["bbox"])
    overlapped = pymupdf.Rect(word_box.x1 - 4, wide.y0, wide.x1, wide.y1)
    neighbor = {
        "bbox": overlapped,
        "w": round(overlapped.width, 1),
        "spans": [spaced],
        "curves": [],
    }
    _tighten_formula_sides(neighbor, page)
    trimmed = pymupdf.Rect(neighbor["bbox"])
    assert trimmed.x0 > word_box.x1 - 0.5

    page.draw_line((70, 78), (96, 78), width=0.8)
    bar = pymupdf.Rect(70, 77.2, 96, 78.8)
    with_bar = {
        "bbox": pymupdf.Rect(bar.x0, wide.y0, wide.x1, wide.y1),
        "w": 40.0,
        "spans": [spaced],
        "curves": [bar],
    }
    _tighten_formula_sides(with_bar, page)
    assert pymupdf.Rect(with_bar["bbox"]).x0 <= bar.x0 + 0.6
    doc.close()


def test_line_wrapped_formula_is_one_placeholder():
    """行末 y_{t-1}= 和下一行行首 s|h_{t-1} 是同一式子。一个占位符、两片，
    不并成横跨整行的大框。行中的公式不和下一行行首的公式并。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, y0, x1, y1=None):
        return {"text": text, "font": font, "size": size,
                "bbox": (x0, y0, x1, y1 if y1 is not None else y0 + size),
                "origin": (x0, (y1 if y1 is not None else y0 + size) - 2)}

    wrapped = _block_with_spans([
        [span("take arg max ", "TimesNewRomanPSMT", 10, 40, 80, 150),
         span("y", "CMMI10", 10, 470, 80, 478),
         span("=", "CMR10", 10, 490, 80, 504)],
        [span("s|h", "CMMI10", 10, 40, 92, 70),
         span(").", "TimesNewRomanPSMT", 10, 72, 92, 90)],
    ], size=10.0)
    sent, formulas = _placeholderize(wrapped)
    assert sent.count("{v") == 1
    assert "s|h" not in sent and "y=" not in sent.replace(" ", "")
    assert len(formulas) == 1
    parts = formulas[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["bbox"].x0 >= 470
    assert parts[1]["bbox"].x0 <= 40
    assert parts[0]["bbox"].x1 < 510 and parts[1]["bbox"].width < 40

    separate = _block_with_spans([
        [span("see ", "TimesNewRomanPSMT", 10, 40, 80, 70),
         span("x", "CMMI10", 10, 80, 80, 90),
         span(" then more words here", "TimesNewRomanPSMT", 10, 92, 80, 220)],
        [span("y", "CMMI10", 10, 40, 92, 50),
         span(" follows", "TimesNewRomanPSMT", 10, 54, 92, 110)],
    ], size=10.0)
    sent2, formulas2 = _placeholderize(separate)
    assert len(formulas2) == 2
    assert "{v1}" in sent2 and "{v2}" in sent2


def test_wrapped_formula_parts_stay_in_order(tmp_path):
    """两片按阅读顺序并排，后半段不再留在下一行行首的源文横坐标。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 80), "y= left", fontsize=11)
    page.insert_text((40, 100), "sh right", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    y_box = d[0].search_for("y=")[0]
    sh_box = d[0].search_for("sh")[0]
    d.close()

    def part(name, rect, text):
        return {
            "bbox": rect, "name": name, "w": round(rect.width, 1), "h": round(rect.height, 1),
            "d": 2.0, "raise": 0.0, "text": text, "has_img": True, "page": 0,
            "png": pymupdf.open(src)[0].get_pixmap(clip=rect, dpi=72).tobytes("png"),
            "spans": [{"bbox": tuple(rect), "text": text, "size": 10, "origin": (rect.x0, rect.y1 - 2)}],
        }

    first = part("f0_1.png", y_box, "y=")
    second = part("f0_1_p1.png", sh_box, "sh")
    first["parts"] = [first, second]
    first["has_img"] = True
    first["mid"] = 0
    rect = pymupdf.Rect(40, 40, 360, 80)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect, pymupdf.Rect(y_box), pymupdf.Rect(sh_box)],
        text="take arg max", size=10, color="#000", bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    out = _render_translated(src, [block], [f"也可以取 arg max({sentinel})。"], "zh-CN", {id(block): [first]})
    page = out[0]
    y_hit = page.search_for("y=")
    sh_hit = page.search_for("sh")
    assert y_hit and sh_hit
    assert y_hit[0].x0 < sh_hit[0].x0
    assert sh_hit[0].x0 - y_hit[0].x1 < 20
    assert abs(y_hit[0].y0 - sh_hit[0].y0) < 8


def test_placeholderize_absorbs_accent_over_formula():
    """ŷ 的 ^ 常是独立 span（正文字体、不被判为公式），留在正文会在公式图边
    多出一个孤符号：落在公式 bbox 上的重音符要并进公式、正文里删掉。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None):
        return {"text": text, "font": font, "size": size,
                "bbox": (x0, y0, x1 if x1 is not None else x0 + 10,
                         y1 if y1 is not None else y0 + 10),
                "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2)}

    b = _block_with_spans([
        [span("an estimate ", "TimesNewRomanPSMT", 10, 0, 50),
         span("y", "CMMI10", 10, 40, 50, x1=48, y1=60),
         span(" coming", "TimesNewRomanPSMT", 10, 60, 50)],
        [span("^", "CMR10", 10, 41, 42, x1=45, y1=48)],  # 帽子是独立 dict 行
    ], size=10.0)
    sent, formulas = _placeholderize(b, 0)
    assert "^" not in sent
    assert len(formulas) == 1
    assert formulas[0]["bbox"].y0 <= 42  # 符头顶部包进裁剪区域
    # 远离公式的 ^ 不吸收（如正文里确实有个符号）
    b2 = _block_with_spans([
        [span("y", "CMMI10", 10, 0, 50, x1=8, y1=60),
         span(" then ^ alone", "TimesNewRomanPSMT", 10, 100, 50)],
    ], size=10.0)
    sent2, _ = _placeholderize(b2, 0)
    assert "^" in sent2



def test_same_line_gap_stays_inside_one_column():
    """公式把一行拆成两块时，间隙小于栏沟就接上。另一栏即使只隔 4pt 也不接。"""
    from app.formats.pdf import TextBlock, _merge_visual_lines
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, x1, y=100.0):
        rect = pymupdf.Rect(x0, y, x1, y + 11)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text,
                         size=10, color="#000", bold=False)

    one = PageGeometry(400, (Column(30, 380),), ())
    merged = _merge_visual_lines(
        [blk("the value", 40, 90), blk("x", 93, 102), blk("is large", 105, 160)],
        {0: one},
    )
    assert len(merged) == 1
    assert merged[0].text == "the value x is large"

    two = PageGeometry(400, (Column(30, 180), Column(186, 380)), (), reads_in_columns=True)
    split = _merge_visual_lines(
        [blk("left side", 40, 180), blk("continues", 184, 280)],
        {0: two},
    )
    assert len(split) == 2


def test_cjk_wrap_merges_and_short_labels_stay():
    """左缘对齐、上行够长的中文换行接回一段。短标签和缩进的新段不接。"""
    from app.formats.pdf import TextBlock, _cross_page_units, _merge_continuations

    def blk(text, y, x0=40.0, x1=280.0, page=0, size=10.0):
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text,
                         size=size, color="#000", bold=False)

    wrapped = _merge_continuations([
        blk("这是一行已经写到栏边还没有句号的正文", 100),
        blk("下一行从左缘接着写。", 114, x1=180),
    ])
    assert len(wrapped) == 1
    assert wrapped[0].text.endswith("接着写。")

    labels = _merge_continuations([
        blk("姓名", 100, x1=70),
        blk("年龄", 114, x1=70),
    ])
    assert len(labels) == 2

    indented = _merge_continuations([
        blk("这是一行已经写到栏边还没有句号的正文", 100),
        blk("新的一段从缩进开始写起。", 114, x0=62, x1=220),
    ])
    assert len(indented) == 2

    pages = _cross_page_units([
        blk("上一页在这里被切断还没写完", 700, page=0),
        blk("下一页从同一边距继续。", 60, page=1),
    ], 10.0)
    assert any(len(unit) == 2 for unit in pages)
    heading = _cross_page_units([
        blk("上一页在这里被切断还没写完", 700, page=0),
        blk("1 引言", 60, page=1),
    ], 10.0)
    assert all(len(unit) == 1 for unit in heading)


def test_caption_label_without_colon_merges():
    """「图 2」「Fig. 1.」后面的注记要接上。标题和表格行不接。"""
    from app.formats.pdf import TextBlock, _merge_caption_fragments

    def blk(text, x0, y0, x1, size=10.0):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 10)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text,
                         size=size, color="#000", bold=False)

    zh = _merge_caption_fragments([
        blk("图 2", 60, 100, 88),
        blk("衰减曲线示意。", 60, 112, 160),
    ])
    assert len(zh) == 1
    assert zh[0].text == "图 2衰减曲线示意。"

    fig = _merge_caption_fragments([
        blk("Fig. 1.", 60, 100, 98),
        blk("Examples of decay", 104, 100, 220),
    ])
    assert len(fig) == 1

    title = _merge_caption_fragments([
        blk("Introduction", 40, 100, 130, size=12.0),
        blk("Scholarly works.", 40, 130, 150, size=10.0),
    ])
    assert len(title) == 2


def test_caption_fragments_merge_into_one_block():
    """'Figure 2:' + 'Examples of decay' + 'schedules.' 三个碎片要拼回一个块：
    各翻各的再塞回小框会换行丑陋、译文也断。标题和无冒号的块不受影响。"""
    from app.formats.pdf import TextBlock, _merge_caption_fragments

    def blk(text, x0, y0, x1, size=10.0):
        return TextBlock(page=0, rect=pymupdf.Rect(x0, y0, x1, y0 + 10),
                         line_rects=[pymupdf.Rect(x0, y0, x1, y0 + 10)],
                         text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("Figure 2:", 60, 100, 105),
        blk("Examples of decay", 112, 100, 220),
        blk("schedules.", 60, 112, 110),
    ]
    merged = _merge_caption_fragments(blocks)
    assert len(merged) == 1
    assert merged[0].text == "Figure 2: Examples of decay schedules."
    assert len(merged[0].line_rects) == 3

    # 标题 + 正文（无冒号、大写开头）不合并
    b2 = [blk("Introduction", 40, 100, 130, size=12.0),
          blk("Scholarly works.", 40, 130, 130, size=10.0)]
    assert len(_merge_caption_fragments(b2)) == 2

    # 表格行（大写开头、无冒号）不合并
    b3 = [blk("Always Sampling", 60, 100, 160),
          blk("Scheduled Sampling 1", 60, 112, 190),
          blk("Baseline LSTM", 60, 124, 150)]
    assert len(_merge_caption_fragments(b3)) == 3


def test_heading_beside_the_first_line_is_not_absorbed():
    """短标题只和正文第一行并排。用块的并集会把它吞进正文。"""
    from app.formats.pdf import TextBlock, _merge_visual_lines

    memo_line = pymupdf.Rect(40, 100, 72, 111)
    memo = TextBlock(page=0, rect=memo_line, line_rects=[memo_line], text="Memo.",
                     size=9, color="#000", bold=False)
    first = pymupdf.Rect(80, 100, 220, 111)
    later = pymupdf.Rect(40, 114, 220, 125)
    para = TextBlock(page=0, rect=first | later, line_rects=[first, later],
                     text="The space of plan alternatives is large.",
                     size=9, color="#000", bold=False)
    assert len(_merge_visual_lines([memo, para])) == 2


def test_right_fragment_joins_the_following_line():
    """同一行右侧的半句接回下一行，不并进左边已经结束的那句。"""
    from app.formats.pdf import TextBlock, _merge_line_tails

    left_lines = [pymupdf.Rect(40, 100, 180, 112), pymupdf.Rect(40, 112, 90, 124)]
    left = TextBlock(page=0, rect=left_lines[0] | left_lines[1], line_rects=left_lines,
                     text="The paragraph already ended.", size=9, color="#000", bold=False)
    frag_line = pymupdf.Rect(100, 112, 180, 124)
    fragment = TextBlock(page=0, rect=frag_line, line_rects=[frag_line],
                         text="It is distinguished from", size=9, color="#000", bold=False)
    next_line = pymupdf.Rect(40, 126, 170, 138)
    nxt = TextBlock(page=0, rect=next_line, line_rects=[next_line],
                    text="other optimizers in several ways:", size=9, color="#000", bold=False)
    merged = _merge_line_tails([left, fragment, nxt])
    assert len(merged) == 2
    assert merged[0].text == "The paragraph already ended."
    assert merged[1].text.startswith("It is distinguished from")
    assert merged[1].text.endswith("several ways:")

    shared_lines = [pymupdf.Rect(40, 126, 200, 138), pymupdf.Rect(40, 140, 200, 152)]
    shared = TextBlock(page=0, rect=shared_lines[0] | shared_lines[1], line_rects=shared_lines,
                       text="For example the next sentence starts here.",
                       size=9, color="#000", bold=False)
    short = TextBlock(page=0, rect=pymupdf.Rect(40, 126, 110, 138),
                      line_rects=[pymupdf.Rect(40, 126, 110, 138)],
                      text="expression-specific.", size=9, color="#000", bold=False)
    blocked = _merge_line_tails([left, fragment, shared, short])
    assert all("expression-specific." != block.text or "distinguished" not in block.text
               for block in blocked)
    assert any(block.text == "expression-specific." for block in blocked)


def test_same_line_sentence_joins_the_paragraph_and_keeps_side_labels():
    """行尾起的下一句并进上一段。短粗体侧标和图内标签不并。续行即使被后一句挡住也先接上。"""
    from app.formats.pdf import TextBlock, _merge_same_line_sentences, _with_source_bold

    def line(text, x0, y0, x1, y1, bold=False):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text,
                         size=9, color="#000", bold=bold)

    def para(text, boxes, bold=False):
        rects = [pymupdf.Rect(*box) for box in boxes]
        union = rects[0]
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=0, rect=union, line_rects=rects, text=text,
                         size=9, color="#000", bold=bold)

    intro = para("In this paper we describe Orca for demanding analytics workloads.",
                 [(316.8, 307.0, 555.9, 316.0), (316.8, 338.4, 448.3, 347.3)])
    distinguished = para("It is distinguished from other optimizers in several important ways:",
                         [(457.3, 338.4, 555.9, 347.3), (316.8, 348.8, 491.7, 357.8)])
    extensibility = para("Extensibility. Certain optimizations are dealt with as an afterthought.",
                         [(316.8, 441.9, 555.9, 450.9), (339.2, 483.8, 384.5, 492.7)])
    multiphase = para("Multi-phase optimizers are notoriously difficult to extend.",
                      [(394.4, 483.8, 556.0, 492.7), (339.2, 494.2, 555.9, 503.2)])
    ended = para("In order to derive statistics.",
                 [(316.8, 386.0, 556.0, 396.0), (316.8, 406.9, 406.2, 415.9)])
    promise = line("Statistics promise computation is", 416.0, 406.9, 556.0, 415.9)
    specific = line("expression-specific.", 316.8, 417.4, 392.7, 426.4)
    example = para("For example the next sentence starts here.",
                   [(401.6, 417.4, 555.9, 426.4), (316.8, 428.0, 556.0, 438.0)])
    label = line("Property Enforcement.", 316.8, 350.0, 426.7, 359.0, bold=True)
    body = para("Orca includes an extensible framework for describing query requirements.",
                [(438.2, 350.0, 555.9, 359.0), (316.8, 360.5, 555.9, 369.5)])
    diagram = line("Get(T1)%", 360.0, 200.0, 381.0, 209.0)
    diagram2 = line("Get(T2)%", 394.6, 200.0, 416.0, 209.0)
    merged = _merge_same_line_sentences([
        intro, distinguished, extensibility, multiphase, label, body,
        ended, promise, example, specific, diagram, diagram2,
    ])
    texts = [block.text for block in merged]
    assert any(text.startswith("In this paper") and "distinguished" in text for text in texts)
    assert any(text.startswith("Extensibility.") and "Multi-phase" in text for text in texts)
    joined = next(text for text in texts if "Statistics promise" in text)
    assert "expression-specific." in joined
    assert "For example" in joined
    assert "In order to derive" in joined
    assert any(text == "Property Enforcement." for text in texts)
    assert any(text.startswith("Orca includes") for text in texts)
    assert any(text == "Get(T1)%" for text in texts)
    assert any(text == "Get(T2)%" for text in texts)

    lead = TextBlock(
        page=0, rect=pymupdf.Rect(40, 40, 200, 52),
        line_rects=[pymupdf.Rect(40, 40, 200, 52)],
        text="Modularity. Using a highly extensible abstraction.",
        size=9, color="#000", bold=False,
        span_lines=[[
            {"text": "Modularity.", "font": "CMBX9", "flags": 16},
            {"text": " Using a highly extensible abstraction.", "font": "CMR9", "flags": 4},
        ]],
    )
    assert _with_source_bold(lead, "模块化。借助高度可扩展的抽象。") == "{b}模块化。{/b}借助高度可扩展的抽象。"
    assert _with_source_bold(lead, "{b}模块化。{/b}借助。") == "{b}模块化。{/b}借助。"
    memo = line("Memo.", 40, 100, 72, 112, bold=True)
    memo.span_lines = [[{"text": "Memo.", "font": "CMBX9", "flags": 16}]]
    assert _with_source_bold(memo, "备忘录。") == "{b}备忘录。{/b}"


def test_finished_condition_line_does_not_swallow_the_next_paragraph():
    """公式条件行以句号结束时，不把下面回到栏边的正文接进来。没写完的右半句仍接下一行。"""
    from app.formats.pdf import TextBlock, _continues_fragment, _merge_same_line_sentences

    formula = pymupdf.Rect(254.4, 329.7, 333.6, 340.5)
    condition = pymupdf.Rect(343.5, 329.9, 384.8, 339.9)
    host = TextBlock(
        page=0, rect=formula | condition, line_rects=[formula, condition],
        text="f(ht-1, yt-1, xt; θ) otherwise.", size=10, color="#000", bold=False,
    )
    guest_lines = [pymupdf.Rect(108.0, 349.4, 504.0, 361.1),
                   pymupdf.Rect(108.0, 360.6, 305.0, 370.5)]
    guest = TextBlock(
        page=0, rect=guest_lines[0] | guest_lines[1], line_rects=guest_lines,
        text="where oh is a vector of 0's with same dimensionality as ht's.",
        size=10, color="#000", bold=False,
    )
    assert not _continues_fragment(host, guest)
    merged = _merge_same_line_sentences([host, guest])
    assert [block.text for block in merged] == [host.text, guest.text]
    assert merged[0].rect.y1 < guest.rect.y0

    left = pymupdf.Rect(316.8, 406.9, 406.2, 415.9)
    right = pymupdf.Rect(416.0, 406.9, 556.0, 415.9)
    open_host = TextBlock(
        page=0, rect=left | right, line_rects=[left, right],
        text="derive statistics. Statistics promise computation is",
        size=9, color="#000", bold=False,
    )
    next_line = pymupdf.Rect(316.8, 417.4, 392.7, 426.4)
    nxt = TextBlock(
        page=0, rect=next_line, line_rects=[next_line],
        text="expression-specific.", size=9, color="#000", bold=False,
    )
    assert _continues_fragment(open_host, nxt)
    joined = _merge_same_line_sentences([open_host, nxt])
    assert len(joined) == 1
    assert "Statistics promise" in joined[0].text
    assert joined[0].text.endswith("expression-specific.")


def test_cjk_bold_strokes_once(tmp_path):
    """汉字粗体描边，提取出来仍是一个字。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=400, height=200)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 280, 80)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect],
        text="Modularity. Using a highly extensible abstraction of metadata.",
        size=9, color="#000000", bold=False,
        span_lines=[[
            {"text": "Modularity.", "font": "CMBX9", "flags": 16},
            {"text": " Using a highly extensible abstraction of metadata.", "font": "CMR9", "flags": 4},
        ]],
    )
    out = _render_translated(src, [block], ["模块化。借助高度可扩展的抽象来描述元数据。"], "zh-CN")
    text = out[0].get_text()
    assert text.count("模") == 1
    assert "2 Tr" in out[0].read_contents().decode("latin1")


def test_shared_baseline_continuation_is_one_translation():
    """下一行和后一段共基线时不并成外框，但仍成对送翻，不插分界，也不接左边已经结束的句子。"""
    from app.formats.pdf import TextBlock, _cross_page_units, _merge_line_tails, _prepare_unit

    def line(text, x0, y0, x1, y1):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text,
                         size=9, color="#000", bold=False)

    def para(text, lines):
        rects = [pymupdf.Rect(*box) for box in lines]
        union = rects[0]
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=0, rect=union, line_rects=rects, text=text,
                         size=9, color="#000", bold=False)

    ended = para("In order to derive statistics.", [(316.8, 386.0, 556.0, 396.0),
                                                    (316.8, 406.9, 406.2, 415.9)])
    promise = line("Statistics promise computation is", 416.0, 406.9, 556.0, 415.9)
    specific = line("expression-specific.", 316.8, 417.4, 392.7, 426.4)
    example = para("For example the next sentence starts here.",
                   [(401.6, 417.4, 555.9, 426.4), (316.8, 428.0, 556.0, 438.0)])
    figure = para("Figure 5 ends.", [(316.8, 616.0, 556.0, 626.0), (316.8, 626.6, 346.0, 635.6)])
    requests = line("For example, InnerJoin(T1,T2) on (a=b) requests", 353.5, 626.6, 555.9, 635.6)
    histograms = line("histograms on T1.a and T2.b.", 316.8, 637.1, 441.3, 646.0)
    requested = para("The requested histograms are used later.",
                     [(449.4, 637.1, 555.9, 646.0), (316.8, 648.0, 556.0, 658.0)])
    heading = line("(2) Statistics Derivation.", 316.8, 200.0, 435.3, 212.0)
    beside = para("At the end of exploration the optimizer stops.",
                  [(444.4, 200.0, 556.0, 212.0), (316.8, 214.0, 556.0, 226.0)])
    # 另一栏会插在半句和续行中间，而且已经写到更下面。不能因此把续行丢掉。
    other_column = line("ORDER BY T1.a where the distribution is hashed.", 53.8, 434.7, 280.0, 446.0)
    blocks = [heading, beside, ended, promise, example, other_column, specific,
              figure, requests, requested, histograms]
    merged = _merge_line_tails(blocks)
    assert any(block.text == "expression-specific." for block in merged)
    assert any(block.text == "histograms on T1.a and T2.b." for block in merged)
    units = _cross_page_units(blocks, 9.0)
    pairs = [unit for unit in units if len(unit) == 2]
    assert pairs == [[promise, specific], [requests, histograms]]
    sent, _, _ = _prepare_unit(pairs[0], 0)
    assert "{|}" not in sent
    assert "expression-specific." in sent
    assert "Statistics promise computation is" in sent


def test_column_wrap_joins_when_geometry_misses_the_gutter():
    """几何栏不准时，仍按正文左缘把左栏页尾接到右栏页首。图注不占这个位置。"""
    from app.formats.pdf import TextBlock, _cross_page_units
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, y, page=0, x1=None):
        rect = pymupdf.Rect(x0, y, x1 if x1 is not None else x0 + 180, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text,
                         size=9, color="#000", bold=False)

    wide = PageGeometry(600, (Column(40, 560),), ())
    filler = blk("Earlier sentence in the left column ends.", 40, 660)
    host = blk("Effective usage of CPUs trans-", 40, 700)
    caption = blk("Figure 8: Optimization jobs dependency graph", 300, 40, x1=520)
    tail = blk("lates to better query plans.", 300, 80)
    later = blk("Parallelizing the optimizer is crucial.", 300, 100)
    units = _cross_page_units([filler, host, caption, tail, later], 9.0, geometries=[wide])
    assert [unit for unit in units if len(unit) == 2] == [[host, tail]]
    # 不传几何时，同页两栏仍按原来的规则分开
    assert all(len(unit) == 1 for unit in _cross_page_units(
        [filler, host, caption, tail, later], 9.0,
    ))

    listing = blk("<dxl:Metadata SystemIds=\"0\">", 40, 40, page=1)
    code = blk("1 0x000e8106df gpos::CException::Raise", 40, 56, page=1)
    prose = blk("tomer issues in the optimizer.", 40, 80, page=1)
    prose2 = blk("The dump can be replayed later.", 40, 96, page=1)
    cus = blk("reproduce and debug cus-", 300, 700)
    cus2 = blk("The tool captures the optimizer state.", 300, 660)
    joined = _cross_page_units(
        [cus2, cus, listing, code, prose, prose2], 9.0, geometries=[wide, wide],
    )
    assert [unit for unit in joined if len(unit) == 2] == [[cus, prose]]


def test_capital_continuation_joins_across_columns_only():
    """换栏可以接在专有名词上。同一栏的大写换行仍分开。"""
    from app.formats.pdf import TextBlock, _cross_page_units, _merge_continuations
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, y, page=0):
        rect = pymupdf.Rect(x0, y, x0 + 180, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text,
                         size=9, color="#000", bold=False)

    wide = PageGeometry(600, (Column(40, 560),), ())
    host = blk("query engines that allow", 300, 700)
    host2 = blk("Several efforts have addressed this.", 300, 660)
    tail = blk("SQL-based processing of data in HDFS.", 40, 60, page=1)
    tail2 = blk("The exchange avoids another platform.", 40, 80, page=1)
    units = _cross_page_units([host2, host, tail, tail2], 9.0, geometries=[wide, wide])
    assert [unit for unit in units if len(unit) == 2] == [[host, tail]]
    assert all(len(unit) == 1 for unit in _cross_page_units([host2, host, tail, tail2], 9.0))

    same_column = _merge_continuations([
        blk("the previous line does not end", 40, 100),
        blk("SQL-based processing starts a new sentence.", 40, 114),
    ])
    assert len(same_column) == 2

    # 另一栏没形成左缘时，页末左栏也不能把下一页大写开头的新段接进来。
    def edge(text, x0, y, page, x1=280.0):
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text,
                         size=9, color="#000", bold=False)

    two = PageGeometry(612, (Column(40, 290), Column(316, 560)), ())
    host = edge("tunately even with this setting, we were", 53.8, 700, 0)
    earlier = edge("To obtain a better coverage across systems,", 53.8, 680, 0)
    nxt = edge("For queries where HAWQ has the most speedups.", 53.8, 56, 1)
    nxt2 = edge("Impala joins the fact tables first.", 53.8, 72, 1)
    units = _cross_page_units(
        [earlier, host, nxt, nxt2], 9.0, geometries=[two, two],
    )
    assert all(len(unit) == 1 for unit in units)
    lower = edge("and the sentence goes on", 53.8, 56, 1)
    joined = _cross_page_units(
        [earlier, host, lower, nxt2], 9.0, geometries=[two, two],
    )
    assert [unit for unit in joined if len(unit) == 2] == [[host, lower]]


def test_same_page_continuations_merge():
    """被公式碎行拆断的段落（'…(like a' | 'sequence)…' | 'that belong…'）要并回
    一个块：各翻各的会产出半截译文、在断点强制换行。标题、表格行不受影响。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, x0=108.0, size=10.0, nwords_pad=0):
        return TextBlock(page=0, rect=pymupdf.Rect(x0, y, 504, y + 11),
                         line_rects=[pymupdf.Rect(x0, y, 504, y + 11)],
                         text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("We are considering supervised tasks where the training set is given (like a", 280),
        blk("sequence) while the target output is a sequence of tokens", 292),
        blk("that belong to a fixed known dictionary.", 303),
        blk("2.1 Model", 330),
        blk("Given a single pair the log probability can be computed as follows", 345),
    ]
    merged = _merge_continuations(blocks)
    assert len(merged) == 3
    assert merged[0].text.endswith("fixed known dictionary.")
    assert merged[1].text == "2.1 Model"

    # 上行有句末标点 → 不并
    b2 = [blk("This is a full sentence with several words.", 100),
          blk("another paragraph starts here with lowercase", 112)]
    assert len(_merge_continuations(b2)) == 2

    # 悬挂缩进大约 2.5em。2em 卡太紧时，特征列表的续行拆成两段并单独缩小。
    hanging = [
        blk("Modularity. Using a highly extensible abstraction meta-", 100, x0=40),
        blk("data and system description stay together.", 112, x0=62),
    ]
    assert len(_merge_continuations(hanging)) == 1
    ended = [
        blk("This feature list item already ended.", 100, x0=40),
        blk("data and system description stay together.", 112, x0=62),
    ]
    assert len(_merge_continuations(ended)) == 2


def test_smaller_body_under_a_heading_splits():
    """12pt 标题和 9pt 导语常在同一 dict 块。导语左缘只让出节号，不能当成缩进留在标题里。"""
    from app.formats.pdf import _split_lines

    items = [
        _line("4.", 316, 326, 674, size=12),
        _line("QUERY OPTIMIZATION", 338, 475, 674, size=12),
        _line("We describe Orca's optimization workflow in Section 4.1.", 326, 556, 689, size=9),
        _line("We then show how the process can be conducted in parallel.", 317, 556, 700, size=9),
    ]
    assert _split_lines(items) == [[0, 1], [2, 3]]
    # 角标短行字数不够，即使更小也不拆。纯数字行本来就会按符号行拆开，这里用序数后缀。
    script = [
        _line("the i", 40, 80, 10, size=12),
        _line("th", 82, 94, 6, size=7),
    ]
    assert _split_lines(script) == [[0, 1]]


def test_heading_does_not_turn_the_following_sentence_into_a_formula():
    """就算标题和导语还在一块里，9pt 正文也不能因为小于 12pt 就被裁成公式图。"""
    from app.formats.pdf import _placeholderize

    def span(text: str, size: float, x0: float, y0: float, font: str) -> dict[str, object]:
        return {
            "text": text, "font": font, "size": size, "flags": 16 if size >= 12 else 0,
            "bbox": (x0, y0, x0 + max(8, len(text) * size * 0.45), y0 + size),
            "origin": (x0, y0 + size * 0.8),
        }

    heading = [
        span("4. ", 12, 0, 0, "NimbusRomNo9L-Medi"),
        span("QUERY OPTIMIZATION", 12, 24, 0, "NimbusRomNo9L-Medi"),
    ]
    words = "We describe Orca optimization workflow in Section".split()
    body = [span(word + " ", 9, index * 42, 16, "CMR9") for index, word in enumerate(words)]
    sent, formulas = _placeholderize(_block_with_spans([heading, body], size=12), 0)
    assert formulas == []
    assert "{v" not in sent
    assert "workflow" in sent


def test_section_number_stays_with_heading():
    """'2.2' + 'Training' 两条 dict 行不能拆：纯数字被当符号行拆开后，数字被过滤
    不遮罩（原文残留）、标题单独翻，两者基线对不齐。"""
    from app.formats.pdf import _split_lines

    items = [
        ("2.2", pymupdf.Rect(108, 84, 120, 94), [{"text": "2.2", "size": 10.0}], None),
        ("Training", pymupdf.Rect(130, 84, 167, 94), [{"text": "Training", "size": 10.0}], None),
    ]
    assert _split_lines(items) == [[0, 1]]
    # 页码和正文不在同一可视行（纵向不重叠），照常拆
    items2 = [
        ("3", pymupdf.Rect(300, 760, 308, 770), [{"text": "3", "size": 10.0}], None),
        ("Next paragraph starts here", pymupdf.Rect(108, 700, 500, 712),
         [{"text": "Next paragraph starts here", "size": 10.0}], None),
    ]
    assert _split_lines(items2) == [[0], [1]]


def test_heading_number_kept_in_source_form():
    """标题章节号保留原文形式：模型按提示词译成中文数字（'第二章第一节'），
    学术论文里要回 '2.1'。段落里以数字开头的句子不受影响。"""
    from app.formats.pdf import _normalize_heading_number as norm

    assert norm("2 Proposed Approach", "第二章 所提方法") == "2 所提方法"
    assert norm("2.1 Model", "第二章第一节 模型") == "2.1 模型"
    assert norm("4.3 Speech Recognition", "第四章第三节 语音识别") == "4.3 语音识别"
    assert norm("2.1 Model", "2.1 模型") == "2.1 模型"  # 幂等
    # 长段落以数字开头：不动
    para = "40 dimensional log Mel filter banks and their first and second order derivatives were used"
    assert norm(para, "使用了 40 维对数 Mel 滤波器组") == "使用了 40 维对数 Mel 滤波器组"
    # 有句末标点的短句：不动
    assert norm("3 states remain.", "还剩 3 个状态。") == "还剩 3 个状态。"
    # 译文没有章节号时不加前缀；跳过块原文必须原样返回
    assert norm("10 epochs were used", "使用了 10 个轮次") == "使用了 10 个轮次"
    assert norm("2.5 hours later", "两个半小时后") == "两个半小时后"
    assert norm("2. Smith 2019", "2. Smith 2019") == "2. Smith 2019"
    assert norm("2 Smith 2019", "2. Smith 2019") == "2 Smith 2019"


def test_heading_number_skips_body_blocks():
    from app.formats.pdf import _heading_number_for

    assert _heading_number_for("p", "2 samples", "第二章 样本") == "第二章 样本"
    assert _heading_number_for("h2", "2 Proposed Approach", "第二章 所提方法") == "2 所提方法"
    # 与正文同字号的节标题（kind 是 p）：带点节号照样改回
    assert _heading_number_for("p", "2.1 Model", "第二章第一节 模型") == "2.1 模型"
    # 正文尺寸的标题但译文没有章节号：不加前缀
    assert _heading_number_for("p", "2.1 Model", "模型") == "模型"
    # 模型写成"第二节"（没有"章"）也改回
    assert _heading_number_for("h2", "2 Proposed Approach", "第二节 所提方法") == "2 所提方法"
    assert _heading_number_for("h2", "2.1 Massively Parallel Processing", "第二点一节 大规模并行处理") == "2.1 大规模并行处理"
    assert _heading_number_for("h2", "4.2 Parallel Query Optimization", "四点二 并行查询优化") == "4.2 并行查询优化"
    assert _heading_number_for("h2", "8. RELATED WORK", "八, 相关工作") == "8 相关工作"
    assert _heading_number_for("h2", "9. SUMMARY", "九, 总结") == "9 总结"
    assert _heading_number_for("h2", "10. REFERENCES", "十, 参考文献") == "10 参考文献"
    assert _heading_number_for("h2", "8.1 Query Optimization Foundations", "八点一 查询优化基础") == "8.1 查询优化基础"
    assert _heading_number_for("h2", "2.2 SQL on Hadoop", "第2.2节 Hadoop 上的 SQL") == "2.2 Hadoop 上的 SQL"


def test_punct_fragment_uses_line_end_not_union():
    """公式碎片落在整块并集内、但离上一行行尾超过 2pt 时不能捐出去遮罩。"""
    from app.formats.pdf import TextBlock, _donate_punct_rects

    line1 = pymupdf.Rect(72, 80, 400, 92)
    line2 = pymupdf.Rect(72, 96, 200, 108)
    prev = TextBlock(page=0, rect=line1 | line2, line_rects=[line1, line2],
                     text="scale the dot products by", size=11, color="#000", bold=False)
    far = pymupdf.Rect(210, 96, 240, 108)
    _donate_punct_rects(prev, [far])
    assert len(prev.line_rects) == 2
    close = pymupdf.Rect(201, 96, 210, 108)
    _donate_punct_rects(prev, [close])
    assert prev.line_rects[-1] == close


def test_cross_page_does_not_merge_numbered_heading():
    from app.formats.pdf import TextBlock, _cross_page_units

    def blk(text, page, size=10.0, y=0, bold=False):
        return TextBlock(page=page, rect=pymupdf.Rect(0, y, 200, y + 12), line_rects=[],
                         text=text, size=size, color="#000", bold=bold)

    blocks = [
        blk("The previous section ends without a period", 0, y=700),
        blk("1 Introduction", 1, y=60),
        blk("3.1 Encoder layers follow", 1, y=80),
    ]
    assert all(len(u) == 1 for u in _cross_page_units(blocks, 10.0))
    # 小写续段仍然合并
    cont = [
        blk("performs quite", 0, y=700),
        blk("well on this split.", 1, y=60),
    ]
    merged = [u for u in _cross_page_units(cont, 10.0) if len(u) == 2]
    assert len(merged) == 1


def test_split_translation_does_not_break_sentinel():
    from app.formats.pdf import _split_translation

    sentinel = "\x01i\x0212\x01/i\x02"
    text = "甲" * 8 + sentinel + "乙" * 8
    a, b = _split_translation(text, 0.5)
    assert a + b == text
    assert sentinel in a or sentinel in b
    assert "\x01" not in a.replace(sentinel, "")
    assert "\x01" not in b.replace(sentinel, "")


def test_cross_page_paragraph_continues_on_the_next_page(tmp_path):
    """跨页译文按满行排。这一页排满就接到下一页，行不画出页边，脚注留在原处。"""
    from app.formats.pdf import SPILL_TAIL, TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    first = doc.new_page(width=420, height=220)
    first.insert_text((40, 168), "performs quite", fontsize=10)
    first.insert_text((40, 200), "1 Note here.", fontsize=8)
    second = doc.new_page(width=420, height=220)
    second.insert_text((40, 48), "well. We hypothesize.", fontsize=10)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0, rect=pymupdf.Rect(40, 140, 300, 174), line_rects=[pymupdf.Rect(40, 156, 180, 174)],
        text="performs quite", size=10, color="#000000", bold=False,
    )
    note = TextBlock(
        page=0, rect=pymupdf.Rect(40, 188, 180, 206), line_rects=[pymupdf.Rect(40, 188, 140, 206)],
        text="1 Note here.", size=8, color="#000000", bold=False,
    )
    tail = TextBlock(
        page=1, rect=pymupdf.Rect(40, 36, 300, 80), line_rects=[pymupdf.Rect(40, 36, 220, 54)],
        text="well. We hypothesize.", size=10, color="#000000", bold=False,
    )
    paragraph = (
        "对于这个问题，模型的表现相当不错。我们推测这与数据集的性质有关，"
        "因此这一段按满行往下排，这一页排满再从下一页接着写，不要把行画出页边。"
        "后面这几行放不进页末的空隙，就从下一页的栏首继续，脚注仍留在原来的位置。"
    )
    out = _render_translated(
        src, [host, note, tail],
        [paragraph, "脚注仍在原处。", SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    page0 = out[0].get_text()
    page1 = out[1].get_text()
    assert "相当不错" in page0 + page1
    assert "well" not in page1
    assert out[0].rect.height == 220
    assert all(span["bbox"][3] <= 221 for block in out[0].get_text("dict")["blocks"]
               if block.get("type") == 0 for line in block["lines"] for span in line["spans"])
    note_top = None
    for block in out[0].get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
        if "脚注" in text:
            note_top = block["bbox"][1]
    assert note_top is not None and note_top > 180
    assert page1.strip()
    assert "相当不错" in page0
    assert "相当" not in page1 or "不错" in page1


def test_split_filled_lines_keeps_breaks_when_the_page_has_room():
    """这一页放得下时返回已经断好的行。返回原文会让调用方按一行去缩字号。"""
    from app.formats.pdf import _split_filled_lines

    text = "在实验研究中我们首先比较旧优化器与新优化器的计划质量并说明新优化器的收益"
    head, rest = _split_filled_lines(text, [], em=8.8, block_size=10, width=120, cjk=True, y0=100, limit=400)
    assert "\n" in head
    assert rest == ""


def test_hyphen_bridge_keeps_the_page_boundary():
    from app.formats.pdf import TextBlock, _prepare_unit

    def blk(text, page):
        return TextBlock(page=page, rect=pymupdf.Rect(40, 40, 200, 52), line_rects=[],
                         text=text, size=9, color="#000", bold=False)

    sent, formulas, name_i = _prepare_unit([blk("debug cus-", 0), blk("tomer issues later", 1)], 0)
    assert "customer" in sent
    assert "{|}" in sent
    assert "cus-" not in sent
    assert "issues later" in sent
    assert formulas == []
    assert name_i == 2


def test_fitted_cross_page_paragraph_keeps_body_size(tmp_path):
    """跨页译文这一页放得下时保持原字号，不再整段挤进一行高的框里。"""
    from app.formats.pdf import SPILL_TAIL, TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    first = doc.new_page(width=420, height=400)
    first.insert_text((40, 220), "In our experimental study we", fontsize=10)
    second = doc.new_page(width=420, height=400)
    second.insert_text((40, 48), "first compare the systems.", fontsize=10)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0, rect=pymupdf.Rect(40, 200, 280, 230),
        line_rects=[pymupdf.Rect(40, 208, 220, 222), pymupdf.Rect(40, 222, 180, 234)],
        text="In our experimental study we", size=10, color="#000000", bold=False,
    )
    tail = TextBlock(
        page=1, rect=pymupdf.Rect(40, 36, 280, 70),
        line_rects=[pymupdf.Rect(40, 36, 220, 50)],
        text="first compare the systems.", size=10, color="#000000", bold=False,
    )
    paragraph = (
        "在实验研究中我们首先比较旧优化器与新优化器的计划质量，"
        "并说明新优化器在大规模数据上的收益，这一段应该保持正文字号。"
    )
    out = _render_translated(
        src, [host, tail], [paragraph, SPILL_TAIL], "zh-CN",
        continuations={id(host): tail},
    )
    sizes = [
        span["size"]
        for block in out[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in span["text"])
    ]
    assert sizes
    assert min(sizes) >= 7


def test_narrow_fragment_is_written_at_the_continuation_width(tmp_path):
    """行尾半句不按自己的窄框排。译文写到续页，字号保持正文。"""
    from app.formats.pdf import SPILL_TAIL, TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=420, height=300)
    doc.new_page(width=420, height=300)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0, rect=pymupdf.Rect(250, 250, 290, 262),
        line_rects=[pymupdf.Rect(250, 250, 290, 262)],
        text="An opti-", size=10, color="#000000", bold=False,
    )
    tail = TextBlock(
        page=1, rect=pymupdf.Rect(40, 40, 280, 90),
        line_rects=[pymupdf.Rect(40, 40, 280, 54)],
        text="mization stage continues here.", size=10, color="#000000", bold=False,
    )
    paragraph = "优化阶段在满足下列任一条件时结束，资源受限的系统也可以在这里停下来。"
    out = _render_translated(
        src, [host, tail], [paragraph, SPILL_TAIL], "zh-CN",
        continuations={id(host): tail},
    )
    sizes = [
        span["size"]
        for page in out
        for block in page.get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in span["text"])
    ]
    assert sizes
    assert min(sizes) >= 7
    assert out[1].get_text().strip()


def test_shared_baseline_continuation_stays_in_its_lines(tmp_path):
    """共基线的续行只写在自己的两行里，不盖住右边的下一段，字号保持正文。"""
    from app.formats.pdf import SPILL_TAIL, TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((140, 126), "For example the next sentence.", fontsize=9)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0, rect=pymupdf.Rect(160, 100, 300, 112),
        line_rects=[pymupdf.Rect(160, 100, 300, 112)],
        text="Statistics promise computation is", size=9, color="#000000", bold=False,
    )
    tail = TextBlock(
        page=0, rect=pymupdf.Rect(40, 114, 130, 126),
        line_rects=[pymupdf.Rect(40, 114, 130, 126)],
        text="expression-specific.", size=9, color="#000000", bold=False,
    )
    neighbor_lines = [pymupdf.Rect(140, 114, 300, 126), pymupdf.Rect(40, 128, 300, 140)]
    neighbor = TextBlock(
        page=0, rect=neighbor_lines[0] | neighbor_lines[1], line_rects=neighbor_lines,
        text="For example the next sentence starts here.", size=9, color="#000000", bold=False,
    )
    paragraph = "统计承诺所描述的计算过程只对特定表达式成立。"
    out = _render_translated(
        src, [host, neighbor, tail],
        [paragraph, neighbor.text, SPILL_TAIL],
        "zh-CN",
        continuations={id(host): tail},
    )
    spans = [
        span
        for block in out[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if any("\u4e00" <= ch <= "\u9fff" for ch in span["text"])
    ]
    assert spans
    assert min(span["size"] for span in spans) >= 7
    assert any(span["bbox"][1] < 113 for span in spans)
    assert any(span["bbox"][1] >= 113 for span in spans)
    for span in spans:
        x0, y0, x1, _y1 = span["bbox"]
        if y0 >= 113:
            assert x1 <= 136
        else:
            assert x0 >= 155
    assert "For example" in out[0].get_text()


def test_cross_page_formulas_keep_own_page_and_name():
    from app.formats.pdf import TextBlock, _assign_restored_parts, _prepare_unit

    def span(text, font, size, x0, y0):
        return {"text": text, "font": font, "size": size, "bbox": (x0, y0, x0 + 12, y0 + 10),
                "origin": (x0, y0 + 8)}

    def block(page, prose):
        spans = [span(prose, "TimesNewRomanPSMT", 12, 0, 0),
                 span("E", "CMMI10", 12, 40, 0)]
        b = _block_with_spans([spans])
        b.page = page
        b.text = prose + "E"
        return b

    unit = [block(0, "sequence "), block(1, "length ")]
    sent, formulas, name_i = _prepare_unit(unit, 0)
    assert name_i == 2
    assert {f["name"] for f in formulas} == {"f0_1.png", "f1_1.png"}
    assert [f["page"] for f in formulas] == [0, 1]
    assert formulas[0]["owner"] == 0 and formulas[1]["owner"] == 1
    assert "{v1}" in sent and "{v2}" in sent
    restored = sent.replace("{v1}", "\x01i\x021\x01/i\x02").replace("{v2}", "\x01i\x022\x01/i\x02")
    parts, lists = _assign_restored_parts(restored, unit, formulas)
    from app.formats.pdf import SPILL_TAIL
    assert parts[1] == SPILL_TAIL and lists[1] == []
    assert [f["page"] for f in lists[0]] == [0, 1]
    assert parts[0].count("\x01i\x021\x01/i\x02") == 1
    assert parts[0].count("\x01i\x022\x01/i\x02") == 1


def test_boundary_echo_keeps_repeated_cells_and_short_phrases():
    from app.formats.pdf import _strip_boundary_echo

    assert _strip_boundary_echo("WSJ only", "WSJ only", "仅 WSJ", "仅 WSJ") == "仅 WSJ"
    assert _strip_boundary_echo("see below", "next", "如下所示", "如下所示的结果") == "如下所示的结果"
    prev = "前文结束序列），而目标输出"
    cur = "序列），而目标输出才是下一段"
    assert _strip_boundary_echo("alpha beta", "gamma delta", prev, cur) == "才是下一段"


def test_tighten_line_height_for_cjk():
    from app.formats.pdf import _tighten_line_height

    cjk = "* {line-height: 1.45; margin: 0;}"
    latin = "* {line-height: 1.2; margin: 0;}"
    assert "line-height: 1.2" in _tighten_line_height(cjk)
    assert "line-height: 1.1" in _tighten_line_height(latin)
    assert _tighten_line_height(cjk) != cjk


def test_preview_override_replaces_shared_translation():
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache())
    runner.segments = [{"s": "2 Proposed", "k": "h2"}]
    runner._indices = {"2 Proposed": [0]}
    runner._done = {"2 Proposed": "第二章 所提"}
    runner._log = ["2 Proposed"]
    runner._index_overrides[0] = "2 所提"
    assert runner.preview()["updates"] == [[0, "2 所提", 0]]


def test_split_translation_balances_style_marks():
    """跨页切分时样式标记要配对：左半补闭、右半补开，且重开的是仍开着的字号。"""
    from app.formats.pdf import _split_translation
    from app.formats.pdf_flow import strip_style_marks

    t = "前面一段结束。{b}后半加粗的内容比较长，跨页了{/b}。"
    left, right = _split_translation(t, 0.55)
    assert "{b}" in left and left.endswith("{/b}")
    assert right.startswith("{b}")
    assert left.count("{b}") == left.count("{/b}")
    assert right.count("{b}") == right.count("{/b}")
    assert strip_style_marks(left) + strip_style_marks(right) == strip_style_marks(t)

    sized = "{z80}小号结束。{/z}{z120}大号内容比较长，会跨页继续写下去直到句号。{/z}"
    left, right = _split_translation(sized, 0.45)
    assert left.endswith("{/z}")
    assert right.startswith("{z120}")
    assert "{z80}" not in right
    assert strip_style_marks(left) + strip_style_marks(right) == strip_style_marks(sized)

    nested = "{z120}{b}{i}强调的内容比较长，会跨页继续写下去直到句号。{/i}{/b}{/z}"
    left, right = _split_translation(nested, 0.4)
    assert left.endswith("{/i}{/b}{/z}")
    assert right.startswith("{z120}{b}{i}")
    assert strip_style_marks(left) + strip_style_marks(right) == strip_style_marks(nested)


def test_save_after_redact_keeps_resources_indirect(tmp_path):
    """redact 把 /Resources 改成内联字典。写回前必须改成间接引用，否则大字典在 garbage 保存时会被写坏。"""
    import pymupdf as pm
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "t.pdf"
    doc = pm.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((40, 60), "First block here")
    for i in range(8):
        page.insert_text((40, 80 + i * 12), f"Body line number {i} with enough text to matter")
    doc.save(src)
    doc.close()

    lines = ["First block here"] + [
        f"Body line number {n} with enough text to matter" for n in range(8)
    ]
    blocks = []
    for index, line in enumerate(lines):
        y = 48 + index * 12
        rect = pm.Rect(40, y, 360, y + 12)
        blocks.append(TextBlock(
            page=0, rect=rect, line_rects=[rect], text=line, size=11, color="#000000", bold=False,
        ))
    trans = [f"译文第{i}段的内容写在这里测试保存" for i in range(len(blocks))]
    out = _render_translated(src, blocks, trans, "zh-CN")
    kind, value = out.xref_get_key(out[0].xref, "Resources")
    assert kind == "xref" and value.endswith(" 0 R")
    data = out.tobytes(garbage=4, deflate=True)
    check = pm.open("pdf", data)
    text = check[0].get_text()
    assert "译文第0段" in text and f"译文第{len(blocks) - 1}段" in text


def test_wrapped_formula_tail_stays_one_placeholder_without_a_wide_box():
    """跨可视行折行的公式收成一个占位符的两片。不并 bbox，否则裁进两行之间的整栏空白。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None):
        return {"text": text, "font": font, "size": size,
                "bbox": (x0, y0, x1 if x1 is not None else x0 + 10,
                         y1 if y1 is not None else y0 + 10),
                "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2)}

    b = _block_with_spans([
        [span("taken as ", "TimesNewRomanPSMT", 10, 100, 100),
         span("y", "CMMI10", 10, 400, 100, x1=405, y1=110),
         span("t−1 =", "CMR7", 7, 405, 104, x1=415, y1=110)],
        [span("s|h", "CMMI10", 10, 108, 114, x1=120, y1=124),
         span("t−1", "CMR7", 7, 120, 117, x1=128, y1=124),
         span("). This", "TimesNewRomanPSMT", 10, 130, 114)],
    ], size=10.0)
    sent, formulas = _placeholderize(b, 0)
    assert len(formulas) == 1
    assert sent.count("{v") == 1
    parts = formulas[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["bbox"].x0 >= 400
    assert parts[1]["bbox"].x0 <= 108
    assert parts[0]["bbox"] | parts[1]["bbox"] != formulas[0]["bbox"]


def test_justify_extras_skip_punctuation_and_short_lines():
    """两端对齐摊在字与字之间，标点前面不拉开。余量太小或缝太少时不拉。"""
    from app.formats.pdf_align import justify_extras

    text = "我们在此提出一种，采样机制已经写完"
    extras = justify_extras(text, 24.0, 10.0)
    assert len(extras) == len(text) - 1
    comma = text.index("，")
    assert extras[comma - 1] == 0.0
    assert extras[0] > 0
    assert justify_extras(text, 1.0, 10.0) == []
    assert justify_extras("短行", 20.0, 10.0) == []
    names = "Zhongxian Gu, Entong Shen, George Caragea 等人继续写"
    extras = justify_extras(names, 40.0, 10.0)
    assert extras[0] == 0.0
    assert extras[names.index(" ")] > 0


def test_caption_word_in_bold_cm_is_not_a_formula():
    from app.formats.pdf import _is_formula_span

    assert _is_formula_span({"font": "CMBX8", "text": "of", "size": 8.0}, 9.0) is False
    assert _is_formula_span({"font": "CMBX10", "text": "x", "size": 10.0}, 10.0) is True


def test_garbled_figure_label_is_recognized():
    from app.formats.pdf import _garbled_label

    assert _garbled_label("(c)$Bo8om(up$sta.s.cs$deriva.on$")
    assert not _garbled_label("Figure 2: Interaction of Orca with database system")


def test_right_fragment_stays_on_its_own_line():
    """侧标让出的第一行仍从这一行起写，左缘用这一行自己的，不整段下移。"""
    from app.formats.pdf import TextBlock, _flow_y0, _inset_write_boxes

    fragment = pymupdf.Rect(190, 40, 360, 52)
    nxt = pymupdf.Rect(40, 56, 360, 68)
    block = TextBlock(
        page=0, rect=fragment | nxt, line_rects=[fragment, nxt],
        text="At the end of exploration.", size=9, color="#000", bold=False,
    )
    assert _flow_y0(block, True) == pytest.approx(40 + 0.9, abs=0.01)
    first, rest = _inset_write_boxes(block, True)
    assert first.x0 == pytest.approx(190)
    assert rest.x0 == pytest.approx(40)
    assert rest.y0 > first.y0
    indent = pymupdf.Rect(52, 40, 360, 52)
    body = pymupdf.Rect(40, 56, 360, 68)
    indented = TextBlock(
        page=0, rect=indent | body, line_rects=[indent, body],
        text="A normal indented paragraph.", size=9, color="#000", bold=False,
    )
    assert _flow_y0(indented, False) == 40
    assert _inset_write_boxes(indented, False) is None
    # 同一基线上的碎片不是下一行。续行从真正的下一行起，不压回第一行。
    fragments = [
        pymupdf.Rect(127.9, 595.2, 134.1, 604.2),
        pymupdf.Rect(141.4, 595.2, 198.1, 604.2),
        pymupdf.Rect(205.5, 595.2, 292.9, 604.2),
        pymupdf.Rect(53.8, 605.7, 292.9, 614.7),
    ]
    torn = TextBlock(
        page=0, rect=fragments[0] | fragments[-1], line_rects=fragments,
        text="is an extensible optimizer framework.", size=9, color="#000", bold=False,
    )
    torn_first, torn_rest = _inset_write_boxes(torn, True)
    assert torn_first.x0 == pytest.approx(127.9)
    assert torn_rest.x0 == pytest.approx(53.8)
    assert torn_rest.y0 > torn_first.y0 + 4


def test_affiliation_tails_split_and_stack():
    from app.formats.pdf import TextBlock, _affiliation_write_rects, _split_affiliation_tails

    names = [
        pymupdf.Rect(100, 120, 400, 132),
        pymupdf.Rect(100, 134, 400, 146),
        pymupdf.Rect(116, 160, 180, 172),
    ]
    author = TextBlock(
        page=0, rect=names[0] | names[2], line_rects=names,
        text="Zhongxian Gu Entong Shen ∗Pivotal Inc.",
        size=12, color="#000", bold=False,
        span_lines=[[{"text": "Zhongxian Gu"}], [{"text": "Entong Shen"}], [{"text": "∗Pivotal Inc."}]],
    )
    datometry = TextBlock(
        page=0, rect=pymupdf.Rect(210, 160, 300, 184),
        line_rects=[pymupdf.Rect(210, 160, 300, 172), pymupdf.Rect(210, 172, 300, 184)],
        text="‡ Datometry Inc. San Francisco", size=10, color="#000", bold=False,
    )
    split = _split_affiliation_tails([author, datometry])
    assert len(split) == 3
    assert "Pivotal" not in split[0].text
    assert "Pivotal" in split[1].text
    stacked = _affiliation_write_rects(split)
    assert stacked[id(split[1])].y0 != stacked[id(split[2])].y0


def test_side_label_stays_beside_the_first_line(tmp_path):
    """短标题留在正文第一行左边。正文从标题右侧起写，下一行再回到栏左缘。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=300).insert_text((40, 80), "heading and sentence", fontsize=11)
    doc.save(src)
    doc.close()
    heading_line = pymupdf.Rect(40, 40, 180, 52)
    fragment = pymupdf.Rect(190, 40, 360, 52)
    nxt = pymupdf.Rect(40, 56, 360, 68)
    heading = TextBlock(
        page=0, rect=heading_line, line_rects=[heading_line],
        text="(2) Statistics Derivation.", size=9, color="#000000", bold=True,
        span_lines=[[{"text": "(2) Statistics Derivation.", "font": "CMBX9", "flags": 16}]],
    )
    body = TextBlock(
        page=0, rect=fragment | nxt, line_rects=[fragment, nxt],
        text="At the end of exploration the memo keeps the space.",
        size=9, color="#000000", bold=False,
    )
    out = _render_translated(
        src, [body, heading],
        ["探索结束时备忘录保存完整的逻辑空间并且继续写下一段说明，后面这几个字回到栏左缘。", "(2) 统计信息推导。"],
        "zh-CN",
    )
    chars = _cjk_chars(out[0])
    heading_chars = [item for item in chars if item[3] < 54 and item[0] in "统计信息推导"]
    first_body = [item for item in chars if item[0] == "探"]
    assert heading_chars and first_body
    heading_right = max(item[2] for item in heading_chars)
    body_left = min(item[1] for item in first_body)
    assert heading_right <= 188
    # 英文正文从 190 起。中文标签更短，第一行要补上让出来的空白，并且不压到标签。
    assert body_left >= heading_right - 0.6
    assert body_left < 170
    assert abs(first_body[0][3] - heading_chars[0][3]) < 4
    later_body = [item for item in chars if item[3] > first_body[0][3] + 4]
    assert later_body
    assert min(item[1] for item in later_body) < 80


def test_inset_rest_reflows_at_the_column_width():
    """侧标让出的第一行按窄宽断完后，剩下的按整栏宽重排，不把窄行的换行留住。"""
    from app.formats.pdf import _split_own_lines

    text = (
        "Orca 包含一个可扩展框架, 用于基于形式化的属性规范描述查询需求和计划特征。"
        "属性有不同类型, 包括逻辑属性。"
    )
    head, rest = _split_own_lines(text, [], 7.92, 9.0, 120.0, 240.0)
    rest_lines = [line for line in rest.split("\n") if line]
    assert head and rest_lines
    assert max(len(line) for line in rest_lines) > len(head)
    assert "框架用于" not in head + rest
    glued = (head + rest.replace("\n", "")).replace(" ", "")
    assert glued == text.replace(" ", "")


def test_inset_continuation_fills_the_column(tmp_path):
    """侧标旁边的正文，续行回到栏左缘并写到栏右缘，字号不再因为窄行太多而被压小。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=400).insert_text((40, 80), "heading and sentence", fontsize=11)
    doc.save(src)
    doc.close()
    heading_line = pymupdf.Rect(40, 40, 180, 52)
    first = pymupdf.Rect(190, 40, 360, 52)
    later = [pymupdf.Rect(40, 56 + i * 12, 360, 68 + i * 12) for i in range(8)]
    heading = TextBlock(
        page=0, rect=heading_line, line_rects=[heading_line],
        text="(2) Statistics Derivation.", size=9, color="#000000", bold=True,
        span_lines=[[{"text": "(2) Statistics Derivation.", "font": "CMBX9", "flags": 16}]],
    )
    body = TextBlock(
        page=0, rect=first | later[-1], line_rects=[first, *later],
        text="At the end of exploration the memo keeps the space and continues.",
        size=9, color="#000000", bold=False,
    )
    text = (
        "探索结束时备忘录保存完整的逻辑空间并且继续写下一段很长的说明"
        "用来占满这一栏的宽度然后再换行继续写到栏的右缘这里应该用整栏宽度"
        "而不是停在第一行那么窄的位置上。"
    )
    out = _render_translated(src, [body, heading], [text, "(2) 统计信息推导。"], "zh-CN")
    chars = _cjk_chars(out[0])
    grouped = _group_chars(chars)
    body_lines = [line for line in grouped if line[0][1] >= 100 or line[0][3] > 58]
    assert len(body_lines) >= 3
    assert 80 < body_lines[0][0][1] < 170
    full = [line for line in body_lines[1:-1] if line[-1][2] > 300]
    assert full, [(line[0][1], line[-1][2], "".join(item[0] for item in line)) for line in body_lines]
    sizes = [
        span["size"]
        for block in out[0].get_text("dict")["blocks"] if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if span["bbox"][1] > 58 and span["bbox"][0] < 80 and span["text"].strip()
    ]
    assert sizes
    assert min(sizes) > 7.5


def test_same_baseline_fragments_do_not_cover_the_lead(tmp_path):
    """同一基线的碎片不是续行。续文从下一行起写，不压到左侧未译的 Cascades 上。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((62, 50), "Cascades", fontsize=9)
    doc.save(src)
    doc.close()
    lead = pymupdf.Rect(62.8, 40, 99, 52)
    fragments = [
        pymupdf.Rect(127.9, 40, 160, 52),
        pymupdf.Rect(168, 40, 220, 52),
        pymupdf.Rect(40, 56, 280, 68),
        pymupdf.Rect(40, 70, 280, 82),
    ]
    cascades = TextBlock(
        page=0, rect=lead, line_rects=[lead], text="Cascades",
        size=9, color="#000000", bold=False,
    )
    body = TextBlock(
        page=0, rect=fragments[0] | fragments[-1], line_rects=fragments,
        text="is an extensible optimizer framework whose principles have been used.",
        size=9, color="#000000", bold=False,
    )
    text = "是一个可扩展的优化器框架，其原则已被用于构建查询优化器，并且继续写到下一行。"
    out = _render_translated(src, [cascades, body], ["Cascades", text], "zh-CN")
    chars = _cjk_chars(out[0])
    first = [item for item in chars if item[0] == "是"]
    assert first
    same_line = [item for item in chars if abs(item[3] - first[0][3]) < 2]
    below = [item for item in chars if item[3] > first[0][3] + 4]
    assert same_line and below
    assert min(item[1] for item in same_line) >= 120
    assert "Cascades" in out[0].get_text()


def test_cross_page_stops_above_the_page_number(tmp_path):
    """跨页译文停在页码带上面。页码是纯数字，不在块列表里，也不能写穿它。"""
    from app.formats.pdf import SPILL_TAIL, TextBlock, _render_translated, _room_below
    from app.formats.pdf_layout import content_floor

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=612, height=792)
    doc.save(src)
    doc.close()
    host = TextBlock(
        page=0, rect=pymupdf.Rect(316.8, 699.8, 556, 719.3),
        line_rects=[
            pymupdf.Rect(325.8, 699.8, 556, 708.8),
            pymupdf.Rect(316.8, 710.3, 556, 719.3),
        ],
        text="Several efforts have addressed interactive processing on Hadoop by creating specialized query engines that allow",
        size=9, color="#000000", bold=False,
    )
    tail = TextBlock(
        page=1, rect=pymupdf.Rect(53.8, 56.8, 293, 180),
        line_rects=[pymupdf.Rect(53.8, 56.8, 293, 66)],
        text="SQL-based processing of data in HDFS.",
        size=9, color="#000000", bold=False,
    )
    paragraph = (
        "一些研究通过创建专用查询引擎来解决交互式处理问题，使得无需使用 MapReduce 即可基于 SQL 处理数据。"
        "这些方法在查询优化器和执行引擎的设计上各不相同。数据库与大数据平台的共置使数据能够以原生方式处理。"
        "后面这几行放不进页末，就从下一页接着写，不能压到页码上。"
    )
    out = _render_translated(
        src, [host, tail], [paragraph, SPILL_TAIL], "zh-CN",
        continuations={id(host): tail},
    )
    floor = content_floor(792)
    bottoms = [
        ch["bbox"][3]
        for block in out[0].get_text("rawdict")["blocks"] if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    assert bottoms
    assert max(bottoms) < floor + 4
    assert max(bottoms) < 745
    assert out[1].get_text().strip()
    note = TextBlock(
        page=0, rect=pymupdf.Rect(53, 750, 200, 762),
        line_rects=[pymupdf.Rect(53, 750, 200, 762)],
        text="1 Note.", size=8, color="#000", bold=False,
    )
    kept = _room_below(out[0], note, 750, 53, 200, [note])
    assert kept > 780


def test_page_margins_stay_with_the_source(tmp_path):
    """页首对齐原文上沿，正文不写进原文下沿以下。页脚带里的脚注仍不额外下刀。"""
    from app.formats.pdf import TextBlock, _render_translated, _room_below

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.save(src)
    doc.close()
    top = pymupdf.Rect(53.8, 56.8, 293.0, 90.0)
    bottom = pymupdf.Rect(53.8, 690.0, 293.0, 719.3)
    blocks = [
        TextBlock(page=0, rect=top, line_rects=[top], text="During query execution, data can be distributed.",
                  size=9, color="#000000", bold=False),
        TextBlock(page=0, rect=bottom, line_rects=[bottom],
                  text="DXL. Decoupling the optimizer from the database system requires a communication mechanism.",
                  size=9, color="#000000", bold=False),
    ]
    long = "在查询执行期间，数据可以通过多种方式分布到各个分段节点，包括广播、哈希和复制。" * 6
    out = _render_translated(src, blocks, ["查询执行期间数据会分布到分段节点。", long], "zh-CN")
    chars = [
        ch for block in out[0].get_text("rawdict")["blocks"] if block.get("type") == 0
        for line in block["lines"] for span in line["spans"] for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    assert chars
    assert min(ch["bbox"][1] for ch in chars) <= 57.6
    assert max(ch["bbox"][3] for ch in chars) <= 719.5
    note = TextBlock(
        page=0, rect=pymupdf.Rect(53, 750, 200, 762),
        line_rects=[pymupdf.Rect(53, 750, 200, 762)],
        text="1 Note.", size=8, color="#000", bold=False,
    )
    assert _room_below(out[0], note, 750, 53, 200, [note]) > 780


def test_cjk_lines_share_the_right_edge(tmp_path):
    """非末行的右缘对齐到写入框。标点贴着前一个字。末行保持左齐。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=240).insert_text((40, 60), "source paragraph", fontsize=11)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 360, 140)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect],
        text="source paragraph that differs", size=12, color="#111111", bold=False,
    )
    text = "我们在此提出一种采样机制，在训练过程中随机决定使用真实的前一个词元，还是来自模型本身的估计值。"
    out = _render_translated(src, [block], [text], "zh-CN")
    chars = _cjk_chars(out[0])
    grouped = _group_chars(chars)
    assert len(grouped) >= 2
    right = rect.x1 + 2
    assert abs(grouped[0][-1][2] - right) < 1.5
    assert grouped[-1][-1][2] < right - 8
    for line in grouped[:-1]:
        for prev, nxt in zip(line, line[1:]):
            if nxt[0] == "，":
                assert nxt[1] - prev[2] < 0.8
    assert rect.y0 - 2 < grouped[0][0][3] < rect.y1


def test_em_dash_line_stays_inside_the_box(tmp_path):
    """破折号和 HMM 按实际字宽断行，不能再多出一个字到右缘外面。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=240).insert_text((40, 60), "source paragraph", fontsize=11)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 280, 160)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect],
        text="source paragraph that differs", size=12, color="#111111", bold=False,
    )
    text = "标准配置——每一帧都经过 HMM 对齐，破折号不能再多挤出一个字到栏外去。"
    out = _render_translated(src, [block], [text], "zh-CN")
    right = rect.x1 + 2
    for item in out[0].get_text("dict")["blocks"]:
        if item.get("type") != 0:
            continue
        for line in item["lines"]:
            for span in line["spans"]:
                if span["bbox"][0] < 20:
                    continue
                assert span["bbox"][2] <= right + 1.0, span["text"]


def test_italic_marks_still_justify_the_line(tmp_path):
    """斜体标记不再改走 HTML。汉字没有斜体字形，行仍两端对齐。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=500, height=240).insert_text((40, 60), "source paragraph", fontsize=11)
    doc.save(src)
    doc.close()
    rect = pymupdf.Rect(40, 40, 360, 140)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect],
        text="source paragraph that differs", size=12, color="#111111", bold=False,
    )
    text = "我们在此提出一种{i}均匀计划采样{/i}机制，在训练过程中随机决定使用真实的前一个词元，还是来自模型本身的估计值。"
    out = _render_translated(src, [block], [text], "zh-CN")
    grouped = _group_chars(_cjk_chars(out[0]))
    assert len(grouped) >= 2
    right = rect.x1 + 2
    assert abs(grouped[0][-1][2] - right) < 1.5
    assert grouped[-1][-1][2] < right - 8


def test_formula_does_not_jump_to_the_line_above(tmp_path):
    """上一行伸到栏边时，下一行的公式仍贴着本行文字。
    上一行的字框会探进本行的写入盒子，不能拿它的右缘当光标。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=240)
    page.insert_text((80, 80), "t", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    box = d[0].search_for("t")[0]
    d.close()
    formula = {
        "bbox": box, "name": "f0_1.png", "w": round(box.width, 1), "h": round(box.height, 1),
        "d": 2.0, "raise": 0.0, "text": "t", "has_img": True, "page": 0, "mid": 0,
        "png": pymupdf.open(src)[0].get_pixmap(clip=box, dpi=72).tobytes("png"),
        "spans": [{"bbox": tuple(box), "text": "t", "size": 10, "origin": (box.x0, box.y1 - 2)}],
    }
    rect = pymupdf.Rect(40, 40, 360, 160)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect],
        text="source paragraph", size=11, color="#000", bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    # 零宽空格让字形写入失败，改走会测量右缘的 HTML 盒子。
    text = (
        "我们在此提出一种采样机制，在训练过程中随机决定使用真实的前一个词元还是估计值。"
        f"在时间{sentinel}, 模型需要上一个标记。\u200b"
    )
    out = _render_translated(src, [block], [text], "zh-CN", {id(block): [formula]})
    page = out[0]
    time = page.search_for("在时间")
    painted = page.search_for("t")
    assert time and painted
    assert painted[0].x0 - time[0].x1 < 8
    assert painted[0].x0 < rect.x1 - 40


def test_italic_run_does_not_open_a_wide_gap_before_the_formula(tmp_path):
    """i-th 两侧各有一个空格。斜体按 0.5em 估宽会把后面的公式推开，应贴着写出的字。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((80, 80), "y=", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    y_box = d[0].search_for("y=")[0]
    d.close()
    formula = {
        "bbox": y_box, "name": "f0_1.png", "w": round(y_box.width, 1), "h": round(y_box.height, 1),
        "d": 2.0, "raise": 0.0, "text": "y=", "has_img": True, "page": 0, "mid": 0,
        "png": pymupdf.open(src)[0].get_pixmap(clip=y_box, dpi=72).tobytes("png"),
        "spans": [{"bbox": tuple(y_box), "text": "y=", "size": 10, "origin": (y_box.x0, y_box.y1 - 2)}],
    }
    rect = pymupdf.Rect(40, 40, 460, 120)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect],
        text="the i-th token", size=11, color="#000", bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    out = _render_translated(
        src, [block], [f"第{{i}} i-th{{/i}}个词元{sentinel}。"], "zh-CN", {id(block): [formula]},
    )
    page = out[0]
    spans = [
        span
        for blk in page.get_text("dict")["blocks"] if blk.get("type") == 0
        for ln in blk["lines"] for span in ln["spans"]
    ]
    chars = [
        ch
        for blk in page.get_text("rawdict")["blocks"] if blk.get("type") == 0
        for ln in blk["lines"] for span in ln["spans"] for ch in span["chars"]
    ]
    di = next(ch for ch in chars if ch["c"] == "第")
    ith = next(ch for ch in chars if ch["c"] == "i" and ch["bbox"][0] > di["bbox"][2])
    aitch = next(ch for ch in chars if ch["c"] == "h" and ch["bbox"][0] > ith["bbox"][0])
    ge = next(ch for ch in chars if ch["c"] == "个" and ch["bbox"][0] > aitch["bbox"][0])
    assert 1.2 <= ith["bbox"][0] - di["bbox"][2] <= 5
    assert 1.2 <= ge["bbox"][0] - aitch["bbox"][2] <= 5
    italic = next(span for span in spans if "i-th" in span["text"])
    assert "Italic" in italic["font"] or "Oblique" in italic["font"] or int(italic.get("flags") or 0) & 2
    yuan = next(span for span in spans if "元" in span["text"])
    painted = page.search_for("y=")
    assert painted
    assert painted[0].x0 - yuan["bbox"][2] < 5


def test_formula_keeps_a_space_before_following_text(tmp_path):
    """公式后面接文字时，中间有一个空格宽的分界，不贴在一起。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)
    page.insert_text((80, 80), "y=", fontsize=11)
    doc.save(src)
    doc.close()
    d = pymupdf.open(src)
    y_box = d[0].search_for("y=")[0]
    d.close()
    formula = {
        "bbox": y_box, "name": "f0_1.png", "w": round(y_box.width, 1), "h": round(y_box.height, 1),
        "d": 2.0, "raise": 0.0, "text": "y=", "has_img": True, "page": 0, "mid": 0,
        "png": pymupdf.open(src)[0].get_pixmap(clip=y_box, dpi=72).tobytes("png"),
        "spans": [{"bbox": tuple(y_box), "text": "y=", "size": 10, "origin": (y_box.x0, y_box.y1 - 2)}],
    }
    rect = pymupdf.Rect(40, 50, 420, 110)
    block = TextBlock(
        page=0, rect=rect, line_rects=[rect, pymupdf.Rect(y_box)],
        text="take the value", size=11, color="#000", bold=False,
    )
    sentinel = "\x01i\x021\x01/i\x02"
    out = _render_translated(
        src, [block], [f"前面{sentinel}后面还有一段说明文字。"], "zh-CN", {id(block): [formula]},
    )
    page = out[0]
    y_hit = page.search_for("y=")
    after = page.search_for("后")
    assert y_hit and after
    gap = after[0].x0 - y_hit[0].x1
    assert gap >= 2.5
    stuck = page.search_for(")。")
    # 标点仍然贴着公式，不在 ) 前加空格
    close = _render_translated(
        src, [block], [f"取{sentinel})。"], "zh-CN", {id(block): [formula]},
    )
    paren = close[0].search_for(")")
    painted = close[0].search_for("y=")
    if paren and painted:
        assert paren[0].x0 - painted[0].x1 < 4


def _cjk_chars(page: pymupdf.Page) -> list[tuple[str, float, float, float]]:
    chars: list[tuple[str, float, float, float]] = []
    for blk in page.get_text("rawdict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for span in ln["spans"]:
                for ch in span["chars"]:
                    if "\u4e00" <= ch["c"] <= "\u9fff" or ch["c"] in "，。":
                        chars.append((ch["c"], ch["bbox"][0], ch["bbox"][2], ch["origin"][1]))
    return chars


def _group_chars(
    chars: list[tuple[str, float, float, float]],
) -> list[list[tuple[str, float, float, float]]]:
    grouped: list[list[tuple[str, float, float, float]]] = []
    for item in chars:
        if not grouped or abs(item[3] - grouped[-1][0][3]) > 2:
            grouped.append([item])
        else:
            grouped[-1].append(item)
    return grouped
