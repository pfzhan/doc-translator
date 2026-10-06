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
        blk("4https://example.com/footnote", 0, size=7.0, y=760),  # 页脚脚注，不参与
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
    assert len(lists[0]) == 1 and lists[0][0]["page"] == 0
    assert len(lists[1]) == 1 and lists[1][0]["page"] == 1
    assert parts[0].count("\x01i\x021\x01/i\x02") == 1
    assert parts[1].count("\x01i\x021\x01/i\x02") == 1
    assert "\x01i\x022\x01/i\x02" not in parts[0] and "\x01i\x022\x01/i\x02" not in parts[1]


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
