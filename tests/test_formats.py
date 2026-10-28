import asyncio
import re
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
    out = render(blocks, ["T:" + t for t in texts], bilingual=False)
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
    out = render(blocks, ["T:" + t for t in texts], bilingual=True)
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


def test_epub2_outline_reads_the_ncx(tmp_path):
    """没有 nav 文档的 EPUB2 按 toc.ncx 列目录，层级跟着 navPoint 嵌套。"""
    from app.formats.epub import book_outline

    ncx = (
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>'
        '<navPoint id="a"><navLabel><text>Chapter One</text></navLabel><content src="ch1.xhtml"/>'
        '<navPoint id="b"><navLabel><text>A Section</text></navLabel><content src="ch1.xhtml#s1"/></navPoint>'
        "</navPoint></navMap></ncx>"
    )
    src = tmp_path / "old.epub"
    _write_epub(
        src,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<itemref idref="ch1"/>',
        {"ch1.xhtml": "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>x</p></body></html>", "toc.ncx": ncx},
    )
    with zipfile.ZipFile(src) as z:
        outline = book_outline(z)
    assert outline["title"] == "Test Book" and outline["spine"] == ["OEBPS/ch1.xhtml"]
    assert [(e["label"], e["anchor"], e["depth"]) for e in outline["toc"]] == [
        ("Chapter One", "", 0), ("A Section", "s1", 1),
    ]


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



def test_epub_named_entities_survive_xml_parsing(tmp_path):
    """XHTML 1.1 章节里的 &rsquo;、&nbsp;、&mdash; 不能被 lxml-xml 删掉，原文和译文都要保留这些字符。"""
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>c1</title></head>'
        "<body><p>Don&rsquo;t stop&nbsp;now &mdash; ok &amp; done</p></body></html>"
    )
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": chapter})
    [out] = run(translate_epub(src, tmp_path, runner(), True, "zh-CN"))
    text = _epub_texts(out)
    assert "Don\u2019t stop\u00a0now \u2014 ok & done" in text
    assert "[zh-CN] Don\u2019t stop" in text

def test_epub_html_style_void_tags_and_bare_ampersands_keep_text(tmp_path):
    """EPUB2 里常见 <br>、<img ...> 不闭合和裸 &。lxml-xml 容错解析会从出错处起删掉 &amp;、&lt;，
    段落还会互相嵌套；这里先补成良构 XML 再解析，SVG 的 viewBox 这种大小写敏感的属性也不受影响。"""
    from lxml import etree

    body = ('<p>First line<br>second &amp; line &lt;tag&gt;</p>'
            '<p>AT&T bought <img src="a.png" alt="logo"> it.</p>'
            '<p>Third paragraph.</p>'
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>')
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(body)})
    [out] = run(translate_epub(src, tmp_path, runner(), True, "zh-CN"))
    data = zipfile.ZipFile(out).read("OEBPS/ch1.xhtml")
    root = etree.fromstring(data)
    texts = ["".join(p.itertext()) for p in root.iter("{http://www.w3.org/1999/xhtml}p")]
    assert texts[:2] == ["First linesecond & line <tag>", "[zh-CN] First line\nsecond & line <tag>".replace("\n", "")]
    assert "AT&T bought  it." in texts and "Third paragraph." in texts
    assert not any(p.find("{http://www.w3.org/1999/xhtml}p") is not None for p in root.iter("{http://www.w3.org/1999/xhtml}p"))
    assert b'viewBox="0 0 10 10"' in data


def test_mobi7_cp1252_decoding(tmp_path):
    """老 MOBI7 的 book.html 按 cp1252 解码：é 等字符不能变成 U+FFFD。"""
    html_file = tmp_path / "book.html"
    html_file.write_bytes("<html><body><p>café et naïve</p></body></html>".encode("cp1252"))
    text = _read_mobi7_html(html_file)
    assert "café et naïve" in text
    assert "\ufffd" not in text
    # 转成 XHTML 后字符也保留
    assert "café et naïve" in _xhtml_from_mobi7(text).decode("utf-8")


def test_mobi_temp_dir_removed_when_unpack_fails(tmp_path, monkeypatch):
    """解包失败（DRM、坏文件）时临时目录也要删掉。"""
    from app.formats import mobi as mobi_format

    work = tmp_path / "mobiex-fail"
    monkeypatch.setattr(mobi_format.tempfile, "mkdtemp", lambda prefix="": (work.mkdir(), str(work))[1])

    def broken(*a, **kw):
        (work / "partial.bin").write_bytes(b"x")
        raise ValueError("DRM")

    monkeypatch.setattr(mobi_format, "unpackBook", broken)
    src = tmp_path / "book.azw3"
    src.write_bytes(b"not a real book")
    with pytest.raises(ValueError):
        run(mobi_format.translate_mobi(src, tmp_path, runner(), False, "zh-CN"))
    assert not work.exists()


def test_mobi_temp_dir_removed_after_cancelled_unpack(tmp_path, monkeypatch):
    """解包中途任务被取消：等解包线程写完再删目录，不能留下线程后来写的文件。"""
    import time

    from app.formats import mobi as mobi_format

    work = tmp_path / "mobiex-cancel"
    monkeypatch.setattr(mobi_format.tempfile, "mkdtemp", lambda prefix="": (work.mkdir(), str(work))[1])

    def slow(src, out, **kw):
        time.sleep(0.3)
        (Path(out) / "mobi7").mkdir()
        (Path(out) / "mobi7" / "book.html").write_text("<html></html>")

    monkeypatch.setattr(mobi_format, "unpackBook", slow)
    src = tmp_path / "book.mobi"
    src.write_bytes(b"not a real book")

    async def scenario():
        task = asyncio.create_task(mobi_format.translate_mobi(src, tmp_path, runner(), False, "zh-CN"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.6)

    run(scenario())
    assert not work.exists()

def test_cancelled_calibre_conversion_is_killed(tmp_path, monkeypatch):
    """转 MOBI 时任务被暂停或删除：ebook-convert 要跟着停，不能在后台接着往输出目录写。"""
    import os

    from app.formats import mobi as mobi_format

    pid_file = tmp_path / "pid"
    exe = tmp_path / "ebook-convert"
    exe.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 30\n")
    exe.chmod(0o755)
    monkeypatch.setattr(mobi_format.shutil, "which", lambda name: str(exe))

    def alive(pid: int) -> bool:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        except FileNotFoundError:
            return False
        return state != "Z"

    async def scenario():
        task = asyncio.create_task(mobi_format._calibre_convert(tmp_path / "a.epub", tmp_path / "a.mobi"))
        while not pid_file.exists() or not pid_file.read_text().strip():
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        pid = int(pid_file.read_text())
        for _ in range(50):
            if not alive(pid):
                return pid
            await asyncio.sleep(0.02)
        os.kill(pid, 9)
        raise AssertionError("ebook-convert still running")

    run(scenario())


def test_markdown_list_does_not_swallow_following_blocks():
    """列表后没有空行时，紧跟的标题 / 引用是独立的块，不能并进最后一个列表项。"""
    blocks = split_blocks("- item\n# Heading\n> quote")
    assert [b.kind for b in blocks] == ["list", "heading", "quote"]
    assert collect_segments(blocks) == ["item", "Heading", "quote"]


def test_markdown_list_and_quote_interrupt_paragraph():
    """段落后紧跟列表或引用、中间没有空行时，列表和引用是独立的块，不能并进段落。"""
    blocks = split_blocks("Steps to install:\n- download\n- run setup\n\nSee below:\n> note here")
    assert [b.kind for b in blocks if b.kind != "raw"] == ["para", "list", "para", "quote"]
    texts = collect_segments(blocks)
    assert texts == ["Steps to install:", "download", "run setup", "See below:", "note here"]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == "T:Steps to install:\n- T:download\n- T:run setup\n\nT:See below:\n> T:note here"


def test_markdown_quote_keeps_paragraphs_alerts_and_lists():
    """引用里的提示框标记、空行分段、列表、代码块都保留，每段单独翻译。"""
    src = "> [!NOTE]\n> Useful information\n> on two lines.\n>\n> Second paragraph.\n> - first point\n> ```\n> code here\n> ```"
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["Useful information on two lines.", "Second paragraph.", "first point"]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == ("> [!NOTE]\n> T:Useful information on two lines.\n>\n> T:Second paragraph.\n"
                   "> - T:first point\n> ```\n> code here\n> ```")
    both = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert "> Second paragraph.\n>\n> T:Second paragraph." in both
    assert "> - first point  \n>   T:first point" in both


def test_markdown_lazy_quote_continuation_stays_in_the_quote():
    """引用段落后面不带 > 的续行还属于这一段（CommonMark 懒续行），一句话不能拆成两段翻译。
    标题、列表、引用里的代码块后面的行不是续行。"""
    src = ("> This is a quote that\ncontinues lazily here.\n> - point one\nstill point one\n\nAfter.\n\n"
           "> quote\n# Heading\n\n> ```\n> code\nnot code\n\n> # Title\nplain")
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["This is a quote that continues lazily here.", "point one still point one", "After.",
                     "quote", "Heading", "not code", "Title", "plain"]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out.startswith("> T:This is a quote that continues lazily here.\n> - T:point one still point one\n\nT:After.")
    both = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert both.startswith("> This is a quote that\ncontinues lazily here.\n>\n> T:This is a quote that continues lazily here.")
    assert out.endswith("> # T:Title\nT:plain") and "> # Title\n> # T:Title\n" in both


def test_markdown_link_definitions_kept_verbatim():
    """链接引用定义原样保留，不能并成一行；脚注定义有正文，照常翻译。"""
    src = "See [docs][d] and ![ci][c].\n\n[d]: https://example.com/docs\n[c]: https://img.shields.io/ci.svg\n\n[^1]: A footnote here."
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["See [docs][d] and ![ci][c].", "[^1]: A footnote here."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert "\n[d]: https://example.com/docs\n[c]: https://img.shields.io/ci.svg\n" in out


def test_markdown_title_skips_heading_without_text(tmp_path):
    """第一个一级标题只有数字时没有可翻译片段，取标题不能越界。"""
    src = tmp_path / "notes.md"
    src.write_text("# 2024\n\nSome body text here.\n", encoding="utf-8")
    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    assert out.read_text(encoding="utf-8") == "# 2024\n\n[zh-CN] Some body text here.\n"


def test_markdown_setext_headings():
    """Setext 标题的下划线不送去翻译，译文下面保留原来的下划线。"""
    blocks = split_blocks("Main Title\n==========\n\nSection Name\n---\n\nBody text.")
    texts = collect_segments(blocks)
    assert texts == ["Main Title", "Section Name", "Body text."]
    from app.formats.markdown import segment_kinds
    assert segment_kinds(blocks) == ["h1", "h2", "p"]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == "T:Main Title\n==========\n\nT:Section Name\n---\n\nT:Body text."
    both = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert both.startswith("Main Title\n==========\n\nT:Main Title\n==========\n")


def test_markdown_toml_front_matter_kept():
    """Hugo 的 +++ front matter 和 YAML 一样原样保留，不送去翻译。"""
    blocks = split_blocks('+++\ntitle = "Demo post"\ndraft = false\n+++\n\nBody text.')
    texts = collect_segments(blocks)
    assert texts == ["Body text."]
    assert render(blocks, ["T"], bilingual=False).startswith('+++\ntitle = "Demo post"\ndraft = false\n+++\n')


def test_markdown_long_fence_not_closed_by_inner_fence():
    """四个反引号的围栏里的 ``` 是内容，不能提前结束代码块。"""
    src = "````markdown\n```python\nprint(1)\n```\n````\n\nAfter the block."
    blocks = split_blocks(src)
    assert collect_segments(blocks) == ["After the block."]
    assert blocks[0].kind == "raw" and blocks[0].lines[-1] == "````"


def test_markdown_loose_list_continuation_translated_in_place():
    """松散列表里缩进的续段属于列表项，照常翻译并保留缩进；多缩进 4 格的才是代码块。"""
    src = ("- item one\n\n    continued paragraph of item one.\n\n        code_in_item()\n\n"
           "Outside paragraph.\n\n    real_code()")
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["item one", "continued paragraph of item one.", "Outside paragraph."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == ("- T:item one\n\n    T:continued paragraph of item one.\n\n        code_in_item()\n\n"
                   "T:Outside paragraph.\n\n    real_code()")


def test_markdown_tab_indented_code_kept_verbatim():
    """Tab 缩进和四个空格一样是代码块；列表项里 Tab 缩进的续段还是正文。"""
    src = "Run this:\n\n\tpip install foo\n\tfoo --help\n\nDone.\n\n- item one\n\n\tcontinued paragraph."
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["Run this:", "Done.", "item one", "continued paragraph."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert "\n\tpip install foo\n\tfoo --help\n" in out and "\n\tT:continued paragraph." in out


def test_markdown_paragraph_starting_with_inline_tag_is_translated():
    """行首是行内标签或自动链接的段落照常翻译；块级标签和整行单个标签仍按 HTML 块原样保留。"""
    src = ("<kbd>Ctrl</kbd> copies the selection.\n\n<https://example.com> is our site.\n\n"
           '<div align="center">\nCentered text\n</div>\n\n<img src="logo.png">\n\n<!-- note -->')
    blocks = split_blocks(src)
    assert collect_segments(blocks) == ["<kbd>Ctrl</kbd> copies the selection.", "<https://example.com> is our site."]


def test_markdown_references_kept_original(tmp_path):
    """参考文献章节保留原文：模型原样返回的引文会被当成回显算失败，条目一多整本书就报错。"""
    src = tmp_path / "paper.md"
    src.write_text(
        "# Paper\n\nBody paragraph here.\n\n## References\n\n"
        "- Vaswani A. Attention is all you need. 2017.\n- Devlin J. BERT. 2019.\n\n"
        "### Older works\n\nSmith J. A book. 1990.\n\n## Appendix\n\nAppendix text.\n",
        encoding="utf-8",
    )
    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    text = out.read_text(encoding="utf-8")
    assert "## [zh-CN] References" in text and "### [zh-CN] Older works" in text
    assert "- Vaswani A. Attention is all you need. 2017." in text
    assert "\nSmith J. A book. 1990.\n" in text
    assert "[zh-CN] Body paragraph here." in text and "[zh-CN] Appendix text." in text


def test_markdown_repeated_text_in_body_and_references(tmp_path):
    """同一句在正文和参考文献里都有：按位置写回，正文那处照常翻译，参考文献那处保留原文。"""
    src = tmp_path / "paper.md"
    src.write_text("# Intro\n\nDeep learning. MIT Press, 2016.\n\n## References\n\nDeep learning. MIT Press, 2016.\n",
                   encoding="utf-8")
    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    assert out.read_text(encoding="utf-8") == (
        "# [zh-CN] Intro\n\n[zh-CN] Deep learning. MIT Press, 2016.\n\n"
        "## [zh-CN] References\n\nDeep learning. MIT Press, 2016.\n")


def test_markdown_numbers_inside_paragraph_stay_in_paragraph():
    """有序列表只有从 1 开始才能打断段落，正文里换行后的年份不是列表。"""
    blocks = split_blocks("It happened in\n1990. Then more text.")
    assert [b.kind for b in blocks] == ["para"]


def test_markdown_display_math_kept_verbatim():
    """$$ 公式块原样保留：不送去翻译，块里 + / - 开头的行也不是列表项。"""
    src = ("Energy is given by\n$$\nL = a\n+ b\n- c\n$$\nwhere m is mass.\n\n"
           "$$ E = mc^2 $$\n\n$$x$$ alone is not a block here.\n\nCosts $$5 today.")
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["Energy is given by", "where m is mass.", "$$x$$ alone is not a block here.", "Costs $$5 today."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert "$$\nL = a\n+ b\n- c\n$$" in out and out.count("E = mc^2") == 1


def test_markdown_unclosed_display_math_is_text():
    """没闭合的 $$ 遇到空行就不算公式，后面的段落照常翻译。"""
    blocks = split_blocks("$$ broken formula\n\nNext paragraph.")
    assert collect_segments(blocks) == ["$$ broken formula", "Next paragraph."]


def test_markdown_tables_with_short_or_aligned_separators():
    """GFM 分隔行每格一个 - 就够，可以带冒号对齐；列数和表头不一样的不是表格。"""
    for sep in ("|-|-|", "| :-: | -: |", ":--|--:"):
        blocks = split_blocks(f"| Name | Value |\n{sep}\n| alpha | one |")
        assert [b.kind for b in blocks] == ["table"], sep
        texts = collect_segments(blocks)
        assert texts == ["Name", "Value", "alpha", "one"]
        out = render(blocks, ["T:" + x for x in texts], bilingual=False)
        assert out == f"| T:Name | T:Value |\n{sep}\n| T:alpha | T:one |"
    # 表头两列、分隔行一列：是 Setext 二级标题
    blocks = split_blocks("Name | Value\n---")
    assert [b.kind for b in blocks] == ["setext"]


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
    """行尾连字符断词接回去，全大写的词也一样。数字后面的、小写字母接大写的是名字里本来的连字符，留着。"""
    assert _join_lines(["word-", "wrap"]) == "wordwrap"
    assert _join_lines(["INTER-", "NATIONAL"]) == "INTERNATIONAL"
    assert _join_lines(["Claude-3.7-", "Sonnet"]) == "Claude-3.7-Sonnet"
    assert _join_lines(["DeepSeek-R1-Distill-", "Qwen-7B"]) == "DeepSeek-R1-Distill-Qwen-7B"
    assert _join_lines(["Donau-", "Dampfschiff"]) == "Donau-Dampfschiff"
    # TeX 在大写字母前断开的 McCallum：全文别处写的是 McCallum，连字符去掉
    from app.formats.pdf import _CAMEL_WORDS
    token = _CAMEL_WORDS.set(frozenset({"McCallum"}))
    try:
        assert _join_lines(["({c0000ff}Mc-", "Callum and"]) == "({c0000ff}McCallum and"
        assert _join_lines(["Witten-", "Bell discounting"]) == "Witten-Bell discounting"
    finally:
        _CAMEL_WORDS.reset(token)
    assert _join_lines(["hello", "world"]) == "hello world"  # 非断词只是换行
    assert _join_lines(["the {v1}", ". This"]) == "the {v1}. This"
    assert _join_lines(["end", ", next"]) == "end, next"


def test_a_url_broken_across_lines_is_joined_without_a_space():
    """网址在 / 处、协议名后面折行，接回来中间没有空格；网址写完了、下一行是普通词的照常加空格。"""
    assert _join_lines(["available at https://github.com/", "tensorflow/tensor2tensor."]) == (
        "available at https://github.com/tensorflow/tensor2tensor."
    )
    assert _join_lines(["University, http:", "//www.ling.ohio-state.edu/", "research/phonetics/E_ToBI/."]) == (
        "University, http://www.ling.ohio-state.edu/research/phonetics/E_ToBI/."
    )
    assert _join_lines(["visit https://site.com/", "for details"]) == "visit https://site.com/ for details"
    assert _join_lines(["the ratio a:", "b holds"]) == "the ratio a: b holds"


def test_a_mark_closed_at_a_line_end_and_reopened_continues():
    """每行各自收起标记。下一行原样打开的接着上一行，断词的连字符才认得出来。"""
    assert _join_lines(["In {i}Proceedings of the 2006 confer-{/i}", "{i}ence on methods{/i}, pages"]) == (
        "In {i}Proceedings of the 2006 conference on methods{/i}, pages"
    )
    assert _join_lines(["{c0000ff}Supervised{/c}", "{c0000ff}learning{/c} of"]) == "{c0000ff}Supervised learning{/c} of"
    # 换了字号、换了强调的不接
    assert _join_lines(["{z80}{b}A{/b}{/z}", "{z90}{b}B{/b}{/z}"]) == "{z80}{b}A{/b}{/z} {z90}{b}B{/b}{/z}"
    assert _join_lines(["a {b}x{/b}", "{i}y{/i}"]) == "a {b}x{/b} {i}y{/i}"


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


def test_a_full_row_split_after_a_sentence_stays_in_the_paragraph():
    """两端对齐的段落里，句号后的宽空格让 PyMuPDF 把一整行拆成两截：前一截从段落左缘起，
    后一截顶到右缘，上下都是整行。这是同一行，不是新段落（摘要第三行 “bles.  To realize”）。"""
    from app.formats.pdf import _split_lines

    def line(text, x0, x1, y0):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 10), [{"size": 9.0, "font": "Times-Roman", "text": text}])

    items = [
        line("processing time, especially for aggregation queries over large ta-", 54.0, 293.8, 190.6),
        line("bles.", 54.0, 70.8, 200.6),
        line("To realize this potential, the query optimizer must know", 78.5, 293.7, 200.6),
        line("how and when to exploit materialized views. This paper presents a", 54.0, 293.8, 210.6),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3]]
    # 单独一行的两截可能是侧标和正文，照旧按句末拆开
    assert _split_lines(items[1:3]) == [[0], [1]]


def test_a_number_wrapped_to_its_own_line_ends_the_sentence():
    """排满的一行没说完，折下来独占一行的数是句尾，不按符号行拆出去。行首的节号后面还有标题，照旧拆开。"""
    from app.formats.pdf import _split_lines

    def line(text, x0, x1, y0):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), [{"size": 10.0, "font": "Times-Roman", "text": text}])

    items = [
        line("perparameter that controls the penalty term; values typically range from 0.01 to", 160.2, 493.2, 121.9),
        line("0.1.", 160.2, 175.1, 133.8),
        line("As illustrated in Fig. 8.12, DPO uses gradient descent with this loss over the", 175.1, 493.2, 145.8),
    ]
    assert _split_lines(items) == [[0, 1], [2]]
    toc = [
        line("9.5 Fine-Tuning for Sequence Labeling: Named Entity Recognition . . . 238", 160.2, 493.2, 121.9),
        line("9.6", 160.2, 175.1, 133.8),
        line("Summary", 190.0, 230.0, 133.8),
    ]
    assert [0] in _split_lines(toc)


def test_pdf_hard_break_splits_lines():
    """无项目符号的硬换行（上行远短于块最宽行、下行大写开头、左对齐）也拆成两段。"""
    from app.formats.pdf import extract_blocks

    doc = _pdf_with_text("Short Name\nA Much Longer Affiliation Line Here")
    texts = [b.text for b in extract_blocks(doc)]
    assert texts == ["Short Name", "A Much Longer Affiliation Line Here"]


def _line(text: str, x0: float, x1: float, y0: float, size: float = 10) -> tuple[str, pymupdf.Rect, list[dict[str, float]]]:
    return (text, pymupdf.Rect(x0, y0, x1, y0 + size + 2), [{"size": size}])


def test_a_table_row_name_is_cut_from_its_numbers_and_the_rows_around():
    """表格的行名和同一行的几格数、上面的分组名各自成段。挨得近的数照样拆开；
    PyMuPDF 并进行名那一行的几格数从行里拿掉，和别的行的数一样留在原页上。"""
    from app.formats.pdf import _drop_number_tails, _split_lines

    def span(text, x0, x1, y):
        return {"text": text, "size": 9.0, "bbox": (x0, y, x1, y + 9), "origin": (x0, y + 7), "font": "Times"}

    def item(parts, y):
        spans = [span(text, x0, x1, y) for text, x0, x1 in parts]
        text = "".join(part[0] for part in parts)
        rect = pymupdf.Rect(parts[0][1], y, parts[-1][2], y + 9)
        return (text, rect, [one for one in spans if one["text"].strip()], spans)

    rows = [
        item([("Top Leaderboard Systems (Dec 10th, 2018)", 102.8, 259.5)], 328.6),
        item([("Human", 81.4, 107.8)], 338.6),
        item([("86.3", 198.5, 214.2), (" ", 214.2, 220.7), ("89.0", 220.7, 236.4), (" ", 236.4, 243.0),
              ("86.9", 243.0, 258.7), (" ", 258.7, 265.2), ("89.5", 265.2, 280.9)], 338.6),
        item([("#1 Single - MIR-MRC (F-Net)", 81.4, 192.0)], 348.6),
        item([("-", 204.8, 207.8)], 348.6),
        item([("74.8", 243.0, 258.7), (" ", 258.7, 265.2), ("78.0", 265.2, 280.9)], 348.6),
        item([("#2 Single - nlnet", 81.4, 140.9)], 358.5),
        item([("74.2", 243.0, 258.7), (" ", 258.7, 265.2), ("77.1", 265.2, 280.9)], 358.5),
        item([("BERT (Ens.+TriviaQA)", 81.4, 190.5), (" ", 190.5, 198.5), ("86.2", 198.5, 214.2),
              (" ", 214.2, 220.7), ("92.2", 220.7, 236.4)], 368.5),
    ]
    rows = _drop_number_tails(rows)
    assert rows[-1][0] == "BERT (Ens.+TriviaQA)"
    texts = [" ".join(rows[i][0] for i in group) for group in _split_lines(rows)]
    assert "Human" in texts and "#1 Single - MIR-MRC (F-Net)" in texts and "#2 Single - nlnet" in texts
    assert "Top Leaderboard Systems (Dec 10th, 2018)" in texts


def test_header_cells_over_number_columns_are_cut_apart():
    """表头一行的几格挨得近，并成了一行。每截正对着下面一列数时拆开，行首行名那一列的表头可以不对。
    两端对齐的正文行拆出来的是半句，不拆。"""
    from app.formats.pdf import _cut_header_cells

    def span(text, x0, x1, y=238.1):
        return {"text": text, "size": 9.0, "bbox": (x0, y, x1, y + 9), "origin": (x0, y + 7), "font": "Times"}

    def item(spans):
        text = "".join(one["text"] for one in spans)
        rect = pymupdf.Rect(spans[0]["bbox"][0], spans[0]["bbox"][1], spans[-1]["bbox"][2], spans[0]["bbox"][3])
        return (text, rect, [one for one in spans if one["text"].strip()], spans)

    columns = [411.8, 452.7, 501.6]
    header = item([span("Fine-tune", 394.6, 429.0), span(" ", 429.0, 435.5), span("Fine-tune", 435.5, 469.9),
                   span(" ", 469.9, 476.5), span("Feature-based", 476.5, 526.7)])
    items, cells = _cut_header_cells([header], columns)
    assert [one[0] for one in items] == ["Fine-tune", "Fine-tune", "Feature-based"] and cells == {0, 1, 2}
    stub = [item([span("System", 317.0, 343.4)]), item([span("Dev F1", 395.0, 428.6), span(" ", 428.6, 435.2),
                                                         span("Test F1", 439.4, 466.0)])]
    items, cells = _cut_header_cells(stub, columns)
    assert [one[0] for one in items] == ["System", "Dev F1", "Test F1"]
    prose = item([span("Second,", 380.0, 406.8), span("there", 412.6, 434.4), span("are", 439.6, 452.9),
                  span("major", 458.1, 483.6), span("computational", 488.7, 551.2)])
    items, cells = _cut_header_cells([prose], [393.4, 470.0])
    assert len(items) == 1 and not cells


def test_a_wrapped_formula_wider_than_a_line_stays_in_pieces():
    """换行切开的式子并成一个占位符的多片。几片并排比一行还宽（跨两行的长网址）时不并：
    一个占位符只能排在一行里，放不下就把整段缩小；分开的两片能在原来的断处换行。"""
    from app.formats.pdf import _join_line_wrapped_formulas

    def span(text, x0, x1, y, font="LMMono10-Regular"):
        return {"text": text, "size": 12.0, "font": font, "flags": 0, "bbox": (x0, y, x1, y + 12),
                "origin": (x0, y + 9)}

    def lines(head_x0, tail_x1):
        return [
            [span("Olympiad ", 70.9, head_x0, 420.0, "URWPalladioL"), span("https://example.org/wiki/AMC_", head_x0, 524.4, 420.0)],
            [span("results?srsltid=AfmBOoqQ6pQic5NCan", 70.9, tail_x1, 433.5),
             span(" positions DeepSeek-R1", tail_x1, 524.4, 433.5, "URWPalladioL")],
        ]

    def runs(span_lines):
        return [{"spans": [span_lines[0][1]], "bbox": pymupdf.Rect(span_lines[0][1]["bbox"]), "pos": (0, 1, 1)},
                {"spans": [span_lines[1][0]], "bbox": pymupdf.Rect(span_lines[1][0]["bbox"]), "pos": (1, 0, 0)}]

    short = lines(400.0, 200.0)
    assert len(_join_line_wrapped_formulas(runs(short), short, 12.0)) == 1
    long = lines(121.2, 470.0)
    assert len(_join_line_wrapped_formulas(runs(long), long, 12.0)) == 2


def test_a_small_caps_heading_takes_the_size_of_its_capitals():
    """小型大写的标题首字母大一号，其余小一号，最长的一截是小号的。块的字号按大写算，
    不然译文比正文还小。往下错开的下标（BERT_LARGE）不算小型大写。"""
    from app.formats.pdf import _block_size

    def span(text, x0, size, base):
        return {"text": text, "size": size, "origin": (x0, base), "bbox": (x0, base - size, x0 + 50, base + 2)}

    heading = [span("I", 123.0, 11.6, 362.0), span("NITIALIZATION", 127.0, 9.3, 362.0)]
    assert _block_size(heading[1], [heading]) == 11.6
    model = [span("BERT", 324.5, 9.0, 135.0), span("LARGE", 347.0, 6.5, 137.5)]
    assert _block_size(model[1], [model]) == 6.5


def test_a_numbered_lemma_and_a_displayed_formula_start_their_own_paragraphs():
    """带编号的引理标签另起一段，上一行停在式子上也一样。两边缩进、只有数学字体、带关系符号的一行
    是单独排的式子，上一行停在逗号上也不接。分式拆出来的求和号那一截不算。"""
    from app.formats.pdf import _split_lines

    def line(text, x0, x1, y, font="NimbusRomNo9L-Regu", flags=0):
        return (text, pymupdf.Rect(x0, y, x1, y + 11),
                [{"text": text, "size": 9.7, "font": font, "flags": flags, "bbox": (x0, y, x1, y + 11)}])

    lemma = [
        line("that contains the ith dimension of the gradients over all iterations till t, g1:t,i = [g1,i, g2,i]",
             105.0, 485.6, 300.0),
        ("Lemma 10.3. Let gt = ∇ft(θt) and g1:t be defined as above and bounded,", pymupdf.Rect(105.0, 314.0, 490.3, 325.0),
         [{"text": "Lemma 10.3.", "size": 9.7, "font": "NimbusRomNo9L-Medi", "flags": 16, "bbox": (105, 314, 160, 325)},
          {"text": " Let gt = ∇ft(θt) and g1:t be defined as above and bounded,", "size": 9.7,
           "font": "NimbusRomNo9L-Regu", "flags": 0, "bbox": (160, 314, 490.3, 325)}]),
    ]
    assert _split_lines(lemma) == [[0], [1]]
    display = [
        line("Lemma 10.2. If a function f : Rd → R is convex, then for all x, y ∈ Rd,", 105.0, 391.0, 209.9),
        line("f(y) ≥ f(x) + ∇f(x)T (y − x)", 235.1, 360.2, 226.0, font="CMMI10"),
        line("The above lemma can be used to upper bound the regret and our proof for the main theorem is",
             105.0, 490.3, 248.0),
    ]
    assert _split_lines(display) == [[0], [1], [2]]


def test_wrapped_letter_period_stays_in_the_paragraph():
    """撑满的上一行没有句末时，下一行的 Q. 是换行，不是列表项。句末之后的 a. 仍拆开。"""
    from app.formats.pdf import _split_lines

    wrapped = [
        _line("corresponding to QA to answer", 72, 540, 414, size=12),
        _line("Q. The cost of answering Q is a function", 72, 540, 427, size=12),
        _line("the simplest cost-model:", 72, 187, 441, size=12),
    ]
    assert _split_lines(wrapped) == [[0, 1, 2]]
    listed = [
        _line("We can now ask the following questions.", 72, 540, 100, size=12),
        _line("a. How many views must we materialize?", 85, 430, 116, size=12),
    ]
    assert _split_lines(listed) == [[0], [1]]
    paren = [
        _line("Note that since the top view is in S.", 72, 500, 80, size=12),
        _line("(a) Let u be the view of least cost", 105, 500, 96, size=12),
        _line("S, there must be at least one view.", 124, 500, 110, size=12),
        _line("(b) If C(v) is smaller then stop.", 104, 480, 126, size=12),
    ]
    assert _split_lines(paren) == [[0], [1, 2], [3]]


def test_a_dash_after_a_full_line_continues_the_sentence():
    """排满的上一行没说完，下一行行首的破折号是句中的，不是弹点。上一行是大一号的等宽网址也一样。
    破折号起头的一项下面的破折号是同级的下一项，缩进、换了字号的是下一级弹点。"""
    from app.formats.pdf import _split_lines

    wrapped = [
        _line("performance of 80%. Most notably, the model attains a score of 143.7/150 in AMC 12 2024",
              70.5, 524.4, 366.6, size=11),
        _line("(https://artofproblemsolving.com/wiki/index.php/2024_AMC_12B_Problems)", 70.5, 525.5, 379.3, size=12),
        _line("- a performance that, when combined with its AIME results, yields a score exceeding the",
              69.4, 524.4, 393.7, size=11),
    ]
    assert _split_lines(wrapped) == [[0, 1, 2]]
    items = [
        _line("– Effect of Number of Training Steps; and", 340.5, 525.5, 168.1, size=10.9),
        _line("– Ablation for Different Masking Proce-", 340.5, 525.5, 183.8, size=10.9),
    ]
    assert _split_lines(items) == [[0], [1]]
    nested = [
        _line("• A relational database layer for Apache HBase", 43.2, 581.3, 101.5, size=25),
        _line("– Query engine", 79.2, 581.3, 127.5, size=22),
    ]
    assert _split_lines(nested) == [[0], [1]]


def test_justified_rows_are_not_a_centered_stack():
    """排满的正文行左缘差一两磅，宽窄几乎一样，不是居中排的一摞。作者名、单位、邮箱宽窄不一。"""
    from app.formats.pdf import _centered_rows

    paragraph = [pymupdf.Rect(x0, y, x1, y + 11) for x0, x1, y in (
        (70.9, 524.4, 285.5), (70.4, 524.4, 326.0), (70.5, 525.5, 379.3), (69.4, 524.4, 393.7))]
    assert not _centered_rows(paragraph, 11.0)
    authors = [pymupdf.Rect(x0, y, x1, y + 10) for x0, x1, y in (
        (424.3, 497.2, 233.5), (424.5, 492.9, 246.0), (422.1, 495.3, 257.2))]
    assert _centered_rows(authors, 10.0)


def test_the_tail_of_a_wrapped_url_does_not_end_the_verbatim_text():
    """折行的网址尾巴落在下一行行首，后面接着正文。行首还是等宽字，没换回正文，不拆段。"""
    from app.formats.pdf import _split_lines

    def span(text, mono):
        return {"text": text, "size": 12.0 if mono else 11.0, "font": "LMMono10-Regular" if mono else "URWPalladioL-Roma",
                "flags": 0}

    lines = [
        ("Olympiad https://artofproblemsolving.com/wiki/index.php/AMC_historical_",
         pymupdf.Rect(70.9, 420.0, 525.0, 432.0),
         [span("Olympiad ", False), span("https://artofproblemsolving.com/wiki/index.php/AMC_historical_", True)]),
        ("results?srsltid=AfmBOoqQ6pQic5NCan_NX1wYgr-aoHgJ33hsq7KSekF-rUwY8TBaBao",
         pymupdf.Rect(70.9, 433.5, 524.4, 445.5),
         [span("results?srsltid=AfmBOoqQ6pQic5NCan_NX1wYgr-aoHgJ33hsq7KSekF-rUwY8TBaBao", True)]),
        ("1). This performance positions DeepSeek-R1 among the nation’s top-tier high school students.",
         pymupdf.Rect(70.6, 447.1, 524.0, 459.1),
         [span("1).", True), span(" This performance positions DeepSeek-R1 among the nation’s top-tier", False)]),
    ]
    assert _split_lines(lines) == [[0, 1, 2]]


def test_an_equals_sign_at_the_end_of_a_list_item_line_keeps_the_item_whole():
    """两端对齐把行尾等号前的空拉到一个字宽以上，等号仍接在这一行，下一行缩在条目号右边的是条目的续行。"""
    from app.formats.pdf import _split_lines

    def piece(text, x0, x1, y0, font="Times-Roman"):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 10), [{"size": 10.0, "font": font, "text": text}])

    items = [
        piece("1. Intractability:", 131.4, 199.1, 474.5),
        ("the case where the integral of the marginal likelihood p(x)", pymupdf.Rect(210.4, 474.5, 485.9, 485.3), [
            {"size": 10.0, "font": "Times-Roman", "text": "the case where the integral of the marginal likelihood"},
            {"size": 10.0, "font": "CMR10", "text": "p(x)"},
        ]),
        piece("=", 496.3, 504.0, 474.5, font="CMR10"),
        piece("R", 143.9, 148.6, 484.6, font="CMEX10"),
        piece("p(z)p(x|z) dz is intractable (so we cannot evaluate or differentiate the marginal like-", 152.2, 504.0, 485.3),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3, 4]]
    # 等号没顶到右缘，这一行没排满，隔一个字宽以上的不接
    loose = [items[1], items[2], piece("where the prior is fixed and the posterior is not", 131.4, 540.0, 500.0)]
    assert _split_lines(loose)[0] == [0]


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


def test_pdf_far_apart_labels_on_one_row_split():
    """同一行上隔开好几个字宽的两截是并排的标签（幻灯片的图注和图里的节点），要拆开。"""
    from app.formats.pdf import _split_lines

    items = [
        _line("Candidate 1:", 61, 154, 119, size=12),
        _line("project", 516, 553, 118, size=12),
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



def test_epub_kbd_text_kept_in_translation_only_mode(tmp_path):
    """段落里只有 <kbd> 这类行内标签时，按键文字要发给翻译服务，仅译文模式下也不能丢。"""
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml("<p>Press <kbd>Ctrl</kbd>+C to copy the selected text.</p>")})
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    z = zipfile.ZipFile(out)
    html = z.read("OEBPS/ch1.xhtml").decode()
    p = BeautifulSoup(html, "lxml").find("p")
    assert "[zh-CN]" in p.get_text() and "Ctrl+C" in p.get_text()
    assert "<kbd>Ctrl</kbd>" in html

def test_epub_bilingual_translation_has_no_duplicate_ids(tmp_path):
    """双语模式下译文段落不能带原文的 id 和分页锚点，否则 id 重复、页码显示两遍。"""
    body = ('<p id="para1"><a id="p12"/>Some <b>bold</b> text and <a id="note3" href="#n3">a note</a> here.</p>'
            '<ul><li><span epub:type="pagebreak" id="p13">13</span>Item with <em>emphasis</em> inside.</li></ul>')
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(body)})
    [out] = run(translate_epub(src, tmp_path, runner(), True, "zh-CN"))
    html = zipfile.ZipFile(out).read("OEBPS/ch1.xhtml").decode()
    for ident in ("para1", "p12", "note3", "p13"):
        assert html.count(f'id="{ident}"') == 1, ident
    assert html.count(">13<") == 1
    assert "[zh-CN]" in html and 'href="#n3"' in html

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

    # 图片平时换成占位符不发给模型；原文自带 {vN} 时才跟着 HTML 走，这里直接查清洗
    src = 'Text {v1} <img src="data:image/png;base64,AAAA"/> with <a href="https://x">link</a>'
    nodes = hb._sanitize_translation(
        src,
        '文字 {v1} <img src="data:image/png;base64,AAAA" onerror="alert(1)"/>'
        '<img src="data:image/svg+xml;base64,PHN2Zw=="/> 和 '
        '<a href="javascript:alert(1)">坏链接</a><a href="https://x">链接</a>')
    soup = BeautifulSoup("<p></p>", "lxml")
    for n in nodes:
        soup.p.append(n)
    raster, svg = soup.find_all("img")
    assert raster["src"].startswith("data:image/png") and not raster.has_attr("onerror")
    assert not svg.has_attr("src")
    bad, good = soup.find_all("a")
    assert not bad.has_attr("href") and good["href"] == "https://x"
    assert hb._unsafe_url("data:image/jpeg;base64,AAAA", "image") is False
    assert hb._unsafe_url("data:image/svg+xml,<svg></svg>", "image") is True


MATH_PARA = ('<p>Let <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math> be a variable '
             'and <img src="eq1.png" alt="y=2x"/> holds.</p>')


def _order(p) -> list[str]:
    """段落里文字和媒体元素的先后：文字取原样，媒体元素取标签名。"""
    return [c.name if getattr(c, "name", None) else str(c) for c in p.contents]


def test_epub_inline_math_and_images_stay_in_place(tmp_path):
    """段落里的 MathML 和图片换成 {vN} 发给模型，译完放回原来的位置；双语译文里也有。"""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml(MATH_PARA), xml=True)
    [(_, text, html)] = hb.find_blocks(soup)
    assert text == "Let {v1} be a variable and {v2} holds." and html is None

    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(MATH_PARA)})
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    p = BeautifulSoup(zipfile.ZipFile(out).read("OEBPS/ch1.xhtml"), "lxml-xml").find("p")
    assert _order(p) == ["[zh-CN] Let ", "math", " be a variable and ", "img", " holds."]

    [out] = run(translate_epub(src, tmp_path, runner(), True, "zh-CN"))
    original, translated = BeautifulSoup(zipfile.ZipFile(out).read("OEBPS/ch1.xhtml"), "lxml-xml").find_all("p")
    assert _order(original) == ["Let ", "math", " be a variable and ", "img", " holds."]
    assert _order(translated) == ["[zh-CN] Let ", "math", " be a variable and ", "img", " holds."]


def test_rich_paragraph_sends_placeholders_instead_of_mathml():
    """富文本段落里的 MathML 不发给模型（<mtext> 会被翻掉），和纯文本路径一样换成占位符。"""
    from app.formats import html_blocks as hb

    body = ('<p>Let <em>f</em> be <math xmlns="http://www.w3.org/1998/Math/MathML">'
            '<mrow><mi>x</mi><mtext>for all</mtext></mrow></math> here.</p>')
    soup = hb.parse(_xhtml(body), xml=True)
    [(el, text, html)] = hb.find_blocks(soup)
    assert html == "Let <em>f</em> be {v1} here." and text == "Let f be {v1} here."
    hb.apply_translation_html(soup, el, html, "设 <em>f</em> 为 {v1}。", False, "zh-CN")
    p = soup.find("p")
    assert _order(p) == ["设 ", "em", " 为 ", "math", "。"]
    assert p.find("mtext").get_text() == "for all"


def test_lost_or_repeated_placeholders_keep_every_formula():
    """模型漏掉、重复或编错占位符：多余的删掉，没放回去的公式按原顺序补在段尾，一个都不丢。"""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml(MATH_PARA), xml=True)
    [(el, text, _)] = hb.find_blocks(soup)
    hb.apply_translation(soup, el, text, "设 {v2} 和 {v2} 以及 {v7}。", False, "zh-CN")
    p = soup.find("p")
    assert _order(p) == ["设 ", "img", " 和 ", " 以及 ", "。", "math"]


def test_identical_images_get_their_own_placeholders():
    """两张一样的图各占一个编号：bs4 的 Tag 相等比的是内容，不能按相等找。"""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml('<p>A <img src="x.png"/> and B <img src="x.png"/> end.</p>'), xml=True)
    [(el, text, _)] = hb.find_blocks(soup)
    assert text == "A {v1} and B {v2} end."
    hb.apply_translation(soup, el, text, "乙 {v2} 甲 {v1} 完。", True, "zh-CN")
    assert len(soup.find_all("img")) == 4


def test_literal_placeholder_text_is_not_a_slot():
    """原文本来就写着 {v1}（模板语法）时不用占位符，原文的 {v1} 原样翻译。"""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml('<p>Use {v1} in the template <img src="x.png"/> here.</p>'), xml=True)
    [(el, text, _)] = hb.find_blocks(soup)
    assert text == "Use {v1} in the template here."
    hb.apply_translation(soup, el, text, "模板里用 {v1}。", False, "zh-CN")
    p = soup.find("p")
    assert _order(p) == ["img", "模板里用 {v1}。"]


def test_source_line_wraps_are_spaces_not_line_breaks(tmp_path):
    """排版过的 XHTML 段落里文字跨行写，换行只是空白；只有 <br> 才是换行。"""
    body = ("<p>Alice was beginning to get very tired of sitting\n    by her sister on the bank,<br/>\n    and of having nothing to do.</p>"
            "<p>\n<br/>First stanza line,<br/>\nsecond line.<br/>\n<br/>\nNext stanza here.<br/>\n</p>")
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(body)})
    r = runner()
    [out] = run(translate_epub(src, tmp_path, r, False, "zh-CN"))
    p, poem = BeautifulSoup(zipfile.ZipFile(out).read("OEBPS/ch1.xhtml"), "lxml-xml").find_all("p")
    assert len(p.find_all("br")) == 1
    assert p.get_text("|") == "[zh-CN] Alice was beginning to get very tired of sitting by her sister on the bank,|and of having nothing to do."
    # 诗节之间连着两个 <br>，空行要留着；段首段尾的 <br> 不算
    assert poem.get_text("|") == "[zh-CN] First stanza line,|second line.|Next stanza here."
    assert [c.name for c in poem.contents if c.name] == ["br", "br", "br"]


def test_body_with_only_bare_text_is_translated(tmp_path):
    """<body> 里直接是文字、没有任何块级元素（MOBI 转出来的章节常见）：照样翻译。"""
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml("Some chapter text.<br/>More <i>text</i>.")})
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    body = BeautifulSoup(zipfile.ZipFile(out).read("OEBPS/ch1.xhtml"), "lxml-xml").find("body")
    assert "[zh-CN] Some chapter text." in body.get_text() and body.find("i") is not None


def test_paragraph_with_only_a_formula_is_not_sent():
    """只有一个公式、没有文字的段落不送去翻译。"""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml('<p><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math></p>'), xml=True)
    assert hb.find_blocks(soup) == []


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


def test_pdf_biblio_skips_alpha_labels_at_uniform_size():
    """作者缩写标签的参考文献，字号和正文一样（点阵字 OCR 后会统一）：OCR 认丢了 [ 的标签、
    单独一行的标签、句号收尾的短续行都不能被当成下一章，整节保留原文。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size=10):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=False)

    blocks = [
        blk("We thank our colleagues for help with the experiments."),
        blk("References"),
        blk("[CS94] S. Chaudhuri and K. Shim. Including Group-By in Query Optimization. In VLDB, 1994."),
        blk("ESS]"),
        blk("Arbor Software Inc. Multidimensional Analysis: Converting Corporate Data into Strate-"),
        blk("GBLP95] J. Gray, A. Bosworth, A. Layman, H. Pirahesh Data Cube: A Relational Aggregation"),
        blk("Operator Generalizing Group-By, Cross-Tab, and Sub-Totals. Microsoft Technical Report"),
        blk("STG]"),
        blk("Databases. White Paper."),
        blk("[YL95] W. P. Yan and P. A. Larson. Eager Aggregation and Lazy Aggregation. In VLDB, 1995."),
    ]
    assert biblio_skips(blocks) == [False, False] + [True] * 8
    # 句号收尾不影响没有句号的下一章标题退出
    tail = blocks[:3] + [blk("Appendix"), blk("Appendix body text continues here.")]
    assert biblio_skips(tail) == [False, False, True, False, False]

def test_pdf_biblio_skips_weigh_sizes_by_text():
    """表格里成串的小字格子按块数比正文多，按块数取的中位落到小字号上，同字号的引文就像标题了。
    中位按字数取。标题比正文小一号、整节参考文献都小一号的（ICASSP），加粗又不比引文小，也是标题。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    body = [blk("Learning widely applicable representations of words has been an active area of research " * 4, 10)
            for _ in range(3)]
    cells = [blk(f"{80 + n}.5", 9) for n in range(20)]
    entries = [blk("Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018.", 10),
               blk("Contextual string embeddings for sequence labeling.", 10),
               blk("Rami Al-Rfou, Dokook Choe, and Llion Jones. 2018.", 10),
               blk("Character-level language modeling with deeper self-attention.", 10)]
    skips = biblio_skips([*body, *cells, blk("References", 12), *entries])
    assert skips[-4:] == [True] * 4
    small = [blk("[1] R. Xu and Q. Liu, \u201cX-Voice,\u201d in Proc. Interspeech, 2023, pp. 1\u20135.", 8),
             blk("[2] M. Yang and J. Hansen, \u201cActivation steering,\u201d in Proc. ICASSP, 2024.", 8)]
    skips = biblio_skips([*[blk("Body text of the paper. " * 20, 9) for _ in range(5)], blk("8. REFERENCES", 8, True), *small])
    assert skips[-2:] == [True, True]


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


def test_a_proof_mark_after_the_full_stop_still_ends_the_paragraph():
    """段末的证毕号从 Symbol 字体里取出来是 \x00，句号在它前面。下一页大写起头的是新段，不接上来。"""
    from app.formats.pdf import TextBlock, _can_continue, _ends_sentence

    def blk(text, page):
        return TextBlock(page=page, rect=pymupdf.Rect(318, 700, 558, 729), line_rects=[],
                         text=text, size=10, color="#000000", bold=False)

    end = blk("Consequently, the query output can be computed from the view. \x00", 0)
    assert _ends_sentence(end.text) and _ends_sentence("which proves the claim. □")
    nxt = blk("As this example illustrated, to correctly test availability of output columns", 1)
    assert not _can_continue(end, nxt, 10, None, allow_capital=True)
    # 公式哨兵的分隔符不当证毕号剥掉
    assert not _ends_sentence("f\x01s\x02i\x01/s\x02(")


def test_a_column_break_inside_brackets_or_after_et_al_continues():
    """换栏、换页处圆括号起头的看括号里的字，句中的 et al. 不算句末，分号、冒号后面小写的接着写。编号、音标、引文另起，
    栏底整行斜体的小标题也不往下接。"""
    from app.formats.pdf import TextBlock, _can_continue

    def blk(text, page):
        return TextBlock(page=page, rect=pymupdf.Rect(318, 700, 558, 729), line_rects=[],
                         text=text, size=10, color="#000000", bold=False)

    def joins(prev, nxt):
        return _can_continue(blk(prev, 0), blk(nxt, 1), 10, None, allow_capital=True)

    assert joins("Accordingly, we use frame-wise root-mean-square", "(RMS) loudness to describe the trajectory.")
    assert joins("any of its variants such as Adagrad", "(Duchi et al., 2011). The normalization of activations")
    assert joins("work on 3D multimodal LLMs [Hong et al. 2023; Mao et al.", "2026; Xu et al. 2024] shows that LLMs")
    assert not joins("the scores are combined as shown below", "(3) where the weights sum to one")
    assert not joins("The Vocal Organs", "[v], [z], and all the English vowels, among others.")
    assert not joins("we use the fact that", "“[M]achines of this character can behave in a complicated manner.”")
    assert not joins("The scores are combined as shown below.", "(RMS) loudness to describe the trajectory.")
    assert joins("the policy is confused on BC trials;", "an explanation that also fails to cover the rest.")
    assert not joins("the policy is confused on BC trials;", "An explanation that also fails to cover the rest.")
    # 冒号后面的行间公式跳过了，下一页小写起头的 where 接着写
    assert joins("into a biaffine scoring function:", "where U, W, and b are weights learned by the model.")
    assert not joins("into a biaffine scoring function:", "We use U, W, and b as weights learned by the model.")
    assert not _can_continue(blk("the context window from that instance:", 0),
                             blk("is traditionally followed by cherry pie, a traditional dessert", 1), 10, None,
                             allow_capital=True, at_edge=False)
    assert joins("closed-loop reasoning in later work (Fu et al., 2024b;", "Mao et al., 2024) share the same confound.")
    assert not joins("closed-loop reasoning in later work (Fu et al., 2024b;", "Results differ. Both (a) and (b) fail).")
    assert joins("with a dual success metric and scripted certificates", "(§5.3). Urban probes stress the agent.")
    assert joins("Stress-tested on the held-out pool", "(Tables S32–S35), the port scores lower.")
    assert not joins("The scores are combined as shown below.", "(Tables S32–S35), the port scores lower.")
    formula = blk("D(·||·) between the data generating process and the model.", 1)
    formula.span_lines = [[{"text": "D", "font": "CMMI10", "flags": 2, "size": 10},
                           {"text": "(·||·) between the data generating process and the model.", "font": "Times", "flags": 0, "size": 10}]]
    assert _can_continue(blk("results from trying to minimise a divergence,", 0), formula, 10, None)
    assert not _can_continue(blk("results from trying to minimise a divergence.", 0), formula, 10, None)
    bullet = blk("• We introduce finite-transition modeling for accelerated MRI.", 1)
    bullet.span_lines = [[{"text": "•", "font": "CMSY7", "flags": 0, "size": 7},
                          {"text": " We introduce finite-transition modeling for accelerated MRI.", "font": "Times", "flags": 0, "size": 10}]]
    assert not _can_continue(blk("loss broadly across the image. At", 0), bullet, 10, None)
    assert joins("similarity recovery comes with higher mean GOP (4.750 vs. 4.733)", "and lower mean WER on the test set.")
    assert not joins("similarity recovery comes with higher mean GOP (4.750 vs. 4.733)", "On the item-matched test set, C wins.")
    heading = blk("E. Energy-Detection Characterization", 0)
    heading.line_rects = [pymupdf.Rect(49, 661, 204, 671)]
    heading.span_lines = [[{"text": heading.text, "font": "NimbusRomNo9L-ReguItal", "flags": 2, "size": 10}]]
    head = blk("This letter proposed an LLM-assisted hierarchical framework", 1)
    assert not _can_continue(heading, head, 10, None, allow_capital=True)
    heading.span_lines = [[{"text": heading.text, "font": "NimbusRomNo9L-Regu", "flags": 0, "size": 10}]]
    assert _can_continue(heading, head, 10, None, allow_capital=True)


def test_a_block_at_the_text_edge_starts_where_the_body_lines_start():
    """冒号下面接着写的半句从正文的左缘起笔；引出的例句、式子缩进或者居中排。
    左缘按同一栏里各段除首行以外的行算，首行缩进的段落不影响，图注居中的续行不算。"""
    from app.formats.pdf import TextBlock, _at_text_edge

    def blk(text, lefts, y, x1=493.0):
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 10) for i, x0 in enumerate(lefts)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = [blk("Apple pie computer peripherals and personal digital assistants. These dimensions", [175, 160, 160], 300),
            blk("Figure 5.3 A spatial visualization of word vectors in two dimensions", [160, 210], 400)]
    example = blk("is traditionally followed by cherry pie, a traditional dessert often mixed", [191, 217, 161], 108, x1=457)
    resumed = blk("then the same kinds of training data would lead the language model to assign", [160, 160, 160], 108)
    assert not _at_text_edge(example, [example, resumed, *body])
    assert _at_text_edge(resumed, [resumed, *body])
    assert not _at_text_edge(blk("fair but brutal", [167], 108, x1=222), body)


def test_a_number_heavy_results_paragraph_is_prose():
    """数字多的结果段字母占比低，仍是正文，能当换栏续句的栏顶。刻度数字那样的短行不是。"""
    from app.formats.pdf import _is_code_line

    assert not _is_code_line(
        "and lower mean WER (0.009 vs. 0.013); paired differences are +0.017 [−0.043, 0.077] "
        "and −0.004 [−0.012, 0.004] (n = 1200/arm), 95% CI.")
    assert _is_code_line("0.1 0.2 0.3 0.4 0.5 0.6 (s)")


def test_a_body_sentence_that_starts_with_figure_is_not_a_caption():
    """编号后面接小写词的是正文（Figure 8(a) shows that …），编号后面是冒号、句点或大写的是图注。"""
    from app.formats.pdf import _is_float_text

    assert not _is_float_text("Figure 8(a) shows that the fidelity converges.")
    assert not _is_float_text("Table 5 evaluates the out-of-distribution results.")
    assert not _is_float_text("Table 1.2 shows the settings.")
    assert _is_float_text("Figure 3: The pipeline.")
    assert _is_float_text("FIG. 8. Temporal and spatial convergence.")
    assert _is_float_text("Table 1.2 Results on the test set")
    assert _is_float_text("Fig. 3 (A) Panel view")


def test_roman_and_appendix_numbers_are_captions():
    """IEEE 的表号是罗马数字（TABLE IV: …），附录的图表号带字母（Figure S1: …）；后面接小写词的仍是正文。"""
    from app.formats.pdf import _is_float_text

    assert _is_float_text("TABLE II: For different values of the number of observed e-folds")
    assert _is_float_text("TABLE I")
    assert _is_float_text("Figure S1: Full OSVD three-view label distributions.")
    assert _is_float_text("Table A2. Hyperparameters")
    assert _is_float_text("Figure B.6: Constrained distributional evolution")
    assert not _is_float_text("Table IV compares inference time.")
    assert not _is_float_text("Figure S1 shows that Satisfaction constitutes the largest share.")
    assert not _is_float_text("Table Index of symbols")


def test_table_cells_do_not_set_the_column_body_size():
    """栏顶一张表一格一块、字号比正文大，正文字号仍按正文块算，栏底的正文段不被当成小一号的脚注。"""
    from app.formats.pdf import TextBlock, _column_body

    def blk(text, y, size, clip=None):
        return TextBlock(page=0, rect=pymupdf.Rect(54, y, 290, y + 20), line_rects=[],
                         text=text, size=size, color="#000000", bold=False, clip=clip)

    cells = [blk(f"AR + VMD {k}", 100 + 12 * k, 11.1, clip=(54, 100, 290, 200)) for k in range(7)]
    body = [blk("Overall, every proposed component helps.", 400 + 60 * k, 10) for k in range(4)]
    heading = blk("Out-of-Distribution Generalization", 650, 10.9)
    kept = _column_body([*cells, *body, heading])
    assert all(block in kept for block in body)


def test_pdf_biblio_numbered_entries_without_a_heading():
    """revtex 的期刊版式不印 References：[1] 起头、编号挨着往下排的一串是参考文献。
    碰到标题停住，另一栏接着排的下一个编号从那里续上。正文里一句 [1] 起头的不算。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size=9.0, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("In future work we will present a more detailed exposition of our methods.", 10),
        blk("[1] B. P. Abbott et al. (LIGO Scientific, Virgo), Phys. Rev. Lett. 116, 061102 (2016)."),
        blk("[2] R. Abbott et al., Phys. Rev. X 13, 041039 (2023), arXiv:2111.03606 [gr-qc]."),
        blk("[3] A. Buonanno and T. Damour, Phys. Rev. D 59, 084006"),
        blk("(1999), arXiv:gr-qc/9811091. [4] A. Buonanno and T. Damour, Phys. Rev. D 62, 064015"),
        blk("(2000), arXiv:gr-qc/0001013."),
        blk("Appendix F: Control-space dimension and optimizer", bold=True),
        blk("Table VIII compares 21- and 27-parameter controls at all three stages.", 10),
        blk("Phys. Rev. D 108, 124035 (2023). [5] L. Pompili et al., Phys. Rev. D 108, 124035 (2023)."),
        blk("[6] https://github.com/DUsseglio/Scattering-Waveform-1SF."),
        blk("SUPPLEMENTAL MATERIAL", bold=True),
        blk("The leading multipole moments take the form below.", 10),
    ]
    assert biblio_skips(blocks) == [False, True, True, True, True, True, False, False, True, True, False, False]
    assert biblio_skips([blk("[1] We first fix the gauge, Phys. Rev. D 59 (1999).", 10),
                         blk("The second step follows.", 10)]) == [False, False]


def test_a_table_with_its_caption_above_is_skipped_at_the_column_top():
    """表的图注排在表上面时，紧接在图注下面的短行是表格，续句是下面第一块排满栏宽的段落。
    图的图注排在图下面，图注下面的短行（式子后面的 subject to）是正文。"""
    from app.formats.pdf import TextBlock, _float_contents

    def blk(text, y, x1, rows=1):
        lines = [pymupdf.Rect(318, y + 11 * i, x1, y + 11 * i + 9) for i in range(rows)]
        return TextBlock(page=0, rect=pymupdf.Rect(318, y, x1, lines[-1].y1), line_rects=lines,
                         text=text, size=9.0, color="#000", bold=False)

    def page(caption_text):
        caption = blk(caption_text, 359, 560, rows=2)
        cells = [blk("Method", 397, 350), blk("w/o point cloud", 412, 380), blk("w/o artist scenes", 434, 390)]
        body = blk("available support surfaces. Figure 7 corroborates this.", 493, 560, rows=4)
        return caption, cells, body

    caption, cells, body = page("Table 3. Controlled ablations on the test set.")
    assert _float_contents([*cells, body], [caption, *cells, body]) == {id(cell) for cell in cells}
    caption, cells, body = page("Fig. 3. Transient feature extraction.")
    assert _float_contents([*cells, body], [caption, *cells, body]) == set()


def test_pdf_biblio_bold_heading_as_large_as_the_entries():
    """条目首行折断成的一串作者名、接在作者名后面的题名短行都不是下一章，哪怕和标题一样大；
    后面加粗的附录标题才是。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=10.0, color="#000000", bold=bold)

    blocks = [
        blk("This work was supported by the National Science Foundation under Grant 2200000."),
        blk("References", bold=True),
        blk("Ahmed M. Alaa, Zaid Ahmad, and Mark van der Laan."),
        blk("Conformal meta-learners for predictive inference of treatment effects. arXiv preprint arXiv:2308.14895, 2023."),
        blk("Anastasios N Angelopoulos, Stephen Bates, Adam"),
        blk("Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. In ICLR, 2024."),
        blk("Jean-Yves Audibert, Sébastien Bubeck, and Rémi"),
        blk("Munos. Best arm identification in multi-armed bandits. In COLT, pages 41–53, 2010."),
        blk("Jiayuan Su, Jing Luo, Hongwei Wang, and Lu Cheng."),
        blk("API is enough: Conformal prediction for large lan-"),
        blk("guage models without logit-access. In Findings of EMNLP, 2024."),
        blk("Proofs and Theoretical Results", bold=True),
        blk("We first restate the main theorem and then prove it in two steps."),
    ]
    assert biblio_skips(blocks) == [False, False, *[True] * 9, False, False]


def test_ieee_section_headings_are_headings():
    """IEEE 的节标题和正文一样大：罗马数字后面全大写（小型大写）、字母编号整行斜体的是标题。
    正文里的人名缩写、带逗号的作者行不是。"""
    from app.formats.pdf_roles import _is_lettered_title

    assert _is_lettered_title("II. METHODS", styled=False)
    assert _is_lettered_title("IV. EXPERIMENTS AND RESULTS", styled=False)
    assert _is_lettered_title("A. Materials", styled=True)
    assert not _is_lettered_title("A. Materials", styled=False)
    assert not _is_lettered_title("B. Smith", styled=False)
    assert not _is_lettered_title("V. Vuletić, and M. D. Lukin,", styled=True)


def test_a_column_does_not_continue_under_a_heading_at_its_top():
    """右栏顶上紧挨着本栏的小标题时，右栏是新的一节，左栏末句不往下接；
    小标题和栏顶之间还隔着别的字（图里认成标题的大字）时照常接。"""
    from app.formats.pdf import TextBlock, _column_continuations

    def blk(text, x0, y, rows=3):
        return TextBlock(page=0, rect=pymupdf.Rect(x0, y, x0 + 245, y + 12 * rows), line_rects=[],
                         text=text, size=10.0, color="#000000", bold=False)

    left = [blk("The arm is described in the next section of this paper in full detail.", 54, 100),
            blk("Our solution is a novel tendon-actuated arm that we describe below.", 54, 200),
            blk("The arm is mechanically simpler than comparative actuators [17], [18],", 54, 600)]
    heading = blk("A. Materials", 313, 89, rows=1)
    right = [blk("the soft structure including the chambers is cast in one piece.", 313, 105),
             blk("To facilitate free sliding of the tendon the ribbon is coated.", 313, 300)]

    def pairs(extra):
        blocks = [*left, heading, *extra, *right]
        return _column_continuations({0: blocks}, 10.0, {id(block) for block in extra}, {id(heading)}, set())

    assert id(left[-1]) not in pairs([])
    heading.rect = pymupdf.Rect(313, 60, 400, 72)
    label = blk("Reward Model", 313, 80, rows=1)
    label.rect = pymupdf.Rect(330, 80, 380, 90)
    assert pairs([label]).get(id(left[-1])) is right[0]
    # 小节号悬在栏边左面（1.4 Related work）的也是栏顶的小标题
    heading.rect = pymupdf.Rect(290, 89, 420, 101)
    assert id(left[-1]) not in pairs([])


def test_the_page_end_is_not_in_a_column_with_body_text_to_its_right(monkeypatch):
    """右栏没分出来、块都没归栏时，最后一栏是左栏：右边还排着正文，左栏栏底不是页末，不接下一页的页首。"""
    from app.formats import pdf
    from app.formats.pdf import TextBlock, _column_continuations

    def blk(text, x0, y, page=0):
        return TextBlock(page=page, rect=pymupdf.Rect(x0, y, x0 + 245, y + 36), line_rects=[],
                         text=text, size=10.0, color="#000000", bold=False)

    left = [blk("Text-to-SQL systems allow users to query structured databases in plain", 54, 100),
            blk("language, and most evaluation focuses on the final generated query and", 54, 600)]
    right = [blk("Even when benchmark performance is relatively strong, generated queries fail.", 313, 100),
             blk("This framing is relevant to uncertainty-aware generation, and we treat the", 313, 600)]
    head = blk("results under the conversational-history condition in the rest of the paper.", 54, 80, page=1)
    monkeypatch.setattr(pdf, "_assigned_columns", lambda blocks, anchors: (
        [[block for block in blocks if block.rect.x0 < 300]], [block for block in blocks if block.rect.x0 >= 300]))
    pairs = _column_continuations({0: [*left, *right], 1: [head]}, 10.0, set(), set(), set())
    assert id(left[-1]) not in pairs


def test_a_flush_paragraph_under_a_table_continues_the_column_above():
    """段落首行缩进的文档里，栏顶的表下面齐平起笔的一段是左栏末段的续段，左栏恰好在句末断开、
    续段大写起头也接上。续段首行缩进的是新的一段；正上方是正文字号的短行（PROMPT、漏认的小标题）的
    是新起的一段；文档不缩进段首时不接"""
    from app.formats.pdf import TextBlock, _column_continuations

    def blk(text, x0, y, rows=3, size=10.0, indent=0.0):
        lines = [pymupdf.Rect(x0 + (indent if i == 0 else 0), y + 12 * i, x0 + 245, y + 12 * i + 10)
                 for i in range(rows)]
        return TextBlock(page=0, rect=pymupdf.Rect(x0, y, x0 + 245, y + 12 * rows), line_rects=lines,
                         text=text, size=size, color="#000000", bold=False)

    host = blk("Ablation studies on fastMRI examine the principal design elements; "
               "the support-restricted objective also degrades performance.", 54, 600)
    left = [blk("Table V compares inference time and peak GPU memory across methods here.", 54, 300), host]
    caption = blk("TABLE IV: Cross-dataset generalization. Models were trained on M4Raw.", 313, 100, rows=1)
    cells = [blk("PSNR", 361, 130, rows=1, size=7.3), blk("FLEX-MRI", 315, 160, rows=1, size=7.3)]
    head = blk("Perturbing the training inputs provides no additional benefit. These results", 313, 190)
    later = blk("Fig. 5 further examines the number of retained transitions in the model.", 313, 400, indent=10)

    def pairs(above, indents=True):
        blocks = [*left, caption, *cells, *above, head, later]
        return _column_continuations({0: blocks}, 10.0, set(), set(), set(), indents=indents)

    assert pairs([]).get(id(host)) is head
    assert id(host) not in pairs([], indents=False)
    assert id(host) not in pairs([blk("PROMPT", 313, 175, rows=1)])
    head.line_rects[0].x0 += 10
    assert id(host) not in pairs([])


def test_pdf_biblio_author_year_entries_after_the_heading():
    """作者-年份体的条目不带编号和括号年份。标题后面连着几条「姓, 名缩写.」起头的就进入参考文献；
    正文里只一句这样开头的不算。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=False)

    book = [
        blk("A potential DAMSL labeling of the conversational fragment.", 9),
        blk("Bibliography", 24),
        blk("Abadi, M., A. Agarwal, P. Barham, Agirre, E. and D. Martinez. 2001.", 8),
        blk("E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin.", 8),
        blk("Agichtein, E. and L. Gravano. 2000.", 8),
        blk("Snowball: Extracting relations from large plain-text collections.", 8),
        blk("Subject Index", 24),
    ]
    assert biblio_skips(book) == [False, False, True, True, True, True, False]
    prose = [
        blk("Bibliography", 14),
        blk("Smith, J. and colleagues kept the sources in a shared folder.", 11),
        blk("The chapter continues here.", 11),
    ]
    assert biblio_skips(prose) == [False] * 3


def test_pdf_biblio_numbered_entry_ending_with_pages():
    """老数据库论文的条目是 会议 年份, 页码。，行尾是页码不是年份，也要进入。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    blocks = [
        blk("Body text of the conclusion.", 9),
        blk("8. References", 12, bold=True),
        blk("[1] S. Agrawal, S. Chaudhuri, V. R. Narasayya: Automated Selection of "
            "Materialized Views and Indexes in SQL Databases. VLDB 2000: 496-505", 8),
        blk("[4] S. Chaudhuri, S. Krishnamurthy, S. Potamianos, K. Shim, Optimizing "
            "Queries with Materialized Views, ICDE 1995, 190-200.", 8),
        blk("[6] G. Graefe, The Cascades Framework for Query Optimization, Data "
            "Engineering Bulletin, 18(3), 1995, 19-29.", 8),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True]
    # 不带编号的正文即使以 年份, 页码 收尾也不算
    prose = [
        blk("Body text.", 9),
        blk("References", 12, bold=True),
        blk("The workshop ran in 2001, 10-12.", 9),
        blk("More body text.", 9),
    ]
    assert biblio_skips(prose) == [False] * 4


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


def test_pdf_biblio_author_lists_and_url_halves_stay_in():
    """ACL 体（作者. 年份. 题名. In 出处）进得去；条目字号比正文大一号时，作者一行、折行的网址半截都不是下一章。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size, bold=False):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=bold)

    acl = [
        blk("Body text of the paper.", 10),
        blk("References", 12, bold=True),
        blk("Alan Akbik, Duncan Blythe, and Roland Vollgraf.", 10),
        blk("2018. Contextual string embeddings for sequence labeling. In Proceedings of COLING, pages 1638–1649.", 10),
        blk("Appendix A", 12, bold=True),
    ]
    assert biblio_skips(acl) == [False, False, True, True, False]
    larger = [
        *[blk("Body text of the paper.", 10) for _ in range(5)],
        blk("References", 13, bold=True),
        blk("E. Akyurek, M. Damani, L. Qiu, H. Guo, Y. Kim, and J. Andreas. The surprising effectiveness. arXiv, 2024.", 10.8),
        blk("Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort", 11),
        blk("H. Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https:", 11),
        blk("//github.com/huggingface/open-r1.", 12),
        blk("1b2feafb99a24-Abstract-Datasets_and_Benchmarks_Track.html.", 12),
        blk("D. Ganguli, L. Lovitt, J. Kernion. Red teaming language models. arXiv, 2022.", 11),
    ]
    assert biblio_skips(larger)[6:] == [True] * 6


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
    out = pymupdf.open("pdf", _subset_fonts_safe(doc.tobytes()))
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
    # 正文字体配套的数学字体、newtx 的数学字体
    assert _font_is_math("XCharterMathMI") and _font_is_math("txmiaX") and _font_is_math("ABCDEF+txsys")
    assert not _font_is_math("XCharter-Roman") and not _font_is_math("txtt")
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


def test_italic_letters_between_math_brackets_are_a_formula():
    """正文是 Times 的书里，P(y = 1) 的 P、y 是正文的斜体，括号、等号是标了 math 的 CMR：单看都不成公式，
    连起来是一个行内公式。括号里的斜体词（P(to|want)）也在公式里。只有斜体词、没有公式标点的不算"""
    from app.formats.pdf import _placeholderize

    x = [0.0]

    def span(text, font, flags=4, math=False):
        x0 = x[0]
        x[0] += 5.0 * len(text)
        out = {"text": text, "font": font, "size": 10.0, "flags": flags,
               "bbox": (x0, 100.0, x[0], 110.0), "origin": (x0, 108.0)}
        return {**out, "math": True} if math else out

    def italic(text):
        return span(text, "NimbusRomNo9L-ReguItal", flags=6)

    def cm(text):
        return span(text, "CMR10", math=True)

    line = [span("the two cases,", "NimbusRomNo9L-Regu"), italic(" P"), cm("("), italic("y"), cm(" ="),
            span(" 1", "NimbusRomNo9L-Regu"), cm(")"), span(" and the word", "NimbusRomNo9L-Regu"),
            italic(" great"), span(" here.", "NimbusRomNo9L-Regu")]
    sent, formulas = _placeholderize(_block_with_spans([line], size=10.0), 0)
    assert sent == "the two cases,{v1} and the word {i}great{/i} here."
    assert [formula["text"] for formula in formulas] == ["P(y = 1)"]

    x[0] = 0.0
    line = [span("probability", "NimbusRomNo9L-Regu"), italic(" P"), cm("("), italic("to"),
            span("|", "CMSY10", flags=6), italic("want"), cm(")"), span(" decreases.", "NimbusRomNo9L-Regu")]
    sent, formulas = _placeholderize(_block_with_spans([line], size=10.0), 0)
    assert sent == "probability{v1} decreases."
    assert [formula["text"] for formula in formulas] == ["P(to|want)"]
    # 无衬线粗体的向量 w、x（CMSSBX10）在 Times 排的书里也只出现在公式里
    from app.formats.pdf import _CM_UPRIGHT_RE
    assert _CM_UPRIGHT_RE.match("CMSSBX10") and _CM_UPRIGHT_RE.match("ABCDEF+CMSS10")


def test_detected_formula_boxes_mark_their_letters():
    """公式检测模型框出的 x、y = 1 是行内公式，y = 1 全是正文的正体字，按字体认不出来。
    框把后面的逗号也框进去时，逗号留在正文里；只框着 0 的框不算公式"""
    from app.formats.pdf import _extract, _placeholderize

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 60), "We call the input", fontsize=10, fontname="tiro")
    x0 = 40 + pymupdf.get_text_length("We call the input ", "tiro", 10)
    page.insert_text((x0, 60), "x", fontsize=10, fontname="tiit")
    x1 = x0 + pymupdf.get_text_length("x", "tiit", 10)
    page.insert_text((x1, 60), ", and set y = 1, then 0 and 1.", fontsize=10, fontname="tiro")
    boxes = [page.search_for("x,")[0], page.search_for("y = 1,")[0], page.search_for("0")[0]]

    class Detector:
        def pages(self):
            yield boxes

    [block], _frames, _scans = _extract(doc, formulas=Detector())
    sent, formulas = _placeholderize(block, 0)
    assert sent == "We call the input {v1}, and set {v2}, then 0 and 1."
    assert [formula["text"] for formula in formulas] == ["x", "y = 1"]
    # y = 1 是从一整段正文 span 里切出来的，记着页面上原来那个 span，裁公式时按它认（见 _outside_formula）
    [line] = page.get_text("dict")["blocks"][0]["lines"][-1:]
    [y_span] = formulas[1]["spans"]
    assert pymupdf.Rect(y_span["source"]) == pymupdf.Rect(line["spans"][-1]["bbox"])
    [block], _frames, _scans = _extract(doc)
    assert _placeholderize(block, 0)[0] == "We call the input {i}x{/i}, and set y = 1, then 0 and 1."


def test_blocks_inside_a_display_formula_box_are_kept():
    """公式检测模型框成行间公式的块留原文（align 里的 s.t.、otherwise、\\text{...}）。
    只有一角碰到框的块照翻；框里的说明句（有三个虚词）、标题也照翻"""
    from app.formats.pdf import TextBlock, _isolated_formula_ids
    from app.formats.pdf_roles import Role

    def block(text, rect, page=0):
        rect = pymupdf.Rect(rect)
        spans = [{"text": text, "font": "TimesNewRomanPSMT", "size": 10.0, "flags": 0}]
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False,
                         span_lines=[spans])

    inside = block("otherwise,", (300, 210, 340, 222))
    # 式子右下角外面的一个词，只有一角碰到框
    below = block("and", (390, 230, 450, 242))
    other_page = block("s.t.", (300, 210, 340, 222), page=1)

    class Detector:
        isolated = {0: [pymupdf.Rect(80, 200, 400, 236)]}

    found = _isolated_formula_ids([inside, below, other_page], Detector())
    assert found == {id(inside)}
    assert _isolated_formula_ids([inside], None) == set()
    note = block("(where c is the correct class)", (300, 210, 390, 222))
    title = block("Phoenix + Calcite", (100, 205, 380, 230))
    # 式子里的 perplexity、red、blue 是实词，没有虚词，不是句子
    words = block("perplexityA(T) = PA(red red red red blue)−1/5", (100, 205, 380, 225))
    blocks = [note, title, inside, words]
    found = _isolated_formula_ids(blocks, Detector(), [Role.BODY, Role.HEADING, Role.BODY, Role.BODY])
    assert found == {id(inside), id(words)}


def test_accent_on_its_own_line_joins_the_formula_once():
    """ŝ 的 ˆ 单成一行，排在 s 那一行后面，自己也判成了式子。并进它盖着的式子以后不再单列一个占位符。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, x1, y0, y1, base):
        return {"text": text, "font": font, "size": size, "flags": 0, "bbox": (x0, y0, x1, y1), "origin": (x0, base)}

    # 前面认出它在式子里，标了 math
    hat = [{**span("\u02c6", "CMR10", 10.0, 311.9, 317.4, 242.0, 252.9, 250.5), "math": True}]
    line = [
        span("span ", "NimbusRomNo9L-Regu", 10.0, 280.0, 307.0, 242.2, 253.1, 250.5),
        span("s", "CMMI10", 10.0, 307.3, 312.4, 242.0, 252.9, 250.5),
        span("i,j", "CMMI8", 8.0, 312.4, 321.1, 245.9, 253.9, 252.1),
        span(". We predict a non-null answer", "NimbusRomNo9L-Regu", 10.0, 322.1, 470.0, 242.2, 253.1, 250.5),
    ]
    sent, formulas = _placeholderize(_block_with_spans([line, hat], size=10.0), 0)
    assert sent == "span {v1}. We predict a non-null answer"
    assert len(formulas) == 1
    assert formulas[0]["bbox"].x1 >= 317.4


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


def test_formulas_set_in_text_italic_stay_whole():
    """老论文用 Times-Italic 写变量、Symbol 写算子，只有小号角标和 Symbol 字形算公式。
    贴着公式的斜体变量、括号、等号也是式子：(T_i.C_p = T_j.C_r) 整个是一个公式，不能拆成
    一半图一半字。离公式远的斜体字母、小写的斜体词（is）仍是正文。"""
    from app.formats.pdf import _placeholderize

    def span(text, x0, x1, font="Times-Italic", size=9.0, y0=652.5):
        y1 = y0 + (11.2 if size > 8 else 7.7)
        return {"text": text, "font": font, "size": size, "flags": 2 if "Italic" in font else 0,
                "bbox": (x0, y0, x1, y1), "origin": (x0, y0 + (8.5 if size > 8 else 5.7))}

    def sub(text, x0, x1, y0=656.9):
        return span(text, x0, x1, size=6.0, y0=y0)

    roman = "Times-Roman"
    line = [
        span("form ", 30.0, 54.0, roman),
        span("(T", 54.0, 61.9), sub("i", 62.0, 63.6), span(".C", 63.6, 71.9), sub("p", 71.9, 74.9),
        span(" ", 74.9, 77.8), span("=", 77.8, 83.9), span(" ", 83.9, 86.8),
        span("T", 86.8, 91.8), sub("j", 91.8, 93.5), span(".C", 93.5, 101.8), sub("r", 101.8, 104.1),
        span(")", 104.1, 107.1),
        span(", where ", 107.1, 134.1, roman),
        span("C", 136.9, 143.0), sub("p", 142.9, 145.9),
        span(" ", 145.9, 148.0, roman), span("is", 148.0, 154.0), span(" a column, and ", 154.0, 230.0, roman),
        span("c", 230.0, 234.0), span(" is a constant.", 234.0, 290.0, roman),
    ]
    sent, formulas = _placeholderize(_block_with_spans([line], size=9.0))
    assert sent == "form {v1}, where {v2} {i}is{/i} a column, and {i}c{/i} is a constant."
    assert formulas[0]["bbox"].x0 == pytest.approx(54.0) and formulas[0]["bbox"].x1 == pytest.approx(107.1)

    y = 697.5
    display = [
        span("(PE", 54.0, 69.5, y0=y), sub("q", 69.5, 72.5, y0=y + 4.4),
        span(" \u2227", 71.8, 83.9, "Symbol", y0=y), span("PR", 83.0, 95.2, y0=y), sub("q", 95.2, 98.2, y0=y + 4.4),
        span(" \u21d2", 121.9, 137.6, "Symbol", y0=y), span("PE", 136.7, 148.9, y0=y), sub("v", 148.9, 151.8, y0=y + 4.4),
        span(" )", 151.8, 156.8, y0=y), span(" \u2227", 155.8, 168.7, "Symbol", y0=y),
    ]
    sent, formulas = _placeholderize(_block_with_spans([display], size=9.0))
    assert re.sub(r"\{v\d+\}|\s", "", sent) == ""

    # 定理正文整段斜体：贴着公式的 If 是词，不是变量
    theorem = [
        span("If", 54.0, 60.0), span(" ", 60.0, 62.5), span("G", 62.5, 69.0, "CMMI10"),
        span(" and D have capacity", 69.0, 150.0),
    ]
    sent, _ = _placeholderize(_block_with_spans([theorem], size=9.0))
    assert sent.startswith("If {v1}")

    # 序数 t-th 的 t 是数学字体，PyMuPDF 也标斜体。放回正文的序数不能再被收进旁边的公式
    ordinal = [
        span("and to predict the", 108.0, 176.3, "NimbusRomNo9L-Regu", 9.9, 206.5),
        {**span(" t", 176.3, 182.4, "CMMI10", 10.0, 206.5), "flags": 2},
        span("th", 182.4, 187.8, "NimbusRomNo9L-Regu", 7.0, 205.0),
        span(" ", 187.8, 190.8, "NimbusRomNo9L-Regu", 9.9, 202.8),
        span("coordinate given", 190.8, 255.9, "NimbusRomNo9L-Regu", 9.9, 206.5),
        {**span(" t", 255.9, 262.0, "CMMI10", 10.0, 206.5), "flags": 2},
        span(" + 1", 262.0, 279.1, "CMR10", 10.0, 206.5),
    ]
    sent, _ = _placeholderize(_block_with_spans([ordinal], size=9.9))
    assert "t-th" in sent


def test_punctuation_alone_does_not_switch_the_font_size():
    """“in PE.” 的句号排成了角标字号。单个标点换字号没有意义，标出来模型会把 {z65} 套到半句话上。"""
    from app.formats.pdf import _placeholderize

    def span(text, size=9.0, font="Times-Roman", x0=0.0):
        return {"text": text, "font": font, "size": size, "flags": 2 if "Italic" in font else 0,
                "bbox": (x0, 0, x0 + 10, size + 2), "origin": (x0, size)}

    line = [span("predicates in", x0=0), span(" PE", font="Times-Italic", x0=60), span(".", size=6.0, x0=72),
            span(" An equivalence class is a set of", x0=75), span(" small", size=7.5, x0=200), span(" words.", x0=230)]
    sent, _ = _placeholderize(_block_with_spans([line], size=9.0))
    assert "{z" not in sent.split("An equivalence")[0]
    assert "{z" in sent.split("An equivalence")[1]


def test_placeholderize_no_formula_keeps_text_identical():
    from app.formats.pdf import _placeholderize

    spans = [{"text": "Plain sentence ", "font": "TimesNewRomanPSMT", "size": 12.0},
             {"text": "without any math.", "font": "ArialMT", "size": 12.0}]
    b = _block_with_spans([spans])
    b.text = "Plain sentence without any math."
    sent, formulas = _placeholderize(b)
    assert sent == b.text and formulas == []


def _symbol_pdf(path, symbol: str):
    """正文中间夹一段 Symbol 字体的字形，提取时变成 {v1} 公式占位符。"""
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=300)
    head = "For the case where the parameter k "
    page.insert_text((40, 60), head, fontsize=11)
    x = 40 + pymupdf.get_text_length(head, fontsize=11)
    page.insert_text((x, 60), symbol, fontname="symb", fontsize=11)
    page.insert_text((x + 9 * len(symbol), 60), " is small, the greedy algorithm works well.", fontsize=11)
    doc.save(path)
    doc.close()


def test_a_display_formula_set_in_text_italic_is_left_as_it_is(tmp_path):
    """Times-Italic 写变量、Symbol 写算子的独立公式行：占位以后没有字可译，整行留原样，
    不送翻、不擦掉重排（两行的式子重排时会被当成一个高公式缩小）。"""
    src = tmp_path / "italic_math.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((40, 60), "The implication test can be split into three separate tests:", fontsize=10)
    x, y = 60.0, 90.0
    for text, font, size, dy in [
        ("(PE", "tiit", 10, 0), ("q", "tiit", 6.5, 1.6), (" \u2227 ", "symb", 10, 0), ("PR", "tiit", 10, 0),
        ("q", "tiit", 6.5, 1.6), (" \u21d2 ", "symb", 10, 0), ("PE", "tiit", 10, 0), ("v", "tiit", 6.5, 1.6),
        (")", "tiit", 10, 0),
    ]:
        page.insert_text((x, y + dy), text, fontname=font, fontsize=size)
        x += pymupdf.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text((40, 120), "The first test is called the equijoin subsumption test.", fontsize=10)
    doc.save(src)
    doc.close()

    tr = MockTranslator("zh-CN")
    sent: list[str] = []
    original = tr.translate_batch

    async def record(texts):
        sent.extend(texts)
        return await original(texts)

    tr.translate_batch = record
    [out] = run(translate_file(src, tmp_path, Runner(tr, cache=MemCache(), skip_same_lang=False), False, "zh-CN"))
    assert not any("PE" in text or "{v" in text for text in sent), sent
    words = [w[4] for w in pymupdf.open(out)[0].get_text("words") if 80 <= w[1] <= 95]
    # 原字还在原处（取词时角标并进前面的词）
    assert words[0].startswith("(PE") and not any("[zh-CN]" in w for w in words)


def test_display_equations_and_text_fractions_are_left_as_they_are(tmp_path):
    """分子分母是文字的分式、只有关系符和一两个名字的式子留原样不送翻。
    链接下划线下面接着写的正文、带等号的图注照翻。"""
    src = tmp_path / "equations.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((40, 40), "We measure the parser with labeled recall and labeled precision.", fontsize=10)
    page.insert_text((127 - pymupdf.get_text_length("labeled recall: =", fontsize=10), 78), "labeled recall: =", fontsize=10)
    numerator, denominator = "# of correct constituents in the parse", "# of total constituents in the parse"
    width = pymupdf.get_text_length(numerator, fontsize=10)
    page.insert_text((130, 73), numerator, fontsize=10)
    page.insert_text((130 + (width - pymupdf.get_text_length(denominator, fontsize=10)) / 2, 86), denominator, fontsize=10)
    page.draw_line((130, 76), (130 + width, 76), width=0.4)
    page.insert_text((150, 120), "T = argmax s(T)", fontsize=10)
    page.insert_text((40, 160), "The scores are listed in the appendix of the paper, see the", fontsize=10)
    page.draw_line((260, 162), (300, 162), width=0.4)
    page.insert_text((40, 172), "online tables for the full results of every model we trained.", fontsize=10)
    page.insert_text((40, 210), "Figure 2: Accuracy with alpha = 0.5.", fontsize=10)
    doc.save(src)
    doc.close()

    tr = MockTranslator("zh-CN")
    sent: list[str] = []
    original = tr.translate_batch

    async def record(texts):
        sent.extend(texts)
        return await original(texts)

    tr.translate_batch = record
    run(translate_file(src, tmp_path, Runner(tr, cache=MemCache(), skip_same_lang=False), False, "zh-CN"))
    assert not any("constituents" in text or "argmax" in text for text in sent), sent
    assert any("see the online tables" in text for text in sent), sent
    assert any(text.startswith("Figure 2") for text in sent), sent


def test_an_equation_in_an_algorithm_box_or_a_caption_is_translated():
    """算法框里的一行有一个词也照翻；图注带编号，等号再多也是图注。"""
    from app.formats.pdf import _equation_block

    def spans(*texts):
        return [[{"text": text, "font": "Times-Roman", "size": 10.0} for text in texts]]

    assert _equation_block(_block_with_spans(spans("T = argmax s(T)")))
    assert _equation_block(_block_with_spans([[{"text": "+ max", "font": "CMSY10", "size": 10.0}]]))
    assert not _equation_block(_block_with_spans(spans("Phoenix + Calcite")))
    assert not _equation_block(_block_with_spans(spans("Figure 2: ", "α = 0.5")))
    cell = _block_with_spans(spans("Output: ", "y = BN(x)"))
    cell.clip = (0, 0, 1, 1)
    assert not _equation_block(cell)


class _PlaceholderDropper(MockTranslator):
    """批量请求时把 {v1} 写成文字丢掉；keep_on_retry 为真时单段重翻会保留占位符。"""

    def __init__(self, *a, keep_on_retry=False, **kw):
        super().__init__(*a, **kw)
        self.keep_on_retry = keep_on_retry
        self.calls = []

    async def translate_batch(self, texts):
        self.calls.append(list(texts))
        retry = len(self.calls) > 1
        out = []
        for text in texts:
            if retry and self.keep_on_retry:
                out.append("当参数 k{v1} 较小时，贪心算法效果很好。")
            else:
                out.append("当参数 k = 2 较小时，贪心算法效果很好。")
        return out


def test_lost_placeholder_is_retried_once(tmp_path):
    """模型丢了 {v1}：单段再翻一次，补上占位符的译文写进缓存，公式照常贴回。"""
    src = tmp_path / "sym.pdf"
    _symbol_pdf(src, "abcd")
    tr = _PlaceholderDropper("zh-CN", keep_on_retry=True)
    cache = MemCache()
    r = Runner(tr, cache=cache, skip_same_lang=False)
    [out] = run(translate_file(src, tmp_path, r, False, "zh-CN"))
    assert len(tr.calls) == 2 and tr.calls[1] == tr.calls[0]
    assert "当参数" in pymupdf.open(out)[0].get_text()
    assert any("{v1}" in v for v in cache.d.values())
    assert r.info["failed"] == 0


def test_lost_small_placeholder_keeps_the_translation(tmp_path):
    """丢的只是一个字形的小公式（模型把它写成了文字）：收下译文，不退回原文。"""
    src = tmp_path / "sym.pdf"
    _symbol_pdf(src, "a")
    r = Runner(_PlaceholderDropper("zh-CN"), cache=MemCache(), skip_same_lang=False)
    [out] = run(translate_file(src, tmp_path, r, False, "zh-CN"))
    text = pymupdf.open(out)[0].get_text()
    assert "贪心算法" in text and "greedy" not in text
    assert r.info["failed"] == 0


def test_lost_large_placeholder_falls_back_and_counts_as_failed(tmp_path):
    """丢的是较长的公式：整段退回原文，并算进失败段数，任务页才会提示。"""
    src = tmp_path / "sym.pdf"
    _symbol_pdf(src, "abcd")
    r = Runner(_PlaceholderDropper("zh-CN"), cache=MemCache(), skip_same_lang=False)
    [out] = run(translate_file(src, tmp_path, r, False, "zh-CN"))
    assert "greedy" in pymupdf.open(out)[0].get_text()
    assert r.info["failed"] == 1

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
    """矢量透传端到端：两个块的公式各落各的槽位、只画一次，画在内容流里前后两段字中间，
    按内容流的顺序读（拖选、复制）时公式夹在原位。

    回归场景：块内序号命名的标记图互相去重，所有块的公式 #1 全堆到第一个块的槽位；
    公式全画在整页最后一条内容流里，拖选正文时选不中公式。
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
    assert sum(s.count(b"/FmF_a_1_png Do") for s in streams) == 1
    assert sum(s.count(b"/FmF_b_1_png Do") for s in streams) == 1
    words = [w[4] for w in page.get_text("words", sort=False)]
    assert [w for w in words if w in ("BETA", "EPSILON", "甲", "乙")] == ["BETA", "甲", "EPSILON", "乙"]

    # 译成英文时每片文字一个 HTML 盒子，公式流插在前一片文字的流后面
    out = _render_translated(src, [b1, b2], ["start \x01i\x021\x01/i\x02 one", "second \x01i\x021\x01/i\x02 two"],
                             "en", {id(b1): [f1], id(b2): [f2]}, archive)
    words = [w[4] for w in out[0].get_text("words", sort=False)]
    assert [w for w in words if w in ("start", "BETA", "one", "second", "EPSILON", "two")] == [
        "start", "BETA", "one", "second", "EPSILON", "two"]
    # 原文被 redact 后公式 glyph 由透传补回：各出现一次，槽位不同
    assert len(page.search_for("BETA")) == 1
    assert len(page.search_for("EPSILON")) == 1
    hit1, hit2 = page.search_for("BETA")[0], page.search_for("EPSILON")[0]
    # 各落各的槽位（块 rect 附近）：带缩放的 cm 平移量算错会让公式整体往下漂
    assert pymupdf.Rect(30, 30, 210, 75).contains(hit1)
    assert pymupdf.Rect(30, 80, 210, 125).contains(hit2)
    # 透传只带公式自己的字。整页原文叠在每个公式底下（哪怕在页面外），浏览器拖选会选中一大片
    words = [w[4] for w in page.get_text("words", flags=pymupdf.TEXTFLAGS_WORDS & ~pymupdf.TEXT_MEDIABOX_CLIP)]
    assert not set(words) & {"alpha", "gamma", "delta", "zeta"}
    assert words.count("BETA") == 1 and words.count("EPSILON") == 1


def test_formulas_on_a_page_filter_their_own_lines():
    """一页的公式按上下对半分，每一份只留自己所在的行，每个公式在最小的一份上过滤，
    留下的字和在整页上过滤一样。没有登记的公式也能找到盖住它的一份。"""
    from app.formats.pdf import _FormulaSources, _keep_marks, _page_form, _pdf_rect

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    for row in range(16):
        page.insert_text((40, 30 + 22 * row), f"line {row} has v{row} inside", fontsize=10)
    whole = _page_form(doc, page)
    formulas = []
    for row in range(16):
        box = page.search_for(f"v{row} ")[0] + (0, 0, -2, 0)
        formulas.append({"bbox": box, "spans": [{"bbox": tuple(box)}]})
    sources = _FormulaSources(doc, page, whole, formulas[:12])

    def shown(form):
        view = doc.new_page(width=400, height=400)
        draw = doc.get_new_xref()
        doc.update_object(draw, "<<>>")
        doc.update_stream(draw, b"q /Fm Do Q")
        doc.xref_set_key(view.xref, "Contents", f"{draw} 0 R")
        doc.xref_set_key(view.xref, "Resources", f"<< /XObject << /Fm {form} 0 R >> >>")
        return sorted(w[4] for w in doc[-1].get_text("words"))

    lines = shown(sources.source(formulas[3]))
    assert "v3" in lines and len(lines) <= 4 * 4 and "v12" not in lines
    for row in (0, 5, 11, 14):
        keep = [_pdf_rect(doc[0], formulas[row]["bbox"] + (-1, -1, 1, 1))]
        source = sources.source(formulas[row])
        assert shown(_keep_marks(doc, source, keep)) == shown(_keep_marks(doc, whole, keep)) == [f"v{row}"]
    assert sources.source(formulas[14]) == whole


def test_glyph_places_match_when_a_column_jitters_in_the_last_digits():
    """过滤后重写的内容流里数字少几位，同一列上下几行行首的字横坐标差万分之几，
    按坐标排序后先后不同。各字仍在原处，位置核对要通过；真偏了的不通过。"""
    from app.formats.pdf import _same_places

    before = [(72.0001, 500.9), (71.9999, 169.0), (72.0, 181.0), (80.0, 169.0)]
    after = [(71.9999, 500.9), (72.0001, 169.0), (71.9998, 181.0), (80.05, 169.0)]
    assert sorted(before) != sorted(after)
    assert _same_places(before, after)
    assert not _same_places(before, after[:3] + [(80.3, 169.0)])
    assert not _same_places(before, after[:3])


def test_a_formula_bitmap_is_rendered_only_when_used(monkeypatch):
    """裁公式图时只记下从哪里裁；矢量贴回失败、要贴位图时才渲染，只渲染一次。"""
    from app.formats.pdf import _FormulaImage, _formula_png

    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "x + y", fontsize=12)
    calls = []
    original = pymupdf.Page.get_pixmap
    monkeypatch.setattr(pymupdf.Page, "get_pixmap", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    formula = {"png": _FormulaImage(doc, 0, pymupdf.Rect(15, 35, 60, 55))}
    assert calls == []
    png = _formula_png(formula)
    assert png.startswith(b"\x89PNG") and _formula_png(formula) is png and len(calls) == 1
    assert _formula_png({"png": b"given"}) == b"given" and _formula_png({}) is None


def test_formula_bitmaps_rendered_in_forked_workers_read_their_own_copy(tmp_path):
    """写回的子进程是 fork 出来的，和父进程共用原文件的读写位置。几个子进程同时用父进程打开的
    文档渲染公式位图，会读到别处的字节，渲染错或报错。子进程要用自己打开的那份。"""
    import multiprocessing

    from app.formats.pdf import _FormulaImage

    src = tmp_path / "book.pdf"
    doc = pymupdf.open()
    for pno in range(12):
        page = doc.new_page(width=300, height=400)
        for row in range(20):
            page.insert_text((20, 30 + row * 18), f"page {pno} row {row} x + y = z", fontsize=11)
    doc.save(src)
    clips = [(pno, pymupdf.Rect(15, 15 + row * 18, 200, 35 + row * 18)) for pno in range(12) for row in range(0, 20, 4)]
    fresh = pymupdf.open(src)
    expected = [fresh[pno].get_pixmap(dpi=300, clip=clip).samples for pno, clip in clips]
    shared = pymupdf.open(src)
    images = [_FormulaImage(shared, pno, clip) for pno, clip in clips]
    pymupdf.TOOLS.store_shrink(100)

    def work(part: int, out) -> None:
        try:
            got = [pymupdf.Pixmap(images[k].png()).samples == expected[k] for k in range(part, len(images), 4)]
            out.put(all(got))
        except Exception:  # noqa: BLE001 - 读坏的文档会报错，算渲染错
            out.put(False)

    context = multiprocessing.get_context("fork")
    out = context.Queue()
    workers = [context.Process(target=work, args=(part, out)) for part in range(4)]
    for worker in workers:
        worker.start()
    results = [out.get(timeout=60) for _ in workers]
    for worker in workers:
        worker.join()
    assert results == [True] * 4


def test_formula_glyphs_stay_in_place_after_a_size_change_on_their_line():
    """同一行中途换过字号（小型大写），MuPDF 删掉前面的字后会把公式算偏。
    偏了就整行留下，别的行照样删掉。"""
    from app.formats.pdf import _keep_marks, _page_form, _pdf_rect

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    font = doc.get_new_xref()
    doc.update_object(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    doc.xref_set_key(page.xref, "Resources", f"<< /Font << /F1 {font} 0 R >> >>")
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(stream, (
        b"BT /F1 10 Tf 40 150 Td [(see the)]TJ /F1 6 Tf [(SMALL CAPITALS)]TJ /F1 10 Tf [( here )]TJ [(X)]TJ ET "
        b"BT /F1 10 Tf 40 130 Td [(next line)]TJ ET"
    ))
    doc.xref_set_key(page.xref, "Contents", f"{stream} 0 R")
    target = page.search_for("X")[0]
    form = _keep_marks(doc, _page_form(doc, page), [_pdf_rect(page, target + (-1, -1, 1, 1))])

    shown = doc.new_page(width=400, height=200)
    draw = doc.get_new_xref()
    doc.update_object(draw, "<<>>")
    doc.update_stream(draw, b"q /Fm Do Q")
    doc.xref_set_key(shown.xref, "Contents", f"{draw} 0 R")
    doc.xref_set_key(shown.xref, "Resources", f"<< /XObject << /Fm {form} 0 R >> >>")
    words = doc[-1].get_text("words")
    assert [w[4] for w in words if w[4] == "X"] == ["X"]
    assert abs(next(w[0] for w in words if w[4] == "X") - target.x0) < 0.2
    assert not any(w[4] in ("next", "line") for w in words)


@pytest.mark.parametrize("box", ["CropBox", "MediaBox"])
def test_cjk_lines_follow_a_shifted_page_box(tmp_path, box):
    """裁剪框或媒体框不从原点起时，逐字写出的中文仍落在原文的位置上。"""
    from app.formats.pdf import _render_translated, extract_blocks

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    for i in range(4):
        page.insert_text((120, 150 + i * 14), "The model computes the value for each input token here.", fontsize=10)
    if box == "CropBox":
        page.set_cropbox(pymupdf.Rect(80, 80, 520, 720))
    else:
        doc.xref_set_key(page.xref, "MediaBox", "[50 50 650 850]")
    src = tmp_path / "shifted.pdf"
    doc.save(src)
    source = pymupdf.open(src)[0].search_for("The model")[0]
    blocks = extract_blocks(pymupdf.open(src))
    assert len(blocks) == 1

    out = _render_translated(src, blocks, ["模型为每个输入词元计算这个值。"], "zh-CN")
    chars = _cjk_chars(out[0])
    assert chars
    assert abs(min(c[1] for c in chars) - source.x0) < 2
    assert abs(min(c[3] for c in chars) - (source.y1 - 2)) < 4


def test_formula_passthrough_follows_a_shifted_crop_box(tmp_path):
    """裁剪框偏移时，公式仍从原位裁出、贴回自己的槽位。"""
    from app.formats.pdf import TextBlock, _marker_png, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=400)
    page.insert_text((130, 150), "alpha BETA gamma", fontsize=10)
    page.set_cropbox(pymupdf.Rect(80, 100, 420, 360))
    doc.save(src)
    doc.close()

    d = pymupdf.open(src)
    rect = d[0].search_for("BETA")[0]
    formula = {"bbox": rect, "name": "a_1.png", "mid": 0,
               "w": round(rect.width, 1), "h": round(rect.height, 1), "d": 2.0,
               "text": "x", "has_img": True,
               "png": d[0].get_pixmap(dpi=150, clip=rect).tobytes("png"),
               "spans": [{"bbox": tuple(rect), "text": "x", "size": 10,
                          "origin": (rect.x0, rect.y1 - 2)}]}
    line = pymupdf.Rect(rect.x0 - 45, rect.y0 - 2, rect.x1 + 60, rect.y1 + 3)
    d.close()
    archive = pymupdf.Archive()
    archive.add(_marker_png(0), "ph_a_1.png")
    block = TextBlock(page=0, rect=line, line_rects=[line], text="alpha BETA gamma",
                      size=10, color="#000", bold=False)
    out = _render_translated(src, [block], ["译文 \x01i\x021\x01/i\x02 甲"], "zh-CN",
                             {id(block): [formula]}, archive)
    hits = out[0].search_for("BETA")
    assert len(hits) == 1
    assert pymupdf.Rect(line.x0 - 10, line.y0 - 10, line.x1 + 10, line.y1 + 10).contains(hits[0])


def test_formula_clip_ignores_tall_bitmap_glyph_boxes(tmp_path):
    """下一行是没有 Unicode 的点阵字时，它报的字体框比墨迹高，会伸进公式的下半截。
    这种框不能拿来挖洞，否则公式的下半截被裁掉。"""
    from app.formats.pdf import TextBlock, _marker_png, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    procs = []
    for _ in range(10):
        xref = doc.get_new_xref()
        doc.update_object(xref, "<<>>")
        doc.update_stream(xref, b"500 0 0 0 400 600 d1 0 0 400 600 re f")
        procs.append(xref)
    glyphs = " ".join(f"/a{i}" for i in range(10))
    charprocs = " ".join(f"/a{i} {xref} 0 R" for i, xref in enumerate(procs))
    font = doc.get_new_xref()
    doc.update_object(font, (
        "<< /Type /Font /Subtype /Type3 /Name /BitmapF /FontBBox [0 -200 500 1800] "
        f"/FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << {charprocs} >> "
        f"/Encoding << /Type /Encoding /Differences [65 {glyphs}] >> "
        f"/FirstChar 65 /LastChar 74 /Widths [{' '.join(['500'] * 10)}] /Resources << >> >>"
    ))
    doc.xref_set_key(page.xref, "Resources",
                     f"<< /Font << /T3 {font} 0 R /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>")
    codes = "".join(f"{65 + i % 10:02X}" for i in range(20))
    content = doc.get_new_xref()
    doc.update_object(content, "<<>>")
    doc.update_stream(content, (
        "BT /F1 10 Tf 50 200 Td (alpha BETA gamma) Tj ET\n"
        f"BT /T3 12 Tf 50 182 Td <{codes}> Tj ET"
    ).encode())
    doc.xref_set_key(page.xref, "Contents", f"{content} 0 R")
    doc.save(src)
    doc.close()

    d = pymupdf.open(src)
    rect = d[0].search_for("BETA")[0]
    bitmap = next(span for block in d[0].get_text("dict")["blocks"] for line in block.get("lines", [])
                  for span in line["spans"] if span["font"] == "BitmapF")
    assert bitmap["bbox"][1] < rect.y1  # 点阵字的框确实伸进了 BETA 的下半截
    def ink_share(pix):
        return sum(1 for v in pix.samples if v < 128) / len(pix.samples)

    ink = ink_share(d[0].get_pixmap(dpi=150, clip=rect, colorspace=pymupdf.csGRAY))
    formula = {"bbox": rect, "name": "a_1.png", "mid": 0,
               "w": round(rect.width, 1), "h": round(rect.height, 1), "d": 2.0,
               "text": "x", "has_img": True,
               "png": d[0].get_pixmap(dpi=150, clip=rect).tobytes("png"),
               "spans": [{"bbox": tuple(rect), "text": "x", "size": 10,
                          "origin": (rect.x0, rect.y1 - 2)}]}
    line = pymupdf.Rect(45, rect.y0 - 2, 200, rect.y1 + 3)
    d.close()
    archive = pymupdf.Archive()
    archive.add(_marker_png(0), "ph_a_1.png")
    block = TextBlock(page=0, rect=line, line_rects=[line], text="alpha BETA gamma",
                      size=10, color="#000", bold=False)
    out = _render_translated(src, [block], ["译文 \x01i\x021\x01/i\x02 甲"], "zh-CN",
                             {id(block): [formula]}, archive)
    slot = out[0].search_for("BETA")[0]
    # 译文行按 0.88 倍字号排，公式跟着缩小，按墨迹占比比较。裁掉下半截时只剩一半左右
    drawn = ink_share(out[0].get_pixmap(dpi=150, clip=slot, colorspace=pymupdf.csGRAY))
    assert drawn >= ink * 0.65

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

    # 角标后面接着箭头，同一个 run 里有正常大小的字：字母仍是角标的底
    arrow = _block_with_spans([[
        span("Set ", "Times-Roman", 10, 300, 456.4, x1=322.4, y1=466.9),
        span("N", "Times-Roman", 10, 327.5, 456.4, x1=335.2, y1=466.9),
        {**span("inf", "CMR7", 7, 335.2, 457.5, x1=344.6, y1=462.7), "math": True},
        {**span("BN", "CMR7", 7, 335.2, 464.2, x1=345.6, y1=469.4), "math": True},
        span(" ←", "CMSY10", 10, 346.6, 459.4, x1=359.2, y1=469.4),
        span(" N now", "Times-Roman", 10, 359.2, 456.4, x1=390, y1=466.9),
    ]], size=10.0)
    sent_a, formulas_a = _placeholderize(arrow)
    assert sent_a.startswith("Set {v1}") and len(formulas_a) == 1

    equals = _block_with_spans([[
        span("E ", "TimesNewRomanPSMT", 10, 0, 50),
        span("=", "TimesNewRomanPSMT", 10, 14, 50, x1=22),
        span(" mc", "TimesNewRomanPSMT", 10, 24, 50),
    ]], size=10.0)
    assert "=" in _placeholderize(equals)[0]


def test_a_script_on_the_next_line_does_not_join_the_letter_above_it():
    """下一行行首 H^enc 的 enc 字框碰到上一行的 X，基线低了大半行：它是 H 的上标，
    不是 X 的下标。X 是正文时不收成角标的底，X 是公式时也不并成一个式子。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, x0, y0, x1, y1, base):
        return {"text": text, "font": font, "size": size, "flags": 0,
                "bbox": (x0, y0, x1, y1), "origin": (x0, base)}

    for x_font in ("CMSSBX10", "CMBX10"):
        block = _block_with_spans([
            [span("is", "NimbusRomNo9L-Regu", 10, 160, 464, 167, 476, 473.3),
             span(" X", x_font, 10, 167, 466, 177, 476, 473.3),
             span(", in cross attention the input is", "NimbusRomNo9L-Regu", 10, 177, 464, 421, 476, 473.3)],
            [span("H", "CMSSBX10", 10, 160, 477, 168, 487, 485.3),
             span("enc", "NimbusRomNo9L-ReguItal", 7.4, 168, 475, 178, 484, 481.7),
             span(" is of shape", "NimbusRomNo9L-Regu", 10, 178, 476, 226, 488, 485.3)],
        ], size=10.0)
        sent, formulas = _placeholderize(block)
        assert not any("X" in f["text"] and "enc" in f["text"] for f in formulas), x_font
        assert any(f["text"] == "enc" for f in formulas), x_font
        assert "H{v" in sent or "H{/b}{v" in sent, sent


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

    # 行末的 ∪ 按 CMSY 的字体框伸到下一行，和下一行行首的 {γ} 交叠：仍是折行的两片
    def ink(item, top, bottom):
        item["ink"] = (item["bbox"][0], top, item["bbox"][2], bottom)
        return item

    union = _block_with_spans([
        [span("the parameters ", "NimbusRomNo9L-Regu", 10, 40, 430.4, 510, 443.7),
         ink(span("Θ", "CMR10", 10, 518.8, 433.4, 526.5, 443.4), 433.8, 440.9),
         span(" ", "CMSY10", 10, 526.5, 433.4, 533.4, 450.4),
         ink(span("∪", "CMSY10", 10, 533.4, 433.4, 540, 450.4), 434.2, 440.9)],
        [ink(span("{", "CMSY10", 10, 40, 443.9, 45, 460.9), 445.3, 455.3),
         ink(span("γ", "CMMI10", 10, 45, 445.3, 51, 455.3), 448.0, 455.3),
         ink(span("}", "CMSY10", 10, 51, 443.9, 56, 460.9), 445.3, 455.3),
         span(" to train", "NimbusRomNo9L-Regu", 10, 58, 442.4, 100, 455.7)],
    ], size=10.0)
    sent_u, formulas_u = _placeholderize(union)
    assert sent_u.count("{v") == 1
    assert len(formulas_u[0]["parts"]) == 2
    assert formulas_u[0]["parts"][0]["bbox"].x0 >= 518
    assert formulas_u[0]["parts"][1]["bbox"].x1 <= 56

    # 行内分式 1/√dk：分子分母都是小字号，分子那一行排在块的最后，仍是同一个式子
    fraction = _block_with_spans([
        [span("we scale the dot products by ", "Times-Roman", 10, 300, 606, 438, 616),
         span("√", "CMSY7", 7, 440.7, 606.2, 447, 619.3),
         span("dk", "CMMI7", 7, 447, 606.2, 455.2, 619.3),
         span(".", "Times-Roman", 10, 456, 606, 459, 616)],
        [span("1", "CMR7", 7, 446.3, 603.6, 450.2, 610.6)],
    ], size=10.0)
    sent_f, formulas_f = _placeholderize(fraction)
    assert sent_f.count("{v") == 1 and sent_f.endswith("{v1}.")
    assert formulas_f[0]["bbox"].y0 < 604

    # 相邻两行行首的脚注记号也是小字号、上下对齐，隔着一行，各是各的
    notes = _block_with_spans([
        [span("†", "CMSY6", 6, 108, 680, 112, 686), span("Work performed at A.", "Times-Roman", 9, 113, 679, 220, 689)],
        [span("‡", "CMSY6", 6, 108, 691, 112, 697), span("Work performed at B.", "Times-Roman", 9, 113, 690, 220, 700)],
    ], size=9.0)
    sent_n, _ = _placeholderize(notes)
    assert sent_n.count("{v") == 2

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

    # 式子收尾的「 + 1」「 = 20」CMR 排成一个 span，Latin Modern 的也一样
    for roman in ("CMR10", "LMRoman10-Regular"):
        tail = _block_with_spans([[
            span("takes only ", roman, 10, 0, 50, 46),
            span("A", "CMMI10", 10, 47, 50, 54),
            span(" + 1", roman, 10, 54, 50, 73),
            span(" values, so ", roman, 10, 77, 50, 120),
            span("A", "CMMI10", 10, 121, 50, 128),
            span(" = 20", roman, 10, 128, 50, 151),
            span(" draws", roman, 10, 154, 50, 180),
        ]], size=10.0)
        sent_a, formulas_a = _placeholderize(tail)
        assert sent_a == "takes only {v1} values, so {v2} draws"
        assert [f["text"].replace(" ", "") for f in formulas_a] == ["A+1", "A=20"]

    # 小数点是 CMMI 的「.」，后半个数和正文排在一个 span 里：「= 0」不收，免得把 0.31 拆开
    decimal = _block_with_spans([[
        span("gives a slightly higher", "CMR10", 10, 0, 50, 100),
        span(" B/T", "CMMI10", 10, 100, 50, 120),
        span(" ", "CMR10", 10, 120, 50, 123),
        span("= 0", "CMR10", 10, 123, 50, 141),
        span(".", "CMMI10", 10, 141, 50, 144),
        span("31 likely", "CMR10", 10, 144, 50, 180),
    ]], size=10.0)
    sent_dec, _ = _placeholderize(decimal)
    assert sent_dec == "gives a slightly higher{v1} = 0.31 likely"

    # 单独的 CMMI「>」贴着后面的「= 10¹¹」，一起收进公式
    bound = _block_with_spans([[
        span("of stellar mass", "CMR10", 10, 0, 50, 70),
        span(" >", "CMMI10", 10, 70, 50, 81),
        span("= 10", "CMR10", 10, 81, 50, 101),
        span("11", "CMR7", 7, 101, 48, 108, 55),
        span(" in the nearby", "CMR10", 10, 108, 50, 160),
    ]], size=10.0)
    sent_b, formulas_b = _placeholderize(bound)
    assert sent_b == "of stellar mass{v1} in the nearby"
    assert formulas_b[0]["text"].replace(" ", "").startswith(">=10")

    # Times 排的式子：斜体 L 后面贴着正体的下标 CE，L 照样收进 ∂ 的式子
    partial = _block_with_spans([[
        span("the derivative ", "Times-Roman", 10, 0, 50, 60),
        span("∂", "CMSY10", 10, 60, 50, 65),
        span("L", "Times-Italic", 10, 65, 50, 71),
        span("CE", "Times-Roman", 10, 71, 50, 84),
        span(" is", "Times-Roman", 10, 84, 50, 96),
    ]], size=10.0)
    _, formulas_p = _placeholderize(partial)
    assert formulas_p[0]["text"].replace(" ", "").startswith("∂L")


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
        [{**span("^", "CMR10", 10, 41, 42, x1=45, y1=48), "origin": (41, 56)}],  # 帽子是独立 dict 行，基线抬高一点
    ], size=10.0)
    sent, formulas = _placeholderize(b, 0)
    assert "^" not in sent
    assert len(formulas) == 1
    assert formulas[0]["bbox"].y0 <= 42  # 符头顶部包进裁剪区域
    # 短音符 M̆、小波浪 Ẽ 也一样
    for accent in ("˘", "˜"):
        b1 = _block_with_spans([
            [span("costs at least ", "TimesNewRomanPSMT", 10, 0, 50),
             span("M", "CMMI10", 10, 40, 50, x1=48, y1=60),
             span(" over", "TimesNewRomanPSMT", 10, 60, 50)],
            [{**span(accent, "CMR10", 10, 41, 42, x1=46, y1=48), "origin": (41, 56)}],
        ], size=10.0)
        sent1, formulas1 = _placeholderize(b1, 0)
        assert accent not in sent1 and len(formulas1) == 1
    # 远离公式的 ^ 不吸收（如正文里确实有个符号）
    b2 = _block_with_spans([
        [span("y", "CMMI10", 10, 0, 50, x1=8, y1=60),
         span(" then ^ alone", "TimesNewRomanPSMT", 10, 100, 50)],
    ], size=10.0)
    sent2, _ = _placeholderize(b2, 0)
    assert "^" in sent2
    # 上一行 naïve 的两点不算到下一行的公式上
    b3 = _block_with_spans([
        [span("usual (na", "TimesNewRomanPSMT", 10, 0, 40, x1=40, y1=50),
         {**span("\u00a8", "TimesNewRomanPSMT", 10, 40, 40, x1=44, y1=50), "origin": (40, 48)},
         span("\u0131ve)", "TimesNewRomanPSMT", 10, 40, 40, x1=60, y1=50)],
        [span("\u2207", "CMSY10", 10, 38, 50, x1=48, y1=62)],
    ], size=10.0)
    sent3, _ = _placeholderize(b3, 0)
    assert "\u00a8" in sent3



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


def test_paragraph_split_by_a_caption_beside_it_is_joined():
    """段落绕着图排，图注在阅读顺序里夹在两半中间。图注在上一半下面、下一半旁边，
    两半接成一段。表格同一行右边的格子和上一格并排，下一行的格子不接上去。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, y0, x1, y1):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000", bold=False)

    head = blk("From the myth about Pygmalion and the statue he fell in love", 160, 241, 493, 265)
    caption = blk("Michelangelo's Moses (1515)", 380, 432, 486, 443)
    tail = blk("with, to Mary Shelley's Gothic novel about Frankenstein", 160, 265, 370, 337)
    merged = _merge_continuations([head, caption, tail])
    assert [b.text[:4] for b in merged] == ["From", "Mich"]
    assert merged[0].text.endswith("Frankenstein")

    left = blk("a claim made by the speaker", 160, 100, 300, 112)
    right = blk("a question by the speaker", 320, 100, 460, 112)
    below = blk("a weak suggestion or listing", 160, 114, 300, 126)
    assert len(_merge_continuations([left, right, below])) == 3


def test_a_larger_heading_below_a_title_is_not_its_wrapped_line():
    """小节名下面紧接着的大一号的是下一节的标题，不是小节名换行的第二行。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, y0, x1, y1, size):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=True,
                         body_size=10)

    title = blk("8.4.4 Limitations of Preference-Based Learning", 160.2, 515.9, 410.3, 531.4, 12)
    section = blk("8.5 Summary", 95.4, 539.6, 201.1, 560.4, 17.2)
    assert len(_merge_continuations([title, section])) == 2


def test_translation_wraps_around_the_figure_its_paragraph_wraps_around():
    """原文段落前几行在图左边让开、图下面恢复整宽。译文碰到图高度的行同样让开，图下面的行写满。
    只有一行在旁边的不算绕排（引文最后一行旁边的出处）。"""
    from app.formats.pdf import TextBlock, _place_plain, _wrap_holes

    image = pymupdf.Rect(380, 268, 493, 428)
    caption = pymupdf.Rect(380, 432, 486, 443)
    narrow = [pymupdf.Rect(160, y, 370, y + 12) for y in range(385, 445, 12)]
    wide = [pymupdf.Rect(160, y, 493, y + 12) for y in range(445, 517, 12)]
    lines = narrow + wide
    block = TextBlock(page=0, rect=pymupdf.Rect(160, 385, 493, 517), line_rects=lines,
                      text="x", size=10, color="#000000", bold=False)
    holes = _wrap_holes(block, [image, caption, pymupdf.Rect(131, 471, 146, 480)])
    assert sorted(map(tuple, holes)) == sorted([tuple(image), tuple(caption)])

    quote = [pymupdf.Rect(167, y, 475, y + 12) for y in (210, 222, 234)] + [pymupdf.Rect(167, 246, 203, 258)]
    quoted = TextBlock(page=0, rect=pymupdf.Rect(167, 210, 475, 258), line_rects=quote,
                       text="x", size=10, color="#000000", bold=False)
    assert _wrap_holes(quoted, [pymupdf.Rect(411, 246, 475, 258)]) == []

    # 跨页接过来的半段按上一页的栏宽写。图在块的框外、写入范围以内，同样让开
    figure = pymupdf.Rect(347, 103, 493, 253)
    tail_lines = [pymupdf.Rect(160, y, 337, y + 12) for y in range(110, 170, 12)]
    tail = TextBlock(page=0, rect=pymupdf.Rect(160, 110, 337, 170), line_rects=tail_lines,
                     text="x", size=10, color="#000000", bold=False)
    assert _wrap_holes(tail, [figure]) == []
    assert _wrap_holes(tail, [figure], pymupdf.Rect(160, 110, 493, 168)) == [figure]

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _place_plain(page, "这一卷讲的是这个激动人心的构想如何实现、带来什么影响。" * 8, [], pymupdf.Rect(160, 386, 495, 521),
                 8.8, 10, True, "left", "", "#000000", "normal", [], None, None, False, holes)
    spans = [span for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])
             for span in line["spans"] if span["text"].strip()]
    assert spans
    for span in spans:
        box = pymupdf.Rect(span["bbox"])
        assert not box.intersects(image) and not box.intersects(caption), span["text"]
    assert any(pymupdf.Rect(span["bbox"]).x1 > 450 and span["bbox"][1] > 443 for span in spans)


def test_a_running_head_does_not_run_into_the_first_line_below():
    """页眉停在 et al. 上，像没写完的半句。认成页眉的块不和下面小写起头的正文相接。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blocks():
        head = TextBlock(page=1, rect=pymupdf.Rect(51, 42, 344, 54), line_rects=[pymupdf.Rect(51, 42, 344, 54)],
                         text="2 Sarkar et al.", size=10.0, color="#000", bold=False)
        lines = [pymupdf.Rect(51, 67 + 12.5 * i, 292, 77 + 12.5 * i) for i in range(3)]
        body = TextBlock(page=1, rect=pymupdf.Rect(51, 67, 292, 102), line_rects=lines,
                         text="rounded by an outer, extended low surface brightness envelope. This came into light.",
                         size=10.0, color="#000", bold=False)
        return head, body

    head, body = blocks()
    assert len(_merge_continuations([head, body])) == 1
    head, body = blocks()
    assert len(_merge_continuations([head, body], margins={id(head)})) == 2


def test_a_word_cut_after_a_run_in_label_continues_at_the_column_edge():
    """行内标签后面的半行从标签右边起笔，行尾断开的词（temper-）在下一行栏边接着写。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blocks(x1=522):
        half = TextBlock(page=0, rect=pymupdf.Rect(163, 620, x1, 631), line_rects=[pymupdf.Rect(163, 620, x1, 631)],
                         text="Models are served through OpenRouter (OpenRouter, 2026) at temper-",
                         size=10.9, color="#000", bold=False)
        lines = [pymupdf.Rect(90, 634 + 13 * i, 522, 645 + 13 * i) for i in range(3)]
        below = TextBlock(page=0, rect=pymupdf.Rect(90, 634, 522, 672), line_rects=lines,
                          text="ature 0 with fixed seeds: all scenarios and every memory design.",
                          size=10.9, color="#000", bold=False)
        return [half, below]

    assert len(_merge_continuations(blocks())) == 1
    assert len(_merge_continuations(blocks(x1=400))) == 2


def test_a_line_inside_an_open_bracket_runs_on_below():
    """括号还没合上就换行，下一行大写或数字起头、在第一句里合上括号，仍是同一句。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, rows=2):
        lines = [pymupdf.Rect(72, y + 24 * i, 540, y + 24 * i + 12) for i in range(rows)]
        return TextBlock(page=0, rect=pymupdf.Rect(72, y, 540, y + 24 * rows - 12), line_rects=lines,
                         text=text, size=12.0, color="#000", bold=False)

    head = "the influence of the prior decreases with sample size (see e.g."
    assert len(_merge_continuations([blk(head, 100), blk("Ghosal et al., 2000; Walker, 2004). The density follows.", 148)])) == 1
    assert len(_merge_continuations([blk(head, 100), blk("Ghosal et al. Then the second sentence starts here.", 148)])) == 2


def test_a_line_after_an_inline_formula_wraps_to_the_column_edge():
    """一行里左边是式子、右边接着写正文，这半行撑到右缘、没写完，下一行回到栏边小写接着写。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blocks(x1=521, text="This no longer agrees with the asymptotic distribution of the MLE under"):
        half = TextBlock(page=0, rect=pymupdf.Rect(154, 414, x1, 425), line_rects=[pymupdf.Rect(154, 414, x1, 425)],
                         text=text, size=10.9, color="#000", bold=False)
        lines = [pymupdf.Rect(89, 434 + 22 * i, 522, 445 + 22 * i) for i in range(3)]
        below = TextBlock(page=0, rect=pymupdf.Rect(89, 434, 522, 489), line_rects=lines,
                          text="model misspecification, which although still centered at the same point.",
                          size=10.9, color="#000", bold=False)
        return [half, below]

    assert len(_merge_continuations(blocks())) == 1
    assert len(_merge_continuations(blocks(x1=400))) == 2
    assert len(_merge_continuations(blocks(text="This no longer agrees with the asymptotic distribution."))) == 2


def test_layout_model_regions_join_a_paragraph_cut_by_a_formula():
    """开了版面模型时，前后两个单元落在模型框出的同一段正文里、切口上又有式子，就是被式子切开的同一段。
    没有模型、上一块句末收住而下一块大写起头（一串列表）、上一块是一行作者名、
    切口上没有式子的（幻灯片上一条条要点）、题词和下面的出处，都不并。"""
    from app.formats.pdf import TextBlock, _region_joins
    from app.formats.pdf_roles import HeuristicLayout

    def blk(text, y, rows=3, x0=72):
        lines = [pymupdf.Rect(x0, y + 12 * i, 540, y + 12 * i + 10) for i in range(rows)]
        return TextBlock(page=0, rect=pymupdf.Rect(x0, y, 540, lines[-1].y1), line_rects=lines,
                         text=text, size=10.0, color="#000", bold=False)

    box = [pymupdf.Rect(70, 500, 545, 600)]

    def joined(head, tail, regions=box):
        formula = blk("Σ x = 1", 540, rows=1, x0=250)
        units = [[head], [formula], [tail]]
        out = _region_joins(units, {id(formula)}, set(), lambda pno: regions)
        return len(out) == 2 and out[0] == [head, tail]

    head = blk("We fix weights α with α > 0 and consider the measure on the line. For a Borel set E,", 505)
    tail = blk("1E is the indicator function of that set, and we let the map act on it.", 562, rows=2)
    assert joined(head, tail)
    assert not joined(head, tail, regions=[])
    near = blk("1E is the indicator function of that set, and we let the map act on it.", 541, rows=2)
    assert not joined(head, near)
    near.span_lines = [[{"text": "1E", "font": "CMMI10", "flags": 0, "size": 10},
                        {"text": " is the indicator function of that set,", "font": "Times", "flags": 0, "size": 10}], []]
    assert joined(head, near)
    ended = blk("We show that the dimension remains similar over time under fixed conditions.", 505)
    assert not joined(ended, blk("We demonstrate that aging changes the dimension of the data.", 552, rows=2))
    authors = blk("Jie Yu, Jianyu Zhang, Zian Pei, Xue Shi, Mohamed Ali", 505, rows=1, x0=190)
    assert not joined(authors, blk("Yumei Liu, Xin Jiang, Quanying Liu, Yi Guo", 552, rows=1, x0=190))
    quote = blk("“The most important attributes of a vector in 3-space are Location, Location, Location”", 505, rows=2)
    assert not joined(quote, blk("Randall Munroe, the hover from https://xkcd.com/2358/ on vectors", 552, rows=2))
    assert HeuristicLayout().text_regions(0) == []


def test_a_full_line_wraps_into_a_capitalized_next_line():
    """段落缩进的首行和页边旁注挤在一块里被单拆出来。它排到右缘、没有句末标点，下一行回到左缘、
    以专名大写起头（RNNs, one where…），仍是同一句。末行没排满、或者有句号的不接。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, x0=160.2, x1=493.2, rows=1):
        lines = [pymupdf.Rect(x0 if i == 0 else 160.2, y + 12 * i, x1, y + 12 * i + 12) for i in range(rows)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    head = "A bidirectional RNN (Schuster and Paliwal, 1997) combines two independent"
    tail = "RNNs, one where the input is processed from the start to the end, and the other"
    assert len(_merge_continuations([blk(head, 610, x0=175.1), blk(tail, 622, rows=2)])) == 1
    assert len(_merge_continuations([blk(head, 610, x0=175.1, x1=420), blk(tail, 622, rows=2)])) == 2
    assert len(_merge_continuations([blk(head + ".", 610, x0=175.1), blk(tail, 622, rows=2)])) == 2


def test_a_document_that_spaces_paragraphs_joins_blocks_at_the_normal_line_gap():
    """新段齐平起笔、上面多空出一截的文档里，按正常行距紧贴着、上一行排满的下一块是同一段，
    上一行恰好在句末断开也一样（双倍行距的稿子一段常切成几块）。空出那一截的另起一段。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    size, pitch = 10.9, 21.0

    def blk(text, y, rows=3, last_x1=522):
        lines = [pymupdf.Rect(89, y + pitch * i, 522 if i < rows - 1 else last_x1, y + pitch * i + 11)
                 for i in range(rows)]
        rect = pymupdf.Rect(89, y, 522, lines[-1].y1)
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=size, color="#000",
                         bold=False, line_gap=10 / size)

    def page(gap):
        blocks, y = [], 60.0
        for _ in range(4):
            blocks.append(blk("This paragraph discusses the prior and the likelihood in turn.", y, last_x1=300))
            y = blocks[-1].rect.y1 + 16
        blocks.append(blk("These assumptions are almost never satisfied in practice.", y))
        blocks.append(blk("As a result, such Bayesian analyses are conducted with care.", blocks[-1].rect.y1 + gap,
                          last_x1=300))
        return blocks

    assert len(_merge_continuations(page(gap=10))) == 5
    assert len(_merge_continuations(page(gap=16))) == 6


def test_a_full_line_in_a_document_that_indents_paragraphs_runs_on_below():
    """段落首行缩进的文档里，排满一整行、恰好在句末断开的一行，下面齐平起笔的一行是同一段的续行。
    带编号的标签、隔着一行的、自己往下悬挂缩进的（续行另成一块也算）、上一行是表格几格的，都另起。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, rows):
        # 一行可以是几截：[(x0, x1), ...]
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12)
                 for i, row in enumerate(rows) for x0, x1 in (row if isinstance(row, list) else [row])]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    head = "The first line opens indented and the second fills the measure to the margin."

    # 并段会改动上一块，每次重新排
    def plan(text, y=316, rows=((72, 400),), tail=(), last=((72, 540),)):
        paragraphs = [blk(body, 100 + 48 * i, [(87, 540), (72, 540), (72, 540), (72, 300)]) for i in range(4)]
        return _merge_continuations([*paragraphs, blk(head, 292, [(87, 540), *last]), blk(text, y, list(rows)), *tail])

    assert len(plan("The next sentence goes on in the same paragraph.")) == 5
    assert len(plan("EXAMPLE 2.1 The next sentence goes on in the same paragraph.")) == 6
    assert len(plan("The next sentence goes on in the same paragraph.", y=328)) == 6
    entry = "The next entry runs to the margin and hangs its continuation"
    assert len(plan(entry + " below it", rows=((72, 540), (90, 300)))) == 6
    merged = plan(entry + ",", rows=((72, 540),), tail=[blk("under its first line.", 328, [(90, 300)])])
    assert [block.text[:14] for block in merged[-2:]] == ["The first line", "The next entry"]
    assert len(plan("The next sentence goes on in the same paragraph.", last=([(72, 200), (300, 540)],))) == 6
    # 只有一行、在句末收住，下面紧贴着缩进的新段：是这一段的末行，不是悬挂缩进
    merged = plan("The paragraph ends on this line.", rows=((72, 300),), tail=[blk(body, 328, [(87, 540), (72, 300)])])
    assert len(merged) == 6 and merged[-2].text.endswith("on this line.")


def test_theorem_labels_do_not_count_as_flush_paragraphs():
    """定理、证明的标签本来就齐平起笔，不算段落不缩进的证据。正文一行的宽度按整行量，行内式子切开的几截并起来。"""
    from app.formats.pdf import TextBlock, _body_measure, _indents_paragraphs

    def blk(text, y, rows):
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12)
                 for i, row in enumerate(rows) for x0, x1 in (row if isinstance(row, list) else [row])]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    labels = ["Lemma 3.4. Let U be a span.", "Proposition 2.4. Let T be bounded.", "2.1. Bosonic Fourier transform.",
              "Theorem 3.3. Suppose that T is linear."]
    blocks = []
    for i in range(3):
        blocks.append(blk(body, 100 + 96 * i, [(87, 540), (72, 300)]))
        blocks.append(blk(body, 124 + 96 * i, [(87, 540), (72, 300)]))
        blocks.append(blk(labels[i], 148 + 96 * i, [(72, 300)]))
    blocks.append(blk(labels[3], 388, [(72, 300)]))
    assert _indents_paragraphs(blocks)
    split = blk(body, 500, [[(72, 200), (210, 540)], [(72, 300), (310, 540)], [(72, 540)], (72, 300)])
    assert _body_measure([split]) == 468.0


def test_one_query_per_line_in_one_block_is_cut_into_lines():
    """PyMuPDF 把一行一句的查询归成一块。每行都早早换行（下一行的第一个词放得下），按行拆开。
    排满的段落照旧是一段。"""
    from app.formats.pdf import _cut_line_list

    def item(text, y, x1):
        rect = pymupdf.Rect(167.1, y, x1, y + 10.8)
        return (text, rect, [{"text": text, "size": 9.0, "bbox": tuple(rect)}], None)

    queries = [
        item("can you tell me about any good cantonese restaurants close by", 356.2, 390.1),
        item("tell me about chez panisse", 367.1, 261.2),
        item("i'm looking for a good place to eat breakfast", 378.1, 326.4),
        item("when is caffe venezia open during the day", 389.1, 317.9),
    ]
    assert _cut_line_list([0, 1, 2, 3], queries, [item[1] for item in queries]) == [[0], [1], [2], [3]]
    # 绕图、表格里的段落比栏窄，参差地折行，下一行的第一个词都放不进最长的那一行
    paragraph = [
        item("Policies, processes,", 100, 240), item("procedures, and", 111, 225),
        item("practices across the", 122, 238), item("organization related", 133, 236),
    ]
    assert _cut_line_list([0, 1, 2, 3], paragraph, [item[1] for item in paragraph]) == [[0, 1, 2, 3]]
    # 绕着右边的图排的几行止于图边，下面一行排满全栏
    around = [
        item("the first full-sentence speech synthesizer, shown partially to the right. His", 100, 475),
        item("device consisted of a bellows to simulate the lungs, a rub-", 111, 380),
        item("ber mouthpiece and a nose aperture, a", 122, 380.5),
        item("reed to simulate the vocal folds, var-", 133, 379.6),
    ]
    assert _cut_line_list([0, 1, 2, 3], around, [item[1] for item in around]) == [[0, 1, 2, 3]]


def test_rows_of_a_table_or_a_list_with_one_opening_are_cut_into_lines():
    """同一行分成几条的先并成一行。各行同一处另起一列（行名和特征），或者每行同一个词起头
    （特征模板），哪怕长短差不多，也按行拆开。"""
    from app.formats.pdf import _cut_line_list

    def item(text, x0, y, x1, size=9.96):
        rect = pymupdf.Rect(x0, y, x1, y + 11.9)
        return (text, rect, [{"text": text, "size": size, "bbox": tuple(rect)}], None)

    def row(parts, y):
        spans = [{"text": text, "size": 9.96, "bbox": (x0, y, x1, y + 11.9)} for text, x0, x1 in parts]
        return (" ".join(p[0] for p in parts), pymupdf.Rect(parts[0][1], y, parts[-1][2], y + 11.9), spans, None)

    table = [
        item("hen", 178.1, 414.7, 192.5),
        row([("+female,", 212.4, 250.1), ("+chicken,", 259.5, 302.9), ("+adult", 311.8, 343.2)], 416.0),
        row([("rooster", 178.1, 209.4), ("-female,", 212.4, 250.1), ("+chicken,", 259.5, 302.9),
             ("+adult", 311.8, 343.2)], 426.7),
        item("chick", 178.1, 438.6, 198.8),
        row([("+chicken,", 212.4, 255.8), ("-adult", 264.7, 296.1)], 439.9),
    ]
    rects = [item[1] for item in table]
    assert _cut_line_list([0, 1, 2, 3, 4], table, rects) == [[0, 1], [2], [3, 4]]
    templates = [
        item("xi contains a particular prefix (perhaps from all prefixes of length ≤2)", 185.7, 589.9, 465.9),
        item("xi contains a particular suffix (perhaps from all suffixes of length ≤2)", 185.7, 601.9, 464.3),
        item("xi’s word shape", 185.7, 613.8, 247.3),
        item("xi’s short word shape", 185.7, 625.8, 269.8),
    ]
    rects = [item[1] for item in templates]
    assert _cut_line_list([0, 1, 2, 3], templates, rects) == [[0], [1], [2], [3]]


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
    # 粗体名字后面跟着公式字体的星号角标，整行仍是粗体
    author = line("Jimmy Lei Ba∗", 40, 120, 100, 132, bold=True)
    author.span_lines = [[{"text": "Jimmy Lei Ba", "font": "NimbusRomNo9L-Medi", "flags": 20},
                          {"text": "∗", "font": "CMSY7", "flags": 7}]]
    assert _with_source_bold(author, "Jimmy Lei Ba{v1}") == "{b}Jimmy Lei Ba{v1}{/b}"


def _reference_same_line_merge(blocks):
    """改成按页合并之前的写法。每并一次就从头再找。"""
    from app.formats.pdf import _absorb_block, _continues_fragment, _sits_on_last_line

    out = list(blocks)
    changed = True
    while changed:
        changed = False
        for index, host in enumerate(out):
            for other_index, guest in enumerate(out):
                if other_index == index or guest.page != host.page:
                    continue
                if not _sits_on_last_line(host, guest) and not _continues_fragment(host, guest):
                    continue
                _absorb_block(host, guest)
                del out[other_index]
                changed = True
                break
            if changed:
                break
    return out


def _random_line_blocks(seed: int):
    import random

    from app.formats.pdf import TextBlock

    rng = random.Random(seed)
    texts = ["Alpha beta gamma delta.", "alpha beta gamma delta", "and the model keeps going",
             "Results are shown here.", "with more words after it"]
    blocks = []
    for _ in range(rng.randint(4, 30)):
        page = rng.choice([0, 1, 2])
        row = rng.randint(0, 8)
        start = rng.choice([40.0, 40.0, 130.0, 210.0])
        count = rng.choice([1, 1, 2, 3])
        rects = []
        for line in range(count):
            y0 = 100 + (row + line) * 10.5
            x0 = start if line == 0 else 40.0
            x1 = 300.0 if line < count - 1 else rng.choice([120.0, 200.0, 300.0])
            rects.append(pymupdf.Rect(x0, y0, max(x1, x0 + 20), y0 + 9))
        union = pymupdf.Rect(rects[0])
        for rect in rects[1:]:
            union |= rect
        blocks.append(TextBlock(page=page, rect=union, line_rects=rects, text=rng.choice(texts),
                                size=9, color="#000", bold=False))
    return blocks


def test_same_line_merge_by_page_matches_the_rescan():
    """按页分开、并完接着找，结果和每次从头重扫一样。"""
    import copy

    from app.formats.pdf import TextBlock, _merge_same_line_sentences

    def shape(blocks):
        return [(b.page, b.text, tuple(b.rect), [tuple(r) for r in b.line_rects]) for b in blocks]

    merges = 0
    for seed in range(300):
        blocks = _random_line_blocks(seed)
        expected = _reference_same_line_merge(copy.deepcopy(blocks))
        assert shape(_merge_same_line_sentences(copy.deepcopy(blocks))) == shape(expected), seed
        merges += len(blocks) - len(expected)
    assert merges > 300

    # 中间那块先接上下一行、回到栏边，才能并进前面那段。并完要回头看前面的块。
    def para(text, boxes):
        rects = [pymupdf.Rect(*box) for box in boxes]
        union = pymupdf.Rect(rects[0])
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=0, rect=union, line_rects=rects, text=text, size=9, color="#000", bold=False)

    chain = [
        para("Alpha beta gamma delta.", [(40, 100, 300, 109), (40, 110.5, 150, 119.5)]),
        para("Results are shown here and", [(155, 110.5, 300, 119.5), (200, 121, 300, 130)]),
        para("with more words after it.", [(40, 131.5, 300, 140.5)]),
    ]
    expected = _reference_same_line_merge(copy.deepcopy(chain))
    assert len(expected) == 1
    assert shape(_merge_same_line_sentences(copy.deepcopy(chain))) == shape(expected)


def test_a_displayed_formula_does_not_swallow_the_prose_below_it():
    """单独排的式子两边都空着，分子分母、分号各是一截，最后一截落在右边。下面回到栏边的正文另起一段。"""
    from app.formats.pdf import TextBlock, _continues_fragment

    lines = [pymupdf.Rect(117, 145, 173, 158), pymupdf.Rect(178, 145, 253, 165), pymupdf.Rect(254, 157, 257, 164)]
    host = TextBlock(
        page=0, rect=lines[0] | lines[1] | lines[2], line_rects=lines,
        text="dNorm(x, X)/dx and dNorm(x, X)/dX ;", size=10, color="#000", bold=False,
    )
    guest_lines = [pymupdf.Rect(72, 175, 301, 188), pymupdf.Rect(72, 187, 301, 200)]
    guest = TextBlock(
        page=0, rect=guest_lines[0] | guest_lines[1], line_rects=guest_lines,
        text="ignoring the latter term would lead to the explosion described above.",
        size=10, color="#000", bold=False,
    )
    assert not _continues_fragment(host, guest)


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


def test_a_column_continues_below_the_table_at_the_top_of_the_next_column():
    """右栏顶上是一张表，表注在表下面。表头、行名都是短行，左栏没写完的句子接在表注下面的正文上。
    栏顶是排满的段落时照常接上。"""
    from app.formats.pdf import TextBlock, _cross_page_units
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, y, x1, rows=1):
        rect = pymupdf.Rect(x0, y, x1, y + 10 * rows)
        lines = [pymupdf.Rect(x0, y + 10 * k, x1, y + 10 * k + 9) for k in range(rows)]
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10, color="#000", bold=False)

    two = PageGeometry(600, (Column(72, 290), Column(307, 526)), (), reads_in_columns=True)
    left = [blk("Following standard practice, we formulate this as a tagging task", 72, 600, 290, rows=6),
            blk("include the maximal context and do not use a CRF", 72, 670, 290, rows=4)]
    table = [blk("System", 317, 67, 343), blk("Fine-tuning approach", 317, 118, 394),
             blk("Feature-based approach (BERT)", 317, 153, 450)]
    caption = blk("Table 7: CoNLL-2003 Named Entity Recognition results.", 307, 236, 526, rows=4)
    tail = blk("layer in the output. We use the representation of the first sub-token", 307, 290, 526, rows=3)
    later = blk("To ablate the fine-tuning approach, we apply the feature-based approach", 307, 330, 526, rows=5)
    units = _cross_page_units([*left, *table, caption, tail, later], 10.0, geometries=[two])
    assert [unit for unit in units if len(unit) == 2] == [[left[1], tail]]


def test_a_column_of_one_paragraph_still_continues_into_the_next_column():
    """左栏上面是缩进排的摘要、下面只有一整段，没有两块同一左缘，这一整段也算一栏，接到右栏顶。
    右栏顶居中排在第二位作者名下面的单位不是栏顶。页上没有一栏是两块对齐的（幻灯片），不另起栏。"""
    from app.formats.pdf import TextBlock, _cross_page_units

    def blk(text, x0, y, x1, rows=1, page=0):
        rect = pymupdf.Rect(x0, y, x1, y + 10 * rows)
        lines = [pymupdf.Rect(x0, y + 10 * k, x1, y + 10 * k + 9) for k in range(rows)]
        return TextBlock(page=page, rect=rect, line_rects=lines, text=text, size=10, color="#000", bold=False)

    abstract = blk("Oddly satisfying videos can evoke strong viewer responses in many ways.", 83, 240, 280, rows=20)
    intro = blk("Accordingly, Oddly Satisfying Video Emotion Anal-", 63, 530, 290, rows=17)
    right = [blk("ysis (OSVEA) requires models to move beyond recognizing objects.", 307, 220, 526, rows=20),
             blk("To address the lack of a dataset, we construct one and annotate it.", 307, 430, 526, rows=20)]
    units = _cross_page_units([abstract, intro, *right], 10.0, geometries=[])
    assert [unit for unit in units if len(unit) == 2] == [[intro, right[0]]]

    author = blk("Shixiang Zhu", 376, 162, 442)
    office = blk("Carnegie Mellon University", 350, 174, 468)
    left = [blk("Generative conformal prediction builds uncertainty sets from samples.", 83, 240, 280, rows=30),
            blk("allowing the set to reflect complex or multimodal predictive distributions", 63, 623, 290, rows=10)]
    tail = blk("(Wang et al., 2023b). Probabilistic conformal prediction samples outputs.", 307, 209, 526, rows=6)
    later = blk("This efficiency, however, depends on drawing enough samples per input.", 307, 275, 526, rows=20)
    units = _cross_page_units([author, office, *left, tail, later], 10.0, geometries=[])
    assert [unit for unit in units if len(unit) == 2] == [[left[1], tail]]

    slide = blk("If we need probabilities we can do one exp at the end p1 = exp(log", 115, 509, 700, rows=3)
    title = blk("This avoids underflow from multiplying many small numbers", 105, 279, 700, rows=3)
    heading = blk("Larger ngrams", 115, 25, 400, page=1)
    bullet = blk("4-grams, 5-grams", 114, 150, 400, page=1)
    more = blk("Large datasets of large n-grams have been released", 114, 216, 700, rows=2, page=1)
    units = _cross_page_units([title, slide, heading, bullet, more], 10.0, geometries=[], halt_ids={id(heading)})
    assert all(len(unit) == 1 for unit in units)


def test_column_tail_drawn_before_its_host_keeps_the_paragraph(tmp_path):
    """内容流先画右栏时，右栏顶的续块排在左栏宿主前面。整段译文仍要写出来。"""
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)

    def para(x, y, lines):
        for i, line in enumerate(lines):
            page.insert_text((x, y + i * 12), line, fontsize=10)

    para(320, 80, ["continues the gamma sentence on the right side", "with more words that end with a stop."])
    para(320, 200, ["Delta paragraph in the right column is here", "and finishes with a final stop."])
    para(50, 80, ["Alpha paragraph text that fills the left column", "and keeps going for a few more words here.",
                  "It ends with a full stop at the end."])
    para(50, 200, ["Beta paragraph text that also fills the column", "and keeps going for several more words.",
                   "It also ends properly with a stop."])
    para(50, 320, ["Gamma begins and we write a short line"])
    src = tmp_path / "right_first.pdf"
    doc.save(src)

    [out] = run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))
    text = " ".join(pymupdf.open(out)[0].get_text().split())
    assert "Gamma begins" in text
    assert "continues the gamma sentence" in text


def test_rotated_pages_lay_out_like_unrotated_ones(tmp_path):
    """旋转 90 度的页面，跨页段落的译文排法和不旋转时一样：不提前停笔，字号不缩。"""
    from app.formats.pdf import translate_pdf

    words = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
             "tempor incididunt ut labore et dolore magna aliqua ").split()
    layouts = []
    for rotation in (0, 90):
        doc = pymupdf.open()
        index = 0
        for pno in range(2):
            page = doc.new_page(width=595, height=842)
            writer = pymupdf.TextWriter(page.rect)
            for y in range(60, 790, 12):
                line: list[str] = []
                while len(" ".join(line)) < 80:
                    line.append(words[index % len(words)])
                    index += 1
                text = " ".join(line)
                writer.append((50, y), ("The " + text) if pno == 0 and y == 60 else text, fontsize=10)
            writer.write_text(page)
            page.set_rotation(rotation)
        src = tmp_path / f"rot{rotation}.pdf"
        doc.save(src)
        [out] = run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))
        layouts.append([
            sorted(
                (round(span["bbox"][1]), round(span["size"], 1))
                for blk in page.get_text("dict")["blocks"] if blk.get("type") == 0
                for ln in blk["lines"] for span in ln["spans"]
            )
            for page in pymupdf.open(out)
        ])
    plain, rotated = layouts
    assert rotated == plain


def test_page_end_does_not_take_a_block_already_paired_to_the_next_column():
    """下一页左栏的块已经接到右栏顶时，上一页页末不能再把它拿走。否则右栏那块不在任何单元里。"""
    from app.formats.pdf import TextBlock, _cross_page_units
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, y, page, size=9.0):
        rect = pymupdf.Rect(x0, y, x0 + 180, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text,
                         size=size, color="#000", bold=False)

    wide = PageGeometry(600, (Column(40, 560),), ())
    end = blk("The optimizer then rewrites the", 40, 700, 0)
    host = blk("query into a plan and it keeps the", 40, 300, 1)
    tail = blk("cheapest alternative for each group.", 300, 60, 1)
    later = blk("Each group holds equivalent expressions.", 300, 200, 1)
    small = blk("Notes about the memo structure here", 40, 600, 1, size=8.0)
    blocks = [end, host, tail, later, small]
    units = _cross_page_units(blocks, 9.0, geometries=[wide, wide], halt_ids=set())
    assert sorted(id(block) for unit in units for block in unit) == sorted(id(block) for block in blocks)
    assert [host, tail] in units


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
    # 行尾断词（fine-）后半截只会是小写，换栏碰上大写起头的是表头或标题。
    broken = blk("score. Without TriviaQA fine-", 300, 700)
    units = _cross_page_units([host2, broken, tail, tail2], 9.0, geometries=[wide, wide])
    assert all(len(unit) == 1 for unit in units)
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


def test_line_end_colon_inside_a_sentence_merges():
    """行尾冒号上下两行都撑满是句中换行，要并成一段；下一行是短的例句、表头就不并。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, x0=142.0, x1=475.0):
        rect = pymupdf.Rect(x0, y, x1, y + 11)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    wrapped = [
        blk("The most common way to use this n-gram hierarchy is called interpolation:", 250, x0=157),
        blk("computing a new probability by interpolating (weighting and combining) the tri-", 262),
    ]
    assert len(_merge_continuations(wrapped)) == 1
    example = [
        blk("Consider the language of certain sheep, which consists of strings like:", 250),
        blk("baa! baaa! baaaa!", 262, x0=200, x1=280),
    ]
    assert len(_merge_continuations(example)) == 2
    intro = [
        blk("the steps are:", 250, x1=210),
        blk("compute the counts and normalize them so that they sum to one over the", 262),
    ]
    assert len(_merge_continuations(intro)) == 2


def test_slide_sub_item_is_not_a_title_fragment():
    """幻灯片上一级列表项后面的下一级项字号大、左缘也不同，不是标题的第二行。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, y0, y1, size):
        rect = pymupdf.Rect(x0, y0, 682, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("• Pushes as much work as possible onto the cluster for parallel execution", 115.2, 176.1, 214.0, 18),
        blk("– Metadata repository", 79.2, 215.3, 240.9, 22),
    ]
    assert [b.text for b in _merge_continuations(blocks)] == [
        "• Pushes as much work as possible onto the cluster for parallel execution",
        "– Metadata repository",
    ]


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


def test_affiliation_under_an_author_name_splits():
    """题名页上作者名下面小一号的单位自成一行，只有两个词也单独成段。
    行首星号比后面的字大一号时，按多数字的字号比，下一行的同字号续行不拆。"""
    from app.formats.pdf import _split_lines

    author = [
        _line("Daniel Jurafsky", 247, 355, 333.4, size=17.2),
        _line("Stanford University", 245, 357, 353.6, size=14.3),
    ]
    assert _split_lines(author) == [[0], [1]]

    def item(spans, x0, x1, y0):
        text = "".join(text for text, _ in spans)
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 14), [{"text": text, "size": size} for text, size in spans])

    note = [
        item([("*", 15), ("also what standalone", 12), (" ", 15)], 61, 186, 155),
        item([("Phoenix compiler ", 12)], 61, 157, 173),
        item([("would generate.", 12)], 61, 148, 187),
    ]
    assert _split_lines(note) == [[0, 1, 2]]


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


def test_a_chapter_number_across_a_rule_is_not_joined_to_the_title():
    """章首的两位数章号和标题挨得近，按词距会连成一句，整行重写时压过中间的竖线。隔着竖线就拆开。"""
    from app.formats.pdf import _cut_at_posts, _split_lines

    items = [
        ("10", pymupdf.Rect(98.4, 132.0, 158.2, 204.1), [{"text": "10", "size": 59.8}], None),
        ("Interpretability", pymupdf.Rect(167.3, 137.3, 331.9, 169.6),
         [{"text": "Interpretability", "size": 24.8}], None),
    ]
    post = pymupdf.Rect(161.2, 110.8, 162.4, 200.8)
    groups = [part for group in _split_lines(items) for part in _cut_at_posts(group, items, [post])]
    assert groups == [[0], [1]]
    # 竖线不在两截之间（表格外框），照常相连
    assert _cut_at_posts([0, 1], items, [pymupdf.Rect(90, 110, 91, 200)]) == [[0, 1]]


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
    # 原文编号带点的，点也留着
    assert norm("1. Exploited the sortedness of join input", "1. 利用了连接输入的有序性") == "1. 利用了连接输入的有序性"


def test_section_and_item_numbers_are_not_sent_for_translation():
    """节号、条目号摘下来不送翻，写回时原样接回去。正文句首的数不摘。"""
    from app.formats.pdf import _peel_number, _with_label

    assert _peel_number("B.3.2. DeepSeek-R1 Cold Start", "p") == ("B.3.2.", "DeepSeek-R1 Cold Start")
    assert _peel_number("4 Experiments", "h2") == ("4", "Experiments")
    assert _peel_number("{b}3{/b} {b}Method{/b}", "h2") == ("{b}3{/b}", "{b}Method{/b}")
    # 粗体从编号延续到文字：编号关上，文字重开
    assert _peel_number("{b}17.2.1 The Encoder{/b} ENCODEC {b}model{/b}", "h3") == (
        "{b}17.2.1{/b}", "{b}The Encoder{/b} ENCODEC {b}model{/b}")
    assert _peel_number("1.{i} Intractability{/i}: the case where", "p") == ("1.", "{i} Intractability{/i}: the case where")
    assert _peel_number("(a) Let u be the view of least cost.", "p") == ("(a)", "Let u be the view of least cost.")
    assert _peel_number("2 samples were drawn from each model and we report the mean.", "p")[0] == ""
    assert _peel_number("1.5 times faster than the baseline, a large gain.", "p")[0] == ""
    assert _peel_number("I. Goodfellow proposed adversarial nets in 2014.", "p")[0] == ""
    assert _peel_number("3.17", "p")[0] == ""
    assert _with_label("B.3.2.", "DeepSeek-R1 冷启动") == "B.3.2. DeepSeek-R1 冷启动"
    assert _with_label("", "实验") == "实验"


def test_a_bitmap_glyph_without_its_own_box_is_kept_by_its_extent():
    """点阵字不报自己的框（宽为 0），只有整套字体的上下界，算出的中心离墨迹半行。
    横向看起笔点、纵向看上下界和保留框交叠，上标仍留下；起笔点在框外的照旧删掉。"""
    from types import SimpleNamespace as NS

    from app.formats.pdf import _GlyphKeeper

    keeper = _GlyphKeeper([(484.9, 240.1, 490.5, 244.7)])
    trm = NS(a=0.1, b=0.0, c=0.0, d=-0.1, e=485.5, f=249.8)
    ctm = NS(a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0)
    font_box = NS(x0=0.0, y0=-65.0, x1=0.0, y1=86.0)
    assert keeper.text_filter(None, None, 0, trm, ctm, font_box) == 0
    beside = NS(a=0.1, b=0.0, c=0.0, d=-0.1, e=495.3, f=249.8)
    assert keeper.text_filter(None, None, 0, beside, ctm, font_box) == 1
    # 报了自己的框的字仍按中心算
    own_box = NS(x0=0.0, y0=-65.0, x1=60.0, y1=126.0)
    assert keeper.text_filter(None, None, 0, trm, ctm, own_box) == 1


def test_stray_braces_around_marks_are_dropped():
    """模型在标记外面多套的花括号去掉。原文本来有花括号（集合 {k=1}）的不动。"""
    from app.formats.pdf import _drop_stray_braces

    sent = "called a {b}fragment{/b}. {c0000ff}Fox Tree{/c} showed {v1}"
    out = _drop_stray_braces("叫做{b}断裂{/b}}。{{c0000ff}Fox Tree{/c}}证明了{v1}", sent)
    assert out == "叫做{b}断裂{/b}。{c0000ff}Fox Tree{/c}证明了{v1}"
    assert _drop_stray_braces("集合 {k=1} 是", "the set {k=1} is") == "集合 {k=1} 是"


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
    assert _heading_number_for("h2", "8. RELATED WORK", "八, 相关工作") == "8. 相关工作"
    assert _heading_number_for("h2", "9. SUMMARY", "九, 总结") == "9. 总结"
    assert _heading_number_for("h2", "10. REFERENCES", "十, 参考文献") == "10. 参考文献"
    assert _heading_number_for("h2", "8.1 Query Optimization Foundations", "八点一 查询优化基础") == "8.1 查询优化基础"
    assert _heading_number_for("h2", "2.2 SQL on Hadoop", "第2.2节 Hadoop 上的 SQL") == "2.2 Hadoop 上的 SQL"
    assert _heading_number_for("h2", "1.3 Related Work", "一.三. 相关工作") == "1.3 相关工作"
    assert _heading_number_for("h2", "2.5 Composite Lattices", "二．五 复合格") == "2.5 复合格"
    # 三级节号，各级之间可能空一格
    assert _heading_number_for("h2", "3.1.3 Dealing with scale", "第三章第一节第三小节 处理规模问题") == "3.1.3 处理规模问题"
    assert _heading_number_for("h2", "10.1.2 Contextual Embeddings", "第十章 第一节 第二小节 上下文嵌入") == "10.1.2 上下文嵌入"


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
    # 隔壁栏图上和这一行同高的刻度数字在行尾左边很远，也不是这一行的
    tick = pymupdf.Rect(20, 96, 28, 106)
    _donate_punct_rects(prev, [tick])
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


def test_split_filled_lines_fills_the_last_lines_of_the_page():
    """页底剩原文一两行的高度。译文按原行距放不下，压紧行距、末行的行距伸出去，仍留在本页。"""
    from app.formats.pdf import _filled_bottom, _split_filled_lines

    text = "人工智能领域的一项名为发展的研究计划的成员，并发表了一篇论文。" * 3
    head, rest = _split_filled_lines(text, [], em=8.8, block_size=10, width=300, cjk=True, y0=700, limit=712)
    assert head and "\n" not in head
    assert rest
    head, rest = _split_filled_lines(text, [], em=8.8, block_size=10, width=300, cjk=True, y0=696.5, limit=719.5)
    assert head.count("\n") == 1
    assert rest
    bottom = _filled_bottom(700, 1, 8.8, True, 712, 792)
    assert bottom - 700 >= 8.8 * 1.45 + 8.8 * 0.3 - 0.1
    assert bottom <= 712 + 0.45 * 8.8 + 0.01
    # 写入框裁在原文正文下沿，limit 就是这条线时，末行的行距伸不出去：少留一行，不让整段缩小
    text = "上界不匹配，因此必须强制执行谓词，这可以简化为一个等值谓词。" * 8
    roomy, _ = _split_filled_lines(text, [], em=7.92, block_size=9, width=240, cjk=True, y0=664.1, limit=722.5)
    capped, _ = _split_filled_lines(
        text, [], em=7.92, block_size=9, width=240, cjk=True, y0=664.1, limit=722.5, page_bottom=722.5,
    )
    assert roomy.count("\n") == 5
    assert capped.count("\n") == 4


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
    # 标题后另起一段，开头就是标题里的词
    head_src, para_src = "4.7.2 Applying softmax in logistic regression", "When we apply softmax for logistic regression"
    head_t, para_t = "4.7.2 在逻辑回归中应用 softmax", "在逻辑回归中应用 softmax 时, 输入与 sigmoid 一样"
    assert _strip_boundary_echo(head_src, para_src, head_t, para_t) == para_t


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


def test_a_loose_mixed_line_stretches_word_spaces_before_the_chinese():
    """中英混排的行余量大时，先拉开词间空格，汉字之间只稍微拉开，不拉成「常 识 推 理」。"""
    from app.formats.pdf_align import justify_extras

    text = "常识推理(0-shot): Hellaswag [28], Winogrande [21], PIQA [4],"
    extras = justify_extras(text, 60.0, 10.0)
    inside = extras[text.index("识")]
    space = extras[text.index(" ")]
    assert 0 < inside <= 1.0 + 1e-9
    # 一处空格前后两个缝合起来算，最多再拉开四分之一个字。余下的留在行尾
    space += extras[text.index(" ") - 1]
    assert space > inside * 2 and space <= 2.5 + 1e-9
    assert sum(extras) < 60.0
    # 没有空格的中文行照旧平均摊
    plain = "我们在此提出一种采样机制已经写完了"
    extras = justify_extras(plain, 16.0, 10.0)
    assert len({round(extra, 6) for extra in extras}) == 1


def test_snake_case_identifiers_are_one_word():
    """l_partkey 这样的标识符是一个词：不从下划线处折行，两端对齐也不在下划线两边拉开。"""
    from app.formats.pdf_align import can_stretch
    from app.formats.pdf_flow import tokenize

    assert tokenize("{l_partkey, o_custkey}") == ["{", "l_partkey", ",", " ", "o_custkey", "}"]
    assert not can_stretch("l", "_") and not can_stretch("_", "p")
    assert can_stretch(",", "中") and can_stretch(",", " ")
    # 网址、邮箱连着写，@ . / 两边也不拉开，也不从中间折行
    email = "jimmy@psi.utoronto.ca"
    assert not any(can_stretch(a, b) for a, b in zip(email, email[1:]))
    assert tokenize("多伦多大学 jimmy@psi.utoronto.ca 获取")[6] == email
    assert tokenize("见 http://www.github.com/goodfeli/adversarial).")[2:] == [
        "http://www.github.com/goodfeli/adversarial", ")", "."]


def test_a_centered_title_breaks_after_a_clause_mark():
    """居中的副标题放不下一行时在逗号后面断开，不留“识别”两个字单独一行；一行放得下就不断。"""
    from app.formats.pdf import _clause_lines
    from app.formats.pdf_typeset import typeset_lines

    text = "借助语言模型介绍自然语言处理, 计算语言学和语音识别"

    def broken(width):
        lines = _clause_lines(text, [], 10.0, 10.0, width, typeset_lines(text, 10.0, width, {}))
        return ["".join(piece.text for piece in line.pieces) for line in lines]

    assert broken(230.0) == ["借助语言模型介绍自然语言处理,", "计算语言学和语音识别"]
    assert broken(260.0) == [text]
    # 整段粗体的作者名单在逗号后面断开，后一行仍是粗体
    names = "{b}Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed{/b}"
    lines = _clause_lines(names, [], 10.0, 10.0, 300.0, typeset_lines(names, 10.0, 300.0, {}))
    texts = ["".join(piece.text for piece in line.pieces) for line in lines]
    assert len(texts) == 2 and all(line.startswith("{b}") and line.endswith("{/b}") for line in texts), texts


def test_an_opening_bracket_does_not_end_a_line():
    """左括号不留在行尾，跟下一个词一起折到下一行（“{” 单独挂在行尾、“o_custkey}” 在下一行）。"""
    from app.formats.pdf_flow import break_lines

    lines = break_lines("视图范围: {l_partkey}, {o_custkey}", 10.0, 64.0, {})
    assert len(lines) > 1
    assert all(not line.rstrip().endswith(("{", "(", "[", "（")) for line in lines), lines
    assert "".join(lines).replace(" ", "") == "视图范围:{l_partkey},{o_custkey}"
    # 左引号同样跟着下一个词走，右引号不起行
    quoted = break_lines("这里, Victoria 有多种不同的指称方式: “Victoria Chen”“Megabucks Banking 的首席财务官”", 9.5, 300.0, {})
    assert all(not line.rstrip().endswith("“") and not line.startswith("”") for line in quoted), quoted


def test_caption_word_in_bold_cm_is_not_a_formula():
    from app.formats.pdf import _is_formula_span

    assert _is_formula_span({"font": "CMBX8", "text": "of", "size": 8.0}, 9.0) is False
    assert _is_formula_span({"font": "CMBX10", "text": "x", "size": 10.0}, 10.0) is True


def test_garbled_figure_label_is_recognized():
    from app.formats.pdf import _garbled_label

    assert _garbled_label("(c)$Bo8om(up$sta.s.cs$deriva.on$")
    assert not _garbled_label("Figure 2: Interaction of Orca with database system")
    assert not _garbled_label("Table 7 | Training costs, assuming the rental price is $2 per GPU hour.")


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


def test_write_helpers_only_need_the_blocks_on_their_page():
    """写入只传同页的块。各个版面判断拿整本书的块和只拿本页的块，结果一样。"""
    import random

    from app.formats.pdf import (
        TextBlock, _affiliation_write_rects, _beside_body, _body_first_starts, _fit_page_margins, _room_below,
    )

    rng = random.Random(5)
    texts = ["∗University of Somewhere", "†Institute of Elsewhere", "Memo.", "(2) Statistics Derivation.",
             "The body line runs on and on", "and the model keeps going.", "Results are shown here."]
    blocks = []
    for _ in range(90):
        x0 = rng.choice([40.0, 60.0, 120.0, 200.0, 320.0])
        y0 = rng.choice([30.0, 60.0, 100.0, 300.0, 700.0, 760.0]) + rng.randint(0, 3) * 12
        lines = [pymupdf.Rect(x0, y0 + i * 12, x0 + rng.choice([60, 140, 240]), y0 + i * 12 + 10)
                 for i in range(rng.choice([1, 1, 2]))]
        union = pymupdf.Rect(lines[0])
        for line in lines[1:]:
            union |= line
        blocks.append(TextBlock(page=rng.choice([0, 1, 2]), rect=union, line_rects=lines,
                                text=rng.choice(texts), size=rng.choice([8.0, 10.0]), color="#000",
                                bold=rng.random() < 0.4))

    def fixed(pno, text, lines, bold=False):
        rects = [pymupdf.Rect(*line) for line in lines]
        union = pymupdf.Rect(rects[0])
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=pno, rect=union, line_rects=rects, text=text, size=10, color="#000", bold=bold)

    # 每页同一位置放并排单位、侧标和它旁边的正文。判断漏了同页条件时，别页的块会混进来。
    for pno in range(3):
        blocks += [
            fixed(pno, "∗University of Somewhere", [(60, 40, 180, 50)]),
            fixed(pno, "†Institute of Elsewhere", [(190, 40, 300, 50)]),
            fixed(pno, "Memo.", [(40, 100, 70, 110)], bold=True),
            fixed(pno, "This body line sits beside the label.", [(72, 100, 300, 110), (40, 112, 300, 122)]),
        ]
    page = pymupdf.open().new_page(width=600, height=800)
    found = {"stacked": 0, "starts": 0, "beside": 0}
    for pno in range(3):
        own = [block for block in blocks if block.page == pno]
        ids = {id(block) for block in own}
        whole = _affiliation_write_rects(blocks)
        stacked = _affiliation_write_rects(own)
        assert stacked == {key: rect for key, rect in whole.items() if key in ids}
        found["stacked"] += len(stacked)
        items = [(block, "译文" + block.text) for block in own]
        for cjk in (True, False):
            starts = _body_first_starts(own, items, cjk, None)
            assert starts == _body_first_starts(blocks, items, cjk, None)
            found["starts"] += len(starts)
            for block in own:
                rect = pymupdf.Rect(block.rect.x0, block.rect.y0, block.rect.x1 + 2, block.rect.y1 + 4)
                assert _fit_page_margins(block, rect, own, 800, cjk) == _fit_page_margins(block, rect, blocks, 800, cjk)
        for block in own:
            beside = _beside_body(block, own)
            assert beside is _beside_body(block, blocks)
            found["beside"] += beside is not None
            args = (page, block, block.rect.y0, block.rect.x0, block.rect.x1)
            assert _room_below(*args, own) == _room_below(*args, blocks)
    assert all(found.values())


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
    """侧标让出的第一行按窄宽断完后，剩下的不带换行，写入时按续行的框重排。
    预先断好的换行会在续行让开图时留下半截短行。"""
    from app.formats.pdf import _split_own_lines

    text = (
        "Orca 包含一个可扩展框架, 用于基于形式化的属性规范描述查询需求和计划特征。"
        "属性有不同类型, 包括逻辑属性。"
    )
    head, rest = _split_own_lines(text, [], 7.92, 9.0, 120.0)
    assert head and rest and "\n" not in rest
    assert "框架用于" not in head + rest
    glued = (head + rest.replace("\n", "")).replace(" ", "")
    assert glued == text.replace(" ", "")
    # 斜体跨过第一行：第一行末尾补了收尾标记，剩下的只接着写后半，斜体在开头重开
    styled = "前面普通{i}斜体跨过第一行的末尾一直延续到下一行{/i}后面普通"
    head, rest = _split_own_lines(styled, [], 10.0, 10.0, 80.0)
    assert head == "前面普通{i}斜体跨过{/i}"
    assert rest == "{i}第一行的末尾一直延续到下一行{/i}后面普通"


def test_a_word_too_wide_for_the_shared_line_moves_whole_to_the_next_line():
    """同一行剩下的半截只有一个短词宽，放不下 BERT 时整个词留给续行，不切成 BER / T。"""
    from app.formats.pdf import _split_own_lines

    assert _split_own_lines("BERT 和 GPT 在所有任务上的表现", [], 8.8, 10.0, 20.0) == ("", "BERT 和 GPT 在所有任务上的表现")
    assert _split_own_lines("{b}BERT{/b} 在所有任务上", [], 8.8, 10.0, 20.0)[0] == ""
    head, rest = _split_own_lines("在所有任务上 BERT 都更好", [], 8.8, 10.0, 20.0)
    assert head == "在所" and rest.startswith("有任务")


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


def test_cross_column_host_stops_above_kept_formula(tmp_path):
    """栏底段落接到另一栏时，译文停在下面留着原字的行间公式上方，余下的写到续块。

    行间公式在提取时丢掉，不在块列表里；只按块算余量，会一直写到本页正文下沿，压在公式上。
    """
    from app.formats.pdf import SPILL_TAIL, TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((150, 186), "dl/dTheta", fontsize=10)
    doc.save(src)
    doc.close()
    formula_top = pymupdf.open(src)[0].get_text("words")[0][1]

    def block(rect, text):
        rect = pymupdf.Rect(rect)
        lines = [pymupdf.Rect(rect.x0, y, rect.x1, y + 10) for y in range(int(rect.y0), int(rect.y1) - 9, 12)]
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10, color="#000000", bold=False)

    host = block((72, 100, 300, 160), "where x is the training data set and the mini-batch is used to approximate it")
    tail = block((311, 60, 540, 90), "Using mini-batches of examples is helpful in several ways.")
    lower = block((311, 300, 540, 400), "While stochastic gradient is simple and effective, it needs tuning.")
    paragraph = "其中训练数据集按小批量逐步处理，每一步都用小批量来近似损失函数的梯度。" * 6
    out = _render_translated(
        src, [host, tail, lower], [paragraph, SPILL_TAIL, "随机梯度下降简单而有效，但需要仔细调节。"], "zh-CN",
        continuations={id(host): tail},
    )
    chars = [
        pymupdf.Rect(ch["bbox"])
        for b in out[0].get_text("rawdict")["blocks"] if b.get("type") == 0
        for line in b["lines"] for span in line["spans"] for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    left = [rect for rect in chars if rect.x1 < 306]
    assert left and max(rect.y1 for rect in left) < formula_top
    assert any(rect.x0 > 306 and rect.y1 < 100 for rect in chars)
    assert "dl/dTheta" in out[0].get_text()


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


def test_last_short_line_keeps_body_size(tmp_path):
    """页末一两行的行框余量留着。裁掉之后中文装不下一行，字号不能因此比上面的正文小。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=612, height=792)
    doc.save(src)
    doc.close()
    above = pymupdf.Rect(72, 640.5, 540, 651.4)
    last = pymupdf.Rect(90, 667.6, 276.9, 678.5)
    heading = pymupdf.Rect(72, 640.0, 280, 652.0)
    pair = pymupdf.Rect(72, 685.5, 540, 710.0)
    blocks = [
        TextBlock(page=0, rect=above, line_rects=[above],
                  text="Copying data on every read and write access slows the DBMS.",
                  size=10.9, color="#000000", bold=False),
        TextBlock(page=0, rect=last, line_rects=[last],
                  text="Default libc malloc is slow. Never use it.",
                  size=10.9, color="#000000", bold=False),
        TextBlock(page=1, rect=heading, line_rects=[heading],
                  text="Optimistic Concurrency Control.",
                  size=12, color="#000000", bold=True),
        TextBlock(
            page=1, rect=pair,
            line_rects=[
                pymupdf.Rect(72, 685.5, 540, 696.4),
                pymupdf.Rect(72, 699.1, 265, 710.0),
            ],
            text="Store all changes in a private workspace and merge them at commit time.",
            size=10.9, color="#000000", bold=False,
        ),
    ]
    out = _render_translated(
        src, blocks,
        [
            "每次读和写都会复制数据，因此数据库会变慢。",
            "默认的 libc malloc 速度很慢。永远不要使用它。",
            "乐观并发控制。",
            "将所有更改先放进私有工作区，提交时再检查冲突并合并。这个做法最早在一九八一年提出，后来被广泛使用。",
        ],
        "zh-CN",
    )

    def sizes(page: pymupdf.Page, y0: float) -> list[float]:
        found = [
            span["size"]
            for block in page.get_text("dict")["blocks"] if block.get("type") == 0
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip() and span["bbox"][1] >= y0
        ]
        assert found
        return found

    body = 10.9 * 0.88
    assert min(sizes(out[0], 660)) >= body - 0.05
    assert min(sizes(out[1], 670)) >= body - 0.05
    bottoms = [
        ch["bbox"][3]
        for page in out
        for block in page.get_text("rawdict")["blocks"] if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        for ch in span["chars"]
        if "\u4e00" <= ch["c"] <= "\u9fff"
    ]
    assert bottoms
    assert max(bottoms) < 730


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


def test_block_of_typewriter_lines_is_code_and_inline_names_are_not():
    """整块几乎都是等宽字（SQL）留原样。句子里夹着几个等宽属性名的照翻。"""
    from app.formats.pdf import _code_block

    def span(text, font):
        return {"text": text, "font": font, "size": 10.0, "bbox": (0, 0, 10, 10), "origin": (0, 8)}

    sql = _block_with_spans([
        [span("SELECT Part, Customer, SUM(SP) ", "Courier"), span("AS ", "Times-Roman"), span("TotalSales", "Courier")],
        [span("FROM R", "Times-Roman")],
    ])
    assert _code_block(sql)
    prose = _block_with_spans([[
        span("There are three dimensions we are interested in: ", "Times-Roman"),
        span("part, supplier, ", "Courier"), span("and customer.", "Times-Roman"),
    ]])
    assert not _code_block(prose)
    # 表格一行：斜体的行名后面是几格等宽记号
    row = _block_with_spans([[
        span("rooster", "NimbusRomNo9L-ReguItal"), span(" -female, +chicken, +adult", "Courier"),
    ]])
    assert _code_block(row)


def test_lone_relation_symbol_from_a_symbol_font_stays_a_formula():
    """CMSY 里的 ≺ 被 OCR 认成 <。单独一个也是公式，不能当成正文里的小于号。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, x0):
        return {"text": text, "font": font, "size": 10.0, "bbox": (x0, 0, x0 + 20, 10), "origin": (x0, 8)}

    block = _block_with_spans([[
        span("Thus (part) ", "Times-Roman", 0), span("<", "CMSY10", 60), span(" (part, customer).", "Times-Roman", 70),
    ]], size=10.0)
    sent, formulas = _placeholderize(block)
    assert sent == "Thus (part) {v1} (part, customer)." and len(formulas) == 1


def test_pdf_biblio_bare_label_line_does_not_exit():
    """悬挂缩进的标签和条目隔得远，[ESS] 单独成一块。它是条目的标签，后面的引文仍跳过。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size=10.7):
        return TextBlock(page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[],
                         text=text, size=size, color="#000000", bold=False)

    blocks = [
        blk("We thank the reviewers for help with the experiments."),
        blk("References", 14.3),
        blk("S. Chaudhuri and K. Shim. Including Group-By in Query Optimization. In VLDB, pages 354-366, 1994."),
        blk("[ESS]"),
        blk("Goetz Graefe. Query Evaluation Techniques for Large Databases. ACM Computing Surveys, 1993."),
        blk("[GBLP95] J. Gray, A. Bosworth. Data Cube: A Relational Aggregation Operator. MSR-TR-95-22, 1995."),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True]


def test_inline_typewriter_words_keep_their_glyphs():
    """句子里的等宽属性名换成占位符，贴回原字。整块都是等宽字的不在这里，那是代码块。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, x0):
        return {"text": text, "font": font, "size": 10.0, "bbox": (x0, 0, x0 + 40, 10), "origin": (x0, 8)}

    block = _block_with_spans([[
        span("the total sales of ", "Times-Roman", 0), span("part", "Courier", 90),
        span(" that was bought", "Times-Roman", 120),
    ]], size=10.0)
    sent, formulas = _placeholderize(block)
    assert sent == "the total sales of {v1} that was bought" and len(formulas) == 1


def test_list_number_keeps_its_point():
    from app.formats.pdf import _list_number

    assert _list_number("1. x is a partial order, and", "1。x 是一个偏序，并且") == "1. x 是一个偏序，并且"
    assert _list_number("2. There is a top element.", "2. 存在一个顶元素。") == "2. 存在一个顶元素。"
    assert _list_number("Plain sentence.", "1。不动") == "1。不动"
    assert _list_number("Figure 1: The eight views", "图一: 八个视图") == "图 1: 八个视图"
    assert _list_number("Figure 12: Bars", "图 12: 条") == "图 12: 条"



def test_sentence_around_formulas_is_translated_and_bare_formula_line_is_not():
    """公式外有 If、then、Otherwise 三个词，是一句话，照翻。For all 加一长串公式仍是公式行。"""
    from app.formats.pdf import extract_blocks

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)

    def write(y, pieces):
        x = 40.0
        for text, font in pieces:
            page.insert_text((x, y), text, fontsize=10, fontname=font)
            x += pymupdf.get_text_length(text, fontname=font, fontsize=10)

    write(60, [("(b) If ", "Times-Roman"), ("C(v) < C(u)", "Symbol"), (", then ", "Times-Roman"),
               ("B = C(v) - C(u)", "Symbol"), (". Otherwise, ", "Times-Roman"), ("B = 0", "Symbol"), (".", "Times-Roman")])
    write(120, [("For all ", "Times-Roman"), ("i: b - x - x - y < a + a + a + a", "Symbol"), (".", "Times-Roman")])
    texts = [block.text for block in extract_blocks(doc)]
    assert any("Otherwise" in text for text in texts)
    assert not any("For all" in text for text in texts)


def _page_data(*lines):
    """get_text("dict") 的最小结构：每个参数是一行的 span 列表 (字体, 文字)。"""
    return {"blocks": [{"type": 0, "lines": [
        {"spans": [{"font": font, "text": text, "size": 10.0} for font, text in spans]} for spans in lines
    ]}]}


def test_computer_modern_upright_is_math_when_the_body_is_times():
    """Times 排的论文里，CMR 只写 log、min、括号和等号，是公式。CM 排的论文里它是正文。"""
    from app.formats.pdf import _is_formula_span, _mark_math_roman

    times = _page_data([("NimbusRomNo9L-Regu", "We train the model to minimize " * 20), ("ABCDEF+CMR10", "log(1")])
    _mark_math_roman([times])
    log_span = times["blocks"][0]["lines"][0]["spans"][1]
    assert log_span.get("math") is True
    assert _is_formula_span({**log_span, "size": 10.0, "origin": (0, 0)}, 10.0)

    modern = _page_data([("CMR10", "We train the model to minimize " * 20), ("CMMI10", "x")])
    _mark_math_roman([modern])
    assert "math" not in modern["blocks"][0]["lines"][0]["spans"][0]


def test_a_font_that_writes_math_alphabet_letters_is_math():
    """字体名单里没有的数学斜体（newtxmath 配 Cochineal 叫 cochMI），字母映到 𝑁、𝑡 这些数学字母，
    同一字体写的括号、逗号也算公式。正文里偶尔夹一个数学字母的字体不算。"""
    from app.formats.pdf import _font_is_math, _mark_math_alphabet, _span_is_math

    data = _page_data(
        [("Cochineal-Roman", "fitted a hyperbola "), ("zxcMI", "𝑁("), ("zxcMI", "𝑡"), ("Cochineal-Roman", " to two millennia")],
        [("Cochineal-Roman", "where 𝑥 is the count of " * 3)],
    )
    _mark_math_alphabet([data])
    first, second = (line["spans"] for line in data["blocks"][0]["lines"])
    assert _span_is_math(first[1]) and _span_is_math(first[2])
    assert not _span_is_math(first[0]) and not _span_is_math(second[0])
    assert _font_is_math("NewTXMI7") and _font_is_math("cochMRM") and not _font_is_math("Cochineal-Italic")


def test_a_line_split_into_two_blocks_is_put_back_together():
    """PyMuPDF 把一行从上标处切开、后半截分进别的块时，挪回前半截后面。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, y0, y1, text):
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [{"text": text, "size": 10.0}]}

    head = line(100, 400, 150, 162, "All operations are element-wise. With β")
    tail = line(398, 440, 150, 163, "1 and β")
    nxt = line(100, 300, 162, 174, "we denote the power t.")
    other = line(100, 400, 200, 212, "Require: a step size")
    data = {"blocks": [{"type": 0, "bbox": (100, 150, 400, 162), "lines": [head]},
                       {"type": 0, "bbox": (100, 150, 440, 212), "lines": [tail, nxt, other]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [head, tail]
    assert data["blocks"][1]["lines"] == [nxt, other]


def test_a_two_column_gutter_is_not_a_split_line():
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, text):
        return {"bbox": (x0, 150, x1, 162), "dir": (1, 0), "spans": [{"text": text, "size": 10.0}]}

    left, right = line(60, 290, "left column text"), line(305, 540, "right column text")
    data = {"blocks": [{"type": 0, "bbox": (60, 150, 290, 162), "lines": [left]},
                       {"type": 0, "bbox": (305, 150, 540, 162), "lines": [right]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [left] and data["blocks"][1]["lines"] == [right]


def test_a_narrow_gutter_next_to_math_or_a_section_number_is_not_a_split_line():
    """栏沟不到一个字宽，右栏一行以公式起头、或是节号时，也不接到左栏的段落后面。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, y0, text, font="Times-Roman", size=10.0, y1=None):
        box = (x0, y0, x1, y1 if y1 is not None else y0 + 13)
        return {"bbox": box, "dir": (1, 0), "spans": [{"text": text, "font": font, "size": size, "bbox": box}]}

    para = [line(72, 301.3, 638 + 12 * i, "ducing internal covariate shift, and in doing so dramati-")
            for i in range(4)]
    formula = line(311, 540, 650, "x = u + b", "CMMI10")
    rest = line(311, 540, 662, "the training set, and E[x] is the mean")
    number = line(311, 326, 659, "3.2", "Times-Bold", 14.3)
    title = line(337.9, 540, 659, "Batch-Normalized Convolutional Net-", "Times-Bold", 14.3)
    data = {"blocks": [{"type": 0, "bbox": (72, 638, 301.3, 687), "lines": list(para)},
                       {"type": 0, "bbox": (311, 650, 540, 675), "lines": [formula, rest]},
                       {"type": 0, "bbox": (311, 659, 540, 672), "lines": [number, title]}]}
    _rejoin_split_lines(data)
    assert [block["lines"] for block in data["blocks"]] == [para, [formula, rest], [number, title]]

    # 右栏一行很高，同时压着左栏段落的两行，左栏只剩一行可比：仍是两栏
    first = line(72, 301.0, 325.6, "where the BN transform is applied independently to each")
    second = line(72, 301.1, 337.5, "dimension of x = Wu, with a separate pair of learned")
    last = line(72, 219.2, 345.9, "parameters per dimension.", y1=362.8)
    tall = line(311, 539.0, 329.1, "JJT . Thus, JJT = I, and so all singular values of J", "CMMI10", y1=346.0)
    data = {"blocks": [{"type": 0, "bbox": (72, 325.6, 301.1, 362.8), "lines": [first, second, last]},
                       {"type": 0, "bbox": (311, 329.1, 539, 346.0), "lines": [tall]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == [tall]

    # 分子分母上下对齐、都止于求和号前，不是一栏
    numerator = line(414.3, 418.2, 82.5, "1", "CMR7", 7.0, 89.4)
    denominator = line(413.5, 419.0, 89.8, "L", "CMMI7", 7.0, 96.8)
    total = line(421.8, 437.8, 81.4, "PL", "CMEX10")
    data = {"blocks": [{"type": 0, "bbox": (413.5, 82.5, 419.0, 102.8), "lines": [numerator, denominator]},
                       {"type": 0, "bbox": (421.8, 81.4, 437.8, 94.4), "lines": [total]}]}
    _rejoin_split_lines(data)
    assert total in data["blocks"][0]["lines"]

    # 栏顶的大号标题和左栏段落同一基线，左栏只剩一整行可比：字号大一截，不是那一行的后半截
    top = line(72, 301.1, 35.5, "not have to readjust to compensate for the change in the")
    tail = line(72, 138.8, 47.5, "distribution of x.")
    heading = [line(311, 318.2, 31, "2", "Times-Bold", 14.3, 50.9),
               line(332.5, 385.3, 31, "Towards", "Times-Bold", 14.3, 50.9)]
    data = {"blocks": [{"type": 0, "bbox": (72, 35.5, 301.1, 60.8), "lines": [top, tail]},
                       {"type": 0, "bbox": (311, 31, 385.3, 50.9), "lines": list(heading)}]}
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == heading


def test_the_hole_for_a_neighbour_stays_inside_its_font_box():
    """上一行网址的精确框往下多出半行，按它挖孔会削掉下一行网址的字头。孔夹在字体框里；
    单画量出的墨迹（字体框量错了）照旧。"""
    from app.formats.pdf import _hole_ink

    def box(span):
        return tuple(round(v, 1) for v in _hole_ink(span))

    url = {"text": " https://artofproblemsolving.com", "bbox": (121.2, 420.0, 525.0, 432.0),
           "ink": (121.2, 420.0, 525.0, 437.5)}
    assert box(url) == (121.2, 420.0, 525.0, 432.0)
    arrow = {"text": "←", "bbox": (130.2, 214.0, 139.9, 232.0), "ink": (130.2, 221.3, 139.9, 226.4)}
    assert box(arrow) == (130.2, 221.3, 139.9, 226.4)
    root = {"text": "√", "bbox": (330.5, 187.2, 337.0, 195.0), "ink": (330.5, 187.2, 337.0, 200.0), "ink_drawn": True}
    assert box(root) == (330.5, 187.2, 337.0, 200.0)


def test_a_formula_whose_font_box_reaches_the_next_line_stays_on_its_own_line():
    """ϵ 的字体框伸到下一行，按字体框算和下一行同一行高，分母就被接到下一行后面，分子跟过去。"""
    from app.formats.pdf import _rejoin_split_lines

    def span(text, font, size, box, ink=None):
        item = {"text": text, "font": font, "size": size, "bbox": box, "origin": (box[0], box[3] - 2)}
        if ink is not None:
            item["ink"] = ink
        return item

    def line(*spans):
        box = pymupdf.Rect(spans[0]["bbox"])
        for item in spans[1:]:
            box |= pymupdf.Rect(item["bbox"])
        return {"bbox": tuple(box), "dir": (1, 0), "spans": list(spans)}

    lead = line(span("Equation (10) reveals that we must predict (", "Times-Roman", 10.0, (108, 187.2, 305, 199.9)))
    minus = line(span("x", "CMBX10", 10.0, (307.6, 187.3, 314, 206.1), (307.6, 190.5, 314, 197.2)),
                 span(" −", "CMSY10", 10.0, (314, 187.3, 327.1, 206.1), (318, 192.8, 327.1, 193.6)))
    numerator = line(span("β", "CMMI7", 7.0, (339.2, 185.3, 344, 192.8), (339.2, 185.6, 344, 191.9)))
    denominator = line(span("√", "CMSY7", 7.0, (330.5, 187.2, 337, 205.7), (330.5, 193.3, 337, 200.0)),
                       span("α", "CMMI7", 7.0, (337, 187.2, 341, 205.7), (337, 195.4, 341, 199.8)),
                       span(" ϵ", "CMMIB10", 10.0, (355.2, 187.2, 361.7, 205.7), (355.2, 194.0, 361.7, 198.4)))
    given = line(span("given x. Since x is available as", "Times-Roman", 10.0, (370.2, 187.4, 504, 198.9)))
    nxt = line(span("input to the model, we may choose the parameterization", "Times-Roman", 10.0, (108, 201.7, 331.3, 211.7)))
    data = {"blocks": [{"type": 0, "bbox": (108, 187.2, 305, 199.9), "lines": [lead]},
                       {"type": 0, "bbox": (307.6, 185.3, 361.7, 206.1), "lines": [minus, numerator, denominator]},
                       {"type": 0, "bbox": (108, 187.4, 504, 211.7), "lines": [given, nxt]}]}
    _rejoin_split_lines(data)
    host = next(block for block in data["blocks"] if lead in block["lines"])
    assert denominator in host["lines"] and numerator in host["lines"]
    assert nxt not in host["lines"]


def test_a_split_line_goes_to_its_nearest_left_neighbour():
    """两个块里都有能接上的左半截：接到挨得最近的那个，不在两个块之间来回挪。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, y0, text, font="CMMI10"):
        box = (x0, y0, x1, y0 + 10)
        return {"bbox": box, "dir": (1, 0), "spans": [{"text": text, "font": font, "size": 10.0, "bbox": box}]}

    # 两个左半截上下错开，各自和右半截同一行高，彼此不是同一行
    far = line(72, 262, 96, "the rule is", "Times-Roman")
    near = line(250, 264, 104, "c =")
    piece = line(265, 300, 100, "d − e")
    data = {"blocks": [{"type": 0, "bbox": (72, 96, 262, 106), "lines": [far]},
                       {"type": 0, "bbox": (250, 104, 264, 114), "lines": [near]},
                       {"type": 0, "bbox": (265, 100, 300, 110), "lines": [piece]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == [near, piece]


def test_a_justified_heading_split_word_by_word_stays_one_paragraph():
    """窄栏标题两端对齐，词距拉到两个字宽，PyMuPDF 一词一行。按整行认回一段。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, x1, y0):
        span = {"text": text, "font": "Times-Bold", "size": 14.3}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 20), [span])

    items = [
        item("2", 311, 318, 31), item("Towards", 333, 385, 31), item("Reducing", 408, 466, 31),
        item("Internal", 490, 540, 31), item("Covariate Shift", 333, 426, 49),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3, 4]]


def test_table_cells_on_one_baseline_stay_apart():
    from app.formats.pdf import _split_lines

    def item(text, x0, x1):
        span = {"text": text, "font": "Times-Roman", "size": 9.0}
        return (text, pymupdf.Rect(x0, 100, x1, 110), [span])

    items = [item("Training", 128, 158), item("Inference needed", 184, 246), item("No difficulties", 437, 489)]
    assert len(_split_lines(items)) == 3


def test_formula_box_height_follows_the_glyph_ink():
    """CMSY 的字体伸量将近两行高。框按字体算会把公式压小，还会裁进上下两行的字。"""
    from app.formats.pdf import _tighten_formula_height

    arrow = {"text": "←", "ink": (130.2, 221.3, 139.9, 226.4)}
    zero = {"text": "0", "ink": (142.6, 219.8, 147.4, 226.5)}
    part = {"bbox": pymupdf.Rect(114.7, 218.8, 147.4, 237.1), "h": 18.3, "d": 10.8, "spans": [arrow, zero]}
    _tighten_formula_height(part)
    assert [round(v, 1) for v in part["bbox"]] == [114.7, 219.2, 147.4, 227.1]
    assert part["h"] == 7.9 and part["d"] == 0.8
    # 减号整个在基线上面（数学轴），收紧后下沿仍在基线上方，不能贴到基线上
    minus = {"text": "−", "ink": (197.2, 713.4, 205.0, 715.0)}
    part = {"bbox": pymupdf.Rect(197.2, 708.9, 205.0, 726.2), "h": 17.3, "d": 9.6, "spans": [minus]}
    _tighten_formula_height(part)
    baseline = 726.2 - 9.6
    assert part["d"] < 0 and abs(part["bbox"].y1 - part["d"] - baseline) < 0.1
    # 点阵字没有墨迹框，照旧用原来的框
    unknown = {"bbox": pymupdf.Rect(0, 0, 10, 20), "h": 20, "d": 5, "spans": [{"text": "x"}]}
    _tighten_formula_height(unknown)
    assert unknown["h"] == 20


def test_a_block_is_sized_by_its_base_line_text_not_a_subscript():
    """d_model 的 model 用正文字体排成小字下标，是这一行最长的一截，块字号仍按基线上的 and 算。"""
    from app.formats.pdf import _main_span

    def span(text, font, size, y):
        return {"text": text, "font": font, "size": size, "origin": (0, y)}

    line = [span("and", "NimbusRomNo9L-Regu", 10, 225.0), span(" W", "CMMI10", 10, 225.0),
            span("model", "NimbusRomNo9L-Regu", 5, 227.6), span(".", "NimbusRomNo9L-Regu", 10, 225.0)]
    assert _main_span([line])["text"] == "and"


def test_a_formula_line_wrapped_from_a_full_line_is_kept():
    """上一行写到版心右缘、没写完一句，下一行 and W^O ∈ … 从左缘接着写，是这句的后半，不是单独的式子。
    叠着排的式子写不到右缘。"""
    from app.formats.pdf import TextBlock, _wraps_from_above
    from app.formats.pdf_frame import TextFrame

    frame = TextFrame(108, 504, 72, 720, 10)
    line = pymupdf.Rect(107.5, 203.1, 502.8, 217.6)
    above = TextBlock(page=4, rect=line, line_rects=[line], text="Where the projections are parameter matrices W Q",
                      size=10, color="#000000", bold=False)
    tail = pymupdf.Rect(108, 215.6, 201.2, 227.9)
    rows = [pymupdf.Rect(108, y, 503, y + 12) for y in (100, 112, 124, 136)]
    assert _wraps_from_above([above], 4, tail, "and W O ∈Rhdv×dmodel.", rows, frame)
    short = pymupdf.Rect(107.5, 203.1, 200, 217.6)
    stacked = TextBlock(page=4, rect=short, line_rects=[short], text="et = Ext", size=10, color="#000000", bold=False)
    assert not _wraps_from_above([stacked], 4, tail, "ht = g(Uht−1 + Wet)", rows, frame)
    # 两栏排时上一行写到的是左栏的右缘，不是版心右缘
    column = pymupdf.Rect(108, 203.1, 297.5, 217.6)
    left = TextBlock(page=4, rect=column, line_rects=[column], text="where the mean is computed over",
                     size=10, color="#000000", bold=False)
    rows = [pymupdf.Rect(108, y, 298, y + 12) for y in (100, 112, 124, 136)]
    assert _wraps_from_above([left], 4, tail, "and x̂ = (x − µ)/σ.", rows, frame)
    # 居中排的图注，末行和上面的整行中线对齐
    first, second = pymupdf.Rect(83.2, 135.0, 511.7, 147.1), pymupdf.Rect(87.9, 149.7, 507.8, 160.6)
    caption = [
        TextBlock(page=4, rect=line, line_rects=[line], text=text, size=10.9, color="#000000", bold=False)
        for line, text in ((first, "Table 15 | Comparison of models on"),
                           (second, "Numbers in bold denote the performance is statistically"))
    ]
    end = pymupdf.Rect(219.6, 161.2, 375.7, 174.2)
    assert _wraps_from_above(caption, 4, end, "significant (t−test with 𝑝< 0.01).", [], frame)


def test_tex_spacing_accents_are_composed_before_translation():
    from app.formats.pdf import _compose_accents

    assert _compose_accents("Universit´e de Montr´eal") == "Université de Montréal"
    assert _compose_accents("Fr´ed´eric, na¨ıve, ˆetre") == "Frédéric, naïve, être"
    assert _compose_accents("Fran¸cois, gar c¸on") == "François, gar çon"
    assert _compose_accents("f(x)´ and x¸") == "f(x)´ and x¸"
    # 字母是斜体、粗体时，标记夹在重音和字母之间
    assert _compose_accents("the estimate ˆ{i}y{/i} and ˜{i}x,r{/i}") == "the estimate {i}ŷ{/i} and {i}x̃,r{/i}"
    assert _compose_accents("output ˆ{b}Y{/b}") == "output {b}Ŷ{/b}"


def test_a_stacked_accent_is_split_from_the_glyphs_before_it():
    """「(ˆ」里的帽子叠在后面的 y 上，切成单独的一截。网址里的 ˜、正则里的 ˆ 占自己的宽度，
    是 ~ ^，不切。"""
    from app.formats.pdf import _PageChars, _split_accents

    doc = pymupdf.open()
    page = doc.new_page()
    # 嵌入字体文件才能写 ˆ ˜；按内置 Times 的单字节编码写会换成别的字
    font = pymupdf.Font("tiro")
    page.insert_font(fontname="T", fontbuffer=font.buffer)
    page.insert_text((100, 100), "(\u02c6", fontname="T", fontsize=10)
    page.insert_text((100 + font.text_length("(", 10), 100), "y", fontname="T", fontsize=10)
    page.insert_text((100, 140), "nyu.edu/\u02dcroweis", fontname="T", fontsize=10)
    page.insert_text((100, 180), 'r"\u02c6The"', fontname="T", fontsize=10)
    data = page.get_text("dict")
    _split_accents(_PageChars(page), data)
    texts = [span["text"] for block in data["blocks"] for line in block["lines"] for span in line["spans"]]
    assert texts == ["(", "\u02c6", "y", "nyu.edu/~roweis", 'r"^The"']


def test_caption_number_is_restored_inside_style_marks():
    from app.formats.pdf import _list_number

    assert _list_number("Figure 2: Samples", "{z110}图二:{/z} 模型样本") == "{z110}图 2:{/z} 模型样本"
    assert _list_number("Table 1: Parzen", "{z110}{b}表一：{/b}{/z} 基于") == "{z110}{b}表 1：{/b}{/z} 基于"


def test_small_caps_heading_is_sent_without_size_marks():
    """小型大写的标题首字母大一号、其余小一号，是一种字，不在词中间插字号标记。"""
    from app.formats.pdf import TextBlock, _placeholderize

    def span(text, size, x0):
        return {"text": text, "font": "NimbusRomNo9L-Regu", "size": size, "flags": 0,
                "bbox": (x0, 100, x0 + 8 * len(text), 112), "origin": (x0, 110)}

    spans = [span("2.1", 12.5, 100), span(" ", 12.5, 120), span("A", 12.5, 125), span("DAM", 10.0, 133),
             span("’", 12.5, 157), span("S", 10.0, 161), span(" ", 10.0, 169), span("UPDATE", 10.0, 173)]
    rect = pymupdf.Rect(100, 100, 230, 112)
    block = TextBlock(0, rect, [rect], "2.1 ADAM’S UPDATE", 10.0, "#000", False, span_lines=[spans])
    sent, _ = _placeholderize(block, 0)
    assert sent == "2.1 ADAM’S UPDATE"


def test_small_caps_words_in_body_text_take_the_body_size():
    """正文里整词小型大写（AGENT）比正文小一号，译成中文后照搬小字号会缩成一团。"""
    from app.formats.pdf import TextBlock, _placeholderize

    def span(text, size, x0, base=110):
        return {"text": text, "font": "NimbusRomNo9L-Regu", "size": size, "flags": 0,
                "bbox": (x0, base - 10, x0 + 5 * len(text), base + 2), "origin": (x0, base)}

    spans = [span("roles like", 9.96, 100), span(" ", 7.97, 150), span("AGENT", 7.97, 154),
             span(", or the", 9.96, 180), span("TM", 7.97, 220, base=106), span(" x", 9.96, 232)]
    rect = pymupdf.Rect(100, 100, 245, 112)
    block = TextBlock(0, rect, [rect], "roles like AGENT, or theTM x", 9.96, "#000", False, span_lines=[spans])
    sent, _ = _placeholderize(block, 0)
    # 抬高的上标不是小型大写，字号照旧
    assert sent.startswith("roles like AGENT, or the{z")


def test_bare_chinese_section_number_is_restored():
    from app.formats.pdf import _normalize_heading_number

    assert _normalize_heading_number("2 ALGORITHM", "二 算法") == "2 算法"
    assert _normalize_heading_number("2.1 Update rule", "二点一 更新规则") == "2.1 更新规则"
    assert _normalize_heading_number("3 Method", "{z125}三{/z} 方法") == "{z125}3{/z} 方法"
    assert _normalize_heading_number("1 A new idea", "一种新想法") == "一种新想法"


def test_fraction_pieces_rejoin_their_line_with_less_overlap():
    """分母、大括号把框往下拉，同一行的两截上下只交叠三四成。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, y0, y1, text):
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [{"text": text, "size": 10.0}]}

    root = line(188, 202.4, 259.0, 271.2, "√αt")
    paren = line(205.8, 211.7, 253.9, 263.8, "(")
    data = {"blocks": [{"type": 0, "bbox": (108, 258, 202, 274), "lines": [line(108, 184, 258, 274, "to compute x ="), root]},
                       {"type": 0, "bbox": (205, 253, 212, 264), "lines": [paren]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"][-1] is paren


def test_math_line_box_follows_glyph_ink():
    """CMSY 的框按字体伸量能有两行高。行框、格子都按墨迹算；根号的起笔点在顶上，也按墨迹。"""
    from app.formats.pdf import _line_box, _text_box

    arrow = {"text": "←", "font": "CMSY10", "size": 9.7, "bbox": (130.2, 218.8, 139.9, 235.6),
             "origin": (130.2, 226.3), "ink": (130.2, 221.3, 139.9, 226.4)}
    word = {"text": "(Initialize", "font": "NimbusRomNo9L-Regu", "size": 9.7, "bbox": (147.4, 217.3, 194.8, 229.0),
            "origin": (147.4, 226.3), "ink": (147.4, 219.7, 194.8, 228.0)}
    assert [round(v, 1) for v in _text_box(arrow)] == [130.2, 220.7, 139.9, 227.0]
    assert _text_box(word) == pymupdf.Rect(147.4, 217.3, 194.8, 229.0)
    line = {"bbox": (130.2, 217.3, 194.8, 235.6), "spans": [arrow, word]}
    assert [round(v, 1) for v in _line_box(line)] == [130.2, 217.3, 194.8, 229.0]


def test_formula_pieces_touching_on_one_line_stay_one_paragraph():
    """分式、大括号把一行式子拆成几截。纯符号的那一截不能把句子切开。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, x1, y0, y1, *spans):
        spans = spans or ((text, "CMMI10", 10.0),)
        return (text, pymupdf.Rect(x0, y0, x1, y1), [{"text": t, "font": f, "size": z} for t, f, z in spans])

    items = [
        item("to compute x =", 108, 184, 258, 268, ("to compute", "NimbusRomNo9L-Regu", 9.9), (" x =", "CMMI10", 10.0)),
        item("1", 193.4, 197.4, 256.4, 262.2, ("1", "CMR7", 7.0)),
        item("√αt", 187.9, 202.4, 263.5, 271.1, ("√αt", "CMSY7", 7.0)),
        item("(", 205.8, 211.7, 253.9, 263.8, ("(", "CMEX10", 10.0)),
        item("xt −", 211.8, 231.3, 260.5, 267.7),
        item("+σz, where the sampling", 303, 504, 258, 268,
             ("+σz", "CMMI10", 10.0), (", where the sampling", "NimbusRomNo9L-Regu", 9.9)),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3, 4, 5]]


def test_a_line_with_three_words_among_formulas_is_translated():
    """Therefore, ∫… = ∫… It follows：公式多、字母占比低，但有三个正文词，是句子。"""
    from app.formats.pdf import _untranslatable

    def span(text, font):
        return {"text": text, "font": font}

    spans = [span("p(ϵ)", "CMMI10"), span("Therefore", "Times-Roman"), span(",", "Times-Roman"),
             span("∫q(z|x)f(z)dz = ∫p(ϵ)f(z)dϵ = ∫p(ϵ)f(g(ϵ,x))dϵ", "CMMI10"), span(".", "Times-Roman"),
             span("It", "Times-Roman"), span("follows", "Times-Roman")]
    text = "p(ϵ) Therefore, ∫q(z|x)f(z)dz = ∫p(ϵ)f(z)dϵ = ∫p(ϵ)f(g(ϵ,x))dϵ. It follows"
    assert not _untranslatable(spans, text)
    assert _untranslatable(spans[:4], "p(ϵ) Therefore, ∫q(z|x)f(z)dz = ∫p(ϵ)f(z)dϵ")


def test_an_inline_fraction_rejoins_its_line_in_reading_order():
    """g(x) = 1/(1+exp(−x))：分母在另一块里，离等号比词距宽；分子排在分母前面。
    挪回来以后按从左到右排，分母才接得上等号。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, y0, y1, *spans):
        # Times 正文里的 CMR 提取时已经标成公式
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [
            {"text": text, "font": font, "size": 10.0, "math": font.startswith("CM")} for text, font in spans]}

    host = line(72, 219.4, 136, 149.3, ("rameters to be learned, and", "Times-Roman"), ("g(x) =", "CMR10"))
    numerator = line(243.2, 247.2, 137.3, 143.2, ("1", "CMR7"))
    denominator = line(225.6, 301, 136, 151.7, ("1+exp(−x)", "CMR10"), (". As", "Times-Roman"))
    data = {"blocks": [{"type": 0, "bbox": (72, 136, 247, 149), "lines": [host, numerator]},
                       {"type": 0, "bbox": (225, 136, 301, 152), "lines": [denominator]}]}
    _rejoin_split_lines(data)
    assert sorted(map(id, data["blocks"][0]["lines"])) == sorted(map(id, [host, numerator, denominator]))


def test_pieces_of_one_row_are_read_left_to_right():
    """分子排在分母前面时，按从左到右重排；上下两行不当成一行。"""
    from app.formats.pdf import _left_to_right

    def item(text, x0, y0, x1, y1):
        return (text, pymupdf.Rect(x0, y0, x1, y1), [])

    host, numerator, denominator = item("g(x) =", 72, 136, 219, 149), item("1", 243, 137, 247, 143), item("1+exp(−x)", 225, 136, 301, 151)
    nxt = item("increases, g′(x) tends to zero", 72, 149.7, 301, 163)
    assert _left_to_right([host, numerator, denominator, nxt]) == [host, denominator, numerator, nxt]


def test_plain_words_a_wide_gap_apart_are_not_rejoined():
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, text):
        return {"bbox": (x0, 136, x1, 149), "dir": (1, 0), "spans": [{"text": text, "font": "Times-Roman", "size": 10.0}]}

    left, right = line(72, 219, "learned, and the"), line(227, 301, "output of the layer")
    data = {"blocks": [{"type": 0, "bbox": (72, 136, 219, 149), "lines": [left]},
                       {"type": 0, "bbox": (227, 136, 301, 149), "lines": [right]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == [right]


def test_a_centred_definition_line_keeps_its_glyphs():
    """居中单独成行的式子里只有几个连接词（if and only if、for all），不当成句子翻。
    从栏左缘起笔、接着上一行写的式子行仍按句子翻。"""
    from app.formats.pdf import _displayed, _left_margins, _untranslatable

    body = {"bbox": (72, 0, 540, 12), "spans": [{"text": "A body line long enough to count as the column margin."}]}
    data = {"blocks": [{"type": 0, "lines": [body] * 4}]}
    margins = _left_margins(data)
    assert margins == [72]
    spans = [{"text": "(a1, …, an) ⪯ (b1, …, bn)", "font": "CMMI10", "size": 10.0},
             {"text": "if and only if", "font": "Times-Roman", "size": 10.0},
             {"text": "ai ⪯ bi", "font": "CMMI10", "size": 10.0},
             {"text": "for all", "font": "Times-Roman", "size": 10.0}, {"text": "i", "font": "CMMI10", "size": 10.0}]
    text = "(a1, …, an) ⪯ (b1, …, bn) if and only if ai ⪯ bi for all i"
    assert _displayed(pymupdf.Rect(215, 100, 400, 112), spans, margins)
    assert not _displayed(pymupdf.Rect(72, 100, 400, 112), spans, margins)
    assert _untranslatable(spans, text, 8)
    assert not _untranslatable(spans, text)


def test_a_lone_numerator_goes_back_to_its_fraction():
    """分子单独剩在一个块里时，挪到正下方那段式子的行旁边。上一行的正文横向也盖得住它，不算。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, y0, y1, *spans):
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [
            {"text": text, "font": font, "size": size, "bbox": box, "ink": box}
            for text, font, size, box in spans]}

    above = line(108, 504, 243, 262, ("where the model predicts noise", "Times-Roman", 9.9, (108, 243, 504, 262)))
    denominator = line(235, 296, 258, 276, ("√1−αt", "CMSY7", 7.0, (235, 264.4, 260, 270.2)))
    numerator = line(243, 251, 256, 263, ("βt", "CMMI7", 7.0, (243.4, 256.2, 250.7, 262.5)))
    data = {"blocks": [{"type": 0, "bbox": (108, 243, 504, 262), "lines": [above]},
                       {"type": 0, "bbox": (235, 258, 296, 276), "lines": [denominator]},
                       {"type": 0, "bbox": (243, 256, 251, 263), "lines": [numerator]}]}
    _rejoin_split_lines(data)
    together = [block["lines"] for block in data["blocks"] if numerator in block["lines"]]
    assert len(together) == 1 and sorted(map(id, together[0])) == sorted(map(id, [denominator, numerator]))
    assert data["blocks"][0]["lines"] == [above]


def test_fraction_bars_drawn_as_one_pixel_images_are_formula_curves():
    """老 TeX 把分式线、根号上的横线画成拉长的一像素图片。和矢量线一样算公式的曲线，
    公式挪走时原处的线才会擦掉；插图不算。"""
    from app.formats.pdf import _page_curve_rects

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    dot = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 1, 1), False)
    dot.clear_with(0)
    bar = pymupdf.Rect(357.6, 598.8, 394.1, 599.3)
    page.insert_image(bar, pixmap=dot, keep_proportion=False)
    photo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    photo.clear_with(128)
    page.insert_image(pymupdf.Rect(100, 100, 300, 300), pixmap=photo)
    # 描边的横线：路径框高为零，画出来有线宽那么粗
    page.draw_line((234.7, 263.1), (259.9, 263.1), width=0.4)
    rects = _page_curve_rects(page)
    assert any(abs(r.x0 - bar.x0) < 0.1 and abs(r.y1 - bar.y1) < 0.1 for r in rects)
    assert not any(r.width > 150 and r.height > 150 for r in rects)
    stroke = next(r for r in rects if abs(r.x1 - 259.9) < 0.5)
    assert stroke.y0 < 263.1 < stroke.y1 and stroke.height >= 0.39

    from app.formats.pdf_runs import intersecting_curves
    assert intersecting_curves(pymupdf.Rect(234.7, 255.9, 295.8, 276.6), rects)


def test_a_formula_only_takes_the_rules_its_glyphs_touch():
    """|m/√v| < 1 的字体框伸到下一行，盖住了下一行 √E[g²] 的根号横线。按墨迹认，横线归下一行的式子。"""
    from app.formats.pdf import _absorb_one_box

    def formula(bbox, inks):
        spans = [{"text": "x", "ink": ink} for ink in inks]
        return {"bbox": pymupdf.Rect(bbox), "d": 3.0, "spans": spans}

    above = formula((334.3, 131.1, 395.7, 175.0), [(334.3, 133.6, 364.0, 145.2), (364.7, 133.9, 373.2, 145.6), (380.0, 135.9, 395.7, 141.2)])
    below = formula((355.1, 143.0, 433.7, 179.1), [(355.1, 145.6, 390.0, 157.7), (391.5, 150.6, 413.0, 158.4), (414.0, 148.9, 433.7, 155.1)])
    own_radical = pymupdf.Rect(364.7, 138.2, 373.2, 138.6)
    next_radical = pymupdf.Rect(391.5, 150.0, 413.0, 150.4)
    _absorb_one_box(above, [own_radical, next_radical])
    _absorb_one_box(below, [own_radical, next_radical])
    assert above["curves"] == [own_radical]
    assert below["curves"] == [next_radical]
    assert abs(above["bbox"].x1 - 395.7) < 0.01


def test_a_fraction_bar_stays_when_its_numerator_stays():
    """分子在另一块里留在原处时，分数线也留着：跟分母挪走，分子就悬在半空。"""
    from app.formats.pdf import _absorb_one_box

    denominator = {"text": "1−α", "origin": (164.2, 500.0), "ink": (164.2, 496.4, 187.5, 502.0)}
    numerator = {"text": "1−α", "origin": (159.7, 492.5), "ink": (159.7, 487.4, 187.5, 494.2)}
    bar = pymupdf.Rect(159.7, 494.9, 187.5, 495.3)
    alone = {"bbox": pymupdf.Rect(164.2, 490.1, 197.2, 507.6), "d": 3.0, "spans": [denominator]}
    _absorb_one_box(alone, [bar], [numerator, denominator])
    assert alone["curves"] == []
    whole = {"bbox": pymupdf.Rect(159.7, 487.0, 197.2, 507.6), "d": 3.0, "spans": [numerator, denominator]}
    _absorb_one_box(whole, [bar], [numerator, denominator])
    assert whole["curves"] == [bar]


def test_a_formula_leaves_the_edge_of_a_proof_box():
    """证毕方框紧贴在行尾公式下面。方框的上边和两条竖边相接，是方框的一边，不跟公式挪走。"""
    from app.formats.pdf import _absorb_one_box

    g = {"text": "g", "origin": (493.0, 571.0), "ink": (493.0, 566.5, 496.9, 574.6)}
    paren = {"text": ")", "origin": (497.6, 569.5), "ink": (497.6, 562.5, 501.5, 572.0)}
    top = pymupdf.Rect(496.9, 575.2, 503.3, 575.6)
    sides = [pymupdf.Rect(496.7, 575.0, 497.1, 582.1), pymupdf.Rect(503.1, 575.0, 503.5, 582.1),
             pymupdf.Rect(496.9, 581.5, 503.3, 581.9)]
    formula = {"bbox": pymupdf.Rect(461.7, 561.9, 501.5, 573.0), "d": 3.0, "spans": [g, paren]}
    _absorb_one_box(formula, [top, *sides], [g, paren])
    assert formula["curves"] == []
    bar = pymupdf.Rect(493.0, 575.2, 501.5, 575.6)
    lone = {"bbox": pymupdf.Rect(461.7, 561.9, 501.5, 573.0), "d": 3.0, "spans": [g, paren]}
    _absorb_one_box(lone, [bar], [g, paren])
    assert lone["curves"] == [bar]


def test_formula_holes_follow_the_neighbours_ink_not_their_font_box():
    """上一行的大括号字体框伸到下一行的 N 上，墨迹没有。按墨迹挖洞，N 不被削掉一角。"""
    from app.formats.pdf import _formula_form, _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    fms = _page_form(doc, page)
    base = {"text": "N", "bbox": (327.5, 456.4, 336.9, 469.7)}
    formula = {"bbox": pymupdf.Rect(base["bbox"]), "spans": [base]}
    brace = {"text": "{", "bbox": (328.0, 445.3, 332.9, 462.3)}

    def hole_bottom(neighbour):
        """孔的下沿（页面坐标）。第一个 re 是公式自己的裁剪框，后面的是孔。"""
        xref = _formula_form(doc, page, fms, formula, "/Fx", [dict(base), neighbour])
        rects = re.findall(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", doc.xref_stream(xref).decode())
        [(_, y, _, _)] = rects[1:]
        return page.rect.height - float(y)

    # N 的墨迹从 460.3 起
    assert hole_bottom(brace) > 460.3
    assert hole_bottom({**brace, "ink": (328.0, 445.6, 332.9, 457.4)}) <= 457.5


def test_a_formula_cut_from_a_text_span_is_not_holed_by_that_span():
    """公式检测模型把「 σ」里的 σ 切出来做公式。页面上还是原来那个「 σ」span，框和公式的 σ 对不上；
    公式带上标、比一行高时，按它挖洞会把 σ 挖掉（译文里只剩上标）。原 span 只留切剩的字，空格没有墨迹，不挖。
    上一行伸下来的字照样挖"""
    from app.formats.pdf import _formula_form, _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    fms = _page_form(doc, page)
    whole = (269.5, 213.2, 277.4, 222.2)
    sigma = {"text": "σ", "bbox": (272.0, 213.2, 277.4, 222.2), "source": whole}
    square = {"text": "2", "bbox": (277.6, 206.0, 280.0, 214.0)}
    formula = {"bbox": pymupdf.Rect(272.0, 206.0, 280.0, 222.2), "spans": [sigma, square]}
    source = {"text": " σ", "bbox": whole,
              "chars": [(" ", pymupdf.Rect(269.5, 213.2, 272.0, 222.2)), ("σ", pymupdf.Rect(272.0, 213.2, 277.4, 222.2))]}
    above = {"text": "p", "bbox": (273.0, 196.0, 276.0, 205.5)}

    def clip_rects(page_spans):
        xref = _formula_form(doc, page, fms, formula, "/Fx", page_spans)
        return re.findall(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", doc.xref_stream(xref).decode())

    assert len(clip_rects([dict(source), dict(square)])) == 1
    assert len(clip_rects([dict(source), dict(square), above])) == 2


def test_list_item_near_the_page_middle_stays_left_aligned():
    """列表项的中线碰巧对上最宽一块的中线，仍是左对齐，不按居中写回。子图下面的 (a) 小标题照旧居中。"""
    from app.formats.pdf import TextBlock, _write_rect

    def blk(text, x0, y0, x1):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 23.7)
        return TextBlock(page=17, rect=rect, line_rects=[rect], text=text, size=20.4, color="#000", bold=False)

    widest = blk("• Apache top-level project since October, 2015", 35.2, 101.8, 477.8)
    sibling = blk("– Optional SQL parser, JDBC server", 71.2, 317.8, 400.0)
    item = blk("– Community-authored rules, adapters", 71.2, 341.8, 431.0)
    rect, align = _write_rect(item, [(widest, ""), (sibling, ""), (item, "")], True, False, None)
    assert align == "left"
    assert rect.x0 == item.rect.x0

    caption = blk("(a) Pixel space nearest neighbors", 197.0, 367.0, 317.0)
    _, align = _write_rect(caption, [(widest, ""), (sibling, ""), (caption, "")], True, False, None)
    assert align == "center"


def test_heading_at_the_frame_edge_stays_left_aligned():
    """2.6.1 小标题从版心左缘起笔。最宽一块是并进旁注、更靠左的段落，中线碰巧对上，仍按左对齐写回。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry

    def blk(text, x0, y0, x1, y1, size=10.0):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=16, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    heading = blk("2.6.1 Character Disjunction: The Square Bracket", 160, 173, 421, 185, 12)
    widest = blk("Figure 2.10 The caret for negation or just to mean the caret.", 96, 695, 490, 705, 9)
    title = blk("Exercises", 280, 300, 373, 312, 12)
    frame = TextFrame(160, 493, 111, 719, 10)
    geo = PageGeometry.from_rects(612, [heading.rect, widest.rect], height=792, frame=frame)
    items = [(heading, ""), (widest, ""), (title, "")]
    rect, align = _write_rect(heading, items, True, False, geo)
    assert align == "left" and rect.x0 == heading.rect.x0
    _, align = _write_rect(title, items, True, False, geo)
    assert align == "center"


def test_epigraph_and_attribution_at_the_frame_right_edge_are_written_flush_right():
    """章首引语和署名每行都收在版心右缘、离左缘很远：按右对齐写回，写入框不往右伸。
    同一行左边还有别的块时，右边那一块只是一行的后半截，照常左对齐。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_align import _place_line
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_typeset import typeset_lines

    def blk(text, rows, size=10.0):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=128, rect=rect, line_rects=rects, text=text, size=size, color="#000", bold=False)

    quote = blk("Numquam ponenda est pluralitas sine necessitate ‘Plurality should never be proposed unless needed’",
                [(296.0, 545.3, 493.2, 557.2), (290.0, 557.2, 493.2, 569.2)])
    author = blk("William of Occam", [(426.8, 570.1, 493.2, 580.9)], 9.0)
    body = blk("There is a problem with learning weights that make the model perfectly match the training data.",
               [(160.2, 611.8, 493.2, 623.8)])
    frame = TextFrame(160.2, 493.2, 100, 720, 10)
    geo = PageGeometry.from_rects(612, [quote.rect, author.rect, body.rect], height=792, frame=frame)
    items = [(quote, ""), (author, ""), (body, "")]
    for block in (quote, author):
        rect, align = _write_rect(block, items, True, False, geo)
        assert align == "right" and rect.x1 == block.rect.x1
    assert _write_rect(body, items, True, False, geo)[1] == "left"
    # 同一行左边是另一块：这一块是一行的后半截
    left = blk("The first half of the row", [(160.2, 700, 280, 712)])
    tail = blk("and the second half", [(300, 700, 493.2, 712)])
    assert _write_rect(tail, [*items, (left, ""), (tail, "")], True, False, geo)[1] == "left"
    # 图在前一页，绕图的正文左边空着，几行起笔对齐
    wrap = blk("Probably the most common visualization method, however, is to project the 100 dimensions",
               [(270, 218, 493.2, 230), (255, 230, 493.2, 242), (255, 242, 493.2, 254)])
    assert _write_rect(wrap, [*items, (wrap, "")], True, False, geo)[1] == "left"
    # 栏的右缘只有这一块排到，是它自己撑出来的
    lone = blk("1 request", [(300, 300, 493.2, 312)])
    short = blk("Count(w | denied the)", [(160.2, 280, 400, 292)])
    lone_geo = PageGeometry.from_rects(612, [lone.rect, short.rect], height=792, frame=frame)
    assert _write_rect(lone, [(short, ""), (lone, "")], True, False, lone_geo)[1] == "left"

    def start(text):
        line = typeset_lines(text, 9.0, 200.0, {})[0]
        ops = _place_line(line, 100.0, 200.0, 9.0, False, "right", frozenset(), {})[0]
        return float(next(op for op in ops if op.endswith("Tm")).split()[4]), line.width

    x, width = start("奥卡姆的威廉")
    assert abs(x + width - 300.0) < 0.5
    # 折行留在行尾的空格看不见，不占靠右的位置
    assert start("William of Occam ")[0] == start("William of Occam")[0]


def test_attribution_short_of_the_frame_edge_by_a_trailing_space_is_written_flush_right():
    """署名末尾带着词间空格，右缘比版心右缘短两三点，仍按右对齐写。引语末行用 \\hfill 推到行尾的署名也是。
    两端对齐的窄栏里一行拆出来的末一个词、和下面格子起笔对齐的表头照常左对齐。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry

    def blk(text, rows, size=10.0):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=75, rect=rect, line_rects=rects, text=text, size=size, color="#000", bold=False)

    frame = TextFrame(142.0, 475.0, 100, 720, 10)

    def align(block, others):
        items = [(other, "") for other in [*others, block]]
        geo = PageGeometry.from_rects(612, [other.rect for other, _ in items], height=792, frame=frame)
        return _write_rect(block, items, True, False, geo)[1]

    quote = blk("You are uniformly charming! cried he, with a smile of associating and now and then I bowed",
                [(167.1, 218, 475.2, 230), (167.1, 230, 431.4, 242)])
    author = blk("Random sentence generated from a Jane Austen trigram model", [(246.8, 243, 472.4, 252)], 9.0)
    body = blk("Predicting is difficult, especially about the future, as the old quip goes.",
               [(142.2, 280, 475.2, 292), (142.2, 292, 475.2, 304)])
    assert align(author, [quote, body]) == "right"
    # 署名和引语末行同在一行
    babbage = blk("On two occasions I have been asked, Pray, Mr. Babbage, if you put into the machine wrong figures",
                  [(167.1, 210, 475.2, 222), (167.1, 222, 203.4, 234)])
    signed = blk("Babbage (1864)", [(411.3, 222, 475.2, 234)])
    assert align(signed, [babbage, body]) == "right"
    # 两端对齐时一行里拆出来的末一个词，左边隔着一个拉开的词距
    para = blk("states yields a finite-transition increment", [(142.2, 300, 475.2, 312), (142.2, 312, 300.0, 324)])
    word = blk("comprising", [(316.5, 312, 472.6, 324)])
    assert align(word, [para, body]) == "left"
    # 表头的左缘和下面格子的起笔对齐
    header = blk("Emotion Label Emotion Explanation", [(300.0, 330, 472.6, 340)])
    cells = blk("A V AV Semantic", [(300.8, 344, 320.0, 354), (340.0, 344, 400.0, 354)])
    assert align(header, [cells, body]) == "left"


def test_two_line_attribution_is_split_like_the_source_and_written_flush_right(tmp_path):
    """引语加署名两行，首行短、起笔靠右：译文照首行的宽度断开，两行都收在原来的右缘。"""
    from app.formats.pdf import TextBlock, _render_translated
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    right = 475.0
    # 每行从右缘往回量出起笔
    for text, y in (("Time will explain.", 230), ("Jane Austen, Persuasion", 240)):
        page.insert_text((right - pymupdf.get_text_length(text, "tiro", 9), y), text, fontsize=9, fontname="tiro")
    body_line = "x" * 55
    for index in range(3):
        page.insert_text((right - pymupdf.get_text_length(body_line, "cour", 10), 300 + index * 12), body_line,
                         fontsize=10, fontname="cour")
    path = tmp_path / "src.pdf"
    doc.save(path)
    page = pymupdf.open(path)[0]

    def blk(text, rows, size):
        rects = [page.search_for(row)[0] for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=0, rect=rect, line_rects=rects, text=text, size=size, color="#000000", bold=False)

    author = blk("Time will explain. Jane Austen, Persuasion", ["Time will explain.", "Jane Austen, Persuasion"], 9.0)
    body = TextBlock(page=0, rect=pymupdf.Rect(page.search_for(body_line)[0]) | page.search_for(body_line)[-1],
                     line_rects=page.search_for(body_line), text=body_line * 3, size=10.0, color="#000000", bold=False)
    frame = TextFrame(body.rect.x0, body.rect.x1, 100, 720, 10)
    geo = PageGeometry.from_rects(612, [author.rect, body.rect], height=792, frame=frame)
    out = _render_translated(path, [author, body], ["时间会说明一切。简·奥斯汀，《劝导》", "正文" * 40], "zh-CN",
                             geometries=[geo])
    lines = [
        (line["bbox"], "".join(span["text"] for span in line["spans"]))
        for block in out[0].get_text("dict", clip=pymupdf.Rect(300, 215, 500, 250))["blocks"]
        for line in block.get("lines", [])
    ]
    assert [text for _, text in lines] == ["时间会说明一切。", "简·奥斯汀，《劝导》"]
    assert all(abs(bbox[2] - author.rect.x1) < 0.5 for bbox, _ in lines)


def test_header_cells_near_the_column_middle_are_not_all_centred_on_it():
    """表头一行的几格中线碰巧都在栏中线附近，整行不在栏中间：照各自的位置写，不都按栏中线居中叠在一起。
    节号和节名分成两块的标题整行对着栏中线，节名仍居中。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry

    def blk(text, x0, x1, y0=676.0, size=10.0):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 11.6)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=True)

    body = TextBlock(page=0, rect=pymupdf.Rect(160.2, 600, 493.2, 660), line_rects=[pymupdf.Rect(160.2, 600, 493.2, 612)],
                     text="Let us see how the cosine computes which of the words is closer", size=10.0, color="#000", bold=False)
    cells = [blk("pie", 313.4, 324.9), blk("data", 332.3, 349.3), blk("computer", 355.7, 392.6)]
    frame = TextFrame(160.0, 493.0, 100, 720, 10)
    items = [(body, ""), *((cell, "") for cell in cells)]
    geo = PageGeometry.from_rects(612, [block.rect for block, _ in items], height=792, frame=frame)
    assert [_write_rect(cell, items, True, False, geo)[1] for cell in cells] == ["left", "left", "left"]
    # 节号另成一块，节名和节号整行都对着栏中线
    number, title = blk("V.", 290.0, 302.0, y0=400), blk("Conclusions", 306.0, 362.0, y0=400)
    items = [(body, ""), (number, ""), (title, "")]
    assert _write_rect(title, items, True, False, geo)[1] == "center"


def test_a_paragraph_with_an_overfull_line_is_written_to_the_frame_edge():
    """一行断不开伸进页边的段落，译文照版心右缘写，不按伸出去的那一行。
    大半行都在版心外的块照旧按自己的宽度写。"""
    from app.formats.pdf import TextBlock, _write_limit, _write_rect
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(rows):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(page=75, rect=rect, line_rects=rects, text="This probabilistic knowledge " * 8,
                         size=10.0, color="#000", bold=False)

    para = blk([(157.1, 600, 475.2, 612), (142.2, 612, 475.2, 624), (142.2, 624, 489.5, 636),
                (142.2, 636, 475.2, 648), (142.2, 648, 262.0, 660)])
    wide = blk([(142.2, 400, 520.0, 412), (142.2, 412, 520.0, 424), (142.2, 424, 300.0, 436)])
    frame = TextFrame(142.0, 475.0, 100, 720, 10)
    geo = PageGeometry.from_rects(612, [para.rect, wide.rect], height=792, frame=frame)
    items = [(para, ""), (wide, "")]
    rect, align = _write_rect(para, items, True, False, geo)
    assert align == "left" and rect.x1 == 477.0
    assert _write_rect(wide, items, True, False, geo)[0].x1 == 522.0
    # 译文放不下往右扩时，也只扩到版心右缘，栏的右缘是被伸出去的那一行撑开的
    assert geo.columns[-1].x1 > 500
    assert _write_limit(para, rect, geo, []) == 477.0
    assert _write_limit(wide, rect, geo, []) > 500
    # 两行的段落伸出去一行。旁注另成一栏时正文栏的右缘被这一行撑开，仍照版心右缘写
    pair = blk([(157.1, 300, 489.5, 312), (142.2, 312, 475.2, 324)])
    geo = PageGeometry(612, (Column(79.4, 127.8), Column(142.2, 489.5)), (), 792, frame=frame)
    assert _write_rect(pair, [(pair, ""), (para, "")], True, False, geo)[0].x1 == 477.0


def test_slide_title_and_a_box_label_below_stay_apart():
    """幻灯片大标题下面图里的小字框不是图注续行，标题旁边的星形里的字也不是标题的下一行。"""
    from app.formats.pdf import TextBlock, _merge_caption_fragments, _merge_continuations

    def blk(text, x0, y0, x1, y1, size):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    title = [blk("Example 3: Improved Plan", 141, 27, 597, 73, 39), blk("scan ‘depts’", 132, 115, 209, 131, 14)]
    assert len(_merge_caption_fragments(title)) == 2
    assert len(_merge_continuations(title)) == 2

    star = [blk("Win", 404, 186, 433, 205, 17), blk("SortRemoveRule sorted on [deptno]", 595, 213, 695, 239, 12)]
    assert len(_merge_continuations(star)) == 2


def test_two_line_title_centered_on_the_page_is_written_centered():
    """幻灯片标题在页面正中。最宽一块是靠左的列表项，中线偏右，标题仍按页面居中写回。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_layout import PageGeometry

    def blk(text, x0, y0, x1, y1, size):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=29, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    title = blk("Beyond Phoenix 4.8 with Apache Calcite", 202, 18, 518, 100, 35.5)
    widest = blk("– To allow other forms of indices (maybe defined as external), e.g., a", 68, 210, 709, 256, 20)
    item = blk("• Get the missing SQL support", 31, 119, 367, 147, 24)
    geo = PageGeometry.from_rects(720, [title.rect, widest.rect, item.rect], height=405)
    rect, align = _write_rect(title, [(title, ""), (widest, ""), (item, "")], True, False, geo)
    assert align == "center"
    assert abs((rect.x0 + rect.x1) / 2 - 360) < 1

    widest_title = blk("Calcite Planning Process", 122, 25, 599, 76, 44)
    labels = [blk("Planner", 198, 104, 235, 115, 10), blk("Logical Plan", 97, 213, 151, 224, 10),
              blk("Best RelNode Graph", 451, 356, 543, 367, 10)]
    geo = PageGeometry.from_rects(720, [widest_title.rect, *(label.rect for label in labels)], height=405)
    items = [(widest_title, ""), *((label, "") for label in labels)]
    _, align = _write_rect(widest_title, items, True, False, geo)
    assert align == "center"


def test_title_page_centered_off_the_page_middle_is_written_centered():
    """题名页各行共用一条中线，但这条线不在页面正中（左右边距不对称），也不在全书版心的中线上。
    每一块都按这条中线居中写回，最宽的书名也一样。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry

    def blk(text, x0, y0, x1, y1, size):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("Speech and Language Processing", 108.2, 112.9, 494.1, 147.7, 28.9),
        blk("with Language Models", 227.7, 198.4, 374.6, 217.6, 15.9),
        blk("Third Edition draft", 236.1, 242.2, 366.2, 263.0, 17.2),
        blk("James H. Martin", 244.0, 395.4, 358.3, 416.1, 17.2),
        blk("University of Colorado at Boulder", 202.5, 416.0, 399.8, 433.1, 14.3),
        blk("Draft of August 19, 2026. Comments and typos welcome!", 100.2, 644.4, 501.0, 665.2, 17.2),
    ]
    geo = PageGeometry.from_rects(612, [b.rect for b in blocks], height=792,
                                  frame=TextFrame(203, 493, 110, 719, 10))
    items = [(b, "") for b in blocks]
    for block in blocks:
        rect, align = _write_rect(block, items, True, False, geo)
        assert align == "center", block.text
        assert abs((rect.x0 + rect.x1) / 2 - 301) < 1, block.text


def test_body_lines_beside_margin_notes_do_not_make_a_centered_page():
    """页边旁注把本页文字的左右范围撑宽，正文整行也不能算成短行。正文行都在版心中线上，
    缩进的例句中线碰巧挨着它，不算整页居中排。"""
    from app.formats.pdf import TextBlock, _centered_axis

    def blk(text, x0, y0, x1, size=10.0):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    body = [blk("Words don't appear out of nowhere. Any particular piece of text", 160, 253 + 12 * i, 493)
            for i in range(6)]
    note = blk("code switching", 101, 530, 146, size=8)
    example = blk("[he was and will remain a friend ... don't worry ... but have faith]", 192, 619, 465)
    title = blk("Corpora", 134, 213, 191, size=14)
    blocks = [*body, note, example, title]
    assert _centered_axis(example, blocks) is None


def test_white_label_on_a_black_box_is_pasted_back_as_is(tmp_path):
    """黑底白字的 Figure 标签不重写成正文颜色，原样贴回，后面的图注照译。"""
    src = tmp_path / "label.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.draw_rect(pymupdf.Rect(40, 92, 92, 104), color=None, fill=(0, 0, 0))
    page.insert_text((42, 102), "Figure 2.4", fontsize=9, fontname="hebo", color=(1, 1, 1))
    page.insert_text((98, 102), "Some selected ASCII codes for English letters.", fontsize=9)
    doc.save(src)
    doc.close()

    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    spans = [
        span for block in pymupdf.open(out)[0].get_text("dict")["blocks"] if block.get("type") == 0
        for line in block["lines"] for span in line["spans"] if span["text"].strip()
    ]
    text = "".join(span["text"] for span in spans)
    assert "[zh-CN]" in text and "Some selected ASCII codes" in text
    label = [span for span in spans if "Figure" in span["text"]]
    assert label and all(span["color"] == 0xFFFFFF for span in label)


def test_bibliography_without_a_heading_stays_original():
    """章末参考文献紧接着习题排，没有 References 标题。一连串「姓, 名缩写.」起头的条目保留原文；
    正文里一句这样开头的话不算。"""
    from app.formats.pdf import TextBlock, biblio_skips

    def blk(text, size=10.0, page=0):
        rect = pymupdf.Rect(40, 40, 300, 50)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("Smith, J. 2020 showed that this works, and we follow that approach here."),
        blk("2.10 Augment the minimum edit distance algorithm to output an alignment."),
        blk("Ahia, O., S. Kumar, H. Gonen, J. Kasai, D. Mortensen,", 8, 1),
        blk("N. Smith, and Y. Tsvetkov. 2023. Do all languages cost the same? EMNLP.", 8, 1),
        blk("Arkadiev, P. M. 2020. Morphology in typology. Oxford.", 8, 1),
        blk("Baayen, R. H. 2001. Word frequency distributions. Springer.", 8, 1),
        blk("Exercises", 10, 1),
        blk("Bellman, R. 1984. Eye of the Hurricane: an autobiography.", 8, 1),
        blk("World Scientific Singapore.", 8, 1),
        blk("Clark, H. H. and J. E. Fox Tree. 2002. Using uh and um in spontaneous speaking.", 8, 1),
        blk("Weizenbaum, J. 1976. Computer Power and Human Reason.", 8, 1),
        blk("From Judgement to Calculation. W.H. Freeman & Co.", 8, 1),
    ]
    assert biblio_skips(blocks) == [False, False, *([True] * 10)]


def test_left_aligned_dialogue_lines_are_not_centered():
    """对话各行左缘对齐。中线碰巧落在版心中线附近的那几行也按左对齐写回。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry

    def blk(text, y0, x1):
        rect = pymupdf.Rect(221, y0, x1, y0 + 9)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=9, color="#000", bold=False)

    first = blk("User: I need some help, that much seems certain.", 189, 453)
    second = blk("ELIZA: WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP", 200, 471)
    geo = PageGeometry.from_rects(612, [first.rect, second.rect], height=792,
                                  frame=TextFrame(160, 493, 111, 719, 10))
    rect, align = _write_rect(first, [(first, ""), (second, "")], True, False, geo)
    assert align == "left" and rect.x0 == first.rect.x0


def test_large_chapter_number_stays_apart_from_the_title():
    """章首 60pt 的数字和 25pt 的标题之间隔着一个大字宽：数字原样留着，标题单独写回原位。
    同字号的 2.2 Training 仍是一行。"""
    from app.formats.pdf import _split_lines

    def item(text, rect, size):
        span = {"text": text, "size": size, "font": "Times", "flags": 0}
        return (text, pymupdf.Rect(rect), [span])

    chapter = [item("2", (98, 142, 128, 202), 59.8), item("Words and Tokens", (167, 142, 364, 166), 24.8)]
    assert _split_lines(chapter) == [[0], [1]]
    section = [item("2.2", (40, 100, 60, 112), 12), item("Training", (70, 100, 130, 112), 12)]
    assert _split_lines(section) == [[0, 1]]


def test_pdf_page_lines_and_tables_are_read_once_per_page(tmp_path, monkeypatch):
    """找表要把整页的字解析一遍，矢量图、图片框每读一次也要几毫秒。
    提取、阅读顺序、角色判断共用一次结果。写回译文时另读，不在这里数。"""
    from app.formats import pdf_layout
    from app.formats.pdf import _extract

    calls: dict[str, list[int]] = {"tables": [], "drawings": [], "images": []}
    tables, drawings, images = pdf_layout.page_tables, pymupdf.Page.get_drawings, pymupdf.Page.get_image_info

    def counted_tables(page, *args):
        calls["tables"].append(page.number)
        return tables(page, *args)

    def counted_drawings(self, *args, **kwargs):
        calls["drawings"].append(self.number)
        return drawings(self, *args, **kwargs)

    def counted_images(self, *args, **kwargs):
        calls["images"].append(self.number)
        return images(self, *args, **kwargs)

    monkeypatch.setattr(pdf_layout, "page_tables", counted_tables)
    monkeypatch.setattr(pymupdf.Page, "get_drawings", counted_drawings)
    monkeypatch.setattr(pymupdf.Page, "get_image_info", counted_images)
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "A body paragraph long enough to be translated here.", fontsize=10)
    _extract(doc)
    assert {key: sorted(pages) for key, pages in calls.items()} == {key: [0, 1, 2] for key in calls}


def test_union_box_matches_rect_union():
    from app.formats.pdf import _union_box

    rects = [
        pymupdf.Rect(378.9772014701512, 294.40010658, 391.92303898, 300.53540323),
        pymupdf.Rect(391.8992945173863, 212.31890825, 415.72914222, 219.31949272),
        pymupdf.Rect(5, 5, 5, 9),  # 零宽的空框不算
        pymupdf.Rect(9, 9, 3, 3),
        pymupdf.Rect(pymupdf.INFINITE_RECT()),
    ]
    for a in rects:
        for b in rects:
            assert tuple(_union_box(a, b)) == tuple(a | b)


def test_pdf_reports_its_preparation_and_writing_stages(tmp_path):
    """整本书的版面分析、写回各要几分钟。逐页报做到哪一页，任务页才不像卡住。"""
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "A body paragraph long enough to be translated here.", fontsize=10)
    src = tmp_path / "pages.pdf"
    doc.save(src)
    stages: list[tuple] = []
    r = runner()
    original = r.report_stage

    def record(name, done=0, total=0, unit=""):
        stages.append((name, done, total, unit))
        original(name, done, total, unit)

    r.report_stage = record
    run(translate_pdf(src, tmp_path, r, False, "zh-CN"))
    assert stages[:4] == [("分析版面", n, 3, "页") for n in (1, 2, 3)] + [("整理段落", 0, 0, "")]
    assert stages[4][0] == "准备译文" and stages[4][1] == 0 and stages[4][3] == "段"
    # 开始翻译时清掉准备阶段；译文到齐后逐页写回
    after = stages[stages.index(("", 0, 0, "")):]
    assert [stage for stage in after if stage[0] == "写回译文"] == [("写回译文", n, 3, "页") for n in (0, 1, 2)]
    assert after[-1] == ("生成文件", 0, 0, "")



def test_the_layout_model_reports_each_page_it_reads(tmp_path, monkeypatch):
    """版面模型一页推两秒左右，在定角色前逐页推好、按页报进度，推完回到「整理段落」"""
    import numpy as np

    from app.formats import pdf, pdf_vision

    class Meta:
        custom_metadata_map = {"names": "{0: 'title'}"}

    class Input:
        name = "images"

    class NoBoxes:
        def __init__(self):
            self.runs = 0

        def get_modelmeta(self):
            return Meta()

        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, _feeds):
            self.runs += 1
            return [np.zeros((1, 0, 6), np.float32)]

    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "A body paragraph long enough to be translated here.", fontsize=10)
    doc.new_page(width=420, height=240)  # 没有字的页不推
    src = tmp_path / "pages.pdf"
    doc.save(src)
    session = NoBoxes()
    monkeypatch.setattr(pdf, "layout_detector", lambda doc, path, boxes=None: pdf_vision.ModelLayout(doc, session))
    stages: list[tuple] = []

    class R:
        def report_stage(self, name, done=0, total=0, unit=""):
            stages.append((name, done, total, unit))

        def set_title(self, title):
            pass

    pdf._plan_pdf(src, R(), "layout.onnx")
    start = stages.index(("版面模型分析", 1, 3, "页"))
    assert stages[start:start + 4] == [("版面模型分析", n, 3, "页") for n in (1, 2, 3)] + [("整理段落", 0, 0, "")]
    assert session.runs == 3


def test_reanalysis_after_a_code_change_reuses_the_model_boxes(tmp_path, monkeypatch):
    """改了代码、存下的分析结果作废时重新分析：两个模型的框从 boxes.json 读回，不再推，结果和第一次一样"""
    import numpy as np

    from app.formats import pdf, pdf_vision
    from app.formats.pdf_vision import FormulaModel

    class Meta:
        def __init__(self, names):
            self.custom_metadata_map = {"names": names}

    class Input:
        name = "images"

    runs = {"mfd": 0, "layout": 0}

    class Formula:
        """在 Q7 上框一个行内公式（中心、宽高、两类的分，没做 NMS）"""

        def __init__(self, page):
            self.page = page

        def get_modelmeta(self):
            return Meta("{0: 'embedding', 1: 'isolated'}")

        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, feeds):
            runs["mfd"] += 1
            height, width = feeds["images"].shape[2:]
            zoom = max(height, width) / max(self.page.rect.width, self.page.rect.height)
            left = (width - round(self.page.rect.width * zoom)) // 2
            top = (height - round(self.page.rect.height * zoom)) // 2
            r = self.page.search_for("Q7")[0]
            x0, y0, x1, y1 = r.x0 * zoom + left, r.y0 * zoom + top, r.x1 * zoom + left, r.y1 * zoom + top
            return [np.array([[[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.9, 0.0]]], np.float32).transpose(0, 2, 1)]

    class Layout:
        def get_modelmeta(self):
            return Meta("{0: 'title'}")

        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, _feeds):
            runs["layout"] += 1
            return [np.zeros((1, 0, 6), np.float32)]

    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "We set Q7 to the value found in the previous section here.", fontsize=10)
    src = tmp_path / "book.pdf"
    doc.save(src)
    for name in ("mfd.onnx", "layout.onnx"):
        (tmp_path / name).write_bytes(b"x")
    shown = pymupdf.open(src)[0]
    monkeypatch.setattr(pdf_vision, "_usable_sessions",
                        lambda path, *args, **kwargs: [Formula(shown) if path.endswith("mfd.onnx") else Layout()])

    class R:
        def report_stage(self, *args, **kwargs):
            pass

        def set_title(self, title):
            pass

    plan_path = tmp_path / "plan.pickle"
    model = FormulaModel(str(tmp_path / "mfd.onnx"))
    first = pdf._saved_plan(src, R(), str(tmp_path / "layout.onnx"), plan_path, model)
    assert runs == {"mfd": 2, "layout": 2} and (tmp_path / "boxes.json").is_file()
    assert any(f.get("text") == "Q7" for formulas in first.formulas for f in formulas)
    monkeypatch.setattr(pdf, "_code_version", lambda: "changed")
    again = pdf._saved_plan(src, R(), str(tmp_path / "layout.onnx"), plan_path, model)
    assert runs == {"mfd": 2, "layout": 2}
    assert again.texts == first.texts and again.skips == first.skips
    assert [[f.get("text") for f in fs] for fs in again.formulas] == [[f.get("text") for f in fs] for fs in first.formulas]

def test_pdf_jobs_take_turns_with_mupdf_and_keep_the_loop_running(tmp_path, monkeypatch):
    """同时跑的两个任务不能一起用 MuPDF（find_tables 会改进程全局的字框开关）；
    用 MuPDF 的部分在线程里做，事件循环一直能响应别的请求。"""
    import threading
    import time

    from app.formats import pdf

    src = tmp_path / "a.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A sentence to translate.", fontsize=11)
    doc.save(src)

    inside = most = 0
    stages: list[tuple] = []
    # 在事件循环所在的主线程里调用的次数
    on_loop: list[str] = []

    def counted(real):
        def wrapper(*args, **kwargs):
            nonlocal inside, most
            if threading.current_thread() is threading.main_thread():
                on_loop.append(real.__name__)
            inside += 1
            most = max(most, inside)
            try:
                time.sleep(0.2)
                return real(*args, **kwargs)
            finally:
                inside -= 1
        return wrapper

    real_geometry = pdf.PageGeometry

    # 以前在事件循环里逐页算
    class SlowGeometry(real_geometry):
        @classmethod
        def from_page(cls, *args, **kwargs):
            if threading.current_thread() is threading.main_thread():
                on_loop.append("from_page")
            return real_geometry.from_page(*args, **kwargs)

    monkeypatch.setattr(pdf, "_extract", counted(pdf._extract))
    monkeypatch.setattr(pdf, "_render_by_parts", counted(pdf._render_by_parts))
    monkeypatch.setattr(pdf, "PageGeometry", SlowGeometry)

    async def scenario():
        for name in "ab":
            (tmp_path / name).mkdir()
        runners = [runner() for _ in "ab"]
        for r in runners:
            r.report_stage = lambda name, done=0, total=0, unit="": stages.append((name, done, total, unit))
        return await asyncio.gather(*(
            pdf.translate_pdf(src, tmp_path / name, r, False, "zh-CN") for name, r in zip("ab", runners)
        ))

    outs = run(scenario())
    # 后到的任务等的时候任务页上显示在等
    assert ("等待其他任务", 0, 0, "") in stages
    # 只看在哪个线程里调用，不量循环停了多久：机器忙的时候线程里的调用也会让循环停上零点几秒
    assert on_loop == []
    assert most == 1
    assert [pymupdf.open(out)[0].get_text() for [out] in outs] == [pymupdf.open(outs[0][0])[0].get_text()] * 2


def test_pdf_keeps_lines_the_model_returns_unchanged(tmp_path):
    """送翻时 TeX 的 `u 合成了 ù。模型原样返回时这行不算译过，留原字，不按合成后的文字重排。"""
    from app.formats.pdf import translate_pdf

    class Echo(MockTranslator):
        async def translate_batch(self, texts):
            return list(texts)

    src = tmp_path / "codes.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 60), "U+00F9 `u LATIN SMALL LETTER U WITH GRAVE", fontsize=10)
    doc.save(src)
    before = pymupdf.open(src)[0].get_text()
    assert "`u" in before

    [out] = run(translate_pdf(src, tmp_path, Runner(Echo("zh-CN"), cache=MemCache()), False, "zh-CN"))
    assert pymupdf.open(out)[0].get_text() == before


def test_light_cell_shading_is_not_a_formula_stroke():
    """浅色格子底色压在字下面，不并进公式框；图注标签的黑底和分式线照收。"""
    from app.formats.pdf import _page_curve_rects

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 35), color=None, fill=(0.77, 0.86, 0.99))
    page.draw_rect(pymupdf.Rect(80, 20, 120, 35), color=None, fill=(0, 0, 0))
    page.draw_rect(pymupdf.Rect(140, 30, 180, 30.6), color=None, fill=(0, 0, 0))
    rects = _page_curve_rects(page)
    assert [round(r.x0) for r in rects] == [80, 140]


def test_shaded_panel_of_monospace_text_is_kept_whole():
    """底色上全是等宽字的一片（码位表、代码）留原字。txtt 名字里没有 Mono，靠等宽标志认。
    没有底色、或者底色上夹着正文字体的，照翻；汉字也带等宽标志，不算。"""
    from app.formats.pdf import TextBlock, _code_panel_ids
    from app.formats.pdf_layout import PageScan

    def block(y, *spans):
        rect = pymupdf.Rect(160, y, 380, y + 10)
        span_lines = [[{"text": text, "font": font, "flags": flags, "size": 10.0, "bbox": tuple(rect)}
                       for text, font, flags in spans]]
        return TextBlock(0, rect, [rect], "".join(text for text, _f, _g in spans), 10.0, "#000000", False, span_lines)

    table = [block(372, ("U+0061 a LATIN SMALL LETTER A", "txtt", 8)),
             block(384, ("GRINNING FACE", "txtt", 8))]
    plain = block(600, ("Men are all alike.", "txtt", 8))
    mixed = [block(452, ("/woodchuck/", "txtt", 8)), block(464, ("interesting links to woodchucks", "NimbusRomNo9L-Regu", 4))]
    chinese = block(532, ("进远违连", "gbsnu8f", 8))
    shades = (pymupdf.Rect(157, 370, 411, 396), pymupdf.Rect(157, 450, 411, 476), pymupdf.Rect(157, 530, 411, 544))
    found = _code_panel_ids([*table, plain, *mixed, chinese], [PageScan((), (), (), shades)])
    assert found == {id(b) for b in table}


def test_shaded_areas_join_cells_and_leave_out_page_background():
    """一格一块底色、格间留缝的表并成一片；铺满整页的纸张底色不算。"""
    from app.formats.pdf_layout import scan_page

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(page.rect, color=None, fill=(0.98, 0.96, 0.9))
    for row in range(3):
        for x0, x1 in ((20, 60), (60.5, 150)):
            page.draw_rect(pymupdf.Rect(x0, 40 + row * 12.5, x1, 52 + row * 12.5), color=None, fill=(0.77, 0.86, 0.99))
    assert [tuple(round(v) for v in area) for area in scan_page(page).shades] == [(20, 40, 150, 77)]


def test_resumed_pdf_job_reads_back_its_layout_analysis(tmp_path, monkeypatch):
    """继续翻译时原文件和代码都没变，读回上次的版面分析，不再逐页分析；写出来的和第一次一样。
    原文件变了就重新分析。"""
    from app.formats import pdf
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((40, 50), "Saved Analysis", fontsize=16)
    page.insert_text((40, 90), "A body paragraph long enough to be translated here.", fontsize=10)
    page.insert_text((40, 130), "for i in range(10): print(i)", fontsize=10, fontname="cour")
    src = tmp_path / "book.pdf"
    doc.save(src)
    saved = tmp_path / "plan.pickle"
    calls = []
    real_extract = pdf._extract
    monkeypatch.setattr(pdf, "_extract", lambda *a, **k: calls.append(1) or real_extract(*a, **k))

    def translate(out):
        out.mkdir()
        r = runner()
        r.translator.title = ""
        stages = []
        r.report_stage = lambda name, *a: stages.append(name)
        [path] = run(translate_pdf(src, out, r, False, "zh-CN", plan_path=saved))
        return pymupdf.open(path), r.translator.title, stages

    first, title, _ = translate(tmp_path / "a")
    assert saved.exists() and len(calls) == 1
    again, title_again, stages = translate(tmp_path / "b")
    assert len(calls) == 1 and "分析版面" not in stages and "读取上次的版面分析结果" in stages
    assert title_again == title == "book"
    assert again[0].get_text() == first[0].get_text()
    assert "for i in range(10)" in again[0].get_text()
    assert again[0].get_pixmap(dpi=72).samples == first[0].get_pixmap(dpi=72).samples

    doc[0].insert_text((40, 170), "Another paragraph added to the source file.", fontsize=10)
    doc.save(src)
    translate(tmp_path / "c")
    assert len(calls) == 2


def test_styles_carry_over_line_breaks():
    """断开的每一行自带标记：跨行的粗体、字号在下一行开头重开，本行末尾收起。"""
    from app.formats.pdf_flow import break_lines

    lines = break_lines("前面的字{z80}{b}这一段很长的小字号粗体文字要跨过换行{/b}{/z}后面", 10, 100, {})
    assert len(lines) >= 2
    assert all(line.count("{b}") == line.count("{/b}") == 1 for line in lines[:-1])
    assert lines[1].startswith("{z80}{b}")


def test_cjk_line_keeps_bold_across_a_formula():
    """公式把一行切成几片，公式后面那片接着用前面开着的粗体，收尾以后回到常规。"""
    import pymupdf

    from app.formats.pdf_align import _place_lines
    from app.formats.pdf_typeset import FormulaMetric, typeset_lines

    text = "前面{b}粗体\x01i\x021\x01/i\x02还是粗体{/b}后面"
    lines = typeset_lines(text, 10, 400, {1: FormulaMetric(10, 10, 8, 2)})
    commands, _slots, _fonts = _place_lines(
        lines, pymupdf.Rect(0, 0, 400, 100), 10, 1.4, "#000000", "left", frozenset({1}), {}, 800,
    )
    stroked, bold = [], False
    for op in commands.splitlines():
        if op == "2 Tr":
            bold = True
        elif op == "0 Tr":
            bold = False
        elif op.endswith("TJ") or op.endswith("Tj"):
            stroked.append(bold)
    assert stroked == [False, True, True, False]


def test_cjk_lines_rise_to_stay_inside_a_tight_first_line_box():
    """原文第一行的行框比汉字字形矮时整段往上提，字形下沿不压到紧贴在下面的线；
    最多提到字形上沿和盒子顶齐平。行框够高时不动。"""
    import pymupdf

    from app.formats.pdf_align import FIRST_BASELINE, _place_lines
    from app.formats.pdf_typeset import typeset_lines

    lines = typeset_lines("用于生成适合人类阅读的解答的提示词。", 10, 400, {})

    def baseline(first_room):
        _commands, slots, _fonts = _place_lines(
            lines, pymupdf.Rect(0, 100, 400, 200), 10, 1.4, "#000000", "left", frozenset(), {}, 800,
            first_room=first_room,
        )
        commands = _commands.splitlines()
        tm = next(op for op in commands if op.endswith(" Tm"))
        return 800 - float(tm.split()[5])

    assert baseline(None) == pytest.approx(100 + 10 * FIRST_BASELINE, abs=0.01)
    assert baseline(13.0) == pytest.approx(100 + 10 * FIRST_BASELINE, abs=0.01)
    assert baseline(11.0) == pytest.approx(100 + 11.0 - 10 * 0.12, abs=0.01)
    assert baseline(5.0) == pytest.approx(100 + 10 * 0.88, abs=0.01)


def test_inline_colors_are_marked_relative_to_the_block():
    """和块同色的不写标记；别的颜色写 {cRRGGBB}，同色的几个词中间的空格不断开；白字（图注标签）不写。
    颜色在字号里面、强调外面，标记成对嵌套。"""
    from app.formats.pdf_flow import Emphasis, writer_for

    plain = Emphasis(False, False)
    spans = [
        {"text": "We cite ", "color": 0, "size": 10},
        {"text": "Clark", "color": 0x0000FF, "size": 10},
        {"text": " ", "color": 0, "size": 10},
        {"text": "2002", "color": 0x0000FF, "size": 10},
        {"text": " here.", "color": 0, "size": 10},
        {"text": "Figure 2.1", "color": 0xFFFFFF, "size": 10},
    ]
    writer = writer_for(spans, 10, 0)
    text = "".join(writer.text(s["text"], plain, s["size"], s["color"]) for s in spans) + writer.close()
    assert text == "We cite {c0000ff}Clark 2002{/c} here.Figure 2.1"

    same = [{"text": "All black text.", "color": 0, "size": 10}]
    writer = writer_for(same, 10, 0)
    assert writer.text("All black text.", plain, 10, 0) + writer.close() == "All black text."

    writer = writer_for([{"text": "abc", "color": 0, "size": 10}, {"text": "x", "color": 0xFF0000, "size": 6}], 10, 0)
    out = writer.text("abc ", Emphasis(True, False), 10, 0) + writer.text("x", Emphasis(True, False), 6, 0xFF0000)
    assert out + writer.close() == "{b}abc {/b}{z60}{cff0000}{b}x{/b}{/c}{/z}"


def test_colors_carry_over_line_breaks():
    """断开的每一行自带标记：跨行的颜色和里面的粗体在下一行开头重开，本行末尾收起。"""
    from app.formats.pdf_flow import break_lines

    lines = break_lines("前面的字{c0000ff}{b}这一段很长的蓝色粗体链接文字要跨过换行{/b}{/c}后面", 10, 100, {})
    assert len(lines) >= 2
    assert all(line.count("{c0000ff}") == line.count("{/c}") == 1 for line in lines[:-1])
    assert lines[1].startswith("{c0000ff}{b}")


def test_pdf_keeps_inline_colors_in_the_translation(tmp_path):
    """正文里的蓝色引文、红色字串译过去还是原来的颜色，跨行的部分也是；其余的字是块的颜色。
    模型弄乱了颜色标记（少了收尾）时整段按块的颜色写，不把后半段染色。"""
    from app.formats.pdf import translate_pdf

    long_blue = "这是一段很长很长的蓝色链接文字需要跨过换行才能写完" * 3

    class Colored(MockTranslator):
        async def translate_batch(self, texts):
            out = []
            for text in texts:
                if "{c0000ff}" not in text:
                    out.append(text)
                elif "broken" in text:
                    out.append("前面的字{c0000ff}没有收尾的蓝色，后面的字也会被染成蓝色吗")
                else:
                    out.append("开头的一些普通文字，" + "{c0000ff}" + long_blue + "{/c}" + "，以及{cff0000}红字{/c}结尾。")
            return out

    src = tmp_path / "colors.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    for y, middle in ((60, "Weizenbaum (1966)"), (140, "broken link")):
        x = 40
        for text, color in (("We cite ", (0, 0, 0)), (middle, (0, 0, 1)), (" and ", (0, 0, 0)),
                            ("xxxxxx", (1, 0, 0)), (" in this body text.", (0, 0, 0))):
            page.insert_text((x, y), text, fontsize=10, color=color)
            x += pymupdf.get_text_length(text, fontsize=10)
    doc.save(src)

    [out] = run(translate_pdf(src, tmp_path, Runner(Colored("zh-CN"), cache=MemCache()), False, "zh-CN"))
    spans = [s for b in pymupdf.open(out)[0].get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
    blue = "".join(s["text"] for s in spans if s["color"] == 0x0000FF)
    red = "".join(s["text"] for s in spans if s["color"] == 0xFF0000)
    assert blue.replace(" ", "") == long_blue
    assert red == "红字"
    assert "没有收尾的蓝色" in "".join(s["text"] for s in spans if s["color"] == 0)


def test_cjk_line_keeps_colors_across_a_formula():
    """公式把一行切成几片，公式后面那片接着用前面开着的颜色，收尾以后回到块的颜色。"""
    import pymupdf

    from app.formats.pdf_align import _place_lines
    from app.formats.pdf_typeset import FormulaMetric, typeset_lines

    text = "前面{c0000ff}蓝色\x01i\x021\x01/i\x02还是蓝色{/c}后面"
    lines = typeset_lines(text, 10, 400, {1: FormulaMetric(10, 10, 8, 2)})
    commands, _slots, _fonts = _place_lines(
        lines, pymupdf.Rect(0, 0, 400, 100), 10, 1.4, "#000000", "left", frozenset({1}), {}, 800,
    )
    painted, now = [], None
    for op in commands.splitlines():
        if op.endswith(" rg"):
            now = op
        elif op.endswith("TJ") or op.endswith("Tj"):
            painted.append(now)
    blue, black = "0.0000 0.0000 1.0000 rg", "0.0000 0.0000 0.0000 rg"
    assert painted == [black, blue, blue, black]


def test_toc_entries_send_only_titles_and_keep_page_numbers(tmp_path):
    """目录条目只送标题：节号、点线和页码不送翻，页码留在原处靠右对齐，点线补到译文末尾。
    点和标题排在同一个 span 里的、没有点线只有页码的（卷名一行）也一样；每条单独成段。"""
    from app.formats.pdf import translate_pdf

    sent: list[str] = []

    class Short(MockTranslator):
        async def translate_batch(self, texts):
            sent.extend(texts)
            return ["标题" + str(index) for index, _ in enumerate(texts)]

    src = tmp_path / "toc.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((40, 50), "Part One: Foundations of Language", fontsize=12)
    page.insert_text((360, 50), "1", fontsize=12)
    rows = (("1.1 What is a Large Language Model", "5"), ("1.2 Tokens", "8"), ("Historical Notes", "12"))
    for row, (title, number) in enumerate(rows):
        y = 70 + row * 14
        page.insert_text((50, y), title, fontsize=10)
        x = 50 + pymupdf.get_text_length(title, fontsize=10) + 6
        while x < 345:
            page.insert_text((x, y), ".", fontsize=10)
            x += 5
        page.insert_text((360, y), number, fontsize=10)
    # 点和标题在同一个 span 里
    page.insert_text((50, 112), "Exercises" + " ." * 45, fontsize=10)
    page.insert_text((360, 112), "13", fontsize=10)
    doc.save(src)
    before = {w[4]: (round(w[0], 1), round(w[1], 1)) for w in pymupdf.open(src)[0].get_text("words") if w[4].isdigit()}

    [out] = run(translate_pdf(src, tmp_path, Runner(Short("zh-CN"), cache=MemCache()), False, "zh-CN"))
    assert sorted(sent) == sorted([
        "Part One: Foundations of Language", "What is a Large Language Model", "Tokens",
        "Historical Notes", "Exercises",
    ])
    result = pymupdf.open(out)[0]
    words = result.get_text("words")
    after = {w[4]: (round(w[0], 1), round(w[1], 1)) for w in words if w[4].isdigit()}
    assert after == before
    assert not any(w[4] in ("Exercises", "Notes", "Tokens") for w in words)
    # 每条的点线从译文末尾接到页码前，中间不空出一截
    for line in [l for b in result.get_text("dict")["blocks"] for l in b.get("lines", [])]:
        text = "".join(s["text"] for s in line["spans"])
        if text.startswith("标题") and "." in text:
            title = next(s for s in line["spans"] if s["text"].startswith("标题"))
            dots = [s for s in line["spans"] if s["text"].strip(" .") == "" and "." in s["text"]]
            assert dots and dots[0]["bbox"][0] - title["bbox"][2] < 8


def _glyph(text, font, size, flags, bbox, ink, math=False):
    span = {"text": text, "font": font, "size": size, "flags": flags, "bbox": bbox, "ink": ink,
            "origin": (bbox[0], ink[3])}
    if math:
        span["math"] = True
    return span


def test_a_formula_at_a_line_start_keeps_its_opening_glue():
    """下一行行首的 P( 是这一行式子的开头。上一行式子里 CMSY 减号的字体框伸到这一行，
    按字体框比会把 P( 挂到上一行的 w_{n-2}w_{n-1}w_n 上，两个式子各缺一截。"""
    from app.formats.pdf import _placeholderize

    reg, ital, cmsy, cmr = "NimbusRomNo9L-Regu", "NimbusRomNo9L-ReguItal", "CMSY10", "CMR10"
    above = [
        _glyph("examples of a particular trigram ", reg, 10.0, 4, (142.2, 191.2, 306.7, 203.2), (142.2, 193.6, 306.7, 200.6)),
        _glyph("w", ital, 10.0, 6, (309.7, 191.2, 316.3, 203.1), (309.7, 196.0, 316.3, 200.6)),
        _glyph("n", ital, 7.4, 6, (316.3, 195.1, 320.0, 203.9), (316.3, 198.7, 320.0, 202.0)),
        _glyph("−", cmsy, 7.4, 6, (320.0, 196.2, 325.8, 209.0), (320.0, 200.0, 325.8, 200.3)),
        _glyph("2", reg, 7.4, 4, (325.8, 195.1, 329.4, 204.0), (325.8, 197.0, 329.4, 201.9)),
        _glyph("w", ital, 10.0, 6, (329.9, 191.2, 336.6, 203.1), (329.9, 196.0, 336.6, 200.6)),
        _glyph("n", ital, 7.4, 6, (336.6, 195.1, 340.3, 203.9), (336.6, 198.7, 340.3, 202.0)),
        _glyph("−", cmsy, 7.4, 6, (340.3, 196.2, 346.0, 209.0), (340.3, 200.0, 346.0, 200.3)),
        _glyph("1", reg, 7.4, 4, (346.0, 195.1, 349.7, 204.0), (346.0, 197.0, 349.7, 201.9)),
        _glyph("w", ital, 10.0, 6, (350.2, 191.2, 356.8, 203.1), (350.2, 196.0, 356.8, 200.6)),
        _glyph("n", ital, 7.4, 6, (356.8, 195.1, 360.5, 203.9), (356.8, 198.7, 360.5, 202.0)),
        _glyph(", we can instead estimate its", reg, 10.0, 4, (361.0, 191.2, 475.2, 203.2), (361.0, 193.6, 475.2, 201.8)),
    ]
    below = [
        _glyph("probability by using the bigram probability", reg, 10.0, 4, (142.2, 203.2, 313.8, 215.2), (142.2, 205.6, 313.8, 214.6)),
        _glyph(" P", ital, 10.0, 6, (313.8, 203.2, 322.3, 215.1), (313.8, 205.9, 322.3, 212.4)),
        _glyph("(", cmr, 10.0, 4, (322.3, 204.9, 326.2, 214.9), (322.3, 204.9, 326.2, 214.9), math=True),
        _glyph("w", ital, 10.0, 6, (326.2, 203.2, 332.9, 215.1), (326.2, 208.0, 332.9, 212.6)),
        _glyph("n", ital, 7.4, 6, (332.9, 207.1, 336.5, 215.9), (332.9, 210.6, 336.5, 214.0)),
        _glyph("|", cmsy, 10.0, 6, (337.0, 204.7, 339.8, 222.0), (337.0, 204.9, 339.8, 214.9)),
        _glyph("w", ital, 10.0, 6, (339.8, 203.2, 346.4, 215.1), (339.8, 208.0, 346.4, 212.6)),
        _glyph("n", ital, 7.4, 6, (346.4, 207.1, 350.1, 215.9), (346.4, 210.7, 350.1, 214.0)),
        _glyph("−", cmsy, 7.4, 6, (350.1, 208.2, 355.9, 221.0), (350.1, 211.9, 355.9, 212.2)),
        _glyph("1", reg, 7.4, 4, (355.9, 207.1, 359.5, 216.0), (355.9, 208.9, 359.5, 213.9)),
        _glyph(")", cmr, 10.0, 4, (360.0, 204.9, 363.9, 214.9), (360.0, 204.9, 363.9, 214.9), math=True),
        _glyph(". Similarly, if we don’t have", reg, 10.0, 4, (363.9, 203.2, 475.2, 215.2), (363.9, 205.6, 475.2, 214.6)),
    ]
    sent, formulas = _placeholderize(_block_with_spans([above, below], size=10.0))
    assert len(formulas) == 2
    assert "P" not in formulas[0]["text"]
    assert formulas[1]["text"].startswith("P(")
    assert "bigram probability" in sent and sent.endswith("{v2}. Similarly, if we don’t have")


def test_a_denominator_split_into_the_next_lines_block_rejoins_its_fraction():
    """count(w)/N 的分母 N 被分进下一行那个块。整块都是 Times 斜体，按字体认不出是式子，
    要按小一号的斜体居中叠在正文行的小斜体上认出来，挪回分子那一行。上面是正文字号的斜体、
    或者只是左缘对齐的上下标时不挪。"""
    from app.formats.pdf import _rejoin_split_lines

    reg, ital, cmr = "NimbusRomNo9L-Regu", "NimbusRomNo9L-ReguItal", "CMR10"

    def line(*spans):
        box = pymupdf.Rect(spans[0]["bbox"])
        for span in spans[1:]:
            box |= pymupdf.Rect(span["bbox"])
        return {"bbox": tuple(box), "dir": (1, 0), "spans": list(spans)}

    def data(numerator):
        score = line(
            _glyph("The backoff terminates in the unigram, which has score", reg, 10.0, 4,
                   (160.2, 307.0, 378.1, 319.1), (160.2, 309.4, 378.1, 316.4)),
            _glyph(" S", ital, 10.0, 6, (378.1, 307.0, 385.2, 318.9), (378.1, 309.6, 385.2, 316.4)),
            _glyph(") =", cmr, 10.0, 4, (395.8, 308.8, 409.2, 318.7), (395.8, 308.8, 409.2, 318.7), math=True),
            _glyph("count", ital, numerator, 7, (412.2, 304.5, 428.4, 313.3), (412.2, 307.3, 428.4, 311.4)),
            _glyph("(", cmr, numerator, 5, (428.9, 305.8, 431.8, 313.2), (428.9, 305.8, 431.8, 313.2), math=True),
            _glyph("w", ital, numerator, 7, (431.8, 304.5, 436.7, 313.3), (431.8, 308.1, 436.7, 311.5)),
            _glyph(")", cmr, numerator, 5, (436.7, 305.8, 439.6, 313.2), (436.7, 305.8, 439.6, 313.2), math=True),
        )
        denominator = line(_glyph("N", ital, 7.4, 6, (423.2, 312.9, 428.1, 321.7), (423.2, 314.9, 428.1, 319.8)))
        tail = line(_glyph(". Brants et al.", reg, 10.0, 4, (440.7, 307.0, 493.2, 319.1), (440.7, 309.4, 493.2, 316.4)))
        rest = line(_glyph("(2007) find that a value of 0.4 worked well", reg, 10.0, 4,
                           (160.2, 319.0, 344.6, 331.0), (160.2, 321.4, 344.6, 328.4)))
        return {"blocks": [{"type": 0, "bbox": score["bbox"], "lines": [score]},
                           {"type": 0, "bbox": (160.2, 307.0, 493.2, 331.0), "lines": [denominator, tail, rest]}]}

    split = data(7.4)
    score, (denominator, tail, rest) = split["blocks"][0]["lines"][0], split["blocks"][1]["lines"]
    _rejoin_split_lines(split)
    assert split["blocks"][0]["lines"] == [score, denominator, tail]
    assert split["blocks"][1]["lines"] == [rest]

    # 上面是正文字号的斜体单词，不是分子：小字不挪
    words = data(10.0)
    small = words["blocks"][1]["lines"][0]
    _rejoin_split_lines(words)
    assert small in words["blocks"][1]["lines"]

    # w^i_{i-k+1} 的上标和下标左缘对齐，不是分式：不挪
    script = data(7.4)
    sup = script["blocks"][1]["lines"][0]
    sup["spans"][0]["bbox"], sup["spans"][0]["ink"] = (412.2, 312.9, 417.1, 321.7), (412.2, 314.9, 417.1, 319.8)
    sup["bbox"] = sup["spans"][0]["bbox"]
    _rejoin_split_lines(script)
    assert sup in script["blocks"][1]["lines"]


def test_a_citation_year_after_et_al_continues_the_sentence():
    """Brants et al. | (2007) find that… 是同一句：et al. 的点不是句末，左括号起头的是引文年份。
    (a) 这样的编号是下一项，(2.11) 是例句的编号，缩进排的是单独的式子，句号收住的上一行也不接。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, x1=493.0):
        rect = pymupdf.Rect(160, y, x1, y + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    cite = [blk("The backoff terminates in the unigram, which has score S(w). Brants et al.", 307),
            blk("(2007) find that a value of 0.4 worked well for λ.", 319, x1=356)]
    assert len(_merge_continuations(cite)) == 1
    item = [blk("the model is trained in two stages, described below", 307),
            blk("(a) pretraining on unlabeled text", 319, x1=320)]
    assert len(_merge_continuations(item)) == 2
    indented = [blk("It could answer questions by parsing them into a logical form", 307),
                TextBlock(page=0, rect=pymupdf.Rect(182, 319, 470, 343),
                          line_rects=[pymupdf.Rect(182, 319, 470, 331), pymupdf.Rect(196, 331, 440, 343)],
                          text="(TEST (FOR SOME X16 / (SEQ SAMPLES) : T ; (CONTAIN’ X16 (NPR* X17 / (QUOTE AL203))",
                          size=10.0, color="#000", bold=False)]
    assert len(_merge_continuations(indented)) == 2
    tag = [blk("where we’ve explicitly marked the spaces between words: _", 307),
           blk("(2.11) set new new renew reset renew", 319, x1=330)]
    assert len(_merge_continuations(tag)) == 2
    ended = [blk("This algorithm does not produce a probability distribution.", 307),
             blk("(Unlike interpolation, the weights are fixed.)", 319, x1=380)]
    assert len(_merge_continuations(ended)) == 2


def test_block_color_follows_most_characters_not_the_longest_span():
    """逐词一个 span 的书里，最长的一截是黑底白字的图注标签。按它取颜色，整条图注的译文写成白字。"""
    from app.formats.pdf import _by_chars

    spans = [{"text": "Figure 10.3", "color": 0xFFFFFF}] + [
        {"text": word, "color": 0} for word in ("The", "all-words", "WSD", "task,", "mapping", "from", "input")
    ]
    assert _by_chars(spans, lambda span: span["color"]) == 0


def test_a_contents_page_without_leader_dots_is_found_by_its_page_numbers():
    """没有点线的目录：标题 Table of Contents，行尾隔得很远、右缘对齐、往下递增的页码。
    表目录页码从头再来、接在目录页后面没有标题的续页也算。不在目录旁边、没有标题的整数列表格不算。"""
    from app.formats.pdf import _toc_pages

    def line(text, x0, x1, y):
        return {"dir": (1, 0), "bbox": (x0, y, x1, y + 12), "spans": [{"text": text, "size": 12.0}]}

    def page(title, rows):
        lines = [line(title, 241, 371, 70)] if title else []
        for k, (entry, number) in enumerate(rows):
            y = 100 + 14.5 * k
            lines += [line(entry, 90, 90 + 6 * len(entry), y), line(str(number), 522 - 6 * len(str(number)), 522, y)]
        return {"blocks": [{"type": 0, "lines": lines}]}

    entries = [("Executive Summary", 1), ("1 Framing Risk", 4), ("1.1 Understanding Risks", 4), ("1.2 Challenges", 5),
               ("2 Audience", 9), ("3 AI Risks", 12), ("Appendix D", 42), ("Table 1 GOVERN", 22), ("Table 2 MAP", 26)]
    contents = page("Table of Contents", entries)
    following = page("", [("Figure 1 Harms", 5), ("Figure 2 Lifecycle", 10), ("Figure 3 Actors", 11),
                          ("Figure 4 Characteristics", 12), ("Figure 5 Functions", 20)])
    counts = page("", [("Romeo", 1), ("salad", 2), ("Falstaff", 4), ("forest", 12), ("battle", 21)])
    assert _toc_pages([counts, contents, following]) == {1, 2}


def test_the_short_centered_last_line_of_a_caption_continues_it():
    """居中排的表格说明末行短，两边都缩进。中线对齐、小写起头，接在上一行后面。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, x1, y):
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.9, color="#000", bold=False)

    caption = [blk("Table 3 | Experimental results at each stage of DeepSeek-R1. Numbers in bold denote the", 84.0, 510.9, 85.5),
               blk("performance is statistically significant (t-test with p < 0.01).", 153.5, 441.5, 98.2)]
    assert len(_merge_continuations(caption)) == 1
    # 中线不对齐的缩进行是别的东西
    offset = [caption[0], blk("performance is statistically significant (t-test with p < 0.01).", 200.0, 488.0, 98.2)]
    assert len(_merge_continuations(offset)) == 2
    # 只剩一个词的末行全是正文词、以句号收尾，也接上
    short = [blk("Figure 7 | The experiment results of Language Consistency (LC) Reward during reinforcement",
                 71.0, 524.0, 439.0), blk("learning.", 276.0, 319.0, 452.5)]
    assert len(_merge_continuations(short)) == 1


def test_a_line_ending_on_a_function_word_continues_into_a_capital():
    """上一行停在 and、of 上，下一行大写起头（专名）也是接着写的；下一行是另一个列表项、
    小一号的脚注时不接。居中段落大写起头的末行接上，题词下面小一号的出处不接。句点起头的接上。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, x1, y, size=10.9):
        rect = pymupdf.Rect(x0, y, x1, y + 11)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False,
                         span_lines=[[{"text": text, "font": "Palatino", "size": size}]])

    def head():
        return blk("safe answers to questions, rather than simply declining to respond). For DeepSeek-V3 and",
                   80, 515, 141)

    assert len(_merge_continuations([head(), blk("DeepSeek-R1, we report results under two configurations", 85, 510, 154)])) == 1
    assert len(_merge_continuations([head(), blk("• DeepSeek-R1 is reported under two configurations", 85, 510, 154)])) == 2
    assert len(_merge_continuations([head(), blk("Available at the project page of the authors", 85, 510, 154, size=9.0)])) == 2
    caption = blk("Table 6 | DeepSeek-R1 Distilled Models, their corresponding Base Models, and Initial Learning",
                  71, 524, 275)
    assert len(_merge_continuations([caption, blk("Rates.", 282, 313, 288)])) == 1
    poem = blk("Dim as an evening shadow in a brook,", 274, 433, 250, size=10.0)
    assert len(_merge_continuations([poem, blk("Thomas Lovell Beddoes, 1851", 298, 408, 263, size=9.0)])) == 2
    words = blk("we counted some representative reflective words, including wait, mistake, however", 71, 524, 300)
    assert len(_merge_continuations([words, blk(". These reflective words were selected by 3 experts", 71, 524, 313)])) == 1


def test_a_centered_formula_or_a_hanging_item_line_keeps_its_place():
    """段落下面居中的公式行不是居中段落的末行；悬挂缩进条目的续行以左括号起头也接上。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, x1, y, lines=None):
        rects = lines or [pymupdf.Rect(x0, y, x1, y + 12)]
        rect = pymupdf.Rect(rects[0])
        for r in rects[1:]:
            rect |= r
        return TextBlock(page=0, rect=rect, line_rects=rects, text=text, size=10.0, color="#000", bold=False)

    formula = [blk("states that if the language is regular in certain ways (to be exact, if it is both stationary and", 160, 493, 300),
               blk("lim -1/n log p(w1:n)", 270, 383, 318)]
    assert len(_merge_continuations(formula)) == 2
    # 成句的公式行也上下空着：rel(A,B) = log(max(|in(A)|,|in(B)|)) ...
    spaced = [formula[0], blk("rel(A,B) = log(max(in(A), in(B))) - log(in(A) and in(B))", 215, 438, 318)]
    assert len(_merge_continuations(spaced)) == 2
    indent = [blk("A hidden Markov model (HMM) allows us to talk about both observed events", 157, 475, 300),
              blk("(like words that we see in the input) and hidden events", 142, 475, 312)]
    assert len(_merge_continuations(indent)) == 1
    item = [blk("brand new NPs: these introduce entities that are discourse-new and hearer-old", 142, 475, 300,
                [pymupdf.Rect(142, 300, 475, 312), pymupdf.Rect(157, 312, 475, 324)]),
            blk("(like Hong Kong, Marie Curie, or the New York Times.", 157, 380, 324)]
    assert len(_merge_continuations(item)) == 1
    # 条目只有一行，没有续行可比：粗体标签起头的条目，悬挂缩进的左括号行也是它的续行
    label = blk("unused NPs: these introduce entities that are discourse-new but hearer-old", 185, 493, 656)
    label.span_lines = [[
        {"text": "unused NPs:", "font": "NimbusRomNo9L-Medi", "flags": 16},
        {"text": " ", "font": "NimbusRomNo9L-Regu", "flags": 0},
        {"text": "these introduce entities that are discourse-new but hearer-old", "font": "NimbusRomNo9L-Regu", "flags": 0},
    ]]
    single = [label, blk("(like Hong Kong, Marie Curie, or the New York Times.", 207, 425, 668)]
    assert len(_merge_continuations(single)) == 1


def test_a_block_inside_an_open_code_fence_continues_with_a_capital():
    """上一块停在没收口的代码围栏里（“‘sql 开了还没关），下一块的 JOIN 是代码的接续；围栏关上以后大写起头照常分开。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y):
        return TextBlock(page=0, rect=pymupdf.Rect(82, y, 514, y + 10), line_rects=[pymupdf.Rect(82, y, 514, y + 10)],
                         text=text, size=10.0, color="#000", bold=False)

    fenced = [blk("the query would look like this: “‘sql SELECT * FROM table1 AS t1 JOIN table2 ON t1.code = table2.code", 332),
              blk("JOIN table3 ON t1.code = table3.code; “‘ This query first selects the top 10 rows", 344)]
    assert len(_merge_continuations(fenced)) == 1
    closed = [blk("Extract the code wrapped by “‘python“‘ in the response and run the test cases", 332),
              blk("Judge whether the answer passes the test cases from the problem statement", 344)]
    assert len(_merge_continuations(closed)) == 2


def test_a_line_opening_with_an_inline_formula_continues_the_sentence():
    """以行内式子起头、从栏边起笔、后面接着正文的一行是上一句的折行；缩进单独排的式子不是。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(spans, x0, x1, y):
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        text = "".join(text for text, _font in spans)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=9.7, color="#000", bold=False,
                         span_lines=[[{"text": text, "font": font, "size": 9.7, "math": font.startswith("CM")}
                                      for text, font in spans]])

    # 正文是 Times，CMR 的 1/ 是公式（提取时标上 math）
    def head():
        return blk([("whereas SGD is low at learning rare features. Adam with", "Times-Roman")], 105, 490, 682)

    wrap = blk([("1/", "CMR10"), ("t decay on its stepsize should theoretically match", "Times-Roman")], 105, 490, 694)
    assert len(_merge_continuations([head(), wrap])) == 1
    display = blk([("x", "CMMI10"), (" = y + z for all inputs", "Times-Roman")], 160, 300, 694)
    assert len(_merge_continuations([head(), display])) == 2
    # 行首的分式碎片另成了块，剩下的从分母起头、排到右缘，下一行回到栏边
    rest = blk([("1−α", "CMR7"), (" had similar results. The first choice is optimal for", "Times-Roman")], 164, 490, 694)
    rest.line_rects = [pymupdf.Rect(164, 694, 490, 706), pymupdf.Rect(105, 706, 490, 718)]
    rest.rect = pymupdf.Rect(105, 694, 490, 718)
    assert len(_merge_continuations([head(), rest])) == 1
    # 上标编号起头的脚注字号小一号，不是折行
    note = blk([("2", "CMR6"), ("The analysis assumes that every possible cell of the data cube exists.", "Times-Roman")],
               164, 490, 694)
    note.line_rects, note.rect, note.size = rest.line_rects, rest.rect, 8.0
    assert len(_merge_continuations([head(), note])) == 2


def test_the_line_gap_is_measured_from_the_documents_body_lines():
    """行框高的字体紧排的两行叠上一点，量出来是负的。正文太少量不出来时用常见值。"""
    from app.formats.pdf import _LINE_GAP, _line_gap

    def page(count, top=100.0, height=13.3, pitch=11.0, size=10.0):
        lines = [
            {"dir": (1, 0), "bbox": (72, top + pitch * k, 520, top + pitch * k + height),
             "spans": [{"text": "words of the body text", "size": size}]}
            for k in range(count)
        ]
        return {"blocks": [{"type": 0, "lines": lines}]}

    assert _line_gap([page(30)]) == -0.25
    # 量的是正文字号的行，标题不算
    assert _line_gap([page(30), page(5, top=40, height=20, pitch=30, size=20)]) == -0.25
    assert _line_gap([page(5)]) == _LINE_GAP


def test_a_line_far_below_by_the_documents_own_spacing_is_not_a_continuation():
    """幻灯片字体行框高，紧排的两行量出来叠着。副标题下面隔开 1.3 个字的表头
    按固定的 1.5 个字算还在行距里，扣掉这份文档的行距就是另起的一块。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y0, y1, size, line_gap):
        rect = pymupdf.Rect(114, y0, 1122, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False,
                         line_gap=line_gap)

    def slide(line_gap):
        return [blk("Out of 9222 sentences", 150, 205, 45.0, line_gap),
                blk("i want to eat chinese food lunch spend", 263, 304, 31.9, line_gap)]

    assert len(_merge_continuations(slide(0.1))) == 1
    assert len(_merge_continuations(slide(-0.15))) == 2


def test_a_line_tail_pair_measures_its_gap_against_the_documents_line_gap():
    """行尾半句和下一行回到栏边的续写，行距也扣掉这份文档紧排两行之间的空。
    提取时配对、写回时再认，两边都看块上带的同一个值。"""
    from app.formats.pdf import TextBlock, _same_page_line_pair

    def pair(line_gap):
        def blk(text, rect):
            return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False,
                             line_gap=line_gap)
        return (blk("which we call the", pymupdf.Rect(300, 100, 480, 112)),
                blk("unigram model of the text", pymupdf.Rect(72, 127, 480, 139)))

    assert _same_page_line_pair(*pair(0.1))
    assert not _same_page_line_pair(*pair(-0.2))


def test_a_running_head_does_not_take_the_first_line_below_it():
    """书的页眉靠右排、不带句号，几截排开像一行的后半句，下一行回到栏边、小写起头。
    认出是页眉就不接，行距卡在哪里都一样。"""
    from app.formats.pdf import TextBlock, _merge_same_line_sentences

    def blocks():
        head = [pymupdf.Rect(378, 82.8, 391.6, 94.8), pymupdf.Rect(401.8, 82.8, 405.3, 94.8),
                pymupdf.Rect(415.5, 82.8, 466.1, 94.8)]
        body = [pymupdf.Rect(160, 104, 493, 116), pymupdf.Rect(160, 116, 493, 128)]
        return [
            TextBlock(page=0, rect=pymupdf.Rect(378, 82.8, 466.1, 94.8), line_rects=head, text="5.5 • WORD2VEC",
                      size=10.0, color="#000", bold=False),
            TextBlock(page=0, rect=pymupdf.Rect(160, 104, 493, 128), line_rects=body,
                      text="automobile are distinct and unrelated; sparse vectors may thus fail to capture",
                      size=10.0, color="#000", bold=False),
        ]

    assert len(_merge_same_line_sentences(blocks())) == 1
    head, body = blocks()
    assert len(_merge_same_line_sentences([head, body], {id(head)})) == 2


def test_a_slide_sub_point_is_not_a_wrapped_title():
    """幻灯片正文就有三十来磅，缩进排的子项不是大标题换行的第二行。
    大字号比这份文档的正文字号算；没量出正文字号时照旧按 12 磅。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def slide(body_size):
        def blk(text, x0, y0, x1, y1):
            rect = pymupdf.Rect(x0, y0, x1, y1)
            return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=27.0, color="#000", bold=False,
                             body_size=body_size)
        return [blk("Example: Deterministic language L = {red,blue, green}", 77, 229, 672, 262),
                blk("Branching factor = 3, any word can be followed by red, blue, green", 163, 272, 903, 305)]

    assert len(_merge_continuations(slide(38.0))) == 2
    assert len(_merge_continuations(slide(0.0))) == 1


def test_a_block_cut_by_a_margin_note_does_not_take_an_indented_new_paragraph():
    """旁注把一段正文切成上下两块，拿掉旁注后接回。下一块自己首行缩进的是另起的一段：
    例句的译文下面接着新段落。不缩进的下一块才是被切开的同一句。"""
    from app.formats.pdf import _note_cut

    def block(*lines):
        return {"lines": [{"dir": (1, 0), "bbox": bbox, "spans": [{"text": text, "size": 10.0}]} for text, bbox in lines]}

    gloss = block(("He said the biggest house in the country was this one", (174.1, 307.2, 396.9, 319.1)))
    paragraph = block(
        ("Alternatively, in languages like Koryak, a Chukotko-Kamchatkan language", (157.1, 323.1, 475.2, 335.1)),
        ("spoken in the northern part of the Kamchatka peninsula in Russia, a single", (142.2, 335.1, 475.2, 347.1)),
    )
    assert not _note_cut(gloss, paragraph)
    upper = block(("Alternatively, in languages like Koryak, a Chukotko-Kamchatkan language", (142.2, 307.2, 475.2, 319.1)))
    lower = block(
        ("spoken in the northern part of the Kamchatka peninsula in Russia, a single", (142.2, 323.1, 475.2, 335.1)),
        ("word may have very many morphemes", (142.2, 335.1, 400.0, 347.1)),
    )
    assert _note_cut(upper, lower)
    # 列表项悬挂缩进的续行下面，比续行靠左起笔的一行是列表完了以后的正文
    item = block(("2. embeddings, vectors that represent the meanings of tokens and concepts in-", (173.0, 234.0, 493.0, 247.0)),
                 ("side the network", (185.0, 247.0, 251.0, 259.0)))
    after = block(("Neural networks are the fundamental computational tool for language pro-", (175.0, 262.0, 493.0, 275.0)))
    assert not _note_cut(item, after)


def test_a_bracket_or_quote_closes_a_sentence_only_after_its_end_mark():
    """右括号、引号收尾的行，前面有句末标点才算一句完了。引文年份 (Vilain et al., 1995)、
    代码 r"cat|dog" 后面接着的是同一句。括号紧跟着函数名、变量的是式子收尾，下面的式子、where 不接。"""
    from app.formats.pdf import TextBlock, _ends_sentence, _merge_continuations

    assert _ends_sentence("found it.)") and _ends_sentence('he said."')
    assert not _ends_sentence("The MUC F-measure (Vilain et al., 1995)")
    assert not _ends_sentence('The pattern r"cat|dog"')
    assert _ends_sentence("yi = softmax(hL i WK)")

    def blk(text, y, x0=142.2):
        rect = pymupdf.Rect(x0, y, 475.2, y + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    citation = [blk("Let’s just explore two of the metrics. The MUC F-measure (Vilain et al., 1995)", 484),
                blk("is based on the number of coreference links (pairs of mentions) common to H and R.", 496)]
    assert len(_merge_continuations(citation)) == 1
    formula = [blk("yi = softmax(hL i WK)", 179, x0=262), blk("ti = argmaxk(yi)", 194, x0=263)]
    assert len(_merge_continuations(formula)) == 2


def test_a_list_ends_where_the_next_block_starts_left_of_the_item():
    """列表项后面第一行比它还靠左起笔的是列表完了以后的正文；悬挂缩进的续行照样接上。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, y0, x1, y1):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    after = [blk("8. none (1)", 85, 264, 139, 276),
             blk("none indicates that there are no attributes in the GROUP BY clause.", 71, 289, 540, 301)]
    assert len(_merge_continuations(after)) == 2
    item = [blk("2. fluency: how fluent the translation is in the target language (is it grammatical,",
                172.7, 267.9, 493.2, 280.9),
            blk("clear, readable, natural).", 185.1, 280.2, 281.2, 292.2)]
    assert len(_merge_continuations(item)) == 1


def test_an_example_number_stays_apart_from_its_sentence():
    """例句编号 (25.4) 是标签，不连进右边的句子：连上以后块的左缘退到编号上，悬挂缩进的第三行就接不上了。"""
    from app.formats.pdf import _same_line_word

    assert not _same_line_word("(25.4)", "Before winter I built a chimney",
                               pymupdf.Rect(142.2, 556.9, 166.3, 568.9), pymupdf.Rect(179.1, 556.6, 458.9, 569.5), 10.0)


def test_numbered_examples_in_one_block_are_split_apart():
    """一块里两行例句 (1.2)、(1.3)，第一行最长，按右缘看不出是硬换行。上一句说完、下一行以例句编号起头，拆开。"""
    from app.formats.pdf import _split_lines

    def item(text, bbox):
        return (text, pymupdf.Rect(bbox), [{"text": text, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0}])

    items = [item("(1.2) It was difficult to make it up the steep hill.", (142.2, 595.0, 335.0, 607.0)),
             item("(1.3) What a difficult decision!", (142.2, 609.0, 268.0, 621.0))]
    assert _split_lines(items) == [[0], [1]]
    wrapped = [item("as derived in the two equations", (142.2, 595.0, 335.0, 607.0)), item("(2.3) we see", (142.2, 609.0, 268.0, 621.0))]
    assert _split_lines(wrapped) == [[0, 1]]


def test_a_paragraph_after_a_hanging_list_line_is_split_off():
    """列表项折行悬挂缩进，下一段首行缩进比续行靠左，是列表完了。首行缩进的段落第二行回到栏边照常接着。"""
    from app.formats.pdf import _split_lines

    def item(text, bbox):
        return (text, pymupdf.Rect(bbox), [{"text": text, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0}])

    items = [item("4. If one word in the question is changed, the human-preferred answer changes", (155, 202, 475, 214)),
             item("to the other party", (167, 214, 235, 226)),
             item("The kind of world knowledge that might be needed to solve the problems", (157, 229, 475, 241))]
    assert _split_lines(items) == [[0, 1], [2]]
    paragraph = [item("The first line of a paragraph is set to the full measure and", (157, 202, 475, 214)),
                 item("The second line goes back to the margin", (142, 214, 475, 226))]
    assert _split_lines(paragraph) == [[0, 1]]


def test_a_colored_example_sentence_stays_original_but_an_epigraph_does_not():
    """正文中间缩进排开的一句彩色例句留原文，下文要拿原句里的词来讲。章首题词也这样排，后面跟着出处，照常翻。"""
    from app.formats.pdf import TextBlock, _attribution, _colored_example
    from app.formats.pdf_frame import TextFrame

    frame = TextFrame(142.2, 475.2, 82.0, 719.0, 10.0)

    def blk(text, rect, color=0x0000FF, size=10.0, flags=0):
        rect = pymupdf.Rect(rect)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#0000ff", bold=False,
                         body_size=10.0, span_lines=[[{"text": text, "size": size, "color": color, "flags": flags}]])

    example = blk("Victoria Chen, CFO of Megabucks Banking, saw her pay jump to $2.3 million.", (167, 456, 475, 480))
    assert _colored_example(example, frame)
    assert not _colored_example(blk(example.text, (167, 456, 475, 480), color=0x404040), frame)
    assert not _colored_example(blk(example.text, (167, 456, 475, 480), flags=2), frame)
    assert not _colored_example(blk(example.text, (142.2, 456, 475, 480)), frame)
    quote = blk("On two occasions I have been asked, will the right answers come out?", (167, 210, 475, 258))
    assert _attribution(quote, blk("Babbage (1864)", (411, 246, 475, 258)))
    assert not _attribution(example, blk("Here Victoria is referred to by many different phrases", (142, 484, 475, 520)))


def test_numbered_and_bracket_labeled_examples_stay_original():
    """编号起头、续行悬挂在编号右边的例句和带 [角标 …] 标注的例句留原文。
    以式子编号起头、续行回到栏边的段落，层号上标 W[1]、整个括号只有一个词的 [EOS] 照翻。"""
    from app.formats.pdf import TextBlock, _example_ids

    def blk(spans, rows):
        rects = [pymupdf.Rect(row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        text = "".join(text for text, _size in spans)
        return TextBlock(page=0, rect=rect, line_rects=rects, text=text, size=10.0, color="#000000", bold=False,
                         span_lines=[[{"text": text, "size": size} for text, size in spans]])

    example = blk([("(24.3) AMD agreed to [buy] Markham, Ontario-based ATI for around $5.4 billion", 10.0),
                   (" in cash and stock, the companies announced Monday.", 10.0)],
                  [(160, 491, 490, 504), (185, 504, 399, 516)])
    reference = blk([("(2.4) follows from the definition of the loss and the bound on the", 10.0),
                     (" gradient given above.", 10.0)], [(160, 491, 490, 504), (160, 504, 300, 516)])
    labeled = blk([("[", 10.0), ("Arg0", 7.0), (" The group] ", 10.0), ("agreed", 10.0), (" [", 10.0), ("Arg1", 7.0),
                   (" it wouldn’t make an offer].", 10.0)], [(185, 300, 400, 312)])
    layers = blk([("We use W", 10.0), ("[", 7.0), ("1", 7.0), ("] and W", 10.0), ("[", 7.0), ("2", 7.0),
                  ("] for the weights.", 10.0)], [(160, 300, 400, 312)])
    tokens = blk([("Then we append an [", 10.0), ("EOS", 7.0), ("] and [", 10.0), ("BOS", 7.0), ("] token.", 10.0)],
                 [(160, 300, 400, 312)])
    found = _example_ids([example, reference, labeled, layers, tokens])
    assert found == {id(example), id(labeled)}


def test_a_verbatim_line_does_not_run_into_the_prose_below():
    """等宽字照录的问答 Q: ... A: Charles 没有句末标点，下一行回到栏边、小写开头的正文不是它的续行。"""
    from app.formats.pdf import TextBlock, _line_tail_match

    def blk(text, rect, font):
        rect = pymupdf.Rect(rect)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False,
                         span_lines=[[{"text": text, "size": 10.0, "font": font, "flags": 8 if font == "txtt" else 0}]])

    answer = blk("A: Charles", (376, 111, 429, 121), "txtt")
    prose = blk("then the same kinds of training data would lead the language model to", (142, 126, 475, 138),
                "NimbusRomNo9L-Regu")
    assert not _line_tail_match(answer, prose)
    tail = blk("A: Charles", (376, 111, 429, 121), "NimbusRomNo9L-Regu")
    assert _line_tail_match(tail, prose)
    # 正文里夹的等宽字折到下一行，两边都是等宽，照常接上
    mixed = blk("1. The prompt is assembled, including", (376, 111, 429, 121), "NimbusRomNo9L-Regu")
    mixed.span_lines.append([{"text": "<user> Who wrote ‘‘The Origin", "size": 10.0, "font": "txtt", "flags": 8}])
    rest = blk('of Species"?', (142, 126, 220, 138), "txtt")
    assert _line_tail_match(mixed, rest)


def test_a_line_in_a_column_of_its_own_font_stays_in_its_block():
    """幻灯片右边一栏等宽字排的示例和左边正文同高，横向还叠上一点。它在自己的块里和上一行
    左缘对齐、字体相同，是这一栏的一行，不当左边正文切下来的后半截挪过去。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(text, bbox, font):
        return {"dir": (1, 0), "bbox": bbox,
                "spans": [{"text": text, "bbox": bbox, "size": 29.0, "font": font, "flags": 0, "color": 0,
                           "origin": (bbox[0], bbox[3] - 7)}]}

    def page(with_mate):
        prose = line("according to its probability p(w|<s>)", (38, 221, 523, 256), "Calibri")
        sample = line("I want", (520, 232, 693, 265), "CourierNewPSMT")
        column = [line("<s> I", (520, 194, 607, 227), "CourierNewPSMT")] if with_mate else []
        return {"blocks": [{"type": 0, "bbox": (38, 221, 523, 256), "lines": [prose]},
                           {"type": 0, "bbox": (520, 194, 693, 265), "lines": [*column, sample]}]}

    data = page(True)
    _rejoin_split_lines(data)
    assert [len(block["lines"]) for block in data["blocks"]] == [1, 2]
    data = page(False)
    _rejoin_split_lines(data)
    assert [len(block["lines"]) for block in data["blocks"]] == [2, 0]


def test_a_small_fraction_after_a_relation_joins_the_line():
    """≃ 后面的 1/L 比正文小一号，和 ≃ 隔着正文的一个字宽。按小字的字宽算接不上，
    分子分母连同后面的求和留在另一块里，被当成公式行跳过，译文里少了右半个式子。"""
    from app.formats.pdf import _rejoin_split_lines

    def line(x0, x1, y0, y1, text, font, size):
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [{"text": text, "font": font, "size": size}]}

    head = line(108, 315, 84, 94, "that a differentiable estimator can be constructed:", "NimbusRomNo9L-Regu", 10.0)
    rel = line(330, 406, 84, 95, "q(z|x)f(z) dz ≃", "CMSY10", 10.0)
    numerator = line(414, 418, 82, 89, "1", "CMR7", 7.0)
    numerator["spans"][0]["math"] = True  # _mark_math_roman 给式子里的正体数字打的标
    denominator = line(414, 419, 90, 97, "L", "CMMI7", 7.0)
    data = {"blocks": [{"type": 0, "bbox": (108, 84, 406, 95), "lines": [head, rel]},
                       {"type": 0, "bbox": (414, 82, 419, 97), "lines": [numerator, denominator]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [head, rel, numerator, denominator]


def test_a_full_line_ending_in_a_call_runs_on_into_the_next_line():
    """行尾的 f(g(x)) 算句末，单独排的式子下面的 where 不接。排满一整行、下一行是正文词起头的照样接；
    下一行是另一个式子的不接。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, font="NimbusRomNo9L-Regu"):
        rect = pymupdf.Rect(108, y, 504, y + 11)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000", bold=False,
                         span_lines=[[{"text": text, "font": font, "size": 10.0}]])

    head = blk("that a differentiable estimator can be constructed: q(z|x)f(z) dz ≃ f(g(x, e))", 84)
    tail = blk("where e ∼ p(e). In section 2.3 we applied this trick.", 96)
    assert len(_merge_continuations([head, tail])) == 1
    formula = blk("we write the next sublayer as t4 = LayerNorm(t3)", 84)
    after = blk("hi = t5 + t3", 96)
    assert len(_merge_continuations([formula, after])) == 2


def test_table_split_keeps_space_spans():
    """两端对齐的行，词间空格是单独的 span。按格子重拼文字时要带上，词才不会粘在一起。"""
    from app.formats.pdf import TextBlock, _split_table_blocks
    from app.formats.pdf_layout import RuledTable

    def span(text, x0, x1):
        return {"text": text, "bbox": (x0, 100, x1, 110), "origin": (x0, 108), "font": "Times", "size": 10.0}

    first = [span("Sample", 140, 170), span(" ", 170, 176), span("minibatch", 176, 220),
             span(" ", 220, 226), span("rule.", 226, 250)]
    second = [{**item, "bbox": (item["bbox"][0], 112, item["bbox"][2], 122),
               "origin": (item["origin"][0], 120)} for item in (span("We", 140, 152), span(" ", 152, 158), span("used", 158, 180))]
    rects = [pymupdf.Rect(140, 100, 250, 110), pymupdf.Rect(140, 112, 180, 122)]
    block = TextBlock(page=0, rect=rects[0] | rects[1], line_rects=rects, text="",
                      size=10, color="#000", bold=False, span_lines=[first, second])
    table = RuledTable(pymupdf.Rect(100, 98, 300, 124),
                       (pymupdf.Rect(100, 98, 300, 111), pymupdf.Rect(100, 111, 300, 124)), algorithm=True)
    parts = _split_table_blocks([block], (table,))
    assert [part.text for part in parts] == ["Sample minibatch rule.", "We used"]


def test_a_split_off_period_keeps_its_own_baseline():
    """「 .」的空格落在根号里下标的基线上，切开后句点要用自己的起笔点，擦除时才认得到。"""
    from app.formats.pdf import _split_covered

    span = {"text": " .", "font": "NimbusRomNo9L-Regu", "size": 10.0, "bbox": (455.2, 605.4, 459.5, 620.7),
            "origin": (455.2, 618.3)}
    chars = {(" .", 455.2, 618.3): [(455.2, 457.0, (455.2, 618.3)), (457.0, 459.5, (457.0, 613.0))]}
    bar = pymupdf.Rect(440.5, 610.3, 456.0, 610.7)
    pieces = _split_covered(span, [bar], chars, ([440.7, 455.2, 457.0], [455.2, 457.0, 459.5]), "math")
    assert [piece["text"] for piece in pieces] == [" ", "."]
    assert pieces[1]["origin"] == (457.0, 613.0)


def test_a_numerator_left_in_its_own_block_rejoins_the_paragraph():
    """分子 g⁴ 和它的下标 T,i 被分进一个单独的块，逐行挪都有同块的行贴着。整块叠在段落里
    分母的式子上、和段落的正文行同高，一起挪回段落。"""
    from app.formats.pdf import _rejoin_split_lines

    reg, cmmi = "NimbusRomNo9L-Regu", "CMMI7"

    def line(*spans):
        box = pymupdf.Rect(spans[0]["bbox"])
        for span in spans[1:]:
            box |= pymupdf.Rect(span["bbox"])
        return {"bbox": tuple(box), "dir": (1, 0), "spans": list(spans)}

    lead = line(_glyph("From,", reg, 9.7, 4, (105.0, 558.9, 128.4, 570.6), (105.0, 561.0, 128.4, 569.9)))
    denominator = line(_glyph("4∥g", cmmi, 6.8, 0, (209.1, 566.4, 220.8, 573.2), (209.1, 566.4, 220.8, 573.2), math=True))
    tail = line(_glyph("T,i", cmmi, 6.8, 0, (312.9, 565.5, 323.1, 572.2), (312.9, 565.5, 323.1, 571.0), math=True),
                _glyph(", we can take square root of both side and", reg, 9.7, 4,
                       (323.6, 558.9, 490.3, 570.6), (323.6, 561.0, 490.3, 569.9)))
    g4 = line(_glyph("g", cmmi, 6.8, 0, (219.6, 556.9, 223.3, 563.7), (219.6, 558.9, 223.3, 563.7), math=True),
              _glyph("4", "CMR5", 4.8, 0, (223.6, 555.4, 226.9, 560.3), (223.6, 555.4, 226.9, 559.1), math=True))
    sub = line(_glyph("T,i", "CMMI5", 4.8, 0, (223.3, 560.3, 232.7, 565.1), (223.3, 560.3, 232.7, 563.9), math=True))
    data = {"blocks": [{"type": 0, "bbox": (105.0, 558.9, 490.3, 573.2), "lines": [lead, denominator, tail]},
                       {"type": 0, "bbox": (219.6, 555.4, 232.7, 565.1), "lines": [g4, sub]}]}
    _rejoin_split_lines(data)
    assert {id(item) for item in data["blocks"][0]["lines"]} == {id(item) for item in (lead, denominator, tail, g4, sub)}
    assert data["blocks"][1]["lines"] == []


def test_a_radical_bar_moves_with_its_formula_under_the_line_above():
    """根号横线上面贴着上一行的降部，横线左端接在本公式根号的右上角，照样跟公式走。"""
    from app.formats.pdf import _absorb_one_box

    radical = {"text": "√", "origin": (339.8, 625.9), "ink": (339.8, 618.0, 348.1, 628.0)}
    alpha = {"text": "α", "origin": (348.1, 625.5), "ink": (348.1, 619.6, 354.5, 625.7)}
    bar = pymupdf.Rect(347.9, 618.0, 358.2, 618.4)
    above = {"text": "p", "origin": (349.0, 613.5), "ink": (349.0, 608.8, 354.0, 616.6)}
    formula = {"bbox": pymupdf.Rect(339.8, 617.0, 358.2, 628.6), "d": 3.0, "spans": [radical, alpha]}
    _absorb_one_box(formula, [bar], [radical, alpha, above])
    assert formula["curves"] == [bar]


def test_a_formula_clip_reaches_into_its_own_gaps_beside_a_comma():
    """𝒩 的笔画伸进它和 ( 之间的空档。公式后面紧挨着逗号时，只有最右边的矩形要让着逗号，
    中间的矩形照常扩，空档不能被切掉。"""
    from app.formats.pdf import _formula_form, _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    fms = _page_form(doc, page)
    spans = [{"text": "x", "bbox": (407.1, 490.1, 413.0, 500.1)}, {"text": "∼N", "bbox": (424.5, 489.9, 444.3, 498.1)},
             {"text": "(", "bbox": (445.8, 490.1, 449.7, 500.1)}, {"text": ")", "bbox": (464.2, 490.1, 468.0, 500.1)}]
    formula = {"bbox": pymupdf.Rect(407.1, 489.9, 468.0, 500.1), "spans": spans}
    # 前面的 for 和后面的逗号都紧挨着公式
    word = {"text": "for", "bbox": (395.2, 489.9, 407.0, 500.0)}
    comma = {"text": ",", "bbox": (468.0, 489.9, 470.6, 500.0)}
    xref = _formula_form(doc, page, fms, formula, "/Fx", [dict(span) for span in spans] + [word, comma])
    rects = [tuple(map(float, found)) for found in re.findall(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", doc.xref_stream(xref).decode())]

    def inside(x, y):
        y = page.rect.height - y
        return sum(x0 <= x <= x0 + w and y0 <= y <= y0 + h for x0, y0, w, h in rects) % 2 == 1

    assert inside(445.0, 491.0)
    assert not inside(469.0, 495.0)


def test_a_displayed_formula_set_in_the_text_italic_is_kept():
    """用正文斜体写变量的书，单独排的式子按字体认不出公式字符。带算符、正文词不超过两个的留原字；
    单独排的整句（L(ŷ,y) = How much ŷ differs from the true y）照翻。"""
    from app.formats.pdf import _untranslatable

    def spans(*parts):
        return [{"text": text, "font": font, "size": 10.0, "flags": 2 if "Ital" in font else 0}
                for text, font in parts]

    roman, italic, sans, cmr = "NimbusRomNo9L-Regu", "NimbusRomNo9L-ReguItal", "CMSSBX10", "CMR10"
    line = spans(("Q", sans), (" =", cmr), (" H", sans), ("dec", italic), ("W", sans), ("Q", sans), (";", roman),
                 ("K", sans), (" =", cmr), (" H", sans), ("enc", italic), ("W", sans), ("K", sans))
    assert _untranslatable(line, "Q = HdecWQ; K = HencWK", 8, True)
    line = spans(("ˆ", roman), ("y", italic), ("i", italic), ("=", cmr), ("argmax", roman), ("char", roman),
                 ("∈", "CMSY10"), ("Alphabet", roman), ("P", italic), ("(", cmr), ("char", roman), (")", cmr))
    assert _untranslatable(line, "ˆyi = argmaxchar∈AlphabetP(char)", 8, True)
    line = spans(("L", italic), ("(", cmr), ("y", italic), (")", cmr), ("=", cmr), ("How much", roman),
                 ("y", italic), (" differs from the true", roman), (" y", italic))
    assert not _untranslatable(line, "L(y) = How much y differs from the true y", 8, True)
    line = spans(("convolution with width-1 kernel:", italic), ("z", sans), ("j", italic), ("=", cmr), ("x", sans))
    assert not _untranslatable(line, "convolution with width-1 kernel: zj = x", 8, True)


def test_a_fragment_inside_a_kept_formula_is_part_of_it():
    """分数的分母 dx 单成一行，落在留原字的式子行（dv · dv）的框里，是式子的一截。"""
    from app.formats.pdf import TextBlock, _inside_kept_formula

    formula = pymupdf.Rect(329.5, 429.0, 357.5, 461.4)
    piece = pymupdf.Rect(347.6, 442.6, 357.5, 454.5)
    block = TextBlock(page=0, rect=piece, line_rects=[piece], text="dx", size=10, color="#000000", bold=False)
    assert _inside_kept_formula(block, [formula])
    label = TextBlock(page=0, rect=piece, line_rects=[piece], text="prior", size=10, color="#000000", bold=False)
    assert not _inside_kept_formula(label, [formula])
    assert not _inside_kept_formula(block, [pymupdf.Rect(100, 100, 200, 120)])


def test_text_under_a_big_cmex_radical_is_part_of_the_formula():
    """CMEX 的大号根号提出来是字母 p，字体框比字形高。横线从它墨迹的右上角起头，线下的 Var 是被开方数。"""
    from app.formats.pdf import _PageChars, _mark_radicands

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((188.4, 123.8), "Var", fontname="tiro", fontsize=9.96)
    data = page.get_text("dict")
    line = data["blocks"][0]["lines"][0]
    line["spans"].insert(0, {
        "text": "p", "font": "CMEX10", "size": 9.96, "flags": 0, "color": 0,
        "bbox": (177.2, 107.7, 188.4, 138.6), "origin": (177.2, 115.1), "ink": (177.2, 114.7, 188.4, 126.6),
    })
    right = line["spans"][1]["bbox"][2]
    _mark_radicands(_PageChars(page), data, [pymupdf.Rect(188.4, 114.6, right, 115.1)])
    marked = [span for span in line["spans"] if span["text"] == "Var"]
    assert marked and marked[0].get("math")


def test_a_smaller_line_in_another_colour_does_not_continue_a_title():
    """幻灯片标题（51 磅灰色）下面小一号的紫色例句不是标题的折行，不并成一段。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    title = pymupdf.Rect(97.9, 63.0, 859.0, 125.3)
    item = pymupdf.Rect(112.3, 151.7, 1087.7, 198.1)
    blocks = [
        TextBlock(page=0, rect=title, line_rects=[title], text="Berkeley Restaurant Project sentences",
                  size=51.0, color="#404040", bold=False, body_size=38.0),
        TextBlock(page=0, rect=item, line_rects=[item], text="can you tell me about any good cantonese restaurants",
                  size=38.0, color="#330099", bold=False, body_size=38.0),
    ]
    assert len(_merge_continuations(blocks)) == 2


def test_an_index_entry_ends_at_its_page_numbers():
    """索引页上一条到页码就完整了：下一行小写起头也是另一条，不并成一段。正文的行很少停在「, 数字」上。"""
    from app.formats.pdf import TextBlock, _index_page, _merge_continuations

    def page(lines):
        return {"blocks": [{"type": 0, "lines": [{"spans": [{"text": line}]} for line in lines]}]}

    entries = [f"term number {i}, {100 + i}" for i in range(12)] + ["sub entry", "linear chain CRF, 414, 415",
                                                                       "Kleene *, 50–52", "fully-connected, 157"]
    assert _index_page(page(entries))
    prose = [f"this is line {i} of an ordinary paragraph that wraps" for i in range(16)]
    assert not _index_page(page(prose))

    def blk(text, y, entry):
        rect = pymupdf.Rect(80, y, 200, y + 9)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=8.0, color="#000", bold=False,
                         span_lines=[[{"text": text, "font": "Times", "size": 8.0}]], entry=entry)

    blocks = [blk("Frump, 497", 100, True), blk("fully-connected, 157", 110, True)]
    assert len(_merge_continuations(blocks)) == 2


def test_no_borderless_table_is_found_on_an_index_page(monkeypatch):
    """索引页上几栏短条目行首对齐，几行凑在一起像无线的格子表。和目录页一样不找表，条目照翻。"""
    from dataclasses import replace

    from app.formats import pdf
    from app.formats.pdf_layout import RuledTable

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    for column, x in enumerate((60, 260)):
        for row in range(14):
            page.insert_text((x, 80 + 12 * row), f"entry term {column}{row}, {100 + row}", fontsize=8)
    real = pdf.scan_page

    def scan(page):
        found = real(page)
        cells = tuple(pymupdf.Rect(x, 72 + 12 * row, x + 150, 84 + 12 * row) for row in range(3) for x in (58, 258))
        return replace(found, tables=(RuledTable(pymupdf.Rect(58, 72, 408, 108), cells),))

    monkeypatch.setattr(pdf, "scan_page", scan)
    blocks, _frames, scans = pdf._extract(doc)
    assert scans[0].tables == ()
    assert all(block.clip is None for block in blocks)


def test_an_index_entry_with_many_page_numbers_is_translated():
    """词目后面一串页码，字母占不到四成，也是一条词目，不当成一行数留原文。只有页码的一行照旧留原文。
    只有一两个字母的词目（F0）也是一条。"""
    from app.formats import pdf

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    for row in range(16):
        page.insert_text((60, 80 + 12 * row), f"entry term {row}, {100 + row}", fontsize=8)
    page.insert_text((60, 290), "perplexity, 27, 76, 88, 203", fontsize=8)
    page.insert_text((60, 302), "CRF, 414", fontsize=8)
    page.insert_text((60, 314), "F0, 340", fontsize=8)
    blocks, _frames, _scans = pdf._extract(doc)
    texts = {block.text for block in blocks}
    assert "perplexity, 27, 76, 88, 203" in texts and "CRF, 414" in texts
    # 只有一个字母的词目也成块，索引重排时跟着挪，不留在原处被压住
    assert "F0, 340" in texts


def test_a_wrapped_index_entry_ends_where_its_page_numbers_do():
    """索引条目折行时续行悬挂缩进。下一行起笔不比这一行靠右就是另一条（see 引用没有页码）；
    页码折到下一行时那一行只有页码，可能分在另一块里；两端对齐拆开的一行按整行看。"""
    from app.formats.pdf import _index_entry_ends

    def item(text, x0, y0, x1):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 9.6), [{"text": text, "size": 8.0}], [])

    see = [item("CFG, see context-free gram-", 179.3, 111.8, 271.3), item("mar", 203.3, 121.2, 215.6),
           item("chain rule, 123, 171", 179.3, 130.4, 242.9)]
    assert _index_entry_ends(see, []) == {1, 2}
    head = [item("Chinese", 179.3, 206.6, 205.0), item("as verb-framed language,", 187.3, 216.1, 271.3)]
    assert 0 not in _index_entry_ends(head, [])
    # 续行几行缩进一样，到下一条从首行的左缘起笔才收住
    long = [item("Transformations and Dis-", 77.4, 518.7, 169.4), item("course Analysis", 101.3, 528.2, 169.4),
            item("Project (TDAP),", 101.3, 537.7, 169.4), item("422", 101.3, 547.1, 113.3),
            item("transition probability", 77.4, 556.6, 144.5)]
    assert _index_entry_ends(long, []) == {3}
    split = [item("collaborative", 179.3, 652.4, 220.6), item("completion,", 233.5, 652.4, 271.3)]
    numbers = [pymupdf.Rect(203.3, 661.6, 215.2, 671.9)]
    assert _index_entry_ends(split, numbers) == {1}
    assert _index_entry_ends(split, []) == set()
    # 续行带着逗号收住，页码和续行缩进一样
    wrapped = [item("as weighted average", 187.3, 225.6, 271.3), item("branching factor,", 203.3, 235.0, 271.3)]
    assert _index_entry_ends(wrapped, [pymupdf.Rect(203.3, 244.5, 211.2, 254.1)]) == {1}
    # 块从一条的续行起头，停在断开的半个词上，下一行和它对齐也是同一条
    broken = [item("for pronoun resolu-", 203.3, 456.0, 271.3), item("tion, 566", 203.3, 465.5, 231.6),
              item("hold, 597", 179.3, 475.0, 210.0)]
    assert _index_entry_ends(broken, []) == {1, 2}


def test_a_paragraph_indented_under_an_inset_line_starts_on_its_own():
    """例句的译文缩进排成一行，下一段的首行缩进比它靠左、比后面几行靠右，是新的一段。
    两端对齐的参考文献一行拆成几截，按整行的左缘比，不拆。居中排的几行不算缩进。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12),
                [{"text": text, "size": 10.0, "bbox": (x0, y0, x1, y0 + 12), "font": "Times", "flags": 0}], [])

    items = [item("Soak the preserved vegetable in water until soft, remove, drain, and chop", 192.1, 463.4, 482.6),
             item("We generally distinguish two broad classes of morphemes: roots, the central", 175.1, 479.9, 493.2),
             item("morpheme of the word, supplying the main meaning, and affixes, adding", 160.2, 491.9, 493.2),
             item("additional meanings of various kinds.", 160.2, 504.2, 320.0)]
    assert _split_lines(items) == [[0], [1, 2, 3]]
    entry = [item("carbon footprints of machine learn-", 223.3, 238.6, 339.3), item("ing.", 223.3, 247.1, 235.5),
             item("Journal of Machine Learning", 242.0, 247.1, 339.3), item("Research, 21(248):1-43.", 223.3, 255.6, 301.9),
             item("Henderson, P., X. Li, D. Jurafsky,", 213.3, 266.7, 339.3)]
    assert any({2, 3} <= set(group) for group in _split_lines(entry))
    # 居中排的作者单位越往下越宽，三行对着同一条中线，不是缩进
    author = [item("Machine Learning Group", 339.9, 182.1, 441.1), item("Universiteit van Amsterdam", 334.4, 193.0, 446.6),
              item("Welling Max at the institute of", 327.7, 203.5, 453.3)]
    assert _split_lines(author) == [[0, 1, 2]]


def test_an_indented_line_after_a_full_line_that_ends_a_sentence_starts_a_paragraph():
    """上一行排满、恰好在句末收住，下一行缩进一个半字宽、排到右缘，再下一行回到左缘：是新一段。
    上下是编号条目的是参考文献悬挂缩进的续行，不拆。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12),
                [{"text": text, "size": 10.0, "bbox": (x0, y0, x1, y0 + 12), "font": "Times", "flags": 0}], [])

    items = [item("This ordering reverses by 10 dB, and the residual advantage", 315.2, 428.9, 559.0),
             item("10 dB, and the residual advantage grows to 4.08 dB at 25 dB.", 315.2, 440.9, 559.0),
             item("The correction target in (9) provides an interpretation of", 330.2, 453.6, 559.0),
             item("this crossover. For slowly varying channels and weak obser-", 315.2, 465.5, 559.0),
             item("baseline achieves lower mean NMSE than both variants.", 315.2, 477.5, 539.5)]
    assert _split_lines(items) == [[0, 1], [2, 3, 4]]
    entry = [item("46. Jim Fuller, Anthony L Piro, and Adam S Jermyn. Slowing the spins of cores.", 114.0, 100.0, 474.0),
             item("Monthly Notices of the Royal Astronomical Society, 485(3):3661, 2019.", 129.0, 112.0, 474.0),
             item("47. Ilya Mandel and Alison Farmer. Merging stellar-mass binary black holes.", 114.0, 124.0, 474.0)]
    assert [0, 1] in _split_lines(entry)


def test_numbers_wrapped_under_an_indented_first_line_stay_in_the_paragraph():
    """段落首行缩进，没说完的一串数折到下一行的左缘（13 tokens: 11865, … | 885, 220, … 0)）：
    这一行和上一行对齐，是句尾，不另成一段。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12),
                [{"text": text, "size": 10.0, "bbox": (x0, y0, x1, y0 + 12), "font": "Times", "flags": 0}], [])

    items = [item("The visualization shows colors to separate out words, but of course the true out-", 157.0, 169.0, 475.0),
             item("put of the tokenizer is simply a sequence of unique token ids. (In case you're in-", 142.0, 181.0, 475.0),
             item("terested, they were the following 13 tokens: 11865, 8923, 11, 31211, 6177, 23919,", 142.0, 193.0, 475.0),
             item("885, 220, 19427, 7633, 18887, 147065, 0)", 142.0, 205.0, 310.0),
             item("Notice that most words are their own token, usually including the leading space.", 157.0, 217.0, 475.0),
             item("Clitics like 's are segmented off when they appear on proper nouns like Jane, but", 142.0, 229.0, 475.0)]
    assert _split_lines(items) == [[0, 1, 2, 3], [4, 5]]
    # 上一行停在句中的 et al. 上，下一行的年份是引文的后半截
    cited = [item("least another star and/or BH in close proximity to carry away", 303.3, 728.3, 558.4),
             item("the excess energy and momentum (Heggie 1975; Tanikawa et al.", 303.3, 739.7, 558.4),
             item("2012).", 303.3, 750.6, 329.0)]
    assert _split_lines(cited) == [[0, 1, 2]]


def test_caption_lines_wrapped_back_under_the_label_stay_in_the_caption():
    """图注的标签（Figure 4.12）和后面的正文隔着一段空，是同一行的两截。折下来的行回到标签的左缘，
    按整行比不是左跳，不另成一段：只剩 (2012). 的一行另成一段时整行留原文，译文写完它还在下一行。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1, font="NimbusRomNo9L-Regu"):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 11),
                [{"text": text, "size": 9.0, "bbox": (x0, y0, x1, y0 + 11), "font": font, "flags": 0}], [])

    label = item("Figure 4.12", 143.2, 254.2, 186.4, "NimbusRomNo9L-Medi")
    first = item("A version of the paired bootstrap algorithm after Berg-Kirkpatrick et al.", 196.3, 254.5, 475.2)
    assert _split_lines([label, first, item("(2012).", 142.2, 265.5, 168.3)]) == [[0, 1, 2]]
    assert _split_lines([label, first, item("(2012), resampling the test set with replacement.", 142.2, 265.5, 380.0)]) \
        == [[0, 1, 2]]


def test_page_numbers_wrapped_below_an_index_entry_stay_with_it():
    """索引条目停在逗号上，页码折到下一行、缩到条目文字下面。页码那一行跟着条目一起送翻，
    条目到页码收住，下一条照常另起。"""
    from app.formats.pdf import _index_entry_ends, _split_lines

    def item(text, x0, y0, x1):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 9.6),
                [{"text": text, "size": 8.0, "bbox": (x0, y0, x1, y0 + 9.6)}], [])

    items = [item("Switchboard, 359", 401.2, 130.6, 457.5), item("Switchboard Corpus,", 401.2, 140.5, 472.4),
             item("338,", 479.3, 140.5, 493.2), item("339, 348, 349, 359", 425.1, 149.7, 484.9),
             item("sycophantic, 29", 401.2, 159.2, 451.3)]
    groups = _split_lines(items, True)
    assert [1, 2, 3] in groups or any({1, 2, 3} <= set(group) for group in groups)
    ends = _index_entry_ends(items, [])
    assert 2 not in ends and 3 in ends and 4 in ends


def test_page_numbers_in_the_next_block_join_the_index_entry():
    """页码折到下一行、分在下一块里。只有页码的一行紧贴在停在逗号上的条目下面，接到条目上；
    下面另起的条目不接。"""
    from app.formats.pdf import TextBlock, _merge_continuations, _wraps_from_above

    def blk(text, x0, y0, x1, entry=False):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 9.6)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=8.0, color="#000", bold=False,
                         entry=entry, index=True)

    host = blk("Berkeley Restaurant Project,", 401.2, 643.0, 493.2, entry=True)
    refs = pymupdf.Rect(425.1, 652.5, 433.1, 662.1)
    assert _wraps_from_above([host], 0, refs, "72", [], None, index_page=True)
    assert not _wraps_from_above([host], 0, refs, "72", [], None, index_page=False)
    merged = _merge_continuations([host, blk("72", 425.1, 652.5, 433.1, entry=True),
                                   blk("BERT", 401.2, 662.0, 421.1)])
    assert [block.text for block in merged] == ["Berkeley Restaurant Project, 72", "BERT"]


def test_a_long_index_entry_takes_another_line_and_pushes_the_column_down():
    """索引一栏排满到下沿。译文要两行的条目多占一行，下面的条目（含留原文的）跟着往下挪，
    整栏行距收紧一点，末条不出下沿；别栏不动。多出的行太多就照旧不挪。"""
    from app.formats.pdf import TextBlock, _index_reflow
    from app.formats.pdf_layout import Column, PageGeometry

    def blk(text, x0, row):
        rect = pymupdf.Rect(x0, 100 + row * 9.6, x0 + 60, 110 + row * 9.6)
        span = {"text": text, "size": 8.0, "font": "Times-Roman", "flags": 0, "color": 0, "bbox": tuple(rect)}
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=8.0, color="#000", bold=False,
                         span_lines=[[span]], entry=True, index=True)

    left = [blk(f"entry {n}, {n}", 80, n) for n in range(10)]
    right = [blk(f"other {n}, {n}", 180, n) for n in range(10)]
    geo = PageGeometry(612, (Column(80, 280),), ())
    items = [(b, f"条目{n}, {n}") for n, b in enumerate(left) if n != 6] + [(b, "其他") for b in right]
    items[2] = (left[2], "在命名实体识别(NER)中, 241, 419")
    bottom = left[-1].rect.y1
    moved = _index_reflow(items, [*left, *right], geo, True, {}, set(), bottom)
    assert moved[id(left[2])].rect.height > left[2].rect.height + 8
    assert id(left[6]) in moved and moved[id(left[6])].rect.y0 > left[6].rect.y0
    assert moved[id(left[9])].rect.y1 <= bottom + 1
    assert not any(id(b) in moved for b in right)
    crowded = [(b, "在命名实体识别(NER)中, 241, 419") for b in left]
    assert _index_reflow(crowded, left, geo, True, {}, set(), bottom) == {}
    # 第 5 条提取时丢了，原字留在原处。上面四条挤不下多出来的一行，不挪，也不压到原字上
    dropped = left[4].rect
    rest = [b for n, b in enumerate(left) if n != 4]
    moved = _index_reflow([item for item in items if item[0] is not left[4]], rest, geo, True, {}, set(),
                          bottom, [dropped])
    assert not any(id(b) in moved for b in rest)


def test_an_index_entry_is_never_centered():
    """缩进的子条目正好落在栏中间，也是靠左排的。"""
    from app.formats.pdf import TextBlock, _write_rect
    from app.formats.pdf_layout import Column, PageGeometry

    geo = PageGeometry(612, (Column(80, 280), Column(300, 500)), ())
    rect = pymupdf.Rect(150, 100, 210, 110)
    plain = TextBlock(page=0, rect=rect, line_rects=[rect], text="in NER, 241, 419", size=8.0, color="#000",
                      bold=False)
    entry = TextBlock(page=0, rect=rect, line_rects=[rect], text="in NER, 241, 419", size=8.0, color="#000",
                      bold=False, entry=True, index=True)
    assert _write_rect(plain, [(plain, "")], True, False, geo)[1] == "center"
    assert _write_rect(entry, [(entry, "")], True, False, geo)[1] == "left"


def test_the_rest_of_a_broken_word_continues_the_line_above():
    """上一行停在连字符断开的半个词上，下一行缩进多少都是接着写的（expres- | sions, 50, 65），
    字母少也不当成一行页码留原文。索引页上比上一行缩进得深的也是续行（as, 51）。"""
    from app.formats.pdf import TextBlock, _wraps_from_above

    def blk(text, rect):
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=8.0, color="#000", bold=False)

    above = blk("anchors in regular expres-", pymupdf.Rect(299.3, 366.9, 391.3, 376.5))
    rest = pymupdf.Rect(323.2, 376.1, 363.5, 386.5)
    assert _wraps_from_above([above], 0, rest, "sions, 50, 65", [], None)
    whole = blk("pipe in regular expressions", pymupdf.Rect(205.3, 121.5, 289.3, 131.1))
    tail = pymupdf.Rect(221.3, 130.7, 239.8, 141.0)
    assert not _wraps_from_above([whole], 0, tail, "as, 51", [], None)
    assert _wraps_from_above([whole], 0, tail, "as, 51", [], None, index_page=True)


def test_one_sentence_per_line_is_not_joined_into_a_paragraph():
    """幻灯片上逐行排的例句：各占一行、左缘对齐，好几处上一行后面还放得下下一行的第一个词，
    小写起头也不并成一段。段落里折行的上一行放不下下一个词，照常接上。"""
    from app.formats.pdf import TextBlock, _line_lists, _merge_continuations
    from app.formats.pdf_frame import TextFrame

    def blk(text, y, x1, size=38.0):
        rect = pymupdf.Rect(112.3, y, x1, y + 46.4)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    frame = TextFrame(97.9, 1088.0, 11.0, 600.0, 38.0)
    examples = [blk("can you tell me about any good cantonese restaurants close by", 151.7, 1087.7),
                blk("tell me about chez panisse", 210.8, 524.4),
                blk("i’m looking for a good place to eat breakfast", 269.8, 795.0),
                blk("when is caffe venezia open during the day", 329.8, 764.4)]
    listed = _line_lists(examples, [frame])
    assert len(_merge_continuations(examples, listed)) == 4
    paragraph = [blk("the model is trained on a large corpus of text and then it is", 151.7, 1087.7),
                 blk("evaluated on a held out set of sentences that were never seen", 210.8, 1085.0),
                 blk("during training so that the estimate of its quality is fair", 269.8, 1080.0),
                 blk("and unbiased", 329.8, 400.0)]
    assert _line_lists(paragraph, [frame]) == set()


def test_a_glyph_arrow_over_a_word_marks_both_as_formula():
    """esvect 的 \\vv{woman} 提出来是「#    »」：没有字母、墨迹是一条细带、基线比词高不到一个字，
    紧贴在 woman 上面。箭头和 woman 是式子，后面的 is close to 照翻。上一行的标点不算。"""
    from app.formats.pdf import _PageChars, _mark_overlays

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((139.5, 597.2), "woman is close to", fontname="tiro", fontsize=9)
    page.insert_text((140.5, 593.4), "#    »", fontname="tiro", fontsize=9)
    page.insert_text((139.5, 586.4), "end of the line above, ...", fontname="tiro", fontsize=9)
    page.insert_text((172.0, 586.4), "____", fontname="tiro", fontsize=9)
    data = page.get_text("dict")
    for block in data["blocks"]:
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].startswith("#"):
                    span["ink"] = (140.5, 589.8, 165.8, 592.5)
                elif span["text"].startswith("woman"):
                    span["ink"] = (139.5, 591.1, 203.7, 597.3)
                elif span["text"].startswith("_"):
                    span["ink"] = (172.0, 586.0, 190.0, 586.6)
                else:
                    span["ink"] = (139.5, 580.0, 230.0, 586.6)
    data["blocks"] = [{"type": 0, "lines": [line for block in data["blocks"] for line in block["lines"]]}]
    _mark_overlays(_PageChars(page), data)
    spans = [span for line in data["blocks"][0]["lines"] for span in line["spans"]]
    marked = {span["text"].strip(): bool(span.get("math")) for span in spans}
    assert marked["woman"] and marked["#    »"]
    assert not marked["is close to"]
    assert not marked["end of the line above, ..."] and not marked["____"]


def test_a_word_grid_is_translated_cell_by_cell():
    """一格一个词的词表：各格自成一条，同一排的几格不连成一句。表下面同一块里的图注标签不算格子。"""
    from app.formats.pdf import _grid_cells

    words = [["dwindle", "move", "soar", "shift"], ["advance", "edge", "swell", "tumble"],
             ["climb", "explode", "swing", "fall"]]
    items = [
        (word, pymupdf.Rect(184 + 60 * col, 180 + 10 * row, 230 + 60 * col, 190 + 10 * row), [], [])
        for row, line in enumerate(words) for col, word in enumerate(line)
    ]
    items.append(("Figure 2.4", pymupdf.Rect(80, 240, 120, 250), [], []))
    items.append(("Some selected ASCII codes for some English letters", pymupdf.Rect(130, 240, 400, 250), [], []))
    cells = _grid_cells(items)
    assert cells == set(range(12))
    prose = [(f"line {i} of an ordinary paragraph that wraps", pymupdf.Rect(80, 100 + 10 * i, 400, 110 + 10 * i), [], [])
             for i in range(12)]
    assert not _grid_cells(prose)


def test_a_relation_word_after_an_equals_sign_in_the_formula_is_dropped():
    """y_c = 收进了式子图，后面的 1 是正文字。模型又写一遍「为 1」，读出来是「y_c = 为 1」。"""
    from app.formats.pdf import _restore_placeholders

    formulas = [{"text": "y\x01b\x02c\x01/b\x02 ="}, {"text": "n\x01b\x020\x01/b\x02"}]
    out = _restore_placeholders("其中 {v1} 为 1, 而 {v2} 为 2", formulas)
    assert out == "其中 \x01i\x021\x01/i\x02 1, 而 \x01i\x022\x01/i\x02 为 2"
    assert _restore_placeholders("而 {v1} 大于 1 时", [{"text": "β <"}]) == "而 \x01i\x021\x01/i\x02 1 时"


def test_a_word_broken_at_the_block_end_continues_in_the_next_block():
    """上一块停在连字符断开的半个词上，只有两个词也接上下一块（retrieval-augmented genera- | tion, 267）。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, x0, y0, x1, entry=False):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 9.6)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=8.0, color="#000", bold=False,
                         span_lines=[[{"text": text, "font": "Times", "size": 8.0}]], entry=entry)

    merged = _merge_continuations([blk("retrieval-augmented genera-", 95.4, 217.0, 187.4),
                                   blk("tion, 267", 119.3, 226.2, 147.7, entry=True)])
    assert [block.text for block in merged] == ["retrieval-augmented generation, 267"]
    # 两个词的标签后面不是断开的词，不接
    kept = _merge_continuations([blk("Figure label", 95.4, 217.0, 150.0), blk("tion, 267", 119.3, 226.2, 147.7)])
    assert len(kept) == 2


def test_a_model_name_broken_at_its_hyphen_continues_with_the_number():
    """行尾停在型号的连字符上（VE- | 8. Both datasets …），下一行的数字是型号的后半截，不是编号。
    接回去连字符留着、中间不加空格"""
    from app.formats.pdf import TextBlock, _join_lines, _merge_continuations

    def blk(text, y0, x1):
        rect = pymupdf.Rect(54.0, y0, x1, y0 + 10.0)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False,
                         span_lines=[[{"text": text, "font": "Times", "size": 10.0}]])

    merged = _merge_continuations([blk("Table 5: Out-of-distribution evaluation on VCE and VE-", 226.0, 292.0),
                                   blk("8. Both datasets are evaluated after re-annotation.", 237.0, 280.0)])
    assert [block.text for block in merged] == [
        "Table 5: Out-of-distribution evaluation on VCE and VE-8. Both datasets are evaluated after re-annotation."]
    assert _join_lines(["control system) and GPT-", "4o (2024-05-13)"]) == "control system) and GPT-4o (2024-05-13)"


def test_the_left_column_of_an_index_page_is_not_a_side_note():
    """索引页分几栏，最左一栏比正文版心靠外，像页边旁注。旁注里的短词留原文，索引条目就没有翻。"""
    from app.formats import pdf

    doc = pymupdf.open()
    for number in range(4):
        page = doc.new_page(width=500, height=700)
        for row in range(40):
            word = "abcdefghij"[number] + "klmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"[row]
            page.insert_text((140, 80 + 14 * row), f"ordinary body text {word} that runs across the column", fontsize=10)
    page = doc.new_page(width=500, height=700)
    for column, x in enumerate((40, 200, 330)):
        for row in range(30):
            page.insert_text((x, 80 + 12 * row), f"term {column} {row}, {100 + row}", fontsize=8)
    blocks, _frames, _scans = pdf._extract(doc)
    index = [block for block in blocks if block.page == 4]
    assert index and not any(block.side_note for block in index)


def test_the_rest_of_a_numbered_example_stays_original():
    """编号例句的第二句、换行的 wine.、IT 对照另成了块：紧贴在下面、和编号后的句子对齐的块也是例句。
    回到编号左缘的正文、段首缩进被旁注切出来的首行照翻。"""
    from app.formats.pdf import TextBlock, _example_ids

    def blk(text, rect, label=""):
        rect = pymupdf.Rect(rect)
        spans = [{"text": text, "bbox": tuple(rect), "size": 10.0}]
        if label:
            cut = rect.x0 + 30
            spans = [{"text": label, "bbox": (rect.x0, rect.y0, cut, rect.y1), "size": 10.0},
                     {"text": text[len(label) + 1:], "bbox": (cut + 2, rect.y0, rect.x1, rect.y1), "size": 10.0}]
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False,
                         span_lines=[spans])

    head = blk("(24.14) I just bought a copy of Thoreau’s Walden.", (160.2, 222.1, 491.1, 234.1), "(24.14)")
    second = blk("That one had been very tattered.", (192.1, 234.1, 474.9, 246.1))
    prose = blk("Note that this NP is ambiguous.", (160.2, 250.8, 493.2, 274.8))
    other = blk("(19.2) We saw the Eiffel Tower flying to Paris.", (142.2, 304.6, 329.9, 316.6), "(19.2)")
    first_line = blk("In coordination ambiguity phrases can be conjoined", (157.1, 320.1, 475.2, 333.0))
    found = _example_ids([head, second, prose, other, first_line])
    assert id(second) in found and id(prose) not in found and id(first_line) not in found


def test_a_displayed_line_inside_a_block_is_its_own_paragraph():
    """缩进排开、没排满的一行例句和下一行的正文在同一块里：例句自成一段，下一行小写起头也不并进来。
    绕图排的段落首行和下一行收在同一处，照旧是一段。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), [{"text": text, "size": 10.0}], [])

    witch = [item("The green witch arrived", 185.1, 473.7, 280.2),
             item("and must translate it into the Spanish sentence:", 160.2, 488.6, 347.0)]
    assert _split_lines(witch) == [[0], [1]]
    wrapped = [item("This volume is about the implementation and im-", 175.1, 384.7, 369.9),
               item("plications of that exciting idea, as realized in large", 160.2, 396.3, 369.9),
               item("language models, and the vibrant interdisciplinary field", 160.2, 408.2, 493.2)]
    assert _split_lines(wrapped) == [[0, 1, 2]]


def test_a_colored_example_on_its_own_line_needs_no_period():
    """单独排开的一行彩色例句常常不带句号（The green witch arrived）。"""
    from app.formats.pdf import TextBlock, _colored_example
    from app.formats.pdf_frame import TextFrame

    frame = TextFrame(160.2, 493.2, 82.0, 719.0, 10.0)
    rect = pymupdf.Rect(185.1, 473.7, 280.2, 485.7)
    block = TextBlock(page=0, rect=rect, line_rects=[rect], text="The green witch arrived", size=10.0,
                      color="#0000ff", bold=False, body_size=10.0,
                      span_lines=[[{"text": "The green witch arrived", "size": 10.0, "color": 0x0000FF, "flags": 0}]])
    assert _colored_example(block, frame)


def test_text_annotated_with_arcs_stays_original():
    """依存弧从词连到词，弧上的圆角框里是标签（nsubj）。句子译成中文以后弧对不上词，句子和标签都留原文。
    曲线图里一头高一头低的线不是弧，旁边的正文照翻。"""
    from app.formats import pdf

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((238, 395), "She announced this in January", fontsize=10)
    page.draw_bezier((275, 386), (275, 372), (246, 372), (246, 384), width=0.4)
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(251, 367, 270, 376), radius=0.3)
    shape.finish(color=(0, 0, 0), fill=(1, 1, 1), width=0.4)
    shape.commit()
    page.insert_text((253, 374), "nsubj", fontsize=7)
    page.draw_bezier((300, 470), (330, 420), (360, 440), (400, 430), width=0.4)
    page.insert_text((300, 480), "The loss falls quickly as training goes on.", fontsize=10)
    scan = pdf.scan_page(page)
    assert len(scan.arcs) == 1 and len(scan.round_boxes) == 1
    blocks, _frames, scans = pdf._extract(doc)
    found = pdf._arc_figure_ids(blocks, scans)
    kept = {block.text for block in blocks if id(block) in found}
    assert kept == {"She announced this in January", "nsubj"}


def test_a_wrapped_formula_with_tall_glyph_boxes_is_split_by_baseline():
    """折行的公式上下两片（(o_i ≻ | o_j|x)）字体框交叠一点，按基线看是两行，收成两片，
    不并成一个横跨整栏的框。数值取自 ed3book p226。"""
    from app.formats.pdf import _placeholderize

    def span(text, font, size, box, base, ink=None):
        out = {"text": text, "font": font, "size": size, "bbox": box, "origin": (box[0], base)}
        if ink is not None:
            out["ink"] = ink
        return out

    italic, roman, symbol, body = "NimbusRomNo9L-ReguItal", "CMR10", "CMSY10", "NimbusRomNo9L-Regu"
    b = _block_with_spans([
        [span("move from the simple assertion ", body, 10, (142.2, 140.1, 450.0, 152.1), 149.3),
         span("(", roman, 10, (453.4, 141.8, 457.2, 151.8), 149.3, (453.4, 141.8, 457.2, 151.8)),
         span("o", italic, 10, (457.2, 140.1, 462.2, 152.0), 149.3),
         span("i", italic, 7.4, (462.2, 144.0, 464.3, 152.8), 150.8),
         span("≻", symbol, 10, (467.5, 141.6, 475.2, 158.9), 149.3, (467.5, 143.9, 475.2, 149.7))],
        [span("o", italic, 10, (142.2, 152.1, 147.2, 163.9), 161.3),
         span(" j", italic, 7.4, (147.2, 154.4, 150.3, 164.7), 161.3),
         span("|", symbol, 10, (151.0, 153.5, 153.8, 170.8), 161.3, (151.0, 153.8, 153.8, 163.8)),
         span("x", italic, 10, (153.8, 152.1, 158.2, 163.9), 161.3),
         span(")", roman, 10, (158.2, 153.8, 162.1, 163.8), 161.3, (158.2, 153.8, 162.1, 163.8)),
         span(" to knowing the value", body, 10, (162.1, 152.1, 300.0, 164.1), 161.3)],
    ], size=10.0)
    sent, formulas = _placeholderize(b, 0)
    assert sent.count("{v") == 1
    parts = formulas[0].get("parts") or []
    assert len(parts) == 2 and parts[0]["bbox"].x0 >= 450 and parts[1]["bbox"].x1 <= 170


def test_formula_pieces_keep_their_source_order():
    """一个公式被正文字号的变量切成几截（{v2} {b}W{/b}{v3}），模型把次序调了，按原文换回去。
    中间夹着中文的不动；两个字母以上的是词，不算公式串里的东西。"""
    from app.formats.pdf import _keep_formula_chains

    source = "compute the value of each{v1} as{v2} {b}W{/b}{v3}."
    assert _keep_formula_chains("计算每个{v1}的值, 即 {b}W{/b}{v3}{v2}。", source) == "计算每个{v1}的值, 即 {v2} {b}W{/b}{v3}。"
    assert _keep_formula_chains("即 {v2} {b}W{/b}{v3}。", source) == "即 {v2} {b}W{/b}{v3}。"
    assert _keep_formula_chains("{v3} 是 {v2} 的一部分", source) == "{v3} 是 {v2} 的一部分"


def test_a_period_after_a_small_denominator_does_not_make_it_a_heading():
    """分母 Σ_j e^{S·T_j} 后面跟着正文字号的句号。句号不算正文字号的字，分母照样接回分式所在的行。"""
    from app.formats.pdf import _larger_text

    denominator = {"spans": [{"text": "j", "font": "CMMI6", "size": 6.0}, {"text": "e", "font": "CMMI8", "size": 8.0},
                             {"text": " .", "font": "NimbusRomNo9L-Regu", "size": 10.9}]}
    sigma = {"spans": [{"text": "P", "font": "CMEX8", "size": 8.0}]}
    assert not _larger_text(denominator, sigma)
    heading = {"spans": [{"text": "4.2 SQuAD", "font": "NimbusRomNo9L-Medi", "size": 12.0}]}
    assert _larger_text(heading, {"spans": [{"text": "words", "font": "NimbusRomNo9L-Regu", "size": 9.0}]})


def _draw_visible_space(page, x, baseline):
    """按 LaTeX 的画法用三条短线画一个 ␣：底线 3 点长，两头各立一条 1.4 点的竖线。"""
    page.draw_line((x, baseline - 1.4), (x, baseline), width=0.4)
    page.draw_line((x, baseline), (x + 3, baseline), width=0.4)
    page.draw_line((x + 3, baseline - 1.4), (x + 3, baseline), width=0.4)


def test_drawn_visible_space_is_read_into_the_text():
    """画出来的 ␣ 提取不到字。盖在空格上的换掉空格，空隙里没有字的补一个 ␣，和前面的字隔开一个空格。
    不在字行基线上的同样形状（图里的刻度）不算。"""
    from app.formats.pdf import _PageChars, _mark_visible_spaces
    from app.formats.pdf_layout import scan_page

    doc = pymupdf.open()
    page = doc.new_page()
    font = pymupdf.Font("cour")
    page.insert_font(fontname="C", fontbuffer=font.buffer)
    page.insert_text((100, 100), "merged to   re.", fontname="C", fontsize=10)
    _draw_visible_space(page, 100 + font.text_length("merged to  ", 10) + 2.8, 100)
    page.insert_text((100, 140), "Next", fontname="C", fontsize=10)
    _draw_visible_space(page, 100 + font.text_length("Next", 10) + 4, 140)
    page.insert_text((100 + font.text_length("Next", 10) + 12, 140), "r (total)", fontname="C", fontsize=10)
    _draw_visible_space(page, 300, 300)
    scan = scan_page(page)
    assert len(scan.visible_spaces) == 3
    data = page.get_text("dict")
    _mark_visible_spaces(_PageChars(page), data, scan.visible_spaces)
    lines = ["".join(span["text"] for span in line["spans"]) for block in data["blocks"] for line in block["lines"]]
    assert lines[0].replace("  ", " ") == "merged to ␣re."
    assert " ".join(lines[1:]) == "Next ␣ r (total)"


def test_index_entries_get_notes_with_their_parent_entry():
    """索引条目的说明：子条目按缩进挂在前面最近一条缩进更浅的条目下面，有子条目的举出前几个。
    跨栏时按各栏自己的左缘量缩进。不在索引页上的单元没有说明。"""
    from app.formats.pdf import TextBlock, _index_notes

    def block(text, x0, y0, index=True):
        rect = pymupdf.Rect(x0, y0, x0 + 80, y0 + 10)
        return [TextBlock(5, rect, [rect], text, 9.0, "#000", False, index=index)]

    units = [
        block("Index", 72, 50, index=False),
        block("MT, {b}276{/b}", 72, 100), block("divergences, {b}278{/b}", 82, 110),
        block("post-editing, 293", 82, 120), block("multinomial logistic regression, 108", 72, 130),
        block("why used for probabilities,", 322, 100), block("log", 312, 90),
    ]
    units = [units[0], units[1], units[2], units[3], units[4], units[6], units[5]]
    texts = [unit[0].text for unit in units]
    notes = _index_notes(units, texts, [False] * len(units))
    assert notes[0] == ""
    assert notes[1] == ("An entry in the subject index at the back of the book; "
                        "its sub-entries include “divergences”, “post-editing”.")
    assert notes[2] == "An entry in the subject index at the back of the book, listed under the entry “MT”."
    assert notes[4] == "An entry in the subject index at the back of the book."
    assert "listed under the entry “log”" in notes[6]


def test_wrapped_monospace_phrase_keeps_its_word_space():
    """等宽例句 my dog is | hairy 在词间断行，两片并排时留出原文空格那么宽的词距；
    网址在词中间断行、式子在运算符后面断行的，两片照旧挨着。"""
    from app.formats.pdf import _PART_GAP, _wrap_gap

    def span(text, x0, x1):
        return {"text": text, "font": "NimbusMonL-Regu", "size": 9.0, "bbox": (x0, 0, x1, 9)}

    line = [span("my", 477.1, 487.9), span(" ", 487.9, 493.3), span("dog", 493.3, 509.4),
            span(" ", 509.4, 514.8), span("is", 514.8, 525.5)]
    assert abs(_wrap_gap(line, [span("hairy", 307.3, 334.2)]) - 5.4) < 0.05
    url = [span("https://artofproblemsolving.com/wiki/index.php/202", 300, 525)]
    assert _wrap_gap(url, [span("5_AIME_II_Problems", 307, 400)]) == _PART_GAP
    assert _wrap_gap([span("a", 500, 505), span(" ", 505, 508), span("+", 508, 515)], [span("b", 307, 312)]) == _PART_GAP


def test_inline_formulas_keep_their_size_next_to_chinese():
    """中文字号比原文小一成，看着一样大。公式按中文的字号比，没缩字号时是原大，不跟着小一成；
    整段缩小时公式跟着缩。"""
    from app.formats.pdf import _formula_slot

    formula = {"has_img": True, "w": 40.0, "h": 8.0, "d": 2.0, "text": "f(x)"}
    scale, _, _ = _formula_slot(formula, 10.0, 8.8, 8.8)
    assert abs(scale - 1.0) < 1e-6
    scale, _, _ = _formula_slot(formula, 10.0, 8.8 * 0.9, 8.8)
    assert abs(scale - 0.9) < 1e-6
    # 不传 reference 时照旧按原文字号比
    scale, _, _ = _formula_slot(formula, 10.0, 8.8)
    assert abs(scale - 0.88) < 1e-6


def test_a_clipped_plot_path_does_not_cover_the_text_below_it():
    """剪进图框的曲线路径框伸到页外几万点。按剪裁区裁过，只剩图框里那一截，盖不住下面的正文。
    整个落在剪裁区外面的路径去掉。"""
    from app.formats.pdf_layout import visible_drawings

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 620), "Otherwise, we take the simulation snapshot before the binary", fontsize=10)
    xref = page.get_contents()[0]
    stream = doc.xref_stream(xref) + (
        b"\nq 81 628 207 154 re W n 1 0 0 rg 90 746 m 279 746 l 279 -31000 l 90 -31000 l h f Q"
        b"\nq 81 628 207 154 re W n 0 1 0 rg 300 700 m 320 700 l 320 720 l h f Q\n"
    )
    doc.update_stream(xref, stream)
    rects = [drawing["rect"] for drawing in visible_drawings(page)]
    assert len(rects) == 1
    assert rects[0].y1 <= 842 - 628 + 1 and rects[0].y0 >= 842 - 782 - 1


def test_a_bold_run_in_label_starts_a_paragraph():
    """上一行在句末收住，下一行以粗体标签起头：几个词的段落小标题（八个词也算）、
    粗体的定理名加编号、Q2 这样的编号条目，都另起一段。按词拆开的 span 之间补上空格再认。"""
    from app.formats.pdf import _label_lead, _split_lines

    def span(text, x0, x1, bold=False):
        return {"text": text, "size": 10.0, "bbox": (x0, 0, x1, 10), "font": "Times-Bold" if bold else "Times",
                "flags": 20 if bold else 4}

    def words(text, x0, bold_words=0):
        out, x = [], x0
        for k, word in enumerate(text.split()):
            out.append(span(word, x, x + 5 * len(word), bold=k < bold_words))
            x += 5 * len(word) + 3
        return out

    assert _label_lead(words("On the Optimality of the Level Schedule. Recall from", 313, 7))
    assert _label_lead(words("Parallelization Across Batches and Batch Instances. A", 313, 6))
    assert _label_lead(words("Proposition 5 (Efficiency beyond oracle radii). Un-", 315, 2))
    assert _label_lead(words("Q2 Which dynamical system functions benefit most", 60, 1))
    assert not _label_lead(words("Important things happen when we look closely", 60, 1))
    assert not _label_lead(words("This whole sentence is set in bold for emphasis and runs on. Then", 60, 12))

    def item(text, x0, y0, x1, bold_words=0):
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), words(text, x0, bold_words), [])

    items = [item("units. Finally, we return to the batch and state when the", 313.2, 190.0, 558.0),
             item("resulting schedule is also optimal for the whole batch.", 313.2, 201.9, 536.2),
             item("Parallelization Across Batches and Batch Instances. A", 313.2, 216.0, 558.0, 6),
             item("batch consists of B evaluations of the same function at dif-", 313.2, 227.9, 558.0)]
    assert _split_lines(items) == [[0, 1], [2, 3]]


def test_an_indented_paragraph_start_is_not_a_continuation():
    """段落首行缩进的文档里，首行缩进、大写的词起头的块是新的一段：上一块停在 and 上
    （下面是单独排的式子）也不接。小写起头、行首是式子的照旧接上。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def blk(text, y, rows):
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12) for i, (x0, x1) in enumerate(rows)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    para = [(64, 299), (54, 299), (54, 200)]

    def plan(text):
        blocks = [blk(body, 100 + 36 * i, para) for i in range(4)]
        return _merge_continuations([*blocks, blk("The forward pass gives its overlap, and", 244, [(54, 299)]),
                                     blk(text, 256, para)])

    assert len(plan("The adjoint state carries the sensitivity of the final error to the control.")) == 6
    assert len(plan("the adjoint state carries the sensitivity of the final error to the control.")) == 5


def test_a_paragraph_continues_after_a_page_of_figures():
    """下一页整页是图，只有图注和紧贴着的分图说明 (C) …：上一页没写完的段落接在再下一页。
    图注下面紧接着正文字号的段落时，那一页有正文，不跳过。"""
    from app.formats.pdf import TextBlock, _cross_page_units

    def blk(text, page, y, rows=3, size=10.0):
        rect = pymupdf.Rect(72, y, 540, y + 12 * rows)
        lines = [pymupdf.Rect(72, y + 12 * k, 540, y + 12 * k + 11) for k in range(rows)]
        return TextBlock(page=page, rect=rect, line_rects=lines, text=text, size=size, color="#000", bold=False)

    host = blk("By the end of training, A is mutually", 0, 600)
    caption = blk("Figure 5: Model comparison on the supervised dataset.", 1, 519, rows=5)
    panel = blk("(C) Difference in summed AIC relative to the best-fitting model.", 1, 584, rows=3)
    tail = blk("excitatory with all units; B and C are mutually inhibitory.", 2, 74, rows=10)
    units = _cross_page_units([host, caption, panel, tail], 10.0)
    assert [host, tail] in units
    body = blk("bie and Shutova (2022), who show that ablating induction heads hurts.", 1, 584, rows=6)
    units = _cross_page_units([host, caption, body, tail], 10.0)
    assert [host, tail] not in units


def test_a_font_that_only_writes_single_marks_beside_formulas_is_math():
    """点阵的空心 1（𝟙）是名字只有编号的 Type3 字体，提出来是 1。全文只写一两个字、多半贴着公式，
    就是公式字体。正文里单独出现的脚注号字体不贴公式，不算。"""
    from app.formats.pdf import _mark_formula_only_fonts, _span_is_math

    def span(font, text, x0, x1):
        return {"font": font, "text": text, "size": 10.0, "bbox": (x0, 50, x1, 60)}

    data = {"blocks": [{"type": 0, "lines": [
        {"spans": [span("CMR10", "the set is", 0, 40), span("CMMI10", "A", 42, 49), span("F74", "1", 49, 55),
                   span("CMSY10", "{", 55, 60)]},
        {"spans": [span("CMR10", "see note", 0, 40), span("F99", "2", 41, 45), span("CMR10", " below", 46, 80)]},
    ]}]}
    _mark_formula_only_fonts([data])
    first, second = (line["spans"] for line in data["blocks"][0]["lines"])
    assert _span_is_math(first[2])
    assert not _span_is_math(second[1]) and not _span_is_math(first[0])


def test_a_roman_numbered_item_does_not_continue_the_paragraph_above():
    """证明里的 (iii) 是另一项。上一段收在行间公式上、没有句号，(iii) 后面大写起头的也不接上来；
    句中列举的 (iii) 后面小写，括号里的小写词，照样接。"""
    from app.formats.pdf import TextBlock, _can_continue

    def blk(text):
        return TextBlock(page=0, rect=pymupdf.Rect(72, 500, 540, 530), line_rects=[],
                         text=text, size=10, color="#000000", bold=False)

    prev = blk("The partition statement is the expansion {v11}")
    assert not _can_continue(prev, blk("(iii) Let {v1} be the sum."), 10, None)
    assert _can_continue(blk("(ii) it must be light, and"), blk("(iii) the arm must have few actuators."), 10, None)
    assert _can_continue(prev, blk("(see below) which holds."), 10, None)

    from app.formats.pdf import _split_lines

    # 上一行停在式子上、没排满，(iii) 另起一项；上一行排满的是段中列举折到了行首
    assert _split_lines([_line("is the expansion = 1", 102, 152, 127), _line("(iii) Let the sum of the first n terms of the series be", 102, 505, 140),
                         _line("written as a product of independent exponentials", 102, 400, 154)]) == [[0], [1, 2]]
    assert _split_lines([_line("the intervening low-density region, regardless of r.", 63, 297, 134),
                         _line("(ii) Such misses propagate through calibration. Be-", 63, 297, 146)]) == [[0, 1]]


def test_the_margin_piece_of_a_wide_caption_line_is_not_a_side_note():
    """伸进页边的图注里，行内式子把一行切成几块，留在页边的 that w[1] 自成一块。上下紧挨的行从同一处
    起笔、写进版心，它是图注的一行，不是旁注。正文旁边的旁注照样是旁注。"""
    from app.formats.pdf import _wide_block_lines
    from app.formats.pdf_frame import FrameLine, TextFrame

    rows = [
        ((96, 570, 140, 581), "Figure 6.18", "a"),
        ((150, 570, 493, 581), "Sample computation graph for a simple 2-layer neural net with", "a"),
        ((95, 581, 493, 592), "and 2 hidden units. We have adjusted the notation a bit to avoid", "a"),
        ((95, 591, 468, 605), "the function that is being computed, and the resulting variable name.", "a"),
        ((95, 605, 125, 619), "that w[1]", "b"),
        ((117, 605, 493, 622), "11 is to be multiplied by x1, and the node z[1] = + means that", "c"),
        ((95, 619.4, 351, 633), "the three nodes that feed into it (the two products, and the bias)", "d"),
        ((100, 300, 150, 310), "regular expressions", "e"),
        ((160, 289, 493, 300), "body text that runs across the whole column of the page here", "f"),
        ((160, 301, 493, 312), "and goes on with more of the same body text below the note", "f"),
    ]
    blocks = {}
    lines, refs = [], []
    for bbox, text, key in rows:
        line = {"bbox": bbox}
        blocks.setdefault(key, {"lines": []})["lines"].append(line)
        lines.append(FrameLine(pymupdf.Rect(bbox), text, 9))
        refs.append((blocks[key], line))
    kept = _wide_block_lines(lines, refs, {0, 4, 7}, TextFrame(160, 493, 108, 719, 10))
    assert kept == {0, 4}


def test_a_numeral_denominator_cut_into_the_next_block_joins_its_numerator():
    """正文字体排的数字分式 24000/75，PyMuPDF 把分母 75 分进下一块。分母挪回分子那一行，
    不然单独成块留原字，压在译文上。"""
    from app.formats.pdf import _rejoin_split_lines

    def span(text, x0, y0, x1, y1, size, base):
        return {"text": text, "bbox": (x0, y0, x1, y1), "origin": (x0, base), "size": size,
                "font": "NimbusRomNo9L-Regu", "flags": 4}

    def line(*spans):
        box = pymupdf.Rect()
        for item in spans:
            box |= pymupdf.Rect(item["bbox"])
        return {"bbox": tuple(box), "dir": (1, 0), "spans": list(spans)}

    head = line(span("this is a downsampling of", 77.4, 623.9, 368.9, 634.7, 9.0, 631.8),
                span("24000", 372.6, 622.2, 390.0, 630.6, 7.0, 627.7))
    denominator = line(span("75", 377.8, 628.9, 384.8, 637.3, 7.0, 634.4))
    nxt = line(span("the encoder and decoder is a quantization step producing", 77.4, 634.9, 352.1, 645.7, 9.0, 642.6))
    data = {"blocks": [{"type": 0, "bbox": head["bbox"], "lines": [head]},
                       {"type": 0, "bbox": (77.4, 628.9, 384.8, 645.7), "lines": [denominator, nxt]}]}
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [head, denominator]
    assert data["blocks"][1]["lines"] == [nxt]


def test_a_script_left_of_a_letter_is_not_its_script():
    """行内分式的分子（小一号）和同一行后面的斜体 x 上下交叠，中间隔着 . Each token in，x 不是它的底。"""
    from app.formats.pdf import _script_base

    x = {"text": " x", "bbox": (223.9, 596.6, 230.4, 608.5), "origin": (223.9, 606.0), "size": 10.0,
         "font": "NimbusRomNo9L-ReguItal"}
    assert not _script_base(x, pymupdf.Rect(146.5, 593.3, 159.8, 607.2), 10.0, math=True, script_y=600.0)
    assert _script_base({**x, "bbox": (140.0, 596.6, 146.5, 608.5)}, pymupdf.Rect(146.5, 600, 149, 609), 10.0,
                        math=True, script_y=607.0)


def test_a_fraction_wrapped_to_the_line_start_ends_the_sentence_above():
    """句子最后的行内分式折到下一行行首（the log of the odds ratio | p/(1−p):），分子分母叠着比一行高，
    后面没有正文词，也是这句的结尾，接回上一段。隔着空、缩进排的是单独的式子。"""
    from app.formats.pdf import TextBlock, _merge_continuations, _wraps_from_above
    from app.formats.pdf_frame import TextFrame

    frame = TextFrame(142, 475, 72, 720, 10)
    rows = [pymupdf.Rect(142.2, y, 475.2, y + 12) for y in (488.8, 500.8, 512.8, 524.7)]
    line = pymupdf.Rect(142.2, 524.7, 475.2, 536.7)
    host = TextBlock(page=0, rect=pymupdf.Rect(142.2, 500.8, 475.2, 536.7), line_rects=rows[1:],
                     text="This is because the logit function is the inverse of the sigmoid. "
                          "The logit function is the log of the odds ratio",
                     size=10, color="#000000", bold=False,
                     span_lines=[[{"text": "The logit function is the log of the odds ratio", "size": 10.0,
                                   "font": "NimbusRomNo9L-Regu", "bbox": tuple(line)}]])
    tail = pymupdf.Rect(143.4, 534.5, 161.0, 556.4)
    assert _wraps_from_above([host], 0, tail, "p 1−p:", rows, frame)
    displayed = pymupdf.Rect(243.8, 548.0, 338.1, 570.0)
    assert not _wraps_from_above([host], 0, displayed, "logit(p) = ln p 1−p", rows, frame)
    # 一行高的数字是下面表格的一格
    assert not _wraps_from_above([host], 0, pymupdf.Rect(143.4, 537.0, 170.0, 548.0), "0.00025", rows, frame)

    def span(text, font):
        return {"text": text, "size": 7.4, "font": font, "bbox": tuple(tail)}

    fraction = TextBlock(page=0, rect=tail, line_rects=[tail], text="1−p: p", size=7.4, color="#000000", bold=False,
                         span_lines=[[span("1", "NimbusRomNo9L-Regu"), span("−", "CMSY10"),
                                      span("p", "NimbusRomNo9L-ReguItal"), span(":", "NimbusRomNo9L-Regu")],
                                     [span("p", "NimbusRomNo9L-ReguItal")]])
    merged = _merge_continuations([host, fraction])
    assert len(merged) == 1 and merged[0].text.endswith("odds ratio 1−p: p")


def test_a_formula_form_box_keeps_the_overhang_of_an_italic_letter():
    """斜体 f 的弯钩和尾巴伸出字宽。Form 的 BBox 也会裁剪，左右放到字形量出来的墨迹，上下照公式的框。
    只放到自己的墨迹为止：a 不伸出字宽，前面 If 的 f 伸进空档的弯钩不带进来。"""
    from app.formats.pdf import _formula_form, _GlyphInk, _page_form

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_font(fontname="ti", fontbuffer=pymupdf.Font("tiit").buffer)
    page.insert_text((350, 526), "If f is a", fontname="ti", fontsize=10)
    spans = {}
    for char in page.get_text("rawdict")["blocks"][0]["lines"][0]["spans"][0]["chars"]:
        if char["c"] in "fa":
            spans.setdefault(char["c"], []).append({"text": char["c"], "bbox": char["bbox"], "origin": char["origin"]})
    fms = _page_form(doc, page)
    glyphs = _GlyphInk(page)

    def box(letter):
        formula = {"bbox": pymupdf.Rect(letter["bbox"]), "spans": [letter]}
        xref = _formula_form(doc, page, fms, formula, "/Fx", [dict(letter)], glyphs)
        x0, y0, x1, y1 = map(float, doc.xref_get_key(xref, "BBox")[1].strip("[]").split())
        return x0, 792 - y1, x1, 792 - y0

    letter = spans["f"][1]
    x0, y0, x1, y1 = box(letter)
    assert x0 < letter["bbox"][0] - 0.5 and x1 > letter["bbox"][2] + 0.5
    assert (round(y0, 1), round(y1, 1)) == (round(letter["bbox"][1], 1), round(letter["bbox"][3], 1))
    a = spans["a"][0]
    x0, _y0, x1, _y1 = box(a)
    assert abs(x0 - round(a["bbox"][0], 2)) < 0.02 and abs(x1 - round(a["bbox"][2], 2)) < 0.02
