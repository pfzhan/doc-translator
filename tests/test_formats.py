import asyncio
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Never

import pytest
from bs4 import BeautifulSoup, Tag

from app.formats import translate_file
from app.formats.epub import _content_docs, translate_epub
from app.formats.markdown import collect_segments, render, split_blocks
from app.formats.mobi import _read_mobi7_html, _xhtml_from_mobi7
from app.formats.pdf_blocks import _join_lines
from app.runner import Runner
from app.translators import Call, MockTranslator, OpenAITranslator
from tests.helpers import SAMPLES, MemCache, _write_epub, run, runner
from tests.test_html_page import _attr, _tag

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


def test_markdown_translated_only() -> None:
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


def test_markdown_bilingual_keeps_original() -> None:
    blocks = split_blocks(MD)
    texts = collect_segments(blocks)
    out = render(blocks, ["T:" + t for t in texts], bilingual=True)
    assert "# Hello World" in out and "# T:Hello World" in out
    assert "This is a paragraph\nspanning two lines.\n\nT:This is a paragraph" in out
    assert "foo<br>T:foo" in out


def _epub_texts(path: Path) -> str:
    z = zipfile.ZipFile(path)
    assert z.infolist()[0].filename == "mimetype"
    assert z.infolist()[0].compress_type == zipfile.ZIP_STORED
    html = "".join(z.read(n).decode() for n in z.namelist() if n.endswith((".xhtml", ".html", ".htm")))
    return BeautifulSoup(html, "lxml").get_text()


@pytest.mark.parametrize("name", ["alice.epub", "alice.mobi", "alice.azw3"])
@pytest.mark.parametrize("bilingual", [True, False])
def test_ebooks(tmp_path: Path, name: str, bilingual: bool) -> None:
    src = SAMPLES / name
    if not src.exists():
        pytest.skip("sample missing")
    outputs = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
    epub = next(p for p in outputs if p.suffix == ".epub")
    text = _epub_texts(epub)
    assert "[zh-CN] Alice was beginning to get very tired" in text
    # The bilingual version keeps the original text; in the translated-only version the original only appears as
    # "[zh-CN] <original>"
    plain = text.replace("[zh-CN] Alice was beginning", "")
    assert ("Alice was beginning to get very tired" in plain) == bilingual


def test_llm_unparseable_batch_falls_back_to_single(monkeypatch: pytest.MonkeyPatch) -> None:
    tr = OpenAITranslator("zh-CN", api_key="x", base_url="http://localhost", model="m")
    calls: list[str] = []

    async def fake_complete(system: str, content: str, call: Call) -> str:
        calls.append(content)
        return "garbage" if len(calls) == 1 else "你好"

    monkeypatch.setattr(tr, "_complete", fake_complete)
    assert run(tr.translate_batch(["a", "b"])) == ["你好", "你好"]  # Batch result unparseable, translate one by one
    assert len(calls) == 3


def test_epub_chapter_with_internal_dtd_not_dropped(tmp_path: Path) -> None:
    """XHTML chapter with an internal DTD subset: lxml-xml silently produces an empty document, so the fallback must
    keep the content."""
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
    # The chapter is not emptied and its paragraphs are translated as usual (&nbsp; collapses to a plain space)
    assert "[zh-CN] Chapter one text here" in text


def test_epub_named_entities_survive_xml_parsing(tmp_path: Path) -> None:
    """lxml-xml must not drop &rsquo;, &nbsp; and &mdash; in an XHTML 1.1 chapter; both the original and the
    translation keep these characters."""
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


def test_epub_html_style_void_tags_and_bare_ampersands_keep_text(tmp_path: Path) -> None:
    """EPUB2 files often have unclosed <br> and <img ...> tags and bare &. lxml-xml's recovering parser drops &amp;
    and &lt; from the first error on and nests paragraphs inside each other. The markup is first repaired into
    well-formed XML and then parsed; case-sensitive attributes such as the SVG viewBox are not affected."""
    from lxml import etree

    body = (
        "<p>First line<br>second &amp; line &lt;tag&gt;</p>"
        '<p>AT&T bought <img src="a.png" alt="logo"> it.</p>'
        "<p>Third paragraph.</p>"
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>'
    )
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(body)})
    [out] = run(translate_epub(src, tmp_path, runner(), True, "zh-CN"))
    data = zipfile.ZipFile(out).read("OEBPS/ch1.xhtml")
    root = etree.fromstring(data)
    texts = ["".join(p.itertext()) for p in root.iter("{http://www.w3.org/1999/xhtml}p")]
    assert texts[:2] == ["First linesecond & line <tag>", "[zh-CN] First line\nsecond & line <tag>".replace("\n", "")]
    assert "AT&T bought  it." in texts and "Third paragraph." in texts
    assert not any(
        p.find("{http://www.w3.org/1999/xhtml}p") is not None for p in root.iter("{http://www.w3.org/1999/xhtml}p")
    )
    assert b'viewBox="0 0 10 10"' in data


def test_mobi7_cp1252_decoding(tmp_path: Path) -> None:
    """book.html of an old MOBI7 file is decoded as cp1252: characters such as é must not become U+FFFD."""
    html_file = tmp_path / "book.html"
    html_file.write_bytes("<html><body><p>café et naïve</p></body></html>".encode("cp1252"))
    text = _read_mobi7_html(html_file)
    assert "café et naïve" in text
    assert "\ufffd" not in text
    # The characters survive the conversion to XHTML too
    assert "café et naïve" in _xhtml_from_mobi7(text).decode("utf-8")


def test_mobi_temp_dir_removed_when_unpack_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The temporary directory is removed when unpacking fails (DRM, broken file) too."""
    from app.formats import mobi as mobi_format

    work = tmp_path / "mobiex-fail"

    def make_work(prefix: str = "") -> str:
        work.mkdir()
        return str(work)

    # mobi uses the tempfile module itself
    monkeypatch.setattr(tempfile, "mkdtemp", make_work)

    def broken(*a: object, **kw: object) -> Never:
        (work / "partial.bin").write_bytes(b"x")
        raise ValueError("DRM")

    monkeypatch.setattr(mobi_format, "unpackBook", broken)
    src = tmp_path / "book.azw3"
    src.write_bytes(b"not a real book")
    with pytest.raises(ValueError):
        run(mobi_format.translate_mobi(src, tmp_path, runner(), False, "zh-CN"))
    assert not work.exists()


def test_mobi_temp_dir_removed_after_cancelled_unpack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Job cancelled during unpacking: the directory is removed after the unpacking thread finishes writing, so no
    file written later by the thread is left behind."""
    import time

    from app.formats import mobi as mobi_format

    work = tmp_path / "mobiex-cancel"

    def make_work(prefix: str = "") -> str:
        work.mkdir()
        return str(work)

    # mobi uses the tempfile module itself
    monkeypatch.setattr(tempfile, "mkdtemp", make_work)

    def slow(src: str, out: str, **kw: object) -> None:
        time.sleep(0.3)
        (Path(out) / "mobi7").mkdir()
        (Path(out) / "mobi7" / "book.html").write_text("<html></html>")

    monkeypatch.setattr(mobi_format, "unpackBook", slow)
    src = tmp_path / "book.mobi"
    src.write_bytes(b"not a real book")

    async def scenario() -> None:
        task = asyncio.create_task(mobi_format.translate_mobi(src, tmp_path, runner(), False, "zh-CN"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.6)

    run(scenario())
    assert not work.exists()


def test_cancelled_calibre_conversion_is_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Job paused or deleted during MOBI conversion: ebook-convert stops too and does not keep writing to the output
    directory in the background."""
    import os

    from app.formats import mobi as mobi_format

    pid_file = tmp_path / "pid"
    exe = tmp_path / "ebook-convert"
    exe.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 30\n")
    exe.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(exe))  # mobi uses the shutil module itself

    def alive(pid: int) -> bool:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        except FileNotFoundError:
            return False
        return state != "Z"

    async def scenario() -> int:
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


def test_markdown_list_does_not_swallow_following_blocks() -> None:
    """A heading or quote right after a list without a blank line is a separate block and is not merged into the last
    list item."""
    blocks = split_blocks("- item\n# Heading\n> quote")
    assert [b.kind for b in blocks] == ["list", "heading", "quote"]
    assert collect_segments(blocks) == ["item", "Heading", "quote"]


def test_markdown_list_and_quote_interrupt_paragraph() -> None:
    """A list or quote right after a paragraph without a blank line is a separate block and is not merged into the
    paragraph."""
    blocks = split_blocks("Steps to install:\n- download\n- run setup\n\nSee below:\n> note here")
    assert [b.kind for b in blocks if b.kind != "raw"] == ["para", "list", "para", "quote"]
    texts = collect_segments(blocks)
    assert texts == ["Steps to install:", "download", "run setup", "See below:", "note here"]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == "T:Steps to install:\n- T:download\n- T:run setup\n\nT:See below:\n> T:note here"


def test_markdown_quote_keeps_paragraphs_alerts_and_lists() -> None:
    """Admonition markers, blank-line paragraph breaks, lists and code blocks inside a quote are kept, and each
    paragraph is translated separately."""
    src = (
        "> [!NOTE]\n> Useful information\n> on two lines.\n>\n> Second paragraph.\n"
        "> - first point\n> ```\n> code here\n> ```"
    )
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["Useful information on two lines.", "Second paragraph.", "first point"]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == (
        "> [!NOTE]\n> T:Useful information on two lines.\n>\n> T:Second paragraph.\n"
        "> - T:first point\n> ```\n> code here\n> ```"
    )
    both = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert "> Second paragraph.\n>\n> T:Second paragraph." in both
    assert "> - first point  \n>   T:first point" in both


def test_markdown_lazy_quote_continuation_stays_in_the_quote() -> None:
    """A line without > after a quoted paragraph still belongs to that paragraph (CommonMark lazy continuation), so
    one sentence is not translated as two. Lines after a heading, a list or a code block inside a quote are not
    continuation lines."""
    src = (
        "> This is a quote that\ncontinues lazily here.\n> - point one\nstill point one\n\nAfter.\n\n"
        "> quote\n# Heading\n\n> ```\n> code\nnot code\n\n> # Title\nplain"
    )
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == [
        "This is a quote that continues lazily here.",
        "point one still point one",
        "After.",
        "quote",
        "Heading",
        "not code",
        "Title",
        "plain",
    ]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out.startswith(
        "> T:This is a quote that continues lazily here.\n> - T:point one still point one\n\nT:After."
    )
    both = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert both.startswith(
        "> This is a quote that\ncontinues lazily here.\n>\n> T:This is a quote that continues lazily here."
    )
    assert out.endswith("> # T:Title\nT:plain") and "> # Title\n> # T:Title\n" in both


def test_markdown_link_definitions_kept_verbatim() -> None:
    """Link reference definitions are kept as they are and not joined into one line; footnote definitions have body
    text and are translated as usual."""
    src = (
        "See [docs][d] and ![ci][c].\n\n[d]: https://example.com/docs\n[c]: https://img.shields.io/ci.svg\n\n"
        "[^1]: A footnote here."
    )
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["See [docs][d] and ![ci][c].", "[^1]: A footnote here."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert "\n[d]: https://example.com/docs\n[c]: https://img.shields.io/ci.svg\n" in out


def test_markdown_title_skips_heading_without_text(tmp_path: Path) -> None:
    """A first level-1 heading made only of digits has no translatable segment; taking the title must not index out of
    range."""
    src = tmp_path / "notes.md"
    src.write_text("# 2024\n\nSome body text here.\n", encoding="utf-8")
    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    assert out.read_text(encoding="utf-8") == "# 2024\n\n[zh-CN] Some body text here.\n"


def test_markdown_toml_front_matter_kept() -> None:
    """Hugo +++ front matter is kept as it is, like YAML, and not sent for translation."""
    blocks = split_blocks('+++\ntitle = "Demo post"\ndraft = false\n+++\n\nBody text.')
    texts = collect_segments(blocks)
    assert texts == ["Body text."]
    assert render(blocks, ["T"], bilingual=False).startswith('+++\ntitle = "Demo post"\ndraft = false\n+++\n')


def test_markdown_long_fence_not_closed_by_inner_fence() -> None:
    """``` inside a four-backtick fence is content and does not end the code block early."""
    src = "````markdown\n```python\nprint(1)\n```\n````\n\nAfter the block."
    blocks = split_blocks(src)
    assert collect_segments(blocks) == ["After the block."]
    assert blocks[0].kind == "raw" and blocks[0].lines[-1] == "````"


def test_markdown_loose_list_continuation_translated_in_place() -> None:
    """An indented continuation paragraph in a loose list belongs to the list item and is translated as usual with its
    indentation kept; only a paragraph indented 4 more spaces is a code block."""
    src = (
        "- item one\n\n    continued paragraph of item one.\n\n        code_in_item()\n\n"
        "Outside paragraph.\n\n    real_code()"
    )
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["item one", "continued paragraph of item one.", "Outside paragraph."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == (
        "- T:item one\n\n    T:continued paragraph of item one.\n\n        code_in_item()\n\n"
        "T:Outside paragraph.\n\n    real_code()"
    )


def test_markdown_tab_indented_code_kept_verbatim() -> None:
    """Tab indentation makes a code block like four spaces; a tab-indented continuation paragraph in a list item is
    still body text."""
    src = "Run this:\n\n\tpip install foo\n\tfoo --help\n\nDone.\n\n- item one\n\n\tcontinued paragraph."
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["Run this:", "Done.", "item one", "continued paragraph."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert "\n\tpip install foo\n\tfoo --help\n" in out and "\n\tT:continued paragraph." in out


def test_markdown_paragraph_starting_with_inline_tag_is_translated() -> None:
    """A paragraph starting with an inline tag or an autolink is translated as usual; block-level tags and a single tag
    on its own line are still kept as HTML blocks."""
    src = (
        "<kbd>Ctrl</kbd> copies the selection.\n\n<https://example.com> is our site.\n\n"
        '<div align="center">\nCentered text\n</div>\n\n<img src="logo.png">\n\n<!-- note -->'
    )
    blocks = split_blocks(src)
    assert collect_segments(blocks) == ["<kbd>Ctrl</kbd> copies the selection.", "<https://example.com> is our site."]


def test_markdown_references_kept_original(tmp_path: Path) -> None:
    """The references section keeps the original text: a citation the model returns unchanged counts as an echo and
    fails, and with many entries the whole book fails."""
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


def test_markdown_repeated_text_in_body_and_references(tmp_path: Path) -> None:
    """The same sentence appears in the body and in the references: writing goes by position, so the body occurrence
    is translated and the references occurrence keeps the original text."""
    src = tmp_path / "paper.md"
    src.write_text(
        "# Intro\n\nDeep learning. MIT Press, 2016.\n\n## References\n\nDeep learning. MIT Press, 2016.\n",
        encoding="utf-8",
    )
    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    assert out.read_text(encoding="utf-8") == (
        "# [zh-CN] Intro\n\n[zh-CN] Deep learning. MIT Press, 2016.\n\n"
        "## [zh-CN] References\n\nDeep learning. MIT Press, 2016.\n"
    )


def test_markdown_numbers_inside_paragraph_stay_in_paragraph() -> None:
    """An ordered list interrupts a paragraph only when it starts at 1; a year after a line break in body text is not a
    list."""
    blocks = split_blocks("It happened in\n1990. Then more text.")
    assert [b.kind for b in blocks] == ["para"]


def test_markdown_display_math_kept_verbatim() -> None:
    """$$ formula blocks are kept as they are: not sent for translation, and lines starting with + or - inside them are
    not list items."""
    src = (
        "Energy is given by\n$$\nL = a\n+ b\n- c\n$$\nwhere m is mass.\n\n"
        "$$ E = mc^2 $$\n\n$$x$$ alone is not a block here.\n\nCosts $$5 today."
    )
    blocks = split_blocks(src)
    texts = collect_segments(blocks)
    assert texts == ["Energy is given by", "where m is mass.", "$$x$$ alone is not a block here.", "Costs $$5 today."]
    out = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert "$$\nL = a\n+ b\n- c\n$$" in out and out.count("E = mc^2") == 1


def test_markdown_unclosed_display_math_is_text() -> None:
    """An unclosed $$ stops being a formula at a blank line, and the following paragraphs are translated as usual."""
    blocks = split_blocks("$$ broken formula\n\nNext paragraph.")
    assert collect_segments(blocks) == ["$$ broken formula", "Next paragraph."]


def test_markdown_tables_with_short_or_aligned_separators() -> None:
    """One - per cell is enough for a GFM delimiter row, and colons may set alignment; a delimiter row whose column
    count differs from the header is not a table."""
    for sep in ("|-|-|", "| :-: | -: |", ":--|--:"):
        blocks = split_blocks(f"| Name | Value |\n{sep}\n| alpha | one |")
        assert [b.kind for b in blocks] == ["table"], sep
        texts = collect_segments(blocks)
        assert texts == ["Name", "Value", "alpha", "one"]
        out = render(blocks, ["T:" + x for x in texts], bilingual=False)
        assert out == f"| T:Name | T:Value |\n{sep}\n| T:alpha | T:one |"
    # Two header columns and one delimiter column: a Setext level-2 heading
    blocks = split_blocks("Name | Value\n---")
    assert [b.kind for b in blocks] == ["setext"]


def test_epub_nav_in_spine_parsed_once(tmp_path: Path) -> None:
    """The EPUB3 nav document is also in the spine: it goes through the label translation path once, without being
    parsed again or losing translations."""
    nav = (
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        "<head><title>Nav</title></head><body>"
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
    assert "OEBPS/nav.xhtml" not in docs  # nav is not parsed again as a body document

    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    with zipfile.ZipFile(out) as z:
        nav_html = z.read("OEBPS/nav.xhtml").decode()
        ch_html = z.read("OEBPS/ch1.xhtml").decode()
    # Each TOC entry is translated exactly once, with no duplicates and no fallback to the original
    assert nav_html.count("[zh-CN] Chapter One") == 1
    assert '<a href="ch1.xhtml">' in nav_html  # Link structure unchanged
    assert "[zh-CN] Body paragraph" in ch_html


def test_a_url_broken_across_lines_is_joined_without_a_space() -> None:
    """A URL wrapped at a / or after the scheme is joined back without a space; when the URL has ended and the next
    line is an ordinary word, a space is added as usual."""
    assert _join_lines(["available at https://github.com/", "tensorflow/tensor2tensor."]) == (
        "available at https://github.com/tensorflow/tensor2tensor."
    )
    assert _join_lines(["University, http:", "//www.ling.ohio-state.edu/", "research/phonetics/E_ToBI/."]) == (
        "University, http://www.ling.ohio-state.edu/research/phonetics/E_ToBI/."
    )
    assert _join_lines(["visit https://site.com/", "for details"]) == "visit https://site.com/ for details"
    assert _join_lines(["the ratio a:", "b holds"]) == "the ratio a: b holds"


def test_a_mark_closed_at_a_line_end_and_reopened_continues() -> None:
    """Each line closes its own marks. A next line that reopens the same marks continues the previous one, so the
    hyphen of a broken word can be recognized."""
    assert _join_lines(["In {i}Proceedings of the 2006 confer-{/i}", "{i}ence on methods{/i}, pages"]) == (
        "In {i}Proceedings of the 2006 conference on methods{/i}, pages"
    )
    assert _join_lines(["{c0000ff}Supervised{/c}", "{c0000ff}learning{/c} of"]) == "{c0000ff}Supervised learning{/c} of"
    # Lines with a different size or emphasis are not joined
    assert _join_lines(["{z80}{b}A{/b}{/z}", "{z90}{b}B{/b}{/z}"]) == "{z80}{b}A{/b}{/z} {z90}{b}B{/b}{/z}"
    assert _join_lines(["a {b}x{/b}", "{i}y{/i}"]) == "a {b}x{/b} {i}y{/i}"


# --- Inline formatting in rich-text paragraphs is kept ---

RICH_CHAPTER = (
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>c1</title></head><body>'
    "<p>Plain paragraph without formatting.</p>"
    '<p>With <b>bold</b> and <a href="https://example.com">a link</a> inside.</p>'
    "</body></html>"
)


def _rich_epub(path: Path) -> None:
    _write_epub(
        path,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>',
        '<itemref idref="ch1"/>',
        {"ch1.xhtml": RICH_CHAPTER},
    )


def test_find_blocks_marks_rich_paragraphs() -> None:
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    blocks = hb.find_blocks(soup)
    assert [html is not None for _, _, html in blocks] == [False, True]
    rich = blocks[1][2]
    assert rich is not None
    assert "<b>bold</b>" in rich and 'href="https://example.com"' in rich


def test_apply_translation_html_keeps_tags_and_strips_script() -> None:
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    el, _, html = hb.find_blocks(soup)[1]
    assert html is not None
    hb.apply_translation_html(
        soup,
        el,
        html,
        '带有<b>粗体</b>和<a href="https://example.com">链接</a><script>alert(1)</script>。',
        False,
        "zh-CN",
    )
    p = soup.find_all("p")[1]
    assert _tag(p.find("b")).get_text() == "粗体"
    assert _attr(_tag(p.find("a")), "href") == "https://example.com"
    assert p.find("script") is None


def test_apply_translation_html_falls_back_when_tags_lost() -> None:
    """The translation lost all inline tags (when the model ignores the rules): fall back to plain text and do not
    write an empty shell."""
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    el, _, html = hb.find_blocks(soup)[1]
    assert html is not None
    hb.apply_translation_html(soup, el, html, "第一行<br>第二行，完全没有标签。", False, "zh-CN")
    p = soup.find_all("p")[1]
    assert p.find("b") is None
    assert p.get_text(separator="\n") == "第一行\n第二行，完全没有标签。"
    assert p.find("br") is not None


def test_pagebreak_anchor_kept_without_rich_tags() -> None:
    """A paragraph whose only markup is a page-break anchor also takes the rich-text path; the anchor is restored when
    the translation drops it."""
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
    assert html is not None
    hb.apply_translation_html(soup, el, html, "章节从这里继续。", False, "zh-CN")
    anchor = soup.find(id="page7")
    assert anchor is not None and "pagebreak" in str(anchor.get("epub:type") or anchor.get("type") or "")


def test_sanitize_drops_event_handlers_and_script_urls() -> None:
    from app.formats import html_blocks as hb

    soup = hb.parse(RICH_CHAPTER, xml=True)
    el, _, html = hb.find_blocks(soup)[1]
    assert html is not None
    hb.apply_translation_html(
        soup,
        el,
        html,
        '<a href="javascript:alert(1)" onclick="alert(1)">链接</a><b onerror="x">粗体</b>',
        False,
        "zh-CN",
    )
    p = soup.find_all("p")[1]
    link = p.find("a")
    assert link is not None and "href" not in link.attrs and "onclick" not in link.attrs
    bold = p.find("b")
    assert bold is not None and "onerror" not in bold.attrs


def test_epub_rich_paragraphs_keep_inline_formatting(tmp_path: Path) -> None:
    src = tmp_path / "book.epub"
    _rich_epub(src)
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    z = zipfile.ZipFile(out)
    html = "".join(z.read(n).decode() for n in z.namelist() if n.endswith((".xhtml", ".html", ".htm")))
    assert "<b>" in html and 'href="https://example.com"' in html
    assert "[zh-CN]" in html


def test_epub_kbd_text_kept_in_translation_only_mode(tmp_path: Path) -> None:
    """When a paragraph has only inline tags such as <kbd>, the key text is sent to the translation service and is not
    lost in the translated-only version either."""
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml("<p>Press <kbd>Ctrl</kbd>+C to copy the selected text.</p>")})
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    z = zipfile.ZipFile(out)
    html = z.read("OEBPS/ch1.xhtml").decode()
    p = _tag(BeautifulSoup(html, "lxml").find("p"))
    assert "[zh-CN]" in p.get_text() and "Ctrl+C" in p.get_text()
    assert "<kbd>Ctrl</kbd>" in html


def test_epub_bilingual_translation_has_no_duplicate_ids(tmp_path: Path) -> None:
    """In the bilingual version the translated paragraph must not carry the original id and page-break anchor,
    otherwise the id is duplicated and the page number shows twice."""
    body = (
        '<p id="para1"><a id="p12"/>Some <b>bold</b> text and <a id="note3" href="#n3">a note</a> here.</p>'
        '<ul><li><span epub:type="pagebreak" id="p13">13</span>Item with <em>emphasis</em> inside.</li></ul>'
    )
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(body)})
    [out] = run(translate_epub(src, tmp_path, runner(), True, "zh-CN"))
    html = zipfile.ZipFile(out).read("OEBPS/ch1.xhtml").decode()
    for ident in ("para1", "p12", "note3", "p13"):
        assert html.count(f'id="{ident}"') == 1, ident
    assert html.count(">13<") == 1
    assert "[zh-CN]" in html and 'href="#n3"' in html


def test_sanitize_rejects_anchor_only_translation() -> None:
    """The original has text and the translation has only the anchor (model misbehaving): the translation is rejected
    and falls back without emptying the original paragraph."""
    from app.formats import html_blocks as hb

    soup = hb.parse('<html><body><p><a id="page42"/>Anchor and plain text here.</p></body></html>', xml=False)
    el, _, html = hb.find_blocks(soup)[0]
    assert html is not None  # The anchor paragraph takes the rich-text path
    hb.apply_translation_html(soup, el, html, '<a id="page42"/>', False, "zh-CN")
    p = _tag(soup.find("p"))
    assert p.get_text()  # Fallback keeps content: the original or a plain-text translation, never an empty shell


def test_sanitize_rejects_numbered_pagebreak_as_only_text() -> None:
    """A pagebreak with a page number is not a surviving translation. A reply with only the anchor, a <br>, or empty
    tags plus the page number must not reduce the body text to the page number."""
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
        text = _tag(soup.find("p")).get_text()
        assert soup.find(id="page7") is not None
        return text

    assert prose in apply(anchor + prose, anchor)
    assert prose in apply(anchor + prose, "<br/>")
    assert "Important" in apply(anchor + "<b>Important</b> words.", anchor + "<b></b>")
    kept = apply(anchor + prose, anchor + "章节从这里继续。")
    assert "章节从这里继续。" in kept


def test_sanitize_keeps_raster_data_image_but_strips_script_url() -> None:
    from app.formats import html_blocks as hb

    # Images are normally replaced by placeholders and not sent to the model; they go with the HTML only when the
    # original already contains {vN}, so sanitizing is checked directly here
    src = 'Text {v1} <img src="data:image/png;base64,AAAA"/> with <a href="https://x">link</a>'
    nodes = hb._sanitize_translation(
        src,
        '文字 {v1} <img src="data:image/png;base64,AAAA" onerror="alert(1)"/>'
        '<img src="data:image/svg+xml;base64,PHN2Zw=="/> 和 '
        '<a href="javascript:alert(1)">坏链接</a><a href="https://x">链接</a>',
    )
    assert nodes is not None
    soup = BeautifulSoup("<p></p>", "lxml")
    p = _tag(soup.p)
    for n in nodes:
        p.append(n)
    raster, svg = soup.find_all("img")
    assert _attr(raster, "src").startswith("data:image/png") and not raster.has_attr("onerror")
    assert not svg.has_attr("src")
    bad, good = soup.find_all("a")
    assert not bad.has_attr("href") and good["href"] == "https://x"
    assert hb._unsafe_url("data:image/jpeg;base64,AAAA", "image") is False
    assert hb._unsafe_url("data:image/svg+xml,<svg></svg>", "image") is True


MATH_PARA = (
    '<p>Let <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math> be a variable '
    'and <img src="eq1.png" alt="y=2x"/> holds.</p>'
)


def _order(p: Tag | None) -> list[str]:
    """Order of text and media elements in a paragraph: text as it is, media elements by tag name."""
    return [c.name if isinstance(c, Tag) else str(c) for c in _tag(p).contents]


def test_epub_inline_math_and_images_stay_in_place(tmp_path: Path) -> None:
    """MathML and images in a paragraph are sent to the model as {vN} and put back at their original positions after
    translation; the bilingual version has them too."""
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


def test_rich_paragraph_sends_placeholders_instead_of_mathml() -> None:
    """MathML in a rich-text paragraph is not sent to the model (<mtext> would get translated); it is replaced by a
    placeholder as on the plain-text path."""
    from app.formats import html_blocks as hb

    body = (
        '<p>Let <em>f</em> be <math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<mrow><mi>x</mi><mtext>for all</mtext></mrow></math> here.</p>"
    )
    soup = hb.parse(_xhtml(body), xml=True)
    [(el, text, html)] = hb.find_blocks(soup)
    assert html == "Let <em>f</em> be {v1} here." and text == "Let f be {v1} here."
    hb.apply_translation_html(soup, el, html, "设 <em>f</em> 为 {v1}。", False, "zh-CN")
    p = _tag(soup.find("p"))
    assert _order(p) == ["设 ", "em", " 为 ", "math", "。"]
    assert _tag(p.find("mtext")).get_text() == "for all"


def test_lost_or_repeated_placeholders_keep_every_formula() -> None:
    """The model drops, repeats or misnumbers placeholders: extra ones are removed, and formulas not put back are
    appended to the paragraph end in their original order, so none is lost."""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml(MATH_PARA), xml=True)
    [(el, text, _)] = hb.find_blocks(soup)
    hb.apply_translation(soup, el, text, "设 {v2} 和 {v2} 以及 {v7}。", False, "zh-CN")
    p = soup.find("p")
    assert _order(p) == ["设 ", "img", " 和 ", " 以及 ", "。", "math"]


def test_identical_images_get_their_own_placeholders() -> None:
    """Two identical images get a number each: bs4 Tag equality compares content, so lookup cannot use equality."""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml('<p>A <img src="x.png"/> and B <img src="x.png"/> end.</p>'), xml=True)
    [(el, text, _)] = hb.find_blocks(soup)
    assert text == "A {v1} and B {v2} end."
    hb.apply_translation(soup, el, text, "乙 {v2} 甲 {v1} 完。", True, "zh-CN")
    assert len(soup.find_all("img")) == 4


def test_literal_placeholder_text_is_not_a_slot() -> None:
    """When the original already contains {v1} (template syntax), no placeholders are used and the original {v1} is
    translated as it is."""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml('<p>Use {v1} in the template <img src="x.png"/> here.</p>'), xml=True)
    [(el, text, _)] = hb.find_blocks(soup)
    assert text == "Use {v1} in the template here."
    hb.apply_translation(soup, el, text, "模板里用 {v1}。", False, "zh-CN")
    p = soup.find("p")
    assert _order(p) == ["img", "模板里用 {v1}。"]


def test_source_line_wraps_are_spaces_not_line_breaks(tmp_path: Path) -> None:
    """In a typeset XHTML paragraph the text spans several source lines and newlines are only whitespace; only <br>
    is a line break."""
    body = (
        "<p>Alice was beginning to get very tired of sitting\n    by her sister on the bank,<br/>\n"
        "    and of having nothing to do.</p>"
        "<p>\n<br/>First stanza line,<br/>\nsecond line.<br/>\n<br/>\nNext stanza here.<br/>\n</p>"
    )
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml(body)})
    r = runner()
    [out] = run(translate_epub(src, tmp_path, r, False, "zh-CN"))
    p, poem = BeautifulSoup(zipfile.ZipFile(out).read("OEBPS/ch1.xhtml"), "lxml-xml").find_all("p")
    assert len(p.find_all("br")) == 1
    assert p.get_text("|") == (
        "[zh-CN] Alice was beginning to get very tired of sitting by her sister on the bank,|"
        "and of having nothing to do."
    )
    # Two consecutive <br> between stanzas keep the blank line; <br> at the start or end of the paragraph does not
    # count
    assert poem.get_text("|") == "[zh-CN] First stanza line,|second line.|Next stanza here."
    assert [c.name for c in poem.contents if isinstance(c, Tag)] == ["br", "br", "br"]


def test_body_with_only_bare_text_is_translated(tmp_path: Path) -> None:
    """Text directly inside <body> without any block-level element (common in chapters converted from MOBI) is
    translated as well."""
    src = tmp_path / "book.epub"
    _one_epub(src, {"ch1.xhtml": _xhtml("Some chapter text.<br/>More <i>text</i>.")})
    [out] = run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))
    body = _tag(BeautifulSoup(zipfile.ZipFile(out).read("OEBPS/ch1.xhtml"), "lxml-xml").find("body"))
    assert "[zh-CN] Some chapter text." in body.get_text() and body.find("i") is not None


def test_paragraph_with_only_a_formula_is_not_sent() -> None:
    """A paragraph with a single formula and no text is not sent for translation."""
    from app.formats import html_blocks as hb

    soup = hb.parse(_xhtml('<p><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>x</mi></math></p>'), xml=True)
    assert hb.find_blocks(soup) == []


# --- The references section keeps the original text ---


def _biblio_epub(path: Path) -> None:
    ch1 = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        "<h1>Chapter One</h1><p>Normal paragraph here.</p>"
        "<h2>Bibliography</h2><p>Smith, J. (2020). Some book. Press.</p>"
        "<h3>Primary sources</h3><p>Jones, A. (2019). Another book.</p>"
        "<h2>Appendix</h2><p>Appendix paragraph here.</p></body></html>"
    )
    biblio = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        "<h1>References</h1><p>Doe, B. (2018). Cited work.</p></body></html>"
    )
    _write_epub(
        path,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="bib" href="bibliography.xhtml" media-type="application/xhtml+xml"/>',
        '<itemref idref="ch1"/><itemref idref="bib"/>',
        {"ch1.xhtml": ch1, "bibliography.xhtml": biblio},
    )


def test_epub_bibliography_kept_original(tmp_path: Path) -> None:
    src = tmp_path / "book.epub"
    _biblio_epub(src)
    r = runner()
    [out] = run(translate_epub(src, tmp_path, r, False, "zh-CN"))
    text = _epub_texts(out)
    # Body and appendix are translated; the references heading is translated; citation entries keep the original
    # text
    assert "[zh-CN] Normal paragraph here." in text
    assert "[zh-CN] Appendix paragraph here." in text
    assert "Smith, J. (2020). Some book. Press." in text and "[zh-CN] Smith" not in text
    assert "Jones, A. (2019). Another book." in text
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    # Marked as skipped in the preview
    updates = {u[0]: u for u in r.preview()["updates"]}
    segs = r.preview()["segments"]
    skipped_srcs = {segs[u[0]]["s"] for u in updates.values() if u[2] == 1}
    assert any("Smith, J." in s for s in skipped_srcs)
    assert any("Doe, B." in s for s in skipped_srcs)


def _one_epub(path: Path, docs: dict[str, str]) -> None:
    items = "".join(
        f'<item id="d{i}" href="{name}" media-type="application/xhtml+xml"/>' for i, name in enumerate(docs)
    )
    spine = "".join(f'<itemref idref="d{i}"/>' for i in range(len(docs)))
    _write_epub(path, items, spine, docs)


def _xhtml(body: str) -> str:
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        f"<body>{body}</body></html>"
    )


def test_epub_chapter_titled_papers_ends_bibliography(tmp_path: Path) -> None:
    """In a flat document h2 Papers is the next chapter and must not keep the original text because the word looks
    like a subheading."""
    src = tmp_path / "book.epub"
    _one_epub(
        src,
        {
            "book.xhtml": _xhtml(
                "<h1>References</h1><p>Doe, B. (2018). Cited work.</p>"
                "<h2>Papers</h2><p>Papers chapter paragraph.</p>"
                "<h2>Appendix</h2><p>Appendix paragraph here.</p>"
            )
        },
    )
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    assert "[zh-CN] Papers chapter paragraph." in text
    assert "[zh-CN] Appendix paragraph here." in text


def test_epub_deeper_heading_ends_flat_bibliography(tmp_path: Path) -> None:
    """In a single file the h2 appendix after the h1 references is the next chapter, so the rest of the document
    must not keep the original text.

    A deeper heading wrapped in the following section ends it too, without relying on heading numbers alone.
    """
    src = tmp_path / "book.epub"
    _one_epub(
        src,
        {
            "book.xhtml": _xhtml(
                "<h1>References</h1><p>Doe, B. (2018). Cited work.</p>"
                "<h2>Appendix</h2><p>Appendix paragraph here.</p>"
                "<section><h1>References</h1><p>Lee, C. (2017). Third work.</p></section>"
                "<section><h2>Appendix</h2><p>Second appendix paragraph.</p></section>"
            )
        },
    )
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    assert "[zh-CN] Appendix paragraph here." in text
    assert "Lee, C. (2017). Third work." in text and "[zh-CN] Lee" not in text
    assert "[zh-CN] Second appendix paragraph." in text


def test_epub_filename_substring_does_not_skip_chapter(tmp_path: Path) -> None:
    src = tmp_path / "book.epub"
    _one_epub(
        src,
        {
            "chapter-references.xhtml": _xhtml("<h1>Cross References in Plato</h1><p>Normal paragraph here.</p>"),
            "bibliographical-notes.xhtml": _xhtml("<h1>Notes</h1><p>Another normal paragraph.</p>"),
        },
    )
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "[zh-CN] Cross References in Plato" in text
    assert "[zh-CN] Normal paragraph here." in text
    assert "[zh-CN] Another normal paragraph." in text


def test_epub_bibliography_file_still_exits_on_next_heading(tmp_path: Path) -> None:
    src = tmp_path / "book.epub"
    _one_epub(
        src,
        {
            "bibliography.xhtml": _xhtml(
                "<h1>References</h1><p>Doe, B. (2018). Cited work.</p><h1>Index</h1><p>Index entry text here.</p>"
            )
        },
    )
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Doe, B. (2018). Cited work." in text and "[zh-CN] Doe" not in text
    assert "[zh-CN] Index entry text here." in text


def test_epub_bibliography_continues_across_spine(tmp_path: Path) -> None:
    src = tmp_path / "book.epub"
    _one_epub(
        src,
        {
            "ch1.xhtml": _xhtml("<h2>Bibliography</h2><p>Smith, J. (2020). Some book. Press.</p>"),
            "index_split_002.xhtml": _xhtml(
                "<p>Jones, A. (2019). Another book.</p><h1>Appendix</h1><p>Appendix paragraph here.</p>"
            ),
        },
    )
    text = _epub_texts(run(translate_epub(src, tmp_path, runner(), False, "zh-CN"))[0])
    assert "Smith, J. (2020). Some book. Press." in text and "[zh-CN] Smith" not in text
    assert "Jones, A. (2019). Another book." in text and "[zh-CN] Jones" not in text
    assert "[zh-CN] Appendix paragraph here." in text


def test_epub_repeated_citation_in_body_is_translated(tmp_path: Path) -> None:
    src = tmp_path / "book.epub"
    cite = "Smith, J. (2020). Some book. Press."
    _one_epub(
        src,
        {
            "ch1.xhtml": _xhtml(
                f"<p>{cite}</p><h2>Bibliography</h2><p>{cite}</p><h2>Appendix</h2><p>Appendix paragraph here.</p>"
            )
        },
    )
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


def test_epub_styled_heading_colon_and_epub_type(tmp_path: Path) -> None:
    src = tmp_path / "book.epub"
    _one_epub(
        src,
        {
            "ch1.xhtml": _xhtml(
                '<p class="heading">Bibliography</p><p>Smith, J. (2020). Some book. Press.</p>'
                "<h2>Appendix</h2><p>Appendix paragraph here.</p>"
                "<h2>参考文献：</h2><p>Doe, B. (2018). Cited work.</p>"
                "<h2>Index</h2><p>Index entry text here.</p>"
                '<section epub:type="bibliography"><h2>Notes</h2>'
                "<p>Lee, C. (2017). Third work.</p></section>"
                "<h2>Afterword</h2><p>Afterword paragraph here.</p>"
                "<p>Bibliography</p><p>This sentence mentions Bibliography but is body text.</p>"
                "<h2>12. References</h2><p>Ng, D. (2016). Fourth work.</p>"
                "<h2>Glossary</h2><p>Glossary paragraph here.</p>"
            )
        },
    )
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


def test_preview_override_replaces_shared_translation() -> None:
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache())
    runner.segments = [{"s": "2 Proposed", "k": "h2"}]
    runner._indices = {"2 Proposed": [0]}
    runner._done = {"2 Proposed": "第二章 所提"}
    runner._log = ["2 Proposed"]
    runner._index_overrides[0] = "2 所提"
    assert runner.preview()["updates"] == [[0, "2 所提", 0]]
