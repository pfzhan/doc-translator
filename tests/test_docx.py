"""Word documents (app/formats/docx.py)."""

import zipfile
from pathlib import Path

from lxml import etree

from app.formats import translate_file
from app.formats.docx import translate_docx
from app.runner import Runner
from app.translators import MockTranslator
from tests.helpers import MemCache, run, runner

NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
STYLES = (
    f'<w:styles {NS}><w:style w:type="paragraph" w:styleId="1"><w:name w:val="heading 1"/>'
    '<w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style></w:styles>'
)


def make_docx(path: Path, body: str, footnotes: str = "", header: str = "", title: str = "") -> Path:
    """A minimal package with the parts Word writes: content types, relationships, styles, and optionally footnotes,
    a header and document properties."""
    rels = [
        f'<Relationship Id="rS" Type="{REL}/styles" Target="styles.xml"/>',
        f'<Relationship Id="rL" Type="{REL}/hyperlink" Target="https://example.com" TargetMode="External"/>',
    ]
    if footnotes:
        rels.append(f'<Relationship Id="rF" Type="{REL}/footnotes" Target="footnotes.xml"/>')
    if header:
        rels.append(f'<Relationship Id="rH" Type="{REL}/header" Target="header1.xml"/>')
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>'
        )
        z.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Type="{REL}/officeDocument" '
            'Target="word/document.xml"/></Relationships>',
        )
        z.writestr("word/_rels/document.xml.rels", f'<Relationships xmlns="{pkg}">{"".join(rels)}</Relationships>')
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/document.xml", f"<w:document {NS}><w:body>{body}</w:body></w:document>")
        if footnotes:
            z.writestr("word/footnotes.xml", f"<w:footnotes {NS}>{footnotes}</w:footnotes>")
        if header:
            z.writestr("word/header1.xml", f"<w:hdr {NS}>{header}</w:hdr>")
        if title:
            z.writestr(
                "docProps/core.xml",
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
                f' xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title></cp:coreProperties>',
            )
    return path


def part(path: Path, name: str = "word/document.xml") -> etree._Element:
    with zipfile.ZipFile(path) as z:
        return etree.fromstring(z.read(name))


def paragraphs(root: etree._Element) -> list[etree._Element]:
    return list(root.iter(f"{W}p"))


def text(p: etree._Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{W}t"))


RICH = (
    '<w:p><w:r><w:t xml:space="preserve">The </w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>red</w:t></w:r>'
    '<w:r><w:t xml:space="preserve"> car is </w:t></w:r><w:hyperlink r:id="rL"><w:r><w:rPr>'
    '<w:rStyle w:val="Hyperlink"/></w:rPr><w:t>here</w:t></w:r></w:hyperlink><w:r><w:t>.</w:t></w:r>'
    '<w:r><w:footnoteReference w:id="1"/></w:r></w:p>'
)
FIELD = (
    '<w:p><w:r><w:t xml:space="preserve">See page </w:t></w:r><w:r><w:fldChar w:fldCharType="begin"/></w:r>'
    '<w:r><w:instrText>PAGE</w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    '<w:r><w:t>3</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>'
    '<w:r><w:t xml:space="preserve"> of the report</w:t></w:r></w:p>'
)
FOOTNOTE = (
    '<w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r><w:r><w:t>A footnote sentence.</w:t></w:r></w:p></w:footnote>'
)


def test_docx_translates_body_tables_headers_and_footnotes(tmp_path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:bookmarkStart w:id="0" w:name="_Toc1"/>'
        '<w:r><w:t>Introduction to data</w:t></w:r><w:bookmarkEnd w:id="0"/></w:p>'
        + RICH
        + "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell text here</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        "<w:p><w:r><w:t>2024</w:t></w:r></w:p>"
    )
    src = make_docx(tmp_path / "a.docx", body, FOOTNOTE, "<w:p><w:r><w:t>Running header text</w:t></w:r></w:p>")
    out = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))[0]
    assert out.name == "a.translated.docx"
    doc = part(out)
    texts = [text(p) for p in paragraphs(doc)]
    assert texts == ["[zh-CN] Introduction to data", "[zh-CN] The red car is here.", "[zh-CN] Cell text here", "2024"]
    # The bookmark around the heading still wraps its text, so cross-references keep working
    heading = list(paragraphs(doc)[0])
    assert [el.tag.split("}")[1] for el in heading] == ["pPr", "bookmarkStart", "r", "bookmarkEnd"]
    # Bold stays on its word, the link keeps its relationship id and the footnote reference stays at the end
    rich = paragraphs(doc)[1]
    bold = [text(r) for r in rich.iter(f"{W}r") if r.find(f"{W}rPr/{W}b") is not None]
    assert bold == ["red"]
    link = rich.find(f"{W}hyperlink")
    assert link.get(f"{{{REL}}}id") == "rL" and text(link) == "here"
    assert rich[-1].find(f"{W}footnoteReference") is not None
    assert text(part(out, "word/footnotes.xml")) == "[zh-CN] A footnote sentence."
    assert text(part(out, "word/header1.xml")) == "[zh-CN] Running header text"


class Reordering(MockTranslator):
    """Moves the formatted word to the front and the content placeholders out of order, as target word order can."""

    async def translate_batch(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            if '<b id="s1">red</b>' in t:
                out.append('<b id="s1">红色的</b>汽车在<a id="s3">这里</a>{v1}。')
            elif "{v2}" in t:
                out.append("见报告第 {v2}3{v1} 页")
            else:
                out.append(f"[{self.target_lang}] {t}")
        return out


def test_docx_follows_the_translation_order_and_keeps_fields_whole(tmp_path: Path) -> None:
    src = make_docx(tmp_path / "a.docx", RICH + FIELD)
    out = run(translate_docx(src, tmp_path, Runner(Reordering("zh-CN"), cache=MemCache()), False))[0]
    rich, field = paragraphs(part(out))
    runs = [(text(r), r.find(f"{W}rPr/{W}b") is not None) for r in rich.iter(f"{W}r") if text(r)]
    assert runs == [("红色的", True), ("汽车在", False), ("这里", False), ("。", False)]
    assert text(rich.find(f"{W}hyperlink")) == "这里"
    # The footnote reference came back where its placeholder was, before the full stop
    tags = [el.tag.split("}")[1] for r in rich.iter(f"{W}r") for el in r if el.tag != f"{W}rPr"]
    assert tags == ["t", "t", "t", "footnoteReference", "t"]
    # The field's end placeholder came first in the translation; its begin still comes before its end
    chars = [c.get(f"{W}fldCharType") for c in field.iter(f"{W}fldChar")]
    assert chars == ["begin", "separate", "end"]
    assert text(field) == "见报告第 3 页"


def test_docx_bilingual_adds_a_copy_without_numbering_or_unique_content(tmp_path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="1"/><w:keepNext/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="3"/></w:numPr>'
        '<w:jc w:val="left"/></w:pPr><w:bookmarkStart w:id="0" w:name="_Toc1"/><w:r><w:t>Methods used</w:t></w:r>'
        '<w:bookmarkEnd w:id="0"/></w:p>' + RICH
    )
    src = make_docx(tmp_path / "a.docx", body, FOOTNOTE)
    out = run(translate_file(src, tmp_path, runner(), True, "zh-CN"))[0]
    assert out.name == "a.bilingual.docx"
    heading, heading_copy, rich, rich_copy = paragraphs(part(out))
    assert text(heading) == "Methods used" and text(heading_copy) == "[zh-CN] Methods used"
    # numId 0 turns numbering off; numPr sits where the schema puts it, after keepNext and before jc
    ppr = heading_copy.find(f"{W}pPr")
    assert [el.tag.split("}")[1] for el in ppr] == ["pStyle", "keepNext", "numPr", "jc"]
    assert ppr.find(f"{W}numPr/{W}numId").get(f"{W}val") == "0"
    assert heading_copy.find(f"{W}bookmarkStart") is None and heading.find(f"{W}bookmarkStart") is not None
    assert rich.find(f".//{W}footnoteReference") is not None
    assert rich_copy.find(f".//{W}footnoteReference") is None
    assert text(rich_copy) == "[zh-CN] The red car is here."
    note, note_copy = paragraphs(part(out, "word/footnotes.xml"))
    assert note.find(f".//{W}footnoteRef") is not None and note_copy.find(f".//{W}footnoteRef") is None


def test_docx_bilingual_section_break_moves_to_the_copy(tmp_path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:jc w:val="left"/><w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr></w:pPr>'
        "<w:r><w:t>Last paragraph of section one.</w:t></w:r></w:p><w:p><w:r><w:t>Next section here.</w:t></w:r></w:p>"
    )
    out = run(translate_file(make_docx(tmp_path / "a.docx", body), tmp_path, runner(), True, "zh-CN"))[0]
    last, last_copy, _, _ = paragraphs(part(out))
    assert last.find(f"{W}pPr/{W}sectPr") is None and last.find(f"{W}pPr/{W}jc") is not None
    assert last_copy.find(f"{W}pPr/{W}sectPr") is not None


def test_docx_bilingual_copy_keeps_field_results_without_the_fields(tmp_path: Path) -> None:
    caption = (
        '<w:p><w:r><w:t xml:space="preserve">Figure </w:t></w:r><w:fldSimple w:instr=" SEQ Figure \\* ARABIC ">'
        '<w:r><w:t>1</w:t></w:r></w:fldSimple><w:r><w:t xml:space="preserve">: A caption here</w:t></w:r></w:p>'
    )
    out = run(translate_file(make_docx(tmp_path / "a.docx", caption + FIELD), tmp_path, runner(), True, "zh-CN"))[0]
    fig, fig_copy, field, field_copy = paragraphs(part(out))
    # Figure numbers count once; the copies show the results as plain text
    assert fig.find(f"{W}fldSimple") is not None and fig_copy.find(f"{W}fldSimple") is None
    assert text(fig_copy) == "[zh-CN] Figure 1: A caption here"
    assert len(field.findall(f".//{W}fldChar")) == 3 and field_copy.find(f".//{W}fldChar") is None
    assert field_copy.find(f".//{W}instrText") is None and text(field_copy) == "[zh-CN] See page 3 of the report"


def test_docx_bookmarks_inside_a_link_stay_around_its_text(tmp_path: Path) -> None:
    body = (
        '<w:p><w:r><w:t xml:space="preserve">See </w:t></w:r><w:hyperlink r:id="rL">'
        '<w:bookmarkStart w:id="5" w:name="target"/><w:r><w:rPr><w:rStyle w:val="Hyperlink"/></w:rPr>'
        '<w:t>this link</w:t></w:r><w:commentRangeStart w:id="2"/><w:bookmarkEnd w:id="5"/></w:hyperlink>'
        '<w:r><w:t xml:space="preserve"> for details.</w:t></w:r></w:p>'
    )
    out = run(translate_file(make_docx(tmp_path / "a.docx", body), tmp_path, runner(), False, "zh-CN"))[0]
    link = paragraphs(part(out))[0].find(f"{W}hyperlink")
    assert [el.tag.split("}")[1] for el in link] == ["bookmarkStart", "r", "commentRangeStart", "bookmarkEnd"]
    assert text(link) == "this link"


def test_docx_placeholders_written_in_the_text_stay(tmp_path: Path) -> None:
    body = (
        '<w:p><w:r><w:t xml:space="preserve">Write {v1} where </w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r>'
        "<w:r><w:t>the value goes.</w:t></w:r></w:p>"
    )
    out = run(translate_file(make_docx(tmp_path / "a.docx", body, FOOTNOTE), tmp_path, runner(), False, "zh-CN"))[0]
    p = paragraphs(part(out))[0]
    assert text(p) == "[zh-CN] Write {v1} where the value goes."
    tags = [el.tag.split("}")[1] for r in p.iter(f"{W}r") for el in r if el.tag != f"{W}rPr"]
    assert tags == ["t", "footnoteReference", "t"]


def test_docx_smart_tag_properties_stay_first_and_are_not_sent(tmp_path: Path) -> None:
    body = (
        '<w:p><w:r><w:t xml:space="preserve">Meet me in </w:t></w:r><w:smartTag w:uri="u" w:element="place">'
        '<w:smartTagPr><w:attr w:name="n" w:val="v"/></w:smartTagPr><w:bookmarkStart w:id="1" w:name="m"/>'
        '<w:r><w:t>Boston</w:t></w:r><w:bookmarkEnd w:id="1"/></w:smartTag><w:r><w:t> tomorrow.</w:t></w:r></w:p>'
    )
    sent: list[str] = []

    class Recording(MockTranslator):
        async def translate_batch(self, texts: list[str]) -> list[str]:
            sent.extend(texts)
            return await super().translate_batch(texts)

    src = make_docx(tmp_path / "a.docx", body)
    out = run(translate_docx(src, tmp_path, Runner(Recording("zh-CN"), cache=MemCache()), False))[0]
    # The properties are not content: no placeholder stands for them
    assert sent == ['Meet me in <span id="s1">Boston</span> tomorrow.']
    tag = paragraphs(part(out))[0].find(f"{W}smartTag")
    assert [el.tag.split("}")[1] for el in tag] == ["smartTagPr", "bookmarkStart", "r", "bookmarkEnd"]


def test_docx_text_beside_a_tab_in_one_run_is_translated(tmp_path: Path) -> None:
    body = "<w:p><w:r><w:t>Name</w:t><w:tab/><w:t>Value here</w:t></w:r></w:p>"
    out = run(translate_file(make_docx(tmp_path / "a.docx", body), tmp_path, runner(), False, "zh-CN"))[0]
    p = paragraphs(part(out))[0]
    assert text(p) == "[zh-CN] NameValue here"
    assert p.find(f".//{W}tab") is not None


def test_docx_bibliography_keeps_the_original(tmp_path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>References</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Smith, J. (2020). Some book. Press.</w:t></w:r></w:p>"
        '<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>Appendix</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Appendix paragraph here.</w:t></w:r></w:p>"
    )
    out = run(translate_file(make_docx(tmp_path / "a.docx", body), tmp_path, runner(), False, "zh-CN"))[0]
    assert [text(p) for p in paragraphs(part(out))] == [
        "[zh-CN] References",
        "Smith, J. (2020). Some book. Press.",
        "[zh-CN] Appendix",
        "[zh-CN] Appendix paragraph here.",
    ]


def test_docx_preview_kinds_and_title(tmp_path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>Chapter heading</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>A list item</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Body paragraph.</w:t></w:r></w:p>"
    )

    class Titled(MockTranslator):
        title = ""  # LLM engines have a title; set_title fills it

    translator = Titled("zh-CN")
    r = Runner(translator, cache=MemCache())
    run(translate_file(make_docx(tmp_path / "a.docx", body, title="Report title"), tmp_path, r, False, "zh-CN"))
    assert r.segments is not None
    assert [s["k"] for s in r.segments] == ["h1", "li", "p"]
    assert translator.title == "Report title"


def test_docx_other_parts_are_copied_unchanged(tmp_path: Path) -> None:
    src = make_docx(tmp_path / "a.docx", "<w:p><w:r><w:t>Hello there</w:t></w:r></w:p>")
    out = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))[0]
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(out) as b:
        assert a.namelist() == b.namelist()
        assert all(a.read(n) == b.read(n) for n in a.namelist() if n != "word/document.xml")
