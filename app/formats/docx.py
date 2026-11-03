"""Word (.docx) translation: paragraphs of the body, tables, text boxes, headers, footers, footnotes and endnotes are
translated; styles, numbering, images, fields and everything else in the package are kept.

A paragraph is sent as one text. Runs with the paragraph's main formatting are plain text; runs with other
formatting (bold, italic, links...) are wrapped in numbered tags, so the translation keeps them on the matching
words. Content that is not text (images, tabs, breaks, footnote references, fields) becomes {vN} and is put back in
place. In the bilingual version each translated paragraph is followed by a copy holding the translation.
"""

import copy
import html
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
from lxml import html as lxml_html

from .html_blocks import BIBLIO_HEADING_RE, translatable

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DC_NS = "http://purl.org/dc/elements/1.1/"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
SUFFIXES = {".docx"}


def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


# Elements around runs that keep their runs in the paragraph's text: links, tracked insertions, smart tags
_WRAPPERS = {w("hyperlink"), w("ins"), w("smartTag"), w("customXml")}
# Properties of a wrapper: its first child, copied to each copy of it
_WRAPPER_PROPS = {w("smartTagPr"), w("customXmlPr")}
# Run children that are text, or carry nothing that has a position in the text
_TEXT_CHILDREN = {w("rPr"), w("t"), w("lastRenderedPageBreak"), w("proofErr")}
# Paragraph children with no position in the text; they stay where they are
_MARKERS = {
    w("pPr"),
    w("bookmarkStart"),
    w("bookmarkEnd"),
    w("commentRangeStart"),
    w("commentRangeEnd"),
    w("proofErr"),
    w("permStart"),
    w("permEnd"),
}
# Content that must appear once in a document: left out of the bilingual copy
_UNIQUE = {
    w("footnoteReference"),
    w("endnoteReference"),
    w("footnoteRef"),
    w("endnoteRef"),
    w("commentReference"),
    w("drawing"),
    w("object"),
    w("pict"),
}
# Field codes: the bilingual copy keeps only a field's result text, so fields that count (figure numbers, SEQ) do not
# count twice and fields spanning paragraphs (a table of contents) stay balanced
_FIELD_CODES = {w("fldChar"), w("instrText"), w("delInstrText")}
_PLACEHOLDER_RE = re.compile(r"\{v(\d+)\}")
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True, remove_blank_text=False)
# Children of w:pPr that come before w:numPr in the schema; Word rejects a document with them out of order
_BEFORE_NUMPR = {w("pStyle"), w("keepNext"), w("keepLines"), w("pageBreakBefore"), w("framePr"), w("widowControl")}
# Word part types translated besides the main document, by relationship type
_PART_TYPES = ("/header", "/footer", "/footnotes", "/endnotes")


@dataclass
class _Segment:
    """Adjacent text runs with the same formatting inside the same wrapper."""

    runs: list[etree._Element]
    wrapper: etree._Element | None
    key: str
    text: str = ""


@dataclass
class _Atom:
    """Adjacent runs or elements that are not text; they become one {vN}."""

    nodes: list[etree._Element]
    wrappers: list[etree._Element | None]


@dataclass
class _Paragraph:
    p: etree._Element
    items: list[_Segment | _Atom] = field(default_factory=list)
    text: str = ""
    html: str | None = None
    kind: str = "p"
    skip: bool = False


def _is_text_run(run: etree._Element) -> bool:
    return all(child.tag in _TEXT_CHILDREN for child in run) and any(child.tag == w("t") for child in run)


def _run_key(run: etree._Element, wrapper: etree._Element | None) -> str:
    rpr = run.find(w("rPr"))
    style = etree.tostring(rpr) if rpr is not None else b""
    return f"{id(wrapper)}:{style!r}"


def _split_mixed_runs(p: etree._Element) -> None:
    """A run holding text together with a tab or a break is split into one run per child, so its text is
    translated and the tab or break keeps its place."""
    for run in list(p.iter(w("r"))):
        content = [c for c in run if c.tag != w("rPr")]
        if len(content) < 2 or _is_text_run(run) or not any(c.tag == w("t") for c in content):
            continue
        rpr = run.find(w("rPr"))
        for child in content:
            piece = etree.Element(w("r"), run.attrib, nsmap=run.nsmap)
            if rpr is not None:
                piece.append(copy.deepcopy(rpr))
            piece.append(child)
            run.addprevious(piece)
        run.getparent().remove(run)


def _items(p: etree._Element) -> list[_Segment | _Atom]:
    """Runs directly in the paragraph or one wrapper deep, grouped into text segments and content. The runs of a
    field's code hold w:instrText and count as content, so the field stays whole; its result is text."""
    _split_mixed_runs(p)
    items: list[_Segment | _Atom] = []
    for child in p:
        if child.tag in _MARKERS:
            continue
        if child.tag in _WRAPPERS:
            members = [(g, child) for g in child if g.tag not in _MARKERS | _WRAPPER_PROPS]
        else:
            members = [(child, None)]
        for node, wrapper in members:
            last = items[-1] if items else None
            if node.tag == w("r") and _is_text_run(node):
                key = _run_key(node, wrapper)
                text = "".join(t.text or "" for t in node.iter(w("t")))
                if isinstance(last, _Segment) and last.key == key:
                    last.runs.append(node)
                    last.text += text
                else:
                    items.append(_Segment([node], wrapper, key, text))
            elif isinstance(last, _Atom):
                last.nodes.append(node)
                last.wrappers.append(wrapper)
            else:
                items.append(_Atom([node], [wrapper]))
    return items


def _tag_of(segment: _Segment) -> str:
    if segment.wrapper is not None and segment.wrapper.tag == w("hyperlink"):
        return "a"
    rpr = segment.runs[0].find(w("rPr"))
    for name in ("b", "i", "u"):
        el = rpr.find(w(name)) if rpr is not None else None
        if el is not None and el.get(w("val"), "true") not in ("0", "false", "none"):
            return name
    return "span"


def _main_key(items: list[_Segment | _Atom]) -> str:
    """Formatting that covers most of the paragraph's text; its runs are sent untagged."""
    sizes: dict[str, int] = {}
    for item in items:
        if isinstance(item, _Segment):
            sizes[item.key] = sizes.get(item.key, 0) + len(item.text.strip())
    return max(sizes, key=lambda k: sizes[k])


def _first_atom(items: list[_Segment | _Atom]) -> int:
    """Content is numbered after the highest {vN} written in the text itself, so that text is not taken for
    content."""
    texts = [i.text for i in items if isinstance(i, _Segment)]
    return max((int(n) for text in texts for n in _PLACEHOLDER_RE.findall(text)), default=0)


def _source(items: list[_Segment | _Atom]) -> tuple[str, str | None]:
    """(plain text, HTML or None). HTML is used only when some text has formatting other than the main one."""
    main = _main_key(items)
    rich = any(isinstance(i, _Segment) and i.key != main and i.text.strip() for i in items)
    plain, marked = [], []
    atom = _first_atom(items)
    for n, item in enumerate(items):
        if isinstance(item, _Atom):
            atom += 1
            plain.append(f"{{v{atom}}}")
            marked.append(f"{{v{atom}}}")
        else:
            plain.append(item.text)
            escaped = html.escape(item.text, quote=False)
            if item.key == main or not item.text.strip():
                marked.append(escaped)
            else:
                tag = _tag_of(item)
                marked.append(f'<{tag} id="s{n}">{escaped}</{tag}>')
    text = " ".join("".join(plain).split())
    return text, (" ".join("".join(marked).split()) if rich else None)


def _pieces(items: list[_Segment | _Atom], translated: str, rich: bool) -> list[tuple[int | None, str]]:
    """Split a translation into (item index, text) in order; None is the main formatting. Tags the source does
    not have count as the formatting around them."""
    if not rich:
        return [(None, translated)]
    try:
        root = lxml_html.fragment_fromstring(translated, create_parent="div")
    except (etree.ParserError, ValueError):
        return [(None, re.sub(r"<[^<>]*>", "", translated))]
    out: list[tuple[int | None, str]] = []

    def collect(el, owner: int | None) -> None:
        mark = el.get("id") or ""
        if mark[:1] == "s" and mark[1:].isdigit() and int(mark[1:]) < len(items):
            if isinstance(items[int(mark[1:])], _Segment):
                owner = int(mark[1:])
        if el.text:
            out.append((owner, el.text))
        for child in el:
            collect(child, owner)
            if child.tail:
                out.append((owner, child.tail))

    collect(root, None)
    return out


def _new_run(template: etree._Element, text: str) -> etree._Element:
    run = etree.Element(w("r"), template.attrib, nsmap=template.nsmap)
    rpr = template.find(w("rPr"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    t = etree.SubElement(run, w("t"))
    t.text = text
    t.set(XML_SPACE, "preserve")
    return run


def _write(para: _Paragraph, translated: str, drop_unique: bool = False) -> None:
    """Replace the paragraph's runs with the translation, in its order. Placeholders put the content back in
    place; content whose placeholder was lost goes after the text, in its original order. With drop_unique (the
    bilingual copy) content that must appear once and field codes are left out."""
    items = para.items
    segments = [i for i in items if isinstance(i, _Segment)]
    atoms = [i for i in items if isinstance(i, _Atom)]
    main = _main_key(items)
    main_segment = next(s for s in segments if s.key == main)
    first = _first_atom(items)
    # Elements to emit: (element, wrapper it belonged to)
    emitted: list[tuple[etree._Element, etree._Element | None]] = []
    placed: set[int] = set()

    def text_out(index: int | None, text: str) -> None:
        if not text:
            return
        segment = items[index] if index is not None else main_segment
        emitted.append((_new_run(segment.runs[0], text), segment.wrapper))

    def atom_out(atom: _Atom) -> None:
        for node, wrapper in zip(atom.nodes, atom.wrappers):
            # A simple field is replaced by the runs of its result
            nodes = [c for c in node if c.tag == w("r")] if drop_unique and node.tag == w("fldSimple") else [node]
            for el in nodes:
                if not (drop_unique and any(d.tag in _UNIQUE or d.tag in _FIELD_CODES for d in el.iter())):
                    emitted.append((el, wrapper))

    for index, text in _pieces(items, translated, para.html is not None):
        pos = 0
        for m in _PLACEHOLDER_RE.finditer(text):
            # {vN} outside the content's numbers is text
            if not first < int(m.group(1)) <= first + len(atoms):
                continue
            text_out(index, text[pos : m.start()])
            # Content keeps its order: a field's end must not come before its start. A placeholder brings along
            # the earlier ones not yet placed
            for n in range(1, int(m.group(1)) - first + 1):
                if n not in placed:
                    placed.add(n)
                    atom_out(atoms[n - 1])
            pos = m.end()
        text_out(index, text[pos:])
    for n, atom in enumerate(atoms, 1):
        if n not in placed:
            atom_out(atom)

    p = para.p
    # Everything that held runs leaves the paragraph; markers (bookmarks, comment ranges) stay where they are
    old = [c for c in p if c.tag not in _MARKERS]
    anchor = old[0] if old else None
    index = list(p).index(anchor) if anchor is not None else len(p)
    for c in old:
        p.remove(c)
    # Copies of each wrapper, in order: a wrapper split by the translation's word order has several
    copies: dict[int, list[etree._Element]] = {}
    last_wrapper = None
    for element, wrapper in emitted:
        if wrapper is None:
            p.insert(index, element)
            index += 1
            last_wrapper = None
            continue
        if last_wrapper is not wrapper:
            shell = etree.Element(wrapper.tag, wrapper.attrib, nsmap=wrapper.nsmap)
            shell.extend(copy.deepcopy(c) for c in wrapper if c.tag in _WRAPPER_PROPS)
            copies.setdefault(id(wrapper), []).append(shell)
            p.insert(index, shell)
            index += 1
            last_wrapper = wrapper
        copies[id(wrapper)][-1].append(element)
    # Markers inside a wrapper (a bookmark on a link's text) go into its copies: those before its runs at the start
    # of the first copy, the others at the end of the last. Without a copy they stay in the paragraph
    for wrapper in old:
        if wrapper.tag not in _WRAPPERS:
            continue
        runs = [i for i, c in enumerate(wrapper) if c.tag not in _MARKERS | _WRAPPER_PROPS]
        lead = [c for i, c in enumerate(wrapper) if c.tag in _MARKERS and i < (runs[0] if runs else len(wrapper))]
        tail = [c for c in wrapper if c.tag in _MARKERS and c not in lead]
        shells = copies.get(id(wrapper))
        if shells is None:
            for c in [*lead, *tail]:
                p.insert(index, c)
                index += 1
            continue
        props = sum(c.tag in _WRAPPER_PROPS for c in shells[0])
        for c in reversed(lead):
            shells[0].insert(props, c)
        shells[-1].extend(tail)


def _style_kinds(zf: zipfile.ZipFile, styles_name: str) -> dict[str, str]:
    """Preview type of each paragraph style: headings by outline level or name, tables of contents, quotes."""
    if styles_name not in zf.namelist():
        return {}
    root = etree.fromstring(zf.read(styles_name), _PARSER)
    kinds = {}
    for style in root.iter(w("style")):
        sid = style.get(w("styleId")) or ""
        name_el = style.find(w("name"))
        name = (name_el.get(w("val")) if name_el is not None else "").lower()
        level = style.find(f"{w('pPr')}/{w('outlineLvl')}")
        if name == "title":
            kinds[sid] = "h1"
        elif level is not None and (level.get(w("val")) or "").isdigit() and int(level.get(w("val"))) < 6:
            kinds[sid] = f"h{int(level.get(w('val'))) + 1}"
        elif m := re.match(r"heading (\d)", name):
            kinds[sid] = f"h{min(int(m.group(1)), 6)}"
        elif name.startswith("toc"):
            kinds[sid] = "toc"
        elif "quote" in name:
            kinds[sid] = "quote"
    return kinds


def _kind(p: etree._Element, styles: dict[str, str]) -> str:
    ppr = p.find(w("pPr"))
    style = ppr.find(w("pStyle")) if ppr is not None else None
    kind = styles.get(style.get(w("val")) if style is not None else "", "")
    if kind:
        return kind
    if ppr is not None and ppr.find(w("numPr")) is not None:
        return "li"
    if any(a.tag == w("tc") for a in p.iterancestors()):
        return "td"
    return "p"


def _bibliography_skips(paras: list[_Paragraph]) -> None:
    """Paragraphs under a bibliography heading keep the original text, until a heading of the same or a higher
    level, as in the other formats."""
    level = 0
    for para in paras:
        if para.kind.startswith("h"):
            lv = int(para.kind[1])
            if len(para.text) <= 40 and BIBLIO_HEADING_RE.match(para.text):
                level = lv
            elif level and lv <= level:
                level = 0
            continue
        para.skip = bool(level)


def _rels(zf: zipfile.ZipFile, part: str) -> list[tuple[str, str]]:
    """(relationship type, target part name) of a part."""
    folder, name = posixpath.split(part)
    rels = posixpath.join(folder, "_rels", f"{name}.rels")
    if rels not in zf.namelist():
        return []
    root = etree.fromstring(zf.read(rels), _PARSER)
    out = []
    for rel in root.iter(f"{{{REL_NS}}}Relationship"):
        if rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target") or ""
        path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
        out.append((rel.get("Type") or "", path))
    return out


def _parts(zf: zipfile.ZipFile) -> tuple[str, list[str], str]:
    """(main document, parts to translate in order, styles part)."""
    main = next((t for kind, t in _rels(zf, "") if kind.endswith("/officeDocument")), "word/document.xml")
    if main not in zf.namelist():
        raise ValueError("不是有效的 Word 文档：缺少 document.xml")
    related = _rels(zf, main)
    others = [t for kind, t in related if kind.endswith(_PART_TYPES) and t in zf.namelist()]
    styles = next((t for kind, t in related if kind.endswith("/styles")), "word/styles.xml")
    return main, [main, *dict.fromkeys(others)], styles


def _title(zf: zipfile.ZipFile) -> str:
    if "docProps/core.xml" not in zf.namelist():
        return ""
    el = etree.fromstring(zf.read("docProps/core.xml"), _PARSER).find(f"{{{DC_NS}}}title")
    return (el.text or "").strip() if el is not None else ""


def _bilingual_copy(para: _Paragraph) -> _Paragraph:
    """A copy of the paragraph after it, to hold the translation. It has no numbering (a list item would count
    twice), no bookmarks (their names must be unique) and no ids of the original. A section break moves to the
    copy, which now ends the section."""
    clone = copy.deepcopy(para.p)
    sect = para.p.find(f"{w('pPr')}/{w('sectPr')}")
    if sect is not None:
        sect.getparent().remove(sect)
    for el in list(clone.iter(*_MARKERS - {w("pPr"), w("proofErr")})):
        el.getparent().remove(el)
    ppr = clone.find(w("pPr"))
    if ppr is None:
        ppr = etree.Element(w("pPr"))
        clone.insert(0, ppr)
    old = ppr.find(w("numPr"))
    if old is not None:
        ppr.remove(old)
    numpr = etree.Element(w("numPr"))
    etree.SubElement(numpr, w("numId")).set(w("val"), "0")
    before = [i for i, child in enumerate(ppr) if child.tag in _BEFORE_NUMPR]
    ppr.insert(before[-1] + 1 if before else 0, numpr)
    for attr in list(clone.attrib):
        if attr.startswith("{http://schemas.microsoft.com/office/word/2010/wordml}"):
            del clone.attrib[attr]
    para.p.addnext(clone)
    return _Paragraph(clone, _items(clone), para.text, para.html, para.kind)


async def translate_docx(src: Path, out_dir: Path, runner, bilingual: bool) -> list[Path]:
    try:
        zin = zipfile.ZipFile(src)
    except zipfile.BadZipFile:
        raise ValueError(f"不是有效的 Word 文档: {src.name}") from None
    with zin:
        main, parts, styles_name = _parts(zin)
        styles = _style_kinds(zin, styles_name)
        trees: dict[str, etree._Element] = {}
        paras: list[_Paragraph] = []
        for part in parts:
            root = etree.fromstring(zin.read(part), _PARSER)
            trees[part] = root
            for p in root.iter(w("p")):
                items = _items(p)
                if not any(isinstance(i, _Segment) for i in items):
                    continue
                text, rich = _source(items)
                if not translatable(_PLACEHOLDER_RE.sub("", text)):
                    continue
                paras.append(_Paragraph(p, items, text, rich, _kind(p, styles)))
        _bibliography_skips(paras)
        first_heading = next((p.text for p in paras if p.kind == "h1"), "")
        runner.set_title(_title(zin) or first_heading or src.stem)
        results = await runner.translate_all(
            [p.html or p.text for p in paras],
            kinds=[p.kind for p in paras],
            preview=True,
            html=[p.html is not None for p in paras],
            skip=[p.skip for p in paras],
        )
        for para, translated in zip(paras, results):
            if not translated.strip() or translated == (para.html or para.text):
                continue
            target = _bilingual_copy(para) if bilingual else para
            _write(target, translated, drop_unique=bilingual)

        suffix = "bilingual" if bilingual else "translated"
        dst = out_dir / f"{src.stem}.{suffix}.docx"
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename in trees:
                    data = etree.tostring(trees[info.filename], xml_declaration=True, encoding="UTF-8", standalone=True)
                else:
                    data = zin.read(info.filename)
                zout.writestr(info, data)
    return [dst]
