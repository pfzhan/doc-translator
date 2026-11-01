"""(X)HTML paragraph translation, shared by EPUB and MOBI.

The approach matches the web page translation of Immersive Translate: find the leaf block elements (block-level
elements with no other block-level element inside) and translate the text of each as one paragraph. In bilingual
mode a translated paragraph of the same type is inserted after the original; in translated-only mode the
paragraph text is replaced.
"""

import copy
import re
from html.entities import html5

from bs4 import BeautifulSoup, NavigableString, Tag
from lxml import etree

from ..languages import RTL_LANGUAGES

BLOCK_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "div",
    "td",
    "th",
    "dt",
    "dd",
    "figcaption",
    "caption",
    "section",
    "article",
    "aside",
    "header",
    "footer",
    "pre",
    "table",
    "tr",
    "ul",
    "ol",
    "dl",
    "figure",
    "nav",
    "body",
    "center",
    "address",
    "summary",
    "details",
    "main",
    "hr",
    "tbody",
    "thead",
}
# Content inside these elements is not translated
SKIP_TAGS = {"pre", "code", "script", "style", "svg", "math", "head", "title", "kbd", "samp", "var"}
# Inline code, keys and variables: the text stays in the paragraph as is and is sent to the translation service
# with it
VERBATIM_TAGS = {"code", "kbd", "samp", "var"}
# The translation is appended inside these elements: inserting a sibling would break list numbering / table
# structure, and a copied heading would show up twice in the table of contents
INNER_TAGS = {"li", "td", "th", "dt", "dd", "caption", "span", "h1", "h2", "h3", "h4", "h5", "h6"}

TRANSLATION_CLASS = "dt-translation"
BILINGUAL_CSS = f".{TRANSLATION_CLASS} {{ opacity: 0.85; }}"
NO_TEXT_RE = re.compile(r"^[\W\d_]*$")
WS_RE = re.compile(r"\s+")

# Inline formatting elements to keep through translation; a paragraph with only <span> and <br> is not rich
# text
RICH_TAGS = {
    "a",
    "b",
    "strong",
    "em",
    "i",
    "u",
    "s",
    "strike",
    "del",
    "ins",
    "sup",
    "sub",
    "code",
    "kbd",
    "samp",
    "var",
    "mark",
    "small",
    "abbr",
    "q",
    "cite",
    "font",
}
# Tags removed together with their text when they appear in a translated fragment
DANGER_TAGS = {"script", "style", "iframe", "object", "embed", "form", "input", "button", "select", "textarea"}


def _local(name: str | None) -> str:
    return (name or "").split(":")[-1].lower()


def _is_block(el) -> bool:
    return isinstance(el, Tag) and _local(el.name) in BLOCK_TAGS


def _in_skip(el: Tag) -> bool:
    return any(isinstance(p, Tag) and _local(p.name) in SKIP_TAGS for p in [el, *el.parents])


def _text_of(el: Tag, slots: bool = True) -> str:
    """Get the plain text of a paragraph: <br> becomes a newline, newlines in the text count as spaces,
    script/style and the like are ignored, and inline code is kept as is and translated with the rest.

    Formulas and images in the paragraph become {v1}, {v2}, and are put back by number after translation (see
    _uses_slots).
    """
    # Matched by id: bs4 Tag equality compares content, so two identical images would count as one
    slot_of = {id(m): i for i, m in enumerate(_media_of(el), 1)} if slots and _uses_slots(el) else {}
    parts = []
    for node in el.descendants:
        if id(node) in slot_of:
            parts.append(f"{{v{slot_of[id(node)]}}}")
        elif isinstance(node, Tag) and _local(node.name) == "br":
            parts.append("\n")
        elif type(node) is NavigableString:  # Excludes subclasses such as comments and CDATA
            parents = [p for p in node.parents if isinstance(p, Tag)]
            if any(id(p) in slot_of for p in parents):
                continue
            if not any(_local(p.name) in SKIP_TAGS - VERBATIM_TAGS for p in parents):
                # In paragraphs wrapped across lines in the source, a newline is whitespace like a space; only
                # <br> is a line break
                parts.append(WS_RE.sub(" ", str(node)))
    return _join_lines("".join(parts))


def _join_lines(text: str) -> str:
    """Collapse whitespace on each line. An empty line left by two consecutive <br> separates the stanzas of a
    poem; such lines are kept in the middle and removed at the start and end."""
    return "\n".join(WS_RE.sub(" ", line).strip() for line in text.split("\n")).strip("\n")


MEDIA_TAGS = {"img", "image", "svg", "math", "video", "audio", "object"}
# Placeholder for inline formulas and images in the translation, written the same way as the PDF formula
# placeholder; the prompt explains it
SLOT_RE = re.compile(r"\{v(\d+)\}")


def _media_of(el: Tag) -> list[Tag]:
    """Media elements in el, in document order; those nested in other media (an image inside svg) are not
    counted separately."""
    media: list[Tag] = []
    seen: set[int] = set()
    for m in el.find_all(lambda t: _local(t.name) in MEDIA_TAGS):
        if not any(id(p) in seen for p in m.parents):
            media.append(m)
            seen.add(id(m))
    return media


def _uses_slots(el: Tag) -> bool:
    """Use placeholders only when there are media elements and the source has no {vN} of its own; otherwise the
    placeholders could not be told apart from the source text."""
    return bool(_media_of(el)) and not SLOT_RE.search(el.get_text())


def _without_slots(text: str) -> str:
    return SLOT_RE.sub("", text)


def translatable(text: str) -> bool:
    return bool(text) and not NO_TEXT_RE.match(text)


def _attr_local(el: Tag, local: str) -> list[str]:
    out: list[str] = []
    for key, val in el.attrs.items():
        if _local(str(key)) != local:
            continue
        if isinstance(val, list):
            out.extend(str(v) for v in val)
        else:
            out.append(str(val))
    return out


def _is_page_anchor(el: Tag) -> bool:
    """Page anchor: epub:type=pagebreak, or an id anchor with no visible text. Writing back plain text would
    remove them."""
    if any("pagebreak" in v.lower() for v in _attr_local(el, "type")):
        return True
    return bool(el.has_attr("id") and not el.get_text(strip=True) and _local(el.name) in ("a", "span"))


def _has_kept_structure(el: Tag) -> bool:
    return el.find(lambda t: isinstance(t, Tag) and (_local(t.name) in RICH_TAGS or _is_page_anchor(t))) is not None


def _inner_html(el: Tag) -> str | None:
    """Return the inner HTML when the block has inline formatting or page anchors to keep; otherwise None
    (plain text path).
    Media elements become {vN} as in the plain text path, so MathML is not sent to the model."""
    if not _has_kept_structure(el):
        return None
    if not _uses_slots(el):
        return "".join(str(c) for c in el.children)
    copied = copy.copy(el)
    for i, m in enumerate(_media_of(copied), 1):
        m.replace_with(NavigableString(f"{{v{i}}}"))
    return "".join(str(c) for c in copied.children)


def find_blocks(soup: BeautifulSoup) -> list[tuple[Tag, str, str | None]]:
    """Return [(element, plain text, rich text HTML or None)]; no element is nested in another."""
    body = soup.find(lambda t: _local(t.name) == "body") or soup
    result = []
    for el in body.find_all(True):
        if not _is_block(el) or _in_skip(el):
            continue
        if any(_is_block(d) for d in el.find_all(True)):
            continue
        text = _text_of(el)
        if translatable(_without_slots(text)):
            result.append((el, text, _inner_html(el)))
    # Bare text directly inside a block container (common in HTML converted from MOBI) is wrapped in a span and
    # translated. Other containers without block children are already leaf blocks found above; <body> is not
    # covered above, so its text is wrapped even when it has no block at all
    for container in [body, *body.find_all(lambda t: _is_block(t))]:
        if _in_skip(container):
            continue
        if not any(_is_block(c) for c in container.children) and not (
            container is body and _local(body.name) == "body"
        ):
            continue
        for run in _inline_runs(container, soup):
            if run.find(_is_block):
                continue  # An inline element containing block elements; the block part was handled above
            text = _text_of(run)
            if translatable(_without_slots(text)):
                result.append((run, text, _inner_html(run)))
    # Sort in document order (bare text is found in the second pass), so that preview and translation follow
    # the reading order
    order = {id(el): i for i, el in enumerate(body.find_all(True))}
    result.sort(key=lambda item: order.get(id(item[0]), len(order)))
    return result


HEADING_KINDS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Citations translated into the target language no longer match the original book. The whole heading must
# match; references appearing inside body text does not count.
_BIBLIO_TITLE = (
    r"(?:selected bibliography|further reading|works cited|bibliography|references|endnotes"
    r"|参考文献|参考书目|引用文献)"
)
BIBLIO_HEADING_RE = re.compile(rf"^(?:\d{{1,3}}[.)]\s+)?{_BIBLIO_TITLE}\s*[:：.。]?\s*$", re.I)
# Subheadings that clearly belong inside a bibliography. Books / "论文" are also common chapter titles and
# are not listed here.
BIBLIO_SUBHEADING_RE = re.compile(
    r"^(?:\d{1,3}[.)]\s+)?"
    r"(?:primary sources?|secondary sources?|archival sources?|online sources?"
    r"|web sources?|internet sources?|printed sources?|unpublished sources?"
    r"|book chapters?"
    r"|主要文献|次要文献|原始文献|一手文献|二手文献|网络文献)"
    r"\s*[:：.。]?\s*$",
    re.I,
)
# One-word subheadings. They count only when lower than the bibliography heading and heading-like, so that
# a line with only Journal in a citation is not matched.
BIBLIO_SHORT_LABEL_RE = re.compile(
    r"^(?:\d{1,3}[.)]\s+)?"
    r"(?:books?|articles?|journals?|essays?|papers?|magazines?|newspapers?"
    r"|图书|期刊|论文|报纸|杂志|文章|著作)"
    r"\s*[:：.。]?\s*$",
    re.I,
)
BIBLIO_EPUB_TYPES = frozenset({"bibliography", "rearnotes", "endnotes"})
_BIBLIO_ROLES = frozenset({"doc-bibliography", "doc-endnotes"})
_HEADING_CLASS_RE = re.compile(r"\b(?:heading|title|chapter|h[1-6])\b", re.I)


def heading_level(el: Tag) -> int:
    """Return the level (1-6) for a heading element, otherwise 0."""
    name = _local(el.name)
    return int(name[1]) if name in HEADING_KINDS else 0


def structural_heading_level(el: Tag) -> int:
    """Level of h1-h6. Short blocks marked as headings by class or role also count, otherwise
    <p class="heading"> would not be recognized."""
    level = heading_level(el)
    if level:
        return level
    text = WS_RE.sub(" ", el.get_text()).strip()
    if not text or len(text) > 80:
        return 0
    classes = " ".join(_attr_local(el, "class"))
    match = re.search(r"\bh([1-6])\b", classes, re.I)
    if match:
        return int(match.group(1))
    roles = {v.lower() for v in _attr_local(el, "role")}
    if "heading" in roles or _HEADING_CLASS_RE.search(classes):
        return 2
    return 0


def _heading_text(el: Tag) -> str:
    return WS_RE.sub(" ", el.get_text()).strip()


def biblio_heading_level(el: Tag) -> int:
    """Return the level (1-6) for a bibliography section heading, otherwise 0."""
    level = structural_heading_level(el)
    if not level:
        return 0
    text = _heading_text(el)
    if len(text) > 40 or not BIBLIO_HEADING_RE.match(text):
        return 0
    return level


def is_biblio_subheading(text: str) -> bool:
    """A subheading inside a bibliography, or another bibliography heading."""
    text = WS_RE.sub(" ", text).strip()
    if len(text) > 40:
        return False
    return bool(BIBLIO_SUBHEADING_RE.match(text) or BIBLIO_HEADING_RE.match(text))


def _type_values(el: Tag) -> set[str]:
    out: set[str] = set()
    for raw in _attr_local(el, "type"):
        out.update(part.strip().lower() for part in raw.split() if part.strip())
    return out


def _biblio_container(el: Tag) -> Tag | None:
    """The ancestor section marked bibliography / rearnotes / endnotes. A heading is not a container itself."""
    for parent in el.parents:
        if not isinstance(parent, Tag):
            continue
        if _type_values(parent) & BIBLIO_EPUB_TYPES:
            return parent
        roles = {v.strip().lower() for v in _attr_local(parent, "role")}
        if roles & _BIBLIO_ROLES:
            return parent
    return None


def _owning_section(heading: Tag) -> Tag | None:
    """The section/article whose first heading is this heading; None when the heading sits directly under
    body."""
    for parent in heading.parents:
        if not isinstance(parent, Tag):
            continue
        name = _local(parent.name)
        if name in {"section", "article"}:
            first = parent.find(lambda t: isinstance(t, Tag) and structural_heading_level(t) > 0)
            return parent if first is heading else None
        if name in {"body", "html"}:
            return None
    return None


def _biblio_should_exit(el: Tag, heading_lv: int, level: int, section: Tag | None) -> bool:
    # In a flat document, an h2 appendix after an h1 bibliography is not a subheading; only deeper headings in
    # the same section stay inside.
    if section is not None and section not in el.parents:
        return True
    if heading_lv <= level:
        return True
    if is_biblio_subheading(_heading_text(el)):
        return False
    return section is None


def bibliography_skips(blocks: list[Tag], *, start_level: int = 0) -> tuple[list[bool], int]:
    """Mark the blocks that keep the original text, and return the level carried to the next spine file (0
    means the bibliography has ended).

    Headings themselves are not skipped. Body text inside an epub:type container is skipped entirely; after
    leaving the container, heading levels decide as before.
    """
    level = start_level if 0 <= start_level <= 6 else 0
    section: Tag | None = None
    skips: list[bool] = []
    ended_in_container = False
    for el in blocks:
        container = _biblio_container(el)
        heading_lv = structural_heading_level(el)
        if container is not None:
            ended_in_container = True
            skips.append(not heading_lv)
            continue
        ended_in_container = False
        biblio_lv = biblio_heading_level(el)
        if biblio_lv:
            if level == 0 or biblio_lv <= level:
                level = biblio_lv
                section = _owning_section(el)
            skips.append(False)
            continue
        if level and heading_lv and _biblio_should_exit(el, heading_lv, level, section):
            level = 0
            section = None
            skips.append(False)
            continue
        skips.append(bool(level))
    if ended_in_container and level == 0:
        level = 1
    return skips, level


def preview_kind(el: Tag) -> str:
    """Paragraph type for the preview: headings keep their level, list items, quotes and table cells each
    have their own type, everything else is a paragraph."""
    name = _local(el.name)
    if name in HEADING_KINDS:
        return name
    if name in ("li", "dt", "dd"):
        return "li"
    if name in ("td", "th", "caption"):
        return "td"
    if name == "blockquote" or any(_local(p.name) == "blockquote" for p in el.parents if isinstance(p, Tag)):
        return "quote"
    return "p"


def _inline_runs(container: Tag, soup) -> list[Tag]:
    """Merge adjacent inline nodes in a container into a <span> so they can be handled as a paragraph."""
    runs, cur = [], []

    def flush():
        if cur and "".join(n.get_text() if isinstance(n, Tag) else str(n) for n in cur).strip():
            wrapper = soup.new_tag("span")
            cur[0].insert_before(wrapper)
            for n in cur:
                wrapper.append(n.extract())
            runs.append(wrapper)
        cur.clear()

    for child in list(container.children):
        if _is_block(child) or (isinstance(child, Tag) and _local(child.name) in SKIP_TAGS):
            flush()
        elif isinstance(child, Tag) or type(child) is NavigableString:
            cur.append(child)
    flush()
    return runs


def _set_text(soup, el: Tag, text: str, media: list[Tag] | None = None):
    """Clear el and write text into it. media holds the media elements of the original paragraph: each {vN}
    in text is replaced by the matching one, and those the translation left out are appended at the end. When
    media is None (the source already contains {vN}), placeholders are not interpreted."""
    el.clear()
    for i, line in enumerate(text.split("\n")):
        if i:
            el.append(soup.new_tag("br"))
        el.append(NavigableString(line))
    if media is not None:
        _place_media(el, media)


def _slot_media(el: Tag, copies: bool) -> list[Tag] | None:
    """Media elements of el, to be placed in the translation. In bilingual mode the original paragraph still
    needs them, so copies are used, with ids removed to avoid duplicates."""
    if not _uses_slots(el):
        return None
    if not copies:
        return [m.extract() for m in _media_of(el)]
    out = []
    for m in _media_of(el):
        clone = copy.copy(m)
        for t in [clone, *clone.find_all(True)]:
            if t.has_attr("id"):
                del t["id"]
        out.append(clone)
    return out


def _place_media(target: Tag, media: list[Tag]) -> None:
    """Replace {vN} in the text of target with the Nth element of media. Duplicate and out-of-range
    placeholders are removed, and unused elements are appended at the end in their original order: a formula
    must not be lost when the model drops its placeholder."""
    used: set[int] = set()
    for node in list(target.descendants):
        if type(node) is not NavigableString or not SLOT_RE.search(node):
            continue
        pieces: list = []
        pos = 0
        for m in SLOT_RE.finditer(node):
            pieces.append(NavigableString(node[pos : m.start()]))
            index = int(m.group(1)) - 1
            if 0 <= index < len(media) and index not in used:
                used.add(index)
                pieces.append(media[index])
            pos = m.end()
        pieces.append(NavigableString(node[pos:]))
        for piece in pieces:
            node.insert_before(piece)
        node.extract()
    for index, m in enumerate(media):
        if index not in used:
            target.append(m)


def lang_attrs(lang: str) -> dict:
    """Language attributes of a translated element: readers use them to choose fonts and line breaking;
    Arabic and similar languages need right-to-left layout."""
    if not lang:
        return {}
    attrs = {"lang": lang, "xml:lang": lang}
    if lang.split("-")[0] in RTL_LANGUAGES:
        attrs["dir"] = "rtl"
    return attrs


def apply_translation(soup, el: Tag, text: str, translated: str, bilingual: bool, lang: str = "") -> Tag | None:
    """Write back a plain text translation and return the element that holds it (el itself / an inner span /
    a cloned block); return None when skipped."""
    if not translated or translated.strip() == text.strip():
        return None
    name = _local(el.name)
    attrs = lang_attrs(lang)
    if not bilingual:
        media = _slot_media(el, copies=False)
        if media is None:
            # The source contains {vN} itself: media elements are placed at the start of the paragraph as before
            media_kept = [m.extract() for m in _media_of(el)]
            _set_text(soup, el, translated)
            for m in reversed(media_kept):
                el.insert(0, m)
        else:
            _set_text(soup, el, translated, media)
        for k, v in attrs.items():
            el[k] = v
        return el
    if name in INNER_TAGS:
        span = soup.new_tag("span", attrs={"class": TRANSLATION_CLASS, **attrs})
        _set_text(soup, span, translated, _slot_media(el, copies=True))
        el.append(soup.new_tag("br"))
        el.append(span)
        return span
    clone = copy.copy(el)
    for attr in ("id", "name"):
        if clone.has_attr(attr):
            del clone[attr]
    classes = clone.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    clone["class"] = [*classes, TRANSLATION_CLASS]
    for k, v in attrs.items():
        clone[k] = v
    _set_text(soup, clone, translated, _slot_media(el, copies=True))
    el.insert_after(clone)
    return clone


def _parse_fragment(html: str) -> Tag | None:
    try:
        return BeautifulSoup(f"<div>{html}</div>", "lxml").find("div")
    except Exception:  # noqa: BLE001 - the translation is model output; any parse error falls back
        return None


_URL_ATTRS = frozenset({"href", "src", "xlink:href"})


# svg+xml can embed scripts; only raster data URIs, which browsers draw as images, are allowed
_RASTER_DATA_RE = re.compile(r"^data:image/(?:png|jpe?g|gif|webp)[;,]")


def _unsafe_url(value: object, tag: str) -> bool:
    """javascript:/vbscript: are always unsafe; data: is allowed only for raster images on img/image."""
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    if v.startswith(("javascript:", "vbscript:")):
        return True
    if not v.startswith("data:"):
        return False
    return not (tag in ("img", "image") and _RASTER_DATA_RE.match(v) is not None)


def _clean_attrs(src: Tag, frag: Tag) -> None:
    """Keep only attributes that appear on tags of the same name in the source, and drop event handlers and
    script URLs."""
    allowed: dict[str, set[str]] = {}
    styles: set[str] = set()
    for t in src.find_all(True):
        allowed.setdefault(_local(t.name), set()).update(str(k) for k in t.attrs)
        if "style" in t.attrs:
            styles.add(str(t.attrs["style"]))
    for t in frag.find_all(True):
        keep = allowed.get(_local(t.name), set())
        for attr in list(t.attrs):
            key = str(attr)
            val = t.attrs[attr]
            if (
                key.lower().startswith("on")
                or key not in keep
                or _unsafe_url(val, _local(t.name))
                or key == "style"
                and str(val) not in styles
            ):
                del t[attr]


def _restore_page_anchors(src: Tag, dest: Tag) -> None:
    """Restore page anchors the translation dropped, otherwise page number links in the book break."""
    have = {t.get("id") for t in dest.find_all(True) if t.has_attr("id")}
    missing = []
    for anchor in src.find_all(True):
        if not _is_page_anchor(anchor):
            continue
        aid = anchor.get("id")
        if aid and aid not in have:
            missing.append(copy.copy(anchor))
            have.add(aid)
    for anchor in reversed(missing):
        dest.insert(0, anchor)


def _prose_of(el: Tag) -> str:
    """Plain text of a paragraph. Page numbers in page anchors are not body text and must not be used to judge
    whether the translation is still there."""
    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, Tag) and _local(node.name) == "br":
            if not any(_is_page_anchor(p) for p in node.parents if isinstance(p, Tag)):
                parts.append("\n")
        elif type(node) is NavigableString:
            parents = [p for p in node.parents if isinstance(p, Tag)]
            if any(_is_page_anchor(p) for p in parents):
                continue
            if any(_local(p.name) in SKIP_TAGS - VERBATIM_TAGS for p in parents):
                continue
            parts.append(WS_RE.sub(" ", str(node)))
    return _join_lines("".join(parts))


def _sanitize_translation(src_html: str, translated_html: str) -> list | None:
    """Validate and clean a translated HTML fragment and return the list of nodes to insert; return None when
    it is rejected (the caller falls back to plain text).

    - Remove dangerous tags such as script/style.
    - Unwrap tags the source does not have (keeping only their text), so the model cannot invent structure.
    - Drop event handlers, script URLs and attributes the source does not have.
    - Reject the fragment when the translation is empty, fails to parse, keeps none of the source's inline
      formatting, or has no text after removing page anchors while the source has body text.
    """
    if not translated_html or not translated_html.strip():
        return None
    src = _parse_fragment(src_html)
    frag = _parse_fragment(translated_html)
    if src is None or frag is None:
        return None
    allowed = {_local(t.name) for t in src.find_all(True)} | {"br"}
    for t in frag.find_all(lambda t: _local(t.name) in DANGER_TAGS):
        t.decompose()
    for t in list(frag.find_all(True)):
        if _local(t.name) not in allowed:
            t.unwrap()
    _clean_attrs(src, frag)
    _restore_page_anchors(src, frag)
    # Page numbers are inside page anchors and do not count as translation after being restored. If the source
    # has text without its anchors and the translation has none, the model emptied the paragraph.
    if _prose_of(src) and not _prose_of(frag):
        return None
    if not frag.get_text(strip=True) and not any(_is_page_anchor(t) for t in frag.find_all(True)):
        return None
    src_rich = src.find(lambda t: _local(t.name) in RICH_TAGS)
    if src_rich and not frag.find(lambda t: _local(t.name) in RICH_TAGS):
        return None
    return list(frag.children)


def _strip_ids(target: Tag) -> None:
    """Remove ids and page anchors from a bilingual translation: the original paragraph still has them, so a
    second copy would duplicate the ids and show the page numbers twice."""
    anchors = [t for t in target.find_all(True) if _is_page_anchor(t)]
    for t in anchors:
        if not any(p in anchors for p in t.parents):
            t.decompose()
    for t in target.find_all(True):
        for attr in ("id", "name"):
            if t.has_attr(attr):
                del t[attr]


def apply_translation_html(soup, el: Tag, src_html: str, translated_html: str, bilingual: bool, lang: str = ""):
    """Write back a rich text paragraph: the translation keeps inline tags (bold, links, etc.). Falls back to
    the plain text path when the translation is rejected."""
    if translated_html.strip() == src_html.strip():
        return
    nodes = _sanitize_translation(src_html, translated_html)
    if nodes is None:
        frag = _parse_fragment(translated_html)
        # <br> becomes a newline; page numbers in page anchors are not translation and must not overwrite the
        # original
        plain = _prose_of(frag) if frag is not None else translated_html
        target = apply_translation(soup, el, _text_of(el), plain, bilingual, lang)
        src = _parse_fragment(src_html)
        # In bilingual mode the page anchors are still in the original paragraph and need no restoring
        if target is not None and src is not None and not bilingual:
            _restore_page_anchors(src, target)
        return
    name = _local(el.name)
    attrs = lang_attrs(lang)
    # When the source contains {vN} itself, media elements were sent to the model with the HTML; if the
    # translation lost all of them, the source's media are appended at the end
    src_frag = _parse_fragment(src_html)
    src_media = src_frag.find_all(lambda t: _local(t.name) in MEDIA_TAGS) if src_frag else []
    has_media = any(
        _local(getattr(n, "name", None)) in MEDIA_TAGS
        or (isinstance(n, Tag) and n.find(lambda t: _local(t.name) in MEDIA_TAGS))
        for n in nodes
    )

    def fill(target: Tag, media: list[Tag] | None):
        target.clear()
        for n in nodes:
            target.append(n)
        if media is not None:
            _place_media(target, media)
        elif src_media and not has_media:
            for m in src_media:
                target.append(m)

    if not bilingual:
        fill(el, _slot_media(el, copies=False))
        for k, v in attrs.items():
            el[k] = v
        return
    if name in INNER_TAGS:
        span = soup.new_tag("span", attrs={"class": TRANSLATION_CLASS, **attrs})
        fill(span, _slot_media(el, copies=True))
        _strip_ids(span)
        el.append(soup.new_tag("br"))
        el.append(span)
        return
    clone = copy.copy(el)
    for attr in ("id", "name"):
        if clone.has_attr(attr):
            del clone[attr]
    classes = clone.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    clone["class"] = [*classes, TRANSLATION_CLASS]
    fill(clone, _slot_media(el, copies=True))
    _strip_ids(clone)
    for k, v in attrs.items():
        clone[k] = v
    el.insert_after(clone)


# The five entities built into XML; other named entities need a DTD, and lxml-xml does not load external DTDs
_XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}
_NAMED_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]*);")
_NAMED_ENTITY_BYTES_RE = re.compile(rb"&([A-Za-z][A-Za-z0-9]*);")


def _numeric_entities(content: bytes | str) -> bytes | str:
    """Convert HTML named entities such as &rsquo; and &nbsp; to numeric references. Unknown entities are kept
    as is."""

    def repl(m: re.Match) -> bytes | str:
        name = m.group(1)
        if isinstance(name, bytes):
            name = name.decode("ascii")
        chars = None if name in _XML_ENTITIES else html5.get(name + ";")
        if chars is None:
            return m.group(0)
        out = "".join(f"&#{ord(c)};" for c in chars)
        return out.encode("ascii") if isinstance(m.group(0), bytes) else out

    if isinstance(content, bytes):
        return _NAMED_ENTITY_BYTES_RE.sub(repl, content)
    return _NAMED_ENTITY_RE.sub(repl, content)


# Void elements written HTML-style: unclosed <br> and <img ...> are invalid in XML
_VOID_TAGS = "area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr"
_OPEN_VOID = rf"""<({_VOID_TAGS})\b((?:[^<>"']|"[^"]*"|'[^']*')*?)\s*(?<!/)>"""
_VOID_END = rf"</(?:{_VOID_TAGS})\s*>"
# An & not followed by an entity reference (AT&T)
_BARE_AMP = r"&(?!#\d+;|#[xX][0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]*;)"


def _sub(pattern: str, repl: str, content: bytes | str) -> bytes | str:
    if isinstance(content, bytes):
        return re.sub(pattern.encode(), repl.encode(), content, flags=re.I)
    return re.sub(pattern, repl, content, flags=re.I)


def _well_formed(content: bytes | str) -> bool:
    data = content.encode("utf-8") if isinstance(content, str) else content
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=True)
    try:
        etree.fromstring(data, parser)
    except etree.XMLSyntaxError:
        return False
    return True


def _close_html_tags(content: bytes | str) -> bytes | str:
    """Turn HTML-style markup into XML: void elements become self-closing, their end tags are removed, and bare
    & is escaped."""
    content = _sub(_OPEN_VOID, r"<\1\2/>", content)
    content = _sub(_VOID_END, "", content)
    return _sub(_BARE_AMP, "&amp;", content)


def parse(content: bytes | str, xml: bool) -> BeautifulSoup:
    if xml:
        # Without the conversion these entities are silently dropped, and characters go missing from both the
        # source and the translation
        content = _numeric_entities(content)
        # On malformed input, the lenient parsing of lxml-xml drops the characters &amp; and &lt; from the error
        # onward, and blocks end up nested in each other, while the whole document is written back from the parse
        # result. First fix the HTML-style markup; if that is not enough, parse as HTML, which loses the case of
        # SVG attributes
        if not _well_formed(content):
            repaired = _close_html_tags(content)
            if not _well_formed(repaired):
                return BeautifulSoup(content, "lxml")
            content = repaired
    soup = BeautifulSoup(content, "lxml-xml" if xml else "lxml")
    # lxml-xml silently produces an empty document for XHTML with an internal DTD subset
    # (<!DOCTYPE html [<!ENTITY nbsp "&#160;">]>). When the source is not empty but no element was parsed,
    # parse again in HTML mode
    if xml and soup.find(True) is None and content.strip():
        soup = BeautifulSoup(content, "lxml")
    return soup


def add_style(soup, css: str = BILINGUAL_CSS):
    head = soup.find(lambda t: _local(t.name) == "head")
    if head is None:
        return
    style = soup.new_tag("style", attrs={"type": "text/css"})
    style.string = css
    head.append(style)


def translate_nav_links(soup) -> list[tuple[Tag, str]]:
    """EPUB3 navigation document: translate only the text of <a>/<span>, without changing the structure."""
    out = []
    for el in soup.find_all(lambda t: _local(t.name) in ("a", "span")):
        if el.find(lambda t: _local(t.name) in ("a", "span")):
            continue
        text = _text_of(el, slots=False)
        if translatable(text):
            out.append((el, text))
    return out


def set_label(el: Tag, text: str, translated: str, bilingual: bool):
    if not translated or translated == text:
        return
    el.string = f"{text} / {translated}" if bilingual else translated
