"""(X)HTML 段落翻译，EPUB / MOBI 共用。

思路和沉浸式翻译的网页翻译一致：找出“叶子块级元素”（自身是块级、内部没有其他块级元素），
把它的文本作为一个段落翻译；双语模式下在原段落后面插入一个同类型的译文段落，
仅译文模式下直接替换段落文本。
"""
import copy
import re

from bs4 import BeautifulSoup, NavigableString, Tag

from ..languages import RTL_LANGUAGES

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div", "td", "th",
    "dt", "dd", "figcaption", "caption", "section", "article", "aside", "header",
    "footer", "pre", "table", "tr", "ul", "ol", "dl", "figure", "nav", "body",
    "center", "address", "summary", "details", "main", "hr", "tbody", "thead",
}
# 这些元素内部的内容不翻译
SKIP_TAGS = {"pre", "code", "script", "style", "svg", "math", "head", "title", "kbd", "samp", "var"}
# 在元素内部追加译文：插入兄弟元素会破坏列表序号 / 表格结构，标题复制一份会让目录出现重复项
INNER_TAGS = {"li", "td", "th", "dt", "dd", "caption", "span", "h1", "h2", "h3", "h4", "h5", "h6"}

TRANSLATION_CLASS = "dt-translation"
BILINGUAL_CSS = f".{TRANSLATION_CLASS} {{ opacity: 0.85; }}"
NO_TEXT_RE = re.compile(r"^[\W\d_]*$")
WS_RE = re.compile(r"\s+")

# 需要在翻译中保留的行内格式元素；只有 <span>、<br> 的段落不算富文本
RICH_TAGS = {
    "a", "b", "strong", "em", "i", "u", "s", "strike", "del", "ins",
    "sup", "sub", "code", "mark", "small", "abbr", "q", "cite", "font",
}
# 译文片段里出现直接删除（不保留文字）的标签
DANGER_TAGS = {"script", "style", "iframe", "object", "embed", "form", "input", "button", "select", "textarea"}


def _local(name: str | None) -> str:
    return (name or "").split(":")[-1].lower()


def _is_block(el) -> bool:
    return isinstance(el, Tag) and _local(el.name) in BLOCK_TAGS


def _in_skip(el: Tag) -> bool:
    for p in [el, *el.parents]:
        if isinstance(p, Tag) and _local(p.name) in SKIP_TAGS:
            return True
    return False


def _text_of(el: Tag) -> str:
    """取段落纯文本；<br> 变成换行，script/style 等忽略，行内 code 保留原样参与翻译。"""
    parts = []
    for node in el.descendants:
        if isinstance(node, Tag) and _local(node.name) == "br":
            parts.append("\n")
        elif type(node) is NavigableString:  # 排除注释、CDATA 等子类
            if not any(_local(p.name) in SKIP_TAGS - {"code"} for p in node.parents if isinstance(p, Tag)):
                parts.append(str(node))
    text = "".join(parts)
    lines = [WS_RE.sub(" ", l).strip() for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


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
    """分页锚点：epub:type=pagebreak，或没有可见文字的 id 锚点。纯文本回写会把它们清掉。"""
    if any("pagebreak" in v.lower() for v in _attr_local(el, "type")):
        return True
    return bool(el.has_attr("id") and not el.get_text(strip=True) and _local(el.name) in ("a", "span"))


def _has_kept_structure(el: Tag) -> bool:
    return el.find(lambda t: isinstance(t, Tag) and (_local(t.name) in RICH_TAGS or _is_page_anchor(t))) is not None


def _inner_html(el: Tag) -> str | None:
    """块内有需要保留的行内格式或分页锚点时返回 inner HTML；否则 None（纯文本路径）。"""
    if not _has_kept_structure(el):
        return None
    return "".join(str(c) for c in el.children)


def find_blocks(soup: BeautifulSoup) -> list[tuple[Tag, str, str | None]]:
    """返回 [(元素, 纯文本, 富文本 HTML 或 None)]，元素之间互不嵌套。"""
    body = soup.find(lambda t: _local(t.name) == "body") or soup
    result = []
    for el in body.find_all(True):
        if not _is_block(el) or _in_skip(el):
            continue
        if any(_is_block(d) for d in el.find_all(True)):
            continue
        text = _text_of(el)
        if translatable(text):
            result.append((el, text, _inner_html(el)))
    # 块级容器里直接挂着的“裸文本”（常见于 MOBI 转出来的 HTML），包成 span 再翻译
    for container in [body, *body.find_all(lambda t: _is_block(t))]:
        if _in_skip(container) or not any(_is_block(c) for c in container.children):
            continue
        for run in _inline_runs(container, soup):
            if run.find(_is_block):
                continue  # 行内元素里包着块级元素，块级部分已在上面处理
            text = _text_of(run)
            if translatable(text):
                result.append((run, text, _inner_html(run)))
    # 按文档顺序排列（裸文本是第二遍才找到的），这样预览和翻译顺序都和阅读顺序一致
    order = {id(el): i for i, el in enumerate(body.find_all(True))}
    result.sort(key=lambda item: order.get(id(item[0]), len(order)))
    return result


HEADING_KINDS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# 引文翻成目标语言后对不上原书。必须整段命中，正文里出现 references 不算。
_BIBLIO_TITLE = (
    r"(?:selected bibliography|further reading|works cited|bibliography|references|endnotes"
    r"|参考文献|参考书目|引用文献)"
)
BIBLIO_HEADING_RE = re.compile(
    rf"^(?:\d{{1,3}}[.)]\s+)?{_BIBLIO_TITLE}\s*[:：.。]?\s*$", re.I)
# 明确是参考文献内部的小标题。Books / 论文 也是常见章名，不放这里。
BIBLIO_SUBHEADING_RE = re.compile(
    r"^(?:\d{1,3}[.)]\s+)?"
    r"(?:primary sources?|secondary sources?|archival sources?|online sources?"
    r"|web sources?|internet sources?|printed sources?|unpublished sources?"
    r"|book chapters?"
    r"|主要文献|次要文献|原始文献|一手文献|二手文献|网络文献)"
    r"\s*[:：.。]?\s*$",
    re.I,
)
# 单词小标题。只有比参考文献标题更小、又像标题时才算，避免引文里单独一行 Journal。
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
    """标题元素返回级别（1-6），否则 0。"""
    name = _local(el.name)
    return int(name[1]) if name in HEADING_KINDS else 0


def structural_heading_level(el: Tag) -> int:
    """h1–h6 的级别。class 或 role 标成标题的短块也算，否则 <p class="heading"> 认不出。"""
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
    """是参考文献章节标题时返回级别（1-6），否则 0。"""
    level = structural_heading_level(el)
    if not level:
        return 0
    text = _heading_text(el)
    if len(text) > 40 or not BIBLIO_HEADING_RE.match(text):
        return 0
    return level


def is_biblio_subheading(text: str) -> bool:
    """参考文献内部的小标题，或又一个参考文献标题。"""
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
    """祖先里带 bibliography / rearnotes / endnotes 的 section。标题自己不算容器。"""
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
    """这个标题是其第一个标题的 section/article；平铺在 body 下则没有。"""
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
    # 平铺文档里 h1 参考文献后的 h2 附录不是小标题；同一 section 里更深的标题才留下。
    if section is not None and section not in el.parents:
        return True
    if heading_lv <= level:
        return True
    if is_biblio_subheading(_heading_text(el)):
        return False
    return section is None


def bibliography_skips(blocks: list[Tag], *, start_level: int = 0) -> tuple[list[bool], int]:
    """标记应保留原文的块，并返回带到下一个 spine 文件的级别（0 表示已退出）。

    标题本身不跳过。epub:type 容器内的正文整段跳过，离开容器后按标题级别继续。
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
            skips.append(False if heading_lv else True)
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
    """预览里的段落类型：标题保留级别，列表、引用、表格单元格各自一类，其他都算段落。"""
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
    """把容器里相邻的行内节点合并成 <span>，便于当作段落处理。"""
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


MEDIA_TAGS = {"img", "image", "svg", "math", "video", "audio", "object"}


def _set_text(soup, el: Tag, text: str, keep_media: bool = True):
    # 图片等媒体元素在清空文字后放回去，避免段落内的插图丢失
    media = []
    if keep_media:
        for m in el.find_all(lambda t: _local(t.name) in MEDIA_TAGS):
            if not any(m in p.descendants for p in media):
                media.append(m)
        media = [m.extract() for m in media]
    el.clear()
    for m in media:
        el.append(m)
    for i, line in enumerate(text.split("\n")):
        if i:
            el.append(soup.new_tag("br"))
        el.append(NavigableString(line))


def lang_attrs(lang: str) -> dict:
    """译文元素的语言属性：阅读器据此选字体、断行；阿拉伯语等需要从右到左排版。"""
    if not lang:
        return {}
    attrs = {"lang": lang, "xml:lang": lang}
    if lang.split("-")[0] in RTL_LANGUAGES:
        attrs["dir"] = "rtl"
    return attrs


def apply_translation(soup, el: Tag, text: str, translated: str, bilingual: bool, lang: str = "") -> Tag | None:
    """回写纯文本译文，返回译文落在的元素（el 本身 / 内部 span / 克隆块）；跳过返回 None。"""
    if not translated or translated.strip() == text.strip():
        return None
    name = _local(el.name)
    attrs = lang_attrs(lang)
    if not bilingual:
        _set_text(soup, el, translated)
        for k, v in attrs.items():
            el[k] = v
        return el
    if name in INNER_TAGS:
        span = soup.new_tag("span", attrs={"class": TRANSLATION_CLASS, **attrs})
        _set_text(soup, span, translated)
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
    _set_text(soup, clone, translated, keep_media=False)
    el.insert_after(clone)
    return clone


def _parse_fragment(html: str) -> Tag | None:
    try:
        return BeautifulSoup(f"<div>{html}</div>", "lxml").find("div")
    except Exception:  # noqa: BLE001 - 译文是模型生成的，任何解析异常都按回退处理
        return None


_URL_ATTRS = frozenset({"href", "src", "xlink:href"})


# svg+xml 能内嵌脚本；只放行浏览器当图片画的栅格 data URI
_RASTER_DATA_RE = re.compile(r"^data:image/(?:png|jpe?g|gif|webp)[;,]")


def _unsafe_url(value: object, tag: str) -> bool:
    """javascript:/vbscript: 一律危险；data: 只放行 img/image 上的栅格图。"""
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    if v.startswith(("javascript:", "vbscript:")):
        return True
    if not v.startswith("data:"):
        return False
    return not (tag in ("img", "image") and _RASTER_DATA_RE.match(v) is not None)


def _clean_attrs(src: Tag, frag: Tag) -> None:
    """只留原文同名标签上出现过的属性，并丢掉事件处理和脚本 URL。"""
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
            if key.lower().startswith("on") or key not in keep or _unsafe_url(val, _local(t.name)):
                del t[attr]
            elif key == "style" and str(val) not in styles:
                del t[attr]


def _restore_page_anchors(src: Tag, dest: Tag) -> None:
    """译文丢掉的分页锚点补回去，否则书内页码链接失效。"""
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
    """段落纯文本。分页锚点里的页码不是正文，不能拿来判断译文还在不在。"""
    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, Tag) and _local(node.name) == "br":
            if not any(_is_page_anchor(p) for p in node.parents if isinstance(p, Tag)):
                parts.append("\n")
        elif type(node) is NavigableString:
            parents = [p for p in node.parents if isinstance(p, Tag)]
            if any(_is_page_anchor(p) for p in parents):
                continue
            if any(_local(p.name) in SKIP_TAGS - {"code"} for p in parents):
                continue
            parts.append(str(node))
    text = "".join(parts)
    lines = [WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _sanitize_translation(src_html: str, translated_html: str) -> list | None:
    """校验并清洗译文 HTML 片段，返回可插入的节点列表；不合格返回 None（调用方回退纯文本）。

    - 剔除 script/style 等危险标签；
    - 原文没有的标签 unwrap（只保留文字），防模型自造结构；
    - 丢掉事件处理、脚本 URL，以及原文没有的属性；
    - 译文为空、解析失败、原文的行内格式一个都没保住，或原文有正文而译文去掉分页锚点后没有文字时，判为不合格。
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
    # 页码在分页锚点里，补回来之后仍不算译文。原文去掉锚点后还有文字、译文没有，就是模型把段落清空了。
    if _prose_of(src) and not _prose_of(frag):
        return None
    if not frag.get_text(strip=True) and not any(_is_page_anchor(t) for t in frag.find_all(True)):
        return None
    src_rich = src.find(lambda t: _local(t.name) in RICH_TAGS)
    if src_rich and not frag.find(lambda t: _local(t.name) in RICH_TAGS):
        return None
    return list(frag.children)


def apply_translation_html(soup, el: Tag, src_html: str, translated_html: str, bilingual: bool, lang: str = ""):
    """富文本段落回写：译文保留行内标签（加粗、链接等）。不合格时回退纯文本路径。"""
    if translated_html.strip() == src_html.strip():
        return
    nodes = _sanitize_translation(src_html, translated_html)
    if nodes is None:
        frag = _parse_fragment(translated_html)
        # <br> 换成换行；分页锚点里的页码不是译文，不能拿来覆盖原文
        plain = _prose_of(frag) if frag is not None else translated_html
        target = apply_translation(soup, el, _text_of(el), plain, bilingual, lang)
        src = _parse_fragment(src_html)
        if target is not None and src is not None:
            _restore_page_anchors(src, target)
        return
    name = _local(el.name)
    attrs = lang_attrs(lang)
    # 译文里媒体元素全丢时，把原文的补在末尾（img 在富文本段落里很少见，位置不苛求）
    src_frag = _parse_fragment(src_html)
    src_media = src_frag.find_all(lambda t: _local(t.name) in MEDIA_TAGS) if src_frag else []
    has_media = any(_local(getattr(n, "name", None)) in MEDIA_TAGS
                    or (isinstance(n, Tag) and n.find(lambda t: _local(t.name) in MEDIA_TAGS)) for n in nodes)

    def fill(target: Tag):
        target.clear()
        for n in nodes:
            target.append(n)
        if src_media and not has_media:
            for m in src_media:
                target.append(m)

    if not bilingual:
        fill(el)
        for k, v in attrs.items():
            el[k] = v
        return
    if name in INNER_TAGS:
        span = soup.new_tag("span", attrs={"class": TRANSLATION_CLASS, **attrs})
        fill(span)
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
    fill(clone)
    for k, v in attrs.items():
        clone[k] = v
    el.insert_after(clone)


def parse(content: bytes | str, xml: bool) -> BeautifulSoup:
    soup = BeautifulSoup(content, "lxml-xml" if xml else "lxml")
    # lxml-xml 遇到带内部 DTD 子集（<!DOCTYPE html [<!ENTITY nbsp "&#160;">]>）的 XHTML 会静默产出空文档，
    # 原文非空但解析不出任何元素时回退到 HTML 模式重新解析
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
    """EPUB3 导航文档：只翻译 <a>/<span> 的文字，不改变结构。"""
    out = []
    for el in soup.find_all(lambda t: _local(t.name) in ("a", "span")):
        if el.find(lambda t: _local(t.name) in ("a", "span")):
            continue
        text = _text_of(el)
        if translatable(text):
            out.append((el, text))
    return out


def set_label(el: Tag, text: str, translated: str, bilingual: bool):
    if not translated or translated == text:
        return
    el.string = f"{text} / {translated}" if bilingual else translated
