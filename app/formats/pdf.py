"""PDF 翻译，保留原版式。

1. 用 PyMuPDF 提取每页的文本块（坐标、字号、颜色、粗体）。
2. 跳过公式、纯数字、旋转文字等不适合翻译的块。
3. 译文版：用 redaction 擦掉原文字（图片和矢量图形保留），在原位置用 HTML 盒子写入译文，字号放不下时自动缩小。
4. 双语版：原页面和译文页面左右并排放在同一页上，方便对照阅读。
"""
import asyncio
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..languages import RTL_LANGUAGES

MATH_FONT_RE = re.compile(r"CMMI|CMSY|CMEX|MSBM|Math|Symbol|STIX|Cambria Math", re.I)
# 精确的数学字体白名单（直接判公式）
MATH_FONT_PRECISE_RE = re.compile(
    r"Asana|FiraMath|STIX|TeXGyre.*Math|XITSMath|LibertinusMath|MathJax|Cambria Math|LatinModern.*Math", re.I)
# 常见正文字体黑名单（优先于启发式，里面的希腊字母/符号靠字符级规则识别）
TEXT_FONT_RE = re.compile(
    r"Times|Arial|Calibri|Minion|Palatino|\bCMR|Charter|Georgia|Helvetica|Verdana|Roboto|Lato|"
    r"Open ?Sans|Source ?Sans|Noto(?!.*Math)|Libertine(?!.*Math)|Garamond|Baskerville|Bookman|"
    r"Courier|Consolas|Menlo|Monaco|Ubuntu|DejaVu|Liberation|FreeSerif|FreeSans|Nimbus|"
    r"Trebuchet|Candara|Constantia|Franklin|Gill|Lucida|Segoe|Optima|Futura|Avenir|Univers|"
    r"Myriad|Frutiger|\bDIN\b|Proxima|Museo|\bPT S|Merriweather|Lora|Crimson|Playfair|Cormorant|"
    r"Spectral|Inter|Work Sans|IBM Plex|Charis|Gentium|Doulos|Andika|"
    r"PingFang|SimSun|SimHei|Songti|Heiti|Kaiti|FangSong|SourceHan|WenQuanYi|YaHei|YouYuan|LiSu|"
    r"STFangsong|STHeiti|STKaiti|STSong|STXihei|Yuanti|Hiragino|Ryumin|GothicBBB|Kozuka|IPA|"
    r"Takao|Sazanami|Motoya|Yomogi|BIZ |M PLUS|GenShin|GenRyu|NotoSansCJK|NotoSerifCJK", re.I)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
NO_TEXT_RE = re.compile(r"^[\W\d_]*$")
# 行首列表标记：符号弹点（可不带空格）、短横线类（须带空格，避免误伤连字符单词）、数字/字母编号
BULLET_RE = re.compile(r"^\s*(?:[•◦▪‣·○■□►»]\s*|[-–—*]\s+|\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+)")
# 新段落的开头：大写字母、数字、引号/括号（小写字母开头多半是自动换行的续行）。
# CJK 不在其中：CJK 行首无法区分新句子和续行，只有上行以句末标点结尾才算新段落（见下）
SEG_START_RE = re.compile(r"^[A-Z0-9À-Þ(\"'“‘\[（【]")
CJK_START_RE = re.compile(r"^[぀-ヿ㐀-鿿가-힯]")
# 不含分号：分号在下面的连接标点里，硬换行不会因此拆段
SENT_ENDS = ("。", "！", "？", "：", "…", ".", "!", "?", ":", '"', "'", "”", "’", "」", "』", ")", "）")


@dataclass
class TextBlock:
    page: int
    rect: pymupdf.Rect
    line_rects: list[pymupdf.Rect]
    text: str
    size: float
    color: str
    bold: bool
    # 每行的原始 span（含空白 span）：重建送翻文本时与原文逐字一致
    span_lines: list[list[dict]] = None


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and re.match(r"[a-zA-ZÀ-ÿ]", line):
            out = out[:-1] + line  # 行尾连字符断词（德语等断词后首字母可能大写）
        elif CJK_RE.search(out[-1]) or CJK_RE.search(line[0]):
            out += line
        else:
            out += " " + line
    return out


def _split_lines(items: list[tuple]) -> list[list[int]]:
    """把一个文本块里的行分成独立段落（返回每组的行下标）。

    幻灯片里多个列表项、甚至不同的图注标签常被 PyMuPDF 归进同一块，整段翻译会把
    项目符号变成行内文字、把不相干的内容拼在一起。三处拆段：
    - 行首带列表标记；
    - 纯符号行（“=”“→”等）单独成段，之后被过滤器跳过、原样保留；
    - 左跳超过约 3 个字号、且两行没有都撑满右边距（并列标签）。1–2em 是首行缩进，
      不管上一行有没有撑满，都不拆。
    - 上行明显没到右边距（硬换行而非自动换行）且下行像新句子开头；上行以逗号等
      连接标点结尾、左跳只是缩进、或仍是同一条列表项的续行，都不算硬换行。
    自动换行的续行（小写开头、上行撑满）仍并入上一段。
    items 的元素是 (行文本, 行 rect, 行 span, ...) 的元组，只用前三个。
    """
    if not items:
        return []
    max_x1 = max(it[1].x1 for it in items)
    groups, cur = [], [0]
    for i in range(1, len(items)):
        text, rect = items[i][0], items[i][1]
        ptext, prect, pspans = items[cur[-1]][0], items[cur[-1]][1], items[cur[-1]][2]
        psize = max((s["size"] for s in pspans), default=10)
        stripped = text.lstrip()
        prev_symbol = bool(NO_TEXT_RE.match(ptext.strip()))  # 纯符号行独立成段，后面的内容不和它拼
        jump = prect.x0 - rect.x0
        indent = 0 < jump <= 3 * psize  # 1–2em 左跳是首行缩进，不是新段落
        # 列表项的续行：缩进在弹点之后的悬挂对齐。同 x0 或左跳的不是续行，可能是下一段
        continuing_list = bool(BULLET_RE.match(ptext)) and not BULLET_RE.match(text) and rect.x0 > prect.x0
        # CJK 行首分不出新句子和续行，要求上行以句末标点结尾才算新段落
        seg_start = bool(SEG_START_RE.match(stripped)) or (
            bool(CJK_START_RE.match(stripped)) and ptext.rstrip().endswith(SENT_ENDS)
        )
        hard_break = (
            max_x1 - prect.x1 > 3 * psize
            and seg_start
            and not prev_symbol
            and not indent
            and not continuing_list
            and not ptext.rstrip().endswith((",", "，", "、", ";", "；", "(", "+", "*", "/", "=", "<", ">"))
        )
        both_full = max_x1 - prect.x1 <= 3 * psize and max_x1 - rect.x1 <= 3 * psize
        left_jump = jump > 3 * psize and not both_full
        # 两行纵向交叠超过一半、且横向区间相交：是同一可视行被上下标/公式拆出来的
        # （y_1^T 的 1 和 T 各算一行，x 区间是交错咬合的），不能在这里拆段。
        # 并排标签横向是分开的（无横向交叠），left_jump 照常拆开。
        overlap = min(prect.y1, rect.y1) - max(prect.y0, rect.y0)
        h_overlap = min(prect.x1, rect.x1) - max(prect.x0, rect.x0)
        same_visual_line = (h_overlap > 0
                            and overlap > min(prect.y1 - prect.y0, rect.y1 - rect.y0) * 0.5)
        if not same_visual_line and (
            BULLET_RE.match(text) or NO_TEXT_RE.match(stripped) or prev_symbol or hard_break or left_jump
        ):
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return groups


def _preview_kind(block: TextBlock, median: float) -> str:
    """短块里，明显大于正文的、以及略大于正文的粗体，当作小标题。

    只看粗体不够：作者名和图注标签也常是粗体，但字号不超过正文。
    """
    if len(block.text) >= 100:
        return "p"
    if block.size >= median * 1.3 or (block.bold and block.size >= median * 1.15):
        return "h2"
    return "p"


def _font_is_math(font: str) -> bool:
    """字体三层判定（BabelDOC 思路）：精确白名单 → 是公式；正文黑名单 → 不是；宽泛启发式 → 是。
    黑名单优先，避免 Times/Arial 里的数学段落被误判。"""
    if MATH_FONT_PRECISE_RE.search(font):
        return True
    if TEXT_FONT_RE.search(font):
        return False
    return bool(MATH_FONT_RE.search(font))


def _is_formula_char(c: str) -> bool:
    """字符级公式判定：希腊字母、数学符号/修饰符、私用区。"""
    if 0x370 <= ord(c) <= 0x3FF:
        return True
    return unicodedata.category(c) in ("Sm", "Sk", "Mn", "Co")


def _math_chars(spans: list[dict]) -> int:
    """块里公式字符数：公式字体的全部字符 + 正文字体里的公式字符。"""
    n = 0
    for s in spans:
        if _font_is_math(s["font"]):
            n += len(s["text"])
        else:
            n += sum(1 for c in s["text"] if _is_formula_char(c))
    return n


# 句末标点：判断页面末尾的块是不是被页边界切断的段落
_SENT_END_PUNCT = tuple(".!?:;…。！？；：\"'”’)]}》")


def _cross_page_units(blocks: list[TextBlock], median: float, skip_ids: set[int] = frozenset()) -> list[list[TextBlock]]:
    """把跨页续段两两合并成翻译单元（返回块列表的列表，每单元 1~2 块）。

    上一页最后一个正文块不以句末标点结尾、下一页第一个块以小写/数字开头且不像标题时，
    视为被页边界切断的同一段。页脚脚注（字号明显小于正文）和指定跳过的块不参与。
    """
    by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)

    units: list[list[TextBlock]] = []
    pages = sorted(by_page)
    used: set[int] = set()
    for pno in pages:
        body = [b for b in by_page[pno] if b.size >= median * 0.8]
        for b in by_page[pno]:
            if id(b) in used:
                continue
            nxt = by_page.get(pno + 1)
            first = next((x for x in (nxt or []) if x.size >= median * 0.8), None)
            head = first.text.lstrip()[:1] if first else ""
            if (body and b is body[-1] and first is not None and id(first) not in used
                    and id(b) not in skip_ids and id(first) not in skip_ids
                    and not b.text.rstrip().endswith(_SENT_END_PUNCT)
                    and (head.islower() or head.isdigit())
                    and first.size <= b.size * 1.4):
                units.append([b, first])
                used.add(id(b))
                used.add(id(first))
            else:
                units.append([b])
                used.add(id(b))
    return units


def _split_translation(t: str, ratio: float) -> tuple[str, str]:
    """把跨页合并段的译文按比例切回两页：优先在比例附近的句读点断开，
    没有合适标点时按空格，再不行硬切。"""
    target = len(t) * ratio
    best = None
    for m in re.finditer(r"[。！？；，、：.!?;,:]", t):
        if best is None or abs(m.end() - target) < abs(best - target):
            best = m.end()
    if best is not None and abs(best - target) <= len(t) * 0.3:
        return t[:best], t[best:]
    # 拉丁文本按空格切
    spaces = [m.start() for m in re.finditer(r"\s", t)]
    if spaces:
        cut = min(spaces, key=lambda s: abs(s - target))
        return t[:cut], t[cut:]
    cut = round(target)
    return t[:cut], t[cut:]


def _merge_visual_lines(blocks: list[TextBlock]) -> list[TextBlock]:
    """合并其实是同一可视行的相邻块（字体在公式处切换时，PyMuPDF 会把一行拆成两块）。

    判定：纵向交叠超过较矮块的一半、且横向区间相交。双栏的左右栏横向不相交，
    不会被误并。不合并的话，两块译文会写进互相交叠的矩形里叠在一起。
    """
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            v_overlap = min(p.rect.y1, b.rect.y1) - max(p.rect.y0, b.rect.y0)
            h_overlap = min(p.rect.x1, b.rect.x1) - max(p.rect.x0, b.rect.x0)
            if (p.page == b.page and h_overlap > 0
                    and v_overlap > min(p.rect.y1 - p.rect.y0, b.rect.y1 - b.rect.y0) * 0.5):
                p.line_rects.extend(b.line_rects)
                if p.span_lines and b.span_lines:
                    p.span_lines.extend(b.span_lines)
                else:
                    p.span_lines = None
                p.text = _join_lines([p.text, b.text])
                p.rect |= b.rect
                continue
        out.append(b)
    return out


def _is_formula_span(span: dict, main_size: float) -> bool:
    """行内公式：数学字体的 span，或明显小于正文的角标 span（引用编号等）。"""
    return _font_is_math(span["font"]) or span["size"] < main_size * 0.79


def _formula_markup(run_spans: list[dict], main_size: float) -> str:
    """把公式 run 拼成带上下标哨兵的字符串（\\x01s\\x02..\\x01/s\\x02 上标，b 为下标）。
    恢复进译文后由渲染层换成 <sup>/<sub>，y_1^T 不再被拍平成 y1T。
    上/下标按 baseline 偏移判定：比正文 baseline 高的是上标，低的是下标。"""
    text_spans = [s for s in run_spans if s["size"] >= main_size * 0.79]
    origins = [s["origin"][1] for s in text_spans if s.get("origin")]
    main_origin = origins[0] if origins else None
    if main_origin is None:
        return "".join(s["text"] for s in run_spans)
    out, role = [], ""
    for s in run_spans:
        r = ""
        if s["size"] < main_size * 0.79 and s.get("origin"):
            dy = s["origin"][1] - main_origin
            if dy < -1:
                r = "s"
            elif dy > 1:
                r = "b"
        if r != role:
            if role:
                out.append(f"\x01/{role}\x02")
            if r:
                out.append(f"\x01{r}\x02")
            role = r
        out.append(s["text"])
    if role:
        out.append(f"\x01/{role}\x02")
    return "".join(out)


def _placeholderize(block: TextBlock) -> tuple[str, list[str]]:
    """把块里的行内公式换成 {vN} 占位符（BabelDOC 的翻译协议），返回 (送翻文本, 公式列表)。

    没有公式时返回的文本与原文逐字一致，缓存照常命中。相邻的公式 span 和它们之间的
    空白合并成一个占位符；上/下标用哨兵标记随公式一起保留。
    """
    if not block.span_lines:
        return block.text, []
    formulas: list[str] = []
    out_lines: list[str] = []
    for spans in block.span_lines:
        parts: list[str] = []
        i, n = 0, len(spans)
        while i < n:
            if _is_formula_span(spans[i], block.size):
                run = spans[i]["text"]
                run_spans = [spans[i]]
                j = i + 1
                while j < n:
                    if not spans[j]["text"].strip():
                        # 空白 span：只有后面还跟着公式时才并入 run；
                        # run 末尾的空白留在原文里，保住公式和正文之间的间隔
                        if any(_is_formula_span(spans[k], block.size) for k in range(j + 1, n)):
                            run += spans[j]["text"]
                            run_spans.append(spans[j])
                            j += 1
                            continue
                        break
                    if not _is_formula_span(spans[j], block.size):
                        break
                    run += spans[j]["text"]
                    run_spans.append(spans[j])
                    j += 1
                if run.strip():
                    formulas.append(_formula_markup(run_spans, block.size).strip())
                    parts.append(f"{{v{len(formulas)}}}")
                else:
                    parts.append(run)
                i = j
            else:
                parts.append(spans[i]["text"])
                i += 1
        out_lines.append("".join(parts))
    return _join_lines(out_lines), formulas


# 译文里游离的 {vN} 占位符（模型把占位符写进了别的段）
_STRAY_PLACEHOLDER_RE = re.compile(r"\{\s*v\s*\d+\s*\}")


def _restore_placeholders(translation: str, formulas: list[str]) -> str | None:
    """译后恢复 {vN}（容忍模型在占位符里塞的空格）；有占位符丢失时返回 None（回退原文）。
    译文里多出来的幻觉占位符直接删掉。"""
    for n, formula in enumerate(formulas, 1):
        pattern = re.compile(rf"\{{\s*v\s*{n}\s*\}}")
        if not pattern.search(translation):
            return None
        translation = pattern.sub(lambda _: formula, translation, count=1)
    return re.sub(r"\{\s*v\s*\d+\s*\}", "", translation)


def extract_blocks(doc: pymupdf.Document) -> list[TextBlock]:
    blocks = []
    for pno, page in enumerate(doc):
        data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
        for b in data["blocks"]:
            if b.get("type") != 0:
                continue
            items: list[tuple[str, pymupdf.Rect, list[dict]]] = []
            for ln in b["lines"]:
                if abs(ln["dir"][1]) > 0.01:  # 旋转 / 竖排文字不处理
                    continue
                text = "".join(s["text"] for s in ln["spans"])
                if not text.strip():
                    continue
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if spans:
                    items.append((text, pymupdf.Rect(ln["bbox"]), spans, ln["spans"]))
            for group in _split_lines(items):
                texts = [items[i][0] for i in group]
                line_rects = [items[i][1] for i in group]
                spans = [s for i in group for s in items[i][2]]
                span_lines = [items[i][3] for i in group]
                text = _join_lines(texts)
                total = sum(len(s["text"]) for s in spans)
                letters = sum(c.isalpha() for c in text)
                if NO_TEXT_RE.match(text) or _math_chars(spans) > total * 0.3 or letters < 2 or letters < len(text) * 0.4:
                    continue
                # 字号、颜色取占比最多的 span
                main = max(spans, key=lambda s: len(s["text"]))
                rect = pymupdf.Rect()
                for r in line_rects:
                    rect |= r
                blocks.append(TextBlock(
                    page=pno,
                    rect=rect,
                    line_rects=line_rects,
                    text=text,
                    size=round(main["size"], 1),
                    color=f"#{main['color']:06x}",
                    bold=bool(main["flags"] & 16) or "Bold" in main["font"],
                    span_lines=span_lines,
                ))
    return _merge_visual_lines(blocks)


def _render_translated(src_path: Path, blocks: list[TextBlock], translations: list[str],
                       target_lang: str = "") -> pymupdf.Document:
    rtl = target_lang.split("-")[0] in RTL_LANGUAGES
    # CJK 字体的行框比拉丁高（约 1.31em vs 1.16em），且字形顶部会越出给定区域：
    # 按原字号写入会和下一行叠在一起，字号缩小并下移补偿
    # 粤语、文言文走同一套 CJK 字体，行框一样偏高
    cjk = target_lang.split("-")[0] in ("zh", "ja", "ko", "yue", "wyw")
    # lang 让 MuPDF 为中日韩选对字形；dir=rtl 让阿拉伯语、希伯来语从右往左排
    div_attrs = f' lang="{html.escape(target_lang)}"' if target_lang else ""
    if rtl:
        div_attrs += ' dir="rtl" style="text-align: right"'
    doc = pymupdf.open(src_path)
    by_page: dict[int, list[tuple[TextBlock, str]]] = {}
    for b, t in zip(blocks, translations):
        if t and t.strip() and t.strip() != b.text.strip():
            by_page.setdefault(b.page, []).append((b, t))

    for pno, items in by_page.items():
        page = doc[pno]
        for b, _ in items:
            for r in b.line_rects:
                page.add_redact_annot(r, fill=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        )
        for b, t in items:
            weight = "bold" if b.bold else "normal"
            body = (
                html.escape(t)
                .replace("\n", "<br>")
                .replace("\x01s\x02", "<sup>")
                .replace("\x01/s\x02", "</sup>")
                .replace("\x01b\x02", "<sub>")
                .replace("\x01/b\x02", "</sub>")
            )
            size = b.size * 0.88 if cjk else b.size
            css = (
                f"* {{font-family: sans-serif; font-size: {size}px; color: {b.color}; "
                f"font-weight: {weight}; line-height: 1.2; margin: 0; padding: 0;}}"
            )
            # 留一点余量，避免译文比原文长时被截断；放不下时走三级收缩阶梯
            y0 = b.rect.y0 + b.size * 0.1 if cjk else b.rect.y0
            rect = pymupdf.Rect(b.rect.x0, y0, b.rect.x1 + 2, b.rect.y1 + b.size * 0.3)
            obstacles = [ob.rect for ob in blocks if ob.page == pno and ob is not b]
            _insert_fitting(page, rect, f"<div{div_attrs}>{body}</div>", css, obstacles)
    return doc


def _expand_right(rect: "pymupdf.Rect", page_width: float, obstacles: list["pymupdf.Rect"]) -> "pymupdf.Rect":
    """排版收缩第一级：把写入框向右扩（上限 90% 页宽），不盖住右侧纵向有交叠的文本块。"""
    x1 = page_width * 0.9
    for o in obstacles:
        if o.x0 <= rect.x0 + 1:  # 不是右侧的块
            continue
        if o.y0 < rect.y1 and o.y1 > rect.y0:  # 纵向上有交叠
            x1 = min(x1, o.x0 - 2)
    return pymupdf.Rect(rect.x0, rect.y0, max(x1, rect.x1), rect.y1)


def _try_insert(page: "pymupdf.Page", rect: "pymupdf.Rect", html_text: str, css: str, scale_low: float) -> bool:
    """insert_htmlbox 放不下返回负值（此时不画内容）。pymupdf 在缩放刚好等于 scale_low 时
    会因浮点误差触发 assert（0.8999999999999999 < 0.9），按没放下来处理。"""
    try:
        return page.insert_htmlbox(rect, html_text, css=css, scale_low=scale_low)[0] >= 0
    except AssertionError:
        return False


def _insert_fitting(page: "pymupdf.Page", rect: "pymupdf.Rect", html_text: str, css: str,
                    obstacles: list["pymupdf.Rect"]):
    """排版三级收缩（BabelDOC 思路）：先右扩 → 压行距 → 最后才缩字号。

    每一级只在放不下（insert_htmlbox 返回负值，此时不会画出内容）时进入下一级，
    放得下的段落和旧的 scale_low=0 直接缩比，字号明显更大。
    """
    if _try_insert(page, rect, html_text, css, 0.9):
        return
    wide = _expand_right(rect, page.rect.width, obstacles)
    if wide.x1 > rect.x1 + 1 and _try_insert(page, wide, html_text, css, 0.9):
        return
    target = wide if wide.x1 > rect.x1 + 1 else rect
    css_tight = css.replace("line-height: 1.2", "line-height: 1.1")
    if _try_insert(page, target, html_text, css_tight, 0.9):
        return
    # 兜底：和原来一样缩到能放下为止
    page.insert_htmlbox(target, html_text, css=css_tight, scale_low=0)


def _render_side_by_side(src: pymupdf.Document, translated: pymupdf.Document) -> pymupdf.Document:
    out = pymupdf.open()
    for i in range(len(src)):
        w, h = src[i].rect.width, src[i].rect.height
        page = out.new_page(width=w * 2, height=h)
        page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, i)
        page.show_pdf_page(pymupdf.Rect(w, 0, w * 2, h), translated, i)
    return out


_BIBLIO_CITE_RE = re.compile(r"\((?:19|20)\d{2}\)|et al\.|doi:|https?://", re.I)
# 行尾 “, 2016.” 在正文里太常见。确认进入时必须同时是编号条目，或带 arXiv/CoRR/doi。
_BIBLIO_YEAR_END_RE = re.compile(r",\s*(?:19|20)\d{2}\s*[.]?\s*$")
_BIBLIO_TRAIL_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*[.]?\s*$|年版\s*[.。]?\s*$")
_BIBLIO_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# 卷期续行：2020, 3(2) / Vol. 3 / Volume 3 / pp. 1-10 / 第3期。
# Volume 后面必须是阿拉伯数字，避免 “volume were”。
_BIBLIO_VOLUME_RE = re.compile(
    r"^(?:19|20)\d{2}\s*[,，(（]\s*(?:[\d(（]|[Vv]ol\.?\s*\d|[Vv]olume\.?\s*\d|[Nn]o\.?(?:\s|\d)|[Pp]p?\.?(?:\s|\d)|第)"
)
# 只认缩写 Vol. 后的罗马数字。不含 M/D，否则 mix / mill 会当成卷号。
_ROMAN_VOL_RE = re.compile(
    r"(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})(?![A-Za-z])",
    re.I,
)
_BIBLIO_VOL_LABEL_RE = re.compile(r"^(?:19|20)\d{2}\s*[,，(（]\s*[Vv]ol\.?\s*")
# 同字号退出用。数字后要是期号、页码或行尾；No/pp 后要是数字；「第」后要是「期」。
# 行首只是年份的下一章（2020, 1. Introduction / 第1章）不能靠这条留下。
_BIBLIO_EXIT_VOLUME_RE = re.compile(
    r"^(?:19|20)\d{2}\s*[,，(（]\s*(?:"
    r"\d+\s*(?:[（(]|[:：]\s*\d|[,，]\s*\d|[.。]?\s*$)"
    r"|\d+\s*[）)]"
    r"|[Vv]ol\.?\s*\d|[Vv]olume\.?\s*\d|[Nn]o\.?\s*\d|[Pp]p?\.?\s*\d"
    r"|第\s*\d+\s*期"
    r")"
)
# 出版社行可以收在 2019。 / 2019年。 / 年版。出版社后面还有字的是正文。
_BIBLIO_IMPRINT_END_RE = re.compile(r"(?:(?:19|20)\d{2}年?|年版)\s*[.。]?\s*$")
_BIBLIO_MARK_RE = re.compile(r"\barXiv\b|\bCoRR\b|doi:|https?://", re.I)
# 题名后的著录位置才算，如 研究[J]. 刊名。句末单独的 [M]。 是正文，不是引文。
_BIBLIO_TYPE_MARK_RE = re.compile(r"(?<=\S)\[[JMDCNRP]\]\s*[.。]\s*\S")
# [1]张三 中间常没有空格。1. 仍要空格，避免把 1.2 当成编号。
_BIBLIO_LEAD_RE = re.compile(r"^(?:\[\d+\]\s*|\d{1,3}[.)]\s+)")
_BIBLIO_WRAP_RE = re.compile(r"^[a-zà-öø-ÿ]")
# 弱信号：只有编号条目才算。“这项研究发表于某出版社。”是正文。
_BIBLIO_CN_PUB_RE = re.compile(r"出版社|年版")


def _biblio_heading_like(block: TextBlock, text: str, median: float) -> bool:
    if not text or len(text) > 120:
        return False
    if block.size > median + 0.4:
        return True
    return bool(block.bold and block.size + 0.4 >= median and len(text) < 80)


def _biblio_entry_citation(text: str) -> bool:
    """标题后面这行像引文，才进入参考文献。正文里的 2017 / arXiv / 出版社 / [J] 不算。"""
    if _BIBLIO_CITE_RE.search(text):
        return True
    # 类型标记 + 年份才是著录条目；“出处见某书[M]。后续还有讨论。”是正文
    if _BIBLIO_TYPE_MARK_RE.search(text) and _BIBLIO_YEAR_RE.search(text):
        return True
    lead = bool(_BIBLIO_LEAD_RE.match(text))
    mark = bool(_BIBLIO_MARK_RE.search(text))
    if _BIBLIO_YEAR_END_RE.search(text) and (lead or mark):
        return True
    return lead and (mark or bool(_BIBLIO_CN_PUB_RE.search(text)))


def _vol_roman_continuation(text: str) -> bool:
    """Vol. III / vol. xii 是卷号。单词 mix / mill 不是。"""
    head = _BIBLIO_VOL_LABEL_RE.match(text)
    if head is None:
        return False
    return _ROMAN_VOL_RE.match(text, head.end()) is not None


def _biblio_exit_continuation(text: str) -> bool:
    """同字号时仍算引文续行。行首像年份的下一章不算。"""
    if _BIBLIO_EXIT_VOLUME_RE.match(text) or _vol_roman_continuation(text):
        return True
    return bool(_BIBLIO_CN_PUB_RE.search(text) and _BIBLIO_IMPRINT_END_RE.search(text))


def _biblio_year_continuation(text: str) -> bool:
    """类型标记的下一块仍是著录。后来随便出现的年份不算。"""
    if _BIBLIO_VOLUME_RE.match(text) or _vol_roman_continuation(text):
        return True
    # 出版社行以年份或「年版」收尾。正文提到出版社后还有下文的不算。
    return bool(_BIBLIO_CN_PUB_RE.search(text) and _BIBLIO_IMPRINT_END_RE.search(text))


def _biblio_citation_line(block: TextBlock, text: str, start_size: float, median: float) -> bool:
    """已经在参考文献里：这行是引文或换行前的条目，不能当成下一章。"""
    if _BIBLIO_CITE_RE.search(text) or _BIBLIO_MARK_RE.search(text):
        return True
    heading = _biblio_heading_like(block, text, median)
    short_heading = heading and len(text) < 40
    # 比起始标题更大，或同字号短标题，仍是下一章。超过 80 字的加粗行先让给出类型标记和行尾年份。
    if block.size > start_size + 0.5 or short_heading or (block.bold and heading):
        return False
    if _BIBLIO_TYPE_MARK_RE.search(text) or _BIBLIO_TRAIL_YEAR_RE.search(text):
        return True
    # 真正的卷期、出版社续行和标题同字号也不是下一章。
    if _biblio_exit_continuation(text):
        return True
    if not _BIBLIO_LEAD_RE.match(text):
        return False
    return True


def biblio_skips(blocks: list[TextBlock]) -> list[bool]:
    """标记参考文献章节的块。标题本身不跳过，章节正文保留原文。

    页眉和目录里的同名短行不能把进入字号改小；同等字号的引文也不是下一章标题。
    """
    from .html_blocks import BIBLIO_HEADING_RE, BIBLIO_SHORT_LABEL_RE, BIBLIO_SUBHEADING_RE

    if not blocks:
        return []
    sizes = sorted(b.size for b in blocks)
    median = sizes[len(sizes) // 2]
    prepared = [(b, re.sub(r"\s+", " ", b.text).strip()) for b in blocks]

    def smaller_subsection(block: TextBlock, text: str, origin: float) -> bool:
        # 同字号或更大的 Books / 论文是下一章，不能当成内部小标题。
        if len(text) >= 40 or block.size + 0.4 >= origin:
            return False
        if BIBLIO_SUBHEADING_RE.match(text):
            return True
        return bool(BIBLIO_SHORT_LABEL_RE.match(text) and _biblio_heading_like(block, text, median))

    def citations_follow(idx: int) -> bool:
        origin = prepared[idx][0].size
        pending_lead = False
        pending_type = False
        for block, text in prepared[idx + 1:idx + 16]:
            # Endnotes / 参考文献是另一节的标题，必须停住，不能把进入字号抬高
            if len(text) < 40 and BIBLIO_HEADING_RE.match(text):
                return False
            if smaller_subsection(block, text, origin):
                pending_lead = False
                pending_type = False
                continue
            # 像标题的行即使含 [J] 或年份也不是引文，先停住
            if _biblio_heading_like(block, text, median) and len(text) < 80:
                return False
            # 编号和出版社、类型标记和年份，都常被拆成相邻两块。只看下一块。
            if (
                _biblio_entry_citation(text)
                or (pending_lead and _BIBLIO_CN_PUB_RE.search(text))
                or (pending_type and _biblio_year_continuation(text))
            ):
                return True
            pending_lead = bool(_BIBLIO_LEAD_RE.match(text))
            pending_type = bool(_BIBLIO_TYPE_MARK_RE.search(text) and not _BIBLIO_YEAR_RE.search(text))
        return False

    def exits(block: TextBlock, text: str, start_size: float, prev_numbered: bool) -> bool:
        if not text or len(text) > 120 or smaller_subsection(block, text, start_size):
            return False
        if block.size + 0.5 < start_size * 0.8:
            return False
        if _biblio_citation_line(block, text, start_size, median):
            return False
        # 引文里单独一行 Journal 不是下一章
        if len(text) < 40 and BIBLIO_SHORT_LABEL_RE.match(text) and not _biblio_heading_like(block, text, median):
            return False
        # 编号条目的换行续行以小写字母开头，不是下一章
        if prev_numbered and _BIBLIO_WRAP_RE.match(text):
            return False
        if _biblio_heading_like(block, text, median):
            return True
        # 加粗长标题（heading_like 的加粗分支有 80 字上限）
        if block.bold and block.size + 0.4 >= median:
            return True
        return len(text) < 60 and block.size >= start_size - 0.5

    skips: list[bool] = []
    in_biblio = False
    start_size = 0.0
    prev_numbered = False
    for i, (block, text) in enumerate(prepared):
        title = len(text) < 40 and bool(BIBLIO_HEADING_RE.match(text))
        if title and in_biblio:
            prev_numbered = False
            if block.size + 0.5 >= start_size * 0.8:
                if block.size > start_size:
                    start_size = block.size
                skips.append(False)
            else:
                skips.append(True)
            continue
        if title and citations_follow(i) and (_biblio_heading_like(block, text, median) or block.size + 0.4 >= median):
            in_biblio = True
            start_size = block.size
            prev_numbered = False
            skips.append(False)
            continue
        if in_biblio and smaller_subsection(block, text, start_size):
            prev_numbered = False
            skips.append(False)
            continue
        if in_biblio and exits(block, text, start_size, prev_numbered):
            in_biblio = False
            prev_numbered = False
            skips.append(False)
            continue
        skips.append(in_biblio)
        if in_biblio and _BIBLIO_LEAD_RE.match(text):
            prev_numbered = True
        elif _BIBLIO_WRAP_RE.match(text):
            pass
        elif (
            len(text) < 40
            and BIBLIO_SHORT_LABEL_RE.match(text)
            and not _biblio_heading_like(block, text, median)
        ):
            pass  # 引文单词不是标题，编号条目的续行还没结束
        else:
            prev_numbered = False
    return skips


def _scan_status(doc: pymupdf.Document) -> str:
    """ok=正常；hidden_text=带隐藏 OCR 文本层（render mode 3，文字不可见但可提取）；
    scanned=完全没有文本层的纯扫描件。

    隐藏层文件照翻：OCR 层的坐标通常和图像对齐，译文写回原位置后是可见的，
    效果和 BabelDOC 的 ocr_workaround 一样。纯扫描件没有可翻的东西，明确拒绝。
    """
    text_pages = hidden_pages = 0
    for page in doc:
        if not page.get_text().strip():
            continue
        text_pages += 1
        if b" 3 Tr" in page.read_contents():  # 渲染模式 3 = 不可见文字
            hidden_pages += 1
    if text_pages == 0:
        return "scanned"
    if text_pages >= 3 and hidden_pages / text_pages > 0.8:
        return "hidden_text"
    return "ok"


def _subset_worker(data: bytes, queue):
    doc = pymupdf.open("pdf", data)
    doc.subset_fonts()
    queue.put(doc.tobytes(garbage=4, deflate=True))


def _subset_fonts_safe(doc: pymupdf.Document) -> pymupdf.Document:
    """子集化放到子进程里跑（BabelDOC 同款）：PyMuPDF 子集化偶尔崩溃或卡死，
    60 秒没结果就放弃，用未子集化的版本——文件大一些，但能看。
    用 fork 上下文：spawn 会重新 import 主模块，在 stdin/REPL 里直接失败。
    不支持 fork 的平台（Windows）退回进程内直接子集化。"""
    import multiprocessing

    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        try:
            doc.subset_fonts()
        except Exception:  # noqa: BLE001
            pass
        return doc
    data = doc.tobytes(garbage=4, deflate=True)
    queue = ctx.Queue()
    proc = ctx.Process(target=_subset_worker, args=(data, queue))
    proc.start()
    try:
        out = queue.get(timeout=60)
        proc.join(timeout=5)
        return pymupdf.open("pdf", out)
    except Exception:  # noqa: BLE001 - 超时或子进程崩溃都回退
        proc.kill()
        return pymupdf.open("pdf", data)


async def translate_pdf(src: Path, out_dir: Path, runner, bilingual: bool, target_lang: str = "") -> list[Path]:
    src_doc = pymupdf.open(src)
    if src_doc.needs_pass:
        raise ValueError("PDF 有密码保护，暂不支持")
    if _scan_status(src_doc) == "scanned":
        raise ValueError("这是没有文本层的扫描版 PDF：请先用 Acrobat、ABBYY 或 WPS 的 OCR 功能识别文字，"
                         "再上传识别后的文件")
    blocks = await asyncio.to_thread(extract_blocks, src_doc)
    if not blocks:
        raise ValueError("没有提取到可用的文字：可能是图片型页面或 OCR 文本层质量太差")
    toc = src_doc.get_toc()  # 输出文件带上原书签
    # PDF 元数据里的标题经常是 "Microsoft Word - xxx.docx" 之类，看起来像文件名就不用
    meta_title = ((src_doc.metadata or {}).get("title") or "").strip()
    if re.search(r"\.(docx?|pdf|tex|indd)$|^untitled$", meta_title, re.I):
        meta_title = ""
    runner.set_title(meta_title or src.stem)
    median = sorted(b.size for b in blocks)[len(blocks) // 2]
    kinds = [_preview_kind(b, median) for b in blocks]
    skips = biblio_skips(blocks)
    skip_ids = {id(b) for b, s in zip(blocks, skips) if s}
    # 跨页续段合并成翻译单元：“quite / well” 这类被页边界切断的句子不再拆成两半各翻各的
    units = _cross_page_units(blocks, median, skip_ids)
    block_kind = {id(b): k for b, k in zip(blocks, kinds)}

    sent_texts, sent_formulas, sent_kinds, sent_skips, flat_units = [], [], [], [], []
    for unit in units:
        texts, formulas = [], []
        for b in unit:
            s, f = _placeholderize(b)
            offset = len(formulas)
            if offset:
                s = re.sub(r"\{v(\d+)\}", lambda m: f"{{v{int(m.group(1)) + offset}}}", s)
            formulas.extend(f)
            texts.append(s)
        flat_units.append(unit)
        sent_texts.append(_join_lines(texts))
        sent_formulas.append(formulas)
        sent_kinds.append(block_kind[id(unit[0])])
        sent_skips.append(all(id(b) in skip_ids for b in unit))

    raw = await runner.translate_all(sent_texts, kinds=sent_kinds, preview=True, skip=sent_skips)

    per_block: dict[int, str] = {}
    for unit, t, sent_text, formulas in zip(flat_units, raw, sent_texts, sent_formulas):
        restored = _restore_placeholders(t, formulas) if formulas else _STRAY_PLACEHOLDER_RE.sub("", t)
        if restored is None:
            parts = [b.text for b in unit]  # 模型弄丢占位符：整单元回退原文，不产出坏文档
            shown = _join_lines([b.text for b in unit])
        else:
            if len(unit) == 1:
                parts = [restored]
            else:
                ratio = len(unit[0].text) / (len(unit[0].text) + len(unit[1].text))
                parts = list(_split_translation(restored, ratio))
            shown = restored
        for b, part in zip(unit, parts):
            per_block[id(b)] = part
        # 预览里显示恢复后的译文（缓存仍按占位符版本存，重跑照样命中）
        if formulas and getattr(runner, "_done", None) and runner._done.get(sent_text) == t:
            runner._done[sent_text] = shown
    translations = [per_block[id(b)] for b in blocks]

    def build():
        translated = _render_translated(src, blocks, translations, target_lang)
        # insert_htmlbox 每次都会嵌入完整的 CJK 字体（十几 MB），必须做子集化；失败则用未子集化版本
        translated = _subset_fonts_safe(translated)
        if bilingual:
            dst = out_dir / f"{src.stem}.bilingual.pdf"
            # 先落盘再读回，并排页面引用的是已经子集化的字体
            tmp = pymupdf.open("pdf", translated.tobytes(garbage=4, deflate=True))
            side = _render_side_by_side(src_doc, tmp)
            if toc:
                side.set_toc(toc)  # 并排页和原页一一对应，书签页码不用映射
            side.save(dst, garbage=4, deflate=True)
        else:
            dst = out_dir / f"{src.stem}.translated.pdf"
            if toc:
                translated.set_toc(toc)
            translated.save(dst, garbage=4, deflate=True)
        return dst

    return [await asyncio.to_thread(build)]
