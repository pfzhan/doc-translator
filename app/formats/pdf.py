"""PDF 翻译，保留原版式。

1. 用 PyMuPDF 提取每页的文本块（坐标、字号、颜色、粗体）。
2. 跳过整块公式、纯数字、旋转文字。行内公式是段落里的一个 run，译后按原样贴回。
3. 译文版：擦掉原文字（图片和矢量图形保留），在原位置写入译文。
   放不下时先在本栏内右扩（不盖住图和其他文字），再压行距，最后缩字号。
   有线表格按格翻译，写回时不越出格子。重复的页眉页脚和页码保留原文。
4. 双语版：原页面和译文页面左右并排放在同一页上，方便对照阅读。
"""
import asyncio
import html
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import pymupdf

from ..languages import RTL_LANGUAGES
from .pdf_align import draw_cjk_lines
from .pdf_flow import (
    STYLE_MARK_RE,
    Emphasis,
    break_lines,
    emphasis_of,
    nowrap_lines,
    peel_leading_space,
    restore_emphasis,
    separate_after_formula,
    strip_style_marks,
    writer_for,
)
from .pdf_layout import (
    PageGeometry, RuledTable, cell_containing, margin_skips, page_tables, reading_order,
)
from .pdf_roles import HeuristicLayout, LayoutItem, Role, is_size_heading
from .pdf_runs import BOUNDARY, FormulaRun, Run, TextRun, expand_box, intersecting_curves, split_page_boundary, strip_boundary
from .pdf_typeset import FormulaMetric, FormulaPiece, TextPiece, TypesetLine, typeset_lines

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
    # 占位符重建后的有序 run。提取阶段还没有，送翻前才填。
    runs: list[Run] = field(default_factory=list)
    # 有线表格的格子。写入不得越出；不同格子的块不能并成一段。
    clip: tuple[float, float, float, float] | None = None


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
        elif line[:1] in ")]}>，。；：、）】」.,":
            out += line  # 换行后的右括号、句号贴回公式，不在标点前加空格
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
    左跳/硬换行的比较基准是"上一条正文字号的行"（参考行）：上下标拆出的 dict
    小行会把左跳量算飞，让段落在公式处被切碎。
    items 的元素是 (行文本, 行 rect, 行 span, ...) 的元组，只用前三个。
    """
    if not items:
        return []
    max_x1 = max(it[1].x1 for it in items)
    groups, cur = [], [0]
    ref = 0  # 参考行：上一条正文字号的行。上下标拆出的 dict 小行不能当参考，
    # 否则左跳量相对小行算，段落会在公式处被切碎（"(like a | sequence)"）
    ref_size = max((s["size"] for s in items[0][2]), default=10)
    for i in range(1, len(items)):
        text, rect = items[i][0], items[i][1]
        ptext, prect, pspans = items[ref][0], items[ref][1], items[ref][2]
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
        qrect = items[i - 1][1]
        overlap = min(qrect.y1, rect.y1) - max(qrect.y0, rect.y0)
        h_overlap = min(qrect.x1, rect.x1) - max(qrect.x0, rect.x0)
        same_visual_line = (h_overlap > 0
                            and overlap > min(qrect.y1 - qrect.y0, rect.y1 - rect.y0) * 0.5)
        # 章节号和标题是同一行的整体（'2.2 Training'）：纯数字的上行不当符号行，
        # 也不按硬换行拆开，否则数字被过滤不遮罩、标题单独翻，基线对不齐
        sec_label = (bool(re.fullmatch(r"\d+(?:\.\d+)*\.?", ptext.strip()))
                     and overlap > min(prect.y1 - prect.y0, rect.y1 - rect.y0) * 0.5)
        if not same_visual_line and not sec_label and (
            BULLET_RE.match(text) or NO_TEXT_RE.match(stripped) or prev_symbol or hard_break or left_jump
        ):
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
        isize = max((s["size"] for s in items[i][2]), default=10)
        if isize >= ref_size * 0.79:
            ref, ref_size = i, isize
    groups.append(cur)
    return groups


def _preview_kind(block: TextBlock, median: float) -> str:
    """短块里，明显大于正文的、以及略大于正文的粗体，当作小标题。

    只看粗体不够：作者名和图注标签也常是粗体，但字号不超过正文。
    有版面角色时，标题以角色为准，这里只留字号兜底。
    """
    return "h2" if is_size_heading(block.text, block.size, block.bold, median) else "p"


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


def _is_symbol_span(text: str) -> bool:
    """正文字体里整段都是希腊字母或数学符号。纯重音留给重音吸收，不当成公式。"""
    stripped = text.strip()
    if not stripped or all(unicodedata.category(c) in ("Sk", "Mn") for c in stripped):
        return False
    return all(_is_formula_char(c) for c in stripped)


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
# '1 Introduction'、'3.1 Encoder' 是章节标题，不能当成跨页续段
_SRC_HEAD_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")


def _different_columns(
    geometries: list[PageGeometry] | None, left: TextBlock, right: TextBlock,
) -> bool:
    """两页里只要有一页是多栏、且两块不在同一栏序号，就不是同一栏的跨页续段。

    右栏页尾接到下一页左栏是另一条规则，不走这里。单栏页面序号都是 0。
    """
    if not geometries or left.page >= len(geometries) or right.page >= len(geometries):
        return False
    a, b = geometries[left.page], geometries[right.page]
    if len(a.columns) < 2 and len(b.columns) < 2:
        return False
    return a.index_of(left.rect) != b.index_of(right.rect)


def _caption_label(text: str) -> bool:
    return bool(_CAPTION_LABEL_RE.match(text.strip()))


def _cjk_page_continue(prev: TextBlock, nxt: TextBlock) -> bool:
    """跨页中文续段没有大小写。短标签、缩进的新段、图注标签不接。"""
    head = nxt.text.lstrip()[:1]
    if not head or CJK_RE.search(head) is None or _caption_label(nxt.text):
        return False
    if len(CJK_RE.findall(prev.text)) < 4:
        return False
    return abs(nxt.rect.x0 - prev.rect.x0) <= prev.size * 1.2


def _can_continue(
    prev: TextBlock, nxt: TextBlock, median: float, halt_ids: set[int] | None,
) -> bool:
    """页末块没有句末标点，页首块以小写、数字或中文开头，且不是标题。"""
    head = nxt.text.lstrip()[:1]
    numbered_heading = bool(_SRC_HEAD_NUM_RE.match(nxt.text.lstrip()))
    if halt_ids is None:
        is_heading = _preview_kind(nxt, median) != "p"
    else:
        is_heading = id(nxt) in halt_ids
    continues = head.islower() or head.isdigit() or _cjk_page_continue(prev, nxt)
    return bool(not numbered_heading and not is_heading and continues
                and not prev.text.rstrip().endswith(_SENT_END_PUNCT)
                and nxt.size <= prev.size * 1.4)


def _column_index(geo: PageGeometry, block: TextBlock) -> int:
    return geo.index_of(block.rect)


def _in_column(geo: PageGeometry, block: TextBlock) -> bool:
    return len(geo.columns) < 2 or not geo.spans_columns(block.rect)


def _column_page_end(
    body: list[TextBlock], geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    """同栏跨页看的页末。多栏页上的跨栏块不占这个位置，否则会抢走右栏末行。"""
    return _column_edge(body, geometries, last=True)


def _column_page_start(
    body: list[TextBlock], geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    return _column_edge(body, geometries, last=False)


def _column_edge(
    body: list[TextBlock], geometries: list[PageGeometry] | None, last: bool,
) -> TextBlock | None:
    if not body:
        return None
    if not geometries:
        return body[-1] if last else body[0]
    ordered = reversed(body) if last else body
    for block in ordered:
        if block.page >= len(geometries) or _in_column(geometries[block.page], block):
            return block
    return None


def _is_last_in_right_column(
    block: TextBlock, page_blocks: list[TextBlock], geo: PageGeometry,
) -> bool:
    if len(geo.columns) < 2 or not _in_column(geo, block):
        return False
    if _column_index(geo, block) != len(geo.columns) - 1:
        return False
    return not any(
        other is not block and _in_column(geo, other)
        and _column_index(geo, other) == len(geo.columns) - 1
        and other.rect.y0 > block.rect.y0 + 1
        for other in page_blocks
    )


def _left_column_head(page_blocks: list[TextBlock], geo: PageGeometry) -> TextBlock | None:
    if len(geo.columns) < 2:
        return None
    heads = [
        block for block in page_blocks
        if _in_column(geo, block) and _column_index(geo, block) == 0
        and not any(
            other is not block and _in_column(geo, other) and _column_index(geo, other) == 0
            and other.rect.y0 < block.rect.y0 - 1
            for other in page_blocks
        )
    ]
    if not heads:
        return None
    return min(heads, key=lambda block: (block.rect.y0, block.rect.x0))


def _cross_page_units(
    blocks: list[TextBlock],
    median: float,
    skip_ids: set[int] = frozenset(),
    geometries: list[PageGeometry] | None = None,
    halt_ids: set[int] | None = None,
    ignore_ids: set[int] | None = None,
) -> list[list[TextBlock]]:
    """把跨页续段两两合并成翻译单元（返回块列表的列表，每单元 1~2 块）。

    上一页最后一个正文块不以句末标点结尾、下一页第一个块以小写/数字开头且不像标题时，
    视为被页边界切断的同一段。页脚脚注（字号明显小于正文）和指定跳过的块不参与。
    同一栏的续段照旧合并。多栏时，只再放开「本页最右栏的末块接到下页最左栏的首块」。
    ignore_ids 是页眉页脚，不占页末、也不挡页首。跨栏标题和整表同样不占栏首栏尾。
    """
    by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    ignored = ignore_ids or set()

    units: list[list[TextBlock]] = []
    pages = sorted(by_page)
    used: set[int] = set()
    for pno in pages:
        body = [b for b in by_page[pno] if b.size >= median * 0.8 and id(b) not in ignored]
        nxt = by_page.get(pno + 1) or []
        nxt_body = [x for x in nxt if x.size >= median * 0.8 and id(x) not in ignored]
        first = _column_page_start(nxt_body, geometries)
        end = _column_page_end(body, geometries)
        head = (
            _left_column_head(nxt_body, geometries[pno + 1])
            if geometries and pno + 1 < len(geometries) else None
        )
        for b in by_page[pno]:
            if id(b) in used:
                continue
            if halt_ids is not None and id(b) in halt_ids:
                units.append([b])
                used.add(id(b))
                continue
            same = bool(
                end is not None and b is end and first is not None and id(first) not in used
                and id(b) not in skip_ids and id(first) not in skip_ids
                and _can_continue(b, first, median, halt_ids)
                and not _different_columns(geometries, b, first)
            )
            cross_target = head if head is not None and id(head) not in used else None
            cross = bool(
                not same and cross_target is not None and geometries is not None
                and pno < len(geometries)
                and _is_last_in_right_column(b, body, geometries[pno])
                and id(b) not in skip_ids and id(cross_target) not in skip_ids
                and _can_continue(b, cross_target, median, halt_ids)
            )
            other = first if same else cross_target
            if (same or cross) and other is not None:
                units.append([b, other])
                used.add(id(b))
                used.add(id(other))
            else:
                units.append([b])
                used.add(id(b))
    return units


# 公式哨兵。硬切不能落在哨兵内部，否则两页都匹配不上。
_SENTINEL_NUM_RE = re.compile(r"\x01i\x02(\d+)\x01/i\x02")


def _snap_cut(text: str, cut: int) -> int:
    cut = max(0, min(len(text), cut))
    for pattern in (_SENTINEL_NUM_RE, STYLE_MARK_RE):
        for m in pattern.finditer(text):
            if m.start() < cut < m.end():
                return m.start() if cut - m.start() <= m.end() - cut else m.end()
    return cut


def _split_translation(t: str, ratio: float) -> tuple[str, str]:
    """把跨页合并段的译文按比例切回两页：优先在比例附近的句读点断开，
    没有合适标点时按空格，再不行硬切。切点避开公式哨兵，且两侧样式标记配对。"""
    target = len(t) * ratio
    best = None
    for m in re.finditer(r"[。！？；，、：.!?;,:]", t):
        if best is None or abs(m.end() - target) < abs(best - target):
            best = m.end()
    if best is not None and abs(best - target) <= len(t) * 0.3:
        cut = _snap_cut(t, best)
        return _balance_marks(t[:cut], t[cut:])
    # 拉丁文本按空格切
    spaces = [m.start() for m in re.finditer(r"\s", t)]
    if spaces:
        cut = _snap_cut(t, min(spaces, key=lambda s: abs(s - target)))
        return _balance_marks(t[:cut], t[cut:])
    cut = _snap_cut(t, round(target))
    return _balance_marks(t[:cut], t[cut:])


def _balance_marks(left: str, right: str) -> tuple[str, str]:
    """切开处补上仍开着的样式标记。字号不嵌套，必须重开还开着的那一个，不能重开更早已经闭合的。"""
    open_stack: list[str] = []
    for matched in STYLE_MARK_RE.finditer(left):
        token = matched.group(0)
        if token.startswith("{/"):
            kind = token[2:-1]
            for index in range(len(open_stack) - 1, -1, -1):
                opened = open_stack[index]
                if (kind == "z" and opened.startswith("{z")) or opened == "{" + kind + "}":
                    del open_stack[index]
                    break
            continue
        open_stack.append(token)
    if not open_stack:
        return left, right
    closes = ["{/z}" if token.startswith("{z") else "{/" + token[1:] for token in reversed(open_stack)]
    return left + "".join(closes), "".join(open_stack) + right


def _blocks_share_column(
    left: TextBlock, right: TextBlock, geometries: dict[int, PageGeometry] | None,
) -> bool:
    """间隙小于栏沟时，仍要确认两块落在同一栏。没有栏信息时只靠间隙本身。"""
    if not geometries or left.page not in geometries:
        return True
    geo = geometries[left.page]
    if len(geo.columns) < 2:
        return True
    return geo.index_of(left.rect) == geo.index_of(right.rect) and not geo.spans_columns(left.rect | right.rect)


def _merge_visual_lines(
    blocks: list[TextBlock], geometries: dict[int, PageGeometry] | None = None,
) -> list[TextBlock]:
    """合并其实是同一可视行的相邻块（字体在公式处切换时，PyMuPDF 会把一行拆成两块）。

    判定：纵向交叠超过较矮块的一半，并且横向相交，或间隙小于 6pt。栏沟至少 12pt，
    这个间隙接不上另一栏。不合并的话，两块译文会写进互相交叠的矩形里叠在一起。
    """
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            v_overlap = min(p.rect.y1, b.rect.y1) - max(p.rect.y0, b.rect.y0)
            h_overlap = min(p.rect.x1, b.rect.x1) - max(p.rect.x0, b.rect.x0)
            gap = b.rect.x0 - p.rect.x1
            same_line = v_overlap > min(p.rect.y1 - p.rect.y0, b.rect.y1 - b.rect.y0) * 0.5
            near = h_overlap > 0 or (0 <= gap < 6 and _blocks_share_column(p, b, geometries))
            if p.page == b.page and p.clip == b.clip and same_line and near:
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
    """行内公式：数学字体的 span，或明显小于正文的角标 span（引用编号等）。
    粗体 CM（CMBX 等）的一两个字符按 mathbf 单字母算公式；粗体单词仍是正文。
    正文字体里的希腊字母和数学符号也是公式。单个拉丁字母不是，除非贴在角标上。"""
    if _font_is_math(span["font"]) or span["size"] < main_size * 0.79:
        return True
    if re.match(r"^CMB", span["font"]) and len(span["text"].strip()) <= 2:
        return True
    return _is_symbol_span(span["text"])


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


# 孤立标点不当公式：<EOS> 这类 token 的尖括号是数学字体，单独抽出来会把 token 拆散、
# 还把上下行的括号粘成跨行怪图。括号/标点留在正文里排版（正文字体本来就画得出）。
_LONE_PUNCT_RE = re.compile(r"^[\s<>[\]{}()|/\\=+\-_.,;:'\"~·，。；：！？（）【】]+$")
# 贴在角标上的正体变量。两个字母会把 of/to 吸进上标，只收一个字母。
_VARIABLE_BASE_RE = re.compile(r"^[A-Za-z]['′]?$")
# 图注标签可以没有冒号：图 2、Fig. 1.
_CAPTION_LABEL_RE = re.compile(
    r"^(?:(?:Figure|Fig\.?|Table|Tab\.?|Scheme|Plate)\s*\d+|(?:图|表)\s*\d+)\s*[:.。：]?\s*$",
    re.I,
)


def _script_base(span: dict, bbox: pymupdf.Rect, block_size: float) -> bool:
    """单字母贴在较小角标上，才并进公式。引用上标在字母右侧，中心出了字框，不收。"""
    if not _VARIABLE_BASE_RE.match(str(span.get("text") or "").strip()):
        return False
    if float(span.get("size") or 0) < block_size * 0.79:
        return False
    raw = span.get("bbox")
    if raw is None:
        return False
    base = pymupdf.Rect(raw)
    y_gap = max(base.y0, bbox.y0) - min(base.y1, bbox.y1)
    center = (bbox.x0 + bbox.x1) / 2
    # 角标可以略伸出字母右缘。再往右就是引用上标，中心已经离开字框。
    return y_gap < block_size and bbox.x0 <= base.x1 + 0.4 and base.x0 - 1 <= center <= base.x1 + 2.0


def _absorb_script_bases(runs: list[dict], span_lines: list[list[dict]], block_size: float) -> None:
    """把贴在角标上的正体变量并进该角标。单词和引用编号留在正文。"""
    taken = {id(span) for run in runs for span in run["spans"]}
    for run in runs:
        if float(run["max_size"]) >= block_size * 0.79:
            continue
        line_no, start, end = run["pos"]
        candidates: list[tuple[int, int, dict]] = []
        if start > 0:
            candidates.append((line_no, start - 1, span_lines[line_no][start - 1]))
        if line_no > 0:
            for index, span in enumerate(span_lines[line_no - 1]):
                candidates.append((line_no - 1, index, span))
        for other_no, index, span in candidates:
            if id(span) in taken or not _script_base(span, run["bbox"], block_size):
                continue
            run["spans"].insert(0, span)
            run["bbox"] |= pymupdf.Rect(span["bbox"])
            if other_no == line_no:
                run["pos"] = (line_no, index, end)
            taken.add(id(span))
            break


# 数学罗马体（CMR）在黑名单里，不当公式字体。贴着公式的数字、括号、算子名仍是式子的一部分。
_CM_ROMAN_RE = re.compile(r"CMR\d")
_OPERATOR_NAMES = (
    "arg|max|min|log|ln|sin|cos|tan|cot|sec|csc|exp|lim|sup|inf|det|dim|"
    "ker|gcd|mod|sgn|deg|hom|Pr|arcsin|sinh|cosh"
)
_GLUE_ATOM = rf"(?:[\d.()\[\]{{}}|]+|(?:{_OPERATOR_NAMES}))"
_MATH_GLUE_RE = re.compile(rf"^(?:{_GLUE_ATOM})(?:\s+{_GLUE_ATOM})*$")
_ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}


def _is_math_glue(span: dict) -> bool:
    """CMR 上的数字、括号或算子名。正文字体里的同形字不收。"""
    if not _CM_ROMAN_RE.search(str(span.get("font") or "")):
        return False
    return bool(_MATH_GLUE_RE.match(str(span.get("text") or "").strip()))


def _attach_span(run: dict, span: dict, line_no: int, index: int) -> None:
    run["spans"].append(span)
    run["bbox"] |= pymupdf.Rect(span["bbox"])
    run["max_size"] = max(float(run["max_size"]), float(span.get("size") or 0))
    cur_line, start, end = run["pos"]
    if line_no == cur_line:
        run["pos"] = (cur_line, min(start, index), max(end, index))


def _gap_is_clear(
    span: dict, run: dict, span_lines: list[list[dict]], taken: set[int],
) -> bool:
    """两框之间不能隔着还没收进公式的字。空白可以。"""
    box = pymupdf.Rect(span["bbox"])
    other = run["bbox"]
    if box.x0 <= other.x0:
        left, right = box.x1, other.x0
    else:
        left, right = other.x1, box.x0
    if right - left <= 0.4:
        return True
    for line in span_lines:
        for item in line:
            if id(item) in taken or id(item) == id(span) or not str(item.get("text") or "").strip():
                continue
            item_box = pymupdf.Rect(item["bbox"])
            if item_box.x1 <= left + 0.2 or item_box.x0 >= right - 0.2:
                continue
            y_overlap = min(box.y1, item_box.y1) - max(box.y0, item_box.y0)
            if y_overlap > 0:
                return False
    return True


def _glue_host(
    span: dict, runs: list[dict], span_lines: list[list[dict]],
    block_size: float, taken: set[int],
) -> dict | None:
    box = pymupdf.Rect(span["bbox"])
    best: tuple[float, dict] | None = None
    for run in runs:
        other = run["bbox"]
        y_overlap = min(box.y1, other.y1) - max(box.y0, other.y0)
        if y_overlap <= 0.3 * min(box.height, other.height):
            continue
        gap = box.x0 - other.x1 if box.x0 >= other.x0 else other.x0 - box.x1
        if gap < 0:
            gap = 0.0
        if gap >= block_size * 0.8 or not _gap_is_clear(span, run, span_lines, taken):
            continue
        if best is None or gap < best[0]:
            best = (gap, run)
    return best[1] if best else None


def _blanks_between(
    span: dict, run: dict, span_lines: list[list[dict]], taken: set[int],
) -> list[tuple[int, int, dict]]:
    """公式和这段胶水之间的空白。收进公式后，送翻文本里不再留一个空档。"""
    box = pymupdf.Rect(span["bbox"])
    other = run["bbox"]
    if box.x0 <= other.x0:
        left, right = box.x1, other.x0
    else:
        left, right = other.x1, box.x0
    if right - left <= 0.4:
        return []
    found: list[tuple[int, int, dict]] = []
    for line_no, line in enumerate(span_lines):
        for index, item in enumerate(line):
            if id(item) in taken or str(item.get("text") or "").strip():
                continue
            item_box = pymupdf.Rect(item["bbox"])
            if item_box.x0 < left - 0.4 or item_box.x1 > right + 0.4:
                continue
            y_overlap = min(box.y1, item_box.y1) - max(box.y0, item_box.y0)
            if y_overlap > 0:
                found.append((line_no, index, item))
    return found


def _absorb_math_glue(
    runs: list[dict], span_lines: list[list[dict]], block_size: float,
) -> None:
    """把贴着公式的 CMR 数字、括号、算子名并进公式。远处的数字仍是正文。"""
    taken = {id(span) for run in runs for span in run["spans"]}
    pending = [
        (line_no, index, span)
        for line_no, line in enumerate(span_lines)
        for index, span in enumerate(line)
        if id(span) not in taken and _is_math_glue(span)
    ]
    while pending:
        attached = False
        still: list[tuple[int, int, dict]] = []
        for line_no, index, span in pending:
            host = _glue_host(span, runs, span_lines, block_size, taken)
            if host is None:
                still.append((line_no, index, span))
                continue
            for blank_line, blank_index, blank in _blanks_between(span, host, span_lines, taken):
                _attach_span(host, blank, blank_line, blank_index)
                taken.add(id(blank))
            _attach_span(host, span, line_no, index)
            taken.add(id(span))
            attached = True
        if not attached:
            return
        pending = still


def _ordinal_reading(run: dict, block_size: float) -> str | None:
    """i^{th}、1^{st} 这类序数。上标不是 st/nd/rd/th 的角标不是序数。"""
    base: list[str] = []
    suffix: list[str] = []
    for span in run["spans"]:
        text = str(span.get("text") or "").strip()
        if not text:
            continue
        if float(span.get("size") or 0) < block_size * 0.79:
            suffix.append(text)
        else:
            base.append(text)
    tail = "".join(suffix).lower()
    head = "".join(base)
    if tail not in _ORDINAL_SUFFIXES or not re.fullmatch(r"[A-Za-z]|\d{1,3}", head):
        return None
    return f"{head}-{tail}"


def _release_ordinals(runs: list[dict], block_size: float) -> tuple[dict[int, str], set[int]]:
    """序数不做成公式图，改成 i-th 送翻，模型才能译成「第 i 个」。"""
    kept: list[dict] = []
    ordinal_at: dict[int, str] = {}
    ordinal_skip: set[int] = set()
    for run in runs:
        reading = _ordinal_reading(run, block_size)
        if reading is None:
            kept.append(run)
            continue
        anchor = next(
            (span for span in run["spans"] if str(span.get("text") or "").strip()
             and float(span.get("size") or 0) >= block_size * 0.79),
            None,
        )
        if anchor is None:
            kept.append(run)
            continue
        ordinal_at[id(anchor)] = reading
        for span in run["spans"]:
            if span is not anchor and str(span.get("text") or "").strip():
                ordinal_skip.add(id(span))
    runs[:] = kept
    return ordinal_at, ordinal_skip


# 重音符号（hat/tilde/bar/dot 等）：落在公式 bbox 上的要并进公式，不能留在正文
_ACCENT_CHARS = {"^", "ˆ", "̂", "~", "̃", "¯", "̄", "´", "`", "˙", "̇", "¨", "̈", "ˇ", "̌", "⃗"}


def _placeholderize(block: TextBlock, block_index: int = 0) -> tuple[str, list[dict]]:
    """把块里的行内公式换成 {vN} 占位符（BabelDOC 的翻译协议），返回 (送翻文本, 公式记录列表)。

    每条公式记录：bbox（原页裁剪区域）、name、w/h/d（落位几何）、text（无图环境回退）。
    渲染时以 show_pdf_page 矢量透传，视觉和原版一致。
    关键归并：上下标常被 PyMuPDF 拆到另一个 dict 行，按行收集的 run 会把一个公式
    拆成互相交叠的两段（画两遍、上标错位），所以先全块收集 run，再把 x 区间交叠的
    归并成一个。没有公式时返回的文本与原文逐字一致，缓存照常命中。
    """
    if not block.span_lines:
        return block.text, []

    # 1) 按 dict 行收集公式 run，并记录在块 span 流里的位置（供阅读序邻接判断）
    runs: list[dict] = []
    for line_no, spans in enumerate(block.span_lines):
        i, n = 0, len(spans)
        while i < n:
            if _is_formula_span(spans[i], block.size):
                run_spans = [spans[i]]
                j = i + 1
                while j < n and (not spans[j]["text"].strip() or _is_formula_span(spans[j], block.size)):
                    run_spans.append(spans[j])
                    j += 1
                while run_spans and not run_spans[-1]["text"].strip():
                    run_spans.pop()
                if run_spans and any(s["text"].strip() for s in run_spans):
                    # 孤立标点 run 不当公式（<EOS> 的尖括号、括号、等号），留在正文里
                    joined = "".join(s["text"] for s in run_spans).strip()
                    if not _LONE_PUNCT_RE.match(joined):
                        bbox = pymupdf.Rect(run_spans[0]["bbox"])
                        for s in run_spans[1:]:
                            bbox |= pymupdf.Rect(s["bbox"])
                        runs.append({"spans": run_spans, "bbox": bbox,
                                     "max_size": max(s["size"] for s in run_spans),
                                     "pos": (line_no, i, j - 1)})
                i = j
            else:
                i += 1
    _absorb_script_bases(runs, block.span_lines, block.size)
    ordinal_at, ordinal_skip = _release_ordinals(runs, block.size)
    _absorb_math_glue(runs, block.span_lines, block.size)

    # 2) 归并 run，两种情形：
    # a) 上下标拆到不同 dict 行：x 交叠够深（≥30% 较窄 run 的宽）且必有一方是小字号
    #    （<0.79 正文）；两个正文字号 run 即使 x 深度交叠也是相邻文本行上碰巧对齐
    #    的两个公式（如行末 (X,Y) 和下一行的 y_1,…,y_T）。
    # b) 阅读序上真正邻接（中间只有空白 span）且 x 间距小于一个字号、纵向相交
    #    （同一可视行）：同一数学表达式被拆成的连续片段（y^i_1, y^i_2, … 的元素），
    #    并回一个公式，否则下标会被占位符间的空格推到离基底很远的位置。
    #    行间隔着正文的不并（(X,Y) vs y_1,…,y_T）。跨行折行的两半不在这里并 bbox。
    member_ids = {id(s) for r in runs for s in r["spans"]}

    def flow_adjacent(a: dict, b: dict) -> bool:
        l1, _, e1 = a["pos"]
        l2, s2, _ = b["pos"]
        for ln in range(l1, l2 + 1):
            line = block.span_lines[ln]
            lo = e1 + 1 if ln == l1 else 0
            hi = s2 if ln == l2 else len(line)
            for s in line[lo:hi]:
                if s["text"].strip() and id(s) not in member_ids:
                    return False
        return True

    runs.sort(key=lambda r: r["pos"])
    merged: list[dict] = []
    for r in runs:
        if merged:
            m = merged[-1]
            x_overlap = min(m["bbox"].x1, r["bbox"].x1) - max(m["bbox"].x0, r["bbox"].x0)
            y_gap = max(m["bbox"].y0, r["bbox"].y0) - min(m["bbox"].y1, r["bbox"].y1)
            min_w = min(m["bbox"].width, r["bbox"].width)
            lo, hi = sorted([m["max_size"], r["max_size"]])
            geometric = (x_overlap > max(1.0, 0.3 * min_w) and y_gap < block.size
                         and lo < hi * 0.79)
            # b) 同一可视行上、阅读序邻接、x 间距小于一个字号。跨行折行的两半
            #    纵向不相交，不在这里并成一个大框。
            y_overlap = min(m["bbox"].y1, r["bbox"].y1) - max(m["bbox"].y0, r["bbox"].y0)
            adjacent = (flow_adjacent(m, r)
                        and r["bbox"].x0 - m["bbox"].x1 < block.size
                        and y_overlap > 0.3 * min(m["bbox"].height, r["bbox"].height))
            if geometric or adjacent:
                m["spans"].extend(r["spans"])
                m["bbox"] |= r["bbox"]
                m["max_size"] = hi
                m_line, m_start, m_end = m["pos"]
                r_line, r_start, r_end = r["pos"]
                if m_line == r_line:
                    m["pos"] = (m_line, min(m_start, r_start), max(m_end, r_end))
                continue
        merged.append(r)

    # 2.5) 吸收落在公式 bbox 上的重音符 span（ŷ 的 ^、x̄ 的 ¯ 等）：
    # 它们常是不被判定为公式的独立 span，留在正文里会在公式图边上多出一个孤符号；
    # 并进 run 后 bbox 上移把符头顶部也包进裁剪，正文里删掉
    for spans in block.span_lines:
        for s in spans:
            if s["text"].strip() not in _ACCENT_CHARS:
                continue
            sb = pymupdf.Rect(s["bbox"])
            cx, cy = (sb.x0 + sb.x1) / 2, (sb.y0 + sb.y1) / 2
            for m in merged:
                if any(s is x for x in m["spans"]):
                    break
                rb = m["bbox"]
                if (rb.x0 - 1 <= cx <= rb.x1 + 1
                        and rb.y0 - block.size * 0.8 <= cy <= rb.y1 + 1):
                    m["spans"].append(s)
                    m["bbox"] |= sb
                    break
    merged = _join_line_wrapped_formulas(merged, block.span_lines, block.size)

    formulas: list[dict] = []
    formula_runs: dict[int, FormulaRun] = {}
    first_span: dict[int, int] = {}  # span id → 公式序号
    span_line = {id(s): li for li, line in enumerate(block.span_lines) for s in line}
    line_x0 = [_line_x0(spans) for spans in block.span_lines]
    for n, m in enumerate(merged, 1):
        bbox = pymupdf.Rect(m["bbox"])
        spans = sorted(m["spans"], key=lambda s: (s["bbox"][0], s["bbox"][1]))
        main_span = max(spans, key=lambda s: s["size"])
        baseline = main_span["origin"][1] if main_span.get("origin") else bbox.y1
        # 孤立角标公式（(1−ε_i)² 的 ²）的 origin 是抬高的：渲染时按它所在
        # 正文行的 baseline 高差抬回去，否则平方落到正文基线上像个普通数字。
        # 高差必须相对本行正文（正文行 origin 的中位数），不能相对全块
        run_lines = {span_line.get(id(s)) for s in m["spans"]}
        line_origins = sorted(s["origin"][1] for li in run_lines if li is not None
                              for s in block.span_lines[li]
                              if s["text"].strip() and s.get("origin") and s["size"] >= block.size * 0.9)
        own_origin = line_origins[len(line_origins) // 2] if line_origins else baseline
        raise_dy = max(0.0, own_origin - baseline)
        anchor = min(m["spans"], key=lambda s: s["bbox"][0])
        anchor_line = span_line.get(id(anchor), 0)
        formula_run = FormulaRun(
            index=n,
            bbox=bbox,
            baseline=baseline,
            dx=round(bbox.x0 - line_x0[anchor_line], 1),
            dy=round(raise_dy, 1),
            descent=round(max(0.0, bbox.y1 - baseline), 1),
            text=_formula_markup(spans, block.size).strip(),
            page=block.page,
            name=f"f{block_index}_{n}.png",
            line=anchor_line,
        )
        formula_runs[n] = formula_run
        record = _formula_record(formula_run, spans)
        extras = m.get("extras") or []
        if extras:
            record["text"] = " ".join(
                piece for piece in [record["text"], *(_formula_markup(extra["spans"], block.size).strip() for extra in extras)]
                if piece
            )
            record["parts"] = [record, *(_part_record(extra, block.page, f"{formula_run.name[:-4]}_p{i}.png") for i, extra in enumerate(extras, 1))]
        formulas.append(record)
        first_span[id(anchor)] = n

    # 3) 按行重建送翻文本，并记下有序 run。公式锚点是一个 FormulaRun，
    # 同行同风格的文字并成一个 TextRun。和底色不同的强调、字号才写标记。
    formula_ids = {id(s) for m in merged for s in m["spans"]}
    formula_ids |= {id(s) for m in merged for extra in m.get("extras") or [] for s in extra["spans"]}
    writer = writer_for(
        [s for line in block.span_lines for s in line if id(s) not in formula_ids],
        block.size,
    )
    runs: list[Run] = []
    out_lines: list[str] = []
    pending: TextRun | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            runs.append(pending)
            pending = None

    for line_no, spans in enumerate(block.span_lines):
        parts: list[str] = []
        for s in spans:
            n = first_span.get(id(s))
            if n is not None:
                flush()
                runs.append(formula_runs[n])
                parts.append(writer.atom(f"{{v{n}}}"))
            elif id(s) in formula_ids or id(s) in ordinal_skip:
                continue  # 已被某个公式吞掉，或是序数的上标
            elif id(s) in ordinal_at:
                reading = ordinal_at[id(s)]
                emphasis = emphasis_of(s.get("font") or "", int(s.get("flags") or 0))
                parts.append(writer.text(reading, emphasis, float(s.get("size") or 0)))
                shown = dict(s)
                shown["text"] = reading
                if pending is not None and not _same_text_style(pending, s, emphasis):
                    flush()
                pending = _extend_text_run(pending, shown, emphasis, line_no)
            else:
                emphasis = emphasis_of(s.get("font") or "", int(s.get("flags") or 0))
                parts.append(writer.text(s["text"], emphasis, float(s.get("size") or 0)))
                if pending is not None and not _same_text_style(pending, s, emphasis):
                    flush()
                pending = _extend_text_run(pending, s, emphasis, line_no)
        flush()
        parts.append(writer.close())
        out_lines.append("".join(parts))
    block.runs = runs
    return _join_lines(out_lines), formulas


def _line_x0(spans: list[dict]) -> float:
    xs = [pymupdf.Rect(s["bbox"]).x0 for s in spans if s.get("bbox") is not None]
    return min(xs) if xs else 0.0


def _span_box(span: dict) -> pymupdf.Rect:
    raw = span.get("bbox")
    return pymupdf.Rect(raw) if raw is not None else pymupdf.Rect(0, 0, 0, 0)


def _same_text_style(pending: TextRun, span: dict, emphasis: Emphasis) -> bool:
    return (
        pending.emphasis == emphasis
        and pending.font == str(span.get("font") or "")
        and abs(pending.size - float(span.get("size") or 0)) < 0.1
    )


def _extend_text_run(pending: TextRun | None, span: dict, emphasis: Emphasis, line: int) -> TextRun:
    """同行、同字体、同强调的相邻 span 并成一个文字 run。"""
    bbox = _span_box(span)
    baseline = span["origin"][1] if span.get("origin") else bbox.y1
    font = str(span.get("font") or "")
    size = float(span.get("size") or 0)
    if pending is not None and _same_text_style(pending, span, emphasis):
        return replace(pending, text=pending.text + str(span.get("text") or ""), bbox=pending.bbox | bbox)
    return TextRun(
        text=str(span.get("text") or ""),
        font=font,
        size=size,
        emphasis=emphasis,
        baseline=baseline,
        bbox=bbox,
        line=line,
    )


def _page_curve_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - 个别页面的矢量表读不出来，当没有曲线
        return []
    rects: list[pymupdf.Rect] = []
    for drawing in drawings:
        raw = drawing.get("rect")
        if raw is None:
            continue
        rect = pymupdf.Rect(raw)
        if not rect.is_empty:
            rects.append(rect)
    return rects


def _absorb_one_box(formula: dict, drawings: list[pymupdf.Rect]) -> None:
    old = pymupdf.Rect(formula["bbox"])
    curves = intersecting_curves(old, drawings)
    formula["curves"] = list(curves)
    if not curves:
        return
    box = expand_box(old, curves)
    baseline = old.y1 - float(formula.get("d") or 0)
    formula["bbox"] = box
    formula["w"] = round(box.width, 1)
    formula["h"] = round(box.height, 1)
    formula["d"] = round(max(0.0, box.y1 - baseline), 1)
    formula["dx"] = round(float(formula.get("dx") or 0) + (box.x0 - old.x0), 1)


def _absorb_curves(formula: dict, drawings: list[pymupdf.Rect], block: TextBlock) -> None:
    """把相交曲线并进公式框，并写回对应的 FormulaRun。换行续片各自收自己的框。"""
    _absorb_one_box(formula, drawings)
    for part in formula.get("parts") or []:
        if part is not formula:
            _absorb_one_box(part, drawings)
    block.runs = [
        replace(
            run,
            bbox=pymupdf.Rect(formula["bbox"]),
            curves=tuple(formula.get("curves") or ()),
            descent=formula["d"],
            dx=formula.get("dx", run.dx),
        )
        if isinstance(run, FormulaRun) and run.name == formula["name"] else run
        for run in block.runs
    ]


def _line_edge(span_lines: list[list[dict]], line_no: int, side: str) -> float:
    edges: list[float] = []
    for span in span_lines[line_no]:
        raw = span.get("bbox")
        if raw is None or not str(span.get("text") or "").strip():
            continue
        box = pymupdf.Rect(raw)
        edges.append(box.x0 if side == "x0" else box.x1)
    if not edges:
        return 0.0
    return min(edges) if side == "x0" else max(edges)


def _only_this_run(run: dict, span_lines: list[list[dict]], head: bool) -> bool:
    line_no, start, end = run["pos"]
    spans = span_lines[line_no][:start] if head else span_lines[line_no][end + 1:]
    return not any(str(span.get("text") or "").strip() for span in spans)


def _continues_wrapped_formula(
    prev: dict, nxt: dict, span_lines: list[list[dict]], block_size: float,
) -> bool:
    """行末公式接到下一行行首，是同一式子被换行切开。中间不能还有正文。"""
    last = (prev.get("extras") or [prev])[-1]
    last_line, _, _ = last["pos"]
    next_line, _, _ = nxt["pos"]
    if next_line != last_line + 1:
        return False
    if not _only_this_run(last, span_lines, head=False) or not _only_this_run(nxt, span_lines, head=True):
        return False
    if _line_edge(span_lines, last_line, "x1") - last["bbox"].x1 > block_size * 1.5:
        return False
    if nxt["bbox"].x0 - _line_edge(span_lines, next_line, "x0") > block_size * 1.2:
        return False
    y_gap = nxt["bbox"].y0 - last["bbox"].y1
    return -block_size * 0.5 <= y_gap < block_size * 1.8


def _join_line_wrapped_formulas(
    runs: list[dict], span_lines: list[list[dict]], block_size: float,
) -> list[dict]:
    """换行切开的公式收成一个占位符的多片。不并 bbox，否则裁进两行之间的整栏空白。"""
    if len(runs) < 2:
        return runs
    joined: list[dict] = []
    for run in runs:
        if joined and _continues_wrapped_formula(joined[-1], run, span_lines, block_size):
            joined[-1].setdefault("extras", []).append(run)
            continue
        joined.append(run)
    return joined


def _part_record(run: dict, page: int, name: str) -> dict:
    bbox = pymupdf.Rect(run["bbox"])
    spans = run["spans"]
    main = max(spans, key=lambda span: span["size"])
    baseline = main["origin"][1] if main.get("origin") else bbox.y1
    return {
        "page": page,
        "bbox": bbox,
        "name": name,
        "w": round(bbox.width, 1),
        "h": round(bbox.height, 1),
        "d": round(max(0.0, bbox.y1 - baseline), 1),
        "raise": 0.0,
        "spans": spans,
        "curves": [],
        "text": _formula_markup(spans, float(main["size"])).strip(),
    }


def _formula_parts(formula: dict) -> list[dict]:
    return formula.get("parts") or [formula]


def _formula_record(run: FormulaRun, spans: list[dict]) -> dict:
    """渲染层仍读这条记录。框和曲线随后可能变，run 再同步回去。"""
    return {
        "page": run.page,
        "bbox": pymupdf.Rect(run.bbox),
        "name": run.name,
        "w": round(run.bbox.width, 1),
        "h": round(run.bbox.height, 1),
        "d": run.descent,
        "raise": run.dy,
        "dx": run.dx,
        "text": run.text,
        "spans": spans,
        "curves": [],
        "owner": run.owner,
    }


# 译文里游离的 {vN} 占位符（模型把占位符写进了别的段）
_STRAY_PLACEHOLDER_RE = re.compile(r"\{\s*v\s*\d+\s*\}")


def _restore_placeholders(translation: str, formulas: list[dict]) -> str | None:
    """译后恢复 {vN}（容忍模型在占位符里塞的空格）；有占位符丢失时返回 None（回退原文）。
    恢复出来的是图片哨兵（\\x01i\\x02N\\x01/i\\x02），渲染层换成对应公式的 <img>。
    译文里多出来的幻觉占位符直接删掉。"""
    for n, _f in enumerate(formulas, 1):
        pattern = re.compile(rf"\{{\s*v\s*{n}\s*\}}")
        if not pattern.search(translation):
            return None
        translation = pattern.sub(lambda _: f"\x01i\x02{n}\x01/i\x02", translation, count=1)
    return re.sub(r"\{\s*v\s*\d+\s*\}", "", translation)


def _prose_units(text: str) -> int:
    """正文词数：拉丁单词（≥2 字母）+ CJK 字符数 / 2。用来区分"公式行"和"含公式的句子"。"""
    latin = len(re.findall(r"[A-Za-z]{2,}", text))
    cjk = len(CJK_RE.findall(text))
    return latin + cjk // 2


def _donate_punct_rects(prev: TextBlock, line_rects: list[pymupdf.Rect]) -> None:
    """标点碎片贴在上一块某一行的行尾时，把矩形并进那一行去遮罩。

    间隙相对该行的 x1，不用整块并集：并集会把右侧公式擦掉，而公式行不会重画。
    """
    if not line_rects or not prev.line_rects:
        return
    fragment = line_rects[0]
    height = fragment.y1 - fragment.y0
    if height <= 0:
        return
    for tail in prev.line_rects:
        overlap = min(tail.y1, fragment.y1) - max(tail.y0, fragment.y0)
        if overlap > 0.5 * height and fragment.x0 - tail.x1 < 2:
            prev.line_rects.extend(line_rects)
            return


def _clip_key(cell: pymupdf.Rect | None) -> tuple[float, float, float, float] | None:
    if cell is None:
        return None
    return (cell.x0, cell.y0, cell.x1, cell.y1)


def _split_table_blocks(blocks: list[TextBlock], tables: tuple[RuledTable, ...]) -> list[TextBlock]:
    """一行里并排的格子拆成独立块。小写开头的邻格不会被硬换行规则切开。"""
    if not tables:
        return blocks
    out: list[TextBlock] = []
    for block in blocks:
        out.extend(_split_one_table_block(block, tables))
    return out


def _split_one_table_block(block: TextBlock, tables: tuple[RuledTable, ...]) -> list[TextBlock]:
    lines = block.span_lines or []
    if len(lines) != len(block.line_rects):
        block.clip = _clip_key(cell_containing(block.rect, tables))
        return [block]
    grouped: dict[tuple[float, float, float, float] | None, list[tuple[pymupdf.Rect, list[dict]]]] = {}
    order: list[tuple[float, float, float, float] | None] = []
    for line_rect, spans in zip(block.line_rects, lines):
        by_cell: dict[tuple[float, float, float, float] | None, list[dict]] = {}
        cell_order: list[tuple[float, float, float, float] | None] = []
        for span in spans:
            if not str(span.get("text", "")).strip():
                continue
            bbox = span.get("bbox")
            key = _clip_key(cell_containing(pymupdf.Rect(bbox), tables) if bbox else None)
            if key not in by_cell:
                cell_order.append(key)
                by_cell[key] = []
            by_cell[key].append(span)
        if not cell_order:
            key = _clip_key(cell_containing(line_rect, tables))
            _remember_cell(grouped, order, key, line_rect, [])
            continue
        if len(cell_order) == 1:
            _remember_cell(grouped, order, cell_order[0], line_rect, by_cell[cell_order[0]])
            continue
        for key in cell_order:
            box = pymupdf.Rect()
            for span in by_cell[key]:
                box |= pymupdf.Rect(span["bbox"])
            _remember_cell(grouped, order, key, box, by_cell[key])
    if len(order) <= 1:
        block.clip = order[0] if order else _clip_key(cell_containing(block.rect, tables))
        return [block]
    parts: list[TextBlock] = []
    for key in order:
        pieces = grouped[key]
        rect = pymupdf.Rect()
        for line_rect, _spans in pieces:
            rect |= line_rect
        text = _join_lines(["".join(str(span.get("text", "")) for span in spans) for _rect, spans in pieces])
        if not text.strip():
            continue
        parts.append(replace(
            block,
            rect=rect,
            line_rects=[line_rect for line_rect, _spans in pieces],
            text=text,
            span_lines=[spans for _rect, spans in pieces],
            clip=key,
            runs=[],
        ))
    return parts or [block]


def _remember_cell(
    grouped: dict[tuple[float, float, float, float] | None, list[tuple[pymupdf.Rect, list[dict]]]],
    order: list[tuple[float, float, float, float] | None],
    key: tuple[float, float, float, float] | None,
    line_rect: pymupdf.Rect,
    spans: list[dict],
) -> None:
    if key not in grouped:
        order.append(key)
        grouped[key] = []
    grouped[key].append((line_rect, spans))


def extract_blocks(doc: pymupdf.Document) -> list[TextBlock]:
    blocks = []
    tables_by_page: list[tuple[RuledTable, ...]] = []
    for pno, page in enumerate(doc):
        tables_by_page.append(page_tables(page))
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
                # 只跳过"纯公式行"：数学字符多且几乎没有正文词。
                # 含公式的正常句子（正文词 ≥4）照翻，公式走占位符，不再整句留成英文
                is_equation = _math_chars(spans) > total * 0.3 and _prose_units(text) < 4
                if NO_TEXT_RE.match(text) or is_equation or letters < 2 or letters < len(text) * 0.4:
                    # 只捐标点碎片。公式行和低字母行不送翻，矩形也不能捐出去：
                    # 捐了会被 redact，而这些行没有占位符、不会重画。
                    if NO_TEXT_RE.match(text) and blocks and blocks[-1].page == pno:
                        _donate_punct_rects(blocks[-1], line_rects)
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
    split: list[TextBlock] = []
    for block in blocks:
        tables = tables_by_page[block.page] if block.page < len(tables_by_page) else ()
        split.extend(_split_table_blocks([block], tables))
    # 栏沟明确时先排成阅读顺序。同页续段只看相邻块，另一栏插在中间就接不上本栏的下一行。
    ordered: list[TextBlock] = []
    by_page: dict[int, list[TextBlock]] = {}
    geometries: dict[int, PageGeometry] = {}
    for block in split:
        by_page.setdefault(block.page, []).append(block)
    for pno in sorted(by_page):
        page_blocks = by_page[pno]
        geo = PageGeometry.from_page(doc[pno], [block.rect for block in page_blocks])
        geometries[pno] = geo
        ordered.extend(reading_order(page_blocks, geo))
    return _merge_continuations(_merge_caption_fragments(_merge_visual_lines(ordered, geometries)))


def _absorb_block(p: TextBlock, b: TextBlock):
    p.text = _join_lines([p.text, b.text])
    p.rect |= b.rect
    p.line_rects.extend(b.line_rects)
    if p.span_lines and b.span_lines:
        p.span_lines.extend(b.span_lines)
    else:
        p.span_lines = None
    if len(b.text) > len(p.text) - len(b.text):  # 风格跟随占多的片段
        p.size, p.color, p.bold = b.size, b.color, b.bold


def _cjk_wrap(prev: TextBlock, nxt: TextBlock) -> bool:
    """中文换行不分大小写，也不按词数计。新段有约 2em 缩进，短标签宽度不够。"""
    head = nxt.text.lstrip()[:1]
    if not head or CJK_RE.search(head) is None:
        return False
    if prev.text.rstrip().endswith(tuple(SENT_ENDS)):
        return False
    if abs(nxt.size - prev.size) > prev.size * 0.2:
        return False
    if abs(nxt.rect.x0 - prev.rect.x0) > prev.size * 0.6:
        return False
    return len(CJK_RE.findall(prev.text)) >= 8 or prev.rect.width >= prev.size * 12


def _merge_continuations(blocks: list[TextBlock]) -> list[TextBlock]:
    """合并同页内被公式/碎片拆断的段落续行：上行无句末标点、下行小写（或闭括号）
    开头、同左 margin、行距正常，就是同一段被拆开的两行。各翻各的会产出半截译文，
    渲染时在断点强制换行（'（如 | 序列）'）。中文换行左缘对齐、上行够长时同样接上。
    英文大写开头的换行仍分开，那和下一句分不开。"""
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            first = b.text[:1]
            # 大字号标题碎片（'…with' + 'Recurrent Neural Networks'）：换行的第二行
            # 往往大写开头、也不带句读。仅当两行左边界明显漂移（居中对齐的标题行）
            # 才并：左对齐的大字号块（小标题、署名行）各自独立
            title_frag = (p.size >= 12 and b.size >= p.size * 0.7
                          and 0 <= b.rect.y0 - p.rect.y1 < 1.5 * p.size
                          and abs(b.rect.x0 - p.rect.x0) > p.size * 0.5
                          and not p.text.rstrip().endswith(tuple(SENT_ENDS))
                          and not b.text.rstrip().endswith(tuple(SENT_ENDS)))
            y_close = -0.8 * p.size <= b.rect.y0 - p.rect.y1 < 1.5 * p.size
            latin = (
                y_close
                and abs(b.rect.x0 - p.rect.x0) < 2 * p.size
                and len(p.text.split()) >= 3  # 一两个词的无标点短块是标题/标签，不是段落
                and not p.text.rstrip().endswith(tuple(SENT_ENDS))
                and (first.islower() or first in "),;%,；，")
            )
            if p.page == b.page and p.clip == b.clip and (title_frag or latin or (y_close and _cjk_wrap(p, b))):
                _absorb_block(p, b)
                continue
        out.append(b)
    return out


def _merge_caption_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    """合并被拆碎的图注/表注：'Figure 2:'、'Examples of decay'、'schedules.' 三个
    碎片各翻各的，再塞回各自小框里换行丑陋。拼回一个块后整句翻译、一行排下。
    规则保守，只拼三类：a) 同一视觉行上短标签后紧跟的片段，标签可以是冒号，
    也可以是「图 2」「Fig. 1.」；b) 这种标签正下方的一行注记；c) 带冒号的
    图注组内、以左对齐小写开头且上行无句末标点的短续行。表格行和标题不并。"""
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            if p.page == b.page and p.clip == b.clip:
                y_ov = min(p.rect.y1, b.rect.y1) - max(p.rect.y0, b.rect.y0)
                same_line = y_ov > 0.5 * min(p.rect.y1 - p.rect.y0, b.rect.y1 - b.rect.y0)
                gap = b.rect.x0 - p.rect.x1
                below = (0 <= b.rect.y0 - p.rect.y1 < 1.2 * p.size
                         and abs(b.rect.x0 - p.rect.x0) < p.size)
                label = (p.text.rstrip().endswith((":", "：")) and len(p.text) < 30) or _caption_label(p.text)
                head = b.text.lstrip()[:1]
                note = bool(head and (head.islower() or CJK_RE.search(head)) and not _caption_label(b.text))
                continuation = (len(p.text) < 60 and b.text[:1].islower()
                                and (":" in p.text or "：" in p.text)
                                and not p.text.rstrip().endswith(tuple(SENT_ENDS)))
                if (same_line and -1 <= gap < 3 * p.size and label) or (below and (continuation or (label and note))):
                    _absorb_block(p, b)
                    continue
        out.append(b)
    return out


def _marker_png(idx: int) -> bytes:
    """1x1 彩色像素：颜色编码公式的全局序号（R + G<<8），供插入后找回落位。
    必须全局唯一：同色的标记图会被 MuPDF 去重成同一 xref，落位全串到第一处。"""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1))
    pix.set_pixel(0, 0, ((idx + 1) & 0xFF, ((idx + 1) >> 8) & 0xFF, 200))
    return pix.tobytes("png")


def _white_pixmap(doc) -> "pymupdf.Pixmap":
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1))
    pix.set_pixel(0, 0, (255, 255, 255))
    return pix


def _page_form(doc: "pymupdf.Document", page: "pymupdf.Page") -> int:
    """把页面内容打包成 Form XObject（在 redact 之前调用），返回 xref。
    供公式透传引用：资源沿用页面自己的，BBox 为整页。"""
    mb = page.rect
    fm = doc.get_new_xref()
    doc.update_object(fm, (
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 {mb.width} {mb.height}] "
        f"/Matrix [1 0 0 1 0 0] /Resources {doc.xref_get_key(page.xref, 'Resources')[1]} >>"
    ))
    doc.update_stream(fm, page.read_contents())
    return fm


def _formula_form(doc: "pymupdf.Document", page: "pymupdf.Page", fmS: int,
                  f: dict, f_name: str, page_spans: list[dict] | None = None) -> int:
    """给单个公式造紧致 BBox 的包裹 Form（按 span 逐个裁剪后引用源页 Form），返回 xref。

    裁剪路径是 run 里每个 span 的矩形并集，不是整个外框矩形：外框矩形会把间隙里
    的邻字笔画（上一行 p 的降部等）也裁进来，在公式上方留下横杠。
    BBox 取并集外框，选择/复制时不会框出一大片。

    裁剪用 even-odd：run 的 span 矩形按侧扩展后取并集（斜体笔画会越出 span 的
    排版 bbox——Y 的右臂能越出 1.2pt，扩展以不碰到邻居为限），再减去外来 span
    的矩形孔（上一行降部等），既不削自己的字，也不沾别人的。
    """
    H = page.rect.height
    bbox = f["bbox"]
    clip_outer = pymupdf.Rect(bbox.x0, H - bbox.y1, bbox.x1, H - bbox.y0)  # PDF 坐标 y 向上
    fmT = doc.get_new_xref()
    doc.update_object(fmT, (
        f"<< /Type /XObject /Subtype /Form /BBox [{clip_outer.x0:.2f} {clip_outer.y0:.2f} "
        f"{clip_outer.x1:.2f} {clip_outer.y1:.2f}] "
        f"/Matrix [1 0 0 1 0 0] /Resources << /XObject << /FmS {fmS} 0 R >> >> >>"
    ))
    # run 的 span 来自 extract_blocks 的另一次 get_text 调用，和 page_spans 是不同
    # 对象，只能按 bbox 相等判断是不是 run 成员，否则兄弟 span（本公式的上标）会被
    # 当成外来 span 被挖洞挖掉
    curve_rects = [pymupdf.Rect(r) for r in f.get("curves") or []]
    run_rects = [pymupdf.Rect(s["bbox"]) for s in f["spans"]] + curve_rects

    def merge_rects(rects: list["pymupdf.Rect"]) -> list["pymupdf.Rect"]:
        out: list[pymupdf.Rect] = []
        for b in rects:
            for i, mb in enumerate(out):
                if not (b & mb).is_empty:
                    out[i] = mb | b
                    break
            else:
                out.append(b)
        return out

    def is_run_member(fr: "pymupdf.Rect") -> bool:
        return any(abs(fr.x0 - rr.x0) < 0.5 and abs(fr.y0 - rr.y0) < 0.5
                   and abs(fr.x1 - rr.x1) < 0.5 and abs(fr.y1 - rr.y1) < 0.5
                   for rr in run_rects)

    # 先并成不相交集合（ŷ 的符头 bbox 和 y 几乎完全重叠，不并掉 even-odd 会把
    # y 的躯干整个关掉），再按侧扩展：每侧扩到 1.5pt 为止、以不贴到邻居 bbox 为限
    base = merge_rects([pymupdf.Rect(s["bbox"]) for s in f["spans"]] + curve_rects)
    union = pymupdf.Rect()
    for rr in base:
        union |= rr
    lim = {"l": 1.5, "r": 1.5, "t": 1.5, "b": 1.5}  # 每侧扩展上限（pt）
    if page_spans:
        for fb in page_spans:
            if not fb["text"].strip():  # 空白 span 没有墨迹，不限扩展也不挖洞
                continue
            fr = pymupdf.Rect(fb["bbox"])
            if is_run_member(fr) or not (fr.y0 < union.y1 and fr.y1 > union.y0):
                continue
            if fr.x1 <= union.x0 + 0.01:      # 左侧邻居
                lim["l"] = max(0.2, min(lim["l"], union.x0 - fr.x1 - 0.3))
            elif fr.x0 >= union.x1 - 0.01:    # 右侧邻居
                lim["r"] = max(0.2, min(lim["r"], fr.x0 - union.x1 - 0.3))
            if fr.y1 <= union.y0 + 0.01:      # 上方邻居（pymupdf y 向下）
                lim["t"] = max(0.2, min(lim["t"], union.y0 - fr.y1 - 0.3))
            elif fr.y0 >= union.y1 - 0.01:    # 下方邻居
                lim["b"] = max(0.2, min(lim["b"], fr.y0 - union.y1 - 0.3))
    our_rects = merge_rects([pymupdf.Rect(b.x0 - lim["l"], b.y0 - lim["t"],
                                          b.x1 + lim["r"], b.y1 + lim["b"]) for b in base])
    ours, holes = [], []
    for b in our_rects:
        r = pymupdf.Rect(b.x0, H - b.y1, b.x1, H - b.y0)
        if r.width > 0.2 and r.height > 0.2:
            ours.append(f"{r.x0:.2f} {r.y0:.2f} {r.width:.2f} {r.height:.2f} re")
    if page_spans and our_rects:
        hole_rects = []
        for fb in page_spans:
            if not fb["text"].strip():  # 空白 span 没有墨迹，限扩展和挖洞都不参与
                continue
            fr = pymupdf.Rect(fb["bbox"])
            if any(abs(fr.x0 - rr.x0) < 0.5 and abs(fr.y0 - rr.y0) < 0.5
                   and abs(fr.x1 - rr.x1) < 0.5 and abs(fr.y1 - rr.y1) < 0.5
                   for rr in run_rects):
                continue
            # 孔 = 外来 span 与我们的每个矩形的交。必须完全落在我们的矩形内：
            # even-odd 裁剪下越出并集的孔会反开，把正要排除的外来笔画整个画出来。
            # 只挖纵向不共线的 span（降部、邻公式的角标等）：同行邻字的 bbox 常常
            # 仅仅贴边且纵向几乎等长，挖掉会把本公式的笔画（y 的左干）削了。
            # 孔不内收：降部墨迹和 span bbox 底边之间本就有缝隙，内收会留下一道残影
            for rr in our_rects:
                y_ov = min(fr.y1, rr.y1) - max(fr.y0, rr.y0)
                if y_ov > 0.7 * (rr.y1 - rr.y0):
                    continue
                hb = fr & rr
                if not hb.is_empty and hb.width >= 0.3 and hb.height >= 0.3:
                    hole_rects.append(hb)
        # 相交的孔合并成一个：两个孔相交处在 even-odd 下会反开
        merged: list[pymupdf.Rect] = []
        for hb in hole_rects:
            for i, mb in enumerate(merged):
                if not (hb & mb).is_empty:
                    merged[i] = mb | hb
                    break
            else:
                merged.append(hb)
        for hb in merged:
            hr = pymupdf.Rect(hb.x0, H - hb.y1, hb.x1, H - hb.y0)
            holes.append(f"{hr.x0:.2f} {hr.y0:.2f} {hr.width:.2f} {hr.height:.2f} re")
    doc.update_stream(fmT, f"q {' '.join(ours + holes)} W* n /FmS Do Q".encode())
    res_v = doc.xref_get_key(page.xref, "Resources")[1]
    # redact 后 /Resources 会变成直接字典（键要带 Resources/ 前缀）；
    # 间接引用则在资源对象上直接设键
    m = re.match(r"(\d+) \d+ R", res_v)
    res_xref, base = (int(m.group(1)), "") if m else (page.xref, "Resources/")
    xo_v = doc.xref_get_key(res_xref, f"{base}XObject")[1]
    if xo_v == "null":
        doc.xref_set_key(res_xref, f"{base}XObject", "<<>>")
        xo_v = "<<"
    m = re.match(r"(\d+) \d+ R", xo_v)
    if m:
        doc.xref_set_key(int(m.group(1)), f_name, f"{fmT} 0 R")
    else:
        doc.xref_set_key(res_xref, f"{base}XObject/{f_name}", f"{fmT} 0 R")
    return fmT


def _place_formula_images(page: "pymupdf.Page", doc: "pymupdf.Document", before: set[int],
                          mid_map: dict[int, dict], fmS: int | None = None,
                          page_spans: list[dict] | None = None):
    """把占位标记换成公式内容并校准基线。

    htmlbox 对 <img> 只做图片底部对齐，公式的 baseline 在图片内部（上下标在 baseline
    之下还有延伸），直接插会整体浮起来。所以先插一个按公式宽高占位的 1x1 标记像素，
    从落位矩形读出文本基线位置，再把公式按“bbox 底 − 公式 baseline”下移贴到精确位置。
    有源页 Form（fmS）时用紧致 BBox 的包裹 Form 做矢量透传（原 glyph、可选中、选择
    不爆框）；失败回退到 PNG 位图。

    标记像素按全局序号（mid）编码，整页一次处理；所有公式绘制集中在最后一条新内容
    流里：insert_htmlbox 每块新建内容流，标记涂白后白块仍在块的内容流里，公式若画在
    之前的流里会被后续块的白块盖住上半截。
    """
    draws = []
    # 渲染后的文本 span：htmlbox 对 img 只做底部对齐，且标记底边和文本 baseline 有
    # ~1pt 系统偏差，公式 baseline 直接锚到同一行文字的 origin 上最稳
    text_spans = [s for blk in page.get_text("dict")["blocks"] if blk.get("type") == 0
                  for ln in blk["lines"] for s in ln["spans"]
                  if s["text"].strip() and s.get("origin")]
    now = {img[0] for img in page.get_images(full=True)}
    for xref in now - before:
        pix = pymupdf.Pixmap(doc, xref)
        p = pix.pixel(0, 0)
        if len(p) < 3 or p[2] != 200:
            continue
        f = mid_map.get(p[0] + (p[1] << 8) - 1)
        if f is None:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        rs = f.get("rs", 1.0)
        # raise：孤立角标公式（²）的 origin 比正文 baseline 高，落位时按高差抬回去
        w, h = f["w"] * rs, f["h"] * rs
        d = (f["d"] - f.get("raise", 0.0)) * rs
        cy = (rect.y0 + rect.y1) / 2
        baseline = None
        best_key = (0.0, float("inf"))
        for s in text_spans:
            sb = s["bbox"]
            if not (sb[1] - 1 <= cy <= sb[3] + 1):  # 不在同一行
                continue
            # CJK line-height 放大后相邻行的 glyph bbox 会交叠：按与标记矩形的
            # y 交叠量取最大（同一行交叠最多），并列再取 x 最近
            overlap = min(sb[3], rect.y1) - max(sb[1], rect.y0)
            dx = max(0.0, sb[0] - rect.x1, rect.x0 - sb[2])
            if (overlap, -dx) > (best_key[0], -best_key[1]):
                best_key, baseline = (overlap, dx), s["origin"][1]
        if baseline is not None:
            target = pymupdf.Rect(rect.x0 + 2.0, baseline + d - h, rect.x0 + 2.0 + w, baseline + d)
        else:
            target = pymupdf.Rect(rect.x0 + 2.0, rect.y0 + d, rect.x0 + 2.0 + w, rect.y0 + d + h)
        page.replace_image(xref, pixmap=_white_pixmap(doc))
        _paint_formula(page, doc, f, target, fmS, page_spans, draws)
    if draws:
        ns = doc.get_new_xref()
        doc.update_object(ns, "<<>>")
        doc.update_stream(ns, ("\n" + "\n".join(draws) + "\n").encode())
        cont_v = doc.xref_get_key(page.xref, "Contents")[1].strip()
        if cont_v.startswith("["):
            doc.xref_set_key(page.xref, "Contents", f"{cont_v[:-1].rstrip()} {ns} 0 R]")
        else:
            doc.xref_set_key(page.xref, "Contents", f"[{cont_v} {ns} 0 R]")


def _paint_formula(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    formula: dict,
    target: "pymupdf.Rect",
    fmS: int | None,
    page_spans: list[dict] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]] | None = None,
) -> None:
    """把公式画进 target。矢量透传失败时回退位图。绘制命令先攒着，最后一条内容流再写。"""
    bbox = formula.get("bbox")
    if fmS is not None and bbox is not None and bbox.width > 0.5 and bbox.height > 0.5 and target.width > 0.5:
        try:
            # 资源名必须全页唯一：块内序号相同的 FmF0 会互相覆盖
            f_name = "FmF_" + re.sub(r"[^A-Za-z0-9_]", "_", str(formula["name"]))
            _formula_form(doc, page, fmS, formula, f_name, page_spans)
            sx = target.width / bbox.width
            sy = target.height / bbox.height
            # 带缩放的仿射：平移量必须吸收缩放，否则公式整体往下漂 (1-sy)*H
            tx = target.x0 - sx * bbox.x0
            ty = (page.rect.height - target.y1) - sy * (page.rect.height - bbox.y1)
            draws.append(f"q {sx:.4f} 0 0 {sy:.4f} {tx:.2f} {ty:.2f} cm /{f_name} Do Q")
            return
        except Exception:  # noqa: BLE001 - 矢量透传失败回退位图
            pass
    png = formula.get("png")
    if not png:
        return
    if deferred_png is not None:
        deferred_png.append((target, png))
        return
    page.insert_image(target, stream=png, keep_proportion=False, overlay=True)


def _formula_metrics(formulas: list[dict], block_size: float, em: float) -> dict[int, FormulaMetric]:
    """按当前字号给出每个公式槽。rs 写回记录，矢量缩放和槽位用同一个数。"""
    metrics: dict[int, FormulaMetric] = {}
    for index, formula in enumerate(formulas, 1):
        scale, slot_width, height = _formula_slot(formula, block_size, em)
        formula["rs"] = scale
        body = float(formula["w"]) * scale if formula.get("has_img") else slot_width
        below = (float(formula.get("d") or 0) - float(formula.get("raise") or 0)) * scale
        metrics[index] = FormulaMetric(slot_width, body, height, below)
    return metrics


def _fit_typeset(
    text: str,
    formulas: list[dict],
    rect: "pymupdf.Rect",
    em: float,
    block_size: float,
    cjk: bool,
    page: "pymupdf.Page",
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None,
) -> tuple[list[TypesetLine], float, float, "pymupdf.Rect"]:
    """放得下就按原字号排。放不下先在本栏右扩，再压行距，最后才缩字号。"""
    gap = 1.45 if cjk else 1.2
    box = rect
    expanded = False
    tightened = False
    lines: list[TypesetLine] = []
    for _ in range(24):
        lines = typeset_lines(text, em, max(box.width, 1.0), _formula_metrics(formulas, block_size, em))
        # 首行字形顶部还有 0.3em 的出头（实测 MuPDF 盒子内容高 = 行数×行距 + ~0.25em），
        # 只算行距会让临界盒子在写入时被静默缩到 0.93 倍
        height = max(len(lines), 1) * em * gap + em * 0.3
        overflows = any(line.width > box.width + 1 for line in lines)
        too_tall = height > box.height + 1
        if not overflows and not too_tall:
            return lines, em, gap, box
        if not expanded and box.width < page.rect.width * 0.6:
            expanded = True
            wide = _expand_right(box, page.rect.width, obstacles, right_limit)
            if wide.x1 > box.x1 + 1:
                box = wide
                continue
        if not tightened:
            gap = 1.1 if gap <= 1.25 else gap - 0.25
            tightened = True
            continue
        if em <= 1.5:
            break
        em = max(1.5, em * 0.9)
    return lines, em, gap, box


def _line_baseline(
    spans: list[dict], line_top: float, line_h: float, em: float, cjk: bool, slot_x: float | None = None,
) -> float:
    """这一行已写入文字的 origin。纯公式行没有文字，用 htmlbox 的经验基线。
    相邻行的字框会交叠，离槽位更近的文字优先，避免公式掉到下一行仍留在原横坐标。"""
    best: tuple[tuple[float, float], float] | None = None
    for span in spans:
        origin = span.get("origin")
        bbox = span.get("bbox")
        if not origin or not bbox:
            continue
        if bbox[3] < line_top - 1 or bbox[1] > line_top + line_h + 1:
            continue
        overlap = min(bbox[3], line_top + line_h) - max(bbox[1], line_top)
        if overlap <= 0:
            continue
        dx = 0.0
        if slot_x is not None:
            if bbox[2] < slot_x:
                dx = slot_x - bbox[2]
            elif bbox[0] > slot_x:
                dx = bbox[0] - slot_x
        near = overlap if dx <= em * 4 else overlap * 0.2
        key = (near, -dx)
        if best is None or key > best[0]:
            best = (key, float(origin[1]))
    if best is not None:
        return best[1]
    return line_top + em * (1.12 if cjk else 1.0)


def _paint_formula_parts(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    formula: dict,
    slot_x: float,
    baseline: float,
    fmS: int | None,
    page_spans: list[dict] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]],
) -> None:
    """一个占位符的几片按阅读顺序并排贴。换行切开的后半段不再留在源文横坐标。"""
    scale = float(formula.get("rs") or 1)
    x = slot_x + 2.0
    for part in _formula_parts(formula):
        width = float(part["w"]) * scale
        height = float(part["h"]) * scale
        below = (float(part.get("d") or 0) - float(part.get("raise") or 0)) * scale
        target = pymupdf.Rect(x, baseline + below - height, x + width, baseline + below)
        _paint_formula(page, doc, part, target, fmS, page_spans, draws, deferred_png)
        x += width + 1.5


def _typeset_line_html(line: TypesetLine, formulas: list[dict]) -> str:
    """一行的 HTML。没有图的公式退回记录里的文字。"""
    parts: list[str] = []
    for piece in line.pieces:
        if isinstance(piece, TextPiece):
            if piece.text:
                parts.append(_inline_marks(restore_emphasis(html.escape(piece.text))))
        elif isinstance(piece, FormulaPiece) and 1 <= piece.index <= len(formulas):
            formula = formulas[piece.index - 1]
            if not formula.get("has_img"):
                parts.append(_inline_marks(html.escape(str(formula.get("text") or ""))))
    return "".join(parts)


def _formula_draw_args(formulas: list[dict]) -> tuple[frozenset[int], dict[int, str]]:
    images = frozenset(index for index, formula in enumerate(formulas, 1) if formula.get("has_img"))
    fallback = {
        index: str(formula.get("text") or "")
        for index, formula in enumerate(formulas, 1)
        if index not in images
    }
    return images, fallback


def _draw_cjk(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    lines: list[TypesetLine],
    formulas: list[dict],
    box: "pymupdf.Rect",
    em: float,
    gap: float,
    color: str,
    centered: bool,
    fmS: int | None,
    page_spans: list[dict] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]],
) -> bool:
    """中文行按字形两端对齐。写不成时返回 False。"""
    images, fallback = _formula_draw_args(formulas)
    slots = draw_cjk_lines(
        page, doc, lines, box, em, gap, color, centered, images, fallback,
    )
    if slots is None:
        return False
    for slot in slots:
        if not 1 <= slot.index <= len(formulas):
            continue
        _paint_formula_parts(
            page, doc, formulas[slot.index - 1], slot.x, slot.baseline,
            fmS, page_spans, draws, deferred_png,
        )
    return True


def _place_plain(
    page: "pymupdf.Page",
    text: str,
    formulas: list[dict],
    rect: "pymupdf.Rect",
    em: float,
    block_size: float,
    cjk: bool,
    centered: bool,
    div_attrs: str,
    color: str,
    weight: str,
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None,
    archive: pymupdf.Archive | None,
) -> None:
    """排版器断行，再写入。中文按字形两端对齐；缺字或带样式标记时退回一个 HTML 盒子。"""
    text = separate_after_formula(text)
    lines, em, gap, box = _fit_typeset(
        text, formulas, rect, em, block_size, cjk, page, obstacles, right_limit,
    )
    if cjk and _draw_cjk(
        page, page.parent, lines, formulas, box, em, gap, color, centered,
        None, None, [], [],
    ):
        return
    rendered = [_typeset_line_html(line, formulas) for line in lines]
    parts = [
        f'<span style="white-space:nowrap">{line_html}</span>'
        for line_html in rendered
        if line_html
    ]
    body = "<br>".join(parts)
    if not body:
        return
    align = "center" if centered else "left"
    css = (
        f"* {{font-family: sans-serif; font-size: {em}px; color: {color}; "
        f"font-weight: {weight}; line-height: {gap}; margin: 0; padding: 0; text-align: {align};}}"
    )
    html_text = f"<div{div_attrs}>{body}</div>"
    if _try_insert(page, box, html_text, css, 0.9, archive):
        return
    _insert_fitting(page, box, html_text, css, obstacles, archive, right_limit)


def _place_typeset(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    block: TextBlock,
    text: str,
    formulas: list[dict],
    rect: "pymupdf.Rect",
    em: float,
    cjk: bool,
    centered: bool,
    div_attrs: str,
    color: str,
    weight: str,
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None,
    fmS: int | None,
    page_spans: list[dict] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]],
) -> None:
    """按行写入文字，公式槽用记录的下沿贴到该行基线。"""
    text = separate_after_formula(text)
    lines, em, gap, box = _fit_typeset(
        text, formulas, rect, em, block.size, cjk, page, obstacles, right_limit,
    )
    if cjk and _draw_cjk(
        page, doc, lines, formulas, box, em, gap, color, centered,
        fmS, page_spans, draws, deferred_png,
    ):
        return
    css = (
        f"* {{font-family: sans-serif; font-size: {em}px; color: {color}; "
        f"font-weight: {weight}; line-height: {gap}; margin: 0; padding: 0; text-align: left;}}"
    )
    line_h = em * gap
    pending: list[tuple[FormulaPiece, float, float]] = []
    y = box.y0
    for line in lines:
        origin_x = box.x0
        if centered and line.width < box.width:
            origin_x += (box.width - line.width) / 2
        for piece in line.pieces:
            if isinstance(piece, TextPiece):
                if not piece.text.strip():
                    continue
                # 公式后的分界空格在单独的盒子开头会被吃掉，改成同样宽的偏移
                body, had_space = peel_leading_space(piece.text)
                lead = em * 0.33 if had_space else 0.0
                fragment = _inline_marks(restore_emphasis(html.escape(body)))
                html_text = f'<div{div_attrs}><span style="white-space:nowrap">{fragment}</span></div>'
                # 单行盒子也要留首行出头的余量（MuPDF 一行内容高 = 行距 + ~0.25em），
                # 否则每个小片都被静默缩到 0.93 倍
                slot = pymupdf.Rect(origin_x + piece.x + lead, y,
                                    origin_x + piece.x + piece.width + 4,
                                    y + line_h + em * 0.3)
                if not _try_insert(page, slot, html_text, css, 0.9):
                    try:
                        page.insert_htmlbox(slot, html_text, css=css, scale_low=0)
                    except AssertionError:
                        pass
            elif isinstance(piece, FormulaPiece) and 1 <= piece.index <= len(formulas):
                pending.append((piece, origin_x + piece.x, y))
        y += line_h
    if not pending:
        return
    spans = [
        span
        for blk in page.get_text("dict")["blocks"] if blk.get("type") == 0
        for ln in blk["lines"] for span in ln["spans"]
        if span.get("text", "").strip() and span.get("origin")
    ]
    for piece, slot_x, line_top in pending:
        baseline = _line_baseline(spans, line_top, line_h, em, cjk, slot_x)
        _paint_formula_parts(
            page, doc, formulas[piece.index - 1], slot_x, baseline, fmS, page_spans, draws, deferred_png,
        )


def _flush_draws(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]] | None = None,
) -> None:
    for target, png in deferred_png or []:
        page.insert_image(target, stream=png, keep_proportion=False, overlay=True)
    if deferred_png is not None:
        deferred_png.clear()
    if not draws:
        return
    ns = doc.get_new_xref()
    doc.update_object(ns, "<<>>")
    doc.update_stream(ns, ("\n" + "\n".join(draws) + "\n").encode())
    cont_v = doc.xref_get_key(page.xref, "Contents")[1].strip()
    if cont_v.startswith("["):
        doc.xref_set_key(page.xref, "Contents", f"{cont_v[:-1].rstrip()} {ns} 0 R]")
    else:
        doc.xref_set_key(page.xref, "Contents", f"[{cont_v} {ns} 0 R]")
    draws.clear()


def _nowrap_parens(body: str) -> str:
    """短括号组包 nowrap span：MuPDF 的 CJK 换行可以在任意处断，会出现
    '（如\\n序列）' 这种断在括号中间的丑行，以及 '（y_1^{t-1}' 把右括号单独
    留在下一行。整组不拆（含公式图的组，括号跟着公式一起走），长组不包（防溢出）。"""
    def repl(m):
        inner = m.group(0)
        # 估算渲染宽度：<img> 按 3 个字符计，其余标签不计
        plain = re.sub(r"<[^>]+>", lambda t: "@@@" if t.group(0).startswith("<img") else "", inner)
        if len(plain) <= 10:
            return f'<span style="white-space: nowrap">{inner}</span>'
        return inner
    return re.sub(r"（[^（）]+）|\([^()]+\)", repl, body)


def _formula_slot(formula: dict, block_size: float, render_size: float) -> tuple[float, float, float]:
    """行内公式槽位 (缩放, 宽, 高)。高超过 1.35em 时压到 1.35em，避免行框被图片撑开。
    换行切开的几片按阅读顺序横排，宽度相加，不把两行空白裁进同一个框。"""
    parts = _formula_parts(formula)
    height = max(float(part.get("h") or 0) for part in parts)
    if not formula.get("has_img") or height <= 0 or block_size <= 0:
        text = str(formula.get("text") or "")
        return 1.0, len(text) * render_size * 0.5, render_size
    scale = min(render_size / block_size, 1.35 * render_size / height)
    gaps = 1.5 * (len(parts) - 1)
    width = sum(float(part["w"]) * scale for part in parts) + gaps + 4.0
    return scale, width, height * scale


def _inline_marks(body: str) -> str:
    return (
        body.replace("\x01s\x02", "<sup>")
        .replace("\x01/s\x02", "</sup>")
        .replace("\x01b\x02", "<sub>")
        .replace("\x01/b\x02", "</sub>")
    )


def _translation_html(
    text: str,
    formulas: list[dict],
    block_size: float,
    render_size: float,
    max_width: float,
) -> str:
    """译文 HTML。含公式哨兵时自己断行，每行 nowrap；否则仍交给 MuPDF 折行。"""

    def img_repl(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if not 1 <= number <= len(formulas):
            return ""
        formula = formulas[number - 1]
        if formula.get("has_img"):
            scale, width, height = _formula_slot(formula, block_size, render_size)
            formula["rs"] = scale
            # 槽位两侧各留 2pt：斜体字形会微微越出 bbox，紧贴排布时像被前后汉字压住
            return (
                f'<img src="ph_{formula["name"]}" '
                f'style="width: {width:.1f}px; height: {height:.1f}px;">'
            )
        return _inline_marks(html.escape(str(formula.get("text") or "")))

    def one(fragment: str) -> str:
        escaped = _inline_marks(restore_emphasis(html.escape(fragment)))
        return re.sub(r"\x01i\x02(\d+)\x01/i\x02", img_repl, escaped)

    if "\x01i\x02" not in text:
        return _nowrap_parens(one(text).replace("\n", "<br>"))
    widths = {
        index: _formula_slot(formula, block_size, render_size)[1]
        for index, formula in enumerate(formulas, 1)
    }
    return nowrap_lines([one(line) for line in break_lines(text, render_size, max_width, widths)])


def _centered_in_cell(block: TextBlock) -> bool:
    """原文在格子里居中。栏中线对不上格子中线，不能拿栏来判断。"""
    clip = block.clip
    if clip is None:
        return False
    cell_mid = (clip[0] + clip[2]) / 2
    text_mid = (block.rect.x0 + block.rect.x1) / 2
    return abs(text_mid - cell_mid) <= 2.0


def _write_rect(
    block: TextBlock,
    items: list[tuple[TextBlock, str]],
    cjk: bool,
    rtl: bool,
    geo: PageGeometry | None,
) -> tuple[pymupdf.Rect, bool]:
    """写入框。多栏且块落在某一栏内时，居中参照该栏；单栏仍用本页最宽块。
    格子里原本居中的文字，改以格子中线居中。"""
    y0 = block.rect.y0 + block.size * 0.1 if cjk else block.rect.y0
    pad = block.size * 0.5 if cjk else block.size * 0.3
    if _centered_in_cell(block) and block.clip is not None:
        clip = block.clip
        rect = pymupdf.Rect(clip[0], y0, clip[2], block.rect.y1 + pad)
        return _clip_write_rect(rect, clip), True
    column = None
    if geo is not None and len(geo.columns) >= 2:
        candidate = geo.column_of(block.rect)
        if block.rect.x0 >= candidate.x0 - 2 and block.rect.x1 <= candidate.x1 + 2:
            column = candidate
    if column is not None:
        col_x0, col_x1 = column.x0, column.x1
        col_width = col_x1 - col_x0
    else:
        widest = max((other.rect for other, _ in items), key=lambda rect: rect.width, default=block.rect)
        col_x0, col_x1 = widest.x0, widest.x1
        col_width = widest.width
    centered = (
        not rtl and len(block.text) < 100 and block.rect.width < col_width * 0.85
        and abs((block.rect.x0 + block.rect.x1) / 2 - (col_x0 + col_x1) / 2) < 15
    )
    if centered:
        rect = pymupdf.Rect(col_x0, y0, col_x1, block.rect.y1 + pad)
    else:
        rect = pymupdf.Rect(block.rect.x0, y0, block.rect.x1 + 2, block.rect.y1 + pad)
    return _clip_write_rect(rect, block.clip), centered


def _clip_write_rect(
    rect: pymupdf.Rect, clip: tuple[float, float, float, float] | None,
) -> pymupdf.Rect:
    """格子左右是硬边界。上下保留行框高度。

    贴着格子下沿的一行，如果把下沿余量裁掉，行框会矮过一行。排版器误以为
    高度不够，把这一格的字号单独缩小，同一张表里就会大小不齐。
    """
    if clip is None:
        return rect
    bounded = pymupdf.Rect(
        max(rect.x0, clip[0] + 0.6),
        rect.y0,
        min(rect.x1, clip[2] - 0.8),
        rect.y1,
    )
    if bounded.width < 4 or bounded.height < 4:
        return rect
    return bounded


def _write_limit(
    block: TextBlock, rect: pymupdf.Rect, geo: PageGeometry | None, obstacles: list[pymupdf.Rect],
) -> float | None:
    limit = geo.right_limit(rect, obstacles) if geo is not None else None
    if block.clip is not None:
        cap = block.clip[2] - 0.8
        limit = cap if limit is None else min(limit, cap)
    return limit


def _render_translated(src_path: Path, blocks: list[TextBlock], translations: list[str],
                       target_lang: str = "", formulas_map: dict | None = None,
                       archive=None, orig: "pymupdf.Document | None" = None,
                       geometries: list[PageGeometry] | None = None) -> pymupdf.Document:
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
        mid_map = {f["mid"]: f for b, _ in items for f in (formulas_map or {}).get(id(b), [])
                   if "mid" in f}
        # redact 之前把整页打包成 Form：公式透传从它里面裁剪原 glyph；
        # 同时收集页面全部 span（检测上标区域被上一行降部侵入的污染）
        page_spans = [s for blk in page.get_text("dict")["blocks"] if blk.get("type") == 0
                      for ln in blk["lines"] for s in ln["spans"]] \
            if mid_map else None
        fmS = _page_form(doc, page) if page_spans else None
        for b, _ in items:
            for r in b.line_rects:
                page.add_redact_annot(r, fill=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        )
        # redaction 会把 /Resources 改写成内联字典；排版器写入后 /XObject 条目上百个，
        # 这么大的内联嵌套字典在保存的 GC（garbage≥3）里会被写坏（第一个表单的条目
        # 指向字体流，内容静默丢失）。规范成间接引用避开这个 MuPDF bug。
        res_v = doc.xref_get_key(page.xref, "Resources")[1]
        if res_v.startswith("<<"):
            res_x = doc.get_new_xref()
            doc.update_object(res_x, res_v)
            doc.xref_set_key(page.xref, "Resources", f"{res_x} 0 R")
        images_before = {img[0] for img in page.get_images(full=True)}
        geo = geometries[pno] if geometries and pno < len(geometries) else None
        draws: list[str] = []
        deferred_png: list[tuple[pymupdf.Rect, bytes]] = []
        for b, t in items:
            t = separate_after_formula(t)
            bformulas = (formulas_map or {}).get(id(b), [])
            weight = "bold" if b.bold else "normal"
            size = b.size * 0.88 if cjk else b.size
            # 栏宽参考：单栏用本页最宽块；多栏用本块所在栏。居中块用整栏宽，
            # 短译文不再被小框挤成孤字行。排版器按这个宽度断行。
            rect, centered = _write_rect(b, items, cjk, rtl, geo)
            obstacles = [ob.rect for ob in blocks if ob.page == pno and ob is not b]
            if geo is not None:
                obstacles = [*obstacles, *geo.figures]
            limit = _write_limit(b, rect, geo, obstacles)
            # 从右往左仍交给 HTML 盒子。其余段落由排版器断行：有公式图时按行落位，
            # 没有公式图时写进一个锁住换行的盒子，避免一行嵌一次整套字体。
            if rtl:
                body = _translation_html(t, bformulas, b.size, size, rect.width)
                align = "right" if not centered else "center"
                css = (
                    f"* {{font-family: sans-serif; font-size: {size}px; color: {b.color}; "
                    f"font-weight: {weight}; line-height: {1.45 if cjk else 1.2}; margin: 0; padding: 0; "
                    f"text-align: {align};}}"
                )
                _insert_fitting(page, rect, f"<div{div_attrs}>{body}</div>", css, obstacles, archive, limit)
            elif any(f.get("has_img") for f in bformulas):
                _place_typeset(
                    page, doc, b, t, bformulas, rect, size, cjk, centered, div_attrs, b.color, weight,
                    obstacles, limit, fmS, page_spans, draws, deferred_png,
                )
            else:
                _place_plain(
                    page, t, bformulas, rect, size, b.size, cjk, centered, div_attrs, b.color, weight,
                    obstacles, limit, archive,
                )
        _flush_draws(page, doc, draws, deferred_png)
        if mid_map:
            _place_formula_images(page, doc, images_before, mid_map, fmS, page_spans)
    return doc


def _expand_right(
    rect: "pymupdf.Rect",
    page_width: float,
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None = None,
) -> "pymupdf.Rect":
    """排版收缩第一级：向右扩，不盖住右侧纵向交叠的文字或图。

    上限是页宽的 90%，有栏缘时再取更小的那个。结果不会小于原来的右缘。
    """
    x1 = page_width * 0.9
    if right_limit is not None:
        x1 = min(x1, right_limit)
    for obstacle in obstacles:
        if obstacle.x0 <= rect.x0 + 1:
            continue
        if obstacle.y0 < rect.y1 and obstacle.y1 > rect.y0:
            x1 = min(x1, obstacle.x0 - 2)
    return pymupdf.Rect(rect.x0, rect.y0, max(x1, rect.x1), rect.y1)


def _try_insert(page: "pymupdf.Page", rect: "pymupdf.Rect", html_text: str, css: str, scale_low: float,
                archive=None) -> bool:
    """insert_htmlbox 放不下返回负值（此时不画内容）。pymupdf 在缩放刚好等于 scale_low 时
    会因浮点误差触发 assert（0.8999999999999999 < 0.9），按没放下来处理。"""
    try:
        return page.insert_htmlbox(rect, html_text, css=css, scale_low=scale_low, archive=archive)[0] >= 0
    except AssertionError:
        return False


def _tighten_line_height(css: str) -> str:
    """压一级行距。CJK 样式是 1.45，写死替换 1.2 时这一步不会生效。"""
    match = re.search(r"line-height:\s*([0-9.]+)", css)
    if match is None:
        return css
    current = float(match.group(1))
    tight = 1.1 if current <= 1.25 else current - 0.25
    return f"{css[:match.start(1)]}{tight:g}{css[match.end(1):]}"


def _insert_fitting(page: "pymupdf.Page", rect: "pymupdf.Rect", html_text: str, css: str,
                    obstacles: list["pymupdf.Rect"], archive=None, right_limit: float | None = None):
    """排版三级收缩（BabelDOC 思路）：先右扩 → 压行距 → 最后才缩字号。

    每一级只在放不下（insert_htmlbox 返回负值，此时不会画出内容）时进入下一级。
    右扩只对明显窄于页宽的块（标签、图注、短行）生效：满栏段落右扩会越过栏边界，
    且扩完不缩字号时 CJK 行框（1.31em）比盒子（按拉丁 1.16em 算）高，会向下溢出
    压到下一行内容。right_limit 是本栏右缘，扩出去也不能越过它。
    """
    if _try_insert(page, rect, html_text, css, 0.9, archive):
        return
    wide = (
        _expand_right(rect, page.rect.width, obstacles, right_limit)
        if rect.width < page.rect.width * 0.6 else rect
    )
    if wide.x1 > rect.x1 + 1 and _try_insert(page, wide, html_text, css, 0.9, archive):
        return
    target = wide
    css_tight = _tighten_line_height(css)
    if _try_insert(page, target, html_text, css_tight, 0.9, archive):
        return
    # 兜底：和原来一样缩到能放下为止
    page.insert_htmlbox(target, html_text, css=css_tight, scale_low=0, archive=archive)


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


_TRANS_HEAD_NUM_RE = re.compile(
    r"^第[0-9零〇一二三四五六七八九十百]+[章节](?:第[0-9零〇一二三四五六七八九十百]+节)?\s*"
    r"|^\d+(?:\.\d+)*\.?[章节]?(?:\s+|$)")


def _retag_runs(runs: list[Run], owner: int, index_offset: int) -> list[Run]:
    """公式序号改成单元内的全局序号，归属记在 run 上。"""
    retagged: list[Run] = []
    for run in runs:
        if isinstance(run, FormulaRun):
            retagged.append(replace(run, owner=owner, index=run.index + index_offset))
        else:
            retagged.append(replace(run, owner=owner))
    return retagged


def _prepare_unit(unit: list[TextBlock], name_start: int) -> tuple[str, list[dict], int]:
    """一个翻译单元的送翻文本和公式记录。公式名按块递增，避免跨页两块撞名。

    两块合成一段时，中间插入 {| }。这是 run 的分界，译完按它切开。
    """
    texts: list[str] = []
    formulas: list[dict] = []
    name_i = name_start
    for ui, block in enumerate(unit):
        sent, found = _placeholderize(block, name_i)
        name_i += 1
        offset = len(formulas)
        if offset:
            sent = re.sub(
                r"\{v(\d+)\}",
                lambda m, off=offset: f"{{v{int(m.group(1)) + off}}}",
                sent,
            )
        for rec in found:
            rec["owner"] = ui
        block.runs = _retag_runs(block.runs, ui, offset)
        formulas.extend(found)
        texts.append(sent)
    if len(texts) == 2:
        return _join_lines([texts[0], BOUNDARY, texts[1]]), formulas, name_i
    return _join_lines(texts), formulas, name_i


def _take_owner(text: str, formulas: list[dict], owner: int) -> tuple[str, list[str]]:
    """留下属于 owner 的公式哨兵，其余原样抽出（保持相对顺序）。"""
    kept: list[str] = []
    foreign: list[str] = []
    pos = 0
    for m in _SENTINEL_NUM_RE.finditer(text):
        kept.append(text[pos:m.start()])
        n = int(m.group(1))
        if 1 <= n <= len(formulas) and formulas[n - 1].get("owner", 0) == owner:
            kept.append(m.group(0))
        elif 1 <= n <= len(formulas):
            foreign.append(m.group(0))
        pos = m.end()
    kept.append(text[pos:])
    return "".join(kept), foreign


def _localize_sentinels(text: str, formulas: list[dict], owner: int) -> tuple[str, list[dict]]:
    """哨兵改成本块公式列表的序号，渲染时按下标取图。"""
    local = [f for f in formulas if f.get("owner", 0) == owner]
    index_map: dict[int, int] = {}
    local_n = 1
    for i, rec in enumerate(formulas, 1):
        if rec.get("owner", 0) == owner:
            index_map[i] = local_n
            local_n += 1

    def repl(m: re.Match[str]) -> str:
        mapped = index_map.get(int(m.group(1)))
        if mapped is None:
            return ""
        return f"\x01i\x02{mapped}\x01/i\x02"

    return _SENTINEL_NUM_RE.sub(repl, text), local


def _assign_restored_parts(
    restored: str, unit: list[TextBlock], formulas: list[dict],
) -> tuple[list[str], list[list[dict]]]:
    """把恢复后的译文按 run 分界分回各块。哨兵仍跟自己的 run 走。

    模型丢掉 {| } 时才退回按原文长度比例切，避免整段堆在一页。
    """
    if len(unit) != 2:
        return [strip_boundary(restored)], [formulas]
    split = split_page_boundary(restored)
    if split is None:
        total = len(unit[0].text) + len(unit[1].text)
        ratio = len(unit[0].text) / total if total else 0.5
        split = _split_translation(restored, ratio)
    left, right = split
    left, to_right = _take_owner(left, formulas, 0)
    right, to_left = _take_owner(right, formulas, 1)
    left_text, left_formulas = _localize_sentinels(left + "".join(to_left), formulas, 0)
    right_text, right_formulas = _localize_sentinels("".join(to_right) + right, formulas, 1)
    return [left_text, right_text], [left_formulas, right_formulas]


def _normalize_heading_number(src_text: str, translation: str) -> str:
    """标题译文开头已有章节号时改回原文编号（'第二章' → '2'）。

    没有章节号就不加前缀。译文与原文相同则原样返回，避免跳过块被重新排版。
    """
    if translation.strip() == src_text.strip():
        return translation
    m = _SRC_HEAD_NUM_RE.match(src_text.strip())
    if not m or len(src_text) > 60 or src_text.rstrip().endswith(tuple(SENT_ENDS)):
        return translation
    if not _TRANS_HEAD_NUM_RE.match(translation.strip()):
        return translation
    t = _TRANS_HEAD_NUM_RE.sub("", translation.strip(), count=1)
    return f"{m.group(1)} {t}" if t else translation


def _heading_number_for(kind: str, src_text: str, translation: str) -> str:
    if kind == "p":
        # 正文尺寸的节标题（'2.1 Model' 与正文同字号，kind 是 p）：带点节号
        # （2.1、3.2.1）几乎不会出现在正文句首；bare 整数开头可能是正文数量
        # （'2 samples'），不动。且译文以章节号开头才改写。
        if not (re.match(r"^\d+(?:\.\d+)+\s", src_text.strip())
                and _TRANS_HEAD_NUM_RE.match(translation.strip())):
            return translation
    return _normalize_heading_number(src_text, translation)


def _strip_boundary_echo(prev_src: str, cur_src: str, prev_t: str, cur_t: str) -> str:
    """去掉模型在段界复读的连接语。

    相同原文会命中同一条缓存，译文必然相同，不能删。无标点的短重叠
    （「如下所示」）是常见短语，也不是断点复读。整段重叠不删。
    """
    if not prev_t or not cur_t or prev_src.strip() == cur_src.strip() or "\x01" in cur_t[:40]:
        return cur_t
    limit = min(40, len(prev_t), len(cur_t) - 1)
    for k in range(limit, 3, -1):
        overlap = cur_t[:k]
        if not prev_t.endswith(overlap) or overlap in prev_src:
            continue
        if k <= 8 and not re.search(r"[，。；：、,.;:）)\]]", overlap):
            continue
        return cur_t[k:]
    return cur_t


def _sync_unit_preview(
    runner: object, units: list[list[TextBlock]], sent_texts: list[str], per_block: dict[int, str],
) -> None:
    """标题改写和段界去重发生在预览写入之后，按段落序号把最终译文补回去。"""
    overrides = getattr(runner, "_index_overrides", None)
    done = getattr(runner, "_done", None)
    if not isinstance(overrides, dict) or not isinstance(done, dict):
        return
    for ui, (unit, sent_text) in enumerate(zip(units, sent_texts)):
        shown = strip_style_marks(_join_lines(per_block.get(id(b), b.text) for b in unit))
        current = done.get(sent_text)
        if current is not None and shown != current:
            overrides[ui] = shown


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
    # 栏和障碍先于角色、也先于跨页合并：图内文字和图注靠这些框来认
    geometries = [
        PageGeometry.from_page(src_doc[i], [b.rect for b in blocks if b.page == i])
        for i in range(src_doc.page_count)
    ]
    roles = HeuristicLayout().classify(
        [LayoutItem(b.page, pymupdf.Rect(b.rect), b.text, b.size, b.bold) for b in blocks],
        geometries,
        median,
    )
    kinds = ["h2" if role is Role.HEADING else "p" for role in roles]
    edges = margin_skips(blocks, geometries)
    skips = [
        bib or role is Role.FIGURE or edge
        for bib, role, edge in zip(biblio_skips(blocks), roles, edges)
    ]
    skip_ids = {id(b) for b, s in zip(blocks, skips) if s}
    margin_ids = {id(b) for b, edge in zip(blocks, edges) if edge}
    halt_ids = {id(b) for b, role in zip(blocks, roles) if role is not Role.BODY} | margin_ids
    units = _cross_page_units(blocks, median, skip_ids, geometries, halt_ids, margin_ids)
    block_kind = {id(b): k for b, k in zip(blocks, kinds)}

    sent_texts: list[str] = []
    sent_formulas: list[list[dict]] = []
    sent_kinds: list[str] = []
    sent_skips: list[bool] = []
    flat_units: list[list[TextBlock]] = []
    name_i = 0
    for unit in units:
        sent, formulas, name_i = _prepare_unit(unit, name_i)
        flat_units.append(unit)
        sent_texts.append(sent)
        sent_formulas.append(formulas)
        sent_kinds.append(block_kind[id(unit[0])])
        sent_skips.append(all(id(b) in skip_ids for b in unit))

    raw = await runner.translate_all(sent_texts, kinds=sent_kinds, preview=True, skip=sent_skips)

    # 公式从它所在的原页裁图。不能用单元最后一块的页码，跨页时会裁错页。
    # 框内相交的曲线并进裁剪，根号、分式线才不会被切掉。
    archive = pymupdf.Archive()
    curve_rects = [_page_curve_rects(page) for page in src_doc]
    for unit, formulas in zip(flat_units, sent_formulas):
        for f in formulas:
            _absorb_curves(f, curve_rects[int(f["page"])], unit[int(f.get("owner") or 0)])
    for formulas in sent_formulas:
        for f in formulas:
            pieces = _formula_parts(f)
            ready = False
            for part in pieces:
                bbox = part["bbox"]
                if bbox.width < 1 or bbox.height < 1:
                    continue
                part["png"] = src_doc[int(part.get("page") or f["page"])].get_pixmap(dpi=300, clip=bbox).tobytes("png")
                part["has_img"] = True
                part["clip"] = bbox + (0.4, 0.4, -0.4, -0.4)
                ready = True
            f["has_img"] = ready
    # 标记像素按全文全局序号编码（同色的会被 MuPDF 去重成同一图片，落位全串）
    marker_idx = 0
    for formulas in sent_formulas:
        for f in formulas:
            if f.get("has_img"):
                f["mid"] = marker_idx
                archive.add(_marker_png(marker_idx), f"ph_{f['name']}")
                marker_idx += 1

    per_block: dict[int, str] = {}
    block_formulas: dict[int, list] = {}
    for unit, t, sent_text, formulas in zip(flat_units, raw, sent_texts, sent_formulas):
        restored = _restore_placeholders(t, formulas) if formulas else _STRAY_PLACEHOLDER_RE.sub("", t)
        if restored is not None:
            restored = separate_after_formula(restored)
        if restored is None:
            parts = [b.text for b in unit]  # 模型弄丢占位符：整单元回退原文，不产出坏文档
            formula_lists: list[list[dict]] = [[] for _ in unit]
            shown = _join_lines(parts)
        else:
            parts, formula_lists = _assign_restored_parts(restored, unit, formulas)
            shown = strip_style_marks(strip_boundary(restored))
        for b, part, flist in zip(unit, parts, formula_lists):
            per_block[id(b)] = _heading_number_for(block_kind[id(b)], b.text, part)
            block_formulas[id(b)] = flist
        # 预览里显示恢复后的译文（缓存仍按占位符版本存，重跑照样命中）
        if formulas and getattr(runner, "_done", None) and runner._done.get(sent_text) == t:
            runner._done[sent_text] = shown
    translations = [per_block[id(b)] for b in blocks]
    for i in range(1, len(blocks)):
        translations[i] = _strip_boundary_echo(
            blocks[i - 1].text, blocks[i].text, translations[i - 1], translations[i],
        )
        per_block[id(blocks[i])] = translations[i]
    _sync_unit_preview(runner, flat_units, sent_texts, per_block)

    def build():
        translated = _render_translated(
            src, blocks, translations, target_lang, block_formulas, archive,
            orig=src_doc, geometries=geometries,
        )
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
