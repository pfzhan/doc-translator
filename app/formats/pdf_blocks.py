"""Text blocks and the helpers that both reading and writing pages use: line joining,
sentence ends, glyph ink and math-font checks."""

import contextvars
import functools
import hashlib
import math
import os
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import pymupdf

from .pdf_flow import (
    STYLE_MARK_RE,
    emphasis_of,
)
from .pdf_layout import (
    PageGeometry,
)
from .pdf_runs import (
    Run,
)

MATH_FONT_RE = re.compile(r"CMMI|CMB?SY|CMEX|MSBM|MSAM|Math|Symbol|StandardSym|STIX|Cambria Math", re.I)
# 精确的数学字体白名单（直接判公式）
# 正文字体配套的数学字体名里带 Math（XCharterMath、CambriaMath），newtx、pxfonts 的数学字体是 txmi、txsy 这样的短名。
# newtxmath 的数学斜体叫 NewTXMI、NewTXMI7，配 Cochineal 时叫 cochMI，数学正体叫 cochMRM。
# Euler 的花体、哥特体（EUSM、EUFM），空心体（bbold、dsfont、bbm）只用来写公式
MATH_FONT_PRECISE_RE = re.compile(
    r"Asana|FiraMath|STIX|TeXGyre.*Math|XITSMath|LibertinusMath|MathJax|Cambria Math|LatinModern.*Math"
    r"|(?-i:Math(?![a-z]))|(?:^|\+)n?[tp]x(?:mi|sy|ex)|(?:^|\+)(?:New[TP]X|coch)B?(?:MI|MRM)\d*$"
    r"|(?:^|\+)(?:EU[SF]M|EURM|EUEX|BBOLD|bbm|dsrom|dsss|DSSerif)",
    re.I,
)
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
    r"Takao|Sazanami|Motoya|Yomogi|BIZ |M PLUS|GenShin|GenRyu|NotoSansCJK|NotoSerifCJK",
    re.I,
)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_PAREN_ITEM_RE = re.compile(r"^\([a-zA-Z]\)\s+")
# 证明分项的 (ii)、(iii) 后面大写起头的是另一项。小写接着写的是句中的列举（(iii) the arm must …）
_ROMAN_ITEM_RE = re.compile(r"^\((?:[ivx]{2,4}|[IVX]{2,4})\)\s+[A-Z]")
# 行首列表标记：符号弹点（可不带空格）、短横线类（须带空格，避免误伤连字符单词）、数字/字母编号
BULLET_RE = re.compile(r"^\s*(?:[•◦▪‣·○■□►»]\s*|[-–—*]\s+|\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+)")
_MARK_ITEM_RE = re.compile(r"^\s*(?:[•◦▪‣·○■□►»]|[-–—*]\s)")
# 不含分号：分号在下面的连接标点里，硬换行不会因此拆段
SENT_ENDS = ("。", "！", "？", "：", "…", ".", "!", "?", ":")
# 句末标点后面收尾的引号、括号（found it.)、he said."）。前面不是句末标点的，
# 像引文年份 (Vilain et al., 1995)、代码 r"cat|dog"，句子还没完
_SENT_CLOSERS = "\"'”’」』)）"


# 段末的证毕号（□、∎，Symbol 字体里取出来是 \x00 或私用区字符）不算正文，句号在它前面。
# \x01、\x02 是公式哨兵的分隔符，不能剥
_END_MARK_RE = re.compile(r"[\s\x00\ue000-\uf8ff□■∎◻◼]+$")


def _ends_sentence(text: str) -> bool:
    """以句末标点收尾，或者以式子收尾。段末的证毕号不算。"""
    text = _END_MARK_RE.sub("", text)
    return text.rstrip(_SENT_CLOSERS).endswith(SENT_ENDS) or _ends_in_call(text)


def _ends_in_call(text: str) -> bool:
    """末尾的右括号配对的左括号紧跟在字母、数字后面：softmax(hW)、P(B|A)、count(... doc)。
    这是式子收尾，单独排的式子下面的 where、with 不接上来。正文里的括号前面隔着空格。"""
    if not text.endswith(")"):
        return False
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                return index > 0 and text[index - 1].isascii() and text[index - 1].isalnum()
    return False


@dataclass(frozen=True)
class Leader:
    """目录条目的点线和页码。点线和标题一起擦掉，页码留在原处；写回时从 dot_x（页码前最右一个点）
    按点距 pitch 往左补点，接到译文末尾。dots 是要擦的点（连同和标题排在一起的那个 span）。
    没有点线、只有页码时 pitch 为 0，不补点。"""

    number: pymupdf.Rect
    dot_x: float
    pitch: float
    baseline: float
    size: float
    color: int
    dots: tuple[dict, ...] = ()


# 正常行距下上一行行框底到下一行行框顶的空，以字号计。正文太少、量不出来时用这个
_LINE_GAP = 0.1


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
    # 版心外的页边旁注。单独翻译，不和正文并段
    side_note: bool = False
    # 目录条目行尾的点线和页码（见 _cut_leaders）。块里只有标题，点线和页码留在原页上
    leader: Leader | None = None
    # 这份文档正文紧排两行之间的空（见 _line_gap）。判断下一行是不是续行，行距按多出它的部分算
    line_gap: float = _LINE_GAP
    # 这份文档的正文字号（见 _body_size）。0 是没量
    body_size: float = 0.0
    # 自成一条：索引页上以页码收尾的一条（见 _index_page）、词表里的一格（见 _grid_cells）。
    # 不和上下的条目并段
    entry: bool = False
    # 是词表里的一格。词表和图一样，栏顶栏底跳过它找正文；索引条目照样占着栏顶栏底
    grid: bool = False
    # 在书后的索引页上（见 _index_page）
    index: bool = False


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        out, line = _carry_marks(out, line)
        if out.endswith("-") and re.match(r"[a-zA-ZÀ-ÿ]", line):
            # 行尾连字符断词接回去。数字后面的是名字里本来的连字符（Claude-3.7-|Sonnet），留着。
            # 小写字母接大写的（Distill-|Qwen、Mc-|Callum）看全文：接起来的词在别处出现过才是断词
            kept = out[-2:-1].isdigit() or (
                out[-2:-1].islower() and line[0].isupper() and _camel_join(out, line) not in _CAMEL_WORDS.get()
            )
            out = out + line if kept else out[:-1] + line
        elif out.endswith("-") and out[-2:-1].isalpha() and line[:1].isdigit():
            out += line  # 型号在连字符处折行（VE-|8、GPT-|4），连字符留着，中间没有空格
        elif CJK_RE.search(out[-1]) or CJK_RE.search(line[0]):
            out += line
        elif line[:1] in ")]}>，。；：、）】」.,":
            out += line  # 换行后的右括号、句号贴回公式，不在标点前加空格
        elif _URL_TAIL_RE.search(out) and _URL_PART_RE.match(line):
            out += line  # 网址在 / 处折行（https://github.com/ | tensorflow/tensor2tensor），中间没有空格
        else:
            out += " " + line
    return out


# 全文里大小写夹在中间的词（McCallum、PropBank）。TeX 在大写字母前断词时，接回去要去掉连字符
_CAMEL_WORDS: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar("camel_words", default=frozenset())


def _camel_join(out: str, line: str) -> str:
    left = re.search(r"[A-Za-z]+(?=-$)", out)
    right = re.match(r"[A-Za-z]+", line)
    return (left.group() if left else "") + (right.group() if right else "")


def _carry_marks(out: str, line: str) -> tuple[str, str]:
    """上一行末尾收起的标记，下一行开头原样打开的，两边都去掉，标记接着上一行：
    {i}confer-{/i} | {i}ence{/i} 接成 {i}conference{/i}，断词的连字符才认得出来。"""
    closers = re.search(r"(?:\{/[a-z]\})+$", out)
    openers = re.match(rf"(?:{STYLE_MARK_RE.pattern})+", line)
    if not closers or not openers or closers.start() == 0 or openers.end() == len(line):
        return out, line
    stack: list[str] = []
    for tag in STYLE_MARK_RE.findall(out[: closers.start()]):
        if tag.startswith("{/"):
            if stack and stack[-1][1] == tag[2]:
                stack.pop()
        else:
            stack.append(tag)
    closed = re.findall(r"\{/([a-z])\}", closers.group())
    if len(closed) > len(stack):
        return out, line
    reopened = stack[len(stack) - len(closed) :]
    if [tag[1] for tag in reversed(reopened)] != closed or "".join(reopened) != openers.group():
        return out, line
    return out[: closers.start()], line[openers.end() :]


# 行尾停在网址的分隔符上；下一行开头的一截里面还有网址的分隔符，是网址的后半
# 也可能折在协议名后面（http: | //www.ling.ohio-state.edu/）
_URL_TAIL_RE = re.compile(r"(?:(?:https?://|www\.)\S*[/._\-#?=&~]|\bhttps?:)$")
_URL_PART_RE = re.compile(r"//\w|[\w%~-]+(?:/|[/._#?=&][\w%~/._#?=&-]*[\w/])")


def _list_item(text: str) -> bool:
    """a.、(a) 和后面大写起头的 (ii) 都是列表项。换行后的 Q. 由调用处另行排除。"""
    stripped = text.lstrip()
    return bool(BULLET_RE.match(text) or _PAREN_ITEM_RE.match(stripped) or _ROMAN_ITEM_RE.match(stripped))


_INK_PAD = 0.6


def _attach_ink(page: pymupdf.Page, data: dict) -> None:
    """给每个 span 记下按字形轮廓量出来的墨迹框（span["ink"]），按文字和起笔点对上。

    get_text 默认的框按字体的上下伸量算，和字形无关。CMSY、CMEX 的伸量是给大号括号、
    根号留的，一个箭头的框就能盖住上下两行。"""
    accurate = page.get_text("dict", flags=_TEXT_FLAGS | pymupdf.TEXT_ACCURATE_BBOXES)
    found: dict[tuple[str, float, float], tuple[float, float, float, float]] = {}
    for block in accurate["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                key = (span["text"], round(span["origin"][0], 1), round(span["origin"][1], 1))
                found[key] = tuple(span["bbox"])
    fonts: dict[str, list[tuple[bytes, bytes]]] | None = None
    glyphs: dict[tuple[str, float, float], list[tuple[int, float, float]]] | None = None
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line["spans"]:
                ink = found.get((span["text"], round(span["origin"][0], 1), round(span["origin"][1], 1)))
                if ink is None:
                    continue
                # 量不出来的字形（txsys 的根号）精确框和默认框一样，按嵌入的字体单画一遍量。
                # 墨迹出了字体框才是字体框量错了，换成墨迹；在框里的照旧
                if ink == tuple(span["bbox"]) and span["text"].strip() and _font_is_math(str(span.get("font") or "")):
                    if fonts is None:
                        fonts = _embedded_fonts(page)
                        glyphs = _char_origins(page)
                    key = (span["text"], round(span["origin"][0], 1), round(span["origin"][1], 1))
                    drawn = _drawn_ink(span, fonts.get(str(span.get("font") or ""), []), (glyphs or {}).get(key, []))
                    if drawn is not None and (drawn[1] < ink[1] - _INK_PAD or drawn[3] > ink[3] + _INK_PAD):
                        ink = drawn
                        span["ink_drawn"] = True
                span["ink"] = ink


def _embedded_fonts(page: pymupdf.Page) -> dict[str, list[tuple[bytes, bytes]]]:
    """页上嵌入的字体文件和它的散列，按去掉子集前缀的名字分组。Type 3 这类取不出文件的不收。"""
    out: dict[str, list[tuple[bytes, bytes]]] = {}
    for xref, _ext, _kind, basefont, *_rest in page.get_fonts():
        try:
            buffer = page.parent.extract_font(xref)[3]
        except (RuntimeError, ValueError):
            continue
        if buffer:
            out.setdefault(basefont.split("+", 1)[-1], []).append((hashlib.sha1(buffer).digest(), buffer))
    return out


def _char_origins(page: pymupdf.Page) -> dict[tuple[str, float, float], list[tuple[int, float, float]]]:
    """每个 span 逐字的字形编号、基线和字宽，按 span 的文字和起笔点对上。空白和对不上编号的字不收。
    span 的起笔点是第一个字的：「 √」的空格和根号不在一条基线上。"""
    gids = {
        (round(char[2][0], 1), round(char[2][1], 1)): char[1]
        for trace in page.get_texttrace()
        for char in trace["chars"]
        if char[1] >= 0
    }
    found: dict[tuple[str, float, float], list[tuple[int, float, float]]] = {}
    for block in page.get_text("rawdict", flags=_TEXT_FLAGS).get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars") or []
                text = "".join(char["c"] for char in chars)
                key = (text, round(span["origin"][0], 1), round(span["origin"][1], 1))
                found[key] = [
                    (gid, char["origin"][1], char["bbox"][2] - char["bbox"][0])
                    for char in chars
                    if not char["c"].isspace()
                    and (gid := gids.get((round(char["origin"][0], 1), round(char["origin"][1], 1)))) is not None
                ]
    return found


def _drawn_ink(
    span: dict,
    fonts: list[tuple[bytes, bytes]],
    glyphs: list[tuple[int, float, float]],
) -> tuple[float, float, float, float] | None:
    """按字形单画出来的墨迹算 span 的上下沿，左右照原框。每个字按自己的基线放。
    按页上用的字形编号画：大号定界符有好几档，CMEX 的宽波浪号按 Unicode 在字体里找不到。
    同名字体可能有好几个子集，字宽对不上的不是这一个。"""
    size = float(span.get("size") or 0)
    if size <= 0 or not glyphs:
        return None
    for digest, buffer in fonts:
        tops: list[float] = []
        bottoms: list[float] = []
        for gid, base, width in glyphs:
            extent = _glyph_extent(digest, buffer, gid)
            if extent is None or abs(extent[2] * size - width) > max(0.3, width * 0.08):
                break
            tops.append(base + extent[0] * size)
            bottoms.append(base + extent[1] * size)
        else:
            x0, _y0, x1, _y1 = span["bbox"]
            return (x0, min(tops), x1, max(bottoms))
    return None


# 字形墨迹的上下沿，键是字体文件的散列和字形编号
_GLYPH_EXTENTS: dict[tuple[bytes, int], tuple[float, float, float, float, float] | None] = {}


def _glyph_extent(digest: bytes, buffer: bytes, gid: int) -> tuple[float, float, float, float, float] | None:
    """字形墨迹的上下沿（基线往下为正）、字宽和墨迹的左右沿，单位是字号。字体里没有这个字形、或画不出墨迹时为 None。"""
    key = (digest, gid)
    if key not in _GLYPH_EXTENTS:
        _GLYPH_EXTENTS[key] = _measure_glyph(buffer, gid)
    return _GLYPH_EXTENTS[key]


def _measure_glyph(buffer: bytes, gid: int) -> tuple[float, float, float, float, float] | None:
    """按字形编号在 100 号字上单画一遍，量墨迹的上下沿、字宽和墨迹的左右沿（离起笔点多远）。"""
    mupdf = pymupdf.mupdf
    try:
        font = mupdf.fz_new_font_from_buffer(None, mupdf.fz_new_buffer_from_copied_data(buffer), 0, 0)
        if not 0 < gid < font.m_internal.glyph_count:
            return None
        advance = mupdf.fz_advance_glyph(font, gid, 0)
        text = mupdf.fz_new_text()
        mupdf.fz_show_glyph(
            text,
            font,
            mupdf.FzMatrix(100, 0, 0, -100, 100, 150),
            gid,
            0,
            0,
            0,
            mupdf.FZ_BIDI_LTR,
            mupdf.FZ_LANG_UNSET,
        )
        canvas = mupdf.fz_new_pixmap(mupdf.fz_device_gray(), 300, 300, mupdf.FzSeparations(), 0)
        mupdf.fz_clear_pixmap_with_value(canvas, 255)
        device = mupdf.fz_new_draw_device(mupdf.FzMatrix(), canvas)
        mupdf.fz_fill_text(device, text, mupdf.FzMatrix(), mupdf.fz_device_gray(), [0.0], 1.0, mupdf.FzColorParams())
        mupdf.fz_close_device(device)
        pix = pymupdf.Pixmap(canvas)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
    except (RuntimeError, ValueError):
        return None
    width = pix.width
    samples = pix.samples
    rows = [y for y in range(pix.height) if min(samples[y * width : (y + 1) * width]) < 128]
    if not rows:
        return None
    columns = [x for x in range(width) if min(samples[x::width]) < 128]
    return (
        (rows[0] - 150) / 100,
        (rows[-1] + 1 - 150) / 100,
        advance,
        (columns[0] - 100) / 100,
        (columns[-1] + 1 - 100) / 100,
    )


def _bare_font_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", name)


class _GlyphInk:
    """本页逐字按字形量出来的墨迹左右沿。get_text 的框只到字宽，斜体 f 的弯钩、尾巴伸在框外。
    字形编号和嵌入的字体要在擦原文之前读，擦过的字读不到。"""

    def __init__(self, page: pymupdf.Page):
        # 字体表里的名字和 texttrace 报的写法不一定一样（Nimbus Roman Italic、NimbusRoman-Italic），去掉空格、连字符再对
        self._fonts: dict[str, list[tuple[bytes, bytes]]] = {}
        for name, found in _embedded_fonts(page).items():
            self._fonts.setdefault(_bare_font_name(name), []).extend(found)
        self._chars = [
            (_bare_font_name(str(trace["font"])), float(trace["size"]), char)
            for trace in page.get_texttrace()
            if tuple(trace["dir"]) == (1.0, 0.0)
            for char in trace["chars"]
            if char[1] >= 0 and char[0] != 32
        ]

    def x_bounds(self, span: dict) -> tuple[float, float] | None:
        """span 里各字墨迹的最左、最右。有一个字量不出来就是 None。"""
        origin = span.get("origin")
        if not origin:
            return None
        x0, _y0, x1, _y1 = span["bbox"]
        lefts: list[float] = []
        rights: list[float] = []
        for font, size, (_code, gid, (ox, oy), box) in self._chars:
            if not x0 - 0.05 <= ox < x1 - 0.05 or abs(oy - origin[1]) > 0.1:
                continue
            for digest, buffer in self._fonts.get(font, []):
                extent = _glyph_extent(digest, buffer, gid)
                if extent is not None and abs(extent[2] * size - (box[2] - box[0])) <= max(
                    0.3, (box[2] - box[0]) * 0.08
                ):
                    lefts.append(ox + extent[3] * size)
                    rights.append(ox + extent[4] * size)
                    break
            else:
                return None
        return (min(lefts), max(rights)) if lefts else None


@functools.lru_cache(maxsize=1024)
def _font_is_math(font: str) -> bool:
    """字体三层判定（BabelDOC 思路）：精确白名单 → 是公式；正文黑名单 → 不是；宽泛启发式 → 是。
    黑名单优先，避免 Times/Arial 里的数学段落被误判。

    每个 span 都要问一次，一本书却只有几十种字体，按字体名记下结果。"""
    if MATH_FONT_PRECISE_RE.search(font):
        return True
    if TEXT_FONT_RE.search(font):
        return False
    return bool(MATH_FONT_RE.search(font))


def _span_is_math(span: dict) -> bool:
    """按字体判公式。math 是提取时按全文字体定下的（见 _mark_math_roman）。"""
    return bool(span.get("math")) or _font_is_math(str(span.get("font") or ""))


# '1 Introduction'、'3.1 Encoder' 是章节标题，不能当成跨页续段
_SRC_HEAD_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+\S")


# 公式哨兵。硬切不能落在哨兵内部，否则两页都匹配不上。
_SENTINEL_NUM_RE = re.compile(r"\x01i\x02(\d+)\x01/i\x02")


def _color_value(color: str) -> int | None:
    """#rrggbb 换成整数，和 span 里的颜色同一种写法。"""
    return int(color[1:], 16) if re.fullmatch(r"#[0-9a-fA-F]{6}", color or "") else None


def _small_caps_sizes(span_lines: list[list[dict]]) -> dict[int, float]:
    """小型大写（INTRODUCTION 首字母大一号、其余小一号）里小的那部分，按大写字号算。

    同一个词里大写接着小一号的大写，中间没有空格，是同一种字的两个字号，
    不是强调。按原样写字号标记会把词从中间切开：{z125}I{/z}NTRODUCTION。
    """
    out: dict[int, float] = {}
    for line in span_lines:
        pairs: dict[float, float] = {}
        for prev, cur in zip(line, line[1:]):
            left, right = str(prev.get("text") or ""), str(cur.get("text") or "")
            if not left or not right or left[-1].isspace() or right[0].isspace():
                continue
            if not (left[-1].isupper() or left[-1] in "’'") or not _all_caps(right):
                continue
            big = out.get(id(prev), float(prev.get("size") or 0))
            small = float(cur.get("size") or 0)
            if big > 0 and 0.6 <= small / big < 0.92:
                out[id(cur)] = big
                pairs[round(small, 1)] = big
        # 原文小写开头的词整个是小号大写（UPDATE RULE），同一行按同样的对应算
        for span in line:
            big = pairs.get(round(float(span.get("size") or 0), 1))
            if big and id(span) not in out and _all_caps(str(span.get("text") or "")):
                out[id(span)] = big
    # 正文里整个词排成小型大写（AGENT、GOVERN），没有大一号的首字母：
    # 落在正文基线上、比正文小一号的全大写词，按正文字号算。中文没有大小写，照搬小字号就缩成一团
    lower = [s for line in span_lines for s in line if any(c.islower() for c in str(s.get("text") or ""))]
    if not lower:
        return out
    body = float(_by_chars(lower, lambda s: round(float(s.get("size") or 0), 1)))
    lower_ids = {id(s) for s in lower}
    for line in span_lines:
        bases = [float(s["origin"][1]) for s in line if id(s) in lower_ids and s.get("origin") is not None]
        # 同一字号连着的一串里有小写（{z90}mouse (N)），是小一号的正文，不是小型大写
        lowered = [False] * len(line)
        start = 0
        for i in range(1, len(line) + 1):
            if i == len(line) or round(float(line[i].get("size") or 0), 1) != round(
                float(line[start].get("size") or 0), 1
            ):
                low = any(c.islower() for s in line[start:i] for c in str(s.get("text") or ""))
                lowered[start:i] = [low] * (i - start)
                start = i
        caps_at = []
        for i, span in enumerate(line):
            text = str(span.get("text") or "")
            size = float(span.get("size") or 0)
            on_base = (
                not bases
                or span.get("origin") is None
                or min(abs(float(span["origin"][1]) - b) for b in bases) <= body * 0.15
            )
            caps_at.append(
                id(span) not in out
                and body > 0
                and 0.6 <= size / body < 0.92
                and on_base
                and not lowered[i]
                and (_all_caps(text) or not text.strip())
            )
        # 空格跟着两边走：夹在小型大写词中间的空格算进去，正文和小型大写之间的空格不动
        for i, span in enumerate(line):
            if not caps_at[i]:
                continue
            if str(span.get("text") or "").strip():
                out[id(span)] = body
                continue
            sides = [j for j in (i - 1, i + 1) if 0 <= j < len(line)]
            if sides and all(id(line[j]) in out or caps_at[j] for j in sides):
                out[id(span)] = body
    return out


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


_TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES


class _CharIndex(list):
    """一页上逐 span 的框和逐字的框，按 span 框的上沿分桶，查一个框（见 find）不用从头比到尾。"""

    def __init__(self, entries: Iterable[tuple[pymupdf.Rect, list[tuple[str, pymupdf.Rect]]]] = ()) -> None:
        super().__init__(entries)
        self._rows: dict[int, list[int]] = {}
        for order, (span_box, _chars) in enumerate(self):
            self._rows.setdefault(math.floor(span_box.y0), []).append(order)

    def find(self, box: pymupdf.Rect) -> list[tuple[str, pymupdf.Rect]] | None:
        """框和 box 对得上的第一个 span 的逐字框（见 _boxes_match），没有时返回 None。
        上沿相差不到半点，所在的桶最多差一格。"""
        row = math.floor(box.y0)
        found = [
            order
            for key in (row - 1, row, row + 1)
            for order in self._rows.get(key, ())
            if _boxes_match(box, self[order][0])
        ]
        return self[min(found)][1] if found else None


def _page_char_index(page: pymupdf.Page) -> _CharIndex:
    data = page.get_text("rawdict", flags=_TEXT_FLAGS)
    index: list[tuple[pymupdf.Rect, list[tuple[str, pymupdf.Rect]]]] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                chars = [
                    (str(char.get("c") or ""), pymupdf.Rect(char["bbox"]))
                    for char in span.get("chars") or []
                    if char.get("bbox") is not None
                ]
                index.append((pymupdf.Rect(span["bbox"]), chars))
    return _CharIndex(index)


def _boxes_match(left: pymupdf.Rect, right: pymupdf.Rect) -> bool:
    return (
        abs(left.x0 - right.x0) < 0.5
        and abs(left.y0 - right.y0) < 0.5
        and abs(left.x1 - right.x1) < 0.5
        and abs(left.y1 - right.y1) < 0.5
    )


def _outside_formula(
    page_spans: list[dict],
    run_spans: Sequence[dict],
    char_index: _CharIndex | None = None,
) -> list[dict]:
    """页面上的 span，公式自己的字除外。

    公式的字可能是从正文 span 里切出来的（公式检测模型框出的 σ、y = 1，根号下的数）。
    页面上的 span 还是切开前的那一个，框和公式的 span 对不上，整个会被当成别人的字：
    公式又高又宽时（带分式、上标）按它挖洞，公式自己的字就被挖掉。这样的 span 只留
    切剩下的非空白字，按连着的几段算。逐字的框取 span 的 chars（见 _attach_chars），没有时查 char_index。"""
    sources = [pymupdf.Rect(span["source"]) for span in run_spans if span.get("source") is not None]
    if not sources:
        return page_spans
    pieces = [pymupdf.Rect(span["bbox"]) for span in run_spans if span.get("bbox") is not None]
    out: list[dict] = []
    for span in page_spans:
        box = pymupdf.Rect(span["bbox"])
        if not any(_boxes_match(box, source) for source in sources):
            out.append(span)
            continue
        chars = span.get("chars")
        if chars is None and char_index is not None:
            chars = char_index.find(box)
        if chars is None:
            out.append(span)
            continue
        group: list[tuple[str, pymupdf.Rect]] = []
        for char, rect in [*chars, ("", None)]:
            middle = (rect.x0 + rect.x1) / 2 if rect is not None else 0.0
            ours = rect is not None and any(
                piece.x0 - 0.1 <= middle <= piece.x1 + 0.1 and min(piece.y1, box.y1) > max(piece.y0, box.y0)
                for piece in pieces
            )
            if rect is not None and char.strip() and not ours:
                group.append((char, rect))
                continue
            if group:
                x0, x1 = min(r.x0 for _c, r in group), max(r.x1 for _c, r in group)
                rest = {**span, "text": "".join(c for c, _r in group), "bbox": (x0, box.y0, x1, box.y1)}
                if span.get("ink") is not None:
                    rest["ink"] = (x0, span["ink"][1], x1, span["ink"][3])
                out.append(rest)
            group = []
    return out


# 换行切开的公式几片并排贴时中间的空（点）
_PART_GAP = 1.5


def _formula_parts(formula: dict) -> list[dict]:
    return formula.get("parts") or [formula]


def _by_chars(spans: list[dict], key: Callable[[dict], object]) -> object:
    """按字数占多数的取值。"""
    counts: dict[object, int] = {}
    for span in spans:
        value = key(span)
        counts[value] = counts.get(value, 0) + len(span["text"].strip())
    return max(counts, key=counts.__getitem__)


_AFFIL_MARK_RE = re.compile(r"^[∗*†‡§¶]")


def _garbled_label(text: str) -> bool:
    """图内标签用符号字体时，$ 顶替了空格。重排会把乱码译出来，留下原字。
    $ 贴着字母才算：「of $2 per hour」的 $ 前面本来就有空格，去掉空白再看会误认。"""
    compact = re.sub(r"\s+", "", text)
    if not compact or len(compact) > 80:
        return False
    return bool(re.search(r"[A-Za-z]\$|\$[A-Za-z]", text))


# 行首是一个词加编号的标签：EXAMPLE 2.1、Lemma 3.、Table 2:
_LABEL_START_RE = re.compile(r"[A-Z][A-Za-z]*\s+\d+(?:\.\d+)*[.:]?\s")
# 行首是小节编号：2.1. Bosonic Fourier transform、3 Method
_NUMBERED_HEAD_RE = re.compile(r"\d+(?:\.\d+)*\.?\s+[A-Z]")


def _styled_edge(block: TextBlock, index: int) -> bool:
    """第一截（index=0）或最后一截（-1）是粗体、斜体，或者字号和块的不一样。
    小型大写的词（MUC (Vilain et al., 1995) and …）按正文字号算，是句中的缩写，不是小标题"""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    if not spans:
        return False
    span = spans[index]
    mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))
    size = _small_caps_sizes(block.span_lines or []).get(id(span), float(span.get("size") or block.size))
    return mark.bold or mark.italic or abs(size - block.size) > block.size * 0.05


def _first_line_indented(block: TextBlock) -> bool:
    """块的首行比下面的行往右缩进半个到三个字宽，下面几行左缘齐，首行和下面的行排到同一右缘。
    居中排的几行右缘跟着左缘一起收进来，不算"""
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    if len(rows) < 2:
        return False
    lefts = [min(rect.x0 for rect in row) for row in rows]
    rights = [max(rect.x1 for rect in row) for row in rows]
    rest = lefts[1:]
    return (
        max(rest) - min(rest) <= 0.5 * block.size
        and 0.5 * block.size < lefts[0] - min(rest) <= _MAX_INDENT_EM * block.size
        and rights[0] >= max(rights[1:]) - 0.5 * block.size
    )


def _indents_paragraphs(blocks: list[TextBlock]) -> bool:
    """这份文档的段落是首行缩进的。看段落的末行（句末收住、没排到右缘）下面紧接着的一块：
    首行缩进一两个字宽的比和末行齐平的多。靠段间空白分段的文档新段齐平起笔，不算。
    定理、引理的标签（Proposition 2.4.、Proof.）和带编号的小标题本来就齐平起笔，不计。
    上下两行的空先扣掉正常行距（见 _line_gap）：行框比行距高的字体，紧接着的两行框会叠上一点"""
    indented = flush = 0
    for p, b in zip(blocks, blocks[1:]):
        if p.page != b.page or not p.line_rects or not b.line_rects or abs(p.size - b.size) > p.size * 0.05:
            continue
        if _styled_edge(b, 0) or _LABEL_START_RE.match(b.text) or _NUMBERED_HEAD_RE.match(b.text):
            continue
        tail, head = p.line_rects[-1], b.line_rects[0]
        if (
            not _ends_sentence(p.text)
            or _list_item(b.text)
            or tail.x1 > max(p.rect.x1, b.rect.x1) - 2 * p.size
            or not -0.3 * p.size <= head.y0 - tail.y1 - p.line_gap * p.size <= 1.5 * p.size
        ):
            continue
        shift = head.x0 - tail.x0
        if p.size * 0.5 < shift <= p.size * _MAX_INDENT_EM:
            indented += 1
        elif abs(shift) <= p.size * 0.3:
            flush += 1
    return indented >= 3 and indented > flush


# 段落首行缩进最多这么多个字宽
_MAX_INDENT_EM = 3


def _numbered_line(text: str) -> bool:
    """行首是编号，后面大写、汉字起头或者没有字：例句、式子、条目的编号（(21.15) Ok, we fly …、(1) For a fixed …）。
    后面小写、标点起头的多是正文里引用的式子折到了行首（(2) for improved …、(5.9), we have）"""
    lead = text.lstrip()
    tag = _EQUATION_TAG_RE.match(lead)
    if tag is None:
        return False
    rest = lead[tag.end() :].lstrip()
    return not rest or not (rest[0].islower() or rest[0] in ",.;:()[]")


# 行首的编号 (2.11)、(3)。四位数的 (2007) 是年份
_EQUATION_TAG_RE = re.compile(r"\((?:\d+\.)+\d+\)|\(\d{1,3}\)")


def _same_page_line_pair(host: TextBlock, tail: TextBlock) -> bool:
    """同一页的行尾半句接到下一短行。换栏和跨页不是这种，分界仍留着。"""
    if host.page != tail.page or len(host.line_rects) > 1:
        return False
    if tail.rect.x0 > host.rect.x0 - max(host.size * 3.5, 28):
        return False
    gap = tail.rect.y0 - host.rect.y1 - host.line_gap * host.size
    return -0.5 * host.size <= gap < 1.5 * host.size


def _is_side_label(block: TextBlock) -> bool:
    """左侧短标签，和正文第一行并排。Memo.、Property Enforcement.、(2) Statistics Derivation."""
    text = block.text.strip()
    if not text.endswith(".") or len(block.line_rects) > 1:
        return False
    if len(text.split()) > 6 or block.rect.width > 150:
        return False
    if block.bold:
        return True
    return bool(re.match(r"^\(?\d+\)", text))


def _diagramish(block: TextBlock) -> bool:
    """图内标签用符号当空格。不和正文并成一段。"""
    text = block.text.strip()
    if not text or _garbled_label(text) or _AFFIL_MARK_RE.match(text):
        return True
    words = text.split()
    junk = sum(text.count(ch) for ch in "%!#$")
    if junk and (len(words) <= 4 or block.rect.width < 90):
        return True
    if "'" in text and block.rect.width < 80 and len(words) <= 3:
        return True
    return sum(ch.isalpha() for ch in text) < 2


class _FormulaImage:
    """公式在原页上的位图。公式在另一页、或者矢量贴回失败时才用到（见 _paint_formula），裁图时只记下从哪一页、
    哪一块裁，用到时才渲染，渲染一次。

    写回在 fork 出来的子进程里时（见 render_in_parts），不能用父进程打开的原文档渲染：
    父子进程共用这个文件的读写位置，几个子进程同时读，读到的是别处的字节，点阵字的字形缺笔画、
    画成黑块，页面树也会读坏。子进程按文件名自己打开一份（见 _own_source）。"""

    def __init__(self, doc: pymupdf.Document, page: int, clip: pymupdf.Rect, spans: Sequence[dict] = ()):
        self.doc, self.page, self.clip = doc, page, pymupdf.Rect(clip)
        self.spans = list(spans)
        self._pid = os.getpid()
        self._area: pymupdf.Rect | None = None
        self._png: bytes | None = None

    def _source(self) -> pymupdf.Document:
        return self.doc if os.getpid() == self._pid else _own_source(self.doc)

    def area(self) -> pymupdf.Rect:
        """实际裁的范围。斜体 f 的弯钩、尾巴伸出字宽，左右放到公式自己的字形墨迹，每侧最多 1.5pt。"""
        if self._area is None:
            area = pymupdf.Rect(self.clip)
            if self.spans:
                ink = _page_glyph_ink(self._source(), self.page)
                bounds = [ink.x_bounds(span) for span in self.spans]
                if bounds and None not in bounds:
                    area.x0 = max(min(area.x0, *(b[0] for b in bounds)), self.clip.x0 - 1.5)
                    area.x1 = min(max(area.x1, *(b[1] for b in bounds)), self.clip.x1 + 1.5)
            self._area = area
        return self._area

    def png(self) -> bytes:
        if self._png is None:
            # 透明底：白底会盖住贴图位置下面的图框线、分隔线
            pixmap = self._source()[self.page].get_pixmap(dpi=300, clip=self.area(), alpha=True)
            self._png = pixmap.tobytes("png")
        return self._png


# 量墨迹要逐字读字形编号、取嵌入字体，同一页的公式挨着裁，只留最近一页
_GLYPH_INK_PAGE: list[tuple[tuple[int, int], "_GlyphInk"]] = []


def _page_glyph_ink(doc: pymupdf.Document, page: int) -> "_GlyphInk":
    key = (id(doc), page)
    if not _GLYPH_INK_PAGE or _GLYPH_INK_PAGE[0][0] != key:
        _GLYPH_INK_PAGE[:] = [(key, _GlyphInk(doc[page]))]
    return _GLYPH_INK_PAGE[0][1]


# 子进程自己打开的原文档，按文件名记。只在子进程里存，子进程写完就退出
_OWN_SOURCES: dict[str, pymupdf.Document] = {}


def _own_source(doc: pymupdf.Document) -> pymupdf.Document:
    """本进程自己打开的那份原文档。从内存打开的文档没有文件可共用，照用原来的。"""
    if not doc.name or not os.path.isfile(doc.name):
        return doc
    if doc.name not in _OWN_SOURCES:
        _OWN_SOURCES[doc.name] = pymupdf.open(doc.name)
    return _OWN_SOURCES[doc.name]


def _source_bold_lead(block: TextBlock) -> str:
    """第一行开头的短粗体短语，以句号结束，后面还有非粗体正文。"""
    if not block.span_lines:
        return ""
    lead: list[str] = []
    saw_plain = False
    for span in block.span_lines[0]:
        text = str(span.get("text") or "")
        if not text.strip():
            if lead:
                lead.append(text)
            continue
        mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))
        if mark.bold and not saw_plain:
            lead.append(text)
            continue
        saw_plain = True
        break
    phrase = "".join(lead).strip()
    if not saw_plain or not phrase.endswith(".") or len(phrase.split()) > 6 or len(phrase) > 40:
        return ""
    return phrase


def _uniform_bold(block: TextBlock) -> bool:
    """正文字都是粗体。公式字体的角标（名字后面的星号）原样贴回，不算。"""
    spans = [
        span
        for line in (block.span_lines or [])
        for span in line
        if str(span.get("text") or "").strip() and not _span_is_math(span)
    ]
    if not spans:
        return False
    return all(emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)).bold for span in spans)


def _wrap_lead_bold(text: str) -> str:
    """译文丢掉了段首粗体时，补到第一个句号。导语都很短，不把整段包进去。"""
    stripped = text.lstrip()
    pad = text[: len(text) - len(stripped)]
    cut = 0
    for index, char in enumerate(stripped):
        if index > 16:
            return text
        if char in "。." and index >= 1:
            cut = index + 1
            break
    if cut <= 1:
        return text
    return f"{pad}{{b}}{stripped[:cut]}{{/b}}{stripped[cut:]}"


def _with_source_bold(block: TextBlock, text: str) -> str:
    """写回时补上源文的粗体。模型经常丢掉 {b}，汉字字形本身也没有粗体。"""
    if not text or text == SPILL_TAIL or "{b}" in text:
        return text
    if _source_bold_lead(block):
        return _wrap_lead_bold(text)
    if _uniform_bold(block):
        return "{b}" + text + "{/b}"
    return text


def _visual_lines(block: TextBlock) -> list[list[pymupdf.Rect]]:
    """同一基线上的碎片并成一行。行框按词切开时，下一碎片不是下一行。"""
    grouped: list[list[pymupdf.Rect]] = []
    gap = max(block.size * 0.4, 2.0)
    for rect in block.line_rects:
        placed = False
        for line in grouped:
            if abs(rect.y0 - line[0].y0) <= gap:
                line.append(rect)
                placed = True
                break
        if not placed:
            grouped.append([rect])
    return grouped


def _column_left(rect: pymupdf.Rect, geo: PageGeometry | None) -> float | None:
    """rect 所在栏（单栏时是版心）的左缘。rect 在左缘左边时返回 None"""
    if geo is None:
        return None
    edges = [geo.column_of(rect).x0] if len(geo.columns) >= 2 else []
    if geo.frame is not None:
        edges.append(geo.frame.x0)
    return max((edge for edge in edges if edge <= rect.x0 + 2), default=None)


def _paragraph_indent(blocks: list[TextBlock]) -> float | None:
    """首行缩进的文档里段首缩进多宽（点）：多行段落首行比下面几行缩进的中位数。不缩进的文档返回 None"""
    if not _indents_paragraphs(blocks):
        return None
    shifts = []
    for block in blocks:
        if _first_line_indented(block):
            lefts = [min(rect.x0 for rect in row) for row in sorted(_visual_lines(block), key=lambda row: row[0].y0)]
            shifts.append(lefts[0] - min(lefts[1:]))
    return statistics.median(shifts) if shifts else None


def _pages_to_write(blocks: list[TextBlock], translations: list[str]) -> dict[int, list[tuple[TextBlock, str]]]:
    """每页要写回的块和写成的样子。译文和原文一样的块不动。"""
    by_page: dict[int, list[tuple[TextBlock, str]]] = {}
    for b, t in zip(blocks, translations):
        if t == SPILL_TAIL or (t and t.strip() and t.strip() != b.text.strip()):
            shown = t if t == SPILL_TAIL else _with_source_bold(b, t)
            by_page.setdefault(b.page, []).append((b, shown))
    return by_page


# 续页块只擦原文。译文按满行写在起始块，排满再接到续页，不画出页边。
SPILL_TAIL = "\x02"


# 节号、条目号：4、4.1、3.2.1.、B.3.2.、1.、(a)、a)、(1)。编号外面可能包着字号、粗体标记
_NUMBER_LABEL_RE = re.compile(
    r"^((?:\{[bizu]\d*\})*)"
    r"(\d{1,3}(?:\.\d{1,3})*\.?|[A-Z](?:\.\d{1,3})+\.?|[A-Z]\.|[IVX]{1,4}\.|\(?[a-z]\)|\(\d{1,2}\))"
    r"((?:\{/[bizu]\})*)(?=\s|\{[bizu]\d*\}\s)"
)


_CAPTION_NUM_RE = re.compile(r"\s*(?:Figure|Fig\.?|Table|Tab\.?)\s*(\d+)\b", re.I)


# 送翻文本里的标记：样式、字号、颜色、公式占位符、栏界
_MARKUP_RE = re.compile(r"\{(?:/?[biu]|z\d+|/z|c[0-9a-f]{6}|/c|v\d+|\|)\}")
