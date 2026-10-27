"""中文行的两端对齐。

MuPDF 的 letter-spacing、word-spacing、text-align:justify 都拉不动纯汉字行。
这里按字形自己写文本：非末行的余量摊到字与字之间，标点前面不拉开。
复制出来的仍是原来的字，不含为对齐插入的空格。
斜体标记不改走 HTML：汉字没有斜体字形，仍用正文字体；拉丁字母换斜体字体。
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pymupdf

from .pdf_flow import open_marks
from .pdf_typeset import FormulaPiece, TextPiece, TypesetLine, line_offsets

# htmlbox 首行 origin 相对盒子顶的比例。和它对齐，换写法后行高不变。
FIRST_BASELINE = 1.108
_FONT_NAME = "DocCJK"
# 汉字字形大约占基线上 0.88em、下 0.12em
_GLYPH_ASCENT = 0.88
_GLYPH_DESCENT = 0.12
_MIN_SLOTS = 6
_MIN_SLACK_EM = 0.15
_MAX_GAP_EM = 0.35
# 词间空格每处最多再拉开多少（em），前后两个缝各摊一半。空格本身约 0.26em，拉满后约半个字宽
_MAX_SPACE_EM = 0.25
# 有词间空格的行里，字与字之间最多拉开多少（em）
_MAX_MIXED_GAP_EM = 0.1
# 句读贴着前一个字，开括号贴着后一个字。空格可以拉开，公式后的分界就靠它。
_NO_GAP_BEFORE = frozenset("。，、；：？！）】」』〉》％%.,;:?!)]}>")
_NO_GAP_AFTER = frozenset("（【「『《〈([{")

_MARK_RE = re.compile(r"\{i\}|\{/i\}|\{b\}|\{/b\}|\{u\}|\{/u\}|\{/z\}|\{z(\d+)\}|\{/c\}|\{c([0-9a-f]{6})\}")
# 下划线在基线下多远、多粗（em）
_UNDERLINE_DROP = 0.1
_UNDERLINE_WIDTH = 0.05
_LATIN_FONTS = {
    (True, False): ("DocIT", "heit"),
    (False, True): ("DocBO", "hebo"),
    (True, True): ("DocBI", "hebi"),
}

# 没有合成字的组合重音（P̂、x̃）。字体里没有组合字形，画成对应的独立重音叠在前一个字上
_MARK_GLYPHS = {
    "\u0300": "`", "\u0301": "´", "\u0302": "ˆ", "\u0303": "˜", "\u0304": "¯", "\u0305": "¯",
    "\u0306": "˘", "\u0307": "˙", "\u0308": "¨", "\u030a": "˚", "\u030c": "ˇ",
}
_MARK_FONT = ("DocRG", "helv")
# 独立重音按小写字母的高度画，叠在大写字母、带上伸部的字母上要抬高（em）
_TALL_LIFT = 0.2
_ITALIC_SLANT = 0.21

_font: pymupdf.Font | None = None
_latin: dict[str, pymupdf.Font] = {}
# 各字体里每个字的宽度（1 号字）
_ADVANCES: dict[pymupdf.Font, dict[str, float]] = {}


@dataclass(frozen=True)
class FormulaSlot:
    index: int
    x: float
    baseline: float


@dataclass(frozen=True)
class _Run:
    text: str
    italic: bool
    bold: bool
    size_pct: int
    # #rrggbb。空串是块的颜色
    color: str = ""
    underline: bool = False


@dataclass(frozen=True)
class _Segment:
    text: str
    font_name: str
    font: pymupdf.Font
    size: float
    bold: bool = False
    color: str = ""
    underline: bool = False
    # 组合重音：不占宽，叠在前一个字上
    mark: bool = False
    italic: bool = False

    @property
    def width(self) -> float:
        if self.mark:
            return 0.0
        return _text_length(self.font, self.text, self.size) if self.text else 0.0


def _text_length(font: pymupdf.Font, text: str, size: float) -> float:
    """同 Font.text_length。它每个字都要问一遍 MuPDF 用哪个字形，这里每个字只问一次，
    按原来的次序累加，结果相同。"""
    advances = _ADVANCES.setdefault(font, {})
    total = 0.0
    for char in text:
        advance = advances.get(char)
        if advance is None:
            advance = advances[char] = font.text_length(char, 1)
        total += advance
    return total * size


def cjk_font() -> pymupdf.Font:
    global _font
    if _font is None:
        _font = pymupdf.Font("china-ss")
    return _font


def _latin_font(italic: bool, bold: bool) -> tuple[str, pymupdf.Font]:
    name, builtin = _LATIN_FONTS[(italic, bold)]
    face = _latin.get(name)
    if face is None:
        face = pymupdf.Font(builtin)
        _latin[name] = face
    return name, face


def _styled_runs(text: str) -> list[_Run]:
    """按 {i} {b} {u} {zN} {cRRGGBB} 切开。标记本身不占字。"""
    italic = False
    bold = False
    underline = False
    size = 100
    color = ""
    runs: list[_Run] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            runs.append(_Run("".join(buf), italic, bold, size, color, underline))
            buf.clear()

    index = 0
    while index < len(text):
        mark = _MARK_RE.match(text, index)
        if mark is None:
            buf.append(text[index])
            index += 1
            continue
        flush()
        token = mark.group(0)
        if token == "{i}":
            italic = True
        elif token == "{/i}":
            italic = False
        elif token == "{b}":
            bold = True
        elif token == "{/b}":
            bold = False
        elif token == "{u}":
            underline = True
        elif token == "{/u}":
            underline = False
        elif token == "{/z}":
            size = 100
        elif token == "{/c}":
            color = ""
        elif mark.group(2):
            color = "#" + mark.group(2)
        else:
            size = int(mark.group(1))
        index = mark.end()
    flush()
    return runs


def _keeps_body_font(char: str) -> bool:
    """汉字和全角符号没有斜体字形，强调也不换字体。"""
    code = ord(char)
    if 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xAC00 <= code <= 0xD7AF:
        return True
    return unicodedata.east_asian_width(char) in ("W", "F")


def _face_for(char: str, italic: bool, bold: bool) -> tuple[str, pymupdf.Font] | None:
    body = cjk_font()
    if _keeps_body_font(char) or not (italic or bold):
        if body.has_glyph(ord(char)):
            return _FONT_NAME, body
        if not (italic or bold):
            return None
    name, face = _latin_font(italic, bold)
    if face.has_glyph(ord(char)):
        return name, face
    if body.has_glyph(ord(char)):
        return _FONT_NAME, body
    return None


def _append_segment(
    segments: list[_Segment],
    buf: list[str],
    face: tuple[str, pymupdf.Font] | None,
    size: float,
    bold: bool,
    color: str = "",
    underline: bool = False,
) -> None:
    if not buf or face is None:
        return
    name, font = face
    text = "".join(buf)
    # 汉字没有粗体字形。粗体只记在片段上，写的时候描一笔，不换字体。
    if (
        segments
        and not segments[-1].mark
        and segments[-1].font_name == name
        and segments[-1].size == size
        and segments[-1].bold == bold
        and segments[-1].color == color
        and segments[-1].underline == underline
    ):
        prev = segments[-1]
        segments[-1] = _Segment(prev.text + text, name, font, size, bold, color, underline)
    else:
        segments.append(_Segment(text, name, font, size, bold, color, underline))
    buf.clear()


def _segments(text: str, em: float) -> list[_Segment] | None:
    """同一字体、同一字号的相邻片段合成一段。缺字返回 None。"""
    if "\x01" in text:
        return None
    segments: list[_Segment] = []
    for run in _styled_runs(text):
        pct = run.size_pct if 50 <= run.size_pct <= 200 else 100
        size = em * pct / 100.0
        buf: list[str] = []
        face: tuple[str, pymupdf.Font] | None = None
        # 拉丁粗体已经换了 hebo。描边只加在仍用正文字体的汉字上。
        stroke = run.bold
        for char in run.text:
            if char in _MARK_GLYPHS and (buf or segments):
                if face is not None:
                    _append_segment(segments, buf, face, size, stroke and face[0] == _FONT_NAME, run.color, run.underline)
                face = None
                name, font = _latin_font(run.italic, run.bold) if run.italic or run.bold else _mark_font()
                segments.append(_Segment(char, name, font, size, False, run.color, False, True, run.italic))
                continue
            picked = _face_for(char, run.italic, run.bold)
            if picked is None:
                return None
            if face is not None and picked != face:
                _append_segment(segments, buf, face, size, stroke and face[0] == _FONT_NAME, run.color, run.underline)
            face = picked
            buf.append(char)
        if face is not None:
            _append_segment(segments, buf, face, size, stroke and face[0] == _FONT_NAME, run.color, run.underline)
    return segments


def _mark_font() -> tuple[str, pymupdf.Font]:
    name, builtin = _MARK_FONT
    face = _latin.get(name)
    if face is None:
        face = pymupdf.Font(builtin)
        _latin[name] = face
    return name, face


def _latin_unit(ch: str) -> bool:
    """词内的拉丁字母、数字、连字符和下划线（l_partkey）。两端对齐不能从这里拉开。"""
    return ch.isascii() and (ch.isalnum() or ch in "-'’_")


def can_stretch(left: str, right: str) -> bool:
    if right in _NO_GAP_BEFORE or left in _NO_GAP_AFTER or right in _MARK_GLYPHS:
        return False
    # 作者名被拉成 Z h o n g，就是把词内的字母也拉开了。只在词与词之间留缝。
    # 网址、邮箱（jimmy@psi.utoronto.ca）中间的 @ . / 也是连着写的一串，拉开就像断成了几段
    if _latin_unit(left) and _latin_unit(right) or _ascii_mark(left) and _ascii_mark(right):
        return False
    return True


def _ascii_mark(ch: str) -> bool:
    return ch.isascii() and ch.isprintable() and not ch.isspace()


def justify_extras(text: str, slack: float, em: float) -> list[float]:
    """每个字后面摊多少点。长度是 len(text)-1。不该拉的行返回空列表。"""
    if em <= 0 or slack < em * _MIN_SLACK_EM or len(text) < 2:
        return []
    return _spread([(text[i], text[i + 1]) for i in range(len(text) - 1)], slack, em)


def _spread(pairs: list[tuple[str, str]], slack: float, em: float) -> list[float]:
    """余量摊到相邻两字之间。词间空格旁的缝按两份摊，字与字之间按一份：中英混排的行里
    拉丁词长、拉不开，平均摊会把汉字拉成「常 识 推 理」。一种缝拉到上限，剩下的给另一种，
    两种都到上限，余下的留在行尾。"""
    stretch = [can_stretch(left, right) for left, right in pairs]
    if sum(stretch) < _MIN_SLOTS:
        return []
    spaces = [ok and (left.isspace() or right.isspace()) for ok, (left, right) in zip(stretch, pairs)]
    n_space = sum(spaces)
    n_char = sum(stretch) - n_space
    unit = slack / (2 * n_space + n_char)
    space_cap = em * _MAX_SPACE_EM / 2
    space_each = min(2 * unit, space_cap) if n_space else 0.0
    # 有词间空格的行，字与字之间只稍微拉开，再多的余量留在行尾
    char_cap = _MAX_MIXED_GAP_EM if n_space else _MAX_GAP_EM
    char_each = min((slack - space_each * n_space) / n_char, em * char_cap) if n_char else 0.0
    if n_space:
        space_each = min((slack - char_each * n_char) / n_space, space_cap)
    if max(space_each, char_each) < 0.05:
        return []
    return [space_each if spaced else char_each if ok else 0.0 for ok, spaced in zip(stretch, spaces)]


def page_origin(page: pymupdf.Page) -> tuple[float, float]:
    """页面坐标原点在 PDF 坐标里的位置，即裁剪框的左上角。

    裁剪框或媒体框不从 0 起时，页面坐标要加上这段偏移才是 PDF 坐标。
    """
    return float(page.cropbox.x0), float(page.mediabox.y1 - page.cropbox.y0)


def _lift(em: float, first_room: float | None) -> float:
    """整段往上提多少。按 htmlbox 的首行基线写，汉字的下沿常比原文第一行的行框低，
    压到紧贴在下面的线（代码框的上沿）。字形下沿超出原文行框时往上提，最多提到字形
    上沿和盒子顶齐平，不越过上面的东西。"""
    if first_room is None:
        return 0.0
    over = em * (FIRST_BASELINE + _GLYPH_DESCENT) - first_room
    return min(max(over, 0.0), em * (FIRST_BASELINE - _GLYPH_ASCENT))


def draw_cjk_lines(
    page: pymupdf.Page,
    doc: pymupdf.Document,
    lines: list[TypesetLine],
    box: pymupdf.Rect,
    em: float,
    gap: float,
    color: str,
    align: str,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
    justify_last: bool = False,
    lanes: Sequence[tuple[float, float]] = (),
    first_room: float | None = None,
    paint: Callable[[FormulaSlot], str] | None = None,
) -> list[FormulaSlot] | None:
    """写成两端对齐的行。缺字时返回 None，调用方改走 HTML 盒子。
    align 是 left、center 或 right，居中、靠右排的行不拉开。

    段落还要接到下一页时，这一页的末行也是满行，同样两端对齐。
    lanes 是开头几行相对盒子左缘的缩进和宽度（让开图的那几行），其余各行占满盒子。
    first_room 是原文第一行从盒子顶到下沿的高度，见 _lift。
    paint 给出画一个公式槽的命令（页面坐标，不含本段的平移），画在内容流里这个槽的位置上：
    拖选、复制时公式夹在前后的字中间，不排到整页最后。
    """
    left, top = page_origin(page)
    placed = _place_lines(
        lines, box, em, gap, color, align, image_indexes, fallback_text, top,
        justify_last, left, lanes, first_room,
    )
    if placed is None:
        return None
    commands, slots, fonts = placed
    for number, slot in enumerate(slots):
        drawn = paint(slot) if paint is not None else ""
        if drawn and left:
            drawn = f"q 1 0 0 1 {-left:.2f} 0 cm\n{drawn}\nQ"
        # 文本对象里不能画 XObject，先结束再接着写；字体、颜色是图形状态，接着写的字照旧
        commands = commands.replace(_slot_mark(number), f"ET\n{drawn}\nBT" if drawn else "")
    if commands:
        for name, face in fonts.items():
            page.insert_font(fontname=name, fontbuffer=face.buffer)
        _append_contents(page, doc, commands)
    return slots


def _place_lines(
    lines: list[TypesetLine],
    box: pymupdf.Rect,
    em: float,
    gap: float,
    color: str,
    align: str,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
    page_top: float,
    justify_last: bool = False,
    page_left: float = 0.0,
    lanes: Sequence[tuple[float, float]] = (),
    first_room: float | None = None,
) -> tuple[str, list[FormulaSlot], dict[str, pymupdf.Font]] | None:
    red, green, blue = _rgb(color)
    chunks: list[str] = ["q"]
    if page_left:
        chunks.append(f"1 0 0 1 {page_left:.2f} 0 cm")
    chunks += [
        "BT",
        f"{red:.4f} {green:.4f} {blue:.4f} rg",
        f"{red:.4f} {green:.4f} {blue:.4f} RG",
    ]
    slots: list[FormulaSlot] = []
    fonts: dict[str, pymupdf.Font] = {}
    last = len(lines) - 1
    offsets, _extra = line_offsets(lines, em, gap)
    wrote = False
    # 公式把一行切成几片，每片从头读标记。前一片还开着的粗体、颜色带到下一片
    opened: list[str] = []
    # 下划线：文本对象里不能画填充，写完字再画
    rules: list[tuple[float, float, float, float, str]] = []
    lift = _lift(em, first_room)
    for index, line in enumerate(lines):
        baseline = box.y0 + em * FIRST_BASELINE - lift + index * em * gap + offsets[index]
        indent, width = lanes[index] if index < len(lanes) else (0.0, box.width)
        row = _place_line(
            line, box.x0 + indent, width, em,
            justify=align == "left" and (justify_last or index != last),
            align=align,
            image_indexes=image_indexes,
            fallback_text=fallback_text,
            opened=opened,
            color=color,
        )
        if row is None:
            return None
        ops, line_slots, line_fonts, line_rules = row
        # 行里的槽标记按行内序号写，换成全段的序号
        ops = [_slot_mark(len(slots) + int(op[len("{line-slot"):-1])) if op.startswith("{line-slot") else op
               for op in ops]
        fonts.update(line_fonts)
        if ops:
            wrote = True
            pdf_y = page_top - baseline
            chunks.extend(op.replace("{Y}", f"{pdf_y:.2f}") for op in ops)
        rules.extend((x0, x1, baseline, size, paint) for x0, x1, size, paint in line_rules)
        for slot_index, slot_x in line_slots:
            slots.append(FormulaSlot(slot_index, slot_x, baseline))
    if not wrote and not slots:
        return "", [], {}
    chunks.append("ET")
    for x0, x1, baseline, size, paint in rules:
        red, green, blue = _rgb(paint or color)
        thick = max(size * _UNDERLINE_WIDTH, 0.4)
        bottom = page_top - (baseline + size * _UNDERLINE_DROP + thick)
        chunks.append(f"{red:.4f} {green:.4f} {blue:.4f} rg {x0:.2f} {bottom:.2f} {x1 - x0:.2f} {thick:.2f} re f")
    chunks.append("Q")
    return "\n".join(chunks) + "\n", slots, fonts


def _trailing_space(pieces: list[tuple[list[_Segment], int | None, float, str]]) -> float:
    """一行末尾空白的宽度。行尾是公式槽时为 0。"""
    width = 0.0
    for segments, slot, _width, _visible in reversed(pieces):
        if slot is not None:
            return width
        for seg in reversed(segments):
            body = seg.text.rstrip()
            if not seg.mark:
                width += _text_length(seg.font, seg.text[len(body):], seg.size)
            if body:
                return width
    return width


def _slot_mark(number: int) -> str:
    """公式槽在文字操作里的位置标记，单占一行，画的时候换成公式（见 draw_cjk_lines）。"""
    return f"{{slot{number}}}"


def _place_line(
    line: TypesetLine,
    x0: float,
    width: float,
    em: float,
    justify: bool,
    align: str,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
    opened: list[str] | None = None,
    color: str = "#000000",
) -> tuple[
    list[str], list[tuple[int, float]], dict[str, pymupdf.Font], list[tuple[float, float, float, str]],
] | None:
    """一行的文本操作（基线 y 先写成 {Y}）、公式槽的横坐标，和要画下划线的几段（起止横坐标、字号、颜色）。
    opened 是前面还开着的样式标记，读完这一行就地更新。color 是块的颜色，片段没有颜色时用它。"""
    pieces: list[tuple[list[_Segment], int | None, float, str]] = []
    fonts: dict[str, pymupdf.Font] = {}
    carried = opened if opened is not None else []
    for piece in line.pieces:
        prefix = "".join(carried)
        if isinstance(piece, TextPiece):
            carried[:] = open_marks(piece.text, carried)
        drawn = _draw_piece(piece, em, image_indexes, fallback_text, prefix)
        if drawn is None:
            return None
        segments, slot = drawn
        for seg in segments:
            fonts[seg.font_name] = seg.font
        if slot is not None:
            pieces.append((segments, slot, piece.slot_width if isinstance(piece, FormulaPiece) else 0.0, ""))
        else:
            visible = "".join(seg.text for seg in segments)
            pieces.append((segments, None, sum(seg.width for seg in segments), visible))
    texts: list[str | None] = [None if slot is not None else visible for _, slot, _, visible in pieces]
    widths = [piece_width for _, _, piece_width, _ in pieces]
    total = sum(widths)
    slack = width - total
    extras = line_extras(texts, slack, em) if justify and slack > 0 else []
    cursor = x0
    # 行尾的空格看不见，靠右、居中按最后一个字对齐
    inked = total - _trailing_space(pieces)
    if inked < width and align != "left":
        cursor += width - inked if align == "right" else (width - inked) / 2
    ops: list[str] = []
    slots: list[tuple[int, float]] = []
    rules: list[tuple[float, float, float, str]] = []
    extra_at = 0
    face_now: tuple[str, float] | None = None
    paint_now = ""
    for segments, slot, piece_width, visible in pieces:
        if slot is not None:
            ops.append(f"{{line-slot{len(slots)}}}")
            slots.append((slot, cursor))
            cursor += piece_width
            continue
        count = max(len(visible) - 1, 0)
        piece_extras = extras[extra_at:extra_at + count]
        extra_at += count
        pos = 0
        # 上一个字的起点、宽度和字，组合重音叠在它上面
        base: tuple[float, float, str] | None = None
        for seg in segments:
            n = len(seg.text)
            if seg.mark:
                if base is not None:
                    ops.extend(_mark_ops(seg, base, color, face_now, paint_now))
                    face_now = (seg.font_name, seg.size)
                    paint_now = seg.color
                pos += n
                if pos < len(visible) and pos - 1 < len(piece_extras):
                    cursor += piece_extras[pos - 1]
                continue
            internal = piece_extras[pos:pos + max(n - 1, 0)]
            glyphs = _glyphs(seg.font, seg.text)
            if glyphs is None:
                return None
            if (seg.font_name, seg.size) != face_now:
                ops.append(f"/{seg.font_name} {seg.size:.2f} Tf")
                face_now = (seg.font_name, seg.size)
            if seg.color != paint_now:
                red, green, blue = _rgb(seg.color or color)
                ops.append(f"{red:.4f} {green:.4f} {blue:.4f} rg")
                ops.append(f"{red:.4f} {green:.4f} {blue:.4f} RG")
                paint_now = seg.color
            # 正文字体没有汉字粗体。填充再描边，字还是一个，复制出来不会变两个。
            if seg.bold:
                ops.append("2 Tr")
                ops.append(f"{max(seg.size * 0.045, 0.2):.2f} w")
            ops.append(f"1 0 0 1 {cursor:.2f} {{Y}} Tm")
            ops.append(_tj(glyphs, internal, seg.size))
            if seg.bold:
                ops.append("0 Tr")
            advance = seg.width + sum(internal)
            if seg.underline and seg.text.strip():
                # 接着上一段下划线画，词间拉开的空也画上
                if rules and abs(rules[-1][1] - cursor) < 0.5 and rules[-1][3] == seg.color:
                    rules[-1] = (rules[-1][0], cursor + advance, rules[-1][2], seg.color)
                else:
                    rules.append((cursor, cursor + advance, seg.size, seg.color))
            last = _text_length(seg.font, seg.text[-1], seg.size) if seg.text else 0.0
            base = (cursor + advance - last, last, seg.text[-1:])
            pos += n
            if pos < len(visible) and pos - 1 < len(piece_extras):
                advance += piece_extras[pos - 1]
            cursor += advance
    if paint_now:
        # 下一行从块的颜色开始
        red, green, blue = _rgb(color)
        ops.append(f"{red:.4f} {green:.4f} {blue:.4f} rg")
        ops.append(f"{red:.4f} {green:.4f} {blue:.4f} RG")
    return ops, slots, fonts, rules


def _mark_ops(
    seg: _Segment, base: tuple[float, float, str], color: str,
    face_now: tuple[str, float] | None, paint_now: str,
) -> list[str]:
    """组合重音画成独立重音，中线对着前一个字的中线。大写字母和带上伸部的字母上抬高一截，
    斜体顺着字的斜度往右挪。"""
    x, width, char = base
    glyph = _MARK_GLYPHS[seg.text]
    gid = seg.font.has_glyph(ord(glyph))
    if not gid:
        return []
    ops: list[str] = []
    if (seg.font_name, seg.size) != face_now:
        ops.append(f"/{seg.font_name} {seg.size:.2f} Tf")
    if seg.color != paint_now:
        red, green, blue = _rgb(seg.color or color)
        ops.append(f"{red:.4f} {green:.4f} {blue:.4f} rg")
        ops.append(f"{red:.4f} {green:.4f} {blue:.4f} RG")
    lift = seg.size * _TALL_LIFT if char.isupper() or char.isdigit() or char in "bdfhklt" else 0.0
    shift = lift * _ITALIC_SLANT if seg.italic else 0.0
    left = x + (width - _text_length(seg.font, glyph, seg.size)) / 2 + shift
    ops.append(f"1 0 0 1 {left:.2f} {{Y}} Tm")
    if lift:
        ops.append(f"{lift:.2f} Ts")
    ops.append(f"[<{gid:04X}>]TJ")
    if lift:
        ops.append("0 Ts")
    return ops


def _draw_piece(
    piece: TextPiece | FormulaPiece,
    em: float,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
    opened: str = "",
) -> tuple[list[_Segment], int | None] | None:
    """文字片返回片段。公式图返回空片段和占位符序号。缺字返回 None。opened 是前面还开着的标记。"""
    if isinstance(piece, FormulaPiece) and piece.index in image_indexes:
        return [], piece.index
    text = opened + piece.text if isinstance(piece, TextPiece) else fallback_text.get(piece.index, "")
    segments = _segments(text, em)
    if segments is None:
        return None
    return segments, None


def line_extras(texts: list[str | None], slack: float, em: float) -> list[float]:
    """把整行的余量摊进各段文字。公式槽是原子，不在这里拆。"""
    if em <= 0 or slack < em * _MIN_SLACK_EM:
        return []
    pairs = [(text[i], text[i + 1]) for text in texts if text and len(text) >= 2 for i in range(len(text) - 1)]
    return _spread(pairs, slack, em)


def _glyphs(font: pymupdf.Font, text: str) -> list[int] | None:
    ids: list[int] = []
    for char in text:
        gid = font.has_glyph(ord(char))
        if not gid:
            return None
        ids.append(gid)
    return ids


def _tj(glyphs: list[int], extras: list[float], em: float) -> str:
    parts: list[str] = ["["]
    for index, gid in enumerate(glyphs):
        if index and index - 1 < len(extras) and extras[index - 1]:
            parts.append(str(int(round(-extras[index - 1] / em * 1000))))
        parts.append(f"<{gid:04X}>")
    parts.append("]TJ")
    return "".join(parts)


def _rgb(color: str) -> tuple[float, float, float]:
    hexes = color.removeprefix("#")
    if len(hexes) != 6:
        return (0.0, 0.0, 0.0)
    try:
        return (
            int(hexes[0:2], 16) / 255.0,
            int(hexes[2:4], 16) / 255.0,
            int(hexes[4:6], 16) / 255.0,
        )
    except ValueError:
        return (0.0, 0.0, 0.0)


def _append_contents(page: pymupdf.Page, doc: pymupdf.Document, commands: str) -> None:
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, commands.encode())
    current = doc.xref_get_key(page.xref, "Contents")[1].strip()
    if current.startswith("["):
        doc.xref_set_key(page.xref, "Contents", f"{current[:-1].rstrip()} {xref} 0 R]")
    elif current and current != "null":
        doc.xref_set_key(page.xref, "Contents", f"[{current} {xref} 0 R]")
    else:
        doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
