"""中文行的两端对齐。

MuPDF 的 letter-spacing、word-spacing、text-align:justify 都拉不动纯汉字行。
这里按字形自己写文本：非末行的余量摊到字与字之间，标点前面不拉开。
复制出来的仍是原来的字，不含为对齐插入的空格。
斜体标记不改走 HTML：汉字没有斜体字形，仍用正文字体；拉丁字母换斜体字体。
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
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

_font: pymupdf.Font | None = None
_latin: dict[str, pymupdf.Font] = {}


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

    @property
    def width(self) -> float:
        return self.font.text_length(self.text, self.size) if self.text else 0.0


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


def _latin_unit(ch: str) -> bool:
    """词内的拉丁字母、数字、连字符和下划线（l_partkey）。两端对齐不能从这里拉开。"""
    return ch.isascii() and (ch.isalnum() or ch in "-'’_")


def can_stretch(left: str, right: str) -> bool:
    if right in _NO_GAP_BEFORE or left in _NO_GAP_AFTER:
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
    slots = [i for i in range(len(text) - 1) if can_stretch(text[i], text[i + 1])]
    if len(slots) < _MIN_SLOTS:
        return []
    each = min(slack / len(slots), em * _MAX_GAP_EM)
    if each < 0.05:
        return []
    extras = [0.0] * (len(text) - 1)
    for index in slots:
        extras[index] = each
    return extras


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
    centered: bool,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
    justify_last: bool = False,
    lanes: Sequence[tuple[float, float]] = (),
    first_room: float | None = None,
) -> list[FormulaSlot] | None:
    """写成两端对齐的行。缺字时返回 None，调用方改走 HTML 盒子。

    段落还要接到下一页时，这一页的末行也是满行，同样两端对齐。
    lanes 是开头几行相对盒子左缘的缩进和宽度（让开图的那几行），其余各行占满盒子。
    first_room 是原文第一行从盒子顶到下沿的高度，见 _lift。
    """
    left, top = page_origin(page)
    placed = _place_lines(
        lines, box, em, gap, color, centered, image_indexes, fallback_text, top,
        justify_last, left, lanes, first_room,
    )
    if placed is None:
        return None
    commands, slots, fonts = placed
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
    centered: bool,
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
            justify=not centered and (justify_last or index != last),
            centered=centered,
            image_indexes=image_indexes,
            fallback_text=fallback_text,
            opened=opened,
            color=color,
        )
        if row is None:
            return None
        ops, line_slots, line_fonts, line_rules = row
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


def _place_line(
    line: TypesetLine,
    x0: float,
    width: float,
    em: float,
    justify: bool,
    centered: bool,
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
    extras = _line_extras(texts, slack, em) if justify and slack > 0 else []
    cursor = x0 + (width - total) / 2 if centered and total < width else x0
    ops: list[str] = []
    slots: list[tuple[int, float]] = []
    rules: list[tuple[float, float, float, str]] = []
    extra_at = 0
    face_now: tuple[str, float] | None = None
    paint_now = ""
    for segments, slot, piece_width, visible in pieces:
        if slot is not None:
            slots.append((slot, cursor))
            cursor += piece_width
            continue
        count = max(len(visible) - 1, 0)
        piece_extras = extras[extra_at:extra_at + count]
        extra_at += count
        pos = 0
        for seg in segments:
            n = len(seg.text)
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


def _line_extras(texts: list[str | None], slack: float, em: float) -> list[float]:
    """把整行的余量摊进各段文字。公式槽是原子，不在这里拆。"""
    slots = 0
    for text in texts:
        if not text:
            continue
        slots += sum(1 for i in range(len(text) - 1) if can_stretch(text[i], text[i + 1]))
    if em <= 0 or slack < em * _MIN_SLACK_EM or slots < _MIN_SLOTS:
        return []
    each = min(slack / slots, em * _MAX_GAP_EM)
    if each < 0.05:
        return []
    extras: list[float] = []
    for text in texts:
        if not text or len(text) < 2:
            continue
        for index in range(len(text) - 1):
            extras.append(each if can_stretch(text[index], text[index + 1]) else 0.0)
    return extras


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
