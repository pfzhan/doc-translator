"""中文行的两端对齐。

MuPDF 的 letter-spacing、word-spacing、text-align:justify 都拉不动纯汉字行。
这里按字形自己写文本：非末行的余量摊到字与字之间，标点前面不拉开。
复制出来的仍是原来的字，不含为对齐插入的空格。
斜体标记不改走 HTML：汉字没有斜体字形，仍用正文字体；拉丁字母换斜体字体。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pymupdf

from .pdf_typeset import FormulaPiece, TextPiece, TypesetLine

# htmlbox 首行 origin 相对盒子顶的比例。和它对齐，换写法后行高不变。
FIRST_BASELINE = 1.108
_FONT_NAME = "DocCJK"
_MIN_SLOTS = 6
_MIN_SLACK_EM = 0.15
_MAX_GAP_EM = 0.35
# 句读贴着前一个字，开括号贴着后一个字。空格可以拉开，公式后的分界就靠它。
_NO_GAP_BEFORE = frozenset("。，、；：？！）】」』〉》％%.,;:?!)]}>")
_NO_GAP_AFTER = frozenset("（【「『《〈([{")

_MARK_RE = re.compile(r"\{i\}|\{/i\}|\{b\}|\{/b\}|\{/z\}|\{z(\d+)\}")
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


@dataclass(frozen=True)
class _Segment:
    text: str
    font_name: str
    font: pymupdf.Font
    size: float

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
    """按 {i} {b} {zN} 切开。标记本身不占字。"""
    italic = False
    bold = False
    size = 100
    runs: list[_Run] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            runs.append(_Run("".join(buf), italic, bold, size))
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
        elif token == "{/z}":
            size = 100
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
) -> None:
    if not buf or face is None:
        return
    name, font = face
    text = "".join(buf)
    if segments and segments[-1].font_name == name and segments[-1].size == size:
        prev = segments[-1]
        segments[-1] = _Segment(prev.text + text, name, font, size)
    else:
        segments.append(_Segment(text, name, font, size))
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
        for char in run.text:
            picked = _face_for(char, run.italic, run.bold)
            if picked is None:
                return None
            if face is not None and picked != face:
                _append_segment(segments, buf, face, size)
            face = picked
            buf.append(char)
        _append_segment(segments, buf, face, size)
    return segments


def _latin_unit(ch: str) -> bool:
    """词内的拉丁字母、数字和连字符。两端对齐不能从这里拉开。"""
    return ch.isascii() and (ch.isalnum() or ch in "-'’")


def can_stretch(left: str, right: str) -> bool:
    if right in _NO_GAP_BEFORE or left in _NO_GAP_AFTER:
        return False
    # 作者名被拉成 Z h o n g，就是把词内的字母也拉开了。只在词与词之间留缝。
    if _latin_unit(left) and _latin_unit(right):
        return False
    return True


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
) -> list[FormulaSlot] | None:
    """写成两端对齐的行。缺字时返回 None，调用方改走 HTML 盒子。

    段落还要接到下一页时，这一页的末行也是满行，同样两端对齐。
    """
    placed = _place_lines(
        # 页顶是 mediabox 的上沿。向下加长页面时页高变了，页顶不变，不能用 page.rect.height。
        lines, box, em, gap, color, centered, image_indexes, fallback_text, page.mediabox.y1,
        justify_last,
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
    page_height: float,
    justify_last: bool = False,
) -> tuple[str, list[FormulaSlot], dict[str, pymupdf.Font]] | None:
    red, green, blue = _rgb(color)
    chunks: list[str] = [
        "q",
        "BT",
        f"{red:.4f} {green:.4f} {blue:.4f} rg",
    ]
    slots: list[FormulaSlot] = []
    fonts: dict[str, pymupdf.Font] = {}
    last = len(lines) - 1
    wrote = False
    for index, line in enumerate(lines):
        baseline = box.y0 + em * FIRST_BASELINE + index * em * gap
        row = _place_line(
            line, box.x0, box.width, em,
            justify=not centered and (justify_last or index != last),
            centered=centered,
            image_indexes=image_indexes,
            fallback_text=fallback_text,
        )
        if row is None:
            return None
        ops, line_slots, line_fonts = row
        fonts.update(line_fonts)
        if ops:
            wrote = True
            pdf_y = page_height - baseline
            chunks.extend(op.replace("{Y}", f"{pdf_y:.2f}") for op in ops)
        for slot_index, slot_x in line_slots:
            slots.append(FormulaSlot(slot_index, slot_x, baseline))
    if not wrote and not slots:
        return "", [], {}
    chunks.append("ET")
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
) -> tuple[list[str], list[tuple[int, float]], dict[str, pymupdf.Font]] | None:
    """一行的文本操作（基线 y 先写成 {Y}）和公式槽的横坐标。"""
    pieces: list[tuple[list[_Segment], int | None, float, str]] = []
    fonts: dict[str, pymupdf.Font] = {}
    for piece in line.pieces:
        drawn = _draw_piece(piece, em, image_indexes, fallback_text)
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
    extra_at = 0
    face_now: tuple[str, float] | None = None
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
            ops.append(f"1 0 0 1 {cursor:.2f} {{Y}} Tm")
            ops.append(_tj(glyphs, internal, seg.size))
            advance = seg.width + sum(internal)
            pos += n
            if pos < len(visible) and pos - 1 < len(piece_extras):
                advance += piece_extras[pos - 1]
            cursor += advance
    return ops, slots, fonts


def _draw_piece(
    piece: TextPiece | FormulaPiece,
    em: float,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
) -> tuple[list[_Segment], int | None] | None:
    """文字片返回片段。公式图返回空片段和占位符序号。缺字返回 None。"""
    if isinstance(piece, FormulaPiece) and piece.index in image_indexes:
        return [], piece.index
    text = piece.text if isinstance(piece, TextPiece) else fallback_text.get(piece.index, "")
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
