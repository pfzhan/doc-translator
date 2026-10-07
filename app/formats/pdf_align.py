"""中文行的两端对齐。

MuPDF 的 letter-spacing、word-spacing、text-align:justify 都拉不动纯汉字行。
这里按字形自己写文本：非末行的余量摊到字与字之间，标点前面不拉开。
复制出来的仍是原来的字，不含为对齐插入的空格。
"""
from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from .pdf_flow import STYLE_MARK_RE
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

_font: pymupdf.Font | None = None


@dataclass(frozen=True)
class FormulaSlot:
    index: int
    x: float
    baseline: float


def cjk_font() -> pymupdf.Font:
    global _font
    if _font is None:
        _font = pymupdf.Font("china-ss")
    return _font


def can_stretch(left: str, right: str) -> bool:
    return right not in _NO_GAP_BEFORE and left not in _NO_GAP_AFTER


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
) -> list[FormulaSlot] | None:
    """写成两端对齐的行。缺字或带样式标记时返回 None，调用方改走 HTML 盒子。"""
    font = cjk_font()
    placed = _place_lines(
        lines, box, em, gap, color, centered, font, image_indexes, fallback_text, page.rect.height,
    )
    if placed is None:
        return None
    commands, slots = placed
    if commands:
        page.insert_font(fontname=_FONT_NAME, fontbuffer=font.buffer)
        _append_contents(page, doc, commands)
    return slots


def _place_lines(
    lines: list[TypesetLine],
    box: pymupdf.Rect,
    em: float,
    gap: float,
    color: str,
    centered: bool,
    font: pymupdf.Font,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
    page_height: float,
) -> tuple[str, list[FormulaSlot]] | None:
    red, green, blue = _rgb(color)
    chunks: list[str] = [
        "q",
        "BT",
        f"{red:.4f} {green:.4f} {blue:.4f} rg",
        f"/{_FONT_NAME} {em:.2f} Tf",
    ]
    slots: list[FormulaSlot] = []
    last = len(lines) - 1
    wrote = False
    for index, line in enumerate(lines):
        baseline = box.y0 + em * FIRST_BASELINE + index * em * gap
        row = _place_line(
            line, box.x0, box.width, em, font,
            justify=not centered and index != last,
            centered=centered,
            image_indexes=image_indexes,
            fallback_text=fallback_text,
        )
        if row is None:
            return None
        ops, line_slots = row
        if ops:
            wrote = True
            pdf_y = page_height - baseline
            chunks.extend(op.replace("{Y}", f"{pdf_y:.2f}") for op in ops)
        for slot_index, slot_x in line_slots:
            slots.append(FormulaSlot(slot_index, slot_x, baseline))
    if not wrote and not slots:
        return "", []
    chunks.append("ET")
    chunks.append("Q")
    return "\n".join(chunks) + "\n", slots


def _place_line(
    line: TypesetLine,
    x0: float,
    width: float,
    em: float,
    font: pymupdf.Font,
    justify: bool,
    centered: bool,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
) -> tuple[list[str], list[tuple[int, float]]] | None:
    """一行的文本操作（基线 y 先写成 {Y}）和公式槽的横坐标。"""
    for piece in line.pieces:
        if isinstance(piece, TextPiece) and (STYLE_MARK_RE.search(piece.text) or "\x01" in piece.text):
            return None
    parts = [_visible(piece, image_indexes, fallback_text) for piece in line.pieces]
    texts: list[str | None] = []
    widths: list[float] = []
    for piece, text in zip(line.pieces, parts):
        if text is None:
            if not isinstance(piece, FormulaPiece):
                return None
            texts.append(None)
            widths.append(piece.slot_width)
            continue
        if "\x01" in text or _glyphs(font, text) is None:
            return None
        texts.append(text)
        widths.append(font.text_length(text, em) if text else 0.0)
    total = sum(widths)
    slack = width - total
    extras = _line_extras(texts, slack, em) if justify and slack > 0 else []
    cursor = x0 + (width - total) / 2 if centered and total < width else x0
    ops: list[str] = []
    slots: list[tuple[int, float]] = []
    extra_at = 0
    for piece, text, piece_width in zip(line.pieces, texts, widths):
        if text is None:
            if isinstance(piece, FormulaPiece):
                slots.append((piece.index, cursor))
            cursor += piece_width
            continue
        count = max(len(text) - 1, 0)
        piece_extras = extras[extra_at:extra_at + count]
        extra_at += count
        glyphs = _glyphs(font, text)
        if glyphs is None:
            return None
        if text:
            ops.append(f"1 0 0 1 {cursor:.2f} {{Y}} Tm")
            ops.append(_tj(glyphs, piece_extras, em))
        cursor += piece_width + sum(piece_extras)
    return ops, slots


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


def _visible(
    piece: TextPiece | FormulaPiece,
    image_indexes: frozenset[int],
    fallback_text: dict[int, str],
) -> str | None:
    """文字片返回要写的字。有图的公式返回 None，表示这里是一个槽。"""
    if isinstance(piece, TextPiece):
        return STYLE_MARK_RE.sub("", piece.text)
    if piece.index in image_indexes:
        return None
    return fallback_text.get(piece.index, "")


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
