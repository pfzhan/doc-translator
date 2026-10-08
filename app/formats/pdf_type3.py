"""Type3 点阵字没有 Unicode 时，按字形把字母认回来。

旧版 Ghostscript 把每个字画成位图，编码名是 /a0、/a1，没有 ToUnicode。
提取出来的是字符码，不是页面上的字母。整页用 tesseract 认字形，再按词重新组行。
点阵字的行框和词对不齐，词心会落在碎片之间，所以不能把词填回旧行框。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

_CUSTOM_NAME = re.compile(r"^[A-Za-z]\d+$")
_SECTION_NO = re.compile(r"\d+(?:\.\d+)*\.?")
# 坏字体占这一页文字的比例。零星符号字体不整页重认。
_BROKEN_SHARE = 0.4
_MIN_BROKEN_CHARS = 20
_OCR_SCALE = 3.0
_MIN_CONF = 15.0
# 词心和点阵字框中心最多差这么多，仍算同一行。正文行距大约 12pt。
_BAND = 8.0
_X_SLACK = 2.0
# 行距内的下一行并进同一块。更大的空隙是标题、署名或新的一节。
_LINE_GAP = 8.0
# PyMuPDF get_text("dict") 的结构。字段随版本变化，只在这层进出。
PageData = dict[str, Any]


@dataclass(frozen=True)
class OcrWord:
    text: str
    rect: pymupdf.Rect
    conf: float


@dataclass(frozen=True)
class GlyphBox:
    """丢弃字符码之前留下的点阵字框，用来对上认出来的词，并取颜色和基线。"""

    rect: pymupdf.Rect
    em: float
    color: int
    origin_y: float
    flags: int


def visual_em(span: dict[str, Any]) -> float:
    """点阵字报出来的 size 是缩放系数，不是磅值。用字框高度换回一个参考磅值。"""
    size = float(span.get("size") or 0)
    ascender = float(span.get("ascender") or 0)
    descender = float(span.get("descender") or 0)
    if 0 < size < 1.5 and ascender > 2:
        box = size * (ascender - descender) if descender < 0 else size * ascender
        return round(min(max(box * 0.72, 6.0), 36.0), 1)
    return round(size, 1) if size else 0.0


def broken_type3_names(doc: pymupdf.Document, page: pymupdf.Page) -> set[str]:
    """没有 ToUnicode、字形名又是 /a0 这种自造名的 Type3 字体。"""
    names: set[str] = set()
    for xref, _ext, subtype, name, _ref, _encoding in page.get_fonts():
        if subtype != "Type3" or not name or not _custom_encoding(doc, xref):
            continue
        names.add(name)
    return names


def segment_ocr_words(words: list[OcrWord]) -> list[list[OcrWord]]:
    """先按基线分成行，行内空隙明显大于词距时再切开。

    点阵字的行框和真正的词对不齐，词心会落在碎片之间。按认出来的词重新组行。
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda word: ((word.rect.y0 + word.rect.y1) / 2, word.rect.x0))
    lines: list[list[OcrWord]] = []
    for word in ordered:
        cy = (word.rect.y0 + word.rect.y1) / 2
        if lines:
            last = lines[-1]
            ly = sum((item.rect.y0 + item.rect.y1) / 2 for item in last) / len(last)
            limit = max(word.rect.height, last[0].rect.height, 8.0) * 0.55
            if abs(cy - ly) <= limit:
                last.append(word)
                continue
        lines.append([word])
    segments: list[list[OcrWord]] = []
    for line in lines:
        line.sort(key=lambda word: word.rect.x0)
        current = [line[0]]
        for prev, word in zip(line, line[1:]):
            gap = word.rect.x0 - prev.rect.x1
            # 节号和标题之间常空出 15–20pt。署名和栏沟更大，仍切开。
            if _lone_section_number(current) and gap <= 22.0:
                current.append(word)
            elif gap > max(prev.rect.height * 1.1, 12.0):
                segments.append(current)
                current = [word]
            else:
                current.append(word)
        segments.append(current)
    return segments


def recover_type3_page(page: pymupdf.Page, data: PageData) -> bool:
    """这一页大部分是自造编码的点阵字时，用认出来的行换掉字符码。

    旧行框盖不住词，所以按字形重新组行。擦除用墨迹框：字框比墨迹高，
    扩进去会压到下一行。没对上点阵的词留下，避免把正常字体再认一遍。
    """
    doc = page.parent
    if doc is None or "blocks" not in data:
        return False
    names = broken_type3_names(doc, page)
    if not names or _broken_share(data, names) < _BROKEN_SHARE:
        return False
    words = ocr_page(page)
    originals = _broken_lines(data, names)
    _drop_broken_lines(data, names)
    rebuilt: list[dict[str, Any]] = []
    for segment in segment_ocr_words(words):
        if not segment:
            continue
        line = _line_from_words(segment, originals)
        if line is not None:
            rebuilt.append(line)
    if not rebuilt:
        raise ValueError("点阵字页面没有认出可翻译的文字。请确认 tesseract 的英文语言包已安装")
    for group in _group_lines(rebuilt):
        data["blocks"].append({"type": 0, "bbox": _union_bbox(group), "lines": group})
    return True


def ocr_page(page: pymupdf.Page, scale: float = _OCR_SCALE) -> list[OcrWord]:
    exe = shutil.which("tesseract")
    if exe is None:
        raise ValueError("这份 PDF 的文字是没有 Unicode 的点阵字，需要安装 tesseract 才能读出原文")
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        pix.save(path)
        proc = subprocess.run(
            [exe, str(path), "stdout", "--psm", "3", "-l", "eng", "tsv"],
            capture_output=True, text=True, timeout=90, check=False,
        )
        if not proc.stdout.strip():
            proc = subprocess.run(
                [exe, str(path), "stdout", "--psm", "3", "tsv"],
                capture_output=True, text=True, timeout=90, check=False,
            )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)
    if not proc.stdout.strip():
        raise ValueError("点阵字页面没有认出文字。请确认 tesseract 的英文语言包已安装")
    return _parse_tsv(proc.stdout, scale)


def _lone_section_number(words: list[OcrWord]) -> bool:
    """这一段目前只有一个节号。行尾的年份不是节号，不能把下一栏吸进来。"""
    if len(words) != 1:
        return False
    return bool(_SECTION_NO.fullmatch(words[0].text.strip()))


def _custom_encoding(doc: pymupdf.Document, xref: int) -> bool:
    if doc.xref_get_key(xref, "ToUnicode")[0] != "null":
        return False
    kind, value = doc.xref_get_key(xref, "Encoding")
    if kind == "name":
        return False
    if kind != "xref":
        return True
    enc_xref = int(value.split()[0])
    names = re.findall(r"/([A-Za-z][A-Za-z0-9.]*)", doc.xref_object(enc_xref))
    custom = [name for name in names if _CUSTOM_NAME.fullmatch(name)]
    return len(custom) >= 8


def _broken_share(data: PageData, names: set[str]) -> float:
    total = 0
    broken = 0
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                count = len(str(span.get("text") or ""))
                total += count
                if span.get("font") in names:
                    broken += count
    if broken < _MIN_BROKEN_CHARS or total == 0:
        return 0.0
    return broken / total


def _broken_lines(data: PageData, names: set[str]) -> list[GlyphBox]:
    """丢掉字符码之前，留下点阵字框。零宽码没有墨迹，不参与对位。"""
    found: list[GlyphBox] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span.get("font") not in names:
                    continue
                raw = span.get("bbox")
                if not raw:
                    continue
                box = pymupdf.Rect(raw)
                if box.is_empty or box.width < 0.4:
                    continue
                origin = span.get("origin")
                origin_y = float(origin[1]) if origin and len(origin) >= 2 else box.y1
                found.append(GlyphBox(
                    rect=box,
                    em=visual_em(span),
                    color=_as_int(span.get("color")),
                    origin_y=origin_y,
                    flags=_as_int(span.get("flags")),
                ))
    return found


def _drop_broken_lines(data: PageData, names: set[str]) -> None:
    """删掉字符码。图片和正常字体留下。混在同一行里的正常字也留下。"""
    kept: list[dict[str, Any]] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            kept.append(block)
            continue
        lines: list[dict[str, Any]] = []
        for line in block["lines"]:
            spans = [span for span in line["spans"] if span.get("font") not in names]
            if not spans:
                continue
            if len(spans) != len(line["spans"]):
                line = dict(line)
                line["spans"] = spans
                line["bbox"] = _union_bbox([{"bbox": span["bbox"]} for span in spans if span.get("bbox")])
            lines.append(line)
        if not lines:
            continue
        block["lines"] = lines
        kept.append(block)
    data["blocks"] = kept


def _line_from_words(segment: list[OcrWord], originals: list[GlyphBox]) -> dict[str, Any] | None:
    """一组词合成一行。没盖住点阵字框的词不是这页的坏字体，不另起一行。"""
    words = _dedupe(sorted(segment, key=lambda word: word.rect.x0))
    if not words:
        return None
    text = " ".join(word.text for word in words).strip()
    if not text:
        return None
    ink = pymupdf.Rect()
    for word in words:
        ink |= word.rect
    matched = [glyph for glyph in originals if _same_band(glyph.rect, ink) and _x_hits(glyph.rect, ink)]
    if not matched:
        return None
    pad = 0.6
    box = pymupdf.Rect(ink.x0 - pad, ink.y0 - pad, ink.x1 + pad, ink.y1 + pad)
    glyph = max(matched, key=lambda item: _overlap_width(item.rect, ink))
    size = _line_size(text, ink.height, glyph.em)
    bbox = (box.x0, box.y0, box.x1, box.y1)
    span: dict[str, Any] = {
        "text": text,
        "bbox": bbox,
        "size": size,
        "font": "Times-Roman",
        "flags": glyph.flags,
        "color": glyph.color,
        "origin": (box.x0, glyph.origin_y),
        "ascender": 0.8,
        "descender": -0.2,
    }
    return {"spans": [span], "wmode": 0, "dir": (1.0, 0.0), "bbox": bbox}


def _group_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """行距内、同一栏的行并成一块，后面的分段才接得上。并排的署名不并。"""
    ordered = sorted(lines, key=lambda line: (float(line["bbox"][1]), float(line["bbox"][0])))
    groups: list[list[dict[str, Any]]] = []
    for line in ordered:
        box = pymupdf.Rect(line["bbox"])
        best: tuple[float, list[dict[str, Any]]] | None = None
        for group in groups:
            prev = pymupdf.Rect(group[-1]["bbox"])
            gap = box.y0 - prev.y1
            if gap > max(prev.height * 0.9, _LINE_GAP) or not _x_aligned(prev, box):
                continue
            if best is None or gap < best[0]:
                best = (gap, group)
        if best is None:
            groups.append([line])
        else:
            best[1].append(line)
    return groups


def _ink_size(text: str, height: float) -> float:
    """墨迹高度换成磅值。同一套点阵字的字框一样高，画出的标题和正文并不一样大。"""
    if height <= 0:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    has_desc = any(ch in "gjpqy" for ch in letters)
    has_asc = any(ch.isupper() or ch in "bdfhiklt" for ch in letters)
    if has_desc:
        factor = 0.95
    elif has_asc:
        factor = 0.72
    else:
        factor = 0.52
    return round(min(max(height / factor, 6.0), 36.0), 1)


def _line_size(text: str, ink_height: float, em: float) -> float:
    guessed = _ink_size(text, ink_height)
    if em > 0 and guessed > em * 1.45:
        return round(em * 1.45, 1)
    return guessed


def _same_band(glyph: pymupdf.Rect, ink: pymupdf.Rect) -> bool:
    glyph_y = (glyph.y0 + glyph.y1) / 2
    ink_y = (ink.y0 + ink.y1) / 2
    return abs(glyph_y - ink_y) <= _BAND


def _x_hits(glyph: pymupdf.Rect, ink: pymupdf.Rect) -> bool:
    return glyph.x1 >= ink.x0 - _X_SLACK and glyph.x0 <= ink.x1 + _X_SLACK


def _overlap_width(glyph: pymupdf.Rect, ink: pymupdf.Rect) -> float:
    return min(glyph.x1, ink.x1) - max(glyph.x0, ink.x0)


def _x_aligned(prev: pymupdf.Rect, nxt: pymupdf.Rect) -> bool:
    overlap = min(prev.x1, nxt.x1) - max(prev.x0, nxt.x0)
    if overlap > min(prev.width, nxt.width) * 0.3:
        return True
    return abs(nxt.x0 - prev.x0) <= max(prev.height, 12.0) * 2


def _union_bbox(lines: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    box = pymupdf.Rect()
    for line in lines:
        raw = line.get("bbox")
        if raw:
            box |= pymupdf.Rect(raw)
    return (box.x0, box.y0, box.x1, box.y1)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _dedupe(words: list[OcrWord]) -> list[OcrWord]:
    ordered = sorted(words, key=lambda word: (word.rect.x0, -word.conf))
    kept: list[OcrWord] = []
    for word in ordered:
        if kept and word.rect.x0 < kept[-1].rect.x1 - 1:
            if word.conf > kept[-1].conf:
                kept[-1] = word
            continue
        kept.append(word)
    return kept


def _parse_tsv(raw: str, scale: float) -> list[OcrWord]:
    words: list[OcrWord] = []
    for row in raw.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        text = parts[11].strip()
        if not text:
            continue
        try:
            conf = float(parts[10])
            left, top, width, height = (float(parts[index]) for index in (6, 7, 8, 9))
        except ValueError:
            continue
        if conf < _MIN_CONF or width < 1 or height < 1 or scale <= 0:
            continue
        words.append(OcrWord(
            text,
            pymupdf.Rect(left / scale, top / scale, (left + width) / scale, (top + height) / scale),
            conf,
        ))
    return words
