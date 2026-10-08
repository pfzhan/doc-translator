"""Type3 点阵字没有 Unicode 时，按字形把字母认回来。

旧版 Ghostscript 把每个字画成位图，编码名是 /a0、/a1，没有 ToUnicode。
提取出来的是字符码，不是页面上的字母。整页用 tesseract 认字形，再按词重新组行。
点阵字的行框和词对不齐，词心会落在碎片之间，所以不能把词填回旧行框。

公式位图会被认成乱码。并进下一句后，写入框盖住公式，译文再压到下一段。
这类词留下原字形。矢量格子上的节点标签，以及用竖线字形画出的表，也不重排。
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
_NUMBERED_LINE = re.compile(r"^\d+(?:\.\d+)*[.,]?\s+\S")
_SQL_LINE = re.compile(r"^(?:SELECT|FROM|WHERE|GROUP|ORDER|HAVING|INSERT|UPDATE|DELETE)\b")
_PIPE_TEXT = re.compile(r"^[|¦│]+$")
_GARBAGE_MARK = set("®∑∏√∞≤≥∈≈≠±×÷@#$^~\\¢£¥§")
# 弯引号和破折号是正文。单独一个破折号不能当成公式乱码丢掉，否则这一句从中间裂开。
_KEEP_MARK = set("‘’“”–—−‐‑‒")
# 矢量笔画靠得这么近，就是同一张图，不是页边的一根线。
_CLUSTER_GAP = 18.0
_ZONE_PAD_X = 16.0
_ZONE_PAD_TOP = 6.0
_ZONE_PAD_BOTTOM = 16.0
_MIN_CLUSTER = 3
_MIN_ZONE_AREA = 2000.0
# 公式位图的词置信度很低。节号偶尔也低，不能一起丢掉。
_NOISE_CONF = 40.0
_JUNK_CONF = 22.0
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
class _Zone:
    """labels_only 只挡短标签。整句正文可以穿过，表和图则整块留下。"""

    rect: pymupdf.Rect
    labels_only: bool = False


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
    zones = [*_drawing_zones(page), *_pipe_zones(words)]
    zones.extend(_label_zones(words, zones))
    words = [word for word in _without_noise(words) if not _hidden(word, zones)]
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


def _without_noise(words: list[OcrWord]) -> list[OcrWord]:
    """丢掉公式位图认出来的乱码，以及贴在乱码旁边的短标点。正文和节号留下。"""
    noise = [word for word in words if _is_noise(word)]
    if not noise:
        return list(words)
    kept: list[OcrWord] = []
    for word in words:
        if _is_noise(word) or _glued_to_noise(word, noise):
            continue
        kept.append(word)
    return kept


def _is_noise(word: OcrWord) -> bool:
    text = word.text.strip()
    if not text or _SECTION_NO.fullmatch(text):
        return False
    if any(ch in _GARBAGE_MARK or (ord(ch) > 127 and ch not in _KEEP_MARK) for ch in text):
        return True
    if word.conf < _JUNK_CONF:
        return True
    if word.conf < _NOISE_CONF and not _plain_token(text):
        return True
    return False


def _plain_token(text: str) -> bool:
    """低置信度里仍可能是正文的：单字母变量、数字、节号、普通单词。"""
    core = text.strip(".,;:()[]")
    if not core:
        return False
    if _SECTION_NO.fullmatch(core) or core.isdigit():
        return True
    if core.isalpha() and len(core) == 1:
        return True
    return len(core) >= 3 and core.replace("-", "").isalpha()


def _glued_to_noise(word: OcrWord, noise: list[OcrWord]) -> bool:
    """乱码旁边的 `(p,`、`c),` 仍是公式的一部分。节号和表格竖线不是。"""
    if not _is_fringe(word):
        return False
    cy = (word.rect.y0 + word.rect.y1) / 2
    for other in noise:
        oy = (other.rect.y0 + other.rect.y1) / 2
        band = max(word.rect.height, other.rect.height, 8.0) * 0.6
        if abs(cy - oy) > band:
            continue
        gap = word.rect.x0 - other.rect.x1 if word.rect.x0 >= other.rect.x0 else other.rect.x0 - word.rect.x1
        if gap <= 8:
            return True
    return False


def _is_fringe(word: OcrWord) -> bool:
    text = word.text.strip()
    if not text or _SECTION_NO.fullmatch(text) or _PIPE_TEXT.fullmatch(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    return len(text) <= 4 and letters <= 1


def _drawing_zones(page: pymupdf.Page) -> list[_Zone]:
    """细笔画凑成的格子、坐标图。单根下划线不成区，节点标签落在区里就留下原字。"""
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - 个别页面的矢量读不出来，当没有图
        return []
    rects: list[pymupdf.Rect] = []
    width, height = page.rect.width, page.rect.height
    for drawing in drawings:
        if _blank_fill(drawing):
            continue
        rect = pymupdf.Rect(drawing.get("rect"))
        if rect.width < 0.4 and rect.height < 4:
            continue
        if rect.height < 0.4 and rect.width < 4:
            continue
        if rect.width > width * 0.85 and rect.height > height * 0.85:
            continue
        rects.append(rect)
    return _cluster_drawings(rects, width, height)


def _cluster_drawings(
    rects: list[pymupdf.Rect], page_width: float, page_height: float,
) -> list[_Zone]:
    """靠在一起的笔画收成一块。标签贴在笔画外侧几个点，框要稍向外扩。"""
    if len(rects) < _MIN_CLUSTER:
        return []
    parent = list(range(len(rects)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in _close_pairs(rects):
        parent[find(left)] = find(right)
    groups: dict[int, list[pymupdf.Rect]] = {}
    for index, rect in enumerate(rects):
        groups.setdefault(find(index), []).append(rect)
    zones: list[_Zone] = []
    for items in groups.values():
        if len(items) < _MIN_CLUSTER:
            continue
        box = pymupdf.Rect()
        for item in items:
            box |= item
        if box.width * box.height < _MIN_ZONE_AREA:
            continue
        padded = pymupdf.Rect(
            box.x0 - _ZONE_PAD_X, box.y0 - _ZONE_PAD_TOP,
            box.x1 + _ZONE_PAD_X, box.y1 + _ZONE_PAD_BOTTOM,
        )
        if page_width > 0 and padded.width > page_width * 0.92 and padded.height > page_height * 0.75:
            continue
        zones.append(_Zone(padded))
    return zones


def _close_pairs(rects: list[pymupdf.Rect]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for index, rect in enumerate(rects):
        for other_index in range(index + 1, len(rects)):
            other = rects[other_index]
            gap_x = max(0.0, max(rect.x0, other.x0) - min(rect.x1, other.x1))
            gap_y = max(0.0, max(rect.y0, other.y0) - min(rect.y1, other.y1))
            if gap_x <= _CLUSTER_GAP and gap_y <= _CLUSTER_GAP:
                pairs.append((index, other_index))
    return pairs


def _blank_fill(drawing: dict[str, Any]) -> bool:
    if drawing.get("type") != "f":
        return False
    fill = drawing.get("fill")
    if not isinstance(fill, (tuple, list)) or len(fill) < 3:
        return False
    return all(float(channel) >= 0.96 for channel in fill[:3])


def _pipe_zones(words: list[OcrWord]) -> list[_Zone]:
    """竖线字形围出的表。线不是矢量，按格拆不开，整表留下原字。"""
    pipes = [word for word in words if _PIPE_TEXT.fullmatch(word.text.strip())]
    columns = _vertical_pipe_columns(pipes, 4, 36.0)
    if not columns:
        return []
    related = [pipe for column in columns for pipe in column]
    seen = {id(pipe) for pipe in related}
    baselines = [(pipe.rect.y0 + pipe.rect.y1) / 2 for pipe in related]
    for pipe in pipes:
        if id(pipe) in seen:
            continue
        cy = (pipe.rect.y0 + pipe.rect.y1) / 2
        if any(abs(cy - base) <= 6 for base in baselines):
            related.append(pipe)
            seen.add(id(pipe))
    box = pymupdf.Rect()
    for pipe in related:
        box |= pipe.rect
    if box.height > 520:
        return []
    xs = sorted({round(pipe.rect.x0, 1) for pipe in related})
    gaps = [right - left for left, right in zip(xs, xs[1:]) if right - left >= 8]
    pitch = sorted(gaps)[len(gaps) // 2] if gaps else 64.0
    rect = pymupdf.Rect(box.x0 - pitch - 8, box.y0 - 4, box.x1 + pitch + 12, box.y1 + 4)
    return [_Zone(rect)]


def _label_zones(words: list[OcrWord], covered: list[_Zone]) -> list[_Zone]:
    """两三根竖线是格子的边。旁边的短标签留下原字，整句正文不挡。"""
    pipes = [word for word in words if _PIPE_TEXT.fullmatch(word.text.strip())]
    zones: list[_Zone] = []
    for column in _vertical_pipe_columns(pipes, 2, 16.0):
        box = pymupdf.Rect()
        for pipe in column:
            box |= pipe.rect
        padded = pymupdf.Rect(box.x0 - 14, box.y0 - 8, box.x1 + 14, box.y1 + 14)
        if any(_mostly_inside(padded, zone.rect) for zone in covered):
            continue
        zones.append(_Zone(padded, labels_only=True))
    return zones


def _vertical_pipe_columns(
    pipes: list[OcrWord], min_count: int, min_span: float,
) -> list[list[OcrWord]]:
    columns: list[list[OcrWord]] = []
    for pipe in sorted(pipes, key=lambda item: item.rect.x0):
        placed = False
        for column in columns:
            anchor = sum(item.rect.x0 for item in column) / len(column)
            if abs(pipe.rect.x0 - anchor) <= 5:
                column.append(pipe)
                placed = True
                break
        if not placed:
            columns.append([pipe])
    kept: list[list[OcrWord]] = []
    for column in columns:
        if len(column) < min_count:
            continue
        top = min(pipe.rect.y0 for pipe in column)
        bottom = max(pipe.rect.y1 for pipe in column)
        if bottom - top < min_span:
            continue
        kept.append(column)
    return kept


def _hidden(word: OcrWord, zones: list[_Zone]) -> bool:
    if not zones:
        return False
    cx = (word.rect.x0 + word.rect.x1) / 2
    cy = (word.rect.y0 + word.rect.y1) / 2
    short = word.rect.width <= 48 and len(word.text.strip()) <= 8
    for zone in zones:
        if not (zone.rect.x0 <= cx <= zone.rect.x1 and zone.rect.y0 <= cy <= zone.rect.y1):
            continue
        if zone.labels_only and not short:
            continue
        return True
    return False


def _mostly_inside(inner: pymupdf.Rect, outer: pymupdf.Rect) -> bool:
    hit = inner & outer
    if hit.is_empty or inner.width < 0.4 or inner.height < 0.4:
        return False
    return hit.width * hit.height >= inner.width * inner.height * 0.6


def _stands_alone(line: dict[str, Any]) -> bool:
    """编号标题、列表项、SQL 行不吞下一行，也不被上一行吞掉。"""
    text = _span_text(line)
    if not text:
        return False
    if _SQL_LINE.match(text):
        return True
    return bool(_NUMBERED_LINE.match(text) and len(text) <= 90)


def _starts_after_hole(prev: pymupdf.Rect, nxt: pymupdf.Rect) -> bool:
    """从远远的右侧起笔。左边是没认出来的公式，悬挂缩进没有这么宽。

    同一基线上的后半句不是这种情况。破折号后面的列举要留在这一句里。
    """
    if _same_baseline(prev, nxt):
        return False
    return nxt.x0 > prev.x0 + 40


def _returns_past_hole(prev: pymupdf.Rect, nxt: pymupdf.Rect) -> bool:
    """下一行回到左缘，上一行却从公式右边起笔。并进去会把公式框进写入范围。"""
    if _same_baseline(prev, nxt):
        return False
    return nxt.x0 < prev.x0 - 40


def _covers_right_neighbor(
    prev: pymupdf.Rect, nxt: pymupdf.Rect, pending: list[pymupdf.Rect],
) -> bool:
    """上一行只是公式左边的半句，右边还有另一半。下一整行并进来会盖住那一半。"""
    if prev.width > 160 or nxt.x0 > prev.x0 + 24:
        return False
    for other in pending:
        if other.x0 <= prev.x1 + 4:
            continue
        if _same_baseline(prev, other) and other.x0 - prev.x1 <= 80:
            return True
    return False


def _same_baseline(prev: pymupdf.Rect, nxt: pymupdf.Rect) -> bool:
    overlap = min(prev.y1, nxt.y1) - max(prev.y0, nxt.y0)
    height = min(prev.height, nxt.height)
    return height > 0 and overlap > height * 0.4


def _snap_group_size(group: list[dict[str, Any]]) -> None:
    """同一段里墨迹高低不一，不能因此写出字号标记。差得太多的是另一级标题，不动。"""
    sizes = [
        float(span["size"])
        for line in group
        for span in line.get("spans") or []
        if isinstance(span.get("size"), (int, float)) and float(span["size"]) > 0
    ]
    if len(sizes) < 2:
        return
    low, high = min(sizes), max(sizes)
    if high > low * 1.35:
        return
    median = sorted(sizes)[len(sizes) // 2]
    for line in group:
        for span in line.get("spans") or []:
            if span.get("size"):
                span["size"] = median


def _span_text(line: dict[str, Any]) -> str:
    return " ".join(str(span.get("text") or "") for span in line.get("spans") or []).strip()


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
    """行距内、同一栏的行并成一块，后面的分段才接得上。并排的署名不并。

    编号标题、列表项和 SQL 行各自一块。公式左边没认出来时，右侧半句不能并进
    上一行：并进去的写入框会盖住公式，再压到下一段。
    """
    ordered = sorted(lines, key=lambda line: (float(line["bbox"][1]), float(line["bbox"][0])))
    groups: list[list[dict[str, Any]]] = []
    for index, line in enumerate(ordered):
        box = pymupdf.Rect(line["bbox"])
        pending = [pymupdf.Rect(item["bbox"]) for item in ordered[index + 1:]]
        placed = [pymupdf.Rect(item["bbox"]) for group in groups for item in group]
        neighbors = [*pending, *placed]
        best: tuple[float, list[dict[str, Any]]] | None = None
        for group in groups:
            prev_line = group[-1]
            prev = pymupdf.Rect(prev_line["bbox"])
            gap = box.y0 - prev.y1
            if gap > max(prev.height * 0.9, _LINE_GAP) or not _x_aligned(prev, box):
                continue
            if _stands_alone(prev_line) or _stands_alone(line):
                continue
            if _starts_after_hole(prev, box) or _returns_past_hole(prev, box):
                continue
            if _covers_right_neighbor(prev, box, neighbors):
                continue
            if best is None or gap < best[0]:
                best = (gap, group)
        if best is None:
            groups.append([line])
        else:
            best[1].append(line)
    for group in groups:
        _snap_group_size(group)
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
    if abs(nxt.x0 - prev.x0) <= max(prev.height, 12.0) * 2:
        return True
    # 同一基线上、空隙不大的后半句。公式空洞比一个破折号宽得多，不从这里接上。
    if _same_baseline(prev, nxt):
        gap = nxt.x0 - prev.x1
        return -2 <= gap <= 24
    return False


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
