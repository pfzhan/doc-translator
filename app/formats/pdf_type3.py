"""Type3 点阵字没有 Unicode 时，按字形把字母认回来。

旧版 Ghostscript 把每个字画成位图，编码名是 /a0、/a1，没有 ToUnicode。
提取出来的是字符码，不是页面上的字母。整页用 tesseract 认字形，再按词重新组行。
点阵字的行框和词对不齐，词心会落在碎片之间，所以不能把词填回旧行框。

公式位图会被认成乱码。并进下一句后，写入框盖住公式，译文再压到下一段。
这类词收成行内占位符，原字形贴回去，句子不再从公式处拆开。居中的公式行、
矢量格子上的节点标签，以及用竖线字形画出的表，留下原字，不重排。
同一页正文用一个字号。缩进的伪代码不揉进正文。
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
# 编号后面跟着像样的标题词才是编号行。公式行开头的 ≥ 常被认成 2，后面是 RAR 这种碎片
_NUMBERED_TITLE = re.compile(r"^\d+(?:\.\d+)*[.,]?\s+[A-Z][a-z]{3,}")
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
# 低于这个置信度的词基本都是公式字形认出来的乱码（rains、iro、Tag 都在 25 上下），正文词很少这么低
_JUNK_CONF = 30.0
# 认得很有把握的词：正文字体靠它们来认
_CONFIDENT_CONF = 90.0
# 词底下数学字体占到这个比例才算公式。贴着公式的 to、if 只占一半
_MATH_FONT_SHARE = 0.8
# 单独一个比较符是点阵公式，不是正文里的小于号。
_OPERATOR = re.compile(r"^[=<>≤≥≠≈∈∉⊂⊃±×÷&]$")
_BULLET_WORD = re.compile(r"^[eco*•·]$")
_DISPLAY_MARK = re.compile(r"[=<>≤≥≠≈∈∉]|[{}]")
_CAPTION_START = re.compile(r"^(?:Figure|Fig\.?|Table|EXAMPLE|Example)\b")
_FOOTNOTE_START = re.compile(r"""^["“”?]\s*[A-Za-z]""")
# 例题标题：全大写的 EXAMPLE 2.1，或 Example 2.1 后面紧跟大写开头的句子。
# 「For instance, in / Example 1.1, we can」是句子换行，不能断开
_EXAMPLE_START = re.compile(r"^EXAMPLE\b|^Example\s+\d+(?:\.\d+)*\.?\s+[A-Z]")
_LETTER_ITEM = re.compile(r"^\([a-zA-Z]\)\s")
# 缩进的伪代码。正文从栏左缘起笔，不在这里。
_LISTING_LINE = re.compile(
    r"(?:"
    r"end;"
    r"|for\b.+\bdo\b.*\bbegin\b"
    r"|select that view\b.+"
    r"|resulting\b.+\bselection;?"
    r"|[A-Za-z]\s*=\s*\{.+"
    r"|[A-Za-z]\s*=\s*S\b.+"
    r")$",
    re.I,
)
_LISTING_X = 95.0
# 下标被认成 B,,、a;、6;。单个逗号是 S,、is,，不能收成公式。
_BROKEN_SUB = re.compile(r"^(?:[A-Za-z]{1,2}[,;]{2,}|[A-Za-z]{1,2};|\d;)$")
_BAR_CHAR = re.compile(r"[|¦│]")
# 公式槽用数学字体，后面的占位符才会把这块裁成图贴回去。
_FORMULA_FONT = "CMMI10"
_FORMULA_MARK = "×"
# 公式空洞比栏沟窄。再宽就是两栏，不能从中间接上。
_HOLE_LIMIT = 160.0
_LEAD_LIMIT = 180.0
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
class _Piece:
    """一行里的一段：正文词，或公式空洞。"""

    kind: str
    text: str
    rect: pymupdf.Rect


@dataclass(frozen=True)
class GlyphBox:
    """丢弃字符码之前留下的点阵字框，用来对上认出来的词，并取颜色和基线。"""

    rect: pymupdf.Rect
    em: float
    color: int
    origin_y: float
    flags: int
    font: str = ""


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
    这一页识别超时或认不出正文时，只去掉字符码，点阵字留在原页，不让整本书失败。
    """
    doc = page.parent
    if doc is None or "blocks" not in data:
        return False
    names = broken_type3_names(doc, page)
    if not names or _broken_share(data, names) < _BROKEN_SHARE:
        return False
    try:
        words = ocr_page(page)
    except subprocess.TimeoutExpired:
        _drop_broken_lines(data, names)
        return False
    zones = [*_drawing_zones(page), *_pipe_zones(words), *_bar_zones(words)]
    zones.extend(_label_zones(words, zones))
    visible = [word for word in words if not _hidden(word, zones)]
    originals = _broken_lines(data, names)
    math = _math_font_words(visible, originals, _glyph_origins(page, names))
    prose = _without_noise(visible)
    kept = {id(word) for word in prose}
    debris = [word for word in visible if id(word) not in kept]
    _drop_broken_lines(data, names)
    math_rects = {tuple(word.rect) for word in prose if id(word) in math}
    rebuilt = _compose_lines(prose, debris, originals, math_rects)
    if not rebuilt:
        return False
    _snap_page_sizes(rebuilt)
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


def _glyph_origins(page: pymupdf.Page, names: set[str]) -> list[tuple[str, float, float]]:
    """点阵字逐字的起笔点（字体名, x, y）。

    字框按字体框算，宽高都和墨迹差好几个点，越往行尾差得越多；起笔点和墨迹对得上。
    """
    found: list[tuple[str, float, float]] = []
    for block in page.get_text("rawdict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span.get("font") not in names:
                    continue
                for char in span.get("chars") or []:
                    x, y = char["origin"]
                    found.append((str(span["font"]), float(x), float(y)))
    return found


def _origin_fonts(word: OcrWord, origins: list[tuple[str, float, float]]) -> dict[str, int]:
    """起笔点落在词里的字各属哪个字体。起笔点在字的顶上一点，竖向放宽几个点。"""
    hits: dict[str, int] = {}
    for font, x, y in origins:
        if word.rect.x0 - 0.8 <= x < word.rect.x1 - 0.5 and word.rect.y0 - 4 <= y <= word.rect.y1 + 4:
            hits[font] = hits.get(font, 0) + 1
    return hits


def _word_fonts(word: OcrWord, glyphs: list[GlyphBox]) -> tuple[str, float]:
    """词底下占得最多的点阵字体，以及它占的比例。点阵字框比墨迹高，只按横向重叠算。"""
    cy = (word.rect.y0 + word.rect.y1) / 2
    share: dict[str, float] = {}
    for glyph in glyphs:
        if not glyph.font or not (glyph.rect.y0 <= cy <= glyph.rect.y1):
            continue
        overlap = min(glyph.rect.x1, word.rect.x1) - max(glyph.rect.x0, word.rect.x0)
        if overlap > 0:
            share[glyph.font] = share.get(glyph.font, 0.0) + overlap
    if not share:
        return "", 0.0
    font = max(share, key=lambda name: share[name])
    return font, share[font] / sum(share.values())


def _confident_word(word: OcrWord, length: int) -> bool:
    core = word.text.strip(".,;:()[]\"'“”‘’")
    return word.conf >= _CONFIDENT_CONF and core.isalpha() and len(core) >= length


# 正文里常排成斜体的拉丁缩写，斜体字体在一页上常常只有这一个词
_LATIN_ABBREV = re.compile(r"^(?:i\.e\.|e\.g\.|cf\.|etc\.|vs\.|viz\.)[,;:]?$", re.I)
# 节号、列表编号、图号、页码（4.3、1.、Figure 9:、11）。换成公式后标题和编号就认不出了。
# 单独一个数字不算：数学斜体的 b 常被认成 6
_NUMBER_TOKEN = re.compile(r"^(?:\d+(?:\.\d+)*[.:]|\d+(?:\.\d+)+|\d{2,})$")


def _math_font_words(
    words: list[OcrWord], glyphs: list[GlyphBox], origins: list[tuple[str, float, float]] | None = None,
) -> set[int]:
    """字形几乎全来自数学字体的词。

    点阵字体名是 A、B、C 这种自造名，分不出 cmr 和 cmmi。正文字体会认出大量高置信度的普通单词，
    数学斜体和符号字体一个也认不出：这一页上没认出过普通单词的字体就当数学字体。
    tesseract 会把数学字形认成 rains、Tag、6（其实是 b）这类词，送去翻译只会出乱码。
    本身是高置信度普通单词的不算：标题和正文斜体在一页上常常只出现几个词，of、to 也在里面。
    单个字母不算普通单词，它们正是 a、b、w 这类变量。

    一两个字的短词（认成 6 的 b、变量 a、x）按逐字起笔点（origins）归属：字框会漂到隔壁的正文上，
    起笔点不会。b 底下是一个数学斜体的字加一个正文字体的间隔字，数学字体占一半就算。
    长词仍按字框比例算：几乎每个字都带着别的字体的间隔字，按起笔点会把 TPC-D、GROUP-BY 也算进来。
    """
    origins = origins or []
    attributed = [(word, _origin_fonts(word, origins), *_word_fonts(word, glyphs)) for word in words]
    # 正文字体仍按字框比例认：按起笔点算，间隔字会把一两个词算到数学字体名下
    prose_fonts = {font for word, hits, font, share in attributed
                   if font and share >= 0.5 and _confident_word(word, 4)}

    def from_math_font(word: OcrWord, hits: dict[str, int], font: str, share: float) -> bool:
        if bool(font) and font not in prose_fonts and share >= _MATH_FONT_SHARE:
            return True
        if hits and len(word.text.strip(".,;:()[]{}")) <= 2:
            math_hits = sum(count for name, count in hits.items() if name not in prose_fonts)
            return math_hits * 2 >= sum(hits.values())
        return False

    return {
        id(word) for word, hits, font, share in attributed
        if from_math_font(word, hits, font, share)
        and not _confident_word(word, 2) and not _LATIN_ABBREV.match(word.text.strip())
        and not _NUMBER_TOKEN.match(word.text.strip())
    }


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


def _compose_lines(
    prose: list[OcrWord], debris: list[OcrWord], glyphs: list[GlyphBox],
    math_rects: set[tuple[float, ...]] | None = None,
) -> list[dict[str, Any]]:
    """同一基线上被公式空洞切开的正文仍是一句。空洞收成公式槽，居中的公式行不重排。

    math_rects 是字形来自数学字体的词。公式行、伪代码先按认出来的原词判断，
    留下来排正文的行再把这些词换成公式槽：先换的话 = 和 S 没了，这两种行就认不出来了。
    """
    lines: list[dict[str, Any]] = []
    for row in _baseline_rows(prose):
        for pieces in _row_pieces(row, debris, glyphs):
            pieces = _without_bullet(pieces)
            if not pieces or _display_formula(pieces) or _listing_line(pieces):
                continue
            pieces = _math_as_formula(pieces, math_rects or set())
            line = _line_from_pieces(_peel_operators(pieces), glyphs)
            if line is not None:
                lines.append(line)
    return lines


def _math_as_formula(pieces: list[_Piece], math_rects: set[tuple[float, ...]]) -> list[_Piece]:
    """数学字体的词换成公式槽，相邻的槽并成一个。

    夹在两个槽中间、没有一个像样单词的碎片（/(n—、)、,）也并进去，
    否则 r = j/(n−j) 会切成好几块，译文里各放各的。
    """
    if not math_rects:
        return pieces
    marked = [piece.kind == "formula" or tuple(piece.rect) in math_rects for piece in pieces]
    out: list[_Piece] = []
    for index, piece in enumerate(pieces):
        is_slot = marked[index]
        if not is_slot and out and out[-1].kind == "formula" and _math_glue(piece.text):
            is_slot = any(marked[index + 1:index + 3])
        if not is_slot:
            out.append(piece)
            continue
        if out and out[-1].kind == "formula":
            out[-1] = _Piece("formula", _FORMULA_MARK, out[-1].rect | piece.rect)
        else:
            out.append(_Piece("formula", _FORMULA_MARK, pymupdf.Rect(piece.rect)))
    return out


def _math_glue(text: str) -> bool:
    """公式中间的碎片：没有两个字母以上的词。"""
    token = text.strip()
    return bool(token) and len(token) <= 6 and not re.search(r"[A-Za-z]{2,}", token)


def _baseline_rows(words: list[OcrWord]) -> list[list[OcrWord]]:
    ordered = sorted(words, key=lambda word: ((word.rect.y0 + word.rect.y1) / 2, word.rect.x0))
    rows: list[list[OcrWord]] = []
    for word in ordered:
        cy = (word.rect.y0 + word.rect.y1) / 2
        if rows:
            last = rows[-1]
            ly = sum((item.rect.y0 + item.rect.y1) / 2 for item in last) / len(last)
            limit = max(word.rect.height, last[0].rect.height, 8.0) * 0.55
            if abs(cy - ly) <= limit:
                last.append(word)
                continue
        rows.append([word])
    for row in rows:
        row.sort(key=lambda word: word.rect.x0)
    return rows


def _row_pieces(
    row: list[OcrWord], debris: list[OcrWord], glyphs: list[GlyphBox],
) -> list[list[_Piece]]:
    if not row:
        return []
    segments: list[list[_Piece]] = []
    current: list[_Piece] = []
    lead = _formula_before(row[0], debris, glyphs)
    if lead is not None:
        current.append(lead)
    current.append(_Piece("prose", row[0].text, row[0].rect))
    for prev, word in zip(row, row[1:]):
        gap = word.rect.x0 - prev.rect.x1
        bridge = _formula_between(prev, word, debris, glyphs)
        section = _section_only(current) and gap <= 22.0
        if section or (bridge is not None and gap <= _HOLE_LIMIT):
            if bridge is not None and not section:
                current.append(bridge)
            current.append(_Piece("prose", word.text, word.rect))
            continue
        if gap > max(prev.rect.height * 1.1, 12.0):
            segments.append(current)
            current = [_Piece("prose", word.text, word.rect)]
            continue
        current.append(_Piece("prose", word.text, word.rect))
    trail = _formula_after(row[-1], debris, glyphs)
    if trail is not None:
        current.append(trail)
    segments.append(current)
    return segments


def _section_only(pieces: list[_Piece]) -> bool:
    prose = [piece for piece in pieces if piece.kind == "prose"]
    return len(prose) == 1 and bool(_SECTION_NO.fullmatch(prose[0].text.strip()))


def _formula_before(
    word: OcrWord, debris: list[OcrWord], glyphs: list[GlyphBox],
) -> _Piece | None:
    box = _evidence_box(
        debris, glyphs, word.rect.x0 - _LEAD_LIMIT, word.rect.x0, word.rect.y0, word.rect.y1,
    )
    if box is None or word.rect.x0 - box.x1 > 24 or box.width < 6:
        return None
    return _Piece("formula", _FORMULA_MARK, box)


def _formula_after(
    word: OcrWord, debris: list[OcrWord], glyphs: list[GlyphBox],
) -> _Piece | None:
    box = _evidence_box(
        debris, glyphs, word.rect.x1, word.rect.x1 + 80, word.rect.y0, word.rect.y1,
    )
    if box is None or box.x0 - word.rect.x1 > 24 or box.width < 6:
        return None
    return _Piece("formula", _FORMULA_MARK, box)


def _formula_between(
    prev: OcrWord, nxt: OcrWord, debris: list[OcrWord], glyphs: list[GlyphBox],
) -> _Piece | None:
    gap = nxt.rect.x0 - prev.rect.x1
    if gap <= max(prev.rect.height * 1.1, 12.0) or gap > _HOLE_LIMIT:
        return None
    y0 = min(prev.rect.y0, nxt.rect.y0)
    y1 = max(prev.rect.y1, nxt.rect.y1)
    box = _evidence_box(debris, glyphs, prev.rect.x1, nxt.rect.x0, y0, y1)
    if box is None or box.width < 6:
        return None
    # 点阵字框横向和墨迹差出好几个点，N_supplier 的 N 会落到 where 那边被漏掉。
    # 两词之间整段都算进来，多出的空白排版时会按墨迹收掉
    return _Piece("formula", _FORMULA_MARK, pymupdf.Rect(prev.rect.x1 + 1.5, box.y0, nxt.rect.x0 - 1.5, box.y1))


def _evidence_box(
    debris: list[OcrWord], glyphs: list[GlyphBox],
    x0: float, x1: float, y0: float, y1: float,
) -> pymupdf.Rect | None:
    """空洞里的乱码或没对上正文的点阵，就是公式墨迹。字框比墨迹高，裁到这一行。"""
    box = pymupdf.Rect()
    # 下标落在基线下面几个点。上一行的字框中心不能算进这一行的空洞。
    debris_band = pymupdf.Rect(x0, y0 - 4, x1, y1 + 8)
    glyph_band = pymupdf.Rect(x0, y0 - 2, x1, y1 + 4)
    for word in debris:
        if _center_in(word.rect, debris_band):
            box |= word.rect
    if box.is_empty:
        for glyph in glyphs:
            if not _center_in(glyph.rect, glyph_band):
                continue
            clipped = pymupdf.Rect(
                glyph.rect.x0, max(glyph.rect.y0, y0 - 2),
                glyph.rect.x1, min(glyph.rect.y1, y1 + 6),
            )
            if clipped.width >= 1 and clipped.height >= 1:
                box |= clipped
    if box.is_empty or box.x0 >= x1 or box.x1 <= x0:
        return None
    return pymupdf.Rect(max(box.x0, x0), box.y0, min(box.x1, x1), box.y1)


def _center_in(rect: pymupdf.Rect, band: pymupdf.Rect) -> bool:
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    return band.x0 <= cx <= band.x1 and band.y0 <= cy <= band.y1


def _without_bullet(pieces: list[_Piece]) -> list[_Piece]:
    """行首的单字母 e 是项目符号的点阵，不是单词。留下原字，句子从后面起翻。"""
    if len(pieces) < 2 or pieces[0].kind != "prose":
        return pieces
    head, nxt = pieces[0], pieces[1]
    if nxt.kind != "prose" or not nxt.text[:1].isupper():
        return pieces
    if head.rect.width <= 14 and _BULLET_WORD.fullmatch(head.text.strip()):
        return pieces[1:]
    return pieces


def _display_formula(pieces: list[_Piece]) -> bool:
    """居中、几乎没有正文词、又带着比较符的一行是公式。认出来的符号是错的，留下原字。"""
    prose = [piece for piece in pieces if piece.kind == "prose"]
    if not prose:
        return True
    box = pymupdf.Rect()
    for piece in prose:
        box |= piece.rect
    if box.x0 < 160:
        return False
    text = " ".join(piece.text for piece in prose)
    # 比较符本身常被认成数学字体、换成了公式槽，这时剩下的词里已经没有 = 了。
    # 靠公式槽认的，整行（连同槽）都要缩进：行首是公式的 (p,s,c) where … is 是正文
    whole = pymupdf.Rect(box)
    for piece in pieces:
        whole |= piece.rect
    has_slot = any(piece.kind == "formula" for piece in pieces) and whole.x0 >= 160
    if len(text) > 90 or not (_DISPLAY_MARK.search(text) or has_slot):
        return False
    if _NUMBERED_TITLE.match(text) or _SQL_LINE.match(text) or _CAPTION_START.match(text):
        return False
    outside = re.sub(r"\([^)]*\)", " ", text)
    long_words = re.findall(r"[A-Za-z]{4,}", outside)
    return len(long_words) <= 1


def _listing_line(pieces: list[_Piece]) -> bool:
    """缩进的 end;、for … do begin、S = {…} 是伪代码。留下原字，不揉进正文。"""
    prose = [piece for piece in pieces if piece.kind == "prose"]
    if not prose:
        return False
    box = pymupdf.Rect()
    for piece in prose:
        box |= piece.rect
    if box.x0 < _LISTING_X:
        return False
    text = " ".join(piece.text for piece in prose).strip()
    return bool(_LISTING_LINE.fullmatch(text))


def _peel_operators(pieces: list[_Piece]) -> list[_Piece]:
    """行内单独的比较符、夹在括号组之间的数字，以及认成 B,, 的下标，是公式。"""
    peeled: list[_Piece] = []
    for index, piece in enumerate(pieces):
        token = piece.text.strip()
        if piece.kind == "formula" or _OPERATOR.fullmatch(token) or _broken_sub(pieces, index):
            peeled.append(_Piece("formula", _FORMULA_MARK, piece.rect))
            continue
        if token.isdigit() and len(token) == 1 and _between_parens(pieces, index):
            peeled.append(_Piece("formula", _FORMULA_MARK, piece.rect))
            continue
        peeled.append(piece)
    return peeled


def _broken_sub(pieces: list[_Piece], index: int) -> bool:
    """B,,、a;、6; 是下标。后面接大写词的 Q; 是句首，S, 是正文。"""
    token = pieces[index].text.strip()
    if not _BROKEN_SUB.fullmatch(token):
        return False
    if token.endswith(";") and ",," not in token and ";" not in token[:-1]:
        nxt = next((piece for piece in pieces[index + 1:] if piece.kind == "prose"), None)
        if nxt is not None and nxt.text[:1].isupper():
            return False
    return True


def _between_parens(pieces: list[_Piece], index: int) -> bool:
    left = next((piece for piece in reversed(pieces[:index]) if piece.kind == "prose"), None)
    right = next((piece for piece in pieces[index + 1:] if piece.kind == "prose"), None)
    if left is None or right is None:
        return False
    return "(" in left.text and ")" in right.text


# 下沿会落到基线以下的字符：降部字母、逗号分号、括号
_DESCENDER_RE = re.compile(r"[gjpqyQ,;()\[\]{}/|@$_]")


def _ocr_baseline(words: list[tuple[str, pymupdf.Rect]], fallback: float) -> float:
    """按认出来的词的下沿取基线。

    点阵字报出来的 origin 不可信：老 Ghostscript 的 Type3 字体常把它放在字框上部，
    公式槽按它算下沉，整个公式会往下掉一行。带降部的词下沿偏低，先排除；都带降部时取中位数。
    """
    bottoms = sorted(rect.y1 for text, rect in words if text.strip() and not _DESCENDER_RE.search(text))
    if not bottoms:
        bottoms = sorted(rect.y1 for text, rect in words if text.strip())
    if not bottoms:
        return fallback
    return bottoms[len(bottoms) // 2]


def _line_from_pieces(pieces: list[_Piece], glyphs: list[GlyphBox]) -> dict[str, Any] | None:
    """正文槽和公式槽合成一行。没盖住点阵字的词不是这页的坏字体，不另起一行。"""
    spans = _spans_from_pieces(pieces)
    prose = [span for span in spans if span["font"] != _FORMULA_FONT]
    if not prose:
        return None
    ink = pymupdf.Rect()
    for span in prose:
        ink |= pymupdf.Rect(span["bbox"])
    matched = [glyph for glyph in glyphs if _same_band(glyph.rect, ink) and _x_hits(glyph.rect, ink)]
    if not matched:
        return None
    glyph = max(matched, key=lambda item: _overlap_width(item.rect, ink))
    text = " ".join(str(span["text"]) for span in prose)
    size = _line_size(text, ink.height, glyph.em)
    baseline = _ocr_baseline([(piece.text, piece.rect) for piece in pieces if piece.kind == "prose"],
                             glyph.origin_y)
    for span in spans:
        span["size"] = size
        span["flags"] = glyph.flags
        span["color"] = glyph.color
        span["origin"] = (span["bbox"][0], baseline)
    box = pymupdf.Rect()
    for span in spans:
        box |= pymupdf.Rect(span["bbox"])
    box = _cover_line_glyphs(box, glyphs)
    bbox = (box.x0, box.y0, box.x1, box.y1)
    return {"spans": spans, "wmode": 0, "dir": (1.0, 0.0), "bbox": bbox}


def _cover_line_glyphs(box: pymupdf.Rect, glyphs: list[GlyphBox]) -> pymupdf.Rect:
    """擦除要盖住这一行的点阵字。字框比墨迹高，只加宽，纵向仍停在这一行。"""
    covered = pymupdf.Rect(box)
    for glyph in glyphs:
        cy = (glyph.rect.y0 + glyph.rect.y1) / 2
        if not (box.y0 - 1 <= cy <= box.y1 + 1):
            continue
        if not _x_hits(glyph.rect, box):
            continue
        covered.x0 = min(covered.x0, glyph.rect.x0)
        covered.x1 = max(covered.x1, glyph.rect.x1)
    return covered


def _spans_from_pieces(pieces: list[_Piece]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    prose: list[_Piece] = []

    def flush(trailing_formula: bool) -> None:
        if not prose:
            return
        box = pymupdf.Rect()
        for piece in prose:
            box |= piece.rect
        text = " ".join(piece.text for piece in prose)
        if spans and spans[-1]["font"] == _FORMULA_FONT:
            text = " " + text
        if trailing_formula:
            text += " "
        spans.append(_span(text, box, "Times-Roman"))
        prose.clear()

    for index, piece in enumerate(pieces):
        if piece.kind == "formula":
            nxt = pieces[index + 1] if index + 1 < len(pieces) else None
            flush(nxt is not None and nxt.kind == "prose")
            spans.append(_span(_FORMULA_MARK, piece.rect, _FORMULA_FONT))
            continue
        prose.append(piece)
    flush(False)
    return spans


def _span(text: str, box: pymupdf.Rect, font: str) -> dict[str, Any]:
    pad = 0.6
    bbox = (box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
    return {
        "text": text,
        "bbox": bbox,
        "size": 10.0,
        "font": font,
        "flags": 0,
        "color": 0,
        "origin": (bbox[0], bbox[3]),
        "ascender": 0.8,
        "descender": -0.2,
    }


def _table_bar(word: OcrWord) -> bool:
    """格子里的竖线。集合记号 {b | a < b} 里的竖线不是表。"""
    text = word.text
    if not _BAR_CHAR.search(text) or any(mark in text for mark in "{}="):
        return False
    return len(text) <= 48


def _formula_row(row: list[OcrWord]) -> bool:
    """同一基线上有大括号，竖线是集合分隔符，不是格子。格子里的等号仍是表。"""
    return any(any(mark in word.text for mark in "{}") for word in row)


def _table_row(row: list[OcrWord]) -> bool:
    """竖线粘在格子文字里，或单独的竖线旁边还有格子文字。光一根竖线是公式分隔符。"""
    bars = [word for word in row if _table_bar(word)]
    if not bars or _formula_row(row):
        return False
    if any(len(word.text.strip()) > 1 for word in bars):
        return True
    return any(not _table_bar(word) for word in row)


def _bar_zones(words: list[OcrWord]) -> list[_Zone]:
    """竖线写在格子文字里，不是单独的词。这样的几行围出一张字形表，整表留下原字。

    竖线常在格子中间。左列，以及夹在中间、没认出竖线的行，都在这张表的纵带里。
    """
    table_rows = [row for row in _baseline_rows(words) if _table_row(row)]
    if len(table_rows) < 2:
        return []
    y0 = min(word.rect.y0 for row in table_rows for word in row)
    y1 = max(word.rect.y1 for row in table_rows for word in row)
    if y1 - y0 < 24 or y1 - y0 > 220:
        return []
    box = pymupdf.Rect()
    for word in words:
        cy = (word.rect.y0 + word.rect.y1) / 2
        if y0 - 4 <= cy <= y1 + 4:
            box |= word.rect
    if box.is_empty or box.height > 240:
        return []
    return [_Zone(pymupdf.Rect(box.x0 - 6, box.y0 - 4, box.x1 + 8, box.y1 + 4))]


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
        inside = zone.rect.x0 <= cx <= zone.rect.x1 and zone.rect.y0 <= cy <= zone.rect.y1
        if inside and (not zone.labels_only or short):
            return True
        # 节点标签贴在笔画上方几个点。整句正文比标签长，不挡。
        if short and not zone.labels_only and zone.rect.x0 - 8 <= cx <= zone.rect.x1 + 8:
            if zone.rect.y0 - 14 <= cy < zone.rect.y0:
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


def _footnote_start(line: dict[str, Any]) -> bool:
    """脚注记号被认成引号或问号。它不是上一段的下一行。"""
    return bool(_FOOTNOTE_START.match(_span_text(line)))


def _example_start(line: dict[str, Any]) -> bool:
    return bool(_EXAMPLE_START.match(_span_text(line)))


def _letter_item(line: dict[str, Any]) -> bool:
    """(a) (b) 各自起条。续行不是这种开头，仍接在这一条上。"""
    return bool(_LETTER_ITEM.match(_span_text(line)))


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
                    font=str(span.get("font") or ""),
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
        "origin": (box.x0, _ocr_baseline([(word.text, word.rect) for word in words], glyph.origin_y)),
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
            if _example_start(line) and not _example_start(prev_line):
                continue
            if _footnote_start(line) and not _footnote_start(prev_line):
                continue
            if _letter_item(line):
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
    """墨迹高度就是磅值。有没有下伸字母不再另乘一档，否则同一段写出两种字号。"""
    del text
    if height <= 0:
        return 0.0
    return round(min(max(height, 6.0), 36.0), 1)


def _prose_height(line: dict[str, Any]) -> float:
    """公式槽的高框不算。一行的字号看正文墨迹。"""
    height = 0.0
    for span in line.get("spans") or []:
        if span.get("font") == _FORMULA_FONT:
            continue
        raw = span.get("bbox")
        if not raw or len(raw) < 4:
            continue
        height = max(height, float(raw[3]) - float(raw[1]))
    return height


def _snap_page_sizes(lines: list[dict[str, Any]]) -> None:
    """同一页正文用一个字号，取正文墨迹高度的中位数。

    字数够成句的行就是正文，墨迹被公式撑高也不另起一档。
    墨迹明显更高的短行是标题，留下原来的字号。
    """
    heights = [
        height
        for line in lines
        if (height := _prose_height(line)) >= 6 and height <= 20
        and sum(ch.isalpha() for ch in _span_text(line)) >= 12
    ]
    if not heights:
        return
    median = sorted(heights)[len(heights) // 2]
    body = round(min(max(median, 8.0), 14.0), 1)
    limit = median * 1.35
    for line in lines:
        height = _prose_height(line)
        letters = sum(ch.isalpha() for ch in _span_text(line))
        if height <= 0:
            continue
        if letters < 12 and height > limit and letters >= 4:
            continue
        for span in line.get("spans") or []:
            if span.get("size"):
                span["size"] = body


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
