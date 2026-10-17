"""Type3 点阵字没有 Unicode 时，按字形把字母认回来。

旧版 Ghostscript 把每个字画成一张位图，编码名是 /a0、/a1，没有 ToUnicode，提出来的是字符码。
同一个字符码在整份文档里画的是同一张位图。整份文档先用 tesseract 认一遍，把认出的字母按位置
对到位图上投票，就知道每张位图是哪个字母。文字层再按位图重建：位置和大小取 d1 给的位图框，
词距看墨迹间隔。OCR 词框的漂移、漏词、个别认错都不再进译文。

数学斜体的 a、v、S 只单独出现，从不进单词；数学符号各次认得不一样。这两类写成数学字体，
上下标按实际大小和位置写出。后面的公式占位、分段和普通 TeX PDF 走同一套逻辑。
"""
from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pymupdf

_CUSTOM_NAME = re.compile(r"^[A-Za-z]\d+$")
# 坏字体占这一页文字的比例。零星符号字体不整页重建。
_BROKEN_SHARE = 0.4
_MIN_BROKEN_CHARS = 20
_OCR_SCALE = 3.0
_OCR_TIMEOUT = 90
# 认得一致的比例。低于它的字形各次认得不一样，是数学符号或数学斜体。
_SURE = 0.8
# 一页的字形里有这么多认出了字母，这一页才重建。否则留下原字形。
_READABLE = 0.8
# 一个字形有四分之一以上出现在单词里，它就是正文字母
_IN_WORD = 0.25
# 词距以 x 高计：0.4 以内是字距，0.6 以上是空格。中间这一档两种都有，
# 按这一行的字距和词距分，这一行没有明显的词距时听 OCR 分词的
_TIGHT = 0.4
_WIDE = 0.6
# 同一行里最大的横向空隙，以字号计。撑开的词距也不到半个字号，再宽是并排的署名、另一栏或表格的另一列
_RUN_GAP = 1.6
# 比本行字号小这么多的是上下标。字号不小、基线却偏开这么多（以字号计）的也是
_SCRIPT = 0.8
_SHIFT = 0.18
# 同一个字母的两张位图像到这个程度，就是同一种字体
_SAME_SHAPE = 0.85
_SAME_MARK = 0.5
# 实心圆点的墨迹占满框的七成以上，字母不到一半
_FILLED = 0.65
_SHAPE_DPI = 600
_SHAPE_GRID = 16
# 基线距在这么多个字号以内的下一行并进同一块。TeX 的行距是 1.2 个字号，行内公式撑高时再多一点
_LEADING = 1.6
_MATH_SYMBOLS = set("<>=+|{}^~*∑∏√∞≤≥≠≈∈∉⊂⊃±×÷→←")
_SENTENCE_MARKS = set(".,;:")
_OPERATORS = set("=<>+-−–—≤≥≠×")
_RELATIONS = set("=<>≤≥≠≺⪯≈∈")
_TRAILING_MARKS = set(",.;:)")
# 节号、编号、年份这样的数
_NUMBER = re.compile(r"\(?\d+(?:\.\d+)*[.:,)]?")
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
# 等宽字格宽的下限，以字号计。等宽字是 0.525；正文词里字距最匀的（each、latter）不到 0.48
_MONO_PITCH = 0.5
# 英文里单独成词的大写字母
_LETTER_WORDS = {"A", "I"}
# 正文字体里很窄的字（0.28 个字号上下）。等宽字体里它们也占满一格
_NARROW = set("iljI.,;:’'!|")
# 各类字母的墨迹高度占字号的比例（Computer Modern、Times 都差不多）
_X_LETTERS = set("acemnorsuvwxz")
_TALL_LETTERS = set("bdhkl") | set("ABCDEFGHIKLMNOPRSTUVWXYZ") | set("0123456789")
_DESCENDERS = set("gpqy")
_UNSTEADY = set("fjJQ")
_HEIGHT_RATIO = {"x": 0.45, "tall": 0.69, "desc": 0.64, "i": 0.67, "t": 0.62}
# 各类字母的字心在基线以上多高
_CENTER_RATIO = {"x": 0.225, "tall": 0.345, "desc": 0.13, "i": 0.335, "t": 0.31}
_QUOTES = set("‘’“”'\"`")
_DASHES = set("-–—−~")
# 减号的墨迹宽度（以字号计）。短横 0.5，长破折号 1.0，连字符 0.3
_MINUS_WIDTH = (0.55, 0.8)
_TEXT_FONT = "Times-Roman"
_MONO_FONT = "Courier"
_MATH_FONT = "CMMI10"
_SYMBOL_FONT = "CMSY10"
# PyMuPDF get_text("dict") 的结构。字段随版本变化，只在这层进出。
PageData = dict[str, Any]
Key = tuple[int, int]
# 细线：\_ 的下划线、行内分数线。比 30pt 长的是表格线、脚注线
_RULE: Key = (-2, 0)
_RULE_HEIGHT = 1.2
_RULE_LENGTH = 30.0
_PAD = 0.6


@dataclass(frozen=True)
class OcrChar:
    text: str
    rect: pymupdf.Rect
    conf: float


@dataclass(frozen=True)
class OcrWord:
    text: str
    rect: pymupdf.Rect
    conf: float
    chars: tuple[OcrChar, ...] = ()


@dataclass(frozen=True)
class Glyph:
    """页面上画出来的一个字形。key 是（字体 xref，字符码），rect 是位图的墨迹框，origin 是起笔点。"""

    key: Key
    rect: pymupdf.Rect
    color: int = 0
    origin: tuple[float, float] | None = None


@dataclass(frozen=True)
class Label:
    text: str
    agreement: float


@dataclass
class _Line:
    glyphs: list[Glyph]
    tokens: list[list[Glyph]] = field(default_factory=list)
    baseline: float = 0.0
    xheight: float = 0.0
    size: float = 0.0


def broken_type3_names(doc: pymupdf.Document, page: pymupdf.Page) -> set[str]:
    """没有 ToUnicode、字形名又是 /a0 这种自造名的 Type3 字体。"""
    return set(_broken_fonts(doc, page).values())


def _broken_fonts(doc: pymupdf.Document, page: pymupdf.Page) -> dict[int, str]:
    found: dict[int, str] = {}
    for xref, _ext, subtype, name, _ref, _encoding in page.get_fonts():
        if subtype == "Type3" and name and _custom_encoding(doc, xref):
            found[xref] = name
    return found


class Type3Reader:
    """一份文档里的点阵字。第一次用到时整份认一遍，之后逐页给出重建好的行。"""

    def __init__(self, doc: pymupdf.Document):
        self.doc = doc
        self._lines: dict[int, list[_Line]] | None = None
        self.labels: dict[Key, Label] = {}
        self.roles: dict[Key, str] = {}
        self.mono: dict[Key, float] = {}
        self.proportional: set[Key] = set()
        self.doubtful: set[Key] = set()

    def lines(self, pno: int) -> list[_Line] | None:
        if self._lines is None:
            self._lines = self._read()
        return self._lines.get(pno)

    def _read(self) -> dict[int, list[_Line]]:
        shapes: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        pages: dict[int, list[tuple[Key, float, float, float, int]]] = {}
        for page in self.doc:
            fonts = _broken_fonts(self.doc, page)
            if not fonts:
                continue
            for xref in fonts:
                if xref not in shapes:
                    shapes[xref] = _font_shapes(self.doc, xref)
            placed = _placed_codes(page, {name: xref for xref, name in fonts.items()}, shapes)
            if len(placed) >= _MIN_BROKEN_CHARS:
                pages[page.number] = placed
        if not pages:
            return {}
        flip = _y_flip(self.doc, pages, shapes)
        glyphs = {
            pno: [_glyph(key, x, y, size, color, shapes, flip) for key, x, y, size, color in placed]
            for pno, placed in pages.items()
        }
        words = _ocr_pages(self.doc, list(glyphs))
        self.labels = _vote_labels(glyphs, words)
        # \_ 和行内分数线是细线，老 Ghostscript 画成小图片。它们是这一行的一部分
        self.labels[_RULE] = Label("_", 1.0)
        for pno, items in glyphs.items():
            items.extend(Glyph(_RULE, rect) for rect in _rules(self.doc[pno]))
        owners = {pno: _word_owners(items, words.get(pno, [])) for pno, items in glyphs.items()}
        lines = {pno: _build_lines(items, self.labels, owners[pno]) for pno, items in glyphs.items()}
        self.roles, self.doubtful = _learn_roles(lines, self.labels, lambda keys: _glyph_shapes(self.doc, glyphs, keys))
        self.mono = _mono_kinds(lines, self.labels)
        self.proportional = _proportional_kinds(lines, self.mono)
        for key in _bullets(lines, self.labels, lambda keys: _ink_share(self.doc, glyphs, keys)):
            self.labels[key] = Label("•", 1.0)
            self.roles[key] = "text"
        bearings = _bearings(lines, self.labels)
        # 等宽字的空格是一整格空白，字距本身也宽；f 的钩子伸进右边的空白。
        # 认出等宽字形、量出各字形的边距以后重新分词
        for pno, page_lines in lines.items():
            for line in page_lines:
                line.tokens = _tokens(line, self.labels, owners[pno], self.mono, bearings)
        return {pno: page_lines for pno, page_lines in lines.items() if _readable(glyphs[pno], self.labels)}


def recover_type3_page(page: pymupdf.Page, data: PageData, reader: Type3Reader | None = None) -> bool:
    """这一页大部分是自造编码的点阵字时，用认出来的文字层换掉字符码。

    认不出来的页（识别超时、字形大多没认出字母）只去掉字符码，点阵字留在原页，不让整本书失败。
    """
    doc = page.parent
    if doc is None or "blocks" not in data:
        return False
    names = broken_type3_names(doc, page)
    if not names or _broken_share(data, names) < _BROKEN_SHARE:
        return False
    reader = reader or Type3Reader(doc)
    lines = reader.lines(page.number)
    _drop_broken_lines(data, names)
    if not lines:
        return False
    built = [line for line in (_line_dict(line, reader) for line in lines) if line is not None]
    if not built:
        return False
    for group in _group_lines(built):
        data["blocks"].append({"type": 0, "bbox": _union_bbox(group), "lines": group})
    return True


# ---- 字体和字形位置 ----


def _font_shapes(doc: pymupdf.Document, xref: int) -> dict[int, tuple[float, float, float, float]]:
    """画了东西的字形在字形空间里的框（已乘 FontMatrix）。只推进笔位的空字形不收。"""
    matrix = _numbers(doc.xref_get_key(xref, "FontMatrix")[1]) or [0.001, 0, 0, 0.001, 0, 0]
    font_box = _numbers(doc.xref_get_key(xref, "FontBBox")[1]) or [0, 0, 1000, 1000]
    procs = dict(re.findall(r"/([^\s/<>\[\]()]+)\s+(\d+)\s+0\s+R", doc.xref_get_key(xref, "CharProcs")[1]))
    shapes: dict[int, tuple[float, float, float, float]] = {}
    for code, name in _encoding_names(doc, xref).items():
        proc = procs.get(name)
        if proc is None:
            continue
        stream = doc.xref_stream(int(proc)) or b""
        found = re.search(rb"\bd([01])\b", stream)
        if found is None or not stream[found.end():].strip():
            continue
        head = _numbers(stream[:found.start()].decode("latin-1"))
        box = head[2:6] if found.group(1) == b"1" and len(head) >= 6 else font_box[:4]
        scale_x, scale_y = matrix[0], matrix[3]
        x0, x1 = sorted((box[0] * scale_x, box[2] * scale_x))
        y0, y1 = sorted((box[1] * scale_y, box[3] * scale_y))
        shapes[code] = (x0, y0, x1, y1)
    return shapes


def _encoding_names(doc: pymupdf.Document, xref: int) -> dict[int, str]:
    kind, value = doc.xref_get_key(xref, "Encoding")
    source = doc.xref_object(int(value.split()[0])) if kind == "xref" else value
    found = re.search(r"/Differences\s*\[(.*?)\]", source, re.S)
    names: dict[int, str] = {}
    if found is None:
        return names
    code = 0
    # 名字之间可以不留空格：[65/a0/a1]
    for token in re.findall(r"/[^\s/\[\]<>()]+|\d+", found.group(1)):
        if token.isdigit():
            code = int(token)
        else:
            names[code] = token[1:]
            code += 1
    return names


def _numbers(raw: str) -> list[float]:
    """PDF 的数可以省掉整数部分：.001。"""
    return [float(item) for item in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", raw)]


def _placed_codes(
    page: pymupdf.Page, fonts: dict[str, int], shapes: dict[int, dict[int, tuple[float, float, float, float]]],
) -> list[tuple[Key, float, float, float, int]]:
    """按内容流顺序列出画了东西的字形：(key, 起笔 x, 起笔 y, 字号, 颜色)。

    get_text 会丢掉和前一个字同位置的零宽字形，texttrace 不丢。
    """
    placed: list[tuple[Key, float, float, float, int]] = []
    for span in page.get_texttrace():
        xref = fonts.get(str(span.get("font")))
        if xref is None or abs(span["dir"][1]) > 0.01:
            continue
        drawn = shapes.get(xref, {})
        color = _rgb(span.get("color"))
        for char in span["chars"]:
            if char[1] in drawn:
                placed.append(((xref, char[1]), char[2][0], char[2][1], float(span["size"]), color))
    return placed


def _glyph(
    key: Key, x: float, y: float, size: float, color: int,
    shapes: dict[int, dict[int, tuple[float, float, float, float]]], flip: bool,
) -> Glyph:
    x0, y0, x1, y1 = shapes[key[0]][key[1]]
    if flip:
        rect = pymupdf.Rect(x + x0 * size, y + y0 * size, x + x1 * size, y + y1 * size)
    else:
        rect = pymupdf.Rect(x + x0 * size, y - y1 * size, x + x1 * size, y - y0 * size)
    return Glyph(key, rect, color, (x, y))


def _y_flip(
    doc: pymupdf.Document, pages: dict[int, list[tuple[Key, float, float, float, int]]],
    shapes: dict[int, dict[int, tuple[float, float, float, float]]],
) -> bool:
    """字形空间朝上还是朝下。Ghostscript 的点阵字用倒置的 Tm，字形往起笔点下方画。

    PyMuPDF 不给出文字矩阵，两种都摆一遍，看哪种盖住页面上的墨迹。
    """
    pno = max(pages, key=lambda number: len(pages[number]))
    pix = doc[pno].get_pixmap(dpi=36, colorspace=pymupdf.csGRAY, alpha=False)
    scale = 36 / 72

    def covered(flip: bool) -> int:
        mask = bytearray(pix.width * pix.height)
        for key, x, y, size, color in pages[pno]:
            rect = _glyph(key, x, y, size, color, shapes, flip).rect
            left, right = max(int(rect.x0 * scale), 0), min(int(rect.x1 * scale) + 1, pix.width)
            for row in range(max(int(rect.y0 * scale), 0), min(int(rect.y1 * scale) + 1, pix.height)):
                mask[row * pix.width + left: row * pix.width + right] = b"\x01" * max(right - left, 0)
        return sum(1 for hit, value in zip(mask, pix.samples) if hit and value < 128)

    return covered(True) > covered(False)


def _rules(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """行内的短横线，画成图片的和画成矢量的都算。"""
    rects = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    try:
        rects.extend(pymupdf.Rect(drawing["rect"]) for drawing in page.get_drawings())
    except Exception:  # noqa: BLE001 - 个别页面的矢量读不出来，当没有
        pass
    return [rect for rect in rects if rect.height <= _RULE_HEIGHT and 1.5 <= rect.width <= _RULE_LENGTH]


def _rgb(color: object) -> int:
    if not isinstance(color, (tuple, list)) or len(color) < 3:
        return 0
    red, green, blue = (min(max(int(float(part) * 255 + 0.5), 0), 255) for part in color[:3])
    return (red << 16) | (green << 8) | blue


# ---- 认字 ----


def _ocr_pages(doc: pymupdf.Document, numbers: list[int]) -> dict[int, list[OcrWord]]:
    """整份文档一起认。渲染在主线程，tesseract 并行跑。某一页超时就不给这页投票。"""
    if shutil.which("tesseract") is None:
        raise ValueError("这份 PDF 的文字是没有 Unicode 的点阵字，需要安装 tesseract 才能读出原文")
    with tempfile.TemporaryDirectory(prefix="type3-") as tmp:
        paths: dict[int, Path] = {}
        for pno in numbers:
            path = Path(tmp) / f"p{pno}.png"
            doc[pno].get_pixmap(matrix=pymupdf.Matrix(_OCR_SCALE, _OCR_SCALE), alpha=False).save(path)
            paths[pno] = path

        def read(pno: int) -> list[OcrWord]:
            try:
                return ocr_image(paths[pno])
            except subprocess.TimeoutExpired:
                return []

        workers = max(1, min(4, os.cpu_count() or 1, len(numbers)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(numbers, pool.map(read, numbers)))


def ocr_image(path: Path, scale: float = _OCR_SCALE) -> list[OcrWord]:
    exe = shutil.which("tesseract")
    if exe is None:
        return []
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    proc = subprocess.run(
        [exe, str(path), "stdout", "--psm", "3", "-l", "eng", "-c", "hocr_char_boxes=1", "hocr"],
        capture_output=True, text=True, timeout=_OCR_TIMEOUT, check=False, env=env,
    )
    if proc.returncode != 0 and "traineddata" in proc.stderr:
        raise ValueError("点阵字页面没有认出文字。请确认 tesseract 的英文语言包已安装")
    parser = _HocrWords(scale)
    parser.feed(proc.stdout)
    return parser.words


class _HocrWords(HTMLParser):
    """hOCR 里的词和逐字框。字框和置信度在 ocrx_cinfo 上。"""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.words: list[OcrWord] = []
        self._stack: list[str] = []
        self._word: tuple[pymupdf.Rect, float] | None = None
        self._chars: list[OcrChar] = []
        self._char: tuple[pymupdf.Rect, float] | None = None
        self._text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            return
        info = dict(attrs)
        kind = info.get("class") or ""
        title = info.get("title") or ""
        self._stack.append(kind)
        if kind == "ocrx_word":
            box = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
            conf = re.search(r"x_wconf ([\d.]+)", title)
            self._word = (self._rect(box), float(conf.group(1)) if conf else 0.0) if box else None
            self._chars = []
        elif kind == "ocrx_cinfo":
            box = re.search(r"x_bboxes (\d+) (\d+) (\d+) (\d+)", title)
            conf = re.search(r"x_conf ([\d.]+)", title)
            self._char = (self._rect(box), float(conf.group(1)) if conf else 0.0) if box else None
            self._text = ""

    def handle_data(self, data: str) -> None:
        if self._char is not None:
            self._text += data

    def handle_endtag(self, tag: str) -> None:
        if tag != "span" or not self._stack:
            return
        kind = self._stack.pop()
        if kind == "ocrx_cinfo" and self._char is not None:
            text = _LIGATURES.get(self._text.strip(), self._text.strip())
            if text:
                self._chars.append(OcrChar(text, *self._char))
            self._char = None
        elif kind == "ocrx_word" and self._word is not None:
            text = "".join(char.text for char in self._chars)
            if text:
                self.words.append(OcrWord(text, self._word[0], self._word[1], tuple(self._chars)))
            self._word = None

    def _rect(self, box: re.Match[str]) -> pymupdf.Rect:
        return pymupdf.Rect(*(int(box.group(index)) / self.scale for index in range(1, 5)))


def _vote_labels(glyphs: dict[int, list[Glyph]], words: dict[int, list[OcrWord]]) -> dict[Key, Label]:
    """认出的字按位置落到字形上投票。

    字数和字形数对得上的词逐个配对，先定下大部分字形。对不上的词（连字 ffi 是一个字形、
    OCR 漏了或多认了一个字）再按已定的字形和位置对齐：认得准的字形吃掉自己那个字，
    剩下的字归没定的字形。
    """
    votes: dict[Key, Counter[str]] = defaultdict(Counter)
    uneven: list[tuple[list[Glyph], tuple[OcrChar, ...]]] = []
    for pno, page_words in words.items():
        items = glyphs.get(pno, [])
        for word in page_words:
            box = pymupdf.Rect(word.rect.x0 - 0.7, word.rect.y0 - 0.7, word.rect.x1 + 0.7, word.rect.y1 + 0.7)
            inside = sorted((glyph for glyph in items if _center_in(glyph.rect, box)), key=lambda glyph: glyph.rect.x0)
            if not inside or not word.chars:
                continue
            # 字数对上时按顺序配：OCR 的字序可靠，逐字框常偏出两三个点
            if len(inside) == len(word.chars):
                for glyph, char in zip(inside, word.chars):
                    votes[glyph.key][char.text] += char.conf
            else:
                uneven.append((inside, word.chars))
    first = _best_labels(votes)
    for inside, chars in uneven:
        for glyph, taken in zip(inside, _align(inside, chars, first)):
            if taken and not _sure(first.get(glyph.key)):
                votes[glyph.key]["".join(char.text for char in taken)] += min(char.conf for char in taken) / 2
    return _best_labels(votes)


def _best_labels(votes: dict[Key, Counter[str]]) -> dict[Key, Label]:
    labels: dict[Key, Label] = {}
    for key, counter in votes.items():
        total = sum(counter.values())
        if total > 0:
            text, weight = counter.most_common(1)[0]
            labels[key] = Label(text, weight / total)
    return labels


def _sure(label: Label | None) -> bool:
    return label is not None and label.agreement >= _SURE


def _align(
    glyphs: list[Glyph], chars: tuple[OcrChar, ...], labels: dict[Key, Label],
) -> list[tuple[OcrChar, ...]]:
    """把认出的字按顺序分给字形。每个字形分 0 到 3 个字，代价最小的分法。

    已认定的字形要分到它自己的字；分到的字和字形横向不重叠要加代价，
    OCR 多认出来的一个点不会把后面的字都挤错位。
    """
    count, total = len(glyphs), len(chars)
    inf = float("inf")
    cost = [[inf] * (total + 1) for _ in range(count + 1)]
    back = [[0] * (total + 1) for _ in range(count + 1)]
    cost[0][0] = 0.0
    for index in range(1, count + 1):
        label = labels.get(glyphs[index - 1].key)
        for used in range(total + 1):
            for take in range(0, min(3, used) + 1):
                prev = cost[index - 1][used - take]
                if prev == inf:
                    continue
                taken = chars[used - take:used]
                text = "".join(char.text for char in taken)
                if _sure(label):
                    step = 0.0 if text == label.text else 2.0
                else:
                    step = 1.5 if take == 0 else 0.3 * (take - 1)
                # 分给一个字形的字要和它横向重叠
                step += sum(1.0 for char in taken if _overlap(glyphs[index - 1].rect, char.rect) <= 0)
                if prev + step < cost[index][used]:
                    cost[index][used] = prev + step
                    back[index][used] = take
    if cost[count][total] == inf:
        return [() for _ in glyphs]
    taken: list[tuple[OcrChar, ...]] = []
    used = total
    for index in range(count, 0, -1):
        take = back[index][used]
        taken.append(tuple(chars[used - take:used]))
        used -= take
    return taken[::-1]


def _readable(glyphs: list[Glyph], labels: dict[Key, Label]) -> bool:
    known = sum(1 for glyph in glyphs if glyph.key in labels)
    return bool(glyphs) and known >= len(glyphs) * _READABLE


# ---- 行、词和字形的角色 ----


def _build_lines(glyphs: list[Glyph], labels: dict[Key, Label], owners: dict[int, int]) -> list[_Line]:
    """按基线把字形排成行，和普通 PDF 的文字一样按位置排，不按内容流的先后。

    认得出字母类别的字形各有一条基线，基线相同的是一行。上下标、分数的分子分母
    字小、基线偏开，自成一小行，再并回它们所在的那一行；认不出类别的符号和标点按高度并进去。
    都并好以后再按大的横向空隙切开：并排的署名、两栏。内容流里分母会让笔头退回分数左边，
    按先后接行会把分数后面的半句接到分母上。
    """
    placed: list[tuple[Glyph, float, float]] = []
    loose: list[Glyph] = []
    for glyph in glyphs:
        size = _glyph_size(glyph, labels)
        if size is None:
            loose.append(glyph)
        else:
            placed.append((glyph, size, _own_baseline(glyph, labels, size)))
    bands: list[list[tuple[Glyph, float, float]]] = []
    for item in sorted(placed, key=lambda item: item[2]):
        band = next((band for band in reversed(bands[-6:]) if _same_band(band, item)), None)
        if band is None:
            bands.append([item])
        else:
            band.append(item)
    ordered = [[item[0] for item in band] for band in sorted(bands, key=len, reverse=True)]
    rows: list[_Line] = []
    for piece in (piece for piece in ordered if len(piece) >= 3):
        # 整体下沉的正文字（N 下面的 supplier）字数也不少，仍是上一段的下标
        host = _host_line(rows, piece, labels)
        if host is None:
            rows.append(_row(piece, labels))
        else:
            host.glyphs.extend(piece)
    pending = [*(piece for piece in ordered if len(piece) < 3), *([glyph] for glyph in loose)]
    while pending:
        pending = _attach(rows, pending, labels)
        if pending:
            # 剩下的小段自己成行：先让最大的一段成行，再看别的能不能并进来
            rows.append(_row(pending.pop(0), labels))
    lines: list[_Line] = []
    for row in rows:
        row.glyphs.sort(key=lambda glyph: glyph.rect.x0)
        _measure(row, labels)
        current = [row.glyphs[0]]
        right = row.glyphs[0].rect.x1
        for glyph in row.glyphs[1:]:
            if glyph.rect.x0 - right > row.size * _RUN_GAP:
                lines.append(_Line(current))
                current = []
            current.append(glyph)
            right = max(right, glyph.rect.x1)
        lines.append(_Line(current))
    # 没落进任何一行的细线是图里的线，不是字
    lines = [line for line in lines if any(glyph.key != _RULE for glyph in line.glyphs)]
    for line in lines:
        _measure(line, labels)
        line.tokens = _tokens(line, labels, owners)
    return sorted(lines, key=lambda line: (line.baseline, line.glyphs[0].rect.x0))


def _row(glyphs: list[Glyph], labels: dict[Key, Label]) -> _Line:
    row = _Line(list(glyphs))
    _measure(row, labels)
    return row


def _attach(rows: list[_Line], pending: list[list[Glyph]], labels: dict[Key, Label]) -> list[list[Glyph]]:
    """小段并进行里，一轮轮并到没有可并的为止：1、.、1 一个挨一个才够得着标题。"""
    extents = {id(row): _extent(row) for row in rows}
    while pending:
        still: list[list[Glyph]] = []
        for piece in pending:
            host = _host_line(rows, piece, labels, extents)
            if host is None:
                still.append(piece)
                continue
            host.glyphs.extend(piece)
            left, right = extents[id(host)]
            extents[id(host)] = (min(left, *(glyph.rect.x0 for glyph in piece)),
                                 max(right, *(glyph.rect.x1 for glyph in piece)))
        if len(still) == len(pending):
            return still
        pending = still
    return []


def _extent(row: _Line) -> tuple[float, float]:
    return min(glyph.rect.x0 for glyph in row.glyphs), max(glyph.rect.x1 for glyph in row.glyphs)


def _own_baseline(glyph: Glyph, labels: dict[Key, Label], size: float) -> float:
    kind = _letter_class(labels[glyph.key].text)
    return glyph.rect.y1 - (size * 0.19 if kind == "desc" else 0.0)


def _same_band(band: list[tuple[Glyph, float, float]], item: tuple[Glyph, float, float]) -> bool:
    """基线差不到字号的一成二、字号也差不多，是同一条基线上的字。"""
    _glyph, size, baseline = item
    sizes = [entry[1] for entry in band[-8:]]
    base = statistics.median(entry[2] for entry in band[-8:])
    ref = statistics.median(sizes)
    return abs(baseline - base) <= min(size, ref) * 0.12 and max(size, ref) <= min(size, ref) * 1.25


def _host_line(
    lines: list[_Line], piece: list[Glyph], labels: dict[Key, Label],
    extents: dict[int, tuple[float, float]] | None = None,
) -> _Line | None:
    """上下标、分数、标点这样的小段并进哪一行：字心落在那一行 x 高的上下 0.75 个字号以内，
    横向挨着那一行。字更大的段不并进小字行。"""
    left = min(glyph.rect.x0 for glyph in piece)
    right = max(glyph.rect.x1 for glyph in piece)
    center = statistics.median((glyph.rect.y0 + glyph.rect.y1) / 2 for glyph in piece)
    sizes = [size for glyph in piece if (size := _glyph_size(glyph, labels))]
    best: tuple[float, _Line] | None = None
    for line in lines:
        if sizes and statistics.median(sizes) > line.size * 1.1:
            continue
        offset = abs(center - (line.baseline - line.xheight / 2))
        if offset > line.size * 0.75 or (best is not None and offset >= best[0]):
            continue
        line_left, line_right = extents[id(line)] if extents else _extent(line)
        reach = line.size * _RUN_GAP
        if line_left - reach <= right and left <= line_right + reach:
            best = (offset, line)
    return best[1] if best is not None else None


def _letter_class(text: str) -> str | None:
    """按墨迹高度分的字母类别。f、j、J、Q 在斜体里会伸到基线以下，高度不定，不归类。"""
    if len(text) != 1 or text in _UNSTEADY:
        return None
    if text in _X_LETTERS:
        return "x"
    if text in _DESCENDERS:
        return "desc"
    if text in ("i", "t"):
        return text
    if text in _TALL_LETTERS:
        return "tall"
    return None


def _glyph_size(glyph: Glyph, labels: dict[Key, Label], sure: bool = True) -> float | None:
    """按字母类别把墨迹高度换回字号。认不准的字形没有字号。"""
    label = labels.get(glyph.key)
    if label is None or (sure and label.agreement < _SURE):
        return None
    kind = _letter_class(label.text)
    if kind is None:
        return None
    return glyph.rect.height / _HEIGHT_RATIO[kind]


def _measure(line: _Line, labels: dict[Key, Label]) -> None:
    sizes = [size for glyph in line.glyphs if (size := _glyph_size(glyph, labels))]
    line.size = statistics.median(sizes) if sizes else statistics.median(g.rect.height for g in line.glyphs) / 0.7
    sitting = [
        glyph for glyph in line.glyphs
        if (size := _glyph_size(glyph, labels)) and size >= line.size * _SCRIPT
        and _letter_class(labels[glyph.key].text) in ("x", "tall", "i", "t")
    ]
    line.baseline = statistics.median(glyph.rect.y1 for glyph in (sitting or line.glyphs))
    line.xheight = line.size * _HEIGHT_RATIO["x"]


def _tokens(
    line: _Line, labels: dict[Key, Label], owners: dict[int, int], mono: dict[Key, float] | None = None,
    bearings: dict[Key, tuple[float, float]] | None = None,
) -> list[list[Glyph]]:
    tokens: list[list[Glyph]] = []
    mono = mono or {}
    bearings = bearings or {}
    gaps = [_gap(left, right, line, bearings) for left, right in zip(line.glyphs, line.glyphs[1:])]
    split = _space_split(gaps)
    for index, glyph in enumerate(line.glyphs):
        prev = tokens[-1][-1] if tokens else None
        pitch = (mono.get(prev.key) or mono.get(glyph.key)) if prev is not None else None
        same_word = prev is not None and owners.get(id(prev)) is not None and owners.get(id(prev)) == owners.get(id(glyph))
        if prev is not None and pitch and (prev.key in mono and glyph.key in mono or same_word):
            # 一边是等宽字、OCR 也认成同一个词（@db.stanford 里只出现一次的 s），按格宽判断
            joined = _middle(glyph) - _middle(prev) <= pitch * 1.5
        else:
            joined = prev is not None and not _space_before(gaps[index - 1], prev, glyph, owners, split)
        if joined:
            tokens[-1].append(glyph)
        else:
            tokens.append([glyph])
    # 逗号句号前面不会有空格。等宽字体里逗号在格子中间，墨迹离前一个字远
    merged: list[list[Glyph]] = []
    for token in tokens:
        marks = all((labels.get(glyph.key) or Label("?", 0)).text in _TRAILING_MARKS for glyph in token)
        if merged and marks and token[0].rect.x0 - max(glyph.rect.x1 for glyph in merged[-1]) <= line.xheight:
            merged[-1].extend(token)
        else:
            merged.append(token)
    return merged


def _gap(left: Glyph, right: Glyph, line: _Line, bearings: dict[Key, tuple[float, float]]) -> float:
    """两个字形墨迹之间的空白，以 x 高计，减掉两边字形自带的边距。"""
    raw = (right.rect.x0 - left.rect.x1) / line.xheight
    return raw - bearings.get(left.key, (0.0, 0.0))[1] - bearings.get(right.key, (0.0, 0.0))[0]


def _space_before(gap: float, prev: Glyph, glyph: Glyph, owners: dict[int, int], split: float | None) -> bool:
    if gap <= _TIGHT:
        return False
    if gap >= _WIDE:
        return True
    if split is not None:
        return gap >= split
    left, right = owners.get(id(prev)), owners.get(id(glyph))
    if left is None or right is None:
        return gap > (_TIGHT + _WIDE) / 2
    return left != right


def _space_split(gaps: list[float]) -> float | None:
    """两端对齐的一行里词距处处一样宽。取这一行字距和词距的中点当分界；
    这一行没有明显的词距时返回 None，交给 OCR 分词。"""
    letters = [gap for gap in gaps if gap <= _TIGHT]
    spaces = [gap for gap in gaps if _WIDE <= gap <= 3.0]
    if len(letters) < 3 or len(spaces) < 2:
        return None
    return (statistics.median(letters) + statistics.median(spaces)) / 2


def _bearings(pages: dict[int, list[_Line]], labels: dict[Key, Label]) -> dict[Key, tuple[float, float]]:
    """各字形左右自带的边距：在认得准的词里，它前后的字距比一般的字距宽多少、窄多少。

    正体 f 的钩子伸到右边，后面的字距总比别的字窄一截，of the 的空格看起来就只剩一半。
    """
    after: dict[Key, list[float]] = defaultdict(list)
    before: dict[Key, list[float]] = defaultdict(list)
    every: list[float] = []
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            if not _sure_word(token, labels):
                continue
            for left, right in zip(token, token[1:]):
                gap = (right.rect.x0 - left.rect.x1) / line.xheight
                after[left.key].append(gap)
                before[right.key].append(gap)
                every.append(gap)
    if not every:
        return {}
    usual = statistics.median(every)

    def shift(values: list[float]) -> float:
        return max(-0.3, min(0.3, statistics.median(values) - usual)) if len(values) >= 3 else 0.0

    return {key: (shift(before.get(key, [])), shift(after.get(key, []))) for key in {*after, *before}}


def _word_owners(glyphs: list[Glyph], words: list[OcrWord]) -> dict[int, int]:
    """每个字形落在 OCR 的第几个词里。词距拿不准时按它分词。"""
    owners: dict[int, int] = {}
    for index, word in enumerate(words):
        box = pymupdf.Rect(word.rect.x0 - 0.7, word.rect.y0 - 0.7, word.rect.x1 + 0.7, word.rect.y1 + 0.7)
        for glyph in glyphs:
            if _center_in(glyph.rect, box):
                owners[id(glyph)] = index
    return owners


def _learn_roles(
    pages: dict[int, list[_Line]], labels: dict[Key, Label],
    shapes: Callable[[set[Key]], dict[Key, np.ndarray]],
) -> tuple[dict[Key, str], set[Key]]:
    """整份文档统计每个字形出现在单词里的比例，定它是正文、数学还是数字标点。

    先用四个字母以上、每个字都认得准的词定下一批正文字形（psc、cp 这种视图名是数学斜体，不算）。
    这样的词里的字母都算在词里。短词再看：同一个词里有别的字母是正文字形，
    又没有常单独出现的变量字母（等宽的 S 在伪代码里也单独当变量用），它就算在词里。
    连字 ff、We 的 W、标题里只出现在 The 里的 T 和 h 这样补上。
    数学斜体的 a、v 几乎从不在词里（v’s 偶尔算一次）。
    从不进词的字母是变量。英文里单独成词的只有 A 和 I：拿位图和正文里的 A、I 比，
    像就是正文（摘要里小一号的 A）。斜体正文和数学斜体长得一样，其他字母比不出来。
    """
    tokens = [
        _without_plural(_main_glyphs(line, token, labels), labels)
        for lines in pages.values() for line in lines for token in line.tokens
    ]
    seen = Counter(glyph.key for token in tokens for glyph in token)
    seeds = {
        glyph.key for token in tokens if _sure_word(token, labels)
        for glyph in token if _letter(labels.get(glyph.key))
    }
    alone = Counter(token[0].key for token in tokens if len(token) == 1 and _letter(labels.get(token[0].key)))
    loner = {key for key, count in alone.items() if count * 3 >= seen[key]}
    in_word: Counter[Key] = Counter()
    for token in tokens:
        if not _alpha_core(token, labels, 2):
            continue
        letters = [glyph for glyph in token if _letter(labels.get(glyph.key))]
        sure = _sure_word(token, labels)
        for glyph in letters:
            others = [item for item in letters if item is not glyph]
            if sure or (any(item.key in seeds for item in others) and not any(item.key in loner for item in others)):
                in_word[glyph.key] += 1
    numeral: Counter[Key] = Counter()
    for token in tokens:
        text = "".join(_text_of(glyph, labels) for glyph in token)
        # 至少两位数字：数学斜体的 b 常被认成 6，后面跟着逗号也像个数
        if sum(ch.isdigit() for ch in text) >= 2 and _NUMBER.fullmatch(text):
            for glyph in token:
                numeral[glyph.key] += 1
    roles = {
        key: _role(labels.get(key), in_word[key], count, numeral[key] * 2 >= count)
        for key, count in seen.items()
    }
    # 横线按宽窄分：减号 0.6 个字号，连字符、短横、长破折号各有各的宽度；比横线高的是箭头
    dashes: dict[Key, list[tuple[float, float]]] = defaultdict(list)
    for line in (line for lines in pages.values() for line in lines):
        for glyph in line.glyphs:
            if _text_of(glyph, labels) in _DASHES:
                dashes[glyph.key].append((glyph.rect.width / line.size, glyph.rect.height / line.size))
    for key, shapes_seen in dashes.items():
        width = statistics.median(shape[0] for shape in shapes_seen)
        height = statistics.median(shape[1] for shape in shapes_seen)
        if height > 0.15 or _MINUS_WIDTH[0] <= width <= _MINUS_WIDTH[1]:
            roles[key] = "symbol"
    # 冠词 A 在句首，变量 A 在句中。句首居多、位图又像正文的 A，才是冠词。
    # 只出现一两次的标点也拿位图和常见的同名字形比
    opening = Counter()
    for lines in pages.values():
        previous = "."
        for line in lines:
            for token in line.tokens:
                if len(token) == 1:
                    if previous[-1:] in ".!?:":
                        opening[token[0].key] += 1
                previous = "".join(_text_of(glyph, labels) for glyph in token) or previous
    words = {
        key for key, role in roles.items()
        if role == "letter" and labels[key].text in _LETTER_WORDS and opening[key] * 2 >= seen[key]
    }
    odd = {
        key for key, role in roles.items()
        if role == "neutral" and seen[key] <= 2 and not labels[key].text.isalnum()
    }
    names = {labels[key].text for key in words | odd}
    refs = {
        key for key, role in roles.items()
        if key not in odd and labels.get(key) is not None and labels[key].text in names
        and (role == "text" or (role == "neutral" and seen[key] >= 4))
    }
    doubtful: set[Key] = set()
    if words or odd:
        images = shapes(words | odd | refs)
        for key in words:
            if _looks_like(key, refs, labels, images, _SAME_SHAPE):
                roles[key] = "text"
        # 脚注星号被认成引号：和真正的引号长得不像，这个字认得不可信
        doubtful = {key for key in odd if not _looks_like(key, refs, labels, images, _SAME_MARK)}
    return {key: "math" if role == "letter" else role for key, role in roles.items()}, doubtful


def _without_plural(token: list[Glyph], labels: dict[Key, Label]) -> list[Glyph]:
    """数学里给变量加复数：the a’s、v’s。去掉 ’s，剩下的单个字母不算单词。"""
    texts = [_text_of(glyph, labels) for glyph in token]
    while texts and not texts[-1].isalnum():
        texts.pop()
    if len(texts) >= 3 and texts[-1] == "s" and texts[-2] in ("’", "'"):
        return token[:len(texts) - 2]
    return token


def _letter(label: Label | None) -> bool:
    return label is not None and label.text.isalpha()


def _core(token: list[Glyph], labels: dict[Key, Label]) -> str:
    """去掉两头的标点、引号和脚注记号（Efficiently* 的星号），剩下中间的字。"""
    texts = [(labels.get(glyph.key) or Label("?", 0)).text for glyph in token]
    start, end = 0, len(texts)
    while start < end and not texts[start].isalpha():
        start += 1
    while end > start and not texts[end - 1].isalpha():
        end -= 1
    return "".join(texts[start:end])


def _alpha_core(token: list[Glyph], labels: dict[Key, Label], length: int) -> bool:
    core = _core(token, labels)
    return sum(ch.isalpha() for ch in core) >= length and core.replace("-", "").replace("’", "").isalpha()


def _sure_word(token: list[Glyph], labels: dict[Key, Label]) -> bool:
    if any((label := labels.get(glyph.key)) is None or label.agreement < _SURE for glyph in token):
        return False
    return _alpha_core(token, labels, 4)


def _role(label: Label | None, in_word: int, count: int, numeral: bool = False) -> str:
    """数字认得不稳（粗体 1 有时认成 l），但大多出现在 1.1、12 这种数里，仍是数字。"""
    if label is None:
        return "symbol"
    if count and in_word / count >= _IN_WORD:
        return "text"
    if any(ch in _MATH_SYMBOLS for ch in label.text):
        return "symbol"
    if label.agreement < _SURE and not (numeral and label.text.isdigit()):
        return "math" if label.text.isalpha() else "symbol"
    if label.text.isalpha():
        return "letter"
    return "neutral"


def _text_of(glyph: Glyph, labels: dict[Key, Label]) -> str:
    label = labels.get(glyph.key)
    return label.text if label is not None else "?"


def _looks_like(
    key: Key, refs: set[Key], labels: dict[Key, Label], images: dict[Key, np.ndarray], threshold: float,
) -> bool:
    image = images.get(key)
    label = labels.get(key)
    if image is None or label is None:
        return False
    same = [images[other] for other in refs if other in images and labels[other].text == label.text]
    return any(float(np.dot(image, other)) >= threshold for other in same)


def _bullets(
    pages: dict[int, list[_Line]], labels: dict[Key, Label], density: Callable[[set[Key]], dict[Key, float]],
) -> set[Key]:
    """项目符号：总是单独站在行首，方方正正，墨迹填满。OCR 常把实心圆点认成 e。"""
    lines = [line for page_lines in pages.values() for line in page_lines]
    seen = Counter(glyph.key for line in lines for glyph in line.glyphs)
    starts = Counter(line.tokens[0][0].key for line in lines if line.tokens and len(line.tokens[0]) == 1)
    shapes: dict[Key, pymupdf.Rect] = {}
    for line in lines:
        for glyph in line.glyphs:
            shapes.setdefault(glyph.key, glyph.rect)
    keys = {
        key for key, count in starts.items()
        if count >= 2 and count * 2 >= seen[key] and key != _RULE
        and 0.8 <= shapes[key].width / max(shapes[key].height, 0.1) <= 1.25
    }
    filled = density(keys) if keys else {}
    return {key for key in keys if filled.get(key, 0.0) >= _FILLED}


def _ink_share(doc: pymupdf.Document, glyphs: dict[int, list[Glyph]], keys: set[Key]) -> dict[Key, float]:
    """字形框里墨迹占多少。"""
    found: dict[Key, float] = {}
    for pno, items in glyphs.items():
        for glyph in items:
            if glyph.key not in keys or glyph.key in found:
                continue
            pix = doc[pno].get_pixmap(dpi=_SHAPE_DPI, clip=glyph.rect, colorspace=pymupdf.csGRAY, alpha=False)
            ink = 1.0 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width] / 255
            found[glyph.key] = float(ink.mean()) if ink.size else 0.0
    return found


def _glyph_shapes(
    doc: pymupdf.Document, glyphs: dict[int, list[Glyph]], keys: set[Key],
) -> dict[Key, np.ndarray]:
    """每种字形取第一次出现的位图，按原比例放进方格，块平均到 16×16，减均值、归一，点积就是相似度。"""
    first: dict[Key, tuple[int, Glyph]] = {}
    for pno, items in glyphs.items():
        for glyph in items:
            if glyph.key in keys and glyph.key not in first:
                first[glyph.key] = (pno, glyph)
    found: dict[Key, np.ndarray] = {}
    for key, (pno, glyph) in first.items():
        pix = doc[pno].get_pixmap(dpi=_SHAPE_DPI, clip=glyph.rect, colorspace=pymupdf.csGRAY, alpha=False)
        if pix.width < 2 or pix.height < 2:
            continue
        ink = 1.0 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width] / 255
        side = max(ink.shape)
        canvas = np.zeros((side, side))
        top, left = (side - ink.shape[0]) // 2, (side - ink.shape[1]) // 2
        canvas[top:top + ink.shape[0], left:left + ink.shape[1]] = ink
        edges = np.linspace(0, side, _SHAPE_GRID + 1).astype(int)[:-1]
        grid = np.add.reduceat(np.add.reduceat(canvas, edges, axis=0), edges, axis=1).flatten()
        grid -= grid.mean()
        norm = float(np.linalg.norm(grid))
        if norm > 0:
            found[key] = grid / norm
    return found


# ---- 写成 PyMuPDF 的行 ----


def _line_dict(line: _Line, reader: Type3Reader) -> dict[str, Any] | None:
    styles = _glyph_styles(line, reader)
    items: list[tuple[str, Glyph]] = []
    for index, token in enumerate(line.tokens):
        if index:
            items.append(("", _SPACE))
        items.extend((styles[id(glyph)], glyph) for glyph in token)
    # 词间空格跟着正文走：公式 span 整个换成占位符，空格放在里面会一起丢掉
    for index, (style, glyph) in enumerate(items):
        if glyph is _SPACE:
            before, after = items[index - 1][0], items[index + 1][0]
            items[index] = (before if before in _PROSE or after not in _PROSE else after, glyph)
    spans: list[dict[str, Any]] = []
    run: list[Glyph] = []
    current = ""
    for style, glyph in items:
        if run and style != current:
            spans.append(_span(run, current, line, reader))
            run = []
        run.append(glyph)
        current = style
    if run:
        spans.append(_span(run, current, line, reader))
    spans = [span for span in spans if span["text"].strip()]
    if not spans:
        return None
    box = pymupdf.Rect()
    for span in spans:
        box |= pymupdf.Rect(span["bbox"])
    return {"spans": spans, "wmode": 0, "dir": (1.0, 0.0), "bbox": (box.x0, box.y0, box.x1, box.y1)}


# 词之间的空格
_SPACE = Glyph((-1, -1), pymupdf.Rect())
# 空格只跟着普通正文走：等宽词和公式一样会换成占位符
_PROSE = ("text",)


def _glyph_styles(line: _Line, reader: Type3Reader) -> dict[int, str]:
    """每个字形写成什么：text、mono、math、symbol、script。

    含数学字形的词整个是公式；夹在两个公式词中间的数字、括号也并进去，(p, s, c) 是一个公式。
    一串公式词末尾的句号逗号是句子的标点，留在正文里。
    """
    mono = _mono_tokens(line, reader)
    kinds: list[str] = []
    for index, token in enumerate(line.tokens):
        roles = [_role_of(glyph, line, reader) for glyph in token if not _is_script(glyph, line, reader)]
        if index in mono:
            # 等宽字体里没有数学：{venky, anand}@db 的括号、@ 都是原样的字
            kinds.append("text")
        elif any(role in ("math", "symbol") for role in roles):
            kinds.append("math")
        elif "text" in roles:
            kinds.append("text")
        else:
            kinds.append("neutral")
    for index, kind in enumerate(kinds):
        if kind != "neutral":
            continue
        before = next((item for item in reversed(kinds[:index]) if item != "neutral"), "")
        after = next((item for item in kinds[index + 1:] if item != "neutral"), "")
        # k = 2、1 − (…)：紧挨着运算符的数也是公式
        # 列表编号 1. 后面跟着 ≺ 也不是被减数：带句号的不算
        operand = (index > 0 and kinds[index - 1] == "math" and _ends_with_operator(line.tokens[index - 1], reader)) or (
            index + 1 < len(kinds) and kinds[index + 1] == "math"
            and _starts_with_operator(line.tokens[index + 1], reader)
            and not _sentence_mark(line.tokens[index][-1], reader)
        )
        kinds[index] = "math" if (before == "math" and after == "math") or operand else "text"
    absorbed = _close_brackets(line.tokens, kinds, reader)
    prose_tails = _hyphen_tails(line, kinds, reader)
    splits = _clause_splits(line.tokens, kinds, reader)
    styles: dict[int, str] = {}
    for index, (token, kind) in enumerate(zip(line.tokens, kinds)):
        tail = len(token)
        phrase_end = (
            index + 1 == len(kinds) or kinds[index + 1] != "math" or _ends_sentence(token, reader)
            or index in splits
        )
        if index in mono and index + 1 not in mono:
            # 一串等宽词末尾的句号逗号是句子的。C1, C2 中间的逗号仍是代码
            while tail > 1 and _sentence_mark(token[tail - 1], reader):
                tail -= 1
        if kind == "math" and phrase_end:
            opened = _phrase_depth(line.tokens, kinds, index, reader)
            while tail > 1:
                last = token[tail - 1]
                if _sentence_mark(last, reader):
                    tail -= 1
                elif _bracket(last, reader) < 0 and opened < 0:
                    # (using d) 的右括号是正文开的，留在正文
                    opened += 1
                    tail -= 1
                else:
                    break
        for position, glyph in enumerate(token):
            if _is_script(glyph, line, reader):
                styles[id(glyph)] = "script"
            elif id(glyph) in prose_tails:
                styles[id(glyph)] = "text"
            elif (kind == "math" and position < tail) or id(glyph) in absorbed:
                styles[id(glyph)] = "symbol" if _role_of(glyph, line, reader) == "symbol" else "math"
            else:
                styles[id(glyph)] = "mono" if index in mono and position < tail else "text"
    return styles


def _hyphen_tails(line: _Line, kinds: list[str], reader: Type3Reader) -> set[int]:
    """n-dimensional 里连字符后面的正文词留在正文，公式只是 n。不然模型看不到这个词，又把它译一遍。"""
    tails: set[int] = set()
    for token, kind in zip(line.tokens, kinds):
        if kind != "math":
            continue
        for position, glyph in enumerate(token):
            if _label_text(glyph, reader) not in "-‐" or position == 0:
                continue
            rest = [item for item in token[position + 1:] if _label_text(item, reader).isalpha()]
            if len(rest) >= 3 and all(reader.roles.get(item.key) == "text" for item in rest):
                tails.update(id(item) for item in token[position:])
            break
    return tails


def _close_brackets(tokens: list[list[Glyph]], kinds: list[str], reader: Type3Reader) -> set[int]:
    """公式里开了括号，括号要在公式里合上：(i + 1)st 的 1) 归公式，st 留在正文。"""
    absorbed: set[int] = set()
    depth = 0
    for token, kind in zip(tokens, kinds):
        if kind == "math":
            for glyph in token:
                depth = max(depth + _bracket(glyph, reader), 0)
            continue
        if depth <= 0:
            continue
        for glyph in token:
            absorbed.add(id(glyph))
            depth += _bracket(glyph, reader)
            if depth <= 0:
                break
        depth = max(depth, 0)
    return absorbed


def _phrase_depth(tokens: list[list[Glyph]], kinds: list[str], end: int, reader: Type3Reader) -> int:
    """以 end 结尾的这串公式词里，左括号比右括号多几个。负数是多出来的右括号。"""
    start = end
    while start > 0 and kinds[start - 1] == "math":
        start -= 1
    return sum(_bracket(glyph, reader) for token in tokens[start:end + 1] for glyph in token)


def _clause_splits(tokens: list[list[Glyph]], kinds: list[str], reader: Type3Reader) -> set[int]:
    """一串公式词里，哪些词末的逗号把公式断开。

    括号里的逗号不断：(p, s, c)。逗号左边这段还没有关系符，是单个变量或一组括号，断开：
    d, e, g 和 h 各是一个公式，⟨L, ⪯⟩, a ≺ b 是两个。左边已经有关系符（i = 1, 2, …, k），
    后面只是列举，除非后面又是一个关系式（a = b, c = d）。
    """
    splits: set[int] = set()
    depth = 0
    related = False
    for index, (token, kind) in enumerate(zip(tokens, kinds)):
        if kind != "math":
            depth, related = 0, False
            continue
        depth = max(depth + sum(_bracket(glyph, reader) for glyph in token), 0)
        related = related or _has_relation(token, reader)
        last = token[-1]
        if depth > 0 or not _sentence_mark(last, reader) or _label_text(last, reader) not in ",;":
            continue
        if not related or _relation_follows(tokens, kinds, index, reader):
            splits.add(index)
            depth, related = 0, False
    return splits


def _has_relation(token: list[Glyph], reader: Type3Reader) -> bool:
    return any(
        reader.roles.get(glyph.key) == "symbol" and _label_text(glyph, reader) in _RELATIONS for glyph in token
    )


def _relation_follows(tokens: list[list[Glyph]], kinds: list[str], index: int, reader: Type3Reader) -> bool:
    for after in range(index + 1, len(tokens)):
        if kinds[after] != "math":
            return False
        if _has_relation(tokens[after], reader):
            return True
        if _sentence_mark(tokens[after][-1], reader) and _label_text(tokens[after][-1], reader) in ",;":
            return False
    return False


def _ends_sentence(token: list[Glyph], reader: Type3Reader) -> bool:
    """size S. c gives：S 后面的句号断开两个公式。省略号 ... 不算。"""
    if len(token) < 2 or not _sentence_mark(token[-1], reader) or _label_text(token[-1], reader) != ".":
        return False
    return _label_text(token[-2], reader) != "."


def _starts_with_operator(token: list[Glyph], reader: Type3Reader) -> bool:
    first = token[0]
    return reader.roles.get(first.key) == "symbol" and _label_text(first, reader) in _OPERATORS


def _ends_with_operator(token: list[Glyph], reader: Type3Reader) -> bool:
    last = token[-1]
    return reader.roles.get(last.key) == "symbol" and _label_text(last, reader) in _OPERATORS


def _bracket(glyph: Glyph, reader: Type3Reader) -> int:
    text = _label_text(glyph, reader)
    return text.count("(") + text.count("[") - text.count(")") - text.count("]")


def _sentence_mark(glyph: Glyph, reader: Type3Reader) -> bool:
    """句子的标点认得准。下标 i 常被认成分号，但认得不稳，不算。"""
    label = reader.labels.get(glyph.key)
    return _sure(label) and label is not None and label.text in _SENTENCE_MARKS


def _role_of(glyph: Glyph, line: _Line, reader: Type3Reader) -> str:
    """细线贴着基线是下划线，是字；在基线以上是分数线，归公式。"""
    if glyph.key == _RULE:
        center = (glyph.rect.y0 + glyph.rect.y1) / 2
        return "neutral" if abs(center - line.baseline) <= line.size * 0.12 else "symbol"
    return reader.roles.get(glyph.key, "symbol")


def _is_script(glyph: Glyph, line: _Line, reader: Type3Reader) -> bool:
    """上下标：字号明显小，或者字号一样、整个字却上下偏开（N 下面正体的 supplier）。

    偏开看字心，不看下沿：斜体的 f 本来就伸到基线以下。
    认不出字母的符号没有字号，下沿悬在 x 高四分之三以上的（脚注记号）算上标。
    引号本来就悬着，连字符在 x 高一半上下，都不算。
    """
    # 认得不稳的字母（下标 N 有时认成 M）类别多半没错，字号照样能估
    size = _glyph_size(glyph, reader.labels, sure=False)
    if size is None:
        text = _label_text(glyph, reader)
        role = reader.roles.get(glyph.key, "symbol")
        trusted = role != "symbol" and glyph.key not in reader.doubtful
        if role == "text" or (trusted and (text in _QUOTES or text in _DASHES)):
            return False
        return glyph.rect.y1 < line.baseline - line.xheight * 0.75
    return _off_size(glyph, line, reader.labels, size)


def _off_size(glyph: Glyph, line: _Line, labels: dict[Key, Label], size: float) -> bool:
    """字号比这一行小一截，或者字心偏离同类字母该在的高度。"""
    if size < line.size * _SCRIPT:
        return True
    kind = _letter_class(labels[glyph.key].text)
    center = (glyph.rect.y0 + glyph.rect.y1) / 2
    expected = line.baseline - _CENTER_RATIO.get(kind or "", 0.3) * line.size
    return abs(center - expected) > line.size * _SHIFT


def _main_glyphs(line: _Line, token: list[Glyph], labels: dict[Key, Label]) -> list[Glyph]:
    """去掉上下标。N 下面的 supplier 和 N 挨在一个词里，不能让 N 看起来像单词的一部分。"""
    kept = []
    for glyph in token:
        size = _glyph_size(glyph, labels, sure=False)
        if size is None or not _off_size(glyph, line, labels, size):
            kept.append(glyph)
    return kept or token


def _mono_tokens(line: _Line, reader: Type3Reader) -> set[int]:
    """一半以上的字母是等宽字形的词。挨着等宽词、字距也是格宽的词（SELECT、AS、FROM）跟着算。"""
    found: dict[int, float] = {}
    for index, token in enumerate(line.tokens):
        # 只看字母：(year) 的括号是正体。下标也不算：N 下面等宽字的 supplier 不让 N 变成等宽字
        letters = [
            glyph for glyph in token
            if _label_text(glyph, reader).isalpha() and not _is_script(glyph, line, reader)
        ]
        known = [reader.mono[glyph.key] for glyph in letters if glyph.key in reader.mono]
        # 等宽字和正文是两套位图，词里一半字母是等宽字形就是等宽词（(day) 的 y 只出现过一次）
        if letters and len(known) * 2 >= len(letters):
            found[index] = statistics.median(known)
    # 字距正好是等宽格宽、字形又从没出现在比例字体的词里（FROM 这一行没有别的等宽词）
    pitches = set(reader.mono.values())
    for index, token in enumerate(line.tokens):
        gaps = _steps(token)
        if index in found or not gaps or min(gaps) <= 0:
            continue
        if any(reader.roles.get(glyph.key) in ("math", "symbol") or glyph.key in reader.proportional
               for glyph in token):
            continue
        pitch = next((value for value in pitches if all(abs(gap - value) <= value * 0.12 for gap in gaps)), None)
        if pitch is not None:
            found[index] = pitch
    grown = bool(found)
    while grown:
        grown = False
        for index, token in enumerate(line.tokens):
            near = [found[other] for other in (index - 1, index + 1) if other in found]
            if index in found or not near:
                continue
            # part p：等宽词旁边的单个变量仍是公式。So、the 这种在正文里出现过的字形也不跟着走
            if any(reader.roles.get(glyph.key) in ("math", "symbol") or glyph.key in reader.proportional
                   for glyph in token):
                continue
            if all(abs(gap - near[0]) <= near[0] * 0.12 for gap in _steps(token)):
                found[index] = near[0]
                grown = True
    return set(found)


def _mono_kinds(pages: dict[int, list[_Line]], labels: dict[Key, Label]) -> dict[Key, float]:
    """等宽字体的字形和它的格宽。

    从长词里找：五个字以上、字心距处处一样、又带着 i、l、逗号这种窄字的词。
    正文字体里窄字和邻字的字心距短一截，凑不齐；数字在哪种字体里都等宽，不算。
    等宽字的格宽是 0.525 个字号，each、latter 这种字距也匀的正文词不到 0.48。
    这样的词至少三个落在同一档格宽（正文和脚注各一档），它们的字形就是等宽字形。
    """
    candidates: list[tuple[float, list[Glyph]]] = []
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            gaps = _steps(token)
            if len(token) < 5 or min(gaps) <= 0 or max(gaps) > min(gaps) * 1.15:
                continue
            if statistics.median(gaps) < line.size * _MONO_PITCH:
                continue
            if any(_text_of(glyph, labels) in _NARROW for glyph in token):
                candidates.append((statistics.median(gaps), token))
    clusters: list[list[tuple[float, list[Glyph]]]] = []
    for value, token in sorted(candidates, key=lambda item: item[0]):
        if clusters and value <= clusters[-1][0][0] * 1.04:
            clusters[-1].append((value, token))
        else:
            clusters.append([(value, token)])
    mono: dict[Key, float] = {}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        pitch = statistics.median(value for value, _token in cluster)
        for _value, token in cluster:
            mono.update((glyph.key, pitch) for glyph in token if not _text_of(glyph, labels).isdigit())
    return mono


def _proportional_kinds(pages: dict[int, list[_Line]], mono: dict[Key, float]) -> set[Key]:
    """出现在字距不匀的词里的字形：比例字体。等宽字体的词字距处处一样。

    (year) 这种正体括号包着等宽字的词字距也不匀，词里有等宽字形的不算。
    """
    found: set[Key] = set()
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            gaps = _steps(token)
            if len(token) < 3 or min(gaps) <= 0 or max(gaps) <= min(gaps) * 1.3:
                continue
            if not any(glyph.key in mono for glyph in token):
                found.update(glyph.key for glyph in token)
    return found


def _steps(token: list[Glyph]) -> list[float]:
    return [_middle(right) - _middle(left) for left, right in zip(token, token[1:])]


def _middle(glyph: Glyph) -> float:
    return (glyph.rect.x0 + glyph.rect.x1) / 2


def _label_text(glyph: Glyph, reader: Type3Reader) -> str:
    return _text_of(glyph, reader.labels)


def _span(run: list[Glyph], style: str, line: _Line, reader: Type3Reader) -> dict[str, Any]:
    glyphs = [glyph for glyph in run if glyph is not _SPACE]
    text = "".join(" " if glyph is _SPACE else _label_text(glyph, reader) for glyph in run)
    ink = pymupdf.Rect()
    for glyph in glyphs:
        ink |= glyph.rect
    if style == "script":
        sizes = [size for glyph in glyphs if (size := _glyph_size(glyph, reader.labels))]
        size = round(statistics.median(sizes), 1) if sizes else round(line.size * 0.7, 1)
        baseline = ink.y1
        box = pymupdf.Rect(ink)
    else:
        size = round(line.size, 1)
        baseline = line.baseline
        box = ink | pymupdf.Rect(ink.x0, baseline - size * 0.72, ink.x1, baseline + size * 0.22)
    # 公式按这个框从原页裁剪，四周留一点余量，位图边上的像素不被切掉
    box = pymupdf.Rect(box.x0 - _PAD, box.y0 - _PAD, box.x1 + _PAD, box.y1 + _PAD)
    font = {
        "text": _TEXT_FONT, "mono": _MONO_FONT, "math": _MATH_FONT, "symbol": _SYMBOL_FONT, "script": _MATH_FONT,
    }[style]
    return {
        "text": text,
        "bbox": (box.x0, box.y0, box.x1, box.y1),
        "size": size,
        "font": font,
        "flags": 0,
        "color": glyphs[0].color if glyphs else 0,
        "origin": (ink.x0, baseline),
        # 擦原文按这些起笔点认字。点阵字的字体框比墨迹高出半行，按框认会认到别人的字
        "origins": [glyph.origin for glyph in glyphs if glyph.origin is not None],
        "ascender": 0.75,
        "descender": -0.25,
    }


def _group_lines(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """行距内、横向交叠、字号相近的行并成一块，和 PyMuPDF 对普通字体的分块一样。分段交给后面。

    行距按基线量。行框会被高括号、下标撑开，两行的框叠在一起不说明什么。
    """
    ordered = sorted(lines, key=lambda line: (_baseline(line), float(line["bbox"][0])))
    groups: list[list[dict[str, Any]]] = []
    for line in ordered:
        box = pymupdf.Rect(line["bbox"])
        size = _line_size(line)
        best: list[dict[str, Any]] | None = None
        for group in groups:
            prev = pymupdf.Rect(group[-1]["bbox"])
            prev_size = _line_size(group[-1])
            step = _baseline(line) - _baseline(group[-1])
            if not (0 < step <= max(size, prev_size) * _LEADING):
                continue
            if min(box.x1, prev.x1) - max(box.x0, prev.x0) <= 0:
                continue
            if max(size, prev_size) > min(size, prev_size) * 1.15:
                continue
            best = group
        if best is None:
            groups.append([line])
        else:
            best.append(line)
    return groups


def _baseline(line: dict[str, Any]) -> float:
    spans = [span for span in line["spans"] if span["font"] in (_TEXT_FONT, _MONO_FONT)] or line["spans"]
    return float(max(spans, key=lambda span: float(span["size"]))["origin"][1])


def _line_size(line: dict[str, Any]) -> float:
    sizes = [float(span["size"]) for span in line["spans"] if span["font"] in (_TEXT_FONT, _MONO_FONT)]
    return max(sizes) if sizes else max(float(span["size"]) for span in line["spans"])


# ---- 共用 ----


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


def _union_bbox(lines: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    box = pymupdf.Rect()
    for line in lines:
        raw = line.get("bbox")
        if raw:
            box |= pymupdf.Rect(raw)
    return (box.x0, box.y0, box.x1, box.y1)


def _center_in(rect: pymupdf.Rect, box: pymupdf.Rect) -> bool:
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    return box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1


def _overlap(left: pymupdf.Rect, right: pymupdf.Rect) -> float:
    return min(left.x1, right.x1) - max(left.x0, right.x0)
