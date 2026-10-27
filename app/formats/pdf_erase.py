"""擦掉要重排的原文：按字形归属删内容流里的绘制指令，不按矩形挖。

矩形擦除会连带删掉压在框边上的别人的字（上一行的降部、图里的标签），框留小了
又会漏掉自己的字。这里让 MuPDF 重写内容流，逐个字形问它属于谁：起笔点落在要重排的
span 的基线上、宽度范围内，就是这个 span 的字。点阵字这类没有可靠字体框的，span 直接
带上每个字形的起笔点（origins），按点认。

线和图不归哪个 span。整个落在要重排的行里的线（\\_ 的下划线、行内分数线）、
紧贴在行基线下的下划线和一像素高的细图跟着删，背景、表格线、插图都不动。
"""
from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import pymupdf
import pymupdf.mupdf as mupdf

# 起笔点和基线的允许偏差（字号的比例）。同一个 span 的字基线完全一致，
# 上下标是另外的 span，另一行至少差一个字号。
_BASELINE_TOLERANCE = 0.15
# 起笔点在 span 左缘外的余量（点）。粗体模拟会把同一个字错开一点再画一遍。
_X_TOLERANCE = 0.5
# 起笔点按这个精度对比，内容流里的坐标和 PyMuPDF 报的坐标只差浮点误差
_ORIGIN_DIGITS = 1
# 细图：老 Ghostscript 把 \_ 和行内分数线画成一像素高的图片
_RULE_IMAGE_HEIGHT = 1.5
# 下划线在基线下多深以内（字号的比例），左右可以比字多出多少（点）
_UNDERLINE_DEPTH = 0.45
_UNDERLINE_REACH = 1.0
# 同一串正文词之间的空档上限（字号的比例）。两端对齐拉宽的空格也在这以内
_WORD_GAP = 0.8

_PAINTED_PATHS = {mupdf.FZ_CULL_PATH_FILL, mupdf.FZ_CULL_PATH_STROKE, mupdf.FZ_CULL_PATH_FILL_STROKE}

# 内容流的词法单元。字符串可以嵌套括号，另外按字扫
_TOKEN_RE = re.compile(rb"\s+|%[^\r\n]*|<<|>>|<[^>]*>|[\[\]{}]|/[^\s()<>\[\]{}/%]*|[^\s()<>\[\]{}/%]+|\(")
_NUMBER_RE = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)")
_INLINE_END_RE = re.compile(rb"\sEI(?=\s|$)")
# 设颜色的操作符：填充色、描边色
_FILL_COLOR = {b"g", b"rg", b"k", b"sc", b"scn"}
_STROKE_COLOR = {b"G", b"RG", b"K", b"SC", b"SCN"}
_PAINT_OPS = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"S", b"s", b"Tj", b"TJ", b"'", b'"', b"Do", b"EI"}


@dataclass
class _Line:
    baseline: float
    size: float
    x0: float
    x1: float


@dataclass
class Erasure:
    """一页上要擦的东西。lines 是要重排的行框：没有 span 的块按它认字，线和细图也按它删。"""

    spans: list[dict] = field(default_factory=list)
    lines: list[pymupdf.Rect] = field(default_factory=list)
    spanless: list[pymupdf.Rect] = field(default_factory=list)
    # 判断一个 span 是不是公式字。公式字下面的横线不当下划线擦
    math: Callable[[dict], bool] = lambda span: False

    def add_block(self, span_lines: Sequence[Sequence[dict]] | None, line_rects: Iterable[pymupdf.Rect]) -> None:
        rects = [pymupdf.Rect(rect) for rect in line_rects]
        self.lines.extend(rects)
        if not span_lines:
            self.spanless.extend(rects)
            return
        for index, spans in enumerate(span_lines):
            shown = [span for span in spans if str(span.get("text") or "").strip()]
            if all("bbox" in span and "origin" in span for span in shown):
                self.spans.extend(shown)
            elif index < len(rects):
                self.spanless.append(rects[index])
        # 并进来的标点碎片只带了行框，没有 span
        if len(rects) > len(span_lines):
            self.spanless.extend(rects[len(span_lines):])


class _Owners:
    """chars 是页面上每个 span 里各个字的起笔点，按 span 的框查。

    span 自己的 origin 只是第一个字的起笔点。第一个字是空格时，它可能落在前面下标的
    基线上，和这一行的字差一点多，拿它当基线会漏擦整段。查得到逐字起笔点的 span 按点认。
    """

    def __init__(self, erasure: Erasure, chars: dict[tuple[float, ...], list[tuple[float, float]]] | None = None):
        self.points: set[tuple[float, float]] = set()
        self.lines: list[_Line] = []
        self.boxes: list[pymupdf.Rect] = []
        chars = chars or {}
        lined: list[dict] = []
        for span in erasure.spans:
            origins = span.get("origins")
            if origins is not None:
                self.points.update(_key(x, y) for x, y in origins)
                continue
            bbox = pymupdf.Rect(span["bbox"])
            direction = span.get("dir", (1, 0))
            if abs(direction[1]) > 0.01:
                self.boxes.append(bbox)
                continue
            size = float(span.get("size") or bbox.height)
            found = chars.get(_box_key(bbox))
            baseline = span["origin"][1]
            if found:
                self.points.update(_key(x, y) for x, y in found)
                baseline = statistics.median(y for _x, y in found)
            # 按基线认仍然要：粗体模拟错开一点再画一遍的字，起笔点和 PyMuPDF 报的不一样
            self.lines.append(_Line(baseline, size, bbox.x0, bbox.x1))
            lined.append(span)
        self.boxes.extend(erasure.spanless)
        # 每个字形都要和这些框比一遍。按 MuPDF 存矩形的单精度先换好，比的结果和 Rect.contains 一样
        self._box_bounds = [single_precision(box) for box in self.boxes]
        self.runs = _prose_runs(
            [line for line, span in zip(self.lines, lined) if not erasure.math(span)]
        )

    def underlines(self, rect: pymupdf.Rect) -> bool:
        """细横线在一串正文词的基线下、降部以内，横向不出这串词：是这些字的下划线（链接常这样画），
        字擦了线也要擦。字体框的下沿常比下划线高一点，按行框认不到。
        公式字下面的横线是分数线、根号线，分母还在原处时擦了分式就断了，不算。"""
        if rect.height > _RULE_IMAGE_HEIGHT:
            return False
        return any(
            run.x0 - _UNDERLINE_REACH <= rect.x0 and rect.x1 <= run.x1 + _UNDERLINE_REACH
            and run.baseline <= rect.y0 and rect.y1 <= run.baseline + run.size * _UNDERLINE_DEPTH
            for run in self.runs
        )

    def owns(self, x: float, y: float) -> bool:
        """x、y 是 MuPDF 换算出的页面坐标，本来就是单精度的值。"""
        if _key(x, y) in self.points:
            return True
        for line in self.lines:
            if (
                abs(y - line.baseline) <= line.size * _BASELINE_TOLERANCE
                and line.x0 - _X_TOLERANCE <= x < line.x1
            ):
                return True
        # 和 MuPDF 判断点在矩形里一样：左、上边算在里面，右、下边不算
        return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in self._box_bounds)


def single_precision(box: pymupdf.Rect) -> tuple[float, float, float, float]:
    """矩形按 MuPDF 存的单精度取值。拿它和单精度的点比，结果和 Rect.contains 一样，省掉每次的换算。"""
    rect = mupdf.FzRect(box.x0, box.y0, box.x1, box.y1)
    return rect.x0, rect.y0, rect.x1, rect.y1


def _prose_runs(lines: list[_Line]) -> list[_Line]:
    """同一基线上挨着的正文 span 连成一串：下划线在词间的空格下面也有一段。
    中间隔着公式（空出一大段）就断开。"""
    runs: list[_Line] = []
    for line in sorted(lines, key=lambda item: (round(item.baseline, 1), item.x0)):
        last = runs[-1] if runs else None
        if (
            last is not None and round(last.baseline, 1) == round(line.baseline, 1)
            and line.x0 - last.x1 <= max(last.size, line.size) * _WORD_GAP
        ):
            runs[-1] = _Line(last.baseline, max(last.size, line.size), last.x0, max(last.x1, line.x1))
        else:
            runs.append(line)
    return runs


def _key(x: float, y: float) -> tuple[float, float]:
    return round(x, _ORIGIN_DIGITS), round(y, _ORIGIN_DIGITS)


def _box_key(box: pymupdf.Rect) -> tuple[float, ...]:
    return tuple(round(value, 1) for value in box)


_BLANKS = frozenset(" \t\n\u00a0\u2002\u2003\u2009\u3000")


def _char_origins(page: pymupdf.Page) -> dict[tuple[float, ...], list[tuple[float, float]]]:
    """页面上每个 span 的逐字起笔点（空白不算），按 span 框查。和提取时用同一组参数。"""
    data = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
    out: dict[tuple[float, ...], list[tuple[float, float]]] = {}
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                # 只跳过真的空格。\r 这类控制字符是没有 Unicode 的字形（CMEX 的大号范数线），照样要擦
                points = [tuple(char["origin"]) for char in span.get("chars") or [] if str(char.get("c") or " ") not in _BLANKS]
                if points:
                    out.setdefault(_box_key(pymupdf.Rect(span["bbox"])), []).extend(points)
    return out


def _page_point(x: float, y: float, ctm, to_page: mupdf.FzMatrix) -> tuple[float, float]:
    """内容流里的点换成页面坐标。和 Point * Matrix 调同一个 MuPDF 函数、结果相同，
    但每个字形都要换一次，省掉 PyMuPDF 包装的那几层参数处理。"""
    point = mupdf.fz_transform_point(mupdf.FzPoint(x, y), mupdf.FzMatrix(ctm.a, ctm.b, ctm.c, ctm.d, ctm.e, ctm.f))
    point = mupdf.fz_transform_point(point, to_page)
    return point.x, point.y


def _rect(r) -> pymupdf.Rect:
    return pymupdf.Rect(r.x0, r.y0, r.x1, r.y1)


class _Sanitizer(mupdf.PdfSanitizeFilterOptions2):
    def __init__(self, page: pymupdf.Page, erasure: Erasure):
        super().__init__()
        # 每个回调都明确开或关，不靠构造函数的默认值
        self.use_virtual_text_filter()
        self.use_virtual_culler()
        self.use_virtual_image_filter()
        self.use_virtual_after_text_object(False)
        self.to_page = page.transformation_matrix
        self.to_page_fz = mupdf.FzMatrix(*self.to_page)
        self.owners = _Owners(erasure, _char_origins(page))
        self.lines = erasure.lines
        self.removed = 0
        # 回调里的异常 MuPDF 只当成「保留」，记下来再抛
        self.error: BaseException | None = None

    def text_filter(self, ctx, ucsbuf, ucslen, trm, ctm, bbox, *_rest):
        try:
            if self.owners.owns(*_page_point(trm.e, trm.f, ctm, self.to_page_fz)):
                self.removed += 1
                return 1
        except BaseException as exc:  # noqa: BLE001
            self.error = self.error or exc
        return 0

    def image_filter(self, ctx, ctm, name, image, scissor):
        # 图片一律原样保留，细图在 culler 里按框删。返回值归 MuPDF 释放，先加一次引用
        return mupdf.ll_fz_keep_image(image)

    def culler(self, ctx, bbox, kind):
        try:
            if kind not in _PAINTED_PATHS and kind != mupdf.FZ_CULL_IMAGE:
                return 0
            rect = _rect(bbox) * self.to_page
            if kind == mupdf.FZ_CULL_IMAGE and rect.height > _RULE_IMAGE_HEIGHT:
                return 0
            if any(_covers(line, rect) for line in self.lines) or self.owners.underlines(rect):
                return 1
        except BaseException as exc:  # noqa: BLE001
            self.error = self.error or exc
        return 0


def _covers(line: pymupdf.Rect, rect: pymupdf.Rect) -> bool:
    return (
        line.x0 <= rect.x0 and rect.x1 <= line.x1
        and line.y0 <= rect.y0 and rect.y1 <= line.y1
        and not math.isinf(rect.width)
    )


class _Factory(mupdf.PdfFilterFactory2):
    def __init__(self, sanitizer: _Sanitizer):
        super().__init__()
        self.use_virtual_filter()
        # 不能叫 options：基类有同名字段，赋值会改掉 C 结构里的指针
        self.sanitizer = sanitizer

    def filter(self, ctx, doc, chain, struct_parents, transform, options):
        return mupdf.ll_pdf_new_sanitize_filter(doc, chain, struct_parents, transform, options, self.sanitizer.internal())


def erase(page: pymupdf.Page, erasure: Erasure) -> int:
    """按归属删掉这一页要重排的字形和线，返回删掉的字形数。

    重写的是页面自己的内容流；用到的表单（Form XObject）按页复制一份再改，
    别的页共用的表单不受影响。
    """
    if not erasure.spans and not erasure.lines:
        return 0
    doc = page.parent
    glyphs = type3_glyphs(page)
    streams = [*page.get_contents(), *(xref for xref, *_rest in page.get_xobjects())]
    bare = any(_paints_in_bare_space(doc.xref_stream(xref) or b"") for xref in streams)
    first_new = doc.xref_length()
    sanitizer = _Sanitizer(page, erasure)
    factory = _Factory(sanitizer)
    options = mupdf.PdfFilterOptions()
    # 构造函数不保证把字段清零，no_update 碰上脏值时过滤结果会被悄悄丢掉
    options.recurse = 1
    options.instance_forms = 1
    options.ascii = 0
    options.no_update = 0
    options.newlines = 0
    options.add_factory(factory.internal())
    mupdf.pdf_filter_page_contents(pymupdf._as_pdf_document(doc), pymupdf._as_pdf_page(page), options)
    restore_glyphs(doc, glyphs)
    if sanitizer.error is not None:
        raise sanitizer.error
    # 重写出来的页面流和表单都是新对象。原流里有不给颜色就画的，补色会改掉它的初始色，整页不补
    if not bare:
        for xref in range(first_new, doc.xref_length()):
            data = doc.xref_stream(xref) if doc.xref_is_stream(xref) else None
            fixed = _restore_dropped_colors(data) if data else data
            if fixed != data:
                doc.update_stream(xref, fixed)
    return sanitizer.removed


def _operators(data: bytes) -> Iterable[tuple[bytes, int, int]]:
    """内容流里的操作符，带上它的操作数起点和它自己的终点。"""
    pos, end, start = 0, len(data), -1
    while pos < end:
        match = _TOKEN_RE.match(data, pos)
        if match is None:
            pos += 1
            continue
        token = match.group()
        if token[:1].isspace() or token[:1] == b"%":
            pos = match.end()
            continue
        if start < 0:
            start = pos
        if token == b"(":
            depth = 0
            while pos < end:
                char = data[pos]
                if char == 0x5C:
                    pos += 2
                    continue
                pos += 1
                depth += (char == 0x28) - (char == 0x29)
                if depth == 0:
                    break
            continue
        pos = match.end()
        if token[:1] in b"<>[]{}/" or _NUMBER_RE.fullmatch(token) or token in (b"true", b"false", b"null"):
            continue
        yield token, start, pos
        start = -1
        if token == b"ID":
            # 内联图的数据是二进制，跳到 EI
            inline = _INLINE_END_RE.search(data, pos)
            pos = end if inline is None else inline.start() + 1


def _paints_in_bare_space(data: bytes) -> bool:
    """换了色彩空间不给颜色就画：按规范画的是这个空间的初始色。"""
    fill = stroke = False
    for op, _start, _end in _operators(data):
        if op == b"cs" or op == b"CS":
            fill, stroke = fill or op == b"cs", stroke or op == b"CS"
        elif op in _FILL_COLOR:
            fill = False
        elif op in _STROKE_COLOR:
            stroke = False
        elif op in _PAINT_OPS and (fill or stroke):
            return True
    return False


def _restore_dropped_colors(data: bytes) -> bytes:
    """MuPDF 重写内容流时，换了色彩空间后的颜色值若和换之前的相同就省掉，可换色彩空间会把
    颜色重置成初始色，白底就画成了黑块。每个后面没跟颜色的 cs、CS 后补回换之前的颜色值。"""
    if re.search(rb"(?<![^\s\]>)])(?:cs|CS)(?![^\s/\[(<%])", data) is None:
        return data
    ops = list(_operators(data))
    current: dict[bool, bytes | None] = {True: None, False: None}
    saved: list[dict[bool, bytes | None]] = []
    pieces: list[bytes] = []
    copied = 0
    for index, (op, start, op_end) in enumerate(ops):
        if op == b"q":
            saved.append(dict(current))
        elif op == b"Q" and saved:
            current = saved.pop()
        elif op in _FILL_COLOR or op in _STROKE_COLOR:
            current[op in _FILL_COLOR] = data[start:op_end - len(op)].strip()
        elif op == b"cs" or op == b"CS":
            fill = op == b"cs"
            follow = ops[index + 1][0] if index + 1 < len(ops) else b""
            values = current[fill]
            if values and follow not in ((b"sc", b"scn") if fill else (b"SC", b"SCN")):
                pieces += [data[copied:op_end], b" ", values, b" scn" if fill else b" SCN"]
                copied = op_end
    if not pieces:
        return data
    pieces.append(data[copied:])
    return b"".join(pieces)


def type3_glyphs(page: pymupdf.Page) -> dict[int, bytes]:
    """这一页用到的 Type3 字体的字形流。MuPDF 递归过滤内容流时也进到字形里，按字形自己的坐标
    问归属，删掉的是全文共用的字形：别的页、贴公式用的原页上同一个字都跟着没了。过滤完原样写回。"""
    doc = page.parent
    glyphs: dict[int, bytes] = {}
    for xref, _ext, kind, *_rest in page.get_fonts(full=True):
        if kind != "Type3" or xref <= 0:
            continue
        procs = doc.xref_get_key(xref, "CharProcs")
        if procs[0] == "xref":
            procs = ("dict", doc.xref_object(int(procs[1].split()[0])))
        for ref in re.findall(r"(\d+) 0 R", procs[1]):
            glyphs[int(ref)] = doc.xref_stream(int(ref))
    return glyphs


def restore_glyphs(doc: pymupdf.Document, glyphs: dict[int, bytes]) -> None:
    for xref, data in glyphs.items():
        if doc.xref_stream(xref) != data:
            doc.update_stream(xref, data)
