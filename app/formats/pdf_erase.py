"""擦掉要重排的原文：按字形归属删内容流里的绘制指令，不按矩形挖。

矩形擦除会连带删掉压在框边上的别人的字（上一行的降部、图里的标签），框留小了
又会漏掉自己的字。这里让 MuPDF 重写内容流，逐个字形问它属于谁：起笔点落在要重排的
span 的基线上、宽度范围内，就是这个 span 的字。点阵字这类没有可靠字体框的，span 直接
带上每个字形的起笔点（origins），按点认。

线和图不归哪个 span。整个落在要重排的行里的线（\\_ 的下划线、行内分数线）和
一像素高的细图跟着删，背景、表格线、插图都不动。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
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

_PAINTED_PATHS = {mupdf.FZ_CULL_PATH_FILL, mupdf.FZ_CULL_PATH_STROKE, mupdf.FZ_CULL_PATH_FILL_STROKE}


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
    def __init__(self, erasure: Erasure):
        self.points: set[tuple[float, float]] = set()
        self.lines: list[_Line] = []
        self.boxes: list[pymupdf.Rect] = []
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
            self.lines.append(_Line(span["origin"][1], float(span.get("size") or bbox.height), bbox.x0, bbox.x1))
        self.boxes.extend(erasure.spanless)

    def owns(self, x: float, y: float) -> bool:
        if _key(x, y) in self.points:
            return True
        for line in self.lines:
            if (
                abs(y - line.baseline) <= line.size * _BASELINE_TOLERANCE
                and line.x0 - _X_TOLERANCE <= x < line.x1
            ):
                return True
        point = pymupdf.Point(x, y)
        return any(box.contains(point) for box in self.boxes)


def _key(x: float, y: float) -> tuple[float, float]:
    return round(x, _ORIGIN_DIGITS), round(y, _ORIGIN_DIGITS)


def _matrix(m) -> pymupdf.Matrix:
    return pymupdf.Matrix(m.a, m.b, m.c, m.d, m.e, m.f)


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
        self.owners = _Owners(erasure)
        self.lines = erasure.lines
        self.removed = 0
        # 回调里的异常 MuPDF 只当成「保留」，记下来再抛
        self.error: BaseException | None = None

    def text_filter(self, ctx, ucsbuf, ucslen, trm, ctm, bbox, *_rest):
        try:
            origin = pymupdf.Point(trm.e, trm.f) * _matrix(ctm) * self.to_page
            if self.owners.owns(origin.x, origin.y):
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
            if any(_covers(line, rect) for line in self.lines):
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
    mupdf.pdf_filter_page_contents(pymupdf._as_pdf_document(page.parent), pymupdf._as_pdf_page(page), options)
    if sanitizer.error is not None:
        raise sanitizer.error
    return sanitizer.removed
