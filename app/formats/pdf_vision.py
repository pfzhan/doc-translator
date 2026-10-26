"""可选的版面模型：图、行间公式、标题、图注由模型从页面图像里框出来，再和规则的结论合并。

公式检测模型（FormulaDetector）另外框出行内公式，提取文字时用。

不是必需的。在「版面模型」页面里开启、给出 YOLO 版面模型的 ONNX 文件
（如 DocLayout-YOLO DocStructBench），并且装了 onnxruntime 时才启用；
少了哪一样、或者模型加载不了，都只用规则（HeuristicLayout），翻译照常进行。

模型只在它擅长的地方说了算：框成图和行间公式的块原样保留，规则当成正文的块
可以改成标题、图注。框成正文的块、表格和页眉页脚仍按规则，规则已经认出来的
图内文字、编号标题不会被模型改掉。框出的正文段落（text_regions）用来把被式子切开的
同一段并回去（见 pdf._region_joins）。
"""
from __future__ import annotations

import ast
import functools
import os
import re
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from .pdf_layout import PageGeometry
from .pdf_roles import HeuristicLayout, LayoutDetector, LayoutItem, Role

_INPUT = 1024
_PAD_GRAY = 114
_MIN_SCORE = 0.4
# 块的面积有这么多落在框里，才算框里的块
_COVER = 0.6
# 公式检测模型的输入边长。训练用的是 768，1024 能多认出单个字母的变量，推理时间是 768 的 1.8 倍；
# 1280 认出的不比 1024 多，推理时间又翻倍。设置里可以选 768
FORMULA_SIZES = (1024, 768)
_FORMULA_SCORE = 0.25
_FORMULA_IOU = 0.45
_INLINE_FORMULA = "embedding"
_DISPLAY_FORMULA = "isolated"

# 模型的类别名到角色。不在这里的类别（正文、表格、页眉页脚）交给规则
_ROLES = {
    "figure": Role.FIGURE,
    "isolate_formula": Role.FIGURE,
    "formula_caption": Role.FIGURE,
    "title": Role.HEADING,
    "figure_caption": Role.CAPTION,
    "table_caption": Role.CAPTION,
}
# 正文里以图表编号起头的一句（Fig. 19.15 sketches the architecture.）：编号后面接着小写的词。
# 这句紧挨在图注下面时模型常把它也框成图注
_BODY_REFERENCE_RE = re.compile(
    r"(?:Figure|Fig\.|Table|Tab\.)\s*(?:[A-Z]\.?)?\d+(?:\.\d+)*\s+(?-i:[a-z])", re.I,
)


@dataclass(frozen=True)
class Region:
    label: str
    rect: pymupdf.Rect
    score: float


def layout_detector(doc: pymupdf.Document, model_path: str = "") -> LayoutDetector:
    """model_path 为空表示没开模型。"""
    session = _usable_session(model_path)
    return HeuristicLayout() if session is None else ModelLayout(doc, session)


@functools.lru_cache(maxsize=2)
def load_session(path: str, threads: int = 0) -> Any:
    """加载模型。threads 是推理用的线程数，0 表示 onnxruntime 自己定（用满所有核）。
    没装 onnxruntime 时抛 ImportError，文件不对时抛 onnxruntime 的异常。"""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    return onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _usable_session(path: str, threads: int = 0) -> Any | None:
    if not path or not Path(path).is_file():
        return None
    try:
        return load_session(path, threads)
    except Exception:  # noqa: BLE001  翻译不因为模型出问题而失败，退回规则
        return None


class ModelLayout:
    """先按规则分好，再用模型框出来的图、公式、标题、图注改写。"""

    def __init__(self, doc: pymupdf.Document, session: Any):
        self.doc = doc
        self.session = session
        names = session.get_modelmeta().custom_metadata_map.get("names", "{}")
        self.names: dict[int, str] = ast.literal_eval(names)
        self.input = session.get_inputs()[0].name
        self._pages: dict[int, list[Region]] = {}

    def classify(
        self,
        items: Sequence[LayoutItem],
        geometries: Sequence[PageGeometry],
        median: float,
    ) -> list[Role]:
        roles = HeuristicLayout().classify(items, geometries, median)
        regions = {page: self.page_regions(page) for page in {item.page for item in items}}
        out: list[Role] = []
        for item, role in zip(items, roles):
            found = _covering(item.rect, regions.get(item.page, []))
            model = _ROLES.get(found.label) if found is not None else None
            if model is Role.CAPTION and _BODY_REFERENCE_RE.match(item.text.lstrip()):
                model = None
            out.append(_merge(role, model))
        return out

    def text_regions(self, pno: int) -> list[pymupdf.Rect]:
        return [region.rect for region in self.page_regions(pno) if region.label == "plain text"]

    def page_regions(self, pno: int) -> list[Region]:
        """每页只推一次：定角色和并段都要用。"""
        if pno not in self._pages:
            self._pages[pno] = self.regions(pno)
        return self._pages[pno]

    def regions(self, pno: int) -> list[Region]:
        """一页上的框，坐标和 PyMuPDF 的文字坐标一致（不转页面旋转）。"""
        page = self.doc[pno]
        batch, to_page = _page_batch(page, _INPUT)
        rows = self.session.run(None, {self.input: batch})[0][0]
        return [
            Region(self.names.get(int(cls), str(int(cls))), to_page(x0, y0, x1, y1), float(score))
            for x0, y0, x1, y1, score, cls in rows if score >= _MIN_SCORE
        ]


def _page_batch(page: pymupdf.Page, size: int) -> tuple[np.ndarray, Callable[..., pymupdf.Rect]]:
    """页面画成模型的输入：长边缩到 size，居中补灰边成正方形。另给把输入里的框换回页面坐标的函数。"""
    zoom = size / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    canvas = np.full((size, size, 3), _PAD_GRAY, np.uint8)
    top = (size - pix.height) // 2
    left = (size - pix.width) // 2
    canvas[top:top + pix.height, left:left + pix.width] = image
    batch = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def to_page(x0: float, y0: float, x1: float, y1: float) -> pymupdf.Rect:
        rect = pymupdf.Rect((x0 - left) / zoom, (y0 - top) / zoom, (x1 - left) / zoom, (y1 - top) / zoom)
        return rect * page.derotation_matrix

    return batch, to_page


@dataclass(frozen=True)
class FormulaModel:
    """公式检测模型的设置：ONNX 文件和输入边长（FORMULA_SIZES 之一）。"""
    path: str
    size: int = FORMULA_SIZES[0]


class FormulaDetector:
    """公式检测模型（Pix2Text MFD）：类别 embedding 是行内公式，isolated 是行间公式。

    规则按字体认公式，正文用 Times 之类时，斜体变量和正文里的斜体词是同一种字体，
    x、P(y = 1) 认不全。模型看页面图像，框出的行内公式在提取时标到字上（见 pdf._mark_detected_formulas）。
    它是 Ultralytics 导出的 YOLO，输出没做 NMS：每列是中心 x、y、宽、高和各类的分。
    """

    def __init__(self, doc: pymupdf.Document, session: Any, size: int = FORMULA_SIZES[0]):
        self.doc = doc
        self.session = session
        self.size = size
        names = session.get_modelmeta().custom_metadata_map.get("names", "{}")
        self.names: dict[int, str] = ast.literal_eval(names)
        self.input = session.get_inputs()[0].name
        # pages() 顺带记下的每页行间公式框（isolated），定哪些块留原文时用（见 pdf._isolated_formula_ids）
        self.isolated: dict[int, list[pymupdf.Rect]] = {}

    def regions(self, pno: int) -> list[Region]:
        batch, to_page = _page_batch(self.doc[pno], self.size)
        return [Region(label, to_page(*box), score) for label, box, score in self._detect(batch)]

    def inline(self, pno: int) -> list[pymupdf.Rect]:
        return [region.rect for region in self.regions(pno) if region.label == _INLINE_FORMULA]

    def pages(self) -> Iterator[list[pymupdf.Rect]]:
        """逐页给出行内公式框。取第 n 页时第 n + 1 页已交给后台线程推理，调用方处理第 n 页的同时
        模型在算下一页，一页的时间是两者里长的那个。画页面、换算坐标都在调用方的线程里做：
        MuPDF 不能在两个线程里同时用同一个文档，后台线程只跑模型。"""
        count = self.doc.page_count
        with ThreadPoolExecutor(max_workers=1) as pool:

            def submit(pno: int) -> tuple[Future, Callable[..., pymupdf.Rect]]:
                batch, to_page = _page_batch(self.doc[pno], self.size)
                return pool.submit(self._detect, batch), to_page

            pending = submit(0) if count else None
            for pno in range(count):
                future, to_page = pending
                pending = submit(pno + 1) if pno + 1 < count else None
                found = future.result()
                self.isolated[pno] = [to_page(*box) for label, box, _score in found if label == _DISPLAY_FORMULA]
                yield [to_page(*box) for label, box, _score in found if label == _INLINE_FORMULA]

    def _detect(self, batch: np.ndarray) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """模型输入里的框：(类别, 输入像素坐标, 分)。不碰 MuPDF，可以放在后台线程里跑。"""
        rows = self.session.run(None, {self.input: batch})[0][0].T
        scores = rows[:, 4:]
        classes = scores.argmax(1)
        best = scores.max(1)
        keep = best >= _FORMULA_SCORE
        rows, classes, best = rows[keep], classes[keep], best[keep]
        boxes = np.stack([rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2,
                          rows[:, 0] + rows[:, 2] / 2, rows[:, 1] + rows[:, 3] / 2], 1)
        found = []
        for cls in np.unique(classes):
            index = np.flatnonzero(classes == cls)
            for k in _nms(boxes[index], best[index]):
                found.append((self.names.get(int(cls), str(int(cls))), tuple(float(v) for v in boxes[index[k]]),
                              float(best[index[k]])))
        return found


def formula_detector(doc: pymupdf.Document, model: FormulaModel | None = None) -> FormulaDetector | None:
    """model 为 None 表示没开。模型用不了，或者类别里没有行内公式（填成了版面模型）时返回 None，只用规则。"""
    session = _usable_session(model.path, formula_threads()) if model is not None else None
    if session is None:
        return None
    try:
        detector = FormulaDetector(doc, session, model.size)
    except (ValueError, SyntaxError):
        return None
    return detector if _INLINE_FORMULA in detector.names.values() else None


def formula_threads() -> int:
    """公式检测模型推理和读页面同时进行（见 FormulaDetector.pages），给读页面的主线程和 Python 留出两个核。
    6 核上 4 个线程最快：25 页的论文用满 6 个要多花六成时间，只用 2 个多花两成多。"""
    return max(1, (os.cpu_count() or 1) - 2)


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """同一类里互相重叠的框只留分最高的。"""
    order = list(np.argsort(-scores))
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        rest = np.array(order, dtype=int)
        if not rest.size:
            break
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        overlap = w * h / (area[i] + area[rest] - w * h + 1e-9)
        order = [int(j) for j in rest[overlap <= _FORMULA_IOU]]
    return keep


def _merge(rule: Role, model: Role | None) -> Role:
    """模型框成图或公式的，原样保留。标题、图注只改规则当成正文的块：
    图里的标签被模型看成图注时，仍按规则留在图里。"""
    if model is Role.FIGURE:
        return model
    if model is not None and rule is Role.BODY:
        return model
    return rule


def _covering(rect: pymupdf.Rect, regions: Sequence[Region]) -> Region | None:
    """盖住块面积最多的框。盖得不够多的不算。"""
    area = rect.width * rect.height
    if area < 1:
        return None
    best: Region | None = None
    best_key = (_COVER, 0.0)
    for region in regions:
        hit = rect & region.rect
        if hit.is_empty:
            continue
        key = (round(hit.width * hit.height / area, 2), region.score)
        if key >= best_key:
            best, best_key = region, key
    return best
