"""可选的版面模型：图、行间公式、标题、图注由模型从页面图像里框出来，再和规则的结论合并。

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
import re
from collections.abc import Sequence
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
def load_session(path: str) -> Any:
    """加载模型。没装 onnxruntime 时抛 ImportError，文件不对时抛 onnxruntime 的异常。"""
    import onnxruntime

    return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])


def _usable_session(path: str) -> Any | None:
    if not path or not Path(path).is_file():
        return None
    try:
        return load_session(path)
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
        zoom = _INPUT / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
        image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        canvas = np.full((_INPUT, _INPUT, 3), _PAD_GRAY, np.uint8)
        top = (_INPUT - pix.height) // 2
        left = (_INPUT - pix.width) // 2
        canvas[top:top + pix.height, left:left + pix.width] = image
        batch = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        rows = self.session.run(None, {self.input: batch})[0][0]
        found: list[Region] = []
        for x0, y0, x1, y1, score, cls in rows:
            if score < _MIN_SCORE:
                continue
            rect = pymupdf.Rect(
                (x0 - left) / zoom, (y0 - top) / zoom, (x1 - left) / zoom, (y1 - top) / zoom,
            ) * page.derotation_matrix
            found.append(Region(self.names.get(int(cls), str(int(cls))), rect, float(score)))
        return found


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
