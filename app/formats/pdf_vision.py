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
import hashlib
import json
import os
import tempfile
import threading
import re
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from .. import background
from .pdf_layout import PageGeometry
from .pdf_roles import HeuristicLayout, LayoutDetector, LayoutItem, Role

_INPUT = 1024
_PAD_GRAY = 114
# 两个模型的输入边长都要是 32 的倍数（YOLO 的下采样步长）
_STRIDE = 32
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


def layout_detector(doc: pymupdf.Document, model_path: str = "", boxes: ModelBoxes | None = None) -> LayoutDetector:
    """model_path 为空表示没开模型。boxes 给了时读回上次存下的框，推了新的页再存回去。"""
    sessions = _usable_sessions(model_path)
    if not sessions:
        return HeuristicLayout()
    layout = ModelLayout(doc, sessions[0])
    if boxes is not None:
        layout.keep_in(boxes, model_path)
    return layout


class ModelBoxes:
    """两个模型框出的框，按任务存在一个 JSON 文件里（见 pdf._saved_plan）。

    框只取决于 PDF、模型文件、输入尺寸，和这个文件里画页面、推理、换算坐标的代码。改了别处的代码，
    存下的分析结果作废、要重新分析时，直接读回框，只重跑规则：一本几百页的书，公式检测要十分钟上下，
    版面模型要二十多分钟。"""

    def __init__(self, path: Path, src: Path):
        self.path = path
        self.src = src

    def load(self, kind: str, model_path: str, size: int) -> Any | None:
        key = self._key(kind, model_path, size)
        if key is None:
            return None
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8")).get(kind) or {}
        except (OSError, ValueError, AttributeError):
            return None
        return saved.get("pages") if saved.get("key") == key else None

    def keep(self, kind: str, model_path: str, size: int, pages: Any) -> None:
        """先写临时文件再改名，写到一半退出不留坏文件。存不下只是下次要重新推。"""
        key = self._key(kind, model_path, size)
        if key is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data[kind] = {"key": key, "pages": pages}
        name = ""
        try:
            handle, name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump(data, out)
            Path(name).replace(self.path)
        except OSError:
            if name:
                Path(name).unlink(missing_ok=True)

    def _key(self, kind: str, model_path: str, size: int) -> str | None:
        model = Path(model_path) if model_path else None
        if model is None or not model.is_file():
            return None
        stat = model.stat()
        digest = hashlib.sha1(self.src.read_bytes())
        digest.update(Path(__file__).read_bytes())
        digest.update(f"{kind}@{size}:{model.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        digest.update(f"{pymupdf.VersionBind}:{_runtime_version()}".encode())
        return digest.hexdigest()


def _runtime_version() -> str:
    try:
        import onnxruntime
    except ImportError:
        return ""
    return onnxruntime.__version__


# 加载过的模型会话，按模型文件记：（线程数, 空转）和几份会话。线程设置改了就换掉，旧的不留
_SESSIONS: dict[str, tuple[tuple[int, bool], list[Any]]] = {}
_SESSIONS_LOCK = threading.Lock()


def load_sessions(path: str, threads: int = 0, spin: bool = True, count: int = 1) -> list[Any]:
    """同一个模型的 count 份会话，几页同时推理时每页用一份（见 FormulaDetector.pages）。
    加载好的留给后面的任务。后面几份加载不了时返回已经有的几份；第一份就加载不了时抛出异常，
    没装 onnxruntime 时是 ImportError，文件不对时是 onnxruntime 的异常。"""
    with _SESSIONS_LOCK:
        setting, sessions = _SESSIONS.get(path, ((threads, spin), []))
        if setting != (threads, spin):
            sessions = []
        try:
            while len(sessions) < count:
                sessions.append(_new_session(path, threads, spin))
        except Exception:
            if not sessions:
                raise
        _SESSIONS[path] = (threads, spin), sessions
        return sessions[:count]


def load_session(path: str, threads: int = 0, spin: bool = True) -> Any:
    return load_sessions(path, threads, spin)[0]


def clear_sessions() -> None:
    """重新生成模型时文件名不变，留着的会话要扔掉，下次加载新文件。正在推理的任务照用手上那份。"""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()


def _new_session(path: str, threads: int, spin: bool) -> Any:
    """threads 是推理用的线程数，0 表示 onnxruntime 自己定（用满所有核）。
    spin 为假时推理线程等活不空转，把核让给同时在跑的其他线程。"""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    if not spin:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _usable_sessions(path: str, threads: int = 0, spin: bool = True, count: int = 1) -> list[Any]:
    """能用的几份会话。没开模型、模型用不了时是空的。"""
    if not path or not Path(path).is_file():
        return []
    try:
        return load_sessions(path, threads, spin, count)
    except Exception:  # noqa: BLE001  翻译不因为模型出问题而失败，退回规则
        return []


class ModelLayout:
    """先按规则分好，再用模型框出来的图、公式、标题、图注改写。"""

    def __init__(self, doc: pymupdf.Document, session: Any):
        self.doc = doc
        self.session = session
        names = session.get_modelmeta().custom_metadata_map.get("names", "{}")
        self.names: dict[int, str] = ast.literal_eval(names)
        self.input = session.get_inputs()[0].name
        self._pages: dict[int, list[Region]] = {}
        # 推完以后存到哪里（见 keep_in）；_fresh 是这次新推了页
        self._boxes: tuple[ModelBoxes, str] | None = None
        self._fresh = False

    def keep_in(self, boxes: ModelBoxes, model_path: str) -> None:
        """读回 boxes 里存下的框，prepare 推完以后有新推的页时存回去。"""
        self._boxes = (boxes, model_path)
        saved = boxes.load("layout", model_path, _INPUT)
        if isinstance(saved, dict):
            for pno, found in saved.items():
                self._pages[int(pno)] = [Region(label, pymupdf.Rect(rect), score) for label, rect, score in found]

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

    def prepare(self, pages: Sequence[int], on_page: Callable[[int, int], None]) -> None:
        """逐页推好，on_page(已推页数, 总页数) 报进度。整本书要推几十分钟，
        不先推好、不报进度时，任务页一直停在「整理段落」，看着像卡住了。"""
        for done, pno in enumerate(pages, 1):
            self.page_regions(pno)
            on_page(done, len(pages))
        if self._boxes is not None and self._fresh:
            boxes, model_path = self._boxes
            boxes.keep("layout", model_path, _INPUT, {
                str(pno): [[r.label, list(r.rect), r.score] for r in found] for pno, found in self._pages.items()
            })

    def page_regions(self, pno: int) -> list[Region]:
        """每页只推一次：定角色和并段都要用。一页要推两三秒，整本书要几十分钟，任务被暂停时在这里停下。"""
        if pno not in self._pages:
            background.check_stop()
            self._pages[pno] = self.regions(pno)
            self._fresh = True
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
    """页面画成模型的输入：长边缩到 size，两边各补灰边到 _STRIDE 的倍数。另给把输入里的框换回页面坐标的函数。
    补成 size × size 的正方形时，信纸大小的页有两成多是灰边，推理时间也跟着多两成多；
    Ultralytics 自己推理时也只补到步长的倍数。"""
    zoom = size / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    height = -(-pix.height // _STRIDE) * _STRIDE
    width = -(-pix.width // _STRIDE) * _STRIDE
    canvas = np.full((height, width, 3), _PAD_GRAY, np.uint8)
    top = (height - pix.height) // 2
    left = (width - pix.width) // 2
    canvas[top:top + pix.height, left:left + pix.width] = image
    batch = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def to_page(x0: float, y0: float, x1: float, y1: float) -> pymupdf.Rect:
        rect = pymupdf.Rect((x0 - left) / zoom, (y0 - top) / zoom, (x1 - left) / zoom, (y1 - top) / zoom)
        return rect * page.derotation_matrix

    return batch, to_page


@dataclass(frozen=True)
class FormulaModel:
    """公式检测模型的设置：ONNX 文件、输入边长（FORMULA_SIZES 之一），同时推理几页、每页用几个线程
    （0 表示按核数定，见 formula_parallel）。后两个只影响快慢，分析结果不变。"""
    path: str
    size: int = FORMULA_SIZES[0]
    workers: int = 0
    threads: int = 0


class FormulaDetector:
    """公式检测模型（Pix2Text MFD）：类别 embedding 是行内公式，isolated 是行间公式。

    规则按字体认公式，正文用 Times 之类时，斜体变量和正文里的斜体词是同一种字体，
    x、P(y = 1) 认不全。模型看页面图像，框出的行内公式在提取时标到字上（见 pdf._mark_detected_formulas）。
    它是 Ultralytics 导出的 YOLO，输出没做 NMS：每列是中心 x、y、宽、高和各类的分。
    """

    def __init__(self, doc: pymupdf.Document, sessions: Sequence[Any], size: int = FORMULA_SIZES[0]):
        self.doc = doc
        # 同一个模型的几份，pages() 里几页同时推理，每页用一份
        self.sessions = list(sessions)
        self.size = size
        names = self.sessions[0].get_modelmeta().custom_metadata_map.get("names", "{}")
        self.names: dict[int, str] = ast.literal_eval(names)
        self.input = self.sessions[0].get_inputs()[0].name
        # pages() 顺带记下的每页行间公式框（isolated），定哪些块留原文时用（见 pdf._isolated_formula_ids）
        self.isolated: dict[int, list[pymupdf.Rect]] = {}
        # 整本推完以后存到哪里（见 formula_detector）
        self.keep: Callable[[list], None] | None = None

    def regions(self, pno: int) -> list[Region]:
        batch, to_page = _page_batch(self.doc[pno], self.size)
        return [Region(label, to_page(*box), score) for label, box, score in self._detect(batch)]

    def inline(self, pno: int) -> list[pymupdf.Rect]:
        return [region.rect for region in self.regions(pno) if region.label == _INLINE_FORMULA]

    def pages(self) -> Iterator[list[pymupdf.Rect]]:
        """逐页给出行内公式框。有 k 份模型时，后台线程同时推理后面的 k 页，每页用一份：
        调用方处理第 n 页的同时，模型在算第 n + 1 到 n + k 页。画页面、换算坐标都在调用方的线程里做：
        MuPDF 不能在两个线程里同时用同一个文档，后台线程只跑模型。"""
        count = self.doc.page_count
        workers = len(self.sessions)
        pages: list[tuple[list[pymupdf.Rect], list[pymupdf.Rect]]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:

            def submit(pno: int) -> tuple[Future, Callable[..., pymupdf.Rect]]:
                batch, to_page = _page_batch(self.doc[pno], self.size)
                return pool.submit(self._detect, batch, pno % workers), to_page

            pending = deque(submit(pno) for pno in range(min(workers, count)))
            for pno in range(count):
                future, to_page = pending.popleft()
                found = future.result()
                # 第 n 页用的那份模型空出来了，交给第 n + k 页
                if pno + workers < count:
                    pending.append(submit(pno + workers))
                self.isolated[pno] = [to_page(*box) for label, box, _score in found if label == _DISPLAY_FORMULA]
                inline = [to_page(*box) for label, box, _score in found if label == _INLINE_FORMULA]
                pages.append((inline, self.isolated[pno]))
                # 最后一页推完就存：调用方取完最后一页就关掉，走不到循环后面。中途停下（暂停）时不存半本
                if pno == count - 1 and self.keep is not None:
                    self.keep([[[list(r) for r in kept], [list(r) for r in isolated]] for kept, isolated in pages])
                yield inline

    def _detect(
        self, batch: np.ndarray, worker: int = 0,
    ) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """用第 worker 份模型推理，给出模型输入里的框：(类别, 输入像素坐标, 分)。不碰 MuPDF，可以放在后台线程里跑。"""
        rows = self.sessions[worker].run(None, {self.input: batch})[0][0].T
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


def formula_detector(
    doc: pymupdf.Document, model: FormulaModel | None = None, boxes: ModelBoxes | None = None,
) -> FormulaDetector | None:
    """model 为 None 表示没开。模型用不了，或者类别里没有行内公式（填成了版面模型）时返回 None，只用规则。
    多份模型里后面的加载不了时，用已经加载好的几份。boxes 里存着这本的框时直接读回，不加载模型；
    没存着时推完整本存进去。"""
    if model is None:
        return None
    saved = boxes.load("formula", model.path, model.size) if boxes is not None else None
    if isinstance(saved, list) and len(saved) == doc.page_count:
        return SavedFormulas(saved)
    workers, threads = formula_parallel(model.workers, model.threads)
    sessions = _usable_sessions(model.path, threads, spin=False, count=workers)
    if not sessions:
        return None
    try:
        detector = FormulaDetector(doc, sessions, model.size)
    except (ValueError, SyntaxError):
        return None
    if _INLINE_FORMULA not in detector.names.values():
        return None
    if boxes is not None:
        detector.keep = lambda pages: boxes.keep("formula", model.path, model.size, pages)
    return detector


class SavedFormulas(FormulaDetector):
    """读回的公式检测框（见 ModelBoxes）：不加载模型，pages() 逐页给出存下的行内公式框，顺带记下行间公式框。"""

    def __init__(self, pages: list):
        self.saved = pages
        self.isolated = {}
        self.keep = None

    def pages(self) -> Iterator[list[pymupdf.Rect]]:
        for pno, (inline, isolated) in enumerate(self.saved):
            self.isolated[pno] = [pymupdf.Rect(rect) for rect in isolated]
            yield [pymupdf.Rect(rect) for rect in inline]


def formula_parallel(workers: int = 0, threads: int = 0) -> tuple[int, int]:
    """公式检测模型同时推理几页、每页用几个线程。0 表示按核数定：4 核以上两页同时推理，核平分给这两页。
    推理线程等活不空转（见 _new_session），线程总数和核数一样多也不会挤掉读页面的主线程。
    6 核上 24 页的论文分析（各跑 6 次取中位数）：一页 4 个线程 44 s，关掉空转 37 s，两页各 3 个线程 35 s。"""
    cores = os.cpu_count() or 1
    workers = workers or (2 if cores >= 4 else 1)
    return workers, threads or max(1, cores // workers)


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
