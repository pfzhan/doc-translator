"""Optional layout model: figures, display formulas, headings and captions are boxed by a model from the page
image and merged with the conclusions of the rules.

The formula detection model (FormulaDetector) additionally boxes inline formulas, used during text extraction.

The model is optional. It is used only when enabled on the "版面模型" page with an ONNX file of a YOLO layout
model (such as DocLayout-YOLO DocStructBench) and with onnxruntime installed; if any of these is missing, or the
model fails to load, only the rules (HeuristicLayout) are used and translation proceeds as usual.

The model decides only where it does well: blocks it boxes as figures or display formulas keep the original,
and blocks the rules took for body text may become headings or captions. Blocks it boxes as body text, tables
and running headers and footers follow the rules, and text in figures and numbered headings that the rules
already recognized are not changed by the model. The body-text regions it boxes (text_regions) are used to
rejoin a paragraph cut by a formula (see pdf_merge._region_joins).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import threading
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
# Input sides of both models must be multiples of 32 (the YOLO downsampling stride)
_STRIDE = 32
_MIN_SCORE = 0.4
# Share of a block's area that must fall in a box for the block to belong to it
_COVER = 0.6
# Input side of the formula detection model. It was trained at 768; 1024 finds more single-letter variables at
# 1.8 times the inference time of 768, and 1280 finds no more than 1024 at twice the time again. 768 can be
# chosen in the settings
FORMULA_SIZES = (1024, 768)
_FORMULA_SCORE = 0.25
_FORMULA_IOU = 0.45
_INLINE_FORMULA = "embedding"
_DISPLAY_FORMULA = "isolated"

# Model class names mapped to roles. Classes not listed (body text, tables, running headers and footers) are left
# to the rules
_ROLES = {
    "figure": Role.FIGURE,
    "isolate_formula": Role.FIGURE,
    "formula_caption": Role.FIGURE,
    "title": Role.HEADING,
    "figure_caption": Role.CAPTION,
    "table_caption": Role.CAPTION,
}
# A body sentence starting with a figure or table number (Fig. 19.15 sketches the architecture.): the number is
# followed by a lowercase word. Right below a caption, the model often boxes this sentence as a caption too
_BODY_REFERENCE_RE = re.compile(
    r"(?:Figure|Fig\.|Table|Tab\.)\s*(?:[A-Z]\.?)?\d+(?:\.\d+)*\s+(?-i:[a-z])",
    re.I,
)


@dataclass(frozen=True)
class Region:
    label: str
    rect: pymupdf.Rect
    score: float


def layout_detector(doc: pymupdf.Document, model_path: str = "", boxes: ModelBoxes | None = None) -> LayoutDetector:
    """An empty model_path means the model is off. With boxes, the boxes saved last time are read back, and newly
    inferred pages are saved back."""
    sessions = _usable_sessions(model_path)
    if not sessions:
        return HeuristicLayout()
    layout = ModelLayout(doc, sessions[0])
    if boxes is not None:
        layout.keep_in(boxes, model_path)
    return layout


class ModelBoxes:
    """Boxes found by both models, stored per job in one JSON file (see pdf._saved_plan).

    The boxes depend only on the PDF, the model file, the input size and the code in this file that renders
    pages, runs inference and converts coordinates. When code elsewhere changes and the saved analysis result is
    discarded and the job analyzed again, the boxes are read back and only the rules run again: for a book of
    several hundred pages, formula detection takes about ten minutes and the layout model over twenty."""

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
        """Write a temporary file and rename it, so an exit halfway leaves no broken file. Failing to save only means
        inferring again next time."""
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


# Loaded model sessions per model file: (threads, spin) and the sessions. A change of thread settings replaces
# them, and the old ones are dropped
_SESSIONS: dict[str, tuple[tuple[int, bool], list[Any]]] = {}
_SESSIONS_LOCK = threading.Lock()


def load_sessions(path: str, threads: int = 0, spin: bool = True, count: int = 1) -> list[Any]:
    """count sessions of the same model, one per page when several pages are inferred at once (see
    FormulaDetector.pages). Loaded sessions are kept for later jobs. When later sessions fail to load, the ones
    already loaded are returned; when the first fails, the exception is raised: ImportError without onnxruntime,
    an onnxruntime exception for a bad file."""
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
    """A regenerated model keeps its file name, so kept sessions are dropped and the new file is loaded next time.
    Jobs in the middle of inference keep using the session they have."""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()


def _new_session(path: str, threads: int, spin: bool) -> Any:
    """threads is the number of inference threads; 0 lets onnxruntime decide (all cores). With spin false,
    inference threads do not spin while waiting for work and leave the cores to other running threads."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    if not spin:
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _usable_sessions(path: str, threads: int = 0, spin: bool = True, count: int = 1) -> list[Any]:
    """The usable sessions. Empty when the model is off or unusable."""
    if not path or not Path(path).is_file():
        return []
    try:
        return load_sessions(path, threads, spin, count)
    except Exception:  # noqa: BLE001 - a model problem must not fail the translation; fall back to the rules
        return []


class ModelLayout:
    """Classify by the rules first, then revise with the figures, formulas, headings and captions boxed by the
    model."""

    def __init__(self, doc: pymupdf.Document, session: Any):
        self.doc = doc
        self.session = session
        names = session.get_modelmeta().custom_metadata_map.get("names", "{}")
        self.names: dict[int, str] = ast.literal_eval(names)
        self.input = session.get_inputs()[0].name
        self._pages: dict[int, list[Region]] = {}
        # Where to save after inference (see keep_in); _fresh means pages were newly inferred this time
        self._boxes: tuple[ModelBoxes, str] | None = None
        self._fresh = False

    def keep_in(self, boxes: ModelBoxes, model_path: str) -> None:
        """Read back the boxes saved in boxes, and save them back after prepare when pages were newly inferred."""
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
        """Infer page by page in advance, reporting progress through on_page(pages done, total pages). A whole book
        takes tens of minutes, and without inferring in advance and reporting progress the job page would stay at
        "整理段落" and look stuck."""
        for done, pno in enumerate(pages, 1):
            self.page_regions(pno)
            on_page(done, len(pages))
        if self._boxes is not None and self._fresh:
            boxes, model_path = self._boxes
            boxes.keep(
                "layout",
                model_path,
                _INPUT,
                {str(pno): [[r.label, list(r.rect), r.score] for r in found] for pno, found in self._pages.items()},
            )

    def page_regions(self, pno: int) -> list[Region]:
        """Each page is inferred once, for both roles and paragraph joining. A page takes two or three seconds and a
        whole book tens of minutes; a paused job stops here."""
        if pno not in self._pages:
            background.check_stop()
            self._pages[pno] = self.regions(pno)
            self._fresh = True
        return self._pages[pno]

    def regions(self, pno: int) -> list[Region]:
        """Boxes on a page, in the same coordinates as PyMuPDF text (page rotation not applied)."""
        page = self.doc[pno]
        batch, to_page = _page_batch(page, _INPUT)
        rows = self.session.run(None, {self.input: batch})[0][0]
        return [
            Region(self.names.get(int(cls), str(int(cls))), to_page(x0, y0, x1, y1), float(score))
            for x0, y0, x1, y1, score, cls in rows
            if score >= _MIN_SCORE
        ]


def _page_batch(page: pymupdf.Page, size: int) -> tuple[np.ndarray, Callable[..., pymupdf.Rect]]:
    """Render the page as model input: the long side scaled to size, both sides padded with gray to a multiple of
    _STRIDE. Also returns a function converting boxes in the input back to page coordinates. Padding to a
    size × size square would make over 20% of a letter-size page gray and add as much inference time;
    Ultralytics itself also pads only to a multiple of the stride."""
    zoom = size / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    height = -(-pix.height // _STRIDE) * _STRIDE
    width = -(-pix.width // _STRIDE) * _STRIDE
    canvas = np.full((height, width, 3), _PAD_GRAY, np.uint8)
    top = (height - pix.height) // 2
    left = (width - pix.width) // 2
    canvas[top : top + pix.height, left : left + pix.width] = image
    batch = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def to_page(x0: float, y0: float, x1: float, y1: float) -> pymupdf.Rect:
        rect = pymupdf.Rect((x0 - left) / zoom, (y0 - top) / zoom, (x1 - left) / zoom, (y1 - top) / zoom)
        return rect * page.derotation_matrix

    return batch, to_page


@dataclass(frozen=True)
class FormulaModel:
    """Settings of the formula detection model: the ONNX file, the input side (one of FORMULA_SIZES), how many
    pages are inferred at once and with how many threads each (0 decides by core count, see formula_parallel).
    The last two affect only speed; the analysis result is the same."""

    path: str
    size: int = FORMULA_SIZES[0]
    workers: int = 0
    threads: int = 0


class FormulaDetector:
    """Formula detection model (Pix2Text MFD): class embedding is an inline formula, isolated a display formula.

    The rules find formulas by font. With body text in Times or similar, italic variables use the same font as
    italic words in the text, and x or P(y = 1) are not all found. The model looks at the page image, and the
    inline formulas it boxes are marked on the characters during extraction (see
    pdf_math._mark_detected_formulas). It is a YOLO exported by Ultralytics, with no NMS on the output: each
    column holds center x, y, width, height and the score of each class.
    """

    def __init__(self, doc: pymupdf.Document, sessions: Sequence[Any], size: int = FORMULA_SIZES[0]):
        self.doc = doc
        # Several sessions of the same model; pages() infers several pages at once, one session per page
        self.sessions = list(sessions)
        self.size = size
        names = self.sessions[0].get_modelmeta().custom_metadata_map.get("names", "{}")
        self.names: dict[int, str] = ast.literal_eval(names)
        self.input = self.sessions[0].get_inputs()[0].name
        # Display formula boxes (isolated) per page, recorded by pages() and used to decide which blocks keep the
        # original (see pdf_skip._isolated_formula_ids)
        self.isolated: dict[int, list[pymupdf.Rect]] = {}
        # Where to save after the whole document is inferred (see formula_detector)
        self.keep: Callable[[list], None] | None = None

    def regions(self, pno: int) -> list[Region]:
        batch, to_page = _page_batch(self.doc[pno], self.size)
        return [Region(label, to_page(*box), score) for label, box, score in self._detect(batch)]

    def inline(self, pno: int) -> list[pymupdf.Rect]:
        return [region.rect for region in self.regions(pno) if region.label == _INLINE_FORMULA]

    def pages(self) -> Iterator[list[pymupdf.Rect]]:
        """Yield inline formula boxes page by page. With k sessions, background threads infer the next k pages at
        once, one session per page: while the caller processes page n, the model works on pages n + 1 to n + k.
        Rendering pages and converting coordinates happen in the caller's thread: MuPDF cannot use one document
        from two threads at once, so background threads only run the model."""
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
                # The session used for page n is free and goes to page n + k
                if pno + workers < count:
                    pending.append(submit(pno + workers))
                self.isolated[pno] = [to_page(*box) for label, box, _score in found if label == _DISPLAY_FORMULA]
                inline = [to_page(*box) for label, box, _score in found if label == _INLINE_FORMULA]
                pages.append((inline, self.isolated[pno]))
                # Save as soon as the last page is inferred: the caller closes the generator after taking the last
                # page, so code after the loop never runs. Stopping halfway (pause) saves nothing
                if pno == count - 1 and self.keep is not None:
                    self.keep([[[list(r) for r in kept], [list(r) for r in isolated]] for kept, isolated in pages])
                yield inline

    def _detect(
        self,
        batch: np.ndarray,
        worker: int = 0,
    ) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """Infer with session number worker and return boxes in the model input: (class, input pixel coordinates,
        score). Does not touch MuPDF and can run in a background thread."""
        rows = self.sessions[worker].run(None, {self.input: batch})[0][0].T
        scores = rows[:, 4:]
        classes = scores.argmax(1)
        best = scores.max(1)
        keep = best >= _FORMULA_SCORE
        rows, classes, best = rows[keep], classes[keep], best[keep]
        boxes = np.stack(
            [
                rows[:, 0] - rows[:, 2] / 2,
                rows[:, 1] - rows[:, 3] / 2,
                rows[:, 0] + rows[:, 2] / 2,
                rows[:, 1] + rows[:, 3] / 2,
            ],
            1,
        )
        found = []
        for cls in np.unique(classes):
            index = np.flatnonzero(classes == cls)
            for k in _nms(boxes[index], best[index]):
                found.append(
                    (
                        self.names.get(int(cls), str(int(cls))),
                        tuple(float(v) for v in boxes[index[k]]),
                        float(best[index[k]]),
                    )
                )
        return found


def formula_detector(
    doc: pymupdf.Document,
    model: FormulaModel | None = None,
    boxes: ModelBoxes | None = None,
) -> FormulaDetector | None:
    """model None means off. Returns None, leaving only the rules, when the model is unusable or has no inline
    formula class (a layout model was given). When later sessions fail to load, the loaded ones are used. When
    boxes holds this document's boxes they are read back without loading the model; otherwise they are saved
    after the whole document is inferred."""
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
    """Formula detection boxes read back (see ModelBoxes): no model is loaded; pages() yields the saved inline
    formula boxes page by page and records the display formula boxes."""

    def __init__(self, pages: list):
        self.saved = pages
        self.isolated = {}
        self.keep = None

    def pages(self) -> Iterator[list[pymupdf.Rect]]:
        for pno, (inline, isolated) in enumerate(self.saved):
            self.isolated[pno] = [pymupdf.Rect(rect) for rect in isolated]
            yield [pymupdf.Rect(rect) for rect in inline]


def formula_parallel(workers: int = 0, threads: int = 0) -> tuple[int, int]:
    """How many pages the formula detection model infers at once, and with how many threads each. 0 decides by
    core count: with 4 or more cores, two pages at once with the cores split between them. Inference threads do
    not spin while waiting (see _new_session), so as many threads as cores do not crowd out the main thread
    reading pages. Analyzing a 24-page paper on 6 cores (median of 6 runs each): one page with 4 threads 44 s,
    without spinning 37 s, two pages with 3 threads each 35 s."""
    cores = os.cpu_count() or 1
    workers = workers or (2 if cores >= 4 else 1)
    return workers, threads or max(1, cores // workers)


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Among overlapping boxes of the same class, keep only the one with the highest score."""
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
    """Blocks the model boxes as figures or formulas keep the original. Headings and captions change only blocks
    the rules took for body text: a label in a figure that the model sees as a caption stays in the figure, as
    the rules decided."""
    if model is Role.FIGURE:
        return model
    if model is not None and rule is Role.BODY:
        return model
    return rule


def _covering(rect: pymupdf.Rect, regions: Sequence[Region]) -> Region | None:
    """The box covering most of the block's area. A box covering too little does not count."""
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
