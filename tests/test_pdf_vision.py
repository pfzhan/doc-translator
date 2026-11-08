import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pymupdf
import pytest

from app.formats import pdf_vision
from app.formats.pdf_layout import PageGeometry
from app.formats.pdf_roles import HeuristicLayout, LayoutItem, Role
from app.formats.pdf_vision import (
    FormulaDetector,
    FormulaModel,
    ModelLayout,
    SavedPages,
    formula_detector,
    layout_detector,
)

# A box the fake model finds: rectangle in page coordinates, score, class
Box = tuple[pymupdf.Rect, float, int]

if TYPE_CHECKING:
    # Optional at run time, so the test that needs it imports it with importorskip
    from onnxruntime import SessionOptions

NAMES = "{0: 'title', 1: 'plain text', 3: 'figure', 4: 'figure_caption', 8: 'isolate_formula'}"


class _Meta:
    custom_metadata_map = {"names": NAMES}


class _Input:
    name = "images"


class FakeSession:
    """Take a box in page coordinates and convert it to pixel coordinates of the model input (long side scaled to the
    input's long side, padded on both sides)."""

    def __init__(self, page_rect: pymupdf.Rect, boxes: list[Box]) -> None:
        self.page_rect = page_rect
        self.boxes = boxes
        self.inputs: list[tuple[int, ...]] = []

    def get_modelmeta(self) -> _Meta:
        return _Meta()

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs: list[str] | None, feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
        batch = feeds["images"]
        self.inputs.append(batch.shape)
        height, width = batch.shape[2:]
        zoom = max(height, width) / max(self.page_rect.width, self.page_rect.height)
        left = (width - round(self.page_rect.width * zoom)) // 2
        top = (height - round(self.page_rect.height * zoom)) // 2
        rows = [
            [rect.x0 * zoom + left, rect.y0 * zoom + top, rect.x1 * zoom + left, rect.y1 * zoom + top, score, cls]
            for rect, score, cls in self.boxes
        ]
        rows.append([0, 0, 10, 10, 0.1, 1])  # score too low, ignored
        return [np.array([rows], dtype=np.float32)]


def _doc(width: float = 400, height: float = 500) -> pymupdf.Document:
    doc = pymupdf.open()
    doc.new_page(width=width, height=height)
    return doc


def _item(text: str, rect: pymupdf.Rect) -> LayoutItem:
    return LayoutItem(0, rect, text, 10, False)


def _geo(figures: Iterable[pymupdf.Rect] = ()) -> PageGeometry:
    return PageGeometry.from_rects(400, [pymupdf.Rect(40, 40, 360, 54)], list(figures), 500)


def _corners(rect: pymupdf.Rect) -> tuple[float, float, float, float]:
    return rect.x0, rect.y0, rect.x1, rect.y1


def _near(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    """Every corner coordinate of a is within a point of b."""
    return all(abs(p - q) < 1 for p, q in zip(_corners(a), _corners(b)))


def test_model_marks_chart_labels_and_display_formulas_as_figures() -> None:
    """Legend text and display formulas without vector boxes or captions are body text to the rules; those the model
    boxes as figures and formulas keep the original."""
    doc = _doc()
    legend = pymupdf.Rect(60, 100, 160, 112)
    formula = pymupdf.Rect(120, 200, 280, 216)
    body = pymupdf.Rect(40, 300, 360, 340)
    session = FakeSession(
        doc[0].rect,
        [
            (pymupdf.Rect(50, 80, 350, 180), 0.9, 3),
            (pymupdf.Rect(110, 195, 290, 220), 0.8, 8),
            (pymupdf.Rect(38, 298, 362, 342), 0.95, 1),
        ],
    )
    items = [
        _item("ResNet-20 ResNet-56", legend),
        _item("MultiHead(Q, K, V) = Concat(head1, ..., headh)", formula),
        _item("A body paragraph that explains the figure above in plain words.", body),
    ]
    roles = ModelLayout(doc, session).classify(items, [_geo()], median=10)
    assert roles == [Role.FIGURE, Role.FIGURE, Role.BODY]
    # A 400 × 500 page: long side 1024, short side 820 padded to 832, the next multiple of 32, not to a square
    assert session.inputs == [(1, 3, 1024, 832)]


def test_model_titles_and_captions_do_not_pull_text_out_of_a_figure() -> None:
    """When the model sees a label in a figure as a caption or heading, text the rules already found inside the figure
    stays in the figure."""
    doc = _doc()
    figure = pymupdf.Rect(60, 60, 340, 260)
    label = pymupdf.Rect(100, 100, 150, 112)
    heading = pymupdf.Rect(40, 300, 150, 314)
    session = FakeSession(
        doc[0].rect,
        [
            (pymupdf.Rect(95, 95, 155, 115), 0.7, 4),
            (pymupdf.Rect(38, 298, 160, 316), 0.9, 0),
        ],
    )
    items = [_item("none 1", label), _item("Acknowledgements", heading)]
    rule = HeuristicLayout().classify(items, [_geo([figure])], median=10)
    assert rule == [Role.FIGURE, Role.BODY]
    roles = ModelLayout(doc, session).classify(items, [_geo([figure])], median=10)
    assert roles == [Role.FIGURE, Role.HEADING]


def test_a_body_sentence_under_a_caption_is_not_a_caption() -> None:
    """Body text right below a caption that starts with a figure number (Fig. 19.15 sketches ...) and that the model
    boxes as a caption is still body text; real captions are still recognized."""
    doc = _doc()
    caption = pymupdf.Rect(60, 270, 340, 292)
    body = pymupdf.Rect(40, 300, 360, 312)
    session = FakeSession(
        doc[0].rect,
        [
            (pymupdf.Rect(58, 268, 342, 294), 0.5, 4),
            (pymupdf.Rect(38, 298, 362, 314), 0.8, 4),
        ],
    )
    items = [
        _item("Figure 19.15 A simplified outline of computing the span scores", caption),
        _item("Fig. 19.15 sketches the architecture. The input word tokens are embedded by", body),
    ]
    roles = ModelLayout(doc, session).classify(items, [_geo()], median=10)
    assert roles[1] is Role.BODY
    assert roles[0] is Role.CAPTION


def test_boxes_map_back_through_page_rotation() -> None:
    """A rotated page is rendered rotated, and boxes must be rotated back to text coordinates."""
    doc = _doc(400, 500)
    page = doc[0]
    page.set_rotation(90)
    target = pymupdf.Rect(40, 300, 360, 340)
    shown = target * page.rotation_matrix
    session = FakeSession(page.rect, [(shown, 0.9, 3)])
    [region] = ModelLayout(doc, session).regions(0)
    assert region.label == "figure"
    assert _near(region.rect, target)
    # Rotated, the page is landscape, so the padding is at top and bottom
    assert session.inputs == [(1, 3, 832, 1024)]


def test_without_a_usable_model_the_rules_are_used(tmp_path: Path) -> None:
    """Disabled, a missing file or a file that is not a model all fall back to the rules, and translation does not
    fail."""
    assert isinstance(layout_detector(_doc()), HeuristicLayout)
    assert isinstance(layout_detector(_doc(), str(tmp_path / "missing.onnx")), HeuristicLayout)
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not a model")
    assert isinstance(layout_detector(_doc(), str(broken)), HeuristicLayout)


def test_text_regions_are_the_plain_text_boxes_and_the_page_is_read_once() -> None:
    """Paragraph joining needs the body-text regions boxed by the model. Pages already inferred when deciding roles
    are not inferred again."""
    doc = _doc()
    body = pymupdf.Rect(38, 298, 362, 342)
    session = FakeSession(doc[0].rect, [(body, 0.95, 1), (pymupdf.Rect(110, 195, 290, 220), 0.8, 8)])
    layout = ModelLayout(doc, session)
    layout.classify(
        [_item("A body paragraph that explains the figure.", pymupdf.Rect(40, 300, 360, 340))], [_geo()], median=10
    )
    [region] = layout.text_regions(0)
    assert _near(region, body)
    assert len(session.inputs) == 1


class _FormulaMeta(_Meta):
    custom_metadata_map = {"names": "{0: 'embedding', 1: 'isolated'}"}


class FormulaSession(FakeSession):
    """YOLO exported by Ultralytics: output [1, 4 + classes, N], each column center, width, height and class scores,
    without NMS."""

    def get_modelmeta(self) -> _FormulaMeta:
        return _FormulaMeta()

    def run(self, _outputs: list[str] | None, feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
        height, width = feeds["images"].shape[2:]
        self.inputs.append(feeds["images"].shape)
        zoom = max(height, width) / max(self.page_rect.width, self.page_rect.height)
        left = (width - round(self.page_rect.width * zoom)) // 2
        top = (height - round(self.page_rect.height * zoom)) // 2
        columns: list[list[float]] = []
        for rect, score, cls in self.boxes:
            x0, y0, x1, y1 = rect.x0 * zoom + left, rect.y0 * zoom + top, rect.x1 * zoom + left, rect.y1 * zoom + top
            scores = [0.0, 0.0]
            scores[cls] = score
            columns.append([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, *scores])
        return [np.array([columns], dtype=np.float32).transpose(0, 2, 1)]


def test_formula_boxes_keep_the_best_of_overlapping_ones() -> None:
    """One formula often has several overlapping boxes, and only the highest-scoring one is kept. Boxes scoring too low
    are dropped, and display formulas are not inline formulas"""
    doc = _doc()
    inline = pymupdf.Rect(100, 100, 140, 112)
    session = FormulaSession(
        doc[0].rect,
        [
            (inline, 0.9, 0),
            (pymupdf.Rect(101, 100, 141, 112), 0.6, 0),
            (pymupdf.Rect(200, 100, 230, 112), 0.1, 0),
            (pymupdf.Rect(80, 200, 320, 240), 0.8, 1),
        ],
    )
    detector = FormulaDetector(doc, [session])
    assert sorted(region.label for region in detector.regions(0)) == ["embedding", "isolated"]
    [found] = detector.inline(0)
    assert _near(found, inline)


def test_formula_detector_needs_an_inline_formula_class(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Disabled or a missing file means rules only. A path to the layout model (no inline formula class) is not used
    either"""
    assert formula_detector(_doc()) is None
    assert formula_detector(_doc(), FormulaModel(str(tmp_path / "missing.onnx"))) is None
    doc = _doc()
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [FakeSession(doc[0].rect, [])])
    assert formula_detector(doc, FormulaModel("layout.onnx")) is None
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [FormulaSession(doc[0].rect, [])])
    assert isinstance(formula_detector(doc, FormulaModel("mfd.onnx")), FormulaDetector)


def test_formula_pages_come_in_page_order_at_the_chosen_size() -> None:
    """Boxes yielded page by page are the same as inferring each page alone (the next page is inferred in a background
    thread). Pages are rendered at the input size from the settings"""
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=400, height=500)
    box = pymupdf.Rect(100, 100, 140, 112)

    class PerPage(FormulaSession):
        """Inference number n returns a box moved down by n * 100, showing which page a box belongs to"""

        def run(self, outputs: list[str] | None, feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
            shift = len(self.inputs) * 100
            self.boxes = [(box + (0, shift, 0, shift), 0.9, 0)]
            return super().run(outputs, feeds)

    session = PerPage(doc[0].rect, [])
    pages = list(FormulaDetector(doc, [session], 768).pages())
    assert [round(found.y0) for [found] in pages] == [100, 200, 300]
    assert session.inputs == [(1, 3, 768, 640)] * 3


def test_formula_inference_splits_the_cores_between_two_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatic: with 4 or more cores two pages at once with the cores split, with fewer cores one page using all of
    them. Explicit settings are followed. Inference threads never spin"""
    seen: list[tuple[int, bool, int]] = []
    doc = _doc()

    def usable(path: str, threads: int = 0, spin: bool = True, count: int = 1) -> list[FakeSession]:
        seen.append((threads, spin, count))
        return [FormulaSession(doc[0].rect, []) for _ in range(count)]

    def sessions(model: FormulaModel) -> int:
        detector = formula_detector(doc, model)
        assert detector is not None
        return len(detector.sessions)

    monkeypatch.setattr(pdf_vision, "_usable_sessions", usable)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert sessions(FormulaModel("mfd.onnx")) == 2
    assert seen == [(3, False, 2)]
    seen.clear()
    assert sessions(FormulaModel("mfd.onnx", workers=3, threads=1)) == 3
    assert seen == [(1, False, 3)]
    seen.clear()
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert sessions(FormulaModel("mfd.onnx")) == 1
    assert seen == [(2, False, 1)]


def test_formula_detector_uses_the_copies_that_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """When later sessions fail to load, the loaded ones are used; only when the first fails are the rules used
    alone"""
    doc = _doc()
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [FormulaSession(doc[0].rect, [])])
    detector = formula_detector(doc, FormulaModel("mfd.onnx", workers=3))
    assert detector is not None and len(detector.sessions) == 1
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [])
    assert formula_detector(doc, FormulaModel("mfd.onnx", workers=3)) is None


def test_several_pages_infer_at_once_each_with_its_own_copy() -> None:
    """With two sessions, pages alternate between them in order, a session never infers two pages at once, and boxes
    are still yielded in page order"""
    import threading
    import time

    doc = pymupdf.open()
    for _ in range(5):
        doc.new_page(width=400, height=500)
    box = pymupdf.Rect(100, 100, 140, 112)
    done: list[str] = []

    class Copy(FormulaSession):
        def __init__(self, name: str, pause: float) -> None:
            super().__init__(doc[0].rect, [])
            self.name = name
            self.pause = pause
            self.lock = threading.Lock()

        def run(self, outputs: list[str] | None, feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
            assert self.lock.acquire(blocking=False), "同一份模型同时推了两页"
            try:
                time.sleep(self.pause)
                shift = len(done) * 100
                done.append(self.name)
                self.boxes = [(box + (0, shift, 0, shift), 0.9, 0)]
                return super().run(outputs, feeds)
            finally:
                self.lock.release()

    # a is slower than b: when b finishes first, the next page for a waits until a is free
    copies = [Copy("a", 0.05), Copy("b", 0.01)]
    detector = FormulaDetector(doc, copies)
    pages = list(detector.pages())
    assert len(pages) == 5 and sorted(done) == ["a", "a", "a", "b", "b"]
    assert [len(c.inputs) for c in copies] == [3, 2]
    assert sorted(detector.isolated) == [0, 1, 2, 3, 4]


def test_formula_sessions_do_not_spin_while_waiting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Inference threads that do not spin give up their cores while waiting; each copy of a model is its own
    session"""
    # Optional: the models need it, translation does not (see the README)
    onnxruntime = pytest.importorskip("onnxruntime")

    made: list[str | None] = []
    broken: set[int] = set()

    class Session:
        def __init__(self, path: str, options: "SessionOptions", providers: list[str]) -> None:
            if len(made) in broken:
                raise RuntimeError("out of memory")
            try:
                made.append(options.get_session_config_entry("session.intra_op.allow_spinning"))
            except RuntimeError:  # not set; onnxruntime spins by default
                made.append(None)

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    pdf_vision.clear_sessions()
    try:
        model = str(tmp_path / "mfd.onnx")
        first, second = pdf_vision.load_sessions(model, 3, spin=False, count=2)
        assert first is not second and made == ["0", "0"]
        # Loaded sessions are kept for later jobs
        assert pdf_vision.load_sessions(model, 3, spin=False, count=2) == [first, second]
        assert pdf_vision.load_sessions(model, 3, spin=False) == [first]
        # The third session fails to load, so the two existing ones are used
        broken.add(2)
        assert pdf_vision.load_sessions(model, 3, spin=False, count=3) == [first, second]
        # Thread settings changed, so the sessions are replaced
        broken.clear()
        assert first not in pdf_vision.load_sessions(model, 2, spin=False)
        pdf_vision.load_session("layout.onnx")
        assert made == ["0", "0", "0", None]
        # After the model is regenerated the new file is loaded
        pdf_vision.clear_sessions()
        assert pdf_vision.load_session("layout.onnx") is not None and len(made) == 5
    finally:
        pdf_vision.clear_sessions()


def test_formula_pages_also_keep_the_display_formula_boxes() -> None:
    """While inline formula boxes are taken page by page, display formula boxes (isolated) are recorded per page, for
    deciding which blocks keep the original"""
    doc = _doc()
    display = pymupdf.Rect(80, 200, 320, 240)
    session = FormulaSession(doc[0].rect, [(pymupdf.Rect(100, 100, 140, 112), 0.9, 0), (display, 0.8, 1)])
    detector = FormulaDetector(doc, [session])
    [inline] = list(detector.pages())
    assert len(inline) == 1
    [found] = detector.isolated[0]
    assert _near(found, display)


def test_layout_inference_stops_when_the_job_is_paused() -> None:
    """The layout model takes over a second per page and runs page by page over the whole book while deciding roles.
    A paused job stops at the next page"""
    import asyncio
    import time

    from app import background

    doc = pymupdf.open()
    for _ in range(100):
        doc.new_page(width=400, height=500)

    class Slow(FakeSession):
        def run(self, outputs: list[str] | None, feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
            time.sleep(0.01)
            return super().run(outputs, feeds)

    session = Slow(doc[0].rect, [])
    layout = ModelLayout(doc, session)

    async def main() -> None:
        task = asyncio.ensure_future(background.run(lambda: [layout.page_regions(p) for p in range(100)]))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert 0 < len(session.inputs) < 50


def test_saved_boxes_come_back_only_for_the_same_pdf_model_and_size(tmp_path: Path) -> None:
    """Boxes are stored by PDF, model file and input size. A change in any of them prevents reading them back; the two
    models are stored separately"""
    import os

    from app.formats.pdf_vision import ModelBoxes

    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    model = tmp_path / "mfd.onnx"
    model.write_bytes(b"x")
    boxes = ModelBoxes(tmp_path / "boxes.json", src)
    pages = [[[[1.0, 2.0, 3.0, 4.0]], []]]
    boxes.keep("formula", str(model), 1024, pages)
    boxes.keep("layout", str(model), 1024, {"0": [["figure", [0.0, 0.0, 5.0, 5.0], 0.9]]})
    again = ModelBoxes(tmp_path / "boxes.json", src)
    assert again.load("formula", str(model), 1024) == pages
    assert again.load("layout", str(model), 1024) == {"0": [["figure", [0.0, 0.0, 5.0, 5.0], 0.9]]}
    assert again.load("formula", str(model), 768) is None
    assert again.load("formula", "", 1024) is None
    stat = model.stat()
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert again.load("formula", str(model), 1024) is None
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    src.write_bytes(b"%PDF-1.4 another")
    assert again.load("formula", str(model), 1024) is None
    # Failing to save (the directory is gone) only means inferring again next time
    ModelBoxes(tmp_path / "gone" / "boxes.json", src).keep("formula", str(model), 1024, pages)


def test_formula_boxes_are_kept_only_after_the_whole_book() -> None:
    """Saved only after the whole document is inferred; stopping halfway on pause saves nothing"""
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=400, height=500)
    session = FormulaSession(doc[0].rect, [(pymupdf.Rect(100, 100, 140, 112), 0.9, 0)])
    kept: list[SavedPages] = []
    detector = FormulaDetector(doc, [session])
    detector.keep = kept.append
    pages = detector.pages()
    next(pages)
    pages.close()
    assert kept == []
    # As during extraction: take page by page and close after the last page
    detector = FormulaDetector(doc, [session])
    detector.keep = kept.append
    pages = detector.pages()
    assert all(len(next(pages)) == 1 for _ in range(3))
    pages.close()
    [saved] = kept
    assert len(saved) == 3 and all(len(inline) == 1 and isolated == [] for inline, isolated in saved)
