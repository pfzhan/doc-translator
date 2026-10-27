import numpy as np
import pymupdf

from app.formats.pdf_layout import PageGeometry
from app.formats.pdf_roles import HeuristicLayout, LayoutItem, Role
from app.formats import pdf_vision
from app.formats.pdf_vision import FormulaDetector, FormulaModel, ModelLayout, formula_detector, layout_detector

NAMES = "{0: 'title', 1: 'plain text', 3: 'figure', 4: 'figure_caption', 8: 'isolate_formula'}"


class _Meta:
    custom_metadata_map = {"names": NAMES}


class _Input:
    name = "images"


class FakeSession:
    """按页面坐标给框，换算成模型输入里的像素坐标（长边缩放到输入的长边、两侧补边）。"""

    def __init__(self, page_rect, boxes):
        self.page_rect = page_rect
        self.boxes = boxes
        self.inputs = []

    def get_modelmeta(self):
        return _Meta()

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feeds):
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
        rows.append([0, 0, 10, 10, 0.1, 1])  # 分数太低，不算
        return [np.array([rows], dtype=np.float32)]


def _doc(width=400, height=500):
    doc = pymupdf.open()
    doc.new_page(width=width, height=height)
    return doc


def _item(text, rect):
    return LayoutItem(0, rect, text, 10, False)


def _geo(figures=()):
    return PageGeometry.from_rects(400, [pymupdf.Rect(40, 40, 360, 54)], list(figures), 500)


def test_model_marks_chart_labels_and_display_formulas_as_figures():
    """图例里的字、行间公式没有矢量框也没有图注时，规则当成正文；模型框成图和公式的留原样。"""
    doc = _doc()
    legend = pymupdf.Rect(60, 100, 160, 112)
    formula = pymupdf.Rect(120, 200, 280, 216)
    body = pymupdf.Rect(40, 300, 360, 340)
    session = FakeSession(doc[0].rect, [
        (pymupdf.Rect(50, 80, 350, 180), 0.9, 3),
        (pymupdf.Rect(110, 195, 290, 220), 0.8, 8),
        (pymupdf.Rect(38, 298, 362, 342), 0.95, 1),
    ])
    items = [
        _item("ResNet-20 ResNet-56", legend),
        _item("MultiHead(Q, K, V) = Concat(head1, ..., headh)", formula),
        _item("A body paragraph that explains the figure above in plain words.", body),
    ]
    roles = ModelLayout(doc, session).classify(items, [_geo()], median=10)
    assert roles == [Role.FIGURE, Role.FIGURE, Role.BODY]
    # 400 × 500 的页：长边 1024，短边 820 补到 32 的倍数 832，不补成正方形
    assert session.inputs == [(1, 3, 1024, 832)]


def test_model_titles_and_captions_do_not_pull_text_out_of_a_figure():
    """图里的标签被模型看成图注或标题时，规则已经认成图内文字的仍留在图里。"""
    doc = _doc()
    figure = pymupdf.Rect(60, 60, 340, 260)
    label = pymupdf.Rect(100, 100, 150, 112)
    heading = pymupdf.Rect(40, 300, 150, 314)
    session = FakeSession(doc[0].rect, [
        (pymupdf.Rect(95, 95, 155, 115), 0.7, 4),
        (pymupdf.Rect(38, 298, 160, 316), 0.9, 0),
    ])
    items = [_item("none 1", label), _item("Acknowledgements", heading)]
    rule = HeuristicLayout().classify(items, [_geo([figure])], median=10)
    assert rule == [Role.FIGURE, Role.BODY]
    roles = ModelLayout(doc, session).classify(items, [_geo([figure])], median=10)
    assert roles == [Role.FIGURE, Role.HEADING]


def test_a_body_sentence_under_a_caption_is_not_a_caption():
    """图注下面紧接着的正文以图号起头（Fig. 19.15 sketches …）、被模型框成图注时，仍是正文；真的图注照样认。"""
    doc = _doc()
    caption = pymupdf.Rect(60, 270, 340, 292)
    body = pymupdf.Rect(40, 300, 360, 312)
    session = FakeSession(doc[0].rect, [
        (pymupdf.Rect(58, 268, 342, 294), 0.5, 4),
        (pymupdf.Rect(38, 298, 362, 314), 0.8, 4),
    ])
    items = [_item("Figure 19.15 A simplified outline of computing the span scores", caption),
             _item("Fig. 19.15 sketches the architecture. The input word tokens are embedded by", body)]
    roles = ModelLayout(doc, session).classify(items, [_geo()], median=10)
    assert roles[1] is Role.BODY
    assert roles[0] is Role.CAPTION


def test_boxes_map_back_through_page_rotation():
    """转过的页面按转过的样子渲染，框要转回文字坐标。"""
    doc = _doc(400, 500)
    page = doc[0]
    page.set_rotation(90)
    target = pymupdf.Rect(40, 300, 360, 340)
    shown = target * page.rotation_matrix
    session = FakeSession(page.rect, [(shown, 0.9, 3)])
    [region] = ModelLayout(doc, session).regions(0)
    assert region.label == "figure"
    assert all(abs(a - b) < 1 for a, b in zip(region.rect, target))
    # 转过以后是横的，补的是上下
    assert session.inputs == [(1, 3, 832, 1024)]


def test_without_a_usable_model_the_rules_are_used(tmp_path):
    """没开、文件不存在、文件不是模型，都退回规则，翻译不因此失败。"""
    assert isinstance(layout_detector(_doc()), HeuristicLayout)
    assert isinstance(layout_detector(_doc(), str(tmp_path / "missing.onnx")), HeuristicLayout)
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"not a model")
    assert isinstance(layout_detector(_doc(), str(broken)), HeuristicLayout)


def test_text_regions_are_the_plain_text_boxes_and_the_page_is_read_once():
    """并段要的是模型框出的正文段落。定角色时已经推过的页不再推一次。"""
    doc = _doc()
    body = pymupdf.Rect(38, 298, 362, 342)
    session = FakeSession(doc[0].rect, [(body, 0.95, 1), (pymupdf.Rect(110, 195, 290, 220), 0.8, 8)])
    layout = ModelLayout(doc, session)
    layout.classify([_item("A body paragraph that explains the figure.", pymupdf.Rect(40, 300, 360, 340))],
                    [_geo()], median=10)
    [region] = layout.text_regions(0)
    assert all(abs(a - b) < 1 for a, b in zip(region, body))
    assert len(session.inputs) == 1


class _FormulaMeta:
    custom_metadata_map = {"names": "{0: 'embedding', 1: 'isolated'}"}


class FormulaSession(FakeSession):
    """Ultralytics 导出的 YOLO：输出 [1, 4 + 类别数, N]，每列是中心、宽高和各类的分，没做 NMS。"""

    def get_modelmeta(self):
        return _FormulaMeta()

    def run(self, _outputs, feeds):
        height, width = feeds["images"].shape[2:]
        self.inputs.append(feeds["images"].shape)
        zoom = max(height, width) / max(self.page_rect.width, self.page_rect.height)
        left = (width - round(self.page_rect.width * zoom)) // 2
        top = (height - round(self.page_rect.height * zoom)) // 2
        columns = []
        for rect, score, cls in self.boxes:
            x0, y0, x1, y1 = rect.x0 * zoom + left, rect.y0 * zoom + top, rect.x1 * zoom + left, rect.y1 * zoom + top
            scores = [0.0, 0.0]
            scores[cls] = score
            columns.append([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, *scores])
        return [np.array([columns], dtype=np.float32).transpose(0, 2, 1)]


def test_formula_boxes_keep_the_best_of_overlapping_ones():
    """同一个公式常有几个重叠的框，只留分最高的。分太低的不要，行间公式不算行内公式"""
    doc = _doc()
    inline = pymupdf.Rect(100, 100, 140, 112)
    session = FormulaSession(doc[0].rect, [
        (inline, 0.9, 0),
        (pymupdf.Rect(101, 100, 141, 112), 0.6, 0),
        (pymupdf.Rect(200, 100, 230, 112), 0.1, 0),
        (pymupdf.Rect(80, 200, 320, 240), 0.8, 1),
    ])
    detector = FormulaDetector(doc, [session])
    assert sorted(region.label for region in detector.regions(0)) == ["embedding", "isolated"]
    [found] = detector.inline(0)
    assert all(abs(a - b) < 1 for a, b in zip(found, inline))


def test_formula_detector_needs_an_inline_formula_class(monkeypatch, tmp_path):
    """没开、文件不存在都只用规则。路径填成版面模型（类别里没有行内公式）也不用"""
    assert formula_detector(_doc()) is None
    assert formula_detector(_doc(), FormulaModel(str(tmp_path / "missing.onnx"))) is None
    doc = _doc()
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [FakeSession(doc[0].rect, [])])
    assert formula_detector(doc, FormulaModel("layout.onnx")) is None
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [FormulaSession(doc[0].rect, [])])
    assert isinstance(formula_detector(doc, FormulaModel("mfd.onnx")), FormulaDetector)


def test_formula_pages_come_in_page_order_at_the_chosen_size():
    """逐页给的框和一页一页单独推的一样（下一页在后台线程里推）。输入尺寸按设置画页面"""
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=400, height=500)
    box = pymupdf.Rect(100, 100, 140, 112)

    class PerPage(FormulaSession):
        """第 n 次推理给出往下挪 n * 100 的框，看得出框对的是哪一页"""

        def run(self, outputs, feeds):
            shift = len(self.inputs) * 100
            self.boxes = [(box + (0, shift, 0, shift), 0.9, 0)]
            return super().run(outputs, feeds)

    session = PerPage(doc[0].rect, [])
    pages = list(FormulaDetector(doc, [session], 768).pages())
    assert [round(found.y0) for [found] in pages] == [100, 200, 300]
    assert session.inputs == [(1, 3, 768, 640)] * 3


def test_formula_inference_splits_the_cores_between_two_pages(monkeypatch):
    """自动时 4 核以上两页同时推理，核平分；核少时一页用满。设置里给了就按设置。推理线程都不空转"""
    seen = []
    doc = _doc()

    def usable(path, threads=0, spin=True, count=1):
        seen.append((threads, spin, count))
        return [FormulaSession(doc[0].rect, []) for _ in range(count)]

    monkeypatch.setattr(pdf_vision, "_usable_sessions", usable)
    monkeypatch.setattr(pdf_vision.os, "cpu_count", lambda: 6)
    assert len(formula_detector(doc, FormulaModel("mfd.onnx")).sessions) == 2
    assert seen == [(3, False, 2)]
    seen.clear()
    assert len(formula_detector(doc, FormulaModel("mfd.onnx", workers=3, threads=1)).sessions) == 3
    assert seen == [(1, False, 3)]
    seen.clear()
    monkeypatch.setattr(pdf_vision.os, "cpu_count", lambda: 2)
    assert len(formula_detector(doc, FormulaModel("mfd.onnx")).sessions) == 1
    assert seen == [(2, False, 1)]


def test_formula_detector_uses_the_copies_that_load(monkeypatch):
    """后面几份模型加载不了时用已经加载好的，第一份都加载不了才只用规则"""
    doc = _doc()
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [FormulaSession(doc[0].rect, [])])
    assert len(formula_detector(doc, FormulaModel("mfd.onnx", workers=3)).sessions) == 1
    monkeypatch.setattr(pdf_vision, "_usable_sessions", lambda path, *args, **kwargs: [])
    assert formula_detector(doc, FormulaModel("mfd.onnx", workers=3)) is None


def test_several_pages_infer_at_once_each_with_its_own_copy():
    """两份模型时页按顺序轮流交给两份，同一份不会同时推两页；给出的框仍按页的顺序"""
    import threading
    import time

    doc = pymupdf.open()
    for _ in range(5):
        doc.new_page(width=400, height=500)
    box = pymupdf.Rect(100, 100, 140, 112)
    done = []

    class Copy(FormulaSession):
        def __init__(self, name, pause):
            super().__init__(doc[0].rect, [])
            self.name = name
            self.pause = pause
            self.lock = threading.Lock()

        def run(self, outputs, feeds):
            assert self.lock.acquire(blocking=False), "同一份模型同时推了两页"
            try:
                time.sleep(self.pause)
                shift = len(done) * 100
                done.append(self.name)
                self.boxes = [(box + (0, shift, 0, shift), 0.9, 0)]
                return super().run(outputs, feeds)
            finally:
                self.lock.release()

    # a 比 b 慢：b 先推完时，下一页要等 a 空出来才能交给 a
    copies = [Copy("a", 0.05), Copy("b", 0.01)]
    detector = FormulaDetector(doc, copies)
    pages = list(detector.pages())
    assert len(pages) == 5 and sorted(done) == ["a", "a", "a", "b", "b"]
    assert [len(c.inputs) for c in copies] == [3, 2]
    assert sorted(detector.isolated) == [0, 1, 2, 3, 4]


def test_formula_sessions_do_not_spin_while_waiting(monkeypatch, tmp_path):
    """不空转的推理线程等活时让出核；同一个模型的几份各是一个会话"""
    import onnxruntime

    made = []
    broken = set()

    class Session:
        def __init__(self, path, options, providers):
            if len(made) in broken:
                raise RuntimeError("out of memory")
            try:
                made.append(options.get_session_config_entry("session.intra_op.allow_spinning"))
            except RuntimeError:  # 没设过，onnxruntime 默认空转
                made.append(None)

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    pdf_vision.clear_sessions()
    try:
        model = str(tmp_path / "mfd.onnx")
        first, second = pdf_vision.load_sessions(model, 3, spin=False, count=2)
        assert first is not second and made == ["0", "0"]
        # 加载过的留给后面的任务
        assert pdf_vision.load_sessions(model, 3, spin=False, count=2) == [first, second]
        assert pdf_vision.load_sessions(model, 3, spin=False) == [first]
        # 第三份加载不了，用已经有的两份
        broken.add(2)
        assert pdf_vision.load_sessions(model, 3, spin=False, count=3) == [first, second]
        # 线程设置改了，换成新的
        broken.clear()
        assert first not in pdf_vision.load_sessions(model, 2, spin=False)
        pdf_vision.load_session("layout.onnx")
        assert made == ["0", "0", "0", None]
        # 重新生成模型以后加载新文件
        pdf_vision.clear_sessions()
        assert pdf_vision.load_session("layout.onnx") is not None and len(made) == 5
    finally:
        pdf_vision.clear_sessions()


def test_formula_pages_also_keep_the_display_formula_boxes():
    """逐页取行内公式框时，行间公式框（isolated）按页记下，定哪些块留原文时用"""
    doc = _doc()
    display = pymupdf.Rect(80, 200, 320, 240)
    session = FormulaSession(doc[0].rect, [(pymupdf.Rect(100, 100, 140, 112), 0.9, 0), (display, 0.8, 1)])
    detector = FormulaDetector(doc, [session])
    [inline] = list(detector.pages())
    assert len(inline) == 1
    [found] = detector.isolated[0]
    assert all(abs(a - b) < 1 for a, b in zip(found, display))


def test_layout_inference_stops_when_the_job_is_paused():
    """版面模型一页推一秒多，整本书在定角色时逐页推。任务被暂停时在下一页停下"""
    import asyncio
    import time

    from app import background

    doc = pymupdf.open()
    for _ in range(100):
        doc.new_page(width=400, height=500)

    class Slow(FakeSession):
        def run(self, outputs, feeds):
            time.sleep(0.01)
            return super().run(outputs, feeds)

    session = Slow(doc[0].rect, [])
    layout = ModelLayout(doc, session)

    async def main():
        task = asyncio.ensure_future(background.run(lambda: [layout.page_regions(p) for p in range(100)]))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert 0 < len(session.inputs) < 50


def test_saved_boxes_come_back_only_for_the_same_pdf_model_and_size(tmp_path):
    """框按 PDF、模型文件、输入尺寸存。换了哪一样都不读回；两个模型分开存"""
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
    # 存不下（目录不在了）只是下次要重新推
    ModelBoxes(tmp_path / "gone" / "boxes.json", src).keep("formula", str(model), 1024, pages)


def test_formula_boxes_are_kept_only_after_the_whole_book():
    """整本推完才存；暂停时中途停下，不存半本"""
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=400, height=500)
    session = FormulaSession(doc[0].rect, [(pymupdf.Rect(100, 100, 140, 112), 0.9, 0)])
    kept = []
    detector = FormulaDetector(doc, [session])
    detector.keep = kept.append
    pages = detector.pages()
    next(pages)
    pages.close()
    assert kept == []
    # 和提取时一样：逐页取，取完最后一页就关掉
    detector = FormulaDetector(doc, [session])
    detector.keep = kept.append
    pages = detector.pages()
    assert all(len(next(pages)) == 1 for _ in range(3))
    pages.close()
    [saved] = kept
    assert len(saved) == 3 and all(len(inline) == 1 and isolated == [] for inline, isolated in saved)
