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
    """按页面坐标给框，换算成模型输入里的像素坐标（长边缩放到 1024、居中补边）。"""

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
        zoom = 1024 / max(self.page_rect.width, self.page_rect.height)
        left = (1024 - round(self.page_rect.width * zoom)) // 2
        top = (1024 - round(self.page_rect.height * zoom)) // 2
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
    assert session.inputs == [(1, 3, 1024, 1024)]


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
        size = feeds["images"].shape[-1]
        self.inputs.append(feeds["images"].shape)
        zoom = size / max(self.page_rect.width, self.page_rect.height)
        left = (size - round(self.page_rect.width * zoom)) // 2
        top = (size - round(self.page_rect.height * zoom)) // 2
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
    detector = FormulaDetector(doc, session)
    assert sorted(region.label for region in detector.regions(0)) == ["embedding", "isolated"]
    [found] = detector.inline(0)
    assert all(abs(a - b) < 1 for a, b in zip(found, inline))


def test_formula_detector_needs_an_inline_formula_class(monkeypatch, tmp_path):
    """没开、文件不存在都只用规则。路径填成版面模型（类别里没有行内公式）也不用"""
    assert formula_detector(_doc()) is None
    assert formula_detector(_doc(), FormulaModel(str(tmp_path / "missing.onnx"))) is None
    doc = _doc()
    monkeypatch.setattr(pdf_vision, "_usable_session", lambda path, threads=0: FakeSession(doc[0].rect, []))
    assert formula_detector(doc, FormulaModel("layout.onnx")) is None
    monkeypatch.setattr(pdf_vision, "_usable_session", lambda path, threads=0: FormulaSession(doc[0].rect, []))
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
    pages = list(FormulaDetector(doc, session, 768).pages())
    assert [round(found.y0) for [found] in pages] == [100, 200, 300]
    assert session.inputs == [(1, 3, 768, 768)] * 3


def test_formula_inference_leaves_two_cores_for_reading_pages(monkeypatch):
    """模型在后台推理时主线程还在读页面，推理线程比核数少两个，至少一个"""
    seen = []
    doc = _doc()

    def usable(path, threads=0):
        seen.append(threads)
        return FormulaSession(doc[0].rect, [])

    monkeypatch.setattr(pdf_vision, "_usable_session", usable)
    monkeypatch.setattr(pdf_vision.os, "cpu_count", lambda: 6)
    formula_detector(doc, FormulaModel("mfd.onnx"))
    monkeypatch.setattr(pdf_vision.os, "cpu_count", lambda: 2)
    formula_detector(doc, FormulaModel("mfd.onnx"))
    assert seen == [4, 1]


def test_formula_pages_also_keep_the_display_formula_boxes():
    """逐页取行内公式框时，行间公式框（isolated）按页记下，定哪些块留原文时用"""
    doc = _doc()
    display = pymupdf.Rect(80, 200, 320, 240)
    session = FormulaSession(doc[0].rect, [(pymupdf.Rect(100, 100, 140, 112), 0.9, 0), (display, 0.8, 1)])
    detector = FormulaDetector(doc, session)
    [inline] = list(detector.pages())
    assert len(inline) == 1
    [found] = detector.isolated[0]
    assert all(abs(a - b) < 1 for a, b in zip(found, display))
