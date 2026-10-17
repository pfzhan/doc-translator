import numpy as np
import pymupdf

from app.formats.pdf_layout import PageGeometry
from app.formats.pdf_roles import HeuristicLayout, LayoutItem, Role
from app.formats.pdf_vision import ModelLayout, layout_detector

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
