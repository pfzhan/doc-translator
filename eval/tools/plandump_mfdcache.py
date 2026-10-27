"""用缓存的公式检测框（mfd/<id>.<标签>.json，tools/mfd_project.py 生成）出计划，不跑模型。
用法: plandump_mfdcache.py <代码目录> <pdf> <输出 json> <公式框 json> <seg|math> [版面框 json]
seg 出 plandump2.py 的格式（给 score.py），math 出带公式文字的格式（给 mathscore.py）。"""
import sys, json
from pathlib import Path
code, src, out, boxes_path, kind = sys.argv[1:6]
sys.path.insert(0, code)
import pymupdf
from app.formats import pdf
boxes = json.load(open(boxes_path))


class Cached:
    def __init__(self):
        self.isolated = {}

    def pages(self):
        for pno, found in enumerate(boxes):
            self.isolated[pno] = [pymupdf.Rect(r) for label, r, _s in found if label == "isolated"]
            yield [pymupdf.Rect(r) for label, r, _s in found if label == "embedding"]


pdf.formula_detector = lambda doc, *args: Cached()
if len(sys.argv) > 6:
    from app.formats import pdf_vision
    regions = json.load(open(sys.argv[6]))

    class Layout(pdf_vision.ModelLayout):
        def __init__(self, doc):
            self.doc = doc; self._pages = {}; self._boxes = None; self._fresh = False

        def regions(self, pno):
            return [pdf_vision.Region(l, pymupdf.Rect(r), s) for l, r, s in regions[pno]]
    pdf.layout_detector = lambda doc, *args: Layout(doc)
class R:
    def report_stage(self, *a, **k): pass
    def set_title(self, t): pass
plan = pdf._plan_pdf(Path(src), R(), "", "cached")
rows = []
for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips):
    if kind == "math":
        rows.append([u[0].page, s, t, [f.get("text", "") for f in fs]])
    else:
        first, last = u[0], u[-1]
        rows.append([first.page, s, t, [], [last.page, [round(v, 1) for v in first.rect], [round(v, 1) for v in last.rect],
                     [round(v, 1) for v in first.line_rects[0]] if first.line_rects else None]])
json.dump(rows, open(out, "w"), ensure_ascii=False)
