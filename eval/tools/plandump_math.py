"""分段计划连同每个单元的公式文字。用法: plandump_math.py <代码目录> <pdf> <输出 json> [框 json]
给了框 json 时开版面模型（读缓存的框）。行: [页, 跳过, 送翻文本, [公式文字...]]"""
import sys, json
from pathlib import Path
code, src, out = sys.argv[1:4]
sys.path.insert(0, code)
from app.formats import pdf
if len(sys.argv) > 4:
    import pymupdf
    from app.formats import pdf_vision
    boxes = json.load(open(sys.argv[4]))

    class Cached(pdf_vision.ModelLayout):
        def __init__(self, doc):
            self.doc = doc; self._pages = {}; self._boxes = None; self._fresh = False

        def regions(self, pno):
            return [pdf_vision.Region(l, pymupdf.Rect(r), s) for l, r, s in boxes[pno]]
    pdf.layout_detector = lambda doc, *args: Cached(doc)
class R:
    def report_stage(self, *a, **k): pass
    def set_title(self, t): pass
plan = pdf._plan_pdf(Path(src), R(), "")
rows = [[u[0].page, s, t, [f.get("text", "") for f in fs]] for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips)]
json.dump(rows, open(out, "w"), ensure_ascii=False)
