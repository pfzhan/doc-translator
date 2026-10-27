"""开版面模型（读缓存的框）出老语料的计划。用法: plandump_model.py <代码目录> <pdf> <框 json> <输出> [nojoin]"""
import sys, json
from pathlib import Path
code, src, cached, out = sys.argv[1:5]
sys.path.insert(0, code)
import pymupdf
from app.formats import pdf, pdf_vision
pages = json.load(open(cached))
class Cached(pdf_vision.ModelLayout):
    def __init__(self, doc):
        self.doc = doc; self._pages = {}; self._boxes = None; self._fresh = False
    def regions(self, pno):
        return [pdf_vision.Region(l, pymupdf.Rect(r), s) for l, r, s in pages[pno]]
pdf.layout_detector = lambda doc, *args: Cached(doc)
if len(sys.argv) > 5 and sys.argv[5] == "nojoin":
    pdf._region_joins = lambda units, *a: units
class R:
    def report_stage(self, *a, **k): pass
    def set_title(self, t): pass
plan = pdf._plan_pdf(Path(src), R(), "cached")
rows = []
for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips):
    rows.append([u[0].page, s, t, [[f.get("text", ""), [round(v, 1) for v in f["bbox"]] if f.get("bbox") is not None else None] for f in fs]])
json.dump(rows, open(out, "w"), ensure_ascii=False)
