"""Segmentation plan with the formula text of each unit. Usage: plandump_math.py <code dir> <pdf> <output json>
[boxes json]. With a boxes json the layout model is enabled (reading cached boxes). Rows: [page, skip, text sent
for translation, [formula text...]]"""

import json
import sys
from pathlib import Path

code, src, out = sys.argv[1:4]
sys.path.insert(0, code)
from app.formats import pdf, pdf_units

if len(sys.argv) > 4:
    import pymupdf

    from app.formats import pdf_vision

    boxes = json.loads(Path(sys.argv[4]).read_text())

    class Cached(pdf_vision.ModelLayout):
        def __init__(self, doc):
            self.doc = doc
            self._pages = {}
            self._boxes = None
            self._fresh = False

        def regions(self, pno):
            return [pdf_vision.Region(label, pymupdf.Rect(r), score) for label, r, score in boxes[pno]]

    pdf_units.layout_detector = lambda doc, *args: Cached(doc)


class R:
    def report_stage(self, *a, **k):
        pass

    def set_title(self, t):
        pass


plan = pdf._plan_pdf(Path(src), R(), "")
rows = [
    [u[0].page, s, t, [f.get("text", "") for f in fs]]
    for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips)
]
Path(out).write_text(json.dumps(rows, ensure_ascii=False))
