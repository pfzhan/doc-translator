"""Produce the plan for the regression corpus with the layout model enabled (reading cached boxes).
Usage: plandump_model.py <code dir> <pdf> <boxes json> <output> [nojoin]"""

import json
import sys
from pathlib import Path

code, src, cached, out = sys.argv[1:5]
sys.path.insert(0, code)
import pymupdf

from app.formats import pdf, pdf_units, pdf_vision

pages = json.loads(Path(cached).read_text())


class Cached(pdf_vision.ModelLayout):
    def __init__(self, doc):
        self.doc = doc
        self._pages = {}
        self._boxes = None
        self._fresh = False

    def regions(self, pno):
        return [pdf_vision.Region(label, pymupdf.Rect(r), score) for label, r, score in pages[pno]]


pdf_units.layout_detector = lambda doc, *args: Cached(doc)
if len(sys.argv) > 5 and sys.argv[5] == "nojoin":
    pdf_units._region_joins = lambda units, *a: units


class R:
    def report_stage(self, *a, **k):
        pass

    def set_title(self, t):
        pass


plan = pdf._plan_pdf(Path(src), R(), "cached")
rows = []
for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips):
    rows.append(
        [
            u[0].page,
            s,
            t,
            [[f.get("text", ""), [round(v, 1) for v in f["bbox"]] if f.get("bbox") is not None else None] for f in fs],
        ]
    )
Path(out).write_text(json.dumps(rows, ensure_ascii=False))
