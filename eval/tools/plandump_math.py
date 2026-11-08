"""Segmentation plan with the formula text of each unit. Usage: plandump_math.py <code dir> <pdf> <output json>
[boxes json]. With a boxes json the layout model is enabled (reading cached boxes). Rows: [page, skip, text sent
for translation, [formula text...]]"""

import json
import sys
from pathlib import Path

code, src, out = sys.argv[1:4]
sys.path.insert(0, code)
from app.formats import pdf, pdf_units
from app.runner import Runner

if len(sys.argv) > 4:
    import pymupdf

    from app.formats import pdf_vision
    from app.formats.pdf_roles import LayoutDetector
    from app.formats.pdf_vision import ModelBoxes

    boxes = json.loads(Path(sys.argv[4]).read_text())

    class Cached(pdf_vision.ModelLayout):
        def __init__(self, doc: pymupdf.Document) -> None:
            self.doc = doc
            self._pages = {}
            self._boxes = None
            self._fresh = False

        def regions(self, pno: int) -> list[pdf_vision.Region]:
            return [pdf_vision.Region(label, pymupdf.Rect(r), score) for label, r, score in boxes[pno]]

    def cached_layout(doc: pymupdf.Document, model_path: str = "", saved: ModelBoxes | None = None) -> LayoutDetector:
        return Cached(doc)

    # pdf_units calls the names it imported, so they are replaced there. setattr: pdf_units does not re-export them, and
    # mypy rejects assigning them as attributes
    setattr(pdf_units, "layout_detector", cached_layout)  # noqa: B010


class R(Runner):
    """Planning only reports stages and sets the title: no translator or cache is set up."""

    def __init__(self) -> None:
        pass

    def report_stage(self, name: str, done: int = 0, total: int = 0, unit: str = "") -> None:
        pass

    def set_title(self, title: str) -> None:
        pass


plan = pdf._plan_pdf(Path(src), R(), "")
rows = [
    [u[0].page, s, t, [f.get("text", "") for f in fs]]
    for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips)
]
Path(out).write_text(json.dumps(rows, ensure_ascii=False))
