"""Produce the plan for the regression corpus with the layout model enabled (reading cached boxes).
Usage: plandump_model.py <code dir> <pdf> <boxes json> <output> [nojoin]"""

import json
import sys
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from pathlib import Path

code, src, cached, out = sys.argv[1:5]
sys.path.insert(0, code)
import pymupdf

from app.formats import pdf, pdf_units, pdf_vision
from app.formats.pdf_blocks import TextBlock
from app.formats.pdf_roles import LayoutDetector
from app.formats.pdf_vision import ModelBoxes
from app.runner import Runner

pages = json.loads(Path(cached).read_text())


class Cached(pdf_vision.ModelLayout):
    def __init__(self, doc: pymupdf.Document) -> None:
        self.doc = doc
        self._pages = {}
        self._boxes = None
        self._fresh = False

    def regions(self, pno: int) -> list[pdf_vision.Region]:
        return [pdf_vision.Region(label, pymupdf.Rect(r), score) for label, r, score in pages[pno]]


def cached_layout(doc: pymupdf.Document, model_path: str = "", boxes: ModelBoxes | None = None) -> LayoutDetector:
    return Cached(doc)


def no_joins(
    units: list[list[TextBlock]],
    skip_ids: AbstractSet[int],
    halt_ids: set[int],
    regions_of: Callable[[int], list[pymupdf.Rect]],
) -> list[list[TextBlock]]:
    return units


# pdf_units calls the names it imported, so they are replaced there. setattr: pdf_units does not re-export them, and
# mypy rejects assigning them as attributes
setattr(pdf_units, "layout_detector", cached_layout)  # noqa: B010
if len(sys.argv) > 5 and sys.argv[5] == "nojoin":
    setattr(pdf_units, "_region_joins", no_joins)  # noqa: B010


class R(Runner):
    """Planning only reports stages and sets the title: no translator or cache is set up."""

    def __init__(self) -> None:
        pass

    def report_stage(self, name: str, done: int = 0, total: int = 0, unit: str = "") -> None:
        pass

    def set_title(self, title: str) -> None:
        pass


def _corners(box: pymupdf.Rect) -> tuple[float, float, float, float]:
    """Corners of a formula box. pymupdf.Rect also takes the tuple an older version of the code may hold there."""
    rect = pymupdf.Rect(box)
    return rect.x0, rect.y0, rect.x1, rect.y1


plan = pdf._plan_pdf(Path(src), R(), "cached")
rows = []
for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips):
    rows.append(
        [
            u[0].page,
            s,
            t,
            [
                [f.get("text", ""), [round(v, 1) for v in _corners(f["bbox"])] if f.get("bbox") is not None else None]
                for f in fs
            ],
        ]
    )
Path(out).write_text(json.dumps(rows, ensure_ascii=False))
