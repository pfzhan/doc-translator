"""Run the formula detection model through the project code (formula_detector + pages(), including several pages
inferred at once) and cache the boxes of each page: one list per page, each box [class, [x0, y0, x1, y1], null] in
PDF page coordinates.
Usage: mfd_project.py <code dir> <model> <pdf> <output json> [side=1024]"""

import json
import sys
from pathlib import Path

code, model, src, out = sys.argv[1:5]
size = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
sys.path.insert(0, code)
import pymupdf

from app.formats.pdf_vision import FormulaModel, formula_detector

doc = pymupdf.open(src)
detector = formula_detector(doc, FormulaModel(model, size))
assert detector is not None, "the formula detection model could not be loaded"


def corners(rect: pymupdf.Rect) -> list[float]:
    return [round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)]


pages = []
for pno, inline in enumerate(detector.pages()):
    found = [["embedding", corners(r), None] for r in inline]
    found += [["isolated", corners(r), None] for r in detector.isolated[pno]]
    pages.append(found)
Path(out).write_text(json.dumps(pages))
