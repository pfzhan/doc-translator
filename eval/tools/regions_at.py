"""Run the layout model at a given input side and cache the boxes of each page.
Usage: regions_at.py <code dir> <model> <side> <threads> <pdf> <output json>"""

import json
import sys
from pathlib import Path

code, model, size, threads, src, out = sys.argv[1:7]
sys.path.insert(0, code)
import pymupdf

from app.formats import pdf_vision
from app.formats.pdf_vision import ModelLayout, load_session

pdf_vision._INPUT = int(size)
doc = pymupdf.open(src)
layout = ModelLayout(doc, load_session(model, int(threads)))


def corners(rect: pymupdf.Rect) -> list[float]:
    return [round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)]


pages = []
for pno in range(doc.page_count):
    pages.append([[r.label, corners(r.rect), round(r.score, 3)] for r in layout.regions(pno)])
Path(out).write_text(json.dumps(pages))
