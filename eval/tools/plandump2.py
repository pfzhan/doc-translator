"""Segmentation plan with the page number and box of the first and last block of each unit.
Usage: plandump2.py <code dir> <pdf> <output json>"""

import json
import sys
from pathlib import Path

code, src, out = sys.argv[1:4]
sys.path.insert(0, code)
import pymupdf

from app.formats import pdf
from app.runner import Runner


class R(Runner):
    """Planning only reports stages and sets the title: no translator or cache is set up."""

    def __init__(self) -> None:
        pass

    def report_stage(self, name: str, done: int = 0, total: int = 0, unit: str = "") -> None:
        pass

    def set_title(self, title: str) -> None:
        pass


def corners(rect: pymupdf.Rect) -> list[float]:
    return [round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)]


plan = pdf._plan_pdf(Path(src), R(), "")
rows = []
for u, t, s in zip(plan.units, plan.texts, plan.skips):
    first, last = u[0], u[-1]
    rows.append(
        [
            first.page,
            s,
            t,
            [],
            [
                last.page,
                corners(first.rect),
                corners(last.rect),
                corners(first.line_rects[0]) if first.line_rects else None,
            ],
        ]
    )
Path(out).write_text(json.dumps(rows, ensure_ascii=False))
