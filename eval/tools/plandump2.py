"""Segmentation plan with the page number and box of the first and last block of each unit.
Usage: plandump2.py <code dir> <pdf> <output json>"""

import json
import sys
from pathlib import Path

code, src, out = sys.argv[1:4]
sys.path.insert(0, code)
from app.formats import pdf


class R:
    def report_stage(self, *a, **k):
        pass

    def set_title(self, t):
        pass


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
                [round(v, 1) for v in first.rect],
                [round(v, 1) for v in last.rect],
                [round(v, 1) for v in first.line_rects[0]] if first.line_rects else None,
            ],
        ]
    )
Path(out).write_text(json.dumps(rows, ensure_ascii=False))
