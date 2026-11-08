import json
import sys
from pathlib import Path

code, src, out = sys.argv[1:4]
sys.path.insert(0, code)
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


plan = pdf._plan_pdf(Path(src), R(), "")
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
