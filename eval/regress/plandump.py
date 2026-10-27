import sys, json
from pathlib import Path
code, src, out = sys.argv[1:4]
sys.path.insert(0, code)
from app.formats import pdf
class R:
    def report_stage(self, *a, **k): pass
    def set_title(self, t): pass
plan = pdf._plan_pdf(Path(src), R(), "")
rows = []
for u, t, fs, s in zip(plan.units, plan.texts, plan.formulas, plan.skips):
    rows.append([u[0].page, s, t, [[f.get("text", ""), [round(v, 1) for v in f["bbox"]] if f.get("bbox") is not None else None] for f in fs]])
json.dump(rows, open(out, "w"), ensure_ascii=False)
