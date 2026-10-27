"""分段计划，带每个单元首块、末块的页码和框。用法: plandump2.py <代码目录> <pdf> <输出 json>"""
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
for u, t, s in zip(plan.units, plan.texts, plan.skips):
    first, last = u[0], u[-1]
    rows.append([first.page, s, t, [], [last.page, [round(v, 1) for v in first.rect], [round(v, 1) for v in last.rect],
                 [round(v, 1) for v in first.line_rects[0]] if first.line_rects else None]])
json.dump(rows, open(out, "w"), ensure_ascii=False)
