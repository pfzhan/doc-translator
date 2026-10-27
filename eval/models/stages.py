"""翻译前各阶段耗时，外加主线程的 cProfile。用法: stages.py <代码目录> <onnx|-> <pdf> <prof 输出>"""
import sys, time, cProfile, pstats, tempfile
from pathlib import Path
code, onnx, src, prof = sys.argv[1:5]
sys.path.insert(0, code)
from app.formats import pdf
from app.formats.pdf_vision import FormulaModel
from app.runner import Cache, Runner

class T:
    source_lang = "auto"
    target_lang = "zh-CN"


runner = Runner(T(), cache=Cache(Path(tempfile.mkdtemp()) / "c.sqlite3"))
marks = []
orig = runner.report_stage
def report(name, done=0, total=0, unit=""):
    if not marks or marks[-1][0] != name:
        marks.append((name, time.perf_counter()))
    orig(name, done, total, unit)
runner.report_stage = report

t0 = time.perf_counter()
profiler = cProfile.Profile()
profiler.enable()
plan = pdf._plan_pdf(Path(src), runner, "", None if onnx == "-" else FormulaModel(onnx))
t1 = time.perf_counter()
unique = list(dict.fromkeys(t for t in plan.texts if t.strip()))
runner._detect(unique)
t2 = time.perf_counter()
profiler.disable()
profiler.dump_stats(prof)
prev_name, prev = "开始（读全文找词）", t0
for name, t in marks:
    print(f"{prev_name:<12} {t - prev:7.1f} s")
    prev_name, prev = name, t
print(f"{prev_name:<12} {t1 - prev:7.1f} s")
print(f"{'语言检测':<12} {t2 - t1:7.1f} s   ({len(unique)} 段)")
print(f"{'合计':<12} {t2 - t0:7.1f} s")
