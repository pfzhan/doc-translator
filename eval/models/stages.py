"""Time of each stage before translation, plus a cProfile of the main thread.
Usage: stages.py <code dir> <onnx|-> <pdf> <prof output>"""

import cProfile
import sys
import tempfile
import time
from pathlib import Path

code, onnx, src, prof = sys.argv[1:5]
sys.path.insert(0, code)
from app.formats import pdf
from app.formats.pdf_vision import FormulaModel
from app.runner import Cache, Runner
from app.translators import MockTranslator

# When each stage started: (stage name, time.perf_counter())
marks: list[tuple[str, float]] = []


class TimedRunner(Runner):
    def report_stage(self, name: str, done: int = 0, total: int = 0, unit: str = "") -> None:
        if not marks or marks[-1][0] != name:
            marks.append((name, time.perf_counter()))
        super().report_stage(name, done, total, unit)


# Nothing is translated: the translator only gives the languages for language detection
runner = TimedRunner(MockTranslator("zh-CN"), cache=Cache(Path(tempfile.mkdtemp()) / "c.sqlite3"))

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
