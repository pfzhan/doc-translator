"""每页轮流跑各个模型/边长，抵消机器负载的漂移。用法: bench2.py <代码目录> <onnx@边长,...> <pdf>..."""
import sys, time
code, variants = sys.argv[1:3]
sys.path.insert(0, code)
import pymupdf
from app.formats.pdf_vision import load_session, _page_batch
runs = []
for v in variants.split(","):
    path, size = v.rsplit("@", 1)
    s = load_session(path)
    runs.append([v.split("/")[-1], s, s.get_inputs()[0].name, int(size), 0.0])
pages = []
for p in sys.argv[3:]:
    d = pymupdf.open(p)
    pages += [d[i] for i in range(min(4, d.page_count))]
for r in runs:
    b, _ = _page_batch(pages[0], r[3]); r[1].run(None, {r[2]: b}); r[1].run(None, {r[2]: b})
for pg in pages:
    for r in runs:
        b, _ = _page_batch(pg, r[3])
        t0 = time.perf_counter(); r[1].run(None, {r[2]: b}); r[4] += time.perf_counter() - t0
for r in runs:
    print(r[0], round(r[4] / len(pages), 2), "s/page", flush=True)
