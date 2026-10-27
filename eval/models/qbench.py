"""几个公式模型逐页轮流推理（每份 3 线程、不空转，和翻译时一份的设置一样）。用法: qbench.py <代码目录> <onnx,...> <pdf>..."""
import sys, time
code, models = sys.argv[1:3]
sys.path.insert(0, code)
import pymupdf
from app.formats.pdf_vision import load_session, _page_batch
runs = []
for m in models.split(","):
    s = load_session(m, 3, False)
    print(m.split("/")[-1], "names" in s.get_modelmeta().custom_metadata_map, flush=True)
    runs.append([m.split("/")[-1], s, s.get_inputs()[0].name, 0.0])
pages = []
for p in sys.argv[3:]:
    d = pymupdf.open(p)
    pages += [d[i] for i in range(min(5, d.page_count))]
b, _ = _page_batch(pages[0], 1024)
for r in runs:
    r[1].run(None, {r[2]: b})
for pg in pages:
    b, _ = _page_batch(pg, 1024)
    for r in runs:
        t0 = time.perf_counter(); r[1].run(None, {r[2]: b}); r[3] += time.perf_counter() - t0
for r in runs:
    print(r[0], round(r[3] / len(pages), 3), "s/page", flush=True)
