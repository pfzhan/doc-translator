"""按指定输入边长跑版面模型，缓存每页的框。用法: regions_at.py <代码目录> <模型> <边长> <线程数> <pdf> <输出 json>"""
import sys, json
code, model, size, threads, src, out = sys.argv[1:7]
sys.path.insert(0, code)
import pymupdf
from app.formats import pdf_vision
from app.formats.pdf_vision import ModelLayout, load_session
pdf_vision._INPUT = int(size)
doc = pymupdf.open(src)
layout = ModelLayout(doc, load_session(model, int(threads)))
pages = []
for pno in range(doc.page_count):
    pages.append([[r.label, [round(v, 1) for v in r.rect], round(r.score, 3)] for r in layout.regions(pno)])
json.dump(pages, open(out, "w"))
