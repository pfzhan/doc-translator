"""用项目代码（formula_detector + pages()，含几页同时推理）跑公式检测模型，缓存每页的框：
每页一个列表，每个框是 [类别, [x0, y0, x1, y1], null]，坐标是 PDF 页面坐标。
用法: mfd_project.py <代码目录> <模型> <pdf> <输出 json> [边长=1024]"""
import json, sys
code, model, src, out = sys.argv[1:5]
size = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
sys.path.insert(0, code)
import pymupdf
from app.formats.pdf_vision import FormulaModel, formula_detector

doc = pymupdf.open(src)
detector = formula_detector(doc, FormulaModel(model, size))
pages = []
for pno, inline in enumerate(detector.pages()):
    found = [["embedding", [round(v, 1) for v in r], None] for r in inline]
    found += [["isolated", [round(v, 1) for v in r], None] for r in detector.isolated[pno]]
    pages.append(found)
json.dump(pages, open(out, "w"))
