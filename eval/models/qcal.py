"""产品代码的抽页 + 可换的校准方法。用法: qcal.py <代码目录> <fp32 onnx> <输出> <minmax|percentile|entropy|avg> [页数] [seed]"""
import sys, time
from pathlib import Path
code, src, out, method = sys.argv[1:5]
count = int(sys.argv[5]) if len(sys.argv) > 5 else 64
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
sys.path.insert(0, code)
import numpy as np, onnx, tempfile
from app import quantize, layout_model
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
import os
skip = set(filter(None, os.environ.get("EXCLUDE", "").split(",")))
# 校准页从 CALIB_PDFS 目录下的 PDF 抽（默认是翻译过的任务），和产品里生成 int8 模型时一样
calib = Path(os.environ.get("CALIB_PDFS", Path(code, "data", "jobs")))
pdfs = sorted(p for p in [*calib.glob("*.pdf"), *calib.glob("*/*.pdf")]
              if "bilingual" not in p.name and "translated" not in p.name and p.name not in skip)
print(len(pdfs), "pdfs", flush=True)
t0 = time.perf_counter()
images = quantize.calibration_images(quantize.pick_pages(pdfs, count, seed), layout_model.sample_page)
print(len(images), "pages", round(time.perf_counter() - t0, 1), "s", flush=True)

class Pages(CalibrationDataReader):
    def __init__(self):
        self.left = iter(images)
    def get_next(self):
        image = next(self.left, None)
        return None if image is None else {"images": image.astype(np.float32) / 255.0}

from onnxruntime.quantization.calibrate import HistogramCalibrater
_collect = HistogramCalibrater.collect_data


class One(CalibrationDataReader):
    def __init__(self, item):
        self.item = item
    def get_next(self):
        item, self.item = self.item, None
        return item


def one_page_at_a_time(self, reader):
    # 原实现把所有页的中间结果都留在内存里再算直方图；直方图能逐页合并
    while (item := reader.get_next()):
        _collect(self, One(item))


HistogramCalibrater.collect_data = one_page_at_a_time

calib = {"minmax": CalibrationMethod.MinMax, "avg": CalibrationMethod.MinMax,
         "percentile": CalibrationMethod.Percentile, "entropy": CalibrationMethod.Entropy}[method]
extra = {"ActivationSymmetric": False}
if method == "avg":
    extra["CalibMovingAverage"] = True
if method == "percentile":
    extra["CalibPercentile"] = 99.999
with tempfile.TemporaryDirectory(dir=os.environ.get("WORK", tempfile.gettempdir())) as tmp:
    pre = str(Path(tmp) / "pre.onnx")
    quant_pre_process(src, pre, skip_symbolic_shape=True)
    names = [n.name for n in onnx.load(pre, load_external_data=False).graph.node]
    t1 = time.perf_counter()
    quantize_static(pre, out, Pages(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                    nodes_to_exclude=quantize.decode_nodes(names), calibrate_method=calib, extra_options=extra)
print(method, count, seed, "quantized in", round(time.perf_counter() - t1, 1), "s", flush=True)
