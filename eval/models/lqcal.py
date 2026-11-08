"""int8 quantization of the layout model (DocLayout-YOLO, the one2one detection head of YOLOv10), with the same
settings as the formula model in the application.
Usage: lqcal.py <code dir> <fp32 onnx> <output> [pages] [seed]; the environment variable EXCLUDE lists PDF file
names left out of calibration"""

import os
import re
import sys
import tempfile
import time
from pathlib import Path

code, src, out = sys.argv[1:4]
count = int(sys.argv[4]) if len(sys.argv) > 4 else 64
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
sys.path.insert(0, code)
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from app import layout_model, quantize

skip = set(filter(None, os.environ.get("EXCLUDE", "").split(",")))
# Calibration pages are sampled from the PDFs under CALIB_PDFS (by default the translated jobs), as when the
# application generates an int8 model
calib = Path(os.environ.get("CALIB_PDFS", Path(code, "data", "jobs")))
pdfs = sorted(
    p
    for p in [*calib.glob("*.pdf"), *calib.glob("*/*.pdf")]
    if "bilingual" not in p.name and "translated" not in p.name and p.name not in skip
)
t0 = time.perf_counter()
images = quantize.calibration_images(
    quantize.pick_pages(pdfs, count, seed), layout_model.sample_page, quantize.SPECS["layout"].size
)
print(len(pdfs), "pdfs", len(images), "pages", round(time.perf_counter() - t0, 1), "s", flush=True)


def head_nodes(names: list[str]) -> list[str]:
    blocks = [int(m.group(1)) for n in names if (m := re.match(r"/model\.(\d+)/", n))]
    head = f"/model.{max(blocks)}/"
    branch = re.compile(r"/(?:one2one_)?cv[23]\.\d+/")
    last = re.compile(r"/(?:one2one_)?cv[23]\.\d+/(?:one2one_)?cv[23]\.\d+\.2/")
    return [n for n in names if n.startswith(head) and (not branch.search(n) or last.search(n))]


# onnxruntime has no type information, so its base class is Any to mypy
class Pages(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self) -> None:
        self.left = iter(images)

    def get_next(self) -> dict[str, np.ndarray] | None:
        image = next(self.left, None)
        return None if image is None else {"images": image.astype(np.float32) / 255.0}


with tempfile.TemporaryDirectory(dir=os.environ.get("WORK", tempfile.gettempdir())) as tmp:
    pre = str(Path(tmp) / "pre.onnx")
    quant_pre_process(src, pre, skip_symbolic_shape=True)
    names = [n.name for n in onnx.load(pre, load_external_data=False).graph.node]
    excluded = head_nodes(names)
    print(
        "excluded",
        len(excluded),
        "of",
        len(names),
        "conv kept fp32:",
        [n for n in excluded if n.endswith("Conv")],
        flush=True,
    )
    t1 = time.perf_counter()
    quantize_static(
        pre,
        out,
        Pages(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        nodes_to_exclude=excluded,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "CalibMovingAverage": True},
    )
print("quantized in", round(time.perf_counter() - t1, 1), "s", flush=True)
