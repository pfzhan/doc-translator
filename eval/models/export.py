"""Export an Ultralytics .pt to ONNX in the same form as Pix2Text MFD (dynamic shape, no NMS).
Usage: export.py <pt>"""

import sys

from ultralytics import YOLO

m = YOLO(sys.argv[1])
print("names", m.names, "task", m.task)
print(
    "args imgsz",
    m.ckpt.get("train_args", {}).get("imgsz") if hasattr(m, "ckpt") and m.ckpt else None,
    "model",
    m.ckpt.get("train_args", {}).get("model") if hasattr(m, "ckpt") and m.ckpt else None,
)
print(m.export(format="onnx", dynamic=True, simplify=True, nms=False, batch=1, imgsz=1280))
