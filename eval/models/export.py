"""把 Ultralytics .pt 导成和 Pix2Text MFD 一样的 ONNX（动态尺寸、不带 NMS）。用法: export.py <pt>"""
import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1])
print("names", m.names, "task", m.task)
print("args imgsz", m.ckpt.get("train_args", {}).get("imgsz") if hasattr(m, "ckpt") and m.ckpt else None,
      "model", m.ckpt.get("train_args", {}).get("model") if hasattr(m, "ckpt") and m.ckpt else None)
print(m.export(format="onnx", dynamic=True, simplify=True, nms=False, batch=1, imgsz=1280))
