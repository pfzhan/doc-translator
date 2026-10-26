"""公式检测模型的 int8 版：在本机用翻译过的 PDF 抽页校准，把权重和中间结果换成 8 位整数，推理快一倍左右。

量化要 onnx 包（onnxruntime 的量化工具依赖它），只在生成时用，翻译时不需要。没装时页面上提示安装，
翻译照常用 fp32 模型。

和一般的量化做法有两处不同：
- 检测头的解码（DFL、Sigmoid、拼框和分）和各分支最后一层卷积留 fp32。框坐标 0～1024 和分 0～1 拼在
  一个张量里，共用一个 int8 刻度时分全变成 0，一个公式也框不出来。
- 权重只用 7 位（reduce_range）。没有 VNNI 指令的 x86 CPU 上，onnxruntime 的 int8 卷积把两个 u8×s8
  乘积加进 16 位，权重用满 8 位时会溢出，第一层的误差就有两倍。
在 36 篇 arXiv 论文上，用另外 17 个 PDF 校准时，行内公式识别正确的比例 98.6%（fp32 是 98.9%），分段不变；
6 核 CPU 上每页推理 1.11 s → 0.56 s。
"""
from __future__ import annotations

import asyncio
import importlib.util
import random
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pymupdf

from .formats import pdf_vision

INSTALL_HINT = ".venv/bin/pip install onnx"
# 校准抽的页数。多抽没有明显好处，每页要多推一遍
CALIBRATION_PAGES = 64
# 翻译时用哪个：int8 文件不存在时用 fp32
PRECISIONS = ("int8", "fp32")


class QuantizeError(Exception):
    pass


def available() -> bool:
    """装了 onnx 和 onnxruntime 才能生成。"""
    return importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


def int8_path(fp32: Path) -> Path:
    """int8 文件放在 fp32 文件旁边：pix2text-mfd-1.5.onnx → pix2text-mfd-1.5.int8.onnx。"""
    return fp32.with_name(f"{fp32.stem}.int8{fp32.suffix}")


def pick_pages(pdfs: list[Path], count: int = CALIBRATION_PAGES, seed: int = 0) -> list[tuple[Path, int]]:
    """从各个 PDF 里平均随机抽页，一共最多 count 页。打不开的文件跳过。"""
    rng = random.Random(seed)
    pools: list[list[tuple[Path, int]]] = []
    for path in pdfs:
        try:
            with pymupdf.open(path) as doc:
                pages = list(range(doc.page_count))
        except Exception:  # noqa: BLE001  坏文件、加了密的文件不拿来校准
            continue
        rng.shuffle(pages)
        if pages:
            pools.append([(path, pno) for pno in pages])
    picks: list[tuple[Path, int]] = []
    # 轮流从每个文件取一页，页数少的文件取完为止
    while len(picks) < count and any(pools):
        for pool in pools:
            if pool and len(picks) < count:
                picks.append(pool.pop())
    return sorted(picks, key=lambda pick: (str(pick[0]), pick[1]))


def calibration_images(picks: list[tuple[Path, int]], fallback: Callable[[], pymupdf.Document]) -> list[np.ndarray]:
    """抽到的页按翻译时的样子画成模型输入，存成 8 位省内存（64 页约 160 MB）。没有页可抽时用 fallback 的样例页。
    要用 MuPDF，调用方先拿到 mupdf_turn。"""
    images = []
    for path, pno in picks:
        with pymupdf.open(path) as doc:
            batch, _ = pdf_vision._page_batch(doc[pno], pdf_vision.FORMULA_SIZES[0])
        images.append(np.rint(batch * 255).astype(np.uint8))
    if not images:
        sample = fallback()
        batch, _ = pdf_vision._page_batch(sample[0], pdf_vision.FORMULA_SIZES[0])
        images.append(np.rint(batch * 255).astype(np.uint8))
    return images


def decode_nodes(names: list[str]) -> list[str]:
    """检测头（最后一个 /model.N/）里不量化的节点：不在 cv2、cv3 分支里的解码部分，和两个分支的最后一层卷积。
    names 是图里全部节点的名字，按 Ultralytics 导出的命名。"""
    blocks = [int(m.group(1)) for name in names if (m := re.match(r"/model\.(\d+)/", name))]
    if not blocks:
        return []
    head = f"/model.{max(blocks)}/"
    branch = re.compile(r"/cv[23]\.\d+/")
    last_conv = re.compile(r"/cv[23]\.\d+/cv[23]\.\d+\.2/")
    return [name for name in names
            if name.startswith(head) and (not branch.search(name) or last_conv.search(name))]


class Quantizer:
    """同一时间只生成一份。在线程里量化，进度给页面轮询；先写临时文件，验过能框出公式才换成正式文件。"""

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.state: dict = {"running": False, "done": 0, "total": 0, "error": ""}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, src: Path, dest: Path, images: list[np.ndarray]) -> None:
        if self.running:
            raise QuantizeError("int8 模型正在生成")
        self.state = {"running": True, "done": 0, "total": len(images), "error": ""}
        self.task = asyncio.create_task(self._run(src, dest, images))

    async def _run(self, src: Path, dest: Path, images: list[np.ndarray]) -> None:
        try:
            await asyncio.to_thread(self._quantize, src, dest, images)
        except Exception as e:  # noqa: BLE001  onnx、onnxruntime 抛的异常类型不固定
            self.state["error"] = str(e) or type(e).__name__
        finally:
            self.state["running"] = False

    def _quantize(self, src: Path, dest: Path, images: list[np.ndarray]) -> None:
        import onnx
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process

        state = self.state

        class Pages(CalibrationDataReader):
            def __init__(self):
                self.left = iter(images)

            def get_next(self):
                image = next(self.left, None)
                if image is None:
                    return None
                state["done"] += 1
                return {"images": image.astype(np.float32) / 255.0}

        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            pre = Path(tmp) / "pre.onnx"
            out = Path(tmp) / "int8.onnx"
            quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
            names = [node.name for node in onnx.load(str(pre), load_external_data=False).graph.node]
            quantize_static(
                str(pre), str(out), Pages(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                reduce_range=True, nodes_to_exclude=decode_nodes(names),
                extra_options={"ActivationSymmetric": False},
            )
            _check(src, out, images[:4])
            out.replace(dest)
        # 重新生成时文件名不变，缓存里的旧会话要扔掉，下次翻译加载新文件
        pdf_vision.load_session.cache_clear()


def _check(src: Path, quantized: Path, images: list[np.ndarray]) -> None:
    """在几张校准页上比 fp32 和 int8 框出的公式数，少了一半以上就不用：量化出了问题，比如检测头的命名对不上。"""
    import onnxruntime

    def count(path: Path) -> int:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        total = 0
        for image in images:
            rows = session.run(None, {name: image.astype(np.float32) / 255.0})[0][0]
            total += int((rows[4:].max(0) >= pdf_vision._FORMULA_SCORE).sum())
        return total

    expected, found = count(src), count(quantized)
    if expected and found < expected / 2:
        raise QuantizeError(f"量化后的模型框出的公式太少（{found}，原模型 {expected}），没有保存")
