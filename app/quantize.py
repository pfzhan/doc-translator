"""int8 versions of the two PDF models (formula detection and layout): calibrated locally on pages sampled from
translated PDFs, with weights and activations converted to 8-bit integers. Formula detection runs about twice as
fast and the layout model over 30% faster.

Quantization needs the onnx package (a dependency of onnxruntime's quantization tools), only while generating,
not while translating. Without it the page suggests installing it, and translation uses the fp32 models as usual.

Three things differ from the usual approach:
- The detection head's decoding (DFL, Sigmoid, concatenating boxes and scores) and the last convolution of each
  branch stay fp32. Box coordinates 0 to 1024 and scores 0 to 1 share one tensor, and with one int8 scale every
  score becomes 0 and no box is found.
- Weights use only 7 bits (reduce_range). On x86 CPUs without VNNI, onnxruntime's int8 convolution adds two
  u8×s8 products into 16 bits, which overflows with full 8-bit weights and doubles the error of the first layer.
- Activation ranges are a moving average of per-page minimums and maximums (CalibMovingAverage) rather than the
  minimum and maximum over all pages. The latter is stretched by individual pages and changes with the sampled
  pages, unstable even with 128 pages.
On 36 arXiv papers, calibrated with 17 translated PDFs (2 of which are among the 36; excluding them gives the same
result):
- Formula detection: inline formulas recognized correctly 98.9%, the same as fp32 (for each of 5 page samples;
  98.5% to 98.7% without the moving average), segmentation unchanged; inference per page on a 6-core CPU
  1.11 s → 0.56 s.
- Layout model: segmentation identical to fp32, 98.6% of boxes match fp32; inference per page 1.05 s → 0.66 s.
The evaluation scripts are described in eval/README.md.
"""

from __future__ import annotations

import asyncio
import importlib.util
import random
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from .formats import pdf_vision

INSTALL_HINT = ".venv/bin/pip install onnx"
# Pages sampled for calibration. More pages bring no clear benefit and each one costs another inference
CALIBRATION_PAGES = 64
# Which version translation uses: fp32 when the int8 file does not exist
PRECISIONS = ("int8", "fp32")


class QuantizeError(Exception):
    pass


@dataclass(frozen=True)
class QuantSpec:
    """How to quantize a model: the size (long side) at which pages are rendered for calibration, and how the
    check after generation counts boxes with enough score in the output."""

    size: int
    boxes: Callable[[np.ndarray], int]


def _formula_boxes(output: np.ndarray) -> int:
    # Each column of the formula detection output is a candidate box: center, width, height and class scores
    return int((output[0][4:].max(0) >= pdf_vision._FORMULA_SCORE).sum())


def _layout_boxes(output: np.ndarray) -> int:
    # Each row of the layout model (YOLOv10) output is a box: corner coordinates, score, class
    return int((output[0][:, 4] >= pdf_vision._MIN_SCORE).sum())


# Keys as in layout_model.MODELS. The formula detection model is calibrated at the default input side
SPECS = {
    "formula": QuantSpec(pdf_vision.FORMULA_SIZES[0], _formula_boxes),
    "layout": QuantSpec(pdf_vision._INPUT, _layout_boxes),
}


def available() -> bool:
    """Generating needs onnx and onnxruntime installed."""
    return importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


def int8_path(fp32: Path) -> Path:
    """The int8 file sits next to the fp32 file: pix2text-mfd-1.5.onnx → pix2text-mfd-1.5.int8.onnx."""
    return fp32.with_name(f"{fp32.stem}.int8{fp32.suffix}")


def pick_pages(pdfs: list[Path], count: int = CALIBRATION_PAGES, seed: int = 0) -> list[tuple[Path, int]]:
    """Sample pages at random, evenly across the PDFs, at most count in total. Files that cannot be opened are
    skipped."""
    rng = random.Random(seed)
    pools: list[list[tuple[Path, int]]] = []
    for path in pdfs:
        try:
            with pymupdf.open(path) as doc:
                pages = list(range(doc.page_count))
        except Exception:  # noqa: BLE001 - broken or encrypted files are not used for calibration
            continue
        rng.shuffle(pages)
        if pages:
            pools.append([(path, pno) for pno in pages])
    picks: list[tuple[Path, int]] = []
    # Take one page from each file in turn, until files with few pages run out
    while len(picks) < count and any(pools):
        for pool in pools:
            if pool and len(picks) < count:
                picks.append(pool.pop())
    return sorted(picks, key=lambda pick: (str(pick[0]), pick[1]))


def calibration_images(
    picks: list[tuple[Path, int]], fallback: Callable[[], pymupdf.Document], size: int
) -> list[np.ndarray]:
    """Render the sampled pages as model input with long side size, as during translation, stored as 8 bits to
    save memory (about 160 MB for 64 pages). Uses the sample pages of fallback when there are no pages to sample.
    This uses MuPDF, so the caller must hold mupdf_turn."""
    images = []
    for path, pno in picks:
        with pymupdf.open(path) as doc:
            batch, _ = pdf_vision._page_batch(doc[pno], size)
        images.append(np.rint(batch * 255).astype(np.uint8))
    if not images:
        sample = fallback()
        batch, _ = pdf_vision._page_batch(sample[0], size)
        images.append(np.rint(batch * 255).astype(np.uint8))
    return images


def decode_nodes(names: list[str]) -> list[str]:
    """Nodes of the detection head (the last /model.N/) left unquantized: the decoding outside the cv2 and cv3
    branches, and the last convolution of both branches. names are all node names in the graph, as named by the
    Ultralytics export; YOLOv10 (the layout model) prefixes branch names with one2one_."""
    blocks = [int(m.group(1)) for name in names if (m := re.match(r"/model\.(\d+)/", name))]
    if not blocks:
        return []
    head = f"/model.{max(blocks)}/"
    branch = re.compile(r"/(?:one2one_)?cv[23]\.\d+/")
    last_conv = re.compile(r"/(?:one2one_)?cv[23]\.\d+/(?:one2one_)?cv[23]\.\d+\.2/")
    return [name for name in names if name.startswith(head) and (not branch.search(name) or last_conv.search(name))]


class Quantizer:
    """Generate one model at a time. Quantizes in a thread with progress polled by the page; writes a temporary
    file and replaces the real one only after the box count check (see _check)."""

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.state: dict = {"running": False, "done": 0, "total": 0, "error": ""}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, src: Path, dest: Path, images: list[np.ndarray], spec: QuantSpec) -> None:
        if self.running:
            raise QuantizeError("int8 模型正在生成")
        self.state = {"running": True, "done": 0, "total": len(images), "error": ""}
        self.task = asyncio.create_task(self._run(src, dest, images, spec))

    async def _run(self, src: Path, dest: Path, images: list[np.ndarray], spec: QuantSpec) -> None:
        try:
            await asyncio.to_thread(self._quantize, src, dest, images, spec)
        except Exception as e:  # noqa: BLE001 - onnx and onnxruntime raise various exception types
            self.state["error"] = str(e) or type(e).__name__
        finally:
            self.state["running"] = False

    def _quantize(self, src: Path, dest: Path, images: list[np.ndarray], spec: QuantSpec) -> None:
        import onnx
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process

        state = self.state

        class Pages(CalibrationDataReader):
            def __init__(self, input_name: str):
                self.input_name = input_name
                self.left = iter(images)

            def get_next(self):
                image = next(self.left, None)
                if image is None:
                    return None
                state["done"] += 1
                return {self.input_name: image.astype(np.float32) / 255.0}

        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            pre = Path(tmp) / "pre.onnx"
            out = Path(tmp) / "int8.onnx"
            quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
            graph = onnx.load(str(pre), load_external_data=False).graph
            names = [node.name for node in graph.node]
            quantize_static(
                str(pre),
                str(out),
                Pages(graph.input[0].name),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=True,
                nodes_to_exclude=decode_nodes(names),
                extra_options={"ActivationSymmetric": False, "CalibMovingAverage": True},
            )
            _check(src, out, images[:4], spec)
            out.replace(dest)
        pdf_vision.clear_sessions()


def _check(src: Path, quantized: Path, images: list[np.ndarray], spec: QuantSpec) -> None:
    """Compare the box counts of fp32 and int8 on a few calibration pages, and reject the int8 model when it finds
    less than half: quantization went wrong, for example because the detection head names did not match."""
    import onnxruntime

    def count(path: Path) -> int:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        total = 0
        for image in images:
            total += spec.boxes(session.run(None, {name: image.astype(np.float32) / 255.0})[0])
        return total

    expected, found = count(src), count(quantized)
    if expected and found < expected / 2:
        raise QuantizeError(f"量化后的模型识别出的区域过少（{found} 个，原模型 {expected} 个），未保存")
