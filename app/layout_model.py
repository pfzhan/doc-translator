"""PDF 用的两个模型的管理：状态、下载、测试。版面模型（DocLayout-YOLO）框图、行间公式、标题、图注，
公式检测模型（Pix2Text MFD）框行内公式。设置（开没开、文件在哪）存在 data/settings.json。

PDF 翻译时怎么用模型在 formats/pdf_vision.py。这里不引入新依赖：下载用 httpx，
推理要的 onnxruntime 由用户自己装，没装时页面上提示，翻译照常只用规则。
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf

from .formats import pdf_vision

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"


@dataclass(frozen=True)
class ModelSpec:
    """一个模型：设置里的键、默认文件名、文件的 SHA3-256（下载完核对，传坏了不留下）、下载来源。"""
    key: str
    file: str
    sha3: str
    sources: dict[str, tuple[str, str]]

    def default_path(self) -> Path:
        return MODELS_DIR / self.file

    def resolve(self, path: str) -> Path:
        """设置里的路径。留空用默认位置，相对路径按项目目录算。"""
        if not path.strip():
            return self.default_path()
        candidate = Path(path.strip()).expanduser()
        return candidate if candidate.is_absolute() else ROOT / candidate


_LAYOUT_FILE = "doclayout_yolo_docstructbench_imgsz1024.onnx"
_LAYOUT_REPO = "wybxc/DocLayout-YOLO-DocStructBench-onnx"
LAYOUT = ModelSpec(
    key="layout_model",
    file=_LAYOUT_FILE,
    sha3="60be061226930524958b5465c8c04af3d7c03bcb0beb66454f5da9f792e3cf2a",
    sources={
        "huggingface": ("Hugging Face", f"https://huggingface.co/{_LAYOUT_REPO}/resolve/main/{_LAYOUT_FILE}"),
        "hf-mirror": ("HF-Mirror（国内镜像）", f"https://hf-mirror.com/{_LAYOUT_REPO}/resolve/main/{_LAYOUT_FILE}"),
        "modelscope": ("ModelScope（魔搭）",
                       f"https://www.modelscope.cn/models/AI-ModelScope/DocLayout-YOLO-DocStructBench-onnx"
                       f"/resolve/master/{_LAYOUT_FILE}"),
    },
)
_FORMULA_FILE = "pix2text-mfd-1.5.onnx"
_FORMULA_REPO = "breezedeus/pix2text-mfd-1.5"
# ModelScope 上没有这个模型
FORMULA = ModelSpec(
    key="formula_model",
    file=_FORMULA_FILE,
    sha3="0fbada0d2a68e75bbc17b81f7b7adfa9d92faeacc0ee3495d4bd9ad1241da8a8",
    sources={
        "huggingface": ("Hugging Face", f"https://huggingface.co/{_FORMULA_REPO}/resolve/main/{_FORMULA_FILE}"),
        "hf-mirror": ("HF-Mirror（国内镜像）", f"https://hf-mirror.com/{_FORMULA_REPO}/resolve/main/{_FORMULA_FILE}"),
    },
)
MODELS = {"layout": LAYOUT, "formula": FORMULA}
INSTALL_HINT = ".venv/bin/pip install onnxruntime"
_CHUNK = 1 << 20


class LayoutModelError(Exception):
    pass


def runtime_version() -> str:
    try:
        import onnxruntime
    except ImportError:
        return ""
    return str(getattr(onnxruntime, "__version__", "") or "unknown")


def file_status(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "size": 0}
    return {"exists": True, "size": path.stat().st_size}


class Downloader:
    """同一时间只下一份。进度给页面轮询；写到 .part，核对通过才换成正式文件。"""

    def __init__(self, spec: ModelSpec, transport: httpx.AsyncBaseTransport | None = None):
        self.spec = spec
        self.transport = transport
        self.task: asyncio.Task | None = None
        self.state: dict = {"running": False, "done": 0, "total": 0, "error": "", "source": ""}

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, source: str, dest: Path) -> None:
        if source not in self.spec.sources:
            raise LayoutModelError("未知的下载来源")
        if self.running:
            raise LayoutModelError("模型正在下载")
        self.state = {"running": True, "done": 0, "total": 0, "error": "", "source": source}
        self.task = asyncio.create_task(self._run(self.spec.sources[source][1], dest))

    async def _run(self, url: str, dest: Path) -> None:
        part = dest.with_name(dest.name + ".part")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha3_256()
            async with httpx.AsyncClient(timeout=60, follow_redirects=True, transport=self.transport) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        raise LayoutModelError(f"下载失败：HTTP {resp.status_code}")
                    self.state["total"] = int(resp.headers.get("content-length") or 0)
                    with part.open("wb") as out:
                        async for chunk in resp.aiter_bytes(_CHUNK):
                            out.write(chunk)
                            digest.update(chunk)
                            self.state["done"] += len(chunk)
            if digest.hexdigest() != self.spec.sha3:
                raise LayoutModelError("下载的文件校验未通过，可能下载不完整，请更换来源后重试")
            part.replace(dest)
        except (httpx.HTTPError, OSError, LayoutModelError) as e:
            self.state["error"] = str(e) or type(e).__name__
            part.unlink(missing_ok=True)
        finally:
            self.state["running"] = False


def check(spec: ModelSpec, path: Path, size: int | None = None) -> dict:
    """加载模型，在一张自造的页面上跑一遍，报出耗时和框出来的类别。size 是公式检测模型的输入边长。"""
    if not runtime_version():
        raise LayoutModelError(f"未安装 onnxruntime，请在项目目录运行：{INSTALL_HINT}")
    if not path.is_file():
        raise LayoutModelError(f"找不到模型文件：{path}")
    start = time.monotonic()
    try:
        # 公式检测模型按自动定的线程数加载，和翻译时的第一份一样
        threads = pdf_vision.formula_parallel()[1] if spec is FORMULA else 0
        session = pdf_vision.load_session(str(path), threads, spin=spec is not FORMULA)
    except Exception as e:  # noqa: BLE001  模型文件不对时 onnxruntime 抛的异常类型不固定
        raise LayoutModelError(f"模型加载失败：{e}") from e
    loaded = time.monotonic()
    doc = sample_page()
    if spec is FORMULA:
        if size not in pdf_vision.FORMULA_SIZES:
            size = pdf_vision.FORMULA_SIZES[0]
        detector = pdf_vision.formula_detector(doc, pdf_vision.FormulaModel(str(path), size, workers=1, threads=threads))
        if detector is None:
            raise LayoutModelError("该文件不是公式检测模型：类别中没有行内公式（embedding）")
        regions = detector.regions(0)
    else:
        regions = pdf_vision.ModelLayout(doc, session).regions(0)
    done = time.monotonic()
    return {
        "load_ms": int((loaded - start) * 1000),
        "page_ms": int((done - loaded) * 1000),
        "labels": sorted({region.label for region in regions}),
    }


def sample_page() -> pymupdf.Document:
    """一页小论文：标题、两段正文、一张带图注的图，一行带行内公式的句子。"""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 90), "1 Introduction", fontsize=16, fontname="hebo")
    body = ("Decision support applications involve complex queries on very large databases. "
            "Since response times should be small, query optimization is critical. ") * 3
    page.insert_textbox(pymupdf.Rect(72, 110, 523, 230), body, fontsize=10, fontname="tiro")
    figure = pymupdf.Rect(150, 260, 445, 460)
    page.draw_rect(figure, color=(0, 0, 0), width=1)
    for i in range(5):
        page.draw_line((170 + i * 55, 440), (195 + i * 55, 300 + i * 25), color=(0.2, 0.3, 0.8), width=2)
    page.insert_text((150, 480), "Figure 1: The lattice of views.", fontsize=9, fontname="tiro")
    page.insert_textbox(pymupdf.Rect(72, 500, 523, 620), body, fontsize=10, fontname="tiro")
    page.insert_text((72, 650), "We return the label y = 1 when", fontsize=10, fontname="tiro")
    page.insert_text((215, 650), "P(y = 1 | x) > 0.5", fontsize=10, fontname="tiit")
    page.insert_text((300, 650), "and the label y = 0 otherwise.", fontsize=10, fontname="tiro")
    return doc
