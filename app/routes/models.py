"""Routes of the layout model and the formula detection model; both have the same API."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from .. import layout_model, quantize, state
from ..formats.pdf import mupdf_turn
from ..formats.pdf_vision import formula_parallel
from ..json_types import JSONDict
from ..model_files import _model_file
from ..settings import SettingsError

router = APIRouter()

# ---------- Layout model, formula detection model ----------
# Both models have the same API: /api/layout-model, /api/formula-model
MODEL_NAMES = {"layout": "版面模型", "formula": "公式检测模型"}


def _spec(kind: str) -> layout_model.ModelSpec:
    if kind not in layout_model.MODELS:
        raise HTTPException(404, "未知的模型")
    return layout_model.MODELS[kind]


def _model_status(kind: str) -> JSONDict:
    spec = _spec(kind)
    conf = state.settings.model(spec.key)
    path = spec.resolve(conf["path"])
    int8 = quantize.int8_path(path)
    extra = {
        "int8": {
            "path": str(int8),
            "file": layout_model.file_status(int8),
            "available": quantize.available(),
            "install_hint": quantize.INSTALL_HINT,
            "state": dict(state.quantizers[kind].state),
        },
        "in_use": _model_file(str(path), conf["precision"]),
    }
    if spec is layout_model.FORMULA:
        # The values that "自动" on the page currently stands for
        workers, threads = formula_parallel()
        extra["auto"] = {"workers": workers, "threads": threads}
    return {
        **conf,
        **extra,
        "resolved": str(path),
        "default_path": str(spec.default_path()),
        "runtime": layout_model.runtime_version(),
        "install_hint": layout_model.INSTALL_HINT,
        "file": layout_model.file_status(path),
        "download": dict(state.model_downloads[kind].state),
        "sources": [{"id": key, "label": label} for key, (label, _url) in spec.sources.items()],
    }


@router.get("/api/{kind}-model")
def get_model(kind: str) -> JSONDict:
    return _model_status(kind)


@router.put("/api/{kind}-model")
def update_model(kind: str, data: JSONDict = Body(...)) -> JSONDict:
    spec = _spec(kind)
    try:
        state.settings.update(
            {
                spec.key: {
                    k: data[k] for k in ("enabled", "path", "size", "precision", "workers", "threads") if k in data
                }
            }
        )
    except SettingsError as e:
        raise HTTPException(400, str(e)) from e
    return _model_status(kind)


@router.post("/api/{kind}-model/download")
async def download_model(kind: str, data: JSONDict = Body(...)) -> JSONDict:
    """Download to the path in the settings (empty means the default location). The page polls GET for
    progress."""
    spec = _spec(kind)
    target = spec.resolve(str(data.get("path", state.settings.model(spec.key)["path"]) or ""))
    try:
        state.model_downloads[kind].start(str(data.get("source") or ""), target)
    except layout_model.LayoutModelError as e:
        raise HTTPException(400, str(e)) from e
    return _model_status(kind)


@router.post("/api/{kind}-model/test")
async def test_model(kind: str, data: JSONDict | None = Body(None)) -> JSONDict:
    """Load the model from the path in the form (possibly unsaved) and run it once on a sample page."""
    spec = _spec(kind)
    conf = state.settings.model(spec.key)
    form: JSONDict = data or {}
    raw = form.get("path", conf["path"])
    size = form.get("size", conf.get("size"))
    path = spec.resolve(str(raw or ""))
    # Test the file that translation would use
    path = Path(_model_file(str(path), str(form.get("precision", conf["precision"]))))
    try:
        async with mupdf_turn():  # The sample page is rendered with MuPDF, taking turns with running jobs
            return await asyncio.to_thread(layout_model.check, spec, path, size)
    except layout_model.LayoutModelError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/api/{kind}-model/quantize")
async def quantize_model(kind: str) -> JSONDict:
    """Generate the int8 version of the model, calibrated on pages sampled from translated PDFs. The page polls
    GET for progress."""
    spec = _spec(kind)
    if not quantize.available():
        raise HTTPException(
            400, f"生成 int8 模型需要先安装 onnx：请在项目目录运行 {quantize.INSTALL_HINT}，安装后重启服务"
        )
    src = spec.resolve(state.settings.model(spec.key)["path"])
    if not src.is_file():
        raise HTTPException(400, f"找不到{MODEL_NAMES[kind]}文件：{src}，请先下载模型")
    # Generate only one at a time: calibration keeps the intermediate results of every page in memory, and
    # generating two together would take twice as much
    if any(quantizer.running for quantizer in state.quantizers.values()):
        raise HTTPException(
            400,
            "int8 模型正在生成" if state.quantizers[kind].running else "另一个模型的 int8 版本正在生成，请在完成后重试",
        )
    pdfs = [
        path
        for job in state.jobs.list()
        if (path := state.jobs.source_path(job)).suffix.lower() == ".pdf" and path.is_file()
    ]
    quant = quantize.SPECS[kind]
    async with mupdf_turn():  # Sampling pages renders with MuPDF, taking turns with running jobs
        picks = await asyncio.to_thread(quantize.pick_pages, pdfs)
        images = await asyncio.to_thread(quantize.calibration_images, picks, layout_model.sample_page, quant.size)
    try:
        state.quantizers[kind].start(src, quantize.int8_path(src), images, quant)
    except quantize.QuantizeError as e:
        raise HTTPException(400, str(e)) from e
    return _model_status(kind)
