"""Web 服务：上传文档 → 后台翻译 → 边翻边预览 → 下载结果；翻译记录持久化在 data/jobs/。"""
import asyncio
import shutil
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import history
from .formats import PREVIEW_FORMATS, SUPPORTED, translate_file
from .history import Job, JobStore
from .languages import GOOGLE_CODES, LANGUAGES
from .prompts import default_prompts
from .runner import Runner
from .services import PROVIDERS, ServiceError, ServiceStore
from .settings import Settings, SettingsError
from .translators import TranslatorError, create_translator

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 200 * 1024 * 1024
CLEANUP_INTERVAL = 3600

store = ServiceStore()
settings = Settings()
jobs = JobStore(history.JOBS_DIR)
# 正在运行的翻译任务，防止被垃圾回收，也用于删除时取消
running: dict[str, asyncio.Task] = {}


def _cleanup():
    jobs.cleanup(settings.retention_days)


@asynccontextmanager
async def lifespan(_app):
    """启动时清理一次过期记录，之后每小时检查一次。"""
    async def loop():
        while True:
            _cleanup()
            await asyncio.sleep(CLEANUP_INTERVAL)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Doc Translator", lifespan=lifespan)


async def _run_job(job: Job, translator, bilingual: bool):
    def progress(done, total):
        job.done, job.total = done, total

    job.status = "running"
    job.runner = Runner(translator, progress)
    jobs.save(job)
    try:
        outputs = await translate_file(jobs.source_path(job), jobs.out_dir(job.id), job.runner, bilingual,
                                       job.target_lang)
        job.outputs = [p.name for p in outputs]
        job.status = "done"
    except asyncio.CancelledError:
        job.status, job.error = "error", "已取消"
        raise
    except (TranslatorError, ValueError) as e:
        job.status, job.error = "error", str(e)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
    finally:
        job.finished = time.time()
        jobs.save_preview(job)
        jobs.save(job)
        running.pop(job.id, None)
        await translator.aclose()


def _start(job: Job, service: dict):
    try:
        translator = create_translator(service, job.target_lang, job.source_lang)
    except TranslatorError as e:
        raise HTTPException(400, str(e))
    running[job.id] = asyncio.create_task(_run_job(job, translator, job.mode == "bilingual"))


@app.get("/api/languages")
def languages():
    """语言列表，顺序和插件一致。google 字段表示谷歌翻译是否支持。"""
    return [
        {"code": code, "name": en, "native": native, "google": code == "auto" or code in GOOGLE_CODES}
        for code, en, native in LANGUAGES
    ]


# ---------- 翻译服务管理 ----------

@app.get("/api/providers")
def providers():
    return PROVIDERS


@app.get("/api/prompts/default")
def default_prompt(target_lang: str = "zh-CN", source_lang: str = "auto"):
    return default_prompts(target_lang, source_lang)


@app.get("/api/services")
def list_services():
    return store.list()


@app.post("/api/services")
def create_service(data: dict = Body(...)):
    try:
        return store.create(data)
    except ServiceError as e:
        raise HTTPException(400, str(e))


@app.put("/api/services/{sid}")
def update_service(sid: str, data: dict = Body(...)):
    try:
        return store.update(sid, data)
    except ServiceError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/services/{sid}")
def delete_service(sid: str):
    try:
        store.delete(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/services/{sid}/default")
def set_default_service(sid: str):
    try:
        store.set_default(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e))
    return store.list()


async def _with_translator(data: dict, fn):
    try:
        translator = create_translator(
            store.resolve(data), data.get("target_lang") or "zh-CN", data.get("source_lang") or "auto"
        )
    except (ServiceError, TranslatorError) as e:
        raise HTTPException(400, str(e))
    try:
        return await fn(translator)
    except TranslatorError as e:
        raise HTTPException(400, str(e))
    finally:
        await translator.aclose()


@app.post("/api/services/test")
async def test_service(data: dict = Body(...)):
    """用表单里当前（可能尚未保存）的配置翻译一句话，验证 Key / 地址 / 模型是否可用。"""
    # 目标语言是英语时用中文例句，否则用英文例句，避免“英译英”
    target = data.get("target_lang") or "zh-CN"
    default_sample = "你好，世界！这是一条翻译测试。" if target == "en" else "Hello, world! This is a translation test."
    sample = data.get("text") or default_sample

    async def run(tr):
        start = time.monotonic()
        [out] = await tr.translate_batch([sample])
        return {"source": sample, "result": out, "ms": int((time.monotonic() - start) * 1000)}

    return await _with_translator(data, run)


@app.post("/api/services/models")
async def fetch_models(data: dict = Body(...)):
    async def run(tr):
        if not hasattr(tr, "list_models"):
            raise TranslatorError("该服务没有模型列表")
        return {"models": await tr.list_models()}

    return await _with_translator(data, run)


# ---------- 翻译任务 / 翻译记录 ----------

def _job_or_404(job_id: str) -> Job:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "翻译记录不存在")
    return job


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    target_lang: str = Form("zh-CN"),
    source_lang: str = Form("auto"),
    service_id: str = Form(""),
    mode: str = Form("bilingual"),
):
    _cleanup()
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"不支持的格式 {ext}，支持: {', '.join(sorted(SUPPORTED))}")
    service = store.get(service_id or store.default_id)
    if not service:
        raise HTTPException(400, "翻译服务不存在")
    # 先检查配置，避免上传完才发现服务不可用
    try:
        await create_translator(service, target_lang, source_lang).aclose()
    except TranslatorError as e:
        raise HTTPException(400, str(e))

    job = Job(id=uuid.uuid4().hex[:12], filename=name, source_lang=source_lang, target_lang=target_lang,
              mode="translated" if mode == "translated" else "bilingual", service_id=service["id"],
              service_name=service.get("name", ""), preview=ext in PREVIEW_FORMATS)
    jobs.out_dir(job.id).mkdir(parents=True)
    size = 0
    with jobs.source_path(job).open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                shutil.rmtree(jobs.dir(job.id), ignore_errors=True)
                raise HTTPException(413, "文件太大（上限 200MB）")
            f.write(chunk)
    job.size = size
    jobs.add(job)
    _start(job, service)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": [j.to_dict() for j in jobs.list()], "disk_usage": jobs.disk_usage(),
            "retention_days": settings.retention_days}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).to_dict()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """删除记录，连同原文件、译文和预览。正在翻译的会先停止。"""
    _job_or_404(job_id)
    task = running.pop(job_id, None)
    if task:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    jobs.delete(job_id)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str, data: dict | None = Body(None)):
    """用同一个文件重新翻译（中断、失败后继续）。已翻好的段落在缓存里，不会重复请求。

    可以传 service_id 换一个服务；换了服务 id 就会按新服务重新翻译。
    必须是 async：_start 会创建后台任务，需要在事件循环里调用。
    """
    job = _job_or_404(job_id)
    if job.status in ("queued", "running"):
        raise HTTPException(400, "这个任务还在翻译中")
    if not jobs.source_path(job).exists():
        raise HTTPException(400, "原文件已不存在，无法重新翻译")
    service = store.get((data or {}).get("service_id") or job.service_id)
    if not service:
        raise HTTPException(400, "原来的翻译服务已不存在，请选择其他服务")
    job.service_id, job.service_name = service["id"], service.get("name", "")
    job.status, job.error, job.done, job.total, job.outputs, job.finished = "queued", "", 0, 0, [], 0.0
    for old in jobs.out_dir(job.id).glob("*"):
        old.unlink()
    (jobs.dir(job.id) / "preview.json").unlink(missing_ok=True)
    _start(job, service)
    jobs.save(job)
    return job.to_dict()


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str, since: int = -1):
    """边翻边预览。since=-1 返回全部段落和已完成译文；之后用返回的 version 增量拉取。"""
    job = _job_or_404(job_id)
    if job.runner is not None:
        return {**job.runner.preview(since), "status": job.status}
    # 已结束的任务（包括重启前的）从快照读
    snap = jobs.load_preview(job)
    if snap is None:
        return {"ready": False, "version": 0, "status": job.status}
    if since >= 0:
        return {"ready": True, "version": snap["version"], "updates": [], "status": job.status}
    return {**snap, "status": job.status}


@app.post("/api/jobs/{job_id}/focus")
def job_focus(job_id: str, data: dict = Body(...)):
    """预览里用户跳到了哪一段，后面优先翻译这附近。"""
    job = _job_or_404(job_id)
    if job.runner is None:
        return {"ok": False}
    try:
        job.runner.focus(int(data.get("index", 0)))
    except (TypeError, ValueError):
        raise HTTPException(400, "index 必须是整数")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str):
    job = _job_or_404(job_id)
    if name in job.outputs:
        path = jobs.out_dir(job.id) / name
        if path.exists():
            return FileResponse(path, filename=name)
    raise HTTPException(404, "文件不存在")


# ---------- 设置 ----------

@app.get("/api/settings")
def get_settings():
    return settings.public()


@app.put("/api/settings")
def update_settings(data: dict = Body(...)):
    try:
        result = settings.update(data)
    except SettingsError as e:
        raise HTTPException(400, str(e))
    _cleanup()  # 改短了保留时间，立即清理
    return result


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
