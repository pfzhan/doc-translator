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
# 正在运行的翻译任务，防止被垃圾回收，也用于暂停、删除时取消
running: dict[str, asyncio.Task] = {}
# 用户点了暂停的任务：取消时据此把状态记成“已暂停”，而不是“已中断”
pause_requested: set[str] = set()
ACTIVE = ("queued", "running")
PROGRESS_SAVE_INTERVAL = 5


def _cleanup():
    jobs.cleanup(settings.retention_days)


@asynccontextmanager
async def lifespan(_app):
    """启动时清理一次过期记录，之后每小时检查一次。"""
    async def loop():
        while True:
            try:
                await asyncio.to_thread(_cleanup)  # 扫磁盘是重 IO，别卡事件循环
            except Exception:  # noqa: BLE001
                traceback.print_exc()  # 清理失败记录日志，下一轮再试，不能让后台任务结束
            await asyncio.sleep(CLEANUP_INTERVAL)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Doc Translator", lifespan=lifespan)


async def _run_job(job: Job, translator, bilingual: bool):
    last_save = 0.0

    def progress(done, total):
        nonlocal last_save
        job.done, job.total = done, total
        # 隔几秒存一次进度，服务意外退出后记录里还能看到翻到了哪里
        if time.time() - last_save > PROGRESS_SAVE_INTERVAL:
            last_save = time.time()
            jobs.save(job)

    job.status = "running"
    job.runner = Runner(translator, progress)
    jobs.save(job)
    try:
        outputs = await translate_file(jobs.source_path(job), jobs.out_dir(job.id), job.runner, bilingual,
                                       job.target_lang)
        job.outputs = [p.name for p in outputs]
        job.status = "done"
    except asyncio.CancelledError:
        if job.id in pause_requested:
            job.status, job.error = "paused", ""
        else:
            # 服务关闭等原因被取消
            job.status, job.error = "interrupted", "翻译被中断。已翻译的段落有缓存，继续翻译会很快。"
        raise
    except (TranslatorError, ValueError) as e:
        job.status, job.error = "error", str(e)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
    finally:
        job.finished = time.time()
        # 快照可能很大，写盘放到线程里
        await asyncio.to_thread(jobs.save_preview, job)
        job.sync_info()
        # 释放 Runner 占用的内存；之后预览接口从 preview.json 快照读
        job.runner = None
        jobs.save(job)
        running.pop(job.id, None)
        pause_requested.discard(job.id)
        await translator.aclose()


def _make_translator(job: Job, service: dict):
    """按服务当前的配置创建翻译引擎。配置无效时抛 400，此时任务状态和文件都还没动。"""
    try:
        return create_translator(service, job.target_lang, job.source_lang)
    except TranslatorError as e:
        raise HTTPException(400, str(e))


def _start(job: Job, translator):
    """开始（或继续）翻译。每次都按服务当前的配置创建引擎，所以继续翻译时用的是服务现在选中的模型。"""
    job.model = getattr(translator, "model", "")
    running[job.id] = asyncio.create_task(_run_job(job, translator, job.mode == "bilingual"))


async def _cancel(job_id: str):
    """取消正在跑的翻译任务，等它真正停下来（写完记录）再返回。"""
    task = running.pop(job_id, None)
    if task and not task.done():
        task.cancel()
        # 只吞被等待任务的取消和异常；当前协程自己被取消时仍会向上传播
        await asyncio.gather(task, return_exceptions=True)


async def _pause(job: Job) -> bool:
    if job.status not in ACTIVE:
        return False
    pause_requested.add(job.id)
    await _cancel(job.id)
    # 还在排队、没开始跑的任务被取消时不会进入 _run_job，状态要在这里改
    if job.status in ACTIVE:
        job.status, job.error, job.finished = "paused", "", time.time()
        jobs.save(job)
    pause_requested.discard(job.id)
    return True


def _resume(job: Job, service_id: str | None = None) -> Job:
    """继续（或重新）翻译同一个文件。已翻好的段落在缓存里，不会重复请求。"""
    if job.status in ACTIVE:
        raise HTTPException(400, "这个任务还在翻译中")
    if not jobs.source_path(job).exists():
        raise HTTPException(400, "原文件已不存在，无法重新翻译")
    service = store.get(service_id or job.service_id)
    if not service:
        raise HTTPException(400, "原来的翻译服务已不存在，请选择其他服务")
    # 先验证配置，通过后才重置状态、删旧译文，否则配置失效时译文已丢
    translator = _make_translator(job, service)
    job.service_id, job.service_name = service["id"], service.get("name", "")
    job.status, job.error, job.done, job.total, job.outputs, job.finished = "queued", "", 0, 0, [], 0.0
    for old in jobs.out_dir(job.id).glob("*"):
        old.unlink()
    (jobs.dir(job.id) / "preview.json").unlink(missing_ok=True)
    _start(job, translator)
    jobs.save(job)
    return job


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


@app.post("/api/services/{sid}/clone")
def clone_service(sid: str):
    try:
        return store.clone_to_local(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e))


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
    await asyncio.to_thread(_cleanup)
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"不支持的格式 {ext}，支持: {', '.join(sorted(SUPPORTED))}")
    service = store.get(service_id or store.effective_default())
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
    try:
        size = 0
        with jobs.source_path(job).open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "文件太大（上限 200MB）")
                # 大文件写盘放到线程里，别卡事件循环
                await asyncio.to_thread(f.write, chunk)
    except BaseException:
        # 上传中途失败（读异常、超大小等）：清掉孤儿目录再抛出
        shutil.rmtree(jobs.dir(job.id), ignore_errors=True)
        raise
    job.size = size
    jobs.add(job)
    _start(job, _make_translator(job, service))
    return job.to_dict()


@app.get("/api/jobs")
async def list_jobs():
    # 遍历所有任务目录算大小，放到线程里避免卡事件循环
    usage = await asyncio.to_thread(jobs.disk_usage)
    return {"jobs": [j.to_dict() for j in jobs.list()], "disk_usage": usage,
            "retention_days": settings.retention_days}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).to_dict()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """删除记录，连同原文件、译文和预览。正在翻译的会先停止。"""
    _job_or_404(job_id)
    await _cancel(job_id)
    jobs.delete(job_id)
    return {"ok": True}


@app.post("/api/jobs/pause-all")
async def pause_all_jobs():
    """暂停所有正在翻译和排队的任务。"""
    targets = [j for j in jobs.list() if j.status in ACTIVE]
    results = await asyncio.gather(*(_pause(j) for j in targets))
    return {"paused": sum(results)}


@app.post("/api/jobs/resume-all")
async def resume_all_jobs():
    """继续所有已暂停和已中断的任务，各自用原来的翻译服务（服务当前选中的模型）。"""
    resumed, failed = 0, []
    for job in jobs.list():
        if job.status not in ("paused", "interrupted"):
            continue
        try:
            _resume(job)
            resumed += 1
        except HTTPException as e:
            failed.append({"id": job.id, "filename": job.filename, "error": e.detail})
    return {"resumed": resumed, "failed": failed}


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """暂停：立即停止发送请求。已翻好的段落在缓存里，继续翻译时直接复用。"""
    job = _job_or_404(job_id)
    if not await _pause(job):
        raise HTTPException(400, "这个任务没有在翻译")
    # 暂停期间记录可能被并发删除，别返回已过期的结果
    if jobs.get(job_id) is None:
        raise HTTPException(404, "翻译记录不存在")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str, data: dict | None = Body(None)):
    """继续翻译（暂停、中断、失败后）。已翻好的段落在缓存里，不会重复请求。

    用的是服务当前的配置：在服务里换了模型，后面的段落就用新模型。
    可以传 service_id 换一个服务；换了服务 id 就会按新服务重新翻译。
    必须是 async：_start 会创建后台任务，需要在事件循环里调用。
    """
    return _resume(_job_or_404(job_id), (data or {}).get("service_id")).to_dict()


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
    try:
        index = int(data.get("index", 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "index 必须是整数")
    if job.runner is None:
        # 任务已结束（Runner 已释放）：聚焦没有意义，直接视为成功；还没开始的保持失败
        return {"ok": job.status not in ACTIVE}
    job.runner.focus(index)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str):
    job = _job_or_404(job_id)
    # job.json 可能被篡改过，挡住带路径分隔符的文件名
    if Path(name).name != name:
        raise HTTPException(404, "文件不存在")
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
async def update_settings(data: dict = Body(...)):
    try:
        result = settings.update(data)
    except SettingsError as e:
        raise HTTPException(400, str(e))
    await asyncio.to_thread(_cleanup)  # 改短了保留时间，立即清理
    return result


class NoCacheStaticFiles(StaticFiles):
    """前端文件不让浏览器凭缓存直接用：每次都带 ETag 问一下服务器，没改过返回 304，改过就拿新的。
    否则改了 app.js 后，浏览器可能继续用旧版本（实际发生过：提交后仍然跳进阅读器）。"""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=ROOT / "static", html=True), name="static")
