"""Web 服务：上传文档 → 后台翻译 → 边翻边预览 → 下载结果；翻译记录持久化在 data/jobs/。"""
import asyncio
import mimetypes
import os
import shutil
import time
import traceback
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import history, layout_model, quantize
from .formats import PREVIEW_FORMATS, SUPPORTED, output_variant, translate_file
from .formats.epub import book_outline
from .formats.pdf import mupdf_turn
from .formats.pdf_vision import FormulaModel, formula_parallel
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
# 正在补生成另一种版本的请求。删除和过期清理要等它写完，第二次请求直接拒绝。
variant_tasks: dict[str, asyncio.Future[None]] = {}
# 过期清理已经决定要删、还没删完的任务。补生成看到它就停，避免删目录时还在写。
deleting: set[str] = set()
# 用户点了暂停的任务：取消时据此把状态记成“已暂停”，而不是“已中断”
pause_requested: set[str] = set()
# 版面模型的下载（同一时间只有一个）
model_downloads = {kind: layout_model.Downloader(spec) for kind, spec in layout_model.MODELS.items()}
# 两个模型的 int8 版在本机生成（见 quantize.py），同一时间只生成一个
quantizers = {kind: quantize.Quantizer() for kind in layout_model.MODELS}
ACTIVE = ("queued", "running")
PROGRESS_SAVE_INTERVAL = 5


async def _cleanup() -> None:
    """删过期记录。是否在补生成只在事件循环里判断，不把那个字典交给别的线程。"""
    expired = await asyncio.to_thread(jobs.expired_ids, settings.retention_days)
    for jid in expired:
        if jid in variant_tasks or jid in running or jid in deleting:
            continue
        deleting.add(jid)
        # 登记和再检查之间没有 await，补生成插不进来
        if jid in variant_tasks:
            deleting.discard(jid)
            continue
        try:
            await asyncio.to_thread(jobs.delete, jid)
        finally:
            deleting.discard(jid)


@asynccontextmanager
async def lifespan(_app):
    """启动时清理一次过期记录，之后每小时检查一次。"""
    async def loop():
        while True:
            try:
                await _cleanup()  # 扫磁盘放在 _cleanup 里的线程，别卡事件循环
            except Exception:  # noqa: BLE001
                traceback.print_exc()  # 清理失败记录日志，下一轮再试，不能让后台任务结束
            await asyncio.sleep(CLEANUP_INTERVAL)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Doc Translator", lifespan=lifespan)
# 只接受本机地址访问。否则恶意网页把自己的域名解析到 127.0.0.1（DNS 重绑定），就能以同源身份调用这些接口，
# 比如用已保存的 Key 去测一个它指定的 API 地址。在局域网里用 IP 访问时，把地址加到环境变量 ALLOWED_HOSTS，逗号分隔
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]",
                 *(h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip())]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """上传超过上限时，在读请求体之前就拒绝。否则 Starlette 会先把整个文件写进临时文件，create_job 里的检查才轮到。"""
    if request.method == "POST" and request.url.path == "/api/jobs":
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        # 留 1MB 给 multipart 的分隔符和表单字段
        if length > MAX_UPLOAD + 1024 * 1024:
            return JSONResponse({"detail": "文件太大（上限 200MB）"}, status_code=413)
    return await call_next(request)


async def _run_job(job: Job, translator, bilingual: bool, fresh: bool = False):
    last_save = 0.0

    def progress(done, total):
        nonlocal last_save
        job.done, job.total = done, total
        # 隔几秒存一次进度，服务意外退出后记录里还能看到翻到了哪里
        if time.time() - last_save > PROGRESS_SAVE_INTERVAL:
            last_save = time.time()
            jobs.save(job)

    job.status = "running"
    job.runner = Runner(translator, progress, fresh=fresh)
    jobs.save(job)
    try:
        outputs = await translate_file(jobs.source_path(job), jobs.out_dir(job.id), job.runner, bilingual,
                                       job.target_lang, layout_model=_model_path("layout"),
                                       plan_path=jobs.plan_path(job.id), formula_model=_formula_model())
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
        # 累计本次运行的 token 用量（继续/重试会加在之前的数上）
        usage = getattr(translator, "usage", None)
        if usage:
            job.prompt_tokens += usage.get("prompt", 0)
            job.completion_tokens += usage.get("completion", 0)
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


def _start(job: Job, translator, fresh: bool = False):
    """开始（或继续）翻译。每次都按服务当前的配置创建引擎，所以继续翻译时用的是服务现在选中的模型。
    fresh 为真时不用已有译文，每一段都重新请求。"""
    job.model = getattr(translator, "model", "")
    running[job.id] = asyncio.create_task(_run_job(job, translator, job.mode == "bilingual", fresh))


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


def _resume(job: Job, service_id: str | None = None, fresh: bool = False) -> Job:
    """继续（或重新）翻译同一个文件。已翻好的段落在缓存里，不会重复请求；fresh 为真时重新分析版面、全部重新请求。"""
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
    # 重新计算预览标记：旧记录可能是加预览功能之前创建的（存的是 False）
    job.preview = Path(job.filename).suffix.lower() in PREVIEW_FORMATS
    for old in jobs.out_dir(job.id).glob("*"):
        old.unlink()
    (jobs.dir(job.id) / "preview.json").unlink(missing_ok=True)
    if fresh:
        # 全部重新翻译从版面分析做起
        jobs.plan_path(job.id).unlink(missing_ok=True)
    _start(job, translator, fresh)
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
    await _cleanup()
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


async def _delete(job_id: str) -> None:
    """删除记录，连同原文件、译文和预览。正在翻译的会先停止；正在补生成的等它写完再删。"""
    await _cancel(job_id)
    task = variant_tasks.get(job_id)
    if task is not None and task is not asyncio.current_task() and not task.done():
        await asyncio.gather(task, return_exceptions=True)
    jobs.delete(job_id)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    _job_or_404(job_id)
    await _delete(job_id)
    return {"ok": True}


@app.post("/api/jobs/delete")
async def delete_jobs(data: dict = Body(...)):
    """批量删除选中的记录。已经不存在的 id 放进 missing 返回，不影响其他的。"""
    ids = data.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise HTTPException(400, "ids 必须是记录 id 列表")
    deleted, missing = 0, []
    for job_id in dict.fromkeys(ids):
        if jobs.get(job_id) is None:
            missing.append(job_id)
            continue
        await _delete(job_id)
        deleted += 1
    return {"deleted": deleted, "missing": missing}


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
    """继续翻译（暂停、中断、失败后），或重译已完成的记录。已翻好的段落在缓存里，不会重复请求。

    用的是服务当前的配置：在服务里换了模型，后面的段落就用新模型。
    可以传 service_id 换一个服务；换了服务 id 就会按新服务重新翻译。
    传 fresh=true 时不用已有译文，每一段都重新请求。
    必须是 async：_start 会创建后台任务，需要在事件循环里调用。
    """
    data = data or {}
    return _resume(_job_or_404(job_id), data.get("service_id"), fresh=bool(data.get("fresh"))).to_dict()


def _variant_service(job: Job, service: dict) -> dict:
    """用当初翻译时的模型去对缓存。不改服务里当前选中的模型。"""
    if not job.model:
        return service
    pinned = dict(service)
    pinned["model"] = job.model
    return pinned


@app.post("/api/jobs/{job_id}/variant")
async def build_variant(job_id: str):
    """生成另一种版本（双语 ⇄ 仅译文）。只按当时的缓存重新排版，不请求翻译服务。"""
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(400, "翻译完成后才能生成另一种版本")
    bilingual = job.mode != "bilingual"
    want = "bilingual" if bilingual else "translated"
    if any(output_variant(name) == want for name in job.outputs):
        return job.to_dict()  # 已经生成过
    if job.id in deleting:
        raise HTTPException(409, "这条记录正在清理，请稍后再试")
    if job.id in variant_tasks:
        raise HTTPException(409, "正在生成另一种版本，请稍候")
    task = asyncio.current_task()
    if task is None:
        raise HTTPException(500, "无法生成另一种版本")
    # 登记必须发生在下一个 await 之前，否则两个请求会同时通过上面的检查
    variant_tasks[job.id] = task
    translator = None
    stage = jobs.dir(job.id) / ".variant"
    try:
        if not jobs.source_path(job).exists():
            raise HTTPException(400, "原文件已不存在，无法生成")
        service = store.get(job.service_id)
        if not service:
            raise HTTPException(400, "原来的翻译服务已不存在，无法生成")
        translator = _make_translator(job, _variant_service(job, service))
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        try:
            outputs = await translate_file(
                jobs.source_path(job), stage, Runner(translator, cache_only=True), bilingual, job.target_lang,
                layout_model=_model_path("layout"), plan_path=jobs.plan_path(job.id),
                formula_model=_formula_model(),
            )
        except (TranslatorError, ValueError) as e:
            raise HTTPException(400, str(e))
        current = jobs.get(job.id)
        if current is None or not jobs.dir(job.id).exists():
            raise HTTPException(404, "翻译记录不存在")
        out_dir = jobs.out_dir(job.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 先落到临时目录，替换进 out/ 后再写入记录，避免写到一半被读到
        for path in outputs:
            path.replace(out_dir / path.name)
            if path.name not in current.outputs:
                current.outputs.append(path.name)
        jobs.save(current)
        if jobs.get(job.id) is None:
            raise HTTPException(404, "翻译记录不存在")
        return current.to_dict()
    finally:
        variant_tasks.pop(job.id, None)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if translator is not None:
            await translator.aclose()


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str, since: int = -1):
    """边翻边预览。since=-1 返回全部段落和已完成译文；之后用返回的 version 增量拉取。"""
    job = _job_or_404(job_id)
    # 这个接口在线程池里跑，任务结束时事件循环会把 job.runner 置空，先取到局部变量再用
    runner = job.runner
    if runner is not None:
        return {**runner.preview(since), "status": job.status}
    # 已结束的任务（包括重启前的）从快照读
    snap = jobs.load_preview(job)
    if snap is None:
        return {"ready": False, "version": 0, "status": job.status}
    if since >= snap["version"]:
        return {"ready": True, "version": snap["version"], "updates": [], "status": job.status}
    if since >= 0:
        # 上次拉取之后到任务结束之间翻好的段落只在快照里。快照不分版本，整份返回，前端会跳过已显示的段落
        return {"ready": True, "version": snap["version"], "updates": snap["updates"], "status": job.status}
    return {**snap, "status": job.status}


@app.post("/api/jobs/{job_id}/focus")
def job_focus(job_id: str, data: dict = Body(...)):
    """预览里用户跳到了哪一段，后面优先翻译这附近。"""
    job = _job_or_404(job_id)
    try:
        index = int(data.get("index", 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "index 必须是整数")
    runner = job.runner
    if runner is None:
        # 任务已结束（Runner 已释放）：聚焦没有意义，直接视为成功；还没开始的保持失败
        return {"ok": job.status not in ACTIVE}
    runner.focus(index)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str, inline: bool = False):
    """inline=1 时让浏览器直接打开（PDF 用自带的阅读器看），不弹下载。"""
    job = _job_or_404(job_id)
    # job.json 可能被篡改过，挡住带路径分隔符的文件名
    if Path(name).name != name:
        raise HTTPException(404, "文件不存在")
    if name in job.outputs:
        path = jobs.out_dir(job.id) / name
        if path.exists():
            return FileResponse(path, filename=name, content_disposition_type="inline" if inline else "attachment")
    raise HTTPException(404, "文件不存在")


# 书里的页面直接从 zip 里取出来给浏览器。书里的脚本一律不跑，免得它在本站的源下调接口；
# 保留同源，书里的字体、样式照常加载，预览页也能跟上读者点进去的章节
_BOOK_PAGE_HEADERS = {
    "Content-Security-Policy": "sandbox allow-same-origin allow-popups; script-src 'none'; object-src 'none'",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-cache",
}


def _epub_output(job_id: str, name: str) -> Path:
    job = _job_or_404(job_id)
    path = jobs.out_dir(job.id) / name
    if Path(name).name != name or name not in job.outputs or not name.lower().endswith(".epub") or not path.exists():
        raise HTTPException(404, "文件不存在")
    return path


@app.get("/api/jobs/{job_id}/epub/{name}")
def epub_outline(job_id: str, name: str):
    """在线预览 EPUB：书名、按 spine 排好的章节和目录。章节内容从下面的接口按成员名取。"""
    path = _epub_output(job_id, name)
    try:
        with zipfile.ZipFile(path) as zf:
            return book_outline(zf)
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise HTTPException(422, f"EPUB 读不出来：{e}") from e


@app.get("/api/jobs/{job_id}/epub/{name}/{member:path}")
def epub_member(job_id: str, name: str, member: str):
    path = _epub_output(job_id, name)
    try:
        with zipfile.ZipFile(path) as zf:
            data = zf.read(member)
    except (zipfile.BadZipFile, KeyError):
        raise HTTPException(404, "文件不存在") from None
    media = mimetypes.guess_type(member)[0] or "application/octet-stream"
    return Response(data, media_type=media, headers=_BOOK_PAGE_HEADERS)


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
    await _cleanup()  # 改短了保留时间，立即清理
    return result


# ---------- 版面模型、公式检测模型 ----------
# 两个模型的接口一样：/api/layout-model、/api/formula-model
MODEL_NAMES = {"layout": "版面模型", "formula": "公式检测模型"}


def _spec(kind: str) -> layout_model.ModelSpec:
    if kind not in layout_model.MODELS:
        raise HTTPException(404, "没有这个模型")
    return layout_model.MODELS[kind]


def _model_path(kind: str) -> str:
    """PDF 翻译用的模型文件（按精度选，见 _model_file）。没开时是空串，只用规则。"""
    spec = layout_model.MODELS[kind]
    conf = settings.model(spec.key)
    return _model_file(str(spec.resolve(conf["path"])), conf["precision"]) if conf["enabled"] else ""


def _formula_model() -> FormulaModel | None:
    path = _model_path("formula")
    conf = settings.formula_model
    return FormulaModel(path, conf["size"], conf["workers"], conf["threads"]) if path else None


def _model_file(path: str, precision: str) -> str:
    """选了 int8、int8 文件也生成了时用它，否则用 fp32 文件。"""
    int8 = quantize.int8_path(Path(path))
    return str(int8) if precision == "int8" and int8.is_file() else path


def _model_status(kind: str) -> dict:
    spec = _spec(kind)
    conf = settings.model(spec.key)
    path = spec.resolve(conf["path"])
    int8 = quantize.int8_path(path)
    extra = {
        "int8": {
            "path": str(int8), "file": layout_model.file_status(int8), "available": quantize.available(),
            "install_hint": quantize.INSTALL_HINT, "state": dict(quantizers[kind].state),
        },
        "in_use": _model_file(str(path), conf["precision"]),
    }
    if spec is layout_model.FORMULA:
        # 页面上「自动」现在对应的值
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
        "download": dict(model_downloads[kind].state),
        "sources": [{"id": key, "label": label} for key, (label, _url) in spec.sources.items()],
    }


@app.get("/api/{kind}-model")
def get_model(kind: str):
    return _model_status(kind)


@app.put("/api/{kind}-model")
def update_model(kind: str, data: dict = Body(...)):
    spec = _spec(kind)
    try:
        settings.update({spec.key: {k: data[k] for k in ("enabled", "path", "size", "precision", "workers", "threads") if k in data}})
    except SettingsError as e:
        raise HTTPException(400, str(e))
    return _model_status(kind)


@app.post("/api/{kind}-model/download")
async def download_model(kind: str, data: dict = Body(...)):
    """下到设置里的路径（留空是默认位置）。页面轮询 GET 看进度。"""
    spec = _spec(kind)
    target = spec.resolve(str(data.get("path", settings.model(spec.key)["path"]) or ""))
    try:
        model_downloads[kind].start(str(data.get("source") or ""), target)
    except layout_model.LayoutModelError as e:
        raise HTTPException(400, str(e))
    return _model_status(kind)


@app.post("/api/{kind}-model/test")
async def test_model(kind: str, data: dict | None = Body(None)):
    """用表单里的路径（可能还没保存）加载模型，在一张样例页上跑一遍。"""
    spec = _spec(kind)
    conf = settings.model(spec.key)
    raw = (data or {}).get("path", conf["path"])
    size = (data or {}).get("size", conf.get("size"))
    path = spec.resolve(str(raw or ""))
    # 测翻译时会用的那个文件
    path = Path(_model_file(str(path), (data or {}).get("precision", conf["precision"])))
    try:
        async with mupdf_turn():  # 样例页用 MuPDF 画，和正在跑的任务错开
            return await asyncio.to_thread(layout_model.check, spec, path, size)
    except layout_model.LayoutModelError as e:
        raise HTTPException(400, str(e))


@app.post("/api/{kind}-model/quantize")
async def quantize_model(kind: str):
    """用翻译过的 PDF 抽页校准，生成模型的 int8 版。页面轮询 GET 看进度。"""
    spec = _spec(kind)
    if not quantize.available():
        raise HTTPException(400, f"生成 int8 模型要先安装 onnx，在项目目录运行：{quantize.INSTALL_HINT}，装好后重启服务")
    src = spec.resolve(settings.model(spec.key)["path"])
    if not src.is_file():
        raise HTTPException(400, f"找不到{MODEL_NAMES[kind]}文件：{src}，请先下载模型")
    # 一次只生成一个：校准时每页的中间结果都留在内存里，两个一起生成要多占一倍
    if any(quantizer.running for quantizer in quantizers.values()):
        raise HTTPException(400, "int8 模型正在生成" if quantizers[kind].running else "另一个模型的 int8 版正在生成，完成后再试")
    pdfs = [path for job in jobs.list() if (path := jobs.source_path(job)).suffix.lower() == ".pdf" and path.is_file()]
    quant = quantize.SPECS[kind]
    async with mupdf_turn():  # 抽页要用 MuPDF 画，和正在跑的任务错开
        picks = await asyncio.to_thread(quantize.pick_pages, pdfs)
        images = await asyncio.to_thread(quantize.calibration_images, picks, layout_model.sample_page, quant.size)
    try:
        quantizers[kind].start(src, quantize.int8_path(src), images, quant)
    except quantize.QuantizeError as e:
        raise HTTPException(400, str(e))
    return _model_status(kind)


class NoCacheStaticFiles(StaticFiles):
    """前端文件不让浏览器凭缓存直接用：每次都带 ETag 问一下服务器，没改过返回 304，改过就拿新的。
    否则改了 app.js 后，浏览器可能继续用旧版本（实际发生过：提交后仍然跳进阅读器）。"""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=ROOT / "static", html=True), name="static")
