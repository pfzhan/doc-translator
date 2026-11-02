"""Web server: upload a document -> translate in the background -> preview during translation -> download the
result. Translation records are persisted in data/jobs/."""

import asyncio
import hashlib
import mimetypes
import os
import shutil
import time
import traceback
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import cleanup, history, layout_model, quantize
from .formats import PREVIEW_FORMATS, SUPPORTED, output_variant, translate_file
from .formats.epub import book_outline
from .formats.pdf import mupdf_turn
from .formats.pdf_output import translated_from_bilingual
from .formats.pdf_vision import FormulaModel, formula_parallel
from .history import Job, JobStore
from .languages import GOOGLE_CODES, LANGUAGES
from .prompts import default_prompts
from .runner import Runner, get_cache
from .services import PROVIDERS, ServiceError, ServiceStore
from .settings import Settings, SettingsError
from .translators import TranslatorError, create_translator

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 200 * 1024 * 1024
# 50 texts of 10,000 characters are at most 2 MB as UTF-8 JSON (the extension sends characters unescaped); twice
# that leaves room for clients that escape them
MAX_WEB_REQUEST = 4 * 1024 * 1024
CLEANUP_INTERVAL = 3600

store = ServiceStore()
settings = Settings()
jobs = JobStore(history.JOBS_DIR)
# Running translation jobs; keeps them from being garbage collected and is used to cancel on pause or delete
running: dict[str, asyncio.Task] = {}
# Requests generating the other version. Deletion and expiry cleanup wait until they finish writing; a second
# request is rejected.
variant_tasks: dict[str, asyncio.Future[None]] = {}
# Jobs being cleaned: those expiry cleanup has decided to delete and not finished deleting, or those whose cache
# the user asked to clear and not finished clearing.
# Generating the other version and resuming translation are rejected for them, so nothing writes during cleanup.
cleaning: set[str] = set()
# Jobs the user paused: on cancellation their status is recorded as "已暂停" instead of "已中断"
pause_requested: set[str] = set()
# Layout model downloads (one at a time)
model_downloads = {kind: layout_model.Downloader(spec) for kind, spec in layout_model.MODELS.items()}
# int8 versions of both models are generated locally (see quantize.py), one at a time
quantizers = {kind: quantize.Quantizer() for kind in layout_model.MODELS}
ACTIVE = ("queued", "running")
PROGRESS_SAVE_INTERVAL = 5


async def _cleanup() -> None:
    """Delete expired records; move layout analysis results from record directories to the shared directory and
    delete those no record uses; replace duplicate source files with hard links.
    Whether the other version is being generated is checked only in the event loop; that dict is not handed to
    other threads."""
    expired = await asyncio.to_thread(jobs.expired_ids, settings.retention_days)
    for jid in expired:
        if jid in variant_tasks or jid in running or jid in cleaning:
            continue
        cleaning.add(jid)
        # There is no await between registering and checking again, so generating the other version cannot get in
        if jid in variant_tasks:
            cleaning.discard(jid)
            continue
        try:
            await _forget_cache(jid)
            await asyncio.to_thread(jobs.delete, jid)
        finally:
            cleaning.discard(jid)
    await asyncio.to_thread(jobs.adopt_layouts)
    await asyncio.to_thread(cleanup.prune_layouts, jobs)
    await asyncio.to_thread(jobs.share_sources)


@asynccontextmanager
async def lifespan(_app):
    """Clean up expired records once at startup, then check every hour."""

    async def loop():
        while True:
            try:
                await _cleanup()  # Disk scans run in threads inside _cleanup, so the event loop is not blocked
            except Exception:  # noqa: BLE001
                traceback.print_exc()  # Log a failed cleanup and retry next round; the background task must not end
            await asyncio.sleep(CLEANUP_INTERVAL)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Doc Translator", lifespan=lifespan)
# Accept only local addresses. Otherwise a malicious web page could resolve its own domain to 127.0.0.1 (DNS
# rebinding) and call these APIs as the same origin, for example to test an API address of its choice with a saved
# Key. When accessing by IP on a LAN, add the address to the environment variable ALLOWED_HOSTS, comma-separated
ALLOWED_HOSTS = [
    "127.0.0.1",
    "localhost",
    "[::1]",
    *(h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()),
]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# Requests that change state are accepted only from this server's own pages, from browser extensions, and from
# clients that send no Origin (curl and other tools). TrustedHostMiddleware does not cover this: any web page the
# user visits can POST to 127.0.0.1 across origins. With a form, or fetch in mode "no-cors" and a simple content
# type, there is no preflight, so the request runs even though the page cannot read the answer: it could pause all
# jobs, delete records or spend the translation quota. Browsers always send Origin on such requests.
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
EXTENSION_ORIGINS = ("chrome-extension://", "moz-extension://")


def _origin_allowed(origin: str | None, host: str) -> bool:
    if origin is None or origin.startswith(EXTENSION_ORIGINS):
        return True
    # Host and port only: behind an HTTPS reverse proxy the page's scheme differs from the one the server sees
    parsed = urlsplit(origin)
    return bool(parsed.netloc) and parsed.netloc.lower() == host.lower()


@app.middleware("http")
async def check_origin(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    if request.method not in SAFE_METHODS and not _origin_allowed(
        request.headers.get("origin"), request.headers.get("host", "")
    ):
        return JSONResponse({"detail": "不接受来自其他网页的请求"}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject an upload above the limit before the request body is read. Otherwise Starlette first writes the whole
    file to a temporary file before the check in create_job runs."""
    if request.method == "POST" and request.url.path == "/api/jobs":
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        # Leave 1 MB for multipart boundaries and form fields
        if length > MAX_UPLOAD + 1024 * 1024:
            return JSONResponse({"detail": "文件太大（上限 200MB）"}, status_code=413)
    elif request.method == "POST" and request.url.path == "/api/web/translate":
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > MAX_WEB_REQUEST:
            return JSONResponse({"detail": "请求太大"}, status_code=413)
    return await call_next(request)


async def _run_job(job: Job, translator, bilingual: bool):
    last_save = 0.0

    def progress(done, total):
        nonlocal last_save
        job.done, job.total = done, total
        # Save progress every few seconds, so the record still shows how far translation got after the server
        # exits unexpectedly
        if time.time() - last_save > PROGRESS_SAVE_INTERVAL:
            last_save = time.time()
            jobs.save(job)

    job.status = "running"
    job.runner = Runner(translator, progress)
    jobs.save(job)
    try:
        outputs = await translate_file(
            jobs.source_path(job),
            jobs.out_dir(job.id),
            job.runner,
            bilingual,
            job.target_lang,
            layout_model=_model_path("layout"),
            plan_path=jobs.plan_path(job),
            formula_model=_formula_model(),
        )
        job.outputs = [p.name for p in outputs]
        job.status = "done"
    except asyncio.CancelledError:
        if job.id in pause_requested:
            job.status, job.error = "paused", ""
        else:
            # Cancelled for reasons such as server shutdown
            job.status, job.error = "interrupted", "翻译已中断。已翻译的段落已缓存，继续翻译时直接复用。"
        raise
    except (TranslatorError, ValueError) as e:
        job.status, job.error = "error", str(e)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
    finally:
        job.finished = time.time()
        # Accumulate the token usage of this run (resume/retry adds to the earlier count)
        usage = getattr(translator, "usage", None)
        if usage:
            job.prompt_tokens += usage.get("prompt", 0)
            job.completion_tokens += usage.get("completion", 0)
        # The snapshot may be large, so it is written in a thread
        await asyncio.to_thread(jobs.save_preview, job)
        job.sync_info()
        job.document = job.runner.document or job.document
        # Release the memory held by the Runner; the preview API then reads from the preview.json snapshot
        job.runner = None
        jobs.save(job)
        running.pop(job.id, None)
        pause_requested.discard(job.id)
        await translator.aclose()


def _make_translator(job: Job, service: dict):
    """Create the translation engine from the service's current configuration. Raise 400 for an invalid
    configuration; job status and files have not been touched at that point."""
    try:
        return create_translator(service, job.target_lang, job.source_lang)
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e


def _start(job: Job, translator):
    """Start (or resume) translation. The engine is created from the service's current configuration each time,
    so resuming uses the model now selected in the service."""
    job.model = getattr(translator, "model", "")
    running[job.id] = asyncio.create_task(_run_job(job, translator, job.mode == "bilingual"))


async def _cancel(job_id: str):
    """Cancel a running translation job and return once it has actually stopped (finished writing its record)."""
    task = running.pop(job_id, None)
    if task and not task.done():
        task.cancel()
        # Swallow only the cancellation and exceptions of the awaited task; cancellation of the current coroutine
        # itself still propagates
        await asyncio.gather(task, return_exceptions=True)


async def _pause(job: Job) -> bool:
    if job.status not in ACTIVE:
        return False
    pause_requested.add(job.id)
    await _cancel(job.id)
    # A job still queued and not started never enters _run_job when cancelled, so its status is changed here
    if job.status in ACTIVE:
        job.status, job.error, job.finished = "paused", "", time.time()
        jobs.save(job)
    pause_requested.discard(job.id)
    return True


async def _resume(job: Job, service_id: str | None = None, fresh: bool = False) -> Job:
    """Resume (or redo) translation of the same file. Translated paragraphs are in the cache and are not requested
    again; with fresh true, what this record left in the caches is cleared first (see _clear), and everything starts
    over from layout analysis with all paragraphs requested again."""
    if job.status in ACTIVE:
        raise HTTPException(400, "该任务正在翻译")
    if job.id in cleaning:
        raise HTTPException(409, "该记录正在清理，请稍后重试")
    if not jobs.source_path(job).exists():
        raise HTTPException(400, "原文件已不存在，无法重新翻译")
    service = store.get(service_id or job.service_id)
    if not service:
        raise HTTPException(400, "原翻译服务已不存在，请选择其他服务")
    # Validate the configuration first and reset status and delete old translations only after it passes;
    # otherwise the translations would be lost when the configuration is invalid
    translator = _make_translator(job, service)
    if fresh:
        try:
            await _clear(job)
        except BaseException:
            await translator.aclose()
            raise
        job.clear_results()
    job.service_id, job.service_name = service["id"], service.get("name", "")
    job.status, job.error, job.done, job.total, job.outputs, job.finished = "queued", "", 0, 0, [], 0.0
    # Recompute the preview flag: old records may have been created before the preview feature (stored as False)
    job.preview = Path(job.filename).suffix.lower() in PREVIEW_FORMATS
    for old in jobs.out_dir(job.id).glob("*"):
        old.unlink()
    (jobs.dir(job.id) / "preview.json").unlink(missing_ok=True)
    _start(job, translator)
    jobs.save(job)
    return job


@app.get("/api/languages")
def languages():
    """Language list in the same order as the plugin. The google field tells whether Google Translate supports
    the language."""
    return [
        {"code": code, "name": en, "native": native, "google": code == "auto" or code in GOOGLE_CODES}
        for code, en, native in LANGUAGES
    ]


# ---------- Translation service management ----------


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
        raise HTTPException(400, str(e)) from e


@app.put("/api/services/{sid}")
def update_service(sid: str, data: dict = Body(...)):
    try:
        return store.update(sid, data)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/api/services/{sid}")
def delete_service(sid: str):
    try:
        store.delete(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@app.post("/api/services/{sid}/default")
def set_default_service(sid: str):
    try:
        store.set_default(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e
    return store.list()


@app.post("/api/services/{sid}/clone")
def clone_service(sid: str):
    try:
        return store.clone_to_local(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e


async def _with_translator(data: dict, fn):
    try:
        translator = create_translator(
            store.resolve(data), data.get("target_lang") or "zh-CN", data.get("source_lang") or "auto"
        )
    except (ServiceError, TranslatorError) as e:
        raise HTTPException(400, str(e)) from e
    try:
        return await fn(translator)
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        await translator.aclose()


@app.post("/api/services/test")
async def test_service(data: dict = Body(...)):
    """Translate one sentence with the configuration currently in the form (possibly unsaved) to check that the
    Key / address / model work."""
    # Use a Chinese sample sentence when the target language is English, otherwise an English one, to avoid
    # translating English into English
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
            raise TranslatorError("该服务不提供模型列表")
        return {"models": await tr.list_models()}

    return await _with_translator(data, run)


# ---------- Web page translation (called by the browser extension in extension/) ----------


class WebTranslateRequest(BaseModel):
    # One request carries the paragraphs near the viewport; the extension sends more as the page scrolls
    texts: list[str] = Field(max_length=50)
    # Aligned with texts: which are HTML fragments (inline formatting the translation keeps, such as links)
    html: list[bool] = Field(default_factory=list, max_length=50)
    target_lang: str = "zh-CN"
    source_lang: str = "auto"
    service_id: str = ""  # empty: the default service
    title: str = Field("", max_length=1000)  # page title, context for LLM engines (cut to 200 characters there)


class WebTranslateResponse(BaseModel):
    # Aligned with texts; None keeps the original (blank, already in the target language, or translated unchanged)
    translations: list[str | None]
    detected_lang: str


MAX_WEB_TEXT = 10000


@app.post("/api/web/translate")
async def web_translate(data: WebTranslateRequest) -> WebTranslateResponse:
    if any(len(t) > MAX_WEB_TEXT for t in data.texts):
        raise HTTPException(400, f"单段文字太长（上限 {MAX_WEB_TEXT} 字）")
    if len(data.html) > len(data.texts):
        raise HTTPException(400, "html 比 texts 长")
    service = store.get(data.service_id or store.effective_default())
    if not service or not service.get("enabled"):
        raise HTTPException(400, "翻译服务不存在或未启用")
    try:
        translator = create_translator(service, data.target_lang, data.source_lang)
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e
    # The document glossary costs one more request per call, too slow for a page being read; the user's glossary
    # still applies
    if hasattr(translator, "auto_glossary"):
        translator.auto_glossary = False
    runner = Runner(translator)
    runner.set_title(data.title)
    try:
        results = await runner.translate_all(data.texts, html=data.html)
    except TranslatorError as e:
        raise HTTPException(502, str(e)) from e
    finally:
        await translator.aclose()
    translations: list[str | None] = [dst if dst != src else None for src, dst in zip(data.texts, results)]
    return WebTranslateResponse(translations=translations, detected_lang=str(runner.info["detected_lang"]))


# ---------- Translation jobs / translation records ----------


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
        raise HTTPException(400, f"不支持的格式 {ext}，支持的格式：{', '.join(sorted(SUPPORTED))}")
    service = store.get(service_id or store.effective_default())
    if not service:
        raise HTTPException(400, "翻译服务不存在")
    # Check the configuration first, so an unusable service is not discovered only after the upload
    try:
        await create_translator(service, target_lang, source_lang).aclose()
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e

    job = Job(
        id=uuid.uuid4().hex[:12],
        filename=name,
        source_lang=source_lang,
        target_lang=target_lang,
        mode="translated" if mode == "translated" else "bilingual",
        service_id=service["id"],
        service_name=service.get("name", ""),
        preview=ext in PREVIEW_FORMATS,
    )
    jobs.out_dir(job.id).mkdir(parents=True)
    try:
        size = 0
        # Compute the digest while writing: cache ownership, layout analysis results and source file
        # deduplication all identify the same book by it
        digest = hashlib.sha1()
        with jobs.source_path(job).open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "文件太大（上限 200MB）")
                digest.update(chunk)
                # Write large files in a thread, so the event loop is not blocked
                await asyncio.to_thread(f.write, chunk)
    except BaseException:
        # Upload failed midway (read error, size exceeded, etc.): remove the orphan directory, then re-raise
        shutil.rmtree(jobs.dir(job.id), ignore_errors=True)
        raise
    job.size, job.document = size, digest.hexdigest()
    await asyncio.to_thread(jobs.share_source, job)
    jobs.add(job)
    _start(job, _make_translator(job, service))
    return job.to_dict()


@app.get("/api/jobs")
async def list_jobs():
    # Walking all job directories to compute sizes runs in a thread, so the event loop is not blocked
    usage = await asyncio.to_thread(jobs.disk_usage)
    return {"jobs": [j.to_dict() for j in jobs.list()], "disk_usage": usage, "retention_days": settings.retention_days}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).to_dict()


def _service_translator(job: Job):
    """Create an engine from the current configuration of the record's service, only to compute cache keys,
    without sending requests. Return None when the service was deleted or its configuration is invalid."""
    service = store.get(job.service_id)
    if not service:
        return None
    try:
        return create_translator(service, job.target_lang, job.source_lang)
    except TranslatorError:
        return None


async def _forget_cache(job_id: str) -> None:
    """Clear this file's cache before deleting a record (when no other record uses it). A failure to clear does
    not affect deletion; the entries are pushed out when the cache fills up."""
    job = jobs.get(job_id)
    if job is None:
        return
    translator = _service_translator(job)
    try:
        await asyncio.to_thread(cleanup.forget, job, jobs, get_cache(), translator)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        if translator is not None:
            await translator.aclose()


async def _delete(job_id: str) -> None:
    """Delete a record together with the source file, translations, preview and this file's cache. A job being
    translated is stopped first; one generating the other version is deleted after it finishes writing."""
    await _cancel(job_id)
    task = variant_tasks.get(job_id)
    if task is not None and task is not asyncio.current_task() and not task.done():
        await asyncio.gather(task, return_exceptions=True)
    await _forget_cache(job_id)
    jobs.delete(job_id)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    _job_or_404(job_id)
    await _delete(job_id)
    return {"ok": True}


@app.post("/api/jobs/delete")
async def delete_jobs(data: dict = Body(...)):
    """Delete the selected records in bulk. Ids that no longer exist are returned in missing and do not affect
    the others."""
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


def _check_idle(job: Job) -> None:
    """Before clearing the cache, make sure the record is not being translated, generating the other version, or
    being cleaned."""
    if job.status in ACTIVE:
        raise HTTPException(409, "该任务正在翻译，请先暂停")
    if job.id in variant_tasks:
        raise HTTPException(409, "正在生成另一种版本，请稍候")
    if job.id in cleaning:
        raise HTTPException(409, "该记录正在清理，请稍后重试")


@app.get("/api/jobs/{job_id}/cache")
async def get_job_cache(job_id: str):
    """List what clearing would remove: number of cache entries, number of translation files, size of generated
    files and layout recognition results, and number of records sharing the same file."""
    job = _job_or_404(job_id)
    translator = _service_translator(job)
    try:
        found = await asyncio.to_thread(cleanup.footprint, job, jobs, get_cache(), translator)
    finally:
        if translator is not None:
            await translator.aclose()
    return found.summary(len(job.outputs))


async def _clear(job: Job, boxes: bool = False) -> int:
    """Clear what a record left in the caches (see cleanup.purge) and return the number of cache entries deleted.
    The record is checked to be idle first; there is no await between registering and checking, so resuming
    translation or generating the other version cannot get in."""
    _check_idle(job)
    cleaning.add(job.id)
    translator = _service_translator(job)
    try:
        return await asyncio.to_thread(cleanup.purge, job, jobs, get_cache(), translator, boxes)
    finally:
        cleaning.discard(job.id)
        if translator is not None:
            await translator.aclose()


@app.delete("/api/jobs/{job_id}/cache")
async def clear_job_cache(job_id: str, boxes: bool = False):
    """Clear what a record left in the caches; the record returns to untranslated. With boxes=true the layout
    model's recognition results are cleared too."""
    job = _job_or_404(job_id)
    removed = await _clear(job, boxes)
    if jobs.get(job.id) is None:
        raise HTTPException(404, "翻译记录不存在")
    job.clear_results()
    jobs.save(job)
    return {**job.to_dict(), "removed": removed}


@app.post("/api/jobs/pause-all")
async def pause_all_jobs():
    """Pause all jobs that are translating or queued."""
    targets = [j for j in jobs.list() if j.status in ACTIVE]
    results = await asyncio.gather(*(_pause(j) for j in targets))
    return {"paused": sum(results)}


@app.post("/api/jobs/resume-all")
async def resume_all_jobs():
    """Resume all paused and interrupted jobs, each with its original translation service (the model currently
    selected in that service)."""
    resumed, failed = 0, []
    for job in jobs.list():
        if job.status not in ("paused", "interrupted"):
            continue
        try:
            await _resume(job)
            resumed += 1
        except HTTPException as e:
            failed.append({"id": job.id, "filename": job.filename, "error": e.detail})
    return {"resumed": resumed, "failed": failed}


@app.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause: stop sending requests at once. Translated paragraphs are in the cache and are reused on resume."""
    job = _job_or_404(job_id)
    if not await _pause(job):
        raise HTTPException(400, "该任务未在翻译")
    # The record may be deleted concurrently during the pause; do not return a stale result
    if jobs.get(job_id) is None:
        raise HTTPException(404, "翻译记录不存在")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str, data: dict | None = Body(None)):
    """Resume translation (after pause, interruption or failure), or retranslate a finished record. Translated
    paragraphs are in the cache and are not requested again.

    The service's current configuration is used: after the model is changed in the service, the remaining
    paragraphs use the new model.
    service_id may be passed to switch services; with a different service id, translation is redone with the new
    service.
    With fresh=true, what this record left in the caches is cleared first, then translation starts over with every
    paragraph requested again.
    Must be async: _start creates a background task and has to be called in the event loop.
    """
    data = data or {}
    job = await _resume(_job_or_404(job_id), data.get("service_id"), fresh=bool(data.get("fresh")))
    return job.to_dict()


def _variant_service(job: Job, service: dict) -> dict:
    """Match the cache with the model used for the original translation. The model currently selected in the
    service is not changed."""
    if not job.model:
        return service
    pinned = dict(service)
    pinned["model"] = job.model
    return pinned


async def _cut_from_bilingual(job: Job, stage: Path) -> list[Path] | None:
    """When a PDF already has a bilingual version, take the translated-only version from it without laying out
    again. Return None when that is not possible."""
    src = jobs.source_path(job)
    made = next((name for name in job.outputs if output_variant(name) == "bilingual"), None)
    if src.suffix.lower() != ".pdf" or made is None or not (jobs.out_dir(job.id) / made).is_file():
        return None
    try:
        async with mupdf_turn():
            data = await asyncio.to_thread(
                translated_from_bilingual,
                src,
                jobs.out_dir(job.id) / made,
                jobs.load_preview(job),
            )
    except (RuntimeError, ValueError):  # The bilingual version cannot be read; lay out again
        return None
    if data is None:
        return None
    dst = stage / f"{src.stem}.translated.pdf"
    await asyncio.to_thread(dst.write_bytes, data)
    return [dst]


@app.post("/api/jobs/{job_id}/variant")
async def build_variant(job_id: str):
    """Generate the other version (bilingual <-> translated-only). The translated-only version of a PDF is taken
    from the bilingual version; everything else is laid out again from the cache of that run, without requests to
    the translation service."""
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(400, "翻译完成后才能生成另一种版本")
    bilingual = job.mode != "bilingual"
    want = "bilingual" if bilingual else "translated"
    if any(output_variant(name) == want for name in job.outputs):
        return job.to_dict()  # Already generated
    if job.id in cleaning:
        raise HTTPException(409, "该记录正在清理，请稍后重试")
    if job.id in variant_tasks:
        raise HTTPException(409, "正在生成另一种版本，请稍候")
    task = asyncio.current_task()
    if task is None:
        raise HTTPException(500, "无法生成另一种版本")
    # Registration must happen before the next await; otherwise two requests could both pass the checks above
    variant_tasks[job.id] = task
    translator = None
    stage = jobs.dir(job.id) / ".variant"
    try:
        if not jobs.source_path(job).exists():
            raise HTTPException(400, "原文件已不存在，无法生成")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        outputs = await _cut_from_bilingual(job, stage) if want == "translated" else None
        if outputs is None:
            service = store.get(job.service_id)
            if not service:
                raise HTTPException(400, "原翻译服务已不存在，无法生成")
            translator = _make_translator(job, _variant_service(job, service))
            try:
                outputs = await translate_file(
                    jobs.source_path(job),
                    stage,
                    Runner(translator, cache_only=True),
                    bilingual,
                    job.target_lang,
                    layout_model=_model_path("layout"),
                    plan_path=jobs.plan_path(job),
                    formula_model=_formula_model(),
                )
            except (TranslatorError, ValueError) as e:
                raise HTTPException(400, str(e)) from e
        current = jobs.get(job.id)
        if current is None or not jobs.dir(job.id).exists():
            raise HTTPException(404, "翻译记录不存在")
        out_dir = jobs.out_dir(job.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write to a staging directory first and update the record after moving into out/, so a half-written
        # result is never read
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
    """Preview during translation. since=-1 returns all paragraphs and finished translations; later calls fetch
    increments with the returned version."""
    job = _job_or_404(job_id)
    # This endpoint runs in the thread pool, and the event loop sets job.runner to None when the job ends, so take
    # it into a local variable first
    runner = job.runner
    if runner is not None:
        return {**runner.preview(since), "status": job.status}
    # Finished jobs (including those from before a restart) are read from the snapshot
    snap = jobs.load_preview(job)
    if snap is None:
        return {"ready": False, "version": 0, "status": job.status}
    if since >= snap["version"]:
        return {"ready": True, "version": snap["version"], "updates": [], "status": job.status}
    if since >= 0:
        # Paragraphs translated between the last fetch and the end of the job are only in the snapshot. The
        # snapshot has no versions, so it is returned whole, and the frontend skips paragraphs already shown
        return {"ready": True, "version": snap["version"], "updates": snap["updates"], "status": job.status}
    return {**snap, "status": job.status}


@app.post("/api/jobs/{job_id}/focus")
def job_focus(job_id: str, data: dict = Body(...)):
    """The paragraph the user jumped to in the preview; translation then prioritizes the area around it."""
    job = _job_or_404(job_id)
    try:
        index = int(data.get("index", 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "index 必须是整数") from None
    runner = job.runner
    if runner is None:
        # The job has ended (Runner released): focusing is meaningless and counts as success; a job not yet
        # started still fails
        return {"ok": job.status not in ACTIVE}
    runner.focus(index)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str, inline: bool = False):
    """With inline=1 the browser opens the file directly (a PDF in its built-in viewer) instead of downloading
    it."""
    job = _job_or_404(job_id)
    # job.json may have been tampered with; reject file names containing path separators
    if Path(name).name != name:
        raise HTTPException(404, "文件不存在")
    if name in job.outputs:
        path = jobs.out_dir(job.id) / name
        if path.exists():
            return FileResponse(path, filename=name, content_disposition_type="inline" if inline else "attachment")
    raise HTTPException(404, "文件不存在")


# Pages of the book are served to the browser straight from the zip. Scripts in the book never run, so they cannot
# call the APIs under this site's origin; same origin is kept, so fonts and styles in the book load as usual and the
# preview page can follow the chapter the reader clicks into
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
    """Online EPUB preview: title, chapters ordered by spine, and table of contents. Chapter content is fetched
    by member name from the endpoint below."""
    path = _epub_output(job_id, name)
    try:
        with zipfile.ZipFile(path) as zf:
            return book_outline(zf)
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise HTTPException(422, f"无法读取 EPUB：{e}") from e


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


# ---------- Settings ----------


@app.get("/api/settings")
def get_settings():
    return settings.public()


@app.put("/api/settings")
async def update_settings(data: dict = Body(...)):
    try:
        result = settings.update(data)
    except SettingsError as e:
        raise HTTPException(400, str(e)) from e
    await _cleanup()  # The retention period may have been shortened; clean up at once
    return result


# ---------- Layout model, formula detection model ----------
# Both models have the same API: /api/layout-model, /api/formula-model
MODEL_NAMES = {"layout": "版面模型", "formula": "公式检测模型"}


def _spec(kind: str) -> layout_model.ModelSpec:
    if kind not in layout_model.MODELS:
        raise HTTPException(404, "未知的模型")
    return layout_model.MODELS[kind]


def _model_path(kind: str) -> str:
    """Model file used for PDF translation (chosen by precision, see _model_file). An empty string when disabled;
    only the rules are used then."""
    spec = layout_model.MODELS[kind]
    conf = settings.model(spec.key)
    return _model_file(str(spec.resolve(conf["path"])), conf["precision"]) if conf["enabled"] else ""


def _formula_model() -> FormulaModel | None:
    path = _model_path("formula")
    conf = settings.formula_model
    return FormulaModel(path, conf["size"], conf["workers"], conf["threads"]) if path else None


def _model_file(path: str, precision: str) -> str:
    """Use the int8 file when int8 is selected and the int8 file has been generated, otherwise the fp32 file."""
    int8 = quantize.int8_path(Path(path))
    return str(int8) if precision == "int8" and int8.is_file() else path


def _model_status(kind: str) -> dict:
    spec = _spec(kind)
    conf = settings.model(spec.key)
    path = spec.resolve(conf["path"])
    int8 = quantize.int8_path(path)
    extra = {
        "int8": {
            "path": str(int8),
            "file": layout_model.file_status(int8),
            "available": quantize.available(),
            "install_hint": quantize.INSTALL_HINT,
            "state": dict(quantizers[kind].state),
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
        settings.update(
            {
                spec.key: {
                    k: data[k] for k in ("enabled", "path", "size", "precision", "workers", "threads") if k in data
                }
            }
        )
    except SettingsError as e:
        raise HTTPException(400, str(e)) from e
    return _model_status(kind)


@app.post("/api/{kind}-model/download")
async def download_model(kind: str, data: dict = Body(...)):
    """Download to the path in the settings (empty means the default location). The page polls GET for
    progress."""
    spec = _spec(kind)
    target = spec.resolve(str(data.get("path", settings.model(spec.key)["path"]) or ""))
    try:
        model_downloads[kind].start(str(data.get("source") or ""), target)
    except layout_model.LayoutModelError as e:
        raise HTTPException(400, str(e)) from e
    return _model_status(kind)


@app.post("/api/{kind}-model/test")
async def test_model(kind: str, data: dict | None = Body(None)):
    """Load the model from the path in the form (possibly unsaved) and run it once on a sample page."""
    spec = _spec(kind)
    conf = settings.model(spec.key)
    raw = (data or {}).get("path", conf["path"])
    size = (data or {}).get("size", conf.get("size"))
    path = spec.resolve(str(raw or ""))
    # Test the file that translation would use
    path = Path(_model_file(str(path), (data or {}).get("precision", conf["precision"])))
    try:
        async with mupdf_turn():  # The sample page is rendered with MuPDF, taking turns with running jobs
            return await asyncio.to_thread(layout_model.check, spec, path, size)
    except layout_model.LayoutModelError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/{kind}-model/quantize")
async def quantize_model(kind: str):
    """Generate the int8 version of the model, calibrated on pages sampled from translated PDFs. The page polls
    GET for progress."""
    spec = _spec(kind)
    if not quantize.available():
        raise HTTPException(
            400, f"生成 int8 模型需要先安装 onnx：请在项目目录运行 {quantize.INSTALL_HINT}，安装后重启服务"
        )
    src = spec.resolve(settings.model(spec.key)["path"])
    if not src.is_file():
        raise HTTPException(400, f"找不到{MODEL_NAMES[kind]}文件：{src}，请先下载模型")
    # Generate only one at a time: calibration keeps the intermediate results of every page in memory, and
    # generating two together would take twice as much
    if any(quantizer.running for quantizer in quantizers.values()):
        raise HTTPException(
            400, "int8 模型正在生成" if quantizers[kind].running else "另一个模型的 int8 版本正在生成，请在完成后重试"
        )
    pdfs = [path for job in jobs.list() if (path := jobs.source_path(job)).suffix.lower() == ".pdf" and path.is_file()]
    quant = quantize.SPECS[kind]
    async with mupdf_turn():  # Sampling pages renders with MuPDF, taking turns with running jobs
        picks = await asyncio.to_thread(quantize.pick_pages, pdfs)
        images = await asyncio.to_thread(quantize.calibration_images, picks, layout_model.sample_page, quant.size)
    try:
        quantizers[kind].start(src, quantize.int8_path(src), images, quant)
    except quantize.QuantizeError as e:
        raise HTTPException(400, str(e)) from e
    return _model_status(kind)


class NoCacheStaticFiles(StaticFiles):
    """Keep the browser from using cached frontend files directly: it asks the server with an ETag each time and
    gets 304 when unchanged, or the new file when changed.
    Otherwise, after app.js changes, the browser may keep using the old version (this happened: after a commit it
    still jumped into the reader)."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", NoCacheStaticFiles(directory=ROOT / "static", html=True), name="static")
