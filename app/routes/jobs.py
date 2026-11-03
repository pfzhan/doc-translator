"""Routes of translation jobs (translation records): upload, progress, preview, files and cache clearing."""

import asyncio
import hashlib
import mimetypes
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .. import cleanup, state
from ..formats import PREVIEW_FORMATS, SUPPORTED, html_page, output_variant, translate_file
from ..formats.epub import book_outline
from ..history import Job
from ..job_control import (
    _cleanup,
    _clear,
    _cut_from_bilingual,
    _delete,
    _make_translator,
    _pause,
    _resume,
    _service_translator,
    _start,
    _variant_service,
)
from ..model_files import _formula_model, _model_path
from ..runner import Runner, get_cache
from ..translators import TranslatorError, create_translator

router = APIRouter()

MAX_UPLOAD = 200 * 1024 * 1024


# ---------- Translation jobs / translation records ----------


def _job_or_404(job_id: str) -> Job:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "翻译记录不存在")
    return job


@router.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    target_lang: str = Form("zh-CN"),
    source_lang: str = Form("auto"),
    service_id: str = Form(""),
    mode: str = Form("bilingual"),
    glossary_ids: list[str] = Form([]),
):
    await _cleanup()
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"不支持的格式 {ext}，支持的格式：{', '.join(sorted(SUPPORTED))}")
    service = state.store.get(service_id or state.store.effective_default())
    if not service:
        raise HTTPException(400, "翻译服务不存在")
    # Check the configuration first, so an unusable service is not discovered only after the upload
    try:
        await create_translator(service, target_lang, source_lang).aclose()
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e

    glossary_ids = [gid for gid in dict.fromkeys(glossary_ids) if state.glossaries.get(gid)]
    job = Job(
        id=uuid.uuid4().hex[:12],
        filename=name,
        source_lang=source_lang,
        target_lang=target_lang,
        mode="translated" if mode == "translated" else "bilingual",
        service_id=service["id"],
        service_name=service.get("name", ""),
        glossary_ids=glossary_ids,
        glossary_text=state.glossaries.entries_text(glossary_ids),
        preview=ext in PREVIEW_FORMATS,
    )
    state.jobs.out_dir(job.id).mkdir(parents=True)
    try:
        size = 0
        # Compute the digest while writing: cache ownership, layout analysis results and source file
        # deduplication all identify the same book by it
        digest = hashlib.sha1()
        with state.jobs.source_path(job).open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "文件太大（上限 200MB）")
                digest.update(chunk)
                # Write large files in a thread, so the event loop is not blocked
                await asyncio.to_thread(f.write, chunk)
    except BaseException:
        # Upload failed midway (read error, size exceeded, etc.): remove the orphan directory, then re-raise
        shutil.rmtree(state.jobs.dir(job.id), ignore_errors=True)
        raise
    job.size, job.document = size, digest.hexdigest()
    await asyncio.to_thread(state.jobs.share_source, job)
    state.jobs.add(job)
    _start(job, _make_translator(job, service))
    return job.to_dict()


@router.get("/api/jobs")
async def list_jobs():
    # Walking all job directories to compute sizes runs in a thread, so the event loop is not blocked
    usage = await asyncio.to_thread(state.jobs.disk_usage)
    return {
        "jobs": [j.to_dict() for j in state.jobs.list()],
        "disk_usage": usage,
        "retention_days": state.settings.retention_days,
    }


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).to_dict()


@router.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    _job_or_404(job_id)
    await _delete(job_id)
    return {"ok": True}


@router.post("/api/jobs/delete")
async def delete_jobs(data: dict = Body(...)):
    """Delete the selected records in bulk. Ids that no longer exist are returned in missing and do not affect
    the others."""
    ids = data.get("ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise HTTPException(400, "ids 必须是记录 id 列表")
    deleted, missing = 0, []
    for job_id in dict.fromkeys(ids):
        if state.jobs.get(job_id) is None:
            missing.append(job_id)
            continue
        await _delete(job_id)
        deleted += 1
    return {"deleted": deleted, "missing": missing}


@router.get("/api/jobs/{job_id}/cache")
async def get_job_cache(job_id: str):
    """List what clearing would remove: number of cache entries, number of translation files, size of generated
    files and layout recognition results, and number of records sharing the same file."""
    job = _job_or_404(job_id)
    translator = _service_translator(job)
    try:
        found = await asyncio.to_thread(cleanup.footprint, job, state.jobs, get_cache(), translator)
    finally:
        if translator is not None:
            await translator.aclose()
    return found.summary(len(job.outputs))


@router.delete("/api/jobs/{job_id}/cache")
async def clear_job_cache(job_id: str, boxes: bool = False):
    """Clear what a record left in the caches; the record returns to untranslated. With boxes=true the layout
    model's recognition results are cleared too."""
    job = _job_or_404(job_id)
    removed = await _clear(job, boxes)
    if state.jobs.get(job.id) is None:
        raise HTTPException(404, "翻译记录不存在")
    job.clear_results()
    state.jobs.save(job)
    return {**job.to_dict(), "removed": removed}


class CacheSweep(BaseModel):
    removed: int  # entries cleared, or that would be
    # Entries saved before the cache recorded whose they are: they may belong to a record, so they stay
    untracked: int


def _record_docs() -> set[str]:
    return {doc for job in state.jobs.list() if (doc := state.jobs.document(job))}


@router.get("/api/cache/unused")
async def unused_cache() -> CacheSweep:
    """How many cache entries belong only to web pages and to records deleted earlier."""
    removed, untracked = await asyncio.to_thread(get_cache().unused, await asyncio.to_thread(_record_docs))
    return CacheSweep(removed=removed, untracked=untracked)


@router.delete("/api/cache/unused")
async def clear_unused_cache() -> CacheSweep:
    """Clear the cache entries that belong only to web pages and to records deleted earlier; what a record uses
    stays. Refused while a job is translating, generating the other version or being cleaned: a new record's
    entries are recorded as they are written, but the list of records is taken before."""
    if (
        any(job.status in state.ACTIVE for job in state.jobs.list())
        or state.running
        or state.variant_tasks
        or state.cleaning
    ):
        raise HTTPException(409, "有任务正在翻译或处理，请等它完成或暂停后再清理")
    docs = await asyncio.to_thread(_record_docs)
    removed = await asyncio.to_thread(get_cache().purge_unused, docs)
    _, untracked = await asyncio.to_thread(get_cache().unused, docs)
    return CacheSweep(removed=removed, untracked=untracked)


@router.post("/api/jobs/pause-all")
async def pause_all_jobs():
    """Pause all jobs that are translating or queued."""
    targets = [j for j in state.jobs.list() if j.status in state.ACTIVE]
    results = await asyncio.gather(*(_pause(j) for j in targets))
    return {"paused": sum(results)}


@router.post("/api/jobs/resume-all")
async def resume_all_jobs():
    """Resume all paused and interrupted jobs, each with its original translation service (the model currently
    selected in that service)."""
    resumed, failed = 0, []
    for job in state.jobs.list():
        if job.status not in ("paused", "interrupted"):
            continue
        try:
            await _resume(job)
            resumed += 1
        except HTTPException as e:
            failed.append({"id": job.id, "filename": job.filename, "error": e.detail})
    return {"resumed": resumed, "failed": failed}


@router.post("/api/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause: stop sending requests at once. Translated paragraphs are in the cache and are reused on resume."""
    job = _job_or_404(job_id)
    if not await _pause(job):
        raise HTTPException(400, "该任务未在翻译")
    # The record may be deleted concurrently during the pause; do not return a stale result
    if state.jobs.get(job_id) is None:
        raise HTTPException(404, "翻译记录不存在")
    return job.to_dict()


@router.post("/api/jobs/{job_id}/retry")
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


@router.post("/api/jobs/{job_id}/variant")
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
    if job.id in state.cleaning:
        raise HTTPException(409, "该记录正在清理，请稍后重试")
    if job.id in state.variant_tasks:
        raise HTTPException(409, "正在生成另一种版本，请稍候")
    task = asyncio.current_task()
    if task is None:
        raise HTTPException(500, "无法生成另一种版本")
    # Registration must happen before the next await; otherwise two requests could both pass the checks above
    state.variant_tasks[job.id] = task
    translator = None
    stage = state.jobs.dir(job.id) / ".variant"
    try:
        if not state.jobs.source_path(job).exists():
            raise HTTPException(400, "原文件已不存在，无法生成")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        outputs = await _cut_from_bilingual(job, stage) if want == "translated" else None
        if outputs is None:
            service = state.store.get(job.service_id)
            if not service:
                raise HTTPException(400, "原翻译服务已不存在，无法生成")
            translator = _make_translator(job, _variant_service(job, service))
            try:
                outputs = await translate_file(
                    state.jobs.source_path(job),
                    stage,
                    Runner(translator, cache_only=True),
                    bilingual,
                    job.target_lang,
                    layout_model=_model_path("layout"),
                    plan_path=state.jobs.plan_path(job),
                    formula_model=_formula_model(),
                )
            except (TranslatorError, ValueError) as e:
                raise HTTPException(400, str(e)) from e
        current = state.jobs.get(job.id)
        if current is None or not state.jobs.dir(job.id).exists():
            raise HTTPException(404, "翻译记录不存在")
        out_dir = state.jobs.out_dir(job.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write to a staging directory first and update the record after moving into out/, so a half-written
        # result is never read
        for path in outputs:
            path.replace(out_dir / path.name)
            if path.name not in current.outputs:
                current.outputs.append(path.name)
        state.jobs.save(current)
        if state.jobs.get(job.id) is None:
            raise HTTPException(404, "翻译记录不存在")
        return current.to_dict()
    finally:
        state.variant_tasks.pop(job.id, None)
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if translator is not None:
            await translator.aclose()


@router.get("/api/jobs/{job_id}/preview")
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
    snap = state.jobs.load_preview(job)
    if snap is None:
        return {"ready": False, "version": 0, "status": job.status}
    if since >= snap["version"]:
        return {"ready": True, "version": snap["version"], "updates": [], "status": job.status}
    if since >= 0:
        # Paragraphs translated between the last fetch and the end of the job are only in the snapshot. The
        # snapshot has no versions, so it is returned whole, and the frontend skips paragraphs already shown
        return {"ready": True, "version": snap["version"], "updates": snap["updates"], "status": job.status}
    return {**snap, "status": job.status}


@router.post("/api/jobs/{job_id}/focus")
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
        return {"ok": job.status not in state.ACTIVE}
    runner.focus(index)
    return {"ok": True}


@router.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str, inline: bool = False):
    """With inline=1 the browser opens the file directly (a PDF in its built-in viewer) instead of downloading
    it."""
    job = _job_or_404(job_id)
    # job.json may have been tampered with; reject file names containing path separators
    if Path(name).name != name:
        raise HTTPException(404, "文件不存在")
    if name in job.outputs:
        path = state.jobs.out_dir(job.id) / name
        if path.exists():
            # A translated web page opened with inline=1 would run its scripts under this site's origin
            headers = _BOOK_PAGE_HEADERS if path.suffix.lower() in html_page.SUFFIXES else None
            return FileResponse(
                path,
                filename=name,
                content_disposition_type="inline" if inline else "attachment",
                headers=headers,
            )
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
    path = state.jobs.out_dir(job.id) / name
    if Path(name).name != name or name not in job.outputs or not name.lower().endswith(".epub") or not path.exists():
        raise HTTPException(404, "文件不存在")
    return path


@router.get("/api/jobs/{job_id}/epub/{name}")
def epub_outline(job_id: str, name: str):
    """Online EPUB preview: title, chapters ordered by spine, and table of contents. Chapter content is fetched
    by member name from the endpoint below."""
    path = _epub_output(job_id, name)
    try:
        with zipfile.ZipFile(path) as zf:
            return book_outline(zf)
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise HTTPException(422, f"无法读取 EPUB：{e}") from e


@router.get("/api/jobs/{job_id}/epub/{name}/{member:path}")
def epub_member(job_id: str, name: str, member: str):
    path = _epub_output(job_id, name)
    try:
        with zipfile.ZipFile(path) as zf:
            data = zf.read(member)
    except (zipfile.BadZipFile, KeyError):
        raise HTTPException(404, "文件不存在") from None
    media = mimetypes.guess_type(member)[0] or "application/octet-stream"
    return Response(data, media_type=media, headers=_BOOK_PAGE_HEADERS)
