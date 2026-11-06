"""Run, pause, resume, delete and clear translation jobs. Shared by the job routes and the server's cleanup."""

import asyncio
import shutil
import time
import traceback
from pathlib import Path

from fastapi import HTTPException

from . import cleanup, state
from .formats import PREVIEW_FORMATS, output_variant, translate_file
from .formats.pdf import mupdf_turn
from .formats.pdf_output import translated_from_bilingual
from .history import Job
from .model_files import _formula_model, _model_path
from .runner import Runner, get_cache
from .translators import TranslatorError, create_translator

PROGRESS_SAVE_INTERVAL = 5
# Directory in the record where a run writes its translations before they replace those in out/
RUN_STAGE = ".run"


def _empty_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)


def _install_outputs(out_dir: Path, outputs: list[Path]) -> list[str]:
    """Replace the translations of the last run in out_dir with the files of this run and return their names. Files
    of the last run this run did not write again (the other version generated earlier) are removed, so out/ holds
    only translations of the same run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [path.name for path in outputs]
    for old in out_dir.iterdir():
        if old.name not in names and old.is_file():
            old.unlink()
    for path in outputs:
        path.replace(out_dir / path.name)
    return names


async def _cleanup_quietly() -> None:
    """Run _cleanup for a request that does something else (an upload, saving the settings). Housekeeping that fails
    is logged and tried again in the hourly round; it must not fail the request."""
    try:
        await _cleanup()
    except Exception:  # noqa: BLE001
        traceback.print_exc()


async def _cleanup() -> None:
    """Delete expired records; move layout analysis results from record directories to the shared directory and
    delete those no record uses; replace duplicate source files with hard links.
    Whether the other version is being generated is checked only in the event loop; that dict is not handed to
    other threads."""
    expired = await asyncio.to_thread(state.jobs.expired_ids, state.settings.retention_days)
    for jid in expired:
        if jid in state.variant_tasks or jid in state.running or jid in state.cleaning:
            continue
        state.cleaning.add(jid)
        # There is no await between registering and checking again, so generating the other version cannot get in
        if jid in state.variant_tasks:
            state.cleaning.discard(jid)
            continue
        try:
            await _forget_cache(jid)
            await asyncio.to_thread(state.jobs.delete, jid)
        finally:
            state.cleaning.discard(jid)
    await asyncio.to_thread(state.jobs.adopt_layouts)
    await asyncio.to_thread(cleanup.prune_layouts, state.jobs)
    await asyncio.to_thread(state.jobs.share_sources)


async def _start_job(job: Job):
    """Translate a job taken from the queue. The engine is created now from the service's current configuration, so
    changes made while the job waited apply; a deleted service or an invalid configuration fails the job."""
    service = state.store.get(job.service_id)
    try:
        if not service:
            raise TranslatorError("原翻译服务已不存在，请选择其他服务")
        translator = create_translator(_with_glossaries(service, job.glossary_text), job.target_lang, job.source_lang)
    except TranslatorError as e:
        job.status, job.error, job.finished = "error", str(e), time.time()
        try:
            state.jobs.save(job)
        finally:
            # The place is freed even when the record cannot be written, or the queue would stop for good
            state.running.pop(job.id, None)
            _dispatch()
        return
    job.model = getattr(translator, "model", "")
    await _run_job(job, translator, job.mode == "bilingual")


async def _run_job(job: Job, translator, bilingual: bool):
    last_save = 0.0

    def progress(done, total):
        nonlocal last_save
        job.done, job.total = done, total
        # Save progress every few seconds, so the record still shows how far translation got after the server
        # exits unexpectedly
        if time.time() - last_save > PROGRESS_SAVE_INTERVAL:
            last_save = time.time()
            state.jobs.save(job)

    # Files are written here and replace the translations of the last run only when this run succeeds
    stage = state.jobs.dir(job.id) / RUN_STAGE
    try:
        # Inside the try: when the record cannot be written at start (a full disk), the job fails and frees its place
        # below instead of staying "running" for good
        job.status = "running"
        job.runner = Runner(translator, progress)
        state.jobs.save(job)
        await asyncio.to_thread(_empty_dir, stage)
        outputs = await translate_file(
            state.jobs.source_path(job),
            stage,
            job.runner,
            bilingual,
            job.target_lang,
            layout_model=_model_path("layout"),
            plan_path=state.jobs.plan_path(job),
            formula_model=_formula_model(),
        )
        # No await between moving the files and recording them: a cancellation cannot leave the two out of step
        job.outputs = _install_outputs(state.jobs.out_dir(job.id), outputs)
        job.status = "done"
    except asyncio.CancelledError:
        if job.id in state.pause_requested:
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
        try:
            job.finished = time.time()
            # Accumulate the token usage of this run (resume/retry adds to the earlier count)
            usage = getattr(translator, "usage", None)
            if usage:
                job.prompt_tokens += usage.get("prompt", 0)
                job.completion_tokens += usage.get("completion", 0)
            # The snapshot may be large, so it is written in a thread
            await asyncio.to_thread(state.jobs.save_preview, job)
            job.sync_info()
            if job.runner is not None:
                job.document = job.runner.document or job.document
            # Release the memory held by the Runner; the preview API then reads from the preview.json snapshot
            job.runner = None
            state.jobs.save(job)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            # The place is freed even when writing the record fails (a full disk), or the queue would stop for good
            state.running.pop(job.id, None)
            state.pause_requested.discard(job.id)
            # The next job in the queue takes the place. A paused or deleted job starts it in _pause or _delete;
            # one interrupted by server shutdown does not
            if job.status in ("done", "error"):
                _dispatch()
            await translator.aclose()


def _with_glossaries(service: dict, text: str) -> dict:
    """The service with entries of the job's glossaries added after its own glossary, which takes precedence. They
    enter the cache key like the service's own entries."""
    if not text:
        return service
    return {**service, "glossary": "\n".join(part for part in (service.get("glossary") or "", text) if part)}


def _make_translator(job: Job, service: dict, glossary_text: str | None = None):
    """Create the translation engine from the service's current configuration. Raise 400 for an invalid
    configuration; job status and files have not been touched at that point.
    Without glossary_text the glossary entries of the latest run are used, so the cache keys match the
    translations made."""
    text = job.glossary_text if glossary_text is None else glossary_text
    try:
        return create_translator(_with_glossaries(service, text), job.target_lang, job.source_lang)
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e


def _start(job: Job):
    """Start (or resume) translation: the job joins the end of the queue and starts once fewer than max_jobs jobs
    are translating. The engine is created from the service's configuration when it starts (see _start_job), so
    resuming uses the model now selected in the service."""
    state.waiting[job.id] = job
    _dispatch()


def _number_queue() -> None:
    for position, job in enumerate(state.waiting.values(), 1):
        job.queue_position = position


def _dispatch() -> None:
    """Start the jobs at the front of the queue while fewer than max_jobs jobs are translating."""
    while state.waiting and len(state.running) < state.settings.max_jobs:
        job = state.waiting.pop(next(iter(state.waiting)))
        job.queue_position = 0
        state.running[job.id] = asyncio.create_task(_start_job(job))
    _number_queue()


def _prioritize(job: Job) -> None:
    """Move a waiting job to the front of the queue: it starts at the next free place. Jobs already translating go
    on."""
    if state.waiting.pop(job.id, None) is None:
        raise HTTPException(400, "该任务不在排队中")
    state.waiting = {job.id: job, **state.waiting}
    _number_queue()


async def _cancel(job_id: str):
    """Cancel a running translation job and return once it has actually stopped (finished writing its record). A
    job still in the queue only leaves it."""
    waiting = state.waiting.pop(job_id, None)
    if waiting is not None:
        waiting.queue_position = 0
        _number_queue()
        return
    task = state.running.get(job_id)
    if task and not task.done():
        task.cancel()
        # Swallow only the cancellation and exceptions of the awaited task; cancellation of the current coroutine
        # itself still propagates
        await asyncio.gather(task, return_exceptions=True)
    # The job keeps its place until it has stopped (a PDF finishes its current page first), so the queue does not
    # start another job meanwhile. A task cancelled before it started never freed its place itself
    state.running.pop(job_id, None)


async def _pause(job: Job) -> bool:
    if job.status not in state.ACTIVE:
        return False
    state.pause_requested.add(job.id)
    await _cancel(job.id)
    # A job still queued and not started never enters _run_job when cancelled, so its status is changed here
    if job.status in state.ACTIVE:
        job.status, job.error, job.finished = "paused", "", time.time()
        state.jobs.save(job)
    state.pause_requested.discard(job.id)
    _dispatch()
    return True


def _check_resumable(job: Job) -> None:
    """A record can start translating again only when it still exists and nothing else is working on it."""
    if state.jobs.get(job.id) is None:
        raise HTTPException(404, "翻译记录不存在")
    if job.status in state.ACTIVE:
        raise HTTPException(400, "该任务正在翻译")
    if job.id in state.cleaning:
        raise HTTPException(409, "该记录正在清理，请稍后重试")
    if job.id in state.variant_tasks:
        raise HTTPException(409, "正在生成另一种版本，请稍候")


async def _resume(job: Job, service_id: str | None = None, fresh: bool = False) -> Job:
    """Resume (or redo) translation of the same file. Translated paragraphs are in the cache and are not requested
    again; with fresh true, what this record left in the caches is cleared first (see _clear), and everything starts
    over from layout analysis with all paragraphs requested again."""
    _check_resumable(job)
    if not state.jobs.source_path(job).exists():
        raise HTTPException(400, "原文件已不存在，无法重新翻译")
    service = state.store.get(service_id or job.service_id)
    if not service:
        raise HTTPException(400, "原翻译服务已不存在，请选择其他服务")
    # Validate the configuration first and reset status only after it passes. The glossaries are read again, so
    # edited entries take effect. The engine made here only checks; the job creates its own when it starts
    glossary_text = state.glossaries.entries_text(job.glossary_ids)
    checked = _make_translator(job, service, glossary_text)
    await checked.aclose()
    # The awaits above let other requests run: a second retry, generating the other version, or deleting the record
    _check_resumable(job)
    if fresh:
        await _clear(job)
        job.clear_results()
        _check_resumable(job)
    job.service_id, job.service_name = service["id"], service.get("name", "")
    # Shown while the job waits; set again when it starts, from the service's configuration then
    job.model = getattr(checked, "model", "")
    job.glossary_text = glossary_text
    # The translations of the last run stay in out/ and in job.outputs until this run succeeds (see _run_job), so a
    # retry that fails or is paused does not lose them
    job.status, job.error, job.done, job.total, job.finished = "queued", "", 0, 0, 0.0
    # Recompute the preview flag: old records may have been created before the preview feature (stored as False)
    job.preview = Path(job.filename).suffix.lower() in PREVIEW_FORMATS
    (state.jobs.dir(job.id) / "preview.json").unlink(missing_ok=True)
    _start(job)
    state.jobs.save(job)
    return job


def _service_translator(job: Job):
    """Create an engine from the current configuration of the record's service, only to compute cache keys,
    without sending requests. Return None when the service was deleted or its configuration is invalid."""
    service = state.store.get(job.service_id)
    if not service:
        return None
    try:
        return create_translator(_with_glossaries(service, job.glossary_text), job.target_lang, job.source_lang)
    except TranslatorError:
        return None


async def _forget_cache(job_id: str) -> None:
    """Clear this file's cache before deleting a record (when no other record uses it). A failure to clear does
    not affect deletion; the entries are pushed out when the cache fills up."""
    job = state.jobs.get(job_id)
    if job is None:
        return
    translator = _service_translator(job)
    try:
        await asyncio.to_thread(cleanup.forget, job, state.jobs, get_cache(), translator)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        if translator is not None:
            await translator.aclose()


async def _delete(job_id: str) -> None:
    """Delete a record together with the source file, translations, preview and this file's cache. A job being
    translated is stopped first; one generating the other version is deleted after it finishes writing."""
    await _cancel(job_id)
    task = state.variant_tasks.get(job_id)
    if task is not None and task is not asyncio.current_task() and not task.done():
        await asyncio.gather(task, return_exceptions=True)
    await _forget_cache(job_id)
    state.jobs.delete(job_id)
    _dispatch()


def _check_idle(job: Job) -> None:
    """Before clearing the cache, make sure the record is not being translated, generating the other version, or
    being cleaned."""
    if job.status in state.ACTIVE:
        raise HTTPException(409, "该任务正在翻译，请先暂停")
    if job.id in state.variant_tasks:
        raise HTTPException(409, "正在生成另一种版本，请稍候")
    if job.id in state.cleaning:
        raise HTTPException(409, "该记录正在清理，请稍后重试")


async def _clear(job: Job, boxes: bool = False) -> int:
    """Clear what a record left in the caches (see cleanup.purge) and return the number of cache entries deleted.
    The record is checked to be idle first; there is no await between registering and checking, so resuming
    translation or generating the other version cannot get in."""
    _check_idle(job)
    state.cleaning.add(job.id)
    translator = _service_translator(job)
    try:
        return await asyncio.to_thread(cleanup.purge, job, state.jobs, get_cache(), translator, boxes)
    finally:
        state.cleaning.discard(job.id)
        if translator is not None:
            await translator.aclose()


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
    src = state.jobs.source_path(job)
    made = next((name for name in job.outputs if output_variant(name) == "bilingual"), None)
    if src.suffix.lower() != ".pdf" or made is None or not (state.jobs.out_dir(job.id) / made).is_file():
        return None
    try:
        async with mupdf_turn():
            data = await asyncio.to_thread(
                translated_from_bilingual,
                src,
                state.jobs.out_dir(job.id) / made,
                state.jobs.load_preview(job),
            )
    except (RuntimeError, ValueError):  # The bilingual version cannot be read; lay out again
        return None
    if data is None:
        return None
    dst = stage / f"{src.stem}.translated.pdf"
    await asyncio.to_thread(dst.write_bytes, data)
    return [dst]
