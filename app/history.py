"""Translation records: one directory data/jobs/<id>/ per job, with metadata in job.json and the source file and
translations in the same directory.

- Records and files are deleted only when the user deletes them from the list or when they exceed the retention
  period (settings.retention_days).
- After a server restart, records are restored from disk; jobs unfinished before the restart are marked "已中断"
  and can be translated again (paragraphs already translated are in the cache).
- Preview: a running job is read live from its Runner; a finished job stores a preview snapshot as preview.json,
  which can be opened at any time.
"""

from __future__ import annotations

import builtins
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from .formats import document_digest
from .json_types import JSONDict
from .langdetect import same_language
from .paths import DATA_DIR
from .request_log import LOG_NAME, RequestStats
from .usage import Usage

if TYPE_CHECKING:
    from .runner import Runner


def _write_atomic(path: Path, text: str) -> None:
    """Write a temporary file and replace the target, so a half-written file is never read.

    The temporary name differs each time: the translation thread updating progress and an API request may save
    the same record at once, and with a shared job.tmp the one replacing first would leave the other without its
    file. Nothing is written when the record has just been deleted.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)


JOBS_DIR = DATA_DIR / "jobs"
DAY = 24 * 3600


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued / running / done / error / interrupted / paused / idle (untranslated after cleanup)
    done: int = 0
    total: int = 0
    error: str = ""
    outputs: list[str] = field(default_factory=list)  # translation file names (in the out/ directory)
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    source_lang: str = "auto"
    target_lang: str = ""
    mode: str = "bilingual"
    service_id: str = ""
    service_name: str = ""
    model: str = ""  # model of the latest translation (updated to the service's selected model when continuing)
    size: int = 0
    preview: bool = False
    detected_lang: str = ""
    skipped: int = 0
    failed: int = 0  # paragraphs where the service returned an empty translation and the original was filled in
    # Token usage reported by LLM APIs, accumulated across continues and retries, as in usage.Usage: prompt_tokens
    # counts every input token (cached ones included), completion_tokens every output token (reasoning included).
    # Google and relays that leave usage out report nothing: tokens_reported stays False and the counts 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    tokens_reported: bool = False
    # Digest of the source file's content, used to register cache entries as belonging to this file. Older records
    # lack it and compute it when needed
    document: str = ""
    # Glossaries of the glossary library chosen when the job was submitted, merged after the service's own glossary
    glossary_ids: list[str] = field(default_factory=list)
    # Entries of those glossaries when the latest run started. The other version and cache clearing use them, so
    # editing or deleting a glossary does not change the cache keys of translations already made
    glossary_text: str = ""
    # Summary of the requests sent upstream by all runs (see request_log.py); None before the first one. The
    # requests themselves are in requests.jsonl in the record's directory
    requests: RequestStats | None = None

    # Runtime fields that are not saved
    runner: Runner | None = field(default=None, repr=False, compare=False)
    # Place in the queue while waiting to start, 1 for the next one; 0 otherwise
    queue_position: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        # job.json holds the summary as a plain object
        if not isinstance(self.requests, RequestStats | None):
            self.requests = RequestStats.from_dict(self.requests)
        # Records from before tokens_reported: counts above 0 were reported
        if self.prompt_tokens or self.completion_tokens:
            self.tokens_reported = True

    def add_usage(self, usage: Usage) -> None:
        """Add the token usage of a run."""
        self.prompt_tokens += usage.input
        self.completion_tokens += usage.output
        self.reasoning_tokens += usage.reasoning
        self.cached_tokens += usage.cached_input
        self.cache_write_tokens += usage.cache_write
        self.tokens_reported = self.tokens_reported or usage.reported

    def sync_info(self) -> None:
        """Copy the live detection results from the Runner to the record."""
        if self.runner is not None:
            info = self.runner.info
            self.detected_lang = info.detected_lang or self.detected_lang
            self.skipped = info.skipped or self.skipped
            # 0 is a valid new count (everything succeeded after a retry), so `or` must not treat it as missing
            if info.failed is not None:
                self.failed = info.failed

    def clear_results(self) -> None:
        """Return to untranslated after the cache is cleared: translations, progress, detection results and usage no
        longer match any cache and are all reset."""
        self.status, self.error = "idle", ""
        self.done = self.total = self.skipped = self.failed = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.reasoning_tokens = self.cached_tokens = self.cache_write_tokens = 0
        self.tokens_reported = False
        self.requests = None
        self.outputs, self.finished, self.detected_lang = [], 0.0, ""

    def to_dict(self) -> JSONDict:
        self.sync_info()
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "created": self.created,
            "finished": self.finished,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "mode": self.mode,
            "service_id": self.service_id,
            "service_name": self.service_name,
            "glossary_ids": self.glossary_ids,
            "queue_position": self.queue_position,
            "model": self.model,
            "size": self.size,
            "detected_lang": self.detected_lang,
            "skipped": self.skipped,
            "failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "tokens_reported": self.tokens_reported,
            "requests": asdict(self.requests) if self.requests else None,
            # The extension's sameLangCheck: warn when the detected source language equals the target language
            "same_lang": same_language(self.detected_lang, self.target_lang),
            "preview": self.preview,
            # The preparation stage before translation and the writing stage after it, present only while running
            "stage": dict(self.runner.stage) if self.runner is not None else None,
            # Unencoded # and ? in file names would be taken by browsers as a fragment and a query string
            "outputs": [{"name": n, "url": f"/api/jobs/{self.id}/files/{quote(n)}"} for n in self.outputs],
        }


# Fields saved to disk: all except the runtime ones
_PERSISTED = [f for f in Job.__dataclass_fields__ if f not in ("runner", "queue_position")]


# Layout analysis results of each book are stored under layout/<file digest>/
_LAYOUT_FILES = ("plan.pickle", "boxes.json")


class JobStore:
    def __init__(self, root: Path = JOBS_DIR, layout_root: Path | None = None):
        self.root = root
        # Layout analysis depends only on the source file, the code and the two local models, not on the job's
        # languages, service or mode, so records of the same book share one copy
        self.layout_root = layout_root or root.parent / "layout"
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self._load()

    # ---------- Paths ----------

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def out_dir(self, job_id: str) -> Path:
        return self.dir(job_id) / "out"

    def source_path(self, job: Job) -> Path:
        return self.dir(job.id) / job.filename

    def requests_path(self, job_id: str) -> Path:
        """Log of the requests the record's runs sent upstream (see request_log.py)."""
        return self.dir(job_id) / LOG_NAME

    def document(self, job: Job) -> str:
        """Digest of the record's file. Older records lack it; it is computed and saved while the source file exists,
        and an empty string is returned when the file is gone too."""
        if not job.document:
            source = self.source_path(job)
            if not source.is_file():
                return ""
            job.document = document_digest(source)
            self.save(job)
        return job.document

    def share_source(self, job: Job) -> bool:
        """When another record already stores the same file, replace this record's source file with a hard link to
        it, so a book takes disk space only once. Source files are read-only after upload, and deleting a record
        only removes one link. When the file system does not support hard links, the record keeps its own copy."""
        path = self.source_path(job)
        for other in self.list():
            if other.id == job.id or self.document(other) != job.document:
                continue
            existing = self.source_path(other)
            if not existing.is_file() or existing.stat().st_size != path.stat().st_size:
                continue
            if existing.stat().st_ino == path.stat().st_ino:
                return True
            link = path.with_name(f"{path.name}.{uuid.uuid4().hex}.link")
            try:
                os.link(existing, link)
                link.replace(path)
                return True
            except OSError:
                link.unlink(missing_ok=True)
                return False
        return False

    def share_sources(self) -> int:
        """Replace duplicate source files of existing records (uploaded before files were shared) with hard links and
        return the number of records handled. A record being translated reads an already open file and is not
        affected by replacing the directory entry."""
        return sum(
            self.share_source(job) for job in self.list() if self.source_path(job).is_file() and self.document(job)
        )

    def layout_dir(self, job: Job) -> Path | None:
        """Directory of this book's layout analysis results, shared by records of the same book. Results previously
        stored in a record directory are moved here on first use."""
        doc = self.document(job)
        if not doc:
            return None
        folder = self.layout_root / doc
        folder.mkdir(parents=True, exist_ok=True)
        for name in _LAYOUT_FILES:
            local = self.dir(job.id) / name
            if local.is_file():
                if (folder / name).exists():
                    local.unlink()
                else:
                    local.replace(folder / name)
        return folder

    def adopt_layouts(self) -> int:
        """Move layout analysis results still stored in record directories to the shared directory and return the
        number of records moved. New records of the same book can then find results analyzed for older ones."""
        moved = 0
        for job in self.list():
            if any((self.dir(job.id) / name).is_file() for name in _LAYOUT_FILES) and self.layout_dir(job):
                moved += 1
        return moved

    def plan_path(self, job: Job) -> Path | None:
        """Layout analysis result of the PDF. Read back when continuing a translation, generating another version or
        uploading the same book again, so pages need not be analyzed again. Regions boxed by the models are stored
        next to it in boxes.json. Returns None, and nothing is saved, when the source file is missing."""
        folder = self.layout_dir(job)
        return folder / "plan.pickle" if folder else None

    # ---------- Reading and writing ----------

    def _load(self) -> None:
        if not self.root.exists():
            return
        for meta in self.root.glob("*/job.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                job = Job(**{k: v for k, v in data.items() if k in _PERSISTED})
            except (ValueError, TypeError, OSError):
                continue  # skip a damaged record and leave its directory in place
            self.jobs[job.id] = job
            # Only records that change are written back: rewriting every job.json on load would race with another
            # process using the same directory
            if job.status in ("queued", "running"):
                job.status = "interrupted"
                job.error = "服务重启导致翻译中断。已翻译的段落已缓存，继续翻译时直接复用。"
                self.save(job)

    def save(self, job: Job) -> None:
        job.sync_info()
        path = self.dir(job.id) / "job.json"
        if not path.parent.exists():
            return  # the record has been deleted
        data = {k: getattr(job, k) for k in _PERSISTED}
        data["requests"] = asdict(job.requests) if job.requests else None
        _write_atomic(path, json.dumps(data, ensure_ascii=False, indent=1))

    def add(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
        self.save(job)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(self.dir(job_id), ignore_errors=True)
        return True

    def expired_ids(self, retention_days: int, now: float | None = None) -> builtins.list[str]:
        """Ids of records past the retention period and not being translated. They are not deleted here, since the
        caller must also skip jobs that are regenerating outputs."""
        if retention_days <= 0:
            return []
        now = now or time.time()
        with self.lock:  # runs in the caller's thread; iteration must exclude add/delete
            return [
                j.id
                for j in list(self.jobs.values())
                if j.status not in ("queued", "running") and now - (j.finished or j.created) > retention_days * DAY
            ]

    def cleanup(
        self, retention_days: int, now: float | None = None, busy: frozenset[str] | None = None
    ) -> builtins.list[str]:
        """Delete records past the retention period (except running ones and those in busy). retention_days=0 keeps
        records forever."""
        busy = busy or frozenset()
        expired = [jid for jid in self.expired_ids(retention_days, now) if jid not in busy]
        for jid in expired:
            self.delete(jid)
        return expired

    # ---------- Preview snapshots ----------

    def save_preview(self, job: Job) -> None:
        if job.runner is None or not job.preview:
            return
        data = job.runner.preview(-1)
        if not data.get("ready"):
            return
        path = self.dir(job.id) / "preview.json"
        if path.parent.exists():
            _write_atomic(path, json.dumps(data, ensure_ascii=False))

    def load_preview(self, job: Job) -> JSONDict | None:
        path = self.dir(job.id) / "preview.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def disk_usage(self) -> int:
        """Disk space used by records and layout analysis results. Source files shared by hard links count once."""
        seen: dict[tuple[int, int], int] = {}
        for root in (self.root, self.layout_root):
            if root.exists():
                for p in root.rglob("*"):
                    if p.is_file():
                        st = p.stat()
                        seen[(st.st_dev, st.st_ino)] = st.st_size
        return sum(seen.values())
