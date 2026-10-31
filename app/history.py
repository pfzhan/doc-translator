"""翻译记录：每个任务一个目录 data/jobs/<id>/，元数据存在 job.json，原文件和译文在同一个目录里。

- 记录和文件只在两种情况下删除：用户在列表里删掉，或者超过保留时间（settings.retention_days）。
- 服务重启后从磁盘恢复记录；重启前没跑完的任务标记为“已中断”，可以重新翻译（已翻好的段落在缓存里）。
- 预览：运行中的任务从 Runner 实时读取；已完成的任务把预览快照存成 preview.json，之后随时能打开。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from .formats import document_digest
from .langdetect import same_language


def _write_atomic(path: Path, text: str) -> None:
    """先写临时文件再替换，避免写到一半被读到。

    临时文件名每次不同：翻译线程更新进度和接口请求可能同时保存同一条记录，
    共用一个 job.tmp 时，先换过去的那个会让另一个找不到文件。记录刚被删掉时什么也不写。
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
DAY = 24 * 3600


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued / running / done / error / interrupted / paused / idle（清理后未翻译）
    done: int = 0
    total: int = 0
    error: str = ""
    outputs: list[str] = field(default_factory=list)  # 译文文件名（在 out/ 目录下）
    created: float = field(default_factory=time.time)
    finished: float = 0.0
    source_lang: str = "auto"
    target_lang: str = ""
    mode: str = "bilingual"
    service_id: str = ""
    service_name: str = ""
    model: str = ""  # 最近一次翻译用的模型（继续翻译时按服务当前选中的模型更新）
    size: int = 0
    preview: bool = False
    detected_lang: str = ""
    skipped: int = 0
    failed: int = 0  # 翻译服务返回空译文、回填原文的段数
    # 大模型接口返回的 token 用量，跨继续/重试累计；谷歌等接口不提供，保持 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 原文件内容的摘要，缓存条目按它登记归属。旧记录没有，用到时再算
    document: str = ""

    # 不落盘的运行时字段
    runner: object = field(default=None, repr=False, compare=False)

    def sync_info(self):
        """把 Runner 里实时的检测结果同步到记录上。"""
        if self.runner is not None:
            info = self.runner.info
            self.detected_lang = info.get("detected_lang", "") or self.detected_lang
            self.skipped = info.get("skipped", 0) or self.skipped
            # 0 是有效的新计数（重试后全部成功），不能用 or 把 0 当成缺省
            if "failed" in info:
                self.failed = int(info["failed"])

    def clear_results(self):
        """清理缓存后回到未翻译：译文、进度、检测结果和用量都不再对应任何缓存，一并清零。"""
        self.status, self.error = "idle", ""
        self.done = self.total = self.skipped = self.failed = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.outputs, self.finished, self.detected_lang = [], 0.0, ""

    def to_dict(self) -> dict:
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
            "model": self.model,
            "size": self.size,
            "detected_lang": self.detected_lang,
            "skipped": self.skipped,
            "failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            # 插件的 sameLangCheck：检测到的源语言和目标语言一致时提示
            "same_lang": same_language(self.detected_lang, self.target_lang),
            "preview": self.preview,
            # 翻译前的准备、翻译后的写回阶段，只在运行中有
            "stage": dict(self.runner.stage) if self.runner is not None else None,
            # 文件名里的 #、? 不编码会被浏览器当成锚点和查询串
            "outputs": [{"name": n, "url": f"/api/jobs/{self.id}/files/{quote(n)}"} for n in self.outputs],
        }


# 落盘的字段：除了运行时的 runner 以外全部
_PERSISTED = [f for f in Job.__dataclass_fields__ if f != "runner"]


# 每本书的版面分析结果放在 layout/<文件摘要>/ 下
_LAYOUT_FILES = ("plan.pickle", "boxes.json")


class JobStore:
    def __init__(self, root: Path = JOBS_DIR, layout_root: Path | None = None):
        self.root = root
        # 版面分析只看原文件、代码和两个本地模型，和任务的语言、服务、模式无关，同一本书的记录共用一份
        self.layout_root = layout_root or root.parent / "layout"
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self._load()

    # ---------- 路径 ----------

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def out_dir(self, job_id: str) -> Path:
        return self.dir(job_id) / "out"

    def source_path(self, job: Job) -> Path:
        return self.dir(job.id) / job.filename

    def document(self, job: Job) -> str:
        """记录对应文件的摘要。旧记录没存，原文件还在就算一次存下；原文件也没了返回空串。"""
        if not job.document:
            source = self.source_path(job)
            if not source.is_file():
                return ""
            job.document = document_digest(source)
            self.save(job)
        return job.document

    def share_source(self, job: Job) -> bool:
        """同一份文件已经有记录存着时，把这条记录的原文件换成指向它的硬链接，同一本书在磁盘上只占一份。
        原文件上传后只读不写，删掉一条记录只是少一个链接。文件系统不支持硬链接时保留自己的副本。"""
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
        """把已有记录里重复的原文件都换成硬链接（共用原文件之前上传的），返回处理的记录数。
        正在翻译的记录读的是已经打开的文件，换掉目录项不影响它。"""
        return sum(self.share_source(job) for job in self.list() if self.source_path(job).is_file() and self.document(job))

    def layout_dir(self, job: Job) -> Path | None:
        """这本书的版面分析结果所在的目录，同一本书的记录共用。以前存在记录目录里的，第一次用到时搬过来。"""
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
        """把还存在记录目录里的版面分析结果都搬到共用目录，返回搬过的记录数。
        同一本书的新记录这样才找得到旧记录分析过的结果。"""
        moved = 0
        for job in self.list():
            if any((self.dir(job.id) / name).is_file() for name in _LAYOUT_FILES) and self.layout_dir(job):
                moved += 1
        return moved

    def plan_path(self, job: Job) -> Path | None:
        """PDF 的版面分析结果。继续翻译、生成另一种版本、同一本书再传一次时读回，不用再逐页分析。
        模型框出的区域存在旁边的 boxes.json 里。原文件不在时返回 None，不存。"""
        folder = self.layout_dir(job)
        return folder / "plan.pickle" if folder else None

    # ---------- 读写 ----------

    def _load(self):
        if not self.root.exists():
            return
        for meta in self.root.glob("*/job.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                job = Job(**{k: v for k, v in data.items() if k in _PERSISTED})
            except (ValueError, TypeError, OSError):
                continue  # 损坏的记录跳过，目录保留不动
            if job.status in ("queued", "running"):
                job.status = "interrupted"
                job.error = "服务重启导致翻译中断。已翻译的段落已缓存，继续翻译时直接复用。"
            self.jobs[job.id] = job
        for job in self.jobs.values():
            self.save(job)

    def save(self, job: Job):
        job.sync_info()
        path = self.dir(job.id) / "job.json"
        if not path.parent.exists():
            return  # 记录已删除
        _write_atomic(path, json.dumps({k: getattr(job, k) for k in _PERSISTED}, ensure_ascii=False, indent=1))

    def add(self, job: Job):
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

    def expired_ids(self, retention_days: int, now: float | None = None) -> list[str]:
        """超过保留时间、且不在翻译中的记录 id。不在这里删，调用方还要避开正在补生成的任务。"""
        if retention_days <= 0:
            return []
        now = now or time.time()
        with self.lock:  # 在调用方的线程里跑，遍历要和 add/delete 互斥
            return [
                j.id for j in list(self.jobs.values())
                if j.status not in ("queued", "running")
                and now - (j.finished or j.created) > retention_days * DAY
            ]

    def cleanup(self, retention_days: int, now: float | None = None,
                busy: frozenset[str] | None = None) -> list[str]:
        """删除超过保留时间的记录（运行中的、以及 busy 里的不删）。retention_days=0 表示永久保留。"""
        busy = busy or frozenset()
        expired = [jid for jid in self.expired_ids(retention_days, now) if jid not in busy]
        for jid in expired:
            self.delete(jid)
        return expired

    # ---------- 预览快照 ----------

    def save_preview(self, job: Job):
        if job.runner is None or not job.preview:
            return
        data = job.runner.preview(-1)
        if not data.get("ready"):
            return
        path = self.dir(job.id) / "preview.json"
        if path.parent.exists():
            _write_atomic(path, json.dumps(data, ensure_ascii=False))

    def load_preview(self, job: Job) -> dict | None:
        path = self.dir(job.id) / "preview.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def disk_usage(self) -> int:
        """记录和版面分析结果占的磁盘。硬链接共用的原文件只算一次。"""
        seen: dict[tuple[int, int], int] = {}
        for root in (self.root, self.layout_root):
            if root.exists():
                for p in root.rglob("*"):
                    if p.is_file():
                        st = p.stat()
                        seen[(st.st_dev, st.st_ino)] = st.st_size
        return sum(seen.values())
