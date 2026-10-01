"""翻译记录：每个任务一个目录 data/jobs/<id>/，元数据存在 job.json，原文件和译文在同一个目录里。

- 记录和文件只在两种情况下删除：用户在列表里删掉，或者超过保留时间（settings.retention_days）。
- 服务重启后从磁盘恢复记录；重启前没跑完的任务标记为“已中断”，可以重新翻译（已翻好的段落在缓存里）。
- 预览：运行中的任务从 Runner 实时读取；已完成的任务把预览快照存成 preview.json，之后随时能打开。
"""
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .langdetect import same_language

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
DAY = 24 * 3600


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued / running / done / error / interrupted
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

    # 不落盘的运行时字段
    runner: object = field(default=None, repr=False, compare=False)

    def sync_info(self):
        """把 Runner 里实时的检测结果同步到记录上。"""
        if self.runner is not None:
            info = self.runner.info
            self.detected_lang = info.get("detected_lang", "") or self.detected_lang
            self.skipped = info.get("skipped", 0) or self.skipped

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
            # 插件的 sameLangCheck：检测到的源语言和目标语言一致时提示
            "same_lang": same_language(self.detected_lang, self.target_lang),
            "preview": self.preview,
            "outputs": [{"name": n, "url": f"/api/jobs/{self.id}/files/{n}"} for n in self.outputs],
        }


# 落盘的字段：除了运行时的 runner 以外全部
_PERSISTED = [f for f in Job.__dataclass_fields__ if f != "runner"]


class JobStore:
    def __init__(self, root: Path = JOBS_DIR):
        self.root = root
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
                job.error = "服务重启，翻译被中断。已翻译的段落有缓存，重新翻译会很快。"
            self.jobs[job.id] = job
        for job in self.jobs.values():
            self.save(job)

    def save(self, job: Job):
        job.sync_info()
        path = self.dir(job.id) / "job.json"
        if not path.parent.exists():
            return  # 记录已删除
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: getattr(job, k) for k in _PERSISTED}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)

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

    def cleanup(self, retention_days: int, now: float | None = None) -> list[str]:
        """删除超过保留时间的记录（运行中的不删）。retention_days=0 表示永久保留。"""
        if retention_days <= 0:
            return []
        now = now or time.time()
        expired = [
            j.id for j in list(self.jobs.values())
            if j.status not in ("queued", "running") and now - (j.finished or j.created) > retention_days * DAY
        ]
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
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load_preview(self, job: Job) -> dict | None:
        path = self.dir(job.id) / "preview.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def disk_usage(self) -> int:
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
