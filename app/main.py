"""Web 服务：上传文档 → 后台翻译 → 轮询进度 → 下载结果。"""
import asyncio
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .formats import SUPPORTED, translate_file
from .prompts import default_prompts
from .runner import Runner
from .services import PROVIDERS, ServiceError, ServiceStore
from .langdetect import same_language
from .languages import GOOGLE_CODES, LANGUAGES
from .translators import TranslatorError, create_translator

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
MAX_UPLOAD = 200 * 1024 * 1024
JOB_TTL = 24 * 3600

app = FastAPI(title="Doc Translator")
store = ServiceStore()


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued / running / done / error
    done: int = 0
    total: int = 0
    error: str = ""
    outputs: list[Path] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    source_lang: str = "auto"
    target_lang: str = ""
    # Runner.info：检测到的文档语言、因已是目标语言而跳过的段落数
    info: dict = field(default_factory=dict)

    def to_dict(self):
        detected = self.info.get("detected_lang", "")
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "detected_lang": detected,
            "skipped": self.info.get("skipped", 0),
            # 插件的 sameLangCheck：检测到的源语言和目标语言一致时提示
            "same_lang": same_language(detected, self.target_lang),
            "outputs": [{"name": p.name, "url": f"/api/jobs/{self.id}/files/{p.name}"} for p in self.outputs],
        }


jobs: dict[str, Job] = {}


def _cleanup_old_jobs():
    now = time.time()
    for jid, job in list(jobs.items()):
        if now - job.created > JOB_TTL and job.status in ("done", "error"):
            shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
            jobs.pop(jid, None)


async def _run_job(job: Job, src: Path, out_dir: Path, translator, bilingual: bool, target_lang: str):
    def progress(done, total):
        job.done, job.total = done, total

    job.status = "running"
    runner = Runner(translator, progress)
    job.info = runner.info
    try:
        job.outputs = await translate_file(src, out_dir, runner, bilingual, target_lang)
        job.status = "done"
    except (TranslatorError, ValueError) as e:
        job.status, job.error = "error", str(e)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
    finally:
        await translator.aclose()


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


# ---------- 翻译任务 ----------

@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    target_lang: str = Form("zh-CN"),
    source_lang: str = Form("auto"),
    service_id: str = Form(""),
    mode: str = Form("bilingual"),
):
    _cleanup_old_jobs()
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"不支持的格式 {ext}，支持: {', '.join(sorted(SUPPORTED))}")
    service = store.get(service_id or store.default_id)
    if not service:
        raise HTTPException(400, "翻译服务不存在")
    try:
        translator = create_translator(service, target_lang, source_lang)
    except TranslatorError as e:
        raise HTTPException(400, str(e))

    job = Job(id=uuid.uuid4().hex[:12], filename=name, source_lang=source_lang, target_lang=target_lang)
    job_dir = JOBS_DIR / job.id
    out_dir = job_dir / "out"
    out_dir.mkdir(parents=True)
    src = job_dir / name
    size = 0
    with src.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                await translator.aclose()
                raise HTTPException(413, "文件太大（上限 200MB）")
            f.write(chunk)

    jobs[job.id] = job
    asyncio.create_task(_run_job(job, src, out_dir, translator, mode == "bilingual", target_lang))
    return job.to_dict()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    for p in job.outputs:
        if p.name == name:
            return FileResponse(p, filename=p.name)
    raise HTTPException(404, "文件不存在")


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
