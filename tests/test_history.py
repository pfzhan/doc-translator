import json
import time

import pytest
from fastapi.testclient import TestClient

from app.history import DAY, Job, JobStore
from app.settings import Settings, SettingsError

MD = b"# A Title\n\nFirst paragraph of the document is here.\n\nSecond paragraph goes right after it.\n"


def wait_done(client, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def submit(client, name="book.md", data=MD, **form):
    form = {"service_id": "mock", "target_lang": "zh-CN", **form}
    r = client.post("/api/jobs", files={"file": (name, data)}, data=form)
    assert r.status_code == 200, r.text
    return r.json()


def test_records_persist_across_restart(app_env, tmp_path):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        assert job["status"] == "done" and job["outputs"]
        assert job["service_name"] == "Mock" and job["size"] == len(MD) and job["finished"] > 0
        listed = client.get("/api/jobs").json()
        assert [j["id"] for j in listed["jobs"]] == [job["id"]] and listed["disk_usage"] > 0

    # 模拟重启：重新从磁盘加载
    store = JobStore(tmp_path / "jobs")
    main.jobs = store
    with TestClient(main.app) as client:
        again = client.get(f"/api/jobs/{job['id']}").json()
        assert again["status"] == "done" and again["outputs"] == job["outputs"]
        # 下载和预览（来自快照）都还能用
        assert client.get(job["outputs"][0]["url"]).status_code == 200
        p = client.get(f"/api/jobs/{job['id']}/preview").json()
        assert p["ready"] and len(p["updates"]) == 3 and p["segments"][0]["k"] == "h1"
        assert client.get(f"/api/jobs/{job['id']}/preview?since={p['version']}").json()["updates"] == []


def test_interrupted_job_after_restart_can_resume(tmp_path):
    root = tmp_path / "jobs"
    (root / "abc").mkdir(parents=True)
    (root / "abc" / "job.json").write_text(json.dumps({"id": "abc", "filename": "x.md", "status": "running",
                                                       "done": 3, "total": 9}))
    store = JobStore(root)
    job = store.get("abc")
    assert job.status == "interrupted" and "中断" in job.error
    # 状态也写回了磁盘
    assert json.loads((root / "abc" / "job.json").read_text())["status"] == "interrupted"


def test_retry_reuses_cache_and_can_switch_service(app_env):
    main = app_env
    calls = []
    from app.translators import MockTranslator

    original = MockTranslator.translate_batch

    async def counting(self, texts):
        calls.append(len(texts))
        return await original(self, texts)

    MockTranslator.translate_batch = counting
    try:
        with TestClient(main.app) as client:
            job = wait_done(client, submit(client)["id"])
            first = sum(calls)
            again = wait_done(client, client.post(f"/api/jobs/{job['id']}/retry", json={}).json()["id"])
            assert again["status"] == "done" and sum(calls) == first  # 同一服务命中缓存

            main.store.services.append({"id": "mock2", "provider": "mock", "builtin": True, "name": "Mock 2",
                                        "enabled": True})
            switched = client.post(f"/api/jobs/{job['id']}/retry", json={"service_id": "mock2"}).json()
            assert switched["service_name"] == "Mock 2"
            wait_done(client, job["id"])
            assert sum(calls) > first  # 换了服务重新翻译
    finally:
        MockTranslator.translate_batch = original


def test_delete_removes_record_and_files(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        job_dir = main.jobs.dir(job["id"])
        assert job_dir.exists()
        assert client.delete(f"/api/jobs/{job['id']}").json() == {"ok": True}
        assert not job_dir.exists()
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404
        assert client.get(job["outputs"][0]["url"]).status_code == 404
        assert client.delete(f"/api/jobs/{job['id']}").status_code == 404


def test_cleanup_respects_retention(tmp_path):
    store = JobStore(tmp_path / "jobs")
    now = time.time()
    for jid, age_days, status in [("old", 40, "done"), ("new", 5, "done"), ("busy", 40, "running")]:
        store.dir(jid).mkdir(parents=True)
        job = Job(id=jid, filename="a.md", status=status, created=now - age_days * DAY,
                  finished=0 if status == "running" else now - age_days * DAY)
        store.add(job)
    assert store.cleanup(0, now) == []  # 0 = 永久保留
    assert store.cleanup(30, now) == ["old"]
    assert not store.dir("old").exists() and store.get("new") and store.get("busy")


def test_settings(tmp_path):
    s = Settings(tmp_path / "settings.json")
    assert s.retention_days == 30
    s.update({"retention_days": 7})
    assert Settings(tmp_path / "settings.json").retention_days == 7  # 落盘
    s.update({"retention_days": 0})
    assert s.retention_days == 0
    for bad in (-1, 99999, "x"):
        with pytest.raises(SettingsError):
            s.update({"retention_days": bad})
    (tmp_path / "broken.json").write_text("{not json")
    assert Settings(tmp_path / "broken.json").retention_days == 30


def test_settings_endpoint_triggers_cleanup(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        record = main.jobs.get(job["id"])
        record.finished = record.created = time.time() - 10 * DAY
        assert client.put("/api/settings", json={"retention_days": 30}).json() == {"retention_days": 30}
        assert client.get(f"/api/jobs/{job['id']}").status_code == 200
        client.put("/api/settings", json={"retention_days": 7})
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404
        assert client.put("/api/settings", json={"retention_days": -3}).status_code == 400


LONG_MD = "\n\n".join(f"Paragraph number {i} of the long test document goes here." for i in range(80)).encode()


@pytest.fixture
def slow_mock(app_env, monkeypatch):
    """每批慢一点，并记下每次请求用的模型（mock 服务的 model 字段）。"""
    import asyncio

    from app.translators import MockTranslator

    calls: list[tuple[str, int]] = []
    original = MockTranslator.translate_batch

    async def slow(self, texts):
        await asyncio.sleep(0.05)
        calls.append((getattr(self, "model", ""), len(texts)))
        return await original(self, texts)

    real_create = app_env.create_translator

    def create(service, *a, **kw):
        tr = real_create(service, *a, **kw)
        tr.model = service.get("model", "")
        return tr

    monkeypatch.setattr(MockTranslator, "translate_batch", slow)
    monkeypatch.setattr(app_env, "create_translator", create)
    return calls


def wait_until(client, job_id, pred, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if pred(job):
            return job
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def test_pause_then_resume_with_current_model(app_env, slow_mock):
    main = app_env
    main.store.get("mock")["model"] = "m-old"
    with TestClient(main.app) as client:
        job_id = submit(client, data=LONG_MD)["id"]
        wait_until(client, job_id, lambda j: j["done"] > 0)
        r = client.post(f"/api/jobs/{job_id}/pause")
        assert r.status_code == 200, r.text
        paused = r.json()
        assert paused["status"] == "paused" and 0 < paused["done"] < paused["total"] and paused["model"] == "m-old"
        translated_before, calls_before = sum(n for _, n in slow_mock), len(slow_mock)
        time.sleep(0.2)
        assert client.get(f"/api/jobs/{job_id}").json()["done"] == paused["done"]  # 真的停了
        assert sum(n for _, n in slow_mock) == translated_before
        assert client.post(f"/api/jobs/{job_id}/pause").status_code == 400  # 已暂停不能再暂停

        # 暂停期间换了模型：继续翻译用新模型，翻好的段落不再请求
        main.store.get("mock")["model"] = "m-new"
        resumed = client.post(f"/api/jobs/{job_id}/retry", json={}).json()
        assert resumed["model"] == "m-new"
        done = wait_done(client, job_id)
        assert done["status"] == "done" and done["model"] == "m-new"
        assert {m for m, _ in slow_mock[:calls_before]} == {"m-old"}
        assert {m for m, _ in slow_mock[calls_before:]} == {"m-new"}
        assert sum(n for _, n in slow_mock) == done["total"]  # 每段只翻了一次


def test_pause_all_and_resume_all(app_env, slow_mock):
    main = app_env
    with TestClient(main.app) as client:
        ids = [submit(client, name=f"b{i}.md", data=LONG_MD + str(i).encode())["id"] for i in range(2)]
        for job_id in ids:
            wait_until(client, job_id, lambda j: j["status"] == "running")
        assert client.post("/api/jobs/pause-all").json() == {"paused": 2}
        assert {client.get(f"/api/jobs/{i}").json()["status"] for i in ids} == {"paused"}
        assert client.post("/api/jobs/pause-all").json() == {"paused": 0}

        # 暂停的记录重启后还是暂停，不会变成“中断”
        reloaded = JobStore(main.jobs.root)
        assert {reloaded.get(i).status for i in ids} == {"paused"}

        assert client.post("/api/jobs/resume-all").json() == {"resumed": 2, "failed": []}
        assert {wait_done(client, i)["status"] for i in ids} == {"done"}
        assert client.post("/api/jobs/resume-all").json() == {"resumed": 0, "failed": []}


def test_pause_queued_job(app_env):
    import asyncio

    main = app_env

    async def scenario():
        job = Job(id="q1", filename="x.md", status="queued")
        main.jobs.dir(job.id).mkdir(parents=True)
        main.jobs.add(job)
        main.running[job.id] = asyncio.create_task(asyncio.sleep(10))  # 还没进入 _run_job 的任务
        assert await main._pause(job)
        return job

    job = asyncio.run(scenario())
    assert job.status == "paused" and job.finished > 0 and "q1" not in main.running
    assert JobStore(main.jobs.root).get("q1").status == "paused"


def test_runner_released_after_finish(app_env):
    """任务结束后 Runner 释放内存，预览改从 preview.json 快照读。"""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        assert main.jobs.get(job["id"]).runner is None
        p = client.get(f"/api/jobs/{job['id']}/preview").json()
        assert p["ready"] and p["status"] == "done" and len(p["updates"]) == 3


def test_resume_with_invalid_service_keeps_result(app_env):
    """继续翻译时配置失效：报 400，但旧译文和任务状态都不能丢。"""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        main.store.services.append({"id": "broken", "provider": "claude", "builtin": False, "name": "Broken",
                                    "enabled": True, "api_key": "", "model": "m"})
        r = client.post(f"/api/jobs/{job['id']}/retry", json={"service_id": "broken"})
        assert r.status_code == 400 and "API Key" in r.json()["detail"]
        after = client.get(f"/api/jobs/{job['id']}").json()
        assert after["status"] == "done" and after["outputs"] == job["outputs"]
        assert client.get(job["outputs"][0]["url"]).status_code == 200


def test_failed_upload_leaves_no_orphan_dir(app_env):
    """上传中途读失败：目录要清理掉，不能留孤儿目录。"""
    import asyncio

    main = app_env

    class BrokenUpload:
        filename = "broken.md"

        async def read(self, size=-1):
            raise OSError("read failed")

    with pytest.raises(OSError):
        asyncio.run(main.create_job(BrokenUpload(), target_lang="zh-CN", source_lang="auto",
                                    service_id="mock", mode="bilingual"))
    root = main.jobs.root
    assert not root.exists() or not list(root.iterdir())


def test_download_rejects_path_traversal(app_env):
    from fastapi import HTTPException

    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        record = main.jobs.get(job["id"])
        record.outputs.append("../job.json")  # 模拟被篡改的 job.json
        with pytest.raises(HTTPException) as e:
            main.download(job["id"], "../job.json")
        assert e.value.status_code == 404
