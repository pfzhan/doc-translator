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
