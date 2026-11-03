import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.history import DAY, Job, JobStore
from app.settings import Settings, SettingsError

MD = b"# A Title\n\nFirst paragraph of the document is here.\n\nSecond paragraph goes right after it.\n"


# A record as the API returns it (JSON)
JobJSON = dict[str, Any]


def wait_done(client: TestClient, job_id: str, timeout: float = 30) -> JobJSON:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job  # type: ignore[no-any-return]
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def submit(client: TestClient, name: str = "book.md", data: bytes = MD, **extra: str) -> JobJSON:
    form = {"service_id": "mock", "target_lang": "zh-CN", **extra}
    r = client.post("/api/jobs", files={"file": (name, data)}, data=form)
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def test_sync_info_clears_failed_after_retry():
    job = Job(id="x", filename="a.pdf", failed=4)

    class Runner:
        info = {"failed": 0, "detected_lang": "en", "skipped": 0}

    job.runner = Runner()
    job.sync_info()
    assert job.failed == 0


def test_records_persist_across_restart(app_env, tmp_path):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        assert job["status"] == "done" and job["outputs"]
        assert job["service_name"] == "Mock" and job["size"] == len(MD) and job["finished"] > 0
        listed = client.get("/api/jobs").json()
        assert [j["id"] for j in listed["jobs"]] == [job["id"]] and listed["disk_usage"] > 0

    # Simulate a restart: load again from disk
    store = JobStore(tmp_path / "jobs")
    main.jobs = store
    with TestClient(main.app) as client:
        again = client.get(f"/api/jobs/{job['id']}").json()
        assert again["status"] == "done" and again["outputs"] == job["outputs"]
        # Downloads and the preview (from the snapshot) still work
        assert client.get(job["outputs"][0]["url"]).status_code == 200
        p = client.get(f"/api/jobs/{job['id']}/preview").json()
        assert p["ready"] and len(p["updates"]) == 3 and p["segments"][0]["k"] == "h1"
        assert client.get(f"/api/jobs/{job['id']}/preview?since={p['version']}").json()["updates"] == []


def test_interrupted_job_after_restart_can_resume(tmp_path):
    root = tmp_path / "jobs"
    (root / "abc").mkdir(parents=True)
    (root / "abc" / "job.json").write_text(
        json.dumps({"id": "abc", "filename": "x.md", "status": "running", "done": 3, "total": 9})
    )
    store = JobStore(root)
    job = store.get("abc")
    assert job.status == "interrupted" and "中断" in job.error
    # The status was written back to disk too
    assert json.loads((root / "abc" / "job.json").read_text())["status"] == "interrupted"


def test_concurrent_saves_of_one_job_do_not_collide(tmp_path):
    """The translation thread updating progress and an API request may save the same record at the same time; a
    temporary file name collision must not raise an error."""
    import threading

    store = JobStore(tmp_path / "jobs")
    job = Job(id="abc123def456", filename="book.md")
    store.dir(job.id).mkdir(parents=True)
    store.add(job)
    errors: list[Exception] = []

    def hammer():
        for _ in range(200):
            try:
                store.save(job)
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert json.loads((store.dir(job.id) / "job.json").read_text(encoding="utf-8"))["id"] == job.id
    assert not list(store.dir(job.id).glob("*.tmp"))


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
            assert again["status"] == "done" and sum(calls) == first  # Same service hits the cache

            main.store.services.append(
                {"id": "mock2", "provider": "mock", "builtin": True, "name": "Mock 2", "enabled": True}
            )
            switched = client.post(f"/api/jobs/{job['id']}/retry", json={"service_id": "mock2"}).json()
            assert switched["service_name"] == "Mock 2"
            wait_done(client, job["id"])
            assert sum(calls) > first  # A different service translates again
    finally:
        MockTranslator.translate_batch = original


def test_redo_can_ignore_cached_translations(app_env):
    """Retranslating with "全部重新翻译" (retranslate everything) checked clears the existing translations first and
    requests every paragraph again; without it the existing translations are reused."""
    main = app_env
    calls = []
    from app.translators import MockTranslator

    original = MockTranslator.translate_batch

    async def counting(self, texts):
        calls.extend(texts)
        return await original(self, texts)

    MockTranslator.translate_batch = counting
    try:
        with TestClient(main.app) as client:
            job = wait_done(client, submit(client)["id"])
            first = len(calls)
            assert first == 3
            again = wait_done(client, client.post(f"/api/jobs/{job['id']}/retry", json={"fresh": True}).json()["id"])
            assert again["status"] == "done" and again["outputs"]
            assert len(calls) == first * 2
            wait_done(client, client.post(f"/api/jobs/{job['id']}/retry", json={}).json()["id"])
            assert len(calls) == first * 2
    finally:
        MockTranslator.translate_batch = original


def test_redo_from_scratch_analyses_the_pdf_again(app_env, monkeypatch):
    """The PDF layout analysis is stored in the job directory and read back on resume and retranslation; with
    "全部重新翻译" checked the layout is analysed again."""
    import pymupdf

    from app.formats import pdf_units

    main = app_env
    calls = []
    real_extract = pdf_units._extract
    monkeypatch.setattr(pdf_units, "_extract", lambda *a, **k: calls.append(1) or real_extract(*a, **k))
    doc = pymupdf.open()
    doc.new_page(width=420, height=240).insert_text(
        (40, 60),
        "A body paragraph long enough to be translated here.",
        fontsize=10,
    )
    doc[0].insert_text((40, 120), "Closing Remarks", fontsize=14)
    doc.set_toc([[1, "Closing Remarks", 1]])
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client, name="paper.pdf", data=doc.tobytes())["id"])
        assert job["status"] == "done" and main.jobs.plan_path(main.jobs.get(job["id"])).exists() and len(calls) == 1
        wait_done(client, client.post(f"/api/jobs/{job['id']}/retry", json={}).json()["id"])
        assert len(calls) == 1
        again = wait_done(client, client.post(f"/api/jobs/{job['id']}/retry", json={"fresh": True}).json()["id"])
        assert again["status"] == "done" and len(calls) == 2


def test_delete_several_records_at_once(app_env):
    """Bulk delete: the selected records are deleted together with their files and unselected ones stay; ids that no
    longer exist are reported without affecting the others."""
    main = app_env
    with TestClient(main.app) as client:
        ids = [wait_done(client, submit(client, name=f"b{i}.md")["id"])["id"] for i in range(3)]
        r = client.post("/api/jobs/delete", json={"ids": [ids[0], ids[2], "nosuchjob"]})
        assert r.status_code == 200, r.text
        assert r.json() == {"deleted": 2, "missing": ["nosuchjob"]}
        left = [j["id"] for j in client.get("/api/jobs").json()["jobs"]]
        assert left == [ids[1]]
        assert not main.jobs.dir(ids[0]).exists() and main.jobs.dir(ids[1]).exists()


def test_retry_recomputes_preview_flag_for_legacy_jobs(app_env):
    """Records created before the preview feature store preview=False; retranslation must recompute it from the
    format."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        assert job["status"] == "done"
        # Simulate an old record: preview=False and no preview snapshot
        legacy = main.jobs.get(job["id"])
        legacy.preview = False
        main.jobs.save(legacy)
        (main.jobs.dir(job["id"]) / "preview.json").unlink(missing_ok=True)

        again = wait_done(client, client.post(f"/api/jobs/{job['id']}/retry", json={}).json()["id"])
        assert again["status"] == "done" and again["preview"] is True
        p = client.get(f"/api/jobs/{job['id']}/preview").json()
        assert p["ready"] and p["segments"]


def test_variant_generates_other_mode(app_env):
    """A finished job can generate the other version later (bilingual <-> translated-only) without requesting the
    translation service again."""
    main = app_env
    from app.translators import MockTranslator

    calls = []
    original = MockTranslator.translate_batch

    async def counting(self, texts):
        calls.append(len(texts))
        return await original(self, texts)

    MockTranslator.translate_batch = counting
    try:
        with TestClient(main.app) as client:
            job = wait_done(client, submit(client)["id"])  # Bilingual by default
            assert job["status"] == "done"
            assert any("bilingual" in o["name"] for o in job["outputs"])
            first = sum(calls)

            again = client.post(f"/api/jobs/{job['id']}/variant").json()
            names = [o["name"] for o in again["outputs"]]
            assert any("translated" in n for n in names) and any("bilingual" in n for n in names)
            assert sum(calls) == first  # Everything hits the cache
            url = next(o["url"] for o in again["outputs"] if "translated" in o["name"])
            assert client.get(url).status_code == 200

            once_more = client.post(f"/api/jobs/{job['id']}/variant").json()
            assert [o["name"] for o in once_more["outputs"]] == names  # Idempotent
    finally:
        MockTranslator.translate_batch = original


def test_pdf_translated_variant_is_cut_from_the_bilingual_output(app_env, monkeypatch):
    """The translated-only version of a PDF is taken from the existing bilingual version without typesetting again,
    and the bookmarks are replaced with translations from the preview snapshot."""
    import pymupdf

    main = app_env
    doc = pymupdf.open()
    doc.new_page(width=420, height=240).insert_text(
        (40, 60),
        "A body paragraph long enough to be translated here.",
        fontsize=10,
    )
    doc[0].insert_text((40, 120), "Closing Remarks", fontsize=14)
    doc.set_toc([[1, "Closing Remarks", 1]])
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client, name="paper.pdf", data=doc.tobytes())["id"])
        assert job["status"] == "done"

        async def no_render(*args: object, **kwargs: object) -> list[object]:
            raise AssertionError("不该重新排版")

        monkeypatch.setattr(main, "translate_file", no_render)
        again = client.post(f"/api/jobs/{job['id']}/variant")
        assert again.status_code == 200, again.text
        url = next(o["url"] for o in again.json()["outputs"] if o["name"] == "paper.translated.pdf")
        cut = pymupdf.open("pdf", client.get(url).content)
        assert cut[0].rect == pymupdf.Rect(0, 0, 420, 240) and "[zh-CN]" in cut[0].get_text()
        assert cut.get_toc() == [[1, "[zh-CN] Closing Remarks", 1]]


def test_variant_refuses_cache_miss_and_closes_client(app_env):
    """When the cache does not match, the request does not retranslate, and the translation client is closed whether
    it succeeds or fails."""
    main = app_env
    from app.runner import get_cache
    from app.translators import MockTranslator

    calls = []
    closed = []
    original = MockTranslator.translate_batch
    original_close = MockTranslator.aclose

    async def counting(self, texts):
        calls.append(len(texts))
        return await original(self, texts)

    async def close(self):
        closed.append(self)
        await original_close(self)

    MockTranslator.translate_batch = counting
    MockTranslator.aclose = close
    try:
        with TestClient(main.app) as client:
            job = wait_done(client, submit(client)["id"])
            before = sum(calls)
            cache = get_cache()
            cache.conn.execute("DELETE FROM t")
            cache.conn.commit()
            missed = client.post(f"/api/jobs/{job['id']}/variant")
            assert missed.status_code == 400
            assert "重新翻译" in missed.json()["detail"]
            assert sum(calls) == before
            assert closed
            assert client.get(job["outputs"][0]["url"]).status_code == 200
            assert job["id"] not in main.variant_tasks
    finally:
        MockTranslator.translate_batch = original
        MockTranslator.aclose = original_close


def test_variant_pins_the_model_that_produced_the_job(app_env, monkeypatch):
    """Generating the other version looks up the cache with the model stored in the job record, not the model the
    service has selected now."""
    main = app_env
    seen: list[str] = []
    original = main._make_translator

    def wrapped(job, service):
        seen.append(str(service.get("model") or ""))
        return original(job, service)

    monkeypatch.setattr(main, "_make_translator", wrapped)
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        seen.clear()
        stored = main.jobs.get(job["id"])
        assert stored is not None
        stored.model = "old-model"
        service = main.store.get("mock")
        service["model"] = "new-model"
        again = client.post(f"/api/jobs/{job['id']}/variant")
        assert again.status_code == 200, again.text
        assert seen == ["old-model"]


def test_variant_matches_marker_at_extension(app_env):
    """When the original file name itself contains .bilingual., the other version can still be generated; the
    substring does not count as an existing output."""
    main = app_env
    from app.formats import output_variant

    with TestClient(main.app) as client:
        job = wait_done(client, submit(client, name="book.bilingual.md", mode="translated")["id"])
        names = [o["name"] for o in job["outputs"]]
        assert names == ["book.bilingual.translated.md"]
        assert output_variant(names[0]) == "translated"
        again = client.post(f"/api/jobs/{job['id']}/variant")
        assert again.status_code == 200, again.text
        both = [o["name"] for o in again.json()["outputs"]]
        assert "book.bilingual.bilingual.md" in both
        assert output_variant("book.bilingual.bilingual.md") == "bilingual"


def test_variant_rejects_second_request_and_discards_partial(app_env, monkeypatch):
    """A second request is rejected while generation is running; a failure halfway through writing leaves the record
    unchanged and removes the temporary directory."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        main.variant_tasks[job["id"]] = object()  # Only checks whether generation is running; no real Task is needed
        busy = client.post(f"/api/jobs/{job['id']}/variant")
        assert busy.status_code == 409
        main.variant_tasks.pop(job["id"])

        async def boom(
            src: object,
            out_dir: object,
            runner: object,
            bilingual: bool,
            target_lang: str,
            layout_model: str = "",
            plan_path: object = None,
            formula_model: str = "",
        ) -> list[object]:
            from pathlib import Path

            (Path(str(out_dir)) / "partial.md").write_text("x", encoding="utf-8")
            raise ValueError("boom")

        monkeypatch.setattr(main, "translate_file", boom)
        failed = client.post(f"/api/jobs/{job['id']}/variant")
        assert failed.status_code == 400
        again = client.get(f"/api/jobs/{job['id']}").json()
        assert [o["name"] for o in again["outputs"]] == [o["name"] for o in job["outputs"]]
        assert not (main.jobs.out_dir(job["id"]) / "partial.md").exists()
        assert not (main.jobs.dir(job["id"]) / ".variant").exists()
        assert job["id"] not in main.variant_tasks


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
        job = Job(
            id=jid,
            filename="a.md",
            status=status,
            created=now - age_days * DAY,
            finished=0 if status == "running" else now - age_days * DAY,
        )
        store.add(job)
    assert store.cleanup(0, now) == []  # 0 = keep forever
    assert store.cleanup(30, now, busy=frozenset({"old"})) == []  # Busy generating another version, so not deleted
    assert store.get("old")
    assert store.cleanup(30, now) == ["old"]
    assert not store.dir("old").exists() and store.get("new") and store.get("busy")


def test_settings(tmp_path):
    s = Settings(tmp_path / "settings.json")
    assert s.retention_days == 30
    s.update({"retention_days": 7})
    assert Settings(tmp_path / "settings.json").retention_days == 7  # Persisted to disk
    s.update({"retention_days": 0})
    assert s.retention_days == 0
    for bad in (-1, 99999, "x"):
        with pytest.raises(SettingsError):
            s.update({"retention_days": bad})
    (tmp_path / "broken.json").write_text("{not json")
    assert Settings(tmp_path / "broken.json").retention_days == 30


def test_cleanup_skips_inflight_variant(app_env):
    """Expiry cleanup does not delete a record whose other version is being generated."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        record = main.jobs.get(job["id"])
        record.finished = record.created = time.time() - 10 * DAY
        main.variant_tasks[job["id"]] = object()
        assert client.put("/api/settings", json={"retention_days": 7}).status_code == 200
        assert client.get(f"/api/jobs/{job['id']}").status_code == 200
        main.variant_tasks.pop(job["id"])
        client.put("/api/settings", json={"retention_days": 7})
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404


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
    """Slows down each batch and records the model used by each request (the model field of the mock service)."""
    import asyncio

    from app.translators import MockTranslator

    calls: list[tuple[str, int]] = []
    original = MockTranslator.translate_batch

    async def slow(self, texts):
        # If it is too fast the whole document is translated before the pause request is sent, especially under high
        # machine load
        await asyncio.sleep(0.2)
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
        assert client.get(f"/api/jobs/{job_id}").json()["done"] == paused["done"]  # It really stopped
        assert sum(n for _, n in slow_mock) == translated_before
        assert client.post(f"/api/jobs/{job_id}/pause").status_code == 400  # A paused job cannot be paused again

        # The model changed while paused: resuming uses the new model and does not request translated paragraphs again
        main.store.get("mock")["model"] = "m-new"
        resumed = client.post(f"/api/jobs/{job_id}/retry", json={}).json()
        assert resumed["model"] == "m-new"
        done = wait_done(client, job_id)
        assert done["status"] == "done" and done["model"] == "m-new"
        assert {m for m, _ in slow_mock[:calls_before]} == {"m-old"}
        assert {m for m, _ in slow_mock[calls_before:]} == {"m-new"}
        assert sum(n for _, n in slow_mock) == done["total"]  # Each paragraph was translated once


def test_pause_all_and_resume_all(app_env, slow_mock):
    main = app_env
    with TestClient(main.app) as client:
        ids = [submit(client, name=f"b{i}.md", data=LONG_MD + str(i).encode())["id"] for i in range(2)]
        for job_id in ids:
            wait_until(client, job_id, lambda j: j["status"] == "running")
        assert client.post("/api/jobs/pause-all").json() == {"paused": 2}
        assert {client.get(f"/api/jobs/{i}").json()["status"] for i in ids} == {"paused"}
        assert client.post("/api/jobs/pause-all").json() == {"paused": 0}

        # A paused record is still paused after a restart and does not become "interrupted"
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
        main.running[job.id] = asyncio.create_task(asyncio.sleep(10))  # A job that has not entered _run_job yet
        assert await main._pause(job)
        return job

    job = asyncio.run(scenario())
    assert job.status == "paused" and job.finished > 0 and "q1" not in main.running
    assert JobStore(main.jobs.root).get("q1").status == "paused"


def test_runner_released_after_finish(app_env):
    """After a job finishes the Runner releases its memory and the preview is read from the preview.json snapshot."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        # The status becomes done first and the Runner is released only after the snapshot is written, one thread
        # switch later
        deadline = time.monotonic() + 10
        while main.jobs.get(job["id"]).runner is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert main.jobs.get(job["id"]).runner is None
        p = client.get(f"/api/jobs/{job['id']}/preview").json()
        assert p["ready"] and p["status"] == "done" and len(p["updates"]) == 3


def test_resume_with_invalid_service_keeps_result(app_env):
    """The configuration is invalid on resume: the response is 400, and the old translations and the job status are
    kept."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        main.store.services.append(
            {
                "id": "broken",
                "provider": "claude",
                "builtin": False,
                "name": "Broken",
                "enabled": True,
                "api_key": "",
                "model": "m",
            }
        )
        r = client.post(f"/api/jobs/{job['id']}/retry", json={"service_id": "broken"})
        assert r.status_code == 400 and "API Key" in r.json()["detail"]
        after = client.get(f"/api/jobs/{job['id']}").json()
        assert after["status"] == "done" and after["outputs"] == job["outputs"]
        assert client.get(job["outputs"][0]["url"]).status_code == 200


def test_failed_upload_leaves_no_orphan_dir(app_env):
    """A read fails halfway through an upload: the directory is removed and no orphan directory is left."""
    import asyncio

    main = app_env

    class BrokenUpload:
        filename = "broken.md"

        async def read(self, size=-1):
            raise OSError("read failed")

    with pytest.raises(OSError):
        asyncio.run(
            main.create_job(
                BrokenUpload(),
                target_lang="zh-CN",
                source_lang="auto",
                service_id="mock",
                mode="bilingual",
                glossary_ids=[],
            )
        )
    root = main.jobs.root
    assert not root.exists() or not list(root.iterdir())


def test_download_rejects_path_traversal(app_env):
    from fastapi import HTTPException

    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        record = main.jobs.get(job["id"])
        record.outputs.append("../job.json")  # Simulate a tampered job.json
        with pytest.raises(HTTPException) as e:
            main.download(job["id"], "../job.json")
        assert e.value.status_code == 404


def test_download_url_encodes_hash_and_question_mark(app_env):
    """When the file name contains # or ?, the download link is encoded; otherwise the browser requests only the part
    before #."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client, name="C# 入门?.md")["id"])
        [out] = job["outputs"]
        assert "#" not in out["url"] and "?" not in out["url"]
        assert client.get(out["url"]).status_code == 200


def test_inline_file_opens_in_the_browser(app_env):
    """With inline=1 the response header is inline and the browser opens the file; by default it is still an
    attachment download."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        [out] = job["outputs"]
        assert client.get(out["url"]).headers["content-disposition"].startswith("attachment")
        inline = client.get(out["url"] + "?inline=1")
        assert inline.status_code == 200
        assert inline.headers["content-disposition"].startswith("inline")


def _small_epub() -> bytes:
    import io
    import zipfile

    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Small Book</dc:title></metadata>'
        '<manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        '<item id="c1" href="text/ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/></manifest>'
        '<spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>'
    )
    nav = (
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>'
        '<nav epub:type="toc"><ol><li><a href="text/ch1.xhtml">One</a>'
        '<ol><li><a href="text/ch1.xhtml#part">Part</a></li></ol></li>'
        '<li><a href="text/ch2.xhtml">Two</a></li></ol></nav></body></html>'
    )
    page = '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>{}</p><script>alert(1)</script></body></html>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/nav.xhtml", nav)
        z.writestr("OEBPS/text/ch1.xhtml", page.format("The first chapter of the book."))
        z.writestr("OEBPS/text/ch2.xhtml", page.format("The second chapter of the book."))
    return buf.getvalue()


def test_epub_output_is_previewed_chapter_by_chapter(app_env):
    """EPUB preview: the table of contents and reading order are read from the book, chapters are fetched by their
    path in the zip, and scripts in the book are not allowed to run."""
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client, name="small.epub", data=_small_epub())["id"])
        [out] = [o for o in job["outputs"] if o["name"].endswith(".epub")]
        base = f"/api/jobs/{job['id']}/epub/{out['name']}"
        outline = client.get(base).json()
        assert outline["title"] == "Small Book"
        assert outline["spine"] == ["OEBPS/text/ch1.xhtml", "OEBPS/text/ch2.xhtml"]
        assert [(e["label"].split()[-1], e["path"], e["anchor"], e["depth"]) for e in outline["toc"]] == [
            ("One", "OEBPS/text/ch1.xhtml", "", 0),
            ("Part", "OEBPS/text/ch1.xhtml", "part", 1),
            ("Two", "OEBPS/text/ch2.xhtml", "", 0),
        ]
        page = client.get(f"{base}/OEBPS/text/ch2.xhtml")
        assert page.status_code == 200 and "second chapter" in page.text
        assert page.headers["content-type"].startswith("application/xhtml+xml")
        assert "script-src 'none'" in page.headers["content-security-policy"]
        assert client.get(f"{base}/OEBPS/text/missing.xhtml").status_code == 404
        assert client.get(f"/api/jobs/{job['id']}/epub/small.epub").status_code == 404


def test_oversized_upload_rejected_before_reading_body(app_env, monkeypatch):
    """A Content-Length above the limit returns 413 at once and the request never reaches create_job (without waiting
    for the whole file to be written to a temporary file)."""
    main = app_env
    monkeypatch.setattr(main, "MAX_UPLOAD", 1000)
    entered = []
    real_get = main.store.get

    def get(sid):
        entered.append(sid)  # create_job looks up the translation service as soon as it is entered
        return real_get(sid)

    monkeypatch.setattr(main.store, "get", get)
    with TestClient(main.app) as client:
        r = client.post(
            "/api/jobs",
            files={"file": ("big.md", b"x" * (2 * 1024 * 1024))},
            data={"service_id": "mock", "target_lang": "zh-CN"},
        )
        assert r.status_code == 413 and "上限" in r.json()["detail"]
        assert not entered
        assert client.get("/api/jobs").json()["jobs"] == []
        assert submit(client)["status"] in ("queued", "running", "done")


def test_running_job_shows_its_preparation_stage():
    from app.runner import Runner
    from app.translators import MockTranslator

    job = Job(id="x", filename="a.pdf")
    assert job.to_dict()["stage"] is None
    job.runner = Runner(MockTranslator("zh-CN"), cache=object())
    job.runner.report_stage("分析版面", 120, 646, "页")
    assert job.to_dict()["stage"] == {"name": "分析版面", "done": 120, "total": 646, "unit": "页"}
