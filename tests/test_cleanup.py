"""Clearing what a record left in caches: registering cache owners, clearing by owner, and clearing through the API
and when records are deleted."""

import asyncio
import hashlib
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from app import cleanup, runner, state
from app.history import Job
from app.runner import Cache, Runner
from app.translators import MockTranslator
from tests.test_history import MD, _counted, _job, submit, wait_done


def _rows(cache: Cache) -> int:
    count: int = cache.conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    return count


def _layout_dir(job: Job) -> Path:
    """The record's layout directory, which the test expects it to have (its source file is there)."""
    folder = state.jobs.layout_dir(job)
    assert folder is not None
    return folder


def _plan_path(job: Job) -> Path:
    """Where the record's layout analysis is saved, which the test expects it to have."""
    path = state.jobs.plan_path(job)
    assert path is not None
    return path


def test_cache_purges_only_entries_claimed_by_the_document(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "c.sqlite3")
    cache.put_many("p", {"a": "A", "b": "B", "c": "C"})
    cache.claim("doc1", "p", ["a", "b"])
    cache.claim("doc2", "p", ["c"])
    owned = cache.owned("doc1")
    assert owned == Cache.keys("p", ["a", "b"])
    assert cache.purge("doc1", owned) == 2
    assert cache.get_many("p", ["a", "b", "c"]) == {"c": "C"}
    assert cache.owned("doc1") == set() and cache.owned("doc2") == Cache.keys("p", ["c"])


def test_owned_includes_extra_keys_still_in_cache(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "c.sqlite3")
    cache.put_many("p", {"old": "O"})
    extra = Cache.keys("p", ["old", "gone"])
    assert cache.owned("doc", extra) == Cache.keys("p", ["old"])


def test_dropped_entries_lose_their_owner(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "c.sqlite3")
    cache.put_many("p", {"a": "A"})
    cache.claim("doc", "p", ["a"])
    cache.drop_many("p", ["a"])
    assert cache.conn.execute("SELECT COUNT(*) FROM owner").fetchone()[0] == 0


def _fill(cache: Cache, owner: str, texts: list[str]) -> None:
    """Store and claim entries ten at a time, as runs and page requests do."""
    for i in range(0, len(texts), 10):
        batch = texts[i : i + 10]
        cache.put_many("p", {t: t.upper() for t in batch})
        if owner:
            cache.claim(owner, "p", batch)


def test_a_full_cache_drops_web_and_ownerless_entries_before_documents(tmp_path: Path) -> None:
    """Browsing used to push a book's translations out of a full cache, so continuing the book or generating its
    other version had to translate them again. Entries a document uses now go last."""
    cache = Cache(tmp_path / "c.sqlite3")
    cache.MAX_ENTRIES = 100
    book = [f"book {i}" for i in range(50)]
    shared = [f"shared {i}" for i in range(5)]
    _fill(cache, "", [f"legacy {i}" for i in range(20)])  # Saved before owners were recorded
    _fill(cache, "bookdoc", book + shared)
    _fill(cache, runner.WEB_CACHE_OWNER, shared)
    _fill(cache, runner.WEB_CACHE_OWNER, [f"page {i}" for i in range(500)])
    assert len(cache.get_many("p", book + shared)) == 55
    assert _rows(cache) <= cache.MAX_ENTRIES
    assert cache.get_many("p", ["legacy 0"]) == {}
    # The newest web entries stay; only older ones were dropped
    assert cache.get_many("p", ["page 499"]) == {"page 499": "PAGE 499"}


def test_a_cache_full_of_documents_drops_their_oldest_entries_first(tmp_path: Path) -> None:
    """With only document entries left, the oldest go, and the entries a run has just stored are kept even though
    they get their owner only after being stored."""
    cache = Cache(tmp_path / "c.sqlite3")
    cache.MAX_ENTRIES = 100
    _fill(cache, "old", [f"old {i}" for i in range(100)])
    _fill(cache, "new", [f"new {i}" for i in range(30)])
    assert len(cache.get_many("p", [f"new {i}" for i in range(30)])) == 30
    assert cache.get_many("p", ["old 0"]) == {} and cache.get_many("p", ["old 99"]) == {"old 99": "OLD 99"}


def test_runner_claims_written_and_hit_entries(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "c.sqlite3")
    tr = MockTranslator("zh-CN")
    first = Runner(tr, cache=cache, skip_same_lang=False)
    first.set_document("doc1")
    asyncio.run(first.translate_all(["Hello world, this is a sentence."]))
    assert len(cache.owned("doc1")) == 1
    # Another file hitting the same cache entry is registered as an owner too
    second = Runner(tr, cache=cache, skip_same_lang=False)
    second.set_document("doc2")
    asyncio.run(second.translate_all(["Hello world, this is a sentence."]))
    assert cache.owned("doc2") == cache.owned("doc1")


def test_clearing_a_job_removes_its_cache_and_generated_files(app_env: ModuleType) -> None:
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        folder = state.jobs.dir(job["id"])
        boxes = _layout_dir(_job(state.jobs, job["id"])) / "boxes.json"
        boxes.write_text("{}")
        info = client.get(f"/api/jobs/{job['id']}/cache").json()
        assert info["entries"] == 3 and info["outputs"] == 1 and info["shared"] == 0
        assert info["generated_bytes"] > 0 and info["boxes_bytes"] == 2

        r = client.delete(f"/api/jobs/{job['id']}/cache")
        assert r.status_code == 200, r.text
        cleared = r.json()
        assert cleared["removed"] == 3 and cleared["status"] == "idle"
        assert cleared["outputs"] == [] and cleared["total"] == 0
        assert _rows(cache) == 0
        assert sorted(p.name for p in folder.iterdir()) == ["book.md", "job.json", "out"]
        assert not any((folder / "out").iterdir()) and boxes.is_file()

        # The record can still be translated again
        client.post(f"/api/jobs/{job['id']}/retry")
        again = wait_done(client, job["id"])
        assert again["status"] == "done" and again["outputs"] and _rows(cache) == 3


def test_clearing_can_include_model_boxes(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        boxes = _layout_dir(_job(state.jobs, job["id"])) / "boxes.json"
        boxes.write_text("{}")
        assert client.delete(f"/api/jobs/{job['id']}/cache?boxes=true").status_code == 200
        assert not boxes.exists()


def test_clearing_is_refused_while_translating(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        _job(state.jobs, job["id"]).status = "running"
        r = client.delete(f"/api/jobs/{job['id']}/cache")
        assert r.status_code == 409


def test_jobs_of_the_same_file_share_cache(app_env: ModuleType) -> None:
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        a = wait_done(client, submit(client)["id"])
        b = wait_done(client, submit(client)["id"])
        assert client.get(f"/api/jobs/{a['id']}/cache").json()["shared"] == 1
        # Delete one of them; the other still uses the cache, so it stays
        client.delete(f"/api/jobs/{a['id']}")
        assert _rows(cache) == 3
        # Delete the last one too, and the cache is cleared with it
        client.delete(f"/api/jobs/{b['id']}")
        assert _rows(cache) == 0


def test_entries_saved_before_ownership_are_found_from_the_saved_plan(
    app_env: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        # Simulate a cache saved before owners were registered: remove the registration, and add an old paragraph that
        # appears only in the last layout analysis
        cache.conn.execute("DELETE FROM owner")
        tr = MockTranslator("zh-CN")
        tr.service_id = "mock"
        tr.detected_source = "en"
        cache.put_many(tr.cache_key, {"An old paragraph.": "旧译文"})
        monkeypatch.setattr(cleanup, "saved_texts", lambda path: [("An old paragraph.", "")])
        assert client.get(f"/api/jobs/{job['id']}/cache").json()["entries"] == 1
        client.delete(f"/api/jobs/{job['id']}/cache")
        assert cache.get_many(tr.cache_key, ["An old paragraph."]) == {}


def test_clearing_unused_cache_keeps_what_records_use(app_env: ModuleType) -> None:
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        kept = wait_done(client, submit(client)["id"])
        other = b"# Another\n\nA paragraph of a book deleted later on.\n"
        gone = wait_done(client, submit(client, name="gone.md", data=other)["id"])
        # Removed without clearing its cache, as records deleted before clearing existed were
        state.jobs.delete(gone["id"])
        r = client.post("/api/web/translate", json={"texts": ["Some text from a web page."], "service_id": "mock"})
        assert r.status_code == 200
        # Saved before the cache recorded owners: nothing tells whose it is, so it stays
        cache.put_many("old-prefix", {"An untracked paragraph.": "旧译文"})
        before = _rows(cache)

        info = client.get("/api/cache/unused").json()
        assert info == {"removed": 3, "untracked": 1}  # the deleted book's title and paragraph, and the page
        assert _rows(cache) == before  # counting deletes nothing
        assert client.delete("/api/cache/unused").json() == {"removed": info["removed"], "untracked": 1}
        assert _rows(cache) == before - info["removed"]
        assert cache.get_many("old-prefix", ["An untracked paragraph."]) == {"An untracked paragraph.": "旧译文"}
        assert client.get(f"/api/jobs/{kept['id']}/cache").json()["entries"] == 3
        assert client.get("/api/cache/unused").json() == {"removed": 0, "untracked": 1}


def test_clearing_unused_cache_waits_for_running_jobs(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        _job(state.jobs, job["id"]).status = "running"
        assert client.delete("/api/cache/unused").status_code == 409


def test_document_digest_is_stored_on_the_job(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        assert _job(state.jobs, job["id"]).document == hashlib.sha1(MD).hexdigest()


def test_redo_from_scratch_clears_stale_entries_first(app_env: ModuleType) -> None:
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        doc = _job(state.jobs, job["id"]).document
        # A paragraph cut by old code and no longer used
        cache.put_many("old-prefix", {"A stale paragraph.": "旧译文"})
        cache.claim(doc, "old-prefix", ["A stale paragraph."])
        r = client.post(f"/api/jobs/{job['id']}/retry", json={"fresh": True})
        assert r.status_code == 200, r.text
        again = wait_done(client, job["id"])
        assert again["status"] == "done" and again["outputs"]
        assert cache.get_many("old-prefix", ["A stale paragraph."]) == {}
        assert _rows(cache) == 3


PDF_TEXT = "A body paragraph long enough to be translated here."


def _pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page(width=420, height=240).insert_text((40, 60), PDF_TEXT, fontsize=10)
    data: bytes = doc.tobytes()
    return data


def test_jobs_of_the_same_book_share_the_layout_analysis(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.formats import pdf_units
    from app.formats.pdf_extract import _extract

    main = app_env
    calls: list[int] = []
    monkeypatch.setattr(pdf_units, "_extract", _counted(_extract, calls))
    data = _pdf()
    with TestClient(main.app) as client:
        a = wait_done(client, submit(client, name="paper.pdf", data=data)["id"])
        b = wait_done(client, submit(client, name="copy.pdf", data=data, mode="translated")["id"])
        assert a["status"] == b["status"] == "done" and len(calls) == 1
        plan = _plan_path(_job(state.jobs, a["id"]))
        assert plan == _plan_path(_job(state.jobs, b["id"])) and plan.is_file()
        # Delete one; the other still uses the analysis result, so it stays; delete the last one too, and it goes
        client.delete(f"/api/jobs/{a['id']}")
        assert plan.is_file()
        client.delete(f"/api/jobs/{b['id']}")
        assert not plan.parent.exists()


def test_layout_saved_in_the_job_folder_moves_to_the_shared_place(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        job = _job(state.jobs, wait_done(client, submit(client)["id"])["id"])
        folder = state.jobs.dir(job.id)
        (folder / "plan.pickle").write_bytes(b"plan")
        (folder / "boxes.json").write_text("{}")
        plan = _plan_path(job)
        assert plan.read_bytes() == b"plan" and (plan.parent / "boxes.json").is_file()
        assert not (folder / "plan.pickle").exists() and not (folder / "boxes.json").exists()


def test_layouts_no_job_uses_are_pruned(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        job = _job(state.jobs, wait_done(client, submit(client)["id"])["id"])
        used = _layout_dir(job)
        stale = state.jobs.layout_root / ("0" * 40)
        stale.mkdir()
        (stale / "plan.pickle").write_bytes(b"old")
        assert cleanup.prune_layouts(state.jobs) == [stale.name]
        assert used.is_dir() and not stale.exists()


def test_the_same_file_uploaded_twice_is_stored_once(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        a = wait_done(client, submit(client)["id"])
        b = wait_done(client, submit(client, name="again.md")["id"])
        first = state.jobs.source_path(_job(state.jobs, a["id"]))
        second = state.jobs.source_path(_job(state.jobs, b["id"]))
        assert first.stat().st_ino == second.stat().st_ino and second.read_bytes() == MD
        usage = client.get("/api/jobs").json()["disk_usage"]
        assert usage == state.jobs.disk_usage()
        client.delete(f"/api/jobs/{a['id']}")
        assert second.read_bytes() == MD
        # Another file with different content is not affected
        c = wait_done(client, submit(client, name="other.md", data=MD + b"\nMore.\n")["id"])
        assert state.jobs.source_path(_job(state.jobs, c["id"])).stat().st_ino != second.stat().st_ino


def test_layouts_left_in_job_folders_are_adopted_for_new_jobs(app_env: ModuleType) -> None:
    main = app_env
    with TestClient(main.app) as client:
        old = _job(state.jobs, wait_done(client, submit(client)["id"])["id"])
        # An old record from before the shared directory: its layout result is still in the record directory
        (state.jobs.dir(old.id) / "boxes.json").write_text("{}")
        shared = state.jobs.layout_root / old.document
        assert state.jobs.adopt_layouts() == 1
        assert (shared / "boxes.json").is_file() and not (state.jobs.dir(old.id) / "boxes.json").exists()
