"""清理一条记录的缓存痕迹：缓存归属登记、按归属清理、接口和删除记录时的连带清理。"""
import asyncio
import hashlib

from fastapi.testclient import TestClient

from app import cleanup, runner
from app.runner import Cache, Runner
from app.translators import MockTranslator
from tests.test_history import MD, submit, wait_done


def _rows(cache: Cache) -> int:
    return cache.conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]


def test_cache_purges_only_entries_claimed_by_the_document(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3")
    cache.put_many("p", {"a": "A", "b": "B", "c": "C"})
    cache.claim("doc1", "p", ["a", "b"])
    cache.claim("doc2", "p", ["c"])
    owned = cache.owned("doc1")
    assert owned == Cache.keys("p", ["a", "b"])
    assert cache.purge("doc1", owned) == 2
    assert cache.get_many("p", ["a", "b", "c"]) == {"c": "C"}
    assert cache.owned("doc1") == set() and cache.owned("doc2") == Cache.keys("p", ["c"])


def test_owned_includes_extra_keys_still_in_cache(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3")
    cache.put_many("p", {"old": "O"})
    extra = Cache.keys("p", ["old", "gone"])
    assert cache.owned("doc", extra) == Cache.keys("p", ["old"])


def test_dropped_entries_lose_their_owner(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3")
    cache.put_many("p", {"a": "A"})
    cache.claim("doc", "p", ["a"])
    cache.drop_many("p", ["a"])
    assert cache.conn.execute("SELECT COUNT(*) FROM owner").fetchone()[0] == 0


def test_runner_claims_written_and_hit_entries(tmp_path):
    cache = Cache(tmp_path / "c.sqlite3")
    tr = MockTranslator("zh-CN")
    first = Runner(tr, cache=cache, skip_same_lang=False)
    first.set_document("doc1")
    asyncio.run(first.translate_all(["Hello world, this is a sentence."]))
    assert len(cache.owned("doc1")) == 1
    # 另一份文件命中同一条缓存，也登记到它名下
    second = Runner(tr, cache=cache, skip_same_lang=False)
    second.set_document("doc2")
    asyncio.run(second.translate_all(["Hello world, this is a sentence."]))
    assert cache.owned("doc2") == cache.owned("doc1")


def test_clearing_a_job_removes_its_cache_and_generated_files(app_env):
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        folder = main.jobs.dir(job["id"])
        boxes = main.jobs.layout_dir(main.jobs.get(job["id"])) / "boxes.json"
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

        # 记录还能重新开始翻译
        client.post(f"/api/jobs/{job['id']}/retry")
        again = wait_done(client, job["id"])
        assert again["status"] == "done" and again["outputs"] and _rows(cache) == 3


def test_clearing_can_include_model_boxes(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        boxes = main.jobs.layout_dir(main.jobs.get(job["id"])) / "boxes.json"
        boxes.write_text("{}")
        assert client.delete(f"/api/jobs/{job['id']}/cache?boxes=true").status_code == 200
        assert not boxes.exists()


def test_clearing_is_refused_while_translating(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        main.jobs.get(job["id"]).status = "running"
        r = client.delete(f"/api/jobs/{job['id']}/cache")
        assert r.status_code == 409


def test_jobs_of_the_same_file_share_cache(app_env):
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        a = wait_done(client, submit(client)["id"])
        b = wait_done(client, submit(client)["id"])
        assert client.get(f"/api/jobs/{a['id']}/cache").json()["shared"] == 1
        # 删掉其中一条，另一条还在用，缓存留着
        client.delete(f"/api/jobs/{a['id']}")
        assert _rows(cache) == 3
        # 最后一条也删掉，缓存跟着清掉
        client.delete(f"/api/jobs/{b['id']}")
        assert _rows(cache) == 0


def test_entries_saved_before_ownership_are_found_from_the_saved_plan(app_env, monkeypatch):
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        # 模拟登记归属之前存下的缓存：去掉登记，再多存一条只在上次版面分析里出现过的旧段落
        cache.conn.execute("DELETE FROM owner")
        tr = MockTranslator("zh-CN")
        tr.service_id = "mock"
        tr.detected_source = "en"
        cache.put_many(tr.cache_key, {"An old paragraph.": "旧译文"})
        monkeypatch.setattr(cleanup, "saved_texts", lambda path: [("An old paragraph.", "")])
        assert client.get(f"/api/jobs/{job['id']}/cache").json()["entries"] == 1
        client.delete(f"/api/jobs/{job['id']}/cache")
        assert cache.get_many(tr.cache_key, ["An old paragraph."]) == {}


def test_document_digest_is_stored_on_the_job(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        assert main.jobs.get(job["id"]).document == hashlib.sha1(MD).hexdigest()


def test_redo_from_scratch_clears_stale_entries_first(app_env):
    main = app_env
    cache = runner.get_cache()
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        doc = main.jobs.get(job["id"]).document
        # 旧代码切出来、现在已经不用的段落
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
    return doc.tobytes()


def test_jobs_of_the_same_book_share_the_layout_analysis(app_env, monkeypatch):
    from app.formats import pdf

    main = app_env
    calls = []
    real_extract = pdf._extract
    monkeypatch.setattr(pdf, "_extract", lambda *a, **k: calls.append(1) or real_extract(*a, **k))
    data = _pdf()
    with TestClient(main.app) as client:
        a = wait_done(client, submit(client, name="paper.pdf", data=data)["id"])
        b = wait_done(client, submit(client, name="copy.pdf", data=data, mode="translated")["id"])
        assert a["status"] == b["status"] == "done" and len(calls) == 1
        plan = main.jobs.plan_path(main.jobs.get(a["id"]))
        assert plan == main.jobs.plan_path(main.jobs.get(b["id"])) and plan.is_file()
        # 删掉一条，另一条还在用，分析结果留着；最后一条也删掉，一起删
        client.delete(f"/api/jobs/{a['id']}")
        assert plan.is_file()
        client.delete(f"/api/jobs/{b['id']}")
        assert not plan.parent.exists()


def test_layout_saved_in_the_job_folder_moves_to_the_shared_place(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = main.jobs.get(wait_done(client, submit(client)["id"])["id"])
        folder = main.jobs.dir(job.id)
        (folder / "plan.pickle").write_bytes(b"plan")
        (folder / "boxes.json").write_text("{}")
        plan = main.jobs.plan_path(job)
        assert plan.read_bytes() == b"plan" and (plan.parent / "boxes.json").is_file()
        assert not (folder / "plan.pickle").exists() and not (folder / "boxes.json").exists()


def test_layouts_no_job_uses_are_pruned(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = main.jobs.get(wait_done(client, submit(client)["id"])["id"])
        used = main.jobs.layout_dir(job)
        stale = main.jobs.layout_root / ("0" * 40)
        stale.mkdir()
        (stale / "plan.pickle").write_bytes(b"old")
        assert cleanup.prune_layouts(main.jobs) == [stale.name]
        assert used.is_dir() and not stale.exists()


def test_the_same_file_uploaded_twice_is_stored_once(app_env):
    main = app_env
    with TestClient(main.app) as client:
        a = wait_done(client, submit(client)["id"])
        b = wait_done(client, submit(client, name="again.md")["id"])
        first = main.jobs.source_path(main.jobs.get(a["id"]))
        second = main.jobs.source_path(main.jobs.get(b["id"]))
        assert first.stat().st_ino == second.stat().st_ino and second.read_bytes() == MD
        usage = client.get("/api/jobs").json()["disk_usage"]
        assert usage == main.jobs.disk_usage()
        client.delete(f"/api/jobs/{a['id']}")
        assert second.read_bytes() == MD
        # 另一份内容不同的文件不受影响
        c = wait_done(client, submit(client, name="other.md", data=MD + b"\nMore.\n")["id"])
        assert main.jobs.source_path(main.jobs.get(c["id"])).stat().st_ino != second.stat().st_ino
