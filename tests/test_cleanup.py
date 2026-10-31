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
        (folder / "boxes.json").write_text("{}")
        info = client.get(f"/api/jobs/{job['id']}/cache").json()
        assert info["entries"] == 3 and info["outputs"] == 1 and info["shared"] == 0
        assert info["generated_bytes"] > 0 and info["boxes_bytes"] == 2

        r = client.delete(f"/api/jobs/{job['id']}/cache")
        assert r.status_code == 200, r.text
        cleared = r.json()
        assert cleared["removed"] == 3 and cleared["status"] == "idle"
        assert cleared["outputs"] == [] and cleared["total"] == 0
        assert _rows(cache) == 0
        assert sorted(p.name for p in folder.iterdir()) == ["book.md", "boxes.json", "job.json", "out"]
        assert not any((folder / "out").iterdir())

        # 记录还能重新开始翻译
        client.post(f"/api/jobs/{job['id']}/retry")
        again = wait_done(client, job["id"])
        assert again["status"] == "done" and again["outputs"] and _rows(cache) == 3


def test_clearing_can_include_model_boxes(app_env):
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        boxes = main.jobs.dir(job["id"]) / "boxes.json"
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
