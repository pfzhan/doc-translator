"""Editing the translation of a paragraph in the reader (PUT /api/jobs/{id}/segments/{index})."""

from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from app import state
from app.edits import EditedCache, EditRejected, with_placeholders
from app.translators import MockTranslator
from tests.test_history import JobJSON, _pending_future, submit, wait_done
from tests.test_preview import MemCache

FIRST = "First paragraph of the document is here."
SECOND = "Second paragraph goes right after it."
MD = f"# A Title\n\n{FIRST}\n\n{SECOND}\n\n{FIRST}\n".encode()
EDIT = "这是人工改过的第一段译文。"


def _file(client: TestClient, job: JobJSON, variant: str) -> str:
    url = next(o["url"] for o in job["outputs"] if f".{variant}." in o["name"])
    text: str = client.get(url).text
    return text


def _index(client: TestClient, job_id: str, source: str) -> int:
    segments = client.get(f"/api/jobs/{job_id}/preview").json()["segments"]
    return next(i for i, s in enumerate(segments) if s["s"] == source)


def test_an_edit_rewrites_the_files_from_the_cache(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    original = MockTranslator.translate_batch

    async def counting(self: MockTranslator, texts: list[str]) -> list[str]:
        calls.append(texts)
        return await original(self, texts)

    monkeypatch.setattr(MockTranslator, "translate_batch", counting)
    with TestClient(app_env.app) as client:
        job = wait_done(client, submit(client, data=MD)["id"])
        with_variant = client.post(f"/api/jobs/{job['id']}/variant").json()
        calls.clear()
        index = _index(client, job["id"], FIRST)

        r = client.put(f"/api/jobs/{job['id']}/segments/{index}", json={"text": f"  {EDIT}\n"})
        assert r.status_code == 200, r.text
        body = r.json()
        # Both places with the same sentence read the edit; the service was not asked
        assert body["updates"] == [[index, EDIT, 0], [index + 2, EDIT, 0]]
        assert calls == []
        for variant in ("bilingual", "translated"):
            assert _file(client, with_variant, variant).count(EDIT) == 2
        assert SECOND in _file(client, with_variant, "bilingual")
        # The reader shows it after a reload
        preview = client.get(f"/api/jobs/{job['id']}/preview").json()
        assert [index, EDIT, 0] in preview["updates"]
        assert job["id"] not in state.variant_tasks

        # The edit is in the cache under the job's key: translating again keeps it without a request
        again = wait_done(client, client.post(f"/api/jobs/{job['id']}/retry", json={}).json()["id"])
        assert again["status"] == "done" and calls == []
        assert _file(client, again, "bilingual").count(EDIT) == 2


def test_an_edit_is_checked(app_env: ModuleType) -> None:
    with TestClient(app_env.app) as client:
        job = wait_done(client, submit(client, data=MD)["id"])
        url = f"/api/jobs/{job['id']}/segments"
        index = _index(client, job["id"], FIRST)
        before = _file(client, job, "bilingual")
        assert client.put(f"{url}/{index}", json={"text": "  "}).status_code == 400
        assert client.put(f"{url}/{index}", json={"text": "x" * 20001}).status_code == 422
        assert client.put(f"{url}/{index}", json={}).status_code == 422
        assert client.put(f"{url}/99", json={"text": EDIT}).status_code == 404
        assert client.put(f"{url}/-1", json={"text": EDIT}).status_code == 404
        assert client.put(f"/api/jobs/missing/segments/{index}", json={"text": EDIT}).status_code == 404
        # The paragraph has no formula or image to put back
        r = client.put(f"{url}/{index}", json={"text": "公式 {v1} 不存在"})
        assert r.status_code == 400 and "{v1}" in r.json()["detail"]

        state.variant_tasks[job["id"]] = _pending_future()
        assert client.put(f"{url}/{index}", json={"text": EDIT}).status_code == 409
        state.variant_tasks.pop(job["id"])
        stored = state.jobs.get(job["id"])
        assert stored is not None
        stored.status = "running"
        assert client.put(f"{url}/{index}", json={"text": EDIT}).status_code == 400
        stored.status = "done"
        assert _file(client, job, "bilingual") == before


def test_an_edit_in_a_pdf(app_env: ModuleType) -> None:
    import pymupdf

    body = "A body paragraph long enough to be translated here."
    doc = pymupdf.open()
    doc.new_page(width=420, height=240).insert_text((40, 60), body, fontsize=10)
    with TestClient(app_env.app) as client:
        job = wait_done(client, submit(client, name="paper.pdf", data=doc.tobytes())["id"])
        assert job["status"] == "done"
        index = _index(client, job["id"], body)
        r = client.put(f"/api/jobs/{job['id']}/segments/{index}", json={"text": "正文段落的人工译文"})
        assert r.status_code == 200, r.text
        url = r.json()["job"]["outputs"][0]["url"]
        written = pymupdf.open("pdf", client.get(url).content)
        assert "正文段落的人工译文" in "".join(page.get_text() for page in written.pages())


def test_with_placeholders_maps_them_by_order_of_appearance() -> None:
    # The reader numbers them as they appear; the cache has the model's own numbers
    assert with_placeholders("先 {v2} 后 {v1}", "{v2} 在后，{v1} 在前") == "{v1} 在后，{v2} 在前"
    assert with_placeholders("没有公式", "仍然没有") == "仍然没有"
    for bad in ("只有 {v1}", "{v1} {v1} {v2}", "{v1} {v3}"):
        with pytest.raises(EditRejected):
            with_placeholders("{v1} 和 {v2}", bad)
    with pytest.raises(EditRejected):
        with_placeholders("x\x01s\x022\x01/s\x02", "x2")


def test_edited_cache_swaps_the_paragraph_until_committed() -> None:
    inner = MemCache({"Hello": "你好", "Hello\n\n[note] greeting": "喂", "Other": "其他"})
    cache = EditedCache(inner, "Hello", "您好")
    found = cache.get_many("p", ["Hello", "Hello\n\n[note] greeting", "Other", "Missing"])
    assert found == {"Hello": "您好", "Hello\n\n[note] greeting": "您好", "Other": "其他"}
    # The document glossary is never the paragraph's
    assert cache.get_many("p:terms", ["Hello"]) == {"Hello": "你好"}
    # What the run stores for the edited keys waits for commit; other entries go through
    cache.put_many("p", {"Hello": "您好！", "Other": "另一个"})
    assert inner.d["Hello"] == "你好" and inner.d["Other"] == "另一个"
    cache.commit()
    assert inner.d["Hello"] == "您好！" and inner.d["Hello\n\n[note] greeting"] == "您好"
