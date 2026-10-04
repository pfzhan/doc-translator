import asyncio
import io
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.runner import Runner
from app.translators import MockTranslator

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class MemCache:
    def __init__(self, data=None):
        self.d = data or {}

    def get_many(self, prefix, texts):
        return {t: self.d[t] for t in texts if t in self.d}

    def put_many(self, prefix, pairs):
        self.d.update(pairs)


class RecordingTranslator(MockTranslator):
    """记录每批翻译的顺序，可以在批次之间插入回调（模拟用户在翻译过程中跳转）。"""

    max_batch_items = 2
    concurrency = 1

    def __init__(self, *a, on_batch=None, **kw):
        super().__init__(*a, **kw)
        self.batches = []
        self.on_batch = on_batch

    async def translate_batch(self, texts):
        self.batches.append(list(texts))
        if self.on_batch:
            self.on_batch(len(self.batches))
        await asyncio.sleep(0)
        return await super().translate_batch(texts)


def paras(n):
    return [f"Paragraph number {i} talks about something quite different from the others." for i in range(n)]


def test_preview_incremental_and_duplicates():
    texts = ["Title of the book", "Hello there my friend", "", "Hello there my friend"]
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    assert runner.preview() == {"ready": False, "version": 0}
    asyncio.run(runner.translate_all(texts, kinds=["h1", "p", "p", "p"], preview=True))

    full = runner.preview()
    assert [s["k"] for s in full["segments"]] == ["h1", "p", "p", "p"]
    # 同一句出现两次，两个位置都会更新；空段落不出现在更新里
    assert sorted(u[0] for u in full["updates"]) == [0, 1, 3]
    assert full["version"] == 2  # 两条不同的原文

    assert runner.preview(since=full["version"])["updates"] == []
    partial = runner.preview(since=1)
    assert "segments" not in partial and len(partial["updates"]) == 2


def test_cached_and_skipped_segments_show_immediately():
    texts = ["Cached sentence here", "这一段本来就是中文，不需要再翻译成中文了。", "Fresh sentence to translate now"]
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache({"Cached sentence here": "已缓存"}))
    asyncio.run(runner.translate_all(texts, preview=True))
    log = runner._log
    # 缓存命中和已是目标语言的段落排在最前面，不用等翻译
    assert set(log[:2]) == {texts[0], texts[1]}
    updates = {u[0]: u for u in runner.preview()["updates"]}
    assert updates[0][1] == "已缓存" and updates[1][2] == 1 and updates[2][1].startswith("[zh-CN]")


def test_first_batch_is_small():
    tr = RecordingTranslator("zh-CN")
    tr.max_batch_items = 20
    asyncio.run(Runner(tr, cache=MemCache(), skip_same_lang=False).translate_all(paras(30)))
    assert [len(b) for b in tr.batches] == [4, 20, 6]


def test_focus_reorders_remaining_batches():
    texts = paras(20)
    runner = None

    def jump(n):
        if n == 1:
            runner.focus(14)  # 第一批翻译时用户跳到了第 14 段

    tr = RecordingTranslator("zh-CN", on_batch=jump)
    runner = Runner(tr, cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all(texts, preview=True))
    order = [texts.index(b[0]) for b in tr.batches]
    assert order[0] == 0
    assert order[1] == 14  # 跳转后先翻关注位置
    assert order[1:4] == [14, 16, 18]  # 然后顺着往后翻
    assert all(o.startswith("[zh-CN]") for o in out)  # 最终全部翻完
    assert sum(len(b) for b in tr.batches) == 20


def test_preview_endpoints(app_env):
    md = b"# Title here\n\nFirst paragraph of the document.\n\n- item one in list\n"
    # with 块里共用一个事件循环，后台翻译任务才能在两次请求之间继续跑
    with TestClient(main.app) as client:
        r = client.post("/api/jobs", files={"file": ("a.md", md)}, data={"service_id": "mock", "target_lang": "zh-CN"})
        job = r.json()
        assert job["preview"] is True
        # 等到任务结束（最多 30 秒），整套测试一起跑时机器负载高，不能用很短的固定超时
        deadline = time.monotonic() + 30
        status = None
        while time.monotonic() < deadline:
            status = client.get(f"/api/jobs/{job['id']}").json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert status["status"] == "done", status
        p = client.get(f"/api/jobs/{job['id']}/preview").json()
        assert p["ready"] and p["status"] == "done", p
        assert [s["k"] for s in p["segments"]] == ["h1", "p", "li"]
        assert len(p["updates"]) == 3
        assert client.post(f"/api/jobs/{job['id']}/focus", json={"index": 2}).json() == {"ok": True}
        assert client.post(f"/api/jobs/{job['id']}/focus", json={"index": "x"}).status_code == 400
        assert client.get("/api/jobs/nope/preview").status_code == 404

        pdf = client.post("/api/jobs", files={"file": ("a.pdf", b"%PDF-1.4")}, data={"service_id": "mock"}).json()
        assert pdf["preview"] is True


def test_epub_preview_kinds_in_reading_order():
    src = SAMPLES / "alice.mobi"
    if not src.exists():
        pytest.skip("sample missing")
    from app.formats import translate_file

    runner = Runner(MockTranslator("zh-CN"), cache=MemCache())
    out = Path(__import__("tempfile").mkdtemp())
    asyncio.run(translate_file(src, out, runner, True, "zh-CN"))
    segs = runner.preview()["segments"]
    kinds = {s["k"] for s in segs}
    assert "toc" in kinds and "p" in kinds
    # MOBI7 的裸文本段落按阅读顺序排列：正文第一句出现在“CHAPTER I”之后
    # 富文本段的 s 带行内标签，匹配前剥掉（前端预览同样处理）
    texts = [__import__("re").sub(r"<[^>]+>", "", s["s"]).strip() for s in segs]
    ch1 = next(i for i, t in enumerate(texts) if t.startswith("CHAPTER I.") and "Rabbit" in t)
    alice = next(i for i, t in enumerate(texts) if t.startswith("Alice was beginning"))
    assert ch1 < alice


def _heading_pdf() -> bytes:
    """正文 10pt，标题 17pt，章节名 12pt 粗体，一句 12pt 非粗体，作者名 10pt 粗体。"""
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((40, 40), "Attention Is All You Need", fontsize=17, fontname="hebo")
    page.insert_text((40, 80), "Introduction", fontsize=12, fontname="hebo")
    page.insert_text((40, 110), "scholarly works.", fontsize=12, fontname="helv")
    page.insert_text((40, 140), "Jakob Uszkoreit", fontsize=10, fontname="hebo")
    y = 180
    for i in range(6):
        page.insert_text(
            (40, y),
            f"This is body paragraph number {i} with enough words to look like a normal line of text.",
            fontsize=10,
            fontname="helv",
        )
        y += 28
    return doc.tobytes()


def test_pdf_preview_marks_section_headings(tmp_path):
    """略大于正文的粗体短块是小标题；同字号的非粗体、以及不大于正文的粗体仍是段落。"""
    from app.formats import translate_file

    src = tmp_path / "paper.pdf"
    src.write_bytes(_heading_pdf())
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache())
    asyncio.run(translate_file(src, tmp_path, runner, False, "zh-CN"))
    kinds = {s["s"]: s["k"] for s in runner.preview()["segments"]}
    assert kinds["Attention Is All You Need"] == "h2"
    assert kinds["Introduction"] == "h2"
    assert kinds["scholarly works."] == "p"
    assert kinds["Jakob Uszkoreit"] == "p"
    body = [s for s in runner.preview()["segments"] if s["s"].startswith("This is body")]
    assert body and all(s["k"] == "p" for s in body)


def _wait_terminal(client, job_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_pdf_preview_headings_survive_the_job(app_env, tmp_path):
    main = app_env
    src = tmp_path / "paper.pdf"
    src.write_bytes(_heading_pdf())
    with TestClient(main.app) as client:
        job = client.post(
            "/api/jobs",
            files={"file": ("paper.pdf", src.read_bytes())},
            data={"service_id": "mock", "target_lang": "zh-CN"},
        ).json()
        done = _wait_terminal(client, job["id"])
        assert done["status"] == "done" and done["preview"] is True
        kinds = {s["s"]: s["k"] for s in client.get(f"/api/jobs/{job['id']}/preview").json()["segments"]}
        assert kinds["Introduction"] == "h2"
        assert kinds["scholarly works."] == "p"


def test_failed_pdf_stays_previewable_but_not_ready(app_env):
    """加密或没有文字的 PDF 预览标记为真，但不会有段落；客户端不该为此继续轮询。"""
    main = app_env
    empty = pymupdf.open()
    empty.new_page()
    locked = pymupdf.open()
    locked.new_page().insert_text((72, 72), "Secret text that cannot be read")
    buf = io.BytesIO()
    locked.save(buf, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    cases = [("empty.pdf", empty.tobytes()), ("locked.pdf", buf.getvalue()), ("bad.pdf", b"%PDF-1.4")]
    with TestClient(main.app) as client:
        for name, data in cases:
            created = client.post("/api/jobs", files={"file": (name, data)}, data={"service_id": "mock"}).json()
            assert created["preview"] is True
            job = _wait_terminal(client, created["id"])
            assert job["status"] == "error", (name, job)
            preview = client.get(f"/api/jobs/{job['id']}/preview").json()
            assert preview["ready"] is False and preview["status"] == "error"
