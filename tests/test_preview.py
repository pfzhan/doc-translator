import asyncio
import io
import re
import time
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main, state
from app.history import Job
from app.runner import Runner
from app.translators import MockTranslator, TranslatorError

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class MemCache:
    def __init__(self, data=None):
        self.d = data or {}

    def get_many(self, prefix, texts):
        return {t: self.d[t] for t in texts if t in self.d}

    def put_many(self, prefix, pairs):
        self.d.update(pairs)

    def drop_many(self, prefix, texts):
        for text in texts:
            self.d.pop(text, None)


class RecordingTranslator(MockTranslator):
    """Records the order of translated batches and can run a callback between batches (simulating a user who jumps
    elsewhere during translation)."""

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
    # The same sentence appears twice and both positions are updated; empty paragraphs do not appear in the updates
    assert sorted(u[0] for u in full["updates"]) == [0, 1, 3]
    assert full["version"] == 2  # Two distinct source texts

    assert runner.preview(since=full["version"])["updates"] == []
    partial = runner.preview(since=1)
    assert "segments" not in partial and len(partial["updates"]) == 2


def test_positional_skip_does_not_suppress_the_same_sentence():
    cite = "Smith, J. (2020). Some book. Press."
    texts = [cite, "Bibliography", cite]
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all(texts, skip=[False, False, True], preview=True))
    assert out[0].startswith("[zh-CN] ") and out[2] == cite
    updates = {u[0]: u for u in runner.preview()["updates"]}
    assert updates[0][2] == 0 and updates[0][1].startswith("[zh-CN]")
    assert updates[2][2] == 1 and updates[2][1] == cite
    assert runner.info["skipped"] == 1


def test_cached_and_skipped_segments_show_immediately():
    texts = ["Cached sentence here", "这一段本来就是中文，不需要再翻译成中文了。", "Fresh sentence to translate now"]
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache({"Cached sentence here": "已缓存"}))
    asyncio.run(runner.translate_all(texts, preview=True))
    log = runner._log
    # Cache hits and paragraphs already in the target language come first without waiting for translation
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
            runner.focus(14)  # The user jumped to paragraph 14 during the first batch

    tr = RecordingTranslator("zh-CN", on_batch=jump)
    runner = Runner(tr, cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all(texts, preview=True))
    order = [texts.index(b[0]) for b in tr.batches]
    assert order[0] == 0
    assert order[1] == 14  # After the jump the focused position is translated first
    assert order[1:4] == [14, 16, 18]  # Then translation continues forward from there
    assert all(o.startswith("[zh-CN]") for o in out)  # Everything is translated in the end
    assert sum(len(b) for b in tr.batches) == 20


def test_preview_endpoints(app_env):
    md = b"# Title here\n\nFirst paragraph of the document.\n\n- item one in list\n"
    # The with block shares one event loop, so the background translation job keeps running between two requests
    with TestClient(main.app) as client:
        r = client.post("/api/jobs", files={"file": ("a.md", md)}, data={"service_id": "mock", "target_lang": "zh-CN"})
        job = r.json()
        assert job["preview"] is True
        # Wait until the job ends (at most 30 seconds); machine load is high when the whole suite runs, so a short
        # fixed timeout does not work
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
    # Plain-text MOBI7 paragraphs are in reading order: the first body sentence comes after "CHAPTER I"
    # The s of a rich-text segment carries inline tags, which are stripped before matching (the frontend preview
    # does the same)
    texts = [__import__("re").sub(r"<[^>]+>", "", s["s"]).strip() for s in segs]
    ch1 = next(i for i, t in enumerate(texts) if t.startswith("CHAPTER I.") and "Rabbit" in t)
    alice = next(i for i, t in enumerate(texts) if t.startswith("Alice was beginning"))
    assert ch1 < alice


def _heading_pdf() -> bytes:
    """Body 10pt, title 17pt, section name 12pt bold, one sentence 12pt non-bold, author name 10pt bold."""
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
    """A short bold block slightly larger than body text is a section heading; non-bold text of the same size and bold
    text no larger than body text are still paragraphs."""
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
    """An encrypted PDF or one without text has the preview flag set but no paragraphs; the client must not keep
    polling for them."""
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


def test_normalize_numbered_units():
    from app.runner import normalize_numbered_units as norm

    texts = ["h1", "toc1", "toc2", "body", "cite", "long"]
    results = [
        "第2卷",
        "第3卷　论三种政体的原则",
        "第卷26　论法律应当与其所规范的事物秩序具有的关系",
        "第一章",
        "《Science》307卷第5708期（2005）：414–16",  # Does not start with "第", left alone
        "第4卷　" + "很长的条目" * 10,  # Too long, left alone
    ]
    fixed = norm(texts, results, "zh-CN")
    assert fixed["h1"] == "第二卷"
    assert fixed["toc1"] == "第三卷　论三种政体的原则"
    assert fixed["toc2"] == "第二十六卷　论法律应当与其所规范的事物秩序具有的关系"
    assert "body" not in fixed and "cite" not in fixed and "long" not in fixed
    assert norm(texts, results, "en") == {}
    from app.runner import fix_numbered_unit as fix

    assert fix("第110章", "zh-CN") == "第一百一十章"
    assert fix("第111章", "zh-CN") == "第一百一十一章"
    assert fix("第210卷", "zh-CN") == "第二百一十卷"
    assert fix("第1010章", "zh-CN") == "第一千零一十章"
    assert fix("第1110章", "zh-CN") == "第一千一百一十章"
    assert fix("第10章", "zh-CN") == "第十章"
    # Volume, issue and page numbers keep Arabic digits
    for cite in ("第25卷第1期", "第25卷第1期（2005）：414–16", "第307卷第5708期", "第2章第3页"):
        assert fix(cite, "zh-CN") == cite


def test_normalize_numbered_units_updates_cache():
    from app.runner import Runner

    class DictTranslator(MockTranslator):
        async def translate_batch(self, texts):
            return ["第2卷" if t == "BOOK 2" else f"[zh-CN] {t}" for t in texts]

    runner = Runner(DictTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all(["BOOK 2", "para one"], kinds=["h1", "p"], preview=True))
    assert out[0] == "第二卷"
    # The cache also stores the normalized version, so the preview and the history snapshot agree
    assert runner.cache.get_many("x", []) == {}
    assert runner._done["BOOK 2"] == "第二卷"
    # Body text is not changed even when the translation is "第2卷", and no rewritten cache entry is written
    body = Runner(DictTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    assert asyncio.run(body.translate_all(["BOOK 2"], kinds=["p"], preview=True)) == ["第2卷"]


def test_normalize_applies_to_cache_hits():
    """Old-style translations in the cache (translated before normalization) are rewritten in place on a hit, and the
    preview shows the new style at once."""
    from app.runner import Runner

    class NeverCalled(MockTranslator):
        async def translate_batch(self, texts):
            raise AssertionError("全部命中缓存，不该发请求")

    cache = MemCache({"BOOK 2": "第2卷", "BOOK 26": "第卷26", "cite": "第25卷第1期"})
    runner = Runner(NeverCalled("zh-CN"), cache=cache, skip_same_lang=False)
    out = asyncio.run(runner.translate_all(["BOOK 2", "BOOK 26", "cite"], kinds=["h1", "toc", "h2"], preview=True))
    assert out == ["第二卷", "第二十六卷", "第25卷第1期"]
    assert cache.d["BOOK 2"] == "第二卷" and cache.d["BOOK 26"] == "第二十六卷"
    assert cache.d["cite"] == "第25卷第1期"
    updates = {u[0]: u[1] for u in runner.preview()["updates"]}
    assert updates[0] == "第二卷" and updates[1] == "第二十六卷" and updates[2] == "第25卷第1期"


def test_normalize_punctuation():
    """Full-width punctuation in Chinese translations becomes half-width (comma-like marks get a following space);
    full stops, title marks and quotation marks stay full-width. New translations and cache hits are both rewritten;
    other languages are left alone."""
    from app.runner import Runner, clean_translation, normalize_punctuation

    assert normalize_punctuation("词元（token）的概率", "zh-CN") == "词元(token)的概率"
    assert normalize_punctuation("训练时，我们随机决定，是否采样", "zh-CN") == "训练时, 我们随机决定, 是否采样"
    assert normalize_punctuation("已有空格， 不重复", "zh-CN") == "已有空格, 不重复"
    assert normalize_punctuation("甲、乙；丙：丁？戊！己", "zh-CN") == "甲, 乙; 丙: 丁? 戊! 己"
    # Full stops, title marks and quotation marks stay full-width
    assert normalize_punctuation("完了。《论文》里说“保持”。", "zh-CN") == "完了。《论文》里说“保持”。"
    # The space added after sentence-final punctuation does not remain at the end
    assert normalize_punctuation("结束了吗？", "zh-CN") == "结束了吗?"
    assert normalize_punctuation("（保持），不变", "en") == "（保持），不变"
    # Section numbers read out in Chinese are restored to the digits of the source
    assert clean_translation("introduced in Section 2.1", "我们将在第二点一节介绍", "zh-CN") == "我们将在第2.1节介绍"
    assert clean_translation("Section 2.2 Hadoop", "第二点二节 Hadoop", "zh-CN") == "第2.2节 Hadoop"
    # An ellipsis between list separators or operators is a mathematical ellipsis and is written once; "……" at the
    # end of a sentence or utterance is left alone
    assert (
        normalize_punctuation("引用表{v1}{i}, {/i}{v2}{i}, ……, {/i}{v3}", "zh-CN")
        == "引用表{v1}{i}, {/i}{v2}{i}, …, {/i}{v3}"
    )
    assert normalize_punctuation("x{v1} + …… + x{v2}", "zh-CN") == "x{v1} + … + x{v2}"
    assert normalize_punctuation("我不知道……", "zh-CN") == "我不知道……"
    assert normalize_punctuation("苹果、香蕉……等水果", "zh-CN") == "苹果, 香蕉……等水果"

    class FullwidthTranslator(MockTranslator):
        async def translate_batch(self, texts):
            return [f"译（{t}）文，完" for t in texts]

    runner = Runner(FullwidthTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    assert asyncio.run(runner.translate_all(["para"], kinds=["p"], preview=True)) == ["译(para)文, 完"]
    assert runner.cache.d["para"] == "译(para)文, 完"

    class NeverCalled(MockTranslator):
        async def translate_batch(self, texts):
            raise AssertionError("全部命中缓存，不该发请求")

    cache = MemCache({"old": "旧（缓存）译文，嗯"})
    hit = Runner(NeverCalled("zh-CN"), cache=cache, skip_same_lang=False)
    assert asyncio.run(hit.translate_all(["old"], kinds=["p"], preview=True)) == ["旧(缓存)译文, 嗯"]
    assert cache.d["old"] == "旧(缓存)译文, 嗯"


def test_fix_numbered_unit_inside_html():
    """Rich-text translations: when a number is wrapped in <small> or nested in <a>, the text nodes are rewritten and
    the tags are kept as they are."""
    from app.runner import fix_numbered_unit

    html = '<a class="c1" href="part0015.html">第<small class="c2">1</small>卷</a>  论一般的法律'
    out = fix_numbered_unit(html, "zh-CN", html=True)
    assert 'href="part0015.html"' in out and '<small class="c2">一</small>' in out
    assert re.sub(r"<[^>]+>", "", out).startswith("第一卷")
    inv = '<a href="x">第<small>卷26</small></a>　论法律'
    assert re.sub(r"<[^>]+>", "", fix_numbered_unit(inv, "zh-CN", html=True)).startswith("第二十六卷")
    # Footnote numbers and page numbers are not merged into the chapter number; plain text with angle brackets is
    # not sent to the parser
    assert fix_numbered_unit("第2<sup>1</sup>章", "zh-CN", html=True) == "第二<sup>1</sup>章"
    page = '<span epub:type="pagebreak" id="page7" role="doc-pagebreak">7</span>第2章'
    assert fix_numbered_unit(page, "zh-CN", html=True) == page.replace("第2章", "第二章")
    mid = '第2<span epub:type="pagebreak" id="page7">7</span>章'
    assert fix_numbered_unit(mid, "zh-CN", html=True) == mid.replace("第2", "第二")
    assert fix_numbered_unit("第2章 <vector>", "zh-CN") == "第二章 <vector>"
    assert fix_numbered_unit("第2章</div>标题还在", "zh-CN", html=True) == "第2章</div>标题还在"
    # Long paragraphs and citation formats are left alone
    cite = '<a href="x">参考文献</a>：《Science》307卷第5708期'
    assert fix_numbered_unit(cite, "zh-CN", html=True) == cite


def test_fix_numbered_unit_with_selfclosed_anchor():
    """Numbers next to self-closing anchors (<a id="x"/>, <span id="x"/>) are normalized too, and the anchor structure
    is not swallowed."""
    from app.runner import fix_numbered_unit as fix

    h = '<a class="calibre1" id="chapter15"/>第<small class="calibre5">15</small>卷'
    out = fix(h, "zh-CN", html=True)
    assert re.sub(r"<[^>]+>", "", out) == "第十五卷"
    assert 'id="chapter15"' in out
    # The anchor is still an empty element; the number must not be swallowed into the <a>
    assert re.search(r"<a [^>]*></a>", out) or "/>" in out
    assert fix('<span id="pg"/>第2卷', "zh-CN", html=True) != '<span id="pg"/>第2卷'
    # With whitespace before the slash it also expands cleanly, and the number does not stay unchanged
    spaced = '<a class="calibre1" id="chapter15" />第<small class="calibre5">15</small>卷'
    sout = fix(spaced, "zh-CN", html=True)
    assert re.sub(r"<[^>]+>", "", sout) == "第十五卷"
    assert 'id="chapter15"' in sout and re.search(r"<a [^>]*></a>", sout)
    span = fix('<span id="pg" />第2卷', "zh-CN", html=True)
    assert re.sub(r"<[^>]+>", "", span) == "第二卷"
    assert re.search(r"<span [^>]*></span>", span)


def test_empty_translations_backfilled_below_threshold():
    """An occasional empty translation is filled with the source text, and the job finishes as usual and counts it;
    only above the threshold does it fail."""

    class FlakyEmpty(MockTranslator):
        def __init__(self, target_lang, empty_at):
            super().__init__(target_lang)
            self.empty_at = empty_at

        async def translate_batch(self, texts):
            return ["" if t in self.empty_at else f"[zh-CN] {t}" for t in texts]

    texts = [f"Sentence number {i} here." for i in range(20)]
    runner = Runner(FlakyEmpty("zh-CN", empty_at={texts[3]}), cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all(texts))
    assert out[3] == texts[3]  # Filled with the source text
    assert runner.info["failed"] == 1
    assert out[4].startswith("[zh-CN]")

    runner2 = Runner(FlakyEmpty("zh-CN", empty_at=set(texts[:8])), cache=MemCache(), skip_same_lang=False)
    with pytest.raises(TranslatorError, match="空译文"):
        asyncio.run(runner2.translate_all(texts))


def test_echo_translations_retried_individually():
    """A paragraph echoed whole in a batch (the translation is still in the source language) is retranslated once on
    its own; it is kept only if it still echoes."""
    calls = []

    def handler(request):
        import json as _json

        n = len(_json.loads(request.content)["messages"][1]["content"].split("[[p"))
        calls.append(n)
        if n > 2:  # Batch: echo the source text
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "[[p0]]\nThis is a fairly long sentence that should really be translated.\n"
                                "[[p1]]\nLSTM\n[[source_end]]"
                            }
                        }
                    ]
                },
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "这是一个相当长的句子，确实应该被翻译出来。"}}]}
        )

    tr = __import__("tests.test_translators", fromlist=["make_openai"]).make_openai(handler)
    long_text = "This is a fairly long sentence that should really be translated."
    runner = Runner(tr, cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all([long_text, "LSTM"]))
    # The echo was fixed by the single retranslation (the comma normalized to half-width)
    assert out[0] == "这是一个相当长的句子, 确实应该被翻译出来。"
    assert out[1] == "LSTM"  # Proper nouns are not affected


def test_cached_prose_echo_is_dropped_and_retried():
    """A whole English sentence echoed back does not stay in the cache. The next re-layout translates it again
    instead of pasting the English back every time."""
    src = (
        "To obtain a better coverage across the different systems, we used a larger "
        "benchmark so the comparison stays fair for every engine involved."
    )
    cache = MemCache({src: src})
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "为了在不同系统之间获得更好的覆盖, 我们使用了更大的基准, "
                            "让每个引擎的比较都公平。"
                        }
                    }
                ]
            },
        )

    tr = __import__("tests.test_translators", fromlist=["make_openai"]).make_openai(handler)
    runner = Runner(tr, cache=cache, skip_same_lang=False)
    out = asyncio.run(runner.translate_all([src]))
    assert calls
    assert "覆盖" in out[0]
    assert cache.d[src] != src


def test_stray_placeholders_cleaned_from_formula_free_blocks():
    """Blocks without formulas: {vN} placeholders from model mismatches or hallucinations are always stripped and do
    not leak into the output."""
    from app.formats.pdf_restore import _STRAY_PLACEHOLDER_RE

    t = "其中{v1}从{v2}到{v3}进行线性递减"
    assert _STRAY_PLACEHOLDER_RE.sub("", t) == "其中从到进行线性递减"
    assert _STRAY_PLACEHOLDER_RE.sub("", "正常译文") == "正常译文"


def test_finished_job_preview_returns_updates_after_last_poll(app_env):
    """When the last batch finishes between two polls and the job ends right after, the next incremental poll gets
    the translations of that batch."""
    job = Job(id="snap1", filename="a.md", preview=True, status="done")
    state.jobs.out_dir(job.id).mkdir(parents=True)
    state.jobs.add(job)
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    asyncio.run(runner.translate_all(paras(3), preview=True))
    job.runner = runner
    state.jobs.save_preview(job)
    job.runner = None
    with TestClient(main.app) as client:
        snap = client.get("/api/jobs/snap1/preview").json()
        late = client.get("/api/jobs/snap1/preview?since=1").json()
        assert late["version"] == snap["version"]
        assert sorted(u[0] for u in late["updates"]) == [0, 1, 2]
        assert "segments" not in late
        assert client.get(f"/api/jobs/snap1/preview?since={snap['version']}").json()["updates"] == []


def test_dotted_section_numbers_stay_in_digits():
    """The source starts with 7.4 or 1.2.3, and the model, following "章节号写中文数字" (write section numbers in
    Chinese numerals), wrote "七点四" or "四.11", or added "第十二章" to a bare chapter number without Chapter: these
    are restored to the original, and old translations in the cache are rewritten on a hit too. Sources that spell out
    the unit and readings that do not match are left alone."""
    from app.runner import Runner
    from app.runner import keep_section_number as keep

    assert keep("7.4 The input", "七点四 输入", "zh-CN") == "7.4 输入"
    assert keep("{b}7.4{/b} Prove that", "{b}七点四{/b} 证明", "zh-CN") == "{b}7.4{/b} 证明"
    assert keep("1.7.1 Inference", "第一点七点一 推理", "zh-CN") == "1.7.1 推理"
    assert keep("24.7.1 Linking", "二十四点七点一 链接", "zh-CN") == "24.7.1 链接"
    assert keep("4.11 Statistical Testing", "四.11 统计检验", "zh-CN") == "4.11 统计检验"
    assert keep("12 Agents", "第十二章 智能体", "zh-CN") == "12 智能体"
    assert keep("CHAPTER 5 Embeddings", "第五章 嵌入", "zh-CN") == "第五章 嵌入"
    assert keep("2 words were found", "找到了两个词", "zh-CN") == "找到了两个词"
    assert keep("1.1 What", "一点十一 什么", "zh-CN") == "一点十一 什么"
    assert keep("7.4 The input", "七点四 输入", "en") == "七点四 输入"
    # The same holds for section numbers, figure numbers and decimals cited in a sentence
    assert (
        keep("introduced in Section {c0000ff}7.6{/c}", "在第 {c0000ff}七点六节{/c}中介绍", "zh-CN")
        == "在第 {c0000ff}7.6节{/c}中介绍"
    )
    assert keep("see Section 16.4.3 and Fig. 17.3", "见十六点四点三节和图十七点三", "zh-CN") == "见16.4.3节和图17.3"
    assert keep("16.2 and 6.2, with p = 0.5", "十六点二和六点二, 取 p = 零点五", "zh-CN") == "16.2和6.2, 取 p = 0.5"

    class NeverCalled(MockTranslator):
        async def translate_batch(self, texts):
            raise AssertionError("全部命中缓存，不该发请求")

    cache = MemCache({"17.2.3 Residual vector quantization": "十七点二点三 残差向量量化"})
    runner = Runner(NeverCalled("zh-CN"), cache=cache, skip_same_lang=False)
    out = asyncio.run(runner.translate_all(["17.2.3 Residual vector quantization"], kinds=["h2"]))
    assert out == ["17.2.3 残差向量量化"]
    assert cache.d["17.2.3 Residual vector quantization"] == "17.2.3 残差向量量化"


def test_cached_translation_with_stray_script_is_dropped_and_retried():
    """A cached translation containing foreign script is dropped and retranslated; when only typesetting, it cannot
    be retranslated and is used as is."""
    src = "Please take a moment to convince yourself."
    cache = MemCache({src: "请花一点时间 убед自己。"})

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "请花一点时间说服自己。"}}]})

    tr = __import__("tests.test_translators", fromlist=["make_openai"]).make_openai(handler)
    out = asyncio.run(Runner(tr, cache=cache, skip_same_lang=False).translate_all([src]))
    assert out[0] == "请花一点时间说服自己。"
    assert cache.d[src] == "请花一点时间说服自己。"
    cache.d[src] = "请花一点时间 убед自己。"
    out = asyncio.run(Runner(tr, cache=cache, skip_same_lang=False, cache_only=True).translate_all([src]))
    assert out[0] == "请花一点时间 убед自己。"


def test_notes_reach_the_prompt_and_key_the_cache():
    """Paragraphs with notes: the note goes into the system prompt with the batch, the cache key is the source text
    plus the note, and translations stored earlier without the note no longer hit."""
    systems = []

    def handler(request):
        import json as _json

        messages = _json.loads(request.content)["messages"]
        systems.append(messages[0]["content"])
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "[[p0]]\n对数\n[[p1]]\n对数似然比, 531"}}]}
        )

    tr = __import__("tests.test_translators", fromlist=["make_openai"]).make_openai(handler)
    cache = MemCache({"log": "日志"})
    texts = ["log", "log likelihood ratio, 531"]
    notes = ["An entry in the subject index at the back of the book.", ""]
    out = asyncio.run(Runner(tr, cache=cache, skip_same_lang=False).translate_all(texts, notes=notes))
    assert out == ["对数", "对数似然比, 531"]
    assert "## Item notes" in systems[0] and "[[p0]] An entry in the subject index" in systems[0]
    assert "[[p1]]" not in systems[0].split("## Item notes")[1]
    assert cache.d["log"] == "日志"
    assert cache.d["log\n\n[note] An entry in the subject index at the back of the book."] == "对数"
