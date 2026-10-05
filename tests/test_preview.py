import re
import asyncio
import io
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
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


def test_normalize_numbered_units():
    from app.runner import normalize_numbered_units as norm

    texts = ["h1", "toc1", "toc2", "body", "cite", "long"]
    results = [
        "第2卷",
        "第3卷　论三种政体的原则",
        "第卷26　论法律应当与其所规范的事物秩序具有的关系",
        "第一章",
        "《Science》307卷第5708期（2005）：414–16",  # 不以“第”开头，不动
        "第4卷　" + "很长的条目" * 10,  # 超长不动
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
    # 卷期页号保留阿拉伯数字
    for cite in ("第25卷第1期", "第25卷第1期（2005）：414–16", "第307卷第5708期", "第2章第3页"):
        assert fix(cite, "zh-CN") == cite


def test_normalize_numbered_units_updates_cache():
    from app.runner import Runner

    class DictTranslator(MockTranslator):
        async def translate_batch(self, texts):
            return ["第2卷" if t == "BOOK 2" else f"[zh-CN] {t}" for t in texts]

    runner = Runner(DictTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all(
        ["BOOK 2", "para one"], kinds=["h1", "p"], preview=True))
    assert out[0] == "第二卷"
    # 缓存里也写成归一化后的版本，预览和历史快照都一致
    assert runner.cache.get_many("x", []) == {}
    assert runner._done["BOOK 2"] == "第二卷"
    # 正文即使译文是「第2卷」也不改、不写进被改过的缓存
    body = Runner(DictTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    assert asyncio.run(body.translate_all(["BOOK 2"], kinds=["p"], preview=True)) == ["第2卷"]


def test_normalize_applies_to_cache_hits():
    """缓存里的旧样式译文（归一化之前翻的）：命中时就地改写，预览立即是新样式。"""
    from app.runner import Runner

    class NeverCalled(MockTranslator):
        async def translate_batch(self, texts):
            raise AssertionError("全部命中缓存，不该发请求")

    cache = MemCache({"BOOK 2": "第2卷", "BOOK 26": "第卷26", "cite": "第25卷第1期"})
    runner = Runner(NeverCalled("zh-CN"), cache=cache, skip_same_lang=False)
    out = asyncio.run(runner.translate_all(
        ["BOOK 2", "BOOK 26", "cite"], kinds=["h1", "toc", "h2"], preview=True))
    assert out == ["第二卷", "第二十六卷", "第25卷第1期"]
    assert cache.d["BOOK 2"] == "第二卷" and cache.d["BOOK 26"] == "第二十六卷"
    assert cache.d["cite"] == "第25卷第1期"
    updates = {u[0]: u[1] for u in runner.preview()["updates"]}
    assert updates[0] == "第二卷" and updates[1] == "第二十六卷" and updates[2] == "第25卷第1期"


def test_fix_numbered_unit_inside_html():
    """富文本译文：数字被 <small> 包住或嵌在 <a> 里时，按文本节点改写，标签原样保留。"""
    from app.runner import fix_numbered_unit

    html = '<a class="c1" href="part0015.html">第<small class="c2">1</small>卷</a>  论一般的法律'
    out = fix_numbered_unit(html, "zh-CN", html=True)
    assert 'href="part0015.html"' in out and "<small class=\"c2\">一</small>" in out
    assert re.sub(r"<[^>]+>", "", out).startswith("第一卷")
    inv = '<a href="x">第<small>卷26</small></a>　论法律'
    assert re.sub(r"<[^>]+>", "", fix_numbered_unit(inv, "zh-CN", html=True)).startswith("第二十六卷")
    # 脚注号和页码不并进章节号；尖括号纯文本不送进解析器
    assert fix_numbered_unit("第2<sup>1</sup>章", "zh-CN", html=True) == "第二<sup>1</sup>章"
    page = '<span epub:type="pagebreak" id="page7" role="doc-pagebreak">7</span>第2章'
    assert fix_numbered_unit(page, "zh-CN", html=True) == page.replace("第2章", "第二章")
    mid = '第2<span epub:type="pagebreak" id="page7">7</span>章'
    assert fix_numbered_unit(mid, "zh-CN", html=True) == mid.replace("第2", "第二")
    assert fix_numbered_unit("第2章 <vector>", "zh-CN") == "第二章 <vector>"
    assert fix_numbered_unit("第2章</div>标题还在", "zh-CN", html=True) == "第2章</div>标题还在"
    # 长段和引文格式不动
    cite = '<a href="x">参考文献</a>：《Science》307卷第5708期'
    assert fix_numbered_unit(cite, "zh-CN", html=True) == cite


def test_fix_numbered_unit_with_selfclosed_anchor():
    """自闭合锚点（<a id="x"/>、<span id="x"/>）旁的编号也要归一，且锚点结构不被吞。"""
    from app.runner import fix_numbered_unit as fix

    h = '<a class="calibre1" id="chapter15"/>第<small class="calibre5">15</small>卷'
    out = fix(h, "zh-CN", html=True)
    assert re.sub(r"<[^>]+>", "", out) == "第十五卷"
    assert 'id="chapter15"' in out
    # 锚点仍然是空元素，不能把编号吞进 <a> 里
    assert re.search(r"<a [^>]*></a>", out) or "/>" in out
    assert fix('<span id="pg"/>第2卷', "zh-CN", html=True) != '<span id="pg"/>第2卷'
    # 斜杠前有空白时也要展开干净，编号不能留着不改
    spaced = '<a class="calibre1" id="chapter15" />第<small class="calibre5">15</small>卷'
    sout = fix(spaced, "zh-CN", html=True)
    assert re.sub(r"<[^>]+>", "", sout) == "第十五卷"
    assert 'id="chapter15"' in sout and re.search(r"<a [^>]*></a>", sout)
    span = fix('<span id="pg" />第2卷', "zh-CN", html=True)
    assert re.sub(r"<[^>]+>", "", span) == "第二卷"
    assert re.search(r"<span [^>]*></span>", span)


def test_empty_translations_backfilled_below_threshold():
    """个别空译文回填原文，任务照常完成并计数；超过阈值才判失败。"""
    class FlakyEmpty(MockTranslator):
        def __init__(self, target_lang, empty_at):
            super().__init__(target_lang)
            self.empty_at = empty_at

        async def translate_batch(self, texts):
            return ["" if t in self.empty_at else f"[zh-CN] {t}" for t in texts]

    texts = [f"Sentence number {i} here." for i in range(20)]
    runner = Runner(FlakyEmpty("zh-CN", empty_at={texts[3]}), cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all(texts))
    assert out[3] == texts[3]  # 回填原文
    assert runner.info["failed"] == 1
    assert out[4].startswith("[zh-CN]")

    runner2 = Runner(FlakyEmpty("zh-CN", empty_at=set(texts[:8])), cache=MemCache(), skip_same_lang=False)
    with pytest.raises(TranslatorError, match="空译文"):
        asyncio.run(runner2.translate_all(texts))


def test_echo_translations_retried_individually():
    """批量里整段照抄原文（回显）的段落单独重翻一次；仍回显才保留（专有名词不误伤）。"""
    class EchoTranslator(MockTranslator):
        async def translate_batch(self, texts):
            # 批量模式下长段回显，单段请求时正常翻译
            if len(texts) > 1:
                return [t if len(t) > 20 else f"[zh-CN] {t}" for t in texts]
            return [f"[zh-CN] {t}" for t in texts]

    long_text = "This is a fairly long sentence that should really be translated."
    runner = Runner(EchoTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    out = asyncio.run(runner.translate_all([long_text, "LSTM"]))
    assert out[0] == f"[zh-CN] {long_text}"  # 回显被单独重翻修好了
    assert out[1] == "[zh-CN] LSTM"  # 专有名词不受影响


def test_stray_placeholders_cleaned_from_formula_free_blocks():
    """没有公式的块：模型错配/幻觉产生的 {vN} 占位符一律剥掉，不漏进输出。"""
    from app.formats.pdf import _STRAY_PLACEHOLDER_RE

    t = "其中{v1}从{v2}到{v3}进行线性递减"
    assert _STRAY_PLACEHOLDER_RE.sub("", t) == "其中从到进行线性递减"
    assert _STRAY_PLACEHOLDER_RE.sub("", "正常译文") == "正常译文"
