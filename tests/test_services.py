import asyncio
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import ServiceError, ServiceStore
from app.translators import (
    ClaudeTranslator,
    GeminiTranslator,
    GrokTranslator,
    OpenAITranslator,
    TranslatorError,
    create_translator,
)


def run(coro):
    return asyncio.run(coro)


def fake_llm(provider: str, log: list):
    """按各家接口格式返回“译文 = 原文前加 T:”的假服务器。"""

    def handler(request: httpx.Request):
        log.append(request)
        if request.method == "GET":
            if provider == "gemini":
                return httpx.Response(200, json={"models": [
                    {"name": "models/gemini-x", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
                ]})
            return httpx.Response(200, json={"data": [{"id": "m-b"}, {"id": "m-a"}]})
        body = json.loads(request.content)
        if provider == "gemini":
            user = body["contents"][0]["parts"][0]["text"]
        else:
            user = body["messages"][-1]["content"]
        reply = fake_marker_reply(user)
        if provider == "claude":
            return httpx.Response(200, json={"content": [{"type": "text", "text": reply}]})
        if provider == "gemini":
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [
                {"text": "thinking...", "thought": True}, {"text": reply}]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": f"<think>hmm</think>\n{reply}"}}]})

    return httpx.MockTransport(handler)


def fake_marker_reply(user: str) -> str:
    """模拟模型：按 [[pN]] 协议逆序返回，单段时直接返回译文。"""
    body = user.split("\n\n", 1)[1]  # 去掉 "翻译为简体中文：" 这一行
    if "[[source_end]]" not in body:
        return "T:" + body
    items = re.findall(r"\[\[(p\d+)\]\]\n(.*?)(?=\n\[\[)", body, re.S)
    return "\n".join(f"[[{pid}]]\nT:{text}" for pid, text in reversed(items))


@pytest.mark.parametrize("provider,cls,url", [
    ("openai", OpenAITranslator, "https://api.openai.com/v1/chat/completions"),
    ("grok", GrokTranslator, "https://api.x.ai/v1/chat/completions"),
    ("claude", ClaudeTranslator, "https://api.anthropic.com/v1/messages"),
    ("gemini", GeminiTranslator, "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"),
])
def test_providers_request_format(provider, cls, url):
    log = []
    svc = {"provider": provider, "api_key": "sk-test", "model": "gemini-x" if provider == "gemini" else "m"}
    tr = create_translator(svc, "zh-CN", transport=fake_llm(provider, log))
    assert isinstance(tr, cls)
    assert run(tr.translate_batch(["hello", "world\nline2"])) == ["T:hello", "T:world\nline2"]
    assert run(tr.translate_batch(["single"])) == ["T:single"]
    req = log[0]
    assert str(req.url) == url
    body = json.loads(req.content)
    if provider == "claude":
        assert req.headers["x-api-key"] == "sk-test"
        assert "anthropic-version" in req.headers
        system = body["system"]
    elif provider == "gemini":
        assert req.headers["x-goog-api-key"] == "sk-test"
        system = body["systemInstruction"]["parts"][0]["text"]
    else:
        assert req.headers["authorization"] == "Bearer sk-test"
        assert body["messages"][0]["role"] == "system"
        system = body["messages"][0]["content"]
    # 沉浸式的简体中文专用提示词 + 多段协议
    assert system.startswith("你是专业的简体中文母语译者")
    assert "[[source_end]]" in system
    run(tr.aclose())


def test_list_models():
    log = []
    tr = create_translator({"provider": "gemini", "api_key": "k", "model": "x"}, "zh-CN",
                           transport=fake_llm("gemini", log))
    assert run(tr.list_models()) == ["gemini-x"]
    tr = create_translator({"provider": "openai", "api_key": "k", "model": "x"}, "zh-CN",
                           transport=fake_llm("openai", log))
    assert run(tr.list_models()) == ["m-a", "m-b"]


def test_custom_base_url_and_missing_key():
    log = []
    tr = create_translator({"provider": "custom", "base_url": "http://localhost:11434/v1/", "model": "qwen"},
                           "en", transport=fake_llm("custom", log))
    run(tr.translate_batch(["你好"]))
    assert str(log[0].url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in log[0].headers
    with pytest.raises(TranslatorError, match="API Key"):
        create_translator({"provider": "claude", "model": "m", "name": "我的 Claude"}, "zh-CN")
    with pytest.raises(TranslatorError, match="API 地址"):
        create_translator({"provider": "custom", "model": "m"}, "zh-CN")


def test_build_messages_and_custom_prompt():
    from app.prompts import build_messages

    system, user = build_messages(["a", "b"], "de", title="My Book")
    assert system.startswith("You are a professional German native translator")
    assert "Title: “My Book”" in system
    assert "{{" not in system
    assert user == "Translate to German:\n\n[[p0]]\na\n[[p1]]\nb\n[[source_end]]"

    system, user = build_messages(["only {{to}} literal"], "zh-CN", system_template="Be a {{to}} poet.",
                                  user_template="Render:")
    assert system.startswith("Be a Simplified Chinese poet.")
    assert "[[source_end]]" not in system and "Return only the processed text" in system
    assert user == "Render:\n\nonly {{to}} literal"  # 原文里的 {{to}} 不能被替换


def test_marker_parse_missing_item_is_retranslated():
    from app.prompts import parse_markers

    assert parse_markers("```\n[[p1]]\nB\n[[p0]]\nA\nmore\n[[end]]\nignored\n```", 2) == {0: "A\nmore", 1: "B"}

    calls = []

    def handler(request):
        user = json.loads(request.content)["messages"][-1]["content"]
        calls.append(user)
        # 批量时故意漏掉 p1
        content = "[[p0]]\nA\n[[p2]]\nC" if "[[source_end]]" in user else "B-single"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = create_translator({"provider": "openai", "api_key": "k", "model": "m"}, "zh-CN",
                           transport=httpx.MockTransport(handler))
    assert run(tr.translate_batch(["a", "b", "c"])) == ["A", "B-single", "C"]
    assert len(calls) == 2


@pytest.mark.parametrize("provider,base,expected", [
    ("openai", "https://relay.example.com/", "https://relay.example.com/v1/chat/completions"),
    ("custom", "https://relay.example.com:1100", "https://relay.example.com:1100/v1/chat/completions"),
    ("openai", "https://relay.example.com/api/v3", "https://relay.example.com/api/v3/chat/completions"),
    ("claude", "https://relay.example.com", "https://relay.example.com/v1/messages"),
    ("gemini", "https://relay.example.com", "https://relay.example.com/v1beta/models/m:generateContent"),
])
def test_domain_only_base_url_gets_version_path(provider, base, expected):
    log = []

    def handler(request):
        log.append(request)
        if provider == "claude":
            return httpx.Response(200, json={"content": [{"type": "text", "text": "好"}]})
        if provider == "gemini":
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "好"}]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

    tr = create_translator({"provider": provider, "api_key": "k", "model": "m", "base_url": base}, "zh-CN",
                           transport=httpx.MockTransport(handler))
    run(tr.translate_batch(["good"]))
    assert str(log[0].url) == expected


def test_html_response_gives_readable_error():
    # 中转站把未知路径返回成首页（200 + HTML），以前会抛 JSONDecodeError 变成 500
    tr = create_translator({"provider": "openai", "api_key": "k", "model": "m", "base_url": "https://relay.example.com/x"},
                           "zh-CN", transport=httpx.MockTransport(
                               lambda r: httpx.Response(200, text="<!doctype html><html></html>",
                                                        headers={"content-type": "text/html"})))
    with pytest.raises(TranslatorError, match="不是 JSON.*/v1"):
        run(tr.translate_batch(["good"]))


def test_transient_400_from_relay_is_retried(monkeypatch):
    # 中转站把上游的偶发故障包装成 400 INVALID_MODEL_ID，同样的请求过几秒就成功
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"message": 'Upstream rejected... {"reason":"INVALID_MODEL_ID"}'}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

    tr = create_translator({"provider": "openai", "api_key": "k", "model": "m"}, "zh-CN",
                           transport=httpx.MockTransport(handler))
    assert run(tr.translate_batch(["good"])) == ["好"]
    assert len(calls) == 2


@pytest.mark.parametrize("status", [401, 403, 404])
def test_config_errors_fail_fast(monkeypatch, status):
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "bad key"}})

    tr = create_translator({"provider": "openai", "api_key": "k", "model": "m"}, "zh-CN",
                           transport=httpx.MockTransport(handler))
    with pytest.raises(TranslatorError, match=str(status)):
        run(tr.translate_batch(["good"]))
    assert len(calls) == 1  # Key 错、没权限、地址错，重试没用，立即报错


def test_batch_rejected_by_relay_is_split(monkeypatch):
    # 实测：某个 8 段的批次每次都被中转站 400 拒绝，拆开后每一部分都能翻
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    sizes = []

    def handler(request):
        user = json.loads(request.content)["messages"][-1]["content"]
        n = user.count("[[p") if "[[source_end]]" in user else 1
        sizes.append(n)
        if n >= 8:
            return httpx.Response(400, json={"error": {"message": "INVALID_MODEL_ID"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": fake_marker_reply(user)}}]})

    tr = create_translator({"provider": "openai", "api_key": "k", "model": "m"}, "zh-CN",
                           transport=httpx.MockTransport(handler))
    texts = [f"para {i}" for i in range(8)]
    assert run(tr.translate_batch(texts)) == [f"T:para {i}" for i in range(8)]
    assert sizes == [8, 8, 4, 4]  # 原样重试一次，然后拆成两半


def test_single_segment_400_still_fails(monkeypatch):
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    tr = create_translator({"provider": "openai", "api_key": "k", "model": "m"}, "zh-CN",
                           transport=httpx.MockTransport(lambda r: httpx.Response(400, json={"error": {"message": "x"}})))
    with pytest.raises(TranslatorError) as e:
        run(tr.translate_batch(["only one"]))
    assert e.value.status == 400


async def _no_sleep(_seconds):
    return None


def test_temperature_retry_without_it():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature'"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

    tr = create_translator({"provider": "openai", "api_key": "k", "model": "o3", "temperature": 0}, "zh-CN",
                           transport=httpx.MockTransport(handler))
    assert run(tr.translate_batch(["good"])) == ["好"]
    assert len(calls) == 2 and "temperature" not in calls[1]


def test_cache_key_follows_service_id_not_model():
    def tr(sid, **kw):
        return create_translator({"id": sid, "provider": "openai", "api_key": "k", "model": "a", **kw}, "zh-CN")

    a, b, c, d = tr("s1"), tr("s1", model="b"), tr("s1", prompt="x"), tr("s2")
    # 同一服务换模型/提示词：已翻好的段落继续复用；换服务 id 才重新翻译
    assert a.cache_key == b.cache_key == c.cache_key != d.cache_key
    # 旧版缓存 key 含模型和提示词，仍然各不相同
    assert len({a.legacy_cache_key, b.legacy_cache_key, c.legacy_cache_key}) == 3


def test_legacy_cache_entries_are_reused_and_migrated(tmp_path):
    from app.runner import Cache, Runner

    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "新译文"}}]})

    svc = {"id": "s1", "provider": "openai", "api_key": "k", "model": "old", "base_url": "https://r.example.com/v1"}
    cache = Cache(tmp_path / "cache.sqlite3")
    old = create_translator(svc, "zh-CN")
    old.detected_source = "en"  # 旧版运行时同样先检测出原文语言再算 key
    text = "A sentence long enough to be translated by the model."
    cache.put_many(old.legacy_cache_key, {text: "旧译文"})

    # 升级后第一次继续翻译（同一模型）：旧缓存命中并迁移到新 key
    same = create_translator(svc, "zh-CN", transport=httpx.MockTransport(handler))
    assert run(Runner(same, cache=cache).translate_all([text])) == ["旧译文"] and calls == []

    # 之后换了模型：迁移过的段落照样复用，新段落才用新模型
    tr = create_translator({**svc, "model": "new"}, "zh-CN", transport=httpx.MockTransport(handler))
    other = "Another sentence that was never translated before."
    out = run(Runner(tr, cache=cache).translate_all([text, other]))
    assert out == ["旧译文", "新译文"] and calls == ["new"]
    assert cache.get_many(tr.cache_key, [text]) == {text: "旧译文"}


def test_cache_reused_only_for_same_service_id():
    from app.runner import Runner

    class Cache:
        def __init__(self):
            self.d = {}

        def get_many(self, prefix, texts):
            return {t: self.d[(prefix, t)] for t in texts if (prefix, t) in self.d}

        def put_many(self, prefix, pairs):
            self.d.update({(prefix, s): t for s, t in pairs.items()})

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "译文"}}]})

    def svc(sid):
        # 两个服务除了 id 完全相同（同一个中转站、同一个模型）
        return {"id": sid, "provider": "openai", "api_key": "k", "model": "m", "base_url": "https://r.example.com/v1"}

    cache = Cache()
    text = ["A sentence long enough to be translated by the model."]
    for sid in ("svc-a", "svc-a", "svc-b"):
        tr = create_translator(svc(sid), "zh-CN", transport=httpx.MockTransport(handler))
        run(Runner(tr, cache=cache).translate_all(text))
    # 第二次同一服务命中缓存；换了服务 id 就重新翻译
    assert len(calls) == 2
    assert {k[0].split(":")[0] for k in cache.d} == {"svc-a", "svc-b"}


def test_store_crud(tmp_path):
    store = ServiceStore(tmp_path / "s.json")
    assert [s["id"] for s in store.services] == ["google"]
    svc = store.create({"provider": "claude", "name": "my-claude", "api_key": "sk-ant-1234567890"})
    assert "api_key" not in svc and svc["api_key_hint"] == "sk-…7890"
    assert svc["model"]  # 有预置模型时自动选第一个

    # Key 留空不覆盖
    store.update(svc["id"], {"api_key": "", "model": "claude-x"})
    assert store.get(svc["id"])["api_key"] == "sk-ant-1234567890"
    assert store.get(svc["id"])["model"] == "claude-x"

    store.set_default(svc["id"])
    with pytest.raises(ServiceError):
        store.update(svc["id"], {"enabled": False})
    with pytest.raises(ServiceError):
        store.delete("google")

    # 重新加载后仍在，文件权限是 600
    store2 = ServiceStore(tmp_path / "s.json")
    assert store2.default_id == svc["id"]
    assert (tmp_path / "s.json").stat().st_mode & 0o777 == 0o600

    store2.delete(svc["id"])
    assert store2.default_id == "google"


def test_api_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", ServiceStore(tmp_path / "s.json"))
    client = TestClient(main.app)
    providers = client.get("/api/providers").json()
    assert {"openai", "claude", "gemini", "grok", "custom"} <= providers.keys()

    r = client.post("/api/services", json={"provider": "grok", "api_key": "xai-abcdefghijk"})
    sid = r.json()["id"]
    listed = client.get("/api/services").json()
    assert all("api_key" not in s for s in listed["services"])
    assert client.post(f"/api/services/{sid}/default").json()["default"] == sid
    assert client.post("/api/services", json={"provider": "nope"}).status_code == 400
    # 缺 Key 时测试接口返回 400 和可读的错误
    r = client.post("/api/services/test", json={"provider": "claude", "model": "m"})
    assert r.status_code == 400 and "API Key" in r.json()["detail"]


def test_corrupted_services_json_falls_back(tmp_path):
    """services.json 损坏时回退到空配置，内置谷歌翻译仍在，不影响启动。"""
    path = tmp_path / "s.json"
    path.write_text("{not json", encoding="utf-8")
    store = ServiceStore(path)
    assert [s["id"] for s in store.services] == ["google"]
    assert store.default_id == "google"
    assert store.list()["default"] == "google"


def test_ccswitch_default_unreadable_db_falls_back(tmp_path):
    """默认服务是 CC Switch 的但数据库暂时读不到：配置保留，实际生效的默认退回谷歌翻译。"""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"default": "ccswitch-claude", "services": [], "ccswitch": {}}),
                    encoding="utf-8")
    store = ServiceStore(path, ccswitch_db=tmp_path / "missing.db")
    assert store.default_id == "ccswitch-claude"  # 原值保留，数据库恢复后仍是默认
    assert store.effective_default() == "google"
    assert store.list()["default"] == "google"


def test_empty_translation_is_not_cached(tmp_path):
    """空译文不写缓存、不算完成：整批按失败处理，重试时空的段落重新请求。"""
    from app.runner import Cache, Runner

    class SometimesEmpty:
        cache_key = "t"
        concurrency = 1
        max_batch_items = 20
        max_batch_chars = 3000
        source_lang = "en"
        target_lang = "zh-CN"

        def __init__(self):
            self.calls = 0

        async def translate_batch(self, texts):
            self.calls += 1
            if self.calls == 1 and len(texts) > 1:
                return ["译:" + texts[0]] + [""] * (len(texts) - 1)  # 第一段正常，其余返回空
            return ["译:" + t for t in texts]

    cache = Cache(tmp_path / "c.sqlite3")
    tr = SometimesEmpty()
    with pytest.raises(TranslatorError, match="空译文"):
        run(Runner(tr, cache=cache).translate_all(["good one", "bad one"]))
    # 空译文没有进缓存，正常的进了
    assert cache.get_many("t", ["good one", "bad one"]) == {"good one": "译:good one"}
    # 重试：缓存命中的不再请求，空译文重新翻译
    assert run(Runner(tr, cache=cache).translate_all(["good one", "bad one"])) == ["译:good one", "译:bad one"]
    assert tr.calls == 2
