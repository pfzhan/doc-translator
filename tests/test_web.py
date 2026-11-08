from collections.abc import Awaitable, Callable
from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from app import state
from app.json_types import JSONDict
from app.runner import get_cache
from app.translators import MockTranslator, TranslatorError, create_translator

EN = "The quick brown fox jumps over the lazy dog near the river bank."
ZH = "这是一段已经是中文的文字，不需要再翻译一遍。"


def post(main: ModuleType, body: JSONDict, **headers: str) -> httpx.Response:
    resp: httpx.Response = TestClient(main.app).post("/api/web/translate", json=body, headers=headers)
    return resp


def test_translates_aligned_and_keeps_skipped(app_env: ModuleType) -> None:
    resp = post(app_env, {"texts": [EN, "  ", ZH], "target_lang": "zh-CN", "service_id": "mock"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["translations"] == [f"[zh-CN] {EN}", None, None]
    assert data["langs"] == ["zh-CN", None, None]
    assert data["detected_lang"]


def test_alt_lang_takes_texts_already_in_target(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    async def aclose(self: MockTranslator) -> None:
        closed.append(self.target_lang)

    monkeypatch.setattr(MockTranslator, "aclose", aclose)
    body = {"texts": [EN, "  ", ZH], "target_lang": "zh-CN", "alt_lang": "en", "service_id": "mock"}
    data = post(app_env, body).json()
    assert data["translations"] == [f"[zh-CN] {EN}", None, f"[en] {ZH}"]
    assert data["langs"] == ["zh-CN", None, "en"]
    assert sorted(closed) == ["en", "zh-CN"]


def test_alt_lang_skips_texts_translated_unchanged(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    async def echoing(self: MockTranslator, texts: list[str]) -> list[str]:
        calls.append((self.target_lang, list(texts)))
        return list(texts)

    monkeypatch.setattr(MockTranslator, "translate_batch", echoing)
    body = {"texts": [EN], "target_lang": "zh-CN", "alt_lang": "en", "service_id": "mock"}
    data = post(app_env, body).json()
    assert data["translations"] == [None]
    assert data["langs"] == [None]
    assert calls == [("zh-CN", [EN])]


def test_alt_lang_same_as_target_keeps_texts(app_env: ModuleType) -> None:
    body = {"texts": [ZH], "target_lang": "zh-CN", "alt_lang": "zh-CN", "service_id": "mock"}
    assert post(app_env, body).json()["translations"] == [None]


def test_unsupported_alt_lang_fails_only_texts_needing_it(app_env: ModuleType) -> None:
    body = {"texts": [EN], "target_lang": "zh-CN", "alt_lang": "xx", "service_id": "mock"}
    assert post(app_env, body).json()["translations"] == [f"[zh-CN] {EN}"]
    resp = post(app_env, {**body, "texts": [EN, ZH]})
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("改译已是目标语言的段落时出错：")


def test_uses_cache_across_requests(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    original: Callable[[MockTranslator, list[str]], Awaitable[list[str]]] = MockTranslator.translate_batch

    async def counting(self: MockTranslator, texts: list[str]) -> list[str]:
        calls.append(list(texts))
        return await original(self, texts)

    monkeypatch.setattr(MockTranslator, "translate_batch", counting)
    body = {"texts": [EN], "service_id": "mock"}
    first = post(app_env, body).json()
    second = post(app_env, body).json()
    assert first == second
    assert calls == [[EN]]


def test_unknown_or_disabled_service(app_env: ModuleType) -> None:
    assert post(app_env, {"texts": [EN], "service_id": "nope"}).status_code == 400
    mock = state.store.get("mock")
    assert mock is not None
    mock["enabled"] = False
    assert post(app_env, {"texts": [EN], "service_id": "mock"}).status_code == 400


def test_limits(app_env: ModuleType) -> None:
    assert post(app_env, {"texts": [EN] * 51, "service_id": "mock"}).status_code == 422
    assert post(app_env, {"texts": ["a" * 10001], "service_id": "mock"}).status_code == 400


def test_html_flags_reach_the_engine(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, bool] = {}
    original: Callable[[MockTranslator, list[str]], Awaitable[list[str]]] = MockTranslator.translate_batch

    async def recording(self: MockTranslator, texts: list[str]) -> list[str]:
        seen.update({t: self.is_html_item(t) for t in texts})
        return await original(self, texts)

    monkeypatch.setattr(MockTranslator, "translate_batch", recording)
    rich = 'Read the <a i="0">project documentation</a> before you start.'
    resp = post(app_env, {"texts": [rich, EN], "html": [True], "service_id": "mock"})
    assert resp.status_code == 200
    assert resp.json()["translations"] == [f"[zh-CN] {rich}", f"[zh-CN] {EN}"]
    assert seen == {rich: True, EN: False}
    assert post(app_env, {"texts": [EN], "html": [True, False], "service_id": "mock"}).status_code == 400


def test_unsupported_language(app_env: ModuleType) -> None:
    assert post(app_env, {"texts": [EN], "service_id": "mock", "target_lang": "xx"}).status_code == 400


def test_request_size_limits(app_env: ModuleType) -> None:
    assert post(app_env, {"texts": [EN], "service_id": "mock", "title": "t" * 1001}).status_code == 422
    big: JSONDict = {"texts": ["a" * 9000] * 50, "service_id": "mock", "padding": "p" * (4 * 1024 * 1024)}
    assert post(app_env, big).status_code == 413


def test_engine_errors_map_to_502(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    async def failing(self: MockTranslator, texts: list[str]) -> list[str]:
        raise TranslatorError("服务出错了")

    async def aclose(self: MockTranslator) -> None:
        closed.append(True)

    monkeypatch.setattr(MockTranslator, "translate_batch", failing)
    monkeypatch.setattr(MockTranslator, "aclose", aclose)
    resp = post(app_env, {"texts": [EN], "service_id": "mock"})
    assert resp.status_code == 502
    assert resp.json()["detail"] == "服务出错了"
    assert closed == [True]


def test_empty_request(app_env: ModuleType) -> None:
    assert post(app_env, {"texts": [], "service_id": "mock"}).json() == {
        "translations": [],
        "langs": [],
        "detected_lang": "",
    }


def test_clear_web_cache(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    original: Callable[[MockTranslator, list[str]], Awaitable[list[str]]] = MockTranslator.translate_batch

    async def counting(self: MockTranslator, texts: list[str]) -> list[str]:
        calls.append(list(texts))
        return await original(self, texts)

    monkeypatch.setattr(MockTranslator, "translate_batch", counting)
    client = TestClient(app_env.app)
    shared = "A sentence that a translated document also contains, word for word."
    # A document translated earlier uses one of the sentences: it keeps it after the web cache is cleared
    cache = get_cache()
    mock = state.store.get("mock")
    assert mock is not None
    translator = create_translator(mock, "zh-CN")
    cache.put_many(translator.cache_key, {shared: f"[zh-CN] {shared}"})
    cache.claim("documentdigest", translator.cache_key, [shared])

    assert post(app_env, {"texts": [EN, shared], "service_id": "mock"}).status_code == 200
    assert calls == [[EN]]
    assert client.get("/api/web/cache").json() == {"entries": 1}
    assert client.delete("/api/web/cache").json() == {"entries": 1}
    assert client.get("/api/web/cache").json() == {"entries": 0}

    # The page's sentence is translated again; the document's stays cached
    assert post(app_env, {"texts": [EN, shared], "service_id": "mock"}).status_code == 200
    assert calls == [[EN], [EN]]
    assert cache.get_many(translator.cache_key, [shared]) == {shared: f"[zh-CN] {shared}"}


def test_clear_web_cache_rejects_other_pages(app_env: ModuleType) -> None:
    resp = TestClient(app_env.app).delete("/api/web/cache", headers={"origin": "https://evil.example"})
    assert resp.status_code == 403


def test_requests_to_one_service_share_its_concurrency(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each web request makes its own engine. The service's concurrency limits them together; a limit of 1 used to
    let 8 simultaneous page requests send 8 upstream requests at once."""
    import asyncio
    import functools

    from app.routes import web

    live = {"now": 0, "peak": 0}

    async def upstream(request: httpx.Request) -> httpx.Response:
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0.02)
        live["now"] -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "敏捷的棕色狐狸跳过了河边的懒狗。"}}]})

    state.store.services.append(
        {
            "id": "llm",
            "provider": "custom",
            "name": "LLM",
            "enabled": True,
            "base_url": "http://upstream.test/v1",
            "model": "m",
            "concurrency": 1,
        }
    )
    monkeypatch.setattr(
        web, "create_translator", functools.partial(create_translator, transport=httpx.MockTransport(upstream))
    )

    async def scenario() -> None:
        requests = [
            web.WebTranslateRequest(texts=[f"{EN} Number {i}."], source_lang="en", service_id="llm") for i in range(8)
        ]
        await asyncio.gather(*(web.web_translate(r) for r in requests))

    asyncio.run(scenario())
    assert live["peak"] == 1
