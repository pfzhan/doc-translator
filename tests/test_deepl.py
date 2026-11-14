"""The DeepL engine: request format, language codes, address choice, errors and retries, and its service type."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main, state
from app.json_types import JSONDict
from app.services import ServiceError, ServiceStore
from app.translators import (
    DEEPL_FREE_URL,
    DEEPL_PRO_URL,
    DeepLTranslator,
    TranslatorError,
    create_translator,
    deepl_base_url,
)
from tests.helpers import run


async def _no_sleep(_seconds: float) -> None:
    return None


def _echo(request: httpx.Request) -> httpx.Response:
    """DeepL's answer to a request: each text prefixed with the target language."""
    body = json.loads(request.content)
    texts: list[str] = body["text"]
    return httpx.Response(
        200,
        json={
            "translations": [{"detected_source_language": "EN", "text": f"{body['target_lang']}:{t}"} for t in texts]
        },
    )


def _deepl(
    handler: httpx.MockTransport, target: str = "zh-CN", source: str = "auto", key: str = "k:fx"
) -> DeepLTranslator:
    return DeepLTranslator(target, source, api_key=key, transport=handler)


def test_request_format() -> None:
    """Texts go as a JSON list with the key in the Authorization header; an automatic source language is left out
    and the document title goes in as context."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _echo(request)

    tr = _deepl(httpx.MockTransport(handler))
    tr.title = "Alice in Wonderland"
    assert run(tr.translate_batch(["one", "two"])) == ["ZH-HANS:one", "ZH-HANS:two"]
    request = seen[0]
    assert str(request.url) == f"{DEEPL_FREE_URL}/translate"
    assert request.headers["authorization"] == "DeepL-Auth-Key k:fx"
    assert json.loads(request.content) == {
        "text": ["one", "two"],
        "target_lang": "ZH-HANS",
        "context": "Alice in Wonderland",
    }
    run(tr.aclose())


@pytest.mark.parametrize(
    "source,target,source_code,target_code",
    [
        ("en", "zh-TW", "EN", "ZH-HANT"),
        ("zh-HK", "en", "ZH", "EN-US"),
        ("pt-br", "pt", "PT", "PT-PT"),
        ("ja", "pt-br", "JA", "PT-BR"),
        ("no", "de", "NB", "DE"),
    ],
)
def test_language_codes(source: str, target: str, source_code: str, target_code: str) -> None:
    """A source language has no regional variant; English, Portuguese and Chinese targets need one."""
    seen: list[JSONDict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _echo(request)

    tr = _deepl(httpx.MockTransport(handler), target, source)
    run(tr.translate_batch(["x"]))
    assert seen[0]["source_lang"] == source_code and seen[0]["target_lang"] == target_code
    run(tr.aclose())


def test_unsupported_languages_fail_at_once() -> None:
    transport = httpx.MockTransport(_echo)
    with pytest.raises(TranslatorError, match="DeepL 不支持目标语言"):
        _deepl(transport, "wyw")
    with pytest.raises(TranslatorError, match="DeepL 不支持源语言"):
        _deepl(transport, "zh-CN", "fil")


@pytest.mark.parametrize(
    "base,key,expected",
    [
        ("", "abc:fx", DEEPL_FREE_URL),
        ("", "abc", DEEPL_PRO_URL),
        ("https://api.deepl.com", "abc:fx", "https://api.deepl.com/v2"),
        ("https://api-free.deepl.com/v2/translate/", "abc", "https://api-free.deepl.com/v2"),
        ("http://localhost:8080/v2", "", "http://localhost:8080/v2"),
    ],
)
def test_base_url(base: str, key: str, expected: str) -> None:
    """Left empty, the address follows the key: a Free key ends in ':fx'. A domain alone gets /v2."""
    assert deepl_base_url(base, key) == expected


def test_html_and_plain_paragraphs_go_separately() -> None:
    """Rich-text paragraphs are sent with tag_handling=html, plain ones without, and come back in order."""
    modes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        mode = body.get("tag_handling", "text")
        modes.append(mode)
        return httpx.Response(200, json={"translations": [{"text": f"{mode}:{t}"} for t in body["text"]]})

    tr = _deepl(httpx.MockTransport(handler))
    tr.html_texts = {"x <b>bold</b> y"}
    out = run(tr.translate_batch(["plain one", "x <b>bold</b> y", "plain two"]))
    assert out == ["text:plain one", "html:x <b>bold</b> y", "text:plain two"]
    assert sorted(modes) == ["html", "text"]
    run(tr.aclose())


@pytest.mark.parametrize(
    "status,message",
    [
        (403, "API Key 无效"),
        (456, "额度已用完"),
        (400, "Value for 'target_lang' not supported"),
    ],
)
def test_errors_retrying_does_not_fix_fail_at_once(status: int, message: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json={"message": "Value for 'target_lang' not supported."})

    tr = _deepl(httpx.MockTransport(handler))
    with pytest.raises(TranslatorError, match=message) as e:
        run(tr.translate_batch(["x"]))
    assert e.value.status == status
    assert len(calls) == 1
    run(tr.aclose())


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limits_and_server_errors_are_retried(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    """429 (DeepL's rate limit) and 5xx are retried with backoff; 429 and 503 also pause the whole service."""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    monkeypatch.setattr("app.translators._clock", lambda: 1000.0)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(status, json={"message": "Too many requests"})
        return _echo(request)

    tr = _deepl(httpx.MockTransport(handler))
    assert run(tr.translate_batch(["x"])) == ["ZH-HANS:x"]
    assert len(calls) == 3
    run(tr.aclose())


def test_retries_run_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    monkeypatch.setattr("app.translators._clock", lambda: 1000.0)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, json={"message": "Too many requests"})

    tr = _deepl(httpx.MockTransport(handler))
    with pytest.raises(TranslatorError, match="429") as e:
        run(tr.translate_batch(["x"]))
    assert e.value.status == 429 and len(calls) == tr.retries
    run(tr.aclose())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"translations": [{"text": "only one"}]}),
        httpx.Response(200, json={"message": "something else"}),
        httpx.Response(200, text="<html>maintenance</html>"),
    ],
)
def test_unexpected_answers_raise(response: httpx.Response) -> None:
    """A translation missing, or an answer that is not DeepL's, is an error rather than a wrong translation."""
    tr = _deepl(httpx.MockTransport(lambda _request: response))
    with pytest.raises(TranslatorError, match="DeepL 返回"):
        run(tr.translate_batch(["a", "b"]))
    run(tr.aclose())


def test_create_translator_from_service() -> None:
    """The service's key, address and batch sizes reach the engine; batches are capped at what DeepL accepts."""
    service: JSONDict = {
        "id": "d1",
        "provider": "deepl",
        "api_key": " secret ",
        "base_url": "",
        "concurrency": 3,
        "max_items": 200,
        "max_chars": 50000,
    }
    tr = create_translator(service, "de", "en")
    assert isinstance(tr, DeepLTranslator)
    assert tr.url == f"{DEEPL_PRO_URL}/translate" and tr.api_key == "secret"
    assert (tr.concurrency, tr.max_batch_items, tr.max_batch_chars) == (3, 50, 30000)
    assert tr.cache_key == "d1:deepl:en:de"
    run(tr.aclose())
    with pytest.raises(TranslatorError, match="需要先填写 API Key"):
        create_translator({"provider": "deepl", "name": "My DeepL"}, "de")


def test_store_creates_deepl_with_its_defaults(tmp_path: Path) -> None:
    """A new DeepL service starts with DeepL's batch sizes; Google stays the one built-in service."""
    store = ServiceStore(tmp_path / "s.json")
    svc = store.create({"provider": "deepl", "api_key": "abcdefghijk:fx"})
    assert svc["name"] == "DeepL" and svc["model"] == ""
    assert (svc["concurrency"], svc["max_items"], svc["max_chars"]) == (2, 50, 5000)
    with pytest.raises(ServiceError):
        store.create({"provider": "google"})


def test_api_lists_deepl_languages_and_tests_the_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state, "store", ServiceStore(tmp_path / "s.json"))
    client = TestClient(main.app)
    providers = client.get("/api/providers").json()
    assert providers["deepl"]["configurable"] and not providers["deepl"]["llm"]
    assert not providers["google"]["configurable"]
    langs = {lang["code"]: lang for lang in client.get("/api/languages").json()}
    assert langs["auto"]["deepl"] and langs["zh-TW"]["deepl"] and not langs["wyw"]["deepl"]
    r = client.post("/api/services/test", json={"provider": "deepl"})
    assert r.status_code == 400 and "API Key" in r.json()["detail"]
