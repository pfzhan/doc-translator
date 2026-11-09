"""The gate every request of a service goes through: its concurrency, and the pause after it reports overload."""

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

from app.json_types import JSONDict
from app.translators import LLMTranslator, TranslatorError, create_translator
from tests.helpers import run

OK = {"choices": [{"message": {"content": "好"}}]}


def _service(sid: str, concurrency: int = 4) -> JSONDict:
    return {"id": sid, "provider": "custom", "base_url": "http://x/v1", "model": "m", "concurrency": concurrency}


def _engine(sid: str, handler: httpx.MockTransport, concurrency: int = 4) -> LLMTranslator:
    engine = create_translator(_service(sid, concurrency), "zh-CN", "en", transport=handler)
    assert isinstance(engine, LLMTranslator)
    return engine


def _record_sleeps(monkeypatch: pytest.MonkeyPatch, then: Callable[[], Awaitable[None]] | None = None) -> list[float]:
    """Replace asyncio.sleep: record each delay, and await then() instead of sleeping. The service's clock stands
    still, so a pause lasts exactly as long as it was set for."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if then is not None:
            await then()

    monkeypatch.setattr("app.translators.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("app.translators._clock", lambda: 1000.0)
    return sleeps


def test_a_429_pauses_every_request_of_the_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """After one request is told to wait, the other engines of the same service wait too before sending, instead
    of each running into the rate limit on its own; another service is not held back."""
    sleeps = _record_sleeps(monkeypatch)
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        sent.append(body)
        if "rate me" in body:
            return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"Retry-After": "5"})
        return httpx.Response(200, json=OK)

    async def scenario() -> None:
        limited = _engine("svc", httpx.MockTransport(handler))
        limited.retries = 1
        same, other = _engine("svc", httpx.MockTransport(handler)), _engine("other", httpx.MockTransport(handler))
        try:
            with pytest.raises(TranslatorError) as e:
                await limited.translate_batch(["rate me"])
            assert e.value.status == 429
            assert sleeps == []  # The request that failed has nothing left to wait for
            assert await other.translate_batch(["Elsewhere."]) == ["好"]
            assert sleeps == []
            assert await same.translate_batch(["Same service."]) == ["好"]
            assert sleeps == [5.0]
        finally:
            for engine in (limited, same, other):
                await engine.aclose()

    run(scenario())
    assert len(sent) == 3


def test_retry_after_a_429_waits_in_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry after a 429 waits out the service's pause once, not a backoff of its own on top."""
    sleeps = _record_sleeps(monkeypatch)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=OK)

    async def scenario() -> list[str]:
        engine = _engine("svc", httpx.MockTransport(handler))
        try:
            return await engine.translate_batch(["Hello."])
        finally:
            await engine.aclose()

    assert run(scenario()) == ["好"]
    assert sleeps == [2.0, 4.0]  # The exponential backoff of attempts 1 and 2, as the service's pause


def test_backoff_gives_the_place_to_other_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request backing off after an error does not keep its place of the gate: with concurrency 1, another
    request is sent while it waits, and the retry comes after."""
    other_sent = asyncio.Event()

    async def until_other_sent() -> None:
        await asyncio.wait_for(other_sent.wait(), 2)

    _record_sleeps(monkeypatch, until_other_sent)
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if "Second" in body:
            sent.append("second")
            other_sent.set()
            return httpx.Response(200, json=OK)
        sent.append("first")
        if sent.count("first") == 1:
            return httpx.Response(500, json={"error": {"message": "upstream hiccup"}})
        return httpx.Response(200, json=OK)

    async def scenario() -> list[list[str]]:
        engine = _engine("svc", httpx.MockTransport(handler), concurrency=1)
        try:
            first = asyncio.create_task(engine.translate_batch(["First."]))
            second = asyncio.create_task(engine.translate_batch(["Second."]))
            return list(await asyncio.gather(first, second))
        finally:
            await engine.aclose()

    assert run(scenario()) == [["好"], ["好"]]
    assert sent == ["first", "second", "first"]
