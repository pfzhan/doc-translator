"""The per-service requests-per-minute limit: a sliding window shared by every engine of the service."""

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from app.json_types import JSONDict
from app.services import ServiceError, ServiceStore
from app.translators import LLMTranslator, create_translator
from tests.helpers import run

OK = {"choices": [{"message": {"content": "好"}}]}


class FakeTime:
    """A clock that moves only when the code under test sleeps, so waits are exact and take no real time."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        # Called once after the next sleep, as if something happened meanwhile
        self.during_sleep: Callable[[], None] | None = None
        monkeypatch.setattr("app.translators._clock", lambda: self.now)
        monkeypatch.setattr("app.translators.asyncio.sleep", self.sleep)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        during = self.during_sleep
        if during is not None:
            self.during_sleep = None
            during()


def _engine(sid: str, rpm: object, sent: list[float], clock: FakeTime) -> LLMTranslator:
    """An engine of service sid whose requests are recorded in sent, by the fake clock's time."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(clock.now)
        return httpx.Response(200, json=OK)

    service: JSONDict = {"id": sid, "provider": "custom", "base_url": "http://x/v1", "model": "m", "rpm": rpm}
    engine = create_translator(service, "zh-CN", "en", transport=httpx.MockTransport(handler))
    assert isinstance(engine, LLMTranslator)
    return engine


def test_requests_beyond_the_limit_wait_for_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """With rpm 2, the third request waits until the first is a minute old, and the jobs of one service share the
    window: a second engine of the same service waits too, while another service does not."""
    clock = FakeTime(monkeypatch)
    sent: list[float] = []
    other_sent: list[float] = []

    async def scenario() -> None:
        first, second = _engine("svc", 2, sent, clock), _engine("svc", 2, sent, clock)
        other = _engine("other", 2, other_sent, clock)
        try:
            assert await first.translate_batch(["One."]) == ["好"]
            clock.now += 10
            assert await first.translate_batch(["Two."]) == ["好"]
            assert await second.translate_batch(["Three."]) == ["好"]
            assert await other.translate_batch(["Elsewhere."]) == ["好"]
            assert await second.translate_batch(["Four."]) == ["好"]
        finally:
            for engine in (first, second, other):
                await engine.aclose()

    run(scenario())
    # Three at 1060 is a minute after One; Four waits for Two (1010), a minute later at 1070
    assert sent == [1000.0, 1010.0, 1060.0, 1070.0]
    assert other_sent == [1060.0]
    assert clock.sleeps == [50.0, 10.0]


@pytest.mark.parametrize("rpm", [None, "", 0])
def test_no_limit_by_default(monkeypatch: pytest.MonkeyPatch, rpm: object) -> None:
    clock = FakeTime(monkeypatch)
    sent: list[float] = []

    async def scenario() -> None:
        engine = _engine("svc", rpm, sent, clock)
        try:
            for i in range(30):
                assert await engine.translate_batch([f"Line {i}."]) == ["好"]
        finally:
            await engine.aclose()

    run(scenario())
    assert len(sent) == 30 and clock.sleeps == []


def test_a_pause_starting_while_waiting_for_the_window_is_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the window has room the request still waits for a 429 pause that began in the meantime."""
    clock = FakeTime(monkeypatch)
    sent: list[float] = []

    async def scenario() -> None:
        engine = _engine("svc", 1, sent, clock)
        try:
            assert await engine.translate_batch(["One."]) == ["好"]
            # While Two waits for the window, another request of the service is told to wait 100 seconds
            clock.during_sleep = lambda: engine._pause(100)
            assert await engine.translate_batch(["Two."]) == ["好"]
        finally:
            await engine.aclose()

    run(scenario())
    assert sent == [1000.0, 1160.0]
    assert clock.sleeps == [60.0, 100.0]


def test_store_cleans_the_limit(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "s.json")
    svc = store.create({"provider": "openai", "name": "o", "api_key": "sk-1234567890", "rpm": "30"})
    assert svc["rpm"] == 30
    assert store.update(svc["id"], {"rpm": ""})["rpm"] is None
    assert store.update(svc["id"], {"rpm": "0"})["rpm"] is None
    assert store.update(svc["id"], {"rpm": 99999})["rpm"] == 10000
    with pytest.raises(ServiceError, match="每分钟请求数"):
        store.update(svc["id"], {"rpm": "fast"})
