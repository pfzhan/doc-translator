"""The per-service reasoning setting: what each provider is sent, and the fallback when the API rejects it."""

import json
from pathlib import Path

import httpx
import pytest

from app.json_types import JSONDict
from app.services import ServiceError, ServiceStore
from app.translators import LLMTranslator, TranslatorError, create_translator
from tests.helpers import run

REPLIES: dict[str, JSONDict] = {
    "openai": {"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}]},
    "responses": {"status": "completed", "output_text": "好"},
    "claude": {"content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn"},
    "gemini": {"candidates": [{"content": {"parts": [{"text": "好"}]}, "finishReason": "STOP"}]},
}


def _engine(provider: str, reasoning: str, sent: list[JSONDict], reject: str = "") -> LLMTranslator:
    """An engine whose requests are recorded in sent. With reject, a request whose body mentions it gets a 400
    naming it, the way an API turns down a parameter it does not take."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if reject and reject in request.content.decode():
            return httpx.Response(400, json={"error": {"message": f"Unsupported parameter: '{reject}'"}})
        return httpx.Response(200, json=REPLIES[provider])

    service: JSONDict = {
        "provider": "openai" if provider == "responses" else provider,
        "api_key": "k",
        "base_url": "https://api.example.com/v1",
        "model": "m",
        "temperature": 0,
        "reasoning": reasoning,
        "api_format": "responses" if provider == "responses" else "",
    }
    engine = create_translator(service, "zh-CN", transport=httpx.MockTransport(handler))
    assert isinstance(engine, LLMTranslator)
    return engine


def _send(provider: str, reasoning: str, reject: str = "") -> list[JSONDict]:
    sent: list[JSONDict] = []
    engine = _engine(provider, reasoning, sent, reject)
    assert run(engine.translate_batch(["good"])) == ["好"]
    run(engine.aclose())
    return sent


@pytest.mark.parametrize("provider", list(REPLIES))
def test_unset_sends_nothing(provider: str) -> None:
    """Without a setting the request is what it was before the setting existed."""
    body = _send(provider, "")[0]
    assert not {"reasoning_effort", "reasoning", "thinking", "output_config"} & set(body)
    assert "thinkingConfig" not in body.get("generationConfig", {})
    assert body.get("temperature", body.get("generationConfig", {}).get("temperature")) == 0
    if provider == "claude":
        assert body["max_tokens"] == 8192


@pytest.mark.parametrize(
    "provider,reasoning,expected",
    [
        ("openai", "minimal", {"reasoning_effort": "minimal", "temperature": 0}),
        ("openai", "none", {"reasoning_effort": "none"}),
        ("responses", "high", {"reasoning": {"effort": "high"}, "temperature": 0}),
        ("claude", "none", {"thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": 8192}),
        (
            "claude",
            "medium",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}, "max_tokens": 16000},
        ),
        # Claude has no minimal effort; the lowest it has is low
        ("claude", "minimal", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}),
        ("gemini", "none", {"generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingBudget": 0}}}),
        ("gemini", "low", {"generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingBudget": 1024}}}),
    ],
)
def test_each_provider_gets_its_own_parameter(provider: str, reasoning: str, expected: JSONDict) -> None:
    body = _send(provider, reasoning)[0]
    assert {k: body.get(k) for k in expected} == expected


def test_claude_thinking_sends_no_temperature() -> None:
    """A thinking Claude model takes no temperature (older ones only 1), so it is left out rather than refused."""
    assert "temperature" not in _send("claude", "high")[0]


@pytest.mark.parametrize(
    "provider,reasoning,reject,key",
    [
        ("openai", "low", "reasoning_effort", "reasoning_effort"),
        ("responses", "low", "reasoning", "reasoning"),
        ("claude", "none", "thinking", "thinking"),
        ("claude", "high", "output_config", "output_config"),
    ],
)
def test_a_rejected_setting_is_dropped_and_the_request_sent_again(
    provider: str, reasoning: str, reject: str, key: str
) -> None:
    """Like temperature: a model or relay that does not know the parameter answers 400; it is dropped for the rest of
    the engine's requests and the request sent once more, so the job carries on with the model's default."""
    sent: list[JSONDict] = []
    engine = _engine(provider, reasoning, sent, reject)
    assert run(engine.translate_batch(["good"])) == ["好"]
    assert run(engine.translate_batch(["more"])) == ["好"]
    run(engine.aclose())
    assert len(sent) == 3 and key in sent[0] and all(key not in body for body in sent[1:])
    assert engine.reasoning == ""
    # Claude's temperature comes back once thinking is no longer asked for
    if provider == "claude":
        assert sent[1]["temperature"] == 0 and sent[1]["max_tokens"] == 8192


def test_temperature_is_dropped_before_reasoning() -> None:
    """A 400 about temperature drops temperature only; the reasoning setting stays."""
    sent = _send("openai", "low", reject="temperature")
    assert len(sent) == 2 and "temperature" not in sent[1] and sent[1]["reasoning_effort"] == "low"


def test_an_unrelated_400_keeps_the_setting() -> None:
    sent: list[JSONDict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "Invalid API key"}})

    engine = create_translator(
        {"provider": "openai", "api_key": "k", "model": "m", "reasoning": "low"},
        "zh-CN",
        transport=httpx.MockTransport(handler),
    )
    assert isinstance(engine, LLMTranslator)
    with pytest.raises(TranslatorError):
        run(engine.translate_batch(["good"]))
    run(engine.aclose())
    assert sent and all(body["reasoning_effort"] == "low" for body in sent) and engine.reasoning == "low"


def test_the_setting_is_not_part_of_the_cache_key() -> None:
    """Like temperature it tunes the same model on the same prompt: changing it does not retranslate the book, and
    dropping it after a 400 mid-run does not move the rest of the job to another cache."""
    keys = {_engine("openai", level, []).cache_key for level in ("", "none", "high")}
    assert len(keys) == 1


def test_store_validates_the_setting(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "s.json")
    svc = store.create({"provider": "openai", "name": "o", "api_key": "sk-1234567890", "reasoning": " high "})
    assert svc["reasoning"] == "high"
    assert store.update(svc["id"], {"reasoning": ""})["reasoning"] == ""
    with pytest.raises(ServiceError, match="推理强度"):
        store.update(svc["id"], {"reasoning": "extreme"})
    with pytest.raises(TranslatorError, match="推理强度"):
        create_translator({"provider": "openai", "api_key": "k", "model": "m", "reasoning": "extreme"}, "zh-CN")
