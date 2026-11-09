"""Token usage in one shape for every provider, tolerant of what relays send."""

import json
from pathlib import Path

import httpx
import pytest

from app.history import Job, JobStore
from app.json_types import JSONDict
from app.request_log import RequestLog, read_records
from app.translators import ClaudeTranslator, GeminiTranslator, LLMTranslator, OpenAITranslator
from app.usage import Usage, count
from tests.helpers import run


@pytest.mark.parametrize(
    "value,expected",
    [(12, 12), ("34", 34), (" 56 ", 56), (7.0, 7), ("8.0", 8), (0, 0), ("0", 0)]
    + [(None, None), ("", None), ("n/a", None), (-1, None), (True, None), ([1], None), ({"n": 1}, None)]
    + [(float("nan"), None), (float("inf"), None)],
)
def test_counts_tolerate_what_relays_send(value: object, expected: int | None) -> None:
    assert count(value) == expected


def _usage_of(provider: str, reply: JSONDict, tmp_path: Path) -> tuple[Usage, list[int | None]]:
    """The usage an engine of provider adds up from one reply, and the token fields of its request record."""
    classes: dict[str, type[LLMTranslator]] = {"claude": ClaudeTranslator, "gemini": GeminiTranslator}
    tr = classes.get(provider, OpenAITranslator)(
        "zh-CN",
        "en",
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        api_format="responses" if provider == "responses" else "",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply)),
    )
    tr.request_log = RequestLog(tmp_path / "requests.jsonl")
    assert run(tr.translate_batch(["Hello."])) == ["你好。"]
    run(tr.aclose())
    tr.request_log.flush()
    [record] = read_records(tmp_path / "requests.jsonl")
    fields = [
        record.input_tokens,
        record.output_tokens,
        record.reasoning_tokens,
        record.cached_tokens,
        record.cache_write_tokens,
    ]
    return tr.usage, fields


CHAT = {"choices": [{"message": {"content": "你好。"}, "finish_reason": "stop"}]}
RESPONSES = {"status": "completed", "output_text": "你好。"}
CLAUDE = {"content": [{"type": "text", "text": "你好。"}], "stop_reason": "end_turn"}
GEMINI = {"candidates": [{"content": {"parts": [{"text": "你好。"}]}, "finishReason": "STOP"}]}


@pytest.mark.parametrize(
    "provider,reply,expected",
    [
        (
            "openai",
            {
                **CHAT,
                "usage": {
                    "prompt_tokens": "1000",  # A relay sending strings
                    "completion_tokens": 300,
                    "prompt_tokens_details": {"cached_tokens": 800},
                    "completion_tokens_details": {"reasoning_tokens": "200"},
                },
            },
            Usage(input=1000, output=300, reasoning=200, cached_input=800, reported=True),
        ),
        (
            "openai",
            {**CHAT, "usage": {"prompt_tokens": 90, "completion_tokens": 10, "prompt_cache_hit_tokens": 64}},
            Usage(input=90, output=10, cached_input=64, reported=True),
        ),
        (
            "responses",
            {
                **RESPONSES,
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 256},
                    "output_tokens_details": {"reasoning_tokens": 64},
                },
            },
            Usage(input=500, output=120, reasoning=64, cached_input=256, reported=True),
        ),
        (
            # Claude leaves the cache out of input_tokens; all three are billed input
            "claude",
            {
                **CLAUDE,
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 300,
                },
            },
            Usage(input=1320, output=50, cached_input=1000, cache_write=300, reported=True),
        ),
        (
            "gemini",
            {
                **GEMINI,
                "usageMetadata": {
                    "promptTokenCount": 400,
                    "candidatesTokenCount": 40,
                    "thoughtsTokenCount": 60,
                    "cachedContentTokenCount": 128,
                },
            },
            Usage(input=400, output=100, reasoning=60, cached_input=128, reported=True),
        ),
        # Real zeros are reported zeros
        ("openai", {**CHAT, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}, Usage(reported=True)),
        # Relays that leave usage out, or send something unreadable, report nothing
        ("openai", CHAT, Usage()),
        ("openai", {**CHAT, "usage": None}, Usage()),
        ("openai", {**CHAT, "usage": "n/a"}, Usage()),
        ("claude", {**CLAUDE, "usage": {"input_tokens": None}}, Usage()),
        ("gemini", GEMINI, Usage()),
    ],
)
def test_usage_of_every_provider(provider: str, reply: JSONDict, expected: Usage, tmp_path: Path) -> None:
    usage, record = _usage_of(provider, reply, tmp_path)
    assert usage == expected
    if expected.reported:
        tokens = [usage.input, usage.output, usage.reasoning, usage.cached_input, usage.cache_write]
        assert record == tokens
    else:
        assert record == [None] * 5  # The request log tells "not reported" from zeros too


def test_usage_adds_up_and_stays_reported() -> None:
    total = Usage()
    total.add(Usage())
    assert not total.reported
    total.add(Usage(input=10, output=2, cached_input=5, reported=True))
    total.add(Usage())
    total.add(Usage(input=1, output=1, reasoning=1, cache_write=3, reported=True))
    assert total == Usage(input=11, output=3, reasoning=1, cached_input=5, cache_write=3, reported=True)


def test_records_keep_their_token_counts(tmp_path: Path) -> None:
    """Records from before the breakdown still say their usage was reported; a run adds every part, and a run
    without usage leaves the counts and the flag as they were."""
    store = JobStore(tmp_path / "jobs")
    store.dir("old").mkdir(parents=True)
    store.dir("none").mkdir(parents=True)
    legacy = {"id": "old", "filename": "a.md", "status": "done", "prompt_tokens": 100, "completion_tokens": 20}
    (store.dir("old") / "job.json").write_text(json.dumps(legacy), encoding="utf-8")
    silent = {"id": "none", "filename": "b.md", "status": "done", "prompt_tokens": 0, "completion_tokens": 0}
    (store.dir("none") / "job.json").write_text(json.dumps(silent), encoding="utf-8")
    loaded = JobStore(tmp_path / "jobs")
    old, none = loaded.jobs["old"], loaded.jobs["none"]
    assert old.tokens_reported and not none.tokens_reported
    none.add_usage(Usage())
    assert not none.to_dict()["tokens_reported"]
    old.add_usage(Usage(input=10, output=5, reasoning=2, cached_input=4, cache_write=1, reported=True))
    data = old.to_dict()
    assert (data["prompt_tokens"], data["completion_tokens"], data["reasoning_tokens"]) == (110, 25, 2)
    assert (data["cached_tokens"], data["cache_write_tokens"], data["tokens_reported"]) == (4, 1, True)
    old.clear_results()
    fresh = Job(id="old", filename="a.md")
    for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cache_write_tokens"):
        assert getattr(old, name) == getattr(fresh, name) == 0
    assert not old.tokens_reported
