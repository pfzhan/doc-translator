"""Replies the model or a filter declined are not taken as translations, whatever text they carry."""

from pathlib import Path

import httpx
import pytest

from app.json_types import JSONDict
from app.request_log import RequestLog, read_records
from app.translators import ClaudeTranslator, GeminiTranslator, LLMTranslator, OpenAITranslator
from tests.helpers import run


def _reply(provider: str, text: str) -> JSONDict:
    """A normal, finished reply in the provider's format."""
    if provider == "openai":
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    if provider == "responses":
        return {"status": "completed", "output_text": text}
    if provider == "claude":
        return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


# Declined replies, each with the beginning of a translation that must not be used
PARTIAL = "[[p0]]\n好。\n[[p1]]\n危"
DECLINED: list[tuple[str, JSONDict]] = [
    ("claude", {"content": [{"type": "text", "text": PARTIAL}], "stop_reason": "refusal"}),
    ("claude", {"content": [{"type": "text", "text": PARTIAL}], "stop_reason": "pause_turn"}),
    ("openai", {"choices": [{"message": {"content": PARTIAL}, "finish_reason": "content_filter"}]}),
    ("openai", {"choices": [{"message": {"content": None, "refusal": "I can't help."}, "finish_reason": "stop"}]}),
    (
        "responses",
        {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}, "output_text": PARTIAL},
    ),
    (
        "responses",
        {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "I can't help."}]}],
        },
    ),
    ("gemini", {"candidates": [{"content": {"parts": [{"text": PARTIAL}]}, "finishReason": "SAFETY"}]}),
    ("gemini", {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}),
]


def _engine(provider: str, declined: JSONDict, log: RequestLog) -> LLMTranslator:
    """A service that declines every request containing "Dangerous" and translates the others."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if "Dangerous" in body:
            return httpx.Response(200, json=declined)
        return httpx.Response(200, json=_reply(provider, "好。"))

    classes: dict[str, type[LLMTranslator]] = {"claude": ClaudeTranslator, "gemini": GeminiTranslator}
    engine = classes.get(provider, OpenAITranslator)(
        "zh-CN",
        "en",
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        api_format="responses" if provider == "responses" else "",
        transport=httpx.MockTransport(handler),
    )
    engine.request_log = log
    return engine


@pytest.mark.parametrize("provider,declined", DECLINED)
def test_a_declined_batch_is_translated_paragraph_by_paragraph(
    provider: str, declined: JSONDict, tmp_path: Path
) -> None:
    """A declined reply counts as empty: the batch is asked again, then translated paragraph by paragraph, so only
    the paragraph that set off the filter stays untranslated (Runner keeps its original and caches nothing). The
    partial text of the declined reply is never used, and the job does not fail."""
    log = RequestLog(tmp_path / "requests.jsonl")
    engine = _engine(provider, declined, log)
    assert run(engine.translate_batch(["Fine.", "Dangerous."])) == ["好。", ""]
    run(engine.aclose())
    log.flush()
    records = read_records(tmp_path / "requests.jsonl")
    # The batch twice, then each paragraph; the declined one is asked twice too
    assert [r.reason for r in records].count("retranslate") == 2
    assert all(r.stop for r in records)  # The request log says why each reply stopped


@pytest.mark.parametrize(
    "provider,reply",
    [
        ("claude", {"content": [{"type": "text", "text": "好。"}], "stop_reason": "stop_sequence"}),
        ("claude", {"content": [{"type": "text", "text": "好。"}]}),  # Relays may leave stop_reason out
        ("openai", {"choices": [{"message": {"content": "好。"}, "finish_reason": "eos"}]}),
        ("openai", {"choices": [{"message": {"content": "好。", "refusal": None}}]}),
        ("gemini", {"candidates": [{"content": {"parts": [{"text": "好。"}]}}]}),
    ],
)
def test_finished_replies_are_taken(provider: str, reply: JSONDict, tmp_path: Path) -> None:
    engine = _engine(provider, reply, RequestLog(tmp_path / "requests.jsonl"))
    assert run(engine.translate_batch(["Dangerous."])) == ["好。"]
    run(engine.aclose())


def test_a_reply_cut_at_the_context_window_is_split(tmp_path: Path) -> None:
    """Claude's model_context_window_exceeded is a cut-off reply like max_tokens: the batch is split."""
    sent: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        sent.append(sum(word in body for word in ("Alpha", "Beta")))
        if sent[-1] > 1:
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": PARTIAL}], "stop_reason": "model_context_window_exceeded"},
            )
        return httpx.Response(200, json=_reply("claude", "好。"))

    engine = ClaudeTranslator(
        "zh-CN",
        "en",
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        transport=httpx.MockTransport(handler),
    )
    assert run(engine.translate_batch(["Alpha one.", "Beta two."])) == ["好。", "好。"]
    run(engine.aclose())
    assert sorted(sent) == [1, 1, 2]
