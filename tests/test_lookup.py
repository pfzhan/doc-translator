"""Looking up a word selected on a web page: the prompt, parsing the model's entry, the engines and the route."""

import json
from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from app.json_types import JSONDict
from app.lookup import MAX_SENSES, parse_entry
from app.prompts import build_lookup_messages
from app.routes import web
from app.translators import ClaudeTranslator, MockTranslator, OpenAITranslator, TranslatorError
from tests.helpers import run

ENTRY: JSONDict = {
    "headword": "bank",
    "phonetic": "/bæŋk/",
    "senses": [{"pos": "n.", "meaning": "银行"}, {"pos": "n.", "meaning": "河岸；堤"}],
    "in_context": "河岸",
}
SENTENCE = "They sat on the bank of the river."


def openai(replies: list[str], seen: list[JSONDict] | None = None, target: str = "zh-CN") -> OpenAITranslator:
    """An engine whose model gives these replies in turn."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

    return OpenAITranslator(
        target, api_key="k", base_url="https://api.example.com/v1", model="m", transport=httpx.MockTransport(handler)
    )


def test_the_prompt_carries_the_selection_its_sentence_and_the_language() -> None:
    system, user = build_lookup_messages("bank", SENTENCE, "zh-CN", title="Rivers")
    assert "Chinese" in system and "Rivers" in system
    assert user == f"Selection: bank\n\nSentence: {SENTENCE}"
    assert build_lookup_messages("bank", " ", "zh-CN")[1].endswith("Sentence: (none)")
    assert "instead" not in system


def test_a_word_in_the_target_language_is_explained_in_the_second_one() -> None:
    system, _ = build_lookup_messages("银行", SENTENCE, "zh-CN", alt_lang="en")
    assert "When the selection itself is Simplified Chinese, write every explanation in English instead." in system
    assert "{{" not in system
    # Without a second language, or with the target as the second one, the target language is used throughout
    assert "instead" not in build_lookup_messages("银行", "", "zh-CN", alt_lang="zh-CN")[0]


def test_placeholders_in_the_page_text_stay_as_written() -> None:
    _, user = build_lookup_messages("{{context}}", "a {{text}} b", "zh-CN")
    assert user == "Selection: {{context}}\n\nSentence: a {{text}} b"


def test_an_entry_is_read_from_a_fence_or_a_sentence_around_it() -> None:
    raw = json.dumps(ENTRY, ensure_ascii=False)
    for reply in (raw, f"```json\n{raw}\n```", f"Here is the entry:\n{raw}\nHope it helps."):
        entry = parse_entry(reply, "banks")
        assert entry is not None
        assert entry.headword == "bank" and entry.phonetic == "/bæŋk/"
        assert [(s.pos, s.meaning) for s in entry.senses] == [("n.", "银行"), ("n.", "河岸；堤")]


def test_missing_fields_are_filled_and_extra_senses_cut() -> None:
    senses = [{"meaning": f"义{i}"} for i in range(MAX_SENSES + 3)] + [{"pos": "n.", "meaning": " "}]
    entry = parse_entry(json.dumps({"senses": senses}), "banks")
    assert entry is not None
    assert entry.headword == "banks" and entry.phonetic == "" and entry.in_context == ""
    assert [s.meaning for s in entry.senses] == [f"义{i}" for i in range(MAX_SENSES)]
    assert entry.senses[0].pos == ""


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "银行；河岸",  # prose, not an entry
        '{"headword": "bank", "senses": [',  # cut off
        '{"headword": bank}',  # not JSON
        "[1, 2]",
        '{"headword": "bank", "senses": "银行"}',  # senses of the wrong shape
        '{"headword": "bank", "senses": []}',  # nothing to show
    ],
)
def test_a_reply_that_is_not_an_entry_gives_none(reply: str) -> None:
    assert parse_entry(reply, "bank") is None


def test_the_model_is_asked_with_the_sentence_and_its_entry_returned() -> None:
    seen: list[JSONDict] = []
    tr = openai([json.dumps(ENTRY)], seen)
    entry = run(tr.lookup("bank", SENTENCE))
    assert entry.in_context == "河岸" and entry.lang == "zh-CN"
    assert len(seen) == 1
    assert SENTENCE in seen[0]["messages"][-1]["content"]


def test_a_reply_out_of_format_is_asked_for_once_more() -> None:
    tr = openai(["银行", json.dumps(ENTRY)])
    assert run(tr.lookup("bank", SENTENCE)).headword == "bank"


def test_a_second_reply_out_of_format_is_an_error() -> None:
    tr = openai(["银行", "还是银行"])
    with pytest.raises(TranslatorError, match="释义"):
        run(tr.lookup("bank", SENTENCE))


def test_a_declined_lookup_is_asked_for_once_more() -> None:
    replies = [
        {"stop_reason": "refusal", "content": [{"type": "text", "text": "I can't"}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(ENTRY)}]},
    ]
    tr = ClaudeTranslator(
        "zh-CN",
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=replies.pop(0))),
    )
    assert run(tr.lookup("bank", SENTENCE)).senses[0].meaning == "银行"
    assert replies == []


@pytest.mark.parametrize(
    ("in_target", "alt", "lang"),
    [(False, "en", "zh-CN"), (True, "en", "en"), (True, "", "zh-CN"), (True, "zh-CN", "zh-CN")],
)
def test_the_entry_says_which_language_it_is_written_in(in_target: bool, alt: str, lang: str) -> None:
    tr = openai([json.dumps({**ENTRY, "in_target_language": in_target, "lang": "fr"})])
    entry = run(tr.lookup("银行", SENTENCE, alt))
    assert entry.lang == lang
    assert "in_target_language" not in entry.model_dump()


def lookup(main: ModuleType, body: JSONDict) -> httpx.Response:
    resp: httpx.Response = TestClient(main.app).post("/api/web/lookup", json=body)
    return resp


def with_model(monkeypatch: pytest.MonkeyPatch, replies: list[str], seen: list[JSONDict]) -> list[str]:
    """Make the route's services LLM engines giving these replies; returns the target languages they were made
    for."""
    targets: list[str] = []

    def create(service: JSONDict, target_lang: str, source_lang: str = "auto") -> OpenAITranslator:
        targets.append(target_lang)
        return openai(replies, seen, target_lang)

    monkeypatch.setattr(web, "create_translator", create)
    return targets


def test_the_route_answers_with_the_entry(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[JSONDict] = []
    reply = {**ENTRY, "headword": "银行", "in_target_language": True}
    targets = with_model(monkeypatch, [json.dumps(reply, ensure_ascii=False)], seen)
    body = {"text": " 银行 ", "context": "他去银行取钱。", "alt_lang": "en", "service_id": "mock", "title": "T"}
    resp = lookup(app_env, body)
    assert resp.status_code == 200
    assert resp.json() == {
        "headword": "银行",
        "phonetic": "/bæŋk/",
        "senses": ENTRY["senses"],
        "in_context": "河岸",
        "lang": "en",
    }
    assert targets == ["zh-CN"]
    system, user = (m["content"] for m in seen[0]["messages"])
    assert "in English instead" in system and "T" in system
    assert user == "Selection: 银行\n\nSentence: 他去银行取钱。"


def test_the_route_closes_the_engine_when_the_lookup_fails(
    app_env: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    with_model(monkeypatch, ["银行", "还是银行"], [])

    async def aclose(self: OpenAITranslator) -> None:
        closed.append(True)

    monkeypatch.setattr(OpenAITranslator, "aclose", aclose)
    resp = lookup(app_env, {"text": "bank", "service_id": "mock"})
    assert resp.status_code == 502
    assert resp.json()["detail"] == "模型没有按格式给出释义"
    assert closed == [True]


def test_an_engine_without_a_model_does_not_look_up(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The extension translates the selection instead, which keeps a word already in the target language apart."""
    closed: list[bool] = []

    async def aclose(self: MockTranslator) -> None:
        closed.append(True)

    monkeypatch.setattr(MockTranslator, "aclose", aclose)
    resp = lookup(app_env, {"text": "bank", "service_id": "mock"})
    assert resp.status_code == 501
    assert closed == [True]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"text": "bank", "service_id": "missing"}, 400),
        ({"text": "bank", "service_id": "mock", "target_lang": "xx"}, 400),
        ({"text": "bank", "service_id": "mock", "alt_lang": "xx"}, 400),
        ({"text": "bank", "service_id": "mock", "alt_lang": "auto"}, 400),
        ({"text": "   ", "service_id": "mock"}, 400),
        ({"text": "", "service_id": "mock"}, 422),
        ({"text": "w" * 101, "service_id": "mock"}, 422),
        ({"text": "bank", "context": "c" * 2001, "service_id": "mock"}, 422),
    ],
)
def test_the_route_rejects_what_it_cannot_look_up(app_env: ModuleType, body: JSONDict, status: int) -> None:
    assert lookup(app_env, body).status_code == status
