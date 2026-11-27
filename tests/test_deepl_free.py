"""DeepL's free web endpoint: request format, language tags, the length limit of a request, the check for answers
that are the original text, errors, and its built-in service."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main, state
from app.deepl_free import MAX_REQUEST_UNITS, pack, split_paragraph, target_tag, utf16_len
from app.services import ServiceError, ServiceStore
from app.translators import DEEPL_FREE_URL_WEB, DeepLFreeTranslator, TranslatorError, create_translator
from tests.helpers import run


async def _no_sleep(_seconds: float) -> None:
    return None


def _echo(request: httpx.Request) -> httpx.Response:
    """The endpoint's answer: each text prefixed with the target language, found to be English."""
    body = json.loads(request.content)
    texts: list[str] = body["text"]
    assert utf16_len("".join(texts)) <= MAX_REQUEST_UNITS, "the endpoint answers a longer request with a 400"
    return httpx.Response(
        200,
        json={
            "translations": [{"detected_source_language": "en", "text": f"{body['target_lang']}:{t}"} for t in texts]
        },
    )


def _unchanged(detected: str) -> httpx.MockTransport:
    """The endpoint's answer to a target it does not take: the original text, with the language it found."""

    def handler(request: httpx.Request) -> httpx.Response:
        texts: list[str] = json.loads(request.content)["text"]
        return httpx.Response(
            200, json={"translations": [{"detected_source_language": detected, "text": t} for t in texts]}
        )

    return httpx.MockTransport(handler)


def _free(handler: httpx.MockTransport, target: str = "zh-CN", source: str = "auto") -> DeepLFreeTranslator:
    return DeepLFreeTranslator(target, source, transport=handler)


def test_request_format() -> None:
    """No key and no context; the target is a BCP 47 tag, as the endpoint answers DE or ZH-HANS with the original."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _echo(request)

    tr = _free(httpx.MockTransport(handler), "zh-CN", "en")
    tr.title = "Alice in Wonderland"
    assert run(tr.translate_batch(["one", "two"])) == ["zh-Hans:one", "zh-Hans:two"]
    request = seen[0]
    assert str(request.url) == DEEPL_FREE_URL_WEB
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {"text": ["one", "two"], "target_lang": "zh-Hans", "source_lang": "EN"}
    run(tr.aclose())


@pytest.mark.parametrize(
    "code,tag", [("ZH-HANS", "zh-Hans"), ("ZH-HANT", "zh-Hant"), ("EN-US", "en-US"), ("PT-BR", "pt-BR"), ("NB", "nb")]
)
def test_target_tags(code: str, tag: str) -> None:
    assert target_tag(code) == tag


def test_thai_is_not_offered() -> None:
    """The endpoint answers a Thai target with the original text, so Thai fails before any request."""
    with pytest.raises(TranslatorError, match="不支持目标语言"):
        DeepLFreeTranslator("th")
    with pytest.raises(TranslatorError, match="不支持源语言"):
        DeepLFreeTranslator("de", "th")


def test_paragraphs_are_packed_within_the_length_limit() -> None:
    """All paragraphs of a request together stay within 1500 UTF-16 code units; answers come back in order."""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["text"]))
        return _echo(request)

    texts = [f"Paragraph {i}. " + "word " * 100 for i in range(12)]  # about 515 units each
    tr = _free(httpx.MockTransport(handler), "de")
    assert run(tr.translate_batch(texts)) == [f"de:{t}" for t in texts]
    assert sizes == [2, 2, 2, 2, 2, 2]
    run(tr.aclose())


def test_emoji_count_twice() -> None:
    """The limit counts UTF-16 code units, as the endpoint does: an emoji is two."""
    assert utf16_len("a😀") == 3
    assert pack([800, 700, 1]) == [[0, 1], [2]]
    assert pack([1] * 120, max_items=50) == [list(range(50)), list(range(50, 100)), list(range(100, 120))]


def test_long_paragraph_is_cut_after_a_sentence() -> None:
    sentence = "The quick brown fox jumps over the lazy dog, again and again, near the quiet bank. "
    text = (sentence * 30).strip()  # about 2500 units
    pieces = split_paragraph(text, html_text=False)
    assert len(pieces) == 2
    assert all(utf16_len(piece) <= MAX_REQUEST_UNITS for piece, _ in pieces)
    assert pieces[0][0].endswith("bank.") and pieces[0][1] == " "
    assert "".join(piece + gap for piece, gap in pieces) == text


def test_cut_pieces_are_joined_with_their_white_space() -> None:
    """Pieces of a cut paragraph go back together with the space between them, which Chinese and Japanese drop
    unless it is a line break."""
    sentence = "The quick brown fox jumps over the lazy dog, again and again, near the quiet bank. "
    text = (sentence * 30).strip()
    de = _free(httpx.MockTransport(_echo), "de")
    [out] = run(de.translate_batch([text]))
    assert out.count("de:") == 2 and "bank. de:The" in out
    zh = _free(httpx.MockTransport(_echo), "zh-CN")
    [out] = run(zh.translate_batch([text]))
    assert "bank.zh-Hans:The" in out
    run(de.aclose())
    run(zh.aclose())


def test_text_without_a_break_is_cut_at_the_limit() -> None:
    text = "x" * 3200
    pieces = split_paragraph(text, html_text=False)
    assert [len(p) for p, _ in pieces] == [1500, 1500, 200]


def test_rich_text_is_cut_outside_its_tags() -> None:
    sentence = 'Read <a i="1">the long guide</a> before <b i="2">you start</b> working on it today. '
    text = (sentence * 25).strip()
    pieces = split_paragraph(text, html_text=True)
    assert len(pieces) > 1
    for piece, _ in pieces:
        assert utf16_len(piece) <= MAX_REQUEST_UNITS
        assert piece.count("<a ") == piece.count("</a>") and piece.count("<b ") == piece.count("</b>")


def test_rich_text_that_cannot_be_cut_loses_its_tags() -> None:
    """A link longer than the limit leaves no place to cut outside a tag; its text goes as escaped HTML instead."""
    text = '<a i="1">' + "Tom & Jerry run. " * 120 + "</a>"
    pieces = split_paragraph(text, html_text=True)
    assert len(pieces) > 1
    joined = "".join(piece + gap for piece, gap in pieces)
    assert "<a" not in joined and "Tom &amp; Jerry" in joined


def test_blank_paragraphs_are_not_sent() -> None:
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["text"])
        return _echo(request)

    tr = _free(httpx.MockTransport(handler), "de")
    assert run(tr.translate_batch(["  ", "Hello"])) == ["  ", "de:Hello"]
    assert sent == [["Hello"]]
    run(tr.aclose())


def test_original_text_answered_for_foreign_text_is_an_error() -> None:
    """The endpoint answers a target it does not take with the original text and a 200; a whole request of foreign
    text answered that way is an error, not a translation that keeps the original."""
    long_text = "This sentence is plainly English and should have been translated into German. " * 4
    tr = _free(_unchanged("en"), "de")
    with pytest.raises(TranslatorError, match="原样返回了原文"):
        run(tr.translate_batch([long_text]))
    run(tr.aclose())


@pytest.mark.parametrize(
    "detected,text",
    [
        ("en", "Ludwig van Beethoven"),  # a name may stay as it is
        ("de", "Dieser Satz ist schon auf Deutsch und muss nicht übersetzt werden, er bleibt so. " * 4),
    ],
)
def test_unchanged_short_or_same_language_text_is_kept(detected: str, text: str) -> None:
    tr = _free(_unchanged(detected), "de")
    assert run(tr.translate_batch([text])) == [text]
    run(tr.aclose())


@pytest.mark.parametrize(
    "status,message",
    [(403, "拒绝了请求"), (404, "已不可用"), (456, "额度已用完"), (400, "免费接口错误 400")],
)
def test_errors_retrying_does_not_fix_fail_at_once(status: int, message: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json={"title": "nope"})

    tr = _free(httpx.MockTransport(handler))
    with pytest.raises(TranslatorError, match=message):
        run(tr.translate_batch(["x"]))
    assert len(calls) == 1
    run(tr.aclose())


@pytest.mark.parametrize("status", [429, 529])
def test_rate_limits_pause_the_service_and_are_retried(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    """529 is the endpoint's "too many requests": like 429 it holds back every request to it."""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    monkeypatch.setattr("app.translators._clock", lambda: 1000.0)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status) if len(calls) < 3 else _echo(request)

    tr = _free(httpx.MockTransport(handler), "de")
    paused: list[float] = []
    monkeypatch.setattr(tr, "_pause", lambda delay: paused.append(delay))
    assert run(tr.translate_batch(["x"])) == ["de:x"]
    assert len(calls) == 3 and len(paused) == 2
    run(tr.aclose())


def test_create_translator_from_service() -> None:
    tr = create_translator({"id": "deepl-free", "provider": "deepl-free"}, "de", "en")
    assert isinstance(tr, DeepLFreeTranslator)
    assert (tr.concurrency, tr.max_batch_items, tr.max_batch_chars) == (2, 50, MAX_REQUEST_UNITS)
    assert tr.cache_key == "deepl-free:deepl-free:en:de"
    run(tr.aclose())


def test_built_in_after_google(tmp_path: Path) -> None:
    """A configuration saved before the free DeepL service existed gets it after Google; it cannot be added again,
    edited beyond enabling, or deleted."""
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps({"default": "google", "services": [{"id": "google", "provider": "google", "builtin": True}]}),
        encoding="utf-8",
    )
    store = ServiceStore(path)
    assert [s["id"] for s in store.services[:2]] == ["google", "deepl-free"]
    assert [s["id"] for s in ServiceStore(tmp_path / "new.json").services[:2]] == ["google", "deepl-free"]
    with pytest.raises(ServiceError):
        store.create({"provider": "deepl-free"})
    with pytest.raises(ServiceError, match="内置服务不能删除"):
        store.delete("deepl-free")
    assert store.update("deepl-free", {"name": "x", "enabled": False})["name"] == "DeepL（免费）"


def test_api_lists_its_languages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state, "store", ServiceStore(tmp_path / "s.json"))
    client = TestClient(main.app)
    provider = client.get("/api/providers").json()["deepl-free"]
    assert not provider["configurable"] and not provider["needs_key"] and not provider["llm"]
    langs = {lang["code"]: lang for lang in client.get("/api/languages").json()}
    assert langs["auto"]["deepl-free"] and langs["zh-TW"]["deepl-free"]
    assert langs["th"]["deepl"] and not langs["th"]["deepl-free"]
    assert any(s["id"] == "deepl-free" for s in client.get("/api/services").json()["services"])


def test_chinese_and_japanese_are_cut_after_a_full_stop() -> None:
    """Full-width marks end a sentence without a space after them, and nothing is lost between the pieces."""
    text = "这是一个用来测试切分的中文句子，它会重复很多次。" * 80
    pieces = split_paragraph(text, html_text=False)
    assert [len(piece) for piece, _ in pieces] == [1488, 432]
    assert pieces[0] == (text[:1488], "") and pieces[0][0].endswith("次。")
    japanese = "今日はとても良い天気ですね！" * 120
    assert all(piece.endswith("！") for piece, _ in split_paragraph(japanese, html_text=False))


def test_a_closing_quote_stays_with_its_sentence() -> None:
    text = "他说：“明天见。”" * 300
    first, _ = split_paragraph(text, html_text=False)[0]
    assert first.endswith("。”")
    english = 'He said "See you tomorrow." ' * 80
    first, gap = split_paragraph(english.strip(), html_text=False)[0]
    assert first.endswith('tomorrow."') and gap == " "


def test_a_full_stop_inside_a_number_or_address_ends_no_sentence() -> None:
    """Without a sentence end, the cut falls before a space rather than inside 3.14 or example.com."""
    text = "pi is 3.14 and the site is example.com " * 60
    pieces = split_paragraph(text.strip(), html_text=False)
    assert len(pieces) == 2
    first, gap = pieces[0]
    assert gap == " "
    # Every word of both pieces is whole: a cut inside 3.14 or example.com would leave "3." or "example."
    for piece, _ in pieces:
        assert set(piece.split()) <= {"pi", "is", "3.14", "and", "the", "site", "example.com"}


def test_line_breaks_and_void_tags_do_not_hold_rich_text_open() -> None:
    """<br> and self-closing tags open nothing, so the text after them can still be cut."""
    for br in ("<br>", "<br/>"):
        text = f"The first line of a long poem ends here.{br}" * 60
        pieces = split_paragraph(text, html_text=True)
        assert len(pieces) == 2
        assert all(utf16_len(piece) <= MAX_REQUEST_UNITS for piece, _ in pieces)
        assert "".join(piece + gap for piece, gap in pieces) == text
