"""Regression tests for translator fixes: concurrency cap, error message tolerance, rate limit retries, marker
collisions."""

import asyncio
import json
import re
from collections.abc import Callable, Coroutine
from typing import TypedDict, Unpack

import httpx
import pytest

from app.json_types import JSONDict
from app.prompts import drop_added_braces, pick_marker
from app.translators import (
    GoogleTranslator,
    LLMTranslator,
    OpenAITranslator,
    TranslatorError,
    _error_text,
)
from tests.helpers import run

# What httpx.MockTransport accepts: a function that answers a request, plain or async
Handler = Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


class _LLMOptions(TypedDict, total=False):
    """The keyword options of LLMTranslator that the tests set."""

    concurrency: int
    temperature: float | None
    api_format: str


async def _no_sleep(_seconds: float) -> None:
    return None


def make_openai(handler: Handler, **kw: Unpack[_LLMOptions]) -> OpenAITranslator:
    return OpenAITranslator(
        "zh-CN",
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        **kw,
    )


def test_concurrency_cap_covers_missing_segment_recovery() -> None:
    """Requests that recover missing paragraphs also go through the concurrency semaphore: peak concurrency must not
    exceed concurrency=2."""
    current = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        try:
            await asyncio.sleep(0.02)  # Give concurrent requests time to overlap
            user = json.loads(request.content)["messages"][-1]["content"]
            if "[[source_end]]" in user:
                # The batch response deliberately returns only the first two paragraphs; the rest trigger the recovery
                # gather
                content = "[[p0]]\nA\n[[p1]]\nB"
            else:
                content = "T:single"
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        finally:
            current -= 1

    tr = make_openai(handler, concurrency=2)
    out = run(tr.translate_batch(["a", "b", "c", "d", "e"]))
    assert out == ["A", "B", "T:single", "T:single", "T:single"]
    assert peak == 2  # Requests did run in parallel and did not exceed the cap
    run(tr.aclose())


def test_shifted_batch_items_are_retranslated_one_by_one() -> None:
    """When the model merges two paragraphs into one and everything after shifts by one, the marker count still
    matches. If some paragraph has mismatched numbers, the batch is retranslated one paragraph at a time."""
    singles: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        if "[[source_end]]" in user:
            content = (
                "[[p0]]\n特伦布莱，1975 年，麦格劳-希尔。\n[[p1]]\n泽纳基斯，1994 年 4 月。\n"
                "[[p2]]\n严和拉森，第 345-357 页，1995 年。"
            )
        else:
            singles.append(user)
            content = "单独重翻，" + ("1994 年" if "1994" in user else "1975 年")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = make_openai(handler)
    out = run(
        tr.translate_batch(
            [
                "Tremblay and Manohar. Discrete Mathematical Structures with Applications to",
                "Computer Science. McGraw Hill Book Company, New York, 1975.",
                "John Xenakis, editor. Multidimensional Databases. April 1994.",
            ]
        )
    )
    # Paragraphs 2 and 3 have mismatched numbers. The translation of paragraph 1 is paragraphs 1 and 2 merged, which
    # cannot be detected, so it is retranslated with them
    assert out == ["单独重翻，1975 年", "单独重翻，1975 年", "单独重翻，1994 年"]
    assert len(singles) == 3
    run(tr.aclose())


def test_items_holding_a_neighbors_translation_are_retranslated() -> None:
    """The model dropped a paragraph and the following ones shifted: a translation carries formula placeholders the
    source does not have, or its length ratio is far from the usual value of the batch. The batch is retranslated one
    paragraph at a time: the translation of paragraph 1 happens to match but cannot be trusted, and not every shifted
    paragraph before paragraph 4 can be detected"""
    singles: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        if "[[source_end]]" in user:
            content = (
                "[[p0]]\n{v1} 将简单循环神经网络表示为前馈网络。前一时间步的隐藏层{v2}与权重矩阵 U 相乘。\n"
                "[[p1]]\n{v1} 将简单循环神经网络表示为前馈网络。前一时间步的隐藏层{v2}与权重矩阵 U 相乘。\n"
                "[[p2]]\nRNN 中的前向推理几乎与前馈网络完全相同, 为了根据输入 {v2} 计算输出 {v1}。\n"
                "[[p3]]\n{v1}{v2} 前向推理。\n"
                "[[p4]]\n我们分别将输入层, 隐藏层和输出层的维度记为 {v1}, {v2} 和 {v3}, 由此得到三个参数矩阵。"
            )
        else:
            singles.append(user)
            content = "单独重翻"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = make_openai(handler)
    out = run(
        tr.translate_batch(
            [
                "Adding this temporal dimension makes RNNs appear to be more complex than non-recurrent "
                "architectures. But in reality, they are not all that different.",
                "{v1} Simple recurrent neural network illustrated as a feedforward network. The hidden "
                "layer{v2} from the prior time step is multiplied by weight matrix U.",
                "Inference in RNNs",
                "Forward inference (mapping a sequence of inputs to a sequence of outputs) in an RNN is nearly "
                "identical to what we have already seen with feedforward networks, to compute an output {v1} "
                "for an input {v2}.",
                "Let us refer to the input, hidden and output layer dimensions as {v1}, {v2}, and {v3} "
                "respectively, which gives three parameter matrices.",
            ]
        )
    )
    # Paragraphs 0 and 2 carry placeholders of other paragraphs, and paragraph 3 is translated as a single word
    assert all(text == "单独重翻" for text in out)
    assert len(singles) == 5
    run(tr.aclose())


def test_a_short_heading_holding_the_next_paragraph_shifts_the_batch() -> None:
    """A three-word heading got the translation of the next paragraph, and that paragraph got the one after it. Short
    sentences are not checked by length ratio, only by a translation that is absurdly long"""
    singles: list[str] = []
    following = (
        "下一步是提取加窗信号的频谱信息；我们需要知道信号在不同频带上包含多少能量。"
        "对于离散时间（采样）信号，要提取离散频带的频谱信息，所使用的工具是离散傅里叶变换，即 DFT。"
    )
    after = "DFT 的输入是加窗信号，对于每一个离散频带，输出是一个复数，表示该频率分量在原始信号中的幅度和相位。"

    async def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        if "[[source_end]]" in user:
            content = (
                f"[[p0]]\n{following}\n[[p1]]\n{after}\n[[p2]]\n{after}\n"
                "[[p3]]\n这里不介绍 DFT 的数学细节，只需指出傅里叶分析依赖于欧拉公式，其中 j 是虚数单位。"
            )
        else:
            singles.append(user)
            content = "单独重翻"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = make_openai(handler)
    out = run(
        tr.translate_batch(
            [
                "Discrete Fourier Transform",
                "The next step is to extract spectral information for our windowed signal; we need to know how much "
                "energy the signal contains at different frequency bands. The tool for extracting spectral information "
                "for discrete frequency bands for a discrete-time (sampled) signal is the discrete Fourier transform "
                "or DFT.",
                "The input to the DFT is a windowed signal, and the output, for each of N discrete frequency bands, "
                "is a complex number representing the magnitude and phase of that frequency component in the "
                "original signal.",
                "We do not introduce the mathematical details of the DFT here, except to note that Fourier analysis "
                "relies on Euler's formula, with j as the imaginary unit.",
            ]
        )
    )
    assert all(text == "单独重翻" for text in out)
    assert len(singles) == 4
    run(tr.aclose())


def test_error_text_tolerates_non_dict_json() -> None:
    """An error response from a relay service may be a JSON array, a plain string or HTML; none of them may raise
    AttributeError."""
    resp = httpx.Response(500, json=[{"error": "upstream boom"}])
    assert "upstream boom" in _error_text(resp)
    resp = httpx.Response(500, json="plain string failure")
    assert "plain string failure" in _error_text(resp)
    resp = httpx.Response(500, text="<html>not json at all</html>")
    assert "not json at all" in _error_text(resp)


def test_non_dict_error_body_surfaces_as_translator_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    tr = make_openai(lambda r: httpx.Response(500, json=["relay exploded"]))
    with pytest.raises(TranslatorError, match="relay exploded") as e:
        run(tr.translate_batch(["good"]))
    assert e.value.status == 500
    run(tr.aclose())


def test_google_200_html_treated_as_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rate-limited Google sometimes returns an HTML captcha page with status 200: retry with rate limit backoff and
    raise TranslatorError once the retries are used up."""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="<html><body>captcha</body></html>", headers={"content-type": "text/html"})

    tr = GoogleTranslator("zh-CN", transport=httpx.MockTransport(handler))
    with pytest.raises(TranslatorError, match="不是 JSON"):
        run(tr.translate_batch(["hello", "world"]))
    assert len(calls) == tr.retries  # Every attempt resends; the error is raised only after all are used up
    run(tr.aclose())


@pytest.mark.parametrize("retry_after,expected", [("2", 2.0), ("70", 60.0)])
def test_retry_after_header_drives_backoff(monkeypatch: pytest.MonkeyPatch, retry_after: str, expected: float) -> None:
    """A 429 with Retry-After waits for that long (at most 60 s) instead of the default exponential backoff."""
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("app.translators.asyncio.sleep", record_sleep)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429, json={"error": {"message": "rate limited"}}, headers={"Retry-After": retry_after}
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

    tr = make_openai(handler)
    assert run(tr.translate_batch(["good"])) == ["好"]
    assert delays == [expected]
    run(tr.aclose())


def test_temperature_strip_does_not_mask_real_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """On the last remaining attempt temperature is no longer stripped: the 400 with the real message is raised
    instead of "已达到最大重试次数" (maximum retries reached)."""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' is not supported"}})

    tr = make_openai(handler, temperature=0)
    tr.retries = 1  # Only one attempt, so the temperature stripping branch is never reached
    with pytest.raises(TranslatorError) as e:
        run(tr.translate_batch(["good"]))
    assert e.value.status == 400
    assert "'temperature' is not supported" in str(e.value)
    assert "已达到最大重试次数" not in str(e.value)
    assert len(calls) == 1
    run(tr.aclose())


def test_pick_marker_avoids_collision() -> None:
    assert pick_marker(["普通一段", "另一段"]) == "p"
    assert pick_marker(["引用别人的标记\n[[p1]]\n还是同一段"]) == "q"
    assert pick_marker(["[[p1]]\n[[q2]]\n两种前缀都被占了"]) == "s"
    assert pick_marker(["内联的 [[p1]] 不算，独占一行才算"]) == "p"


def test_marker_collision_end_to_end() -> None:
    """When the source has [[p1]] on a line of its own, the request uses [[qN]] instead, and parsing does not treat
    the [[p1]] line of the source as a protocol marker."""
    bodies: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        bodies.append(user)
        # The translation keeps the [[p1]] line of the source
        content = "[[q0]]\n译A\n[[p1]]\n译A 续\n[[q1]]\n译B"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = make_openai(handler)
    texts = ["第一段\n[[p1]]\n还是第一段", "第二段"]
    assert run(tr.translate_batch(texts)) == ["译A\n[[p1]]\n译A 续", "译B"]
    assert len(bodies) == 1  # Protocol markers did not shift, so no recovery is needed
    assert re.search(r"^\[\[q0\]\]$", bodies[0], re.M)
    assert re.search(r"^\[\[q1\]\]$", bodies[0], re.M)
    # The request body has no [[pN]] protocol markers (the [[p1]] of the source is content, not a separator)
    assert not re.search(r"^\[\[p0\]\]$", bodies[0], re.M)
    run(tr.aclose())


def test_google_splits_html_and_plain_batches() -> None:
    """Mixed batch: paragraphs with tags go through format=html, plain text through format=text, and the results are
    reassembled in the original order."""
    from urllib.parse import parse_qs

    formats: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fmt = request.url.params["format"]
        formats.append(fmt)
        qs = parse_qs(request.content.decode())["q"]
        return httpx.Response(200, json=[[f"{fmt}:{q}", "en"] for q in qs])

    tr = GoogleTranslator("zh-CN", transport=httpx.MockTransport(handler))
    tr.html_texts = {"x <b>bold</b> y"}
    out = run(tr.translate_batch(["plain one", "x <b>bold</b> y", "plain two"]))
    assert out == ["text:plain one", "html:x <b>bold</b> y", "text:plain two"]
    assert sorted(formats) == ["html", "text"]
    # Angle brackets in body text are not rich text and must not switch to format=html because they look like tags
    formats.clear()
    out = run(tr.translate_batch(["see <Note> and <b>"]))
    assert out == ["text:see <Note> and <b>"]
    assert formats == ["text"]
    # No split into groups when everything is plain text
    out = run(tr.translate_batch(["a", "b"]))
    assert out == ["text:a", "text:b"]
    run(tr.aclose())


def test_llm_token_usage_accumulated() -> None:
    """The usage returned by the API accumulates into translator.usage, counting both batch and recovery requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "译"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    tr = make_openai(handler)
    run(tr.translate_batch(["one"]))
    run(tr.translate_batch(["two"]))
    assert tr.usage == {"prompt": 200, "completion": 40}
    run(tr.aclose())


def test_gemini_token_usage_mapping() -> None:
    """The field names of Gemini usageMetadata map to prompt/completion."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "译"}]}}],
                "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "thoughtsTokenCount": 30},
            },
        )

    from app.translators import GeminiTranslator

    tr = GeminiTranslator(
        "zh-CN", api_key="x", base_url="http://localhost", model="m", transport=httpx.MockTransport(handler)
    )
    assert run(tr.translate_batch(["one"])) == ["译"]
    assert tr.usage == {"prompt": 50, "completion": 40}
    run(tr.aclose())


def test_reply_without_content_is_asked_again() -> None:
    """A relay service sometimes replies with a message without content (finish_reason is still stop). The same
    request is sent again so the whole book does not fail."""
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200, json={"choices": [{"index": 0, "message": {"role": "assistant"}, "finish_reason": "stop"}]}
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "译文"}}]})

    tr = make_openai(handler)
    assert run(tr.translate_batch(["source"])) == ["译文"]
    assert calls == 2
    run(tr.aclose())


def test_reply_that_stays_empty_is_an_empty_translation() -> None:
    """A reply that stays empty returns an empty translation: the Runner fills in the source text, and the whole
    document fails only when there are many empty paragraphs."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}]})

    tr = make_openai(handler)
    assert run(tr.translate_batch(["source"])) == [""]
    run(tr.aclose())


def test_braces_the_model_added_around_marks_are_dropped() -> None:
    """When the source has {n,m}, the model wraps the marks in braces too. Only braces absent at the same position in
    the source are removed."""
    src = "include {b}disjunction{/b} of symbols, {b}counters{/b} (*, +, and {n,m})"
    assert drop_added_braces(src, "包括{{b}析取{/b}}, {{b}计数器{/b}}(*, + 和 {n,m})") == (
        "包括{b}析取{/b}, {b}计数器{/b}(*, + 和 {n,m})"
    )
    assert drop_added_braces("a {b}x{/b} b", "叫做{b}断裂{/b}}和{{/b}") == "叫做{b}断裂{/b}和{/b}"
    assert drop_added_braces("as {c0000ff}Chen{/c} notes", "正如{{c0000ff}Chen{/c}}所说") == "正如{c0000ff}Chen{/c}所说"
    # Set braces already in the source stay
    src = "samples a group {{i}o{/i}{v1}, {i}o{/i}{v2}}"
    assert drop_added_braces(src, "采样一组 {{i}o{/i}{v1}, {i}o{/i}{v2}}") == "采样一组 {{i}o{/i}{v1}, {i}o{/i}{v2}}"
    # Braces not adjacent to a mark are left alone
    assert drop_added_braces("{b}x{/b}", "{b}x{/b} 和 {n,m}") == "{b}x{/b} 和 {n,m}"


def test_marks_written_with_spaces_inside_are_restored() -> None:
    """The model sometimes writes { i}m{/i} or { |}, which are not recognised as marks. Marks present in the source
    are restored to the standard form; others are left alone."""
    from app.prompts import tidy_marks

    src = "a simplified model {i}m{/i} between models{v1} and{v2}, the {|} lower"
    dst = "简化模型{ i}m{/i}, 在两个模型{v1 }和{ v2}之间较{ |}低 {/ i}"
    assert tidy_marks(src, dst) == "简化模型{i}m{/i}, 在两个模型{v1}和{v2}之间较{|}低 {/i}"
    assert tidy_marks("a set {n,m} of {b}x{/b}", "集合 { b} 和 { n,m}") == "集合 {b} 和 { n,m}"
    assert tidy_marks("plain text", "字面的 { i} 不动") == "字面的 { i} 不动"
    # Zero-width characters, question marks, square brackets, a misspelt closing mark, a closing mark with a color code
    src = "if {i}surprised{/i} or {i}just{/i} is {b}hot{/b} for {cb30000}POSITION{/c}"
    dst = "如果{\u200b\u200bi}surprised{/i}或{ i? }just{/i}中的[i]x{/i}是{b}热点{/b>, {cb30000}职务{/cb30000}"
    assert tidy_marks(src, dst) == "如果{i}surprised{/i}或{i}just{/i}中的{i}x{/i}是{b}热点{/b}, {cb30000}职务{/c}"
    assert tidy_marks("x[i] for {i}i{/i}", "x[i] 对 {i}i{/i}") == "x[i] 对 {i}i{/i}"


def test_stray_chars_marks_text_neither_language_uses() -> None:
    """A translation containing a script used by neither the source nor the target language is bad; text already in
    the source, text the target language uses and Greek letters in formulas do not count."""
    from app.translators import stray_chars

    assert stray_chars("Please take a moment to convince yourself", "请花一点时间 убед自己", "zh-CN")
    assert stray_chars("with a small change", "通过做 მცირე修改", "zh-CN")
    assert not stray_chars("the Russian word Привет", "俄语词 Привет", "zh-CN")
    assert not stray_chars("mu-law, 339", "μ律, 339", "zh-CN")
    assert not stray_chars("a text", "日本語のテキスト", "ja")
    assert not stray_chars("a text", "Привет", "ru")
    assert stray_chars("a text", "テキスト", "zh-CN")


def test_items_with_stray_chars_are_retranslated() -> None:
    """A paragraph in a batch containing foreign script is retranslated on its own; if the single retranslation still
    contains it, the request is sent again."""
    singles: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        if "[[source_end]]" in user:
            content = "[[p0]]\n请花一点时间 убед自己。\n[[p1]]\n我们开发了一套方法。"
        else:
            singles.append(user)
            content = "请花一点时间 убед自己。" if len(singles) == 1 else "请花一点时间说服自己。"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = make_openai(handler)
    out = run(tr.translate_batch(["Please take a moment to convince yourself.", "We developed a method."]))
    assert out == ["请花一点时间说服自己。", "我们开发了一套方法。"]
    assert len(singles) == 2
    run(tr.aclose())


def test_stray_chars_marks_symbols_absent_from_source() -> None:
    """A translation with miscellaneous symbols, emoji or private use area characters absent from the source is bad;
    those already in the source do not count."""
    from app.translators import stray_chars

    src = "How AI Risks Differ from Traditional Software Risks"
    assert stray_chars(src, "AI 风险与传统软件风险的差异♀♀♀♀♀♀", "zh-CN")
    assert stray_chars("Results", "结果", "zh-CN")
    assert not stray_chars("a ✓ marks the best", "✓ 表示最好", "zh-CN")
    assert not stray_chars("Figure 2 | The pipeline", "图 2｜流程", "zh-CN")
    assert not stray_chars("90 degrees", "90°", "zh-CN")


def test_single_item_note_is_written_into_the_system_prompt() -> None:
    """With a single paragraph the note is written at the end of the system prompt, and the source text stays only in
    the user message."""
    from app.prompts import build_messages

    system, user = build_messages(["log"], "zh-CN", notes=["An entry in the subject index."])
    assert system.endswith("Note: An entry in the subject index.")
    assert user.endswith("log") and "Note" not in user
    system, _ = build_messages(["log"], "zh-CN", notes=[""])
    assert "## Item notes" not in system


def test_a_reply_asking_for_the_full_text_is_not_kept() -> None:
    """When an example sentence the book leaves unfinished on purpose is retranslated on its own, the model replies
    "请提供完整的原文" (please provide the full source text). The usual length ratio of the batch reveals it; if asking
    again gives the same reply, the result is empty and the Runner fills in the source text"""
    fragment = "IBM, a company with a long..."
    asks: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        if "[[source_end]]" in user:
            content = "[[p0]]\n{v1} 串位的译文。\n[[p1]]\n第二段。\n[[p2]]\n第三段。\n[[p3]]\n第四段。"
        elif fragment in user:
            asks.append(body["messages"][0]["content"])
            content = (
                '我需要您提供完整的源文本才能进行翻译。您提到"IBM, a company with a long...", '
                "但这句话似乎未完成。请提供完整的 source item(源文本内容), 我将按照您的要求将其翻译为简体中文。"
            )
        else:
            content = "单独重翻的译文，长度和原文相称。"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = make_openai(handler)
    long = "This sentence is long enough for the length ratio to be compared with the rest of the batch."
    out = run(tr.translate_batch([long, fragment, long + " Again.", long + " Once more."]))
    assert out[1] == "" and out[0] == out[2] == out[3] == "单独重翻的译文，长度和原文相称。"
    assert len(asks) == 2 and "never ask for more context" in asks[0]
    run(tr.aclose())


TRUNCATED_SOURCES = ["The first short sentence.", "The second short sentence."]
TRUNCATED_TRANSLATIONS = {"first": "第一句短句。", "second": "第二句短句。"}


def _truncating_reply(provider: str, text: str, truncated: bool) -> JSONDict:
    """A reply in the provider's format, cut off at the output limit when truncated."""
    if provider == "openai":
        return {"choices": [{"message": {"content": text}, "finish_reason": "length" if truncated else "stop"}]}
    if provider == "responses":
        if truncated:
            return {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output_text": text}
        return {"status": "completed", "output_text": text}
    if provider == "claude":
        return {"content": [{"type": "text", "text": text}], "stop_reason": "max_tokens" if truncated else "end_turn"}
    finish = "MAX_TOKENS" if truncated else "STOP"
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}]}


def _truncating_translator(provider: str, cut_single: bool = False) -> LLMTranslator:
    """A model that runs out of output for a batch of two paragraphs and translates one paragraph in full, or cuts
    that off too with cut_single."""
    from app.translators import ClaudeTranslator, GeminiTranslator

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        words = [word for word in TRUNCATED_TRANSLATIONS if word in body]
        if len(words) > 1:
            # The first paragraph is complete and the second stops halfway, markers and all
            first, second = (TRUNCATED_TRANSLATIONS[word] for word in words)
            return httpx.Response(200, json=_truncating_reply(provider, f"[[p0]]\n{first}\n[[p1]]\n{second[:2]}", True))
        text = TRUNCATED_TRANSLATIONS[words[0]]
        return httpx.Response(200, json=_truncating_reply(provider, text[:2] if cut_single else text, cut_single))

    classes: dict[str, type[LLMTranslator]] = {"claude": ClaudeTranslator, "gemini": GeminiTranslator}
    cls = classes.get(provider, OpenAITranslator)
    extra: _LLMOptions = {"api_format": "responses"} if provider == "responses" else {}
    return cls(
        "zh-CN",
        "en",
        api_key="k",
        base_url="https://api.example.com/v1",
        model="m",
        transport=httpx.MockTransport(handler),
        **extra,
    )


@pytest.mark.parametrize("provider", ["openai", "responses", "claude", "gemini"])
def test_a_batch_cut_off_at_the_output_limit_is_split(provider: str) -> None:
    """A reply stopped at the model's output limit holds only part of the batch; it used to be accepted (and cached)
    as the translation. The batch is now translated again in halves."""

    async def scenario() -> list[str]:
        tr = _truncating_translator(provider)
        try:
            return await tr.translate_batch(TRUNCATED_SOURCES)
        finally:
            await tr.aclose()

    assert run(scenario()) == list(TRUNCATED_TRANSLATIONS.values())


@pytest.mark.parametrize("provider", ["openai", "claude"])
def test_a_single_paragraph_cut_off_is_an_empty_translation(provider: str) -> None:
    """One paragraph that still does not fit gives an empty translation: Runner keeps the original, caches nothing
    and counts the paragraph as failed."""

    async def scenario() -> list[str]:
        tr = _truncating_translator(provider, cut_single=True)
        try:
            return await tr.translate_batch(TRUNCATED_SOURCES[:1])
        finally:
            await tr.aclose()

    assert run(scenario()) == [""]


def test_engines_of_one_service_share_a_gate_and_others_do_not() -> None:
    """Two engines from the same service share its concurrency; another service, or an engine without a service,
    has its own."""
    from app.translators import create_translator

    live = {"now": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0.02)
        live["now"] -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "译文"}}]})

    def service(sid: str) -> JSONDict:
        return {"id": sid, "provider": "custom", "base_url": "http://x/v1", "model": "m", "concurrency": 2}

    async def peak(services: list[str]) -> int:
        live["peak"] = 0
        engines = [
            create_translator(service(s), "zh-CN", "en", transport=httpx.MockTransport(handler)) for s in services
        ]
        try:
            await asyncio.gather(*(e.translate_batch([f"Text {i}"]) for e in engines for i in range(3)))
        finally:
            for e in engines:
                await e.aclose()
        return live["peak"]

    assert run(peak(["a", "a"])) == 2
    assert run(peak(["a", "b"])) == 4


def test_every_engine_takes_the_document_title() -> None:
    """Runner.set_title fills the title of whatever engine it has; only LLM engines put it into the prompt."""
    from app.runner import Runner
    from app.translators import MockTranslator
    from tests.helpers import MemCache

    translator = MockTranslator("zh-CN")
    Runner(translator, cache=MemCache()).set_title("  A Long Report Title  " + "x" * 300)
    assert translator.title.startswith("A Long Report Title") and len(translator.title) == 200
