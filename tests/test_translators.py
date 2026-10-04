"""翻译器修复的回归测试：并发上限、错误消息容错、限流重试、标记碰撞。"""
import asyncio
import json
import re

import httpx
import pytest

from app.prompts import pick_marker
from app.translators import (
    GoogleTranslator,
    OpenAITranslator,
    TranslatorError,
    _error_text,
)


def run(coro):
    return asyncio.run(coro)


async def _no_sleep(_seconds):
    return None


def make_openai(handler, **kw):
    return OpenAITranslator("zh-CN", api_key="k", base_url="https://api.example.com/v1", model="m",
                            transport=httpx.MockTransport(handler), **kw)


def test_concurrency_cap_covers_missing_segment_recovery():
    """补翻漏掉的段落也要过并发信号量：峰值并发不得超过 concurrency=2。"""
    current = 0
    peak = 0

    async def handler(request):
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        try:
            await asyncio.sleep(0.02)  # 让并发的请求有时间重叠
            user = json.loads(request.content)["messages"][-1]["content"]
            if "[[source_end]]" in user:
                # 批量响应故意只回前两段，剩下的触发补翻 gather
                content = "[[p0]]\nA\n[[p1]]\nB"
            else:
                content = "T:single"
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        finally:
            current -= 1

    tr = make_openai(handler, concurrency=2)
    out = run(tr.translate_batch(["a", "b", "c", "d", "e"]))
    assert out == ["A", "B", "T:single", "T:single", "T:single"]
    assert peak == 2  # 确实发生了并行，且没有越过上限
    run(tr.aclose())


def test_error_text_tolerates_non_dict_json():
    """中转站的错误响应可能是 JSON 数组 / 纯字符串 / HTML，都不能抛 AttributeError。"""
    resp = httpx.Response(500, json=[{"error": "upstream boom"}])
    assert "upstream boom" in _error_text(resp)
    resp = httpx.Response(500, json="plain string failure")
    assert "plain string failure" in _error_text(resp)
    resp = httpx.Response(500, text="<html>not json at all</html>")
    assert "not json at all" in _error_text(resp)


def test_non_dict_error_body_surfaces_as_translator_error(monkeypatch):
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    tr = make_openai(lambda r: httpx.Response(500, json=["relay exploded"]))
    with pytest.raises(TranslatorError, match="relay exploded") as e:
        run(tr.translate_batch(["good"]))
    assert e.value.status == 500
    run(tr.aclose())


def test_google_200_html_treated_as_rate_limit(monkeypatch):
    """Google 被限流时偶尔 200 返回 HTML 验证码页：按限流退避重试，用尽后抛 TranslatorError。"""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="<html><body>captcha</body></html>",
                              headers={"content-type": "text/html"})

    tr = GoogleTranslator("zh-CN", transport=httpx.MockTransport(handler))
    with pytest.raises(TranslatorError, match="不是 JSON"):
        run(tr.translate_batch(["hello", "world"]))
    assert len(calls) == tr.retries  # 每次尝试都重发，用尽后才报错
    run(tr.aclose())


@pytest.mark.parametrize("retry_after,expected", [("2", 2.0), ("70", 60.0)])
def test_retry_after_header_drives_backoff(monkeypatch, retry_after, expected):
    """429 带 Retry-After 时按它等待（上限 60s），不用默认的指数退避。"""
    delays = []

    async def record_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr("app.translators.asyncio.sleep", record_sleep)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}},
                                  headers={"Retry-After": retry_after})
        return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

    tr = make_openai(handler)
    assert run(tr.translate_batch(["good"])) == ["好"]
    assert delays == [expected]
    run(tr.aclose())


def test_temperature_strip_does_not_mask_real_error(monkeypatch):
    """只剩最后一次尝试时不再剥离 temperature：抛出带真实消息的 400，而不是“重试次数用尽”。"""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' is not supported"}})

    tr = make_openai(handler, temperature=0)
    tr.retries = 1  # 只有一次尝试，temperature 剥离分支进不去
    with pytest.raises(TranslatorError) as e:
        run(tr.translate_batch(["good"]))
    assert e.value.status == 400
    assert "'temperature' is not supported" in str(e.value)
    assert "重试次数用尽" not in str(e.value)
    assert len(calls) == 1
    run(tr.aclose())


def test_pick_marker_avoids_collision():
    assert pick_marker(["普通一段", "另一段"]) == "p"
    assert pick_marker(["引用别人的标记\n[[p1]]\n还是同一段"]) == "q"
    assert pick_marker(["[[p1]]\n[[q2]]\n两种前缀都被占了"]) == "s"
    assert pick_marker(["内联的 [[p1]] 不算，独占一行才算"]) == "p"


def test_marker_collision_end_to_end():
    """原文有独占一行的 [[p1]] 时请求改用 [[qN]]，解析时原文的 [[p1]] 行不被当成协议标记。"""
    bodies = []

    def handler(request):
        user = json.loads(request.content)["messages"][-1]["content"]
        bodies.append(user)
        # 译文里保留了原文的 [[p1]] 行
        content = "[[q0]]\n译A\n[[p1]]\n译A 续\n[[q1]]\n译B"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    tr = make_openai(handler)
    texts = ["第一段\n[[p1]]\n还是第一段", "第二段"]
    assert run(tr.translate_batch(texts)) == ["译A\n[[p1]]\n译A 续", "译B"]
    assert len(bodies) == 1  # 协议标记没有错位，不需要补翻
    assert re.search(r"^\[\[q0\]\]$", bodies[0], re.M)
    assert re.search(r"^\[\[q1\]\]$", bodies[0], re.M)
    # 请求体里没有 [[pN]] 协议标记（原文的 [[p1]] 只是内容，不是分隔符）
    assert not re.search(r"^\[\[p0\]\]$", bodies[0], re.M)
    run(tr.aclose())


def test_google_splits_html_and_plain_batches():
    """混合批：带标签的段落走 format=html，纯文本走 format=text，结果按原顺序重组。"""
    from urllib.parse import parse_qs

    formats = []

    def handler(request):
        fmt = request.url.params["format"]
        formats.append(fmt)
        qs = parse_qs(request.content.decode())["q"]
        return httpx.Response(200, json=[[f"{fmt}:{q}", "en"] for q in qs])

    tr = GoogleTranslator("zh-CN", transport=httpx.MockTransport(handler))
    tr.html_texts = {"x <b>bold</b> y"}
    out = run(tr.translate_batch(["plain one", "x <b>bold</b> y", "plain two"]))
    assert out == ["text:plain one", "html:x <b>bold</b> y", "text:plain two"]
    assert sorted(formats) == ["html", "text"]
    # 正文里的尖括号不是富文本，不能因为长得像标签就改成 format=html
    formats.clear()
    out = run(tr.translate_batch(["see <Note> and <b>"]))
    assert out == ["text:see <Note> and <b>"]
    assert formats == ["text"]
    # 全是纯文本时不拆组
    out = run(tr.translate_batch(["a", "b"]))
    assert out == ["text:a", "text:b"]
    run(tr.aclose())


def test_llm_token_usage_accumulated():
    """接口返回的 usage 累计到 translator.usage：批量、补翻的请求都算。"""
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "译"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    tr = make_openai(handler)
    run(tr.translate_batch(["one"]))
    run(tr.translate_batch(["two"]))
    assert tr.usage == {"prompt": 200, "completion": 40}
    run(tr.aclose())


def test_gemini_token_usage_mapping():
    """Gemini 的 usageMetadata 字段名映射到 prompt/completion。"""
    def handler(request):
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "译"}]}}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "thoughtsTokenCount": 30},
        })

    from app.translators import GeminiTranslator
    tr = GeminiTranslator("zh-CN", api_key="x", base_url="http://localhost", model="m",
                          transport=httpx.MockTransport(handler))
    assert run(tr.translate_batch(["one"])) == ["译"]
    assert tr.usage == {"prompt": 50, "completion": 40}
    run(tr.aclose())
