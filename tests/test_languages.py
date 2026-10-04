import asyncio
import json

import httpx
import pytest

from app import langdetect
from app.languages import LANGUAGES
from app.prompts import build_messages, default_prompts
from app.runner import Runner
from app.translators import GoogleTranslator, MockTranslator, TranslatorError, create_translator


def run(coro):
    return asyncio.run(coro)


class MemCache:
    def get_many(self, prefix, texts):
        return {}

    def put_many(self, prefix, pairs):
        pass


def test_language_table_from_plugin():
    codes = [c for c, _, _ in LANGUAGES]
    assert codes[:5] == ["auto", "zh-CN", "zh-TW", "zh-HK", "en"]
    assert len(codes) == 128
    assert dict((c, n) for c, n, _ in LANGUAGES)["wyw"] == "Classical Chinese"


@pytest.mark.parametrize("src,dst,expected", [
    ("auto", "zh-CN", "你是专业的简体中文母语译者"),
    ("en", "zh-CN", "你是专业的简体中文母语译者"),       # auto2zh-CN 对任何源语言生效
    ("wyw", "zh-CN", "你是精通古文的学者"),             # wyw2zh-CN 覆盖 auto2zh-CN
    ("auto", "wyw", "你是专业的文言文译者"),
    ("auto", "zh-CN-NE", "你是专业的东北话翻译"),        # extends auto2zh-CN 再覆盖
    ("auto", "ja", "あなたはプロの日本語ネイティブ翻訳者です"),
    ("auto", "de", "You are a professional {{to}} native translator"),  # 无专门版本，用通用模板
])
def test_lang_overrides_match_plugin_rules(src, dst, expected):
    assert default_prompts(dst, src)["system"].startswith(expected)


def test_zh_prompts_ask_for_chinese_numerals():
    assert "中文数字" in default_prompts("zh-CN")["system"]
    assert "中文數字" in default_prompts("zh-TW")["system"]
    assert "中文数字" in default_prompts("zh-CN-NE")["system"]
    assert "中文数字" in default_prompts("zh-CN", "wyw")["system"]
    system, _ = build_messages(["Book 2"], "zh-HK")
    assert "第二卷" in system


def test_from_and_to_placeholders():
    system, user = build_messages(["a", "b"], "it", source_lang="ja")
    assert system.startswith("You are a professional Italian native translator")
    assert user.startswith("Translate to Italian:")
    _, user = build_messages(["x"], "it", source_lang="ja", user_template="From {{from}} to {{to}}: {{text}}")
    assert user == "From Japanese to Italian: x"
    _, user = build_messages(["x"], "it", user_template="{{from}}|{{text}}")
    assert user == "Auto Detect|x"


def test_detect():
    assert langdetect.detect("Alice was beginning to get very tired of sitting by her sister on the bank") == "en"
    assert langdetect.detect("Die Katze sitzt auf der Matte und schläft den ganzen Tag lang ruhig") == "de"
    assert langdetect.detect("爱丽丝开始觉得很累了，她和姐姐坐在河边") == "zh-CN"
    assert langdetect.detect("愛麗絲開始覺得很累了，她和姐姐坐在河邊") == "zh-TW"
    assert langdetect.detect("这一段本来就是中文，不需要再翻译成中文了。") == "zh-CN"
    assert langdetect.detect("吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。") == "ja"
    assert langdetect.detect("안녕하세요 오늘은 날씨가 정말 좋네요 산책하러 갈까요") == "ko"
    assert langdetect.detect("OK") is None  # 太短不判断


def test_same_language_rules():
    assert langdetect.same_language("en", "en")
    assert langdetect.same_language("pt", "pt-br")
    assert not langdetect.same_language("zh-CN", "zh-TW")   # 简繁互译不算相同
    assert not langdetect.same_language("zh-CN", "wyw")
    assert langdetect.same_language("zh-TW", "zh-HK")
    assert not langdetect.same_language(None, "en")


def test_runner_skips_paragraphs_already_in_target_and_detects_source():
    tr = MockTranslator("zh-CN")
    runner = Runner(tr, cache=MemCache())
    texts = [
        "Alice was beginning to get very tired of sitting by her sister on the bank.",
        "So she was considering in her own mind, as well as she could, for the hot day.",
        "这一段本来就是中文，不需要再翻译成中文了。",
    ]
    out = run(runner.translate_all(texts))
    assert out[0].startswith("[zh-CN] ")
    assert out[2] == texts[2]
    assert runner.info == {"detected_lang": "en", "skipped": 1}
    assert tr.effective_source == "en"

    # 关闭跳过后全部翻译
    runner = Runner(MockTranslator("zh-CN"), cache=MemCache(), skip_same_lang=False)
    assert run(runner.translate_all(texts))[2].startswith("[zh-CN] ")


def test_llm_uses_detected_source_in_prompt():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    tr = create_translator({"provider": "openai", "api_key": "k", "model": "m", "user_prompt": "{{from}}->{{to}}: {{text}}"},
                           "fr", transport=httpx.MockTransport(handler))
    runner = Runner(tr, cache=MemCache())
    run(runner.translate_all(["Alice was beginning to get very tired of sitting by her sister on the bank."]))
    assert seen[0]["messages"][1]["content"].startswith("English->French: Alice")


def test_google_language_codes():
    log = []

    def handler(request):
        log.append(request)
        return httpx.Response(200, json=["ok"])

    tr = GoogleTranslator("fil", "pt-br", transport=httpx.MockTransport(handler))
    run(tr.translate_batch(["hello"]))
    assert log[0].url.params["tl"] == "tl" and log[0].url.params["sl"] == "pt"
    with pytest.raises(TranslatorError, match="谷歌翻译不支持"):
        GoogleTranslator("wyw")
    with pytest.raises(TranslatorError, match="不支持的目标语言"):
        MockTranslator("xx-YY")


def test_build_messages_html_suffix():
    system, _ = build_messages(["x <b>y</b>"], "zh-CN", contains_html=True)
    assert "HTML fragments" in system and "href" in system and "word order" in system
    assert "reorder" not in system
    system, _ = build_messages(["plain"], "zh-CN")
    assert "HTML fragments" not in system
