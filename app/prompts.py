"""AI 翻译提示词，取自沉浸式翻译 1.33.3 的默认配置（default_config.json 的 translationServices.ai）。

用的是它的“AI 批量标记协议”（aiBatch marker protocol）：
- 输入：每段前面一行 [[p0]]、[[p1]]…，最后一行 [[source_end]]
- 输出：同样的 [[pN]] 标题行 + 译文，模型可以按任意顺序返回
- 系统提示词 = 任务提示词（taskSystemPrompt）+ 协议说明（protocolSystemPrompt）
- 只有一段时不用协议，直接让模型输出译文

按语言覆盖（langOverrides）和插件的匹配规则一致：每项 id 是 "<from>2<to>"，
from 是 auto 或当前源语言、to 是 auto 或当前目标语言时生效，按顺序叠加，extends 先叠加被继承的项。
所以 "auto2zh-CN" 对任何源语言生效，"wyw2zh-CN" 只在源语言是文言文时生效并覆盖前者。

占位符：{{to}} 目标语言，{{from}} 源语言，{{text}} 原文，{{title_prompt}} 文档标题上下文，
{{summary_prompt}} / {{terms_prompt}} / {{imt_style_guide}} 暂未使用（留空），方便以后加摘要和术语表。
"""
import re

from .languages import LANG_OVERRIDES, LANGUAGES

# {{to}} / {{from}} 用英文语言名，和插件一致
LANG_EN = {code: en for code, en, _ in LANGUAGES}

TASK_SYSTEM_PROMPT = """You are a professional {{to}} native translator. Translate each source item accurately and fluently into {{to}}.

## Translation Requirements
- Treat each source item as a complete translation unit, including a single word, short phrase, heading, label, or site name. Do not ask for more context, a full article, or additional text.
- Return only the translated content for each item, following the required response format. Do not add explanations, apologies, greetings, or commentary. If an item should remain untranslated, return it unchanged.
- Preserve its meaning, tone, paragraph structure, and meaningful formatting.
- Keep code, placeholders, and content that must not be translated unchanged.
- Use supplied document context and terminology consistently.

{{title_prompt}}{{summary_prompt}}{{terms_prompt}}

{{imt_style_guide}}"""

USER_PROMPT = "Translate to {{to}}:\n\n{{text}}"

# 多段时附加在系统提示词后面的协议说明（插件里术语提取关闭时的版本）
PROTOCOL_SYSTEM_PROMPT = """

## Multi-item processing protocol
The input contains source items. Each item starts with its exact request id on
one title line, followed by that item's source text. The source list ends with
exactly one [[source_end]] line. Process every item independently according to
the active task instructions, using the other items only as context. Return each
supplied id exactly once in any order, followed by only its processed text.
After all output blocks, processed text may span multiple lines. Preserve every id
title byte-for-byte.

Do not wrap the entire response or protocol envelope in a Markdown fence.
Fenced code blocks that belong to an item's processed content may be preserved.
Do not include explanations outside processed text."""

# 只有一段时附加的说明
SINGLE_ITEM_SUFFIX = """
Return only the processed text, without source text, markers, titles,
explanations, or an enclosing Markdown fence. Preserve fenced code blocks that
belong to the processed content."""

TITLE_PROMPT = "\n\n## Context Awareness\nDocument Metadata:\nTitle: “{{imt_title}}”"

SOURCE_END = "[[source_end]]"
MARKER_RE = re.compile(r"^\[\[(p\d+)\]\]$")
CONTROL_RE = re.compile(r"^\[\[(?:p\d+|source_end|terms|term|end)\]\]$")
# 插件的 removeResRegexs：去掉推理模型输出的 <think> 段
THINK_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*|^\s*</think>\s*")

_OVERRIDES = {o["id"]: o for o in LANG_OVERRIDES}


def lang_name(code: str) -> str:
    return LANG_EN.get(code, code)


def default_prompts(target_lang: str, source_lang: str = "auto") -> dict:
    """按插件的 langOverrides 规则算出某个语言对的默认提示词（未填充占位符）。"""
    result = {"system": TASK_SYSTEM_PROMPT, "user": USER_PROMPT}

    def apply(item):
        for key in ("system", "user"):
            if item.get(key):
                result[key] = item[key]

    for oid, item in _OVERRIDES.items():
        src, _, dst = oid.partition("2")
        if src in ("auto", source_lang) and dst in ("auto", target_lang):
            if item.get("extends") in _OVERRIDES:
                apply(_OVERRIDES[item["extends"]])
            apply(item)
    return result


def fill(template: str, values: dict) -> str:
    # 先展开 {{title_prompt}} 这类片段，片段里可能还有 {{imt_title}}
    for _ in range(2):
        template = re.sub(r"\{\{(\w+)\}\}", lambda m: values.get(m.group(1), m.group(0)), template)
    return template


def build_messages(texts: list[str], target_lang: str, *, source_lang: str = "auto", title: str = "",
                   system_template: str = "", user_template: str = "") -> tuple[str, str]:
    """返回 (system, user)。自定义模板为空时用默认模板。source_lang 是用户选的或检测出的源语言。"""
    defaults = default_prompts(target_lang, source_lang)
    system_t = system_template.strip() or defaults["system"]
    user_t = user_template.strip() or defaults["user"]
    if "{{text}}" not in user_t:
        user_t += "\n\n{{text}}"
    values = {
        "to": lang_name(target_lang),
        # 不知道源语言时，插件也是把 {{from}} 填成 "Auto Detect"
        "from": lang_name(source_lang or "auto"),
        "title_prompt": TITLE_PROMPT if title else "",
        "imt_title": title,
        "summary_prompt": "",
        "terms_prompt": "",
        "imt_style_guide": "",
    }
    if len(texts) == 1:
        system = fill(system_t, values).strip() + "\n" + SINGLE_ITEM_SUFFIX
        body = texts[0]
    else:
        system = fill(system_t, values).strip() + PROTOCOL_SYSTEM_PROMPT
        body = "\n".join(f"[[p{i}]]\n{t}" for i, t in enumerate(texts)) + "\n" + SOURCE_END
    # {{text}} 最后替换，避免原文里恰好有 {{xxx}} 被当成占位符
    user = fill(user_t.replace("{{text}}", "\0TEXT\0"), values).replace("\0TEXT\0", body)
    return system, user


def clean_output(content: str) -> str:
    content = THINK_RE.sub("", content).strip()
    # 去掉包住整个回复的 ``` 围栏
    m = re.fullmatch(r"```[\w-]*\n([\s\S]*?)\n```", content)
    return m.group(1).strip() if m else content


def parse_markers(content: str, count: int) -> dict[int, str]:
    """解析 [[pN]] 格式的回复，返回 {序号: 译文}；缺失的序号不在结果里。"""
    result: dict[int, str] = {}
    current: int | None = None
    buf: list[str] = []

    def flush():
        if current is not None and 0 <= current < count and current not in result:
            result[current] = "\n".join(buf).strip()

    for line in clean_output(content).split("\n"):
        stripped = line.strip()
        if CONTROL_RE.match(stripped):
            flush()
            m = MARKER_RE.match(stripped)
            current = int(m.group(1)[1:]) if m else None
            buf = []
            if stripped in ("[[terms]]", "[[end]]"):
                break
        elif current is not None:
            buf.append(line)
    flush()
    return {k: v for k, v in result.items() if v}
