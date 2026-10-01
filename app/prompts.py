"""AI 翻译提示词，取自沉浸式翻译 1.33.3 的默认配置（default_config.json 的 translationServices.ai）。

用的是它的“AI 批量标记协议”（aiBatch marker protocol）：
- 输入：每段前面一行 [[p0]]、[[p1]]…，最后一行 [[source_end]]
- 输出：同样的 [[pN]] 标题行 + 译文，模型可以按任意顺序返回
- 系统提示词 = 任务提示词（taskSystemPrompt，按目标语言有专门版本）+ 协议说明（protocolSystemPrompt）
- 只有一段时不用协议，直接让模型输出译文

占位符：{{to}} 目标语言，{{text}} 原文，{{title_prompt}} 文档标题上下文，
{{summary_prompt}} / {{terms_prompt}} / {{imt_style_guide}} 暂未使用（留空），方便以后加摘要和术语表。
"""
import re

# {{to}} 在通用提示词里用英文语言名，和插件一致
LANG_EN = {
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese (Taiwan)",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
}

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

# 插件按目标语言覆盖的提示词（langOverrides，id 为 auto2<lang>）
LANG_OVERRIDES = {
    "zh-CN": {
        "system": "你是专业的简体中文母语译者。请将每个 source item 分别准确、流畅地翻译为简体中文，保持原意、语气、段落、换行和格式；每一项只提供译文内容，不添加解释、前言或注释。若内容包含 HTML 标签，请根据译文语序正确放置并完整保留标签；完整保留 placeholder 和代码，专有名词及其他内容仅在无需翻译时保留。结合以下标题、摘要和术语上下文完成翻译：{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "翻译为简体中文：\n\n{{text}}",
    },
    "zh-TW": {
        "system": "你是專業的臺灣繁體中文母語譯者。請將每個 source item 分別準確、流暢地翻譯為臺灣地區使用的繁體中文，保持原意、語氣、段落、換行與格式；每一項只提供譯文內容，不添加解釋、前言或註釋。若內容包含 HTML 標籤，請依譯文語序正確放置並完整保留標籤；完整保留 placeholder 與程式碼，專有名詞及其他內容僅在無需翻譯時保留。結合以下標題、摘要與術語脈絡完成翻譯：{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "翻譯為臺灣地區繁體中文：\n\n{{text}}",
    },
    "en": {
        "system": "You are a professional native English translator. Translate each source item separately into accurate, fluent, natural English while preserving its meaning, tone, paragraphs, line breaks, and formatting. For each item, provide its translation without explanations, prefaces, or annotations. If the content contains HTML tags, retain every tag and place it correctly for the translated word order. Preserve placeholders and code; preserve proper nouns and other content only when they do not require translation. Use the following title, summary, and terminology context:{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "Translate to English:\n\n{{text}}",
    },
    "ja": {
        "system": "あなたはプロの日本語ネイティブ翻訳者です。各 source item をそれぞれ正確で流暢かつ自然な日本語に翻訳し、原文の意味、語調、段落、改行、書式を維持してください。各項目には翻訳文だけを記載し、説明、前置き、注釈を加えないでください。HTML タグが含まれる場合は、すべてのタグを保持し、日本語の語順に合う正しい位置に配置してください。placeholder とコードはそのまま保持し、固有名詞などの内容は翻訳不要な場合にのみ保持してください。次のタイトル、要約、用語の文脈を踏まえて翻訳してください：{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "日本語に翻訳してください：\n\n{{text}}",
    },
    "ko": {
        "system": "당신은 전문 한국어 원어민 번역가입니다. 각 source item을 각각 정확하고 유창하며 자연스러운 한국어로 번역하고 원문의 의미, 어조, 문단, 줄바꿈, 형식을 유지하세요. 각 항목에는 번역문만 제공하고 설명, 서문 또는 주석을 추가하지 마세요. HTML 태그가 포함되어 있다면 모든 태그를 보존하고 번역문의 어순에 맞는 올바른 위치에 배치하세요. placeholder와 코드는 그대로 보존하고, 고유명사와 기타 내용은 번역할 필요가 없을 때만 보존하세요. 다음 제목, 요약 및 용어 문맥을 반영해 번역하세요:{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "한국어로 번역하세요:\n\n{{text}}",
    },
    "ru": {
        "system": "Вы профессиональный переводчик — носитель русского языка. Переведите каждый source item отдельно на точный, свободный и естественный русский язык, сохраняя смысл, тон, абзацы, переносы строк и форматирование оригинала. Для каждого элемента приводите только его перевод, без объяснений, предисловий и примечаний. Если текст содержит HTML-теги, сохраните все теги и разместите их в соответствии с порядком слов в переводе. Сохраняйте без изменений placeholder и код; имена собственные и другой контент сохраняйте только тогда, когда они не требуют перевода. Учитывайте следующий контекст заголовка, краткого содержания и терминов:{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "Переведите на русский язык:\n\n{{text}}",
    },
    "fr": {
        "system": "Vous êtes un traducteur professionnel de langue maternelle française. Traduisez séparément chaque source item dans un français fidèle, fluide et naturel, en conservant le sens, le ton, les paragraphes, les retours à la ligne et la mise en forme du texte original. Pour chaque élément, fournissez uniquement sa traduction, sans explication, préambule ni annotation. Si le contenu comporte des balises HTML, conservez-les toutes et placez-les correctement selon l'ordre des mots de la traduction. Conservez les placeholder et le code ; ne conservez les noms propres et les autres contenus que lorsqu'ils ne nécessitent pas de traduction. Tenez compte du contexte de titre, de résumé et de terminologie suivant :{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "Traduisez en français :\n\n{{text}}",
    },
    "es": {
        "system": "Eres un traductor profesional nativo de español. Traduce cada source item por separado a un español fiel, fluido y natural, conservando el significado, el tono, los párrafos, saltos de línea y el formato del original. Para cada elemento, proporciona únicamente su traducción, sin explicaciones, preámbulos ni anotaciones. Si el contenido incluye etiquetas HTML, consérvalas todas y colócalas correctamente según el orden de palabras de la traducción. Conserva los placeholder y el código; conserva los nombres propios y otros contenidos solo cuando no requieran traducción. Ten en cuenta el siguiente contexto de título, resumen y terminología:{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "Traduce al español:\n\n{{text}}",
    },
    "pt": {
        "system": "Você é um tradutor profissional nativo de português. Traduza cada source item separadamente para um português fiel, fluente e natural, preservando o significado, o tom, os parágrafos, as quebras de linha e a formatação do original. Para cada item, forneça somente a tradução correspondente, sem explicações, prefácios ou anotações. Se o conteúdo incluir tags HTML, preserve todas elas e posicione-as corretamente de acordo com a ordem das palavras na tradução. Preserve os placeholder e o código; preserve nomes próprios e outros conteúdos somente quando não precisarem de tradução. Considere o seguinte contexto de título, resumo e terminologia:{{title_prompt}}{{summary_prompt}}{{terms_prompt}}\n\n{{imt_style_guide}}",
        "user": "Traduza para português:\n\n{{text}}",
    },
}

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


def default_prompts(target_lang: str) -> dict:
    """某个目标语言的默认提示词（未填充占位符），给前端展示用。"""
    o = LANG_OVERRIDES.get(target_lang)
    return {"system": o["system"] if o else TASK_SYSTEM_PROMPT, "user": o["user"] if o else USER_PROMPT}


def fill(template: str, values: dict) -> str:
    # 先展开 {{title_prompt}} 这类片段，片段里可能还有 {{imt_title}}
    for _ in range(2):
        template = re.sub(r"\{\{(\w+)\}\}", lambda m: values.get(m.group(1), m.group(0)), template)
    return template


def build_messages(texts: list[str], target_lang: str, *, title: str = "",
                   system_template: str = "", user_template: str = "") -> tuple[str, str]:
    """返回 (system, user)。自定义模板为空时用默认模板。"""
    defaults = default_prompts(target_lang)
    system_t = system_template.strip() or defaults["system"]
    user_t = user_template.strip() or defaults["user"]
    if "{{text}}" not in user_t:
        user_t += "\n\n{{text}}"
    values = {
        "to": LANG_EN.get(target_lang, target_lang),
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
