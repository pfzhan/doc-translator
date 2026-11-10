"""AI translation prompts, taken from the default configuration of Immersive Translate 1.33.3
(translationServices.ai in default_config.json).

They use its "AI batch marker protocol" (aiBatch marker protocol):
- Input: each paragraph is preceded by a line [[p0]], [[p1]], ..., and the last line is [[source_end]]
- Output: the same [[pN]] heading lines followed by the translations; the model may return them in any order
- System prompt = task prompt (taskSystemPrompt) + protocol description (protocolSystemPrompt)
- With a single paragraph the protocol is not used and the model outputs the translation directly

Per-language overrides (langOverrides) follow the plugin's matching rules: each item id is "<from>2<to>", and an
item applies when from is auto or the current source language and to is auto or the current target language.
Matching items are applied in order, and extends applies the inherited item first. So "auto2zh-CN" applies to
any source language, while "wyw2zh-CN" applies only when the source language is Classical Chinese and overrides
the former.

Placeholders: {{to}} target language, {{from}} source language, {{text}} source text, {{title_prompt}} document
title context. {{summary_prompt}} / {{terms_prompt}} / {{imt_style_guide}} are not used yet (left empty), so that
a summary and a glossary can be added later.
"""

import re

from .languages import LANG_OVERRIDES, LANGUAGES

# {{to}} / {{from}} use English language names, as the plugin does
LANG_EN = {code: en for code, en, _ in LANGUAGES}

TASK_SYSTEM_PROMPT = """You are a professional {{to}} native translator. Translate each source item accurately and fluently into {{to}}.

## Translation Requirements
- Treat each source item as a complete translation unit, including a single word, short phrase, heading, label, or site name. Do not ask for more context, a full article, or additional text.
- Return only the translated content for each item, following the required response format. Do not add explanations, apologies, greetings, or commentary. If an item should remain untranslated, return it unchanged.
- Preserve its meaning, tone, paragraph structure, and meaningful formatting.
- Keep code, placeholders, and content that must not be translated unchanged.
- Use supplied document context and terminology consistently.
- Translate repeated phrases of the same kind uniformly across the whole document, and keep one translation for each recurring term. When the target language is Chinese and a heading or contents entry names its unit (Chapter 2, Volume II, Part 3), write the number as a Chinese numeral (第二章, 第二卷, 第三部分). Keep bare numbers and dotted section numbers (2, 7.4, 1.2.3) in Arabic digits.

{{title_prompt}}{{summary_prompt}}{{terms_prompt}}

{{imt_style_guide}}"""

USER_PROMPT = "Translate to {{to}}:\n\n{{text}}"

# Protocol description appended to the system prompt for several paragraphs (the plugin's version with term
# extraction turned off)
PROTOCOL_SYSTEM_PROMPT = """

## Multi-item processing protocol
The input contains source items. Each item starts with its exact request id on
one title line, followed by that item's source text. The source list ends with
exactly one [[source_end]] line. Process every item independently according to
the active task instructions, using the other items only as context. Return each
supplied id exactly once in any order, followed by only its processed text.
An item may be a single word, a label, or an unfinished sentence; process it as
given and never ask for more context or the full text.
After all output blocks, processed text may span multiple lines. Preserve every id
title byte-for-byte.

Do not wrap the entire response or protocol envelope in a Markdown fence.
Fenced code blocks that belong to an item's processed content may be preserved.
Do not include explanations outside processed text."""

# Instruction appended when there is only one paragraph. When an example sentence that a book leaves unfinished
# on purpose ("IBM, a company with a long...") is sent alone, the model replies "请提供完整的原文"
SINGLE_ITEM_SUFFIX = """
The text may be a single word, a label, or an unfinished sentence; process it as
given and never ask for more context or the full text.
Return only the processed text, without source text, markers, titles,
explanations, or an enclosing Markdown fence. Preserve fenced code blocks that
belong to the processed content."""

# Bold, italic, font size and color in a block that differ from the body text. The model must carry the markers
# along so that the original styles can be written back after translation.
STYLE_SUFFIX = """

## Style placeholders
Some items contain {b} {/b}, {i} {/i}, {u} {/u}, {zN} {/z}, or {cRRGGBB} {/c}. They mark
bold, italic, underlined text, a different font size, or a text color (links and citations
are often colored).
Keep every marker unchanged, and move it with the words it wraps. Do not translate,
drop, or invent markers."""

_STYLE_IN_TEXT = re.compile(r"\{/?[biu]\}|\{z\d+\}|\{c[0-9a-f]{6}\}")

# Run boundary of a paragraph carried over to the next page. The model keeps it unchanged. When writing, the two
# parts are typeset as one paragraph that fills the first part before continuing; a unit with a display formula or
# an example sentence between its parts is split back into two blocks at the boundary (see pdf_merge._seam_apart).
BOUNDARY_SUFFIX = """

## Page-boundary placeholder
Some items contain a {|} placeholder. It marks where one paragraph crosses a
page. Keep {|} and every {vN} unchanged, in the same order. Do not translate,
drop, or move the braces."""

# Placeholders for inline formulas (formulas in PDFs, MathML and images in e-book paragraphs). The model often
# guesses {v1} from the context and writes it out as "="; once a placeholder is lost, the whole paragraph
# falls back to the original text.
FORMULA_SUFFIX = """

## Formula placeholders
Some items contain {v1}, {v2}, ... placeholders. Each stands for a formula, symbol,
or image that is put back in place after translation. Keep every {vN} exactly once, in
the matching place of the translation. Do not replace it with your guess of what it
contains, and do not drop or renumber it."""

_FORMULA_IN_TEXT = re.compile(r"\{v\d+\}")

# Appended when the batch contains HTML fragments (e-book rich-text paragraphs): keep inline formatting (bold,
# links)
HTML_SUFFIX = """

## HTML fragments
Some items are HTML fragments. Keep every tag name and attribute (including href
and class) and translate only the text between tags. Move a tag together with the
words it wraps so the markup follows the target word order. Do not add, drop, or
nest tags."""

# Notes on where an item appears in the document (which entry a subentry on an index page belongs to). Used only
# to choose word senses; not translated and not output
NOTES_SUFFIX = """

## Item notes
Some items come with a note on where they appear in the document. Use the note only
to choose the sense of the words. Translate the item itself; do not translate the note
or add it to the output."""

TITLE_PROMPT = "\n\n## Context Awareness\nDocument Metadata:\nTitle: “{{imt_title}}”"

# Glossary: one "source = translation" per line, kept consistent during translation
TERMS_PROMPT = "\n\n## Glossary\nUse these translations consistently:\n{{glossary}}"

# Fixing term translations before translating: the candidates are counted locally from the full text, and the
# model is asked to drop ordinary words
TERMS_SYSTEM_PROMPT = """You are preparing the terminology list for translating a document into {{to}}.{{title_prompt}}

The user lists candidate terms that recur in the document. They were counted mechanically, so
many are not terms. For each candidate that names a concept, method, object, or proper noun of
the document's field, give the single {{to}} rendering a specialist would use, so that every
occurrence is translated the same way. A longer term that contains a shorter kept term reuses
the shorter term's rendering (if "tokenization" is kept, "word tokenization" builds on the same
translation). Keep acronyms, product names, and code identifiers
unchanged when that is the usual practice.

Leave out:
- everyday words and phrases that need no fixed translation (result, value, step, large models),
  including single words of the field that are translated the obvious way in every context;
- pieces cut out of a longer phrase, idiom, or sentence ("account the cost" from "take into
  account the cost", "results suggest");
- variable names and symbols from formulas.

Write one line per kept term: the term exactly as given, " = ", then the translation.
Output nothing else."""

TERMS_USER_PROMPT = "Document excerpt:\n{{excerpt}}\n\nCandidate terms:\n{{terms}}"

# Looking up a word or short phrase selected on a web page (see lookup.py); the sentence it was selected from tells
# which sense is meant
LOOKUP_SYSTEM_PROMPT = """You are a dictionary for a reader of {{to}}. The user selected a word or short phrase on a web page and gives the sentence it was selected from.{{title_prompt}}

Write its dictionary entry, every explanation in {{to}}:
- headword: the selection in its dictionary form (the base form of an inflected word; a phrase stays whole).
- phonetic: its usual pronunciation, in IPA between slashes; empty when the language has no such convention.
- senses: its common senses, most common first, at most 5. For each, "pos" is the part of speech as an English
  abbreviation (n., v., adj., adv., prep., conj., pron., num., int., phr.) and "meaning" gives one to three short
  {{to}} renderings, separated by semicolons.
- in_context: what it means in the given sentence, as a short {{to}} phrase; empty when no sentence is given.

Reply with one JSON object and nothing else:
{"headword": "", "phonetic": "", "senses": [{"pos": "", "meaning": ""}], "in_context": ""}"""

LOOKUP_USER_PROMPT = "Selection: {{text}}\n\nSentence: {{context}}"

SOURCE_END = "[[source_end]]"
MARKER_RE = re.compile(r"^\[\[(p\d+)\]\]$", re.I)
CONTROL_RE = re.compile(r"^\[\[(?:p\d+|source_end|terms|term|end)\]\]$", re.I)
# The model sometimes writes the closing marker in upper case, or appends it to the last line of the
# translation: "……8200。[[END]]"
TAIL_CONTROL_RE = re.compile(r"\s*\[\[(?:source_end|terms|term|end)\]\]\s*$", re.I)


def pick_marker(texts: list[str]) -> str:
    """Choose the paragraph marker prefix. When the source text itself has [[pN]] on a line of its own, another
    prefix is used so that the text is not cut there as a protocol marker."""
    for prefix in ("p", "q", "s"):
        if not any(re.search(rf"^\[\[{prefix}\d+\]\]$", t, re.M) for t in texts):
            return prefix
    return "p"  # Edge case where all three prefixes are taken: keep the default


# The plugin's removeResRegexs: remove the <think> section output by reasoning models
THINK_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*|^\s*</think>\s*")

_OVERRIDES = {o["id"]: o for o in LANG_OVERRIDES}


def lang_name(code: str) -> str:
    return LANG_EN.get(code, code)


def default_prompts(target_lang: str, source_lang: str = "auto") -> dict[str, str]:
    """Compute the default prompts for a language pair by the plugin's langOverrides rules (placeholders not
    filled)."""
    result = {"system": TASK_SYSTEM_PROMPT, "user": USER_PROMPT}

    def apply(item: dict[str, str]) -> None:
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


def fill(template: str, values: dict[str, str]) -> str:
    # Expand fragments such as {{title_prompt}} first; a fragment may contain {{imt_title}} in turn
    for _ in range(2):
        template = re.sub(r"\{\{(\w+)\}\}", lambda m: values.get(m.group(1), m.group(0)), template)
    return template


def build_messages(
    texts: list[str],
    target_lang: str,
    *,
    source_lang: str = "auto",
    title: str = "",
    system_template: str = "",
    user_template: str = "",
    contains_html: bool = False,
    glossary: str = "",
    notes: list[str] | None = None,
) -> tuple[str, str]:
    """Return (system, user). The default templates are used when the custom templates are empty. source_lang is
    the source language chosen by the user or detected.
    notes is aligned with texts and holds notes on where each paragraph appears in the document; an empty string
    means no note."""
    defaults = default_prompts(target_lang, source_lang)
    system_t = system_template.strip() or defaults["system"]
    user_t = user_template.strip() or defaults["user"]
    if "{{text}}" not in user_t:
        user_t += "\n\n{{text}}"
    # Glossary: one "source = translation" per line, with extra blank lines removed
    terms = "\n".join(line for line in (glossary or "").splitlines() if line.strip())
    values = {
        "to": lang_name(target_lang),
        # When the source language is unknown, the plugin also fills {{from}} with "Auto Detect"
        "from": lang_name(source_lang or "auto"),
        "title_prompt": TITLE_PROMPT if title else "",
        "imt_title": title,
        "summary_prompt": "",
        "terms_prompt": TERMS_PROMPT.replace("{{glossary}}", terms) if terms else "",
        "glossary": terms,
        "imt_style_guide": "",
    }
    if len(texts) == 1:
        system = fill(system_t, values).strip() + "\n" + SINGLE_ITEM_SUFFIX
        body = texts[0]
    else:
        system = fill(system_t, values).strip() + PROTOCOL_SYSTEM_PROMPT
        marker = pick_marker(texts)
        body = "\n".join(f"[[{marker}{i}]]\n{t}" for i, t in enumerate(texts)) + "\n" + SOURCE_END
    if contains_html:
        system += HTML_SUFFIX
    if any("{|}" in text for text in texts):
        system += BOUNDARY_SUFFIX
    if any(_FORMULA_IN_TEXT.search(text) for text in texts):
        system += FORMULA_SUFFIX
    if any(_STYLE_IN_TEXT.search(text) for text in texts):
        system += STYLE_SUFFIX
    noted = [(i, note) for i, note in enumerate(notes or []) if note]
    if noted:
        system += NOTES_SUFFIX + "\n"
        if len(texts) == 1:
            system += f"Note: {noted[0][1]}"
        else:
            system += "\n".join(f"[[{marker}{i}]] {note}" for i, note in noted)
    # {{text}} is replaced last, so that a {{xxx}} that happens to be in the source text is not taken as a
    # placeholder
    user = fill(user_t.replace("{{text}}", "\0TEXT\0"), values).replace("\0TEXT\0", body)
    return system, user


def build_terms_messages(
    terms: list[str], target_lang: str, *, source_lang: str = "auto", title: str = "", excerpt: str = ""
) -> tuple[str, str]:
    """(system, user) for fixing term translations."""
    values = {
        "to": lang_name(target_lang),
        "from": lang_name(source_lang or "auto"),
        "title_prompt": TITLE_PROMPT if title else "",
        "imt_title": title,
    }
    system = fill(TERMS_SYSTEM_PROMPT, values)
    user = TERMS_USER_PROMPT.replace("{{terms}}", "\n".join(terms)).replace("{{excerpt}}", excerpt.strip() or "(none)")
    return system, user


def build_lookup_messages(text: str, context: str, target_lang: str, *, title: str = "") -> tuple[str, str]:
    """(system, user) for looking up a selected word or phrase; context is the sentence it was selected from."""
    values = {"to": lang_name(target_lang), "title_prompt": TITLE_PROMPT if title else "", "imt_title": title}
    system = fill(LOOKUP_SYSTEM_PROMPT, values)
    # Both filled in one pass, so that a {{context}} in the selected text is not taken as a placeholder
    parts = {"text": text.strip(), "context": context.strip() or "(none)"}
    user = re.sub(r"\{\{(text|context)\}\}", lambda m: parts[m.group(1)], LOOKUP_USER_PROMPT)
    return system, user


def clean_output(content: str) -> str:
    content = THINK_RE.sub("", content).strip()
    # Remove a ``` fence around the whole reply
    m = re.fullmatch(r"```[\w-]*\n([\s\S]*?)\n```", content)
    return m.group(1).strip() if m else content


def strip_control_marks(text: str) -> str:
    """Remove protocol markers that leaked into the translation ([[END]], [[source_end]]). Old cache entries may
    already contain them."""
    lines = [line for line in text.split("\n") if not CONTROL_RE.match(line.strip())]
    return TAIL_CONTROL_RE.sub("", "\n".join(lines)).strip()


# A [[word]] that looks like a protocol marker
_BRACKET_WORD_RE = re.compile(r"[ \t]*\[\[/?[A-Za-z_][\w-]*\]\]")


def drop_invented_marks(src: str, dst: str) -> str:
    """Remove a [[word]] that is not in the source text. The model occasionally invents one modelled on the
    protocol markers ([[toolong]], [[/INST]]) and appends it to the translation."""
    if "[[" not in dst:
        return dst
    kept = {match.group().strip() for match in _BRACKET_WORD_RE.finditer(src)}
    lines: list[str] = []
    for line in dst.split("\n"):
        cleaned = _BRACKET_WORD_RE.sub(lambda match: match.group() if match.group().strip() in kept else "", line)
        # A line holding only such a marker is removed entirely
        if cleaned == line or cleaned.strip():
            lines.append(cleaned)
    return "\n".join(lines)


# Markers in the text sent for translation: on/off markers for emphasis, font size and color, formula
# placeholders, and page boundaries
_MARK_TOKEN_RE = re.compile(r"\{(?:/?[biuzc]|z\d+|c[0-9a-f]{6}|v\d+|\|)\}")
_PLACEHOLDER_RE = re.compile(r"\{v\d+\}")


def without_marks(text: str) -> str:
    return _MARK_TOKEN_RE.sub("", text)


def placeholders(text: str) -> set[str]:
    """Formula placeholders {v1}, {v2}, ..."""
    return set(_PLACEHOLDER_RE.findall(text))


def _braces_at_marks(text: str) -> tuple[list[int], list[int]]:
    """Positions of literal { and } directly adjacent to markers. The braces of the markers themselves do not
    count."""
    marks = [(m.start(), m.end()) for m in _MARK_TOKEN_RE.finditer(text)]
    starts = {start for start, _end in marks}
    ends = {end for _start, end in marks}
    inside = {i for start, end in marks for i in range(start, end)}
    before = [i for i, ch in enumerate(text) if ch == "{" and i not in inside and i + 1 in starts]
    after = [i for i, ch in enumerate(text) if ch == "}" and i not in inside and i in ends]
    return before, after


# Malformed markers: whitespace, zero-width characters or question marks inside the braces ({ i}, {​i}, { i? }),
# square brackets instead of braces ([i]), a closing > ({/b>}), or a color closing marker carrying the color code
# ({/cb30000})
_LOOSE_MARK_RE = re.compile(r"[{\[][^{}\[\]<>\n]{1,14}[}\]>]")
_NOT_MARK_CHAR_RE = re.compile(r"[^A-Za-z0-9/|]")
_CLOSE_WITH_ARG_RE = re.compile(r"^/(c)[0-9a-f]{6}$|^/(z)\d+$")


def tidy_marks(src: str, dst: str) -> str:
    """The model occasionally writes a marker in a malformed way; it is then not recognized as a marker and is
    rendered as a literal i or /cb30000. Restore the standard form, using only markers present in the source text;
    the same forms that already occur in the source body text ({ i}, x[i]) are left alone."""
    marks = set(_MARK_TOKEN_RE.findall(src))

    def tidy(match: re.Match[str]) -> str:
        text = match.group(0)
        if text in marks or text in src:
            return text
        core = _NOT_MARK_CHAR_RE.sub("", text[1:-1])
        core = _CLOSE_WITH_ARG_RE.sub(lambda close: "/" + (close.group(1) or close.group(2)), core)
        mark = "{" + core + "}"
        return mark if mark in marks else text

    return _LOOSE_MARK_RE.sub(tidy, dst)


def drop_added_braces(src: str, dst: str) -> str:
    """Remove braces the model wrapped around markers: "{{b}填充词{/b}}", {/b}}, {{c0000ff}Chen{/c}}.
    When the source body text has braces such as {n,m}, the model wraps the markers in a layer of braces too.
    Only braces directly adjacent to markers are removed, and only when the source text has none at the same
    positions; the set {{i}o{/i}{v1}, …} is in the source text and is kept."""
    if "{" not in dst:
        return dst
    src_before, src_after = _braces_at_marks(src)
    before, after = _braces_at_marks(dst)
    drop = set()
    if before and not src_before:
        drop.update(before)
    if after and not src_after:
        drop.update(after)
    if not drop:
        return dst
    return "".join(ch for i, ch in enumerate(dst) if i not in drop)


def parse_markers(content: str, count: int, marker: str = "p") -> dict[int, str]:
    """Parse a reply in [[pN]] format and return {index: translation}; missing indices are not in the result.
    marker must match the one used by build_messages."""
    marker_re = MARKER_RE if marker == "p" else re.compile(rf"^\[\[({marker}\d+)\]\]$", re.I)
    control_re = (
        CONTROL_RE if marker == "p" else re.compile(rf"^\[\[(?:{marker}\d+|source_end|terms|term|end)\]\]$", re.I)
    )
    result: dict[int, str] = {}
    current: int | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None and 0 <= current < count and current not in result:
            result[current] = "\n".join(buf).strip()

    for line in clean_output(content).split("\n"):
        stripped = line.strip()
        if control_re.match(stripped):
            flush()
            m = marker_re.match(stripped)
            current = int(m.group(1)[1:]) if m else None
            buf = []
            if stripped.lower() in ("[[terms]]", "[[end]]"):
                break
        elif current is not None:
            tail = TAIL_CONTROL_RE.search(line)
            if tail:
                buf.append(line[: tail.start()])
                flush()
                current = None
                continue
            buf.append(line)
    flush()
    return {k: v for k, v in result.items() if v}
