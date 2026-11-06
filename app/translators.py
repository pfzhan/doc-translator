"""Translation engines. Each engine implements translate_batch(texts) -> list[str].

LLM engines (OpenAI / Grok / Gemini / Claude / OpenAI-compatible) share the same batch translation logic; the
prompts and the [[pN]] marker protocol are taken from Immersive Translate (see prompts.py). Paragraphs the model
leaves out are translated again one by one.
Providers differ only in request format, which subclasses implement in _complete.
"""

import asyncio
import hashlib
import re
import unicodedata
import weakref
from collections.abc import Iterable
from urllib.parse import urlparse

import httpx

from . import glossary as glossary_mod
from . import prompts
from .json_types import JSONDict
from .languages import GOOGLE_CODES, LANGUAGES

# Language list for the frontend drop-down: [(code, English name, native name)], in the same order as the plugin
LANG_CODES = {code for code, _, _ in LANGUAGES}


class TranslatorError(Exception):
    def __init__(self, message: str = "", status: int | None = None) -> None:
        super().__init__(message)
        # HTTP status code returned by the API; None when there is none, such as for network or parse errors
        self.status = status


class TruncatedError(TranslatorError):
    """The model stopped at its output limit, so the reply holds only part of the translation. A batch is split and
    translated again in halves; for a single paragraph the result is empty and Runner keeps the original."""

    def __init__(self) -> None:
        super().__init__("模型输出达到长度上限，译文不完整")


# Concurrency gates by event loop, then by (service id, concurrency); see Translator._sem. A semaphore belongs to the
# loop it is first used in, and the tests run each scenario in a loop of its own
_GATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, int], asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)


class Translator:
    # Maximum paragraphs and characters per batch, used by Runner to split batches
    max_batch_items = 20
    max_batch_chars = 3000
    concurrency = 4

    def __init__(self, target_lang: str, source_lang: str = "auto") -> None:
        if target_lang not in LANG_CODES or target_lang == "auto":
            raise TranslatorError(f"不支持的目标语言: {target_lang}")
        if source_lang not in LANG_CODES:
            raise TranslatorError(f"不支持的源语言: {source_lang}")
        self.target_lang = target_lang
        self.source_lang = source_lang
        # When the source language is "自动检测", Runner fills this in after detecting the document language; used
        # for {{from}} in the prompt
        self.detected_source = ""
        # Translation service id, part of the cache key: a different service translates again, the same service
        # reuses the cache
        self.service_id = ""
        # Token usage returned by the API (LLM engines accumulate it in _complete); stays 0 for engines that do not
        # report it, such as Google
        self.usage = {"prompt": 0, "completion": 0}
        # Filled in by Runner from the caller's flags: these source texts are e-book rich-text fragments, which
        # cannot be guessed from angle brackets
        self.html_texts: set[str] = set()
        # Filled in by Runner from the notes the caller gives: source text -> note on where it appears in the
        # document (used only by LLM engines)
        self.notes: dict[str, str] = {}

    @property
    def _sem(self) -> asyncio.Semaphore:
        """The concurrency gate every request of this engine goes through, including the batch splitting and
        retranslation inside translate_batch, which would otherwise bypass Runner's worker limit.

        Engines made from the same service share one gate, so the service's concurrency limits all of its requests
        together: several jobs, web page requests and service tests. Each job and request makes its own engine, and
        a gate per engine let them add up to many times the limit and run into rate limits. An engine without a
        service id (tests) has a gate of its own."""
        loop = asyncio.get_running_loop()
        gates = _GATES.setdefault(loop, {})
        key = (self.service_id, self.concurrency) if self.service_id else (f"engine:{id(self)}", self.concurrency)
        gate = gates.get(key)
        if gate is None:
            gate = gates[key] = asyncio.Semaphore(self.concurrency)
            if not self.service_id:
                weakref.finalize(self, gates.pop, key, None)
        return gate

    def is_html_item(self, text: str) -> bool:
        return text in self.html_texts

    def _record_usage(self, prompt: int | None, completion: int | None) -> None:
        self.usage["prompt"] += prompt or 0
        self.usage["completion"] += completion or 0

    @property
    def effective_source(self) -> str:
        return self.detected_source if self.source_lang == "auto" and self.detected_source else self.source_lang

    @property
    def cache_key(self) -> str:
        return f"{self.service_id}:{type(self).__name__}:{self.source_lang}:{self.target_lang}"

    async def translate_batch(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300]
    # Error responses from relay services are not always JSON objects; they may be arrays or strings
    if not isinstance(data, dict):
        return str(data)[:300]
    err = data.get("error", data)
    msg = err.get("message") if isinstance(err, dict) else err
    return str(msg or resp.text)[:300]


# Numbers of three or more digits, such as years, page numbers and identifiers, should be present unchanged
# before and after translation
_NUMBER_RE = re.compile(r"\d{3,}")


def _numbers_mismatch(source: str, translation: str) -> bool:
    """None of the source numbers is in the translation, but other numbers are: most likely the model merged two
    paragraphs into one and the following translations are all shifted."""
    src = set(_NUMBER_RE.findall(source))
    dst = set(_NUMBER_RE.findall(translation))
    return bool(src) and bool(dst) and not (src & dst)


# When the translation/source length ratio deviates from the typical value by this factor, most likely the model
# shifted adjacent paragraphs or stopped halfway.
# Measured for English to Chinese: the typical value is about 0.35, normal translations go as low as about 0.58
# times that, and verbatim copies as high as about 3 times
_RATIO_LOW = 0.45
_RATIO_HIGH = 4.0
# Minimum source length without markers for comparing lengths; ratios of short sentences vary widely anyway
_RATIO_MIN_CHARS = 60
# Minimum number of paragraphs for computing the typical value
_RATIO_SAMPLES = 3


def _length_ratio(source: str, translation: str) -> float | None:
    length = len(prompts.without_marks(source).strip())
    if length < _RATIO_MIN_CHARS:
        return None
    return len(prompts.without_marks(translation).strip()) / length


def typical_ratio(pairs: Iterable[tuple[str, str]]) -> float | None:
    """Median length ratio of a batch of (source, translation). Not computed when there are too few
    paragraphs."""
    ratios = sorted(r for source, translation in pairs if (r := _length_ratio(source, translation)) is not None)
    if len(ratios) < _RATIO_SAMPLES:
        return None
    return ratios[len(ratios) // 2]


def misplaced(source: str, translation: str, typical: float | None) -> bool:
    """The translation most likely belongs to another paragraph: it carries formula placeholders the source lacks,
    or its length ratio is far from the typical value of the batch. Ratios of short sentences vary widely anyway,
    so only the upper bound is checked: counting the source as _RATIO_MIN_CHARS, a translation still much longer
    has picked up another paragraph (a three-word heading got the translation of the next paragraph)"""
    if prompts.placeholders(translation) - prompts.placeholders(source):
        return True
    if not typical:
        return False
    ratio = _length_ratio(source, translation)
    if ratio is not None:
        return not typical * _RATIO_LOW <= ratio <= typical * _RATIO_HIGH
    return len(prompts.without_marks(translation).strip()) > typical * _RATIO_HIGH * _RATIO_MIN_CHARS


# Scripts used to detect foreign text, taken from the first word of a character's Unicode name. Latin and Greek
# are not among them: code, personal names and formula symbols ("μ律", λ) may appear in translations into any
# language
_SCRIPTS = frozenset(
    {
        "CYRILLIC",
        "ARMENIAN",
        "HEBREW",
        "ARABIC",
        "SYRIAC",
        "THAANA",
        "DEVANAGARI",
        "BENGALI",
        "GURMUKHI",
        "GUJARATI",
        "ORIYA",
        "TAMIL",
        "TELUGU",
        "KANNADA",
        "MALAYALAM",
        "SINHALA",
        "THAI",
        "LAO",
        "TIBETAN",
        "MYANMAR",
        "GEORGIAN",
        "HANGUL",
        "ETHIOPIC",
        "KHMER",
        "MONGOLIAN",
        "HIRAGANA",
        "KATAKANA",
        "CJK",
    }
)
# Scripts not visible in the native name: Japanese kana
_EXTRA_SCRIPTS = {"ja": {"HIRAGANA", "KATAKANA"}}
_NATIVE_NAMES = {code: native for code, _, native in LANGUAGES}


def _scripts(text: str) -> set[str]:
    found = set()
    for ch in text:
        if ch.isalpha() and ord(ch) > 0x2FF:
            name = unicodedata.name(ch, "").removeprefix("HALFWIDTH ").split(" ", 1)[0]
            if name in _SCRIPTS:
                found.add(name)
    return found


def _symbols(text: str) -> set[str]:
    """Miscellaneous symbols, dingbats, emoji and private-use characters. Such characters in a translation should
    all come from the source text."""
    return {ch for ch in text if 0x2600 <= ord(ch) <= 0x27BF or ord(ch) >= 0x1F000 or unicodedata.category(ch) == "Co"}


def stray_chars(source: str, translation: str, target_lang: str) -> bool:
    """The translation contains a script used by neither the source text nor the target language, such as half a
    Russian word in a Chinese translation ("убед自己"), or symbols the source lacks (a run of "♀♀♀" after a
    heading). The scripts of the target language are read from its native name ("简体中文", "Русский")."""
    if _symbols(translation) - _symbols(source):
        return True
    allowed = _scripts(source) | _scripts(_NATIVE_NAMES.get(target_lang, ""))
    allowed |= _EXTRA_SCRIPTS.get(target_lang, set())
    return bool(_scripts(translation) - allowed)


# These status codes mean our configuration is wrong (bad Key, no permission, wrong address); retrying will not
# help, so report the error at once
FATAL_STATUS = {401, 403, 404}


def _retryable(status: int) -> bool:
    return status not in FATAL_STATUS


def _retry_delay(resp: httpx.Response, default: float) -> float:
    """Retry delay: use the Retry-After response header when present (capped at 60 s), otherwise the given
    exponential backoff value."""
    try:
        delay = float(resp.headers.get("retry-after") or 0)
    except ValueError:
        delay = 0
    return min(delay, 60) if delay > 0 else default


def _json_or_error(resp: httpx.Response, url: str) -> JSONDict:
    """Parse a JSON response. Give a readable error when a web page comes back (common when the API address lacks
    /v1 and the request lands on the relay service's home page)."""
    try:
        data = resp.json()
    except ValueError:
        ctype = resp.headers.get("content-type", "")
        hint = (
            "，请检查 API 地址是否完整（通常以 /v1 结尾）"
            if "html" in ctype or resp.text.lstrip().startswith("<")
            else ""
        )
        raise TranslatorError(f"接口返回的不是 JSON（{ctype or '未知类型'}）：{url}{hint}") from None
    if not isinstance(data, dict):
        raise TranslatorError(f"返回格式异常: {str(data)[:200]}")
    return data


def _with_default_path(url: str, path: str) -> str:
    """Append the version path when the API address is only a domain (no path), as clients such as the OpenAI SDK
    and Cherry Studio do."""
    url = url.strip().rstrip("/")
    if url and not urlparse(url).path:
        url += path
    return url


# Version path appended when only a domain is given: /v1 for Claude and the OpenAI family, /v1beta for Gemini
DEFAULT_API_PATH = {"openai": "/v1", "grok": "/v1", "custom": "/v1", "claude": "/v1", "gemini": "/v1beta"}


class LLMTranslator(Translator):
    provider = ""
    retries = 4

    def __init__(
        self,
        target_lang: str,
        source_lang: str = "auto",
        *,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        concurrency: int = 4,
        max_items: int = 20,
        max_chars: int = 3000,
        temperature: float | None = 0.0,
        prompt: str = "",
        user_prompt: str = "",
        api_format: str = "",
        auth_style: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
        glossary: str = "",
        auto_glossary: bool = True,
    ) -> None:
        super().__init__(target_lang, source_lang)
        if not model:
            raise TranslatorError("未设置模型")
        # api_format: the OpenAI family may choose "responses" (Responses API); the default is Chat Completions
        # auth_style: Claude may choose "bearer" (Authorization: Bearer, common with relay services); the default is
        # x-api-key
        self.api_format = api_format
        self.auth_style = auth_style
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = max(1, int(concurrency))
        self.max_batch_items = max(1, int(max_items))
        self.max_batch_chars = max(200, int(max_chars))
        self.temperature = temperature
        self.prompt = prompt.strip()
        self.user_prompt = user_prompt.strip()
        self.glossary = glossary
        # Find terms in the document and fix their translations before translating (Runner calls define_terms and
        # the result goes into doc_glossary)
        self.auto_glossary = auto_glossary
        self.doc_glossary: list[glossary_mod.Entry] = []
        # Document title, filled into {{title_prompt}} of the prompt as context
        self.title = ""
        self.client = httpx.AsyncClient(timeout=180, transport=transport)

    @property
    def cache_key(self) -> str:
        # Model, prompts and glossary are part of the cache key: changing them retranslates the whole book
        # automatically, and changing back still hits that configuration's own cache.
        # The source language affects the prompts ({{from}}, overrides such as wyw2zh-CN), so the actual detected
        # source language is used
        p = hashlib.sha1(f"{self.prompt}\0{self.user_prompt}\0{self.glossary}".encode()).hexdigest()[:8]
        key = f"{self.service_id}:{self.provider}:{self.model}:{p}:{self.effective_source}:{self.target_lang}"
        # The document glossary also affects the translation. Without one the key is unchanged, so earlier cache
        # entries still hit
        if self.doc_glossary:
            key += ":g" + hashlib.sha1(glossary_mod.render(self.doc_glossary).encode()).hexdigest()[:8]
        return key

    def glossary_for(self, texts: list[str]) -> str:
        """Glossary for this batch: the user's entries plus those found in the document, keeping only entries that
        occur in the source text.

        Lines from the user not written as "source = translation" cannot be tied to any paragraph and go with every
        batch.
        """
        lines = [line.strip() for line in self.glossary.splitlines() if line.strip()]
        loose = [line for line in lines if not glossary_mod.parse(line)]
        entries = glossary_mod.merge(glossary_mod.parse(self.glossary), self.doc_glossary)
        return "\n".join([*loose, glossary_mod.render(glossary_mod.relevant(entries, texts))]).strip()

    async def define_terms(self, terms: list[str], excerpt: str = "") -> list[glossary_mod.Entry]:
        """Fix the translations of the candidate terms in one request. Candidates the model dropped or rewrote are
        not accepted."""
        system, user = prompts.build_terms_messages(
            terms,
            self.target_lang,
            source_lang=self.effective_source,
            title=self.title,
            excerpt=excerpt,
        )
        async with self._sem:
            content = await self._complete(system, user)
        wanted = {term.casefold(): term for term in terms}
        return [
            (wanted[source.casefold()], target)
            for source, target in glossary_mod.parse(prompts.clean_output(content))
            if source.casefold() in wanted
        ]

    def build_messages(self, texts: list[str]) -> tuple[str, str]:
        return prompts.build_messages(
            texts,
            self.target_lang,
            source_lang=self.effective_source,
            title=self.title,
            system_template=self.prompt,
            user_template=self.user_prompt,
            contains_html=any(self.is_html_item(t) for t in texts),
            glossary=self.glossary_for(texts),
            notes=[self.notes.get(t, "") for t in texts],
        )

    async def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    async def _post(self, url: str, payload: JSONDict, headers: dict[str, str]) -> JSONDict:
        """POST with retries. Some reasoning models do not accept temperature; on such a 400 it is removed and the
        request is sent again.

        Relay services often wrap occasional upstream failures (account pool rotation, a channel going offline
        temporarily) as a 400, and the same request succeeds a few seconds later, so a 400 is treated as retryable
        too, except when it clearly comes from a configuration error on our side.
        """
        for attempt in range(self.retries):
            last = attempt == self.retries - 1
            try:
                resp = await self.client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as e:
                if last:
                    raise TranslatorError(f"请求失败: {e}") from e
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return _json_or_error(resp, url)
            msg = _error_text(resp)
            # The last attempt does not strip temperature and resend, so the real error is raised
            if resp.status_code == 400 and "temperature" in msg.lower() and self.temperature is not None and not last:
                self.temperature = None
                payload = {k: v for k, v in payload.items() if k != "temperature"}
                gen = payload.get("generationConfig")
                if isinstance(gen, dict):
                    payload["generationConfig"] = {k: v for k, v in gen.items() if k != "temperature"}
                continue
            # A 400 is retried unchanged only once (for occasional failures); if it fails again, translate_batch splits
            # the batch instead of waiting here in vain
            if _retryable(resp.status_code) and not last and not (resp.status_code == 400 and attempt >= 1):
                await asyncio.sleep(_retry_delay(resp, 2 ** (attempt + 1)))
                continue
            raise TranslatorError(f"接口错误 {resp.status_code}: {msg}", status=resp.status_code)
        raise TranslatorError("已达到最大重试次数")

    async def _get(self, url: str, headers: dict[str, str], params: JSONDict | None = None) -> JSONDict:
        try:
            resp = await self.client.get(url, headers=headers, params=params)
        except httpx.HTTPError as e:
            raise TranslatorError(f"请求失败: {e}") from e
        if resp.status_code != 200:
            raise TranslatorError(f"接口错误 {resp.status_code}: {_error_text(resp)}")
        return _json_or_error(resp, url)

    async def list_models(self) -> list[str]:
        raise NotImplementedError

    async def translate_batch(self, texts: list[str]) -> list[str]:
        system, user = self.build_messages(texts)
        try:
            # The only place that actually sends requests; recursive batch splitting / retranslation ends up here too,
            # all limited by the same semaphore
            async with self._sem:
                content = await self._complete(system, user)
                # An empty reply is mostly an occasional upstream failure; ask again unchanged. If it is still empty,
                # a batch is retranslated paragraph by paragraph below, and for a single paragraph Runner falls back
                # to the source text
                if not content.strip():
                    content = await self._complete(system, user)
        except TranslatorError as e:
            # Relay services consistently reject some combinations of paragraphs (measured: the same batch of 8
            # paragraphs got a 400 every time, while each part translated after splitting).
            # Retrying unchanged does not help, so split into two halves translated separately. This is done only
            # for 400: splitting does not help configuration errors (401/403/404), and 429 and 5xx were already
            # retried with backoff in _post, so splitting would only multiply requests.
            # A reply cut off at the model's output limit is split the same way: half the batch needs half the output
            truncated = isinstance(e, TruncatedError)
            if len(texts) == 1 and truncated:
                return [""]  # Runner keeps the original and does not cache it
            if len(texts) == 1 or (e.status != 400 and not truncated):
                raise
            mid = len(texts) // 2
            # Translate both halves in parallel: serially, the deeper the splitting goes, the longer the last
            # paragraphs wait in the queue
            left, right = await asyncio.gather(self.translate_batch(texts[:mid]), self.translate_batch(texts[mid:]))
            return left + right
        if len(texts) == 1:
            out = prompts.clean_output(content)
            # Ask again for a translation carrying foreign text; if it still does, accept it, and Runner drops and
            # retranslates it at the next re-layout
            if stray_chars(texts[0], out, self.target_lang):
                try:
                    async with self._sem:
                        again = prompts.clean_output(await self._complete(system, user))
                except TruncatedError:
                    again = ""
                if again.strip() and not stray_chars(texts[0], again, self.target_lang):
                    out = again
            return [out]
        found = prompts.parse_markers(content, len(texts), marker=prompts.pick_marker(texts))
        # Retranslate paragraphs the model left out (the same idea as the plugin's recovery, filling in only the
        # missing ones), sent in parallel. Translations carrying foreign text are retranslated too.
        # Mismatched numbers, formula placeholders or lengths mean shifted translations: the marker count matches,
        # but the content belongs to other paragraphs. When the model leaves out or merges one paragraph,
        # the following ones all shift by one, and only one of them with a very different length is detectable,
        # so the whole batch is retranslated paragraph by paragraph
        typical = typical_ratio((texts[i], found[i]) for i in found)
        shifted = any(_numbers_mismatch(texts[i], found[i]) or misplaced(texts[i], found[i], typical) for i in found)
        missing = [
            i
            for i in range(len(texts))
            if shifted or i not in found or stray_chars(texts[i], found[i], self.target_lang)
        ]
        if missing:
            redone = await asyncio.gather(*(self.translate_batch([texts[i]]) for i in missing))
            for i, [text] in zip(missing, redone):
                found[i] = text
            # A reply to a single retranslation may also not be a translation ("请提供完整的原文"), so check again
            # against the typical length ratio of the batch after retranslation.
            # If asking once more is still wrong, return empty; Runner falls back to the source text and nothing is
            # cached
            typical = typical_ratio((texts[i], found[i]) for i in found)
            wrong = [i for i in missing if misplaced(texts[i], found[i], typical)]
            rechecked = await asyncio.gather(*(self.translate_batch([texts[i]]) for i in wrong))
            for i, [retried] in zip(wrong, rechecked):
                found[i] = "" if misplaced(texts[i], retried, typical) else retried
        return [found[i] for i in range(len(texts))]

    async def aclose(self) -> None:
        await self.client.aclose()


class OpenAITranslator(LLMTranslator):
    """OpenAI Chat Completions format. Grok and the various OpenAI-compatible services use it too."""

    provider = "openai"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _complete(self, system: str, user: str) -> str:
        if self.api_format == "responses":
            return await self._complete_responses(system, user)
        payload: JSONDict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = await self._post(f"{self.base_url}/chat/completions", payload, self._headers())
        usage = data.get("usage") or {}
        self._record_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise TranslatorError(f"返回格式异常: {str(data)[:200]}") from e
        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            raise TruncatedError()
        # Relay services occasionally return a message without content; treat it as an empty reply, like an empty
        # reply from other APIs
        return (message.get("content") if isinstance(message, dict) else "") or ""

    async def _complete_responses(self, system: str, user: str) -> str:
        """OpenAI Responses API (the default for Codex / Grok Build)."""
        payload = {"model": self.model, "instructions": system, "input": user, "store": False}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = await self._post(f"{self.base_url}/responses", payload, self._headers())
        usage = data.get("usage") or {}
        self._record_usage(usage.get("input_tokens"), usage.get("output_tokens"))
        if data.get("status") == "incomplete" and (data.get("incomplete_details") or {}).get("reason") == (
            "max_output_tokens"
        ):
            raise TruncatedError()
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text
        texts = [
            c.get("text", "")
            for item in data.get("output") or []
            if item.get("type") == "message"
            for c in item.get("content") or []
            if c.get("type") in ("output_text", "text")
        ]
        if not texts and data.get("status") not in (None, "completed"):
            raise TranslatorError(
                f"返回状态异常: {data.get('status')} {str(data.get('incomplete_details') or '')[:200]}"
            )
        return "".join(texts)

    async def list_models(self) -> list[str]:
        data = await self._get(f"{self.base_url}/models", self._headers())
        return sorted(m["id"] for m in data.get("data", []))


class GrokTranslator(OpenAITranslator):
    provider = "grok"


class CustomOpenAITranslator(OpenAITranslator):
    provider = "custom"


class ClaudeTranslator(LLMTranslator):
    """Anthropic Messages API."""

    provider = "claude"
    api_version = "2023-06-01"

    def _headers(self) -> dict[str, str]:
        if self.auth_style == "bearer":
            return {"Authorization": f"Bearer {self.api_key}", "anthropic-version": self.api_version}
        return {"x-api-key": self.api_key, "anthropic-version": self.api_version}

    async def _complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 8192,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = await self._post(f"{self.base_url}/messages", payload, self._headers())
        usage = data.get("usage") or {}
        self._record_usage(usage.get("input_tokens"), usage.get("output_tokens"))
        if data.get("stop_reason") == "max_tokens":
            raise TruncatedError()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    async def list_models(self) -> list[str]:
        data = await self._get(f"{self.base_url}/models", self._headers(), {"limit": 1000})
        return [m["id"] for m in data.get("data", [])]


class GeminiTranslator(LLMTranslator):
    """Google Gemini generateContent API."""

    provider = "gemini"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    async def _complete(self, system: str, user: str) -> str:
        gen = {}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        model = self.model.removeprefix("models/")
        data = await self._post(f"{self.base_url}/models/{model}:generateContent", payload, self._headers())
        usage = data.get("usageMetadata") or {}
        # Thinking tokens are billed; counting only candidates would undercount the usage of reasoning models
        completion = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
        self._record_usage(usage.get("promptTokenCount"), completion)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "无返回内容")
            raise TranslatorError(f"Gemini 未返回结果: {reason}")
        if candidates[0].get("finishReason") == "MAX_TOKENS":
            raise TruncatedError()
        parts = (candidates[0].get("content") or {}).get("parts", [])
        # Thinking models may include thought parts; take only the main text
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    async def list_models(self) -> list[str]:
        data = await self._get(f"{self.base_url}/models", self._headers(), {"pageSize": 1000})
        return [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]


class GoogleTranslator(Translator):
    """Free API of the Google web version; needs no Key and suits trial use. Heavy use may be rate-limited."""

    # translate_a/t accepts several q values at once, which cuts the number of requests and the chance of being
    # rate-limited
    max_batch_items = 50
    max_batch_chars = 4500
    concurrency = 3
    retries = 6

    def __init__(
        self, target_lang: str, source_lang: str = "auto", transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        super().__init__(target_lang, source_lang)
        # The plugin's Google language mapping: zh-HK->zh-TW, pt-br->pt, pt->pt-PT, fil->tl; languages not in the
        # table are not supported
        if target_lang not in GOOGLE_CODES:
            raise TranslatorError(f"谷歌翻译不支持目标语言「{prompts.lang_name(target_lang)}」，请换用 AI 翻译服务")
        if source_lang != "auto" and source_lang not in GOOGLE_CODES:
            raise TranslatorError(
                f"谷歌翻译不支持源语言「{prompts.lang_name(source_lang)}」，请改为自动检测或换用 AI 翻译服务"
            )
        self.tl = GOOGLE_CODES[target_lang]
        self.sl = "auto" if source_lang == "auto" else GOOGLE_CODES[source_lang]
        self.client = httpx.AsyncClient(timeout=60, transport=transport)

    @property
    def cache_key(self) -> str:
        return f"{self.service_id}:google:{self.source_lang}:{self.target_lang}"

    async def translate_batch(self, texts: list[str]) -> list[str]:
        # A mixed batch is split into two groups by the caller's flags: the rich-text group uses format=html to keep
        # inline formatting, and the plain-text group keeps format=text so that characters such as & are not
        # entity-escaped. Nothing is guessed from angle brackets.
        html_idx = [i for i, t in enumerate(texts) if self.is_html_item(t)]
        if not html_idx or len(html_idx) == len(texts):
            return await self._request(texts, html=bool(html_idx))
        plain_idx = [i for i in range(len(texts)) if i not in set(html_idx)]
        html_out, plain_out = await asyncio.gather(
            self._request([texts[i] for i in html_idx], html=True),
            self._request([texts[i] for i in plain_idx], html=False),
        )
        out = [""] * len(texts)
        for i, t in zip(html_idx, html_out):
            out[i] = t
        for i, t in zip(plain_idx, plain_out):
            out[i] = t
        return out

    async def _request(self, texts: list[str], html: bool) -> list[str]:
        params = {"client": "gtx", "sl": self.sl, "tl": self.tl, "format": "html" if html else "text"}
        for attempt in range(self.retries):
            last = attempt == self.retries - 1
            try:
                async with self._sem:
                    resp = await self.client.post(
                        "https://translate.googleapis.com/translate_a/t", params=params, data={"q": texts}
                    )
            except httpx.HTTPError as e:
                if last:
                    raise TranslatorError(f"Google 请求失败: {e}") from e
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                # When rate-limited, a 200 with an HTML captcha page occasionally comes back; a JSON parse failure is
                # treated as retryable rate limiting
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                if isinstance(data, list):
                    # One item per q: [translation, detected language] with sl=auto, otherwise the translation itself
                    out = [d[0] if isinstance(d, list) else d for d in data]
                    if len(out) != len(texts):
                        raise TranslatorError("Google 返回的译文条数与请求不一致")
                    return out
                if last:
                    raise TranslatorError("Google 返回的不是 JSON（可能已被限流，请稍后重试或改用 AI 翻译服务）")
                await asyncio.sleep(_retry_delay(resp, min(2 ** (attempt + 1), 30)))
                continue
            # A 302 redirect to the captcha page / 429 both mean rate limiting; retry with backoff
            if resp.status_code not in (302, 429, 500, 503) or last:
                raise TranslatorError(
                    f"Google 接口错误 {resp.status_code}（可能已被限流，请稍后重试或改用 AI 翻译服务）"
                )
            await asyncio.sleep(_retry_delay(resp, min(2 ** (attempt + 1), 30)))
        raise TranslatorError("已达到最大重试次数")

    async def aclose(self) -> None:
        await self.client.aclose()


class MockTranslator(Translator):
    """For tests: prefix the text with a marker."""

    async def translate_batch(self, texts: list[str]) -> list[str]:
        return [f"[{self.target_lang}] {t}" for t in texts]


LLM_CLASSES = {
    "openai": OpenAITranslator,
    "claude": ClaudeTranslator,
    "gemini": GeminiTranslator,
    "grok": GrokTranslator,
    "custom": CustomOpenAITranslator,
}


def create_translator(
    service: JSONDict,
    target_lang: str,
    source_lang: str = "auto",
    transport: httpx.AsyncBaseTransport | None = None,
) -> Translator:
    """Create an engine from a translation service configuration (see services.py)."""
    translator = _build_translator(service, target_lang, source_lang, transport)
    translator.service_id = service.get("id") or ""
    return translator


def _build_translator(
    service: JSONDict, target_lang: str, source_lang: str, transport: httpx.AsyncBaseTransport | None
) -> Translator:
    from .services import PROVIDERS

    provider = str(service.get("provider") or "")
    if provider == "google":
        return GoogleTranslator(target_lang, source_lang, transport=transport)
    if provider == "mock":
        return MockTranslator(target_lang, source_lang)
    cls = LLM_CLASSES.get(provider)
    if cls is None:
        raise TranslatorError(f"未知的服务类型: {provider}")
    meta = PROVIDERS[provider]
    api_key = (service.get("api_key") or "").strip()
    base_url = (service.get("base_url") or "").strip() or meta["base_url"]
    if not base_url:
        raise TranslatorError("未设置 API 地址")
    base_url = _with_default_path(base_url, DEFAULT_API_PATH[provider])
    if meta["needs_key"] and not api_key:
        raise TranslatorError(f"「{service.get('name') or meta['label']}」需要先填写 API Key")
    temperature = service.get("temperature")
    return cls(
        target_lang,
        source_lang,
        api_key=api_key,
        base_url=base_url,
        model=(service.get("model") or "").strip(),
        concurrency=service.get("concurrency") or 4,
        max_items=service.get("max_items") or 20,
        max_chars=service.get("max_chars") or 3000,
        temperature=None if temperature in (None, "") else float(temperature),
        prompt=service.get("prompt") or "",
        user_prompt=service.get("user_prompt") or "",
        api_format=service.get("api_format") or "",
        auth_style=service.get("auth_style") or "",
        transport=transport,
        glossary=service.get("glossary") or "",
        auto_glossary=service.get("auto_glossary", True) is not False,
    )
