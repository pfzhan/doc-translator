"""Translation engines. Each engine implements translate_batch(texts) -> list[str].

LLM engines (OpenAI / Grok / Gemini / Claude / OpenAI-compatible) share the same batch translation logic; the
prompts and the [[pN]] marker protocol are taken from Immersive Translate (see prompts.py). Paragraphs the model
leaves out are translated again one by one.
Providers differ only in request format, which subclasses implement in _complete.
"""

import asyncio
import hashlib
import logging
import re
import time
import unicodedata
import weakref
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

from . import glossary as glossary_mod
from . import prompts
from .deepl_free import DEEPL_FREE_CODES, MAX_REQUEST_UNITS, pack, split_paragraph, target_tag, utf16_len
from .json_types import JSONDict
from .languages import DEEPL_CODES, GOOGLE_CODES, LANGUAGES
from .lookup import WordEntry, parse_entry
from .request_log import RequestLog, RequestReason, RequestRecord
from .usage import Usage, nested, reported

log = logging.getLogger(__name__)

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


@dataclass
class _Service:
    """What the engines of one service share: its concurrency gate, the pause after it said it was overloaded and the
    times of its recent requests for the requests-per-minute limit."""

    # Semaphores by concurrency: changing the service's concurrency gives the jobs started after it a new gate
    gates: dict[int, asyncio.Semaphore] = field(default_factory=dict)
    # No request of the service is sent before this time (_clock) after a 429 or 503
    paused_until: float = 0.0
    # When (_clock) each request of the last _RATE_WINDOW seconds was sent, oldest first
    sent: deque[float] = field(default_factory=deque)


# Shared state of services by event loop, then by service id; see Translator._service. A semaphore belongs to the loop
# it is first used in, and the tests run each scenario in a loop of its own
_SERVICES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _Service]] = weakref.WeakKeyDictionary()

# The clock of service pauses; a name of its own so tests can move it without touching the event loop's clock
_clock = time.monotonic

# Statuses that mean the whole service is overloaded or rate-limited, not just this request
_OVERLOADED = {429, 503}

# The window of the requests-per-minute limit, in seconds
_RATE_WINDOW = 60.0


@dataclass
class Call:
    """What one request is for, besides its prompt; it goes into the request log with every attempt."""

    reason: RequestReason
    items: int  # paragraphs in the request
    chars: int  # characters of their source text

    @classmethod
    def of(cls, texts: list[str], reason: RequestReason) -> "Call":
        return cls(reason, len(texts), sum(len(t) for t in texts))


@dataclass
class _Attempt:
    """Timing of one attempt of a request."""

    number: int  # 1 for the first attempt
    started: float  # wall clock
    gate_wait: float
    duration: float = 0.0


class Translator:
    # Maximum paragraphs and characters per batch, used by Runner to split batches
    max_batch_items = 20
    max_batch_chars = 3000
    concurrency = 4
    # Requests per minute the service allows, counted over a sliding window shared by all its engines; None is no
    # limit
    rpm: int | None = None

    def __init__(self, target_lang: str, source_lang: str = "auto") -> None:
        if target_lang not in LANG_CODES or target_lang == "auto":
            raise TranslatorError(f"不支持的目标语言: {target_lang}")
        if source_lang not in LANG_CODES:
            raise TranslatorError(f"不支持的源语言: {source_lang}")
        self.target_lang = target_lang
        self.source_lang = source_lang
        # Document title, context for the engines that use it (LLMs fill it into {{title_prompt}}); set by Runner
        self.title = ""
        # When the source language is "自动检测", Runner fills this in after detecting the document language; used
        # for {{from}} in the prompt
        self.detected_source = ""
        # Translation service id, part of the cache key: a different service translates again, the same service
        # reuses the cache
        self.service_id = ""
        # Token usage the API reported, summed over every request (LLM engines add to it in _post); stays unreported
        # for engines without it, such as Google, and relays that leave it out
        self.usage = Usage()
        # Where every request sent upstream is recorded; set by the job that runs this engine (see request_log.py)
        self.request_log: RequestLog | None = None
        # Filled in by Runner from the caller's flags: these source texts are e-book rich-text fragments, which
        # cannot be guessed from angle brackets
        self.html_texts: set[str] = set()
        # Filled in by Runner from the notes the caller gives: source text -> note on where it appears in the
        # document (used only by LLM engines)
        self.notes: dict[str, str] = {}

    @property
    def _service(self) -> _Service:
        """What this engine shares with the other engines of its service.

        Engines made from the same service share one gate, so the service's concurrency limits all of its requests
        together: several jobs, web page requests and service tests. Each job and request makes its own engine, and
        a gate per engine let them add up to many times the limit and run into rate limits. They share the pause
        after a 429 or 503 for the same reason. An engine without a service id (tests) has a state of its own."""
        services = _SERVICES.setdefault(asyncio.get_running_loop(), {})
        key = self.service_id or f"engine:{id(self)}"
        shared = services.get(key)
        if shared is None:
            shared = services[key] = _Service()
            if not self.service_id:
                weakref.finalize(self, services.pop, key, None)
        return shared

    @property
    def _sem(self) -> asyncio.Semaphore:
        """The concurrency gate every request of this engine goes through, including the batch splitting and
        retranslation inside translate_batch, which would otherwise bypass Runner's worker limit."""
        gates = self._service.gates
        gate = gates.get(self.concurrency)
        if gate is None:
            gate = gates[self.concurrency] = asyncio.Semaphore(self.concurrency)
        return gate

    @asynccontextmanager
    async def _slot(self) -> AsyncIterator[float]:
        """Hold a place of the concurrency gate for one attempt, once the service's pause is over and its
        requests-per-minute limit lets it go; yield the seconds it took to get there.

        A place is held only while a request is on the wire: backoff between attempts happens outside, so a request
        waiting to retry does not keep others of the service from being sent. The pause and the rate limit are
        waited out with the place held, since every other request of the service waits for them too."""
        start = time.perf_counter()
        async with self._sem:
            await self._wait_turn()
            yield time.perf_counter() - start

    async def _wait_turn(self) -> None:
        """Wait out the service's pause, then for the requests-per-minute window to have room, and count this request
        in it. A pause that starts while waiting for room (another request got a 429) is waited out as well."""
        sent = self._service.sent
        while True:
            await self._wait_pause()
            if not self.rpm:
                return
            now = _clock()
            while sent and sent[0] <= now - _RATE_WINDOW:
                sent.popleft()
            if len(sent) < self.rpm:
                sent.append(now)
                return
            # The oldest request leaves the window at sent[0] + _RATE_WINDOW. The window holds more than this
            # engine's rpm only if another engine of the service has a higher one; the oldest still leaves first
            await asyncio.sleep(sent[0] + _RATE_WINDOW - now)

    async def _wait_pause(self) -> None:
        """Sleep until the service's pause is over. A pause extended while sleeping (another 429) is waited out too;
        a pause that did not change is not checked again, so a clock that stands still cannot loop."""
        shared = self._service
        slept_until = 0.0
        while (until := shared.paused_until) > slept_until:
            slept_until = until
            wait = until - _clock()
            if wait <= 0:
                break
            await asyncio.sleep(wait)

    def _pause(self, seconds: float) -> None:
        """Hold back every request of the service for seconds, after it answered that it is overloaded."""
        shared = self._service
        shared.paused_until = max(shared.paused_until, _clock() + seconds)

    @staticmethod
    def _start_attempt(number: int, gate_wait: float) -> _Attempt:
        return _Attempt(number=number, started=time.time(), gate_wait=gate_wait)

    def _log_request(
        self,
        call: Call,
        attempt: _Attempt,
        status: int | None,
        error: str = "",
        model: str = "",
        usage: Usage | None = None,
        stop: str = "",
    ) -> None:
        """Record one attempt: a debug line in the server log, and a line in the job's request log."""
        tokens = usage if usage is not None and usage.reported else None
        log.debug(
            "%s %s attempt %d: status %s%s, gate %.2fs, http %.2fs, %d items, %d chars, tokens %s, model %s",
            self.service_id or type(self).__name__,
            call.reason,
            attempt.number,
            status,
            f" ({error or stop})" if error or stop else "",
            attempt.gate_wait,
            attempt.duration,
            call.items,
            call.chars,
            f"{tokens.input}/{tokens.output}" if tokens else "-",
            model or "-",
        )
        if self.request_log is None:
            return
        self.request_log.add(
            RequestRecord(
                started=attempt.started,
                gate_wait=attempt.gate_wait,
                duration=attempt.duration,
                attempt=attempt.number,
                status=status,
                error=error,
                reason=call.reason,
                items=call.items,
                chars=call.chars,
                model=model,
                stop=stop,
                input_tokens=tokens.input if tokens else None,
                output_tokens=tokens.output if tokens else None,
                reasoning_tokens=tokens.reasoning if tokens else None,
                cached_tokens=tokens.cached_input if tokens else None,
                cache_write_tokens=tokens.cache_write if tokens else None,
            )
        )

    def is_html_item(self, text: str) -> bool:
        return text in self.html_texts

    async def _by_format(
        self, texts: list[str], request: Callable[[list[str], bool], Awaitable[list[str]]]
    ) -> list[str]:
        """Translate rich-text and plain-text paragraphs in separate requests, for machine translation engines that
        take a format switch, and put the results back in order. A mixed batch is split into two groups by the
        caller's flags; nothing is guessed from angle brackets."""
        html_idx = [i for i, t in enumerate(texts) if self.is_html_item(t)]
        if not html_idx or len(html_idx) == len(texts):
            return await request(texts, bool(html_idx))
        html_set = set(html_idx)
        plain_idx = [i for i in range(len(texts)) if i not in html_set]
        html_out, plain_out = await asyncio.gather(
            request([texts[i] for i in html_idx], True),
            request([texts[i] for i in plain_idx], False),
        )
        out = [""] * len(texts)
        for i, t in zip(html_idx, html_out):
            out[i] = t
        for i, t in zip(plain_idx, plain_out):
            out[i] = t
        return out

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


def _failure_kind(error: httpx.HTTPError | TimeoutError) -> str:
    """How a request without a response failed, for the request log: past its deadline (see LLMTranslator._post),
    an httpx timeout, or another network error."""
    if isinstance(error, TimeoutError):
        return "deadline"
    return "timeout" if isinstance(error, httpx.TimeoutException) else "network"


def _attempt_deadline(chars: int) -> float:
    """Seconds one attempt of a request with chars characters of source text may take. The model writes about as
    much as it reads, so time grows with the batch: 60 s plus 1 s per 25 characters (3000 characters, the default
    batch, get 180 s, the httpx timeout of one read), at least 90 s for slow first tokens and at most 300 s."""
    return min(300.0, max(90.0, 60 + chars / 25))


# All attempts of one request together may spend this many attempt deadlines in HTTP calls
_DEADLINE_BUDGET = 2.5
# An attempt is not started with less than this share of a deadline left; it would most likely run out
_DEADLINE_LEFTOVER = 0.1


def _retry_after(resp: httpx.Response) -> float | None:
    """Seconds the response asks to wait: retry-after-ms (OpenAI), or Retry-After as seconds or as an HTTP date.
    None when it asks for nothing usable."""
    try:
        ms = float(resp.headers.get("retry-after-ms") or 0)
    except ValueError:
        ms = 0
    if ms > 0:
        return ms / 1000
    value = (resp.headers.get("retry-after") or "").strip()
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        delay = (when - datetime.now(UTC)).total_seconds()
    return delay if delay > 0 else None


def _retry_delay(resp: httpx.Response, default: float) -> float:
    """Retry delay: what the response asks for (capped at 60 s), otherwise the given exponential backoff value."""
    delay = _retry_after(resp)
    return min(delay, 60) if delay else default


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


def _incomplete_reason(data: JSONDict) -> str:
    """Why a Responses API reply is incomplete (max_output_tokens, content_filter)."""
    details = data.get("incomplete_details")
    return str(details.get("reason") or "") if isinstance(details, dict) else ""


def _with_default_path(url: str, path: str) -> str:
    """Append the version path when the API address is only a domain (no path), as clients such as the OpenAI SDK
    and Cherry Studio do."""
    url = url.strip().rstrip("/")
    if url and not urlparse(url).path:
        url += path
    return url


# Reasoning settings a service can have: "" sends nothing (the model's default), "none" turns reasoning off where
# the model allows it, and the rest ask for more or less of it. Each engine maps them to its provider's parameter
REASONING_LEVELS = ("", "none", "minimal", "low", "medium", "high")
# A 400 whose message mentions one of these is taken as the API rejecting the reasoning setting
_REASONING_WORDS = ("reasoning", "thinking", "effort", "output_config")
# Claude's effort has no "minimal"; the lowest it has is "low"
_CLAUDE_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}
# Gemini's thinking budget in tokens; 0 turns thinking off on models that allow it
_GEMINI_THINKING_BUDGET = {"none": 0, "minimal": 512, "low": 1024, "medium": 8192, "high": 24576}
# Claude's max_tokens: what a translation needs, and more when the model thinks, since thinking counts against it
_CLAUDE_MAX_TOKENS = 8192
_CLAUDE_MAX_TOKENS_THINKING = 16000


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
        reasoning: str = "",
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
        if reasoning not in REASONING_LEVELS:
            raise TranslatorError(f"不支持的推理强度: {reasoning}")
        # How hard a reasoning model thinks (see REASONING_LEVELS); "" sends nothing and leaves it to the model
        self.reasoning = reasoning
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
        content = await self._complete(system, user, Call.of(terms, "terms"))
        wanted = {term.casefold(): term for term in terms}
        return [
            (wanted[source.casefold()], target)
            for source, target in glossary_mod.parse(prompts.clean_output(content))
            if source.casefold() in wanted
        ]

    async def lookup(self, text: str, context: str = "", alt_lang: str = "") -> WordEntry:
        """A dictionary entry for a word or short phrase selected on a web page; context is the sentence it was
        selected from. Written in target_lang, or in alt_lang when the word is itself in target_lang (as selected
        text is translated).

        The model writes the entry as JSON (see prompts.LOOKUP_SYSTEM_PROMPT). A reply that is not such an entry
        (an empty or declined one, prose, broken JSON) is asked for once more, then raised as an error."""
        alt_lang = alt_lang if alt_lang != self.target_lang else ""
        system, user = prompts.build_lookup_messages(
            text, context, self.target_lang, alt_lang=alt_lang, title=self.title
        )
        call = Call("lookup", 1, len(text) + len(context))
        for _ in range(2):
            entry = parse_entry(await self._complete(system, user, call), text)
            if entry is not None:
                return entry.written_in(self.target_lang, alt_lang)
            call = Call("repeat", 1, call.chars)
        raise TranslatorError("模型没有按格式给出释义")

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

    async def _complete(self, system: str, user: str, call: Call) -> str:
        raise NotImplementedError

    def _reported_usage(self, data: JSONDict) -> Usage:
        """Token usage of a reply, in the shape of Usage whatever the provider's field names."""
        raise NotImplementedError

    def _reported_model(self, data: JSONDict) -> str:
        """The model that answered, as the reply names it."""
        return str(data.get("model") or "")

    def _reported_stop(self, data: JSONDict) -> str:
        """Why the model stopped, as the reply says (end_turn, length, content_filter, ...), for the request log."""
        return ""

    def _declined(self, why: str) -> str:
        """The model or the provider's filter declined the request (a refusal, a content filter, a blocked prompt):
        the reply is not a translation, even when it carries some text.

        It is treated as an empty reply rather than an error. translate_batch asks once more, then translates a
        batch paragraph by paragraph, so the paragraph that set off the filter is found and the others are still
        translated; one paragraph that is declined again comes back empty, and Runner keeps its original without
        caching it. Raising an error would fail the whole job on one paragraph, and retrying the same request
        unchanged mostly gets the same answer."""
        log.info("%s declined a request (%s)", self.service_id or self.provider, why)
        return ""

    async def _post(self, url: str, build: Callable[[], JSONDict], headers: dict[str, str], call: Call) -> JSONDict:
        """POST the payload build() makes, with retries. Some reasoning models do not accept temperature, and some
        models or relays do not know the reasoning setting; on such a 400 it is dropped and the payload built and
        sent again.

        Relay services often wrap occasional upstream failures (account pool rotation, a channel going offline
        temporarily) as a 400, and the same request succeeds a few seconds later, so a 400 is treated as retryable
        too, except when it clearly comes from a configuration error on our side.

        Each attempt takes a place of the service's gate (see _slot) and gives it back before backing off. A 429 or
        503 pauses the whole service for the delay it asks for, so the other requests stop too instead of each
        running into the limit on its own; the retry then waits out that pause in the gate.

        httpx's timeout applies to each phase (connecting, each read), so an answer trickling in, or a relay keeping
        the connection open, could hold a worker for many minutes over the retries. Each attempt therefore has a
        deadline from the size of the batch (_attempt_deadline), and an attempt past it is retried like a network
        error; all attempts together get _DEADLINE_BUDGET times that, so one stuck batch cannot hold a worker for
        long. Time waiting for the gate does not count: it is spent queueing, not on this request.
        """
        payload = build()
        limit = _attempt_deadline(call.chars)
        budget = limit * _DEADLINE_BUDGET
        spent = 0.0
        for attempt in range(self.retries):
            if budget - spent <= limit * _DEADLINE_LEFTOVER:
                raise TranslatorError(f"请求超时：{attempt} 次尝试共 {spent:.0f} 秒仍未完成")
            last = attempt == self.retries - 1
            failure: httpx.HTTPError | TimeoutError | None = None
            async with self._slot() as waited:
                timing = self._start_attempt(attempt + 1, waited)
                clock = time.perf_counter()
                allowed = min(limit, budget - spent)
                try:
                    async with asyncio.timeout(allowed):
                        resp = await self.client.post(url, json=payload, headers=headers)
                except (httpx.HTTPError, TimeoutError) as e:
                    failure = e
                timing.duration = time.perf_counter() - clock
                # An attempt cut off by its deadline counts as the time it was allowed: the event loop wakes it a little
                # late, more so on a busy machine, and that delay is not time the request took
                spent += allowed if isinstance(failure, TimeoutError) else timing.duration
            if failure is not None:
                self._log_request(call, timing, None, _failure_kind(failure))
                if last:
                    if isinstance(failure, TimeoutError):
                        raise TranslatorError(f"请求超时：{self.retries} 次尝试共 {spent:.0f} 秒仍未完成") from failure
                    raise TranslatorError(f"请求失败: {failure}") from failure
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                try:
                    data = _json_or_error(resp, url)
                except TranslatorError:
                    self._log_request(call, timing, 200, "format")
                    raise
                usage = self._reported_usage(data)
                self.usage.add(usage)
                model, stop = self._reported_model(data), self._reported_stop(data)
                self._log_request(call, timing, 200, model=model, usage=usage, stop=stop)
                return data
            self._log_request(call, timing, resp.status_code, "http")
            delay = _retry_delay(resp, 2 ** (attempt + 1))
            if resp.status_code in _OVERLOADED:
                self._pause(delay)
            msg = _error_text(resp)
            # A parameter the model does not take is dropped for the rest of this engine's requests and the payload
            # built again: temperature first (a model thinking may refuse it, and say so mentioning thinking), then
            # the reasoning setting. The last attempt does not drop anything, so the real error is raised
            if resp.status_code == 400 and not last:
                lowered = msg.lower()
                if "temperature" in lowered and self.temperature is not None:
                    self.temperature = None
                    payload = build()
                    continue
                if self.reasoning and any(word in lowered for word in _REASONING_WORDS):
                    log.info("%s rejected the reasoning setting %r: %s", self.model, self.reasoning, msg)
                    self.reasoning = ""
                    payload = build()
                    continue
            # A 400 is retried unchanged only once (for occasional failures); if it fails again, translate_batch splits
            # the batch instead of waiting here in vain
            if _retryable(resp.status_code) and not last and not (resp.status_code == 400 and attempt >= 1):
                # After a 429/503 the next attempt waits out the service's pause in the gate
                if resp.status_code not in _OVERLOADED:
                    await asyncio.sleep(delay)
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
        return await self._translate(texts, "normal")

    async def _translate(self, texts: list[str], reason: RequestReason) -> list[str]:
        """translate_batch, told why the batch is sent; the reason goes into the request log."""
        system, user = self.build_messages(texts)
        try:
            # Every request, including those of the recursive batch splitting / retranslation below, goes through the
            # service's gate in _post, one attempt at a time
            content = await self._complete(system, user, Call.of(texts, reason))
            # An empty reply is mostly an occasional upstream failure; ask again unchanged. If it is still empty, a
            # batch is retranslated paragraph by paragraph below, and for a single paragraph Runner falls back to the
            # source text
            if not content.strip():
                content = await self._complete(system, user, Call.of(texts, "repeat"))
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
            left, right = await asyncio.gather(
                self._translate(texts[:mid], "split"), self._translate(texts[mid:], "split")
            )
            return left + right
        if len(texts) == 1:
            out = prompts.clean_output(content)
            # Ask again for a translation carrying foreign text; if it still does, accept it, and Runner drops and
            # retranslates it at the next re-layout
            if stray_chars(texts[0], out, self.target_lang):
                try:
                    again = prompts.clean_output(await self._complete(system, user, Call.of(texts, "repeat")))
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
            redone = await asyncio.gather(*(self._translate([texts[i]], "retranslate") for i in missing))
            for i, [text] in zip(missing, redone):
                found[i] = text
            # A reply to a single retranslation may also not be a translation ("请提供完整的原文"), so check again
            # against the typical length ratio of the batch after retranslation.
            # If asking once more is still wrong, return empty; Runner falls back to the source text and nothing is
            # cached
            typical = typical_ratio((texts[i], found[i]) for i in found)
            wrong = [i for i in missing if misplaced(texts[i], found[i], typical)]
            rechecked = await asyncio.gather(*(self._translate([texts[i]], "retranslate") for i in wrong))
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

    def _reported_usage(self, data: JSONDict) -> Usage:
        usage = data.get("usage")
        if self.api_format == "responses":
            return reported(
                nested(usage, "input_tokens"),
                nested(usage, "output_tokens"),
                reasoning=nested(usage, "output_tokens_details", "reasoning_tokens"),
                cached_input=nested(usage, "input_tokens_details", "cached_tokens"),
            )
        # Chat Completions; DeepSeek reports its cache hits as prompt_cache_hit_tokens instead
        cached = nested(usage, "prompt_tokens_details", "cached_tokens")
        return reported(
            nested(usage, "prompt_tokens"),
            nested(usage, "completion_tokens"),
            reasoning=nested(usage, "completion_tokens_details", "reasoning_tokens"),
            cached_input=cached if cached is not None else nested(usage, "prompt_cache_hit_tokens"),
        )

    def _reported_stop(self, data: JSONDict) -> str:
        if self.api_format == "responses":
            status = str(data.get("status") or "")
            return f"{status}: {_incomplete_reason(data)}" if status == "incomplete" else status
        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        return str(choice.get("finish_reason") or "") if isinstance(choice, dict) else ""

    async def _complete(self, system: str, user: str, call: Call) -> str:
        if self.api_format == "responses":
            return await self._complete_responses(system, user, call)

        def build() -> JSONDict:
            payload: JSONDict = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.reasoning:
                payload["reasoning_effort"] = self.reasoning
            return payload

        data = await self._post(f"{self.base_url}/chat/completions", build, self._headers(), call)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise TranslatorError(f"返回格式异常: {str(data)[:200]}") from e
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        if finish == "length":
            raise TruncatedError()
        # Relays name the normal stop differently ("stop", "eos", none at all), so only a filtered or refused reply
        # is turned down; content cut by the filter may be a partial translation
        if finish == "content_filter":
            return self._declined("content_filter")
        if isinstance(message, dict) and message.get("refusal"):
            return self._declined("refusal")
        # Relay services occasionally return a message without content; treat it as an empty reply, like an empty
        # reply from other APIs
        return (message.get("content") if isinstance(message, dict) else "") or ""

    async def _complete_responses(self, system: str, user: str, call: Call) -> str:
        """OpenAI Responses API (the default for Codex / Grok Build)."""

        def build() -> JSONDict:
            payload: JSONDict = {"model": self.model, "instructions": system, "input": user, "store": False}
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.reasoning:
                payload["reasoning"] = {"effort": self.reasoning}
            return payload

        data = await self._post(f"{self.base_url}/responses", build, self._headers(), call)
        if data.get("status") == "incomplete":
            reason = _incomplete_reason(data)
            if reason == "max_output_tokens":
                raise TruncatedError()
            # Stopped for another reason (content_filter): whatever text came is not a whole translation
            return self._declined(f"incomplete: {reason}")
        contents = [
            c
            for item in data.get("output") or []
            if isinstance(item, dict) and item.get("type") == "message"
            for c in item.get("content") or []
            if isinstance(c, dict)
        ]
        if any(c.get("type") == "refusal" for c in contents):
            return self._declined("refusal")
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text
        texts = [c.get("text", "") for c in contents if c.get("type") in ("output_text", "text")]
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

    def _reported_usage(self, data: JSONDict) -> Usage:
        # input_tokens leaves out the tokens read from and written to the prompt cache, which are billed too
        usage = data.get("usage")
        fresh = nested(usage, "input_tokens")
        cache_read = nested(usage, "cache_read_input_tokens")
        cache_write = nested(usage, "cache_creation_input_tokens")
        total = None if fresh is None else fresh + (cache_read or 0) + (cache_write or 0)
        return reported(total, nested(usage, "output_tokens"), cached_input=cache_read, cache_write=cache_write)

    def _reported_stop(self, data: JSONDict) -> str:
        return str(data.get("stop_reason") or "")

    async def _complete(self, system: str, user: str, call: Call) -> str:

        def build() -> JSONDict:
            thinking = self.reasoning not in ("", "none")
            payload: JSONDict = {
                "model": self.model,
                "max_tokens": _CLAUDE_MAX_TOKENS_THINKING if thinking else _CLAUDE_MAX_TOKENS,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            if thinking:
                # Adaptive thinking with an effort level; a thinking model takes no temperature
                payload["thinking"] = {"type": "adaptive"}
                payload["output_config"] = {"effort": _CLAUDE_EFFORT[self.reasoning]}
            else:
                if self.reasoning == "none":
                    payload["thinking"] = {"type": "disabled"}
                if self.temperature is not None:
                    payload["temperature"] = self.temperature
            return payload

        data = await self._post(f"{self.base_url}/messages", build, self._headers(), call)
        stop = data.get("stop_reason")
        # The reply ran out of room: output limit, or the context window with the prompt
        if stop in ("max_tokens", "model_context_window_exceeded"):
            raise TruncatedError()
        # Only a finished turn is a translation; a refusal (or a pause, a tool call) is not, whatever text it has.
        # Some relays leave stop_reason out
        if stop not in (None, "end_turn", "stop_sequence"):
            return self._declined(str(stop))
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    async def list_models(self) -> list[str]:
        data = await self._get(f"{self.base_url}/models", self._headers(), {"limit": 1000})
        return [m["id"] for m in data.get("data", [])]


class GeminiTranslator(LLMTranslator):
    """Google Gemini generateContent API."""

    provider = "gemini"

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    def _reported_usage(self, data: JSONDict) -> Usage:
        usage = data.get("usageMetadata")
        candidates = nested(usage, "candidatesTokenCount")
        thoughts = nested(usage, "thoughtsTokenCount")
        # Thinking tokens are billed; counting only candidates would undercount the usage of reasoning models
        output = None if candidates is None and thoughts is None else (candidates or 0) + (thoughts or 0)
        return reported(
            nested(usage, "promptTokenCount"),
            output,
            reasoning=thoughts,
            cached_input=nested(usage, "cachedContentTokenCount"),
        )

    def _reported_model(self, data: JSONDict) -> str:
        return str(data.get("modelVersion") or "")

    def _reported_stop(self, data: JSONDict) -> str:
        candidates = data.get("candidates")
        first = candidates[0] if isinstance(candidates, list) and candidates else None
        if isinstance(first, dict):
            return str(first.get("finishReason") or "")
        feedback = data.get("promptFeedback")
        return f"blocked: {feedback.get('blockReason')}" if isinstance(feedback, dict) else ""

    async def _complete(self, system: str, user: str, call: Call) -> str:

        def build() -> JSONDict:
            gen: JSONDict = {}
            if self.temperature is not None:
                gen["temperature"] = self.temperature
            if self.reasoning:
                gen["thinkingConfig"] = {"thinkingBudget": _GEMINI_THINKING_BUDGET[self.reasoning]}
            return {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen,
            }

        model = self.model.removeprefix("models/")
        data = await self._post(f"{self.base_url}/models/{model}:generateContent", build, self._headers(), call)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            # The prompt was blocked: like a refusal, another grouping of the paragraphs may pass
            if reason:
                return self._declined(f"blocked: {reason}")
            raise TranslatorError("Gemini 未返回结果: 无返回内容")
        finish = candidates[0].get("finishReason")
        if finish == "MAX_TOKENS":
            raise TruncatedError()
        # SAFETY, RECITATION, PROHIBITED_CONTENT and the like: stopped by a filter, not finished
        if finish not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
            return self._declined(str(finish))
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
        # The rich-text group uses format=html to keep inline formatting, and the plain-text group keeps format=text
        # so that characters such as & are not entity-escaped
        return await self._by_format(texts, self._request)

    async def _request(self, texts: list[str], html: bool) -> list[str]:
        params = {"client": "gtx", "sl": self.sl, "tl": self.tl, "format": "html" if html else "text"}
        call = Call.of(texts, "normal")
        for attempt in range(self.retries):
            last = attempt == self.retries - 1
            async with self._slot() as waited:
                timing = self._start_attempt(attempt + 1, waited)
                clock = time.perf_counter()
                failure: httpx.HTTPError | None = None
                try:
                    resp = await self.client.post(
                        "https://translate.googleapis.com/translate_a/t", params=params, data={"q": texts}
                    )
                except httpx.HTTPError as e:
                    failure = e
                timing.duration = time.perf_counter() - clock
            if failure is not None:
                self._log_request(call, timing, None, _failure_kind(failure))
                if last:
                    raise TranslatorError(f"Google 请求失败: {failure}") from failure
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                # When rate-limited, a 200 with an HTML captcha page occasionally comes back; a JSON parse failure is
                # treated as retryable rate limiting
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                self._log_request(call, timing, 200, "" if isinstance(data, list) else "format")
                if isinstance(data, list):
                    # One item per q: [translation, detected language] with sl=auto, otherwise the translation itself
                    out = [d[0] if isinstance(d, list) else d for d in data]
                    if len(out) != len(texts):
                        raise TranslatorError("Google 返回的译文条数与请求不一致")
                    return out
                # Rate limiting holds back every request to Google, like a 429 below; the retry waits in the gate
                self._pause(_retry_delay(resp, min(2 ** (attempt + 1), 30)))
                if last:
                    raise TranslatorError("Google 返回的不是 JSON（可能已被限流，请稍后重试或改用 AI 翻译服务）")
                continue
            self._log_request(call, timing, resp.status_code, "http")
            # A 302 redirect to the captcha page / 429 both mean rate limiting; retry with backoff
            delay = _retry_delay(resp, min(2 ** (attempt + 1), 30))
            if resp.status_code in (302, *_OVERLOADED):
                self._pause(delay)
            if resp.status_code not in (302, 429, 500, 503) or last:
                raise TranslatorError(
                    f"Google 接口错误 {resp.status_code}（可能已被限流，请稍后重试或改用 AI 翻译服务）"
                )
            if resp.status_code == 500:
                await asyncio.sleep(delay)
        raise TranslatorError("已达到最大重试次数")

    async def aclose(self) -> None:
        await self.client.aclose()


def deepl_base_url(base_url: str, api_key: str) -> str:
    """The /v2 address of the DeepL API. Left empty, it follows the key: a DeepL API Free key ends in ":fx" and only
    works on api-free.deepl.com, any other key on api.deepl.com. A domain alone gets /v2, and an address pasted with
    /translate at the end loses it."""
    url = base_url.strip().rstrip("/")
    if not url:
        return DEEPL_FREE_URL if api_key.strip().endswith(":fx") else DEEPL_PRO_URL
    url = url.removesuffix("/translate")
    return _with_default_path(url, "/v2")


DEEPL_FREE_URL = "https://api-free.deepl.com/v2"
DEEPL_PRO_URL = "https://api.deepl.com/v2"


class DeepLTranslator(Translator):
    """The official DeepL API (Free or Pro), which needs an authentication key. Machine translation like Google:
    no prompt, glossary or token usage; the document title goes in as context, which DeepL does not bill."""

    retries = 5
    # Statuses that hold back every request to the service: DeepL's rate limit (429) and overload (503)
    _overloaded: set[int] = _OVERLOADED
    # DeepL takes up to 50 texts in one request, within a 128 KiB body: 30000 characters stay under it even when
    # every character takes three bytes, as Chinese does, with room for JSON escapes
    _MAX_ITEMS = 50
    _MAX_CHARS = 30000

    def __init__(
        self,
        target_lang: str,
        source_lang: str = "auto",
        *,
        api_key: str,
        base_url: str = "",
        concurrency: int = 2,
        max_items: int = 50,
        max_chars: int = 5000,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(target_lang, source_lang)
        if target_lang not in DEEPL_CODES:
            raise TranslatorError(f"DeepL 不支持目标语言「{prompts.lang_name(target_lang)}」，请换用 AI 翻译服务")
        if source_lang != "auto" and source_lang not in DEEPL_CODES:
            raise TranslatorError(
                f"DeepL 不支持源语言「{prompts.lang_name(source_lang)}」，请改为自动检测或换用 AI 翻译服务"
            )
        self.target_code = DEEPL_CODES[target_lang][1]
        # Left out of the request for automatic detection
        self.source_code = "" if source_lang == "auto" else DEEPL_CODES[source_lang][0]
        self.api_key = api_key
        self.url = deepl_base_url(base_url, api_key) + "/translate"
        self.concurrency = concurrency
        self.max_batch_items = max(1, min(max_items, self._MAX_ITEMS))
        self.max_batch_chars = max(1, min(max_chars, self._MAX_CHARS))
        self.client = httpx.AsyncClient(timeout=60, transport=transport)

    @property
    def cache_key(self) -> str:
        return f"{self.service_id}:deepl:{self.source_lang}:{self.target_lang}"

    async def translate_batch(self, texts: list[str]) -> list[str]:
        # tag_handling=html for rich text keeps its inline tags in place; plain text is sent as text, so that a
        # stray < or & in it is not taken for markup
        return await self._by_format(texts, self._request)

    def _body(self, texts: list[str], html: bool) -> JSONDict:
        body: JSONDict = {"text": texts, "target_lang": self.target_code}
        if self.source_code:
            body["source_lang"] = self.source_code
        if html:
            body["tag_handling"] = "html"
        if self.title:
            body["context"] = self.title
        return body

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"DeepL-Auth-Key {self.api_key}"}

    async def _request(self, texts: list[str], html: bool) -> list[str]:
        body = self._body(texts, html)
        headers = self._headers()
        call = Call.of(texts, "normal")
        for attempt in range(self.retries):
            last = attempt == self.retries - 1
            async with self._slot() as waited:
                timing = self._start_attempt(attempt + 1, waited)
                clock = time.perf_counter()
                failure: httpx.HTTPError | None = None
                try:
                    resp = await self.client.post(self.url, json=body, headers=headers)
                except httpx.HTTPError as e:
                    failure = e
                timing.duration = time.perf_counter() - clock
            if failure is not None:
                self._log_request(call, timing, None, _failure_kind(failure))
                if last:
                    raise TranslatorError(f"DeepL 请求失败: {failure}") from failure
                await asyncio.sleep(2**attempt)
                continue
            status = resp.status_code
            if status == 200:
                out = self._translations(resp, texts)
                self._log_request(call, timing, 200, "" if out is not None else "format")
                if out is None:
                    raise TranslatorError(f"DeepL 返回格式异常: {resp.text[:200]}")
                return out
            self._log_request(call, timing, status, "http")
            # 429 is DeepL's rate limit and 503 an overload: both hold back every request to it. Other 5xx are
            # passing failures of this request; DeepL's documentation asks to retry both kinds with backoff
            delay = _retry_delay(resp, min(2 ** (attempt + 1), 30))
            if status in self._overloaded:
                self._pause(delay)
            elif status >= 500:
                if not last:
                    await asyncio.sleep(delay)
            else:
                raise TranslatorError(self._error(resp), status=status)
            if last:
                raise TranslatorError(f"DeepL 接口错误 {status}: {_error_text(resp)}", status=status)
        raise TranslatorError("已达到最大重试次数")

    def _error(self, resp: httpx.Response) -> str:
        """A readable message for an error that retrying does not fix."""
        return _deepl_error(resp)

    def _translations(self, resp: httpx.Response, texts: list[str]) -> list[str] | None:
        """The translations of a 200 response, in request order; None when it is not what DeepL answers."""
        count = len(texts)
        try:
            data = resp.json()
        except ValueError:
            return None
        items = data.get("translations") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return None
        out = [item.get("text") if isinstance(item, dict) else None for item in items]
        if not all(isinstance(t, str) for t in out):
            return None
        if len(out) != count:
            raise TranslatorError("DeepL 返回的译文条数与请求不一致")
        return [t for t in out if isinstance(t, str)]

    async def aclose(self) -> None:
        await self.client.aclose()


def _deepl_error(resp: httpx.Response) -> str:
    """A readable message for a DeepL error that retrying does not fix."""
    detail = _error_text(resp)
    match resp.status_code:
        case 401 | 403:
            return (
                "DeepL API Key 无效，或与 API 地址不匹配（免费版 Key 以 :fx 结尾，只能用 api-free.deepl.com）"
                f": {detail}"
            )
        case 456:
            return "DeepL 本期的字符额度已用完，请等下个计费周期或升级套餐"
        case 413:
            return "请求过大，请在高级设置中调小每批最多字符数"
        case _:
            return f"DeepL 接口错误 {resp.status_code}: {detail}"


DEEPL_FREE_URL_WEB = "https://oneshot-free.www.deepl.com/v1/translate"
# A request answered with every paragraph unchanged, although DeepL found them in another language, is taken for
# the endpoint failing silently (see app/deepl_free.py) once it holds this many letters; below that it may be
# names and terms that stay as they are
_UNCHANGED_LETTERS = 200


class DeepLFreeTranslator(DeepLTranslator):
    """DeepL's free web endpoint (oneshot-free), which needs no key. It is not part of DeepL's public API: DeepL may
    rate-limit it, change it or take it away. A request carries at most 1500 UTF-16 code units, so paragraphs are
    packed into requests by that limit and longer ones are cut after a sentence (see app/deepl_free.py)."""

    retries = 6
    # 529 is the endpoint's "too many requests", as 429
    _overloaded = {*_OVERLOADED, 529}

    def __init__(
        self, target_lang: str, source_lang: str = "auto", transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        if target_lang in LANG_CODES and target_lang not in DEEPL_FREE_CODES:
            raise TranslatorError(
                f"DeepL（免费）不支持目标语言「{prompts.lang_name(target_lang)}」，请换用其他翻译服务"
            )
        if source_lang in LANG_CODES and source_lang != "auto" and source_lang not in DEEPL_FREE_CODES:
            raise TranslatorError(
                f"DeepL（免费）不支持源语言「{prompts.lang_name(source_lang)}」，请改为自动检测或换用其他翻译服务"
            )
        super().__init__(
            target_lang,
            source_lang,
            api_key="",
            concurrency=2,
            max_items=50,
            max_chars=MAX_REQUEST_UNITS,
            transport=transport,
        )
        self.url = DEEPL_FREE_URL_WEB
        self.target_code = target_tag(DEEPL_CODES[target_lang][1])
        # Spaces between the pieces of a cut paragraph are dropped in languages that do not put them between sentences
        self._joins_without_space = self.target_code.split("-")[0] in ("zh", "ja")

    @property
    def cache_key(self) -> str:
        return f"{self.service_id}:deepl-free:{self.source_lang}:{self.target_lang}"

    async def translate_batch(self, texts: list[str]) -> list[str]:
        return await self._by_format(texts, self._request_pieces)

    async def _request_pieces(self, texts: list[str], html: bool) -> list[str]:
        """Translate paragraphs of one format in as many requests as the length limit needs."""
        pieces = [(owner, piece, gap) for owner, text in enumerate(texts) for piece, gap in split_paragraph(text, html)]
        # Blank pieces are not sent; they stay as they are
        sent = [k for k, (_, piece, _) in enumerate(pieces) if piece.strip()]
        groups = pack([utf16_len(pieces[k][1]) for k in sent], max_items=self.max_batch_items)
        results = await asyncio.gather(*(self._request([pieces[sent[j]][1] for j in g], html) for g in groups))
        translated = [piece for _, piece, _ in pieces]
        for group, result in zip(groups, results):
            for j, text in zip(group, result):
                translated[sent[j]] = text
        out = [""] * len(texts)
        for (owner, _, gap), text in zip(pieces, translated):
            if gap and self._joins_without_space and "\n" not in gap:
                gap = ""
            out[owner] += text + gap
        return out

    def _headers(self) -> dict[str, str]:
        return {}

    def _body(self, texts: list[str], html: bool) -> JSONDict:
        body = super()._body(texts, html)
        # The endpoint takes context as an object and rejects the API's string with a 400
        body.pop("context", None)
        return body

    def _error(self, resp: httpx.Response) -> str:
        detail = _error_text(resp)
        match resp.status_code:
            case 401 | 403:
                return f"DeepL 免费接口拒绝了请求（可能限制了本机的访问），请稍后再试或换用其他翻译服务: {detail}"
            case 404:
                return "DeepL 免费接口已不可用（404），请换用其他翻译服务"
            case 456:
                return "DeepL 免费接口的额度已用完，请稍后再试或换用其他翻译服务"
            case _:
                return f"DeepL 免费接口错误 {resp.status_code}: {detail}"

    def _translations(self, resp: httpx.Response, texts: list[str]) -> list[str] | None:
        out = super()._translations(resp, texts)
        if out is not None and out == texts and self._foreign(resp):
            letters = sum(len(re.findall(r"[^\W\d_]", t)) for t in texts)
            if letters >= _UNCHANGED_LETTERS:
                raise TranslatorError("DeepL 免费接口原样返回了原文，没有翻译（接口可能已变更），请换用其他翻译服务")
        return out

    def _foreign(self, resp: httpx.Response) -> bool:
        """Whether DeepL found any paragraph of the response in a language other than the target."""
        target = self.target_code.split("-")[0]
        data = resp.json()
        items = data.get("translations") if isinstance(data, dict) else None
        detected = [item.get("detected_source_language") for item in items or [] if isinstance(item, dict)]
        return any(isinstance(d, str) and d.split("-")[0].lower() != target for d in detected)


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
    rpm = service.get("rpm")
    translator.rpm = rpm if isinstance(rpm, int) and not isinstance(rpm, bool) and rpm > 0 else None
    return translator


def _build_translator(
    service: JSONDict, target_lang: str, source_lang: str, transport: httpx.AsyncBaseTransport | None
) -> Translator:
    from .services import PROVIDERS

    provider = str(service.get("provider") or "")
    if provider == "google":
        return GoogleTranslator(target_lang, source_lang, transport=transport)
    if provider == "deepl-free":
        return DeepLFreeTranslator(target_lang, source_lang, transport=transport)
    if provider == "mock":
        return MockTranslator(target_lang, source_lang)
    if provider == "deepl":
        key = (service.get("api_key") or "").strip()
        if not key:
            raise TranslatorError(f"「{service.get('name') or 'DeepL'}」需要先填写 API Key")
        return DeepLTranslator(
            target_lang,
            source_lang,
            api_key=key,
            base_url=service.get("base_url") or "",
            concurrency=service.get("concurrency") or 2,
            max_items=service.get("max_items") or 50,
            max_chars=service.get("max_chars") or 5000,
            transport=transport,
        )
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
        reasoning=str(service.get("reasoning") or ""),
        prompt=service.get("prompt") or "",
        user_prompt=service.get("user_prompt") or "",
        api_format=service.get("api_format") or "",
        auth_style=service.get("auth_style") or "",
        transport=transport,
        glossary=service.get("glossary") or "",
        auto_glossary=service.get("auto_glossary", True) is not False,
    )
