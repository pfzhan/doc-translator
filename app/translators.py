"""翻译引擎。每个引擎实现 translate_batch(texts) -> list[str]。

大模型引擎（OpenAI / Grok / Gemini / Claude / OpenAI 兼容）共用同一套批量翻译逻辑，
提示词和 [[pN]] 标记协议取自沉浸式翻译（见 prompts.py）。模型漏掉的段落会单独补翻。
各家只在请求格式上不同，由子类实现 _complete。
"""
import asyncio
import hashlib
import re
import unicodedata
from urllib.parse import urlparse

import httpx

from . import glossary as glossary_mod
from . import prompts
from .languages import GOOGLE_CODES, LANGUAGES

# 前端下拉框用的语言列表：[(代码, 英文名, 本地名)]，顺序与插件一致
LANG_CODES = {code for code, _, _ in LANGUAGES}


class TranslatorError(Exception):
    def __init__(self, message: str = "", status: int | None = None):
        super().__init__(message)
        # 接口返回的 HTTP 状态码；网络错误、解析错误等没有状态码时为 None
        self.status = status


class Translator:
    # 单批最多多少段、多少字符，由 Runner 用来切批
    max_batch_items = 20
    max_batch_chars = 3000
    concurrency = 4

    def __init__(self, target_lang: str, source_lang: str = "auto"):
        if target_lang not in LANG_CODES or target_lang == "auto":
            raise TranslatorError(f"不支持的目标语言: {target_lang}")
        if source_lang not in LANG_CODES:
            raise TranslatorError(f"不支持的源语言: {source_lang}")
        self.target_lang = target_lang
        self.source_lang = source_lang
        # 源语言选“自动检测”时，由 Runner 检测文档语言后填进来，用于提示词的 {{from}}
        self.detected_source = ""
        # 翻译服务 id，算进缓存 key：换了服务就重新翻译，同一个服务重用缓存
        self.service_id = ""
        # 接口返回的 token 用量（大模型引擎在 _complete 里累计）；谷歌等不提供的保持 0
        self.usage = {"prompt": 0, "completion": 0}
        # 由 Runner 按调用方标记填入：这些原文是电子书富文本片段，不能从尖括号猜
        self.html_texts: set[str] = set()
        # 由 Runner 按调用方给的说明填入：原文 → 它在文档里所处位置的说明（只有大模型引擎用得上）
        self.notes: dict[str, str] = {}

    def is_html_item(self, text: str) -> bool:
        return text in self.html_texts

    def _record_usage(self, prompt: int, completion: int):
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

    async def aclose(self):
        pass


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300]
    # 中转站的错误响应不一定是 JSON 对象，可能是数组或字符串
    if not isinstance(data, dict):
        return str(data)[:300]
    err = data.get("error", data)
    msg = err.get("message") if isinstance(err, dict) else err
    return str(msg or resp.text)[:300]


# 年份、页码、编号这类三位以上的数字，翻译前后应该原样都在
_NUMBER_RE = re.compile(r"\d{3,}")


def _numbers_mismatch(source: str, translation: str) -> bool:
    """原文的数字译文里一个都没有，却有别的数字：多半是模型把两段并成一段，后面的译文整体错了位。"""
    src = set(_NUMBER_RE.findall(source))
    dst = set(_NUMBER_RE.findall(translation))
    return bool(src) and bool(dst) and not (src & dst)


# 译文和原文的长度比偏离常见值这么多倍，多半是模型把相邻几段串了位，或者译到一半断了。
# 实测英译中常见值约 0.35，正常译文最低约 0.58 倍，原样照抄的最高约 3 倍
_RATIO_LOW = 0.45
_RATIO_HIGH = 4.0
# 原文去掉标记后至少这么长才比长度，短句的比例本来就散
_RATIO_MIN_CHARS = 60
# 算常见值至少要这么多段
_RATIO_SAMPLES = 3


def _length_ratio(source: str, translation: str) -> float | None:
    length = len(prompts.without_marks(source).strip())
    if length < _RATIO_MIN_CHARS:
        return None
    return len(prompts.without_marks(translation).strip()) / length


def typical_ratio(pairs) -> float | None:
    """一批 (原文, 译文) 的长度比中位数。段数太少时不算。"""
    ratios = sorted(r for source, translation in pairs if (r := _length_ratio(source, translation)) is not None)
    if len(ratios) < _RATIO_SAMPLES:
        return None
    return ratios[len(ratios) // 2]


def misplaced(source: str, translation: str, typical: float | None) -> bool:
    """译文多半不是这一段的：带着原文没有的公式占位符，或者长度比远离同批的常见值。"""
    if prompts.placeholders(translation) - prompts.placeholders(source):
        return True
    ratio = _length_ratio(source, translation)
    return bool(typical) and ratio is not None and not typical * _RATIO_LOW <= ratio <= typical * _RATIO_HIGH


# 判断夹带外文用的文字系统，取字符 Unicode 名称的第一个词。拉丁字母、希腊字母不在其中：
# 代码、人名、公式符号（μ律、λ）在任何语言的译文里都可能出现
_SCRIPTS = frozenset({
    "CYRILLIC", "ARMENIAN", "HEBREW", "ARABIC", "SYRIAC", "THAANA", "DEVANAGARI", "BENGALI",
    "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "THAI",
    "LAO", "TIBETAN", "MYANMAR", "GEORGIAN", "HANGUL", "ETHIOPIC", "KHMER", "MONGOLIAN",
    "HIRAGANA", "KATAKANA", "CJK",
})
# 本地名里看不出的文字系统：日文的假名
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
    """杂项符号、装饰符号、表情和私用区字符。译文里的这类字符应当都来自原文。"""
    return {ch for ch in text if 0x2600 <= ord(ch) <= 0x27BF or ord(ch) >= 0x1F000
            or unicodedata.category(ch) == "Co"}


def stray_chars(source: str, translation: str, target_lang: str) -> bool:
    """译文里夹着原文和目标语言都不用的文字，例如中文译文里冒出半个俄文词「убед自己」，
    或者原文没有的符号（标题后面多出一串「♀♀♀」）。目标语言用的文字从它的本地名（简体中文、Русский）里看。"""
    if _symbols(translation) - _symbols(source):
        return True
    allowed = _scripts(source) | _scripts(_NATIVE_NAMES.get(target_lang, ""))
    allowed |= _EXTRA_SCRIPTS.get(target_lang, set())
    return bool(_scripts(translation) - allowed)


# 这些状态码说明是我们这边的配置有问题（Key 错、没权限、地址错），重试也不会好，立即报错
FATAL_STATUS = {401, 403, 404}


def _retryable(status: int) -> bool:
    return status not in FATAL_STATUS


def _retry_delay(resp: httpx.Response, default: float) -> float:
    """重试间隔：响应头带 Retry-After 时优先使用（上限 60s），否则用传入的指数退避值。"""
    try:
        delay = float(resp.headers.get("retry-after") or 0)
    except ValueError:
        delay = 0
    return min(delay, 60) if delay > 0 else default


def _json_or_error(resp: httpx.Response, url: str) -> dict:
    """解析 JSON 响应。返回的是网页（常见于 API 地址少了 /v1，请求落到了中转站首页）时给出可读的错误。"""
    try:
        data = resp.json()
    except ValueError:
        ctype = resp.headers.get("content-type", "")
        hint = "，请检查 API 地址是否完整（通常以 /v1 结尾）" if "html" in ctype or resp.text.lstrip().startswith("<") else ""
        raise TranslatorError(f"接口返回的不是 JSON（{ctype or '未知类型'}）：{url}{hint}") from None
    if not isinstance(data, dict):
        raise TranslatorError(f"返回格式异常: {str(data)[:200]}")
    return data


def _with_default_path(url: str, path: str) -> str:
    """API 地址只填了域名（没有路径）时补上版本路径，和 OpenAI SDK、Cherry Studio 等客户端的习惯一致。"""
    url = url.strip().rstrip("/")
    if url and not urlparse(url).path:
        url += path
    return url


# 只填了域名时补的版本路径：Claude 和 OpenAI 系是 /v1，Gemini 是 /v1beta
DEFAULT_API_PATH = {"openai": "/v1", "grok": "/v1", "custom": "/v1", "claude": "/v1", "gemini": "/v1beta"}


class LLMTranslator(Translator):
    provider = ""
    retries = 4

    def __init__(self, target_lang, source_lang="auto", *, api_key="", base_url="", model="",
                 concurrency=4, max_items=20, max_chars=3000, temperature=0.0, prompt="", user_prompt="",
                 api_format="", auth_style="", transport=None, glossary="", auto_glossary=True):
        super().__init__(target_lang, source_lang)
        if not model:
            raise TranslatorError("未设置模型")
        # api_format：OpenAI 系可选 "responses"（Responses API），默认 Chat Completions
        # auth_style：Claude 可选 "bearer"（Authorization: Bearer，中转站常用），默认 x-api-key
        self.api_format = api_format
        self.auth_style = auth_style
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = max(1, int(concurrency))
        # 并发上限：translate_batch 里拆批 / 补翻的 gather 也要过这个信号量，否则会绕过 Runner 的 worker 限制
        self._sem = asyncio.Semaphore(self.concurrency)
        self.max_batch_items = max(1, int(max_items))
        self.max_batch_chars = max(200, int(max_chars))
        self.temperature = temperature
        self.prompt = prompt.strip()
        self.user_prompt = user_prompt.strip()
        self.glossary = glossary
        # 翻译前从文档里找术语、定译名（Runner 调 define_terms，结果放进 doc_glossary）
        self.auto_glossary = auto_glossary
        self.doc_glossary: list[glossary_mod.Entry] = []
        # 文档标题，作为上下文填进提示词的 {{title_prompt}}
        self.title = ""
        self.client = httpx.AsyncClient(timeout=180, transport=transport)

    @property
    def cache_key(self):
        # 模型、提示词、术语表进缓存键：换了就自动整本重翻，换回来仍命中该配置自己的缓存。
        # 源语言影响提示词（{{from}}、wyw2zh-CN 这类覆盖），用检测后的实际源语言
        p = hashlib.sha1(f"{self.prompt}\0{self.user_prompt}\0{self.glossary}".encode()).hexdigest()[:8]
        key = (f"{self.service_id}:{self.provider}:{self.model}:{p}:"
               f"{self.effective_source}:{self.target_lang}")
        # 文档术语表也影响译文。没有时键不变，以前的缓存照样命中
        if self.doc_glossary:
            key += ":g" + hashlib.sha1(glossary_mod.render(self.doc_glossary).encode()).hexdigest()[:8]
        return key

    def glossary_for(self, texts: list[str]) -> str:
        """这一批用的术语表：用户填的加文档里找到的，只留原文里出现了的条目。

        用户填的行没写成“原文 = 译文”的，看不出跟哪段有关，每批都带上。
        """
        lines = [line.strip() for line in self.glossary.splitlines() if line.strip()]
        loose = [line for line in lines if not glossary_mod.parse(line)]
        entries = glossary_mod.merge(glossary_mod.parse(self.glossary), self.doc_glossary)
        return "\n".join([*loose, glossary_mod.render(glossary_mod.relevant(entries, texts))]).strip()

    async def define_terms(self, terms: list[str], excerpt: str = "") -> list[glossary_mod.Entry]:
        """一次请求定下候选术语的译名。模型删掉的、改了写法的候选不收。"""
        system, user = prompts.build_terms_messages(
            terms, self.target_lang, source_lang=self.effective_source, title=self.title, excerpt=excerpt,
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
            texts, self.target_lang, source_lang=self.effective_source, title=self.title,
            system_template=self.prompt, user_template=self.user_prompt,
            contains_html=any(self.is_html_item(t) for t in texts),
            glossary=self.glossary_for(texts), notes=[self.notes.get(t, "") for t in texts],
        )

    async def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    async def _post(self, url: str, payload: dict, headers: dict) -> dict:
        """带重试的 POST。部分推理模型不接受 temperature，遇到这种 400 会去掉后重发。

        中转站经常把上游的偶发故障（号池轮换、渠道临时下线）原样包装成 400 返回，
        同样的请求过几秒就能成功，所以除了明确是我们这边配置错误的情况，400 也按可重试处理。
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
            # 最后一次尝试不再剥离 temperature 重发，让真实错误抛出来
            if (resp.status_code == 400 and "temperature" in msg.lower()
                    and self.temperature is not None and not last):
                self.temperature = None
                payload = {k: v for k, v in payload.items() if k != "temperature"}
                gen = payload.get("generationConfig")
                if isinstance(gen, dict):
                    payload["generationConfig"] = {k: v for k, v in gen.items() if k != "temperature"}
                continue
            # 400 只原样重试一次（应付偶发故障）；再失败就交给 translate_batch 拆批，不在这里白等
            if _retryable(resp.status_code) and not last and not (resp.status_code == 400 and attempt >= 1):
                await asyncio.sleep(_retry_delay(resp, 2 ** (attempt + 1)))
                continue
            raise TranslatorError(f"接口错误 {resp.status_code}: {msg}", status=resp.status_code)
        raise TranslatorError("已达到最大重试次数")

    async def _get(self, url: str, headers: dict, params: dict | None = None) -> dict:
        try:
            resp = await self.client.get(url, headers=headers, params=params)
        except httpx.HTTPError as e:
            raise TranslatorError(f"请求失败: {e}") from e
        if resp.status_code != 200:
            raise TranslatorError(f"接口错误 {resp.status_code}: {_error_text(resp)}")
        return _json_or_error(resp, url)

    async def list_models(self) -> list[str]:
        raise NotImplementedError

    async def translate_batch(self, texts):
        system, user = self.build_messages(texts)
        try:
            # 实际发请求的唯一入口，递归拆批 / 补翻最终也回到这里，都受同一个信号量限制
            async with self._sem:
                content = await self._complete(system, user)
                # 回空多是上游偶发故障，原样再问一次。仍是空的，批量时下面逐段补翻，单段时 Runner 回填原文
                if not content.strip():
                    content = await self._complete(system, user)
        except TranslatorError as e:
            # 中转站会稳定拒绝某些段落组合（实测同一批 8 段每次都 400，拆开后每一部分都能翻），
            # 原样重试没用，拆成两半各自翻。只对 400 这样做：配置错误（401/403/404）拆了也没用，
            # 429、5xx 已经在 _post 里退避重试过了，再拆只会放大请求量
            if len(texts) == 1 or e.status != 400:
                raise
            mid = len(texts) // 2
            # 两半并行翻：串行的话批次越拆越深，最后几段会排很久的队
            left, right = await asyncio.gather(self.translate_batch(texts[:mid]), self.translate_batch(texts[mid:]))
            return left + right
        if len(texts) == 1:
            out = prompts.clean_output(content)
            # 夹带外文的译文再问一次，仍然夹带就照收，由 Runner 下次重排时丢掉重翻
            if stray_chars(texts[0], out, self.target_lang):
                async with self._sem:
                    again = prompts.clean_output(await self._complete(system, user))
                if again.strip() and not stray_chars(texts[0], again, self.target_lang):
                    out = again
            return [out]
        found = prompts.parse_markers(content, len(texts), marker=prompts.pick_marker(texts))
        # 模型漏掉的段落补翻（和插件的 recovery 思路一样，只补缺失的），并行发出。
        # 数字、公式占位符、长度对不上的也单独重翻：标记数量对得上，内容却可能错了位。夹带外文的同样重翻
        typical = typical_ratio((texts[i], found[i]) for i in found)
        missing = [i for i in range(len(texts))
                   if i not in found or _numbers_mismatch(texts[i], found[i])
                   or misplaced(texts[i], found[i], typical)
                   or stray_chars(texts[i], found[i], self.target_lang)]
        if missing:
            redone = await asyncio.gather(*(self.translate_batch([texts[i]]) for i in missing))
            for i, [text] in zip(missing, redone):
                found[i] = text
        return [found[i] for i in range(len(texts))]

    async def aclose(self):
        await self.client.aclose()


class OpenAITranslator(LLMTranslator):
    """OpenAI Chat Completions 格式。Grok 和各种 OpenAI 兼容服务也走这里。"""

    provider = "openai"

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _complete(self, system, user):
        if self.api_format == "responses":
            return await self._complete_responses(system, user)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = await self._post(f"{self.base_url}/chat/completions", payload, self._headers())
        usage = data.get("usage") or {}
        self._record_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise TranslatorError(f"返回格式异常: {str(data)[:200]}") from e
        # 中转站偶尔回一条没有 content 的消息，和其他接口回空一样当空回复
        return (message.get("content") if isinstance(message, dict) else "") or ""

    async def _complete_responses(self, system, user):
        """OpenAI Responses API（Codex / Grok Build 默认用这个）。"""
        payload = {"model": self.model, "instructions": system, "input": user, "store": False}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = await self._post(f"{self.base_url}/responses", payload, self._headers())
        usage = data.get("usage") or {}
        self._record_usage(usage.get("input_tokens"), usage.get("output_tokens"))
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        texts = [
            c.get("text", "")
            for item in data.get("output") or []
            if item.get("type") == "message"
            for c in item.get("content") or []
            if c.get("type") in ("output_text", "text")
        ]
        if not texts and data.get("status") not in (None, "completed"):
            raise TranslatorError(f"返回状态异常: {data.get('status')} {str(data.get('incomplete_details') or '')[:200]}")
        return "".join(texts)

    async def list_models(self):
        data = await self._get(f"{self.base_url}/models", self._headers())
        return sorted(m["id"] for m in data.get("data", []))


class GrokTranslator(OpenAITranslator):
    provider = "grok"


class CustomOpenAITranslator(OpenAITranslator):
    provider = "custom"


class ClaudeTranslator(LLMTranslator):
    """Anthropic Messages API。"""

    provider = "claude"
    api_version = "2023-06-01"

    def _headers(self):
        if self.auth_style == "bearer":
            return {"Authorization": f"Bearer {self.api_key}", "anthropic-version": self.api_version}
        return {"x-api-key": self.api_key, "anthropic-version": self.api_version}

    async def _complete(self, system, user):
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
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    async def list_models(self):
        data = await self._get(f"{self.base_url}/models", self._headers(), {"limit": 1000})
        return [m["id"] for m in data.get("data", [])]


class GeminiTranslator(LLMTranslator):
    """Google Gemini generateContent API。"""

    provider = "gemini"

    def _headers(self):
        return {"x-goog-api-key": self.api_key}

    async def _complete(self, system, user):
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
        # 思考 token 会计费，只记 candidates 会把推理模型的用量算少
        completion = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
        self._record_usage(usage.get("promptTokenCount"), completion)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "无返回内容")
            raise TranslatorError(f"Gemini 未返回结果: {reason}")
        parts = (candidates[0].get("content") or {}).get("parts", [])
        # 思考模型可能带 thought 片段，只取正文
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    async def list_models(self):
        data = await self._get(f"{self.base_url}/models", self._headers(), {"pageSize": 1000})
        return [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]


class GoogleTranslator(Translator):
    """Google 网页版免费接口，无需 Key，适合试用；量大时可能被限流。"""

    # translate_a/t 支持一次传多个 q，大幅减少请求数，降低被限流的概率
    max_batch_items = 50
    max_batch_chars = 4500
    concurrency = 3
    retries = 6

    def __init__(self, target_lang, source_lang="auto", transport=None):
        super().__init__(target_lang, source_lang)
        # 插件的 Google 语言映射：zh-HK→zh-TW、pt-br→pt、pt→pt-PT、fil→tl，不在表里的语言不支持
        if target_lang not in GOOGLE_CODES:
            raise TranslatorError(f"谷歌翻译不支持目标语言「{prompts.lang_name(target_lang)}」，请换用 AI 翻译服务")
        if source_lang != "auto" and source_lang not in GOOGLE_CODES:
            raise TranslatorError(f"谷歌翻译不支持源语言「{prompts.lang_name(source_lang)}」，请改为自动检测或换用 AI 翻译服务")
        self.tl = GOOGLE_CODES[target_lang]
        self.sl = "auto" if source_lang == "auto" else GOOGLE_CODES[source_lang]
        # 和 LLM 引擎一样，实际发请求都过同一个信号量
        self._sem = asyncio.Semaphore(self.concurrency)
        self.client = httpx.AsyncClient(timeout=60, transport=transport)

    @property
    def cache_key(self):
        return f"{self.service_id}:google:{self.source_lang}:{self.target_lang}"

    async def translate_batch(self, texts):
        # 混合批按调用方标记拆成两组：富文本组用 format=html 保留行内格式，
        # 纯文本组保持 format=text，避免 & 等字符被实体转义。不从尖括号猜测。
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

    async def _request(self, texts, html: bool):
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
                # 被限流时偶尔 200 返回 HTML 验证码页，JSON 解析失败按可重试的限流处理
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                if isinstance(data, list):
                    # 每个 q 对应一项：sl=auto 时是 [译文, 检测到的语言]，否则直接是译文
                    out = [d[0] if isinstance(d, list) else d for d in data]
                    if len(out) != len(texts):
                        raise TranslatorError("Google 返回的译文条数与请求不一致")
                    return out
                if last:
                    raise TranslatorError("Google 返回的不是 JSON（可能已被限流，请稍后重试或改用 AI 翻译服务）")
                await asyncio.sleep(_retry_delay(resp, min(2 ** (attempt + 1), 30)))
                continue
            # 302 跳到验证码页 / 429 都是限流，退避重试
            if resp.status_code not in (302, 429, 500, 503) or last:
                raise TranslatorError(f"Google 接口错误 {resp.status_code}（可能已被限流，请稍后重试或改用 AI 翻译服务）")
            await asyncio.sleep(_retry_delay(resp, min(2 ** (attempt + 1), 30)))
        raise TranslatorError("已达到最大重试次数")

    async def aclose(self):
        await self.client.aclose()


class MockTranslator(Translator):
    """测试用：在文本前加上标记。"""

    async def translate_batch(self, texts):
        return [f"[{self.target_lang}] {t}" for t in texts]


LLM_CLASSES = {
    "openai": OpenAITranslator,
    "claude": ClaudeTranslator,
    "gemini": GeminiTranslator,
    "grok": GrokTranslator,
    "custom": CustomOpenAITranslator,
}


def create_translator(service: dict, target_lang: str, source_lang: str = "auto", transport=None) -> Translator:
    """根据翻译服务配置（见 services.py）创建引擎。"""
    translator = _build_translator(service, target_lang, source_lang, transport)
    translator.service_id = service.get("id") or ""
    return translator


def _build_translator(service: dict, target_lang: str, source_lang: str, transport) -> Translator:
    from .services import PROVIDERS

    provider = service.get("provider")
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
