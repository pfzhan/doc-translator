"""翻译引擎。每个引擎实现 translate_batch(texts) -> list[str]。

大模型引擎（OpenAI / Grok / Gemini / Claude / OpenAI 兼容）共用同一套批量翻译逻辑，
提示词和 [[pN]] 标记协议取自沉浸式翻译（见 prompts.py）。模型漏掉的段落会单独补翻。
各家只在请求格式上不同，由子类实现 _complete。
"""
import asyncio
import hashlib

import httpx

from . import prompts
from .languages import GOOGLE_CODES, LANGUAGES

# 前端下拉框用的语言列表：[(代码, 英文名, 本地名)]，顺序与插件一致
LANG_CODES = {code for code, _, _ in LANGUAGES}


class TranslatorError(Exception):
    pass


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

    @property
    def effective_source(self) -> str:
        return self.detected_source if self.source_lang == "auto" and self.detected_source else self.source_lang

    @property
    def cache_key(self) -> str:
        return f"{type(self).__name__}:{self.source_lang}:{self.target_lang}"

    async def translate_batch(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    async def aclose(self):
        pass


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error", data)
        msg = err.get("message") if isinstance(err, dict) else err
        return str(msg or resp.text)[:300]
    except ValueError:
        return resp.text[:300]


class LLMTranslator(Translator):
    provider = ""
    retries = 4

    def __init__(self, target_lang, source_lang="auto", *, api_key="", base_url="", model="",
                 concurrency=4, max_items=20, max_chars=3000, temperature=0.0, prompt="", user_prompt="",
                 api_format="", auth_style="", transport=None):
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
        self.max_batch_items = max(1, int(max_items))
        self.max_batch_chars = max(200, int(max_chars))
        self.temperature = temperature
        self.prompt = prompt.strip()
        self.user_prompt = user_prompt.strip()
        # 文档标题，作为上下文填进提示词的 {{title_prompt}}
        self.title = ""
        self.client = httpx.AsyncClient(timeout=180, transport=transport)

    @property
    def cache_key(self):
        # 模型或提示词变了，译文也会不同，都要算进缓存 key
        p = hashlib.sha1(f"{self.prompt}\0{self.user_prompt}".encode()).hexdigest()[:8]
        # 源语言影响提示词（{{from}}、wyw2zh-CN 这类覆盖），用检测后的实际源语言
        return f"{self.provider}:{self.model}:{p}:{self.effective_source}:{self.target_lang}"

    def build_messages(self, texts: list[str]) -> tuple[str, str]:
        return prompts.build_messages(
            texts, self.target_lang, source_lang=self.effective_source, title=self.title,
            system_template=self.prompt, user_template=self.user_prompt,
        )

    async def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    async def _post(self, url: str, payload: dict, headers: dict) -> dict:
        """带重试的 POST。部分推理模型不接受 temperature，遇到这种 400 会去掉后重发。"""
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
                return resp.json()
            msg = _error_text(resp)
            if resp.status_code == 400 and "temperature" in msg.lower() and self.temperature is not None:
                self.temperature = None
                payload = {k: v for k, v in payload.items() if k != "temperature"}
                gen = payload.get("generationConfig")
                if isinstance(gen, dict):
                    payload["generationConfig"] = {k: v for k, v in gen.items() if k != "temperature"}
                continue
            if (resp.status_code == 429 or resp.status_code >= 500) and not last:
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            raise TranslatorError(f"接口错误 {resp.status_code}: {msg}")
        raise TranslatorError("重试次数用尽")

    async def _get(self, url: str, headers: dict, params: dict | None = None) -> dict:
        try:
            resp = await self.client.get(url, headers=headers, params=params)
        except httpx.HTTPError as e:
            raise TranslatorError(f"请求失败: {e}") from e
        if resp.status_code != 200:
            raise TranslatorError(f"接口错误 {resp.status_code}: {_error_text(resp)}")
        return resp.json()

    async def list_models(self) -> list[str]:
        raise NotImplementedError

    async def translate_batch(self, texts):
        system, user = self.build_messages(texts)
        content = await self._complete(system, user)
        if len(texts) == 1:
            return [prompts.clean_output(content)]
        found = prompts.parse_markers(content, len(texts))
        # 模型漏掉的段落逐条补翻（和插件的 recovery 思路一样，只补缺失的）
        missing = [i for i in range(len(texts)) if i not in found]
        for i in missing:
            [found[i]] = await self.translate_batch([texts[i]])
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
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise TranslatorError(f"返回格式异常: {str(data)[:200]}") from e

    async def _complete_responses(self, system, user):
        """OpenAI Responses API（Codex / Grok Build 默认用这个）。"""
        payload = {"model": self.model, "instructions": system, "input": user, "store": False}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = await self._post(f"{self.base_url}/responses", payload, self._headers())
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
        self.client = httpx.AsyncClient(timeout=60, transport=transport)

    @property
    def cache_key(self):
        return f"google:{self.source_lang}:{self.target_lang}"

    async def translate_batch(self, texts):
        params = {"client": "gtx", "sl": self.sl, "tl": self.tl, "format": "text"}
        for attempt in range(self.retries):
            last = attempt == self.retries - 1
            try:
                resp = await self.client.post(
                    "https://translate.googleapis.com/translate_a/t", params=params, data={"q": texts}
                )
            except httpx.HTTPError as e:
                if last:
                    raise TranslatorError(f"Google 请求失败: {e}") from e
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                # 每个 q 对应一项：sl=auto 时是 [译文, 检测到的语言]，否则直接是译文
                out = [d[0] if isinstance(d, list) else d for d in resp.json()]
                if len(out) != len(texts):
                    raise TranslatorError("Google 返回条数不一致")
                return out
            # 302 跳到验证码页 / 429 都是限流，退避重试
            if resp.status_code not in (302, 429, 500, 503) or last:
                raise TranslatorError(f"Google 接口错误 {resp.status_code}（可能被限流，稍后重试或换用 AI 翻译）")
            await asyncio.sleep(min(2 ** (attempt + 1), 30))
        raise TranslatorError("重试次数用尽")

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
    )
