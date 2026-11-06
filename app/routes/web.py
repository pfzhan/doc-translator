"""Web page translation, called by the browser extension in extension/."""

import asyncio
from contextlib import AsyncExitStack

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import state
from ..runner import WEB_CACHE_OWNER, Runner, get_cache
from ..translators import Translator, TranslatorError, create_translator

router = APIRouter()

# 50 texts of 10,000 characters are at most 2 MB as UTF-8 JSON (the extension sends characters unescaped); twice
# that leaves room for clients that escape them
MAX_WEB_REQUEST = 4 * 1024 * 1024


# ---------- Web page translation (called by the browser extension in extension/) ----------


class WebTranslateRequest(BaseModel):
    # One request carries the paragraphs near the viewport; the extension sends more as the page scrolls
    texts: list[str] = Field(max_length=50)
    # Aligned with texts: which are HTML fragments (inline formatting the translation keeps, such as links)
    html: list[bool] = Field(default_factory=list, max_length=50)
    target_lang: str = "zh-CN"
    source_lang: str = "auto"
    service_id: str = ""  # empty: the default service
    title: str = Field("", max_length=1000)  # page title, context for LLM engines (cut to 200 characters there)
    # Texts already in target_lang are translated into this language instead (for selected text); empty: they are
    # kept as they are
    alt_lang: str = ""


class WebTranslateResponse(BaseModel):
    # Aligned with texts; None keeps the original (blank, already in the target language, or translated unchanged)
    translations: list[str | None]
    # Aligned with texts: the language of each translation, target_lang or alt_lang; None where the original is kept
    langs: list[str | None]
    detected_lang: str


MAX_WEB_TEXT = 10000


@router.post("/api/web/translate")
async def web_translate(data: WebTranslateRequest) -> WebTranslateResponse:
    if any(len(t) > MAX_WEB_TEXT for t in data.texts):
        raise HTTPException(400, f"单段文字太长（上限 {MAX_WEB_TEXT} 字）")
    if len(data.html) > len(data.texts):
        raise HTTPException(400, "html 比 texts 长")
    service = state.store.get(data.service_id or state.store.effective_default())
    if not service or not service.get("enabled"):
        raise HTTPException(400, "翻译服务不存在或未启用")
    try:
        translator = create_translator(service, data.target_lang, data.source_lang)
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e
    async with AsyncExitStack() as stack:
        stack.push_async_callback(translator.aclose)
        try:
            results, runner = await _translate(translator, data.texts, data.html, data.title)
        except TranslatorError as e:
            raise HTTPException(502, str(e)) from e
        langs = [data.target_lang] * len(data.texts)
        again = []
        if data.alt_lang and data.alt_lang != data.target_lang:
            again = [i for i, text in enumerate(data.texts) if text in runner.same_lang]
        if again:
            # Created only when a text needs it, so that an alt_lang the service does not support fails no other
            # request. The source is detected again: target_lang may not be a source language the service accepts
            try:
                alt_translator = create_translator(service, data.alt_lang, "auto")
            except TranslatorError as e:
                raise HTTPException(400, f"改译已是目标语言的段落时出错：{e}") from e
            stack.push_async_callback(alt_translator.aclose)
            html = [i < len(data.html) and data.html[i] for i in again]
            try:
                alt_results, _ = await _translate(alt_translator, [data.texts[i] for i in again], html, data.title)
            except TranslatorError as e:
                raise HTTPException(502, str(e)) from e
            for i, dst in zip(again, alt_results):
                results[i] = dst
                langs[i] = data.alt_lang
    translations: list[str | None] = [dst if dst != src else None for src, dst in zip(data.texts, results)]
    return WebTranslateResponse(
        translations=translations,
        langs=[lang if dst is not None else None for lang, dst in zip(langs, translations)],
        detected_lang=runner.info.detected_lang,
    )


async def _translate(
    translator: Translator, texts: list[str], html: list[bool], title: str
) -> tuple[list[str], Runner]:
    """Translate the texts of a web page; the runner tells which of them were already in the target language."""
    # The document glossary costs one more request per call, too slow for a page being read; the user's glossary
    # still applies
    if hasattr(translator, "auto_glossary"):
        translator.auto_glossary = False
    runner = Runner(translator)
    # Web page translations are recorded under one owner, so that they can be cleared without touching documents
    runner.set_document(WEB_CACHE_OWNER)
    runner.set_title(title)
    return await runner.translate_all(texts, html=html), runner


class WebCacheInfo(BaseModel):
    entries: int


@router.get("/api/web/cache")
async def web_cache() -> WebCacheInfo:
    """How many cached translations clearing the web page cache would remove."""
    return WebCacheInfo(entries=await asyncio.to_thread(get_cache().count_only, WEB_CACHE_OWNER))


@router.delete("/api/web/cache")
async def clear_web_cache() -> WebCacheInfo:
    """Clear the translations of web pages. Sentences a document also uses stay, for that document."""
    return WebCacheInfo(entries=await asyncio.to_thread(get_cache().purge_only, WEB_CACHE_OWNER))
