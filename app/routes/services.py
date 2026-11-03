"""Routes of the translation services: the service list, the language and prompt tables, and testing a service."""

import time

from fastapi import APIRouter, Body, HTTPException

from .. import state
from ..languages import GOOGLE_CODES, LANGUAGES
from ..prompts import default_prompts
from ..services import PROVIDERS, ServiceError
from ..translators import TranslatorError, create_translator

router = APIRouter()


@router.get("/api/languages")
def languages():
    """Language list in the same order as the plugin. The google field tells whether Google Translate supports
    the language."""
    return [
        {"code": code, "name": en, "native": native, "google": code == "auto" or code in GOOGLE_CODES}
        for code, en, native in LANGUAGES
    ]


# ---------- Translation service management ----------


@router.get("/api/providers")
def providers():
    return PROVIDERS


@router.get("/api/prompts/default")
def default_prompt(target_lang: str = "zh-CN", source_lang: str = "auto"):
    return default_prompts(target_lang, source_lang)


@router.get("/api/services")
def list_services():
    return state.store.list()


@router.post("/api/services")
def create_service(data: dict = Body(...)):
    try:
        return state.store.create(data)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e


@router.put("/api/services/{sid}")
def update_service(sid: str, data: dict = Body(...)):
    try:
        return state.store.update(sid, data)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/api/services/{sid}")
def delete_service(sid: str):
    try:
        state.store.delete(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}


@router.post("/api/services/{sid}/default")
def set_default_service(sid: str):
    try:
        state.store.set_default(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e
    return state.store.list()


@router.post("/api/services/{sid}/clone")
def clone_service(sid: str):
    try:
        return state.store.clone_to_local(sid)
    except ServiceError as e:
        raise HTTPException(400, str(e)) from e


async def _with_translator(data: dict, fn):
    try:
        translator = create_translator(
            state.store.resolve(data), data.get("target_lang") or "zh-CN", data.get("source_lang") or "auto"
        )
    except (ServiceError, TranslatorError) as e:
        raise HTTPException(400, str(e)) from e
    try:
        return await fn(translator)
    except TranslatorError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        await translator.aclose()


@router.post("/api/services/test")
async def test_service(data: dict = Body(...)):
    """Translate one sentence with the configuration currently in the form (possibly unsaved) to check that the
    Key / address / model work."""
    # Use a Chinese sample sentence when the target language is English, otherwise an English one, to avoid
    # translating English into English
    target = data.get("target_lang") or "zh-CN"
    default_sample = "你好，世界！这是一条翻译测试。" if target == "en" else "Hello, world! This is a translation test."
    sample = data.get("text") or default_sample

    async def run(tr):
        start = time.monotonic()
        [out] = await tr.translate_batch([sample])
        return {"source": sample, "result": out, "ms": int((time.monotonic() - start) * 1000)}

    return await _with_translator(data, run)


@router.post("/api/services/models")
async def fetch_models(data: dict = Body(...)):
    async def run(tr):
        if not hasattr(tr, "list_models"):
            raise TranslatorError("该服务不提供模型列表")
        return {"models": await tr.list_models()}

    return await _with_translator(data, run)
