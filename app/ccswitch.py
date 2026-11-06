"""Read the provider configuration currently active in CC Switch and use it as a translation service.

CC Switch stores its configuration in the providers table of ~/.cc-switch/cc-switch.db. Each app (claude / codex /
gemini / grokbuild) has one record with is_current=1, and settings_config is the app's raw configuration:
- claude: ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY and ANTHROPIC_MODEL in env
- codex: auth.OPENAI_API_KEY + config.toml (model_provider, model, model_providers.<id>.base_url / wire_api)
- gemini: GOOGLE_GEMINI_BASE_URL, GEMINI_API_KEY and GEMINI_MODEL in env (possibly from the common configuration)
- grokbuild: base_url / api_key / api_backend in [models].default and [model."<name>"] of config.toml

The database is read (opened read-only) on each use, so switching providers in CC Switch takes effect at once,
and keys are never copied into this project. Configurations logged in with an official account (OAuth) have no
API key, cannot be used for translation directly, and are marked unavailable.
"""

import json
import os
import re
import sqlite3
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from .json_types import JSONDict

DB_PATH = Path(os.environ.get("CC_SWITCH_DB", "~/.cc-switch/cc-switch.db")).expanduser()

# CC Switch app → (service id in this project, engine type, display name)
APPS = {
    "claude": ("ccswitch-claude", "claude", "Claude"),
    "codex": ("ccswitch-codex", "openai", "ChatGPT"),
    "gemini": ("ccswitch-gemini", "gemini", "Gemini"),
    "grokbuild": ("ccswitch-grok", "grok", "Grok"),
}

# Claude Code model name suffix, such as claude-opus-5.5[1M] for a 1M context, which the API does not accept
MODEL_SUFFIX_RE = re.compile(r"\[[^\]]*\]$")


class Unavailable(Exception):
    pass


def _append_path(url: str, suffix: str, keep: tuple[str, ...] = ()) -> str:
    """SDKs append a version to the base URL; this completes it to the versioned URL this project needs."""
    url = url.rstrip("/")
    if url.endswith((suffix, *keep)):
        return url
    return url + suffix


def _openai_base(url: str) -> str:
    # OpenAI-compatible URLs usually end in /v1; add it when there is only a host without a path
    url = url.rstrip("/")
    return url + "/v1" if not urlparse(url).path else url


def _parse_claude(cfg: JSONDict, common: JSONDict) -> JSONDict:
    env = {**common.get("env", {}), **cfg.get("env", {})}
    token = env.get("ANTHROPIC_AUTH_TOKEN", "")
    key = env.get("ANTHROPIC_API_KEY", "")
    if not (token or key):
        raise Unavailable("当前以 Claude 官方账号登录，没有 API Key")
    model = next(
        (
            env[k]
            for k in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            )
            if env.get(k)
        ),
        "",
    )
    return {
        "base_url": _append_path(env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com", "/v1"),
        "api_key": token or key,
        # ANTHROPIC_AUTH_TOKEN is sent as Authorization: Bearer and ANTHROPIC_API_KEY as x-api-key, as in Claude Code
        "auth_style": "bearer" if token else "",
        "model": MODEL_SUFFIX_RE.sub("", model) or "claude-sonnet-5-5",
        "api_format": "",
    }


def _parse_codex(cfg: JSONDict, common_toml: str, use_common: bool) -> JSONDict:
    conf = tomllib.loads(common_toml) if use_common and common_toml else {}
    conf.update(tomllib.loads(cfg.get("config") or ""))
    provider_id = conf.get("model_provider") or "openai"
    mp = (conf.get("model_providers") or {}).get(provider_id, {})
    auth = cfg.get("auth") or {}
    key = auth.get("OPENAI_API_KEY") or (os.environ.get(mp["env_key"], "") if mp.get("env_key") else "")
    if not key:
        raise Unavailable("当前以 ChatGPT 账号登录，没有 API Key")
    return {
        "base_url": _openai_base(mp.get("base_url") or "https://api.openai.com/v1"),
        "api_key": key,
        "auth_style": "",
        "model": conf.get("model") or "gpt-4.1-mini",
        # The default wire_api of Codex is responses
        "api_format": "chat" if mp.get("wire_api") == "chat" else "responses",
    }


def _parse_gemini(cfg: JSONDict, common: JSONDict, use_common: bool) -> JSONDict:
    # The common configuration of Gemini is a set of environment variables
    common_env = (common.get("env", common) if use_common else {}) or {}
    env = {**common_env, **cfg.get("env", {})}
    key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or ""
    if not key:
        raise Unavailable("当前以 Google 账号登录，没有 API Key")
    base = env.get("GOOGLE_GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com"
    return {
        "base_url": _append_path(base, "/v1beta", keep=("/v1",)),
        "api_key": key,
        "auth_style": "",
        "model": env.get("GEMINI_MODEL") or "gemini-2.5-flash",
        "api_format": "",
    }


def _parse_grok(cfg: JSONDict) -> JSONDict:
    conf = tomllib.loads(cfg.get("config") or "")
    models = conf.get("model") or {}
    default = (conf.get("models") or {}).get("default")
    m = models.get(default) or next(iter(models.values()), None)
    if not m or not m.get("api_key"):
        raise Unavailable("当前以 Grok 官方账号登录（OAuth），没有 API Key")
    return {
        "base_url": _openai_base(m.get("base_url") or "https://api.x.ai/v1"),
        "api_key": m["api_key"],
        "auth_style": "",
        "model": m.get("model") or default or "grok-4",
        "api_format": "responses" if m.get("api_backend") == "responses" else "chat",
    }


def _common_configs(con: sqlite3.Connection) -> JSONDict:
    out = {}
    for key, value in con.execute("SELECT key, value FROM settings WHERE key LIKE 'common_config_%'"):
        out[key.removeprefix("common_config_")] = value or ""
    return out


def read_current(db_path: Path | None = None) -> list[JSONDict]:
    """Return the active configuration of each app. An empty list when the database does not exist."""
    path = Path(db_path or DB_PATH)
    if not path.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        try:
            rows = con.execute(
                "SELECT app_type, name, settings_config, meta FROM providers WHERE is_current = 1"
            ).fetchall()
            common = _common_configs(con)
        finally:
            con.close()
    except sqlite3.Error:
        return []

    result = []
    for app, name, raw_cfg, raw_meta in rows:
        if app not in APPS:
            continue
        sid, provider, label = APPS[app]
        item = {
            "id": sid,
            "app": app,
            "provider": provider,
            "label": label,
            "provider_name": name,
            "available": True,
            "reason": "",
            "base_url": "",
            "api_key": "",
            "auth_style": "",
            "model": "",
            "api_format": "",
        }
        try:
            cfg = json.loads(raw_cfg or "{}")
            meta = json.loads(raw_meta or "{}")
            use_common = bool(meta.get("commonConfigEnabled"))
            if app == "claude":
                common_cfg = json.loads(common.get("claude") or "{}") if use_common else {}
                item.update(_parse_claude(cfg, common_cfg))
            elif app == "codex":
                item.update(_parse_codex(cfg, common.get("codex", ""), use_common))
            elif app == "gemini":
                item.update(_parse_gemini(cfg, json.loads(common.get("gemini") or "{}"), use_common))
            elif app == "grokbuild":
                item.update(_parse_grok(cfg))
        except Unavailable as e:
            item.update(available=False, reason=str(e))
        except (ValueError, tomllib.TOMLDecodeError, AttributeError, TypeError) as e:
            item.update(available=False, reason=f"配置解析失败：{e}")
        result.append(item)
    order = list(APPS)
    return sorted(result, key=lambda x: order.index(x["app"]))
