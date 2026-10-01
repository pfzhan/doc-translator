"""读取 CC Switch 里当前生效的供应商配置，作为翻译服务使用。

CC Switch 把配置存在 ~/.cc-switch/cc-switch.db 的 providers 表里，每个应用（claude / codex / gemini /
grokbuild）有一条 is_current=1 的记录，settings_config 是该应用的原始配置：
- claude：env 里的 ANTHROPIC_BASE_URL、ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY、ANTHROPIC_MODEL
- codex：auth.OPENAI_API_KEY + config.toml（model_provider、model、model_providers.<id>.base_url / wire_api）
- gemini：env 里的 GOOGLE_GEMINI_BASE_URL、GEMINI_API_KEY、GEMINI_MODEL（可来自通用配置）
- grokbuild：config.toml 的 [models].default 和 [model."<name>"] 里的 base_url / api_key / api_backend

每次使用时现读数据库（只读打开），所以在 CC Switch 里切换供应商后立即生效，Key 也不会复制到本项目里。
官方账号登录（OAuth）的配置没有 API Key，不能直接用于翻译，会标记为不可用。
"""
import json
import os
import re
import sqlite3
import tomllib
from pathlib import Path
from urllib.parse import urlparse

DB_PATH = Path(os.environ.get("CC_SWITCH_DB", "~/.cc-switch/cc-switch.db")).expanduser()

# CC Switch 应用 → (本项目服务 id, 引擎类型, 显示名)
APPS = {
    "claude": ("ccswitch-claude", "claude", "Claude"),
    "codex": ("ccswitch-codex", "openai", "ChatGPT"),
    "gemini": ("ccswitch-gemini", "gemini", "Gemini"),
    "grokbuild": ("ccswitch-grok", "grok", "Grok"),
}

# Claude Code 的模型名后缀，如 claude-opus-5.5[1M] 表示 1M 上下文，接口不认
MODEL_SUFFIX_RE = re.compile(r"\[[^\]]*\]$")


class Unavailable(Exception):
    pass


def _append_path(url: str, suffix: str, keep: tuple[str, ...] = ()) -> str:
    """SDK 会在 base url 后面拼版本号；这里补成本项目需要的“带版本号”的地址。"""
    url = url.rstrip("/")
    if url.endswith((suffix, *keep)):
        return url
    return url + suffix


def _openai_base(url: str) -> str:
    # OpenAI 兼容地址通常带 /v1；只有域名没有路径时补上
    url = url.rstrip("/")
    return url + "/v1" if not urlparse(url).path else url


def _parse_claude(cfg: dict, common: dict) -> dict:
    env = {**common.get("env", {}), **cfg.get("env", {})}
    token = env.get("ANTHROPIC_AUTH_TOKEN", "")
    key = env.get("ANTHROPIC_API_KEY", "")
    if not (token or key):
        raise Unavailable("使用的是 Claude 官方账号登录，没有 API Key")
    model = next((env[k] for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                                    "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL") if env.get(k)), "")
    return {
        "base_url": _append_path(env.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com", "/v1"),
        "api_key": token or key,
        # ANTHROPIC_AUTH_TOKEN 走 Authorization: Bearer，ANTHROPIC_API_KEY 走 x-api-key，和 Claude Code 一致
        "auth_style": "bearer" if token else "",
        "model": MODEL_SUFFIX_RE.sub("", model) or "claude-sonnet-5-5",
        "api_format": "",
    }


def _parse_codex(cfg: dict, common_toml: str, use_common: bool) -> dict:
    conf = tomllib.loads(common_toml) if use_common and common_toml else {}
    conf.update(tomllib.loads(cfg.get("config") or ""))
    provider_id = conf.get("model_provider") or "openai"
    mp = (conf.get("model_providers") or {}).get(provider_id, {})
    auth = cfg.get("auth") or {}
    key = auth.get("OPENAI_API_KEY") or (os.environ.get(mp["env_key"], "") if mp.get("env_key") else "")
    if not key:
        raise Unavailable("使用的是 ChatGPT 账号登录，没有 API Key")
    return {
        "base_url": _openai_base(mp.get("base_url") or "https://api.openai.com/v1"),
        "api_key": key,
        "auth_style": "",
        "model": conf.get("model") or "gpt-4.1-mini",
        # Codex 默认 wire_api 是 responses
        "api_format": "chat" if mp.get("wire_api") == "chat" else "responses",
    }


def _parse_gemini(cfg: dict, common: dict, use_common: bool) -> dict:
    # Gemini 的通用配置就是一组环境变量
    common_env = (common.get("env", common) if use_common else {}) or {}
    env = {**common_env, **cfg.get("env", {})}
    key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or ""
    if not key:
        raise Unavailable("使用的是 Google 账号登录，没有 API Key")
    base = env.get("GOOGLE_GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com"
    return {
        "base_url": _append_path(base, "/v1beta", keep=("/v1",)),
        "api_key": key,
        "auth_style": "",
        "model": env.get("GEMINI_MODEL") or "gemini-2.5-flash",
        "api_format": "",
    }


def _parse_grok(cfg: dict) -> dict:
    conf = tomllib.loads(cfg.get("config") or "")
    models = conf.get("model") or {}
    default = (conf.get("models") or {}).get("default")
    m = models.get(default) or next(iter(models.values()), None)
    if not m or not m.get("api_key"):
        raise Unavailable("使用的是 Grok 官方账号登录（OAuth），没有 API Key")
    return {
        "base_url": _openai_base(m.get("base_url") or "https://api.x.ai/v1"),
        "api_key": m["api_key"],
        "auth_style": "",
        "model": m.get("model") or default or "grok-4",
        "api_format": "responses" if m.get("api_backend") == "responses" else "chat",
    }


def _common_configs(con) -> dict:
    out = {}
    for key, value in con.execute("SELECT key, value FROM settings WHERE key LIKE 'common_config_%'"):
        out[key.removeprefix("common_config_")] = value or ""
    return out


def read_current(db_path: Path | None = None) -> list[dict]:
    """返回各应用当前生效的配置。数据库不存在时返回空列表。"""
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
