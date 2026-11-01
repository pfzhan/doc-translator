"""Translation service configuration: built-in services plus user-defined ones, saved in data/services.json.

A service is one engine configuration (type, name, key, URL, model, concurrency and so on). Several services of the
same type can be added, such as two OpenAI services with different keys, or a Claude service through a proxy URL.
"""

import json
import threading
import uuid
from pathlib import Path

from . import ccswitch

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "services.json"

# Preset models are only a convenience and may be outdated; the page can fetch the model list ("获取模型列表") or
# take a model name typed by hand
PROVIDERS = {
    "google": {
        "label": "谷歌翻译",
        "desc": "免费的机器翻译，无需 API Key。文档较大时可能被限流。",
        "base_url": "",
        "needs_key": False,
        "llm": False,
        "models": [],
    },
    "openai": {
        "label": "OpenAI",
        "desc": "OpenAI 的 GPT 系列模型，理解上下文，译文自然流畅。",
        "base_url": "https://api.openai.com/v1",
        "needs_key": True,
        "llm": True,
        "key_url": "https://platform.openai.com/api-keys",
        "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5"],
    },
    "claude": {
        "label": "Claude",
        "desc": "Anthropic 的 Claude 系列模型，擅长长文本和细腻的语气。",
        "base_url": "https://api.anthropic.com/v1",
        "needs_key": True,
        "llm": True,
        "key_url": "https://console.anthropic.com/settings/keys",
        "models": ["claude-sonnet-5-5", "claude-haiku-4-5-20251001", "claude-opus-5-5", "claude-fable-5-1"],
    },
    "gemini": {
        "label": "Gemini",
        "desc": "Google 的 Gemini 系列模型，速度快，有免费额度。",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "needs_key": True,
        "llm": True,
        "key_url": "https://aistudio.google.com/apikey",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    },
    "grok": {
        "label": "Grok",
        "desc": "xAI 的 Grok 系列模型，接口兼容 OpenAI 格式。",
        "base_url": "https://api.x.ai/v1",
        "needs_key": True,
        "llm": True,
        "key_url": "https://console.x.ai",
        "models": ["grok-4", "grok-3", "grok-3-mini"],
    },
    "custom": {
        "label": "OpenAI 兼容接口",
        "desc": "任何兼容 OpenAI Chat Completions 格式的服务：DeepSeek、通义千问、Kimi、OpenRouter、Ollama、各类中转站等。",  # noqa: E501
        "base_url": "",
        "needs_key": False,
        "llm": True,
        "models": [],
    },
}

# Fields the user can edit, with their defaults
FIELDS = {
    "name": "",
    "enabled": True,
    "api_key": "",
    "base_url": "",
    "model": "",
    "concurrency": 4,
    "max_items": 20,
    "max_chars": 3000,
    "temperature": 0,
    "prompt": "",
    "user_prompt": "",
    "glossary": "",
    "auto_glossary": True,
}

BUILTIN = [{"id": "google", "provider": "google", "builtin": True, **FIELDS, "name": "谷歌翻译"}]


class ServiceError(Exception):
    pass


def _mask(key: str) -> str:
    if not key:
        return ""
    return key[:3] + "…" + key[-4:] if len(key) > 10 else "…" * 3


# Fields of CC Switch services that can be adjusted here; connection details (URL, key) always come from CC Switch
CCSWITCH_LOCAL_FIELDS = {
    "enabled",
    "model",
    "concurrency",
    "max_items",
    "max_chars",
    "temperature",
    "prompt",
    "user_prompt",
    "glossary",
    "auto_glossary",
}


def public(service: dict) -> dict:
    """The version returned to the frontend, without the plain-text key."""
    out = {k: v for k, v in service.items() if k != "api_key"}
    out["api_key_hint"] = _mask(service.get("api_key", ""))
    return out


def _clean(data: dict) -> dict:
    out = {}
    for k, default in FIELDS.items():
        if k not in data:
            continue
        v = data[k]
        if isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int) and k != "temperature":
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ServiceError(f"{k} 必须是整数") from None
            v = max(1, min(v, {"concurrency": 32, "max_items": 200, "max_chars": 50000}[k]))
        elif k == "temperature":
            if v in (None, ""):
                v = None
            else:
                try:
                    v = max(0.0, min(float(v), 2.0))
                except (TypeError, ValueError):
                    raise ServiceError("temperature 必须是数字") from None
        else:
            v = str(v or "") if k in ("prompt", "user_prompt", "glossary") else str(v or "").strip()
        out[k] = v
    return out


class ServiceStore:
    def __init__(self, path: Path = DATA_PATH, ccswitch_db: Path | None = None):
        self.path = path
        self.ccswitch_db = Path(ccswitch_db or ccswitch.DB_PATH)
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        data = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (ValueError, OSError):
                pass  # a damaged file falls back to an empty configuration and does not block startup
        self.services: list[dict] = data.get("services", [])
        self.default_id: str = data.get("default", "google")
        # Local settings of CC Switch services: {service id: {enabled, model, prompt, ...}}
        self.ccswitch_settings: dict[str, dict] = data.get("ccswitch", {})
        # Built-in services always exist; the user can enable or disable them
        for b in BUILTIN:
            if not any(s["id"] == b["id"] for s in self.services):
                self.services.insert(0, dict(b))
        # CC Switch services may be temporarily unreadable (for example when the database is locked); the default is
        # kept and list() handles the fallback
        if not self.get(self.default_id) and not self.default_id.startswith("ccswitch-"):
            self.default_id = "google"

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"default": self.default_id, "services": self.services, "ccswitch": self.ccswitch_settings},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.chmod(0o600)  # contains API keys; readable and writable by the current user only
        tmp.replace(self.path)

    # ---------- CC Switch ----------

    def _ccswitch_services(self) -> list[dict]:
        """The provider currently active in CC Switch. Connection details are read each time; only settings such as
        enabled, concurrency and prompt are stored locally."""
        out = []
        for cc in ccswitch.read_current(self.ccswitch_db):
            local = self.ccswitch_settings.get(cc["id"], {})
            svc = {
                "id": cc["id"],
                "provider": cc["provider"],
                "builtin": True,
                "source": "ccswitch",
                **FIELDS,
                "enabled": cc["available"],
                **{k: v for k, v in local.items() if k in CCSWITCH_LOCAL_FIELDS},
                # These come from CC Switch and cannot be changed here
                "name": f"{cc['label']} · {cc['provider_name']}",
                "base_url": cc["base_url"],
                "api_key": cc["api_key"],
                "model": local.get("model") or cc["model"],
                "ccswitch_model": cc["model"],
                "api_format": cc["api_format"],
                "auth_style": cc["auth_style"],
                "available": cc["available"],
                "unavailable_reason": cc["reason"],
            }
            if not cc["available"]:
                svc["enabled"] = False
            out.append(svc)
        return out

    def _all(self) -> list[dict]:
        return [*self.services, *self._ccswitch_services()]

    def get(self, sid: str) -> dict | None:
        return next((s for s in self._all() if s["id"] == sid), None)

    def effective_default(self) -> str:
        """Id of the effective default service: falls back to Google Translate when the configured service does not
        exist or is disabled (for example when CC Switch is temporarily unreadable)."""
        svc = self.get(self.default_id)
        return self.default_id if svc and svc.get("enabled") else "google"

    def list(self) -> dict:
        all_services = self._all()
        return {
            "default": self.effective_default(),
            "services": [public(s) for s in all_services],
            "ccswitch": {"db": str(self.ccswitch_db), "found": self.ccswitch_db.exists()},
        }

    def _update_ccswitch(self, svc: dict, data: dict) -> dict:
        changes = {k: v for k, v in _clean(data).items() if k in CCSWITCH_LOCAL_FIELDS}
        if changes.get("enabled") and not svc["available"]:
            raise ServiceError(f"无法启用：{svc['unavailable_reason']}")
        if changes.get("enabled") is False and self.default_id == svc["id"]:
            raise ServiceError("默认服务不能关闭，请先将其他服务设为默认")
        # A model equal to CC Switch's is not saved separately, so changing the model in CC Switch takes effect here
        if "model" in changes and changes["model"] in ("", svc["ccswitch_model"]):
            changes["model"] = ""
        self.ccswitch_settings.setdefault(svc["id"], {}).update(changes)
        self._save()
        return public(self.get(svc["id"]))

    def create(self, data: dict) -> dict:
        provider = data.get("provider")
        if provider not in PROVIDERS or provider == "google":
            raise ServiceError("不支持的服务类型")
        meta = PROVIDERS[provider]
        with self.lock:
            svc = {"id": uuid.uuid4().hex[:8], "provider": provider, "builtin": False, **FIELDS}
            svc["name"] = meta["label"]
            svc["model"] = meta["models"][0] if meta["models"] else ""
            svc.update(_clean(data))
            if not svc["name"]:
                svc["name"] = meta["label"]
            self.services.append(svc)
            self._save()
        return public(svc)

    def update(self, sid: str, data: dict) -> dict:
        with self.lock:
            svc = self.get(sid)
            if not svc:
                raise ServiceError("服务不存在")
            if svc.get("source") == "ccswitch":
                return self._update_ccswitch(svc, data)
            changes = _clean(data)
            # An empty key means unchanged; clearing the key requires clear_api_key explicitly
            if not changes.get("api_key") and not data.get("clear_api_key"):
                changes.pop("api_key", None)
            if svc.get("builtin"):
                changes = {k: v for k, v in changes.items() if k == "enabled"}
            if "name" in changes and not changes["name"]:
                changes.pop("name")
            if changes.get("enabled") is False and self.default_id == sid:
                raise ServiceError("默认服务不能关闭，请先将其他服务设为默认")
            svc.update(changes)
            self._save()
        return public(svc)

    def delete(self, sid: str):
        with self.lock:
            svc = self.get(sid)
            if not svc:
                raise ServiceError("服务不存在")
            if svc.get("builtin"):
                raise ServiceError(
                    "内置服务不能删除" if svc.get("source") != "ccswitch" else "CC Switch 的服务不能删除，可将其关闭"
                )
            self.services.remove(svc)
            if self.default_id == sid:
                self.default_id = "google"
            self._save()

    def clone_to_local(self, sid: str) -> dict:
        """Copy a CC Switch service into a local service: connection details such as URL, key and model are fixed
        and no longer follow CC Switch."""
        with self.lock:
            svc = self.get(sid)
            if not svc:
                raise ServiceError("服务不存在")
            if svc.get("source") != "ccswitch":
                raise ServiceError("本地服务无需复制，可直接编辑")
            if not svc["available"]:
                raise ServiceError(f"无法复制：{svc['unavailable_reason']}")
            clone = {
                "id": uuid.uuid4().hex[:8],
                "provider": svc["provider"],
                "builtin": False,
                **{k: svc.get(k, v) for k, v in FIELDS.items()},
                "name": f"{svc['name']}（本地）",
            }
            # api_format/auth_style are not in FIELDS but are needed to create the engine (Claude models served
            # through Responses/Bearer)
            for k in ("api_format", "auth_style"):
                if svc.get(k):
                    clone[k] = svc[k]
            self.services.append(clone)
            self._save()
        return public(clone)

    def set_default(self, sid: str):
        with self.lock:
            svc = self.get(sid)
            if not svc:
                raise ServiceError("服务不存在")
            if svc.get("source") == "ccswitch":
                if not svc["available"]:
                    raise ServiceError(f"无法设为默认：{svc['unavailable_reason']}")
                self.ccswitch_settings.setdefault(sid, {})["enabled"] = True
            else:
                svc["enabled"] = True
            self.default_id = sid
            self._save()

    def resolve(self, data: dict) -> dict:
        """Merge the (possibly unsaved) configuration from the frontend form with the saved one, for testing the
        service and fetching the model list."""
        sid = data.get("id")
        found = self.get(sid) if sid else None
        base = dict(found) if found else {"provider": data.get("provider"), **FIELDS}
        changes = _clean(data)
        if not changes.get("api_key"):
            changes.pop("api_key", None)
        if base.get("source") == "ccswitch":
            if not base["available"]:
                raise ServiceError(base["unavailable_reason"])
            # URL and key come from CC Switch; the form can temporarily override only local settings
            changes = {k: v for k, v in changes.items() if k in CCSWITCH_LOCAL_FIELDS}
            if not changes.get("model"):
                changes.pop("model", None)
        base.update(changes)
        if base.get("provider") not in PROVIDERS:
            raise ServiceError("不支持的服务类型")
        return base
