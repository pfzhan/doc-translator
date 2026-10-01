"""翻译服务配置：内置服务 + 用户自定义服务，保存在 data/services.json。

一个“服务”就是一份引擎配置（类型、名称、Key、地址、模型、并发等），
同一种类型可以添加多个，比如两个不同 Key 的 OpenAI、或者一个走代理地址的 Claude。
"""
import json
import threading
import uuid
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "services.json"

# 预置模型只是方便选择，可能不是最新的；页面上可以“获取模型列表”或手动输入模型名
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
        "desc": "任何兼容 OpenAI Chat Completions 格式的服务：DeepSeek、通义千问、Kimi、OpenRouter、Ollama、各类中转站等。",
        "base_url": "",
        "needs_key": False,
        "llm": True,
        "models": [],
    },
}

# 用户可编辑的字段及默认值
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
}

BUILTIN = [{"id": "google", "provider": "google", "builtin": True, **FIELDS, "name": "谷歌翻译"}]


class ServiceError(Exception):
    pass


def _mask(key: str) -> str:
    if not key:
        return ""
    return key[:3] + "…" + key[-4:] if len(key) > 10 else "…" * 3


def public(service: dict) -> dict:
    """返回给前端的版本：不带明文 Key。"""
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
                raise ServiceError(f"{k} 必须是整数")
            v = max(1, min(v, {"concurrency": 32, "max_items": 200, "max_chars": 50000}[k]))
        elif k == "temperature":
            if v in (None, ""):
                v = None
            else:
                try:
                    v = max(0.0, min(float(v), 2.0))
                except (TypeError, ValueError):
                    raise ServiceError("temperature 必须是数字")
        else:
            v = str(v or "") if k in ("prompt", "user_prompt") else str(v or "").strip()
        out[k] = v
    return out


class ServiceStore:
    def __init__(self, path: Path = DATA_PATH):
        self.path = path
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            data = {}
        self.services: list[dict] = data.get("services", [])
        self.default_id: str = data.get("default", "google")
        # 内置服务始终存在，用户可以改启用状态
        for b in BUILTIN:
            if not any(s["id"] == b["id"] for s in self.services):
                self.services.insert(0, dict(b))
        if not self.get(self.default_id):
            self.default_id = "google"

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"default": self.default_id, "services": self.services}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.chmod(0o600)  # 里面有 API Key，只允许当前用户读写
        tmp.replace(self.path)

    def get(self, sid: str) -> dict | None:
        return next((s for s in self.services if s["id"] == sid), None)

    def list(self) -> dict:
        return {"default": self.default_id, "services": [public(s) for s in self.services]}

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
            changes = _clean(data)
            # Key 留空表示不修改；要清空 Key 需显式传 clear_api_key
            if not changes.get("api_key") and not data.get("clear_api_key"):
                changes.pop("api_key", None)
            if svc.get("builtin"):
                changes = {k: v for k, v in changes.items() if k == "enabled"}
            if "name" in changes and not changes["name"]:
                changes.pop("name")
            if changes.get("enabled") is False and self.default_id == sid:
                raise ServiceError("默认服务不能关闭，请先把别的服务设为默认")
            svc.update(changes)
            self._save()
        return public(svc)

    def delete(self, sid: str):
        with self.lock:
            svc = self.get(sid)
            if not svc:
                raise ServiceError("服务不存在")
            if svc.get("builtin"):
                raise ServiceError("内置服务不能删除")
            self.services.remove(svc)
            if self.default_id == sid:
                self.default_id = "google"
            self._save()

    def set_default(self, sid: str):
        with self.lock:
            svc = self.get(sid)
            if not svc:
                raise ServiceError("服务不存在")
            svc["enabled"] = True
            self.default_id = sid
            self._save()

    def resolve(self, data: dict) -> dict:
        """把前端表单里的（可能未保存的）配置和已保存的配置合并，用于测试服务、获取模型列表。"""
        sid = data.get("id")
        base = dict(self.get(sid)) if sid and self.get(sid) else {"provider": data.get("provider"), **FIELDS}
        changes = _clean(data)
        if not changes.get("api_key"):
            changes.pop("api_key", None)
        base.update(changes)
        if base.get("provider") not in PROVIDERS:
            raise ServiceError("不支持的服务类型")
        return base
