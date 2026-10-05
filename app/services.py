"""翻译服务配置：内置服务 + 用户自定义服务，保存在 data/services.json。

一个“服务”就是一份引擎配置（类型、名称、Key、地址、模型、并发等），
同一种类型可以添加多个，比如两个不同 Key 的 OpenAI、或者一个走代理地址的 Claude。
"""
import json
import threading
import uuid
from pathlib import Path

from . import ccswitch

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
    "glossary": "",
}

BUILTIN = [{"id": "google", "provider": "google", "builtin": True, **FIELDS, "name": "谷歌翻译"}]


class ServiceError(Exception):
    pass


def _mask(key: str) -> str:
    if not key:
        return ""
    return key[:3] + "…" + key[-4:] if len(key) > 10 else "…" * 3


# CC Switch 服务可以在本项目里调整的字段；连接信息（地址、Key）始终以 CC Switch 为准
CCSWITCH_LOCAL_FIELDS = {"enabled", "model", "concurrency", "max_items", "max_chars", "temperature",
                         "prompt", "user_prompt", "glossary"}


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
                pass  # 文件损坏时回退到空配置，不影响启动
        self.services: list[dict] = data.get("services", [])
        self.default_id: str = data.get("default", "google")
        # CC Switch 服务的本地设置：{服务 id: {enabled, model, prompt, ...}}
        self.ccswitch_settings: dict[str, dict] = data.get("ccswitch", {})
        # 内置服务始终存在，用户可以改启用状态
        for b in BUILTIN:
            if not any(s["id"] == b["id"] for s in self.services):
                self.services.insert(0, dict(b))
        # CC Switch 的服务可能暂时读不到（比如数据库被占用），默认值保留，list() 时再兜底
        if not self.get(self.default_id) and not self.default_id.startswith("ccswitch-"):
            self.default_id = "google"

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"default": self.default_id, "services": self.services, "ccswitch": self.ccswitch_settings},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.chmod(0o600)  # 里面有 API Key，只允许当前用户读写
        tmp.replace(self.path)

    # ---------- CC Switch ----------

    def _ccswitch_services(self) -> list[dict]:
        """CC Switch 当前生效的供应商。连接信息每次现读；本地只保存开关、并发、提示词等设置。"""
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
                # 下面这些来自 CC Switch，不能在本项目里改
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
        """实际生效的默认服务 id：配置的服务不存在或已关闭（比如 CC Switch 暂时读不到）时退回谷歌翻译。"""
        svc = self.get(self.default_id)
        return self.default_id if svc and svc.get("enabled") else "google"

    def list(self) -> dict:
        all_services = self._all()
        return {"default": self.effective_default(), "services": [public(s) for s in all_services],
                "ccswitch": {"db": str(self.ccswitch_db), "found": self.ccswitch_db.exists()}}

    def _update_ccswitch(self, svc: dict, data: dict) -> dict:
        changes = {k: v for k, v in _clean(data).items() if k in CCSWITCH_LOCAL_FIELDS}
        if changes.get("enabled") and not svc["available"]:
            raise ServiceError(f"无法启用：{svc['unavailable_reason']}")
        if changes.get("enabled") is False and self.default_id == svc["id"]:
            raise ServiceError("默认服务不能关闭，请先把别的服务设为默认")
        # 模型和 CC Switch 一致时不单独保存，这样在 CC Switch 里改模型也会跟着变
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
                raise ServiceError("内置服务不能删除" if svc.get("source") != "ccswitch" else
                                   "CC Switch 的服务不能删除，可以关闭它")
            self.services.remove(svc)
            if self.default_id == sid:
                self.default_id = "google"
            self._save()

    def clone_to_local(self, sid: str) -> dict:
        """把 CC Switch 服务复制成本地服务：地址、Key、模型等连接信息固化，之后不再跟随 CC Switch。"""
        with self.lock:
            svc = self.get(sid)
            if not svc:
                raise ServiceError("服务不存在")
            if svc.get("source") != "ccswitch":
                raise ServiceError("本地服务不需要复制，直接编辑即可")
            if not svc["available"]:
                raise ServiceError(f"无法复制：{svc['unavailable_reason']}")
            clone = {"id": uuid.uuid4().hex[:8], "provider": svc["provider"], "builtin": False,
                     **{k: svc.get(k, v) for k, v in FIELDS.items()},
                     "name": f"{svc['name']}（本地）"}
            # api_format/auth_style 不在 FIELDS 里，但创建引擎时要用（Claude 系走 Responses/Bearer 的情况）
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
        """把前端表单里的（可能未保存的）配置和已保存的配置合并，用于测试服务、获取模型列表。"""
        sid = data.get("id")
        found = self.get(sid) if sid else None
        base = dict(found) if found else {"provider": data.get("provider"), **FIELDS}
        changes = _clean(data)
        if not changes.get("api_key"):
            changes.pop("api_key", None)
        if base.get("source") == "ccswitch":
            if not base["available"]:
                raise ServiceError(base["unavailable_reason"])
            # 地址和 Key 以 CC Switch 为准，表单里只有本地设置能临时覆盖
            changes = {k: v for k, v in changes.items() if k in CCSWITCH_LOCAL_FIELDS}
            if not changes.get("model"):
                changes.pop("model", None)
        base.update(changes)
        if base.get("provider") not in PROVIDERS:
            raise ServiceError("不支持的服务类型")
        return base
