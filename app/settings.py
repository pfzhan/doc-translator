"""应用设置，保存在 data/settings.json。

- retention_days：翻译记录的保留时间，0 表示永久保留。
- layout_model：PDF 版面模型，{"enabled": 是否使用, "path": 模型文件，空串表示默认位置}。

可以在页面上改，也可以直接改这个文件后重启服务。
"""
import json
import threading
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"
DEFAULTS = {"retention_days": 30, "layout_model": {"enabled": False, "path": ""}}
MAX_RETENTION_DAYS = 3650


class SettingsError(Exception):
    pass


class Settings:
    def __init__(self, path: Path = PATH):
        self.path = path
        self.lock = threading.Lock()
        self.data = dict(DEFAULTS)
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except (ValueError, OSError):
                pass  # 文件损坏时用默认值，不影响启动

    @property
    def retention_days(self) -> int:
        try:
            return max(0, min(int(self.data.get("retention_days", DEFAULTS["retention_days"])), MAX_RETENTION_DAYS))
        except (TypeError, ValueError):
            return DEFAULTS["retention_days"]

    @property
    def layout_model(self) -> dict:
        value = self.data.get("layout_model")
        if not isinstance(value, dict):
            value = {}
        return {"enabled": bool(value.get("enabled", False)), "path": str(value.get("path") or "")}

    def public(self) -> dict:
        # 版面模型有自己的页面和接口（/api/layout-model），这里不带
        return {"retention_days": self.retention_days}

    def update(self, changes: dict) -> dict:
        if "retention_days" in changes:
            try:
                days = int(changes["retention_days"])
            except (TypeError, ValueError):
                raise SettingsError("保留天数必须是整数")
            if not 0 <= days <= MAX_RETENTION_DAYS:
                raise SettingsError(f"保留天数需要在 0 到 {MAX_RETENTION_DAYS} 之间（0 表示永久保留）")
            with self.lock:
                self.data["retention_days"] = days
                self._save()
        if "layout_model" in changes:
            value = changes["layout_model"]
            if not isinstance(value, dict):
                raise SettingsError("版面模型设置格式不对")
            merged = {**self.layout_model, **value}
            with self.lock:
                self.data["layout_model"] = {
                    "enabled": bool(merged.get("enabled")),
                    "path": str(merged.get("path") or "").strip(),
                }
                self._save()
        return self.public()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
