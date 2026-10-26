"""应用设置，保存在 data/settings.json。

- retention_days：翻译记录的保留时间，0 表示永久保留。
- layout_model：PDF 版面模型，{"enabled": 是否使用, "path": 模型文件，空串表示默认位置}。
- formula_model：PDF 公式检测模型，格式同上，另有 "size": 输入边长（1024 或 768）。默认开启、用 1024：
  行内公式认得准不准直接影响译文，模型文件下好了就用，没下时只用规则。

可以在页面上改，也可以直接改这个文件后重启服务。
"""
import json
import threading
from pathlib import Path

from .formats.pdf_vision import FORMULA_SIZES

PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"
DEFAULTS = {
    "retention_days": 30,
    "layout_model": {"enabled": False, "path": ""},
    "formula_model": {"enabled": True, "path": "", "size": FORMULA_SIZES[0]},
}
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
        return self.model("layout_model")

    @property
    def formula_model(self) -> dict:
        return self.model("formula_model")

    def model(self, key: str) -> dict:
        """layout_model 或 formula_model 的设置。"""
        value = self.data.get(key)
        if not isinstance(value, dict):
            value = {}
        default = DEFAULTS[key]
        conf = {"enabled": bool(value.get("enabled", default["enabled"])), "path": str(value.get("path") or "")}
        if "size" in default:
            conf["size"] = value.get("size") if value.get("size") in FORMULA_SIZES else default["size"]
        return conf

    def public(self) -> dict:
        # 两个模型有自己的页面和接口（/api/layout-model、/api/formula-model），这里不带
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
        for key, name in (("layout_model", "版面模型"), ("formula_model", "公式检测模型")):
            if key not in changes:
                continue
            value = changes[key]
            if not isinstance(value, dict):
                raise SettingsError(f"{name}设置格式不对")
            merged = {**self.model(key), **value}
            conf = {"enabled": bool(merged.get("enabled")), "path": str(merged.get("path") or "").strip()}
            if "size" in DEFAULTS[key]:
                try:
                    conf["size"] = int(merged.get("size"))
                except (TypeError, ValueError):
                    conf["size"] = 0
                if conf["size"] not in FORMULA_SIZES:
                    raise SettingsError("输入尺寸只能是 " + " 或 ".join(str(size) for size in FORMULA_SIZES))
            with self.lock:
                self.data[key] = conf
                self._save()
        return self.public()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
