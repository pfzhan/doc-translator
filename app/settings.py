"""Application settings, saved in data/settings.json.

- retention_days: how long translation records are kept; 0 keeps them forever.
- max_jobs: how many jobs translate at once; the others wait in the queue in order.
- layout_model: the PDF layout model, {"enabled": whether to use it, "path": model file, empty for the default
  location, "precision": int8 or fp32}. precision defaults to int8: the int8 file is used once generated (see
  quantize.py), fp32 otherwise.
- formula_model: the PDF formula detection model, in the same format plus "size": the input side (1024 or 768).
  Enabled by default at 1024: recognizing inline formulas correctly directly affects the translation, so the model
  is used once downloaded, and only the rules are used before that.

The settings can be changed on the page, or by editing this file and restarting the server.
"""

import json
import threading
from pathlib import Path

from .formats.pdf_vision import FORMULA_SIZES
from .paths import DATA_DIR
from .quantize import PRECISIONS

PATH = DATA_DIR / "settings.json"
DEFAULTS = {
    "retention_days": 30,
    "max_jobs": 2,
    "layout_model": {"enabled": False, "path": "", "precision": PRECISIONS[0]},
    # workers and threads are how many pages the formula detection model infers at once and with how many threads
    # each; 0 decides by core count
    "formula_model": {
        "enabled": True,
        "path": "",
        "size": FORMULA_SIZES[0],
        "precision": PRECISIONS[0],
        "workers": 0,
        "threads": 0,
    },
}
MAX_RETENTION_DAYS = 3650
MAX_JOBS = 8
# Upper limits for the pages inferred at once by the formula detection model and the threads per page
MAX_FORMULA_WORKERS = 8
MAX_FORMULA_THREADS = 64


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
                pass  # a damaged file falls back to the defaults and does not block startup

    @property
    def retention_days(self) -> int:
        try:
            return max(0, min(int(self.data.get("retention_days", DEFAULTS["retention_days"])), MAX_RETENTION_DAYS))
        except (TypeError, ValueError):
            return DEFAULTS["retention_days"]

    @property
    def max_jobs(self) -> int:
        try:
            return max(1, min(int(self.data.get("max_jobs", DEFAULTS["max_jobs"])), MAX_JOBS))
        except (TypeError, ValueError):
            return DEFAULTS["max_jobs"]

    @property
    def layout_model(self) -> dict:
        return self.model("layout_model")

    @property
    def formula_model(self) -> dict:
        return self.model("formula_model")

    def model(self, key: str) -> dict:
        """Settings of layout_model or formula_model."""
        value = self.data.get(key)
        if not isinstance(value, dict):
            value = {}
        default = DEFAULTS[key]
        conf = {"enabled": bool(value.get("enabled", default["enabled"])), "path": str(value.get("path") or "")}
        if "size" in default:
            conf["size"] = value.get("size") if value.get("size") in FORMULA_SIZES else default["size"]
        if "precision" in default:
            conf["precision"] = value.get("precision") if value.get("precision") in PRECISIONS else default["precision"]
        for name, limit in (("workers", MAX_FORMULA_WORKERS), ("threads", MAX_FORMULA_THREADS)):
            if name in default:
                count = value.get(name)
                valid = isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= limit
                conf[name] = count if valid else default[name]
        return conf

    def public(self) -> dict:
        # The two models have their own pages and endpoints (/api/layout-model, /api/formula-model) and are not
        # included here
        return {"retention_days": self.retention_days, "max_jobs": self.max_jobs}

    def update(self, changes: dict) -> dict:
        if "retention_days" in changes:
            try:
                days = int(changes["retention_days"])
            except (TypeError, ValueError):
                raise SettingsError("保留天数必须是整数") from None
            if not 0 <= days <= MAX_RETENTION_DAYS:
                raise SettingsError(f"保留天数须在 0 到 {MAX_RETENTION_DAYS} 之间（0 表示永久保留）")
            with self.lock:
                self.data["retention_days"] = days
                self._save()
        if "max_jobs" in changes:
            count = changes["max_jobs"]
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_JOBS:
                raise SettingsError(f"同时翻译的任务数须为 1 到 {MAX_JOBS} 的整数")
            with self.lock:
                self.data["max_jobs"] = count
                self._save()
        for key, name in (("layout_model", "版面模型"), ("formula_model", "公式检测模型")):
            if key not in changes:
                continue
            value = changes[key]
            if not isinstance(value, dict):
                raise SettingsError(f"{name}设置格式错误")
            merged = {**self.model(key), **value}
            conf = {"enabled": bool(merged.get("enabled")), "path": str(merged.get("path") or "").strip()}
            if "size" in DEFAULTS[key]:
                try:
                    conf["size"] = int(merged.get("size"))
                except (TypeError, ValueError):
                    conf["size"] = 0
                if conf["size"] not in FORMULA_SIZES:
                    raise SettingsError("输入尺寸只能是 " + " 或 ".join(str(size) for size in FORMULA_SIZES))
            if "precision" in DEFAULTS[key]:
                conf["precision"] = merged.get("precision")
                if conf["precision"] not in PRECISIONS:
                    raise SettingsError("精度只能是 " + " 或 ".join(PRECISIONS))
            for name, label, limit in (
                ("workers", "同时推理的页数", MAX_FORMULA_WORKERS),
                ("threads", "每页推理线程数", MAX_FORMULA_THREADS),
            ):
                if name not in DEFAULTS[key]:
                    continue
                count = merged.get(name)
                if count in (None, ""):
                    count = 0
                if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= limit:
                    raise SettingsError(f"{label}须为 0 到 {limit} 的整数（0 表示自动）")
                conf[name] = count
            with self.lock:
                self.data[key] = conf
                self._save()
        return self.public()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
