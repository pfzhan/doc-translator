"""应用设置，保存在 data/settings.json。

- retention_days：翻译记录的保留时间，0 表示永久保留。
- layout_model：PDF 版面模型，{"enabled": 是否使用, "path": 模型文件，空串表示默认位置,
  "precision": 用 int8 还是 fp32}。precision 默认 int8：生成了 int8 文件就用它（见 quantize.py），没生成时用 fp32。
- formula_model：PDF 公式检测模型，格式同上，另有 "size": 输入边长（1024 或 768）。默认开启、用 1024：
  行内公式认得准不准直接影响译文，模型文件下好了就用，没下时只用规则。

可以在页面上改，也可以直接改这个文件后重启服务。
"""
import json
import threading
from pathlib import Path

from .formats.pdf_vision import FORMULA_SIZES
from .quantize import PRECISIONS

PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"
DEFAULTS = {
    "retention_days": 30,
    "layout_model": {"enabled": False, "path": "", "precision": PRECISIONS[0]},
    # workers、threads 是公式检测模型同时推理几页、每页几个线程，0 表示按核数定
    "formula_model": {"enabled": True, "path": "", "size": FORMULA_SIZES[0], "precision": PRECISIONS[0],
                      "workers": 0, "threads": 0},
}
MAX_RETENTION_DAYS = 3650
# 公式检测模型同时推理的页数、每页推理线程数的上限
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
        if "precision" in default:
            conf["precision"] = value.get("precision") if value.get("precision") in PRECISIONS else default["precision"]
        for name, limit in (("workers", MAX_FORMULA_WORKERS), ("threads", MAX_FORMULA_THREADS)):
            if name in default:
                count = value.get(name)
                valid = isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= limit
                conf[name] = count if valid else default[name]
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
                raise SettingsError(f"保留天数须在 0 到 {MAX_RETENTION_DAYS} 之间（0 表示永久保留）")
            with self.lock:
                self.data["retention_days"] = days
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
            for name, label, limit in (("workers", "同时推理的页数", MAX_FORMULA_WORKERS),
                                       ("threads", "每页推理线程数", MAX_FORMULA_THREADS)):
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
