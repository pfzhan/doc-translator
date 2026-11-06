"""Glossary library, saved in data/glossaries.json: named glossaries (by field, by book series) that a job picks
when it is submitted. Each one holds its entries as text, one "source = translation" per line, the same format as
the glossary of a translation service."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from . import glossary
from .paths import DATA_DIR

DATA_PATH = DATA_DIR / "glossaries.json"
MAX_NAME = 80


class GlossaryError(Exception):
    pass


class GlossaryNotFound(GlossaryError):
    pass


class GlossaryStore:
    def __init__(self, path: Path = DATA_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.items: list[dict] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("glossaries"), list):
                    self.items = [g for g in loaded["glossaries"] if isinstance(g, dict) and g.get("id")]
            except (ValueError, OSError):
                pass  # a damaged file falls back to an empty library and does not block startup

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"glossaries": self.items}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _clean(data: dict) -> dict:
        name = str(data.get("name") or "").strip()
        if not name:
            raise GlossaryError("请填写术语表名称")
        if len(name) > MAX_NAME:
            raise GlossaryError(f"名称不能超过 {MAX_NAME} 个字符")
        # Lines that are not entries are dropped, so what is saved is what translation uses
        entries = glossary.render(glossary.parse(str(data.get("entries") or "")))
        return {"name": name, "entries": entries}

    def _find(self, gid: str) -> dict:
        for g in self.items:
            if g["id"] == gid:
                return g
        raise GlossaryNotFound("术语表不存在")

    def list(self) -> list[dict]:
        with self.lock:
            return [{**g, "count": len(glossary.parse(g.get("entries", "")))} for g in self.items]

    def get(self, gid: str) -> dict | None:
        with self.lock:
            return next((dict(g) for g in self.items if g["id"] == gid), None)

    def create(self, data: dict) -> dict:
        with self.lock:
            item = {"id": uuid.uuid4().hex[:8], **self._clean(data)}
            self.items.append(item)
            self._save()
            return dict(item)

    def update(self, gid: str, data: dict) -> dict:
        with self.lock:
            item = self._find(gid)
            item.update(self._clean(data))
            self._save()
            return dict(item)

    def delete(self, gid: str):
        with self.lock:
            self.items.remove(self._find(gid))
            self._save()

    def entries_text(self, ids: list[str]) -> str:
        """Entries of the glossaries with these ids, in that order. Ids of deleted glossaries are skipped."""
        with self.lock:
            by_id = {g["id"]: g.get("entries", "") for g in self.items}
        return "\n".join(by_id[gid] for gid in ids if by_id.get(gid))
