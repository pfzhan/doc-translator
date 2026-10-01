"""批量翻译调度：去重、缓存、切批、并发、进度回调。"""
import asyncio
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from .translators import Translator

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.sqlite3"


class Cache:
    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    @staticmethod
    def _key(prefix: str, text: str) -> str:
        return hashlib.sha1(f"{prefix}\n{text}".encode()).hexdigest()

    def get_many(self, prefix: str, texts: list[str]) -> dict[str, str]:
        out = {}
        with self.lock:
            for t in texts:
                row = self.conn.execute("SELECT v FROM t WHERE k=?", (self._key(prefix, t),)).fetchone()
                if row:
                    out[t] = row[0]
        return out

    def put_many(self, prefix: str, pairs: dict[str, str]):
        with self.lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO t (k, v) VALUES (?, ?)",
                [(self._key(prefix, s), d) for s, d in pairs.items()],
            )
            self.conn.commit()


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


class Runner:
    def __init__(self, translator: Translator, progress: Callable[[int, int], None] | None = None, cache=None):
        self.translator = translator
        self.progress = progress or (lambda done, total: None)
        self.cache = cache if cache is not None else get_cache()

    def set_title(self, title: str):
        """文档标题作为翻译上下文（只有大模型引擎用得上）。"""
        if hasattr(self.translator, "title"):
            self.translator.title = (title or "").strip()[:200]

    def _make_batches(self, texts: list[str]) -> list[list[str]]:
        tr = self.translator
        batches, cur, size = [], [], 0
        for t in texts:
            if cur and (len(cur) >= tr.max_batch_items or size + len(t) > tr.max_batch_chars):
                batches.append(cur)
                cur, size = [], 0
            cur.append(t)
            size += len(t)
        if cur:
            batches.append(cur)
        return batches

    async def translate_all(self, texts: list[str]) -> list[str]:
        """按原顺序返回译文；空白文本原样返回。"""
        unique = list(dict.fromkeys(t for t in texts if t.strip()))
        prefix = self.translator.cache_key
        done_map = self.cache.get_many(prefix, unique)
        todo = [t for t in unique if t not in done_map]

        total = len(unique)
        done = len(done_map)
        self.progress(done, total)

        sem = asyncio.Semaphore(self.translator.concurrency)

        async def run(batch):
            nonlocal done
            async with sem:
                result = await self.translator.translate_batch(batch)
            pairs = dict(zip(batch, result))
            self.cache.put_many(prefix, pairs)
            done_map.update(pairs)
            done += len(batch)
            self.progress(done, total)

        await asyncio.gather(*(run(b) for b in self._make_batches(todo)))
        return [done_map.get(t, t) if t.strip() else t for t in texts]
