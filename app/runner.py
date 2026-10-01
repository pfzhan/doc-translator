"""批量翻译调度：去重、缓存、切批、并发、进度回调，以及边翻边预览。

并发用固定数量的 worker 从待翻批次里取活：默认按文档顺序，预览里用户滚动或跳转后，
优先取关注位置之后的批次（类似沉浸式翻译“翻译可视区域”的思路，但整本书最终都会翻完）。
"""
import asyncio
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from . import langdetect
from .translators import Translator, TranslatorError

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.sqlite3"


class Cache:
    # 缓存只增不减会无限膨胀，超过上限时删掉最旧的一批
    MAX_ENTRIES = 50000

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
            (n,) = self.conn.execute("SELECT COUNT(*) FROM t").fetchone()
            if n > self.MAX_ENTRIES:
                # rowid 随写入递增（REPLACE 刷新过的条目也算新），删掉最旧的，留一成余量
                self.conn.execute(
                    "DELETE FROM t WHERE rowid IN (SELECT rowid FROM t ORDER BY rowid LIMIT ?)",
                    (n - self.MAX_ENTRIES + self.MAX_ENTRIES // 10,),
                )
            self.conn.commit()


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


class Runner:
    # 第一批只发几段，让预览尽快出现；之后恢复正常批量
    first_batch_items = 4

    def __init__(self, translator: Translator, progress: Callable[[int, int], None] | None = None, cache=None,
                 skip_same_lang: bool = True):
        self.translator = translator
        self.progress = progress or (lambda done, total: None)
        self.cache = cache if cache is not None else get_cache()
        # 已经是目标语言的段落不翻（插件的段落语言检测）
        self.skip_same_lang = skip_same_lang
        # 给任务页展示：检测到的文档语言、跳过的段落数
        self.info = {"detected_lang": "", "skipped": 0}

        # ---- 边翻边预览 ----
        # segments：文档顺序的段落 [{"s": 原文, "k": 类型}]，translate_all(preview=True) 时填充
        self.segments: list[dict] | None = None
        self._indices: dict[str, list[int]] = {}  # 原文 → 出现的段落序号（同一句可能出现多次）
        self._done: dict[str, str] = {}
        self._skipped: set[str] = set()
        # 完成顺序的日志，下标就是版本号；预览接口按 since 增量返回
        self._log: list[str] = []
        # 用户正在看的位置（段落序号），优先翻译它后面的内容
        self._focus: int | None = None

    def set_title(self, title: str):
        """文档标题作为翻译上下文（只有大模型引擎用得上）。"""
        if hasattr(self.translator, "title"):
            self.translator.title = (title or "").strip()[:200]

    def _make_batches(self, texts: list[str]) -> list[list[str]]:
        tr = self.translator
        batches, cur, size = [], [], 0
        for t in texts:
            limit = min(self.first_batch_items, tr.max_batch_items) if not batches else tr.max_batch_items
            if cur and (len(cur) >= limit or size + len(t) > tr.max_batch_chars):
                batches.append(cur)
                cur, size = [], 0
            cur.append(t)
            size += len(t)
        if cur:
            batches.append(cur)
        return batches

    # ---------- 预览 ----------

    def focus(self, index: int):
        """预览里用户滚动或跳转到了第 index 段，之后优先翻译这附近。"""
        self._focus = max(0, int(index))

    def _record(self, texts: list[str]):
        self._log.extend(texts)

    def preview(self, since: int = -1) -> dict:
        """since < 0 时返回全部段落和所有已完成的译文；否则只返回版本 since 之后完成的。"""
        if self.segments is None:
            return {"ready": False, "version": 0}
        start = 0 if since < 0 else min(since, len(self._log))
        updates = []
        for text in self._log[start:]:
            translated = self._done.get(text, text)
            skipped = 1 if text in self._skipped else 0
            for i in self._indices.get(text, ()):
                updates.append([i, translated, skipped])
        out = {"ready": True, "version": len(self._log), "updates": updates, "focus": self._focus}
        if since < 0:
            out["segments"] = self.segments
        return out

    def _next_batch(self, pending: list[tuple[int, list[str]]]):
        """取下一批：有关注位置时取它之后最近的一批，否则按文档顺序。"""
        if not pending:
            return None
        pick = 0
        if self._focus is not None:
            pick = next((n for n, (pos, _) in enumerate(pending) if pos >= self._focus), 0)
        return pending.pop(pick)[1]

    def _detect(self, unique: list[str]) -> set[str]:
        """检测文档语言，返回需要跳过（已是目标语言）的段落。"""
        tr = self.translator
        doc_lang = langdetect.detect_document(unique)
        self.info["detected_lang"] = doc_lang or ""
        if tr.source_lang == "auto" and doc_lang:
            tr.detected_source = doc_lang
        if not self.skip_same_lang:
            return set()
        return {t for t in unique if langdetect.same_language(langdetect.detect(t), tr.target_lang)}

    async def translate_all(self, texts: list[str], kinds: list[str] | None = None,
                            preview: bool = False) -> list[str]:
        """按原顺序返回译文；空白文本和已是目标语言的段落原样返回。

        preview=True 时记录段落供预览接口读取；kinds 是每段的类型（h1~h6 / p / li / quote / td / toc），
        只影响预览的显示样式。
        """
        unique = list(dict.fromkeys(t for t in texts if t.strip()))
        skipped = await asyncio.to_thread(self._detect, unique)
        self.info["skipped"] = len(skipped)
        unique = [t for t in unique if t not in skipped]
        prefix = self.translator.cache_key
        done_map = self.cache.get_many(prefix, unique)
        # 旧版缓存 key 里带了模型和提示词，读不到时用旧 key 再查一次，查到的迁移到新 key
        legacy = getattr(self.translator, "legacy_cache_key", None)
        if legacy and legacy != prefix:
            old = self.cache.get_many(legacy, [t for t in unique if t not in done_map])
            if old:
                self.cache.put_many(prefix, old)
                done_map.update(old)
        todo = [t for t in unique if t not in done_map]

        # 每段原文第一次出现的位置，用来按“离关注位置的远近”挑批次
        first_pos: dict[str, int] = {}
        for i, t in enumerate(texts):
            first_pos.setdefault(t, i)

        if preview:
            self._indices = {}
            for i, t in enumerate(texts):
                if t.strip():
                    self._indices.setdefault(t, []).append(i)
            self._done = done_map
            self._skipped = skipped
            self._log = []
            self.segments = [{"s": t, "k": (kinds[i] if kinds else "p")} for i, t in enumerate(texts)]
            # 已是目标语言的和缓存命中的，预览里直接显示
            self._record([t for t in dict.fromkeys(texts) if t in skipped or t in done_map])

        total = len(unique)
        done = len(done_map)
        self.progress(done, total)

        pending = [(first_pos[b[0]], b) for b in self._make_batches(todo)]

        async def worker():
            nonlocal done
            while (batch := self._next_batch(pending)) is not None:
                try:
                    result = await self.translator.translate_batch(batch)
                except BaseException:
                    pending.clear()  # 出错时让其他并发任务也尽快停下
                    raise
                pairs = dict(zip(batch, result))
                # 空译文不写缓存、不计入完成（否则会永久缓存空结果），按失败处理
                good = {s: d for s, d in pairs.items() if d and d.strip()}
                empty = [s for s in batch if s not in good]
                if good:
                    self.cache.put_many(prefix, good)
                    done_map.update(good)
                    if preview:
                        self._record(list(good))
                    done += len(good)
                    self.progress(done, total)
                if empty:
                    pending.clear()  # 让其他并发任务也尽快停下
                    raise TranslatorError(f"翻译服务返回了空译文（{len(empty)} 段），请重试或更换服务")

        workers = min(self.translator.concurrency, len(pending))
        await asyncio.gather(*(worker() for _ in range(workers)))
        return [done_map.get(t, t) if t.strip() else t for t in texts]
