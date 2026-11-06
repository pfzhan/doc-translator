"""Batch translation scheduling: deduplication, cache, batching, concurrency, progress callbacks, and preview
during translation.

A fixed number of workers take batches from the pending ones: in document order by default; after the user
scrolls or jumps in the preview, batches after the focus position come first (similar to the "翻译可视区域"
approach of Immersive Translate, though the whole book is still translated in the end).
"""

import asyncio
import hashlib
import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from . import background, glossary, langdetect
from .paths import DATA_DIR
from .prompts import drop_added_braces, drop_invented_marks, strip_control_marks, tidy_marks
from .translators import Translator, TranslatorError, misplaced, stray_chars, typical_ratio

CACHE_PATH = DATA_DIR / "cache.sqlite3"
# Owner of web page translations in the cache's owner table, where documents are recorded by the digest of their
# content (a hex string, so the two cannot collide)
WEB_CACHE_OWNER = "web"
# Documents with fewer candidate terms than this (short texts, dialogues) are not worth an extra request
_MIN_TERMS = 3


class Cache:
    # A cache that only grows would bloat without limit; above this limit the oldest entries are deleted
    MAX_ENTRIES = 50000

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, v TEXT)")
        # ok = 1: the translation passed the checks for shifted batches (see Runner.translate_all), so it is not
        # checked against the rest of the document again. Storing a translation again resets it
        if "ok" not in {row[1] for row in self.conn.execute("PRAGMA table_info(t)")}:
            self.conn.execute("ALTER TABLE t ADD COLUMN ok INTEGER NOT NULL DEFAULT 0")
        # Which files (digests of file content) the entries belong to. Keys are hashes that do not show which file
        # they came from, so cleaning up a job deletes by this table
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS owner (doc TEXT NOT NULL, k TEXT NOT NULL, PRIMARY KEY (doc, k)) WITHOUT ROWID"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS owner_k ON owner (k)")
        self.lock = threading.Lock()

    @staticmethod
    def _key(prefix: str, text: str) -> str:
        return hashlib.sha1(f"{prefix}\n{text}".encode()).hexdigest()

    @classmethod
    def keys(cls, prefix: str, texts) -> set[str]:
        return {cls._key(prefix, text) for text in texts}

    def claim(self, doc: str, prefix: str, texts: list[str]) -> None:
        """Record that file doc uses these entries. Recorded on both write and hit, so an entry stored earlier gets
        an owner once it is used again."""
        if not doc or not texts:
            return
        with self.lock:
            self.conn.executemany(
                "INSERT OR IGNORE INTO owner (doc, k) VALUES (?, ?)", [(doc, key) for key in self.keys(prefix, texts)]
            )
            self.conn.commit()

    def owned(self, doc: str, extra: set[str] = frozenset()) -> set[str]:
        """Entries owned by file doc that are still in the cache. extra holds keys computed from other clues; only
        those present in the cache are kept."""
        with self.lock:
            found = {
                k
                for (k,) in self.conn.execute(
                    "SELECT owner.k FROM owner JOIN t ON t.k = owner.k WHERE owner.doc=?", (doc,)
                )
            }
            found |= {k for k in extra - found if self.conn.execute("SELECT 1 FROM t WHERE k=?", (k,)).fetchone()}
        return found

    def purge(self, doc: str, keys: set[str]) -> int:
        """Delete these entries and the records of file doc, and return the number of entries deleted. Records of
        other files for these entries are deleted too, so no dangling references remain."""
        with self.lock:
            rows = [(k,) for k in keys]
            removed = self.conn.executemany("DELETE FROM t WHERE k=?", rows).rowcount
            self.conn.executemany("DELETE FROM owner WHERE k=?", rows)
            self.conn.execute("DELETE FROM owner WHERE doc=?", (doc,))
            self.conn.commit()
        return removed

    def count_only(self, doc: str) -> int:
        """How many entries in the cache are used by doc and by no other file."""
        with self.lock:
            (n,) = self.conn.execute(_ONLY_OWNED_COUNT, (doc,)).fetchone()
        return int(n)

    def purge_only(self, doc: str) -> int:
        """Delete the entries used by doc alone and forget doc's records, and return the number deleted. Entries
        other files use too stay for them."""
        with self.lock:
            keys = [(k,) for (k,) in self.conn.execute(_ONLY_OWNED, (doc,))]
            removed = self.conn.executemany("DELETE FROM t WHERE k=?", keys).rowcount
            self.conn.execute("DELETE FROM owner WHERE doc=?", (doc,))
            self.conn.commit()
        return removed

    def unused(self, used_docs: set[str]) -> tuple[int, int]:
        """(entries whose owners are all outside used_docs, entries with no owner at all). Entries saved before
        owners were registered have none, and nothing tells whose they are."""
        with self.lock:
            self._fill_used(used_docs)
            (unused,) = self.conn.execute(f"SELECT COUNT(*) FROM t WHERE {_UNUSED}").fetchone()
            (untracked,) = self.conn.execute("SELECT COUNT(*) FROM t WHERE k NOT IN (SELECT k FROM owner)").fetchone()
        return int(unused), int(untracked)

    def purge_unused(self, used_docs: set[str]) -> int:
        """Delete the entries whose owners are all outside used_docs, with their owner records; entries with no owner
        stay. Return the number deleted."""
        with self.lock:
            self._fill_used(used_docs)
            removed = self.conn.execute(f"DELETE FROM t WHERE {_UNUSED}").rowcount
            self.conn.execute("DELETE FROM owner WHERE k NOT IN (SELECT k FROM t)")
            self.conn.commit()
        return int(removed)

    def _fill_used(self, used_docs: set[str]) -> None:
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS used (doc TEXT PRIMARY KEY)")
        self.conn.execute("DELETE FROM used")
        self.conn.executemany("INSERT OR IGNORE INTO used (doc) VALUES (?)", [(d,) for d in used_docs])

    def get_many(self, prefix: str, texts: list[str]) -> dict[str, str]:
        out = {}
        with self.lock:
            for t in texts:
                row = self.conn.execute("SELECT v FROM t WHERE k=?", (self._key(prefix, t),)).fetchone()
                if row:
                    out[t] = row[0]
        return out

    def verified(self, prefix: str, texts: list[str]) -> set[str]:
        """The texts whose stored translation has passed the checks (see mark_verified)."""
        out = set()
        with self.lock:
            for t in texts:
                row = self.conn.execute("SELECT ok FROM t WHERE k=?", (self._key(prefix, t),)).fetchone()
                if row and row[0]:
                    out.add(t)
        return out

    def mark_verified(self, prefix: str, texts: list[str]) -> None:
        """Record that the stored translations of these texts passed the checks for shifted batches."""
        if not texts:
            return
        with self.lock:
            self.conn.executemany("UPDATE t SET ok=1 WHERE k=?", [(self._key(prefix, t),) for t in texts])
            self.conn.commit()

    def drop_many(self, prefix: str, texts: list[str]) -> None:
        """Drop cache entries that should not be kept. A whole English sentence echoed back must be translated
        again next time."""
        if not texts:
            return
        rows = [(self._key(prefix, text),) for text in texts]
        with self.lock:
            self.conn.executemany("DELETE FROM t WHERE k=?", rows)
            self.conn.executemany("DELETE FROM owner WHERE k=?", rows)
            self.conn.commit()

    def put_many(self, prefix: str, pairs: dict[str, str]):
        with self.lock:
            (last,) = self.conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM t").fetchone()
            self.conn.executemany(
                "INSERT OR REPLACE INTO t (k, v) VALUES (?, ?)",
                [(self._key(prefix, s), d) for s, d in pairs.items()],
            )
            (n,) = self.conn.execute("SELECT COUNT(*) FROM t").fetchone()
            if n > self.MAX_ENTRIES:
                # Leave 10% headroom
                self._evict(n - self.MAX_ENTRIES + self.MAX_ENTRIES // 10, last)
            self.conn.commit()

    def _evict(self, count: int, last: int) -> None:
        """Delete count entries, oldest first (rowid increases with each write; entries refreshed by REPLACE count
        as new). Entries no document uses go first: web page translations and entries saved before owners were
        recorded. Otherwise browsing could push a book's translations out, and continuing it or generating its other
        version would have to translate (and pay for) them again. The entries this write just stored have rowids
        above last and have no owner yet, so the first round leaves them alone."""
        removed = self.conn.execute(
            "DELETE FROM t WHERE rowid IN (SELECT rowid FROM t WHERE rowid <= ? AND NOT EXISTS"
            " (SELECT 1 FROM owner o WHERE o.k = t.k AND o.doc != ?) ORDER BY rowid LIMIT ?)",
            (last, WEB_CACHE_OWNER, count),
        ).rowcount
        if removed < count:
            self.conn.execute(
                "DELETE FROM t WHERE rowid IN (SELECT rowid FROM t ORDER BY rowid LIMIT ?)", (count - removed,)
            )
        self.conn.execute("DELETE FROM owner WHERE k NOT IN (SELECT k FROM t)")


_cache: Cache | None = None

# Entries in the cache that doc uses and no other owner does
_ONLY_OWNED = """
    SELECT o.k FROM owner o JOIN t ON t.k = o.k
    WHERE o.doc = ? AND NOT EXISTS (SELECT 1 FROM owner other WHERE other.k = o.k AND other.doc != o.doc)
"""
_ONLY_OWNED_COUNT = f"SELECT COUNT(*) FROM ({_ONLY_OWNED})"
# Entries that have owners, none of them in the temporary table used (see Cache.unused)
_UNUSED = """
    k IN (SELECT k FROM owner)
    AND NOT EXISTS (SELECT 1 FROM owner o JOIN used u ON u.doc = o.doc WHERE o.k = t.k)
"""


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


class Runner:
    # The first batch sends only a few paragraphs so that the preview appears quickly; later batches are normal
    # size
    first_batch_items = 4

    def __init__(
        self,
        translator: Translator,
        progress: Callable[[int, int], None] | None = None,
        cache=None,
        skip_same_lang: bool = True,
        cache_only: bool = False,
    ):
        self.translator = translator
        self.progress = progress or (lambda done, total: None)
        self.cache = cache if cache is not None else get_cache()
        # All reads and writes go through it, which also records which file the entries belong to
        self.owned_cache = _OwnedCache(self.cache, lambda: self.document)
        # Paragraphs already in the target language are not translated (the plugin's paragraph language detection)
        self.skip_same_lang = skip_same_lang
        # Typeset only, without requests to the translation service: fail at once when the cache is missing
        # entries, so that nothing is silently retranslated with another configuration
        self.cache_only = cache_only
        # Shown on the job page: detected document language, number of skipped paragraphs
        self.info = {"detected_lang": "", "skipped": 0}
        # Texts the last translate_all kept because they are already in the target language
        self.same_lang: set[str] = set()
        # How far preparation before translation and writing after translation have got (see report_stage)
        self.stage = {"name": "", "done": 0, "total": 0, "unit": ""}
        # Digest of the file content (set_document); the glossary is cached by it
        self.document = ""
        # Preview translations overridden by paragraph index (heading rewrites, deduplication at paragraph
        # boundaries); the same sentence may differ at different positions
        self._index_overrides: dict[int, str] = {}
        # Cache used by the last translate_all: keys include each paragraph's note (see _NotedCache)
        self.keyed_cache = None

        # ---- Preview during translation ----
        # segments: paragraphs in document order [{"s": source text, "k": kind}], filled by
        # translate_all(preview=True)
        self.segments: list[dict] | None = None
        self._indices: dict[str, list[int]] = {}  # source text -> its paragraph indices (may occur several times)
        self._done: dict[str, str] = {}
        self._skipped: set[str] = set()
        self._skip_idx: set[int] = set()
        # Log in order of completion; the index is the version number. The preview API returns increments by since
        self._log: list[str] = []
        # Position the user is viewing (paragraph index); content after it is translated first
        self._focus: int | None = None

    def set_document(self, digest: str):
        """The file being translated (content digest). The same file keeps the glossary fixed the first time."""
        self.document = digest

    def set_title(self, title: str):
        """Document title as translation context (used only by LLM engines)."""
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

    # ---------- Preview ----------

    def focus(self, index: int):
        """The user scrolled or jumped to paragraph index in the preview; translate around it first from now on."""
        self._focus = max(0, int(index))

    def _record(self, texts: list[str]):
        self._log.extend(texts)

    def report_stage(self, name: str, done: int = 0, total: int = 0, unit: str = "") -> None:
        """Report how far preparation before translation or writing after translation has got; unit is the unit
        counted (pages, paragraphs). Layout analysis and writing of a whole book each take several minutes, and
        the job page shows progress from this. Safe to call from a thread: it only replaces one dict as a whole.
        When called in a thread started by background.run, a paused job stops here (raises background.Stopped)."""
        self.stage = {"name": name, "done": done, "total": total, "unit": unit}
        background.check_stop()

    def preview(self, since: int = -1) -> dict:
        """With since < 0, return all paragraphs and all finished translations; otherwise only those finished
        after version since."""
        if self.segments is None:
            return {"ready": False, "version": 0}
        start = 0 if since < 0 else min(since, len(self._log))
        updates = []
        for text in self._log[start:]:
            translated = self._done.get(text, text)
            for i in self._indices.get(text, ()):
                # References are skipped by position: the same sentence in the body text still shows the
                # translation
                if i in self._skip_idx or text in self._skipped:
                    updates.append([i, text, 1])
                else:
                    updates.append([i, self._index_overrides.get(i, translated), 0])
        out = {"ready": True, "version": len(self._log), "updates": updates, "focus": self._focus}
        if since < 0:
            out["segments"] = self.segments
        return out

    def _next_batch(self, pending: list[tuple[int, list[str]]]):
        """Take the next batch: the nearest one after the focus position if there is one, otherwise in document
        order."""
        if not pending:
            return None
        pick = 0
        if self._focus is not None:
            pick = next((n for n, (pos, _) in enumerate(pending) if pos >= self._focus), 0)
        return pending.pop(pick)[1]

    def _detect(self, unique: list[str]) -> set[str]:
        """Detect the document language and return the paragraphs to skip (already in the target language)."""
        tr = self.translator
        doc_lang = langdetect.detect_document(unique)
        self.info["detected_lang"] = doc_lang or ""
        if tr.source_lang == "auto" and doc_lang:
            tr.detected_source = doc_lang
        if not self.skip_same_lang:
            return set()
        return {t for t in unique if langdetect.same_language(langdetect.detect(t), tr.target_lang)}

    async def _prepare_glossary(self, texts: list[str]) -> None:
        """Find terms in the full text and ask the model to fix their translations once; every batch is then
        translated with them.

        The result goes into the cache: resuming, retrying or generating the other version of the same document
        uses the same table, so that the paragraph cache still matches. When the file is known, the table is looked
        up by file content, so the original table is kept even when the extraction rules change and the candidate
        list changes with them; otherwise a changed glossary would make the paragraph cache of the whole document
        miss. When the file is unknown, or the table was stored before this change, it is looked up by the
        candidate list. When only typesetting, no request is made, and without a cached table no glossary is used.
        A failure to fix the translations does not affect translation; this run just has no glossary.
        """
        tr = self.translator
        if not getattr(tr, "auto_glossary", False) or not hasattr(tr, "define_terms"):
            return
        tr.doc_glossary = []
        terms = glossary.candidates(texts)
        if len(terms) < _MIN_TERMS:
            return
        listing = "\n".join(terms)
        prefix = tr.cache_key + ":terms"
        keys = ([f"file:{self.document}"] if self.document else []) + [listing]
        cached = self.owned_cache.get_many(prefix, keys)
        found = next((cached[key] for key in keys if key in cached), None)
        if found is None:
            if self.cache_only:
                return
            excerpt = glossary.excerpt(texts)
            try:
                found = glossary.render(await tr.define_terms(terms, excerpt))
            except TranslatorError:
                # If the service itself has a problem, the following translation requests will report it
                return
            self.owned_cache.put_many(prefix, {key: found for key in keys})
        elif keys[0] not in cached:
            self.owned_cache.put_many(prefix, {keys[0]: found})
        tr.doc_glossary = glossary.parse(found)
        self.info["terms"] = len(tr.doc_glossary)

    async def translate_all(
        self,
        texts: list[str],
        kinds: list[str] | None = None,
        preview: bool = False,
        html: list[bool] | None = None,
        skip: list[bool] | None = None,
        notes: list[str] | None = None,
    ) -> list[str]:
        """Return translations in the original order; blank texts and paragraphs already in the target language
        are returned unchanged.

        With preview=True the paragraphs are recorded for the preview API; kinds is the kind of each paragraph
        (h1~h6 / p / li / quote / td / toc) and affects only the display style of the preview. html is aligned
        with texts and marks which paragraphs are e-book rich-text fragments.
        skip is aligned with texts and marks paragraphs where the caller asks to keep the original text (such as
        a references section); the preview marks them as skipped.
        The flag applies by position: the same sentence is still translated where it occurs in the body text.
        notes is aligned with texts and holds notes on where each paragraph appears in the document (which entry
        a subentry of an index belongs to); they go to the model with the batch to choose word senses.
        A sentence may be translated differently with and without a note, so the cache stores by source text plus
        note. When a sentence occurs several times, the first note is used.
        """
        n = len(texts)
        # The preparation stage ends here. Progress from now on is the number of translated paragraphs; the format
        # reports its own writing stage later
        self.report_stage("")
        skip_flags = list(skip or [])
        if len(skip_flags) < n:
            skip_flags.extend([False] * (n - len(skip_flags)))
        else:
            skip_flags = skip_flags[:n]
        html_flags = list(html or [])
        if len(html_flags) < n:
            html_flags.extend([False] * (n - len(html_flags)))
        # Only paragraphs flagged by the caller are sent as HTML; a <Note> in body text is not a tag
        unique = list(dict.fromkeys(t for t in texts if t.strip()))
        lang_skipped = await asyncio.to_thread(self._detect, unique)
        self.same_lang = lang_skipped

        def kept(i: int) -> bool:
            text = texts[i]
            return bool(text.strip()) and (skip_flags[i] or text in lang_skipped)

        self.info["skipped"] = sum(1 for i in range(n) if kept(i))
        self.translator.html_texts = {texts[i] for i in range(n) if html_flags[i] and texts[i].strip() and not kept(i)}
        need = list(dict.fromkeys(texts[i] for i in range(n) if texts[i].strip() and not kept(i)))
        note_of: dict[str, str] = {}
        for text, note in zip(texts, notes or []):
            if note and text not in note_of:
                note_of[text] = note
        self.translator.notes = note_of

        def key(text: str) -> str:
            return noted_key(text, note_of.get(text, ""))

        # Translations that the format retranslates itself after translation (see
        # pdf_restore._retry_lost_placeholders) are also written back with this key
        self.keyed_cache = cache = _NotedCache(self.owned_cache, key)
        # The glossary must be fixed before the cache keys are computed: it is part of the key
        await self._prepare_glossary(need)
        prefix = self.translator.cache_key
        tl = self.translator.target_lang
        done_map = cache.get_many(prefix, need)
        # If a whole English sentence echoed back got into the cache, the next re-layout would still be English.
        # Drop it so it is translated again this time.
        echoed = [src for src, dst in done_map.items() if _is_prose_echo(src, dst)]
        # Translations that were shifted to the wrong paragraph or carried foreign text in earlier batch
        # translation also got into the cache; drop them and translate again too. When only typesetting they cannot
        # be retranslated and are kept.
        # Other paragraphs in a shifted batch were most likely shifted by one as well, so paragraphs within one
        # batch length before and after are dropped too.
        # Only translations not checked before are compared with the rest of the document: those stored by a batch
        # that passed the engine's own checks, or that passed here once, are left alone. A long but correct
        # translation would otherwise be dropped on every run, and its neighbors with it, paid for again each time
        trusted = cache.verified(prefix, list(done_map))
        if not self.cache_only:
            unchecked = set(done_map) - trusted
            typical = typical_ratio(done_map.items())
            shifted = {src for src in unchecked if src not in echoed and misplaced(src, done_map[src], typical)}
            shifted |= {
                src for src in _batch_around(need, shifted, self.translator.max_batch_chars) if src in unchecked
            }
            echoed += [
                src
                for src, dst in done_map.items()
                if src not in echoed and (src in shifted or stray_chars(src, dst, tl))
            ]
        if echoed:
            done_map = {src: dst for src, dst in done_map.items() if src not in set(echoed)}
            if callable(getattr(self.cache, "drop_many", None)):
                cache.drop_many(prefix, echoed)
        headings = {t for t, k in zip(texts, kinds or []) if k in _HEADING_KINDS}
        html_sources = self.translator.html_texts

        def fixed(src: str, dst: str) -> str:
            dst = clean_translation(src, dst, tl)
            if src not in headings:
                return dst
            return fix_numbered_unit(dst, tl, html=src in html_sources)

        # Old cache entries for headings and contents may still read "第2卷", and translations may still carry
        # full-width parentheses/commas or a leaked [[END]]: rewrite them on a hit, so the preview needs no
        # retranslation
        rewrites = {t: d2 for t, d in done_map.items() if (d2 := fixed(t, d)) != d}
        if rewrites:
            cache.put_many(prefix, rewrites)
            done_map.update(rewrites)
        # Hits that passed the checks above are not checked again next time. Storing a rewrite cleared the flag, so
        # it is set again for rewrites of entries that had it. When only typesetting nothing was checked
        passed = set() if self.cache_only else set(done_map)
        cache.mark_verified(prefix, list((passed - trusted) | (rewrites.keys() & trusted)))
        todo = [t for t in need if t not in done_map]
        if self.cache_only and todo:
            raise ValueError("部分段落不在翻译缓存中，无法仅重新排版，请重新翻译。")

        # Position of the first occurrence of each source text, used to pick batches by distance from the focus
        # position
        first_pos: dict[str, int] = {}
        for i, t in enumerate(texts):
            if t.strip() and not kept(i):
                first_pos.setdefault(t, i)

        if preview:
            self._indices = {}
            for i, t in enumerate(texts):
                if t.strip():
                    self._indices.setdefault(t, []).append(i)
            self._done = done_map
            self._index_overrides = {}
            self._skipped = lang_skipped
            self._skip_idx = {i for i in range(n) if skip_flags[i]}
            self._log = []
            self.segments = [{"s": t, "k": (kinds[i] if kinds else "p")} for i, t in enumerate(texts)]
            # Paragraphs already in the target language, skipped whole by the caller, or found in the cache are shown
            # in the preview at once. When a sentence is skipped only at some positions, each position is marked
            # once the translation arrives.
            immediate: list[str] = []
            seen: set[str] = set()
            for t in texts:
                if not t.strip() or t in seen:
                    continue
                seen.add(t)
                indexes = self._indices[t]
                if all(kept(i) for i in indexes) or t in done_map:
                    immediate.append(t)
            self._record(immediate)

        total = len(need)
        done = len(done_map)
        self.progress(done, total)
        failed: list[str] = []  # Empty results: fall back to the source text; fail only above a threshold

        pending = [(first_pos[b[0]], b) for b in self._make_batches(todo)]

        async def worker():
            nonlocal done
            while (batch := self._next_batch(pending)) is not None:
                try:
                    result = await self.translator.translate_batch(batch)
                except BaseException:
                    pending.clear()  # On error, make the other concurrent workers stop soon too
                    raise
                pairs = dict(zip(batch, result))
                # Echo detection (LLM engines only): a long paragraph whose translation is still detected as the source
                # language was not translated (common in batch mode). Retranslate each such paragraph once; if the
                # retry is still in the source language, the first result is kept (no worse for proper nouns and code)
                from .translators import LLMTranslator

                src_lang = getattr(self.translator, "detected_source", "") or getattr(
                    self.translator, "source_lang", ""
                )
                if (
                    isinstance(self.translator, LLMTranslator)
                    and src_lang
                    and src_lang != self.translator.target_lang
                    and src_lang != "auto"
                ):
                    echoes = [
                        s
                        for s, d in pairs.items()
                        if d and d.strip() and len(d) > 20 and langdetect.detect(d) == src_lang
                    ]
                    if echoes:
                        retried = await asyncio.gather(*(self.translator.translate_batch([s]) for s in echoes))
                        for s, (rt,) in zip(echoes, retried):
                            if rt and rt.strip() and langdetect.detect(rt) != src_lang:
                                pairs[s] = rt
                # Empty translations are not cached and not counted as done (otherwise an empty result would be
                # cached for good); they are treated as failures.
                # "第2卷" in headings and contents is changed to Chinese numerals before storing
                good = {s: fixed(s, d) for s, d in pairs.items() if d and d.strip() and not _is_prose_echo(s, d)}
                empty = [s for s in batch if s not in good]
                if good:
                    cache.put_many(prefix, good)
                    # The engine already checked these against the rest of their batch
                    cache.mark_verified(prefix, list(good))
                    done_map.update(good)
                    if preview:
                        self._record(list(good))
                    done += len(good)
                if empty:
                    # Fallback policy (the same as BabelDOC): a few empty translations keep the original text instead
                    # of failing the whole book, but when all fail, or failures exceed the threshold, this must not
                    # return as success
                    failed.extend(empty)
                    done += len(empty)
                self.progress(done, total)
                # When the API returns immediately (mock, local model), this loop would never yield the event loop
                # and the server would not respond during the whole translation
                await asyncio.sleep(0)

        workers = min(self.translator.concurrency, len(pending))
        await asyncio.gather(*(worker() for _ in range(workers)))
        self.info["failed"] = len(failed)
        if failed and (len(failed) == total or len(failed) > max(3, total // 10)):
            raise TranslatorError(f"翻译服务持续返回空译文（{len(failed)} / {total} 段），请重试或更换服务")
        return [texts[i] if not texts[i].strip() or kept(i) else done_map.get(texts[i], texts[i]) for i in range(n)]


def _batch_around(order: list[str], flagged: set[str], limit: int) -> set[str]:
    """Paragraphs in order within one batch length (limit characters) before and after each flagged
    paragraph."""
    out: set[str] = set()
    for index, text in enumerate(order):
        if text not in flagged:
            continue
        for step in (-1, 1):
            size, other = 0, index + step
            while 0 <= other < len(order) and size + len(order[other]) <= limit:
                size += len(order[other])
                out.add(order[other])
                other += step
    return out


def noted_key(text: str, note: str) -> str:
    """The cache key for a paragraph: a paragraph with a note includes the note, so the same sentence can be
    translated differently at different positions."""
    return f"{text}\n\n[note] {note}" if note else text


class _OwnedCache:
    """Forward to the underlying cache, recording stored and hit entries under the current file (doc() is the
    digest of the file content). When the underlying cache does not support recording (the in-memory cache in
    tests) or the file is unknown, it only forwards."""

    def __init__(self, cache, doc: Callable[[], str]):
        self.cache = cache
        self.doc = doc

    def _claim(self, prefix: str, texts: list[str]) -> None:
        claim = getattr(self.cache, "claim", None)
        if callable(claim):
            claim(self.doc(), prefix, texts)

    def get_many(self, prefix: str, texts: list[str]) -> dict[str, str]:
        found = self.cache.get_many(prefix, texts)
        self._claim(prefix, list(found))
        return found

    def put_many(self, prefix: str, pairs: dict[str, str]) -> None:
        self.cache.put_many(prefix, pairs)
        self._claim(prefix, list(pairs))

    def drop_many(self, prefix: str, texts: list[str]) -> None:
        self.cache.drop_many(prefix, texts)

    def verified(self, prefix: str, texts: list[str]) -> set[str]:
        """Caches without the flag (the in-memory cache in tests) treat every hit as unchecked."""
        verified = getattr(self.cache, "verified", None)
        return verified(prefix, texts) if callable(verified) else set()

    def mark_verified(self, prefix: str, texts: list[str]) -> None:
        mark = getattr(self.cache, "mark_verified", None)
        if callable(mark):
            mark(prefix, texts)


class _NotedCache:
    """Store and fetch translations by source text, with the key replaced by key(source): a paragraph with a note
    includes the note in its key."""

    def __init__(self, cache, key: Callable[[str], str]):
        self.cache = cache
        self.key = key

    def get_many(self, prefix: str, texts: list[str]) -> dict[str, str]:
        keys = {text: self.key(text) for text in texts}
        found = self.cache.get_many(prefix, list(dict.fromkeys(keys.values())))
        return {text: found[k] for text, k in keys.items() if k in found}

    def put_many(self, prefix: str, pairs: dict[str, str]) -> None:
        self.cache.put_many(prefix, {self.key(text): value for text, value in pairs.items()})

    def drop_many(self, prefix: str, texts: list[str]) -> None:
        self.cache.drop_many(prefix, [self.key(text) for text in texts])

    def verified(self, prefix: str, texts: list[str]) -> set[str]:
        keys = {text: self.key(text) for text in texts}
        found = self.cache.verified(prefix, list(dict.fromkeys(keys.values())))
        return {text for text, k in keys.items() if k in found}

    def mark_verified(self, prefix: str, texts: list[str]) -> None:
        self.cache.mark_verified(prefix, [self.key(text) for text in texts])


_CN_DIGITS = "零一二三四五六七八九"
_HEADING_KINDS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "toc"})
_PARENS = str.maketrans("（）", "()")
# Full stops, title marks and quotation marks stay full-width; other full-width punctuation becomes half-width
# followed by a space (an existing space is not doubled)
_PUNCT_HALF = {"，": ",", "、": ",", "；": ";", "：": ":", "？": "?", "！": "!"}
_PUNCT_RE = re.compile(r"([，、；：？！])\s*")
# Dotted numbers and decimals in the source text: 7.6, 16.4.3, 0.5
_DOTTED_RE = re.compile(r"(?<![\d.])\d{1,4}(?:\.\d{1,4})+(?![\d.])")
_CN_NUMERALS = "一二三四五六七八九十百千零〇"
# "第2卷", "第２卷", and "第卷26" garbled by the model. The units exclude page/issue/number ("页/期/号"), which
# keep Arabic digits
_UNIT_HEAD_RE = re.compile(r"^第([0-9０-９]{1,4})(?=[卷章部篇集册])")
_UNIT_INVERTED_RE = re.compile(r"^第([卷章部篇集册])([0-9０-９]{1,4})")
# "第25卷第1期", "第3页", "第5号": volume, issue, page and number references are not chapter headings
_CITE_RE = re.compile(r"第[0-9０-９]{1,4}\s*[期页号]|[卷章部篇集册]\s*第?\s*[0-9０-９]{1,4}\s*期")
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")
_NUM_CHARS = frozenset("第0123456789０１２３４５６７８９卷章部篇集册")


# An ellipsis with list separators or operators on both sides is a mathematical ellipsis (T1, T2, …, Tn;
# x1 + … + xn) and is written once. The Chinese "……" ends a sentence or utterance and never sits between two
# commas. Style markers and placeholders may come in between
_MARKS = r"(?:\{[^{}\s]{1,10}\}|\x01[^\x02]{0,8}\x02)*"
_LIST_SEP = r"[,，、;；+\-=<>≤≥×·]"
_LIST_ELLIPSIS_RE = re.compile(rf"({_LIST_SEP}[ \t]*{_MARKS}[ \t]*)(?:……|⋯⋯|\.{{6}})({_MARKS}[ \t]*{_LIST_SEP})")


def normalize_punctuation(dst: str, target_lang: str) -> str:
    """Make full-width punctuation in Chinese translations half-width: parentheses are replaced directly; commas,
    enumeration commas, semicolons, colons, question marks and exclamation marks become half-width followed by a
    space. Full stops, title marks and quotation marks stay full-width, since they are part of Chinese typesetting,
    while full-width , ; : ? ! are as wide as a Chinese character and look wide and clumsy in body text."""
    if not target_lang.startswith("zh"):
        return dst
    out = _PUNCT_RE.sub(lambda m: _PUNCT_HALF[m.group(1)] + " ", dst.translate(_PARENS))
    out = _LIST_ELLIPSIS_RE.sub(r"\1…\2", out)
    return out.rstrip() if out.endswith(" ") else out


def _is_prose_echo(src: str, dst: str) -> bool:
    """A whole English body-text sentence was echoed back unchanged. Code and short labels do not count; leaving
    them is no worse."""
    plain = dst.strip()
    if not plain or plain.startswith("<") or "dxl:" in plain.lower():
        return False
    words = re.findall(r"[A-Za-z]{4,}", plain)
    if len(words) < 12 or len(plain) < 80:
        return False
    if len(re.findall(r"[\u4e00-\u9fff]", plain)) >= 4:
        return False
    folded_src = re.sub(r"\s+", " ", src.strip())
    folded = re.sub(r"\s+", " ", plain)
    return folded == folded_src or folded.startswith(folded_src[:40])


def _cn_number(n: int, *, higher: bool = False) -> str:
    """Convert 1–9999 to Chinese. A standalone 10–19 is "十 / 十一"; as the remainder of a higher place it gets a
    leading "一" ("一百一十")."""
    if n < 0:
        return str(n)
    if n < 10:
        return _CN_DIGITS[n]
    if n < 20:
        rest = _CN_DIGITS[n % 10] if n % 10 else ""
        return ("一" if higher else "") + "十" + rest
    if n < 100:
        q, r = divmod(n, 10)
        return _CN_DIGITS[q] + "十" + (_CN_DIGITS[r] if r else "")
    if n < 1000:
        q, r = divmod(n, 100)
        tail = "" if not r else ("零" if r < 10 else "") + _cn_number(r, higher=True)
        return _CN_DIGITS[q] + "百" + tail
    q, r = divmod(n, 1000)
    tail = "" if not r else ("零" if r < 100 else "") + _cn_number(r, higher=True)
    return _CN_DIGITS[q] + "千" + tail


def _unit_span(text: str) -> tuple[int, int, str] | None:
    """The span of visible text to replace (relative to text) and its replacement. Return None when not
    applicable.

    The normal form replaces only the digits ("第2卷": 2 becomes "二") and leaves the unit in place.
    The inverted form "第卷26" replaces unit and digits together with "二十六卷".
    """
    stripped = text.strip()
    if len(stripped) > 40 or not stripped.startswith("第") or _CITE_RE.search(stripped):
        return None
    lead = len(text) - len(text.lstrip())
    body = text[lead:]
    inverted = _UNIT_INVERTED_RE.match(body)
    if inverted:
        unit, digits = inverted.group(1), inverted.group(2)
        cn = _cn_number(int(digits.translate(_FULLWIDTH)))
        return lead + 1, lead + inverted.end(), cn + unit
    head = _UNIT_HEAD_RE.match(body)
    if not head:
        return None
    cn = _cn_number(int(head.group(1).translate(_FULLWIDTH)))
    return lead + 1, lead + head.end(), cn


def fix_numbered_unit(dst: str, target_lang: str, *, html: bool = False) -> str:
    """Change an Arabic chapter number at the start of a heading or contents entry to Chinese numerals. Return the
    text unchanged when not applicable.

    Tags are parsed only with html=True (rich text flagged by the caller). Angle brackets in plain text get only
    string replacement.
    Short citations followed by issue, page or number ("期", "页", "号") are not changed.
    """
    if not target_lang.startswith("zh"):
        return dst
    if html:
        return _fix_html_unit(dst)
    span = _unit_span(dst.strip())
    if span is None:
        return dst
    start, end, replacement = span
    core = dst.strip()
    new = core[:start] + replacement + core[end:]
    return dst.replace(core, new, 1) if new != core else dst


# Self-closing inline tags (calibre's <a id="x"/>, <span id="x"/> anchors):
# html.parser treats them as open tags that swallow the following content, so they are expanded to explicit
# closing tags before parsing. Whitespace before the slash is dropped; otherwise the serialized result does not
# match the expanded string and the number is not changed.
_SELFCLOSED_INLINE_RE = re.compile(r"<(a|span)\b([^>]*?)\s*/>")


def _fix_html_unit(dst: str) -> str:
    """In rich text, change only the text nodes holding the digits. Return the text unchanged when the parser
    altered tags or dropped text."""
    from bs4 import BeautifulSoup, NavigableString, Tag

    from .formats.html_blocks import _is_page_anchor

    expanded = _SELFCLOSED_INLINE_RE.sub(r"<\1\2></\1>", dst)
    soup = BeautifulSoup(f"<dt-frag>{expanded}</dt-frag>", "html.parser")
    frag = soup.find("dt-frag")
    if not isinstance(frag, Tag):
        return dst
    if "".join(str(child) for child in frag.children) != expanded:
        return dst
    nodes: list[NavigableString] = []
    for node in list(frag.descendants):
        if type(node) is not NavigableString or not str(node):
            continue
        parents = [p for p in node.parents if isinstance(p, Tag)]
        if any(_is_page_anchor(p) or p.name in ("sup", "sub") for p in parents):
            continue
        nodes.append(node)
    span = _unit_span("".join(str(node) for node in nodes))
    if span is None:
        return dst
    start, end, replacement = span
    pos = 0
    pieces: list[tuple[NavigableString, int, int]] = []
    for node in nodes:
        text = str(node)
        ns, ne = pos, pos + len(text)
        if ne > start and ns < end:
            a, b = max(start - ns, 0), min(end - ns, len(text))
            if any(ch not in _NUM_CHARS for ch in text[a:b]):
                return dst
            pieces.append((node, a, b))
        pos = ne
    if not pieces:
        return dst
    first, a, b = pieces[0]
    text = str(first)
    first.replace_with(NavigableString(text[:a] + replacement + text[b:]))
    for node, a, b in pieces[1:]:
        text = str(node)
        node.replace_with(NavigableString(text[:a] + text[b:]))
    return "".join(str(child) for child in frag.children)


# Number at the start of a paragraph: a dotted section number (7.4, 1.2.3) or a chapter number directly followed
# by a word (12 Agents). Style markers or tags may precede it
_LEADING_MARKS = r"\s*(?:\{[^{}\s]{1,8}\}|<[^<>]+>)*\s*"
_SECTION_NUMBER_RE = re.compile(_LEADING_MARKS + r"(\d{1,3}(?:\.\d{1,3})+)(?![\d.])")
_BARE_NUMBER_RE = re.compile(_LEADING_MARKS + r"(\d{1,3})\s+(?=[^\W\d_])")


def clean_translation(src: str, dst: str, target_lang: str) -> str:
    """Tidy a model translation before storing: remove leaked protocol markers and invented markers, straighten
    style markers, and restore punctuation and numbers to the form of the source text."""
    dst = drop_invented_marks(src, strip_control_marks(dst))
    dst = drop_added_braces(src, tidy_marks(src, dst))
    return keep_section_number(src, normalize_punctuation(dst, target_lang), target_lang)


def _readings(number: str) -> list[str]:
    """Chinese readings of a dotted number, longest first: "十六点四点三"; the decimal part may also be read digit
    by digit ("二点一零")."""
    head, *rest = number.split(".")
    whole = _cn_number(int(head))
    readings = {
        "点".join([whole, *(_cn_number(int(part)) for part in rest)]),
        "点".join([whole, *("".join(_CN_DIGITS[int(digit)] for digit in part) for part in rest)]),
    }
    return sorted(readings, key=len, reverse=True)


def keep_dotted_numbers(src: str, dst: str) -> str:
    """Restore 7.6, 16.4.3 from the source text when they were read out as "七点六", "十六点四点三"."""
    if "点" not in dst:
        return dst
    for number in sorted(set(_DOTTED_RE.findall(src)), key=len, reverse=True):
        for reading in _readings(number):
            dst = re.sub(rf"(?<![{_CN_NUMERALS}]){reading}(?![{_CN_NUMERALS}点])", number, dst)
    return dst


def keep_section_number(src: str, dst: str, target_lang: str) -> str:
    """Restore the original when the source number is in Arabic digits but the translation writes "七点四",
    "四.11", "第一点七点一", or adds "第十二章" to a bare chapter number. The prompt used to ask for
    Chinese numerals for all chapter numbers in contents and headings, and the model also changed dotted section
    numbers and chapter numbers without "Chapter", some on a page and not others. Old cached translations also
    pass through here on a hit."""
    if not target_lang.startswith("zh"):
        return dst
    dst = keep_dotted_numbers(src, dst)
    dotted = _SECTION_NUMBER_RE.match(src)
    if dotted is not None:
        number = dotted.group(1)
        parts = number.split(".")
        spelled = r"(?:点|\.)".join(f"(?:{part}|{_cn_number(int(part))})" for part in parts)
        pattern = re.compile(f"^({_LEADING_MARKS})第?{spelled}(?![点.\\d一二三四五六七八九十零])")
        return pattern.sub(lambda found: found.group(1) + number, dst, count=1)
    bare = _BARE_NUMBER_RE.match(src)
    if bare is not None:
        number = bare.group(1)
        pattern = re.compile(f"^({_LEADING_MARKS})第{_cn_number(int(number))}(?:章|部分|卷|篇)\\s*")
        return pattern.sub(lambda found: found.group(1) + number + " ", dst, count=1)
    return dst


def normalize_numbered_units(texts: list[str], results: list[str], target_lang: str) -> dict[str, str]:
    """Batch version of fix_numbered_unit; return {source text: corrected translation}."""
    fixed: dict[str, str] = {}
    for src, dst in zip(texts, results):
        new = fix_numbered_unit(dst, target_lang)
        if new != dst:
            fixed.setdefault(src, new)
    return fixed
