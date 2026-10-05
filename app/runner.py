"""批量翻译调度：去重、缓存、切批、并发、进度回调，以及边翻边预览。

并发用固定数量的 worker 从待翻批次里取活：默认按文档顺序，预览里用户滚动或跳转后，
优先取关注位置之后的批次（类似沉浸式翻译“翻译可视区域”的思路，但整本书最终都会翻完）。
"""
import asyncio
import hashlib
import re
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
                 skip_same_lang: bool = True, cache_only: bool = False):
        self.translator = translator
        self.progress = progress or (lambda done, total: None)
        self.cache = cache if cache is not None else get_cache()
        # 已经是目标语言的段落不翻（插件的段落语言检测）
        self.skip_same_lang = skip_same_lang
        # 只排版、不请求翻译服务：缺缓存时直接失败，避免用另一套配置悄悄重译
        self.cache_only = cache_only
        # 给任务页展示：检测到的文档语言、跳过的段落数
        self.info = {"detected_lang": "", "skipped": 0}

        # ---- 边翻边预览 ----
        # segments：文档顺序的段落 [{"s": 原文, "k": 类型}]，translate_all(preview=True) 时填充
        self.segments: list[dict] | None = None
        self._indices: dict[str, list[int]] = {}  # 原文 → 出现的段落序号（同一句可能出现多次）
        self._done: dict[str, str] = {}
        self._skipped: set[str] = set()
        self._skip_idx: set[int] = set()
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
            for i in self._indices.get(text, ()):
                # 参考文献按位置跳过：同一句在正文里仍显示译文
                if i in self._skip_idx or text in self._skipped:
                    updates.append([i, text, 1])
                else:
                    updates.append([i, translated, 0])
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
                            preview: bool = False, html: list[bool] | None = None,
                            skip: list[bool] | None = None) -> list[str]:
        """按原顺序返回译文；空白文本和已是目标语言的段落原样返回。

        preview=True 时记录段落供预览接口读取；kinds 是每段的类型（h1~h6 / p / li / quote / td / toc），
        只影响预览的显示样式。html 与 texts 对齐，标记哪些段是电子书富文本片段。
        skip 与 texts 对齐，标记调用方要求保留原文的段落（如参考文献章节），预览里标注为跳过。
        这个标记按位置生效：同一句在正文里出现时仍会翻译。
        """
        n = len(texts)
        skip_flags = list(skip or [])
        if len(skip_flags) < n:
            skip_flags.extend([False] * (n - len(skip_flags)))
        else:
            skip_flags = skip_flags[:n]
        html_flags = list(html or [])
        if len(html_flags) < n:
            html_flags.extend([False] * (n - len(html_flags)))
        # 只有调用方标了的段才按 HTML 发送；正文里的 <Note> 不是标签
        unique = list(dict.fromkeys(t for t in texts if t.strip()))
        lang_skipped = await asyncio.to_thread(self._detect, unique)

        def kept(i: int) -> bool:
            text = texts[i]
            return bool(text.strip()) and (skip_flags[i] or text in lang_skipped)

        self.info["skipped"] = sum(1 for i in range(n) if kept(i))
        self.translator.html_texts = {
            texts[i] for i in range(n) if html_flags[i] and texts[i].strip() and not kept(i)
        }
        need = list(dict.fromkeys(texts[i] for i in range(n) if texts[i].strip() and not kept(i)))
        prefix = self.translator.cache_key
        tl = self.translator.target_lang
        done_map = self.cache.get_many(prefix, need)
        headings = {t for t, k in zip(texts, kinds or []) if k in _HEADING_KINDS}
        html_sources = self.translator.html_texts

        def fixed(src: str, dst: str) -> str:
            if src not in headings:
                return dst
            return fix_numbered_unit(dst, tl, html=src in html_sources)

        # 标题和目录的旧缓存可能还是「第2卷」：命中时改写成中文数字，预览不用重翻
        rewrites = {t: d2 for t, d in done_map.items() if (d2 := fixed(t, d)) != d}
        if rewrites:
            self.cache.put_many(prefix, rewrites)
            done_map.update(rewrites)
        todo = [t for t in need if t not in done_map]
        if self.cache_only and todo:
            raise ValueError("有段落不在翻译缓存里，无法只重新排版。请重新翻译。")

        # 每段原文第一次出现的位置，用来按“离关注位置的远近”挑批次
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
            self._skipped = lang_skipped
            self._skip_idx = {i for i in range(n) if skip_flags[i]}
            self._log = []
            self.segments = [{"s": t, "k": (kinds[i] if kinds else "p")} for i, t in enumerate(texts)]
            # 已是目标语言的、调用方整句跳过的、缓存命中的，预览里直接显示。
            # 同一句只有部分位置跳过时，等译文出来再按位置分别标。
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
        failed: list[str] = []  # 返回空译文的段落：回填原文，超过阈值才判失败

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
                # 原样回显不是翻译：模型在批量模式下偶尔整段照抄原文。单独重翻一次，
                # 仍回显才保留（专有名词、编号本就不变，不会误伤）
                echoes = [s for s, d in pairs.items() if d and d.strip() == s.strip() and len(s) > 20]
                if echoes:
                    retried = await asyncio.gather(*(self.translator.translate_batch([s]) for s in echoes))
                    for s, (rt,) in zip(echoes, retried):
                        if rt and rt.strip() and rt.strip() != s.strip():
                            pairs[s] = rt
                # 空译文不写缓存、不计入完成（否则会永久缓存空结果），按失败处理。
                # 标题和目录的「第2卷」在入库前改成中文数字
                good = {s: fixed(s, d) for s, d in pairs.items() if d and d.strip()}
                empty = [s for s in batch if s not in good]
                if good:
                    self.cache.put_many(prefix, good)
                    done_map.update(good)
                    if preview:
                        self._record(list(good))
                    done += len(good)
                if empty:
                    # 回填策略（BabelDOC 同款）：个别段落空译文保留原文，不让整本书失败；
                    # 但服务持续吐空译文（坏了）时不能全本回填，最后按阈值判失败
                    failed.extend(empty)
                    done += len(empty)
                self.progress(done, total)

        workers = min(self.translator.concurrency, len(pending))
        await asyncio.gather(*(worker() for _ in range(workers)))
        self.info["failed"] = len(failed)
        if len(failed) > max(3, total // 10):
            raise TranslatorError(f"翻译服务持续返回空译文（{len(failed)} / {total} 段），请重试或更换服务")
        return [
            texts[i] if not texts[i].strip() or kept(i) else done_map.get(texts[i], texts[i])
            for i in range(n)
        ]


_CN_DIGITS = "零一二三四五六七八九"
_HEADING_KINDS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "toc"})
# 第2卷、第２卷，以及模型翻串了的“第卷26”。单位不含页/期/号，那些保留阿拉伯数字
_UNIT_HEAD_RE = re.compile(r"^第([0-9０-９]{1,4})(?=[卷章部篇集册])")
_UNIT_INVERTED_RE = re.compile(r"^第([卷章部篇集册])([0-9０-９]{1,4})")
# 第25卷第1期、第3页、第5号：卷期页号不是章节标题
_CITE_RE = re.compile(r"第[0-9０-９]{1,4}\s*[期页号]|[卷章部篇集册]\s*第?\s*[0-9０-９]{1,4}\s*期")
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")
_NUM_CHARS = frozenset("第0123456789０１２３４５６７８９卷章部篇集册")


def _cn_number(n: int, *, higher: bool = False) -> str:
    """1–9999 转中文。单独的 10–19 是「十 / 十一」；作为更高位的余数时补「一」（一百一十）。"""
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
    """可见文字里要替换的区间（相对 text）和替换串。不适用时返回 None。

    普通形式只替换数字（「第2卷」→ 把 2 换成「二」），单位留在原处。
    颠倒形式「第卷26」把单位和数字一起换成「二十六卷」。
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
    """标题、目录开头的阿拉伯数字章节编号改成中文数字。不适用时原样返回。

    html=True 才解析标签（调用方标记的富文本）。纯文本里的尖括号只做字符串替换。
    后面跟着期、页、号的短引文不改。
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


# 自闭合的行内标签（calibre 的 <a id="x"/>、<span id="x"/> 锚点）：
# html.parser 会把它们当成开放标签吞掉后面的内容，先展开成显式闭合再解析。
# 斜杠前的空白要丢掉，否则序列化结果和展开串对不上，编号不会改。
_SELFCLOSED_INLINE_RE = re.compile(r"<(a|span)\b([^>]*?)\s*/>")


def _fix_html_unit(dst: str) -> str:
    """富文本只改数字所在的文本节点。解析器改动了标签或丢掉文字时原样返回。"""
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


def normalize_numbered_units(texts: list[str], results: list[str], target_lang: str) -> dict[str, str]:
    """批量版的 fix_numbered_unit，返回 {原文: 修正后译文}。"""
    fixed: dict[str, str] = {}
    for src, dst in zip(texts, results):
        new = fix_numbered_unit(dst, target_lang)
        if new != dst:
            fixed.setdefault(src, new)
    return fixed
