"""术语表：用户在翻译服务里填的，加上翻译前从整份文档里找出来的。

找术语分两步。先在本地数：同一串实词在全文反复出现（data cube、greedy algorithm），
全大写的缩写（OLAP、TPC-D），以及出现很多次的长词。这一步只求不漏。
再把候选一次交给模型，按文档标题定下译名，普通词由它删掉。
翻译时每一批只带这一批原文里出现了的条目，提示词不随术语表变长。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

Entry = tuple[str, str]

# 送翻文本里的占位符（{v1}、{b}、{|}）和电子书片段的标签，数词时去掉
_MARKUP_RE = re.compile(r"\{[^{}\s]{1,8}\}|<[^<>]+>")
# 句内的停顿处断开，术语不跨标点。撇号只在词边上才是引号，夹在字母中间是词的一部分（don't）
_CLAUSE_RE = re.compile(
    r"[.,;:!?()\[\]{}\"“”‘/|=+*<>…—–]+|(?<![^\W\d_])['’]|['’](?![^\W\d_])|\s-\s|\n"
)
# 所有格 view's set：在这里断开，不让 s 和后面的词连成一条
_POSSESSIVE_RE = re.compile(r"(?<=[^\W\d_])['’]s\b")
_WORD_RE = re.compile(r"[^\W\d_][\w’'-]*[^\W_]|[^\W\d_]")
# 汉字、假名、谚文。这类文字词间没有空格，按空格数词数不出东西
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

# 英文虚词和论文里的套话。术语里不含这些词，它们也不单独成词
_STOPWORDS = frozenset("""
a an the and or nor but if then else when while where which who whom whose what that this these those
of in on at by for to from with without within into onto over under about above below between among
through during before after since until upon via per as than so such too very also only even just
is are was were be been being am do does did done doing have has had having can could may might must
shall should will would not no yes any all some each every both either neither one two three other another
more most less least many much few several same own its it they them their there here we our us you your
he she his her him i me my mine yours ours theirs itself themselves ourselves herself himself
how why however therefore thus hence because although though whether
let lets let's use used using uses given give gives show shows shown see seen say said note
example examples figure section table chapter paper case cases way ways part parts number numbers
first second third last next new old good best well better large small high low simple
e.g i.e etc vs al fig eq ref
""".split())

_MIN_PHRASE = 3     # 多词术语至少出现几次
_MIN_ACRONYM = 2    # 缩写至少出现几次
_MIN_WORD = 6       # 单个词至少出现几次
_MIN_WORD_LEN = 4   # 单个词至少几个字母，更短的多是普通词
_MAX_WORDS = 4      # 多词术语最长几个词
_LIMIT = 80         # 交给模型的候选上限
_EXCERPT = 1500     # 给模型看的开头一段（字符）
# 在这么大比例的段落里都出现的短词按虚词算（法语、德语等没有内置虚词表的语言靠它）
_COMMON_SHARE = 0.3


def parse(text: str) -> list[Entry]:
    """每行“原文 = 译文”。没有等号的行、两边有一边为空的行忽略。"""
    entries: list[Entry] = []
    for line in (text or "").splitlines():
        source, sep, target = line.partition("=")
        source, target = source.strip(), target.strip()
        if sep and source and target:
            entries.append((source, target))
    return entries


def render(entries: Iterable[Entry]) -> str:
    return "\n".join(f"{source} = {target}" for source, target in entries)


def merge(user: Sequence[Entry], found: Sequence[Entry]) -> list[Entry]:
    """用户填的优先。同一个原文（不分大小写）只留一条。"""
    seen: set[str] = set()
    out: list[Entry] = []
    for source, target in [*user, *found]:
        key = source.casefold()
        if key not in seen:
            seen.add(key)
            out.append((source, target))
    return out


def relevant(entries: Sequence[Entry], texts: Sequence[str]) -> list[Entry]:
    """只留这一批原文里出现了的条目。英文复数（-s、-es）也算出现。"""
    if not entries:
        return []
    haystack = "\n".join(texts)
    return [entry for entry in entries if _mentions(haystack, entry[0])]


def _mentions(haystack: str, term: str) -> bool:
    words = term.split()
    last = words[-1]
    lower = last.casefold()
    if lower.endswith("ies") and len(lower) > 4:
        last, tail = last[:-3], "(?:y|ies)"
    elif lower.endswith("y") and len(lower) > 2:
        last, tail = last[:-1], "(?:y|ies)"
    elif lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        last, tail = last[:-1], "(?:s|es)?"
    else:
        tail = "(?:s|es)?"
    body = r"\s+".join(map(re.escape, [*words[:-1], last]))
    # 缩写分大小写：SQL 不等于句子里的 sql，IS 不等于 is
    flags = 0 if _is_acronym(term) else re.I
    return re.search(r"(?<![\w-])" + body + tail + r"(?![\w-])", haystack, flags) is not None


def excerpt(texts: Sequence[str]) -> str:
    """文档开头的几段，让模型知道是哪个领域。短的标题、作者行跳过。"""
    out: list[str] = []
    size = 0
    for text in texts:
        text = " ".join(_MARKUP_RE.sub(" ", text).split())
        if len(text) < 80:
            continue
        out.append(text[:_EXCERPT - size])
        size += len(out[-1])
        if size >= _EXCERPT:
            break
    return "\n".join(out)


def candidates(texts: Sequence[str]) -> list[str]:
    """在本地找候选术语，按重要程度排好。中日韩为主的文档返回空。"""
    clean = [_POSSESSIVE_RE.sub("\n", _MARKUP_RE.sub(" ", text)) for text in texts if text.strip()]
    if not clean:
        return []
    joined = "".join(clean)
    if len(_CJK_RE.findall(joined)) > len(joined) * 0.2:
        return []
    clauses = [
        [word for word in _WORD_RE.findall(clause)]
        for text in clean for clause in _CLAUSE_RE.split(text)
    ]
    common = _common_words(clean)

    def plain(word: str) -> bool:
        return word.casefold() not in _STOPWORDS and word.casefold() not in common

    forms: dict[str, Counter[str]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    for words in clauses:
        for start in range(len(words)):
            for size in range(1, _MAX_WORDS + 1):
                gram = words[start:start + size]
                if len(gram) < size or not all(plain(word) for word in gram):
                    continue
                if size > 1 and any(_is_acronym(word) for word in gram):
                    continue
                # 表格相邻两格写着同一个词时会连成 “semi-supervised semi-supervised”
                if len({word.casefold() for word in gram}) < size:
                    continue
                surface = " ".join(gram)
                key = _key(surface)
                forms[key][surface] += 1
                counts[key] += 1

    scored: list[tuple[float, str]] = []
    for key, count in counts.items():
        surface = _surface(forms[key])
        size = len(surface.split())
        if size == 1 and _is_acronym(surface):
            if count >= _MIN_ACRONYM:
                scored.append((count * 1.5, surface))
        elif size == 1:
            if count >= _MIN_WORD and len(surface) >= _MIN_WORD_LEN:
                scored.append((count * 0.5, surface))
        elif count >= _MIN_PHRASE:
            scored.append((count * size, surface))
    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    return _drop_covered([surface for _score, surface in scored], counts)[:_LIMIT]


def _common_words(texts: Sequence[str]) -> set[str]:
    """在很多段落里都出现的短词。文档太短时不算，免得把术语当成虚词。"""
    if len(texts) < 20:
        return set()
    seen: Counter[str] = Counter()
    for text in texts:
        seen.update({word.casefold() for word in _WORD_RE.findall(text) if len(word) <= 4})
    return {word for word, count in seen.items() if count >= len(texts) * _COMMON_SHARE}


def _is_acronym(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return len(letters) >= 2 and sum(c.isupper() for c in letters) >= 2


def _key(surface: str) -> str:
    """大小写和英文复数不算不同的词。缩写保留大小写：IS 和 is 不是一个词。"""
    words = surface.split()
    if len(words) == 1 and _is_acronym(surface):
        return surface
    last = words[-1].casefold()
    if len(last) > 4 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    return " ".join([*(word.casefold() for word in words[:-1]), last])


def _surface(forms: Counter[str]) -> str:
    """出现最多的写法。句首大写的那种不算，除非全文只有这种写法。"""
    ranked = sorted(forms.items(), key=lambda item: (-item[1], item[0]))
    lower = [form for form, _count in ranked if form[:1].islower()]
    return lower[0] if lower else ranked[0][0]


def _drop_covered(terms: list[str], counts: Counter[str]) -> list[str]:
    """data 和 cube 几乎只出现在 data cube 里时，只留 data cube。"""
    keys = {term: _key(term) for term in terms}
    out: list[str] = []
    for term in terms:
        key = keys[term]
        inside = [
            other for other in terms
            if other is not term and len(other) > len(term) and f" {key} " in f" {keys[other]} "
        ]
        if any(counts[keys[other]] >= counts[key] * 0.8 for other in inside):
            continue
        out.append(term)
    return out
