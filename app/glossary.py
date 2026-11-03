"""Glossary: the entries the user entered for the translation service, plus those found in the whole document
before translation.

Terms are found in two steps. First they are counted locally: sequences of content words repeated across the text
(data cube, greedy algorithm), all-caps acronyms (OLAP, TPC-D), and long words occurring many times. This step only
aims to miss nothing. The candidates are then given to the model at once, which fixes translations based on the
document title and removes common words. Each translation batch carries only the entries occurring in its source
text, so the prompt does not grow with the glossary.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

Entry = tuple[str, str]

# Placeholders in the text sent for translation ({v1}, {b}, {|}) and tags of ebook fragments, removed when
# counting words
_MARKUP_RE = re.compile(r"\{[^{}\s]{1,8}\}|<[^<>]+>")
# Split at pauses within a sentence; terms do not cross punctuation. An apostrophe is a quote only at a word
# boundary; between letters it is part of the word (don't)
_CLAUSE_RE = re.compile(r"[.,;:!?()\[\]{}\"“”‘/|=+*<>…—–]+|(?<![^\W\d_])['’]|['’](?![^\W\d_])|\s-\s|\n")
# Possessives such as view's set: split here so that s does not join the following word
_POSSESSIVE_RE = re.compile(r"(?<=[^\W\d_])['’]s\b")
_WORD_RE = re.compile(r"[^\W\d_][\w’'-]*[^\W_]|[^\W\d_]")
# Han characters, kana and hangul. These scripts have no spaces between words, so counting by spaces finds nothing
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

# English function words and stock phrases of papers. Terms do not contain them, nor are they terms on their own
_STOPWORDS = frozenset(
    """
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
""".split()
)

# First cells of the header row of an imported table
_HEADERS = frozenset({"原文", "source", "term", "术语"})

_MIN_PHRASE = 3  # minimum occurrences of a multi-word term
_MIN_ACRONYM = 2  # minimum occurrences of an acronym
_MIN_WORD = 6  # minimum occurrences of a single word
_MIN_WORD_LEN = 4  # minimum letters of a single word; shorter ones are mostly common words
_MAX_WORDS = 4  # maximum words of a multi-word term
_LIMIT = 80  # maximum candidates given to the model
_EXCERPT = 1500  # characters of the document start shown to the model
# Short words occurring in at least this share of paragraphs count as function words (needed for languages such
# as French and German without a built-in list)
_COMMON_SHARE = 0.3


def parse(text: str) -> list[Entry]:
    """One "source = translation" per line. Lines without an equals sign or with an empty side are ignored. An
    equals sign with spaces around it separates first, so a term may contain one (E=mc2 = 质能方程)."""
    entries: list[Entry] = []
    for line in (text or "").splitlines():
        source, sep, target = line.partition(" = ") if " = " in line else line.partition("=")
        source, target = source.strip(), target.strip()
        if sep and source and target:
            entries.append((source, target))
    return entries


def render(entries: Iterable[Entry]) -> str:
    return "\n".join(f"{source} = {target}" for source, target in entries)


def parse_table(text: str) -> list[Entry]:
    """Entries from an imported file: CSV or TSV with the source in the first column and the translation in the
    second, or "source = translation" lines. A header row of 原文 / source is skipped.

    A file with a tab anywhere is TSV and one with an equals sign on every line is "source = translation" lines,
    whose terms may contain commas; anything else is CSV."""
    text = (text or "").lstrip("\ufeff")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if "\t" in text:
        delimiter = "\t"
    elif all("=" in line for line in lines):
        return parse(text)
    else:
        delimiter = ","
    entries: list[Entry] = []
    for row in csv.reader(lines, delimiter=delimiter):
        if len(row) < 2:
            continue
        source, target = row[0].strip(), row[1].strip()
        if source and target:
            entries.append((source, target))
    if entries and entries[0][0].casefold() in _HEADERS:
        entries.pop(0)
    return entries


def to_csv(entries: Iterable[Entry]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("原文", "译文"))
    writer.writerows(entries)
    return out.getvalue()


def merge(user: Sequence[Entry], found: Sequence[Entry]) -> list[Entry]:
    """User entries take precedence. Only one entry is kept per source term (case-insensitive)."""
    seen: set[str] = set()
    out: list[Entry] = []
    for source, target in [*user, *found]:
        key = source.casefold()
        if key not in seen:
            seen.add(key)
            out.append((source, target))
    return out


def relevant(entries: Sequence[Entry], texts: Sequence[str]) -> list[Entry]:
    """Keep only the entries occurring in this batch's source text. English plurals (-s, -es) count as occurring.

    Style markers often fall inside a term (word{b} type{/b}) and are removed before searching.
    """
    if not entries:
        return []
    haystack = _MARKUP_RE.sub(" ", "\n".join(texts))
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
    # Acronyms are case-sensitive: SQL is not sql in a sentence, IS is not is
    flags = 0 if _is_acronym(term) else re.I
    return re.search(r"(?<![\w-])" + body + tail + r"(?![\w-])", haystack, flags) is not None


def excerpt(texts: Sequence[str]) -> str:
    """The first paragraphs of the document, so the model knows the field. Short titles and author lines are
    skipped."""
    out: list[str] = []
    size = 0
    for text in texts:
        text = " ".join(_MARKUP_RE.sub(" ", text).split())
        if len(text) < 80:
            continue
        out.append(text[: _EXCERPT - size])
        size += len(out[-1])
        if size >= _EXCERPT:
            break
    return "\n".join(out)


def candidates(texts: Sequence[str]) -> list[str]:
    """Find candidate terms locally, sorted by importance. Empty for documents mainly in Chinese, Japanese or
    Korean."""
    clean = [_POSSESSIVE_RE.sub("\n", _MARKUP_RE.sub(" ", text)) for text in texts if text.strip()]
    if not clean:
        return []
    joined = "".join(clean)
    if len(_CJK_RE.findall(joined)) > len(joined) * 0.2:
        return []
    clauses = [[word for word in _WORD_RE.findall(clause)] for text in clean for clause in _CLAUSE_RE.split(text)]
    common = _common_words(clean)

    def plain(word: str) -> bool:
        return word.casefold() not in _STOPWORDS and word.casefold() not in common

    forms: dict[str, Counter[str]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    for words in clauses:
        for start in range(len(words)):
            for size in range(1, _MAX_WORDS + 1):
                gram = words[start : start + size]
                if len(gram) < size or not all(plain(word) for word in gram):
                    continue
                if size > 1 and any(_is_acronym(word) for word in gram):
                    continue
                # Two adjacent table cells with the same word would join into “semi-supervised semi-supervised”
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
    """Short words occurring in many paragraphs. Not computed for short documents, so terms are not taken for
    function words."""
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
    """Case and English plurals do not make different words. Acronyms keep their case: IS and is are different
    words."""
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
    """The most frequent spelling. The sentence-initial capitalized form does not count unless it is the only one."""
    ranked = sorted(forms.items(), key=lambda item: (-item[1], item[0]))
    lower = [form for form, _count in ranked if form[:1].islower()]
    return lower[0] if lower else ranked[0][0]


def _drop_covered(terms: list[str], counts: Counter[str]) -> list[str]:
    """When data and cube occur almost only within data cube, keep only data cube."""
    keys = {term: _key(term) for term in terms}
    # Consecutive words of a phrase → longer phrases containing them. A book has tens of thousands of candidates,
    # too many to compare pairwise
    holders: dict[str, list[str]] = defaultdict(list)
    for other in terms:
        words = keys[other].split()
        for size in range(1, len(words)):
            for start in range(len(words) - size + 1):
                holders[" ".join(words[start : start + size])].append(other)
    out: list[str] = []
    for term in terms:
        key = keys[term]
        if any(len(other) > len(term) and counts[keys[other]] >= counts[key] * 0.8 for other in holders.get(key, ())):
            continue
        out.append(term)
    return out
