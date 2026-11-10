"""Translations the user corrects by hand in the reader.

An edited translation goes into the translation cache under the key the record's own translation has there, and
the record's files are made again from the cache, as when generating the other version. The key depends on the
service, model, prompt, glossaries, document glossary and, in PDFs, the note of the paragraph; rather than work it
out again, the files are made by a run that only reads the cache (Runner with cache_only), through EditedCache,
which hands that run the edited translation wherever it asks for the paragraph's. Those are the keys the edit is
stored under once the run succeeded.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from .runner import TranslationCache, noted_key

# Formula and image placeholders: the files put a formula or an image back at each one. The reader shows them
# numbered in the order they appear, so the user can move them; PDFs show formula markers (\x01i\x02...) instead,
# which the reader writes as such placeholders too
_SLOT_RE = re.compile(r"\{v(\d+)\}")
# Other markers the files read: sub- and superscripts of PDF formulas
_MARKER_RE = re.compile(r"\x01[^\x02]*\x02")
# What a cache key adds after the paragraph's text when the paragraph has a note (see runner.noted_key)
_NOTE_START = noted_key("", " ").rstrip()


class EditRejected(ValueError):
    """The edited translation cannot replace the one in the cache."""


def with_placeholders(before: str, after: str) -> str:
    """after, written with its placeholders numbered by order of appearance, given the placeholders of before: the
    n-th of before's, in the order they appear there, for {vn}. Raise EditRejected when after does not have each
    of them exactly once, or drops other markers before has."""
    slots = [m.group(0) for m in _SLOT_RE.finditer(before)]
    used = [int(m.group(1)) for m in _SLOT_RE.finditer(after)]
    if sorted(used) != list(range(1, len(slots) + 1)):
        shown = "、".join(f"{{v{n}}}" for n in range(1, len(slots) + 1))
        if not slots:
            raise EditRejected("这一段没有公式或图片，译文里不能有 {v1} 这样的占位符")
        raise EditRejected(f"{shown} 是这一段的公式或图片，修改后的译文要各保留一次")
    if Counter(_MARKER_RE.findall(before)) != Counter(_MARKER_RE.findall(after)):
        raise EditRejected("这一段含有无法在阅读器里修改的公式标记")
    return _SLOT_RE.sub(lambda m: slots[int(m.group(1)) - 1], after)


@dataclass
class EditedCache:
    """The translation cache as a cache-only run sees it with the translation of source replaced by text, whose
    placeholders are numbered by order of appearance (see with_placeholders).

    Lookups of source, with any note, get text wherever the cache has a translation for them; a miss stays a miss,
    so a record whose cache no longer matches fails as it would without the edit. What the run stores for those
    keys (its tidying of the punctuation) is held back, and commit writes it, or text, once the run succeeded.
    """

    inner: TranslationCache
    source: str
    text: str
    # (prefix, key) of each entry the edit replaces -> what the run would store there
    edited: dict[tuple[str, str], str] = field(default_factory=dict)

    def _is_source(self, prefix: str, key: str) -> bool:
        # The document glossary is stored with the prefix of the translations plus ":terms"
        if prefix.endswith(":terms"):
            return False
        return key == self.source or key.startswith(self.source + _NOTE_START)

    def get_many(self, prefix: str, texts: list[str]) -> dict[str, str]:
        found = self.inner.get_many(prefix, texts)
        for key in texts:
            if key not in found or not self._is_source(prefix, key):
                continue
            text = with_placeholders(found[key], self.text)
            found[key] = self.edited.setdefault((prefix, key), text)
        return found

    def put_many(self, prefix: str, pairs: dict[str, str]) -> None:
        kept = {key: value for key, value in pairs.items() if (prefix, key) not in self.edited}
        for key, value in pairs.items():
            if (prefix, key) in self.edited:
                self.edited[(prefix, key)] = value
        if kept:
            self.inner.put_many(prefix, kept)

    def drop_many(self, prefix: str, texts: list[str]) -> None:
        drop = getattr(self.inner, "drop_many", None)
        rest = [key for key in texts if (prefix, key) not in self.edited]
        if callable(drop) and rest:
            drop(prefix, rest)

    def claim(self, doc: str, prefix: str, texts: list[str]) -> None:
        claim = getattr(self.inner, "claim", None)
        if callable(claim):
            claim(doc, prefix, texts)

    def verified(self, prefix: str, texts: list[str]) -> set[str]:
        verified = getattr(self.inner, "verified", None)
        return set(verified(prefix, texts)) if callable(verified) else set()

    def mark_verified(self, prefix: str, texts: list[str]) -> None:
        mark = getattr(self.inner, "mark_verified", None)
        if callable(mark):
            mark(prefix, texts)

    def commit(self) -> None:
        """Store the edit under every key it replaced."""
        by_prefix: dict[str, dict[str, str]] = {}
        for (prefix, key), value in self.edited.items():
            by_prefix.setdefault(prefix, {})[key] = value
        for prefix, pairs in by_prefix.items():
            self.inner.put_many(prefix, pairs)
        # A translation typed by the user is not checked against its neighbors as a shifted batch would be
        mark = getattr(self.inner, "mark_verified", None)
        if callable(mark):
            for prefix, pairs in by_prefix.items():
                mark(prefix, list(pairs))
