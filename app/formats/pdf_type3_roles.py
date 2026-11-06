"""Decide which Type3 glyphs are body text and which are math, digits or punctuation, from how often each
occurs inside words."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np

from .pdf_type3_glyphs import (
    _DASHES,
    _SURE,
    Glyph,
    Key,
    Label,
    _alpha_core,
    _glyph_size,
    _Line,
    _off_size,
    _sure_word,
    _text_of,
)

# A glyph with more than a quarter of its occurrences inside words is a body-text letter
_IN_WORD = 0.25


# Two bitmaps of the same letter with this similarity are the same font
_SAME_SHAPE = 0.85


_SAME_MARK = 0.5


_MATH_SYMBOLS = set("<>=+|{}^~*∑∏√∞≤≥≠≈∈∉⊂⊃±×÷→←")


# Numbers such as section numbers, item numbers and years
_NUMBER = re.compile(r"\(?\d+(?:\.\d+)*[.:,)]?")


# Capital letters that form a word on their own in English
_LETTER_WORDS = {"A", "I"}


# Ink width of the minus sign (in font sizes). En dash 0.5, em dash 1.0, hyphen 0.3
_MINUS_WIDTH = (0.55, 0.8)


def _learn_roles(
    pages: dict[int, list[_Line]],
    labels: dict[Key, Label],
    shapes: Callable[[set[Key]], dict[Key, np.ndarray]],
) -> tuple[dict[Key, str], set[Key]]:
    """Count across the document how often each glyph occurs inside words, and decide whether it is body text, math,
    or a digit or punctuation mark.

    First, words of four or more letters with every character recognized reliably settle a set of body-text glyphs
    (view names such as psc and cp are math italic and do not count). All letters of such words count as inside a
    word. Short words come next: a letter counts as inside a word when another letter of the same word is a body-text
    glyph and none of them is a variable letter that often stands alone (monospace S is also used alone as a variable
    in pseudocode). This fills in the ligature ff, the W of We, and T and h that occur in headings only in The.
    Math italic a and v almost never occur in words (v’s counts once in a while).
    Letters that never occur in a word are variables. In English only A and I form words on their own: their bitmaps
    are compared with A and I in body text, and a similar one is body text (the A one size smaller in the abstract).
    Italic body text and math italic look the same, so other letters cannot be told apart by comparison.
    """
    tokens = [
        _without_plural(_main_glyphs(line, token, labels), labels)
        for lines in pages.values()
        for line in lines
        for token in line.tokens
    ]
    seen = Counter(glyph.key for token in tokens for glyph in token)
    seeds = {
        glyph.key for token in tokens if _sure_word(token, labels) for glyph in token if _letter(labels.get(glyph.key))
    }
    alone = Counter(token[0].key for token in tokens if len(token) == 1 and _letter(labels.get(token[0].key)))
    loner = {key for key, count in alone.items() if count * 3 >= seen[key]}
    in_word: Counter[Key] = Counter()
    for token in tokens:
        if not _alpha_core(token, labels, 2):
            continue
        letters = [glyph for glyph in token if _letter(labels.get(glyph.key))]
        sure = _sure_word(token, labels)
        for glyph in letters:
            others = [item for item in letters if item is not glyph]
            if sure or (any(item.key in seeds for item in others) and not any(item.key in loner for item in others)):
                in_word[glyph.key] += 1
    numeral: Counter[Key] = Counter()
    for token in tokens:
        text = "".join(_text_of(glyph, labels) for glyph in token)
        # At least two digits: math italic b is often read as 6, and followed by a comma it also looks like a number
        if sum(ch.isdigit() for ch in text) >= 2 and _NUMBER.fullmatch(text):
            for glyph in token:
                numeral[glyph.key] += 1
    roles = {key: _role(labels.get(key), in_word[key], count, numeral[key] * 2 >= count) for key, count in seen.items()}
    # Dashes are told apart by width: the minus sign is 0.6 font sizes, and hyphen, en dash and em dash each have
    # their own width; marks taller than a dash are arrows
    dashes: dict[Key, list[tuple[float, float]]] = defaultdict(list)
    for line in (line for lines in pages.values() for line in lines):
        for glyph in line.glyphs:
            if _text_of(glyph, labels) in _DASHES:
                dashes[glyph.key].append((glyph.rect.width / line.size, glyph.rect.height / line.size))
    for key, shapes_seen in dashes.items():
        width = statistics.median(shape[0] for shape in shapes_seen)
        height = statistics.median(shape[1] for shape in shapes_seen)
        if height > 0.15 or _MINUS_WIDTH[0] <= width <= _MINUS_WIDTH[1]:
            roles[key] = "symbol"
    # The article A starts a sentence, the variable A sits inside one. It is the article only when it mostly starts
    # sentences and its bitmap looks like the body-text A.
    # Punctuation that occurs only once or twice is also compared by bitmap with common glyphs of the same name
    opening: Counter[Key] = Counter()
    for lines in pages.values():
        previous = "."
        for line in lines:
            for token in line.tokens:
                if len(token) == 1:
                    if previous[-1:] in ".!?:":
                        opening[token[0].key] += 1
                previous = "".join(_text_of(glyph, labels) for glyph in token) or previous
    words = {
        key
        for key, role in roles.items()
        if role == "letter" and labels[key].text in _LETTER_WORDS and opening[key] * 2 >= seen[key]
    }
    odd = {
        key for key, role in roles.items() if role == "neutral" and seen[key] <= 2 and not labels[key].text.isalnum()
    }
    names = {labels[key].text for key in words | odd}
    refs = {
        key
        for key, role in roles.items()
        if key not in odd
        and labels.get(key) is not None
        and labels[key].text in names
        and (role == "text" or (role == "neutral" and seen[key] >= 4))
    }
    doubtful: set[Key] = set()
    if words or odd:
        images = shapes(words | odd | refs)
        for key in words:
            if _looks_like(key, refs, labels, images, _SAME_SHAPE):
                roles[key] = "text"
        # A footnote asterisk read as a quote mark does not look like real quote marks, so its recognition is not
        # trusted
        doubtful = {key for key in odd if not _looks_like(key, refs, labels, images, _SAME_MARK)}
    return {key: "math" if role == "letter" else role for key, role in roles.items()}, doubtful


def _without_plural(token: list[Glyph], labels: dict[Key, Label]) -> list[Glyph]:
    """Plurals of variables in math: the a’s, v’s. After removing ’s, the single letter left is not a word."""
    texts = [_text_of(glyph, labels) for glyph in token]
    while texts and not texts[-1].isalnum():
        texts.pop()
    if len(texts) >= 3 and texts[-1] == "s" and texts[-2] in ("’", "'"):
        return token[: len(texts) - 2]
    return token


def _letter(label: Label | None) -> bool:
    return label is not None and label.text.isalpha()


def _role(label: Label | None, in_word: int, count: int, numeral: bool = False) -> str:
    """Digits are recognized unsteadily (bold 1 is sometimes read as l), but they mostly occur in numbers such as 1.1
    and 12, so they are still digits."""
    if label is None:
        return "symbol"
    if count and in_word / count >= _IN_WORD:
        return "text"
    if any(ch in _MATH_SYMBOLS for ch in label.text):
        return "symbol"
    if label.agreement < _SURE and not (numeral and label.text.isdigit()):
        return "math" if label.text.isalpha() else "symbol"
    if label.text.isalpha():
        return "letter"
    return "neutral"


def _looks_like(
    key: Key,
    refs: set[Key],
    labels: dict[Key, Label],
    images: dict[Key, np.ndarray],
    threshold: float,
) -> bool:
    image = images.get(key)
    label = labels.get(key)
    if image is None or label is None:
        return False
    same = [images[other] for other in refs if other in images and labels[other].text == label.text]
    return any(float(np.dot(image, other)) >= threshold for other in same)


def _main_glyphs(line: _Line, token: list[Glyph], labels: dict[Key, Label]) -> list[Glyph]:
    """Remove superscripts and subscripts. The supplier below N sits in one word with N and must not make N look
    like part of a word."""
    kept = []
    for glyph in token:
        size = _glyph_size(glyph, labels, sure=False)
        if size is None or not _off_size(glyph, line, labels, size):
            kept.append(glyph)
    return kept or token
