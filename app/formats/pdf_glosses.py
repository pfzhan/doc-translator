"""Find interlinear glosses and the numbered examples they belong to."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

from .pdf_drawings import (
    _TextWord,
    page_words,
)
from .pdf_tables import _word_rows

# Interlinear gloss: tolerance for aligned left edges, minimum number of words, and share of the words in each of the
# two lines that must align
_GLOSS_ALIGN = 1.0


_GLOSS_MIN = 4


_GLOSS_SHARE = 0.75


# Role labels below the words of an example sentence; small capitals extract as all capitals: AGENT, ARG0, ARGM-TMP
# Example numbers: (22.27), (3), (4a)
_NUMBER_LABEL_RE = re.compile(r"\(\d+(?:\.\d+)*[a-z]?\)")


# Example sentences numbered by chapter: (22.9), (3.4a)
_CHAPTER_NUMBER_RE = re.compile(r"\(\d+\.\d+[a-z]?\)")


_ROLE_LABEL_RE = re.compile(r"[A-Z][A-Z0-9-]{2,}")


# Grammatical abbreviations common in the Leipzig Glossing Rules. Single letters (S, A, P) are too common elsewhere
# and do not count
_LEIPZIG = frozenset(
    "SG PL DU NOM ACC GEN DAT ERG ABS LOC ABL ALL INS INSTR COM VOC PST PRS FUT PFV IPFV PRF IMPF "
    "PROG NEG DEF INDF CLF COP AUX POSS REFL RECP CAUS PASS APPL TOP FOC DECL IMP SBJ OBJ OBL DEM "
    "PTCP INF NMLZ COMP REL".split()
)


_PERSON_RE = re.compile(r"[123](?:SG|PL|DU)")


_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


# Syllables in a transcription line are separated by spaces and punctuation. Extracted tone marks are separate
# characters (z`ai, zh¯u) and do not separate syllables
_SYLLABLE_SPLIT_RE = re.compile(r"[\s,.;:!?，。；：！？、()（）\[\]“”\"'‘’]+")


_MORPHEME_SPLIT_RE = re.compile(r"[-.=\[\]<>:~()\s]+")


def interlinear_glosses(
    page: pymupdf.Page,
    words: Sequence[_TextWord] | None = None,
) -> tuple[pymupdf.Rect, ...]:
    """Interlinear glosses of linguistic examples: one line of original text, one line of gloss. Translating the gloss
    line word by word makes no sense, and any change of word order breaks the alignment, so both lines keep the
    original text. Two arrangements are detected:

    - Word-by-word alignment: every word of the lower line shares its left edge with some word of the upper line.
      Adjacent lines of body text have only a few words aligned by chance, so at least 75% of the words of each line
      must align. With only two or three words, the upper line must also be in a non-Latin script such as Chinese
      characters or kana, and the lower line in Latin letters.
    - Morpheme-by-morpheme gloss: both lines start at the same x, both consist of morphemes joined by hyphens, and
      the gloss line contains grammatical abbreviations such as 1SG and PFV (Leipzig Glossing Rules).
    - Character-by-character transcription: the upper line is Chinese characters and the lower line starts at the
      same x, consists only of Latin letters and has as many syllables as there are characters. Chinese characters
      have no spaces between them, so word-by-word alignment is impossible.
    - Role labels: the lower line consists of all-capital labels one size smaller (AGENT, ARG0, ARGM-TMP), each
      starting below some word of the upper line. A label refers to a whole phrase and aligns only with its first
      word.

    Tables align like this over many consecutive rows, while a gloss is an isolated pair of lines. words holds the
    words of this page; read now when not given.
    """
    words = list(page_words(page) if words is None else words)
    rows = _with_centered_numbers(_word_rows(words))
    pairs = [_glossed(upper, lower) for upper, lower in zip(rows, rows[1:])]
    found: list[pymupdf.Rect] = []
    for index, paired in enumerate(pairs):
        if not paired or (index > 0 and pairs[index - 1]) or (index + 1 < len(pairs) and pairs[index + 1]):
            continue
        rect = pymupdf.Rect()
        for word in (*rows[index], *rows[index + 1]):
            rect |= pymupdf.Rect(word.x0, word.y0, word.x1, word.y1)
        found.append(rect)
    return tuple(found)


def _with_centered_numbers(rows: list[list[_TextWord]]) -> list[list[_TextWord]]:
    """An example number (22.27) set level with the middle between the example and the label line below it forms a
    line of its own between the two. Merging it into the upper line puts the example and its labels next to each
    other."""
    out: list[list[_TextWord]] = []
    for index, row in enumerate(rows):
        below = rows[index + 1] if index + 1 < len(rows) else None
        if (
            out
            and below is not None
            and len(row) == 1
            and _NUMBER_LABEL_RE.fullmatch(row[0].text)
            and row[0].x1 < min(word.x0 for word in (*out[-1], *below))
            and row[0].y0 < max(word.y1 for word in out[-1])
            and row[0].y1 > min(word.y0 for word in below)
        ):
            out[-1] = [*out[-1], *row]
            continue
        out.append(row)
    return out


@dataclass(frozen=True)
class ExampleNumber:
    """An example number at the start of a line. rect is the number, row is the line of the number, text_x0 is
    where the text after the number starts"""

    rect: pymupdf.Rect
    row: pymupdf.Rect
    text_x0: float


def example_numbers(words: Sequence[_TextWord]) -> tuple[ExampleNumber, ...]:
    """Example numbers at the start of a line: (22.9), (22.11). A number has few letters, so extraction often leaves
    it outside any block while the sentence after it forms a block of its own, or the whole line ((22.12) fall.01)
    forms no block. Numbers with other text right to their left do not count (numbers cited in body text)."""
    found: list[ExampleNumber] = []
    for row in _word_rows(words):
        for index, word in enumerate(row):
            if not _CHAPTER_NUMBER_RE.fullmatch(word.text):
                continue
            height = word.y1 - word.y0
            if index and word.x0 - row[index - 1].x1 < height * 2:
                continue
            rect = pymupdf.Rect(word.x0, word.y0, word.x1, word.y1)
            for other in row[index + 1 :]:
                if other.x0 - rect.x1 > height * 3:
                    break  # another column
                rect |= pymupdf.Rect(other.x0, other.y0, other.x1, other.y1)
            text_x0 = row[index + 1].x0 if index + 1 < len(row) else word.x1
            found.append(ExampleNumber(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), rect, text_x0))
    return tuple(found)


def _glossed(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    height = max(word.y1 - word.y0 for word in upper)
    if lower[0].y0 - upper[0].y0 > height * 1.6:
        return False
    starts = [word.x0 for word in upper]
    # A gloss line consists of words, not a row of a table of numbers
    aligned = [
        word
        for word in lower
        if any(abs(word.x0 - x) <= _GLOSS_ALIGN for x in starts) and any(ch.isalpha() for ch in word.text)
    ]
    hit = [x for x in starts if any(abs(word.x0 - x) <= _GLOSS_ALIGN for word in lower)]
    # When most aligned words equal the word at the same place in the upper line, these are two list items with the
    # same start (Else search from right to left for ...); a gloss line is in another language, so the word at the
    # same place differs
    same = [
        word
        for word in aligned
        if any(abs(word.x0 - other.x0) <= _GLOSS_ALIGN and word.text == other.text for other in upper)
    ]
    if aligned and len(same) >= len(aligned) * 0.5:
        return False
    if len(aligned) >= len(lower) * _GLOSS_SHARE and len(hit) >= len(upper) * _GLOSS_SHARE:
        if len(aligned) >= _GLOSS_MIN:
            return True
        if len(aligned) >= 2 and _script_share(upper, latin=False) >= 0.5 and _script_share(lower, latin=True) == 1:
            return True
    return _morpheme_gloss(upper, lower) or _romanized(upper, lower) or _role_labels(upper, lower)


def _role_labels(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    """Above a label is a word one size larger, and the label lies entirely right below that word. These do not
    count: abbreviations in side notes (DFT, RNN), with side-note words of the same small size above them; subscripts
    (CLS below h), which overlap the line above; capitals in figures, with numbers above them. The size is compared
    with the shortest label in the row, since the subscript of ARG0 makes its box taller."""
    height = min(word.y1 - word.y0 for word in lower)
    return all(
        _ROLE_LABEL_RE.fullmatch(word.text) and any(_labels(word, other, height) for other in upper) for word in lower
    )


def _labels(label: _TextWord, word: _TextWord, height: float) -> bool:
    return (
        word.x0 - _GLOSS_ALIGN * 2 <= label.x0 <= word.x1
        and height < (word.y1 - word.y0) * 0.9
        and word.y1 - 1 <= label.y0 <= word.y1 + height * 0.35
        and any(ch.isalpha() for ch in word.text)
    )


def _romanized(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    if abs(lower[0].x0 - upper[0].x0) > _GLOSS_ALIGN or _script_share(lower, latin=True) < 1:
        return False
    text = "".join(word.text for word in upper)
    hanzi = len(_HANZI_RE.findall(text))
    if hanzi < _GLOSS_MIN or hanzi < len(re.sub(r"[\s\W\d_]", "", text)) * 0.9:
        return False
    syllables = [token for token in _SYLLABLE_SPLIT_RE.split(" ".join(word.text for word in lower)) if token]
    return len(syllables) == hanzi


def _script_share(row: Sequence[_TextWord], latin: bool) -> float:
    """Share of Latin-letter words (latin=True) or of non-Latin-script words among the words with letters in this row.
    Pinyin tone marks in PDFs are often extracted as separate modifier letters (wˇo); characters in that range also
    count as Latin letters."""
    lettered = [word for word in row if any(ch.isalpha() for ch in word.text)]
    if not lettered:
        return 0.0
    matched = [
        word
        for word in lettered
        if all(ord(ch) <= 0x24F or 0x2B0 <= ord(ch) <= 0x2FF for ch in word.text if ch.isalpha()) == latin
    ]
    return len(matched) / len(lettered)


def _morpheme_gloss(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    chains = [word for word in lower if word.text.count("-") >= 2]
    if not chains or sum(len(word.text) for word in chains) < sum(len(word.text) for word in lower) * 0.6:
        return False
    first = min(chains, key=lambda word: word.x0)
    if not any(abs(word.x0 - first.x0) <= _GLOSS_ALIGN and word.text.count("-") >= 2 for word in upper):
        return False
    tokens = _MORPHEME_SPLIT_RE.split(" ".join(word.text for word in chains))
    return any(token in _LEIPZIG or _PERSON_RE.fullmatch(token) for token in tokens)
