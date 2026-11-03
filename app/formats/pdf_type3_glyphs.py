"""Glyph, label and line records of the Type3 reader, with the thresholds and letter classes that its recognizers
share."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import pymupdf

# Agreement ratio of the recognition. Glyphs below it are recognized differently each time: math symbols or
# math italic.
_SURE = 0.8


# Glyphs this much smaller than the line's font size are superscripts or subscripts. So are glyphs of full size
# whose baseline is shifted by this much (in font sizes)
_SCRIPT = 0.8


_SHIFT = 0.18


# Ink height of each letter class as a share of the font size (about the same in Computer Modern and Times)
_X_LETTERS = set("acemnorsuvwxz")


_TALL_LETTERS = set("bdhkl") | set("ABCDEFGHIKLMNOPRSTUVWXYZ") | set("0123456789")


_DESCENDERS = set("gpqy")


_UNSTEADY = set("fjJQ")


_HEIGHT_RATIO = {"x": 0.45, "tall": 0.69, "desc": 0.64, "i": 0.67, "t": 0.62}


# Height of the center of each letter class above the baseline
_CENTER_RATIO = {"x": 0.225, "tall": 0.345, "desc": 0.13, "i": 0.335, "t": 0.31}


_DASHES = set("-–—−~")


_TEXT_FONT = "Times-Roman"


_MONO_FONT = "Courier"


Key = tuple[int, int]


# Thin rules: the underline of \_ and inline fraction bars. Rules longer than 30pt are table rules and footnote rules
_RULE: Key = (-2, 0)


@dataclass(frozen=True)
class OcrChar:
    text: str
    rect: pymupdf.Rect
    conf: float


@dataclass(frozen=True)
class OcrWord:
    text: str
    rect: pymupdf.Rect
    conf: float
    chars: tuple[OcrChar, ...] = ()


@dataclass(frozen=True)
class Glyph:
    """One glyph drawn on the page. key is (font xref, character code), rect is the ink box of the bitmap, origin is
    the glyph origin."""

    key: Key
    rect: pymupdf.Rect
    color: int = 0
    origin: tuple[float, float] | None = None


@dataclass(frozen=True)
class Label:
    text: str
    agreement: float


@dataclass
class _Line:
    glyphs: list[Glyph]
    tokens: list[list[Glyph]] = field(default_factory=list)
    baseline: float = 0.0
    xheight: float = 0.0
    size: float = 0.0


def _sure(label: Label | None) -> bool:
    return label is not None and label.agreement >= _SURE


def _letter_class(text: str) -> str | None:
    """Letter class by ink height. f, j, J and Q reach below the baseline in italic and have no fixed height, so they
    get no class."""
    if len(text) != 1 or text in _UNSTEADY:
        return None
    if text in _X_LETTERS:
        return "x"
    if text in _DESCENDERS:
        return "desc"
    if text in ("i", "t"):
        return text
    if text in _TALL_LETTERS:
        return "tall"
    return None


def _glyph_size(glyph: Glyph, labels: dict[Key, Label], sure: bool = True) -> float | None:
    """Convert the ink height back to a font size by letter class. Glyphs not recognized reliably have no font
    size."""
    label = labels.get(glyph.key)
    if label is None or (sure and label.agreement < _SURE):
        return None
    kind = _letter_class(label.text)
    if kind is None:
        return None
    return glyph.rect.height / _HEIGHT_RATIO[kind]


class _GlyphCenters:
    """The glyphs of one page sorted by the vertical coordinate of their center. A page has a thousand glyphs and
    hundreds of words, and comparing every word with every glyph is too slow, so a box lookup only checks the range
    whose centers lie between the top and bottom of the box."""

    def __init__(self, glyphs: list[Glyph]) -> None:
        self._glyphs = glyphs
        self._order = sorted(range(len(glyphs)), key=lambda index: (glyphs[index].rect.y0 + glyphs[index].rect.y1) / 2)
        self._middles = [(glyphs[index].rect.y0 + glyphs[index].rect.y1) / 2 for index in self._order]

    def inside(self, box: pymupdf.Rect) -> list[Glyph]:
        """Glyphs whose center lies in box, in their original order."""
        low = bisect.bisect_left(self._middles, box.y0)
        high = bisect.bisect_right(self._middles, box.y1)
        found = sorted(index for index in self._order[low:high] if _center_in(self._glyphs[index].rect, box))
        return [self._glyphs[index] for index in found]


def _core(token: list[Glyph], labels: dict[Key, Label]) -> str:
    """Strip punctuation, quote marks and footnote marks (the asterisk of Efficiently*) from both ends and return the
    characters in between."""
    texts = [(labels.get(glyph.key) or Label("?", 0)).text for glyph in token]
    start, end = 0, len(texts)
    while start < end and not texts[start].isalpha():
        start += 1
    while end > start and not texts[end - 1].isalpha():
        end -= 1
    return "".join(texts[start:end])


def _alpha_core(token: list[Glyph], labels: dict[Key, Label], length: int) -> bool:
    core = _core(token, labels)
    return sum(ch.isalpha() for ch in core) >= length and core.replace("-", "").replace("’", "").isalpha()


def _sure_word(token: list[Glyph], labels: dict[Key, Label]) -> bool:
    if any((label := labels.get(glyph.key)) is None or label.agreement < _SURE for glyph in token):
        return False
    return _alpha_core(token, labels, 4)


def _text_of(glyph: Glyph, labels: dict[Key, Label]) -> str:
    label = labels.get(glyph.key)
    return label.text if label is not None else "?"


def _off_size(glyph: Glyph, line: _Line, labels: dict[Key, Label], size: float) -> bool:
    """The font size is clearly smaller than the line's, or the center is off the height expected for its letter
    class."""
    if size < line.size * _SCRIPT:
        return True
    kind = _letter_class(labels[glyph.key].text)
    center = (glyph.rect.y0 + glyph.rect.y1) / 2
    expected = line.baseline - _CENTER_RATIO.get(kind or "", 0.3) * line.size
    return abs(center - expected) > line.size * _SHIFT


def _steps(token: list[Glyph]) -> list[float]:
    return [_middle(right) - _middle(left) for left, right in zip(token, token[1:])]


def _middle(glyph: Glyph) -> float:
    return (glyph.rect.x0 + glyph.rect.x1) / 2


def _center_in(rect: pymupdf.Rect, box: pymupdf.Rect) -> bool:
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    return box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1
