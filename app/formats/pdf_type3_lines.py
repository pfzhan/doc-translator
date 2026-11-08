"""Turn Type3 glyph runs into line dicts with fonts, sizes and styles."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import pymupdf

from .pdf_type3_glyphs import (
    _DASHES,
    _MONO_FONT,
    _RULE,
    _TEXT_FONT,
    Glyph,
    _glyph_size,
    _Line,
    _off_size,
    _steps,
    _sure,
    _text_of,
)
from .pdf_types import Line, Span

if TYPE_CHECKING:
    from .pdf_type3 import Type3Reader

_SENTENCE_MARKS = set(".,;:")


_OPERATORS = set("=<>+-−–—≤≥≠×")


_RELATIONS = set("=<>≤≥≠≺⪯≈∈")


_QUOTES = set("‘’“”'\"`")


_MATH_FONT = "CMMI10"


_SYMBOL_FONT = "CMSY10"


_PAD = 0.6


# ---- Writing PyMuPDF lines ----


def _line_dict(line: _Line, reader: Type3Reader) -> Line | None:
    styles = _glyph_styles(line, reader)
    items: list[tuple[str, Glyph]] = []
    for index, token in enumerate(line.tokens):
        if index:
            items.append(("", _SPACE))
        items.extend((styles[id(glyph)], glyph) for glyph in token)
    # Spaces between words go with the body text: a formula span is replaced as a whole by a placeholder, and a space
    # inside it would be lost with it
    for index, (_style, glyph) in enumerate(items):
        if glyph is _SPACE:
            before, after = items[index - 1][0], items[index + 1][0]
            items[index] = (before if before in _PROSE or after not in _PROSE else after, glyph)
    spans: list[Span] = []
    run: list[Glyph] = []
    current = ""
    for style, glyph in items:
        if run and style != current:
            spans.append(_span(run, current, line, reader))
            run = []
        run.append(glyph)
        current = style
    if run:
        spans.append(_span(run, current, line, reader))
    spans = [span for span in spans if span["text"].strip()]
    if not spans:
        return None
    box = pymupdf.Rect()
    for span in spans:
        box |= pymupdf.Rect(span["bbox"])
    return {"spans": spans, "wmode": 0, "dir": (1.0, 0.0), "bbox": (box.x0, box.y0, box.x1, box.y1)}


# Space between words
_SPACE = Glyph((-1, -1), pymupdf.Rect())


# Spaces only go with plain body text: monospace words are replaced by placeholders like formulas
_PROSE = ("text",)


def _glyph_styles(line: _Line, reader: Type3Reader) -> dict[int, str]:
    """What each glyph is written as: text, mono, math, symbol, script.

    A word containing math glyphs is a formula as a whole; digits and brackets between two formula words join it, so
    (p, s, c) is one formula. A period or comma at the end of a run of formula words is sentence punctuation and stays
    in the body text.
    """
    mono = _mono_tokens(line, reader)
    kinds: list[str] = []
    for index, token in enumerate(line.tokens):
        roles = [_role_of(glyph, line, reader) for glyph in token if not _is_script(glyph, line, reader)]
        if index in mono:
            # Monospace fonts have no math: the braces and @ in {venky, anand}@db are literal characters
            kinds.append("text")
        elif any(role in ("math", "symbol") for role in roles):
            kinds.append("math")
        elif "text" in roles:
            kinds.append("text")
        else:
            kinds.append("neutral")
    for index, kind in enumerate(kinds):
        if kind != "neutral":
            continue
        before = next((item for item in reversed(kinds[:index]) if item != "neutral"), "")
        after = next((item for item in kinds[index + 1 :] if item != "neutral"), "")
        # k = 2, 1 − (…): a number right next to an operator is also part of the formula
        # A list number 1. followed by ≺ is not a minuend either: numbers with a period do not count
        operand = (
            index > 0 and kinds[index - 1] == "math" and _ends_with_operator(line.tokens[index - 1], reader)
        ) or (
            index + 1 < len(kinds)
            and kinds[index + 1] == "math"
            and _starts_with_operator(line.tokens[index + 1], reader)
            and not _sentence_mark(line.tokens[index][-1], reader)
        )
        kinds[index] = "math" if (before == "math" and after == "math") or operand else "text"
    absorbed = _close_brackets(line.tokens, kinds, reader)
    prose_tails = _hyphen_tails(line, kinds, reader)
    splits = _clause_splits(line.tokens, kinds, reader)
    styles: dict[int, str] = {}
    for index, (token, kind) in enumerate(zip(line.tokens, kinds)):
        tail = len(token)
        phrase_end = (
            index + 1 == len(kinds) or kinds[index + 1] != "math" or _ends_sentence(token, reader) or index in splits
        )
        if index in mono and index + 1 not in mono:
            # A period or comma at the end of a run of monospace words belongs to the sentence. The comma in C1, C2
            # is still code
            while tail > 1 and _sentence_mark(token[tail - 1], reader):
                tail -= 1
        if kind == "math" and phrase_end:
            opened = _phrase_depth(line.tokens, kinds, index, reader)
            while tail > 1:
                last = token[tail - 1]
                if _sentence_mark(last, reader):
                    tail -= 1
                elif _bracket(last, reader) < 0 and opened < 0:
                    # The closing parenthesis of (using d) was opened in the body text and stays there
                    opened += 1
                    tail -= 1
                else:
                    break
        for position, glyph in enumerate(token):
            if _is_script(glyph, line, reader):
                styles[id(glyph)] = "script"
            elif id(glyph) in prose_tails:
                styles[id(glyph)] = "text"
            elif (kind == "math" and position < tail) or id(glyph) in absorbed:
                styles[id(glyph)] = "symbol" if _role_of(glyph, line, reader) == "symbol" else "math"
            else:
                styles[id(glyph)] = "mono" if index in mono and position < tail else "text"
    return styles


def _hyphen_tails(line: _Line, kinds: list[str], reader: Type3Reader) -> set[int]:
    """In n-dimensional, the body-text word after the hyphen stays in the body text and only n is a formula.
    Otherwise the model does not see the word and translates it again."""
    tails: set[int] = set()
    for token, kind in zip(line.tokens, kinds):
        if kind != "math":
            continue
        for position, glyph in enumerate(token):
            if _label_text(glyph, reader) not in "-‐" or position == 0:
                continue
            rest = [item for item in token[position + 1 :] if _label_text(item, reader).isalpha()]
            if len(rest) >= 3 and all(reader.roles.get(item.key) == "text" for item in rest):
                tails.update(id(item) for item in token[position:])
            break
    return tails


def _close_brackets(tokens: list[list[Glyph]], kinds: list[str], reader: Type3Reader) -> set[int]:
    """A bracket opened in a formula must close in the formula: in (i + 1)st, 1) goes to the formula and st stays in
    the body text."""
    absorbed: set[int] = set()
    depth = 0
    for token, kind in zip(tokens, kinds):
        if kind == "math":
            for glyph in token:
                depth = max(depth + _bracket(glyph, reader), 0)
            continue
        if depth <= 0:
            continue
        for glyph in token:
            absorbed.add(id(glyph))
            depth += _bracket(glyph, reader)
            if depth <= 0:
                break
        depth = max(depth, 0)
    return absorbed


def _phrase_depth(tokens: list[list[Glyph]], kinds: list[str], end: int, reader: Type3Reader) -> int:
    """How many more opening than closing brackets the run of formula words ending at end has. A negative number
    means extra closing brackets."""
    start = end
    while start > 0 and kinds[start - 1] == "math":
        start -= 1
    return sum(_bracket(glyph, reader) for token in tokens[start : end + 1] for glyph in token)


def _clause_splits(tokens: list[list[Glyph]], kinds: list[str], reader: Type3Reader) -> set[int]:
    """The words in a run of formula words whose trailing comma splits the formula.

    A comma inside brackets does not split: (p, s, c). When the part left of the comma has no relation symbol yet and
    is a single variable or one bracketed group, it splits: d, e, g and h are one formula each, ⟨L, ⪯⟩, a ≺ b is two.
    When the left part already has a relation symbol (i = 1, 2, …, k), what follows is only an enumeration, unless
    another relation follows (a = b, c = d).
    """
    splits: set[int] = set()
    depth = 0
    related = False
    for index, (token, kind) in enumerate(zip(tokens, kinds)):
        if kind != "math":
            depth, related = 0, False
            continue
        depth = max(depth + sum(_bracket(glyph, reader) for glyph in token), 0)
        related = related or _has_relation(token, reader)
        last = token[-1]
        if depth > 0 or not _sentence_mark(last, reader) or _label_text(last, reader) not in ",;":
            continue
        if not related or _relation_follows(tokens, kinds, index, reader):
            splits.add(index)
            depth, related = 0, False
    return splits


def _has_relation(token: list[Glyph], reader: Type3Reader) -> bool:
    return any(reader.roles.get(glyph.key) == "symbol" and _label_text(glyph, reader) in _RELATIONS for glyph in token)


def _relation_follows(tokens: list[list[Glyph]], kinds: list[str], index: int, reader: Type3Reader) -> bool:
    for after in range(index + 1, len(tokens)):
        if kinds[after] != "math":
            return False
        if _has_relation(tokens[after], reader):
            return True
        if _sentence_mark(tokens[after][-1], reader) and _label_text(tokens[after][-1], reader) in ",;":
            return False
    return False


def _ends_sentence(token: list[Glyph], reader: Type3Reader) -> bool:
    """size S. c gives: the period after S separates two formulas. An ellipsis ... does not count."""
    if len(token) < 2 or not _sentence_mark(token[-1], reader) or _label_text(token[-1], reader) != ".":
        return False
    return _label_text(token[-2], reader) != "."


def _starts_with_operator(token: list[Glyph], reader: Type3Reader) -> bool:
    first = token[0]
    return reader.roles.get(first.key) == "symbol" and _label_text(first, reader) in _OPERATORS


def _ends_with_operator(token: list[Glyph], reader: Type3Reader) -> bool:
    last = token[-1]
    return reader.roles.get(last.key) == "symbol" and _label_text(last, reader) in _OPERATORS


def _bracket(glyph: Glyph, reader: Type3Reader) -> int:
    text = _label_text(glyph, reader)
    return text.count("(") + text.count("[") - text.count(")") - text.count("]")


def _sentence_mark(glyph: Glyph, reader: Type3Reader) -> bool:
    """Sentence punctuation recognized reliably. Subscript i is often read as a semicolon, but unsteadily, so it does
    not count."""
    label = reader.labels.get(glyph.key)
    return _sure(label) and label is not None and label.text in _SENTENCE_MARKS


def _role_of(glyph: Glyph, line: _Line, reader: Type3Reader) -> str:
    """A thin rule at the baseline is an underline and counts as a character; above the baseline it is a fraction
    bar and belongs to the formula."""
    if glyph.key == _RULE:
        center = (glyph.rect.y0 + glyph.rect.y1) / 2
        return "neutral" if abs(center - line.baseline) <= line.size * 0.12 else "symbol"
    return reader.roles.get(glyph.key, "symbol")


def _is_script(glyph: Glyph, line: _Line, reader: Type3Reader) -> bool:
    """Superscript or subscript: a clearly smaller font size, or the same size with the whole glyph shifted up or
    down (the upright supplier below N).

    The shift is measured at the glyph center, not the bottom edge: italic f reaches below the baseline anyway.
    Symbols without a recognized letter have no font size; those whose bottom edge hangs above three quarters of the
    x-height (footnote marks) count as superscripts. Quote marks hang anyway and hyphens sit around half the
    x-height; neither counts.
    """
    # For letters recognized unsteadily (subscript N is sometimes read as M) the class is usually right, so the font
    # size can still be estimated
    size = _glyph_size(glyph, reader.labels, sure=False)
    if size is None:
        text = _label_text(glyph, reader)
        role = reader.roles.get(glyph.key, "symbol")
        trusted = role != "symbol" and glyph.key not in reader.doubtful
        if role == "text" or (trusted and (text in _QUOTES or text in _DASHES)):
            return False
        return glyph.rect.y1 < line.baseline - line.xheight * 0.75
    return _off_size(glyph, line, reader.labels, size)


def _mono_tokens(line: _Line, reader: Type3Reader) -> set[int]:
    """Words in which at least half of the letters are monospace glyphs. Words next to a monospace word whose letter
    spacing is also the cell width (SELECT, AS, FROM) count as well."""
    found: dict[int, float] = {}
    for index, token in enumerate(line.tokens):
        # Only letters count: the parentheses of (year) are upright. Subscripts do not count either: the monospace
        # supplier below N does not make N monospace
        letters = [
            glyph for glyph in token if _label_text(glyph, reader).isalpha() and not _is_script(glyph, line, reader)
        ]
        known = [reader.mono[glyph.key] for glyph in letters if glyph.key in reader.mono]
        # Monospace and body text use two sets of bitmaps; a word with half of its letters in monospace glyphs is a
        # monospace word (the y of (day) occurs only once)
        if letters and len(known) * 2 >= len(letters):
            found[index] = statistics.median(known)
    # Letter spacing exactly at a monospace cell width, with glyphs that never occur in proportional-font words (the
    # FROM line has no other monospace word)
    pitches = set(reader.mono.values())
    for index, token in enumerate(line.tokens):
        gaps = _steps(token)
        if index in found or not gaps or min(gaps) <= 0:
            continue
        if any(
            reader.roles.get(glyph.key) in ("math", "symbol") or glyph.key in reader.proportional for glyph in token
        ):
            continue
        pitch = next((value for value in pitches if all(abs(gap - value) <= value * 0.12 for gap in gaps)), None)
        if pitch is not None:
            found[index] = pitch
    grown = bool(found)
    while grown:
        grown = False
        for index, token in enumerate(line.tokens):
            near = [found[other] for other in (index - 1, index + 1) if other in found]
            if index in found or not near:
                continue
            # part p: a single variable next to a monospace word is still a formula. Glyphs seen in body text, as in
            # So and the, do not follow either
            if any(
                reader.roles.get(glyph.key) in ("math", "symbol") or glyph.key in reader.proportional for glyph in token
            ):
                continue
            if all(abs(gap - near[0]) <= near[0] * 0.12 for gap in _steps(token)):
                found[index] = near[0]
                grown = True
    return set(found)


def _label_text(glyph: Glyph, reader: Type3Reader) -> str:
    return _text_of(glyph, reader.labels)


def _span(run: list[Glyph], style: str, line: _Line, reader: Type3Reader) -> Span:
    glyphs = [glyph for glyph in run if glyph is not _SPACE]
    text = "".join(" " if glyph is _SPACE else _label_text(glyph, reader) for glyph in run)
    ink = pymupdf.Rect()
    for glyph in glyphs:
        ink |= glyph.rect
    if style == "script":
        sizes = [size for glyph in glyphs if (size := _glyph_size(glyph, reader.labels))]
        size = round(statistics.median(sizes), 1) if sizes else round(line.size * 0.7, 1)
        baseline = ink.y1
        box = pymupdf.Rect(ink)
    else:
        size = round(line.size, 1)
        baseline = line.baseline
        box = ink | pymupdf.Rect(ink.x0, baseline - size * 0.72, ink.x1, baseline + size * 0.22)
    # Formulas are cropped from the original page with this box; a small margin on all sides keeps the pixels at the
    # bitmap edges
    box = pymupdf.Rect(box.x0 - _PAD, box.y0 - _PAD, box.x1 + _PAD, box.y1 + _PAD)
    font = {
        "text": _TEXT_FONT,
        "mono": _MONO_FONT,
        "math": _MATH_FONT,
        "symbol": _SYMBOL_FONT,
        "script": _MATH_FONT,
    }[style]
    return {
        "text": text,
        "bbox": (box.x0, box.y0, box.x1, box.y1),
        "size": size,
        "font": font,
        "flags": 0,
        "color": glyphs[0].color if glyphs else 0,
        "origin": (ink.x0, baseline),
        # Erasing the original text finds characters by these origins. The font box of a bitmap font is half a line
        # taller than the ink, so finding them by box would catch other characters
        "origins": [glyph.origin for glyph in glyphs if glyph.origin is not None],
        "ascender": 0.75,
        "descender": -0.25,
    }
