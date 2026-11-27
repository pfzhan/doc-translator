"""Helpers for DeepL's free web endpoint (DeepLFreeTranslator in translators.py).

The endpoint takes the request body of the DeepL API without a key, with three differences found by trying it:
- The text of one request, all paragraphs together, may be at most 1500 UTF-16 code units; more is a 400.
- Target languages are BCP 47 tags (de, en-US, zh-Hans). The API's upper-case codes (DE, ZH-HANS), and target
  languages it does not support, are answered with a 200 that carries the original text, not with an error.
- context is an object, not the string the API takes, so the document title is not sent.
"""

import html
import re

from .languages import DEEPL_CODES

# All paragraphs of one request together, in UTF-16 code units
MAX_REQUEST_UNITS = 1500

# Targets the endpoint answers with the original text (checked on 2026-11-27 for every language of DEEPL_CODES)
_UNSUPPORTED = {"th"}
DEEPL_FREE_CODES: dict[str, tuple[str, str]] = {
    code: pair for code, pair in DEEPL_CODES.items() if code not in _UNSUPPORTED
}

# Where a sentence ends: ASCII marks need white space after them (3.14, example.com), full-width ones do not
_FULL_WIDTH_ENDS = "。！？"
_ASCII_ENDS = ".!?…"
_CLOSERS = "\"'”’」』）)]"
_TAG_RE = re.compile(r"<(/?)([A-Za-z][\w-]*)[^>]*?(/?)>")
_VOID_TAGS = {"br", "hr", "img", "wbr", "input", "col", "area", "source", "track", "embed", "meta", "link"}


def target_tag(code: str) -> str:
    """A DeepL API language code as the BCP 47 tag the free endpoint takes: ZH-HANS -> zh-Hans, EN-US -> en-US,
    NB -> nb."""
    base, *rest = code.split("-")
    return "-".join([base.lower(), *(part.title() if len(part) == 4 else part.upper() for part in rest)])


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_paragraph(text: str, html_text: bool, limit: int = MAX_REQUEST_UNITS) -> list[tuple[str, str]]:
    """Cut a paragraph longer than limit into pieces that fit, preferably after a sentence. Each piece comes with
    the white space that followed it, to put back between the translations; a paragraph that fits is one piece.

    Rich text is cut only outside its inline tags. One that cannot be cut there (a single link longer than the
    limit) loses its tags and is sent as escaped text, which keeps it valid HTML."""
    if utf16_len(text) <= limit:
        return [(text, "")]
    pieces = _split(text, _cuttable(text) if html_text else None, limit)
    if pieces is None:
        plain = html.escape(html.unescape(_TAG_RE.sub("", text)), quote=False)
        pieces = _split(plain, None, limit)
    assert pieces is not None  # text without tags can always be cut
    return pieces


def pack(lengths: list[int], limit: int = MAX_REQUEST_UNITS, max_items: int = 50) -> list[list[int]]:
    """Group consecutive pieces, by their UTF-16 lengths, into requests within the limits. Returns the indexes of
    each request's pieces."""
    groups: list[list[int]] = []
    size = 0
    for i, length in enumerate(lengths):
        if not groups or size + length > limit or len(groups[-1]) >= max_items:
            groups.append([])
            size = 0
        groups[-1].append(i)
        size += length
    return groups


def _cuttable(text: str) -> list[bool]:
    """Whether rich text may be cut before each position: only in text outside every inline tag."""
    ok = [False] * (len(text) + 1)
    depth = 0
    pos = 0
    for m in _TAG_RE.finditer(text):
        if depth == 0:
            for i in range(pos, m.start() + 1):
                ok[i] = True
        closing, name, self_closing = m.group(1), m.group(2).lower(), m.group(3)
        if closing:
            depth = max(0, depth - 1)
        elif not self_closing and name not in _VOID_TAGS:
            depth += 1
        pos = m.end()
    if depth == 0:
        for i in range(pos, len(text) + 1):
            ok[i] = True
    return ok


def _split(text: str, ok: list[bool] | None, limit: int) -> list[tuple[str, str]] | None:
    pieces: list[tuple[str, str]] = []
    start = 0
    while utf16_len(text[start:]) > limit:
        cut = _cut(text, start, _fit(text, start, limit), ok)
        if cut is None:
            return None
        rest = cut
        while rest < len(text) and text[rest].isspace():
            rest += 1
        pieces.append((text[start:cut], text[cut:rest]))
        start = rest
    pieces.append((text[start:], ""))
    return pieces


def _fit(text: str, start: int, limit: int) -> int:
    """The furthest end such that text[start:end] is within limit UTF-16 code units."""
    units = 0
    end = start
    while end < len(text):
        units += 2 if ord(text[end]) > 0xFFFF else 1
        if units > limit:
            break
        end += 1
    return end


def _cut(text: str, start: int, end: int, ok: list[bool] | None) -> int | None:
    """Where to end the piece that starts at start and may reach end: after the last sentence, else before the last
    white space, else at end; None when rich text allows none of these."""
    allowed = range(end, start, -1)
    can = (lambda i: ok[i]) if ok is not None else (lambda i: True)
    for i in allowed:
        if can(i) and _ends_sentence(text, i):
            return i
    for i in allowed:
        if can(i) and i < len(text) and text[i].isspace() and not text[i - 1].isspace():
            return i
    return next((i for i in allowed if can(i)), None)


def _ends_sentence(text: str, i: int) -> bool:
    """Whether a sentence ends just before position i, closing quotes and brackets included."""
    j = i
    while j > 0 and text[j - 1] in _CLOSERS:
        j -= 1
    if j == 0:
        return False
    mark = text[j - 1]
    if mark in _FULL_WIDTH_ENDS:
        return True
    return mark in _ASCII_ENDS and i < len(text) and text[i].isspace()
