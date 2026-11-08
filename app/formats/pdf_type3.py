"""Recognize the letters of Type3 bitmap fonts without Unicode from their glyph shapes.

Old versions of Ghostscript draw every character as a bitmap with encoding names /a0, /a1 and no ToUnicode, so
text extraction yields character codes. A character code draws the same bitmap throughout the document. The whole
document is first recognized with tesseract, and the recognized letters vote by position on the bitmaps, which
tells which letter each bitmap is. The text layer is then rebuilt from the bitmaps: position and size come from the
bitmap box given by d1, word spacing from the gaps between the ink. Drift of OCR word boxes, missed words and
occasional misreadings no longer reach the translation.

Math italic a, v, S only occur on their own and never inside words; math symbols are recognized differently each
time. These two kinds are written in math fonts, and superscripts and subscripts are written with their actual size
and position. Formula placeholders and paragraph splitting downstream follow the same logic as ordinary TeX PDFs."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np
import pymupdf

from .pdf_type3_glyphs import (
    _HEIGHT_RATIO,
    _MONO_FONT,
    _RULE,
    _SCRIPT,
    _TEXT_FONT,
    Glyph,
    Key,
    Label,
    OcrWord,
    _glyph_size,
    _GlyphCenters,
    _letter_class,
    _Line,
    _middle,
    _steps,
    _sure_word,
    _text_of,
)
from .pdf_type3_lines import _line_dict
from .pdf_type3_ocr import _ocr_pages, _vote_labels
from .pdf_type3_roles import _learn_roles
from .pdf_types import Block, Line, PageDict

_CUSTOM_NAME = re.compile(r"^[A-Za-z]\d+$")


# Share of the page's text in broken fonts. A few characters in a symbol font do not rebuild the whole page.
_BROKEN_SHARE = 0.4


_MIN_BROKEN_CHARS = 20


# A page is rebuilt only when this share of its glyphs has a recognized letter. Otherwise the original glyphs stay.
_READABLE = 0.8


# Gaps in x-heights: up to 0.4 is letter spacing, from 0.6 on is a space. The band in between has both; it is
# split by the letter and word spacing of the line, and when the line has no clear word spacing OCR words decide
_TIGHT = 0.4


_WIDE = 0.6


# Largest horizontal gap within one line, in font sizes. Even stretched word spaces stay under half a font size;
# wider gaps separate side-by-side author names, another column, or another table column
_RUN_GAP = 1.6


# The ink of a filled dot covers more than 70% of its box, a letter less than half
_FILLED = 0.65


_SHAPE_DPI = 600


_SHAPE_GRID = 16


# The next line joins the same block when its baseline is within this many font sizes. TeX leading is 1.2 font
# sizes, a bit more where an inline formula makes the line taller
_LEADING = 1.6


_TRAILING_MARKS = set(",.;:)")


# Lower bound of the monospace cell width, in font sizes. Monospace is 0.525; the most evenly spaced body-text
# words (each, latter) stay below 0.48
_MONO_PITCH = 0.5


# Characters that are very narrow in body-text fonts (around 0.28 font sizes). In monospace fonts they also fill a
# whole cell
_NARROW = set("iljI.,;:’'!|")


_RULE_HEIGHT = 1.2


_RULE_LENGTH = 30.0


def broken_type3_names(doc: pymupdf.Document, page: pymupdf.Page) -> set[str]:
    """Type3 fonts without ToUnicode whose glyph names are made-up names such as /a0."""
    return set(_broken_fonts(doc, page).values())


def _broken_fonts(doc: pymupdf.Document, page: pymupdf.Page) -> dict[int, str]:
    found: dict[int, str] = {}
    for xref, _ext, subtype, name, _ref, _encoding in page.get_fonts():
        if subtype == "Type3" and name and _custom_encoding(doc, xref):
            found[xref] = name
    return found


class Type3Reader:
    """The bitmap fonts of one document. The whole document is recognized on first use; after that the rebuilt lines
    are returned page by page."""

    def __init__(self, doc: pymupdf.Document):
        self.doc = doc
        self._lines: dict[int, list[_Line]] | None = None
        self.labels: dict[Key, Label] = {}
        self.roles: dict[Key, str] = {}
        self.mono: dict[Key, float] = {}
        self.proportional: set[Key] = set()
        self.doubtful: set[Key] = set()

    def lines(self, pno: int) -> list[_Line] | None:
        if self._lines is None:
            self._lines = self._read()
        return self._lines.get(pno)

    def _read(self) -> dict[int, list[_Line]]:
        shapes: dict[int, dict[int, tuple[float, float, float, float]]] = {}
        pages: dict[int, list[tuple[Key, float, float, float, int]]] = {}
        for page in self.doc.pages():
            fonts = _broken_fonts(self.doc, page)
            if not fonts:
                continue
            for xref in fonts:
                if xref not in shapes:
                    shapes[xref] = _font_shapes(self.doc, xref)
            placed = _placed_codes(page, {name: xref for xref, name in fonts.items()}, shapes)
            if len(placed) >= _MIN_BROKEN_CHARS:
                pages[page.number] = placed
        if not pages:
            return {}
        flip = _y_flip(self.doc, pages, shapes)
        glyphs = {
            pno: [_glyph(key, x, y, size, color, shapes, flip) for key, x, y, size, color in placed]
            for pno, placed in pages.items()
        }
        words = _ocr_pages(self.doc, list(glyphs))
        self.labels = _vote_labels(glyphs, words)
        # \_ and inline fraction bars are thin rules that old Ghostscript draws as small images. They belong to the line
        self.labels[_RULE] = Label("_", 1.0)
        for pno, items in glyphs.items():
            items.extend(Glyph(_RULE, rect) for rect in _rules(self.doc[pno]))
        owners = {pno: _word_owners(items, words.get(pno, [])) for pno, items in glyphs.items()}
        lines = {pno: _build_lines(items, self.labels, owners[pno]) for pno, items in glyphs.items()}
        self.roles, self.doubtful = _learn_roles(lines, self.labels, lambda keys: _glyph_shapes(self.doc, glyphs, keys))
        self.mono = _mono_kinds(lines, self.labels)
        self.proportional = _proportional_kinds(lines, self.mono)
        for key in _bullets(lines, self.labels, lambda keys: _ink_share(self.doc, glyphs, keys)):
            self.labels[key] = Label("•", 1.0)
            self.roles[key] = "text"
        bearings = _bearings(lines, self.labels)
        # A monospace space is a whole blank cell and the letter spacing itself is wide; the hook of f reaches into the
        # blank on its right. Words are split again after monospace glyphs are found and side bearings measured
        for pno, page_lines in lines.items():
            for line in page_lines:
                line.tokens = _tokens(line, self.labels, owners[pno], self.mono, bearings)
        return {pno: page_lines for pno, page_lines in lines.items() if _readable(glyphs[pno], self.labels)}


def recover_type3_page(page: pymupdf.Page, data: PageDict, reader: Type3Reader | None = None) -> bool:
    """When most of this page is bitmap font text with a made-up encoding, replace the character codes with the
    recognized text layer.

    Pages that cannot be recognized (OCR timeout, most glyphs without a recognized letter) only lose the character
    codes; the bitmap glyphs stay on the original page, so the whole book does not fail.
    """
    doc = page.parent
    if doc is None or "blocks" not in data:
        return False
    names = broken_type3_names(doc, page)
    if not names or _broken_share(data, names) < _BROKEN_SHARE:
        return False
    reader = reader or Type3Reader(doc)
    lines = reader.lines(page.number)
    _drop_broken_lines(data, names)
    if not lines:
        return False
    built = [line for line in (_line_dict(line, reader) for line in lines) if line is not None]
    if not built:
        return False
    for group in _group_lines(built):
        data["blocks"].append({"type": 0, "bbox": _union_bbox(group), "lines": group})
    return True


# ---- Fonts and glyph positions ----


def _font_shapes(doc: pymupdf.Document, xref: int) -> dict[int, tuple[float, float, float, float]]:
    """Boxes in glyph space (FontMatrix applied) of the glyphs that draw something. Empty glyphs that only advance
    the pen are left out."""
    matrix = _numbers(doc.xref_get_key(xref, "FontMatrix")[1]) or [0.001, 0, 0, 0.001, 0, 0]
    font_box = _numbers(doc.xref_get_key(xref, "FontBBox")[1]) or [0, 0, 1000, 1000]
    procs = dict(re.findall(r"/([^\s/<>\[\]()]+)\s+(\d+)\s+0\s+R", doc.xref_get_key(xref, "CharProcs")[1]))
    shapes: dict[int, tuple[float, float, float, float]] = {}
    for code, name in _encoding_names(doc, xref).items():
        proc = procs.get(name)
        if proc is None:
            continue
        stream = doc.xref_stream(int(proc)) or b""
        found = re.search(rb"\bd([01])\b", stream)
        if found is None or not stream[found.end() :].strip():
            continue
        head = _numbers(stream[: found.start()].decode("latin-1"))
        box = head[2:6] if found.group(1) == b"1" and len(head) >= 6 else font_box[:4]
        scale_x, scale_y = matrix[0], matrix[3]
        x0, x1 = sorted((box[0] * scale_x, box[2] * scale_x))
        y0, y1 = sorted((box[1] * scale_y, box[3] * scale_y))
        shapes[code] = (x0, y0, x1, y1)
    return shapes


def _encoding_names(doc: pymupdf.Document, xref: int) -> dict[int, str]:
    kind, value = doc.xref_get_key(xref, "Encoding")
    source = doc.xref_object(int(value.split()[0])) if kind == "xref" else value
    found = re.search(r"/Differences\s*\[(.*?)\]", source, re.S)
    names: dict[int, str] = {}
    if found is None:
        return names
    code = 0
    # Names need not be separated by spaces: [65/a0/a1]
    for token in re.findall(r"/[^\s/\[\]<>()]+|\d+", found.group(1)):
        if token.isdigit():
            code = int(token)
        else:
            names[code] = token[1:]
            code += 1
    return names


def _numbers(raw: str) -> list[float]:
    """PDF numbers may omit the integer part: .001."""
    return [float(item) for item in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", raw)]


def _placed_codes(
    page: pymupdf.Page,
    fonts: dict[str, int],
    shapes: dict[int, dict[int, tuple[float, float, float, float]]],
) -> list[tuple[Key, float, float, float, int]]:
    """List the glyphs that draw something in content stream order: (key, origin x, origin y, font size, color).

    get_text drops zero-width glyphs at the same position as the previous character; texttrace keeps them.
    """
    placed: list[tuple[Key, float, float, float, int]] = []
    for span in page.get_texttrace():
        xref = fonts.get(str(span.get("font")))
        if xref is None or abs(span["dir"][1]) > 0.01:
            continue
        drawn = shapes.get(xref, {})
        color = _rgb(span.get("color"))
        for char in span["chars"]:
            if char[1] in drawn:
                placed.append(((xref, char[1]), char[2][0], char[2][1], float(span["size"]), color))
    return placed


def _glyph(
    key: Key,
    x: float,
    y: float,
    size: float,
    color: int,
    shapes: dict[int, dict[int, tuple[float, float, float, float]]],
    flip: bool,
) -> Glyph:
    x0, y0, x1, y1 = shapes[key[0]][key[1]]
    if flip:
        rect = pymupdf.Rect(x + x0 * size, y + y0 * size, x + x1 * size, y + y1 * size)
    else:
        rect = pymupdf.Rect(x + x0 * size, y - y1 * size, x + x1 * size, y - y0 * size)
    return Glyph(key, rect, color, (x, y))


def _y_flip(
    doc: pymupdf.Document,
    pages: dict[int, list[tuple[Key, float, float, float, int]]],
    shapes: dict[int, dict[int, tuple[float, float, float, float]]],
) -> bool:
    """Whether glyph space points up or down. Ghostscript bitmap fonts use an inverted Tm and draw glyphs below the
    origin.

    PyMuPDF does not expose the text matrix, so glyphs are placed both ways to see which one covers the ink on the
    page.
    """
    pno = max(pages, key=lambda number: len(pages[number]))
    pix = doc[pno].get_pixmap(dpi=36, colorspace=pymupdf.csGRAY, alpha=False)
    scale = 36 / 72

    def covered(flip: bool) -> int:
        mask = bytearray(pix.width * pix.height)
        for key, x, y, size, color in pages[pno]:
            rect = _glyph(key, x, y, size, color, shapes, flip).rect
            left, right = max(int(rect.x0 * scale), 0), min(int(rect.x1 * scale) + 1, pix.width)
            for row in range(max(int(rect.y0 * scale), 0), min(int(rect.y1 * scale) + 1, pix.height)):
                mask[row * pix.width + left : row * pix.width + right] = b"\x01" * max(right - left, 0)
        return sum(1 for hit, value in zip(mask, pix.samples) if hit and value < 128)

    return covered(True) > covered(False)


def _rules(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Short horizontal rules within lines, whether drawn as images or as vectors."""
    rects = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    try:
        rects.extend(pymupdf.Rect(drawing["rect"]) for drawing in page.get_drawings())
    except Exception:  # noqa: BLE001 - vector graphics of some pages cannot be read; treat as none
        pass
    return [rect for rect in rects if rect.height <= _RULE_HEIGHT and 1.5 <= rect.width <= _RULE_LENGTH]


def _rgb(color: object) -> int:
    if not isinstance(color, (tuple, list)) or len(color) < 3:
        return 0
    red, green, blue = (min(max(int(float(part) * 255 + 0.5), 0), 255) for part in color[:3])
    return (red << 16) | (green << 8) | blue


def _readable(glyphs: list[Glyph], labels: dict[Key, Label]) -> bool:
    known = sum(1 for glyph in glyphs if glyph.key in labels)
    return bool(glyphs) and known >= len(glyphs) * _READABLE


# ---- Lines, words and glyph roles ----


def _build_lines(glyphs: list[Glyph], labels: dict[Key, Label], owners: dict[int, int]) -> list[_Line]:
    """Arrange glyphs into lines by baseline, ordered by position like the text of ordinary PDFs, and not by content
    stream order.

    Each glyph with a recognized letter class has its own baseline, and glyphs with the same baseline form a line.
    Superscripts, subscripts, numerators and denominators are smaller and off the baseline, so they first form small
    lines of their own and are then merged back into the line they belong to; symbols and punctuation without a
    recognized class are merged by height. After merging, lines are cut at large horizontal gaps: side-by-side author
    names, two columns. In the content stream a denominator moves the pen back to the left of the fraction, and joining
    lines in stream order would attach the rest of the sentence after the fraction to the denominator.
    """
    placed: list[tuple[Glyph, float, float]] = []
    loose: list[Glyph] = []
    for glyph in glyphs:
        size = _glyph_size(glyph, labels)
        if size is None:
            loose.append(glyph)
        else:
            placed.append((glyph, size, _own_baseline(glyph, labels, size)))
    bands: list[list[tuple[Glyph, float, float]]] = []
    for item in sorted(placed, key=lambda item: item[2]):
        band = next((band for band in reversed(bands[-6:]) if _same_band(band, item)), None)
        if band is None:
            bands.append([item])
        else:
            band.append(item)
    ordered = [[item[0] for item in band] for band in sorted(bands, key=len, reverse=True)]
    rows: list[_Line] = []
    for piece in (piece for piece in ordered if len(piece) >= 3):
        # Body-text glyphs lowered as a whole (supplier below N) can be many, and are still a subscript of an earlier
        # piece
        host = _host_line(rows, piece, labels)
        if host is None:
            rows.append(_row(piece, labels))
        else:
            host.glyphs.extend(piece)
    pending = [*(piece for piece in ordered if len(piece) < 3), *([glyph] for glyph in loose)]
    while pending:
        pending = _attach(rows, pending, labels)
        if pending:
            # The remaining small pieces form lines of their own: the largest piece becomes a line first, then the
            # others are checked again for merging into it
            rows.append(_row(pending.pop(0), labels))
    lines: list[_Line] = []
    for row in rows:
        row.glyphs.sort(key=lambda glyph: glyph.rect.x0)
        _measure(row, labels)
        current = [row.glyphs[0]]
        right = row.glyphs[0].rect.x1
        for glyph in row.glyphs[1:]:
            if glyph.rect.x0 - right > row.size * _RUN_GAP:
                lines.append(_Line(current))
                current = []
            current.append(glyph)
            right = max(right, glyph.rect.x1)
        lines.append(_Line(current))
    # Thin rules that fall into no line are lines of a figure, not characters
    lines = [line for line in lines if any(glyph.key != _RULE for glyph in line.glyphs)]
    for line in lines:
        _measure(line, labels)
        line.tokens = _tokens(line, labels, owners)
    return sorted(lines, key=lambda line: (line.baseline, line.glyphs[0].rect.x0))


def _row(glyphs: list[Glyph], labels: dict[Key, Label]) -> _Line:
    row = _Line(list(glyphs))
    _measure(row, labels)
    return row


def _attach(rows: list[_Line], pending: list[list[Glyph]], labels: dict[Key, Label]) -> list[list[Glyph]]:
    """Merge small pieces into lines round by round until none can be merged: "1", "." and "1" reach the heading
    only one after another."""
    extents = {id(row): _extent(row) for row in rows}
    while pending:
        still: list[list[Glyph]] = []
        for piece in pending:
            host = _host_line(rows, piece, labels, extents)
            if host is None:
                still.append(piece)
                continue
            host.glyphs.extend(piece)
            left, right = extents[id(host)]
            extents[id(host)] = (
                min(left, *(glyph.rect.x0 for glyph in piece)),
                max(right, *(glyph.rect.x1 for glyph in piece)),
            )
        if len(still) == len(pending):
            return still
        pending = still
    return []


def _extent(row: _Line) -> tuple[float, float]:
    return min(glyph.rect.x0 for glyph in row.glyphs), max(glyph.rect.x1 for glyph in row.glyphs)


def _own_baseline(glyph: Glyph, labels: dict[Key, Label], size: float) -> float:
    kind = _letter_class(labels[glyph.key].text)
    return glyph.rect.y1 - (size * 0.19 if kind == "desc" else 0.0)


def _same_band(band: list[tuple[Glyph, float, float]], item: tuple[Glyph, float, float]) -> bool:
    """Glyphs whose baselines differ by less than 12% of the font size and whose font sizes are similar sit on the
    same baseline."""
    _glyph, size, baseline = item
    sizes = [entry[1] for entry in band[-8:]]
    base = statistics.median(entry[2] for entry in band[-8:])
    ref = statistics.median(sizes)
    return abs(baseline - base) <= min(size, ref) * 0.12 and max(size, ref) <= min(size, ref) * 1.25


def _host_line(
    lines: list[_Line],
    piece: list[Glyph],
    labels: dict[Key, Label],
    extents: dict[int, tuple[float, float]] | None = None,
) -> _Line | None:
    """The line that a small piece such as a superscript, subscript, fraction or punctuation joins: the piece's center
    lies within 0.75 font sizes of the middle of the line's x-height, and the piece is horizontally next to the line.
    Pieces in a larger size do not join lines in a smaller size."""
    left = min(glyph.rect.x0 for glyph in piece)
    right = max(glyph.rect.x1 for glyph in piece)
    center = statistics.median((glyph.rect.y0 + glyph.rect.y1) / 2 for glyph in piece)
    sizes = [size for glyph in piece if (size := _glyph_size(glyph, labels))]
    best: tuple[float, _Line] | None = None
    for line in lines:
        if sizes and statistics.median(sizes) > line.size * 1.1:
            continue
        offset = abs(center - (line.baseline - line.xheight / 2))
        if offset > line.size * 0.75 or (best is not None and offset >= best[0]):
            continue
        line_left, line_right = extents[id(line)] if extents else _extent(line)
        reach = line.size * _RUN_GAP
        if line_left - reach <= right and left <= line_right + reach:
            best = (offset, line)
    return best[1] if best is not None else None


def _measure(line: _Line, labels: dict[Key, Label]) -> None:
    sizes = [size for glyph in line.glyphs if (size := _glyph_size(glyph, labels))]
    line.size = statistics.median(sizes) if sizes else statistics.median(g.rect.height for g in line.glyphs) / 0.7
    sitting = [
        glyph
        for glyph in line.glyphs
        if (size := _glyph_size(glyph, labels))
        and size >= line.size * _SCRIPT
        and _letter_class(labels[glyph.key].text) in ("x", "tall", "i", "t")
    ]
    line.baseline = statistics.median(glyph.rect.y1 for glyph in (sitting or line.glyphs))
    line.xheight = line.size * _HEIGHT_RATIO["x"]


def _tokens(
    line: _Line,
    labels: dict[Key, Label],
    owners: dict[int, int],
    mono: dict[Key, float] | None = None,
    bearings: dict[Key, tuple[float, float]] | None = None,
) -> list[list[Glyph]]:
    tokens: list[list[Glyph]] = []
    mono = mono or {}
    bearings = bearings or {}
    gaps = [_gap(left, right, line, bearings) for left, right in zip(line.glyphs, line.glyphs[1:])]
    split = _space_split(gaps)
    for index, glyph in enumerate(line.glyphs):
        prev = tokens[-1][-1] if tokens else None
        pitch = (mono.get(prev.key) or mono.get(glyph.key)) if prev is not None else None
        same_word = (
            prev is not None and owners.get(id(prev)) is not None and owners.get(id(prev)) == owners.get(id(glyph))
        )
        if prev is not None and pitch and (prev.key in mono and glyph.key in mono or same_word):
            # One side is monospace and OCR also reads both as one word (the s in @db.stanford, which occurs only
            # once): decide by the cell width
            joined = _middle(glyph) - _middle(prev) <= pitch * 1.5
        else:
            joined = prev is not None and not _space_before(gaps[index - 1], prev, glyph, owners, split)
        if joined:
            tokens[-1].append(glyph)
        else:
            tokens.append([glyph])
    # No space precedes a comma or period. In monospace fonts the comma sits in the middle of its cell, with its ink
    # far from the previous character
    merged: list[list[Glyph]] = []
    for token in tokens:
        marks = all((labels.get(glyph.key) or Label("?", 0)).text in _TRAILING_MARKS for glyph in token)
        if merged and marks and token[0].rect.x0 - max(glyph.rect.x1 for glyph in merged[-1]) <= line.xheight:
            merged[-1].extend(token)
        else:
            merged.append(token)
    return merged


def _gap(left: Glyph, right: Glyph, line: _Line, bearings: dict[Key, tuple[float, float]]) -> float:
    """Blank space between the ink of two glyphs, in x-heights, minus the side bearings of both glyphs."""
    raw = (right.rect.x0 - left.rect.x1) / line.xheight
    return raw - bearings.get(left.key, (0.0, 0.0))[1] - bearings.get(right.key, (0.0, 0.0))[0]


def _space_before(gap: float, prev: Glyph, glyph: Glyph, owners: dict[int, int], split: float | None) -> bool:
    if gap <= _TIGHT:
        return False
    if gap >= _WIDE:
        return True
    if split is not None:
        return gap >= split
    left, right = owners.get(id(prev)), owners.get(id(glyph))
    if left is None or right is None:
        return gap > (_TIGHT + _WIDE) / 2
    return left != right


def _space_split(gaps: list[float]) -> float | None:
    """In a justified line all word spaces have the same width. The midpoint between the line's letter spacing and
    word spacing is the threshold; when the line has no clear word spacing, return None and leave it to the OCR
    words."""
    letters = [gap for gap in gaps if gap <= _TIGHT]
    spaces = [gap for gap in gaps if _WIDE <= gap <= 3.0]
    if len(letters) < 3 or len(spaces) < 2:
        return None
    return (statistics.median(letters) + statistics.median(spaces)) / 2


def _bearings(pages: dict[int, list[_Line]], labels: dict[Key, Label]) -> dict[Key, tuple[float, float]]:
    """Left and right side bearings of each glyph: how much wider or narrower than usual the letter spacing before
    and after it is in reliably recognized words.

    The hook of upright f reaches to the right, so the spacing after it is always narrower than after other letters,
    and the space in "of the" looks only half as wide.
    """
    after: dict[Key, list[float]] = defaultdict(list)
    before: dict[Key, list[float]] = defaultdict(list)
    every: list[float] = []
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            if not _sure_word(token, labels):
                continue
            for left, right in zip(token, token[1:]):
                gap = (right.rect.x0 - left.rect.x1) / line.xheight
                after[left.key].append(gap)
                before[right.key].append(gap)
                every.append(gap)
    if not every:
        return {}
    usual = statistics.median(every)

    def shift(values: list[float]) -> float:
        return max(-0.3, min(0.3, statistics.median(values) - usual)) if len(values) >= 3 else 0.0

    return {key: (shift(before.get(key, [])), shift(after.get(key, []))) for key in {*after, *before}}


def _word_owners(glyphs: list[Glyph], words: list[OcrWord]) -> dict[int, int]:
    """Index of the OCR word each glyph falls in. Words are split by it when the word spacing is ambiguous."""
    owners: dict[int, int] = {}
    centers = _GlyphCenters(glyphs)
    for index, word in enumerate(words):
        box = pymupdf.Rect(word.rect.x0 - 0.7, word.rect.y0 - 0.7, word.rect.x1 + 0.7, word.rect.y1 + 0.7)
        for glyph in centers.inside(box):
            owners[id(glyph)] = index
    return owners


def _bullets(
    pages: dict[int, list[_Line]],
    labels: dict[Key, Label],
    density: Callable[[set[Key]], dict[Key, float]],
) -> set[Key]:
    """Bullets: always alone at the start of a line, square, and filled with ink. OCR often reads a filled dot as e."""
    lines = [line for page_lines in pages.values() for line in page_lines]
    seen = Counter(glyph.key for line in lines for glyph in line.glyphs)
    starts = Counter(line.tokens[0][0].key for line in lines if line.tokens and len(line.tokens[0]) == 1)
    shapes: dict[Key, pymupdf.Rect] = {}
    for line in lines:
        for glyph in line.glyphs:
            shapes.setdefault(glyph.key, glyph.rect)
    keys = {
        key
        for key, count in starts.items()
        if count >= 2
        and count * 2 >= seen[key]
        and key != _RULE
        and 0.8 <= shapes[key].width / max(shapes[key].height, 0.1) <= 1.25
    }
    filled = density(keys) if keys else {}
    return {key for key in keys if filled.get(key, 0.0) >= _FILLED}


def _ink_share(doc: pymupdf.Document, glyphs: dict[int, list[Glyph]], keys: set[Key]) -> dict[Key, float]:
    """Share of the glyph box covered by ink."""
    found: dict[Key, float] = {}
    for pno, items in glyphs.items():
        for glyph in items:
            if glyph.key not in keys or glyph.key in found:
                continue
            pix = doc[pno].get_pixmap(dpi=_SHAPE_DPI, clip=glyph.rect, colorspace=pymupdf.csGRAY, alpha=False)
            ink = 1.0 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width] / 255
            found[glyph.key] = float(ink.mean()) if ink.size else 0.0
    return found


def _glyph_shapes(
    doc: pymupdf.Document,
    glyphs: dict[int, list[Glyph]],
    keys: set[Key],
) -> dict[Key, np.ndarray]:
    """For each glyph kind, take the bitmap of its first occurrence, place it in a square keeping its aspect ratio,
    block-average it to 16×16, subtract the mean and normalize; the dot product is the similarity."""
    first: dict[Key, tuple[int, Glyph]] = {}
    for pno, items in glyphs.items():
        for glyph in items:
            if glyph.key in keys and glyph.key not in first:
                first[glyph.key] = (pno, glyph)
    found: dict[Key, np.ndarray] = {}
    for key, (pno, glyph) in first.items():
        pix = doc[pno].get_pixmap(dpi=_SHAPE_DPI, clip=glyph.rect, colorspace=pymupdf.csGRAY, alpha=False)
        if pix.width < 2 or pix.height < 2:
            continue
        ink = 1.0 - np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width] / 255
        side = max(ink.shape)
        canvas = np.zeros((side, side))
        top, left = (side - ink.shape[0]) // 2, (side - ink.shape[1]) // 2
        canvas[top : top + ink.shape[0], left : left + ink.shape[1]] = ink
        edges = np.linspace(0, side, _SHAPE_GRID + 1).astype(int)[:-1]
        grid = np.add.reduceat(np.add.reduceat(canvas, edges, axis=0), edges, axis=1).flatten()
        grid -= grid.mean()
        norm = float(np.linalg.norm(grid))
        if norm > 0:
            found[key] = grid / norm
    return found


def _mono_kinds(pages: dict[int, list[_Line]], labels: dict[Key, Label]) -> dict[Key, float]:
    """Monospace glyphs and their cell widths.

    They are found in long words: five or more characters, the same distance between centers throughout, and a narrow
    character such as i, l or a comma. In body-text fonts the center distance between a narrow character and its
    neighbors is shorter, so the distances do not come out equal; digits have equal widths in every font and do not
    count. The monospace cell width is 0.525 font sizes; evenly spaced body-text words such as each and latter stay
    below 0.48. When at least three such words fall in the same cell-width band (one band for body text, one for
    footnotes), their glyphs are monospace glyphs.
    """
    candidates: list[tuple[float, list[Glyph]]] = []
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            gaps = _steps(token)
            if len(token) < 5 or min(gaps) <= 0 or max(gaps) > min(gaps) * 1.15:
                continue
            if statistics.median(gaps) < line.size * _MONO_PITCH:
                continue
            if any(_text_of(glyph, labels) in _NARROW for glyph in token):
                candidates.append((statistics.median(gaps), token))
    clusters: list[list[tuple[float, list[Glyph]]]] = []
    for value, token in sorted(candidates, key=lambda item: item[0]):
        if clusters and value <= clusters[-1][0][0] * 1.04:
            clusters[-1].append((value, token))
        else:
            clusters.append([(value, token)])
    mono: dict[Key, float] = {}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        pitch = statistics.median(value for value, _token in cluster)
        for _value, token in cluster:
            mono.update((glyph.key, pitch) for glyph in token if not _text_of(glyph, labels).isdigit())
    return mono


def _proportional_kinds(pages: dict[int, list[_Line]], mono: dict[Key, float]) -> set[Key]:
    """Glyphs that occur in words with uneven letter spacing: proportional fonts. Words in a monospace font have the
    same spacing throughout.

    Words such as (year), with upright parentheses around monospace characters, also have uneven spacing, so words
    containing monospace glyphs do not count.
    """
    found: set[Key] = set()
    for line in (line for lines in pages.values() for line in lines):
        for token in line.tokens:
            gaps = _steps(token)
            if len(token) < 3 or min(gaps) <= 0 or max(gaps) <= min(gaps) * 1.3:
                continue
            if not any(glyph.key in mono for glyph in token):
                found.update(glyph.key for glyph in token)
    return found


def _group_lines(lines: list[Line]) -> list[list[Line]]:
    """Lines within the leading that overlap horizontally and have similar font sizes are grouped into one block, as
    PyMuPDF does for ordinary fonts. Paragraph splitting is left to later steps.

    Leading is measured between baselines. Line boxes are stretched by tall brackets and subscripts, so overlapping
    boxes of two lines mean nothing.
    """
    ordered = sorted(lines, key=lambda line: (_baseline(line), float(line["bbox"][0])))
    groups: list[list[Line]] = []
    for line in ordered:
        box = pymupdf.Rect(line["bbox"])
        size = _line_size(line)
        best: list[Line] | None = None
        for group in groups:
            prev = pymupdf.Rect(group[-1]["bbox"])
            prev_size = _line_size(group[-1])
            step = _baseline(line) - _baseline(group[-1])
            if not (0 < step <= max(size, prev_size) * _LEADING):
                continue
            if min(box.x1, prev.x1) - max(box.x0, prev.x0) <= 0:
                continue
            if max(size, prev_size) > min(size, prev_size) * 1.15:
                continue
            best = group
        if best is None:
            groups.append([line])
        else:
            best.append(line)
    return groups


def _baseline(line: Line) -> float:
    spans = [span for span in line["spans"] if span["font"] in (_TEXT_FONT, _MONO_FONT)] or line["spans"]
    return float(max(spans, key=lambda span: float(span["size"]))["origin"][1])


def _line_size(line: Line) -> float:
    sizes = [float(span["size"]) for span in line["spans"] if span["font"] in (_TEXT_FONT, _MONO_FONT)]
    return max(sizes) if sizes else max(float(span["size"]) for span in line["spans"])


# ---- Shared helpers ----


def _custom_encoding(doc: pymupdf.Document, xref: int) -> bool:
    if doc.xref_get_key(xref, "ToUnicode")[0] != "null":
        return False
    kind, value = doc.xref_get_key(xref, "Encoding")
    if kind == "name":
        return False
    if kind != "xref":
        return True
    enc_xref = int(value.split()[0])
    names = re.findall(r"/([A-Za-z][A-Za-z0-9.]*)", doc.xref_object(enc_xref))
    custom = [name for name in names if _CUSTOM_NAME.fullmatch(name)]
    return len(custom) >= 8


def _broken_share(data: PageDict, names: set[str]) -> float:
    total = 0
    broken = 0
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                count = len(str(span.get("text") or ""))
                total += count
                if span.get("font") in names:
                    broken += count
    if broken < _MIN_BROKEN_CHARS or total == 0:
        return 0.0
    return broken / total


def _drop_broken_lines(data: PageDict, names: set[str]) -> None:
    """Remove the character codes. Images and normal fonts stay, and so do normal characters mixed into the same
    line."""
    kept: list[Block] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            kept.append(block)
            continue
        lines: list[Line] = []
        for line in block["lines"]:
            spans = [span for span in line["spans"] if span.get("font") not in names]
            if not spans:
                continue
            if len(spans) != len(line["spans"]):
                line = line.copy()
                line["spans"] = spans
                line["bbox"] = _union_bbox([{"bbox": span["bbox"]} for span in spans if span.get("bbox")])
            lines.append(line)
        if not lines:
            continue
        block["lines"] = lines
        kept.append(block)
    data["blocks"] = kept


def _union_bbox(lines: list[Line]) -> tuple[float, float, float, float]:
    box = pymupdf.Rect()
    for line in lines:
        raw = line.get("bbox")
        if raw:
            box |= pymupdf.Rect(raw)
    return (box.x0, box.y0, box.x1, box.y1)
