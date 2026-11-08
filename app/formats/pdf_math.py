"""Mark the spans on a page that belong to formulas: math fonts and alphabets, radicands,
fractions, accents, scripts and underlines."""

import math
import re
import struct
from collections.abc import Sequence
from typing import Literal

import pymupdf

from .pdf_blocks import _INK_PAD, CJK_RE, _by_chars, _font_is_math, _span_is_math
from .pdf_types import Line, LineItem, PageDict, Span, text_dict


def _line_box(line: Line) -> pymupdf.Rect:
    """Box of a line of text. Spans in math fonts only count one character height around the baseline.

    PyMuPDF sizes boxes by the font ascent and descent. CMSY and CMEX reserve that room for large brackets
    and radicals, so one arrow can stretch the line box to two lines high, and the translation would then
    start at the position of the line above.

    The analysis computes this several times per line, so it takes minima and maxima of coordinates instead
    of building a Rect per span and taking unions. MuPDF computes the union of two Rects in single
    precision; with two or more spans the result is stored in single precision as well, so the box equals
    the Rect union and comparisons close to a threshold do not change."""
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    count = 0
    for span in line["spans"]:
        if not span["text"].strip() or "bbox" not in span:
            continue
        left, top, right, bottom = _text_bounds(span)
        if left >= right or top >= bottom:
            continue
        x0, y0, x1, y1 = min(x0, left), min(y0, top), max(x1, right), max(y1, bottom)
        count += 1
    if not count:
        return pymupdf.Rect(line["bbox"])
    if count == 1:
        return pymupdf.Rect(x0, y0, x1, y1)
    return pymupdf.Rect(struct.unpack("4f", struct.pack("4f", x0, y0, x1, y1)))


def _text_bounds(span: Span) -> tuple[float, float, float, float]:
    """The four coordinates of _text_box."""
    x0, y0, x1, y1 = span["bbox"]
    ink = span.get("ink")
    if not _span_is_math(span) or not ink:
        return x0, y0, x1, y1
    top, bottom = ink[1] - _INK_PAD, ink[3] + _INK_PAD
    if span.get("ink_drawn"):
        # Ink measured by drawing the glyph alone is not clipped to the font box, because the font box is wrong
        # (the txsys radical reaches most of a character below the baseline). Include the origin: accents such
        # as a wide tilde lie wholly above it, and by ink alone they would leave the line they belong to
        base = float(span["origin"][1])
        return x0, min(top, base), x1, max(bottom, base)
    return x0, max(y0, top), x1, min(y1, bottom)


class _PageChars:
    """Per-character boxes of a page (see _read_char_boxes), read on first use. The underline, radical and
    accent marking steps share them, so each page is read once."""

    def __init__(self, page: pymupdf.Page):
        self.page = page
        self._boxes: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] | None = None

    def boxes(self) -> dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]]:
        if self._boxes is None:
            self._boxes = _read_char_boxes(self.page)
        return self._boxes


# How far below the baseline an underline lies (fraction of the font size). Underlines in the corpus are
# between 0.11 and 0.3; lines touching the baseline are strikethroughs or table rules, lines further down
# are fraction bars or table rules of the next line
_UNDERLINE_TOP = 0.05
_UNDERLINE_DEPTH = 0.35


def _mark_underlines(page_chars: _PageChars, data: PageDict, strokes: Sequence[pymupdf.Rect]) -> None:
    """Lines drawn under text (the marked columns {A, D, G} in example 6 of opt_views). The line is not part
    of the font, so it is recognized as a thin horizontal rule below the baseline, within the descender, and
    not wider than the line of text. The characters above it become separate spans marked underline and are
    sent as {u}. Rules under formula characters are fraction or radical bars and do not count."""
    if not strokes:
        return
    lines = [line for block in data.get("blocks", []) for line in block.get("lines", []) if line.get("spans")]
    # A fraction bar has the denominator right below it, an underline has line spacing below it. Measure the
    # glyph ink boxes, not the font boxes. A line below that is itself underlined (a column of links) is not
    # a denominator
    inks = [
        (id(span), pymupdf.Rect(span.get("ink") or span["bbox"]))
        for line in lines
        for span in line["spans"]
        if str(span.get("text") or "").strip()
        and not _underline_strokes(span, strokes, line["bbox"][0], line["bbox"][2])
    ]
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] | None = None
    seen: dict[tuple[float, float, float, float], bool] = {}

    def visible(rect: pymupdf.Rect) -> bool:
        key = (rect.x0, rect.y0, rect.x1, rect.y1)
        if key not in seen:
            seen[key] = _painted_stroke(page_chars.page, rect)
        return seen[key]

    for line in lines:
        spans = line["spans"]
        left = min(span["bbox"][0] for span in spans)
        right = max(span["bbox"][2] for span in spans)
        main = max(float(span.get("size") or 0) for span in spans)
        # Baseline of the body text in this line. A numerator is raised above it; underlined text sits on it
        body = [
            span
            for span in spans
            if float(span.get("size") or 0) >= main * 0.85 and str(span.get("text") or "").strip()
        ]
        baseline = float(_by_chars(body, lambda span: round(span["origin"][1], 1))) if body else 0.0
        out: list[Span] = []
        for span in spans:
            size = float(span.get("size") or 10.0)
            raised = abs(span["origin"][1] - baseline) > size * 0.15
            under = (
                []
                if size < main * 0.85 or raised
                else [
                    rect
                    for rect in _underline_strokes(span, strokes, left, right)
                    if not any(
                        other != id(span)
                        and rect.y1 - 0.5 <= ink.y0 <= rect.y1 + size * 0.25
                        and min(ink.x1, rect.x1) - max(ink.x0, rect.x0) > 0.5
                        for other, ink in inks
                    )
                    and visible(rect)
                ]
            )
            if not under:
                out.append(span)
                continue
            if chars is None:
                chars = page_chars.boxes()
            out.extend(_split_covered(span, under, chars, _line_edges(spans, chars), "underline"))
        line["spans"] = out


# Radical signs. The bar starts at the top right corner of the sign
_RADICALS = "√∛∜"


def _mark_radicands(page_chars: _PageChars, data: PageDict, strokes: Sequence[pymupdf.Rect]) -> None:
    """Mark the text under a radical bar, the radicand, as formula (the 81 of √81). When body text and
    formulas use the same fonts, the font cannot tell them apart: 81 would stay in the body text and the
    radical become a formula of its own, each typeset separately. Large CMEX radicals extract as the letter
    p, so every glyph in a large delimiter font is recognized by position: it is a radical only if a bar
    starts at its top right corner."""
    lines = [line for block in data.get("blocks", []) for line in block.get("lines", []) if line.get("spans")]
    # The font box of a large radical is taller than the glyph; the bar meets the top right of the ink
    radicals = [
        pymupdf.Rect(span["ink"]) if span.get("ink") else _glyph_box(span)
        for line in lines
        for span in line["spans"]
        if str(span.get("text") or "").strip()
        and (
            str(span.get("text") or "").strip()[-1:] in _RADICALS
            or _BIG_DELIMITER_FONT_RE.search(str(span.get("font") or ""))
        )
    ]
    bars = [
        stroke
        for stroke in strokes
        if any(
            abs(stroke.x0 - radical.x1) <= 1.5 and radical.y0 - 1.5 <= stroke.y0 <= radical.y0 + 2.0
            for radical in radicals
        )
    ]
    if not bars:
        return
    chars = page_chars.boxes()
    for line in lines:
        out: list[Span] = []
        for span in line["spans"]:
            size = float(span.get("size") or 10.0)
            base = float(span["origin"][1])
            box = pymupdf.Rect(span["bbox"])
            over = (
                []
                if _span_is_math(span) or not str(span.get("text") or "").strip()
                else [
                    bar
                    for bar in bars
                    if base - size * 1.1 <= bar.y0 <= base - size * 0.5
                    and min(bar.x1, box.x1) - max(bar.x0, box.x0) > 0.5
                ]
            )
            out.extend(_split_covered(span, over, chars, _line_edges(line["spans"], chars), "math") if over else [span])
        line["spans"] = out


# A long arrow or bar over a word covers at most this many characters; longer ones are table separators
_OVERLAY_CHARS = 12


def _mark_overlays(page_chars: _PageChars, data: PageDict) -> None:
    """Vector arrows drawn with glyphs over a word (esvect's \vv{king} extracts as "#    »") and overlines:
    spans with no letters or digits whose ink is a thin band wider than a character, resting on the ink of
    another word. Together with the covered characters they form one formula and are marked as such;
    otherwise the arrow would be typeset as text and come out as "# » man" piled together."""
    for block in data.get("blocks", []):
        spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
        overlays = []
        for span in spans:
            text = str(span.get("text") or "").strip()
            ink = span.get("ink")
            size = float(span.get("size") or 10.0)
            if not text or ink is None or any(char.isalnum() for char in text):
                continue
            box = pymupdf.Rect(ink)
            if box.width >= size * 0.8 and box.height <= size * 0.45:
                overlays.append((span, box, size))
        if not overlays:
            continue
        chars = page_chars.boxes()
        for overlay, box, size in overlays:
            covered: list[tuple[Span, pymupdf.Rect]] = []
            for span in spans:
                ink = span.get("ink")
                if span is overlay or ink is None or not any(char.isalpha() for char in str(span.get("text") or "")):
                    continue
                word = pymupdf.Rect(ink)
                if min(word.x1, box.x1) - max(word.x0, box.x0) <= 0:
                    continue
                # Compare baselines: the ink top of a whole span is raised by later letters with ascenders (the l
                # of close). The arrow's own baseline is less than a character height above the word's; dots,
                # ellipses and fill-in blanks of the line above are a whole line away
                base = float(span["origin"][1])
                lift = base - float(overlay["origin"][1])
                if 0 <= lift <= size * 0.8 and base - size * 1.3 <= box.y0 and box.y1 <= base - size * 0.45:
                    covered.append((span, word))
            if not covered:
                continue
            boxes = [
                (x0, x1)
                for span, _word in covered
                for x0, x1, _origin in chars.get(
                    (str(span["text"]), round(span["origin"][0], 1), round(span["origin"][1], 1))
                )
                or []
                if box.x0 <= (x0 + x1) / 2 <= box.x1
            ]
            if not boxes or len(boxes) > _OVERLAY_CHARS:
                continue
            under = pymupdf.Rect(min(x0 for x0, _ in boxes), box.y0, max(x1 for _, x1 in boxes), box.y1)
            overlay["math"] = True
            targets = {id(span) for span, _word in covered}
            for line in block.get("lines", []):
                out: list[Span] = []
                for span in line.get("spans", []):
                    if id(span) in targets:
                        out.extend(_split_covered(span, [under], chars, ([under.x0], [under.x1]), "math"))
                    else:
                        out.append(span)
                line["spans"] = out


# Boxes from the formula detection model often include the punctuation after a formula ($x$, boxed as "x,").
# The punctuation stays in the body text so the translation can use Chinese punctuation
_FORMULA_TRAIL = ",.;:?!"
# A box holding only numbers (0, .5, (0, 1)) looks like body text and is not a formula, so that one sentence
# does not have 0 as a placeholder and 1 as text
_NUMERIC_RE = re.compile(r"[\d\s.,()%+\-−]*")


def _mark_detected_formulas(page_chars: _PageChars, data: PageDict, rects: Sequence[pymupdf.Rect]) -> None:
    """Cut out the characters inside inline formula boxes from the formula detection model and mark them
    detected_formula. The mark only counts when recognizing inline formula spans (see _is_formula_span);
    roles, skipping and paragraph joining still use the font-based math mark."""
    if not rects:
        return
    chars = page_chars.boxes()
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            box = pymupdf.Rect(line["bbox"])
            here = [
                kept
                for rect in rects
                if min(rect.y1, box.y1) > max(rect.y0, box.y0) and min(rect.x1, box.x1) > max(rect.x0, box.x0)
                for kept in [_trim_formula_box(rect, spans, chars)]
                if kept is not None
            ]
            if not here:
                continue
            out: list[Span] = []
            for span in spans:
                hit = (
                    []
                    if _span_is_math(span) or not span["text"].strip()
                    else [rect for rect in here if _box_holds_span(rect, span)]
                )
                # Character edges are matched to the box edges. The box comes from an image and is slightly off,
                # so exact alignment as for underlines cannot be required
                out.extend(
                    _split_covered(span, hit, chars, ([r.x0 for r in hit], [r.x1 for r in hit]), "detected_formula")
                    if hit
                    else [span]
                )
            line["spans"] = out


def _box_holds_span(rect: pymupdf.Rect, span: Span) -> bool:
    """The box and the span overlap horizontally and by at least half vertically, so boxes of the lines above
    and below that touch this line do not count."""
    box = pymupdf.Rect(span["bbox"])
    return min(rect.y1, box.y1) - max(rect.y0, box.y0) >= 0.5 * min(box.height, rect.height) and min(
        rect.x1, box.x1
    ) > max(rect.x0, box.x0)


def _trim_formula_box(
    rect: pymupdf.Rect,
    spans: list[Span],
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]],
) -> pymupdf.Rect | None:
    """Fit the box to the characters of this line: give up trailing punctuation, and drop boxes that only
    cover numbers."""
    covered: list[tuple[float, str]] = []
    for span in spans:
        if not _box_holds_span(rect, span):
            continue
        boxes = chars.get((str(span["text"]), round(span["origin"][0], 1), round(span["origin"][1], 1))) or []
        covered += [
            (x0, char)
            for char, (x0, x1, _origin) in zip(str(span["text"]), boxes)
            if rect.x0 - 0.5 <= (x0 + x1) / 2 <= rect.x1 + 0.5
        ]
    covered.sort()
    trimmed = pymupdf.Rect(rect)
    while len(covered) > 1 and covered[-1][1] in _FORMULA_TRAIL:
        trimmed.x1 = covered.pop()[0]
    if covered and _NUMERIC_RE.fullmatch("".join(char for _x, char in covered)):
        return None
    return trimmed


# Accent glyphs drawn on their own (\hat, \bar, the ´ of Montr\'eal). ASCII ^ ~ ` have other uses in text and
# code and are not included
_STACKED_ACCENTS = "ˆ˜¯´˙¨ˇ˘˚"
# A ˆ or ˜ that takes up its own width is the ^ or ~ of code, a regex or a URL (r"^The", nyu.edu/~roweis);
# monospaced fonts draw them with these glyphs too
_ASCII_ACCENTS = {"ˆ": "^", "˜": "~"}


def _split_accents(page_chars: _PageChars, data: PageDict) -> None:
    """An accent often shares a span with the character before it ("(ˆ", "log ˆ"), and per span it cannot be
    seen that it sits over the next character or formula. Cut it into a piece of its own: over a formula it
    joins the formula, over a letter it is composed into ŷ or é before translation. A ˆ or ˜ that takes up
    its own width is not an accent; it becomes ^ or ~ again and is not composed with the next letter."""
    lines = [line for block in data.get("blocks", []) for line in block.get("lines", []) if line.get("spans")]
    if not any(
        any(char in _STACKED_ACCENTS for char in text)
        for line in lines
        for text in (str(span.get("text") or "") for span in line["spans"])
    ):
        return
    chars = page_chars.boxes()
    for line in lines:
        out: list[Span] = []
        spans = line["spans"]
        for index, span in enumerate(spans):
            text = str(span.get("text") or "")
            boxes = chars.get((text, round(span["origin"][0], 1), round(span["origin"][1], 1)))
            if not any(char in _STACKED_ACCENTS for char in text) or boxes is None or len(boxes) != len(text):
                out.append(span)
                continue
            # Left edge of the next character: the next one in this span, or the start of the next span
            follow = [x0 for x0, _x1, _origin in boxes[1:]]
            follow.append(spans[index + 1]["bbox"][0] if index + 1 < len(spans) else float("inf"))
            # An accent stacked over the next character takes no width; the next character starts inside it
            stacked = [
                char in _STACKED_ACCENTS and after < x1 - (x1 - x0) * 0.3
                for char, (x0, x1, _origin), after in zip(text, boxes, follow)
            ]
            # Right edge of the previous character: an accent stacked over it (sˆ in a formula) is an accent too
            before = [spans[index - 1]["bbox"][2] if index else float("-inf")]
            before += [x1 for _x0, x1, _origin in boxes[:-1]]
            # It takes up its own width only if more text follows on the line, starting at its right edge.
            # A ˆ alone on a line sits over another line
            spaced = "".join(
                _ASCII_ACCENTS.get(char, char)
                if not on and after != float("inf") and prior <= x0 + (x1 - x0) * 0.3
                else char
                for char, on, (x0, x1, _origin), prior, after in zip(text, stacked, boxes, before, follow)
            )
            if spaced != text:
                text = spaced
                span = {**span, "text": text}
            if len(text.strip()) <= 1:
                out.append(span)
                continue
            cuts = sorted(
                {cut for index, on in enumerate(stacked) if on for cut in (index, index + 1)} - {0, len(text)}
            )
            for start, end in zip([0, *cuts], [*cuts, len(text)]):
                out.append(
                    span
                    if not cuts
                    else {
                        **span,
                        "text": text[start:end],
                        "origin": boxes[start][2],
                        "bbox": (boxes[start][0], span["bbox"][1], boxes[end - 1][1], span["bbox"][3]),
                        **(
                            {"ink": (boxes[start][0], span["ink"][1], boxes[end - 1][1], span["ink"][3])}
                            if "ink" in span
                            else {}
                        ),
                    }
                )
        line["spans"] = out


VISIBLE_SPACE = "\u2423"


def _mark_visible_spaces(page_chars: _PageChars, data: PageDict, cups: Sequence[pymupdf.Rect]) -> None:
    """A drawn visible space ␣ is not in the font and extracts as a plain space or as nothing, so
    "merged to ␣re" reads as "merged to re". Where one sits on the baseline of a line in a gap between
    characters, write ␣ in its place: it replaces the space it covers, or is added as a new span when the gap
    holds no character. When it is apart from the characters before or after it, a space is kept between."""
    if not cups:
        return
    lines = [
        line
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        if line.get("spans") and abs(line["dir"][1]) <= 0.01
    ]
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] | None = None
    for cup in cups:
        middle = (cup.x0 + cup.x1) / 2
        row = [
            (line, span)
            for line in lines
            for span in line["spans"]
            if abs(float(span["origin"][1]) - cup.y1) <= float(span.get("size") or 10.0) * 0.12
            and span["bbox"][1] < cup.y0
        ]
        covering = [(line, span) for line, span in row if span["bbox"][0] < middle < span["bbox"][2]]
        if covering:
            line, span = covering[0]
            text = str(span.get("text") or "")
            if chars is None:
                chars = page_chars.boxes()
            boxes = chars.get((text, round(span["origin"][0], 1), round(span["origin"][1], 1)))
            if boxes is None or len(boxes) != len(text):
                boxes = [(span["bbox"][0], span["bbox"][2], span["origin"])] if len(text) == 1 else []
            hit = next((i for i, (x0, x1, _o) in enumerate(boxes) if x0 <= middle <= x1), None)
            if hit is None or not text[hit].isspace():
                continue
            x0, x1, _origin = boxes[hit]
            lead = " " if cup.x0 - x0 > 1.0 and (hit == 0 or not text[hit - 1].isspace()) else ""
            tail = " " if x1 - cup.x1 > 1.0 and hit + 1 < len(text) and not text[hit + 1].isspace() else ""
            span["text"] = text[:hit] + lead + VISIBLE_SPACE + tail + text[hit + 1 :]
            continue
        # No character in the gap: attach it to the nearer span, after a span on the left or before one on the
        # right. It does not count if neither side has a character within one em
        inked = [(line, span) for line, span in row if str(span.get("text") or "").strip()]
        before = [item for item in inked if item[1]["bbox"][2] <= middle]
        after = [item for item in inked if item[1]["bbox"][0] >= middle]
        left = max(before, key=lambda item: item[1]["bbox"][2], default=None)
        right = min(after, key=lambda item: item[1]["bbox"][0], default=None)
        size = float((left or right or (None, {}))[1].get("size") or 10.0)
        near_left = left is not None and cup.x0 - left[1]["bbox"][2] <= size
        near_right = right is not None and right[1]["bbox"][0] - cup.x1 <= size
        if not near_left and not near_right:
            continue
        gap_left = cup.x0 - left[1]["bbox"][2] if near_left and left is not None else float("inf")
        gap_right = right[1]["bbox"][0] - cup.x1 if near_right and right is not None else float("inf")
        to_left = gap_left <= gap_right
        chosen = left if to_left else right
        # At least one side is near, so its gap is finite and the side picked by the smaller gap exists
        assert chosen is not None
        line, neighbor = chosen
        spans = line["spans"]
        place = spans.index(neighbor) + (1 if to_left else 0)
        lead = " " if gap_left > 1.0 and (to_left or place > 0) else ""
        tail = " " if gap_right > 1.0 and (not to_left or place < len(spans)) else ""
        box = (cup.x0, neighbor["bbox"][1], cup.x1, neighbor["bbox"][3])
        spans.insert(
            place,
            {
                **neighbor,
                "text": lead + VISIBLE_SPACE + tail,
                "bbox": box,
                "origin": (cup.x0, neighbor["origin"][1]),
                **({"ink": (cup.x0, cup.y0, cup.x1, cup.y1)} if "ink" in neighbor else {}),
            },
        )
        line["bbox"] = tuple(pymupdf.Rect(line["bbox"]) | pymupdf.Rect(box))


def _glyph_box(span: Span) -> pymupdf.Rect:
    """Box taken by the glyph. When the font box is wrong and the ink was measured by drawing the glyph
    alone, the top and bottom come from the ink."""
    box = pymupdf.Rect(span["bbox"])
    ink = span.get("ink")
    if ink and span.get("ink_drawn"):
        return pymupdf.Rect(box.x0, ink[1], box.x1, ink[3])
    return box


def _painted_stroke(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    """The line is actually visible: get_drawings also lists lines outside the clip path or covered by other
    content. Render the thin band the line occupies and require dark pixels in almost every column.
    Descenders only cross a few columns."""
    band = pymupdf.Rect(rect.x0, rect.y0 - 0.4, rect.x1, rect.y1 + 0.4)
    try:
        pix = page.get_pixmap(clip=band, dpi=144, colorspace=pymupdf.csGRAY, alpha=False)
    except Exception:  # noqa: BLE001 - not recognized if it cannot be rendered
        return False
    if pix.width <= 0 or pix.height <= 0:
        return False
    samples = pix.samples
    dark = sum(1 for x in range(pix.width) if any(samples[y * pix.stride + x] < 200 for y in range(pix.height)))
    return bool(dark >= pix.width * 0.8)


def _underline_strokes(
    span: Span,
    strokes: Sequence[pymupdf.Rect],
    left: float,
    right: float,
) -> list[pymupdf.Rect]:
    text = str(span.get("text") or "")
    if not text.strip() or _span_is_math(span):
        return []
    size = float(span.get("size") or 10.0)
    base = float(span["origin"][1])
    box = pymupdf.Rect(span["bbox"])
    return [
        rect
        for rect in strokes
        if base + size * _UNDERLINE_TOP <= rect.y0
        and rect.y1 <= base + size * _UNDERLINE_DEPTH
        and rect.x0 >= left - 1
        and rect.x1 <= right + 1
        and min(rect.x1, box.x1) - max(rect.x0, box.x0) > 0.5
    ]


def _read_char_boxes(
    page: pymupdf.Page,
) -> dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]]:
    found: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] = {}
    raw = text_dict(page, "rawdict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars") or []
                text = "".join(char["c"] for char in chars)
                key = (text, round(span["origin"][0], 1), round(span["origin"][1], 1))
                found[key] = [(char["bbox"][0], char["bbox"][2], char["origin"]) for char in chars]
    return found


def _line_edges(
    spans: list[Span],
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]],
) -> tuple[list[float], list[float]]:
    """Left and right edges of every character in a line."""
    lefts: list[float] = []
    rights: list[float] = []
    for span in spans:
        boxes = chars.get((str(span["text"]), round(span["origin"][0], 1), round(span["origin"][1], 1))) or []
        lefts += [x0 for x0, _x1, _ox in boxes]
        rights += [x1 for _x0, x1, _ox in boxes]
    return lefts, rights


def _split_covered(
    span: Span,
    under: list[pymupdf.Rect],
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]],
    edges: tuple[list[float], list[float]] = ([], []),
    flag: Literal["underline", "math", "detected_formula"] = "underline",
) -> list[Span]:
    """Cut a span by character into pieces covered by the lines and pieces outside them, and mark the covered
    pieces with flag (underline, or the number under a radical). Without matching per-character boxes, the
    span is marked only if all of it lies within a line."""
    text = str(span["text"])
    boxes = chars.get((text, round(span["origin"][0], 1), round(span["origin"][1], 1)))

    def covered(x0: float, x1: float) -> bool:
        middle = (x0 + x1) / 2
        return any(rect.x0 - 0.5 <= middle <= rect.x1 + 0.5 for rect in under)

    if boxes is None or len(boxes) != len(text):
        box = pymupdf.Rect(span["bbox"])
        if (
            all(rect.x0 - 1 <= box.x0 for rect in under)
            and covered(box.x0, box.x0 + 0.1)
            and covered(box.x1 - 0.1, box.x1)
        ):
            marked = span.copy()
            marked[flag] = True
            return [marked]
        return [span]
    # An underline runs from the left edge of one character to the right edge of another. Dashed lines in
    # figures, legend lines and fraction bars do not match character edges. The text under one line may be
    # split into several spans, so the edges of the whole line are used
    reach = max(1.0, float(span.get("size") or 10.0) * 0.15)
    lefts = edges[0] or [x0 for x0, _x1, _ox in boxes]
    rights = edges[1] or [x1 for _x0, x1, _ox in boxes]
    under = [
        rect
        for rect in under
        if any(abs(rect.x0 - x0) <= reach for x0 in lefts) and any(abs(rect.x1 - x1) <= reach for x1 in rights)
    ]
    # The underscore character itself is not underlined text (some PDFs draw _ as a line)
    marks = [covered(x0, x1) and char != "_" for char, (x0, x1, _ox) in zip(text, boxes)]
    if not any(marks):
        return [span]
    pieces: list[Span] = []
    start = 0
    y0, y1 = span["bbox"][1], span["bbox"][3]
    for index in range(1, len(text) + 1):
        if index < len(text) and marks[index] == marks[start]:
            continue
        x0, x1 = boxes[start][0], boxes[index - 1][1]
        # The origin is that of the piece's first character. The span's origin belongs to its first character:
        # in " ." the space sits on the baseline of the subscript before it, and a period copying that origin
        # would not be found by baseline when erasing and would stay on the page
        piece: Span = {**span, "text": text[start:index], "bbox": (x0, y0, x1, y1), "origin": boxes[start][2]}
        # The page still holds the span from before the cut; cropping formulas uses it to tell which characters
        # belong to the formula (see _outside_formula)
        piece["source"] = span.get("source") or span["bbox"]
        if "ink" in span:
            piece["ink"] = (x0, span["ink"][1], x1, span["ink"][3])
        if marks[start]:
            piece[flag] = True
        else:
            piece.pop(flag, None)
        pieces.append(piece)
        start = index
    return pieces


def _script_parts(line: Line, spans: list[Span]) -> list[LineItem]:
    """On a page in a Western language, a line with Chinese text, a gap of half an em or more and then Western
    text is two columns side by side (an epigraph with the Chinese original on the left and the English
    translation on the right). Split it into two pieces: the Chinese piece is kept as foreign text and the
    Western piece is translated. Returns an empty list for other lines"""
    for k in range(1, len(spans)):
        if _script_seam(spans[:k], spans[k:]):
            parts: list[LineItem] = []
            for part in (spans[:k], spans[k:]):
                sub: Line = {**line, "spans": part}
                parts.append(("".join(s["text"] for s in part), _line_box(sub), part, part))
            return parts
    return []


def _script_seam(left: list[Span], right: list[Span]) -> bool:
    """The left piece is Chinese only, the right piece starts with Western text, and the gap between them is
    half an em or more (see _script_parts)"""
    left = [span for span in left if str(span.get("text") or "").strip()]
    right = [span for span in right if str(span.get("text") or "").strip()]
    if not left or not right:
        return False
    text = "".join(str(span.get("text") or "") for span in left)
    size = float(right[0].get("size") or 10)
    return (
        bool(CJK_RE.search(text))
        and not re.search(r"[A-Za-z]", text)
        and re.match(r"\s*[A-Za-z]", str(right[0].get("text") or "")) is not None
        and float(right[0]["bbox"][0]) - float(left[-1]["bbox"][2]) >= size * 0.5
    )


# Upright Computer Modern fonts. When the body text is set in CM they are body text; when it is set in Times
# or similar, they only appear in formulas (\mathrm, \log, brackets, equals signs, \mathbf, bold sans-serif
# vectors w, x).
_CM_UPRIGHT_RE = re.compile(r"^(?:[A-Z]{6}\+)?(?:CMR|CMBX|CMSS(?:BX|I)?|CMSL)\d", re.I)


# Mathematical alphanumeric symbols (𝑥, 𝜁, 𝐀). A font whose ToUnicode maps letters here is a formula font
_MATH_ALPHABET_RE = re.compile(r"[\U0001D400-\U0001D7FF\u210E]")


def _mark_math_alphabet(pages: list[PageDict]) -> None:
    """Mark every span math in fonts that map most of their letters to the mathematical alphanumeric block.
    This catches math italics that the font name lists miss (the names newtxmath gets with various text
    fonts), and commas and brackets in the same font count as formula as well."""
    letters: dict[str, int] = {}
    mathy: dict[str, int] = {}
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    font = str(span.get("font") or "")
                    text = str(span.get("text") or "")
                    letters[font] = letters.get(font, 0) + sum(char.isalpha() for char in text)
                    mathy[font] = mathy.get(font, 0) + len(_MATH_ALPHABET_RE.findall(text))
    fonts = {font for font, count in mathy.items() if count and count * 2 >= letters[font]}
    if not fonts:
        return
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span.get("font") in fonts:
                        span["math"] = True


def _mark_formula_only_fonts(pages: list[PageDict]) -> None:
    """Mark every span math in fonts used only for one or two characters at a time, mostly next to formulas.
    A bitmap double-struck 1 (𝟙) is a Type 3 font named only by a number such as F74, and extracts as a
    plain 1."""
    spans_of: dict[str, int] = {}
    touching: dict[str, int] = {}
    longer: set[str] = set()
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                shown = [span for span in line["spans"] if str(span.get("text") or "").strip()]
                for index, span in enumerate(shown):
                    font = str(span.get("font") or "")
                    if _span_is_math(span):
                        continue
                    if len(span["text"].strip()) > 2:
                        longer.add(font)
                        continue
                    spans_of[font] = spans_of.get(font, 0) + 1
                    box = span["bbox"]
                    neighbours = shown[max(index - 1, 0) : index] + shown[index + 1 : index + 2]
                    if any(
                        _span_is_math(other)
                        and min(abs(other["bbox"][0] - box[2]), abs(box[0] - other["bbox"][2])) < 1.5
                        for other in neighbours
                    ):
                        touching[font] = touching.get(font, 0) + 1
    fonts = {font for font, count in spans_of.items() if font not in longer and touching.get(font, 0) * 2 >= count}
    if not fonts:
        return
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span.get("font") in fonts:
                        span["math"] = True


def _mark_math_roman(pages: list[PageDict]) -> None:
    """In documents whose body text is not Computer Modern, spans in upright CM fonts are formulas; mark them
    math.

    Font name lists can only say what a font usually is. CMR is the body text in a paper set in CM, but in a
    paper set in Times it is only used for log, min, brackets and equals signs, so the body font of this
    document decides.
    """
    counts: dict[str, int] = {}
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    font = str(span.get("font") or "").split("+")[-1]
                    counts[font] = counts.get(font, 0) + len(span["text"].strip())
    if not counts:
        return
    body = max(counts, key=counts.__getitem__)
    if _CM_UPRIGHT_RE.match(body) or _font_is_math(body):
        return
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if _CM_UPRIGHT_RE.match(str(span.get("font") or "")):
                        span["math"] = True


def _prose_units(text: str) -> int:
    """Number of prose words: Latin words (2 or more letters) plus CJK characters / 2. Tells a line of
    formulas from a sentence containing formulas."""
    latin = len(re.findall(r"[A-Za-z]{2,}", text))
    cjk = len(CJK_RE.findall(text))
    return latin + cjk // 2


# How far the numerator baseline is above the fraction bar and the denominator baseline below it (fraction of
# the font size). The bar is on the math axis; the baselines of an equals sign or label beside it are a
# little lower
_NUMERATOR_RISE = (0.1, 0.75)
_DENOMINATOR_DROP = (0.4, 1.3)
_FRACTION_AXIS = 0.5
# Tolerance for centering numerator and denominator (fraction of the bar length), and the distance from the
# bar ends within which neighboring text counts as next to it (fraction of the font size)
_FRACTION_CENTER = 0.1
_FRACTION_NEIGHBOR = 1.5


def _fraction_parts(data: PageDict, rules: Sequence[pymupdf.Rect]) -> set[int]:
    """The text above and below fraction bars whose numerator or denominator contains prose words
    (labeled recall = # of correct ... / # of total ...).

    The bar is a thin horizontal rule. Numerator and denominator are each one line, lie within the length
    of the bar and are centered on it, with no other text next to the bar ends on the same line; outside the
    bar ends, on the math axis, there is an equals sign, a label or similar. A link underline is followed by
    more body text and a table rule has no text beyond its ends, so neither matches. Purely mathematical
    fractions are not handled here: inline fractions in a paragraph split a line into pieces, and counting
    words per piece is unreliable.
    """
    spans = [
        span
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        if abs(line["dir"][1]) <= 0.01
        for span in line["spans"]
        if span["text"].strip()
    ]
    found: set[int] = set()
    for bar in rules:
        axis = (bar.y0 + bar.y1) / 2
        inside = [span for span in spans if bar.x0 - 1 <= span["bbox"][0] and span["bbox"][2] <= bar.x1 + 1]
        over = [span for span in inside if _within(axis - span["origin"][1], span["size"], _NUMERATOR_RISE)]
        under = [span for span in inside if _within(span["origin"][1] - axis, span["size"], _DENOMINATOR_DROP)]
        if not over or not under or max(_plain_units(over), _plain_units(under)) < 2:
            continue
        inner = {id(span) for span in inside}
        if not all(_centered_alone(row, spans, inner, bar) for row in (over, under)):
            continue
        if any(_beside_bar(span, bar, axis) for span in spans if id(span) not in inner):
            found.update(id(span) for span in (*over, *under))
    return found


def _plain_units(spans: list[Span]) -> int:
    return _prose_units(" ".join(span["text"] for span in spans if not _span_is_math(span)))


def _within(offset: float, size: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] * size <= offset <= bounds[1] * size


def _centered_alone(row: list[Span], spans: list[Span], inner: set[int], bar: pymupdf.Rect) -> bool:
    x0 = min(span["bbox"][0] for span in row)
    x1 = max(span["bbox"][2] for span in row)
    if abs((x0 + x1) / 2 - (bar.x0 + bar.x1) / 2) > max(1.5, bar.width * _FRACTION_CENTER):
        return False
    baselines = {round(span["origin"][1]) for span in row}
    for span in spans:
        if id(span) in inner or not any(abs(span["origin"][1] - y) <= 1 for y in baselines):
            continue
        reach = span["size"] * _FRACTION_NEIGHBOR
        if span["bbox"][2] > bar.x0 - reach and span["bbox"][0] < bar.x1 + reach:
            return False
    return True


def _beside_bar(span: Span, bar: pymupdf.Rect, axis: float) -> bool:
    """Text beyond a bar end near the math axis: an equals sign or label before the fraction, an operator
    after it."""
    if not 0 <= span["origin"][1] - axis <= span["size"] * _FRACTION_AXIS:
        return False
    reach = span["size"] * _FRACTION_NEIGHBOR
    return bool(bar.x0 - reach <= span["bbox"][2] <= bar.x0 + 1 or bar.x1 - 1 <= span["bbox"][0] <= bar.x1 + reach)


# Fonts of extensible delimiters such as large braces and summation signs. Inline formulas use them too, so
# only blocks away from the left margin are recognized
_BIG_DELIMITER_FONT_RE = re.compile(r"CMEX|cmex|LMMathExtension", re.I)
