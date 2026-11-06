"""Find algorithm boxes and the pseudocode rows inside them."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pymupdf

from .pdf_drawings import (
    _NEAR,
    RuledTable,
    _rule_segments,
    _TextWord,
)
from .pdf_types import PdfDict

# Boundary between two adjacent rows of an algorithm box, as a share of the line spacing below the upper baseline
_ROW_CUT = 0.35


_SYMBOL_FONT_RE = re.compile(r"(?:^|\+)(?:CMSY|CMEX|MSAM|MSBM)", re.I)


_ALGORITHM_RE = re.compile(r"^\s*(?:Algorithm|Procedure|算法)\s*\d+", re.I)


_ALGORITHM_WORD_RE = re.compile(r"Algorithm|Procedure|算法", re.I)


def algorithm_tables(
    page: pymupdf.Page,
    drawings: list[PdfDict] | None = None,
    words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """Treat pseudocode in an algorithm box as a one-column table with one cell per line.

    Pseudocode has one statement per line and indentation is structure; the lines do not form a paragraph. PyMuPDF
    often puts several lines into one block, and later merging would join them into a reflowed paragraph. As cells,
    each line is translated on its own, written back to its original line and not merged with other lines. The extent
    of an algorithm box is set by its frame rules: from the rule below the caption to the next rule below it.
    words holds the words of this page. A page where no word contains Algorithm, Procedure or "算法" has no algorithm
    caption, so the text of the whole page need not be read again.
    """
    if words is not None and not any(_ALGORITHM_WORD_RE.search(word.text) for word in words):
        return ()
    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return ()
    lines = [
        line
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        if abs(line["dir"][1]) <= 0.01
    ]
    captions = [caption for line in lines for caption in _algorithm_captions(line)]
    if not captions:
        return ()
    horizontal, _vertical = _rule_segments(page, drawings)
    tables: list[RuledTable] = []
    for box, size in captions:
        body = _box_below(box, size, horizontal) or _box_above(box, size, horizontal)
        if body is None:
            continue
        rows = _code_rows(lines, body, horizontal)
        if len(rows) >= 2:
            tables.append(RuledTable(body, rows, algorithm=True))
    return tuple(tables)


def _algorithm_captions(line: PdfDict) -> list[tuple[pymupdf.Rect, float]]:
    """Algorithm captions in one line (box, font size). Two algorithm boxes side by side often have their captions in
    the same line, separated by a column gap."""
    spans = [span for span in line["spans"] if span["text"].strip()]
    starts: list[int] = []
    for index, span in enumerate(spans):
        if index and pymupdf.Rect(span["bbox"]).x0 - pymupdf.Rect(spans[index - 1]["bbox"]).x1 < span["size"] * 2:
            continue
        if _ALGORITHM_RE.match("".join(item["text"] for item in spans[index:])):
            starts.append(index)
    found: list[tuple[pymupdf.Rect, float]] = []
    for begin, end in zip(starts, [*starts[1:], len(spans)]):
        box = pymupdf.Rect()
        for span in spans[begin:end]:
            box |= pymupdf.Rect(span["bbox"])
        found.append((box, max(span["size"] for span in spans[begin:end])))
    return found


def _box_below(caption: pymupdf.Rect, size: float, horizontal: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """Caption on top: the first rule below the caption is the top of the box, the next one below is the bottom. The
    caption may have several lines."""
    rules = sorted(
        (rule for rule in horizontal if rule.y0 >= caption.y1 - _NEAR and _under(rule, caption, size)),
        key=lambda rule: rule.y0,
    )
    rules = _distinct_rules(rules)
    if len(rules) < 2 or rules[0].y0 - caption.y1 > size * 8:
        return None
    top, bottom = rules[0], rules[1]
    return pymupdf.Rect(top.x0, top.y0, top.x1, bottom.y0)


def _box_above(caption: pymupdf.Rect, size: float, horizontal: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """Caption at the bottom: the rule right above the caption is the bottom of the box, the next rule of the same
    width further up is the top."""
    rules = sorted(
        (rule for rule in horizontal if rule.y1 <= caption.y0 + _NEAR and _under(rule, caption, size)),
        key=lambda rule: -rule.y0,
    )
    rules = _distinct_rules(rules)
    if len(rules) < 2 or caption.y0 - rules[0].y1 > size * 2:
        return None
    bottom = rules[0]
    top = next(
        (rule for rule in rules[1:] if abs(rule.x0 - bottom.x0) < _NEAR and abs(rule.x1 - bottom.x1) < _NEAR), None
    )
    if top is None:
        return None
    return pymupdf.Rect(bottom.x0, top.y1, bottom.x1, bottom.y0)


_LINE_LABEL_RE = re.compile(r"^\s*\d+\s*[:.]")


# Line ends of an unfinished sentence: a lowercase letter, a comma, a hyphen of a broken word
_OPEN_END_RE = re.compile(r"[a-z,]$|[a-z]-$")


def _join_wrapped_rows(
    rows: list[pymupdf.Rect],
    baselines: list[float],
    spans: list[PdfDict],
    main: float,
) -> list[pymupdf.Rect]:
    """In an algorithm with line numbers at the line starts (1: 2: ...), a line without a number after the numbered
    lines is a wrapped part of the previous statement and joins the previous cell. In an algorithm without line
    numbers, when the previous line reaches the right edge of the box and its sentence is unfinished, the next line is
    its wrapped part.

    If the two halves were sent for translation separately, a half sentence would have too few body-text words and
    would be kept in English as a formula. Input: and Output: before the first line number are one cell each."""
    labeled: list[bool] = []
    full: list[bool] = []
    for baseline in baselines:
        line_spans = sorted(
            (span for span in spans if abs(span["origin"][1] - baseline) <= main * 0.4),
            key=lambda span: span["origin"][0],
        )
        text = "".join(span["text"] for span in line_spans).rstrip()
        labeled.append(bool(_LINE_LABEL_RE.match(text)))
        # A full line has even spacing; a right-aligned comment (// mean) also reaches the right edge, but with a wide
        # gap after the statement
        boxes = [span["bbox"] for span in line_spans if "bbox" in span and span["text"].strip()]
        right = max((box[2] for box in boxes), default=0.0)
        gaps = [b[0] - a[2] for a, b in zip(boxes, boxes[1:])]
        full.append(
            bool(rows)
            and right >= rows[0].x1 - main * 0.5
            and max(gaps, default=0.0) < main * 1.5
            and bool(_OPEN_END_RE.search(text))
        )
    if sum(labeled) < 2:
        merged: list[pymupdf.Rect] = []
        for index, row in enumerate(rows):
            if index and full[index - 1]:
                merged[-1] = merged[-1] | row
            else:
                merged.append(row)
        return merged
    first = labeled.index(True)
    out: list[pymupdf.Rect] = []
    for index, (row, has_label) in enumerate(zip(rows, labeled)):
        if index > first and not has_label:
            out[-1] = out[-1] | row
        else:
            out.append(row)
    return out


def _under(rule: pymupdf.Rect, caption: pymupdf.Rect, size: float) -> bool:
    """The rule starts at the left edge of the caption and covers the start of the caption. Rules of another
    algorithm box beside it do not count."""
    return rule.x0 <= caption.x0 + _NEAR and rule.x1 >= min(caption.x1, caption.x0 + size * 4)


def _distinct_rules(rules: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """One rule may be both a line segment and an edge of a box; of rules lying on top of each other only one is
    kept."""
    out: list[pymupdf.Rect] = []
    for rule in rules:
        if out and abs(rule.y0 - out[-1].y0) < _NEAR and abs(rule.x0 - out[-1].x0) < _NEAR:
            continue
        out.append(rule)
    return out


def _code_rows(
    lines: list[PdfDict],
    body: pymupdf.Rect,
    rules: Sequence[pymupdf.Rect] = (),
) -> tuple[pymupdf.Rect, ...]:
    """One cell per line in the box. Lines are split by baselines at body-text size, and superscripts and subscripts
    fall into the line they belong to. rules holds the horizontal rules in the box: fraction bars and radical bars are
    short, frame lines and separators run across the whole box."""
    spans = [
        span
        for line in lines
        for span in line["spans"]
        if span["text"].strip() and body.contains(pymupdf.Point(span["origin"]))
    ]
    if not spans:
        return ()
    main = max(span["size"] for span in spans)
    bars = [rule for rule in rules if rule.width < body.width * 0.5 and body.intersects(rule)]
    baselines: list[float] = []
    # Symbol-font glyphs such as radicals and large braces do not have their origin on the baseline, so they are not
    # used to split lines; numerators and denominators of display formulas are as large as body text but sit against
    # a fraction bar, so they are not used either
    steady = [
        span
        for span in spans
        if span["size"] >= main * 0.85 and not _SYMBOL_FONT_RE.search(span["font"]) and not _on_bar(span, bars, main)
    ]
    for y in sorted(span["origin"][1] for span in steady):
        if not baselines or y - baselines[-1] > main * 0.4:
            baselines.append(y)
    if len(baselines) < 2:
        return ()
    # The boundary sits close to the upper baseline: subscripts and descenders are only slightly below the baseline,
    # while superscripts (the st of 1st) are raised high and would fall into the upper line with a boundary midway
    cuts = [body.y0, *(a + (b - a) * _ROW_CUT for a, b in zip(baselines, baselines[1:])), body.y1]
    rows = [pymupdf.Rect(body.x0, top, body.x1, bottom) for top, bottom in zip(cuts, cuts[1:])]
    return tuple(_join_wrapped_rows(rows, baselines, spans, main))


def _on_bar(span: PdfDict, bars: Sequence[pymupdf.Rect], size: float) -> bool:
    """The span sits right above or below a short horizontal rule and within its horizontal extent: a numerator, a
    denominator, or text under a radical."""
    if not bars:
        return False
    box = pymupdf.Rect(span["bbox"])
    for bar in bars:
        if bar.x0 - 1 > box.x0 or box.x1 > bar.x1 + 1:
            continue
        if -1 <= bar.y0 - box.y1 <= size * 0.3 or -1 <= box.y0 - bar.y1 <= size * 0.3:
            return True
    return False
