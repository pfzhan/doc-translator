"""Text frame: the left, right, top and bottom edges of body lines across the whole document, and the margin
notes outside the frame.

Looking at one page alone, the left edge of a column is taken from the leftmost block, which is often a running
header, a protruding section number or a margin note, while the body text starts further right. The frame is
voted on by the long body lines of the whole document: most long lines start at the same place and end at the
same place. Books have different inner and outer margins on odd and even pages, which are counted separately.
When a page has enough long lines of its own that disagree with the pages on the same side (a bibliography
switching to two columns), the page's own edges are used.

A margin note is a short word outside the frame, one size smaller than the body text, on the baseline of a body
line. PyMuPDF often puts it in the same block as that body line, and unless split off it is merged into the
sentence. Margin notes align along one edge (those on the left are right-aligned), and an edge needs several of
them; scattered small text does not count.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

_PROSE_CHARS = 40  # only lines at least this long vote
_MIN_LINES = 12  # minimum long lines in a group
_EDGE_TOL = 2.0  # tolerance for the same edge
_EDGE_SHARE = 0.5  # an edge needs at least this share of the top edge's votes: with two columns both left edges
# qualify
_PAGE_SHIFT = 12.0  # when the page's edge is this far from that of pages on the same side, the page's own is used
_TOP_TOL = 3.0
_NOTE_GAP = 2.0  # minimum distance of a margin note from the frame
_NOTE_SMALL = 0.9  # maximum size of a margin note relative to the body text
_LANE_ALIGN = 1.5
_LANE_MIN = 3  # minimum margin notes along one edge
_NOTE_SYMBOL_CHARS = 3  # maximum characters of a margin note at body size (ŷ is extracted as ˆ and y)
_PITCH_TOL = 0.1  # a line pitch differing by this share from the document's separates two margin notes
_PITCH_MIN = 10  # minimum line pitches of margin notes in one size across the document for comparison


@dataclass(frozen=True)
class FrameLine:
    rect: pymupdf.Rect
    text: str
    size: float


@dataclass(frozen=True)
class TextFrame:
    x0: float
    x1: float
    # Top of the first body line and bottom of the last. None when pages on the same side do not agree
    y0: float | None
    y1: float | None
    size: float


def text_frames(pages: Sequence[Sequence[FrameLine]]) -> list[TextFrame | None]:
    """The text frame of each page. None for pages with too few long lines to vote.

    A long line with the same text at the same height on several pages is a running header or footer
    (Published as a conference paper at ICLR 2015); even at body size it does not vote, or the top of the frame
    would land on the header."""
    running = _running_lines(pages)
    prose = [
        [line for line in page if len(line.text.strip()) >= _PROSE_CHARS and _line_key(line) not in running]
        for page in pages
    ]
    whole = _frame_of(prose)
    shared: dict[int, TextFrame | None] = {}
    for parity in (0, 1):
        shared[parity] = _frame_of([lines for index, lines in enumerate(prose) if index % 2 == parity]) or whole
    frames: list[TextFrame | None] = []
    for index, lines in enumerate(prose):
        frame = shared[index % 2]
        own = _frame_of([lines], with_rows=False) if len(lines) >= _MIN_LINES else None
        if own is not None and frame is not None:
            # The edge of pages on the same side still has a number of long lines starting and ending on this page;
            # the lines reaching further out are a few full-width captions, wide tables and the like
            x0 = frame.x0 if _holds([line.rect.x0 for line in lines], frame.x0) else own.x0
            x1 = frame.x1 if _holds([line.rect.x1 for line in lines], frame.x1) else own.x1
            own = TextFrame(x0, x1, own.y0, own.y1, own.size)
        if own is not None and (
            frame is None or abs(own.x0 - frame.x0) > _PAGE_SHIFT or abs(own.x1 - frame.x1) > _PAGE_SHIFT
        ):
            frame = TextFrame(
                own.x0,
                own.x1,
                frame.y0 if frame is not None else None,
                frame.y1 if frame is not None else None,
                own.size,
            )
        frames.append(frame)
    return frames


def _holds(values: Sequence[float], edge: float) -> bool:
    """The edge has enough votes among these lines: at least _EDGE_SHARE of the top edge's."""
    counter = Counter(round(value) for value in values)
    if not counter:
        return False
    best = max(_near(counter, value) for value in counter)
    return _near(counter, round(edge)) >= best * _EDGE_SHARE


def _line_key(line: FrameLine) -> tuple[int, str]:
    return round(line.rect.y0), re.sub(r"\d+", "#", " ".join(line.text.split()))


def _running_lines(pages: Sequence[Sequence[FrameLine]]) -> set[tuple[int, str]]:
    """The same line of text at the same height on two or more pages (digits ignored, since running headers often
    carry page numbers)."""
    seen: Counter[tuple[int, str]] = Counter()
    for page in pages:
        seen.update({_line_key(line) for line in page if line.text.strip()})
    return {key for key, count in seen.items() if count >= 2}


def margin_notes(pages: Sequence[Sequence[FrameLine]], frames: Sequence[TextFrame | None]) -> list[set[int]]:
    """Indexes of the lines on each page that are margin notes."""
    candidates: list[tuple[int, int, str]] = []
    # A few characters at body size (ŷ in a margin note) count only when aligned with the margin note edge
    symbols: list[tuple[int, int, str]] = []
    for pno, (lines, frame) in enumerate(zip(pages, frames)):
        if frame is None:
            continue
        for index, line in enumerate(lines):
            # Letters are not required: a margin note may be a symbol such as *?. Page numbers and protruding
            # section numbers are no smaller than the body text and are excluded
            if not line.text.strip():
                continue
            side = (
                "left"
                if line.rect.x1 <= frame.x0 - _NOTE_GAP
                else "right"
                if line.rect.x0 >= frame.x1 + _NOTE_GAP
                else ""
            )
            if not side:
                continue
            if line.size <= frame.size * _NOTE_SMALL:
                candidates.append((pno, index, side))
            elif len("".join(line.text.split())) <= _NOTE_SYMBOL_CHARS and re.search(r"[^\W\d_]", line.text):
                # Page numbers and item numbers (1.) are digits and do not count
                symbols.append((pno, index, side))
    edges: dict[tuple[str, int], Counter[int]] = {}
    for pno, index, side in candidates:
        rect = pages[pno][index].rect
        counter = edges.setdefault((side, pno % 2), Counter())
        counter[round(rect.x0)] += 1
        counter[round(rect.x1)] += 1
    found: list[set[int]] = [set() for _ in pages]
    for pno, index, side in candidates:
        rect = pages[pno][index].rect
        counter = edges[(side, pno % 2)]
        if any(_support(counter, edge) >= _LANE_MIN for edge in (rect.x0, rect.x1)):
            found[pno].add(index)
    # The edge toward the frame: notes on the left are right-aligned and notes on the right left-aligned. The
    # outer edges are ragged and not compared
    inner_edges: dict[tuple[str, int], Counter[int]] = {}
    for pno, index, side in candidates:
        rect = pages[pno][index].rect
        inner_edges.setdefault((side, pno % 2), Counter())[round(rect.x1 if side == "left" else rect.x0)] += 1
    for pno, index, side in symbols:
        rect = pages[pno][index].rect
        inner = rect.x1 if side == "left" else rect.x0
        if _support(inner_edges.get((side, pno % 2), Counter()), inner) >= _LANE_MIN:
            found[pno].add(index)
    return found


def note_pitch(pages: Sequence[Sequence[FrameLine]], notes: Sequence[set[int]]) -> dict[float, float]:
    """Line pitches within a margin note across the document, recorded per font size. See note_groups."""
    counters: dict[float, Counter[float]] = {}
    for lines, found in zip(pages, notes):
        picked = [lines[index] for index in sorted(found)]
        for group in _stacked_notes(picked):
            for upper, lower in zip(group, group[1:]):
                above, below = picked[upper], picked[lower]
                pitch = below.rect.y0 - above.rect.y0
                if abs(above.size - below.size) < 0.1 and pitch > above.size * 0.5:
                    counters.setdefault(round(above.size, 1), Counter())[round(pitch * 2) / 2] += 1
    pitches: dict[float, float] = {}
    for size, counter in counters.items():
        pitch, count = counter.most_common(1)[0]
        if count >= _PITCH_MIN:
            pitches[size] = pitch
    return pitches


def note_groups(lines: Sequence[FrameLine], pitches: dict[float, float] | None = None) -> list[list[int]]:
    """Group the margin note lines of a page into notes: two lines right above each other are one note
    (regular / expressions), lines a body line pitch apart are two (Herdan's Law, Heaps' Law). Returns the line
    indexes of each note. When one body line introduces two terms, their notes sit right above each other, even
    closer than the line pitch (AED directly above listen attend / and spell). Within one note the pitch is
    constant, and where it departs from the pitch of margin notes across the document (pitches, see
    note_pitch) two notes meet. Pieces of one split line have a pitch near zero and do not count."""
    groups = _stacked_notes(lines)
    if not pitches:
        return groups
    parts: list[list[int]] = []
    for group in groups:
        parts.append([group[0]])
        for upper, lower in zip(group, group[1:]):
            above, below = lines[upper], lines[lower]
            pitch = below.rect.y0 - above.rect.y0
            regular = pitches.get(round(above.size, 1))
            if (
                regular is not None
                and abs(above.size - below.size) < 0.1
                and pitch > above.size * 0.5
                and abs(pitch - regular) > regular * _PITCH_TOL
            ):
                parts.append([lower])
            else:
                parts[-1].append(lower)
    return parts


def _stacked_notes(lines: Sequence[FrameLine]) -> list[list[int]]:
    """Group margin note lines that are right above each other"""
    order = sorted(range(len(lines)), key=lambda index: (lines[index].rect.y0, lines[index].rect.x0))
    groups: list[list[int]] = []
    for index in order:
        line = lines[index]
        if groups:
            last = lines[groups[-1][-1]]
            pitch = line.rect.y0 - last.rect.y0
            same_side = (line.rect.x1 < last.rect.x1 + line.size * 8) and (line.rect.x0 > last.rect.x0 - line.size * 8)
            if same_side and 0 <= pitch <= max(line.size, last.size) * 1.3:
                groups[-1].append(index)
                continue
        groups.append([index])
    return groups


def _support(counter: Counter[int], edge: float) -> int:
    return sum(count for value, count in counter.items() if abs(value - edge) <= _LANE_ALIGN)


def _frame_of(pages: Sequence[Sequence[FrameLine]], with_rows: bool = True) -> TextFrame | None:
    lines = [line for page in pages for line in page]
    if len(lines) < _MIN_LINES:
        return None
    x0 = _edge([line.rect.x0 for line in lines], low=True)
    x1 = _edge([line.rect.x1 for line in lines], low=False)
    sizes = Counter(round(line.size * 2) / 2 for line in lines)
    size = sizes.most_common(1)[0][0]
    y0 = y1 = None
    if with_rows:
        # Only lines at body size inside the frame: long running headers are long enough to vote, but are one size
        # smaller or outside the frame
        body = [
            [
                line
                for line in page
                if line.rect.x0 >= x0 - _EDGE_TOL
                and line.rect.x1 <= x1 + _EDGE_TOL
                and abs(line.size - size) <= size * 0.1
            ]
            for page in pages
        ]
        y0 = _mode([min(line.rect.y0 for line in page) for page in body if page], _TOP_TOL)
        y1 = _mode([max(line.rect.y1 for line in page) for page in body if page], _TOP_TOL)
    return TextFrame(x0, x1, y0, y1, size)


def _edge(values: Sequence[float], low: bool) -> float:
    """The outermost of the edges with enough votes."""
    counter = Counter(round(value) for value in values)
    support = {value: _near(counter, value) for value in counter}
    best = max(support.values())
    eligible = [value for value, votes in support.items() if votes >= best * _EDGE_SHARE]
    pick = min(eligible) if low else max(eligible)
    # Take the most frequent value near this edge rather than the border of the cluster
    return float(
        max(
            (value for value in counter if abs(value - pick) <= _EDGE_TOL),
            key=lambda value: (counter[value], -abs(value - pick)),
        )
    )


def _near(counter: Counter[int], value: int) -> int:
    return sum(count for other, count in counter.items() if abs(other - value) <= _EDGE_TOL)


def _mode(values: Sequence[float], tol: float) -> float | None:
    """The most frequent of the positions that agree on at least two pages."""
    if not values:
        return None
    counter = Counter(round(value) for value in values)
    support = {value: sum(count for other, count in counter.items() if abs(other - value) <= tol) for value in counter}
    value, votes = max(support.items(), key=lambda item: (item[1], -item[0]))
    if votes < 2:
        return None
    return float(value)
