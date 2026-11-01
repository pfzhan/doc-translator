"""The typesetter breaks lines itself. A formula is an unbreakable slot whose x position is set by the width of
the text before it.

Vertical offsets are not computed here: the caller takes the actual baseline of the line's text and adds the
lower extent recorded for the formula (below).
Paragraphs without formula images use the same line breaking and are then written into a box with the line
breaks locked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .pdf_flow import break_lines, formula_index, measure_token, tokenize


@dataclass(frozen=True)
class FormulaMetric:
    slot_width: float
    body_width: float
    height: float
    below: float


@dataclass(frozen=True)
class TextPiece:
    text: str
    x: float
    width: float


@dataclass(frozen=True)
class FormulaPiece:
    index: int
    x: float
    slot_width: float
    body_width: float
    height: float
    below: float


Piece = TextPiece | FormulaPiece
# Width (points) left free on each side of a formula slot, separating the formula from the text around it
FORMULA_PAD = 2.0


def line_lead(line: TypesetLine) -> float:
    """The padding left of the slot when a line starts with a formula slot. Left-aligned lines move left by this
    much so that the formula lines up with the text on the lines above and below"""
    return FORMULA_PAD if line.pieces and isinstance(line.pieces[0], FormulaPiece) else 0.0


# How far a formula may extend into the line spacing above and below (em). Formulas that used to be squeezed
# to 1.35em all fit in this range
_FORMULA_SPILL = 0.2


@dataclass(frozen=True)
class TypesetLine:
    pieces: tuple[Piece, ...]
    width: float


def typeset_lines(
    text: str,
    em: float,
    max_width: float,
    metrics: dict[int, FormulaMetric],
    first_widths: Sequence[float] = (),
) -> list[TypesetLine]:
    """Break lines using the slot widths, then split each line into text pieces and formula slots. For
    first_widths see break_lines."""
    widths = {index: metric.slot_width for index, metric in metrics.items()}
    raw_lines = break_lines(text, em, max_width, widths, first_widths)
    return [_line_of(raw, em, widths, metrics) for raw in raw_lines]


def _line_of(
    text: str,
    em: float,
    widths: dict[int, float],
    metrics: dict[int, FormulaMetric],
) -> TypesetLine:
    pieces: list[Piece] = []
    buf: list[str] = []
    buf_width = 0.0
    cursor = 0.0

    def flush() -> None:
        nonlocal buf_width, cursor
        if not buf:
            return
        pieces.append(TextPiece("".join(buf), cursor, buf_width))
        cursor += buf_width
        buf.clear()
        buf_width = 0.0

    for token in tokenize(text):
        index = formula_index(token)
        if index is None:
            buf.append(token)
            buf_width += measure_token(token, em, widths)
            continue
        flush()
        metric = metrics.get(index)
        slot = metric.slot_width if metric is not None else measure_token(token, em, widths)
        if metric is not None:
            pieces.append(
                FormulaPiece(
                    index=index,
                    x=cursor,
                    slot_width=slot,
                    body_width=metric.body_width,
                    height=metric.height,
                    below=metric.below,
                )
            )
        cursor += slot
    flush()
    width = pieces[-1].x + _piece_width(pieces[-1]) if pieces else 0.0
    return TypesetLine(tuple(pieces), width)


def line_offsets(lines: Sequence[TypesetLine], em: float, gap: float) -> tuple[list[float], float]:
    """How far the top of each line moves down from its position under normal line spacing, and the extra
    height of the whole paragraph.

    Inline formulas with radicals or sums with limits are taller than a line of text. At normal line spacing
    they would overlap the lines above and below, and shrunk to one line height they become unreadable, so the
    spacing of that line is widened, as in the original. Formulas that fit in the normal line spacing do not
    widen it.
    The first line does not move down: the original already leaves room for the formula above this line, and
    moving down would overlap the block written separately below."""
    half = (gap - 1) / 2
    room_above = em * (0.88 + half + _FORMULA_SPILL)
    room_below = em * (0.12 + half + _FORMULA_SPILL)
    offsets: list[float] = []
    extra = 0.0
    for index, line in enumerate(lines):
        above = below = 0.0
        for piece in line.pieces:
            if isinstance(piece, FormulaPiece):
                above = max(above, piece.height - piece.below - room_above)
                below = max(below, piece.below - room_below)
        if index:
            extra += above
        offsets.append(extra)
        extra += below
    return offsets, extra


def _piece_width(piece: Piece) -> float:
    if isinstance(piece, FormulaPiece):
        return piece.slot_width
    return piece.width
