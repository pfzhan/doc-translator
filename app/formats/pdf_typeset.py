"""排版器自己断行。公式是不可拆的槽位，横坐标由前面的文字宽度决定。

纵向偏移不在这里算：调用方拿这一行文字的真实基线，加上公式记录的下沿。
没有公式的段落仍交给 HTML 盒子。
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class TypesetLine:
    pieces: tuple[Piece, ...]
    width: float


def typeset_lines(
    text: str,
    em: float,
    max_width: float,
    metrics: dict[int, FormulaMetric],
) -> list[TypesetLine]:
    """按槽位宽度断行，再把每一行拆成文字块和公式槽。"""
    widths = {index: metric.slot_width for index, metric in metrics.items()}
    return [_line_of(raw, em, widths, metrics) for raw in break_lines(text, em, max_width, widths)]


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
            pieces.append(FormulaPiece(
                index=index,
                x=cursor,
                slot_width=slot,
                body_width=metric.body_width,
                height=metric.height,
                below=metric.below,
            ))
        cursor += slot
    flush()
    width = pieces[-1].x + _piece_width(pieces[-1]) if pieces else 0.0
    return TypesetLine(tuple(pieces), width)


def _piece_width(piece: Piece) -> float:
    if isinstance(piece, FormulaPiece):
        return piece.slot_width
    return piece.width
