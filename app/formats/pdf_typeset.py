"""排版器自己断行。公式是不可拆的槽位，横坐标由前面的文字宽度决定。

纵向偏移不在这里算：调用方拿这一行文字的真实基线，加上公式记录的下沿。
没有公式图的段落用同一套断行，再写进一个锁住换行的盒子。
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
# 公式槽左右各空出的宽度（点），公式和两边的字隔开
FORMULA_PAD = 2.0


def line_lead(line: TypesetLine) -> float:
    """行首是公式槽时，槽左边空出的那一截。左对齐的行往左挪这么多，公式和上下行的字对齐"""
    return FORMULA_PAD if line.pieces and isinstance(line.pieces[0], FormulaPiece) else 0.0
# 公式可以伸进上下行距的部分（em）。原来压到 1.35em 的公式都在这个范围里
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
    """按槽位宽度断行，再把每一行拆成文字块和公式槽。first_widths 见 break_lines。"""
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


def line_offsets(lines: Sequence[TypesetLine], em: float, gap: float) -> tuple[list[float], float]:
    """每行顶部比按行距排的位置多往下挪多少，以及整段多出的高度。

    根号、求和带上下限的行内公式比一行字高。按行距排会压到上下行，缩到一行高又看不清，
    和原文一样把这一行的行距撑开。正常行距里本来容得下的公式不撑。
    第一行不往下挪：原文这一行上面本来留着公式的地方，往下挪反倒压到下面另写的一块。"""
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
