"""Reflow index entries of a translated page into their columns."""

from collections.abc import Sequence
from dataclasses import replace

import pymupdf

from .pdf_blocks import (
    SPILL_TAIL,
    TextBlock,
    _span_is_math,
    _visual_lines,
)
from .pdf_flow import (
    separate_after_formula,
    space_around_latin,
)
from .pdf_layout import (
    PageGeometry,
)
from .pdf_place import (
    _formula_metrics,
)
from .pdf_typeset import (
    typeset_lines,
)
from .pdf_write_geometry import _source_marked


def _index_reflow(
    items: list[tuple[TextBlock, str]],
    blocks: list[TextBlock],
    geo: PageGeometry | None,
    cjk: bool,
    formulas_map: dict[int, list[dict]],
    fixed: set[int],
    bottom: float,
    ink: Sequence[pymupdf.Rect] = (),
) -> dict[int, TextBlock]:
    """In an index column, move entries whose translation takes more lines than the source down by the source line
    pitch, together with the entries below them.

    Entries are set one per line with no gaps, so without moving, the extra line has no room and the font size has
    to shrink. The blank space between letter groups absorbs it first, and if that is not enough the line spacing
    is tightened (see _restack). Entries that keep the original text move too. Paragraphs crossing columns or pages
    and source glyphs that belong to no block (ink, entries dropped during extraction) cannot move, so entries move
    only down to them. Returns copies of the moved entries with their boxes and line boxes already moved.
    """
    if geo is None or not any(b.index for b in blocks):
        return {}
    texts = {id(b): t for b, t in items if t != SPILL_TAIL}
    # Entries that keep the original text (BERT, CKY) move too, and once moved they are rewritten from the source
    texts.update(
        (id(b), _source_marked(b))
        for b in blocks
        if b.index and id(b) not in texts and not any(_span_is_math(s) for line in b.span_lines or [] for s in line)
    )
    moved: dict[int, TextBlock] = {}
    for column, right in _index_columns(blocks, geo):
        tops = sorted(row[0].y0 for b in column for row in _visual_lines(b))
        steps = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if not steps:
            continue
        pitch = sorted(steps)[len(steps) // 2]
        segment: list[tuple[TextBlock, int, int]] = []
        x0, x1 = min(b.rect.x0 for b in column), max(b.rect.x1 for b in column)
        stops = [rect for rect in ink if x0 <= (rect.x0 + rect.x1) / 2 <= x1]
        for b in sorted(
            [*column, *stops], key=lambda item: item.y0 if isinstance(item, pymupdf.Rect) else item.rect.y0
        ):
            if isinstance(b, pymupdf.Rect):
                moved.update(_restack(segment, pitch, b.y0))
                segment = []
                continue
            movable = b.index and id(b) in texts and id(b) not in fixed and pitch < b.size * 2
            if movable:
                rows = len(_visual_lines(b))
                em = b.size * 0.88 if cjk else b.size
                text = space_around_latin(separate_after_formula(texts[id(b)]))
                metrics = _formula_metrics(formulas_map.get(id(b), []), b.size, em, b.size * 0.88 if cjk else None)
                need = len(typeset_lines(text, em, max(right - b.rect.x0, 1.0), metrics))
                segment.append((b, rows, need))
                continue
            moved.update(_restack(segment, pitch, b.rect.y0))
            segment = []
        moved.update(_restack(segment, pitch, bottom))
    return moved


def _index_columns(blocks: list[TextBlock], geo: PageGeometry) -> list[tuple[list[TextBlock], float]]:
    """Split an index page into columns that do not overlap horizontally, sort each column top to bottom, and give
    the right edge each column can write to.

    Index lines are ragged, and layout analysis often finds only one column. Entries are written only within their
    column and do not overlap entries of other columns horizontally. The split looks only at index entries, not at
    a heading spanning several columns at the top of the page; that heading goes, like other blocks, into the
    columns it overlaps horizontally, where it blocks entries.
    """
    groups: list[list[TextBlock]] = []
    for b in sorted((b for b in blocks if b.index), key=lambda b: b.rect.x0):
        if groups and b.rect.x0 < max(other.rect.x1 for other in groups[-1]):
            groups[-1].append(b)
        else:
            groups.append([b])
    out = []
    for index, group in enumerate(groups):
        x0, x1 = min(b.rect.x0 for b in group), max(b.rect.x1 for b in group)
        group += [b for b in blocks if not b.index and b.rect.x0 < x1 and b.rect.x1 > x0]
        group.sort(key=lambda b: b.rect.y0)
        right = geo.column_of(group[0].rect).x1
        if index + 1 < len(groups):
            right = min(right, min(b.rect.x0 for b in groups[index + 1]) - 2)
        out.append((group, right))
    return out


def _restack(segment: list[tuple[TextBlock, int, int]], pitch: float, limit: float) -> dict[int, TextBlock]:
    """Stack a run of entries top to bottom: each starts at its original position or below the last line of the
    previous entry, and takes the number of lines it needs.

    An index column often runs down to the bottom of the body, with no blank lines to borrow. Beyond limit, the line
    spacing of the whole run is tightened together, by at most 10%, less than a point per line, which reads better
    than shrinking the font size of a single entry; if that is still not enough, nothing moves.
    """
    if not segment or not any(need > rows for _b, rows, need in segment):
        return {}
    placed: list[tuple[TextBlock, float, float, float]] = []
    cursor = None
    for b, rows, need in segment:
        top = b.line_rects[0].y0 if b.line_rects else b.rect.y0
        new_top = max(top, cursor if cursor is not None else top)
        grow = max(0, need - rows) * pitch
        cursor = new_top + max(need, rows) * pitch
        placed.append((b, top, new_top, grow))
    first = placed[0][1]
    last, top, new_top, grow = placed[-1]
    end = last.rect.y1 + new_top - top + grow
    scale = min(1.0, (limit - first) / (end - first)) if end > first else 1.0
    if scale < 0.9:
        return {}
    out: dict[int, TextBlock] = {}
    for b, top, new_top, grow in placed:
        new_top = first + (new_top - first) * scale
        if abs(new_top - top) < 0.05 and not grow:
            continue
        out[id(b)] = replace(
            b,
            rect=pymupdf.Rect(
                b.rect.x0, b.rect.y0 + new_top - top, b.rect.x1, b.rect.y1 + new_top - top + grow * scale
            ),
            line_rects=[
                pymupdf.Rect(
                    rect.x0, new_top + (rect.y0 - top) * scale, rect.x1, new_top + (rect.y0 - top) * scale + rect.height
                )
                for rect in b.line_rects
            ],
        )
    return out
