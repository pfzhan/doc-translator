"""Join the end of one column to the top of the next column or page."""

import re
from collections import Counter

from .pdf_blocks import (
    _AFFIL_MARK_RE,
    TextBlock,
    _ends_sentence,
    _visual_lines,
)
from .pdf_continuations import _is_float_text
from .pdf_math import _prose_units
from .pdf_page_continue import _at_text_edge, _can_continue, _is_page_footnote, _next_text_page, _resumes

# A section number may hang at most this many em to the left of the column edge
_HANGING_HEAD_EM = 3


_CODE_LINE_RE = re.compile(r"^(?:<|\d+\s+0x[0-9a-fA-F])")


# A block with this many words is body text regardless of its share of letters
_PROSE_WORDS = 8


_CLUSTER_GAP = 28.0


def _is_code_line(text: str) -> bool:
    """XML, stack traces and labels inside figures. At the top of a column they must not take the place of a
    continuation.

    Short continuations (well., tomer) are still body text; a label has no spaces and contains digits or
    symbols. Result paragraphs full of numbers (and lower mean WER (0.009 vs. 0.013); paired ...) have a low
    share of letters but enough words to be body text.
    """
    stripped = text.strip()
    if not stripped or _CODE_LINE_RE.match(stripped):
        return True
    if stripped[:1] in "<{" or "::" in stripped or '="' in stripped or "/>" in stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < len(stripped) * 0.45 and _prose_units(stripped) < _PROSE_WORDS:
        return True
    return " " not in stripped and re.search(r"[\d%$<>!#_]", stripped) is not None


def _continuation_prose(
    block: TextBlock,
    median: float,
    skip_ids: set[int],
    halt_ids: set[int] | None,
    ignored: set[int],
) -> bool:
    if id(block) in skip_ids or id(block) in ignored:
        return False
    # Contents and index entries do not join their neighbors but still occupy the top and bottom of a column:
    # if they were skipped, a headword without a page number at the bottom would join an entry in the middle
    # of the next column. Cells of word lists are skipped like figures
    listed = (block.leader is not None or block.entry) and not block.grid
    if halt_ids is not None and id(block) in halt_ids and not listed:
        return False
    if block.size < median * 0.8 or _is_page_footnote(block, median):
        return False
    if _AFFIL_MARK_RE.match(block.text.lstrip()):
        return False
    return not _is_float_text(block.text) and not _is_code_line(block.text)


def _column_anchors(blocks: list[TextBlock]) -> list[float]:
    """Left edges of the body text. A column needs at least two blocks at the same left edge, so the right half
    of a single line does not start a column. When the page already has such a column, a single block of
    three or more lines also counts (another column holding one paragraph, below an indented abstract).

    The next column starts beyond the right edge of the previous one. Indented examples and quotations lie
    inside the previous column and do not start a column; otherwise their last block would be taken as the
    page end and the start of the next page's paragraph joined to it.
    """
    if not blocks:
        return []
    groups: list[list[TextBlock]] = []
    for block in sorted(blocks, key=lambda block: block.rect.x0):
        if groups and block.rect.x0 - groups[-1][0].rect.x0 < 8:
            groups[-1].append(block)
        else:
            groups.append([block])
    merged: list[float] = []
    right = None
    paired = any(len(group) >= 2 for group in groups)
    for group in groups:
        if len(group) < 2 and not (paired and len(_visual_lines(group[0])) >= 3):
            continue
        anchor = sum(block.rect.x0 for block in group) / len(group)
        if merged and (anchor - merged[-1] < _CLUSTER_GAP or (right is not None and anchor < right - 4)):
            continue
        merged.append(anchor)
        edges = sorted(block.rect.x1 for block in group)
        right = edges[len(edges) // 2]
    return merged


def _anchor_index(block: TextBlock, anchors: list[float]) -> int | None:
    if not anchors:
        return None
    best = min(range(len(anchors)), key=lambda index: abs(block.rect.x0 - anchors[index]))
    if abs(block.rect.x0 - anchors[best]) <= max(block.size * 4, 36):
        return best
    return None


def _assigned_columns(
    blocks: list[TextBlock],
    anchors: list[float],
) -> tuple[list[list[TextBlock]], list[TextBlock]]:
    columns: list[list[TextBlock]] = [[] for _ in anchors]
    orphans: list[TextBlock] = []
    for block in blocks:
        index = _anchor_index(block, anchors)
        if index is None:
            orphans.append(block)
        else:
            columns[index].append(block)
    return columns, orphans


def _column_body(column: list[TextBlock]) -> list[TextBlock]:
    """Smaller footnotes at the end of a column do not take the place of a continuation. If the whole column
    is in that size, everything stays.

    Table cells are left out when finding the body size: a table at the top of a column has one block per
    cell and would outnumber the body text.
    """
    if not column:
        return []
    sizes = sorted(block.size for block in column if block.clip is None) or sorted(block.size for block in column)
    body_size = sizes[len(sizes) // 2]
    body = [block for block in column if block.size + 0.6 >= body_size]
    return body or column


def _column_top(blocks: list[TextBlock]) -> TextBlock:
    return min(blocks, key=lambda block: (block.rect.y0, block.rect.x0))


def _column_bottom(blocks: list[TextBlock]) -> TextBlock:
    return max(blocks, key=lambda block: (block.rect.y1, block.rect.x0))


def _fragment_host(tail: TextBlock, orphans: list[TextBlock]) -> TextBlock:
    """When the body text at the end of a column already ends a sentence, an unfinished half sentence to its
    right on the same line is where the continuation starts.

    Line boxes are used rather than the block's union box, which would cover the half sentence on the right
    so that it no longer counts as attached to the line end.
    """
    if not _ends_sentence(tail.text):
        return tail
    lines = tail.line_rects or [tail.rect]
    for orphan in orphans:
        if orphan.page != tail.page or len(orphan.line_rects) > 1:
            continue
        if _ends_sentence(orphan.text):
            continue
        height = orphan.rect.height or orphan.size
        for line in lines:
            overlap = min(orphan.rect.y1, line.y1) - max(orphan.rect.y0, line.y0)
            if overlap <= min(height, line.height) * 0.5:
                continue
            gap = orphan.rect.x0 - line.x1
            if -2 <= gap < 16:
                return orphan
    return tail


def _column_wrap(host: TextBlock, tail: TextBlock) -> bool:
    """A move to another column needs left edges a column apart. A capital at the top of the page in the same
    column still starts the next paragraph."""
    return abs(host.rect.x0 - tail.rect.x0) > max(host.size * 4, 36.0)


def _try_column_pair(
    pairs: dict[int, TextBlock],
    tails: set[int],
    host: TextBlock | None,
    tail: TextBlock | None,
    median: float,
    halt_ids: set[int] | None,
    skip_ids: set[int],
    allow_capital: bool = True,
    flush: bool = False,
    at_edge: bool = True,
) -> None:
    if host is None or tail is None or host is tail:
        return
    if id(host) in pairs or id(host) in tails or id(tail) in pairs or id(tail) in tails:
        return
    if id(host) in skip_ids or id(tail) in skip_ids:
        return
    if halt_ids is not None and id(host) in halt_ids:
        return
    if abs(host.size - tail.size) > max(host.size, 1.0) * 0.25:
        return
    if not _can_continue(host, tail, median, halt_ids, allow_capital=allow_capital, flush=flush, at_edge=at_edge):
        return
    pairs[id(host)] = tail
    tails.add(id(tail))


def _float_contents(column: list[TextBlock], page_blocks: list[TextBlock]) -> set[int]:
    """Text of a table (or figure) at the top of a column: from the column top down to the caption below it
    (Table 7: ...), no line reaches the right edge of the column. Table headers and row names are short lines;
    paragraph lines fill the column width, so a paragraph at the top does not count. When a table caption is
    above the table and the column starts right below it, the table runs from the top down to the first block
    that fills the column width. A figure caption is below the figure, and short lines below it are body text
    (subject to after a formula) and do not count. The column's right edge is where justified body lines most
    often end; figures wider than the body text do not count."""
    if not column:
        return set()
    x0 = min(block.rect.x0 for block in column)
    x1 = max(block.rect.x1 for block in column)
    ends = Counter(round(line.x1) for block in column for line in (block.line_rects or [block.rect]))
    right, count = ends.most_common(1)[0]
    if count < 3:
        return set()
    captions = [
        block for block in page_blocks if _is_float_text(block.text) and block.rect.x0 < x1 and block.rect.x1 > x0
    ]
    if not captions:
        return set()
    caption = min(captions, key=lambda block: block.rect.y0)

    def short(block: TextBlock) -> bool:
        return all(line.x1 < right - block.size * 2 for line in (block.line_rects or [block.rect]))

    top = min(column, key=lambda block: block.rect.y0)
    over = [block for block in captions if block.rect.y1 <= top.rect.y0 + 1]
    near = max(over, key=lambda block: block.rect.y1) if over else None
    if near is not None and re.match(r"(?i)tab", near.text.lstrip()) and top.rect.y0 - near.rect.y1 <= 2 * top.size:
        contents: set[int] = set()
        for block in sorted(column, key=lambda block: block.rect.y0):
            if not short(block):
                return contents
            contents.add(id(block))
        return set()
    above = [block for block in column if block.rect.y1 <= caption.rect.y0 + 1]
    if not above or not all(short(block) for block in above):
        return set()
    return {id(block) for block in above}


def _column_continuations(
    by_page: dict[int, list[TextBlock]],
    median: float,
    skip_ids: set[int],
    halt_ids: set[int] | None,
    ignored: set[int],
    aside_ids: set[int] = frozenset(),
    indents: bool = False,
) -> dict[int, TextBlock]:
    """Join the end of the left column to the top of the right column, and the end of the right column to the
    left column of the next page.

    Figures, lists and captions at the top of a column are passed over and the continuation is looked for
    below them, and so are tables there (see _float_contents). Columns are found from left edges, so this
    works when the geometric columns are wrong and does not depend on reads_in_columns. Continuation in the
    same column across pages does not allow capitals: when the other column has no left edge, the left
    column at the page end must not swallow a new paragraph on the next page. Flush starts in documents that
    indent first lines are the exception (indents, see _can_continue).
    """
    pairs: dict[int, TextBlock] = {}
    tails: set[int] = set()
    pages = sorted(by_page)
    prose: dict[int, list[TextBlock]] = {
        pno: [block for block in blocks if _continuation_prose(block, median, skip_ids, halt_ids, ignored)]
        for pno, blocks in by_page.items()
    }
    columns_of: dict[int, list[list[TextBlock]]] = {}
    orphans_of: dict[int, list[TextBlock]] = {}
    for pno, blocks in prose.items():
        columns, orphans = _assigned_columns(blocks, _column_anchors(blocks))
        columns_of[pno] = [column for column in columns if column]
        orphans_of[pno] = orphans

    def head_of(pno: int, column: list[TextBlock]) -> TextBlock | None:
        contents = _float_contents(column, by_page.get(pno) or [])
        body = _column_body([block for block in column if id(block) not in contents])
        # A single line centered below a line that belongs to no column is an author or affiliation set in
        # columns (Shixiang Zhu | Carnegie Mellon University), not a column top. Below table cells or a heading
        # across both columns is the column top
        orphans = orphans_of.get(pno) or []
        body = [
            block
            for block in body
            if not (
                len(_visual_lines(block)) == 1
                and any(
                    0 <= block.rect.y0 - other.rect.y1 <= 1.5 * block.size
                    and abs((block.rect.x0 + block.rect.x1) - (other.rect.x0 + other.rect.x1)) / 2 <= block.size
                    for other in orphans
                )
            )
        ]
        if not body:
            return None
        head = _column_top(body)
        return None if heading_above(pno, column, head) else head

    def heading_above(pno: int, column: list[TextBlock], head: TextBlock) -> bool:
        """A subheading right above the column top within this column (A. Materials): the column starts a new
        section and holds no continuation. A section number hanging left of the column edge (1.4 Related work)
        counts too. A heading across several columns does not, nor does large text in a figure taken for a
        heading (with other figure text in between)"""
        if halt_ids is None:
            return False
        above = [
            block
            for block in by_page.get(pno) or []
            if id(block) not in ignored
            and block is not head
            and block.rect.y1 <= head.rect.y0 + 1
            and block.rect.x0 < head.rect.x1
            and block.rect.x1 > head.rect.x0
        ]
        if not above:
            return False
        near = max(above, key=lambda block: block.rect.y1)
        x0 = min(block.rect.x0 for block in column)
        x1 = max(block.rect.x1 for block in column)
        return (
            id(near) in halt_ids
            and id(near) not in aside_ids
            and id(near) not in skip_ids
            and not near.side_note
            and near.leader is None
            and not near.entry
            and near.rect.x0 >= x0 - _HANGING_HEAD_EM * near.size
            and near.rect.x1 <= x1 + near.size
        )

    def host_of(pno: int, column: list[TextBlock]) -> TextBlock | None:
        body = _column_body(column)
        if not body:
            return None
        return _fragment_host(_column_bottom(body), orphans_of.get(pno) or [])

    for pno in pages:
        columns = columns_of.get(pno) or []
        for index in range(len(columns) - 1):
            host = host_of(pno, columns[index])
            # A continuation is not larger than the first half. Larger text at the top of the right column is a
            # title or byline above both columns, and the continuation is below it
            head = head_of(
                pno, [block for block in columns[index + 1] if host is None or block.size <= host.size * 1.1]
            )
            # Only text that runs to the bottom of the left column and resumes at the top of the right one moves
            # across columns. Columns that start side by side are not joined.
            if host is None or head is None or host.rect.y0 <= head.rect.y0 + host.size:
                continue
            if host.rect.x1 > head.rect.x0 + 4:
                continue
            _try_column_pair(
                pairs,
                tails,
                host,
                head,
                median,
                halt_ids,
                skip_ids,
                flush=indents and _resumes(head, by_page.get(pno) or [], skip_ids, ignored | aside_ids),
                at_edge=_at_text_edge(head, by_page.get(pno) or []),
            )
        later = _next_text_page(by_page, pno, median, skip_ids, ignored)
        nxt = columns_of.get(later) or []
        if not columns or not nxt:
            continue
        host = host_of(pno, columns[-1])
        # When the right column was not found (none of its blocks got a column), the last column is not the
        # rightmost: a block with body text to its right is not the page end. Nor is one with body text below it
        # that spans it from left to right in a size no smaller: a table in a figure whose cells share a left edge
        # was taken for a column, with a full-width paragraph below. Footnotes are smaller and labels beside
        # display formulas narrower, so neither blocks it
        if host is not None and any(
            block.rect.x0 >= host.rect.x1 - host.size
            and block.rect.y1 > host.rect.y0
            or block is not host
            and block.rect.y0 >= host.rect.y1 - 1
            and block.size >= host.size - 0.5
            and block.rect.x0 <= host.rect.x0 + host.size
            and block.rect.x1 >= host.rect.x1 - host.size
            for block in prose[pno]
        ):
            continue
        head = head_of(later, nxt[0])
        _try_column_pair(
            pairs,
            tails,
            host,
            head,
            median,
            halt_ids,
            skip_ids,
            allow_capital=bool(host and head and _column_wrap(host, head)),
            flush=bool(indents and head and _resumes(head, by_page.get(later) or [], skip_ids, ignored | aside_ids)),
            at_edge=head is None or _at_text_edge(head, by_page.get(later) or []),
        )
    return pairs
