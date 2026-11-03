"""Find where a paragraph continues: on the next page, in the next column or in the next
layout-model region, and pair the blocks into one translation unit."""

import re
from collections.abc import Callable, Sequence

import pymupdf

from .pdf_blocks import (
    _END_MARK_RE,
    _LABEL_START_RE,
    TextBlock,
    _indents_paragraphs,
    _list_item,
    _same_page_line_pair,
    _span_is_math,
    _styled_edge,
    _visual_lines,
)
from .pdf_columns import _column_continuations
from .pdf_continuations import _is_float_text
from .pdf_layout import (
    PageGeometry,
)
from .pdf_page_continue import _at_text_edge, _can_continue, _is_page_footnote, _next_text_page, _resumes
from .pdf_paragraphs import _earlier_on_page, _is_line_tail, _line_tail_blocked, _line_tail_match


def _different_columns(
    geometries: list[PageGeometry] | None,
    left: TextBlock,
    right: TextBlock,
) -> bool:
    """If either page has several columns and the two blocks have different column indexes, the second block
    is not the continuation of the first in the same column.

    The right column's end continuing in the left column of the next page is a separate rule handled
    elsewhere. On single-column pages the index is always 0.
    """
    if not geometries or left.page >= len(geometries) or right.page >= len(geometries):
        return False
    a, b = geometries[left.page], geometries[right.page]
    if len(a.columns) < 2 and len(b.columns) < 2:
        return False
    return a.index_of(left.rect) != b.index_of(right.rect)


def _page_edge_blocks(
    blocks: list[TextBlock],
    median: float,
    ignored: set[int],
) -> list[TextBlock]:
    """Only body text counts at the top and bottom of a page. Smaller numbered footnotes are translated where
    they are and do not take the page-end position."""
    return [
        block
        for block in blocks
        if block.size >= median * 0.8 and id(block) not in ignored and not _is_page_footnote(block, median)
    ]


def _column_index(geo: PageGeometry, block: TextBlock) -> int:
    return geo.index_of(block.rect)


def _in_column(geo: PageGeometry, block: TextBlock) -> bool:
    return len(geo.columns) < 2 or not geo.spans_columns(block.rect)


def _column_page_end(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    """The page end used for continuation in the same column. A block spanning columns on a multi-column
    page does not take this position, or it would take it from the last line of the right column."""
    return _column_edge(body, geometries, last=True)


def _column_page_start(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    return _column_edge(body, geometries, last=False)


def _column_edge(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
    last: bool,
) -> TextBlock | None:
    if not body:
        return None
    if not geometries:
        return body[-1] if last else body[0]
    ordered = reversed(body) if last else body
    for block in ordered:
        if block.page >= len(geometries) or _in_column(geometries[block.page], block):
            return block
    return None


def _is_last_in_right_column(
    block: TextBlock,
    page_blocks: list[TextBlock],
    geo: PageGeometry,
) -> bool:
    if len(geo.columns) < 2 or not _in_column(geo, block):
        return False
    if _column_index(geo, block) != len(geo.columns) - 1:
        return False
    return not any(
        other is not block
        and _in_column(geo, other)
        and _column_index(geo, other) == len(geo.columns) - 1
        and other.rect.y0 > block.rect.y0 + 1
        for other in page_blocks
    )


def _left_column_head(page_blocks: list[TextBlock], geo: PageGeometry) -> TextBlock | None:
    if len(geo.columns) < 2:
        return None
    heads = [
        block
        for block in page_blocks
        if _in_column(geo, block)
        and _column_index(geo, block) == 0
        and not any(
            other is not block
            and _in_column(geo, other)
            and _column_index(geo, other) == 0
            and other.rect.y0 < block.rect.y0 - 1
            for other in page_blocks
        )
    ]
    if not heads:
        return None
    return min(heads, key=lambda block: (block.rect.y0, block.rect.x0))


def _region_joins(
    units: list[list[TextBlock]],
    skip_ids: set[int],
    halt_ids: set[int],
    regions_of: Callable[[int], list[pymupdf.Rect]],
) -> list[list[TextBlock]]:
    """With the layout model enabled: two consecutive units on the same page (possibly with skipped blocks
    between them) that fall in the same body-text box of the model, with a formula at the cut, are one
    paragraph cut by the formula (... with{formula} For a Borel set ... | 1{formula} is the indicator ...).
    The first block must be paragraph text: the model also puts a list of author names or the lines of an
    algorithm into one box (see _region_prose). The model often boxes a list as one region, so a first block
    ending a sentence followed by a capitalized one is not joined (... under fixed conditions. | We show
    that ...). Headings, captions, running headers and footers, list items and run-in bold or italic labels
    are not joined either. Without the model the units are returned unchanged."""
    pages: dict[int, list[pymupdf.Rect]] = {}

    def region(block: TextBlock) -> pymupdf.Rect | None:
        if block.page not in pages:
            pages[block.page] = regions_of(block.page)
        index = _region_index(block.rect, pages[block.page])
        return None if index is None else pages[block.page][index]

    out: list[list[TextBlock]] = []
    last: int | None = None
    for unit in units:
        if all(id(block) in skip_ids for block in unit):
            out.append(unit)
            continue
        if last is not None and _region_continues(out[last][-1], unit[0], halt_ids):
            box = region(out[last][-1])
            if box is not None and box == region(unit[0]) and _region_prose(out[last][-1], box):
                out[last] = [*out[last], *unit]
                continue
        out.append(unit)
        last = len(out) - 1
    return out


def _region_prose(block: TextBlock, box: pymupdf.Rect) -> bool:
    """Paragraph text within the box: a few lowercase words (a list of author names has none), and more than
    one line, or a single line filling the box width. Slide bullet points, algorithm steps and prompt
    templates are short lines that the model also puts into one box."""
    if len(re.findall(r"\b[a-z]{3,}\b", block.text)) < _REGION_WORDS:
        return False
    return len(_visual_lines(block)) >= 2 or (
        block.rect.x0 - box.x0 <= 1.5 * block.size and box.x1 - block.rect.x1 <= 1.5 * block.size
    )


def _region_continues(prev: TextBlock, nxt: TextBlock, halt_ids: set[int]) -> bool:
    if prev.page != nxt.page or id(prev) in halt_ids or id(nxt) in halt_ids:
        return False
    if _list_item(nxt.text) or _LABEL_START_RE.match(nxt.text) or _styled_edge(nxt, 0) or _ALGO_LINE_RE.match(nxt.text):
        return False
    # A formula at the cut: in the last line of the first block or the first line of the next, or a gap where a
    # display formula sits
    if not (_math_on_line(prev, -1) or _math_on_line(nxt, 0) or nxt.rect.y0 - prev.rect.y1 >= _REGION_GAP * prev.size):
        return False
    letter = re.search(r"[A-Za-z]", nxt.text)
    # A closing quote also ends a sentence: a capitalized line below an epigraph is its source
    # (... Location, Location” | Randall Munroe)
    ended = _END_MARK_RE.sub("", prev.text).rstrip().endswith(tuple('.!?;:。！？；：”"'))
    return not (ended and letter is not None and letter.group().isupper())


def _region_index(rect: pymupdf.Rect, regions: list[pymupdf.Rect]) -> int | None:
    """The box that holds most of the block's area."""
    area = rect.width * rect.height
    if area < 1:
        return None
    best, best_share = None, _REGION_COVER
    for index, box in enumerate(regions):
        hit = rect & box
        if hit.is_empty:
            continue
        share = hit.width * hit.height / area
        if share >= best_share:
            best, best_share = index, share
    return best


# Share of a block's area that must fall in a body-text box of the model for the block to belong to it
_REGION_COVER = 0.5


# Minimum number of lowercase words in the first block when joining by model regions
_REGION_WORDS = 3


# A vertical gap of this many font sizes between two blocks holds a display formula
_REGION_GAP = 1.5


# Algorithm line numbers: 6: until converged
_ALGO_LINE_RE = re.compile(r"\d{1,3}:\s")


def _cross_page_units(
    blocks: list[TextBlock],
    median: float,
    skip_ids: set[int] = frozenset(),
    geometries: list[PageGeometry] | None = None,
    halt_ids: set[int] | None = None,
    ignore_ids: set[int] | None = None,
    aside_ids: set[int] = frozenset(),
) -> list[list[TextBlock]]:
    """Pair paragraphs carried over to the next page into translation units (a list of block lists, one or two
    blocks per unit).

    When the last body block of a page does not end with sentence punctuation and the first block of the next
    page starts with a lowercase letter or digit and does not look like a heading, both are one paragraph cut
    by the page break. Footnotes (clearly smaller than the body text, or one size smaller and starting with a
    number or link) and blocks to skip take no part. Continuations in the same column are joined as before.
    With several columns, the only addition is that the last block of the rightmost column may continue in the
    first block of the leftmost column on the next page. ignore_ids are running headers and footers: they
    neither take the page end nor block the page top. Headings across columns and whole tables do not take a
    column top or bottom either. aside_ids are figures and captions: when the next page starts with a figure,
    the cut paragraph continues below its caption.
    """
    by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    ignored = ignore_ids or set()
    indents = _indents_paragraphs(blocks)
    # Without geometry the plain page-end and page-top rules apply. With geometry, continuations across columns
    # are joined by left edges. Half sentences on a shared baseline do not depend on column geometry: the next
    # line is still sent paired with it, only without merging their boxes.
    extra: dict[int, TextBlock] = {}
    taken: set[int] = set()
    for host_id, tail in _shared_line_pairs(blocks).items():
        if host_id in skip_ids or id(tail) in skip_ids:
            continue
        if host_id in ignored or id(tail) in ignored:
            continue
        if halt_ids is not None and (host_id in halt_ids or id(tail) in halt_ids):
            continue
        extra[host_id] = tail
        taken.add(host_id)
        taken.add(id(tail))
    if geometries is not None:
        for host_id, tail in _column_continuations(
            by_page,
            median,
            skip_ids,
            halt_ids,
            ignored,
            aside_ids,
            indents,
        ).items():
            if host_id in taken or id(tail) in taken:
                continue
            extra[host_id] = tail
            taken.add(host_id)
            taken.add(id(tail))

    units: list[list[TextBlock]] = []
    pages = sorted(by_page)
    used: set[int] = {id(tail) for tail in extra.values()}
    for pno in pages:
        body = _page_edge_blocks(by_page[pno], median, ignored | aside_ids)
        later = _next_text_page(by_page, pno, median, skip_ids, ignored)
        nxt = by_page.get(later) or []
        nxt_body = _page_edge_blocks(nxt, median, ignored | aside_ids)
        first = _column_page_start(nxt_body, geometries)
        end = _column_page_end(body, geometries)
        head = _left_column_head(nxt_body, geometries[later]) if geometries and later < len(geometries) else None
        for b in by_page[pno]:
            if id(b) in used:
                continue
            if id(b) in extra:
                tail = extra[id(b)]
                units.append([b, tail])
                used.add(id(b))
                used.add(id(tail))
                continue
            if halt_ids is not None and id(b) in halt_ids:
                units.append([b])
                used.add(id(b))
                continue
            # A block already paired with another on the same page cannot also join the end of the previous page,
            # or its partner would be left out of every unit
            same = bool(
                end is not None
                and b is end
                and first is not None
                and id(first) not in used
                and id(first) not in extra
                and id(b) not in skip_ids
                and id(first) not in skip_ids
                and _can_continue(
                    b,
                    first,
                    median,
                    halt_ids,
                    flush=indents and _resumes(first, nxt, skip_ids, ignored | aside_ids),
                    at_edge=_at_text_edge(first, nxt),
                )
                and not _different_columns(geometries, b, first)
            )
            cross_target = head if head is not None and id(head) not in used and id(head) not in extra else None
            cross = bool(
                not same
                and cross_target is not None
                and geometries is not None
                and pno < len(geometries)
                and _is_last_in_right_column(b, body, geometries[pno])
                and id(b) not in skip_ids
                and id(cross_target) not in skip_ids
                and _can_continue(
                    b,
                    cross_target,
                    median,
                    halt_ids,
                    flush=indents and _resumes(cross_target, nxt, skip_ids, ignored | aside_ids),
                    at_edge=_at_text_edge(cross_target, nxt),
                )
            )
            other = first if same else cross_target
            if (same or cross) and other is not None:
                units.append([b, other])
                used.add(id(b))
                used.add(id(other))
            else:
                units.append([b])
                used.add(id(b))
    return units


def _seam_apart(
    unit: list[TextBlock],
    by_page: dict[int, list[TextBlock]],
    geometries: Sequence[PageGeometry],
    covered: set[int],
    shown: set[int],
    median: float,
    words_of: Callable[[int], list[tuple]],
) -> bool:
    """A two-block unit across pages, across columns or cut by a formula, with a display formula or example
    between the blocks: right below the first block or right above the second there is a body-text block, or
    text that is in no block (a display formula dropped during extraction). Such a unit is split back into
    two blocks at the boundary when written (see _assign_restored_parts); filling and continuing would write
    the second half before the formula or example. When what lies next to them is a figure, table, caption,
    running header or footer, footnote (covered holds figures, captions, table cells and running headers and
    footers) or nothing, the blocks are only separated by a page or column break"""
    host, tail = unit
    if host.page == tail.page and _same_page_line_pair(host, tail):
        return False
    stacked = host.page == tail.page and tail.rect.y0 >= host.rect.y1 - 1 and _x_overlap(host.rect, tail.rect)
    context = (by_page, geometries, covered, shown, median, words_of)
    if _content_next(host, tail, *context, below=True):
        return True
    return not stacked and _content_next(tail, host, *context, below=False)


def _content_next(
    block: TextBlock,
    partner: TextBlock,
    by_page: dict[int, list[TextBlock]],
    geometries: Sequence[PageGeometry],
    covered: set[int],
    shown: set[int],
    median: float,
    words_of: Callable[[int], list[tuple]],
    below: bool,
) -> bool:
    """The nearest thing directly below the block (above when below is false) is body content: a recognized
    display formula or example (shown), a block not in covered, not inside a figure or table box and not a
    footnote, or text inside the text frame that belongs to no block. The layout model takes display formulas
    for figures, so shown is checked first. A block one size smaller is the rest of a caption or a margin
    note; a much smaller one is a formula extracted as a block and counts as content. The partner on the same
    page and anything past it are not considered"""
    geo = geometries[block.page] if block.page < len(geometries) else None
    floats = [*geo.figures, *(table.rect for table in geo.tables)] if geo is not None else []
    frame = geo.frame if geo is not None else None
    edge = None if frame is None else (frame.y1 if below else frame.y0)
    page_blocks = by_page.get(block.page) or []
    found: list[tuple[float, bool]] = []

    def beyond(y0: float, y1: float) -> bool:
        if below:
            return y0 >= block.rect.y1 - 1 and not (partner.page == block.page and y0 >= partner.rect.y0 - 1)
        return y1 <= block.rect.y0 + 1 and not (partner.page == block.page and y1 <= partner.rect.y1 + 1)

    def distance(y0: float, y1: float) -> float:
        return y0 - block.rect.y1 if below else block.rect.y0 - y1

    for rect in floats:
        if _x_overlap(rect, block.rect) and beyond(rect.y0, rect.y1):
            found.append((distance(rect.y0, rect.y1), False))
    for other in page_blocks:
        if other is block or other is partner or not _x_overlap(other.rect, block.rect):
            continue
        if not beyond(other.rect.y0, other.rect.y1):
            continue
        cover = id(other) not in shown and (
            id(other) in covered
            or _is_page_footnote(other, median)
            or _is_float_text(other.text)
            or any(rect.contains(other.rect) for rect in floats)
            or block.size * 0.8 <= other.size < block.size - 0.6
        )
        found.append((distance(other.rect.y0, other.rect.y1), not cover))
    # Dropped display formulas: text a little apart from the block, inside the text frame, and in no block or
    # figure or table box
    for x0, y0, x1, y1, *_rest in words_of(block.page):
        word = pymupdf.Rect(x0, y0, x1, y1)
        if not _x_overlap(word, block.rect) or not beyond(y0, y1) or distance(y0, y1) < 0.3 * block.size:
            continue
        if edge is not None and (y0 > edge + 2 if below else y1 < edge - 2):
            continue
        middle = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any((other.rect + (-1, -1, 1, 1)).contains(middle) for other in page_blocks):
            continue
        if any(rect.contains(middle) for rect in floats):
            continue
        found.append((distance(y0, y1), True))
    return bool(found) and min(found)[1]


def _x_overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1


def _math_on_line(block: TextBlock, index: int) -> bool:
    """Line index of the block (-1 for the last) contains a math font."""
    lines = block.span_lines or []
    if not lines:
        return False
    return any(_span_is_math(span) for span in lines[index] if str(span.get("text") or "").strip())


def _shared_line_pairs(blocks: list[TextBlock]) -> dict[int, TextBlock]:
    """A continuation line on the same baseline as another paragraph is not merged into the box but is still
    sent paired. Merging it would make the writing cover that paragraph.

    The reading order may put another column between the half sentence and its continuation. The
    continuation's vertical position still matches the half sentence, so the search must not stop because
    the column in between has moved further down.
    """
    pairs: dict[int, TextBlock] = {}
    claimed: set[int] = set()
    earlier = _earlier_on_page(blocks)
    for index, block in enumerate(blocks):
        if id(block) in claimed or not _is_line_tail(block, earlier[index]):
            continue
        for nxt in blocks[index + 1 :]:
            if nxt.page != block.page:
                break
            if nxt.rect.y0 > block.rect.y1 + block.size * 2:
                continue
            if not _line_tail_match(block, nxt):
                continue
            if not _line_tail_blocked(nxt, block, blocks) or id(nxt) in claimed:
                break
            pairs[id(block)] = nxt
            claimed.add(id(block))
            claimed.add(id(nxt))
            break
    return pairs
