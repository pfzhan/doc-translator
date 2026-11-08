"""How far each paragraph can write down its page, and how the paragraphs below move to make room.

A translation that needs more height than its source can reach down into free space below, borrow free space from
the movable paragraph right below (which then moves down), move up into free space above, or carry its last lines
over to the next page. Nothing here writes to the page: _WriteContext holds what every page of the document shares
(including the flows of pages not written yet, which a paragraph carried over to them needs), and _PageFlow answers
these questions for one page.
"""

import math
from collections.abc import Callable, Collection
from dataclasses import dataclass

import pymupdf

from ..languages import RTL_LANGUAGES
from .pdf_align import SQUEEZE
from .pdf_blocks import SPILL_TAIL, TextBlock, _pages_to_write, _paragraph_indent
from .pdf_drawings import text_page_rect
from .pdf_flow import separate_after_formula, space_around_latin
from .pdf_layout import PageGeometry
from .pdf_place import _box_height, _formula_metrics, _glyph_bottom
from .pdf_types import Formula
from .pdf_typeset import typeset_lines
from .pdf_write_geometry import (
    _affiliation_write_rects,
    _beside_body,
    _body_ink,
    _drawn_marks,
    _fit_page_margins,
    _flow_y0,
    _indent_edge,
    _indented_start,
    _inset_write_boxes,
    _kept_ink,
    _room_below,
    _row_neighbor,
    _split_filled_lines,
)
from .pdf_write_rect import _write_rect

# The first half of a half sentence that shares a baseline with its continuation (see _same_page_line_pair): the text
# staying on the host's line, the host's own line box, the formulas, and whether the continuation gets text too
_SharedHead = tuple[str, pymupdf.Rect, list[Formula], bool]


class _WriteContext:
    """What every page of one document being written shares: the source blocks by page, the translations to write,
    the target script, and caches of the ink and flows of pages. A paragraph carried over to the next page estimates
    its room there before that page is written, so these live for the whole document rather than one page."""

    def __init__(
        self,
        doc: pymupdf.Document,
        blocks: list[TextBlock],
        translations: list[str],
        target_lang: str,
        formulas_map: dict[int, list[Formula]] | None,
        geometries: list[PageGeometry] | None,
        continuations: dict[int, TextBlock] | None,
        pages: Collection[int] | None,
    ) -> None:
        self.doc = doc
        self.rtl = target_lang.split("-")[0] in RTL_LANGUAGES
        # Line boxes of CJK fonts are taller than Latin ones (about 1.31em vs 1.16em), and glyph tops go past the
        # given area: written at the original font size they would overlap the next line, so the font size is
        # reduced and the text moved down to compensate
        # Cantonese and Classical Chinese use the same CJK fonts, with the same taller line boxes
        self.cjk = target_lang.split("-")[0] in ("zh", "ja", "ko", "yue", "wyw")
        # Line spacing of the translation, in font sizes
        self.gap = 1.45 if self.cjk else 1.2
        self.formulas: dict[int, list[Formula]] = formulas_map or {}
        self.geometries = geometries
        self.continuations: dict[int, TextBlock] = continuations or {}
        self.by_page = _pages_to_write(blocks, translations)
        if pages is not None:
            self.by_page = {pno: items for pno, items in self.by_page.items() if pno in pages}
        # Writing compares only with blocks on the same page. Scanning the whole book for every block is slow with
        # hundreds of pages.
        self.page_blocks: dict[int, list[TextBlock]] = {}
        for b in blocks:
            self.page_blocks.setdefault(b.page, []).append(b)
        # Width of the paragraph indent; one-line paragraphs recognized by it are written from the margin (see
        # _indent_edge)
        self.indent = _paragraph_indent(blocks)
        # Glyphs left in place and drawn figure boxes on each page (see _room_below). Replaced with the post-erase
        # ones when writing reaches the page
        self.page_inks: dict[int, tuple[list[pymupdf.Rect], list[pymupdf.Rect]]] = {}
        # _body_ink of page pno, and the page_inks entry it was computed from
        self._body_inks: dict[int, tuple[list[pymupdf.Rect], list[pymupdf.Rect]]] = {}
        self._tail_rooms: dict[tuple[int, float, float], float] = {}
        self._later_flows: dict[int, _PageFlow] = {}

    def em(self, size: float) -> float:
        """The font size the translation of a block of font size size is written at (see cjk)."""
        return size * 0.88 if self.cjk else size

    def geometry(self, pno: int) -> PageGeometry | None:
        """The page geometry of page pno, when one was given."""
        return self.geometries[pno] if self.geometries and pno < len(self.geometries) else None

    def body_ink(self, pno: int) -> list[pymupdf.Rect]:
        """_body_ink of page pno, recomputed when the page's ink was replaced after erasing."""
        kept, marks = self.page_inks[pno]
        if pno not in self._body_inks or self._body_inks[pno][0] is not kept:
            self._body_inks[pno] = kept, _body_ink(self.doc[pno], kept, marks)
        return self._body_inks[pno][1]

    def unwritten_ink(self, pno: int) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
        """The page_inks entry of page pno, read from the unerased page when writing has not reached it yet."""
        if pno not in self.page_inks:
            page = self.doc[pno]
            self.page_inks[pno] = _kept_ink(page, self.page_blocks.get(pno, [])), _drawn_marks(page)
        return self.page_inks[pno]

    def tail_room(self, tail: TextBlock, width: float, em: float) -> float:
        """How far down the tail can write from its origin, computed the same way as room when writing the tail
        (SPILL_TAIL). The tail's page is not written yet and the source glyphs are all in blocks, so glyphs left in
        place and figure boxes can still be recognized.
        When a figure or another block sits beside the tail within width to its right, lines broken at width
        would have to go around it there and wrap into more lines, so the tail is counted as having no room.
        Free space in the paragraph right below, which the tail can borrow to reach down when written (see
        spare_below), is included; when the run below has to move up, the space needed for that is not.
        em is the font size of the continued text"""
        key = (id(tail), round(width, 2), round(em, 2))
        if key not in self._tail_rooms:
            self._tail_rooms[key] = self._measure_tail_room(tail, width, em)
        return self._tail_rooms[key]

    def _measure_tail_room(self, tail: TextBlock, width: float, em: float) -> float:
        """tail_room, uncached."""
        tail_page = self.doc[tail.page]
        same = self.page_blocks.get(tail.page, [])
        kept, marks = self.unwritten_ink(tail.page)
        x0, x1 = tail.rect.x0, tail.rect.x0 + width
        if any(
            tail.rect.x1 - 2 <= ob.x0 < x1 - 4 and min(ob.y1, tail.rect.y1) - max(ob.y0, tail.rect.y0) > tail.size * 0.5
            for ob in (*marks, *(other.rect for other in same if other is not tail))
        ):
            return 0.0
        y0 = _flow_y0(tail, self.cjk)
        room = _room_below(tail_page, tail, y0, x0, x1, same, kept, marks, self.body_ink(tail.page))
        if tail.page in self.by_page:
            flow = self.later_flow(tail.page)
            below = flow.next_movable(tail, pymupdf.Rect(x0, y0, x1, room), room)
            if below is not None:
                # Reach down to half a line above the lowest origin of the paragraph below (as in _fit_typeset),
                # converted to the same measure as room: the box exceeds room by the line spacing below the last
                # line. When the run below does not fit at normal line spacing and has to move up, the tail ends
                # half a line above the raised origin (as in latest_start)
                start = flow.latest_start(*below)
                gap = self.gap
                reach = start - em * (gap * 0.5 - 0.3) - (gap - 1) * em
                if start > below[1].y0:
                    room = max(room, reach)
                elif start < below[1].y0 - 0.5:
                    room = min(room, reach)
        return room - y0

    def later_flow(self, pno: int) -> "_PageFlow":
        """The _PageFlow of page pno, which is not written yet. A paragraph below that is carried over to the next
        page does not move along (see hosts in _PageFlow)"""
        if pno not in self._later_flows:
            same = self.page_blocks.get(pno, [])
            kept, marks = self.unwritten_ink(pno)
            self._later_flows[pno] = _PageFlow(
                self,
                pno,
                self.by_page[pno],
                kept,
                marks,
                written_ids=set(),
                stacked=_affiliation_write_rects(same),
                reflowed={},
                shared_head={},
                hosts=False,
            )
        return self._later_flows[pno]

    def spill_target(self, block: TextBlock, width: float, em: float) -> tuple[TextBlock, pymupdf.Rect, float] | None:
        """When the last paragraph at the bottom of the page does not fit, its last lines are written in front of
        the first paragraph of the next page, and that paragraph and the movable paragraphs below it move down.
        Returns that paragraph, its box, and how much height can be written in front of it (as in tail_room,
        measured from the origin to where writing can reach). None when the next page starts with a block that
        cannot move, such as a heading or caption, or when its column is narrower than this paragraph"""
        later = self.by_page.get(block.page + 1)
        if not later:
            return None
        body = [b for b, t in later if t != SPILL_TAIL and not _off_body_size(b)]
        if not body:
            return None
        head = body[0]
        if any(
            other is not head
            and other.rect.y1 <= head.rect.y0 + 1
            and min(other.rect.x1, head.rect.x1) - max(other.rect.x0, head.rect.x0) > 8
            for other, _ in later
        ):
            return None
        flow = self.later_flow(head.page)
        box = flow.movable_box(head, None)
        if box is None or box.width < width - 2:
            return None
        gap = self.gap
        # Half a line below the last of the added lines is the origin of the next paragraph (as in claims);
        # converted to the next_room of _split_filled_lines
        lowest = flow.latest_start(head, box) - em * gap * 0.5
        return head, box, lowest - box.y0 + em * 0.3 - (gap - 1) * em - 1


class _PageFlow:
    """spare_below, next_movable, latest_start and movable_box of page pno. Used when writing this page, and also
    when writing a paragraph carried over from the previous page, to estimate how much the tail can borrow; then
    this page is not erased yet, items are the blocks to write, kept and marks come from the unerased page, and no
    block has been written.
    When hosts is false, a paragraph below that is carried over to the next page does not move along: moving it
    down depends on the page after, and estimating a tail does not look further ahead"""

    def __init__(
        self,
        ctx: _WriteContext,
        pno: int,
        items: list[tuple[TextBlock, str]],
        kept: list[pymupdf.Rect],
        marks: list[pymupdf.Rect],
        written_ids: set[int],
        stacked: dict[int, pymupdf.Rect],
        reflowed: dict[int, TextBlock],
        shared_head: dict[int, _SharedHead],
        hosts: bool = True,
    ) -> None:
        self.ctx = ctx
        self.page = ctx.doc[pno]
        self.same_page = ctx.page_blocks.get(pno, [])
        self.bounds = text_page_rect(self.page)
        self.geo = ctx.geometry(pno)
        self.visible = [(b, t) for b, t in items if t != SPILL_TAIL]
        self.texts = {id(b): t for b, t in items}
        self.kept = kept
        self.marks = marks
        # Blocks already written on the page; the writer adds to this set as it writes
        self.written_ids = written_ids
        self.stacked = stacked
        self.reflowed = reflowed
        self.shared_head = shared_head
        self.hosts = hosts
        # Height of each paragraph below at normal line spacing, and its height and line count with slightly
        # tightened letter spacing (see SQUEEZE). They depend only on the translation and box width, so computing
        # them once is enough
        self._heights: dict[int, tuple[float, float, int]] = {}
        self._host_starts: dict[int, float] = {}

    def _room(self, block: TextBlock, box: pymupdf.Rect) -> float:
        """_room_below the top of box, across its width."""
        return _room_below(
            self.page,
            block,
            box.y0,
            box.x0,
            box.x1,
            self.same_page,
            self.kept,
            self.marks,
            self.ctx.body_ink(block.page),
        )

    def plain_rect(self, block: TextBlock) -> pymupdf.Rect | None:
        """The box a block written as an ordinary paragraph (not carried over, not split into two boxes, not a cell
        or index entry) is written in, computed the same way as when writing below. None for other ways of
        writing"""
        ctx = self.ctx
        text = self.texts.get(id(block))
        if (
            text is None
            or text == SPILL_TAIL
            or block.index
            or block.clip is not None
            or block.side_note
            or id(block) in ctx.continuations
            or id(block) in self.stacked
            or id(block) in self.shared_head
            or id(block) in self.reflowed
            or _inset_write_boxes(block, ctx.cjk) is not None
            or _beside_body(block, self.same_page) is not None
            or _row_neighbor(block, self.same_page, self.geo)
        ):
            return None
        rect, align = _write_rect(block, self.visible, ctx.cjk, ctx.rtl, self.geo)
        if align != "left":
            return None
        edge = _indent_edge(block, self.geo, ctx.indent)
        if edge is not None:
            rect = pymupdf.Rect(edge, rect.y0, rect.x1, rect.y1)
        return _fit_page_margins(block, rect, self.same_page, self.bounds.height, ctx.cjk, ctx.body_ink(block.page))

    def next_movable(self, block: TextBlock, rect: pymupdf.Rect, room: float) -> tuple[TextBlock, pymupdf.Rect] | None:
        """When room stops at the paragraph right below, and that paragraph is not written yet, is written as an
        ordinary paragraph and can move up or down, return it and its box"""
        below = [
            other
            for other in self.same_page
            if other is not block
            and other.rect.y0 > rect.y0 + 1
            and not (other.rect.x1 < rect.x0 + 4 or other.rect.x0 > rect.x1 - 4)
        ]
        nearest = min(below, key=lambda other: other.rect.y0, default=None)
        if nearest is None or id(nearest) in self.written_ids or abs(nearest.rect.y0 - 2 - room) > 0.5:
            return None
        box = self.movable_box(nearest, block)
        if box is None and self.hosts:
            box = self.host_box(nearest, block)
        return None if box is None else (nearest, box)

    def movable_box(self, block: TextBlock, above: TextBlock | None = None) -> pymupdf.Rect | None:
        """The box a paragraph that can move up or down is written in (see plain_rect), or None when it cannot move.
        above is the paragraph right above it"""
        # Only body paragraphs move. Labels in a figure (words on a tree diagram) each match a position in the
        # figure and would no longer match if moved
        if _off_body_size(block):
            return None
        box = self.plain_rect(block)
        if box is None or self.pinned(block, box, above):
            return None
        return box

    def pinned(self, block: TextBlock, box: pymupdf.Rect, above: TextBlock | None) -> bool:
        """The paragraph's box contains something that does not move with it"""
        return any(_pins(ink, block, box) for ink in (*self.kept, *self.marks)) or any(
            _pins(other.rect, block, box, other)
            for other in self.same_page
            if other is not block and other is not above
        )

    def host_box(self, block: TextBlock, above: TextBlock | None = None) -> pymupdf.Rect | None:
        """The box a paragraph carried over to the next page (next column) is written in when written as a host
        (not split into two boxes, not a half sentence, not on the same line as its continuation). When it moves
        down, fewer lines stay on this page and the extra lines go to the tail"""
        ctx = self.ctx
        text = self.texts.get(id(block))
        tail = ctx.continuations.get(id(block))
        if (
            text is None
            or text == SPILL_TAIL
            or tail is None
            or block.index
            or block.clip is not None
            or block.side_note
            or id(block) in self.stacked
            or id(block) in self.shared_head
            or id(block) in self.reflowed
            or _off_body_size(block)
            or _inset_write_boxes(block, ctx.cjk) is not None
            or _beside_body(block, self.same_page) is not None
        ):
            return None
        rect, align = _write_rect(block, self.visible, ctx.cjk, ctx.rtl, self.geo)
        if align != "left" or (len(block.line_rects) <= 1 and rect.width < tail.rect.width * 0.7):
            return None
        left = _indented_start(block, tail, ctx.geometries)
        if left is not None:
            rect = pymupdf.Rect(left, rect.y0, rect.x1, rect.y1)
        return None if self.pinned(block, rect, above) else rect

    def host_latest(self, block: TextBlock, box: pymupdf.Rect) -> float:
        """The lowest origin of a paragraph carried over to the next page at which the lines staying on this page fit
        at normal line spacing and the remaining lines also fit at normal line spacing in the tail. When they do not
        fit even at the original position, it does not move and starts there"""
        if id(block) in self._host_starts:
            return self._host_starts[id(block)]
        ctx = self.ctx
        size = ctx.em(block.size)
        gap = ctx.gap
        formulas = ctx.formulas.get(id(block), [])
        room = self._room(block, box)
        floor = _fit_page_margins(
            block,
            pymupdf.Rect(box.x0, box.y0, box.x1, self.bounds.y1),
            self.same_page,
            self.bounds.height,
            ctx.cjk,
            ctx.body_ink(block.page),
        ).y1
        next_room = ctx.tail_room(ctx.continuations[id(block)], box.width, size)
        bottom = min(room + (gap - 1) * size, floor)
        lines = _PageBreak(
            self.texts[id(block)], formulas, size, block.size, box.width, ctx.cjk, room, floor, next_room
        )

        def fits(y0: float) -> bool:
            head, rest = lines.split(y0)
            if not head or lines.squeezed(head, y0):
                return False
            return not rest or (rest.count("\n") + 1) * size * gap + size * 0.3 <= next_room + (gap - 1) * size + 1

        latest = box.y0
        if fits(box.y0):
            latest = _last_fitting(fits, latest, bottom)
        self._host_starts[id(block)] = latest
        return latest

    def latest_start(self, block: TextBlock, box: pymupdf.Rect) -> float:
        """The lowest origin of a movable paragraph (see next_movable) at which it and the run of movable paragraphs
        below it all still fit at normal line spacing. Below the original origin, the difference is how far it can
        move down, lent to the paragraph above; above it, how far it has to move up.
        When the paragraph below has to move up, this one must end half a line above it, so the original box bottom
        can no longer be used"""
        ctx = self.ctx
        if id(block) in ctx.continuations:
            return self.host_latest(block, box)
        size = ctx.em(block.size)
        gap = ctx.gap
        normal, squeezed, count = self._paragraph_heights(block, box)
        room = self._room(block, box)
        fixed = room
        # Write in the box down to the last source line (the box is taller than the line boxes by the 0.3em
        # overhang above the first line)
        own = min(box.y1, block.rect.y1 + size * 0.3)
        below = self.next_movable(block, box, room)
        if below is not None:
            start = self.latest_start(*below)
            if start > below[1].y0:
                room = max(room, start)
            elif start < below[1].y0 - 0.5:
                room, own = min(room, start), -math.inf
        # When the box cannot hold the text, write down to half a line above the block below, as _fit_typeset does
        # when growing downward. A small difference between the estimated and written width loses a line, so 1pt is
        # left
        reach = room - size * (gap * 0.5 - 0.3)
        latest = max(own, reach) - normal - 1
        # When the text fits at the original position (measured as in _fit_typeset, see _glyph_bottom, with
        # slightly tightened letter spacing allowed), it need not move up; the part of the box below the last
        # source line goes unused only when the block moves down
        # The blank part below the last line extending past the box bottom is measured against what stays fixed
        # below; space lent by the paragraph below does not count, because that paragraph moves down half a line
        # into it
        bottom = _glyph_bottom(box.y1, fixed, size, gap) if ctx.cjk and count > 1 else box.y1
        if own > -math.inf and box.y0 + squeezed <= bottom + 1:
            latest = max(latest, box.y0)
        return latest

    def _paragraph_heights(self, block: TextBlock, box: pymupdf.Rect) -> tuple[float, float, int]:
        """The paragraph's height at normal line spacing, and its height and line count with slightly tightened
        letter spacing, at the width of box (cached, see _heights)."""
        if id(block) not in self._heights:
            ctx = self.ctx
            size = ctx.em(block.size)
            # Break lines by formula slots as _fit_typeset does; lines with tall formulas get more spacing
            text = space_around_latin(separate_after_formula(self.texts[id(block)]))
            formulas = ctx.formulas.get(id(block), [])
            metrics = _formula_metrics(formulas, block.size, size, ctx.em(block.size))
            plain, tight = (
                typeset_lines(text, size, max(box.width, 1.0) * squeeze, metrics)
                for squeeze in (1.0, 1 + SQUEEZE if ctx.cjk else 1.0)
            )
            self._heights[id(block)] = (
                _box_height(plain, size, ctx.gap),
                _box_height(tight, size, ctx.gap),
                len(tight),
            )
        return self._heights[id(block)]

    def spare_below(self, block: TextBlock, rect: pymupdf.Rect, room: float) -> tuple[TextBlock, float] | None:
        """When room stops at the paragraph right below: the height left free after that paragraph is written at
        normal line spacing. This paragraph can borrow it to reach down, and that paragraph moves down. Returns that
        paragraph and its lowest origin.
        That paragraph can itself borrow from the one below it, paragraph after paragraph moving down, and what it
        borrows counts toward how far it can write"""
        below = self.next_movable(block, rect, room)
        if below is None:
            return None
        start = self.latest_start(*below)
        if start <= below[1].y0:
            return None
        return below[0], start


@dataclass(frozen=True)
class _PageBreak:
    """A paragraph whose last lines go to another box (the tail on the next page or column, or in front of the first
    paragraph of the next page): splitting it into the lines staying here and the rest. room and floor bound how
    far down it can write here (see _split_filled_lines), next_room how much height the rest has there; size is the
    font size written, block_size the source block's"""

    text: str
    formulas: list[Formula]
    size: float
    block_size: float
    width: float
    cjk: bool
    room: float
    floor: float
    next_room: float

    def split(self, y0: float) -> tuple[str, str]:
        """The lines staying in this box when it starts at y0, and the rest."""
        return _split_filled_lines(
            self.text,
            self.formulas,
            self.size,
            self.block_size,
            self.width,
            self.cjk,
            y0,
            self.room,
            self.floor,
            self.next_room,
        )

    def squeezed(self, head: str, y0: float) -> bool:
        """The lines staying on this page do not fit at normal line spacing (lines counted as in
        _split_filled_lines)"""
        gap = 1.45 if self.cjk else 1.2
        lines = head.count("\n") + 1
        size = self.size
        return bool(head) and y0 + lines * size * gap + size * 0.3 > min(self.room + (gap - 1) * size, self.floor) + 1


def _last_fitting(fits: Callable[[float], bool], low: float, high: float) -> float:
    """Bisect between low, where fits holds, and high for the highest value where it still holds (8 steps)."""
    for _ in range(8):
        middle = (low + high) / 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def _off_body_size(block: TextBlock) -> bool:
    """The block is not at the body font size (labels in a figure, headings), so it does not move."""
    return bool(block.body_size and abs(block.size - block.body_size) > 0.5)


def _pins(ink: pymupdf.Rect, block: TextBlock, box: pymupdf.Rect, other: TextBlock | None = None) -> bool:
    """ink (or the block other) lies in the paragraph's box and does not move with it (see _PageFlow.pinned)."""
    # Inline formulas are pasted at their placeholders in the translation and move with the paragraph. Glyphs left
    # in place (dropped display formulas, equation numbers), figure boxes and other blocks side by side in the box
    # (Airlines-Wagner) do not move with the paragraph, which would cover them if moved. Things reaching into the
    # top from above (formula boxes) are avoided by lift_limit when moving up; the box of the paragraph below
    # overlaps the bottom of this one, moves down with it, and is handled by room
    if not (ink.y0 < block.rect.y1 - 1 and ink.y1 > block.rect.y0 + 1 and ink.x0 < box.x1 and ink.x1 > box.x0):
        return False
    if ink.y0 < block.rect.y0 - 1 and ink.y1 < block.rect.y1 - 1:
        return False
    return not (
        other is not None
        and ink.y0 > block.rect.y0 + 1
        and ink.y1 > block.rect.y1 + 1
        and min(ink.x1, box.x1) - max(ink.x0, box.x0) > box.width * 0.5
    )
