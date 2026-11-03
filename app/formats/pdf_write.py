"""Erase the original text of each page and write the translations in its place."""

import html
import math
from collections.abc import Callable, Collection
from pathlib import Path

import pymupdf

from ..languages import RTL_LANGUAGES
from .pdf_align import SQUEEZE
from .pdf_blocks import (
    SPILL_TAIL,
    TextBlock,
    _attach_ink,
    _formula_parts,
    _pages_to_write,
    _paragraph_indent,
    _same_page_line_pair,
    _span_is_math,
    _with_source_bold,
)
from .pdf_erase import Erasure, erase, restore_glyphs, type3_glyphs
from .pdf_flow import (
    measure_width,
    separate_after_formula,
    space_around_latin,
)
from .pdf_formula_images import _place_formula_images
from .pdf_index_reflow import _index_reflow
from .pdf_layout import (
    PageGeometry,
    text_page_rect,
)
from .pdf_order import ReadingOrder
from .pdf_place import (
    _box_height,
    _expand_left,
    _formula_metrics,
    _FormulaSources,
    _glyph_bottom,
    _insert_fitting,
    _place_plain,
    _place_typeset,
)
from .pdf_type3 import broken_type3_names
from .pdf_typeset import (
    typeset_lines,
)
from .pdf_write_geometry import (
    _affiliation_write_rects,
    _attach_chars,
    _beside_body,
    _beside_rows,
    _body_first_starts,
    _body_ink,
    _Carry,
    _clip_write_rect,
    _closes_clause,
    _drawn_marks,
    _extend_leader,
    _filled_bottom,
    _first_row_room,
    _fit_page_margins,
    _flow_y0,
    _flush_png,
    _hang_indent,
    _hosts_first,
    _indent_edge,
    _indented_start,
    _inset_write_boxes,
    _kept_ink,
    _on_rows,
    _own_line_box,
    _page_content_bottom,
    _page_form,
    _room_below,
    _row_neighbor,
    _source_marked,
    _split_filled_lines,
    _split_own_lines,
    _translation_html,
    _wrap_holes,
    _write_limit,
)
from .pdf_write_rect import _write_rect


def _render_translated(
    src_path: Path,
    blocks: list[TextBlock],
    translations: list[str],
    target_lang: str = "",
    formulas_map: dict | None = None,
    archive=None,
    geometries: list[PageGeometry] | None = None,
    continuations: dict[int, TextBlock] | None = None,
    on_page: Callable[[int, int], None] | None = None,
    pages: Collection[int] | None = None,
) -> pymupdf.Document:
    """on_page(pages written, pages to write) is called before each page is written. When pages is given, only
    those pages are written."""
    rtl = target_lang.split("-")[0] in RTL_LANGUAGES
    # Line boxes of CJK fonts are taller than Latin ones (about 1.31em vs 1.16em), and glyph tops go past the given
    # area: written at the original font size they would overlap the next line, so the font size is reduced and
    # the text moved down to compensate
    # Cantonese and Classical Chinese use the same CJK fonts, with the same taller line boxes
    cjk = target_lang.split("-")[0] in ("zh", "ja", "ko", "yue", "wyw")
    # lang lets MuPDF choose the right glyphs for Chinese, Japanese and Korean; dir=rtl sets Arabic and Hebrew
    # right to left
    div_attrs = f' lang="{html.escape(target_lang)}"' if target_lang else ""
    if rtl:
        div_attrs += ' dir="rtl" style="text-align: right"'
    continuations = continuations or {}
    # Formulas of a paragraph carried over to the next page are recorded under the block on the previous page; when
    # the text continues on this page, the fraction bars in place must be erased according to the tail
    spill_hosts = {id(tail): host for host, tail in continuations.items()}
    carried: dict[int, _Carry] = {}
    doc = pymupdf.open(src_path)
    by_page = _pages_to_write(blocks, translations)
    if pages is not None:
        by_page = {pno: items for pno, items in by_page.items() if pno in pages}

    # Writing compares only with blocks on the same page. Scanning the whole book for every block is slow with
    # hundreds of pages.
    page_blocks: dict[int, list[TextBlock]] = {}
    for b in blocks:
        page_blocks.setdefault(b.page, []).append(b)
    # Width of the paragraph indent; one-line paragraphs recognized by it are written from the margin (see
    # _indent_edge)
    indent = _paragraph_indent(blocks)
    # Glyphs left in place and drawn figure boxes on each page (see _room_below). Replaced with the post-erase ones
    # when writing reaches the page
    page_inks: dict[int, tuple[list[pymupdf.Rect], list[pymupdf.Rect]]] = {}
    # _body_ink of page pno, and the page_inks entry it was computed from
    body_inks: dict[int, tuple[list[pymupdf.Rect], list[pymupdf.Rect]]] = {}

    def body_ink(pno: int) -> list[pymupdf.Rect]:
        kept, marks = page_inks[pno]
        if pno not in body_inks or body_inks[pno][0] is not kept:
            body_inks[pno] = kept, _body_ink(doc[pno], kept, marks)
        return body_inks[pno][1]

    def page_flow(
        pno: int,
        items: list[tuple[TextBlock, str]],
        kept: list[pymupdf.Rect],
        marks: list[pymupdf.Rect],
        written_ids: set[int],
        stacked: dict,
        reflowed: dict,
        shared_head: dict,
        hosts: bool = True,
    ) -> tuple[Callable, Callable, Callable, Callable]:
        """spare_below, next_movable, latest_start and movable_box of page pno. Used when writing this page, and
        also when writing a paragraph carried over from the previous page, to estimate how much the tail can
        borrow; then this page is not erased yet, items are the blocks to write, kept and marks come from the
        unerased page, and no block has been written.
        When hosts is false, a paragraph below that is carried over to the next page does not move along: moving it
        down depends on the page after, and estimating a tail does not look further ahead"""
        page = doc[pno]
        same_page = page_blocks.get(pno, [])
        bounds = text_page_rect(page)
        geo = geometries[pno] if geometries and pno < len(geometries) else None
        visible = [(b, t) for b, t in items if t != SPILL_TAIL]
        texts = {id(b): t for b, t in items}
        # Height of each paragraph below at normal line spacing, and its height and line count with slightly
        # tightened letter spacing (see SQUEEZE). They depend only on the translation and box width, so computing
        # them once is enough
        heights: dict[int, tuple[float, float, int]] = {}
        host_starts: dict[int, float] = {}

        def plain_rect(block: TextBlock) -> pymupdf.Rect | None:
            """The box a block written as an ordinary paragraph (not carried over, not split into two boxes, not a
            cell or index entry) is written in, computed the same way as when writing below. None for other ways of
            writing"""
            text = texts.get(id(block))
            if (
                text is None
                or text == SPILL_TAIL
                or block.index
                or block.clip is not None
                or block.side_note
                or id(block) in continuations
                or id(block) in stacked
                or id(block) in shared_head
                or id(block) in reflowed
                or _inset_write_boxes(block, cjk) is not None
                or _beside_body(block, same_page) is not None
                or _row_neighbor(block, same_page, geo)
            ):
                return None
            rect, align = _write_rect(block, visible, cjk, rtl, geo)
            if align != "left":
                return None
            edge = _indent_edge(block, geo, indent)
            if edge is not None:
                rect = pymupdf.Rect(edge, rect.y0, rect.x1, rect.y1)
            return _fit_page_margins(block, rect, same_page, bounds.height, cjk, body_ink(block.page))

        def next_movable(block: TextBlock, rect: pymupdf.Rect, room: float) -> tuple[TextBlock, pymupdf.Rect] | None:
            """When room stops at the paragraph right below, and that paragraph is not written yet, is written as
            an ordinary paragraph and can move up or down, return it and its box"""
            below = [
                other
                for other in same_page
                if other is not block
                and other.rect.y0 > rect.y0 + 1
                and not (other.rect.x1 < rect.x0 + 4 or other.rect.x0 > rect.x1 - 4)
            ]
            nearest = min(below, key=lambda other: other.rect.y0, default=None)
            if nearest is None or id(nearest) in written_ids or abs(nearest.rect.y0 - 2 - room) > 0.5:
                return None
            box = movable_box(nearest, block)
            if box is None and hosts:
                box = host_box(nearest, block)
            return None if box is None else (nearest, box)

        def movable_box(block: TextBlock, above: TextBlock | None = None) -> pymupdf.Rect | None:
            """The box a paragraph that can move up or down is written in (see plain_rect), or None when it cannot
            move. above is the paragraph right above it"""
            # Only body paragraphs move. Labels in a figure (words on a tree diagram) each match a position in the
            # figure and would no longer match if moved
            if block.body_size and abs(block.size - block.body_size) > 0.5:
                return None
            box = plain_rect(block)
            if box is None or pinned(block, box, above):
                return None
            return box

        def pinned(block: TextBlock, box: pymupdf.Rect, above: TextBlock | None) -> bool:
            """The paragraph's box contains something that does not move with it"""

            # Inline formulas are pasted at their placeholders in the translation and move with the paragraph.
            # Glyphs left in place (dropped display formulas, equation numbers), figure boxes and other blocks side
            # by side in the box (Airlines-Wagner) do not move with the paragraph, which would cover them if moved.
            # Things reaching into the top from above (formula boxes) are avoided by lift_limit when moving up;
            # the box of the paragraph below overlaps the bottom of this one, moves down with it, and is handled by
            # room
            def fixed(ink: pymupdf.Rect, other: TextBlock | None = None) -> bool:
                if not (
                    ink.y0 < block.rect.y1 - 1 and ink.y1 > block.rect.y0 + 1 and ink.x0 < box.x1 and ink.x1 > box.x0
                ):
                    return False
                if ink.y0 < block.rect.y0 - 1 and ink.y1 < block.rect.y1 - 1:
                    return False
                return not (
                    other is not None
                    and ink.y0 > block.rect.y0 + 1
                    and ink.y1 > block.rect.y1 + 1
                    and min(ink.x1, box.x1) - max(ink.x0, box.x0) > box.width * 0.5
                )

            return any(fixed(ink) for ink in (*kept, *marks)) or any(
                fixed(other.rect, other) for other in same_page if other is not block and other is not above
            )

        def host_box(block: TextBlock, above: TextBlock | None = None) -> pymupdf.Rect | None:
            """The box a paragraph carried over to the next page (next column) is written in when written as a host
            (not split into two boxes, not a half sentence, not on the same line as its continuation). When it
            moves down, fewer lines stay on this page and the extra lines go to the tail"""
            text = texts.get(id(block))
            tail = continuations.get(id(block))
            if (
                text is None
                or text == SPILL_TAIL
                or tail is None
                or block.index
                or block.clip is not None
                or block.side_note
                or id(block) in stacked
                or id(block) in shared_head
                or id(block) in reflowed
                or (block.body_size and abs(block.size - block.body_size) > 0.5)
                or _inset_write_boxes(block, cjk) is not None
                or _beside_body(block, same_page) is not None
            ):
                return None
            rect, align = _write_rect(block, visible, cjk, rtl, geo)
            if align != "left" or (len(block.line_rects) <= 1 and rect.width < tail.rect.width * 0.7):
                return None
            left = _indented_start(block, tail, geometries)
            if left is not None:
                rect = pymupdf.Rect(left, rect.y0, rect.x1, rect.y1)
            return None if pinned(block, rect, above) else rect

        def host_latest(block: TextBlock, box: pymupdf.Rect) -> float:
            """The lowest origin of a paragraph carried over to the next page at which the lines staying on this page
            fit at normal line spacing and the remaining lines also fit at normal line spacing in the tail. When
            they do not fit even at the original position, it does not move and starts there"""
            if id(block) in host_starts:
                return host_starts[id(block)]
            size = block.size * 0.88 if cjk else block.size
            gap = 1.45 if cjk else 1.2
            formulas = (formulas_map or {}).get(id(block), [])
            room = _room_below(page, block, box.y0, box.x0, box.x1, same_page, kept, marks, body_ink(block.page))
            floor = _fit_page_margins(
                block,
                pymupdf.Rect(box.x0, box.y0, box.x1, bounds.y1),
                same_page,
                bounds.height,
                cjk,
                body_ink(block.page),
            ).y1
            next_room = tail_room(continuations[id(block)], box.width, size)
            bottom = min(room + (gap - 1) * size, floor)

            def fits(y0: float) -> bool:
                head, rest = _split_filled_lines(
                    texts[id(block)],
                    formulas,
                    size,
                    block.size,
                    box.width,
                    cjk,
                    y0,
                    room,
                    floor,
                    next_room,
                )
                if not head or y0 + (head.count("\n") + 1) * size * gap + size * 0.3 > bottom + 1:
                    return False
                return not rest or (rest.count("\n") + 1) * size * gap + size * 0.3 <= next_room + (gap - 1) * size + 1

            latest = box.y0
            if fits(box.y0):
                high = bottom
                for _ in range(8):
                    middle = (latest + high) / 2
                    if fits(middle):
                        latest = middle
                    else:
                        high = middle
            host_starts[id(block)] = latest
            return latest

        def latest_start(block: TextBlock, box: pymupdf.Rect) -> float:
            """The lowest origin of a movable paragraph (see next_movable) at which it and the run of movable
            paragraphs below it all still fit at normal line spacing. Below the original origin, the difference is
            how far it can move down, lent to the paragraph above; above it, how far it has to move up.
            When the paragraph below has to move up, this one must end half a line above it, so the original box
            bottom can no longer be used"""
            if id(block) in continuations:
                return host_latest(block, box)
            size = block.size * 0.88 if cjk else block.size
            gap = 1.45 if cjk else 1.2
            if id(block) not in heights:
                # Break lines by formula slots as _fit_typeset does; lines with tall formulas get more spacing
                text = space_around_latin(separate_after_formula(texts[id(block)]))
                formulas = (formulas_map or {}).get(id(block), [])
                metrics = _formula_metrics(formulas, block.size, size, block.size * 0.88 if cjk else block.size)
                plain, tight = (
                    typeset_lines(text, size, max(box.width, 1.0) * squeeze, metrics)
                    for squeeze in (1.0, 1 + SQUEEZE if cjk else 1.0)
                )
                heights[id(block)] = (_box_height(plain, size, gap), _box_height(tight, size, gap), len(tight))
            normal, squeezed, count = heights[id(block)]
            room = _room_below(page, block, box.y0, box.x0, box.x1, same_page, kept, marks, body_ink(block.page))
            fixed = room
            # Write in the box down to the last source line (the box is taller than the line boxes by the 0.3em
            # overhang above the first line)
            own = min(box.y1, block.rect.y1 + size * 0.3)
            below = next_movable(block, box, room)
            if below is not None:
                start = latest_start(*below)
                if start > below[1].y0:
                    room = max(room, start)
                elif start < below[1].y0 - 0.5:
                    room, own = min(room, start), -math.inf
            # When the box cannot hold the text, write down to half a line above the block below, as _fit_typeset
            # does when growing downward. A small difference between the estimated and written width loses a line,
            # so 1pt is left
            reach = room - size * (gap * 0.5 - 0.3)
            latest = max(own, reach) - normal - 1
            # When the text fits at the original position (measured as in _fit_typeset, see _glyph_bottom, with
            # slightly tightened letter spacing allowed), it need not move up; the part of the box below the last
            # source line goes unused only when the block moves down
            # The blank part below the last line extending past the box bottom is measured against what stays fixed
            # below; space lent by the paragraph below does not count, because that paragraph moves down half a
            # line into it
            bottom = _glyph_bottom(box.y1, fixed, size, gap) if cjk and count > 1 else box.y1
            if own > -math.inf and box.y0 + squeezed <= bottom + 1:
                latest = max(latest, box.y0)
            return latest

        def spare_below(block: TextBlock, rect: pymupdf.Rect, room: float) -> tuple[TextBlock, float] | None:
            """When room stops at the paragraph right below: the height left free after that paragraph is written at
            normal line spacing. This paragraph can borrow it to reach down, and that paragraph moves down. Returns
            that paragraph and its lowest origin.
            That paragraph can itself borrow from the one below it, paragraph after paragraph moving down, and what
            it borrows counts toward how far it can write"""
            below = next_movable(block, rect, room)
            if below is None:
                return None
            start = latest_start(*below)
            if start <= below[1].y0:
                return None
            return below[0], start

        return spare_below, next_movable, latest_start, movable_box

    def tail_room(tail: TextBlock, width: float, em: float) -> float:
        """How far down the tail can write from its origin, computed the same way as room when writing the tail
        (SPILL_TAIL). The tail's page is not written yet and the source glyphs are all in blocks, so glyphs left in
        place and figure boxes can still be recognized.
        When a figure or another block sits beside the tail within width to its right, lines broken at width
        would have to go around it there and wrap into more lines, so the tail is counted as having no room.
        Free space in the paragraph right below, which the tail can borrow to reach down when written (see
        spare_below), is included; when the run below has to move up, the space needed for that is not.
        em is the font size of the continued text"""
        key = (id(tail), round(width, 2), round(em, 2))
        if key not in tail_rooms:
            tail_rooms[key] = measure_tail_room(tail, width, em)
        return tail_rooms[key]

    tail_rooms: dict[tuple[int, float, float], float] = {}

    def later_flow(pno: int) -> tuple[Callable, Callable, Callable, Callable]:
        """page_flow of page pno, which is not written yet. A paragraph below that is carried over to the next page
        does not move along (see hosts in page_flow)"""
        if pno not in later_flows:
            later_page = doc[pno]
            same = page_blocks.get(pno, [])
            if pno not in page_inks:
                page_inks[pno] = _kept_ink(later_page, same), _drawn_marks(later_page)
            kept, marks = page_inks[pno]
            later_flows[pno] = page_flow(
                pno,
                by_page[pno],
                kept,
                marks,
                set(),
                _affiliation_write_rects(same),
                {},
                {},
                False,
            )
        return later_flows[pno]

    later_flows: dict[int, tuple[Callable, Callable, Callable, Callable]] = {}

    def measure_tail_room(tail: TextBlock, width: float, em: float) -> float:
        tail_page = doc[tail.page]
        same = page_blocks.get(tail.page, [])
        if tail.page not in page_inks:
            page_inks[tail.page] = _kept_ink(tail_page, same), _drawn_marks(tail_page)
        kept, marks = page_inks[tail.page]
        x0, x1 = tail.rect.x0, tail.rect.x0 + width
        if any(
            tail.rect.x1 - 2 <= ob.x0 < x1 - 4 and min(ob.y1, tail.rect.y1) - max(ob.y0, tail.rect.y0) > tail.size * 0.5
            for ob in (*marks, *(other.rect for other in same if other is not tail))
        ):
            return 0.0
        y0 = _flow_y0(tail, cjk)
        room = _room_below(tail_page, tail, y0, x0, x1, same, kept, marks, body_ink(tail.page))
        if tail.page in by_page:
            _, next_movable, latest_start, _ = later_flow(tail.page)
            below = next_movable(tail, pymupdf.Rect(x0, y0, x1, room), room)
            if below is not None:
                # Reach down to half a line above the lowest origin of the paragraph below (as in _fit_typeset),
                # converted to the same measure as room: the box exceeds room by the line spacing below the last
                # line. When the run below does not fit at normal line spacing and has to move up, the tail ends
                # half a line above the raised origin (as in latest_start)
                start = latest_start(*below)
                gap = 1.45 if cjk else 1.2
                reach = start - em * (gap * 0.5 - 0.3) - (gap - 1) * em
                if start > below[1].y0:
                    room = max(room, reach)
                elif start < below[1].y0 - 0.5:
                    room = min(room, reach)
        return room - y0

    def spill_target(block: TextBlock, width: float, em: float) -> tuple[TextBlock, pymupdf.Rect, float] | None:
        """When the last paragraph at the bottom of the page does not fit, its last lines are written in front of
        the first paragraph of the next page, and that paragraph and the movable paragraphs below it move down.
        Returns that paragraph, its box, and how much height can be written in front of it (as in tail_room,
        measured from the origin to where writing can reach). None when the next page starts with a block that
        cannot move, such as a heading or caption, or when its column is narrower than this paragraph"""
        later = by_page.get(block.page + 1)
        if not later:
            return None
        body = [b for b, t in later if t != SPILL_TAIL and not (b.body_size and abs(b.size - b.body_size) > 0.5)]
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
        _, _, latest_start, movable_box = later_flow(head.page)
        box = movable_box(head)
        if box is None or box.width < width - 2:
            return None
        gap = 1.45 if cjk else 1.2
        # Half a line below the last of the added lines is the origin of the next paragraph (as in claims);
        # converted to the next_room of _split_filled_lines
        lowest = latest_start(head, box) - em * gap * 0.5
        return head, box, lowest - box.y0 + em * 0.3 - (gap - 1) * em - 1

    # Last lines of a paragraph at the bottom of a page written on the next page, keyed by the first paragraph of
    # that page
    spilled: dict[int, tuple[pymupdf.Rect, _Carry]] = {}

    for written, (pno, items) in enumerate(by_page.items()):
        if on_page is not None:
            on_page(written, len(by_page))
        page = doc[pno]
        same_page = page_blocks.get(pno, [])
        # page.rect of a rotated page is rotated; comparing with text coordinates needs the unrotated page box
        bounds = text_page_rect(page)
        items = _hosts_first(items, continuations)
        visible = [(b, t) for b, t in items if t != SPILL_TAIL]
        geo = geometries[pno] if geometries and pno < len(geometries) else None
        mid_map = {f["mid"]: f for b, _ in visible for f in (formulas_map or {}).get(id(b), []) if "mid" in f}
        # Before redaction, pack the whole page into a Form: formula pass-through clips the original glyphs from it;
        # also collect all spans of the page (to detect superscript areas intruded on by descenders from the line
        # above). Bitmap glyphs without Unicode report the font box, half a line taller than the ink; cutting
        # holes with them would cut away the formula itself, so they are left out
        bitmap_fonts = broken_type3_names(doc, page) if mid_map else set()
        page_spans = None
        if mid_map:
            page_data = page.get_text("dict")
            _attach_ink(page, page_data)
            page_spans = [
                s
                for blk in page_data["blocks"]
                if blk.get("type") == 0
                for ln in blk["lines"]
                for s in ln["spans"]
                if s.get("font") not in bitmap_fonts
            ]
            # When the glyphs of a formula were cut out of a body span, clipping the formula identifies them by
            # character, which can no longer be looked up after erasing
            if any(
                span.get("source")
                for f in mid_map.values()
                for part in _formula_parts(f)
                for span in part.get("spans") or []
            ):
                _attach_chars(page, page_spans)
        forms = _FormulaSources(doc, page, _page_form(doc, page), mid_map.values()) if mid_map else None
        # Formulas paste the original page packed before erasing and are not affected. Bitmap glyphs are shared by
        # the whole document, and the filters for erasing and clipping formulas change them, so they are written
        # back unchanged after this page is done
        glyphs = type3_glyphs(page) if mid_map else {}
        stacked = _affiliation_write_rects(same_page)
        fixed = {*stacked, *continuations, *(id(tail) for tail in continuations.values())}
        reflowed = _index_reflow(
            items,
            same_page,
            geo,
            cjk,
            formulas_map or {},
            fixed,
            _page_content_bottom(same_page, pno, bounds.height),
            _kept_ink(page, same_page) if any(b.index for b in same_page) else (),
        )
        # Other blocks extending right or wrapping avoid the moved index entries at their new positions
        layout = [reflowed.get(id(ob), ob) for ob in same_page]
        written = {id(b) for b, _ in items}
        items = [
            *items,
            *(
                (b, _with_source_bold(b, _source_marked(b)))
                for b in same_page
                if id(b) in reflowed and id(b) not in written
            ),
        ]
        erasure = Erasure(math=_span_is_math)
        for b, _ in items:
            erasure.add_block(b.span_lines, b.line_rects or [b.rect])
            # The leader of a table of contents entry is erased too, and after the translation is written it is
            # filled in from the page number back to the end of the translation
            if b.leader is not None:
                erasure.spans.extend(b.leader.dots)
            # Fraction bars and radical bars taken into a formula move with the formula image, so the originals
            # must be erased. After line boxes are tightened to the ink, fraction bars often lie outside them
            erasure.lines.extend(
                pymupdf.Rect(curve) + (-0.5, -0.5, 0.5, 0.5)
                for key in (id(b), spill_hosts.get(id(b)))
                for f in (formulas_map or {}).get(key, [])
                for part in _formula_parts(f)
                for curve in part.get("curves") or []
                if curve is not None and int(part.get("page", f.get("page", pno))) == pno
            )
            # A formula image is cut out by its whole box and pasted at the new position, so every glyph whose
            # origin is in the box must be erased in place. Glyphs without Unicode (the large norm bars of CMEX
            # extract as \r) are treated as blank and are not in any span, so erasing by span misses them
            erasure.spanless.extend(
                pymupdf.Rect(part["bbox"])
                for key in (id(b), spill_hosts.get(id(b)))
                for f in (formulas_map or {}).get(key, [])
                for part in _formula_parts(f)
                if part.get("bbox") is not None and int(part.get("page", f.get("page", pno))) == pno
            )
        # Before erasing, place markers at the source positions; after writing, the translation is inserted back
        # there, so drag selection and copying follow the reading order
        order = ReadingOrder(page, [b for b, _ in items])
        erase(page, erasure)
        order.after_erase()
        # Glyphs still in place after erasing (dropped display formulas, equation numbers) must be avoided when the
        # translation extends right or continues
        kept = _kept_ink(page, same_page)
        # Continued text reaching down must also avoid figure boxes and images, which are not in blocks. Read
        # before writing the translation: the formulas written are drawn too
        spills = any(t == SPILL_TAIL or id(b) in continuations for b, t in items)
        marks = _drawn_marks(page) if spills else []
        page_inks[pno] = kept, marks
        # Rewriting the content stream turns /Resources into an inline dictionary; after the typesetter writes,
        # /XObject has hundreds of entries, and such a large nested inline dictionary gets corrupted by the GC on
        # save (garbage≥3) (the entry of the first form points to a font stream and content is silently lost).
        # Making it an indirect reference avoids this MuPDF bug.
        res_v = doc.xref_get_key(page.xref, "Resources")[1]
        if res_v.startswith("<<"):
            res_x = doc.get_new_xref()
            doc.update_object(res_x, res_v)
            doc.xref_set_key(page.xref, "Resources", f"{res_x} 0 R")
        images_before = {img[0] for img in page.get_images(full=True)}
        deferred_png: list[tuple[pymupdf.Rect, bytes]] = []
        starts = _body_first_starts(same_page, items, cjk, formulas_map)

        def write_floor(block: TextBlock, rect: pymupdf.Rect) -> float:
            """How low the write box can reach: when writing, _fit_page_margins trims the box at the source body
            bottom."""
            box = pymupdf.Rect(rect.x0, rect.y0, rect.x1, bounds.y1)
            return _fit_page_margins(block, box, same_page, bounds.height, cjk, body_ink(block.page)).y1

        def write_one(
            block: TextBlock,
            text: str,
            rect: pymupdf.Rect,
            align: str,
            formulas: list[dict] | None = None,
            justify_last: bool = False,
            # source: the original block when block is a moved copy (index entry); the reading order marker is on
            # the original block
            x_cap: float | None = None,
            source: TextBlock | None = None,
            # The second half of continued text is written at the previous block's font size (see _Carry)
            block_size: float | None = None,
            # Lowered origin and how far down to reach: see _fit_typeset
            lower: tuple[float, float] | None = None,
            room: float | None = None,
            # The origin was pushed down by the paragraph above, so the first line is no longer raised to match
            # the first source line
            pushed: bool = False,
            # room is lent by the paragraph below (see _fit_typeset)
            lent: bool = False,
        ) -> None:
            rect = _fit_page_margins(block, rect, same_page, bounds.height, cjk, body_ink(block.page))
            written_ids.add(id(source or block))
            before = order.snapshot()
            bformulas = formulas if formulas is not None else (formulas_map or {}).get(id(block), [])
            weight = "bold" if block.bold else "normal"
            block_size = block.size if block_size is None else block_size
            size = block_size * 0.88 if cjk else block_size
            obstacles = [ob.rect for ob in layout if ob is not block]
            # Table of contents page numbers stay on the original page, and the translation must not run over them
            obstacles += [ob.leader.number for ob in same_page if ob.leader is not None]
            obstacles += [ink for ink in kept if _on_rows(ink, block.line_rects or [block.rect])]
            if geo is not None:
                # Ruled tables and algorithm boxes have drawn borders, and extending right must not cover them
                obstacles = [*obstacles, *geo.figures, *(table.rect for table in geo.tables)]
            beside = _beside_rows(obstacles, rect, block.line_rects or [block.rect])
            limit = _write_limit(block, rect, geo, beside)
            if x_cap is not None:
                limit = x_cap if limit is None else min(limit, x_cap)
            # Right-aligned text ends at the original right edge without extending right; when it does not fit it
            # extends left, at most to 10% of the page width or the cell's left edge
            left_limit = None
            if align == "right":
                limit = rect.x1
                left_limit = block.clip[0] + 0.8 if block.clip is not None else bounds.width * 0.1
            # The half of a paragraph carried over from the previous page is written at that page's column width, so
            # figures to avoid may lie outside the block's box
            holes = _wrap_holes(block, obstacles, rect | block.rect)
            # For items whose number or bullet starts at the margin, the body aligns as in the source in a column to
            # the right of the number. The second half of continued text and moved copies are not measured by the
            # original block's lines
            hang = (
                _hang_indent(block)
                if align == "left" and source is None and formulas is None and abs(rect.x0 - block.rect.x0) <= 1
                else 0.0
            )
            first_room = None if pushed else _first_row_room(block, rect)
            # Right-to-left text still goes to an HTML box. Other paragraphs are broken by the typesetter: with
            # formula images they are placed line by line, without them they are written into one box with locked
            # line breaks, so the whole font is not embedded once per line.
            if rtl:
                body = _translation_html(text, bformulas, block_size, size, rect.width)
                css = (
                    f"* {{font-family: sans-serif; font-size: {size}px; color: {block.color}; "
                    f"font-weight: {weight}; line-height: {1.45 if cjk else 1.2}; margin: 0; padding: 0; "
                    f"text-align: {'center' if align == 'center' else 'right'};}}"
                )
                _insert_fitting(page, rect, f"<div{div_attrs}>{body}</div>", css, beside, archive, limit)
            elif any(f.get("has_img") for f in bformulas):
                end = _place_typeset(
                    page,
                    doc,
                    block_size,
                    text,
                    bformulas,
                    rect,
                    size,
                    cjk,
                    align,
                    div_attrs,
                    block.color,
                    weight,
                    beside,
                    limit,
                    forms,
                    page_spans,
                    deferred_png,
                    justify_last,
                    holes,
                    first_room,
                    lower,
                    room,
                    left_limit,
                    lent,
                    hang,
                )
                ends[id(source or block)] = (source or block, end, size)
            else:
                end = _place_plain(
                    page,
                    text,
                    bformulas,
                    rect,
                    size,
                    block_size,
                    cjk,
                    align,
                    div_attrs,
                    block.color,
                    weight,
                    beside,
                    limit,
                    archive,
                    justify_last,
                    holes,
                    first_room,
                    lower,
                    room,
                    left_limit,
                    lent,
                    hang,
                )
                if end is not None:
                    ends[id(source or block)] = (source or block, end, size)
            if block.leader is not None and not page.rotation:
                _extend_leader(page, block, block.leader)
            # Extending right or left and wrapping write outside the box, so use the leftmost and rightmost reach
            left = _expand_left(rect, left_limit, beside).x0 if left_limit is not None else rect.x0
            reach = pymupdf.Rect(left, rect.y0, max(rect.x1, limit or rect.x1), rect.y1) | block.rect
            order.record(source or block, before, reach + (-2, -2, 2, 2))

        # Blocks written on this page: original block, bottom of the last line box, font size
        ends: dict[int, tuple[TextBlock, float, float]] = {}
        # Blocks already written on this page; after the paragraph above reaches down, the highest origin of each
        # paragraph (see spare_below)
        written_ids: set[int] = set()
        claims: dict[int, float] = {}

        def paragraph_room(block: TextBlock, rect: pymupdf.Rect) -> float | None:
            """How far down the paragraph can write (see _room_below). None for vertically adjacent table cells
            (with other cells beside them) and index entries, which each match their original lines"""
            if block.index or _row_neighbor(block, same_page, geo):
                return None
            return _room_below(page, block, rect.y0, rect.x0, rect.x1, same_page, kept, marks, body_ink(block.page))

        def lift_limit(block: TextBlock, rect: pymupdf.Rect, size: float) -> float | None:
            """How far up the paragraph can move at most: half a line below the nearest thing above. Written blocks
            count by their last line; untranslated blocks, glyphs left in place and figure boxes count at their
            original position. None when a block above has not had its turn yet (growing down, it would cover this
            one) or there is nothing above (top of the column)"""
            pending = {id(other) for other, _ in items} - reached
            bottoms = []
            for other in same_page:
                if (
                    other is block
                    or other.rect.y0 > block.rect.y0 - 1
                    or other.rect.x1 < rect.x0 + 4
                    or other.rect.x0 > rect.x1 - 4
                ):
                    continue
                if id(other) in pending:
                    return None
                bottoms.append(ends[id(other)][1] if id(other) in ends else other.rect.y1)
            bottoms += [
                ink.y1
                for ink in (*kept, *marks)
                if ink.y0 < block.rect.y0 - 1 and ink.x1 > rect.x0 + 4 and ink.x0 < rect.x1 - 4
            ]
            if not bottoms:
                return None
            return max(bottoms) + size * (1.45 if cjk else 1.2) * 0.5

        def paragraph_top(block: TextBlock, rect: pymupdf.Rect, room: float | None) -> tuple[float, float] | None:
            """When the translation of the paragraph right above fills its box, its last line touches this
            paragraph's origin and the two read as one paragraph. The translation has no indents, so paragraphs are
            separated only by vertical space: the origin is at least half a line below the last line above.
            When the block above does not end at a sentence end, semicolon or colon, the two are halves of one
            broken sentence and are not separated.
            room is the bottom computed by paragraph_room; without it nothing moves."""
            if room is None:
                return None
            above = [
                (end, size)
                for other, end, size in ends.values()
                if not other.index
                and _closes_clause(other.text)
                and other.rect.y1 <= block.rect.y0 + 2
                and block.rect.y0 - other.rect.y1 < block.size * 1.5
                and min(other.rect.x1, block.rect.x1) - max(other.rect.x0, block.rect.x0) > block.size * 2
                and not _row_neighbor(other, same_page, geo)
            ]
            if not above:
                return None
            end, size = max(above)
            wanted = end + size * (1.45 if cjk else 1.2) * 0.5
            if wanted <= rect.y0:
                return None
            return wanted, room

        # The continuation line may come before the half sentence. The second half goes into carry first so it is
        # not lost when writing.
        shared_head: dict[int, tuple[str, pymupdf.Rect, list[dict], bool]] = {}
        for b, t in items:
            if t == SPILL_TAIL:
                continue
            tail = continuations.get(id(b))
            if tail is None or not _same_page_line_pair(b, tail):
                continue
            bformulas = (formulas_map or {}).get(id(b), [])
            size = b.size * 0.88 if cjk else b.size
            host_box = _own_line_box(b, cjk)
            tail_box = _own_line_box(tail, cjk)
            head, rest = _split_own_lines(
                t,
                bformulas,
                size,
                b.size,
                host_box.width,
            )
            shared_head[id(b)] = (head, host_box, bformulas, bool(rest.strip()))
            carried[id(tail)] = _Carry(rest, bformulas, tail_box.width, b.size, True)
        spare_below, next_movable, latest_start, movable_box = page_flow(
            pno,
            items,
            kept,
            marks,
            written_ids,
            stacked,
            reflowed,
            shared_head,
        )
        # For a movable paragraph: the paragraph right above that it follows (see next_movable), and its own box
        follows: dict[int, tuple[TextBlock, pymupdf.Rect]] = {}
        # Blocks that have had their turn. A tail whose translation moved entirely to its host, or a host written
        # entirely on the continuation page, writes nothing when its turn comes
        reached: set[int] = set()

        def spill_last_lines(
            block: TextBlock,
            text: str,
            rect: pymupdf.Rect,
            align: str,
            room: float,
            pushed: bool,
        ) -> bool:
            """This paragraph is the last body paragraph on the page, does not fit at normal line spacing, and has no
            free space above or below: the lines staying on this page are written at normal line spacing, and the
            rest continues at the top of the next page (see spill_target). Not done when the remaining lines contain
            inline formulas, because formula images are only pasted on this page"""
            index = next(i for i, (other, _) in enumerate(items) if other is block)
            if any(
                t != SPILL_TAIL and not (other.body_size and abs(other.size - other.body_size) > 0.5)
                for other, t in items[index + 1 :]
            ) or any(
                other.rect.y0 > block.rect.y0 + 1
                and min(other.rect.x1, rect.x1) - max(other.rect.x0, rect.x0) > 8
                and not (other.body_size and abs(other.size - other.body_size) > 0.5)
                for other in same_page
                if other is not block
            ):
                return False
            size = block.size * 0.88 if cjk else block.size
            target = spill_target(block, rect.width, size)
            if target is None:
                return False
            head_block, box, next_room = target
            bformulas = (formulas_map or {}).get(id(block), [])
            floor = write_floor(block, rect)
            head, rest = _split_filled_lines(
                text,
                bformulas,
                size,
                block.size,
                rect.width,
                cjk,
                rect.y0,
                room,
                floor,
                next_room,
            )
            gap = 1.45 if cjk else 1.2
            lines = head.count("\n") + 1
            if (
                not head
                or not rest
                or "\x01" in rest
                or rect.y0 + lines * size * gap + size * 0.3 > min(room + (gap - 1) * size, floor) + 1
                or (rest.count("\n") + 1) * size * gap + size * 0.3 > next_room + (gap - 1) * size + 1
            ):
                return False
            bottom = _filled_bottom(rect.y0, lines, size, cjk, room, floor)
            write_one(
                block,
                head,
                pymupdf.Rect(rect.x0, rect.y0, rect.x1, max(bottom, rect.y0 + 4)),
                align,
                bformulas,
                True,
                pushed=pushed,
            )
            spilled[id(head_block)] = (box, _Carry(rest, bformulas, rect.width, block.size))
            return True

        for b, t in items:
            reached.add(id(b))
            if t == SPILL_TAIL:
                carry = carried.get(id(b))
                if carry is None or not carry.text.strip():
                    continue
                if carry.own_line:
                    box = _own_line_box(b, cjk)
                    write_one(b, carry.text, box, "left", carry.formulas, False, box.x1, block_size=carry.size)
                    continue
                rect, align = _write_rect(b, visible or [(b, carry.text)], cjk, rtl, geo)
                room = _room_below(
                    page, b, rect.y0, rect.x0, rect.x0 + carry.width, same_page, kept, marks, body_ink(b.page)
                )
                # The line spacing below the last line is blank, so the box may extend past room while the glyphs
                # stay within room (as in _filled_bottom)
                size = carry.size * 0.88 if cjk else carry.size
                bottom = room + ((1.45 if cjk else 1.2) - 1) * size
                placed = pymupdf.Rect(rect.x0, rect.y0, rect.x0 + carry.width, max(bottom, rect.y0 + 4))
                # The previous page counted, via tail_room, on borrowing from the paragraph below; when the text does
                # not fit here it reaches into that space, and that paragraph moves down
                borrowed = spare_below(b, placed, room)
                write_one(
                    b,
                    carry.text,
                    placed,
                    align,
                    carry.formulas,
                    False,
                    block_size=carry.size,
                    room=None if borrowed is None else borrowed[1],
                    lent=True,
                )
                if borrowed is not None and id(b) in ends:
                    _, end, size = ends[id(b)]
                    if end + size * 0.3 > placed.y1 + 0.5:
                        claims[id(borrowed[0])] = end + size * (1.45 if cjk else 1.2) * 0.5
                continue
            prepared = shared_head.get(id(b))
            if prepared is not None:
                head, host_box, bformulas, justify = prepared
                if head:
                    write_one(b, head, host_box, "left", bformulas, justify, host_box.x1)
                continue
            # Reference column width: the widest block on the page for one column, the block's own column for
            # several. Centered blocks use the full column width, so short translations are not squeezed by a small
            # box into lines with single characters. The typesetter breaks lines at this width.
            rect, align = _write_rect(b, visible, cjk, rtl, geo)
            inset = _inset_write_boxes(b, cjk)
            if inset is not None and id(b) not in stacked:
                first_box, rest_box = inset
                # A right-aligned epigraph with attribution has a short first line starting further right: it is
                # still broken into two boxes, and every line ends at the original right edge
                side = "right" if align == "right" else "left"
                if side == "right":
                    first_box = pymupdf.Rect(first_box.x0, first_box.y0, rect.x1, first_box.y1)
                    rest_box = pymupdf.Rect(rest_box.x0, rest_box.y0, rect.x1, rest_box.y1)
                moved = starts.get(id(b))
                if moved is not None:
                    first_box = _clip_write_rect(
                        pymupdf.Rect(moved, first_box.y0, first_box.x1, first_box.y1),
                        b.clip,
                    )
                bformulas = (formulas_map or {}).get(id(b), [])
                size = b.size * 0.88 if cjk else b.size
                head, rest = _split_own_lines(
                    t,
                    bformulas,
                    size,
                    b.size,
                    first_box.width,
                )
                if head:
                    write_one(b, head, first_box, side, bformulas, bool(rest.strip()), first_box.x1)
                tail = continuations.get(id(b))
                if rest.strip() and tail is None:
                    write_one(b, rest, rest_box, side, bformulas, False, rest_box.x1)
                elif rest.strip() and tail is not None:
                    room = _room_below(
                        page, b, rest_box.y0, rest_box.x0, rest_box.x1, same_page, kept, marks, body_ink(b.page)
                    )
                    floor = write_floor(b, rest_box)
                    placed, spill = _split_filled_lines(
                        rest,
                        bformulas,
                        size,
                        b.size,
                        rest_box.width,
                        cjk,
                        rest_box.y0,
                        room,
                        floor,
                        tail_room(tail, rest_box.width, size),
                    )
                    if placed:
                        lines = placed.count("\n") + 1
                        bottom = _filled_bottom(rest_box.y0, lines, size, cjk, room, floor)
                        placed_rect = pymupdf.Rect(
                            rest_box.x0,
                            rest_box.y0,
                            rest_box.x1,
                            max(bottom, rest_box.y0 + 4),
                        )
                        write_one(b, placed, placed_rect, side, bformulas, bool(spill), rest_box.x1)
                    if spill:
                        carried[id(tail)] = _Carry(spill, bformulas, rest_box.width, b.size)
                continue
            if id(b) in stacked:
                rect, align = stacked[id(b)], "left"
            shifted = reflowed.get(id(b))
            if shifted is not None:
                rect = pymupdf.Rect(
                    rect.x0,
                    rect.y0 + shifted.rect.y0 - b.rect.y0,
                    rect.x1,
                    rect.y1 + shifted.rect.y1 - b.rect.y1,
                )
                write_one(shifted, t, rect, align, (formulas_map or {}).get(id(b), []), source=b)
                continue
            if b.side_note and geo is not None and geo.frame is not None and b.rect.x1 <= geo.frame.x0:
                # A margin note on the left is right-aligned against the body. When it does not fit on one line it
                # starts from the left as usual
                width = measure_width(t, b.size * 0.88 if cjk else b.size)
                if width < rect.width:
                    rect = pymupdf.Rect(b.rect.x1 - width - 1, rect.y0, b.rect.x1 + 1, rect.y1)
                write_one(b, t, rect, "left", None, False, rect.x1)
                continue
            beside = _beside_body(b, same_page)
            if beside is not None and id(b) not in stacked:
                cap = beside.line_rects[0].x0 - 2
                moved = starts.get(id(beside))
                if moved is not None:
                    cap = min(cap, moved - 2)
                rect = pymupdf.Rect(rect.x0, rect.y0, min(rect.x1, max(cap, rect.x0 + 4)), rect.y1)
                write_one(b, t, rect, "left", None, False, rect.x1)
                continue
            tail = continuations.get(id(b))
            if tail is None:
                edge = _indent_edge(b, geo, indent) if align == "left" and id(b) not in stacked else None
                if edge is not None:
                    rect = pymupdf.Rect(edge, rect.y0, rect.x1, rect.y1)
                carry = spilled.pop(id(b), None)
                if carry is not None:
                    # The last lines of the paragraph at the bottom of the previous page come before this paragraph,
                    # which starts half a line below them
                    box, carry = carry
                    size = carry.size * 0.88 if cjk else carry.size
                    lines = carry.text.count("\n") + 1
                    placed = pymupdf.Rect(
                        box.x0,
                        box.y0,
                        box.x0 + carry.width,
                        box.y0 + lines * size * (1.45 if cjk else 1.2) + size * 0.3,
                    )
                    write_one(b, carry.text, placed, "left", [], False, block_size=carry.size)
                    if id(b) in ends:
                        claims[id(b)] = ends[id(b)][1] + size * (1.45 if cjk else 1.2) * 0.5
                claim = claims.get(id(b))
                pushed = claim is not None and claim > rect.y0
                # Blocks below are looked up from this paragraph's original position. The lowered origin may already
                # be below the top of the next paragraph, and searching down from there would miss it
                own = rect
                if pushed:
                    rect = pymupdf.Rect(rect.x0, claim, rect.x1, rect.y1)
                # This paragraph and the run below it do not fit at normal line spacing and there is space above:
                # move up, at most to half a line below the nearest thing above. The paragraph above was already
                # moved up the same way when written, so the free space above accumulates paragraph by paragraph
                link = follows.get(id(b))
                box = link[1] if link is not None else movable_box(b)
                if box is not None:
                    latest = latest_start(b, box)
                    up = lift_limit(b, rect, b.size * 0.88 if cjk else b.size) if latest < rect.y0 - 0.5 else None
                    if up is not None and max(up, latest) < rect.y0 - 0.5:
                        rect = pymupdf.Rect(rect.x0, max(up, latest), rect.x1, rect.y1)
                        pushed = True
                room = paragraph_room(b, own) if align == "left" and id(b) not in stacked else None
                if room is not None:
                    below = next_movable(b, own, room)
                    if below is not None:
                        follows[id(below[0])] = (b, below[1])
                # Text in a cell does not reach into the next cell even when it does not fit
                grow = room if b.clip is None else None
                borrowed = spare_below(b, own, grow) if grow is not None else None
                if borrowed is not None:
                    grow = max(grow, borrowed[1])
                if box is not None and grow is not None and borrowed is None and latest < rect.y0 - 0.5:
                    if spill_last_lines(b, t, rect, align, grow, pushed):
                        continue
                write_one(
                    b,
                    t,
                    rect,
                    align,
                    lower=paragraph_top(b, rect, room),
                    room=grow,
                    pushed=pushed,
                    lent=borrowed is not None,
                )
                if borrowed is not None and id(b) in ends:
                    _, end, size = ends[id(b)]
                    # The box reached past the original bottom (the box is 0.3em taller than the last line box) into
                    # the box of the paragraph below: that paragraph's origin moves to half a line below the last line
                    if end + size * 0.3 > rect.y1 + 0.5:
                        claims[id(borrowed[0])] = end + size * (1.45 if cjk else 1.2) * 0.5
                continue
            bformulas = (formulas_map or {}).get(id(b), [])
            # A half sentence at the end of a line is only a few words wide. Breaking at its width would squeeze the
            # continuation page too. The source is erased and the whole paragraph is written on the continuation
            # page at that page's column width.
            if len(b.line_rects) <= 1 and rect.width < tail.rect.width * 0.7:
                carried[id(tail)] = _Carry(t, bformulas, tail.rect.width, b.size)
                continue
            left = _indented_start(b, tail, geometries)
            if left is not None:
                rect = pymupdf.Rect(left, rect.y0, rect.x1, rect.y1)
            size = b.size * 0.88 if cjk else b.size
            room = _room_below(page, b, rect.y0, rect.x0, rect.x1, same_page, kept, marks, body_ink(b.page))
            floor = write_floor(b, rect)
            next_room = tail_room(tail, rect.width, size)
            # The paragraph above borrowed the space freed by this paragraph moving down (see host_latest)
            claim = claims.get(id(b))
            if claim is not None and claim > rect.y0:
                rect = pymupdf.Rect(rect.x0, claim, rect.x1, rect.y1)

            def split(y0: float) -> tuple[str, str]:
                return _split_filled_lines(t, bformulas, size, b.size, rect.width, cjk, y0, room, floor, next_room)

            def squeezed(head: str, y0: float) -> bool:
                """The lines staying on this page do not fit at normal line spacing (lines counted as in
                _split_filled_lines)"""
                gap = 1.45 if cjk else 1.2
                lines = head.count("\n") + 1
                return bool(head) and y0 + lines * size * gap + size * 0.3 > min(room + (gap - 1) * size, floor) + 1

            head, rest = split(rect.y0)
            # Does not fit at normal line spacing and there is space above: move up. The higher it moves, the more
            # lines this page holds, so find the lowest origin at which they fit
            up = lift_limit(b, rect, size) if squeezed(head, rect.y0) else None
            if up is not None and up < rect.y0 - 0.5 and not squeezed(split(up)[0], up):
                low, high = up, rect.y0
                for _ in range(8):
                    middle = (low + high) / 2
                    if squeezed(split(middle)[0], middle):
                        high = middle
                    else:
                        low = middle
                rect = pymupdf.Rect(rect.x0, low, rect.x1, rect.y1)
                head, rest = split(low)
            elif up is not None and up < rect.y0 - 0.5:
                # The extra lines gained by moving up are still squeezed (the next page cannot take them): keep the
                # line count and move up to where these lines fit at normal line spacing
                gap = 1.45 if cjk else 1.2
                lines = head.count("\n") + 1
                start = min(room + (gap - 1) * size, floor) - lines * size * gap - size * 0.3
                if up <= start < rect.y0 - 0.5:
                    rect = pymupdf.Rect(rect.x0, start, rect.x1, rect.y1)
            if head:
                lines = head.count("\n") + 1
                bottom = _filled_bottom(rect.y0, lines, size, cjk, room, floor)
                head_rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, max(bottom, rect.y0 + 4))
                write_one(b, head, head_rect, align, bformulas, bool(rest))
            if rest:
                carried[id(tail)] = _Carry(rest, bformulas, rect.width, b.size)
        _flush_png(page, deferred_png)
        if mid_map:
            _place_formula_images(page, doc, images_before, mid_map, forms, page_spans)
            restore_glyphs(doc, glyphs)
        order.assemble()
    return doc
