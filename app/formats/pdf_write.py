"""Erase the original text of each page and write the translations in its place."""

import html
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .pdf_blocks import (
    SPILL_TAIL,
    TextBlock,
    _attach_ink,
    _formula_parts,
    _same_page_line_pair,
    _span_is_math,
    _with_source_bold,
)
from .pdf_drawings import text_page_rect
from .pdf_erase import Erasure, erase, restore_glyphs, type3_glyphs
from .pdf_flow import measure_width
from .pdf_formula_images import _place_formula_images
from .pdf_index_reflow import _index_reflow
from .pdf_layout import PageGeometry
from .pdf_order import ReadingOrder
from .pdf_place import (
    _expand_left,
    _insert_fitting,
    _place_plain,
    _place_typeset,
)
from .pdf_place_formulas import _FormulaSources
from .pdf_type3 import broken_type3_names
from .pdf_types import PdfDict
from .pdf_write_flow import _last_fitting, _off_body_size, _PageBreak, _PageFlow, _SharedHead, _WriteContext
from .pdf_write_geometry import (
    _affiliation_write_rects,
    _attach_chars,
    _beside_body,
    _beside_rows,
    _body_first_starts,
    _Carry,
    _clip_write_rect,
    _closes_clause,
    _drawn_marks,
    _extend_leader,
    _filled_bottom,
    _first_row_room,
    _fit_page_margins,
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
    formulas_map: dict[int, list[PdfDict]] | None = None,
    archive: pymupdf.Archive | None = None,
    geometries: list[PageGeometry] | None = None,
    continuations: dict[int, TextBlock] | None = None,
    on_page: Callable[[int, int], None] | None = None,
    pages: Collection[int] | None = None,
) -> pymupdf.Document:
    """on_page(pages written, pages to write) is called before each page is written. When pages is given, only
    those pages are written."""
    doc = pymupdf.open(src_path)
    ctx = _WriteContext(doc, blocks, translations, target_lang, formulas_map, geometries, continuations, pages)
    _DocumentWriter(ctx, target_lang, archive).write(on_page)
    return doc


class _DocumentWriter:
    """Writes the pages of a document one after another, and holds what a page passes on to the pages after it:
    the text carried over to a tail block, and the last lines spilled in front of the next page's first
    paragraph."""

    def __init__(self, ctx: _WriteContext, target_lang: str, archive: pymupdf.Archive | None) -> None:
        self.ctx = ctx
        self.archive = archive
        # lang lets MuPDF choose the right glyphs for Chinese, Japanese and Korean; dir=rtl sets Arabic and Hebrew
        # right to left
        self.div_attrs = f' lang="{html.escape(target_lang)}"' if target_lang else ""
        if ctx.rtl:
            self.div_attrs += ' dir="rtl" style="text-align: right"'
        # Formulas of a paragraph carried over to the next page are recorded under the block on the previous page;
        # when the text continues on this page, the fraction bars in place must be erased according to the tail
        self.spill_hosts = {id(tail): host for host, tail in ctx.continuations.items()}
        # Text a paragraph carries over to its tail block, keyed by the tail
        self.carried: dict[int, _Carry] = {}
        # Last lines of a paragraph at the bottom of a page written on the next page, keyed by the first paragraph
        # of that page
        self.spilled: dict[int, tuple[pymupdf.Rect, _Carry]] = {}

    def write(self, on_page: Callable[[int, int], None] | None) -> None:
        """Write every page that has translations, calling on_page(pages written, pages to write) before each."""
        for written, (pno, items) in enumerate(self.ctx.by_page.items()):
            if on_page is not None:
                on_page(written, len(self.ctx.by_page))
            _PageWriter(self, pno, items).write()


@dataclass(frozen=True)
class _Placement:
    """Where and how write_one typesets one translation: the box, the alignment, the font size written and the
    source block's, and how far the text may extend right, left and down (see _place_typeset)."""

    rect: pymupdf.Rect
    align: str
    size: float
    block_size: float
    beside: list[pymupdf.Rect]
    limit: float | None
    left_limit: float | None
    holes: list[pymupdf.Rect]
    first_room: float | None
    hang: float
    justify_last: bool
    lower: tuple[float, float] | None
    room: float | None
    lent: bool


class _PageWriter:
    """Writes the translations of one page. Constructing it prepares the page: collects the spans formulas are
    clipped from, moves index entries, erases the source text and reads what ink is left; write() then writes the
    blocks in order. Each block's translation may need more room than its source, so writing a block can move the
    paragraphs below it (claims, follows) or carry its last lines to the next page (see _PageFlow)."""

    def __init__(self, writer: _DocumentWriter, pno: int, items: list[tuple[TextBlock, str]]) -> None:
        ctx = writer.ctx
        self.writer = writer
        self.ctx = ctx
        self.doc = ctx.doc
        self.cjk = ctx.cjk
        self.pno = pno
        self.page = ctx.doc[pno]
        self.same_page = ctx.page_blocks.get(pno, [])
        # page.rect of a rotated page is rotated; comparing with text coordinates needs the unrotated page box
        self.bounds = text_page_rect(self.page)
        items = _hosts_first(items, ctx.continuations)
        self.visible = [(b, t) for b, t in items if t != SPILL_TAIL]
        self.geo = ctx.geometry(pno)
        self.mid_map: dict[int, PdfDict] = {
            f["mid"]: f for b, _ in self.visible for f in ctx.formulas.get(id(b), []) if "mid" in f
        }
        self.page_spans, self.forms, self.glyphs = self._formula_sources()
        self.stacked = _affiliation_write_rects(self.same_page)
        self.reflowed = self._reflow_index(items)
        # Other blocks extending right or wrapping avoid the moved index entries at their new positions
        self.layout = [self.reflowed.get(id(ob), ob) for ob in self.same_page]
        item_ids = {id(b) for b, _ in items}
        self.items = [
            *items,
            *(
                (b, _with_source_bold(b, _source_marked(b)))
                for b in self.same_page
                if id(b) in self.reflowed and id(b) not in item_ids
            ),
        ]
        erasure = self._erasure()
        # Before erasing, place markers at the source positions; after writing, the translation is inserted back
        # there, so drag selection and copying follow the reading order
        self.order = ReadingOrder(self.page, [b for b, _ in self.items])
        erase(self.page, erasure)
        self.order.after_erase()
        # Glyphs still in place after erasing (dropped display formulas, equation numbers) must be avoided when the
        # translation extends right or continues
        self.kept = _kept_ink(self.page, self.same_page)
        # Continued text reaching down must also avoid figure boxes and images, which are not in blocks. Read
        # before writing the translation: the formulas written are drawn too
        spills = any(t == SPILL_TAIL or id(b) in ctx.continuations for b, t in self.items)
        self.marks = _drawn_marks(self.page) if spills else []
        ctx.page_inks[pno] = self.kept, self.marks
        self._make_resources_indirect()
        self.images_before = {img[0] for img in self.page.get_images(full=True)}
        self.deferred_png: list[tuple[pymupdf.Rect, bytes]] = []
        self.starts = _body_first_starts(self.same_page, self.items, self.cjk, ctx.formulas)
        # Blocks written on this page: original block, bottom of the last line box, font size
        self.ends: dict[int, tuple[TextBlock, float, float]] = {}
        # Blocks already written on this page; after the paragraph above reaches down, the highest origin of each
        # paragraph (see spare_below)
        self.written_ids: set[int] = set()
        self.claims: dict[int, float] = {}
        self.shared_head = self._share_heads()
        self.flow = _PageFlow(
            ctx,
            pno,
            self.items,
            self.kept,
            self.marks,
            self.written_ids,
            self.stacked,
            self.reflowed,
            self.shared_head,
        )
        # For a movable paragraph: the paragraph right above that it follows (see next_movable), and its own box
        self.follows: dict[int, tuple[TextBlock, pymupdf.Rect]] = {}
        # Blocks that have had their turn. A tail whose translation moved entirely to its host, or a host written
        # entirely on the continuation page, writes nothing when its turn comes
        self.reached: set[int] = set()

    # Preparing the page

    def _formula_sources(self) -> tuple[list[PdfDict] | None, _FormulaSources | None, dict[int, bytes]]:
        """What formula images on this page are cut from, read before the page is erased: the spans of the page,
        the page packed into a Form, and the bitmap glyphs to restore afterwards. Nothing when the page has no
        formulas."""
        doc, page, mid_map = self.doc, self.page, self.mid_map
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
        return page_spans, forms, glyphs

    def _reflow_index(self, items: list[tuple[TextBlock, str]]) -> dict[int, TextBlock]:
        """Index entries moved down to make room for translations taking more lines (see _index_reflow)."""
        ctx = self.ctx
        fixed = {*self.stacked, *ctx.continuations, *(id(tail) for tail in ctx.continuations.values())}
        return _index_reflow(
            items,
            self.same_page,
            self.geo,
            self.cjk,
            ctx.formulas,
            fixed,
            _page_content_bottom(self.same_page, self.pno, self.bounds.height),
            _kept_ink(self.page, self.same_page) if any(b.index for b in self.same_page) else (),
        )

    def _erasure(self) -> Erasure:
        """What to erase from the page: the source text of every block written, and what formula images take
        along."""
        formulas, spill_hosts, pno = self.ctx.formulas, self.writer.spill_hosts, self.pno
        erasure = Erasure(math=_span_is_math)
        for b, _ in self.items:
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
                if key is not None
                for f in formulas.get(key, [])
                for part in _formula_parts(f)
                for curve in part.get("curves") or []
                if curve is not None and _part_page(part, f, pno) == pno
            )
            # A formula image is cut out by its whole box and pasted at the new position, so every glyph whose
            # origin is in the box must be erased in place. Glyphs without Unicode (the large norm bars of CMEX
            # extract as \r) are treated as blank and are not in any span, so erasing by span misses them
            erasure.spanless.extend(
                pymupdf.Rect(part["bbox"])
                for key in (id(b), spill_hosts.get(id(b)))
                if key is not None
                for f in formulas.get(key, [])
                for part in _formula_parts(f)
                if part.get("bbox") is not None and _part_page(part, f, pno) == pno
            )
        return erasure

    def _make_resources_indirect(self) -> None:
        """Make the page's /Resources an indirect object."""
        doc, page = self.doc, self.page
        # Rewriting the content stream turns /Resources into an inline dictionary; after the typesetter writes,
        # /XObject has hundreds of entries, and such a large nested inline dictionary gets corrupted by the GC on
        # save (garbage≥3) (the entry of the first form points to a font stream and content is silently lost).
        # Making it an indirect reference avoids this MuPDF bug.
        res_v = doc.xref_get_key(page.xref, "Resources")[1]
        if res_v.startswith("<<"):
            res_x = doc.get_new_xref()
            doc.update_object(res_x, res_v)
            doc.xref_set_key(page.xref, "Resources", f"{res_x} 0 R")

    def _share_heads(self) -> dict[int, _SharedHead]:
        """Split each half sentence sharing a baseline with its continuation into the part on its own line and the
        part carried to the continuation."""
        ctx = self.ctx
        # The continuation line may come before the half sentence. The second half goes into carry first so it is
        # not lost when writing.
        shared_head: dict[int, _SharedHead] = {}
        for b, t in self.items:
            if t == SPILL_TAIL:
                continue
            tail = ctx.continuations.get(id(b))
            if tail is None or not _same_page_line_pair(b, tail):
                continue
            bformulas = ctx.formulas.get(id(b), [])
            size = ctx.em(b.size)
            host_box = _own_line_box(b, self.cjk)
            tail_box = _own_line_box(tail, self.cjk)
            head, rest = _split_own_lines(
                t,
                bformulas,
                size,
                b.size,
                host_box.width,
            )
            shared_head[id(b)] = (head, host_box, bformulas, bool(rest.strip()))
            self.writer.carried[id(tail)] = _Carry(rest, bformulas, tail_box.width, b.size, True)
        return shared_head

    # Writing one translation

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

    def write_floor(self, block: TextBlock, rect: pymupdf.Rect) -> float:
        """How low the write box can reach: when writing, _fit_page_margins trims the box at the source body
        bottom."""
        box = pymupdf.Rect(rect.x0, rect.y0, rect.x1, self.bounds.y1)
        return _fit_page_margins(
            block, box, self.same_page, self.bounds.height, self.cjk, self.ctx.body_ink(block.page)
        ).y1

    def write_one(
        self,
        block: TextBlock,
        text: str,
        rect: pymupdf.Rect,
        align: str,
        formulas: list[PdfDict] | None = None,
        justify_last: bool = False,
        # source: the original block when block is a moved copy (index entry); the reading order marker is on the
        # original block
        x_cap: float | None = None,
        source: TextBlock | None = None,
        # The second half of continued text is written at the previous block's font size (see _Carry)
        block_size: float | None = None,
        # Lowered origin and how far down to reach: see _fit_typeset
        lower: tuple[float, float] | None = None,
        room: float | None = None,
        # The origin was pushed down by the paragraph above, so the first line is no longer raised to match the
        # first source line
        pushed: bool = False,
        # room is lent by the paragraph below (see _fit_typeset)
        lent: bool = False,
    ) -> None:
        """Write text, the translation of block, into rect: fit the box to the page margins, work out how far the
        text may extend right, left and down, typeset it, and record where it ended (ends) and the area its reading
        order marker covers."""
        ctx = self.ctx
        rect = _fit_page_margins(block, rect, self.same_page, self.bounds.height, self.cjk, ctx.body_ink(block.page))
        self.written_ids.add(id(source or block))
        before = self.order.snapshot()
        bformulas = formulas if formulas is not None else ctx.formulas.get(id(block), [])
        block_size = block.size if block_size is None else block_size
        size = ctx.em(block_size)
        obstacles = self._obstacles(block)
        beside = _beside_rows(obstacles, rect, block.line_rects or [block.rect])
        limit = _write_limit(block, rect, self.geo, beside)
        if x_cap is not None:
            limit = x_cap if limit is None else min(limit, x_cap)
        # Right-aligned text ends at the original right edge without extending right; when it does not fit it
        # extends left, at most to 10% of the page width or the cell's left edge
        left_limit = None
        if align == "right":
            limit = rect.x1
            left_limit = block.clip[0] + 0.8 if block.clip is not None else self.bounds.width * 0.1
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
        placement = _Placement(
            rect,
            align,
            size,
            block_size,
            beside,
            limit,
            left_limit,
            holes,
            first_room,
            hang,
            justify_last,
            lower,
            room,
            lent,
        )
        end = self._typeset(block, text, bformulas, placement)
        if end is not None:
            self.ends[id(source or block)] = (source or block, end, size)
        if block.leader is not None and not self.page.rotation:
            _extend_leader(self.page, block, block.leader)
        # Extending right or left and wrapping write outside the box, so use the leftmost and rightmost reach
        left = _expand_left(rect, left_limit, beside).x0 if left_limit is not None else rect.x0
        reach = pymupdf.Rect(left, rect.y0, max(rect.x1, limit or rect.x1), rect.y1) | block.rect
        self.order.record(source or block, before, reach + (-2, -2, 2, 2))

    def _obstacles(self, block: TextBlock) -> list[pymupdf.Rect]:
        """What the translation of block must not run over when it extends right or wraps."""
        obstacles = [ob.rect for ob in self.layout if ob is not block]
        # Table of contents page numbers stay on the original page, and the translation must not run over them
        obstacles += [ob.leader.number for ob in self.same_page if ob.leader is not None]
        obstacles += [ink for ink in self.kept if _on_rows(ink, block.line_rects or [block.rect])]
        if self.geo is not None:
            # Ruled tables and algorithm boxes have drawn borders, and extending right must not cover them
            obstacles = [*obstacles, *self.geo.figures, *(table.rect for table in self.geo.tables)]
        return obstacles

    def _typeset(self, block: TextBlock, text: str, formulas: list[PdfDict], at: _Placement) -> float | None:
        """Draw text on the page as placed by at. Returns the bottom of the last line box, or None when it is not
        known (right-to-left text, or text the typesetter did not write)."""
        cjk, page, div_attrs = self.cjk, self.page, self.writer.div_attrs
        weight = "bold" if block.bold else "normal"
        # Right-to-left text still goes to an HTML box. Other paragraphs are broken by the typesetter: with formula
        # images they are placed line by line, without them they are written into one box with locked line breaks,
        # so the whole font is not embedded once per line.
        if self.ctx.rtl:
            body = _translation_html(text, formulas, at.block_size, at.size, at.rect.width)
            css = (
                f"* {{font-family: sans-serif; font-size: {at.size}px; color: {block.color}; "
                f"font-weight: {weight}; line-height: {1.45 if cjk else 1.2}; margin: 0; padding: 0; "
                f"text-align: {'center' if at.align == 'center' else 'right'};}}"
            )
            _insert_fitting(
                page, at.rect, f"<div{div_attrs}>{body}</div>", css, at.beside, self.writer.archive, at.limit
            )
            return None
        if any(f.get("has_img") for f in formulas):
            return _place_typeset(
                page,
                self.doc,
                at.block_size,
                text,
                formulas,
                at.rect,
                at.size,
                cjk,
                at.align,
                div_attrs,
                block.color,
                weight,
                at.beside,
                at.limit,
                self.forms,
                self.page_spans,
                self.deferred_png,
                at.justify_last,
                at.holes,
                at.first_room,
                at.lower,
                at.room,
                at.left_limit,
                at.lent,
                at.hang,
            )
        return _place_plain(
            page,
            text,
            formulas,
            at.rect,
            at.size,
            at.block_size,
            cjk,
            at.align,
            div_attrs,
            block.color,
            weight,
            at.beside,
            at.limit,
            self.writer.archive,
            at.justify_last,
            at.holes,
            at.first_room,
            at.lower,
            at.room,
            at.left_limit,
            at.lent,
            at.hang,
        )

    # Where a paragraph can write

    def paragraph_room(self, block: TextBlock, rect: pymupdf.Rect) -> float | None:
        """How far down the paragraph can write (see _room_below). None for vertically adjacent table cells (with
        other cells beside them) and index entries, which each match their original lines"""
        if block.index or _row_neighbor(block, self.same_page, self.geo):
            return None
        return self._room(block, rect)

    def lift_limit(self, block: TextBlock, rect: pymupdf.Rect, size: float) -> float | None:
        """How far up the paragraph can move at most: half a line below the nearest thing above. Written blocks count
        by their last line; untranslated blocks, glyphs left in place and figure boxes count at their original
        position. None when a block above has not had its turn yet (growing down, it would cover this one) or there
        is nothing above (top of the column)"""
        pending = {id(other) for other, _ in self.items} - self.reached
        bottoms = []
        for other in self.same_page:
            if (
                other is block
                or other.rect.y0 > block.rect.y0 - 1
                or other.rect.x1 < rect.x0 + 4
                or other.rect.x0 > rect.x1 - 4
            ):
                continue
            if id(other) in pending:
                return None
            bottoms.append(self.ends[id(other)][1] if id(other) in self.ends else other.rect.y1)
        bottoms += [
            ink.y1
            for ink in (*self.kept, *self.marks)
            if ink.y0 < block.rect.y0 - 1 and ink.x1 > rect.x0 + 4 and ink.x0 < rect.x1 - 4
        ]
        if not bottoms:
            return None
        return max(bottoms) + size * self.ctx.gap * 0.5

    def paragraph_top(self, block: TextBlock, rect: pymupdf.Rect, room: float | None) -> tuple[float, float] | None:
        """When the translation of the paragraph right above fills its box, its last line touches this paragraph's
        origin and the two read as one paragraph. The translation has no indents, so paragraphs are separated only
        by vertical space: the origin is at least half a line below the last line above.
        When the block above does not end at a sentence end, semicolon or colon, the two are halves of one broken
        sentence and are not separated.
        room is the bottom computed by paragraph_room; without it nothing moves."""
        if room is None:
            return None
        above = [
            (end, size)
            for other, end, size in self.ends.values()
            if not other.index
            and _closes_clause(other.text)
            and other.rect.y1 <= block.rect.y0 + 2
            and block.rect.y0 - other.rect.y1 < block.size * 1.5
            and min(other.rect.x1, block.rect.x1) - max(other.rect.x0, block.rect.x0) > block.size * 2
            and not _row_neighbor(other, self.same_page, self.geo)
        ]
        if not above:
            return None
        end, size = max(above)
        wanted = end + size * self.ctx.gap * 0.5
        if wanted <= rect.y0:
            return None
        return wanted, room

    def _push_below(self, block: TextBlock, borrowed: tuple[TextBlock, float] | None, bottom: float) -> None:
        """After block borrowed space from the paragraph below (see spare_below), move that paragraph's origin down
        when the translation reached into it."""
        if borrowed is not None and id(block) in self.ends:
            _, end, size = self.ends[id(block)]
            # The box reached past the original bottom (the box is 0.3em taller than the last line box) into the box
            # of the paragraph below: that paragraph's origin moves to half a line below the last line
            if end + size * 0.3 > bottom + 0.5:
                self.claims[id(borrowed[0])] = end + size * self.ctx.gap * 0.5

    # Writing the blocks in order

    def write(self) -> None:
        """Write every block of the page in order, then paste formula images and the reading order."""
        for b, t in self.items:
            self.reached.add(id(b))
            self._write_item(b, t)
        self._finish()

    def _write_item(self, b: TextBlock, t: str) -> None:
        """Write the translation t of block b the way its kind of block is written."""
        ctx = self.ctx
        if t == SPILL_TAIL:
            self._write_tail(b)
            return
        prepared = self.shared_head.get(id(b))
        if prepared is not None:
            head, host_box, bformulas, justify = prepared
            if head:
                self.write_one(b, head, host_box, "left", bformulas, justify, host_box.x1)
            return
        # Reference column width: the widest block on the page for one column, the block's own column for
        # several. Centered blocks use the full column width, so short translations are not squeezed by a small
        # box into lines with single characters. The typesetter breaks lines at this width.
        rect, align = _write_rect(b, self.visible, self.cjk, ctx.rtl, self.geo)
        inset = _inset_write_boxes(b, self.cjk)
        if inset is not None and id(b) not in self.stacked:
            self._write_inset(b, t, rect, align, inset)
            return
        if id(b) in self.stacked:
            rect, align = self.stacked[id(b)], "left"
        shifted = self.reflowed.get(id(b))
        if shifted is not None:
            rect = pymupdf.Rect(
                rect.x0,
                rect.y0 + shifted.rect.y0 - b.rect.y0,
                rect.x1,
                rect.y1 + shifted.rect.y1 - b.rect.y1,
            )
            self.write_one(shifted, t, rect, align, ctx.formulas.get(id(b), []), source=b)
            return
        if b.side_note and self.geo is not None and self.geo.frame is not None and b.rect.x1 <= self.geo.frame.x0:
            # A margin note on the left is right-aligned against the body. When it does not fit on one line it
            # starts from the left as usual
            width = measure_width(t, ctx.em(b.size))
            if width < rect.width:
                rect = pymupdf.Rect(b.rect.x1 - width - 1, rect.y0, b.rect.x1 + 1, rect.y1)
            self.write_one(b, t, rect, "left", None, False, rect.x1)
            return
        beside = _beside_body(b, self.same_page)
        if beside is not None and id(b) not in self.stacked:
            cap = beside.line_rects[0].x0 - 2
            moved = self.starts.get(id(beside))
            if moved is not None:
                cap = min(cap, moved - 2)
            rect = pymupdf.Rect(rect.x0, rect.y0, min(rect.x1, max(cap, rect.x0 + 4)), rect.y1)
            self.write_one(b, t, rect, "left", None, False, rect.x1)
            return
        tail = ctx.continuations.get(id(b))
        if tail is None:
            self._write_paragraph(b, t, rect, align)
        else:
            self._write_host(b, t, rect, align, tail)

    def _write_tail(self, b: TextBlock) -> None:
        """Write the text a paragraph on the previous page (or column) carried over to its tail block b."""
        ctx = self.ctx
        carry = self.writer.carried.get(id(b))
        if carry is None or not carry.text.strip():
            return
        if carry.own_line:
            box = _own_line_box(b, self.cjk)
            self.write_one(b, carry.text, box, "left", carry.formulas, False, box.x1, block_size=carry.size)
            return
        rect, align = _write_rect(b, self.visible or [(b, carry.text)], self.cjk, ctx.rtl, self.geo)
        room = _room_below(
            self.page,
            b,
            rect.y0,
            rect.x0,
            rect.x0 + carry.width,
            self.same_page,
            self.kept,
            self.marks,
            ctx.body_ink(b.page),
        )
        # The line spacing below the last line is blank, so the box may extend past room while the glyphs stay
        # within room (as in _filled_bottom)
        size = ctx.em(carry.size)
        bottom = room + (ctx.gap - 1) * size
        placed = pymupdf.Rect(rect.x0, rect.y0, rect.x0 + carry.width, max(bottom, rect.y0 + 4))
        # The previous page counted, via tail_room, on borrowing from the paragraph below; when the text does not
        # fit here it reaches into that space, and that paragraph moves down
        borrowed = self.flow.spare_below(b, placed, room)
        self.write_one(
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
        self._push_below(b, borrowed, placed.y1)

    def _write_inset(
        self,
        b: TextBlock,
        t: str,
        rect: pymupdf.Rect,
        align: str,
        inset: tuple[pymupdf.Rect, pymupdf.Rect],
    ) -> None:
        """Write a paragraph whose first line starts further in (or out) than the rest in two boxes: the first line,
        and the lines below it (see _inset_write_boxes)."""
        ctx = self.ctx
        first_box, rest_box = inset
        # A right-aligned epigraph with attribution has a short first line starting further right: it is still
        # broken into two boxes, and every line ends at the original right edge
        side = "right" if align == "right" else "left"
        if side == "right":
            first_box = pymupdf.Rect(first_box.x0, first_box.y0, rect.x1, first_box.y1)
            rest_box = pymupdf.Rect(rest_box.x0, rest_box.y0, rect.x1, rest_box.y1)
        moved = self.starts.get(id(b))
        if moved is not None:
            first_box = _clip_write_rect(
                pymupdf.Rect(moved, first_box.y0, first_box.x1, first_box.y1),
                b.clip,
            )
        bformulas = ctx.formulas.get(id(b), [])
        size = ctx.em(b.size)
        head, rest = _split_own_lines(
            t,
            bformulas,
            size,
            b.size,
            first_box.width,
        )
        if head:
            self.write_one(b, head, first_box, side, bformulas, bool(rest.strip()), first_box.x1)
        tail = ctx.continuations.get(id(b))
        if rest.strip() and tail is None:
            self.write_one(b, rest, rest_box, side, bformulas, False, rest_box.x1)
        elif rest.strip() and tail is not None:
            self._write_inset_rest(b, rest, rest_box, side, bformulas, tail)

    def _write_inset_rest(
        self,
        b: TextBlock,
        rest: str,
        rest_box: pymupdf.Rect,
        side: str,
        bformulas: list[PdfDict],
        tail: TextBlock,
    ) -> None:
        """Write the lines below the first line of a two-box paragraph that is carried over to the next page: what
        fits here, and the rest carried to its tail."""
        ctx = self.ctx
        size = ctx.em(b.size)
        room = self._room(b, rest_box)
        floor = self.write_floor(b, rest_box)
        placed_text, spill = _PageBreak(
            rest,
            bformulas,
            size,
            b.size,
            rest_box.width,
            self.cjk,
            room,
            floor,
            ctx.tail_room(tail, rest_box.width, size),
        ).split(rest_box.y0)
        if placed_text:
            lines = placed_text.count("\n") + 1
            bottom = _filled_bottom(rest_box.y0, lines, size, self.cjk, room, floor)
            placed_rect = pymupdf.Rect(
                rest_box.x0,
                rest_box.y0,
                rest_box.x1,
                max(bottom, rest_box.y0 + 4),
            )
            self.write_one(b, placed_text, placed_rect, side, bformulas, bool(spill), rest_box.x1)
        if spill:
            self.writer.carried[id(tail)] = _Carry(spill, bformulas, rest_box.width, b.size)

    def _write_paragraph(self, b: TextBlock, t: str, rect: pymupdf.Rect, align: str) -> None:
        """Write a paragraph that ends on this page. It may start lower (pushed down by the paragraph above), move
        up into free space above, reach down into free space below or borrowed from the paragraph below, or, as the
        last paragraph on the page, carry its last lines to the next page."""
        edge = _indent_edge(b, self.geo, self.ctx.indent) if align == "left" and id(b) not in self.stacked else None
        if edge is not None:
            rect = pymupdf.Rect(edge, rect.y0, rect.x1, rect.y1)
        self._write_spilled_lines(b)
        claim = self.claims.get(id(b))
        pushed = claim is not None and claim > rect.y0
        # Blocks below are looked up from this paragraph's original position. The lowered origin may already be
        # below the top of the next paragraph, and searching down from there would miss it
        own = rect
        if pushed:
            rect = pymupdf.Rect(rect.x0, claim, rect.x1, rect.y1)
        rect, lifted, latest = self._lift_paragraph(b, rect)
        pushed = pushed or lifted
        own_room = self.paragraph_room(b, own) if align == "left" and id(b) not in self.stacked else None
        if own_room is not None:
            below = self.flow.next_movable(b, own, own_room)
            if below is not None:
                self.follows[id(below[0])] = (b, below[1])
        # Text in a cell does not reach into the next cell even when it does not fit
        grow = own_room if b.clip is None else None
        borrowed = self.flow.spare_below(b, own, grow) if grow is not None else None
        if borrowed is not None:
            assert grow is not None  # spare_below is only asked when grow is set
            grow = max(grow, borrowed[1])
        if latest is not None and grow is not None and borrowed is None and latest < rect.y0 - 0.5:
            if self.spill_last_lines(b, t, rect, align, grow, pushed):
                return
        self.write_one(
            b,
            t,
            rect,
            align,
            lower=self.paragraph_top(b, rect, own_room),
            room=grow,
            pushed=pushed,
            lent=borrowed is not None,
        )
        self._push_below(b, borrowed, rect.y1)

    def _write_spilled_lines(self, b: TextBlock) -> None:
        """Write the last lines the paragraph at the bottom of the previous page spilled in front of b (see
        spill_last_lines), and push b down below them."""
        spilled_lines = self.writer.spilled.pop(id(b), None)
        if spilled_lines is None:
            return
        # The last lines of the paragraph at the bottom of the previous page come before this paragraph, which
        # starts half a line below them
        box, carry = spilled_lines
        size = self.ctx.em(carry.size)
        lines = carry.text.count("\n") + 1
        placed = pymupdf.Rect(
            box.x0,
            box.y0,
            box.x0 + carry.width,
            box.y0 + lines * size * self.ctx.gap + size * 0.3,
        )
        self.write_one(b, carry.text, placed, "left", [], False, block_size=carry.size)
        if id(b) in self.ends:
            self.claims[id(b)] = self.ends[id(b)][1] + size * self.ctx.gap * 0.5

    def _lift_paragraph(self, b: TextBlock, rect: pymupdf.Rect) -> tuple[pymupdf.Rect, bool, float | None]:
        """Move a movable paragraph up when needed. Returns its box, whether it moved, and its lowest origin (see
        latest_start; None when the paragraph cannot move)."""
        # This paragraph and the run below it do not fit at normal line spacing and there is space above: move up,
        # at most to half a line below the nearest thing above. The paragraph above was already moved up the same
        # way when written, so the free space above accumulates paragraph by paragraph
        link = self.follows.get(id(b))
        free_box = link[1] if link is not None else self.flow.movable_box(b, None)
        if free_box is None:
            return rect, False, None
        latest = self.flow.latest_start(b, free_box)
        up = self.lift_limit(b, rect, self.ctx.em(b.size)) if latest < rect.y0 - 0.5 else None
        if up is not None and max(up, latest) < rect.y0 - 0.5:
            return pymupdf.Rect(rect.x0, max(up, latest), rect.x1, rect.y1), True, latest
        return rect, False, latest

    def spill_last_lines(
        self,
        block: TextBlock,
        text: str,
        rect: pymupdf.Rect,
        align: str,
        room: float,
        pushed: bool,
    ) -> bool:
        """This paragraph is the last body paragraph on the page, does not fit at normal line spacing, and has no free
        space above or below: the lines staying on this page are written at normal line spacing, and the rest
        continues at the top of the next page (see spill_target). Not done when the remaining lines contain inline
        formulas, because formula images are only pasted on this page"""
        ctx = self.ctx
        index = next(i for i, (other, _) in enumerate(self.items) if other is block)
        if any(t != SPILL_TAIL and not _off_body_size(other) for other, t in self.items[index + 1 :]) or any(
            other.rect.y0 > block.rect.y0 + 1
            and min(other.rect.x1, rect.x1) - max(other.rect.x0, rect.x0) > 8
            and not _off_body_size(other)
            for other in self.same_page
            if other is not block
        ):
            return False
        size = ctx.em(block.size)
        target = ctx.spill_target(block, rect.width, size)
        if target is None:
            return False
        head_block, box, next_room = target
        bformulas = ctx.formulas.get(id(block), [])
        floor = self.write_floor(block, rect)
        lines = _PageBreak(text, bformulas, size, block.size, rect.width, self.cjk, room, floor, next_room)
        head, rest = lines.split(rect.y0)
        gap = ctx.gap
        if (
            not head
            or not rest
            or "\x01" in rest
            or lines.squeezed(head, rect.y0)
            or (rest.count("\n") + 1) * size * gap + size * 0.3 > next_room + (gap - 1) * size + 1
        ):
            return False
        bottom = _filled_bottom(rect.y0, head.count("\n") + 1, size, self.cjk, room, floor)
        self.write_one(
            block,
            head,
            pymupdf.Rect(rect.x0, rect.y0, rect.x1, max(bottom, rect.y0 + 4)),
            align,
            bformulas,
            True,
            pushed=pushed,
        )
        self.writer.spilled[id(head_block)] = (box, _Carry(rest, bformulas, rect.width, block.size))
        return True

    def _write_host(self, b: TextBlock, t: str, rect: pymupdf.Rect, align: str, tail: TextBlock) -> None:
        """Write a paragraph carried over to the next page (or column): the lines that fit here, and the rest
        carried to its tail block."""
        ctx = self.ctx
        bformulas = ctx.formulas.get(id(b), [])
        # A half sentence at the end of a line is only a few words wide. Breaking at its width would squeeze the
        # continuation page too. The source is erased and the whole paragraph is written on the continuation page
        # at that page's column width.
        if len(b.line_rects) <= 1 and rect.width < tail.rect.width * 0.7:
            self.writer.carried[id(tail)] = _Carry(t, bformulas, tail.rect.width, b.size)
            return
        left = _indented_start(b, tail, ctx.geometries)
        if left is not None:
            rect = pymupdf.Rect(left, rect.y0, rect.x1, rect.y1)
        size = ctx.em(b.size)
        room = self._room(b, rect)
        floor = self.write_floor(b, rect)
        next_room = ctx.tail_room(tail, rect.width, size)
        # The paragraph above borrowed the space freed by this paragraph moving down (see host_latest)
        claim = self.claims.get(id(b))
        if claim is not None and claim > rect.y0:
            rect = pymupdf.Rect(rect.x0, claim, rect.x1, rect.y1)
        lines = _PageBreak(t, bformulas, size, b.size, rect.width, self.cjk, room, floor, next_room)
        rect, head, rest = self._lift_host(b, rect, lines)
        if head:
            bottom = _filled_bottom(rect.y0, head.count("\n") + 1, size, self.cjk, room, floor)
            head_rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, max(bottom, rect.y0 + 4))
            self.write_one(b, head, head_rect, align, bformulas, bool(rest))
        if rest:
            self.writer.carried[id(tail)] = _Carry(rest, bformulas, rect.width, b.size)

    def _lift_host(self, b: TextBlock, rect: pymupdf.Rect, lines: _PageBreak) -> tuple[pymupdf.Rect, str, str]:
        """Move a paragraph carried over to the next page up when the lines staying here do not fit at normal line
        spacing. Returns its box, the lines staying here and the rest."""
        size = lines.size
        head, rest = lines.split(rect.y0)
        # Does not fit at normal line spacing and there is space above: move up. The higher it moves, the more lines
        # this page holds, so find the lowest origin at which they fit
        up = self.lift_limit(b, rect, size) if lines.squeezed(head, rect.y0) else None
        if up is not None and up < rect.y0 - 0.5 and not lines.squeezed(lines.split(up)[0], up):
            low = _last_fitting(lambda y0: not lines.squeezed(lines.split(y0)[0], y0), up, rect.y0)
            head, rest = lines.split(low)
            return pymupdf.Rect(rect.x0, low, rect.x1, rect.y1), head, rest
        if up is not None and up < rect.y0 - 0.5:
            # The extra lines gained by moving up are still squeezed (the next page cannot take them): keep the line
            # count and move up to where these lines fit at normal line spacing
            gap = self.ctx.gap
            count = head.count("\n") + 1
            start = min(lines.room + (gap - 1) * size, lines.floor) - count * size * gap - size * 0.3
            if up <= start < rect.y0 - 0.5:
                rect = pymupdf.Rect(rect.x0, start, rect.x1, rect.y1)
        return rect, head, rest

    def _finish(self) -> None:
        """Draw the deferred images and formula images, restore the bitmap glyphs and put the reading order
        together."""
        _flush_png(self.page, self.deferred_png)
        if self.mid_map:
            _place_formula_images(self.page, self.doc, self.images_before, self.mid_map, self.forms, self.page_spans)
            restore_glyphs(self.doc, self.glyphs)
        self.order.assemble()


def _part_page(part: PdfDict, formula: PdfDict, pno: int) -> int:
    """The page a formula part lies on: its own, else the formula's, else the page being written."""
    # Bound to an int first: with an Any default, mypy picks dict.get's None-default overload
    default: int = formula.get("page", pno)
    return int(part.get("page", default))
