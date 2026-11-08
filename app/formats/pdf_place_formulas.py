"""Draw inline formulas onto the page: wrapper forms cropped from the source page, vector pass-through and
the bitmap fallback."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pymupdf
import pymupdf.mupdf as mupdf

from .pdf_align import page_origin
from .pdf_blocks import _PART_GAP, _formula_parts, _FormulaImage, _GlyphInk, _outside_formula
from .pdf_types import Formula, Span
from .pdf_typeset import (
    FORMULA_PAD,
)


def _part_lead(part: Formula, scale: float) -> float:
    """The gap between this part and the previous one, scaled with the formula."""
    return max(_PART_GAP, float(part.get("lead") or 0) * scale)


def _formula_png(formula: Formula) -> bytes | None:
    """The bitmap of a formula: tests give a PNG directly, the normal pipeline gives a _FormulaImage."""
    png = formula.get("png")
    return png.png() if isinstance(png, _FormulaImage) else png


def _formula_png_rect(formula: Formula, target: pymupdf.Rect) -> pymupdf.Rect:
    """Where the bitmap goes. The parts of the cropped image that extend past the formula box (see
    _FormulaImage.area) are placed on both sides, scaled by the same factor as the formula box to target."""
    png, bbox = formula.get("png"), formula.get("bbox")
    if not isinstance(png, _FormulaImage) or bbox is None or bbox.width <= 0 or bbox.height <= 0:
        return target
    area = png.area()
    sx, sy = target.width / bbox.width, target.height / bbox.height
    return pymupdf.Rect(
        target.x0 + (area.x0 - bbox.x0) * sx,
        target.y0 + (area.y0 - bbox.y0) * sy,
        target.x1 + (area.x1 - bbox.x1) * sx,
        target.y1 + (area.y1 - bbox.y1) * sy,
    )


def _pdf_rect(page: "pymupdf.Page", rect: "pymupdf.Rect") -> "pymupdf.Rect":
    """Convert a rectangle in page coordinates to PDF coordinates (y up). When the crop box does not start at the
    origin, its offset is added."""
    left, top = page_origin(page)
    return pymupdf.Rect(rect.x0 + left, top - rect.y1, rect.x1 + left, top - rect.y0)


class _GlyphKeeper(mupdf.PdfSanitizeFilterOptions2):
    """Delete glyphs only: glyphs whose center lies outside keep are deleted, and the positions of the kept ones
    are recorded to be checked after filtering. Paths and images are not deleted; the clipping path hides them.
    Deleting paths by position would also delete the strokes inside Type3 glyph procedures (bitmap glyphs such
    as 𝟙 would disappear). The callback runs once per glyph, so the simpler it is, the faster."""

    def __init__(self, keep: list[tuple[float, float, float, float]]) -> None:
        # The SWIG bindings are unannotated; mypy accepts calls into them only through the binding's own class
        # (untyped_calls_exclude), not through this subclass (as in pdf_erase)
        mupdf.PdfSanitizeFilterOptions2.__init__(self)
        base: mupdf.PdfSanitizeFilterOptions2 = self
        self.keep = keep
        self.kept: list[tuple[float, float]] = []
        base.use_virtual_text_filter()
        # The filter also calls this callback when there are images, and raises an error if it is not implemented
        base.use_virtual_image_filter()

    def image_filter(
        self, ctx: mupdf.fz_context, ctm: mupdf.fz_matrix, name: str, image: mupdf.fz_image, scissor: mupdf.fz_rect
    ) -> mupdf.fz_image:
        # The returned image is released once after use, so keeping it unchanged needs one extra reference first
        kept: mupdf.fz_image = mupdf.ll_fz_keep_image(image)
        return kept

    def text_filter(
        self,
        ctx: mupdf.fz_context,
        ucs: object,
        length: int,
        trm: mupdf.fz_matrix,
        ctm: mupdf.fz_matrix,
        bbox: mupdf.fz_rect,
        *rest: object,
    ) -> int:
        # The glyph center is computed from the glyph's own box. The box of the whole font is wider than a
        # parenthesis, and the center would fall outside the parenthesis. Returning 1 deletes the glyph
        gx, gy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
        x, y = gx * trm.a + gy * trm.c + trm.e, gx * trm.b + gy * trm.d + trm.f
        x, y = x * ctm.a + y * ctm.c + ctm.e, x * ctm.b + y * ctm.d + ctm.f
        # Bitmap glyphs often do not report their own box (width 0), only the vertical bounds of the whole font,
        # and the computed center is half a line away from the ink. In that case the origin is used horizontally,
        # and vertically the glyph is kept as long as those bounds overlap a keep box: the clipping path hides the
        # strokes
        reach = None
        if bbox.x0 == bbox.x1:
            ends = [gy_ * trm.d + trm.f for gy_ in (bbox.y0, bbox.y1)]
            ends = [x * ctm.b + y_ * ctm.d + ctm.f for y_ in ends]
            reach = (min(ends), max(ends))
        for x0, y0, x1, y1 in self.keep:
            if x0 <= x <= x1 and (y0 <= y <= y1 or reach is not None and reach[0] < y1 and reach[1] > y0):
                self.kept.append((x, y))
                return 0
        return 1


class _SanitizeFactory(mupdf.PdfFilterFactory2):
    # The attribute must not be called options: that is a field name of the underlying struct, and assigning to it
    # writes into the C struct
    def __init__(self, sanitize: mupdf.PdfSanitizeFilterOptions2) -> None:
        # Base-class calls for the same reason as in _GlyphKeeper
        mupdf.PdfFilterFactory2.__init__(self)
        base: mupdf.PdfFilterFactory2 = self
        self.sanitize = sanitize
        base.use_virtual_filter()

    def filter(
        self,
        ctx: mupdf.fz_context,
        doc: mupdf.pdf_document,
        chain: mupdf.pdf_processor,
        struct_parents: int,
        transform: mupdf.fz_matrix,
        options: mupdf.pdf_filter_options,
    ) -> mupdf.pdf_processor:
        processor: mupdf.pdf_processor = mupdf.ll_pdf_new_sanitize_filter(
            doc,
            chain,
            struct_parents,
            transform,
            options,
            self.sanitize.internal(),
        )
        return processor


def _filter_form(
    doc: "pymupdf.Document", form: int, keep: list[tuple[float, float, float, float]], recurse: bool = True
) -> tuple[int, list[tuple[float, float]]]:
    """Filter a Form once and return the xref of the new Form and the positions of the kept glyphs. When recurse
    is false, only this level is filtered and nested Forms are referenced unchanged."""
    keeper = _GlyphKeeper(keep)
    factory: mupdf.PdfFilterFactory2 = _SanitizeFactory(keeper)
    options = mupdf.PdfFilterOptions()
    # Forms nested in the page (figures) are filtered too. The result is stored as a new object because the page
    # itself still uses the original one, and changing it in place would delete the figure.
    # When nested Forms are not entered, they are not copied either: copying makes MuPDF rewrite the content
    # stream, and the figure's bytes would differ from the original (see _lines_form)
    options.recurse = int(recurse)
    options.instance_forms = int(recurse)
    options.add_factory(factory.internal())
    pdoc = doc.this if isinstance(doc.this, mupdf.PdfDocument) else mupdf.PdfDocument(doc.this)
    ref = mupdf.pdf_new_indirect(pdoc, form, 0)
    res = mupdf.pdf_dict_gets(ref, "Resources")
    out = mupdf.pdf_filter_xobject_instance(ref, res, mupdf.FzMatrix(), options, mupdf.PdfCycleList())
    return mupdf.pdf_to_num(out), keeper.kept


_EVERYWHERE = [(-1e9, -1e9, 1e9, 1e9)]


def _keeps_places(doc: "pymupdf.Document", xref: int, kept: list[tuple[float, float]], recurse: bool = True) -> bool:
    """Every glyph in the filtered Form is still at its original position. recurse is the same as when filtering."""
    if not xref:
        return False
    _same, now = _filter_form(doc, xref, _EVERYWHERE, recurse)
    return _same_places(kept, now)


def _same_places(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """The two sets of glyph positions match one to one, each within 0.1. Sorting and comparing in order does not
    work: the glyphs at the start of several lines in one column differ in x by a few ten-thousandths, the sort
    order gets mixed up, and glyphs from different lines are compared. Here the positions are put into cells of
    0.1, and each glyph is looked up in its own cell and the neighbouring ones."""
    if len(a) != len(b):
        return False
    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for x, y in b:
        cells.setdefault((round(x * 10), round(y * 10)), []).append((x, y))
    for x, y in a:
        cx, cy = round(x * 10), round(y * 10)
        for key in [(cx + i, cy + j) for i in (-1, 0, 1) for j in (-1, 0, 1)]:
            cell = cells.get(key, [])
            hit = next((k for k, (u, v) in enumerate(cell) if abs(u - x) < 0.1 and abs(v - y) < 0.1), None)
            if hit is not None:
                cell.pop(hit)
                break
        else:
            return False
    return True


def _keep_marks(doc: "pymupdf.Document", form: int, keep: list["pymupdf.Rect"]) -> int:
    """Copy the Form keeping only the glyphs inside keep, and return the xref of the new Form. keep is in PDF
    coordinates (y up).

    Formula pass-through references the whole source page. The clipping path hides only the strokes, not the
    text layer: when selecting by dragging or copying in a browser, every formula has a whole page of invisible
    source text under it, and one drag selects a large area.
    When MuPDF deletes earlier glyphs on the same line, a line that changes font size midway (THEME in small caps)
    gets its later glyphs placed a few points off, outside the clip box. So the positions are checked after
    filtering: if they moved, whole lines are kept (deleting other lines does not affect this line), and if they
    still move, the original Form is used.
    """
    boxes = [(k.x0, k.y0, k.x1, k.y1) for k in keep]
    rows = [(-1e9, y0, 1e9, y1) for _x0, y0, _x1, y1 in boxes]
    try:
        for area in (boxes, rows):
            xref, kept = _filter_form(doc, form, area)
            if _keeps_places(doc, xref, kept):
                return xref
    except Exception:
        pass
    return form


def _merge_bands(bands: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for y0, y1 in sorted(bands):
        if merged and y0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])
    return [(y0, y1) for y0, y1 in merged]


def _lines_form(doc: "pymupdf.Document", form: int, bands: list[tuple[float, float]]) -> int:
    """Filter this level of the Form once, keeping only the glyphs in bands (vertical ranges in PDF coordinates),
    and return the xref of the new Form. The original Form is used when the position check fails.

    Nested Forms (figures) are not entered and are referenced unchanged; they are left to the last pass of each
    formula (_keep_marks). Every MuPDF filter pass removes one level of empty q Q, so the resulting bytes depend
    on the pass in which the glyphs of a figure are deleted. If the figures of different formulas go through
    different numbers of passes, they have the same content and different bytes, merging duplicate objects cannot
    merge them, and a large figure is stored several times in the file."""
    try:
        xref, kept = _filter_form(doc, form, [(-1e9, y0, 1e9, y1) for y0, y1 in bands], recurse=False)
        if _keeps_places(doc, xref, kept, recurse=False):
            return xref
    except Exception:  # noqa: BLE001 - as in _keep_marks, use the original Form when filtering fails
        pass
    return form


@dataclass
class _LinesNode:
    bands: list[tuple[float, float]]
    xref: int
    children: list["_LinesNode"]


class _FormulaSources:
    """Source Forms referenced by formula pass-through on this page.

    Before pass-through, each formula filters the source Form once (see _keep_marks), with one callback per glyph.
    If every filter ran on the whole page, a page with dozens of formulas would walk through all its glyphs dozens
    of times. Here the formulas on the page are split in half by vertical position, each half keeps only the lines
    it lies on, and the split continues until each part has at most four formulas: each level walks the whole
    page only once in total, and each formula is finally filtered only on the glyphs of its own lines.
    The lines kept are the formula boxes extended by 2pt above and below. This covers the clip box of
    _formula_form (extended by at most 1.5pt above and below), so the glyphs to keep when filtering by box or by
    line all lie inside. A lookup finds by position the smallest part that covers the whole formula, so formulas
    that were not registered are found too; at worst it is the whole page."""

    def __init__(self, doc: "pymupdf.Document", page: "pymupdf.Page", form: int, formulas: Iterable[Formula]) -> None:
        # Keep only the page top needed for converting to PDF coordinates (see _pdf_rect), not the page object
        self.top = page_origin(page)[1]
        self.glyphs = _GlyphInk(page)
        group: list[tuple[tuple[float, float], Formula]] = []
        for formula in formulas:
            for part in _formula_parts(formula):
                source_page = part.get("page")
                band = self._band(part)
                if band is not None and (source_page is None or int(source_page) == page.number):
                    group.append((band, part))

        def build(source: int, group: list[tuple[tuple[float, float], Formula]]) -> _LinesNode:
            bands = _merge_bands([band for band, _part in group])
            node = _LinesNode(bands, _lines_form(doc, source, bands), [])
            if len(group) > 4:
                group = sorted(group, key=lambda item: item[0])
                half = len(group) // 2
                node.children = [build(node.xref, sub) for sub in (group[:half], group[half:])]
            return node

        self.root = _LinesNode([], form, [build(form, group)] if group else [])

    def _band(self, formula: Formula) -> tuple[float, float] | None:
        rects = [s["bbox"] for s in formula.get("spans") or []] + list(formula.get("curves") or [])
        if formula.get("bbox") is not None:
            rects.append(formula["bbox"])
        if not rects:
            return None
        return self.top - max(rect[3] for rect in rects) - 2, self.top - min(rect[1] for rect in rects) + 2

    def source(self, formula: Formula) -> int:
        band = self._band(formula)
        node = self.root
        while band is not None:
            inner = next(
                (child for child in node.children if any(y0 <= band[0] and band[1] <= y1 for y0, y1 in child.bands)),
                None,
            )
            if inner is None:
                break
            node = inner
        return node.xref


def _hole_ink(span: Span) -> pymupdf.Rect:
    """The hole a foreign span cuts in the formula clip: its ink box, clipped to its font box. An exact box larger
    than the font box (the exact box of LMMono extends half a line downward) is not ink, so the hole follows the
    font box; otherwise it would shave off the tops of the glyphs on the next line. Ink measured by drawing the
    glyph on its own is used unchanged."""
    box = pymupdf.Rect(span["bbox"])
    if span.get("ink") is None:
        return box
    ink = pymupdf.Rect(span["ink"])
    return ink if span.get("ink_drawn") else ink & box


def _formula_form(
    doc: "pymupdf.Document",
    page: "pymupdf.Page",
    fmS: int,
    f: Formula,
    f_name: str,
    page_spans: list[Span] | None = None,
    glyphs: _GlyphInk | None = None,
) -> int:
    """Build a wrapper Form with a tight BBox for one formula (clipped span by span, then referencing the source
    page Form) and return its xref.

    The clipping path is the union of the rectangles of the spans in the run, not the whole bounding rectangle:
    the bounding rectangle would also take in strokes of neighbouring glyphs in the gaps (the descender of a p on
    the line above, and so on) and leave a bar above the formula.
    The BBox is the bounding box of the union, so selecting/copying does not frame a large area.

    The clip uses even-odd: the span rectangles of the run are extended per side and united (italic strokes go
    past the typeset bbox of a span; the right arm of Y can go 1.2pt past it, and extension stops before touching
    a neighbour), then the rectangular holes of foreign spans (descenders from the line above and so on) are
    subtracted, so the formula's own glyphs are not shaved and nothing from other glyphs is picked up.
    The clip does not hide the text layer, so the referenced source page Form first has the glyphs outside the
    clip region removed and keeps only those of this formula.
    """
    bbox = f["bbox"]
    clip_outer = _pdf_rect(page, bbox)
    # The spans of the run come from a separate get_text call in extract_blocks and are different objects from
    # page_spans, so run membership can only be decided by equal bboxes; otherwise sibling spans (superscripts of
    # this formula) would be treated as foreign spans and cut out as holes
    curve_rects = [pymupdf.Rect(r) for r in f.get("curves") or []]
    run_rects = [pymupdf.Rect(s["bbox"]) for s in f["spans"]] + curve_rects
    if page_spans:
        page_spans = _outside_formula(page_spans, f["spans"])

    def merge_rects(rects: list["pymupdf.Rect"]) -> list["pymupdf.Rect"]:
        out: list[pymupdf.Rect] = []
        for b in rects:
            for i, mb in enumerate(out):
                if not (b & mb).is_empty:
                    out[i] = mb | b
                    break
            else:
                out.append(b)
        return out

    def is_run_member(fr: "pymupdf.Rect") -> bool:
        return any(
            abs(fr.x0 - rr.x0) < 0.5
            and abs(fr.y0 - rr.y0) < 0.5
            and abs(fr.x1 - rr.x1) < 0.5
            and abs(fr.y1 - rr.y1) < 0.5
            for rr in run_rects
        )

    # First merge into disjoint sets (the bbox of the hat on ŷ almost fully overlaps y, and without merging,
    # even-odd would switch off the whole stem of y), then extend per side: up to 1.5pt per side, stopping before
    # touching a neighbour's bbox
    base = merge_rects([pymupdf.Rect(s["bbox"]) for s in f["spans"]] + curve_rects)
    union = pymupdf.Rect()
    for rr in base:
        union |= rr
    lim = {"t": 1.5, "b": 1.5}  # upper limit of extension above and below (pt); left and right also 1.5
    # Edges of the left and right neighbours. Each rectangle extends by its own distance to the neighbours: with
    # the bounding box of the whole formula, a comma right next to it would let every rectangle inside extend only
    # 0.2, and the stroke of 𝒩 reaching into the space before the next glyph would be cut off
    left_edges: list[float] = []
    right_edges: list[float] = []
    if page_spans:
        for fb in page_spans:
            if not fb["text"].strip():  # blank spans have no ink; they neither limit extension nor cut holes
                continue
            # First use coordinates to exclude spans not at the formula's height: a page has thousands, and building
            # a Rect for each is slow
            if not (fb["bbox"][1] < union.y1 and fb["bbox"][3] > union.y0):
                continue
            fr = pymupdf.Rect(fb["bbox"])
            if is_run_member(fr) or not (fr.y0 < union.y1 and fr.y1 > union.y0):
                continue
            if fr.x1 <= union.x0 + 0.01:  # left neighbour
                left_edges.append(fr.x1)
            elif fr.x0 >= union.x1 - 0.01:  # right neighbour
                right_edges.append(fr.x0)
            if fr.y1 <= union.y0 + 0.01:  # neighbour above (pymupdf y points down)
                lim["t"] = max(0.2, min(lim["t"], union.y0 - fr.y1 - 0.3))
            elif fr.y0 >= union.y1 - 0.01:  # neighbour below
                lim["b"] = max(0.2, min(lim["b"], fr.y0 - union.y1 - 0.3))

    def reach(gaps: list[float]) -> float:
        return max(0.2, min([1.5, *(gap - 0.3 for gap in gaps)]))

    our_rects = merge_rects(
        [
            pymupdf.Rect(
                b.x0 - reach([b.x0 - x for x in left_edges]),
                b.y0 - lim["t"],
                b.x1 + reach([x - b.x1 for x in right_edges]),
                b.y1 + lim["b"],
            )
            for b in base
        ]
    )
    ours, holes = [], []
    for b in our_rects:
        r = _pdf_rect(page, b)
        if r.width > 0.2 and r.height > 0.2:
            ours.append(f"{r.x0:.2f} {r.y0:.2f} {r.width:.2f} {r.height:.2f} re")
    if page_spans and our_rects:
        hole_rects = []
        reach_x0 = min(r.x0 for r in our_rects)
        reach_y0 = min(r.y0 for r in our_rects)
        reach_x1 = max(r.x1 for r in our_rects)
        reach_y1 = max(r.y1 for r in our_rects)

        def touches(box: Sequence[float]) -> bool:
            return box[0] <= reach_x1 and box[2] >= reach_x0 and box[1] <= reach_y1 and box[3] >= reach_y0

        for fb in page_spans:
            if not fb["text"].strip():  # blank spans have no ink; they play no part in limits or holes
                continue
            # Holes lie in the span's font box or ink box; a span where neither touches the clip box cannot cut a hole
            if not touches(fb["bbox"]) and (fb.get("ink") is None or not touches(fb["ink"])):
                continue
            fr = pymupdf.Rect(fb["bbox"])
            if any(
                abs(fr.x0 - rr.x0) < 0.5
                and abs(fr.y0 - rr.y0) < 0.5
                and abs(fr.x1 - rr.x1) < 0.5
                and abs(fr.y1 - rr.y1) < 0.5
                for rr in run_rects
            ):
                continue
            # Whether to cut is decided by the font box, and the hole follows the ink: the font boxes of parentheses
            # and arrows are a good deal taller than the glyphs and reach into the glyphs on the next line
            ink = _hole_ink(fb)
            # hole = intersection of the foreign span with each of our rectangles. It must lie entirely inside our
            # rectangles: under even-odd clipping, a hole that goes past the union is inverted and draws the whole
            # foreign stroke it was meant to exclude.
            # Only spans that are not vertically aligned are cut (descenders, indices of a neighbouring formula and so
            # on): the bbox of a neighbouring glyph on the same line often just touches the edge and has almost the
            # same vertical extent, and cutting it would shave the strokes of this formula (the left stem of y).
            # Holes are not shrunk: there is already a gap between descender ink and the bottom of the span bbox,
            # and shrinking would leave a faint trace
            for rr in our_rects:
                y_ov = min(fr.y1, rr.y1) - max(fr.y0, rr.y0)
                if y_ov > 0.7 * (rr.y1 - rr.y0):
                    continue
                hb = ink & rr
                if not hb.is_empty and hb.width >= 0.3 and hb.height >= 0.3:
                    hole_rects.append(hb)
        # Merge intersecting holes into one: under even-odd, the area where two holes intersect is inverted
        merged: list[pymupdf.Rect] = []
        for hb in hole_rects:
            for i, mb in enumerate(merged):
                if not (hb & mb).is_empty:
                    merged[i] = mb | hb
                    break
            else:
                merged.append(hb)
        for hb in merged:
            hr = _pdf_rect(page, hb)
            holes.append(f"{hr.x0:.2f} {hr.y0:.2f} {hr.width:.2f} {hr.height:.2f} re")
    # The glyphs to keep are identified by the formula's own span boxes, not by the extended clip box: MuPDF only
    # gives the vertical bounds of the whole font, so the computed center of a glyph on the next line is 0.25em
    # above its baseline, falls right into the strip added below the clip box, and becomes invisible text in the
    # formula image
    source = _keep_marks(doc, fmS, [_pdf_rect(page, r) for r in base]) if our_rects else fmS
    # The BBox clips too. The hook and tail of an italic f reach past its character width and would be cut off by
    # the formula box: extend left and right to the ink measured from the glyphs, no further than the clipping
    # path. Extend only to the formula's own ink, so strokes of neighbouring glyphs reaching into the space (the f
    # of If) are not brought in
    bounds = [glyphs.x_bounds(span) for span in f["spans"]] if glyphs is not None and our_rects else []
    inks = [ink for ink in bounds if ink is not None]
    if inks:
        left = max(min(ink[0] for ink in inks), min(r.x0 for r in our_rects))
        right = min(max(ink[1] for ink in inks), max(r.x1 for r in our_rects))
        ink_rect = _pdf_rect(page, pymupdf.Rect(left, bbox.y0, right, bbox.y1))
        clip_outer = pymupdf.Rect(
            min(clip_outer.x0, ink_rect.x0), clip_outer.y0, max(clip_outer.x1, ink_rect.x1), clip_outer.y1
        )
    fmT: int = doc.get_new_xref()
    doc.update_object(
        fmT,
        (
            f"<< /Type /XObject /Subtype /Form /BBox [{clip_outer.x0:.2f} {clip_outer.y0:.2f} "
            f"{clip_outer.x1:.2f} {clip_outer.y1:.2f}] "
            f"/Matrix [1 0 0 1 0 0] /Resources << /XObject << /FmS {source} 0 R >> >> >>"
        ),
    )
    doc.update_stream(fmT, f"q {' '.join(ours + holes)} W* n /FmS Do Q".encode())
    res_v = doc.xref_get_key(page.xref, "Resources")[1]
    # After redaction /Resources becomes a direct dictionary (keys need the Resources/ prefix);
    # for an indirect reference the keys are set directly on the resource object
    m = re.match(r"(\d+) \d+ R", res_v)
    res_xref, key_prefix = (int(m.group(1)), "") if m else (page.xref, "Resources/")
    xo_v = doc.xref_get_key(res_xref, f"{key_prefix}XObject")[1]
    if xo_v == "null":
        doc.xref_set_key(res_xref, f"{key_prefix}XObject", "<<>>")
        xo_v = "<<"
    m = re.match(r"(\d+) \d+ R", xo_v)
    if m:
        doc.xref_set_key(int(m.group(1)), f_name, f"{fmT} 0 R")
    else:
        doc.xref_set_key(res_xref, f"{key_prefix}XObject/{f_name}", f"{fmT} 0 R")
    return fmT


def _paint_formula(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    formula: Formula,
    target: "pymupdf.Rect",
    forms: "_FormulaSources | None",
    page_spans: list[Span] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]] | None = None,
) -> None:
    """Draw the formula into target. Falls back to a bitmap when vector pass-through fails. Drawing commands are
    collected and written in one content stream at the end.

    If the formula comes from another page, this page's Form does not contain its strokes, so only the already
    cropped image can be pasted.
    """
    bbox = formula.get("bbox")
    source_page = formula.get("page")
    same_page = source_page is None or int(source_page) == page.number
    if (
        same_page
        and forms is not None
        and bbox is not None
        and bbox.width > 0.5
        and bbox.height > 0.5
        and target.width > 0.5
    ):
        try:
            # The resource name must be unique on the page: FmF0 entries with the same index within their blocks
            # would overwrite each other
            f_name = "FmF_" + re.sub(r"[^A-Za-z0-9_]", "_", str(formula["name"]))
            _formula_form(doc, page, forms.source(formula), formula, f_name, page_spans, forms.glyphs)
            sx = target.width / bbox.width
            sy = target.height / bbox.height
            # Affine transform with scaling: the translation must absorb the scale, otherwise the whole formula
            # drifts down by (1-sy)*H
            source = _pdf_rect(page, bbox)
            placed = _pdf_rect(page, target)
            tx = placed.x0 - sx * source.x0
            ty = placed.y0 - sy * source.y0
            draws.append(f"q {sx:.4f} 0 0 {sy:.4f} {tx:.2f} {ty:.2f} cm /{f_name} Do Q")
            return
        except Exception:  # noqa: BLE001 - fall back to a bitmap when vector pass-through fails
            pass
    png = _formula_png(formula)
    if not png:
        return
    target = _formula_png_rect(formula, target)
    if deferred_png is not None:
        deferred_png.append((target, png))
        return
    page.insert_image(target, stream=png, keep_proportion=False, overlay=True)


def _paint_formula_parts(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    formula: Formula,
    slot_x: float,
    baseline: float,
    forms: "_FormulaSources | None",
    page_spans: list[Span] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]],
) -> None:
    """Paste the parts of one placeholder side by side in reading order. The second half split off by a line break
    no longer stays at its x in the source."""
    scale = float(formula.get("rs") or 1)
    x = slot_x + FORMULA_PAD
    for index, part in enumerate(_formula_parts(formula)):
        if index:
            x += _part_lead(part, scale)
        width = float(part["w"]) * scale
        height = float(part["h"]) * scale
        below = (float(part.get("d") or 0) - float(part.get("raise") or 0)) * scale
        target = pymupdf.Rect(x, baseline + below - height, x + width, baseline + below)
        _paint_formula(page, doc, part, target, forms, page_spans, draws, deferred_png)
        x += width
