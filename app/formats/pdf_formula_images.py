"""Paste the cropped images of inline formulas into the typeset lines."""

import pymupdf

from .pdf_place_formulas import _FormulaSources, _paint_formula
from .pdf_types import Formula, Span, text_dict


def _white_pixmap(doc: "pymupdf.Document") -> "pymupdf.Pixmap":
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1))
    pix.set_pixel(0, 0, (255, 255, 255))
    return pix


def _place_formula_images(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    before: set[int],
    mid_map: dict[int, Formula],
    forms: "_FormulaSources | None" = None,
    page_spans: list[Span] | None = None,
) -> None:
    """Replace the placeholder markers with formula content and correct the baseline.

    htmlbox aligns an <img> only by its bottom, while the baseline of a formula lies inside the image (sub- and
    superscripts extend below the baseline), so a formula inserted directly would float up as a whole. A 1x1
    marker pixel sized to the formula's width and height is inserted first, the text baseline is read from the
    rectangle where it lands, and the formula is then moved down by "bbox bottom − formula baseline" to its exact
    position. When a source page Form (forms) is available, a wrapper Form with a tight BBox does vector
    pass-through (original glyphs, selectable, selection does not blow up the box); on failure it falls back to a
    PNG bitmap.

    Marker pixels encode a global index (mid), and the whole page is processed at once. All formula drawing goes
    into one new content stream at the end: insert_htmlbox creates a new content stream for each block, and after
    a marker is painted white the white patch stays in that block's stream, so a formula drawn in an earlier
    stream would have its upper half covered by the white patches of later blocks.
    """
    draws: list[str] = []
    # Text spans after rendering: htmlbox aligns img only by its bottom, and the marker bottom has a systematic
    # offset of ~1pt from the text baseline, so anchoring the formula baseline to the origin of text on the same
    # line is the most reliable
    text_spans = [
        s
        for blk in text_dict(page)["blocks"]
        if blk.get("type") == 0
        for ln in blk["lines"]
        for s in ln["spans"]
        if s["text"].strip() and s.get("origin")
    ]
    now = {img[0] for img in page.get_images(full=True)}
    for xref in now - before:
        pix = pymupdf.Pixmap(doc, xref)
        p = pix.pixel(0, 0)
        if len(p) < 3 or p[2] != 200:
            continue
        f = mid_map.get(p[0] + (p[1] << 8) - 1)
        if f is None:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        rs = f.get("rs", 1.0)
        # raise: the origin of a standalone superscript formula (²) is above the body baseline, and placement
        # moves it back up by the difference
        w, h = f["w"] * rs, f["h"] * rs
        d = (f["d"] - f.get("raise", 0.0)) * rs
        cy = (rect.y0 + rect.y1) / 2
        baseline = None
        best_key = (0.0, float("inf"))
        for s in text_spans:
            sb = s["bbox"]
            if not (sb[1] - 1 <= cy <= sb[3] + 1):  # not on the same line
                continue
            # With the enlarged CJK line-height, glyph bboxes of adjacent lines overlap: take the largest y overlap
            # with the marker rectangle (the same line overlaps most), and on a tie the nearest in x
            overlap = min(sb[3], rect.y1) - max(sb[1], rect.y0)
            dx = max(0.0, sb[0] - rect.x1, rect.x0 - sb[2])
            if (overlap, -dx) > (best_key[0], -best_key[1]):
                best_key, baseline = (overlap, dx), s["origin"][1]
        if baseline is not None:
            target = pymupdf.Rect(rect.x0 + 2.0, baseline + d - h, rect.x0 + 2.0 + w, baseline + d)
        else:
            target = pymupdf.Rect(rect.x0 + 2.0, rect.y0 + d, rect.x0 + 2.0 + w, rect.y0 + d + h)
        page.replace_image(xref, pixmap=_white_pixmap(doc))
        _paint_formula(page, doc, f, target, forms, page_spans, draws)
    if draws:
        ns = doc.get_new_xref()
        doc.update_object(ns, "<<>>")
        doc.update_stream(ns, ("\n" + "\n".join(draws) + "\n").encode())
        cont_v = doc.xref_get_key(page.xref, "Contents")[1].strip()
        if cont_v.startswith("["):
            doc.xref_set_key(page.xref, "Contents", f"{cont_v[:-1].rstrip()} {ns} 0 R]")
        else:
            doc.xref_set_key(page.xref, "Contents", f"[{cont_v} {ns} 0 R]")
