"""Crop each inline formula from the original page into an image to paste into the translation."""

from dataclasses import replace

import pymupdf

from .. import background
from .pdf_blocks import (
    _INK_PAD,
    _TEXT_FLAGS,
    TextBlock,
    _attach_ink,
    _boxes_match,
    _CharIndex,
    _formula_parts,
    _FormulaImage,
    _outside_formula,
    _page_char_index,
)
from .pdf_drawings import light_fill, text_page_rect, visible_drawings
from .pdf_runs import (
    FormulaRun,
    expand_box,
    intersecting_curves,
)


def _page_curve_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Vector lines, plus one-pixel images stretched into thin rules: old TeX draws fraction bars and radical
    bars that way. A stroked line counts by the area it paints: the path box excludes the line width, so a
    horizontal line has a box of zero height."""
    drawings = visible_drawings(page)
    rects: list[pymupdf.Rect] = []
    for drawing in drawings:
        raw = drawing.get("rect")
        # A light fill is a background with text on top, not a stroke. Adding it to the formula box would crop
        # the text of the cells above and below into the image and erase the fill when writing
        if raw is None or light_fill(drawing):
            continue
        rect = pymupdf.Rect(raw)
        half = float(drawing.get("width") or 0) / 2 if "s" in str(drawing.get("type") or "") else 0.0
        if half > 0:
            rect = rect + (-half, -half, half, half)
        if not rect.is_empty:
            rects.append(rect)
    try:
        images = page.get_image_info()
    except Exception:  # noqa: BLE001
        images = []
    for info in images:
        rect = pymupdf.Rect(info["bbox"])
        if not rect.is_empty and min(rect.width, rect.height) <= _RULE_IMAGE_THICK:
            rects.append(rect)
    return rects


def _absorb_one_box(formula: dict, drawings: list[pymupdf.Rect], glyphs: list[dict] = ()) -> None:
    """glyphs are the spans of the page (with ink), used to recognize rules that other text also touches."""
    old = pymupdf.Rect(formula["bbox"])
    # Search by glyph ink: the font boxes of math fonts reach into the lines above and below and would touch
    # the fraction and radical bars of other formulas
    ink = _spans_ink(formula.get("spans") or [])
    probe = old if ink is None or ink.is_empty else ink + (-_INK_PAD, -_INK_PAD, _INK_PAD, _INK_PAD)
    own = {_origin_key(span) for span in formula.get("spans") or [] if span.get("origin")}
    spans = formula.get("spans") or []
    curves = [
        curve
        for curve in intersecting_curves(probe, drawings)
        if _radical_bar(curve, spans) or not _rests_on(curve, glyphs, own)
    ]
    curves = [curve for curve in curves if not _joins_outer_shape(curve, curves, drawings, probe)]
    formula["curves"] = list(curves)
    if not curves:
        return
    box = expand_box(old, curves)
    baseline = old.y1 - float(formula.get("d") or 0)
    formula["bbox"] = box
    formula["w"] = round(box.width, 1)
    formula["h"] = round(box.height, 1)
    formula["d"] = round(max(0.0, box.y1 - baseline), 1)
    formula["dx"] = round(float(formula.get("dx") or 0) + (box.x0 - old.x0), 1)


# Images thinner than this are rules, not figures
_RULE_IMAGE_THICK = 1.5
# Margin kept outside the ink so that antialiasing and italic overhangs are not cut off. The same on both sides.
_FORMULA_SIDE_PAD = 0.45
_FORMULA_INK_DPI = 180


def _page_text_spans(page: pymupdf.Page) -> list[dict]:
    data = page.get_text("dict", flags=_TEXT_FLAGS)
    return [
        span for block in data["blocks"] if block.get("type") == 0 for line in block["lines"] for span in line["spans"]
    ]


def _page_glyphs(page: pymupdf.Page) -> list[dict]:
    """The spans of the page, with glyph ink."""
    data = page.get_text("dict", flags=_TEXT_FLAGS)
    _attach_ink(page, data)
    return [
        span for block in data["blocks"] if block.get("type") == 0 for line in block["lines"] for span in line["spans"]
    ]


def _span_ink_boxes(span: dict, char_index: _CharIndex) -> list[pymupdf.Rect]:
    """Boxes of the non-blank characters. Blanks have no ink; without character data the whole span counts as
    one box if it has text."""
    raw = span.get("bbox")
    if raw is None:
        return []
    box = pymupdf.Rect(raw)
    chars = char_index.find(box)
    if chars is not None:
        return [rect for text, rect in chars if text.strip()]
    if str(span.get("text") or "").strip():
        return [box]
    return []


def _ink_x_bounds(
    pix: pymupdf.Pixmap,
    clip: pymupdf.Rect,
    ours: list[pymupdf.Rect],
    foreign: list[pymupdf.Rect],
) -> tuple[float, float] | None:
    """Left and right bounds of this formula's ink. Neighboring text does not count; italic strokes that
    overhang the character box do."""
    width, height = pix.width, pix.height
    if width < 2 or height < 1:
        return None
    samples = pix.samples
    channels = pix.n
    step_x = clip.width / width
    step_y = clip.height / height

    def is_dark(x: int, y: int) -> bool:
        offset = (y * width + x) * channels
        if channels >= 3:
            return samples[offset] < 245 or samples[offset + 1] < 245 or samples[offset + 2] < 245
        return samples[offset] < 245

    def near(rects: list[pymupdf.Rect], px: float, py: float, pad: float) -> bool:
        return any(rect.x0 - pad <= px <= rect.x1 + pad and rect.y0 - pad <= py <= rect.y1 + pad for rect in rects)

    def is_ours(x: int, y: int) -> bool:
        px = clip.x0 + (x + 0.5) * step_x
        py = clip.y0 + (y + 0.5) * step_y
        # Ink inside a character box or just past it is ours. Ink further away comes from a neighbor reaching in
        # and must not stop the edge from moving in.
        if near(ours, px, py, 0.35):
            return True
        if any(rect.x0 <= px <= rect.x1 and rect.y0 <= py <= rect.y1 for rect in foreign):
            return False
        if not ours:
            return True
        return near(ours, px, py, 1.5)

    def column_has_ink(x: int) -> bool:
        return any(is_dark(x, y) and is_ours(x, y) for y in range(height))

    left = next((x for x in range(width) if column_has_ink(x)), None)
    if left is None:
        return None
    right = next(x for x in range(width - 1, left - 1, -1) if column_has_ink(x))
    return clip.x0 + left * step_x, clip.x0 + (right + 1) * step_x


def _tighten_formula_sides(
    part: dict,
    page: pymupdf.Page,
    page_spans: list[dict] | None = None,
    char_index: _CharIndex | None = None,
    display: pymupdf.DisplayList | None = None,
) -> None:
    """Shrink the left and right sides of the formula box to the same narrow margin outside the ink. The top,
    bottom and spacing inside the formula stay. display is the page's display list, shared by the formulas on
    the page so that the page is not parsed again for every crop."""
    raw = part.get("bbox")
    if raw is None:
        return
    bbox = pymupdf.Rect(raw)
    if bbox.width < 1.2 or bbox.height < 1:
        return
    if char_index is None:
        char_index = _page_char_index(page)
    if page_spans is None:
        page_spans = _page_text_spans(page)
    own_spans = [pymupdf.Rect(span["bbox"]) for span in part.get("spans") or [] if span.get("bbox") is not None]
    ours = [box for span in part.get("spans") or [] for box in _span_ink_boxes(span, char_index)]
    # Glyphs with wrong font boxes (radicals in txsys and CMSY) have boxes above their ink. Add the ink measured
    # by drawing them alone; otherwise the lower half of the radical counts as a neighbor's ink and the left
    # edge moves into the middle of the radical
    ours.extend(pymupdf.Rect(span["ink"]) for span in part.get("spans") or [] if span.get("ink_drawn"))
    ours.extend(pymupdf.Rect(curve) for curve in part.get("curves") or [] if curve is not None)
    foreign: list[pymupdf.Rect] = []
    for span in _outside_formula(page_spans, part.get("spans") or [], char_index):
        if not str(span.get("text") or "").strip() or span.get("bbox") is None:
            continue
        # Rule out distant spans by coordinates first: a page has thousands of spans to compare with each formula
        x0, y0, x1, y1 = span["bbox"]
        if y1 < bbox.y0 - 1 or y0 > bbox.y1 + 1 or x1 < bbox.x0 - 1 or x0 > bbox.x1 + 1:
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if any(_boxes_match(rect, own) for own in own_spans):
            continue
        foreign.append(rect)
    if page.rotation:
        return  # a rotated page renders rotated, and ink found by column would not match the text coordinates
    clip = bbox & text_page_rect(page)
    if clip.width < 1 or clip.height < 1:
        return
    if display is None:
        pix = page.get_pixmap(dpi=_FORMULA_INK_DPI, clip=clip, alpha=False)
    else:
        zoom = _FORMULA_INK_DPI / 72
        pix = display.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False, clip=clip)
    bounds = _ink_x_bounds(
        pix,
        clip,
        ours,
        foreign,
    )
    if bounds is None:
        return
    ink_x0, ink_x1 = bounds
    new_x0 = max(bbox.x0, ink_x0 - _FORMULA_SIDE_PAD)
    new_x1 = min(bbox.x1, ink_x1 + _FORMULA_SIDE_PAD)
    if new_x1 - new_x0 < 1.2:
        return
    if new_x0 <= bbox.x0 + 0.35 and new_x1 >= bbox.x1 - 0.35:
        return
    width = round(new_x1 - new_x0, 1)
    if new_x0 + width > bbox.x1:
        width = round(bbox.x1 - new_x0, 1)
    if width < 1.2:
        return
    part["bbox"] = pymupdf.Rect(new_x0, bbox.y0, new_x0 + width, bbox.y1)
    part["w"] = width
    if "dx" in part:
        part["dx"] = round(float(part.get("dx") or 0) + (new_x0 - bbox.x0), 1)


def _tighten_formula_height(part: dict) -> None:
    """Shrink the top and bottom of the formula box to a narrow margin outside the glyph ink.

    A box from the font ascent and descent is much taller than the glyphs. Layout scales a formula by its box
    height, so formulas with arrows or summation signs would come out very small, and cropping from the
    source page would also take in text from the lines above and below. Spans without measurable ink
    (bitmap glyphs) keep their original box.
    """
    raw = part.get("bbox")
    if raw is None:
        return
    bbox = pymupdf.Rect(raw)
    ink = _spans_ink(part.get("spans") or [])
    if ink is None:
        return
    for curve in part.get("curves") or []:
        if curve is not None:
            ink |= pymupdf.Rect(curve)
    if ink.is_empty:
        return
    top = max(bbox.y0, ink.y0 - _INK_PAD)
    bottom = min(bbox.y1, ink.y1 + _INK_PAD)
    if bottom - top < 1 or (top <= bbox.y0 + 0.3 and bottom >= bbox.y1 - 0.3):
        return
    part["bbox"] = pymupdf.Rect(bbox.x0, top, bbox.x1, bottom)
    part["h"] = round(bottom - top, 1)
    # The ink of a minus or equals sign lies wholly above the baseline. When the bottom is above the baseline,
    # d is negative and placement raises the image accordingly
    part["d"] = round(float(part.get("d") or 0) - (bbox.y1 - bottom), 1)


def _origin_key(span: dict) -> tuple[float, float]:
    return round(span["origin"][0], 1), round(span["origin"][1], 1)


# Glyph ink this close to a rule touches it: numerator and denominator sit one or two points from the bar
_RESTING_GAP = 2.0


def _joins_outer_shape(
    curve: pymupdf.Rect,
    curves: list[pymupdf.Rect],
    drawings: list[pymupdf.Rect],
    probe: pymupdf.Rect,
) -> bool:
    """The rule meets another line outside the formula end to end: it is one side of a shape such as a QED box
    or a table rule, not a fraction bar. Moving it would leave a gap in the box and an extra bar under the
    formula. Solid fills (the black background of a caption label) do not count."""
    if min(curve.width, curve.height) > _RULE_IMAGE_THICK:
        return False
    reach = curve + (-_INK_PAD, -_INK_PAD, _INK_PAD, _INK_PAD)
    room = probe + (-1, -1, 1, 1)
    for other in drawings:
        if other == curve or any(other == own for own in curves) or room.contains(other):
            continue
        if not (other & reach).is_empty:
            return True
    return False


def _radical_bar(curve: pymupdf.Rect, spans: list[dict]) -> bool:
    """The left end of the rule meets the top right corner of a glyph in this formula: it is a radical bar.
    The radicand is below the bar, and descenders of the line above often touch it from above, which must not
    keep it in place. Large CMEX radicals extract as letters, so they are recognized by position."""
    if curve.height > _RULE_IMAGE_THICK:
        return False
    for span in spans:
        if not str(span.get("text") or "").strip():
            continue
        box = pymupdf.Rect(span.get("ink") or span["bbox"])
        if abs(curve.x0 - box.x1) <= _RADICAL_JOIN and abs(curve.y0 - box.y0) <= _RADICAL_JOIN:
            return True
    return False


# Maximum distance (points) between the left end of a radical bar and the top right of the radical sign
_RADICAL_JOIN = 1.0


def _rests_on(curve: pymupdf.Rect, glyphs: list[dict], own: set[tuple[float, float]]) -> bool:
    """The rule touches text outside the formula above or below it. When the numerator is in another block
    that stays in place, the fraction bar is shared and must not move with the formula."""
    if curve.height > _RULE_IMAGE_THICK:
        return False
    for span in glyphs:
        found = span.get("ink")
        if (
            found is None
            or not span.get("origin")
            or _origin_key(span) in own
            or not str(span.get("text") or "").strip()
        ):
            continue
        ink = pymupdf.Rect(found)
        if min(ink.x1, curve.x1) - max(ink.x0, curve.x0) < 1:
            continue
        if -0.5 <= curve.y0 - ink.y1 <= _RESTING_GAP or -0.5 <= ink.y0 - curve.y1 <= _RESTING_GAP:
            return True
    return False


def _spans_ink(spans: list[dict]) -> pymupdf.Rect | None:
    """Union of the glyph ink of several spans. Returns None if any span has no measurable ink (bitmap glyphs)."""
    ink = pymupdf.Rect()
    for span in spans:
        if not str(span.get("text") or "").strip():
            continue
        found = span.get("ink")
        if found is None:
            return None
        found = pymupdf.Rect(found)
        if not found.is_empty:
            ink |= found
    return ink


def _absorb_curves(
    formula: dict,
    drawings: list[pymupdf.Rect],
    block: TextBlock,
    glyphs: list[dict] = (),
) -> None:
    """Add intersecting curves to the formula box and update the matching FormulaRun. Each piece of a formula
    broken across lines takes its own curves."""
    _absorb_one_box(formula, drawings, glyphs)
    for part in formula.get("parts") or []:
        if part is not formula:
            _absorb_one_box(part, drawings, glyphs)
    block.runs = [
        replace(
            run,
            bbox=pymupdf.Rect(formula["bbox"]),
            curves=tuple(formula.get("curves") or ()),
            descent=formula["d"],
            dx=formula.get("dx", run.dx),
        )
        if isinstance(run, FormulaRun) and run.name == formula["name"]
        else run
        for run in block.runs
    ]


def _marker_png(idx: int) -> bytes:
    """A 1x1 colored pixel whose color encodes the global formula index (R + G<<8), used to find where the
    formula landed after insertion. It must be unique across the document: MuPDF merges marker images of the
    same color into one xref, and every formula would then land at the first one."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1))
    pix.set_pixel(0, 0, ((idx + 1) & 0xFF, ((idx + 1) >> 8) & 0xFF, 200))
    return pix.tobytes("png")


def _crop_formulas(
    src_doc: pymupdf.Document,
    flat_units: list[list[TextBlock]],
    sent_formulas: list[list[dict]],
) -> pymupdf.Archive:
    # A formula is cropped from the source page it is on. The page of the unit's last block would be wrong
    # for a unit across two pages. Curves intersecting the box join the crop so radicals and fraction bars
    # are not cut off.
    archive = pymupdf.Archive()
    curve_rects = [_page_curve_rects(page) for page in src_doc]
    glyphs: dict[int, list[dict]] = {}
    for unit, formulas in zip(flat_units, sent_formulas):
        background.check_stop()
        for f in formulas:
            page_no = int(f["page"])
            if page_no not in glyphs:
                glyphs[page_no] = _page_glyphs(src_doc[page_no])
            _absorb_curves(f, curve_rects[page_no], unit[int(f.get("owner") or 0)], glyphs[page_no])
    span_cache: dict[int, list[dict]] = {}
    char_cache: dict[int, _CharIndex] = {}
    # Keep only the display list of the latest page: formulas come roughly in page order, and keeping every
    # page would use too much memory
    display: tuple[int, pymupdf.DisplayList] | None = None
    for formulas in sent_formulas:
        background.check_stop()
        for f in formulas:
            pieces = _formula_parts(f)
            ready = False
            for part in pieces:
                # Leading spaces in character boxes leave wide sides on some formulas and none on others.
                # Shrink to the ink first.
                part_page = int(part.get("page") or f["page"])
                if part_page not in span_cache:
                    span_cache[part_page] = _page_text_spans(src_doc[part_page])
                    char_cache[part_page] = _page_char_index(src_doc[part_page])
                if display is None or display[0] != part_page:
                    display = part_page, src_doc[part_page].get_displaylist()
                _tighten_formula_height(part)
                _tighten_formula_sides(
                    part,
                    src_doc[part_page],
                    span_cache[part_page],
                    char_cache[part_page],
                    display[1],
                )
                bbox = pymupdf.Rect(part["bbox"])
                if bbox.width < 1 or bbox.height < 1:
                    continue
                part["png"] = _FormulaImage(src_doc, part_page, bbox, part.get("spans") or ())
                part["has_img"] = True
                ready = True
            f["has_img"] = ready
    # Marker pixels encode the global index (MuPDF would merge same-colored markers into one image and mix up
    # the placements)
    marker_idx = 0
    for formulas in sent_formulas:
        for f in formulas:
            if f.get("has_img"):
                f["mid"] = marker_idx
                archive.add(_marker_png(marker_idx), f"ph_{f['name']}")
                marker_idx += 1
    return archive
