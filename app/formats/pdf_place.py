"""Lay out translated text in a block's box: line breaking, inline formula images, and
shrinking the line height or font size when the text does not fit."""

import html
import re

import pymupdf

from .pdf_align import FIRST_BASELINE, SQUEEZE, draw_cjk_lines
from .pdf_blocks import _formula_parts
from .pdf_drawings import text_page_rect
from .pdf_flow import (
    balance_marks,
    measure_width,
    peel_leading_space,
    restore_emphasis,
    separate_after_formula,
    space_around_latin,
    strip_style_marks,
)
from .pdf_place_formulas import _FormulaSources, _paint_formula_parts, _part_lead
from .pdf_typeset import (
    FORMULA_PAD,
    FormulaMetric,
    FormulaPiece,
    TextPiece,
    TypesetLine,
    line_lead,
    line_offsets,
    typeset_lines,
)

# Maximum height of an inline formula, in em. Up to this height, formulas with radicals or sums with limits are
# written at their original size and the line spacing grows with them
_INLINE_FORMULA_EM = 2.6


# Height a formula is reduced to when the box cannot hold the enlarged line spacing: exactly one line
_LINE_FORMULA_EM = 1.35


# Fractions and sums under a radical usually reach less than 1.5 em above the baseline and less than 1 em below it
_FORMULA_ASCENT_EM = 1.8


_FORMULA_DESCENT_EM = 1.3


def _formula_metrics(
    formulas: list[dict],
    block_size: float,
    em: float,
    reference: float | None = None,
) -> dict[int, FormulaMetric]:
    """Give the slot of each formula at the current font size. rs is written back to the record, so the vector
    scaling and the slot use the same number. reference: see _formula_slot."""
    metrics: dict[int, FormulaMetric] = {}
    for index, formula in enumerate(formulas, 1):
        scale, slot_width, height = _formula_slot(formula, block_size, em, reference)
        formula["rs"] = scale
        body = float(formula["w"]) * scale if formula.get("has_img") else slot_width
        below = (float(formula.get("d") or 0) - float(formula.get("raise") or 0)) * scale
        metrics[index] = FormulaMetric(slot_width, body, height, below)
    return metrics


def _fit_typeset(
    text: str,
    formulas: list[dict],
    rect: "pymupdf.Rect",
    em: float,
    block_size: float,
    cjk: bool,
    page: "pymupdf.Page",
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None,
    holes: list["pymupdf.Rect"] = (),
    lower: tuple[float, float] | None = None,
    room: float | None = None,
    left_limit: float | None = None,
    lent: bool = False,
    hang: float = 0.0,
) -> tuple[list[TypesetLine], float, float, "pymupdf.Rect", list[tuple[float, float]]]:
    """Typeset at the original font size if the text fits. If not, first extend right within the column, then
    reach down into blank space, then tighten the letter spacing slightly, then the line spacing, and only then
    shrink the font size.
    holes are figures the source text flows around. Lines that touch them move to one side, and the indents and
    widths of those lines are returned as well.
    lower is (desired origin, top of the block below): when the text does not fill the box, the top moves down, at
    most to the desired origin. The bottom of the box reaches past the last source line, and text moved down must
    not touch the block below.
    room is the top of the block below (see _room_below). When the bottom extends downward, half a line is left
    between the last line and the block below, the same as the gap between paragraphs.
    left_limit is how far left a right-aligned block can extend. When given, the box extends left, not right.
    When lent is true, room is the lowest origin of the paragraph below after it lends space (see spare_below). That
    paragraph moves to half a line below the last line, so the blank part below the last line must not extend past
    the box bottom (see _glyph_bottom).
    hang is how many points the source body of a hanging-indent item is indented more than its number (see
    _hang_indent)."""
    gap = 1.45 if cjk else 1.2
    reference = block_size * 0.88 if cjk else block_size
    box = rect
    # Extending to here leaves half a line between the bottom of the last line box and room. The box is taller than
    # the line boxes by the 0.3em overhang above the first line
    reach = None if room is None else room - em * (gap * 0.5 - 0.3)
    expanded = False
    grown = False
    squeezed = False
    tightened = False
    capped = False
    # Width to break lines at. With tightened letter spacing it is a little wider than the box, and each line is
    # compressed back to the box width when written (see SQUEEZE)
    squeeze = 1.0
    for formula in formulas:
        formula.pop("cap", None)
    lines: list[TypesetLine] = []
    lanes: list[tuple[float, float]] = []
    for _ in range(24):
        lanes = _hole_lanes(box, holes, em, gap)
        wrapped = bool(lanes)
        if hang:
            lanes = _hang_lanes(box, lanes, hang, _label_width(text, em), em, gap, len(text))
        metrics = _formula_metrics(formulas, block_size, em, reference)
        lines = typeset_lines(text, em, max(box.width, 1.0) * squeeze, metrics, [width for _, width in lanes])
        extra = line_offsets(lines, em, gap)[1]
        height = _box_height(lines, em, gap)
        overflows = any(
            line.width > (lanes[index][1] if index < len(lanes) else box.width * squeeze) + 1
            for index, line in enumerate(lines)
        )
        # The blank part below the last line extends past the box bottom only to keep the normal line spacing. If
        # the text still does not fit with tightened line spacing, measure against the box bottom as before and
        # shrink the font size if needed.
        # With a single line there is no line spacing to keep: tightening the line spacing does not move the
        # glyphs, the bottom of the last line box is a little higher, and the paragraph below can move up
        relax = cjk and len(lines) > 1 and room is not None and not tightened and not lent
        bottom = _glyph_bottom(box.y1, room, em, gap) if relax else box.y1
        too_tall = box.y0 + height > bottom + 1
        if not overflows and not too_tall:
            # Lines that wrap around figures were computed from the figure positions, so the box cannot move any
            # more; a hanging indent moves with the box
            if lower is not None and not wrapped:
                top, floor = lower
                lowered = min(top, min(box.y1, floor) - height)
                if lowered > box.y0:
                    box = pymupdf.Rect(box.x0, lowered, box.x1, box.y1)
            return lines, em, gap, box, lanes
        page_width = text_page_rect(page).width
        if not expanded and box.width < page_width * 0.6:
            expanded = True
            if left_limit is not None:
                wide = _expand_left(box, left_limit, obstacles)
            else:
                wide = _expand_right(box, page_width, obstacles, right_limit)
            if wide.width > box.width + 1:
                box = wide
                continue
        if too_tall and not overflows and not grown and reach is not None and reach > box.y1 + 0.5:
            grown = True
            box = pymupdf.Rect(box.x0, box.y0, box.x1, min(reach, box.y0 + height))
            continue
        # When the last line has only a few characters, tightening the letter spacing slightly pulls that line
        # back, which is less visible than tightening the line spacing
        if too_tall and not overflows and not squeezed and cjk and not lanes:
            squeezed = True
            if len(typeset_lines(text, em, max(box.width, 1.0) * (1 + SQUEEZE), metrics)) < len(lines):
                squeeze = 1 + SQUEEZE
                continue
        if not tightened:
            gap = 1.1 if gap <= 1.25 else gap - 0.25
            tightened = True
            continue
        # If the lines opened up by tall formulas still do not fit with tightened line spacing, reduce the formulas
        # back to one line height and keep the font size
        if too_tall and extra > 0 and not capped:
            capped = True
            for formula in formulas:
                formula["cap"] = _LINE_FORMULA_EM
            continue
        if em <= 1.5:
            break
        em = max(1.5, em * 0.9)
    return lines, em, gap, box, lanes


def _box_height(lines: list[TypesetLine], em: float, gap: float) -> float:
    """Height of the box the typeset lines need. The glyph tops of the first line add a 0.3em overhang (measured:
    MuPDF box content height = lines × line spacing + ~0.25em); counting only the line spacing makes a borderline
    box get silently shrunk to 0.93 when written"""
    return max(len(lines), 1) * em * gap + em * 0.3 + line_offsets(lines, em, gap)[1]


def _glyph_bottom(bottom: float, room: float, em: float, gap: float) -> float:
    """How far a Chinese paragraph drawn glyph by glyph can write in its box (measured by _box_height). The line
    spacing below the glyphs of the last line is blank and may extend past the box bottom; half of that spacing is
    still left before room, so the glyphs do not touch the block below. When the box bottom already extends past
    room (table of contents entries one right after another), it still writes to the box bottom. When a paragraph
    carried over to the next page is split into lines, this blank part also extends past room (see
    _filled_bottom)"""
    return max(bottom, min(bottom + (gap - 1) * em, room + (gap - 1) * em * 0.5))


def _typeset_end(lines: list[TypesetLine], em: float, gap: float, box: "pymupdf.Rect") -> float:
    """Bottom of the last line box of the typeset lines"""
    return box.y0 + max(len(lines), 1) * em * gap + line_offsets(lines, em, gap)[1]


# Clause punctuation after which a centered title may break when it wraps
_CLAUSE_MARKS = "，、；：,;:"


def _clause_lines(
    text: str,
    formulas: list[dict],
    em: float,
    block_size: float,
    width: float,
    lines: list[TypesetLine],
    reference: float | None = None,
) -> list[TypesetLine]:
    """When a centered title or subtitle wraps onto several lines, break after clause punctuation so that one or
    two characters are not left alone on a line. There may not be more lines than with breaking by width; when
    several punctuation marks work, take the break whose widest line is the narrowest."""
    metrics = _formula_metrics(formulas, block_size, em, reference)
    best, best_width = lines, None
    for index, char in enumerate(text[:-1]):
        if char not in _CLAUSE_MARKS:
            continue
        head, tail = text[: index + 1].rstrip(), text[index + 1 :].strip()
        if not head or not tail:
            continue
        # When an author list that is bold throughout breaks at a comma, the second half must reopen the bold
        head, tail = balance_marks(head, tail)
        parts = typeset_lines(head, em, width, metrics) + typeset_lines(tail, em, width, metrics)
        widest = max(line.width for line in parts)
        if len(parts) > len(lines) or widest > width + 1:
            continue
        if best_width is None or widest < best_width:
            best, best_width = parts, widest
    return best


def _hole_lanes(
    box: "pymupdf.Rect",
    holes: list["pymupdf.Rect"],
    em: float,
    gap: float,
) -> list[tuple[float, float]]:
    """Indent from the left edge of the box and width of the first lines. Lines whose glyphs fall within the height
    of a figure move to the other side of the figure, half a character from it. Lines below the figure are not
    listed and use the full box width."""
    if not holes:
        return []
    pitch = em * gap
    bottom = max(hole.y1 for hole in holes)
    lanes: list[tuple[float, float]] = []
    keep = 0
    top = box.y0
    while top < bottom:
        # Glyphs occupy about 0.88em above the baseline and 0.12em below
        glyph_top = top + em * (FIRST_BASELINE - 0.88)
        glyph_bottom = top + em * (FIRST_BASELINE + 0.12)
        x0, x1 = box.x0, box.x1
        for hole in holes:
            if min(hole.y1, glyph_bottom) - max(hole.y0, glyph_top) <= 0 or hole.x0 >= x1 or hole.x1 <= x0:
                continue
            if hole.x0 - box.x0 > box.x1 - hole.x1:
                x1 = min(x1, hole.x0 - em * 0.5)
            else:
                x0 = max(x0, hole.x1 + em * 0.5)
        lanes.append((x0 - box.x0, max(x1 - x0, em)))
        if x0 > box.x0 or x1 < box.x1:
            keep = len(lanes)
        top += pitch
    return lanes[:keep]


# Number or bullet at the start of the translation, with the blank after it
_HANG_LABEL_RE = re.compile(r"\s*(?:[•◦▪‣·○■□►»]|[-–—*]|\(?(?:\d+(?:\.\d+)*|[a-zA-Z]|[ivxlc]+)[.)]?)\s+")


def _label_width(text: str, em: float) -> float:
    """Width of the number or bullet at the start of the translation together with the blank after it;
    continuation lines start here. Returns 0 when there is no number"""
    matched = _HANG_LABEL_RE.match(strip_style_marks(text))
    return measure_width(matched.group(0), em) if matched else 0.0


def _hang_lanes(
    box: "pymupdf.Rect",
    lanes: list[tuple[float, float]],
    hang: float,
    label: float,
    em: float,
    gap: float,
    chars: int,
) -> list[tuple[float, float]]:
    """Lines of a hanging indent; lines that wrap around figures also move clear of them. The body aligns at hang
    to the right of the box's left edge, in the same column as the description paragraphs that follow the item;
    label is the width of the translation's number with its blank (see _label_width), and the first line moves
    right so that the body after the number also starts at this column. When the number is wider than hang, the
    column moves right with it. The number of lines is unknown before breaking and each line holds at least one
    character, so one entry is listed per character"""
    column = max(hang, label)
    out = []
    for index in range(max(chars, 1)):
        indent, width = lanes[index] if index < len(lanes) else (0.0, box.width)
        shift = column - label if index == 0 and label else column if index else 0.0
        if indent < shift:
            width, indent = max(width - (shift - indent), em), shift
        out.append((indent, width))
    return out


def _line_baseline(
    spans: list[dict],
    line_top: float,
    line_h: float,
    em: float,
    cjk: bool,
    slot_x: float | None = None,
) -> float:
    """Origin of the text already written on this line. A line with only formulas has no text and uses the
    empirical htmlbox baseline. Glyph boxes of adjacent lines overlap, so text closer to the slot wins; this keeps
    a formula that moved to the next line from staying at its old x."""
    best: tuple[tuple[float, float], float] | None = None
    for span in spans:
        origin = span.get("origin")
        bbox = span.get("bbox")
        if not origin or not bbox:
            continue
        if bbox[3] < line_top - 1 or bbox[1] > line_top + line_h + 1:
            continue
        overlap = min(bbox[3], line_top + line_h) - max(bbox[1], line_top)
        if overlap <= 0:
            continue
        dx = 0.0
        if slot_x is not None:
            if bbox[2] < slot_x:
                dx = slot_x - bbox[2]
            elif bbox[0] > slot_x:
                dx = bbox[0] - slot_x
        near = overlap if dx <= em * 4 else overlap * 0.2
        key = (near, -dx)
        if best is None or key > best[0]:
            best = (key, float(origin[1]))
    if best is not None:
        return best[1]
    return line_top + em * (1.12 if cjk else 1.0)


def _typeset_line_html(line: TypesetLine, formulas: list[dict]) -> str:
    """HTML of one line. Formulas without an image fall back to the text in their record."""
    parts: list[str] = []
    for piece in line.pieces:
        if isinstance(piece, TextPiece):
            if piece.text:
                parts.append(_inline_marks(restore_emphasis(html.escape(piece.text))))
        elif isinstance(piece, FormulaPiece) and 1 <= piece.index <= len(formulas):
            formula = formulas[piece.index - 1]
            if not formula.get("has_img"):
                parts.append(_inline_marks(html.escape(str(formula.get("text") or ""))))
    return "".join(parts)


def _formula_draw_args(formulas: list[dict]) -> tuple[frozenset[int], dict[int, str]]:
    images = frozenset(index for index, formula in enumerate(formulas, 1) if formula.get("has_img"))
    fallback = {
        index: str(formula.get("text") or "") for index, formula in enumerate(formulas, 1) if index not in images
    }
    return images, fallback


def _draw_cjk(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    lines: list[TypesetLine],
    formulas: list[dict],
    box: "pymupdf.Rect",
    em: float,
    gap: float,
    color: str,
    align: str,
    forms: "_FormulaSources | None",
    page_spans: list[dict] | None,
    deferred_png: list[tuple["pymupdf.Rect", bytes]],
    justify_last: bool = False,
    lanes: list[tuple[float, float]] = (),
    first_room: float | None = None,
) -> bool:
    """Justify Chinese lines by glyph and draw each formula at its position in the content stream (see
    draw_cjk_lines). Returns False when glyphs are missing."""
    images, fallback = _formula_draw_args(formulas)

    def paint(slot) -> str:
        if not 1 <= slot.index <= len(formulas):
            return ""
        drawn: list[str] = []
        _paint_formula_parts(
            page,
            doc,
            formulas[slot.index - 1],
            slot.x,
            slot.baseline,
            forms,
            page_spans,
            drawn,
            deferred_png,
        )
        return "\n".join(drawn)

    slots = draw_cjk_lines(
        page,
        doc,
        lines,
        box,
        em,
        gap,
        color,
        align,
        images,
        fallback,
        justify_last,
        lanes,
        first_room,
        paint,
    )
    return slots is not None


def _place_plain(
    page: "pymupdf.Page",
    text: str,
    formulas: list[dict],
    rect: "pymupdf.Rect",
    em: float,
    block_size: float,
    cjk: bool,
    align: str,
    div_attrs: str,
    color: str,
    weight: str,
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None,
    archive: pymupdf.Archive | None,
    justify_last: bool = False,
    holes: list["pymupdf.Rect"] = (),
    first_room: float | None = None,
    lower: tuple[float, float] | None = None,
    room: float | None = None,
    left_limit: float | None = None,
    lent: bool = False,
    hang: float = 0.0,
) -> float | None:
    """Break lines with the typesetter, then write them. Chinese is justified by glyph; when glyphs are missing, it
    falls back to one HTML box. lower, room, left_limit, lent, hang: see _fit_typeset. Returns the bottom of the
    last line box, or None when nothing was written."""
    text = space_around_latin(separate_after_formula(text))
    lines, em, gap, box, lanes = _fit_typeset(
        text,
        formulas,
        rect,
        em,
        block_size,
        cjk,
        page,
        obstacles,
        right_limit,
        holes,
        lower,
        room,
        left_limit,
        lent,
        hang,
    )
    if box.y0 > rect.y0:
        # The origin moved down, so the first line is no longer raised to match the first source line
        first_room = None
    end = _typeset_end(lines, em, gap, box)
    if align == "center" and len(lines) > 1 and not any(indent for indent, _ in lanes):
        lines = _clause_lines(text, formulas, em, block_size, box.width, lines, block_size * 0.88 if cjk else None)
    if cjk and _draw_cjk(
        page,
        page.parent,
        lines,
        formulas,
        box,
        em,
        gap,
        color,
        align,
        None,
        None,
        [],
        justify_last,
        lanes,
        first_room,
    ):
        return end
    rendered = [_typeset_line_html(line, formulas) for line in lines]
    if any(indent for indent, _ in lanes):
        # With a figure on the left, the lines beside it are indented individually. MuPDF ignores inline margin and
        # padding, so each line is written as a block
        parts = [
            f'<div style="margin-left:{lanes[index][0] if index < len(lanes) else 0:.2f}px">'
            f'<span style="white-space:nowrap">{line_html}</span></div>'
            for index, line_html in enumerate(rendered)
        ]
        body = "".join(parts) if any(rendered) else ""
    else:
        parts = [f'<span style="white-space:nowrap">{line_html}</span>' for line_html in rendered if line_html]
        body = "<br>".join(parts)
    if not body:
        return None
    css = (
        f"* {{font-family: sans-serif; font-size: {em}px; color: {color}; "
        f"font-weight: {weight}; line-height: {gap}; margin: 0; padding: 0; text-align: {align};}}"
    )
    html_text = f"<div{div_attrs}>{body}</div>"
    if cjk:
        # Whether the text fits is measured by the glyphs of the last line (see _glyph_bottom). The HTML box needs
        # the full line-spacing height, otherwise the whole paragraph gets shrunk
        box = pymupdf.Rect(box.x0, box.y0, box.x1, max(box.y1, box.y0 + _box_height(lines, em, gap)))
    if not _try_insert(page, box, html_text, css, 0.9, archive):
        _insert_fitting(page, box, html_text, css, obstacles, archive, right_limit)
    return end


def _written_right(page: "pymupdf.Page", x0: float, y0: float, y1: float) -> float:
    """Right edge of the text just written for this entry. When the estimated width is off, the next slot follows
    this edge. Only glyphs whose baseline lies in this box count. Glyph boxes of the line above reach into the top
    of the box, and their right edge would push the formula to the column edge."""
    right = x0
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                box = span.get("bbox")
                origin = span.get("origin")
                if not box or not origin or not str(span.get("text") or "").strip():
                    continue
                baseline = float(origin[1])
                if baseline < y0 or baseline > y1 + 1 or float(box[0]) < x0 - 1:
                    continue
                right = max(right, float(box[2]))
    return right


def _place_typeset(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    block_size: float,
    text: str,
    formulas: list[dict],
    rect: "pymupdf.Rect",
    em: float,
    cjk: bool,
    align: str,
    div_attrs: str,
    color: str,
    weight: str,
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None,
    forms: "_FormulaSources | None",
    page_spans: list[dict] | None,
    deferred_png: list[tuple["pymupdf.Rect", bytes]],
    justify_last: bool = False,
    holes: list["pymupdf.Rect"] = (),
    first_room: float | None = None,
    lower: tuple[float, float] | None = None,
    room: float | None = None,
    left_limit: float | None = None,
    lent: bool = False,
    hang: float = 0.0,
) -> float:
    """Write the text line by line, and paste formula slots onto the line's baseline by their recorded bottom.
    lower, room, hang and the return value: see _place_plain."""
    text = space_around_latin(separate_after_formula(text))
    lines, em, gap, box, lanes = _fit_typeset(
        text,
        formulas,
        rect,
        em,
        block_size,
        cjk,
        page,
        obstacles,
        right_limit,
        holes,
        lower,
        room,
        left_limit,
        lent,
        hang,
    )
    if box.y0 > rect.y0:
        first_room = None
    end = _typeset_end(lines, em, gap, box)
    if cjk and _draw_cjk(
        page,
        doc,
        lines,
        formulas,
        box,
        em,
        gap,
        color,
        align,
        forms,
        page_spans,
        deferred_png,
        justify_last,
        lanes,
        first_room,
    ):
        return end
    css = (
        f"* {{font-family: sans-serif; font-size: {em}px; color: {color}; "
        f"font-weight: {weight}; line-height: {gap}; margin: 0; padding: 0; text-align: left;}}"
    )
    line_h = em * gap
    # Each formula records the last content stream after the text before it is written. Once the baseline is
    # measured, the formula is drawn after that stream, so drag selection finds it between the text around it
    pending: list[tuple[FormulaPiece, float, float, int | None]] = []
    offsets, _extra = line_offsets(lines, em, gap)
    for index, line in enumerate(lines):
        y = box.y0 + index * line_h + offsets[index]
        indent, width = lanes[index] if index < len(lanes) else (0.0, box.width)
        origin_x = box.x0 + indent - (line_lead(line) if align == "left" else 0.0)
        if align != "left" and line.width < width:
            origin_x += width - line.width if align == "right" else (width - line.width) / 2
        # Latin letters are estimated at 0.5em, so narrow words like i-th come out too wide. A formula that followed
        # the estimate would be pushed away. Text continues from the right edge actually written, and only the
        # slot's own 2pt is left on each side.
        cursor = origin_x
        after_formula = False
        for piece in line.pieces:
            if isinstance(piece, TextPiece):
                if not piece.text.strip():
                    continue
                body, had_space = peel_leading_space(piece.text)
                if had_space and not after_formula:
                    cursor += em * 0.33
                fragment = _inline_marks(restore_emphasis(html.escape(body)))
                html_text = f'<div{div_attrs}><span style="white-space:nowrap">{fragment}</span></div>'
                # The box is widened from the estimate so narrow words are not shrunk; placement uses the right edge
                # written, not this width.
                slot = pymupdf.Rect(
                    cursor,
                    y,
                    cursor + max(piece.width, 4) + 4,
                    y + line_h + em * 0.3,
                )
                if not _try_insert(page, slot, html_text, css, 0.9):
                    try:
                        page.insert_htmlbox(slot, html_text, css=css, scale_low=0)
                    except AssertionError:
                        pass
                written = _written_right(page, cursor, slot.y0, slot.y1)
                cursor = written if written > cursor + 0.4 else cursor + piece.width
                after_formula = False
            elif isinstance(piece, FormulaPiece) and 1 <= piece.index <= len(formulas):
                pending.append((piece, cursor, y, _last_contents(doc, page)))
                cursor += piece.slot_width
                after_formula = True
    if not pending:
        return end
    spans = [
        span
        for blk in page.get_text("dict")["blocks"]
        if blk.get("type") == 0
        for ln in blk["lines"]
        for span in ln["spans"]
        if span.get("text", "").strip() and span.get("origin")
    ]
    after_stream: dict[int | None, list[str]] = {}
    for piece, slot_x, line_top, after in pending:
        baseline = _line_baseline(spans, line_top, line_h, em, cjk, slot_x)
        _paint_formula_parts(
            page,
            doc,
            formulas[piece.index - 1],
            slot_x,
            baseline,
            forms,
            page_spans,
            after_stream.setdefault(after, []),
            deferred_png,
        )
    for after, drawn in after_stream.items():
        if drawn:
            _insert_contents_after(page, doc, after, "\n".join(drawn))
    return end


def _last_contents(doc: "pymupdf.Document", page: "pymupdf.Page") -> int | None:
    """xref of the last content stream in the page's Contents."""
    refs = re.findall(r"(\d+) 0 R", doc.xref_get_key(page.xref, "Contents")[1])
    return int(refs[-1]) if refs else None


def _insert_contents_after(page: "pymupdf.Page", doc: "pymupdf.Document", after: int | None, commands: str) -> None:
    """Create a content stream and put it after the stream after in Contents, or last when after is not found.

    The position is found by xref: the first write of insert_htmlbox adds a q stream at the front, so a recorded
    index would be off."""
    stream = doc.get_new_xref()
    doc.update_object(stream, "<<>>")
    doc.update_stream(stream, ("\n" + commands + "\n").encode())
    refs = [int(ref) for ref in re.findall(r"(\d+) 0 R", doc.xref_get_key(page.xref, "Contents")[1])]
    at = refs.index(after) + 1 if after in refs else len(refs)
    refs.insert(at, stream)
    doc.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{ref} 0 R" for ref in refs) + "]")


def _formula_slot(
    formula: dict,
    block_size: float,
    render_size: float,
    reference: float | None = None,
) -> tuple[float, float, float]:
    """Inline formula slot (scale, width, height). A height above _INLINE_FORMULA_EM is reduced to it, and a line
    with a taller formula opens up its line spacing (see line_offsets). When the box cannot hold it, _fit_typeset
    records cap on the formula, which reduces it to one line height.
    Parts split by line breaks are laid out horizontally in reading order with their widths added, so the space
    between the two lines is not cropped into one box.
    reference is the translation font size at which the formula keeps its original size. Chinese is set 10%
    smaller than the source font size (so it looks as large as the source), and a formula scaled by render_size /
    block_size would become 10% smaller too and look small among Chinese characters. For Chinese the scale is
    taken against reference, so without font shrinking the formula keeps its original size."""
    parts = _formula_parts(formula)
    height = max(float(part.get("h") or 0) for part in parts)
    if not formula.get("has_img") or height <= 0 or block_size <= 0:
        text = str(formula.get("text") or "")
        return 1.0, len(text) * render_size * 0.5, render_size
    cap = float(formula.get("cap") or _INLINE_FORMULA_EM)
    # A formula reaching far above or below the baseline has most likely taken in glyphs of an adjacent line and has
    # the wrong baseline. Enlarging it would only cover other lines, so it is still reduced to one line height
    ascent = max(float(part.get("h") or 0) - float(part.get("d") or 0) for part in parts)
    descent = max(float(part.get("d") or 0) for part in parts)
    if ascent > _FORMULA_ASCENT_EM * block_size or descent > _FORMULA_DESCENT_EM * block_size:
        cap = min(cap, _LINE_FORMULA_EM)
    scale = min(render_size / (reference or block_size), cap * render_size / height)
    gaps = sum(_part_lead(part, scale) for part in parts[1:])
    width = sum(float(part["w"]) * scale for part in parts) + gaps + 2 * FORMULA_PAD
    return scale, width, height * scale


def _inline_marks(body: str) -> str:
    return (
        body.replace("\x01s\x02", "<sup>")
        .replace("\x01/s\x02", "</sup>")
        .replace("\x01b\x02", "<sub>")
        .replace("\x01/b\x02", "</sub>")
    )


def _expand_right(
    rect: "pymupdf.Rect",
    page_width: float,
    obstacles: list["pymupdf.Rect"],
    right_limit: float | None = None,
) -> "pymupdf.Rect":
    """First shrink stage of typesetting: extend right without covering text or figures on the right that overlap
    vertically.

    The limit is 90% of the page width, or the column edge when given and smaller. The result never ends left of
    the original right edge.
    """
    x1 = page_width * 0.9
    if right_limit is not None:
        x1 = min(x1, right_limit)
    for obstacle in obstacles:
        if obstacle.x0 <= rect.x0 + 1:
            continue
        if obstacle.y0 < rect.y1 and obstacle.y1 > rect.y0:
            x1 = min(x1, obstacle.x0 - 2)
    return pymupdf.Rect(rect.x0, rect.y0, max(x1, rect.x1), rect.y1)


def _expand_left(rect: "pymupdf.Rect", x0: float, obstacles: list["pymupdf.Rect"]) -> "pymupdf.Rect":
    """Extend a right-aligned block to the left, at most to x0, without covering text or figures on the left that
    overlap vertically. The result never starts right of the original left edge."""
    for obstacle in obstacles:
        if obstacle.x1 >= rect.x1 - 1:
            continue
        if obstacle.y0 < rect.y1 and obstacle.y1 > rect.y0:
            x0 = max(x0, obstacle.x1 + 2)
    return pymupdf.Rect(min(x0, rect.x0), rect.y0, rect.x1, rect.y1)


def _try_insert(
    page: "pymupdf.Page", rect: "pymupdf.Rect", html_text: str, css: str, scale_low: float, archive=None
) -> bool:
    """insert_htmlbox returns a negative value when the text does not fit (and draws nothing). When the scale is
    exactly scale_low, pymupdf can trip an assert from floating-point error (0.8999999999999999 < 0.9); this is
    treated as not fitting."""
    try:
        return page.insert_htmlbox(rect, html_text, css=css, scale_low=scale_low, archive=archive)[0] >= 0
    except AssertionError:
        return False


def _tighten_line_height(css: str) -> str:
    """Tighten the line spacing by one step. The CJK style uses 1.45, so a hard-coded replacement of 1.2 would
    have no effect."""
    match = re.search(r"line-height:\s*([0-9.]+)", css)
    if match is None:
        return css
    current = float(match.group(1))
    tight = 1.1 if current <= 1.25 else current - 0.25
    return f"{css[: match.start(1)]}{tight:g}{css[match.end(1) :]}"


def _insert_fitting(
    page: "pymupdf.Page",
    rect: "pymupdf.Rect",
    html_text: str,
    css: str,
    obstacles: list["pymupdf.Rect"],
    archive=None,
    right_limit: float | None = None,
):
    """Three shrink stages of typesetting (the BabelDOC approach): extend right → tighten the line spacing → only
    then shrink the font size.

    Each stage moves on to the next only when the text does not fit (insert_htmlbox returns a negative value and
    draws nothing).
    Extending right applies only to blocks clearly narrower than the page width (labels, captions, short lines):
    a full-column paragraph extended right would cross the column boundary, and without shrinking the font after
    extending, CJK line boxes (1.31em) are taller than the box (computed with the Latin 1.16em), overflow downward
    and cover the next line of content. right_limit is the right edge of this column, and extension may not cross it.
    """
    if _try_insert(page, rect, html_text, css, 0.9, archive):
        return
    page_width = text_page_rect(page).width
    wide = _expand_right(rect, page_width, obstacles, right_limit) if rect.width < page_width * 0.6 else rect
    if wide.x1 > rect.x1 + 1 and _try_insert(page, wide, html_text, css, 0.9, archive):
        return
    target = wide
    css_tight = _tighten_line_height(css)
    if _try_insert(page, target, html_text, css_tight, 0.9, archive):
        return
    # Fallback: as before, shrink until the text fits
    page.insert_htmlbox(target, html_text, css=css_tight, scale_low=0, archive=archive)
