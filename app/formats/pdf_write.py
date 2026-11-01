"""Erase the original text of each page and write the translations in its place."""

import bisect
import html
import math
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf

from ..languages import RTL_LANGUAGES
from .pdf_align import SQUEEZE
from .pdf_blocks import (
    _AFFIL_MARK_RE,
    _LABEL_START_RE,
    _MARK_ITEM_RE,
    _SENTINEL_NUM_RE,
    SPILL_TAIL,
    Leader,
    TextBlock,
    _attach_ink,
    _color_value,
    _column_left,
    _diagramish,
    _ends_sentence,
    _formula_parts,
    _is_side_label,
    _join_lines,
    _list_item,
    _numbered_line,
    _page_char_index,
    _pages_to_write,
    _paragraph_indent,
    _same_page_line_pair,
    _span_is_math,
    _visual_lines,
    _with_source_bold,
)
from .pdf_erase import Erasure, erase, restore_glyphs, single_precision, type3_glyphs
from .pdf_flow import (
    STYLE_MARK_RE,
    balance_marks,
    break_lines,
    measure_width,
    nowrap_lines,
    restore_emphasis,
    separate_after_formula,
    space_around_latin,
    span_emphasis,
    writer_for,
)
from .pdf_layout import (
    PageGeometry,
    content_floor,
    is_page_number,
    text_page_rect,
    visible_drawings,
)
from .pdf_order import ReadingOrder
from .pdf_place import (
    _box_height,
    _expand_left,
    _formula_metrics,
    _formula_slot,
    _FormulaSources,
    _glyph_bottom,
    _inline_marks,
    _insert_fitting,
    _paint_formula,
    _place_plain,
    _place_typeset,
)
from .pdf_type3 import broken_type3_names
from .pdf_typeset import (
    typeset_lines,
)

# Tolerance for the line centers on a page set centered throughout. A comma at the end of a line or an italic overhang
# shifts the center by a point or two
_AXIS_TOL = 2.0


def _attach_chars(page: pymupdf.Page, spans: list[dict]) -> None:
    """Record the per-character boxes of each span (span["chars"]), matched by the span's box. Once the page has
    been erased they can no longer be looked up."""
    index = {_rect_key(rect): chars for rect, chars in _page_char_index(page)}
    for span in spans:
        chars = index.get(_rect_key(pymupdf.Rect(span["bbox"])))
        if chars is not None:
            span["chars"] = chars


def _rect_key(rect: pymupdf.Rect) -> tuple[float, float, float, float]:
    return (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))


def _white_pixmap(doc) -> "pymupdf.Pixmap":
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1))
    pix.set_pixel(0, 0, (255, 255, 255))
    return pix


def _page_form(doc: "pymupdf.Document", page: "pymupdf.Page") -> int:
    """Pack the page content into a Form XObject (called before redaction) and return its xref.
    Formula pass-through references it: it uses the page's own resources, and the BBox is the whole page."""
    mb = page.mediabox
    fm = doc.get_new_xref()
    doc.update_object(
        fm,
        (
            f"<< /Type /XObject /Subtype /Form /BBox [{mb.x0} {mb.y0} {mb.x1} {mb.y1}] "
            f"/Matrix [1 0 0 1 0 0] /Resources {doc.xref_get_key(page.xref, 'Resources')[1]} >>"
        ),
    )
    doc.update_stream(fm, page.read_contents())
    return fm


def _place_formula_images(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    before: set[int],
    mid_map: dict[int, dict],
    forms: "_FormulaSources | None" = None,
    page_spans: list[dict] | None = None,
):
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
    draws = []
    # Text spans after rendering: htmlbox aligns img only by its bottom, and the marker bottom has a systematic
    # offset of ~1pt from the text baseline, so anchoring the formula baseline to the origin of text on the same
    # line is the most reliable
    text_spans = [
        s
        for blk in page.get_text("dict")["blocks"]
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


def _closes_clause(text: str) -> bool:
    """Ends with sentence-final punctuation, a semicolon or a colon (list items often end with a semicolon).
    Trailing digits are footnote numbers and do not count"""
    text = text.rstrip().rstrip("0123456789")
    return _ends_sentence(text) or text.endswith((";", "；", ":", "："))


def _row_neighbor(block: TextBlock, blocks: list[TextBlock], geo: PageGeometry | None) -> bool:
    """Another block sits beside this one on the same row in this column (another table cell). Margin notes and
    body text in another column do not count"""
    column = geo.index_of(block.rect) if geo is not None and len(geo.columns) >= 2 else None
    return any(
        other is not block
        and other.page == block.page
        and not other.side_note
        and (column is None or geo.index_of(other.rect) == column)
        and not (other.rect.x0 < block.rect.x1 and block.rect.x0 < other.rect.x1)
        and min(other.rect.y1, block.rect.y1) - max(other.rect.y0, block.rect.y0)
        > min(other.rect.height, block.rect.height) * 0.5
        for other in blocks
    )


def _wrap_holes(
    block: TextBlock,
    obstacles: list["pymupdf.Rect"],
    area: "pymupdf.Rect | None" = None,
) -> list["pymupdf.Rect"]:
    """Figures and captions the source text flows around. They lie in the area to write (the block's box by
    default) against its left or right edge, no line of the block overlaps them, and at least two lines beside them
    make room. Writing the whole block as one box would put the translation over the figure."""
    lines = block.line_rects
    if len(lines) < 3:
        return []
    area = area or block.rect
    found: list[pymupdf.Rect] = []
    beside: set[int] = set()
    for obstacle in obstacles:
        cut = pymupdf.Rect(obstacle) & area
        if cut.is_empty or cut.height < block.size * 0.8:
            continue
        # The other side must leave 30% of the width; formulas and tables across the whole block do not count
        right = obstacle.x1 >= area.x1 - block.size and obstacle.x0 > area.x0 + area.width * 0.3
        left = obstacle.x0 <= area.x0 + block.size and obstacle.x1 < area.x1 - area.width * 0.3
        if not (right or left):
            continue
        rows: set[int] = set()
        for index, line in enumerate(lines):
            if min(line.y1, obstacle.y1) - max(line.y0, obstacle.y0) < line.height * 0.5:
                continue
            if (right and line.x1 > obstacle.x0 + 1) or (left and line.x0 < obstacle.x1 - 1):
                break
            rows.add(index)
        else:
            found.append(obstacle)
            beside |= rows
    return found if len(beside) >= 2 else []


def _first_row_room(block: TextBlock, rect: "pymupdf.Rect") -> float | None:
    """Height from the top of the write box to the bottom of the first source line box. None when the box does not
    start at the first line (the half of a paragraph carried over from the previous page)."""
    if not block.line_rects:
        return None
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    bottom = max(r.y1 for r in rows[0])
    if not block.rect.y0 - block.size <= rect.y0 < bottom:
        return None
    return bottom - rect.y0


# Start of a hanging-indent item: a number (3.1, 2.), a bullet, or a label like (a)
_HANG_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*\.?\s")


def _hang_indent(block: TextBlock) -> float:
    """For an item whose number or bullet starts at the margin and whose following lines align with the body after
    the number (exercise 3.1 Write out …): how many points continuation lines are indented more than the first.
    Returns 0 for blocks not set this way"""
    # The number of a section heading (1.1 What is a Large Language Model?) is also spaced off from the title; the
    # number stays where it is
    if block.body_size and block.size > block.body_size * 1.1:
        return 0.0
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    if len(rows) == 1:
        return _label_gap(block)
    if not rows:
        return 0.0
    first = min(rect.x0 for rect in rows[0])
    rest = [min(rect.x0 for rect in row) for row in rows[1:]]
    hang = min(rest) - first
    if max(rest) - min(rest) > block.size * 0.5 or not block.size * 0.5 < hang <= block.size * 5:
        return 0.0
    lead = block.text.lstrip()
    return hang if _list_item(lead) or _HANG_NUMBER_RE.match(lead) else 0.0


def _label_gap(block: TextBlock) -> float:
    """For a one-line item whose number, in the body font size, stands apart with at least half a character of space
    before the body (3.8 | Write a program …): how many points the body is indented more than the number. Returns 0
    for items not set this way"""
    spans = sorted(
        (
            span
            for line in block.span_lines or []
            for span in line
            if str(span.get("text") or "").strip() and span.get("bbox") is not None
        ),
        key=lambda span: float(span["bbox"][0]),
    )
    if (
        len(spans) < 2
        or not _HANG_NUMBER_RE.match(str(spans[0]["text"]).strip() + " ")
        and not _list_item(str(spans[0]["text"]).strip() + " x")
    ):
        return 0.0
    # A footnote's superscript number is also followed by a space, and footnote continuation lines start at the margin
    if float(spans[0].get("size") or 0) < block.size * 0.9:
        return 0.0
    gap = float(spans[1]["bbox"][0]) - float(spans[0]["bbox"][2])
    hang = float(spans[1]["bbox"][0]) - float(spans[0]["bbox"][0])
    return hang if gap >= block.size * 0.5 and hang <= block.size * 5 else 0.0


def _extend_leader(page: "pymupdf.Page", block: TextBlock, leader: Leader) -> None:
    """The leader was erased together with the heading. Add dots leftward from the page number with the original
    dot pitch and baseline, up to the end of the translation."""
    if leader.pitch <= 0:
        return
    band = pymupdf.Rect(block.rect.x0 - 1, block.rect.y0 - 1, leader.number.x0 - 0.5, block.rect.y1 + 1)
    right = band.x0
    for item in page.get_text("dict", clip=band)["blocks"]:
        for line in item.get("lines", []):
            for span in line["spans"]:
                # Entries above and below are close, and their boxes reach into this line. Only glyphs with their
                # baseline on this line count
                if str(span.get("text") or "").strip() and abs(span["origin"][1] - leader.baseline) < leader.size * 0.6:
                    right = max(right, float(span["bbox"][2]))
    if right <= band.x0:
        return
    dots: list[float] = []
    x = leader.dot_x
    while x > right + leader.pitch * 0.5:
        dots.append(x)
        x -= leader.pitch
    if not dots:
        return
    font = pymupdf.Font("tiro")
    writer = pymupdf.TextWriter(page.rect)
    for x in dots:
        writer.append((x, leader.baseline), ".", font=font, fontsize=leader.size)
    color = ((leader.color >> 16 & 0xFF) / 255, (leader.color >> 8 & 0xFF) / 255, (leader.color & 0xFF) / 255)
    writer.write_text(page, color=color)


def _flush_png(page: "pymupdf.Page", deferred_png: list[tuple["pymupdf.Rect", bytes]]) -> None:
    """Formulas whose vector pass-through failed and fell back to bitmaps are pasted after the whole page is written."""
    for target, png in deferred_png:
        page.insert_image(target, stream=png, keep_proportion=False, overlay=True)
    deferred_png.clear()


def _nowrap_parens(body: str) -> str:
    """Wrap short parenthesized groups in a nowrap span: MuPDF's CJK line breaking can break anywhere, producing
    bad lines like "（如\\n序列）" broken inside the parentheses, or "（y_1^{t-1}" with the closing parenthesis left
    alone on the next line. A group is kept whole (in a group with a formula image, the parentheses move with the
    formula); long groups are not wrapped (to avoid overflow)."""

    def repl(m):
        inner = m.group(0)
        # Estimated rendered width: <img> counts as 3 characters, other tags count as none
        plain = re.sub(r"<[^>]+>", lambda t: "@@@" if t.group(0).startswith("<img") else "", inner)
        if len(plain) <= 10:
            return f'<span style="white-space: nowrap">{inner}</span>'
        return inner

    return re.sub(r"（[^（）]+）|\([^()]+\)", repl, body)


def _translation_html(
    text: str,
    formulas: list[dict],
    block_size: float,
    render_size: float,
    max_width: float,
) -> str:
    """HTML of the translation. With formula sentinels it breaks the lines itself and makes each line nowrap;
    otherwise MuPDF does the wrapping."""

    def img_repl(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if not 1 <= number <= len(formulas):
            return ""
        formula = formulas[number - 1]
        if formula.get("has_img"):
            scale, width, height = _formula_slot(formula, block_size, render_size)
            formula["rs"] = scale
            # Leave 2pt on each side of the slot: italic glyphs go slightly past their bbox and, set tight, look
            # pressed by the Chinese characters around them
            return f'<img src="ph_{formula["name"]}" style="width: {width:.1f}px; height: {height:.1f}px;">'
        return _inline_marks(html.escape(str(formula.get("text") or "")))

    def one(fragment: str) -> str:
        escaped = _inline_marks(restore_emphasis(html.escape(fragment)))
        return re.sub(r"\x01i\x02(\d+)\x01/i\x02", img_repl, escaped)

    if "\x01i\x02" not in text:
        return _nowrap_parens(one(text).replace("\n", "<br>"))
    widths = {index: _formula_slot(formula, block_size, render_size)[1] for index, formula in enumerate(formulas, 1)}
    return nowrap_lines([one(line) for line in break_lines(text, render_size, max_width, widths)])


def _centered_in_cell(block: TextBlock) -> bool:
    """The source text is centered in its cell. The column center does not match the cell center, so the column
    cannot be used to decide."""
    clip = block.clip
    if clip is None:
        return False
    cell_mid = (clip[0] + clip[2]) / 2
    text_mid = (block.rect.x0 + block.rect.x1) / 2
    return abs(text_mid - cell_mid) <= 2.0


def _flow_y0(block: TextBlock, cjk: bool) -> float:
    """Where writing starts. A first line indented for a side label is written by a separate box from its own left
    edge; here writing still starts at the first line."""
    origin = block.line_rects[0].y0 if block.line_rects else block.rect.y0
    return origin + block.size * 0.1 if cjk else origin


def _inset_write_boxes(block: TextBlock, cjk: bool) -> tuple[pymupdf.Rect, pymupdf.Rect] | None:
    """When the first line starts further right than the following lines, the first line is written from its own
    left edge and the rest from the column's left edge.

    Continuation starts at the next visual line. If fragments on the same baseline were taken as the next line, the
    continued text would cover the first line and the label on its left.
    """
    lines = _visual_lines(block)
    if len(lines) < 2 or block.size <= 0:
        return None
    first_line = lines[0]
    first_left = min(rect.x0 for rect in first_line)
    rest_left = min(rect.x0 for line in lines[1:] for rect in line)
    if first_left <= rest_left + max(block.size * 2, 16):
        return None
    right = max(rect.x1 for rect in block.line_rects) + 2
    origin = min(rect.y0 for rect in first_line)
    y0 = origin + block.size * 0.1 if cjk else origin
    size = block.size * 0.88 if cjk else block.size
    line_gap = 1.45 if cjk else 1.2
    first_box = pymupdf.Rect(first_left, y0, right, y0 + size * line_gap + size * 0.3)
    rest_origin = min(rect.y0 for rect in lines[1])
    rest_y = rest_origin + block.size * 0.1 if cjk else rest_origin
    pad = block.size * 0.5 if cjk else block.size * 0.3
    rest_box = pymupdf.Rect(rest_left, rest_y, right, block.rect.y1 + pad)
    return _clip_write_rect(first_box, block.clip), _clip_write_rect(rest_box, block.clip)


def _short_left_label(block: TextBlock) -> bool:
    """A short label to the left of the first body line. Memo., Property Enforcement., (2) Statistics Derivation."""
    text = block.text.strip()
    if not text.endswith(".") or len(text.split()) > 6 or block.rect.width > 150:
        return False
    return not _diagramish(block)


def _label_beside(body: TextBlock, blocks: list[TextBlock]) -> TextBlock | None:
    """A short label beside the first body line whose source text nearly touches the body's left edge."""
    if not body.line_rects:
        return None
    first = body.line_rects[0]
    found: TextBlock | None = None
    for other in blocks:
        if other is body or other.page != body.page or not _short_left_label(other):
            continue
        height = min(other.rect.height, first.height) or 1.0
        overlap = min(other.rect.y1, first.y1) - max(other.rect.y0, first.y0)
        if overlap <= height * 0.5:
            continue
        gap = first.x0 - other.rect.x1
        if -2 <= gap < 40 and (found is None or other.rect.x0 < found.rect.x0):
            found = other
    return found


def _body_first_starts(
    blocks: list[TextBlock],
    items: list[tuple[TextBlock, str]],
    cjk: bool,
    formulas_map: dict[int, list[dict]] | None,
) -> dict[int, float]:
    """Fill the first-line space left free by a short translated label. When the translation is wider than the
    English, the body moves to the right of the label."""
    out: dict[int, float] = {}
    by_id = {id(block): text for block, text in items}
    known = formulas_map or {}
    for block, _text in items:
        inset = _inset_write_boxes(block, cjk)
        if inset is None:
            continue
        first_box = inset[0]
        label = _label_beside(block, blocks)
        if label is None:
            continue
        translation = by_id.get(id(label))
        if translation is None or translation.strip() == label.text.strip():
            continue
        formulas = known.get(id(label), [])
        if _SENTINEL_NUM_RE.search(translation) and not formulas:
            continue
        em = label.size * 0.88 if cjk else label.size
        metrics = _formula_metrics(formulas, label.size, em, em)
        widths = {index: metric.slot_width for index, metric in metrics.items()}
        start = label.rect.x0 + measure_width(translation, em, widths) + 8.0
        if start >= first_box.x1 - max(em * 4, 24):
            continue
        if abs(start - first_box.x0) < 1:
            continue
        out[id(block)] = start
    return out


def _beside_body(block: TextBlock, blocks: list[TextBlock]) -> TextBlock | None:
    """Body text to the right of a short side label, on the same baseline. The label may not extend into this line."""
    if not _is_side_label(block):
        return None
    found: TextBlock | None = None
    for other in blocks:
        if other is block or other.page != block.page or not other.line_rects:
            continue
        first = other.line_rects[0]
        height = min(block.rect.height, first.height) or 1.0
        overlap = min(block.rect.y1, first.y1) - max(block.rect.y0, first.y0)
        if overlap <= height * 0.5:
            continue
        gap = first.x0 - block.rect.x1
        if -2 <= gap < 40 and (found is None or first.x0 < found.line_rects[0].x0):
            found = other
    return found


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
        if b.index and id(b) not in texts and not any(_span_is_math(s) for line in b.span_lines for s in line)
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


def _source_marked(block: TextBlock) -> str:
    """The source text with bold and italic marks written out span by span. When a block that keeps the original
    text is rewritten, bold page numbers stay bold."""
    spans = [s for line in block.span_lines for s in line]
    writer = writer_for(spans, block.size, _color_value(block.color))
    return _join_lines(
        [
            "".join(
                writer.text(str(s["text"]), span_emphasis(s), float(s.get("size") or 0), s.get("color")) for s in line
            )
            + writer.close()
            for line in block.span_lines
        ]
    )


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


def _affiliation_write_rects(blocks: list[TextBlock]) -> dict[int, pymupdf.Rect]:
    """Write author affiliations that sit side by side one per line instead. Chinese is wider than the source, and
    side by side they would overlap."""
    marked = [
        block
        for block in blocks
        if block.clip is None
        and block.rect.height <= 32
        and len(block.line_rects) <= 2
        and _AFFIL_MARK_RE.match(block.text.lstrip())
    ]
    used: set[int] = set()
    out: dict[int, pymupdf.Rect] = {}
    for block in marked:
        if id(block) in used:
            continue
        group = [block]
        used.add(id(block))
        grew = True
        while grew:
            grew = False
            for other in marked:
                if id(other) in used or other.page != block.page:
                    continue
                if any(
                    min(item.rect.y1, other.rect.y1) - max(item.rect.y0, other.rect.y0)
                    >= min(item.rect.height, other.rect.height) * 0.5
                    and 0 <= other.rect.x0 - item.rect.x1 < 40
                    for item in group
                ) or any(
                    min(item.rect.y1, other.rect.y1) - max(item.rect.y0, other.rect.y0)
                    >= min(item.rect.height, other.rect.height) * 0.5
                    and 0 <= item.rect.x0 - other.rect.x1 < 40
                    for item in group
                ):
                    group.append(other)
                    used.add(id(other))
                    grew = True
        if len(group) < 2:
            continue
        group.sort(key=lambda item: item.rect.x0)
        top = min(item.rect.y0 for item in group)
        left = min(item.rect.x0 for item in group)
        right = max(item.rect.x1 for item in group)
        members = {id(item) for item in group}

        def overlaps_row(other: TextBlock) -> bool:
            return other.rect.x1 > left - 2 and other.rect.x0 < right - 4

        above = [
            other.rect.y1
            for other in blocks
            if other.page == block.page
            and id(other) not in members
            and other.rect.y1 <= top + 1
            and overlaps_row(other)
        ]
        if above:
            top = max(max(above) + 1, top - 8)
        below = [
            other.rect.y0
            for other in blocks
            if other.page == block.page and id(other) not in members and other.rect.y0 > top + 4 and overlaps_row(other)
        ]
        floor = min(below) - 3 if below else top + 12 * len(group)
        slot = max(9.0, (floor - top) / len(group))
        y = top
        for item in group:
            out[id(item)] = pymupdf.Rect(left, y, max(right, left + 80), y + slot)
            y += slot
    return out


def _row_span(block: TextBlock, others: list[TextBlock], col_x0: float, col_x1: float) -> pymupdf.Rect:
    """Horizontal extent of this block together with the blocks next to it on the same row in the column (less
    than 1.5 character widths apart). When a line is split into several blocks, the whole line counts: a heading
    split into section number and section name is centered as a whole line; the cells of a table row together are
    not in the middle of the column. For a multi-line block, every line must have a block on the same row; a block
    whose second line, below the table title, sits on the row of the table header still uses its own extent."""
    rows = block.line_rects or [block.rect]

    def same_row(rect: pymupdf.Rect, row: pymupdf.Rect) -> bool:
        return min(rect.y1, row.y1) - max(rect.y0, row.y0) > min(rect.height, row.height) * 0.5

    mates = [
        other.rect
        for other in others
        if other is not block
        and other.page == block.page
        and col_x0 - 2 <= other.rect.x0
        and other.rect.x1 <= col_x1 + 2
        and any(same_row(other.rect, row) for row in rows)
    ]
    span = pymupdf.Rect(block.rect)
    if not all(any(same_row(mate, row) for mate in mates) for row in rows):
        return span
    grown = True
    while grown:
        grown = False
        for mate in mates:
            near = span.x0 - block.size * 1.5 <= mate.x1 and mate.x0 <= span.x1 + block.size * 1.5
            if near and (mate.x0 < span.x0 or mate.x1 > span.x1):
                span = pymupdf.Rect(min(span.x0, mate.x0), span.y0, max(span.x1, mate.x1), span.y1)
                grown = True
    return span


def _left_aligned_stack(block: TextBlock, others: list[TextBlock]) -> bool:
    """A block right above or below shares the left edge but not the right edge: this is a left-aligned group of
    lines (dialogue, example sentences), and a line whose center happens to fall near the column center is not
    centered. Two centered lines of the same width align on both sides and are not included."""
    for other in others:
        if other is block or other.page != block.page:
            continue
        if abs(other.rect.x0 - block.rect.x0) > 1 or abs(other.rect.x1 - block.rect.x1) <= 2:
            continue
        gap = max(other.rect.y0 - block.rect.y1, block.rect.y0 - other.rect.y1)
        if gap < block.size * 2:
            return True
    return False


def _centered_axis(block: TextBlock, others: list[TextBlock]) -> float | None:
    """Covers and title pages are centered throughout, and most of their lines of varying width share one center
    line. A block is centered when each of its lines is on this center line, which is returned.
    Lines about as wide as the widest line do not count, because the body lines in the text frame all share one
    center line; margin notes widen the horizontal extent of the page's text, so the comparison is with the widest
    line. Lines that also share the left edge are a column of numbers of equal width and are not centered."""
    page = [other for other in others if other.page == block.page]
    lines = [line for other in page for line in (other.line_rects or [other.rect])]
    if not lines:
        return None
    widest = max(line.width for line in lines)
    short = sorted(
        (line for line in lines if line.width < widest * 0.85),
        key=lambda line: (line.x0 + line.x1) / 2,
    )
    if len(short) < 3:
        return None
    mids = [(line.x0 + line.x1) / 2 for line in short]
    group: list[pymupdf.Rect] = []
    for i, mid in enumerate(mids):
        hi = bisect.bisect_right(mids, mid + 2 * _AXIS_TOL)
        if hi - i > len(group):
            group = short[i:hi]
    if len(group) < max(3, len(short) * 0.6):
        return None
    if max(line.x0 for line in group) - min(line.x0 for line in group) < block.size:
        return None
    axis = sum((line.x0 + line.x1) / 2 for line in group) / len(group)
    own = block.line_rects or [block.rect]
    if any(abs((line.x0 + line.x1) / 2 - axis) > _AXIS_TOL for line in own):
        return None
    return axis


def _write_rect(
    block: TextBlock,
    items: list[tuple[TextBlock, str]],
    cjk: bool,
    rtl: bool,
    geo: PageGeometry | None,
) -> tuple[pymupdf.Rect, str]:
    """Write box and alignment (left, center, right). With several columns and the block inside one of them,
    centering refers to that column; with one column it refers to the text frame if there is one, otherwise to the
    widest block on the page. Text originally centered in a table cell is centered on the cell center instead.
    A block starting at the column's left edge is left-aligned, even if its center happens to match."""
    y0 = _flow_y0(block, cjk)
    pad = block.size * 0.5 if cjk else block.size * 0.3
    if _centered_in_cell(block) and block.clip is not None:
        clip = block.clip
        rect = pymupdf.Rect(clip[0], y0, clip[2], block.rect.y1 + pad)
        return _clip_write_rect(rect, clip), "center"
    column = None
    if geo is not None and len(geo.columns) >= 2:
        candidate = geo.column_of(block.rect)
        if block.rect.x0 >= candidate.x0 - 2 and _inside_right(block, candidate.x1):
            column = candidate
    # Right alignment refers to the edges of the column or text frame, not the right edge of the widest block
    margin: tuple[float, float] | None = None
    if column is not None:
        col_x0, col_x1 = column.x0, column.x1
        col_width = col_x1 - col_x0
        margin = col_x0, col_x1
    elif (
        geo is not None
        and geo.frame is not None
        and (geo.frame.x0 - 2 <= block.rect.x0 and _inside_right(block, geo.frame.x1))
    ):
        col_x0, col_x1 = geo.frame.x0, geo.frame.x1
        col_width = col_x1 - col_x0
        margin = col_x0, col_x1
    else:
        if geo is not None and len(geo.columns) == 1:
            only = geo.columns[0]
            if only.x0 - 2 <= block.rect.x0 and _inside_right(block, only.x1):
                margin = only.x0, only.x1
        widest = max((other.rect for other, _ in items), key=lambda rect: rect.width, default=block.rect)
        col_x0, col_x1 = widest.x0, widest.x1
        col_width = widest.width
    # Text larger than most on the page with every line on the page center line is a centered heading: on a
    # slide the widest block is a left-aligned list item whose center is not the page center; when the heading
    # itself is the widest block, it is compared with the page width. Body text of a paper has symmetric margins,
    # so full lines are also on the page center line, but its font size is not large, so it does not count.
    page_mid = geo.width / 2 if geo is not None and column is None and geo.width > 0 else None
    sizes = sorted(other.size for other, _ in items)
    on_page_mid = (
        page_mid is not None
        and block.size > sizes[len(sizes) // 2] * 1.15
        and block.rect.width < geo.width * 0.85
        and all(abs((line.x0 + line.x1) / 2 - page_mid) < 4 for line in block.line_rects or [block.rect])
    )
    # List items starting with a bullet or dash are left-aligned, even if their center happens to match the column
    # center. The same holds for index entries, whose indented subentries often fall right in the middle of the
    # column. Centered headings such as the "(a)" subfigure caption or "1. 引言" are not included.
    # When a line is split into several blocks, the whole line must be on the column center too (see _row_span):
    # the cells of a table row written on the column center would overlap
    col_mid = (col_x0 + col_x1) / 2
    span = _row_span(block, [other for other, _ in items], col_x0, col_x1)
    centered = (
        not rtl
        and not block.index
        and len(block.text) < 100
        and not _left_aligned_stack(block, [other for other, _ in items])
        and (
            (
                block.rect.width < col_width * 0.85
                and block.rect.x0 > col_x0 + 2
                and abs((block.rect.x0 + block.rect.x1) / 2 - col_mid) < 15
                and abs((span.x0 + span.x1) / 2 - col_mid) < 15
            )
            or on_page_mid
        )
        and not _MARK_ITEM_RE.match(block.text)
    )
    right = (
        margin is not None
        and not centered
        and not rtl
        and _flush_right(block, [other for other, _ in items], geo.figures if geo is not None else (), *margin)
    )
    axis = None
    # A long subtitle with several lines on the page's center line and a single font size is also centered. The
    # last line of a body paragraph is short and does not match the center line; author names followed by smaller
    # affiliations are laid out line by line in other ways
    long_title = (
        len(block.line_rects) >= 2
        and len(
            {
                round(span.get("size", block.size))
                for line in block.span_lines or []
                for span in line
                if str(span.get("text") or "").strip()
            }
        )
        == 1
    )
    if (
        not centered
        and not right
        and not rtl
        and not block.index
        and (len(block.text) < 100 or long_title)
        and not _MARK_ITEM_RE.match(block.text)
    ):
        axis = _centered_axis(block, [other for other, _ in items])
    if axis is not None:
        # With asymmetric page margins neither the text frame nor the widest block is on this center line, so the
        # width comes from the distance between the center line and the sides of the page's text
        left = min(other.rect.x0 for other, _ in items if other.page == block.page)
        right = max(other.rect.x1 for other, _ in items if other.page == block.page)
        half = max(block.rect.width / 2, min(axis - left, right - axis))
        rect = pymupdf.Rect(axis - half, y0, axis + half, block.rect.y1 + pad)
        return _clip_write_rect(rect, block.clip), "center"
    if centered and on_page_mid:
        half = max(block.rect.width / 2, min(page_mid - col_x0, col_x1 - page_mid))
        rect = pymupdf.Rect(page_mid - half, y0, page_mid + half, block.rect.y1 + pad)
    elif centered:
        rect = pymupdf.Rect(col_x0, y0, col_x1, block.rect.y1 + pad)
    elif right:
        return _clip_write_rect(
            pymupdf.Rect(block.rect.x0, y0, block.rect.x1, block.rect.y1 + pad), block.clip
        ), "right"
    else:
        # Lines reaching into the margin do not count, and the translation breaks at the right edge of the column
        # or text frame. When margin notes form a column of their own, these lines also push out the right edge
        # of the body column
        edges = [
            edge for edge in (margin[1] if margin is not None else None, _frame_x1(block, geo)) if edge is not None
        ]
        x1 = min(edges) if edges and block.rect.x1 > min(edges) + 2 else block.rect.x1
        rect = pymupdf.Rect(block.rect.x0, y0, x1 + 2, block.rect.y1 + pad)
    return _clip_write_rect(rect, block.clip), "center" if centered else "left"


def _inside_right(block: TextBlock, x1: float) -> bool:
    """The block ends within x1. It also counts when only a few lines push the block's right edge out: a long word
    or citation that TeX cannot break makes a line reach into the margin, as long as the paragraph has at least as
    many lines ending at x1 as lines reaching past it."""
    if block.rect.x1 <= x1 + 2:
        return True
    out = [row.x1 for row in block.line_rects if row.x1 > x1 + 2]
    held = sum(abs(row.x1 - x1) <= 2 for row in block.line_rects)
    return bool(out) and max(out) >= block.rect.x1 - 1 and len(out) <= held


def _frame_x1(block: TextBlock, geo: PageGeometry | None) -> float | None:
    """Right edge of the text frame when the block is inside it."""
    frame = geo.frame if geo is not None else None
    if frame is not None and frame.x0 - 2 <= block.rect.x0 and _inside_right(block, frame.x1):
        return frame.x1
    return None


def _flush_right(
    block: TextBlock,
    others: list[TextBlock],
    figures: tuple[pymupdf.Rect, ...],
    col_x0: float,
    col_x1: float,
) -> bool:
    """A right-aligned block: an epigraph or attribution at the start of a chapter. Every line ends at the column's
    right edge, and the left edge is more than a quarter of the column width from the column's left edge.
    An attribution set by TeX often ends with an interword space, and italics add slant correction, so the glyphs
    end about half a character width short of the column's right edge.
    A wider block on the page must reach the column's right edge; when this block alone defines the column, the
    right edge is its own.
    Two lines starting at the same x mean left-aligned body text, even with space on the left (body text wrapping
    around a figure has empty space on the left when the figure is on the previous page).
    It does not count when another block or figure lies to the left on the same row of the column: the right part
    of a line split into several blocks, or body text wrapping around a figure.
    Lines with another block within a few character widths to the left do not count either: the last word split
    off a line in a narrow justified column. An attribution pushed to the end of the epigraph's last line with
    \\hfill is far from the glyphs to its left.
    A left edge aligned with the start of the lines above or below does not count either: left-aligned text such
    as table headers and lists."""
    if block.index or block.clip is not None or block.rect.x0 < col_x0 + (col_x1 - col_x0) * 0.25:
        return False
    rows = block.line_rects or [block.rect]
    if any(not -1.5 <= col_x1 - row.x1 <= max(1.5, block.size * 0.6) for row in rows):
        return False
    starts = sorted(row.x0 for row in rows)
    if any(later - earlier < 0.5 for earlier, later in zip(starts, starts[1:])):
        return False
    near = [other for other in others if other is not block and other.page == block.page]
    if not any(other.rect.x0 < block.rect.x0 - 1 and abs(other.rect.x1 - col_x1) <= 1.5 for other in near):
        return False
    other_rows = [row for other in near for row in other.line_rects or [other.rect]]
    if any(
        abs(row.x0 - block.rect.x0) <= 1 and max(row.y0 - block.rect.y1, block.rect.y0 - row.y1) < block.size * 2
        for row in other_rows
    ):
        return False
    close = [row for row in other_rows if row.x1 > block.rect.x0 - block.size * 3]
    return not any(
        col_x0 < rect.x1 <= block.rect.x0 + 1
        and min(rect.y1, block.rect.y1) - max(rect.y0, block.rect.y0) > block.size * 0.3
        for rect in [*(other.rect for other in near), *close, *figures]
    )


def _clip_write_rect(
    rect: pymupdf.Rect,
    clip: tuple[float, float, float, float] | None,
) -> pymupdf.Rect:
    """The left and right sides of a cell are hard limits. Vertically the line box height is kept.

    For a line against the bottom of the cell, trimming the margin below would make the line box shorter than a
    line. The typesetter would then take the height as too small and shrink the font size of this cell alone,
    leaving mixed sizes in one table.
    """
    if clip is None:
        return rect
    bounded = pymupdf.Rect(
        max(rect.x0, clip[0] + 0.6),
        rect.y0,
        min(rect.x1, clip[2] - 0.8),
        rect.y1,
    )
    if bounded.width < 4 or bounded.height < 4:
        return rect
    return bounded


def _write_limit(
    block: TextBlock,
    rect: pymupdf.Rect,
    geo: PageGeometry | None,
    obstacles: list[pymupdf.Rect],
) -> float | None:
    limit = geo.right_limit(rect, obstacles) if geo is not None else None
    frame_x1 = _frame_x1(block, geo)
    if limit is not None and frame_x1 is not None:
        # A paragraph in the text frame extends right only up to the frame's right edge. Lines reaching into the
        # margin push out the column's right edge
        limit = min(limit, frame_x1 + 2)
    if block.clip is not None:
        cap = block.clip[2] - 0.8
        limit = cap if limit is None else min(limit, cap)
    return limit


@dataclass(frozen=True)
class _Carry:
    """The second half written on the continuation page. An ordinary carried-over paragraph is broken by column
    width; a half sentence sharing a baseline is broken again by the width of its continuation line.
    size is the font size of the previous block: with lines broken at that size, a tail block with a slightly
    larger size cannot hold them, and one character is pushed out at the end of each line"""

    text: str
    formulas: list[dict]
    width: float
    size: float
    own_line: bool = False


def _page_content_bottom(
    blocks: list[TextBlock],
    page: int,
    height: float,
    ink: Sequence[pymupdf.Rect] = (),
) -> float:
    """Bottom of the source body on this page. Blocks in the page number band do not count; the translation is not
    written below this line. ink is the glyphs and figure boxes that count as body and stay in place (see
    _body_ink)"""
    floor = content_floor(height)
    bottoms = [block.rect.y1 for block in blocks if block.page == page and block.rect.y0 < floor - 4]
    bottoms += [rect.y1 for rect in ink if rect.y0 < floor - 4]
    if not bottoms:
        return floor
    return max(bottoms)


def _is_column_top(block: TextBlock, blocks: list[TextBlock]) -> bool:
    """The topmost block of its column. The top of the page is not moved down, so the top margin matches the
    source."""
    for other in blocks:
        if other is block or other.page != block.page or other.rect.y0 >= block.rect.y0 - 0.5:
            continue
        overlap = min(other.rect.x1, block.rect.x1) - max(other.rect.x0, block.rect.x0)
        if overlap > 8:
            return False
    return True


def _keeps_line_pad(block: TextBlock) -> bool:
    """Only one or two lines at the end of the page. With the line box margin trimmed, Chinese cannot fit one line,
    and the font size would be shrunk for this block alone.

    Taller paragraphs still stop at the source bottom. A long translation may not use this margin to write into the
    bottom margin.
    """
    if block.size <= 0:
        return False
    return block.rect.height <= block.size * 2.6


def _fit_page_margins(
    block: TextBlock,
    rect: pymupdf.Rect,
    blocks: list[TextBlock],
    height: float,
    cjk: bool,
    ink: Sequence[pymupdf.Rect] = (),
) -> pymupdf.Rect:
    """At the top of the page, align with the source top; body text is not written below the source bottom. Cell
    bottoms are not trimmed, so the font size inside a table is not shrunk for one cell.

    The line box margin of one or two lines at the end of the page is kept. That little height only fits this line
    and does not pull the paragraph into the bottom margin.
    """
    y0, y1 = rect.y0, rect.y1
    if cjk and _is_column_top(block, blocks):
        # Move the whole box up instead of making it taller. A taller box would turn one line into two.
        shift = block.size * 0.1
        y0 = max(0.0, y0 - shift)
        y1 -= shift
    floor = content_floor(height)
    if block.clip is None and block.rect.y0 < floor - 4 and not (cjk and _keeps_line_pad(block)):
        bottom = _page_content_bottom(blocks, block.page, height, ink)
        if block.rect.y0 < bottom - 4:
            y1 = min(y1, bottom)
    if y1 - y0 < 4:
        return rect
    return pymupdf.Rect(rect.x0, y0, rect.x1, y1)


def _kept_ink(page: "pymupdf.Page", blocks: list[TextBlock]) -> list[pymupdf.Rect]:
    """Glyphs still in place after erasing that belong to no block, such as display formulas dropped during
    extraction. Table of contents leaders and glyphs moved along with formula images are already erased and are not
    included."""
    # A page has thousands of words, each compared with every line box. Points and boxes are converted to MuPDF's
    # single precision first, so the result matches the Rect checks
    boxes = [single_precision(rect + (-1, -1, 1, 1)) for block in blocks for rect in (block.rect, *block.line_rects)]
    kept = []
    for word in page.get_text("words"):
        center = pymupdf.mupdf.FzPoint((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
        x, y = center.x, center.y
        if not any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in boxes):
            kept.append(pymupdf.Rect(word[:4]))
    return kept


def _body_ink(page: "pymupdf.Page", kept: list[pymupdf.Rect], marks: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Glyphs and figure boxes that stay in place and count as body (see _page_content_bottom): display formulas and
    figures at the bottom of the page. A row holding only the page number does not count; page numbers are often
    plain digits, dropped during extraction and left in place after erasing (see _kept_ink)"""
    words = {tuple(pymupdf.Rect(word[:4])): word[4] for word in page.get_text("words")}
    rows: list[list[pymupdf.Rect]] = []
    for rect in sorted(kept, key=lambda rect: rect.y0):
        if rows and rect.y0 < max(other.y1 for other in rows[-1]) - rect.height * 0.5:
            rows[-1].append(rect)
        else:
            rows.append([rect])
    ink = list(marks)
    for row in rows:
        text = " ".join(words.get(tuple(rect), "") for rect in sorted(row, key=lambda rect: rect.x0))
        if not is_page_number(text):
            ink += row
    return ink


def _beside_rows(
    obstacles: list[pymupdf.Rect],
    rect: pymupdf.Rect,
    rows: Sequence[pymupdf.Rect],
) -> list[pymupdf.Rect]:
    """Obstacles in the way when extending right: only those beside some source line count. The write box has
    extra space above and below the source lines, and the boxes of the lines above and below reach into those few
    points without being beside the block (an indented subentry in an index overlaps the bottom of the entry
    above). A whole extra line at the bottom of the box counts as a line, because text will be written there."""
    top, bottom = min(row.y0 for row in rows), max(row.y1 for row in rows)
    pitch = max(row.height for row in rows)
    bands = _row_bands(rows)
    if rect.y1 - bottom > pitch * 0.5:
        bands.append(pymupdf.Rect(rect.x0, bottom, rect.x1, rect.y1))
    if top - rect.y0 > pitch * 0.5:
        bands.append(pymupdf.Rect(rect.x0, rect.y0, rect.x1, top))
    return [
        ob
        for ob in obstacles
        if any(min(ob.y1, band.y1) - max(ob.y0, band.y0) > min(ob.height, band.height) * 0.5 for band in bands)
        or not (ob.y0 < rect.y1 and ob.y1 > rect.y0)
    ]


def _row_bands(rows: Sequence[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """The source lines, plus bands made of two vertically adjacent lines less than half a line height apart.
    Glyphs between two lines (the equation number to the right of a display formula with cases) overlap each line
    by less than half, and the combined band by more than half."""
    bands = list(rows)
    ordered = sorted(rows, key=lambda row: row.y0)
    for upper, lower in zip(ordered, ordered[1:]):
        if lower.y0 - upper.y1 < min(upper.height, lower.height) * 0.5:
            bands.append(upper | lower)
    return bands


def _on_rows(ink: pymupdf.Rect, rows: Sequence[pymupdf.Rect]) -> bool:
    """These glyphs overlap a line (or two adjacent lines combined) vertically by more than half their own height:
    they are beside these lines, not on the line above or below."""
    return any(min(ink.y1, band.y1) - max(ink.y0, band.y0) > ink.height * 0.5 for band in _row_bands(rows))


def _drawn_marks(page: "pymupdf.Page") -> list[pymupdf.Rect]:
    """Lines, boxes, color fills and images drawn on the page. Shapes filled only with white are invisible and do not
    count."""
    marks = [
        pymupdf.Rect(path["rect"])
        for path in visible_drawings(page)
        if path.get("color") is not None or tuple(path.get("fill") or ()) != (1, 1, 1)
    ]
    marks.extend(pymupdf.Rect(info["bbox"]) for info in page.get_image_info())
    return marks


def _indented_start(
    block: TextBlock,
    tail: TextBlock,
    geometries: list[PageGeometry] | None,
) -> float | None:
    """When only the indented first line of a paragraph stays on this page (column) and the paragraph continues on
    the next page: the left edge of this column. Only when the tail starts at the left edge of its column is it a
    body paragraph; then the translation is written from the column's left edge like other paragraphs, with no
    indent. An indented tail is the hanging indent of a list item and is written from the first line's left edge
    as before."""
    if not geometries or len(block.line_rects) != 1 or block.size <= 0:
        return None

    def left_edge(rect: pymupdf.Rect, page: int) -> float | None:
        return _column_left(rect, geometries[page] if page < len(geometries) else None)

    tail_edge = left_edge(tail.rect, tail.page)
    if tail_edge is None or tail.rect.x0 - tail_edge > 2:
        return None
    edge = left_edge(block.rect, block.page)
    if edge is None or not block.size * 0.5 < block.rect.x0 - edge <= block.size * 3:
        return None
    return edge


def _indent_edge(block: TextBlock, geo: PageGeometry | None, indent: float | None) -> float | None:
    """When a one-line paragraph is indented from the column's left edge by exactly one paragraph indent: the
    column's left edge. The translation of a multi-line paragraph is written from the column's left edge without
    the first-line indent, and a one-line paragraph also starts at the margin. An indent of a different width means
    a block indented as a whole, such as a quotation or code; list items and labelled lines do not count. The
    paragraph has only this line, so its sentence ends on it; a line without sentence-final punctuation is a row
    name in a table (BiDAF+ELMo (Single))"""
    if indent is None or len(block.line_rects) != 1 or not _ends_sentence(block.text.rstrip().rstrip("]")):
        return None
    edge = _column_left(block.rect, geo)
    if edge is None or abs(block.rect.x0 - edge - indent) > block.size * 0.15:
        return None
    if _list_item(block.text) or _LABEL_START_RE.match(block.text) or _numbered_line(block.text):
        return None
    return edge


def _room_below(
    page: "pymupdf.Page",
    block: TextBlock,
    y0: float,
    x0: float,
    x1: float,
    blocks: list[TextBlock],
    kept: Sequence[pymupdf.Rect] = (),
    marks: Sequence[pymupdf.Rect] = (),
    body: Sequence[pymupdf.Rect] = (),
) -> float:
    """How far down this column can still be written. Stops above the page number band, and also before the
    footnotes or other blocks that follow.

    Page numbers are often plain digits, dropped during extraction, and do not necessarily overlap this column in x.
    When the block itself is already in the footer band, no further cut is made, so footnotes at the bottom are not
    shrunk by it.
    Blocks above the body bottom stop at the source body bottom and do not write into the bottom margin. body is the
    glyphs and figure boxes that stay in place and count as body (see _body_ink); the body bottom extends below
    them.
    kept is the glyphs left in place (see _kept_ink), marks is the drawn figure boxes and images (see
    _drawn_marks); those in the body area below the block must be avoided. A box in marks is the line itself, and
    the line spacing below the last line may extend past the bottom (see _filled_bottom), so with only 2pt of
    clearance the glyphs would sit on the line; half the font size is left here instead.
    """
    bounds = text_page_rect(page)
    limit = bounds.y1 - 6
    floor = content_floor(bounds.height)
    if y0 < floor - 4:
        limit = min(limit, floor)
        bottom = _page_content_bottom(blocks, block.page, bounds.height, body)
        if y0 < bottom - 4:
            limit = min(limit, bottom)
    for other in blocks:
        if other.page != block.page or other is block or other.rect.y0 <= y0 + 1:
            continue
        if other.rect.x1 < x0 + 4 or other.rect.x0 > x1 - 4:
            continue
        limit = min(limit, other.rect.y0 - 2)
    for rects, clearance in ((kept, 2.0), (marks, block.size * 0.5)):
        for ink in rects:
            if ink.y0 < max(y0, block.rect.y1) - 1 or ink.y0 >= floor:
                continue
            if ink.x1 < x0 + 4 or ink.x0 > x1 - 4:
                continue
            limit = min(limit, ink.y0 - clearance)
    return limit


def _split_filled_lines(
    text: str,
    formulas: list[dict],
    em: float,
    block_size: float,
    width: float,
    cjk: bool,
    y0: float,
    limit: float,
    page_bottom: float | None = None,
    next_room: float = 0.0,
) -> tuple[str, str]:
    """Split at full lines. As many full lines as fit on this page stay here; the rest continues on the next page.

    next_room is the height the tail can write from its origin downward. When the remaining lines fit there at
    normal line spacing, the lines on this page are counted at normal line spacing. Otherwise, two source lines at
    the bottom of the page cannot hold two translated lines at the source line spacing, and the rest pushed to the
    next page would have to shrink its font size; then, as in _fit_typeset, the line spacing is tightened first and
    the lines are counted after that. The line spacing below the last line is blank and may also extend past limit
    (see _filled_bottom), but the write box ends at page_bottom: when limit is the source body bottom, the box is
    trimmed there, the extra part counted would not fit, and these lines would be shrunk as a whole."""
    prepared = space_around_latin(separate_after_formula(text))
    metrics = _formula_metrics(formulas, block_size, em, block_size * 0.88 if cjk else None)
    widths = {index: metric.slot_width for index, metric in metrics.items()}
    raw = break_lines(prepared, em, max(width, 1.0), widths)

    def fitting(gap: float, room: float) -> int:
        """How many lines fit in a height of room at line spacing gap"""
        count = 0
        while count < len(raw) and (count + 1) * em * gap + em * 0.3 <= room + 1:
            count += 1
        return count

    def page_room(gap: float) -> float:
        bottom = limit + (gap - 1) * em
        return (bottom if page_bottom is None else min(bottom, page_bottom)) - y0

    normal = 1.45 if cjk else 1.2
    count = fitting(1.2 if cjk else 1.1, page_room(1.2 if cjk else 1.1))
    roomy = fitting(normal, page_room(normal))
    if 0 < roomy < count and fitting(normal, next_room + (normal - 1) * em) >= len(raw) - roomy:
        count = roomy
    if count >= len(raw):
        # Return the lines already broken. If the unbroken text were returned, the caller would measure the box as
        # one line, the whole paragraph would be squeezed into one line height, and the font would become unreadable.
        return "\n".join(raw), ""
    if count <= 0:
        return "", prepared
    return "\n".join(raw[:count]), "\n".join(raw[count:])


def _filled_bottom(y0: float, lines: int, em: float, cjk: bool, room: float, page_bottom: float) -> float:
    """Bottom of the write box for the lines _split_filled_lines keeps on this page. The box needs the full
    line-spacing height; below the glyphs of the last line there is a blank part, roughly the amount by which the
    line spacing exceeds the font size. This blank part may extend past room while the glyphs stay within room."""
    gap = 1.45 if cjk else 1.2
    return min(y0 + lines * em * gap + em * 0.3, room + (gap - 1) * em, page_bottom)


def _own_line_box(block: TextBlock, cjk: bool) -> pymupdf.Rect:
    """A half sentence or continuation line occupies only its own line. The box is at least one line high and does
    not go past this line on the left or right."""
    origin = block.line_rects[0].y0 if block.line_rects else block.rect.y0
    y0 = origin + block.size * 0.1 if cjk else origin
    size = block.size * 0.88 if cjk else block.size
    pad = block.size * 0.5 if cjk else block.size * 0.3
    gap = 1.45 if cjk else 1.2
    bottom = max(block.rect.y1 + pad, y0 + size * gap + size * 0.3)
    rect = pymupdf.Rect(block.rect.x0, y0, block.rect.x1 + 2, bottom)
    return _clip_write_rect(rect, block.clip)


# Style marks that break_lines closes at the end of a line when the style continues onto the next line
_TRAILING_CLOSERS_RE = re.compile(r"(?:\{/[bizcu]\})+$")


def _after_first_line(source: str, head: str) -> str:
    """Cut the first line off the source. The space consumed by the line break must not remain at the start of the
    next line. When bold or italic continues past the first line, break_lines adds closing marks at the end of the
    line that the source does not have: they are removed before comparing, and the rest reopens these marks."""
    if not head:
        return source
    if source.startswith(head):
        return source[len(head) :].lstrip(" \t")
    body = _TRAILING_CLOSERS_RE.sub("", head)
    if not body or not source.startswith(body):
        return source
    return balance_marks(body, source[len(body) :].lstrip(" \t"))[1]


def _split_own_lines(
    text: str,
    formulas: list[dict],
    em: float,
    block_size: float,
    host_width: float,
) -> tuple[str, str]:
    """Break the first line at the width of the half sentence and leave the rest unbroken; it is broken again by
    the continuation line's own box when written.

    Line breaks made at the narrow width must not stay in the second half. break_lines treats a newline as a hard
    break, and short lines are not joined again in a wider box, so the body text beside the side label would stay
    as narrow as the first line. Breaking in advance at the full width does not work either: when there is a
    figure beside the continuation lines, writing must also avoid the figure, and the pre-broken lines would leave
    short half lines.
    """
    prepared = space_around_latin(separate_after_formula(text))
    # em is the unshrunk translation font size; formulas keep their original size against it (see _formula_slot)
    metrics = _formula_metrics(formulas, block_size, em, em)
    widths = {index: metric.slot_width for index, metric in metrics.items()}
    raw = break_lines(prepared, em, max(host_width, 1.0), widths)
    if not raw:
        return "", ""
    if len(raw) == 1:
        return raw[0], ""
    rest = _after_first_line(prepared, raw[0]).strip()
    # When the space for the half sentence is too narrow for the first word, break_lines cuts it by character
    # (BER / T). The following lines are wider, so the whole word is left for the continuation line
    shown = STYLE_MARK_RE.sub("", raw[0]).rstrip()
    visible = STYLE_MARK_RE.sub("", prepared)
    joint = visible[len(shown) - 1 : len(shown) + 1] if shown and visible.startswith(shown) else ""
    if not shown or len(joint) == 2 and all(ch.isascii() and ch.isalnum() for ch in joint):
        return "", prepared
    return raw[0], rest


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


def _hosts_first(
    items: list[tuple[TextBlock, str]],
    continuations: dict[int, TextBlock],
) -> list[tuple[TextBlock, str]]:
    """Move tails on the same page after their hosts.

    When the content stream draws the right column first, a tail at the top of the right column comes before its
    host in the left column. When its turn comes there is no continued translation yet, the source is already
    erased, and the whole paragraph is lost.
    """
    host_of = {id(tail): host_id for host_id, tail in continuations.items()}
    position = {id(b): index for index, (b, _t) in enumerate(items)}
    waiting: dict[int, list[tuple[TextBlock, str]]] = {}
    ordered: list[tuple[TextBlock, str]] = []
    for b, t in items:
        host_id = host_of.get(id(b))
        if host_id is not None and position.get(host_id, -1) > position[id(b)]:
            waiting.setdefault(host_id, []).append((b, t))
            continue
        ordered.append((b, t))
        ordered.extend(waiting.pop(id(b), []))
    return ordered
