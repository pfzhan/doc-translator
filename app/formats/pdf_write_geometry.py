"""Decisions about where the translated lines go: room beside and below the original, hanging indents, labels,
page margins and the area of ink that must not be covered."""

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

from .pdf_blocks import (
    _AFFIL_MARK_RE,
    _LABEL_START_RE,
    _SENTINEL_NUM_RE,
    Leader,
    TextBlock,
    _color_value,
    _column_left,
    _diagramish,
    _ends_sentence,
    _is_side_label,
    _join_lines,
    _list_item,
    _numbered_line,
    _page_char_index,
    _visual_lines,
)
from .pdf_erase import single_precision
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
from .pdf_place import (
    _formula_metrics,
    _formula_slot,
    _inline_marks,
)


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
