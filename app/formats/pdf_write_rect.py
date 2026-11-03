"""Write one translated block into its rectangle: align it to the original's axis, edge or label column and
flush it to the frame."""

import bisect

import pymupdf

from .pdf_blocks import (
    _MARK_ITEM_RE,
    TextBlock,
)
from .pdf_layout import (
    PageGeometry,
)
from .pdf_write_geometry import _clip_write_rect, _flow_y0, _frame_x1, _inside_right

# Tolerance for the line centers on a page set centered throughout. A comma at the end of a line or an italic overhang
# shifts the center by a point or two
_AXIS_TOL = 2.0


def _centered_in_cell(block: TextBlock) -> bool:
    """The source text is centered in its cell. The column center does not match the cell center, so the column
    cannot be used to decide."""
    clip = block.clip
    if clip is None:
        return False
    cell_mid = (clip[0] + clip[2]) / 2
    text_mid = (block.rect.x0 + block.rect.x1) / 2
    return abs(text_mid - cell_mid) <= 2.0


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
