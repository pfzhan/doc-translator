"""Page geometry: find columns from the horizontal gaps between text blocks, and collect figures, vector graphics
and ruled tables as boundaries.

Does not depend on the layout model. Headings spanning columns (wider than 60% of the page width) take no part in
column detection. A column gap is at least 12pt wide, with three or more blocks on each side that overlap deeply
enough vertically. The gap need not sit in the middle of the page, so three columns can be separated too. When a
figure takes up the text on one side, the figure edge still acts as a column wall and writing does not cross it.
Tables with horizontal and vertical rules, and short-cell tables whose column edges align, limit writing to their
cells. Whole paragraphs of body text are not tables.
Running headers, footers and page numbers are skipped separately and take no part in column detection.
When the text column gaps are clear, the reading order runs column by column from top to bottom; pages split only by
figure edges are not reordered.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import pymupdf

from .pdf_frame import TextFrame

_GUTTER_MIN = 12.0
_SPAN_MAX = 0.60
_MIN_SIDE = 3
_MIN_Y_OVERLAP = 40.0
_USABLE_W = 8.0
_USABLE_H = 3.0
_FIGURE_MIN = 8.0
_DRAW_MIN = 24.0
_DRAW_AREA = 800.0
_PAGE_FILL = 0.85
_GROUND_SHARE = 2.0  # Maximum ratio of a figure's white ground to the area of the curves and images inside it
_GROUND_REACH = 24.0  # Maximum distance of titles and axis labels from the curves
_GROUND_MARKS = 3  # Minimum number of small marks on a white ground that holds no complete figure
_FRAME_REACH = 12.0  # Maximum distance of a frame line from the top or bottom edge of the white ground
_WALL_MIN_H = 48.0
_WALL_MIN_W = 36.0
_CELL_MIN_W = 18.0
_CELL_MIN_H = 8.0
_CELL_NARROW = 16.0
_MAX_TABLE_COLS = 12
# Table rules: at most this thick and at least this long
_RULE_THICK = 1.5
_RULE_MIN = 5.0
# Snap distance of find_tables: filled shapes thinner than this also count as rules
_SNAP = 3.0
# Distance within which a rule and a cell edge count as aligned
_NEAR = 2.0
# Boundary between two adjacent rows of an algorithm box, as a share of the line spacing below the upper baseline
_ROW_CUT = 0.35
_TEXT_JOIN = 8.0
_COL_ALIGN = 5.0
_MIN_TEXT_ROWS = 3
_MIN_PITCH = 8.0
# Interlinear gloss: tolerance for aligned left edges, minimum number of words, and share of the words in each of the
# two lines that must align
_GLOSS_ALIGN = 1.0
_GLOSS_MIN = 4
_GLOSS_SHARE = 0.75
# Role labels below the words of an example sentence; small capitals extract as all capitals: AGENT, ARG0, ARGM-TMP
# Example numbers: (22.27), (3), (4a)
_NUMBER_LABEL_RE = re.compile(r"\(\d+(?:\.\d+)*[a-z]?\)")
# Example sentences numbered by chapter: (22.9), (3.4a)
_CHAPTER_NUMBER_RE = re.compile(r"\(\d+\.\d+[a-z]?\)")
_ROLE_LABEL_RE = re.compile(r"[A-Z][A-Z0-9-]{2,}")
# Grammatical abbreviations common in the Leipzig Glossing Rules. Single letters (S, A, P) are too common elsewhere
# and do not count
_LEIPZIG = frozenset(
    "SG PL DU NOM ACC GEN DAT ERG ABS LOC ABL ALL INS INSTR COM VOC PST PRS FUT PFV IPFV PRF IMPF "
    "PROG NEG DEF INDF CLF COP AUX POSS REFL RECP CAUS PASS APPL TOP FOC DECL IMP SBJ OBJ OBL DEM "
    "PTCP INF NMLZ COMP REL".split()
)
_PERSON_RE = re.compile(r"[123](?:SG|PL|DU)")
_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
# Syllables in a transcription line are separated by spaces and punctuation. Extracted tone marks are separate
# characters (z`ai, zh¯u) and do not separate syllables
_SYLLABLE_SPLIT_RE = re.compile(r"[\s,.;:!?，。；：！？、()（）\[\]“”\"'‘’]+")
_MORPHEME_SPLIT_RE = re.compile(r"[-.=\[\]<>:~()\s]+")
_MAX_PITCH = 36.0
_SHORT_WORDS = 3
_SHORT_WIDTH = 80.0
_SHORT_SHARE = 0.8
# Maximum number of text line heights between the top rule and the middle rule of a three-line table
_HEAD_ROWS = 4.0
# Minimum length of the rules of a three-line table. Underlines of words and fraction bars in body text are much
# shorter
_BAND_MIN_W = 120.0
_BAND_SHORT_SHARE = 0.6
_LABEL_RE = re.compile(r"(?:Figure|Fig\.|Table|Tab\.)\s*\d+(?:\.\d+)*\b")
_INSIDE = 0.6
_MARGIN_BAND = 0.07
_MARGIN_MIN = 42.0
_MARGIN_HEIGHT = 22.0
_MARGIN_LEN = 80
# A short line at the same height (within tolerance) as this many repeated running headers is also a running header
_MARGIN_ROW = 1.5
_MARGIN_ROWS = 3
_PAGE_NO_RE = re.compile(
    r"^(?:[-–—]\s*)?(?:(?:page|p\.)\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*页)?(?:\s*[-–—])?$"
    r"|^第\s*\d{1,4}\s*页$",
    re.I,
)
_ROMAN_RE = re.compile(
    r"^m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$",
    re.I,
)


@dataclass(frozen=True)
class Column:
    x0: float
    x1: float


@dataclass(frozen=True)
class RuledTable:
    rect: pymupdf.Rect
    cells: tuple[pymupdf.Rect, ...]
    # Pseudocode in an algorithm box is also cut into cells as a one-column table, but is translated line by line.
    # Other tables keep the original text
    algorithm: bool = False


@dataclass(frozen=True)
class PageGeometry:
    width: float
    columns: tuple[Column, ...]
    figures: tuple[pymupdf.Rect, ...]
    height: float = 0.0
    tables: tuple[RuledTable, ...] = ()
    # Figure edges can block writing, but with fewer than three text blocks on one side they cannot drive reordering.
    reads_in_columns: bool = False
    # Text frame voted across the whole document. The column's left edge is the leftmost block on this page, the
    # frame's left edge is where body text starts
    frame: TextFrame | None = None

    def index_of(self, rect: pymupdf.Rect) -> int:
        """The column the block's center falls in. A center in the gap between columns takes the nearest column."""
        center = (rect.x0 + rect.x1) / 2
        for index, column in enumerate(self.columns):
            if column.x0 - 1 <= center <= column.x1 + 1:
                return index
        return min(
            range(len(self.columns)),
            key=lambda index: _interval_distance(center, self.columns[index]),
        )

    def column_of(self, rect: pymupdf.Rect) -> Column:
        return self.columns[self.index_of(rect)]

    def spans_columns(self, rect: pymupdf.Rect) -> bool:
        """Headings and whole tables that span columns. Their center falls into some column, but they cannot serve as
        the start or end of a column."""
        return _spans_columns(rect, self)

    def cell_of(self, rect: pymupdf.Rect) -> pymupdf.Rect | None:
        return cell_containing(rect, self.tables)

    def right_limit(self, rect: pymupdf.Rect, obstacles: list[pymupdf.Rect]) -> float:
        """Right edge of the column, stopping before obstacles on the right that overlap vertically. It can be less
        than rect.x1; callers must not shrink the box because of that."""
        limit = self.column_of(rect).x1
        cell = self.cell_of(rect)
        if cell is not None:
            limit = min(limit, cell.x1 - 0.8)
        for obstacle in obstacles:
            if obstacle.x0 <= rect.x0 + 1:
                continue
            if _overlaps_y(rect, obstacle):
                limit = min(limit, obstacle.x0 - 2)
        return limit

    @staticmethod
    def from_rects(
        page_width: float,
        rects: list[pymupdf.Rect],
        figures: list[pymupdf.Rect] | None = None,
        height: float = 0.0,
        tables: Sequence[RuledTable] | None = None,
        frame: TextFrame | None = None,
    ) -> PageGeometry:
        table_tuple = tuple(tables or ())
        table_zones = [table.rect for table in table_tuple]
        # A table's outer frame is also a large vector shape and must not count as a figure, or the text in its cells
        # would be skipped as a whole.
        # A shape clearly larger than the table is a figure: a chart with a grid, where the grid was detected as a
        # table and the title and axis labels lie outside the grid
        figure_tuple = tuple(
            figure
            for figure in (figures or ())
            if not any(
                _mostly_inside(figure, [zone]) and figure.width * figure.height <= zone.width * zone.height * 1.2
                for zone in table_zones
            )
        )
        voting = _voting_rects(rects, figure_tuple, table_tuple, page_width, height)
        usable = [rect for rect in voting if rect.width >= _USABLE_W and rect.height >= _USABLE_H]
        body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
        columns = _columns_of(page_width, usable, _cutting_figures(figure_tuple, page_width, height))
        return PageGeometry(
            page_width,
            columns,
            figure_tuple,
            height,
            table_tuple,
            reads_in_columns=bool(_find_gutters(body)),
            frame=frame,
        )

    @staticmethod
    def from_page(
        page: pymupdf.Page,
        text_rects: list[pymupdf.Rect],
        frame: TextFrame | None = None,
        scan: PageScan | None = None,
    ) -> PageGeometry:
        """scan holds the tables and figures already read for this page. Read now when not given."""
        scan = scan or scan_page(page)
        bounds = text_page_rect(page)
        return PageGeometry.from_rects(
            bounds.width,
            text_rects,
            list(scan.figures),
            bounds.height,
            scan.tables,
            frame,
        )


@dataclass(frozen=True)
class PageScan:
    """Tables, figures and words of one page. Extraction, reading order, role detection and interlinear glosses all
    use them; finding tables parses the rules and text of each page, and every read of rules, image boxes or words
    takes from a few to tens of milliseconds, so each page is read only once."""

    tables: tuple[RuledTable, ...]
    figures: tuple[pymupdf.Rect, ...]
    words: tuple[_TextWord, ...]
    # Areas of connected light shading merged into one: the background of a table or a text box
    shades: tuple[pymupdf.Rect, ...] = ()
    # Thin horizontal rules: table rules, fraction bars, underlines
    rules: tuple[pymupdf.Rect, ...] = ()
    # All thin horizontal strokes, down to short ones one letter wide (the underline of a single letter is shorter
    # than the lower bound of rules)
    strokes: tuple[pymupdf.Rect, ...] = ()
    # Thin vertical rules: vertical table rules, separators between the chapter number and title at a chapter start
    posts: tuple[pymupdf.Rect, ...] = ()
    # Stroked-only arcs (dependency arcs, attention arcs) and filled rounded boxes (labels on the arcs)
    arcs: tuple[pymupdf.Rect, ...] = ()
    round_boxes: tuple[pymupdf.Rect, ...] = ()
    # Visible spaces ␣ drawn with three short lines (LaTeX \verb* and \textvisiblespace); the font has no such
    # character
    visible_spaces: tuple[pymupdf.Rect, ...] = ()


def scan_page(page: pymupdf.Page) -> PageScan:
    images = _image_infos(page)
    visible, raw = _drawing_paths(page)
    rules = _image_rules(images)
    drawings = [*visible, *rules]
    words = page_words(page)
    horizontal, vertical = _rule_segments(page, drawings)
    # Figures are still found with the unclipped path boxes: after clipping, shapes that covered the whole page and
    # were ignored as background would turn up and change which text inside figures is kept
    return PageScan(
        page_tables(page, drawings, words),
        tuple(_figure_rects(page, [*raw, *rules], images)),
        words,
        _shaded_areas(drawings, text_page_rect(page)),
        tuple(horizontal),
        tuple(_thin_strokes(drawings)),
        tuple(vertical),
        *_curved_paths(drawings),
        visible_spaces=tuple(_visible_spaces(drawings)),
    )


# All four corners of a rounded box are curves; an arc is at least this wide
_ROUND_CORNERS = 4
_ARC_MIN_WIDTH = 6.0
# Maximum distance of the two feet of an arc from the bottom edge of its box
_ARC_FOOT = 3.0


def _curved_paths(drawings: list[dict]) -> tuple[tuple[pymupdf.Rect, ...], tuple[pymupdf.Rect, ...]]:
    """A stroked-only arch with curves is an arc: start and end points apart, both on the bottom edge of the box, with
    the middle raised. Lines in a plot, with one end high and one low, and circles and rounded boxes, whose ends
    meet, do not count. A filled small box with curves at all four corners is a rounded box."""
    arcs: list[pymupdf.Rect] = []
    boxes: list[pymupdf.Rect] = []
    for drawing in drawings:
        items = drawing.get("items") or []
        curves = sum(1 for item in items if item[0] == "c")
        if not curves:
            continue
        rect = pymupdf.Rect(drawing["rect"])
        start, end = items[0][1], items[-1][-1]
        arched = (
            isinstance(start, pymupdf.Point)
            and isinstance(end, pymupdf.Point)
            and abs(start.x - end.x) >= _ARC_MIN_WIDTH * 0.5
            and abs(start.y - rect.y1) <= _ARC_FOOT
            and abs(end.y - rect.y1) <= _ARC_FOOT
            and rect.height >= _ARC_FOOT * 2
        )
        if drawing.get("type") == "s" and rect.width >= _ARC_MIN_WIDTH and arched:
            arcs.append(rect)
        elif drawing.get("fill") is not None and curves >= _ROUND_CORNERS:
            boxes.append(rect)
    return tuple(arcs), tuple(boxes)


# Length of the bottom line of a visible space (points). Drawn 0.3 em wide, about 3 points at body-text size
_VISIBLE_SPACE_WIDTH = (1.0, 6.0)


def _visible_spaces(drawings: list[dict]) -> list[pymupdf.Rect]:
    """A short horizontal line with a shorter vertical line standing at each end, whose lower ends sit on the
    horizontal line: the shape of ␣. Only the shape is detected; extraction decides whether it falls in a gap within a
    line of text."""
    flats: list[pymupdf.Rect] = []
    stems: list[pymupdf.Rect] = []
    low, high = _VISIBLE_SPACE_WIDTH
    for drawing in drawings:
        for item in drawing.get("items", ()):
            if item[0] != "l":
                continue
            rect = pymupdf.Rect(item[1], item[2]).normalize()
            if rect.height < 0.05 and low <= rect.width <= high:
                flats.append(rect)
            elif rect.width < 0.05 and 0.3 <= rect.height <= high:
                stems.append(rect)
    found: list[pymupdf.Rect] = []
    for flat in flats:
        ends = [
            [
                stem
                for stem in stems
                if abs(stem.x0 - x) <= 0.5 and abs(stem.y1 - flat.y1) <= 0.5 and stem.height < flat.width
            ]
            for x in (flat.x0, flat.x1)
        ]
        if all(ends):
            found.append(pymupdf.Rect(flat.x0, min(stem.y0 for stem in ends[0] + ends[1]), flat.x1, flat.y1))
    return found


# Minimum width of an underline (points). Shorter ones are strokes of characters or marks
_STROKE_MIN = 1.0


# An underline drawn as one flat polygon (links in slides) has an outline at most this tall
_STROKE_SHAPE_THICK = 3.0


def _thin_strokes(drawings: list[dict]) -> list[pymupdf.Rect]:
    """Thin horizontal strokes of any length: each line segment and each flat filled rectangle counts as one. Another
    path whose whole outline is a flat band (a filled long thin polygon) counts as one; taller outlines are edges of
    shapes (trapezoids, rounded boxes, dashed boxes) and do not count."""
    found: list[pymupdf.Rect] = []
    for drawing in drawings:
        pieces: list[pymupdf.Rect] | None = []
        for item in drawing.get("items", ()):
            if item[0] == "l":
                rect = pymupdf.Rect(item[1], item[2]).normalize()
            elif item[0] == "re":
                rect = pymupdf.Rect(item[1]).normalize()
            else:
                pieces = None
                break
            if rect.height > _RULE_THICK or rect.width < rect.height:
                pieces = None
                break
            pieces.append(rect)
        if pieces is None:
            outline = pymupdf.Rect(drawing.get("rect") or ())
            pieces = [outline] if 0 < outline.height <= _STROKE_SHAPE_THICK or outline.height == 0 else []
        found.extend(rect for rect in pieces if rect.width >= _STROKE_MIN)
    return found


def _image_infos(page: pymupdf.Page) -> list[dict]:
    """Images on this page, with their boxes reduced to the clipping paths."""
    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001 - if they cannot be read, treat the page as having no images
        return []
    return _clip_images(page, infos) if infos else infos


# Description of an image block in the text layer: <image: DeviceRGB, width: 678, height: 678, bpc: 8>
_IMAGE_SIZE_RE = re.compile(r"width: (\d+), height: (\d+)")


def _clip_images(page: pymupdf.Page, infos: list[dict]) -> list[dict]:
    """Replace image boxes with the part visible after clipping. get_image_info ignores clipping paths: a sample image
    of 20 lines shows only its top three lines, yet its box reaches the bottom of the page, and the two paragraphs it
    covers are taken as text inside the image. Reading the text layer with clipping gives image blocks whose boxes are
    the visible parts; they are matched back to the images by pixel size and position. Images without a match keep
    their original box."""
    try:
        blocks = page.get_text("blocks", flags=pymupdf.TEXT_PRESERVE_IMAGES | pymupdf.TEXT_CLIP)
    except Exception:  # noqa: BLE001 - if it cannot be read, keep the original boxes
        return infos
    shown: list[tuple[pymupdf.Rect, tuple[int, int]]] = []
    for block in blocks:
        size = _IMAGE_SIZE_RE.search(str(block[4])) if block[6] == 1 else None
        if size is not None:
            shown.append((pymupdf.Rect(block[:4]), (int(size.group(1)), int(size.group(2)))))
    out: list[dict] = []
    for info in infos:
        whole = pymupdf.Rect(info["bbox"])
        found = next(
            (
                n
                for n, (rect, size) in enumerate(shown)
                if size == (info.get("width"), info.get("height")) and rect in whole + (-1, -1, 1, 1)
            ),
            None,
        )
        if found is None:
            out.append(info)
            continue
        rect, _size = shown.pop(found)
        out.append({**info, "bbox": tuple(rect)} if rect != whole else info)
    return out


def visible_drawings(page: pymupdf.Page) -> list[dict]:
    """Paths from get_drawings, with boxes clipped to their clipping region. The box of a curve path clipped into a
    figure can reach tens of thousands of points off the page, and the original box would cover the whole column of
    body text below. Paths entirely outside their clipping region are invisible and are dropped"""
    return _drawing_paths(page)[0]


def _drawing_paths(page: pymupdf.Page) -> tuple[list[dict], list[dict]]:
    """Read the paths of this page once and return two lists: clipped (see visible_drawings) and unclipped."""
    try:
        paths = page.get_drawings(extended=True)
    except Exception:  # noqa: BLE001 - the vector graphics of some pages cannot be read; treat as no rules
        return [], []
    out: list[dict] = []
    raw: list[dict] = []
    clips: list[tuple[int, pymupdf.Rect]] = []
    for path in paths:
        level = int(path.get("level") or 0)
        # A clipping region applies to paths nested below its level and lapses once a path is back at its level
        clips = [clip for clip in clips if clip[0] < level]
        kind = path.get("type")
        if kind == "clip":
            if path.get("scissor") is not None:
                clips.append((level, pymupdf.Rect(path["scissor"])))
            continue
        if kind == "group" or path.get("rect") is None:
            continue
        raw.append(path)
        rect = pymupdf.Rect(path["rect"])
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        for _, scissor in clips:
            x0, y0, x1, y1 = max(x0, scissor.x0), max(y0, scissor.y0), min(x1, scissor.x1), min(y1, scissor.y1)
        if x0 > x1 or y0 > y1:
            continue
        out.append({**path, "rect": pymupdf.Rect(x0, y0, x1, y1)} if clips else path)
    return out, raw


def _page_drawings(page: pymupdf.Page, images: list[dict] | None = None) -> list[dict]:
    """Vector graphics of this page plus thin rules drawn as images. images holds the image boxes of this page; read
    now when not given."""
    return [*visible_drawings(page), *_image_rules(_image_infos(page) if images is None else images)]


def light_fill(drawing: dict) -> bool:
    """Light filled-only shapes made of rectangles and thicker than a rule in both directions: the shading of table
    cells and text boxes. Dark shapes do not count: caption labels are white text on black."""
    rect = pymupdf.Rect(drawing["rect"])
    items = drawing.get("items") or []
    fill = drawing.get("fill")
    return (
        drawing.get("type") == "f"
        and bool(items)
        and all(item[0] in ("re", "qu") for item in items)
        and min(rect.width, rect.height) > _RULE_THICK
        and fill is not None
        and len(fill) == 3
        and 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2] > 0.6
    )


def _shaded_areas(drawings: list[dict], bounds: pymupdf.Rect) -> tuple[pymupdf.Rect, ...]:
    """Merge adjacent light shapes into areas. A table often has one shaded shape per cell, with a narrow gap between
    cells. A shape covering the whole page is the paper color and does not count."""
    areas: list[pymupdf.Rect] = []
    for drawing in drawings:
        area = pymupdf.Rect(drawing["rect"])
        if not light_fill(drawing) or (
            area.width > bounds.width * _PAGE_FILL and area.height > bounds.height * _PAGE_FILL
        ):
            continue
        while True:
            near = [other for other in areas if (area + (-_NEAR, -_NEAR, _NEAR, _NEAR)).intersects(other)]
            if not near:
                break
            for other in near:
                areas.remove(other)
                area |= other
        areas.append(area)
    return tuple(areas)


def page_words(page: pymupdf.Page) -> tuple[_TextWord, ...]:
    return tuple(
        _TextWord(float(item[0]), float(item[1]), float(item[2]), float(item[3]), str(item[4]))
        for item in page.get_text("words")
        if str(item[4]).strip()
    )


class _MarginBlock(Protocol):
    page: int
    rect: pymupdf.Rect
    text: str


def text_page_rect(page: pymupdf.Page) -> pymupdf.Rect:
    """The page box that text coordinates refer to. On a rotated page page.rect is rotated, while extracted
    coordinates are not."""
    return pymupdf.Rect(0, 0, page.cropbox.width, page.cropbox.height)


def cell_containing(rect: pymupdf.Rect, tables: Sequence[RuledTable]) -> pymupdf.Rect | None:
    """The cell that holds most of the block. Body text outside the rules returns None."""
    area = rect.width * rect.height
    if area < 1:
        return None
    best: pymupdf.Rect | None = None
    best_area = 0.0
    for table in tables:
        if (rect & table.rect).is_empty:
            continue
        for cell in table.cells:
            hit = rect & cell
            if hit.is_empty or hit.width < 0.4 or hit.height < 0.3:
                continue
            overlap = hit.width * hit.height
            if overlap > best_area:
                best, best_area = cell, overlap
    if best is None or best_area < area * 0.5:
        return None
    return best


class _OrderedBlock(Protocol):
    rect: pymupdf.Rect
    clip: tuple[float, float, float, float] | None


_BlockT = TypeVar("_BlockT", bound=_OrderedBlock)


@dataclass(frozen=True)
class _ReadUnit(Generic[_BlockT]):
    y0: float
    x0: float
    column: int
    order: int
    spanning: bool
    blocks: tuple[_BlockT, ...]


def reading_order(blocks: Sequence[_BlockT], geo: PageGeometry) -> list[_BlockT]:
    """With clear column gaps, order column by column from top to bottom. Headings and whole tables spanning columns
    stay at their vertical position and are not put into any column.

    Single-column pages and pages split only by figure edges keep the given order. Cells inside a table already have
    a row order, and tables spanning columns are not split up.
    """
    if not geo.reads_in_columns or len(geo.columns) < 2 or len(blocks) < 2:
        return list(blocks)
    units = _read_units(blocks, geo)
    separators = sorted(
        (unit for unit in units if unit.spanning),
        key=lambda unit: (unit.y0, unit.order),
    )
    flow = [unit for unit in units if not unit.spanning]
    ordered: list[_BlockT] = []
    cursor = -1e9
    for separator in separators:
        band = [unit for unit in flow if cursor <= unit.y0 < separator.y0]
        ordered.extend(_emit_band(band))
        ordered.extend(separator.blocks)
        taken = {id(unit) for unit in band}
        flow = [unit for unit in flow if id(unit) not in taken]
        cursor = separator.y0
    ordered.extend(_emit_band(flow))
    return ordered


def _read_units(blocks: Sequence[_BlockT], geo: PageGeometry) -> list[_ReadUnit[_BlockT]]:
    spanning_tables = tuple(table for table in geo.tables if _spans_columns(table.rect, geo))
    grouped: dict[int, list[tuple[int, _BlockT]]] = {}
    loose: list[tuple[int, _BlockT]] = []
    for order, block in enumerate(blocks):
        table_index = _spanning_table_index(block, spanning_tables)
        if table_index is None:
            loose.append((order, block))
        else:
            grouped.setdefault(table_index, []).append((order, block))
    units: list[_ReadUnit[_BlockT]] = []
    for index, pairs in grouped.items():
        table = spanning_tables[index]
        cells = tuple(block for _order, block in pairs)
        units.append(
            _ReadUnit(
                table.rect.y0,
                table.rect.x0,
                geo.index_of(table.rect),
                pairs[0][0],
                True,
                cells,
            )
        )
    for order, block in loose:
        units.append(
            _ReadUnit(
                block.rect.y0,
                block.rect.x0,
                geo.index_of(block.rect),
                order,
                _spans_columns(block.rect, geo),
                (block,),
            )
        )
    return units


def _emit_band(units: Sequence[_ReadUnit[_BlockT]]) -> list[_BlockT]:
    ordered: list[_BlockT] = []
    for unit in sorted(units, key=lambda unit: (unit.column, unit.y0, unit.x0, unit.order)):
        ordered.extend(unit.blocks)
    return ordered


def _spans_columns(rect: pymupdf.Rect, geo: PageGeometry) -> bool:
    if geo.width > 0 and rect.width > geo.width * _SPAN_MAX:
        return True
    hits = 0
    for column in geo.columns:
        overlap = min(rect.x1, column.x1) - max(rect.x0, column.x0)
        if overlap > _GUTTER_MIN:
            hits += 1
    return hits >= 2


def _spanning_table_index(block: _OrderedBlock, tables: Sequence[RuledTable]) -> int | None:
    if block.clip is None:
        return None
    rect = pymupdf.Rect(block.clip)
    for index, table in enumerate(tables):
        if cell_containing(rect, (table,)) is not None:
            return index
    return None


def ruled_tables(page: pymupdf.Page, drawings: list[dict] | None = None) -> tuple[RuledTable, ...]:
    """Only cells enclosed by horizontal and vertical rules count. False tables formed by aligned text and grids of
    formula strokes are left out. drawings holds the vector graphics of this page (including rules drawn as images,
    see _page_drawings); read now when not given.

    find_tables first converts every character of the page into a dict, which takes tens of milliseconds per page.
    When the page does not have two horizontal and two vertical rules, no cell can be enclosed and the call is
    skipped; otherwise only the text in the band around the rules is taken (see _table_clip)."""
    if drawings is None:
        drawings = _page_drawings(page)
    if not _grid_possible(drawings):
        return ()
    pymupdf.no_recommend_layout()
    try:
        found = page.find_tables(
            clip=_table_clip(page, drawings),
            vertical_strategy="lines_strict",
            horizontal_strategy="lines_strict",
            paths=drawings,
        )
    except Exception:  # noqa: BLE001 - the rules of some pages cannot be read; treat as no tables
        return ()
    tables: list[RuledTable] = []
    rules: tuple[list[pymupdf.Rect], list[pymupdf.Rect]] | None = None
    for table in found.tables:
        cells = tuple(pymupdf.Rect(cell) for cell in table.cells if cell)
        if not _usable_table(cells):
            continue
        if rules is None:
            rules = _rule_segments(page, drawings)
        tables.append(_with_open_edges(RuledTable(pymupdf.Rect(table.bbox), cells), *rules))
    return tuple(tables)


def _table_clip(page: pymupdf.Page, drawings: Sequence[dict]) -> pymupdf.Rect | None:
    """The area from which find_tables takes text. All rules lie inside it, so the tables found are the same as
    without a limit; only the text outside the area is not converted.

    find_tables uses the text for two decisions: a table area must contain text to count as a table, and a patch of
    connected rules must contain text before an outer frame is added to it. The second compares the top and bottom
    edges of character boxes after conversion by the page transformation matrix, which amounts to comparing
    vertically flipped positions, so besides the band around the rules the area also includes that band flipped
    vertically. The bounding box of a rule may have zero width or height, and Rect union would skip it as empty, so
    the extremes are taken by coordinate. Rotated pages get no limit."""
    if page.rotation or not drawings:
        return None
    rects = [pymupdf.Rect(drawing["rect"]) for drawing in drawings]
    area = pymupdf.Rect(
        min(r.x0 for r in rects),
        min(r.y0 for r in rects),
        max(r.x1 for r in rects),
        max(r.y1 for r in rects),
    )
    flipped = area * ~page.transformation_matrix
    top, bottom = min(area.y0, flipped.y0), max(area.y1, flipped.y1)
    return (
        pymupdf.Rect(
            area.x0 - _TABLE_CLIP_PAD, top - _TABLE_CLIP_PAD, area.x1 + _TABLE_CLIP_PAD, bottom + _TABLE_CLIP_PAD
        )
        & page.rect
    )


# Margin of the text area around the rules: MuPDF decides whether to take a character by its own measured box,
# which differs slightly from the reported box
_TABLE_CLIP_PAD = 5.0


def _image_rules(images: Sequence[dict]) -> list[dict]:
    """Thin strips made by stretching one-pixel images, written as filled rectangles in the get_drawings format.

    Old TeX and some converters draw frame lines with images, which get_drawings does not read, so find_tables does not
    see them either."""
    out: list[dict] = []
    for info in images:
        rect = pymupdf.Rect(info["bbox"])
        if min(rect.width, rect.height) <= _RULE_THICK and max(rect.width, rect.height) >= _RULE_MIN:
            out.append(
                {
                    "type": "f",
                    "rect": rect,
                    "items": [("re", rect, 1)],
                    "closePath": False,
                    "fill": (0.0, 0.0, 0.0),
                    "color": None,
                    "width": 0.0,
                }
            )
    return out


def _grid_possible(drawings: Sequence[dict]) -> bool:
    """At least two horizontal and two vertical rules.

    Counted the way lines_strict takes edges: line segments and the edges of rectangles and quads all count,
    regardless of length (short lines are joined into long ones); filled shapes without a stroke that are wider than
    the snap distance in both directions are not rules."""
    horizontal = vertical = 0
    for drawing in drawings:
        rect = pymupdf.Rect(drawing.get("rect"))
        if drawing.get("type") == "f" and rect.width > _SNAP and rect.height > _SNAP:
            continue
        for item in drawing.get("items", ()):
            if item[0] == "l":
                box = pymupdf.Rect(item[1], item[2]).normalize()
            elif item[0] == "re":
                box = pymupdf.Rect(item[1]).normalize()
            elif item[0] == "qu":
                box = pymupdf.Quad(item[1]).rect
            else:
                continue
            thin_h, thin_v = box.height <= _SNAP, box.width <= _SNAP
            if item[0] != "l" and not (thin_h or thin_v):
                horizontal, vertical = horizontal + 2, vertical + 2
            else:
                horizontal += thin_h
                vertical += thin_v
            if horizontal >= 2 and vertical >= 2:
                return True
    return False


def _rule_segments(
    page: pymupdf.Page,
    drawings: list[dict] | None = None,
) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """Thin horizontal and vertical rules on the page. drawings holds the vector graphics of this page (including
    rules drawn as images); read now when not given.

    Line segments, long thin rectangles and the four edges of stroked rectangles all count."""
    rects: list[pymupdf.Rect] = []
    if drawings is None:
        drawings = _page_drawings(page)
    for drawing in drawings:
        stroked = "s" in str(drawing.get("type") or "")
        for item in drawing.get("items", ()):
            if item[0] == "l":
                rects.append(pymupdf.Rect(item[1], item[2]).normalize())
            elif item[0] == "re":
                rect = pymupdf.Rect(item[1]).normalize()
                rects.append(rect)
                if stroked:
                    rects.extend(_edges_of(rect))
    horizontal = [rect for rect in rects if rect.height <= _RULE_THICK and rect.width >= _RULE_MIN]
    vertical = [rect for rect in rects if rect.width <= _RULE_THICK and rect.height >= _RULE_MIN]
    return horizontal, vertical


def _edges_of(rect: pymupdf.Rect) -> list[pymupdf.Rect]:
    return [
        pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0),
        pymupdf.Rect(rect.x0, rect.y1, rect.x1, rect.y1),
        pymupdf.Rect(rect.x0, rect.y0, rect.x0, rect.y1),
        pymupdf.Rect(rect.x1, rect.y0, rect.x1, rect.y1),
    ]


def _with_open_edges(
    table: RuledTable,
    horizontal: list[pymupdf.Rect],
    vertical: list[pymupdf.Rect],
) -> RuledTable:
    """Add the outer rows and columns that have no outer frame.

    Many papers draw only the rules between cells: horizontal rules reach past the outermost vertical rule, and
    vertical rules reach past the top and bottom horizontal rules. find_tables only detects cells with rules on all
    four sides, so the header, the first and last columns and the last row are missed. The grid is completed up to
    where the rules reach; adjacent cells without a rule between them (headers spanning columns) are merged into one.
    """
    box = pymupdf.Rect(table.rect)
    for _ in range(3):
        ys = [
            v
            for v in vertical
            if box.x0 - _NEAR <= v.x0 <= box.x1 + _NEAR and v.y1 >= box.y0 - _NEAR and v.y0 <= box.y1 + _NEAR
        ]
        xs = [
            h
            for h in horizontal
            if box.y0 - _NEAR <= h.y0 <= box.y1 + _NEAR and h.x1 >= box.x0 - _NEAR and h.x0 <= box.x1 + _NEAR
        ]
        grown = pymupdf.Rect(
            min([box.x0, *(h.x0 for h in xs)]),
            min([box.y0, *(v.y0 for v in ys)]),
            max([box.x1, *(h.x1 for h in xs)]),
            max([box.y1, *(v.y1 for v in ys)]),
        )
        if all(abs(a - b) < 0.5 for a, b in zip(grown, box)):
            break
        box = grown
    if all(abs(a - b) < 0.5 for a, b in zip(box, table.rect)):
        return table
    col_edges = _edges([box.x0, box.x1, *(v.x0 for v in ys)])
    row_edges = _edges([box.y0, box.y1, *(h.y0 for h in xs)])
    found: list[pymupdf.Rect] = []
    for top, bottom in zip(row_edges, row_edges[1:]):
        row: list[pymupdf.Rect] = []
        for left, right in zip(col_edges, col_edges[1:]):
            center = pymupdf.Point((left + right) / 2, (top + bottom) / 2)
            if any(old.contains(center) for old in table.cells):
                found.extend(row)
                row = []
                continue
            cell = pymupdf.Rect(left, top, right, bottom)
            if row and abs(row[-1].x1 - left) < _NEAR and not _ruled(vertical, left, top, bottom, upright=True):
                row[-1] = row[-1] | cell
            else:
                row.append(cell)
        found.extend(row)
    added = _merge_rows(found, horizontal)
    return RuledTable(box, (*table.cells, *added))


def _edges(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if not out or value - out[-1] > _NEAR:
            out.append(value)
    return out


def _ruled(segments: list[pymupdf.Rect], at: float, start: float, end: float, *, upright: bool) -> bool:
    """Rules at at cover most of the range from start to end. With upright true, look for vertical rules, otherwise
    horizontal ones."""
    covered = 0.0
    for seg in segments:
        pos = seg.x0 if upright else seg.y0
        if abs(pos - at) > _NEAR:
            continue
        lo, hi = (seg.y0, seg.y1) if upright else (seg.x0, seg.x1)
        covered += max(0.0, min(hi, end) - max(lo, start))
    return covered >= (end - start) * 0.6


def _merge_rows(cells: list[pymupdf.Rect], horizontal: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Merge outer cells that are vertically adjacent, aligned left and right, and not separated by a horizontal rule
    (row names spanning rows)."""
    out: list[pymupdf.Rect] = []
    for cell in sorted(cells, key=lambda c: (c.x0, c.y0)):
        prev = out[-1] if out else None
        if (
            prev is not None
            and abs(prev.x0 - cell.x0) < _NEAR
            and abs(prev.x1 - cell.x1) < _NEAR
            and abs(prev.y1 - cell.y0) < _NEAR
            and not _ruled(horizontal, cell.y0, cell.x0, cell.x1, upright=False)
        ):
            out[-1] = prev | cell
            continue
        out.append(cell)
    return out


def page_tables(
    page: pymupdf.Page,
    drawings: list[dict] | None = None,
    words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """Ruled tables and tables made of shaded cells, plus short-cell tables with aligned column edges and algorithm
    boxes. Two-column body text and whole paragraphs are not included. drawings and words hold the vector graphics
    and words already read for this page; read now when not given."""
    if drawings is None:
        drawings = _page_drawings(page)
    if words is None:
        words = page_words(page)
    ruled = tuple(_with_text_rows(table, words) for table in ruled_tables(page, drawings))
    ruled = (*ruled, *filled_tables(page, drawings, words, ruled))
    ruled = (*ruled, *column_ruled_tables(_rule_segments(page, drawings)[1], words, ruled))
    algorithms = algorithm_tables(page, drawings, words)
    banded = banded_tables(_rule_segments(page, drawings)[0], words, (*ruled, *algorithms))
    ruled = (*(table for table in ruled if not _mostly_inside(table.rect, [band.rect for band in banded])), *banded)
    # Unruled cell tables are found by aligned rows in the first place, so an aligned header is already included and
    # there is no search upward
    ruled = _with_header_rows(ruled, words)
    unruled = unruled_tables(page, (*ruled, *algorithms), words)
    if unruled:
        unruled = _with_ruled_heads(unruled, words, _rule_segments(page, drawings)[0])
    return (*ruled, *unruled, *algorithms)


def _with_header_rows(tables: Sequence[RuledTable], words: Sequence[_TextWord]) -> tuple[RuledTable, ...]:
    """In a three-line table a rule separates the header from the body, and the rules only yield the body. Rows right
    above the table, horizontally within its extent, with every word inside one column of the table (not crossing a
    column line), covering three or more columns and with at most three words per column are header rows, and each
    row joins the table as one cell. Descriptions and body text above the table are continuous sentences that cross
    the column lines, and the search stops there."""
    zones = [table.rect for table in tables]
    out: list[RuledTable] = []
    for table in tables:
        rect = table.rect
        lines = sorted({x for cell in table.cells for x in (cell.x0, cell.x1) if rect.x0 + 2 < x < rect.x1 - 2})
        above = [
            word
            for word in words
            if word.x0 >= rect.x0 - 2
            and word.x1 <= rect.x1 + 2
            and word.y1 <= rect.y0 + 1
            and not _mostly_inside(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), zones)
        ]
        top = rect.y0
        headers: list[pymupdf.Rect] = []
        for row in reversed(_word_rows(above)):
            box = pymupdf.Rect(
                min(w.x0 for w in row), min(w.y0 for w in row), max(w.x1 for w in row), max(w.y1 for w in row)
            )
            if top - box.y1 > box.height or not _header_row(row, lines):
                break
            headers.append(pymupdf.Rect(rect.x0, box.y0 - 1, rect.x1, min(box.y1 + 1, top)))
            top = box.y0
        if headers:
            grown = pymupdf.Rect(rect.x0, top - 1, rect.x1, rect.y1)
            out.append(RuledTable(grown, (*reversed(headers), *table.cells), table.algorithm))
        else:
            out.append(table)
    return tuple(out)


def column_ruled_tables(
    vertical: Sequence[pymupdf.Rect],
    words: Sequence[_TextWord],
    known: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """Tables with only column lines: no outer frame and only one horizontal rule below the header, so find_tables
    cannot enclose cells; numbers are centered, so alignment of left and right edges does not detect them either.
    When several vertical lines have the same vertical extent, each text row within that extent has text on both sides
    of every line, and most cells are short, columns are split by the lines and rows by the text. A separator between
    two columns of body text has long sentences on both sides and does not count."""
    lines = _joined_verticals(vertical)
    tables: list[RuledTable] = []
    used: set[int] = set()
    for index, line in enumerate(lines):
        if index in used:
            continue
        group = [
            other
            for other in lines
            if other.height > 0
            and min(line.y1, other.y1) - max(line.y0, other.y0) >= 0.8 * min(line.height, other.height)
        ]
        used.update(lines.index(other) for other in group)
        xs = sorted(other.x0 for other in group)
        top, bottom = min(other.y0 for other in group), max(other.y1 for other in group)
        inside = [word for word in words if top < (word.y0 + word.y1) / 2 < bottom]
        rows = _word_rows(inside)
        if len(rows) < _MIN_TEXT_ROWS:
            continue
        sided = [row for row in rows if all(any(w.x1 <= x for w in row) and any(w.x0 >= x for w in row) for x in xs)]
        if len(sided) < len(rows) * 0.8:
            continue
        pieces = [cell for row in rows for cell in _row_cells(row)]
        if (
            sum(1 for cell in pieces if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
            < len(pieces) * _SHORT_SHARE
        ):
            continue
        left = min(w.x0 for w in inside) - 2
        right = max(w.x1 for w in inside) + 2
        area = pymupdf.Rect(left, top, right, bottom)
        if any(_mostly_inside(area, [table.rect]) or _mostly_inside(table.rect, [area]) for table in known):
            continue
        tops = [min(w.y0 for w in row) for row in rows]
        edges = [top, *((max(w.y1 for w in upper) + lower_top) / 2 for upper, lower_top in zip(rows, tops[1:])), bottom]
        cols = [left, *xs, right]
        cells = tuple(
            pymupdf.Rect(x0, y0, x1, y1) for y0, y1 in zip(edges, edges[1:]) for x0, x1 in zip(cols, cols[1:])
        )
        tables.append(RuledTable(area, cells))
    return tuple(tables)


def banded_tables(
    horizontal: Sequence[pymupdf.Rect],
    words: Sequence[_TextWord],
    known: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """Three-line tables: the top rule, the middle rule below the header and the bottom rule have the same length, with
    no vertical rules. The header often wraps and is centered, groups of data are separated by further rules, and the
    row spacing is uneven, so a search by alignment finds only one section. With three or more horizontal rules whose
    ends align and mostly short cells between them, the table is split into sections by the rules, and each section
    into rows and cells by the text. A table already found inside this area is replaced by this one; an area inside an
    existing table is not counted again."""
    rules = sorted((rule for rule in horizontal if rule.width >= _BAND_MIN_W), key=lambda rule: rule.y0)
    tables: list[RuledTable] = []
    used: set[int] = set()
    for index, rule in enumerate(rules):
        if index in used:
            continue
        group: list[pymupdf.Rect] = []
        for other_index, other in enumerate(rules):
            if abs(other.x0 - rule.x0) <= _NEAR and abs(other.x1 - rule.x1) <= _NEAR:
                used.add(other_index)
                if not group or other.y0 - group[-1].y0 > _NEAR:
                    group.append(other)
        if len(group) < 3:
            continue
        left, right = rule.x0, rule.x1
        bands: list[list[list[_TextWord]]] = []
        for upper, lower in zip(group, group[1:]):
            inside = [
                word
                for word in words
                if upper.y1 < (word.y0 + word.y1) / 2 < lower.y0
                and word.x0 >= left - _NEAR
                and word.x1 <= right + _NEAR
            ]
            bands.append(_word_rows(inside))
        rows = [row for band in bands for row in band]
        if len(rows) < _MIN_TEXT_ROWS or any(not band for band in bands):
            continue
        # Rules with a caption between them enclose a figure (grammar rules, example boxes), and the caption is
        # translated as usual
        if any(_LABEL_RE.match(" ".join(word.text for word in row)) for row in rows):
            continue
        # Backed by three rules, a lower share of short cells is enough: column names in the header and expressions
        # such as O(k·n·d) are longer than data cells.
        # A line of body text or pseudocode forms one cell spanning more than half the table
        pieces = [cell for row in rows for cell in _row_cells(row)]
        short = sum(1 for cell in pieces if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
        # A first row below a rule with a single short cell is a group name (Conditional, μ prediction) and is not
        # counted as a row without cells
        split = sum(
            1
            for band in bands
            for index, row in enumerate(band)
            if len(cells := _row_cells(row)) >= 2
            or (index == 0 and cells[0].words <= _SHORT_WORDS and cells[0].width <= _SHORT_WIDTH)
        )
        # A cell spanning more than half the table indicates body text only when it is the only cell in its row; row
        # names of a two-column table take up most of the width anyway
        if (
            short < len(pieces) * _BAND_SHORT_SHARE
            or split < len(rows) * 0.8
            or any(len(cells := _row_cells(row)) == 1 and cells[0].width > (right - left) / 2 for row in rows)
        ):
            continue
        area = pymupdf.Rect(left, group[0].y0, right, group[-1].y1)
        if any(_mostly_inside(area, [table.rect]) for table in known):
            continue
        cells: list[pymupdf.Rect] = []
        for (upper, lower), band in zip(zip(group, group[1:]), bands):
            tops = [min(w.y0 for w in row) for row in band]
            edges = [upper.y1, *((max(w.y1 for w in row) + top) / 2 for row, top in zip(band, tops[1:])), lower.y0]
            for row, y0, y1 in zip(band, edges, edges[1:]):
                boxes = _row_cells(row)
                cuts = [left, *((a.x1 + b.x0) / 2 for a, b in zip(boxes, boxes[1:])), right]
                cells.extend(pymupdf.Rect(x0, y0, x1, y1) for x0, x1 in zip(cuts, cuts[1:]))
        tables.append(RuledTable(area, tuple(cells)))
    return tuple(tables)


def _joined_verticals(vertical: Sequence[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Join vertical segments that meet end to end at the same x coordinate. Table rules drawn row by row have one
    segment per cell."""
    out: list[pymupdf.Rect] = []
    for seg in sorted(vertical, key=lambda r: (round(r.x0), r.y0)):
        last = out[-1] if out else None
        if last is not None and abs(last.x0 - seg.x0) < 1 and seg.y0 - last.y1 <= _SNAP:
            out[-1] = pymupdf.Rect(last.x0, last.y0, last.x1, max(last.y1, seg.y1))
        else:
            out.append(pymupdf.Rect(seg))
    return out


def _with_text_rows(table: RuledTable, words: Sequence[_TextWord]) -> RuledTable:
    """In a table with only column lines and no rules between rows, one cell holds several stacked lines. When the
    cells of one band have text in lines aligned across them, each line spanning two or more cells, a row name on every
    line and mostly short cells, the lines are unruled rows and are cut at the gaps between them. A wrapped header
    often has cells with only one line, a wrapped sentence stays within one cell, and long sentences do not count as
    short cells."""
    bands: dict[tuple[int, int], list[pymupdf.Rect]] = {}
    for cell in table.cells:
        bands.setdefault((round(cell.y0), round(cell.y1)), []).append(cell)
    cells: list[pymupdf.Rect] = []
    for band in bands.values():
        cells.extend(_split_band(band, words) if len(band) >= 2 else band)
    if len(cells) == len(table.cells):
        return table
    return RuledTable(table.rect, tuple(cells), table.algorithm)


def _split_band(band: Sequence[pymupdf.Rect], words: Sequence[_TextWord]) -> list[pymupdf.Rect]:
    def holder(word: _TextWord) -> int | None:
        center = pymupdf.Point((word.x0 + word.x1) / 2, (word.y0 + word.y1) / 2)
        return next((i for i, cell in enumerate(band) if cell.contains(center)), None)

    inside = [word for word in words if holder(word) is not None]
    rows = _word_rows(inside)
    if len(rows) < 2:
        return list(band)
    # The leftmost column with text holds the row names
    names = min({i for i in map(holder, inside) if i is not None}, key=lambda i: band[i].x0)
    pieces: list[_TextCell] = []
    for row in rows:
        by_cell: dict[int | None, list[_TextWord]] = {}
        for word in row:
            by_cell.setdefault(holder(word), []).append(word)
        if len(by_cell) < 2 or names not in by_cell:
            return list(band)
        for part in by_cell.values():
            pieces.extend(_row_cells(part))
    short = sum(1 for piece in pieces if piece.words <= _SHORT_WORDS and piece.width <= _SHORT_WIDTH)
    if short < len(pieces) * _SHORT_SHARE:
        return list(band)
    cuts = [
        (max(word.y1 for word in upper) + min(word.y0 for word in lower)) / 2 for upper, lower in zip(rows, rows[1:])
    ]
    out: list[pymupdf.Rect] = []
    for cell in band:
        edges = [cell.y0, *(y for y in cuts if cell.y0 < y < cell.y1), cell.y1]
        out.extend(pymupdf.Rect(cell.x0, top, cell.x1, bottom) for top, bottom in zip(edges, edges[1:]))
    return out


def _with_ruled_heads(
    tables: Sequence[RuledTable],
    words: Sequence[_TextWord],
    horizontal: Sequence[pymupdf.Rect],
) -> tuple[RuledTable, ...]:
    """The header of a three-line table sits between the top rule and the middle rule. Header text is often wider than
    the data columns and centered, so its left and right edges do not match the body, and a table found by alignment
    holds only the body. When a middle rule across the whole table lies right above the body and a top rule of the
    same length lies further up, the text between the two rules is the header; it is split into cells at the gaps
    between columns and added to the table."""
    out: list[RuledTable] = []
    for table in tables:
        rect = table.rect
        row_height = min(cell.height for cell in table.cells)
        spans = [h for h in horizontal if h.x0 <= rect.x0 + _NEAR and h.x1 >= rect.x1 - _NEAR]
        mids = [h for h in spans if h.y1 <= rect.y0 + _NEAR and rect.y0 - h.y0 <= row_height]
        if not mids:
            out.append(table)
            continue
        mid = max(mids, key=lambda h: h.y0)
        tops = [
            h
            for h in spans
            if abs(h.x0 - mid.x0) <= _NEAR
            and abs(h.x1 - mid.x1) <= _NEAR
            and row_height * 0.5 < mid.y0 - h.y0 <= row_height * _HEAD_ROWS
        ]
        if not tops:
            out.append(table)
            continue
        top = max(tops, key=lambda h: h.y0)
        band = [
            word
            for word in words
            if top.y1 < (word.y0 + word.y1) / 2 < mid.y0 and word.x0 >= mid.x0 - _NEAR and word.x1 <= mid.x1 + _NEAR
        ]
        heads = _head_cells([_row_cells(row) for row in _word_rows(band)], mid.x0, mid.x1, top.y1, mid.y0)
        if len(heads) < 2:
            out.append(table)
            continue
        grown = pymupdf.Rect(min(rect.x0, mid.x0), top.y0, max(rect.x1, mid.x1), rect.y1)
        out.append(RuledTable(grown, (*heads, *table.cells), table.algorithm))
    return tuple(out)


def _head_cells(
    rows: Sequence[Sequence[_TextCell]],
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> list[pymupdf.Rect]:
    """Split header text into cells by column. Text right below the nearest cell above it, when that cell overlaps
    only this one piece of text, is a wrapped line of the same cell; a group name spanning two columns is a cell of its
    own. Cells extend left and right to the middle of the gap to the neighboring cell, and up and down to the bounds
    of their header section."""
    groups: list[pymupdf.Rect] = []
    for row in rows:
        boxes = [pymupdf.Rect(cell.x0, cell.y0, cell.x1, cell.y1) for cell in row]
        start = len(groups)
        for box in boxes:
            hits = [g for g in groups[:start] if g.x0 < box.x1 and box.x0 < g.x1]
            near = max(hits, key=lambda g: g.y1, default=None)
            if near is not None and sum(near.x0 < b.x1 and b.x0 < near.x1 for b in boxes) == 1:
                near |= box
            else:
                groups.append(box)
    cells: list[pymupdf.Rect] = []
    for box in groups:
        side = [g for g in groups if g is not box and g.y0 < box.y1 and box.y0 < g.y1]
        x0 = max([left, *((g.x1 + box.x0) / 2 for g in side if g.x1 <= box.x0)])
        x1 = min([right, *((box.x1 + g.x0) / 2 for g in side if g.x0 >= box.x1)])
        y0 = max([top, *((g.y1 + box.y0) / 2 for g in groups if g.y1 <= box.y0 and g.x0 < box.x1 and box.x0 < g.x1)])
        y1 = min([bottom, *((box.y1 + g.y0) / 2 for g in groups if g.y0 >= box.y1 and g.x0 < box.x1 and box.x0 < g.x1)])
        cells.append(pymupdf.Rect(x0, y0, x1, y1))
    return cells


def _header_row(row: Sequence[_TextWord], lines: Sequence[float]) -> bool:
    """Text joined by normal word spacing is a sentence (the last line of a paragraph, Levenshtein in Eq. 2.20.);
    header columns are separated by gaps."""
    if len(_row_cells(row)) < 2:
        return False
    if any(word.x0 < x - 1 and word.x1 > x + 1 for word in row for x in lines):
        return False
    columns: dict[int, int] = {}
    for word in row:
        column = bisect.bisect(lines, (word.x0 + word.x1) / 2)
        columns[column] = columns.get(column, 0) + 1
    return len(columns) >= 3 and max(columns.values()) <= _SHORT_WORDS


def filled_tables(
    page: pymupdf.Page,
    drawings: list[dict],
    words: Sequence[_TextWord],
    ruled: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """Tables with one shaded shape per cell, the cells tiling an area (LaTeX \\rowcolor, \\cellcolor). Such tables
    often have only a few horizontal rules and no vertical ones, so the rules yield no cells and the text of all cells
    would be merged into one paragraph for translation.

    The shaded shapes must form at least two rows and two columns and tile the area, at least two cells must contain
    text, and all text in the area must fall into some cell. One shaded shape for the whole area is a text box, and
    one shape per line is the shading of a code box; neither counts. White cells laid out in a figure with labels
    spanning several cells do not count either. Tables already found by rules are not counted again.
    """
    bounds = text_page_rect(page)
    fills = [
        pymupdf.Rect(drawing["rect"])
        for drawing in drawings
        if light_fill(drawing)
        and min(drawing["fill"]) < 0.97
        and not (
            drawing["rect"].width > bounds.width * _PAGE_FILL and drawing["rect"].height > bounds.height * _PAGE_FILL
        )
    ]
    groups: list[list[pymupdf.Rect]] = []
    for fill in fills:
        near = [group for group in groups if any((fill + (-_NEAR, -_NEAR, _NEAR, _NEAR)).intersects(r) for r in group)]
        merged = [fill]
        for group in near:
            groups.remove(group)
            merged.extend(group)
        groups.append(merged)
    tables: list[RuledTable] = []
    for cells in groups:
        area = pymupdf.Rect(cells[0])
        for cell in cells[1:]:
            area |= cell
        if any(_mostly_inside(area, [table.rect]) for table in ruled):
            continue
        rows = _edges([cell.y0 for cell in cells])
        cols = _edges([cell.x0 for cell in cells])
        if len(cols) == 1 and _spaced_rows(cells, words):
            tables.append(RuledTable(area, tuple(sorted(cells, key=lambda c: c.y0))))
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        # Cells must not overlap and together must tile the area: nested shading (a shape inside a box) adds extra
        # area
        covered = _tiled_area(cells)
        if covered is None or abs(covered - area.get_area()) > area.get_area() * 0.05:
            continue
        boxes = [pymupdf.Rect(w.x0, w.y0, w.x1, w.y1) for w in words]
        boxes = [box for box in boxes if _mostly_inside(box, [area])]
        holders = [[cell for cell in cells if _holds(cell, box)] for box in boxes]
        if not all(holders):
            continue
        if len({id(found[0]) for found in holders}) >= 2:
            tables.append(RuledTable(area, tuple(sorted(cells, key=lambda c: (c.y0, c.x0)))))
    return tuple(tables)


def _tiled_area(cells: Sequence[pymupdf.Rect]) -> float | None:
    """Total area of the shaded shapes. Adjacent cells overlap in a strip along their shared edge (LaTeX paints a
    little beyond each cell so that no white seam shows between cells), and that strip is subtracted. When most of
    one shape lies on another, it is a shape inside a box, and None is returned."""
    total = sum(cell.get_area() for cell in cells)
    for index, cell in enumerate(cells):
        for other in cells[index + 1 :]:
            hit = cell & other
            if hit.is_empty:
                continue
            if hit.get_area() > min(cell.get_area(), other.get_area()) * 0.5:
                return None
            total -= hit.get_area()
    return total


def _spaced_rows(cells: Sequence[pymupdf.Rect], words: Sequence[_TextWord]) -> bool:
    """A one-column table shaded row by row (\\rowcolor): three or more rows, one line of text per row, each split into
    parts by large gaps (example sentence, blank line to fill in, answer). The shading of a code box also has one
    shape per line, but each line is one continuous sentence."""
    if len(cells) < _MIN_TEXT_ROWS:
        return False
    for cell in cells:
        inside = [word for word in words if _holds(cell, pymupdf.Rect(word.x0, word.y0, word.x1, word.y1))]
        rows = _word_rows(inside)
        if len(rows) != 1 or len(_row_cells(rows[0])) < 2:
            return False
    return True


def _holds(cell: pymupdf.Rect, box: pymupdf.Rect) -> bool:
    """The word lies entirely in the cell, sticking out only slightly left or right. A word spanning two cells shows
    that the shading is not laid out by cells."""
    middle = (box.y0 + box.y1) / 2
    return cell.x0 - _NEAR <= box.x0 and box.x1 <= cell.x1 + _NEAR and cell.y0 <= middle <= cell.y1


_SYMBOL_FONT_RE = re.compile(r"(?:^|\+)(?:CMSY|CMEX|MSAM|MSBM)", re.I)
_ALGORITHM_RE = re.compile(r"^\s*(?:Algorithm|Procedure|算法)\s*\d+", re.I)
_ALGORITHM_WORD_RE = re.compile(r"Algorithm|Procedure|算法", re.I)


def algorithm_tables(
    page: pymupdf.Page,
    drawings: list[dict] | None = None,
    words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """Treat pseudocode in an algorithm box as a one-column table with one cell per line.

    Pseudocode has one statement per line and indentation is structure; the lines do not form a paragraph. PyMuPDF
    often puts several lines into one block, and later merging would join them into a reflowed paragraph. As cells,
    each line is translated on its own, written back to its original line and not merged with other lines. The extent
    of an algorithm box is set by its frame rules: from the rule below the caption to the next rule below it.
    words holds the words of this page. A page where no word contains Algorithm, Procedure or "算法" has no algorithm
    caption, so the text of the whole page need not be read again.
    """
    if words is not None and not any(_ALGORITHM_WORD_RE.search(word.text) for word in words):
        return ()
    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return ()
    lines = [
        line
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        if abs(line["dir"][1]) <= 0.01
    ]
    captions = [caption for line in lines for caption in _algorithm_captions(line)]
    if not captions:
        return ()
    horizontal, _vertical = _rule_segments(page, drawings)
    tables: list[RuledTable] = []
    for box, size in captions:
        body = _box_below(box, size, horizontal) or _box_above(box, size, horizontal)
        if body is None:
            continue
        rows = _code_rows(lines, body, horizontal)
        if len(rows) >= 2:
            tables.append(RuledTable(body, rows, algorithm=True))
    return tuple(tables)


def _algorithm_captions(line: dict) -> list[tuple[pymupdf.Rect, float]]:
    """Algorithm captions in one line (box, font size). Two algorithm boxes side by side often have their captions in
    the same line, separated by a column gap."""
    spans = [span for span in line["spans"] if span["text"].strip()]
    starts: list[int] = []
    for index, span in enumerate(spans):
        if index and pymupdf.Rect(span["bbox"]).x0 - pymupdf.Rect(spans[index - 1]["bbox"]).x1 < span["size"] * 2:
            continue
        if _ALGORITHM_RE.match("".join(item["text"] for item in spans[index:])):
            starts.append(index)
    found: list[tuple[pymupdf.Rect, float]] = []
    for begin, end in zip(starts, [*starts[1:], len(spans)]):
        box = pymupdf.Rect()
        for span in spans[begin:end]:
            box |= pymupdf.Rect(span["bbox"])
        found.append((box, max(span["size"] for span in spans[begin:end])))
    return found


def _box_below(caption: pymupdf.Rect, size: float, horizontal: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """Caption on top: the first rule below the caption is the top of the box, the next one below is the bottom. The
    caption may have several lines."""
    rules = sorted(
        (rule for rule in horizontal if rule.y0 >= caption.y1 - _NEAR and _under(rule, caption, size)),
        key=lambda rule: rule.y0,
    )
    rules = _distinct_rules(rules)
    if len(rules) < 2 or rules[0].y0 - caption.y1 > size * 8:
        return None
    top, bottom = rules[0], rules[1]
    return pymupdf.Rect(top.x0, top.y0, top.x1, bottom.y0)


def _box_above(caption: pymupdf.Rect, size: float, horizontal: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """Caption at the bottom: the rule right above the caption is the bottom of the box, the next rule of the same
    width further up is the top."""
    rules = sorted(
        (rule for rule in horizontal if rule.y1 <= caption.y0 + _NEAR and _under(rule, caption, size)),
        key=lambda rule: -rule.y0,
    )
    rules = _distinct_rules(rules)
    if len(rules) < 2 or caption.y0 - rules[0].y1 > size * 2:
        return None
    bottom = rules[0]
    top = next(
        (rule for rule in rules[1:] if abs(rule.x0 - bottom.x0) < _NEAR and abs(rule.x1 - bottom.x1) < _NEAR), None
    )
    if top is None:
        return None
    return pymupdf.Rect(bottom.x0, top.y1, bottom.x1, bottom.y0)


_LINE_LABEL_RE = re.compile(r"^\s*\d+\s*[:.]")
# Line ends of an unfinished sentence: a lowercase letter, a comma, a hyphen of a broken word
_OPEN_END_RE = re.compile(r"[a-z,]$|[a-z]-$")


def _join_wrapped_rows(
    rows: list[pymupdf.Rect],
    baselines: list[float],
    spans: list[dict],
    main: float,
) -> list[pymupdf.Rect]:
    """In an algorithm with line numbers at the line starts (1: 2: ...), a line without a number after the numbered
    lines is a wrapped part of the previous statement and joins the previous cell. In an algorithm without line
    numbers, when the previous line reaches the right edge of the box and its sentence is unfinished, the next line is
    its wrapped part.

    If the two halves were sent for translation separately, a half sentence would have too few body-text words and
    would be kept in English as a formula. Input: and Output: before the first line number are one cell each."""
    labeled: list[bool] = []
    full: list[bool] = []
    for baseline in baselines:
        row = sorted(
            (span for span in spans if abs(span["origin"][1] - baseline) <= main * 0.4),
            key=lambda span: span["origin"][0],
        )
        text = "".join(span["text"] for span in row).rstrip()
        labeled.append(bool(_LINE_LABEL_RE.match(text)))
        # A full line has even spacing; a right-aligned comment (// mean) also reaches the right edge, but with a wide
        # gap after the statement
        boxes = [span["bbox"] for span in row if "bbox" in span and span["text"].strip()]
        right = max((box[2] for box in boxes), default=0.0)
        gaps = [b[0] - a[2] for a, b in zip(boxes, boxes[1:])]
        full.append(
            bool(rows)
            and right >= rows[0].x1 - main * 0.5
            and max(gaps, default=0.0) < main * 1.5
            and bool(_OPEN_END_RE.search(text))
        )
    if sum(labeled) < 2:
        out = []
        for index, row in enumerate(rows):
            if index and full[index - 1]:
                out[-1] = out[-1] | row
            else:
                out.append(row)
        return out
    first = labeled.index(True)
    out: list[pymupdf.Rect] = []
    for index, (row, has_label) in enumerate(zip(rows, labeled)):
        if index > first and not has_label:
            out[-1] = out[-1] | row
        else:
            out.append(row)
    return out


def _under(rule: pymupdf.Rect, caption: pymupdf.Rect, size: float) -> bool:
    """The rule starts at the left edge of the caption and covers the start of the caption. Rules of another
    algorithm box beside it do not count."""
    return rule.x0 <= caption.x0 + _NEAR and rule.x1 >= min(caption.x1, caption.x0 + size * 4)


def _distinct_rules(rules: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """One rule may be both a line segment and an edge of a box; of rules lying on top of each other only one is
    kept."""
    out: list[pymupdf.Rect] = []
    for rule in rules:
        if out and abs(rule.y0 - out[-1].y0) < _NEAR and abs(rule.x0 - out[-1].x0) < _NEAR:
            continue
        out.append(rule)
    return out


def _code_rows(
    lines: list[dict],
    body: pymupdf.Rect,
    rules: Sequence[pymupdf.Rect] = (),
) -> tuple[pymupdf.Rect, ...]:
    """One cell per line in the box. Lines are split by baselines at body-text size, and superscripts and subscripts
    fall into the line they belong to. rules holds the horizontal rules in the box: fraction bars and radical bars are
    short, frame lines and separators run across the whole box."""
    spans = [
        span
        for line in lines
        for span in line["spans"]
        if span["text"].strip() and body.contains(pymupdf.Point(span["origin"]))
    ]
    if not spans:
        return ()
    main = max(span["size"] for span in spans)
    bars = [rule for rule in rules if rule.width < body.width * 0.5 and body.intersects(rule)]
    baselines: list[float] = []
    # Symbol-font glyphs such as radicals and large braces do not have their origin on the baseline, so they are not
    # used to split lines; numerators and denominators of display formulas are as large as body text but sit against
    # a fraction bar, so they are not used either
    steady = [
        span
        for span in spans
        if span["size"] >= main * 0.85 and not _SYMBOL_FONT_RE.search(span["font"]) and not _on_bar(span, bars, main)
    ]
    for y in sorted(span["origin"][1] for span in steady):
        if not baselines or y - baselines[-1] > main * 0.4:
            baselines.append(y)
    if len(baselines) < 2:
        return ()
    # The boundary sits close to the upper baseline: subscripts and descenders are only slightly below the baseline,
    # while superscripts (the st of 1st) are raised high and would fall into the upper line with a boundary midway
    cuts = [body.y0, *(a + (b - a) * _ROW_CUT for a, b in zip(baselines, baselines[1:])), body.y1]
    rows = [pymupdf.Rect(body.x0, top, body.x1, bottom) for top, bottom in zip(cuts, cuts[1:])]
    return tuple(_join_wrapped_rows(rows, baselines, spans, main))


def _on_bar(span: dict, bars: Sequence[pymupdf.Rect], size: float) -> bool:
    """The span sits right above or below a short horizontal rule and within its horizontal extent: a numerator, a
    denominator, or text under a radical."""
    if not bars:
        return False
    box = pymupdf.Rect(span["bbox"])
    for bar in bars:
        if bar.x0 - 1 > box.x0 or box.x1 > bar.x1 + 1:
            continue
        if -1 <= bar.y0 - box.y1 <= size * 0.3 or -1 <= box.y0 - bar.y1 <= size * 0.3:
            return True
    return False


@dataclass(frozen=True)
class _TextWord:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass(frozen=True)
class _TextCell:
    x0: float
    y0: float
    x1: float
    y1: float
    words: int
    text: str = ""

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def unruled_tables(
    page: pymupdf.Page,
    ruled: Sequence[RuledTable] = (),
    words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """Without rules, only grids of short cells count: column left or right edges align and row heights repeat. words
    holds the words of this page; read now when not given.

    Words at word spacing are merged into one cell. Long lines across a wide column gap are body text, not tables.
    Tables with empty cells, or with two adjacent columns so close that they merge into one cell, have a different
    number of cells per row and are not detected by row alignment. The remaining rows are checked again by the columns
    that the left edges of their cells fall on (see _sparse_grid).
    """
    zones = [table.rect for table in ruled]
    words = [
        word
        for word in (page_words(page) if words is None else words)
        if not _mostly_inside(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), zones)
    ]
    rows = [_row_cells(row) for row in _word_rows(words)]
    tables: list[RuledTable] = []
    index = 0
    while index < len(rows):
        run = [rows[index]]
        cursor = index + 1
        while cursor < len(rows) and _continues_grid(run, rows[cursor]):
            run.append(rows[cursor])
            cursor += 1
        table = _grid_table(run, text_page_rect(page).width)
        if table is not None:
            tables.append(table)
            index = cursor
        else:
            index += 1
    taken = [table.rect for table in tables]
    free = [
        not any(_mostly_inside(pymupdf.Rect(cell.x0, cell.y0, cell.x1, cell.y1), taken) for cell in row) for row in rows
    ]
    # On a two-column page a table in the left column sits next to body text in the right column, and column
    # alignment would join them into one table, so the two sides of the column gap are detected separately
    width = text_page_rect(page).width
    gutters = _find_gutters(
        [
            pymupdf.Rect(cell.x0, cell.y0, cell.x1, cell.y1)
            for row in rows
            for cell in row
            if cell.x1 - cell.x0 <= width * _SPAN_MAX
        ]
    )
    index = 0
    while index < len(rows):
        cursor = index
        while (
            cursor < len(rows)
            and free[cursor]
            and (cursor == index or _continues_rows(rows[index:cursor], rows[cursor]))
            and len(rows[cursor]) >= 2
        ):
            cursor += 1
        table = _sparse_grid(rows[index:cursor], gutters) if cursor - index >= _MIN_TEXT_ROWS else None
        if table is not None:
            tables.append(table)
            index = cursor
        else:
            index += 1
    return tuple(tables)


def _continues_rows(run: Sequence[Sequence[_TextCell]], row: Sequence[_TextCell]) -> bool:
    """The row spacing matches the rows before. The number of cells does not matter"""
    pitch = row[0].y0 - run[-1][0].y0
    first = run[1][0].y0 - run[0][0].y0 if len(run) > 1 else pitch
    return _MIN_PITCH <= pitch <= _MAX_PITCH and abs(pitch - first) <= max(4.0, first * 0.3)


# Page numbers at the end of an index entry: hallucinate, 31, 253
_PAGE_REFS_RE = re.compile(r",\s*\d+$")


def _sparse_grid(
    rows: Sequence[Sequence[_TextCell]],
    gutters: Sequence[tuple[float, float]] = (),
) -> RuledTable | None:
    """Tables with a different number of cells per row (three columns such as TMP when? yesterday evening, with an
    empty cell in some rows): the left edge of every cell falls on some column, and each column is used by at least
    two rows; three or more columns are each used by half of the rows, and adjacent columns are well apart.
    The last column may hold longer example sentences or explanations, the columns before it short cells of one or two
    words. Index subentries are indented only slightly more than entries, so the columns are too close and do not
    count; index entries in several columns ending with page numbers (hallucinate, 31, 253) do not count either.
    A first or last row with long sentences in the earlier columns (a caption below the table) is not part of the
    table. Two adjacent columns separated by the page's column gap hold the text of two separate columns and do not
    count."""
    flat = [cell for row in rows for cell in row]
    if sum(1 for cell in flat if _PAGE_REFS_RE.search(cell.text)) >= len(flat) * 0.3:
        return None
    heads = [cell for row in rows for cell in row[:-1]]
    short = sum(1 for cell in heads if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
    if not heads or short < len(heads) * _SHORT_SHARE:
        return None
    columns: list[list[tuple[int, _TextCell]]] = []
    for row_index, cell in sorted(
        ((index, cell) for index, row in enumerate(rows) for cell in row),
        key=lambda item: item[1].x0,
    ):
        if columns and cell.x0 - columns[-1][-1][1].x0 <= _COL_ALIGN:
            columns[-1].append((row_index, cell))
        else:
            columns.append([(row_index, cell)])
    used = [len({row_index for row_index, _ in column}) for column in columns]
    if min(used) < 2 or sum(1 for count in used if count * 2 >= len(rows)) < 3:
        return None
    anchors = [min(cell.x0 for _, cell in column) for column in columns]
    if any(right - left < _COL_ALIGN * 3 for left, right in zip(anchors, anchors[1:])):
        return None
    if any(
        left < start and end <= right + _COL_ALIGN
        for left, right in zip(anchors, anchors[1:])
        for start, end in gutters
    ):
        return None

    # A caption below the table (Figure 18.12 The A transition ...) has the same row spacing and also falls on the
    # columns, but its long sentence is not in the last column
    def long_head(row: Sequence[_TextCell]) -> bool:
        return any(cell.x0 < anchors[-1] - _COL_ALIGN and cell.words > _SHORT_WORDS for cell in row)

    if long_head(rows[0]) or long_head(rows[-1]):
        inner = rows[1 if long_head(rows[0]) else 0 : -1 if long_head(rows[-1]) else None]
        return _sparse_grid(inner, gutters) if len(inner) >= _MIN_TEXT_ROWS else None
    rights = [anchor - 3.0 for anchor in anchors[1:]]
    rights.append(max(max(cell.x1 for _, cell in columns[-1]) + 8.0, anchors[-1] + 28.0))
    tops = [min(cell.y0 for cell in row) for row in rows]
    bottoms = [top - 1.0 for top in tops[1:]] + [max(cell.y1 for cell in rows[-1]) + 2.0]
    cells = tuple(
        pymupdf.Rect(anchor - 1.0, top - 1.0, max(right, anchor + 18.0), bottom)
        for top, bottom in zip(tops, bottoms)
        for anchor, right in zip(anchors, rights)
    )
    if not _usable_table(cells):
        return None
    rect = pymupdf.Rect()
    for cell in cells:
        rect |= cell
    return RuledTable(rect, cells)


def interlinear_glosses(
    page: pymupdf.Page,
    words: Sequence[_TextWord] | None = None,
) -> tuple[pymupdf.Rect, ...]:
    """Interlinear glosses of linguistic examples: one line of original text, one line of gloss. Translating the gloss
    line word by word makes no sense, and any change of word order breaks the alignment, so both lines keep the
    original text. Two arrangements are detected:

    - Word-by-word alignment: every word of the lower line shares its left edge with some word of the upper line.
      Adjacent lines of body text have only a few words aligned by chance, so at least 75% of the words of each line
      must align. With only two or three words, the upper line must also be in a non-Latin script such as Chinese
      characters or kana, and the lower line in Latin letters.
    - Morpheme-by-morpheme gloss: both lines start at the same x, both consist of morphemes joined by hyphens, and
      the gloss line contains grammatical abbreviations such as 1SG and PFV (Leipzig Glossing Rules).
    - Character-by-character transcription: the upper line is Chinese characters and the lower line starts at the
      same x, consists only of Latin letters and has as many syllables as there are characters. Chinese characters
      have no spaces between them, so word-by-word alignment is impossible.
    - Role labels: the lower line consists of all-capital labels one size smaller (AGENT, ARG0, ARGM-TMP), each
      starting below some word of the upper line. A label refers to a whole phrase and aligns only with its first
      word.

    Tables align like this over many consecutive rows, while a gloss is an isolated pair of lines. words holds the
    words of this page; read now when not given.
    """
    words = list(page_words(page) if words is None else words)
    rows = _with_centered_numbers(_word_rows(words))
    pairs = [_glossed(upper, lower) for upper, lower in zip(rows, rows[1:])]
    found: list[pymupdf.Rect] = []
    for index, paired in enumerate(pairs):
        if not paired or (index > 0 and pairs[index - 1]) or (index + 1 < len(pairs) and pairs[index + 1]):
            continue
        rect = pymupdf.Rect()
        for word in (*rows[index], *rows[index + 1]):
            rect |= pymupdf.Rect(word.x0, word.y0, word.x1, word.y1)
        found.append(rect)
    return tuple(found)


def _with_centered_numbers(rows: list[list[_TextWord]]) -> list[list[_TextWord]]:
    """An example number (22.27) set level with the middle between the example and the label line below it forms a
    line of its own between the two. Merging it into the upper line puts the example and its labels next to each
    other."""
    out: list[list[_TextWord]] = []
    for index, row in enumerate(rows):
        below = rows[index + 1] if index + 1 < len(rows) else None
        if (
            out
            and below is not None
            and len(row) == 1
            and _NUMBER_LABEL_RE.fullmatch(row[0].text)
            and row[0].x1 < min(word.x0 for word in (*out[-1], *below))
            and row[0].y0 < max(word.y1 for word in out[-1])
            and row[0].y1 > min(word.y0 for word in below)
        ):
            out[-1] = [*out[-1], *row]
            continue
        out.append(row)
    return out


@dataclass(frozen=True)
class ExampleNumber:
    """An example number at the start of a line. rect is the number, row is the line of the number, text_x0 is
    where the text after the number starts"""

    rect: pymupdf.Rect
    row: pymupdf.Rect
    text_x0: float


def example_numbers(words: Sequence[_TextWord]) -> tuple[ExampleNumber, ...]:
    """Example numbers at the start of a line: (22.9), (22.11). A number has few letters, so extraction often leaves
    it outside any block while the sentence after it forms a block of its own, or the whole line ((22.12) fall.01)
    forms no block. Numbers with other text right to their left do not count (numbers cited in body text)."""
    found: list[ExampleNumber] = []
    for row in _word_rows(words):
        for index, word in enumerate(row):
            if not _CHAPTER_NUMBER_RE.fullmatch(word.text):
                continue
            height = word.y1 - word.y0
            if index and word.x0 - row[index - 1].x1 < height * 2:
                continue
            rect = pymupdf.Rect(word.x0, word.y0, word.x1, word.y1)
            for other in row[index + 1 :]:
                if other.x0 - rect.x1 > height * 3:
                    break  # another column
                rect |= pymupdf.Rect(other.x0, other.y0, other.x1, other.y1)
            text_x0 = row[index + 1].x0 if index + 1 < len(row) else word.x1
            found.append(ExampleNumber(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), rect, text_x0))
    return tuple(found)


def _glossed(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    height = max(word.y1 - word.y0 for word in upper)
    if lower[0].y0 - upper[0].y0 > height * 1.6:
        return False
    starts = [word.x0 for word in upper]
    # A gloss line consists of words, not a row of a table of numbers
    aligned = [
        word
        for word in lower
        if any(abs(word.x0 - x) <= _GLOSS_ALIGN for x in starts) and any(ch.isalpha() for ch in word.text)
    ]
    hit = [x for x in starts if any(abs(word.x0 - x) <= _GLOSS_ALIGN for word in lower)]
    # When most aligned words equal the word at the same place in the upper line, these are two list items with the
    # same start (Else search from right to left for ...); a gloss line is in another language, so the word at the
    # same place differs
    same = [
        word
        for word in aligned
        if any(abs(word.x0 - other.x0) <= _GLOSS_ALIGN and word.text == other.text for other in upper)
    ]
    if aligned and len(same) >= len(aligned) * 0.5:
        return False
    if len(aligned) >= len(lower) * _GLOSS_SHARE and len(hit) >= len(upper) * _GLOSS_SHARE:
        if len(aligned) >= _GLOSS_MIN:
            return True
        if len(aligned) >= 2 and _script_share(upper, latin=False) >= 0.5 and _script_share(lower, latin=True) == 1:
            return True
    return _morpheme_gloss(upper, lower) or _romanized(upper, lower) or _role_labels(upper, lower)


def _role_labels(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    """Above a label is a word one size larger, and the label lies entirely right below that word. These do not
    count: abbreviations in side notes (DFT, RNN), with side-note words of the same small size above them; subscripts
    (CLS below h), which overlap the line above; capitals in figures, with numbers above them. The size is compared
    with the shortest label in the row, since the subscript of ARG0 makes its box taller."""
    height = min(word.y1 - word.y0 for word in lower)
    return all(
        _ROLE_LABEL_RE.fullmatch(word.text) and any(_labels(word, other, height) for other in upper) for word in lower
    )


def _labels(label: _TextWord, word: _TextWord, height: float) -> bool:
    return (
        word.x0 - _GLOSS_ALIGN * 2 <= label.x0 <= word.x1
        and height < (word.y1 - word.y0) * 0.9
        and word.y1 - 1 <= label.y0 <= word.y1 + height * 0.35
        and any(ch.isalpha() for ch in word.text)
    )


def _romanized(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    if abs(lower[0].x0 - upper[0].x0) > _GLOSS_ALIGN or _script_share(lower, latin=True) < 1:
        return False
    text = "".join(word.text for word in upper)
    hanzi = len(_HANZI_RE.findall(text))
    if hanzi < _GLOSS_MIN or hanzi < len(re.sub(r"[\s\W\d_]", "", text)) * 0.9:
        return False
    syllables = [token for token in _SYLLABLE_SPLIT_RE.split(" ".join(word.text for word in lower)) if token]
    return len(syllables) == hanzi


def _script_share(row: Sequence[_TextWord], latin: bool) -> float:
    """Share of Latin-letter words (latin=True) or of non-Latin-script words among the words with letters in this row.
    Pinyin tone marks in PDFs are often extracted as separate modifier letters (wˇo); characters in that range also
    count as Latin letters."""
    lettered = [word for word in row if any(ch.isalpha() for ch in word.text)]
    if not lettered:
        return 0.0
    matched = [
        word
        for word in lettered
        if all(ord(ch) <= 0x24F or 0x2B0 <= ord(ch) <= 0x2FF for ch in word.text if ch.isalpha()) == latin
    ]
    return len(matched) / len(lettered)


def _morpheme_gloss(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    chains = [word for word in lower if word.text.count("-") >= 2]
    if not chains or sum(len(word.text) for word in chains) < sum(len(word.text) for word in lower) * 0.6:
        return False
    first = min(chains, key=lambda word: word.x0)
    if not any(abs(word.x0 - first.x0) <= _GLOSS_ALIGN and word.text.count("-") >= 2 for word in upper):
        return False
    tokens = _MORPHEME_SPLIT_RE.split(" ".join(word.text for word in chains))
    return any(token in _LEIPZIG or _PERSON_RE.fullmatch(token) for token in tokens)


def _word_rows(words: Sequence[_TextWord]) -> list[list[_TextWord]]:
    rows: list[list[_TextWord]] = []
    for word in sorted(words, key=lambda item: (item.y0, item.x0)):
        if rows and abs(word.y0 - rows[-1][0].y0) <= 4.0:
            rows[-1].append(word)
        else:
            rows.append([word])
    for row in rows:
        row.sort(key=lambda item: item.x0)
    return rows


def _row_cells(row: Sequence[_TextWord]) -> list[_TextCell]:
    if not row:
        return []
    groups: list[list[_TextWord]] = [[row[0]]]
    for word in row[1:]:
        if word.x0 - groups[-1][-1].x1 <= _TEXT_JOIN:
            groups[-1].append(word)
        else:
            groups.append([word])
    cells: list[_TextCell] = []
    for group in groups:
        cells.append(
            _TextCell(
                min(word.x0 for word in group),
                min(word.y0 for word in group),
                max(word.x1 for word in group),
                max(word.y1 for word in group),
                len(group),
                " ".join(word.text for word in group),
            )
        )
    return cells


def _continues_grid(run: Sequence[Sequence[_TextCell]], row: Sequence[_TextCell]) -> bool:
    if len(row) != len(run[0]) or len(row) < 2:
        return False
    pitch = row[0].y0 - run[-1][0].y0
    if pitch < _MIN_PITCH or pitch > _MAX_PITCH:
        return False
    pitches = [run[i + 1][0].y0 - run[i][0].y0 for i in range(len(run) - 1)]
    pitches.append(pitch)
    median = sorted(pitches)[len(pitches) // 2]
    if abs(pitch - median) > max(4.0, median * 0.3):
        return False
    return all(_edge_aligned(anchor, cell) for anchor, cell in zip(run[0], row))


def _edge_aligned(anchor: _TextCell, cell: _TextCell) -> bool:
    return abs(cell.x0 - anchor.x0) <= _COL_ALIGN or abs(cell.x1 - anchor.x1) <= _COL_ALIGN


def _grid_table(rows: Sequence[Sequence[_TextCell]], page_width: float) -> RuledTable | None:
    if len(rows) < _MIN_TEXT_ROWS or len(rows[0]) < 2:
        return None
    flat = [cell for row in rows for cell in row]
    short = sum(1 for cell in flat if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
    if short < len(flat) * _SHORT_SHARE:
        return None
    widths = sorted(cell.width for cell in flat)
    if widths[len(widths) // 2] > min(88.0, max(page_width, 1.0) * 0.28):
        return None
    cells = _grid_cells(rows)
    if not _usable_table(cells):
        return None
    rect = pymupdf.Rect()
    for cell in cells:
        rect |= cell
    return RuledTable(rect, cells)


def _grid_cells(rows: Sequence[Sequence[_TextCell]]) -> tuple[pymupdf.Rect, ...]:
    columns = len(rows[0])
    anchors = [min(row[col].x0 for row in rows) for col in range(columns)]
    rights: list[float] = []
    for col in range(columns - 1):
        rights.append(anchors[col + 1] - 3.0)
    last_text = max(row[-1].x1 for row in rows)
    rights.append(max(last_text + 8.0, anchors[-1] + 28.0))
    tops = [min(cell.y0 for cell in row) for row in rows]
    bottoms: list[float] = []
    for index in range(len(rows) - 1):
        bottoms.append(tops[index + 1] - 1.0)
    bottoms.append(max(cell.y1 for cell in rows[-1]) + 2.0)
    cells: list[pymupdf.Rect] = []
    for top, bottom, _row in zip(tops, bottoms, rows):
        for anchor, right in zip(anchors, rights):
            cells.append(pymupdf.Rect(anchor - 1.0, top - 1.0, max(right, anchor + 18.0), bottom))
    return tuple(cells)


def margin_skips(blocks: Sequence[_MarginBlock], geometries: Sequence[PageGeometry]) -> list[bool]:
    """Short lines repeated in the page margins, and page numbers, keep the original text.

    Only the top and bottom of the page are checked: a fixed narrow band, or the area outside the text frame. Text
    counts as a running header or footer only when the same text appears on the same side on two or more pages,
    ignoring digits (running headers often contain page and chapter numbers). Running headers of a book change with
    the section, so one that appears only once also counts when it sits at the same height as the running headers of
    other pages. Page numbers differ on every page and are detected separately. Numbers inside cells are not page
    numbers.
    """
    zones: list[str | None] = []
    for block in blocks:
        zones.append(_margin_zone(block, geometries))
    seen: dict[tuple[str, str], set[int]] = {}
    for block, zone in zip(blocks, zones):
        if zone is None:
            continue
        key = (zone, _norm_margin(block.text))
        if not key[1] or len(key[1]) > _MARGIN_LEN:
            continue
        seen.setdefault(key, set()).add(block.page)
    repeated = {key for key, pages in seen.items() if len(pages) >= 2}
    rows: dict[str, list[float]] = {}
    for block, zone in zip(blocks, zones):
        if zone is not None and (zone, _norm_margin(block.text)) in repeated:
            rows.setdefault(zone, []).append(block.rect.y0)
    skips: list[bool] = []
    for block, zone in zip(blocks, zones):
        if zone is None:
            skips.append(False)
            continue
        key = (zone, _norm_margin(block.text))
        same_row = (
            0 < len(key[1]) <= _MARGIN_LEN
            and sum(1 for y in rows.get(zone, ()) if abs(y - block.rect.y0) <= _MARGIN_ROW) >= _MARGIN_ROWS
        )
        skips.append(key in repeated or same_row or is_page_number(block.text))
    return skips


def _interval_distance(x: float, column: Column) -> float:
    if x < column.x0:
        return column.x0 - x
    if x > column.x1:
        return x - column.x1
    return 0.0


def _overlaps_y(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.y0 < b.y1 and a.y1 > b.y0


def _mostly_inside(rect: pymupdf.Rect, zones: Sequence[pymupdf.Rect]) -> bool:
    area = rect.width * rect.height
    if area < 1:
        return False
    for zone in zones:
        hit = rect & zone
        if hit.is_empty or hit.width < 0.4 or hit.height < 0.3:
            continue
        if hit.width * hit.height >= area * _INSIDE:
            return True
    return False


def _page_filling(rect: pymupdf.Rect, page_width: float, page_height: float) -> bool:
    if page_height <= 0:
        return False
    return rect.width > page_width * _PAGE_FILL and rect.height > page_height * _PAGE_FILL


def _voting_rects(
    rects: Sequence[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
    tables: Sequence[RuledTable],
    page_width: float,
    page_height: float,
) -> list[pymupdf.Rect]:
    """Text inside tables and labels inside figures take no part in column detection, so that gaps in cells or
    figures are not taken as column gaps."""
    zones = [table.rect for table in tables]
    zones.extend(figure for figure in figures if not _page_filling(figure, page_width, page_height))
    return [rect for rect in rects if not _mostly_inside(rect, zones)]


def _cutting_figures(
    figures: Sequence[pymupdf.Rect],
    page_width: float,
    page_height: float,
) -> list[pymupdf.Rect]:
    return [
        figure
        for figure in figures
        if figure.width >= _WALL_MIN_W
        and figure.height >= _WALL_MIN_H
        and not _page_filling(figure, page_width, page_height)
    ]


def _columns_of(
    page_width: float,
    usable: list[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
) -> tuple[Column, ...]:
    if not usable:
        return (Column(0.0, page_width),)
    body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
    gutters = _merge_gaps([*_find_gutters(body), *_figure_walls(body, figures)])
    if not gutters:
        return (Column(min(rect.x0 for rect in usable), max(rect.x1 for rect in usable)),)
    return _columns_from_gutters(body, gutters, usable)


def _find_gutters(body: list[pymupdf.Rect]) -> list[tuple[float, float]]:
    if len(body) < _MIN_SIDE * 2:
        return []
    found: list[tuple[float, float]] = []
    for left_edge, right_edge in _gaps(_merged_x(body)):
        if right_edge - left_edge < _GUTTER_MIN:
            continue
        mid = (left_edge + right_edge) / 2
        left = [rect for rect in body if (rect.x0 + rect.x1) / 2 < mid]
        right = [rect for rect in body if (rect.x0 + rect.x1) / 2 >= mid]
        if len(left) < _MIN_SIDE or len(right) < _MIN_SIDE:
            continue
        if _group_y_overlap(left, right) < _MIN_Y_OVERLAP:
            continue
        found.append((left_edge, right_edge))
    return found


def _figure_walls(
    body: list[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
) -> list[tuple[float, float]]:
    """With fewer than three text blocks beside a figure, the figure edge still makes that side one column."""
    walls: list[tuple[float, float]] = []
    for figure in figures:
        beside = [rect for rect in body if _overlaps_y(rect, figure) and _x_gap(rect, figure) >= _GUTTER_MIN]
        left = [rect for rect in beside if rect.x1 <= figure.x0 + 1]
        right = [rect for rect in beside if rect.x0 >= figure.x1 - 1]
        if len(left) >= 2 and _group_y_overlap(left, [figure]) >= _MIN_Y_OVERLAP:
            edge = max(rect.x1 for rect in left)
            if figure.x0 - edge >= _GUTTER_MIN:
                walls.append((edge, figure.x0))
        if len(right) >= 2 and _group_y_overlap(right, [figure]) >= _MIN_Y_OVERLAP:
            edge = min(rect.x0 for rect in right)
            if edge - figure.x1 >= _GUTTER_MIN:
                walls.append((figure.x1, edge))
    return walls


def _x_gap(rect: pymupdf.Rect, other: pymupdf.Rect) -> float:
    if rect.x1 <= other.x0:
        return other.x0 - rect.x1
    if rect.x0 >= other.x1:
        return rect.x0 - other.x1
    return 0.0


def _columns_from_gutters(
    body: list[pymupdf.Rect],
    gutters: list[tuple[float, float]],
    usable: list[pymupdf.Rect],
) -> tuple[Column, ...]:
    mids = [(left + right) / 2 for left, right in gutters]
    bounds = [-1e9, *mids, 1e9]
    edges: list[tuple[float | None, float | None]] = [(None, gutters[0][0])]
    for index in range(len(gutters) - 1):
        edges.append((gutters[index][1], gutters[index + 1][0]))
    edges.append((gutters[-1][1], None))
    columns: list[Column] = []
    voters = body or usable
    for (x_start, x_end), (lo, hi) in zip(edges, zip(bounds, bounds[1:])):
        group = [rect for rect in voters if lo < (rect.x0 + rect.x1) / 2 <= hi]
        if not group:
            continue
        columns.append(
            Column(
                min(rect.x0 for rect in group) if x_start is None else x_start,
                max(rect.x1 for rect in group) if x_end is None else x_end,
            )
        )
    if columns:
        return tuple(columns)
    return (Column(min(rect.x0 for rect in usable), max(rect.x1 for rect in usable)),)


def _merge_gaps(gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for x0, x1 in sorted(gaps):
        if x1 - x0 < 1:
            continue
        if merged and x0 <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    return merged


def _merged_x(rects: list[pymupdf.Rect]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for x0, x1 in sorted((rect.x0, rect.x1) for rect in rects):
        if merged and x0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    return merged


def _gaps(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(prev[1], nxt[0]) for prev, nxt in zip(intervals, intervals[1:]) if nxt[0] > prev[1]]


def _group_y_overlap(left: Sequence[pymupdf.Rect], right: Sequence[pymupdf.Rect]) -> float:
    if not left or not right:
        return 0.0
    overlap = min(max(rect.y1 for rect in left), max(rect.y1 for rect in right))
    overlap -= max(min(rect.y0 for rect in left), min(rect.y0 for rect in right))
    return overlap


def _usable_table(cells: tuple[pymupdf.Rect, ...]) -> bool:
    if len(cells) < 2:
        return False
    # Table cells do not overlap. Boxes inside boxes form a flowchart; treated as a table, all text in a box would be
    # merged into one cell
    for index, cell in enumerate(cells):
        for other in cells[index + 1 :]:
            hit = cell & other
            smaller = min(cell.width * cell.height, other.width * other.height)
            if not hit.is_empty and hit.width * hit.height > smaller * 0.25:
                return False
    widths = sorted(cell.width for cell in cells)
    heights = sorted(cell.height for cell in cells)
    if widths[len(widths) // 2] < _CELL_MIN_W or heights[len(heights) // 2] < _CELL_MIN_H:
        return False
    if sum(1 for cell in cells if cell.width < _CELL_NARROW) > len(cells) * 0.4:
        return False
    columns = len({round(cell.x0) for cell in cells})
    return columns <= _MAX_TABLE_COLS


def content_floor(height: float) -> float:
    """Lowest edge that body text can be written to. Below it are page numbers and the footer; text continued
    across pages stops here."""
    if height <= 0:
        return 0.0
    return height - max(_MARGIN_MIN, height * _MARGIN_BAND)


def _margin_zone(block: _MarginBlock, geometries: Sequence[PageGeometry]) -> str | None:
    if block.page < 0 or block.page >= len(geometries):
        return None
    geo = geometries[block.page]
    if geo.height <= 0 or block.rect.height > _MARGIN_HEIGHT:
        return None
    if geo.cell_of(block.rect) is not None:
        return None
    band = max(_MARGIN_MIN, geo.height * _MARGIN_BAND)
    frame = geo.frame
    if block.rect.y1 <= band or (frame is not None and frame.y0 is not None and block.rect.y1 <= frame.y0 - 1):
        return "header"
    floor = content_floor(geo.height)
    if block.rect.y0 >= floor or (frame is not None and frame.y1 is not None and block.rect.y0 >= frame.y1 + 1):
        return "footer"
    return None


def _norm_margin(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text).strip().casefold())


def is_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 16:
        return False
    if _PAGE_NO_RE.match(stripped):
        return True
    return len(stripped) >= 1 and _ROMAN_RE.match(stripped) is not None


def _is_blank_fill(drawing: dict) -> bool:
    """Near-white, filled rectangles only, no stroke. This is background shading, not a figure. Stroked boxes and
    images still count as figures."""
    if drawing.get("type") != "f":
        return False
    fill = drawing.get("fill")
    if not isinstance(fill, (tuple, list)) or len(fill) < 3:
        return False
    if any(float(channel) < 0.96 for channel in fill[:3]):
        return False
    items = drawing.get("items") or []
    return bool(items) and all(isinstance(item, (tuple, list)) and bool(item) and item[0] == "re" for item in items)


def _figure_rects(
    page: pymupdf.Page,
    drawings: list[dict] | None = None,
    images: list[dict] | None = None,
) -> list[pymupdf.Rect]:
    bounds = text_page_rect(page)
    width, height = bounds.width, bounds.height
    found: list[pymupdf.Rect] = []
    for info in _image_infos(page) if images is None else images:
        rect = pymupdf.Rect(info.get("bbox"))
        if rect.width > _FIGURE_MIN and rect.height > _FIGURE_MIN:
            found.append(rect)
    if drawings is None:
        drawings = visible_drawings(page)
    grounds: list[pymupdf.Rect] = []
    marks: list[pymupdf.Rect] = []
    ruled: list[pymupdf.Rect] = []
    for drawing in drawings:
        rect = pymupdf.Rect(drawing.get("rect"))
        if _is_mark(drawing):
            marks.append(rect)
        elif drawing.get("type") == "s":
            ruled.extend(_flat_strokes(drawing))
        if rect.width < _DRAW_MIN or rect.height < _DRAW_MIN or rect.width * rect.height < _DRAW_AREA:
            continue
        if rect.width > width * _PAGE_FILL and rect.height > height * _PAGE_FILL:
            continue
        if _is_blank_fill(drawing):
            grounds.append(rect)
            continue
        found.append(rect)
    framed = [ground for ground in grounds if _framed(ground, ruled)]
    return [*found, *_figure_grounds(grounds, found, marks, framed)]


def _is_mark(drawing: dict) -> bool:
    """Small marks in a figure: colored ellipses circling words, dashed arrows. Solid thin lines are frame lines or
    underlines and do not count"""
    items = drawing.get("items") or []
    if any(item[0] == "c" for item in items):
        return True
    if drawing.get("dashes") and str(drawing.get("dashes")).strip("[] 0"):
        return True
    fill = drawing.get("fill")
    return isinstance(fill, (tuple, list)) and len(fill) >= 3 and any(float(channel) < 0.96 for channel in fill[:3])


def _flat_strokes(drawing: dict) -> list[pymupdf.Rect]:
    """Horizontal lines in a stroked path: line segments, and the top and bottom edges of rectangles."""
    out: list[pymupdf.Rect] = []
    for item in drawing.get("items") or []:
        if item[0] == "l":
            rect = pymupdf.Rect(item[1], item[2]).normalize()
            if rect.height < 1.5:
                out.append(rect)
        elif item[0] == "re":
            rect = pymupdf.Rect(item[1])
            out += [pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0), pymupdf.Rect(rect.x0, rect.y1, rect.x1, rect.y1)]
    return out


def _framed(ground: pymupdf.Rect, ruled: Sequence[pymupdf.Rect]) -> bool:
    """A horizontal line above and below the white ground, each spanning the whole ground and close to its top or
    bottom edge: the white ground is a framed figure."""

    def edge(y: float) -> bool:
        return any(
            abs(line.y0 - y) <= _FRAME_REACH and line.x0 <= ground.x0 + 2 and line.x1 >= ground.x1 - 2 for line in ruled
        )

    return edge(ground.y0) and edge(ground.y1)


def _figure_grounds(
    grounds: Sequence[pymupdf.Rect],
    figures: Sequence[pymupdf.Rect],
    marks: Sequence[pymupdf.Rect] = (),
    framed: Sequence[pymupdf.Rect] = (),
) -> list[pymupdf.Rect]:
    """Plotting programs (matplotlib) first lay down a white ground and write the title, axis labels and legend on it,
    with the curves only in the middle part. A white ground that holds curves or images and is at most twice their
    size is a figure. White grounds laid under body text and figures are much larger than the figures.
    Titles and axis labels sit right next to the curves, so the white ground is only taken up to a margin around the
    curves: some white grounds extend down into the body text.
    A framed white ground with a passage of sample text, colored ellipses circling words and dashed lines connecting
    them (ed3book Figure 4.2) is also a figure: when there are enough of these small marks and they cover most of the
    white ground, they count like curves. An unframed white ground lies under body text, and scattered curves and
    dashed lines on it do not count"""
    kept: list[pymupdf.Rect] = []
    for ground in grounds:
        inside = [figure for figure in figures if _mostly_inside(figure, [ground])]
        dots = [mark for mark in marks if ground in framed and _mostly_inside(mark, [ground])]
        if inside or len(dots) >= _GROUND_MARKS:
            inside += dots
        if not inside:
            continue
        drawn = pymupdf.Rect(inside[0])
        for figure in inside[1:]:
            drawn |= figure
        if ground.width * ground.height <= drawn.width * drawn.height * _GROUND_SHARE:
            kept.append(ground & (drawn + (-_GROUND_REACH, -_GROUND_REACH, _GROUND_REACH, _GROUND_REACH)))
    return kept
