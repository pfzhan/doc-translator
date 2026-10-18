"""页面几何：从文本块的横向间隔认出栏，把图、矢量图形和有线表格收成边界。

不依赖版面模型。跨栏标题（宽过页宽六成）不参与分栏。
栏间空档至少 12pt，左右各有三块以上、纵向交叠够深。空档不必落在页宽正中，
所以三栏也能分开。图把一侧的文字吃掉时，图缘仍是栏墙，写入不会跨过去。
有横竖线的表格，以及列缘对齐的短格子表，用格子限制写入。整段正文不算表。
页眉、页脚和页码另作跳过，不参与分栏。
文字栏沟明确时，阅读顺序按栏从上到下；只靠图缘切开的页面不重排。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar

import pymupdf

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
_WALL_MIN_H = 48.0
_WALL_MIN_W = 36.0
_CELL_MIN_W = 18.0
_CELL_MIN_H = 8.0
_CELL_NARROW = 16.0
_MAX_TABLE_COLS = 12
# 表格线：不超过这么粗、至少这么长
_RULE_THICK = 1.5
_RULE_MIN = 5.0
# 线和格子边算对齐的距离
_NEAR = 2.0
# 算法框相邻两行的分界，从上一行基线往下占行距的比例
_ROW_CUT = 0.35
_TEXT_JOIN = 8.0
_COL_ALIGN = 5.0
_MIN_TEXT_ROWS = 3
_MIN_PITCH = 8.0
_MAX_PITCH = 36.0
_SHORT_WORDS = 3
_SHORT_WIDTH = 80.0
_SHORT_SHARE = 0.8
_INSIDE = 0.6
_MARGIN_BAND = 0.07
_MARGIN_MIN = 42.0
_MARGIN_HEIGHT = 22.0
_MARGIN_LEN = 80
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


@dataclass(frozen=True)
class PageGeometry:
    width: float
    columns: tuple[Column, ...]
    figures: tuple[pymupdf.Rect, ...]
    height: float = 0.0
    tables: tuple[RuledTable, ...] = ()
    # 图缘可以挡住写入，但一侧文字不够三块时不能据此重排。
    reads_in_columns: bool = False

    def index_of(self, rect: pymupdf.Rect) -> int:
        """块中心落在哪一栏。落在栏间空白时取最近的一栏。"""
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
        """跨栏标题和整表。中心会落进某一栏，不能拿来当栏首或栏尾。"""
        return _spans_columns(rect, self)

    def cell_of(self, rect: pymupdf.Rect) -> pymupdf.Rect | None:
        return cell_containing(rect, self.tables)

    def right_limit(self, rect: pymupdf.Rect, obstacles: list[pymupdf.Rect]) -> float:
        """本栏右缘，并在纵向交叠的右侧障碍前停下。可以小于 rect.x1，调用方不得因此把框缩小。"""
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
    ) -> PageGeometry:
        table_tuple = tuple(tables or ())
        table_zones = [table.rect for table in table_tuple]
        # 表格外框也是一块大矢量，不能再当成图，否则格子里的字会被整段跳过
        figure_tuple = tuple(
            figure for figure in (figures or ()) if not _mostly_inside(figure, table_zones)
        )
        voting = _voting_rects(rects, figure_tuple, table_tuple, page_width, height)
        usable = [rect for rect in voting if rect.width >= _USABLE_W and rect.height >= _USABLE_H]
        body = [rect for rect in usable if rect.width <= page_width * _SPAN_MAX]
        columns = _columns_of(page_width, usable, _cutting_figures(figure_tuple, page_width, height))
        return PageGeometry(
            page_width, columns, figure_tuple, height, table_tuple,
            reads_in_columns=bool(_find_gutters(body)),
        )

    @staticmethod
    def from_page(page: pymupdf.Page, text_rects: list[pymupdf.Rect]) -> PageGeometry:
        bounds = text_page_rect(page)
        return PageGeometry.from_rects(
            bounds.width,
            text_rects,
            _figure_rects(page),
            bounds.height,
            page_tables(page),
        )


class _MarginBlock(Protocol):
    page: int
    rect: pymupdf.Rect
    text: str


def text_page_rect(page: pymupdf.Page) -> pymupdf.Rect:
    """文字坐标所在的页面框。旋转页的 page.rect 是转过的，提取出来的坐标没转。"""
    return pymupdf.Rect(0, 0, page.cropbox.width, page.cropbox.height)


def cell_containing(rect: pymupdf.Rect, tables: Sequence[RuledTable]) -> pymupdf.Rect | None:
    """块的大部分落在哪个格子里。落在线外的正文返回 None。"""
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
    """栏沟明确时按栏从上到下。跨栏标题和整表留在纵向位置，不塞进某一栏。

    单栏、以及只靠图缘切开的页面保持传入顺序。表内格子已有行序，跨栏表不拆开。
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
        units.append(_ReadUnit(
            table.rect.y0, table.rect.x0, geo.index_of(table.rect), pairs[0][0], True, cells,
        ))
    for order, block in loose:
        units.append(_ReadUnit(
            block.rect.y0,
            block.rect.x0,
            geo.index_of(block.rect),
            order,
            _spans_columns(block.rect, geo),
            (block,),
        ))
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


def ruled_tables(page: pymupdf.Page) -> tuple[RuledTable, ...]:
    """只认横竖线围出的格子。文字对齐出来的假表、公式笔画网格不要。"""
    pymupdf.no_recommend_layout()
    try:
        found = page.find_tables(vertical_strategy="lines_strict", horizontal_strategy="lines_strict")
    except Exception:  # noqa: BLE001 - 个别页面的线读不出来，当没有表
        return ()
    tables: list[RuledTable] = []
    rules: tuple[list[pymupdf.Rect], list[pymupdf.Rect]] | None = None
    for table in found.tables:
        cells = tuple(pymupdf.Rect(cell) for cell in table.cells if cell)
        if not _usable_table(cells):
            continue
        if rules is None:
            rules = _rule_segments(page)
        tables.append(_with_open_edges(RuledTable(pymupdf.Rect(table.bbox), cells), *rules))
    return tuple(tables)


def _rule_segments(page: pymupdf.Page) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """页面上的细横线和细竖线。

    线段、细长的矩形、描边矩形的四条边都算；老 TeX 和一些转换器把框线画成一像素宽的
    图片，也算。"""
    rects: list[pymupdf.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - 读不出来就当没有线
        drawings = []
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
    try:
        rects.extend(pymupdf.Rect(info["bbox"]) for info in page.get_image_info())
    except Exception:  # noqa: BLE001
        pass
    horizontal = [rect for rect in rects if rect.height <= _RULE_THICK and rect.width >= _RULE_MIN]
    vertical = [rect for rect in rects if rect.width <= _RULE_THICK and rect.height >= _RULE_MIN]
    return horizontal, vertical


def _edges_of(rect: pymupdf.Rect) -> list[pymupdf.Rect]:
    return [
        pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0), pymupdf.Rect(rect.x0, rect.y1, rect.x1, rect.y1),
        pymupdf.Rect(rect.x0, rect.y0, rect.x0, rect.y1), pymupdf.Rect(rect.x1, rect.y0, rect.x1, rect.y1),
    ]


def _with_open_edges(
    table: RuledTable, horizontal: list[pymupdf.Rect], vertical: list[pymupdf.Rect],
) -> RuledTable:
    """补上没有外框的边行、边列。

    很多论文的表只画格子之间的线：横线伸出最外一条竖线，竖线伸过最上、最下一条横线。
    find_tables 只认四面都有线的格子，表头、首列、末列、末行就漏了。按线伸出去的范围
    把网格补全；相邻两格之间没有线隔开的（跨列的表头）并成一格。
    """
    box = pymupdf.Rect(table.rect)
    for _ in range(3):
        ys = [v for v in vertical if box.x0 - _NEAR <= v.x0 <= box.x1 + _NEAR
              and v.y1 >= box.y0 - _NEAR and v.y0 <= box.y1 + _NEAR]
        xs = [h for h in horizontal if box.y0 - _NEAR <= h.y0 <= box.y1 + _NEAR
              and h.x1 >= box.x0 - _NEAR and h.x0 <= box.x1 + _NEAR]
        grown = pymupdf.Rect(
            min([box.x0, *(h.x0 for h in xs)]), min([box.y0, *(v.y0 for v in ys)]),
            max([box.x1, *(h.x1 for h in xs)]), max([box.y1, *(v.y1 for v in ys)]),
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
    """at 处有线盖住 start 到 end 的大半。upright 为真时找竖线，否则找横线。"""
    covered = 0.0
    for seg in segments:
        pos = seg.x0 if upright else seg.y0
        if abs(pos - at) > _NEAR:
            continue
        lo, hi = (seg.y0, seg.y1) if upright else (seg.x0, seg.x1)
        covered += max(0.0, min(hi, end) - max(lo, start))
    return covered >= (end - start) * 0.6


def _merge_rows(cells: list[pymupdf.Rect], horizontal: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """上下相邻、左右对齐、中间没有横线的边格并成一格（跨行的行名）。"""
    out: list[pymupdf.Rect] = []
    for cell in sorted(cells, key=lambda c: (c.x0, c.y0)):
        prev = out[-1] if out else None
        if (
            prev is not None and abs(prev.x0 - cell.x0) < _NEAR and abs(prev.x1 - cell.x1) < _NEAR
            and abs(prev.y1 - cell.y0) < _NEAR and not _ruled(horizontal, cell.y0, cell.x0, cell.x1, upright=False)
        ):
            out[-1] = prev | cell
            continue
        out.append(cell)
    return out


def page_tables(page: pymupdf.Page) -> tuple[RuledTable, ...]:
    """有线表，再加上列缘对齐的短格子表和算法框。两栏正文和整段文字不在这里。"""
    ruled = ruled_tables(page)
    algorithms = algorithm_tables(page)
    return (*ruled, *unruled_tables(page, (*ruled, *algorithms)), *algorithms)


_SYMBOL_FONT_RE = re.compile(r"(?:^|\+)(?:CMSY|CMEX|MSAM|MSBM)", re.I)
_ALGORITHM_RE = re.compile(r"^\s*(?:Algorithm|Procedure|算法)\s*\d+", re.I)


def algorithm_tables(page: pymupdf.Page) -> tuple[RuledTable, ...]:
    """算法框里的伪代码当成一列的表，每一行是一格。

    伪代码一行一句，缩进是结构，行和行不是一段话。PyMuPDF 常把几行收进同一块，
    后面的合并又会把它们接成段落重排。做成格子以后，每行单独翻译、写回原行，
    不和别的行合并。算法框的范围按框线定：图注下面那条横线到再下面一条横线。
    """
    try:
        data = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return ()
    lines = [
        line for block in data.get("blocks", []) if block.get("type") == 0
        for line in block["lines"] if abs(line["dir"][1]) <= 0.01
    ]
    captions = [caption for line in lines for caption in _algorithm_captions(line)]
    if not captions:
        return ()
    horizontal, _vertical = _rule_segments(page)
    tables: list[RuledTable] = []
    for box, size in captions:
        body = _box_below(box, size, horizontal) or _box_above(box, size, horizontal)
        if body is None:
            continue
        rows = _code_rows(lines, body, horizontal)
        if len(rows) >= 2:
            tables.append(RuledTable(body, rows))
    return tuple(tables)


def _algorithm_captions(line: dict) -> list[tuple[pymupdf.Rect, float]]:
    """一行里的算法图注（框、字号）。并排的两个算法框，图注常在同一行里，中间隔着栏沟。"""
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
    """图注在上：图注下面的第一条线是框顶，再下面一条是框底。图注可能有好几行。"""
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
    """图注在下：紧挨图注上方的线是框底，再往上一条同宽的线是框顶。"""
    rules = sorted(
        (rule for rule in horizontal if rule.y1 <= caption.y0 + _NEAR and _under(rule, caption, size)),
        key=lambda rule: -rule.y0,
    )
    rules = _distinct_rules(rules)
    if len(rules) < 2 or caption.y0 - rules[0].y1 > size * 2:
        return None
    bottom = rules[0]
    top = next((rule for rule in rules[1:] if abs(rule.x0 - bottom.x0) < _NEAR and abs(rule.x1 - bottom.x1) < _NEAR), None)
    if top is None:
        return None
    return pymupdf.Rect(bottom.x0, top.y1, bottom.x1, bottom.y0)


_LINE_LABEL_RE = re.compile(r"^\s*\d+\s*[:.]")


def _join_wrapped_rows(
    rows: list[pymupdf.Rect], baselines: list[float], spans: list[dict], main: float,
) -> list[pymupdf.Rect]:
    """行首有行号（1: 2: …）的算法里，行号之后没有行号的行是上一句折下来的，并进上一格。

    折开的两半各自送翻，半句里正文词太少，会被当成公式留下英文。
    第一个行号之前的 Input:、Output: 各是一格。"""
    labeled: list[bool] = []
    for baseline in baselines:
        row = sorted(
            (span for span in spans if abs(span["origin"][1] - baseline) <= main * 0.4),
            key=lambda span: span["origin"][0],
        )
        labeled.append(bool(_LINE_LABEL_RE.match("".join(span["text"] for span in row))))
    if sum(labeled) < 2:
        return rows
    first = labeled.index(True)
    out: list[pymupdf.Rect] = []
    for index, (row, has_label) in enumerate(zip(rows, labeled)):
        if index > first and not has_label:
            out[-1] = out[-1] | row
        else:
            out.append(row)
    return out


def _under(rule: pymupdf.Rect, caption: pymupdf.Rect, size: float) -> bool:
    """线从图注左缘起、盖过图注开头。并排的另一个算法框的线在旁边，不算。"""
    return rule.x0 <= caption.x0 + _NEAR and rule.x1 >= min(caption.x1, caption.x0 + size * 4)


def _distinct_rules(rules: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """同一条线可能既是线段又是框的一边，叠在一起的只留一条。"""
    out: list[pymupdf.Rect] = []
    for rule in rules:
        if out and abs(rule.y0 - out[-1].y0) < _NEAR and abs(rule.x0 - out[-1].x0) < _NEAR:
            continue
        out.append(rule)
    return out


def _code_rows(
    lines: list[dict], body: pymupdf.Rect, rules: Sequence[pymupdf.Rect] = (),
) -> tuple[pymupdf.Rect, ...]:
    """框里每一行一格。行按正文字号的基线分，上下标落在它们所属的行里。
    rules 是框里的横线：分数线、根号上的横线短，框线和分隔线横贯整个框。"""
    spans = [
        span for line in lines for span in line["spans"]
        if span["text"].strip() and body.contains(pymupdf.Point(span["origin"]))
    ]
    if not spans:
        return ()
    main = max(span["size"] for span in spans)
    bars = [rule for rule in rules if rule.width < body.width * 0.5 and body.intersects(rule)]
    baselines: list[float] = []
    # 根号、大括号这类符号字体的起笔点不在基线上，不拿它们分行；
    # 行间公式的分子分母和正文一样大，贴着分数线，也不拿它们分行
    steady = [
        span for span in spans
        if span["size"] >= main * 0.85 and not _SYMBOL_FONT_RE.search(span["font"])
        and not _on_bar(span, bars, main)
    ]
    for y in sorted(span["origin"][1] for span in steady):
        if not baselines or y - baselines[-1] > main * 0.4:
            baselines.append(y)
    if len(baselines) < 2:
        return ()
    # 分界放在偏上一行基线的地方：下标和降部只比基线低一点，上标（1st 的 st）却抬得很高，
    # 放在正中会被划进上一行
    cuts = [body.y0, *(a + (b - a) * _ROW_CUT for a, b in zip(baselines, baselines[1:])), body.y1]
    rows = [pymupdf.Rect(body.x0, top, body.x1, bottom) for top, bottom in zip(cuts, cuts[1:])]
    return tuple(_join_wrapped_rows(rows, baselines, spans, main))


def _on_bar(span: dict, bars: Sequence[pymupdf.Rect], size: float) -> bool:
    """span 紧贴在一条短横线的上面或下面，横向落在线的范围里：分子、分母、根号里的字。"""
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

    @property
    def width(self) -> float:
        return self.x1 - self.x0


def unruled_tables(page: pymupdf.Page, ruled: Sequence[RuledTable] = ()) -> tuple[RuledTable, ...]:
    """没有线时，只认短格子组成的网格：列的左缘或右缘对齐，行高重复。

    词间距并成一格。栏间大空档上的长行是正文，不是表。
    """
    zones = [table.rect for table in ruled]
    words = [
        _TextWord(float(item[0]), float(item[1]), float(item[2]), float(item[3]), str(item[4]))
        for item in page.get_text("words")
        if str(item[4]).strip()
        and not _mostly_inside(pymupdf.Rect(item[:4]), zones)
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
    return tuple(tables)


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
        cells.append(_TextCell(
            min(word.x0 for word in group),
            min(word.y0 for word in group),
            max(word.x1 for word in group),
            max(word.y1 for word in group),
            len(group),
        ))
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
    for top, bottom, row in zip(tops, bottoms, rows):
        for anchor, right in zip(anchors, rights):
            cells.append(pymupdf.Rect(anchor - 1.0, top - 1.0, max(right, anchor + 18.0), bottom))
    return tuple(cells)


def margin_skips(blocks: Sequence[_MarginBlock], geometries: Sequence[PageGeometry]) -> list[bool]:
    """页边重复的短行，以及页码，保留原文。

    只看页顶和页底一条窄带。同一侧、同一文本出现在两页以上才算页眉页脚。
    页码每页不同，单独认。格子里的数字不是页码。
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
    skips: list[bool] = []
    for block, zone in zip(blocks, zones):
        if zone is None:
            skips.append(False)
            continue
        key = (zone, _norm_margin(block.text))
        skips.append(key in repeated or _is_page_number(block.text))
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
    """表内文字和图内标签不参与分栏，避免把格子或图中的缝当成栏沟。"""
    zones = [table.rect for table in tables]
    zones.extend(
        figure for figure in figures if not _page_filling(figure, page_width, page_height)
    )
    return [rect for rect in rects if not _mostly_inside(rect, zones)]


def _cutting_figures(
    figures: Sequence[pymupdf.Rect], page_width: float, page_height: float,
) -> list[pymupdf.Rect]:
    return [
        figure for figure in figures
        if figure.width >= _WALL_MIN_W and figure.height >= _WALL_MIN_H
        and not _page_filling(figure, page_width, page_height)
    ]


def _columns_of(
    page_width: float, usable: list[pymupdf.Rect], figures: Sequence[pymupdf.Rect],
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
    body: list[pymupdf.Rect], figures: Sequence[pymupdf.Rect],
) -> list[tuple[float, float]]:
    """图旁文字不够三块时，图缘仍把这一侧收成一栏。"""
    walls: list[tuple[float, float]] = []
    for figure in figures:
        beside = [
            rect for rect in body
            if _overlaps_y(rect, figure) and _x_gap(rect, figure) >= _GUTTER_MIN
        ]
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
        columns.append(Column(
            min(rect.x0 for rect in group) if x_start is None else x_start,
            max(rect.x1 for rect in group) if x_end is None else x_end,
        ))
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
    return [
        (prev[1], nxt[0])
        for prev, nxt in zip(intervals, intervals[1:])
        if nxt[0] > prev[1]
    ]


def _group_y_overlap(left: Sequence[pymupdf.Rect], right: Sequence[pymupdf.Rect]) -> float:
    if not left or not right:
        return 0.0
    overlap = min(max(rect.y1 for rect in left), max(rect.y1 for rect in right))
    overlap -= max(min(rect.y0 for rect in left), min(rect.y0 for rect in right))
    return overlap


def _usable_table(cells: tuple[pymupdf.Rect, ...]) -> bool:
    if len(cells) < 2:
        return False
    # 表的格子互不重叠。框套框的是流程图，当表的话框里的字全并成一格
    for index, cell in enumerate(cells):
        for other in cells[index + 1:]:
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
    """正文能写到的最下沿。再往下是页码和页脚，跨页续写停在这里。"""
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
    if block.rect.y1 <= band:
        return "header"
    if block.rect.y0 >= content_floor(geo.height):
        return "footer"
    return None


def _norm_margin(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _is_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 16:
        return False
    if _PAGE_NO_RE.match(stripped):
        return True
    return len(stripped) >= 1 and _ROMAN_RE.match(stripped) is not None


def _is_blank_fill(drawing: dict) -> bool:
    """近白、只有矩形填充、没有描边。这是底色，不是图。描边框和图片仍算图。"""
    if drawing.get("type") != "f":
        return False
    fill = drawing.get("fill")
    if not isinstance(fill, (tuple, list)) or len(fill) < 3:
        return False
    if any(float(channel) < 0.96 for channel in fill[:3]):
        return False
    items = drawing.get("items") or []
    return bool(items) and all(
        isinstance(item, (tuple, list)) and bool(item) and item[0] == "re" for item in items
    )


def _figure_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    bounds = text_page_rect(page)
    width, height = bounds.width, bounds.height
    found: list[pymupdf.Rect] = []
    for info in page.get_image_info():
        rect = pymupdf.Rect(info.get("bbox"))
        if rect.width > _FIGURE_MIN and rect.height > _FIGURE_MIN:
            found.append(rect)
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - 个别页面的矢量表读不出来，当没有图
        drawings = []
    for drawing in drawings:
        if _is_blank_fill(drawing):
            continue
        rect = pymupdf.Rect(drawing.get("rect"))
        if rect.width < _DRAW_MIN or rect.height < _DRAW_MIN:
            continue
        if rect.width * rect.height < _DRAW_AREA:
            continue
        if rect.width > width * _PAGE_FILL and rect.height > height * _PAGE_FILL:
            continue
        found.append(rect)
    return found
