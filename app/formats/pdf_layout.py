"""页面几何：从文本块的横向间隔认出栏，把图、矢量图形和有线表格收成边界。

不依赖版面模型。跨栏标题（宽过页宽六成）不参与分栏。
栏间空档至少 12pt，左右各有三块以上、纵向交叠够深。空档不必落在页宽正中，
所以三栏也能分开。图把一侧的文字吃掉时，图缘仍是栏墙，写入不会跨过去。
有横竖线的表格，以及列缘对齐的短格子表，用格子限制写入。整段正文不算表。
页眉、页脚和页码另作跳过，不参与分栏。
文字栏沟明确时，阅读顺序按栏从上到下；只靠图缘切开的页面不重排。
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar

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
_GROUND_SHARE = 2.0  # 图的白底最多是里面曲线、图片所占范围的几倍
_GROUND_REACH = 24.0  # 标题、坐标轴名离曲线最远多少
_WALL_MIN_H = 48.0
_WALL_MIN_W = 36.0
_CELL_MIN_W = 18.0
_CELL_MIN_H = 8.0
_CELL_NARROW = 16.0
_MAX_TABLE_COLS = 12
# 表格线：不超过这么粗、至少这么长
_RULE_THICK = 1.5
_RULE_MIN = 5.0
# find_tables 的吸附距离：比它细的填充块也当线
_SNAP = 3.0
# 线和格子边算对齐的距离
_NEAR = 2.0
# 算法框相邻两行的分界，从上一行基线往下占行距的比例
_ROW_CUT = 0.35
_TEXT_JOIN = 8.0
_COL_ALIGN = 5.0
_MIN_TEXT_ROWS = 3
_MIN_PITCH = 8.0
# 逐词对译：左缘对齐的容差、至少几个词、两行各有多少比例的词对得上
_GLOSS_ALIGN = 1.0
_GLOSS_MIN = 4
_GLOSS_SHARE = 0.75
# 莱比锡标注规则里常用的语法缩写。单个字母（S、A、P）在别处太常见，不算
_LEIPZIG = frozenset(
    "SG PL DU NOM ACC GEN DAT ERG ABS LOC ABL ALL INS INSTR COM VOC PST PRS FUT PFV IPFV PRF IMPF "
    "PROG NEG DEF INDF CLF COP AUX POSS REFL RECP CAUS PASS APPL TOP FOC DECL IMP SBJ OBJ OBL DEM "
    "PTCP INF NMLZ COMP REL".split()
)
_PERSON_RE = re.compile(r"[123](?:SG|PL|DU)")
_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
# 注音行里的音节按空格和标点分开。抽出来的声调符号是单独的字符（z`ai、zh¯u），不算分隔
_SYLLABLE_SPLIT_RE = re.compile(r"[\s,.;:!?，。；：！？、()（）\[\]“”\"'‘’]+")
_MORPHEME_SPLIT_RE = re.compile(r"[-.=\[\]<>:~()\s]+")
_MAX_PITCH = 36.0
_SHORT_WORDS = 3
_SHORT_WIDTH = 80.0
_SHORT_SHARE = 0.8
# 三线表顶线到中线之间最多几行字高
_HEAD_ROWS = 4.0
# 三线表的横线至少这么长。正文里给词加的下划线、分式线都短得多
_BAND_MIN_W = 120.0
_BAND_SHORT_SHARE = 0.6
_LABEL_RE = re.compile(r"(?:Figure|Fig\.|Table|Tab\.)\s*\d+(?:\.\d+)*\b")
_INSIDE = 0.6
_MARGIN_BAND = 0.07
_MARGIN_MIN = 42.0
_MARGIN_HEIGHT = 22.0
_MARGIN_LEN = 80
# 和这么多条重复页眉落在同一高度（容差内）的短行也是页眉
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
    # 算法框里的伪代码也按一列的表切格子，但要逐行翻译。别的表格留原文
    algorithm: bool = False


@dataclass(frozen=True)
class PageGeometry:
    width: float
    columns: tuple[Column, ...]
    figures: tuple[pymupdf.Rect, ...]
    height: float = 0.0
    tables: tuple[RuledTable, ...] = ()
    # 图缘可以挡住写入，但一侧文字不够三块时不能据此重排。
    reads_in_columns: bool = False
    # 整份文档投出来的版心。栏左缘是本页最左的块，版心左缘是正文起笔的地方
    frame: TextFrame | None = None

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
        frame: TextFrame | None = None,
    ) -> PageGeometry:
        table_tuple = tuple(tables or ())
        table_zones = [table.rect for table in table_tuple]
        # 表格外框也是一块大矢量，不能再当成图，否则格子里的字会被整段跳过。
        # 比表大出一圈的是图：带网格的坐标图，网格认成了表，标题、坐标轴名在网格外面
        figure_tuple = tuple(
            figure for figure in (figures or ())
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
            page_width, columns, figure_tuple, height, table_tuple,
            reads_in_columns=bool(_find_gutters(body)), frame=frame,
        )

    @staticmethod
    def from_page(
        page: pymupdf.Page, text_rects: list[pymupdf.Rect], frame: TextFrame | None = None,
        scan: PageScan | None = None,
    ) -> PageGeometry:
        """scan 是这一页已经读出的表格和图。没给时现读。"""
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
    """一页的表格、图和词。提取、阅读顺序、角色判断、逐词对译都要用；
    找表要逐页解析线和字，线、图片框、词每读一次都要几毫秒到几十毫秒，所以每页只读一次。"""
    tables: tuple[RuledTable, ...]
    figures: tuple[pymupdf.Rect, ...]
    words: tuple[_TextWord, ...]
    # 相连的浅色底并成的一片：一张表、一个文本框的底色
    shades: tuple[pymupdf.Rect, ...] = ()


def scan_page(page: pymupdf.Page) -> PageScan:
    images = _image_infos(page)
    drawings = _page_drawings(page, images)
    words = page_words(page)
    return PageScan(
        page_tables(page, drawings, words), tuple(_figure_rects(page, drawings, images)), words,
        _shaded_areas(drawings, text_page_rect(page)),
    )


def _image_infos(page: pymupdf.Page) -> list[dict]:
    try:
        return page.get_image_info()
    except Exception:  # noqa: BLE001 - 读不出来就当没有图片
        return []


def _page_drawings(page: pymupdf.Page, images: list[dict] | None = None) -> list[dict]:
    """本页的矢量图，加上图片画的细线。images 是本页的图片框，没给时现读。"""
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 - 个别页面的线读不出来，当没有线
        drawings = []
    return [*drawings, *_image_rules(_image_infos(page) if images is None else images)]


def light_fill(drawing: dict) -> bool:
    """只填色、由矩形拼成、两边都比线粗的浅色块：表格格子、文本框的底色。
    深色块不算：图注标签是黑底白字。"""
    rect = pymupdf.Rect(drawing["rect"])
    items = drawing.get("items") or []
    fill = drawing.get("fill")
    return (
        drawing.get("type") == "f" and bool(items) and all(item[0] in ("re", "qu") for item in items)
        and min(rect.width, rect.height) > _RULE_THICK
        and fill is not None and len(fill) == 3 and 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2] > 0.6
    )


def _shaded_areas(drawings: list[dict], bounds: pymupdf.Rect) -> tuple[pymupdf.Rect, ...]:
    """挨着的浅色块并成一片。一张表常是每格一块底色，格子之间留一道细缝。
    铺满整页的是纸张底色，不算。"""
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
        for item in page.get_text("words") if str(item[4]).strip()
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


def ruled_tables(page: pymupdf.Page, drawings: list[dict] | None = None) -> tuple[RuledTable, ...]:
    """只认横竖线围出的格子。文字对齐出来的假表、公式笔画网格不要。drawings 是本页的矢量图
    （含图片画的线，见 _page_drawings），没给时现读。

    find_tables 每次先把整页的字逐个转成字典，一页要几十毫秒。页面上凑不出两横两竖的线，
    就围不出格子，不必调用。"""
    if drawings is None:
        drawings = _page_drawings(page)
    if not _grid_possible(drawings):
        return ()
    pymupdf.no_recommend_layout()
    try:
        found = page.find_tables(
            vertical_strategy="lines_strict", horizontal_strategy="lines_strict", paths=drawings,
        )
    except Exception:  # noqa: BLE001 - 个别页面的线读不出来，当没有表
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


def _image_rules(images: Sequence[dict]) -> list[dict]:
    """一像素宽的图片拉成的细条，按 get_drawings 的格式写成填充矩形。

    老 TeX 和一些转换器用图片画框线，get_drawings 读不到，find_tables 也就看不见。"""
    out: list[dict] = []
    for info in images:
        rect = pymupdf.Rect(info["bbox"])
        if min(rect.width, rect.height) <= _RULE_THICK and max(rect.width, rect.height) >= _RULE_MIN:
            out.append({
                "type": "f", "rect": rect, "items": [("re", rect, 1)], "closePath": False,
                "fill": (0.0, 0.0, 0.0), "color": None, "width": 0.0,
            })
    return out


def _grid_possible(drawings: Sequence[dict]) -> bool:
    """横线、竖线各至少两条。

    按 lines_strict 取边的口径数：线段、矩形和四边形的边都算，不分长短（短线会接成长线）；
    只填充不描边、两个方向都比吸附距离宽的块不当线。"""
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
    page: pymupdf.Page, drawings: list[dict] | None = None,
) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """页面上的细横线和细竖线。drawings 是本页的矢量图（含图片画的线），没给时现读。

    线段、细长的矩形、描边矩形的四条边都算。"""
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


def page_tables(
    page: pymupdf.Page, drawings: list[dict] | None = None, words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """有线表、底色格子拼成的表，再加上列缘对齐的短格子表和算法框。两栏正文和整段文字不在这里。
    drawings、words 是本页已读出的矢量图和词，没给时现读。"""
    if drawings is None:
        drawings = _page_drawings(page)
    if words is None:
        words = page_words(page)
    ruled = tuple(_with_text_rows(table, words) for table in ruled_tables(page, drawings))
    ruled = (*ruled, *filled_tables(page, drawings, words, ruled))
    ruled = (*ruled, *column_ruled_tables(_rule_segments(page, drawings)[1], words, ruled))
    algorithms = algorithm_tables(page, drawings)
    banded = banded_tables(_rule_segments(page, drawings)[0], words, (*ruled, *algorithms))
    ruled = (*(table for table in ruled if not _mostly_inside(table.rect, [band.rect for band in banded])), *banded)
    # 无线的格子表本来就是按对齐的行找的，表头对齐的话已经在里面，不再往上找
    ruled = _with_header_rows(ruled, words)
    unruled = unruled_tables(page, (*ruled, *algorithms), words)
    if unruled:
        unruled = _with_ruled_heads(unruled, words, _rule_segments(page, drawings)[0])
    return (*ruled, *unruled, *algorithms)


def _with_header_rows(tables: Sequence[RuledTable], words: Sequence[_TextWord]) -> tuple[RuledTable, ...]:
    """三线表的表头和表身隔着横线，按线只找得到表身。紧贴在表上面、横向落在表的范围里、
    每个词都落在表的某一列里（不跨列线）、占三列以上、每列不过三个词的行是表头，
    一行一格并进这张表。表上面的说明、正文是连着排的句子，会跨过列线，到那里为止。"""
    zones = [table.rect for table in tables]
    out: list[RuledTable] = []
    for table in tables:
        rect = table.rect
        lines = sorted({x for cell in table.cells for x in (cell.x0, cell.x1) if rect.x0 + 2 < x < rect.x1 - 2})
        above = [
            word for word in words
            if word.x0 >= rect.x0 - 2 and word.x1 <= rect.x1 + 2 and word.y1 <= rect.y0 + 1
            and not _mostly_inside(pymupdf.Rect(word.x0, word.y0, word.x1, word.y1), zones)
        ]
        top = rect.y0
        headers: list[pymupdf.Rect] = []
        for row in reversed(_word_rows(above)):
            box = pymupdf.Rect(min(w.x0 for w in row), min(w.y0 for w in row),
                               max(w.x1 for w in row), max(w.y1 for w in row))
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
    vertical: Sequence[pymupdf.Rect], words: Sequence[_TextWord], known: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """只画了列线的表：没有外框，横线也只有表头下面一条，find_tables 围不出格子；
    数字居中排，按左右缘对齐也认不出。几条竖线上下一样长，竖线范围里每行字在每条竖线
    两边都有，多是短格，就按竖线分列、按字分行。两栏正文中间的分隔线两边是长句，不算。"""
    lines = _joined_verticals(vertical)
    tables: list[RuledTable] = []
    used: set[int] = set()
    for index, line in enumerate(lines):
        if index in used:
            continue
        group = [other for other in lines if other.height > 0
                 and min(line.y1, other.y1) - max(line.y0, other.y0) >= 0.8 * min(line.height, other.height)]
        used.update(lines.index(other) for other in group)
        xs = sorted(other.x0 for other in group)
        top, bottom = min(other.y0 for other in group), max(other.y1 for other in group)
        inside = [word for word in words if top < (word.y0 + word.y1) / 2 < bottom]
        rows = _word_rows(inside)
        if len(rows) < _MIN_TEXT_ROWS:
            continue
        sided = [row for row in rows
                 if all(any(w.x1 <= x for w in row) and any(w.x0 >= x for w in row) for x in xs)]
        if len(sided) < len(rows) * 0.8:
            continue
        pieces = [cell for row in rows for cell in _row_cells(row)]
        if sum(1 for cell in pieces if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH) < len(pieces) * _SHORT_SHARE:
            continue
        left = min(w.x0 for w in inside) - 2
        right = max(w.x1 for w in inside) + 2
        area = pymupdf.Rect(left, top, right, bottom)
        if any(_mostly_inside(area, [table.rect]) or _mostly_inside(table.rect, [area]) for table in known):
            continue
        tops = [min(w.y0 for w in row) for row in rows]
        edges = [top, *((max(w.y1 for w in upper) + lower_top) / 2 for upper, lower_top in zip(rows, tops[1:])), bottom]
        cols = [left, *xs, right]
        cells = tuple(pymupdf.Rect(x0, y0, x1, y1) for y0, y1 in zip(edges, edges[1:]) for x0, x1 in zip(cols, cols[1:]))
        tables.append(RuledTable(area, cells))
    return tuple(tables)


def banded_tables(
    horizontal: Sequence[pymupdf.Rect], words: Sequence[_TextWord], known: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """三线表：顶线、表头下的中线、底线一样长，中间没有竖线。表头常折行、居中，
    几组数据之间还隔着横线，行距不齐，按对齐找只找得到其中一段。
    三条以上首尾对齐的横线，线间的字多是短格，就按线分段、段里按字分行分格。
    已经找到的表落在里面的，换成这一张；这一片落在已有的表里的，不另算。"""
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
                word for word in words
                if upper.y1 < (word.y0 + word.y1) / 2 < lower.y0
                and word.x0 >= left - _NEAR and word.x1 <= right + _NEAR
            ]
            bands.append(_word_rows(inside))
        rows = [row for band in bands for row in band]
        if len(rows) < _MIN_TEXT_ROWS or any(not band for band in bands):
            continue
        # 线间排着图注的是图（文法规则、例句框），图注要照常翻
        if any(_LABEL_RE.match(" ".join(word.text for word in row)) for row in rows):
            continue
        # 有三条线撑着，短格占得少一些也算：表头的列名、O(k·n·d) 这样的式子比数据格长。
        # 正文、伪代码一行连成一格，跨过半张表
        pieces = [cell for row in rows for cell in _row_cells(row)]
        short = sum(1 for cell in pieces if cell.words <= _SHORT_WORDS and cell.width <= _SHORT_WIDTH)
        # 横线下第一行只有一个短格的是分组名（Conditional、μ prediction），不算没分格的行
        split = sum(
            1 for band in bands for index, row in enumerate(band)
            if len(cells := _row_cells(row)) >= 2
            or (index == 0 and cells[0].words <= _SHORT_WORDS and cells[0].width <= _SHORT_WIDTH)
        )
        # 跨过半张表的格子只在一行一格时才说明是正文；两列表的行名本来就占大半
        if (short < len(pieces) * _BAND_SHORT_SHARE or split < len(rows) * 0.8
                or any(len(cells := _row_cells(row)) == 1 and cells[0].width > (right - left) / 2 for row in rows)):
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
    """同一横坐标上首尾相接的竖线段接成一条。逐行画的表格线是一格一段。"""
    out: list[pymupdf.Rect] = []
    for seg in sorted(vertical, key=lambda r: (round(r.x0), r.y0)):
        last = out[-1] if out else None
        if last is not None and abs(last.x0 - seg.x0) < 1 and seg.y0 - last.y1 <= _SNAP:
            out[-1] = pymupdf.Rect(last.x0, last.y0, last.x1, max(last.y1, seg.y1))
        else:
            out.append(pymupdf.Rect(seg))
    return out


def _with_text_rows(table: RuledTable, words: Sequence[_TextWord]) -> RuledTable:
    """只画了列线、行与行之间没有横线的表，一格里叠着好几行。同一排的几格里，字一行行
    左右对齐、每行跨两格以上、行名每行都有、多是短格的，是没画线的行，按行间空档切开。
    折了行的表头常有只占一行的格，句子折行只在一格里，长句也不算短格。"""
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
    # 最左边有字的一列是行名
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
        (max(word.y1 for word in upper) + min(word.y0 for word in lower)) / 2
        for upper, lower in zip(rows, rows[1:])
    ]
    out: list[pymupdf.Rect] = []
    for cell in band:
        edges = [cell.y0, *(y for y in cuts if cell.y0 < y < cell.y1), cell.y1]
        out.extend(pymupdf.Rect(cell.x0, top, cell.x1, bottom) for top, bottom in zip(edges, edges[1:]))
    return out


def _with_ruled_heads(
    tables: Sequence[RuledTable], words: Sequence[_TextWord], horizontal: Sequence[pymupdf.Rect],
) -> tuple[RuledTable, ...]:
    """三线表的表头夹在顶线和中线之间。表头的字常比数据列宽、居中排，左右缘和表身对不上，
    按对齐找出来的表只有表身。表身上方紧贴一条横贯全表的中线，再往上有一条一样长的顶线，
    两线之间的字就是表头，按列间空档分格并进表里。"""
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
            h for h in spans if abs(h.x0 - mid.x0) <= _NEAR and abs(h.x1 - mid.x1) <= _NEAR
            and row_height * 0.5 < mid.y0 - h.y0 <= row_height * _HEAD_ROWS
        ]
        if not tops:
            out.append(table)
            continue
        top = max(tops, key=lambda h: h.y0)
        band = [
            word for word in words
            if top.y1 < (word.y0 + word.y1) / 2 < mid.y0
            and word.x0 >= mid.x0 - _NEAR and word.x1 <= mid.x1 + _NEAR
        ]
        heads = _head_cells([_row_cells(row) for row in _word_rows(band)], mid.x0, mid.x1, top.y1, mid.y0)
        if len(heads) < 2:
            out.append(table)
            continue
        grown = pymupdf.Rect(min(rect.x0, mid.x0), top.y0, max(rect.x1, mid.x1), rect.y1)
        out.append(RuledTable(grown, (*heads, *table.cells), table.algorithm))
    return tuple(out)


def _head_cells(
    rows: Sequence[Sequence[_TextCell]], left: float, right: float, top: float, bottom: float,
) -> list[pymupdf.Rect]:
    """表头的字按列分格。一行字接在正上方最近的那格下面，那格又只和这一个字相交的，
    是同一格折了行；跨在两列上面的组名单独一格。格子左右伸到和邻格的空档中线，
    上下伸到所在的那段表头。"""
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
    """按词距连成一片的是一句话（段落末行 Levenshtein in Eq. 2.20.），表头各列之间隔着空档。"""
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
    page: pymupdf.Page, drawings: list[dict], words: Sequence[_TextWord], ruled: Sequence[RuledTable] = (),
) -> tuple[RuledTable, ...]:
    """每格一块底色、格子挨着格子铺满一片的表（LaTeX 的 \\rowcolor、\\cellcolor）。
    这种表常常只有上下几条横线，没有竖线，按线找不出格子，各格的字会被并成一段去翻译。

    底色块要拼成至少两行两列、铺满这一片，有两格以上有字，片里的字都落在某一格里。
    一整块底色是文本框，一行一块的是代码框的底纹，都不算；图里铺的白底格子
    压着跨好几格的图注，也不算。已经按线找到的表不再算一遍。
    """
    bounds = text_page_rect(page)
    fills = [
        pymupdf.Rect(drawing["rect"]) for drawing in drawings
        if light_fill(drawing) and min(drawing["fill"]) < 0.97 and not (
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
        # 格子互不交叠、合起来铺满整片：嵌套的底色（框里再套一块）面积会多出来
        if abs(sum(cell.get_area() for cell in cells) - area.get_area()) > area.get_area() * 0.05:
            continue
        boxes = [pymupdf.Rect(w.x0, w.y0, w.x1, w.y1) for w in words]
        boxes = [box for box in boxes if _mostly_inside(box, [area])]
        holders = [[cell for cell in cells if _holds(cell, box)] for box in boxes]
        if not all(holders):
            continue
        if len({id(found[0]) for found in holders}) >= 2:
            tables.append(RuledTable(area, tuple(sorted(cells, key=lambda c: (c.y0, c.x0)))))
    return tuple(tables)


def _spaced_rows(cells: Sequence[pymupdf.Rect], words: Sequence[_TextWord]) -> bool:
    """一列逐行底色的表（\\rowcolor）：三行以上，每行一行字，都隔着大空档分成几段
    （例句、填空的横线、答案）。代码框的底纹也是一行一块，但一行是连着的一句。"""
    if len(cells) < _MIN_TEXT_ROWS:
        return False
    for cell in cells:
        inside = [word for word in words if _holds(cell, pymupdf.Rect(word.x0, word.y0, word.x1, word.y1))]
        rows = _word_rows(inside)
        if len(rows) != 1 or len(_row_cells(rows[0])) < 2:
            return False
    return True


def _holds(cell: pymupdf.Rect, box: pymupdf.Rect) -> bool:
    """词整个落在格子里，左右只许出头一点。跨两格的词说明这片底色不是按格子铺的。"""
    middle = (box.y0 + box.y1) / 2
    return cell.x0 - _NEAR <= box.x0 and box.x1 <= cell.x1 + _NEAR and cell.y0 <= middle <= cell.y1


_SYMBOL_FONT_RE = re.compile(r"(?:^|\+)(?:CMSY|CMEX|MSAM|MSBM)", re.I)
_ALGORITHM_RE = re.compile(r"^\s*(?:Algorithm|Procedure|算法)\s*\d+", re.I)


def algorithm_tables(page: pymupdf.Page, drawings: list[dict] | None = None) -> tuple[RuledTable, ...]:
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


def unruled_tables(
    page: pymupdf.Page, ruled: Sequence[RuledTable] = (), words: Sequence[_TextWord] | None = None,
) -> tuple[RuledTable, ...]:
    """没有线时，只认短格子组成的网格：列的左缘或右缘对齐，行高重复。words 是本页的词，没给时现读。

    词间距并成一格。栏间大空档上的长行是正文，不是表。
    """
    zones = [table.rect for table in ruled]
    words = [
        word for word in (page_words(page) if words is None else words)
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
    return tuple(tables)


def interlinear_glosses(
    page: pymupdf.Page, words: Sequence[_TextWord] | None = None,
) -> tuple[pymupdf.Rect, ...]:
    """语言学例句的逐词对译：原文一行、对译一行。逐词翻译对译行没有意义，
    词序一变也对不上，两行都留原文。认两种排法：

    - 逐词对齐：下一行每个词都和上一行的某个词左缘对齐。正文相邻两行碰巧对齐的词
      只有零星几个，要求两行各有七成五以上的词对得上。只有两三个词时，还要上一行
      是汉字、假名这类非拉丁文字，下一行是拉丁字母。
    - 逐语素对译：两行从同一处起笔，都是连字符串起来的语素，对译行里有 1SG、PFV
      这样的语法标注缩写（莱比锡标注规则）。
    - 逐字注音：上一行是汉字，下一行从同一处起笔，全是拉丁字母，音节数和汉字数一样。
      汉字之间不空格，没法一个词一个词地对齐。

    表格一连好几行都这样对齐，对译只有孤零零的两行。words 是本页的词，没给时现读。
    """
    words = list(page_words(page) if words is None else words)
    rows = _word_rows(words)
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


def _glossed(upper: Sequence[_TextWord], lower: Sequence[_TextWord]) -> bool:
    height = max(word.y1 - word.y0 for word in upper)
    if lower[0].y0 - upper[0].y0 > height * 1.6:
        return False
    starts = [word.x0 for word in upper]
    # 对译行是词，不是数字表的一行
    aligned = [
        word for word in lower
        if any(abs(word.x0 - x) <= _GLOSS_ALIGN for x in starts) and any(ch.isalpha() for ch in word.text)
    ]
    hit = [x for x in starts if any(abs(word.x0 - x) <= _GLOSS_ALIGN for word in lower)]
    if len(aligned) >= len(lower) * _GLOSS_SHARE and len(hit) >= len(upper) * _GLOSS_SHARE:
        if len(aligned) >= _GLOSS_MIN:
            return True
        if len(aligned) >= 2 and _script_share(upper, latin=False) >= 0.5 and _script_share(lower, latin=True) == 1:
            return True
    return _morpheme_gloss(upper, lower) or _romanized(upper, lower)


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
    """这一行带字母的词里，拉丁字母词（latin=True）或非拉丁文字词所占的比例。
    PDF 里拼音的声调常抽成单独的修饰字母（wˇo），这一段字符也算拉丁字母。"""
    lettered = [word for word in row if any(ch.isalpha() for ch in word.text)]
    if not lettered:
        return 0.0
    matched = [
        word for word in lettered
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

    只看页顶和页底：固定的一条窄带，或者版心以外。同一侧、同一文本出现在两页以上
    才算页眉页脚，文本里的数字不算（页眉常带着页码和章号）。书的页眉随小节变，
    只出现一次的那条和别页的页眉落在同一高度，也算。
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
        skips.append(key in repeated or same_row or _is_page_number(block.text))
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
    frame = geo.frame
    if block.rect.y1 <= band or (frame is not None and frame.y0 is not None and block.rect.y1 <= frame.y0 - 1):
        return "header"
    floor = content_floor(geo.height)
    if block.rect.y0 >= floor or (frame is not None and frame.y1 is not None and block.rect.y0 >= frame.y1 + 1):
        return "footer"
    return None


def _norm_margin(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text).strip().casefold())


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


def _figure_rects(
    page: pymupdf.Page, drawings: list[dict] | None = None, images: list[dict] | None = None,
) -> list[pymupdf.Rect]:
    bounds = text_page_rect(page)
    width, height = bounds.width, bounds.height
    found: list[pymupdf.Rect] = []
    for info in page.get_image_info() if images is None else images:
        rect = pymupdf.Rect(info.get("bbox"))
        if rect.width > _FIGURE_MIN and rect.height > _FIGURE_MIN:
            found.append(rect)
    if drawings is None:
        try:
            drawings = page.get_drawings()
        except Exception:  # noqa: BLE001 - 个别页面的矢量表读不出来，当没有图
            drawings = []
    grounds: list[pymupdf.Rect] = []
    for drawing in drawings:
        rect = pymupdf.Rect(drawing.get("rect"))
        if rect.width < _DRAW_MIN or rect.height < _DRAW_MIN:
            continue
        if rect.width * rect.height < _DRAW_AREA:
            continue
        if rect.width > width * _PAGE_FILL and rect.height > height * _PAGE_FILL:
            continue
        if _is_blank_fill(drawing):
            grounds.append(rect)
            continue
        found.append(rect)
    return [*found, *_figure_grounds(grounds, found)]


def _figure_grounds(grounds: Sequence[pymupdf.Rect], figures: Sequence[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """画图程序（matplotlib）先铺一块白底，标题、坐标轴名、图例都写在白底上，曲线只占中间一块。
    白底里装着曲线、图片，又不比它们大出一倍的，是图。铺在正文和图下面的白底比图大得多。
    标题、坐标轴名紧挨着曲线，白底只算到曲线外面一圈：有的白底往下一直铺到正文里。"""
    kept: list[pymupdf.Rect] = []
    for ground in grounds:
        inside = [figure for figure in figures if _mostly_inside(figure, [ground])]
        if not inside:
            continue
        drawn = pymupdf.Rect(inside[0])
        for figure in inside[1:]:
            drawn |= figure
        if ground.width * ground.height <= drawn.width * drawn.height * _GROUND_SHARE:
            kept.append(ground & (drawn + (-_GROUND_REACH, -_GROUND_REACH, _GROUND_REACH, _GROUND_REACH)))
    return kept
