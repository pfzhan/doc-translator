"""Lay out translated text in a block's box: line breaking, inline formula images, and
shrinking the line height or font size when the text does not fit."""

import html
import re
from dataclasses import dataclass

import pymupdf

from .pdf_align import FIRST_BASELINE, SQUEEZE, draw_cjk_lines, page_origin
from .pdf_blocks import _PART_GAP, _formula_parts, _FormulaImage, _GlyphInk, _outside_formula
from .pdf_flow import (
    balance_marks,
    measure_width,
    peel_leading_space,
    restore_emphasis,
    separate_after_formula,
    space_around_latin,
    strip_style_marks,
)
from .pdf_layout import (
    text_page_rect,
)
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

# 行内公式最高几个字高。根号、求和带上下限的式子在这以内按原大写，行距跟着撑开
_INLINE_FORMULA_EM = 2.6
# 盒子放不下撑开的行距时，公式压到的高度：正好在一行里
_LINE_FORMULA_EM = 1.35
# 分式、根号下求和，基线以上一般不到一个半字高，以下不到一个字高
_FORMULA_ASCENT_EM = 1.8
_FORMULA_DESCENT_EM = 1.3


def _part_lead(part: dict, scale: float) -> float:
    """这一片和前一片之间的空，按公式的缩放算。"""
    return max(_PART_GAP, float(part.get("lead") or 0) * scale)


def _formula_png(formula: dict) -> bytes | None:
    """公式的位图：测试里直接给 PNG，正式流程里是 _FormulaImage。"""
    png = formula.get("png")
    return png.png() if isinstance(png, _FormulaImage) else png


def _formula_png_rect(formula: dict, target: pymupdf.Rect) -> pymupdf.Rect:
    """位图贴到哪里。裁图比公式框宽出的部分（见 _FormulaImage.area）按公式框到 target 的缩放放到两侧。"""
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
    """页面坐标的矩形换成 PDF 坐标（y 向上）。裁剪框不从原点起时要加上偏移。"""
    left, top = page_origin(page)
    return pymupdf.Rect(rect.x0 + left, top - rect.y1, rect.x1 + left, top - rect.y0)


class _GlyphKeeper(pymupdf.mupdf.PdfSanitizeFilterOptions2):
    """只删字形：中心落在 keep 外的删掉，留下的记下位置，过滤后拿来核对。
    路径和图片不删，裁剪路径挡得住。按位置删路径会把 Type3 字形过程里的笔画也删掉
    （𝟙 这种点阵字就没了）。每个字形回调一次，写得越简单越快。"""

    def __init__(self, keep: list[tuple[float, float, float, float]]):
        super().__init__()
        self.keep = keep
        self.kept: list[tuple[float, float]] = []
        self.use_virtual_text_filter()
        # 有图片时过滤器还要调这个回调，不实现会报错
        self.use_virtual_image_filter()

    def image_filter(self, ctx, ctm, name, image, scissor):
        # 返回的图片用完会被释放一次，原样留下要先加一次引用
        return pymupdf.mupdf.ll_fz_keep_image(image)

    def text_filter(self, ctx, ucs, length, trm, ctm, bbox, *rest):
        # 字形中心按这个字自己的框算，整套字体的框比括号宽，中心会落到括号外面。返回 1 是删掉
        gx, gy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
        x, y = gx * trm.a + gy * trm.c + trm.e, gx * trm.b + gy * trm.d + trm.f
        x, y = x * ctm.a + y * ctm.c + ctm.e, x * ctm.b + y * ctm.d + ctm.f
        # 点阵字常不报自己的框（宽为 0），只有整套字体的上下界，算出的中心离墨迹半行。
        # 这时横向看起笔点，纵向只要上下界和保留框交叠就留下：笔画有裁剪路径挡着
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


class _SanitizeFactory(pymupdf.mupdf.PdfFilterFactory2):
    # 属性不能叫 options：那是底层结构体的字段名，赋值会写进 C 结构体
    def __init__(self, sanitize: pymupdf.mupdf.PdfSanitizeFilterOptions2):
        super().__init__()
        self.sanitize = sanitize
        self.use_virtual_filter()

    def filter(self, ctx, doc, chain, struct_parents, transform, options):
        return pymupdf.mupdf.ll_pdf_new_sanitize_filter(
            doc,
            chain,
            struct_parents,
            transform,
            options,
            self.sanitize.internal(),
        )


def _filter_form(
    doc: "pymupdf.Document", form: int, keep: list[tuple[float, float, float, float]], recurse: bool = True
) -> tuple[int, list[tuple[float, float]]]:
    """过滤一遍 Form，返回新 Form 的 xref 和留下的字的位置。recurse 为假时只过滤这一层，套着的 Form 原样引用。"""
    mupdf = pymupdf.mupdf
    keeper = _GlyphKeeper(keep)
    factory = _SanitizeFactory(keeper)
    options = mupdf.PdfFilterOptions()
    # 页里套的 Form（图）也要过滤。过滤结果另存新对象，原对象页面自己还在用，直接改会把图删掉。
    # 不进套着的 Form 时也不复制：复制时 MuPDF 会改写内容流，图就和原来的字节不同了（见 _lines_form）
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
    """过滤后的 Form 里每个字还在原来的位置上。recurse 和过滤时一样。"""
    if not xref:
        return False
    _same, now = _filter_form(doc, xref, _EVERYWHERE, recurse)
    return _same_places(kept, now)


def _same_places(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """两组字形位置一一配得上，各差不到 0.1。不能排序后逐个比：同一列上下几行行首的字，
    横坐标差万分之几，排出来的先后就乱了，拿不同行的字来比。这里按 0.1 的格子分好，
    每个字在自己和相邻的格子里找。"""
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
    """复制 Form，只留 keep 里的字形，返回新 Form 的 xref。keep 是 PDF 坐标（y 向上）。

    公式透传引用的是整页原文，裁剪路径只挡得住笔画，挡不住文字层：浏览器里拖选、
    复制时，每个公式底下都叠着一整页隐形原文，一拖就选中一大片。
    MuPDF 删掉同一行前面的字时，中途换过字号的行（小型大写的 THEME）会把后面的字
    算偏几个点，落到裁剪框外。所以过滤后核对位置：偏了就整行留下（删别的行不影响
    本行），还偏就退回原 Form。
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
    """过滤一遍 Form 这一层，只留 bands（PDF 坐标的纵向区间）里的字，返回新 Form 的 xref。位置核对不过时用原 Form。

    套着的 Form（图）不进去，原样引用，留给每个公式最后那一遍（_keep_marks）。MuPDF 每过滤一遍都会去掉
    一层空的 q Q，图里的字在哪一遍删掉，删完的字节也不一样。各公式的图要是过滤的遍数不同，
    内容相同而字节不同，合并重复对象时合不掉，一张大图会在文件里存好几份。"""
    try:
        xref, kept = _filter_form(doc, form, [(-1e9, y0, 1e9, y1) for y0, y1 in bands], recurse=False)
        if _keeps_places(doc, xref, kept, recurse=False):
            return xref
    except Exception:  # noqa: BLE001 - 和 _keep_marks 一样，过滤出错时用原 Form
        pass
    return form


@dataclass
class _LinesNode:
    bands: list[tuple[float, float]]
    xref: int
    children: list["_LinesNode"]


class _FormulaSources:
    """本页公式透传时引用的源 Form。

    每个公式透传前都要过滤一遍源 Form（见 _keep_marks），每个字形回调一次。都在整页上过滤，
    一页几十个公式就要把整页的字走几十遍。这里把本页的公式按上下位置对半分，每一半只留自己所在的行，
    再往下分，到每份不超过四个公式为止：每层合起来只走一遍整页，每个公式最后只在自己那几行的字上过滤。
    留下的行是公式的框上下各放 2pt，盖住 _formula_form 的裁剪框（上下最多扩 1.5pt），
    按框、按行过滤时要留的字都在里面。取的时候按位置找最小的、整个盖住这个公式的一份，
    没有登记的公式也能找到，最差是整页。"""

    def __init__(self, doc: "pymupdf.Document", page: "pymupdf.Page", form: int, formulas):
        # 只记换算 PDF 坐标要的页面上沿（见 _pdf_rect），不留页面对象
        self.top = page_origin(page)[1]
        self.glyphs = _GlyphInk(page)
        group: list[tuple[tuple[float, float], dict]] = []
        for formula in formulas:
            for part in _formula_parts(formula):
                source_page = part.get("page")
                band = self._band(part)
                if band is not None and (source_page is None or int(source_page) == page.number):
                    group.append((band, part))

        def build(source: int, group: list[tuple[tuple[float, float], dict]]) -> _LinesNode:
            bands = _merge_bands([band for band, _part in group])
            node = _LinesNode(bands, _lines_form(doc, source, bands), [])
            if len(group) > 4:
                group = sorted(group, key=lambda item: item[0])
                half = len(group) // 2
                node.children = [build(node.xref, sub) for sub in (group[:half], group[half:])]
            return node

        self.root = _LinesNode([], form, [build(form, group)] if group else [])

    def _band(self, formula: dict) -> tuple[float, float] | None:
        rects = [s["bbox"] for s in formula.get("spans") or []] + list(formula.get("curves") or [])
        if formula.get("bbox") is not None:
            rects.append(formula["bbox"])
        if not rects:
            return None
        return self.top - max(rect[3] for rect in rects) - 2, self.top - min(rect[1] for rect in rects) + 2

    def source(self, formula: dict) -> int:
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


def _hole_ink(span: dict) -> pymupdf.Rect:
    """外来 span 在公式裁剪里挖的孔：墨迹框，夹在字体框里。精确框比字体框还大的
    （LMMono 的精确框往下多出半行）不是墨迹，按字体框挖，不然会削掉下一行的字头。单画量出的照旧。"""
    box = pymupdf.Rect(span["bbox"])
    if span.get("ink") is None:
        return box
    ink = pymupdf.Rect(span["ink"])
    return ink if span.get("ink_drawn") else ink & box


def _formula_form(
    doc: "pymupdf.Document",
    page: "pymupdf.Page",
    fmS: int,
    f: dict,
    f_name: str,
    page_spans: list[dict] | None = None,
    glyphs: _GlyphInk | None = None,
) -> int:
    """给单个公式造紧致 BBox 的包裹 Form（按 span 逐个裁剪后引用源页 Form），返回 xref。

    裁剪路径是 run 里每个 span 的矩形并集，不是整个外框矩形：外框矩形会把间隙里
    的邻字笔画（上一行 p 的降部等）也裁进来，在公式上方留下横杠。
    BBox 取并集外框，选择/复制时不会框出一大片。

    裁剪用 even-odd：run 的 span 矩形按侧扩展后取并集（斜体笔画会越出 span 的
    排版 bbox——Y 的右臂能越出 1.2pt，扩展以不碰到邻居为限），再减去外来 span
    的矩形孔（上一行降部等），既不削自己的字，也不沾别人的。
    裁剪挡不住文字层，引用的源页 Form 先去掉裁剪区外的字形，只留本公式的。
    """
    bbox = f["bbox"]
    clip_outer = _pdf_rect(page, bbox)
    # run 的 span 来自 extract_blocks 的另一次 get_text 调用，和 page_spans 是不同
    # 对象，只能按 bbox 相等判断是不是 run 成员，否则兄弟 span（本公式的上标）会被
    # 当成外来 span 被挖洞挖掉
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

    # 先并成不相交集合（ŷ 的符头 bbox 和 y 几乎完全重叠，不并掉 even-odd 会把
    # y 的躯干整个关掉），再按侧扩展：每侧扩到 1.5pt 为止、以不贴到邻居 bbox 为限
    base = merge_rects([pymupdf.Rect(s["bbox"]) for s in f["spans"]] + curve_rects)
    union = pymupdf.Rect()
    for rr in base:
        union |= rr
    lim = {"t": 1.5, "b": 1.5}  # 上下扩展上限（pt），左右也是 1.5
    # 左右邻居的边。每个矩形按自己到邻居的距离扩：按整个公式的外框算，紧挨着一个逗号时
    # 里面每个矩形都只扩 0.2，𝒩 伸进下一个字前空档里的笔画就被切断
    left_edges: list[float] = []
    right_edges: list[float] = []
    if page_spans:
        for fb in page_spans:
            if not fb["text"].strip():  # 空白 span 没有墨迹，不限扩展也不挖洞
                continue
            # 先用坐标排除和公式不在同一高度的 span：一页上千个，每个都建 Rect 很慢
            if not (fb["bbox"][1] < union.y1 and fb["bbox"][3] > union.y0):
                continue
            fr = pymupdf.Rect(fb["bbox"])
            if is_run_member(fr) or not (fr.y0 < union.y1 and fr.y1 > union.y0):
                continue
            if fr.x1 <= union.x0 + 0.01:  # 左侧邻居
                left_edges.append(fr.x1)
            elif fr.x0 >= union.x1 - 0.01:  # 右侧邻居
                right_edges.append(fr.x0)
            if fr.y1 <= union.y0 + 0.01:  # 上方邻居（pymupdf y 向下）
                lim["t"] = max(0.2, min(lim["t"], union.y0 - fr.y1 - 0.3))
            elif fr.y0 >= union.y1 - 0.01:  # 下方邻居
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

        def touches(box) -> bool:
            return box[0] <= reach_x1 and box[2] >= reach_x0 and box[1] <= reach_y1 and box[3] >= reach_y0

        for fb in page_spans:
            if not fb["text"].strip():  # 空白 span 没有墨迹，限扩展和挖洞都不参与
                continue
            # 孔落在 span 的字体框或墨迹框里，两个都碰不到裁剪框的 span 挖不出孔
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
            # 挖不挖按字体框判断，孔按墨迹挖：括号、箭头的字体框比字形高出一截，会伸到下一行的字上
            ink = _hole_ink(fb)
            # 孔 = 外来 span 与我们的每个矩形的交。必须完全落在我们的矩形内：
            # even-odd 裁剪下越出并集的孔会反开，把正要排除的外来笔画整个画出来。
            # 只挖纵向不共线的 span（降部、邻公式的角标等）：同行邻字的 bbox 常常
            # 仅仅贴边且纵向几乎等长，挖掉会把本公式的笔画（y 的左干）削了。
            # 孔不内收：降部墨迹和 span bbox 底边之间本就有缝隙，内收会留下一道残影
            for rr in our_rects:
                y_ov = min(fr.y1, rr.y1) - max(fr.y0, rr.y0)
                if y_ov > 0.7 * (rr.y1 - rr.y0):
                    continue
                hb = ink & rr
                if not hb.is_empty and hb.width >= 0.3 and hb.height >= 0.3:
                    hole_rects.append(hb)
        # 相交的孔合并成一个：两个孔相交处在 even-odd 下会反开
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
    # 留哪些字按公式自己的 span 框认，不按往外扩过的裁剪框：MuPDF 只给整套字体的上下界，
    # 下一行的字算出的中心在基线上 0.25em，正好落进裁剪框下沿扩出的那一条，成了公式图里的隐形字
    source = _keep_marks(doc, fmS, [_pdf_rect(page, r) for r in base]) if our_rects else fmS
    # BBox 也会裁剪。斜体 f 的弯钩和尾巴伸出字宽，按公式框会被切掉：左右放到字形量出来的墨迹，
    # 不超过裁剪路径。只放到自己的墨迹为止，旁边的字（If 的 f）伸进空档的笔画不带进来
    inks = [glyphs.x_bounds(span) for span in f["spans"]] if glyphs is not None and our_rects else []
    inks = [ink for ink in inks if ink is not None]
    if inks:
        left = max(min(ink[0] for ink in inks), min(r.x0 for r in our_rects))
        right = min(max(ink[1] for ink in inks), max(r.x1 for r in our_rects))
        reach = _pdf_rect(page, pymupdf.Rect(left, bbox.y0, right, bbox.y1))
        clip_outer = pymupdf.Rect(
            min(clip_outer.x0, reach.x0), clip_outer.y0, max(clip_outer.x1, reach.x1), clip_outer.y1
        )
    fmT = doc.get_new_xref()
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
    # redact 后 /Resources 会变成直接字典（键要带 Resources/ 前缀）；
    # 间接引用则在资源对象上直接设键
    m = re.match(r"(\d+) \d+ R", res_v)
    res_xref, base = (int(m.group(1)), "") if m else (page.xref, "Resources/")
    xo_v = doc.xref_get_key(res_xref, f"{base}XObject")[1]
    if xo_v == "null":
        doc.xref_set_key(res_xref, f"{base}XObject", "<<>>")
        xo_v = "<<"
    m = re.match(r"(\d+) \d+ R", xo_v)
    if m:
        doc.xref_set_key(int(m.group(1)), f_name, f"{fmT} 0 R")
    else:
        doc.xref_set_key(res_xref, f"{base}XObject/{f_name}", f"{fmT} 0 R")
    return fmT


def _paint_formula(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    formula: dict,
    target: "pymupdf.Rect",
    forms: "_FormulaSources | None",
    page_spans: list[dict] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]] | None = None,
) -> None:
    """把公式画进 target。矢量透传失败时回退位图。绘制命令先攒着，最后一条内容流再写。

    公式若来自另一页，本页的 Form 里没有它的笔画，只能贴已经裁好的图。
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
            # 资源名必须全页唯一：块内序号相同的 FmF0 会互相覆盖
            f_name = "FmF_" + re.sub(r"[^A-Za-z0-9_]", "_", str(formula["name"]))
            _formula_form(doc, page, forms.source(formula), formula, f_name, page_spans, forms.glyphs)
            sx = target.width / bbox.width
            sy = target.height / bbox.height
            # 带缩放的仿射：平移量必须吸收缩放，否则公式整体往下漂 (1-sy)*H
            source = _pdf_rect(page, bbox)
            placed = _pdf_rect(page, target)
            tx = placed.x0 - sx * source.x0
            ty = placed.y0 - sy * source.y0
            draws.append(f"q {sx:.4f} 0 0 {sy:.4f} {tx:.2f} {ty:.2f} cm /{f_name} Do Q")
            return
        except Exception:  # noqa: BLE001 - 矢量透传失败回退位图
            pass
    png = _formula_png(formula)
    if not png:
        return
    target = _formula_png_rect(formula, target)
    if deferred_png is not None:
        deferred_png.append((target, png))
        return
    page.insert_image(target, stream=png, keep_proportion=False, overlay=True)


def _formula_metrics(
    formulas: list[dict],
    block_size: float,
    em: float,
    reference: float | None = None,
) -> dict[int, FormulaMetric]:
    """按当前字号给出每个公式槽。rs 写回记录，矢量缩放和槽位用同一个数。reference 见 _formula_slot。"""
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
    """放得下就按原字号排。放不下先在本栏右扩，再往下伸进空白，再压一点字距，再压行距，最后才缩字号。
    holes 是原文绕开的图，碰到它的行让到一边，另外返回这几行的缩进和宽度。
    lower 是（想要的起笔，下面一块的上沿）：盒子写不满时上沿往下挪，最多挪到想要的起笔。
    框的下沿伸过原文末行，挪下去的字不能碰到下面一块。
    room 是下面一块的上沿（见 _room_below）。下沿往下伸时，末行和下面一块之间留出半行，和分段的空隙一样。
    left_limit 是靠右排的块往左扩能到哪里，给了就往左扩，不往右扩。
    lent 为真时 room 是下面一段借出地方后最低的起笔（见 spare_below），它要挪到末行下面半行，
    末行下面那截空白就不能伸过框底（见 _glyph_bottom）。
    hang 是悬挂缩进的条目里原文正文比题号缩进多少点（见 _hang_indent）。"""
    gap = 1.45 if cjk else 1.2
    reference = block_size * 0.88 if cjk else block_size
    box = rect
    # 伸到这里，末行行框下沿离 room 还有半行。盒子高度比末行行框多出首行上方 0.3em 的出头
    reach = None if room is None else room - em * (gap * 0.5 - 0.3)
    expanded = False
    grown = False
    squeezed = False
    tightened = False
    capped = False
    # 按多宽断行。压字距时比盒子宽一点，写的时候各行压回盒子宽（见 SQUEEZE）
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
        # 末行下面那截空白伸过框底只为保住正常行距。压了行距还放不下，照旧按框底量，该缩字号就缩。
        # 只有一行时没有行距要保，压了行距字形不动，末行行框下沿还高一些，下面一段能往上提
        relax = cjk and len(lines) > 1 and room is not None and not tightened and not lent
        bottom = _glyph_bottom(box.y1, room, em, gap) if relax else box.y1
        too_tall = box.y0 + height > bottom + 1
        if not overflows and not too_tall:
            # 绕图的行按图的位置算好了，盒子不能再挪；悬挂缩进跟着盒子走
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
        # 末行只剩几个字时，字距压一点就能收回这一行，比压行距看不出来
        if too_tall and not overflows and not squeezed and cjk and not lanes:
            squeezed = True
            if len(typeset_lines(text, em, max(box.width, 1.0) * (1 + SQUEEZE), metrics)) < len(lines):
                squeeze = 1 + SQUEEZE
                continue
        if not tightened:
            gap = 1.1 if gap <= 1.25 else gap - 0.25
            tightened = True
            continue
        # 压了行距，高公式撑开的那几行还是放不下，就把公式压回一行高，字号不动
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
    """排好的几行要多高的盒子。首行字形顶部还有 0.3em 的出头（实测 MuPDF 盒子内容高 = 行数×行距 + ~0.25em），
    只算行距会让临界盒子在写入时被静默缩到 0.93 倍"""
    return max(len(lines), 1) * em * gap + em * 0.3 + line_offsets(lines, em, gap)[1]


def _glyph_bottom(bottom: float, room: float, em: float, gap: float) -> float:
    """逐字画的中文段，盒子（按 _box_height 量）能写到哪里。末行字形下面那截行距是空白，
    可以伸过框底；离 room 还要空出半截行距，字不贴到下面一块上。框底本来就伸过 room 的（目录条目一条挨一条），
    仍写到框底。跨页段落拆行时同样让这截空白伸过 room（见 _filled_bottom）"""
    return max(bottom, min(bottom + (gap - 1) * em, room + (gap - 1) * em * 0.5))


def _typeset_end(lines: list[TypesetLine], em: float, gap: float, box: "pymupdf.Rect") -> float:
    """排好的几行末行行框的下沿"""
    return box.y0 + max(len(lines), 1) * em * gap + line_offsets(lines, em, gap)[1]


# 居中的题名折行时可以断开的小句标点
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
    """居中排的题名、副标题折成几行时，在小句标点后面断开，不留一两个字单独一行。
    行数不能比按宽度断的多；几处标点都行的，取各行最宽的那行最窄的断法。"""
    metrics = _formula_metrics(formulas, block_size, em, reference)
    best, best_width = lines, None
    for index, char in enumerate(text[:-1]):
        if char not in _CLAUSE_MARKS:
            continue
        head, tail = text[: index + 1].rstrip(), text[index + 1 :].strip()
        if not head or not tail:
            continue
        # 整段粗体的作者名单在逗号处断开，后半截要重开粗体
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
    """开头几行相对盒子左缘的缩进和宽度。字形落在图的高度里的行让到图的另一侧，
    和图隔半个字。图以下的行不列出，按整个盒子宽。"""
    if not holes:
        return []
    pitch = em * gap
    bottom = max(hole.y1 for hole in holes)
    lanes: list[tuple[float, float]] = []
    keep = 0
    top = box.y0
    while top < bottom:
        # 字形大约占基线上 0.88em、下 0.12em
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


# 译文开头的题号、圆点连同后面的空白
_HANG_LABEL_RE = re.compile(r"\s*(?:[•◦▪‣·○■□►»]|[-–—*]|\(?(?:\d+(?:\.\d+)*|[a-zA-Z]|[ivxlc]+)[.)]?)\s+")


def _label_width(text: str, em: float) -> float:
    """译文开头的题号、圆点连同后面的空白有多宽，续行从这里起笔。没有题号时返回 0"""
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
    """悬挂缩进的各行，绕图的行再让开图。正文对齐在盒子左缘往右 hang 处，和条目下面另起的说明段落同一列；
    label 是译文题号连同空白的宽度（见 _label_width），首行往右挪，题号后面的正文也从这一列起笔。
    题号比 hang 宽时这一列跟着往右。断行前不知道有几行，每行至少一个字，按字数列满"""
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
    """这一行已写入文字的 origin。纯公式行没有文字，用 htmlbox 的经验基线。
    相邻行的字框会交叠，离槽位更近的文字优先，避免公式掉到下一行仍留在原横坐标。"""
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


def _paint_formula_parts(
    page: "pymupdf.Page",
    doc: "pymupdf.Document",
    formula: dict,
    slot_x: float,
    baseline: float,
    forms: "_FormulaSources | None",
    page_spans: list[dict] | None,
    draws: list[str],
    deferred_png: list[tuple["pymupdf.Rect", bytes]],
) -> None:
    """一个占位符的几片按阅读顺序并排贴。换行切开的后半段不再留在源文横坐标。"""
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


def _typeset_line_html(line: TypesetLine, formulas: list[dict]) -> str:
    """一行的 HTML。没有图的公式退回记录里的文字。"""
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
    """中文行按字形两端对齐，公式画在内容流里它所在的位置（见 draw_cjk_lines）。缺字时返回 False。"""
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
    """排版器断行，再写入。中文按字形两端对齐；缺字时退回一个 HTML 盒子。
    lower、room、left_limit、lent、hang 见 _fit_typeset。返回末行行框的下沿，没写时返回 None。"""
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
        # 起笔挪下来了，首行不再按原文第一行往上提
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
        # 图在左边时，旁边几行各自缩进。行内的 margin、padding MuPDF 不认，每行写成一个块
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
        # 放没放下按末行字形量（见 _glyph_bottom），HTML 盒子要有整行距高，否则整段被缩小
        box = pymupdf.Rect(box.x0, box.y0, box.x1, max(box.y1, box.y0 + _box_height(lines, em, gap)))
    if not _try_insert(page, box, html_text, css, 0.9, archive):
        _insert_fitting(page, box, html_text, css, obstacles, archive, right_limit)
    return end


def _written_right(page: "pymupdf.Page", x0: float, y0: float, y1: float) -> float:
    """这一条刚写进去的文字的右缘。估算宽度不准时，下一槽跟着它走。
    只认基线落在本盒子里的字。上一行的字框会探进盒子上沿，拿它的右缘会把公式推到栏边。"""
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
    """按行写入文字，公式槽用记录的下沿贴到该行基线。lower、room、hang 和返回值见 _place_plain。"""
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
    # 公式记下前面一片文字写完时最后一条内容流，量完基线画在它后面，拖选时夹在前后的字中间
    pending: list[tuple[FormulaPiece, float, float, int | None]] = []
    offsets, _extra = line_offsets(lines, em, gap)
    for index, line in enumerate(lines):
        y = box.y0 + index * line_h + offsets[index]
        indent, width = lanes[index] if index < len(lanes) else (0.0, box.width)
        origin_x = box.x0 + indent - (line_lead(line) if align == "left" else 0.0)
        if align != "left" and line.width < width:
            origin_x += width - line.width if align == "right" else (width - line.width) / 2
        # 拉丁字母按 0.5em 估算，i-th 这种窄词会偏宽。公式若跟估算走，就会被推开。
        # 文字按实际写出的右缘接着排，两侧只留公式槽自己的 2pt。
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
                # 盒子按估算加宽，避免窄词被缩字；落位不用这个宽，用写出的右缘。
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
    """页面 Contents 里最后一条内容流的 xref。"""
    refs = re.findall(r"(\d+) 0 R", doc.xref_get_key(page.xref, "Contents")[1])
    return int(refs[-1]) if refs else None


def _insert_contents_after(page: "pymupdf.Page", doc: "pymupdf.Document", after: int | None, commands: str) -> None:
    """新建一条内容流，排在 Contents 里 after 这条后面；找不到 after 时排在最后。

    按 xref 找位置：insert_htmlbox 第一次写时会在最前面补一条 q，按下标记会错位。"""
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
    """行内公式槽位 (缩放, 宽, 高)。高超过 _INLINE_FORMULA_EM 时压下来，再高的那一行撑开行距（见 line_offsets）。
    盒子放不下时 _fit_typeset 在公式上记 cap，压到一行高。
    换行切开的几片按阅读顺序横排，宽度相加，不把两行空白裁进同一个框。
    reference 是公式保持原大时的译文字号。中文比原文的字号小一成（看着和原文一样大），
    公式按 render_size / block_size 缩就跟着小一成，夹在汉字里显得小；中文按 reference 比，没缩字号就是原大。"""
    parts = _formula_parts(formula)
    height = max(float(part.get("h") or 0) for part in parts)
    if not formula.get("has_img") or height <= 0 or block_size <= 0:
        text = str(formula.get("text") or "")
        return 1.0, len(text) * render_size * 0.5, render_size
    cap = float(formula.get("cap") or _INLINE_FORMULA_EM)
    # 基线上下伸出一大截的，多半把相邻一行的字也框了进来，基线也认错了。放大只会盖住别的行，仍压到一行高
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
    """排版收缩第一级：向右扩，不盖住右侧纵向交叠的文字或图。

    上限是页宽的 90%，有栏缘时再取更小的那个。结果不会小于原来的右缘。
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
    """靠右排的块向左扩，最多到 x0，不盖住左侧纵向交叠的文字或图。结果不会小于原来的左缘。"""
    for obstacle in obstacles:
        if obstacle.x1 >= rect.x1 - 1:
            continue
        if obstacle.y0 < rect.y1 and obstacle.y1 > rect.y0:
            x0 = max(x0, obstacle.x1 + 2)
    return pymupdf.Rect(min(x0, rect.x0), rect.y0, rect.x1, rect.y1)


def _try_insert(
    page: "pymupdf.Page", rect: "pymupdf.Rect", html_text: str, css: str, scale_low: float, archive=None
) -> bool:
    """insert_htmlbox 放不下返回负值（此时不画内容）。pymupdf 在缩放刚好等于 scale_low 时
    会因浮点误差触发 assert（0.8999999999999999 < 0.9），按没放下来处理。"""
    try:
        return page.insert_htmlbox(rect, html_text, css=css, scale_low=scale_low, archive=archive)[0] >= 0
    except AssertionError:
        return False


def _tighten_line_height(css: str) -> str:
    """压一级行距。CJK 样式是 1.45，写死替换 1.2 时这一步不会生效。"""
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
    """排版三级收缩（BabelDOC 思路）：先右扩 → 压行距 → 最后才缩字号。

    每一级只在放不下（insert_htmlbox 返回负值，此时不会画出内容）时进入下一级。
    右扩只对明显窄于页宽的块（标签、图注、短行）生效：满栏段落右扩会越过栏边界，
    且扩完不缩字号时 CJK 行框（1.31em）比盒子（按拉丁 1.16em 算）高，会向下溢出
    压到下一行内容。right_limit 是本栏右缘，扩出去也不能越过它。
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
    # 兜底：和原来一样缩到能放下为止
    page.insert_htmlbox(target, html_text, css=css_tight, scale_low=0, archive=archive)
