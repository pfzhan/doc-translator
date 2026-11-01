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
from .pdf_layout import (
    light_fill,
    text_page_rect,
    visible_drawings,
)
from .pdf_runs import (
    FormulaRun,
    expand_box,
    intersecting_curves,
)


def _page_curve_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """矢量线条，加上拉长成细线的一像素图片：老 TeX 的分式线、根号横线是这样画的。
    描边的线按画出来的范围算：路径框不含线宽，一条横线的框高为零。"""
    drawings = visible_drawings(page)
    rects: list[pymupdf.Rect] = []
    for drawing in drawings:
        raw = drawing.get("rect")
        # 浅色底不是笔画，字压在上面。并进公式框会把上下格的字一起裁进图里，写回时底色也跟着擦掉
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
    """glyphs 是页面上的 span（带墨迹），用来认出别人也贴着的横线。"""
    old = pymupdf.Rect(formula["bbox"])
    # 按字形墨迹找：数学字体的字体框伸到上下行，会碰到别的式子的分数线、根号横线
    ink = _spans_ink(formula.get("spans") or [])
    probe = old if ink is None or ink.is_empty else ink + (-_INK_PAD, -_INK_PAD, _INK_PAD, _INK_PAD)
    own = {_origin_key(span) for span in formula.get("spans") or [] if span.get("origin")}
    spans = formula.get("spans") or []
    curves = [
        curve for curve in intersecting_curves(probe, drawings)
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


# 比这细的图片是线，不是插图
_RULE_IMAGE_THICK = 1.5
# 墨迹外再留这一条，避免抗锯齿和斜体笔尖被切掉。左右用同一个数。
_FORMULA_SIDE_PAD = 0.45
_FORMULA_INK_DPI = 180


def _page_text_spans(page: pymupdf.Page) -> list[dict]:
    data = page.get_text("dict", flags=_TEXT_FLAGS)
    return [
        span
        for block in data["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    ]


def _page_glyphs(page: pymupdf.Page) -> list[dict]:
    """页面上的 span，带字形墨迹。"""
    data = page.get_text("dict", flags=_TEXT_FLAGS)
    _attach_ink(page, data)
    return [span for block in data["blocks"] if block.get("type") == 0 for line in block["lines"] for span in line["spans"]]


def _span_ink_boxes(span: dict, char_index: _CharIndex) -> list[pymupdf.Rect]:
    """非空白字符的框。空白不算墨迹；没有字符数据时整段非空文字算一块。"""
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
    """本公式墨迹的左右边界。邻居的字不算，斜体越出字框的笔尖算。"""
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
        return any(
            rect.x0 - pad <= px <= rect.x1 + pad and rect.y0 - pad <= py <= rect.y1 + pad
            for rect in rects
        )

    def is_ours(x: int, y: int) -> bool:
        px = clip.x0 + (x + 0.5) * step_x
        py = clip.y0 + (y + 0.5) * step_y
        # 字框内和紧贴字框的笔尖算自己的。再远的墨迹是邻字伸进来的，不能挡住收边。
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
    """公式框左右收到墨迹外同一条窄边。上下和式子内部的间距不动。
    display 是这一页的显示列表，同一页的公式共用，不用每裁一块都重新解析整页。"""
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
    own_spans = [
        pymupdf.Rect(span["bbox"])
        for span in part.get("spans") or []
        if span.get("bbox") is not None
    ]
    ours = [box for span in part.get("spans") or [] for box in _span_ink_boxes(span, char_index)]
    # 字体框量错的字（txsys、CMSY 的根号）框在墨迹上方，按单画量出的墨迹补上，不然根号的下半截
    # 当成邻字的墨迹，左边收到根号中间
    ours.extend(pymupdf.Rect(span["ink"]) for span in part.get("spans") or [] if span.get("ink_drawn"))
    ours.extend(
        pymupdf.Rect(curve) for curve in part.get("curves") or [] if curve is not None
    )
    foreign: list[pymupdf.Rect] = []
    for span in _outside_formula(page_spans, part.get("spans") or [], char_index):
        if not str(span.get("text") or "").strip() or span.get("bbox") is None:
            continue
        # 先按坐标排除离得远的：一页的字有上千段，每块公式都要比一遍
        x0, y0, x1, y1 = span["bbox"]
        if y1 < bbox.y0 - 1 or y0 > bbox.y1 + 1 or x1 < bbox.x0 - 1 or x0 > bbox.x1 + 1:
            continue
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if any(_boxes_match(rect, own) for own in own_spans):
            continue
        foreign.append(rect)
    if page.rotation:
        return  # 转过的页面渲染出来也是转过的，按列找墨迹对不上文字坐标
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
    """公式框上下收到字形墨迹外一条窄边。

    按字体伸量算的框比字高得多，排版时按框高缩放公式，箭头、求和号所在的公式会被压得很小；
    从原页裁剪时还会把上下两行的字一起裁进来。认不出的 span（点阵字）照旧用原来的框。
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
    # 减号、等号的墨迹整个在基线上面，下沿比基线高时 d 是负的，落位时照样抬上去
    part["d"] = round(float(part.get("d") or 0) - (bbox.y1 - bottom), 1)


def _origin_key(span: dict) -> tuple[float, float]:
    return round(span["origin"][0], 1), round(span["origin"][1], 1)


# 字形墨迹离横线这么近算贴着：分子分母和分数线之间只隔一两磅
_RESTING_GAP = 2.0


def _joins_outer_shape(
    curve: pymupdf.Rect, curves: list[pymupdf.Rect], drawings: list[pymupdf.Rect], probe: pymupdf.Rect,
) -> bool:
    """横线和公式外的另一条线首尾相接：是证毕方框、表格线这类图形的一边，不是分数线。
    挪走一边，原处的方框就缺了口，公式下面多出一道杠。实心的底块（图注标签的黑底）不算。"""
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
    """横线左端接在本公式一个字的右上角：是根号的横线。被开方数在线下面，线上面常贴着上一行的降部，
    不能因此留在原处。CMEX 的大号根号提出来是字母，按位置认。"""
    if curve.height > _RULE_IMAGE_THICK:
        return False
    for span in spans:
        if not str(span.get("text") or "").strip():
            continue
        box = pymupdf.Rect(span.get("ink") or span["bbox"])
        if abs(curve.x0 - box.x1) <= _RADICAL_JOIN and abs(curve.y0 - box.y0) <= _RADICAL_JOIN:
            return True
    return False


# 根号横线的左端和根号右上角相距在这以内（点）
_RADICAL_JOIN = 1.0


def _rests_on(curve: pymupdf.Rect, glyphs: list[dict], own: set[tuple[float, float]]) -> bool:
    """横线上下贴着公式以外的字：分子在另一块里留在原处时，分数线是它们共用的，不能跟着公式挪走。"""
    if curve.height > _RULE_IMAGE_THICK:
        return False
    for span in glyphs:
        found = span.get("ink")
        if found is None or not span.get("origin") or _origin_key(span) in own or not str(span.get("text") or "").strip():
            continue
        ink = pymupdf.Rect(found)
        if min(ink.x1, curve.x1) - max(ink.x0, curve.x0) < 1:
            continue
        if -0.5 <= curve.y0 - ink.y1 <= _RESTING_GAP or -0.5 <= ink.y0 - curve.y1 <= _RESTING_GAP:
            return True
    return False


def _spans_ink(spans: list[dict]) -> pymupdf.Rect | None:
    """几个 span 的字形墨迹并起来。有一个量不出墨迹（点阵字）就不算，返回 None。"""
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
    formula: dict, drawings: list[pymupdf.Rect], block: TextBlock, glyphs: list[dict] = (),
) -> None:
    """把相交曲线并进公式框，并写回对应的 FormulaRun。换行续片各自收自己的框。"""
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
        if isinstance(run, FormulaRun) and run.name == formula["name"] else run
        for run in block.runs
    ]


def _marker_png(idx: int) -> bytes:
    """1x1 彩色像素：颜色编码公式的全局序号（R + G<<8），供插入后找回落位。
    必须全局唯一：同色的标记图会被 MuPDF 去重成同一 xref，落位全串到第一处。"""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1))
    pix.set_pixel(0, 0, ((idx + 1) & 0xFF, ((idx + 1) >> 8) & 0xFF, 200))
    return pix.tobytes("png")


def _crop_formulas(
    src_doc: pymupdf.Document, flat_units: list[list[TextBlock]], sent_formulas: list[list[dict]],
) -> pymupdf.Archive:
    # 公式从它所在的原页裁图。不能用单元最后一块的页码，跨页时会裁错页。
    # 框内相交的曲线并进裁剪，根号、分式线才不会被切掉。
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
    # 只留最近一页的显示列表：公式大致按页序排，整本都留着太占内存
    display: tuple[int, pymupdf.DisplayList] | None = None
    for formulas in sent_formulas:
        background.check_stop()
        for f in formulas:
            pieces = _formula_parts(f)
            ready = False
            for part in pieces:
                # 字框里的前导空格会让有的公式两侧很宽、有的又贴着笔画。先收到墨迹。
                part_page = int(part.get("page") or f["page"])
                if part_page not in span_cache:
                    span_cache[part_page] = _page_text_spans(src_doc[part_page])
                    char_cache[part_page] = _page_char_index(src_doc[part_page])
                if display is None or display[0] != part_page:
                    display = part_page, src_doc[part_page].get_displaylist()
                _tighten_formula_height(part)
                _tighten_formula_sides(
                    part, src_doc[part_page], span_cache[part_page], char_cache[part_page], display[1],
                )
                bbox = pymupdf.Rect(part["bbox"])
                if bbox.width < 1 or bbox.height < 1:
                    continue
                part["png"] = _FormulaImage(src_doc, part_page, bbox, part.get("spans") or ())
                part["has_img"] = True
                ready = True
            f["has_img"] = ready
    # 标记像素按全文全局序号编码（同色的会被 MuPDF 去重成同一图片，落位全串）
    marker_idx = 0
    for formulas in sent_formulas:
        for f in formulas:
            if f.get("has_img"):
                f["mid"] = marker_idx
                archive.add(_marker_png(marker_idx), f"ph_{f['name']}")
                marker_idx += 1
    return archive
