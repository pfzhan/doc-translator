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

# 整页居中排时，各行中线的偏差。行尾的逗号、斜体的外伸会让中线偏一两点
_AXIS_TOL = 2.0


def _attach_chars(page: pymupdf.Page, spans: list[dict]) -> None:
    """给 span 记下逐字的框（span["chars"]），按 span 的框对上。页面擦过以后再查就查不到了。"""
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
    """把页面内容打包成 Form XObject（在 redact 之前调用），返回 xref。
    供公式透传引用：资源沿用页面自己的，BBox 为整页。"""
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
    """把占位标记换成公式内容并校准基线。

    htmlbox 对 <img> 只做图片底部对齐，公式的 baseline 在图片内部（上下标在 baseline
    之下还有延伸），直接插会整体浮起来。所以先插一个按公式宽高占位的 1x1 标记像素，
    从落位矩形读出文本基线位置，再把公式按“bbox 底 − 公式 baseline”下移贴到精确位置。
    有源页 Form（forms）时用紧致 BBox 的包裹 Form 做矢量透传（原 glyph、可选中、选择
    不爆框）；失败回退到 PNG 位图。

    标记像素按全局序号（mid）编码，整页一次处理；所有公式绘制集中在最后一条新内容
    流里：insert_htmlbox 每块新建内容流，标记涂白后白块仍在块的内容流里，公式若画在
    之前的流里会被后续块的白块盖住上半截。
    """
    draws = []
    # 渲染后的文本 span：htmlbox 对 img 只做底部对齐，且标记底边和文本 baseline 有
    # ~1pt 系统偏差，公式 baseline 直接锚到同一行文字的 origin 上最稳
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
        # raise：孤立角标公式（²）的 origin 比正文 baseline 高，落位时按高差抬回去
        w, h = f["w"] * rs, f["h"] * rs
        d = (f["d"] - f.get("raise", 0.0)) * rs
        cy = (rect.y0 + rect.y1) / 2
        baseline = None
        best_key = (0.0, float("inf"))
        for s in text_spans:
            sb = s["bbox"]
            if not (sb[1] - 1 <= cy <= sb[3] + 1):  # 不在同一行
                continue
            # CJK line-height 放大后相邻行的 glyph bbox 会交叠：按与标记矩形的
            # y 交叠量取最大（同一行交叠最多），并列再取 x 最近
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
    """以句末标点、分号、冒号收尾（列表项常以分号收尾）。末尾的数字是脚注号，不算"""
    text = text.rstrip().rstrip("0123456789")
    return _ends_sentence(text) or text.endswith((";", "；", ":", "："))


def _row_neighbor(block: TextBlock, blocks: list[TextBlock], geo: PageGeometry | None) -> bool:
    """本栏里同一行旁边还有别的块（表格的另一格）。页边的旁注、另一栏的正文不算"""
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
    """原文绕着排开的图和图注。它们落在要写的范围里（默认是块的框）、贴着左缘或右缘，
    块里没有一行压上去，旁边至少有两行让开了。整块按一个框写，译文会盖到图上。"""
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
        # 另一侧要留出三成宽，横贯整块的公式、表格不算
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
    """写的盒子顶到原文第一行行框下沿的高度。盒子不从第一行起（跨页接过来的半段）时不算。"""
    if not block.line_rects:
        return None
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    bottom = max(r.y1 for r in rows[0])
    if not block.rect.y0 - block.size <= rect.y0 < bottom:
        return None
    return bottom - rect.y0


# 悬挂缩进的条目起头：题号（3.1、2.）、圆点、(a) 这类编号
_HANG_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*\.?\s")


def _hang_indent(block: TextBlock) -> float:
    """题号、圆点顶格，下面各行都和题号后面的正文对齐（练习题 3.1 Write out …）：续行比首行缩进多少点。
    不是这样排的块返回 0"""
    # 节标题（1.1 What is a Large Language Model?）的编号也和标题空开一截，编号留在原处
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
    """只有一行的条目：和正文同字号的题号单独一截，和后面的正文之间空出半个字以上（3.8 | Write a program …），
    正文比题号缩进多少点。不是这样排的返回 0"""
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
    # 脚注的上标编号后面也空开一截，脚注的续行顶格
    if float(spans[0].get("size") or 0) < block.size * 0.9:
        return 0.0
    gap = float(spans[1]["bbox"][0]) - float(spans[0]["bbox"][2])
    hang = float(spans[1]["bbox"][0]) - float(spans[0]["bbox"][0])
    return hang if gap >= block.size * 0.5 and hang <= block.size * 5 else 0.0


def _extend_leader(page: "pymupdf.Page", block: TextBlock, leader: Leader) -> None:
    """点线和标题一起擦掉了。按原来的点距、基线从页码前往左补点，接到译文末尾。"""
    if leader.pitch <= 0:
        return
    band = pymupdf.Rect(block.rect.x0 - 1, block.rect.y0 - 1, leader.number.x0 - 0.5, block.rect.y1 + 1)
    right = band.x0
    for item in page.get_text("dict", clip=band)["blocks"]:
        for line in item.get("lines", []):
            for span in line["spans"]:
                # 上下两条挨得近，框会探进这一行。只认基线在这一行的字
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
    """矢量透传失败、退回位图的公式，整页写完再贴。"""
    for target, png in deferred_png:
        page.insert_image(target, stream=png, keep_proportion=False, overlay=True)
    deferred_png.clear()


def _nowrap_parens(body: str) -> str:
    """短括号组包 nowrap span：MuPDF 的 CJK 换行可以在任意处断，会出现
    '（如\\n序列）' 这种断在括号中间的丑行，以及 '（y_1^{t-1}' 把右括号单独
    留在下一行。整组不拆（含公式图的组，括号跟着公式一起走），长组不包（防溢出）。"""

    def repl(m):
        inner = m.group(0)
        # 估算渲染宽度：<img> 按 3 个字符计，其余标签不计
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
    """译文 HTML。含公式哨兵时自己断行，每行 nowrap；否则仍交给 MuPDF 折行。"""

    def img_repl(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if not 1 <= number <= len(formulas):
            return ""
        formula = formulas[number - 1]
        if formula.get("has_img"):
            scale, width, height = _formula_slot(formula, block_size, render_size)
            formula["rs"] = scale
            # 槽位两侧各留 2pt：斜体字形会微微越出 bbox，紧贴排布时像被前后汉字压住
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
    """原文在格子里居中。栏中线对不上格子中线，不能拿栏来判断。"""
    clip = block.clip
    if clip is None:
        return False
    cell_mid = (clip[0] + clip[2]) / 2
    text_mid = (block.rect.x0 + block.rect.x1) / 2
    return abs(text_mid - cell_mid) <= 2.0


def _flow_y0(block: TextBlock, cjk: bool) -> float:
    """写入起点。侧标让出的第一行由单独的框从自己的左缘写，这里仍从第一行起。"""
    origin = block.line_rects[0].y0 if block.line_rects else block.rect.y0
    return origin + block.size * 0.1 if cjk else origin


def _inset_write_boxes(block: TextBlock, cjk: bool) -> tuple[pymupdf.Rect, pymupdf.Rect] | None:
    """第一行比后面几行靠右时，第一行从自己的左缘写，后面的行从栏左缘写。

    续行从下一视觉行起。同一基线的碎片如果当成下一行，续文会压到第一行和左侧标签上。
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
    """正文第一行左边的短标签。Memo.、Property Enforcement.、(2) Statistics Derivation."""
    text = block.text.strip()
    if not text.endswith(".") or len(text.split()) > 6 or block.rect.width > 150:
        return False
    return not _diagramish(block)


def _label_beside(body: TextBlock, blocks: list[TextBlock]) -> TextBlock | None:
    """和正文第一行并排、源文几乎贴到正文左缘的短标签。"""
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
    """短译文标签让出的第一行空白补上。译文比英文宽时，正文让到标签右边。"""
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
    """短侧标右边、同一基线上的正文。标签不能扩进这一行。"""
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
    """索引一栏里译文比原文多占行的条目，连同下面的条目按原文行距往下挪。

    一条一行紧挨着排，不挪的话多出来的一行没有地方，只能缩小字号。字母分组之间的空白先拿来抵，
    不够再收紧行距（见 _restack）。留原文的条目也跟着挪。跨栏跨页的段落、不属于任何块的原字
    （ink，提取时丢掉的条目）挪不了，挪到它们上面为止。返回挪过的条目的副本，框和行框都已挪好。
    """
    if geo is None or not any(b.index for b in blocks):
        return {}
    texts = {id(b): t for b, t in items if t != SPILL_TAIL}
    # 留原文的条目（BERT、CKY）也跟着挪，挪了就按原文重写
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
    """原文按 span 写出粗体、斜体标记。重写留原文的块时，粗体的页码照旧是粗体。"""
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
    """索引页按左右不重叠分栏，栏内从上到下排好，另给出每栏能写到的右缘。

    索引的行参差不齐，版面分析常常只认出一栏。条目只在本栏里写，横向和别栏的条目不相交。
    分栏只看索引条目，页首横跨几栏的标题不算；它和别的块一起放进左右相交的栏里，挡住条目。
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
    """一段条目从上往下排：每条从原位或上一条末行之下起，占它要的行数。

    索引一栏常常排到正文下沿，没有空行可借。超出 limit 时整段行距一起收紧，至多收一成，
    一行少不到一点，比单独缩小一条的字号好认；再不够就不挪。
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
    """并排的作者单位改成逐行写下。中文比原文宽，并排写会叠在一起。"""
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
    """这一块连同栏里同一行挨着它的块（间隔不到一个半字宽）占的横向范围。一行拆成几块时
    看整行：节号和节名分成两块的标题整行居中；表格一行的几格连起来不在栏中间。
    多行的块每一行都要有同一行的块才算，表题下面一行连着表头的块仍按自己的范围看。"""
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
    """上下紧挨的一块和它左缘对齐、右缘不齐：是左对齐排的一组行（对话、例句），
    某一行的中线碰巧落在栏中线附近也不是居中。宽度相同的两块居中行左右都齐，不在其内。"""
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
    """封面、题名页整页居中排，本页宽窄不一的行大多共用一条中线。块的每一行都在这条中线上
    就是居中的，返回这条中线。
    和最宽一行差不多宽的行不计数，版心里的正文行都在同一条中线上；页边旁注会把本页文字的左右范围撑宽，
    所以拿最宽的行比。左缘也相同的行是一列等宽的数，不是居中。"""
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
    """写入框和对齐方式（left、center、right）。多栏且块落在某一栏内时，居中参照该栏；
    单栏有版心时参照版心，没有时用本页最宽块。格子里原本居中的文字，改以格子中线居中。
    从栏左缘起笔的块是左对齐的，中线碰巧对上也不算居中。"""
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
    # 靠右排参照栏或版心的左右缘，本页最宽块的右缘不算
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
    # 字比本页多数大、每行都正对页面中线的是居中的标题：幻灯片最宽的一块是靠左的列表项，
    # 中线不在页面正中；标题本身是最宽的一块时，跟页宽比。论文正文左右边距对称，
    # 整行也正对页面中线，字号不大，不算。
    page_mid = geo.width / 2 if geo is not None and column is None and geo.width > 0 else None
    sizes = sorted(other.size for other, _ in items)
    on_page_mid = (
        page_mid is not None
        and block.size > sizes[len(sizes) // 2] * 1.15
        and block.rect.width < geo.width * 0.85
        and all(abs((line.x0 + line.x1) / 2 - page_mid) < 4 for line in block.line_rects or [block.rect])
    )
    # 弹点、短横线起头的列表项是靠左排的，中线碰巧对上栏中线也不算居中。索引的条目也一样，
    # 缩进的子条目常常正好落在栏中间。(a) 子图小标题和 1. 引言这种居中的标题不在其内。
    # 一行拆成几块时整行也要对着栏中线（见 _row_span）：表格一行的几格都按栏中线写会叠在一起
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
    # 几行都对着本页中线、字号一样的长副标题也是居中排的。正文段的末行短，对不上中线；
    # 作者名下面接着小一号的单位，各行另有排法
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
        # 页面左右边距不对称时版心和最宽块都不在这条中线上，按中线到本页文字两侧的距离留宽
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
        # 伸进页边的那几行不算，译文照栏或版心的右缘断行。旁注另成一栏时，正文栏的右缘也会被这几行撑出去
        edges = [
            edge for edge in (margin[1] if margin is not None else None, _frame_x1(block, geo)) if edge is not None
        ]
        x1 = min(edges) if edges and block.rect.x1 > min(edges) + 2 else block.rect.x1
        rect = pymupdf.Rect(block.rect.x0, y0, x1 + 2, block.rect.y1 + pad)
    return _clip_write_rect(rect, block.clip), "center" if centered else "left"


def _inside_right(block: TextBlock, x1: float) -> bool:
    """块收在 x1 以内。块的右缘只由个别几行撑出去的也算：TeX 断不开的长词、引用会让一行
    伸进页边，同一段收在 x1 的行不比伸出去的少。"""
    if block.rect.x1 <= x1 + 2:
        return True
    out = [row.x1 for row in block.line_rects if row.x1 > x1 + 2]
    held = sum(abs(row.x1 - x1) <= 2 for row in block.line_rects)
    return bool(out) and max(out) >= block.rect.x1 - 1 and len(out) <= held


def _frame_x1(block: TextBlock, geo: PageGeometry | None) -> float | None:
    """块在版心里时版心的右缘。"""
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
    """靠右排的块：章首的引语、署名。每行都收在栏的右缘，左缘离栏的左缘有栏宽的四分之一以上。
    TeX 排的署名末尾常带着一个词间空格，斜体还有倾斜补偿，字的右缘会比栏的右缘短半个字宽左右。
    栏的右缘要有本页更宽的块排到，栏只由这一块撑出来时右缘就是它自己的右缘。
    有两行起笔对齐的是左对齐的正文，左边空着也不算（图在前一页时绕图的正文左边是空的）。
    同一行栏里左边还有别的块或图的不算：一行拆成几块时的右边那一块，绕着图排的正文。
    左边几个字宽以内有别的块的行也不算：两端对齐的窄栏里一行拆出来的末一个词。引语末行用
    \\hfill 推到行尾的署名离左边的字很远。
    左缘和上下几行的起笔对齐的也不算：表头、列表这类左对齐的字。"""
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
    """格子左右是硬边界。上下保留行框高度。

    贴着格子下沿的一行，如果把下沿余量裁掉，行框会矮过一行。排版器误以为
    高度不够，把这一格的字号单独缩小，同一张表里就会大小不齐。
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
        # 版心里的段落右扩到版心右缘为止。栏的右缘会被伸进页边的行撑出去
        limit = min(limit, frame_x1 + 2)
    if block.clip is not None:
        cap = block.clip[2] - 0.8
        limit = cap if limit is None else min(limit, cap)
    return limit


@dataclass(frozen=True)
class _Carry:
    """续页接着写的后半。普通跨页按栏宽断好；共基线的半句按续行自己的宽度再断。
    size 是上一块的字号：后半按它断的行，续块的字号稍大一点就放不下，每行末尾挤出一个字"""

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
    """这一页原文正文的下沿。页码带里的块不算，译文不写进这条线以下。
    ink 是算在正文里、留在原处的字和图框（见 _body_ink）"""
    floor = content_floor(height)
    bottoms = [block.rect.y1 for block in blocks if block.page == page and block.rect.y0 < floor - 4]
    bottoms += [rect.y1 for rect in ink if rect.y0 < floor - 4]
    if not bottoms:
        return floor
    return max(bottoms)


def _is_column_top(block: TextBlock, blocks: list[TextBlock]) -> bool:
    """这一栏最靠上的块。页首不再下移，上边距跟原文对齐。"""
    for other in blocks:
        if other is block or other.page != block.page or other.rect.y0 >= block.rect.y0 - 0.5:
            continue
        overlap = min(other.rect.x1, block.rect.x1) - max(other.rect.x0, block.rect.x0)
        if overlap > 8:
            return False
    return True


def _keeps_line_pad(block: TextBlock) -> bool:
    """页末只有一两行。行框余量裁掉之后，中文装不下一行，字号会被单独缩小。

    更高的段落仍停在原文下沿。长译文不能借这点余量写进下边距。
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
    """页首对齐原文上沿，正文不写进原文下沿以下。格子下沿不裁，避免表内字号被单独缩小。

    页末一两行的行框余量留着。那一点高度只够放下这一行，不会把段落拉进下边距。
    """
    y0, y1 = rect.y0, rect.y1
    if cjk and _is_column_top(block, blocks):
        # 整框上移，不把框拉高。拉高之后一行会换到两行。
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
    """擦除以后原处还留着、不属于任何块的字，如提取时丢掉的行间公式。
    目录的点线、随公式图搬走的字形已经擦掉，不在其内。"""
    # 一页几千个词，每个都和所有行框比。点和框都按 MuPDF 的单精度先换好，结果和 Rect 的判断一样
    boxes = [single_precision(rect + (-1, -1, 1, 1)) for block in blocks for rect in (block.rect, *block.line_rects)]
    kept = []
    for word in page.get_text("words"):
        center = pymupdf.mupdf.FzPoint((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
        x, y = center.x, center.y
        if not any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in boxes):
            kept.append(pymupdf.Rect(word[:4]))
    return kept


def _body_ink(page: "pymupdf.Page", kept: list[pymupdf.Rect], marks: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """留在原处、算在正文里的字和图框（见 _page_content_bottom）：页底的行间公式、图。
    只有页码的那一行不算，页码常常是纯数字，提取时被丢掉，擦完还留在原处（见 _kept_ink）"""
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
    """右扩时挡路的障碍：和原文某一行并排的才算。写入框比原文的行上下多出留白，
    上一行、下一行的框会探进这几点，不在旁边（索引里缩进的子条目压着上一条的下沿）。
    框往下多出来的整行也算一行，那里将来要写字。"""
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
    """原文的各行，加上上下相邻、间隔不到半行高的两行合起来的带。夹在两行中间的字
    （分情况的行间公式右边的式号）和每一行都只交叠一小半，和合起来的带交叠过半。"""
    bands = list(rows)
    ordered = sorted(rows, key=lambda row: row.y0)
    for upper, lower in zip(ordered, ordered[1:]):
        if lower.y0 - upper.y1 < min(upper.height, lower.height) * 0.5:
            bands.append(upper | lower)
    return bands


def _on_rows(ink: pymupdf.Rect, rows: Sequence[pymupdf.Rect]) -> bool:
    """这处字和某一行（或相邻两行合起来）的上下交叠过它自己高度的一半：在这几行的旁边，
    不是上一行、下一行。"""
    return any(min(ink.y1, band.y1) - max(ink.y0, band.y0) > ink.height * 0.5 for band in _row_bands(rows))


def _drawn_marks(page: "pymupdf.Page") -> list[pymupdf.Rect]:
    """页上画出来的线、框、色块和图片。只填白色的块看不见，不算。"""
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
    """只有段首缩进的一行留在本页（栏），段落接到下一页：本栏的左缘。
    续块从它那一栏的左缘起笔才是正文段落，译文和别的段落一样从栏左缘写，不留缩进。
    续块缩着排的是列表项的悬挂缩进，照旧从第一行的左缘写。"""
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
    """只有一行的段落比栏左缘正好缩进一个段首缩进：栏左缘。多行段落的译文从栏左缘写，首行缩进不留，
    只有一行的也一样顶格。缩进的宽度对不上的是引语、代码这些整块缩着排的，列表项、带标签的不算。
    段落只有这一行，句子就在这一行收住；没有句末标点的是表格里的行名（BiDAF+ELMo (Single)）"""
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
    """这一栏往下还能写到哪里。停在页码带上面，也停在后面的脚注或其他块前面。

    页码常常是纯数字，提取时被丢掉，x 方向也不一定和这一栏相交。
    块自己已经在页脚带里时不再下刀，贴底的脚注不会因此被缩掉。
    正文下沿以上的块停在原文正文下沿，不写进下边距。body 是算在正文里的留在原处的字和图框（见 _body_ink），
    正文下沿算到它们下面。
    kept 是原处留着的字（见 _kept_ink），marks 是画出来的图框、图片（见 _drawn_marks），
    块下方正文区里的都要让开。marks 的框就是线条本身，末行下面的行距又可以伸过下沿（见 _filled_bottom），
    只让 2pt 字就贴在线上，这里空出半个字号。
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
    """按满行切开。这一页放得下几行整行，就留几行；剩下的从下一页继续。

    next_room 是续块从起笔往下能写的高度。余下的行按正常行距在那里写得下时，本页按正常行距数行数。
    写不下时，页底原文的两行按原行距放不下两行译文，余下的挤到下一页只能缩小字号，
    这时和 _fit_typeset 一样先压行距再数行数。末行下面的行距是空白，
    也可以伸过 limit（见 _filled_bottom），但写入框到 page_bottom 为止：limit 就是原文正文下沿时，
    框在那里被裁掉，多算的这截放不下，这几行会被整体缩小。"""
    prepared = space_around_latin(separate_after_formula(text))
    metrics = _formula_metrics(formulas, block_size, em, block_size * 0.88 if cjk else None)
    widths = {index: metric.slot_width for index, metric in metrics.items()}
    raw = break_lines(prepared, em, max(width, 1.0), widths)

    def fitting(gap: float, room: float) -> int:
        """高 room 的地方按行距 gap 放得下几行"""
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
        # 返回已经断好的行。返回原文的话，调用方会按一行去量盒子，
        # 整段就被挤进一行高，字号缩到看不清。
        return "\n".join(raw), ""
    if count <= 0:
        return "", prepared
    return "\n".join(raw[:count]), "\n".join(raw[count:])


def _filled_bottom(y0: float, lines: int, em: float, cjk: bool, room: float, page_bottom: float) -> float:
    """_split_filled_lines 留在本页的几行，写入框的下沿。框要有整行距高，末行字形下面
    还有一截空白，大致是行距超出字号的那部分。这截空白可以伸过 room，字仍在 room 以内。"""
    gap = 1.45 if cjk else 1.2
    return min(y0 + lines * em * gap + em * 0.3, room + (gap - 1) * em, page_bottom)


def _own_line_box(block: TextBlock, cjk: bool) -> pymupdf.Rect:
    """半句和续行只占自己这一行。框高至少一行，左右不超出这一行。"""
    origin = block.line_rects[0].y0 if block.line_rects else block.rect.y0
    y0 = origin + block.size * 0.1 if cjk else origin
    size = block.size * 0.88 if cjk else block.size
    pad = block.size * 0.5 if cjk else block.size * 0.3
    gap = 1.45 if cjk else 1.2
    bottom = max(block.rect.y1 + pad, y0 + size * gap + size * 0.3)
    rect = pymupdf.Rect(block.rect.x0, y0, block.rect.x1 + 2, bottom)
    return _clip_write_rect(rect, block.clip)


# break_lines 在行末收起跨行的样式标记
_TRAILING_CLOSERS_RE = re.compile(r"(?:\{/[bizcu]\})+$")


def _after_first_line(source: str, head: str) -> str:
    """第一行从原文切下去。断行吃掉的那个空格不要留在下一行行首。
    粗体、斜体跨过第一行时，break_lines 在行末补了收尾标记，原文里没有：去掉再比，剩下的重开这些标记。"""
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
    """第一行按半句的宽度断，剩下的不断行，写入时按续行自己的框再断。

    按窄宽断好的换行不能留在后半里。break_lines 把换行当成硬断，
    短行再送进更宽的框也不会并回去，侧标旁边的正文就会一直只有第一行那么窄。
    按整宽预先断好也不行：续行旁边有图时，写入还要让开图，预断的换行会留下半截短行。
    """
    prepared = space_around_latin(separate_after_formula(text))
    # em 是没缩过的译文字号，公式按它保持原大（见 _formula_slot）
    metrics = _formula_metrics(formulas, block_size, em, em)
    widths = {index: metric.slot_width for index, metric in metrics.items()}
    raw = break_lines(prepared, em, max(host_width, 1.0), widths)
    if not raw:
        return "", ""
    if len(raw) == 1:
        return raw[0], ""
    rest = _after_first_line(prepared, raw[0]).strip()
    # 半句的位置太窄，放不下第一个词时 break_lines 按字符硬切（BER / T）。
    # 后面的行比这里宽，这个词整个留给续行
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
    """on_page(已写页数, 要写的页数) 在每写一页之前调用。pages 给出时只写这些页。"""
    rtl = target_lang.split("-")[0] in RTL_LANGUAGES
    # CJK 字体的行框比拉丁高（约 1.31em vs 1.16em），且字形顶部会越出给定区域：
    # 按原字号写入会和下一行叠在一起，字号缩小并下移补偿
    # 粤语、文言文走同一套 CJK 字体，行框一样偏高
    cjk = target_lang.split("-")[0] in ("zh", "ja", "ko", "yue", "wyw")
    # lang 让 MuPDF 为中日韩选对字形；dir=rtl 让阿拉伯语、希伯来语从右往左排
    div_attrs = f' lang="{html.escape(target_lang)}"' if target_lang else ""
    if rtl:
        div_attrs += ' dir="rtl" style="text-align: right"'
    continuations = continuations or {}
    # 跨页段落的公式记在上一页那块名下，续写到这一页时原处的分数线要按续块擦
    spill_hosts = {id(tail): host for host, tail in continuations.items()}
    carried: dict[int, _Carry] = {}
    doc = pymupdf.open(src_path)
    by_page = _pages_to_write(blocks, translations)
    if pages is not None:
        by_page = {pno: items for pno, items in by_page.items() if pno in pages}

    # 写入时只和同页的块比较。每块都扫一遍整本书，几百页就很慢。
    page_blocks: dict[int, list[TextBlock]] = {}
    for b in blocks:
        page_blocks.setdefault(b.page, []).append(b)
    # 段首缩进的宽度，只有一行的段落按它认出来顶格写（见 _indent_edge）
    indent = _paragraph_indent(blocks)
    # 各页留在原处的字和画出来的图框（见 _room_below）。写到这一页时换成擦完以后的
    page_inks: dict[int, tuple[list[pymupdf.Rect], list[pymupdf.Rect]]] = {}
    # 第 pno 页的 _body_ink，和算它用的 page_inks 那一份
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
        """第 pno 页的 spare_below、next_movable、latest_start、movable_box。写这一页的时候用；写上一页的跨页段落时也用它估续块能借到多少，
        那时这一页还没擦，items 是要写的块，kept、marks 按没擦的页认，没有写过的块。
        hosts 为假时下面的跨页段落不跟着挪：它往下挪要看再下一页，估续块的时候不往后看"""
        page = doc[pno]
        same_page = page_blocks.get(pno, [])
        bounds = text_page_rect(page)
        geo = geometries[pno] if geometries and pno < len(geometries) else None
        visible = [(b, t) for b, t in items if t != SPILL_TAIL]
        texts = {id(b): t for b, t in items}
        # 下面各段按正常行距排出来要多高，压一点字距（见 SQUEEZE）后要多高、排几行。只看译文和框宽，算一次就够
        heights: dict[int, tuple[float, float, int]] = {}
        host_starts: dict[int, float] = {}

        def plain_rect(block: TextBlock) -> pymupdf.Rect | None:
            """按普通段落写的块（不跨页、不分两个框、不是格子和索引）写在哪个框里，和下面写的时候一样算。
            别的写法返回 None"""
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
            """room 停在下面紧挨着的一段上，那一段还没写、按普通段落写、能跟着上下挪时，返回它和它的框"""
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
            """能上下挪的一段写在哪个框里（见 plain_rect），不能挪时 None。above 是紧挨着它上面的一段"""
            # 挪的只能是正文段落。图里的标签（树状图上的词）各自对着图上的位置，挪了就对不上
            if block.body_size and abs(block.size - block.body_size) > 0.5:
                return None
            box = plain_rect(block)
            if box is None or pinned(block, box, above):
                return None
            return box

        def pinned(block: TextBlock, box: pymupdf.Rect, above: TextBlock | None) -> bool:
            """这一段的框里有不跟着它挪的东西"""

            # 行内公式按译文里的占位贴，跟着段落走。留在原处的字（丢掉的行间公式、式号）和图框、
            # 框里并排的别的块（Airlines-Wagner）不跟着段落挪，那一段挪了就压上了。
            # 从上面伸进顶上一截的（公式框）往上提时按 lift_limit 让开；
            # 下面一段的框和这一段底下叠着一截，它跟着往下挪，由 room 管
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
            """跨到下一页（下一栏）的段落按宿主写（不是分两个框、半句、和续行同一行的写法）时写在哪个框里。
            它往下挪，留在这一页的行少了，多出的行写到续块那里"""
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
            """跨页段落最晚从哪里起笔：留在这一页的行照正常行距放得下，余下的行续块那里也照正常行距写得下。
            原处就放不下时不挪，原处起笔"""
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
            """能挪的一段（见 next_movable）最晚从哪里起笔，它和下面一串能挪的段落才都照正常行距写得下。
            比原来的起笔低是它能往下挪多少，借给上面一段；比原来高是它得往上提多少。
            下面一段得往上提时，这一段要在它上面半行处收住，原来的框底就不能再用了"""
            if id(block) in continuations:
                return host_latest(block, box)
            size = block.size * 0.88 if cjk else block.size
            gap = 1.45 if cjk else 1.2
            if id(block) not in heights:
                # 和 _fit_typeset 一样按公式槽断行，公式高的行行距跟着加
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
            # 在框里写到原文末行为止（盒子比末行行框多出首行上方 0.3em 的出头）
            own = min(box.y1, block.rect.y1 + size * 0.3)
            below = next_movable(block, box, room)
            if below is not None:
                start = latest_start(*below)
                if start > below[1].y0:
                    room = max(room, start)
                elif start < below[1].y0 - 0.5:
                    room, own = min(room, start), -math.inf
            # 框里放不下时和 _fit_typeset 往下长一样，写到下面一块往上半行处。
            # 估的宽度和写的时候差一点就少算一行，留 1pt
            reach = room - size * (gap * 0.5 - 0.3)
            latest = max(own, reach) - normal - 1
            # 在原处写得下（和 _fit_typeset 一样量，见 _glyph_bottom，也可以压一点字距）就不用往上提，
            # 原文末行往下的那截框只是挪下去时不用
            # 末行下面那截空白伸过框底时按下面不动的东西量，下面一段借出来的地方它要挪下去半行，不算
            bottom = _glyph_bottom(box.y1, fixed, size, gap) if cjk and count > 1 else box.y1
            if own > -math.inf and box.y0 + squeezed <= bottom + 1:
                latest = max(latest, box.y0)
            return latest

        def spare_below(block: TextBlock, rect: pymupdf.Rect, room: float) -> tuple[TextBlock, float] | None:
            """room 停在下面紧挨着的一段上，那一段按正常行距写完还空着的高度：这一段可以借来往下伸，
            那一段跟着往下挪。返回那一段和它最低能从哪里起笔。
            那一段自己也能向它下面的一段借，一段接一段往下挪，借到的算在它能写到的地方里"""
            below = next_movable(block, rect, room)
            if below is None:
                return None
            start = latest_start(*below)
            if start <= below[1].y0:
                return None
            return below[0], start

        return spare_below, next_movable, latest_start, movable_box

    def tail_room(tail: TextBlock, width: float, em: float) -> float:
        """续块从起笔往下能写多高，和写续块（SPILL_TAIL）时的 room 一样算。
        续块那一页还没写，原文的字都在块里，留在原处的字、图框照样认得出来。
        续块右边 width 以内并排着图、别的块时，按 width 断的行到那里要绕开，折成更多行，不算它有地方。
        下面紧挨着的段落还空着的话，续块写的时候能借来往下伸（见 spare_below），算在里面；
        下面一串得往上提时，提上去要用的地方不算。
        em 是续写的字号"""
        key = (id(tail), round(width, 2), round(em, 2))
        if key not in tail_rooms:
            tail_rooms[key] = measure_tail_room(tail, width, em)
        return tail_rooms[key]

    tail_rooms: dict[tuple[int, float, float], float] = {}

    def later_flow(pno: int) -> tuple[Callable, Callable, Callable, Callable]:
        """还没写的第 pno 页的 page_flow。下面的跨页段落不跟着挪（见 page_flow 的 hosts）"""
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
                # 往下伸到下面那段最低起笔往上半行（同 _fit_typeset），换成和 room 一样的算法：框比 room 多出末行下面的行距。
                # 下面那一串照正常行距写不下、得往上提时，续块在提上去的起笔上面半行收住（同 latest_start）
                start = latest_start(*below)
                gap = 1.45 if cjk else 1.2
                reach = start - em * (gap * 0.5 - 0.3) - (gap - 1) * em
                if start > below[1].y0:
                    room = max(room, reach)
                elif start < below[1].y0 - 0.5:
                    room = min(room, reach)
        return room - y0

    def spill_target(block: TextBlock, width: float, em: float) -> tuple[TextBlock, pymupdf.Rect, float] | None:
        """页底最后一段放不下时，末几行接到下一页页首那一段前面写：那一段和它下面能挪的段落往下让。
        返回那一段、它的框、它前面能接着写多高（和 tail_room 一样，从起笔量到能写到的地方）。
        下一页页首是标题、图注这类不能挪的块，或者栏比这一段窄时没有"""
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
        # 接着写的几行末行下面半行是下一段的起笔（同 claims）；换成 _split_filled_lines 的 next_room
        lowest = latest_start(head, box) - em * gap * 0.5
        return head, box, lowest - box.y0 + em * 0.3 - (gap - 1) * em - 1

    # 页底一段接到下一页写的末几行，按下一页页首那一段记
    spilled: dict[int, tuple[pymupdf.Rect, _Carry]] = {}

    for written, (pno, items) in enumerate(by_page.items()):
        if on_page is not None:
            on_page(written, len(by_page))
        page = doc[pno]
        same_page = page_blocks.get(pno, [])
        # 旋转页的 page.rect 是转过的，和文字坐标比较要用没转的页面框
        bounds = text_page_rect(page)
        items = _hosts_first(items, continuations)
        visible = [(b, t) for b, t in items if t != SPILL_TAIL]
        geo = geometries[pno] if geometries and pno < len(geometries) else None
        mid_map = {f["mid"]: f for b, _ in visible for f in (formulas_map or {}).get(id(b), []) if "mid" in f}
        # redact 之前把整页打包成 Form：公式透传从它里面裁剪原 glyph；
        # 同时收集页面全部 span（检测上标区域被上一行降部侵入的污染）。
        # 没有 Unicode 的点阵字报的是字体框，比墨迹高出半行，拿来挖洞会把公式自己挖掉，不参与
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
            # 公式的字从正文 span 里切出来时，裁公式要按字认，擦了就查不到了
            if any(
                span.get("source")
                for f in mid_map.values()
                for part in _formula_parts(f)
                for span in part.get("spans") or []
            ):
                _attach_chars(page, page_spans)
        forms = _FormulaSources(doc, page, _page_form(doc, page), mid_map.values()) if mid_map else None
        # 公式贴的是擦除前打包的原页，不受影响。点阵字的字形是全文共用的，擦除、裁公式的过滤会改到它，
        # 这一页写完原样写回
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
        # 挪过的索引条目，别的块右扩、绕排时按挪过的位置让开
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
            # 目录条目的点线一起擦掉，写完译文再从页码前补到译文末尾
            if b.leader is not None:
                erasure.spans.extend(b.leader.dots)
            # 公式收进来的分数线、根号横线随公式图贴到新位置，原处的要擦掉。
            # 行框按墨迹收紧以后，分数线常落在行框外面
            erasure.lines.extend(
                pymupdf.Rect(curve) + (-0.5, -0.5, 0.5, 0.5)
                for key in (id(b), spill_hosts.get(id(b)))
                for f in (formulas_map or {}).get(key, [])
                for part in _formula_parts(f)
                for curve in part.get("curves") or []
                if curve is not None and int(part.get("page", f.get("page", pno))) == pno
            )
            # 公式图按框整块裁下来贴到新位置，框里起笔的字形原处都要擦。没有 Unicode 的字形
            # （CMEX 的大号范数线提出来是 \r）当成空白，不在 span 里，按 span 擦不到
            erasure.spanless.extend(
                pymupdf.Rect(part["bbox"])
                for key in (id(b), spill_hosts.get(id(b)))
                for f in (formulas_map or {}).get(key, [])
                for part in _formula_parts(f)
                if part.get("bbox") is not None and int(part.get("page", f.get("page", pno))) == pno
            )
        # 擦之前在原文的位置放好标记，写完把译文插回去，拖选、复制按阅读顺序走
        order = ReadingOrder(page, [b for b, _ in items])
        erase(page, erasure)
        order.after_erase()
        # 擦完还在原处的字（丢掉的行间公式、式号），译文右扩、续写都要让开
        kept = _kept_ink(page, same_page)
        # 续写往下伸时还要让开图框、图片，它们不在块里。写译文之前读：写进去的公式也是画出来的
        spills = any(t == SPILL_TAIL or id(b) in continuations for b, t in items)
        marks = _drawn_marks(page) if spills else []
        page_inks[pno] = kept, marks
        # 重写内容流会把 /Resources 改写成内联字典；排版器写入后 /XObject 条目上百个，
        # 这么大的内联嵌套字典在保存的 GC（garbage≥3）里会被写坏（第一个表单的条目
        # 指向字体流，内容静默丢失）。规范成间接引用避开这个 MuPDF bug。
        res_v = doc.xref_get_key(page.xref, "Resources")[1]
        if res_v.startswith("<<"):
            res_x = doc.get_new_xref()
            doc.update_object(res_x, res_v)
            doc.xref_set_key(page.xref, "Resources", f"{res_x} 0 R")
        images_before = {img[0] for img in page.get_images(full=True)}
        deferred_png: list[tuple[pymupdf.Rect, bytes]] = []
        starts = _body_first_starts(same_page, items, cjk, formulas_map)

        def write_floor(block: TextBlock, rect: pymupdf.Rect) -> float:
            """写入框最低能到哪里：写的时候 _fit_page_margins 会把框裁在原文正文下沿。"""
            box = pymupdf.Rect(rect.x0, rect.y0, rect.x1, bounds.y1)
            return _fit_page_margins(block, box, same_page, bounds.height, cjk, body_ink(block.page)).y1

        def write_one(
            block: TextBlock,
            text: str,
            rect: pymupdf.Rect,
            align: str,
            formulas: list[dict] | None = None,
            justify_last: bool = False,
            # source：block 是挪过位置的副本（索引条目）时的原块，阅读顺序的标记放在原块上
            x_cap: float | None = None,
            source: TextBlock | None = None,
            # 续写的后半按上一块的字号写（见 _Carry）
            block_size: float | None = None,
            # 往下挪的起笔、往下伸到哪里，见 _fit_typeset
            lower: tuple[float, float] | None = None,
            room: float | None = None,
            # 起笔被上一段推下来了，首行不再按原文第一行往上提
            pushed: bool = False,
            # room 是下面一段借出来的（见 _fit_typeset）
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
            # 目录的页码留在原页上，译文不能写过去
            obstacles += [ob.leader.number for ob in same_page if ob.leader is not None]
            obstacles += [ink for ink in kept if _on_rows(ink, block.line_rects or [block.rect])]
            if geo is not None:
                # 带框线的表和算法框画着外框，右扩也不能压上去
                obstacles = [*obstacles, *geo.figures, *(table.rect for table in geo.tables)]
            beside = _beside_rows(obstacles, rect, block.line_rects or [block.rect])
            limit = _write_limit(block, rect, geo, beside)
            if x_cap is not None:
                limit = x_cap if limit is None else min(limit, x_cap)
            # 靠右排的收在原来的右缘，不往右扩，放不下时往左扩，最多到页宽 10% 处或格子左缘
            left_limit = None
            if align == "right":
                limit = rect.x1
                left_limit = block.clip[0] + 0.8 if block.clip is not None else bounds.width * 0.1
            # 跨页接过来的半段按上一页的栏宽写，要让开的图可能在块的框外面
            holes = _wrap_holes(block, obstacles, rect | block.rect)
            # 题号、圆点顶格的条目，正文照原文对齐在题号右边的一列。续写的后半、挪过位置的副本不按原块的行算
            hang = (
                _hang_indent(block)
                if align == "left" and source is None and formulas is None and abs(rect.x0 - block.rect.x0) <= 1
                else 0.0
            )
            first_room = None if pushed else _first_row_room(block, rect)
            # 从右往左仍交给 HTML 盒子。其余段落由排版器断行：有公式图时按行落位，
            # 没有公式图时写进一个锁住换行的盒子，避免一行嵌一次整套字体。
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
            # 右扩、左扩、绕排时字会写出框外，按能写到的最左、最右边算
            left = _expand_left(rect, left_limit, beside).x0 if left_limit is not None else rect.x0
            reach = pymupdf.Rect(left, rect.y0, max(rect.x1, limit or rect.x1), rect.y1) | block.rect
            order.record(source or block, before, reach + (-2, -2, 2, 2))

        # 本页写好的块：原块、末行行框的下沿、字号
        ends: dict[int, tuple[TextBlock, float, float]] = {}
        # 本页已经写过的块，上一段伸下来以后各段最高从哪里起笔（见 spare_below）
        written_ids: set[int] = set()
        claims: dict[int, float] = {}

        def paragraph_room(block: TextBlock, rect: pymupdf.Rect) -> float | None:
            """段落往下能写到哪里（见 _room_below）。表格里上下相邻的格子（旁边还有别的格子）、
            索引的条目各自对着原来的行，没有"""
            if block.index or _row_neighbor(block, same_page, geo):
                return None
            return _room_below(page, block, rect.y0, rect.x0, rect.x1, same_page, kept, marks, body_ink(block.page))

        def lift_limit(block: TextBlock, rect: pymupdf.Rect, size: float) -> float | None:
            """往上提最多提到哪里：上面最近的东西下面半行。写好的块按末行算，没译的块、留在原处的字、
            图框按原处算。上面还有没轮到的块（它往下长会压上来）、上面什么都没有（栏顶）时不提"""
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
            """上面紧挨着的一段译文写满了自己的框，末行贴着这一段的起笔，两段读起来像一段。
            译文不缩进，段落只靠上下的空隙分开：起笔至少比上一段末行低半行。
            上一块没有在句末、分号、冒号收住的是同一句断开的两半，不隔开。
            room 是 paragraph_room 算出的下沿，没有时不挪。"""
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

        # 续行可能排在半句前面。先把后半放进 carry，写的时候才不会丢掉。
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
        # 能挪的一段紧挨着上面哪一段写（见 next_movable），和它自己的框
        follows: dict[int, tuple[TextBlock, pymupdf.Rect]] = {}
        # 已经轮到过的块。译文全挪去宿主的续块、整段写去续页的宿主轮到了也不写字
        reached: set[int] = set()

        def spill_last_lines(
            block: TextBlock,
            text: str,
            rect: pymupdf.Rect,
            align: str,
            room: float,
            pushed: bool,
        ) -> bool:
            """这一段是这一页最后一段正文，照正常行距放不下、上下都没有空：留在这一页的行照正常行距写，
            余下的接到下一页页首（见 spill_target）。余下的行里有行内公式时不接，公式图只贴在本页"""
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
                # 末行下面的行距是空白，框可以伸过 room，字仍在 room 以内（同 _filled_bottom）
                size = carry.size * 0.88 if cjk else carry.size
                bottom = room + ((1.45 if cjk else 1.2) - 1) * size
                placed = pymupdf.Rect(rect.x0, rect.y0, rect.x0 + carry.width, max(bottom, rect.y0 + 4))
                # 上一页按 tail_room 算过能向下面一段借的，这里写不下就伸进去，那一段跟着往下挪
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
            # 栏宽参考：单栏用本页最宽块；多栏用本块所在栏。居中块用整栏宽，
            # 短译文不再被小框挤成孤字行。排版器按这个宽度断行。
            rect, align = _write_rect(b, visible, cjk, rtl, geo)
            inset = _inset_write_boxes(b, cjk)
            if inset is not None and id(b) not in stacked:
                first_box, rest_box = inset
                # 靠右排的引语加署名首行短、起笔靠右：照样按两个框断行，每行收在原来的右缘
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
                # 左侧的旁注靠右对齐，贴着正文。放不下一行时照常从左起笔
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
                    # 上一页页底那一段的末几行接在这一段前面，这一段从它下面半行起笔
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
                # 下面有哪些块按这一段原来的位置找。挪下去的起笔可能已经低过下一段的顶，从那里往下找会漏掉它
                own = rect
                if pushed:
                    rect = pymupdf.Rect(rect.x0, claim, rect.x1, rect.y1)
                # 这一段和下面一串段落照正常行距写不下，上面还空着：往上提，最多提到上面最近的东西下面半行。
                # 上面一段写的时候已经按同样的算法提过，上面的空一段段攒下来
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
                # 格子里的字写不下也不伸进下一格
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
                    # 盒子伸过了原来的下沿（盒子比末行行框多出 0.3em），伸进了下面一段的框：
                    # 那一段的起笔挪到末行下面半行
                    if end + size * 0.3 > rect.y1 + 0.5:
                        claims[id(borrowed[0])] = end + size * (1.45 if cjk else 1.2) * 0.5
                continue
            bformulas = (formulas_map or {}).get(id(b), [])
            # 行尾半句只有几个词宽。按它的宽度断行，续页也会被挤小。
            # 原文擦掉，整段按续页的栏宽写在续页上。
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
            # 上面一段借了这一段往下挪出的空（见 host_latest）
            claim = claims.get(id(b))
            if claim is not None and claim > rect.y0:
                rect = pymupdf.Rect(rect.x0, claim, rect.x1, rect.y1)

            def split(y0: float) -> tuple[str, str]:
                return _split_filled_lines(t, bformulas, size, b.size, rect.width, cjk, y0, room, floor, next_room)

            def squeezed(head: str, y0: float) -> bool:
                """留在这一页的几行照正常行距放不下（同 _split_filled_lines 数行数）"""
                gap = 1.45 if cjk else 1.2
                lines = head.count("\n") + 1
                return bool(head) and y0 + lines * size * gap + size * 0.3 > min(room + (gap - 1) * size, floor) + 1

            head, rest = split(rect.y0)
            # 照正常行距放不下、上面还空着：往上提。提得越高这一页放的行越多，找放得下的最低起笔
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
                # 提上去多放的行照样挤（下一页接不住）：行数不变，提到这几行照正常行距放得下的地方
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
    """同页的续块挪到宿主后面。

    内容流先画右栏时，右栏顶的续块会排在左栏宿主前面。轮到它时还没有接续的译文，
    原文已经擦掉，这一段就整段没了。
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
