"""Mark the spans on a page that belong to formulas: math fonts and alphabets, radicands,
fractions, accents, scripts and underlines."""
import math
import re
import struct
from collections.abc import Sequence

import pymupdf

from .pdf_blocks import _INK_PAD, CJK_RE, _by_chars, _font_is_math, _span_is_math


def _line_box(line: dict) -> pymupdf.Rect:
    """一行字的框。数学字体的 span 只算基线上下一个字高。

    PyMuPDF 按字体的上下伸量给框，CMSY、CMEX 的伸量是给大号括号、根号留的，
    一个箭头就能把行框撑到两行高。行框偏高，译文就从上一行的位置写起。

    分析时每行要算好几遍，按坐标取最值，不逐个建 Rect 求并集。两个 Rect 求并集在 MuPDF 里按单精度算，
    有两截以上时结果也按单精度存，和用 Rect 求并集得到的框一样，挨着阈值的比较才不变。"""
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    count = 0
    for span in line["spans"]:
        if not span["text"].strip() or "bbox" not in span:
            continue
        left, top, right, bottom = _text_bounds(span)
        if left >= right or top >= bottom:
            continue
        x0, y0, x1, y1 = min(x0, left), min(y0, top), max(x1, right), max(y1, bottom)
        count += 1
    if not count:
        return pymupdf.Rect(line["bbox"])
    if count == 1:
        return pymupdf.Rect(x0, y0, x1, y1)
    return pymupdf.Rect(struct.unpack("4f", struct.pack("4f", x0, y0, x1, y1)))


def _text_bounds(span: dict) -> tuple[float, float, float, float]:
    """_text_box 的四个坐标。"""
    x0, y0, x1, y1 = span["bbox"]
    ink = span.get("ink")
    if not _span_is_math(span) or not ink:
        return x0, y0, x1, y1
    top, bottom = ink[1] - _INK_PAD, ink[3] + _INK_PAD
    if span.get("ink_drawn"):
        # 单画量出的墨迹不夹在字体框里：字体框量错了（txsys 的根号往基线下伸了大半个字）。
        # 连上起笔点：宽波浪号这类重音整个在起笔点上方，只看墨迹就离开了它排的那一行
        base = float(span["origin"][1])
        return x0, min(top, base), x1, max(bottom, base)
    return x0, max(y0, top), x1, min(y1, bottom)


class _PageChars:
    """一页逐字的框（见 _read_char_boxes），第一次用到时才读。下划线、根号、重音等几步标记共用，
    同一页只读一次。"""

    def __init__(self, page: pymupdf.Page):
        self.page = page
        self._boxes: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] | None = None

    def boxes(self) -> dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]]:
        if self._boxes is None:
            self._boxes = _read_char_boxes(self.page)
        return self._boxes


# 下划线在基线下多深（字号的比例）。语料里的下划线在 0.11–0.3 之间；贴着基线的是删除线、
# 表格线，再往下是下一行的分数线、表格线
_UNDERLINE_TOP = 0.05
_UNDERLINE_DEPTH = 0.35


def _mark_underlines(page_chars: _PageChars, data: dict, strokes: Sequence[pymupdf.Rect]) -> None:
    """字下面画的线（opt_views 例 6 里标出的列 {A, D, G}）。线不在字体里，按细横线认：
    在字的基线下、降部以内，横向不出这一行。线下面的字切成单独的 span，标上 underline，
    送翻时写成 {u}。公式字下面的横线是分数线、根号线，不算。"""
    if not strokes:
        return
    lines = [line for block in data.get("blocks", []) for line in block.get("lines", []) if line.get("spans")]
    # 分数线下面紧挨着分母，下划线下面空着一段行距。量字形的墨迹框，不量字体框。
    # 下面那行自己也画着下划线的（一列链接）不是分母
    inks = [
        (id(span), pymupdf.Rect(span.get("ink") or span["bbox"]))
        for line in lines for span in line["spans"]
        if str(span.get("text") or "").strip()
        and not _underline_strokes(span, strokes, line["bbox"][0], line["bbox"][2])
    ]
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] | None = None
    seen: dict[tuple[float, float, float, float], bool] = {}

    def visible(rect: pymupdf.Rect) -> bool:
        key = (rect.x0, rect.y0, rect.x1, rect.y1)
        if key not in seen:
            seen[key] = _painted_stroke(page_chars.page, rect)
        return seen[key]

    for line in lines:
        spans = line["spans"]
        left = min(span["bbox"][0] for span in spans)
        right = max(span["bbox"][2] for span in spans)
        main = max(float(span.get("size") or 0) for span in spans)
        # 这一行正文的基线。分子抬在基线上面，下划线的字落在基线上
        body = [span for span in spans if float(span.get("size") or 0) >= main * 0.85 and str(span.get("text") or "").strip()]
        baseline = float(_by_chars(body, lambda span: round(span["origin"][1], 1))) if body else 0.0
        out: list[dict] = []
        for span in spans:
            size = float(span.get("size") or 10.0)
            raised = abs(span["origin"][1] - baseline) > size * 0.15
            under = [] if size < main * 0.85 or raised else [
                rect for rect in _underline_strokes(span, strokes, left, right)
                if not any(
                    other != id(span) and rect.y1 - 0.5 <= ink.y0 <= rect.y1 + size * 0.25
                    and min(ink.x1, rect.x1) - max(ink.x0, rect.x0) > 0.5
                    for other, ink in inks
                ) and visible(rect)
            ]
            if not under:
                out.append(span)
                continue
            if chars is None:
                chars = page_chars.boxes()
            out.extend(_split_covered(span, under, chars, _line_edges(spans, chars), "underline"))
        line["spans"] = out


# 根号。横线从根号右上角起笔
_RADICALS = "√∛∜"


def _mark_radicands(page_chars: _PageChars, data: dict, strokes: Sequence[pymupdf.Rect]) -> None:
    """根号横线下面的字是被开方数，标成公式（√81 的 81）。正文和公式用同一套字体时，
    按字体分不出来，81 留在正文里，根号单独成了一个公式，两边各排各的。
    CMEX 的大号根号提出来是字母 p，大号定界符字体里的字都按位置认：横线从它的右上角起头才是根号。"""
    lines = [line for block in data.get("blocks", []) for line in block.get("lines", []) if line.get("spans")]
    # 大号根号的字体框比字形高出一截，横线接在墨迹的右上角
    radicals = [
        pymupdf.Rect(span["ink"]) if span.get("ink") else _glyph_box(span)
        for line in lines for span in line["spans"]
        if str(span.get("text") or "").strip() and (
            str(span.get("text") or "").strip()[-1:] in _RADICALS
            or _BIG_DELIMITER_FONT_RE.search(str(span.get("font") or ""))
        )
    ]
    bars = [
        stroke for stroke in strokes
        if any(abs(stroke.x0 - radical.x1) <= 1.5 and radical.y0 - 1.5 <= stroke.y0 <= radical.y0 + 2.0 for radical in radicals)
    ]
    if not bars:
        return
    chars = page_chars.boxes()
    for line in lines:
        out: list[dict] = []
        for span in line["spans"]:
            size = float(span.get("size") or 10.0)
            base = float(span["origin"][1])
            box = pymupdf.Rect(span["bbox"])
            over = [] if _span_is_math(span) or not str(span.get("text") or "").strip() else [
                bar for bar in bars
                if base - size * 1.1 <= bar.y0 <= base - size * 0.5 and min(bar.x1, box.x1) - max(bar.x0, box.x0) > 0.5
            ]
            out.extend(_split_covered(span, over, chars, _line_edges(line["spans"], chars), "math") if over else [span])
        line["spans"] = out


# 叠在词上面的长箭头、横线最多盖住这么多个字：再长的是表格里的分隔线
_OVERLAY_CHARS = 12


def _mark_overlays(page_chars: _PageChars, data: dict) -> None:
    """用字形画在词上面的向量箭头（esvect 的 \vv{king} 提出来是「#    »」）和上横线：没有字母数字、
    墨迹是一条比一个字宽的细带，紧贴在别的词的墨迹上面。它和盖住的字是一个式子，标成公式，
    不然箭头当成字排进译文，「# » man」叠在一起。"""
    for block in data.get("blocks", []):
        spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
        overlays = []
        for span in spans:
            text = str(span.get("text") or "").strip()
            ink = span.get("ink")
            size = float(span.get("size") or 10.0)
            if not text or ink is None or any(char.isalnum() for char in text):
                continue
            box = pymupdf.Rect(ink)
            if box.width >= size * 0.8 and box.height <= size * 0.45:
                overlays.append((span, box, size))
        if not overlays:
            continue
        chars = page_chars.boxes()
        for overlay, box, size in overlays:
            covered: list[tuple[dict, pymupdf.Rect]] = []
            for span in spans:
                ink = span.get("ink")
                if span is overlay or ink is None or not any(char.isalpha() for char in str(span.get("text") or "")):
                    continue
                word = pymupdf.Rect(ink)
                if min(word.x1, box.x1) - max(word.x0, box.x0) <= 0:
                    continue
                # 按基线比：整个 span 的墨迹顶会被后面带升部的字（close 的 l）抬高。
                # 箭头自己的基线离词的基线不到一个字高；上一行的标点、省略号、填空横线隔着一整行
                base = float(span["origin"][1])
                lift = base - float(overlay["origin"][1])
                if 0 <= lift <= size * 0.8 and base - size * 1.3 <= box.y0 and box.y1 <= base - size * 0.45:
                    covered.append((span, word))
            if not covered:
                continue
            boxes = [
                (x0, x1) for span, _word in covered
                for x0, x1, _origin in chars.get((str(span["text"]), round(span["origin"][0], 1), round(span["origin"][1], 1))) or []
                if box.x0 <= (x0 + x1) / 2 <= box.x1
            ]
            if not boxes or len(boxes) > _OVERLAY_CHARS:
                continue
            under = pymupdf.Rect(min(x0 for x0, _ in boxes), box.y0, max(x1 for _, x1 in boxes), box.y1)
            overlay["math"] = True
            targets = {id(span) for span, _word in covered}
            for line in block.get("lines", []):
                out: list[dict] = []
                for span in line.get("spans", []):
                    if id(span) in targets:
                        out.extend(_split_covered(span, [under], chars, ([under.x0], [under.x1]), "math"))
                    else:
                        out.append(span)
                line["spans"] = out


# 公式检测模型的框常把公式后面的句读也框进去（$x$, 框成「x,」）。句读留在正文里，译文才排得成中文标点
_FORMULA_TRAIL = ",.;:?!"
# 框里只有数字（0、.5、(0, 1)）的排出来和正文一样，不当公式，免得同一句里 0 是占位符、1 不是
_NUMERIC_RE = re.compile(r"[\d\s.,()%+\-−]*")


def _mark_detected_formulas(page_chars: _PageChars, data: dict, rects: Sequence[pymupdf.Rect]) -> None:
    """公式检测模型框出的行内公式，框里的字按字切出来标上 detected_formula。
    只在认行内公式的 span 时算数（见 _is_formula_span），角色、跳过、拼段仍按字体定下的 math。"""
    if not rects:
        return
    chars = page_chars.boxes()
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            box = pymupdf.Rect(line["bbox"])
            here = [
                kept for rect in rects
                if min(rect.y1, box.y1) > max(rect.y0, box.y0) and min(rect.x1, box.x1) > max(rect.x0, box.x0)
                for kept in [_trim_formula_box(rect, spans, chars)] if kept is not None
            ]
            if not here:
                continue
            out: list[dict] = []
            for span in spans:
                hit = [] if _span_is_math(span) or not span["text"].strip() else [
                    rect for rect in here if _box_holds_span(rect, span)
                ]
                # 字边按框边对：框是从图像上来的，和字边差一点，不能按下划线那样要求对齐
                out.extend(
                    _split_covered(span, hit, chars, ([r.x0 for r in hit], [r.x1 for r in hit]), "detected_formula")
                    if hit else [span]
                )
            line["spans"] = out


def _box_holds_span(rect: pymupdf.Rect, span: dict) -> bool:
    """框和 span 左右相交，上下有一半以上重合：上下两行的框碰到这一行的边不算。"""
    box = pymupdf.Rect(span["bbox"])
    return (
        min(rect.y1, box.y1) - max(rect.y0, box.y0) >= 0.5 * min(box.height, rect.height)
        and min(rect.x1, box.x1) > max(rect.x0, box.x0)
    )


def _trim_formula_box(
    rect: pymupdf.Rect, spans: list[dict],
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]],
) -> pymupdf.Rect | None:
    """框落到这一行的字上：右端的句读让出去，只框着数字的不要。"""
    covered: list[tuple[float, str]] = []
    for span in spans:
        if not _box_holds_span(rect, span):
            continue
        boxes = chars.get((str(span["text"]), round(span["origin"][0], 1), round(span["origin"][1], 1))) or []
        covered += [
            (x0, char) for char, (x0, x1, _origin) in zip(str(span["text"]), boxes)
            if rect.x0 - 0.5 <= (x0 + x1) / 2 <= rect.x1 + 0.5
        ]
    covered.sort()
    trimmed = pymupdf.Rect(rect)
    while len(covered) > 1 and covered[-1][1] in _FORMULA_TRAIL:
        trimmed.x1 = covered.pop()[0]
    if covered and _NUMERIC_RE.fullmatch("".join(char for _x, char in covered)):
        return None
    return trimmed


# 单独画出来的重音字形（\hat、\bar、Montr\'eal 的 ´）。ASCII 的 ^ ~ ` 在正文、代码里另有用处，不算
_STACKED_ACCENTS = "ˆ˜¯´˙¨ˇ˘˚"
# 占自己宽度的 ˆ ˜ 是代码、正则、网址里的 ^ ~（r"^The"、nyu.edu/~roweis），等宽字体也画成这两个字形
_ASCII_ACCENTS = {"ˆ": "^", "˜": "~"}


def _split_accents(page_chars: _PageChars, data: dict) -> None:
    """重音和前面的字常在同一个 span 里（「(ˆ」「log ˆ」），按 span 认不出它盖在后面的字或公式上。
    切成单独的一截：盖在公式上的并进公式，盖在字母上的送翻前合成 ŷ、é。
    占自己宽度的 ˆ ˜ 不是重音，改回 ^ ~，送翻前不和后面的字母合成。"""
    lines = [line for block in data.get("blocks", []) for line in block.get("lines", []) if line.get("spans")]
    if not any(
        any(char in _STACKED_ACCENTS for char in text)
        for line in lines for text in (str(span.get("text") or "") for span in line["spans"])
    ):
        return
    chars = page_chars.boxes()
    for line in lines:
        out: list[dict] = []
        spans = line["spans"]
        for index, span in enumerate(spans):
            text = str(span.get("text") or "")
            boxes = chars.get((text, round(span["origin"][0], 1), round(span["origin"][1], 1)))
            if not any(char in _STACKED_ACCENTS for char in text) or boxes is None or len(boxes) != len(text):
                out.append(span)
                continue
            # 后一个字的左边：同一截里的下一个字，或者下一截的起头
            follow = [x0 for x0, _x1, _origin in boxes[1:]]
            follow.append(spans[index + 1]["bbox"][0] if index + 1 < len(spans) else float("inf"))
            # 叠在后一个字上的重音不占位，后一个字从它中间起笔
            stacked = [
                char in _STACKED_ACCENTS and after < x1 - (x1 - x0) * 0.3
                for char, (x0, x1, _origin), after in zip(text, boxes, follow)
            ]
            # 前一个字的右边：叠在前一个字上的（公式里的 sˆ）也是重音
            before = [spans[index - 1]["bbox"][2] if index else float("-inf")]
            before += [x1 for _x0, x1, _origin in boxes[:-1]]
            # 后面同一行有字、从它右边起笔，才能断定它占自己的宽度。单成一行的 ˆ 叠在别的行上
            spaced = "".join(
                _ASCII_ACCENTS.get(char, char)
                if not on and after != float("inf") and prior <= x0 + (x1 - x0) * 0.3 else char
                for char, on, (x0, x1, _origin), prior, after in zip(text, stacked, boxes, before, follow)
            )
            if spaced != text:
                text = spaced
                span = {**span, "text": text}
            if len(text.strip()) <= 1:
                out.append(span)
                continue
            cuts = sorted({
                cut for index, on in enumerate(stacked) if on for cut in (index, index + 1)
            } - {0, len(text)})
            for start, end in zip([0, *cuts], [*cuts, len(text)]):
                out.append(span if not cuts else {
                    **span, "text": text[start:end], "origin": boxes[start][2],
                    "bbox": (boxes[start][0], span["bbox"][1], boxes[end - 1][1], span["bbox"][3]),
                    **({"ink": (boxes[start][0], span["ink"][1], boxes[end - 1][1], span["ink"][3])} if "ink" in span else {}),
                })
        line["spans"] = out


VISIBLE_SPACE = "\u2423"


def _mark_visible_spaces(page_chars: _PageChars, data: dict, cups: Sequence[pymupdf.Rect]) -> None:
    """画出来的可见空格 ␣ 不在字体里，提取出来只剩一个普通空格或者一段空白，
    merged to ␣re 读成 merged to re。落在一行字的基线上、字的空隙里的，在原处写上 ␣：
    盖着空格的换掉这个空格，空隙里没有字的补一截。和前后的字隔着一段的，中间留一个空格。"""
    if not cups:
        return
    lines = [
        line for block in data.get("blocks", []) if block.get("type") == 0 for line in block.get("lines", [])
        if line.get("spans") and abs(line["dir"][1]) <= 0.01
    ]
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] | None = None
    for cup in cups:
        middle = (cup.x0 + cup.x1) / 2
        row = [
            (line, span) for line in lines for span in line["spans"]
            if abs(float(span["origin"][1]) - cup.y1) <= float(span.get("size") or 10.0) * 0.12
            and span["bbox"][1] < cup.y0
        ]
        covering = [(line, span) for line, span in row if span["bbox"][0] < middle < span["bbox"][2]]
        if covering:
            line, span = covering[0]
            text = str(span.get("text") or "")
            if chars is None:
                chars = page_chars.boxes()
            boxes = chars.get((text, round(span["origin"][0], 1), round(span["origin"][1], 1)))
            if boxes is None or len(boxes) != len(text):
                boxes = [(span["bbox"][0], span["bbox"][2], tuple(span["origin"]))] if len(text) == 1 else []
            hit = next((i for i, (x0, x1, _o) in enumerate(boxes) if x0 <= middle <= x1), None)
            if hit is None or not text[hit].isspace():
                continue
            x0, x1, _origin = boxes[hit]
            lead = " " if cup.x0 - x0 > 1.0 and (hit == 0 or not text[hit - 1].isspace()) else ""
            tail = " " if x1 - cup.x1 > 1.0 and hit + 1 < len(text) and not text[hit + 1].isspace() else ""
            span["text"] = text[:hit] + lead + VISIBLE_SPACE + tail + text[hit + 1:]
            continue
        # 空隙里没有字：接在离得近的那截上，左边的接在后面，右边的接在前面。两边一个字宽内都没有字的不算
        inked = [(line, span) for line, span in row if str(span.get("text") or "").strip()]
        before = [item for item in inked if item[1]["bbox"][2] <= middle]
        after = [item for item in inked if item[1]["bbox"][0] >= middle]
        left = max(before, key=lambda item: item[1]["bbox"][2], default=None)
        right = min(after, key=lambda item: item[1]["bbox"][0], default=None)
        size = float((left or right or (None, {}))[1].get("size") or 10.0)
        near_left = left is not None and cup.x0 - left[1]["bbox"][2] <= size
        near_right = right is not None and right[1]["bbox"][0] - cup.x1 <= size
        if not near_left and not near_right:
            continue
        gap_left = cup.x0 - left[1]["bbox"][2] if near_left else float("inf")
        gap_right = right[1]["bbox"][0] - cup.x1 if near_right else float("inf")
        to_left = gap_left <= gap_right
        line, neighbor = left if to_left else right
        spans = line["spans"]
        place = spans.index(neighbor) + (1 if to_left else 0)
        lead = " " if gap_left > 1.0 and (to_left or place > 0) else ""
        tail = " " if gap_right > 1.0 and (not to_left or place < len(spans)) else ""
        box = (cup.x0, neighbor["bbox"][1], cup.x1, neighbor["bbox"][3])
        spans.insert(place, {
            **neighbor, "text": lead + VISIBLE_SPACE + tail, "bbox": box,
            "origin": (cup.x0, neighbor["origin"][1]),
            **({"ink": (cup.x0, cup.y0, cup.x1, cup.y1)} if "ink" in neighbor else {}),
        })
        line["bbox"] = tuple(pymupdf.Rect(line["bbox"]) | pymupdf.Rect(box))


def _glyph_box(span: dict) -> pymupdf.Rect:
    """字形占的框。字体框量错、墨迹是单画量出来的，上下按墨迹算。"""
    box = pymupdf.Rect(span["bbox"])
    ink = span.get("ink")
    if ink and span.get("ink_drawn"):
        return pymupdf.Rect(box.x0, ink[1], box.x1, ink[3])
    return box


def _painted_stroke(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    """线真的画出来了：剪裁路径外面、被盖住的线也在 get_drawings 里。按线所在的一条细带渲染，
    几乎每一列都有深色才算。字的降部只穿过其中几列。"""
    band = pymupdf.Rect(rect.x0, rect.y0 - 0.4, rect.x1, rect.y1 + 0.4)
    try:
        pix = page.get_pixmap(clip=band, dpi=144, colorspace=pymupdf.csGRAY, alpha=False)
    except Exception:  # noqa: BLE001 - 渲染不了就不认
        return False
    if pix.width <= 0 or pix.height <= 0:
        return False
    samples = pix.samples
    dark = sum(
        1 for x in range(pix.width)
        if any(samples[y * pix.stride + x] < 200 for y in range(pix.height))
    )
    return dark >= pix.width * 0.8


def _underline_strokes(
    span: dict, strokes: Sequence[pymupdf.Rect], left: float, right: float,
) -> list[pymupdf.Rect]:
    text = str(span.get("text") or "")
    if not text.strip() or _span_is_math(span):
        return []
    size = float(span.get("size") or 10.0)
    base = float(span["origin"][1])
    box = pymupdf.Rect(span["bbox"])
    return [
        rect for rect in strokes
        if base + size * _UNDERLINE_TOP <= rect.y0 and rect.y1 <= base + size * _UNDERLINE_DEPTH
        and rect.x0 >= left - 1 and rect.x1 <= right + 1
        and min(rect.x1, box.x1) - max(rect.x0, box.x0) > 0.5
    ]


def _read_char_boxes(
    page: pymupdf.Page,
) -> dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]]:
    found: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]] = {}
    raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars") or []
                text = "".join(char["c"] for char in chars)
                key = (text, round(span["origin"][0], 1), round(span["origin"][1], 1))
                found[key] = [(char["bbox"][0], char["bbox"][2], tuple(char["origin"])) for char in chars]
    return found


def _line_edges(
    spans: list[dict], chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]],
) -> tuple[list[float], list[float]]:
    """一行里所有字的左边和右边。"""
    lefts: list[float] = []
    rights: list[float] = []
    for span in spans:
        boxes = chars.get((str(span["text"]), round(span["origin"][0], 1), round(span["origin"][1], 1))) or []
        lefts += [x0 for x0, _x1, _ox in boxes]
        rights += [x1 for _x0, x1, _ox in boxes]
    return lefts, rights


def _split_covered(
    span: dict, under: list[pymupdf.Rect],
    chars: dict[tuple[str, float, float], list[tuple[float, float, tuple[float, float]]]],
    edges: tuple[list[float], list[float]] = ([], []),
    flag: str = "underline",
) -> list[dict]:
    """span 按字切成线盖着的和线外的几截，盖着的标上 flag（下划线、根号里的数）。
    逐字的框对不上时，整个 span 都在线的范围里才标。"""
    text = str(span["text"])
    boxes = chars.get((text, round(span["origin"][0], 1), round(span["origin"][1], 1)))

    def covered(x0: float, x1: float) -> bool:
        middle = (x0 + x1) / 2
        return any(rect.x0 - 0.5 <= middle <= rect.x1 + 0.5 for rect in under)

    if boxes is None or len(boxes) != len(text):
        box = pymupdf.Rect(span["bbox"])
        if all(rect.x0 - 1 <= box.x0 for rect in under) and covered(box.x0, box.x0 + 0.1) and covered(box.x1 - 0.1, box.x1):
            return [{**span, flag: True}]
        return [span]
    # 下划线从某个字的左边画到某个字的右边。图里的虚线、图例线、分数线和字边对不上。
    # 一条线下面的字可能分成几个 span，按整行的字边对
    reach = max(1.0, float(span.get("size") or 10.0) * 0.15)
    lefts = edges[0] or [x0 for x0, _x1, _ox in boxes]
    rights = edges[1] or [x1 for _x0, x1, _ox in boxes]
    under = [
        rect for rect in under
        if any(abs(rect.x0 - x0) <= reach for x0 in lefts) and any(abs(rect.x1 - x1) <= reach for x1 in rights)
    ]
    # 下划线字符本身不算画了下划线（有的 PDF 把 _ 画成一条线）
    marks = [covered(x0, x1) and char != "_" for char, (x0, x1, _ox) in zip(text, boxes)]
    if not any(marks):
        return [span]
    pieces: list[dict] = []
    start = 0
    y0, y1 = span["bbox"][1], span["bbox"][3]
    for index in range(1, len(text) + 1):
        if index < len(text) and marks[index] == marks[start]:
            continue
        x0, x1 = boxes[start][0], boxes[index - 1][1]
        # 起笔点取这一截第一个字的。span 的起笔点是第一个字的：「 .」的空格落在前面下标的基线上，
        # 切下来的句点照抄它，擦除时按基线认不到，句点留在原处
        piece = {**span, "text": text[start:index], "bbox": (x0, y0, x1, y1), "origin": boxes[start][2]}
        # 页面上的 span 还是切开前的那一个，裁公式时按它认出哪些字是公式自己的（见 _outside_formula）
        piece["source"] = span.get("source") or tuple(span["bbox"])
        if "ink" in span:
            piece["ink"] = (x0, span["ink"][1], x1, span["ink"][3])
        if marks[start]:
            piece[flag] = True
        else:
            piece.pop(flag, None)
        pieces.append(piece)
        start = index
    return pieces


def _script_parts(line: dict, spans: list[dict]) -> list[tuple]:
    """西文页上一行里一截中文、空开半个字以上、接着一截西文的，是并排的两栏（左栏中文原句、右栏英译的题词），
    拆成两截：中文那截按外文留原文，西文接着译。不是这样的行返回空列表"""
    for k in range(1, len(spans)):
        if _script_seam(spans[:k], spans[k:]):
            parts = []
            for part in (spans[:k], spans[k:]):
                sub = dict(line, spans=part)
                parts.append(("".join(s["text"] for s in part), _line_box(sub), part, part))
            return parts
    return []


def _script_seam(left: list[dict], right: list[dict]) -> bool:
    """左边一截只有中文、右边一截西文起头，中间空开半个字以上（见 _script_parts）"""
    left = [span for span in left if str(span.get("text") or "").strip()]
    right = [span for span in right if str(span.get("text") or "").strip()]
    if not left or not right:
        return False
    text = "".join(str(span.get("text") or "") for span in left)
    size = float(right[0].get("size") or 10)
    return (bool(CJK_RE.search(text)) and not re.search(r"[A-Za-z]", text)
            and re.match(r"\s*[A-Za-z]", str(right[0].get("text") or "")) is not None
            and float(right[0]["bbox"][0]) - float(left[-1]["bbox"][2]) >= size * 0.5)


# Computer Modern 的正体。正文是 CM 时它们是正文；正文用 Times 之类时只出现在公式里
# （\mathrm、\log、括号、等号、\mathbf，无衬线粗体的向量 w、x）。
_CM_UPRIGHT_RE = re.compile(r"^(?:[A-Z]{6}\+)?(?:CMR|CMBX|CMSS(?:BX|I)?|CMSL)\d", re.I)


# 数学字母区（𝑥、𝜁、𝐀）。字体的 ToUnicode 把字母映到这里，这个字体就是写公式的
_MATH_ALPHABET_RE = re.compile(r"[\U0001D400-\U0001D7FF\u210E]")


def _mark_math_alphabet(pages: list[dict]) -> None:
    """字母大半映到数学字母区的字体，所有 span 都标上 math。字体名单认不全的数学斜体
    （newtxmath 配各种正文字体时的名字）照样认得出，同一字体里的逗号、括号也跟着算公式。"""
    letters: dict[str, int] = {}
    mathy: dict[str, int] = {}
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    font = str(span.get("font") or "")
                    text = str(span.get("text") or "")
                    letters[font] = letters.get(font, 0) + sum(char.isalpha() for char in text)
                    mathy[font] = mathy.get(font, 0) + len(_MATH_ALPHABET_RE.findall(text))
    fonts = {font for font, count in mathy.items() if count and count * 2 >= letters[font]}
    if not fonts:
        return
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span.get("font") in fonts:
                        span["math"] = True


def _mark_formula_only_fonts(pages: list[dict]) -> None:
    """全文只写一两个字、多半贴着公式的字体，所有 span 都标上 math。
    点阵的空心 1（𝟙）是 Type3 字体，名字只是 F74 这样的编号，提出来是普通的 1。"""
    spans_of: dict[str, int] = {}
    touching: dict[str, int] = {}
    longer: set[str] = set()
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                shown = [span for span in line["spans"] if str(span.get("text") or "").strip()]
                for index, span in enumerate(shown):
                    font = str(span.get("font") or "")
                    if _span_is_math(span):
                        continue
                    if len(span["text"].strip()) > 2:
                        longer.add(font)
                        continue
                    spans_of[font] = spans_of.get(font, 0) + 1
                    box = span["bbox"]
                    neighbours = shown[max(index - 1, 0):index] + shown[index + 1:index + 2]
                    if any(_span_is_math(other) and min(abs(other["bbox"][0] - box[2]), abs(box[0] - other["bbox"][2])) < 1.5
                           for other in neighbours):
                        touching[font] = touching.get(font, 0) + 1
    fonts = {font for font, count in spans_of.items() if font not in longer and touching.get(font, 0) * 2 >= count}
    if not fonts:
        return
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span.get("font") in fonts:
                        span["math"] = True


def _mark_math_roman(pages: list[dict]) -> None:
    """正文不是 Computer Modern 的文档里，CM 正体的 span 都是公式，标上 math。

    字体名单只能管到“这种字体通常是什么”。CMR 在一篇 CM 排的论文里是正文，在 Times 排的
    论文里只用来写 log、min、括号和等号，要看这份文档的正文用的是什么。
    """
    counts: dict[str, int] = {}
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    font = str(span.get("font") or "").split("+")[-1]
                    counts[font] = counts.get(font, 0) + len(span["text"].strip())
    if not counts:
        return
    body = max(counts, key=counts.__getitem__)
    if _CM_UPRIGHT_RE.match(body) or _font_is_math(body):
        return
    for data in pages:
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if _CM_UPRIGHT_RE.match(str(span.get("font") or "")):
                        span["math"] = True


def _prose_units(text: str) -> int:
    """正文词数：拉丁单词（≥2 字母）+ CJK 字符数 / 2。用来区分"公式行"和"含公式的句子"。"""
    latin = len(re.findall(r"[A-Za-z]{2,}", text))
    cjk = len(CJK_RE.findall(text))
    return latin + cjk // 2


# 分子的基线在分数线上方、分母的基线在下方多远（字号的比例）。分数线在数学轴上，
# 旁边的等号、标签的基线比它低一点
_NUMERATOR_RISE = (0.1, 0.75)
_DENOMINATOR_DROP = (0.4, 1.3)
_FRACTION_AXIS = 0.5
# 分子、分母居中的容差（线长的比例），旁边的字离线端多远以内算挨着（字号的比例）
_FRACTION_CENTER = 0.1
_FRACTION_NEIGHBOR = 1.5


def _fraction_parts(data: dict, rules: Sequence[pymupdf.Rect]) -> set[int]:
    """分子或分母里有正文词的分式（labeled recall = # of correct … / # of total …）上下的字。

    分数线是一条细横线，分子、分母各成一行，都落在线的长度以内、在线上居中，同一行里
    再没有挨着线端的字；线端外面在数学轴上有等号、标签之类。链接下划线下一行是接着写的
    正文，表格线两端外面没有字，都不是。纯数学的分式不在这里认：段落里的行内分式把一行
    拆成几截，按截数词认不准。
    """
    spans = [
        span for block in data.get("blocks", []) if block.get("type") == 0
        for line in block["lines"] if abs(line["dir"][1]) <= 0.01
        for span in line["spans"] if span["text"].strip()
    ]
    found: set[int] = set()
    for bar in rules:
        axis = (bar.y0 + bar.y1) / 2
        inside = [span for span in spans if bar.x0 - 1 <= span["bbox"][0] and span["bbox"][2] <= bar.x1 + 1]
        over = [span for span in inside if _within(axis - span["origin"][1], span["size"], _NUMERATOR_RISE)]
        under = [span for span in inside if _within(span["origin"][1] - axis, span["size"], _DENOMINATOR_DROP)]
        if not over or not under or max(_plain_units(over), _plain_units(under)) < 2:
            continue
        inner = {id(span) for span in inside}
        if not all(_centered_alone(row, spans, inner, bar) for row in (over, under)):
            continue
        if any(_beside_bar(span, bar, axis) for span in spans if id(span) not in inner):
            found.update(id(span) for span in (*over, *under))
    return found


def _plain_units(spans: list[dict]) -> int:
    return _prose_units(" ".join(span["text"] for span in spans if not _span_is_math(span)))


def _within(offset: float, size: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] * size <= offset <= bounds[1] * size


def _centered_alone(row: list[dict], spans: list[dict], inner: set[int], bar: pymupdf.Rect) -> bool:
    x0 = min(span["bbox"][0] for span in row)
    x1 = max(span["bbox"][2] for span in row)
    if abs((x0 + x1) / 2 - (bar.x0 + bar.x1) / 2) > max(1.5, bar.width * _FRACTION_CENTER):
        return False
    baselines = {round(span["origin"][1]) for span in row}
    for span in spans:
        if id(span) in inner or not any(abs(span["origin"][1] - y) <= 1 for y in baselines):
            continue
        reach = span["size"] * _FRACTION_NEIGHBOR
        if span["bbox"][2] > bar.x0 - reach and span["bbox"][0] < bar.x1 + reach:
            return False
    return True


def _beside_bar(span: dict, bar: pymupdf.Rect, axis: float) -> bool:
    """线端外面、数学轴附近的字：分式前的等号、标签，后面的运算符。"""
    if not 0 <= span["origin"][1] - axis <= span["size"] * _FRACTION_AXIS:
        return False
    reach = span["size"] * _FRACTION_NEIGHBOR
    return bar.x0 - reach <= span["bbox"][2] <= bar.x0 + 1 or bar.x1 - 1 <= span["bbox"][0] <= bar.x1 + reach


# 可伸长的大括号、大求和号这类定界符的字体。行内公式也会用到，所以只认离开左缘的块
_BIG_DELIMITER_FONT_RE = re.compile(r"CMEX|cmex|LMMathExtension", re.I)
