"""Turn text blocks into translation units: inline formulas become placeholders, section
numbers are peeled off, and each unit gets a note on where it sits in the book."""

import functools
import re
import struct
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import pymupdf

from .. import prompts
from .pdf_blocks import (
    _CAPTION_NUM_RE,
    _MARKUP_RE,
    _NUMBER_LABEL_RE,
    _PART_GAP,
    CJK_RE,
    TextBlock,
    _color_value,
    _ends_sentence,
    _garbled_label,
    _join_lines,
    _same_page_line_pair,
    _small_caps_sizes,
    _span_is_math,
    _visual_lines,
)
from .pdf_extract import (
    _INDEX_REF_RE,
    _ITALIC_FONT_RE,
    _OPERATOR_NAMES,
    _RELATION_RE,
    MONO_FONT_RE,
    _block_slice,
    _block_spans,
    _clip_key,
    _extract,
    _is_formula_char,
    _mono_span,
    _small_band,
    _text_box,
    _vivid,
)
from .pdf_flow import (
    Emphasis,
    span_emphasis,
    strip_style_marks,
    writer_for,
)
from .pdf_frame import TextFrame
from .pdf_layout import (
    PageGeometry,
    example_numbers,
    interlinear_glosses,
    margin_skips,
)
from .pdf_math import _glyph_box, _plain_units, _prose_units
from .pdf_merge import _all_styled, _cross_page_units, _region_joins, _seam_apart
from .pdf_roles import LayoutItem, Role
from .pdf_runs import (
    BOUNDARY,
    FormulaRun,
    Run,
    TextRun,
)
from .pdf_skip import (
    _arc_figure_ids,
    _code_panel_ids,
    _color_keyed_ids,
    _concordance_ids,
    _display_example_ids,
    _display_formula_ids,
    _example_ids,
    _foreign_ids,
    _gloss_ids,
    _in_gloss,
    _isolated_formula_ids,
    _item_example_ids,
    _median_size,
    _role_pattern_ids,
    biblio_skips,
)
from .pdf_vision import FormulaModel, ModelBoxes, formula_detector, layout_detector


def _is_symbol_span(text: str) -> bool:
    """正文字体里整段都是希腊字母或数学符号。纯重音留给重音吸收，不当成公式。"""
    stripped = text.strip()
    if not stripped or all(unicodedata.category(c) in ("Sk", "Mn") for c in stripped):
        return False
    return all(_is_formula_char(c) for c in stripped)


def _smaller_prose_line(spans: list[dict], main_size: float) -> bool:
    """这一行整体都比块里最大的标题小，而且是完整句子。那是标题下的正文，不是角标。"""
    visible = [span for span in spans if str(span.get("text") or "").strip()]
    if not visible or main_size <= 0:
        return False
    if max(float(span.get("size") or 0) for span in visible) >= main_size * 0.9:
        return False
    return _prose_units("".join(str(span.get("text") or "") for span in visible)) >= 4


def _light_text(span: dict) -> bool:
    color = int(span.get("color") or 0)
    red, green, blue = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return bool(span["text"].strip()) and 0.299 * red + 0.587 * green + 0.114 * blue >= 220


def _is_formula_span(span: dict, main_size: float, *, prose_line: bool = False, code: bool = False) -> bool:
    """行内公式：数学字体的 span、公式检测模型框出的字，或明显小于正文的角标 span（引用编号等）。
    粗体 CM（CMBX 等）的一两个字符按 mathbf 单字母算公式；粗体单词仍是正文。
    正文字体里的希腊字母和数学符号也是公式。单个拉丁字母不是，除非贴在角标上。
    prose_line 为真时，整行都是比标题小的正文，不能单凭字号收成公式。"""
    if _span_is_math(span) or span.get("detected_formula"):
        return True
    # 近白的字压在深色底上（黑底白字的图注标签）。重写会用正文的颜色，和底色混成一片，原样贴回
    if _light_text(span):
        return True
    # 句子里的等宽词是属性名、函数名，照原样贴回，不让模型有的译有的不译
    if code and MONO_FONT_RE.search(span["font"]) and re.search(r"[A-Za-z]", span["text"]):
        return True
    if span["size"] < main_size * 0.79 and not prose_line:
        return True
    if re.match(r"^CMB", span["font"]) and len(span["text"].strip()) <= 2:
        # 图注里的 of / to 也是两个字母，但是词，不是 mathbf 变量。
        token = span["text"].strip()
        if len(token) == 2 and token.isalpha() and token.islower():
            return False
        return True
    return _is_symbol_span(span["text"])


def _formula_markup(run_spans: list[dict], main_size: float) -> str:
    """把公式 run 拼成带上下标哨兵的字符串（\\x01s\\x02..\\x01/s\\x02 上标，b 为下标）。
    恢复进译文后由渲染层换成 <sup>/<sub>，y_1^T 不再被拍平成 y1T。
    上/下标按 baseline 偏移判定：比正文 baseline 高的是上标，低的是下标。"""
    text_spans = [s for s in run_spans if s["size"] >= main_size * 0.79]
    origins = [s["origin"][1] for s in text_spans if s.get("origin")]
    main_origin = origins[0] if origins else None
    if main_origin is None:
        return "".join(s["text"] for s in run_spans)
    out, role = [], ""
    for s in run_spans:
        r = ""
        if s["size"] < main_size * 0.79 and s.get("origin"):
            dy = s["origin"][1] - main_origin
            if dy < -1:
                r = "s"
            elif dy > 1:
                r = "b"
        if r != role:
            if role:
                out.append(f"\x01/{role}\x02")
            if r:
                out.append(f"\x01{r}\x02")
            role = r
        out.append(s["text"])
    if role:
        out.append(f"\x01/{role}\x02")
    return "".join(out)


# 孤立标点不当公式：<EOS> 这类 token 的尖括号是数学字体，单独抽出来会把 token 拆散、
# 还把上下行的括号粘成跨行怪图。括号/标点留在正文里排版（正文字体本来就画得出）。
# 符号字体里没有正文标点：CMSY 的 < 是 ≺ 这类关系符，单独一个也是公式
_SYMBOL_FONT_RE = re.compile(r"^CMSY|^MSAM|^MSBM", re.I)
_LONE_PUNCT_RE = re.compile(r"^[\s<>[\]{}()|/\\=+\-_.,;:'\"~·，。；：！？（）【】]+$")
# 贴在角标上的正体变量。两个字母会把 of/to 吸进上标，只收一个字母。
_VARIABLE_BASE_RE = re.compile(r"^[A-Za-z]['′]?$")


# 后面的数字算进公式的运算符
_ASSIGN_OPS = {"=", "<", ">", "+", "-", "−"}


def _letter_formula_runs(span_lines: list[list[dict]], runs: list[dict]) -> list[dict]:
    """正文字体的斜体单字母，夹着按全文字体标成公式的括号、等号（P(y = 1)、f(x)）：
    P、y 是正文的斜体，括号、等号各自是孤立标点，单看哪个都不成公式。同一行上连着的一串里
    既有标成公式的标点、又有斜体单字母的，整串是行内公式。中间的数字是公式里的，首尾的不收。
    等号后面的数字也收（y = 1）。公式的括号里的斜体词、已经认出的符号也在串里（P(to|want) 的 to、|、want），
    被整串盖住的公式并进来。返回新的整串，盖住的公式从 runs 里拿掉"""
    taken = {id(span) for run in runs for span in run["spans"]}

    def math_mark(span: dict) -> bool:
        return bool(span.get("math")) and _LONE_PUNCT_RE.match(span["text"].strip() or " ") is not None

    def italic(span: dict) -> bool:
        return bool(int(span.get("flags") or 0) & 2)

    def letter(span: dict) -> bool:
        return italic(span) and _VARIABLE_BASE_RE.match(span["text"].strip()) is not None

    def edge(span: dict) -> bool:
        return math_mark(span) or letter(span) or id(span) in taken

    def assigned(window: list[dict]) -> bool:
        """串尾是数字，前面紧挨着的是标成公式的运算符（= 1、< 0）"""
        words = [span for span in window if span["text"].strip()]
        return (
            len(words) >= 2
            and words[-1]["text"].strip().isdigit()
            and math_mark(words[-2])
            and words[-2]["text"].strip() in _ASSIGN_OPS
        )

    found: list[dict] = []
    for line_no, spans in enumerate(span_lines):
        i, n = 0, len(spans)
        while i < n:
            if not (math_mark(spans[i]) or letter(spans[i])):
                i += 1
                continue
            depth = 0
            j = i
            while True:
                text = spans[j]["text"]
                if math_mark(spans[j]) or id(spans[j]) in taken:
                    depth = max(0, depth + text.count("(") - text.count(")"))
                if j + 1 >= n:
                    break
                nxt = spans[j + 1]
                word = nxt["text"].strip()
                if not (not word or edge(nxt) or word.isdigit() or (depth > 0 and italic(nxt) and word.isalpha())):
                    break
                j += 1
            while j > i and not edge(spans[j]) and not assigned(spans[i : j + 1]):
                j -= 1
            window = spans[i : j + 1]
            if any(math_mark(span) for span in window) and any(letter(span) for span in window):
                bbox = _glyph_box(window[0])
                for span in window[1:]:
                    bbox |= _glyph_box(span)
                found.append(
                    {
                        "spans": window,
                        "bbox": bbox,
                        "max_size": max(span["size"] for span in window),
                        "pos": (line_no, i, j),
                    }
                )
            i = j + 1
    covered = {id(span) for run in found for span in run["spans"]}
    runs[:] = [run for run in runs if not all(id(span) in covered for span in run["spans"] if span["text"].strip())]
    return found


def _script_base(
    span: dict,
    bbox: pymupdf.Rect,
    block_size: float,
    math: bool = False,
    script_y: float | None = None,
) -> bool:
    """单字母贴在较小角标上，才并进公式。引用上标在字母右侧，中心出了字框，不收；
    数学字体的角标（inf、tr 这样宽的）不是引用，只要紧贴字母右缘。
    script_y 是角标的基线，下标比底低不过半个字。
    下一行 H^enc 的 enc 字框碰到上一行的 X，基线却低了大半行，不是 X 的下标。"""
    if not _VARIABLE_BASE_RE.match(str(span.get("text") or "").strip()):
        return False
    if float(span.get("size") or 0) < block_size * 0.79:
        return False
    raw = span.get("bbox")
    if raw is None:
        return False
    base = pymupdf.Rect(raw)
    y_gap = max(base.y0, bbox.y0) - min(base.y1, bbox.y1)
    center = (bbox.x0 + bbox.x1) / 2
    # 角标可以略伸出字母右缘。再往右就是引用上标，中心已经离开字框。
    # 同一行上的底和角标字框叠着或贴着；上下隔开几磅的是上一行的字母（w_end 的 w 不是下一行 mention 的底）。
    # 整个在字母左边的不是它的角标：行内分式的分子和同一行后面的 x 上下交叠，中间隔着正文
    if y_gap >= block_size * 0.3 or bbox.x0 > base.x1 + 0.4 or bbox.x1 <= base.x0:
        return False
    if script_y is not None and span.get("origin"):
        drop = script_y - float(span["origin"][1])
        if drop > block_size * 0.5:
            return False
    return math or base.x0 - 1 <= center <= base.x1 + 2.0


def _has_text_base(run: dict, span_lines: list[list[dict]], members: set[int], block_size: float) -> bool:
    """只有小字的 run 左边紧贴着同一行一个正文字号、不在公式里的字：那个字是它的底。"""
    if run["max_size"] >= block_size * 0.79:
        return False
    box = run["bbox"]
    for line in span_lines:
        for span in line:
            if (
                id(span) in members
                or not str(span.get("text") or "").strip()
                or float(span.get("size") or 0) < block_size * 0.79
            ):
                continue
            base = _glyph_box(span)
            overlap = min(base.y1, box.y1) - max(base.y0, box.y0)
            if -0.5 <= box.x0 - base.x1 <= block_size * 0.25 and overlap > box.height * 0.5:
                return True
    return False


def _absorb_script_bases(runs: list[dict], span_lines: list[list[dict]], block_size: float) -> None:
    """把贴在角标上的正体变量并进该角标。单词和引用编号留在正文。
    只看 run 开头的角标：N 的上下标后面接着箭头、等号时，字母仍是它们的底。"""
    taken = {id(span) for run in runs for span in run["spans"]}
    for run in runs:
        scripts = pymupdf.Rect()
        math = False
        script_y: float | None = None
        for span in run["spans"]:
            if not str(span.get("text") or "").strip():
                continue
            if float(span.get("size") or 0) >= block_size * 0.79:
                break
            scripts |= pymupdf.Rect(span["bbox"])
            math = math or _span_is_math(span)
            if script_y is None and span.get("origin"):
                script_y = float(span["origin"][1])
        if scripts.is_empty:
            continue
        line_no, start, end = run["pos"]
        candidates: list[tuple[int, int, dict]] = []
        if start > 0:
            candidates.append((line_no, start - 1, span_lines[line_no][start - 1]))
        if line_no > 0:
            for index, span in enumerate(span_lines[line_no - 1]):
                candidates.append((line_no - 1, index, span))
        for other_no, index, span in candidates:
            if id(span) in taken or not _script_base(span, scripts, block_size, math, script_y):
                continue
            run["spans"].insert(0, span)
            run["bbox"] |= pymupdf.Rect(span["bbox"])
            # 有了正文字号的底就不再是单独的角标，不能再按拆开的上下标并到别的式子上
            run["max_size"] = max(float(run["max_size"]), float(span.get("size") or 0))
            if other_no == line_no:
                run["pos"] = (line_no, index, end)
            taken.add(id(span))
            break


# 数学罗马体（CMR、Latin Modern 的 LMRoman）在黑名单里，不当公式字体。贴着公式的数字、括号、
# 等号加号这类算子、算子名仍是式子的一部分
_CM_ROMAN_RE = re.compile(r"CMR\d|LMRoman\d")
_GLUE_ATOM = rf"(?:[\d.()\[\]{{}}|=+<>≤≥×]+|(?:{_OPERATOR_NAMES}))"
_MATH_GLUE_RE = re.compile(rf"^(?:{_GLUE_ATOM})(?:\s+{_GLUE_ATOM})*$")
_ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}


def _absorb_inner_blanks(runs: list[dict], span_lines: list[list[dict]]) -> None:
    """同一行上夹在一个公式两个成员之间的空白 span 并进公式。几段并成一个以后，
    中间的空格不属于哪一段，留在正文里就成了占位符后面多出来的一个空格。"""
    owner = {id(span): index for index, run in enumerate(runs) for span in run["spans"]}
    for line in span_lines:
        shown = [index for index, span in enumerate(line) if str(span.get("text") or "").strip()]
        for left, right in zip(shown, shown[1:]):
            # 两边最近的字都在同一个公式里，中间夹着的空白才是它的
            run = owner.get(id(line[left]))
            if run is None or owner.get(id(line[right])) != run:
                continue
            for span in line[left + 1 : right]:
                if id(span) not in owner:
                    runs[run]["spans"].append(span)
                    owner[id(span)] = run


_ITALIC_GLUE_RE = re.compile(r"^[\s()\[\]{}.,=<>+\-*/'A-Za-z0-9]+$")
_ITALIC_OPERATOR_RE = re.compile(r"[()\[\]{}=<>+\-*/]")


def _is_math_glue(span: dict) -> bool:
    """CMR 上的数字、括号、算子或算子名；正文斜体写的变量、括号、算子。正文正体里的同形字不收。
    数学字体的 span 只收单个标点：别的没进公式的是有意放回正文的（序数 t-th 的 t）。"""
    font = str(span.get("font") or "")
    text = str(span.get("text") or "").strip()
    if _CM_ROMAN_RE.search(font):
        return bool(_MATH_GLUE_RE.match(text))
    if _span_is_math(span):
        # 单独成段时当标点放回正文的 >、小数点，贴着公式就是式子里的（> = 10、0.31）
        return bool(text) and bool(_LONE_PUNCT_RE.match(text))
    if not (_ITALIC_FONT_RE.search(font) or int(span.get("flags") or 0) & 2):
        return False
    if not text or not _ITALIC_GLUE_RE.match(text):
        return False
    operator = bool(_ITALIC_OPERATOR_RE.search(text))
    for word in re.findall(r"[A-Za-z]+", text):
        if len(word) == 1 or (len(word) == 2 and word.isupper()):
            continue
        if len(word) == 2 and operator:
            continue
        return False
    return operator or bool(re.search(r"[A-Za-z0-9]", text))


def _attach_span(run: dict, span: dict, line_no: int, index: int) -> None:
    run["spans"].append(span)
    run["bbox"] |= pymupdf.Rect(span["bbox"])
    run["max_size"] = max(float(run["max_size"]), float(span.get("size") or 0))
    cur_line, start, end = run["pos"]
    if line_no == cur_line:
        run["pos"] = (cur_line, min(start, index), max(end, index))


def _gap_is_clear(
    span: dict,
    run: dict,
    span_lines: list[list[dict]],
    taken: set[int],
) -> bool:
    """两框之间不能隔着还没收进公式的字。空白可以。"""
    box = pymupdf.Rect(span["bbox"])
    other = run["bbox"]
    if box.x0 <= other.x0:
        left, right = box.x1, other.x0
    else:
        left, right = other.x1, box.x0
    if right - left <= 0.4:
        return True
    for line in span_lines:
        for item in line:
            if id(item) in taken or id(item) == id(span) or not str(item.get("text") or "").strip():
                continue
            item_box = pymupdf.Rect(item["bbox"])
            if item_box.x1 <= left + 0.2 or item_box.x0 >= right - 0.2:
                continue
            y_overlap = min(box.y1, item_box.y1) - max(box.y0, item_box.y0)
            if y_overlap > 0:
                return False
    return True


def _glue_host(
    span: dict,
    runs: list[dict],
    span_lines: list[list[dict]],
    block_size: float,
    taken: set[int],
) -> dict | None:
    box = pymupdf.Rect(span["bbox"])
    # 纵向按字高比：CMSY 的减号、竖线的字体框伸到下一行，下一行行首的 P( 会挂到上一行的式子上
    text = _text_box(span)
    best: tuple[float, dict] | None = None
    for run in runs:
        other = run["bbox"]
        ink = _run_text_box(run)
        y_overlap = min(text.y1, ink.y1) - max(text.y0, ink.y0)
        if y_overlap <= 0.3 * min(text.height, ink.height):
            continue
        gap = box.x0 - other.x1 if box.x0 >= other.x0 else other.x0 - box.x1
        if gap < 0:
            gap = 0.0
        if gap >= block_size * 0.8 or not _gap_is_clear(span, run, span_lines, taken):
            continue
        if best is None or gap < best[0]:
            best = (gap, run)
    return best[1] if best else None


def _blanks_between(
    span: dict,
    run: dict,
    span_lines: list[list[dict]],
    taken: set[int],
) -> list[tuple[int, int, dict]]:
    """公式和这段胶水之间的空白。收进公式后，送翻文本里不再留一个空档。"""
    box = pymupdf.Rect(span["bbox"])
    other = run["bbox"]
    if box.x0 <= other.x0:
        left, right = box.x1, other.x0
    else:
        left, right = other.x1, box.x0
    if right - left <= 0.4:
        return []
    found: list[tuple[int, int, dict]] = []
    for line_no, line in enumerate(span_lines):
        for index, item in enumerate(line):
            if id(item) in taken or str(item.get("text") or "").strip():
                continue
            item_box = pymupdf.Rect(item["bbox"])
            if item_box.x0 < left - 0.4 or item_box.x1 > right + 0.4:
                continue
            y_overlap = min(box.y1, item_box.y1) - max(box.y0, item_box.y0)
            if y_overlap > 0:
                found.append((line_no, index, item))
    return found


def _absorb_math_glue(
    runs: list[dict],
    span_lines: list[list[dict]],
    block_size: float,
) -> None:
    """把贴着公式的 CMR 数字、括号、算子名并进公式。远处的数字仍是正文。"""
    taken = {id(span) for run in runs for span in run["spans"]}
    pending = [
        (line_no, index, span)
        for line_no, line in enumerate(span_lines)
        for index, span in enumerate(line)
        if id(span) not in taken
        and _is_math_glue(span)
        # 斜体变量后面贴着下标、正文是常事（∂L 接 CE），只有正体的数字、算子、标点怕拆开一个数
        and not (
            (_CM_ROMAN_RE.search(str(span.get("font") or "")) or _span_is_math(span))
            and _runs_into_word(line, index, taken)
        )
    ]
    while pending:
        attached = False
        still: list[tuple[int, int, dict]] = []
        for line_no, index, span in pending:
            host = _glue_host(span, runs, span_lines, block_size, taken)
            if host is None:
                still.append((line_no, index, span))
                continue
            for blank_line, blank_index, blank in _blanks_between(span, host, span_lines, taken):
                _attach_span(host, blank, blank_line, blank_index)
                taken.add(id(blank))
            _attach_span(host, span, line_no, index)
            taken.add(id(span))
            attached = True
        if not attached:
            return
        pending = still


def _runs_into_word(line: list[dict], index: int, taken: set[int]) -> bool:
    """span 后面没隔空格就接着正文的字母、数字：B/T = 0.31 的「= 0」后面是「.」和「31 likely」。
    收进公式会把一个数拆成两半，这一串都留在正文。中间也能收的 span 跳过去接着看。"""
    for item in line[index + 1 :]:
        text = str(item.get("text") or "")
        if id(item) in taken or not text or text[0].isspace():
            return False
        if _is_math_glue(item):
            if text[-1].isspace():
                return False
            continue
        word = text.split()[0]
        if re.search(r"[A-Za-z0-9]", word):
            return True
        if len(word) < len(text):
            return False
    return False


def _ordinal_reading(run: dict, block_size: float) -> str | None:
    """i^{th}、1^{st} 这类序数。上标不是 st/nd/rd/th 的角标不是序数。"""
    base: list[str] = []
    suffix: list[str] = []
    for span in run["spans"]:
        text = str(span.get("text") or "").strip()
        if not text:
            continue
        if float(span.get("size") or 0) < block_size * 0.79:
            suffix.append(text)
        else:
            base.append(text)
    tail = "".join(suffix).lower()
    head = "".join(base)
    if tail not in _ORDINAL_SUFFIXES or not re.fullmatch(r"[A-Za-z]|\d{1,3}", head):
        return None
    return f"{head}-{tail}"


def _release_ordinals(runs: list[dict], block_size: float) -> tuple[dict[int, str], set[int]]:
    """序数不做成公式图，改成 i-th 送翻，模型才能译成「第 i 个」。"""
    kept: list[dict] = []
    ordinal_at: dict[int, str] = {}
    ordinal_skip: set[int] = set()
    for run in runs:
        reading = _ordinal_reading(run, block_size)
        if reading is None:
            kept.append(run)
            continue
        anchor = next(
            (
                span
                for span in run["spans"]
                if str(span.get("text") or "").strip() and float(span.get("size") or 0) >= block_size * 0.79
            ),
            None,
        )
        if anchor is None:
            kept.append(run)
            continue
        ordinal_at[id(anchor)] = reading
        for span in run["spans"]:
            if span is not anchor and str(span.get("text") or "").strip():
                ordinal_skip.add(id(span))
    runs[:] = kept
    return ordinal_at, ordinal_skip


# 重音符号（hat/tilde/bar/dot 等）：落在公式 bbox 上的要并进公式，不能留在正文
_ACCENT_CHARS = {"^", "ˆ", "̂", "~", "˜", "̃", "¯", "̄", "´", "`", "˙", "̇", "¨", "̈", "ˇ", "̌", "˘", "̆", "˚", "̊", "⃗"}


def _placeholderize(block: TextBlock, block_index: int = 0) -> tuple[str, list[dict]]:
    """把块里的行内公式换成 {vN} 占位符（BabelDOC 的翻译协议），返回 (送翻文本, 公式记录列表)。

    每条公式记录：bbox（原页裁剪区域）、name、w/h/d（落位几何）、text（无图环境回退）。
    渲染时以 show_pdf_page 矢量透传，视觉和原版一致。
    关键归并：上下标常被 PyMuPDF 拆到另一个 dict 行，按行收集的 run 会把一个公式
    拆成互相交叠的两段（画两遍、上标错位），所以先全块收集 run，再把 x 区间交叠的
    归并成一个。没有公式时返回的文本与原文逐字一致，缓存照常命中。
    """
    if not block.span_lines:
        return block.text, []

    # 1) 按 dict 行收集公式 run，并记录在块 span 流里的位置（供阅读序邻接判断）
    runs: list[dict] = []
    code = not _code_block(block)
    for line_no, spans in enumerate(block.span_lines):
        prose_line = _smaller_prose_line(spans, block.size)
        i, n = 0, len(spans)
        while i < n:
            if _is_formula_span(spans[i], block.size, prose_line=prose_line, code=code):
                run_spans = [spans[i]]
                j = i + 1
                while j < n and (
                    not spans[j]["text"].strip()
                    or _is_formula_span(spans[j], block.size, prose_line=prose_line, code=code)
                ):
                    run_spans.append(spans[j])
                    j += 1
                while run_spans and not run_spans[-1]["text"].strip():
                    run_spans.pop()
                if run_spans and any(s["text"].strip() for s in run_spans):
                    # 孤立标点 run 不当公式（<EOS> 的尖括号、括号、等号），留在正文里
                    joined = "".join(s["text"] for s in run_spans).strip()
                    if not _LONE_PUNCT_RE.match(joined) or any(_SYMBOL_FONT_RE.search(s["font"]) for s in run_spans):
                        bbox = _glyph_box(run_spans[0])
                        for s in run_spans[1:]:
                            bbox |= _glyph_box(s)
                        runs.append(
                            {
                                "spans": run_spans,
                                "bbox": bbox,
                                "max_size": max(s["size"] for s in run_spans),
                                "pos": (line_no, i, j - 1),
                            }
                        )
                i = j
            else:
                i += 1
    runs.extend(_letter_formula_runs(block.span_lines, runs))
    _absorb_script_bases(runs, block.span_lines, block.size)
    ordinal_at, ordinal_skip = _release_ordinals(runs, block.size)
    _absorb_math_glue(runs, block.span_lines, block.size)

    # 2) 归并 run，三种情形：
    # a) 上下标拆到不同 dict 行：x 交叠够深（≥30% 较窄 run 的宽）且必有一方是小字号
    #    （<0.79 正文）；两个正文字号 run 即使 x 深度交叠也是相邻文本行上碰巧对齐
    #    的两个公式（如行末 (X,Y) 和下一行的 y_1,…,y_T）。
    # b) 阅读序上真正邻接（中间只有空白 span）且 x 间距小于一个字号、纵向相交
    #    （同一可视行）：同一数学表达式被拆成的连续片段（y^i_1, y^i_2, … 的元素），
    #    并回一个公式，否则下标会被占位符间的空格推到离基底很远的位置。
    #    行间隔着正文的不并（(X,Y) vs y_1,…,y_T）。跨行折行的两半不在这里并 bbox。
    # c) 行内分式的分子分母：上下贴着、左右居中（见 _fraction_pair），阅读序上不必相邻。
    member_ids = {id(s) for r in runs for s in r["spans"]}

    def flow_adjacent(a: dict, b: dict) -> bool:
        l1, _, e1 = a["pos"]
        l2, s2, _ = b["pos"]
        for ln in range(l1, l2 + 1):
            line = block.span_lines[ln]
            lo = e1 + 1 if ln == l1 else 0
            hi = s2 if ln == l2 else len(line)
            for s in line[lo:hi]:
                if s["text"].strip() and id(s) not in member_ids:
                    return False
        return True

    runs.sort(key=lambda r: r["pos"])

    # 只有小字的 run 紧贴在本行一个正文的字后面，是那个字的角标（H^enc 的 H 是正文的粗体）。
    # 下一行行首的 H^enc 正好在上一行 X 的下面，不能按 x 交叠并成 X 的下标
    based = {id(r) for r in runs if _has_text_base(r, block.span_lines, member_ids, block.size)}

    def geometric(m: dict, r: dict) -> bool:
        if any(id(x) in based and x["max_size"] < block.size * 0.79 for x in (m, r)):
            return False
        x_overlap = min(m["bbox"].x1, r["bbox"].x1) - max(m["bbox"].x0, r["bbox"].x0)
        y_gap = max(m["bbox"].y0, r["bbox"].y0) - min(m["bbox"].y1, r["bbox"].y1)
        min_w = min(m["bbox"].width, r["bbox"].width)
        lo, hi = sorted([m["max_size"], r["max_size"]])
        return x_overlap > max(1.0, 0.3 * min_w) and y_gap < block.size and lo < hi * 0.79

    def adjacent(m: dict, r: dict) -> bool:
        # b) 同一可视行上、阅读序邻接、x 间距小于一个字号。跨行折行的两半
        #    纵向不相交，不在这里并成一个大框。
        y_overlap = min(m["bbox"].y1, r["bbox"].y1) - max(m["bbox"].y0, r["bbox"].y0)
        # 前一段收住一行，后一段整个在它左下方：折到了下一行行首。
        # 分子右边还有式子，分母的根号也在左下方，不算折行
        # 数学字体的字框很高，上下两行的框会交叠，再按基线比：低了半行以上就是下一行
        wrapped = (
            r["bbox"].x1 <= m["bbox"].x0
            and (_run_text_box(r).y0 >= _run_text_box(m).y1 or _run_baseline(r) - _run_baseline(m) > block.size * 0.6)
            and _ends_row(m, block.span_lines)
        )
        return (
            flow_adjacent(m, r)
            and r["bbox"].x0 - m["bbox"].x1 < block.size
            and not wrapped
            and y_overlap > 0.3 * min(m["bbox"].height, r["bbox"].height)
        )

    merged: list[dict] = []
    # 刚收过一段的 run。分子并进前面的分母以后，分子的下标要接着找到它
    recent: dict | None = None
    for r in runs:
        target = next((m for m in (recent, merged[-1] if merged else None) if m is not None and geometric(m, r)), None)
        if target is None:
            target = next((m for m in reversed(merged) if _fraction_pair(m, r, block.size)), None)
        if target is None and merged and adjacent(merged[-1], r):
            target = merged[-1]
        if target is None:
            merged.append(r)
            recent = r
            continue
        target["spans"].extend(r["spans"])
        target["bbox"] |= r["bbox"]
        target["max_size"] = max(target["max_size"], r["max_size"])
        m_line, m_start, m_end = target["pos"]
        r_line, r_start, r_end = r["pos"]
        if m_line == r_line:
            target["pos"] = (m_line, min(m_start, r_start), max(m_end, r_end))
        recent = target

    _absorb_inner_blanks(merged, block.span_lines)

    # 2.5) 吸收落在公式 bbox 上的重音符 span（ŷ 的 ^、x̄ 的 ¯ 等）：
    # 它们常是不被判定为公式的独立 span，留在正文里会在公式图边上多出一个孤符号；
    # 并进 run 后 bbox 上移把符头顶部也包进裁剪，正文里删掉
    for spans in block.span_lines:
        for s in spans:
            if s["text"].strip() not in _ACCENT_CHARS:
                continue
            sb = pymupdf.Rect(s["bbox"])
            # 重音盖在公式里同一行的某一截字上：横向交叠，基线相近（大写字母上的帽子抬高一点）。
            # 按公式框比，上一行 naïve 的两点会算到下一行的公式上
            base = float(s["origin"][1]) if s.get("origin") else (sb.y0 + sb.y1) / 2

            def covers(span: dict) -> bool:
                box = pymupdf.Rect(span["bbox"])
                raised = float(span["origin"][1]) - base
                return min(box.x1, sb.x1) - max(box.x0, sb.x0) > 0 and -0.2 <= raised / block.size <= 0.6

            # 重音自己可能单独成了一个式子（CMR10 的 ˆ 单成一行）。并进它盖着的式子以后，
            # 原来那个就不再单列，不然送翻文本里多出一个只有 ˆ 的占位符
            owner = next((m for m in merged if any(s is x for x in m["spans"])), None)
            if owner is not None and any(
                str(x.get("text") or "").strip() not in _ACCENT_CHARS
                for x in owner["spans"]
                if str(x.get("text") or "").strip()
            ):
                continue
            for m in merged:
                if m is owner:
                    continue
                if any(covers(x) for x in m["spans"] if x.get("origin") and str(x.get("text") or "").strip()):
                    m["spans"].append(s)
                    m["bbox"] |= sb
                    if owner is not None:
                        merged.remove(owner)
                    break
    merged = _join_line_wrapped_formulas(merged, block.span_lines, block.size)

    formulas: list[dict] = []
    formula_runs: dict[int, FormulaRun] = {}
    first_span: dict[int, int] = {}  # span id → 公式序号
    span_line = {id(s): li for li, line in enumerate(block.span_lines) for s in line}

    def text_baseline(spans: list[dict]) -> float | None:
        """几截字所在行正文的基线：这几行里正文字号的字 origin 的中位数，不能按全块算"""
        lines = {span_line.get(id(span)) for span in spans}
        origins = sorted(
            span["origin"][1]
            for li in lines
            if li is not None
            for span in block.span_lines[li]
            if span["text"].strip() and span.get("origin") and span["size"] >= block.size * 0.9
        )
        return origins[len(origins) // 2] if origins else None

    line_x0 = [_line_x0(spans) for spans in block.span_lines]
    for n, m in enumerate(merged, 1):
        bbox = pymupdf.Rect(m["bbox"])
        spans = sorted(m["spans"], key=lambda s: (s["bbox"][0], s["bbox"][1]))
        main_span = max(spans, key=lambda s: s["size"])
        baseline = main_span["origin"][1] if main_span.get("origin") else bbox.y1
        # 孤立角标公式（(1−ε_i)² 的 ²）的 origin 是抬高的：渲染时按它所在
        # 正文行的 baseline 高差抬回去，否则平方落到正文基线上像个普通数字。
        # 高差必须相对本行正文（正文行 origin 的中位数），不能相对全块
        own_origin = text_baseline(m["spans"]) or baseline
        raise_dy = max(0.0, own_origin - baseline)
        anchor = min(m["spans"], key=lambda s: s["bbox"][0])
        anchor_line = span_line.get(id(anchor), 0)
        formula_run = FormulaRun(
            index=n,
            bbox=bbox,
            baseline=baseline,
            dx=round(bbox.x0 - line_x0[anchor_line], 1),
            dy=round(raise_dy, 1),
            descent=round(max(0.0, bbox.y1 - baseline), 1),
            text=_formula_markup(spans, block.size).strip(),
            page=block.page,
            name=f"f{block_index}_{n}.png",
            line=anchor_line,
        )
        formula_runs[n] = formula_run
        record = _formula_record(formula_run, spans)
        extras = m.get("extras") or []
        if extras:
            record["text"] = " ".join(
                piece
                for piece in [
                    record["text"],
                    *(_formula_markup(extra["spans"], block.size).strip() for extra in extras),
                ]
                if piece
            )
            record["parts"] = [
                record,
                *(
                    _part_record(extra, block.page, f"{formula_run.name[:-4]}_p{i}.png", text_baseline(extra["spans"]))
                    for i, extra in enumerate(extras, 1)
                ),
            ]
            for part, before, after in zip(record["parts"][1:], [m, *extras], extras):
                part["lead"] = _wrap_gap(before["spans"], after["spans"])
        formulas.append(record)
        first_span[id(anchor)] = n

    # 3) 按行重建送翻文本，并记下有序 run。公式锚点是一个 FormulaRun，
    # 同行同风格的文字并成一个 TextRun。和底色不同的强调、字号才写标记。
    formula_ids = {id(s) for m in merged for s in m["spans"]}
    formula_ids |= {id(s) for m in merged for extra in m.get("extras") or [] for s in extra["spans"]}
    caps = _small_caps_sizes(block.span_lines)

    def size_of(span: dict) -> float:
        return caps.get(id(span), float(span.get("size") or 0))

    text_spans = [s for line in block.span_lines for s in line if id(s) not in formula_ids]
    sizes = {round(size_of(s), 1) for s in text_spans if str(s.get("text") or "").strip()}
    # 整块小型大写：大小写各一个字号，其实是一种字，不写字号标记
    base_size = sizes.pop() if caps and len(sizes) == 1 else block.size
    writer = writer_for([{**s, "size": size_of(s)} for s in text_spans], base_size, _color_value(block.color))
    runs: list[Run] = []
    out_lines: list[str] = []
    pending: TextRun | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            runs.append(pending)
            pending = None

    for line_no, spans in enumerate(block.span_lines):
        parts: list[str] = []
        for s in spans:
            n = first_span.get(id(s))
            if n is not None:
                flush()
                runs.append(formula_runs[n])
                parts.append(writer.atom(f"{{v{n}}}"))
            elif id(s) in formula_ids or id(s) in ordinal_skip:
                continue  # 已被某个公式吞掉，或是序数的上标
            elif id(s) in ordinal_at:
                reading = ordinal_at[id(s)]
                emphasis = span_emphasis(s)
                parts.append(writer.text(reading, emphasis, size_of(s), s.get("color")))
                shown = dict(s)
                shown["text"] = reading
                if pending is not None and not _same_text_style(pending, s, emphasis):
                    flush()
                pending = _extend_text_run(pending, shown, emphasis, line_no)
            else:
                emphasis = span_emphasis(s)
                parts.append(writer.text(s["text"], emphasis, size_of(s), s.get("color")))
                if pending is not None and not _same_text_style(pending, s, emphasis):
                    flush()
                pending = _extend_text_run(pending, s, emphasis, line_no)
        flush()
        parts.append(writer.close())
        out_lines.append("".join(parts))
    block.runs = runs
    return _join_lines(out_lines), formulas


def _line_x0(spans: list[dict]) -> float:
    xs = [pymupdf.Rect(s["bbox"]).x0 for s in spans if s.get("bbox") is not None]
    return min(xs) if xs else 0.0


def _span_box(span: dict) -> pymupdf.Rect:
    raw = span.get("bbox")
    return pymupdf.Rect(raw) if raw is not None else pymupdf.Rect(0, 0, 0, 0)


def _union_box(a: pymupdf.Rect, b: pymupdf.Rect) -> pymupdf.Rect:
    """和 a | b 一样：空框不算，结果按 MuPDF 存成单精度。
    每个 span 都要并一次，Rect 的 | 要先复制再逐项检查参数，慢。"""
    if a.is_infinite or b.is_infinite:
        return a | b
    if b.is_empty:
        return pymupdf.Rect(a)
    if a.is_empty:
        return pymupdf.Rect(b)
    box = (min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))
    return pymupdf.Rect(struct.unpack("4f", struct.pack("4f", *box)))


def _same_text_style(pending: TextRun, span: dict, emphasis: Emphasis) -> bool:
    return (
        pending.emphasis == emphasis
        and pending.font == str(span.get("font") or "")
        and abs(pending.size - float(span.get("size") or 0)) < 0.1
    )


def _extend_text_run(pending: TextRun | None, span: dict, emphasis: Emphasis, line: int) -> TextRun:
    """同行、同字体、同强调的相邻 span 并成一个文字 run。"""
    bbox = _span_box(span)
    baseline = span["origin"][1] if span.get("origin") else bbox.y1
    font = str(span.get("font") or "")
    size = float(span.get("size") or 0)
    if pending is not None and _same_text_style(pending, span, emphasis):
        return replace(pending, text=pending.text + str(span.get("text") or ""), bbox=_union_box(pending.bbox, bbox))
    return TextRun(
        text=str(span.get("text") or ""),
        font=font,
        size=size,
        emphasis=emphasis,
        baseline=baseline,
        bbox=bbox,
        line=line,
    )


def _line_edge(span_lines: list[list[dict]], line_no: int, side: str) -> float:
    edges: list[float] = []
    for span in span_lines[line_no]:
        raw = span.get("bbox")
        if raw is None or not str(span.get("text") or "").strip():
            continue
        box = pymupdf.Rect(raw)
        edges.append(box.x0 if side == "x0" else box.x1)
    if not edges:
        return 0.0
    return min(edges) if side == "x0" else max(edges)


def _only_this_run(run: dict, span_lines: list[list[dict]], head: bool) -> bool:
    line_no, start, end = run["pos"]
    spans = span_lines[line_no][:start] if head else span_lines[line_no][end + 1 :]
    return not any(str(span.get("text") or "").strip() for span in spans)


def _continues_wrapped_formula(
    prev: dict,
    nxt: dict,
    span_lines: list[list[dict]],
    block_size: float,
) -> bool:
    """行末公式接到下一行行首，是同一式子被换行切开。中间不能还有正文。"""
    last = (prev.get("extras") or [prev])[-1]
    last_line, _, _ = last["pos"]
    next_line, _, _ = nxt["pos"]
    if next_line != last_line + 1:
        return False
    if not _only_this_run(last, span_lines, head=False) or not _only_this_run(nxt, span_lines, head=True):
        return False
    if _line_edge(span_lines, last_line, "x1") - last["bbox"].x1 > block_size * 1.5:
        return False
    if nxt["bbox"].x0 - _line_edge(span_lines, next_line, "x0") > block_size * 1.2:
        return False
    # 按字高量行距：CMSY 的字体框伸到下一行，两行看着像交叠
    y_gap = _run_text_box(nxt).y0 - _run_text_box(last).y1
    return -block_size * 0.5 <= y_gap < block_size * 1.8


def _ends_row(run: dict, span_lines: list[list[dict]]) -> bool:
    """run 右边、同一行高上没有别的字。"""
    box = _run_text_box(run)
    own = {id(span) for span in run["spans"]}
    for spans in span_lines:
        for span in spans:
            if id(span) in own or not str(span.get("text") or "").strip() or "bbox" not in span:
                continue
            other = _text_box(span)
            overlap = min(box.y1, other.y1) - max(box.y0, other.y0)
            if other.x0 >= box.x1 - 1 and overlap > min(box.height, other.height) * 0.5:
                return False
    return True


def _run_text_box(run: dict) -> pymupdf.Rect:
    box = pymupdf.Rect()
    for span in run["spans"]:
        if str(span.get("text") or "").strip() and "bbox" in span:
            box |= _text_box(span)
    return box if not box.is_empty else pymupdf.Rect(run["bbox"])


def _fraction_pair(a: dict, b: dict, block_size: float) -> bool:
    """一段全是小字号，和另一段里的一截小字号上下贴着、左右居中：是同一个行内分式的分子分母。
    相邻两行行首的脚注记号也上下对齐，但隔着一行，不贴着。"""
    small = block_size * 0.79
    for whole, other in ((a, b), (b, a)):
        if whole["max_size"] >= small:
            continue
        piece = _small_band(other, whole["bbox"], small)
        if piece is None:
            continue
        box = whole["bbox"]
        lo = min(whole["max_size"], max(float(span["size"]) for span in other["spans"]))
        overlap = min(box.x1, piece.x1) - max(box.x0, piece.x0)
        gap = max(box.y0, piece.y0) - min(box.y1, piece.y1)
        centered = abs((box.x0 + box.x1) - (piece.x0 + piece.x1)) / 2 <= max(1.0, 0.15 * min(box.width, piece.width))
        if overlap > max(1.0, 0.3 * min(box.width, piece.width)) and gap < lo * 0.3 and centered:
            return True
    return False


def _run_baseline(run: dict) -> float:
    """公式 run 里正文字号的那几截的基线，取中位数。角标抬高、压低，不算。"""
    sizes = [float(span.get("size") or 0) for span in run["spans"] if span.get("origin") is not None]
    top = max(sizes, default=0.0)
    lows = sorted(
        float(span["origin"][1])
        for span in run["spans"]
        if span.get("origin") is not None and float(span.get("size") or 0) >= top * 0.9
    )
    return lows[len(lows) // 2] if lows else float(run["bbox"].y1)


def _join_line_wrapped_formulas(
    runs: list[dict],
    span_lines: list[list[dict]],
    block_size: float,
) -> list[dict]:
    """换行切开的公式收成一个占位符的多片。不并 bbox，否则裁进两行之间的整栏空白。
    几片并排起来比一行还宽的不收（跨两行的长网址）：占位符整个排在一行里，放不下就把整段缩小，
    分开的几片可以像原文一样在断处换行。"""
    if len(runs) < 2:
        return runs
    line_width = max(
        (
            max(float(span["bbox"][2]) for span in line) - min(float(span["bbox"][0]) for span in line)
            for line in span_lines
            if line
        ),
        default=0.0,
    )
    joined: list[dict] = []
    for run in runs:
        if joined and _continues_wrapped_formula(joined[-1], run, span_lines, block_size):
            pieces = [joined[-1], *joined[-1].get("extras", []), run]
            if not line_width or sum(pymupdf.Rect(piece["bbox"]).width for piece in pieces) <= line_width:
                joined[-1].setdefault("extras", []).append(run)
                continue
        joined.append(run)
    return joined


def _part_record(run: dict, page: int, name: str, baseline: float | None = None) -> dict:
    """换行切开的公式后面的一片。baseline 是这一片所在行正文的基线；分式的分子分母一样大，
    按片里最大的字量会量到分母的基线上，贴的时候整个分式往上抬"""
    bbox = pymupdf.Rect(run["bbox"])
    spans = run["spans"]
    main = max(spans, key=lambda span: span["size"])
    if baseline is None:
        baseline = main["origin"][1] if main.get("origin") else bbox.y1
    return {
        "page": page,
        "bbox": bbox,
        "name": name,
        "w": round(bbox.width, 1),
        "h": round(bbox.height, 1),
        "d": round(max(0.0, bbox.y1 - baseline), 1),
        "raise": 0.0,
        "spans": spans,
        "curves": [],
        "text": _formula_markup(spans, float(main["size"])).strip(),
    }


# 一片里有空格隔开的两个词：是一句等宽字体的例句，断行处也是词间
_SPACED_WORDS_RE = re.compile(r"[^\W_]\s+[^\W_]")


def _wrap_gap(before: list[dict], after: list[dict]) -> float:
    """换行切开的公式两片之间的空。等宽字体的例句在词间断行（my dog is | hairy），留一个词距，
    取片里空格的宽度。片里的词本来就用空格隔开、断处两边都是字母或数字才算；
    网址、标识符在词中间断行，式子在运算符后面断行，都挨着排。"""

    def text(spans: list[dict]) -> str:
        return "".join(str(span.get("text") or "") for span in sorted(spans, key=lambda span: span["bbox"][0]))

    head, tail = text(before), text(after)
    last, first = head.rstrip()[-1:], tail.lstrip()[:1]
    if not (last.isalnum() and first.isalnum()) or not (_SPACED_WORDS_RE.search(head) or _SPACED_WORDS_RE.search(tail)):
        return _PART_GAP
    spaces = sorted(
        span["bbox"][2] - span["bbox"][0]
        for span in [*before, *after]
        if str(span.get("text") or "") and not str(span.get("text") or "").strip()
    )
    return max(_PART_GAP, spaces[len(spaces) // 2]) if spaces else _PART_GAP


def _formula_record(run: FormulaRun, spans: list[dict]) -> dict:
    """渲染层仍读这条记录。框和曲线随后可能变，run 再同步回去。"""
    return {
        "page": run.page,
        "bbox": pymupdf.Rect(run.bbox),
        "name": run.name,
        "w": round(run.bbox.width, 1),
        "h": round(run.bbox.height, 1),
        "d": run.descent,
        "raise": run.dy,
        "dx": run.dx,
        "text": run.text,
        "spans": spans,
        "curves": [],
        "owner": run.owner,
    }


_OPERATOR_CHARS = "+−×±"
# 式子里的正文词不到这个数：argmax、Precision 这种名字
_EQUATION_PROSE = 3


def _equation_block(block: TextBlock) -> bool:
    """带关系符或公式字体的运算符、式子外几乎没有正文词的块：T̂ = argmax s(T)、
    Precision = |R|、对齐的第二行 + max[…]。并完块再判断：一个式子常被拆成好几截，
    逐截判断会把剩下的截断开。图注带编号，不算；算法框里的一行有一个词也照翻（Output: …）。"""
    if block.clip is not None:
        return False
    spans = [span for line in block.span_lines or [] for span in line if span["text"].strip()]
    text = " ".join(span["text"] for span in spans)
    if _CAPTION_NUM_RE.match(text) or _plain_units(spans) >= _EQUATION_PROSE:
        return False
    return bool(_RELATION_RE.search(text)) or any(
        _span_is_math(span) and any(c in _OPERATOR_CHARS for c in span["text"]) for span in spans
    )


def _margin_term(block: TextBlock) -> bool:
    """页边的术语注（perplexity、log probability）：几个词、不成句。留原文。
    译文比原文宽，挤在窄窄的页边要换好几行，还会顶到上下的注；原文的术语也正是读者要对照的。
    成句的旁注照常翻译。"""
    text = block.text.strip()
    return block.side_note and len(text.split()) <= 4 and not text.endswith((".", "?", "!", "。", "？", "！"))


def _colored_examples(blocks: list[TextBlock], geometries: list[PageGeometry]) -> set[int]:
    """正文中间单独排开的一句彩色例句（讲指代、句法时举的句子）。下文拿原句里的英文词来讲，留原文。

    整块一种鲜明的颜色、正体不加粗、正文字号，左边缩进一两个字宽，说完一句。
    章首题词也这样排，但后面跟着出处，照常翻译。"""
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if not block.side_note:
            by_page.setdefault(block.page, []).append(block)
    found: set[int] = set()
    for page_blocks in by_page.values():
        page_blocks.sort(key=lambda block: block.rect.y0)
        for index, block in enumerate(page_blocks):
            frame = geometries[block.page].frame
            if frame is None or not _colored_example(block, frame):
                continue
            below = page_blocks[index + 1] if index + 1 < len(page_blocks) else None
            if below is None or not _attribution(block, below):
                found.add(id(block))
    return found


def _attribution(quote: TextBlock, below: TextBlock) -> bool:
    """题词的出处：紧跟在下面，小一号，或者是靠右排的短行（可以和末行同一行）。"""
    if below.rect.y0 - quote.rect.y1 >= quote.size * 1.5:
        return False
    flush_right = below.rect.width < quote.rect.width * 0.5 and abs(below.rect.x1 - quote.rect.x1) < quote.size
    return below.size < quote.size * 0.95 or flush_right


def _colored_example(block: TextBlock, frame: TextFrame) -> bool:
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    # 单独排开的一行例句常常不带句号（The green witch arrived）
    if (
        not spans
        or block.bold
        or len(block.text.split()) < 4
        or not (_ends_sentence(block.text) or len(_visual_lines(block)) == 1)
    ):
        return False
    if block.body_size and abs(block.size - block.body_size) > 0.5:
        return False
    colors = {int(span.get("color") or 0) for span in spans}
    if len(colors) != 1:
        return False
    if not _vivid(colors.pop()):
        return False
    if sum(len(span["text"]) for span in spans if int(span.get("flags") or 0) & 2) * 2 > sum(
        len(span["text"]) for span in spans
    ):
        return False
    indent = block.rect.x0 - frame.x0
    return block.size * 1.5 <= indent <= block.size * 4 and block.rect.x1 <= frame.x1 + 2


def _mono_share(block: TextBlock) -> float:
    """块里正文字（公式除外）有多少是等宽字体。只数字母：行首的计数、编号用正文字体排，
    不说明这一行是正文（语料表的 2 n e w）。"""
    total = mono = 0
    for span in _block_spans(block):
        if _span_is_math(span):
            continue
        count = sum(1 for ch in str(span["text"]) if ch.isalpha() and not CJK_RE.match(ch))
        total += count
        if _mono_span(span):
            mono += count
    return mono / total if total else 0.0


def _code_block(block: TextBlock) -> bool:
    """整块正文几乎都是等宽字体：程序、SQL、命令行，还有照录的提示词、对话。
    译了会把关键字也换掉，照录的文字也不再是原样，留下原字。

    句子里夹着的等宽单词（属性名、函数名）不在这里。块里八成以上是等宽字才算，
    FROM R 这种短行认不出等宽也不影响整块。
    正文字只有一个词、等宽字占三分之二以上的，是表格一行的行名带着几格记号（rooster -female,
    +chicken, +adult），和同表的其他行一样留原字。
    """
    share = _mono_share(block)
    if share >= 0.8:
        return True
    plain = " ".join(
        str(span["text"]) for span in _block_spans(block) if not _span_is_math(span) and not _mono_span(span)
    )
    return share >= 2 / 3 and sum(1 for word in plain.split() if any(ch.isalpha() for ch in word)) == 1


def _split_gloss_lines(
    blocks: list[TextBlock],
    glosses: dict[int, tuple[pymupdf.Rect, ...]],
) -> list[TextBlock]:
    """对译行和下面的意译常分在同一块里。按行拆开，对译留原文，意译另成一块：不在编号例句里的照翻。"""
    out: list[TextBlock] = []
    for block in blocks:
        rects = glosses.get(block.page, ())
        lines = block.line_rects
        if not rects or len(lines) < 2 or not block.span_lines or len(block.span_lines) != len(lines):
            out.append(block)
            continue
        flags = [_in_gloss(line, rects) for line in lines]
        if all(flags) or not any(flags):
            out.append(block)
            continue
        start = 0
        for index in range(1, len(lines) + 1):
            if index == len(lines) or flags[index] != flags[start]:
                out.append(_block_slice(block, start, index))
                start = index
    return out


def _table_cell(block: TextBlock, geo: PageGeometry) -> bool:
    """表格里的字不翻译，留原文。算法框的伪代码虽然也切成格子，照样逐行翻译。"""
    return block.clip is not None and any(
        not table.algorithm and any(_clip_key(cell) == block.clip for cell in table.cells) for table in geo.tables
    )


def _scan_status(doc: pymupdf.Document) -> str:
    """ok=正常；hidden_text=带隐藏 OCR 文本层（render mode 3，文字不可见但可提取）；
    scanned=完全没有文本层的纯扫描件。

    隐藏层文件照翻：OCR 层的坐标通常和图像对齐，译文写回原位置后是可见的，
    效果和 BabelDOC 的 ocr_workaround 一样。纯扫描件没有可翻的东西，明确拒绝。
    """
    text_pages = hidden_pages = 0
    for page in doc:
        if not page.get_text().strip():
            continue
        text_pages += 1
        if b" 3 Tr" in page.read_contents():  # 渲染模式 3 = 不可见文字
            hidden_pages += 1
    if text_pages == 0:
        return "scanned"
    if text_pages >= 3 and hidden_pages / text_pages > 0.8:
        return "hidden_text"
    return "ok"


def _retag_runs(runs: list[Run], owner: int, index_offset: int) -> list[Run]:
    """公式序号改成单元内的全局序号，归属记在 run 上。"""
    retagged: list[Run] = []
    for run in runs:
        if isinstance(run, FormulaRun):
            retagged.append(replace(run, owner=owner, index=run.index + index_offset))
        else:
            retagged.append(replace(run, owner=owner))
    return retagged


def _bridge_hyphen(left: str, right: str) -> tuple[str, str]:
    """行尾连字符断词时，把词补全再送翻。分界仍留在两页之间。"""
    stripped = left.rstrip()
    if not stripped.endswith("-"):
        return left, right
    rest = right.lstrip()
    head, sep, tail = rest.partition(" ")
    if not head or re.match(r"[A-Za-zÀ-ÿ]", head) is None:
        return left, right
    return stripped[:-1] + head, tail if sep else ""


# TeX 的 \' \` \^ \" \~ 把重音画成单独的字形，提出来是“Montr´eal”“Fr´ed´eric”。
# 送翻前合成 é：模型认得出人名地名，写回的译文里也不再出现散开的重音。
_SPACING_ACCENTS = {
    "´": "\u0301",
    "`": "\u0300",
    "ˆ": "\u0302",
    "¨": "\u0308",
    "˜": "\u0303",
    "ˇ": "\u030c",
    "˘": "\u0306",
    "˙": "\u0307",
    "˚": "\u030a",
}
# 重音和字母之间可能隔着样式标记：ŷ 的 y 是斜体，提出来是 ˆ{i}y{/i}
_ACCENT_BEFORE_RE = re.compile(
    "([" + "".join(_SPACING_ACCENTS) + r"])((?:\{[biu]\}|\{z\d+\}|\{c[0-9a-f]{6}\})*)([A-Za-zıȷ])"
)
# 软音符只挂在 c、s、t 下面，提出来可能在字母前，也可能在后：c¸、¸c → ç
_CEDILLA_RE = re.compile(r"([cCsStT])\s?¸|¸\s?([cCsStT])")


def _compose_accents(text: str) -> str:
    def before(m: re.Match[str]) -> str:
        letter = {"ı": "i", "ȷ": "j"}.get(m.group(3), m.group(3))
        return m.group(2) + unicodedata.normalize("NFC", letter + _SPACING_ACCENTS[m.group(1)])

    text = _ACCENT_BEFORE_RE.sub(before, text)
    return _CEDILLA_RE.sub(lambda m: unicodedata.normalize("NFC", (m.group(1) or m.group(2)) + "\u0327"), text)


def _only_formulas(sent: str) -> bool:
    """送翻文本里除了公式占位符、样式标记，只剩标点和空白。"""
    return not re.search(r"[^\W\d_]", _MARKUP_RE.sub("", sent))


def _prepare_unit(unit: list[TextBlock], name_start: int) -> tuple[str, list[dict], int]:
    """一个翻译单元的送翻文本和公式记录。公式名按块递增，避免跨页两块撞名。

    两块合成一段时，中间插入 {| }。同一页共基线的半句已经是一句，不插分界。
    译完收成一段，按满行排，排满再接到下一页。
    行尾连字符先把词补全，模型看到的是完整单词，分界仍在。
    """
    texts: list[str] = []
    formulas: list[dict] = []
    name_i = name_start
    for ui, block in enumerate(unit):
        sent, found = _placeholderize(block, name_i)
        name_i += 1
        offset = len(formulas)
        if offset:
            sent = re.sub(
                r"\{v(\d+)\}",
                lambda m, off=offset: f"{{v{int(m.group(1)) + off}}}",
                sent,
            )
        for rec in found:
            rec["owner"] = ui
        block.runs = _retag_runs(block.runs, ui, offset)
        formulas.extend(found)
        texts.append(sent)
    if len(texts) == 2:
        left, right = _bridge_hyphen(texts[0], texts[1])
        # 同一页的半句已经是一句。换栏和跨页仍留分界，写回再按满行接。
        if _same_page_line_pair(unit[0], unit[1]):
            return _compose_accents(_join_lines([left, right] if right else [left])), formulas, name_i
        parts = [left, BOUNDARY]
        if right:
            parts.append(right)
        return _compose_accents(_join_lines(parts)), formulas, name_i
    return _compose_accents(_join_lines(texts)), formulas, name_i


def _peel_number(sent: str, kind: str) -> tuple[str, str]:
    """摘下单元开头的节号、条目号，只送后面的文字，写回时原样接回去（见 _with_label）。
    连编号一起送，模型会把 4 Experiments 译成「第四 实验」，把 B.3.2 写成 B.三.二、1. 写成「一，」。
    不以点收尾的数（2 samples、1.5 times）、单个大写字母、罗马数字也可能是正文句首
    （I. Goodfellow），只在标题和不成句的短行上摘。"""
    m = _NUMBER_LABEL_RE.match(sent)
    if not m:
        return "", sent
    number = m.group(2)
    rest = sent[m.end() :]
    if not re.search(r"[^\W\d_]", strip_style_marks(rest)):
        return "", sent
    sure = bool(
        re.fullmatch(r"\d{1,3}\.|\d{1,3}(?:\.\d{1,3})+\.|[A-Z](?:\.\d{1,3})+\.?|\(?[a-z]\)|\(\d{1,2}\)", number)
    )
    if not sure and kind == "p" and (len(sent) > 80 or _ends_sentence(sent)):
        return "", sent
    # 粗体从编号一直延续到标题文字（{b}17.2.1 The Encoder{/b}）：编号这边关上，文字这边重开
    closed = re.findall(r"\{/([bizu])\}", m.group(3))
    still_open: list[str] = []
    for mark in re.findall(r"\{[bizu]\d*\}", m.group(1)):
        if mark[1] in closed:
            closed.remove(mark[1])
        else:
            still_open.append(mark)
    label = m.group(0) + "".join(f"{{/{mark[1]}}}" for mark in reversed(still_open))
    return label, "".join(still_open) + rest.lstrip()


def _label_styled(block: TextBlock) -> bool:
    """开头一段（图注的 Figure 2.4）和正文主体的字体不同。颜色不算：正文里的图号常是蓝色链接。"""
    spans = [span for line in block.span_lines or [] for span in line if span["text"].strip()]
    if len(spans) < 2:
        return False
    main = max(spans, key=lambda span: len(span["text"]))
    first = spans[0]
    return first is not main and (
        first["font"] != main["font"] or bool(first["flags"] & 16) != bool(main["flags"] & 16)
    )


_REPORT_UNITS = 200


@dataclass
class _PdfPlan:
    """送翻前准备好的东西：原书、块、每页的几何，和按单元整理好的送翻文本。"""

    src_doc: pymupdf.Document
    blocks: list[TextBlock]
    geometries: list[PageGeometry]
    toc: list
    units: list[list[TextBlock]]
    texts: list[str]
    formulas: list[list[dict]]
    kinds: list[str]
    skips: list[bool]
    skip_ids: set[int]
    block_kind: dict[int, str]
    title: str = ""
    labels: list[str] = field(default_factory=list)  # 每个单元开头摘下的编号，见 _peel_number
    notes: list[str] = field(default_factory=list)  # 每个单元在书里所处位置的说明，见 _index_notes
    aparts: list[bool] = field(default_factory=list)  # 两块中间隔着行间公式、例句的单元，见 _seam_apart


def _plan_document(
    src: Path,
    src_doc: pymupdf.Document,
    runner,
    layout_model: str,
    formula_model: FormulaModel | None = None,
    boxes: ModelBoxes | None = None,
) -> _PdfPlan:
    # 逐页读线、表、字占准备时间的大半，按页报进度
    formulas = formula_detector(src_doc, formula_model, boxes)
    blocks, frames, scans = _extract(
        src_doc,
        lambda done, total: runner.report_stage("分析版面", done, total, "页"),
        formulas,
    )
    runner.report_stage("整理段落")
    glosses = {
        pno: interlinear_glosses(src_doc[pno], scans[pno].words) for pno in sorted({block.page for block in blocks})
    }
    blocks = _split_gloss_lines(blocks, glosses)
    # 纯扫描件提不出块。有块的文件不必再逐页查文本层
    if not blocks and _scan_status(src_doc) == "scanned":
        raise ValueError(
            "此 PDF 为没有文本层的扫描件，请先使用 Acrobat、ABBYY 或 WPS 的 OCR 功能识别文字，再上传识别后的文件"
        )
    if not blocks:
        raise ValueError("未提取到可用的文字：页面可能为图片，或 OCR 文本层质量过低")
    toc = src_doc.get_toc()  # 输出文件带上原书签
    # PDF 元数据里的标题经常是 "Microsoft Word - xxx.docx" 之类，看起来像文件名就不用
    meta_title = ((src_doc.metadata or {}).get("title") or "").strip()
    if re.search(r"\.(docx?|pdf|tex|indd)$|^untitled$", meta_title, re.I):
        meta_title = ""
    title = meta_title or src.stem
    runner.set_title(title)
    median = _median_size(blocks)
    # 栏和障碍先于角色、也先于跨页合并：图内文字和图注靠这些框来认
    rects_by_page: dict[int, list[pymupdf.Rect]] = {}
    for b in blocks:
        rects_by_page.setdefault(b.page, []).append(b.rect)
    geometries = [
        PageGeometry.from_page(src_doc[i], rects_by_page.get(i, []), frames[i], scans[i])
        for i in range(src_doc.page_count)
    ]
    detector = layout_detector(src_doc, layout_model, boxes)
    # 开了版面模型时逐页推好、按页报进度，推完回到「整理段落」
    modeled: list[int] = []

    def on_model_page(done: int, total: int) -> None:
        modeled.append(done)
        runner.report_stage("版面模型分析", done, total, "页")

    detector.prepare(sorted({b.page for b in blocks}), on_model_page)
    if modeled:
        runner.report_stage("整理段落")
    roles = detector.classify(
        [
            LayoutItem(b.page, pymupdf.Rect(b.rect), b.text, b.size, b.bold, _label_styled(b), _all_styled(b))
            for b in blocks
        ],
        geometries,
        median,
    )
    edges = margin_skips(blocks, geometries)
    displayed = (
        _display_formula_ids(blocks, geometries)
        | _isolated_formula_ids(blocks, formulas, roles)
        | _gloss_ids(blocks, glosses)
        | _code_panel_ids(blocks, scans)
        | _foreign_ids(blocks)
        | _colored_examples(blocks, geometries)
        | _color_keyed_ids(blocks)
        | _item_example_ids(blocks)
        | _arc_figure_ids(blocks, scans)
    )
    numbers = {pno: example_numbers(scans[pno].words) for pno in sorted({block.page for block in blocks})}
    examples = _example_ids(blocks, numbers) | _role_pattern_ids(blocks) | _concordance_ids(blocks)
    displayed |= _display_example_ids(blocks, geometries, displayed | examples)
    # 页眉不进大纲，旁注、目录条目不是标题
    kinds = [
        "h2" if role is Role.HEADING and not edge and not block.side_note and block.leader is None else "p"
        for role, edge, block in zip(roles, edges, blocks)
    ]
    cell_ids = {id(block) for block in blocks if _table_cell(block, geometries[block.page])}
    # 书后的索引留原文：条目按原文字母排序，页码指向原书页，译过来既查不到，也和页码对不上
    skips = [
        bib
        or role is Role.FIGURE
        or edge
        or _garbled_label(block.text)
        or _code_block(block)
        or _margin_term(block)
        or id(block) in displayed
        or id(block) in cell_ids
        or _equation_block(block)
        or id(block) in examples
        or block.index
        for bib, role, edge, block in zip(biblio_skips(blocks), roles, edges, blocks)
    ]
    skip_ids = {id(b) for b, s in zip(blocks, skips) if s}
    margin_ids = {id(b) for b, edge in zip(blocks, edges) if edge}
    # 目录条目到点线就完了，索引条目到页码就完了，不和下一栏、下一页的条目并段
    halt_ids = {
        id(b)
        for b, role in zip(blocks, roles)
        if role is not Role.BODY or b.side_note or b.leader is not None or b.entry
    } | margin_ids
    aside_ids = {id(b) for b, role in zip(blocks, roles) if role in (Role.CAPTION, Role.FIGURE)}
    # 跨页合并照旧把格子当正文看：栏底是格子时不接下一页，格子上面的表头、图里的字不能顶上去当页末
    # 跨页的例句：下一页接着的半句也留原文。先让例句照正文接段，找出接在它后面的块
    if examples:
        skip_ids |= {
            id(block)
            for unit in _cross_page_units(
                blocks,
                median,
                skip_ids - cell_ids - examples,
                geometries,
                halt_ids,
                margin_ids,
                aside_ids,
            )
            for index, block in enumerate(unit)
            if any(id(head) in examples for head in unit[:index])
        }
    units = _cross_page_units(blocks, median, skip_ids - cell_ids, geometries, halt_ids, margin_ids, aside_ids)
    units = _region_joins(units, skip_ids, halt_ids, detector.text_regions)
    block_kind = {id(b): k for b, k in zip(blocks, kinds)}
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    covered = aside_ids | margin_ids | cell_ids
    shown = displayed | examples | {id(block) for block in blocks if _equation_block(block)}
    words_of = functools.cache(lambda pno: src_doc[pno].get_text("words"))
    aparts = [
        len(unit) == 2 and _seam_apart(unit, by_page, geometries, covered, shown, median, words_of) for unit in units
    ]

    sent_texts: list[str] = []
    sent_formulas: list[list[dict]] = []
    sent_kinds: list[str] = []
    sent_skips: list[bool] = []
    sent_labels: list[str] = []
    flat_units: list[list[TextBlock]] = []
    name_i = 0
    for unit_no, unit in enumerate(units):
        # 一本书有上万个单元，要算好几秒
        if unit_no % _REPORT_UNITS == 0:
            runner.report_stage("准备译文", unit_no, len(units), "段")
        # 整单元都跳过时不打占位符、也不标斜体。标记后的字符串和原文不同，
        # 写回会把原字擦掉重排，参考文献的悬挂缩进和行尾断词就没了。
        if all(id(b) in skip_ids for b in unit):
            flat_units.append(unit)
            sent_texts.append(_join_lines(b.text for b in unit))
            sent_formulas.append([])
            sent_kinds.append(block_kind[id(unit[0])])
            sent_skips.append(True)
            sent_labels.append("")
            continue
        sent, formulas, name_i = _prepare_unit(unit, name_i)
        flat_units.append(unit)
        sent_kinds.append(block_kind[id(unit[0])])
        if formulas and _only_formulas(sent):
            # 占位以后没有字可译：正文斜体写变量的独立公式行。和跳过的块一样留原字，
            # 擦掉重排会把两行的式子当成一个高公式缩小
            sent_texts.append(_join_lines(b.text for b in unit))
            sent_formulas.append([])
            sent_skips.append(True)
            sent_labels.append("")
            continue
        label, sent = _peel_number(sent, sent_kinds[-1])
        sent_labels.append(label)
        sent_texts.append(sent)
        sent_formulas.append(formulas)
        sent_skips.append(False)

    return _PdfPlan(
        src_doc,
        blocks,
        geometries,
        toc,
        flat_units,
        sent_texts,
        sent_formulas,
        sent_kinds,
        sent_skips,
        skip_ids,
        block_kind,
        title,
        sent_labels,
        _index_notes(flat_units, sent_texts, sent_skips),
        aparts,
    )


# 比所在那一栏的条目左缘缩进这么多（字号的倍数）才是子条目
_SUBENTRY_INDENT = 0.4
# 说明里最多举几个子条目
_NOTE_CHILDREN = 3


def _index_notes(units: list[list[TextBlock]], texts: list[str], skips: list[bool]) -> list[str]:
    """索引条目单独送翻没有上下文，log 译成「日志」，MT 下面的 divergences 译成「散度」。
    给每条写一句说明：它是书后索引的条目，子条目在哪一条下面，有子条目的举出前几个。
    子条目按缩进认：比同一栏最靠左的条目缩进的，挂在前面最近一条缩进更浅的条目下面。"""
    notes = [""] * len(units)
    by_page: dict[int, list[pymupdf.Rect]] = {}
    for unit in units:
        if unit[0].index:
            by_page.setdefault(unit[0].page, []).append(unit[0].rect)
    parents: list[int | None] = [None] * len(units)
    children: dict[int, list[int]] = {}
    stack: list[tuple[float, int]] = []
    for n, (unit, skip) in enumerate(zip(units, skips)):
        block = unit[0]
        if not block.index:
            stack = []
            continue
        if skip:
            continue
        rect = block.rect
        left = min(other.x0 for other in by_page[block.page] if other.x0 < rect.x1 and other.x1 > rect.x0)
        indent = rect.x0 - left
        while stack and stack[-1][0] >= indent - block.size * _SUBENTRY_INDENT:
            stack.pop()
        if stack:
            parents[n] = stack[-1][1]
            children.setdefault(stack[-1][1], []).append(n)
        stack.append((indent, n))

    def head(n: int) -> str:
        return _INDEX_REF_RE.sub("", prompts.without_marks(texts[n])).strip().rstrip(",")

    for n, unit in enumerate(units):
        if not unit[0].index or skips[n]:
            continue
        note = "An entry in the subject index at the back of the book"
        if parents[n] is not None:
            note += f", listed under the entry “{head(parents[n])}”"
        if children.get(n):
            note += "; its sub-entries include " + ", ".join(f"“{head(c)}”" for c in children[n][:_NOTE_CHILDREN])
        notes[n] = note + "."
    return notes
