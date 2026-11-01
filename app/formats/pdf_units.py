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
    """The whole span in a text font consists of Greek letters or math symbols. Pure accents are left to accent
    absorption and are not treated as a formula."""
    stripped = text.strip()
    if not stripped or all(unicodedata.category(c) in ("Sk", "Mn") for c in stripped):
        return False
    return all(_is_formula_char(c) for c in stripped)


def _smaller_prose_line(spans: list[dict], main_size: float) -> bool:
    """The whole line is smaller than the largest heading in the block and is a complete sentence. That is body text
    under the heading, not a script."""
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
    """Inline formula: a span in a math font, characters boxed by the formula detection model, or a script span
    clearly smaller than body text (citation numbers and the like).
    One or two characters in bold CM (CMBX and similar) count as a single mathbf letter; bold words are still body text.
    Greek letters and math symbols in a text font are formulas too. A single Latin letter is not, unless it is attached
    to a script.
    When prose_line is true, the whole line is body text smaller than the heading and size alone cannot make it a
    formula."""
    if _span_is_math(span) or span.get("detected_formula"):
        return True
    # Near-white text on a dark background (white-on-black caption labels). Rewriting would use the body color and
    # blend into the background, so it is pasted back unchanged
    if _light_text(span):
        return True
    # Monospaced words in a sentence are attribute or function names; they are pasted back unchanged so the model
    # cannot translate some of them and leave others
    if code and MONO_FONT_RE.search(span["font"]) and re.search(r"[A-Za-z]", span["text"]):
        return True
    if span["size"] < main_size * 0.79 and not prose_line:
        return True
    if re.match(r"^CMB", span["font"]) and len(span["text"].strip()) <= 2:
        # of / to in a caption are also two letters, but they are words, not mathbf variables.
        token = span["text"].strip()
        if len(token) == 2 and token.isalpha() and token.islower():
            return False
        return True
    return _is_symbol_span(span["text"])


def _formula_markup(run_spans: list[dict], main_size: float) -> str:
    """Join a formula run into a string with superscript and subscript sentinels (\\x01s\\x02..\\x01/s\\x02 for a
    superscript, b for a subscript).
    After they are restored into the translation the rendering layer turns them into <sup>/<sub>, so y_1^T is no
    longer flattened into y1T.
    Superscript and subscript are decided by baseline offset: above the body baseline is a superscript, below it a
    subscript."""
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


# Isolated punctuation is not a formula: the angle brackets of a token such as <EOS> are in a math font, and pulling
# them out alone would break the token apart and glue brackets of adjacent lines into odd images spanning lines.
# Brackets and punctuation stay in the body text for typesetting (the text font can draw them).
# Symbol fonts have no text punctuation: < in CMSY is a relation symbol such as ≺, and even a lone one is a formula
_SYMBOL_FONT_RE = re.compile(r"^CMSY|^MSAM|^MSBM", re.I)
_LONE_PUNCT_RE = re.compile(r"^[\s<>[\]{}()|/\\=+\-_.,;:'\"~·，。；：！？（）【】]+$")
# Upright variable attached to a script. Two letters would pull of/to into a superscript, so only one letter is taken.
_VARIABLE_BASE_RE = re.compile(r"^[A-Za-z]['′]?$")


# Operators whose following number belongs to the formula
_ASSIGN_OPS = {"=", "<", ">", "+", "-", "−"}


def _letter_formula_runs(span_lines: list[list[dict]], runs: list[dict]) -> list[dict]:
    """Single italic letters in a text font mixed with brackets and equals signs that the whole-text font pass marked
    as formula (P(y = 1), f(x)): P and y are body italics and the brackets and equals signs are each isolated
    punctuation, so none of them is a formula alone. A contiguous sequence on one line that contains both punctuation
    marked as formula and single italic letters is an inline formula. Digits inside it belong to the formula; digits
    at either end are not taken. A digit after an equals sign is taken too (y = 1). Italic words inside the formula's
    brackets and symbols already recognized are part of the sequence (to, | and want in P(to|want)), and formulas the
    sequence covers are merged into it. Returns the new sequences; covered formulas are removed from runs"""
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
        """The sequence ends with a number directly preceded by an operator marked as formula (= 1, < 0)"""
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
    """A single letter is merged into a formula only when it is attached to a smaller script. A citation superscript
    sits to the right of the letter with its center outside the letter's box and is not taken; a script in a math
    font (wide ones such as inf or tr) is not a citation and only needs to touch the letter's right edge.
    script_y is the baseline of the script; a subscript is no more than half a character below its base.
    The box of enc in H^enc on the next line touches the X on the line above, but its baseline is most of a line lower,
    so it is not a subscript of X."""
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
    # A script may extend slightly past the letter's right edge. Further right it is a citation superscript whose
    # center has left the letter's box.
    # On one line the boxes of a base and its script overlap or touch; a letter a few points above or below is on the
    # previous line (the w of w_end is not the base of mention on the next line).
    # A script entirely left of the letter is not its script: the numerator of an inline fraction overlaps vertically
    # with an x later on the same line, with body text between them
    if y_gap >= block_size * 0.3 or bbox.x0 > base.x1 + 0.4 or bbox.x1 <= base.x0:
        return False
    if script_y is not None and span.get("origin"):
        drop = script_y - float(span["origin"][1])
        if drop > block_size * 0.5:
            return False
    return math or base.x0 - 1 <= center <= base.x1 + 2.0


def _has_text_base(run: dict, span_lines: list[list[dict]], members: set[int], block_size: float) -> bool:
    """A run of small characters only has a character at body size, not in a formula, directly to its left on the same
    line: that character is its base."""
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
    """Merge upright variables attached to a script into that script. Words and citation numbers stay in body text.
    Only the scripts at the start of a run are considered: when the scripts of N are followed by an arrow or equals
    sign, the letter is still their base."""
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
            # With a base at body size it is no longer a lone script and must not be merged into another formula as a
            # detached script
            run["max_size"] = max(float(run["max_size"]), float(span.get("size") or 0))
            if other_no == line_no:
                run["pos"] = (line_no, index, end)
            taken.add(id(span))
            break


# Math roman fonts (CMR, LMRoman of Latin Modern) are on the blocklist and are not treated as math fonts. Digits,
# brackets, operators such as equals and plus signs, and operator names attached to a formula are still part of it
_CM_ROMAN_RE = re.compile(r"CMR\d|LMRoman\d")
_GLUE_ATOM = rf"(?:[\d.()\[\]{{}}|=+<>≤≥×]+|(?:{_OPERATOR_NAMES}))"
_MATH_GLUE_RE = re.compile(rf"^(?:{_GLUE_ATOM})(?:\s+{_GLUE_ATOM})*$")
_ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}


def _absorb_inner_blanks(runs: list[dict], span_lines: list[list[dict]]) -> None:
    """Merge whitespace spans between two members of one formula on the same line into the formula. Once several
    pieces are merged into one, a space between them belongs to no piece, and left in body text it becomes an extra
    space after the placeholder."""
    owner = {id(span): index for index, run in enumerate(runs) for span in run["spans"]}
    for line in span_lines:
        shown = [index for index, span in enumerate(line) if str(span.get("text") or "").strip()]
        for left, right in zip(shown, shown[1:]):
            # The whitespace belongs to the formula only when the nearest characters on both sides are in it
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
    """Digits, brackets, operators or operator names in CMR; variables, brackets and operators in a body italic. The
    same characters in an upright text font are not taken.
    Spans in a math font are taken only as single punctuation: other ones outside a formula were put back into body
    text on purpose (the t of the ordinal t-th)."""
    font = str(span.get("font") or "")
    text = str(span.get("text") or "").strip()
    if _CM_ROMAN_RE.search(font):
        return bool(_MATH_GLUE_RE.match(text))
    if _span_is_math(span):
        # > and decimal points put back into body text as punctuation when alone belong to the formula when attached
        # to it (> = 10, 0.31)
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
    """No characters outside a formula may lie between the two boxes. Whitespace may."""
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
    # Vertical comparison uses character height: the font boxes of the CMSY minus sign and vertical bar reach the next
    # line, and P( at the start of the next line would attach to the formula on the line above
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
    """Whitespace between the formula and this glue span. Once it is merged into the formula, the text sent for
    translation has no leftover gap."""
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
    """Merge CMR digits, brackets and operator names attached to a formula into it. Distant digits stay body text."""
    taken = {id(span) for run in runs for span in run["spans"]}
    pending = [
        (line_no, index, span)
        for line_no, line in enumerate(span_lines)
        for index, span in enumerate(line)
        if id(span) not in taken
        and _is_math_glue(span)
        # An italic variable is often followed directly by a subscript or body text (∂L then CE); only upright digits,
        # operators and punctuation risk splitting a number
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
    """The span is followed without a space by letters or digits of body text: in B/T = 0.31, "= 0" is followed by "."
    and "31 likely". Merging it into a formula would split a number in two, so the whole sequence stays in body text.
    Spans in between that could also be merged are skipped over."""
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
    """Ordinals such as i^{th} and 1^{st}. A script whose superscript is not st/nd/rd/th is not an ordinal."""
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
    """Ordinals are not turned into formula images; they are sent for translation as i-th so the model can translate
    them as "第 i 个"."""
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


# Accent marks (hat/tilde/bar/dot and others): those falling on a formula bbox must be merged into it and must not
# stay in body text
_ACCENT_CHARS = {"^", "ˆ", "̂", "~", "˜", "̃", "¯", "̄", "´", "`", "˙", "̇", "¨", "̈", "ˇ", "̌", "˘", "̆", "˚", "̊", "⃗"}


def _placeholderize(block: TextBlock, block_index: int = 0) -> tuple[str, list[dict]]:
    """Replace the inline formulas in the block with {vN} placeholders (the BabelDOC translation protocol) and return
    (text sent for translation, list of formula records).

    Each formula record has bbox (crop area on the original page), name, w/h/d (placement geometry) and text (fallback
    when images are unavailable).
    When rendering, formulas are passed through as vectors with show_pdf_page and look the same as in the original.
    Key merging step: PyMuPDF often puts superscripts and subscripts in a separate dict line, and runs collected per
    line would split one formula into two overlapping pieces (drawn twice, superscript misplaced). So runs are first
    collected over the whole block, and runs whose x ranges overlap are merged into one. Without formulas the returned
    text matches the original character for character, so the cache still hits.
    """
    if not block.span_lines:
        return block.text, []

    # 1) Collect formula runs per dict line and record their position in the block's span stream (for checking
    # adjacency in reading order)
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
                    # Runs of isolated punctuation are not formulas (the angle brackets of <EOS>, brackets, equals
                    # signs) and stay in body text
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

    # 2) Merge runs, in three cases:
    # a) Scripts split into different dict lines: the x overlap is deep enough (≥30% of the width of the narrower run)
    #    and at least one side is small (<0.79 of body size); two runs at body size, even with deep x overlap, are two
    #    formulas on adjacent text lines that happen to align (such as (X,Y) at a line end and y_1,…,y_T on the next).
    # b) Truly adjacent in reading order (only whitespace spans between), x gap smaller than one font size, and
    #    vertically overlapping (the same visual line): consecutive pieces of one math expression (the elements of
    #    y^i_1, y^i_2, …) are merged back into one formula, or the spaces between placeholders would push the
    #    subscripts far from their base.
    #    Runs separated by body text between lines are not merged ((X,Y) vs y_1,…,y_T). The bboxes of the two halves of
    #    a formula wrapped across lines are not merged here.
    # c) Numerator and denominator of an inline fraction: touching vertically and centered horizontally (see
    #    _fraction_pair); they need not be adjacent in reading order.
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

    # A run of small characters only, directly after a body text character on the same line, is that character's
    # script (the H of H^enc is in the body bold).
    # H^enc at the start of the next line sits right below the X on the line above and must not become a subscript of
    # X by x overlap
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
        # b) On the same visual line, adjacent in reading order, x gap smaller than one font size. The two halves of a
        #    formula wrapped across lines do not overlap vertically and are not merged into one large box here.
        y_overlap = min(m["bbox"].y1, r["bbox"].y1) - max(m["bbox"].y0, r["bbox"].y0)
        # The first piece ends a line and the second lies entirely below and to its left: it wrapped to the start of
        # the next line.
        # A numerator with more formula to its right, with the radical of the denominator also below left, is not a
        # wrap
        # Math font boxes are tall and boxes of two lines can overlap, so baselines are compared as well: more than
        # half a line lower means the next line
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
    # The run that most recently took in a piece. After a numerator is merged into the preceding denominator, the
    # numerator's subscript must still find it
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

    # 2.5) Absorb accent spans that fall on a formula bbox (the ^ of ŷ, the ¯ of x̄, and so on):
    # they are often separate spans not judged to be formulas, and left in body text they add a stray symbol beside
    # the formula image;
    # once merged into the run, the bbox extends upward so the crop includes the top of the accent, and it is removed
    # from body text
    for spans in block.span_lines:
        for s in spans:
            if s["text"].strip() not in _ACCENT_CHARS:
                continue
            sb = pymupdf.Rect(s["bbox"])
            # The accent covers a span of the formula on the same line: horizontal overlap and a similar baseline (a
            # hat over a capital letter sits a little higher).
            # Compared against the formula box instead, the two dots of naïve on the line above would be assigned to a
            # formula on the next line
            base = float(s["origin"][1]) if s.get("origin") else (sb.y0 + sb.y1) / 2

            def covers(span: dict) -> bool:
                box = pymupdf.Rect(span["bbox"])
                raised = float(span["origin"][1]) - base
                return min(box.x1, sb.x1) - max(box.x0, sb.x0) > 0 and -0.2 <= raised / block.size <= 0.6

            # The accent may have become a formula of its own (ˆ of CMR10 on a line by itself). After it is merged
            # into the formula it covers, the original is no longer listed, or the text sent for translation would get
            # an extra placeholder holding only ˆ
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
    first_span: dict[int, int] = {}  # span id → formula number
    span_line = {id(s): li for li, line in enumerate(block.span_lines) for s in line}

    def text_baseline(spans: list[dict]) -> float | None:
        """Body baseline of the lines that hold these spans: the median origin of body-size characters on those lines,
        not over the whole block"""
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
        # The origin of an isolated script formula (the ² of (1−ε_i)²) is raised: when rendering it is raised back by
        # its offset from the baseline of its body line, or the square would land on the body baseline like an
        # ordinary digit.
        # The offset must be relative to the body text of its own line (median origin of the body line), not to the
        # whole block
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

    # 3) Rebuild the text sent for translation line by line and record the ordered runs. A formula anchor is one
    # FormulaRun, and text on the same line in the same style becomes one TextRun. Marks are written only for emphasis
    # and sizes that differ from the base style.
    formula_ids = {id(s) for m in merged for s in m["spans"]}
    formula_ids |= {id(s) for m in merged for extra in m.get("extras") or [] for s in extra["spans"]}
    caps = _small_caps_sizes(block.span_lines)

    def size_of(span: dict) -> float:
        return caps.get(id(span), float(span.get("size") or 0))

    text_spans = [s for line in block.span_lines for s in line if id(s) not in formula_ids]
    sizes = {round(size_of(s), 1) for s in text_spans if str(s.get("text") or "").strip()}
    # The whole block in small caps: capitals and small letters have one size each but are one typeface, so no size
    # marks are written
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
                continue  # Already taken by a formula, or the superscript of an ordinal
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
    """Same as a | b: empty boxes are ignored and the result is stored in single precision as MuPDF does.
    It runs once per span, and Rect's | copies first and then checks each argument, which is slow."""
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
    """Merge adjacent spans on the same line with the same font and emphasis into one text run."""
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
    """A formula at a line end continues at the start of the next line: one formula split by a line break. No body
    text may lie between."""
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
    # Measure line spacing by character height: CMSY font boxes reach the next line and make two lines look overlapped
    y_gap = _run_text_box(nxt).y0 - _run_text_box(last).y1
    return -block_size * 0.5 <= y_gap < block_size * 1.8


def _ends_row(run: dict, span_lines: list[list[dict]]) -> bool:
    """No other characters to the right of run at the same line height."""
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
    """One run is entirely small and touches a small band of the other run vertically, centered horizontally: they are
    the numerator and denominator of one inline fraction.
    Footnote marks at the start of two adjacent lines also align vertically, but they are a line apart and do not
    touch."""
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
    """Median baseline of the body-size spans in a formula run. Raised and lowered scripts do not count."""
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
    """Collect a formula split by line breaks into several pieces of one placeholder. The bboxes are not merged, or
    the crop would include the full column width of blank space between the two lines.
    Pieces wider than a line when placed side by side are not collected (a long URL across two lines): a placeholder
    is set entirely on one line, and if it does not fit the whole paragraph is shrunk, while separate pieces can break
    at the same place as in the original."""
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
    """A later piece of a formula split by a line break. baseline is the body baseline of the line holding this piece;
    the numerator and denominator of a fraction are the same size, so measuring by the largest character in the piece
    would hit the denominator's baseline and the whole fraction would be pasted too high"""
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


# Two words separated by a space within a piece: it is a monospaced example sentence, and the break also falls
# between words
_SPACED_WORDS_RE = re.compile(r"[^\W_]\s+[^\W_]")


def _wrap_gap(before: list[dict], after: list[dict]) -> float:
    """Gap between two pieces of a formula split by a line break. A monospaced example sentence breaks between words
    (my dog is | hairy) and gets one word space, taken as the width of the spaces in the pieces. This applies only
    when the words in the pieces are already separated by spaces and both sides of the break are letters or digits;
    URLs and identifiers break inside a word and formulas break after an operator, and those pieces are set flush."""

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
    """The rendering layer still reads this record. The box and curves may change later, and the run is synced back."""
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
# A formula has fewer prose words than this: names such as argmax and Precision
_EQUATION_PROSE = 3


def _equation_block(block: TextBlock) -> bool:
    """A block with a relation symbol or an operator in a math font and almost no prose words outside the formula:
    T̂ = argmax s(T), Precision = |R|, an aligned second line + max[…]. Checked after blocks are merged: a formula is
    often split into several spans, and checking span by span would cut off the remaining ones. Numbered captions do
    not count; a line in an algorithm box with a single word is still translated (Output: ...)."""
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
    """A term in the margin (perplexity, log probability): a few words, not a sentence. The original is kept.
    The translation is wider than the original, would wrap over several lines in the narrow margin and push into the
    notes above and below; the original term is also what the reader wants to compare against.
    Marginal notes that form sentences are translated as usual."""
    text = block.text.strip()
    return block.side_note and len(text.split()) <= 4 and not text.endswith((".", "?", "!", "。", "？", "！"))


def _colored_examples(blocks: list[TextBlock], geometries: list[PageGeometry]) -> set[int]:
    """A colored example sentence set apart in the body text (sentences illustrating coreference or syntax). The text
    that follows discusses English words of the original sentence, so the original is kept.

    The whole block is in one vivid color, upright and not bold, at body size, indented one or two em on the left,
    and forms a complete sentence.
    Chapter epigraphs are set the same way but are followed by their source and are translated as usual."""
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
    """Source of an epigraph: directly below, one size smaller, or a short flush-right line (possibly on the same line
    as the last line)."""
    if below.rect.y0 - quote.rect.y1 >= quote.size * 1.5:
        return False
    flush_right = below.rect.width < quote.rect.width * 0.5 and abs(below.rect.x1 - quote.rect.x1) < quote.size
    return below.size < quote.size * 0.95 or flush_right


def _colored_example(block: TextBlock, frame: TextFrame) -> bool:
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    # A single example line set apart often has no period (The green witch arrived)
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
    """Share of the block's text characters (excluding formulas) set in a monospaced font. Only letters are counted:
    counts and numbers at the start of a line are set in the text font and do not make the line body text (2 n e w in
    a corpus table)."""
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
    """Almost all text in the block is monospaced: programs, SQL, command lines, and verbatim prompts and dialogs.
    Translating them would also replace keywords and the verbatim text would no longer be verbatim, so the original
    is kept.

    Monospaced words inside a sentence (attribute names, function names) are not handled here. A block counts only
    when at least 80% of it is monospaced, so a short line such as FROM R whose monospacing goes unrecognized does not
    affect the whole block.
    A block whose non-monospaced text is a single word, with at least two thirds monospaced, is a table row name with
    a few cells of marks (rooster -female, +chicken, +adult) and keeps the original like the other rows of the table.
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
    """Interlinear gloss lines and the free translation below them often end up in one block. Split them by line: the
    gloss keeps the original, and the free translation becomes its own block, translated as usual unless it is in a
    numbered example."""
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
    """Text in a table is not translated and keeps the original. Pseudocode in an algorithm box is also cut into
    cells but is still translated line by line."""
    return block.clip is not None and any(
        not table.algorithm and any(_clip_key(cell) == block.clip for cell in table.cells) for table in geo.tables
    )


def _scan_status(doc: pymupdf.Document) -> str:
    """ok = normal; hidden_text = has a hidden OCR text layer (render mode 3, invisible but extractable);
    scanned = a pure scan with no text layer at all.

    Files with a hidden layer are translated: the coordinates of the OCR layer usually align with the image, and the
    translation written back in place is visible, with the same effect as BabelDOC's ocr_workaround. A pure scan has
    nothing to translate and is rejected explicitly.
    """
    text_pages = hidden_pages = 0
    for page in doc:
        if not page.get_text().strip():
            continue
        text_pages += 1
        if b" 3 Tr" in page.read_contents():  # render mode 3 = invisible text
            hidden_pages += 1
    if text_pages == 0:
        return "scanned"
    if text_pages >= 3 and hidden_pages / text_pages > 0.8:
        return "hidden_text"
    return "ok"


def _retag_runs(runs: list[Run], owner: int, index_offset: int) -> list[Run]:
    """Renumber formulas with numbers global to the unit and record the owner on the run."""
    retagged: list[Run] = []
    for run in runs:
        if isinstance(run, FormulaRun):
            retagged.append(replace(run, owner=owner, index=run.index + index_offset))
        else:
            retagged.append(replace(run, owner=owner))
    return retagged


def _bridge_hyphen(left: str, right: str) -> tuple[str, str]:
    """When a word is hyphenated at the line end, complete the word before sending it for translation. The boundary
    stays between the two pages."""
    stripped = left.rstrip()
    if not stripped.endswith("-"):
        return left, right
    rest = right.lstrip()
    head, sep, tail = rest.partition(" ")
    if not head or re.match(r"[A-Za-zÀ-ÿ]", head) is None:
        return left, right
    return stripped[:-1] + head, tail if sep else ""


# TeX's \' \` \^ \" \~ draw accents as separate glyphs, extracted as "Montr´eal" and "Fr´ed´eric".
# They are composed into é before translation: the model recognizes the person and place names, and the written
# translation no longer contains detached accents.
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
# Style marks may lie between the accent and the letter: the y of ŷ is italic and is extracted as ˆ{i}y{/i}
_ACCENT_BEFORE_RE = re.compile(
    "([" + "".join(_SPACING_ACCENTS) + r"])((?:\{[biu]\}|\{z\d+\}|\{c[0-9a-f]{6}\})*)([A-Za-zıȷ])"
)
# A cedilla hangs only under c, s and t, and is extracted either before or after the letter: c¸, ¸c → ç
_CEDILLA_RE = re.compile(r"([cCsStT])\s?¸|¸\s?([cCsStT])")


def _compose_accents(text: str) -> str:
    def before(m: re.Match[str]) -> str:
        letter = {"ı": "i", "ȷ": "j"}.get(m.group(3), m.group(3))
        return m.group(2) + unicodedata.normalize("NFC", letter + _SPACING_ACCENTS[m.group(1)])

    text = _ACCENT_BEFORE_RE.sub(before, text)
    return _CEDILLA_RE.sub(lambda m: unicodedata.normalize("NFC", (m.group(1) or m.group(2)) + "\u0327"), text)


def _only_formulas(sent: str) -> bool:
    """Apart from formula placeholders and style marks, the text sent for translation holds only punctuation and
    whitespace."""
    return not re.search(r"[^\W\d_]", _MARKUP_RE.sub("", sent))


def _prepare_unit(unit: list[TextBlock], name_start: int) -> tuple[str, list[dict], int]:
    """Text sent for translation and formula records of one translation unit. Formula names increase per block so
    that two blocks across a page break do not collide.

    When two blocks form one paragraph, {| } is inserted between them. Half sentences on the same page sharing a
    baseline are already one sentence and get no boundary.
    After translation the paragraph is set in full lines and continues on the next page once the page is full.
    A word hyphenated at the line end is completed first, so the model sees the whole word, and the boundary remains.
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
        # Half sentences on the same page are already one sentence. Column and page breaks keep the boundary and are
        # joined in full lines when writing.
        if _same_page_line_pair(unit[0], unit[1]):
            return _compose_accents(_join_lines([left, right] if right else [left])), formulas, name_i
        parts = [left, BOUNDARY]
        if right:
            parts.append(right)
        return _compose_accents(_join_lines(parts)), formulas, name_i
    return _compose_accents(_join_lines(texts)), formulas, name_i


def _peel_number(sent: str, kind: str) -> tuple[str, str]:
    """Peel the section or item number off the start of the unit, send only the text after it, and attach it back
    unchanged when writing (see _with_label).
    Sent together with the number, the model translates 4 Experiments as "第四 实验", writes B.3.2 as "B.三.二" and
    1. as "一，".
    Numbers not ending with a period (2 samples, 1.5 times), single capital letters and Roman numerals can also start
    a sentence of body text (I. Goodfellow), so these are peeled only from headings and short lines that are not
    sentences."""
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
    # Bold runs from the number into the heading text ({b}17.2.1 The Encoder{/b}): close it on the number side and
    # reopen it on the text side
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
    """The leading span (Figure 2.4 of a caption) has a different font from the main text. Color does not count:
    figure numbers in body text are often blue links."""
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
    """Everything prepared before translation: the source document, blocks, per-page geometry, and the text sent for
    translation arranged by unit."""

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
    labels: list[str] = field(default_factory=list)  # Number peeled off the start of each unit, see _peel_number
    notes: list[str] = field(default_factory=list)  # Note on where each unit sits in the book, see _index_notes
    aparts: list[bool] = field(default_factory=list)  # Display formula or example between its blocks, see _seam_apart


def _plan_document(
    src: Path,
    src_doc: pymupdf.Document,
    runner,
    layout_model: str,
    formula_model: FormulaModel | None = None,
    boxes: ModelBoxes | None = None,
) -> _PdfPlan:
    # Reading lines, tables and text page by page takes most of the preparation time, so progress is reported per
    # page
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
    # A pure scan yields no blocks. Files with blocks need no page-by-page check of the text layer
    if not blocks and _scan_status(src_doc) == "scanned":
        raise ValueError(
            "此 PDF 为没有文本层的扫描件，请先使用 Acrobat、ABBYY 或 WPS 的 OCR 功能识别文字，再上传识别后的文件"
        )
    if not blocks:
        raise ValueError("未提取到可用的文字：页面可能为图片，或 OCR 文本层质量过低")
    toc = src_doc.get_toc()  # The output file keeps the original bookmarks
    # The title in PDF metadata is often something like "Microsoft Word - xxx.docx"; it is not used when it looks
    # like a file name
    meta_title = ((src_doc.metadata or {}).get("title") or "").strip()
    if re.search(r"\.(docx?|pdf|tex|indd)$|^untitled$", meta_title, re.I):
        meta_title = ""
    title = meta_title or src.stem
    runner.set_title(title)
    median = _median_size(blocks)
    # Columns and obstacles come before roles and before merging across pages: text inside figures and captions are
    # recognized by these boxes
    rects_by_page: dict[int, list[pymupdf.Rect]] = {}
    for b in blocks:
        rects_by_page.setdefault(b.page, []).append(b.rect)
    geometries = [
        PageGeometry.from_page(src_doc[i], rects_by_page.get(i, []), frames[i], scans[i])
        for i in range(src_doc.page_count)
    ]
    detector = layout_detector(src_doc, layout_model, boxes)
    # With the layout model enabled, run inference page by page and report progress per page, then return to the
    # "整理段落" stage
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
    # Running headers do not go into the outline; marginal notes and table of contents entries are not headings
    kinds = [
        "h2" if role is Role.HEADING and not edge and not block.side_note and block.leader is None else "p"
        for role, edge, block in zip(roles, edges, blocks)
    ]
    cell_ids = {id(block) for block in blocks if _table_cell(block, geometries[block.page])}
    # The index at the back of the book keeps the original: entries are sorted alphabetically in the original and the
    # page numbers point to pages of the original, so a translated index could not be looked up and would not match
    # the page numbers
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
    # A table of contents entry ends at its leader and an index entry ends at its page number; neither is merged
    # with entries in the next column or on the next page
    halt_ids = {
        id(b)
        for b, role in zip(blocks, roles)
        if role is not Role.BODY or b.side_note or b.leader is not None or b.entry
    } | margin_ids
    aside_ids = {id(b) for b, role in zip(blocks, roles) if role in (Role.CAPTION, Role.FIGURE)}
    # Merging across pages still treats table cells as body text: when a cell is at the bottom of a column, nothing
    # is joined from the next page, and the table header above the cell or text in a figure cannot move up to become
    # the end of the page
    # Example sentences across pages: the half sentence continued on the next page also keeps the original. First
    # let examples join like body text, and find the blocks joined after them
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
        # A book has tens of thousands of units, which takes several seconds
        if unit_no % _REPORT_UNITS == 0:
            runner.report_stage("准备译文", unit_no, len(units), "段")
        # When the whole unit is skipped, no placeholders or italic marks are added. A marked string differs from the
        # original, and writing would erase and reflow the original text, losing the hanging indents and line-end
        # hyphenation of the bibliography.
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
            # Nothing left to translate after placeholders: a standalone formula line with variables in body italic.
            # Like skipped blocks it keeps the original text; erasing and reflowing would shrink a two-line formula as
            # one tall formula
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


# An entry is a sub-entry only when indented this much (multiple of font size) from the left edge of entries in
# its column
_SUBENTRY_INDENT = 0.4
# Maximum number of sub-entries listed in a note
_NOTE_CHILDREN = 3


def _index_notes(units: list[list[TextBlock]], texts: list[str], skips: list[bool]) -> list[str]:
    """An index entry sent for translation on its own has no context: log becomes "日志", and divergences under MT
    becomes "散度". Each entry gets a one-sentence note: it is an entry of the index at the back of the book, which
    entry a sub-entry is listed under, and the first few sub-entries of an entry that has them.
    Sub-entries are recognized by indent: an entry indented relative to the leftmost entry of its column belongs under
    the nearest preceding entry with a smaller indent."""
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
