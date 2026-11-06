"""Recognize the formula runs of a line: absorb the glue, scripts and ordinals around them and join formulas
wrapped over lines."""

import re
import struct
import unicodedata
from dataclasses import replace

import pymupdf

from .pdf_blocks import (
    _span_is_math,
)
from .pdf_flow import (
    Emphasis,
)
from .pdf_math import _glyph_box, _prose_units
from .pdf_paragraphs import _OPERATOR_NAMES, _is_formula_char
from .pdf_rejoin import _ITALIC_FONT_RE, _small_band
from .pdf_rows import _text_box
from .pdf_runs import (
    TextRun,
)
from .pdf_split_lines import MONO_FONT_RE
from .pdf_types import PdfDict


def _is_symbol_span(text: str) -> bool:
    """The whole span in a text font consists of Greek letters or math symbols. Pure accents are left to accent
    absorption and are not treated as a formula."""
    stripped = text.strip()
    if not stripped or all(unicodedata.category(c) in ("Sk", "Mn") for c in stripped):
        return False
    return all(_is_formula_char(c) for c in stripped)


def _smaller_prose_line(spans: list[PdfDict], main_size: float) -> bool:
    """The whole line is smaller than the largest heading in the block and is a complete sentence. That is body text
    under the heading, not a script."""
    visible = [span for span in spans if str(span.get("text") or "").strip()]
    if not visible or main_size <= 0:
        return False
    if max(float(span.get("size") or 0) for span in visible) >= main_size * 0.9:
        return False
    return _prose_units("".join(str(span.get("text") or "") for span in visible)) >= 4


def _light_text(span: PdfDict) -> bool:
    color = int(span.get("color") or 0)
    red, green, blue = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return bool(span["text"].strip()) and 0.299 * red + 0.587 * green + 0.114 * blue >= 220


def _is_formula_span(span: PdfDict, main_size: float, *, prose_line: bool = False, code: bool = False) -> bool:
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


def _formula_markup(run_spans: list[PdfDict], main_size: float) -> str:
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


def _letter_formula_runs(span_lines: list[list[PdfDict]], runs: list[PdfDict]) -> list[PdfDict]:
    """Single italic letters in a text font mixed with brackets and equals signs that the whole-text font pass marked
    as formula (P(y = 1), f(x)): P and y are body italics and the brackets and equals signs are each isolated
    punctuation, so none of them is a formula alone. A contiguous sequence on one line that contains both punctuation
    marked as formula and single italic letters is an inline formula. Digits inside it belong to the formula; digits
    at either end are not taken. A digit after an equals sign is taken too (y = 1). Italic words inside the formula's
    brackets and symbols already recognized are part of the sequence (to, | and want in P(to|want)), and formulas the
    sequence covers are merged into it. Returns the new sequences; covered formulas are removed from runs"""
    taken = {id(span) for run in runs for span in run["spans"]}

    def math_mark(span: PdfDict) -> bool:
        return bool(span.get("math")) and _LONE_PUNCT_RE.match(span["text"].strip() or " ") is not None

    def italic(span: PdfDict) -> bool:
        return bool(int(span.get("flags") or 0) & 2)

    def letter(span: PdfDict) -> bool:
        return italic(span) and _VARIABLE_BASE_RE.match(span["text"].strip()) is not None

    def edge(span: PdfDict) -> bool:
        return math_mark(span) or letter(span) or id(span) in taken

    def assigned(window: list[PdfDict]) -> bool:
        """The sequence ends with a number directly preceded by an operator marked as formula (= 1, < 0)"""
        words = [span for span in window if span["text"].strip()]
        return (
            len(words) >= 2
            and words[-1]["text"].strip().isdigit()
            and math_mark(words[-2])
            and words[-2]["text"].strip() in _ASSIGN_OPS
        )

    found: list[PdfDict] = []
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
    span: PdfDict,
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


def _has_text_base(run: PdfDict, span_lines: list[list[PdfDict]], members: set[int], block_size: float) -> bool:
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


def _absorb_script_bases(runs: list[PdfDict], span_lines: list[list[PdfDict]], block_size: float) -> None:
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
        candidates: list[tuple[int, int, PdfDict]] = []
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


def _absorb_inner_blanks(runs: list[PdfDict], span_lines: list[list[PdfDict]]) -> None:
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


def _is_math_glue(span: PdfDict) -> bool:
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


def _attach_span(run: PdfDict, span: PdfDict, line_no: int, index: int) -> None:
    run["spans"].append(span)
    run["bbox"] |= pymupdf.Rect(span["bbox"])
    run["max_size"] = max(float(run["max_size"]), float(span.get("size") or 0))
    cur_line, start, end = run["pos"]
    if line_no == cur_line:
        run["pos"] = (cur_line, min(start, index), max(end, index))


def _gap_is_clear(
    span: PdfDict,
    run: PdfDict,
    span_lines: list[list[PdfDict]],
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
    span: PdfDict,
    runs: list[PdfDict],
    span_lines: list[list[PdfDict]],
    block_size: float,
    taken: set[int],
) -> PdfDict | None:
    box = pymupdf.Rect(span["bbox"])
    # Vertical comparison uses character height: the font boxes of the CMSY minus sign and vertical bar reach the next
    # line, and P( at the start of the next line would attach to the formula on the line above
    text = _text_box(span)
    best: tuple[float, PdfDict] | None = None
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
    span: PdfDict,
    run: PdfDict,
    span_lines: list[list[PdfDict]],
    taken: set[int],
) -> list[tuple[int, int, PdfDict]]:
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
    found: list[tuple[int, int, PdfDict]] = []
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
    runs: list[PdfDict],
    span_lines: list[list[PdfDict]],
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
        still: list[tuple[int, int, PdfDict]] = []
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


def _runs_into_word(line: list[PdfDict], index: int, taken: set[int]) -> bool:
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


def _ordinal_reading(run: PdfDict, block_size: float) -> str | None:
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


def _release_ordinals(runs: list[PdfDict], block_size: float) -> tuple[dict[int, str], set[int]]:
    """Ordinals are not turned into formula images; they are sent for translation as i-th so the model can translate
    them as "第 i 个"."""
    kept: list[PdfDict] = []
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


def _line_x0(spans: list[PdfDict]) -> float:
    xs = [pymupdf.Rect(s["bbox"]).x0 for s in spans if s.get("bbox") is not None]
    return min(xs) if xs else 0.0


def _span_box(span: PdfDict) -> pymupdf.Rect:
    raw = span.get("bbox")
    return pymupdf.Rect(raw) if raw is not None else pymupdf.Rect(0, 0, 0, 0)


def _union_box(a: pymupdf.Rect, b: pymupdf.Rect) -> pymupdf.Rect:
    """Same as a | b: empty boxes are ignored and the result is stored in single precision as MuPDF does.
    It runs once per span, and Rect's | copies first and then checks each argument, which is slow."""
    if a.is_infinite or b.is_infinite:
        union: pymupdf.Rect = a | b
        return union
    if b.is_empty:
        return pymupdf.Rect(a)
    if a.is_empty:
        return pymupdf.Rect(b)
    box = (min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))
    return pymupdf.Rect(struct.unpack("4f", struct.pack("4f", *box)))


def _same_text_style(pending: TextRun, span: PdfDict, emphasis: Emphasis) -> bool:
    return (
        pending.emphasis == emphasis
        and pending.font == str(span.get("font") or "")
        and abs(pending.size - float(span.get("size") or 0)) < 0.1
    )


def _extend_text_run(pending: TextRun | None, span: PdfDict, emphasis: Emphasis, line: int) -> TextRun:
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


def _line_edge(span_lines: list[list[PdfDict]], line_no: int, side: str) -> float:
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


def _only_this_run(run: PdfDict, span_lines: list[list[PdfDict]], head: bool) -> bool:
    line_no, start, end = run["pos"]
    spans = span_lines[line_no][:start] if head else span_lines[line_no][end + 1 :]
    return not any(str(span.get("text") or "").strip() for span in spans)


def _continues_wrapped_formula(
    prev: PdfDict,
    nxt: PdfDict,
    span_lines: list[list[PdfDict]],
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


def _ends_row(run: PdfDict, span_lines: list[list[PdfDict]]) -> bool:
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


def _run_text_box(run: PdfDict) -> pymupdf.Rect:
    box = pymupdf.Rect()
    for span in run["spans"]:
        if str(span.get("text") or "").strip() and "bbox" in span:
            box |= _text_box(span)
    return box if not box.is_empty else pymupdf.Rect(run["bbox"])


def _fraction_pair(a: PdfDict, b: PdfDict, block_size: float) -> bool:
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


def _run_baseline(run: PdfDict) -> float:
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
    runs: list[PdfDict],
    span_lines: list[list[PdfDict]],
    block_size: float,
) -> list[PdfDict]:
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
    joined: list[PdfDict] = []
    for run in runs:
        if joined and _continues_wrapped_formula(joined[-1], run, span_lines, block_size):
            pieces = [joined[-1], *joined[-1].get("extras", []), run]
            if not line_width or sum(pymupdf.Rect(piece["bbox"]).width for piece in pieces) <= line_width:
                joined[-1].setdefault("extras", []).append(run)
                continue
        joined.append(run)
    return joined
