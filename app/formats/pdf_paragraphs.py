"""Join the lines and fragments of a page that continue one paragraph: wrapped lines, line tails, same-line
sentences, caption fragments and table cells."""

import re
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import replace

import pymupdf

from .pdf_blocks import (
    _AFFIL_MARK_RE,
    CJK_RE,
    TextBlock,
    _diagramish,
    _ends_sentence,
    _is_side_label,
    _join_lines,
    _span_is_math,
    _visual_lines,
)
from .pdf_drawings import RuledTable
from .pdf_frame import TextFrame
from .pdf_layout import PageGeometry, cell_containing
from .pdf_math import (
    _prose_units,
    _script_seam,
)
from .pdf_rows import (
    _INDEX_REFS_ONLY_RE,
    _LIST_RUN,
    _breaks_early,
    _same_opening,
    _stacked_rows,
    _text_box,
    _voted_right,
)
from .pdf_split_lines import _OPERATOR_RE, NO_TEXT_RE, _label_lead, _typewriter_line
from .pdf_types import PdfDict


def _verbatim_block(block: TextBlock) -> bool:
    """The whole block is verbatim text in a monospaced font (example dialogue, code), set as separate
    lines."""
    return bool(block.span_lines) and all(_typewriter_line(spans) for spans in block.span_lines or [])


def _verbatim_seam(host: TextBlock, guest: TextBlock) -> bool:
    """One side is verbatim monospaced text set on its own, and the line touching it on the other side is body
    text. A verbatim line has no sentence-final punctuation, yet the lowercase body text below it is not its
    continuation. Code inside body text that happens to fall at the start or end of a line is joined as
    usual."""
    if not host.span_lines or not guest.span_lines:
        return False
    return (
        _verbatim_block(host)
        and not _typewriter_line(guest.span_lines[0])
        or _verbatim_block(guest)
        and not _typewriter_line(host.span_lines[-1])
    )


def _is_formula_char(c: str) -> bool:
    """Per-character formula test: Greek letters, math symbols and modifiers, Private Use Area."""
    if 0x370 <= ord(c) <= 0x3FF:
        return True
    return unicodedata.category(c) in ("Sm", "Sk", "Mn", "Co")


def _math_chars(spans: list[PdfDict]) -> int:
    """Number of formula characters in a block: all characters in math fonts plus formula characters in body
    fonts."""
    n = 0
    for s in spans:
        if _span_is_math(s):
            n += len(s["text"])
        else:
            n += sum(1 for c in s["text"] if _is_formula_char(c))
    return n


def _caption_label(text: str) -> bool:
    return bool(_CAPTION_LABEL_RE.match(text.strip()))


def _overlaps_line(block: TextBlock, other: TextBlock) -> bool:
    lines = other.line_rects or [other.rect]
    height = block.rect.height or block.size
    for line in lines:
        overlap = min(block.rect.y1, line.y1) - max(block.rect.y0, line.y0)
        if overlap > min(height, line.height) * 0.5:
            return True
    return False


def _blocks_share_column(
    left: TextBlock,
    right: TextBlock,
    geometries: dict[int, PageGeometry] | None,
) -> bool:
    """Even with a gap narrower than the column gap, confirm that the two blocks are in the same column.
    Without column information, rely on the gap alone."""
    if not geometries or left.page not in geometries:
        return True
    geo = geometries[left.page]
    if len(geo.columns) < 2:
        return True
    return geo.index_of(left.rect) == geo.index_of(right.rect) and not geo.spans_columns(left.rect | right.rect)


def _line_spans(block: TextBlock) -> list[list[PdfDict]]:
    """Spans per line, one entry for each of line_rects. When they do not match, each line gets an empty list
    and no decision is made by span."""
    rows = len(block.line_rects) or 1
    if block.span_lines and len(block.span_lines) == rows:
        return [list(line) for line in block.span_lines]
    return [[] for _ in range(rows)]


def _shares_visual_row(
    left: TextBlock,
    right: TextBlock,
    geometries: dict[int, PageGeometry] | None,
) -> bool:
    """Whether the two blocks sit on the same baseline and are adjacent horizontally.

    Line boxes are used, not the union of the blocks. A short heading on the left sits beside only the first
    body line; the union would cover it and the paragraph would swallow the heading. A column gap is at least
    12pt, so a 6pt gap cannot reach another column.
    """
    left_lines = list(zip(left.line_rects or [left.rect], _line_spans(left)))
    right_lines = list(zip(right.line_rects or [right.rect], _line_spans(right)))
    for a, a_spans in left_lines:
        for b, b_spans in right_lines:
            overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
            if overlap <= min(a.height, b.height) * 0.5 or _stacked_rows(a_spans, b_spans):
                continue
            horizontal = min(a.x1, b.x1) - max(a.x0, b.x0)
            gap = b.x0 - a.x1 if b.x0 >= a.x0 else a.x0 - b.x1
            if horizontal > 0 or (
                0 <= gap < 6
                and _blocks_share_column(left, right, geometries)
                and not (b.x0 >= a.x0 and _script_seam(a_spans, b_spans))
            ):
                return True
    return False


def _merge_visual_lines(
    blocks: list[TextBlock],
    geometries: dict[int, PageGeometry] | None = None,
) -> list[TextBlock]:
    """Merge adjacent blocks that are really one visual line (PyMuPDF splits a line into two blocks where the
    font switches at a formula).

    Unmerged, the two translations would be written into overlapping rectangles on top of each other.
    """
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            if (
                p.page == b.page
                and p.clip == b.clip
                and not (p.side_note or b.side_note)
                and _shares_visual_row(p, b, geometries)
            ):
                p.line_rects.extend(b.line_rects)
                if p.span_lines and b.span_lines:
                    p.span_lines.extend(b.span_lines)
                else:
                    p.span_lines = None
                p.text = _join_lines([p.text, b.text])
                p.rect |= b.rect
                continue
        out.append(b)
    return out


# A caption label may have no colon: "图 2", "Fig. 1."
_CAPTION_LABEL_RE = re.compile(
    r"^(?:(?:Figure|Fig\.?|Table|Tab\.?|Scheme|Plate)\s*\d+|(?:图|表)\s*\d+)\s*[:.。：]?\s*$",
    re.I,
)


_OPERATOR_NAMES = (
    "arg|max|min|log|ln|sin|cos|tan|cot|sec|csc|exp|lim|sup|inf|det|dim|ker|gcd|mod|sgn|deg|hom|Pr|arcsin|sinh|cosh"
)


def _donate_punct_rects(prev: TextBlock, line_rects: list[pymupdf.Rect]) -> None:
    """When a punctuation fragment sits at the end of a line of the previous block, add its rectangles to that
    line for masking.

    The gap is measured from that line's x1, not from the union of the block: the union would erase the
    formula on the right, and formula lines are not redrawn.
    The fragment must be near the end of the line: tick numbers on a figure in the neighbouring column at the
    same height do not belong to this line, and merging them would stretch the write box into that column.
    """
    if not line_rects or not prev.line_rects:
        return
    fragment = line_rects[0]
    height = fragment.y1 - fragment.y0
    if height <= 0:
        return
    for tail in prev.line_rects:
        overlap = min(tail.y1, fragment.y1) - max(tail.y0, fragment.y0)
        if overlap > 0.5 * height and -height < fragment.x0 - tail.x1 < 2:
            prev.line_rects.extend(line_rects)
            return


def _clip_key(cell: pymupdf.Rect | None) -> tuple[float, float, float, float] | None:
    if cell is None:
        return None
    return (cell.x0, cell.y0, cell.x1, cell.y1)


def _split_table_blocks(blocks: list[TextBlock], tables: tuple[RuledTable, ...]) -> list[TextBlock]:
    """Split cells side by side in one line into separate blocks. A neighbouring cell that starts in lowercase
    would not be cut by the hard break rule."""
    if not tables:
        return blocks
    out: list[TextBlock] = []
    for block in blocks:
        out.extend(_split_one_table_block(block, tables))
    return out


def _split_one_table_block(block: TextBlock, tables: tuple[RuledTable, ...]) -> list[TextBlock]:
    lines = block.span_lines or []
    if len(lines) != len(block.line_rects):
        block.clip = _clip_key(cell_containing(block.rect, tables))
        return [block]
    grouped: dict[tuple[float, float, float, float] | None, list[tuple[pymupdf.Rect, list[PdfDict]]]] = {}
    order: list[tuple[float, float, float, float] | None] = []
    for line_rect, spans in zip(block.line_rects, lines):
        by_cell: dict[tuple[float, float, float, float] | None, list[PdfDict]] = {}
        cell_order: list[tuple[float, float, float, float] | None] = []
        for span in spans:
            if not str(span.get("text", "")).strip():
                # In justified lines the space between words is often a span of its own; it goes into the same
                # cell as the piece before it, and without it the words run together
                if cell_order:
                    by_cell[cell_order[-1]].append(span)
                continue
            bbox = span.get("bbox")
            key = _clip_key(cell_containing(_text_box(span), tables) if bbox else None)
            if key not in by_cell:
                cell_order.append(key)
                by_cell[key] = []
            by_cell[key].append(span)
        if not cell_order:
            key = _clip_key(cell_containing(line_rect, tables))
            _remember_cell(grouped, order, key, line_rect, [])
            continue
        if len(cell_order) == 1:
            _remember_cell(grouped, order, cell_order[0], line_rect, by_cell[cell_order[0]])
            continue
        for key in cell_order:
            box = pymupdf.Rect()
            for span in by_cell[key]:
                if str(span.get("text", "")).strip():
                    box |= pymupdf.Rect(span["bbox"])
            _remember_cell(grouped, order, key, box, by_cell[key])
    if len(order) <= 1:
        block.clip = order[0] if order else _clip_key(cell_containing(block.rect, tables))
        return [block]
    parts: list[TextBlock] = []
    for key in order:
        pieces = grouped[key]
        rect = pymupdf.Rect()
        for line_rect, _spans in pieces:
            rect |= line_rect
        text = _join_lines(["".join(str(span.get("text", "")) for span in spans) for _rect, spans in pieces])
        if not text.strip():
            continue
        parts.append(
            replace(
                block,
                rect=rect,
                line_rects=[line_rect for line_rect, _spans in pieces],
                text=text,
                span_lines=[spans for _rect, spans in pieces],
                clip=key,
                runs=[],
            )
        )
    return parts or [block]


def _remember_cell(
    grouped: dict[tuple[float, float, float, float] | None, list[tuple[pymupdf.Rect, list[PdfDict]]]],
    order: list[tuple[float, float, float, float] | None],
    key: tuple[float, float, float, float] | None,
    line_rect: pymupdf.Rect,
    spans: list[PdfDict],
) -> None:
    if key not in grouped:
        order.append(key)
        grouped[key] = []
    grouped[key].append((line_rect, spans))


_LIST_ITEM_RE = re.compile(r"^\d+[\.,]\s+[A-Za-z]")


def _numbered_list_item(text: str) -> bool:
    """'5. part (0.2M)' has a low share of letters but is still a list item, not a formula line."""
    stripped = text.strip()
    if not _LIST_ITEM_RE.match(stripped) or len(stripped) > 80:
        return False
    return any(len(word) >= 3 and word.isalpha() for word in re.findall(r"[A-Za-z]+", stripped))


def _index_refs_tail(block: TextBlock, rect: pymupdf.Rect) -> bool:
    """rect is a line of page numbers wrapped from the last line of block: right below the last line, starting
    between its left and right edges."""
    last = block.line_rects[-1]
    return (
        -block.size * 0.3 <= rect.y0 - last.y1 <= block.size * 0.5 and last.x0 - block.size * 0.5 <= rect.x0 < last.x1
    )


def _wraps_from_above(
    blocks: list[TextBlock],
    pno: int,
    rect: pymupdf.Rect,
    text: str,
    rows: Sequence[pymupdf.Rect],
    frame: TextFrame | None,
    index_page: bool = False,
) -> bool:
    """A math line that starts with a lowercase body word and continues from the left edge of the body line
    above (and W^O ∈ ℝ^{hd_v×d_model}.) is wrapped from the sentence above, not a display expression: the line
    above runs to the right edge of the column without finishing a sentence, and this line sits right below
    it.
    When the line above ends with half a word broken by a hyphen, this line is the second half of that word. On
    an index page, a line indented deeper than the line above is a continuation of the previous entry (pipe in
    regular expressions | as, 51).
    The last line of a centered caption (significant (t-test with p ≤ 0.01).) shares the center line of the
    line above.
    Stacked expression lines (e_t = Ex_t, h_t = g(...)) do not reach the right edge.
    An index entry stops at a comma and its page numbers wrap to the next line, in the next block: a line of
    only page numbers right below is joined as well."""
    page_blocks = [block for block in blocks if block.page == pno and block.line_rects]
    if index_page and _INDEX_REFS_ONLY_RE.match(text.strip()):
        return any(block.text.rstrip().endswith(",") and _index_refs_tail(block, rect) for block in page_blocks[-1:])
    worded = re.match(r"[a-z]{2,}\b", text.lstrip()) is not None
    for block in reversed(page_blocks):
        last = block.line_rects[-1]
        if rect.y0 - last.y1 > 0.6 * block.size:
            # Blocks are in reading order; earlier ones are all higher up or in another column
            continue
        # What wraps is one line: with more lines stacked below, it is a separately set expression. The
        # stacked numerator and denominator of an inline fraction are taller than one line
        if (
            rect.height > block.size * (1.6 if worded else 2.5)
            or rect.y0 - last.y1 < -0.5 * block.size
            or _ends_sentence(block.text)
        ):
            continue
        # The left-aligned line above runs to the right edge of the column. In a centered paragraph each line
        # is a block, and a line above as wide as the line above it is a full line
        left = (
            abs(block.rect.x0 - rect.x0) <= block.size * 0.5
            and last.x1 >= _column_right(block.rect.x0, block.size, rows, frame) - block.size
        )
        if not worded:
            # A line not starting with a word is accepted only as an inline fraction with stacked numerator
            # and denominator, left-aligned and right below the line above (the odds ratio | p/(1−p):).
            # A display expression has space above it. Numbers and symbols one line tall belong to tables and
            # figures
            if left and rect.height > block.size * 1.6 and rect.y0 - last.y1 <= 0.3 * block.size:
                return True
            continue
        centered = abs((last.x0 + last.x1) / 2 - (rect.x0 + rect.x1) / 2) <= block.size and any(
            0 <= last.y0 - line.y1 <= block.size
            and abs(line.x0 - last.x0) <= block.size
            and abs(line.x1 - last.x1) <= block.size
            for other in page_blocks
            for line in other.line_rects
            if line is not last
        )
        # The line above stops at half a word broken by a hyphen (expres- | sions, 50, 65): it continues
        # whatever the indent.
        # A continuation line of an index entry has a hanging indent, and the remaining half word with its page
        # numbers has few letters and looks like a line of page numbers
        broken = (
            re.search(r"[A-Za-z]-$", block.text.rstrip()) is not None
            and block.rect.x0 - block.size * 0.5 <= rect.x0 < last.x1
        )
        hanging = index_page and last.x0 + block.size * 0.5 < rect.x0 < last.x1
        if left or centered or broken or hanging:
            return True
    return False


def _column_right(x0: float, size: float, rows: Sequence[pymupdf.Rect], frame: TextFrame | None) -> float:
    """Right edge of the column of a block starting at x0: 40% or more of the lines on this page starting at
    the same left edge end at the same x (justified).
    Ragged lines such as indexes and code give no such edge, and the right edge of the text frame is used."""
    edge = _voted_right(x0, size, rows)
    if edge is not None:
        return edge
    return frame.x1 if frame is not None else float("inf")


def _untranslatable(spans: list[PdfDict], text: str, min_prose: int = 3, displayed: bool = False) -> bool:
    """Lines that are pure symbols, pure formula, or have almost no letters are not sent for translation and
    keep the original text.

    Only "pure formula lines" are skipped: many math characters and almost no body words. Three body words
    outside the formula make a sentence (If ... then ... Otherwise ...), which is translated, with the formula
    going through a placeholder. Letters inside a formula are not words:
    letters recognized from math symbols in bitmap fonts are garbage.

    A line set on its own with an operator and not a single body word is also an expression. In books that set
    variables in body italics (Q = H_dec W^Q, ŷ = argmax_{w∈V} P(w|w_<t)), formula characters cannot be
    recognized by font.
    """
    if NO_TEXT_RE.match(text):
        return True
    if displayed and _OPERATOR_RE.search(text) and not _prose_words(spans, text):
        return True
    total = sum(len(span["text"]) for span in spans)
    # Blank spans have been filtered out, so join with spaces, or It and follows would become one word
    prose = _prose_units(" ".join(span["text"] for span in spans if not _span_is_math(span)))
    if _math_chars(spans) > total * 0.3 and prose < min_prose:
        return True
    # Enough body words make a sentence (Therefore, ... It follows), translated however much formula it has and
    # however low its share of letters.
    # The few-letters rule is meant for labels and lines of numbers
    if prose >= max(min_prose, 3):
        return False
    letters = sum(c.isalpha() for c in text)
    return (letters < 2 or letters < len(text) * 0.4) and not _numbered_list_item(text)


_MATH_NAMES = frozenset(_OPERATOR_NAMES.split("|")) | {"argmax", "argmin", "softmax", "tanh", "sigmoid"}


def _prose_words(spans: list[PdfDict], text: str) -> list[str]:
    """Words in the line separated by spaces on both sides and written in a body font (with lowercase letters,
    two or more letters).
    Words written together with symbols (H_dec, argmax_{char∈Alphabet}), all-capital matrix names (HU) and
    operator names (argmax, log) do not count."""
    upright = set()
    for span in spans:
        if not _span_is_math(span):
            upright.update(re.findall(r"[A-Za-z]{2,}", span["text"]))
    words = []
    for token in text.split():
        word = token.strip(",.;:!?\"'“”‘’()[]{}")
        if (
            len(word) >= 2
            and word.isascii()
            and word.isalpha()
            and word in upright
            and any(c.islower() for c in word)
            and word not in _MATH_NAMES
        ):
            words.append(word)
    return words


def _merge_cells(blocks: list[TextBlock], unchecked: set[int]) -> list[TextBlock]:
    """Merge the blocks in one cell into one block, ordered top to bottom and left to right. A cell is one
    translation unit.

    PyMuPDF splits the text of a cell into several blocks (line numbers one size smaller, formulas in a block
    of their own). Sent for translation separately, each half sentence has too few body words and is taken as
    a formula, leaving the English. Blocks in unchecked have not been tested for formulas yet; the test runs
    after merging.
    A cell is one complete sentence and is translated when it has a single body word (for k = 1 ... K do);
    cells of pure formula or pure numbers keep the original text.
    """
    cells: dict[tuple[int, tuple[float, float, float, float] | None], list[TextBlock]] = {}
    for block in blocks:
        if block.clip is not None:
            cells.setdefault((block.page, block.clip), []).append(block)
    out: list[TextBlock] = []
    done: set[tuple[int, tuple[float, float, float, float] | None]] = set()
    for block in blocks:
        key = (block.page, block.clip)
        if block.clip is None:
            if id(block) in unchecked and _untranslatable(_block_spans(block), block.text):
                continue
            out.append(block)
            continue
        if key in done:
            continue
        done.add(key)
        parts = [
            part
            for part in _reading_sorted(cells[key])
            if not (id(part) in unchecked and _displayed_in_cell(part, cells[key]))
        ]
        for group in _cell_groups(parts):
            merged = group[0] if len(group) == 1 else _joined_cell(group)
            if any(id(part) in unchecked for part in group) and _untranslatable(_block_spans(merged), merged.text, 1):
                continue
            out.append(merged)
    return out


def _cell_groups(parts: list[TextBlock]) -> list[list[TextBlock]]:
    """Adjacent pieces in a cell are one sentence. A piece two ems away, or a blank line apart, is another
    sentence: when the boxes of a flowchart are taken as cells, the node labels in a box must not be merged
    into one sentence."""
    groups: list[list[TextBlock]] = []
    for part in parts:
        near = [group for group in groups if any(_close_in_cell(part, other) for other in group)]
        if not near:
            groups.append([part])
            continue
        near[0].append(part)
        for group in near[1:]:
            near[0].extend(group)
            groups.remove(group)
    return [_reading_sorted(group) if len(group) > 1 else group for group in groups]


def _close_in_cell(a: TextBlock, b: TextBlock) -> bool:
    size = max(a.size, b.size)
    dx = max(a.rect.x0 - b.rect.x1, b.rect.x0 - a.rect.x1, 0)
    dy = max(a.rect.y0 - b.rect.y1, b.rect.y0 - a.rect.y1, 0)
    return dx <= size * 2 and dy <= size


def _displayed_in_cell(part: TextBlock, parts: list[TextBlock]) -> bool:
    """A self-contained expression in a cell: on a line of its own and indented more than two ems from the
    text in the cell (a centered formula under an algorithm step), or more than two ems from the text on the
    same row (the expression of an algorithm step and the comment to its right).
    It keeps the original text and is not merged into the sentence: once merged, letters such as E and Var are
    translated as words, and the display formula is squeezed to one line height."""
    others = [other for other in parts if other is not part]
    if not others or not _untranslatable(_block_spans(part), part.text):
        return False
    beside = [
        other
        for other in others
        if min(other.rect.y1, part.rect.y1) - max(other.rect.y0, part.rect.y0)
        > min(other.rect.height, part.rect.height) * 0.35
    ]
    if beside:
        # A line number is not a sentence either, but it is not an expression, so it stays in this step
        if not any(_span_is_math(span) for span in _block_spans(part)):
            return False
        texts = [other for other in beside if not _untranslatable(_block_spans(other), other.text)]
        return bool(texts) and all(
            max(other.rect.x0 - part.rect.x1, part.rect.x0 - other.rect.x1) > part.size * 2 for other in texts
        )
    left = min(other.rect.x0 for other in others)
    return part.rect.x0 - left > part.size * 2.5


def _reading_sorted(parts: list[TextBlock]) -> list[TextBlock]:
    """Pieces on the same row are ordered left to right, different rows top to bottom. The box of a smaller
    line number has a slightly lower top, so box tops cannot be used for ordering."""
    rows: list[list[TextBlock]] = []
    for part in sorted(parts, key=lambda item: (item.rect.y0 + item.rect.y1) / 2):
        center = (part.rect.y0 + part.rect.y1) / 2
        if rows:
            last = rows[-1][-1]
            if abs(center - (last.rect.y0 + last.rect.y1) / 2) <= max(part.size, last.size) * 0.5:
                rows[-1].append(part)
                continue
        rows.append([part])
    return [part for row in rows for part in sorted(row, key=lambda item: item.rect.x0)]


def _joined_cell(parts: list[TextBlock]) -> TextBlock:
    main = max(parts, key=lambda part: len(part.text))
    rect = pymupdf.Rect()
    for part in parts:
        rect |= part.rect
    has_spans = all(part.span_lines for part in parts)
    return replace(
        main,
        rect=rect,
        line_rects=[line for part in parts for line in part.line_rects],
        text=_join_lines([part.text for part in parts]),
        span_lines=[line for part in parts for line in part.span_lines or []] if has_spans else None,
        runs=[],
    )


def _block_spans(block: TextBlock) -> list[PdfDict]:
    return [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]


def _line_text(span_lines: list[list[PdfDict]], index: int) -> str:
    return "".join(str(span.get("text") or "") for span in span_lines[index])


def _block_slice(block: TextBlock, start: int, end: int) -> TextBlock:
    lines = block.line_rects[start:end]
    spans = block.span_lines[start:end] if block.span_lines else None
    texts = [_line_text(spans, i) for i in range(len(spans))] if spans is not None else []
    rect = pymupdf.Rect()
    for line in lines:
        rect |= line
    return TextBlock(
        page=block.page,
        rect=rect,
        line_rects=list(lines),
        text=_join_lines(texts) if texts else block.text,
        size=block.size,
        color=block.color,
        bold=block.bold,
        span_lines=list(spans) if spans is not None else None,
        clip=block.clip,
        line_gap=block.line_gap,
        body_size=block.body_size,
    )


def _affiliation_neighbor(blocks: list[TextBlock], block: TextBlock, line: pymupdf.Rect) -> bool:
    """Whether a short affiliation line sits beside this line on its right."""
    for other in blocks:
        if other is block or other.page != block.page or other.clip is not None:
            continue
        if other.rect.height > 32 or len(other.line_rects) > 2:
            continue
        overlap = min(line.y1, other.rect.y1) - max(line.y0, other.rect.y0)
        if overlap < min(line.height, other.rect.height) * 0.5:
            continue
        gap = other.rect.x0 - line.x1
        if 0 <= gap < 40 and _AFFIL_MARK_RE.match(other.text.lstrip()):
            return True
    return False


def _split_affiliation_tails(blocks: list[TextBlock]) -> list[TextBlock]:
    """When the affiliation at the end of an author block sits beside other affiliations, split it into a
    separate block. Otherwise the translation would cover them."""
    out: list[TextBlock] = []
    for block in blocks:
        spans = block.span_lines
        if spans is None or block.clip is not None or len(block.line_rects) < 3:
            out.append(block)
            continue
        if len(spans) != len(block.line_rects):
            out.append(block)
            continue
        cut: int | None = None
        for index, line in enumerate(block.line_rects):
            if index == 0 or line.width > block.rect.width * 0.5:
                continue
            if not _AFFIL_MARK_RE.match(_line_text(spans, index).strip()):
                continue
            if _affiliation_neighbor(blocks, block, line):
                cut = index
                break
        if cut is None:
            out.append(block)
            continue
        out.append(_block_slice(block, 0, cut))
        out.append(_block_slice(block, cut, len(block.line_rects)))
    return out


def _absorb_block(p: TextBlock, b: TextBlock) -> None:
    p.text = _join_lines([p.text, b.text])
    p.rect |= b.rect
    p.line_rects.extend(b.line_rects)
    if p.span_lines and b.span_lines:
        p.span_lines.extend(b.span_lines)
    else:
        p.span_lines = None
    if len(b.text) > len(p.text) - len(b.text):  # Style follows the larger fragment
        p.size, p.color, p.bold = b.size, b.color, b.bold
    # The merged block ends at b's last line: when b is the last line of an index entry, so is the merged block
    p.entry = b.entry


def _line_lists(blocks: list[TextBlock], frames: Sequence[TextFrame | None]) -> set[int]:
    """Blocks in a list with one sentence per line (examples on a slide). Several blocks, one line each,
    aligned on the left with equal line spacing, where at more than half of the line breaks the first word of
    the next line would still fit after the line above, or every line starts with the same word: set line by
    line, not wrapped from one paragraph.
    A paragraph wraps line by line, and a word that fits on the line above is not moved to the next. Lines set
    around a figure end at the figure's edge; enough lines end at the same x that the column's right edge is
    taken at the figure's edge (see _column_right)."""
    rows_of: dict[int, list[pymupdf.Rect]] = {}
    for block in blocks:
        rows_of.setdefault(block.page, []).extend(block.line_rects or [block.rect])
    listed: set[int] = set()
    run: list[TextBlock] = []

    def close() -> None:
        if len(run) < _LIST_RUN:
            return
        frame = frames[run[0].page] if run[0].page < len(frames) else None
        if frame is None:
            return
        rows = rows_of[run[0].page]
        if _same_opening([block.text for block in run]) or _breaks_early(
            [(block.rect, block.text, block.size) for block in run],
            lambda rect, size: _column_right(rect.x0, size, rows, frame),
        ):
            listed.update(id(block) for block in run)

    for block in blocks:
        single = bool(block.line_rects) and len(_visual_lines(block)) == 1
        if run and single:
            last = run[-1]
            pitch = block.rect.y0 - last.rect.y0
            first_pitch = run[1].rect.y0 - run[0].rect.y0 if len(run) >= 2 else pitch
            if (
                block.page == last.page
                and abs(block.rect.x0 - last.rect.x0) <= last.size * 0.5
                and abs(block.size - last.size) <= last.size * 0.05
                and 0 < pitch <= last.size * 2.0
                and abs(pitch - first_pitch) <= last.size * 0.2
            ):
                run.append(block)
                continue
        close()
        run = [block] if single else []
    close()
    return listed


def _is_line_tail(block: TextBlock, earlier: list[TextBlock]) -> bool:
    """This line sits right after the end of a line of a block to its left: half a sentence, not a new
    paragraph."""
    if len(block.line_rects) > 1 or _ends_sentence(block.text) or block.side_note:
        return False
    for other in earlier:
        if other.page != block.page or other.clip != block.clip or other.side_note:
            continue
        lines = other.line_rects or [other.rect]
        height = block.rect.height or block.size
        for line in lines:
            overlap = min(block.rect.y1, line.y1) - max(block.rect.y0, line.y0)
            if overlap <= min(height, line.height) * 0.5:
                continue
            gap = block.rect.x0 - line.x1
            if -2 <= gap < 12 and block.rect.x0 > line.x0 + max(block.size * 3, 24):
                return True
    return False


def _earlier_on_page(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """For each block, the blocks before it on the same page. Half sentences are compared only with lines on
    their own page; slicing the whole book at once takes a long time for books of several hundred pages."""
    seen: dict[int, list[TextBlock]] = {}
    out: list[list[TextBlock]] = []
    for block in blocks:
        same = seen.setdefault(block.page, [])
        out.append(list(same))
        same.append(block)
    return out


def _line_tail_blocked(nxt: TextBlock, fragment: TextBlock, blocks: list[TextBlock]) -> bool:
    """Do not merge when the continuing line also touches another block. Merged, the writing would cover that
    block."""
    for other in blocks:
        if other is nxt or other is fragment or other.page != nxt.page:
            continue
        if not _overlaps_line(nxt, other):
            continue
        if other.rect.x0 < nxt.rect.x1 and other.rect.x1 > nxt.rect.x0:
            return True
    return False


def _line_tail_match(block: TextBlock, nxt: TextBlock) -> bool:
    """Only a next line that returns to the column edge and starts in lowercase continues this half
    sentence."""
    if nxt.page != block.page or nxt.clip != block.clip or nxt.side_note:
        return False
    if abs(nxt.size - block.size) > block.size * 0.2:
        return False
    if nxt.rect.x0 > block.rect.x0 - max(block.size * 3.5, 28):
        return False
    gap = nxt.rect.y0 - block.rect.y1 - block.line_gap * block.size
    if not (-0.5 * block.size <= gap < 1.5 * block.size):
        return False
    return bool(nxt.text.lstrip()[:1].islower()) and not _verbatim_seam(block, nxt)


def _line_tail_continuation(
    block: TextBlock,
    later: list[TextBlock],
    blocks: list[TextBlock],
) -> TextBlock | None:
    for nxt in later:
        if nxt.page != block.page or nxt.rect.y0 > block.rect.y1 + block.size * 2:
            return None
        if not _line_tail_match(block, nxt) or _line_tail_blocked(nxt, block, blocks):
            continue
        return nxt
    return None


def _merge_line_tails(blocks: list[TextBlock], edges: Collection[int] = ()) -> list[TextBlock]:
    """Join a half sentence on the right of a line with its continuation on the next line back at the column
    edge into one paragraph. It is not merged into the sentence on its left.
    edges are running headers and footers, used neither as half sentences nor as continuations."""
    used: set[int] = set()
    out: list[TextBlock] = []
    earlier = _earlier_on_page(blocks)
    for index, block in enumerate(blocks):
        if id(block) in used:
            continue
        if id(block) not in edges and not block.entry and _is_line_tail(block, earlier[index]):
            later = [other for other in blocks[index + 1 :] if id(other) not in edges]
            nxt = _line_tail_continuation(block, later, blocks)
            if nxt is not None and id(nxt) not in used and not nxt.entry:
                _absorb_block(block, nxt)
                used.add(id(nxt))
        out.append(block)
    return out


def _returns_to_margin(host: TextBlock, guest: TextBlock) -> bool:
    """Only when the later lines return to the host's column left edge is it the same paragraph. Affiliation
    lines side by side do not return."""
    rest = guest.line_rects[1:]
    if not rest:
        return True
    host_left = min(line.x0 for line in host.line_rects)
    rest_left = min(line.x0 for line in rest)
    return abs(rest_left - host_left) <= max(host.size * 4, 36)


def _sits_on_last_line(host: TextBlock, guest: TextBlock) -> bool:
    """The next sentence starts to the right of the last line of the previous sentence. In two-column papers
    this is still the same paragraph."""
    if host.page != guest.page or host.clip != guest.clip:
        return False
    if not host.line_rects or not guest.line_rects or host.size <= 0:
        return False
    if abs(host.size - guest.size) > host.size * 0.2:
        return False
    # Compare positions before text: blocks on a page are compared pairwise, and most pairs are ruled out here
    last = host.line_rects[-1]
    first = guest.line_rects[0]
    height = min(last.height, first.height) or 1.0
    overlap = min(last.y1, first.y1) - max(last.y0, first.y0)
    if overlap <= height * 0.5:
        return False
    gap = first.x0 - last.x1
    if not (-2 <= gap < 16):
        return False
    if first.x0 <= last.x0 + max(host.size * 3, 24):
        return False
    if _is_side_label(host) or _diagramish(host) or _diagramish(guest):
        return False
    return _returns_to_margin(host, guest)


def _continues_fragment(host: TextBlock, guest: TextBlock) -> bool:
    """The line after a half sentence at a line end returns to the column edge and starts in lowercase. It is
    joined first, even when the next sentence sits to the right on this line.

    A right half line that already reaches a sentence end is no half sentence. The otherwise. of a formula
    condition line must not pull in the body text below.
    """
    if host.page != guest.page or host.clip != guest.clip:
        return False
    if not host.line_rects or not guest.line_rects or host.size <= 0:
        return False
    if _ends_sentence(host.text):
        return False
    if abs(host.size - guest.size) > host.size * 0.2:
        return False
    last = host.line_rects[-1]
    left = min(line.x0 for line in host.line_rects)
    if last.x0 <= left + max(host.size * 3, 24):
        return False
    first = guest.line_rects[0]
    if first.x0 > last.x0 - max(host.size * 3.5, 28):
        return False
    # A half sentence at a line end wraps to the next line, and this line runs to the right edge of the column.
    # A display expression has space on both sides, and the body text below starts a new paragraph;
    # limits under an expression are short pieces of math and are still joined
    if last.x1 < first.x1 - host.size * 3 and _prose_units(guest.text) >= 4:
        return False
    gap = first.y0 - last.y1 - host.line_gap * host.size
    if not (-0.5 * host.size <= gap < 1.5 * host.size):
        return False
    head = guest.text.lstrip()[:1]
    if not (head and (head.islower() or head in "),;%,；，")):
        return False
    return not (_diagramish(guest) or _is_side_label(guest) or _verbatim_seam(host, guest))


def _merge_same_line_sentences(blocks: list[TextBlock], edges: Collection[int] = ()) -> list[TextBlock]:
    """When a sentence ends mid-line and the next sentence starts to its right on the same line, merge them
    into one paragraph laid out at the column width.

    Short bold side labels and labels inside figures are not merged. A continuation line is joined first even
    when it shares a baseline with the following sentence, which is merged after it.
    Reading order can put the left half line after the right half, so adjacent blocks alone are not enough.
    The search stays within one page. Comparing the whole book pairwise takes a long time for books of several
    hundred pages.
    edges are running headers and footers and take no part.
    """
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if id(block) not in edges:
            by_page.setdefault(block.page, []).append(block)
    absorbed: set[int] = set()
    for page_blocks in by_page.values():
        absorbed |= _merge_page_line_sentences(page_blocks)
    return [block for block in blocks if id(block) not in absorbed]


def _same_line_sentence(host: TextBlock, guest: TextBlock) -> bool:
    # Margin notes, index entries and word grid cells each stand on their own
    if host.side_note or guest.side_note or host.entry or guest.entry:
        return False
    return _sits_on_last_line(host, guest) or _continues_fragment(host, guest)


def _merge_page_line_sentences(blocks: list[TextBlock]) -> set[int]:
    """Go through the hosts on a page in order, each taking the first block that can be merged into it.
    Returns the blocks merged away.

    After a merge the search does not restart from the beginning. The earlier hosts had no block to merge,
    and only the block that just changed may join after one of them.
    """
    out = list(blocks)
    absorbed: set[int] = set()
    index = 0
    while index < len(out):
        host = out[index]
        guest_index = next(
            (other for other, guest in enumerate(out) if other != index and _same_line_sentence(host, guest)),
            None,
        )
        if guest_index is None:
            index += 1
            continue
        absorbed.add(id(out[guest_index]))
        _absorb_block(host, out[guest_index])
        del out[guest_index]
        if guest_index < index:
            index -= 1
        while True:
            earlier = next((other for other in range(index) if _same_line_sentence(out[other], out[index])), None)
            if earlier is None:
                break
            absorbed.add(id(out[index]))
            _absorb_block(out[earlier], out[index])
            del out[index]
            index = earlier
    return absorbed


def _merge_caption_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    """Merge figure and table captions broken into fragments: 'Figure 2:', 'Examples of decay' and
    'schedules.' translated separately and put back into their own small boxes wrap badly. Joined back into
    one block, the sentence is translated whole and laid out on one line.
    The rules are conservative and join only three kinds: a) a fragment right after a short label on the same
    visual line, where the label may end with a colon or be "图 2" or "Fig. 1."; b) a one-line note right
    below such a label; c) within a caption group with a colon, a short continuation line that is
    left-aligned, starts in lowercase, and follows a line without sentence-final punctuation. Table rows and
    headings are not merged."""
    out: list[TextBlock] = []
    for b in blocks:
        if out:
            p = out[-1]
            # An index entry ends at its page numbers (perplexity:coined by, 89 has a colon but is no caption
            # label)
            if p.page == b.page and p.clip == b.clip and not (p.side_note or b.side_note or p.entry):
                y_ov = min(p.rect.y1, b.rect.y1) - max(p.rect.y0, b.rect.y0)
                same_line = y_ov > 0.5 * min(p.rect.y1 - p.rect.y0, b.rect.y1 - b.rect.y0)
                gap = b.rect.x0 - p.rect.x1
                # Notes and continuation lines have the caption's size. Small text under a large heading is a
                # box in a figure
                below = (
                    -0.1 * p.size <= b.rect.y0 - p.rect.y1 - p.line_gap * p.size < 1.1 * p.size
                    and abs(b.rect.x0 - p.rect.x0) < p.size
                    and abs(b.size - p.size) <= p.size * 0.2
                )
                label = (p.text.rstrip().endswith((":", "：")) and len(p.text) < 30) or _caption_label(p.text)
                head = b.text.lstrip()[:1]
                note = bool(head and (head.islower() or CJK_RE.search(head)) and not _caption_label(b.text))
                # A line that itself starts with a bold label is the next entry of a definition list
                # (mouse² : ... like a mouse | bank¹ : ...), not a continuation line
                continuation = (
                    len(p.text) < 60
                    and b.text[:1].islower()
                    and (":" in p.text or "：" in p.text)
                    and not _ends_sentence(p.text)
                    and not (
                        b.span_lines
                        and _label_lead([span for span in b.span_lines[0] if str(span.get("text") or "").strip()])
                    )
                )
                # A definition-style label is followed by a column of values: several lines, each starting at
                # the values' left edge (View ranges:  {...}, | {...}).
                # Merged into the label, the translation would wrap back under the label and the column would
                # be lost. Caption labels are merged as before
                rows = _visual_lines(b) if b.line_rects else []
                column = (
                    not _caption_label(p.text)
                    and len(rows) >= 2
                    and b.rect.x0 > p.rect.x1
                    and all(abs(min(rect.x0 for rect in row) - b.rect.x0) <= 1.0 for row in rows)
                )
                if (same_line and -1 <= gap < 3 * p.size and label and not column) or (
                    below and (continuation or (label and note))
                ):
                    _absorb_block(p, b)
                    continue
        out.append(b)
    return out
