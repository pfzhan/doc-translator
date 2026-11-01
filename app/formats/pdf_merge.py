"""Find where a paragraph continues: on the next page, in the next column or in the next
layout-model region, and pair the blocks into one translation unit."""

import re
from collections import Counter
from collections.abc import Callable, Sequence

import pymupdf

from .pdf_blocks import (
    _AFFIL_MARK_RE,
    _END_MARK_RE,
    _EQUATION_TAG_RE,
    _LABEL_START_RE,
    _NUMBERED_HEAD_RE,
    _SRC_HEAD_NUM_RE,
    CJK_RE,
    TextBlock,
    _ends_sentence,
    _indents_paragraphs,
    _list_item,
    _numbered_line,
    _same_page_line_pair,
    _span_is_math,
    _styled_edge,
    _visual_lines,
)
from .pdf_extract import (
    _FOOTNOTE_LEAD_RE,
    _MID_ABBREV_RE,
    _caption_label,
    _closes_first,
    _earlier_on_page,
    _is_float_text,
    _is_line_tail,
    _line_tail_blocked,
    _line_tail_match,
    _opens_with_math,
    _unclosed_bracket,
)
from .pdf_flow import (
    emphasis_of,
)
from .pdf_layout import (
    PageGeometry,
)
from .pdf_math import _prose_units
from .pdf_roles import is_size_heading


def _preview_kind(block: TextBlock, median: float) -> str:
    """Among short blocks, those clearly larger than the body text, and bold ones slightly larger, count as
    subheadings.

    Bold alone is not enough: author names and caption labels are often bold but no larger than the body.
    When layout roles are available they decide headings; this is only the font-size fallback.
    """
    return "h2" if is_size_heading(block.text, block.size, block.bold, median) else "p"


# Sentence-ending punctuation: tells whether the last block on a page is a paragraph cut by the page break
_SENT_END_PUNCT = tuple(".!?:;…。！？；：\"'”’)]}》")


def _different_columns(
    geometries: list[PageGeometry] | None,
    left: TextBlock,
    right: TextBlock,
) -> bool:
    """If either page has several columns and the two blocks have different column indexes, the second block
    is not the continuation of the first in the same column.

    The right column's end continuing in the left column of the next page is a separate rule handled
    elsewhere. On single-column pages the index is always 0.
    """
    if not geometries or left.page >= len(geometries) or right.page >= len(geometries):
        return False
    a, b = geometries[left.page], geometries[right.page]
    if len(a.columns) < 2 and len(b.columns) < 2:
        return False
    return a.index_of(left.rect) != b.index_of(right.rect)


def _cjk_page_continue(prev: TextBlock, nxt: TextBlock) -> bool:
    """Chinese has no letter case to show a continuation across pages. Short labels, indented new paragraphs
    and caption labels are not joined."""
    head = nxt.text.lstrip()[:1]
    if not head or CJK_RE.search(head) is None or _caption_label(nxt.text):
        return False
    if len(CJK_RE.findall(prev.text)) < 4:
        return False
    return abs(nxt.rect.x0 - prev.rect.x0) <= prev.size * 1.2


def _is_page_footnote(block: TextBlock, median: float) -> bool:
    if block.size + 0.6 >= median:
        return False
    return _FOOTNOTE_LEAD_RE.match(block.text.lstrip()) is not None


def _can_continue(
    prev: TextBlock,
    nxt: TextBlock,
    median: float,
    halt_ids: set[int] | None,
    allow_capital: bool = False,
    flush: bool = False,
    at_edge: bool = True,
) -> bool:
    """The block at the end of a page has no sentence-ending punctuation, and the block at the top of the next
    starts with a lowercase letter, a digit or Chinese, and is not a heading.

    A capital letter is allowed when moving to another column: an unfinished sentence often breaks at a
    proper noun such as SQL. Within one column a capital still ends the paragraph, since it cannot be told
    from the next sentence. Table cells stand alone: columns of cells look like text columns, and the last
    cell of one column must not join the header of the next. flush marks, in a document that indents the
    first line of paragraphs, a block at the top of a page or column (or below a figure at the column top)
    that starts flush (see _flush_start): new paragraphs are indented, so a flush one continues the previous
    paragraph, and it is joined even when the previous block happens to end a sentence and this one starts
    with a capital. at_edge means the first line of the next block starts at the left edge of the column's
    body text (see _at_text_edge).
    """
    if prev.clip is not None or nxt.clip is not None:
        return False
    # A short all-italic or all-bold line at the bottom of a column is a subheading
    # (E. Energy-Detection Characterization) and does not continue in the next column
    if len(_visual_lines(prev)) == 1 and len(prev.text.split()) <= 8 and _all_styled(prev):
        return False
    # For an opening parenthesis look at the text inside: (RMS) loudness, (0, 1), which. Numbers such as (3) and
    # (a) do not count. Square brackets and quotes are not looked into: [v], [z] are phonetic symbols, and
    # “[M]achines starts a new quotation
    opening = nxt.text.lstrip()
    if not _EQUATION_TAG_RE.match(opening) and not _list_item(opening):
        opening = opening.lstrip("(")
    head = opening[:1]
    numbered_heading = bool(_SRC_HEAD_NUM_RE.match(nxt.text.lstrip()))
    if halt_ids is None:
        is_heading = _preview_kind(nxt, median) != "p"
    else:
        is_heading = id(nxt) in halt_ids
    # A block starting with an inline formula (... minimise a divergence, | 𝐷(⋅||⋅) between ...) also continues:
    # a capital letter in a formula does not start a sentence. Bullets in a math font (• We introduce ...) do not
    # count
    math_start = _opens_with_math(nxt) and not _list_item(nxt.text)
    continues = head.islower() or head.isdigit() or _cjk_page_continue(prev, nxt) or math_start
    # The second half of a word hyphenated at the line end (fine-) is lowercase, so capitals are not allowed
    # even across columns: a capitalized block at the top of the right column is a table header or a heading
    broken_word = bool(re.search(r"[a-z]-$", prev.text.rstrip()))
    if allow_capital and not broken_word and head.isalpha() and head.isupper() and len(prev.text.split()) >= 3:
        continues = True
    # The next block starts with a parenthetical followed by punctuation or lowercase:
    # (Tables S32–S35), the port ..., (§5.3). Urban ...
    lead = nxt.text.lstrip()
    if (
        re.match(r"\([^()]{1,60}\)\s*(?:[.,;:]|[a-z])", lead)
        and not _EQUATION_TAG_RE.match(lead)
        and not _list_item(lead)
    ):
        continues = True
    # An abbreviation inside a sentence (Mao et al. | 2026) does not end it. Lowercase text after a semicolon,
    # a colon, or a closing parenthesis without a period inside continues the same sentence
    # (... on BC trials; | an explanation ..., ... GOP (4.750 vs. 4.733) | and lower ...). When a display
    # formula after a colon is skipped, the where on the next page follows the colon
    # (... scoring function: | where U, W ...). An indented or centered line below a colon is an example or
    # formula being introduced (... from that instance: | is traditionally followed by cherry pie) and is not
    # joined: a continuation starts at the left edge of the body text
    tail = _END_MARK_RE.sub("", prev.text)
    closers = "\"'”’)]}》"
    weak = (
        tail.endswith((";", "；"))
        or (at_edge and tail.endswith((":", "：")))
        or (tail.endswith(tuple(closers)) and not tail.rstrip(closers).endswith(_SENT_END_PUNCT))
    )
    ended = (
        tail.endswith(_SENT_END_PUNCT)
        and not _MID_ABBREV_RE.search(prev.text.rstrip())
        and not (weak and head.islower())
    )
    # A bracket is still open and the next block closes it in its first sentence:
    # (Fu et al., 2024b; | Mao et al., 2024) share ...
    if _unclosed_bracket(prev.text) and _closes_first(nxt.text):
        continues, ended = True, False
    # Body text only: flush text below a smaller figure or table caption does not continue the caption.
    # Only capitalized starts are allowed here: lowercase text below a sentence end (restric- | tion from ...)
    # is the second half of another broken block. When the page ends with a numbered example
    # ((24.31) IBM announced ...), flush text on the next page continues the paragraph before the example, not
    # the example
    if (
        flush
        and head.isupper()
        and len(prev.text.split()) >= _FLUSH_HOST_WORDS
        and abs(prev.size - nxt.size) <= prev.size * 0.05
        and _flush_start(nxt)
        and not _numbered_line(prev.text)
    ):
        continues, ended = True, False
    return bool(
        not numbered_heading
        and not is_heading
        and continues
        and not ended
        and not _is_float_text(nxt.text)
        and nxt.size <= prev.size * 1.4
    )


def _flush_start(block: TextBlock) -> bool:
    """The block is a paragraph filling the column width whose first line starts at the column edge like the
    lines below it. Run-in bold or italic headings, numbered labels, section numbers and list items start
    flush anyway and do not count"""
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    if len(rows) < 2 or _styled_edge(block, 0) or _list_item(block.text):
        return False
    if _LABEL_START_RE.match(block.text) or _NUMBERED_HEAD_RE.match(block.text):
        return False
    lefts = [min(rect.x0 for rect in row) for row in rows]
    rights = [max(rect.x1 for rect in row) for row in rows]
    return max(lefts) - min(lefts) <= 0.3 * block.size and rights[0] >= max(rights[1:]) - 0.5 * block.size


def _at_text_edge(head: TextBlock, page_blocks: list[TextBlock]) -> bool:
    """The first line of the block starts at the left edge of the column's body text. The edge is the most
    common left of all lines except the first in multi-line blocks on the page that overlap it horizontally
    (itself included), which works for first-line and hanging indents alike. Continuation lines of captions
    are centered and do not count. With no lines to measure, the block counts as at the edge"""
    lefts = Counter(
        round(min(rect.x0 for rect in row))
        for block in page_blocks
        if block.rect.x0 < head.rect.x1 and block.rect.x1 > head.rect.x0 and not _is_float_text(block.text)
        for row in sorted(_visual_lines(block), key=lambda row: row[0].y0)[1:]
    )
    if not lefts:
        return True
    rows = sorted(_visual_lines(head), key=lambda row: row[0].y0)
    first = min(rect.x0 for rect in rows[0]) if rows else head.rect.x0
    return abs(first - lefts.most_common(1)[0][0]) <= 0.3 * head.size


def _resumes(head: TextBlock, page_blocks: list[TextBlock], skip_ids: set[int], passed: set[int]) -> bool:
    """The block is where text from the previous column or page would resume: nothing is directly above it
    (top of a column or page), or what is above is a skipped figure or formula, a smaller table cell, or a
    caption (passed holds running headers and footers, figures and captions). When a short line at body size
    is directly above (a missed heading such as B.3.1. RL Data, PROMPT in a prompt box), the block starts a
    new paragraph"""
    above = [
        block
        for block in page_blocks
        if block is not head
        and block.rect.y1 <= head.rect.y0 + 1
        and block.rect.x0 < head.rect.x1
        and block.rect.x1 > head.rect.x0
    ]
    if not above:
        return True
    near = max(above, key=lambda block: block.rect.y1)
    return id(near) in skip_ids or id(near) in passed or _is_float_text(near.text) or near.size < head.size * 0.9


# A section number may hang at most this many em to the left of the column edge
_HANGING_HEAD_EM = 3
# For a flush continuation to be joined, the previous block needs at least this many words (a paragraph)
_FLUSH_HOST_WORDS = 8


def _page_edge_blocks(
    blocks: list[TextBlock],
    median: float,
    ignored: set[int],
) -> list[TextBlock]:
    """Only body text counts at the top and bottom of a page. Smaller numbered footnotes are translated where
    they are and do not take the page-end position."""
    return [
        block
        for block in blocks
        if block.size >= median * 0.8 and id(block) not in ignored and not _is_page_footnote(block, median)
    ]


def _column_index(geo: PageGeometry, block: TextBlock) -> int:
    return geo.index_of(block.rect)


def _in_column(geo: PageGeometry, block: TextBlock) -> bool:
    return len(geo.columns) < 2 or not geo.spans_columns(block.rect)


def _column_page_end(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    """The page end used for continuation in the same column. A block spanning columns on a multi-column
    page does not take this position, or it would take it from the last line of the right column."""
    return _column_edge(body, geometries, last=True)


def _column_page_start(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    return _column_edge(body, geometries, last=False)


def _column_edge(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
    last: bool,
) -> TextBlock | None:
    if not body:
        return None
    if not geometries:
        return body[-1] if last else body[0]
    ordered = reversed(body) if last else body
    for block in ordered:
        if block.page >= len(geometries) or _in_column(geometries[block.page], block):
            return block
    return None


def _is_last_in_right_column(
    block: TextBlock,
    page_blocks: list[TextBlock],
    geo: PageGeometry,
) -> bool:
    if len(geo.columns) < 2 or not _in_column(geo, block):
        return False
    if _column_index(geo, block) != len(geo.columns) - 1:
        return False
    return not any(
        other is not block
        and _in_column(geo, other)
        and _column_index(geo, other) == len(geo.columns) - 1
        and other.rect.y0 > block.rect.y0 + 1
        for other in page_blocks
    )


def _left_column_head(page_blocks: list[TextBlock], geo: PageGeometry) -> TextBlock | None:
    if len(geo.columns) < 2:
        return None
    heads = [
        block
        for block in page_blocks
        if _in_column(geo, block)
        and _column_index(geo, block) == 0
        and not any(
            other is not block
            and _in_column(geo, other)
            and _column_index(geo, other) == 0
            and other.rect.y0 < block.rect.y0 - 1
            for other in page_blocks
        )
    ]
    if not heads:
        return None
    return min(heads, key=lambda block: (block.rect.y0, block.rect.x0))


_CODE_LINE_RE = re.compile(r"^(?:<|\d+\s+0x[0-9a-fA-F])")
# A block with this many words is body text regardless of its share of letters
_PROSE_WORDS = 8
_CLUSTER_GAP = 28.0


def _is_code_line(text: str) -> bool:
    """XML, stack traces and labels inside figures. At the top of a column they must not take the place of a
    continuation.

    Short continuations (well., tomer) are still body text; a label has no spaces and contains digits or
    symbols. Result paragraphs full of numbers (and lower mean WER (0.009 vs. 0.013); paired ...) have a low
    share of letters but enough words to be body text.
    """
    stripped = text.strip()
    if not stripped or _CODE_LINE_RE.match(stripped):
        return True
    if stripped[:1] in "<{" or "::" in stripped or '="' in stripped or "/>" in stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < len(stripped) * 0.45 and _prose_units(stripped) < _PROSE_WORDS:
        return True
    return " " not in stripped and re.search(r"[\d%$<>!#_]", stripped) is not None


def _continuation_prose(
    block: TextBlock,
    median: float,
    skip_ids: set[int],
    halt_ids: set[int] | None,
    ignored: set[int],
) -> bool:
    if id(block) in skip_ids or id(block) in ignored:
        return False
    # Contents and index entries do not join their neighbors but still occupy the top and bottom of a column:
    # if they were skipped, a headword without a page number at the bottom would join an entry in the middle
    # of the next column. Cells of word lists are skipped like figures
    listed = (block.leader is not None or block.entry) and not block.grid
    if halt_ids is not None and id(block) in halt_ids and not listed:
        return False
    if block.size < median * 0.8 or _is_page_footnote(block, median):
        return False
    if _AFFIL_MARK_RE.match(block.text.lstrip()):
        return False
    return not _is_float_text(block.text) and not _is_code_line(block.text)


def _column_anchors(blocks: list[TextBlock]) -> list[float]:
    """Left edges of the body text. A column needs at least two blocks at the same left edge, so the right half
    of a single line does not start a column. When the page already has such a column, a single block of
    three or more lines also counts (another column holding one paragraph, below an indented abstract).

    The next column starts beyond the right edge of the previous one. Indented examples and quotations lie
    inside the previous column and do not start a column; otherwise their last block would be taken as the
    page end and the start of the next page's paragraph joined to it.
    """
    if not blocks:
        return []
    groups: list[list[TextBlock]] = []
    for block in sorted(blocks, key=lambda block: block.rect.x0):
        if groups and block.rect.x0 - groups[-1][0].rect.x0 < 8:
            groups[-1].append(block)
        else:
            groups.append([block])
    merged: list[float] = []
    right = None
    paired = any(len(group) >= 2 for group in groups)
    for group in groups:
        if len(group) < 2 and not (paired and len(_visual_lines(group[0])) >= 3):
            continue
        anchor = sum(block.rect.x0 for block in group) / len(group)
        if merged and (anchor - merged[-1] < _CLUSTER_GAP or (right is not None and anchor < right - 4)):
            continue
        merged.append(anchor)
        edges = sorted(block.rect.x1 for block in group)
        right = edges[len(edges) // 2]
    return merged


def _anchor_index(block: TextBlock, anchors: list[float]) -> int | None:
    if not anchors:
        return None
    best = min(range(len(anchors)), key=lambda index: abs(block.rect.x0 - anchors[index]))
    if abs(block.rect.x0 - anchors[best]) <= max(block.size * 4, 36):
        return best
    return None


def _assigned_columns(
    blocks: list[TextBlock],
    anchors: list[float],
) -> tuple[list[list[TextBlock]], list[TextBlock]]:
    columns: list[list[TextBlock]] = [[] for _ in anchors]
    orphans: list[TextBlock] = []
    for block in blocks:
        index = _anchor_index(block, anchors)
        if index is None:
            orphans.append(block)
        else:
            columns[index].append(block)
    return columns, orphans


def _column_body(column: list[TextBlock]) -> list[TextBlock]:
    """Smaller footnotes at the end of a column do not take the place of a continuation. If the whole column
    is in that size, everything stays.

    Table cells are left out when finding the body size: a table at the top of a column has one block per
    cell and would outnumber the body text.
    """
    if not column:
        return []
    sizes = sorted(block.size for block in column if block.clip is None) or sorted(block.size for block in column)
    body_size = sizes[len(sizes) // 2]
    body = [block for block in column if block.size + 0.6 >= body_size]
    return body or column


def _column_top(blocks: list[TextBlock]) -> TextBlock:
    return min(blocks, key=lambda block: (block.rect.y0, block.rect.x0))


def _column_bottom(blocks: list[TextBlock]) -> TextBlock:
    return max(blocks, key=lambda block: (block.rect.y1, block.rect.x0))


def _fragment_host(tail: TextBlock, orphans: list[TextBlock]) -> TextBlock:
    """When the body text at the end of a column already ends a sentence, an unfinished half sentence to its
    right on the same line is where the continuation starts.

    Line boxes are used rather than the block's union box, which would cover the half sentence on the right
    so that it no longer counts as attached to the line end.
    """
    if not _ends_sentence(tail.text):
        return tail
    lines = tail.line_rects or [tail.rect]
    for orphan in orphans:
        if orphan.page != tail.page or len(orphan.line_rects) > 1:
            continue
        if _ends_sentence(orphan.text):
            continue
        height = orphan.rect.height or orphan.size
        for line in lines:
            overlap = min(orphan.rect.y1, line.y1) - max(orphan.rect.y0, line.y0)
            if overlap <= min(height, line.height) * 0.5:
                continue
            gap = orphan.rect.x0 - line.x1
            if -2 <= gap < 16:
                return orphan
    return tail


def _column_wrap(host: TextBlock, tail: TextBlock) -> bool:
    """A move to another column needs left edges a column apart. A capital at the top of the page in the same
    column still starts the next paragraph."""
    return abs(host.rect.x0 - tail.rect.x0) > max(host.size * 4, 36.0)


def _try_column_pair(
    pairs: dict[int, TextBlock],
    tails: set[int],
    host: TextBlock | None,
    tail: TextBlock | None,
    median: float,
    halt_ids: set[int] | None,
    skip_ids: set[int],
    allow_capital: bool = True,
    flush: bool = False,
    at_edge: bool = True,
) -> None:
    if host is None or tail is None or host is tail:
        return
    if id(host) in pairs or id(host) in tails or id(tail) in pairs or id(tail) in tails:
        return
    if id(host) in skip_ids or id(tail) in skip_ids:
        return
    if halt_ids is not None and id(host) in halt_ids:
        return
    if abs(host.size - tail.size) > max(host.size, 1.0) * 0.25:
        return
    if not _can_continue(host, tail, median, halt_ids, allow_capital=allow_capital, flush=flush, at_edge=at_edge):
        return
    pairs[id(host)] = tail
    tails.add(id(tail))


def _float_contents(column: list[TextBlock], page_blocks: list[TextBlock]) -> set[int]:
    """Text of a table (or figure) at the top of a column: from the column top down to the caption below it
    (Table 7: ...), no line reaches the right edge of the column. Table headers and row names are short lines;
    paragraph lines fill the column width, so a paragraph at the top does not count. When a table caption is
    above the table and the column starts right below it, the table runs from the top down to the first block
    that fills the column width. A figure caption is below the figure, and short lines below it are body text
    (subject to after a formula) and do not count. The column's right edge is where justified body lines most
    often end; figures wider than the body text do not count."""
    if not column:
        return set()
    x0 = min(block.rect.x0 for block in column)
    x1 = max(block.rect.x1 for block in column)
    ends = Counter(round(line.x1) for block in column for line in (block.line_rects or [block.rect]))
    right, count = ends.most_common(1)[0]
    if count < 3:
        return set()
    captions = [
        block for block in page_blocks if _is_float_text(block.text) and block.rect.x0 < x1 and block.rect.x1 > x0
    ]
    if not captions:
        return set()
    caption = min(captions, key=lambda block: block.rect.y0)

    def short(block: TextBlock) -> bool:
        return all(line.x1 < right - block.size * 2 for line in (block.line_rects or [block.rect]))

    top = min(column, key=lambda block: block.rect.y0)
    over = [block for block in captions if block.rect.y1 <= top.rect.y0 + 1]
    near = max(over, key=lambda block: block.rect.y1) if over else None
    if near is not None and re.match(r"(?i)tab", near.text.lstrip()) and top.rect.y0 - near.rect.y1 <= 2 * top.size:
        contents: set[int] = set()
        for block in sorted(column, key=lambda block: block.rect.y0):
            if not short(block):
                return contents
            contents.add(id(block))
        return set()
    above = [block for block in column if block.rect.y1 <= caption.rect.y0 + 1]
    if not above or not all(short(block) for block in above):
        return set()
    return {id(block) for block in above}


def _column_continuations(
    by_page: dict[int, list[TextBlock]],
    median: float,
    skip_ids: set[int],
    halt_ids: set[int] | None,
    ignored: set[int],
    aside_ids: set[int] = frozenset(),
    indents: bool = False,
) -> dict[int, TextBlock]:
    """Join the end of the left column to the top of the right column, and the end of the right column to the
    left column of the next page.

    Figures, lists and captions at the top of a column are passed over and the continuation is looked for
    below them, and so are tables there (see _float_contents). Columns are found from left edges, so this
    works when the geometric columns are wrong and does not depend on reads_in_columns. Continuation in the
    same column across pages does not allow capitals: when the other column has no left edge, the left
    column at the page end must not swallow a new paragraph on the next page. Flush starts in documents that
    indent first lines are the exception (indents, see _can_continue).
    """
    pairs: dict[int, TextBlock] = {}
    tails: set[int] = set()
    pages = sorted(by_page)
    prose: dict[int, list[TextBlock]] = {
        pno: [block for block in blocks if _continuation_prose(block, median, skip_ids, halt_ids, ignored)]
        for pno, blocks in by_page.items()
    }
    columns_of: dict[int, list[list[TextBlock]]] = {}
    orphans_of: dict[int, list[TextBlock]] = {}
    for pno, blocks in prose.items():
        columns, orphans = _assigned_columns(blocks, _column_anchors(blocks))
        columns_of[pno] = [column for column in columns if column]
        orphans_of[pno] = orphans

    def head_of(pno: int, column: list[TextBlock]) -> TextBlock | None:
        contents = _float_contents(column, by_page.get(pno) or [])
        body = _column_body([block for block in column if id(block) not in contents])
        # A single line centered below a line that belongs to no column is an author or affiliation set in
        # columns (Shixiang Zhu | Carnegie Mellon University), not a column top. Below table cells or a heading
        # across both columns is the column top
        orphans = orphans_of.get(pno) or []
        body = [
            block
            for block in body
            if not (
                len(_visual_lines(block)) == 1
                and any(
                    0 <= block.rect.y0 - other.rect.y1 <= 1.5 * block.size
                    and abs((block.rect.x0 + block.rect.x1) - (other.rect.x0 + other.rect.x1)) / 2 <= block.size
                    for other in orphans
                )
            )
        ]
        if not body:
            return None
        head = _column_top(body)
        return None if heading_above(pno, column, head) else head

    def heading_above(pno: int, column: list[TextBlock], head: TextBlock) -> bool:
        """A subheading right above the column top within this column (A. Materials): the column starts a new
        section and holds no continuation. A section number hanging left of the column edge (1.4 Related work)
        counts too. A heading across several columns does not, nor does large text in a figure taken for a
        heading (with other figure text in between)"""
        if halt_ids is None:
            return False
        above = [
            block
            for block in by_page.get(pno) or []
            if id(block) not in ignored
            and block is not head
            and block.rect.y1 <= head.rect.y0 + 1
            and block.rect.x0 < head.rect.x1
            and block.rect.x1 > head.rect.x0
        ]
        if not above:
            return False
        near = max(above, key=lambda block: block.rect.y1)
        x0 = min(block.rect.x0 for block in column)
        x1 = max(block.rect.x1 for block in column)
        return (
            id(near) in halt_ids
            and id(near) not in aside_ids
            and id(near) not in skip_ids
            and not near.side_note
            and near.leader is None
            and not near.entry
            and near.rect.x0 >= x0 - _HANGING_HEAD_EM * near.size
            and near.rect.x1 <= x1 + near.size
        )

    def host_of(pno: int, column: list[TextBlock]) -> TextBlock | None:
        body = _column_body(column)
        if not body:
            return None
        return _fragment_host(_column_bottom(body), orphans_of.get(pno) or [])

    for pno in pages:
        columns = columns_of.get(pno) or []
        for index in range(len(columns) - 1):
            host = host_of(pno, columns[index])
            # A continuation is not larger than the first half. Larger text at the top of the right column is a
            # title or byline above both columns, and the continuation is below it
            head = head_of(
                pno, [block for block in columns[index + 1] if host is None or block.size <= host.size * 1.1]
            )
            # Only text that runs to the bottom of the left column and resumes at the top of the right one moves
            # across columns. Columns that start side by side are not joined.
            if host is None or head is None or host.rect.y0 <= head.rect.y0 + host.size:
                continue
            if host.rect.x1 > head.rect.x0 + 4:
                continue
            _try_column_pair(
                pairs,
                tails,
                host,
                head,
                median,
                halt_ids,
                skip_ids,
                flush=indents and _resumes(head, by_page.get(pno) or [], skip_ids, ignored | aside_ids),
                at_edge=_at_text_edge(head, by_page.get(pno) or []),
            )
        later = _next_text_page(by_page, pno, median, skip_ids, ignored)
        nxt = columns_of.get(later) or []
        if not columns or not nxt:
            continue
        host = host_of(pno, columns[-1])
        # When the right column was not found (none of its blocks got a column), the last column is not the
        # rightmost: a block with body text to its right is not the page end. Nor is one with body text below it
        # that spans it from left to right in a size no smaller: a table in a figure whose cells share a left edge
        # was taken for a column, with a full-width paragraph below. Footnotes are smaller and labels beside
        # display formulas narrower, so neither blocks it
        if host is not None and any(
            block.rect.x0 >= host.rect.x1 - host.size
            and block.rect.y1 > host.rect.y0
            or block is not host
            and block.rect.y0 >= host.rect.y1 - 1
            and block.size >= host.size - 0.5
            and block.rect.x0 <= host.rect.x0 + host.size
            and block.rect.x1 >= host.rect.x1 - host.size
            for block in prose[pno]
        ):
            continue
        head = head_of(later, nxt[0])
        _try_column_pair(
            pairs,
            tails,
            host,
            head,
            median,
            halt_ids,
            skip_ids,
            allow_capital=bool(host and head and _column_wrap(host, head)),
            flush=bool(indents and head and _resumes(head, by_page.get(later) or [], skip_ids, ignored | aside_ids)),
            at_edge=head is None or _at_text_edge(head, by_page.get(later) or []),
        )
    return pairs


def _region_joins(
    units: list[list[TextBlock]],
    skip_ids: set[int],
    halt_ids: set[int],
    regions_of: Callable[[int], list[pymupdf.Rect]],
) -> list[list[TextBlock]]:
    """With the layout model enabled: two consecutive units on the same page (possibly with skipped blocks
    between them) that fall in the same body-text box of the model, with a formula at the cut, are one
    paragraph cut by the formula (... with{formula} For a Borel set ... | 1{formula} is the indicator ...).
    The first block must be paragraph text: the model also puts a list of author names or the lines of an
    algorithm into one box (see _region_prose). The model often boxes a list as one region, so a first block
    ending a sentence followed by a capitalized one is not joined (... under fixed conditions. | We show
    that ...). Headings, captions, running headers and footers, list items and run-in bold or italic labels
    are not joined either. Without the model the units are returned unchanged."""
    pages: dict[int, list[pymupdf.Rect]] = {}

    def region(block: TextBlock) -> pymupdf.Rect | None:
        if block.page not in pages:
            pages[block.page] = regions_of(block.page)
        index = _region_index(block.rect, pages[block.page])
        return None if index is None else pages[block.page][index]

    out: list[list[TextBlock]] = []
    last: int | None = None
    for unit in units:
        if all(id(block) in skip_ids for block in unit):
            out.append(unit)
            continue
        if last is not None and _region_continues(out[last][-1], unit[0], halt_ids):
            box = region(out[last][-1])
            if box is not None and box == region(unit[0]) and _region_prose(out[last][-1], box):
                out[last] = [*out[last], *unit]
                continue
        out.append(unit)
        last = len(out) - 1
    return out


def _region_prose(block: TextBlock, box: pymupdf.Rect) -> bool:
    """Paragraph text within the box: a few lowercase words (a list of author names has none), and more than
    one line, or a single line filling the box width. Slide bullet points, algorithm steps and prompt
    templates are short lines that the model also puts into one box."""
    if len(re.findall(r"\b[a-z]{3,}\b", block.text)) < _REGION_WORDS:
        return False
    return len(_visual_lines(block)) >= 2 or (
        block.rect.x0 - box.x0 <= 1.5 * block.size and box.x1 - block.rect.x1 <= 1.5 * block.size
    )


def _region_continues(prev: TextBlock, nxt: TextBlock, halt_ids: set[int]) -> bool:
    if prev.page != nxt.page or id(prev) in halt_ids or id(nxt) in halt_ids:
        return False
    if _list_item(nxt.text) or _LABEL_START_RE.match(nxt.text) or _styled_edge(nxt, 0) or _ALGO_LINE_RE.match(nxt.text):
        return False
    # A formula at the cut: in the last line of the first block or the first line of the next, or a gap where a
    # display formula sits
    if not (_math_on_line(prev, -1) or _math_on_line(nxt, 0) or nxt.rect.y0 - prev.rect.y1 >= _REGION_GAP * prev.size):
        return False
    letter = re.search(r"[A-Za-z]", nxt.text)
    # A closing quote also ends a sentence: a capitalized line below an epigraph is its source
    # (... Location, Location” | Randall Munroe)
    ended = _END_MARK_RE.sub("", prev.text).rstrip().endswith(tuple('.!?;:。！？；：”"'))
    return not (ended and letter is not None and letter.group().isupper())


def _region_index(rect: pymupdf.Rect, regions: list[pymupdf.Rect]) -> int | None:
    """The box that holds most of the block's area."""
    area = rect.width * rect.height
    if area < 1:
        return None
    best, best_share = None, _REGION_COVER
    for index, box in enumerate(regions):
        hit = rect & box
        if hit.is_empty:
            continue
        share = hit.width * hit.height / area
        if share >= best_share:
            best, best_share = index, share
    return best


# Share of a block's area that must fall in a body-text box of the model for the block to belong to it
_REGION_COVER = 0.5
# Minimum number of lowercase words in the first block when joining by model regions
_REGION_WORDS = 3
# A vertical gap of this many font sizes between two blocks holds a display formula
_REGION_GAP = 1.5
# Algorithm line numbers: 6: until converged
_ALGO_LINE_RE = re.compile(r"\d{1,3}:\s")
# Maximum number of following full-page figures or tables passed over when looking for a continuation
_FLOAT_PAGES = 2
# Subfigure labels: (C), (b), C.
_PANEL_LABEL_RE = re.compile(r"\(?[A-Za-z]\)\s|[A-Z]\.\s")


def _next_text_page(
    by_page: dict[int, list[TextBlock]],
    pno: int,
    median: float,
    skip_ids: set[int],
    ignored: set[int],
) -> int:
    """The first page after pno with body text. Pages holding only figures, captions (with the subfigure
    descriptions right below them) and running headers and footers are passed over: an unfinished paragraph
    continues on the page after them. A subfigure description starts with a label such as (C) or is smaller
    than the body text; a paragraph at body size right below a caption is body text (where the paragraph
    broken on the previous page continues)"""
    later = pno + 1
    while later <= pno + _FLOAT_PAGES and _float_page(by_page.get(later) or [], median, skip_ids, ignored):
        later += 1
    return later if later <= pno + _FLOAT_PAGES else pno + 1


def _float_page(blocks: list[TextBlock], median: float, skip_ids: set[int], ignored: set[int]) -> bool:
    prose = [
        block
        for block in blocks
        if id(block) not in ignored and id(block) not in skip_ids and block.size >= median * 0.8
    ]
    captions = [block for block in prose if _is_float_text(block.text)]
    for block in sorted(prose, key=lambda block: block.rect.y0):
        if block in captions:
            continue
        panel = _PANEL_LABEL_RE.match(block.text.lstrip()) is not None or block.size < median * 0.95
        if panel and any(
            0 <= block.rect.y0 - caption.rect.y1 <= 1.5 * block.size
            and abs(block.rect.x0 - caption.rect.x0) <= block.size
            for caption in captions
        ):
            captions.append(block)
            continue
        return False
    return True


def _cross_page_units(
    blocks: list[TextBlock],
    median: float,
    skip_ids: set[int] = frozenset(),
    geometries: list[PageGeometry] | None = None,
    halt_ids: set[int] | None = None,
    ignore_ids: set[int] | None = None,
    aside_ids: set[int] = frozenset(),
) -> list[list[TextBlock]]:
    """Pair paragraphs carried over to the next page into translation units (a list of block lists, one or two
    blocks per unit).

    When the last body block of a page does not end with sentence punctuation and the first block of the next
    page starts with a lowercase letter or digit and does not look like a heading, both are one paragraph cut
    by the page break. Footnotes (clearly smaller than the body text, or one size smaller and starting with a
    number or link) and blocks to skip take no part. Continuations in the same column are joined as before.
    With several columns, the only addition is that the last block of the rightmost column may continue in the
    first block of the leftmost column on the next page. ignore_ids are running headers and footers: they
    neither take the page end nor block the page top. Headings across columns and whole tables do not take a
    column top or bottom either. aside_ids are figures and captions: when the next page starts with a figure,
    the cut paragraph continues below its caption.
    """
    by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    ignored = ignore_ids or set()
    indents = _indents_paragraphs(blocks)
    # Without geometry the plain page-end and page-top rules apply. With geometry, continuations across columns
    # are joined by left edges. Half sentences on a shared baseline do not depend on column geometry: the next
    # line is still sent paired with it, only without merging their boxes.
    extra: dict[int, TextBlock] = {}
    taken: set[int] = set()
    for host_id, tail in _shared_line_pairs(blocks).items():
        if host_id in skip_ids or id(tail) in skip_ids:
            continue
        if host_id in ignored or id(tail) in ignored:
            continue
        if halt_ids is not None and (host_id in halt_ids or id(tail) in halt_ids):
            continue
        extra[host_id] = tail
        taken.add(host_id)
        taken.add(id(tail))
    if geometries is not None:
        for host_id, tail in _column_continuations(
            by_page,
            median,
            skip_ids,
            halt_ids,
            ignored,
            aside_ids,
            indents,
        ).items():
            if host_id in taken or id(tail) in taken:
                continue
            extra[host_id] = tail
            taken.add(host_id)
            taken.add(id(tail))

    units: list[list[TextBlock]] = []
    pages = sorted(by_page)
    used: set[int] = {id(tail) for tail in extra.values()}
    for pno in pages:
        body = _page_edge_blocks(by_page[pno], median, ignored | aside_ids)
        later = _next_text_page(by_page, pno, median, skip_ids, ignored)
        nxt = by_page.get(later) or []
        nxt_body = _page_edge_blocks(nxt, median, ignored | aside_ids)
        first = _column_page_start(nxt_body, geometries)
        end = _column_page_end(body, geometries)
        head = _left_column_head(nxt_body, geometries[later]) if geometries and later < len(geometries) else None
        for b in by_page[pno]:
            if id(b) in used:
                continue
            if id(b) in extra:
                tail = extra[id(b)]
                units.append([b, tail])
                used.add(id(b))
                used.add(id(tail))
                continue
            if halt_ids is not None and id(b) in halt_ids:
                units.append([b])
                used.add(id(b))
                continue
            # A block already paired with another on the same page cannot also join the end of the previous page,
            # or its partner would be left out of every unit
            same = bool(
                end is not None
                and b is end
                and first is not None
                and id(first) not in used
                and id(first) not in extra
                and id(b) not in skip_ids
                and id(first) not in skip_ids
                and _can_continue(
                    b,
                    first,
                    median,
                    halt_ids,
                    flush=indents and _resumes(first, nxt, skip_ids, ignored | aside_ids),
                    at_edge=_at_text_edge(first, nxt),
                )
                and not _different_columns(geometries, b, first)
            )
            cross_target = head if head is not None and id(head) not in used and id(head) not in extra else None
            cross = bool(
                not same
                and cross_target is not None
                and geometries is not None
                and pno < len(geometries)
                and _is_last_in_right_column(b, body, geometries[pno])
                and id(b) not in skip_ids
                and id(cross_target) not in skip_ids
                and _can_continue(
                    b,
                    cross_target,
                    median,
                    halt_ids,
                    flush=indents and _resumes(cross_target, nxt, skip_ids, ignored | aside_ids),
                    at_edge=_at_text_edge(cross_target, nxt),
                )
            )
            other = first if same else cross_target
            if (same or cross) and other is not None:
                units.append([b, other])
                used.add(id(b))
                used.add(id(other))
            else:
                units.append([b])
                used.add(id(b))
    return units


def _seam_apart(
    unit: list[TextBlock],
    by_page: dict[int, list[TextBlock]],
    geometries: Sequence[PageGeometry],
    covered: set[int],
    shown: set[int],
    median: float,
    words_of: Callable[[int], list[tuple]],
) -> bool:
    """A two-block unit across pages, across columns or cut by a formula, with a display formula or example
    between the blocks: right below the first block or right above the second there is a body-text block, or
    text that is in no block (a display formula dropped during extraction). Such a unit is split back into
    two blocks at the boundary when written (see _assign_restored_parts); filling and continuing would write
    the second half before the formula or example. When what lies next to them is a figure, table, caption,
    running header or footer, footnote (covered holds figures, captions, table cells and running headers and
    footers) or nothing, the blocks are only separated by a page or column break"""
    host, tail = unit
    if host.page == tail.page and _same_page_line_pair(host, tail):
        return False
    stacked = host.page == tail.page and tail.rect.y0 >= host.rect.y1 - 1 and _x_overlap(host.rect, tail.rect)
    context = (by_page, geometries, covered, shown, median, words_of)
    if _content_next(host, tail, *context, below=True):
        return True
    return not stacked and _content_next(tail, host, *context, below=False)


def _content_next(
    block: TextBlock,
    partner: TextBlock,
    by_page: dict[int, list[TextBlock]],
    geometries: Sequence[PageGeometry],
    covered: set[int],
    shown: set[int],
    median: float,
    words_of: Callable[[int], list[tuple]],
    below: bool,
) -> bool:
    """The nearest thing directly below the block (above when below is false) is body content: a recognized
    display formula or example (shown), a block not in covered, not inside a figure or table box and not a
    footnote, or text inside the text frame that belongs to no block. The layout model takes display formulas
    for figures, so shown is checked first. A block one size smaller is the rest of a caption or a margin
    note; a much smaller one is a formula extracted as a block and counts as content. The partner on the same
    page and anything past it are not considered"""
    geo = geometries[block.page] if block.page < len(geometries) else None
    floats = [*geo.figures, *(table.rect for table in geo.tables)] if geo is not None else []
    frame = geo.frame if geo is not None else None
    edge = None if frame is None else (frame.y1 if below else frame.y0)
    page_blocks = by_page.get(block.page) or []
    found: list[tuple[float, bool]] = []

    def beyond(y0: float, y1: float) -> bool:
        if below:
            return y0 >= block.rect.y1 - 1 and not (partner.page == block.page and y0 >= partner.rect.y0 - 1)
        return y1 <= block.rect.y0 + 1 and not (partner.page == block.page and y1 <= partner.rect.y1 + 1)

    def distance(y0: float, y1: float) -> float:
        return y0 - block.rect.y1 if below else block.rect.y0 - y1

    for rect in floats:
        if _x_overlap(rect, block.rect) and beyond(rect.y0, rect.y1):
            found.append((distance(rect.y0, rect.y1), False))
    for other in page_blocks:
        if other is block or other is partner or not _x_overlap(other.rect, block.rect):
            continue
        if not beyond(other.rect.y0, other.rect.y1):
            continue
        cover = id(other) not in shown and (
            id(other) in covered
            or _is_page_footnote(other, median)
            or _is_float_text(other.text)
            or any(rect.contains(other.rect) for rect in floats)
            or block.size * 0.8 <= other.size < block.size - 0.6
        )
        found.append((distance(other.rect.y0, other.rect.y1), not cover))
    # Dropped display formulas: text a little apart from the block, inside the text frame, and in no block or
    # figure or table box
    for x0, y0, x1, y1, *_rest in words_of(block.page):
        word = pymupdf.Rect(x0, y0, x1, y1)
        if not _x_overlap(word, block.rect) or not beyond(y0, y1) or distance(y0, y1) < 0.3 * block.size:
            continue
        if edge is not None and (y0 > edge + 2 if below else y1 < edge - 2):
            continue
        middle = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any((other.rect + (-1, -1, 1, 1)).contains(middle) for other in page_blocks):
            continue
        if any(rect.contains(middle) for rect in floats):
            continue
        found.append((distance(y0, y1), True))
    return bool(found) and min(found)[1]


def _x_overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1


def _all_styled(block: TextBlock) -> bool:
    """Every span is bold or italic."""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    marks = [emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)) for span in spans]
    return bool(marks) and all(mark.bold or mark.italic for mark in marks)


def _math_on_line(block: TextBlock, index: int) -> bool:
    """Line index of the block (-1 for the last) contains a math font."""
    lines = block.span_lines or []
    if not lines:
        return False
    return any(_span_is_math(span) for span in lines[index] if str(span.get("text") or "").strip())


def _shared_line_pairs(blocks: list[TextBlock]) -> dict[int, TextBlock]:
    """A continuation line on the same baseline as another paragraph is not merged into the box but is still
    sent paired. Merging it would make the writing cover that paragraph.

    The reading order may put another column between the half sentence and its continuation. The
    continuation's vertical position still matches the half sentence, so the search must not stop because
    the column in between has moved further down.
    """
    pairs: dict[int, TextBlock] = {}
    claimed: set[int] = set()
    earlier = _earlier_on_page(blocks)
    for index, block in enumerate(blocks):
        if id(block) in claimed or not _is_line_tail(block, earlier[index]):
            continue
        for nxt in blocks[index + 1 :]:
            if nxt.page != block.page:
                break
            if nxt.rect.y0 > block.rect.y1 + block.size * 2:
                continue
            if not _line_tail_match(block, nxt):
                continue
            if not _line_tail_blocked(nxt, block, blocks) or id(nxt) in claimed:
                break
            pairs[id(block)] = nxt
            claimed.add(id(block))
            claimed.add(id(nxt))
            break
    return pairs
