"""Decide whether the last block of a page or column continues in the block that follows: sentence ends,
footnotes, flush starts and floating pages."""

import re
from collections import Counter
from collections.abc import Set as AbstractSet

from .pdf_blocks import (
    _END_MARK_RE,
    _EQUATION_TAG_RE,
    _LABEL_START_RE,
    _NUMBERED_HEAD_RE,
    _SRC_HEAD_NUM_RE,
    CJK_RE,
    TextBlock,
    _list_item,
    _numbered_line,
    _styled_edge,
    _visual_lines,
)
from .pdf_continuations import _FOOTNOTE_LEAD_RE, _closes_first, _is_float_text, _opens_with_math, _unclosed_bracket
from .pdf_flow import (
    emphasis_of,
)
from .pdf_paragraphs import _caption_label
from .pdf_roles import is_size_heading
from .pdf_split_lines import _MID_ABBREV_RE


def _preview_kind(block: TextBlock, median: float) -> str:
    """Among short blocks, those clearly larger than the body text, and bold ones slightly larger, count as
    subheadings.

    Bold alone is not enough: author names and caption labels are often bold but no larger than the body.
    When layout roles are available they decide headings; this is only the font-size fallback.
    """
    return "h2" if is_size_heading(block.text, block.size, block.bold, median) else "p"


# Sentence-ending punctuation: tells whether the last block on a page is a paragraph cut by the page break
_SENT_END_PUNCT = tuple(".!?:;…。！？；：\"'”’)]}》")


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


def _resumes(
    head: TextBlock, page_blocks: list[TextBlock], skip_ids: AbstractSet[int], passed: AbstractSet[int]
) -> bool:
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


# For a flush continuation to be joined, the previous block needs at least this many words (a paragraph)
_FLUSH_HOST_WORDS = 8


# Maximum number of following full-page figures or tables passed over when looking for a continuation
_FLOAT_PAGES = 2


# Subfigure labels: (C), (b), C.
_PANEL_LABEL_RE = re.compile(r"\(?[A-Za-z]\)\s|[A-Z]\.\s")


def _next_text_page(
    by_page: dict[int, list[TextBlock]],
    pno: int,
    median: float,
    skip_ids: AbstractSet[int],
    ignored: AbstractSet[int],
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


def _float_page(blocks: list[TextBlock], median: float, skip_ids: AbstractSet[int], ignored: AbstractSet[int]) -> bool:
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


def _all_styled(block: TextBlock) -> bool:
    """Every span is bold or italic."""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    marks = [emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)) for span in spans]
    return bool(marks) and all(mark.bold or mark.italic for mark in marks)
