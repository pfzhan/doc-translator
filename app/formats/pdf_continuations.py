"""Merge the continuation lines of a paragraph that formulas or fragments broke apart within a page."""

import re
from collections import Counter
from collections.abc import Collection

import pymupdf

from .pdf_blocks import (
    _EQUATION_TAG_RE,
    _LABEL_START_RE,
    _MAX_INDENT_EM,
    _NUMBERED_HEAD_RE,
    CJK_RE,
    TextBlock,
    _ends_in_call,
    _ends_sentence,
    _first_line_indented,
    _indents_paragraphs,
    _list_item,
    _numbered_line,
    _span_is_math,
    _styled_edge,
    _visual_lines,
)
from .pdf_math import (
    _prose_units,
)
from .pdf_paragraphs import _absorb_block, _index_refs_tail, _verbatim_seam
from .pdf_rows import _INDEX_REFS_ONLY_RE
from .pdf_split_lines import _MID_ABBREV_RE, _centered_rows, _label_lead, _open_operator_end, _row_boxes, _row_style


def _new_voice(host: TextBlock, guest: TextBlock) -> bool:
    """The next block differs in size by more than 10% and also changes color: an example under a slide title,
    or a caption under body text, and not a wrapped line of the same paragraph.
    Within one paragraph the body size and color stay the same, and emphasized words and formulas inside it
    never make up the majority."""
    return abs(guest.size - host.size) > host.size * 0.12 and guest.color != host.color


def _restyled_seam(host: TextBlock, guest: TextBlock) -> bool:
    """The two blocks are part of one centered stack of lines, and the lines on either side of the seam change
    style: bold author names followed by an affiliation in a regular font, an affiliation followed by an email
    address in a monospaced font. Each line is one item, not a wrapped line of a paragraph. A line split into
    several pieces is compared as a whole line."""
    if not host.span_lines or not guest.span_lines or not host.line_rects or not guest.line_rects:
        return False
    if len(host.span_lines) != len(host.line_rects) or len(guest.span_lines) != len(guest.line_rects):
        return False
    if not _centered_rows(_row_boxes(host.line_rects + guest.line_rects), host.size):
        return False
    gap = max(host.size * 0.4, 2.0)

    def row_spans(block: TextBlock, at: pymupdf.Rect) -> list[dict]:
        return [
            span
            for rect, spans in zip(block.line_rects, block.span_lines)
            if abs(rect.y0 - at.y0) <= gap
            for span in spans
        ]

    upper = row_spans(host, host.line_rects[-1])
    lower = row_spans(guest, guest.line_rects[0])
    return _row_style(upper) != _row_style(lower)


# A footnote is one size smaller than body text and starts with a number (1 Although / 2Note / 4https://).
# Section numbers 1. / 2.2 are not footnotes. A footnote at the bottom of the page must not take the place of
# the page end of a paragraph carried over to the next page, or a paragraph broken after "quite" would not join
# "well" on the next page.
_FOOTNOTE_LEAD_RE = re.compile(r"^\d{1,2}(?!\.\d)(?:https?://|\s*[A-Za-z])", re.I)


def _unclosed_bracket(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth:
            depth -= 1
    return depth > 0


def _closes_first(text: str) -> bool:
    """The first sentence of the block contains an unmatched closing bracket."""
    depth = 0
    for match in re.finditer(r"[()\[\]]|[.!?](?=\s+[A-Z])", text[:300]):
        mark = match.group()
        if mark in "([":
            depth += 1
        elif mark in ")]":
            if depth == 0:
                return True
            depth -= 1
        elif depth == 0 and not _MID_ABBREV_RE.search(text[: match.end()]):
            return False
    return False


# A caption or listing at the top of a column does not take the place of the continued sentence. Figure 3 shows
# and Figure 8(a) shows in body text are not captions: a lowercase word follows the number.
# The number may also be Roman (IEEE's TABLE IV) or carry an appendix letter (Table A1, Figure S3, Figure B.6)
_FLOAT_HEAD_RE = re.compile(
    r"^(?:Figure|Fig\.?|Table|Tab\.?|Listing|List\.?|Scheme|Plate|Algorithm|图|表|清单)"
    r"\s*(?:(?-i:[A-Z])?\.?\d++(?:\.\d+)*+|(?-i:[IVXLC]++))\b"
    r"(?!(?:\s*\((?-i:[a-z])\))?\s+(?-i:[a-z]))",
    re.I,
)


def _is_float_text(text: str) -> bool:
    """A caption or listing title at the top of a column. The body text after it is what continues the
    paragraph before."""
    return _FLOAT_HEAD_RE.match(text.strip()) is not None


def _plain_words(block: TextBlock) -> int:
    """Number of body words outside formulas."""
    return _prose_units(
        " ".join(
            str(span.get("text") or "") for line in block.span_lines or [] for span in line if not _span_is_math(span)
        )
    )


def _cjk_wrap(prev: TextBlock, nxt: TextBlock) -> bool:
    """Chinese line breaks have no letter case and are not counted in words. A new paragraph has an indent of
    about 2 em, and a short label is not wide enough."""
    head = nxt.text.lstrip()[:1]
    if not head or CJK_RE.search(head) is None:
        return False
    if _ends_sentence(prev.text):
        return False
    if abs(nxt.size - prev.size) > prev.size * 0.2:
        return False
    if abs(nxt.rect.x0 - prev.rect.x0) > prev.size * 0.6:
        return False
    return len(CJK_RE.findall(prev.text)) >= 8 or prev.rect.width >= prev.size * 12


def _merge_continuations(
    blocks: list[TextBlock],
    listed: Collection[int] = frozenset(),
    margins: Collection[int] = frozenset(),
) -> list[TextBlock]:
    """Merge continuation lines of a paragraph that formulas or fragments broke apart within a page: when the
    line above has no sentence-final punctuation, the line below starts in lowercase (or with a closing
    bracket), the left margin is the same and the line spacing is normal, they are two lines of one paragraph
    split apart. Translated separately they produce half translations, and rendering forces a line break at
    the cut ("（如 | 序列）"). Chinese line breaks are joined the same way when the left edges align and the
    line above is long enough.
    English line breaks before a capital stay separate, since they cannot be told from the next sentence;
    the exception is a line above that stops at a word such as and, of or the.
    When a paragraph is set around a figure, the caption falls between the two halves in reading order. When
    that block lies below the first half and beside the second, it is skipped and the block before it is
    compared. margins are running headers and footers, which are not joined to the blocks above and below."""
    out: list[TextBlock] = []
    indents = _indents_paragraphs(blocks)
    # In documents with paragraph indents, a line starting flush below a sentence end is a caption, an example
    # or the half sentence after an expression, so no paragraph spacing can be measured
    space = 0.0 if indents else _paragraph_space(blocks)
    measure = _body_measure(blocks) if indents or space else 0.0
    for index, b in enumerate(blocks):
        hosts = out[-1:]
        if len(out) >= 2 and _beside_wrap(out[-1], out[-2], b):
            hosts.append(out[-2])
        for p in hosts:
            first = b.text[:1]
            # Fragments of a large heading ('...with' + 'Recurrent Neural Networks'): the second line of the
            # wrap often starts with a capital and has no punctuation. Merge only when the left edges of the two
            # lines clearly shift (centered heading lines): left-aligned large blocks (subheadings, byline
            # lines) stay separate.
            # A byline sits almost a line below the title and its left edge also shifts, but it is not the
            # title's next line.
            # A line with a list marker is the next item of a slide, whose left edge changes with the level; it
            # is not the title's next line either.
            # The two lines of a wrapped heading always overlap horizontally; one offset sideways is a label
            # beside it.
            # Large is measured against this document's body size: slide body text is around thirty points,
            # and indented subitems are not a wrapped heading.
            # Both lines of a wrapped heading are the same size. One size larger right after is the next
            # heading level (8.5 Summary under the 8.4.4 subsection name)
            title_frag = (
                p.size >= max(12, p.body_size * 1.1)
                and p.size * 0.7 <= b.size <= p.size * 1.1
                and 0 <= b.rect.y0 - p.rect.y1 < 0.75 * p.size
                and abs(b.rect.x0 - p.rect.x0) > p.size * 0.5
                and min(p.rect.x1, b.rect.x1) > max(p.rect.x0, b.rect.x0)
                and not _list_item(b.text)
                and not _ends_sentence(p.text)
                and not _ends_sentence(b.text)
            )
            y_close = -0.9 * p.size <= b.rect.y0 - p.rect.y1 - p.line_gap * p.size < 1.4 * p.size
            # A hanging indent is about 2-3 em. With a tight 2 em limit, continuation lines of a feature list are
            # split into two paragraphs and the short line is then shrunk on its own. A jump to the left still
            # uses 2 em, so that another column is not joined.
            x_jump = b.rect.x0 - p.rect.x0
            x_close = abs(x_jump) < 2 * p.size or 0 < x_jump <= max(4 * p.size, 36)
            # The last line of a centered paragraph (a table note) is short, indented on both sides, shares the
            # center line of the line above, and sits right below it at normal line spacing.
            # A centered formula line below a paragraph has more space around it, and spans more than one line
            # or is no sentence. A last line of only one or two words (the learning. of a caption) that is all
            # body words and ends with a period is not a formula either
            last_y1 = p.line_rects[-1].y1 if p.line_rects else p.rect.y1
            words = b.text.split()
            centered_tail = (
                len(_visual_lines(b)) == 1
                and b.rect.y0 - last_y1 - p.line_gap * p.size <= 0.3 * p.size
                and b.rect.x0 > p.rect.x0 + p.size
                and b.rect.x1 < p.rect.x1 - p.size
                and abs((b.rect.x0 + b.rect.x1) - (p.rect.x0 + p.rect.x1)) / 2 <= p.size
                and not _list_item(b.text)
                and (_prose_units(b.text) >= 3 or (_ends_sentence(b.text) and _prose_units(b.text) == len(words)))
            )
            # When the lines on both sides of a colon at a line end reach the right edge, it is a wrap inside a
            # sentence ("is called interpolation: | computing ...").
            # A next line that is not full is an example or list introduced by the colon
            last_x1 = p.line_rects[-1].x1 if p.line_rects else p.rect.x1
            next_x1 = b.line_rects[0].x1 if b.line_rects else b.rect.x1
            colon_wrap = (
                p.text.rstrip().endswith(":")
                and abs(last_x1 - next_x1) <= p.size
                and last_x1 >= max(p.rect.x1, b.rect.x1) - p.size
            )
            # Ending with f(x) counts as a sentence end so that a where below a display expression is not joined.
            # When the last line runs from the next line's left edge to the right edge and the next line starts
            # with a body word, it is a wrap inside a sentence and the expression just falls at the line end
            call_wrap = _ends_in_call(p.text.rstrip()) and _opens_with_word(b) and _wraps_into(p, b)
            # A line that starts with an inline expression followed by body text (Adam with | 1/√t decay on its
            # stepsize) is a wrap inside a sentence: it starts at the column edge, or the expression fragment at
            # its start became a block of its own while the rest runs to the right edge and the next line
            # returns to the column edge.
            # A display expression is indented, does not reach the right edge and has few body words; a footnote
            # starting with a superscript number is one size smaller
            first_x0 = b.line_rects[0].x0 if b.line_rects else b.rect.x0
            rows = sorted(_visual_lines(b), key=lambda row: row[0].y0) if b.line_rects else []
            math_wrap = (
                _opens_with_math(b)
                and _plain_words(b) >= _MATH_WRAP_WORDS
                and abs(b.size - p.size) <= p.size * 0.1
                and (
                    abs(first_x0 - p.rect.x0) <= p.size * 0.5
                    or (
                        len(rows) >= 2
                        and abs(max(rect.x1 for rect in rows[0]) - p.rect.x1) <= p.size * 0.5
                        and abs(min(rect.x0 for rect in rows[1]) - p.rect.x0) <= p.size * 0.5
                    )
                )
            )
            # An inline fraction at the end of a sentence wraps to the start of the next line with no body words
            # after it (the odds ratio | p/(1−p):): the stacked numerator and denominator are taller than one
            # line, start at the column edge, and sit right below a line above that runs to the right edge. A
            # display expression is indented and has space above it
            formula_tail = (
                bool(b.line_rects)
                and _plain_words(b) == 0
                and b.rect.height > p.size * 1.6
                and abs(first_x0 - p.rect.x0) <= p.size * 0.5
                and last_x1 >= p.rect.x1 - p.size
                and b.rect.y0 - last_y1 <= 0.3 * p.size
            )
            # An abbreviation inside a sentence (Brants et al. | (2007)) is not a sentence end. A continuation
            # line starting with an opening parenthesis is a citation year or a parenthetical when it is no
            # further right than the leftmost line of the paragraph above (a paragraph with a first-line
            # indent), shares a left edge with one of its lines (continuation of an entry with a hanging indent),
            # or is centered below a centered paragraph.
            # A label such as (a) is the next item, and (2.11) is the number of an example or formula;
            # an indented one is a display expression (a LUNAR logical form), and one offset to the right is a
            # column header such as (Acc).
            # When an entry starting with a bold label has only one line, there is no continuation line to
            # compare with, and a line with a hanging indent of one or two ems is also its continuation
            label_hang = (
                bool(p.span_lines)
                and 0 < b.rect.x0 - p.rect.x0 <= 3 * p.size
                and _label_lead([span for span in p.span_lines[0] if str(span.get("text") or "").strip()])
            )
            # The bracket is still open and the next line closes it in its first sentence
            # ((see e.g. | Ghosal et al., 2000), (Chérief- | Abdellatif ...))
            bracketed = _unclosed_bracket(p.text) and _closes_first(b.text)
            open_paren = (
                first == "("
                and not _list_item(b.text)
                and not _EQUATION_TAG_RE.match(b.text)
                and (
                    centered_tail
                    or label_hang
                    or b.rect.x0 <= p.rect.x0 + p.size * 0.5
                    or any(abs(b.rect.x0 - rect.x0) <= p.size * 0.5 for rect in p.line_rects)
                )
            )
            # When the line above stops at and or of, the next line continues even when it starts with a capital.
            # Not when the next line is another list item (...; and | • ...) or a footnote one size smaller
            same_size = abs(b.size - p.size) <= p.size * 0.05
            dangling = _dangling(p.text) and not _list_item(b.text) and not _opens_with_math(b) and same_size
            # The last line of a centered paragraph is joined even when it starts with a capital (Rates., MATH
            # dataset.); the smaller attribution under an epigraph is not
            centered_end = centered_tail and same_size
            # A centered paragraph (a list of authors) that wraps at a comma continues even when the next line
            # starts with a capital. The whole stack shares one center line and the font stays the same
            centered_comma = (
                same_size
                and p.text.rstrip().endswith((",", "，"))
                and not _list_item(b.text)
                and bool(p.line_rects)
                and bool(b.line_rects)
                and _centered_rows(_row_boxes(p.line_rects + b.line_rects), p.size)
            )
            # The indented first line of a paragraph shared a block with a margin note and was split off on its
            # own: when this line runs to the right edge without sentence-final punctuation and the next line is
            # one indent further left, it is a wrap inside a sentence, joined even when the next line starts
            # with a capital (a proper noun)
            full_wrap = (
                same_size and not _list_item(b.text) and len(_visual_lines(p)) == 1 and _wraps_into(p, b, indented=True)
            )
            # A first line after a list item that starts even further left is body text after the list has
            # ended. Continuation lines align with the item text or have a hanging indent and never move left
            list_end = _list_item(p.text) and first_x0 < p.rect.x0 - p.size * 0.5
            # The block above stops at half a broken word (retrieval-augmented genera- | tion, 267), and the next
            # line finishes the word in lowercase.
            # A digit after the hyphen is a model name broken across lines (VE- | 8. Both ..., GPT- | 4), not a
            # number label on the next line.
            # Joined even with only two words: headings and labels do not stop at a hyphen
            # A half line after an inline label or inline expression (Backbones. | Models are ... at temper-,
            # ... This no longer agrees with the MLE under) starts to their right, and the next line returns to
            # the column edge. When this line and the next block share a right edge, they are in the same column
            row_tail = len(_visual_lines(p)) == 1 and abs(p.rect.x1 - b.rect.x1) <= p.size and first_x0 < p.rect.x0
            word_cut = (
                same_size
                and y_close
                and re.search(r"[A-Za-z]{2}-$", p.text.rstrip()) is not None
                and (first.islower() or first.isdigit())
                and first_x0 < p.rect.x1
                and (p.rect.x0 - p.size <= first_x0 or row_tail)
            )
            row_wrap = same_size and row_tail and first.islower() and not _ends_sentence(p.text)
            # A line set off on its own (the indented example The green witch arrived): the next line returns to
            # the left to continue the body text and is not its continuation.
            # An indented first line cut into a block by a margin note reaches the right edge and is as wide as
            # the next line
            displayed_line = (
                len(_visual_lines(p)) == 1 and p.rect.x0 > first_x0 + p.size and p.rect.x1 < b.rect.x1 - p.size
            )
            # In documents with first-line paragraph indents, a line starting flush below a line that runs to the
            # right edge continues the same paragraph,
            # even when the line above happens to break at a sentence end and the next line starts with a
            # capital (in double-spaced manuscripts each line is a block of its own).
            # When the block above holds only the indented first line of a paragraph (cut off by a margin note),
            # a next line one indent further left is also a continuation.
            # A line starting with a change to bold, italic or another size is an unindented subheading
            # (Execution Accuracy (EX) We..., Lemma 2.3.)
            # "Runs to the right edge" is compared with the width of a full body line; a table cell or a short
            # heading would be full against its own narrow box.
            # Numbered labels (EXAMPLE 2.1, Table 3:) also start anew.
            # The next line must sit right below at normal line spacing: one a line further down is a reference
            # with a hanging indent whose continuation line between became a block of its own.
            # A next block with its own hanging indent below is the next reference; a line above whose pieces
            # are several ems apart is a table row.
            # In documents that separate paragraphs by spacing, a new paragraph has extra space above it: only a
            # line right below at normal line spacing is in the same paragraph.
            # A line starting with a number ((21.15) Ok, we fly ...) is an example or an expression, set flush
            # below the line that introduces it as a block of its own, and is not joined with the body text
            # below either.
            # A number followed by a lowercase start is an expression cited in body text wrapped to the line
            # start ((4.5) and (4.6), we have), joined when it sits right below at normal line spacing;
            # an example starting in lowercase ((24.25) her pay jumped ...) has extra space above it
            row_gap = b.rect.y0 - last_y1 - p.line_gap * p.size
            # The line above runs to the right edge and stops at an equals or plus sign (... and λ2 = | 1/2, what
            # is ...), and the next line starts at the same left edge right below at normal line spacing:
            # it is the other half of the expression, joined even when it starts with a digit. A display
            # expression line does not reach the right edge and has more space below; a line starting with a
            # capitalized word or a footnote number is the next sentence or the next entry
            open_end = (
                same_size
                and _open_operator_end(p.text)
                and row_gap <= 0.3 * p.size
                and abs(first_x0 - p.rect.x0) <= 0.5 * p.size
                and last_x1 >= max(p.rect.x1, b.rect.x1) - p.size
                and bool(p.line_rects)
                and _row_width(_visual_lines(p)[-1]) >= b.rect.width * 0.8
                and not _FOOTNOTE_LEAD_RE.match(b.text.lstrip())
                and not re.match(r"[A-Z][a-z]", b.text.lstrip())
            )
            flush_wrap = (
                same_size
                and ((indents and row_gap <= 0.5 * p.size) or (space and row_gap <= space / 2))
                and not _list_item(b.text)
                and not _LABEL_START_RE.match(b.text)
                and not _numbered_line(b.text)
                and not _numbered_line(p.text)
                and not (_EQUATION_TAG_RE.match(b.text.lstrip()) and row_gap > 0.15 * p.size)
                and not _is_float_text(b.text)
                and not _styled_edge(b, 0)
                and not _styled_edge(p, -1)
                and (_wraps_into(p, b) or full_wrap)
                and bool(p.line_rects)
                and _row_width(_visual_lines(p)[-1]) >= measure * 0.9
                and not _hangs(b, blocks[index + 1] if index + 1 < len(blocks) else None)
                and _row_gap(_visual_lines(p)[-1]) <= 1.5 * p.size
            )
            # The block above stops inside an unclosed code fence (```sql opened and not closed), and the next
            # line continues the code, joined even when it starts with a capital
            in_fence = len(_FENCE_RE.findall(p.text)) % 2 == 1 and same_size
            latin = (
                y_close
                and (x_close or centered_tail or math_wrap or centered_comma or word_cut or row_wrap)
                and not list_end
                and not displayed_line
                and (len(p.text.split()) >= 3 or word_cut)  # 1-2 unpunctuated words: heading or label, not a paragraph
                # A continuation has the size of the line above; small text under a large heading is in a figure
                and p.size * 0.7 <= b.size <= p.size / 0.7
                and (
                    colon_wrap
                    or call_wrap
                    or flush_wrap
                    or _MID_ABBREV_RE.search(p.text.rstrip())
                    or not _ends_sentence(p.text)
                )
                and (
                    first.islower()
                    or first in ".),;%,；，"
                    or open_paren
                    or math_wrap
                    or formula_tail
                    or centered_end
                    or dangling
                    or open_end
                    or centered_comma
                    or in_fence
                    or full_wrap
                    or flush_wrap
                    or bracketed
                    or word_cut
                )
                and not _verbatim_seam(p, b)
                # In documents with first-line paragraph indents, a block whose first line is indented one or
                # two ems more than the lines below and starts with a capitalized word is a new paragraph, not
                # joined even when the block above stops at and (half a sentence before an expression). A line
                # starting with an expression (W 1,∞ ...) does not count
                and not (
                    indents
                    and re.match(r"[A-Z][a-z]", b.text.lstrip()) is not None
                    and not _opens_with_math(b)
                    and not word_cut
                    and _first_line_indented(b)
                )
            )
            # An index entry stops at a comma and its page numbers wrap into the next block (see
            # _wraps_from_above): the entry ends at the page numbers
            if (
                p.index
                and p.page == b.page
                and p.text.rstrip().endswith(",")
                and p.line_rects
                and _INDEX_REFS_ONLY_RE.match(b.text.strip())
                and _index_refs_tail(p, b.rect)
            ):
                _absorb_block(p, b)
                break
            # A contents entry is complete at its leader and page number and is not merged with the entries or
            # part titles above and below
            if (
                p.page == b.page
                and p.clip == b.clip
                and not (p.side_note or b.side_note)
                and id(p) not in margins
                and id(b) not in margins
                and p.leader is None
                and b.leader is None
                and not p.entry
                and not (id(p) in listed and id(b) in listed)
                and (title_frag or latin or (y_close and _cjk_wrap(p, b)))
                and not _restyled_seam(p, b)
                and not _new_voice(p, b)
            ):
                _absorb_block(p, b)
                break
        else:
            out.append(b)
    return out


def _body_measure(blocks: list[TextBlock]) -> float:
    """Width of a full body line: the most common width (rounded to points) of the lines in multi-line blocks
    other than their last lines.
    Inline expressions cut a line into several pieces, so whole lines are measured"""
    widths: Counter[int] = Counter()
    for block in blocks:
        rows = _visual_lines(block)
        if len(rows) >= 3:
            widths.update(round(_row_width(row)) for row in rows[:-1])
    return float(widths.most_common(1)[0][0]) if widths else 0.0


def _row_width(row: list[pymupdf.Rect]) -> float:
    return max(rect.x1 for rect in row) - min(rect.x0 for rect in row)


def _row_gap(row: list[pymupdf.Rect]) -> float:
    """The widest gap between two adjacent pieces of a line."""
    pieces = sorted(row, key=lambda rect: rect.x0)
    return max((b.x0 - a.x1 for a, b in zip(pieces, pieces[1:])), default=0.0)


def _hangs(block: TextBlock, nxt: TextBlock | None = None) -> bool:
    """A hanging indent with the first line flush and the lines below indented to the right (one reference
    entry).
    With only one line, look at nxt right below: the continuation lines became a block of their own. A line
    ending at a sentence end is the last line of a paragraph, with an indented new paragraph below"""
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    if not rows:
        return False
    left = min(rect.x0 for rect in rows[0])
    if len(rows) == 1:
        if nxt is None or not nxt.line_rects or nxt.page != block.page or _ends_sentence(block.text):
            return False
        head = nxt.line_rects[0]
        bottom = max(rect.y1 for rect in rows[0])
        return (
            0 <= head.y0 - bottom <= block.size
            and abs(nxt.size - block.size) <= block.size * 0.05
            and block.size * 0.5 < head.x0 - left <= block.size * _MAX_INDENT_EM
        )
    return all(min(rect.x0 for rect in row) - left > block.size * 0.5 for row in rows[1:])


def _paragraph_space(blocks: list[TextBlock]) -> float:
    """In documents that separate paragraphs by spacing, the extra space above a new paragraph beyond normal
    line spacing (in points). Returns 0 for documents not set this way.
    Looks at blocks starting flush below the last line of a paragraph (ending a sentence, short of the right
    edge); it counts only when most of them have extra space of about half a character height or more"""
    gaps: list[tuple[float, float]] = []
    for p, b in zip(blocks, blocks[1:]):
        if p.page != b.page or not p.line_rects or not b.line_rects or abs(p.size - b.size) > p.size * 0.05:
            continue
        if _styled_edge(b, 0) or _LABEL_START_RE.match(b.text) or _NUMBERED_HEAD_RE.match(b.text) or _list_item(b.text):
            continue
        tail, head = p.line_rects[-1], b.line_rects[0]
        if (
            not _ends_sentence(p.text)
            or tail.x1 > max(p.rect.x1, b.rect.x1) - 2 * p.size
            or abs(head.x0 - tail.x0) > p.size * 0.3
        ):
            continue
        gap = head.y0 - tail.y1 - p.line_gap * p.size
        if 0 <= gap <= 1.5 * p.size:
            gaps.append((gap, p.size))
    if len(gaps) < _SPACED_PARAGRAPHS:
        return 0.0
    space, size = sorted(gaps)[len(gaps) // 2]
    return space if space >= 0.4 * size else 0.0


# Paragraph spacing must be measurable below at least this many last lines of paragraphs
_SPACED_PARAGRAPHS = 3


# Minimum number of body words in a continuation line that starts with an expression
_MATH_WRAP_WORDS = 3


# Markdown code fence. LaTeX sets ``` as curly quotes “‘
_FENCE_RE = re.compile(r"```|“‘|‘‘‘")


# Sentences do not stop at these words: when a line ends with one, the next line continues even when it starts
# with a capital (a proper noun)
_DANGLING_WORDS = frozenset(
    "a an the and or of to in for with by from on at as that which whose than its their our his her".split()
)


def _dangling(text: str) -> bool:
    words = text.split()
    return bool(words) and words[-1] in _DANGLING_WORDS


def _opens_with_math(block: TextBlock) -> bool:
    """The first piece is in a math font."""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    return bool(spans) and _span_is_math(spans[0])


def _opens_with_word(block: TextBlock) -> bool:
    """The first piece is not in a math font and starts with a lowercase word. An expression on the next line
    (t3 = t2 + x, hi = t5 + t3) does not count."""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    return (
        bool(spans) and not _span_is_math(spans[0]) and bool(re.match(r"[a-z]{2,}\b(?!\s*[\[(←=∈<>≤≥+])", block.text))
    )


def _wraps_into(block: TextBlock, nxt: TextBlock, indented: bool = False) -> bool:
    """The last line of block (with the numerators, denominators and limits on the same row) shares a left
    edge with the first line of nxt and runs to the right edge of both blocks.
    With indented, the last line of block is the indented first line of a paragraph, and the first line of
    nxt must be one indent further left."""
    lines = block.line_rects or [block.rect]
    tail = lines[-1]
    row = [
        rect for rect in lines if min(rect.y1, tail.y1) - max(rect.y0, tail.y0) >= min(rect.height, tail.height) * 0.5
    ]
    first = nxt.line_rects[0] if nxt.line_rects else nxt.rect
    shift = min(rect.x0 for rect in row) - first.x0
    if indented:
        lead = block.size * 0.5 < shift <= block.size * _MAX_INDENT_EM
    else:
        lead = abs(shift) <= block.size * 0.5
    return lead and max(rect.x1 for rect in row) >= max(block.rect.x1, nxt.rect.x1) - block.size * 0.5


def _beside_wrap(other: TextBlock, head: TextBlock, tail: TextBlock) -> bool:
    """other lies below head, the first half of the paragraph, and beside tail, the second half, horizontally
    within the width of head: the space the paragraph makes room for (a caption). A cell to the right on the
    same table row sits beside head and does not count."""
    if other.page != tail.page or other.rect.y0 < head.rect.y1 - 1:
        return False
    if other.rect.x0 >= tail.rect.x1:
        return head.rect.x1 > other.rect.x0
    if other.rect.x1 <= tail.rect.x0:
        return head.rect.x0 < other.rect.x1
    return False
