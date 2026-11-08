"""Decide which blocks keep their original text: examples, glosses, code, display formulas,
foreign-language passages and the bibliography."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

from .. import langdetect
from .pdf_blocks import (
    _MARK_ITEM_RE,
    CJK_RE,
    TextBlock,
    _column_left,
    _list_item,
    _paragraph_indent,
    _span_is_math,
    _visual_lines,
)
from .pdf_continuations import _plain_words
from .pdf_flow import (
    emphasis_of,
)
from .pdf_glosses import ExampleNumber
from .pdf_layout import PageGeometry, PageScan
from .pdf_math import _BIG_DELIMITER_FONT_RE
from .pdf_merge import _x_overlap
from .pdf_paragraphs import _block_spans
from .pdf_roles import Role
from .pdf_rows import _vivid, _vivid_share
from .pdf_split_lines import _EXAMPLE_NUMBER_RE, _RELATION_RE, _mono_span
from .pdf_types import Span
from .pdf_vision import FormulaDetector

# Maximum length of a bracketed label in an example sentence (Arg0, ArgM-TMP, PER)
_BRACKET_LABEL_CHARS = 10


# Minimum number of letters for detecting a foreign-language passage. An item of an epigraph has only one or
# two words and is detected together with its neighbors
_FOREIGN_LETTERS = 40


_FOREIGN_CJK = 8


_HAN_RE = re.compile(r"[㐀-鿿]")


_KANA_HANGUL_RE = re.compile(r"[぀-ヿ가-힯]")


# "Label: gloss" rows listed below an example sentence (Arg1: Proposition, Cf (U2): {John, piano})
_GLOSS_ROW_RE = re.compile(r"[^\s:：]{1,8}(?:\s?\([^)]*\))?\s?[:：]\s")


def _display_example_ids(blocks: list[TextBlock], geometries: list[PageGeometry], skipped: set[int]) -> set[int]:
    """In a book that indents first lines, a single line or a few left-aligned lines set apart in the body text
    and indented deeper than a paragraph start: example sentences illustrating a linguistic point
    (a clear majority of senators back/VBP the bill, *fair and brutal) and relation notation
    (Giraffe is-a ruminant is-a ungulate ...). The text that follows discusses words of the original
    sentence, so the original is kept. These do not count: a line ending with a colon introduces what
    follows (Then repeat iteratively until convergence:), a different font size marks the source of an
    epigraph, a line centered in the column is a caption (Listing 3 | Prompt for ...), a line with an equals
    or inequality sign explains a formula (x_i contains a particular prefix (... length ≤ 2)), a "label: gloss"
    line is a gloss for the reader, and a line right below a list item is a wrapped line of that item. When
    the blocks right above or below share its left edge, or it or an adjacent block has another block beside
    it on the same line, it is a line of a table, an interlinear gloss or an entry with a hanging indent
    (the table header corpus, the free translation “Yao Ming reaches the finals”), handled by the rules for
    those"""
    indent = _paragraph_indent(blocks)
    if indent is None:
        return set()
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if not block.side_note and block.clip is None:
            by_page.setdefault(block.page, []).append(block)
    found: set[int] = set()
    for page, page_blocks in by_page.items():
        page_blocks.sort(key=lambda block: block.rect.y0)
        geo = geometries[page] if page < len(geometries) else None
        runs: list[list[TextBlock]] = []
        for block in page_blocks:
            if (
                id(block) in skipped
                or len(block.line_rects) != 1
                or block.index
                or block.entry
                or block.leader is not None
                or _list_item(block.text)
                or block.text.rstrip().endswith((":", "："))
                or _GLOSS_ROW_RE.match(block.text)
                or not block.body_size
                or abs(block.size - block.body_size) > 0.5
            ):
                continue
            edge = _column_left(block.rect, geo)
            if edge is None or not indent + block.size * 0.5 < block.rect.x0 - edge <= block.size * 8:
                continue
            last = runs[-1][-1] if runs else None
            if (
                last is not None
                and abs(block.rect.x0 - last.rect.x0) <= 1
                and -2 <= block.rect.y0 - last.rect.y1 <= block.size * 1.5
            ):
                runs[-1].append(block)
            else:
                runs.append([block])
        for run in runs:
            first, last = run[0], run[-1]
            size = first.size
            edge = _column_left(first.rect, geo)
            # A block only joins a run when it has a column edge, which needs a page geometry
            assert geo is not None and edge is not None
            right = geo.column_of(first.rect).x1 if len(geo.columns) >= 2 else geo.frame.x1 if geo.frame else None
            if (
                len(run) == 1
                and right is not None
                and abs((first.rect.x0 + first.rect.x1) / 2 - (edge + right) / 2) < size * 0.5
            ):
                continue
            if any(_RELATION_RE.search(block.text) for block in run):
                continue
            members = {id(block) for block in run}
            # Lines right above and below: the last line of each block above, the first line of each block below
            above = [
                (other, other.line_rects[-1])
                for other in page_blocks
                if id(other) not in members
                and other.rect.y1 <= first.rect.y0 + 2
                and first.rect.y0 - other.rect.y1 <= size * 1.5
                and _x_overlap(other.rect, first.rect)
            ]
            below = [
                (other, other.line_rects[0])
                for other in page_blocks
                if id(other) not in members
                and other.rect.y0 >= last.rect.y1 - 2
                and other.rect.y0 - last.rect.y1 <= size * 1.5
                and _x_overlap(other.rect, last.rect)
            ]
            if (
                not above
                or not below
                or any(_list_item(other.text) for other, _ in above)
                or any(abs(x0 - first.rect.x0) <= 1 for other, row in above + below for x0 in (other.rect.x0, row.x0))
            ):
                continue
            # It or an adjacent block has another block beside it on the same line (pinyin, a gloss, another table
            # column): a line of an interlinear gloss or a table
            if any(
                other is not block
                and not other.side_note
                and not _x_overlap(other.rect, block.rect)
                and min(other.rect.y1, block.rect.y1) - max(other.rect.y0, block.rect.y0) > block.rect.height * 0.5
                and (right is None or edge <= other.rect.x0 < right)
                for block in run + [other for other, _ in above + below]
                for other in page_blocks
            ):
                continue
            found |= members
    return found


# Minimum number of colors when a pattern and an example are matched by color. Links and emphasis in body text
# use only one or two
_KEY_COLORS = 3


def _color_keys(block: TextBlock) -> list[tuple[pymupdf.Rect, tuple[int, ...]]]:
    """The box of each line in the block (spans on the same baseline merged), and the order in which vivid colors
    appear in the line"""
    rows: list[tuple[pymupdf.Rect, list[int]]] = []
    gap = max(block.size * 0.4, 2.0)
    for rect, spans in zip(block.line_rects, block.span_lines or []):
        colors = [int(span.get("color") or 0) for span in spans if str(span.get("text") or "").strip()]
        colors = [color for color in colors if _vivid(color)]
        if rows and abs(rect.y0 - rows[-1][0].y0) <= gap:
            rows[-1] = (rows[-1][0] | rect, rows[-1][1] + colors)
        else:
            rows.append((pymupdf.Rect(rect), colors))
    return [(rect, tuple(dict.fromkeys(colors))) for rect, colors in rows]


def _color_keyed_ids(blocks: list[TextBlock]) -> set[int]:
    """A pattern and an example on two consecutive lines, with matching parts marked by the same colors in the
    same order (PER, POSITION of ORG: above George Marshall, Secretary of State of the United States, with
    PER and George Marshall both blue). Translation changes the word order and the colors no longer match,
    so both lines keep the original text."""
    rows: dict[int, list[tuple[pymupdf.Rect, tuple[int, ...], TextBlock]]] = {}
    for block in blocks:
        if block.clip is None and not block.side_note:
            rows.setdefault(block.page, []).extend((rect, key, block) for rect, key in _color_keys(block))
    found: set[int] = set()
    for page_rows in rows.values():
        page_rows.sort(key=lambda row: row[0].y0)
        for (upper, key, block), (lower, other, below) in zip(page_rows, page_rows[1:]):
            if (
                len(key) >= _KEY_COLORS
                and key == other
                and _x_overlap(upper, lower)
                and -2 <= lower.y0 - upper.y1 <= block.size * 1.5
            ):
                found |= {id(block), id(below)}
    return found


def _item_example_ids(blocks: list[TextBlock]) -> set[int]:
    """Examples on their own lines below a list item (American, Airlines, Tim ... below Bag-of-words in M1
    and M2, M2: -1 spokesman): indented right of the bullet, with most letters in vivid colors, they are
    words taken from an example sentence and keep the original text. Several consecutive lines below the
    same item count. For telling them from wrapped lines of the item see _cut_item_breaks, which also only
    looks at items starting with a bullet."""
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if block.clip is None and not block.side_note:
            by_page.setdefault(block.page, []).append(block)
    found: set[int] = set()
    for page_blocks in by_page.values():
        page_blocks.sort(key=lambda block: (block.rect.y0, block.rect.x0))
        # The item each recognized example line belongs to: the item's left edge
        roots: dict[int, float] = {}
        for index, block in enumerate(page_blocks):
            if len(_visual_lines(block)) != 1 or _MARK_ITEM_RE.match(block.text) or not _vivid_letters(block):
                continue
            above = [
                other
                for other in page_blocks[:index]
                if _x_overlap(other.rect, block.rect) and -2 <= block.rect.y0 - other.rect.y1 <= block.size * 1.5
            ]
            if not above:
                continue
            nearest = max(above, key=lambda other: other.rect.y1)
            root = roots.get(id(nearest), nearest.rect.x0 if _MARK_ITEM_RE.match(nearest.text) else None)
            if root is not None and block.rect.x0 > root + block.size * 0.3:
                roots[id(block)] = root
                found.add(id(block))
    return found


def _vivid_letters(block: TextBlock) -> bool:
    """More than half of the letters in the block are in vivid colors"""
    return _vivid_share(_block_spans(block)) > 0.5


def _arc_figure_ids(blocks: list[TextBlock], scans: Sequence[PageScan]) -> set[int]:
    """Example sentences annotated with arcs (dependency or attention arcs linking word to word) and the labels
    in rounded boxes on the arcs (nsubj, coreference attention). Both ends of an arc sit above words of the
    sentence and would no longer match them in Chinese, so the original is kept. The top of the sentence meets
    the lower ends of the arcs; a label is inside a rounded box that overlaps an arc."""
    found: set[int] = set()
    for block in blocks:
        if block.page >= len(scans):
            continue
        scan = scans[block.page]
        if not scan.arcs:
            continue
        rows = block.line_rects or [block.rect]
        if any(
            abs(arc.y1 - row.y0) <= block.size * 0.4 and row.x0 - 2 <= arc.x0 and arc.x1 <= row.x1 + 2
            for arc in scan.arcs
            for row in rows
        ):
            found.add(id(block))
            continue
        for box in scan.round_boxes:
            framed = pymupdf.Rect(box.x0 - 2, box.y0 - 2, box.x1 + 2, box.y1 + 2)
            if framed.contains(block.rect) and any(not (box & arc).is_empty for arc in scan.arcs):
                found.add(id(block))
                break
    return found


def _example_ids(blocks: list[TextBlock], numbers: dict[int, tuple[ExampleNumber, ...]] | None = None) -> set[int]:
    """Example sentences illustrating a linguistic point: sentences starting with a number such as (24.3) whose
    continuation lines hang right of the number, and sentences with two or more [label ...] annotations
    ([Arg0 The group] agreed ..., [PER Jane] of [ORG United]). The text that follows discusses words of the
    original, and the annotations sit between the words, so the original is kept. numbers holds the example
    numbers at line starts on each page (see example_numbers): when a number is not part of a block, the block
    right of it on the same line is an example; when the whole line formed no block ((22.12) fall.01), the
    indented lines below (Arg1: ..., Ex1: ...) are the example.

    A body paragraph starting with an equation number ((2.4) follows from ...) has continuation lines back at
    the column edge and does not count."""
    found: set[int] = set()
    heads: list[_ExampleHead] = []
    numbers = numbers or {}
    for block in blocks:
        if block.clip is not None or block.side_note:
            continue
        if _EXAMPLE_NUMBER_RE.match(block.text) and _hanging(block):
            found.add(id(block))
            heads.append(
                _ExampleHead(block.page, block.rect, block.rect.x0, _after_number_x0(block), block.size, id(block))
            )
        elif (number := _example_number_before(block, numbers.get(block.page, ()))) is not None:
            found.add(id(block))
            heads.append(_ExampleHead(block.page, block.rect, number.rect.x0, block.rect.x0, block.size, id(block)))
        elif _labeled_brackets(block) >= 2:
            found.add(id(block))
    for page, page_numbers in numbers.items():
        on_page = [block for block in blocks if block.page == page]
        for number in page_numbers:
            if not any(
                min(block.rect.y1, number.row.y1) - max(block.rect.y0, number.row.y0) > number.row.height * 0.5
                and block.rect.x0 < number.row.x1
                and block.rect.x1 > number.row.x0
                for block in on_page
            ):
                heads.append(_ExampleHead(page, number.row, number.rect.x0, number.text_x0, 0.0, None))
    return found | _example_tails(blocks, heads, numbers, _paragraph_indent(blocks))


@dataclass(frozen=True)
class _ExampleHead:
    """The first line of a numbered example. block_id is the block it is in, or None when the line formed no
    block, in which case size is taken from the first block below. number_x0 is the left edge of the number,
    text_x0 where the sentence after the number starts"""

    page: int
    rect: pymupdf.Rect
    number_x0: float
    text_x0: float
    size: float
    block_id: int | None


def _example_number_before(block: TextBlock, numbers: tuple[ExampleNumber, ...]) -> ExampleNumber | None:
    """The first line of the block is right of a standalone example number on the same line"""
    first = block.line_rects[0] if block.line_rects else block.rect
    return next(
        (
            number
            for number in numbers
            if number.rect.x1 <= first.x0
            and first.x0 - number.rect.x1 <= block.size * 5
            and min(number.rect.y1, first.y1) - max(number.rect.y0, first.y0)
            > min(number.rect.height, first.height) * 0.5
        ),
        None,
    )


def _example_tails(
    blocks: list[TextBlock],
    heads: list[_ExampleHead],
    numbers: dict[int, tuple[ExampleNumber, ...]],
    indent: float | None = None,
) -> set[int]:
    """Later lines of a numbered example that formed separate blocks (the second sentence of (24.14), wine.
    wrapped within an example, the IT and JA lines comparing languages, Arg0: ... and Ex1: ... below a
    frame file entry, the free translation “Yao Ming reaches the finals” below a gloss): a block right
    below the example, in the same size, starting right of the midpoint between the number and the sentence,
    belongs to the example. Body text returns to the left edge of the number; indent is the first-line indent
    of body paragraphs, and a block starting that far in is the next paragraph. Stops at the next number."""
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if block.clip is None and not block.side_note:
            by_page.setdefault(block.page, []).append(block)
    for page_blocks in by_page.values():
        page_blocks.sort(key=lambda block: block.rect.y0)
    head_ids = {head.block_id for head in heads}
    tails: set[int] = set()
    for head in heads:
        left = (head.number_x0 + head.text_x0) / 2
        stop = min(
            (
                number.rect.y0
                for number in numbers.get(head.page, ())
                if number.rect.y0 > head.rect.y0 + number.rect.height * 0.5
                and number.rect.x0 < head.rect.x1
                and number.rect.x1 > head.rect.x0
            ),
            default=math.inf,
        )
        last, size = head.rect, head.size
        for block in by_page.get(head.page, ()):
            if id(block) == head.block_id or block.rect.y0 < head.rect.y0 + head.rect.height * 0.5:
                continue
            if block.rect.x0 >= head.rect.x1 or block.rect.x1 <= head.rect.x0:
                continue  # another column
            size = size or block.size
            if indent is not None:
                left = max(left, head.number_x0 + indent + size * 0.5)
            # A line set after a blank line within the example (Ex1: below the Arg lines) starts aligned with the
            # previous line
            gap = size * (1.6 if last is not head.rect and abs(block.rect.x0 - last.x0) <= 1 else 0.6)
            if (
                block.rect.y0 - last.y1 > gap
                or block.rect.y0 >= stop - 1
                or block.rect.x0 < left
                or abs(block.size - size) > size * 0.1
                or _EXAMPLE_NUMBER_RE.match(block.text)
                or id(block) in head_ids
            ):
                break
            tails.add(id(block))
            last = block.rect
    return tails


def _after_number_x0(block: TextBlock) -> float:
    """x coordinate of the first character after the example number. When the number and the sentence share a
    span, it is estimated from the character count."""
    matched = _EXAMPLE_NUMBER_RE.match(block.text)
    # The number and the sentence may be in two lines at the same height
    lines = block.span_lines or [[]]
    first = [span for span in lines[0] if str(span.get("text") or "").strip() and span.get("bbox") is not None]
    top = pymupdf.Rect(first[0]["bbox"]) if first else None
    spans = sorted(
        (
            span
            for line in lines
            for span in line
            if str(span.get("text") or "").strip()
            and span.get("bbox") is not None
            and top is not None
            and abs(pymupdf.Rect(span["bbox"]).y1 - top.y1) <= block.size * 0.3
        ),
        key=lambda span: span["bbox"][0],
    )
    if matched is None or not spans:
        return block.rect.x0 + block.size * 3
    label = len(matched.group(0)[:-1].rstrip())
    seen = 0
    for span in spans:
        text = str(span["text"])
        start = len(text) - len(text.lstrip())
        stripped = text.strip()
        if seen + len(stripped) > label:
            box = pymupdf.Rect(span["bbox"])
            offset = max(label - seen, 0) + start
            while offset < len(text) and text[offset].isspace():
                offset += 1
            x: float = box.x0 + box.width * offset / max(len(text), 1)
            return x
        seen += len(stripped)
    return block.rect.x1


def _hanging(block: TextBlock) -> bool:
    """Every line after the first starts right of the first line. A single line also counts."""
    starts = [min(rect.x0 for rect in row) for row in sorted(_visual_lines(block), key=lambda row: row[0].y0)]
    return all(x0 > starts[0] + block.size for x0 in starts[1:])


def _labeled_brackets(block: TextBlock) -> int:
    """Number of times a left bracket is followed by a smaller letter label and the label by more words.
    The superscript in a layer index W[1] is a digit and [EOS] holds a single word; neither counts."""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    count = 0
    for left, label, after in zip(spans, spans[1:], spans[2:]):
        text = str(label.get("text") or "").strip()
        if (
            str(left.get("text") or "").rstrip().endswith("[")
            and float(label.get("size") or 0) < block.size * 0.95
            and len(text) <= _BRACKET_LABEL_CHARS
            and " " not in text
            and sum(c.isalpha() for c in text) >= 2
            and not str(after.get("text") or "").lstrip().startswith("]")
        ):
            count += 1
    return count


# Realizations of arguments listed line by line: AGENT/Subject, THEME/Object, INSTRUMENT/PPwith
_ROLE_PATTERN_RE = re.compile(r"[A-Z]{3,}/\w+(?:,\s*[A-Z]{3,}/\w+)*")


def _role_pattern_ids(blocks: list[TextBlock]) -> set[int]:
    """Role notation listed line by line (AGENT/Subject, THEME/Object). The text that follows discusses the role
    names, so the original is kept. Examples with roles marked below the words are found as glosses, see
    interlinear_glosses."""
    return {
        id(block)
        for block in blocks
        if block.clip is None
        and not block.side_note
        and len(block.line_rects) <= 3
        and _ROLE_PATTERN_RE.fullmatch(block.text.strip())
    }


def _concordance_ids(blocks: list[TextBlock]) -> set[int]:
    """Corpus concordance lines: a bold word in the middle of each line, left-aligned across lines, with its
    context from the corpus on both sides (is traditionally followed by | cherry | pie, a traditional
    dessert). The contexts are cut-off fragments that do not form sentences when read together, and the
    text that follows discusses the middle word, so the original is kept."""
    found: set[int] = set()
    for block in blocks:
        if block.clip is not None or block.side_note:
            continue
        spans = sorted(
            (
                span
                for line in block.span_lines or []
                for span in line
                if str(span.get("text") or "").strip() and span.get("origin") is not None
            ),
            key=lambda span: float(span["origin"][1]),
        )
        rows: list[list[Span]] = []
        for span in spans:
            if rows and float(span["origin"][1]) - float(rows[-1][0]["origin"][1]) <= block.size * 0.3:
                rows[-1].append(span)
            else:
                rows.append([span])
        if len(rows) < 3:
            continue
        lefts = []
        for row in rows:
            row.sort(key=lambda span: float(span["bbox"][0]))
            bold = [
                i
                for i, span in enumerate(row)
                if emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)).bold
            ]
            # The bold words are contiguous, with context on both sides
            if not bold or bold != list(range(bold[0], bold[-1] + 1)) or bold[0] == 0 or bold[-1] == len(row) - 1:
                break
            lefts.append(float(row[bold[0]]["bbox"][0]))
        else:
            if max(lefts) - min(lefts) <= block.size * 0.2:
                found.add(id(block))
    return found


def _foreign_ids(blocks: list[TextBlock]) -> set[int]:
    """Passages in a language other than the document's: a Spanish epigraph, a German quotation or Basque
    examples in an English book. The author kept them in the original on purpose and usually gives a
    translation nearby, so the original is kept.

    Epigraphs are set in two columns with one or two words per item, too little to detect alone. Adjacent
    blocks on the same page with the same size and color are detected together, and if the group is foreign
    the whole group keeps the original; a foreign passage within body text is detected on its own. Too few
    letters or too low a confidence does not count, and nothing is decided when the document language
    cannot be detected.
    """
    doc_lang = langdetect.detect_document([block.text for block in blocks])
    if not doc_lang:
        return set()

    cjk_doc = doc_lang.split("-")[0] in ("zh", "ja", "ko")

    def foreign(text: str) -> bool:
        # One Han character carries as much as several letters: a Chinese sentence of ten or so characters
        # without Latin letters in a Western book (the original of an epigraph) is detected despite the
        # letter count
        if not cjk_doc and len(CJK_RE.findall(text)) >= _FOREIGN_CJK and not re.search(r"[A-Za-z]", text):
            return True
        lang = langdetect.detect(text, min_letters=_FOREIGN_LETTERS)
        return bool(lang and lang.split("-")[0] != doc_lang.split("-")[0])

    found = {id(block) for block in blocks if foreign(block.text)}
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if not block.side_note:
            by_page.setdefault(block.page, []).append(block)
    for page_blocks in by_page.values():
        groups: list[list[TextBlock]] = []
        for block in page_blocks:
            reach = block.rect + (-block.size, -block.size * 0.6, block.size, block.size * 0.6)
            near = [
                group
                for group in groups
                if any(
                    abs(other.size - block.size) < 0.3 and other.color == block.color and reach.intersects(other.rect)
                    for other in group
                )
            ]
            merged = [block]
            for group in near:
                groups.remove(group)
                merged.extend(group)
            groups.append(merged)
        for group in groups:
            if len(group) > 1 and foreign("\n".join(block.text for block in group)):
                found.update(id(block) for block in group)
    if not cjk_doc:
        found |= _rendering_ids(blocks, found)
    return found


def _han_line(block: TextBlock) -> bool:
    """The first line of the block is a Chinese sentence: eight or more Han characters, with no Latin letters,
    kana or hangul. Pinyin may follow below"""
    first = "".join(str(span.get("text") or "") for span in block.span_lines[0]) if block.span_lines else block.text
    return (
        len(_HAN_RE.findall(first)) >= _FOREIGN_CJK
        and not re.search(r"[A-Za-z]", first)
        and not _KANA_HANGUL_RE.search(first)
    )


def _rendering_ids(blocks: list[TextBlock], found: set[int]) -> set[int]:
    """The author's Western translation next to an original Chinese sentence (see _han_line): starting right of
    it on the same line (the epigraph 荃者所以在鱼 | Nets are for fish;), or right below it with the same left
    edge (In my backyard, ... below the characters and pinyin in an example box), together with left-aligned
    continuation lines below. The original is right there, so translating back into Chinese is pointless and
    the original is kept."""
    sources = [block for block in blocks if _han_line(block)]
    out: set[int] = set()
    for source in sources:
        for block in blocks:
            if block.page != source.page or id(block) in found or id(block) in out:
                continue
            r, s = block.rect, source.rect
            beside = (
                min(r.y1, s.y1) - max(r.y0, s.y0) > min(r.height, s.height) * 0.5
                and 0 <= r.x0 - s.x1 <= 3 * source.size
            )
            below = 0 <= r.y0 - s.y1 + 1 <= source.size * 0.5 and abs(r.x0 - s.x0) <= source.size
            if beside or below:
                out.add(id(block))
    # Continuation lines: right below, same left edge, same size
    grown = True
    while grown:
        grown = False
        for block in blocks:
            if id(block) in found or id(block) in out:
                continue
            for other in blocks:
                if (
                    id(other) in out
                    and other.page == block.page
                    and abs(block.rect.x0 - other.rect.x0) <= 1
                    and 0 <= block.rect.y0 - other.rect.y1 + 1 <= block.size * 0.5
                    and abs(block.size - other.size) <= 0.3
                ):
                    out.add(id(block))
                    grown = True
                    break
    return out


def _latin_chars(text: str) -> int:
    """Count only Latin letters and digits when counting monospaced characters. Han characters, kana and
    full-width punctuation also carry the monospace flag and are not counted."""
    return sum(1 for ch in text if ch.isalnum() and not CJK_RE.match(ch))


def _code_panel_ids(blocks: list[TextBlock], scans: list[PageScan]) -> set[int]:
    """At least 80% of the text on a shaded area is monospaced: program code, code point tables, command output.
    The whole area keeps the original text.

    Counted over the whole area: comment lines and line numbers inside it fall short of 80% on their own and
    keep the original along with the area. Shaded areas with much text in the body font are not affected.
    """
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    found: set[int] = set()
    for pno, page_blocks in by_page.items():
        for area in scans[pno].shades:
            inside = [block for block in page_blocks if (block.rect & area).get_area() > block.rect.get_area() * 0.6]
            spans = [span for block in inside for span in _block_spans(block) if not _span_is_math(span)]
            total = mono = 0
            for span in spans:
                count = _latin_chars(str(span["text"]))
                total += count
                if _mono_span(span):
                    mono += count
            if total and mono >= total * 0.8:
                found.update(id(block) for block in inside)
    return found


# Glyphs of the big-delimiter font that also appear inline: wide hats and tildes (b-g) and the two smallest
# radicals (p, q). Only extended brackets and summation or integral signs show a display formula. The
# text-size summation sign counts too: the denominator of a displayed fraction uses it
_BIG_GLYPH_RE = re.compile(r"[^\sb-gpq]")


# With this many words outside formulas the block is a paragraph. Inline integral signs use the same font, and
# right columns and list continuation lines also start away from the left edge of the text frame
_DISPLAY_FORMULA_MAX_WORDS = 24


# Share of a block's area that must fall in a display formula box of the formula detection model for the block
# to count as a display formula (like _COVER for the layout model)
_ISOLATED_COVER = 0.6


# Function words: articles, copulas, prepositions, conjunctions, pronouns. A display formula box with this many
# of them holds an explanatory sentence
_FUNCTION_WORDS = frozenset(
    """
a an the is are was were be been being am do does did has have had can could will would shall should may might must
of to in for with by from on at as into onto over under about between than that which who whom whose what where
when why how this these those it its they their them we our us you your he his she her him i my not no nor
and or but if then so each every any all some such much many
""".split()
)


_SENTENCE_FUNCTION_WORDS = 3


def _function_words(text: str) -> int:
    return sum(1 for word in re.findall(r"[A-Za-z]+", text) if word.lower() in _FUNCTION_WORDS)


def _display_formula_ids(blocks: list[TextBlock], geometries: list[PageGeometry]) -> set[int]:
    """Display formulas set in the body font: the brace of a piecewise function groups several lines, and
    min or ins-cost(target[j]) in them look like words, so word counts miss them. A block with an extended
    delimiter that does not start at the left edge of the text frame is a formula, and so are blocks inside
    its box (the condition lines right of the brace)."""
    found: list[TextBlock] = []
    for block in blocks:
        geo = geometries[block.page] if block.page < len(geometries) else None
        frame = geo.frame if geo is not None else None
        if frame is None:
            continue
        # A wide centered formula leaves only a little space on the left, as much as on the right; a paragraph
        # runs to the right edge
        left, right = block.rect.x0 - frame.x0, frame.x1 - block.rect.x1
        centered = left > block.size and abs(left - right) <= block.size
        if left <= block.size * 2 and not centered:
            continue
        if (
            any(
                _BIG_DELIMITER_FONT_RE.search(str(span.get("font") or ""))
                and _BIG_GLYPH_RE.search(str(span.get("text") or ""))
                for line in block.span_lines or []
                for span in line
            )
            and _plain_words(block) < _DISPLAY_FORMULA_MAX_WORDS
        ):
            found.append(block)
    ids = {id(block) for block in found}
    for block in blocks:
        area = block.rect.width * block.rect.height
        if id(block) in ids or area < 1:
            continue
        for formula in found:
            if formula.page != block.page:
                continue
            hit = block.rect & (formula.rect + (-2, -2, 2, 2))
            if not hit.is_empty and hit.width * hit.height >= area * 0.7:
                ids.add(id(block))
                break
    return ids


def _isolated_formula_ids(
    blocks: list[TextBlock],
    formulas: FormulaDetector | None,
    roles: Sequence[Role] | None = None,
) -> set[int]:
    """Blocks that the formula detection model boxes as display formulas keep the original text: most of the
    block's area falls in the box. Display formulas typeset in Word use the body font, so fonts do not reveal
    them, and without extended delimiters _display_formula_ids misses them too.

    Sentences explaining something inside the box are translated ((where c is the correct class),
    L(ŷ, y) = How much ŷ differs from the true y). Whether text is a sentence is decided by function words,
    not by the word count: the words of a formula are content words and variable names
    (perplexity_A(T) = P_A(red red red red blue)), and counting words would take it for a sentence and
    translate it as "困惑度 = P(红红红红蓝)". Headings and captions do not count: the model often boxes large
    slide titles (Phoenix + Calcite) as display formulas."""
    if formulas is None:
        return set()
    ids: set[int] = set()
    for index, block in enumerate(blocks):
        area = block.rect.width * block.rect.height
        if area < 1 or roles is not None and roles[index] in (Role.HEADING, Role.CAPTION):
            continue
        if _function_words(block.text) >= _SENTENCE_FUNCTION_WORDS:
            continue
        for rect in formulas.isolated.get(block.page, ()):
            hit = block.rect & rect
            if not hit.is_empty and hit.width * hit.height >= area * _ISOLATED_COVER:
                ids.add(id(block))
                break
    return ids


def _in_gloss(rect: pymupdf.Rect, glosses: tuple[pymupdf.Rect, ...]) -> bool:
    area = rect.width * rect.height
    for gloss in glosses:
        hit = rect & (gloss + (-1, -1, 1, 1))
        if area >= 1 and not hit.is_empty and hit.width * hit.height >= area * 0.7:
            return True
    return False


def _gloss_ids(blocks: list[TextBlock], glosses: dict[int, tuple[pymupdf.Rect, ...]]) -> set[int]:
    """Blocks inside the two lines of an interlinear gloss."""
    return {id(block) for block in blocks if _in_gloss(block.rect, glosses.get(block.page, ()))}
