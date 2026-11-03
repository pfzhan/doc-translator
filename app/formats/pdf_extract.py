"""Read PDF pages into text blocks: detect contents and index pages, the text frame and margin notes, and run
the line and paragraph stages (pdf_rows, pdf_paragraphs) on each page."""

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace

import pymupdf

from .pdf_blocks import (
    _LINE_GAP,
    CJK_RE,
    TextBlock,
    _attach_ink,
    _by_chars,
    _ends_sentence,
    _join_lines,
    _list_item,
    _small_caps_sizes,
    _span_is_math,
)
from .pdf_frame import FrameLine, TextFrame, margin_notes, note_groups, note_pitch, text_frames
from .pdf_layout import (
    PageGeometry,
    PageScan,
    RuledTable,
    cell_containing,
    margin_skips,
    reading_order,
    scan_page,
)
from .pdf_math import (
    _fraction_parts,
    _line_box,
    _mark_detected_formulas,
    _mark_formula_only_fonts,
    _mark_math_alphabet,
    _mark_math_roman,
    _mark_overlays,
    _mark_radicands,
    _mark_underlines,
    _mark_visible_spaces,
    _PageChars,
    _script_parts,
    _split_accents,
)
from .pdf_paragraphs import (
    _donate_punct_rects,
    _line_lists,
    _merge_caption_fragments,
    _merge_cells,
    _merge_continuations,
    _merge_line_tails,
    _merge_same_line_sentences,
    _merge_visual_lines,
    _split_affiliation_tails,
    _split_table_blocks,
    _untranslatable,
    _wraps_from_above,
)
from .pdf_rows import (
    _INDEX_REF_RE,
    _INDEX_REFS_ONLY_RE,
    _LEADER_RE,
    NO_TEXT_RE,
    _cut_after,
    _cut_at_posts,
    _cut_header_cells,
    _cut_item_breaks,
    _cut_leaders,
    _cut_line_list,
    _cut_runs,
    _drop_number_tails,
    _grid_cells,
    _index_entry_ends,
    _left_to_right,
    _near_votes,
    _number_cell,
    _rejoin_split_lines,
    _split_lines,
)
from .pdf_type3 import Type3Reader, recover_type3_page
from .pdf_vision import FormulaDetector


def _latin_page(data: dict) -> bool:
    """More than 90% of the characters on the page are not CJK"""
    text = "".join(
        span["text"]
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
    )
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and len(CJK_RE.findall(text)) < letters * 0.1


def _has_leader_line(data: dict) -> bool:
    return any(
        _LEADER_RE.search("".join(span["text"] for span in line["spans"]))
        for block in data.get("blocks", [])
        for line in block.get("lines", [])
    )


# Title of a contents page
_TOC_TITLE_RE = re.compile(r"(?:(?:table|summary) of )?contents|list of (?:figures|tables)|目\s*录", re.I)


# A page needs at least this many entries in its page number column to count as contents
_TOC_MIN_ENTRIES = 5


# A page is an index page only when it has at least this many lines and this share of them end with page numbers
_INDEX_MIN_LINES = 15


_INDEX_SHARE = 0.4


def _index_page(data: dict) -> bool:
    """An index page at the back of a book: 40% or more of the lines end with ", page number". Body lines
    rarely stop at a comma followed by a number."""
    texts = [
        "".join(span["text"] for span in line["spans"]).strip()
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
    ]
    texts = [text for text in texts if text]
    if len(texts) < _INDEX_MIN_LINES:
        return False
    return sum(bool(_INDEX_REF_RE.search(text)) for text in texts) >= len(texts) * _INDEX_SHARE


def _index_ref_rows(data: dict) -> list[pymupdf.Rect]:
    """Lines on an index page that hold only page numbers. PyMuPDF often puts such a line and the entry on the
    line above into two blocks, so they are collected across the whole page."""
    return [
        _line_box(line)
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        if _INDEX_REFS_ONLY_RE.match("".join(span["text"] for span in line["spans"]).strip())
    ]


def _toc_pages(pages: list[dict]) -> set[int]:
    """Contents pages without leaders: a title line, and page numbers far from the rest of each line,
    right-aligned and increasing down the page.
    When the contents span several pages, the later pages have no title; a page that follows a contents page
    and has the same page number column also counts.
    A page number column alone easily mistakes a table with a column of integers for contents, so a title or a
    preceding contents page is required."""
    found: set[int] = set()
    previous = False
    for pno, data in enumerate(pages):
        lines = [
            line
            for block in data.get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            if abs(line["dir"][1]) <= 0.01
        ]
        titled = any(_TOC_TITLE_RE.fullmatch("".join(span["text"] for span in line["spans"]).strip()) for line in lines)
        if _page_number_column(lines) and (titled or previous):
            found.add(pno)
            previous = True
        else:
            previous = _has_leader_line(data)
    return found


def _page_number_column(lines: list[dict]) -> bool:
    """Page numbers at the end of lines: a line holding only one number, with other text more than two ems to
    its left on the same row.
    Such page numbers form a right-aligned column that mostly does not decrease from top to bottom. When a
    list of tables follows the contents, the page numbers start over, so a few drops are allowed."""
    found: list[tuple[float, float, int]] = []
    for line in lines:
        text = "".join(span["text"] for span in line["spans"]).strip()
        if not re.fullmatch(r"\d{1,4}", text):
            continue
        box = pymupdf.Rect(line["bbox"])
        size = max((float(span["size"]) for span in line["spans"]), default=10.0)
        for other in lines:
            if other is line:
                continue
            left = pymupdf.Rect(other["bbox"])
            overlap = min(left.y1, box.y1) - max(left.y0, box.y0)
            if overlap >= min(left.height, box.height) * 0.5 and box.x0 - left.x1 > size * 2:
                found.append((box.x1, box.y0, int(text)))
                break
    best = max(
        ([(y0, number) for x1, y0, number in found if abs(x1 - edge) <= 1.5] for edge in {x for x, _, _ in found}),
        key=len,
        default=[],
    )
    numbers = [number for _, number in sorted(best)]
    drops = sum(1 for a, b in zip(numbers, numbers[1:]) if b < a)
    return len(numbers) >= _TOC_MIN_ENTRIES and drops <= len(numbers) // 5


def _number_columns(data: dict) -> list[float]:
    """Center lines of the number columns in tables on this page: three or more number cells whose centers
    fall at the same x (within one point)."""
    centers = sorted(
        (float(span["bbox"][0]) + float(span["bbox"][2])) / 2
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if str(span["text"]).strip()
        and _number_cell(str(span["text"]))
        and not str(span["text"]).rstrip().endswith(".")
    )
    columns: list[float] = []
    run: list[float] = []
    for center in centers:
        if run and center - run[0] > 1:
            if len(run) >= 3:
                columns.append(sum(run) / len(run))
            run = []
        run.append(center)
    if len(run) >= 3:
        columns.append(sum(run) / len(run))
    return columns


def _block_size(main: dict, span_lines: list[list[dict]]) -> float:
    """Font size of the block: the size of its longest piece. In a small-caps heading (INITIALIZATION, with a
    larger initial) the longest piece is the small part, so the size of the capitals is used. The capitals
    count only on the same baseline; one offset downward is a subscript (BERT_LARGE)."""
    size = float(main["size"])
    big = _small_caps_sizes(span_lines).get(id(main))
    if big is None or main.get("origin") is None:
        return size
    base = float(main["origin"][1])
    for line in span_lines:
        for span in line:
            if (
                span is not main
                and str(span.get("text") or "").strip()
                and span.get("origin") is not None
                and abs(float(span.get("size") or 0) - big) <= 0.1
                and abs(float(span["origin"][1]) - base) <= big * 0.1
            ):
                return big
    return size


# Tolerance for aligning lines in a wide block with the position of margin notes. The first column of a table
# is often indented slightly more than the caption label (99 against 96)
_WIDE_ALIGN = 4.0


def _frame_lines(data: dict) -> tuple[list[FrameLine], list[tuple[dict, dict]]]:
    """Lines used to vote on the text frame, and the block and line dict of each line."""
    lines: list[FrameLine] = []
    refs: list[tuple[dict, dict]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            if abs(line["dir"][1]) > 0.01:
                continue
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = "".join(span["text"] for span in line["spans"])
            lines.append(FrameLine(pymupdf.Rect(line["bbox"]), text, max(span["size"] for span in spans)))
            refs.append((block, line))
    return lines, refs


def _detach_margin_notes(
    data: dict,
    lines: list[FrameLine],
    refs: list[tuple[dict, dict]],
    notes: set[int],
    pitches: dict[float, float] | None = None,
) -> None:
    """Move margin note lines out of their original blocks; each margin note becomes its own block, placed at
    the end of the page.

    Left in place, a note would separate the two pieces of body text it sits between, and the two pieces could
    not be joined back into one paragraph.
    """
    if not notes:
        return
    picked = sorted(notes)
    touched: set[int] = set()
    for group in note_groups([lines[index] for index in picked], pitches):
        members = [refs[picked[index]] for index in group]
        rect = pymupdf.Rect()
        for block, line in members:
            block["lines"].remove(line)
            touched.add(id(block))
            rect |= pymupdf.Rect(line["bbox"])
        data["blocks"].append(
            {
                "type": 0,
                "bbox": tuple(rect),
                "lines": [line for _, line in members],
                "side_note": True,
            }
        )
    _rejoin_note_cuts(data, touched)
    kept = []
    for block in data["blocks"]:
        if block.get("type") == 0:
            if not block["lines"]:
                continue
            box = pymupdf.Rect()
            for line in block["lines"]:
                box |= pymupdf.Rect(line["bbox"])
            block["bbox"] = tuple(box)
        kept.append(block)
    data["blocks"] = kept


# A line this long counts as a body line; at least this many of them starting at the same x (within
# _EDGE_NEAR) make an edge of the body text on this page
_WIDE_TEXT = 40


_WIDE_ROWS = 3


_EDGE_NEAR = 2


def _wide_block_lines(
    lines: list[FrameLine],
    refs: list[tuple[dict, dict]],
    notes: set[int],
    frame: TextFrame | None,
) -> set[int]:
    """Lines of wide tables and wide captions reaching into the margin that fall where margin notes would be.
    When another line in the block starts at the same edge and runs into the text frame, this line is part of
    the block (the label of Figure 3.7, with the next caption line starting at the same x), not a margin note.
    The same holds when another line starts deep in the margin and runs into the text frame: body text does
    not start in the margin, so the whole block reaches into the margin (the label of Figure 6.17 followed by
    the caption, with the tion of on the next line also in the margin)."""
    if frame is None:
        return set()
    note_lines = {id(refs[index][1]) for index in notes}
    # The outermost x where body text on this page starts and ends (several lines are there). The text frame
    # is voted across pages on the same side, and the body text of some pages extends beyond it
    long_lines = [line.rect for line in lines if len(line.text.strip()) >= _WIDE_TEXT]
    starts = Counter(round(rect.x0) for rect in long_lines)
    ends = Counter(round(rect.x1) for rect in long_lines)
    body_x0 = min([frame.x0, *(x for x in starts if _near_votes(starts, x, _EDGE_NEAR) >= _WIDE_ROWS)])
    body_x1 = max([frame.x1, *(x for x in ends if _near_votes(ends, x, _EDGE_NEAR) >= _WIDE_ROWS)])
    kept: set[int] = set()
    for index in notes:
        block, _line = refs[index]
        rect = lines[index].rect
        left = rect.x1 <= frame.x0
        for other in block["lines"]:
            if id(other) in note_lines:
                continue
            box = pymupdf.Rect(other["bbox"])
            deep = lines[index].size * 2
            if left and (abs(box.x0 - rect.x0) <= _WIDE_ALIGN or box.x0 <= body_x0 - deep) and box.x1 > frame.x0:
                kept.add(index)
                break
            if not left and (abs(box.x1 - rect.x1) <= _WIDE_ALIGN or box.x1 >= body_x1 + deep) and box.x0 < frame.x1:
                kept.add(index)
                break
        if index in kept:
            continue
        # When an inline expression cuts a caption line into several blocks, the piece left in the margin
        # (that w[1]) is a block of its own.
        # When the line right above or below starts at the same x and runs into the text frame, this piece is
        # also a caption line
        pitch = lines[index].size * 0.6
        for other_index, other in enumerate(lines):
            if other_index in notes:
                continue
            box = other.rect
            if min(abs(box.y0 - rect.y1), abs(rect.y0 - box.y1)) > pitch:
                continue
            if left and abs(box.x0 - rect.x0) <= _WIDE_ALIGN and box.x1 > frame.x0:
                kept.add(index)
                break
            if not left and abs(box.x1 - rect.x1) <= _WIDE_ALIGN and box.x0 < frame.x1:
                kept.add(index)
                break
    return kept


def _rejoin_note_cuts(data: dict, touched: set[int]) -> None:
    """PyMuPDF often cuts a paragraph into two blocks at the line where a margin note sits. Once the notes are
    removed, two blocks right above each other and cut mid-sentence are joined back into one: a line break
    before a capital (Standard / Code for ...) is not recognized by block merging. When the upper block ends
    with a sentence end, or the lower block starts indented, they are two paragraphs."""
    blocks = [block for block in data["blocks"] if block.get("type") == 0 and not block.get("side_note")]
    index = 0
    while index < len(blocks) - 1:
        upper, lower = blocks[index], blocks[index + 1]
        if (id(upper) in touched or id(lower) in touched) and _note_cut(upper, lower):
            upper["lines"].extend(lower["lines"])
            lower["lines"] = []
            touched.add(id(upper))
            del blocks[index + 1]
            continue
        index += 1


def _note_cut(upper: dict, lower: dict) -> bool:
    above = [line for line in upper["lines"] if abs(line["dir"][1]) <= 0.01]
    below = [line for line in lower["lines"] if abs(line["dir"][1]) <= 0.01]
    if not above or not below:
        return False
    last = pymupdf.Rect(max(above, key=lambda line: line["bbox"][3])["bbox"])
    first = pymupdf.Rect(min(below, key=lambda line: line["bbox"][1])["bbox"])
    size = max((span["size"] for line in above for span in line["spans"] if span["text"].strip()), default=0)
    lower_size = max((span["size"] for line in below for span in line["spans"] if span["text"].strip()), default=0)
    if size <= 0 or abs(lower_size - size) > size * 0.15:
        return False
    if not -size * 0.3 <= first.y0 - last.y1 <= size * 0.6:
        return False
    # A lower block indented to the right is a new paragraph; one back to the left at the column edge can be
    # the continuation of a line with a first-line indent
    if not last.x0 - size * 3 <= first.x0 <= last.x0 + 2:
        return False
    # Continuation lines of a list item align with the item text; a lower block starting further left is body
    # text after the list has ended
    head = min(above, key=lambda line: line["bbox"][1])
    if _list_item("".join(span["text"] for span in head["spans"])) and first.x0 < last.x0 - size * 0.5:
        return False
    # A lower block with its own first-line indent (the first line right of the lines below) is a new
    # paragraph: a new paragraph following the translation of an example
    rest = [line["bbox"][0] for line in below if line["bbox"][1] >= first.y1 - 1 and line["bbox"][2] > first.x0]
    if rest and first.x0 > min(rest) + size * 0.5:
        return False
    tail = "".join(span["text"] for line in above if pymupdf.Rect(line["bbox"]) == last for span in line["spans"])
    return not _ends_sentence(tail)


def extract_blocks(doc: pymupdf.Document) -> list[TextBlock]:
    return _extract(doc)[0]


def _extract(
    doc: pymupdf.Document,
    on_page: Callable[[int, int], None] | None = None,
    formulas: FormulaDetector | None = None,
) -> tuple[list[TextBlock], list[TextFrame | None], list[PageScan]]:
    """Text blocks, and for each page the text frame and the tables, figures and words. on_page(pages read,
    total pages) is called after each page is read.
    formulas is the formula detection model; when given, the inline formulas it boxes are marked on the
    characters. The model runs ahead on the next page in a background thread (see FormulaDetector.pages)."""
    blocks = []
    scans: list[PageScan] = []
    tables_by_page: list[tuple[RuledTable, ...]] = []
    type3 = Type3Reader(doc)
    pages: list[dict] = []
    pending_cells: set[int] = set()
    detected = formulas.pages() if formulas is not None else None
    try:
        for page in doc:
            scans.append(scan_page(page))
            tables_by_page.append(scans[-1].tables)
            data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
            _attach_ink(page, data)
            page_chars = _PageChars(page)
            _mark_underlines(page_chars, data, scans[-1].strokes)
            _mark_radicands(page_chars, data, scans[-1].strokes)
            _mark_overlays(page_chars, data)
            _split_accents(page_chars, data)
            if detected is not None:
                _mark_detected_formulas(page_chars, data, next(detected))
            _mark_visible_spaces(page_chars, data, scans[-1].visible_spaces)
            # Bitmap glyphs with a custom encoding extract as character codes. They are recognized as letters
            # before the paragraph splitting below.
            recover_type3_page(page, data, type3)
            pages.append(data)
            if on_page is not None:
                on_page(len(pages), doc.page_count)
    finally:
        if detected is not None:
            # Stop background inference. When stopped midway (see background.check_stop), this thread also
            # waits for it to finish the page in hand
            detected.close()
    _mark_math_roman(pages)
    _mark_math_alphabet(pages)
    _mark_formula_only_fonts(pages)
    # No table search on contents pages: the page number column and the section number column align, short
    # entries would be taken as a cell table, and cells are not translated.
    # The same for index pages: several columns of short entries align at their starts, and a few rows of
    # them look like a table without rules
    toc_pages = _toc_pages(pages)
    index_pages = {pno for pno, data in enumerate(pages) if _index_page(data)}
    for pno in toc_pages | index_pages:
        tables_by_page[pno] = ()
        scans[pno] = replace(scans[pno], tables=())
    for data in pages:
        _rejoin_split_lines(data)
    line_gap = _line_gap(pages)
    body_size = _body_size(_sized_lines(pages))
    page_lines = [_frame_lines(data) for data in pages]
    frames = text_frames([lines for lines, _refs in page_lines])
    notes = margin_notes([lines for lines, _refs in page_lines], frames)
    pitches = note_pitch([lines for lines, _refs in page_lines], notes)
    for pno, (data, (lines, refs), found, frame) in enumerate(zip(pages, page_lines, notes, frames)):
        # An index page has several columns, and the leftmost lies outside the body text frame, looking like
        # margin notes. Index pages have no margin notes
        if pno in index_pages:
            continue
        _detach_margin_notes(data, lines, refs, found - _wide_block_lines(lines, refs, found, frame), pitches)
    absorbed: set[int] = set()
    for pno, data in enumerate(pages):
        margins = _left_margins(data)
        # Start and end of each line on this page, used to find the right edge of the column a line is in
        # Math lines on this page that keep the original text; fragments inside them (the denominator dx) are
        # math too
        kept_rows: list[pymupdf.Rect] = []
        page_start = len(blocks)
        row_ends = [
            _line_box(line)
            for block in data["blocks"]
            if block.get("type") == 0
            for line in block["lines"]
            if "".join(span["text"] for span in line["spans"]).strip()
        ]
        fractions = _fraction_parts(data, scans[pno].rules)
        toc_page = _has_leader_line(data) or pno in toc_pages
        index_page = _index_page(data)
        ref_rows = _index_ref_rows(data) if index_page else []
        # Page numbers on an index page form several columns, and words of entries happen to line up with them
        number_columns = [] if index_page else _number_columns(data)
        latin_page = _latin_page(data)
        for b in data["blocks"]:
            if b.get("type") != 0:
                continue
            items: list[tuple[str, pymupdf.Rect, list[dict]]] = []
            # Chinese source sentences split off (see _script_parts)
            cjk_heads: set[int] = set()
            for ln in b["lines"]:
                if abs(ln["dir"][1]) > 0.01:  # Rotated or vertical text is not handled
                    continue
                text = "".join(s["text"] for s in ln["spans"])
                if not text.strip():
                    continue
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                parts = _script_parts(ln, spans) if latin_page else []
                if parts:
                    cjk_heads.add(id(parts[0][2]))
                items.extend(parts or [(text, _line_box(ln), spans, ln["spans"])])
            items = _left_to_right(items)
            items, leaders = _cut_leaders(items, toc_page)
            items = _drop_number_tails(items)
            items, header_cells = _cut_header_cells(items, number_columns)
            side_note = bool(b.get("side_note"))
            # One margin note is one block and is not split by the body text rules
            groups = [list(range(len(items)))] if side_note and items else _split_lines(items, index_page)
            # A line with a leader is the last line of a contents entry, and the next line is another entry. The
            # same goes for a line ending with page numbers on an index page, an entry whose page numbers wrap
            # to the next line, and one cell of a word grid
            cells = _grid_cells(items)
            entry_ends = cells | header_cells | (_index_entry_ends(items, ref_rows) if index_page else set())
            # A Chinese source sentence is cut off on both sides and forms its own group
            entry_ends |= {j for i, item in enumerate(items) if id(item[2]) in cjk_heads for j in (i - 1, i) if j >= 0}
            groups = [part for group in groups for part in _cut_after(group, {*leaders, *entry_ends})]
            groups = [part for group in groups for part in _cut_at_posts(group, items, scans[pno].posts)]
            # A line holding a text fraction forms its own group, keeps the original text, and is not joined
            # with the body text above and below
            formula_rows = {i for i, item in enumerate(items) if any(id(span) in fractions for span in item[2])}
            groups = [part for group in groups for part in _cut_runs(group, formula_rows)]
            groups = [part for group in groups for part in _cut_line_list(group, items, row_ends)]
            groups = [part for group in groups for part in _cut_item_breaks(group, items, row_ends)]
            for group in groups:
                texts = [items[i][0] for i in group]
                line_rects = [items[i][1] for i in group]
                spans = [s for i in group for s in items[i][2]]
                span_lines = [items[i][3] for i in group]
                text = _join_lines(texts)
                rect = pymupdf.Rect()
                for r in line_rects:
                    rect |= r
                # Fragments inside a cell are kept for now: whether they are a formula is decided after the
                # pieces of the cell are merged
                in_cell = cell_containing(rect, tables_by_page[pno]) is not None
                # The few connecting words in a centered single-line expression (a definition, if and only if)
                # do not form a sentence, so the original text is kept:
                # the model often changes the mathematical meaning when it rearranges such lines
                displayed = len(group) == 1 and _displayed(rect, spans, margins)
                fraction = group[0] in formula_rows
                # The page numbers after an index entry do not count toward the share of letters:
                # perplexity, 27, 76, 88, 203 is one entry, not a row of numbers
                judged = _INDEX_REF_RE.sub("", text) if index_page else text
                # An index entry ending with page numbers is still an entry however few letters it has (F0, 340);
                # it stays a block with the original text and moves along when the index is rearranged
                entry = index_page and judged != text and bool(judged.strip()) and not fraction
                if (
                    not in_cell
                    and not entry
                    and (fraction or _untranslatable(spans, judged, _DISPLAY_PROSE if displayed else 3, displayed))
                    and not _wraps_from_above(blocks, pno, rect, text, row_ends, frames[pno], index_page)
                ):
                    # Only punctuation fragments are donated. Formula lines and lines with few letters are not
                    # sent for translation, and their rectangles must not be donated either:
                    # donated rectangles are erased, and these lines have no placeholder and are not redrawn.
                    if NO_TEXT_RE.match(text) and blocks and blocks[-1].page == pno:
                        _donate_punct_rects(blocks[-1], line_rects)
                    else:
                        kept_rows.append(rect)
                    continue
                # The size comes from the longest span. Color and weight come from the majority by character
                # count: in books with one span per word, the longest piece is often the caption label Figure
                # 10.3 in white on black, and the whole caption would be written in white
                main = _main_span([items[i][2] for i in group])
                main_size = _block_size(main, span_lines)
                color = _by_chars(spans, lambda s: s["color"])
                bold = _by_chars(spans, lambda s: bool(s["flags"] & 16) or "Bold" in s["font"])
                if in_cell:
                    pending_cells.add(len(blocks))
                blocks.append(
                    TextBlock(
                        page=pno,
                        rect=rect,
                        line_rects=line_rects,
                        text=text,
                        size=round(main_size, 1),
                        color=f"#{color:06x}",
                        bold=bold,
                        span_lines=span_lines,
                        side_note=side_note,
                        leader=leaders.get(group[-1]),
                        line_gap=line_gap,
                        body_size=body_size,
                        entry=group[-1] in entry_ends,
                        grid=group[-1] in cells,
                        index=index_page,
                    )
                )
        absorbed.update(id(block) for block in blocks[page_start:] if _inside_kept_formula(block, kept_rows))
    split: list[TextBlock] = []
    unchecked: set[int] = set()
    for index, block in enumerate(blocks):
        if id(block) in absorbed:
            continue
        tables = tables_by_page[block.page] if block.page < len(tables_by_page) else ()
        parts = _split_table_blocks([block], tables)
        if index in pending_cells:
            unchecked.update(id(part) for part in parts)
        split.extend(parts)
    split = _merge_cells(split, unchecked)
    # When the column gaps are clear, sort into reading order first. Continuation within a page looks only at
    # adjacent blocks, and another column in between would keep a block from joining the next line of its
    # own column.
    ordered: list[TextBlock] = []
    by_page: dict[int, list[TextBlock]] = {}
    geometries: dict[int, PageGeometry] = {}
    for block in split:
        by_page.setdefault(block.page, []).append(block)
    for pno in sorted(by_page):
        page_blocks = by_page[pno]
        geo = PageGeometry.from_page(doc[pno], [block.rect for block in page_blocks], scan=scans[pno])
        geometries[pno] = geo
        ordered.extend(reading_order(page_blocks, geo))
    lines_first = _merge_caption_fragments(_merge_visual_lines(ordered, geometries))
    # Running headers and footers are not joined to half sentences of body text: a book's running header is
    # set on the right without a period and looks like the second half of a line;
    # a paper's running header stops at et al. and looks like an unfinished sentence.
    # A running header often lies below the narrow strip at the top of the page and above the text frame, so it
    # is only recognized with the text frame
    margin_geometries = [
        PageGeometry.from_page(doc[pno], [block.rect for block in by_page.get(pno, [])], frames[pno], scans[pno])
        for pno in range(doc.page_count)
    ]
    heads = {id(block) for block, edge in zip(lines_first, margin_skips(lines_first, margin_geometries)) if edge}
    merged = _merge_continuations(lines_first, _line_lists(lines_first, frames), heads)
    edges = {id(block) for block, edge in zip(merged, margin_skips(merged, margin_geometries)) if edge}
    merged = _merge_same_line_sentences(_merge_line_tails(merged, edges), edges)
    return _split_affiliation_tails(merged), frames, scans


_DISPLAY_PROSE = 8


# Measuring the line spacing needs at least this many pairs of adjacent body lines
_LINE_GAP_PAIRS = 20


def _sized_lines(pages: list[dict]) -> list[list[tuple[pymupdf.Rect, float, int]]]:
    """The horizontal text lines of each page, with the size holding most characters and the character count,
    from top to bottom."""
    rows: list[list[tuple[pymupdf.Rect, float, int]]] = []
    for data in pages:
        row: list[tuple[pymupdf.Rect, float, int]] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [span for span in line["spans"] if span["text"].strip()]
                if abs(line["dir"][1]) > 0.01 or not spans:
                    continue
                size = round(float(_by_chars(spans, lambda span: span["size"])), 1)
                row.append((pymupdf.Rect(line["bbox"]), size, sum(len(span["text"].strip()) for span in spans)))
        rows.append(sorted(row, key=lambda item: (item[0].y0, item[0].x0)))
    return rows


def _body_size(rows: list[list[tuple[pymupdf.Rect, float, int]]]) -> float:
    """Body font size: the size with the most characters. 0 when there are no characters."""
    chars: dict[float, int] = {}
    for row in rows:
        for _box, size, count in row:
            chars[size] = chars.get(size, 0) + count
    return max(chars, key=chars.__getitem__, default=0.0)


def _line_gap(pages: list[dict]) -> float:
    """The gap between adjacent body lines across the document, in units of font size, taking the mode. Line
    box height varies with the font's ascender and descender: for two lines set equally tight, some fonts
    measure a gap of 0.2 em and others an overlap of 0.2 em.
    This amount is subtracted first when deciding whether the next line follows directly."""
    rows = _sized_lines(pages)
    body = _body_size(rows)
    if body <= 0:
        return _LINE_GAP
    counts: dict[float, int] = {}
    for row in rows:
        lines = [box for box, size, _count in row if abs(size - body) <= 0.3]
        for index, box in enumerate(lines):
            # The next line right below in the same column: overlapping horizontally by more than half, not on
            # the same row as this line
            for below in lines[index + 1 :]:
                if below.y0 > box.y1 + body * 3:
                    break
                if below.y0 < box.y1 - body * 0.3:
                    continue
                if min(box.x1, below.x1) - max(box.x0, below.x0) < min(box.width, below.width) * 0.5:
                    continue
                gap = round((below.y0 - box.y1) / body * 20) / 20
                counts[gap] = counts.get(gap, 0) + 1
                break
    if sum(counts.values()) < _LINE_GAP_PAIRS:
        return _LINE_GAP
    return max(counts, key=counts.__getitem__)


def _left_margins(data: dict) -> list[float]:
    """Left edges of body lines on this page (one per column): at least three long lines start there."""
    starts: list[float] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            if len("".join(span["text"] for span in line["spans"]).strip()) >= 40:
                starts.append(round(line["bbox"][0]))
    margins: list[float] = []
    for x in sorted(set(starts)):
        if sum(1 for start in starts if abs(start - x) <= 2) >= 3 and all(abs(x - m) > 2 for m in margins):
            margins.append(x)
    return margins


def _displayed(rect: pymupdf.Rect, spans: list[dict], margins: list[float]) -> bool:
    """An expression on a line of its own, more than two ems from the left edge of its column. An expression
    continuing the line above within a paragraph starts at the left edge."""
    left = [m for m in margins if m <= rect.x0 + 2]
    if not left:
        return False
    size = max((float(span.get("size") or 0) for span in spans), default=10.0)
    return rect.x0 - max(left) > size * 2.5


def _main_span(lines: list[list[dict]]) -> dict:
    """The piece that sets the block's size and color: the longest body-font piece on the baseline. In lines
    with much math the longest piece is often a subscript (the model of d_model, set small in the body font);
    a size taken from it is half the body size and does not join the line above."""
    kept: list[dict] = []
    for spans in lines:
        weights: Counter[int] = Counter()
        for span in spans:
            weights[round(float(span["origin"][1]))] += len(span["text"].strip())
        base = weights.most_common(1)[0][0] if weights else None
        kept += [
            span for span in spans if base is None or abs(float(span["origin"][1]) - base) <= float(span["size"]) * 0.2
        ]
    words = [span for span in kept if not _span_is_math(span) and span["text"].strip()]
    pool = words or kept or [span for spans in lines for span in spans]
    return max(pool, key=lambda span: len(span["text"]))


def _inside_kept_formula(block: TextBlock, kept: Sequence[pymupdf.Rect]) -> bool:
    """A one-line fragment with no word of three or more letters, mostly inside the box of a math line that
    keeps the original text: it is a piece of the expression (the denominator dx of a fraction) and is not
    sent for translation on its own. An annotation on a brace below an expression (prior) is a word and is
    translated."""
    if len(block.line_rects) != 1 or re.search(r"[A-Za-z]{3,}", block.text) or CJK_RE.search(block.text):
        return False
    area = block.rect.get_area()
    return area > 0 and any((block.rect & row).get_area() >= area * 0.6 for row in kept)
