"""Turn text blocks into translation units: inline formulas become placeholders, section
numbers are peeled off, and each unit gets a note on where it sits in the book."""

import functools
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

from .. import prompts
from .pdf_biblio import _median_size, biblio_skips
from .pdf_blocks import (
    _MARKUP_RE,
    _NUMBER_LABEL_RE,
    TextBlock,
    _color_value,
    _ends_sentence,
    _garbled_label,
    _join_lines,
    _same_page_line_pair,
    _small_caps_sizes,
)
from .pdf_extract import _extract
from .pdf_flow import (
    span_emphasis,
    strip_style_marks,
    writer_for,
)
from .pdf_formula_runs import (
    _ACCENT_CHARS,
    _LONE_PUNCT_RE,
    _SYMBOL_FONT_RE,
    _absorb_inner_blanks,
    _absorb_math_glue,
    _absorb_script_bases,
    _ends_row,
    _extend_text_run,
    _formula_markup,
    _fraction_pair,
    _has_text_base,
    _is_formula_span,
    _join_line_wrapped_formulas,
    _letter_formula_runs,
    _line_x0,
    _release_ordinals,
    _run_baseline,
    _run_text_box,
    _same_text_style,
    _smaller_prose_line,
)
from .pdf_glosses import example_numbers, interlinear_glosses
from .pdf_layout import PageGeometry, margin_skips
from .pdf_math import _glyph_box
from .pdf_merge import _cross_page_units, _region_joins, _seam_apart
from .pdf_page_continue import _all_styled
from .pdf_roles import LayoutItem, Role
from .pdf_rows import (
    _INDEX_REF_RE,
)
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
    _isolated_formula_ids,
    _item_example_ids,
    _role_pattern_ids,
)
from .pdf_types import PdfDict
from .pdf_unit_blocks import (
    _ACCENT_BEFORE_RE,
    _CEDILLA_RE,
    _SPACING_ACCENTS,
    _bridge_hyphen,
    _code_block,
    _colored_examples,
    _equation_block,
    _formula_record,
    _margin_term,
    _part_record,
    _retag_runs,
    _scan_status,
    _split_gloss_lines,
    _table_cell,
    _wrap_gap,
)
from .pdf_vision import FormulaModel, ModelBoxes, formula_detector, layout_detector

if TYPE_CHECKING:
    from ..runner import Runner


def _placeholderize(block: TextBlock, block_index: int = 0) -> tuple[str, list[PdfDict]]:
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
    span_lines: list[list[PdfDict]] = block.span_lines

    # 1) Collect formula runs per dict line and record their position in the block's span stream (for checking
    # adjacency in reading order)
    runs: list[PdfDict] = []
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

    def flow_adjacent(a: PdfDict, b: PdfDict) -> bool:
        l1, _, e1 = a["pos"]
        l2, s2, _ = b["pos"]
        for ln in range(l1, l2 + 1):
            line = span_lines[ln]
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

    def geometric(m: PdfDict, r: PdfDict) -> bool:
        if any(id(x) in based and x["max_size"] < block.size * 0.79 for x in (m, r)):
            return False
        x_overlap = min(m["bbox"].x1, r["bbox"].x1) - max(m["bbox"].x0, r["bbox"].x0)
        y_gap = max(m["bbox"].y0, r["bbox"].y0) - min(m["bbox"].y1, r["bbox"].y1)
        min_w = min(m["bbox"].width, r["bbox"].width)
        lo, hi = sorted([m["max_size"], r["max_size"]])
        return bool(x_overlap > max(1.0, 0.3 * min_w) and y_gap < block.size and lo < hi * 0.79)

    def adjacent(m: PdfDict, r: PdfDict) -> bool:
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
            and _ends_row(m, span_lines)
        )
        return (
            flow_adjacent(m, r)
            and r["bbox"].x0 - m["bbox"].x1 < block.size
            and not wrapped
            and y_overlap > 0.3 * min(m["bbox"].height, r["bbox"].height)
        )

    merged: list[PdfDict] = []
    # The run that most recently took in a piece. After a numerator is merged into the preceding denominator, the
    # numerator's subscript must still find it
    recent: PdfDict | None = None
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

            def covers(span: PdfDict) -> bool:
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

    formulas: list[PdfDict] = []
    formula_runs: dict[int, FormulaRun] = {}
    first_span: dict[int, int] = {}  # span id → formula number
    span_line = {id(s): li for li, line in enumerate(block.span_lines) for s in line}

    def text_baseline(spans: list[PdfDict]) -> float | None:
        """Body baseline of the lines that hold these spans: the median origin of body-size characters on those lines,
        not over the whole block"""
        lines = {span_line.get(id(span)) for span in spans}
        origins = sorted(
            span["origin"][1]
            for li in lines
            if li is not None
            for span in span_lines[li]
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

    def size_of(span: PdfDict) -> float:
        return caps.get(id(span), float(span.get("size") or 0))

    text_spans = [s for line in block.span_lines for s in line if id(s) not in formula_ids]
    sizes = {round(size_of(s), 1) for s in text_spans if str(s.get("text") or "").strip()}
    # The whole block in small caps: capitals and small letters have one size each but are one typeface, so no size
    # marks are written
    base_size = sizes.pop() if caps and len(sizes) == 1 else block.size
    writer = writer_for([{**s, "size": size_of(s)} for s in text_spans], base_size, _color_value(block.color))
    ordered: list[Run] = []
    out_lines: list[str] = []
    pending: TextRun | None = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            ordered.append(pending)
            pending = None

    for line_no, spans in enumerate(block.span_lines):
        parts: list[str] = []
        for s in spans:
            number = first_span.get(id(s))
            if number is not None:
                flush()
                ordered.append(formula_runs[number])
                parts.append(writer.atom(f"{{v{number}}}"))
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
    block.runs = ordered
    return _join_lines(out_lines), formulas


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


def _prepare_unit(unit: list[TextBlock], name_start: int) -> tuple[str, list[PdfDict], int]:
    """Text sent for translation and formula records of one translation unit. Formula names increase per block so
    that two blocks across a page break do not collide.

    When two blocks form one paragraph, {| } is inserted between them. Half sentences on the same page sharing a
    baseline are already one sentence and get no boundary.
    After translation the paragraph is set in full lines and continues on the next page once the page is full.
    A word hyphenated at the line end is completed first, so the model sees the whole word, and the boundary remains.
    """
    texts: list[str] = []
    formulas: list[PdfDict] = []
    name_i = name_start
    for ui, block in enumerate(unit):
        sent, found = _placeholderize(block, name_i)
        name_i += 1
        offset = len(formulas)
        if offset:
            sent = re.sub(
                r"\{v(\d+)\}",
                lambda m: f"{{v{int(m.group(1)) + offset}}}",
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
    toc: list[list[int | str]]  # get_toc entries: [level, title, page]
    units: list[list[TextBlock]]
    texts: list[str]
    formulas: list[list[PdfDict]]
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
    runner: "Runner",
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
    sent_formulas: list[list[PdfDict]] = []
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
        sent, unit_formulas, name_i = _prepare_unit(unit, name_i)
        flat_units.append(unit)
        sent_kinds.append(block_kind[id(unit[0])])
        if unit_formulas and _only_formulas(sent):
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
        sent_formulas.append(unit_formulas)
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
        parent = parents[n]
        if parent is not None:
            note += f", listed under the entry “{head(parent)}”"
        if children.get(n):
            note += "; its sub-entries include " + ", ".join(f"“{head(c)}”" for c in children[n][:_NOTE_CHILDREN])
        notes[n] = note + "."
    return notes
