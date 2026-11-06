"""Build the translation records of special blocks: equations, colored examples, code, glosses and table cells."""

import re
from dataclasses import replace

import pymupdf

from .pdf_blocks import (
    _CAPTION_NUM_RE,
    _PART_GAP,
    CJK_RE,
    TextBlock,
    _ends_sentence,
    _span_is_math,
    _visual_lines,
)
from .pdf_formula_runs import _formula_markup
from .pdf_frame import TextFrame
from .pdf_layout import PageGeometry
from .pdf_math import _plain_units
from .pdf_paragraphs import _block_slice, _block_spans, _clip_key
from .pdf_rows import _vivid
from .pdf_runs import (
    FormulaRun,
    Run,
)
from .pdf_skip import (
    _in_gloss,
)
from .pdf_split_lines import _RELATION_RE, _mono_span
from .pdf_types import PdfDict


def _part_record(run: PdfDict, page: int, name: str, baseline: float | None = None) -> PdfDict:
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


def _wrap_gap(before: list[PdfDict], after: list[PdfDict]) -> float:
    """Gap between two pieces of a formula split by a line break. A monospaced example sentence breaks between words
    (my dog is | hairy) and gets one word space, taken as the width of the spaces in the pieces. This applies only
    when the words in the pieces are already separated by spaces and both sides of the break are letters or digits;
    URLs and identifiers break inside a word and formulas break after an operator, and those pieces are set flush."""

    def text(spans: list[PdfDict]) -> str:
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


def _formula_record(run: FormulaRun, spans: list[PdfDict]) -> PdfDict:
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
    for page in doc.pages():
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
