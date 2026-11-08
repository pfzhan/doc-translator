"""Joining lines and blocks into paragraphs (pdf_paragraphs, pdf_merge)."""

from collections.abc import Iterable, Sequence

import pymupdf
import pytest

from app.formats.pdf_blocks import TextBlock
from app.formats.pdf_types import PdfDict

# A line of a test block: one (x0, x1) piece, or several pieces cut by inline formulas
_Row = tuple[float, float] | list[tuple[float, float]]


def _box(rect: pymupdf.Rect) -> tuple[float, float, float, float]:
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def test_a_proof_mark_after_the_full_stop_still_ends_the_paragraph() -> None:
    """The end-of-proof mark at the paragraph end comes out of the Symbol font as \x00, with the period before it. A
    paragraph starting with a capital on the next page is a new paragraph and is not joined."""
    from app.formats.pdf_blocks import TextBlock, _ends_sentence
    from app.formats.pdf_page_continue import _can_continue

    def blk(text: str, page: int) -> TextBlock:
        return TextBlock(
            page=page,
            rect=pymupdf.Rect(318, 700, 558, 729),
            line_rects=[],
            text=text,
            size=10,
            color="#000000",
            bold=False,
        )

    end = blk("Consequently, the query output can be computed from the view. \x00", 0)
    assert _ends_sentence(end.text) and _ends_sentence("which proves the claim. □")
    nxt = blk("As this example illustrated, to correctly test availability of output columns", 1)
    assert not _can_continue(end, nxt, 10, None, allow_capital=True)
    # The separators of formula sentinels are not stripped as end-of-proof marks
    assert not _ends_sentence("f\x01s\x02i\x01/s\x02(")


def test_a_column_break_inside_brackets_or_after_et_al_continues() -> None:
    """At a column or page change, a line starting with a parenthesis is judged by the text inside it, et al. inside
    a sentence is no sentence end, and a lowercase line after a semicolon or colon continues the text. Numbered
    items, phonetic transcriptions and citations start anew, and an all-italic subheading at the column bottom is
    not continued either."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_page_continue import _can_continue

    def blk(text: str, page: int) -> TextBlock:
        return TextBlock(
            page=page,
            rect=pymupdf.Rect(318, 700, 558, 729),
            line_rects=[],
            text=text,
            size=10,
            color="#000000",
            bold=False,
        )

    def joins(prev: str, nxt: str) -> bool:
        return _can_continue(blk(prev, 0), blk(nxt, 1), 10, None, allow_capital=True)

    assert joins("Accordingly, we use frame-wise root-mean-square", "(RMS) loudness to describe the trajectory.")
    assert joins("any of its variants such as Adagrad", "(Duchi et al., 2011). The normalization of activations")
    assert joins("work on 3D multimodal LLMs [Hong et al. 2023; Mao et al.", "2026; Xu et al. 2024] shows that LLMs")
    assert not joins("the scores are combined as shown below", "(3) where the weights sum to one")
    assert not joins("The Vocal Organs", "[v], [z], and all the English vowels, among others.")
    assert not joins("we use the fact that", "“[M]achines of this character can behave in a complicated manner.”")
    assert not joins("The scores are combined as shown below.", "(RMS) loudness to describe the trajectory.")
    assert joins("the policy is confused on BC trials;", "an explanation that also fails to cover the rest.")
    assert not joins("the policy is confused on BC trials;", "An explanation that also fails to cover the rest.")
    # The display formula after the colon is skipped, and where, starting lowercase on the next page, continues the
    # text
    assert joins("into a biaffine scoring function:", "where U, W, and b are weights learned by the model.")
    assert not joins("into a biaffine scoring function:", "We use U, W, and b as weights learned by the model.")
    assert not _can_continue(
        blk("the context window from that instance:", 0),
        blk("is traditionally followed by cherry pie, a traditional dessert", 1),
        10,
        None,
        allow_capital=True,
        at_edge=False,
    )
    assert joins("closed-loop reasoning in later work (Fu et al., 2024b;", "Mao et al., 2024) share the same confound.")
    assert not joins(
        "closed-loop reasoning in later work (Fu et al., 2024b;", "Results differ. Both (a) and (b) fail)."
    )
    assert joins("with a dual success metric and scripted certificates", "(§5.3). Urban probes stress the agent.")
    assert joins("Stress-tested on the held-out pool", "(Tables S32–S35), the port scores lower.")
    assert not joins("The scores are combined as shown below.", "(Tables S32–S35), the port scores lower.")
    formula = blk("D(·||·) between the data generating process and the model.", 1)
    formula.span_lines = [
        [
            {"text": "D", "font": "CMMI10", "flags": 2, "size": 10},
            {
                "text": "(·||·) between the data generating process and the model.",
                "font": "Times",
                "flags": 0,
                "size": 10,
            },
        ]
    ]
    assert _can_continue(blk("results from trying to minimise a divergence,", 0), formula, 10, None)
    assert not _can_continue(blk("results from trying to minimise a divergence.", 0), formula, 10, None)
    bullet = blk("• We introduce finite-transition modeling for accelerated MRI.", 1)
    bullet.span_lines = [
        [
            {"text": "•", "font": "CMSY7", "flags": 0, "size": 7},
            {
                "text": " We introduce finite-transition modeling for accelerated MRI.",
                "font": "Times",
                "flags": 0,
                "size": 10,
            },
        ]
    ]
    assert not _can_continue(blk("loss broadly across the image. At", 0), bullet, 10, None)
    assert joins(
        "similarity recovery comes with higher mean GOP (4.750 vs. 4.733)", "and lower mean WER on the test set."
    )
    assert not joins(
        "similarity recovery comes with higher mean GOP (4.750 vs. 4.733)", "On the item-matched test set, C wins."
    )
    heading = blk("E. Energy-Detection Characterization", 0)
    heading.line_rects = [pymupdf.Rect(49, 661, 204, 671)]
    heading.span_lines = [[{"text": heading.text, "font": "NimbusRomNo9L-ReguItal", "flags": 2, "size": 10}]]
    head = blk("This letter proposed an LLM-assisted hierarchical framework", 1)
    assert not _can_continue(heading, head, 10, None, allow_capital=True)
    heading.span_lines = [[{"text": heading.text, "font": "NimbusRomNo9L-Regu", "flags": 0, "size": 10}]]
    assert _can_continue(heading, head, 10, None, allow_capital=True)


def test_a_block_at_the_text_edge_starts_where_the_body_lines_start() -> None:
    """A half sentence continuing below a colon starts at the left edge of the body text; introduced examples and
    formulas are indented or centered. The left edge comes from all lines except the first of each paragraph in
    the same column, so first-line indents have no effect, and centered continuation lines of captions do not
    count."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_page_continue import _at_text_edge

    def blk(text: str, lefts: list[float], y: float, x1: float = 493.0) -> TextBlock:
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 10) for i, x0 in enumerate(lefts)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = [
        blk("Apple pie computer peripherals and personal digital assistants. These dimensions", [175, 160, 160], 300),
        blk("Figure 5.3 A spatial visualization of word vectors in two dimensions", [160, 210], 400),
    ]
    example = blk(
        "is traditionally followed by cherry pie, a traditional dessert often mixed", [191, 217, 161], 108, x1=457
    )
    resumed = blk("then the same kinds of training data would lead the language model to assign", [160, 160, 160], 108)
    assert not _at_text_edge(example, [example, resumed, *body])
    assert _at_text_edge(resumed, [resumed, *body])
    assert not _at_text_edge(blk("fair but brutal", [167], 108, x1=222), body)


def test_a_number_heavy_results_paragraph_is_prose() -> None:
    """A results paragraph full of numbers has a low share of letters but is still body text and can be the column top
    that continues a sentence. Short lines such as axis tick numbers are not."""
    from app.formats.pdf_columns import _is_code_line

    assert not _is_code_line(
        "and lower mean WER (0.009 vs. 0.013); paired differences are +0.017 [−0.043, 0.077] "
        "and −0.004 [−0.012, 0.004] (n = 1200/arm), 95% CI."
    )
    assert _is_code_line("0.1 0.2 0.3 0.4 0.5 0.6 (s)")


def test_a_body_sentence_that_starts_with_figure_is_not_a_caption() -> None:
    """A number followed by a lowercase word is body text (Figure 8(a) shows that …); a number followed by a colon,
    a period or a capital is a caption."""
    from app.formats.pdf_continuations import _is_float_text

    assert not _is_float_text("Figure 8(a) shows that the fidelity converges.")
    assert not _is_float_text("Table 5 evaluates the out-of-distribution results.")
    assert not _is_float_text("Table 1.2 shows the settings.")
    assert _is_float_text("Figure 3: The pipeline.")
    assert _is_float_text("FIG. 8. Temporal and spatial convergence.")
    assert _is_float_text("Table 1.2 Results on the test set")
    assert _is_float_text("Fig. 3 (A) Panel view")


def test_roman_and_appendix_numbers_are_captions() -> None:
    """IEEE table numbers are Roman numerals (TABLE IV: …) and appendix figure and table numbers carry a letter
    (Figure S1: …); followed by a lowercase word they are still body text."""
    from app.formats.pdf_continuations import _is_float_text

    assert _is_float_text("TABLE II: For different values of the number of observed e-folds")
    assert _is_float_text("TABLE I")
    assert _is_float_text("Figure S1: Full OSVD three-view label distributions.")
    assert _is_float_text("Table A2. Hyperparameters")
    assert _is_float_text("Figure B.6: Constrained distributional evolution")
    assert not _is_float_text("Table IV compares inference time.")
    assert not _is_float_text("Figure S1 shows that Satisfaction constitutes the largest share.")
    assert not _is_float_text("Table Index of symbols")


def test_table_cells_do_not_set_the_column_body_size() -> None:
    """A table at the column top has one block per cell at a size larger than the body; the body size still comes from
    the body blocks, so the body paragraph at the column bottom is not taken for a smaller footnote."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_columns import _column_body

    def blk(text: str, y: float, size: float, clip: tuple[float, float, float, float] | None = None) -> TextBlock:
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(54, y, 290, y + 20),
            line_rects=[],
            text=text,
            size=size,
            color="#000000",
            bold=False,
            clip=clip,
        )

    cells = [blk(f"AR + VMD {k}", 100 + 12 * k, 11.1, clip=(54, 100, 290, 200)) for k in range(7)]
    body = [blk("Overall, every proposed component helps.", 400 + 60 * k, 10) for k in range(4)]
    heading = blk("Out-of-Distribution Generalization", 650, 10.9)
    kept = _column_body([*cells, *body, heading])
    assert all(block in kept for block in body)


def test_a_table_with_its_caption_above_is_skipped_at_the_column_top() -> None:
    """When a table caption is set above the table, the short lines right below the caption are the table, and the
    continuing sentence is the first block below that fills the column width. A figure caption is set below the
    figure, so short lines below it (subject to after a formula) are body text."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_columns import _float_contents

    def blk(text: str, y: float, x1: float, rows: int = 1) -> TextBlock:
        lines = [pymupdf.Rect(318, y + 11 * i, x1, y + 11 * i + 9) for i in range(rows)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(318, y, x1, lines[-1].y1),
            line_rects=lines,
            text=text,
            size=9.0,
            color="#000",
            bold=False,
        )

    def page(caption_text: str) -> tuple[TextBlock, list[TextBlock], TextBlock]:
        caption = blk(caption_text, 359, 560, rows=2)
        cells = [blk("Method", 397, 350), blk("w/o point cloud", 412, 380), blk("w/o artist scenes", 434, 390)]
        body = blk("available support surfaces. Figure 7 corroborates this.", 493, 560, rows=4)
        return caption, cells, body

    caption, cells, body = page("Table 3. Controlled ablations on the test set.")
    assert _float_contents([*cells, body], [caption, *cells, body]) == {id(cell) for cell in cells}
    caption, cells, body = page("Fig. 3. Transient feature extraction.")
    assert _float_contents([*cells, body], [caption, *cells, body]) == set()


def test_a_column_does_not_continue_under_a_heading_at_its_top() -> None:
    """When a subheading of the right column sits right at its top, the right column starts a new section and the last
    sentence of the left column is not continued there; when other text lies between the subheading and the column
    top (large text in a figure recognized as a heading), it is continued as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_columns import _column_continuations

    def blk(text: str, x0: float, y: float, rows: int = 3) -> TextBlock:
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(x0, y, x0 + 245, y + 12 * rows),
            line_rects=[],
            text=text,
            size=10.0,
            color="#000000",
            bold=False,
        )

    left = [
        blk("The arm is described in the next section of this paper in full detail.", 54, 100),
        blk("Our solution is a novel tendon-actuated arm that we describe below.", 54, 200),
        blk("The arm is mechanically simpler than comparative actuators [17], [18],", 54, 600),
    ]
    heading = blk("A. Materials", 313, 89, rows=1)
    right = [
        blk("the soft structure including the chambers is cast in one piece.", 313, 105),
        blk("To facilitate free sliding of the tendon the ribbon is coated.", 313, 300),
    ]

    def pairs(extra: list[TextBlock]) -> dict[int, TextBlock]:
        blocks = [*left, heading, *extra, *right]
        return _column_continuations({0: blocks}, 10.0, {id(block) for block in extra}, {id(heading)}, set())

    assert id(left[-1]) not in pairs([])
    heading.rect = pymupdf.Rect(313, 60, 400, 72)
    label = blk("Reward Model", 313, 80, rows=1)
    label.rect = pymupdf.Rect(330, 80, 380, 90)
    assert pairs([label]).get(id(left[-1])) is right[0]
    # A subheading whose number hangs left of the column edge (1.4 Related work) is also a column-top subheading
    heading.rect = pymupdf.Rect(290, 89, 420, 101)
    assert id(left[-1]) not in pairs([])


def test_the_page_end_is_not_in_a_column_with_body_text_to_its_right(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the right column was not detected and no block was assigned to a column, the last column is the left one:
    body text still runs on the right, so the bottom of the left column is not the page end and does not continue at
    the top of the next page."""
    from app.formats import pdf_columns
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_columns import _column_continuations

    def blk(text: str, x0: float, y: float, page: int = 0) -> TextBlock:
        return TextBlock(
            page=page,
            rect=pymupdf.Rect(x0, y, x0 + 245, y + 36),
            line_rects=[],
            text=text,
            size=10.0,
            color="#000000",
            bold=False,
        )

    left = [
        blk("Text-to-SQL systems allow users to query structured databases in plain", 54, 100),
        blk("language, and most evaluation focuses on the final generated query and", 54, 600),
    ]
    right = [
        blk("Even when benchmark performance is relatively strong, generated queries fail.", 313, 100),
        blk("This framing is relevant to uncertainty-aware generation, and we treat the", 313, 600),
    ]
    head = blk("results under the conversational-history condition in the rest of the paper.", 54, 80, page=1)
    monkeypatch.setattr(
        pdf_columns,
        "_assigned_columns",
        lambda blocks, anchors: (
            [[block for block in blocks if block.rect.x0 < 300]],
            [block for block in blocks if block.rect.x0 >= 300],
        ),
    )
    pairs = _column_continuations({0: [*left, *right], 1: [head]}, 10.0, set(), set(), set())
    assert id(left[-1]) not in pairs


def test_a_flush_paragraph_under_a_table_continues_the_column_above() -> None:
    """In a document with first-line indents, a paragraph starting flush below a table at the column top continues
    the last paragraph of the left column; it is joined even when the left column breaks exactly at a sentence end
    and the continuation starts with a capital. A continuation with a first-line indent is a new paragraph; one
    right below a short body-size line (PROMPT, a missed subheading) starts a new paragraph; in a document without
    first-line indents it is not joined"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_columns import _column_continuations

    def blk(text: str, x0: float, y: float, rows: int = 3, size: float = 10.0, indent: float = 0.0) -> TextBlock:
        lines = [
            pymupdf.Rect(x0 + (indent if i == 0 else 0), y + 12 * i, x0 + 245, y + 12 * i + 10) for i in range(rows)
        ]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(x0, y, x0 + 245, y + 12 * rows),
            line_rects=lines,
            text=text,
            size=size,
            color="#000000",
            bold=False,
        )

    host = blk(
        "Ablation studies on fastMRI examine the principal design elements; "
        "the support-restricted objective also degrades performance.",
        54,
        600,
    )
    left = [blk("Table V compares inference time and peak GPU memory across methods here.", 54, 300), host]
    caption = blk("TABLE IV: Cross-dataset generalization. Models were trained on M4Raw.", 313, 100, rows=1)
    cells = [blk("PSNR", 361, 130, rows=1, size=7.3), blk("FLEX-MRI", 315, 160, rows=1, size=7.3)]
    head = blk("Perturbing the training inputs provides no additional benefit. These results", 313, 190)
    later = blk("Fig. 5 further examines the number of retained transitions in the model.", 313, 400, indent=10)

    def pairs(above: list[TextBlock], indents: bool = True) -> dict[int, TextBlock]:
        blocks = [*left, caption, *cells, *above, head, later]
        return _column_continuations({0: blocks}, 10.0, set(), set(), set(), indents=indents)

    assert pairs([]).get(id(host)) is head
    assert id(host) not in pairs([], indents=False)
    assert id(host) not in pairs([blk("PROMPT", 313, 175, rows=1)])
    head.line_rects[0].x0 += 10
    assert id(host) not in pairs([])


def test_formula_font_and_char_rules() -> None:
    from app.formats.pdf_blocks import _font_is_math
    from app.formats.pdf_paragraphs import _is_formula_char, _math_chars

    assert _font_is_math("CMMI10") and _font_is_math("STIXTwoMath")
    assert _font_is_math("LibertinusMath-Regular")
    # Math fonts that come with a text font, and the newtx math fonts
    assert _font_is_math("XCharterMathMI") and _font_is_math("txmiaX") and _font_is_math("ABCDEF+txsys")
    assert not _font_is_math("XCharter-Roman") and not _font_is_math("txtt")
    assert not _font_is_math("TimesNewRomanPSMT") and not _font_is_math("ArialMT")
    assert not _font_is_math("NotoSansCJK-Regular")
    # The blocklist comes first: Noto text fonts do not match the broad *Sans* heuristic
    assert _is_formula_char("α") and _is_formula_char("∑") and _is_formula_char("∂")
    assert not _is_formula_char("a") and not _is_formula_char("中")
    spans = [
        {"text": "E = mc", "font": "TimesNewRomanPSMT"},
        {"text": "αβγ", "font": "TimesNewRomanPSMT"},
        {"text": " + more text here", "font": "ArialMT"},
    ]
    assert _math_chars(spans) == 5  # Three for αβγ, plus = and + (category Sm math symbols)


def test_cross_page_units_merge_continuation() -> None:
    """Merging a paragraph carried over to the next page: when the last block on a page does not end with sentence-final
    punctuation and the first block on the next page starts lowercase, they form one translation unit."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, page: int, size: float = 10.0, y: float = 0) -> TextBlock:
        return TextBlock(
            page=page,
            rect=pymupdf.Rect(0, y, 100, y + 10),
            line_rects=[],
            text=text,
            size=size,
            color="#000",
            bold=False,
        )

    blocks = [
        blk("It can be seen that the model performs quite", 0, y=700),
        blk("1 Although one could also use another projection.", 0, size=9.0, y=740),
        blk("4https://example.com/footnote", 0, size=9.0, y=760),  # A smaller footnote does not count as the page end
        blk("well. We hypothesize that this has to do with it.", 1, y=60),
        blk("A new section starts here.", 1, y=90),
    ]
    units = _cross_page_units(blocks, 10.0)
    merged = [u for u in units if len(u) == 2]
    assert len(merged) == 1
    assert merged[0][0].text.endswith("quite") and merged[0][1].text.startswith("well")
    # Complete sentences are not merged
    blocks2 = [
        blk("This sentence is complete.", 0, y=700),
        blk("another paragraph starts here.", 1, y=60),
    ]
    assert all(len(u) == 1 for u in _cross_page_units(blocks2, 10.0))


def test_same_line_gap_stays_inside_one_column() -> None:
    """When a formula splits a line into two blocks, they are joined if the gap is narrower than the column gap. A block
    in the other column is not joined even 4pt away."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_paragraphs import _merge_visual_lines

    def blk(text: str, x0: float, x1: float, y: float = 100.0) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 11)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000", bold=False)

    one = PageGeometry(400, (Column(30, 380),), ())
    merged = _merge_visual_lines(
        [blk("the value", 40, 90), blk("x", 93, 102), blk("is large", 105, 160)],
        {0: one},
    )
    assert len(merged) == 1
    assert merged[0].text == "the value x is large"

    two = PageGeometry(400, (Column(30, 180), Column(186, 380)), (), reads_in_columns=True)
    split = _merge_visual_lines(
        [blk("left side", 40, 180), blk("continues", 184, 280)],
        {0: two},
    )
    assert len(split) == 2


def test_paragraph_split_by_a_caption_beside_it_is_joined() -> None:
    """A paragraph set around a figure has the caption between its two halves in reading order. With the caption
    below the first half and beside the second, the two halves are joined into one paragraph. A cell to the right
    in the same table row sits beside the previous cell, and a cell in the next row is not joined to it."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000", bold=False)

    head = blk("From the myth about Pygmalion and the statue he fell in love", 160, 241, 493, 265)
    caption = blk("Michelangelo's Moses (1515)", 380, 432, 486, 443)
    tail = blk("with, to Mary Shelley's Gothic novel about Frankenstein", 160, 265, 370, 337)
    merged = _merge_continuations([head, caption, tail])
    assert [b.text[:4] for b in merged] == ["From", "Mich"]
    assert merged[0].text.endswith("Frankenstein")

    left = blk("a claim made by the speaker", 160, 100, 300, 112)
    right = blk("a question by the speaker", 320, 100, 460, 112)
    below = blk("a weak suggestion or listing", 160, 114, 300, 126)
    assert len(_merge_continuations([left, right, below])) == 3


def test_a_larger_heading_below_a_title_is_not_its_wrapped_line() -> None:
    """A line one size larger right below a subsection name is the heading of the next section, not the second line of
    a wrapped subsection name."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, y0: float, x1: float, y1: float, size: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(
            page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=True, body_size=10
        )

    title = blk("8.4.4 Limitations of Preference-Based Learning", 160.2, 515.9, 410.3, 531.4, 12)
    section = blk("8.5 Summary", 95.4, 539.6, 201.1, 560.4, 17.2)
    assert len(_merge_continuations([title, section])) == 2


def test_a_running_head_does_not_run_into_the_first_line_below() -> None:
    """A running header ending on et al. looks like an unfinished sentence. A block recognized as a running header is
    not joined to the body text below that starts lowercase."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blocks() -> tuple[TextBlock, TextBlock]:
        head = TextBlock(
            page=1,
            rect=pymupdf.Rect(51, 42, 344, 54),
            line_rects=[pymupdf.Rect(51, 42, 344, 54)],
            text="2 Sarkar et al.",
            size=10.0,
            color="#000",
            bold=False,
        )
        lines = [pymupdf.Rect(51, 67 + 12.5 * i, 292, 77 + 12.5 * i) for i in range(3)]
        body = TextBlock(
            page=1,
            rect=pymupdf.Rect(51, 67, 292, 102),
            line_rects=lines,
            text="rounded by an outer, extended low surface brightness envelope. This came into light.",
            size=10.0,
            color="#000",
            bold=False,
        )
        return head, body

    head, body = blocks()
    assert len(_merge_continuations([head, body])) == 1
    head, body = blocks()
    assert len(_merge_continuations([head, body], margins={id(head)})) == 2


def test_a_word_cut_after_a_run_in_label_continues_at_the_column_edge() -> None:
    """The half line after an inline label starts right of the label, and the word broken at the line end (temper-)
    continues at the column edge on the next line."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blocks(x1: float = 522) -> list[TextBlock]:
        half = TextBlock(
            page=0,
            rect=pymupdf.Rect(163, 620, x1, 631),
            line_rects=[pymupdf.Rect(163, 620, x1, 631)],
            text="Models are served through OpenRouter (OpenRouter, 2026) at temper-",
            size=10.9,
            color="#000",
            bold=False,
        )
        lines = [pymupdf.Rect(90, 634 + 13 * i, 522, 645 + 13 * i) for i in range(3)]
        below = TextBlock(
            page=0,
            rect=pymupdf.Rect(90, 634, 522, 672),
            line_rects=lines,
            text="ature 0 with fixed seeds: all scenarios and every memory design.",
            size=10.9,
            color="#000",
            bold=False,
        )
        return [half, below]

    assert len(_merge_continuations(blocks())) == 1
    assert len(_merge_continuations(blocks(x1=400))) == 2


def test_a_line_inside_an_open_bracket_runs_on_below() -> None:
    """A line break inside an open parenthesis, with the next line starting with a capital or digit and closing the
    parenthesis in its first sentence, is still the same sentence."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, rows: int = 2) -> TextBlock:
        lines = [pymupdf.Rect(72, y + 24 * i, 540, y + 24 * i + 12) for i in range(rows)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(72, y, 540, y + 24 * rows - 12),
            line_rects=lines,
            text=text,
            size=12.0,
            color="#000",
            bold=False,
        )

    head = "the influence of the prior decreases with sample size (see e.g."
    assert (
        len(
            _merge_continuations([blk(head, 100), blk("Ghosal et al., 2000; Walker, 2004). The density follows.", 148)])
        )
        == 1
    )
    assert (
        len(_merge_continuations([blk(head, 100), blk("Ghosal et al. Then the second sentence starts here.", 148)]))
        == 2
    )


def test_a_line_after_an_inline_formula_wraps_to_the_column_edge() -> None:
    """A line with a formula on the left and body text continuing on the right: this half line reaches the right edge
    unfinished, and the next line continues lowercase at the column edge."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blocks(
        x1: float = 521, text: str = "This no longer agrees with the asymptotic distribution of the MLE under"
    ) -> list[TextBlock]:
        half = TextBlock(
            page=0,
            rect=pymupdf.Rect(154, 414, x1, 425),
            line_rects=[pymupdf.Rect(154, 414, x1, 425)],
            text=text,
            size=10.9,
            color="#000",
            bold=False,
        )
        lines = [pymupdf.Rect(89, 434 + 22 * i, 522, 445 + 22 * i) for i in range(3)]
        below = TextBlock(
            page=0,
            rect=pymupdf.Rect(89, 434, 522, 489),
            line_rects=lines,
            text="model misspecification, which although still centered at the same point.",
            size=10.9,
            color="#000",
            bold=False,
        )
        return [half, below]

    assert len(_merge_continuations(blocks())) == 1
    assert len(_merge_continuations(blocks(x1=400))) == 2
    assert len(_merge_continuations(blocks(text="This no longer agrees with the asymptotic distribution."))) == 2


def test_layout_model_regions_join_a_paragraph_cut_by_a_formula() -> None:
    """With the layout model enabled, two consecutive units inside the same body paragraph box of the model, with a
    formula at the cut, are one paragraph cut by the formula. No merge without the model, when the previous block
    ends a sentence and the next starts with a capital (a list), when the previous block is a line of author
    names, without a formula at the cut (bullet points on a slide), or for an epigraph and its source below."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_merge import _region_joins
    from app.formats.pdf_roles import HeuristicLayout

    def blk(text: str, y: float, rows: int = 3, x0: float = 72) -> TextBlock:
        lines = [pymupdf.Rect(x0, y + 12 * i, 540, y + 12 * i + 10) for i in range(rows)]
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(x0, y, 540, lines[-1].y1),
            line_rects=lines,
            text=text,
            size=10.0,
            color="#000",
            bold=False,
        )

    box = [pymupdf.Rect(70, 500, 545, 600)]

    def joined(head: TextBlock, tail: TextBlock, regions: list[pymupdf.Rect] = box) -> bool:
        formula = blk("Σ x = 1", 540, rows=1, x0=250)
        units = [[head], [formula], [tail]]
        out = _region_joins(units, {id(formula)}, set(), lambda pno: regions)
        return len(out) == 2 and out[0] == [head, tail]

    head = blk("We fix weights α with α > 0 and consider the measure on the line. For a Borel set E,", 505)
    tail = blk("1E is the indicator function of that set, and we let the map act on it.", 562, rows=2)
    assert joined(head, tail)
    assert not joined(head, tail, regions=[])
    near = blk("1E is the indicator function of that set, and we let the map act on it.", 541, rows=2)
    assert not joined(head, near)
    near.span_lines = [
        [
            {"text": "1E", "font": "CMMI10", "flags": 0, "size": 10},
            {"text": " is the indicator function of that set,", "font": "Times", "flags": 0, "size": 10},
        ],
        [],
    ]
    assert joined(head, near)
    ended = blk("We show that the dimension remains similar over time under fixed conditions.", 505)
    assert not joined(ended, blk("We demonstrate that aging changes the dimension of the data.", 552, rows=2))
    authors = blk("Jie Yu, Jianyu Zhang, Zian Pei, Xue Shi, Mohamed Ali", 505, rows=1, x0=190)
    assert not joined(authors, blk("Yumei Liu, Xin Jiang, Quanying Liu, Yi Guo", 552, rows=1, x0=190))
    quote = blk("“The most important attributes of a vector in 3-space are Location, Location, Location”", 505, rows=2)
    assert not joined(quote, blk("Randall Munroe, the hover from https://xkcd.com/2358/ on vectors", 552, rows=2))
    assert HeuristicLayout().text_regions(0) == []


def test_a_full_line_wraps_into_a_capitalized_next_line() -> None:
    """The indented first line of a paragraph, crowded into one block with a side note in the margin, is split out on
    its own. It reaches the right edge without sentence-final punctuation, and the next line returns to the left
    edge starting with a capitalized proper name (RNNs, one where…): still the same sentence. A last line that is
    not full, or that has a period, is not joined."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, x0: float = 160.2, x1: float = 493.2, rows: int = 1) -> TextBlock:
        lines = [pymupdf.Rect(x0 if i == 0 else 160.2, y + 12 * i, x1, y + 12 * i + 12) for i in range(rows)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    head = "A bidirectional RNN (Schuster and Paliwal, 1997) combines two independent"
    tail = "RNNs, one where the input is processed from the start to the end, and the other"
    assert len(_merge_continuations([blk(head, 610, x0=175.1), blk(tail, 622, rows=2)])) == 1
    assert len(_merge_continuations([blk(head, 610, x0=175.1, x1=420), blk(tail, 622, rows=2)])) == 2
    assert len(_merge_continuations([blk(head + ".", 610, x0=175.1), blk(tail, 622, rows=2)])) == 2


def test_an_indented_first_line_cut_by_a_margin_note_runs_on_after_a_full_stop() -> None:
    """In a document with first-line indents whose line boxes are taller than the line pitch, so the boxes of
    adjacent lines overlap a little (ed3book: 12pt pitch, 12.9pt line box), the first-line indent is still
    recognized. An indented first line cut into a block of its own by a side note, reaching the right edge and
    breaking exactly at a sentence end, is followed by a line back at the left edge that continues the same
    paragraph."""
    from app.formats.pdf_blocks import TextBlock, _indents_paragraphs
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, rows: Sequence[tuple[float, float]]) -> TextBlock:
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12.9) for i, (x0, x1) in enumerate(rows)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    paragraphs = [
        blk(body, 100 + 48 * i, [(175.1, 493.2), (160.2, 493.2), (160.2, 493.2), (160.2, 300)]) for i in range(4)
    ]
    assert _indents_paragraphs(paragraphs)
    head = blk("For example in batch training we compute the gradient over the entire dataset.", 292, [(175.1, 493.2)])
    tail = blk(
        "By seeing so many examples, batch training offers a superb estimate of the direction.",
        304,
        [(160.2, 493.2), (160.2, 493.2), (160.2, 426.5)],
    )
    merged = _merge_continuations([*paragraphs, head, tail])
    assert len(merged) == 5 and merged[-1].text.startswith("For example") and merged[-1].text.endswith("direction.")


def test_a_numbered_example_under_a_full_line_stays_on_its_own() -> None:
    """In a document with first-line indents, a numbered example sits flush below the line introducing it, with a
    little more space than the line pitch. It forms a block of its own and is not merged into the paragraph above,
    and the body text continuing flush below is not merged into it. The same across pages: when the page ends with
    an example, a paragraph starting flush on the next page is not joined after it."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_page_continue import _can_continue

    def blk(text: str, y: float, rows: Sequence[tuple[float, float]], page: int = 0) -> TextBlock:
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12.9) for i, (x0, x1) in enumerate(rows)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=page, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    paragraphs = [
        blk(body, 100 + 48 * i, [(157.1, 475.2), (142.2, 475.2), (142.2, 475.2), (142.2, 300)]) for i in range(4)
    ]
    lead = blk(
        "The relation between tenses and time is not simple. The present tense can be used to refer "
        "to a future event, as in this example:",
        292,
        [(157.1, 475.2), (142.2, 475.2)],
    )
    example = blk("(21.15) Ok, we fly from San Francisco to Boston at 10.", 320, [(142.2, 364.1)])
    after = blk(
        "Here the present tense refers to an event that will happen later, and the sentence is fine.",
        337,
        [(142.2, 475.2), (142.2, 475.2), (142.2, 300)],
    )
    merged = _merge_continuations([*paragraphs, lead, example, after])
    assert [block.text[:7] for block in merged[-3:]] == ["The rel", "(21.15)", "Here th"]
    # A formula reference in body text wrapped to the line start, continuing lowercase and sitting at normal line
    # pitch below the previous line: the same paragraph
    cited = blk(
        "For the odd generators we put a = 2j − 1 and b = 2j and then apply the two relations. By",
        292,
        [(157.1, 475.2), (142.2, 475.2)],
    )
    wrapped = blk(
        "(4.5) and (4.6), we have the identity that the paper needs in the next section.",
        316,
        [(142.2, 475.2), (142.2, 300)],
    )
    assert len(_merge_continuations([*paragraphs, cited, wrapped])) == 5

    ended = blk(
        "(24.31) IBM announced a new machine translation product yesterday. They have been working on it for 20 years.",
        660,
        [(142.2, 475.2), (174.1, 400)],
    )
    second = blk(
        "Second, singular they has become much more common, in which they is used to describe singular individuals.",
        110,
        [(142.2, 475.2), (142.2, 475.2), (142.2, 300)],
        page=1,
    )
    assert not _can_continue(ended, second, 10, None, flush=True)
    assert _can_continue(
        blk(
            "However, algorithms cannot enforce number agreement too strictly, as the example shows.",
            660,
            [(142.2, 475.2), (142.2, 400)],
        ),
        second,
        10,
        None,
        flush=True,
    )


def test_a_cross_page_unit_with_a_display_in_between_is_written_apart() -> None:
    """Two blocks across a page break separated by a display formula (dropped during text extraction, leaving only
    glyphs outside any block) or a skipped example are cut at the boundary when writing. When only footnotes and
    the page number lie below the previous block and only the running header and captions above the next, it is
    just a page change: fill the page, then continue."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_merge import PageWord, _seam_apart

    def blk(text: str, page: int, y0: float, y1: float, size: float = 10.0) -> TextBlock:
        rect = pymupdf.Rect(142, y0, 475, y1)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    geo = PageGeometry(612, (Column(142, 475),), (), 792, frame=TextFrame(142, 475, 109, 718, 10))
    host = blk("Let's assume the initial weights are all set to 0, and the learning rate is 0.1:", 0, 653, 679)
    tail = blk("The single update step requires that we compute the gradient, multiplied by the rate.", 1, 110, 134)
    header = blk("4.6 GRADIENT DESCENT", 1, 83, 95)
    note = blk("1 Note that for infinitesimals we use the notation below.", 0, 700, 709, size=8.0)
    page_number: PageWord = (300.0, 752.0, 306.0, 762.0, "4", 0, 0, 0)
    display: PageWord = (290.0, 688.0, 330.0, 699.0, "w1=w2=b=0", 0, 0, 0)

    def apart(
        page0: list[TextBlock],
        page1: list[TextBlock],
        words0: list[PageWord],
        shown: Iterable[int] = frozenset(),
        covered: Iterable[int] = frozenset(),
    ) -> bool:
        by_page = {0: [host, *page0], 1: [tail, *page1]}
        return _seam_apart(
            [host, tail],
            by_page,
            [geo, geo],
            {id(header), *covered},
            set(shown),
            10.0,
            lambda pno: words0 if pno == 0 else [],
        )

    assert not apart([note], [header], [page_number])
    assert apart([], [header], [display, page_number])
    example = blk("(24.31) IBM announced a new machine translation product yesterday.", 0, 695, 718)
    assert apart([example], [header], [], {id(example)})
    # The same when the layout model recognizes examples and formulas as figures (covered); only a figure not
    # recognized as an example does not count
    assert apart([example], [header], [], {id(example)}, {id(example)})
    assert not apart([example], [header], [], set(), {id(example)})


def test_a_document_that_spaces_paragraphs_joins_blocks_at_the_normal_line_gap() -> None:
    """In a document where new paragraphs start flush with extra space above, a next block at normal line pitch below
    a full line is the same paragraph, also when that line breaks exactly at a sentence end (double-spaced
    manuscripts often cut one paragraph into several blocks). A block with the extra space above starts a new
    paragraph."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    size, pitch = 10.9, 21.0

    def blk(text: str, y: float, rows: int = 3, last_x1: float = 522) -> TextBlock:
        lines = [
            pymupdf.Rect(89, y + pitch * i, 522 if i < rows - 1 else last_x1, y + pitch * i + 11) for i in range(rows)
        ]
        rect = pymupdf.Rect(89, y, 522, lines[-1].y1)
        return TextBlock(
            page=0, rect=rect, line_rects=lines, text=text, size=size, color="#000", bold=False, line_gap=10 / size
        )

    def page(gap: float) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        y = 60.0
        for _ in range(4):
            blocks.append(blk("This paragraph discusses the prior and the likelihood in turn.", y, last_x1=300))
            y = blocks[-1].rect.y1 + 16
        blocks.append(blk("These assumptions are almost never satisfied in practice.", y))
        blocks.append(
            blk("As a result, such Bayesian analyses are conducted with care.", blocks[-1].rect.y1 + gap, last_x1=300)
        )
        return blocks

    assert len(_merge_continuations(page(gap=10))) == 5
    assert len(_merge_continuations(page(gap=16))) == 6


def test_a_full_line_in_a_document_that_indents_paragraphs_runs_on_below() -> None:
    """In a document with first-line indents, after a full line breaking exactly at a sentence end, a line starting
    flush below continues the same paragraph. A numbered label, a line one line apart, a line with its own hanging
    indent below (also when the continuation forms a separate block), and a line below table cells each start a
    new paragraph."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, rows: Sequence[_Row]) -> TextBlock:
        # A row may have several pieces: [(x0, x1), ...]
        lines = [
            pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12)
            for i, row in enumerate(rows)
            for x0, x1 in (row if isinstance(row, list) else [row])
        ]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    head = "The first line opens indented and the second fills the measure to the margin."

    # Merging changes the previous block, so the blocks are rebuilt each time
    def plan(
        text: str,
        y: float = 316,
        rows: Sequence[_Row] = ((72, 400),),
        tail: Sequence[TextBlock] = (),
        last: Sequence[_Row] = ((72, 540),),
    ) -> list[TextBlock]:
        paragraphs = [blk(body, 100 + 48 * i, [(87, 540), (72, 540), (72, 540), (72, 300)]) for i in range(4)]
        return _merge_continuations([*paragraphs, blk(head, 292, [(87, 540), *last]), blk(text, y, list(rows)), *tail])

    assert len(plan("The next sentence goes on in the same paragraph.")) == 5
    assert len(plan("EXAMPLE 2.1 The next sentence goes on in the same paragraph.")) == 6
    assert len(plan("The next sentence goes on in the same paragraph.", y=328)) == 6
    entry = "The next entry runs to the margin and hangs its continuation"
    assert len(plan(entry + " below it", rows=((72, 540), (90, 300)))) == 6
    merged = plan(entry + ",", rows=((72, 540),), tail=[blk("under its first line.", 328, [(90, 300)])])
    assert [block.text[:14] for block in merged[-2:]] == ["The first line", "The next entry"]
    assert len(plan("The next sentence goes on in the same paragraph.", last=([(72, 200), (300, 540)],))) == 6
    # A single line ending a sentence, with an indented new paragraph right below: it is the last line of the
    # paragraph, not a hanging indent
    merged = plan("The paragraph ends on this line.", rows=((72, 300),), tail=[blk(body, 328, [(87, 540), (72, 300)])])
    assert len(merged) == 6 and merged[-2].text.endswith("on this line.")


def test_theorem_labels_do_not_count_as_flush_paragraphs() -> None:
    """Theorem and proof labels always start flush and are no evidence of unindented paragraphs. The width of a body
    line is measured over the whole line, joining the pieces cut by inline formulas."""
    from app.formats.pdf_blocks import TextBlock, _indents_paragraphs
    from app.formats.pdf_continuations import _body_measure

    def blk(text: str, y: float, rows: Sequence[_Row]) -> TextBlock:
        lines = [
            pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12)
            for i, row in enumerate(rows)
            for x0, x1 in (row if isinstance(row, list) else [row])
        ]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    labels = [
        "Lemma 3.4. Let U be a span.",
        "Proposition 2.4. Let T be bounded.",
        "2.1. Bosonic Fourier transform.",
        "Theorem 3.3. Suppose that T is linear.",
    ]
    blocks: list[TextBlock] = []
    for i in range(3):
        blocks.append(blk(body, 100 + 96 * i, [(87, 540), (72, 300)]))
        blocks.append(blk(body, 124 + 96 * i, [(87, 540), (72, 300)]))
        blocks.append(blk(labels[i], 148 + 96 * i, [(72, 300)]))
    blocks.append(blk(labels[3], 388, [(72, 300)]))
    assert _indents_paragraphs(blocks)
    split = blk(body, 500, [[(72, 200), (210, 540)], [(72, 300), (310, 540)], [(72, 540)], (72, 300)])
    assert _body_measure([split]) == 468.0


def test_cjk_wrap_merges_and_short_labels_stay() -> None:
    """Chinese lines with aligned left edges and a long enough previous line are joined back into one paragraph. Short
    labels and indented new paragraphs are not joined."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, y: float, x0: float = 40.0, x1: float = 280.0, page: int = 0, size: float = 10.0) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    wrapped = _merge_continuations(
        [
            blk("这是一行已经写到栏边还没有句号的正文", 100),
            blk("下一行从左缘接着写。", 114, x1=180),
        ]
    )
    assert len(wrapped) == 1
    assert wrapped[0].text.endswith("接着写。")

    labels = _merge_continuations(
        [
            blk("姓名", 100, x1=70),
            blk("年龄", 114, x1=70),
        ]
    )
    assert len(labels) == 2

    indented = _merge_continuations(
        [
            blk("这是一行已经写到栏边还没有句号的正文", 100),
            blk("新的一段从缩进开始写起。", 114, x0=62, x1=220),
        ]
    )
    assert len(indented) == 2

    pages = _cross_page_units(
        [
            blk("上一页在这里被切断还没写完", 700, page=0),
            blk("下一页从同一边距继续。", 60, page=1),
        ],
        10.0,
    )
    assert any(len(unit) == 2 for unit in pages)
    heading = _cross_page_units(
        [
            blk("上一页在这里被切断还没写完", 700, page=0),
            blk("1 引言", 60, page=1),
        ],
        10.0,
    )
    assert all(len(unit) == 1 for unit in heading)


def test_caption_label_without_colon_merges() -> None:
    """The note after "图 2" or "Fig. 1." is joined. Headings and table rows are not."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_caption_fragments

    def blk(text: str, x0: float, y0: float, x1: float, size: float = 10.0) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y0 + 10)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    zh = _merge_caption_fragments(
        [
            blk("图 2", 60, 100, 88),
            blk("衰减曲线示意。", 60, 112, 160),
        ]
    )
    assert len(zh) == 1
    assert zh[0].text == "图 2衰减曲线示意。"

    fig = _merge_caption_fragments(
        [
            blk("Fig. 1.", 60, 100, 98),
            blk("Examples of decay", 104, 100, 220),
        ]
    )
    assert len(fig) == 1

    title = _merge_caption_fragments(
        [
            blk("Introduction", 40, 100, 130, size=12.0),
            blk("Scholarly works.", 40, 130, 150, size=10.0),
        ]
    )
    assert len(title) == 2


def test_caption_fragments_merge_into_one_block() -> None:
    """The fragments 'Figure 2:' + 'Examples of decay' + 'schedules.' are joined back into one block: translated
    separately and put back into small boxes they wrap badly and the translation is broken up. Headings and blocks
    without a colon are not affected."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_caption_fragments

    def blk(text: str, x0: float, y0: float, x1: float, size: float = 10.0) -> TextBlock:
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(x0, y0, x1, y0 + 10),
            line_rects=[pymupdf.Rect(x0, y0, x1, y0 + 10)],
            text=text,
            size=size,
            color="#000",
            bold=False,
        )

    blocks = [
        blk("Figure 2:", 60, 100, 105),
        blk("Examples of decay", 112, 100, 220),
        blk("schedules.", 60, 112, 110),
    ]
    merged = _merge_caption_fragments(blocks)
    assert len(merged) == 1
    assert merged[0].text == "Figure 2: Examples of decay schedules."
    assert len(merged[0].line_rects) == 3

    # Heading plus body text (no colon, starting with a capital) is not merged
    b2 = [blk("Introduction", 40, 100, 130, size=12.0), blk("Scholarly works.", 40, 130, 130, size=10.0)]
    assert len(_merge_caption_fragments(b2)) == 2

    # Table rows (starting with a capital, no colon) are not merged
    b3 = [
        blk("Always Sampling", 60, 100, 160),
        blk("Scheduled Sampling 1", 60, 112, 190),
        blk("Baseline LSTM", 60, 124, 150),
    ]
    assert len(_merge_caption_fragments(b3)) == 3


def test_heading_beside_the_first_line_is_not_absorbed() -> None:
    """A short heading sits beside only the first body line. Using the union of the blocks would swallow it into the
    body text."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_visual_lines

    memo_line = pymupdf.Rect(40, 100, 72, 111)
    memo = TextBlock(page=0, rect=memo_line, line_rects=[memo_line], text="Memo.", size=9, color="#000", bold=False)
    first = pymupdf.Rect(80, 100, 220, 111)
    later = pymupdf.Rect(40, 114, 220, 125)
    para = TextBlock(
        page=0,
        rect=first | later,
        line_rects=[first, later],
        text="The space of plan alternatives is large.",
        size=9,
        color="#000",
        bold=False,
    )
    assert len(_merge_visual_lines([memo, para])) == 2


def test_right_fragment_joins_the_following_line() -> None:
    """A half sentence on the right of a line joins the next line, not the finished sentence on the left."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_line_tails

    left_lines = [pymupdf.Rect(40, 100, 180, 112), pymupdf.Rect(40, 112, 90, 124)]
    left = TextBlock(
        page=0,
        rect=left_lines[0] | left_lines[1],
        line_rects=left_lines,
        text="The paragraph already ended.",
        size=9,
        color="#000",
        bold=False,
    )
    frag_line = pymupdf.Rect(100, 112, 180, 124)
    fragment = TextBlock(
        page=0,
        rect=frag_line,
        line_rects=[frag_line],
        text="It is distinguished from",
        size=9,
        color="#000",
        bold=False,
    )
    next_line = pymupdf.Rect(40, 126, 170, 138)
    nxt = TextBlock(
        page=0,
        rect=next_line,
        line_rects=[next_line],
        text="other optimizers in several ways:",
        size=9,
        color="#000",
        bold=False,
    )
    merged = _merge_line_tails([left, fragment, nxt])
    assert len(merged) == 2
    assert merged[0].text == "The paragraph already ended."
    assert merged[1].text.startswith("It is distinguished from")
    assert merged[1].text.endswith("several ways:")

    shared_lines = [pymupdf.Rect(40, 126, 200, 138), pymupdf.Rect(40, 140, 200, 152)]
    shared = TextBlock(
        page=0,
        rect=shared_lines[0] | shared_lines[1],
        line_rects=shared_lines,
        text="For example the next sentence starts here.",
        size=9,
        color="#000",
        bold=False,
    )
    short = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 126, 110, 138),
        line_rects=[pymupdf.Rect(40, 126, 110, 138)],
        text="expression-specific.",
        size=9,
        color="#000",
        bold=False,
    )
    blocked = _merge_line_tails([left, fragment, shared, short])
    assert all(block.text != "expression-specific." or "distinguished" not in block.text for block in blocked)
    assert any(block.text == "expression-specific." for block in blocked)


def test_same_line_sentence_joins_the_paragraph_and_keeps_side_labels() -> None:
    """A next sentence starting at the line end joins the previous paragraph. Short bold side labels and labels inside
    figures do not. A continuation line is joined first even when a following sentence is in the way."""
    from app.formats.pdf_blocks import TextBlock, _with_source_bold
    from app.formats.pdf_paragraphs import _merge_same_line_sentences

    def line(text: str, x0: float, y0: float, x1: float, y1: float, bold: bool = False) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=9, color="#000", bold=bold)

    def para(text: str, boxes: list[tuple[float, float, float, float]], bold: bool = False) -> TextBlock:
        rects = [pymupdf.Rect(*box) for box in boxes]
        union = rects[0]
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=0, rect=union, line_rects=rects, text=text, size=9, color="#000", bold=bold)

    intro = para(
        "In this paper we describe Orca for demanding analytics workloads.",
        [(316.8, 307.0, 555.9, 316.0), (316.8, 338.4, 448.3, 347.3)],
    )
    distinguished = para(
        "It is distinguished from other optimizers in several important ways:",
        [(457.3, 338.4, 555.9, 347.3), (316.8, 348.8, 491.7, 357.8)],
    )
    extensibility = para(
        "Extensibility. Certain optimizations are dealt with as an afterthought.",
        [(316.8, 441.9, 555.9, 450.9), (339.2, 483.8, 384.5, 492.7)],
    )
    multiphase = para(
        "Multi-phase optimizers are notoriously difficult to extend.",
        [(394.4, 483.8, 556.0, 492.7), (339.2, 494.2, 555.9, 503.2)],
    )
    ended = para("In order to derive statistics.", [(316.8, 386.0, 556.0, 396.0), (316.8, 406.9, 406.2, 415.9)])
    promise = line("Statistics promise computation is", 416.0, 406.9, 556.0, 415.9)
    specific = line("expression-specific.", 316.8, 417.4, 392.7, 426.4)
    example = para(
        "For example the next sentence starts here.", [(401.6, 417.4, 555.9, 426.4), (316.8, 428.0, 556.0, 438.0)]
    )
    label = line("Property Enforcement.", 316.8, 350.0, 426.7, 359.0, bold=True)
    body = para(
        "Orca includes an extensible framework for describing query requirements.",
        [(438.2, 350.0, 555.9, 359.0), (316.8, 360.5, 555.9, 369.5)],
    )
    diagram = line("Get(T1)%", 360.0, 200.0, 381.0, 209.0)
    diagram2 = line("Get(T2)%", 394.6, 200.0, 416.0, 209.0)
    merged = _merge_same_line_sentences(
        [
            intro,
            distinguished,
            extensibility,
            multiphase,
            label,
            body,
            ended,
            promise,
            example,
            specific,
            diagram,
            diagram2,
        ]
    )
    texts = [block.text for block in merged]
    assert any(text.startswith("In this paper") and "distinguished" in text for text in texts)
    assert any(text.startswith("Extensibility.") and "Multi-phase" in text for text in texts)
    joined = next(text for text in texts if "Statistics promise" in text)
    assert "expression-specific." in joined
    assert "For example" in joined
    assert "In order to derive" in joined
    assert any(text == "Property Enforcement." for text in texts)
    assert any(text.startswith("Orca includes") for text in texts)
    assert any(text == "Get(T1)%" for text in texts)
    assert any(text == "Get(T2)%" for text in texts)

    lead = TextBlock(
        page=0,
        rect=pymupdf.Rect(40, 40, 200, 52),
        line_rects=[pymupdf.Rect(40, 40, 200, 52)],
        text="Modularity. Using a highly extensible abstraction.",
        size=9,
        color="#000",
        bold=False,
        span_lines=[
            [
                {"text": "Modularity.", "font": "CMBX9", "flags": 16},
                {"text": " Using a highly extensible abstraction.", "font": "CMR9", "flags": 4},
            ]
        ],
    )
    assert _with_source_bold(lead, "模块化。借助高度可扩展的抽象。") == "{b}模块化。{/b}借助高度可扩展的抽象。"
    assert _with_source_bold(lead, "{b}模块化。{/b}借助。") == "{b}模块化。{/b}借助。"
    memo = line("Memo.", 40, 100, 72, 112, bold=True)
    memo.span_lines = [[{"text": "Memo.", "font": "CMBX9", "flags": 16}]]
    assert _with_source_bold(memo, "备忘录。") == "{b}备忘录。{/b}"
    # A bold name followed by an asterisk script in a formula font: the whole line is still bold
    author = line("Jimmy Lei Ba∗", 40, 120, 100, 132, bold=True)
    author.span_lines = [
        [
            {"text": "Jimmy Lei Ba", "font": "NimbusRomNo9L-Medi", "flags": 20},
            {"text": "∗", "font": "CMSY7", "flags": 7},
        ]
    ]
    assert _with_source_bold(author, "Jimmy Lei Ba{v1}") == "{b}Jimmy Lei Ba{v1}{/b}"


def _reference_same_line_merge(blocks: list[TextBlock]) -> list[TextBlock]:
    """The implementation before merging by page. Each merge restarts the search from the beginning."""
    from app.formats.pdf_paragraphs import _absorb_block, _continues_fragment, _sits_on_last_line

    out = list(blocks)
    changed = True
    while changed:
        changed = False
        for index, host in enumerate(out):
            for other_index, guest in enumerate(out):
                if other_index == index or guest.page != host.page:
                    continue
                if not _sits_on_last_line(host, guest) and not _continues_fragment(host, guest):
                    continue
                _absorb_block(host, guest)
                del out[other_index]
                changed = True
                break
            if changed:
                break
    return out


def _random_line_blocks(seed: int) -> list[TextBlock]:
    import random

    rng = random.Random(seed)
    texts = [
        "Alpha beta gamma delta.",
        "alpha beta gamma delta",
        "and the model keeps going",
        "Results are shown here.",
        "with more words after it",
    ]
    blocks: list[TextBlock] = []
    for _ in range(rng.randint(4, 30)):
        page = rng.choice([0, 1, 2])
        row = rng.randint(0, 8)
        start = rng.choice([40.0, 40.0, 130.0, 210.0])
        count = rng.choice([1, 1, 2, 3])
        rects: list[pymupdf.Rect] = []
        for line in range(count):
            y0 = 100 + (row + line) * 10.5
            x0 = start if line == 0 else 40.0
            x1 = 300.0 if line < count - 1 else rng.choice([120.0, 200.0, 300.0])
            rects.append(pymupdf.Rect(x0, y0, max(x1, x0 + 20), y0 + 9))
        union = pymupdf.Rect(rects[0])
        for rect in rects[1:]:
            union |= rect
        blocks.append(
            TextBlock(page=page, rect=union, line_rects=rects, text=rng.choice(texts), size=9, color="#000", bold=False)
        )
    return blocks


def test_same_line_merge_by_page_matches_the_rescan() -> None:
    """Splitting by page and continuing the search after each merge gives the same result as rescanning from the start
    every time."""
    import copy

    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_same_line_sentences

    def shape(
        blocks: list[TextBlock],
    ) -> list[tuple[int, str, tuple[float, float, float, float], list[tuple[float, float, float, float]]]]:
        return [(b.page, b.text, _box(b.rect), [_box(r) for r in b.line_rects]) for b in blocks]

    merges = 0
    for seed in range(300):
        blocks = _random_line_blocks(seed)
        expected = _reference_same_line_merge(copy.deepcopy(blocks))
        assert shape(_merge_same_line_sentences(copy.deepcopy(blocks))) == shape(expected), seed
        merges += len(blocks) - len(expected)
    assert merges > 300

    # The middle block must first join the next line and return to the column edge before it can merge into the
    # paragraph before it. After merging, the earlier blocks are checked again.
    def para(text: str, boxes: list[tuple[float, float, float, float]]) -> TextBlock:
        rects = [pymupdf.Rect(*box) for box in boxes]
        union = pymupdf.Rect(rects[0])
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=0, rect=union, line_rects=rects, text=text, size=9, color="#000", bold=False)

    chain = [
        para("Alpha beta gamma delta.", [(40, 100, 300, 109), (40, 110.5, 150, 119.5)]),
        para("Results are shown here and", [(155, 110.5, 300, 119.5), (200, 121, 300, 130)]),
        para("with more words after it.", [(40, 131.5, 300, 140.5)]),
    ]
    expected = _reference_same_line_merge(copy.deepcopy(chain))
    assert len(expected) == 1
    assert shape(_merge_same_line_sentences(copy.deepcopy(chain))) == shape(expected)


def test_a_displayed_formula_does_not_swallow_the_prose_below_it() -> None:
    """A displayed formula has space on both sides; numerator, denominator and semicolon are separate pieces, and the
    last piece lands on the right. Body text below at the column edge starts a new paragraph."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _continues_fragment

    lines = [pymupdf.Rect(117, 145, 173, 158), pymupdf.Rect(178, 145, 253, 165), pymupdf.Rect(254, 157, 257, 164)]
    host = TextBlock(
        page=0,
        rect=lines[0] | lines[1] | lines[2],
        line_rects=lines,
        text="dNorm(x, X)/dx and dNorm(x, X)/dX ;",
        size=10,
        color="#000",
        bold=False,
    )
    guest_lines = [pymupdf.Rect(72, 175, 301, 188), pymupdf.Rect(72, 187, 301, 200)]
    guest = TextBlock(
        page=0,
        rect=guest_lines[0] | guest_lines[1],
        line_rects=guest_lines,
        text="ignoring the latter term would lead to the explosion described above.",
        size=10,
        color="#000",
        bold=False,
    )
    assert not _continues_fragment(host, guest)


def test_finished_condition_line_does_not_swallow_the_next_paragraph() -> None:
    """When a formula condition line ends with a period, the body text below at the column edge is not joined to it.
    An unfinished right half sentence still joins the next line."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _continues_fragment, _merge_same_line_sentences

    formula = pymupdf.Rect(254.4, 329.7, 333.6, 340.5)
    condition = pymupdf.Rect(343.5, 329.9, 384.8, 339.9)
    host = TextBlock(
        page=0,
        rect=formula | condition,
        line_rects=[formula, condition],
        text="f(ht-1, yt-1, xt; θ) otherwise.",
        size=10,
        color="#000",
        bold=False,
    )
    guest_lines = [pymupdf.Rect(108.0, 349.4, 504.0, 361.1), pymupdf.Rect(108.0, 360.6, 305.0, 370.5)]
    guest = TextBlock(
        page=0,
        rect=guest_lines[0] | guest_lines[1],
        line_rects=guest_lines,
        text="where oh is a vector of 0's with same dimensionality as ht's.",
        size=10,
        color="#000",
        bold=False,
    )
    assert not _continues_fragment(host, guest)
    merged = _merge_same_line_sentences([host, guest])
    assert [block.text for block in merged] == [host.text, guest.text]
    assert merged[0].rect.y1 < guest.rect.y0

    left = pymupdf.Rect(316.8, 406.9, 406.2, 415.9)
    right = pymupdf.Rect(416.0, 406.9, 556.0, 415.9)
    open_host = TextBlock(
        page=0,
        rect=left | right,
        line_rects=[left, right],
        text="derive statistics. Statistics promise computation is",
        size=9,
        color="#000",
        bold=False,
    )
    next_line = pymupdf.Rect(316.8, 417.4, 392.7, 426.4)
    nxt = TextBlock(
        page=0,
        rect=next_line,
        line_rects=[next_line],
        text="expression-specific.",
        size=9,
        color="#000",
        bold=False,
    )
    assert _continues_fragment(open_host, nxt)
    joined = _merge_same_line_sentences([open_host, nxt])
    assert len(joined) == 1
    assert "Statistics promise" in joined[0].text
    assert joined[0].text.endswith("expression-specific.")


def test_shared_baseline_continuation_is_one_translation() -> None:
    """When the next line shares a baseline with the following paragraph, they are not merged into one outer box but
    are still sent for translation as a pair, without a boundary between them and without joining the finished
    sentence on the left."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_merge import _cross_page_units
    from app.formats.pdf_paragraphs import _merge_line_tails
    from app.formats.pdf_units import _prepare_unit

    def line(text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=9, color="#000", bold=False)

    def para(text: str, lines: list[tuple[float, float, float, float]]) -> TextBlock:
        rects = [pymupdf.Rect(*box) for box in lines]
        union = rects[0]
        for rect in rects[1:]:
            union |= rect
        return TextBlock(page=0, rect=union, line_rects=rects, text=text, size=9, color="#000", bold=False)

    ended = para("In order to derive statistics.", [(316.8, 386.0, 556.0, 396.0), (316.8, 406.9, 406.2, 415.9)])
    promise = line("Statistics promise computation is", 416.0, 406.9, 556.0, 415.9)
    specific = line("expression-specific.", 316.8, 417.4, 392.7, 426.4)
    example = para(
        "For example the next sentence starts here.", [(401.6, 417.4, 555.9, 426.4), (316.8, 428.0, 556.0, 438.0)]
    )
    figure = para("Figure 5 ends.", [(316.8, 616.0, 556.0, 626.0), (316.8, 626.6, 346.0, 635.6)])
    requests = line("For example, InnerJoin(T1,T2) on (a=b) requests", 353.5, 626.6, 555.9, 635.6)
    histograms = line("histograms on T1.a and T2.b.", 316.8, 637.1, 441.3, 646.0)
    requested = para(
        "The requested histograms are used later.", [(449.4, 637.1, 555.9, 646.0), (316.8, 648.0, 556.0, 658.0)]
    )
    heading = line("(2) Statistics Derivation.", 316.8, 200.0, 435.3, 212.0)
    beside = para(
        "At the end of exploration the optimizer stops.", [(444.4, 200.0, 556.0, 212.0), (316.8, 214.0, 556.0, 226.0)]
    )
    # The other column comes between the half sentence and its continuation line and already reaches further down.
    # The continuation line must not be lost because of that.
    other_column = line("ORDER BY T1.a where the distribution is hashed.", 53.8, 434.7, 280.0, 446.0)
    blocks = [heading, beside, ended, promise, example, other_column, specific, figure, requests, requested, histograms]
    merged = _merge_line_tails(blocks)
    assert any(block.text == "expression-specific." for block in merged)
    assert any(block.text == "histograms on T1.a and T2.b." for block in merged)
    units = _cross_page_units(blocks, 9.0)
    pairs = [unit for unit in units if len(unit) == 2]
    assert pairs == [[promise, specific], [requests, histograms]]
    sent, _, _ = _prepare_unit(pairs[0], 0)
    assert "{|}" not in sent
    assert "expression-specific." in sent
    assert "Statistics promise computation is" in sent


def test_column_wrap_joins_when_geometry_misses_the_gutter() -> None:
    """When the geometric columns are inaccurate, the end of the left column is still joined to the top of the right
    column by the left edge of the body text. A caption does not take this position."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, x0: float, y: float, page: int = 0, x1: float | None = None) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1 if x1 is not None else x0 + 180, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=9, color="#000", bold=False)

    wide = PageGeometry(600, (Column(40, 560),), ())
    filler = blk("Earlier sentence in the left column ends.", 40, 660)
    host = blk("Effective usage of CPUs trans-", 40, 700)
    caption = blk("Figure 8: Optimization jobs dependency graph", 300, 40, x1=520)
    tail = blk("lates to better query plans.", 300, 80)
    later = blk("Parallelizing the optimizer is crucial.", 300, 100)
    units = _cross_page_units([filler, host, caption, tail, later], 9.0, geometries=[wide])
    assert [unit for unit in units if len(unit) == 2] == [[host, tail]]
    # Without geometry, two columns on the same page are still kept apart by the original rules
    assert all(
        len(unit) == 1
        for unit in _cross_page_units(
            [filler, host, caption, tail, later],
            9.0,
        )
    )

    listing = blk('<dxl:Metadata SystemIds="0">', 40, 40, page=1)
    code = blk("1 0x000e8106df gpos::CException::Raise", 40, 56, page=1)
    prose = blk("tomer issues in the optimizer.", 40, 80, page=1)
    prose2 = blk("The dump can be replayed later.", 40, 96, page=1)
    cus = blk("reproduce and debug cus-", 300, 700)
    cus2 = blk("The tool captures the optimizer state.", 300, 660)
    joined = _cross_page_units(
        [cus2, cus, listing, code, prose, prose2],
        9.0,
        geometries=[wide, wide],
    )
    assert [unit for unit in joined if len(unit) == 2] == [[cus, prose]]


def test_a_column_continues_below_the_table_at_the_top_of_the_next_column() -> None:
    """The right column starts with a table with its caption below it. Header and row names are short lines, and the
    unfinished sentence of the left column continues in the body text below the caption. When the column top is a
    full paragraph, it is joined as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, x0: float, y: float, x1: float, rows: int = 1) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 10 * rows)
        lines = [pymupdf.Rect(x0, y + 10 * k, x1, y + 10 * k + 9) for k in range(rows)]
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10, color="#000", bold=False)

    two = PageGeometry(600, (Column(72, 290), Column(307, 526)), (), reads_in_columns=True)
    left = [
        blk("Following standard practice, we formulate this as a tagging task", 72, 600, 290, rows=6),
        blk("include the maximal context and do not use a CRF", 72, 670, 290, rows=4),
    ]
    table = [
        blk("System", 317, 67, 343),
        blk("Fine-tuning approach", 317, 118, 394),
        blk("Feature-based approach (BERT)", 317, 153, 450),
    ]
    caption = blk("Table 7: CoNLL-2003 Named Entity Recognition results.", 307, 236, 526, rows=4)
    tail = blk("layer in the output. We use the representation of the first sub-token", 307, 290, 526, rows=3)
    later = blk("To ablate the fine-tuning approach, we apply the feature-based approach", 307, 330, 526, rows=5)
    units = _cross_page_units([*left, *table, caption, tail, later], 10.0, geometries=[two])
    assert [unit for unit in units if len(unit) == 2] == [[left[1], tail]]


def test_a_column_of_one_paragraph_still_continues_into_the_next_column() -> None:
    """The left column has an indented abstract on top and a single paragraph below, so no two blocks share a left
    edge; that paragraph still counts as a column and joins the top of the right column. An affiliation centered
    below the second author name at the top of the right column is not the column top. When no column on the page
    has two aligned blocks (slides), no column is formed."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, x0: float, y: float, x1: float, rows: int = 1, page: int = 0) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 10 * rows)
        lines = [pymupdf.Rect(x0, y + 10 * k, x1, y + 10 * k + 9) for k in range(rows)]
        return TextBlock(page=page, rect=rect, line_rects=lines, text=text, size=10, color="#000", bold=False)

    abstract = blk("Oddly satisfying videos can evoke strong viewer responses in many ways.", 83, 240, 280, rows=20)
    intro = blk("Accordingly, Oddly Satisfying Video Emotion Anal-", 63, 530, 290, rows=17)
    right = [
        blk("ysis (OSVEA) requires models to move beyond recognizing objects.", 307, 220, 526, rows=20),
        blk("To address the lack of a dataset, we construct one and annotate it.", 307, 430, 526, rows=20),
    ]
    units = _cross_page_units([abstract, intro, *right], 10.0, geometries=[])
    assert [unit for unit in units if len(unit) == 2] == [[intro, right[0]]]

    author = blk("Shixiang Zhu", 376, 162, 442)
    office = blk("Carnegie Mellon University", 350, 174, 468)
    left = [
        blk("Generative conformal prediction builds uncertainty sets from samples.", 83, 240, 280, rows=30),
        blk("allowing the set to reflect complex or multimodal predictive distributions", 63, 623, 290, rows=10),
    ]
    tail = blk("(Wang et al., 2023b). Probabilistic conformal prediction samples outputs.", 307, 209, 526, rows=6)
    later = blk("This efficiency, however, depends on drawing enough samples per input.", 307, 275, 526, rows=20)
    units = _cross_page_units([author, office, *left, tail, later], 10.0, geometries=[])
    assert [unit for unit in units if len(unit) == 2] == [[left[1], tail]]

    slide = blk("If we need probabilities we can do one exp at the end p1 = exp(log", 115, 509, 700, rows=3)
    title = blk("This avoids underflow from multiplying many small numbers", 105, 279, 700, rows=3)
    heading = blk("Larger ngrams", 115, 25, 400, page=1)
    bullet = blk("4-grams, 5-grams", 114, 150, 400, page=1)
    more = blk("Large datasets of large n-grams have been released", 114, 216, 700, rows=2, page=1)
    units = _cross_page_units([title, slide, heading, bullet, more], 10.0, geometries=[], halt_ids={id(heading)})
    assert all(len(unit) == 1 for unit in units)


def test_page_end_does_not_take_a_block_already_paired_to_the_next_column() -> None:
    """When a left-column block on the next page is already joined to the top of the right column, the end of the
    previous page must not take it as well. Otherwise the right-column block would be in no unit."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, x0: float, y: float, page: int, size: float = 9.0) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x0 + 180, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    wide = PageGeometry(600, (Column(40, 560),), ())
    end = blk("The optimizer then rewrites the", 40, 700, 0)
    host = blk("query into a plan and it keeps the", 40, 300, 1)
    tail = blk("cheapest alternative for each group.", 300, 60, 1)
    later = blk("Each group holds equivalent expressions.", 300, 200, 1)
    small = blk("Notes about the memo structure here", 40, 600, 1, size=8.0)
    blocks = [end, host, tail, later, small]
    units = _cross_page_units(blocks, 9.0, geometries=[wide, wide], halt_ids=set())
    assert sorted(id(block) for unit in units for block in unit) == sorted(id(block) for block in blocks)
    assert [host, tail] in units


def test_capital_continuation_joins_across_columns_only() -> None:
    """A column change may continue with a proper noun. A capitalized line break within the same column still splits."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, x0: float, y: float, page: int = 0) -> TextBlock:

        rect = pymupdf.Rect(x0, y, x0 + 180, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=9, color="#000", bold=False)

    wide = PageGeometry(600, (Column(40, 560),), ())
    host = blk("query engines that allow", 300, 700)
    host2 = blk("Several efforts have addressed this.", 300, 660)
    tail = blk("SQL-based processing of data in HDFS.", 40, 60, page=1)
    tail2 = blk("The exchange avoids another platform.", 40, 80, page=1)
    units = _cross_page_units([host2, host, tail, tail2], 9.0, geometries=[wide, wide])
    assert [unit for unit in units if len(unit) == 2] == [[host, tail]]
    # The second half of a word hyphenated at the line end (fine-) is always lowercase; a capitalized start after a
    # column change is a table header or heading.
    broken = blk("score. Without TriviaQA fine-", 300, 700)
    units = _cross_page_units([host2, broken, tail, tail2], 9.0, geometries=[wide, wide])
    assert all(len(unit) == 1 for unit in units)
    assert all(len(unit) == 1 for unit in _cross_page_units([host2, host, tail, tail2], 9.0))

    same_column = _merge_continuations(
        [
            blk("the previous line does not end", 40, 100),
            blk("SQL-based processing starts a new sentence.", 40, 114),
        ]
    )
    assert len(same_column) == 2

    # When the other column has no left edge, the left column at the page end must not take in a new paragraph
    # starting with a capital on the next page either.
    def edge(text: str, x0: float, y: float, page: int, x1: float = 280.0) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=9, color="#000", bold=False)

    two = PageGeometry(612, (Column(40, 290), Column(316, 560)), ())
    host = edge("tunately even with this setting, we were", 53.8, 700, 0)
    earlier = edge("To obtain a better coverage across systems,", 53.8, 680, 0)
    nxt = edge("For queries where HAWQ has the most speedups.", 53.8, 56, 1)
    nxt2 = edge("Impala joins the fact tables first.", 53.8, 72, 1)
    units = _cross_page_units(
        [earlier, host, nxt, nxt2],
        9.0,
        geometries=[two, two],
    )
    assert all(len(unit) == 1 for unit in units)
    lower = edge("and the sentence goes on", 53.8, 56, 1)
    joined = _cross_page_units(
        [earlier, host, lower, nxt2],
        9.0,
        geometries=[two, two],
    )
    assert [unit for unit in joined if len(unit) == 2] == [[host, lower]]


def test_same_page_continuations_merge() -> None:
    """A paragraph broken up by formula fragment lines ('…(like a' | 'sequence)…' | 'that belong…') is merged back
    into one block: translated separately the pieces give half translations with forced line breaks at the cuts.
    Headings and table rows are not affected."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, x0: float = 108.0, size: float = 10.0, nwords_pad: int = 0) -> TextBlock:
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(x0, y, 504, y + 11),
            line_rects=[pymupdf.Rect(x0, y, 504, y + 11)],
            text=text,
            size=size,
            color="#000",
            bold=False,
        )

    blocks = [
        blk("We are considering supervised tasks where the training set is given (like a", 280),
        blk("sequence) while the target output is a sequence of tokens", 292),
        blk("that belong to a fixed known dictionary.", 303),
        blk("2.1 Model", 330),
        blk("Given a single pair the log probability can be computed as follows", 345),
    ]
    merged = _merge_continuations(blocks)
    assert len(merged) == 3
    assert merged[0].text.endswith("fixed known dictionary.")
    assert merged[1].text == "2.1 Model"

    # The previous line has sentence-final punctuation, so no merge
    b2 = [
        blk("This is a full sentence with several words.", 100),
        blk("another paragraph starts here with lowercase", 112),
    ]
    assert len(_merge_continuations(b2)) == 2

    # The hanging indent is about 2.5em. With a 2em limit, which is too tight, continuation lines of a feature list
    # split into two paragraphs that shrink separately.
    hanging = [
        blk("Modularity. Using a highly extensible abstraction meta-", 100, x0=40),
        blk("data and system description stay together.", 112, x0=62),
    ]
    assert len(_merge_continuations(hanging)) == 1
    ended = [
        blk("This feature list item already ended.", 100, x0=40),
        blk("data and system description stay together.", 112, x0=62),
    ]
    assert len(_merge_continuations(ended)) == 2


def test_line_end_colon_inside_a_sentence_merges() -> None:
    """A colon at the line end with full lines above and below is a line break inside a sentence and is merged into one
    paragraph; when the next line is a short example or a table header, it is not merged."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, x0: float = 142.0, x1: float = 475.0) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 11)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    wrapped = [
        blk("The most common way to use this n-gram hierarchy is called interpolation:", 250, x0=157),
        blk("computing a new probability by interpolating (weighting and combining) the tri-", 262),
    ]
    assert len(_merge_continuations(wrapped)) == 1
    example = [
        blk("Consider the language of certain sheep, which consists of strings like:", 250),
        blk("baa! baaa! baaaa!", 262, x0=200, x1=280),
    ]
    assert len(_merge_continuations(example)) == 2
    intro = [
        blk("the steps are:", 250, x1=210),
        blk("compute the counts and normalize them so that they sum to one over the", 262),
    ]
    assert len(_merge_continuations(intro)) == 2


def test_slide_sub_item_is_not_a_title_fragment() -> None:
    """On a slide, a second-level item after a first-level list item has a larger font size and a different left edge,
    and is not the second line of a heading."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, y0: float, y1: float, size: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, 682, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("• Pushes as much work as possible onto the cluster for parallel execution", 115.2, 176.1, 214.0, 18),
        blk("– Metadata repository", 79.2, 215.3, 240.9, 22),
    ]
    assert [b.text for b in _merge_continuations(blocks)] == [
        "• Pushes as much work as possible onto the cluster for parallel execution",
        "– Metadata repository",
    ]


def test_punct_fragment_uses_line_end_not_union() -> None:
    """A formula fragment inside the block union but more than 2pt from the end of the previous line must not be
    donated for masking."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _donate_punct_rects

    line1 = pymupdf.Rect(72, 80, 400, 92)
    line2 = pymupdf.Rect(72, 96, 200, 108)
    prev = TextBlock(
        page=0,
        rect=line1 | line2,
        line_rects=[line1, line2],
        text="scale the dot products by",
        size=11,
        color="#000",
        bold=False,
    )
    far = pymupdf.Rect(210, 96, 240, 108)
    _donate_punct_rects(prev, [far])
    assert len(prev.line_rects) == 2
    # A tick number on a figure in the neighboring column, at the height of this line but far left of its end, does
    # not belong to this line either
    tick = pymupdf.Rect(20, 96, 28, 106)
    _donate_punct_rects(prev, [tick])
    assert len(prev.line_rects) == 2
    close = pymupdf.Rect(201, 96, 210, 108)
    _donate_punct_rects(prev, [close])
    assert prev.line_rects[-1] == close


def test_cross_page_does_not_merge_numbered_heading() -> None:
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, page: int, size: float = 10.0, y: float = 0, bold: bool = False) -> TextBlock:
        return TextBlock(
            page=page,
            rect=pymupdf.Rect(0, y, 200, y + 12),
            line_rects=[],
            text=text,
            size=size,
            color="#000",
            bold=bold,
        )

    blocks = [
        blk("The previous section ends without a period", 0, y=700),
        blk("1 Introduction", 1, y=60),
        blk("3.1 Encoder layers follow", 1, y=80),
    ]
    assert all(len(u) == 1 for u in _cross_page_units(blocks, 10.0))
    # A lowercase continuation paragraph is still merged
    cont = [
        blk("performs quite", 0, y=700),
        blk("well on this split.", 1, y=60),
    ]
    merged = [u for u in _cross_page_units(cont, 10.0) if len(u) == 2]
    assert len(merged) == 1


def test_affiliation_tails_split_and_stack() -> None:
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _split_affiliation_tails
    from app.formats.pdf_write_geometry import _affiliation_write_rects

    names = [
        pymupdf.Rect(100, 120, 400, 132),
        pymupdf.Rect(100, 134, 400, 146),
        pymupdf.Rect(116, 160, 180, 172),
    ]
    author = TextBlock(
        page=0,
        rect=names[0] | names[2],
        line_rects=names,
        text="Zhongxian Gu Entong Shen ∗Pivotal Inc.",
        size=12,
        color="#000",
        bold=False,
        span_lines=[[{"text": "Zhongxian Gu"}], [{"text": "Entong Shen"}], [{"text": "∗Pivotal Inc."}]],
    )
    datometry = TextBlock(
        page=0,
        rect=pymupdf.Rect(210, 160, 300, 184),
        line_rects=[pymupdf.Rect(210, 160, 300, 172), pymupdf.Rect(210, 172, 300, 184)],
        text="‡ Datometry Inc. San Francisco",
        size=10,
        color="#000",
        bold=False,
    )
    split = _split_affiliation_tails([author, datometry])
    assert len(split) == 3
    assert "Pivotal" not in split[0].text
    assert "Pivotal" in split[1].text
    stacked = _affiliation_write_rects(split)
    assert stacked[id(split[1])].y0 != stacked[id(split[2])].y0


def test_a_formula_line_wrapped_from_a_full_line_is_kept() -> None:
    """The previous line reaches the right edge of the text area without finishing the sentence, and the next line
    and W^O ∈ … continues from the left edge: it is the second half of the sentence, not a displayed formula.
    Stacked formulas do not reach the right edge."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_paragraphs import _wraps_from_above

    frame = TextFrame(108, 504, 72, 720, 10)
    line = pymupdf.Rect(107.5, 203.1, 502.8, 217.6)
    above = TextBlock(
        page=4,
        rect=line,
        line_rects=[line],
        text="Where the projections are parameter matrices W Q",
        size=10,
        color="#000000",
        bold=False,
    )
    tail = pymupdf.Rect(108, 215.6, 201.2, 227.9)
    rows = [pymupdf.Rect(108, y, 503, y + 12) for y in (100, 112, 124, 136)]
    assert _wraps_from_above([above], 4, tail, "and W O ∈Rhdv×dmodel.", rows, frame)
    short = pymupdf.Rect(107.5, 203.1, 200, 217.6)
    stacked = TextBlock(page=4, rect=short, line_rects=[short], text="et = Ext", size=10, color="#000000", bold=False)
    assert not _wraps_from_above([stacked], 4, tail, "ht = g(Uht−1 + Wet)", rows, frame)
    # In a two-column layout the previous line reaches the right edge of the left column, not of the text area
    column = pymupdf.Rect(108, 203.1, 297.5, 217.6)
    left = TextBlock(
        page=4,
        rect=column,
        line_rects=[column],
        text="where the mean is computed over",
        size=10,
        color="#000000",
        bold=False,
    )
    rows = [pymupdf.Rect(108, y, 298, y + 12) for y in (100, 112, 124, 136)]
    assert _wraps_from_above([left], 4, tail, "and x̂ = (x − µ)/σ.", rows, frame)
    # A centered caption: the last line shares the center line of the full lines above
    first, second = pymupdf.Rect(83.2, 135.0, 511.7, 147.1), pymupdf.Rect(87.9, 149.7, 507.8, 160.6)
    caption = [
        TextBlock(page=4, rect=line, line_rects=[line], text=text, size=10.9, color="#000000", bold=False)
        for line, text in (
            (first, "Table 15 | Comparison of models on"),
            (second, "Numbers in bold denote the performance is statistically"),
        )
    ]
    end = pymupdf.Rect(219.6, 161.2, 375.7, 174.2)
    assert _wraps_from_above(caption, 4, end, "significant (t−test with 𝑝< 0.01).", [], frame)


def test_a_line_with_three_words_among_formulas_is_translated() -> None:
    """Therefore, ∫… = ∫… It follows: many formulas and a low share of letters, but three body words make it a
    sentence."""
    from app.formats.pdf_paragraphs import _untranslatable

    def span(text: str, font: str) -> PdfDict:
        return {"text": text, "font": font}

    spans = [
        span("p(ϵ)", "CMMI10"),
        span("Therefore", "Times-Roman"),
        span(",", "Times-Roman"),
        span("∫q(z|x)f(z)dz = ∫p(ϵ)f(z)dϵ = ∫p(ϵ)f(g(ϵ,x))dϵ", "CMMI10"),
        span(".", "Times-Roman"),
        span("It", "Times-Roman"),
        span("follows", "Times-Roman"),
    ]
    text = "p(ϵ) Therefore, ∫q(z|x)f(z)dz = ∫p(ϵ)f(z)dϵ = ∫p(ϵ)f(g(ϵ,x))dϵ. It follows"
    assert not _untranslatable(spans, text)
    assert _untranslatable(spans[:4], "p(ϵ) Therefore, ∫q(z|x)f(z)dz = ∫p(ϵ)f(z)dϵ")


def test_a_centred_definition_line_keeps_its_glyphs() -> None:
    """A centered displayed formula with only a few connecting words (if and only if, for all) is not translated as a
    sentence. A formula line starting at the left edge of the column and continuing the previous line is still
    translated as a sentence."""
    from app.formats.pdf_extract import _displayed, _left_margins
    from app.formats.pdf_paragraphs import _untranslatable

    body = {"bbox": (72, 0, 540, 12), "spans": [{"text": "A body line long enough to count as the column margin."}]}
    data = {"blocks": [{"type": 0, "lines": [body] * 4}]}
    margins = _left_margins(data)
    assert margins == [72]
    spans = [
        {"text": "(a1, …, an) ⪯ (b1, …, bn)", "font": "CMMI10", "size": 10.0},
        {"text": "if and only if", "font": "Times-Roman", "size": 10.0},
        {"text": "ai ⪯ bi", "font": "CMMI10", "size": 10.0},
        {"text": "for all", "font": "Times-Roman", "size": 10.0},
        {"text": "i", "font": "CMMI10", "size": 10.0},
    ]
    text = "(a1, …, an) ⪯ (b1, …, bn) if and only if ai ⪯ bi for all i"
    assert _displayed(pymupdf.Rect(215, 100, 400, 112), spans, margins)
    assert not _displayed(pymupdf.Rect(72, 100, 400, 112), spans, margins)
    assert _untranslatable(spans, text, 8)
    assert not _untranslatable(spans, text)


def test_slide_title_and_a_box_label_below_stay_apart() -> None:
    """Small text boxes in a figure below a large slide title are not caption continuation lines, and the text in a
    star shape beside the title is not the next line of the title."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_paragraphs import _merge_caption_fragments

    def blk(text: str, x0: float, y0: float, x1: float, y1: float, size: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    title = [blk("Example 3: Improved Plan", 141, 27, 597, 73, 39), blk("scan ‘depts’", 132, 115, 209, 131, 14)]
    assert len(_merge_caption_fragments(title)) == 2
    assert len(_merge_continuations(title)) == 2

    star = [blk("Win", 404, 186, 433, 205, 17), blk("SortRemoveRule sorted on [deptno]", 595, 213, 695, 239, 12)]
    assert len(_merge_continuations(star)) == 2


def test_a_citation_year_after_et_al_continues_the_sentence() -> None:
    """Brants et al. | (2007) find that… is one sentence: the period of et al. is no sentence end, and the line starting
    with a parenthesis is the citation year. A number such as (a) is the next item, (2.11) is an example number, an
    indented line is a displayed formula, and a previous line ending with a period is not joined either."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, x1: float = 493.0) -> TextBlock:
        rect = pymupdf.Rect(160, y, x1, y + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    cite = [
        blk("The backoff terminates in the unigram, which has score S(w). Brants et al.", 307),
        blk("(2007) find that a value of 0.4 worked well for λ.", 319, x1=356),
    ]
    assert len(_merge_continuations(cite)) == 1
    item = [
        blk("the model is trained in two stages, described below", 307),
        blk("(a) pretraining on unlabeled text", 319, x1=320),
    ]
    assert len(_merge_continuations(item)) == 2
    indented = [
        blk("It could answer questions by parsing them into a logical form", 307),
        TextBlock(
            page=0,
            rect=pymupdf.Rect(182, 319, 470, 343),
            line_rects=[pymupdf.Rect(182, 319, 470, 331), pymupdf.Rect(196, 331, 440, 343)],
            text="(TEST (FOR SOME X16 / (SEQ SAMPLES) : T ; (CONTAIN’ X16 (NPR* X17 / (QUOTE AL203))",
            size=10.0,
            color="#000",
            bold=False,
        ),
    ]
    assert len(_merge_continuations(indented)) == 2
    tag = [
        blk("where we’ve explicitly marked the spaces between words: _", 307),
        blk("(2.11) set new new renew reset renew", 319, x1=330),
    ]
    assert len(_merge_continuations(tag)) == 2
    ended = [
        blk("This algorithm does not produce a probability distribution.", 307),
        blk("(Unlike interpolation, the weights are fixed.)", 319, x1=380),
    ]
    assert len(_merge_continuations(ended)) == 2


def test_the_short_centered_last_line_of_a_caption_continues_it() -> None:
    """The last line of a centered table note is short and indented on both sides. With a matching center line and a
    lowercase start, it is joined to the previous line."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, x1: float, y: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.9, color="#000", bold=False)

    caption = [
        blk(
            "Table 3 | Experimental results at each stage of DeepSeek-R1. Numbers in bold denote the", 84.0, 510.9, 85.5
        ),
        blk("performance is statistically significant (t-test with p < 0.01).", 153.5, 441.5, 98.2),
    ]
    assert len(_merge_continuations(caption)) == 1
    # An indented line with a different center line is something else
    offset = [caption[0], blk("performance is statistically significant (t-test with p < 0.01).", 200.0, 488.0, 98.2)]
    assert len(_merge_continuations(offset)) == 2
    # A last line with a single word, all body words and ending with a period, is joined too
    short = [
        blk(
            "Figure 7 | The experiment results of Language Consistency (LC) Reward during reinforcement",
            71.0,
            524.0,
            439.0,
        ),
        blk("learning.", 276.0, 319.0, 452.5),
    ]
    assert len(_merge_continuations(short)) == 1


def test_a_line_ending_on_a_function_word_continues_into_a_capital() -> None:
    """When the previous line stops on and or of, a next line starting with a capital (a proper name) continues it;
    a next line that is another list item or a smaller footnote is not joined. The last line of a centered
    paragraph starting with a capital is joined, a smaller source below an epigraph is not. A line starting with a
    period is joined."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, x1: float, y: float, size: float = 10.9) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 11)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=size,
            color="#000",
            bold=False,
            span_lines=[[{"text": text, "font": "Palatino", "size": size}]],
        )

    def head() -> TextBlock:
        return blk(
            "safe answers to questions, rather than simply declining to respond). For DeepSeek-V3 and", 80, 515, 141
        )

    assert (
        len(
            _merge_continuations([head(), blk("DeepSeek-R1, we report results under two configurations", 85, 510, 154)])
        )
        == 1
    )
    assert (
        len(_merge_continuations([head(), blk("• DeepSeek-R1 is reported under two configurations", 85, 510, 154)]))
        == 2
    )
    assert (
        len(_merge_continuations([head(), blk("Available at the project page of the authors", 85, 510, 154, size=9.0)]))
        == 2
    )
    caption = blk(
        "Table 6 | DeepSeek-R1 Distilled Models, their corresponding Base Models, and Initial Learning", 71, 524, 275
    )
    assert len(_merge_continuations([caption, blk("Rates.", 282, 313, 288)])) == 1
    poem = blk("Dim as an evening shadow in a brook,", 274, 433, 250, size=10.0)
    assert len(_merge_continuations([poem, blk("Thomas Lovell Beddoes, 1851", 298, 408, 263, size=9.0)])) == 2
    words = blk("we counted some representative reflective words, including wait, mistake, however", 71, 524, 300)
    assert (
        len(_merge_continuations([words, blk(". These reflective words were selected by 3 experts", 71, 524, 313)]))
        == 1
    )


def test_a_centered_formula_or_a_hanging_item_line_keeps_its_place() -> None:
    """A centered formula line below a paragraph is not the last line of a centered paragraph; a continuation line of a
    hanging-indent entry is joined even when it starts with an opening parenthesis."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, x1: float, y: float, lines: list[pymupdf.Rect] | None = None) -> TextBlock:
        rects = lines or [pymupdf.Rect(x0, y, x1, y + 12)]
        rect = pymupdf.Rect(rects[0])
        for r in rects[1:]:
            rect |= r
        return TextBlock(page=0, rect=rect, line_rects=rects, text=text, size=10.0, color="#000", bold=False)

    formula = [
        blk(
            "states that if the language is regular in certain ways (to be exact, if it is both stationary and",
            160,
            493,
            300,
        ),
        blk("lim -1/n log p(w1:n)", 270, 383, 318),
    ]
    assert len(_merge_continuations(formula)) == 2
    # A formula line forming a sentence also has space above and below: rel(A,B) = log(max(|in(A)|,|in(B)|)) ...
    spaced = [formula[0], blk("rel(A,B) = log(max(in(A), in(B))) - log(in(A) and in(B))", 215, 438, 318)]
    assert len(_merge_continuations(spaced)) == 2
    indent = [
        blk("A hidden Markov model (HMM) allows us to talk about both observed events", 157, 475, 300),
        blk("(like words that we see in the input) and hidden events", 142, 475, 312),
    ]
    assert len(_merge_continuations(indent)) == 1
    item = [
        blk(
            "brand new NPs: these introduce entities that are discourse-new and hearer-old",
            142,
            475,
            300,
            [pymupdf.Rect(142, 300, 475, 312), pymupdf.Rect(157, 312, 475, 324)],
        ),
        blk("(like Hong Kong, Marie Curie, or the New York Times.", 157, 380, 324),
    ]
    assert len(_merge_continuations(item)) == 1
    # The entry has one line and no continuation line to compare with: for an entry starting with a bold label,
    # the hanging-indent line starting with an opening parenthesis is its continuation line too
    label = blk("unused NPs: these introduce entities that are discourse-new but hearer-old", 185, 493, 656)
    label.span_lines = [
        [
            {"text": "unused NPs:", "font": "NimbusRomNo9L-Medi", "flags": 16},
            {"text": " ", "font": "NimbusRomNo9L-Regu", "flags": 0},
            {
                "text": "these introduce entities that are discourse-new but hearer-old",
                "font": "NimbusRomNo9L-Regu",
                "flags": 0,
            },
        ]
    ]
    single = [label, blk("(like Hong Kong, Marie Curie, or the New York Times.", 207, 425, 668)]
    assert len(_merge_continuations(single)) == 1


def test_a_block_inside_an_open_code_fence_continues_with_a_capital() -> None:
    """The previous block stops inside an unclosed code fence ("‘sql opened and not yet closed), so JOIN in the next
    block continues the code; once the fence is closed, a capitalized start splits as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float) -> TextBlock:
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(82, y, 514, y + 10),
            line_rects=[pymupdf.Rect(82, y, 514, y + 10)],
            text=text,
            size=10.0,
            color="#000",
            bold=False,
        )

    fenced = [
        blk(
            "the query would look like this: “‘sql SELECT * FROM table1 AS t1 JOIN table2 ON t1.code = table2.code", 332
        ),
        blk("JOIN table3 ON t1.code = table3.code; “‘ This query first selects the top 10 rows", 344),
    ]
    assert len(_merge_continuations(fenced)) == 1
    closed = [
        blk("Extract the code wrapped by “‘python“‘ in the response and run the test cases", 332),
        blk("Judge whether the answer passes the test cases from the problem statement", 344),
    ]
    assert len(_merge_continuations(closed)) == 2


def test_a_line_opening_with_an_inline_formula_continues_the_sentence() -> None:
    """A line starting with an inline formula at the column edge and continuing with body text is a wrapped line of
    the previous sentence; an indented displayed formula is not."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(spans: list[tuple[str, str]], x0: float, x1: float, y: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y, x1, y + 12)
        text = "".join(text for text, _font in spans)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=9.7,
            color="#000",
            bold=False,
            span_lines=[
                [{"text": text, "font": font, "size": 9.7, "math": font.startswith("CM")} for text, font in spans]
            ],
        )

    # The body text is Times, so the CMR 1/ is a formula (marked math during extraction)
    def head() -> TextBlock:
        return blk([("whereas SGD is low at learning rare features. Adam with", "Times-Roman")], 105, 490, 682)

    wrap = blk([("1/", "CMR10"), ("t decay on its stepsize should theoretically match", "Times-Roman")], 105, 490, 694)
    assert len(_merge_continuations([head(), wrap])) == 1
    display = blk([("x", "CMMI10"), (" = y + z for all inputs", "Times-Roman")], 160, 300, 694)
    assert len(_merge_continuations([head(), display])) == 2
    # The fraction fragment at the line start became a separate block; the rest starts with the denominator and
    # reaches the right edge, and the next line returns to the column edge
    rest = blk(
        [("1−α", "CMR7"), (" had similar results. The first choice is optimal for", "Times-Roman")], 164, 490, 694
    )
    rest.line_rects = [pymupdf.Rect(164, 694, 490, 706), pymupdf.Rect(105, 706, 490, 718)]
    rest.rect = pymupdf.Rect(105, 694, 490, 718)
    assert len(_merge_continuations([head(), rest])) == 1
    # A footnote starting with a superscript number is one size smaller and is no wrapped line
    note = blk(
        [("2", "CMR6"), ("The analysis assumes that every possible cell of the data cube exists.", "Times-Roman")],
        164,
        490,
        694,
    )
    note.line_rects, note.rect, note.size = rest.line_rects, rest.rect, 8.0
    assert len(_merge_continuations([head(), note])) == 2


def test_a_line_far_below_by_the_documents_own_spacing_is_not_a_continuation() -> None:
    """Slide fonts have tall line boxes, so two tightly set lines measure as overlapping. A table header 1.3 em below
    a subtitle is still within line spacing by a fixed 1.5 em, but with the line gap of this document subtracted
    it is a separate block."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y0: float, y1: float, size: float, line_gap: float) -> TextBlock:
        rect = pymupdf.Rect(114, y0, 1122, y1)
        return TextBlock(
            page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False, line_gap=line_gap
        )

    def slide(line_gap: float) -> list[TextBlock]:
        return [
            blk("Out of 9222 sentences", 150, 205, 45.0, line_gap),
            blk("i want to eat chinese food lunch spend", 263, 304, 31.9, line_gap),
        ]

    assert len(_merge_continuations(slide(0.1))) == 1
    assert len(_merge_continuations(slide(-0.15))) == 2


def test_a_running_head_does_not_take_the_first_line_below_it() -> None:
    """A book running header is right-aligned without a period; its pieces look like the second half of a line, and
    the next line returns to the column edge starting lowercase. Once recognized as a running header it is not
    joined, wherever the line spacing threshold lies."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_same_line_sentences

    def blocks() -> list[TextBlock]:
        head = [
            pymupdf.Rect(378, 82.8, 391.6, 94.8),
            pymupdf.Rect(401.8, 82.8, 405.3, 94.8),
            pymupdf.Rect(415.5, 82.8, 466.1, 94.8),
        ]
        body = [pymupdf.Rect(160, 104, 493, 116), pymupdf.Rect(160, 116, 493, 128)]
        return [
            TextBlock(
                page=0,
                rect=pymupdf.Rect(378, 82.8, 466.1, 94.8),
                line_rects=head,
                text="5.5 • WORD2VEC",
                size=10.0,
                color="#000",
                bold=False,
            ),
            TextBlock(
                page=0,
                rect=pymupdf.Rect(160, 104, 493, 128),
                line_rects=body,
                text="automobile are distinct and unrelated; sparse vectors may thus fail to capture",
                size=10.0,
                color="#000",
                bold=False,
            ),
        ]

    assert len(_merge_same_line_sentences(blocks())) == 1
    head, body = blocks()
    assert len(_merge_same_line_sentences([head, body], {id(head)})) == 2


def test_a_slide_sub_point_is_not_a_wrapped_title() -> None:
    """Slide body text is around 30pt, so an indented sub-item is not the second line of a wrapped large heading.
    Large sizes are compared with the body size of this document; when no body size was measured, 12pt is used
    as before."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def slide(body_size: float) -> list[TextBlock]:
        def blk(text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
            rect = pymupdf.Rect(x0, y0, x1, y1)
            return TextBlock(
                page=0,
                rect=rect,
                line_rects=[rect],
                text=text,
                size=27.0,
                color="#000",
                bold=False,
                body_size=body_size,
            )

        return [
            blk("Example: Deterministic language L = {red,blue, green}", 77, 229, 672, 262),
            blk("Branching factor = 3, any word can be followed by red, blue, green", 163, 272, 903, 305),
        ]

    assert len(_merge_continuations(slide(38.0))) == 2
    assert len(_merge_continuations(slide(0.0))) == 1


def test_a_bracket_or_quote_closes_a_sentence_only_after_its_end_mark() -> None:
    """A line ending with a closing parenthesis or quote ends a sentence only when sentence-final punctuation comes
    before it. What follows a citation year (Vilain et al., 1995) or code r"cat|dog" is the same sentence. A
    parenthesis right after a function name or variable ends a formula, and a formula or where below it is not
    joined."""
    from app.formats.pdf_blocks import TextBlock, _ends_sentence
    from app.formats.pdf_continuations import _merge_continuations

    assert _ends_sentence("found it.)") and _ends_sentence('he said."')
    assert not _ends_sentence("The MUC F-measure (Vilain et al., 1995)")
    assert not _ends_sentence('The pattern r"cat|dog"')
    assert _ends_sentence("yi = softmax(hL i WK)")

    def blk(text: str, y: float, x0: float = 142.2) -> TextBlock:
        rect = pymupdf.Rect(x0, y, 475.2, y + 12)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    citation = [
        blk("Let’s just explore two of the metrics. The MUC F-measure (Vilain et al., 1995)", 484),
        blk("is based on the number of coreference links (pairs of mentions) common to H and R.", 496),
    ]
    assert len(_merge_continuations(citation)) == 1
    formula = [blk("yi = softmax(hL i WK)", 179, x0=262), blk("ti = argmaxk(yi)", 194, x0=263)]
    assert len(_merge_continuations(formula)) == 2


def test_a_list_ends_where_the_next_block_starts_left_of_the_item() -> None:
    """The first line after a list item that starts further left than the item is body text after the list;
    hanging-indent continuation lines are joined as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, y0: float, x1: float, y1: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y1)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    after = [
        blk("8. none (1)", 85, 264, 139, 276),
        blk("none indicates that there are no attributes in the GROUP BY clause.", 71, 289, 540, 301),
    ]
    assert len(_merge_continuations(after)) == 2
    item = [
        blk(
            "2. fluency: how fluent the translation is in the target language (is it grammatical,",
            172.7,
            267.9,
            493.2,
            280.9,
        ),
        blk("clear, readable, natural).", 185.1, 280.2, 281.2, 292.2),
    ]
    assert len(_merge_continuations(item)) == 1


def test_a_verbatim_line_does_not_run_into_the_prose_below() -> None:
    """A question and answer reproduced in monospace, Q: ... A: Charles, has no sentence-final punctuation; the body
    text on the next line, back at the column edge and starting lowercase, is not its continuation line."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _line_tail_match

    def blk(text: str, box: tuple[float, float, float, float], font: str) -> TextBlock:
        rect = pymupdf.Rect(box)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=10.0,
            color="#000",
            bold=False,
            span_lines=[[{"text": text, "size": 10.0, "font": font, "flags": 8 if font == "txtt" else 0}]],
        )

    answer = blk("A: Charles", (376, 111, 429, 121), "txtt")
    prose = blk(
        "then the same kinds of training data would lead the language model to",
        (142, 126, 475, 138),
        "NimbusRomNo9L-Regu",
    )
    assert not _line_tail_match(answer, prose)
    tail = blk("A: Charles", (376, 111, 429, 121), "NimbusRomNo9L-Regu")
    assert _line_tail_match(tail, prose)
    # Monospace text inside body text wraps to the next line; both sides are monospace, so they are joined as usual
    mixed = blk("1. The prompt is assembled, including", (376, 111, 429, 121), "NimbusRomNo9L-Regu")
    assert mixed.span_lines is not None
    mixed.span_lines.append([{"text": "<user> Who wrote ‘‘The Origin", "size": 10.0, "font": "txtt", "flags": 8}])
    rest = blk('of Species"?', (142, 126, 220, 138), "txtt")
    assert _line_tail_match(mixed, rest)


def test_a_full_line_ending_in_a_call_runs_on_into_the_next_line() -> None:
    """f(g(x)) at the line end counts as a sentence end, so where below a displayed formula is not joined. A full line
    followed by a line starting with a body word is joined as usual; a following line that is another formula is
    not."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, font: str = "NimbusRomNo9L-Regu") -> TextBlock:
        rect = pymupdf.Rect(108, y, 504, y + 11)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=10,
            color="#000",
            bold=False,
            span_lines=[[{"text": text, "font": font, "size": 10.0}]],
        )

    head = blk("that a differentiable estimator can be constructed: q(z|x)f(z) dz ≃ f(g(x, e))", 84)
    tail = blk("where e ∼ p(e). In section 2.3 we applied this trick.", 96)
    assert len(_merge_continuations([head, tail])) == 1
    formula = blk("we write the next sublayer as t4 = LayerNorm(t3)", 84)
    after = blk("hi = t5 + t3", 96)
    assert len(_merge_continuations([formula, after])) == 2


def test_table_split_keeps_space_spans() -> None:
    """In justified lines the word spaces are separate spans. They must be included when text is reassembled by cells,
    otherwise the words run together."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_drawings import RuledTable
    from app.formats.pdf_paragraphs import _split_table_blocks

    def span(text: str, x0: float, x1: float) -> PdfDict:
        return {"text": text, "bbox": (x0, 100, x1, 110), "origin": (x0, 108), "font": "Times", "size": 10.0}

    first = [
        span("Sample", 140, 170),
        span(" ", 170, 176),
        span("minibatch", 176, 220),
        span(" ", 220, 226),
        span("rule.", 226, 250),
    ]
    second = [
        {**item, "bbox": (item["bbox"][0], 112, item["bbox"][2], 122), "origin": (item["origin"][0], 120)}
        for item in (span("We", 140, 152), span(" ", 152, 158), span("used", 158, 180))
    ]
    rects = [pymupdf.Rect(140, 100, 250, 110), pymupdf.Rect(140, 112, 180, 122)]
    block = TextBlock(
        page=0,
        rect=rects[0] | rects[1],
        line_rects=rects,
        text="",
        size=10,
        color="#000",
        bold=False,
        span_lines=[first, second],
    )
    table = RuledTable(
        pymupdf.Rect(100, 98, 300, 124),
        (pymupdf.Rect(100, 98, 300, 111), pymupdf.Rect(100, 111, 300, 124)),
        algorithm=True,
    )
    parts = _split_table_blocks([block], (table,))
    assert [part.text for part in parts] == ["Sample minibatch rule.", "We used"]


def test_a_displayed_formula_set_in_the_text_italic_is_kept() -> None:
    """In a book writing variables in body italics, the font does not reveal formula characters in displayed
    formulas. Those with an operator and at most two body words keep the original glyphs; a whole sentence set on
    its own line (L(ŷ,y) = How much ŷ differs from the true y) is translated."""
    from app.formats.pdf_paragraphs import _untranslatable

    def spans(*parts: tuple[str, str]) -> list[PdfDict]:
        return [{"text": text, "font": font, "size": 10.0, "flags": 2 if "Ital" in font else 0} for text, font in parts]

    roman, italic, sans, cmr = "NimbusRomNo9L-Regu", "NimbusRomNo9L-ReguItal", "CMSSBX10", "CMR10"
    line = spans(
        ("Q", sans),
        (" =", cmr),
        (" H", sans),
        ("dec", italic),
        ("W", sans),
        ("Q", sans),
        (";", roman),
        ("K", sans),
        (" =", cmr),
        (" H", sans),
        ("enc", italic),
        ("W", sans),
        ("K", sans),
    )
    assert _untranslatable(line, "Q = HdecWQ; K = HencWK", 8, True)
    line = spans(
        ("ˆ", roman),
        ("y", italic),
        ("i", italic),
        ("=", cmr),
        ("argmax", roman),
        ("char", roman),
        ("∈", "CMSY10"),
        ("Alphabet", roman),
        ("P", italic),
        ("(", cmr),
        ("char", roman),
        (")", cmr),
    )
    assert _untranslatable(line, "ˆyi = argmaxchar∈AlphabetP(char)", 8, True)
    line = spans(
        ("L", italic),
        ("(", cmr),
        ("y", italic),
        (")", cmr),
        ("=", cmr),
        ("How much", roman),
        ("y", italic),
        (" differs from the true", roman),
        (" y", italic),
    )
    assert not _untranslatable(line, "L(y) = How much y differs from the true y", 8, True)
    line = spans(("convolution with width-1 kernel:", italic), ("z", sans), ("j", italic), ("=", cmr), ("x", sans))
    assert not _untranslatable(line, "convolution with width-1 kernel: zj = x", 8, True)


def test_a_smaller_line_in_another_colour_does_not_continue_a_title() -> None:
    """A smaller purple example below a slide title (51pt, gray) is not a wrapped line of the title and is not merged
    with it."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    title = pymupdf.Rect(97.9, 63.0, 859.0, 125.3)
    item = pymupdf.Rect(112.3, 151.7, 1087.7, 198.1)
    blocks = [
        TextBlock(
            page=0,
            rect=title,
            line_rects=[title],
            text="Berkeley Restaurant Project sentences",
            size=51.0,
            color="#404040",
            bold=False,
            body_size=38.0,
        ),
        TextBlock(
            page=0,
            rect=item,
            line_rects=[item],
            text="can you tell me about any good cantonese restaurants",
            size=38.0,
            color="#330099",
            bold=False,
            body_size=38.0,
        ),
    ]
    assert len(_merge_continuations(blocks)) == 2


def test_an_index_entry_ends_at_its_page_numbers() -> None:
    """On an index page an entry is complete at its page number: a next line starting lowercase is another entry and is
    not merged. Body lines rarely stop on ", number"."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_extract import _index_page

    def page(lines: list[str]) -> PdfDict:
        return {"blocks": [{"type": 0, "lines": [{"spans": [{"text": line}]} for line in lines]}]}

    entries = [f"term number {i}, {100 + i}" for i in range(12)] + [
        "sub entry",
        "linear chain CRF, 414, 415",
        "Kleene *, 50–52",
        "fully-connected, 157",
    ]
    assert _index_page(page(entries))
    prose = [f"this is line {i} of an ordinary paragraph that wraps" for i in range(16)]
    assert not _index_page(page(prose))

    def blk(text: str, y: float, entry: bool) -> TextBlock:
        rect = pymupdf.Rect(80, y, 200, y + 9)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=8.0,
            color="#000",
            bold=False,
            span_lines=[[{"text": text, "font": "Times", "size": 8.0}]],
            entry=entry,
        )

    blocks = [blk("Frump, 497", 100, True), blk("fully-connected, 157", 110, True)]
    assert len(_merge_continuations(blocks)) == 2


def test_page_numbers_in_the_next_block_join_the_index_entry() -> None:
    """The page numbers wrap to the next line and land in the next block. A line with only page numbers right below an
    entry ending on a comma is joined to the entry; the entry starting below is not."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_paragraphs import _wraps_from_above

    def blk(text: str, x0: float, y0: float, x1: float, entry: bool = False) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y0 + 9.6)
        return TextBlock(
            page=0, rect=rect, line_rects=[rect], text=text, size=8.0, color="#000", bold=False, entry=entry, index=True
        )

    host = blk("Berkeley Restaurant Project,", 401.2, 643.0, 493.2, entry=True)
    refs = pymupdf.Rect(425.1, 652.5, 433.1, 662.1)
    assert _wraps_from_above([host], 0, refs, "72", [], None, index_page=True)
    assert not _wraps_from_above([host], 0, refs, "72", [], None, index_page=False)
    merged = _merge_continuations([host, blk("72", 425.1, 652.5, 433.1, entry=True), blk("BERT", 401.2, 662.0, 421.1)])
    assert [block.text for block in merged] == ["Berkeley Restaurant Project, 72", "BERT"]


def test_the_rest_of_a_broken_word_continues_the_line_above() -> None:
    """When the previous line stops on half a hyphenated word, the next line continues it at any indent
    (expres- | sions, 50, 65), and despite few letters it is not kept in the original as a line of page numbers.
    On an index page a line indented deeper than the previous one is a continuation line too (as, 51)."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _wraps_from_above

    def blk(text: str, rect: pymupdf.Rect) -> TextBlock:
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=8.0, color="#000", bold=False)

    above = blk("anchors in regular expres-", pymupdf.Rect(299.3, 366.9, 391.3, 376.5))
    rest = pymupdf.Rect(323.2, 376.1, 363.5, 386.5)
    assert _wraps_from_above([above], 0, rest, "sions, 50, 65", [], None)
    whole = blk("pipe in regular expressions", pymupdf.Rect(205.3, 121.5, 289.3, 131.1))
    tail = pymupdf.Rect(221.3, 130.7, 239.8, 141.0)
    assert not _wraps_from_above([whole], 0, tail, "as, 51", [], None)
    assert _wraps_from_above([whole], 0, tail, "as, 51", [], None, index_page=True)


def test_one_sentence_per_line_is_not_joined_into_a_paragraph() -> None:
    """Examples set line by line on a slide each take one line with aligned left edges; in several places the first
    word of the next line would still fit after the previous line, and even with a lowercase start they are not
    merged. In a paragraph, a wrapped line with no room for the next word is joined as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_paragraphs import _line_lists

    def blk(text: str, y: float, x1: float, size: float = 38.0) -> TextBlock:
        rect = pymupdf.Rect(112.3, y, x1, y + 46.4)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    frame = TextFrame(97.9, 1088.0, 11.0, 600.0, 38.0)
    examples = [
        blk("can you tell me about any good cantonese restaurants close by", 151.7, 1087.7),
        blk("tell me about chez panisse", 210.8, 524.4),
        blk("i’m looking for a good place to eat breakfast", 269.8, 795.0),
        blk("when is caffe venezia open during the day", 329.8, 764.4),
    ]
    listed = _line_lists(examples, [frame])
    assert len(_merge_continuations(examples, listed)) == 4
    paragraph = [
        blk("the model is trained on a large corpus of text and then it is", 151.7, 1087.7),
        blk("evaluated on a held out set of sentences that were never seen", 210.8, 1085.0),
        blk("during training so that the estimate of its quality is fair", 269.8, 1080.0),
        blk("and unbiased", 329.8, 400.0),
    ]
    assert _line_lists(paragraph, [frame]) == set()


def test_a_word_broken_at_the_block_end_continues_in_the_next_block() -> None:
    """When the previous block stops on half a hyphenated word, it is joined to the next block even with only two words
    (retrieval-augmented genera- | tion, 267)."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, x0: float, y0: float, x1: float, entry: bool = False) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y0 + 9.6)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=8.0,
            color="#000",
            bold=False,
            span_lines=[[{"text": text, "font": "Times", "size": 8.0}]],
            entry=entry,
        )

    merged = _merge_continuations(
        [blk("retrieval-augmented genera-", 95.4, 217.0, 187.4), blk("tion, 267", 119.3, 226.2, 147.7, entry=True)]
    )
    assert [block.text for block in merged] == ["retrieval-augmented generation, 267"]
    # A two-word label not followed by a broken word is not joined
    kept = _merge_continuations([blk("Figure label", 95.4, 217.0, 150.0), blk("tion, 267", 119.3, 226.2, 147.7)])
    assert len(kept) == 2


def test_a_model_name_broken_at_its_hyphen_continues_with_the_number() -> None:
    """The line ends on the hyphen of a model name (VE- | 8. Both datasets …), so the number on the next line is the
    second half of the model name, not a list number. When joined, the hyphen stays and no space is inserted"""
    from app.formats.pdf_blocks import TextBlock, _join_lines
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y0: float, x1: float) -> TextBlock:
        rect = pymupdf.Rect(54.0, y0, x1, y0 + 10.0)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=10.0,
            color="#000",
            bold=False,
            span_lines=[[{"text": text, "font": "Times", "size": 10.0}]],
        )

    merged = _merge_continuations(
        [
            blk("Table 5: Out-of-distribution evaluation on VCE and VE-", 226.0, 292.0),
            blk("8. Both datasets are evaluated after re-annotation.", 237.0, 280.0),
        ]
    )
    assert [block.text for block in merged] == [
        "Table 5: Out-of-distribution evaluation on VCE and VE-8. Both datasets are evaluated after re-annotation."
    ]
    assert _join_lines(["control system) and GPT-", "4o (2024-05-13)"]) == "control system) and GPT-4o (2024-05-13)"


def test_an_indented_paragraph_start_is_not_a_continuation() -> None:
    """In a document with first-line indents, a block with a first-line indent starting with a capitalized word is a
    new paragraph and is not joined even when the previous block stops on and (with a displayed formula below).
    A block starting lowercase or with a formula at the line start is joined as before."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y: float, rows: Sequence[tuple[float, float]]) -> TextBlock:
        lines = [pymupdf.Rect(x0, y + 12 * i, x1, y + 12 * i + 12) for i, (x0, x1) in enumerate(rows)]
        rect = pymupdf.Rect(lines[0])
        for line in lines:
            rect |= line
        return TextBlock(page=0, rect=rect, line_rects=lines, text=text, size=10.0, color="#000", bold=False)

    body = "Words of a paragraph run across the column and wrap onto the next line. It ends here."
    para = [(64, 299), (54, 299), (54, 200)]

    def plan(text: str) -> list[TextBlock]:
        blocks = [blk(body, 100 + 36 * i, para) for i in range(4)]
        return _merge_continuations(
            [*blocks, blk("The forward pass gives its overlap, and", 244, [(54, 299)]), blk(text, 256, para)]
        )

    assert len(plan("The adjoint state carries the sensitivity of the final error to the control.")) == 6
    assert len(plan("the adjoint state carries the sensitivity of the final error to the control.")) == 5


def test_a_paragraph_continues_after_a_page_of_figures() -> None:
    """The next page is a full-page figure with only a caption and the subfigure note (C) … right next to it: the
    unfinished paragraph of the previous page continues on the page after that. When a body-size paragraph follows
    right below the caption, that page has body text and is not skipped."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_merge import _cross_page_units

    def blk(text: str, page: int, y: float, rows: int = 3, size: float = 10.0) -> TextBlock:
        rect = pymupdf.Rect(72, y, 540, y + 12 * rows)
        lines = [pymupdf.Rect(72, y + 12 * k, 540, y + 12 * k + 11) for k in range(rows)]
        return TextBlock(page=page, rect=rect, line_rects=lines, text=text, size=size, color="#000", bold=False)

    host = blk("By the end of training, A is mutually", 0, 600)
    caption = blk("Figure 5: Model comparison on the supervised dataset.", 1, 519, rows=5)
    panel = blk("(C) Difference in summed AIC relative to the best-fitting model.", 1, 584, rows=3)
    tail = blk("excitatory with all units; B and C are mutually inhibitory.", 2, 74, rows=10)
    units = _cross_page_units([host, caption, panel, tail], 10.0)
    assert [host, tail] in units
    body = blk("bie and Shutova (2022), who show that ablating induction heads hurts.", 1, 584, rows=6)
    units = _cross_page_units([host, caption, body, tail], 10.0)
    assert [host, tail] not in units


def test_a_fraction_wrapped_to_the_line_start_ends_the_sentence_above() -> None:
    """An inline fraction at the end of a sentence wraps to the start of the next line (the log of the odds ratio |
    p/(1−p):); the stacked numerator and denominator are taller than a line and no body words follow, but it still
    ends the sentence and is joined back to the previous paragraph. One set apart by space and indented is a
    displayed formula."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_paragraphs import _wraps_from_above

    frame = TextFrame(142, 475, 72, 720, 10)
    rows = [pymupdf.Rect(142.2, y, 475.2, y + 12) for y in (488.8, 500.8, 512.8, 524.7)]
    line = pymupdf.Rect(142.2, 524.7, 475.2, 536.7)
    host = TextBlock(
        page=0,
        rect=pymupdf.Rect(142.2, 500.8, 475.2, 536.7),
        line_rects=rows[1:],
        text="This is because the logit function is the inverse of the sigmoid. "
        "The logit function is the log of the odds ratio",
        size=10,
        color="#000000",
        bold=False,
        span_lines=[
            [
                {
                    "text": "The logit function is the log of the odds ratio",
                    "size": 10.0,
                    "font": "NimbusRomNo9L-Regu",
                    "bbox": _box(line),
                }
            ]
        ],
    )
    tail = pymupdf.Rect(143.4, 534.5, 161.0, 556.4)
    assert _wraps_from_above([host], 0, tail, "p 1−p:", rows, frame)
    displayed = pymupdf.Rect(243.8, 548.0, 338.1, 570.0)
    assert not _wraps_from_above([host], 0, displayed, "logit(p) = ln p 1−p", rows, frame)
    # A number one line high is a cell of the table below
    assert not _wraps_from_above([host], 0, pymupdf.Rect(143.4, 537.0, 170.0, 548.0), "0.00025", rows, frame)

    def span(text: str, font: str) -> PdfDict:
        return {"text": text, "size": 7.4, "font": font, "bbox": _box(tail)}

    fraction = TextBlock(
        page=0,
        rect=tail,
        line_rects=[tail],
        text="1−p: p",
        size=7.4,
        color="#000000",
        bold=False,
        span_lines=[
            [
                span("1", "NimbusRomNo9L-Regu"),
                span("−", "CMSY10"),
                span("p", "NimbusRomNo9L-ReguItal"),
                span(":", "NimbusRomNo9L-Regu"),
            ],
            [span("p", "NimbusRomNo9L-ReguItal")],
        ],
    )
    merged = _merge_continuations([host, fraction])
    assert len(merged) == 1 and merged[0].text.endswith("odds ratio 1−p: p")


def test_a_line_ending_on_an_equals_sign_runs_on() -> None:
    """When the previous line stops on an equals sign, a next line starting with a digit is the other half of the
    formula and is joined into one paragraph."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def blk(text: str, y0: float, y1: float) -> TextBlock:
        rect = pymupdf.Rect(167, y0, 475, y1)
        span: PdfDict = {"text": text, "font": "NimbusRomNo9L-Regu", "size": 10.0, "flags": 0}
        return TextBlock(
            page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000", bold=False, span_lines=[[span]]
        )

    head = blk(
        "If we use linear interpolation smoothing between a maximum-likelihood bigram model and a "
        "maximum-likelihood unigram model with λ1 = 1 2 and λ2 =",
        654,
        680,
    )
    tail = blk("2, what is P(Sam|am)? Include <s> and </s> in your counts just like any other token.", 678, 703)
    assert len(_merge_continuations([head, tail])) == 1
