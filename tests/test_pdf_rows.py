"""Splitting lines into rows and cells (pdf_rows)."""

import pymupdf

from app.formats.pdf_types import Line, LineItem, PageDict, Span
from tests.helpers import _glyph


def _item(text: str, rect: pymupdf.Rect, spans: list[Span]) -> LineItem:
    """A text line whose spans all carry text, so the spans with text are all the spans of the line."""
    return (text, rect, spans, spans)


def _bbox(rect: pymupdf.Rect) -> tuple[float, float, float, float]:
    return (rect.x0, rect.y0, rect.x1, rect.y1)


def test_mid_line_capital_stays_in_the_sentence() -> None:
    """When a line is split into words, a capitalized word inside the line stays in the sentence. The next sentence
    after a sentence end, and markup, are still split off."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _line("The", 325.8, 341.6, 590.3, size=9),
        _line("extracted", 350.8, 388.4, 590.3, size=9),
        _line("plan", 397.6, 414.9, 590.3, size=9),
        _line("is", 424.1, 430.3, 590.3, size=9),
        _line("serialized", 439.4, 476.3, 590.3, size=9),
        _line("in", 485.5, 493.2, 590.3, size=9),
        _line("DXL", 502.3, 519.9, 590.3, size=9),
        _line("format", 529.0, 555.9, 590.3, size=9),
        _line("and", 316.8, 331.6, 600.7, size=9),
        _line("shipped.", 339.9, 370.9, 600.7, size=9),
    ]
    groups = _split_lines(items)
    assert groups[0] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert 8 in groups[1]

    ended = [
        _line("workloads.", 316.8, 448.3, 338.4, size=9),
        _line("It is distinguished from", 457.3, 555.9, 338.4, size=9),
    ]
    assert _split_lines(ended) == [[0], [1]]
    markup = [
        _line("<dxl:Ident", 53.8, 110.0, 80, size=8),
        _line('ColId="0" Name="a"/>', 116.0, 220.0, 80, size=8),
    ]
    assert _split_lines(markup) == [[0], [1]]


def test_a_full_row_split_after_a_sentence_stays_in_the_paragraph() -> None:
    """In a justified paragraph the wide space after a period makes PyMuPDF split a full line into two pieces: the
    first starts at the left edge of the paragraph, the second reaches the right edge, and the lines above and
    below are full. This is one line and does not start a new paragraph (third line of the abstract,
    "bles.  To realize")."""
    from app.formats.pdf_split_lines import _split_lines

    def line(text: str, x0: float, x1: float, y0: float) -> LineItem:
        return _item(text, pymupdf.Rect(x0, y0, x1, y0 + 10), [{"size": 9.0, "font": "Times-Roman", "text": text}])

    items = [
        line("processing time, especially for aggregation queries over large ta-", 54.0, 293.8, 190.6),
        line("bles.", 54.0, 70.8, 200.6),
        line("To realize this potential, the query optimizer must know", 78.5, 293.7, 200.6),
        line("how and when to exploit materialized views. This paper presents a", 54.0, 293.8, 210.6),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3]]
    # Two pieces on a line of their own may be a side label and body text, so they are still split at the sentence end
    assert _split_lines(items[1:3]) == [[0], [1]]


def test_a_number_wrapped_to_its_own_line_ends_the_sentence() -> None:
    """After a full line with an unfinished sentence, a number wrapped onto its own line ends the sentence and is not
    split off as a symbol line. A section number at the start of a line is followed by a heading and is still
    split off."""
    from app.formats.pdf_split_lines import _split_lines

    def line(text: str, x0: float, x1: float, y0: float) -> LineItem:
        return _item(text, pymupdf.Rect(x0, y0, x1, y0 + 12), [{"size": 10.0, "font": "Times-Roman", "text": text}])

    items = [
        line("perparameter that controls the penalty term; values typically range from 0.01 to", 160.2, 493.2, 121.9),
        line("0.1.", 160.2, 175.1, 133.8),
        line("As illustrated in Fig. 8.12, DPO uses gradient descent with this loss over the", 175.1, 493.2, 145.8),
    ]
    assert _split_lines(items) == [[0, 1], [2]]
    toc = [
        line("9.5 Fine-Tuning for Sequence Labeling: Named Entity Recognition . . . 238", 160.2, 493.2, 121.9),
        line("9.6", 160.2, 175.1, 133.8),
        line("Summary", 190.0, 230.0, 133.8),
    ]
    assert [0] in _split_lines(toc)


def _line(text: str, x0: float, x1: float, y0: float, size: float = 10) -> LineItem:
    return _item(text, pymupdf.Rect(x0, y0, x1, y0 + size + 2), [{"size": size}])


def test_lines_wrapping_around_a_figure_stay_in_one_paragraph() -> None:
    """In a paragraph set around a figure, the lines beside the figure end at the same x, far from the right edge of
    the block. Smilodon, starting with a capital below a line that fills the narrow column, is a wrapped line and
    no hard break. A line ending well before the right edge of the narrow column is still split as a hard break."""
    from app.formats.pdf_split_lines import _split_lines

    def line(text: str, x0: float, x1: float, y0: float) -> LineItem:
        return _item(text, pymupdf.Rect(x0, y0, x1, y0 + 12), [{"size": 10.0, "font": "Times-Roman", "text": text}])

    items = [
        line(
            "recognizable by its long canines. Five million years ago or so, a completely different",
            160.2,
            493.2,
            326.9,
        ),
        line("saber-tooth tiger called Thylacosmilus lived", 160.2, 341.5, 338.9),
        line("in Argentina and other parts of South Amer-", 160.2, 341.5, 350.8),
        line("ica. Thylacosmilus was a marsupial whereas", 160.2, 341.5, 362.8),
        line("Smilodon was a placental mammal, but Thy-", 160.2, 341.5, 374.8),
        line("lacosmilus had the same long upper canines", 160.2, 341.5, 386.7),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3, 4, 5]]
    items[3] = line("ica. Thylacosmilus was a marsupial.", 160.2, 300.0, 362.8)
    assert _split_lines(items) == [[0, 1, 2, 3], [4, 5]]
    # The first line beside the figure ends with a period, the line above fills the full width, and the line below
    # ends at the same x
    first = [
        line(
            "one of the earliest goals of computer language processing. In fact, speech processing", 160.2, 493.2, 298.1
        ),
        line("predates the computer by many decades!", 160.2, 324.5, 310.0),
        line("The first machine that recognized speech", 160.2, 324.5, 322.0),
        line("was a toy from the 1920s.", 160.2, 324.5, 333.9),
    ]
    assert _split_lines(first) == [[0, 1, 2, 3]]
    # The answers of the exercises take one line each and a few lines apart happen to end at the same x; the next
    # exercise starts a new paragraph
    answers = [
        line("The pronoun in the sentence above refers to", 82.2, 260.0, 294.8),
        line("Tom", 82.2, 102.1, 307.7),
        line("He said the manuscript was never published, so he wanted it back before anyone", 82.2, 513.1, 318.8),
        line("saw it.", 82.2, 112.1, 330.7),
        line("The pronoun in the sentence above refers to", 82.2, 260.0, 342.7),
        line("Manuscript", 82.2, 102.1, 354.6),
        line("Many years later the young man crossed the river to meet the bride he had chosen", 82.2, 513.1, 366.6),
    ]
    assert _split_lines(answers)[-1] == [6]


def test_a_table_row_name_is_cut_from_its_numbers_and_the_rows_around() -> None:
    """A table row name, the number cells in the same row and the group name above each become a paragraph of their
    own. Numbers close together are split as well; the number cells PyMuPDF merged into the row-name line are taken
    out of the line and stay on the original page like the numbers of the other rows."""
    from app.formats.pdf_rows import _drop_number_tails
    from app.formats.pdf_split_lines import _split_lines

    def span(text: str, x0: float, x1: float, y: float) -> Span:
        return {"text": text, "size": 9.0, "bbox": (x0, y, x1, y + 9), "origin": (x0, y + 7), "font": "Times"}

    def item(parts: list[tuple[str, float, float]], y: float) -> LineItem:
        spans = [span(text, x0, x1, y) for text, x0, x1 in parts]
        text = "".join(part[0] for part in parts)
        rect = pymupdf.Rect(parts[0][1], y, parts[-1][2], y + 9)
        return (text, rect, [one for one in spans if one["text"].strip()], spans)

    rows = [
        item([("Top Leaderboard Systems (Dec 10th, 2018)", 102.8, 259.5)], 328.6),
        item([("Human", 81.4, 107.8)], 338.6),
        item(
            [
                ("86.3", 198.5, 214.2),
                (" ", 214.2, 220.7),
                ("89.0", 220.7, 236.4),
                (" ", 236.4, 243.0),
                ("86.9", 243.0, 258.7),
                (" ", 258.7, 265.2),
                ("89.5", 265.2, 280.9),
            ],
            338.6,
        ),
        item([("#1 Single - MIR-MRC (F-Net)", 81.4, 192.0)], 348.6),
        item([("-", 204.8, 207.8)], 348.6),
        item([("74.8", 243.0, 258.7), (" ", 258.7, 265.2), ("78.0", 265.2, 280.9)], 348.6),
        item([("#2 Single - nlnet", 81.4, 140.9)], 358.5),
        item([("74.2", 243.0, 258.7), (" ", 258.7, 265.2), ("77.1", 265.2, 280.9)], 358.5),
        item(
            [
                ("BERT (Ens.+TriviaQA)", 81.4, 190.5),
                (" ", 190.5, 198.5),
                ("86.2", 198.5, 214.2),
                (" ", 214.2, 220.7),
                ("92.2", 220.7, 236.4),
            ],
            368.5,
        ),
    ]
    rows = _drop_number_tails(rows)
    assert rows[-1][0] == "BERT (Ens.+TriviaQA)"
    texts = [" ".join(rows[i][0] for i in group) for group in _split_lines(rows)]
    assert "Human" in texts and "#1 Single - MIR-MRC (F-Net)" in texts and "#2 Single - nlnet" in texts
    assert "Top Leaderboard Systems (Dec 10th, 2018)" in texts


def test_header_cells_over_number_columns_are_cut_apart() -> None:
    """The cells of a header row sit close together and were merged into one line. They are split when each piece is
    right above a column of numbers; the header of the leading row-name column need not line up. Splitting a
    justified body line would give half sentences, so it is not split."""
    from app.formats.pdf_rows import _cut_header_cells

    def span(text: str, x0: float, x1: float, y: float = 238.1) -> Span:
        return {"text": text, "size": 9.0, "bbox": (x0, y, x1, y + 9), "origin": (x0, y + 7), "font": "Times"}

    def item(spans: list[Span]) -> LineItem:
        text = "".join(one["text"] for one in spans)
        rect = pymupdf.Rect(spans[0]["bbox"][0], spans[0]["bbox"][1], spans[-1]["bbox"][2], spans[0]["bbox"][3])
        return (text, rect, [one for one in spans if one["text"].strip()], spans)

    columns = [411.8, 452.7, 501.6]
    header = item(
        [
            span("Fine-tune", 394.6, 429.0),
            span(" ", 429.0, 435.5),
            span("Fine-tune", 435.5, 469.9),
            span(" ", 469.9, 476.5),
            span("Feature-based", 476.5, 526.7),
        ]
    )
    items, cells = _cut_header_cells([header], columns)
    assert [one[0] for one in items] == ["Fine-tune", "Fine-tune", "Feature-based"] and cells == {0, 1, 2}
    stub = [
        item([span("System", 317.0, 343.4)]),
        item([span("Dev F1", 395.0, 428.6), span(" ", 428.6, 435.2), span("Test F1", 439.4, 466.0)]),
    ]
    items, cells = _cut_header_cells(stub, columns)
    assert [one[0] for one in items] == ["System", "Dev F1", "Test F1"]
    prose = item(
        [
            span("Second,", 380.0, 406.8),
            span("there", 412.6, 434.4),
            span("are", 439.6, 452.9),
            span("major", 458.1, 483.6),
            span("computational", 488.7, 551.2),
        ]
    )
    items, cells = _cut_header_cells([prose], [393.4, 470.0])
    assert len(items) == 1 and not cells


def test_a_numbered_lemma_and_a_displayed_formula_start_their_own_paragraphs() -> None:
    """A numbered lemma label starts a new paragraph, also when the previous line ends in a formula. A line indented
    on both sides, with only math fonts and a relation symbol, is a displayed formula and is not joined even when
    the previous line ends with a comma. The summation piece split out of a fraction does not count."""
    from app.formats.pdf_split_lines import _split_lines

    def line(text: str, x0: float, x1: float, y: float, font: str = "NimbusRomNo9L-Regu", flags: int = 0) -> LineItem:
        return _item(
            text,
            pymupdf.Rect(x0, y, x1, y + 11),
            [{"text": text, "size": 9.7, "font": font, "flags": flags, "bbox": (x0, y, x1, y + 11)}],
        )

    lemma = [
        line(
            "that contains the ith dimension of the gradients over all iterations till t, g1:t,i = [g1,i, g2,i]",
            105.0,
            485.6,
            300.0,
        ),
        _item(
            "Lemma 10.3. Let gt = ∇ft(θt) and g1:t be defined as above and bounded,",
            pymupdf.Rect(105.0, 314.0, 490.3, 325.0),
            [
                {
                    "text": "Lemma 10.3.",
                    "size": 9.7,
                    "font": "NimbusRomNo9L-Medi",
                    "flags": 16,
                    "bbox": (105, 314, 160, 325),
                },
                {
                    "text": " Let gt = ∇ft(θt) and g1:t be defined as above and bounded,",
                    "size": 9.7,
                    "font": "NimbusRomNo9L-Regu",
                    "flags": 0,
                    "bbox": (160, 314, 490.3, 325),
                },
            ],
        ),
    ]
    assert _split_lines(lemma) == [[0], [1]]
    display = [
        line("Lemma 10.2. If a function f : Rd → R is convex, then for all x, y ∈ Rd,", 105.0, 391.0, 209.9),
        line("f(y) ≥ f(x) + ∇f(x)T (y − x)", 235.1, 360.2, 226.0, font="CMMI10"),
        line(
            "The above lemma can be used to upper bound the regret and our proof for the main theorem is",
            105.0,
            490.3,
            248.0,
        ),
    ]
    assert _split_lines(display) == [[0], [1], [2]]


def test_wrapped_letter_period_stays_in_the_paragraph() -> None:
    """When the full previous line has no sentence end, Q. on the next line is a wrapped line and no list item. a.
    after a sentence end is still split off."""
    from app.formats.pdf_split_lines import _split_lines

    wrapped = [
        _line("corresponding to QA to answer", 72, 540, 414, size=12),
        _line("Q. The cost of answering Q is a function", 72, 540, 427, size=12),
        _line("the simplest cost-model:", 72, 187, 441, size=12),
    ]
    assert _split_lines(wrapped) == [[0, 1, 2]]
    listed = [
        _line("We can now ask the following questions.", 72, 540, 100, size=12),
        _line("a. How many views must we materialize?", 85, 430, 116, size=12),
    ]
    assert _split_lines(listed) == [[0], [1]]
    paren = [
        _line("Note that since the top view is in S.", 72, 500, 80, size=12),
        _line("(a) Let u be the view of least cost", 105, 500, 96, size=12),
        _line("S, there must be at least one view.", 124, 500, 110, size=12),
        _line("(b) If C(v) is smaller then stop.", 104, 480, 126, size=12),
    ]
    assert _split_lines(paren) == [[0], [1, 2], [3]]


def test_a_dash_after_a_full_line_continues_the_sentence() -> None:
    """When the full previous line has an unfinished sentence, a dash at the start of the next line belongs to the
    sentence and is no bullet. The same holds when the previous line is a monospace URL one size larger. A dash
    below an item that starts with a dash is the next item at the same level; an indented dash or one in another
    font size is a bullet one level down."""
    from app.formats.pdf_split_lines import _split_lines

    wrapped = [
        _line(
            "performance of 80%. Most notably, the model attains a score of 143.7/150 in AMC 12 2024",
            70.5,
            524.4,
            366.6,
            size=11,
        ),
        _line("(https://artofproblemsolving.com/wiki/index.php/2024_AMC_12B_Problems)", 70.5, 525.5, 379.3, size=12),
        _line(
            "- a performance that, when combined with its AIME results, yields a score exceeding the",
            69.4,
            524.4,
            393.7,
            size=11,
        ),
    ]
    assert _split_lines(wrapped) == [[0, 1, 2]]
    items = [
        _line("– Effect of Number of Training Steps; and", 340.5, 525.5, 168.1, size=10.9),
        _line("– Ablation for Different Masking Proce-", 340.5, 525.5, 183.8, size=10.9),
    ]
    assert _split_lines(items) == [[0], [1]]
    nested = [
        _line("• A relational database layer for Apache HBase", 43.2, 581.3, 101.5, size=25),
        _line("– Query engine", 79.2, 581.3, 127.5, size=22),
    ]
    assert _split_lines(nested) == [[0], [1]]


def test_justified_rows_are_not_a_centered_stack() -> None:
    """Full body lines whose left edges differ by a point or two have almost the same width and are no stack of
    centered lines. Author names, affiliations and emails vary in width."""
    from app.formats.pdf_split_lines import _centered_rows

    paragraph = [
        pymupdf.Rect(x0, y, x1, y + 11)
        for x0, x1, y in ((70.9, 524.4, 285.5), (70.4, 524.4, 326.0), (70.5, 525.5, 379.3), (69.4, 524.4, 393.7))
    ]
    assert not _centered_rows(paragraph, 11.0)
    authors = [
        pymupdf.Rect(x0, y, x1, y + 10)
        for x0, x1, y in ((424.3, 497.2, 233.5), (424.5, 492.9, 246.0), (422.1, 495.3, 257.2))
    ]
    assert _centered_rows(authors, 10.0)


def test_the_tail_of_a_wrapped_url_does_not_end_the_verbatim_text() -> None:
    """The tail of a wrapped URL lands at the start of the next line and body text follows it. The line start is still
    monospace and has not switched back to the body font, so the paragraph is not split."""
    from app.formats.pdf_split_lines import _split_lines

    def span(text: str, mono: bool) -> Span:
        return {
            "text": text,
            "size": 12.0 if mono else 11.0,
            "font": "LMMono10-Regular" if mono else "URWPalladioL-Roma",
            "flags": 0,
        }

    lines = [
        _item(
            "Olympiad https://artofproblemsolving.com/wiki/index.php/AMC_historical_",
            pymupdf.Rect(70.9, 420.0, 525.0, 432.0),
            [span("Olympiad ", False), span("https://artofproblemsolving.com/wiki/index.php/AMC_historical_", True)],
        ),
        _item(
            "results?srsltid=AfmBOoqQ6pQic5NCan_NX1wYgr-aoHgJ33hsq7KSekF-rUwY8TBaBao",
            pymupdf.Rect(70.9, 433.5, 524.4, 445.5),
            [span("results?srsltid=AfmBOoqQ6pQic5NCan_NX1wYgr-aoHgJ33hsq7KSekF-rUwY8TBaBao", True)],
        ),
        _item(
            "1). This performance positions DeepSeek-R1 among the nation’s top-tier high school students.",
            pymupdf.Rect(70.6, 447.1, 524.0, 459.1),
            [span("1).", True), span(" This performance positions DeepSeek-R1 among the nation’s top-tier", False)],
        ),
    ]
    assert _split_lines(lines) == [[0, 1, 2]]


def test_an_equals_sign_at_the_end_of_a_list_item_line_keeps_the_item_whole() -> None:
    """Justification stretches the space before an equals sign at the line end to more than one em; the equals sign
    still joins this line, and the next line indented to the right of the entry number continues the entry."""
    from app.formats.pdf_split_lines import _split_lines

    def piece(text: str, x0: float, x1: float, y0: float, font: str = "Times-Roman") -> LineItem:
        return _item(text, pymupdf.Rect(x0, y0, x1, y0 + 10), [{"size": 10.0, "font": font, "text": text}])

    items = [
        piece("1. Intractability:", 131.4, 199.1, 474.5),
        _item(
            "the case where the integral of the marginal likelihood p(x)",
            pymupdf.Rect(210.4, 474.5, 485.9, 485.3),
            [
                {"size": 10.0, "font": "Times-Roman", "text": "the case where the integral of the marginal likelihood"},
                {"size": 10.0, "font": "CMR10", "text": "p(x)"},
            ],
        ),
        piece("=", 496.3, 504.0, 474.5, font="CMR10"),
        piece("R", 143.9, 148.6, 484.6, font="CMEX10"),
        piece(
            "p(z)p(x|z) dz is intractable (so we cannot evaluate or differentiate the marginal like-",
            152.2,
            504.0,
            485.3,
        ),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3, 4]]
    # The equals sign does not reach the right edge, so the line is not full and a piece more than one em away is
    # not joined
    loose = [items[1], items[2], piece("where the prior is fixed and the posterior is not", 131.4, 540.0, 500.0)]
    assert _split_lines(loose)[0] == [0]


def test_pdf_left_jump_keeps_full_width_wrap() -> None:
    """An indented first line with both lines reaching the right margin is an automatic wrap and is not split into two
    paragraphs."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _line("started", 120, 400, 10),
        _line("the effort continues here", 108, 400, 24),
    ]
    assert _split_lines(items) == [[0, 1]]


def test_pdf_short_indent_stays_one_segment() -> None:
    """A short indented first line followed by a longer continuation line is still an automatic wrap and is not split.
    A Chinese continuation line is not split just because it looks like a new sentence."""
    from app.formats.pdf_split_lines import _split_lines

    latin = [
        _line("Hello", 64, 140, 10, size=12),
        _line("And the sentence continues much further", 40, 400, 26, size=12),
    ]
    cjk = [
        _line("开始", 64, 120, 10, size=12),
        _line("续行比第一行长很多", 40, 360, 26, size=12),
    ]
    assert _split_lines(latin) == [[0, 1]]
    assert _split_lines(cjk) == [[0, 1]]


def test_pdf_list_continuation_keeps_capital() -> None:
    """A continuation line of a list item belongs to the item even when it starts with a capital, and does not start a
    new paragraph."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _line("- First item", 40, 120, 10),
        _line("Continues Here", 55, 220, 24),
    ]
    assert _split_lines(items) == [[0, 1]]


def test_pdf_cjk_indent_wrap_stays_one_segment() -> None:
    """A 2em first-line indent with a continuation line that is not full is still one paragraph."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _line("第一行撑满右边距", 64, 400, 10, size=12),
        _line("续行", 40, 80, 26, size=12),
    ]
    assert _split_lines(items) == [[0, 1]]


def test_pdf_left_jump_splits_stacked_labels() -> None:
    """An upper line that is not full and a lower line jumping clearly to the left are parallel short labels and are
    split."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _line("Fig", 180, 210, 10),
        _line("A longer caption here", 40, 300, 24),
    ]
    assert _split_lines(items) == [[0], [1]]


def test_pdf_far_apart_labels_on_one_row_split() -> None:
    """Two pieces on the same line several em apart are side-by-side labels (a slide caption and a node in the
    figure) and are split."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _line("Candidate 1:", 61, 154, 119, size=12),
        _line("project", 516, 553, 118, size=12),
    ]
    assert _split_lines(items) == [[0], [1]]


def test_pdf_cjk_straight_quote_ends_sentence() -> None:
    """An ASCII straight quote at a sentence end counts as a sentence end like a curly quote; a semicolon joins
    clauses and does not split the paragraph."""
    from app.formats.pdf_split_lines import _split_lines

    def pair(end: str) -> list[LineItem]:
        return [
            _line(f"完整一句。{end}", 40, 140, 10, size=12),
            _line("而这是一行明显更长的内容用来撑满整行的宽度", 40, 400, 26, size=12),
        ]

    assert _split_lines(pair('"')) == [[0], [1]]
    assert _split_lines(pair("'")) == [[0], [1]]
    assert _split_lines(pair("”")) == [[0], [1]]
    assert _split_lines(pair("；")) == [[0, 1]]


def test_one_query_per_line_in_one_block_is_cut_into_lines() -> None:
    """PyMuPDF groups queries of one sentence per line into one block. Every line breaks early (the first word of the
    next line would fit), so the block is split by line. A full paragraph stays one paragraph."""
    from app.formats.pdf_rows import _cut_line_list

    def item(text: str, y: float, x1: float) -> LineItem:
        rect = pymupdf.Rect(167.1, y, x1, y + 10.8)
        return _item(text, rect, [{"text": text, "size": 9.0, "bbox": _bbox(rect)}])

    queries = [
        item("can you tell me about any good cantonese restaurants close by", 356.2, 390.1),
        item("tell me about chez panisse", 367.1, 261.2),
        item("i'm looking for a good place to eat breakfast", 378.1, 326.4),
        item("when is caffe venezia open during the day", 389.1, 317.9),
    ]
    assert _cut_line_list([0, 1, 2, 3], queries, [item[1] for item in queries]) == [[0], [1], [2], [3]]
    # Paragraphs around figures or inside tables are narrower than the column and wrap raggedly; the first word of
    # the next line never fits on the longest line
    paragraph = [
        item("Policies, processes,", 100, 240),
        item("procedures, and", 111, 225),
        item("practices across the", 122, 238),
        item("organization related", 133, 236),
    ]
    assert _cut_line_list([0, 1, 2, 3], paragraph, [item[1] for item in paragraph]) == [[0, 1, 2, 3]]
    # Lines around a figure on the right stop at the figure edge, and the line below fills the whole column
    around = [
        item("the first full-sentence speech synthesizer, shown partially to the right. His", 100, 475),
        item("device consisted of a bellows to simulate the lungs, a rub-", 111, 380),
        item("ber mouthpiece and a nose aperture, a", 122, 380.5),
        item("reed to simulate the vocal folds, var-", 133, 379.6),
    ]
    assert _cut_line_list([0, 1, 2, 3], around, [item[1] for item in around]) == [[0, 1, 2, 3]]


def test_rows_of_a_table_or_a_list_with_one_opening_are_cut_into_lines() -> None:
    """A line split into several pieces is first joined into one line. When every line starts a new column at the
    same x (row name and features), or every line starts with the same word (feature templates), the block is split
    by line even if the lines have similar lengths."""
    from app.formats.pdf_rows import _cut_line_list

    def item(text: str, x0: float, y: float, x1: float, size: float = 9.96) -> LineItem:
        rect = pymupdf.Rect(x0, y, x1, y + 11.9)
        return _item(text, rect, [{"text": text, "size": size, "bbox": _bbox(rect)}])

    def row(parts: list[tuple[str, float, float]], y: float) -> LineItem:
        spans: list[Span] = [{"text": text, "size": 9.96, "bbox": (x0, y, x1, y + 11.9)} for text, x0, x1 in parts]
        return _item(" ".join(p[0] for p in parts), pymupdf.Rect(parts[0][1], y, parts[-1][2], y + 11.9), spans)

    table = [
        item("hen", 178.1, 414.7, 192.5),
        row([("+female,", 212.4, 250.1), ("+chicken,", 259.5, 302.9), ("+adult", 311.8, 343.2)], 416.0),
        row(
            [
                ("rooster", 178.1, 209.4),
                ("-female,", 212.4, 250.1),
                ("+chicken,", 259.5, 302.9),
                ("+adult", 311.8, 343.2),
            ],
            426.7,
        ),
        item("chick", 178.1, 438.6, 198.8),
        row([("+chicken,", 212.4, 255.8), ("-adult", 264.7, 296.1)], 439.9),
    ]
    rects = [item[1] for item in table]
    assert _cut_line_list([0, 1, 2, 3, 4], table, rects) == [[0, 1], [2], [3, 4]]
    templates = [
        item("xi contains a particular prefix (perhaps from all prefixes of length ≤2)", 185.7, 589.9, 465.9),
        item("xi contains a particular suffix (perhaps from all suffixes of length ≤2)", 185.7, 601.9, 464.3),
        item("xi’s word shape", 185.7, 613.8, 247.3),
        item("xi’s short word shape", 185.7, 625.8, 269.8),
    ]
    rects = [item[1] for item in templates]
    assert _cut_line_list([0, 1, 2, 3], templates, rects) == [[0], [1], [2], [3]]


def test_smaller_body_under_a_heading_splits() -> None:
    """A 12pt heading and a 9pt lead paragraph are often in the same dict block. The left edge of the lead only makes
    room for the section number and must not be taken as an indent that keeps it in the heading."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _line("4.", 316, 326, 674, size=12),
        _line("QUERY OPTIMIZATION", 338, 475, 674, size=12),
        _line("We describe Orca's optimization workflow in Section 4.1.", 326, 556, 689, size=9),
        _line("We then show how the process can be conducted in parallel.", 317, 556, 700, size=9),
    ]
    assert _split_lines(items) == [[0, 1], [2, 3]]
    # A short script line has too few characters and is not split even though it is smaller. A digits-only line is
    # split as a symbol line anyway, so an ordinal suffix is used here.
    script = [
        _line("the i", 40, 80, 10, size=12),
        _line("th", 82, 94, 6, size=7),
    ]
    assert _split_lines(script) == [[0, 1]]


def test_affiliation_under_an_author_name_splits() -> None:
    """On a title page, an affiliation one size smaller below an author name forms a line of its own and becomes a
    separate paragraph even with only two words. When an asterisk at the line start is one size larger than the
    text after it, the comparison uses the size of the majority of characters, and a continuation line of the same
    size on the next line is not split."""
    from app.formats.pdf_split_lines import _split_lines

    author = [
        _line("Daniel Jurafsky", 247, 355, 333.4, size=17.2),
        _line("Stanford University", 245, 357, 353.6, size=14.3),
    ]
    assert _split_lines(author) == [[0], [1]]

    def item(spans: list[tuple[str, float]], x0: float, x1: float, y0: float) -> LineItem:
        text = "".join(text for text, _ in spans)
        return _item(text, pymupdf.Rect(x0, y0, x1, y0 + 14), [{"text": text, "size": size} for text, size in spans])

    note = [
        item([("*", 15), ("also what standalone", 12), (" ", 15)], 61, 186, 155),
        item([("Phoenix compiler ", 12)], 61, 157, 173),
        item([("would generate.", 12)], 61, 148, 187),
    ]
    assert _split_lines(note) == [[0, 1, 2]]


def test_section_number_stays_with_heading() -> None:
    """The dict lines '2.2' + 'Training' must not be split: once the digits-only line is split off as a symbol line,
    the number is filtered out and not masked (the original stays visible), the heading is translated alone, and
    the baselines of the two do not line up."""
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _item("2.2", pymupdf.Rect(108, 84, 120, 94), [{"text": "2.2", "size": 10.0}]),
        _item("Training", pymupdf.Rect(130, 84, 167, 94), [{"text": "Training", "size": 10.0}]),
    ]
    assert _split_lines(items) == [[0, 1]]
    # The page number and body text are not on the same visual line (no vertical overlap), so they are split as
    # usual
    items2 = [
        _item("3", pymupdf.Rect(300, 760, 308, 770), [{"text": "3", "size": 10.0}]),
        _item(
            "Next paragraph starts here",
            pymupdf.Rect(108, 700, 500, 712),
            [{"text": "Next paragraph starts here", "size": 10.0}],
        ),
    ]
    assert _split_lines(items2) == [[0], [1]]


def test_a_chapter_number_across_a_rule_is_not_joined_to_the_title() -> None:
    """A two-digit chapter number at the chapter start sits close to the heading, so word spacing would join them into
    one sentence and rewriting the line would cover the vertical bar between them. A vertical bar between them
    splits them."""
    from app.formats.pdf_rows import _cut_at_posts
    from app.formats.pdf_split_lines import _split_lines

    items = [
        _item("10", pymupdf.Rect(98.4, 132.0, 158.2, 204.1), [{"text": "10", "size": 59.8}]),
        _item(
            "Interpretability",
            pymupdf.Rect(167.3, 137.3, 331.9, 169.6),
            [{"text": "Interpretability", "size": 24.8}],
        ),
    ]
    post = pymupdf.Rect(161.2, 110.8, 162.4, 200.8)
    groups = [part for group in _split_lines(items) for part in _cut_at_posts(group, items, [post])]
    assert groups == [[0], [1]]
    # The vertical bar is not between the two pieces (a table border), so they are joined as usual
    assert _cut_at_posts([0, 1], items, [pymupdf.Rect(90, 110, 91, 200)]) == [[0, 1]]


def test_a_line_split_into_two_blocks_is_put_back_together() -> None:
    """When PyMuPDF cuts a line at a superscript and puts the second half into another block, it is moved back after
    the first half."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(x0: float, x1: float, y0: float, y1: float, text: str) -> Line:
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [{"text": text, "size": 10.0}]}

    head = line(100, 400, 150, 162, "All operations are element-wise. With β")
    tail = line(398, 440, 150, 163, "1 and β")
    nxt = line(100, 300, 162, 174, "we denote the power t.")
    other = line(100, 400, 200, 212, "Require: a step size")
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (100, 150, 400, 162), "lines": [head]},
            {"type": 0, "bbox": (100, 150, 440, 212), "lines": [tail, nxt, other]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [head, tail]
    assert data["blocks"][1]["lines"] == [nxt, other]


def test_a_two_column_gutter_is_not_a_split_line() -> None:
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(x0: float, x1: float, text: str) -> Line:
        return {"bbox": (x0, 150, x1, 162), "dir": (1, 0), "spans": [{"text": text, "size": 10.0}]}

    left, right = line(60, 290, "left column text"), line(305, 540, "right column text")
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (60, 150, 290, 162), "lines": [left]},
            {"type": 0, "bbox": (305, 150, 540, 162), "lines": [right]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [left] and data["blocks"][1]["lines"] == [right]


def test_a_narrow_gutter_next_to_math_or_a_section_number_is_not_a_split_line() -> None:
    """With a column gap narrower than one em, a right-column line starting with a formula or a section number is not
    joined after the left-column paragraph either."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(
        x0: float,
        x1: float,
        y0: float,
        text: str,
        font: str = "Times-Roman",
        size: float = 10.0,
        y1: float | None = None,
    ) -> Line:
        box = (x0, y0, x1, y1 if y1 is not None else y0 + 13)
        return {"bbox": box, "dir": (1, 0), "spans": [{"text": text, "font": font, "size": size, "bbox": box}]}

    para = [
        line(72, 301.3, 638 + 12 * i, "ducing internal covariate shift, and in doing so dramati-") for i in range(4)
    ]
    formula = line(311, 540, 650, "x = u + b", "CMMI10")
    rest = line(311, 540, 662, "the training set, and E[x] is the mean")
    number = line(311, 326, 659, "3.2", "Times-Bold", 14.3)
    title = line(337.9, 540, 659, "Batch-Normalized Convolutional Net-", "Times-Bold", 14.3)
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (72, 638, 301.3, 687), "lines": list(para)},
            {"type": 0, "bbox": (311, 650, 540, 675), "lines": [formula, rest]},
            {"type": 0, "bbox": (311, 659, 540, 672), "lines": [number, title]},
        ]
    }
    _rejoin_split_lines(data)
    assert [block["lines"] for block in data["blocks"]] == [para, [formula, rest], [number, title]]

    # A tall right-column line overlaps two lines of the left-column paragraph, leaving only one left line to
    # compare: still two columns
    first = line(72, 301.0, 325.6, "where the BN transform is applied independently to each")
    second = line(72, 301.1, 337.5, "dimension of x = Wu, with a separate pair of learned")
    last = line(72, 219.2, 345.9, "parameters per dimension.", y1=362.8)
    tall = line(311, 539.0, 329.1, "JJT . Thus, JJT = I, and so all singular values of J", "CMMI10", y1=346.0)
    data = {
        "blocks": [
            {"type": 0, "bbox": (72, 325.6, 301.1, 362.8), "lines": [first, second, last]},
            {"type": 0, "bbox": (311, 329.1, 539, 346.0), "lines": [tall]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == [tall]

    # Numerator and denominator are vertically aligned and both end before the summation sign: not a column
    numerator = line(414.3, 418.2, 82.5, "1", "CMR7", 7.0, 89.4)
    denominator = line(413.5, 419.0, 89.8, "L", "CMMI7", 7.0, 96.8)
    total = line(421.8, 437.8, 81.4, "PL", "CMEX10")
    data = {
        "blocks": [
            {"type": 0, "bbox": (413.5, 82.5, 419.0, 102.8), "lines": [numerator, denominator]},
            {"type": 0, "bbox": (421.8, 81.4, 437.8, 94.4), "lines": [total]},
        ]
    }
    _rejoin_split_lines(data)
    assert total in data["blocks"][0]["lines"]

    # A large heading at the column top shares a baseline with the left-column paragraph, leaving one full line to
    # compare: the font size is much larger, so it is not the second half of that line
    top = line(72, 301.1, 35.5, "not have to readjust to compensate for the change in the")
    tail = line(72, 138.8, 47.5, "distribution of x.")
    heading: list[Line] = [
        line(311, 318.2, 31, "2", "Times-Bold", 14.3, 50.9),
        line(332.5, 385.3, 31, "Towards", "Times-Bold", 14.3, 50.9),
    ]
    data = {
        "blocks": [
            {"type": 0, "bbox": (72, 35.5, 301.1, 60.8), "lines": [top, tail]},
            {"type": 0, "bbox": (311, 31, 385.3, 50.9), "lines": list(heading)},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == heading


def test_a_formula_whose_font_box_reaches_the_next_line_stays_on_its_own_line() -> None:
    """The font box of ϵ reaches into the next line. Measured by the font box, the denominator is on the same row
    as the next line, so it would be attached after that line and the numerator would follow it."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    Box = tuple[float, float, float, float]

    def span(text: str, font: str, size: float, box: Box, ink: Box | None = None) -> Span:
        item: Span = {"text": text, "font": font, "size": size, "bbox": box, "origin": (box[0], box[3] - 2)}
        if ink is not None:
            item["ink"] = ink
        return item

    def line(*spans: Span) -> Line:
        box = pymupdf.Rect(spans[0]["bbox"])
        for item in spans[1:]:
            box |= pymupdf.Rect(item["bbox"])
        return {"bbox": _bbox(box), "dir": (1, 0), "spans": list(spans)}

    lead = line(span("Equation (10) reveals that we must predict (", "Times-Roman", 10.0, (108, 187.2, 305, 199.9)))
    minus = line(
        span("x", "CMBX10", 10.0, (307.6, 187.3, 314, 206.1), (307.6, 190.5, 314, 197.2)),
        span(" −", "CMSY10", 10.0, (314, 187.3, 327.1, 206.1), (318, 192.8, 327.1, 193.6)),
    )
    numerator = line(span("β", "CMMI7", 7.0, (339.2, 185.3, 344, 192.8), (339.2, 185.6, 344, 191.9)))
    denominator = line(
        span("√", "CMSY7", 7.0, (330.5, 187.2, 337, 205.7), (330.5, 193.3, 337, 200.0)),
        span("α", "CMMI7", 7.0, (337, 187.2, 341, 205.7), (337, 195.4, 341, 199.8)),
        span(" ϵ", "CMMIB10", 10.0, (355.2, 187.2, 361.7, 205.7), (355.2, 194.0, 361.7, 198.4)),
    )
    given = line(span("given x. Since x is available as", "Times-Roman", 10.0, (370.2, 187.4, 504, 198.9)))
    nxt = line(
        span("input to the model, we may choose the parameterization", "Times-Roman", 10.0, (108, 201.7, 331.3, 211.7))
    )
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (108, 187.2, 305, 199.9), "lines": [lead]},
            {"type": 0, "bbox": (307.6, 185.3, 361.7, 206.1), "lines": [minus, numerator, denominator]},
            {"type": 0, "bbox": (108, 187.4, 504, 211.7), "lines": [given, nxt]},
        ]
    }
    _rejoin_split_lines(data)
    host = next(block for block in data["blocks"] if lead in block["lines"])
    assert denominator in host["lines"] and numerator in host["lines"]
    assert nxt not in host["lines"]


def test_a_split_line_goes_to_its_nearest_left_neighbour() -> None:
    """Two blocks both have a left half that could be joined: the piece joins the nearest one and does not move back
    and forth between the two blocks."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(x0: float, x1: float, y0: float, text: str, font: str = "CMMI10") -> Line:
        box = (x0, y0, x1, y0 + 10)
        return {"bbox": box, "dir": (1, 0), "spans": [{"text": text, "font": font, "size": 10.0, "bbox": box}]}

    # The two left halves are vertically offset, each shares the line height of the right half, and they are not on
    # the same line as each other
    far = line(72, 262, 96, "the rule is", "Times-Roman")
    near = line(250, 264, 104, "c =")
    piece = line(265, 300, 100, "d − e")
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (72, 96, 262, 106), "lines": [far]},
            {"type": 0, "bbox": (250, 104, 264, 114), "lines": [near]},
            {"type": 0, "bbox": (265, 100, 300, 110), "lines": [piece]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == [near, piece]


def test_a_justified_heading_split_word_by_word_stays_one_paragraph() -> None:
    """A justified heading in a narrow column has word spacing stretched to two em, so PyMuPDF makes one line per word.
    It is recognized as one paragraph by whole line."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, x1: float, y0: float) -> LineItem:
        span: Span = {"text": text, "font": "Times-Bold", "size": 14.3}
        return _item(text, pymupdf.Rect(x0, y0, x1, y0 + 20), [span])

    items = [
        item("2", 311, 318, 31),
        item("Towards", 333, 385, 31),
        item("Reducing", 408, 466, 31),
        item("Internal", 490, 540, 31),
        item("Covariate Shift", 333, 426, 49),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3, 4]]


def test_table_cells_on_one_baseline_stay_apart() -> None:
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, x1: float) -> LineItem:
        span: Span = {"text": text, "font": "Times-Roman", "size": 9.0}
        return _item(text, pymupdf.Rect(x0, 100, x1, 110), [span])

    items = [item("Training", 128, 158), item("Inference needed", 184, 246), item("No difficulties", 437, 489)]
    assert len(_split_lines(items)) == 3


def test_fraction_pieces_rejoin_their_line_with_less_overlap() -> None:
    """Denominators and large braces pull the box down, so two pieces of the same line overlap vertically by only 30 to
    40 percent."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(x0: float, x1: float, y0: float, y1: float, text: str) -> Line:
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [{"text": text, "size": 10.0}]}

    root = line(188, 202.4, 259.0, 271.2, "√αt")
    paren = line(205.8, 211.7, 253.9, 263.8, "(")
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (108, 258, 202, 274), "lines": [line(108, 184, 258, 274, "to compute x ="), root]},
            {"type": 0, "bbox": (205, 253, 212, 264), "lines": [paren]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"][-1] is paren


def test_math_line_box_follows_glyph_ink() -> None:
    """A CMSY box from the font extent can be two lines high. Line boxes and cells are computed from ink; the radical
    has its origin at the top, so it is measured by ink as well."""
    from app.formats.pdf_math import _line_box
    from app.formats.pdf_rows import _text_box

    arrow: Span = {
        "text": "←",
        "font": "CMSY10",
        "size": 9.7,
        "bbox": (130.2, 218.8, 139.9, 235.6),
        "origin": (130.2, 226.3),
        "ink": (130.2, 221.3, 139.9, 226.4),
    }
    word: Span = {
        "text": "(Initialize",
        "font": "NimbusRomNo9L-Regu",
        "size": 9.7,
        "bbox": (147.4, 217.3, 194.8, 229.0),
        "origin": (147.4, 226.3),
        "ink": (147.4, 219.7, 194.8, 228.0),
    }
    assert [round(v, 1) for v in _bbox(_text_box(arrow))] == [130.2, 220.7, 139.9, 227.0]
    assert _text_box(word) == pymupdf.Rect(147.4, 217.3, 194.8, 229.0)
    line: Line = {"bbox": (130.2, 217.3, 194.8, 235.6), "spans": [arrow, word]}
    assert [round(v, 1) for v in _bbox(_line_box(line))] == [130.2, 217.3, 194.8, 229.0]


def test_formula_pieces_touching_on_one_line_stay_one_paragraph() -> None:
    """Fractions and large braces split a formula line into several pieces. A piece of symbols only must not cut the
    sentence apart."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, x1: float, y0: float, y1: float, *spans: tuple[str, str, float]) -> LineItem:
        spans = spans or ((text, "CMMI10", 10.0),)
        return _item(text, pymupdf.Rect(x0, y0, x1, y1), [{"text": t, "font": f, "size": z} for t, f, z in spans])

    items = [
        item("to compute x =", 108, 184, 258, 268, ("to compute", "NimbusRomNo9L-Regu", 9.9), (" x =", "CMMI10", 10.0)),
        item("1", 193.4, 197.4, 256.4, 262.2, ("1", "CMR7", 7.0)),
        item("√αt", 187.9, 202.4, 263.5, 271.1, ("√αt", "CMSY7", 7.0)),
        item("(", 205.8, 211.7, 253.9, 263.8, ("(", "CMEX10", 10.0)),
        item("xt −", 211.8, 231.3, 260.5, 267.7),
        item(
            "+σz, where the sampling",
            303,
            504,
            258,
            268,
            ("+σz", "CMMI10", 10.0),
            (", where the sampling", "NimbusRomNo9L-Regu", 9.9),
        ),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3, 4, 5]]


def test_an_inline_fraction_rejoins_its_line_in_reading_order() -> None:
    """g(x) = 1/(1+exp(−x)): the denominator is in another block, farther from the equals sign than a word space, and
    the numerator comes before the denominator. After moving them back, they are ordered left to right so the
    denominator can join the equals sign."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(x0: float, x1: float, y0: float, y1: float, *spans: tuple[str, str]) -> Line:
        # CMR inside Times body text is already marked as formula during extraction
        return {
            "bbox": (x0, y0, x1, y1),
            "dir": (1, 0),
            "spans": [
                {"text": text, "font": font, "size": 10.0, "math": font.startswith("CM")} for text, font in spans
            ],
        }

    host = line(72, 219.4, 136, 149.3, ("rameters to be learned, and", "Times-Roman"), ("g(x) =", "CMR10"))
    numerator = line(243.2, 247.2, 137.3, 143.2, ("1", "CMR7"))
    denominator = line(225.6, 301, 136, 151.7, ("1+exp(−x)", "CMR10"), (". As", "Times-Roman"))
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (72, 136, 247, 149), "lines": [host, numerator]},
            {"type": 0, "bbox": (225, 136, 301, 152), "lines": [denominator]},
        ]
    }
    _rejoin_split_lines(data)
    assert sorted(map(id, data["blocks"][0]["lines"])) == sorted(map(id, [host, numerator, denominator]))


def test_pieces_of_one_row_are_read_left_to_right() -> None:
    """When the numerator comes before the denominator, they are reordered left to right; two lines one above the other
    are not treated as one line."""
    from app.formats.pdf_rows import _left_to_right

    def item(text: str, x0: float, y0: float, x1: float, y1: float) -> LineItem:
        return _item(text, pymupdf.Rect(x0, y0, x1, y1), [])

    host, numerator, denominator = (
        item("g(x) =", 72, 136, 219, 149),
        item("1", 243, 137, 247, 143),
        item("1+exp(−x)", 225, 136, 301, 151),
    )
    nxt = item("increases, g′(x) tends to zero", 72, 149.7, 301, 163)
    assert _left_to_right([host, numerator, denominator, nxt]) == [host, denominator, numerator, nxt]


def test_plain_words_a_wide_gap_apart_are_not_rejoined() -> None:
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(x0: float, x1: float, text: str) -> Line:
        return {
            "bbox": (x0, 136, x1, 149),
            "dir": (1, 0),
            "spans": [{"text": text, "font": "Times-Roman", "size": 10.0}],
        }

    left, right = line(72, 219, "learned, and the"), line(227, 301, "output of the layer")
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (72, 136, 219, 149), "lines": [left]},
            {"type": 0, "bbox": (227, 136, 301, 149), "lines": [right]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][1]["lines"] == [right]


def test_a_lone_numerator_goes_back_to_its_fraction() -> None:
    """A numerator left alone in a block is moved next to the formula line right below it. Body text on the previous
    line also covers it horizontally, but that does not count."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(
        x0: float, x1: float, y0: float, y1: float, *spans: tuple[str, str, float, tuple[float, float, float, float]]
    ) -> Line:
        return {
            "bbox": (x0, y0, x1, y1),
            "dir": (1, 0),
            "spans": [
                {"text": text, "font": font, "size": size, "bbox": box, "ink": box} for text, font, size, box in spans
            ],
        }

    above = line(108, 504, 243, 262, ("where the model predicts noise", "Times-Roman", 9.9, (108, 243, 504, 262)))
    denominator = line(235, 296, 258, 276, ("√1−αt", "CMSY7", 7.0, (235, 264.4, 260, 270.2)))
    numerator = line(243, 251, 256, 263, ("βt", "CMMI7", 7.0, (243.4, 256.2, 250.7, 262.5)))
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (108, 243, 504, 262), "lines": [above]},
            {"type": 0, "bbox": (235, 258, 296, 276), "lines": [denominator]},
            {"type": 0, "bbox": (243, 256, 251, 263), "lines": [numerator]},
        ]
    }
    _rejoin_split_lines(data)
    together = [block["lines"] for block in data["blocks"] if numerator in block["lines"]]
    assert len(together) == 1 and sorted(map(id, together[0])) == sorted(map(id, [denominator, numerator]))
    assert data["blocks"][0]["lines"] == [above]


def test_large_chapter_number_stays_apart_from_the_title() -> None:
    """A 60pt number and a 25pt heading at a chapter start are one large em apart: the number stays as it is, and the
    heading is written back alone at its original position. 2.2 Training at a single font size is still one
    line."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, rect: tuple[float, float, float, float], size: float) -> LineItem:
        span: Span = {"text": text, "size": size, "font": "Times", "flags": 0}
        return _item(text, pymupdf.Rect(rect), [span])

    chapter = [item("2", (98, 142, 128, 202), 59.8), item("Words and Tokens", (167, 142, 364, 166), 24.8)]
    assert _split_lines(chapter) == [[0], [1]]
    section = [item("2.2", (40, 100, 60, 112), 12), item("Training", (70, 100, 130, 112), 12)]
    assert _split_lines(section) == [[0, 1]]


def test_a_long_toc_title_with_two_leader_dots_keeps_its_page_number() -> None:
    """When a long title reaches close to the page number, only two dots of leader remain. On a contents page the
    leader and page number are still cut off and the page number stays in place. Such lines on other pages are
    left alone."""
    from app.formats.pdf_rows import _cut_leaders

    def span(text: str, x0: float, x1: float, y: float = 278.7) -> Span:
        return {"text": text, "size": 10.0, "bbox": (x0, y, x1, y + 12), "origin": (x0, y + 9.5), "color": 0}

    parts = [
        ("Fine-Tuning for Sequence Labeling:", 203.0, 347.9),
        (" ", 347.9, 350.9),
        ("Named Entity Recognition", 350.9, 457.8),
        (" ", 457.8, 461.6),
        (".", 461.6, 464.1),
        (" ", 464.1, 469.0),
        (".", 469.0, 471.5),
        (" ", 471.5, 478.3),
        ("238", 478.3, 493.2),
    ]
    spans = [span(*part) for part in parts]
    item: LineItem = (
        "".join(part[0] for part in parts),
        pymupdf.Rect(203.0, 278.7, 493.2, 290.7),
        [one for one in spans if one["text"].strip()],
        spans,
    )
    [cut], leaders = _cut_leaders([item], toc_page=True)
    assert cut[0].rstrip() == "Fine-Tuning for Sequence Labeling: Named Entity Recognition"
    assert leaders[0].number.x0 == 478.3 and leaders[0].pitch > 0
    [kept], leaders = _cut_leaders([item])
    assert kept[0] == item[0] and not leaders


def test_a_denominator_split_into_the_next_lines_block_rejoins_its_fraction() -> None:
    """The denominator N of count(w)/N was put into the block of the next line. The whole block is Times italic, so
    the font does not reveal a formula; it is recognized as a smaller italic centered below small italics in the
    body line and moved back to the line of the numerator. It is not moved when the italics above are at body size,
    or when they are only a superscript and subscript aligned at the left edge."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    reg, ital, cmr = "NimbusRomNo9L-Regu", "NimbusRomNo9L-ReguItal", "CMR10"

    def line(*spans: Span) -> Line:
        box = pymupdf.Rect(spans[0]["bbox"])
        for span in spans[1:]:
            box |= pymupdf.Rect(span["bbox"])
        return {"bbox": _bbox(box), "dir": (1, 0), "spans": list(spans)}

    def data(numerator: float) -> PageDict:
        score = line(
            _glyph(
                "The backoff terminates in the unigram, which has score",
                reg,
                10.0,
                4,
                (160.2, 307.0, 378.1, 319.1),
                (160.2, 309.4, 378.1, 316.4),
            ),
            _glyph(" S", ital, 10.0, 6, (378.1, 307.0, 385.2, 318.9), (378.1, 309.6, 385.2, 316.4)),
            _glyph(") =", cmr, 10.0, 4, (395.8, 308.8, 409.2, 318.7), (395.8, 308.8, 409.2, 318.7), math=True),
            _glyph("count", ital, numerator, 7, (412.2, 304.5, 428.4, 313.3), (412.2, 307.3, 428.4, 311.4)),
            _glyph("(", cmr, numerator, 5, (428.9, 305.8, 431.8, 313.2), (428.9, 305.8, 431.8, 313.2), math=True),
            _glyph("w", ital, numerator, 7, (431.8, 304.5, 436.7, 313.3), (431.8, 308.1, 436.7, 311.5)),
            _glyph(")", cmr, numerator, 5, (436.7, 305.8, 439.6, 313.2), (436.7, 305.8, 439.6, 313.2), math=True),
        )
        denominator = line(_glyph("N", ital, 7.4, 6, (423.2, 312.9, 428.1, 321.7), (423.2, 314.9, 428.1, 319.8)))
        tail = line(_glyph(". Brants et al.", reg, 10.0, 4, (440.7, 307.0, 493.2, 319.1), (440.7, 309.4, 493.2, 316.4)))
        rest = line(
            _glyph(
                "(2007) find that a value of 0.4 worked well",
                reg,
                10.0,
                4,
                (160.2, 319.0, 344.6, 331.0),
                (160.2, 321.4, 344.6, 328.4),
            )
        )
        return {
            "blocks": [
                {"type": 0, "bbox": score["bbox"], "lines": [score]},
                {"type": 0, "bbox": (160.2, 307.0, 493.2, 331.0), "lines": [denominator, tail, rest]},
            ]
        }

    split = data(7.4)
    score, (denominator, tail, rest) = split["blocks"][0]["lines"][0], split["blocks"][1]["lines"]
    _rejoin_split_lines(split)
    assert split["blocks"][0]["lines"] == [score, denominator, tail]
    assert split["blocks"][1]["lines"] == [rest]

    # The italic word above is at body size and is no numerator: the small text does not move
    words = data(10.0)
    small = words["blocks"][1]["lines"][0]
    _rejoin_split_lines(words)
    assert small in words["blocks"][1]["lines"]

    # The superscript and subscript of w^i_{i-k+1} share a left edge and are no fraction: nothing moves
    script = data(7.4)
    sup = script["blocks"][1]["lines"][0]
    sup["spans"][0]["bbox"], sup["spans"][0]["ink"] = (412.2, 312.9, 417.1, 321.7), (412.2, 314.9, 417.1, 319.8)
    sup["bbox"] = sup["spans"][0]["bbox"]
    _rejoin_split_lines(script)
    assert sup in script["blocks"][1]["lines"]


def test_an_example_number_stays_apart_from_its_sentence() -> None:
    """The example number (25.4) is a label and is not joined to the sentence on its right: once joined, the left edge
    of the block moves back to the number and the third, hanging-indent line can no longer be joined."""
    from app.formats.pdf_split_lines import _same_line_word

    assert not _same_line_word(
        "(25.4)",
        "Before winter I built a chimney",
        pymupdf.Rect(142.2, 556.9, 166.3, 568.9),
        pymupdf.Rect(179.1, 556.6, 458.9, 569.5),
        10.0,
    )


def test_numbered_examples_in_one_block_are_split_apart() -> None:
    """One block holds two example lines (1.2) and (1.3), the first being the longest, so the right edge does not show
    a hard break. The previous sentence is finished and the next line starts with an example number, so they split."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, bbox: tuple[float, float, float, float]) -> LineItem:
        return _item(text, pymupdf.Rect(bbox), [{"text": text, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0}])

    items = [
        item("(1.2) It was difficult to make it up the steep hill.", (142.2, 595.0, 335.0, 607.0)),
        item("(1.3) What a difficult decision!", (142.2, 609.0, 268.0, 621.0)),
    ]
    assert _split_lines(items) == [[0], [1]]
    wrapped = [
        item("as derived in the two equations", (142.2, 595.0, 335.0, 607.0)),
        item("(2.3) we see", (142.2, 609.0, 268.0, 621.0)),
    ]
    assert _split_lines(wrapped) == [[0, 1]]


def test_a_paragraph_after_a_hanging_list_line_is_split_off() -> None:
    """A list item wraps with a hanging indent, and the next paragraph has a first-line indent left of the continuation
    lines: the list has ended. The second line of a paragraph with a first-line indent returns to the column edge
    and is joined as usual."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, bbox: tuple[float, float, float, float]) -> LineItem:
        return _item(text, pymupdf.Rect(bbox), [{"text": text, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0}])

    items = [
        item("4. If one word in the question is changed, the human-preferred answer changes", (155, 202, 475, 214)),
        item("to the other party", (167, 214, 235, 226)),
        item("The kind of world knowledge that might be needed to solve the problems", (157, 229, 475, 241)),
    ]
    assert _split_lines(items) == [[0, 1], [2]]
    paragraph = [
        item("The first line of a paragraph is set to the full measure and", (157, 202, 475, 214)),
        item("The second line goes back to the margin", (142, 214, 475, 226)),
    ]
    assert _split_lines(paragraph) == [[0, 1]]


def test_a_line_in_a_column_of_its_own_font_stays_in_its_block() -> None:
    """A monospace example in the right column of a slide is level with the body text on the left and overlaps it a
    little horizontally. In its own block it shares the left edge and font of the line above, so it is a line of
    that column and is not moved over as the cut-off second half of the body text on the left."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(text: str, bbox: tuple[float, float, float, float], font: str) -> Line:
        return {
            "dir": (1, 0),
            "bbox": bbox,
            "spans": [
                {
                    "text": text,
                    "bbox": bbox,
                    "size": 29.0,
                    "font": font,
                    "flags": 0,
                    "color": 0,
                    "origin": (bbox[0], bbox[3] - 7),
                }
            ],
        }

    def page(with_mate: bool) -> PageDict:
        prose = line("according to its probability p(w|<s>)", (38, 221, 523, 256), "Calibri")
        sample = line("I want", (520, 232, 693, 265), "CourierNewPSMT")
        column = [line("<s> I", (520, 194, 607, 227), "CourierNewPSMT")] if with_mate else []
        return {
            "blocks": [
                {"type": 0, "bbox": (38, 221, 523, 256), "lines": [prose]},
                {"type": 0, "bbox": (520, 194, 693, 265), "lines": [*column, sample]},
            ]
        }

    data = page(True)
    _rejoin_split_lines(data)
    assert [len(block["lines"]) for block in data["blocks"]] == [1, 2]
    data = page(False)
    _rejoin_split_lines(data)
    assert [len(block["lines"]) for block in data["blocks"]] == [2, 0]


def test_a_small_fraction_after_a_relation_joins_the_line() -> None:
    """The 1/L after ≃ is one size smaller than the body and one body em away from ≃. Measured by the small em it
    would not be joined, so numerator and denominator together with the following sum would stay in another
    block and be skipped as a formula line, leaving the right half of the formula missing from the translation."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def line(x0: float, x1: float, y0: float, y1: float, text: str, font: str, size: float) -> Line:
        return {"bbox": (x0, y0, x1, y1), "dir": (1, 0), "spans": [{"text": text, "font": font, "size": size}]}

    head = line(108, 315, 84, 94, "that a differentiable estimator can be constructed:", "NimbusRomNo9L-Regu", 10.0)
    rel = line(330, 406, 84, 95, "q(z|x)f(z) dz ≃", "CMSY10", 10.0)
    numerator = line(414, 418, 82, 89, "1", "CMR7", 7.0)
    numerator["spans"][0]["math"] = True  # The mark _mark_math_roman puts on upright digits in a formula
    denominator = line(414, 419, 90, 97, "L", "CMMI7", 7.0)
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (108, 84, 406, 95), "lines": [head, rel]},
            {"type": 0, "bbox": (414, 82, 419, 97), "lines": [numerator, denominator]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [head, rel, numerator, denominator]


def test_a_numerator_left_in_its_own_block_rejoins_the_paragraph() -> None:
    """The numerator g⁴ and its subscript T,i were put into a separate block, so moving line by line always finds a
    line of the same block next to it. The whole block sits on the denominator formula in the paragraph, level
    with the body lines of the paragraph, and moves back into the paragraph as one."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    reg, cmmi = "NimbusRomNo9L-Regu", "CMMI7"

    def line(*spans: Span) -> Line:
        box = pymupdf.Rect(spans[0]["bbox"])
        for span in spans[1:]:
            box |= pymupdf.Rect(span["bbox"])
        return {"bbox": _bbox(box), "dir": (1, 0), "spans": list(spans)}

    lead = line(_glyph("From,", reg, 9.7, 4, (105.0, 558.9, 128.4, 570.6), (105.0, 561.0, 128.4, 569.9)))
    denominator = line(
        _glyph("4∥g", cmmi, 6.8, 0, (209.1, 566.4, 220.8, 573.2), (209.1, 566.4, 220.8, 573.2), math=True)
    )
    tail = line(
        _glyph("T,i", cmmi, 6.8, 0, (312.9, 565.5, 323.1, 572.2), (312.9, 565.5, 323.1, 571.0), math=True),
        _glyph(
            ", we can take square root of both side and",
            reg,
            9.7,
            4,
            (323.6, 558.9, 490.3, 570.6),
            (323.6, 561.0, 490.3, 569.9),
        ),
    )
    g4 = line(
        _glyph("g", cmmi, 6.8, 0, (219.6, 556.9, 223.3, 563.7), (219.6, 558.9, 223.3, 563.7), math=True),
        _glyph("4", "CMR5", 4.8, 0, (223.6, 555.4, 226.9, 560.3), (223.6, 555.4, 226.9, 559.1), math=True),
    )
    sub = line(_glyph("T,i", "CMMI5", 4.8, 0, (223.3, 560.3, 232.7, 565.1), (223.3, 560.3, 232.7, 563.9), math=True))
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": (105.0, 558.9, 490.3, 573.2), "lines": [lead, denominator, tail]},
            {"type": 0, "bbox": (219.6, 555.4, 232.7, 565.1), "lines": [g4, sub]},
        ]
    }
    _rejoin_split_lines(data)
    assert {id(item) for item in data["blocks"][0]["lines"]} == {
        id(item) for item in (lead, denominator, tail, g4, sub)
    }
    assert data["blocks"][1]["lines"] == []


def test_a_wrapped_index_entry_ends_where_its_page_numbers_do() -> None:
    """A wrapped index entry has hanging-indent continuation lines. A next line starting no further right than this
    one is another entry (see references have no page numbers); page numbers wrapped to the next line form a line
    of only page numbers, possibly in another block; a line split by justification is judged as a whole line."""
    from app.formats.pdf_rows import _index_entry_ends

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 9.6), [{"text": text, "size": 8.0}], [])

    see = [
        item("CFG, see context-free gram-", 179.3, 111.8, 271.3),
        item("mar", 203.3, 121.2, 215.6),
        item("chain rule, 123, 171", 179.3, 130.4, 242.9),
    ]
    assert _index_entry_ends(see, []) == {1, 2}
    head = [item("Chinese", 179.3, 206.6, 205.0), item("as verb-framed language,", 187.3, 216.1, 271.3)]
    assert 0 not in _index_entry_ends(head, [])
    # Continuation lines share the same indent; the entry ends only when the next entry starts at the left edge of
    # the first line
    long = [
        item("Transformations and Dis-", 77.4, 518.7, 169.4),
        item("course Analysis", 101.3, 528.2, 169.4),
        item("Project (TDAP),", 101.3, 537.7, 169.4),
        item("422", 101.3, 547.1, 113.3),
        item("transition probability", 77.4, 556.6, 144.5),
    ]
    assert _index_entry_ends(long, []) == {3}
    split = [item("collaborative", 179.3, 652.4, 220.6), item("completion,", 233.5, 652.4, 271.3)]
    numbers = [pymupdf.Rect(203.3, 661.6, 215.2, 671.9)]
    assert _index_entry_ends(split, numbers) == {1}
    assert _index_entry_ends(split, []) == set()
    # The continuation line ends with a comma, and the page number has the same indent as the continuation line
    wrapped = [item("as weighted average", 187.3, 225.6, 271.3), item("branching factor,", 203.3, 235.0, 271.3)]
    assert _index_entry_ends(wrapped, [pymupdf.Rect(203.3, 244.5, 211.2, 254.1)]) == {1}
    # The block starts with a continuation line of an entry and stops on half a broken word; the next line aligned
    # with it belongs to the same entry
    broken = [
        item("for pronoun resolu-", 203.3, 456.0, 271.3),
        item("tion, 566", 203.3, 465.5, 231.6),
        item("hold, 597", 179.3, 475.0, 210.0),
    ]
    assert _index_entry_ends(broken, []) == {1, 2}


def test_a_paragraph_indented_under_an_inset_line_starts_on_its_own() -> None:
    """The translation of an example is set as one indented line; the next paragraph has a first-line indent left of
    it and right of the lines that follow, so it is a new paragraph. A justified reference line split into pieces
    is compared by the left edge of the whole line and is not split. Centered lines do not count as indented."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        return (
            text,
            pymupdf.Rect(x0, y0, x1, y0 + 12),
            [{"text": text, "size": 10.0, "bbox": (x0, y0, x1, y0 + 12), "font": "Times", "flags": 0}],
            [],
        )

    items = [
        item("Soak the preserved vegetable in water until soft, remove, drain, and chop", 192.1, 463.4, 482.6),
        item("We generally distinguish two broad classes of morphemes: roots, the central", 175.1, 479.9, 493.2),
        item("morpheme of the word, supplying the main meaning, and affixes, adding", 160.2, 491.9, 493.2),
        item("additional meanings of various kinds.", 160.2, 504.2, 320.0),
    ]
    assert _split_lines(items) == [[0], [1, 2, 3]]
    entry = [
        item("carbon footprints of machine learn-", 223.3, 238.6, 339.3),
        item("ing.", 223.3, 247.1, 235.5),
        item("Journal of Machine Learning", 242.0, 247.1, 339.3),
        item("Research, 21(248):1-43.", 223.3, 255.6, 301.9),
        item("Henderson, P., X. Li, D. Jurafsky,", 213.3, 266.7, 339.3),
    ]
    assert any({2, 3} <= set(group) for group in _split_lines(entry))
    # A centered author affiliation gets wider line by line; the three lines share one center line and are not
    # indented
    author = [
        item("Machine Learning Group", 339.9, 182.1, 441.1),
        item("Universiteit van Amsterdam", 334.4, 193.0, 446.6),
        item("Welling Max at the institute of", 327.7, 203.5, 453.3),
    ]
    assert _split_lines(author) == [[0, 1, 2]]


def test_an_indented_line_after_a_full_line_that_ends_a_sentence_starts_a_paragraph() -> None:
    """The previous line is full and ends exactly at a sentence end, the next line is indented by one and a half em
    and reaches the right edge, and the line after it returns to the left edge: a new paragraph. With numbered
    entries above and below, it is a hanging-indent continuation line of a reference and is not split."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        return (
            text,
            pymupdf.Rect(x0, y0, x1, y0 + 12),
            [{"text": text, "size": 10.0, "bbox": (x0, y0, x1, y0 + 12), "font": "Times", "flags": 0}],
            [],
        )

    items = [
        item("This ordering reverses by 10 dB, and the residual advantage", 315.2, 428.9, 559.0),
        item("10 dB, and the residual advantage grows to 4.08 dB at 25 dB.", 315.2, 440.9, 559.0),
        item("The correction target in (9) provides an interpretation of", 330.2, 453.6, 559.0),
        item("this crossover. For slowly varying channels and weak obser-", 315.2, 465.5, 559.0),
        item("baseline achieves lower mean NMSE than both variants.", 315.2, 477.5, 539.5),
    ]
    assert _split_lines(items) == [[0, 1], [2, 3, 4]]
    entry = [
        item("46. Jim Fuller, Anthony L Piro, and Adam S Jermyn. Slowing the spins of cores.", 114.0, 100.0, 474.0),
        item("Monthly Notices of the Royal Astronomical Society, 485(3):3661, 2019.", 129.0, 112.0, 474.0),
        item("47. Ilya Mandel and Alison Farmer. Merging stellar-mass binary black holes.", 114.0, 124.0, 474.0),
    ]
    assert [0, 1] in _split_lines(entry)


def test_numbers_wrapped_under_an_indented_first_line_stay_in_the_paragraph() -> None:
    """A paragraph with a first-line indent has an unfinished list of numbers wrapped to the left edge of the next line
    (13 tokens: 11865, … | 885, 220, … 0)): this line aligns with the previous one and ends the sentence, so it
    does not become a new paragraph."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        return (
            text,
            pymupdf.Rect(x0, y0, x1, y0 + 12),
            [{"text": text, "size": 10.0, "bbox": (x0, y0, x1, y0 + 12), "font": "Times", "flags": 0}],
            [],
        )

    items = [
        item("The visualization shows colors to separate out words, but of course the true out-", 157.0, 169.0, 475.0),
        item("put of the tokenizer is simply a sequence of unique token ids. (In case you're in-", 142.0, 181.0, 475.0),
        item("terested, they were the following 13 tokens: 11865, 8923, 11, 31211, 6177, 23919,", 142.0, 193.0, 475.0),
        item("885, 220, 19427, 7633, 18887, 147065, 0)", 142.0, 205.0, 310.0),
        item("Notice that most words are their own token, usually including the leading space.", 157.0, 217.0, 475.0),
        item("Clitics like 's are segmented off when they appear on proper nouns like Jane, but", 142.0, 229.0, 475.0),
    ]
    assert _split_lines(items) == [[0, 1, 2, 3], [4, 5]]
    # The previous line stops on et al. inside a sentence, and the year on the next line is the second half of the
    # citation
    cited = [
        item("least another star and/or BH in close proximity to carry away", 303.3, 728.3, 558.4),
        item("the excess energy and momentum (Heggie 1975; Tanikawa et al.", 303.3, 739.7, 558.4),
        item("2012).", 303.3, 750.6, 329.0),
    ]
    assert _split_lines(cited) == [[0, 1, 2]]


def test_caption_lines_wrapped_back_under_the_label_stay_in_the_caption() -> None:
    """A caption label (Figure 4.12) and the text after it are separated by a gap but are two pieces of one line. The
    wrapped line returns to the left edge of the label, which is no jump to the left when compared by whole line,
    so it does not start a new paragraph: if the line with only (2012). became a paragraph of its own, it would
    keep the original text and still sit on the next line after the translation."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float, font: str = "NimbusRomNo9L-Regu") -> LineItem:
        return (
            text,
            pymupdf.Rect(x0, y0, x1, y0 + 11),
            [{"text": text, "size": 9.0, "bbox": (x0, y0, x1, y0 + 11), "font": font, "flags": 0}],
            [],
        )

    label = item("Figure 4.12", 143.2, 254.2, 186.4, "NimbusRomNo9L-Medi")
    first = item("A version of the paired bootstrap algorithm after Berg-Kirkpatrick et al.", 196.3, 254.5, 475.2)
    assert _split_lines([label, first, item("(2012).", 142.2, 265.5, 168.3)]) == [[0, 1, 2]]
    assert _split_lines(
        [label, first, item("(2012), resampling the test set with replacement.", 142.2, 265.5, 380.0)]
    ) == [[0, 1, 2]]


def test_page_numbers_wrapped_below_an_index_entry_stay_with_it() -> None:
    """An index entry stops on a comma, and its page numbers wrap to the next line, indented below the entry text.
    The page number line is sent for translation together with the entry, the entry ends at the page numbers, and
    the next entry starts anew as usual."""
    from app.formats.pdf_rows import _index_entry_ends
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        return (
            text,
            pymupdf.Rect(x0, y0, x1, y0 + 9.6),
            [{"text": text, "size": 8.0, "bbox": (x0, y0, x1, y0 + 9.6)}],
            [],
        )

    items = [
        item("Switchboard, 359", 401.2, 130.6, 457.5),
        item("Switchboard Corpus,", 401.2, 140.5, 472.4),
        item("338,", 479.3, 140.5, 493.2),
        item("339, 348, 349, 359", 425.1, 149.7, 484.9),
        item("sycophantic, 29", 401.2, 159.2, 451.3),
    ]
    groups = _split_lines(items, True)
    assert [1, 2, 3] in groups or any({1, 2, 3} <= set(group) for group in groups)
    ends = _index_entry_ends(items, [])
    assert 2 not in ends and 3 in ends and 4 in ends


def test_a_word_grid_is_translated_cell_by_cell() -> None:
    """A word table with one word per cell: each cell is its own entry, and cells in the same row are not joined into a
    sentence. A caption label in the same block below the table is not a cell."""
    from app.formats.pdf_rows import _grid_cells

    words = [
        ["dwindle", "move", "soar", "shift"],
        ["advance", "edge", "swell", "tumble"],
        ["climb", "explode", "swing", "fall"],
    ]
    items: list[LineItem] = [
        (word, pymupdf.Rect(184 + 60 * col, 180 + 10 * row, 230 + 60 * col, 190 + 10 * row), [], [])
        for row, line in enumerate(words)
        for col, word in enumerate(line)
    ]
    items.append(("Figure 2.4", pymupdf.Rect(80, 240, 120, 250), [], []))
    items.append(("Some selected ASCII codes for some English letters", pymupdf.Rect(130, 240, 400, 250), [], []))
    cells = _grid_cells(items)
    assert cells == set(range(12))
    prose: list[LineItem] = [
        (f"line {i} of an ordinary paragraph that wraps", pymupdf.Rect(80, 100 + 10 * i, 400, 110 + 10 * i), [], [])
        for i in range(12)
    ]
    assert not _grid_cells(prose)


def test_a_displayed_line_inside_a_block_is_its_own_paragraph() -> None:
    """An indented example line that is not full shares a block with the body text on the next line: the example is a
    paragraph of its own, and the next line is not merged into it even when it starts lowercase. In a paragraph
    set around a figure, the first and next lines end at the same x and are still one paragraph."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), [{"text": text, "size": 10.0}], [])

    witch = [
        item("The green witch arrived", 185.1, 473.7, 280.2),
        item("and must translate it into the Spanish sentence:", 160.2, 488.6, 347.0),
    ]
    assert _split_lines(witch) == [[0], [1]]
    wrapped = [
        item("This volume is about the implementation and im-", 175.1, 384.7, 369.9),
        item("plications of that exciting idea, as realized in large", 160.2, 396.3, 369.9),
        item("language models, and the vibrant interdisciplinary field", 160.2, 408.2, 493.2),
    ]
    assert _split_lines(wrapped) == [[0, 1, 2]]


def test_a_period_after_a_small_denominator_does_not_make_it_a_heading() -> None:
    """The denominator Σ_j e^{S·T_j} is followed by a body-size period. The period does not count as a body-size
    character, so the denominator is still joined back to the line of the fraction."""
    from app.formats.pdf_rejoin import _larger_text

    denominator: Line = {
        "spans": [
            {"text": "j", "font": "CMMI6", "size": 6.0},
            {"text": "e", "font": "CMMI8", "size": 8.0},
            {"text": " .", "font": "NimbusRomNo9L-Regu", "size": 10.9},
        ]
    }
    sigma: Line = {"spans": [{"text": "P", "font": "CMEX8", "size": 8.0}]}
    assert not _larger_text(denominator, sigma)
    heading: Line = {"spans": [{"text": "4.2 SQuAD", "font": "NimbusRomNo9L-Medi", "size": 12.0}]}
    assert _larger_text(heading, {"spans": [{"text": "words", "font": "NimbusRomNo9L-Regu", "size": 9.0}]})


def test_a_bold_run_in_label_starts_a_paragraph() -> None:
    """The previous line ends a sentence and the next line starts with a bold label: a paragraph heading of a few
    words (eight words count too), a bold theorem name with a number, and a numbered item such as Q2 all start a
    new paragraph. Spaces are added between spans split by word before recognizing them."""
    from app.formats.pdf_split_lines import _label_lead, _split_lines

    def span(text: str, x0: float, x1: float, bold: bool = False) -> Span:
        return {
            "text": text,
            "size": 10.0,
            "bbox": (x0, 0, x1, 10),
            "font": "Times-Bold" if bold else "Times",
            "flags": 20 if bold else 4,
        }

    def words(text: str, x0: float, bold_words: int = 0) -> list[Span]:
        out: list[Span] = []
        x = x0
        for k, word in enumerate(text.split()):
            out.append(span(word, x, x + 5 * len(word), bold=k < bold_words))
            x += 5 * len(word) + 3
        return out

    assert _label_lead(words("On the Optimality of the Level Schedule. Recall from", 313, 7))
    assert _label_lead(words("Parallelization Across Batches and Batch Instances. A", 313, 6))
    assert _label_lead(words("Proposition 5 (Efficiency beyond oracle radii). Un-", 315, 2))
    assert _label_lead(words("Q2 Which dynamical system functions benefit most", 60, 1))
    assert not _label_lead(words("Important things happen when we look closely", 60, 1))
    assert not _label_lead(words("This whole sentence is set in bold for emphasis and runs on. Then", 60, 12))

    def item(text: str, x0: float, y0: float, x1: float, bold_words: int = 0) -> LineItem:
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), words(text, x0, bold_words), [])

    items = [
        item("units. Finally, we return to the batch and state when the", 313.2, 190.0, 558.0),
        item("resulting schedule is also optimal for the whole batch.", 313.2, 201.9, 536.2),
        item("Parallelization Across Batches and Batch Instances. A", 313.2, 216.0, 558.0, 6),
        item("batch consists of B evaluations of the same function at dif-", 313.2, 227.9, 558.0),
    ]
    assert _split_lines(items) == [[0, 1], [2, 3]]


def test_a_roman_numbered_item_does_not_continue_the_paragraph_above() -> None:
    """(iii) in a proof is another item. When the previous paragraph ends on a display formula without a period, an
    (iii) followed by a capital is not joined to it either; an (iii) in an enumeration inside a sentence is
    followed by lowercase, as is a lowercase word in parentheses, and is joined as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_page_continue import _can_continue

    def blk(text: str) -> TextBlock:
        return TextBlock(
            page=0, rect=pymupdf.Rect(72, 500, 540, 530), line_rects=[], text=text, size=10, color="#000000", bold=False
        )

    prev = blk("The partition statement is the expansion {v11}")
    assert not _can_continue(prev, blk("(iii) Let {v1} be the sum."), 10, None)
    assert _can_continue(blk("(ii) it must be light, and"), blk("(iii) the arm must have few actuators."), 10, None)
    assert _can_continue(prev, blk("(see below) which holds."), 10, None)

    from app.formats.pdf_split_lines import _split_lines

    # The previous line stops on a formula and is not full, so (iii) starts a new item; after a full line it is an
    # enumeration inside the paragraph wrapped to the line start
    assert _split_lines(
        [
            _line("is the expansion = 1", 102, 152, 127),
            _line("(iii) Let the sum of the first n terms of the series be", 102, 505, 140),
            _line("written as a product of independent exponentials", 102, 400, 154),
        ]
    ) == [[0], [1, 2]]
    assert _split_lines(
        [
            _line("the intervening low-density region, regardless of r.", 63, 297, 134),
            _line("(ii) Such misses propagate through calibration. Be-", 63, 297, 146),
        ]
    ) == [[0, 1]]


def test_a_numeral_denominator_cut_into_the_next_block_joins_its_numerator() -> None:
    """For the numeric fraction 24000/75 in the body font, PyMuPDF puts the denominator 75 into the next block. The
    denominator is moved back to the line of the numerator; otherwise it would be a block of its own keeping the
    original glyphs, on top of the translation."""
    from app.formats.pdf_rejoin import _rejoin_split_lines

    def span(text: str, x0: float, y0: float, x1: float, y1: float, size: float, base: float) -> Span:
        return {
            "text": text,
            "bbox": (x0, y0, x1, y1),
            "origin": (x0, base),
            "size": size,
            "font": "NimbusRomNo9L-Regu",
            "flags": 4,
        }

    def line(*spans: Span) -> Line:
        box = pymupdf.Rect()
        for item in spans:
            box |= pymupdf.Rect(item["bbox"])
        return {"bbox": _bbox(box), "dir": (1, 0), "spans": list(spans)}

    head = line(
        span("this is a downsampling of", 77.4, 623.9, 368.9, 634.7, 9.0, 631.8),
        span("24000", 372.6, 622.2, 390.0, 630.6, 7.0, 627.7),
    )
    denominator = line(span("75", 377.8, 628.9, 384.8, 637.3, 7.0, 634.4))
    nxt = line(span("the encoder and decoder is a quantization step producing", 77.4, 634.9, 352.1, 645.7, 9.0, 642.6))
    data: PageDict = {
        "blocks": [
            {"type": 0, "bbox": head["bbox"], "lines": [head]},
            {"type": 0, "bbox": (77.4, 628.9, 384.8, 645.7), "lines": [denominator, nxt]},
        ]
    }
    _rejoin_split_lines(data)
    assert data["blocks"][0]["lines"] == [head, denominator]
    assert data["blocks"][1]["lines"] == [nxt]


def test_examples_listed_under_a_bullet_get_their_own_line() -> None:
    """A bullet item breaks before reaching the right edge of the column although the first word of the next line
    would fit: the next line is an example listed under the item, so they split. When the item fills its line, a
    next line whose letters are mostly colored, below a line with no colored text, also splits. A continuation
    line wrapped from a full line does not split, and neither does a reference starting with name initials"""
    from app.formats.pdf_rows import _cut_item_breaks

    def item(text: str, rect: tuple[float, float, float, float], color: int = 0) -> LineItem:
        return _item(text, pymupdf.Rect(rect), [{"text": text, "size": 10.0, "color": color}])

    body = [pymupdf.Rect(142, 100 + 12 * i, 475, 110 + 12 * i) for i in range(5)]
    listed = [
        item("• Bag-of-words and bigrams in M1 and M2", (159, 234, 334, 246)),
        item("American, Airlines, Tim, Wagner, American Airlines, Tim Wagner", (167, 246, 433, 258), 0x0000FF),
    ]
    assert _cut_item_breaks([0, 1], listed, body + [item[1] for item in listed]) == [[0], [1]]
    full = [
        item("• Entity Level of M1 and M2 (from the set NAME, NOMINAL, PRONOUN)", (159, 367, 469, 379)),
        item("M1: NAME [it or he would be PRONOUN]", (167, 379, 343, 391), 0x0000FF),
    ]
    assert _cut_item_breaks([0, 1], full, body) == [[0], [1]]
    wrapped = [
        item("• Number of entities between the two arguments of the relation, counted", (159, 404, 475, 416)),
        item("over the whole sentence", (167, 416, 280, 428)),
    ]
    assert _cut_item_breaks([0, 1], wrapped, body) == [[0, 1]]
    reference = [
        item("E. H. Hovy, V. Ng, and M. Strube.", (159, 404, 330, 416)),
        item("Scoring coreference partitions", (167, 416, 330, 428), 0x0000FF),
    ]
    assert _cut_item_breaks([0, 1], reference, body) == [[0, 1]]
