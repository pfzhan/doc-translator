import itertools
import string

import pymupdf

from app.formats.pdf_extract import extract_blocks
from app.formats.pdf_frame import FrameLine, margin_notes, note_groups, note_pitch, text_frames
from app.formats.pdf_types import LineItem, PdfDict

_LINE = "a body line of running text that is long enough to vote"
# Every body line has different text; the same sentence at the same height on every page is a running header
_TAGS = ("".join(pair) for pair in itertools.product(string.ascii_lowercase, repeat=3))


def _lines(x0: float, x1: float, count: int = 14, top: float = 100) -> list[FrameLine]:
    return [
        FrameLine(pymupdf.Rect(x0, top + 12 * i, x1, top + 12 * i + 10), f"{_LINE} {next(_TAGS)}", 10)
        for i in range(count)
    ]


def test_frame_follows_body_lines_not_the_leftmost_block() -> None:
    header = FrameLine(pymupdf.Rect(77, 84, 143, 94), "4 CHAPTER 2 • WORDS AND TOKENS ...........", 8)
    frames = text_frames([_lines(160, 493), [header, *_lines(142, 475)], _lines(160, 493), [header, *_lines(142, 475)]])
    first, second = frames[0], frames[1]
    assert first is not None and second is not None
    assert (first.x0, first.x1) == (160, 493)
    assert (second.x0, second.x1) == (142, 475)
    assert second.y0 == 100


def test_a_wide_caption_does_not_move_the_frame_of_its_page() -> None:
    """A full-width caption starts at the page margin, while the body on this page still starts at the left edge of
    pages on the same side."""
    caption = _lines(77, 475, count=6, top=300)
    frames = text_frames(
        [_lines(160, 493), [*_lines(142, 475, count=12), *caption], _lines(160, 493), _lines(142, 475)]
    )
    frame = frames[1]
    assert frame is not None
    assert (frame.x0, frame.x1) == (142, 475)


def test_a_running_header_set_like_body_text_is_outside_the_frame() -> None:
    """A running header as large and as long as body text, the same sentence at the same height on every page, is not
    the first line of the text frame."""
    header = "Published as a conference paper at ICLR 2015"
    pages = [[FrameLine(pymupdf.Rect(105, 61, 285, 73), header, 10), *_lines(108, 504, top=116)] for _ in range(4)]
    assert all(frame is not None and frame.y0 == 116 for frame in text_frames(pages))


def test_two_columns_span_both_columns() -> None:
    page = [*_lines(40, 290), *_lines(316, 560)]
    [frame] = text_frames([page])
    assert frame is not None
    assert (frame.x0, frame.x1) == (40, 560)


def test_small_terms_on_a_lane_beside_the_frame_are_notes() -> None:
    body = _lines(142, 475)
    notes = [
        FrameLine(pymupdf.Rect(84, 161, 128, 169), "Herdan's Law", 7.2),
        FrameLine(pymupdf.Rect(91, 173, 128, 180), "Heaps' Law", 7.2),
        FrameLine(pymupdf.Rect(82, 281, 128, 288), "function words", 7.2),
        # Page numbers and protruding section numbers are as large as body text and are not margin notes
        FrameLine(pymupdf.Rect(77, 84, 82, 94), "4", 10),
        FrameLine(pymupdf.Rect(77, 300, 99, 315), "2.1", 17),
    ]
    page = [*body, *notes]
    frames = text_frames([page])
    found = margin_notes([page], frames)[0]
    assert found == {len(body), len(body) + 1, len(body) + 2}


def test_a_body_size_symbol_on_the_note_lane_is_a_note() -> None:
    """ŷ in a margin note is as large as body text but aligned with the margin note edge, so it is a margin note too.
    Protruding section numbers are not on that edge."""
    body = _lines(142, 475)
    notes = [
        FrameLine(pymupdf.Rect(84, 161, 128, 169), "Herdan's Law", 7.2),
        FrameLine(pymupdf.Rect(91, 173, 128, 180), "Heaps' Law", 7.2),
        FrameLine(pymupdf.Rect(82, 281, 128, 288), "function words", 7.2),
        FrameLine(pymupdf.Rect(123.4, 300, 128, 312), "\u02c6y", 10),
        FrameLine(pymupdf.Rect(77, 330, 99, 345), "2.1", 10),
    ]
    page = [*body, *notes]
    found = margin_notes([page], text_frames([page]))[0]
    assert found == {len(body), len(body) + 1, len(body) + 2, len(body) + 3}


def test_two_line_note_is_one_note_and_neighbours_stay_apart() -> None:
    lines = [
        FrameLine(pymupdf.Rect(123, 623, 146, 630), "regular", 7.2),
        FrameLine(pymupdf.Rect(111, 630, 146, 637), "expressions", 7.2),
        FrameLine(pymupdf.Rect(84, 661, 146, 669), "Herdan's Law", 7.2),
        FrameLine(pymupdf.Rect(91, 673, 146, 680), "Heaps' Law", 7.2),
    ]
    assert note_groups(lines) == [[0, 1], [2], [3]]


def test_stacked_notes_split_where_the_pitch_is_off() -> None:
    """One body line introduces two terms, and their margin notes sit right above each other: listen attend / and
    spell starts after a larger gap below AED, and vector / semantics and representation / learning are stacked
    tighter than the line pitch. The pitch within one note matches that of the whole document"""

    def note(y0: float, text: str) -> FrameLine:
        return FrameLine(pymupdf.Rect(110, y0, 146, y0 + 9.4), text, 7.2)

    regular = [[note(100 + 30 * i, "regular"), note(106.5 + 30 * i, "expressions")] for i in range(12)]
    pitches = note_pitch(regular, [{0, 1}] * len(regular))
    assert pitches == {7.2: 6.5}
    stacked = [note(290.9, "AED"), note(298.9, "listen attend"), note(305.3, "and spell")]
    assert note_groups(stacked, pitches) == [[0], [1, 2]]
    tight = [note(636.5, "vector"), note(643.0, "semantics"), note(647.8, "representation"), note(654.3, "learning")]
    assert note_groups(tight, pitches) == [[0, 1], [2, 3]]
    assert note_groups(tight) == [[0, 1, 2, 3]]


def test_margin_note_on_a_body_line_is_split_off() -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    words = (
        "The body of this paragraph keeps going across many lines so that the "
        "frame has enough long lines to vote on where the text starts "
    ).split()
    y = 120
    lines = []
    for i in range(16):
        lines.append(" ".join(words[(i * 3) % 20 : (i * 3) % 20 + 9]))
    for i, line in enumerate(lines):
        page.insert_text((160, y + 12 * i), line, fontsize=10)
    for row, note in ((2, "tokenization"), (6, "regular"), (10, "edit distance")):
        width = pymupdf.get_text_length(note, fontsize=7)
        page.insert_text((146 - width, y + 12 * row), note, fontsize=7)
    blocks = extract_blocks(doc)
    notes = [block.text for block in blocks if block.side_note]
    assert notes == ["tokenization", "regular", "edit distance"]
    body = " ".join(block.text for block in blocks if not block.side_note)
    assert "tokenization" not in body and "regular" not in body and "edit distance" not in body


def test_label_of_a_caption_wider_than_the_body_is_not_a_margin_note() -> None:
    """The caption of a wide table starts at the page margin, with its label where margin notes are. The next caption
    line starts at the same place and runs into the text frame, so the label stays in the caption. Real margin notes
    are still split off."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    words = (
        "The body of this paragraph keeps going across many lines so that the "
        "frame has enough long lines to vote on where the text starts "
    ).split()
    for i in range(16):
        page.insert_text((160, 120 + 12 * i), " ".join(words[(i * 3) % 20 : (i * 3) % 20 + 9]), fontsize=10)
    for row, note in ((2, "tokenization"), (6, "regular"), (10, "edit distance")):
        page.insert_text((96, 120 + 12 * row), note, fontsize=7)
    page.insert_text((96, 330), "Figure 3.7", fontsize=9, fontname="hebo")
    page.insert_text((150, 330), "Add-one smoothed bigram probabilities for eight of the words in the", fontsize=9)
    page.insert_text((96, 341), "corpus of 9332 sentences. Previously-zero probabilities are in gray.", fontsize=9)
    blocks = extract_blocks(doc)
    assert [block.text for block in blocks if block.side_note] == ["tokenization", "regular", "edit distance"]
    [caption] = [block for block in blocks if "Add-one" in block.text]
    assert caption.text.startswith("Figure 3.7") and "corpus of 9332" in caption.text


def test_a_caption_spanning_the_margin_keeps_its_short_lines() -> None:
    """The whole caption reaches into the margin: the line after the label starts deep in the margin and runs into the
    text frame, and the next line has only a few words, also where margin notes are (Figure 6.17 ... computa- |
    tion of). Both the label and those words stay in the caption."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    words = (
        "The body of this paragraph keeps going across many lines so that the "
        "frame has enough long lines to vote on where the text starts "
    ).split()
    for i in range(16):
        page.insert_text((160, 120 + 12 * i), " ".join(words[(i * 3) % 20 : (i * 3) % 20 + 9]), fontsize=10)
    for row, note in ((2, "tokenization"), (6, "regular"), (10, "edit distance")):
        page.insert_text((78, 120 + 12 * row), note, fontsize=7)
    # The label is white on black and drawn after the caption, so PyMuPDF makes it a separate line
    page.insert_text((132, 330), "Computation graph for the function, showing the backward pass computa-", fontsize=9)
    page.insert_text((78, 341), "tion of it.", fontsize=9)
    page.insert_text((78, 330), "Figure 6.17", fontsize=9, fontname="hebo")
    blocks = extract_blocks(doc)
    assert [block.text for block in blocks if block.side_note] == ["tokenization", "regular", "edit distance"]
    [caption] = [block for block in blocks if "Computation graph" in block.text]
    assert caption.text.startswith("Figure 6.17") and caption.text.endswith("computation of it.")


def test_displayed_formula_with_a_big_brace_keeps_its_rows() -> None:
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_skip import _display_formula_ids

    def block(text: str, rect: tuple[float, float, float, float], font: str = "NimbusRomNo9L-Regu") -> TextBlock:
        span: PdfDict = {"text": text, "font": font, "size": 10.0}
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(rect),
            line_rects=[pymupdf.Rect(rect)],
            text=text,
            size=10,
            color="#000000",
            bold=False,
            span_lines=[[span]],
        )

    brace = block("D[i−1, j −1]+ 2; if source[i] ≠ target[j]", (237, 408, 432, 432), "CMEX10")
    condition = block("0; if source[i] = target[j]", (325, 420, 432, 430))
    body = block("Under this version of Levenshtein, the computation for D[i, j] becomes:", (142, 380, 475, 390))
    inline = block("a sum written inline uses the extension font too", (142, 440, 475, 450), "CMEX10")
    geo = PageGeometry.from_rects(612, [], [], 792, frame=TextFrame(142, 475, 111, 719, 10))
    # List continuation lines and right-column paragraphs also start away from the left edge, and integral signs
    # inside sentences use the same font, so the word count marks them as sentences
    item = TextBlock(
        page=0,
        rect=pymupdf.Rect(170, 460, 475, 500),
        line_rects=[pymupdf.Rect(170, 460, 475, 500)],
        text="",
        size=10,
        color="#000000",
        bold=False,
        span_lines=[
            [
                {"text": "R", "font": "CMEX10", "size": 10.0},
                {
                    "text": "pθ(z)pθ(x|z) dz is intractable (so we cannot evaluate or differentiate the marginal "
                    "likelihood), where the true posterior density is intractable (so the EM algorithm "
                    "cannot be used), and where the required integrals",
                    "font": "NimbusRomNo9L-Regu",
                    "size": 10.0,
                },
            ]
        ],
    )
    # Inline wide hats and small radicals use this font too and do not indicate a display formula
    accent = TextBlock(
        page=0,
        rect=pymupdf.Rect(300, 520, 475, 540),
        line_rects=[pymupdf.Rect(300, 520, 475, 540)],
        text="",
        size=10,
        color="#000000",
        bold=False,
        span_lines=[
            [
                {"text": "Let the normalized values be x", "font": "NimbusRomNo9L-Regu", "size": 10.0},
                {"text": "b", "font": "CMEX10", "size": 10.0},
                {
                    "text": ", and their transformations be y. We refer to the transform",
                    "font": "NimbusRomNo9L-Regu",
                    "size": 10.0,
                },
            ]
        ],
    )
    ids = _display_formula_ids([brace, condition, body, inline, item, accent], [geo])
    assert ids == {id(brace), id(condition)}


def test_a_centered_wide_formula_close_to_the_left_edge_is_displayed() -> None:
    """A wide centered formula leaves less than two characters of space on the left, as much as on the right."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry
    from app.formats.pdf_skip import _display_formula_ids

    rect = pymupdf.Rect(160.5, 640, 457.7, 655)
    formula = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="",
        size=10,
        color="#000000",
        bold=False,
        span_lines=[
            [
                {"text": "p(x1,...,xn) =", "font": "CMMI10", "size": 10.0},
                {"text": "Y", "font": "CMEX10", "size": 10.0},
                {"text": "p(xi|R(q) ; Answer the following question... ; q ;x<i)", "font": "CMMI10", "size": 10.0},
            ]
        ],
    )
    geo = PageGeometry.from_rects(612, [], [], 792, frame=TextFrame(142, 475, 111, 719, 10))
    assert _display_formula_ids([formula], [geo]) == {id(formula)}


def test_body_cut_at_a_margin_note_is_joined_back() -> None:
    from app.formats.pdf_extract import _rejoin_note_cuts

    def line(text: str, x0: float, y0: float, x1: float, size: float = 10.0) -> PdfDict:
        return {
            "dir": (1, 0),
            "bbox": (x0, y0, x1, y0 + size),
            "spans": [{"text": text, "size": size, "font": "Times", "flags": 0, "color": 0}],
        }

    upper: PdfDict = {
        "type": 0,
        "lines": [
            line("Starting in the 1960s, the characters (like the ones used", 160, 341, 493),
            line("in this sentence), were represented with ASCII (American Standard", 160, 353, 493),
        ],
    }
    lower: PdfDict = {
        "type": 0,
        "lines": [line("Code for Information Interchange). ASCII used one byte.", 160, 365, 493)],
    }
    after: PdfDict = {"type": 0, "lines": [line("Chinese has about 100,000 characters in Unicode.", 175, 377, 493)]}
    data: PdfDict = {"blocks": [upper, lower, after]}
    _rejoin_note_cuts(data, {id(upper)})
    assert len(upper["lines"]) == 3 and lower["lines"] == []
    assert after["lines"]  # the previous line ends a sentence and the next block is indented: a new paragraph


def test_a_bold_caption_label_starts_its_own_paragraph() -> None:
    """The caption of a wide figure is less than three characters left of the figure text, like an indent. The label
    is set alone in bold and is the first line of the caption. A body sentence starting with Figure 3.8 keeps its
    font and is not split."""
    from app.formats.pdf_split_lines import _split_lines

    def item(*spans: tuple[str, str], x0: float, y0: float) -> LineItem:
        parts: list[PdfDict] = [
            {"text": text, "size": 9.0 if font.endswith("Medi") else 8.0, "font": font, "flags": 0}
            for text, font in spans
        ]
        width = sum(len(text) for text, _font in spans) * 4.5
        return ("".join(text for text, _font in spans), pymupdf.Rect(x0, y0, x0 + width, y0 + 9), parts, parts)

    figure = [
        item(("anchorTimeID=” t1 ”> the weekend </TIMEX3>, marking", "Courier"), x0=98, y0=147),
        item(("weeks </TIMEX3>.", "Courier"), x0=98, y0=166),
        item(
            ("Figure 21.17", "Times-Medi"),
            (" TimeML markup including normalized values.", "Times-Roman"),
            x0=78,
            y0=176,
        ),
    ]
    assert _split_lines(figure) == [[0, 1], [2]]
    body = [
        item(("We can see the counts of all bigrams in the corpus", "Times-Roman"), x0=98, y0=147),
        item(("Figure 3.8 shows the reconstructed counts.", "Times-Roman"), x0=78, y0=158),
    ]
    assert _split_lines(body) == [[0, 1]]


def test_typewriter_dialogue_keeps_one_line_per_turn() -> None:
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float, font: str = "txtt", flags: int = 8) -> LineItem:
        span: PdfDict = {"text": text, "size": 9.0, "font": font, "flags": flags}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 9), [span], [span])

    turns = [
        item("ELIZA: WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP", 221, 200, 471),
        item("User: Perhaps I could learn to get along with my mother.", 221, 211, 490),
    ]
    assert _split_lines(turns) == [[0], [1]]
    wrapped = [
        item("They picnicked by the pool, then lay back on the grass and", 209, 200, 471),
        item("looked at the stars.", 209, 211, 300),
    ]
    assert _split_lines(wrapped) == [[0, 1]]


def test_prose_after_typewriter_lines_starts_a_new_paragraph() -> None:
    """A small list item in monospace is directly followed by the first body line. The first-line indent differs only
    slightly from the item's left edge, and they must still be split."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float, size: float, font: str, flags: int) -> LineItem:
        span: PdfDict = {"text": text, "size": size, "font": font, "flags": flags}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + size), [span], [span])

    items = [
        item("illustrate its explanations with examples, thought experiments", 167, 577, 489, 8.0, "txtt", 12),
        item("self-destructive behavior", 167, 600, 272, 8.0, "txtt", 12),
        item(
            "Prompts can also be used in more sophisticated ways in a method",
            157,
            611,
            475,
            10.0,
            "NimbusRomNo9L-Regu",
            4,
        ),
        item(
            "time compute. One representative example is chain-of-thought", 142, 623, 475, 10.0, "NimbusRomNo9L-Regu", 4
        ),
    ]
    assert _split_lines(items) == [[0, 1], [2, 3]]


def test_a_larger_line_below_a_slide_formula_starts_a_new_paragraph() -> None:
    """On a slide, an indented formula is followed by a subheading one size larger whose left edge is just over one
    character away, like a first-line indent; they must be split. The rest of a sentence returning to body size
    after small capitals is not split."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float, size: float, font: str = "TimesNewRomanPSMT") -> LineItem:
        span: PdfDict = {"text": text, "size": size, "font": font, "flags": 0}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + size * 1.2), [span], [span])

    slide = [
        item(" P(A,B,C,D) = P(A) P(B|A) P(C|A,B) P(D|A,B,C)", 161, 380, 865, 35.0),
        item("The Chain Rule in General", 113, 433, 547, 42.0, "Calibri"),
        item("  P(x1,x2,x3,…,xn) = P(x1)P(x2|x1)P(x3|x1,x2)…P(xn|x1,…,xn-1)", 104, 499, 1118, 42.0),
    ]
    assert _split_lines(slide) == [[0], [1, 2]]
    reference = [
        item("The KNOWREF", 359, 297, 475, 6.4),
        item("coreference corpus: Removing gen-", 359, 305, 475, 8.0),
    ]
    assert _split_lines(reference) == [[0, 1]]


def test_the_short_last_line_of_an_example_does_not_open_the_next_paragraph() -> None:
    """The hanging continuation line of an example is not full and ends with a period, and the next paragraph starts
    at the column edge, just over one character to the left; this is not a first-line indent. The full first line of
    an indented paragraph still continues as before."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        span: PdfDict = {"text": text, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), [span], [span])

    example = [
        item("of graphics chips.", 185, 529, 256),
        item("Event mentions are much harder to detect than entity mentions, since they can be ver-", 160, 543, 493),
        item("bal as well as nominal.", 160, 555, 300),
    ]
    assert _split_lines(example) == [[0], [1, 2]]
    indented = [
        item("Event mentions are much harder to detect than entity mentions, since they", 175, 529, 493),
        item("can be verbal as well as nominal.", 160, 543, 300),
    ]
    assert _split_lines(indented) == [[0, 1]]
    # The block starts with a continuation line of the previous item, so its first line is indented too. After two
    # indented lines, the line back at the column edge is the next item
    mid_item = [
        item("was there a task? Was the language originally spoken conversation, edited", 167, 335, 475),
        item("text, social media communication, monologue vs. dialogue?", 167, 347, 407),
        item("Language variety: What language (including dialect/region) was the corpus in?", 147, 362, 468),
    ]
    assert _split_lines(mid_item) == [[0, 1], [2]]


def test_a_bold_label_after_a_finished_sentence_opens_the_next_item() -> None:
    """A definition list with one nearly full line per item, so hard line breaks do not show the items. After a
    finished sentence, a line starting with a bold label is the next item. A bold word that wraps to a line start
    mid-sentence is not split."""
    from app.formats.pdf_split_lines import _split_lines

    def item(label: str, rest: str, y0: float, x1: float) -> LineItem:
        spans: list[PdfDict] = (
            [{"text": label, "size": 10.0, "font": "NimbusRomNo9L-Medi", "flags": 16}] if label else []
        )
        spans.append({"text": rest, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0})
        return (label + rest, pymupdf.Rect(147, y0, x1, y0 + 10), spans, spans)

    rows = [
        item("Language variety", ": What language (including dialect/region) was the corpus in?", 362, 468),
        item("Speaker demographics", ": What was, e.g., the age or gender of the text’s authors?", 377, 468),
        item("Collection process", ": How big is the data? If it is a subsample how was it sampled?", 391, 475),
    ]
    assert _split_lines(rows) == [[0], [1], [2]]
    wrapped = [
        item("", "Tokenization is usually done in one of two ways, which we will call the", 362, 475),
        item("two kinds", ": top-down tokenization and bottom-up tokenization of the text", 377, 475),
    ]
    assert _split_lines(wrapped) == [[0, 1]]


def test_a_bold_label_after_a_short_unfinished_item_opens_the_next_item() -> None:
    """Definition list items are partial example sentences without sentence-ending punctuation (mouse² : ... a quiet
    animal like a mouse). The previous item also starts with a bold label and does not reach the right edge, so the
    bold label on the next line is the next item"""
    from app.formats.pdf_split_lines import _split_lines

    def item(label: str, rest: str, y0: float, x1: float) -> LineItem:
        spans: list[PdfDict] = [
            {"text": label, "size": 10.0, "font": "CMSSBX10", "flags": 20},
            {"text": " :" + rest, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 4},
        ]
        return (label + " :" + rest, pymupdf.Rect(142, y0, x1, y0 + 15), spans, spans)

    rows = [
        item("mouse1", " .... a mouse controlling a computer system in 1968.", 569, 389),
        item("mouse2", " .... a quiet animal like a mouse", 581, 307),
        item("bank1", " ...a bank can hold the investments in a custodial account ...", 593, 412),
        item("bank2", " ...as agriculture burgeons on the east bank, the river ...", 605, 393),
    ]
    assert _split_lines(rows) == [[0], [1], [2], [3]]
    # Once split into two blocks, they are not joined back as caption continuation lines either
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_caption_fragments

    boxes = [pymupdf.Rect(142, 584, 307, 596), pymupdf.Rect(142, 596, 412, 608)]
    blocks = [
        TextBlock(
            page=0,
            rect=box,
            line_rects=[box],
            text=text,
            size=10.0,
            color="#000",
            bold=False,
            span_lines=[spans],
            line_gap=0.0,
        )
        for (text, _rect, spans, _all), box in zip(rows[1:3], boxes)
    ]
    assert len(_merge_caption_fragments(blocks)) == 2


def test_a_heading_across_a_narrow_gutter_is_not_joined_to_the_left_line() -> None:
    """A left-column line is level with a numbered heading in the right column, separated by a gutter one character
    wide. The section number starts at the right column's edge and is not the second half of the left line."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    left = ["BN-Baseline: Same as Inception with Batch Normal-", "ization before each nonlinearity."]
    for i, text in enumerate(left):
        page.insert_text((72, 320 + 12 * i), text, fontsize=10)
    page.insert_text((311, 318), "4.2.3", fontsize=10, fontname="hebo")
    page.insert_text((352, 318), "Ensemble Classification", fontsize=10, fontname="hebo")
    for i in range(6):
        page.insert_text((311, 340 + 12 * i), f"The current reported best results on ImageNet line {i}", fontsize=10)
    from app.formats.pdf_rejoin import _rejoin_split_lines

    data = page.get_text("dict")
    _rejoin_split_lines(data)
    left_block = next(block for block in data["blocks"] if "Normal-" in block["lines"][0]["spans"][0]["text"])
    assert ["".join(span["text"] for span in line["spans"]) for line in left_block["lines"]] == left


def test_an_inline_enumeration_wrapping_onto_a_new_line_is_not_a_list() -> None:
    """In a caption, (a) is mid-line and (b) happens to start a line after a full one: a wrapped line of the same
    paragraph. When every item starts a line, it is a list."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, y: float, x1: float = 504) -> LineItem:
        span: PdfDict = {"text": text, "size": 9.0, "font": "Times-Roman", "flags": 0}
        return (text, pymupdf.Rect(108, y, x1, y + 10), [span], [span])

    caption = [
        item("transformed samples. G contracts in regions of high density and expands in regions of low. (a)", 524),
        item(
            "Consider an adversarial pair near convergence: pg is similar to pdata and D is a partially accurate.", 534
        ),
        item("(b) In the inner loop of the algorithm D is trained to discriminate samples from data, converging", 544),
    ]
    assert _split_lines(caption) == [[0, 1, 2]]
    listed = [
        item("(a) Consider an adversarial pair near convergence.", 524, 330),
        item("(b) In the inner loop of the algorithm D is trained.", 534, 330),
    ]
    assert _split_lines(listed) == [[0], [1]]


def test_blocks_in_a_flagged_monospace_font_are_code() -> None:
    """Fonts such as txtt have no Mono in their names, only the PDF monospace flag. A block entirely in such a font
    keeps the original."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_unit_blocks import _code_block

    def block(*spans: tuple[str, str, int]) -> TextBlock:
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(0, 0, 100, 10),
            line_rects=[],
            text="".join(t for t, _, _ in spans),
            size=10,
            color="#000",
            bold=False,
            span_lines=[[{"text": t, "font": f, "flags": flags, "size": 10} for t, f, flags in spans]],
        )

    assert _code_block(block(("<system> You are a helpful assistant.", "txtt", 12)))
    assert not _code_block(
        block(
            ("The prompt ", "NimbusRomNo9L-Regu", 4),
            ("<system>", "txtt", 12),
            (" comes first in the context window.", "NimbusRomNo9L-Regu", 4),
        )
    )
    # Chinese fonts often carry the monospace flag too and are not code
    assert not _code_block(block(("系统提示词放在上下文的最前面。", "SimSun", 8)))
    # The monospace font of the LaTeX times package is URW's NimbusMonL, without Mono in its name and not always
    # flagged as monospace
    assert _code_block(block(("dpkingma@gmail.com", "NimbusMonL-Regu", 4)))
    # A row of a corpus table: the count at the start is in the body font, the words after it in monospace. Digits
    # do not count as body text
    assert _code_block(block(("2", "NimbusRomNo9L-Regu", 4), ("n", "txtt", 12), (" e w", "txtt", 12)))


def test_gloss_line_is_split_from_its_free_translation() -> None:
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _gloss_ids
    from app.formats.pdf_unit_blocks import _split_gloss_lines

    lines = [pymupdf.Rect(174, 400, 399, 410), pymupdf.Rect(174, 414, 389, 424)]
    spans = [
        [{"text": "1SG.S-E-midnight-big-E-sew-1SG.S[PFV]", "size": 10.0}],
        [{"text": "“I sewed a lot of yurt covers.”", "size": 10.0}],
    ]
    block = TextBlock(
        page=0,
        rect=lines[0] | lines[1],
        line_rects=lines,
        text="",
        size=10,
        color="#000000",
        bold=False,
        span_lines=spans,
    )
    glosses: dict[int, tuple[pymupdf.Rect, ...]] = {0: (pymupdf.Rect(174, 388, 399, 410),)}
    gloss, free = _split_gloss_lines([block], glosses)
    assert gloss.text.startswith("1SG.S") and free.text.startswith("“I sewed")
    assert _gloss_ids([gloss, free], glosses) == {id(gloss)}


def test_centered_author_rows_stay_one_item_per_row() -> None:
    """A centered author block: a bold name, an affiliation in the regular font and an email in monospace, one line
    each. A next line in another font is another item and is not joined with the line above. An affiliation wrapped
    onto two lines in the same font is still one item."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    def centered(text: str, axis: float, y: float, font: str) -> None:
        width = pymupdf.get_text_length(text, fontname=font, fontsize=10)
        page.insert_text((axis - width / 2, y), text, fontsize=10, fontname=font)

    for axis, name, mail in (
        (215, "Diederik P. Kingma", "dpkingma@gmail.com"),
        (390, "Max Welling", "welling.max@gmail.com"),
    ):
        centered(name, axis, 180, "hebo")
        centered("Machine Learning Group", axis, 191, "helv")
        centered("Universiteit van Amsterdam", axis, 202, "helv")
        centered(mail, axis, 213, "cour")
    texts = [block.text for block in extract_blocks(doc)]
    assert texts.count("Machine Learning Group Universiteit van Amsterdam") == 2
    for row in ("Diederik P. Kingma", "dpkingma@gmail.com", "Max Welling", "welling.max@gmail.com"):
        assert row in texts


def test_a_centered_name_list_wrapping_at_commas_stays_one_unit() -> None:
    """A centered author list wraps at commas and PyMuPDF splits it into several blocks. A next line starting with a
    capital still continues it; a name left alone on a line would be transliterated."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_continuations import _merge_continuations

    def row(text: str, x0: float, y0: float, x1: float) -> TextBlock:
        rect = pymupdf.Rect(x0, y0, x1, y0 + 10)
        span: PdfDict = {"text": text, "font": "NimbusRomNo9L-Medi", "flags": 20, "size": 10.0}
        return TextBlock(
            page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000", bold=True, span_lines=[[span]]
        )

    rows = [
        row("Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,", 133.9, 179.7, 478.1),
        row("Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,", 137.8, 190.6, 474.2),
        row("William El Sayed", 269.0, 201.5, 343.0),
    ]
    [merged] = _merge_continuations(rows)
    assert merged.text.endswith("Lacroix, William El Sayed")
    # Two left-aligned lines are not centered
    left = [
        row("Results are listed in the table below, for each model,", 108, 300, 380),
        row("Accuracy improves with scale.", 108, 311, 250),
    ]
    assert len(_merge_continuations(left)) == 2


def test_a_formula_wrapping_after_an_operator_keeps_its_tail() -> None:
    """A line ending at > is continued by 100) at the start of the next line, the second half of an expression, not
    a number or a symbol line. A display formula whose numerator starts right of the equals sign is still split;
    numbers after ===Instructions=== are still a list."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y: float, x1: float = 504) -> LineItem:
        span: PdfDict = {"text": text, "size": 10.0, "font": "Times-Roman", "flags": 0}
        return (text, pymupdf.Rect(x0, y, x1, y + 10), [span], [span])

    wrapped = [
        item("we can still evaluate the predicate (l_quantity*l_extendedprice >", 72, 100),
        item("100) without the columns l_quantity and l_extendedprice.", 72, 111, 330),
    ]
    assert _split_lines(wrapped) == [[0, 1]]
    item_end = [item("one predicate of this type: (l_quantity*l_extendedprice >", 90, 100), item("100).", 90, 111, 112)]
    assert _split_lines(item_end) == [[0, 1]]
    display = [item("cos(cherry, information) =", 147, 140, 280), item("442 * 5 + 8 * 3982 + 2 * 3325", 300, 130, 400)]
    assert _split_lines(display) == [[0], [1]]
    marker = [item("===Instructions===", 72, 100, 160), item("1. Carefully compare the answers.", 72, 111, 300)]
    assert _split_lines(marker) == [[0], [1]]


def test_a_justified_row_with_an_italic_variable_stays_one_row() -> None:
    """PyMuPDF splits a justified line by words, one piece holding the italic variable AVG(E). It is still one line and
    the paragraph is not split in the middle."""
    from app.formats.pdf_split_lines import _split_lines

    def piece(x0: float, x1: float, y: float, *spans: tuple[str, str]) -> LineItem:
        text = "".join(t for t, _ in spans)
        parts: list[PdfDict] = [{"text": t, "font": f, "size": 10.0, "flags": 0} for t, f in spans]
        return (text, pymupdf.Rect(x0, y, x1, y + 12), parts, parts)

    roman = "Times-Roman"
    items = [
        piece(54.0, 293.9, 560.5, ("tain an output column that matches exactly (taking into account", roman)),
        piece(54.0, 81.0, 570.6, ("column", roman)),
        piece(97.4, 149.1, 570.6, ("equivalences).", roman)),
        piece(165.4, 196.3, 570.6, ("AVG(", roman), ("E", "Times-Italic"), (")", roman)),
        piece(212.7, 218.7, 570.6, ("is", roman)),
        piece(235.0, 270.4, 570.6, ("converted", roman)),
        piece(286.8, 293.8, 570.6, ("to", roman)),
        piece(54.0, 136.2, 580.5, ("SUM(", roman), ("E", "Times-Italic"), (")/count_big(*).", roman)),
    ]
    assert _split_lines(items) == [list(range(len(items)))]


def test_a_line_opening_with_a_closing_bracket_continues_the_row() -> None:
    """A monospace [ split off at the line end, with the next line continuing with ]). This is not a symbol line
    followed by a new paragraph."""
    from app.formats.pdf_split_lines import _split_lines

    def piece(x0: float, x1: float, y: float, *spans: tuple[str, str]) -> LineItem:
        text = "".join(t for t, _ in spans)
        parts: list[PdfDict] = [{"text": t, "font": f, "size": 10.0, "flags": 0} for t, f in spans]
        return (text, pymupdf.Rect(x0, y, x1, y + 10), parts, parts)

    roman, mono = "NimbusRomNo9L-Regu", "txtt"
    items = [
        piece(
            160.2, 493.2, 280, ("the . to mean period and not the wildcard. By contrast, the regular expression", roman)
        ),
        piece(
            160.2,
            493.2,
            292,
            ('r"The dog.$"', mono),
            (" would match The dog. but also The dog! and The dogo. As", roman),
        ),
        piece(
            160.2,
            461.8,
            304,
            ("we’ll discuss below, all the special characters we’ve defined so far (", roman),
            ("* + ?", mono),
        ),
        piece(472.3, 477.5, 304, (".", mono)),
        piece(488.0, 493.2, 304, ("[", mono)),
        piece(160.2, 408.8, 316, ("]", mono), (") need to be backslashed when we mean to use them literally.", roman)),
    ]
    assert _split_lines(items) == [list(range(len(items)))]


def test_a_figure_row_pressed_onto_the_caption_is_its_own_row() -> None:
    """An operation line in a figure overlaps the first caption line, with ink boxes overlapping by more than half,
    the same size and baselines 4pt apart: two separate lines, not merged with the caption. The small numerator and
    denominator of an inline fraction are on other baselines, and that line is still one line."""
    from app.formats.pdf_rows import _left_to_right
    from app.formats.pdf_split_lines import _split_lines

    def item(
        text: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        base: float,
        font: str = "NimbusRomNo9L-Regu",
        size: float = 9.0,
    ) -> LineItem:
        span: PdfDict = {"text": text, "size": size, "font": font, "flags": 0, "origin": (x0, base)}
        return (text, pymupdf.Rect(x0, y0, x1, y1), [span], [span])

    items = _left_to_right(
        [
            item("d s s", 264, 600, 288, 609, 607.3, font="txtt"),
            item("i s", 301, 600, 315, 609, 607.3, font="txtt"),
            item("Figure 2.17", 143, 604, 186, 613, 611.3, font="NimbusRomNo9L-Medi"),
            item(
                "Representing the minimum edit distance between two strings as an alignment.", 196, 604, 475, 613, 611.3
            ),
            item(
                "The final row gives the operation list for converting the top string into the bottom string: d for",
                142,
                615,
                475,
                624,
                622.2,
            ),
        ]
    )
    assert [text for text, *_ in items][:2] == ["d s s", "i s"]
    assert _split_lines(items) == [[0, 1], [2, 3, 4]]

    def fraction_line(x0: float, y0: float, x1: float, y1: float, pieces: list[tuple[str, float, float]]) -> LineItem:
        spans: list[PdfDict] = [
            {"text": text, "size": size, "font": "NimbusRomNo9L-Regu", "flags": 0, "origin": (x0, base)}
            for text, size, base in pieces
        ]
        return ("".join(text for text, _, _ in pieces), pymupdf.Rect(x0, y0, x1, y1), spans, spans)

    row = [
        fraction_line(
            142, 175, 332, 189, [("word follows each other with equal probability", 10.0, 185.9), ("1", 7.4, 182.0)]
        ),
        fraction_line(
            329, 177, 475, 192, [("3", 7.4, 189.4), (" (it was trained on a training set with", 10.0, 185.9)]
        ),
    ]
    assert _split_lines(_left_to_right(row)) == [[0, 1]]
    # The two lines went into two blocks, which are not merged as one line either
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_visual_lines

    def blk(entry: LineItem) -> TextBlock:
        text, rect, spans, _all = entry
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(rect),
            line_rects=[pymupdf.Rect(rect)],
            text=text,
            size=9.0,
            color="#000",
            bold=False,
            span_lines=[spans],
        )

    ops = item("d s s i s", 264, 600, 315, 609, 607.3, font="txtt")
    caption = item(
        "Representing the minimum edit distance between two strings as an alignment.", 196, 604, 475, 613, 611.3
    )
    assert len(_merge_visual_lines([blk(ops), blk(caption)])) == 2


def test_underlined_letters_are_marked_and_a_fraction_bar_is_not() -> None:
    """A line drawn under text (the columns marked in example 6) is an underline, marking only the characters above
    it; the bar of a small inline fraction has the denominator right below and is not an underline."""
    from app.formats.pdf_blocks import _attach_ink
    from app.formats.pdf_layout import scan_page
    from app.formats.pdf_math import _mark_underlines, _PageChars

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 100), "outputs {A, D} and more", fontsize=10, fontname="tiro")
    left = 72 + pymupdf.get_text_length("outputs {", fontname="tiro", fontsize=10)
    width = pymupdf.get_text_length("A", fontname="tiro", fontsize=10)
    page.draw_line((left, 101.1), (left + width, 101.1), width=0.5)
    page.insert_text((72, 160), "a half", fontsize=10, fontname="tiro")
    page.insert_text((110, 157), "1", fontsize=7, fontname="tiro")
    page.draw_line((110, 158.2), (113.9, 158.2), width=0.4)
    page.insert_text((110, 164.5), "2", fontsize=7, fontname="tiro")
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
    _attach_ink(page, data)
    _mark_underlines(_PageChars(page), data, scan_page(page).strokes)
    spans = [span for block in data["blocks"] for line in block["lines"] for span in line["spans"]]
    assert [span["text"] for span in spans if span.get("underline")] == ["A"]
    assert "".join(span["text"] for span in spans).startswith("outputs {A, D} and more")


def test_digits_under_a_radical_bar_join_the_formula() -> None:
    """The 81 of √81 is in the body font. Text under the radical bar is marked as formula, the period is not."""
    from app.formats.pdf_blocks import _attach_ink, _span_is_math
    from app.formats.pdf_layout import scan_page
    from app.formats.pdf_math import _mark_radicands, _PageChars

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 100), "6 + 3 =", fontsize=10, fontname="tiro")
    root = 72 + pymupdf.get_text_length("6 + 3 = ", fontname="tiro", fontsize=10)
    page.insert_text((root, 100), "\u221a", fontsize=10, fontname="china-s")
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
    radical = next(
        span
        for block in data["blocks"]
        for line in block["lines"]
        for span in line["spans"]
        if "\u221a" in span["text"]
    )
    left = radical["bbox"][2]
    page.insert_text((left, 100), "81. Find the least", fontsize=10, fontname="tiro")
    digits = pymupdf.get_text_length("81", fontname="tiro", fontsize=10)
    page.draw_line((left, radical["bbox"][1] + 0.5), (left + digits, radical["bbox"][1] + 0.5), width=0.4)
    data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
    _attach_ink(page, data)
    _mark_radicands(_PageChars(page), data, scan_page(page).strokes)
    spans = [span for block in data["blocks"] for line in block["lines"] for span in line["spans"]]
    assert [span["text"] for span in spans if span.get("math")] == ["81"]
    assert any(span["text"].startswith(". Find") and not _span_is_math(span) for span in spans)


def test_a_glyph_is_measured_by_drawing_it_alone() -> None:
    """A glyph with a wrong font box (the radical of txsys) is drawn alone to measure its ink: top and bottom relative
    to the baseline, left and right relative to the origin, in font sizes. The hook and tail of italic f extend
    beyond its advance width."""
    from app.formats.pdf_blocks import _measure_glyph

    font = pymupdf.Font("tiro")
    measured = _measure_glyph(font.buffer, font.has_glyph(ord("g")))
    assert measured is not None
    top, bottom, width, left, right = measured
    assert -0.6 < top < 0 < bottom < 0.4 and width == font.glyph_advance(ord("g"))
    assert 0 <= left < right <= width + 0.03
    measured = _measure_glyph(font.buffer, font.has_glyph(ord("x")))
    assert measured is not None
    top, bottom, _width, _left, _right = measured
    assert -0.6 < top < 0 and abs(bottom) < 0.03
    italic = pymupdf.Font("tiit")
    measured = _measure_glyph(italic.buffer, italic.has_glyph(ord("f")))
    assert measured is not None
    _top, _bottom, width, left, right = measured
    assert left < -0.1 and right > width + 0.1
    assert _measure_glyph(font.buffer, font.has_glyph(ord("\u4e2d"))) is None


def test_glyphs_are_matched_by_the_glyph_id_used_on_the_page() -> None:
    """Each character is recorded by the glyph id used on the page, whitespace excluded. Fonts with a custom encoding
    (CMEX) have no glyphs to find by Unicode."""
    from app.formats.pdf_blocks import _char_origins

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "g x", fontname="tiro", fontsize=10)
    font = pymupdf.Font("tiro")
    [glyphs] = _char_origins(page).values()
    assert [gid for gid, _base, _width in glyphs] == [font.has_glyph(ord("g")), font.has_glyph(ord("x"))]
    assert all(abs(base - 100) < 0.01 for _gid, base, _width in glyphs)


def test_an_underline_is_written_as_a_mark_and_drawn_under_the_words() -> None:
    """Underlines are sent for translation as {u}, even when the whole block is underlined; the Chinese is written
    with a line below the text."""
    from app.formats.pdf_align import _place_line
    from app.formats.pdf_flow import span_emphasis, writer_for
    from app.formats.pdf_typeset import typeset_lines

    plain: PdfDict = {"text": "Now consider ", "font": "Times-Roman", "flags": 0}
    marked: PdfDict = {"text": "A", "font": "Times-Roman", "flags": 0, "underline": True}
    writer = writer_for([plain, marked])
    text = writer.text(plain["text"], span_emphasis(plain)) + writer.text("A", span_emphasis(marked)) + writer.close()
    assert text == "Now consider {u}A{/u}"
    whole = writer_for([{**marked, "text": "Step 1:"}])
    assert whole.text("Step 1:", span_emphasis(marked)) + whole.close() == "{u}Step 1:{/u}"

    line = typeset_lines("步骤 {u}A{/u} 列", 10.0, 200.0, {})[0]
    placed = _place_line(line, 0.0, 200.0, 10.0, False, "left", frozenset(), {})
    assert placed is not None
    rules = placed[3]
    assert len(rules) == 1 and 0 < rules[0][1] - rules[0][0] < 10


def test_a_value_column_beside_a_word_label_is_its_own_paragraph() -> None:
    """Labels and values in two columns: the next line of a value starts at the value's left edge. The value becomes
    its own paragraph and the label is translated alone; a hanging indent after an item number is still one item."""
    from app.formats.pdf_split_lines import _split_lines

    def item(text: str, x0: float, y0: float, x1: float) -> LineItem:
        span: PdfDict = {"text": text, "size": 9.0, "font": "Times-Roman", "flags": 0}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), [span], [span])

    ranges = [
        item("Step 3: Compute ranges.", 318, 594, 407),
        item("View ranges:", 318, 604, 366),
        item("{l_partkey, p_partkey} in (150, +inf),", 390, 604, 522),
        item("{o_custkey} in (50, 500)", 390, 614, 482),
        item("Query ranges:", 318, 627, 369),
        item("{l_partkey, p_partkey} in (150, 160),", 390, 627, 524),
        item("{o_custkey} in (123, 123)", 390, 637, 484),
    ]
    assert _split_lines(ranges) == [[0], [1], [2, 3], [4], [5, 6]]
    numbered = [
        item("2.1", 318, 604, 330),
        item("Translate a sentence from one language you know to another. Now", 345, 604, 558),
        item("tokenize both sentences with a tokenizer visualizer.", 345, 614, 520),
    ]
    assert _split_lines(numbered) == [[0, 1, 2]]


def test_a_word_label_does_not_absorb_the_value_column_block_beside_it() -> None:
    """When label and value are in two blocks, the value is a column of several lines and is not merged into the
    label; the rest of a sentence after a caption label is still merged."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_paragraphs import _merge_caption_fragments

    def blk(text: str, rects: list[tuple[float, float, float, float]]) -> TextBlock:

        rect = pymupdf.Rect(rects[0])
        for other in rects[1:]:
            rect |= other
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[pymupdf.Rect(r) for r in rects],
            text=text,
            size=9.0,
            color="#000",
            bold=False,
        )

    label = blk("View ranges:", [(318, 604, 366, 616)])
    values = blk(
        "{l_partkey, p_partkey} in (150, +inf), {o_custkey} in (50, 500)", [(390, 604, 522, 616), (390, 614, 482, 626)]
    )
    assert len(_merge_caption_fragments([label, values])) == 2
    caption = blk("Figure 2:", [(318, 604, 350, 616)])
    tail = blk("Examples of decay schedules for the learning rate", [(356, 604, 540, 616)])
    assert len(_merge_caption_fragments([caption, tail])) == 1
