import pymupdf

from app.formats.pdf import extract_blocks
from app.formats.pdf_frame import FrameLine, margin_notes, note_groups, text_frames

_LINE = "a body line of running text that is long enough to vote"


def _lines(x0: float, x1: float, count: int = 14, top: float = 100) -> list[FrameLine]:
    return [FrameLine(pymupdf.Rect(x0, top + 12 * i, x1, top + 12 * i + 10), _LINE, 10) for i in range(count)]


def test_frame_follows_body_lines_not_the_leftmost_block():
    header = FrameLine(pymupdf.Rect(77, 84, 143, 94), "4 CHAPTER 2 • WORDS AND TOKENS ...........", 8)
    even = [header, *_lines(142, 475)]
    odd = _lines(160, 493)
    frames = text_frames([odd, even, odd, even])
    assert (frames[0].x0, frames[0].x1) == (160, 493)
    assert (frames[1].x0, frames[1].x1) == (142, 475)
    assert frames[1].y0 == 100


def test_two_columns_span_both_columns():
    page = [*_lines(40, 290), *_lines(316, 560)]
    [frame] = text_frames([page])
    assert (frame.x0, frame.x1) == (40, 560)


def test_small_terms_on_a_lane_beside_the_frame_are_notes():
    body = _lines(142, 475)
    notes = [
        FrameLine(pymupdf.Rect(84, 161, 128, 169), "Herdan's Law", 7.2),
        FrameLine(pymupdf.Rect(91, 173, 128, 180), "Heaps' Law", 7.2),
        FrameLine(pymupdf.Rect(82, 281, 128, 288), "function words", 7.2),
        # 页码、外凸的节号和正文一样大，不是旁注
        FrameLine(pymupdf.Rect(77, 84, 82, 94), "4", 10),
        FrameLine(pymupdf.Rect(77, 300, 99, 315), "2.1", 17),
    ]
    page = [*body, *notes]
    frames = text_frames([page])
    found = margin_notes([page], frames)[0]
    assert found == {len(body), len(body) + 1, len(body) + 2}


def test_two_line_note_is_one_note_and_neighbours_stay_apart():
    lines = [
        FrameLine(pymupdf.Rect(123, 623, 146, 630), "regular", 7.2),
        FrameLine(pymupdf.Rect(111, 630, 146, 637), "expressions", 7.2),
        FrameLine(pymupdf.Rect(84, 661, 146, 669), "Herdan's Law", 7.2),
        FrameLine(pymupdf.Rect(91, 673, 146, 680), "Heaps' Law", 7.2),
    ]
    assert note_groups(lines) == [[0, 1], [2], [3]]


def test_margin_note_on_a_body_line_is_split_off():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    words = ("The body of this paragraph keeps going across many lines so that the "
             "frame has enough long lines to vote on where the text starts ").split()
    y = 120
    lines = []
    for i in range(16):
        lines.append(" ".join(words[(i * 3) % 20:(i * 3) % 20 + 9]))
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


def test_label_of_a_caption_wider_than_the_body_is_not_a_margin_note():
    """宽表的图注从页边起笔，标签落在旁注的位置。下一行图注从同一处起笔、写进版心，
    标签留在图注里。真正的旁注照旧拆出来。"""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    words = ("The body of this paragraph keeps going across many lines so that the "
             "frame has enough long lines to vote on where the text starts ").split()
    for i in range(16):
        page.insert_text((160, 120 + 12 * i), " ".join(words[(i * 3) % 20:(i * 3) % 20 + 9]), fontsize=10)
    for row, note in ((2, "tokenization"), (6, "regular"), (10, "edit distance")):
        page.insert_text((96, 120 + 12 * row), note, fontsize=7)
    page.insert_text((96, 330), "Figure 3.7", fontsize=9, fontname="hebo")
    page.insert_text((150, 330), "Add-one smoothed bigram probabilities for eight of the words in the", fontsize=9)
    page.insert_text((96, 341), "corpus of 9332 sentences. Previously-zero probabilities are in gray.", fontsize=9)
    blocks = extract_blocks(doc)
    assert [block.text for block in blocks if block.side_note] == ["tokenization", "regular", "edit distance"]
    [caption] = [block for block in blocks if "Add-one" in block.text]
    assert caption.text.startswith("Figure 3.7") and "corpus of 9332" in caption.text


def test_displayed_formula_with_a_big_brace_keeps_its_rows():
    from app.formats.pdf import TextBlock, _display_formula_ids
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import PageGeometry

    def block(text, rect, font="NimbusRomNo9L-Regu"):
        span = {"text": text, "font": font, "size": 10.0}
        return TextBlock(page=0, rect=pymupdf.Rect(rect), line_rects=[pymupdf.Rect(rect)], text=text,
                         size=10, color="#000000", bold=False, span_lines=[[span]])

    brace = block("D[i−1, j −1]+ 2; if source[i] ≠ target[j]", (237, 408, 432, 432), "CMEX10")
    condition = block("0; if source[i] = target[j]", (325, 420, 432, 430))
    body = block("Under this version of Levenshtein, the computation for D[i, j] becomes:", (142, 380, 475, 390))
    inline = block("a sum written inline uses the extension font too", (142, 440, 475, 450), "CMEX10")
    geo = PageGeometry.from_rects(612, [], [], 792, frame=TextFrame(142, 475, 111, 719, 10))
    # 列表续行、右栏的段落也离开左缘，句中的积分号用同一种字体，按词数认成句子
    item = TextBlock(
        page=0, rect=pymupdf.Rect(170, 460, 475, 500), line_rects=[pymupdf.Rect(170, 460, 475, 500)],
        text="", size=10, color="#000000", bold=False,
        span_lines=[[{"text": "R", "font": "CMEX10", "size": 10.0},
                     {"text": "pθ(z)pθ(x|z) dz is intractable (so we cannot evaluate or differentiate the marginal "
                              "likelihood), where the true posterior density is intractable (so the EM algorithm "
                              "cannot be used), and where the required integrals",
                      "font": "NimbusRomNo9L-Regu", "size": 10.0}]],
    )
    # 行内的宽帽子、小根号也是这种字体，不说明是单独排的式子
    accent = TextBlock(
        page=0, rect=pymupdf.Rect(300, 520, 475, 540), line_rects=[pymupdf.Rect(300, 520, 475, 540)],
        text="", size=10, color="#000000", bold=False,
        span_lines=[[{"text": "Let the normalized values be x", "font": "NimbusRomNo9L-Regu", "size": 10.0},
                     {"text": "b", "font": "CMEX10", "size": 10.0},
                     {"text": ", and their transformations be y. We refer to the transform", "font": "NimbusRomNo9L-Regu",
                      "size": 10.0}]],
    )
    ids = _display_formula_ids([brace, condition, body, inline, item, accent], [geo])
    assert ids == {id(brace), id(condition)}


def test_body_cut_at_a_margin_note_is_joined_back():
    from app.formats.pdf import _rejoin_note_cuts

    def line(text, x0, y0, x1, size=10.0):
        return {"dir": (1, 0), "bbox": (x0, y0, x1, y0 + size),
                "spans": [{"text": text, "size": size, "font": "Times", "flags": 0, "color": 0}]}

    upper = {"type": 0, "lines": [line("Starting in the 1960s, the characters (like the ones used", 160, 341, 493),
                                  line("in this sentence), were represented with ASCII (American Standard", 160, 353, 493)]}
    lower = {"type": 0, "lines": [line("Code for Information Interchange). ASCII used one byte.", 160, 365, 493)]}
    after = {"type": 0, "lines": [line("Chinese has about 100,000 characters in Unicode.", 175, 377, 493)]}
    data = {"blocks": [upper, lower, after]}
    _rejoin_note_cuts(data, {id(upper)})
    assert len(upper["lines"]) == 3 and lower["lines"] == []
    assert after["lines"]  # 上一行以句末收尾、下一块缩进：新的一段


def test_a_bold_caption_label_starts_its_own_paragraph():
    """宽图的图注比图里的字靠左不到三个字宽，像缩进。标签单独排成粗体，是图注的第一行。
    正文里以 Figure 3.8 开头的句子字体不变，不拆。"""
    from app.formats.pdf import _split_lines

    def item(*spans, x0, y0):
        parts = [{"text": text, "size": 9.0 if font.endswith("Medi") else 8.0, "font": font, "flags": 0}
                 for text, font in spans]
        width = sum(len(text) for text, _font in spans) * 4.5
        return ("".join(text for text, _font in spans), pymupdf.Rect(x0, y0, x0 + width, y0 + 9), parts)

    figure = [
        item(("anchorTimeID=” t1 ”> the weekend </TIMEX3>, marking", "Courier"), x0=98, y0=147),
        item(("weeks </TIMEX3>.", "Courier"), x0=98, y0=166),
        item(("Figure 21.17", "Times-Medi"), (" TimeML markup including normalized values.", "Times-Roman"),
             x0=78, y0=176),
    ]
    assert _split_lines(figure) == [[0, 1], [2]]
    body = [
        item(("We can see the counts of all bigrams in the corpus", "Times-Roman"), x0=98, y0=147),
        item(("Figure 3.8 shows the reconstructed counts.", "Times-Roman"), x0=78, y0=158),
    ]
    assert _split_lines(body) == [[0, 1]]


def test_typewriter_dialogue_keeps_one_line_per_turn():
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1, font="txtt", flags=8):
        span = {"text": text, "size": 9.0, "font": font, "flags": flags}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 9), [span])

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


def test_prose_after_typewriter_lines_starts_a_new_paragraph():
    """等宽字排的小号列表项后面紧跟正文首行。首行缩进和列表项的左缘只差一点，也要拆开。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1, size, font, flags):
        span = {"text": text, "size": size, "font": font, "flags": flags}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + size), [span])

    items = [
        item("illustrate its explanations with examples, thought experiments", 167, 577, 489, 8.0, "txtt", 12),
        item("self-destructive behavior", 167, 600, 272, 8.0, "txtt", 12),
        item("Prompts can also be used in more sophisticated ways in a method", 157, 611, 475, 10.0,
             "NimbusRomNo9L-Regu", 4),
        item("time compute. One representative example is chain-of-thought", 142, 623, 475, 10.0,
             "NimbusRomNo9L-Regu", 4),
    ]
    assert _split_lines(items) == [[0, 1], [2, 3]]


def test_a_larger_line_below_a_slide_formula_starts_a_new_paragraph():
    """幻灯片里缩进的式子下面接着大一号的小标题，左缘只差一个多字宽，像首行缩进，要拆开。
    小型大写的词后面回到正文字号、接着写的半句不拆。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1, size, font="TimesNewRomanPSMT"):
        span = {"text": text, "size": size, "font": font, "flags": 0}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + size * 1.2), [span])

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


def test_blocks_in_a_flagged_monospace_font_are_code():
    """txtt 这类字体名字里没有 Mono，只有 PDF 的等宽标志。整块是这种字就留原文。"""
    from app.formats.pdf import TextBlock, _code_block

    def block(*spans):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 100, 10), line_rects=[], text="".join(t for t, _, _ in spans),
            size=10, color="#000", bold=False,
            span_lines=[[{"text": t, "font": f, "flags": flags, "size": 10} for t, f, flags in spans]],
        )

    assert _code_block(block(("<system> You are a helpful assistant.", "txtt", 12)))
    assert not _code_block(block(("The prompt ", "NimbusRomNo9L-Regu", 4), ("<system>", "txtt", 12),
                                 (" comes first in the context window.", "NimbusRomNo9L-Regu", 4)))
    # 汉字字体也常带等宽标志，不算代码
    assert not _code_block(block(("系统提示词放在上下文的最前面。", "SimSun", 8)))


def test_gloss_line_is_split_from_its_free_translation():
    from app.formats.pdf import TextBlock, _gloss_ids, _split_gloss_lines

    lines = [pymupdf.Rect(174, 400, 399, 410), pymupdf.Rect(174, 414, 389, 424)]
    spans = [[{"text": "1SG.S-E-midnight-big-E-sew-1SG.S[PFV]", "size": 10.0}],
             [{"text": "“I sewed a lot of yurt covers.”", "size": 10.0}]]
    block = TextBlock(page=0, rect=lines[0] | lines[1], line_rects=lines, text="", size=10,
                      color="#000000", bold=False, span_lines=spans)
    glosses = {0: (pymupdf.Rect(174, 388, 399, 410),)}
    gloss, free = _split_gloss_lines([block], glosses)
    assert gloss.text.startswith("1SG.S") and free.text.startswith("“I sewed")
    assert _gloss_ids([gloss, free], glosses) == {id(gloss)}
