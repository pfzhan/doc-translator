import itertools
import string

import pymupdf

from app.formats.pdf import extract_blocks
from app.formats.pdf_frame import FrameLine, margin_notes, note_groups, text_frames

_LINE = "a body line of running text that is long enough to vote"
# 正文每行的字都不同；各页同一高度、同一句话的是页眉
_TAGS = ("".join(pair) for pair in itertools.product(string.ascii_lowercase, repeat=3))


def _lines(x0: float, x1: float, count: int = 14, top: float = 100) -> list[FrameLine]:
    return [
        FrameLine(pymupdf.Rect(x0, top + 12 * i, x1, top + 12 * i + 10), f"{_LINE} {next(_TAGS)}", 10)
        for i in range(count)
    ]


def test_frame_follows_body_lines_not_the_leftmost_block():
    header = FrameLine(pymupdf.Rect(77, 84, 143, 94), "4 CHAPTER 2 • WORDS AND TOKENS ...........", 8)
    frames = text_frames([_lines(160, 493), [header, *_lines(142, 475)], _lines(160, 493), [header, *_lines(142, 475)]])
    assert (frames[0].x0, frames[0].x1) == (160, 493)
    assert (frames[1].x0, frames[1].x1) == (142, 475)
    assert frames[1].y0 == 100


def test_a_running_header_set_like_body_text_is_outside_the_frame():
    """页眉和正文一样大、一样长，每页同一高度同一句话，不算版心的第一行。"""
    header = "Published as a conference paper at ICLR 2015"
    pages = [[FrameLine(pymupdf.Rect(105, 61, 285, 73), header, 10), *_lines(108, 504, top=116)] for _ in range(4)]
    assert all(frame.y0 == 116 for frame in text_frames(pages))


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


def test_the_short_last_line_of_an_example_does_not_open_the_next_paragraph():
    """例句的悬挂续行没排满、以句号收尾，下一段从栏边起笔，只左跳一个多字宽，不是首行缩进。
    首行缩进的段落第一行排满，照样接着。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y0, x1):
        span = {"text": text, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0}
        return (text, pymupdf.Rect(x0, y0, x1, y0 + 12), [span])

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
    # 块从上一条的续行起头，第一行也缩着。两行缩进之后回到栏边的是下一条
    mid_item = [
        item("was there a task? Was the language originally spoken conversation, edited", 167, 335, 475),
        item("text, social media communication, monologue vs. dialogue?", 167, 347, 407),
        item("Language variety: What language (including dialect/region) was the corpus in?", 147, 362, 468),
    ]
    assert _split_lines(mid_item) == [[0, 1], [2]]


def test_a_bold_label_after_a_finished_sentence_opens_the_next_item():
    """定义列表里一行一条，排得差不多满，按硬换行看不出来。上一句说完、下一行以粗体标签起头的是下一条。
    句中折行落到行首的粗体词不拆。"""
    from app.formats.pdf import _split_lines

    def item(label, rest, y0, x1):
        spans = [{"text": label, "size": 10.0, "font": "NimbusRomNo9L-Medi", "flags": 16}] if label else []
        spans.append({"text": rest, "size": 10.0, "font": "NimbusRomNo9L-Regu", "flags": 0})
        return (label + rest, pymupdf.Rect(147, y0, x1, y0 + 10), spans)

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


def test_a_heading_across_a_narrow_gutter_is_not_joined_to_the_left_line():
    """左栏一行和右栏的节号标题同高，中间只隔一个字宽的栏沟。节号从右栏栏边起笔，不是左边那行的后半截。"""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    left = ["BN-Baseline: Same as Inception with Batch Normal-", "ization before each nonlinearity."]
    for i, text in enumerate(left):
        page.insert_text((72, 320 + 12 * i), text, fontsize=10)
    page.insert_text((311, 318), "4.2.3", fontsize=10, fontname="hebo")
    page.insert_text((352, 318), "Ensemble Classification", fontsize=10, fontname="hebo")
    for i in range(6):
        page.insert_text((311, 340 + 12 * i), f"The current reported best results on ImageNet line {i}", fontsize=10)
    from app.formats.pdf import _rejoin_split_lines

    data = page.get_text("dict")
    _rejoin_split_lines(data)
    left_block = next(block for block in data["blocks"] if "Normal-" in block["lines"][0]["spans"][0]["text"])
    assert ["".join(span["text"] for span in line["spans"]) for line in left_block["lines"]] == left


def test_an_inline_enumeration_wrapping_onto_a_new_line_is_not_a_list():
    """图注里 (a) 写在行中间，(b) 碰巧落到行首、上一行排满：是同一段的折行。各项都在行首的是列表。"""
    from app.formats.pdf import _split_lines

    def item(text, y, x1=504):
        span = {"text": text, "size": 9.0, "font": "Times-Roman", "flags": 0}
        return (text, pymupdf.Rect(108, y, x1, y + 10), [span])

    caption = [
        item("transformed samples. G contracts in regions of high density and expands in regions of low. (a)", 524),
        item("Consider an adversarial pair near convergence: pg is similar to pdata and D is a partially accurate.", 534),
        item("(b) In the inner loop of the algorithm D is trained to discriminate samples from data, converging", 544),
    ]
    assert _split_lines(caption) == [[0, 1, 2]]
    listed = [item("(a) Consider an adversarial pair near convergence.", 524, 330),
              item("(b) In the inner loop of the algorithm D is trained.", 534, 330)]
    assert _split_lines(listed) == [[0], [1]]


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
    # LaTeX times 宏包的等宽字是 URW 的 NimbusMonL，名字里没有 Mono，也不一定带等宽标志
    assert _code_block(block(("dpkingma@gmail.com", "NimbusMonL-Regu", 4)))
    # 语料表一行：行首的计数用正文字体，后面是等宽字的词。数字不算正文
    assert _code_block(block(("2", "NimbusRomNo9L-Regu", 4), ("n", "txtt", 12), (" e w", "txtt", 12)))


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


def test_centered_author_rows_stay_one_item_per_row():
    """作者栏居中排：粗体名字、常规字体的单位、等宽字的邮箱各占一行。换了字体的下一行是另一项，
    不和上一行并成一段。同一字体折成两行的单位仍是一项。"""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    def centered(text, axis, y, font):
        width = pymupdf.get_text_length(text, fontname=font, fontsize=10)
        page.insert_text((axis - width / 2, y), text, fontsize=10, fontname=font)

    for axis, name, mail in ((215, "Diederik P. Kingma", "dpkingma@gmail.com"),
                             (390, "Max Welling", "welling.max@gmail.com")):
        centered(name, axis, 180, "hebo")
        centered("Machine Learning Group", axis, 191, "helv")
        centered("Universiteit van Amsterdam", axis, 202, "helv")
        centered(mail, axis, 213, "cour")
    texts = [block.text for block in extract_blocks(doc)]
    assert texts.count("Machine Learning Group Universiteit van Amsterdam") == 2
    for row in ("Diederik P. Kingma", "dpkingma@gmail.com", "Max Welling", "welling.max@gmail.com"):
        assert row in texts


def test_a_centered_name_list_wrapping_at_commas_stays_one_unit():
    """作者名单居中排、在逗号处折行，PyMuPDF 分成了几块。下一行大写起头也是接着写的，
    拆开后单独一行的名字会被音译。"""
    from app.formats.pdf import TextBlock, _merge_continuations

    def row(text, x0, y0, x1):
        rect = pymupdf.Rect(x0, y0, x1, y0 + 10)
        span = {"text": text, "font": "NimbusRomNo9L-Medi", "flags": 20, "size": 10.0}
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000",
                         bold=True, span_lines=[[span]])

    rows = [row("Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,", 133.9, 179.7, 478.1),
            row("Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,", 137.8, 190.6, 474.2),
            row("William El Sayed", 269.0, 201.5, 343.0)]
    [merged] = _merge_continuations(rows)
    assert merged.text.endswith("Lacroix, William El Sayed")
    # 左对齐的两行不算居中排
    left = [row("Results are listed in the table below, for each model,", 108, 300, 380),
            row("Accuracy improves with scale.", 108, 311, 250)]
    assert len(_merge_continuations(left)) == 2


def test_a_formula_wrapping_after_an_operator_keeps_its_tail():
    """行尾停在 > 上，下一行行首的 100) 是式子的后半截，不是编号也不是符号行。
    单独排的式子里分子从等号右边起笔，仍拆开；===Instructions=== 后面的编号照常是列表。"""
    from app.formats.pdf import _split_lines

    def item(text, x0, y, x1=504):
        span = {"text": text, "size": 10.0, "font": "Times-Roman", "flags": 0}
        return (text, pymupdf.Rect(x0, y, x1, y + 10), [span])

    wrapped = [item("we can still evaluate the predicate (l_quantity*l_extendedprice >", 72, 100),
               item("100) without the columns l_quantity and l_extendedprice.", 72, 111, 330)]
    assert _split_lines(wrapped) == [[0, 1]]
    item_end = [item("one predicate of this type: (l_quantity*l_extendedprice >", 90, 100),
                item("100).", 90, 111, 112)]
    assert _split_lines(item_end) == [[0, 1]]
    display = [item("cos(cherry, information) =", 147, 140, 280),
               item("442 * 5 + 8 * 3982 + 2 * 3325", 300, 130, 400)]
    assert _split_lines(display) == [[0], [1]]
    marker = [item("===Instructions===", 72, 100, 160), item("1. Carefully compare the answers.", 72, 111, 300)]
    assert _split_lines(marker) == [[0], [1]]


def test_a_justified_row_with_an_italic_variable_stays_one_row():
    """两端对齐的一行被 PyMuPDF 按词拆开，其中一截夹着斜体的变量 AVG(E)。仍是同一行，不从中间拆段。"""
    from app.formats.pdf import _split_lines

    def piece(x0, x1, y, *spans):
        text = "".join(t for t, _ in spans)
        return (text, pymupdf.Rect(x0, y, x1, y + 12),
                [{"text": t, "font": f, "size": 10.0, "flags": 0} for t, f in spans])

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


def test_a_line_opening_with_a_closing_bracket_continues_the_row():
    """行尾单拆出来一个等宽的 [，下一行以 ]) 起头接着写。不是符号行后面另起一段。"""
    from app.formats.pdf import _split_lines

    def piece(x0, x1, y, *spans):
        text = "".join(t for t, _ in spans)
        return (text, pymupdf.Rect(x0, y, x1, y + 10),
                [{"text": t, "font": f, "size": 10.0, "flags": 0} for t, f in spans])

    roman, mono = "NimbusRomNo9L-Regu", "txtt"
    items = [
        piece(160.2, 493.2, 280, ("the . to mean period and not the wildcard. By contrast, the regular expression", roman)),
        piece(160.2, 493.2, 292, ("r\"The dog.$\"", mono), (" would match The dog. but also The dog! and The dogo. As", roman)),
        piece(160.2, 461.8, 304, ("we’ll discuss below, all the special characters we’ve defined so far (", roman),
              ("* + ?", mono)),
        piece(472.3, 477.5, 304, (".", mono)),
        piece(488.0, 493.2, 304, ("[", mono)),
        piece(160.2, 408.8, 316, ("]", mono), (") need to be backslashed when we mean to use them literally.", roman)),
    ]
    assert _split_lines(items) == [list(range(len(items)))]


def test_a_figure_row_pressed_onto_the_caption_is_its_own_row():
    """图里的操作行压在图注第一行上，墨迹框交叠过半，同字号、基线差四磅，是上下两行，不和图注并成一行。
    行内分式的小号分子分母在别的基线上，那一行照样是一行。"""
    from app.formats.pdf import _left_to_right, _split_lines

    def item(text, x0, y0, x1, y1, base, font="NimbusRomNo9L-Regu", size=9.0):
        span = {"text": text, "size": size, "font": font, "flags": 0, "origin": (x0, base)}
        return (text, pymupdf.Rect(x0, y0, x1, y1), [span])

    items = _left_to_right([
        item("d s s", 264, 600, 288, 609, 607.3, font="txtt"),
        item("i s", 301, 600, 315, 609, 607.3, font="txtt"),
        item("Figure 2.17", 143, 604, 186, 613, 611.3, font="NimbusRomNo9L-Medi"),
        item("Representing the minimum edit distance between two strings as an alignment.", 196, 604, 475, 613, 611.3),
        item("The final row gives the operation list for converting the top string into the bottom string: d for",
             142, 615, 475, 624, 622.2),
    ])
    assert [text for text, *_ in items][:2] == ["d s s", "i s"]
    assert _split_lines(items) == [[0, 1], [2, 3, 4]]

    def fraction_line(x0, y0, x1, y1, pieces):
        spans = [{"text": text, "size": size, "font": "NimbusRomNo9L-Regu", "flags": 0, "origin": (x0, base)}
                 for text, size, base in pieces]
        return ("".join(text for text, _, _ in pieces), pymupdf.Rect(x0, y0, x1, y1), spans)

    row = [
        fraction_line(142, 175, 332, 189, [("word follows each other with equal probability", 10.0, 185.9),
                                           ("1", 7.4, 182.0)]),
        fraction_line(329, 177, 475, 192, [("3", 7.4, 189.4), (" (it was trained on a training set with", 10.0, 185.9)]),
    ]
    assert _split_lines(_left_to_right(row)) == [[0, 1]]
    # 两行分进了两块，块之间也不按同一行并
    from app.formats.pdf import TextBlock, _merge_visual_lines

    def blk(entry):
        text, rect, spans = entry
        return TextBlock(page=0, rect=pymupdf.Rect(rect), line_rects=[pymupdf.Rect(rect)], text=text, size=9.0,
                         color="#000", bold=False, span_lines=[spans])

    ops = item("d s s i s", 264, 600, 315, 609, 607.3, font="txtt")
    caption = item("Representing the minimum edit distance between two strings as an alignment.", 196, 604, 475, 613, 611.3)
    assert len(_merge_visual_lines([blk(ops), blk(caption)])) == 2
