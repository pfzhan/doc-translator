from pathlib import Path

import pymupdf

from app.formats.pdf import TextBlock, _cross_page_units
from app.formats.pdf_layout import PageGeometry
from app.formats.pdf_roles import HeuristicLayout, LayoutItem, Role


def _item(text: str, rect: pymupdf.Rect, size: float = 10, bold: bool = False, page: int = 0) -> LayoutItem:
    return LayoutItem(page, rect, text, size, bold)


def _geo(figures: list[pymupdf.Rect], width: float = 400, height: float = 500) -> PageGeometry:
    return PageGeometry.from_rects(width, [pymupdf.Rect(40, 40, 180, 54)], figures, height)


def test_size_heading_stays_a_heading_and_author_stays_body():
    items = [
        _item("Introduction", pymupdf.Rect(40, 40, 160, 54), size=12, bold=True),
        _item("Jakob Uszkoreit", pymupdf.Rect(40, 70, 160, 82), size=10, bold=True),
        _item("A body sentence long enough to stay a paragraph.", pymupdf.Rect(40, 100, 300, 114)),
    ]
    roles = HeuristicLayout().classify(items, [_geo([])], median=10)
    assert roles == [Role.HEADING, Role.BODY, Role.BODY]


def test_numbered_title_is_a_heading_but_a_quantity_is_not():
    items = [
        _item("3.1 Encoder", pymupdf.Rect(40, 40, 140, 54)),
        _item("1 引言", pymupdf.Rect(40, 60, 120, 74)),
        _item("2 samples", pymupdf.Rect(40, 80, 140, 94)),
        _item("2 Samples were collected today.", pymupdf.Rect(40, 100, 280, 114)),
    ]
    roles = HeuristicLayout().classify(items, [_geo([])], median=10)
    assert roles == [Role.HEADING, Role.HEADING, Role.BODY, Role.BODY]


def test_text_inside_a_figure_is_a_figure_and_a_caption_is_not():
    figure = pymupdf.Rect(80, 80, 220, 180)
    items = [
        _item("axis", pymupdf.Rect(100, 110, 130, 122), size=8),
        _item("Figure 1: a plot.", pymupdf.Rect(80, 184, 200, 198)),
        _item("Figure 1 shows the plot in detail.", pymupdf.Rect(80, 184, 280, 198)),
        _item("Body paragraph under the caption stays body text.", pymupdf.Rect(40, 220, 300, 234)),
    ]
    roles = HeuristicLayout().classify(items, [_geo([figure])], median=10)
    assert roles == [Role.FIGURE, Role.CAPTION, Role.BODY, Role.BODY]


def test_a_caption_on_the_figure_ground_stays_a_caption():
    """画图程序的白底连图注一起包住时，以图注标签起头的仍是图注，标签后面可以是竖线。"""
    ground = pymupdf.Rect(60, 60, 340, 260)
    items = [
        _item("Rollout", pymupdf.Rect(100, 100, 140, 110), size=7),
        _item("Figure 5 | Overview of our RL framework.", pymupdf.Rect(120, 240, 300, 252)),
    ]
    roles = HeuristicLayout().classify(items, [_geo([ground])], median=10)
    assert roles == [Role.FIGURE, Role.CAPTION]


def test_page_filling_image_does_not_swallow_the_text_layer():
    page = pymupdf.Rect(0, 0, 400, 500)
    items = [_item("Recognized line of the scanned page.", pymupdf.Rect(40, 80, 300, 94))]
    roles = HeuristicLayout().classify(items, [_geo([page], height=500)], median=10)
    assert roles == [Role.BODY]


def test_cross_page_does_not_merge_a_caption():
    left = TextBlock(
        page=0, rect=pymupdf.Rect(40, 700, 200, 714), line_rects=[],
        text="the model performs quite", size=10, color="#000", bold=False,
    )
    caption = TextBlock(
        page=1, rect=pymupdf.Rect(40, 40, 180, 54), line_rects=[],
        text="examples of the plot", size=10, color="#000", bold=False,
    )
    units = _cross_page_units([left, caption], 10.0, halt_ids={id(caption)})
    assert all(len(unit) == 1 for unit in units)


def test_figure_label_stays_original_and_caption_is_translated(tmp_path: Path):
    from app.formats import translate_file
    from tests.test_formats import run, runner

    src = tmp_path / "fig.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 16))
    pix.set_rect(pix.irect, (180, 180, 180))
    page.insert_image(pymupdf.Rect(80, 70, 220, 160), pixmap=pix)
    page.insert_text((100, 120), "axis", fontsize=9)
    page.insert_text((80, 176), "Figure 1: a plot.", fontsize=10)
    page.insert_text((40, 40), "Body paragraph with enough letters here.", fontsize=11)
    doc.save(src)
    doc.close()

    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    # 正文下面空着，译文长了就折到下一行，不缩字号
    text = " ".join(pymupdf.open(out)[0].get_text().split())
    assert "axis" in text and "[zh-CN] axis" not in text
    assert "[zh-CN] Figure 1: a plot." in text
    assert "[zh-CN] Body paragraph with enough letters here." in text


def test_labels_and_listing_above_a_figure_caption_are_the_figure():
    """picture 环境用字形画线，没有矢量框。图注上方、到上一段正文为止都是图。
    两张图叠在一起时，上一张的图注是下一张图的上界，不被吞进去。"""
    items = [
        _item("A body paragraph that ends right before the figure starts.", pymupdf.Rect(40, 40, 360, 54)),
        _item("psc 6M", pymupdf.Rect(170, 70, 210, 80)),
        _item("S = S union {v};", pymupdf.Rect(80, 90, 200, 100)),
        _item("Figure 1: The eight views.", pymupdf.Rect(120, 120, 300, 132)),
        _item("none 1", pymupdf.Rect(170, 150, 210, 160)),
        _item("Figure 2: Hierarchy of time attributes", pymupdf.Rect(120, 180, 320, 192)),
        _item("Next body paragraph starts at the left margin again.", pymupdf.Rect(40, 210, 360, 224)),
    ]
    geo = PageGeometry.from_rects(400, [item.rect for item in items], [], 500)
    roles = HeuristicLayout().classify(items, [geo], median=10)
    assert roles == [Role.BODY, Role.FIGURE, Role.FIGURE, Role.CAPTION, Role.FIGURE, Role.CAPTION, Role.BODY]


def test_body_starting_right_of_outdented_headings_bounds_the_figure():
    """版心左缘在 160：页眉和外凸的节标题更靠左，正文段落仍是图区的上界。"""
    from app.formats.pdf_frame import TextFrame

    items = [
        _item("2.3 • UNICODE", pymupdf.Rect(398, 84, 476, 94), size=8),
        _item("2.3 Unicode", pymupdf.Rect(95, 218, 192, 235), size=17),
        _item("Another option we could consider is the level of the individual character.",
              pymupdf.Rect(160, 257, 493, 315)),
        _item("Ch Hex Dec", pymupdf.Rect(163, 438, 221, 448)),
        _item("Figure 2.4 Some selected ASCII codes for some English letters.",
              pymupdf.Rect(160, 499, 493, 519), size=9),
        _item("But ASCII is of course insufficient since there are lots of other characters.",
              pymupdf.Rect(160, 533, 493, 579)),
    ]
    frame = TextFrame(160, 493, 111, 719, 10)
    geo = PageGeometry.from_rects(612, [item.rect for item in items], [], 792, frame=frame)
    roles = HeuristicLayout().classify(items, [geo], median=10)
    assert roles[2] is Role.BODY and roles[3] is Role.FIGURE and roles[4] is Role.CAPTION


def test_a_sentence_that_names_a_figure_is_not_its_caption():
    items = [
        _item("We'll come back to precision and recall in Chapter 4.", pymupdf.Rect(160, 277, 491, 287)),
        _item("2.6.6 More Operators", pymupdf.Rect(160, 304, 279, 316), size=12),
        _item("Figure 2.13 shows some useful aliases for common ranges:", pymupdf.Rect(160, 324, 395, 334)),
        _item("any digit", pymupdf.Rect(293, 360, 328, 370)),
        _item("Figure 2.13 Aliases for common sets of characters.", pymupdf.Rect(161, 433, 353, 442), size=9),
    ]
    geo = PageGeometry.from_rects(612, [item.rect for item in items], [], 792)
    roles = HeuristicLayout().classify(items, [geo], median=10)
    assert roles == [Role.BODY, Role.HEADING, Role.BODY, Role.FIGURE, Role.CAPTION]


def test_smaller_pseudocode_above_a_caption_is_the_figure():
    items = [
        _item("A body paragraph that ends right before the algorithm box starts.", pymupdf.Rect(160, 60, 493, 100)),
        _item("V ← all unique characters in C # initial set of tokens is characters",
              pymupdf.Rect(170, 144, 427, 164), size=9),
        _item("Figure 2.6 The training part of the BPE algorithm.", pymupdf.Rect(160, 234, 493, 265), size=9),
    ]
    geo = PageGeometry.from_rects(612, [item.rect for item in items], [], 792)
    roles = HeuristicLayout().classify(items, [geo], median=10)
    assert roles == [Role.BODY, Role.FIGURE, Role.CAPTION]


def test_a_paragraph_in_a_framed_box_is_body_text():
    """带底色的圆角框按图算，框里成段的几句话仍是正文；图里的短标签照旧算图。"""
    box = pymupdf.Rect(100, 100, 500, 300)
    paragraph = (
        "The Framework is intended to be flexible and to augment existing risk practices which should align "
        "with applicable laws, regulations, and norms. Organizations should follow existing regulations and "
        "guidelines for risk criteria, tolerance, and response established by organizational, domain, "
        "discipline, sector, or professional requirements."
    )
    items = [
        LayoutItem(0, pymupdf.Rect(110, 110, 490, 290), paragraph, 10, False),
        LayoutItem(0, pymupdf.Rect(120, 120, 180, 130), "Encoder output", 8, False),
    ]
    roles = HeuristicLayout().classify(items, [_geo([box])], median=10)
    assert roles == [Role.BODY, Role.FIGURE]


def test_a_prompt_box_is_translated_as_a_whole():
    """框里大半是成段的正文（以冒号引出下文的一段也算），框里成句的列表项、示例和正文一起算正文，
    不留半框原文。两三个词的短标签照旧算图。"""
    box = pymupdf.Rect(70, 330, 525, 700)
    first = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
        "assistants to the user prompt displayed below. You will be given assistant A's answer and assistant "
        "B's answer. Your job is to evaluate which assistant's answer is better."
    )
    second = (
        "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer "
        "correctly responds to the prompt or follows the instructions. After providing your explanation, you "
        "must output only one of the following choices as your final verdict with a label:"
    )
    items = [
        LayoutItem(0, pymupdf.Rect(86, 348, 509, 440), first, 11, False),
        LayoutItem(0, pymupdf.Rect(86, 443, 510, 600), second, 11, False),
        LayoutItem(0, pymupdf.Rect(86, 604, 302, 616), "1. Assistant A is significantly better: [[A>>B]]", 11, False),
        LayoutItem(0, pymupdf.Rect(86, 631, 251, 643), "3. Tie, relatively the same: [[A=B]]", 11, False),
        LayoutItem(0, pymupdf.Rect(400, 660, 450, 670), "Encoder output", 8, False),
    ]
    roles = HeuristicLayout().classify(items, [_geo([box], 612, 792)], median=11)
    # 带编号的短列表项可能认成标题，都会送翻
    assert [role is Role.FIGURE for role in roles] == [False, False, False, False, True]
