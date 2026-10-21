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
    ids = _display_formula_ids([brace, condition, body, inline], [geo])
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
