import pymupdf

from app.formats.pdf import TextBlock, _absorb_curves, _assign_restored_parts, _placeholderize, _prepare_unit
from app.formats.pdf_runs import FormulaRun, TextRun, intersecting_curves, split_page_boundary, strip_boundary
from app.prompts import build_messages


def _span(text, font="TimesNewRomanPSMT", size=12.0, flags=0, x0=0.0, y0=0.0, x1=None):
    width = 12 if x1 is None else x1 - x0
    return {
        "text": text,
        "font": font,
        "size": size,
        "flags": flags,
        "bbox": (x0, y0, x0 + width, y0 + 10),
        "origin": (x0, y0 + 8),
    }


def _block(span_lines, size=12.0):
    return TextBlock(
        page=0, rect=pymupdf.Rect(0, 0, 200, 20), line_rects=[],
        text="", size=size, color="#000", bold=False, span_lines=span_lines,
    )


def test_placeholderize_builds_ordered_runs():
    block = _block([[
        _span("The formula ", x0=0),
        _span("E=mc", font="CMMI10", x0=70, x1=100),
        _span("2", font="CMR10", size=8.0, x0=96, y0=-3, x1=104),
        _span(" shows energy", x0=110),
    ]])
    sent, formulas = _placeholderize(block, 0)
    assert sent == "The formula {v1} shows energy"
    assert [type(run) for run in block.runs] == [TextRun, FormulaRun, TextRun]
    formula = block.runs[1]
    assert isinstance(formula, FormulaRun)
    assert formula.index == 1 and formula.name == "f0_1.png"
    assert formula.dx == round(formulas[0]["bbox"].x0 - 0, 1)
    assert formula.dy >= 0 and formula.descent >= 0
    assert formula.baseline > 0
    assert block.runs[0].text == "The formula "
    assert block.runs[2].text == " shows energy"


def test_mixed_emphasis_is_on_the_text_run():
    block = _block([[
        _span("See "),
        _span("this", flags=16, x0=20),
        _span(" word", x0=40),
    ]])
    sent, formulas = _placeholderize(block)
    assert sent == "See {b}this{/b} word" and formulas == []
    assert [run.text for run in block.runs] == ["See ", "this", " word"]
    assert block.runs[1].emphasis.bold and not block.runs[0].emphasis.bold


def test_prepare_unit_marks_the_run_boundary():
    def block(page, prose):
        spans = [_span(prose, x0=0), _span("E", font="CMMI10", x0=40)]
        item = _block([spans])
        item.page = page
        item.text = prose + "E"
        return item

    sent, formulas, name_i = _prepare_unit([block(0, "sequence "), block(1, "length ")], 0)
    assert name_i == 2 and "{|}" in sent
    assert sent.index("{v1}") < sent.index("{|}") < sent.index("{v2}")
    assert formulas[0]["owner"] == 0 and formulas[1]["owner"] == 1
    restored = (
        sent.replace("{v1}", "\x01i\x021\x01/i\x02")
        .replace("{v2}", "\x01i\x022\x01/i\x02")
    )
    left, right = split_page_boundary(restored)
    assert left is not None and "{|}" not in left and "{|}" not in right


def test_assign_keeps_the_cross_page_paragraph_together():
    """跨页译文整段留在起始页。汉字分界不加空格，否则「相当」和「不错」会分开。"""
    from app.formats.pdf import SPILL_TAIL

    left = TextBlock(
        page=0, rect=pymupdf.Rect(0, 0, 40, 12), line_rects=[],
        text="abcd", size=10, color="#000", bold=False,
    )
    right = TextBlock(
        page=1, rect=pymupdf.Rect(0, 0, 80, 12), line_rects=[],
        text="efghijklmnop", size=10, color="#000", bold=False,
    )
    parts, lists = _assign_restored_parts("表现相当{| }不错。我们推测原因在此。", [left, right], [])
    assert parts == ["表现相当不错。我们推测原因在此。", SPILL_TAIL]
    assert lists == [[], []]
    assert strip_boundary("甲乙丙丁戊己庚辛，壬癸。{|}续页。") == "甲乙丙丁戊己庚辛，壬癸。 续页。"


def test_stroke_just_outside_the_glyph_box_joins_the_formula():
    """分式线贴在框外、根号竖笔贴在左侧时要收进裁剪。通栏线和下划线不收。"""
    box = pymupdf.Rect(40, 40, 70, 54)
    bar = pymupdf.Rect(38, 36.8, 74, 38.2)  # 高出框 1.8pt 的分式线
    radical = pymupdf.Rect(32, 38, 39, 56)  # 贴在左侧的根号
    rule = pymupdf.Rect(0, 37, 400, 38.2)  # 贴着框、但是通栏
    underline = pymupdf.Rect(10, 52, 38, 53.2)  # 左侧单词的下划线
    curves = intersecting_curves(box, [bar, radical, rule, underline])
    assert pymupdf.Rect(bar) in curves
    assert pymupdf.Rect(radical) in curves
    assert pymupdf.Rect(rule) not in curves
    assert pymupdf.Rect(underline) not in curves


def test_intersecting_curve_joins_the_formula_box():
    box = pymupdf.Rect(40, 40, 70, 54)
    bar = pymupdf.Rect(42, 38, 68, 42)  # 压在框顶上的分式线
    rule = pymupdf.Rect(0, 40, 400, 41)  # 通栏线，不并
    curves = intersecting_curves(box, [bar, rule])
    assert curves == (pymupdf.Rect(bar),)
    formula = {
        "bbox": pymupdf.Rect(box), "d": 4.0, "dx": 10.0, "name": "f0_1.png",
        "owner": 0, "curves": [],
    }
    block = TextBlock(
        page=0, rect=box, line_rects=[], text="E", size=12, color="#000", bold=False,
        runs=[FormulaRun(
            index=1, bbox=pymupdf.Rect(box), baseline=50, dx=10, dy=0, descent=4,
            text="E", page=0, name="f0_1.png", line=0,
        )],
    )
    _absorb_curves(formula, [bar, rule], block)
    assert formula["bbox"].y0 == 38
    assert formula["h"] > 14
    assert len(formula["curves"]) == 1
    assert block.runs[0].bbox.y0 == 38
    assert block.runs[0].curves == (pymupdf.Rect(bar),)


def test_boundary_placeholder_is_mentioned_to_the_model():
    system, _ = build_messages(["quite {|} well"], "zh-CN")
    assert "{|}" in system and "{vN}" in system
    plain, _ = build_messages(["quite well"], "zh-CN")
    assert "Page-boundary" not in plain


def test_size_difference_is_a_style_mark():
    block = _block([[
        _span("See ", size=12),
        _span("this", size=10, flags=16, x0=20),
        _span(" word", size=12, x0=40),
    ]])
    sent, formulas = _placeholderize(block)
    assert sent == "See {z85}{b}this{/b}{/z} word" and formulas == []
    assert block.runs[1].size == 10


def test_size_mark_does_not_swallow_a_formula():
    block = _block([[
        _span("see ", size=10),
        _span("E", font="CMMI10", size=12, x0=30),
        _span(" note", size=10, x0=50),
    ]], size=12)
    sent, formulas = _placeholderize(block)
    assert formulas and "{v1}" in sent
    assert sent.index("{/z}") < sent.index("{v1}")
    assert "{z" not in sent[sent.index("{v1}"):sent.index("{v1}") + 4]


def test_style_placeholder_is_mentioned_to_the_model():
    system, _ = build_messages(["See {b}this{/b}"], "zh-CN")
    assert "{b}" in system and "{zN}" in system
    sized, _ = build_messages(["See {z85}this{/z}"], "zh-CN")
    assert "{zN}" in sized
    plain, _ = build_messages(["See this"], "zh-CN")
    assert "Style placeholders" not in plain
