import pymupdf

from app.formats.pdf_blocks import TextBlock
from app.formats.pdf_crop import _absorb_curves
from app.formats.pdf_restore import _assign_restored_parts
from app.formats.pdf_runs import FormulaRun, TextRun, intersecting_curves, split_page_boundary, strip_boundary
from app.formats.pdf_types import PdfDict
from app.formats.pdf_units import _placeholderize, _prepare_unit
from app.prompts import build_messages


def _span(
    text: str,
    font: str = "TimesNewRomanPSMT",
    size: float = 12.0,
    flags: int = 0,
    x0: float = 0.0,
    y0: float = 0.0,
    x1: float | None = None,
) -> PdfDict:
    width = 12 if x1 is None else x1 - x0
    return {
        "text": text,
        "font": font,
        "size": size,
        "flags": flags,
        "bbox": (x0, y0, x0 + width, y0 + 10),
        "origin": (x0, y0 + 8),
    }


def _block(span_lines: list[list[PdfDict]], size: float = 12.0) -> TextBlock:
    return TextBlock(
        page=0,
        rect=pymupdf.Rect(0, 0, 200, 20),
        line_rects=[],
        text="",
        size=size,
        color="#000",
        bold=False,
        span_lines=span_lines,
    )


def test_placeholderize_builds_ordered_runs() -> None:
    block = _block(
        [
            [
                _span("The formula ", x0=0),
                _span("E=mc", font="CMMI10", x0=70, x1=100),
                _span("2", font="CMR10", size=8.0, x0=96, y0=-3, x1=104),
                _span(" shows energy", x0=110),
            ]
        ]
    )
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


def test_mixed_emphasis_is_on_the_text_run() -> None:
    block = _block(
        [
            [
                _span("See "),
                _span("this", flags=16, x0=20),
                _span(" word", x0=40),
            ]
        ]
    )
    sent, formulas = _placeholderize(block)
    assert sent == "See {b}this{/b} word" and formulas == []
    assert [run.text for run in block.runs] == ["See ", "this", " word"]
    plain, bold = block.runs[0], block.runs[1]
    assert isinstance(plain, TextRun) and isinstance(bold, TextRun)
    assert bold.emphasis.bold and not plain.emphasis.bold


def test_prepare_unit_marks_the_run_boundary() -> None:
    def block(page: int, prose: str) -> TextBlock:
        spans = [_span(prose, x0=0), _span("E", font="CMMI10", x0=40)]
        item = _block([spans])
        item.page = page
        item.text = prose + "E"
        return item

    sent, formulas, name_i = _prepare_unit([block(0, "sequence "), block(1, "length ")], 0)
    assert name_i == 2 and "{|}" in sent
    assert sent.index("{v1}") < sent.index("{|}") < sent.index("{v2}")
    assert formulas[0]["owner"] == 0 and formulas[1]["owner"] == 1
    restored = sent.replace("{v1}", "\x01i\x021\x01/i\x02").replace("{v2}", "\x01i\x022\x01/i\x02")
    split = split_page_boundary(restored)
    assert split is not None
    left, right = split
    assert "{|}" not in left and "{|}" not in right


def test_assign_keeps_the_cross_page_paragraph_together() -> None:
    """A translation across pages stays whole on the starting page. No space is added at a boundary between Chinese
    characters, or "相当" and "不错" would be separated."""
    from app.formats.pdf_blocks import SPILL_TAIL

    left = TextBlock(
        page=0,
        rect=pymupdf.Rect(0, 0, 40, 12),
        line_rects=[],
        text="abcd",
        size=10,
        color="#000",
        bold=False,
    )
    right = TextBlock(
        page=1,
        rect=pymupdf.Rect(0, 0, 80, 12),
        line_rects=[],
        text="efghijklmnop",
        size=10,
        color="#000",
        bold=False,
    )
    parts, lists = _assign_restored_parts("表现相当{| }不错。我们推测原因在此。", [left, right], [])
    assert parts == ["表现相当不错。我们推测原因在此。", SPILL_TAIL]
    assert lists == [[], []]
    assert strip_boundary("甲乙丙丁戊己庚辛，壬癸。{|}续页。") == "甲乙丙丁戊己庚辛，壬癸。 续页。"


def test_assign_splits_a_paragraph_cut_by_a_display_at_the_boundary() -> None:
    """Two blocks separated by a display formula or example are split back at the boundary, and formula sentinels go
    with their own block. When the model drops the boundary, the whole paragraph stays in the starting block."""
    from app.formats.pdf_blocks import SPILL_TAIL

    left = TextBlock(
        page=0,
        rect=pymupdf.Rect(0, 0, 40, 12),
        line_rects=[],
        text="the initial learning rate E is 0.1:",
        size=10,
        color="#000",
        bold=False,
    )
    right = TextBlock(
        page=1,
        rect=pymupdf.Rect(0, 0, 80, 12),
        line_rects=[],
        text="The single update step requires F.",
        size=10,
        color="#000",
        bold=False,
    )
    formulas = [{"name": "a", "owner": 0}, {"name": "b", "owner": 1}]
    restored = "初始学习率\x01i\x021\x01/i\x02为 0.1：{|}单次更新步骤要用到\x01i\x022\x01/i\x02。"
    parts, lists = _assign_restored_parts(restored, [left, right], formulas, apart=True)
    assert parts == ["初始学习率\x01i\x021\x01/i\x02为 0.1：", "单次更新步骤要用到\x01i\x021\x01/i\x02。"]
    assert lists == [[formulas[0]], [formulas[1]]]
    parts, _ = _assign_restored_parts("初始学习率为 0.1：单次更新步骤。", [left, right], [], apart=True)
    assert parts == ["初始学习率为 0.1：单次更新步骤。", SPILL_TAIL]


def test_stroke_just_outside_the_glyph_box_joins_the_formula() -> None:
    """A fraction bar just outside the box or a radical stroke on the left is taken into the clip. Full-width rules
    and underlines are not."""
    box = pymupdf.Rect(40, 40, 70, 54)
    bar = pymupdf.Rect(38, 36.8, 74, 38.2)  # fraction bar 1.8pt above the box
    radical = pymupdf.Rect(32, 38, 39, 56)  # radical on the left
    rule = pymupdf.Rect(0, 37, 400, 38.2)  # touching the box but full width
    underline = pymupdf.Rect(10, 52, 38, 53.2)  # underline of the word on the left
    curves = intersecting_curves(box, [bar, radical, rule, underline])
    assert pymupdf.Rect(bar) in curves
    assert pymupdf.Rect(radical) in curves
    assert pymupdf.Rect(rule) not in curves
    assert pymupdf.Rect(underline) not in curves


def test_intersecting_curve_joins_the_formula_box() -> None:
    box = pymupdf.Rect(40, 40, 70, 54)
    bar = pymupdf.Rect(42, 38, 68, 42)  # fraction bar over the box top
    rule = pymupdf.Rect(0, 40, 400, 41)  # full-width rule, not merged
    curves = intersecting_curves(box, [bar, rule])
    assert curves == (pymupdf.Rect(bar),)
    formula: PdfDict = {
        "bbox": pymupdf.Rect(box),
        "d": 4.0,
        "dx": 10.0,
        "name": "f0_1.png",
        "owner": 0,
        "curves": [],
    }
    block = TextBlock(
        page=0,
        rect=box,
        line_rects=[],
        text="E",
        size=12,
        color="#000",
        bold=False,
        runs=[
            FormulaRun(
                index=1,
                bbox=pymupdf.Rect(box),
                baseline=50,
                dx=10,
                dy=0,
                descent=4,
                text="E",
                page=0,
                name="f0_1.png",
                line=0,
            )
        ],
    )
    _absorb_curves(formula, [bar, rule], block)
    assert formula["bbox"].y0 == 38
    assert formula["h"] > 14
    assert len(formula["curves"]) == 1
    run = block.runs[0]
    assert isinstance(run, FormulaRun)
    assert run.bbox.y0 == 38
    assert run.curves == (pymupdf.Rect(bar),)


def test_boundary_placeholder_is_mentioned_to_the_model() -> None:
    system, _ = build_messages(["quite {|} well"], "zh-CN")
    assert "{|}" in system and "{vN}" in system
    plain, _ = build_messages(["quite well"], "zh-CN")
    assert "Page-boundary" not in plain


def test_size_difference_is_a_style_mark() -> None:
    block = _block(
        [
            [
                _span("See ", size=12),
                _span("this", size=10, flags=16, x0=20),
                _span(" word", size=12, x0=40),
            ]
        ]
    )
    sent, formulas = _placeholderize(block)
    assert sent == "See {z85}{b}this{/b}{/z} word" and formulas == []
    sized = block.runs[1]
    assert isinstance(sized, TextRun)
    assert sized.size == 10


def test_size_mark_does_not_swallow_a_formula() -> None:
    block = _block(
        [
            [
                _span("see ", size=10),
                _span("E", font="CMMI10", size=12, x0=30),
                _span(" note", size=10, x0=50),
            ]
        ],
        size=12,
    )
    sent, formulas = _placeholderize(block)
    assert formulas and "{v1}" in sent
    assert sent.index("{/z}") < sent.index("{v1}")
    assert "{z" not in sent[sent.index("{v1}") : sent.index("{v1}") + 4]


def test_style_placeholder_is_mentioned_to_the_model() -> None:
    system, _ = build_messages(["See {b}this{/b}"], "zh-CN")
    assert "{b}" in system and "{zN}" in system
    sized, _ = build_messages(["See {z85}this{/z}"], "zh-CN")
    assert "{zN}" in sized
    plain, _ = build_messages(["See this"], "zh-CN")
    assert "Style placeholders" not in plain
