import html

from app.formats.pdf_flow import (
    Emphasis,
    EmphasisWriter,
    break_lines,
    emphasis_of,
    measure_token,
    nowrap_lines,
    restore_emphasis,
    size_percent,
    strip_style_marks,
    writer_for,
)
from app.formats.pdf_types import Span

_SENTINEL = "\x01i\x021\x01/i\x02"


def test_formula_sentinel_is_not_split() -> None:
    text = "甲" * 8 + _SENTINEL + "乙" * 8
    lines = break_lines(text, em=10, max_width=50, formula_widths={1: 30})
    assert "".join(lines) == text
    assert sum(line.count(_SENTINEL) for line in lines) == 1
    assert all("\x01i\x02" not in line or _SENTINEL in line for line in lines)


def test_wide_formula_takes_its_own_line() -> None:
    lines = break_lines("甲" + _SENTINEL + "乙", em=10, max_width=20, formula_widths={1: 40})
    assert _SENTINEL in lines
    assert lines[lines.index(_SENTINEL)] == _SENTINEL


def test_cjk_wraps_by_character() -> None:
    assert break_lines("甲乙丙丁戊", em=10, max_width=25, formula_widths={}) == ["甲乙", "丙丁", "戊"]


def test_first_lines_take_their_own_widths() -> None:
    """The first lines making room for a figure break at their own widths, the other lines at the full width."""
    lines = break_lines("甲" * 12, em=10, max_width=50, formula_widths={}, first_widths=[20, 30])
    assert lines == ["甲甲", "甲甲甲", "甲甲甲甲甲", "甲甲"]


def test_a_paren_group_wider_than_a_narrow_line_still_breaks() -> None:
    """Bracket groups are locked against the narrowest line. A group that fits the full width but not a narrow line
    must not roll back repeatedly."""
    lines = break_lines("甲（乙丙）丁", em=10, max_width=60, formula_widths={}, first_widths=[30])
    assert "".join(lines) == "甲（乙丙）丁"


def test_decimal_number_is_not_split_at_its_point() -> None:
    """The line must not break after the decimal point of "2.07 秒"."""
    lines = break_lines("甲乙丙丁戊 2.07 秒", em=10, max_width=70, formula_widths={})
    assert any("2.07" in line for line in lines)
    assert not any(line.rstrip().endswith("2.") for line in lines)


def test_a_long_hyphenated_word_breaks_after_its_hyphen() -> None:
    """When (Super-NaturalInstructions does not fit, the previous line breaks early and the Chinese is spread out.
    A line may break after a hyphen with words of three or more letters on both sides; GPT-4 and x-axis do not
    break."""
    text = "由于许多任务彼此相似(Super-NaturalInstructions 包含 25 个数据集"
    lines = break_lines(text, em=10, max_width=200, formula_widths={})
    assert lines[0].endswith("(Super-") and "".join(lines) == text
    lines = break_lines("由于许多任务彼此相似, GPT-4 和 x-axis", em=10, max_width=110, formula_widths={})
    assert not any(line.endswith(("GPT-", "x-")) for line in lines)


def test_punctuation_does_not_start_a_line() -> None:
    lines = break_lines("甲乙丙丁。", em=10, max_width=40, formula_widths={})
    assert lines == ["甲乙丙", "丁。"]
    assert all(not line.startswith("。") for line in lines)


def test_halfwidth_punct_drops_the_trailing_space() -> None:
    lines = break_lines("甲乙, 丙丁", em=10, max_width=32, formula_widths={})
    assert any(line.endswith(",") for line in lines)
    assert all(not line.endswith(", ") for line in lines)
    assert "".join(lines).replace(" ", "") == "甲乙,丙丁"


def test_short_parentheses_stay_on_one_line() -> None:
    text = "甲甲甲（如序列）乙乙乙"
    lines = break_lines(text, em=10, max_width=50, formula_widths={})
    assert any("（如序列）" in line for line in lines)
    assert all("（" not in line or "）" in line for line in lines)
    ascii_lines = break_lines("see (term) next", em=10, max_width=40, formula_widths={})
    assert any("(term)" in line for line in ascii_lines)


def test_parentheses_around_a_formula_stay_together() -> None:
    text = "甲（" + _SENTINEL + "）乙"
    lines = break_lines(text, em=10, max_width=40, formula_widths={1: 20})
    assert any("（" + _SENTINEL + "）" in line for line in lines)


def test_long_parentheses_may_split() -> None:
    lines = break_lines("（" + "甲" * 20 + "）", em=10, max_width=50, formula_widths={})
    assert len(lines) > 1
    assert not any("（" in line and "）" in line for line in lines)


def test_style_marks_have_zero_width() -> None:
    assert break_lines("{b}甲乙丙{/b}", em=10, max_width=30, formula_widths={}) == ["{b}甲乙丙{/b}"]
    assert break_lines("{z85}甲乙{/z}丙", em=10, max_width=30, formula_widths={}) == ["{z85}甲乙{/z}丙"]


def test_nowrap_html_keeps_each_line_intact() -> None:
    html_text = nowrap_lines(["甲" + _SENTINEL, "乙"])
    assert html_text.count("white-space:nowrap") == 2
    assert "<br>" in html_text
    assert _SENTINEL in html_text


def test_restore_emphasis_after_escape() -> None:
    assert restore_emphasis(html.escape("{b}a&b{/b}")) == "<b>a&amp;b</b>"
    assert restore_emphasis("{i}x{/i}") == "<i>x</i>"
    assert restore_emphasis(html.escape("{z85}a&b{/z}")) == '<span style="font-size:85%">a&amp;b</span>'
    assert strip_style_marks("See {z85}{b}this{/b}{/z} word") == "See this word"


def test_writer_marks_only_a_mixed_block() -> None:
    writer = EmphasisWriter(Emphasis(False, False), mixed=True)
    assert writer.text("see ", Emphasis(False, False)) + writer.text(
        "this", Emphasis(True, False)
    ) + writer.close() == ("see {b}this{/b}")
    plain = EmphasisWriter(Emphasis(False, False), mixed=False)
    assert plain.text("see this", Emphasis(True, False)) + plain.atom("{v1}") + plain.close() == "see this{v1}"


def test_writer_marks_size_outside_emphasis() -> None:
    writer = EmphasisWriter(Emphasis(False, False), mixed=True, base_size=12)
    sent = (
        writer.text("See ", Emphasis(False, False), 12)
        + writer.text("this", Emphasis(True, False), 10)
        + writer.text(" word", Emphasis(False, False), 12)
        + writer.close()
    )
    assert sent == "See {z85}{b}this{/b}{/z} word"
    assert size_percent(11.5, 12) == 100
    assert size_percent(10, 12) == 85


def test_writer_for_marks_the_differing_span() -> None:
    spans: list[Span] = [
        {"text": "See ", "font": "TimesNewRomanPSMT", "flags": 0},
        {"text": "this", "font": "TimesNewRomanPS-BoldMT", "flags": 16},
        {"text": " word", "font": "TimesNewRomanPSMT", "flags": 0},
    ]
    writer = writer_for(spans)
    assert writer.mixed
    assert writer.text("See ", emphasis_of("TimesNewRomanPSMT", 0)) == "See "
    assert writer.text("this", emphasis_of("TimesNewRomanPS-BoldMT", 16)) == "{b}this"
    uniform = writer_for([{"text": "All regular words", "font": "Times", "flags": 0}])
    assert not uniform.mixed
    assert uniform.text("All regular words", Emphasis(False, False)) == "All regular words"


def _span(text: str, bold: bool = False, italic: bool = False) -> Span:
    font = "Times-" + ("Bold" if bold else "") + ("Italic" if italic else "") if bold or italic else "Times-Roman"
    return {"text": text, "font": font, "flags": (16 if bold else 0) | (2 if italic else 0)}


def _write(spans: list[Span]) -> str:
    writer = writer_for(spans)
    return "".join(writer.text(s["text"], emphasis_of(s["font"], s["flags"])) for s in spans) + writer.close()


def test_a_bold_label_before_body_split_into_short_spans_keeps_its_mark() -> None:
    """When citations cut the body into many short spans, the longest one is the bold label. Bold is written only
    through {b}, so the base style must not be bold."""
    spans: list[Span] = [
        _span("•"),
        _span(" ", bold=True),
        _span("Commonsense Reasoning (0-shot):", bold=True),
        _span(" Hellaswag ["),
        _span("28"),
        _span("], Winogrande ["),
        _span("21"),
        _span("]"),
    ]
    assert _write(spans) == "• {b}Commonsense Reasoning (0-shot):{/b} Hellaswag [28], Winogrande [21]"
    # Bold is marked even when it has one character more than the body
    spans = [
        _span("• N-gram"),
        _span(" language models", bold=True),
        _span(" can be evaluated on a"),
        _span(" test set", bold=True),
        _span(" using"),
        _span(" perplexity.", bold=True),
    ]
    assert (
        _write(spans)
        == "• N-gram {b}language models{/b} can be evaluated on a {b}test set{/b} using {b}perplexity.{/b}"
    )
    # A fully bold block has a bold base and no markers
    assert _write([_span("Bold title", bold=True), _span(" 3")]) == "Bold title 3"


def test_spaces_do_not_open_a_mark_and_close_the_one_that_ends() -> None:
    spans: list[Span] = [
        _span("Theorem 1.", bold=True),
        _span(" ", bold=False),
        _span("The minimum is"),
        _span(" ", italic=True),
        _span("unique", italic=True),
    ]
    assert _write(spans) == "{b}Theorem 1.{/b} The minimum is {i}unique{/i}"


def test_a_mostly_italic_statement_keeps_italic_as_base() -> None:
    """The italic body of a theorem has an italic base, without {i} on every span."""
    writer = writer_for(
        [_span("Theorem 1.", bold=True), _span(" Assume that the function has bounded gradients", italic=True)]
    )
    assert writer.base == Emphasis(False, True)


def test_paren_group_at_width_edge_terminates() -> None:
    """A bracket group whose width falls in the tolerance band must not loop forever: the locking condition must match
    the overflow test of filling."""
    text = "前面占位" + "（一二三四五六七）" + "尾巴"
    for width in (7.7, 7.8, 7.9, 8.0, 8.1):
        lines = break_lines(text, em=1.0, max_width=width, formula_widths={})
        assert "".join(lines) == text


def test_oversize_word_split_to_fit() -> None:
    """Overwide words such as long URLs are cut by character instead of running past the right edge with nowrap."""
    text = "见 http://example.com/very/long/path 结束"
    lines = break_lines(text, em=1.0, max_width=8.0, formula_widths={})
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    assert all(len(line) <= 20 for line in lines)


def test_restore_emphasis_keeps_italic_tag() -> None:
    """Italic stays <i>. The fallback font has no bold Chinese, so <b> would neither emphasize the Chinese nor keep the
    Latin italic."""
    assert restore_emphasis("正常{i}强调{/i}文字") == "正常<i>强调</i>文字"


def test_space_around_latin_balances_both_sides() -> None:
    """One space each between Chinese and i-th. Spaces inside markers move out, and English sentences are unchanged."""
    from app.formats.pdf_flow import space_around_latin

    assert space_around_latin("第{i} i-th{/i}个") == "第 {i}i-th{/i} 个"
    assert space_around_latin("第i-th个") == "第 i-th 个"
    assert space_around_latin("the i-th mini-batch") == "the i-th mini-batch"
    sentinel = "\x01i\x021\x01/i\x02"
    assert space_around_latin(f"词元{sentinel},") == f"词元{sentinel},"


def test_separate_after_formula_adds_one_space() -> None:
    """A space is added when text directly follows a formula. Not before punctuation, existing spaces or style
    markers, and never twice."""
    from app.formats.pdf_flow import separate_after_formula

    sent = "\x01i\x021\x01/i\x02"
    assert separate_after_formula(f"使用{sent}时") == f"使用{sent} 时"
    assert separate_after_formula(f"取 arg max{sent})。") == f"取 arg max{sent})。"
    assert separate_after_formula(f"已有{sent} 空格") == f"已有{sent} 空格"
    assert separate_after_formula(f"{sent}{{b}}时") == f"{sent}{{b}} 时"
    assert separate_after_formula("没有公式") == "没有公式"


def test_fullwidth_punctuation_measures_one_em() -> None:
    """Full-width punctuation (，。（） and so on) is not in the CJK regex but still counts as 1em by
    east_asian_width. Underestimating the line width at 0.5em would push justified lines past the right edge."""
    assert measure_token("，", 10.0, {}) == 10.0
    assert measure_token("（", 10.0, {}) == 10.0
    assert measure_token("a", 10.0, {}) >= 5.0


def test_wide_glyphs_wrap_before_they_pass_the_edge() -> None:
    """Dashes and M are wider than 0.5em. Estimated at half a character, they would stick out past the line end."""
    assert measure_token("—", 10.0, {}) >= 10.0
    assert measure_token("M", 10.0, {}) > 5.0
    assert break_lines("甲乙丙丁——", em=10, max_width=50, formula_widths={}) != ["甲乙丙丁——"]
    lines = break_lines("甲甲甲HMM", em=10, max_width=45, formula_widths={})
    assert lines[0] == "甲甲甲"
    assert "HMM" in lines[1]


def test_a_word_split_by_style_marks_is_not_broken_at_the_mark() -> None:
    """{b}R{/b}epresentations is one word, so the line must not break after the bold initial; an opening bracket
    before the word moves with it."""
    from app.formats.pdf_flow import break_lines

    text = "双向编码器表示”({b}B{/b}idirectional {b}E{/b}ncoder {b}R{/b}epresentations from {b}T{/b}ransformers)的缩写"
    for width in (100, 120, 140, 160):
        lines = break_lines(text, 9.0, width, {})
        for line in lines:
            assert not line.rstrip().endswith(("{b}R{/b}", "{b}B{/b}", "{b}T{/b}", "("))
        assert "".join(lines).replace("{b}", "").replace("{/b}", "").replace(" ", "") == text.replace(
            "{b}", ""
        ).replace("{/b}", "").replace(" ", "")


def test_slack_before_a_long_word_is_shared_with_the_line_above() -> None:
    """A long word at the start of the next line did not fit, leaving this line several characters short. Chinese
    characters move down from the end of the line above, so both lines are a little short and justification
    spreads each only slightly. The last line is left however short it is."""
    from app.formats.pdf_flow import _break_lines, measure_width

    text = (
        "目前为止所展示的所有结果都采用了微调方法即在预训练模型上添加一个简单的分类层并在下游任务上"
        "对所有参数进行联合微调然而基于特征的方法也具有一定优势首先并非所有任务都能轻易地用 "
        "Transformer 编码器架构表示因此需要额外添加特定于任务的模型架构"
    )
    lines = _break_lines(text, 10, 230, {})
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    # Without balancing, the line above Transformer is nearly six characters short
    slack = [230 - measure_width(line, 10) for line in lines[:-1]]
    assert max(slack) < 30


def test_narrow_latin_letters_measure_as_drawn() -> None:
    """i, l and commas are measured at their written width, not the 0.5em default. Measured too wide, mixed Chinese
    and English paragraphs break before lines are full, and justification at its limit still falls short of the
    right edge. Measured widths are never less than the written ones, so lines stay within the right edge."""
    from app.formats.pdf_align import _segments
    from app.formats.pdf_flow import measure_width

    for text in ("illicit little", "{i}decisions, steep hills{/i}", "{b}difficult{/b}", "可能会让你了解到, "):
        segments = _segments(text, 10.0)
        assert segments is not None
        drawn = sum(segment.width for segment in segments)
        assert drawn <= measure_width(text, 10.0) <= drawn * 1.1
    text = (
        "分布假说的核心思想是: 看到 {i}difficult{/i} 与 {i}hard{/i} 一样, 出现在相似的语境词"
        "({i}decisions, steep hills{/i})附近, 可能会让你了解到, {i}difficult{/i} 的含义与 {i}hard{/i} 相近。"
    )
    assert len(break_lines(text, em=8.8, max_width=335, formula_widths={})) == 2


def test_a_formula_at_the_line_start_lines_up_with_the_text_below() -> None:
    """With a formula at the start of a line, the blank left part of its slot is not drawn, so the formula and the
    text of the next line start at the same left edge."""
    import pymupdf
    import pytest

    from app.formats.pdf_align import draw_cjk_lines
    from app.formats.pdf_typeset import FORMULA_PAD, FormulaMetric, typeset_lines

    doc = pymupdf.open()
    page = doc.new_page()
    box = pymupdf.Rect(100, 100, 300, 200)
    metrics = {1: FormulaMetric(6 + 2 * FORMULA_PAD, 6, 8.8, 0)}
    lines = typeset_lines(
        _SENTINEL + " 个词类型。用各种 n 元语法计数和其他记号来表示估计的公式，后面还有很多字要换到下一行去写",
        8.8,
        box.width,
        metrics,
    )
    assert len(lines) >= 2
    slots = draw_cjk_lines(page, doc, lines, box, 8.8, 1.45, "#000000", "left", frozenset({1}), {})
    assert slots is not None
    assert slots[0].x + FORMULA_PAD == pytest.approx(box.x0, abs=0.01)
