import html

from app.formats.pdf_flow import (
    Emphasis,
    EmphasisWriter,
    break_lines,
    emphasis_of,
    nowrap_lines,
    restore_emphasis,
    size_percent,
    strip_style_marks,
    writer_for,
)

_SENTINEL = "\x01i\x021\x01/i\x02"


def test_formula_sentinel_is_not_split():
    text = "甲" * 8 + _SENTINEL + "乙" * 8
    lines = break_lines(text, em=10, max_width=50, formula_widths={1: 30})
    assert "".join(lines) == text
    assert sum(line.count(_SENTINEL) for line in lines) == 1
    assert all("\x01i\x02" not in line or _SENTINEL in line for line in lines)


def test_wide_formula_takes_its_own_line():
    lines = break_lines("甲" + _SENTINEL + "乙", em=10, max_width=20, formula_widths={1: 40})
    assert _SENTINEL in lines
    assert lines[lines.index(_SENTINEL)] == _SENTINEL


def test_cjk_wraps_by_character():
    assert break_lines("甲乙丙丁戊", em=10, max_width=25, formula_widths={}) == ["甲乙", "丙丁", "戊"]


def test_style_marks_have_zero_width():
    assert break_lines("{b}甲乙丙{/b}", em=10, max_width=30, formula_widths={}) == ["{b}甲乙丙{/b}"]
    assert break_lines("{z85}甲乙{/z}丙", em=10, max_width=30, formula_widths={}) == ["{z85}甲乙{/z}丙"]


def test_nowrap_html_keeps_each_line_intact():
    html_text = nowrap_lines(["甲" + _SENTINEL, "乙"])
    assert html_text.count("white-space:nowrap") == 2
    assert "<br>" in html_text
    assert _SENTINEL in html_text


def test_restore_emphasis_after_escape():
    assert restore_emphasis(html.escape("{b}a&b{/b}")) == "<b>a&amp;b</b>"
    assert restore_emphasis("{i}x{/i}") == "<i>x</i>"
    assert restore_emphasis(html.escape("{z85}a&b{/z}")) == '<span style="font-size:85%">a&amp;b</span>'
    assert strip_style_marks("See {z85}{b}this{/b}{/z} word") == "See this word"


def test_writer_marks_only_a_mixed_block():
    writer = EmphasisWriter(Emphasis(False, False), mixed=True)
    assert writer.text("see ", Emphasis(False, False)) + writer.text("this", Emphasis(True, False)) + writer.close() == (
        "see {b}this{/b}"
    )
    plain = EmphasisWriter(Emphasis(False, False), mixed=False)
    assert plain.text("see this", Emphasis(True, False)) + plain.atom("{v1}") + plain.close() == "see this{v1}"


def test_writer_marks_size_outside_emphasis():
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


def test_writer_for_uses_the_longest_span_as_base():
    spans = [
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
