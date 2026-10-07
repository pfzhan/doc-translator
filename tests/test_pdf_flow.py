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


def test_short_parentheses_stay_on_one_line():
    text = "甲甲甲（如序列）乙乙乙"
    lines = break_lines(text, em=10, max_width=50, formula_widths={})
    assert any("（如序列）" in line for line in lines)
    assert all("（" not in line or "）" in line for line in lines)
    ascii_lines = break_lines("see (term) next", em=10, max_width=40, formula_widths={})
    assert any("(term)" in line for line in ascii_lines)


def test_parentheses_around_a_formula_stay_together():
    text = "甲（" + _SENTINEL + "）乙"
    lines = break_lines(text, em=10, max_width=40, formula_widths={1: 20})
    assert any("（" + _SENTINEL + "）" in line for line in lines)


def test_long_parentheses_may_split():
    lines = break_lines("（" + "甲" * 20 + "）", em=10, max_width=50, formula_widths={})
    assert len(lines) > 1
    assert not any("（" in line and "）" in line for line in lines)


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


def test_paren_group_at_width_edge_terminates():
    """括号组宽度落在容差带内时不能死循环：锁组条件必须和填充溢出判定一致。"""
    text = "前面占位" + "（一二三四五六七）" + "尾巴"
    for width in (7.7, 7.8, 7.9, 8.0, 8.1):
        lines = break_lines(text, em=1.0, max_width=width, formula_widths={})
        assert "".join(lines) == text


def test_oversize_word_split_to_fit():
    """长 URL 这类超宽词按字符硬切，不能 nowrap 越出右缘。"""
    text = "见 http://example.com/very/long/path 结束"
    lines = break_lines(text, em=1.0, max_width=8.0, formula_widths={})
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    assert all(len(line) <= 20 for line in lines)


def test_restore_emphasis_keeps_italic_tag():
    """斜体保持 <i>。回退字体没有汉字粗体，改成 <b> 不能强调汉字，还会丢掉拉丁斜体。"""
    assert restore_emphasis("正常{i}强调{/i}文字") == "正常<i>强调</i>文字"


def test_space_around_latin_balances_both_sides():
    """汉字和 i-th 之间各留一个空格。标记里的空格挪出来，英文句子不动。"""
    from app.formats.pdf_flow import space_around_latin

    assert space_around_latin("第{i} i-th{/i}个") == "第 {i}i-th{/i} 个"
    assert space_around_latin("第i-th个") == "第 i-th 个"
    assert space_around_latin("the i-th mini-batch") == "the i-th mini-batch"
    sentinel = "\x01i\x021\x01/i\x02"
    assert space_around_latin(f"词元{sentinel},") == f"词元{sentinel},"


def test_separate_after_formula_adds_one_space():
    """公式后面直接接文字时补一个空格。标点、已有空格、样式标记都不重复补。"""
    from app.formats.pdf_flow import separate_after_formula

    sent = "\x01i\x021\x01/i\x02"
    assert separate_after_formula(f"使用{sent}时") == f"使用{sent} 时"
    assert separate_after_formula(f"取 arg max{sent})。") == f"取 arg max{sent})。"
    assert separate_after_formula(f"已有{sent} 空格") == f"已有{sent} 空格"
    assert separate_after_formula(f"{sent}{{b}}时") == f"{sent}{{b}} 时"
    assert separate_after_formula("没有公式") == "没有公式"


def test_fullwidth_punctuation_measures_one_em():
    """全角标点（，。（）等）不在 CJK 正则里，按 east_asian_width 也得算 1em。
    按 0.5em 低估行宽，两端对齐时会把行推出右缘。"""
    assert measure_token("，", 10.0, {}) == 10.0
    assert measure_token("（", 10.0, {}) == 10.0
    assert measure_token("a", 10.0, {}) >= 5.0


def test_wide_glyphs_wrap_before_they_pass_the_edge():
    """破折号和 M 比 0.5em 宽。按半字宽估算时，这两个字会多挤在行尾外面。"""
    assert measure_token("—", 10.0, {}) >= 10.0
    assert measure_token("M", 10.0, {}) > 5.0
    assert break_lines("甲乙丙丁——", em=10, max_width=50, formula_widths={}) != ["甲乙丙丁——"]
    lines = break_lines("甲甲甲HMM", em=10, max_width=45, formula_widths={})
    assert lines[0] == "甲甲甲"
    assert "HMM" in lines[1]
