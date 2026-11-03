"""Translation units, formulas and restoring translations (pdf_units, pdf_restore, pdf_math)."""

import asyncio
import re

import pymupdf
import pytest

from app.formats import translate_file
from tests.helpers import SAMPLES, _glyph, _pdf_with_text, run, runner


def test_a_wrapped_formula_wider_than_a_line_stays_in_pieces():
    """Pieces of a formula cut by a line break are joined into one placeholder. When the pieces side by side are wider
    than a line (a long URL across two lines) they are not joined: one placeholder must fit on a single line, so a
    placeholder that does not fit shrinks the whole paragraph, while two separate pieces can break at the original
    break point."""
    from app.formats.pdf_formula_runs import _join_line_wrapped_formulas

    def span(text, x0, x1, y, font="LMMono10-Regular"):
        return {
            "text": text,
            "size": 12.0,
            "font": font,
            "flags": 0,
            "bbox": (x0, y, x1, y + 12),
            "origin": (x0, y + 9),
        }

    def lines(head_x0, tail_x1):
        return [
            [
                span("Olympiad ", 70.9, head_x0, 420.0, "URWPalladioL"),
                span("https://example.org/wiki/AMC_", head_x0, 524.4, 420.0),
            ],
            [
                span("results?srsltid=AfmBOoqQ6pQic5NCan", 70.9, tail_x1, 433.5),
                span(" positions DeepSeek-R1", tail_x1, 524.4, 433.5, "URWPalladioL"),
            ],
        ]

    def runs(span_lines):
        return [
            {"spans": [span_lines[0][1]], "bbox": pymupdf.Rect(span_lines[0][1]["bbox"]), "pos": (0, 1, 1)},
            {"spans": [span_lines[1][0]], "bbox": pymupdf.Rect(span_lines[1][0]["bbox"]), "pos": (1, 0, 0)},
        ]

    short = lines(400.0, 200.0)
    assert len(_join_line_wrapped_formulas(runs(short), short, 12.0)) == 1
    long = lines(121.2, 470.0)
    assert len(_join_line_wrapped_formulas(runs(long), long, 12.0)) == 2


def test_scan_status_detection():
    from app.formats.pdf_unit_blocks import _scan_status

    # Normal PDF
    doc = _pdf_with_text("Visible paragraph text here.")
    assert _scan_status(doc) == "ok"
    # With a hidden OCR text layer (render mode 3)
    doc = pymupdf.open()
    for _ in range(4):
        p = doc.new_page()
        p.insert_text((50, 50), "hidden layer text here", fontsize=12, render_mode=3)
    assert _scan_status(doc) == "hidden_text"
    # Image-only scan
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page().insert_image(
            pymupdf.Rect(10, 10, 60, 60), pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 5, 5))
        )
    assert _scan_status(doc) == "scanned"


def _block_with_spans(span_lines, size=12.0):
    from app.formats.pdf_blocks import TextBlock

    return TextBlock(
        page=0,
        rect=pymupdf.Rect(0, 0, 1, 1),
        line_rects=[],
        text="",
        size=size,
        color="#000",
        bold=False,
        span_lines=span_lines,
    )


def test_italic_letters_between_math_brackets_are_a_formula():
    """In a book set in Times, the P and y of P(y = 1) are body-font italics and the parentheses and equals sign are
    CMR marked as math: none is a formula on its own, together they form one inline formula. Italic words inside
    the parentheses (P(to|want)) are part of the formula too. Italic words without formula punctuation do not
    count"""
    from app.formats.pdf_units import _placeholderize

    x = [0.0]

    def span(text, font, flags=4, math=False):
        x0 = x[0]
        x[0] += 5.0 * len(text)
        out = {
            "text": text,
            "font": font,
            "size": 10.0,
            "flags": flags,
            "bbox": (x0, 100.0, x[0], 110.0),
            "origin": (x0, 108.0),
        }
        return {**out, "math": True} if math else out

    def italic(text):
        return span(text, "NimbusRomNo9L-ReguItal", flags=6)

    def cm(text):
        return span(text, "CMR10", math=True)

    line = [
        span("the two cases,", "NimbusRomNo9L-Regu"),
        italic(" P"),
        cm("("),
        italic("y"),
        cm(" ="),
        span(" 1", "NimbusRomNo9L-Regu"),
        cm(")"),
        span(" and the word", "NimbusRomNo9L-Regu"),
        italic(" great"),
        span(" here.", "NimbusRomNo9L-Regu"),
    ]
    sent, formulas = _placeholderize(_block_with_spans([line], size=10.0), 0)
    assert sent == "the two cases,{v1} and the word {i}great{/i} here."
    assert [formula["text"] for formula in formulas] == ["P(y = 1)"]

    x[0] = 0.0
    line = [
        span("probability", "NimbusRomNo9L-Regu"),
        italic(" P"),
        cm("("),
        italic("to"),
        span("|", "CMSY10", flags=6),
        italic("want"),
        cm(")"),
        span(" decreases.", "NimbusRomNo9L-Regu"),
    ]
    sent, formulas = _placeholderize(_block_with_spans([line], size=10.0), 0)
    assert sent == "probability{v1} decreases."
    assert [formula["text"] for formula in formulas] == ["P(to|want)"]
    # The sans-serif bold vectors w and x (CMSSBX10) also appear only in formulas in a book set in Times
    from app.formats.pdf_math import _CM_UPRIGHT_RE

    assert _CM_UPRIGHT_RE.match("CMSSBX10") and _CM_UPRIGHT_RE.match("ABCDEF+CMSS10")


def test_detected_formula_boxes_mark_their_letters():
    """x and y = 1 boxed by the formula detection model are inline formulas; y = 1 is set entirely in the upright body
    font and cannot be recognized by font. When the box also covers the comma after it, the comma stays in the
    body text; a box around just 0 is not a formula"""
    from app.formats.pdf_extract import _extract
    from app.formats.pdf_units import _placeholderize

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 60), "We call the input", fontsize=10, fontname="tiro")
    x0 = 40 + pymupdf.get_text_length("We call the input ", "tiro", 10)
    page.insert_text((x0, 60), "x", fontsize=10, fontname="tiit")
    x1 = x0 + pymupdf.get_text_length("x", "tiit", 10)
    page.insert_text((x1, 60), ", and set y = 1, then 0 and 1.", fontsize=10, fontname="tiro")
    boxes = [page.search_for("x,")[0], page.search_for("y = 1,")[0], page.search_for("0")[0]]

    class Detector:
        def pages(self):
            yield boxes

    [block], _frames, _scans = _extract(doc, formulas=Detector())
    sent, formulas = _placeholderize(block, 0)
    assert sent == "We call the input {v1}, and set {v2}, then 0 and 1."
    assert [formula["text"] for formula in formulas] == ["x", "y = 1"]
    # y = 1 was cut out of a whole body span, so it records the original span on the page, which cropping uses to
    # recognize it (see _outside_formula)
    [line] = page.get_text("dict")["blocks"][0]["lines"][-1:]
    [y_span] = formulas[1]["spans"]
    assert pymupdf.Rect(y_span["source"]) == pymupdf.Rect(line["spans"][-1]["bbox"])
    [block], _frames, _scans = _extract(doc)
    assert _placeholderize(block, 0)[0] == "We call the input {i}x{/i}, and set y = 1, then 0 and 1."


def test_accent_on_its_own_line_joins_the_formula_once():
    """The ˆ of ŝ forms a line of its own after the line with s and is judged a formula by itself. Once merged into the
    formula it covers, it no longer gets a placeholder of its own."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, x1, y0, y1, base):
        return {"text": text, "font": font, "size": size, "flags": 0, "bbox": (x0, y0, x1, y1), "origin": (x0, base)}

    # Recognized earlier as part of the formula and marked math
    hat = [{**span("\u02c6", "CMR10", 10.0, 311.9, 317.4, 242.0, 252.9, 250.5), "math": True}]
    line = [
        span("span ", "NimbusRomNo9L-Regu", 10.0, 280.0, 307.0, 242.2, 253.1, 250.5),
        span("s", "CMMI10", 10.0, 307.3, 312.4, 242.0, 252.9, 250.5),
        span("i,j", "CMMI8", 8.0, 312.4, 321.1, 245.9, 253.9, 252.1),
        span(". We predict a non-null answer", "NimbusRomNo9L-Regu", 10.0, 322.1, 470.0, 242.2, 253.1, 250.5),
    ]
    sent, formulas = _placeholderize(_block_with_spans([line, hat], size=10.0), 0)
    assert sent == "span {v1}. We predict a non-null answer"
    assert len(formulas) == 1
    assert formulas[0]["bbox"].x1 >= 317.4


def test_placeholderize_and_restore_roundtrip():
    from app.formats.pdf_restore import _restore_placeholders
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0=0, y0=0):
        return {"text": text, "font": font, "size": size, "bbox": (x0, y0, x0 + 10, y0 + 10), "origin": (x0, y0 + 8)}

    spans = [
        span("The formula ", "TimesNewRomanPSMT", 12.0),
        span("E=mc", "CMMI10", 12.0, x0=60),
        span("2", "CMR10", 8.5, x0=80, y0=-2),  # The superscript joins the formula run
        span(" shows energy", "TimesNewRomanPSMT", 12.0, x0=90),
    ]
    b = _block_with_spans([spans])
    sent, formulas = _placeholderize(b, 0)
    assert sent == "The formula {v1} shows energy"
    assert len(formulas) == 1
    f = formulas[0]
    assert f["name"] == "f0_1.png" and f["h"] > 0 and f["bbox"].width > 0
    out = _restore_placeholders("公式 { v 1 } 说明了能量", formulas)
    assert out == "公式 \x01i\x021\x01/i\x02 说明了能量"
    # Lost placeholder gives None (fall back to the original); hallucinated placeholders are removed
    assert _restore_placeholders("没有占位符", formulas) is None
    assert _restore_placeholders("公式 {v1} {v9}", formulas) == "公式 \x01i\x021\x01/i\x02 "


def test_placeholderize_marks_mixed_emphasis_only():
    from app.formats.pdf_units import _placeholderize

    def span(text, font="TimesNewRomanPSMT", flags=0, x0=0):
        return {
            "text": text,
            "font": font,
            "size": 12.0,
            "flags": flags,
            "bbox": (x0, 0, x0 + 20, 12),
            "origin": (x0, 10),
        }

    mixed = _block_with_spans(
        [
            [
                span("See "),
                span("this", flags=16, x0=20),
                span(" word", x0=40),
            ]
        ]
    )
    sent, formulas = _placeholderize(mixed)
    assert sent == "See {b}this{/b} word" and formulas == []

    italic = _block_with_spans(
        [
            [
                span("A "),
                span("term", font="TimesNewRomanPS-ItalicMT", x0=12),
                span(" here", x0=30),
            ]
        ]
    )
    assert _placeholderize(italic)[0] == "A {i}term{/i} here"

    uniform = _block_with_spans([[span("All bold words here", flags=16)]])
    assert "{b}" not in _placeholderize(uniform)[0]


def test_formulas_set_in_text_italic_stay_whole():
    """Old papers set variables in Times-Italic and operators in Symbol, so only small scripts and Symbol glyphs count
    as formulas. Italic variables, parentheses and equals signs next to a formula are part of it too:
    (T_i.C_p = T_j.C_r) is one whole formula and must not be split into half image, half text. Italic letters far
    from a formula and lowercase italic words (is) are still body text."""
    from app.formats.pdf_units import _placeholderize

    def span(text, x0, x1, font="Times-Italic", size=9.0, y0=652.5):
        y1 = y0 + (11.2 if size > 8 else 7.7)
        return {
            "text": text,
            "font": font,
            "size": size,
            "flags": 2 if "Italic" in font else 0,
            "bbox": (x0, y0, x1, y1),
            "origin": (x0, y0 + (8.5 if size > 8 else 5.7)),
        }

    def sub(text, x0, x1, y0=656.9):
        return span(text, x0, x1, size=6.0, y0=y0)

    roman = "Times-Roman"
    line = [
        span("form ", 30.0, 54.0, roman),
        span("(T", 54.0, 61.9),
        sub("i", 62.0, 63.6),
        span(".C", 63.6, 71.9),
        sub("p", 71.9, 74.9),
        span(" ", 74.9, 77.8),
        span("=", 77.8, 83.9),
        span(" ", 83.9, 86.8),
        span("T", 86.8, 91.8),
        sub("j", 91.8, 93.5),
        span(".C", 93.5, 101.8),
        sub("r", 101.8, 104.1),
        span(")", 104.1, 107.1),
        span(", where ", 107.1, 134.1, roman),
        span("C", 136.9, 143.0),
        sub("p", 142.9, 145.9),
        span(" ", 145.9, 148.0, roman),
        span("is", 148.0, 154.0),
        span(" a column, and ", 154.0, 230.0, roman),
        span("c", 230.0, 234.0),
        span(" is a constant.", 234.0, 290.0, roman),
    ]
    sent, formulas = _placeholderize(_block_with_spans([line], size=9.0))
    assert sent == "form {v1}, where {v2} {i}is{/i} a column, and {i}c{/i} is a constant."
    assert formulas[0]["bbox"].x0 == pytest.approx(54.0) and formulas[0]["bbox"].x1 == pytest.approx(107.1)

    y = 697.5
    display = [
        span("(PE", 54.0, 69.5, y0=y),
        sub("q", 69.5, 72.5, y0=y + 4.4),
        span(" \u2227", 71.8, 83.9, "Symbol", y0=y),
        span("PR", 83.0, 95.2, y0=y),
        sub("q", 95.2, 98.2, y0=y + 4.4),
        span(" \u21d2", 121.9, 137.6, "Symbol", y0=y),
        span("PE", 136.7, 148.9, y0=y),
        sub("v", 148.9, 151.8, y0=y + 4.4),
        span(" )", 151.8, 156.8, y0=y),
        span(" \u2227", 155.8, 168.7, "Symbol", y0=y),
    ]
    sent, formulas = _placeholderize(_block_with_spans([display], size=9.0))
    assert re.sub(r"\{v\d+\}|\s", "", sent) == ""

    # The theorem body is all italic: If next to a formula is a word, not a variable
    theorem = [
        span("If", 54.0, 60.0),
        span(" ", 60.0, 62.5),
        span("G", 62.5, 69.0, "CMMI10"),
        span(" and D have capacity", 69.0, 150.0),
    ]
    sent, _ = _placeholderize(_block_with_spans([theorem], size=9.0))
    assert sent.startswith("If {v1}")

    # The t of the ordinal t-th is in a math font, and PyMuPDF marks it italic too. An ordinal put back into the body
    # text must not be pulled into the formula next to it again
    ordinal = [
        span("and to predict the", 108.0, 176.3, "NimbusRomNo9L-Regu", 9.9, 206.5),
        {**span(" t", 176.3, 182.4, "CMMI10", 10.0, 206.5), "flags": 2},
        span("th", 182.4, 187.8, "NimbusRomNo9L-Regu", 7.0, 205.0),
        span(" ", 187.8, 190.8, "NimbusRomNo9L-Regu", 9.9, 202.8),
        span("coordinate given", 190.8, 255.9, "NimbusRomNo9L-Regu", 9.9, 206.5),
        {**span(" t", 255.9, 262.0, "CMMI10", 10.0, 206.5), "flags": 2},
        span(" + 1", 262.0, 279.1, "CMR10", 10.0, 206.5),
    ]
    sent, _ = _placeholderize(_block_with_spans([ordinal], size=9.9))
    assert "t-th" in sent


def test_punctuation_alone_does_not_switch_the_font_size():
    """The period of "in PE." is set at script size. A size change on a single punctuation mark carries no meaning, and
    if marked, the model applies {z65} to half a sentence."""
    from app.formats.pdf_units import _placeholderize

    def span(text, size=9.0, font="Times-Roman", x0=0.0):
        return {
            "text": text,
            "font": font,
            "size": size,
            "flags": 2 if "Italic" in font else 0,
            "bbox": (x0, 0, x0 + 10, size + 2),
            "origin": (x0, size),
        }

    line = [
        span("predicates in", x0=0),
        span(" PE", font="Times-Italic", x0=60),
        span(".", size=6.0, x0=72),
        span(" An equivalence class is a set of", x0=75),
        span(" small", size=7.5, x0=200),
        span(" words.", x0=230),
    ]
    sent, _ = _placeholderize(_block_with_spans([line], size=9.0))
    assert "{z" not in sent.split("An equivalence")[0]
    assert "{z" in sent.split("An equivalence")[1]


def test_placeholderize_no_formula_keeps_text_identical():
    from app.formats.pdf_units import _placeholderize

    spans = [
        {"text": "Plain sentence ", "font": "TimesNewRomanPSMT", "size": 12.0},
        {"text": "without any math.", "font": "ArialMT", "size": 12.0},
    ]
    b = _block_with_spans([spans])
    b.text = "Plain sentence without any math."
    sent, formulas = _placeholderize(b)
    assert sent == b.text and formulas == []


def test_an_equation_in_an_algorithm_box_or_a_caption_is_translated():
    """A line in an algorithm box with a single word is translated; a caption with a number is a caption no matter how
    many equals signs it has."""
    from app.formats.pdf_unit_blocks import _equation_block

    def spans(*texts):
        return [[{"text": text, "font": "Times-Roman", "size": 10.0} for text in texts]]

    assert _equation_block(_block_with_spans(spans("T = argmax s(T)")))
    assert _equation_block(_block_with_spans([[{"text": "+ max", "font": "CMSY10", "size": 10.0}]]))
    assert not _equation_block(_block_with_spans(spans("Phoenix + Calcite")))
    assert not _equation_block(_block_with_spans(spans("Figure 2: ", "α = 0.5")))
    cell = _block_with_spans(spans("Output: ", "y = BN(x)"))
    cell.clip = (0, 0, 1, 1)
    assert not _equation_block(cell)


def test_attention_formula_placeholders_end_to_end(tmp_path):
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    from app.formats.pdf_extract import extract_blocks
    from app.formats.pdf_units import _placeholderize

    blocks = extract_blocks(pymupdf.open(src))
    sent_meta = [_placeholderize(b) for b in blocks]
    n_ph = sum(len(f) for _, f in sent_meta)
    assert n_ph > 50  # The paper has many inline formulas and scripts
    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    text = pymupdf.open(out)[0].get_text() + pymupdf.open(out)[1].get_text()
    assert "{v" not in text  # All restored


def test_split_translation_at_clause_boundary():
    from app.formats.pdf_restore import _split_translation

    t = "可以看出，基线模型的性能优于采样模型。这是意料之中的。然而还可以看出模型的表现相当好。我们推测原因在此。"
    a, b = _split_translation(t, 0.6)
    assert a.endswith(("。", "，", "；")) and len(a) > len(b) * 0.8
    assert a + b == t
    # Latin text is split at spaces
    a2, b2 = _split_translation("the quick brown fox jumps over the lazy dog", 0.5)
    assert a2.endswith("fox") and a2 + b2 == "the quick brown fox jumps over the lazy dog"


def test_formula_runs_merge_requires_deep_x_overlap():
    """Merging runs across dict lines: scripts (deep overlap in x) merge into one formula; separate formulas on two
    adjacent lines (touching by only 1pt in x) must not merge."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None):
        return {
            "text": text,
            "font": font,
            "size": size,
            "bbox": (x0, y0, x1 if x1 is not None else x0 + 10, y1 if y1 is not None else y0 + 10),
            "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2),
        }

    # y and the superscript T are on two lines and almost fully overlap in x, so they merge into one
    b = _block_with_spans(
        [
            [span("see ", "TimesNewRomanPSMT", 10, 0, 50), span("y", "CMMI10", 10, 50, 50)],
            [span("T", "CMR7", 7, 55, 44, x1=59, y1=50)],
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(b, 0)
    assert len(formulas) == 1 and sent.count("{v1}") == 1

    # θ⋆ ends the first line and X^i is on the second, touching by only 1pt in x, so they are two separate formulas
    b2 = _block_with_spans(
        [
            [
                span("use ", "TimesNewRomanPSMT", 10, 0, 50),
                span("θ", "CMMI10", 10, 50, 50, x1=58),
                span(" then", "TimesNewRomanPSMT", 10, 62, 50),
            ],
            [
                span("and ", "TimesNewRomanPSMT", 10, 0, 65),
                span("X", "CMMI10", 10, 57, 65, x1=65, y1=75),
                span("i", "CMR7", 7, 60, 62, x1=63, y1=68),
            ],
        ],
        size=10.0,
    )
    sent2, formulas2 = _placeholderize(b2, 1)
    assert len(formulas2) == 2
    assert "{v1}" in sent2 and "{v2}" in sent2


def test_body_font_symbols_and_script_bases_become_formulas():
    """Greek letters in Times are formulas. A single letter touching a script joins the script. Italic words, an
    isolated equals sign and the a before a citation superscript stay in the body text."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None, flags=0):
        return {
            "text": text,
            "font": font,
            "size": size,
            "flags": flags,
            "bbox": (x0, y0, x1 if x1 is not None else x0 + 8, y1 if y1 is not None else y0 + 10),
            "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2),
        }

    greek = _block_with_spans(
        [
            [
                span("see ", "TimesNewRomanPSMT", 10, 0, 50),
                span("α", "TimesNewRomanPSMT", 10, 28, 50, x1=36),
                span(" here", "TimesNewRomanPSMT", 10, 38, 50),
            ]
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(greek)
    assert sent == "see {v1} here" and len(formulas) == 1

    base = _block_with_spans(
        [
            [span("let ", "TimesNewRomanPSMT", 10, 0, 50), span("x", "TimesNewRomanPSMT", 10, 30, 50, x1=36)],
            [span("i", "TimesNewRomanPSMT", 7, 34, 44, x1=38, y1=50)],
        ],
        size=10.0,
    )
    sent_b, formulas_b = _placeholderize(base)
    assert sent_b == "let {v1}" and len(formulas_b) == 1
    assert "x" not in sent_b

    cite = _block_with_spans(
        [
            [
                span("a", "TimesNewRomanPSMT", 10, 0, 50, x1=6),
                span("1", "TimesNewRomanPSMT", 7, 8, 46, x1=12, y1=52),
                span(" word", "TimesNewRomanPSMT", 10, 14, 50),
            ]
        ],
        size=10.0,
    )
    sent_c, _ = _placeholderize(cite)
    assert sent_c.startswith("a") and "{v1}" in sent_c

    # The script is followed by an arrow, and the same run has normal-size text: the letter is still the base of the
    # script
    arrow = _block_with_spans(
        [
            [
                span("Set ", "Times-Roman", 10, 300, 456.4, x1=322.4, y1=466.9),
                span("N", "Times-Roman", 10, 327.5, 456.4, x1=335.2, y1=466.9),
                {**span("inf", "CMR7", 7, 335.2, 457.5, x1=344.6, y1=462.7), "math": True},
                {**span("BN", "CMR7", 7, 335.2, 464.2, x1=345.6, y1=469.4), "math": True},
                span(" ←", "CMSY10", 10, 346.6, 459.4, x1=359.2, y1=469.4),
                span(" N now", "Times-Roman", 10, 359.2, 456.4, x1=390, y1=466.9),
            ]
        ],
        size=10.0,
    )
    sent_a, formulas_a = _placeholderize(arrow)
    assert sent_a.startswith("Set {v1}") and len(formulas_a) == 1

    equals = _block_with_spans(
        [
            [
                span("E ", "TimesNewRomanPSMT", 10, 0, 50),
                span("=", "TimesNewRomanPSMT", 10, 14, 50, x1=22),
                span(" mc", "TimesNewRomanPSMT", 10, 24, 50),
            ]
        ],
        size=10.0,
    )
    assert "=" in _placeholderize(equals)[0]


def test_a_script_on_the_next_line_does_not_join_the_letter_above_it():
    """The glyph box of enc in H^enc at the start of the next line touches the X on the line above, with a baseline
    most of a line lower: it is the superscript of H, not a subscript of X. When X is body text it is not taken as
    the base of the script, and when X is a formula the two are not merged into one formula."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, y0, x1, y1, base):
        return {"text": text, "font": font, "size": size, "flags": 0, "bbox": (x0, y0, x1, y1), "origin": (x0, base)}

    for x_font in ("CMSSBX10", "CMBX10"):
        block = _block_with_spans(
            [
                [
                    span("is", "NimbusRomNo9L-Regu", 10, 160, 464, 167, 476, 473.3),
                    span(" X", x_font, 10, 167, 466, 177, 476, 473.3),
                    span(", in cross attention the input is", "NimbusRomNo9L-Regu", 10, 177, 464, 421, 476, 473.3),
                ],
                [
                    span("H", "CMSSBX10", 10, 160, 477, 168, 487, 485.3),
                    span("enc", "NimbusRomNo9L-ReguItal", 7.4, 168, 475, 178, 484, 481.7),
                    span(" is of shape", "NimbusRomNo9L-Regu", 10, 178, 476, 226, 488, 485.3),
                ],
            ],
            size=10.0,
        )
        sent, formulas = _placeholderize(block)
        assert not any("X" in f["text"] and "enc" in f["text"] for f in formulas), x_font
        assert any(f["text"] == "enc" for f in formulas), x_font
        assert "H{v" in sent or "H{/b}{v" in sent, sent


def test_math_roman_pieces_join_the_formula_and_ordinals_stay_text():
    """CMR digits, parentheses and operator names touching a formula join the same placeholder. i^{th} stays in the
    body text and is sent for translation as i-th. Distant digits and digits in Times are not taken in."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, y0, x1, y1=None):
        return {
            "text": text,
            "font": font,
            "size": size,
            "bbox": (x0, y0, x1, y1 if y1 is not None else y0 + size),
            "origin": (x0, (y1 if y1 is not None else y0 + size) - 2),
        }

    equation = _block_with_spans(
        [
            [
                span("When", "NimbusRomNo9L-Regu", 10, 0, 50, 28),
                span("ε", "CMMI10", 10, 30, 50, 36),
                span("i", "CMMI7", 7, 36, 53, 40, 60),
                span("=", "CMR10", 10, 44, 50, 52),
                span(" ", "CMR10", 10, 52, 50, 56),
                span("1", "CMR10", 10, 56, 50, 62),
                span(",", "NimbusRomNo9L-Regu", 10, 62, 50, 66),
            ]
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(equation)
    assert len(formulas) == 1
    assert sent == "When{v1},"
    assert "1" in formulas[0]["text"]

    wrapped = _block_with_spans(
        [
            [
                span("taken as the ", "NimbusRomNo9L-Regu", 10, 40, 80, 130),
                span("arg max", "CMR10", 10, 130, 80, 180),
                span("s", "CMMI7", 7, 180, 84, 186, 90),
                span("P", "CMMI10", 10, 188, 80, 196),
                span("(", "CMR10", 10, 197, 80, 203),
                span("y", "CMMI10", 10, 203, 80, 210),
                span("=", "CMR10", 10, 220, 80, 228),
            ],
            [
                span("s", "CMMI10", 10, 40, 94, 46),
                span(")", "CMR10", 10, 48, 94, 54),
                span(". This", "NimbusRomNo9L-Regu", 10, 56, 94, 90),
            ],
        ],
        size=10.0,
    )
    sent_w, formulas_w = _placeholderize(wrapped)
    assert sent_w.count("{v") == 1
    assert "arg" not in sent_w and "P" not in sent_w
    assert "{v1}." in sent_w and " ." not in sent_w
    assert "arg max" in formulas_w[0]["text"]
    assert len(formulas_w[0]["parts"]) == 2
    assert formulas_w[0]["parts"][0]["bbox"].x0 >= 130
    assert formulas_w[0]["parts"][1]["bbox"].x0 <= 40

    # The ∪ at the line end reaches into the next line by its CMSY font box and overlaps the {γ} at the start of
    # that line: they are still two pieces of a wrapped formula
    def ink(item, top, bottom):
        item["ink"] = (item["bbox"][0], top, item["bbox"][2], bottom)
        return item

    union = _block_with_spans(
        [
            [
                span("the parameters ", "NimbusRomNo9L-Regu", 10, 40, 430.4, 510, 443.7),
                ink(span("Θ", "CMR10", 10, 518.8, 433.4, 526.5, 443.4), 433.8, 440.9),
                span(" ", "CMSY10", 10, 526.5, 433.4, 533.4, 450.4),
                ink(span("∪", "CMSY10", 10, 533.4, 433.4, 540, 450.4), 434.2, 440.9),
            ],
            [
                ink(span("{", "CMSY10", 10, 40, 443.9, 45, 460.9), 445.3, 455.3),
                ink(span("γ", "CMMI10", 10, 45, 445.3, 51, 455.3), 448.0, 455.3),
                ink(span("}", "CMSY10", 10, 51, 443.9, 56, 460.9), 445.3, 455.3),
                span(" to train", "NimbusRomNo9L-Regu", 10, 58, 442.4, 100, 455.7),
            ],
        ],
        size=10.0,
    )
    sent_u, formulas_u = _placeholderize(union)
    assert sent_u.count("{v") == 1
    assert len(formulas_u[0]["parts"]) == 2
    assert formulas_u[0]["parts"][0]["bbox"].x0 >= 518
    assert formulas_u[0]["parts"][1]["bbox"].x1 <= 56

    # Inline fraction 1/√dk: numerator and denominator are both small, and the numerator line comes last in the
    # block; it is still one formula
    fraction = _block_with_spans(
        [
            [
                span("we scale the dot products by ", "Times-Roman", 10, 300, 606, 438, 616),
                span("√", "CMSY7", 7, 440.7, 606.2, 447, 619.3),
                span("dk", "CMMI7", 7, 447, 606.2, 455.2, 619.3),
                span(".", "Times-Roman", 10, 456, 606, 459, 616),
            ],
            [span("1", "CMR7", 7, 446.3, 603.6, 450.2, 610.6)],
        ],
        size=10.0,
    )
    sent_f, formulas_f = _placeholderize(fraction)
    assert sent_f.count("{v") == 1 and sent_f.endswith("{v1}.")
    assert formulas_f[0]["bbox"].y0 < 604

    # Footnote marks at the start of two lines are also small and vertically aligned, one line apart; each is
    # separate
    notes = _block_with_spans(
        [
            [
                span("†", "CMSY6", 6, 108, 680, 112, 686),
                span("Work performed at A.", "Times-Roman", 9, 113, 679, 220, 689),
            ],
            [
                span("‡", "CMSY6", 6, 108, 691, 112, 697),
                span("Work performed at B.", "Times-Roman", 9, 113, 690, 220, 700),
            ],
        ],
        size=9.0,
    )
    sent_n, _ = _placeholderize(notes)
    assert sent_n.count("{v") == 2

    ordinal = _block_with_spans(
        [
            [
                span("of the ", "NimbusRomNo9L-Regu", 10, 0, 50, 40),
                span("i", "CMMI10", 10, 40, 50, 44),
                span("th", "CMMI7", 7, 44, 44, 54, 51),
                span(" mini-batch", "NimbusRomNo9L-Regu", 10, 56, 50, 120),
            ]
        ],
        size=10.0,
    )
    sent_o, formulas_o = _placeholderize(ordinal)
    assert formulas_o == []
    assert sent_o == "of the i-th mini-batch"

    distant = _block_with_spans(
        [
            [
                span("x", "CMMI10", 10, 0, 50, 8),
                span(" see page ", "NimbusRomNo9L-Regu", 10, 10, 50, 70),
                span("1", "CMR10", 10, 80, 50, 86),
            ]
        ],
        size=10.0,
    )
    sent_d, formulas_d = _placeholderize(distant)
    assert len(formulas_d) == 1 and "1" in sent_d

    times_digit = _block_with_spans(
        [
            [
                span("x", "CMMI10", 10, 0, 50, 8),
                span("1", "TimesNewRomanPSMT", 10, 10, 50, 16),
            ]
        ],
        size=10.0,
    )
    sent_t, _ = _placeholderize(times_digit)
    assert "1" in sent_t

    # The closing " + 1" and " = 20" of a formula are set as one CMR span, and the same holds for Latin Modern
    for roman in ("CMR10", "LMRoman10-Regular"):
        tail = _block_with_spans(
            [
                [
                    span("takes only ", roman, 10, 0, 50, 46),
                    span("A", "CMMI10", 10, 47, 50, 54),
                    span(" + 1", roman, 10, 54, 50, 73),
                    span(" values, so ", roman, 10, 77, 50, 120),
                    span("A", "CMMI10", 10, 121, 50, 128),
                    span(" = 20", roman, 10, 128, 50, 151),
                    span(" draws", roman, 10, 154, 50, 180),
                ]
            ],
            size=10.0,
        )
        sent_a, formulas_a = _placeholderize(tail)
        assert sent_a == "takes only {v1} values, so {v2} draws"
        assert [f["text"].replace(" ", "") for f in formulas_a] == ["A+1", "A=20"]

    # The decimal point is a CMMI ".", and the second half of the number shares a span with body text: "= 0" is not
    # taken in, so 0.31 is not split
    decimal = _block_with_spans(
        [
            [
                span("gives a slightly higher", "CMR10", 10, 0, 50, 100),
                span(" B/T", "CMMI10", 10, 100, 50, 120),
                span(" ", "CMR10", 10, 120, 50, 123),
                span("= 0", "CMR10", 10, 123, 50, 141),
                span(".", "CMMI10", 10, 141, 50, 144),
                span("31 likely", "CMR10", 10, 144, 50, 180),
            ]
        ],
        size=10.0,
    )
    sent_dec, _ = _placeholderize(decimal)
    assert sent_dec == "gives a slightly higher{v1} = 0.31 likely"

    # A lone CMMI ">" touching the following "= 10¹¹" is taken into the formula with it
    bound = _block_with_spans(
        [
            [
                span("of stellar mass", "CMR10", 10, 0, 50, 70),
                span(" >", "CMMI10", 10, 70, 50, 81),
                span("= 10", "CMR10", 10, 81, 50, 101),
                span("11", "CMR7", 7, 101, 48, 108, 55),
                span(" in the nearby", "CMR10", 10, 108, 50, 160),
            ]
        ],
        size=10.0,
    )
    sent_b, formulas_b = _placeholderize(bound)
    assert sent_b == "of stellar mass{v1} in the nearby"
    assert formulas_b[0]["text"].replace(" ", "").startswith(">=10")

    # A formula set in Times: the italic L is followed by the upright subscript CE, and L is still taken into the ∂
    # formula
    partial = _block_with_spans(
        [
            [
                span("the derivative ", "Times-Roman", 10, 0, 50, 60),
                span("∂", "CMSY10", 10, 60, 50, 65),
                span("L", "Times-Italic", 10, 65, 50, 71),
                span("CE", "Times-Roman", 10, 71, 50, 84),
                span(" is", "Times-Roman", 10, 84, 50, 96),
            ]
        ],
        size=10.0,
    )
    _, formulas_p = _placeholderize(partial)
    assert formulas_p[0]["text"].replace(" ", "").startswith("∂L")


def test_line_wrapped_formula_is_one_placeholder():
    """y_{t-1}= at the line end and s|h_{t-1} at the start of the next line are one formula: one placeholder, two
    pieces, not merged into a large box spanning the whole line. A formula in the middle of a line is not merged
    with a formula at the start of the next line."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, y0, x1, y1=None):
        return {
            "text": text,
            "font": font,
            "size": size,
            "bbox": (x0, y0, x1, y1 if y1 is not None else y0 + size),
            "origin": (x0, (y1 if y1 is not None else y0 + size) - 2),
        }

    wrapped = _block_with_spans(
        [
            [
                span("take arg max ", "TimesNewRomanPSMT", 10, 40, 80, 150),
                span("y", "CMMI10", 10, 470, 80, 478),
                span("=", "CMR10", 10, 490, 80, 504),
            ],
            [span("s|h", "CMMI10", 10, 40, 92, 70), span(").", "TimesNewRomanPSMT", 10, 72, 92, 90)],
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(wrapped)
    assert sent.count("{v") == 1
    assert "s|h" not in sent and "y=" not in sent.replace(" ", "")
    assert len(formulas) == 1
    parts = formulas[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["bbox"].x0 >= 470
    assert parts[1]["bbox"].x0 <= 40
    assert parts[0]["bbox"].x1 < 510 and parts[1]["bbox"].width < 40

    separate = _block_with_spans(
        [
            [
                span("see ", "TimesNewRomanPSMT", 10, 40, 80, 70),
                span("x", "CMMI10", 10, 80, 80, 90),
                span(" then more words here", "TimesNewRomanPSMT", 10, 92, 80, 220),
            ],
            [span("y", "CMMI10", 10, 40, 92, 50), span(" follows", "TimesNewRomanPSMT", 10, 54, 92, 110)],
        ],
        size=10.0,
    )
    sent2, formulas2 = _placeholderize(separate)
    assert len(formulas2) == 2
    assert "{v1}" in sent2 and "{v2}" in sent2


def test_placeholderize_absorbs_accent_over_formula():
    """The ^ of ŷ is often a separate span (body font, not judged a formula); left in the body text it adds a stray
    symbol next to the formula image. An accent that falls on the formula bbox joins the formula and is removed
    from the body text."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None):
        return {
            "text": text,
            "font": font,
            "size": size,
            "bbox": (x0, y0, x1 if x1 is not None else x0 + 10, y1 if y1 is not None else y0 + 10),
            "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2),
        }

    b = _block_with_spans(
        [
            [
                span("an estimate ", "TimesNewRomanPSMT", 10, 0, 50),
                span("y", "CMMI10", 10, 40, 50, x1=48, y1=60),
                span(" coming", "TimesNewRomanPSMT", 10, 60, 50),
            ],
            [{**span("^", "CMR10", 10, 41, 42, x1=45, y1=48), "origin": (41, 56)}],  # Hat: own dict line, raised base
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(b, 0)
    assert "^" not in sent
    assert len(formulas) == 1
    assert formulas[0]["bbox"].y0 <= 42  # The top of the accent is inside the crop area
    # The same for the breve in M̆ and the tilde in Ẽ
    for accent in ("˘", "˜"):
        b1 = _block_with_spans(
            [
                [
                    span("costs at least ", "TimesNewRomanPSMT", 10, 0, 50),
                    span("M", "CMMI10", 10, 40, 50, x1=48, y1=60),
                    span(" over", "TimesNewRomanPSMT", 10, 60, 50),
                ],
                [{**span(accent, "CMR10", 10, 41, 42, x1=46, y1=48), "origin": (41, 56)}],
            ],
            size=10.0,
        )
        sent1, formulas1 = _placeholderize(b1, 0)
        assert accent not in sent1 and len(formulas1) == 1
    # A ^ far from any formula is not absorbed (the body text may really contain the symbol)
    b2 = _block_with_spans(
        [
            [span("y", "CMMI10", 10, 0, 50, x1=8, y1=60), span(" then ^ alone", "TimesNewRomanPSMT", 10, 100, 50)],
        ],
        size=10.0,
    )
    sent2, _ = _placeholderize(b2, 0)
    assert "^" in sent2
    # The diaeresis of naïve on the line above does not count toward the formula on the next line
    b3 = _block_with_spans(
        [
            [
                span("usual (na", "TimesNewRomanPSMT", 10, 0, 40, x1=40, y1=50),
                {**span("\u00a8", "TimesNewRomanPSMT", 10, 40, 40, x1=44, y1=50), "origin": (40, 48)},
                span("\u0131ve)", "TimesNewRomanPSMT", 10, 40, 40, x1=60, y1=50),
            ],
            [span("\u2207", "CMSY10", 10, 38, 50, x1=48, y1=62)],
        ],
        size=10.0,
    )
    sent3, _ = _placeholderize(b3, 0)
    assert "\u00a8" in sent3


def test_heading_does_not_turn_the_following_sentence_into_a_formula():
    """Even when the heading and the lead paragraph are still in one block, 9pt body text must not be cropped as a
    formula image for being smaller than 12pt."""
    from app.formats.pdf_units import _placeholderize

    def span(text: str, size: float, x0: float, y0: float, font: str) -> dict[str, object]:
        return {
            "text": text,
            "font": font,
            "size": size,
            "flags": 16 if size >= 12 else 0,
            "bbox": (x0, y0, x0 + max(8, len(text) * size * 0.45), y0 + size),
            "origin": (x0, y0 + size * 0.8),
        }

    heading = [
        span("4. ", 12, 0, 0, "NimbusRomNo9L-Medi"),
        span("QUERY OPTIMIZATION", 12, 24, 0, "NimbusRomNo9L-Medi"),
    ]
    words = "We describe Orca optimization workflow in Section".split()
    body = [span(word + " ", 9, index * 42, 16, "CMR9") for index, word in enumerate(words)]
    sent, formulas = _placeholderize(_block_with_spans([heading, body], size=12), 0)
    assert formulas == []
    assert "{v" not in sent
    assert "workflow" in sent


def test_heading_number_kept_in_source_form():
    """Heading section numbers keep their original form: the model follows the prompt and writes Chinese numerals
    ("第二章第一节"), which academic papers need back as '2.1'. Sentences in paragraphs that start with a number are
    not affected."""
    from app.formats.pdf_restore import _normalize_heading_number as norm

    assert norm("2 Proposed Approach", "第二章 所提方法") == "2 所提方法"
    assert norm("2.1 Model", "第二章第一节 模型") == "2.1 模型"
    assert norm("4.3 Speech Recognition", "第四章第三节 语音识别") == "4.3 语音识别"
    assert norm("2.1 Model", "2.1 模型") == "2.1 模型"  # Idempotent
    # A long paragraph starting with a number: unchanged
    para = "40 dimensional log Mel filter banks and their first and second order derivatives were used"
    assert norm(para, "使用了 40 维对数 Mel 滤波器组") == "使用了 40 维对数 Mel 滤波器组"
    # A short sentence with sentence-final punctuation: unchanged
    assert norm("3 states remain.", "还剩 3 个状态。") == "还剩 3 个状态。"
    # No prefix when the translation has no section number; for a skipped block the original must come back
    # unchanged
    assert norm("10 epochs were used", "使用了 10 个轮次") == "使用了 10 个轮次"
    assert norm("2.5 hours later", "两个半小时后") == "两个半小时后"
    assert norm("2. Smith 2019", "2. Smith 2019") == "2. Smith 2019"
    assert norm("2 Smith 2019", "2. Smith 2019") == "2 Smith 2019"
    # When the original number has a trailing period, the period stays
    assert norm("1. Exploited the sortedness of join input", "1. 利用了连接输入的有序性") == "1. 利用了连接输入的有序性"


def test_section_and_item_numbers_are_not_sent_for_translation():
    """Section and item numbers are taken off before translation and put back unchanged when writing. A number at the
    start of a body sentence is not taken off."""
    from app.formats.pdf_restore import _with_label
    from app.formats.pdf_units import _peel_number

    assert _peel_number("B.3.2. DeepSeek-R1 Cold Start", "p") == ("B.3.2.", "DeepSeek-R1 Cold Start")
    assert _peel_number("4 Experiments", "h2") == ("4", "Experiments")
    assert _peel_number("{b}3{/b} {b}Method{/b}", "h2") == ("{b}3{/b}", "{b}Method{/b}")
    # Bold runs from the number into the text: it is closed on the number and reopened on the text
    assert _peel_number("{b}17.2.1 The Encoder{/b} ENCODEC {b}model{/b}", "h3") == (
        "{b}17.2.1{/b}",
        "{b}The Encoder{/b} ENCODEC {b}model{/b}",
    )
    assert _peel_number("1.{i} Intractability{/i}: the case where", "p") == (
        "1.",
        "{i} Intractability{/i}: the case where",
    )
    assert _peel_number("(a) Let u be the view of least cost.", "p") == ("(a)", "Let u be the view of least cost.")
    assert _peel_number("2 samples were drawn from each model and we report the mean.", "p")[0] == ""
    assert _peel_number("1.5 times faster than the baseline, a large gain.", "p")[0] == ""
    assert _peel_number("I. Goodfellow proposed adversarial nets in 2014.", "p")[0] == ""
    assert _peel_number("3.17", "p")[0] == ""
    assert _with_label("B.3.2.", "DeepSeek-R1 冷启动") == "B.3.2. DeepSeek-R1 冷启动"
    assert _with_label("", "实验") == "实验"


def test_stray_braces_around_marks_are_dropped():
    """Extra braces the model puts around marks are removed. Braces present in the original (the set {k=1}) stay."""
    from app.formats.pdf_restore import _drop_stray_braces

    sent = "called a {b}fragment{/b}. {c0000ff}Fox Tree{/c} showed {v1}"
    out = _drop_stray_braces("叫做{b}断裂{/b}}。{{c0000ff}Fox Tree{/c}}证明了{v1}", sent)
    assert out == "叫做{b}断裂{/b}。{c0000ff}Fox Tree{/c}证明了{v1}"
    assert _drop_stray_braces("集合 {k=1} 是", "the set {k=1} is") == "集合 {k=1} 是"


def test_heading_number_skips_body_blocks():
    from app.formats.pdf_restore import _heading_number_for

    assert _heading_number_for("p", "2 samples", "第二章 样本") == "第二章 样本"
    assert _heading_number_for("h2", "2 Proposed Approach", "第二章 所提方法") == "2 所提方法"
    # A section heading at body size (kind is p): a dotted section number is changed back as well
    assert _heading_number_for("p", "2.1 Model", "第二章第一节 模型") == "2.1 模型"
    # A heading at body size whose translation has no section number: no prefix
    assert _heading_number_for("p", "2.1 Model", "模型") == "模型"
    # Also changed back when the model writes "第二节" (without "章")
    assert _heading_number_for("h2", "2 Proposed Approach", "第二节 所提方法") == "2 所提方法"
    assert (
        _heading_number_for("h2", "2.1 Massively Parallel Processing", "第二点一节 大规模并行处理")
        == "2.1 大规模并行处理"
    )
    assert _heading_number_for("h2", "4.2 Parallel Query Optimization", "四点二 并行查询优化") == "4.2 并行查询优化"
    assert _heading_number_for("h2", "8. RELATED WORK", "八, 相关工作") == "8. 相关工作"
    assert _heading_number_for("h2", "9. SUMMARY", "九, 总结") == "9. 总结"
    assert _heading_number_for("h2", "10. REFERENCES", "十, 参考文献") == "10. 参考文献"
    assert _heading_number_for("h2", "8.1 Query Optimization Foundations", "八点一 查询优化基础") == "8.1 查询优化基础"
    assert _heading_number_for("h2", "2.2 SQL on Hadoop", "第2.2节 Hadoop 上的 SQL") == "2.2 Hadoop 上的 SQL"
    assert _heading_number_for("h2", "1.3 Related Work", "一.三. 相关工作") == "1.3 相关工作"
    assert _heading_number_for("h2", "2.5 Composite Lattices", "二．五 复合格") == "2.5 复合格"
    # Three-level section numbers, possibly with a space between levels
    assert (
        _heading_number_for("h2", "3.1.3 Dealing with scale", "第三章第一节第三小节 处理规模问题")
        == "3.1.3 处理规模问题"
    )
    assert (
        _heading_number_for("h2", "10.1.2 Contextual Embeddings", "第十章 第一节 第二小节 上下文嵌入")
        == "10.1.2 上下文嵌入"
    )


def test_split_translation_does_not_break_sentinel():
    from app.formats.pdf_restore import _split_translation

    sentinel = "\x01i\x0212\x01/i\x02"
    text = "甲" * 8 + sentinel + "乙" * 8
    a, b = _split_translation(text, 0.5)
    assert a + b == text
    assert sentinel in a or sentinel in b
    assert "\x01" not in a.replace(sentinel, "")
    assert "\x01" not in b.replace(sentinel, "")


def test_hyphen_bridge_keeps_the_page_boundary():
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_units import _prepare_unit

    def blk(text, page):
        return TextBlock(
            page=page, rect=pymupdf.Rect(40, 40, 200, 52), line_rects=[], text=text, size=9, color="#000", bold=False
        )

    sent, formulas, name_i = _prepare_unit([blk("debug cus-", 0), blk("tomer issues later", 1)], 0)
    assert "customer" in sent
    assert "{|}" in sent
    assert "cus-" not in sent
    assert "issues later" in sent
    assert formulas == []
    assert name_i == 2


def test_cross_page_formulas_keep_own_page_and_name():
    from app.formats.pdf_restore import _assign_restored_parts
    from app.formats.pdf_units import _prepare_unit

    def span(text, font, size, x0, y0):
        return {"text": text, "font": font, "size": size, "bbox": (x0, y0, x0 + 12, y0 + 10), "origin": (x0, y0 + 8)}

    def block(page, prose):
        spans = [span(prose, "TimesNewRomanPSMT", 12, 0, 0), span("E", "CMMI10", 12, 40, 0)]
        b = _block_with_spans([spans])
        b.page = page
        b.text = prose + "E"
        return b

    unit = [block(0, "sequence "), block(1, "length ")]
    sent, formulas, name_i = _prepare_unit(unit, 0)
    assert name_i == 2
    assert {f["name"] for f in formulas} == {"f0_1.png", "f1_1.png"}
    assert [f["page"] for f in formulas] == [0, 1]
    assert formulas[0]["owner"] == 0 and formulas[1]["owner"] == 1
    assert "{v1}" in sent and "{v2}" in sent
    restored = sent.replace("{v1}", "\x01i\x021\x01/i\x02").replace("{v2}", "\x01i\x022\x01/i\x02")
    parts, lists = _assign_restored_parts(restored, unit, formulas)
    from app.formats.pdf_blocks import SPILL_TAIL

    assert parts[1] == SPILL_TAIL and lists[1] == []
    assert [f["page"] for f in lists[0]] == [0, 1]
    assert parts[0].count("\x01i\x021\x01/i\x02") == 1
    assert parts[0].count("\x01i\x022\x01/i\x02") == 1


def test_boundary_echo_keeps_repeated_cells_and_short_phrases():
    from app.formats.pdf_restore import _strip_boundary_echo

    assert _strip_boundary_echo("WSJ only", "WSJ only", "仅 WSJ", "仅 WSJ") == "仅 WSJ"
    assert _strip_boundary_echo("see below", "next", "如下所示", "如下所示的结果") == "如下所示的结果"
    prev = "前文结束序列），而目标输出"
    cur = "序列），而目标输出才是下一段"
    assert _strip_boundary_echo("alpha beta", "gamma delta", prev, cur) == "才是下一段"
    # A new paragraph after a heading that starts with words from the heading
    head_src, para_src = (
        "4.7.2 Applying softmax in logistic regression",
        "When we apply softmax for logistic regression",
    )
    head_t, para_t = "4.7.2 在逻辑回归中应用 softmax", "在逻辑回归中应用 softmax 时, 输入与 sigmoid 一样"
    assert _strip_boundary_echo(head_src, para_src, head_t, para_t) == para_t


def test_split_translation_balances_style_marks():
    """Style marks stay paired when text is split across pages: the left half gets the closing marks, the right half
    the opening ones, and the font size reopened is the one still open."""
    from app.formats.pdf_flow import strip_style_marks
    from app.formats.pdf_restore import _split_translation

    t = "前面一段结束。{b}后半加粗的内容比较长，跨页了{/b}。"
    left, right = _split_translation(t, 0.55)
    assert "{b}" in left and left.endswith("{/b}")
    assert right.startswith("{b}")
    assert left.count("{b}") == left.count("{/b}")
    assert right.count("{b}") == right.count("{/b}")
    assert strip_style_marks(left) + strip_style_marks(right) == strip_style_marks(t)

    sized = "{z80}小号结束。{/z}{z120}大号内容比较长，会跨页继续写下去直到句号。{/z}"
    left, right = _split_translation(sized, 0.45)
    assert left.endswith("{/z}")
    assert right.startswith("{z120}")
    assert "{z80}" not in right
    assert strip_style_marks(left) + strip_style_marks(right) == strip_style_marks(sized)

    nested = "{z120}{b}{i}强调的内容比较长，会跨页继续写下去直到句号。{/i}{/b}{/z}"
    left, right = _split_translation(nested, 0.4)
    assert left.endswith("{/i}{/b}{/z}")
    assert right.startswith("{z120}{b}{i}")
    assert strip_style_marks(left) + strip_style_marks(right) == strip_style_marks(nested)


def test_wrapped_formula_tail_stays_one_placeholder_without_a_wide_box():
    """A formula wrapped across visual lines becomes two pieces of one placeholder. The bboxes are not merged,
    otherwise the crop would include the full-width blank space between the two lines."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, y0, x1=None, y1=None):
        return {
            "text": text,
            "font": font,
            "size": size,
            "bbox": (x0, y0, x1 if x1 is not None else x0 + 10, y1 if y1 is not None else y0 + 10),
            "origin": (x0, (y1 if y1 is not None else y0 + 10) - 2),
        }

    b = _block_with_spans(
        [
            [
                span("taken as ", "TimesNewRomanPSMT", 10, 100, 100),
                span("y", "CMMI10", 10, 400, 100, x1=405, y1=110),
                span("t−1 =", "CMR7", 7, 405, 104, x1=415, y1=110),
            ],
            [
                span("s|h", "CMMI10", 10, 108, 114, x1=120, y1=124),
                span("t−1", "CMR7", 7, 120, 117, x1=128, y1=124),
                span("). This", "TimesNewRomanPSMT", 10, 130, 114),
            ],
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(b, 0)
    assert len(formulas) == 1
    assert sent.count("{v") == 1
    parts = formulas[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["bbox"].x0 >= 400
    assert parts[1]["bbox"].x0 <= 108
    assert parts[0]["bbox"] | parts[1]["bbox"] != formulas[0]["bbox"]


def test_caption_word_in_bold_cm_is_not_a_formula():
    from app.formats.pdf_formula_runs import _is_formula_span

    assert _is_formula_span({"font": "CMBX8", "text": "of", "size": 8.0}, 9.0) is False
    assert _is_formula_span({"font": "CMBX10", "text": "x", "size": 10.0}, 10.0) is True


def test_block_of_typewriter_lines_is_code_and_inline_names_are_not():
    """A block that is almost all monospace (SQL) is kept as it is. A sentence with a few monospace attribute names in
    it is translated."""
    from app.formats.pdf_unit_blocks import _code_block

    def span(text, font):
        return {"text": text, "font": font, "size": 10.0, "bbox": (0, 0, 10, 10), "origin": (0, 8)}

    sql = _block_with_spans(
        [
            [
                span("SELECT Part, Customer, SUM(SP) ", "Courier"),
                span("AS ", "Times-Roman"),
                span("TotalSales", "Courier"),
            ],
            [span("FROM R", "Times-Roman")],
        ]
    )
    assert _code_block(sql)
    prose = _block_with_spans(
        [
            [
                span("There are three dimensions we are interested in: ", "Times-Roman"),
                span("part, supplier, ", "Courier"),
                span("and customer.", "Times-Roman"),
            ]
        ]
    )
    assert not _code_block(prose)
    # A table row: an italic row name followed by several cells of monospace notation
    row = _block_with_spans(
        [
            [
                span("rooster", "NimbusRomNo9L-ReguItal"),
                span(" -female, +chicken, +adult", "Courier"),
            ]
        ]
    )
    assert _code_block(row)


def test_lone_relation_symbol_from_a_symbol_font_stays_a_formula():
    """The ≺ in CMSY is recognized by OCR as <. Even on its own it is a formula and must not be taken for a less-than
    sign in body text."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, x0):
        return {"text": text, "font": font, "size": 10.0, "bbox": (x0, 0, x0 + 20, 10), "origin": (x0, 8)}

    block = _block_with_spans(
        [
            [
                span("Thus (part) ", "Times-Roman", 0),
                span("<", "CMSY10", 60),
                span(" (part, customer).", "Times-Roman", 70),
            ]
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(block)
    assert sent == "Thus (part) {v1} (part, customer)." and len(formulas) == 1


def test_inline_typewriter_words_keep_their_glyphs():
    """Monospace attribute names in a sentence are replaced by placeholders and put back as the original glyphs. Blocks
    entirely in monospace are not handled here; those are code blocks."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, x0):
        return {"text": text, "font": font, "size": 10.0, "bbox": (x0, 0, x0 + 40, 10), "origin": (x0, 8)}

    block = _block_with_spans(
        [
            [
                span("the total sales of ", "Times-Roman", 0),
                span("part", "Courier", 90),
                span(" that was bought", "Times-Roman", 120),
            ]
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(block)
    assert sent == "the total sales of {v1} that was bought" and len(formulas) == 1


def test_list_number_keeps_its_point():
    from app.formats.pdf_restore import _list_number

    assert _list_number("1. x is a partial order, and", "1。x 是一个偏序，并且") == "1. x 是一个偏序，并且"
    assert _list_number("2. There is a top element.", "2. 存在一个顶元素。") == "2. 存在一个顶元素。"
    assert _list_number("Plain sentence.", "1。不动") == "1。不动"
    assert _list_number("Figure 1: The eight views", "图一: 八个视图") == "图 1: 八个视图"
    assert _list_number("Figure 12: Bars", "图 12: 条") == "图 12: 条"


def _page_data(*lines):
    """Minimal get_text("dict") structure: each argument is the span list of one line as (font, text)."""
    return {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"font": font, "text": text, "size": 10.0} for font, text in spans]} for spans in lines
                ],
            }
        ]
    }


def test_computer_modern_upright_is_math_when_the_body_is_times():
    """In a paper set in Times, CMR only writes log, min, parentheses and equals signs, so it is a formula. In a paper
    set in CM it is body text."""
    from app.formats.pdf_formula_runs import _is_formula_span
    from app.formats.pdf_math import _mark_math_roman

    times = _page_data([("NimbusRomNo9L-Regu", "We train the model to minimize " * 20), ("ABCDEF+CMR10", "log(1")])
    _mark_math_roman([times])
    log_span = times["blocks"][0]["lines"][0]["spans"][1]
    assert log_span.get("math") is True
    assert _is_formula_span({**log_span, "size": 10.0, "origin": (0, 0)}, 10.0)

    modern = _page_data([("CMR10", "We train the model to minimize " * 20), ("CMMI10", "x")])
    _mark_math_roman([modern])
    assert "math" not in modern["blocks"][0]["lines"][0]["spans"][0]


def test_a_font_that_writes_math_alphabet_letters_is_math():
    """A math italic missing from the font list (newtxmath with Cochineal is called cochMI) maps letters to math
    letters such as 𝑁 and 𝑡, and parentheses and commas in the same font count as formula too. A font that only
    occasionally puts a math letter into body text does not count."""
    from app.formats.pdf_blocks import _font_is_math, _span_is_math
    from app.formats.pdf_math import _mark_math_alphabet

    data = _page_data(
        [
            ("Cochineal-Roman", "fitted a hyperbola "),
            ("zxcMI", "𝑁("),
            ("zxcMI", "𝑡"),
            ("Cochineal-Roman", " to two millennia"),
        ],
        [("Cochineal-Roman", "where 𝑥 is the count of " * 3)],
    )
    _mark_math_alphabet([data])
    first, second = (line["spans"] for line in data["blocks"][0]["lines"])
    assert _span_is_math(first[1]) and _span_is_math(first[2])
    assert not _span_is_math(first[0]) and not _span_is_math(second[0])
    assert _font_is_math("NewTXMI7") and _font_is_math("cochMRM") and not _font_is_math("Cochineal-Italic")


def test_tex_spacing_accents_are_composed_before_translation():
    from app.formats.pdf_units import _compose_accents

    assert _compose_accents("Universit´e de Montr´eal") == "Université de Montréal"
    assert _compose_accents("Fr´ed´eric, na¨ıve, ˆetre") == "Frédéric, naïve, être"
    assert _compose_accents("Fran¸cois, gar c¸on") == "François, gar çon"
    assert _compose_accents("f(x)´ and x¸") == "f(x)´ and x¸"
    # When the letter is italic or bold, the mark sits between the accent and the letter
    assert _compose_accents("the estimate ˆ{i}y{/i} and ˜{i}x,r{/i}") == "the estimate {i}ŷ{/i} and {i}x̃,r{/i}"
    assert _compose_accents("output ˆ{b}Y{/b}") == "output {b}Ŷ{/b}"


def test_a_stacked_accent_is_split_from_the_glyphs_before_it():
    """The hat in "(ˆ" overlaps the following y and is cut into a separate piece. The ˜ in a URL and the ˆ in a regex
    take their own width; they are ~ and ^ and are not cut."""
    from app.formats.pdf_math import _PageChars, _split_accents

    doc = pymupdf.open()
    page = doc.new_page()
    # Writing ˆ and ˜ needs an embedded font file; the single-byte encoding of the built-in Times would turn them
    # into other characters
    font = pymupdf.Font("tiro")
    page.insert_font(fontname="T", fontbuffer=font.buffer)
    page.insert_text((100, 100), "(\u02c6", fontname="T", fontsize=10)
    page.insert_text((100 + font.text_length("(", 10), 100), "y", fontname="T", fontsize=10)
    page.insert_text((100, 140), "nyu.edu/\u02dcroweis", fontname="T", fontsize=10)
    page.insert_text((100, 180), 'r"\u02c6The"', fontname="T", fontsize=10)
    data = page.get_text("dict")
    _split_accents(_PageChars(page), data)
    texts = [span["text"] for block in data["blocks"] for line in block["lines"] for span in line["spans"]]
    assert texts == ["(", "\u02c6", "y", "nyu.edu/~roweis", 'r"^The"']


def test_caption_number_is_restored_inside_style_marks():
    from app.formats.pdf_restore import _list_number

    assert _list_number("Figure 2: Samples", "{z110}图二:{/z} 模型样本") == "{z110}图 2:{/z} 模型样本"
    assert _list_number("Table 1: Parzen", "{z110}{b}表一：{/b}{/z} 基于") == "{z110}{b}表 1：{/b}{/z} 基于"


def test_small_caps_heading_is_sent_without_size_marks():
    """A small-caps heading with the initial one size larger and the rest one size smaller is one font, so no size mark
    is inserted inside the word."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_units import _placeholderize

    def span(text, size, x0):
        return {
            "text": text,
            "font": "NimbusRomNo9L-Regu",
            "size": size,
            "flags": 0,
            "bbox": (x0, 100, x0 + 8 * len(text), 112),
            "origin": (x0, 110),
        }

    spans = [
        span("2.1", 12.5, 100),
        span(" ", 12.5, 120),
        span("A", 12.5, 125),
        span("DAM", 10.0, 133),
        span("’", 12.5, 157),
        span("S", 10.0, 161),
        span(" ", 10.0, 169),
        span("UPDATE", 10.0, 173),
    ]
    rect = pymupdf.Rect(100, 100, 230, 112)
    block = TextBlock(0, rect, [rect], "2.1 ADAM’S UPDATE", 10.0, "#000", False, span_lines=[spans])
    sent, _ = _placeholderize(block, 0)
    assert sent == "2.1 ADAM’S UPDATE"


def test_small_caps_words_in_body_text_take_the_body_size():
    """A whole word in small caps in body text (AGENT) is one size smaller than the body; copying the small size into
    the Chinese translation would cram it together."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_units import _placeholderize

    def span(text, size, x0, base=110):
        return {
            "text": text,
            "font": "NimbusRomNo9L-Regu",
            "size": size,
            "flags": 0,
            "bbox": (x0, base - 10, x0 + 5 * len(text), base + 2),
            "origin": (x0, base),
        }

    spans = [
        span("roles like", 9.96, 100),
        span(" ", 7.97, 150),
        span("AGENT", 7.97, 154),
        span(", or the", 9.96, 180),
        span("TM", 7.97, 220, base=106),
        span(" x", 9.96, 232),
    ]
    rect = pymupdf.Rect(100, 100, 245, 112)
    block = TextBlock(0, rect, [rect], "roles like AGENT, or theTM x", 9.96, "#000", False, span_lines=[spans])
    sent, _ = _placeholderize(block, 0)
    # A raised superscript is not small caps and keeps its font size
    assert sent.startswith("roles like AGENT, or the{z")


def test_bare_chinese_section_number_is_restored():
    from app.formats.pdf_restore import _normalize_heading_number

    assert _normalize_heading_number("2 ALGORITHM", "二 算法") == "2 算法"
    assert _normalize_heading_number("2.1 Update rule", "二点一 更新规则") == "2.1 更新规则"
    assert _normalize_heading_number("3 Method", "{z125}三{/z} 方法") == "{z125}3{/z} 方法"
    assert _normalize_heading_number("1 A new idea", "一种新想法") == "一种新想法"


def test_union_box_matches_rect_union():
    from app.formats.pdf_formula_runs import _union_box

    rects = [
        pymupdf.Rect(378.9772014701512, 294.40010658, 391.92303898, 300.53540323),
        pymupdf.Rect(391.8992945173863, 212.31890825, 415.72914222, 219.31949272),
        pymupdf.Rect(5, 5, 5, 9),  # A zero-width empty box does not count
        pymupdf.Rect(9, 9, 3, 3),
        pymupdf.Rect(pymupdf.INFINITE_RECT()),
    ]
    for a in rects:
        for b in rects:
            assert tuple(_union_box(a, b)) == tuple(a | b)


def test_pdf_jobs_take_turns_with_mupdf_and_keep_the_loop_running(tmp_path, monkeypatch):
    """Two jobs running at the same time must not use MuPDF together (find_tables changes a process-wide glyph box
    setting); the MuPDF work runs in a thread, so the event loop keeps answering other requests."""
    import threading
    import time

    from app.formats import pdf, pdf_extract, pdf_units

    src = tmp_path / "a.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A sentence to translate.", fontsize=11)
    doc.save(src)

    inside = most = 0
    stages: list[tuple] = []
    # Calls made on the main thread that runs the event loop
    on_loop: list[str] = []

    def counted(real):
        def wrapper(*args, **kwargs):
            nonlocal inside, most
            if threading.current_thread() is threading.main_thread():
                on_loop.append(real.__name__)
            inside += 1
            most = max(most, inside)
            try:
                time.sleep(0.2)
                return real(*args, **kwargs)
            finally:
                inside -= 1

        return wrapper

    real_geometry = pdf_units.PageGeometry

    # Previously computed page by page on the event loop
    class SlowGeometry(real_geometry):
        @classmethod
        def from_page(cls, *args, **kwargs):
            if threading.current_thread() is threading.main_thread():
                on_loop.append("from_page")
            return real_geometry.from_page(*args, **kwargs)

    monkeypatch.setattr(pdf_units, "_extract", counted(pdf_units._extract))
    monkeypatch.setattr(pdf, "_render_by_parts", counted(pdf._render_by_parts))
    monkeypatch.setattr(pdf_extract, "PageGeometry", SlowGeometry)
    monkeypatch.setattr(pdf_units, "PageGeometry", SlowGeometry)

    async def scenario():
        for name in "ab":
            (tmp_path / name).mkdir()
        runners = [runner() for _ in "ab"]
        for r in runners:
            r.report_stage = lambda name, done=0, total=0, unit="": stages.append((name, done, total, unit))
        return await asyncio.gather(
            *(pdf.translate_pdf(src, tmp_path / name, r, False, "zh-CN") for name, r in zip("ab", runners))
        )

    outs = run(scenario())
    # While the later job waits, its job page shows that it is waiting
    assert ("等待其他任务", 0, 0, "") in stages
    # Only the calling thread is checked, not how long the loop stalls: on a busy machine calls in a thread also
    # stall the loop for a fraction of a second
    assert on_loop == []
    assert most == 1
    assert [pymupdf.open(out)[0].get_text() for [out] in outs] == [pymupdf.open(outs[0][0])[0].get_text()] * 2


def test_resumed_pdf_job_reads_back_its_layout_analysis(tmp_path, monkeypatch):
    """When a job is resumed and neither the source file nor the code has changed, the previous layout analysis is
    read back instead of analyzing page by page again; the output matches the first run. When the source file has
    changed, it is analyzed again."""
    from app.formats import pdf_units
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((40, 50), "Saved Analysis", fontsize=16)
    page.insert_text((40, 90), "A body paragraph long enough to be translated here.", fontsize=10)
    page.insert_text((40, 130), "for i in range(10): print(i)", fontsize=10, fontname="cour")
    src = tmp_path / "book.pdf"
    doc.save(src)
    saved = tmp_path / "plan.pickle"
    calls = []
    real_extract = pdf_units._extract
    monkeypatch.setattr(pdf_units, "_extract", lambda *a, **k: calls.append(1) or real_extract(*a, **k))

    def translate(out):
        out.mkdir()
        r = runner()
        r.translator.title = ""
        stages = []
        r.report_stage = lambda name, *a: stages.append(name)
        [path] = run(translate_pdf(src, out, r, False, "zh-CN", plan_path=saved))
        return pymupdf.open(path), r.translator.title, stages

    first, title, _ = translate(tmp_path / "a")
    assert saved.exists() and len(calls) == 1
    again, title_again, stages = translate(tmp_path / "b")
    assert len(calls) == 1 and "分析版面" not in stages and "读取上次的版面分析结果" in stages
    assert title_again == title == "book"
    assert again[0].get_text() == first[0].get_text()
    assert "for i in range(10)" in again[0].get_text()
    assert again[0].get_pixmap(dpi=72).samples == first[0].get_pixmap(dpi=72).samples

    doc[0].insert_text((40, 170), "Another paragraph added to the source file.", fontsize=10)
    doc.save(src)
    translate(tmp_path / "c")
    assert len(calls) == 2


def test_a_formula_at_a_line_start_keeps_its_opening_glue():
    """P( at the start of the next line begins the formula of that line. The font box of a CMSY minus sign in the
    formula on the line above reaches into this line, so comparing font boxes would attach P( to w_{n-2}w_{n-1}w_n
    on the line above and each formula would lose a piece."""
    from app.formats.pdf_units import _placeholderize

    reg, ital, cmsy, cmr = "NimbusRomNo9L-Regu", "NimbusRomNo9L-ReguItal", "CMSY10", "CMR10"
    above = [
        _glyph(
            "examples of a particular trigram ",
            reg,
            10.0,
            4,
            (142.2, 191.2, 306.7, 203.2),
            (142.2, 193.6, 306.7, 200.6),
        ),
        _glyph("w", ital, 10.0, 6, (309.7, 191.2, 316.3, 203.1), (309.7, 196.0, 316.3, 200.6)),
        _glyph("n", ital, 7.4, 6, (316.3, 195.1, 320.0, 203.9), (316.3, 198.7, 320.0, 202.0)),
        _glyph("−", cmsy, 7.4, 6, (320.0, 196.2, 325.8, 209.0), (320.0, 200.0, 325.8, 200.3)),
        _glyph("2", reg, 7.4, 4, (325.8, 195.1, 329.4, 204.0), (325.8, 197.0, 329.4, 201.9)),
        _glyph("w", ital, 10.0, 6, (329.9, 191.2, 336.6, 203.1), (329.9, 196.0, 336.6, 200.6)),
        _glyph("n", ital, 7.4, 6, (336.6, 195.1, 340.3, 203.9), (336.6, 198.7, 340.3, 202.0)),
        _glyph("−", cmsy, 7.4, 6, (340.3, 196.2, 346.0, 209.0), (340.3, 200.0, 346.0, 200.3)),
        _glyph("1", reg, 7.4, 4, (346.0, 195.1, 349.7, 204.0), (346.0, 197.0, 349.7, 201.9)),
        _glyph("w", ital, 10.0, 6, (350.2, 191.2, 356.8, 203.1), (350.2, 196.0, 356.8, 200.6)),
        _glyph("n", ital, 7.4, 6, (356.8, 195.1, 360.5, 203.9), (356.8, 198.7, 360.5, 202.0)),
        _glyph(
            ", we can instead estimate its", reg, 10.0, 4, (361.0, 191.2, 475.2, 203.2), (361.0, 193.6, 475.2, 201.8)
        ),
    ]
    below = [
        _glyph(
            "probability by using the bigram probability",
            reg,
            10.0,
            4,
            (142.2, 203.2, 313.8, 215.2),
            (142.2, 205.6, 313.8, 214.6),
        ),
        _glyph(" P", ital, 10.0, 6, (313.8, 203.2, 322.3, 215.1), (313.8, 205.9, 322.3, 212.4)),
        _glyph("(", cmr, 10.0, 4, (322.3, 204.9, 326.2, 214.9), (322.3, 204.9, 326.2, 214.9), math=True),
        _glyph("w", ital, 10.0, 6, (326.2, 203.2, 332.9, 215.1), (326.2, 208.0, 332.9, 212.6)),
        _glyph("n", ital, 7.4, 6, (332.9, 207.1, 336.5, 215.9), (332.9, 210.6, 336.5, 214.0)),
        _glyph("|", cmsy, 10.0, 6, (337.0, 204.7, 339.8, 222.0), (337.0, 204.9, 339.8, 214.9)),
        _glyph("w", ital, 10.0, 6, (339.8, 203.2, 346.4, 215.1), (339.8, 208.0, 346.4, 212.6)),
        _glyph("n", ital, 7.4, 6, (346.4, 207.1, 350.1, 215.9), (346.4, 210.7, 350.1, 214.0)),
        _glyph("−", cmsy, 7.4, 6, (350.1, 208.2, 355.9, 221.0), (350.1, 211.9, 355.9, 212.2)),
        _glyph("1", reg, 7.4, 4, (355.9, 207.1, 359.5, 216.0), (355.9, 208.9, 359.5, 213.9)),
        _glyph(")", cmr, 10.0, 4, (360.0, 204.9, 363.9, 214.9), (360.0, 204.9, 363.9, 214.9), math=True),
        _glyph(
            ". Similarly, if we don’t have", reg, 10.0, 4, (363.9, 203.2, 475.2, 215.2), (363.9, 205.6, 475.2, 214.6)
        ),
    ]
    sent, formulas = _placeholderize(_block_with_spans([above, below], size=10.0))
    assert len(formulas) == 2
    assert "P" not in formulas[0]["text"]
    assert formulas[1]["text"].startswith("P(")
    assert "bigram probability" in sent and sent.endswith("{v2}. Similarly, if we don’t have")


def test_a_colored_example_sentence_stays_original_but_an_epigraph_does_not():
    """A colored example sentence set off with an indent in the middle of body text keeps the original text, because the
    following text discusses words of the original sentence. A chapter epigraph is set the same way with its source
    after it and is translated as usual."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_unit_blocks import _attribution, _colored_example

    frame = TextFrame(142.2, 475.2, 82.0, 719.0, 10.0)

    def blk(text, rect, color=0x0000FF, size=10.0, flags=0):
        rect = pymupdf.Rect(rect)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=size,
            color="#0000ff",
            bold=False,
            body_size=10.0,
            span_lines=[[{"text": text, "size": size, "color": color, "flags": flags}]],
        )

    example = blk("Victoria Chen, CFO of Megabucks Banking, saw her pay jump to $2.3 million.", (167, 456, 475, 480))
    assert _colored_example(example, frame)
    assert not _colored_example(blk(example.text, (167, 456, 475, 480), color=0x404040), frame)
    assert not _colored_example(blk(example.text, (167, 456, 475, 480), flags=2), frame)
    assert not _colored_example(blk(example.text, (142.2, 456, 475, 480)), frame)
    quote = blk("On two occasions I have been asked, will the right answers come out?", (167, 210, 475, 258))
    assert _attribution(quote, blk("Babbage (1864)", (411, 246, 475, 258)))
    assert not _attribution(
        example, blk("Here Victoria is referred to by many different phrases", (142, 484, 475, 520))
    )


def test_a_split_off_period_keeps_its_own_baseline():
    """The space of " ." lies on the baseline of a subscript inside a radical; after the split the period must use its
    own origin so erasing can find it."""
    from app.formats.pdf_math import _split_covered

    span = {
        "text": " .",
        "font": "NimbusRomNo9L-Regu",
        "size": 10.0,
        "bbox": (455.2, 605.4, 459.5, 620.7),
        "origin": (455.2, 618.3),
    }
    chars = {(" .", 455.2, 618.3): [(455.2, 457.0, (455.2, 618.3)), (457.0, 459.5, (457.0, 613.0))]}
    bar = pymupdf.Rect(440.5, 610.3, 456.0, 610.7)
    pieces = _split_covered(span, [bar], chars, ([440.7, 455.2, 457.0], [455.2, 457.0, 459.5]), "math")
    assert [piece["text"] for piece in pieces] == [" ", "."]
    assert pieces[1]["origin"] == (457.0, 613.0)


def test_text_under_a_big_cmex_radical_is_part_of_the_formula():
    """The large CMEX radical is extracted as the letter p, with a font box taller than the glyph. The bar starts at
    the upper right corner of its ink, and the Var below the bar is the radicand."""
    from app.formats.pdf_math import _mark_radicands, _PageChars

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((188.4, 123.8), "Var", fontname="tiro", fontsize=9.96)
    data = page.get_text("dict")
    line = data["blocks"][0]["lines"][0]
    line["spans"].insert(
        0,
        {
            "text": "p",
            "font": "CMEX10",
            "size": 9.96,
            "flags": 0,
            "color": 0,
            "bbox": (177.2, 107.7, 188.4, 138.6),
            "origin": (177.2, 115.1),
            "ink": (177.2, 114.7, 188.4, 126.6),
        },
    )
    right = line["spans"][1]["bbox"][2]
    _mark_radicands(_PageChars(page), data, [pymupdf.Rect(188.4, 114.6, right, 115.1)])
    marked = [span for span in line["spans"] if span["text"] == "Var"]
    assert marked and marked[0].get("math")


def test_a_glyph_arrow_over_a_word_marks_both_as_formula():
    """esvect's \\vv{woman} is extracted as "#    »": no letters, ink that is a thin band, and a baseline less than an
    em above the word, right on top of woman. The arrow and woman are a formula, and the is close to after them is
    translated. Punctuation on the previous line does not count."""
    from app.formats.pdf_math import _mark_overlays, _PageChars

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((139.5, 597.2), "woman is close to", fontname="tiro", fontsize=9)
    page.insert_text((140.5, 593.4), "#    »", fontname="tiro", fontsize=9)
    page.insert_text((139.5, 586.4), "end of the line above, ...", fontname="tiro", fontsize=9)
    page.insert_text((172.0, 586.4), "____", fontname="tiro", fontsize=9)
    data = page.get_text("dict")
    for block in data["blocks"]:
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].startswith("#"):
                    span["ink"] = (140.5, 589.8, 165.8, 592.5)
                elif span["text"].startswith("woman"):
                    span["ink"] = (139.5, 591.1, 203.7, 597.3)
                elif span["text"].startswith("_"):
                    span["ink"] = (172.0, 586.0, 190.0, 586.6)
                else:
                    span["ink"] = (139.5, 580.0, 230.0, 586.6)
    data["blocks"] = [{"type": 0, "lines": [line for block in data["blocks"] for line in block["lines"]]}]
    _mark_overlays(_PageChars(page), data)
    spans = [span for line in data["blocks"][0]["lines"] for span in line["spans"]]
    marked = {span["text"].strip(): bool(span.get("math")) for span in spans}
    assert marked["woman"] and marked["#    »"]
    assert not marked["is close to"]
    assert not marked["end of the line above, ..."] and not marked["____"]


def test_a_relation_word_after_an_equals_sign_in_the_formula_is_dropped():
    """y_c = went into the formula image, and the 1 after it is body text. The model writes "为 1" once more, so the
    result reads "y_c = 为 1"."""
    from app.formats.pdf_restore import _restore_placeholders

    formulas = [{"text": "y\x01b\x02c\x01/b\x02 ="}, {"text": "n\x01b\x020\x01/b\x02"}]
    out = _restore_placeholders("其中 {v1} 为 1, 而 {v2} 为 2", formulas)
    assert out == "其中 \x01i\x021\x01/i\x02 1, 而 \x01i\x022\x01/i\x02 为 2"
    assert _restore_placeholders("而 {v1} 大于 1 时", [{"text": "β <"}]) == "而 \x01i\x021\x01/i\x02 1 时"


def test_a_colored_example_on_its_own_line_needs_no_period():
    """A colored example set on its own line often has no period (The green witch arrived)."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_unit_blocks import _colored_example

    frame = TextFrame(160.2, 493.2, 82.0, 719.0, 10.0)
    rect = pymupdf.Rect(185.1, 473.7, 280.2, 485.7)
    block = TextBlock(
        page=0,
        rect=rect,
        line_rects=[rect],
        text="The green witch arrived",
        size=10.0,
        color="#0000ff",
        bold=False,
        body_size=10.0,
        span_lines=[[{"text": "The green witch arrived", "size": 10.0, "color": 0x0000FF, "flags": 0}]],
    )
    assert _colored_example(block, frame)


def test_a_wrapped_formula_with_tall_glyph_boxes_is_split_by_baseline():
    """The upper and lower pieces of a wrapped formula ((o_i ≻ | o_j|x)) have slightly overlapping font boxes; by
    baseline they are two lines, so they become two pieces and are not merged into one box spanning the column.
    Values from ed3book p226."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, box, base, ink=None):
        out = {"text": text, "font": font, "size": size, "bbox": box, "origin": (box[0], base)}
        if ink is not None:
            out["ink"] = ink
        return out

    italic, roman, symbol, body = "NimbusRomNo9L-ReguItal", "CMR10", "CMSY10", "NimbusRomNo9L-Regu"
    b = _block_with_spans(
        [
            [
                span("move from the simple assertion ", body, 10, (142.2, 140.1, 450.0, 152.1), 149.3),
                span("(", roman, 10, (453.4, 141.8, 457.2, 151.8), 149.3, (453.4, 141.8, 457.2, 151.8)),
                span("o", italic, 10, (457.2, 140.1, 462.2, 152.0), 149.3),
                span("i", italic, 7.4, (462.2, 144.0, 464.3, 152.8), 150.8),
                span("≻", symbol, 10, (467.5, 141.6, 475.2, 158.9), 149.3, (467.5, 143.9, 475.2, 149.7)),
            ],
            [
                span("o", italic, 10, (142.2, 152.1, 147.2, 163.9), 161.3),
                span(" j", italic, 7.4, (147.2, 154.4, 150.3, 164.7), 161.3),
                span("|", symbol, 10, (151.0, 153.5, 153.8, 170.8), 161.3, (151.0, 153.8, 153.8, 163.8)),
                span("x", italic, 10, (153.8, 152.1, 158.2, 163.9), 161.3),
                span(")", roman, 10, (158.2, 153.8, 162.1, 163.8), 161.3, (158.2, 153.8, 162.1, 163.8)),
                span(" to knowing the value", body, 10, (162.1, 152.1, 300.0, 164.1), 161.3),
            ],
        ],
        size=10.0,
    )
    sent, formulas = _placeholderize(b, 0)
    assert sent.count("{v") == 1
    parts = formulas[0].get("parts") or []
    assert len(parts) == 2 and parts[0]["bbox"].x0 >= 450 and parts[1]["bbox"].x1 <= 170


def test_formula_pieces_keep_their_source_order():
    """A formula cut into pieces by a body-size variable ({v2} {b}W{/b}{v3}) whose order the model changed is put
    back in the original order. Pieces with Chinese between them are left alone; two or more letters form a word,
    which is not part of the formula sequence."""
    from app.formats.pdf_restore import _keep_formula_chains

    source = "compute the value of each{v1} as{v2} {b}W{/b}{v3}."
    assert (
        _keep_formula_chains("计算每个{v1}的值, 即 {b}W{/b}{v3}{v2}。", source)
        == "计算每个{v1}的值, 即 {v2} {b}W{/b}{v3}。"
    )
    assert _keep_formula_chains("即 {v2} {b}W{/b}{v3}。", source) == "即 {v2} {b}W{/b}{v3}。"
    assert _keep_formula_chains("{v3} 是 {v2} 的一部分", source) == "{v3} 是 {v2} 的一部分"


def _draw_visible_space(page, x, baseline):
    """Draw a ␣ from three short lines the way LaTeX does: a 3pt base line with a 1.4pt vertical line at each end."""
    page.draw_line((x, baseline - 1.4), (x, baseline), width=0.4)
    page.draw_line((x, baseline), (x + 3, baseline), width=0.4)
    page.draw_line((x + 3, baseline - 1.4), (x + 3, baseline), width=0.4)


def test_drawn_visible_space_is_read_into_the_text():
    """A drawn ␣ yields no text on extraction. One covering a space replaces the space, and one in a gap without text
    adds a ␣, separated from the preceding text by a space. The same shape off the text baseline (a tick mark in a
    figure) does not count."""
    from app.formats.pdf_layout import scan_page
    from app.formats.pdf_math import _mark_visible_spaces, _PageChars

    doc = pymupdf.open()
    page = doc.new_page()
    font = pymupdf.Font("cour")
    page.insert_font(fontname="C", fontbuffer=font.buffer)
    page.insert_text((100, 100), "merged to   re.", fontname="C", fontsize=10)
    _draw_visible_space(page, 100 + font.text_length("merged to  ", 10) + 2.8, 100)
    page.insert_text((100, 140), "Next", fontname="C", fontsize=10)
    _draw_visible_space(page, 100 + font.text_length("Next", 10) + 4, 140)
    page.insert_text((100 + font.text_length("Next", 10) + 12, 140), "r (total)", fontname="C", fontsize=10)
    _draw_visible_space(page, 300, 300)
    scan = scan_page(page)
    assert len(scan.visible_spaces) == 3
    data = page.get_text("dict")
    _mark_visible_spaces(_PageChars(page), data, scan.visible_spaces)
    lines = ["".join(span["text"] for span in line["spans"]) for block in data["blocks"] for line in block["lines"]]
    assert lines[0].replace("  ", " ") == "merged to ␣re."
    assert " ".join(lines[1:]) == "Next ␣ r (total)"


def test_index_entries_get_notes_with_their_parent_entry():
    """Notes for index entries: a subentry is attached by indent to the nearest preceding entry with a shallower
    indent, and an entry with subentries lists the first few. Across columns the indent is measured from the left
    edge of each column. Units not on an index page have no note."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_units import _index_notes

    def block(text, x0, y0, index=True):
        rect = pymupdf.Rect(x0, y0, x0 + 80, y0 + 10)
        return [TextBlock(5, rect, [rect], text, 9.0, "#000", False, index=index)]

    units = [
        block("Index", 72, 50, index=False),
        block("MT, {b}276{/b}", 72, 100),
        block("divergences, {b}278{/b}", 82, 110),
        block("post-editing, 293", 82, 120),
        block("multinomial logistic regression, 108", 72, 130),
        block("why used for probabilities,", 322, 100),
        block("log", 312, 90),
    ]
    units = [units[0], units[1], units[2], units[3], units[4], units[6], units[5]]
    texts = [unit[0].text for unit in units]
    notes = _index_notes(units, texts, [False] * len(units))
    assert notes[0] == ""
    assert notes[1] == (
        "An entry in the subject index at the back of the book; its sub-entries include “divergences”, “post-editing”."
    )
    assert notes[2] == "An entry in the subject index at the back of the book, listed under the entry “MT”."
    assert notes[4] == "An entry in the subject index at the back of the book."
    assert "listed under the entry “log”" in notes[6]


def test_wrapped_monospace_phrase_keeps_its_word_space():
    """The monospace example my dog is | hairy breaks between words, so the two pieces side by side keep a word space
    as wide as the original space; for a URL broken inside a word or a formula broken after an operator, the two
    pieces still touch."""
    from app.formats.pdf_blocks import _PART_GAP
    from app.formats.pdf_unit_blocks import _wrap_gap

    def span(text, x0, x1):
        return {"text": text, "font": "NimbusMonL-Regu", "size": 9.0, "bbox": (x0, 0, x1, 9)}

    line = [
        span("my", 477.1, 487.9),
        span(" ", 487.9, 493.3),
        span("dog", 493.3, 509.4),
        span(" ", 509.4, 514.8),
        span("is", 514.8, 525.5),
    ]
    assert abs(_wrap_gap(line, [span("hairy", 307.3, 334.2)]) - 5.4) < 0.05
    url = [span("https://artofproblemsolving.com/wiki/index.php/202", 300, 525)]
    assert _wrap_gap(url, [span("5_AIME_II_Problems", 307, 400)]) == _PART_GAP
    assert (
        _wrap_gap([span("a", 500, 505), span(" ", 505, 508), span("+", 508, 515)], [span("b", 307, 312)]) == _PART_GAP
    )


def test_a_font_that_only_writes_single_marks_beside_formulas_is_math():
    """The bitmap double-struck 1 (𝟙) is a Type3 font whose name is only a number, and it is extracted as 1. A font
    that writes only one or two characters in the whole text, mostly next to formulas, is a formula font. A
    footnote mark font appearing alone in body text is not next to formulas and does not count."""
    from app.formats.pdf_blocks import _span_is_math
    from app.formats.pdf_math import _mark_formula_only_fonts

    def span(font, text, x0, x1):
        return {"font": font, "text": text, "size": 10.0, "bbox": (x0, 50, x1, 60)}

    data = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            span("CMR10", "the set is", 0, 40),
                            span("CMMI10", "A", 42, 49),
                            span("F74", "1", 49, 55),
                            span("CMSY10", "{", 55, 60),
                        ]
                    },
                    {
                        "spans": [
                            span("CMR10", "see note", 0, 40),
                            span("F99", "2", 41, 45),
                            span("CMR10", " below", 46, 80),
                        ]
                    },
                ],
            }
        ]
    }
    _mark_formula_only_fonts([data])
    first, second = (line["spans"] for line in data["blocks"][0]["lines"])
    assert _span_is_math(first[2])
    assert not _span_is_math(second[1]) and not _span_is_math(first[0])


def test_a_script_left_of_a_letter_is_not_its_script():
    """The numerator of an inline fraction (one size smaller) overlaps vertically with an italic x later on the same
    line, with . Each token in between them; x is not its base."""
    from app.formats.pdf_formula_runs import _script_base

    x = {
        "text": " x",
        "bbox": (223.9, 596.6, 230.4, 608.5),
        "origin": (223.9, 606.0),
        "size": 10.0,
        "font": "NimbusRomNo9L-ReguItal",
    }
    assert not _script_base(x, pymupdf.Rect(146.5, 593.3, 159.8, 607.2), 10.0, math=True, script_y=600.0)
    assert _script_base(
        {**x, "bbox": (140.0, 596.6, 146.5, 608.5)}, pymupdf.Rect(146.5, 600, 149, 609), 10.0, math=True, script_y=607.0
    )


def test_a_wrapped_fraction_piece_sits_on_its_line_baseline():
    """A fraction wrapped to the start of the next line (… and λ2 = | 1/2, what is …) measures its descent from the
    baseline of the body text on that line, not from the baseline of the denominator: numerator and denominator are
    the same size, so measuring to the denominator would raise the whole fraction on placement and push the
    numerator onto the previous line."""
    from app.formats.pdf_units import _placeholderize

    def span(text, font, size, x0, x1, base, top=None):
        return {
            "text": text,
            "font": font,
            "size": size,
            "bbox": (x0, top if top is not None else base - size * 0.8, x1, base + size * 0.2),
            "origin": (x0, base),
        }

    block = _block_with_spans(
        [
            [
                span("unigram model with ", "NimbusRomNo9L-Regu", 10, 40, 400, 676.4),
                span("λ", "CMMI10", 10, 453.5, 459, 676.4),
                span("2", "CMR7", 7, 459, 463, 678.0),
                span("=", "CMR10", 10, 466, 475, 676.4),
            ],
            [
                span("1", "CMR7", 7, 168.3, 172, 683.0, top=677.5),
                span("2", "CMR7", 7, 168.3, 172, 691.7),
                span(", what is the probability", "NimbusRomNo9L-Regu", 10, 174, 300, 688.4),
            ],
        ],
        size=10.0,
    )
    _sent, formulas = _placeholderize(block)
    parts = formulas[0]["parts"]
    assert len(parts) == 2
    assert parts[1]["d"] == pytest.approx(693.1 - 688.4, abs=0.2)
