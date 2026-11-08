"""PDF translation end to end and the stages not covered by the other PDF test files."""

import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate

import pymupdf
import pytest

from app.formats import translate_file
from app.formats.markdown import collect_segments, render, split_blocks
from app.formats.pdf_blocks import _join_lines
from app.formats.pdf_types import Block, Line, PageDict, Span
from app.runner import Runner
from app.translators import MockTranslator
from tests.helpers import SAMPLES, MemCache, _pdf_with_text, _write_epub, run, runner

# A preparation stage as reported to the runner: name, done, total, unit
Stage = tuple[str, int, int, str]
# A box as its corner coordinates: x0, y0, x1, y1
Corners = tuple[float, float, float, float]


def _bbox(rect: pymupdf.Rect) -> Corners:
    return rect.x0, rect.y0, rect.x1, rect.y1


def test_pdf(tmp_path: Path) -> None:
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    for bilingual in (False, True):
        [out] = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
        doc = pymupdf.open(out)
        assert len(doc) == len(pymupdf.open(src))
        text = doc[0].get_text()
        assert "[zh-CN]" in text
        if bilingual:
            assert doc[0].rect.width == pytest.approx(pymupdf.open(src)[0].rect.width * 2)


def test_epub2_outline_reads_the_ncx(tmp_path: Path) -> None:
    """An EPUB2 without a nav document takes its table of contents from toc.ncx, nested by navPoint."""
    from app.formats.epub import book_outline

    ncx = (
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>'
        '<navPoint id="a"><navLabel><text>Chapter One</text></navLabel><content src="ch1.xhtml"/>'
        '<navPoint id="b"><navLabel><text>A Section</text></navLabel><content src="ch1.xhtml#s1"/></navPoint>'
        "</navPoint></navMap></ncx>"
    )
    src = tmp_path / "old.epub"
    _write_epub(
        src,
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        '<itemref idref="ch1"/>',
        {"ch1.xhtml": "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>x</p></body></html>", "toc.ncx": ncx},
    )
    with zipfile.ZipFile(src) as z:
        outline = book_outline(z)
    assert outline["title"] == "Test Book" and outline["spine"] == ["OEBPS/ch1.xhtml"]
    assert [(e["label"], e["anchor"], e["depth"]) for e in outline["toc"]] == [
        ("Chapter One", "", 0),
        ("A Section", "s1", 1),
    ]


def test_markdown_setext_headings() -> None:
    """The underline of a Setext heading is not sent for translation; the original underline stays below the
    translation."""
    blocks = split_blocks("Main Title\n==========\n\nSection Name\n---\n\nBody text.")
    texts = collect_segments(blocks)
    assert texts == ["Main Title", "Section Name", "Body text."]
    from app.formats.markdown import segment_kinds

    assert segment_kinds(blocks) == ["h1", "h2", "p"]
    out = render(blocks, ["T:" + x for x in texts], bilingual=False)
    assert out == "T:Main Title\n==========\n\nT:Section Name\n---\n\nT:Body text."
    both = render(blocks, ["T:" + x for x in texts], bilingual=True)
    assert both.startswith("Main Title\n==========\n\nT:Main Title\n==========\n")


def test_pdf_hyphen_joins_uppercase_word() -> None:
    """Words hyphenated at a line end are joined back, all-caps words too. A hyphen after a digit, or between a
    lowercase and an uppercase letter, belongs to the name and stays."""
    assert _join_lines(["word-", "wrap"]) == "wordwrap"
    assert _join_lines(["INTER-", "NATIONAL"]) == "INTERNATIONAL"
    assert _join_lines(["Claude-3.7-", "Sonnet"]) == "Claude-3.7-Sonnet"
    assert _join_lines(["DeepSeek-R1-Distill-", "Qwen-7B"]) == "DeepSeek-R1-Distill-Qwen-7B"
    assert _join_lines(["Donau-", "Dampfschiff"]) == "Donau-Dampfschiff"
    # McCallum broken by TeX before the capital letter: elsewhere the text writes McCallum, so the hyphen goes
    from app.formats.pdf_blocks import _CAMEL_WORDS

    token = _CAMEL_WORDS.set(frozenset({"McCallum"}))
    try:
        assert _join_lines(["({c0000ff}Mc-", "Callum and"]) == "({c0000ff}McCallum and"
        assert _join_lines(["Witten-", "Bell discounting"]) == "Witten-Bell discounting"
    finally:
        _CAMEL_WORDS.reset(token)
    assert _join_lines(["hello", "world"]) == "hello world"  # Without hyphenation it is just a line break
    assert _join_lines(["the {v1}", ". This"]) == "the {v1}. This"
    assert _join_lines(["end", ", next"]) == "end, next"


def test_pdf_list_items_split_into_segments() -> None:
    """List items in one block each become a separate translation unit (common in slides); wrapped continuation
    lines still join the previous item."""
    from app.formats.pdf_extract import extract_blocks

    doc = _pdf_with_text("- First item here\n- Second item continues\nonto next line\n- Third item")
    texts = [b.text for b in extract_blocks(doc)]
    assert texts == ["- First item here", "- Second item continues onto next line", "- Third item"]


def test_pdf_hard_break_splits_lines() -> None:
    """A hard line break without a bullet (upper line much shorter than the widest line of the block, lower line
    starting with a capital, left-aligned) also splits into two paragraphs."""
    from app.formats.pdf_extract import extract_blocks

    doc = _pdf_with_text("Short Name\nA Much Longer Affiliation Line Here")
    texts = [b.text for b in extract_blocks(doc)]
    assert texts == ["Short Name", "A Much Longer Affiliation Line Here"]


def test_a_small_caps_heading_takes_the_size_of_its_capitals() -> None:
    """In a small-caps heading the initial is one size larger and the rest one size smaller, so the longest piece is
    the small one. The block size follows the capitals, otherwise the translation would be smaller than the body
    text. A lowered subscript (BERT_LARGE) is not small caps."""
    from app.formats.pdf_extract import _block_size

    def span(text: str, x0: float, size: float, base: float) -> Span:
        return {"text": text, "size": size, "origin": (x0, base), "bbox": (x0, base - size, x0 + 50, base + 2)}

    heading = [span("I", 123.0, 11.6, 362.0), span("NITIALIZATION", 127.0, 9.3, 362.0)]
    assert _block_size(heading[1], [heading]) == 11.6
    model = [span("BERT", 324.5, 9.0, 135.0), span("LARGE", 347.0, 6.5, 137.5)]
    assert _block_size(model[1], [model]) == 6.5


def test_pdf_wrapped_body_on_attention_sample() -> None:
    """A first-line indent in the body of a paper must not cut a sentence in the middle."""
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    from app.formats.pdf_extract import extract_blocks

    texts = [b.text for b in extract_blocks(pymupdf.open(src))]
    for sentence in (
        "started the effort",
        "Each layer has two sub-layers",
        "independent random variables",
        "added to the sub-layer input",
    ):
        assert any(sentence in t for t in texts), sentence


def test_pdf_list_continuation_must_be_indented() -> None:
    """A new line at the same x0 after a list item is not a continuation line (it may be the next paragraph); only an
    indented line with hanging alignment is."""
    from app.formats.pdf_extract import extract_blocks

    doc = _pdf_with_text("- Short item\nNew paragraph at same indent")
    assert [b.text for b in extract_blocks(doc)] == ["- Short item", "New paragraph at same indent"]


def test_pdf_cjk_continuation_needs_sentence_end() -> None:
    """CJK line starts: when the previous line does not end with sentence-final punctuation the line is a
    continuation; the paragraph splits only after sentence-final punctuation."""
    from app.formats.pdf_extract import extract_blocks

    # The first line is clearly shorter than the second (a sign of a hard break), but it does not end a sentence,
    # so there is no split
    doc = _pdf_with_text("这是一个短行\n而这是一行明显更长的内容用来撑满整行的宽度", fontname="china-s")
    assert len(extract_blocks(doc)) == 1
    doc = _pdf_with_text("这是完整的一句。\n而这是一行明显更长的内容用来撑满整行的宽度", fontname="china-s")
    assert len(extract_blocks(doc)) == 2


def test_ieee_section_headings_are_headings() -> None:
    """IEEE section headings have the body size: all caps (small caps) after a Roman numeral, or a whole italic line
    with a letter number, is a heading. Name initials in body text and author lines with commas are not."""
    from app.formats.pdf_roles import _is_lettered_title

    assert _is_lettered_title("II. METHODS", styled=False)
    assert _is_lettered_title("IV. EXPERIMENTS AND RESULTS", styled=False)
    assert _is_lettered_title("A. Materials", styled=True)
    assert not _is_lettered_title("A. Materials", styled=False)
    assert not _is_lettered_title("B. Smith", styled=False)
    assert not _is_lettered_title("V. Vuletić, and M. D. Lukin,", styled=True)


def test_translated_version_cut_from_the_bilingual_one_matches_a_fresh_render(tmp_path: Path) -> None:
    """The translated-only version is taken from the bilingual version: text, links, page labels and bookmarks match
    a fresh layout. With rotated pages, or with bookmarks but no preview snapshot, it cannot be taken and a fresh
    layout is used."""
    from app.formats.pdf import translate_pdf
    from app.formats.pdf_output import translated_from_bilingual

    doc = pymupdf.open()
    for n in range(2):
        doc.new_page(width=420, height=240).insert_text(
            (40, 60),
            f"A body paragraph long enough to be translated on page {n + 1}.",
            fontsize=10,
        )
    doc[1].insert_text((40, 120), "2 Second Part", fontsize=14)
    doc[0].insert_link(
        {"kind": pymupdf.LINK_GOTO, "from": pymupdf.Rect(40, 50, 200, 62), "page": 1, "to": pymupdf.Point(0, 0)}
    )
    doc.set_page_labels([{"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 1}])
    doc.set_toc([[1, "Second Part", 2]])
    src = tmp_path / "paper.pdf"
    doc.save(src)
    (tmp_path / "b").mkdir()
    (tmp_path / "t").mkdir()
    first = runner()
    [both] = run(translate_pdf(src, tmp_path / "b", first, True, "zh-CN"))
    [alone] = run(translate_pdf(src, tmp_path / "t", runner(), False, "zh-CN"))
    assert translated_from_bilingual(src, both, None) is None
    data = translated_from_bilingual(src, both, first.preview(-1))
    assert data is not None
    cut, fresh = pymupdf.open("pdf", data), pymupdf.open(alone)
    assert len(cut) == len(fresh) == 2 and "[zh-CN]" in cut[0].get_text()
    for a, b in zip(cut.pages(), fresh.pages()):
        assert a.rect == b.rect and a.get_text() == b.get_text()
        assert [(link["from"], link.get("page")) for link in a.get_links()] == [
            (link["from"], link.get("page")) for link in b.get_links()
        ]
        assert a.get_pixmap(dpi=50).samples == b.get_pixmap(dpi=50).samples
    assert cut[0].get_links() and cut.get_page_labels() == fresh.get_page_labels()
    assert cut.get_toc() == fresh.get_toc() == [[1, "[zh-CN] Second Part", 2]]
    assert pymupdf.open(both).get_toc() == [[1, "Second Part", 2]]  # Bilingual version keeps its bookmarks

    doc[1].set_rotation(90)
    doc.save(tmp_path / "turned.pdf")
    assert translated_from_bilingual(tmp_path / "turned.pdf", both, first.preview(-1)) is None


def test_bookmarks_take_the_translation_shown_on_the_page_they_point_to() -> None:
    """When the same heading has different translations (translated once on the contents page and once in the body),
    the one on the target page or the page after it is used, and the section number is dropped when the original
    bookmark has none. Bookmarks whose translation equals the original, or whose paragraph was skipped, keep the
    original text."""
    from app.formats.pdf_output import translate_bookmarks

    doc = pymupdf.open()
    for text in ["Ways", "", "2 Approaches"]:
        doc.new_page().insert_text((50, 50), text)
    doc.set_toc([[1, "Methods", 2], [1, "Results", 3], [1, "Notes", 3]])
    preview = {
        "segments": [
            {"s": "Methods", "k": "p"},
            {"s": "Methods", "k": "h2"},
            {"s": "Results", "k": "h2"},
            {"s": "Notes", "k": "h2"},
        ],
        "updates": [[0, "Ways", 0], [1, "Approaches", 0], [1, "2 Approaches", 0], [2, "Results", 0], [3, "Notes", 1]],
    }
    translate_bookmarks(doc, preview)
    assert doc.get_toc() == [[1, "Approaches", 2], [1, "Results", 3], [1, "Notes", 3]]


def test_skipped_bibliography_keeps_hanging_indent(tmp_path: Path) -> None:
    """Skipped citations stay on the original page. An italic journal name must not trigger erasing and re-layout,
    otherwise the hanging indent and line-end hyphenation are lost."""
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    page = doc.new_page(width=420, height=240)
    roman = pymupdf.Font("tiro")
    italic = pymupdf.Font("tiit")
    bold = pymupdf.Font("tibo")
    writer = pymupdf.TextWriter(page.rect)
    writer.append((40, 36), "A body paragraph long enough to set the median size.", font=roman, fontsize=10)
    writer.append((40, 70), "References", font=bold, fontsize=14)
    _, cursor = writer.append((40, 100), "[1] Smith, J. (2020). A paper. In ", font=roman, fontsize=9)
    writer.append(cursor, "Proceedings of the Inter-", font=italic, fontsize=9)
    _, cursor = writer.append((54, 112), "national Conference, ", font=italic, fontsize=9)
    writer.append(cursor, "ICML, 2009.", font=roman, fontsize=9)
    writer.write_text(page)
    src = tmp_path / "refs.pdf"
    doc.save(src)

    before = pymupdf.open(src)[0]
    national = before.search_for("national")
    proceedings = before.search_for("Proceedings")
    assert national and proceedings

    [out] = run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))
    page = pymupdf.open(out)[0]
    assert "[zh-CN]" in page.get_text()
    assert "Inter- national" not in page.get_text()
    kept = page.search_for("national")
    kept_proc = page.search_for("Proceedings")
    assert kept and kept_proc
    assert abs(kept[0].x0 - national[0].x0) < 0.8
    assert abs(kept_proc[0].x0 - proceedings[0].x0) < 0.8
    fonts = {
        span["font"]
        for block in page.get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if "Proceedings" in span["text"] or "national" in span["text"]
    }
    assert fonts
    assert all("Times" in name or "NimbusRom" in name for name in fonts)


def test_pdf_scanned_error_guides_ocr(tmp_path: Path) -> None:
    from app.formats.pdf import translate_pdf

    src = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_image(
        pymupdf.Rect(10, 10, 60, 60), pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 5, 5))
    )
    doc.save(src)
    with pytest.raises(ValueError, match="OCR"):
        run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))


def test_pdf_bookmarks_migrated(tmp_path: Path) -> None:
    doc = _pdf_with_text("Chapter one content here for testing.")
    doc.set_toc([(1, "Chapter One", 1)])
    src = tmp_path / "book.pdf"
    doc.save(src)
    for bilingual in (False, True):
        [out] = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
        assert pymupdf.open(out).get_toc() == [[1, "Chapter One", 1]]


def test_subset_fonts_safe_returns_valid_doc() -> None:
    from app.formats.pdf_output import _subset_fonts_safe

    doc = _pdf_with_text("Some text for subsetting.")
    out = pymupdf.open("pdf", _subset_fonts_safe(doc.tobytes()))
    assert out.page_count == doc.page_count and "Some text" in out[0].get_text()


def _symbol_pdf(path: Path, symbol: str) -> None:
    """A run of Symbol-font glyphs in the middle of body text becomes a {v1} formula placeholder on extraction."""
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=300)
    head = "For the case where the parameter k "
    page.insert_text((40, 60), head, fontsize=11)
    x = 40 + pymupdf.get_text_length(head, fontsize=11)
    page.insert_text((x, 60), symbol, fontname="symb", fontsize=11)
    page.insert_text((x + 9 * len(symbol), 60), " is small, the greedy algorithm works well.", fontsize=11)
    doc.save(path)
    doc.close()


def test_a_display_formula_set_in_text_italic_is_left_as_it_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone formula line with variables in Times-Italic and operators in Symbol: after placeholders there is
    no text to translate, so the line is kept as it is, not sent for translation and not erased and laid out again
    (a two-line formula would be shrunk as one tall formula on re-layout)."""
    src = tmp_path / "italic_math.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((40, 60), "The implication test can be split into three separate tests:", fontsize=10)
    x, y = 60.0, 90.0
    for text, font, size, dy in [
        ("(PE", "tiit", 10, 0),
        ("q", "tiit", 6.5, 1.6),
        (" \u2227 ", "symb", 10, 0),
        ("PR", "tiit", 10, 0),
        ("q", "tiit", 6.5, 1.6),
        (" \u21d2 ", "symb", 10, 0),
        ("PE", "tiit", 10, 0),
        ("v", "tiit", 6.5, 1.6),
        (")", "tiit", 10, 0),
    ]:
        page.insert_text((x, y + dy), text, fontname=font, fontsize=size)
        x += pymupdf.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text((40, 120), "The first test is called the equijoin subsumption test.", fontsize=10)
    doc.save(src)
    doc.close()

    tr = MockTranslator("zh-CN")
    sent: list[str] = []
    original = tr.translate_batch

    async def record(texts: list[str]) -> list[str]:
        sent.extend(texts)
        return await original(texts)

    monkeypatch.setattr(tr, "translate_batch", record)
    [out] = run(translate_file(src, tmp_path, Runner(tr, cache=MemCache(), skip_same_lang=False), False, "zh-CN"))
    assert not any("PE" in text or "{v" in text for text in sent), sent
    words = [w[4] for w in pymupdf.open(out)[0].get_text("words") if 80 <= w[1] <= 95]
    # The original glyphs are still in place (scripts join the preceding word when words are extracted)
    assert words[0].startswith("(PE") and not any("[zh-CN]" in w for w in words)


def test_display_equations_and_text_fractions_are_left_as_they_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fraction with words as numerator and denominator, and a formula with only a relation symbol and one or two
    names, are kept as they are and not sent for translation. Body text continuing below a link underline and a
    caption with an equals sign are translated."""
    src = tmp_path / "equations.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((40, 40), "We measure the parser with labeled recall and labeled precision.", fontsize=10)
    page.insert_text(
        (127 - pymupdf.get_text_length("labeled recall: =", fontsize=10), 78), "labeled recall: =", fontsize=10
    )
    numerator, denominator = "# of correct constituents in the parse", "# of total constituents in the parse"
    width = pymupdf.get_text_length(numerator, fontsize=10)
    page.insert_text((130, 73), numerator, fontsize=10)
    page.insert_text(
        (130 + (width - pymupdf.get_text_length(denominator, fontsize=10)) / 2, 86), denominator, fontsize=10
    )
    page.draw_line((130, 76), (130 + width, 76), width=0.4)
    page.insert_text((150, 120), "T = argmax s(T)", fontsize=10)
    page.insert_text((40, 160), "The scores are listed in the appendix of the paper, see the", fontsize=10)
    page.draw_line((260, 162), (300, 162), width=0.4)
    page.insert_text((40, 172), "online tables for the full results of every model we trained.", fontsize=10)
    page.insert_text((40, 210), "Figure 2: Accuracy with alpha = 0.5.", fontsize=10)
    doc.save(src)
    doc.close()

    tr = MockTranslator("zh-CN")
    sent: list[str] = []
    original = tr.translate_batch

    async def record(texts: list[str]) -> list[str]:
        sent.extend(texts)
        return await original(texts)

    monkeypatch.setattr(tr, "translate_batch", record)
    run(translate_file(src, tmp_path, Runner(tr, cache=MemCache(), skip_same_lang=False), False, "zh-CN"))
    assert not any("constituents" in text or "argmax" in text for text in sent), sent
    assert any("see the online tables" in text for text in sent), sent
    assert any(text.startswith("Figure 2") for text in sent), sent


class _PlaceholderDropper(MockTranslator):
    """A batch request writes {v1} out as text and loses it; with keep_on_retry set, retranslating the single segment
    keeps the placeholder."""

    def __init__(self, target_lang: str, *, keep_on_retry: bool = False) -> None:
        super().__init__(target_lang)
        self.keep_on_retry = keep_on_retry
        self.calls: list[list[str]] = []

    async def translate_batch(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        retry = len(self.calls) > 1
        out: list[str] = []
        for _text in texts:
            if retry and self.keep_on_retry:
                out.append("当参数 k{v1} 较小时，贪心算法效果很好。")
            else:
                out.append("当参数 k = 2 较小时，贪心算法效果很好。")
        return out


def test_lost_placeholder_is_retried_once(tmp_path: Path) -> None:
    """The model lost {v1}: the segment is translated once more on its own, the translation with the placeholder
    restored goes into the cache, and the formula is put back as usual."""
    src = tmp_path / "sym.pdf"
    _symbol_pdf(src, "abcd")
    tr = _PlaceholderDropper("zh-CN", keep_on_retry=True)
    cache = MemCache()
    r = Runner(tr, cache=cache, skip_same_lang=False)
    [out] = run(translate_file(src, tmp_path, r, False, "zh-CN"))
    assert len(tr.calls) == 2 and tr.calls[1] == tr.calls[0]
    assert "当参数" in pymupdf.open(out)[0].get_text()
    assert any("{v1}" in v for v in cache.d.values())
    assert r.info.failed == 0


def test_lost_small_placeholder_keeps_the_translation(tmp_path: Path) -> None:
    """Only a one-glyph formula was lost (the model wrote it out as text): the translation is accepted without falling
    back to the original."""
    src = tmp_path / "sym.pdf"
    _symbol_pdf(src, "a")
    r = Runner(_PlaceholderDropper("zh-CN"), cache=MemCache(), skip_same_lang=False)
    [out] = run(translate_file(src, tmp_path, r, False, "zh-CN"))
    text = pymupdf.open(out)[0].get_text()
    assert "贪心算法" in text and "greedy" not in text
    assert r.info.failed == 0


def test_lost_large_placeholder_falls_back_and_counts_as_failed(tmp_path: Path) -> None:
    """A longer formula was lost: the whole segment falls back to the original and counts as a failed segment, so the
    job page shows a warning."""
    src = tmp_path / "sym.pdf"
    _symbol_pdf(src, "abcd")
    r = Runner(_PlaceholderDropper("zh-CN"), cache=MemCache(), skip_same_lang=False)
    [out] = run(translate_file(src, tmp_path, r, False, "zh-CN"))
    assert "greedy" in pymupdf.open(out)[0].get_text()
    assert r.info.failed == 1


def test_equation_vs_prose_with_formula() -> None:
    """Formula lines (many math characters, few body words) are skipped; ordinary sentences containing formulas are
    translated."""
    from app.formats.pdf_extract import extract_blocks

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)
    # A sentence with a formula: many body words, so it must be translated
    page.insert_textbox(
        pymupdf.Rect(30, 20, 480, 45),
        "The sequence y_1, y_2, ..., y_T has a variable number of tokens here",
        fontsize=11,
    )
    # A formula line: almost no body words, so it is skipped
    page.insert_textbox(pymupdf.Rect(30, 60, 480, 85), "x + y = z", fontsize=11)
    blocks = extract_blocks(doc)
    texts = [b.text for b in blocks]
    assert any("sequence" in t for t in texts)
    assert not any("x + y = z" in t for t in texts)


def test_formula_bitmaps_rendered_in_forked_workers_read_their_own_copy(tmp_path: Path) -> None:
    """The writing subprocesses are forked and share the file position of the original file with the parent. Several
    subprocesses rendering formula bitmaps from the document the parent opened at the same time read bytes from
    other places and render wrongly or fail. Each subprocess must use a document it opened itself."""
    import multiprocessing
    from multiprocessing.queues import Queue

    from app.formats.pdf_blocks import _FormulaImage

    src = tmp_path / "book.pdf"
    doc = pymupdf.open()
    for pno in range(12):
        page = doc.new_page(width=300, height=400)
        for row in range(20):
            page.insert_text((20, 30 + row * 18), f"page {pno} row {row} x + y = z", fontsize=11)
    doc.save(src)
    clips = [(pno, pymupdf.Rect(15, 15 + row * 18, 200, 35 + row * 18)) for pno in range(12) for row in range(0, 20, 4)]
    fresh = pymupdf.open(src)
    expected = [fresh[pno].get_pixmap(dpi=300, clip=clip, alpha=True).samples for pno, clip in clips]
    shared = pymupdf.open(src)
    images = [_FormulaImage(shared, pno, clip) for pno, clip in clips]
    pymupdf.TOOLS.store_shrink(100)

    def work(part: int, out: Queue[bool]) -> None:
        try:
            got = [pymupdf.Pixmap(images[k].png()).samples == expected[k] for k in range(part, len(images), 4)]
            out.put(all(got))
        except Exception:  # noqa: BLE001 - a corrupted read raises, which counts as a wrong render
            out.put(False)

    context = multiprocessing.get_context("fork")
    out = context.Queue()
    workers = [context.Process(target=work, args=(part, out)) for part in range(4)]
    for worker in workers:
        worker.start()
    results = [out.get(timeout=60) for _ in workers]
    for worker in workers:
        worker.join()
    assert results == [True] * 4


def test_a_small_caps_word_at_the_line_start_is_not_a_style_change() -> None:
    """An all-caps word one size smaller at the line start is a small-caps abbreviation (MUC (Vilain et al., 1995)
    and …) and counts at the body size; body text starting in bold or one size smaller still counts as a style
    change."""
    from app.formats.pdf_blocks import TextBlock, _styled_edge

    def span(text: str, x0: float, size: float = 10.0, font: str = "NimbusRomNo9L-Regu", flags: int = 0) -> Span:
        return {
            "text": text,
            "font": font,
            "size": size,
            "flags": flags,
            "bbox": (x0, 433, x0 + 20, 444),
            "origin": (x0, 441.5),
        }

    def blk(spans: list[Span]) -> TextBlock:
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(142, 433, 475, 480),
            line_rects=[],
            text="",
            size=10,
            color="#000",
            bold=False,
            span_lines=[spans],
        )

    assert not _styled_edge(blk([span("MUC", 142.4, 8.0), span(" (Vilain et al., 1995) and", 162)]), 0)
    assert _styled_edge(
        blk([span("Lemma 2.3.", 142.4, font="NimbusRomNo9L-Medi", flags=16), span(" Let x be", 190)]), 0
    )
    assert _styled_edge(blk([span("note", 142.4, 8.0), span(" and the rest", 162)]), 0)


def test_column_tail_drawn_before_its_host_keeps_the_paragraph(tmp_path: Path) -> None:
    """When the content stream draws the right column first, the tail block at the top of the right column comes before
    its host in the left column. The whole translation must still be written."""
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)

    def para(x: float, y: float, lines: list[str]) -> None:
        for i, line in enumerate(lines):
            page.insert_text((x, y + i * 12), line, fontsize=10)

    para(320, 80, ["continues the gamma sentence on the right side", "with more words that end with a stop."])
    para(320, 200, ["Delta paragraph in the right column is here", "and finishes with a final stop."])
    para(
        50,
        80,
        [
            "Alpha paragraph text that fills the left column",
            "and keeps going for a few more words here.",
            "It ends with a full stop at the end.",
        ],
    )
    para(
        50,
        200,
        [
            "Beta paragraph text that also fills the column",
            "and keeps going for several more words.",
            "It also ends properly with a stop.",
        ],
    )
    para(50, 320, ["Gamma begins and we write a short line"])
    src = tmp_path / "right_first.pdf"
    doc.save(src)

    [out] = run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))
    text = " ".join(pymupdf.open(out)[0].get_text().split())
    assert "Gamma begins" in text
    assert "continues the gamma sentence" in text


def test_rotated_pages_lay_out_like_unrotated_ones(tmp_path: Path) -> None:
    """On a page rotated by 90 degrees, a paragraph carried over to the next page is laid out as on an unrotated page:
    writing does not stop early and the font size does not shrink."""
    from app.formats.pdf import translate_pdf

    words = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua "
    ).split()
    layouts = []
    for rotation in (0, 90):
        doc = pymupdf.open()
        index = 0
        for pno in range(2):
            page = doc.new_page(width=595, height=842)
            writer = pymupdf.TextWriter(page.rect)
            for y in range(60, 790, 12):
                line: list[str] = []
                while len(" ".join(line)) < 80:
                    line.append(words[index % len(words)])
                    index += 1
                text = " ".join(line)
                writer.append((50, y), ("The " + text) if pno == 0 and y == 60 else text, fontsize=10)
            writer.write_text(page)
            page.set_rotation(rotation)
        src = tmp_path / f"rot{rotation}.pdf"
        doc.save(src)
        [out] = run(translate_pdf(src, tmp_path, runner(), False, "zh-CN"))
        layouts.append(
            [
                sorted(
                    (round(span["bbox"][1]), round(span["size"], 1))
                    for blk in page.get_text("dict")["blocks"]
                    if blk.get("type") == 0
                    for ln in blk["lines"]
                    for span in ln["spans"]
                )
                for page in pymupdf.open(out).pages()
            ]
        )
    plain, rotated = layouts
    assert rotated == plain


def test_justify_extras_skip_punctuation_and_short_lines() -> None:
    """Justification spreads the slack between characters and does not open space before punctuation. With too little
    slack or too few gaps the line is not stretched."""
    from app.formats.pdf_align import justify_extras

    text = "我们在此提出一种，采样机制已经写完"
    extras = justify_extras(text, 24.0, 10.0)
    assert len(extras) == len(text) - 1
    comma = text.index("，")
    assert extras[comma - 1] == 0.0
    assert extras[0] > 0
    assert justify_extras(text, 1.0, 10.0) == []
    assert justify_extras("短行", 20.0, 10.0) == []
    names = "Zhongxian Gu, Entong Shen, George Caragea 等人继续写"
    extras = justify_extras(names, 40.0, 10.0)
    assert extras[0] == 0.0
    assert extras[names.index(" ")] > 0


def test_a_loose_mixed_line_stretches_word_spaces_before_the_chinese() -> None:
    """On a line mixing Chinese and English with a lot of slack, word spaces stretch first and Chinese characters only
    a little, so the result is not "常 识 推 理"."""
    from app.formats.pdf_align import justify_extras

    text = "常识推理(0-shot): Hellaswag [28], Winogrande [21], PIQA [4],"
    extras = justify_extras(text, 60.0, 10.0)
    inside = extras[text.index("识")]
    space = extras[text.index(" ")]
    assert 0 < inside <= 1.0 + 1e-9
    # The two gaps before and after a space count together and stretch by at most another quarter em. The rest
    # stays at the line end
    space += extras[text.index(" ") - 1]
    assert space > inside * 2 and space <= 2.5 + 1e-9
    assert sum(extras) < 60.0
    # A Chinese line without spaces is still spread evenly
    plain = "我们在此提出一种采样机制已经写完了"
    extras = justify_extras(plain, 16.0, 10.0)
    assert len({round(extra, 6) for extra in extras}) == 1


def test_snake_case_identifiers_are_one_word() -> None:
    """An identifier such as l_partkey is one word: it does not wrap at the underscore, and justification does not open
    space around the underscore."""
    from app.formats.pdf_align import can_stretch
    from app.formats.pdf_flow import tokenize

    assert tokenize("{l_partkey, o_custkey}") == ["{", "l_partkey", ",", " ", "o_custkey", "}"]
    assert not can_stretch("l", "_") and not can_stretch("_", "p")
    assert can_stretch(",", "中") and can_stretch(",", " ")
    # URLs and emails are written as one piece, with no space opened around @ . / and no wrapping inside them
    email = "jimmy@psi.utoronto.ca"
    assert not any(can_stretch(a, b) for a, b in zip(email, email[1:]))
    assert tokenize("多伦多大学 jimmy@psi.utoronto.ca 获取")[6] == email
    assert tokenize("见 http://www.github.com/goodfeli/adversarial).")[2:] == [
        "http://www.github.com/goodfeli/adversarial",
        ")",
        ".",
    ]


def test_an_opening_bracket_does_not_end_a_line() -> None:
    """An opening bracket does not stay at the line end and wraps to the next line with the following word (instead of
    "{" hanging alone at the line end and "o_custkey}" on the next line)."""
    from app.formats.pdf_flow import break_lines

    lines = break_lines("视图范围: {l_partkey}, {o_custkey}", 10.0, 64.0, {})
    assert len(lines) > 1
    assert all(not line.rstrip().endswith(("{", "(", "[", "（")) for line in lines), lines
    assert "".join(lines).replace(" ", "") == "视图范围:{l_partkey},{o_custkey}"
    # An opening quote also goes with the following word, and a closing quote does not start a line
    quoted = break_lines(
        "这里, Victoria 有多种不同的指称方式: “Victoria Chen”“Megabucks Banking 的首席财务官”", 9.5, 300.0, {}
    )
    assert all(not line.rstrip().endswith("“") and not line.startswith("”") for line in quoted), quoted


def test_garbled_figure_label_is_recognized() -> None:
    from app.formats.pdf_blocks import _garbled_label

    assert _garbled_label("(c)$Bo8om(up$sta.s.cs$deriva.on$")
    assert not _garbled_label("Figure 2: Interaction of Orca with database system")
    assert not _garbled_label("Table 7 | Training costs, assuming the rental price is $2 per GPU hour.")


def test_sentence_around_formulas_is_translated_and_bare_formula_line_is_not() -> None:
    """The three words If, then and Otherwise outside the formula make it a sentence, so it is translated. For all with
    a long formula is still a formula line."""
    from app.formats.pdf_extract import extract_blocks

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=200)

    def write(y: float, pieces: list[tuple[str, str]]) -> None:
        x = 40.0
        for text, font in pieces:
            page.insert_text((x, y), text, fontsize=10, fontname=font)
            x += pymupdf.get_text_length(text, fontname=font, fontsize=10)

    write(
        60,
        [
            ("(b) If ", "Times-Roman"),
            ("C(v) < C(u)", "Symbol"),
            (", then ", "Times-Roman"),
            ("B = C(v) - C(u)", "Symbol"),
            (". Otherwise, ", "Times-Roman"),
            ("B = 0", "Symbol"),
            (".", "Times-Roman"),
        ],
    )
    write(120, [("For all ", "Times-Roman"), ("i: b - x - x - y < a + a + a + a", "Symbol"), (".", "Times-Roman")])
    texts = [block.text for block in extract_blocks(doc)]
    assert any("Otherwise" in text for text in texts)
    assert not any("For all" in text for text in texts)


def test_a_block_is_sized_by_its_base_line_text_not_a_subscript() -> None:
    """The model of d_model is a small subscript in the body font and the longest piece of the line; the block size
    still follows the and on the baseline."""
    from app.formats.pdf_extract import _main_span

    def span(text: str, font: str, size: float, y: float) -> Span:
        return {"text": text, "font": font, "size": size, "origin": (0, y)}

    line = [
        span("and", "NimbusRomNo9L-Regu", 10, 225.0),
        span(" W", "CMMI10", 10, 225.0),
        span("model", "NimbusRomNo9L-Regu", 5, 227.6),
        span(".", "NimbusRomNo9L-Regu", 10, 225.0),
    ]
    assert _main_span([line])["text"] == "and"


def test_white_label_on_a_black_box_is_pasted_back_as_is(tmp_path: Path) -> None:
    """A white-on-black Figure label is not rewritten in the body text color but put back as it is; the caption after
    it is translated."""
    src = tmp_path / "label.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.draw_rect(pymupdf.Rect(40, 92, 92, 104), color=None, fill=(0, 0, 0))
    page.insert_text((42, 102), "Figure 2.4", fontsize=9, fontname="hebo", color=(1, 1, 1))
    page.insert_text((98, 102), "Some selected ASCII codes for English letters.", fontsize=9)
    doc.save(src)
    doc.close()

    [out] = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))
    spans: list[Span] = [
        span
        for block in pymupdf.open(out)[0].get_text("dict")["blocks"]
        if block.get("type") == 0
        for line in block["lines"]
        for span in line["spans"]
        if span["text"].strip()
    ]
    text = "".join(span["text"] for span in spans)
    assert "[zh-CN]" in text and "Some selected ASCII codes" in text
    label = [span for span in spans if "Figure" in span["text"]]
    assert label and all(span["color"] == 0xFFFFFF for span in label)


def test_pdf_page_lines_and_tables_are_read_once_per_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding tables parses all text on the page, and each read of vector drawings or image boxes takes a few
    milliseconds too. Extraction, reading order and role detection share one result. Writing the translation reads
    them separately, which is not counted here."""
    from app.formats import pdf_layout
    from app.formats.pdf_extract import _extract
    from app.formats.pdf_tables import page_tables

    calls: dict[str, list[int]] = {"tables": [], "drawings": [], "images": []}

    def counted[**P, R](
        key: str, read: Callable[Concatenate[pymupdf.Page, P], R]
    ) -> Callable[Concatenate[pymupdf.Page, P], R]:
        def call(page: pymupdf.Page, /, *args: P.args, **kwargs: P.kwargs) -> R:
            calls[key].append(page.number)
            return read(page, *args, **kwargs)

        return call

    monkeypatch.setattr(pdf_layout, "page_tables", counted("tables", page_tables))
    monkeypatch.setattr(pymupdf.Page, "get_drawings", counted("drawings", pymupdf.Page.get_drawings))
    monkeypatch.setattr(pymupdf.Page, "get_image_info", counted("images", pymupdf.Page.get_image_info))
    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "A body paragraph long enough to be translated here.", fontsize=10)
    _extract(doc)
    assert {key: sorted(pages) for key, pages in calls.items()} == {key: [0, 1, 2] for key in calls}


def test_pdf_reports_its_preparation_and_writing_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Layout analysis and writing each take several minutes for a whole book. Progress is reported page by page so the
    job page does not look stuck."""
    from app.formats.pdf import translate_pdf

    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "A body paragraph long enough to be translated here.", fontsize=10)
    src = tmp_path / "pages.pdf"
    doc.save(src)
    stages: list[Stage] = []
    r = runner()
    original = r.report_stage

    def record(name: str, done: int = 0, total: int = 0, unit: str = "") -> None:
        stages.append((name, done, total, unit))
        original(name, done, total, unit)

    monkeypatch.setattr(r, "report_stage", record)
    run(translate_pdf(src, tmp_path, r, False, "zh-CN"))
    assert stages[:4] == [("分析版面", n, 3, "页") for n in (1, 2, 3)] + [("整理段落", 0, 0, "")]
    assert stages[4][0] == "准备译文" and stages[4][1] == 0 and stages[4][3] == "段"
    # Translation start clears the preparation stage; once all translations are in, writing proceeds page by page
    after = stages[stages.index(("", 0, 0, "")) :]
    assert [stage for stage in after if stage[0] == "写回译文"] == [("写回译文", n, 3, "页") for n in (0, 1, 2)]
    assert after[-1] == ("生成文件", 0, 0, "")


def test_the_layout_model_reports_each_page_it_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The layout model takes about two seconds per page; it runs page by page before roles are assigned, reporting
    progress per page, and then returns to the "整理段落" stage"""
    import numpy as np
    import numpy.typing as npt

    from app.formats import pdf, pdf_units, pdf_vision

    class Meta:
        custom_metadata_map = {"names": "{0: 'title'}"}

    class Input:
        name = "images"

    class NoBoxes:
        def __init__(self) -> None:
            self.runs = 0

        def get_modelmeta(self) -> Meta:
            return Meta()

        def get_inputs(self) -> list[Input]:
            return [Input()]

        def run(self, _outputs: list[str] | None, _feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
            self.runs += 1
            return [np.zeros((1, 0, 6), np.float32)]

    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "A body paragraph long enough to be translated here.", fontsize=10)
    doc.new_page(width=420, height=240)  # Pages without text are not run
    src = tmp_path / "pages.pdf"
    doc.save(src)
    session = NoBoxes()
    monkeypatch.setattr(
        pdf_units, "layout_detector", lambda doc, path, boxes=None: pdf_vision.ModelLayout(doc, session)
    )
    stages: list[Stage] = []

    def record(name: str, done: int = 0, total: int = 0, unit: str = "") -> None:
        stages.append((name, done, total, unit))

    r = runner()
    monkeypatch.setattr(r, "report_stage", record)
    pdf._plan_pdf(src, r, "layout.onnx")
    start = stages.index(("版面模型分析", 1, 3, "页"))
    assert stages[start : start + 4] == [("版面模型分析", n, 3, "页") for n in (1, 2, 3)] + [("整理段落", 0, 0, "")]
    assert session.runs == 3


def test_reanalysis_after_a_code_change_reuses_the_model_boxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a code change invalidates the saved analysis result, the analysis runs again: the boxes of both models are
    read back from boxes.json without running the models, and the result matches the first run"""
    import numpy as np
    import numpy.typing as npt

    from app.formats import pdf, pdf_vision
    from app.formats.pdf_vision import FormulaModel

    class Meta:
        def __init__(self, names: str) -> None:
            self.custom_metadata_map = {"names": names}

    class Input:
        name = "images"

    runs = {"mfd": 0, "layout": 0}

    class Formula:
        """Box one inline formula on Q7 (center, width and height, scores for both classes, before NMS)"""

        def __init__(self, page: pymupdf.Page) -> None:
            self.page = page

        def get_modelmeta(self) -> Meta:
            return Meta("{0: 'embedding', 1: 'isolated'}")

        def get_inputs(self) -> list[Input]:
            return [Input()]

        def run(self, _outputs: list[str] | None, feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
            runs["mfd"] += 1
            height, width = feeds["images"].shape[2:]
            zoom = max(height, width) / max(self.page.rect.width, self.page.rect.height)
            left = (width - round(self.page.rect.width * zoom)) // 2
            top = (height - round(self.page.rect.height * zoom)) // 2
            r = self.page.search_for("Q7")[0]
            x0, y0, x1, y1 = r.x0 * zoom + left, r.y0 * zoom + top, r.x1 * zoom + left, r.y1 * zoom + top
            return [
                np.array([[[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.9, 0.0]]], np.float32).transpose(0, 2, 1)
            ]

    class Layout:
        def get_modelmeta(self) -> Meta:
            return Meta("{0: 'title'}")

        def get_inputs(self) -> list[Input]:
            return [Input()]

        def run(self, _outputs: list[str] | None, _feeds: dict[str, np.ndarray]) -> list[npt.NDArray[np.float32]]:
            runs["layout"] += 1
            return [np.zeros((1, 0, 6), np.float32)]

    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=420, height=240)
        page.insert_text((40, 60), "We set Q7 to the value found in the previous section here.", fontsize=10)
    src = tmp_path / "book.pdf"
    doc.save(src)
    for name in ("mfd.onnx", "layout.onnx"):
        (tmp_path / name).write_bytes(b"x")
    shown = pymupdf.open(src)[0]
    monkeypatch.setattr(
        pdf_vision,
        "_usable_sessions",
        lambda path, *args, **kwargs: [Formula(shown) if path.endswith("mfd.onnx") else Layout()],
    )

    plan_path = tmp_path / "plan.pickle"
    model = FormulaModel(str(tmp_path / "mfd.onnx"))
    first = pdf._saved_plan(src, runner(), str(tmp_path / "layout.onnx"), plan_path, model)
    assert runs == {"mfd": 2, "layout": 2} and (tmp_path / "boxes.json").is_file()
    assert any(f.get("text") == "Q7" for formulas in first.formulas for f in formulas)
    monkeypatch.setattr(pdf, "_code_version", lambda: "changed")
    again = pdf._saved_plan(src, runner(), str(tmp_path / "layout.onnx"), plan_path, model)
    assert runs == {"mfd": 2, "layout": 2}
    assert again.texts == first.texts and again.skips == first.skips
    assert [[f.get("text") for f in fs] for fs in again.formulas] == [
        [f.get("text") for f in fs] for fs in first.formulas
    ]


def test_pdf_keeps_lines_the_model_returns_unchanged(tmp_path: Path) -> None:
    """TeX's `u was composed into ù in the text sent for translation. When the model returns the line unchanged, it
    does not count as translated: the original glyphs stay and the line is not laid out again with the composed
    text."""
    from app.formats.pdf import translate_pdf

    class Echo(MockTranslator):
        async def translate_batch(self, texts: list[str]) -> list[str]:
            return list(texts)

    src = tmp_path / "codes.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((40, 60), "U+00F9 `u LATIN SMALL LETTER U WITH GRAVE", fontsize=10)
    doc.save(src)
    before = pymupdf.open(src)[0].get_text()
    assert "`u" in before

    [out] = run(translate_pdf(src, tmp_path, Runner(Echo("zh-CN"), cache=MemCache()), False, "zh-CN"))
    assert pymupdf.open(out)[0].get_text() == before


def test_shaded_areas_join_cells_and_leave_out_page_background() -> None:
    """A table with one background per cell and gaps between cells is merged into one area; a paper background
    covering the whole page does not count."""
    from app.formats.pdf_layout import scan_page

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.draw_rect(page.rect, color=None, fill=(0.98, 0.96, 0.9))
    for row in range(3):
        for x0, x1 in ((20, 60), (60.5, 150)):
            page.draw_rect(pymupdf.Rect(x0, 40 + row * 12.5, x1, 52 + row * 12.5), color=None, fill=(0.77, 0.86, 0.99))
    assert [tuple(round(v) for v in _bbox(area)) for area in scan_page(page).shades] == [(20, 40, 150, 77)]


def test_styles_carry_over_line_breaks() -> None:
    """Each broken line carries its own marks: bold and font size spanning lines are closed at the end of a line and
    reopened at the start of the next."""
    from app.formats.pdf_flow import break_lines

    lines = break_lines("前面的字{z80}{b}这一段很长的小字号粗体文字要跨过换行{/b}{/z}后面", 10, 100, {})
    assert len(lines) >= 2
    assert all(line.count("{b}") == line.count("{/b}") == 1 for line in lines[:-1])
    assert lines[1].startswith("{z80}{b}")


def test_cjk_line_keeps_bold_across_a_formula() -> None:
    """A formula cuts a line into pieces; the piece after the formula continues the bold opened before it and returns
    to regular after the closing mark."""
    import pymupdf

    from app.formats.pdf_align import _place_lines
    from app.formats.pdf_typeset import FormulaMetric, typeset_lines

    text = "前面{b}粗体\x01i\x021\x01/i\x02还是粗体{/b}后面"
    lines = typeset_lines(text, 10, 400, {1: FormulaMetric(10, 10, 8, 2)})
    placed = _place_lines(
        lines,
        pymupdf.Rect(0, 0, 400, 100),
        10,
        1.4,
        "#000000",
        "left",
        frozenset({1}),
        {},
        800,
    )
    assert placed is not None
    commands, _slots, _fonts = placed
    stroked, bold = [], False
    for op in commands.splitlines():
        if op == "2 Tr":
            bold = True
        elif op == "0 Tr":
            bold = False
        elif op.endswith("TJ") or op.endswith("Tj"):
            stroked.append(bold)
    assert stroked == [False, True, True, False]


def test_cjk_lines_rise_to_stay_inside_a_tight_first_line_box() -> None:
    """When the line box of the first original line is shorter than the Chinese glyphs, the whole paragraph moves up
    so the bottom of the glyphs does not overlap the line right below; at most until the glyph tops are level with
    the top of the box. When the line box is tall enough, nothing moves."""
    import pymupdf

    from app.formats.pdf_align import FIRST_BASELINE, _place_lines
    from app.formats.pdf_typeset import typeset_lines

    lines = typeset_lines("用于生成适合人类阅读的解答的提示词。", 10, 400, {})

    def baseline(first_room: float | None) -> float:
        placed = _place_lines(
            lines,
            pymupdf.Rect(0, 100, 400, 200),
            10,
            1.4,
            "#000000",
            "left",
            frozenset(),
            {},
            800,
            first_room=first_room,
        )
        assert placed is not None
        commands = placed[0].splitlines()
        tm = next(op for op in commands if op.endswith(" Tm"))
        return 800 - float(tm.split()[5])

    assert baseline(None) == pytest.approx(100 + 10 * FIRST_BASELINE, abs=0.01)
    assert baseline(13.0) == pytest.approx(100 + 10 * FIRST_BASELINE, abs=0.01)
    assert baseline(11.0) == pytest.approx(100 + 11.0 - 10 * 0.12, abs=0.01)
    assert baseline(5.0) == pytest.approx(100 + 10 * 0.88, abs=0.01)


def test_inline_colors_are_marked_relative_to_the_block() -> None:
    """Text in the block color gets no mark; other colors get {cRRGGBB}, and spaces between words of the same color do
    not break the mark; white text (caption labels) gets none. Color goes inside font size and outside emphasis,
    and marks nest in pairs."""
    from app.formats.pdf_flow import Emphasis, writer_for

    plain = Emphasis(False, False)
    spans: list[Span] = [
        {"text": "We cite ", "color": 0, "size": 10},
        {"text": "Clark", "color": 0x0000FF, "size": 10},
        {"text": " ", "color": 0, "size": 10},
        {"text": "2002", "color": 0x0000FF, "size": 10},
        {"text": " here.", "color": 0, "size": 10},
        {"text": "Figure 2.1", "color": 0xFFFFFF, "size": 10},
    ]
    writer = writer_for(spans, 10, 0)
    text = "".join(writer.text(s["text"], plain, s["size"], s["color"]) for s in spans) + writer.close()
    assert text == "We cite {c0000ff}Clark 2002{/c} here.Figure 2.1"

    same: list[Span] = [{"text": "All black text.", "color": 0, "size": 10}]
    writer = writer_for(same, 10, 0)
    assert writer.text("All black text.", plain, 10, 0) + writer.close() == "All black text."

    writer = writer_for([{"text": "abc", "color": 0, "size": 10}, {"text": "x", "color": 0xFF0000, "size": 6}], 10, 0)
    out = writer.text("abc ", Emphasis(True, False), 10, 0) + writer.text("x", Emphasis(True, False), 6, 0xFF0000)
    assert out + writer.close() == "{b}abc {/b}{z60}{cff0000}{b}x{/b}{/c}{/z}"


def test_colors_carry_over_line_breaks() -> None:
    """Each broken line carries its own marks: a color spanning lines and the bold inside it are closed at the end of a
    line and reopened at the start of the next."""
    from app.formats.pdf_flow import break_lines

    lines = break_lines("前面的字{c0000ff}{b}这一段很长的蓝色粗体链接文字要跨过换行{/b}{/c}后面", 10, 100, {})
    assert len(lines) >= 2
    assert all(line.count("{c0000ff}") == line.count("{/c}") == 1 for line in lines[:-1])
    assert lines[1].startswith("{c0000ff}{b}")


def test_pdf_keeps_inline_colors_in_the_translation(tmp_path: Path) -> None:
    """Blue citations and red strings in body text keep their colors in the translation, including the parts spanning
    lines; the rest of the text is in the block color. When the model garbles the color marks (a missing closing
    mark), the whole paragraph is written in the block color and the second half is not colored."""
    from app.formats.pdf import translate_pdf

    long_blue = "这是一段很长很长的蓝色链接文字需要跨过换行才能写完" * 3

    class Colored(MockTranslator):
        async def translate_batch(self, texts: list[str]) -> list[str]:
            out = []
            for text in texts:
                if "{c0000ff}" not in text:
                    out.append(text)
                elif "broken" in text:
                    out.append("前面的字{c0000ff}没有收尾的蓝色，后面的字也会被染成蓝色吗")
                else:
                    out.append(
                        "开头的一些普通文字，" + "{c0000ff}" + long_blue + "{/c}" + "，以及{cff0000}红字{/c}结尾。"
                    )
            return out

    src = tmp_path / "colors.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    for y, middle in ((60, "Weizenbaum (1966)"), (140, "broken link")):
        x = 40.0
        for text, color in (
            ("We cite ", (0, 0, 0)),
            (middle, (0, 0, 1)),
            (" and ", (0, 0, 0)),
            ("xxxxxx", (1, 0, 0)),
            (" in this body text.", (0, 0, 0)),
        ):
            page.insert_text((x, y), text, fontsize=10, color=color)
            x += pymupdf.get_text_length(text, fontsize=10)
    doc.save(src)

    [out] = run(translate_pdf(src, tmp_path, Runner(Colored("zh-CN"), cache=MemCache()), False, "zh-CN"))
    spans: list[Span] = [
        s for b in pymupdf.open(out)[0].get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"]
    ]
    blue = "".join(s["text"] for s in spans if s["color"] == 0x0000FF)
    red = "".join(s["text"] for s in spans if s["color"] == 0xFF0000)
    assert blue.replace(" ", "") == long_blue
    assert red == "红字"
    assert "没有收尾的蓝色" in "".join(s["text"] for s in spans if s["color"] == 0)


def test_cjk_line_keeps_colors_across_a_formula() -> None:
    """A formula cuts a line into pieces; the piece after the formula continues the color opened before it and returns
    to the block color after the closing mark."""
    import pymupdf

    from app.formats.pdf_align import _place_lines
    from app.formats.pdf_typeset import FormulaMetric, typeset_lines

    text = "前面{c0000ff}蓝色\x01i\x021\x01/i\x02还是蓝色{/c}后面"
    lines = typeset_lines(text, 10, 400, {1: FormulaMetric(10, 10, 8, 2)})
    placed = _place_lines(
        lines,
        pymupdf.Rect(0, 0, 400, 100),
        10,
        1.4,
        "#000000",
        "left",
        frozenset({1}),
        {},
        800,
    )
    assert placed is not None
    commands, _slots, _fonts = placed
    painted, now = [], None
    for op in commands.splitlines():
        if op.endswith(" rg"):
            now = op
        elif op.endswith("TJ") or op.endswith("Tj"):
            painted.append(now)
    blue, black = "0.0000 0.0000 1.0000 rg", "0.0000 0.0000 0.0000 rg"
    assert painted == [black, blue, blue, black]


def test_toc_entries_send_only_titles_and_keep_page_numbers(tmp_path: Path) -> None:
    """Only the title of a contents entry is sent for translation: section number, leader and page number are not;
    the page number stays in place right-aligned, and the leader is extended to the end of the translation. The
    same holds when the dots share a span with the title, and for entries with a page number but no leader (a
    volume name line); each entry is a paragraph of its own."""
    from app.formats.pdf import translate_pdf

    sent: list[str] = []

    class Short(MockTranslator):
        async def translate_batch(self, texts: list[str]) -> list[str]:
            sent.extend(texts)
            return ["标题" + str(index) for index, _ in enumerate(texts)]

    src = tmp_path / "toc.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((40, 50), "Part One: Foundations of Language", fontsize=12)
    page.insert_text((360, 50), "1", fontsize=12)
    rows = (("1.1 What is a Large Language Model", "5"), ("1.2 Tokens", "8"), ("Historical Notes", "12"))
    for row, (title, number) in enumerate(rows):
        y = 70 + row * 14
        page.insert_text((50, y), title, fontsize=10)
        x = 50 + pymupdf.get_text_length(title, fontsize=10) + 6
        while x < 345:
            page.insert_text((x, y), ".", fontsize=10)
            x += 5
        page.insert_text((360, y), number, fontsize=10)
    # Dots and title in the same span
    page.insert_text((50, 112), "Exercises" + " ." * 45, fontsize=10)
    page.insert_text((360, 112), "13", fontsize=10)
    doc.save(src)
    before = {w[4]: (round(w[0], 1), round(w[1], 1)) for w in pymupdf.open(src)[0].get_text("words") if w[4].isdigit()}

    [out] = run(translate_pdf(src, tmp_path, Runner(Short("zh-CN"), cache=MemCache()), False, "zh-CN"))
    assert sorted(sent) == sorted(
        [
            "Part One: Foundations of Language",
            "What is a Large Language Model",
            "Tokens",
            "Historical Notes",
            "Exercises",
        ]
    )
    result = pymupdf.open(out)[0]
    words = result.get_text("words")
    after = {w[4]: (round(w[0], 1), round(w[1], 1)) for w in words if w[4].isdigit()}
    assert after == before
    assert not any(w[4] in ("Exercises", "Notes", "Tokens") for w in words)
    # Each leader runs from the end of the translation to the page number without a gap
    for line in [line for b in result.get_text("dict")["blocks"] for line in b.get("lines", [])]:
        text = "".join(s["text"] for s in line["spans"])
        if text.startswith("标题") and "." in text:
            title = next(s for s in line["spans"] if s["text"].startswith("标题"))
            dots = [s for s in line["spans"] if s["text"].strip(" .") == "" and "." in s["text"]]
            assert dots and dots[0]["bbox"][0] - title["bbox"][2] < 8


def test_block_color_follows_most_characters_not_the_longest_span() -> None:
    """In a book with one span per word, the longest piece is a white-on-black caption label. Taking the color from it
    would write the translation of the whole caption in white."""
    from app.formats.pdf_blocks import _by_chars

    words: list[Span] = [
        {"text": word, "color": 0} for word in ("The", "all-words", "WSD", "task,", "mapping", "from", "input")
    ]
    spans: list[Span] = [{"text": "Figure 10.3", "color": 0xFFFFFF}, *words]
    assert _by_chars(spans, lambda span: span["color"]) == 0


def test_a_contents_page_without_leader_dots_is_found_by_its_page_numbers() -> None:
    """A table of contents without leaders: the heading Table of Contents, and page numbers far from the line ends,
    right-aligned and increasing downward. A list of tables whose page numbers start over, and a continuation page
    without a heading after a contents page, count too. A table with a column of integers and no heading that is
    not next to a contents page does not."""
    from app.formats.pdf_extract import _toc_pages

    def line(text: str, x0: float, x1: float, y: float) -> Line:
        return {"dir": (1, 0), "bbox": (x0, y, x1, y + 12), "spans": [{"text": text, "size": 12.0}]}

    def page(title: str, rows: list[tuple[str, int]]) -> PageDict:
        lines = [line(title, 241, 371, 70)] if title else []
        for k, (entry, number) in enumerate(rows):
            y = 100 + 14.5 * k
            lines += [line(entry, 90, 90 + 6 * len(entry), y), line(str(number), 522 - 6 * len(str(number)), 522, y)]
        return {"blocks": [{"type": 0, "lines": lines}]}

    entries = [
        ("Executive Summary", 1),
        ("1 Framing Risk", 4),
        ("1.1 Understanding Risks", 4),
        ("1.2 Challenges", 5),
        ("2 Audience", 9),
        ("3 AI Risks", 12),
        ("Appendix D", 42),
        ("Table 1 GOVERN", 22),
        ("Table 2 MAP", 26),
    ]
    contents = page("Table of Contents", entries)
    following = page(
        "",
        [
            ("Figure 1 Harms", 5),
            ("Figure 2 Lifecycle", 10),
            ("Figure 3 Actors", 11),
            ("Figure 4 Characteristics", 12),
            ("Figure 5 Functions", 20),
        ],
    )
    counts = page("", [("Romeo", 1), ("salad", 2), ("Falstaff", 4), ("forest", 12), ("battle", 21)])
    assert _toc_pages([counts, contents, following]) == {1, 2}


def test_the_line_gap_is_measured_from_the_documents_body_lines() -> None:
    """Two tightly set lines in a font with tall line boxes overlap a little, so the measured gap is negative. When
    there is too little body text to measure, a common value is used."""
    from app.formats.pdf_blocks import _LINE_GAP
    from app.formats.pdf_extract import _line_gap

    def page(count: int, top: float = 100.0, height: float = 13.3, pitch: float = 11.0, size: float = 10.0) -> PageDict:
        lines: list[Line] = [
            {
                "dir": (1, 0),
                "bbox": (72, top + pitch * k, 520, top + pitch * k + height),
                "spans": [{"text": "words of the body text", "size": size}],
            }
            for k in range(count)
        ]
        return {"blocks": [{"type": 0, "lines": lines}]}

    assert _line_gap([page(30)]) == -0.25
    # Only lines at body size are measured; headings do not count
    assert _line_gap([page(30), page(5, top=40, height=20, pitch=30, size=20)]) == -0.25
    assert _line_gap([page(5)]) == _LINE_GAP


def test_a_line_tail_pair_measures_its_gap_against_the_documents_line_gap() -> None:
    """For a half sentence at the line end and its continuation back at the column edge, the line spacing also
    subtracts the gap between two tightly set lines of this document. Pairing at extraction and recognizing again
    at writing both use the same value stored on the block."""
    from app.formats.pdf_blocks import TextBlock, _same_page_line_pair

    def pair(line_gap: float) -> tuple[TextBlock, TextBlock]:
        def blk(text: str, rect: pymupdf.Rect) -> TextBlock:
            return TextBlock(
                page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False, line_gap=line_gap
            )

        return (
            blk("which we call the", pymupdf.Rect(300, 100, 480, 112)),
            blk("unigram model of the text", pymupdf.Rect(72, 127, 480, 139)),
        )

    assert _same_page_line_pair(*pair(0.1))
    assert not _same_page_line_pair(*pair(-0.2))


def test_a_block_cut_by_a_margin_note_does_not_take_an_indented_new_paragraph() -> None:
    """A side note cuts a body paragraph into an upper and a lower block, which are joined back once the side note is
    removed. A next block with its own first-line indent starts a new paragraph: a new paragraph follows the
    translation of an example. Only an unindented next block is the same sentence cut apart."""
    from app.formats.pdf_extract import _note_cut

    def block(*lines: tuple[str, Corners]) -> Block:
        return {
            "lines": [{"dir": (1, 0), "bbox": bbox, "spans": [{"text": text, "size": 10.0}]} for text, bbox in lines]
        }

    gloss = block(("He said the biggest house in the country was this one", (174.1, 307.2, 396.9, 319.1)))
    paragraph = block(
        ("Alternatively, in languages like Koryak, a Chukotko-Kamchatkan language", (157.1, 323.1, 475.2, 335.1)),
        ("spoken in the northern part of the Kamchatka peninsula in Russia, a single", (142.2, 335.1, 475.2, 347.1)),
    )
    assert not _note_cut(gloss, paragraph)
    upper = block(
        ("Alternatively, in languages like Koryak, a Chukotko-Kamchatkan language", (142.2, 307.2, 475.2, 319.1))
    )
    lower = block(
        ("spoken in the northern part of the Kamchatka peninsula in Russia, a single", (142.2, 323.1, 475.2, 335.1)),
        ("word may have very many morphemes", (142.2, 335.1, 400.0, 347.1)),
    )
    assert _note_cut(upper, lower)
    # Below the hanging-indent continuation lines of a list item, a line starting left of them is body text after
    # the list
    item = block(
        ("2. embeddings, vectors that represent the meanings of tokens and concepts in-", (173.0, 234.0, 493.0, 247.0)),
        ("side the network", (185.0, 247.0, 251.0, 259.0)),
    )
    after = block(
        ("Neural networks are the fundamental computational tool for language pro-", (175.0, 262.0, 493.0, 275.0))
    )
    assert not _note_cut(item, after)


def test_a_fragment_inside_a_kept_formula_is_part_of_it() -> None:
    """The denominator dx of a fraction forms its own line inside the box of a formula line that keeps its original
    glyphs (dv · dv), so it is a piece of that formula."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_extract import _inside_kept_formula

    formula = pymupdf.Rect(329.5, 429.0, 357.5, 461.4)
    piece = pymupdf.Rect(347.6, 442.6, 357.5, 454.5)
    block = TextBlock(page=0, rect=piece, line_rects=[piece], text="dx", size=10, color="#000000", bold=False)
    assert _inside_kept_formula(block, [formula])
    label = TextBlock(page=0, rect=piece, line_rects=[piece], text="prior", size=10, color="#000000", bold=False)
    assert not _inside_kept_formula(label, [formula])
    assert not _inside_kept_formula(block, [pymupdf.Rect(100, 100, 200, 120)])


def test_no_borderless_table_is_found_on_an_index_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """On an index page, columns of short entries with aligned line starts look like a table without rules. As on a
    contents page, no table search runs, and the entries are translated."""
    from dataclasses import replace

    from app.formats import pdf_extract
    from app.formats.pdf_drawings import RuledTable
    from app.formats.pdf_layout import PageScan, scan_page

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    for column, x in enumerate((60, 260)):
        for row in range(14):
            page.insert_text((x, 80 + 12 * row), f"entry term {column}{row}, {100 + row}", fontsize=8)

    def scan(page: pymupdf.Page) -> PageScan:
        found = scan_page(page)
        cells = tuple(pymupdf.Rect(x, 72 + 12 * row, x + 150, 84 + 12 * row) for row in range(3) for x in (58, 258))
        return replace(found, tables=(RuledTable(pymupdf.Rect(58, 72, 408, 108), cells),))

    monkeypatch.setattr(pdf_extract, "scan_page", scan)
    blocks, _frames, scans = pdf_extract._extract(doc)
    assert scans[0].tables == ()
    assert all(block.clip is None for block in blocks)


def test_an_index_entry_with_many_page_numbers_is_translated() -> None:
    """A headword followed by a run of page numbers, with letters making up less than 40 percent, is still an entry
    and does not keep the original text as a line of numbers. A line with only page numbers still keeps the
    original text. A headword of one or two letters (F0) is an entry too."""
    from app.formats import pdf_extract

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    for row in range(16):
        page.insert_text((60, 80 + 12 * row), f"entry term {row}, {100 + row}", fontsize=8)
    page.insert_text((60, 290), "perplexity, 27, 76, 88, 203", fontsize=8)
    page.insert_text((60, 302), "CRF, 414", fontsize=8)
    page.insert_text((60, 314), "F0, 340", fontsize=8)
    blocks, _frames, _scans = pdf_extract._extract(doc)
    texts = {block.text for block in blocks}
    assert "perplexity, 27, 76, 88, 203" in texts and "CRF, 414" in texts
    # A one-letter headword forms a block too and moves during index reflow instead of staying in place and being
    # covered
    assert "F0, 340" in texts


def test_the_index_at_the_back_stays_original(tmp_path: Path) -> None:
    """The index at the back of a book is sorted by the original alphabet and its page numbers point to the original
    pages, so the whole page keeps the original text. The body text before it is translated."""
    from app.formats.pdf import _plan_pdf

    doc = pymupdf.open()
    body = doc.new_page(width=500, height=700)
    body.insert_text((60, 80), "Perplexity measures how well a model predicts a sample of text.", fontsize=10)
    index = doc.new_page(width=500, height=700)
    for row in range(16):
        index.insert_text((60, 80 + 12 * row), f"entry term {row}, {100 + row}", fontsize=8)
    src = tmp_path / "book.pdf"
    doc.save(src)
    plan = _plan_pdf(src, runner(), "")
    skipped = {text: skip for text, skip in zip(plan.texts, plan.skips)}
    assert not skipped["Perplexity measures how well a model predicts a sample of text."]
    entries = [skip for text, skip in skipped.items() if text.startswith("entry term")]
    assert len(entries) == 16 and all(entries)


def test_the_left_column_of_an_index_page_is_not_a_side_note() -> None:
    """An index page has several columns, and the leftmost lies outside the text area of the body, like side notes in
    the margin. Short words in side notes keep the original text, so taking the column for side notes would leave
    the index entries untranslated."""
    from app.formats import pdf_extract

    doc = pymupdf.open()
    for number in range(4):
        page = doc.new_page(width=500, height=700)
        for row in range(40):
            word = "abcdefghij"[number] + "klmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"[row]
            page.insert_text(
                (140, 80 + 14 * row), f"ordinary body text {word} that runs across the column", fontsize=10
            )
    page = doc.new_page(width=500, height=700)
    for column, x in enumerate((40, 200, 330)):
        for row in range(30):
            page.insert_text((x, 80 + 12 * row), f"term {column} {row}, {100 + row}", fontsize=8)
    blocks, _frames, _scans = pdf_extract._extract(doc)
    index = [block for block in blocks if block.page == 4]
    assert index and not any(block.side_note for block in index)


def test_a_clipped_plot_path_does_not_cover_the_text_below_it() -> None:
    """The path box of a curve clipped into a figure frame reaches tens of thousands of points off the page. Clipped
    to the clip area, only the part inside the frame remains, so it cannot cover the body text below. Paths entirely
    outside the clip area are removed."""
    from app.formats.pdf_drawings import visible_drawings

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 620), "Otherwise, we take the simulation snapshot before the binary", fontsize=10)
    xref = page.get_contents()[0]
    stream = doc.xref_stream(xref) + (
        b"\nq 81 628 207 154 re W n 1 0 0 rg 90 746 m 279 746 l 279 -31000 l 90 -31000 l h f Q"
        b"\nq 81 628 207 154 re W n 0 1 0 rg 300 700 m 320 700 l 320 720 l h f Q\n"
    )
    doc.update_stream(xref, stream)
    rects = [drawing["rect"] for drawing in visible_drawings(page)]
    assert len(rects) == 1
    assert rects[0].y1 <= 842 - 628 + 1 and rects[0].y0 >= 842 - 782 - 1


def test_the_margin_piece_of_a_wide_caption_line_is_not_a_side_note() -> None:
    """In a caption reaching into the page margin, inline formulas cut a line into blocks, and that w[1] left in the
    margin forms a block of its own. The lines right above and below start at the same x and run into the text
    area, so it is a line of the caption, not a side note. A side note beside body text is still a side note."""
    from app.formats.pdf_extract import _wide_block_lines
    from app.formats.pdf_frame import FrameLine, TextFrame

    rows = [
        ((96, 570, 140, 581), "Figure 6.18", "a"),
        ((150, 570, 493, 581), "Sample computation graph for a simple 2-layer neural net with", "a"),
        ((95, 581, 493, 592), "and 2 hidden units. We have adjusted the notation a bit to avoid", "a"),
        ((95, 591, 468, 605), "the function that is being computed, and the resulting variable name.", "a"),
        ((95, 605, 125, 619), "that w[1]", "b"),
        ((117, 605, 493, 622), "11 is to be multiplied by x1, and the node z[1] = + means that", "c"),
        ((95, 619.4, 351, 633), "the three nodes that feed into it (the two products, and the bias)", "d"),
        ((100, 300, 150, 310), "regular expressions", "e"),
        ((160, 289, 493, 300), "body text that runs across the whole column of the page here", "f"),
        ((160, 301, 493, 312), "and goes on with more of the same body text below the note", "f"),
    ]
    blocks: dict[str, Block] = {}
    lines: list[FrameLine] = []
    refs: list[tuple[Block, Line]] = []
    for bbox, text, key in rows:
        line: Line = {"bbox": bbox}
        blocks.setdefault(key, {"lines": []})["lines"].append(line)
        lines.append(FrameLine(pymupdf.Rect(bbox), text, 9))
        refs.append((blocks[key], line))
    kept = _wide_block_lines(lines, refs, {0, 4, 7}, TextFrame(160, 493, 108, 719, 10))
    assert kept == {0, 4}
