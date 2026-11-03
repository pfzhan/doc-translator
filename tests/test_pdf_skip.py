"""Choosing the blocks that keep the original text (pdf_skip)."""

import pymupdf
import pytest

from tests.helpers import SAMPLES


def test_pdf_biblio_skips_alpha_labels_at_uniform_size():
    """References with author-abbreviation labels at the body size (OCR of bitmap fonts makes the sizes uniform): a
    label whose [ was lost by OCR, a label on its own line and a short continuation line ending with a period must
    not be taken for the next chapter, and the whole section keeps the original text."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size=10):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=False
        )

    blocks = [
        blk("We thank our colleagues for help with the experiments."),
        blk("References"),
        blk("[CS94] S. Chaudhuri and K. Shim. Including Group-By in Query Optimization. In VLDB, 1994."),
        blk("ESS]"),
        blk("Arbor Software Inc. Multidimensional Analysis: Converting Corporate Data into Strate-"),
        blk("GBLP95] J. Gray, A. Bosworth, A. Layman, H. Pirahesh Data Cube: A Relational Aggregation"),
        blk("Operator Generalizing Group-By, Cross-Tab, and Sub-Totals. Microsoft Technical Report"),
        blk("STG]"),
        blk("Databases. White Paper."),
        blk("[YL95] W. P. Yan and P. A. Larson. Eager Aggregation and Lazy Aggregation. In VLDB, 1995."),
    ]
    assert biblio_skips(blocks) == [False, False] + [True] * 8
    # Entries ending with a period do not keep the next chapter heading, which has no period, from ending the
    # section
    tail = blocks[:3] + [blk("Appendix"), blk("Appendix body text continues here.")]
    assert biblio_skips(tail) == [False, False, True, False, False]


def test_pdf_biblio_skips_weigh_sizes_by_text():
    """Runs of small-text table cells outnumber the body blocks, so a median by block count lands on the small size
    and citations at body size look like headings. The median is taken by character count. A heading one size
    smaller than the body, in a references section that is one size smaller throughout (ICASSP), is bold and no
    smaller than the citations, so it is a heading too."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    body = [
        blk("Learning widely applicable representations of words has been an active area of research " * 4, 10)
        for _ in range(3)
    ]
    cells = [blk(f"{80 + n}.5", 9) for n in range(20)]
    entries = [
        blk("Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018.", 10),
        blk("Contextual string embeddings for sequence labeling.", 10),
        blk("Rami Al-Rfou, Dokook Choe, and Llion Jones. 2018.", 10),
        blk("Character-level language modeling with deeper self-attention.", 10),
    ]
    skips = biblio_skips([*body, *cells, blk("References", 12), *entries])
    assert skips[-4:] == [True] * 4
    small = [
        blk("[1] R. Xu and Q. Liu, \u201cX-Voice,\u201d in Proc. Interspeech, 2023, pp. 1\u20135.", 8),
        blk("[2] M. Yang and J. Hansen, \u201cActivation steering,\u201d in Proc. ICASSP, 2024.", 8),
    ]
    skips = biblio_skips(
        [*[blk("Body text of the paper. " * 20, 9) for _ in range(5)], blk("8. REFERENCES", 8, True), *small]
    )
    assert skips[-2:] == [True, True]


def test_pdf_biblio_skips_follows_font_size():
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, page=0, bold=False):
        return TextBlock(
            page=page, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("Chapter body text.", 10),
        blk("References", 14),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Jones, A. (2019). Another.", 10),
        blk("Index", 14),
        blk("Index entry text.", 10),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, False, False]
    # A running header must not lower the threshold; a citation of the same size is no exit heading; a slightly
    # smaller next chapter heading ends the section
    mixed = [
        blk("Chapter body text.", 11),
        blk("References", 16),
        blk("Smith, J. (2020). Some book.", 12),
        blk("References", 9),
        blk("Jones, A. (2019). Another.", 12),
        blk("Appendix: supplementary notes for the study of this volume", 14),
        blk("Appendix paragraph here.", 11),
    ]
    assert biblio_skips(mixed) == [False, False, True, True, True, False, False]
    # References in a table of contents has no citations after it, so the section is not entered
    toc = [
        blk("References", 14),
        blk("A normal chapter paragraph without a citation year.", 11),
        blk("More body text here.", 11),
    ]
    assert biblio_skips(toc) == [False, False, False]
    # When the heading and the citations have the same size, the citations still keep the original text and the
    # next heading ends the section
    same = [
        blk("Intro paragraph.", 11),
        blk("参考文献：", 12),
        blk("Smith, J. (2020). Some book.", 12),
        blk("Index", 12),
        blk("Index entry text.", 11),
    ]
    assert biblio_skips(same) == [False, False, True, False, False]


def test_pdf_biblio_entry_needs_a_citation_not_a_year_in_prose():
    """Years and the word arXiv in body text, and a smaller next heading, do not confirm entering the references
    section."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    prose = [
        blk("Chapter body text.", 11),
        blk("References", 14),
        blk("In 2017 the authors published a follow-up.", 11),
        blk("WMT 2014 is a common benchmark.", 11),
        blk("See the arXiv preprint for details.", 11),
        blk("More body text here.", 11),
    ]
    assert biblio_skips(prose) == [False] * 6
    zh = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14),
        blk("这项工作发表于2016年。", 11),
        blk("后续正文还在这里。", 11),
    ]
    assert biblio_skips(zh) == [False, False, False, False]
    # A 14pt table-of-contents heading followed by a smaller bold chapter: the lookahead must stop and not swallow
    # the chapter
    toc = [
        blk("Contents line.", 11),
        blk("References", 14),
        blk("Introduction", 11, bold=True),
        blk("In 2017 the authors wrote this chapter.", 11),
        blk("The chapter continues here.", 11),
    ]
    assert biblio_skips(toc) == [False] * 5


def test_pdf_biblio_numbered_entries_without_a_heading():
    """The revtex journal layout does not print References: a run starting at [1] with consecutive numbers below is
    the references section. It stops at a heading and resumes at the next number continuing in another column. A
    body sentence starting with [1] does not count."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size=9.0, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("In future work we will present a more detailed exposition of our methods.", 10),
        blk("[1] B. P. Abbott et al. (LIGO Scientific, Virgo), Phys. Rev. Lett. 116, 061102 (2016)."),
        blk("[2] R. Abbott et al., Phys. Rev. X 13, 041039 (2023), arXiv:2111.03606 [gr-qc]."),
        blk("[3] A. Buonanno and T. Damour, Phys. Rev. D 59, 084006"),
        blk("(1999), arXiv:gr-qc/9811091. [4] A. Buonanno and T. Damour, Phys. Rev. D 62, 064015"),
        blk("(2000), arXiv:gr-qc/0001013."),
        blk("Appendix F: Control-space dimension and optimizer", bold=True),
        blk("Table VIII compares 21- and 27-parameter controls at all three stages.", 10),
        blk("Phys. Rev. D 108, 124035 (2023). [5] L. Pompili et al., Phys. Rev. D 108, 124035 (2023)."),
        blk("[6] https://github.com/DUsseglio/Scattering-Waveform-1SF."),
        blk("SUPPLEMENTAL MATERIAL", bold=True),
        blk("The leading multipole moments take the form below.", 10),
    ]
    assert biblio_skips(blocks) == [False, True, True, True, True, True, False, False, True, True, False, False]
    assert biblio_skips(
        [blk("[1] We first fix the gauge, Phys. Rev. D 59 (1999).", 10), blk("The second step follows.", 10)]
    ) == [False, False]


def test_pdf_biblio_bold_heading_as_large_as_the_entries():
    """A run of author names from the broken first line of an entry, and short title lines after the author names,
    are not the next chapter even at heading size; the bold appendix heading after them is."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=10.0, color="#000000", bold=bold
        )

    blocks = [
        blk("This work was supported by the National Science Foundation under Grant 2200000."),
        blk("References", bold=True),
        blk("Ahmed M. Alaa, Zaid Ahmad, and Mark van der Laan."),
        blk(
            "Conformal meta-learners for predictive inference of treatment effects. "
            "arXiv preprint arXiv:2308.14895, 2023."
        ),
        blk("Anastasios N Angelopoulos, Stephen Bates, Adam"),
        blk("Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. In ICLR, 2024."),
        blk("Jean-Yves Audibert, Sébastien Bubeck, and Rémi"),
        blk("Munos. Best arm identification in multi-armed bandits. In COLT, pages 41–53, 2010."),
        blk("Jiayuan Su, Jing Luo, Hongwei Wang, and Lu Cheng."),
        blk("API is enough: Conformal prediction for large lan-"),
        blk("guage models without logit-access. In Findings of EMNLP, 2024."),
        blk("Proofs and Theoretical Results", bold=True),
        blk("We first restate the main theorem and then prove it in two steps."),
    ]
    assert biblio_skips(blocks) == [False, False, *[True] * 9, False, False]


def test_pdf_biblio_author_year_entries_after_the_heading():
    """Author-year entries have no numbers and no parenthesized years. Several consecutive entries starting with
    "Surname, Initials." after the heading enter the references section; a single body sentence starting this way
    does not count."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=False
        )

    book = [
        blk("A potential DAMSL labeling of the conversational fragment.", 9),
        blk("Bibliography", 24),
        blk("Abadi, M., A. Agarwal, P. Barham, Agirre, E. and D. Martinez. 2001.", 8),
        blk("E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin.", 8),
        blk("Agichtein, E. and L. Gravano. 2000.", 8),
        blk("Snowball: Extracting relations from large plain-text collections.", 8),
        blk("Subject Index", 24),
    ]
    assert biblio_skips(book) == [False, False, True, True, True, True, False]
    prose = [
        blk("Bibliography", 14),
        blk("Smith, J. and colleagues kept the sources in a shared folder.", 11),
        blk("The chapter continues here.", 11),
    ]
    assert biblio_skips(prose) == [False] * 3


def test_pdf_biblio_numbered_entry_ending_with_pages():
    """Entries in old database papers read "venue year, pages", so the line ends with pages instead of a year; they
    also enter the references section."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("Body text of the conclusion.", 9),
        blk("8. References", 12, bold=True),
        blk(
            "[1] S. Agrawal, S. Chaudhuri, V. R. Narasayya: Automated Selection of "
            "Materialized Views and Indexes in SQL Databases. VLDB 2000: 496-505",
            8,
        ),
        blk(
            "[4] S. Chaudhuri, S. Krishnamurthy, S. Potamianos, K. Shim, Optimizing "
            "Queries with Materialized Views, ICDE 1995, 190-200.",
            8,
        ),
        blk(
            "[6] G. Graefe, The Cascades Framework for Query Optimization, Data "
            "Engineering Bulletin, 18(3), 1995, 19-29.",
            8,
        ),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True]
    # Unnumbered body text does not count even when it ends with "year, pages"
    prose = [
        blk("Body text.", 9),
        blk("References", 12, bold=True),
        blk("The workshop ran in 2001, 10-12.", 9),
        blk("More body text.", 9),
    ]
    assert biblio_skips(prose) == [False] * 4


def test_pdf_biblio_same_size_citations_do_not_exit():
    """A bare year at the line end, CoRR, numbered entries and wrapped continuation lines are not the next chapter; a
    bold heading at body size ends the section."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    same = [
        blk("Intro paragraph.", 11),
        blk("References", 12, bold=True),
        blk("Vaswani, A. Attention is all you need. CoRR, abs/1706.03762, 2017.", 12),
        blk("1. Zhang, S. A paper title goes here. 2017.", 12),
        blk("[2] Dzmitry Bahdanau and Yoshua Bengio. Neural machine translation by jointly", 12),
        blk("learning to align and translate. CoRR, abs/1409.0473, 2014.", 12),
        blk("张三. 书名. 2016.", 12),
        blk("Index", 12, bold=True),
        blk("Index entry text.", 11),
    ]
    assert biblio_skips(same) == [False, False, True, True, True, True, True, False, False]
    bold_next = [
        blk("Chapter body text.", 10),
        blk("Another body line.", 10),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Acknowledgements", 10.1, bold=True),
        blk("Thanks to the reviewers.", 10),
    ]
    assert biblio_skips(bold_next) == [False, False, False, True, False, False]
    numbered_heading = [
        blk("Chapter body text.", 10),
        blk("References", 12, bold=True),
        blk("[1] Smith, J. A paper. CoRR, abs/1706.03762, 2017.", 10),
        blk("[1] Appendix", 14, bold=True),
        blk("Appendix paragraph here.", 10),
    ]
    assert biblio_skips(numbered_heading) == [False, False, True, False, False]
    appendix = [
        blk("Chapter body text.", 10),
        blk("Another body line.", 10),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("1. Appendix", 10.1, bold=True),
        blk("Appendix paragraph here.", 10),
    ]
    assert biblio_skips(appendix) == [False, False, False, True, False, False]


def test_pdf_biblio_author_lists_and_url_halves_stay_in():
    """ACL style (Author. Year. Title. In Venue) enters the references section; when the entries are one size larger
    than the body, an author line and half of a wrapped URL are not the next chapter."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    acl = [
        blk("Body text of the paper.", 10),
        blk("References", 12, bold=True),
        blk("Alan Akbik, Duncan Blythe, and Roland Vollgraf.", 10),
        blk("2018. Contextual string embeddings for sequence labeling. In Proceedings of COLING, pages 1638–1649.", 10),
        blk("Appendix A", 12, bold=True),
    ]
    assert biblio_skips(acl) == [False, False, True, True, False]
    larger = [
        *[blk("Body text of the paper.", 10) for _ in range(5)],
        blk("References", 13, bold=True),
        blk(
            "E. Akyurek, M. Damani, L. Qiu, H. Guo, Y. Kim, and J. Andreas. The surprising effectiveness. arXiv, 2024.",
            10.8,
        ),
        blk("Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort", 11),
        blk("H. Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https:", 11),
        blk("//github.com/huggingface/open-r1.", 12),
        blk("1b2feafb99a24-Abstract-Datasets_and_Benchmarks_Track.html.", 12),
        blk("D. Ganguli, L. Lovitt, J. Kernion. Red teaming language models. arXiv, 2022.", 11),
    ]
    assert biblio_skips(larger)[6:] == [True] * 6


def test_pdf_biblio_skips_on_attention_paper():
    """arXiv-style citations (year at the line end without parentheses) are recognized too: the whole References
    section of attention.pdf is skipped."""
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_extract import extract_blocks

    blocks = extract_blocks(pymupdf.open(src))
    skips = biblio_skips(blocks)
    refs = next(i for i, b in enumerate(blocks) if b.text.strip() == "References")
    viz = next(i for i, b in enumerate(blocks) if b.text.strip() == "Attention Visualizations")
    assert skips.index(True) == refs + 1
    assert blocks[refs + 1].text.strip().startswith("[")
    assert all(skips[refs + 1 : viz])
    assert not skips[viz]
    assert not any(skips[viz + 1 :])


def test_pdf_biblio_entry_with_bold_subheadings():
    """Bold subheadings (Books/Articles) under References followed by citations also enter the references section."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000", bold=bold
        )

    blocks = [
        blk("Body paragraph.", 10),
        blk("References", 14, bold=True),
        blk("Books", 11, bold=True),
        blk("Smith, J. (2020). Some book title here. Press.", 10),
        blk("Articles", 11, bold=True),
        blk("Doe, A. (2019). Some article title. Journal.", 10),
        blk("Index", 14, bold=True),
        blk("Index entry.", 10),
    ]
    assert biblio_skips(blocks) == [False, False, False, True, False, True, False, False]


def test_pdf_biblio_section_title_stops_lookahead():
    """A larger References must not enter through Endnotes, otherwise the next heading at body size cannot end the
    section."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("Chapter body text.", 10),
        blk("Another body line.", 10),
        blk("References", 16, bold=True),
        blk("This paragraph is still the chapter.", 10),
        blk("Endnotes", 12, bold=True),
        blk("See Smith et al. (2018). A cited work.", 10),
        blk("Acknowledgements", 10.1, bold=True),
        blk("Thanks to the reviewers.", 10),
    ]
    assert biblio_skips(blocks) == [False, False, False, False, False, True, False, False]


def test_pdf_biblio_journal_line_does_not_end_numbered_entry():
    """Journal on a line of its own inside a citation is no subheading, and the lowercase continuation lines after it
    stay in the references section."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("Intro paragraph.", 11),
        blk("References", 12, bold=True),
        blk("[2] Dzmitry Bahdanau and Yoshua Bengio. Neural machine translation by jointly", 12),
        blk("Journal", 12),
        blk("of medicine and was widely cited afterward.", 12),
        blk("Smith, J. (2019). Another paper.", 12),
        blk("Index", 12, bold=True),
        blk("Index entry text.", 11),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True, False, False]


def test_pdf_biblio_chapter_titled_books_or_paper_does_not_enter():
    """Books and "论文" at the same size or larger are the next chapter. A smaller Paper is still an inner
    subheading."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    books = [
        blk("Body paragraph.", 10),
        blk("References", 14, bold=True),
        blk("Books", 16, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Chapter continues here.", 10),
    ]
    assert biblio_skips(books) == [False, False, False, False, False]
    thesis = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("论文", 12, bold=True),
        blk("这项研究发表于某出版社。", 11),
        blk("后续正文还在这里。", 11),
    ]
    assert biblio_skips(thesis) == [False, False, False, False, False]
    paper = [
        blk("Body paragraph.", 10),
        blk("References", 14, bold=True),
        blk("Paper", 11, bold=True),
        blk("Smith, J. (2020). Some book.", 10),
        blk("Index", 14, bold=True),
        blk("Index entry.", 10),
    ]
    assert biblio_skips(paper) == [False, False, False, True, False, False]


def test_pdf_biblio_chinese_gbt_citations():
    """Chinese GB/T citations ([J][M] type markers): a smaller "论文" is a group heading, the citations are skipped,
    and body text mentioning a publisher does not enter the references section."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("论文", 12, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("李四. 某书[M]. 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(blocks) == [False, False, False, True, True, False, False]


def test_pdf_biblio_type_mark_does_not_swallow_headings():
    """Inline [J]/[M] is no citation. A next chapter heading containing a marker ends the section, and the heading
    itself is not skipped."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    prose = [
        blk("Chapter body text.", 10),
        blk("References", 16, bold=True),
        blk("期刊论文的类型标识为[J]，专著为[M]。", 10),
        blk("See note [J]", 10),
        blk("The concentration [M] was measured carefully.", 10),
        blk("Endnotes", 12, bold=True),
        blk("See Smith et al. (2018). A cited work.", 10),
        blk("Acknowledgements", 10.1, bold=True),
        blk("Thanks to the reviewers.", 10),
    ]
    assert biblio_skips(prose) == [False, False, False, False, False, False, True, False, False]
    appendix = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("Appendix [C]", 14, bold=True),
        blk("Appendix paragraph.", 11),
    ]
    assert biblio_skips(appendix) == [False, False, True, False, False]
    cn = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("附录[J]", 14, bold=True),
        blk("附录正文。", 11),
    ]
    assert biblio_skips(cn) == [False, False, True, False, False]
    larger = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("第2章 [M]", 16, bold=True),
        blk("章正文。", 11),
    ]
    assert biblio_skips(larger) == [False, False, True, False, False]


def test_pdf_biblio_numbered_publisher_entry_without_space():
    """No space after [1], or the publisher in the next block, still enters. A publisher in unnumbered body text does
    not."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    tight = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1]张三. 书名. 北京: 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(tight) == [False, False, True, False, False]
    split = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1] 张三. 书名.", 11),
        blk("北京: 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(split) == [False, False, True, True, False, False]
    prose = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("这项研究发表于某出版社。", 11),
        blk("后续正文还在这里。", 11),
    ]
    assert biblio_skips(prose) == [False, False, False, False]


def test_pdf_biblio_edition_heading_exits():
    """A short numbered heading ending with an edition year ("年版") is the next chapter, and the heading itself is not
    skipped. Real bibliographic lines are still skipped."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    year = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报, 2020, 3(2): 1-10.", 11),
        blk("1. 2019年版", 14, bold=True),
        blk("后续正文。", 11),
    ]
    assert biblio_skips(year) == [False, False, True, False, False]
    press = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1]张三. 书名. 人民出版社, 2019年版.", 11),
        blk("1. 人民出版社2019年版", 14, bold=True),
        blk("后续正文。", 11),
    ]
    assert biblio_skips(press) == [False, False, True, False, False]


def test_pdf_biblio_type_mark_in_prose_does_not_enter():
    """A "[M]。" in the middle of a body sentence is no citation (no year), so the body text after it must not keep
    the original text."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("Body text.", 11),
        blk("References", 14, bold=True),
        blk("这个概念的出处见某书[M]。后续还有讨论。", 11),
        blk("正文继续。", 11),
    ]
    assert biblio_skips(blocks) == [False, False, False, False]
    later = [
        blk("Body text.", 11),
        blk("References", 14, bold=True),
        blk("这个概念的出处见某书[M]。后续还有讨论。", 11),
        blk("2020年又有新的讨论。", 11),
        blk("正文继续。", 11),
    ]
    assert biblio_skips(later) == [False, False, False, False, False]


def test_pdf_biblio_type_mark_year_on_next_block_enters():
    """A type marker and year split apart still enter. The next block must be a volume/issue or a publisher, not any
    following text."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    volume = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某某研究[J]. 某某学报", 11),
        blk("2020, 3(2): 1-10.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(volume) == [False, False, True, True, False, False]
    numbered = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("[1]张三. 某某研究[J]. 某某学报", 11),
        blk("2020, 3(2): 1-10.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(numbered) == [False, False, True, True, False, False]
    press = [
        blk("这是正文段落。", 11),
        blk("参考文献", 14, bold=True),
        blk("张三. 某书[M]. 书名", 11),
        blk("北京: 人民出版社, 2019.", 11),
        blk("Index", 14, bold=True),
        blk("索引条目。", 11),
    ]
    assert biblio_skips(press) == [False, False, True, True, False, False]


def test_pdf_biblio_year_continuation_tightened():
    """Confirming a year on a split line must not be loose enough to accept body text: an English sentence starting
    with a year and body text mentioning a publisher and a year are not continuation lines."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    def scenario(second: str):
        return biblio_skips(
            [
                blk("参考文献", 14, bold=True),
                blk("见某书[M]。如下。", 11),
                blk(second, 11),
            ]
        )

    assert scenario("2020, we showed more.") == [False, False, False]
    assert scenario("2020, volume were higher than expected.") == [False, False, False]
    assert scenario("2020, Volume of the series was larger.") == [False, False, False]
    assert scenario("这项研究2020年发表于某出版社。") == [False, False, False]
    assert scenario("人民出版社, 2019年出版.") == [False, False, False]
    # Volume/issue pages, "第N期", and publisher lines ending with "年" or a Chinese full stop still count
    assert scenario("2020, 3(2): 1-10.") == [False, True, True]
    assert scenario("2020, Vol. 3") == [False, True, True]
    assert scenario("2020, Volume 3") == [False, True, True]
    assert scenario("2020, volume. 12, no. 2") == [False, True, True]
    assert scenario("2020, pp. 1-10.") == [False, True, True]
    assert scenario("2020, 第3期") == [False, True, True]
    assert scenario("北京: 人民出版社, 2019.") == [False, True, True]
    assert scenario("北京: 人民出版社, 2019年.") == [False, True, True]
    assert scenario("北京: 人民出版社, 2019。") == [False, True, True]
    assert scenario("北京: 人民出版社2019年.") == [False, True, True]


def test_pdf_biblio_vol_abbreviation_also_needs_digit():
    """The abbreviation Vol works like the full Volume: a number must follow, so "2020, Vol were…" is body text."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    def scenario(second: str):
        return biblio_skips(
            [
                blk("参考文献", 14, bold=True),
                blk("见某书[M]。如下。", 11),
                blk(second, 11),
            ]
        )

    assert scenario("2020, Vol were higher than expected.") == [False, False, False]
    assert scenario("2020, Vol of the series was larger.") == [False, False, False]
    assert scenario("2020, Vol. mix") == [False, False, False]
    assert scenario("2020, Vol. mill") == [False, False, False]
    assert scenario("2020, Vol. 3") == [False, True, True]
    assert scenario("2020, Vol 3") == [False, True, True]
    assert scenario("2020, Vol. III") == [False, True, True]
    assert scenario("2020, Vol. IV, pp. 1-10.") == [False, True, True]
    assert scenario("2020, Vol. xii, pp. 1-10.") == [False, True, True]


def test_pdf_biblio_volume_lines_protected_at_same_size():
    """When the references heading has the body size, volume/issue and publisher continuation lines must not be taken
    for the next chapter either."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    blocks = [
        blk("Body text.", 12),
        blk("参考文献", 12, bold=True),
        blk("张三. 某某研究[J]. 某某学报", 12),
        blk("2020, 3(2): 1-10.", 12),
        blk("李四. 某书[M]. 书名", 12),
        blk("2020, Vol. III", 12),
        blk("王五. 另一本书[M]. 书名", 12),
        blk("北京: 人民出版社, 2019年。", 12),
        blk("Index", 12, bold=True),
        blk("索引条目。", 12),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True, True, True, False, False]


def test_pdf_biblio_year_prefixed_chapter_still_exits():
    """A next chapter starting with a year is no volume/issue continuation line. A long bold heading ends the section
    too."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    def scenario(line: str, *, bold: bool = False) -> list[bool]:
        body = (
            "这是后续正文，需要被翻译，不能因为上一行像年份就留在参考文献里。"
            "这里再补上一整句，让整段明显超过六十个字，避免它本身被当成同字号短标题。"
        )
        return biblio_skips(
            [
                blk("Body text.", 12),
                blk("References", 12, bold=True),
                blk("Smith, J. (2020). Some book.", 12),
                blk(line, 12, bold=bold),
                blk(body, 12),
            ]
        )

    assert scenario("2020, 1. Introduction") == [False, False, True, False, False]
    assert scenario("2020, 3. Results") == [False, False, True, False, False]
    assert scenario("2020, 第1章 结论") == [False, False, True, False, False]
    assert scenario("2020, No further reading") == [False, False, True, False, False]
    assert scenario("2020, p < 0.05.") == [False, False, True, False, False]
    assert scenario("2020, 3 days later.") == [False, False, True, False, False]
    assert scenario("2020, 1. Introduction to machine translation systems", bold=True) == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert scenario("2020, 3(2): 1-10.") == [False, False, True, True, True]
    assert scenario("2020, Vol. III") == [False, False, True, True, True]


def test_pdf_biblio_long_bold_chapter_title_exits():
    """A bold chapter heading longer than 80 characters also ends the section (the bold branch of heading_like has an
    80-character limit and cannot be relied on)."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    body = "这是后续正文，需要被翻译，这里再补上一整句，让整段明显超过六十个字，避免被当成短标题。"
    blocks = [
        blk("Body text.", 12),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 12),
        blk("Chapter 12: On the relationship between government forms and the laws that govern them", 12, bold=True),
        blk(body, 12),
    ]
    assert biblio_skips(blocks) == [False, False, True, False, False]


def test_pdf_biblio_long_bold_citation_stays():
    """A bold bibliographic line longer than 80 characters is not the next chapter. Lines with a type marker or a year
    at the end are still skipped."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size, bold=False):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=bold
        )

    gbt = (
        "张三. 某某研究的题目在这里继续写长，把期刊论文的背景、方法和结论都写进题名，"
        "直到这一行确实超过八十个字。[J]. 某某学报, 2020, 3(2): 1-10."
    )
    smith = "Smith, J. A long paper title goes here and continues past eighty characters. 2019."
    assert len(gbt) > 80 and len(smith) > 80
    body = "Jones, K. (2018). Another citation that must stay untranslated."
    blocks = [
        blk("Body text.", 12),
        blk("References", 12, bold=True),
        blk("Smith, J. (2020). Some book.", 12),
        blk(gbt, 12, bold=True),
        blk(smith, 12, bold=True),
        blk(body, 12),
        blk("Index", 12, bold=True),
        blk("Index entry text.", 12),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True, False, False]


def test_blocks_inside_a_display_formula_box_are_kept():
    """Blocks the formula detection model boxes as a display formula keep the original text (s.t., otherwise and
    \\text{...} in an align). A block touching the box with only one corner is translated; an explanatory sentence
    inside the box (with three function words) and a heading are translated too"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_roles import Role
    from app.formats.pdf_skip import _isolated_formula_ids

    def block(text, rect, page=0):
        rect = pymupdf.Rect(rect)
        spans = [{"text": text, "font": "TimesNewRomanPSMT", "size": 10.0, "flags": 0}]
        return TextBlock(
            page=page, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False, span_lines=[spans]
        )

    inside = block("otherwise,", (300, 210, 340, 222))
    # A word outside the lower right corner of the formula touches the box with only one corner
    below = block("and", (390, 230, 450, 242))
    other_page = block("s.t.", (300, 210, 340, 222), page=1)

    class Detector:
        isolated = {0: [pymupdf.Rect(80, 200, 400, 236)]}

    found = _isolated_formula_ids([inside, below, other_page], Detector())
    assert found == {id(inside)}
    assert _isolated_formula_ids([inside], None) == set()
    note = block("(where c is the correct class)", (300, 210, 390, 222))
    title = block("Phoenix + Calcite", (100, 205, 380, 230))
    # perplexity, red and blue in the formula are content words without function words, so this is no sentence
    words = block("perplexityA(T) = PA(red red red red blue)−1/5", (100, 205, 380, 225))
    blocks = [note, title, inside, words]
    found = _isolated_formula_ids(blocks, Detector(), [Role.BODY, Role.HEADING, Role.BODY, Role.BODY])
    assert found == {id(inside), id(words)}


def test_a_concordance_with_aligned_bold_keywords_stays_original():
    """Search results with one bold word in the middle of each line, aligned left edges and cut-off context on both
    sides keep the original text; scattered bold terms in body text do not count."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _concordance_ids

    def span(text, x0, x1, y, bold=False):
        return {
            "text": text,
            "font": "NimbusRomNo9L-Medi" if bold else "NimbusRomNo9L-Regu",
            "size": 10.0,
            "flags": 16 if bold else 0,
            "bbox": (x0, y - 8, x1, y + 2),
            "origin": (x0, y),
        }

    def blk(span_lines):
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(161, 108, 492, 157),
            line_rects=[],
            text="",
            size=10,
            color="#000",
            bold=False,
            span_lines=span_lines,
        )

    concordance = blk(
        [
            [span("is traditionally followed by", 191.0, 298.9, 117.5), span("cherry", 304.9, 333.1, 117.5, True)],
            [span("pie, a traditional dessert", 361.6, 456.7, 117.5)],
            [span("often mixed, such as", 217.2, 298.9, 129.5), span("strawberry", 304.9, 352.5, 129.5, True)],
            [span("rhubarb pie. Apple pie", 361.6, 451.8, 129.5)],
            [
                span("computer peripherals and personal", 161.4, 298.9, 141.4),
                span("digital", 304.9, 332.0, 141.4, True),
            ],
            [span("assistants. These devices usually", 361.6, 492.0, 141.4)],
        ]
    )
    prose = blk(
        [
            [
                span("we call this a", 161.4, 230.0, 117.5),
                span("vector", 233.0, 260.0, 117.5, True),
                span("and it is", 263.0, 300.0, 117.5),
            ],
            [
                span("a list of numbers, the", 161.4, 260.0, 129.5),
                span("dimension", 263.0, 310.0, 129.5, True),
                span("of which is", 313.0, 360.0, 129.5),
            ],
            [
                span("the length of the", 161.4, 240.0, 141.4),
                span("vector space", 243.0, 300.0, 141.4, True),
                span("below", 303.0, 330.0, 141.4),
            ],
        ]
    )
    assert _concordance_ids([concordance, prose]) == {id(concordance)}


def test_pdf_biblio_bare_label_line_does_not_exit():
    """With a hanging indent the label is far from its entry, so [ESS] forms a block of its own. It is the label of the
    entry, and the citation after it is still skipped."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size=10.7):
        return TextBlock(
            page=0, rect=pymupdf.Rect(0, 0, 1, 1), line_rects=[], text=text, size=size, color="#000000", bold=False
        )

    blocks = [
        blk("We thank the reviewers for help with the experiments."),
        blk("References", 14.3),
        blk("S. Chaudhuri and K. Shim. Including Group-By in Query Optimization. In VLDB, pages 354-366, 1994."),
        blk("[ESS]"),
        blk("Goetz Graefe. Query Evaluation Techniques for Large Databases. ACM Computing Surveys, 1993."),
        blk("[GBLP95] J. Gray, A. Bosworth. Data Cube: A Relational Aggregation Operator. MSR-TR-95-22, 1995."),
    ]
    assert biblio_skips(blocks) == [False, False, True, True, True, True]


def test_bibliography_without_a_heading_stays_original():
    """References at a chapter end follow the exercises directly, without a References heading. A run of entries
    starting with "Surname, Initials." keeps the original text; a single body sentence starting this way does not
    count."""
    from app.formats.pdf_biblio import biblio_skips
    from app.formats.pdf_blocks import TextBlock

    def blk(text, size=10.0, page=0):
        rect = pymupdf.Rect(40, 40, 300, 50)
        return TextBlock(page=page, rect=rect, line_rects=[rect], text=text, size=size, color="#000", bold=False)

    blocks = [
        blk("Smith, J. 2020 showed that this works, and we follow that approach here."),
        blk("2.10 Augment the minimum edit distance algorithm to output an alignment."),
        blk("Ahia, O., S. Kumar, H. Gonen, J. Kasai, D. Mortensen,", 8, 1),
        blk("N. Smith, and Y. Tsvetkov. 2023. Do all languages cost the same? EMNLP.", 8, 1),
        blk("Arkadiev, P. M. 2020. Morphology in typology. Oxford.", 8, 1),
        blk("Baayen, R. H. 2001. Word frequency distributions. Springer.", 8, 1),
        blk("Exercises", 10, 1),
        blk("Bellman, R. 1984. Eye of the Hurricane: an autobiography.", 8, 1),
        blk("World Scientific Singapore.", 8, 1),
        blk("Clark, H. H. and J. E. Fox Tree. 2002. Using uh and um in spontaneous speaking.", 8, 1),
        blk("Weizenbaum, J. 1976. Computer Power and Human Reason.", 8, 1),
        blk("From Judgement to Calculation. W.H. Freeman & Co.", 8, 1),
    ]
    assert biblio_skips(blocks) == [False, False, *([True] * 10)]


def test_shaded_panel_of_monospace_text_is_kept_whole():
    """An area of monospace text on a background (code point tables, code) keeps the original glyphs. txtt has no
    Mono in its name and is recognized by the monospace flag. Text without a background, or with body font mixed
    in on the background, is translated; Chinese characters carry the monospace flag too and do not count."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_layout import PageScan
    from app.formats.pdf_skip import _code_panel_ids

    def block(y, *spans):
        rect = pymupdf.Rect(160, y, 380, y + 10)
        span_lines = [
            [
                {"text": text, "font": font, "flags": flags, "size": 10.0, "bbox": tuple(rect)}
                for text, font, flags in spans
            ]
        ]
        return TextBlock(0, rect, [rect], "".join(text for text, _f, _g in spans), 10.0, "#000000", False, span_lines)

    table = [block(372, ("U+0061 a LATIN SMALL LETTER A", "txtt", 8)), block(384, ("GRINNING FACE", "txtt", 8))]
    plain = block(600, ("Men are all alike.", "txtt", 8))
    mixed = [
        block(452, ("/woodchuck/", "txtt", 8)),
        block(464, ("interesting links to woodchucks", "NimbusRomNo9L-Regu", 4)),
    ]
    chinese = block(532, ("进远违连", "gbsnu8f", 8))
    shades = (pymupdf.Rect(157, 370, 411, 396), pymupdf.Rect(157, 450, 411, 476), pymupdf.Rect(157, 530, 411, 544))
    found = _code_panel_ids([*table, plain, *mixed, chinese], [PageScan((), (), (), shades)])
    assert found == {id(b) for b in table}


def test_numbered_and_bracket_labeled_examples_stay_original():
    """Examples starting with a number whose continuation lines hang right of the number, and examples with [subscript
    …] bracket annotation, keep the original text. A paragraph starting with an equation number whose continuation
    lines return to the column edge, the layer superscript W[1], and [EOS] with a single word in the brackets are
    translated."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _example_ids

    def blk(spans, rows):
        rects = [pymupdf.Rect(row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        text = "".join(text for text, _size in spans)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=rects,
            text=text,
            size=10.0,
            color="#000000",
            bold=False,
            span_lines=[[{"text": text, "size": size} for text, size in spans]],
        )

    example = blk(
        [
            ("(24.3) AMD agreed to [buy] Markham, Ontario-based ATI for around $5.4 billion", 10.0),
            (" in cash and stock, the companies announced Monday.", 10.0),
        ],
        [(160, 491, 490, 504), (185, 504, 399, 516)],
    )
    reference = blk(
        [
            ("(2.4) follows from the definition of the loss and the bound on the", 10.0),
            (" gradient given above.", 10.0),
        ],
        [(160, 491, 490, 504), (160, 504, 300, 516)],
    )
    labeled = blk(
        [
            ("[", 10.0),
            ("Arg0", 7.0),
            (" The group] ", 10.0),
            ("agreed", 10.0),
            (" [", 10.0),
            ("Arg1", 7.0),
            (" it wouldn’t make an offer].", 10.0),
        ],
        [(185, 300, 400, 312)],
    )
    layers = blk(
        [
            ("We use W", 10.0),
            ("[", 7.0),
            ("1", 7.0),
            ("] and W", 10.0),
            ("[", 7.0),
            ("2", 7.0),
            ("] for the weights.", 10.0),
        ],
        [(160, 300, 400, 312)],
    )
    tokens = blk(
        [("Then we append an [", 10.0), ("EOS", 7.0), ("] and [", 10.0), ("BOS", 7.0), ("] token.", 10.0)],
        [(160, 300, 400, 312)],
    )
    found = _example_ids([example, reference, labeled, layers, tokens])
    assert found == {id(example), id(labeled)}


def test_the_rest_of_a_numbered_example_stays_original():
    """The second sentence of a numbered example, the wrapped wine. and the IT gloss became separate blocks: a block
    right below and aligned with the sentence after the number is part of the example too. Body text back at the
    left edge of the number, and a first line with an indent cut out by a side note, are translated."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _example_ids

    def blk(text, rect, label=""):
        rect = pymupdf.Rect(rect)
        spans = [{"text": text, "bbox": tuple(rect), "size": 10.0}]
        if label:
            cut = rect.x0 + 30
            spans = [
                {"text": label, "bbox": (rect.x0, rect.y0, cut, rect.y1), "size": 10.0},
                {"text": text[len(label) + 1 :], "bbox": (cut + 2, rect.y0, rect.x1, rect.y1), "size": 10.0},
            ]
        return TextBlock(
            page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False, span_lines=[spans]
        )

    head = blk("(24.14) I just bought a copy of Thoreau’s Walden.", (160.2, 222.1, 491.1, 234.1), "(24.14)")
    second = blk("That one had been very tattered.", (192.1, 234.1, 474.9, 246.1))
    prose = blk("Note that this NP is ambiguous.", (160.2, 250.8, 493.2, 274.8))
    other = blk("(19.2) We saw the Eiffel Tower flying to Paris.", (142.2, 304.6, 329.9, 316.6), "(19.2)")
    first_line = blk("In coordination ambiguity phrases can be conjoined", (157.1, 320.1, 475.2, 333.0))
    found = _example_ids([head, second, prose, other, first_line])
    assert id(second) in found and id(prose) not in found and id(first_line) not in found


def test_sentences_after_a_separate_example_number_stay_original():
    """The number (22.9) has few letters and did not get into a block during extraction, so the following a. Shelly
    cut … became a block of its own: a block starting on the same line right of the number is an example, and so
    are the aligned b. below and the free translation below a word-by-word gloss. The next body paragraph and body
    sentences referring to the number are translated."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_glosses import ExampleNumber
    from app.formats.pdf_skip import _example_ids

    def blk(text, rect):
        rect = pymupdf.Rect(rect)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    first = blk("a. Shelly cut the banana with a knife.", (190.1, 352.2, 340.0, 364.2))
    second = blk("b. The knife cut the banana.", (189.9, 364.7, 303.0, 376.7))
    prose = blk("In addition to the fragmentation problem, there are cases", (157.1, 408.6, 475.0, 420.6))
    glossed = blk("姚明进入总决赛", (190.0, 380.0, 280.0, 392.0))
    free = blk("“Yao Ming reaches the finals”", (190.0, 392.5, 300.0, 404.5))
    cited = blk("as shown in (22.9) the instrument can be the subject.", (142.2, 440.0, 400.0, 452.0))
    numbers = {
        0: (
            ExampleNumber(pymupdf.Rect(142.2, 352.2, 166.3, 364.2), pymupdf.Rect(142.2, 352.2, 340.0, 364.2), 190.1),
            ExampleNumber(pymupdf.Rect(160.0, 380.0, 180.0, 392.0), pymupdf.Rect(160.0, 380.0, 280.0, 392.0), 190.0),
        )
    }
    found = _example_ids([first, second, glossed, free, prose, cited], numbers)
    assert found == {id(first), id(second), id(glossed), id(free)}


def test_lines_under_a_number_row_without_a_block_stay_original():
    """The line (22.12) fall.01 has few letters and formed no block: Arg1: … indented below the number and Ex1:
    aligned after a blank line are examples. The next paragraph starting with a first-line indent does not
    count"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_glosses import ExampleNumber
    from app.formats.pdf_skip import _example_ids, _example_tails, _ExampleHead

    def blk(text, rect):
        rect = pymupdf.Rect(rect)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10.0, color="#000", bold=False)

    roles = [
        blk("Arg1: Logical subject, patient, thing falling", (188.1, 466.5, 364.0, 478.5)),
        blk("Arg2: Extent, amount fallen", (188.1, 478.5, 303.0, 490.5)),
    ]
    label = blk("Ex1:", (188.1, 500.0, 207.0, 512.0))
    note = blk("Note that there is no Arg0 role for fall.", (175.1, 514.0, 400.0, 526.0))
    body = blk(
        "Here the body text goes on across the whole column with no indent at all.", (160.2, 540.0, 493.0, 564.0)
    )
    numbers = {
        0: (
            ExampleNumber(pymupdf.Rect(160.2, 455.4, 189.3, 467.4), pymupdf.Rect(160.2, 455.4, 221.0, 467.4), 194.2),
            ExampleNumber(pymupdf.Rect(160.2, 596.4, 189.3, 608.4), pymupdf.Rect(160.2, 596.4, 330.0, 608.4), 194.2),
        )
    }
    found = _example_ids([*roles, label, note, body], numbers)
    assert found == {id(block) for block in (*roles, label)}

    # The number (5.6) is short and the sentence starts at 185; body paragraphs are indented by 15, so a one-line
    # paragraph starts at 175
    short = ExampleNumber(pymupdf.Rect(160.2, 377.0, 179.0, 389.0), pymupdf.Rect(160.2, 377.0, 361.0, 389.0), 185.0)
    head = _ExampleHead(0, short.row, short.rect.x0, short.text_x0, 0.0, None)
    paragraph = blk(
        "And suppose that you had seen many of these context words in other contexts:", (175.1, 393.0, 493.0, 405.0)
    )
    assert _example_tails([paragraph], [head], {0: (short,)}, indent=15.0) == set()


def test_text_annotated_with_arcs_stays_original():
    """Dependency arcs connect words, with labels (nsubj) in rounded boxes on the arcs. Once the sentence is
    translated into Chinese the arcs no longer match the words, so the sentence and labels keep the original text.
    A line in a chart that is high at one end and low at the other is no arc, and the body text beside it is
    translated."""
    from app.formats import pdf_extract, pdf_skip

    doc = pymupdf.open()
    page = doc.new_page(width=500, height=700)
    page.insert_text((238, 395), "She announced this in January", fontsize=10)
    page.draw_bezier((275, 386), (275, 372), (246, 372), (246, 384), width=0.4)
    shape = page.new_shape()
    shape.draw_rect(pymupdf.Rect(251, 367, 270, 376), radius=0.3)
    shape.finish(color=(0, 0, 0), fill=(1, 1, 1), width=0.4)
    shape.commit()
    page.insert_text((253, 374), "nsubj", fontsize=7)
    page.draw_bezier((300, 470), (330, 420), (360, 440), (400, 430), width=0.4)
    page.insert_text((300, 480), "The loss falls quickly as training goes on.", fontsize=10)
    scan = pdf_extract.scan_page(page)
    assert len(scan.arcs) == 1 and len(scan.round_boxes) == 1
    blocks, _frames, scans = pdf_extract._extract(doc)
    found = pdf_skip._arc_figure_ids(blocks, scans)
    kept = {block.text for block in blocks if id(block) in found}
    assert kept == {"She announced this in January", "nsubj"}


def test_role_pattern_lines_keep_the_original():
    """Role notation listed line by line (AGENT/Subject, THEME/Object) keeps the original text; a body sentence
    mentioning AGENT is translated"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _role_pattern_ids

    def block(text: str) -> TextBlock:
        rect = pymupdf.Rect(189, 510, 417, 522)
        return TextBlock(page=0, rect=rect, line_rects=[rect], text=text, size=10, color="#000000", bold=False)

    patterns = [
        block("AGENT/Subject, THEME/Object"),
        block("AGENT/Subject, THEME/Object, INSTRUMENT/PPwith"),
        block("THEME/Subject"),
    ]
    prose = block("These examples suggest that break has AGENT, THEME, and INSTRUMENT.")
    found = _role_pattern_ids([*patterns, prose])
    assert found == {id(b) for b in patterns}


def test_a_displayed_example_line_keeps_the_original():
    """In a book with first-line indents, a line set off in the middle of body text and indented deeper than a
    paragraph start is an example and keeps the original text. Lead-in sentences ending with a colon, sources one
    size smaller, centered captions, wrapped lines of list items, "label: explanation" lines, formula descriptions
    with an equals sign, and free translations below word-by-word glosses do not count"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_frame import TextFrame
    from app.formats.pdf_layout import Column, PageGeometry
    from app.formats.pdf_skip import _display_example_ids

    def blk(rows, text, size=10.0):
        rects = [pymupdf.Rect(*row) for row in rows]
        rect = pymupdf.Rect(rects[0])
        for row in rects[1:]:
            rect |= row
        return TextBlock(
            page=0, rect=rect, line_rects=rects, text=text, size=size, color="#000000", bold=False, body_size=10.0
        )

    def para(y, text="For example WordNet has the is-a or hypernym relation between classes."):
        return blk([(175, y, 493, y + 12), (160, y + 12, 400, y + 24)], text)

    def line(y, x0, x1, text, size=10.0):
        return blk([(x0, y, x1, y + 12)], text, size)

    geo = PageGeometry(612, (Column(160, 493),), (), 792, frame=TextFrame(160, 493, 109, 718, 10))
    blocks = [para(110), para(134), para(158)]
    example = line(184, 185, 446, "Giraffe is-a ruminant is-a ungulate is-a mammal is-a vertebrate ...")
    lead = line(224, 185, 400, "Then repeat iteratively until convergence:")
    source = line(264, 185, 470, "The seven circumstances, associated with Hermagoras and Aristotle", size=8.0)
    caption = line(304, 200, 453, "Listing 3 | Prompt for a simple math problem.")
    item = line(344, 175, 480, "a. Count the parameters in one attention layer.")
    folded = line(356, 188, 300, "Does your answer depend on A?")
    gloss = line(396, 188, 300, "Arg1: Proposition")
    hanzi = line(436, 188, 260, "姚明进入总决赛")
    pinyin = line(436, 290, 400, "yáo míng jìn rù zǒng jué sài")
    free = line(448, 188, 320, "“Yao Ming reaches the finals”")
    formula = line(488, 200, 440, "convolution with width-1 kernel: z = x w")
    blocks += [
        example,
        para(198),
        lead,
        para(238),
        source,
        para(278),
        caption,
        para(318),
        item,
        folded,
        para(370),
        gloss,
        para(410),
        hanzi,
        pinyin,
        free,
        para(462),
        formula,
        para(502),
    ]
    found = _display_example_ids(blocks, [geo], set())
    assert id(example) in found
    for name, block in (
        ("lead", lead),
        ("source", source),
        ("caption", caption),
        ("folded", folded),
        ("item", item),
        ("gloss", gloss),
        ("free", free),
        ("formula", formula),
    ):
        assert id(block) not in found, name
    assert not found & {id(block) for block in blocks if len(block.line_rects) > 1}


def test_patterns_keyed_to_examples_by_color_stay_original():
    """The line below a pattern is an example that marks the matching parts with the same colors in the same order
    (PER and George Marshall are both blue). A translation would change the word order and break the color
    match, so both lines keep the original text. Lines with a different color order, or with only one or two
    colors, are translated"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _color_keyed_ids

    blue, green, red = 0x0000FF, 0x008000, 0xB30000

    def blk(parts, y0):
        rect = pymupdf.Rect(178, y0, 400, y0 + 12)
        text = "".join(text for text, _color in parts)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=10.0,
            color="#000000",
            bold=False,
            span_lines=[[{"text": text, "size": 10.0, "color": color} for text, color in parts]],
        )

    pattern = blk([("PER,", blue), (" POSITION", green), (" of", blue), (" ORG", red), (":", blue)], 268)
    example = blk(
        [("George Marshall", blue), (",", 0), (" Secretary of State", green), (" of", 0), (" the United States", red)],
        280,
    )
    shuffled = blk([("Truman", green), (" appointed", 0), (" Marshall", blue), (" Secretary", red)], 330)
    linked = blk([("see", 0), (" Hearst", blue), (" and", 0), (" Fig. 21.4", red)], 342)
    assert _color_keyed_ids([pattern, example, shuffled, linked]) == {id(pattern), id(example)}


def test_colored_lines_under_a_bullet_stay_original():
    """An example listed on its own line below a bullet item, with mostly colored letters (M2: -1 spokesman), is made
    of words taken from an example sentence and keeps the original text; several consecutive lines under the same
    item all count. Black explanatory text and colored lines not below an item are translated"""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _item_example_ids

    def blk(parts, rect):
        rect = pymupdf.Rect(rect)
        text = "".join(text for text, _color in parts)
        return TextBlock(
            page=0,
            rect=rect,
            line_rects=[rect],
            text=text,
            size=10.0,
            color="#000000",
            bold=False,
            span_lines=[[{"text": text, "size": 10.0, "color": color} for text, color in parts]],
        )

    blue = 0x0000FF
    bullet = blk([("• Words or bigrams in particular positions", 0)], (159, 260, 328, 272))
    first = blk([("M2:", 0), (" -1 spokesman", blue)], (167, 272, 242, 284))
    second = blk([("M2:", 0), (" +1 said", blue)], (167, 284, 216, 296))
    note = blk([("• Number of entities between the arguments", 0)], (159, 404, 439, 416))
    plain = blk([("in this case one, for AMR", 0)], (167, 416, 300, 428))
    prose = blk([("The features are described in", 0)], (142, 500, 475, 512))
    loose = blk([("Zhang et al., 2017", blue)], (167, 512, 300, 524))
    found = _item_example_ids([bullet, first, second, note, plain, prose, loose])
    assert found == {id(first), id(second)}


def test_a_chinese_line_beside_its_english_rendering_is_kept_apart():
    """An epigraph in an English book with the Chinese original in the left column and the English translation in the
    right on the same line: the two pieces split, and the Chinese original keeps the original text. Western text
    following Chinese at normal character spacing is not split off."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_math import _script_seam
    from app.formats.pdf_skip import _foreign_ids

    def span(text, x0, x1):
        return {
            "text": text,
            "font": "Times",
            "size": 10.0,
            "flags": 0,
            "bbox": (x0, 206, x1, 216),
            "origin": (x0, 216),
        }

    assert _script_seam(
        [span("忘", 262.8, 272.8), span("荃", 272.8, 282.7)], [span("Nets are for fish;", 288.7, 355.4)]
    )
    assert not _script_seam([span("使用", 100, 120)], [span("Python", 122.5, 150)])

    def blk(text, y):
        return TextBlock(
            page=0, rect=pymupdf.Rect(160, y, 490, y + 10), line_rects=[], text=text, size=10, color="#000", bold=False
        )

    prose = [
        blk(text, 300 + 40 * i)
        for i, text in enumerate(
            [
                "The asphalt that Los Angeles is famous for occurs mainly on its freeways.",
                "In this chapter we introduce embeddings, vector representations of the meaning of words.",
                "The linguistic field that studies embeddings and their meanings is called vector semantics.",
            ]
        )
    ]
    quote = blk("荃者所以在鱼，得鱼而忘荃", 206)
    assert _foreign_ids([quote, *prose]) == {id(quote)}


def test_the_english_rendering_below_a_chinese_example_stays_original():
    """In an example box, the English translation right below the Chinese characters and pinyin is the author's own and
    keeps the original text; body text below it, after a gap and at a new left edge, is translated."""
    from app.formats.pdf_blocks import TextBlock
    from app.formats.pdf_skip import _foreign_ids

    def blk(text, x0, y0, y1, first=None):
        span = {"text": first or text, "font": "Times", "size": 10.0, "flags": 0}
        return TextBlock(
            page=0,
            rect=pymupdf.Rect(x0, y0, 470, y1),
            line_rects=[],
            text=text,
            size=10,
            color="#000",
            bold=False,
            span_lines=[[span]],
        )

    chinese = blk(
        "在我的后园，可以看见墙外有两株树，一株是 zài wǒ de hòu yuán",
        164,
        412,
        437,
        first="在我的后园，可以看见墙外有两株树，一株是",
    )
    english = blk("In my backyard, you can see two trees beyond the fence. One is a ...", 164, 436, 448)
    prose = [
        blk(text, 160, 454 + 40 * i, 490 + 40 * i)
        for i, text in enumerate(
            [
                "Here you might expect that a language model should assign a high probability to continuations.",
                "Fig. 1.2 shows this idea of a probability distribution more clearly than the text does.",
                "What good is it to assign probabilities to words? One intuition is that a model can predict text.",
            ]
        )
    ]
    found = _foreign_ids([chinese, english, *prose])
    assert id(english) in found and not any(id(block) in found for block in prose)
