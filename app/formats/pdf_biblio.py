"""Find the bibliography and reference entries of a document, which keep the original text."""

import re

from .pdf_blocks import (
    TextBlock,
)


def _median_size(blocks: list[TextBlock]) -> float:
    """Median font size weighted by character count. Counted per block, the hundreds of short blocks in small
    type in an index or a figure would pull it down."""
    total = sum(len(block.text) for block in blocks)
    seen = 0
    for block in sorted(blocks, key=lambda block: block.size):
        seen += len(block.text)
        if seen * 2 >= total:
            return block.size
    return 0.0


_BIBLIO_CITE_RE = re.compile(r"\((?:19|20)\d{2}\)|et al\.|doi:|https?://", re.I)


# A line ending in “, 2016.” is too common in body text. To enter the bibliography the line must also be a
# numbered entry or contain arXiv/CoRR/doi.
_BIBLIO_YEAR_END_RE = re.compile(r",\s*(?:19|20)\d{2}\s*[.]?\s*$")


# Older database papers end with page numbers: ICDE 1995, 190-200. / VLDB 2000: 496-505. Numbered entries only.
_BIBLIO_PAGES_END_RE = re.compile(r"(?:19|20)\d{2}\s*[,:]\s*(?:pp?\.\s*)?\d+\s*[-–]\s*\d+\s*[.]?\s*$")


_BIBLIO_TRAIL_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*[.]?\s*$|年版\s*[.。]?\s*$")


_BIBLIO_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


# Volume and issue continuation lines: 2020, 3(2) / Vol. 3 / Volume 3 / pp. 1-10 / 第3期.
# Volume must be followed by an Arabic numeral, to exclude “volume were”.
_BIBLIO_VOLUME_RE = re.compile(
    r"^(?:19|20)\d{2}\s*[,，(（]\s*(?:[\d(（]|[Vv]ol\.?\s*\d|[Vv]olume\.?\s*\d|[Nn]o\.?(?:\s|\d)|[Pp]p?\.?(?:\s|\d)|第)"
)


# Roman numerals only after the abbreviation Vol. M and D are excluded, or mix / mill would be taken for volumes.
_ROMAN_VOL_RE = re.compile(
    r"(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})(?![A-Za-z])",
    re.I,
)


_BIBLIO_VOL_LABEL_RE = re.compile(r"^(?:19|20)\d{2}\s*[,，(（]\s*[Vv]ol\.?\s*")


# Used when deciding exit at the same size. A number must be followed by an issue, pages or the line end; No/pp
# by a number; "第" by "期". The next chapter starting with just a year (2020, 1. Introduction / 第1章) must not
# be kept by this rule.
_BIBLIO_EXIT_VOLUME_RE = re.compile(
    r"^(?:19|20)\d{2}\s*[,，(（]\s*(?:"
    r"\d+\s*(?:[（(]|[:：]\s*\d|[,，]\s*\d|[.。]?\s*$)"
    r"|\d+\s*[）)]"
    r"|[Vv]ol\.?\s*\d|[Vv]olume\.?\s*\d|[Nn]o\.?\s*\d|[Pp]p?\.?\s*\d"
    r"|第\s*\d+\s*期"
    r")"
)


# A publisher line may end with 2019。 / 2019年。 / 年版。 Text continuing after the publisher is body text.
_BIBLIO_IMPRINT_END_RE = re.compile(r"(?:(?:19|20)\d{2}年?|年版)\s*[.。]?\s*$")


_BIBLIO_MARK_RE = re.compile(r"\barXiv\b|\bCoRR\b|doi:|https?://", re.I)


# Only the position after a title counts, as in 研究[J]. 刊名. A lone [M]。 at a sentence end is body text.
_BIBLIO_TYPE_MARK_RE = re.compile(r"(?<=\S)\[[JMDCNRP]\]\s*[.。]\s*\S")


# [1]张三 often has no space in between. 1. still needs a space so that 1.2 is not taken for a number.
# Older papers use labels of author initials and year: [CS94], [GBLP95], [TPCD]. OCR of bitmap fonts often
# loses the opening [, leaving ESS].
_BIBLIO_LEAD_RE = re.compile(r"^(?:\[\d+\]\s*|\d{1,3}[.)]\s+|\[?[A-Z][A-Za-z]{0,8}\+?\d{0,4}[a-z]?\](?:\s+|$))")


_BIBLIO_WRAP_RE = re.compile(r"^[a-zà-öø-ÿ]")


# Weak signal: counts only for numbered entries. “这项研究发表于某出版社。” is body text.
_BIBLIO_CN_PUB_RE = re.compile(r"出版社|年版")


# A list of author names: Y. Bai, A. Jones, K. Ndousse, ...; Alan Akbik, Duncan Blythe, and Roland Vollgraf.
# Headings and body text do not start with two or three "name, name," in a row
_AUTHOR_NAME = r"(?:(?:[A-Z]\.[\s-]?){1,3}[A-Z][\w'’-]+|[A-Z][a-z][\w'’-]*(?:\s(?:[A-Z]\.|[A-Z][\w'’-]+)){1,3})"


_AUTHOR_LIST_RE = re.compile(rf"^{_AUTHOR_NAME}(?:,\s+(?:and\s+)?{_AUTHOR_NAME}){{2,}}(?:,|\.|$)")


# A line of author names only: two or more comma-separated names of at least two words each (middle initials
# may lack the period, lowercase particles such as van der are allowed); when the line wraps, the last may be
# only a first name (after at least two full names). Old fonts set accents as separate glyphs (S´ebastien)
_NAME_WORD = r"[A-Z][\w´`¨˜ˆ'’.-]*"


_FULL_NAME = rf"{_NAME_WORD}(?:\s+(?:(?:van|von|der|den|de|da|di|du|del|la|le)\s+)*{_NAME_WORD})+"


_NAMES_LINE_RE = re.compile(
    rf"^{_FULL_NAME}(?:,\s+(?:and\s+)?{_FULL_NAME})*,\s+(?:and\s+)?{_FULL_NAME}[,.]?$"
    rf"|^{_FULL_NAME}(?:,\s+(?:and\s+)?{_FULL_NAME})+,\s+(?:and\s+)?{_NAME_WORD}[,.]?$"
)


# ACL style: Authors. 2018. Title. In Proceedings ..., pages 1–10. The year is a sentence of its own after the
# authors and is followed by the venue
_BIBLIO_ACL_YEAR_RE = re.compile(r"(?:^|\.\s)(?:19|20)\d{2}[a-z]?\.\s+\S")


_BIBLIO_VENUE_RE = re.compile(r"\bIn\s+(?:Proc|[A-Z]{2,})|\bpages\s+\d+\s*[–-]\s*\d+|\bJournal\b")


def _biblio_heading_like(block: TextBlock, text: str, median: float) -> bool:
    if not text or len(text) > 120:
        return False
    if block.size > median + 0.4:
        return True
    return bool(block.bold and block.size + 0.4 >= median and len(text) < 80)


def _biblio_entry_citation(text: str) -> bool:
    """The bibliography is entered only when the line after the heading looks like a citation. 2017 / arXiv /
    出版社 / [J] in body text do not count."""
    if _BIBLIO_CITE_RE.search(text) or _author_entry(text):
        return True
    # A type mark plus a year makes an entry; “出处见某书[M]。后续还有讨论。” is body text
    if _BIBLIO_TYPE_MARK_RE.search(text) and _BIBLIO_YEAR_RE.search(text):
        return True
    lead = bool(_BIBLIO_LEAD_RE.match(text))
    mark = bool(_BIBLIO_MARK_RE.search(text))
    if (_BIBLIO_YEAR_END_RE.search(text) or _BIBLIO_PAGES_END_RE.search(text)) and (lead or mark):
        return True
    return lead and (mark or bool(_BIBLIO_CN_PUB_RE.search(text)))


def _author_entry(text: str) -> bool:
    """Starts with a list of author names, or follows the ACL style "Authors. Year. Title. In Venue". The
    bibliography is often set a little larger than the body text, and by size the author line looks like a
    heading, so this is how it is recognized."""
    if _AUTHOR_LIST_RE.match(text):
        return True
    return bool(_BIBLIO_ACL_YEAR_RE.search(text) and _BIBLIO_VENUE_RE.search(text))


def _vol_roman_continuation(text: str) -> bool:
    """Vol. III / vol. xii are volume numbers. The words mix / mill are not."""
    head = _BIBLIO_VOL_LABEL_RE.match(text)
    if head is None:
        return False
    return _ROMAN_VOL_RE.match(text, head.end()) is not None


def _biblio_exit_continuation(text: str) -> bool:
    """Still a citation continuation at the same size. The next chapter with a year-like start is not."""
    if _BIBLIO_EXIT_VOLUME_RE.match(text) or _vol_roman_continuation(text):
        return True
    return bool(_BIBLIO_CN_PUB_RE.search(text) and _BIBLIO_IMPRINT_END_RE.search(text))


def _biblio_year_continuation(text: str) -> bool:
    """The block after a type mark is still part of the entry. A year appearing later does not count."""
    if _BIBLIO_VOLUME_RE.match(text) or _vol_roman_continuation(text):
        return True
    # A publisher line ends with a year or "年版". Body text mentioning a publisher and continuing does not count.
    return bool(_BIBLIO_CN_PUB_RE.search(text) and _BIBLIO_IMPRINT_END_RE.search(text))


def _biblio_citation_line(block: TextBlock, text: str, start_size: float, median: float) -> bool:
    """Already in the bibliography: the line is a citation or part of a wrapped entry and must not be taken for
    the next chapter."""
    if _BIBLIO_CITE_RE.search(text) or _BIBLIO_MARK_RE.search(text) or _author_entry(text):
        return True
    heading = _biblio_heading_like(block, text, median)
    short_heading = heading and len(text) < 40
    # Larger than the starting heading, or a short heading at the same size, is still the next chapter. Bold lines
    # over 80 characters are first checked for a type mark and a trailing year.
    if block.size > start_size + 0.5 or short_heading or (block.bold and heading):
        return False
    if _BIBLIO_TYPE_MARK_RE.search(text) or _BIBLIO_TRAIL_YEAR_RE.search(text):
        return True
    # Real volume, issue and publisher continuation lines at the heading's size are not the next chapter either.
    if _biblio_exit_continuation(text):
        return True
    if not _BIBLIO_LEAD_RE.match(text):
        return False
    return True


def biblio_skips(blocks: list[TextBlock]) -> list[bool]:
    """Mark the blocks of bibliography sections. The heading itself is not skipped; the section body keeps the
    original text.

    Short lines with the same name in running headers and the table of contents must not lower the entry size;
    a citation at the same size is not the next chapter heading either.
    """
    from .html_blocks import BIBLIO_HEADING_RE, BIBLIO_SHORT_LABEL_RE, BIBLIO_SUBHEADING_RE

    if not blocks:
        return []
    # Median weighted by characters: counted per block, joining continuation lines leaves fewer body blocks, the
    # median drops to a small size and the bibliography heading is missed
    median = median_all = _median_size(blocks)
    prepared = [(b, re.sub(r"\s+", " ", b.text).strip()) for b in blocks]

    def smaller_subsection(block: TextBlock, text: str, origin: float) -> bool:
        # Books / 论文 at the same size or larger is the next chapter, not a subheading inside.
        if len(text) >= 40 or block.size + 0.4 >= origin:
            return False
        if BIBLIO_SUBHEADING_RE.match(text):
            return True
        return bool(BIBLIO_SHORT_LABEL_RE.match(text) and _biblio_heading_like(block, text, median))

    def citations_follow(idx: int) -> bool:
        origin = prepared[idx][0].size
        pending_lead = False
        pending_type = False
        # Author-year entries have no number and no year in parentheses (Abadi, M., A. Agarwal, ... 2001.); only
        # several in a row count
        author_leads = 0
        for block, text in prepared[idx + 1 : idx + 16]:
            # Endnotes / 参考文献 is the heading of another section: stop here and do not raise the entry size
            if len(text) < 40 and BIBLIO_HEADING_RE.match(text):
                return False
            if smaller_subsection(block, text, origin):
                pending_lead = False
                pending_type = False
                continue
            # A heading-like line is not a citation even with [J] or a year; stop
            if _biblio_heading_like(block, text, median) and len(text) < 80:
                return False
            # A number and the publisher, or a type mark and the year, are often split into adjacent blocks. Only
            # the next block is checked.
            if (
                _biblio_entry_citation(text)
                or (pending_lead and _BIBLIO_CN_PUB_RE.search(text))
                or (pending_type and _biblio_year_continuation(text))
            ):
                return True
            if _AUTHOR_LEAD_RE.match(text):
                author_leads += 1
                if author_leads >= _AUTHOR_LEADS_TO_ENTER:
                    return True
            pending_lead = bool(_BIBLIO_LEAD_RE.match(text))
            pending_type = bool(_BIBLIO_TYPE_MARK_RE.search(text) and not _BIBLIO_YEAR_RE.search(text))
        return False

    def exits(
        block: TextBlock,
        text: str,
        start_size: float,
        prev_numbered: bool,
        entry_size: float,
        prev_text: str,
    ) -> bool:
        # Heading-likeness is judged against the entry size: some bibliographies are one size larger than the
        # body text, and against the document median the entries would look like headings
        median = max(median_all, entry_size)
        if not text or len(text) > 120 or smaller_subsection(block, text, start_size):
            return False
        # A number on its own line in a hanging indent, [ESS] or [12], is an entry label, not the next chapter
        if _BIBLIO_LEAD_RE.fullmatch(text + " "):
            return False
        # A line starting with mid-sentence punctuation, or a URL without spaces, is the wrapped rest of the
        # previous entry (wrapped after https:, the next line is //github.com/...;
        # ...-Abstract-Datasets_and_Benchmarks_Track.html.)
        if text[:1] in "/.,;:)]}-–_" or (" " not in text and re.search(r"[/_]|\.[A-Za-z]{2,}\.?$", text)):
            return False
        if block.size + 0.5 < start_size * 0.8:
            return False
        if _biblio_citation_line(block, text, start_size, median):
            return False
        # Journal alone on a line in a citation is not the next chapter
        if len(text) < 40 and BIBLIO_SHORT_LABEL_RE.match(text) and not _biblio_heading_like(block, text, median):
            return False
        # A wrapped line of a numbered entry starts lowercase and is not the next chapter.
        # A short line ending with a period (Databases. White Paper.) is also part of a broken entry: after OCR of
        # bitmap fonts all sizes are the same, so size cannot tell them apart
        if prev_numbered and (_BIBLIO_WRAP_RE.match(text) or text.endswith(".")):
            return False
        if _biblio_heading_like(block, text, median):
            return True
        # Long bold headings (the bold branch of heading_like has an 80-character limit)
        if block.bold and block.size + 0.4 >= median:
            return True
        # When the first line of an entry wraps, it may hold only author names (Ahmed M. Alaa, Zaid Ahmad, and Mark
        # van der Laan.), and a short line after the names is the title (API is enough: Conformal prediction for
        # large lan-); neither is the next chapter
        names = _NAMES_LINE_RE.match(text) or _NAMES_LINE_RE.match(prev_text)
        return len(text) < 60 and block.size >= start_size - 0.5 and not names

    skips: list[bool] = []
    in_biblio = False
    start_size = 0.0
    prev_numbered = False
    # Sizes of the entries already taken into the bibliography
    entry_sizes: list[float] = []
    for i, (block, text) in enumerate(prepared):
        title = len(text) < 40 and bool(BIBLIO_HEADING_RE.match(text))
        if title and in_biblio:
            prev_numbered = False
            if block.size + 0.5 >= start_size * 0.8:
                if block.size > start_size:
                    start_size = block.size
                skips.append(False)
            else:
                skips.append(True)
            continue
        # Layouts where the heading is smaller than the body text (ICASSP sets the whole bibliography one size
        # smaller): a bold line no smaller than the citations below is also the section heading
        following = sorted(b.size for b, _t in prepared[i + 1 : i + 6]) or [block.size]
        if (
            title
            and citations_follow(i)
            and (
                _biblio_heading_like(block, text, median)
                or block.size + 0.4 >= median
                or (block.bold and block.size + 0.4 >= following[len(following) // 2])
            )
        ):
            in_biblio = True
            start_size = block.size
            prev_numbered = False
            entry_sizes = []
            skips.append(False)
            continue
        if in_biblio and smaller_subsection(block, text, start_size):
            prev_numbered = False
            skips.append(False)
            continue
        entry_size = sorted(entry_sizes)[len(entry_sizes) // 2] if entry_sizes else 0.0
        if in_biblio and exits(block, text, start_size, prev_numbered, entry_size, prepared[i - 1][1] if i else ""):
            in_biblio = False
            prev_numbered = False
            skips.append(False)
            continue
        skips.append(in_biblio)
        if in_biblio:
            entry_sizes.append(block.size)
        if in_biblio and _BIBLIO_LEAD_RE.match(text):
            prev_numbered = True
        elif _BIBLIO_WRAP_RE.match(text):
            pass
        elif len(text) < 40 and BIBLIO_SHORT_LABEL_RE.match(text) and not _biblio_heading_like(block, text, median):
            pass  # a single word in a citation is not a heading; the numbered entry continues
        else:
            prev_numbered = False
    for index in _headless_bibliography(prepared) | _headless_numbered_bibliography(prepared, median):
        skips[index] = True
    return skips


# Start of an author-year entry: Ahia, O.,  /  Arkadiev, P. M. 2020.  /  Clark, H. H. and  /  Smith, J. (2020)
_AUTHOR_LEAD_RE = re.compile(
    r"^[A-Z][^\s,.;:()]{1,30}(?:[\s-][A-Za-z][^\s,.;:()]{1,30}){0,3},\s"
    r"(?:[A-Z]\.(?:\s?-?[A-Z]\.)*|[A-Z][a-z]+)"
    r"(?:,|\s(?:and|&)\s|\s\(?(?:19|20)\d{2})"
)


_AUTHOR_LEADS_TO_ENTER = 2  # minimum blocks after a bibliography heading that start like author-year entries


_ENTRY_WINDOW = 8  # number of blocks looked ahead


_ENTRY_LEADS = 4  # minimum entry starts among them


_ENTRY_GAP = 4  # the bibliography ends after this many blocks without a new entry


def _headless_bibliography(prepared: list[tuple[TextBlock, str]]) -> set[int]:
    """A bibliography without a heading: books often set it right after the exercises at a chapter end, without
    References.

    It is a run of entries in the same size starting with "Surname, Initials.". A body sentence occasionally
    starts this way but never adds up to enough. Running headers and wrapped lines of entries in between are
    included. A block clearly larger is the next section and ends it.
    """
    leads = [bool(_AUTHOR_LEAD_RE.match(text)) for _, text in prepared]
    found: set[int] = set()
    index = 0
    while index < len(prepared):
        size = prepared[index][0].size

        # All entries share one size. A body sentence starting this way has a different size and is not counted
        # with the entries after it
        def entry(at: int) -> bool:
            return leads[at] and abs(prepared[at][0].size - size) <= 0.5

        window = range(index, min(index + _ENTRY_WINDOW, len(prepared)))
        if not leads[index] or sum(entry(at) for at in window) < _ENTRY_LEADS:
            index += 1
            continue
        last = index
        cursor = index + 1
        while cursor < len(prepared) and cursor - last <= _ENTRY_GAP:
            if entry(cursor):
                last = cursor
            cursor += 1
        end = last + 1
        # Continuation lines of the last entry: no larger in size
        while end < cursor and prepared[end][0].size <= size + 0.5 and not leads[end]:
            end += 1
        found.update(range(index, end))
        index = max(end, index + 1)
    return found


# Start of a numbered entry: [12] A. Buonanno. When the end of the previous entry and the next entry share a
# block, the number follows a period
_NUMBERED_LEAD_RE = re.compile(r"(?:^|[.)]\s)\[(\d{1,3})\]\s+\S")


def _headless_numbered_bibliography(prepared: list[tuple[TextBlock, str]], median: float) -> set[int]:
    """A numbered bibliography without a heading (revtex journal layouts do not print References): starting
    from a block that begins with [1], consecutive numbers reaching [4] or more within the next few blocks.
    It ends where the numbers break, several blocks pass without a new entry, the size grows, or a heading
    appears (SUPPLEMENTAL MATERIAL, Appendix F in another column). A block after the heading starting with
    the next number continues the entries in another column, and detection resumes there.
    """
    numbers = [[int(n) for n in _NUMBERED_LEAD_RE.findall(text)] for _, text in prepared]

    def stops(block: TextBlock, text: str, size: float) -> bool:
        return block.size > size + 0.5 or _biblio_heading_like(block, text, median) or (block.bold and len(text) < 80)

    found: set[int] = set()
    resume = None
    index = 0
    while index < len(prepared):
        block, text = prepared[index]
        opens = text.startswith("[1] ") and _biblio_entry_citation(text)
        if not opens and (resume is None or resume not in numbers[index]):
            index += 1
            continue
        if opens:
            first = 1
        else:
            # Without an opening "[1] " entry the check above only lets a block through when resume is set
            assert resume is not None
            first = resume
        size = block.size
        expected = first
        last = cursor = index
        while cursor < len(prepared) and cursor - last <= _ENTRY_GAP and not stops(*prepared[cursor], size):
            for number in numbers[cursor]:
                if number == expected:
                    expected += 1
                    last = cursor
            cursor += 1
        if first == 1 and expected - 1 < _ENTRY_LEADS:
            index += 1
            continue
        end = last + 1
        # Continuation lines of the last entry: no more numbers
        while end < cursor and not numbers[end]:
            end += 1
        found.update(range(index, end))
        resume = expected
        index = end
    return found
