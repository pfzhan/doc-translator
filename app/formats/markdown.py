"""Markdown translation: split into blocks, skip code blocks / front matter / HTML blocks, and keep the
structure of lists, quotes and tables."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .html_blocks import BIBLIO_HEADING_RE

if TYPE_CHECKING:
    from ..runner import Runner

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# A line starting with $$ opens a formula block, which runs to the line ending with $$
MATH_OPEN_RE = re.compile(r"^\s{0,3}\$\$")
HEADING_RE = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*?)(\s+#+\s*)?$")
LIST_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?)(.*)$")
# Start column of list item content: the marker plus the following spaces; a task box counts as content
LIST_CONTENT_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
QUOTE_RE = re.compile(r"^(\s*(?:>\s?)+)(.*)$")
# GFM table separator row: at least one - per cell, optionally with colons for alignment
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
HR_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
# Setext heading underline: a line of === under a paragraph makes a level 1 heading, --- a level 2 heading
SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
HTML_TAG_RE = re.compile(r"^\s{0,3}</?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)")
# The whole line is one complete start or end tag, with no text after it
LONE_TAG_RE = re.compile(r"^\s{0,3}</?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*)?/?>\s*$")
# Block-level tags that open an HTML block in CommonMark. A line starting with an inline tag such as <kbd> or
# <a> is an ordinary paragraph
HTML_BLOCK_TAGS = frozenset(
    """
address article aside base basefont blockquote body caption center col colgroup dd details dialog dir div dl
dt fieldset figcaption figure footer form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend
li link main menu menuitem nav noframes ol optgroup option p param search section summary table tbody td
tfoot th thead title tr track ul pre script style textarea
""".split()
)
# Link reference definition: [id]: https://... A footnote [^1]: has body text and does not count
LINK_DEF_RE = re.compile(r"^\s{0,3}\[(?!\^)[^\]]+\]:\s*\S")
# GitHub alert marker: > [!NOTE]
ALERT_RE = re.compile(r"^\[![A-Za-z]+\]$")
# Only links / images / symbols, no translatable text
NO_TEXT_RE = re.compile(r"^[\W\d_]*$")


def _translatable(text: str) -> bool:
    stripped = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # Remove images
    stripped = re.sub(r"`[^`]*`", "", stripped)  # Remove inline code
    stripped = re.sub(r"https?://\S+", "", stripped)
    return bool(stripped.strip()) and not NO_TEXT_RE.match(stripped.strip())


def _starts_html_block(line: str) -> bool:
    """A comment, <!DOCTYPE>, a block-level tag, or a line holding a single tag starts an HTML block; an
    autolink <https://...> does not."""
    stripped = line.lstrip()
    if stripped.startswith(("<!--", "<?", "<!")):
        return True
    m = HTML_TAG_RE.match(line)
    if not m:
        return False
    return m.group(1).lower() in HTML_BLOCK_TAGS or bool(LONE_TAG_RE.match(line))


def _closes_fence(line: str, fence: str) -> bool:
    """A closing fence uses the same character, is no shorter than the opening fence, and has nothing after it.
    A fence of four backticks can contain ```."""
    stripped = line.strip()
    return len(stripped) >= len(fence) and stripped == fence[0] * len(stripped)


def _math_end(lines: list[str], start: int) -> int | None:
    """Last line of the $$ formula block starting at lines[start]. $$x$$ followed by more text is an inline
    formula. A block not closed before a blank line is not a formula block, so that one missing $$ does not
    swallow the rest of the document."""
    rest = lines[start].strip()[2:]
    if "$$" in rest:
        return start if rest.rstrip().endswith("$$") and rest.count("$$") == 1 else None
    for j in range(start + 1, len(lines)):
        if not lines[j].strip():
            return None
        if lines[j].rstrip().endswith("$$"):
            return j
    return None


def _interrupts_paragraph(line: str) -> bool:
    """CommonMark: a quote, a non-empty bullet list item, or an ordered list item starting at 1 can directly
    follow a paragraph without a blank line in between."""
    if QUOTE_RE.match(line):
        return True
    m = LIST_RE.match(line)
    if not m or not m.group(2).strip():
        return False
    number = re.match(r"\s*(\d+)", m.group(1))
    return number is None or int(number.group(1)) == 1


@dataclass
class Block:
    kind: str  # raw / heading / setext / para / list / quote / table
    lines: list[str]
    # Each translatable segment: (line index, prefix, text)
    segments: list[tuple[int, str, str]] = field(default_factory=list)


def split_blocks(text: str) -> list[Block]:
    lines = text.split("\n")
    blocks: list[Block] = []
    i = 0
    # YAML front matter (---) and Hugo TOML front matter (+++)
    if lines and lines[0].strip() in ("---", "+++"):
        ends = ("---", "...") if lines[0].strip() == "---" else ("+++",)
        for j in range(1, len(lines)):
            if lines[j].strip() in ends:
                blocks.append(Block("raw", lines[: j + 1]))
                i = j + 1
                break

    para: list[str] = []
    # Content start column of the most recent list item. A paragraph after a blank line indented to this column
    # still belongs to that list item
    list_offset: int | None = None

    def flush() -> None:
        if para:
            blocks.append(Block("para", para.copy()))
            para.clear()

    while i < len(lines):
        line = lines[i]
        m = FENCE_RE.match(line)
        if m:
            flush()
            fence = m.group(1)
            j = i + 1
            while j < len(lines) and not _closes_fence(lines[j], fence):
                j += 1
            blocks.append(Block("raw", lines[i : j + 1]))
            i = j + 1
            continue
        if not line.strip():
            flush()
            blocks.append(Block("raw", [line]))
            i += 1
            continue
        end = _math_end(lines, i) if MATH_OPEN_RE.match(line) else None
        if end is not None:
            # Formulas are kept as is: lines such as + b or - c inside the block must not be taken as list items
            flush()
            blocks.append(Block("raw", lines[i : end + 1]))
            i = end + 1
            continue
        # Per CommonMark, a tab advances to the next column that is a multiple of 4
        expanded = line.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip(" "))
        if list_offset is not None and indent < list_offset and not para:
            list_offset = None
        # An indented code block in a list item needs 4 more columns of indentation than the item content
        in_item = list_offset is not None and indent < list_offset + 4
        if para and SETEXT_RE.match(line):
            blocks.append(Block("setext", para + [line]))
            para.clear()
            i += 1
            continue
        if HR_RE.match(line) or indent >= 4 and not para and not in_item:
            # Thematic break / indented code block
            flush()
            blocks.append(Block("raw", [line]))
            i += 1
            continue
        if HEADING_RE.match(line):
            flush()
            blocks.append(Block("heading", [line]))
            i += 1
            continue
        if _starts_html_block(line) and not para:
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            blocks.append(Block("raw", lines[i:j]))
            i = j
            continue
        if LINK_DEF_RE.match(line) and not para:
            # Joining consecutive definitions into one paragraph would invalidate the later ones; keep them as is
            j = i
            while j < len(lines) and LINK_DEF_RE.match(lines[j]):
                j += 1
            blocks.append(Block("raw", lines[i:j]))
            i = j
            continue
        if i + 1 < len(lines) and _starts_table(line, lines[i + 1]):
            flush()
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                j += 1
            blocks.append(Block("table", lines[i:j]))
            i = j
            continue
        if para and _interrupts_paragraph(line):
            flush()
        if LIST_RE.match(line) and not para:
            j = i
            while j < len(lines) and lines[j].strip() and not FENCE_RE.match(lines[j]):
                # A heading / quote / HTML block / thematic break right after a list is a new block structure and
                # must not be merged into the last list item
                if (
                    j > i
                    and not LIST_RE.match(lines[j])
                    and (
                        HEADING_RE.match(lines[j])
                        or QUOTE_RE.match(lines[j])
                        or _starts_html_block(lines[j])
                        or HR_RE.match(lines[j])
                    )
                ):
                    break
                j += 1
            blocks.append(Block("list", lines[i:j]))
            last_item = _list_items(lines[i:j])[-1][3][0]
            # The last item starts with its own marker line (a list block starts with one)
            marker = LIST_CONTENT_RE.match(last_item)
            assert marker is not None
            list_offset = len(marker.group(0))
            i = j
            continue
        if QUOTE_RE.match(line) and not para:
            j = _quote_end(lines, i)
            blocks.append(Block("quote", lines[i:j]))
            i = j
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


def _starts_table(header: str, separator: str) -> bool:
    """GFM: the header row has |, the next line is a separator row, and both have the same number of columns.
    With different column counts, --- is a Setext heading underline."""
    if "|" not in header or not TABLE_SEP_RE.match(separator):
        return False
    return len(_split_cells(header)[1]) == len(_split_cells(separator)[1])


def _quote_end(lines: list[str], start: int) -> int:
    """End line (exclusive) of a quote block. A line without > that directly follows paragraph text in the
    quote and does not open a new block is a lazy continuation line of that paragraph (CommonMark) and still
    belongs to the quote. Headings, alert markers and code blocks cannot be continued lazily."""
    j = start
    fence = ""
    in_text = False
    while j < len(lines):
        m = QUOTE_RE.match(lines[j])
        if m:
            body = m.group(2)
            fm = FENCE_RE.match(body)
            if fence or fm:
                fence = "" if fence and _closes_fence(body, fence) else (fence or (fm.group(1) if fm else ""))
                in_text = False
            else:
                in_text = bool(body.strip()) and not (
                    ALERT_RE.match(body.strip()) or HEADING_RE.match(body) or HR_RE.match(body)
                )
        elif not (in_text and _continues_paragraph(lines[j])):
            break
        j += 1
    return j


def _continues_paragraph(line: str) -> bool:
    """Non-empty and not the start of a heading, code fence, thematic break, list item, HTML block or formula
    block."""
    return bool(line.strip()) and not (
        HEADING_RE.match(line)
        or FENCE_RE.match(line)
        or HR_RE.match(line)
        or LIST_RE.match(line)
        or _starts_html_block(line)
        or MATH_OPEN_RE.match(line)
    )


def _split_cells(line: str) -> tuple[str, list[str], str]:
    s = line.strip()
    lead = "|" if s.startswith("|") else ""
    trail = "|" if s.endswith("|") and len(s) > 1 else ""
    inner = s[len(lead) : len(s) - len(trail)]
    return lead, [c.strip() for c in re.split(r"(?<!\\)\|", inner)], trail


def collect_segments(blocks: list[Block]) -> list[str]:
    """Fill the segments of each block and return all texts to translate."""
    texts: list[str] = []
    for b in blocks:
        if b.kind == "heading":
            m = HEADING_RE.match(b.lines[0])
            assert m is not None  # heading blocks are the lines HEADING_RE matched
            b.segments = [(0, m.group(1), m.group(2))]
        elif b.kind == "para":
            # The prefix is the first-line indentation: a continuation paragraph in a list item must stay in its
            # column
            lead = b.lines[0][: len(b.lines[0]) - len(b.lines[0].lstrip())]
            b.segments = [(0, lead, " ".join(line.strip() for line in b.lines))]
        elif b.kind == "setext":
            b.segments = [(0, "", " ".join(line.strip() for line in b.lines[:-1]))]
        elif b.kind == "list":
            b.segments = [(idx, pre, text) for idx, pre, text, _ in _list_items(b.lines)]
        elif b.kind == "quote":
            b.segments = [
                (idx, qpre + mpre, text) for kind, idx, qpre, mpre, text, _ in _quote_units(b.lines) if kind != "raw"
            ]
        elif b.kind == "table":
            for idx, line in enumerate(b.lines):
                if idx == 1:
                    continue
                _, cells, _ = _split_cells(line)
                for c in cells:
                    b.segments.append((idx, "", c))
        if b.kind in ("heading", "setext", "para", "quote"):
            # Blocks without translatable text (only images, only links, etc.) are output as is, like raw
            b.segments = [s for s in b.segments if _translatable(s[2])]
        texts.extend(s[2] for s in b.segments if _translatable(s[2]))
    return texts


def segment_kinds(blocks: list[Block]) -> list[str]:
    """Paragraph types for the preview, one per text returned by collect_segments."""
    kinds = []
    for b in blocks:
        for _, pre, text in b.segments:
            if not _translatable(text):
                continue
            if b.kind == "heading":
                kinds.append(f"h{min(pre.count('#'), 6)}")
            elif b.kind == "setext":
                kinds.append("h1" if b.lines[-1].strip().startswith("=") else "h2")
            else:
                kinds.append({"list": "li", "table": "td", "quote": "quote"}.get(b.kind, "p"))
    return kinds


def segment_skips(blocks: list[Block]) -> list[bool]:
    """One flag per text returned by collect_segments: paragraphs in a bibliography section keep the original
    text, as in EPUB and PDF.

    Headings themselves are translated as usual. A heading of the same or a higher level ends the section.
    """
    skips: list[bool] = []
    level = 0
    for b in blocks:
        if b.kind in ("heading", "setext"):
            if b.kind == "heading":
                m = HEADING_RE.match(b.lines[0])
                assert m is not None  # heading blocks are the lines HEADING_RE matched
                lv = m.group(1).count("#")
                title = m.group(2).strip()
            else:
                lv = 1 if b.lines[-1].strip().startswith("=") else 2
                title = " ".join(line.strip() for line in b.lines[:-1])
            if len(title) <= 40 and BIBLIO_HEADING_RE.match(title):
                level = lv
            elif level and lv <= level:
                level = 0
            skips.extend(False for _, _, text in b.segments if _translatable(text))
            continue
        skips.extend(bool(level) for _, _, text in b.segments if _translatable(text))
    return skips


def _list_items(lines: list[str]) -> list[tuple[int, str, str, list[str]]]:
    """Split a list block into items: (start line index, marker prefix, joined text, raw lines). Lazy
    continuation lines join the previous item."""
    items: list[tuple[int, str, str, list[str]]] = []
    for idx, line in enumerate(lines):
        m = LIST_RE.match(line)
        if m or not items:
            pre, text = (m.group(1), m.group(2)) if m else ("", line.strip())
            items.append((idx, pre, text, [line]))
        else:
            start, pre, text, raw = items[-1]
            items[-1] = (start, pre, text + " " + line.strip(), raw + [line])
    return items


def _quote_units(lines: list[str]) -> list[tuple[str, int, str, str, str, list[str]]]:
    """Split a quote block into (kind, start line index, quote prefix, list marker, joined text, raw lines).

    Kind raw is output as is: empty quote lines, alert markers and code blocks inside the quote. para, item and
    heading are translated: each paragraph between empty quote lines, each list item and each heading in the
    quote is a separate segment, and a change of quote level also starts a new one.
    """
    units: list[tuple[str, int, str, str, str, list[str]]] = []
    fence = ""
    for idx, line in enumerate(lines):
        m = QUOTE_RE.match(line)
        if m is None:
            # Lazy continuation line: _quote_end only lets it in after paragraph text
            if units and units[-1][0] in ("para", "item"):
                kind, start, lpre, mpre, text, raw = units[-1]
                units[-1] = (kind, start, lpre, mpre, text + " " + line.strip(), raw + [line])
            else:
                units.append(("raw", idx, "", "", line, [line]))
            continue
        qpre, body = m.group(1), m.group(2)
        fm = FENCE_RE.match(body)
        if fence or fm:
            if fence and _closes_fence(body, fence):
                fence = ""
            elif fm:
                fence = fm.group(1)
            units.append(("raw", idx, qpre, "", body, [line]))
            continue
        if not body.strip() or ALERT_RE.match(body.strip()):
            units.append(("raw", idx, qpre, "", body, [line]))
            continue
        heading = HEADING_RE.match(body)
        if heading:
            units.append(("heading", idx, qpre, heading.group(1), heading.group(2), [line]))
            continue
        item = LIST_RE.match(body)
        if item:
            units.append(("item", idx, qpre, item.group(1), item.group(2), [line]))
            continue
        last = units[-1] if units else None
        if last and last[0] in ("para", "item") and last[2].count(">") == qpre.count(">"):
            kind, start, lpre, mpre, text, raw = last
            units[-1] = (kind, start, lpre, mpre, text + " " + body.strip(), raw + [line])
        else:
            units.append(("para", idx, qpre, "", body.strip(), [line]))
    return units


def render(blocks: list[Block], translations: list[str], bilingual: bool) -> str:
    """translations match the texts returned by collect_segments by position. They are not looked up by source
    text: when the same sentence appears once in the body and once in the bibliography, one is translated and
    the other keeps the original."""
    out: list[str] = []
    pending = iter(translations)

    def t(_s: str) -> str:
        return next(pending)

    for b in blocks:
        if b.kind == "raw" or not b.segments:
            out.extend(b.lines)
        elif b.kind == "heading":
            _, pre, text = b.segments[0]
            if bilingual:
                out.append(b.lines[0])
                out.append("")
            out.append(pre + t(text))
        elif b.kind == "setext":
            text = b.segments[0][2]
            if bilingual:
                out.extend(b.lines)
                out.append("")
            out.append(t(text))
            out.append(b.lines[-1])
        elif b.kind == "para":
            _, lead, text = b.segments[0]
            if bilingual:
                out.extend(b.lines)
                out.append("")
            out.append(lead + t(text))
        elif b.kind == "quote":
            for kind, _, qpre, mpre, text, raw in _quote_units(b.lines):
                if kind == "raw" or not _translatable(text):
                    out.extend(raw)
                elif kind == "heading":
                    if bilingual:
                        out.extend(raw)
                    out.append(qpre + mpre + t(text))
                elif kind == "item":
                    if bilingual:
                        out.append(qpre + mpre + text + "  ")
                        out.append(qpre + " " * len(mpre) + t(text))
                    else:
                        out.append(qpre + mpre + t(text))
                else:
                    if bilingual:
                        out.extend(raw)
                        out.append(qpre.rstrip())
                    out.append(qpre + t(text))
        elif b.kind == "list":
            for _, pre, text, raw in _list_items(b.lines):
                if not _translatable(text):
                    out.extend(raw)
                elif bilingual:
                    # Source + hard line break + translation, kept in the same list item
                    out.append(pre + text + "  ")
                    out.append(" " * len(pre) + t(text))
                else:
                    out.append(pre + t(text))
        elif b.kind == "table":
            for idx, line in enumerate(b.lines):
                if idx == 1:
                    out.append(line)
                    continue
                lead, cells, trail = _split_cells(line)
                new_cells = []
                for c in cells:
                    if not _translatable(c):
                        new_cells.append(c)
                    elif bilingual:
                        new_cells.append(f"{c}<br>{t(c)}")
                    else:
                        new_cells.append(t(c))
                row = " | ".join(new_cells)
                out.append((lead + " " if lead else "") + row + (" " + trail if trail else ""))
    return "\n".join(out)


async def translate_markdown(src: Path, out_dir: Path, runner: "Runner", bilingual: bool) -> list[Path]:
    text = src.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    blocks = split_blocks(text)
    texts = collect_segments(blocks)
    # Title: the first level 1 heading, or the file name if there is none
    # A heading without translatable text, such as "# 2024", has no segments
    h1 = next(
        (
            b.segments[0][2]
            for b in blocks
            if b.segments
            and (
                b.kind == "heading"
                and b.lines[0].lstrip().startswith("# ")
                or b.kind == "setext"
                and b.lines[-1].strip().startswith("=")
            )
        ),
        "",
    )
    runner.set_title(h1 or src.stem)
    results = await runner.translate_all(texts, kinds=segment_kinds(blocks), preview=True, skip=segment_skips(blocks))
    suffix = "bilingual" if bilingual else "translated"
    dst = out_dir / f"{src.stem}.{suffix}{src.suffix}"
    dst.write_text(render(blocks, results, bilingual), encoding="utf-8")
    return [dst]
