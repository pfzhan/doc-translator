"""Subtitle translation: SRT, WebVTT and ASS / SSA. Timings, cue numbers, cue settings and styles are kept; only the
text of each cue is translated.

Each cue is one paragraph, with its lines joined. A cue whose lines all start with a dash holds two speakers and is
translated line by line. The bilingual version keeps the original lines and adds the translation below them.
"""

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from bs4 import UnicodeDammit

from .html_blocks import translatable

if TYPE_CHECKING:
    from ..runner import Runner

TIMING_RE = re.compile(r"^\s*(?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3}\s+-->\s+(?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3}")
# Speaker dashes of a cue holding two speakers: - Where to? / - Home.
DASH_RE = re.compile(r"^\s*[-–—]\s*")
# Tags inside SRT and VTT text: <i>, <b>, <font color=...>, <c.yellow>, <v Anna>, <00:00:01.000>
TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>|<\d[\d:.]*>")
# ASS override blocks such as {\an8} or {\i1}; SRT files converted from ASS keep them too
OVERRIDE_RE = re.compile(r"\{\\[^{}]*\}")
# Override blocks at the start of an SRT cue, such as {\an8} placing it at the top; players read them as in ASS
LEAD_RE = re.compile(r"^(?:\{\\[^{}]*\})+")
WHOLE_ITALIC_RE = re.compile(r"^<i>(.*)</i>$", re.S)
VOICE_RE = re.compile(r"^<v(?:\.[\w.-]+)?\s+[^>]*>")
# ASS drawing mode ({\p1}): the text is vector drawing commands, not words
DRAWING_RE = re.compile(r"\\p[1-9]")
# Han characters, kana and hangul: lines in these scripts join without a space
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
SUFFIXES = {".srt", ".vtt", ".ass", ".ssa"}
# Override blocks at the start of an ASS dialogue text
ASS_LEAD_RE = re.compile(r"^(?:\{[^{}]*\})*")


@dataclass
class Cue:
    """Text lines of one cue in the parsed file: lines[start:end] are replaced when writing."""

    start: int
    end: int
    lines: list[str]
    # Translated parts, one per paragraph sent: the whole cue, or each speaker line
    parts: list[str] = field(default_factory=list)
    # ASS only: the fields before the text and the override block kept at the start of the text
    head: str = ""
    lead: str = ""


def _common_chinese(char: str, codec: str) -> bool:
    """Whether a character is among the common ones of a Chinese encoding: GB2312 for GBK, symbols and level 1
    characters for Big5."""
    try:
        code = int.from_bytes(char.encode(codec))
    except UnicodeEncodeError:
        return False
    return codec == "gb2312" or 0xA140 <= code <= 0xC67E


def _chinese(data: bytes) -> str | None:
    """GBK or Big5 text. Both encodings accept most byte pairs, so the reading whose non-ASCII characters are almost
    all common ones is taken."""
    for codec, common in (("gb18030", "gb2312"), ("big5", "big5")):
        try:
            text = data.decode(codec)
        except UnicodeDecodeError:
            continue
        chars = [char for char in text if ord(char) > 0x7F]
        if CJK_RE.search(text) and sum(_common_chinese(char, common) for char in chars) >= len(chars) * 0.9:
            return text
    return None


def read_text(src: Path) -> str:
    """Subtitles often come in a legacy encoding (GBK, Big5, Windows-1252). UTF-8 and BOMs are tried first, then
    the Chinese encodings, then the encoding is guessed."""
    data = src.read_bytes()
    for bom, codec in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if data.startswith(bom):
            return data.decode(codec, errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        guessed = _chinese(data) or UnicodeDammit(data).unicode_markup
        return guessed if guessed is not None else data.decode("utf-8", errors="replace")


def _join(lines: list[str]) -> str:
    """Lines of one paragraph: Chinese, Japanese and Korean lines join without a space."""
    out = ""
    for line in lines:
        line = line.strip()
        if out and line and not (CJK_RE.match(out[-1]) and CJK_RE.match(line[0])):
            out += " "
        out += line
    return out


def _prefix(pattern: re.Pattern[str], text: str) -> str:
    """The text that pattern matches at the start of text, or "" when it does not match."""
    m = pattern.match(text)
    return m.group(0) if m else ""


def _speaker_lines(lines: list[str]) -> bool:
    return len(lines) > 1 and all(DASH_RE.match(line) for line in lines)


def _plain(text: str) -> str:
    return html.unescape(TAG_RE.sub("", OVERRIDE_RE.sub("", text)))


def _texts(lines: list[str]) -> list[str]:
    """Paragraphs sent for one SRT or VTT cue."""
    if _speaker_lines(lines):
        return [_plain(DASH_RE.sub("", line)).strip() for line in lines]
    return [_join([_plain(line) for line in lines])]


def parse_cues(lines: list[str]) -> list[Cue]:
    """SRT and VTT: the text of a cue is the lines after its timing line, up to a blank line or, in files missing
    that blank line, up to the next cue's number and timing line. VTT headers, NOTE, STYLE and REGION blocks have no
    timing line and are kept as they are."""

    def next_cue(i: int) -> bool:
        if TIMING_RE.match(lines[i]):
            return True
        return lines[i].strip().isdigit() and i + 1 < len(lines) and TIMING_RE.match(lines[i + 1]) is not None

    cues = []
    i = 0
    while i < len(lines):
        if TIMING_RE.match(lines[i]):
            start = end = i + 1
            while end < len(lines) and lines[end].strip() and not next_cue(end):
                end += 1
            if end > start:
                cues.append(Cue(start, end, lines[start:end]))
            i = end
        else:
            i += 1
    return cues


def _ass_fields(lines: list[str]) -> int:
    """Number of fields of a Dialogue line in the [Events] section; Text is the last one."""
    for line in lines:
        if line.lower().startswith("format:") and "text" in line.lower():
            return len(line.split(":", 1)[1].split(","))
    return 10


def parse_ass(lines: list[str]) -> list[Cue]:
    """ASS / SSA: Dialogue lines in [Events]. The override block at the start of the text (position, style) is kept;
    those inside the text are dropped from the translation. \\N is a line break."""
    cues = []
    section = ""
    fields = 10
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.lower()
            if section == "[events]":
                fields = _ass_fields(lines[i + 1 :])
            continue
        if section != "[events]" or not stripped.lower().startswith("dialogue:"):
            continue
        parts = line.split(",", fields - 1)
        if len(parts) < fields:
            continue
        text = parts[-1]
        if DRAWING_RE.search("".join(OVERRIDE_RE.findall(text))):
            continue
        lead = _prefix(ASS_LEAD_RE, text)
        body = re.split(r"\\[Nn]", OVERRIDE_RE.sub("", text[len(lead) :]).replace("\\h", " "))
        cues.append(Cue(i, i + 1, body, head=",".join(parts[:-1]) + ",", lead=lead))
    return cues


def _cue_texts(cue: Cue, ass: bool) -> list[str]:
    if ass:
        return [line.strip() for line in cue.lines] if _speaker_lines(cue.lines) else [_join(cue.lines)]
    return _texts(cue.lines)


def _render_srt(cue: Cue, bilingual: bool, vtt: bool) -> list[str]:
    """Translated text lines of an SRT or VTT cue. A cue set wholly in italics keeps the italics, a VTT voice tag
    at the start keeps its speaker and override blocks at the start stay in front."""
    first = cue.lines[0].strip()
    lead = _prefix(LEAD_RE, first)
    first = first[len(lead) :]
    voice = _prefix(VOICE_RE, first) if vtt else ""
    whole = WHOLE_ITALIC_RE.match("\n".join([first, *(line.strip() for line in cue.lines[1:])]))

    def wrap(text: str) -> str:
        text = html.escape(text, quote=False) if vtt else text
        return f"<i>{text}</i>" if whole else text

    if _speaker_lines(cue.lines):
        out = [f"{_prefix(DASH_RE, line)}{wrap(part)}" for line, part in zip(cue.lines, cue.parts)]
    else:
        out = [voice + wrap(cue.parts[0])]
    out[0] = lead + out[0]
    return [*cue.lines, *out] if bilingual else out


def _render_ass(cue: Cue, original: str, bilingual: bool) -> str:
    if _speaker_lines(cue.lines):
        text = "\\N".join(f"{_prefix(DASH_RE, line)}{part}" for line, part in zip(cue.lines, cue.parts))
    else:
        text = cue.parts[0]
    if bilingual:
        return f"{cue.head}{original}\\N{text}"
    return f"{cue.head}{cue.lead}{text}"


def _ass_title(lines: list[str]) -> str:
    for line in lines:
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return ""


async def translate_subtitles(src: Path, out_dir: Path, runner: "Runner", bilingual: bool) -> list[Path]:
    content = read_text(src)
    newline = "\r\n" if "\r\n" in content else "\n"
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ext = src.suffix.lower()
    ass = ext in (".ass", ".ssa")
    cues = parse_ass(lines) if ass else parse_cues(lines)
    if not cues:
        raise ValueError("没有找到字幕文本")
    groups = [_cue_texts(cue, ass) for cue in cues]
    texts = [text for group in groups for text in group]
    runner.set_title((_ass_title(lines) if ass else "") or src.stem)
    results = await runner.translate_all(
        texts, kinds=["p"] * len(texts), preview=True, skip=[not translatable(t) for t in texts]
    )
    pos = 0
    for cue, group in zip(cues, groups):
        cue.parts = [r if translatable(t) else t for t, r in zip(group, results[pos : pos + len(group)])]
        pos += len(group)
    # Replace from the end so the line numbers of earlier cues stay valid
    for cue in reversed(cues):
        if all(part == text for part, text in zip(cue.parts, _cue_texts(cue, ass))):
            continue
        if ass:
            lines[cue.start : cue.end] = [_render_ass(cue, lines[cue.start][len(cue.head) :], bilingual)]
        else:
            lines[cue.start : cue.end] = _render_srt(cue, bilingual, ext == ".vtt")
    suffix = "bilingual" if bilingual else "translated"
    dst = out_dir / f"{src.stem}.{suffix}{src.suffix}"
    dst.write_text(newline.join(lines), encoding="utf-8")
    return [dst]
