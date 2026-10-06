"""译文断行，以及块内粗体、斜体的占位符。

公式哨兵是不可拆的原子。样式标记不占宽，只在块内强调不一致时写出，
整块同一风格仍交给块级 font-weight，送翻文本和原来逐字一致。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTINEL_RE = re.compile(r"\x01i\x02(\d+)\x01/i\x02")
STYLE_MARK_RE = re.compile(r"\{/?[bi]\}")
_ATOM_RE = re.compile(r"\x01i\x02\d+\x01/i\x02|\{/?[bi]\}")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

_BOLD_FLAG = 16
_ITALIC_FLAG = 2


@dataclass(frozen=True)
class Emphasis:
    bold: bool
    italic: bool


class EmphasisWriter:
    """相对块的底色写出 {b} {/b} {i} {/i}。公式占位符是原子，不进强调。"""

    def __init__(self, base: Emphasis, mixed: bool) -> None:
        self.base = base
        self.mixed = mixed
        self._bold = False
        self._italic = False

    def text(self, text: str, emphasis: Emphasis) -> str:
        if not text or not self.mixed:
            return text
        return self._marks(emphasis) + text

    def atom(self, text: str) -> str:
        if not self.mixed:
            return text
        return self._marks(self.base) + text

    def close(self) -> str:
        if not self.mixed:
            return ""
        return self._marks(self.base)

    def _marks(self, emphasis: Emphasis) -> str:
        want_bold = emphasis.bold and not self.base.bold
        want_italic = emphasis.italic and not self.base.italic
        parts: list[str] = []
        if self._italic and not want_italic:
            parts.append("{/i}")
            self._italic = False
        if self._bold and not want_bold:
            parts.append("{/b}")
            self._bold = False
        if want_bold and not self._bold:
            parts.append("{b}")
            self._bold = True
        if want_italic and not self._italic:
            parts.append("{i}")
            self._italic = True
        return "".join(parts)


def emphasis_of(font: str, flags: int) -> Emphasis:
    bold = bool(flags & _BOLD_FLAG) or "Bold" in font
    italic = bool(flags & _ITALIC_FLAG) or "Italic" in font or "Oblique" in font
    return Emphasis(bold, italic)


def writer_for(spans: list[dict[str, object]]) -> EmphasisWriter:
    """底色取最长的非公式 span。只有别的 span 和它不同时才写标记。"""
    weighted: list[tuple[Emphasis, int]] = []
    for span in spans:
        text = str(span.get("text") or "")
        if not text.strip():
            continue
        mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))
        weighted.append((mark, len(text)))
    if not weighted:
        return EmphasisWriter(Emphasis(False, False), False)
    base = max(weighted, key=lambda item: item[1])[0]
    mixed = any(mark != base for mark, _length in weighted)
    return EmphasisWriter(base, mixed)


def restore_emphasis(escaped_html: str) -> str:
    """html.escape 之后把样式标记换成标签。标记本身没有需要转义的字符。"""
    return (
        escaped_html.replace("{b}", "<b>")
        .replace("{/b}", "</b>")
        .replace("{i}", "<i>")
        .replace("{/i}", "</i>")
    )


def break_lines(
    text: str,
    em: float,
    max_width: float,
    formula_widths: dict[int, float],
) -> list[str]:
    """按估算宽度断行。哨兵和超宽公式各自占一整行，绝不拆开。"""
    if "\n" in text:
        lines: list[str] = []
        for part in text.split("\n"):
            lines.extend(break_lines(part, em, max_width, formula_widths))
        return lines
    if not text or max_width <= 0:
        return [text]
    lines: list[str] = []
    buf: list[str] = []
    width = 0.0

    def flush() -> None:
        nonlocal width
        if buf:
            lines.append("".join(buf))
            buf.clear()
        width = 0.0

    for token in _tokenize(text):
        token_width = _token_width(token, em, formula_widths)
        if not token.isspace() and token_width > max_width:
            flush()
            lines.append(token)
            continue
        if buf and width + token_width > max_width:
            flush()
            if token.isspace():
                continue
        buf.append(token)
        width += token_width
    flush()
    return lines or [text]


def nowrap_lines(lines: list[str]) -> str:
    """每行一个 nowrap span。和括号组同一写法，MuPDF 不能再把公式从中间折开。"""
    return "<br>".join(f'<span style="white-space:nowrap">{line}</span>' for line in lines)


def tokenize(text: str) -> list[str]:
    return _tokenize(text)


def measure_token(token: str, em: float, formula_widths: dict[int, float]) -> float:
    return _token_width(token, em, formula_widths)


def formula_index(token: str) -> int | None:
    matched = _SENTINEL_RE.fullmatch(token)
    return int(matched.group(1)) if matched else None


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        atom = _ATOM_RE.match(text, index)
        if atom:
            tokens.append(atom.group(0))
            index = atom.end()
            continue
        char = text[index]
        if char.isspace() and char != "\n":
            end = index + 1
            while end < length and text[end].isspace() and text[end] != "\n":
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        if _is_word_char(char):
            end = index + 1
            while end < length and _is_word_char(text[end]):
                end += 1
            tokens.append(text[index:end])
            index = end
            continue
        tokens.append(char)
        index += 1
    return tokens


def _is_word_char(char: str) -> bool:
    if _CJK_RE.match(char):
        return False
    return char.isalnum() or char in "-'"


def _token_width(token: str, em: float, formula_widths: dict[int, float]) -> float:
    sentinel = _SENTINEL_RE.fullmatch(token)
    if sentinel:
        return formula_widths.get(int(sentinel.group(1)), em)
    if STYLE_MARK_RE.fullmatch(token):
        return 0.0
    width = 0.0
    for char in token:
        if _CJK_RE.match(char):
            width += em
        elif char.isspace():
            width += em * 0.33
        else:
            width += em * 0.5
    return width
