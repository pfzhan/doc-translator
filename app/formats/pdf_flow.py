"""译文断行，以及块内粗体、斜体、字号差的占位符。

公式哨兵是不可拆的原子。短括号组放得下时也不从中间切开。
样式标记不占宽，只在和块的底色不同时写出。
整块同一风格仍交给块级字号和字重，送翻文本和原来逐字一致。
字号标记是相对底色的百分比，写回时按这个比例缩放，跟着收缩阶梯一起变。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SENTINEL_RE = re.compile(r"\x01i\x02(\d+)\x01/i\x02")
STYLE_MARK_RE = re.compile(r"\{z\d+\}|\{/z\}|\{/?[bi]\}")
_ATOM_RE = re.compile(rf"\x01i\x02\d+\x01/i\x02|{STYLE_MARK_RE.pattern}")
_SIZE_OPEN_RE = re.compile(r"\{z(\d+)\}")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

_BOLD_FLAG = 16
_ITALIC_FLAG = 2
_PAREN_CLOSE = {"（": "）", "(": ")"}
_PAREN_LIMIT = 10


@dataclass(frozen=True)
class Emphasis:
    bold: bool
    italic: bool


class EmphasisWriter:
    """相对块的底色写出 {b} {i} {zN}。公式占位符是原子，不进强调，也不进字号。"""

    def __init__(self, base: Emphasis, mixed: bool, base_size: float = 0) -> None:
        self.base = base
        self.mixed = mixed
        self.base_size = base_size
        self._bold = False
        self._italic = False
        self._size = 100

    def text(self, text: str, emphasis: Emphasis, size: float = 0) -> str:
        if not text or not self.mixed:
            return text
        percent = size_percent(size, self.base_size) if size and self.base_size else 100
        return self._marks(emphasis, percent) + text

    def atom(self, text: str) -> str:
        if not self.mixed:
            return text
        return self._marks(self.base, 100) + text

    def close(self) -> str:
        if not self.mixed:
            return ""
        return self._marks(self.base, 100)

    def _marks(self, emphasis: Emphasis, size_pct: int) -> str:
        """字号是外层。换字号时先收起强调，换完再按目标强调打开，标记始终成对。"""
        if size_pct == self._size:
            return self._emphasis_marks(emphasis)
        return self._emphasis_marks(self.base) + self._switch_size(size_pct) + self._emphasis_marks(emphasis)

    def _switch_size(self, size_pct: int) -> str:
        parts: list[str] = []
        if self._size != 100:
            parts.append("{/z}")
        if size_pct != 100:
            parts.append(f"{{z{size_pct}}}")
        self._size = size_pct
        return "".join(parts)

    def _emphasis_marks(self, emphasis: Emphasis) -> str:
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


def size_percent(size: float, base: float) -> int:
    """相对块底色的字号，四舍五入到 5%。差不到 8% 视为相同，不写标记。"""
    if base <= 0 or size <= 0:
        return 100
    ratio = size / base
    if 0.92 <= ratio <= 1.08:
        return 100
    percent = int(round(ratio * 20.0)) * 5
    return min(200, max(50, percent))


def emphasis_of(font: str, flags: int) -> Emphasis:
    bold = bool(flags & _BOLD_FLAG) or "Bold" in font
    italic = bool(flags & _ITALIC_FLAG) or "Italic" in font or "Oblique" in font
    return Emphasis(bold, italic)


def writer_for(spans: list[dict[str, object]], base_size: float = 0) -> EmphasisWriter:
    """底色取最长的非公式 span。只有别的 span 在强调或字号上和它不同时才写标记。"""
    weighted: list[tuple[Emphasis, int]] = []
    size_mixed = False
    for span in spans:
        text = str(span.get("text") or "")
        if not text.strip():
            continue
        mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))
        weighted.append((mark, len(text)))
        if base_size and size_percent(float(span.get("size") or 0), base_size) != 100:
            size_mixed = True
    if not weighted:
        return EmphasisWriter(Emphasis(False, False), False, base_size)
    base = max(weighted, key=lambda item: item[1])[0]
    mixed = size_mixed or any(mark != base for mark, _length in weighted)
    return EmphasisWriter(base, mixed, base_size)


def restore_emphasis(escaped_html: str, cjk: bool = False) -> str:
    """html.escape 之后把样式标记换成标签。标记本身没有需要转义的字符。
    CJK 字体没有斜体字形，<i> 画了等于没画；映射成 <b> 保住强调。"""
    tagged = (
        escaped_html.replace("{b}", "<b>")
        .replace("{/b}", "</b>")
        .replace("{i}", "<b>" if cjk else "<i>")
        .replace("{/i}", "</b>" if cjk else "</i>")
        .replace("{/z}", "</span>")
    )
    return _SIZE_OPEN_RE.sub(r'<span style="font-size:\1%">', tagged)


def strip_style_marks(text: str) -> str:
    """预览里去掉样式标记。写回 PDF 时仍要这些标记。"""
    return STYLE_MARK_RE.sub("", text)


def break_lines(
    text: str,
    em: float,
    max_width: float,
    formula_widths: dict[int, float],
) -> list[str]:
    """按估算宽度断行。哨兵、超宽公式、放得下的短括号组都不拆开。"""
    if "\n" in text:
        lines: list[str] = []
        for part in text.split("\n"):
            lines.extend(break_lines(part, em, max_width, formula_widths))
        return lines
    if not text or max_width <= 0:
        return [text]
    tokens = _tokenize(text)
    group_of = _paren_groups(tokens, em, formula_widths, max_width)
    lines: list[str] = []
    buf: list[tuple[int, str]] = []
    width = 0.0

    def flush() -> None:
        nonlocal width
        if buf:
            lines.append("".join(token for _, token in buf))
            buf.clear()
        width = 0.0

    index = 0
    while index < len(tokens):
        token = tokens[index]
        token_width = _token_width(token, em, formula_widths)
        if not token.isspace() and token_width > max_width:
            flush()
            if _SENTINEL_RE.fullmatch(token):
                lines.append(token)  # 公式超宽：独占一行（调用方会缩它）
            else:
                # 超宽普通词（长 URL 等）按字符硬切，不 nowrap 越出右缘
                chunk, cw = "", 0.0
                for ch in token:
                    chw = _token_width(ch, em, formula_widths)
                    if chunk and cw + chw > max_width:
                        lines.append(chunk)
                        chunk, cw = "", 0.0
                    chunk += ch
                    cw += chw
                if chunk:
                    buf.append((index, chunk))
                    width += cw
            index += 1
            continue
        if buf and width + token_width > max_width:
            start = group_of.get(index)
            if start is not None and any(token_index >= start for token_index, _ in buf):
                buf[:] = [(token_index, item) for token_index, item in buf if token_index < start]
                width = sum(_token_width(item, em, formula_widths) for _, item in buf)
                flush()
                index = start
                continue
            flush()
            if token.isspace():
                index += 1
                continue
        buf.append((index, token))
        width += token_width
        index += 1
    flush()
    return lines or [text]


def _visible_len(token: str) -> int:
    """括号组的长度不计样式标记。公式哨兵算一个字，不按哨兵原文的字符数。"""
    if STYLE_MARK_RE.fullmatch(token):
        return 0
    if _SENTINEL_RE.fullmatch(token):
        return 1
    return len(token)


def _paren_groups(
    tokens: list[str],
    em: float,
    formula_widths: dict[int, float],
    max_width: float,
) -> dict[int, int]:
    """短括号组里每个 token 指向开括号的下标。组本身超宽时不锁，允许切开。"""
    group_of: dict[int, int] = {}
    index = 0
    count = len(tokens)
    while index < count:
        close = _PAREN_CLOSE.get(tokens[index])
        if close is None:
            index += 1
            continue
        end: int | None = None
        chars = 0
        width = 0.0
        for cursor in range(index, count):
            chars += _visible_len(tokens[cursor])
            width += _token_width(tokens[cursor], em, formula_widths)
            if chars > _PAREN_LIMIT:
                break
            if cursor > index and tokens[cursor] in _PAREN_CLOSE:
                break
            if tokens[cursor] == close and cursor > index:
                end = cursor
                break
        # 锁组条件和填充的溢出判定必须一致：放宽到 +0.5 会让组在填充时超宽，
        # 触发回滚后又因组已锁再次回滚，死循环
        if end is not None and width <= max_width:
            for cursor in range(index, end + 1):
                group_of[cursor] = index
            index = end + 1
        else:
            index += 1
    return group_of


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
        if _CJK_RE.match(char) or unicodedata.east_asian_width(char) in ("W", "F"):
            width += em
        elif char.isspace():
            width += em * 0.33
        else:
            width += em * 0.5
    return width
