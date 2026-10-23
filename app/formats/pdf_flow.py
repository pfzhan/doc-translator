"""译文断行，以及块内粗体、斜体、字号、颜色差的占位符。

公式哨兵是不可拆的原子。短括号组放得下时也不从中间切开。
样式标记不占宽，只在和块的底色不同时写出。
整块同一风格仍交给块级字号和字重，送翻文本和原来逐字一致。
字号标记是相对底色的百分比，写回时按这个比例缩放，跟着收缩阶梯一起变。
"""
from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

_SENTINEL_RE = re.compile(r"\x01i\x02(\d+)\x01/i\x02")
STYLE_MARK_RE = re.compile(r"\{z\d+\}|\{/z\}|\{c[0-9a-f]{6}\}|\{/c\}|\{/?[biu]\}")
_ATOM_RE = re.compile(rf"\x01i\x02\d+\x01/i\x02|{STYLE_MARK_RE.pattern}")
# 网址、邮箱整个不折行。结尾的句号、括号、逗号是句子的，不算在里面
_LINK_RE = re.compile(r"(?:https?://|www\.)[!-~]*[A-Za-z0-9/#=_-]|[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", re.A)
_SIZE_OPEN_RE = re.compile(r"\{z(\d+)\}")
COLOR_MARK_RE = re.compile(r"\{c[0-9a-f]{6}\}|\{/c\}")
_COLOR_OPEN_RE = re.compile(r"\{c([0-9a-f]{6})\}")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")

_BOLD_FLAG = 16
_ITALIC_FLAG = 2
_PAREN_CLOSE = {"（": "）", "(": ")"}
_PAREN_LIMIT = 10
# 这些标点不能出现在行首。放不下时把前一个字带到下一行，标点跟在它后面。
_NO_LINE_START = frozenset("。，、；：？！）】」』〉》”’％%.,;:?!)]}>")
# 左括号、左引号不能留在行尾，跟后面的字一起折到下一行。
_NO_LINE_END = frozenset("（【「『《〈“‘([{")
# 半角标点在行尾时，后面的空格是标点自带的间隔，不留在这一行。
_HALF_PUNCT = frozenset(".,;:?!)]}%")


@dataclass(frozen=True)
class Emphasis:
    bold: bool
    italic: bool
    # 原文字下面画了线。线不在字体里，提取时按字下面的细横线认出来（span["underline"]）
    underline: bool = False


class EmphasisWriter:
    """相对块的底色写出 {b} {i} {u} {zN} {cRRGGBB}。公式占位符是原子，不进强调、字号和颜色。
    写回时块级只有字号和粗细，下划线总按标记写，整块都有下划线也一样。"""

    def __init__(self, base: Emphasis, mixed: bool, base_size: float = 0, base_color: int | None = None) -> None:
        self.base = base
        self.mixed = mixed
        self.base_size = base_size
        self.base_color = base_color
        self._bold = False
        self._italic = False
        self._underline = False
        self._size = 100
        # 当前写着的颜色。None 是块的颜色
        self._color: int | None = None

    def text(self, text: str, emphasis: Emphasis, size: float = 0, color: int | None = None) -> str:
        if not text or not self.mixed:
            return text
        percent = size_percent(size, self.base_size) if size and self.base_size else 100
        # 只有标点的一截不换字号：句号排成角标字号时，{z65}.{/z} 会被模型套到半句话上。
        # 纯空白照旧，标签后面的空格还在字号标记外面
        if text.strip() and not re.search(r"[^\W_]", text):
            percent = self._size
        # 纯空白不换颜色：同色的两个词中间不断开
        shade = self._color if not text.strip() else marked_color(color, self.base_color)
        return self._marks(emphasis, percent, shade) + text

    def atom(self, text: str) -> str:
        if not self.mixed:
            return text
        return self._marks(self.base, 100) + text

    def close(self) -> str:
        if not self.mixed:
            return ""
        return self._marks(self.base, 100)

    def _marks(self, emphasis: Emphasis, size_pct: int, color: int | None = None) -> str:
        """由外到内是字号、颜色、强调。换外层时先收起里面的，换完再打开，标记始终成对嵌套。"""
        if size_pct == self._size and color == self._color:
            return self._emphasis_marks(emphasis)
        parts = [self._emphasis_marks(self.base)]
        if size_pct != self._size:
            parts.append(self._switch_color(None))
            parts.append(self._switch_size(size_pct))
        parts.append(self._switch_color(color))
        parts.append(self._emphasis_marks(emphasis))
        return "".join(parts)

    def _switch_color(self, color: int | None) -> str:
        if color == self._color:
            return ""
        parts: list[str] = []
        if self._color is not None:
            parts.append("{/c}")
        if color is not None:
            parts.append(f"{{c{color:06x}}}")
        self._color = color
        return "".join(parts)

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
        want_underline = emphasis.underline
        parts: list[str] = []
        if self._underline and not want_underline:
            parts.append("{/u}")
            self._underline = False
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
        if want_underline and not self._underline:
            parts.append("{u}")
            self._underline = True
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


def span_emphasis(span: dict[str, object]) -> Emphasis:
    """span 的粗细、斜体，加上提取时认出的下划线。"""
    mark = emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0))  # type: ignore[arg-type]
    return Emphasis(mark.bold, mark.italic, bool(span.get("underline")))


def writer_for(
    spans: list[dict[str, object]], base_size: float = 0, base_color: int | None = None,
) -> EmphasisWriter:
    """底色取最长的非公式 span。只有别的 span 在强调、字号或颜色上和它不同时才写标记。
    base_color 是块的颜色，写回时整块用它；和它不同的字才写颜色标记。"""
    weighted: list[tuple[Emphasis, int]] = []
    other_mixed = False
    for span in spans:
        text = str(span.get("text") or "")
        if not text.strip():
            continue
        mark = span_emphasis(span)
        weighted.append((mark, len(text)))
        if base_size and size_percent(float(span.get("size") or 0), base_size) != 100:
            other_mixed = True
        if marked_color(span.get("color"), base_color) is not None:
            other_mixed = True
    if not weighted:
        return EmphasisWriter(Emphasis(False, False), False, base_size, base_color)
    longest = max(weighted, key=lambda item: item[1])[0]
    base = Emphasis(longest.bold, longest.italic)
    mixed = other_mixed or any(mark != base for mark, _length in weighted)
    return EmphasisWriter(base, mixed, base_size, base_color)


def marked_color(color: object, base_color: int | None) -> int | None:
    """要写颜色标记的颜色。和块同色的不写；很浅的不写：黑底白字的图注标签整个贴回，
    白字写成译文会落在白底上看不见。块本身是浅色字时也不写。"""
    if base_color is None or not isinstance(color, int) or color == base_color:
        return None
    if _light(color) or _light(base_color):
        return None
    return color


def _light(color: int) -> bool:
    red, green, blue = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
    return 0.299 * red + 0.587 * green + 0.114 * blue > 0.8 * 255


def restore_emphasis(escaped_html: str) -> str:
    """html.escape 之后把样式标记换成标签。标记本身没有需要转义的字符。
    斜体保持 <i>：回退字体没有汉字粗体，改成 <b> 既不能强调汉字，又会丢掉拉丁斜体。"""
    tagged = (
        escaped_html.replace("{b}", "<b>")
        .replace("{/b}", "</b>")
        .replace("{i}", "<i>")
        .replace("{/i}", "</i>")
        .replace("{u}", "<u>")
        .replace("{/u}", "</u>")
        .replace("{/z}", "</span>")
        .replace("{/c}", "</span>")
    )
    tagged = _COLOR_OPEN_RE.sub(r'<span style="color:#\1">', tagged)
    return _SIZE_OPEN_RE.sub(r'<span style="font-size:\1%">', tagged)


def strip_style_marks(text: str) -> str:
    """预览里去掉样式标记。写回 PDF 时仍要这些标记。"""
    return STYLE_MARK_RE.sub("", text)


def break_lines(
    text: str,
    em: float,
    max_width: float,
    formula_widths: dict[int, float],
    first_widths: Sequence[float] = (),
) -> list[str]:
    """按估算宽度断行。哨兵、超宽公式、放得下的短括号组都不拆开。
    每行自带样式标记：跨行的粗体、颜色在下一行开头重开，在本行末尾收起。
    各行是分开写的（HTML 每行一个不折行的盒子，中文逐片画），不补的话下一行就丢了样式。
    first_widths 是开头几行各自的宽度（让开图的那几行），其余各行按 max_width。"""
    return balance_lines(_break_lines(text, em, max_width, formula_widths, first_widths))


def balance_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    opened: list[str] = []
    for line in lines:
        if not STYLE_MARK_RE.sub("", line):
            out.append(line)
            opened = open_marks(line, opened)
            continue
        text = "".join(opened) + line
        opened = open_marks(text)
        out.append(text + _closing(opened))
    return out


def open_marks(text: str, opened: list[str] | None = None) -> list[str]:
    """读完 text 后还开着的样式标记，由外到内。opened 是 text 之前已经开着的。"""
    stack = list(opened or ())
    for matched in STYLE_MARK_RE.finditer(text):
        token = matched.group(0)
        if not token.startswith("{/"):
            stack.append(token)
            continue
        for index in range(len(stack) - 1, -1, -1):
            if stack[index][1] == token[2]:
                del stack[index]
                break
    return stack


def _closing(opened: list[str]) -> str:
    return "".join("{/" + token[1] + "}" for token in reversed(opened))


def balance_marks(left: str, right: str) -> tuple[str, str]:
    """切开处补上仍开着的样式标记。字号、颜色不嵌套，必须重开还开着的那一个，不能重开更早已经闭合的。"""
    opened = open_marks(left)
    if not opened:
        return left, right
    return left + _closing(opened), "".join(opened) + right


def _break_lines(
    text: str,
    em: float,
    max_width: float,
    formula_widths: dict[int, float],
    first_widths: Sequence[float] = (),
) -> list[str]:
    if "\n" in text:
        lines: list[str] = []
        for part in text.split("\n"):
            lines.extend(_break_lines(part, em, max_width, formula_widths, first_widths[len(lines):]))
        return lines
    if not text or max_width <= 0:
        return [text]
    tokens = _tokenize(text)
    # 括号组按最窄的一行锁，否则组在窄行放不下，回滚后又回到同一处
    group_of = _paren_groups(tokens, em, formula_widths, min((max_width, *first_widths)))
    lines: list[str] = []
    buf: list[tuple[int, str]] = []
    width = 0.0

    def room() -> float:
        return first_widths[len(lines)] if len(lines) < len(first_widths) else max_width

    def flush() -> None:
        nonlocal width
        if buf:
            lines.append(_trim_line_end("".join(token for _, token in buf)))
            buf.clear()
        width = 0.0

    index = 0
    while index < len(tokens):
        token = tokens[index]
        token_width = _token_width(token, em, formula_widths)
        if not token.isspace() and token_width > room():
            flush()
            if _SENTINEL_RE.fullmatch(token):
                lines.append(token)  # 公式超宽：独占一行（调用方会缩它）
            else:
                # 超宽普通词（长 URL 等）按字符硬切，不 nowrap 越出右缘
                chunk, cw = "", 0.0
                for ch in token:
                    chw = _token_width(ch, em, formula_widths)
                    if chunk and cw + chw > room():
                        moved = _detach_char(chunk) if ch in _NO_LINE_START else None
                        if moved is not None:
                            head, tail = moved
                            if head:
                                lines.append(_trim_line_end(head))
                            chunk, cw = tail, _token_width(tail, em, formula_widths)
                        else:
                            lines.append(_trim_line_end(chunk))
                            chunk, cw = "", 0.0
                    chunk += ch
                    cw += chw
                if chunk:
                    buf.append((index, chunk))
                    width += cw
            index += 1
            continue
        if buf and width + token_width > room():
            start = group_of.get(index)
            if start is not None and any(token_index >= start for token_index, _ in buf):
                buf[:] = [(token_index, item) for token_index, item in buf if token_index < start]
                width = sum(_token_width(item, em, formula_widths) for _, item in buf)
                flush()
                index = start
                continue
            if token in _NO_LINE_START:
                detached = _detach_for_punct(buf)
                if detached is not None:
                    detached_width = sum(_token_width(item, em, formula_widths) for _, item in detached)
                    if detached_width + token_width <= room():
                        flush()
                        buf.extend(detached)
                        width = detached_width
                    else:
                        buf.extend(detached)
                        flush()
                        if token.isspace():
                            index += 1
                            continue
                else:
                    flush()
                    if token.isspace():
                        index += 1
                        continue
            else:
                carried = _detach_opener(buf)
                flush()
                if carried:
                    buf.extend(carried)
                    width = sum(_token_width(item, em, formula_widths) for _, item in carried)
                if token.isspace():
                    index += 1
                    continue
        buf.append((index, token))
        width += token_width
        index += 1
    flush()
    return lines or [text]


def _trim_line_end(line: str) -> str:
    """半角标点在行尾时去掉后面的空格。样式标记不算字。"""
    body = line
    marks = ""
    while True:
        matched = re.search(r"(\{/?[biu]\}|\{/z\}|\{z\d+\})$", body)
        if matched is None:
            break
        marks = matched.group(0) + marks
        body = body[:matched.start()]
    stripped = body.rstrip(" \t")
    if stripped != body and stripped and stripped[-1] in _HALF_PUNCT:
        return stripped + marks
    return line


def _detach_char(chunk: str) -> tuple[str, str] | None:
    """超宽硬切时，把最后一个字留给行首的标点。"""
    if len(chunk) < 2 or chunk[-1] in _NO_LINE_START:
        return None
    return chunk[:-1], chunk[-1]


def _detach_opener(buf: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """行尾的左括号（连同贴着它的样式标记）摘下来，放到下一行开头。整行只有括号时不动。"""
    start = len(buf)
    while start > 0 and STYLE_MARK_RE.fullmatch(buf[start - 1][1]):
        start -= 1
    if start == 0 or buf[start - 1][1] not in _NO_LINE_END:
        return []
    start -= 1
    if not any(not item.isspace() and not STYLE_MARK_RE.fullmatch(item) for _, item in buf[:start]):
        return []
    detached = buf[start:]
    del buf[start:]
    return detached


def _detach_for_punct(buf: list[tuple[int, str]]) -> list[tuple[int, str]] | None:
    """行尾标点放不下时，带上前一个字。行里没有可带的字就不动。"""
    end = len(buf)
    while end > 0 and (
        STYLE_MARK_RE.fullmatch(buf[end - 1][1]) or buf[end - 1][1].isspace() or buf[end - 1][1] in _NO_LINE_START
    ):
        end -= 1
    if end == 0:
        return None
    start = end - 1
    while start > 0 and STYLE_MARK_RE.fullmatch(buf[start - 1][1]):
        start -= 1
    detached = buf[start:]
    del buf[start:]
    return detached


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


# 公式后面直接接文字时要有一个空格。标点和已有空白不补，补第二次也不再加。
_NO_FORMULA_SPACE = frozenset("。，、；：？！）】」』〉》％%.,;:?!)]}>（【「『《〈([{")


_MARK_RUN = r"(?:\{/?[biu]\}|\{z\d+\}|\{/z\})*"
_CJK_CHAR = r"[぀-ヿ㐀-鿿가-힯]"
_LATIN_CHAR = r"[A-Za-z]"


def space_around_latin(text: str) -> str:
    """汉字和夹在中间的拉丁字母两侧各留一个空格。

    序数写成 {i} i-th{/i} 时，空格在斜体里面，后面的汉字又贴着字母。
    把标记内的空格挪出来，边界上没有的补一个，已经有的收成一个。
    """
    if not text or not _CJK_RE.search(text):
        return text
    text = re.sub(r"(\{(?:[biu]|z\d+)\})\s+", r" \1", text)
    text = re.sub(r"\s+(\{/(?:[biu]|z)\})", r"\1 ", text)
    text = re.sub(
        rf"({_CJK_CHAR})[ \t]*({_MARK_RUN})[ \t]*({_LATIN_CHAR})",
        r"\1 \2\3",
        text,
    )
    return re.sub(
        rf"({_LATIN_CHAR})[ \t]*({_MARK_RUN})[ \t]*({_CJK_CHAR})",
        r"\1\2 \3",
        text,
    )


def separate_after_formula(text: str) -> str:
    """公式哨兵后面如果直接接字母、数字或汉字，补一个空格。"""
    if "\x01i\x02" not in text:
        return text
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        matched = _SENTINEL_RE.match(text, index)
        if matched is None:
            out.append(text[index])
            index += 1
            continue
        out.append(matched.group(0))
        index = matched.end()
        while True:
            mark = STYLE_MARK_RE.match(text, index)
            if mark is None:
                break
            out.append(mark.group(0))
            index = mark.end()
        nxt = text[index:index + 1]
        if nxt and not nxt.isspace() and nxt not in _NO_FORMULA_SPACE and _wants_formula_space(nxt):
            out.append(" ")
    return "".join(out)


def peel_leading_space(text: str) -> tuple[str, bool]:
    """去掉样式标记之后的第一个空格。HTML 盒子会吃掉开头的普通空格。"""
    index = 0
    while True:
        mark = STYLE_MARK_RE.match(text, index)
        if mark is None:
            break
        index = mark.end()
    if index < len(text) and text[index] == " ":
        return text[:index] + text[index + 1:], True
    return text, False


def _wants_formula_space(char: str) -> bool:
    if char.isalpha() or char.isdigit():
        return True
    if _CJK_RE.match(char):
        return True
    return unicodedata.east_asian_width(char) in ("W", "F")


def _inside_number(text: str, index: int) -> bool:
    return (
        text[index] in ".,"
        and 0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        atom = _ATOM_RE.match(text, index) or _LINK_RE.match(text, index)
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
            # 2.07、10,000 里夹在数字中间的点和逗号属于这个数，不能从这里断行
            while end < length and (_is_word_char(text[end]) or _inside_number(text, end)):
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
    # l_partkey 这样的标识符不能从下划线处折行
    return char.isalnum() or char in "-'_"


_drawing_fonts: tuple[pymupdf.Font, ...] | None = None


def _drawing_fonts_loaded() -> tuple[pymupdf.Font, ...]:
    """断行要用真正写出的字宽。破折号和 M 按 0.5em 估，一行会多挤出一个字。"""
    global _drawing_fonts
    if _drawing_fonts is None:
        _drawing_fonts = (
            pymupdf.Font("china-ss"),
            pymupdf.Font("heit"),
            pymupdf.Font("hebo"),
        )
    return _drawing_fonts


def measure_width(text: str, em: float, formula_widths: dict[int, float] | None = None) -> float:
    """一段译文占多宽。样式标记为 0，公式哨兵用槽宽，缺槽宽时按一个 em。"""
    widths = formula_widths or {}
    cleaned = strip_style_marks(text)
    total = 0.0
    pos = 0
    for match in _SENTINEL_RE.finditer(cleaned):
        total += sum(_char_width(ch, em) for ch in cleaned[pos:match.start()])
        total += widths.get(int(match.group(1)), em)
        pos = match.end()
    total += sum(_char_width(ch, em) for ch in cleaned[pos:])
    return total


# 断行时每个字都要量一次，一本书要量上百万次；字宽只看字和字号，量过的记下来
@functools.lru_cache(maxsize=65536)
def _char_width(char: str, em: float) -> float:
    if _CJK_RE.match(char) or unicodedata.east_asian_width(char) in ("W", "F"):
        estimate = em
    elif char.isspace():
        estimate = em * 0.33
    else:
        estimate = em * 0.5
    drawn = 0.0
    code = ord(char)
    for font in _drawing_fonts_loaded():
        if font.has_glyph(code):
            drawn = max(drawn, font.text_length(char, em))
    return max(estimate, drawn)


def _token_width(token: str, em: float, formula_widths: dict[int, float]) -> float:
    sentinel = _SENTINEL_RE.fullmatch(token)
    if sentinel:
        return formula_widths.get(int(sentinel.group(1)), em)
    if STYLE_MARK_RE.fullmatch(token):
        return 0.0
    return sum(_char_width(char, em) for char in token)
