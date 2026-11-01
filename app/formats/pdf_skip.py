"""Decide which blocks keep their original text: examples, glosses, code, display formulas,
foreign-language passages and the bibliography."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

from .. import langdetect
from .pdf_blocks import (
    _MARK_ITEM_RE,
    CJK_RE,
    TextBlock,
    _column_left,
    _list_item,
    _paragraph_indent,
    _span_is_math,
    _visual_lines,
)
from .pdf_extract import _EXAMPLE_NUMBER_RE, _RELATION_RE, _block_spans, _mono_span, _plain_words, _vivid, _vivid_share
from .pdf_flow import (
    emphasis_of,
)
from .pdf_layout import (
    ExampleNumber,
    PageGeometry,
    PageScan,
)
from .pdf_math import _BIG_DELIMITER_FONT_RE
from .pdf_merge import _x_overlap
from .pdf_roles import Role
from .pdf_vision import FormulaDetector

# 例句里方括号标注的角标（Arg0、ArgM-TMP、PER）最多几个字
_BRACKET_LABEL_CHARS = 10


def _median_size(blocks: list[TextBlock]) -> float:
    """按字数取中位的字号。按块数取时，索引、图里成百上千个小字的短块会把它拉下来。"""
    total = sum(len(block.text) for block in blocks)
    seen = 0
    for block in sorted(blocks, key=lambda block: block.size):
        seen += len(block.text)
        if seen * 2 >= total:
            return block.size
    return 0.0


# 认外文段落时一段至少要有的字母数。题记里的一个小项只有一两个词，要和旁边的合起来认
_FOREIGN_LETTERS = 40
_FOREIGN_CJK = 8
_HAN_RE = re.compile(r"[㐀-鿿]")
_KANA_HANGUL_RE = re.compile(r"[぀-ヿ가-힯]")


# 例句底下逐行列的「标签: 释义」（Arg1: Proposition、Cf (U2): {John, piano}）
_GLOSS_ROW_RE = re.compile(r"[^\s:：]{1,8}(?:\s?\([^)]*\))?\s?[:：]\s")


def _display_example_ids(blocks: list[TextBlock], geometries: list[PageGeometry], skipped: set[int]) -> set[int]:
    """首行缩进的书里，正文中间单独排开、比段首缩进缩得更深的一行或左缘对齐的几行：讲语言现象时举的例句
    （a clear majority of senators back/VBP the bill、*fair and brutal）、关系的记法
    （Giraffe is-a ruminant is-a ungulate …）。下文拿原句里的词来讲，留原文。
    以冒号收尾的是引出下文的一句（Then repeat iteratively until convergence:），
    字号和正文不一样的是题词的出处，在栏里居中的是图注（Listing 3 | Prompt for …），
    带等号、不等号的是式子的说明（x_i contains a particular prefix (… length ≤ 2)），
    「标签: 释义」一行是给人看的释义，紧接在列表项下面的是这一项折下来的行，都不算。
    上下紧挨着的块和它左缘对齐、或者它和紧挨着的块同一行旁边还有块时，它是表格、逐词对译的例句、悬挂缩进的条目里的一行
    （表头 corpus、意译 “Yao Ming reaches the finals”），由那一类的规则处理"""
    indent = _paragraph_indent(blocks)
    if indent is None:
        return set()
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if not block.side_note and block.clip is None:
            by_page.setdefault(block.page, []).append(block)
    found: set[int] = set()
    for page, page_blocks in by_page.items():
        page_blocks.sort(key=lambda block: block.rect.y0)
        geo = geometries[page] if page < len(geometries) else None
        runs: list[list[TextBlock]] = []
        for block in page_blocks:
            if (
                id(block) in skipped
                or len(block.line_rects) != 1
                or block.index
                or block.entry
                or block.leader is not None
                or _list_item(block.text)
                or block.text.rstrip().endswith((":", "："))
                or _GLOSS_ROW_RE.match(block.text)
                or not block.body_size
                or abs(block.size - block.body_size) > 0.5
            ):
                continue
            edge = _column_left(block.rect, geo)
            if edge is None or not indent + block.size * 0.5 < block.rect.x0 - edge <= block.size * 8:
                continue
            last = runs[-1][-1] if runs else None
            if (
                last is not None
                and abs(block.rect.x0 - last.rect.x0) <= 1
                and -2 <= block.rect.y0 - last.rect.y1 <= block.size * 1.5
            ):
                runs[-1].append(block)
            else:
                runs.append([block])
        for run in runs:
            first, last = run[0], run[-1]
            size = first.size
            edge = _column_left(first.rect, geo)
            right = geo.column_of(first.rect).x1 if len(geo.columns) >= 2 else geo.frame.x1 if geo.frame else None
            if (
                len(run) == 1
                and right is not None
                and abs((first.rect.x0 + first.rect.x1) / 2 - (edge + right) / 2) < size * 0.5
            ):
                continue
            if any(_RELATION_RE.search(block.text) for block in run):
                continue
            members = {id(block) for block in run}
            # 上下紧挨着的行：上面取各块的末行，下面取首行
            above = [
                (other, other.line_rects[-1])
                for other in page_blocks
                if id(other) not in members
                and other.rect.y1 <= first.rect.y0 + 2
                and first.rect.y0 - other.rect.y1 <= size * 1.5
                and _x_overlap(other.rect, first.rect)
            ]
            below = [
                (other, other.line_rects[0])
                for other in page_blocks
                if id(other) not in members
                and other.rect.y0 >= last.rect.y1 - 2
                and other.rect.y0 - last.rect.y1 <= size * 1.5
                and _x_overlap(other.rect, last.rect)
            ]
            if (
                not above
                or not below
                or any(_list_item(other.text) for other, _ in above)
                or any(abs(x0 - first.rect.x0) <= 1 for other, row in above + below for x0 in (other.rect.x0, row.x0))
            ):
                continue
            # 它或紧挨着的块同一行旁边还有块（拼音、对译、表格的另一列），是逐词对译的例句或表格里的一行
            if any(
                other is not block
                and not other.side_note
                and not _x_overlap(other.rect, block.rect)
                and min(other.rect.y1, block.rect.y1) - max(other.rect.y0, block.rect.y0) > block.rect.height * 0.5
                and (right is None or edge <= other.rect.x0 < right)
                for block in run + [other for other, _ in above + below]
                for other in page_blocks
            ):
                continue
            found |= members
    return found


# 模式和例句按颜色对应时至少用几种颜色。正文的链接、强调只用一两种
_KEY_COLORS = 3


def _color_keys(block: TextBlock) -> list[tuple[pymupdf.Rect, tuple[int, ...]]]:
    """块里每一行（同一基线的几截并起来）的框，和行里鲜明颜色先后出现的顺序"""
    rows: list[tuple[pymupdf.Rect, list[int]]] = []
    gap = max(block.size * 0.4, 2.0)
    for rect, spans in zip(block.line_rects, block.span_lines or []):
        colors = [int(span.get("color") or 0) for span in spans if str(span.get("text") or "").strip()]
        colors = [color for color in colors if _vivid(color)]
        if rows and abs(rect.y0 - rows[-1][0].y0) <= gap:
            rows[-1] = (rows[-1][0] | rect, rows[-1][1] + colors)
        else:
            rows.append((pymupdf.Rect(rect), colors))
    return [(rect, tuple(dict.fromkeys(colors))) for rect, colors in rows]


def _color_keyed_ids(blocks: list[TextBlock]) -> set[int]:
    """模式和例句上下两行排，用同几种颜色按同样的先后标出对应的部分
    （PER, POSITION of ORG: 下面是 George Marshall, Secretary of State of the United States，
    PER 和 George Marshall 同是蓝色）。译了语序一变，颜色就对不上了，两行都留原文。"""
    rows: dict[int, list[tuple[pymupdf.Rect, tuple[int, ...], TextBlock]]] = {}
    for block in blocks:
        if block.clip is None and not block.side_note:
            rows.setdefault(block.page, []).extend((rect, key, block) for rect, key in _color_keys(block))
    found: set[int] = set()
    for page_rows in rows.values():
        page_rows.sort(key=lambda row: row[0].y0)
        for (upper, key, block), (lower, other, below) in zip(page_rows, page_rows[1:]):
            if (
                len(key) >= _KEY_COLORS
                and key == other
                and _x_overlap(upper, lower)
                and -2 <= lower.y0 - upper.y1 <= block.size * 1.5
            ):
                found |= {id(block), id(below)}
    return found


def _item_example_ids(blocks: list[TextBlock]) -> set[int]:
    """列表项下面另起一行列的例子（Bag-of-words in M1 and M2 下面的 American, Airlines, Tim …，
    M2: -1 spokesman）：缩在弹点右边，字母大半用鲜明的颜色标着，是从例句里摘出来的词，留原文。
    同一项下面接连几行都算。和条目折下来的行分开见 _cut_item_breaks，也只看弹点起头的条目。"""
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if block.clip is None and not block.side_note:
            by_page.setdefault(block.page, []).append(block)
    found: set[int] = set()
    for page_blocks in by_page.values():
        page_blocks.sort(key=lambda block: (block.rect.y0, block.rect.x0))
        # 已经认出的例子行挂在哪一项下面：项的左缘
        roots: dict[int, float] = {}
        for index, block in enumerate(page_blocks):
            if len(_visual_lines(block)) != 1 or _MARK_ITEM_RE.match(block.text) or not _vivid_letters(block):
                continue
            above = [
                other
                for other in page_blocks[:index]
                if _x_overlap(other.rect, block.rect) and -2 <= block.rect.y0 - other.rect.y1 <= block.size * 1.5
            ]
            if not above:
                continue
            nearest = max(above, key=lambda other: other.rect.y1)
            root = roots.get(id(nearest), nearest.rect.x0 if _MARK_ITEM_RE.match(nearest.text) else None)
            if root is not None and block.rect.x0 > root + block.size * 0.3:
                roots[id(block)] = root
                found.add(id(block))
    return found


def _vivid_letters(block: TextBlock) -> bool:
    """块里一半以上的字母是鲜明的颜色"""
    return _vivid_share(_block_spans(block)) > 0.5


def _arc_figure_ids(blocks: list[TextBlock], scans: Sequence[PageScan]) -> set[int]:
    """用弧线标注的例句（依存弧、注意力弧从词连到词）和弧上圆角框里的标签（nsubj、coreference attention）。
    弧的两端落在句子里某个词的上方，译成中文以后对不上词，留原文。
    句子的上缘接着弧线的下端；标签在圆角框里，框压着弧线。"""
    found: set[int] = set()
    for block in blocks:
        if block.page >= len(scans):
            continue
        scan = scans[block.page]
        if not scan.arcs:
            continue
        rows = block.line_rects or [block.rect]
        if any(
            abs(arc.y1 - row.y0) <= block.size * 0.4 and row.x0 - 2 <= arc.x0 and arc.x1 <= row.x1 + 2
            for arc in scan.arcs
            for row in rows
        ):
            found.add(id(block))
            continue
        for box in scan.round_boxes:
            framed = pymupdf.Rect(box.x0 - 2, box.y0 - 2, box.x1 + 2, box.y1 + 2)
            if framed.contains(block.rect) and any(not (box & arc).is_empty for arc in scan.arcs):
                found.add(id(block))
                break
    return found


def _example_ids(blocks: list[TextBlock], numbers: dict[int, tuple[ExampleNumber, ...]] | None = None) -> set[int]:
    """讲语言现象时举的例句：(24.3) 编号起头、续行悬挂在编号右边的句子，和带两处以上
    [角标 …] 标注的句子（[Arg0 The group] agreed …、[PER Jane] of [ORG United]）。
    下文拿原句里的词来讲，标注也夹在词中间，留原文。
    numbers 是各页行首的例句编号（见 example_numbers）：编号没进块，同一行紧接在它右边的块是例句；
    整行都没成块的（(22.12) fall.01），下面缩进的几行（Arg1: …、Ex1: …）是例句。

    正文以式子编号起头（(2.4) follows from …）的段落，续行回到栏边，不算。"""
    found: set[int] = set()
    heads: list[_ExampleHead] = []
    numbers = numbers or {}
    for block in blocks:
        if block.clip is not None or block.side_note:
            continue
        if _EXAMPLE_NUMBER_RE.match(block.text) and _hanging(block):
            found.add(id(block))
            heads.append(
                _ExampleHead(block.page, block.rect, block.rect.x0, _after_number_x0(block), block.size, id(block))
            )
        elif (number := _example_number_before(block, numbers.get(block.page, ()))) is not None:
            found.add(id(block))
            heads.append(_ExampleHead(block.page, block.rect, number.rect.x0, block.rect.x0, block.size, id(block)))
        elif _labeled_brackets(block) >= 2:
            found.add(id(block))
    for page, page_numbers in numbers.items():
        on_page = [block for block in blocks if block.page == page]
        for number in page_numbers:
            if not any(
                min(block.rect.y1, number.row.y1) - max(block.rect.y0, number.row.y0) > number.row.height * 0.5
                and block.rect.x0 < number.row.x1
                and block.rect.x1 > number.row.x0
                for block in on_page
            ):
                heads.append(_ExampleHead(page, number.row, number.rect.x0, number.text_x0, 0.0, None))
    return found | _example_tails(blocks, heads, numbers, _paragraph_indent(blocks))


@dataclass(frozen=True)
class _ExampleHead:
    """编号例句的第一行。block_id 是它所在的块，整行没成块时为 None，size 这时取下面第一块的字号。
    number_x0 是编号的左缘，text_x0 是编号后面的句子从哪里起笔"""

    page: int
    rect: pymupdf.Rect
    number_x0: float
    text_x0: float
    size: float
    block_id: int | None


def _example_number_before(block: TextBlock, numbers: tuple[ExampleNumber, ...]) -> ExampleNumber | None:
    """块的第一行紧接在一个单独的例句编号右边，同一行"""
    first = block.line_rects[0] if block.line_rects else block.rect
    return next(
        (
            number
            for number in numbers
            if number.rect.x1 <= first.x0
            and first.x0 - number.rect.x1 <= block.size * 5
            and min(number.rect.y1, first.y1) - max(number.rect.y0, first.y0)
            > min(number.rect.height, first.height) * 0.5
        ),
        None,
    )


def _example_tails(
    blocks: list[TextBlock],
    heads: list[_ExampleHead],
    numbers: dict[int, tuple[ExampleNumber, ...]],
    indent: float | None = None,
) -> set[int]:
    """编号例句的后几行另成了块（(24.14) 的第二句、例句里换行的 wine.、IT、JA 几种语言的对照、
    框架文件条目下面的 Arg0: …、Ex1: …、对译下面的意译 “Yao Ming reaches the finals”）：紧贴在例句下面、字号一样、起笔在编号和句子之间偏右的块，也是这个例句。
    正文回到编号的左缘起笔；indent 是正文段首缩进多宽，缩进这么多起笔的是下一段。到下一个编号为止。"""
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if block.clip is None and not block.side_note:
            by_page.setdefault(block.page, []).append(block)
    for page_blocks in by_page.values():
        page_blocks.sort(key=lambda block: block.rect.y0)
    head_ids = {head.block_id for head in heads}
    tails: set[int] = set()
    for head in heads:
        left = (head.number_x0 + head.text_x0) / 2
        stop = min(
            (
                number.rect.y0
                for number in numbers.get(head.page, ())
                if number.rect.y0 > head.rect.y0 + number.rect.height * 0.5
                and number.rect.x0 < head.rect.x1
                and number.rect.x1 > head.rect.x0
            ),
            default=math.inf,
        )
        last, size = head.rect, head.size
        for block in by_page.get(head.page, ()):
            if id(block) == head.block_id or block.rect.y0 < head.rect.y0 + head.rect.height * 0.5:
                continue
            if block.rect.x0 >= head.rect.x1 or block.rect.x1 <= head.rect.x0:
                continue  # 另一栏
            size = size or block.size
            if indent is not None:
                left = max(left, head.number_x0 + indent + size * 0.5)
            # 例句里空一行接着排的（Arg 几行下面的 Ex1:）和上一行对齐起笔
            gap = size * (1.6 if last is not head.rect and abs(block.rect.x0 - last.x0) <= 1 else 0.6)
            if (
                block.rect.y0 - last.y1 > gap
                or block.rect.y0 >= stop - 1
                or block.rect.x0 < left
                or abs(block.size - size) > size * 0.1
                or _EXAMPLE_NUMBER_RE.match(block.text)
                or id(block) in head_ids
            ):
                break
            tails.add(id(block))
            last = block.rect
    return tails


def _after_number_x0(block: TextBlock) -> float:
    """例句编号后面第一个字的横坐标。编号和句子在同一个 span 里时按字数比例估。"""
    matched = _EXAMPLE_NUMBER_RE.match(block.text)
    # 编号和句子可能分在同一高度的两行里
    lines = block.span_lines or [[]]
    first = [span for span in lines[0] if str(span.get("text") or "").strip() and span.get("bbox") is not None]
    top = pymupdf.Rect(first[0]["bbox"]) if first else None
    spans = sorted(
        (
            span
            for line in lines
            for span in line
            if str(span.get("text") or "").strip()
            and span.get("bbox") is not None
            and top is not None
            and abs(pymupdf.Rect(span["bbox"]).y1 - top.y1) <= block.size * 0.3
        ),
        key=lambda span: span["bbox"][0],
    )
    if matched is None or not spans:
        return block.rect.x0 + block.size * 3
    label = len(matched.group(0)[:-1].rstrip())
    seen = 0
    for span in spans:
        text = str(span["text"])
        start = len(text) - len(text.lstrip())
        stripped = text.strip()
        if seen + len(stripped) > label:
            box = pymupdf.Rect(span["bbox"])
            offset = max(label - seen, 0) + start
            while offset < len(text) and text[offset].isspace():
                offset += 1
            return box.x0 + box.width * offset / max(len(text), 1)
        seen += len(stripped)
    return block.rect.x1


def _hanging(block: TextBlock) -> bool:
    """首行以下各行都比首行靠右起笔。只有一行的也算。"""
    starts = [min(rect.x0 for rect in row) for row in sorted(_visual_lines(block), key=lambda row: row[0].y0)]
    return all(x0 > starts[0] + block.size for x0 in starts[1:])


def _labeled_brackets(block: TextBlock) -> int:
    """左方括号后面紧跟一个小一号的字母标签、标签后面还有词的次数。
    层号 W[1] 的上标是数字，[EOS] 这样整个括号里只有一个词，都不算。"""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    count = 0
    for left, label, after in zip(spans, spans[1:], spans[2:]):
        text = str(label.get("text") or "").strip()
        if (
            str(left.get("text") or "").rstrip().endswith("[")
            and float(label.get("size") or 0) < block.size * 0.95
            and len(text) <= _BRACKET_LABEL_CHARS
            and " " not in text
            and sum(c.isalpha() for c in text) >= 2
            and not str(after.get("text") or "").lstrip().startswith("]")
        ):
            count += 1
    return count


# 论元的几种实现方式逐行列出：AGENT/Subject, THEME/Object, INSTRUMENT/PPwith
_ROLE_PATTERN_RE = re.compile(r"[A-Z]{3,}/\w+(?:,\s*[A-Z]{3,}/\w+)*")


def _role_pattern_ids(blocks: list[TextBlock]) -> set[int]:
    """逐行列出的角色记法（AGENT/Subject, THEME/Object）。下文拿记法里的角色名来讲，留原文。
    词下面标着角色的例句按对译认，见 interlinear_glosses。"""
    return {
        id(block)
        for block in blocks
        if block.clip is None
        and not block.side_note
        and len(block.line_rects) <= 3
        and _ROLE_PATTERN_RE.fullmatch(block.text.strip())
    }


def _concordance_ids(blocks: list[TextBlock]) -> set[int]:
    """逐行列出的语料检索结果：每行中间一个粗体的词，各行的这个词左缘对齐，两边是它在语料里的前后文
    （is traditionally followed by | cherry | pie, a traditional dessert）。
    前后文是截下来的半句，几行连起来不成句，下文拿中间的词来讲，留原文。"""
    found: set[int] = set()
    for block in blocks:
        if block.clip is not None or block.side_note:
            continue
        spans = sorted(
            (
                span
                for line in block.span_lines or []
                for span in line
                if str(span.get("text") or "").strip() and span.get("origin") is not None
            ),
            key=lambda span: float(span["origin"][1]),
        )
        rows: list[list[dict]] = []
        for span in spans:
            if rows and float(span["origin"][1]) - float(rows[-1][0]["origin"][1]) <= block.size * 0.3:
                rows[-1].append(span)
            else:
                rows.append([span])
        if len(rows) < 3:
            continue
        lefts = []
        for row in rows:
            row.sort(key=lambda span: float(span["bbox"][0]))
            bold = [
                i
                for i, span in enumerate(row)
                if emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)).bold
            ]
            # 粗体的词连成一截，左右都有前后文
            if not bold or bold != list(range(bold[0], bold[-1] + 1)) or bold[0] == 0 or bold[-1] == len(row) - 1:
                break
            lefts.append(float(row[bold[0]]["bbox"][0]))
        else:
            if max(lefts) - min(lefts) <= block.size * 0.2:
                found.add(id(block))
    return found


def _foreign_ids(blocks: list[TextBlock]) -> set[int]:
    """和全文不是同一种语言的段落：英文书里的西班牙文题记、德文引文、巴斯克语例句。
    作者有意留着原文，旁边多半已经给了译文，留原文。

    题记分两列排，一项只有一两个词，单独认不准。同一页上字号、颜色相同、挨在一起的块
    合起来认，整组是外文就整组留原文；正文里夹着的一段外文按自己认。
    字母不够、置信度不够的不算，正文语言认不出来时不做判断。
    """
    doc_lang = langdetect.detect_document([block.text for block in blocks])
    if not doc_lang:
        return set()

    cjk_doc = doc_lang.split("-")[0] in ("zh", "ja", "ko")

    def foreign(text: str) -> bool:
        # 一个汉字顶好几个字母：西文书里十来个字、不夹西文的一句中文（题词的原句）字母数不够也认得出
        if not cjk_doc and len(CJK_RE.findall(text)) >= _FOREIGN_CJK and not re.search(r"[A-Za-z]", text):
            return True
        lang = langdetect.detect(text, min_letters=_FOREIGN_LETTERS)
        return bool(lang) and lang.split("-")[0] != doc_lang.split("-")[0]

    found = {id(block) for block in blocks if foreign(block.text)}
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        if not block.side_note:
            by_page.setdefault(block.page, []).append(block)
    for page_blocks in by_page.values():
        groups: list[list[TextBlock]] = []
        for block in page_blocks:
            reach = block.rect + (-block.size, -block.size * 0.6, block.size, block.size * 0.6)
            near = [
                group
                for group in groups
                if any(
                    abs(other.size - block.size) < 0.3 and other.color == block.color and reach.intersects(other.rect)
                    for other in group
                )
            ]
            merged = [block]
            for group in near:
                groups.remove(group)
                merged.extend(group)
            groups.append(merged)
        for group in groups:
            if len(group) > 1 and foreign("\n".join(block.text for block in group)):
                found.update(id(block) for block in group)
    if not cjk_doc:
        found |= _rendering_ids(blocks, found)
    return found


def _han_line(block: TextBlock) -> bool:
    """块的第一行是一句中文：八个以上汉字，不夹西文字母、假名、谚文。下面可以跟着拼音"""
    first = "".join(str(span.get("text") or "") for span in block.span_lines[0]) if block.span_lines else block.text
    return (
        len(_HAN_RE.findall(first)) >= _FOREIGN_CJK
        and not re.search(r"[A-Za-z]", first)
        and not _KANA_HANGUL_RE.search(first)
    )


def _rendering_ids(blocks: list[TextBlock], found: set[int]) -> set[int]:
    """中文原句（见 _han_line）旁边作者给的西文译文：同一行右边起笔的（题词 荃者所以在鱼 | Nets are for fish;），
    或者紧贴在下面、左缘对齐的（例句框里汉字、拼音下面的 In my backyard, …），连同下面左缘对齐的续行。
    原句就在旁边，再译回中文没有意义，留原文。"""
    sources = [block for block in blocks if _han_line(block)]
    out: set[int] = set()
    for source in sources:
        for block in blocks:
            if block.page != source.page or id(block) in found or id(block) in out:
                continue
            r, s = block.rect, source.rect
            beside = (
                min(r.y1, s.y1) - max(r.y0, s.y0) > min(r.height, s.height) * 0.5
                and 0 <= r.x0 - s.x1 <= 3 * source.size
            )
            below = 0 <= r.y0 - s.y1 + 1 <= source.size * 0.5 and abs(r.x0 - s.x0) <= source.size
            if beside or below:
                out.add(id(block))
    # 续行：紧贴在下面、左缘对齐、同字号
    grown = True
    while grown:
        grown = False
        for block in blocks:
            if id(block) in found or id(block) in out:
                continue
            for other in blocks:
                if (
                    id(other) in out
                    and other.page == block.page
                    and abs(block.rect.x0 - other.rect.x0) <= 1
                    and 0 <= block.rect.y0 - other.rect.y1 + 1 <= block.size * 0.5
                    and abs(block.size - other.size) <= 0.3
                ):
                    out.add(id(block))
                    grown = True
                    break
    return out


def _latin_chars(text: str) -> int:
    """数等宽字时只数拉丁字母和数字。汉字、假名和全角标点也带等宽标志，不计数。"""
    return sum(1 for ch in text if ch.isalnum() and not CJK_RE.match(ch))


def _code_panel_ids(blocks: list[TextBlock], scans: list[PageScan]) -> set[int]:
    """一片底色上的字八成以上是等宽字：程序、码位表、命令行输出。整片留原字。

    按整片算：片里夹着的注释行、行号单独看不够八成，跟着整片留原字。
    底色上还有不少正文字体的不受影响。
    """
    by_page: dict[int, list[TextBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    found: set[int] = set()
    for pno, page_blocks in by_page.items():
        for area in scans[pno].shades:
            inside = [block for block in page_blocks if (block.rect & area).get_area() > block.rect.get_area() * 0.6]
            spans = [span for block in inside for span in _block_spans(block) if not _span_is_math(span)]
            total = mono = 0
            for span in spans:
                count = _latin_chars(str(span["text"]))
                total += count
                if _mono_span(span):
                    mono += count
            if total and mono >= total * 0.8:
                found.update(id(block) for block in inside)
    return found


# 大号字体里正文行内也用的字形：宽帽子、宽波浪（b–g），最小的两号根号（p、q）。
# 伸长括号、求和积分号才说明是单独排的式子。文中号的求和号也算：单独排的分式里分母用的就是它
_BIG_GLYPH_RE = re.compile(r"[^\sb-gpq]")
# 公式外的词数到这个数就是成段的句子。行内积分号也用这种字体，右栏、列表续行又都离开版心左缘
_DISPLAY_FORMULA_MAX_WORDS = 24
# 块的面积有这么多落在公式检测模型的行间公式框里，才算行间公式（同版面模型的 _COVER）
_ISOLATED_COVER = 0.6
# 冠词、系动词、介词、连词、代词这类虚词。行间公式框里有这么多个虚词的是说明句
_FUNCTION_WORDS = frozenset(
    """
a an the is are was were be been being am do does did has have had can could will would shall should may might must
of to in for with by from on at as into onto over under about between than that which who whom whose what where
when why how this these those it its they their them we our us you your he his she her him i my not no nor
and or but if then so each every any all some such much many
""".split()
)
_SENTENCE_FUNCTION_WORDS = 3


def _function_words(text: str) -> int:
    return sum(1 for word in re.findall(r"[A-Za-z]+", text) if word.lower() in _FUNCTION_WORDS)


def _display_formula_ids(blocks: list[TextBlock], geometries: list[PageGeometry]) -> set[int]:
    """用正文字体排的行间公式：分段函数的大括号把几行式子并成一组，式子里的
    min、ins-cost(target[j]) 看起来像词，按词数认不出来。带伸长定界符、又不从版心左缘
    起笔的块是公式；落在它框里的块（大括号右边的条件行）也是。"""
    found: list[TextBlock] = []
    for block in blocks:
        geo = geometries[block.page] if block.page < len(geometries) else None
        frame = geo.frame if geo is not None else None
        if frame is None:
            continue
        # 居中排的宽式子左边只空出一点，和右边空的一样多；成段的正文写到右缘
        left, right = block.rect.x0 - frame.x0, frame.x1 - block.rect.x1
        centered = left > block.size and abs(left - right) <= block.size
        if left <= block.size * 2 and not centered:
            continue
        if (
            any(
                _BIG_DELIMITER_FONT_RE.search(str(span.get("font") or ""))
                and _BIG_GLYPH_RE.search(str(span.get("text") or ""))
                for line in block.span_lines or []
                for span in line
            )
            and _plain_words(block) < _DISPLAY_FORMULA_MAX_WORDS
        ):
            found.append(block)
    ids = {id(block) for block in found}
    for block in blocks:
        area = block.rect.width * block.rect.height
        if id(block) in ids or area < 1:
            continue
        for formula in found:
            if formula.page != block.page:
                continue
            hit = block.rect & (formula.rect + (-2, -2, 2, 2))
            if not hit.is_empty and hit.width * hit.height >= area * 0.7:
                ids.add(id(block))
                break
    return ids


def _isolated_formula_ids(
    blocks: list[TextBlock],
    formulas: FormulaDetector | None,
    roles: Sequence[Role] | None = None,
) -> set[int]:
    """公式检测模型框成行间公式的块留原文：块的面积大半落在框里。Word 排的行间公式用正文字体，
    按字体认不出来，没有伸长的定界符时 _display_formula_ids 也认不到。

    框里成句的说明照翻（(where c is the correct class)、L(ŷ, y) = How much ŷ differs from the true y）。
    成不成句看虚词，不看正文词数：式子里的词是实词和变量名（perplexity_A(T) = P_A(red red red red blue)），
    数正文词会把它当成句子，译成「困惑度 = P(红红红红蓝)」。
    标题、图注不算：幻灯片上的大字标题（Phoenix + Calcite）模型也常框成行间公式。"""
    if formulas is None:
        return set()
    ids: set[int] = set()
    for index, block in enumerate(blocks):
        area = block.rect.width * block.rect.height
        if area < 1 or roles is not None and roles[index] in (Role.HEADING, Role.CAPTION):
            continue
        if _function_words(block.text) >= _SENTENCE_FUNCTION_WORDS:
            continue
        for rect in formulas.isolated.get(block.page, ()):
            hit = block.rect & rect
            if not hit.is_empty and hit.width * hit.height >= area * _ISOLATED_COVER:
                ids.add(id(block))
                break
    return ids


def _in_gloss(rect: pymupdf.Rect, glosses: tuple[pymupdf.Rect, ...]) -> bool:
    area = rect.width * rect.height
    for gloss in glosses:
        hit = rect & (gloss + (-1, -1, 1, 1))
        if area >= 1 and not hit.is_empty and hit.width * hit.height >= area * 0.7:
            return True
    return False


def _gloss_ids(blocks: list[TextBlock], glosses: dict[int, tuple[pymupdf.Rect, ...]]) -> set[int]:
    """落在逐词对译两行里的块。"""
    return {id(block) for block in blocks if _in_gloss(block.rect, glosses.get(block.page, ()))}


_BIBLIO_CITE_RE = re.compile(r"\((?:19|20)\d{2}\)|et al\.|doi:|https?://", re.I)
# 行尾 “, 2016.” 在正文里太常见。确认进入时必须同时是编号条目，或带 arXiv/CoRR/doi。
_BIBLIO_YEAR_END_RE = re.compile(r",\s*(?:19|20)\d{2}\s*[.]?\s*$")
# 老数据库论文收在页码：ICDE 1995, 190-200. / VLDB 2000: 496-505。同样只认编号条目。
_BIBLIO_PAGES_END_RE = re.compile(r"(?:19|20)\d{2}\s*[,:]\s*(?:pp?\.\s*)?\d+\s*[-–]\s*\d+\s*[.]?\s*$")
_BIBLIO_TRAIL_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*[.]?\s*$|年版\s*[.。]?\s*$")
_BIBLIO_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# 卷期续行：2020, 3(2) / Vol. 3 / Volume 3 / pp. 1-10 / 第3期。
# Volume 后面必须是阿拉伯数字，避免 “volume were”。
_BIBLIO_VOLUME_RE = re.compile(
    r"^(?:19|20)\d{2}\s*[,，(（]\s*(?:[\d(（]|[Vv]ol\.?\s*\d|[Vv]olume\.?\s*\d|[Nn]o\.?(?:\s|\d)|[Pp]p?\.?(?:\s|\d)|第)"
)
# 只认缩写 Vol. 后的罗马数字。不含 M/D，否则 mix / mill 会当成卷号。
_ROMAN_VOL_RE = re.compile(
    r"(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})(?![A-Za-z])",
    re.I,
)
_BIBLIO_VOL_LABEL_RE = re.compile(r"^(?:19|20)\d{2}\s*[,，(（]\s*[Vv]ol\.?\s*")
# 同字号退出用。数字后要是期号、页码或行尾；No/pp 后要是数字；「第」后要是「期」。
# 行首只是年份的下一章（2020, 1. Introduction / 第1章）不能靠这条留下。
_BIBLIO_EXIT_VOLUME_RE = re.compile(
    r"^(?:19|20)\d{2}\s*[,，(（]\s*(?:"
    r"\d+\s*(?:[（(]|[:：]\s*\d|[,，]\s*\d|[.。]?\s*$)"
    r"|\d+\s*[）)]"
    r"|[Vv]ol\.?\s*\d|[Vv]olume\.?\s*\d|[Nn]o\.?\s*\d|[Pp]p?\.?\s*\d"
    r"|第\s*\d+\s*期"
    r")"
)
# 出版社行可以收在 2019。 / 2019年。 / 年版。出版社后面还有字的是正文。
_BIBLIO_IMPRINT_END_RE = re.compile(r"(?:(?:19|20)\d{2}年?|年版)\s*[.。]?\s*$")
_BIBLIO_MARK_RE = re.compile(r"\barXiv\b|\bCoRR\b|doi:|https?://", re.I)
# 题名后的著录位置才算，如 研究[J]. 刊名。句末单独的 [M]。 是正文，不是引文。
_BIBLIO_TYPE_MARK_RE = re.compile(r"(?<=\S)\[[JMDCNRP]\]\s*[.。]\s*\S")
# [1]张三 中间常没有空格。1. 仍要空格，避免把 1.2 当成编号。
# 老论文常用作者缩写加年份的标签：[CS94]、[GBLP95]、[TPCD]。点阵字 OCR 常把开头的 [ 认丢，只剩 ESS]。
_BIBLIO_LEAD_RE = re.compile(r"^(?:\[\d+\]\s*|\d{1,3}[.)]\s+|\[?[A-Z][A-Za-z]{0,8}\+?\d{0,4}[a-z]?\](?:\s+|$))")
_BIBLIO_WRAP_RE = re.compile(r"^[a-zà-öø-ÿ]")
# 弱信号：只有编号条目才算。“这项研究发表于某出版社。”是正文。
_BIBLIO_CN_PUB_RE = re.compile(r"出版社|年版")
# 一串作者名：Y. Bai, A. Jones, K. Ndousse, …；Alan Akbik, Duncan Blythe, and Roland Vollgraf.
# 标题和正文不会以连着两三个「名, 名,」起头
_AUTHOR_NAME = r"(?:(?:[A-Z]\.[\s-]?){1,3}[A-Z][\w'’-]+|[A-Z][a-z][\w'’-]*(?:\s(?:[A-Z]\.|[A-Z][\w'’-]+)){1,3})"
_AUTHOR_LIST_RE = re.compile(rf"^{_AUTHOR_NAME}(?:,\s+(?:and\s+)?{_AUTHOR_NAME}){{2,}}(?:,|\.|$)")
# 一行只有作者名：逗号隔开的两个以上名字，每个至少两个词（名、中间名缩写可以不带点、van der 这类小写前缀），
# 最后一个折行时只剩名（前面至少两个完整的名字）。老字体把重音拆成单独的符号（S´ebastien）
_NAME_WORD = r"[A-Z][\w´`¨˜ˆ'’.-]*"
_FULL_NAME = rf"{_NAME_WORD}(?:\s+(?:(?:van|von|der|den|de|da|di|du|del|la|le)\s+)*{_NAME_WORD})+"
_NAMES_LINE_RE = re.compile(
    rf"^{_FULL_NAME}(?:,\s+(?:and\s+)?{_FULL_NAME})*,\s+(?:and\s+)?{_FULL_NAME}[,.]?$"
    rf"|^{_FULL_NAME}(?:,\s+(?:and\s+)?{_FULL_NAME})+,\s+(?:and\s+)?{_NAME_WORD}[,.]?$"
)
# ACL 体：作者. 2018. 题名. In Proceedings …, pages 1–10. 年份在作者后面单独一句，后面是出处
_BIBLIO_ACL_YEAR_RE = re.compile(r"(?:^|\.\s)(?:19|20)\d{2}[a-z]?\.\s+\S")
_BIBLIO_VENUE_RE = re.compile(r"\bIn\s+(?:Proc|[A-Z]{2,})|\bpages\s+\d+\s*[–-]\s*\d+|\bJournal\b")


def _biblio_heading_like(block: TextBlock, text: str, median: float) -> bool:
    if not text or len(text) > 120:
        return False
    if block.size > median + 0.4:
        return True
    return bool(block.bold and block.size + 0.4 >= median and len(text) < 80)


def _biblio_entry_citation(text: str) -> bool:
    """标题后面这行像引文，才进入参考文献。正文里的 2017 / arXiv / 出版社 / [J] 不算。"""
    if _BIBLIO_CITE_RE.search(text) or _author_entry(text):
        return True
    # 类型标记 + 年份才是著录条目；“出处见某书[M]。后续还有讨论。”是正文
    if _BIBLIO_TYPE_MARK_RE.search(text) and _BIBLIO_YEAR_RE.search(text):
        return True
    lead = bool(_BIBLIO_LEAD_RE.match(text))
    mark = bool(_BIBLIO_MARK_RE.search(text))
    if (_BIBLIO_YEAR_END_RE.search(text) or _BIBLIO_PAGES_END_RE.search(text)) and (lead or mark):
        return True
    return lead and (mark or bool(_BIBLIO_CN_PUB_RE.search(text)))


def _author_entry(text: str) -> bool:
    """一串作者名起头，或者 ACL 体的「作者. 年份. 题名. In 出处」。参考文献的字号常比正文大一点，
    作者那一行按字号像标题，要靠这个认出来。"""
    if _AUTHOR_LIST_RE.match(text):
        return True
    return bool(_BIBLIO_ACL_YEAR_RE.search(text) and _BIBLIO_VENUE_RE.search(text))


def _vol_roman_continuation(text: str) -> bool:
    """Vol. III / vol. xii 是卷号。单词 mix / mill 不是。"""
    head = _BIBLIO_VOL_LABEL_RE.match(text)
    if head is None:
        return False
    return _ROMAN_VOL_RE.match(text, head.end()) is not None


def _biblio_exit_continuation(text: str) -> bool:
    """同字号时仍算引文续行。行首像年份的下一章不算。"""
    if _BIBLIO_EXIT_VOLUME_RE.match(text) or _vol_roman_continuation(text):
        return True
    return bool(_BIBLIO_CN_PUB_RE.search(text) and _BIBLIO_IMPRINT_END_RE.search(text))


def _biblio_year_continuation(text: str) -> bool:
    """类型标记的下一块仍是著录。后来随便出现的年份不算。"""
    if _BIBLIO_VOLUME_RE.match(text) or _vol_roman_continuation(text):
        return True
    # 出版社行以年份或「年版」收尾。正文提到出版社后还有下文的不算。
    return bool(_BIBLIO_CN_PUB_RE.search(text) and _BIBLIO_IMPRINT_END_RE.search(text))


def _biblio_citation_line(block: TextBlock, text: str, start_size: float, median: float) -> bool:
    """已经在参考文献里：这行是引文或换行前的条目，不能当成下一章。"""
    if _BIBLIO_CITE_RE.search(text) or _BIBLIO_MARK_RE.search(text) or _author_entry(text):
        return True
    heading = _biblio_heading_like(block, text, median)
    short_heading = heading and len(text) < 40
    # 比起始标题更大，或同字号短标题，仍是下一章。超过 80 字的加粗行先让给出类型标记和行尾年份。
    if block.size > start_size + 0.5 or short_heading or (block.bold and heading):
        return False
    if _BIBLIO_TYPE_MARK_RE.search(text) or _BIBLIO_TRAIL_YEAR_RE.search(text):
        return True
    # 真正的卷期、出版社续行和标题同字号也不是下一章。
    if _biblio_exit_continuation(text):
        return True
    if not _BIBLIO_LEAD_RE.match(text):
        return False
    return True


def biblio_skips(blocks: list[TextBlock]) -> list[bool]:
    """标记参考文献章节的块。标题本身不跳过，章节正文保留原文。

    页眉和目录里的同名短行不能把进入字号改小；同等字号的引文也不是下一章标题。
    """
    from .html_blocks import BIBLIO_HEADING_RE, BIBLIO_SHORT_LABEL_RE, BIBLIO_SUBHEADING_RE

    if not blocks:
        return []
    # 按字数取中位：按块数取时，正文接上续行少了几块，中位就掉到小字号上，参考文献的标题认不出来
    median = median_all = _median_size(blocks)
    prepared = [(b, re.sub(r"\s+", " ", b.text).strip()) for b in blocks]

    def smaller_subsection(block: TextBlock, text: str, origin: float) -> bool:
        # 同字号或更大的 Books / 论文是下一章，不能当成内部小标题。
        if len(text) >= 40 or block.size + 0.4 >= origin:
            return False
        if BIBLIO_SUBHEADING_RE.match(text):
            return True
        return bool(BIBLIO_SHORT_LABEL_RE.match(text) and _biblio_heading_like(block, text, median))

    def citations_follow(idx: int) -> bool:
        origin = prepared[idx][0].size
        pending_lead = False
        pending_type = False
        # 作者-年份体的条目不带编号和括号年份（Abadi, M., A. Agarwal, … 2001.），连着几条才算
        author_leads = 0
        for block, text in prepared[idx + 1 : idx + 16]:
            # Endnotes / 参考文献是另一节的标题，必须停住，不能把进入字号抬高
            if len(text) < 40 and BIBLIO_HEADING_RE.match(text):
                return False
            if smaller_subsection(block, text, origin):
                pending_lead = False
                pending_type = False
                continue
            # 像标题的行即使含 [J] 或年份也不是引文，先停住
            if _biblio_heading_like(block, text, median) and len(text) < 80:
                return False
            # 编号和出版社、类型标记和年份，都常被拆成相邻两块。只看下一块。
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
        # 像不像标题按条目的字号比：有的参考文献整体比正文大一号，按全文的中位字号，条目也像标题
        median = max(median_all, entry_size)
        if not text or len(text) > 120 or smaller_subsection(block, text, start_size):
            return False
        # 悬挂缩进里单独成行的编号 [ESS]、[12] 是条目的标签，不是下一章
        if _BIBLIO_LEAD_RE.fullmatch(text + " "):
            return False
        # 句中标点起头的、没有空格的一整串网址，是上一条折过来的半截
        # （https: 后面折行，下一行是 //github.com/…；…-Abstract-Datasets_and_Benchmarks_Track.html.）
        if text[:1] in "/.,;:)]}-–_" or (" " not in text and re.search(r"[/_]|\.[A-Za-z]{2,}\.?$", text)):
            return False
        if block.size + 0.5 < start_size * 0.8:
            return False
        if _biblio_citation_line(block, text, start_size, median):
            return False
        # 引文里单独一行 Journal 不是下一章
        if len(text) < 40 and BIBLIO_SHORT_LABEL_RE.match(text) and not _biblio_heading_like(block, text, median):
            return False
        # 编号条目的换行续行以小写字母开头，不是下一章。
        # 句号收尾的短行（Databases. White Paper.）也是条目断开的一截：点阵字 OCR 后字号都统一了，靠字号分不开
        if prev_numbered and (_BIBLIO_WRAP_RE.match(text) or text.endswith(".")):
            return False
        if _biblio_heading_like(block, text, median):
            return True
        # 加粗长标题（heading_like 的加粗分支有 80 字上限）
        if block.bold and block.size + 0.4 >= median:
            return True
        # 条目首行折断时只剩一串作者名（Ahmed M. Alaa, Zaid Ahmad, and Mark van der Laan.），
        # 接在作者名后面的短行是题名（API is enough: Conformal prediction for large lan-），都不是下一章
        names = _NAMES_LINE_RE.match(text) or _NAMES_LINE_RE.match(prev_text)
        return len(text) < 60 and block.size >= start_size - 0.5 and not names

    skips: list[bool] = []
    in_biblio = False
    start_size = 0.0
    prev_numbered = False
    # 已经认进参考文献的条目字号
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
        # 标题比正文小的排法（ICASSP 整节参考文献都小一号）：加粗、不比下面几条引文小，也是这一节的标题
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
            pass  # 引文单词不是标题，编号条目的续行还没结束
        else:
            prev_numbered = False
    for index in _headless_bibliography(prepared) | _headless_numbered_bibliography(prepared, median):
        skips[index] = True
    return skips


# 作者-年份体的著录条目开头：Ahia, O.,  /  Arkadiev, P. M. 2020.  /  Clark, H. H. and  /  Smith, J. (2020)
_AUTHOR_LEAD_RE = re.compile(
    r"^[A-Z][^\s,.;:()]{1,30}(?:[\s-][A-Za-z][^\s,.;:()]{1,30}){0,3},\s"
    r"(?:[A-Z]\.(?:\s?-?[A-Z]\.)*|[A-Z][a-z]+)"
    r"(?:,|\s(?:and|&)\s|\s\(?(?:19|20)\d{2})"
)
_AUTHOR_LEADS_TO_ENTER = 2  # 参考文献标题后面至少几块以作者-年份条目起头
_ENTRY_WINDOW = 8  # 往后看几块
_ENTRY_LEADS = 4  # 其中至少几块是条目开头
_ENTRY_GAP = 4  # 连续这么多块没有新条目，参考文献就结束了


def _headless_bibliography(prepared: list[tuple[TextBlock, str]]) -> set[int]:
    """没有标题的参考文献：书的章末常紧接着习题排，不写 References。

    一连串同字号、以「姓, 名缩写.」起头的条目就是。正文里偶尔有一句这样开头，凑不够数。
    中间夹着的页眉、条目断开的续行一并算进去。字号明显变大的块是下一节，到此为止。
    """
    leads = [bool(_AUTHOR_LEAD_RE.match(text)) for _, text in prepared]
    found: set[int] = set()
    index = 0
    while index < len(prepared):
        size = prepared[index][0].size

        # 条目都是同一个字号。正文里这样开头的一句话字号不同，不和后面的条目凑数
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
        # 最后一条的续行：字号不变大
        while end < cursor and prepared[end][0].size <= size + 0.5 and not leads[end]:
            end += 1
        found.update(range(index, end))
        index = max(end, index + 1)
    return found


# 编号条目的开头：[12] A. Buonanno。上一条的续行和下一条挤在一块里时，编号跟在句点后面
_NUMBERED_LEAD_RE = re.compile(r"(?:^|[.)]\s)\[(\d{1,3})\]\s+\S")


def _headless_numbered_bibliography(prepared: list[tuple[TextBlock, str]], median: float) -> set[int]:
    """没有标题的编号参考文献（revtex 的期刊版式不印 References）：从 [1] 起头的块开始，
    往后几块里编号挨着排到 [4] 以上就是。编号连不上、隔了几块没有新条目、字号变大，
    或者碰到标题（SUPPLEMENTAL MATERIAL、另一栏的 Appendix F），到此为止。
    标题后面以下一个编号起头的块是另一栏接着排的条目，从那里接着认。
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
        first = 1 if opens else resume
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
        # 最后一条的续行：不再有编号
        while end < cursor and not numbers[end]:
            end += 1
        found.update(range(index, end))
        resume = expected
        index = end
    return found
