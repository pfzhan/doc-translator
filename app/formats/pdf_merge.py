"""Find where a paragraph continues: on the next page, in the next column or in the next
layout-model region, and pair the blocks into one translation unit."""

import re
from collections import Counter
from collections.abc import Callable, Sequence

import pymupdf

from .pdf_blocks import (
    _AFFIL_MARK_RE,
    _END_MARK_RE,
    _EQUATION_TAG_RE,
    _LABEL_START_RE,
    _NUMBERED_HEAD_RE,
    _SRC_HEAD_NUM_RE,
    CJK_RE,
    TextBlock,
    _ends_sentence,
    _indents_paragraphs,
    _list_item,
    _numbered_line,
    _same_page_line_pair,
    _span_is_math,
    _styled_edge,
    _visual_lines,
)
from .pdf_extract import (
    _FOOTNOTE_LEAD_RE,
    _MID_ABBREV_RE,
    _caption_label,
    _closes_first,
    _earlier_on_page,
    _is_float_text,
    _is_line_tail,
    _line_tail_blocked,
    _line_tail_match,
    _opens_with_math,
    _unclosed_bracket,
)
from .pdf_flow import (
    emphasis_of,
)
from .pdf_layout import (
    PageGeometry,
)
from .pdf_math import _prose_units
from .pdf_roles import is_size_heading


def _preview_kind(block: TextBlock, median: float) -> str:
    """短块里，明显大于正文的、以及略大于正文的粗体，当作小标题。

    只看粗体不够：作者名和图注标签也常是粗体，但字号不超过正文。
    有版面角色时，标题以角色为准，这里只留字号兜底。
    """
    return "h2" if is_size_heading(block.text, block.size, block.bold, median) else "p"


# 句末标点：判断页面末尾的块是不是被页边界切断的段落
_SENT_END_PUNCT = tuple(".!?:;…。！？；：\"'”’)]}》")


def _different_columns(
    geometries: list[PageGeometry] | None,
    left: TextBlock,
    right: TextBlock,
) -> bool:
    """两页里只要有一页是多栏、且两块不在同一栏序号，就不是同一栏的跨页续段。

    右栏页尾接到下一页左栏是另一条规则，不走这里。单栏页面序号都是 0。
    """
    if not geometries or left.page >= len(geometries) or right.page >= len(geometries):
        return False
    a, b = geometries[left.page], geometries[right.page]
    if len(a.columns) < 2 and len(b.columns) < 2:
        return False
    return a.index_of(left.rect) != b.index_of(right.rect)


def _cjk_page_continue(prev: TextBlock, nxt: TextBlock) -> bool:
    """跨页中文续段没有大小写。短标签、缩进的新段、图注标签不接。"""
    head = nxt.text.lstrip()[:1]
    if not head or CJK_RE.search(head) is None or _caption_label(nxt.text):
        return False
    if len(CJK_RE.findall(prev.text)) < 4:
        return False
    return abs(nxt.rect.x0 - prev.rect.x0) <= prev.size * 1.2


def _is_page_footnote(block: TextBlock, median: float) -> bool:
    if block.size + 0.6 >= median:
        return False
    return _FOOTNOTE_LEAD_RE.match(block.text.lstrip()) is not None


def _can_continue(
    prev: TextBlock,
    nxt: TextBlock,
    median: float,
    halt_ids: set[int] | None,
    allow_capital: bool = False,
    flush: bool = False,
    at_edge: bool = True,
) -> bool:
    """页末块没有句末标点，页首块以小写、数字或中文开头，且不是标题。

    换栏时允许大写开头：没写完的句子常常断在 SQL 这种专有名词上。
    同一栏里的普通换行仍不放开，那和下一句分不开。
    表格的格子各是各的：格子的列看起来也像栏，上一列的末格不能接到下一列的表头。
    flush 是段落首行缩进的文档里，页首、栏首（或者栏顶的图表下面）齐平起笔的一段（见 _flush_start）：
    新段都缩进，齐平的是上一段的续段，上一块恰好在句末断开、下一块大写起头也接上。
    at_edge 是下一块的首行从本栏正文的左缘起笔（见 _at_text_edge）。
    """
    if prev.clip is not None or nxt.clip is not None:
        return False
    # 栏底整行斜体、粗体的短行是小标题（E. Energy-Detection Characterization），不往下一栏接
    if len(_visual_lines(prev)) == 1 and len(prev.text.split()) <= 8 and _all_styled(prev):
        return False
    # 圆括号起头的看里面的字：(RMS) loudness、(0, 1), which。(3)、(a) 这样的编号不算。
    # 方括号、引号里的不看：[v], [z] 是音标，“[M]achines 是另起的引文
    opening = nxt.text.lstrip()
    if not _EQUATION_TAG_RE.match(opening) and not _list_item(opening):
        opening = opening.lstrip("(")
    head = opening[:1]
    numbered_heading = bool(_SRC_HEAD_NUM_RE.match(nxt.text.lstrip()))
    if halt_ids is None:
        is_heading = _preview_kind(nxt, median) != "p"
    else:
        is_heading = id(nxt) in halt_ids
    # 以行内式子起头的（… minimise a divergence, | 𝐷(⋅||⋅) between …）也是接着写：式子的大写字母不是句首。
    # 数学字体的项目符号（• We introduce …）不算
    math_start = _opens_with_math(nxt) and not _list_item(nxt.text)
    continues = head.islower() or head.isdigit() or _cjk_page_continue(prev, nxt) or math_start
    # 行尾断开的词（fine-）后半截是小写，换栏也不放开大写：右栏顶上大写起头的是表头、标题
    broken_word = bool(re.search(r"[a-z]-$", prev.text.rstrip()))
    if allow_capital and not broken_word and head.isalpha() and head.isupper() and len(prev.text.split()) >= 3:
        continues = True
    # 下一块以括注起头，括注后面是标点或小写：(Tables S32–S35), the port …、(§5.3). Urban …
    lead = nxt.text.lstrip()
    if (
        re.match(r"\([^()]{1,60}\)\s*(?:[.,;:]|[a-z])", lead)
        and not _EQUATION_TAG_RE.match(lead)
        and not _list_item(lead)
    ):
        continues = True
    # 句中的缩写（Mao et al. | 2026）不是句末。分号、冒号、里面没有句号的右括号后面小写接着写的是同一句
    # （… on BC trials; | an explanation …、… GOP (4.750 vs. 4.733) | and lower …）。
    # 冒号后面的行间公式跳过了不译时，下一页的 where 接在冒号后面（… scoring function: | where U, W …）。
    # 冒号下面缩进、居中排的是引出的例句、式子（… from that instance: | is traditionally followed by cherry pie），
    # 不接：接着写的半句从正文的左缘起笔
    tail = _END_MARK_RE.sub("", prev.text)
    closers = "\"'”’)]}》"
    weak = (
        tail.endswith((";", "；"))
        or (at_edge and tail.endswith((":", "：")))
        or (tail.endswith(tuple(closers)) and not tail.rstrip(closers).endswith(_SENT_END_PUNCT))
    )
    ended = (
        tail.endswith(_SENT_END_PUNCT)
        and not _MID_ABBREV_RE.search(prev.text.rstrip())
        and not (weak and head.islower())
    )
    # 括号还没合上，下一块在第一句里合上它：(Fu et al., 2024b; | Mao et al., 2024) share …
    if _unclosed_bracket(prev.text) and _closes_first(nxt.text):
        continues, ended = True, False
    # 只看正文：小一号的图注、表注下面齐平起笔的正文不是它们的续段。
    # 只放开大写起头的：句末下面小写起头的（restric- | tion from …）是别的块断开的后半截。
    # 页末是编号起头的例句（(24.31) IBM announced …）时，下一页齐平起笔的是例句前面那段的续段，不接在例句后面
    if (
        flush
        and head.isupper()
        and len(prev.text.split()) >= _FLUSH_HOST_WORDS
        and abs(prev.size - nxt.size) <= prev.size * 0.05
        and _flush_start(nxt)
        and not _numbered_line(prev.text)
    ):
        continues, ended = True, False
    return bool(
        not numbered_heading
        and not is_heading
        and continues
        and not ended
        and not _is_float_text(nxt.text)
        and nxt.size <= prev.size * 1.4
    )


def _flush_start(block: TextBlock) -> bool:
    """块是排满栏宽的一段，首行和下面的行一样从栏边起笔。行首换了粗体、斜体的小标题，
    带编号的标签、小节号和列表项本来就齐平起笔，不算"""
    rows = sorted(_visual_lines(block), key=lambda row: row[0].y0)
    if len(rows) < 2 or _styled_edge(block, 0) or _list_item(block.text):
        return False
    if _LABEL_START_RE.match(block.text) or _NUMBERED_HEAD_RE.match(block.text):
        return False
    lefts = [min(rect.x0 for rect in row) for row in rows]
    rights = [max(rect.x1 for rect in row) for row in rows]
    return max(lefts) - min(lefts) <= 0.3 * block.size and rights[0] >= max(rights[1:]) - 0.5 * block.size


def _at_text_edge(head: TextBlock, page_blocks: list[TextBlock]) -> bool:
    """块的首行从本栏正文的左缘起笔。左缘取同一页上和它左右交叠的多行块（连同它自己）里，
    除首行以外各行最常见的左边，首行缩进、悬挂缩进的段落也量得准。图注的续行居中排，不算。
    没有能量的行时按在左缘算"""
    lefts = Counter(
        round(min(rect.x0 for rect in row))
        for block in page_blocks
        if block.rect.x0 < head.rect.x1 and block.rect.x1 > head.rect.x0 and not _is_float_text(block.text)
        for row in sorted(_visual_lines(block), key=lambda row: row[0].y0)[1:]
    )
    if not lefts:
        return True
    rows = sorted(_visual_lines(head), key=lambda row: row[0].y0)
    first = min(rect.x0 for rect in rows[0]) if rows else head.rect.x0
    return abs(first - lefts.most_common(1)[0][0]) <= 0.3 * head.size


def _resumes(head: TextBlock, page_blocks: list[TextBlock], skip_ids: set[int], passed: set[int]) -> bool:
    """块是接着上一栏、上一页往下写的位置：正上方没有块（栏顶、页首），或者正上方是跳过的图、式子、
    小一号的表格格子，或者图注（passed 里是页眉页脚、图和图注）。正上方是一行正文字号的短行
    （漏认的小标题 B.3.1. RL Data、提示词框里的 PROMPT）时，下面是新起的一段"""
    above = [
        block
        for block in page_blocks
        if block is not head
        and block.rect.y1 <= head.rect.y0 + 1
        and block.rect.x0 < head.rect.x1
        and block.rect.x1 > head.rect.x0
    ]
    if not above:
        return True
    near = max(above, key=lambda block: block.rect.y1)
    return id(near) in skip_ids or id(near) in passed or _is_float_text(near.text) or near.size < head.size * 0.9


# 小节号悬在栏边左面最多这么多字宽
_HANGING_HEAD_EM = 3
# 齐平起笔的续段接上去，上一块至少要有这么多词（成段的正文）
_FLUSH_HOST_WORDS = 8


def _page_edge_blocks(
    blocks: list[TextBlock],
    median: float,
    ignored: set[int],
) -> list[TextBlock]:
    """页首页尾只看正文。小一号的编号脚注留在原处翻译，不占跨页位置。"""
    return [
        block
        for block in blocks
        if block.size >= median * 0.8 and id(block) not in ignored and not _is_page_footnote(block, median)
    ]


def _column_index(geo: PageGeometry, block: TextBlock) -> int:
    return geo.index_of(block.rect)


def _in_column(geo: PageGeometry, block: TextBlock) -> bool:
    return len(geo.columns) < 2 or not geo.spans_columns(block.rect)


def _column_page_end(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    """同栏跨页看的页末。多栏页上的跨栏块不占这个位置，否则会抢走右栏末行。"""
    return _column_edge(body, geometries, last=True)


def _column_page_start(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
) -> TextBlock | None:
    return _column_edge(body, geometries, last=False)


def _column_edge(
    body: list[TextBlock],
    geometries: list[PageGeometry] | None,
    last: bool,
) -> TextBlock | None:
    if not body:
        return None
    if not geometries:
        return body[-1] if last else body[0]
    ordered = reversed(body) if last else body
    for block in ordered:
        if block.page >= len(geometries) or _in_column(geometries[block.page], block):
            return block
    return None


def _is_last_in_right_column(
    block: TextBlock,
    page_blocks: list[TextBlock],
    geo: PageGeometry,
) -> bool:
    if len(geo.columns) < 2 or not _in_column(geo, block):
        return False
    if _column_index(geo, block) != len(geo.columns) - 1:
        return False
    return not any(
        other is not block
        and _in_column(geo, other)
        and _column_index(geo, other) == len(geo.columns) - 1
        and other.rect.y0 > block.rect.y0 + 1
        for other in page_blocks
    )


def _left_column_head(page_blocks: list[TextBlock], geo: PageGeometry) -> TextBlock | None:
    if len(geo.columns) < 2:
        return None
    heads = [
        block
        for block in page_blocks
        if _in_column(geo, block)
        and _column_index(geo, block) == 0
        and not any(
            other is not block
            and _in_column(geo, other)
            and _column_index(geo, other) == 0
            and other.rect.y0 < block.rect.y0 - 1
            for other in page_blocks
        )
    ]
    if not heads:
        return None
    return min(heads, key=lambda block: (block.rect.y0, block.rect.x0))


_CODE_LINE_RE = re.compile(r"^(?:<|\d+\s+0x[0-9a-fA-F])")
# 这么多词的块是正文，不看字母占比
_PROSE_WORDS = 8
_CLUSTER_GAP = 28.0


def _is_code_line(text: str) -> bool:
    """XML、堆栈和图内标签。它们贴在栏顶时不能占续句的位置。

    短续句（well.、tomer）仍是正文。没有空格、又夹着数字或符号的才是标签。
    数字多的结果段（and lower mean WER (0.009 vs. 0.013); paired …）字母占比低，词数够的仍是正文。
    """
    stripped = text.strip()
    if not stripped or _CODE_LINE_RE.match(stripped):
        return True
    if stripped[:1] in "<{" or "::" in stripped or '="' in stripped or "/>" in stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < len(stripped) * 0.45 and _prose_units(stripped) < _PROSE_WORDS:
        return True
    return " " not in stripped and re.search(r"[\d%$<>!#_]", stripped) is not None


def _continuation_prose(
    block: TextBlock,
    median: float,
    skip_ids: set[int],
    halt_ids: set[int] | None,
    ignored: set[int],
) -> bool:
    if id(block) in skip_ids or id(block) in ignored:
        return False
    # 目录、索引条目不接上下文，但照样占着栏顶栏底：跳过它们，栏底一个没页码的词头
    # 会接到下一栏中间的条目上。词表的格子和图一样跳过
    listed = (block.leader is not None or block.entry) and not block.grid
    if halt_ids is not None and id(block) in halt_ids and not listed:
        return False
    if block.size < median * 0.8 or _is_page_footnote(block, median):
        return False
    if _AFFIL_MARK_RE.match(block.text.lstrip()):
        return False
    return not _is_float_text(block.text) and not _is_code_line(block.text)


def _column_anchors(blocks: list[TextBlock]) -> list[float]:
    """正文左缘。至少两块落在同一左缘才算一栏，单行右半句不另起栏。
    页上已经有这样的一栏时，另一栏只有一整段（左栏上面是缩进排的摘要），三行以上的一块也算。

    下一栏从上一栏的右缘以外起笔。缩进排的例句、引文落在上一栏中间，不另起栏：
    不然它的最后一块会被当成页末，把下一页的段首接过来。
    """
    if not blocks:
        return []
    groups: list[list[TextBlock]] = []
    for block in sorted(blocks, key=lambda block: block.rect.x0):
        if groups and block.rect.x0 - groups[-1][0].rect.x0 < 8:
            groups[-1].append(block)
        else:
            groups.append([block])
    merged: list[float] = []
    right = None
    paired = any(len(group) >= 2 for group in groups)
    for group in groups:
        if len(group) < 2 and not (paired and len(_visual_lines(group[0])) >= 3):
            continue
        anchor = sum(block.rect.x0 for block in group) / len(group)
        if merged and (anchor - merged[-1] < _CLUSTER_GAP or (right is not None and anchor < right - 4)):
            continue
        merged.append(anchor)
        edges = sorted(block.rect.x1 for block in group)
        right = edges[len(edges) // 2]
    return merged


def _anchor_index(block: TextBlock, anchors: list[float]) -> int | None:
    if not anchors:
        return None
    best = min(range(len(anchors)), key=lambda index: abs(block.rect.x0 - anchors[index]))
    if abs(block.rect.x0 - anchors[best]) <= max(block.size * 4, 36):
        return best
    return None


def _assigned_columns(
    blocks: list[TextBlock],
    anchors: list[float],
) -> tuple[list[list[TextBlock]], list[TextBlock]]:
    columns: list[list[TextBlock]] = [[] for _ in anchors]
    orphans: list[TextBlock] = []
    for block in blocks:
        index = _anchor_index(block, anchors)
        if index is None:
            orphans.append(block)
        else:
            columns[index].append(block)
    return columns, orphans


def _column_body(column: list[TextBlock]) -> list[TextBlock]:
    """栏末的小一号脚注不占续句的位置。整栏都是这个字号时全部留下。

    正文字号不算表格的格子：栏顶一张表一格一块，按块数会压过正文。
    """
    if not column:
        return []
    sizes = sorted(block.size for block in column if block.clip is None) or sorted(block.size for block in column)
    body_size = sizes[len(sizes) // 2]
    body = [block for block in column if block.size + 0.6 >= body_size]
    return body or column


def _column_top(blocks: list[TextBlock]) -> TextBlock:
    return min(blocks, key=lambda block: (block.rect.y0, block.rect.x0))


def _column_bottom(blocks: list[TextBlock]) -> TextBlock:
    return max(blocks, key=lambda block: (block.rect.y1, block.rect.x0))


def _fragment_host(tail: TextBlock, orphans: list[TextBlock]) -> TextBlock:
    """栏末正文已经句号结束时，同一行右侧没写完的半句才是续句的起点。

    用行框而不是块的并集。并集会盖住右侧那半句，半句就不算贴在行尾了。
    """
    if not _ends_sentence(tail.text):
        return tail
    lines = tail.line_rects or [tail.rect]
    for orphan in orphans:
        if orphan.page != tail.page or len(orphan.line_rects) > 1:
            continue
        if _ends_sentence(orphan.text):
            continue
        height = orphan.rect.height or orphan.size
        for line in lines:
            overlap = min(orphan.rect.y1, line.y1) - max(orphan.rect.y0, line.y0)
            if overlap <= min(height, line.height) * 0.5:
                continue
            gap = orphan.rect.x0 - line.x1
            if -2 <= gap < 16:
                return orphan
    return tail


def _column_wrap(host: TextBlock, tail: TextBlock) -> bool:
    """左缘差出一栏，才是换栏。同一栏页首的大写仍是下一段。"""
    return abs(host.rect.x0 - tail.rect.x0) > max(host.size * 4, 36.0)


def _try_column_pair(
    pairs: dict[int, TextBlock],
    tails: set[int],
    host: TextBlock | None,
    tail: TextBlock | None,
    median: float,
    halt_ids: set[int] | None,
    skip_ids: set[int],
    allow_capital: bool = True,
    flush: bool = False,
    at_edge: bool = True,
) -> None:
    if host is None or tail is None or host is tail:
        return
    if id(host) in pairs or id(host) in tails or id(tail) in pairs or id(tail) in tails:
        return
    if id(host) in skip_ids or id(tail) in skip_ids:
        return
    if halt_ids is not None and id(host) in halt_ids:
        return
    if abs(host.size - tail.size) > max(host.size, 1.0) * 0.25:
        return
    if not _can_continue(host, tail, median, halt_ids, allow_capital=allow_capital, flush=flush, at_edge=at_edge):
        return
    pairs[id(host)] = tail
    tails.add(id(tail))


def _float_contents(column: list[TextBlock], page_blocks: list[TextBlock]) -> set[int]:
    """栏顶一张表（图）的字：从栏顶到下面的图注（Table 7: …）之间，每一行都没排到栏的右缘。
    表头、行名都是短行；段落的行排满栏宽，栏顶是段落时不算。
    表的图注排在表上面、整栏都紧接在图注下面的，从栏顶往下数到第一块排满栏宽的段落为止。
    图的图注排在图下面，下面的短行是正文（式子后面的 subject to），不算。
    栏的右缘按两端对齐的正文行最常收在的地方算，栏里比正文宽的图不算。"""
    if not column:
        return set()
    x0 = min(block.rect.x0 for block in column)
    x1 = max(block.rect.x1 for block in column)
    ends = Counter(round(line.x1) for block in column for line in (block.line_rects or [block.rect]))
    right, count = ends.most_common(1)[0]
    if count < 3:
        return set()
    captions = [
        block for block in page_blocks if _is_float_text(block.text) and block.rect.x0 < x1 and block.rect.x1 > x0
    ]
    if not captions:
        return set()
    caption = min(captions, key=lambda block: block.rect.y0)

    def short(block: TextBlock) -> bool:
        return all(line.x1 < right - block.size * 2 for line in (block.line_rects or [block.rect]))

    top = min(column, key=lambda block: block.rect.y0)
    over = [block for block in captions if block.rect.y1 <= top.rect.y0 + 1]
    near = max(over, key=lambda block: block.rect.y1) if over else None
    if near is not None and re.match(r"(?i)tab", near.text.lstrip()) and top.rect.y0 - near.rect.y1 <= 2 * top.size:
        contents: set[int] = set()
        for block in sorted(column, key=lambda block: block.rect.y0):
            if not short(block):
                return contents
            contents.add(id(block))
        return set()
    above = [block for block in column if block.rect.y1 <= caption.rect.y0 + 1]
    if not above or not all(short(block) for block in above):
        return set()
    return {id(block) for block in above}


def _column_continuations(
    by_page: dict[int, list[TextBlock]],
    median: float,
    skip_ids: set[int],
    halt_ids: set[int] | None,
    ignored: set[int],
    aside_ids: set[int] = frozenset(),
    indents: bool = False,
) -> dict[int, TextBlock]:
    """左栏页尾接到右栏页首，右栏页尾接到下一页左栏。

    栏顶的图、清单和图注跳过，续句在它们下面，栏顶的表也是（见 _float_contents）。几何栏不准时也按左缘分，
    不依赖 reads_in_columns。同一栏跨页不放开大写：另一栏没形成左缘时，
    页末左栏不能把下一页新起的一段吞进去。段落首行缩进的文档里齐平起笔的除外（indents，见 _can_continue）。
    """
    pairs: dict[int, TextBlock] = {}
    tails: set[int] = set()
    pages = sorted(by_page)
    prose: dict[int, list[TextBlock]] = {
        pno: [block for block in blocks if _continuation_prose(block, median, skip_ids, halt_ids, ignored)]
        for pno, blocks in by_page.items()
    }
    columns_of: dict[int, list[list[TextBlock]]] = {}
    orphans_of: dict[int, list[TextBlock]] = {}
    for pno, blocks in prose.items():
        columns, orphans = _assigned_columns(blocks, _column_anchors(blocks))
        columns_of[pno] = [column for column in columns if column]
        orphans_of[pno] = orphans

    def head_of(pno: int, column: list[TextBlock]) -> TextBlock | None:
        contents = _float_contents(column, by_page.get(pno) or [])
        body = _column_body([block for block in column if id(block) not in contents])
        # 一行居中排在不归哪一栏的一行底下的，是分栏排的作者、单位（Shixiang Zhu | Carnegie Mellon University），
        # 不是栏顶。表格的格子、横跨两栏的标题下面就是栏顶
        orphans = orphans_of.get(pno) or []
        body = [
            block
            for block in body
            if not (
                len(_visual_lines(block)) == 1
                and any(
                    0 <= block.rect.y0 - other.rect.y1 <= 1.5 * block.size
                    and abs((block.rect.x0 + block.rect.x1) - (other.rect.x0 + other.rect.x1)) / 2 <= block.size
                    for other in orphans
                )
            )
        ]
        if not body:
            return None
        head = _column_top(body)
        return None if heading_above(pno, column, head) else head

    def heading_above(pno: int, column: list[TextBlock], head: TextBlock) -> bool:
        """栏顶正上方紧挨着本栏的小标题（A. Materials），这一栏是新的一节，没有续句。
        小节号悬在栏边左面的（1.4 Related work）也算。
        横跨几栏的大标题不算，图里认成标题的大字（中间还隔着图里别的字）也不算"""
        if halt_ids is None:
            return False
        above = [
            block
            for block in by_page.get(pno) or []
            if id(block) not in ignored
            and block is not head
            and block.rect.y1 <= head.rect.y0 + 1
            and block.rect.x0 < head.rect.x1
            and block.rect.x1 > head.rect.x0
        ]
        if not above:
            return False
        near = max(above, key=lambda block: block.rect.y1)
        x0 = min(block.rect.x0 for block in column)
        x1 = max(block.rect.x1 for block in column)
        return (
            id(near) in halt_ids
            and id(near) not in aside_ids
            and id(near) not in skip_ids
            and not near.side_note
            and near.leader is None
            and not near.entry
            and near.rect.x0 >= x0 - _HANGING_HEAD_EM * near.size
            and near.rect.x1 <= x1 + near.size
        )

    def host_of(pno: int, column: list[TextBlock]) -> TextBlock | None:
        body = _column_body(column)
        if not body:
            return None
        return _fragment_host(_column_bottom(body), orphans_of.get(pno) or [])

    for pno in pages:
        columns = columns_of.get(pno) or []
        for index in range(len(columns) - 1):
            host = host_of(pno, columns[index])
            # 续句不比上半句的字大。右栏顶上比正文大的是两栏上面的标题、署名，续句在它们下面
            head = head_of(
                pno, [block for block in columns[index + 1] if host is None or block.size <= host.size * 1.1]
            )
            # 左栏写到底、再跳到右栏顶，才是换栏续句。并排开头不接。
            if host is None or head is None or host.rect.y0 <= head.rect.y0 + host.size:
                continue
            if host.rect.x1 > head.rect.x0 + 4:
                continue
            _try_column_pair(
                pairs,
                tails,
                host,
                head,
                median,
                halt_ids,
                skip_ids,
                flush=indents and _resumes(head, by_page.get(pno) or [], skip_ids, ignored | aside_ids),
                at_edge=_at_text_edge(head, by_page.get(pno) or []),
            )
        later = _next_text_page(by_page, pno, median, skip_ids, ignored)
        nxt = columns_of.get(later) or []
        if not columns or not nxt:
            continue
        host = host_of(pno, columns[-1])
        # 右栏没分出来（右栏的块都没归栏）时最后一栏不是最右一栏：右边还排着正文的不是页末。
        # 下面还有从左到右盖过这一块、字不比它小的正文也不是页末：图里的表格有一列格子左缘对齐，认成了一栏，
        # 底下接着通栏的段落。页脚注的字小，行间公式旁的标注窄，都不挡
        if host is not None and any(
            block.rect.x0 >= host.rect.x1 - host.size
            and block.rect.y1 > host.rect.y0
            or block is not host
            and block.rect.y0 >= host.rect.y1 - 1
            and block.size >= host.size - 0.5
            and block.rect.x0 <= host.rect.x0 + host.size
            and block.rect.x1 >= host.rect.x1 - host.size
            for block in prose[pno]
        ):
            continue
        head = head_of(later, nxt[0])
        _try_column_pair(
            pairs,
            tails,
            host,
            head,
            median,
            halt_ids,
            skip_ids,
            allow_capital=bool(host and head and _column_wrap(host, head)),
            flush=bool(indents and head and _resumes(head, by_page.get(later) or [], skip_ids, ignored | aside_ids)),
            at_edge=head is None or _at_text_edge(head, by_page.get(later) or []),
        )
    return pairs


def _region_joins(
    units: list[list[TextBlock]],
    skip_ids: set[int],
    halt_ids: set[int],
    regions_of: Callable[[int], list[pymupdf.Rect]],
) -> list[list[TextBlock]]:
    """开了版面模型时：同一页上前后两个单元（中间可以隔着跳过的块）落在模型框出的同一段正文里，
    切口上又有式子，就是被式子切开的同一段（… with{式子} For a Borel set … | 1{式子} is the indicator …）。
    上一块要是成段的正文：模型也会把一串作者名、算法的一行行框成一块（见 _region_prose）。
    模型常把一串列表框成一块，上一块句末收住、下一块大写起头的不并（… under fixed conditions. | We show that …）。
    标题、图注、页眉页脚、列表项、行首的粗斜体标签也不并。没有模型时原样返回。"""
    pages: dict[int, list[pymupdf.Rect]] = {}

    def region(block: TextBlock) -> pymupdf.Rect | None:
        if block.page not in pages:
            pages[block.page] = regions_of(block.page)
        index = _region_index(block.rect, pages[block.page])
        return None if index is None else pages[block.page][index]

    out: list[list[TextBlock]] = []
    last: int | None = None
    for unit in units:
        if all(id(block) in skip_ids for block in unit):
            out.append(unit)
            continue
        if last is not None and _region_continues(out[last][-1], unit[0], halt_ids):
            box = region(out[last][-1])
            if box is not None and box == region(unit[0]) and _region_prose(out[last][-1], box):
                out[last] = [*out[last], *unit]
                continue
        out.append(unit)
        last = len(out) - 1
    return out


def _region_prose(block: TextBlock, box: pymupdf.Rect) -> bool:
    """框里成段的正文：有几个小写的词（一串作者名没有），不止一行，或者单独一行撑满框宽。
    幻灯片的要点、算法的一步步、提示词模板都是一行一行的短句，模型也框成一块。"""
    if len(re.findall(r"\b[a-z]{3,}\b", block.text)) < _REGION_WORDS:
        return False
    return len(_visual_lines(block)) >= 2 or (
        block.rect.x0 - box.x0 <= 1.5 * block.size and box.x1 - block.rect.x1 <= 1.5 * block.size
    )


def _region_continues(prev: TextBlock, nxt: TextBlock, halt_ids: set[int]) -> bool:
    if prev.page != nxt.page or id(prev) in halt_ids or id(nxt) in halt_ids:
        return False
    if _list_item(nxt.text) or _LABEL_START_RE.match(nxt.text) or _styled_edge(nxt, 0) or _ALGO_LINE_RE.match(nxt.text):
        return False
    # 切口上有式子：上一块的末行、下一块的首行里有式子，或者中间空出一行行间公式
    if not (_math_on_line(prev, -1) or _math_on_line(nxt, 0) or nxt.rect.y0 - prev.rect.y1 >= _REGION_GAP * prev.size):
        return False
    letter = re.search(r"[A-Za-z]", nxt.text)
    # 引号收住的也算句末：题词下面大写起头的一行是出处（… Location, Location” | Randall Munroe）
    ended = _END_MARK_RE.sub("", prev.text).rstrip().endswith(tuple('.!?;:。！？；：”"'))
    return not (ended and letter is not None and letter.group().isupper())


def _region_index(rect: pymupdf.Rect, regions: list[pymupdf.Rect]) -> int | None:
    """块的面积大半落在哪个框里。"""
    area = rect.width * rect.height
    if area < 1:
        return None
    best, best_share = None, _REGION_COVER
    for index, box in enumerate(regions):
        hit = rect & box
        if hit.is_empty:
            continue
        share = hit.width * hit.height / area
        if share >= best_share:
            best, best_share = index, share
    return best


# 块的面积有这么多落在模型的正文框里，才算框里的块
_REGION_COVER = 0.5
# 模型并段时上一块至少有这么多小写的词
_REGION_WORDS = 3
# 上下两块之间空出这么多（字号的倍数）就是中间排了一行行间公式
_REGION_GAP = 1.5
# 算法的行号：6: until converged
_ALGO_LINE_RE = re.compile(r"\d{1,3}:\s")
# 往后最多跳过这么多页整页的图表，去找续段
_FLOAT_PAGES = 2
# 分图号：(C)、(b)、C.
_PANEL_LABEL_RE = re.compile(r"\(?[A-Za-z]\)\s|[A-Z]\.\s")


def _next_text_page(
    by_page: dict[int, list[TextBlock]],
    pno: int,
    median: float,
    skip_ids: set[int],
    ignored: set[int],
) -> int:
    """pno 之后第一个有正文的页。整页只有图、图注（连同紧贴在图注下面的分图说明）和页眉页脚的页跳过：
    上一页没写完的段落接在它后面那一页。分图说明以 (C) 这样的分图号起头，或者比正文小一号；
    图注下面紧接着的正文字号的段落是正文（上一页断开的那段接在这里）"""
    later = pno + 1
    while later <= pno + _FLOAT_PAGES and _float_page(by_page.get(later) or [], median, skip_ids, ignored):
        later += 1
    return later if later <= pno + _FLOAT_PAGES else pno + 1


def _float_page(blocks: list[TextBlock], median: float, skip_ids: set[int], ignored: set[int]) -> bool:
    prose = [
        block
        for block in blocks
        if id(block) not in ignored and id(block) not in skip_ids and block.size >= median * 0.8
    ]
    captions = [block for block in prose if _is_float_text(block.text)]
    for block in sorted(prose, key=lambda block: block.rect.y0):
        if block in captions:
            continue
        panel = _PANEL_LABEL_RE.match(block.text.lstrip()) is not None or block.size < median * 0.95
        if panel and any(
            0 <= block.rect.y0 - caption.rect.y1 <= 1.5 * block.size
            and abs(block.rect.x0 - caption.rect.x0) <= block.size
            for caption in captions
        ):
            captions.append(block)
            continue
        return False
    return True


def _cross_page_units(
    blocks: list[TextBlock],
    median: float,
    skip_ids: set[int] = frozenset(),
    geometries: list[PageGeometry] | None = None,
    halt_ids: set[int] | None = None,
    ignore_ids: set[int] | None = None,
    aside_ids: set[int] = frozenset(),
) -> list[list[TextBlock]]:
    """把跨页续段两两合并成翻译单元（返回块列表的列表，每单元 1~2 块）。

    上一页最后一个正文块不以句末标点结尾、下一页第一个块以小写/数字开头且不像标题时，
    视为被页边界切断的同一段。页脚脚注（字号明显小于正文，或小一号且以编号、链接开头）和指定跳过的块不参与。
    同一栏的续段照旧合并。多栏时，只再放开「本页最右栏的末块接到下页最左栏的首块」。
    ignore_ids 是页眉页脚，不占页末、也不挡页首。跨栏标题和整表同样不占栏首栏尾。
    aside_ids 是图和图注：下一页先排了一张图，被切断的那段接在图注下面。
    """
    by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    ignored = ignore_ids or set()
    indents = _indents_paragraphs(blocks)
    # 不传几何时保持旧的页末/页首规则。传了几何才按左缘把换栏续句接上。
    # 共基线的半句不依赖栏几何：下一行仍和它成对送翻，只是不并成一个外框。
    extra: dict[int, TextBlock] = {}
    taken: set[int] = set()
    for host_id, tail in _shared_line_pairs(blocks).items():
        if host_id in skip_ids or id(tail) in skip_ids:
            continue
        if host_id in ignored or id(tail) in ignored:
            continue
        if halt_ids is not None and (host_id in halt_ids or id(tail) in halt_ids):
            continue
        extra[host_id] = tail
        taken.add(host_id)
        taken.add(id(tail))
    if geometries is not None:
        for host_id, tail in _column_continuations(
            by_page,
            median,
            skip_ids,
            halt_ids,
            ignored,
            aside_ids,
            indents,
        ).items():
            if host_id in taken or id(tail) in taken:
                continue
            extra[host_id] = tail
            taken.add(host_id)
            taken.add(id(tail))

    units: list[list[TextBlock]] = []
    pages = sorted(by_page)
    used: set[int] = {id(tail) for tail in extra.values()}
    for pno in pages:
        body = _page_edge_blocks(by_page[pno], median, ignored | aside_ids)
        later = _next_text_page(by_page, pno, median, skip_ids, ignored)
        nxt = by_page.get(later) or []
        nxt_body = _page_edge_blocks(nxt, median, ignored | aside_ids)
        first = _column_page_start(nxt_body, geometries)
        end = _column_page_end(body, geometries)
        head = _left_column_head(nxt_body, geometries[later]) if geometries and later < len(geometries) else None
        for b in by_page[pno]:
            if id(b) in used:
                continue
            if id(b) in extra:
                tail = extra[id(b)]
                units.append([b, tail])
                used.add(id(b))
                used.add(id(tail))
                continue
            if halt_ids is not None and id(b) in halt_ids:
                units.append([b])
                used.add(id(b))
                continue
            # 已经和同页另一块成对的块，不能再接到上一页页末，否则它那一半不在任何单元里
            same = bool(
                end is not None
                and b is end
                and first is not None
                and id(first) not in used
                and id(first) not in extra
                and id(b) not in skip_ids
                and id(first) not in skip_ids
                and _can_continue(
                    b,
                    first,
                    median,
                    halt_ids,
                    flush=indents and _resumes(first, nxt, skip_ids, ignored | aside_ids),
                    at_edge=_at_text_edge(first, nxt),
                )
                and not _different_columns(geometries, b, first)
            )
            cross_target = head if head is not None and id(head) not in used and id(head) not in extra else None
            cross = bool(
                not same
                and cross_target is not None
                and geometries is not None
                and pno < len(geometries)
                and _is_last_in_right_column(b, body, geometries[pno])
                and id(b) not in skip_ids
                and id(cross_target) not in skip_ids
                and _can_continue(
                    b,
                    cross_target,
                    median,
                    halt_ids,
                    flush=indents and _resumes(cross_target, nxt, skip_ids, ignored | aside_ids),
                    at_edge=_at_text_edge(cross_target, nxt),
                )
            )
            other = first if same else cross_target
            if (same or cross) and other is not None:
                units.append([b, other])
                used.add(id(b))
                used.add(id(other))
            else:
                units.append([b])
                used.add(id(b))
    return units


def _seam_apart(
    unit: list[TextBlock],
    by_page: dict[int, list[TextBlock]],
    geometries: Sequence[PageGeometry],
    covered: set[int],
    shown: set[int],
    median: float,
    words_of: Callable[[int], list[tuple]],
) -> bool:
    """跨页、跨栏、被式子切开的两块单元，中间隔着行间公式、例句：上一块紧下面、下一块紧上面是正文的块，
    或者是不在任何块里的字（抽字时丢掉的行间公式）。这样的单元写回时按分界切回两块（见 _assign_restored_parts），
    排满再接会把后半段写到公式、例句前面。紧挨着的是图表、图注、页眉页脚、脚注（covered 里是图、图注、表格的格子
    和页眉页脚）或者什么都没有时，两块之间只是换页、换栏"""
    host, tail = unit
    if host.page == tail.page and _same_page_line_pair(host, tail):
        return False
    stacked = host.page == tail.page and tail.rect.y0 >= host.rect.y1 - 1 and _x_overlap(host.rect, tail.rect)
    context = (by_page, geometries, covered, shown, median, words_of)
    if _content_next(host, tail, *context, below=True):
        return True
    return not stacked and _content_next(tail, host, *context, below=False)


def _content_next(
    block: TextBlock,
    partner: TextBlock,
    by_page: dict[int, list[TextBlock]],
    geometries: Sequence[PageGeometry],
    covered: set[int],
    shown: set[int],
    median: float,
    words_of: Callable[[int], list[tuple]],
    below: bool,
) -> bool:
    """块的正下方（below 为假时是正上方）离它最近的东西是正文：认出来的行间公式、例句（shown），
    不在 covered 里、不在图表框里、不是脚注的块，或者版心里不属于任何块的字。版面模型会把行间公式认成图，
    所以 shown 先看。小一号的块是图注接着的半句、旁注，小得多的是抽成块的公式，算正文。
    同一页上的 partner 和它后面的不看"""
    geo = geometries[block.page] if block.page < len(geometries) else None
    floats = [*geo.figures, *(table.rect for table in geo.tables)] if geo is not None else []
    frame = geo.frame if geo is not None else None
    edge = None if frame is None else (frame.y1 if below else frame.y0)
    page_blocks = by_page.get(block.page) or []
    found: list[tuple[float, bool]] = []

    def beyond(y0: float, y1: float) -> bool:
        if below:
            return y0 >= block.rect.y1 - 1 and not (partner.page == block.page and y0 >= partner.rect.y0 - 1)
        return y1 <= block.rect.y0 + 1 and not (partner.page == block.page and y1 <= partner.rect.y1 + 1)

    def distance(y0: float, y1: float) -> float:
        return y0 - block.rect.y1 if below else block.rect.y0 - y1

    for rect in floats:
        if _x_overlap(rect, block.rect) and beyond(rect.y0, rect.y1):
            found.append((distance(rect.y0, rect.y1), False))
    for other in page_blocks:
        if other is block or other is partner or not _x_overlap(other.rect, block.rect):
            continue
        if not beyond(other.rect.y0, other.rect.y1):
            continue
        cover = id(other) not in shown and (
            id(other) in covered
            or _is_page_footnote(other, median)
            or _is_float_text(other.text)
            or any(rect.contains(other.rect) for rect in floats)
            or block.size * 0.8 <= other.size < block.size - 0.6
        )
        found.append((distance(other.rect.y0, other.rect.y1), not cover))
    # 丢掉的行间公式：和块隔开一点、在版心以内、不落在任何块和图表框里的字
    for x0, y0, x1, y1, *_rest in words_of(block.page):
        word = pymupdf.Rect(x0, y0, x1, y1)
        if not _x_overlap(word, block.rect) or not beyond(y0, y1) or distance(y0, y1) < 0.3 * block.size:
            continue
        if edge is not None and (y0 > edge + 2 if below else y1 < edge - 2):
            continue
        middle = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any((other.rect + (-1, -1, 1, 1)).contains(middle) for other in page_blocks):
            continue
        if any(rect.contains(middle) for rect in floats):
            continue
        found.append((distance(y0, y1), True))
    return bool(found) and min(found)[1]


def _x_overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1


def _all_styled(block: TextBlock) -> bool:
    """每一截都是粗体或斜体。"""
    spans = [span for line in block.span_lines or [] for span in line if str(span.get("text") or "").strip()]
    marks = [emphasis_of(str(span.get("font") or ""), int(span.get("flags") or 0)) for span in spans]
    return bool(marks) and all(mark.bold or mark.italic for mark in marks)


def _math_on_line(block: TextBlock, index: int) -> bool:
    """块的第几行（-1 是末行）里有公式字体。"""
    lines = block.span_lines or []
    if not lines:
        return False
    return any(_span_is_math(span) for span in lines[index] if str(span.get("text") or "").strip())


def _shared_line_pairs(blocks: list[TextBlock]) -> dict[int, TextBlock]:
    """续行和另一段共基线时不并进外框，但仍成对送翻。并进去写回会盖住那一段。

    阅读顺序会把另一栏插在半句和续行中间。续行的纵坐标仍贴着半句，
    不能因为中间那栏已经往下走了就停。
    """
    pairs: dict[int, TextBlock] = {}
    claimed: set[int] = set()
    earlier = _earlier_on_page(blocks)
    for index, block in enumerate(blocks):
        if id(block) in claimed or not _is_line_tail(block, earlier[index]):
            continue
        for nxt in blocks[index + 1 :]:
            if nxt.page != block.page:
                break
            if nxt.rect.y0 > block.rect.y1 + block.size * 2:
                continue
            if not _line_tail_match(block, nxt):
                continue
            if not _line_tail_blocked(nxt, block, blocks) or id(nxt) in claimed:
                break
            pairs[id(block)] = nxt
            claimed.add(id(block))
            claimed.add(id(nxt))
            break
    return pairs
