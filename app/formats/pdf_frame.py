"""正文版心：整份文档里正文行的左右缘、上下缘，以及版心外的页边旁注。

单看一页时，栏的左缘取的是最靠左的块，常落在页眉、外凸的节号或页边旁注上，
正文其实从更靠右的地方起笔。版心由整份文档的正文长行投票：多数长行从同一处起笔、
止于同一处。书的奇偶页内外侧边距不同，分开统计。本页长行够多、又和同侧页对不上时
（参考文献改排两栏），用本页自己的左右缘。

页边旁注是版心外、比正文小一号的短词，和某一行正文共基线。PyMuPDF 常把它和
那一行正文分进同一块，不拆开就并进句子里。旁注沿一条边对齐（左侧的靠右对齐），
同一条边上的旁注要有好几条，零散的小字不算。
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf

_PROSE_CHARS = 40     # 够这么长的行才投票
_MIN_LINES = 12       # 一组里至少这么多长行
_EDGE_TOL = 2.0       # 同一条缘的容差
_EDGE_SHARE = 0.5     # 票数不低于最多那条缘的这个比例才算缘：两栏时两栏的左缘都够
_PAGE_SHIFT = 12.0    # 本页的缘和同侧页差这么多，改用本页的
_TOP_TOL = 3.0
_NOTE_GAP = 2.0       # 旁注离版心至少这么远
_NOTE_SMALL = 0.9     # 旁注字号不超过正文的这个比例
_LANE_ALIGN = 1.5
_LANE_MIN = 3         # 同一条边上至少这么多条旁注


@dataclass(frozen=True)
class FrameLine:
    rect: pymupdf.Rect
    text: str
    size: float


@dataclass(frozen=True)
class TextFrame:
    x0: float
    x1: float
    # 正文第一行的上缘、最后一行的下缘。同侧页里对不出一致的位置时为 None
    y0: float | None
    y1: float | None
    size: float


def text_frames(pages: Sequence[Sequence[FrameLine]]) -> list[TextFrame | None]:
    """每页的版心。长行太少、投不出来的页为 None。

    好几页同一高度、同一句话的长行是页眉页脚（Published as a conference paper at ICLR 2015），
    字号和正文一样，也不投票，不然版心的上缘落在页眉上。"""
    running = _running_lines(pages)
    prose = [
        [line for line in page if len(line.text.strip()) >= _PROSE_CHARS and _line_key(line) not in running]
        for page in pages
    ]
    whole = _frame_of(prose)
    shared: dict[int, TextFrame | None] = {}
    for parity in (0, 1):
        shared[parity] = _frame_of([lines for index, lines in enumerate(prose) if index % 2 == parity]) or whole
    frames: list[TextFrame | None] = []
    for index, lines in enumerate(prose):
        frame = shared[index % 2]
        own = _frame_of([lines], with_rows=False) if len(lines) >= _MIN_LINES else None
        if own is not None and frame is not None:
            # 同侧页的缘在本页仍有一批长行起笔、收尾，本页外扩的是通栏图注、宽表这类少数行
            x0 = frame.x0 if _holds([line.rect.x0 for line in lines], frame.x0) else own.x0
            x1 = frame.x1 if _holds([line.rect.x1 for line in lines], frame.x1) else own.x1
            own = TextFrame(x0, x1, own.y0, own.y1, own.size)
        if own is not None and (
            frame is None
            or abs(own.x0 - frame.x0) > _PAGE_SHIFT or abs(own.x1 - frame.x1) > _PAGE_SHIFT
        ):
            frame = TextFrame(
                own.x0, own.x1,
                frame.y0 if frame is not None else None,
                frame.y1 if frame is not None else None,
                own.size,
            )
        frames.append(frame)
    return frames


def _holds(values: Sequence[float], edge: float) -> bool:
    """这条缘在这些行里票数够：不低于最多那条缘的 _EDGE_SHARE。"""
    counter = Counter(round(value) for value in values)
    if not counter:
        return False
    best = max(_near(counter, value) for value in counter)
    return _near(counter, round(edge)) >= best * _EDGE_SHARE


def _line_key(line: FrameLine) -> tuple[int, str]:
    return round(line.rect.y0), re.sub(r"\d+", "#", " ".join(line.text.split()))


def _running_lines(pages: Sequence[Sequence[FrameLine]]) -> set[tuple[int, str]]:
    """在两页以上同一高度出现的同一行字（数字不算，页眉常带页码）。"""
    seen: Counter[tuple[int, str]] = Counter()
    for page in pages:
        seen.update({_line_key(line) for line in page if line.text.strip()})
    return {key for key, count in seen.items() if count >= 2}


def margin_notes(pages: Sequence[Sequence[FrameLine]], frames: Sequence[TextFrame | None]) -> list[set[int]]:
    """每页里是页边旁注的行（下标）。"""
    candidates: list[tuple[int, int, str]] = []
    for pno, (lines, frame) in enumerate(zip(pages, frames)):
        if frame is None:
            continue
        for index, line in enumerate(lines):
            # 不要求有字母：旁注也可以是 *? 这样的记号。页码和外凸的节号不比正文小，不在其内
            if not line.text.strip() or line.size > frame.size * _NOTE_SMALL:
                continue
            if line.rect.x1 <= frame.x0 - _NOTE_GAP:
                candidates.append((pno, index, "left"))
            elif line.rect.x0 >= frame.x1 + _NOTE_GAP:
                candidates.append((pno, index, "right"))
    edges: dict[tuple[str, int], Counter[int]] = {}
    for pno, index, side in candidates:
        rect = pages[pno][index].rect
        counter = edges.setdefault((side, pno % 2), Counter())
        counter[round(rect.x0)] += 1
        counter[round(rect.x1)] += 1
    found: list[set[int]] = [set() for _ in pages]
    for pno, index, side in candidates:
        rect = pages[pno][index].rect
        counter = edges[(side, pno % 2)]
        if any(_support(counter, edge) >= _LANE_MIN for edge in (rect.x0, rect.x1)):
            found[pno].add(index)
    return found


def note_groups(lines: Sequence[FrameLine]) -> list[list[int]]:
    """一页的旁注行合成几条旁注：上下紧挨的两行是同一条（regular / expressions），
    隔开一行正文行距的是两条（Herdan's Law、Heaps' Law）。返回每条的行下标。"""
    order = sorted(range(len(lines)), key=lambda index: (lines[index].rect.y0, lines[index].rect.x0))
    groups: list[list[int]] = []
    for index in order:
        line = lines[index]
        if groups:
            last = lines[groups[-1][-1]]
            pitch = line.rect.y0 - last.rect.y0
            same_side = (line.rect.x1 < last.rect.x1 + line.size * 8) and (line.rect.x0 > last.rect.x0 - line.size * 8)
            if same_side and 0 <= pitch <= max(line.size, last.size) * 1.3:
                groups[-1].append(index)
                continue
        groups.append([index])
    return groups


def _support(counter: Counter[int], edge: float) -> int:
    return sum(count for value, count in counter.items() if abs(value - edge) <= _LANE_ALIGN)


def _frame_of(pages: Sequence[Sequence[FrameLine]], with_rows: bool = True) -> TextFrame | None:
    lines = [line for page in pages for line in page]
    if len(lines) < _MIN_LINES:
        return None
    x0 = _edge([line.rect.x0 for line in lines], low=True)
    x1 = _edge([line.rect.x1 for line in lines], low=False)
    sizes = Counter(round(line.size * 2) / 2 for line in lines)
    size = sizes.most_common(1)[0][0]
    y0 = y1 = None
    if with_rows:
        # 只看版心里正文字号的行：长页眉也够投票的长度，但字号小一号或在版心外
        body = [
            [line for line in page
             if line.rect.x0 >= x0 - _EDGE_TOL and line.rect.x1 <= x1 + _EDGE_TOL
             and abs(line.size - size) <= size * 0.1]
            for page in pages
        ]
        y0 = _mode([min(line.rect.y0 for line in page) for page in body if page], _TOP_TOL)
        y1 = _mode([max(line.rect.y1 for line in page) for page in body if page], _TOP_TOL)
    return TextFrame(x0, x1, y0, y1, size)


def _edge(values: Sequence[float], low: bool) -> float:
    """票数够的缘里最靠外的一条。"""
    counter = Counter(round(value) for value in values)
    support = {value: _near(counter, value) for value in counter}
    best = max(support.values())
    eligible = [value for value, votes in support.items() if votes >= best * _EDGE_SHARE]
    pick = min(eligible) if low else max(eligible)
    # 取这条缘附近出现最多的那个值，不取簇的边上
    return float(max(
        (value for value in counter if abs(value - pick) <= _EDGE_TOL),
        key=lambda value: (counter[value], -abs(value - pick)),
    ))


def _near(counter: Counter[int], value: int) -> int:
    return sum(count for other, count in counter.items() if abs(other - value) <= _EDGE_TOL)


def _mode(values: Sequence[float], tol: float) -> float | None:
    """至少两页对得上的位置里最多的那个。"""
    if not values:
        return None
    counter = Counter(round(value) for value in values)
    support = {value: sum(count for other, count in counter.items() if abs(other - value) <= tol)
               for value in counter}
    value, votes = max(support.items(), key=lambda item: (item[1], -item[0]))
    if votes < 2:
        return None
    return float(value)
