"""译文插回原文在内容流里的位置，拖选、复制时按阅读顺序走。

译文原来都追加在页面内容流的最后。浏览器、阅读器拖选文字按内容流的顺序：没翻的表格、页眉、
旁注排在前面，译文排在后面，从正文往下拖进表格，选区会往回跳。原页的内容流本来就是阅读顺序。

擦除之前，在每串要重写的字的位置放一个标记（/DTrN MP，什么也不画）。标记放在两个文本对象之间，
或者文本对象里一个换行定位操作（Td、TD、Tm、T*、'、"）前面：在那里先结束文本对象，放标记，
再开始一个文本对象，用 Tm 恢复当时的文本行矩阵，后面的字位置不变。擦除时 MuPDF 重写内容流，
标记留在原处。整页写完后在标记处把原页的流切开，把这些段落写出的流插进去。

插进去的流套一层：当时的 CTM 取逆，回到页面坐标；文字状态、颜色、透明度复位。
下列情况这一串不挪，照旧画在最后：
- 后面还有不透明的填充、图片、Form 压在写出的位置上，挪了会被盖住；
- 当时的裁剪不是矩形，或者没有盖住写出的位置。
自己解析出的字形数和 MuPDF 的对不上、字体编码认不出、页面旋转过时整页不挪。
"""
from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import pymupdf
import pymupdf.mupdf as mupdf

from .pdf_erase import _NUMBER_RE, _operators, _page_point

_MARK_RE = re.compile(rb"/DTr(\d+)\s+MP")
_POSITION = {b"Td", b"TD", b"Tm", b"T*"}
_SHOW = {b"Tj", b"TJ", b"'", b'"'}
_FILLS = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}
_PATH_END = _FILLS | {b"S", b"s", b"n"}
# 查字形归属时行框按纵坐标分桶的高度（点）
_ROW = 10
# 比这细的填充是表格线、分数线，压不住字
_THIN = 1.5
_GS_NAME = "DTrGS"
_GS = "<< /Type /ExtGState /CA 1 /ca 1 /BM /Normal /SMask /None /AIS false /OP false /op false /OPM 0 >>"
_RESET = "0 Tc 0 Tw 100 Tz 0 Ts 0 Tr 0 g 0 G 1 w 0 J 0 j [] 0 d"


class _Probe(mupdf.PdfSanitizeFilterOptions2):
    """只读一遍顶层内容流：每个文本对象里各个字形的起笔点（页面坐标）。"""

    def __init__(self, page: pymupdf.Page):
        super().__init__()
        self.use_virtual_text_filter()
        self.use_virtual_after_text_object()
        self.use_virtual_culler(False)
        self.use_virtual_image_filter()
        self.to_page = mupdf.FzMatrix(*page.transformation_matrix)
        self.objects: list[list[tuple[float, float]]] = [[]]
        self.error: BaseException | None = None

    def text_filter(self, ctx, ucsbuf, ucslen, trm, ctm, bbox, *_rest):
        try:
            self.objects[-1].append(_page_point(trm.e, trm.f, ctm, self.to_page))
        except BaseException as exc:  # noqa: BLE001 - 回调里的异常 MuPDF 接不住，记下来再抛
            self.error = self.error or exc
        return 0

    def image_filter(self, ctx, ctm, name, image, scissor):
        return mupdf.ll_fz_keep_image(image)

    def after_text_object(self, *_args):
        self.objects.append([])


class _Factory(mupdf.PdfFilterFactory2):
    def __init__(self, probe: _Probe):
        super().__init__()
        self.use_virtual_filter()
        self.probe = probe

    def filter(self, ctx, doc, chain, struct_parents, transform, options):
        return mupdf.ll_pdf_new_sanitize_filter(doc, chain, struct_parents, transform, options, self.probe.internal())


def _probe(page: pymupdf.Page) -> list[list[tuple[float, float]]]:
    probe = _Probe(page)
    factory = _Factory(probe)
    options = mupdf.PdfFilterOptions()
    options.recurse = 0
    options.instance_forms = 0
    options.ascii = 0
    options.no_update = 1
    options.newlines = 0
    options.add_factory(factory.internal())
    mupdf.pdf_filter_page_contents(pymupdf._as_pdf_document(page.parent), pymupdf._as_pdf_page(page), options)
    if probe.error is not None:
        raise probe.error
    return probe.objects[:-1]


def _literal_length(data: bytes, start: int) -> tuple[int, int]:
    """data[start] 是 ( 的字面字符串：解码后的字节数和字符串之后的位置。"""
    count, pos, depth = 0, start + 1, 1
    while pos < len(data):
        char = data[pos]
        if char == 0x5C:
            pos += 1
            if pos < len(data) and 0x30 <= data[pos] <= 0x37:
                end = pos
                while end < len(data) and end - pos < 3 and 0x30 <= data[end] <= 0x37:
                    end += 1
                pos, count = end, count + 1
                continue
            if pos < len(data) and data[pos] in (0x0A, 0x0D):
                pos += 1
                continue
            pos, count = pos + 1, count + 1
            continue
        if char == 0x28:
            depth += 1
        elif char == 0x29:
            depth -= 1
            if depth == 0:
                return count, pos + 1
        pos, count = pos + 1, count + 1
    return count, pos


def _string_bytes(operands: bytes) -> int:
    """操作数里各个字符串解码后的字节数之和。"""
    total, pos = 0, 0
    while pos < len(operands):
        char = operands[pos]
        if char == 0x28:
            count, pos = _literal_length(operands, pos)
            total += count
            continue
        if char == 0x3C and operands[pos:pos + 2] != b"<<":
            end = operands.find(b">", pos)
            if end < 0:
                break
            digits = len(re.sub(rb"\s", b"", operands[pos + 1:end]))
            total += (digits + 1) // 2
            pos = end + 1
            continue
        pos += 1
    return total


def _num(value: float) -> str:
    """PDF 里的数不能写成科学计数法。"""
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _box_of(corners: list[tuple[float, float]]) -> pymupdf.Rect | None:
    """四个点（可以回到起点）围成横平竖直的矩形时返回它。"""
    if corners and corners[-1] == corners[0]:
        corners = corners[:-1]
    if len(corners) != 4:
        return None
    xs = sorted({round(x, 3) for x, _ in corners})
    ys = sorted({round(y, 3) for _, y in corners})
    if len(xs) != 2 or len(ys) != 2:
        return None
    # 相邻两点只差一个坐标：是框，不是交叉的折线
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        if (round(x0, 3) != round(x1, 3)) == (round(y0, 3) != round(y1, 3)):
            return None
    return pymupdf.Rect(xs[0], ys[0], xs[1], ys[1])


def _numbers(operands: bytes) -> list[float]:
    return [float(value) for value in _NUMBER_RE.findall(operands)]


@dataclass
class _Op:
    """顶层流里的一个操作符：位置、在哪个文本对象里、执行前的状态。"""
    name: bytes
    start: int
    group: int | None = None
    glyphs: int = 0
    line: pymupdf.Matrix | None = None  # 执行前的文本行矩阵
    ctm: pymupdf.Matrix | None = None
    clips: list[pymupdf.Rect] | None = None  # 页面坐标；None 表示有不是矩形的裁剪


@dataclass
class _Mark:
    number: int
    at: int  # 在原流里的字节位置
    inside: bool  # 在文本对象里面，要先结束再开始
    line: pymupdf.Matrix | None  # 在里面时，当时的文本行矩阵
    blocks: list[int] = field(default_factory=list)


@dataclass
class _Cut:
    """擦除后的流里一个标记处的状态。MuPDF 重写时会把 cm 这类状态推迟到要画时才写，
    标记处的 CTM 要在重写后的流上算。"""
    end: int  # 标记之后的字节位置
    ctm: pymupdf.Matrix
    clips: list[pymupdf.Rect] | None
    paints: list[pymupdf.Rect]  # 标记之后画的不透明区域，页面坐标


class ReadingOrder:
    """一页的阅读顺序：擦除前 mark()，写每段时 record()，整页写完 assemble()。"""

    def __init__(self, page: pymupdf.Page, moved: Iterable, enabled: bool = True):
        self.page = page
        self.doc = page.parent
        self.marks: list[_Mark] = []
        self.streams: dict[int, list[int]] = {}
        self.rects: dict[int, list[pymupdf.Rect]] = {}
        self.base: list[int] = []
        self._spans: list[pymupdf.Rect] | None = None
        if not enabled or page.rotation:
            return
        try:
            self._mark(list(moved))
        except Exception:  # noqa: BLE001 - 解析不了的页照旧把译文画在最后
            self.marks = []

    # ---- 擦除前：找位置、放标记 ----

    def _fonts(self) -> dict[str, int | None]:
        """页面字体资源名 → 每个字的编码字节数；认不出的编码记 None。"""
        widths: dict[str, int | None] = {}
        for _xref, _ext, kind, _base, name, encoding, *_rest in self.page.get_fonts(full=True):
            if kind == "Type0":
                widths[name] = 2 if encoding in ("Identity-H", "Identity-V") else None
            else:
                widths[name] = 1
        return widths

    def _xobjects(self) -> dict[str, pymupdf.Rect | None]:
        """XObject 资源名 → 在用户空间里画的范围（图片是单位方块）。"""
        out: dict[str, pymupdf.Rect | None] = {}
        for img in self.page.get_images(full=True):
            out[img[7]] = pymupdf.Rect(0, 0, 1, 1)
        for xref, name, _invoker, bbox in self.page.get_xobjects():
            rect = pymupdf.Rect(bbox)
            kind, value = self.doc.xref_get_key(xref, "Matrix")
            if kind == "array":
                values = _numbers(value.encode())
                if len(values) == 6:
                    rect = rect * pymupdf.Matrix(*values)
            out[name] = rect
        return out

    def _walk(self, raw: bytes, strict: bool = True) -> list[_Op] | None:
        """走一遍顶层流，记下每个操作符执行前的状态和每个 show 画几个字。
        strict 时认不出的字体、操作数返回 None；否则不数字形，只跟状态。"""
        fonts = self._fonts()
        xobjects = self._xobjects()
        to_page = self.page.transformation_matrix
        whole = self.page.rect
        ops: list[_Op] = []
        # 图形状态栈：CTM、裁剪、TL、Tr
        ctm = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
        clips: list[pymupdf.Rect] | None = []
        leading, render = 0.0, 0
        stack: list[tuple] = []
        group: int | None = None
        groups = 0
        line = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
        width: int | None = 1
        # 路径的范围按坐标记：一个点的 Rect 是空的，PyMuPDF 取并集时会把它丢掉
        path: list[float] | None = None  # [x0, y0, x1, y1]
        path_rect: pymupdf.Rect | None = None  # 路径只有一个 re 时是它
        corners: list[tuple[float, float]] = []  # 只有 m、l 时的各个点，四点围成横平竖直的框也算矩形
        path_parts = 0
        clipping = False
        self.paints: list[tuple[int, pymupdf.Rect]] = []
        self.unsplittable: set[int] = set()
        for name, start, end in _operators(raw):
            operands = raw[start:end - len(name)]
            op = _Op(name, start, group, line=line if group is not None else None, ctm=ctm,
                     clips=None if clips is None else list(clips))
            ops.append(op)
            if name == b"q":
                stack.append((ctm, clips, leading, render))
            elif name == b"Q":
                if stack:
                    ctm, clips, leading, render = stack.pop()
            elif name == b"cm":
                values = _numbers(operands)
                if len(values) != 6:
                    return None
                ctm = pymupdf.Matrix(*values) * ctm
            elif name == b"BT":
                group, groups = groups, groups + 1
                op.group = group
                line = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
                op.line = line
            elif name == b"ET":
                group = None
            elif name == b"TL":
                values = _numbers(operands)
                leading = values[-1] if values else leading
            elif name == b"Tr":
                values = _numbers(operands)
                render = int(values[-1]) if values else render
            elif name == b"Tf":
                found = re.findall(rb"/([^\s/\[\]<>()]+)", operands)
                width = fonts.get(found[0].decode("latin1")) if found else None
            elif name in (b"Td", b"TD"):
                values = _numbers(operands)
                if len(values) != 2:
                    return None
                if name == b"TD":
                    leading = -values[1]
                line = pymupdf.Matrix(1, 0, 0, 1, *values) * line
            elif name == b"Tm":
                values = _numbers(operands)
                if len(values) != 6:
                    return None
                line = pymupdf.Matrix(*values)
            elif name == b"T*":
                line = pymupdf.Matrix(1, 0, 0, 1, 0, -leading) * line
            if name in _SHOW and group is not None:
                if width is None:
                    if strict:
                        return None
                    continue
                if name in (b"'", b'"'):
                    line = pymupdf.Matrix(1, 0, 0, 1, 0, -leading) * line
                op.glyphs = _string_bytes(operands) // width
                if render >= 4:
                    self.unsplittable.add(group)
            # 路径和裁剪
            if name in (b"m", b"l", b"c", b"v", b"y", b"re"):
                values = _numbers(operands)
                if name == b"re" and len(values) == 4:
                    x, y, w, h = values
                    xs, ys = [x, x + w], [y, y + h]
                    path_rect = pymupdf.Rect(min(xs), min(ys), max(xs), max(ys)) if path_parts == 0 else None
                    corners = []
                else:
                    if len(values) < 2:
                        return None
                    xs, ys = values[0::2], values[1::2]
                    path_rect = None
                    if name in (b"m", b"l") and (name == b"l" or path_parts == 0):
                        corners.append((xs[0], ys[0]))
                    else:
                        corners = [(float("nan"), float("nan"))]
                if path is None:
                    path = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    path = [min(path[0], *xs), min(path[1], *ys), max(path[2], *xs), max(path[3], *ys)]
                path_parts += 1
            elif name in (b"W", b"W*"):
                clipping = True
            elif name in _PATH_END:
                if name in _FILLS and path is not None:
                    painted = pymupdf.Rect(path) * ctm * to_page
                    if min(painted.width, painted.height) >= _THIN:
                        self.paints.append((len(ops) - 1, painted))
                if clipping and clips is not None:
                    axis = abs(ctm.b) < 1e-9 and abs(ctm.c) < 1e-9 or abs(ctm.a) < 1e-9 and abs(ctm.d) < 1e-9
                    if path_rect is None:
                        path_rect = _box_of(corners)
                    if path_rect is not None and axis:
                        clips = [*clips, pymupdf.Rect(path_rect) * ctm * to_page]
                    else:
                        clips = None
                path, path_rect, corners, path_parts, clipping = None, None, [], 0, False
            elif name == b"Do":
                found = re.findall(rb"/([^\s/\[\]<>()]+)", operands)
                area = xobjects.get(found[0].decode("latin1")) if found else None
                self.paints.append((len(ops) - 1, area * ctm * to_page if area is not None else whole))
            elif name == b"EI":
                self.paints.append((len(ops) - 1, pymupdf.Rect(0, 0, 1, 1) * ctm * to_page))
            elif name == b"sh":
                self.paints.append((len(ops) - 1, whole))
        return ops

    def _mark(self, moved: list) -> None:
        raw = self.page.read_contents()
        ops = self._walk(raw)
        if ops is None:
            return
        objects = _probe(self.page)
        # 每个文本对象自己数出的字形数要和 MuPDF 的一样，字形才能对到操作符上
        counted: dict[int, int] = {}
        for op in ops:
            if op.group is not None:
                counted[op.group] = counted.get(op.group, 0) + op.glyphs
        if len(objects) != len(counted) or any(counted.get(i, 0) != len(pts) for i, pts in enumerate(objects)):
            return
        # 行框按纵坐标分桶，一页几千个字形逐个查归属
        rows: dict[int, list[tuple[pymupdf.Rect, int]]] = {}
        for block in moved:
            for r in block.line_rects or [block.rect]:
                area = r + (-1, -1, 1, 1)
                for row in range(int(area.y0 // _ROW), int(area.y1 // _ROW) + 1):
                    rows.setdefault(row, []).append((area, id(block)))

        def owner(x: float, y: float) -> int | None:
            for area, key in rows.get(int(y // _ROW), ()):
                if area.x0 <= x <= area.x1 and area.y0 <= y <= area.y1:
                    return key
            return None

        # 按流的顺序排成一串字：(操作符序号, 归属)
        glyphs: list[tuple[int, int | None]] = []
        cursor = iter(objects)
        points: list[tuple[float, float]] = []
        taken = 0
        for index, op in enumerate(ops):
            if op.name == b"BT":
                points, taken = next(cursor), 0
            for x, y in points[taken:taken + op.glyphs]:
                glyphs.append((index, owner(x, y)))
            taken += op.glyphs

        def between(left: int, right: int) -> int | None:
            """glyphs[left] 和 glyphs[right] 之间能放标记的操作符序号。"""
            a, b = glyphs[left][0], glyphs[right][0]
            if ops[a].group != ops[b].group:
                # 后一个文本对象的 BT 前面
                return next(i for i in range(b, -1, -1) if ops[i].name == b"BT")
            if ops[a].group in self.unsplittable:
                return None
            for i in range(b, a, -1):
                if ops[i].name in _POSITION or ops[i].name in (b"'", b'"'):
                    return i
            return None

        # 两个字之间画过东西就断开：插在前一个字的位置，后一段会落到这些东西下面
        painted = sorted(index for index, _rect in self.paints)

        def paints_between(a: int, b: int) -> bool:
            k = bisect.bisect_right(painted, a)
            return k < len(painted) and painted[k] < b

        assigned: set[int] = set()
        i = 0
        while i < len(glyphs):
            if glyphs[i][1] is None:
                i += 1
                continue
            j = i
            while (j + 1 < len(glyphs) and glyphs[j + 1][1] is not None
                   and not paints_between(glyphs[j][0], glyphs[j + 1][0])):
                j += 1
            run = list(dict.fromkeys(key for _, key in glyphs[i:j + 1] if key not in assigned))
            at = None
            if run:
                if i == 0:
                    at = next(k for k in range(glyphs[i][0], -1, -1) if ops[k].name == b"BT")
                else:
                    at = between(i - 1, i)
                if at is None:
                    at = len(ops) if j == len(glyphs) - 1 else between(j, j + 1)
            if at is not None and run:
                self._add(ops, at, len(raw), run)
                assigned.update(run)
            i = j + 1
        if self.marks:
            self._write(raw)

    def _add(self, ops: list[_Op], at: int, end: int, blocks: list[int]) -> None:
        if at >= len(ops):
            self.marks.append(_Mark(len(self.marks), end, False, None, blocks))
        else:
            op = ops[at]
            self.marks.append(_Mark(len(self.marks), op.start, op.name != b"BT", op.line, blocks))

    def _write(self, raw: bytes) -> None:
        pieces, last = [], 0
        for mark in sorted(self.marks, key=lambda m: m.at):
            pieces.append(raw[last:mark.at])
            if mark.inside:
                line = " ".join(_num(v) for v in mark.line)
                pieces.append(f"\nET\n/DTr{mark.number} MP\nBT\n{line} Tm\n".encode())
            else:
                pieces.append(f"\n/DTr{mark.number} MP\n".encode())
            last = mark.at
        pieces.append(raw[last:])
        stream = self.doc.get_new_xref()
        self.doc.update_object(stream, "<<>>")
        self.doc.update_stream(stream, b"".join(pieces))
        self.doc.xref_set_key(self.page.xref, "Contents", f"{stream} 0 R")

    # ---- 写回时：记下每段写出的流 ----

    def after_erase(self) -> None:
        """擦完、写之前调用。先把流包平衡：insert_htmlbox 第一次写时会补 q/Q，补的流不算哪一段的。"""
        if not self.marks:
            return
        self.page.wrap_contents()
        self.base = self.page.get_contents()

    def snapshot(self) -> set[int]:
        return set(self.page.get_contents()) if self.marks else set()

    def record(self, block, before: set[int], rect: pymupdf.Rect) -> None:
        if not self.marks:
            return
        added = [xref for xref in self.page.get_contents() if xref not in before]
        self.streams.setdefault(id(block), []).extend(added)
        self.rects.setdefault(id(block), []).append(pymupdf.Rect(rect))

    # ---- 整页写完：在标记处插回 ----

    def _extent(self, rect: pymupdf.Rect) -> pymupdf.Rect:
        """写入框里实际写出的字的范围。框按能写到的最远处给，比字大，拿它判断会不会被盖住太保守。"""
        if self._spans is None:
            self._spans = [pymupdf.Rect(span["bbox"]) for block in self.page.get_text("dict")["blocks"]
                           if block.get("type") == 0 for line in block["lines"] for span in line["spans"]
                           if span["text"].strip()]
        found = pymupdf.Rect()
        for span in self._spans:
            middle = (span.tl + span.br) / 2
            if rect.contains(middle):
                found |= span
        return found if not found.is_empty else rect

    def _movable(self, key: int, cut: _Cut) -> bool:
        """这一段写出的字插到 cut 处不会被裁掉、不会被后面画的东西盖住。"""
        rects = [self._extent(rect) for rect in self.rects.get(key, [])]
        if not rects or cut.clips is None or abs(cut.ctm.a * cut.ctm.d - cut.ctm.b * cut.ctm.c) < 1e-9:
            return False
        for rect in rects:
            if not all(clip.contains(rect) for clip in cut.clips):
                return False
            for paint in cut.paints:
                overlap = paint & rect
                if not overlap.is_empty and overlap.width * overlap.height > 1:
                    return False
        return True

    def assemble(self) -> None:
        if not self.marks:
            return
        contents = self.page.get_contents()
        marked = next((xref for xref in self.base if _MARK_RE.search(self.doc.xref_stream(xref) or b"")), None)
        if marked is None:
            return
        data = self.doc.xref_stream(marked)
        ops = self._walk(data, strict=False)
        if ops is None:
            return
        by_number = {mark.number: mark for mark in self.marks}
        moving: dict[int, tuple[list[int], _Cut]] = {}
        for index, op in enumerate(ops):
            found = _MARK_RE.match(data, op.start) if op.name == b"MP" else None
            # 标记要在文本对象外面才能切开
            if found is None or op.group is not None:
                continue
            mark = by_number.get(int(found.group(1)))
            if mark is None:
                continue
            cut = _Cut(found.end(), op.ctm, op.clips, [rect for at, rect in self.paints if at > index])
            blocks = [key for key in mark.blocks if self._movable(key, cut)]
            if blocks:
                moving[found.end()] = (blocks, cut)
        if not moving:
            return
        self._ensure_state()
        moved_streams: set[int] = set()
        refs: list[int] = []
        for xref in contents:
            if xref != marked:
                continue
            last = 0
            for end in sorted(moving):
                refs.append(self._stream(data[last:end]))
                blocks, cut = moving[end]
                streams = [xref for key in blocks for xref in self.streams.get(key, [])]
                inverse = ~cut.ctm
                refs.append(self._stream(
                    f"q {inverse.a:.6f} {inverse.b:.6f} {inverse.c:.6f} {inverse.d:.6f} "
                    f"{inverse.e:.4f} {inverse.f:.4f} cm {_RESET} /{_GS_NAME} gs".encode()
                ))
                refs.extend(streams)
                refs.append(self._stream(b"Q"))
                moved_streams.update(streams)
                last = end
            refs.append(self._stream(data[last:]))
        order = []
        for xref in contents:
            if xref == marked:
                order.extend(refs)
            elif xref not in moved_streams:
                order.append(xref)
        self.doc.xref_set_key(self.page.xref, "Contents", "[" + " ".join(f"{x} 0 R" for x in order) + "]")

    def _stream(self, data: bytes) -> int:
        xref = self.doc.get_new_xref()
        self.doc.update_object(xref, "<<>>")
        self.doc.update_stream(xref, b"\n" + data + b"\n")
        return xref

    def _ensure_state(self) -> None:
        """页面资源里加一个复位用的 ExtGState。"""
        kind, value = self.doc.xref_get_key(self.page.xref, "Resources")
        found = re.match(r"(\d+) 0 R", value) if kind == "xref" else None
        res, prefix = (int(found.group(1)), "") if found else (self.page.xref, "Resources/")
        kind, value = self.doc.xref_get_key(res, f"{prefix}ExtGState")
        if kind == "xref":
            self.doc.xref_set_key(int(value.split()[0]), _GS_NAME, _GS)
        elif kind == "dict":
            self.doc.xref_set_key(res, f"{prefix}ExtGState/{_GS_NAME}", _GS)
        else:
            self.doc.xref_set_key(res, f"{prefix}ExtGState", f"<< /{_GS_NAME} {_GS} >>")
