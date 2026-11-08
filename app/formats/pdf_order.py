"""Insert translations at the original text's position in the content stream, so that selecting and copying
follow the reading order.

Translations used to be appended at the end of the page content stream. Browsers and readers select text in
content stream order: untranslated tables, running headers and margin notes come first and the translations
last, so dragging from the body down into a table makes the selection jump back. The content stream of the
original page is already in reading order.

Before erasing, a marker (/DTrN MP, which draws nothing) is placed where each run of text to rewrite starts.
It goes between two text objects, or inside a text object before a line positioning operator (Td, TD, Tm, T*,
', "): the text object is ended there, the marker placed, and a new text object begun, with Tm restoring the
text line matrix so the following glyphs keep their positions. A paragraph whose text is in a Form (labels in
a figure) has no glyphs in the top-level stream, and its marker goes after the Do that draws the Form. MuPDF
rewrites the content stream while erasing and the markers stay in place. After the page is written, the
original stream is cut at the markers and the streams written for these paragraphs are inserted.

Each inserted stream is wrapped: the inverse of the CTM at that point returns to page coordinates, and text
state, color and transparency are reset. The written area is computed from the glyph outlines the paragraph
itself wrote. A run is not moved, and is drawn at the end as before, when:
- opaque fills, images, Forms or shadings drawn later overlap the written area and would cover it. Their
  extent is intersected with the bounding box of the clipping path in effect;
- the clip in effect is not a rectangle, or does not cover the written area.
Nothing on the page moves when the glyph count parsed here differs from MuPDF's, a font encoding is not
recognized, or the page is rotated.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import pymupdf
import pymupdf.mupdf as mupdf

from .pdf_blocks import TextBlock
from .pdf_erase import _NUMBER_RE, _operators, _page_point
from .pdf_types import text_dict

_MARK_RE = re.compile(rb"/DTr(\d+)\s+MP")
_POSITION = {b"Td", b"TD", b"Tm", b"T*"}
_SHOW = {b"Tj", b"TJ", b"'", b'"'}
_FILLS = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}
_PATH_END = _FILLS | {b"S", b"s", b"n"}
# Height in points of the vertical buckets of line boxes used to find which line a glyph belongs to
_ROW = 10
# Fills and images thinner than this are table rules or fraction bars and do not cover text
_THIN = 1.5
_GS_NAME = "DTrGS"
_GS = "<< /Type /ExtGState /CA 1 /ca 1 /BM /Normal /SMask /None /AIS false /OP false /op false /OPM 0 >>"
_RESET = "0 Tc 0 Tw 100 Tz 0 Ts 0 Tr 0 g 0 G 1 w 0 J 0 j [] 0 d"


class _Probe(mupdf.PdfSanitizeFilterOptions2):
    """Read the top-level content stream once: the origin of each glyph in each text object (page coordinates)."""

    def __init__(self, page: pymupdf.Page) -> None:
        # The SWIG bindings are unannotated; mypy accepts calls into them only through the binding's own class
        # (untyped_calls_exclude), not through this subclass
        mupdf.PdfSanitizeFilterOptions2.__init__(self)
        base: mupdf.PdfSanitizeFilterOptions2 = self
        base.use_virtual_text_filter()
        base.use_virtual_after_text_object()
        base.use_virtual_culler(False)
        base.use_virtual_image_filter()
        self.to_page = mupdf.FzMatrix(*page.transformation_matrix)
        self.objects: list[list[tuple[float, float]]] = [[]]
        self.error: BaseException | None = None

    def text_filter(
        self,
        ctx: mupdf.fz_context,
        ucsbuf: object,
        ucslen: int,
        trm: mupdf.fz_matrix,
        ctm: mupdf.fz_matrix,
        bbox: mupdf.fz_rect,
        *_rest: object,
    ) -> int:
        try:
            self.objects[-1].append(_page_point(trm.e, trm.f, ctm, self.to_page))
        except BaseException as exc:  # noqa: BLE001 - MuPDF cannot handle exceptions in the callback; keep and reraise
            self.error = self.error or exc
        return 0

    def image_filter(
        self, ctx: mupdf.fz_context, ctm: mupdf.fz_matrix, name: str, image: mupdf.fz_image, scissor: mupdf.fz_rect
    ) -> mupdf.fz_image:
        kept: mupdf.fz_image = mupdf.ll_fz_keep_image(image)
        return kept

    def after_text_object(self, *_args: object) -> None:
        self.objects.append([])


class _Factory(mupdf.PdfFilterFactory2):
    def __init__(self, probe: _Probe) -> None:
        # Base-class calls for the same reason as in _Probe
        mupdf.PdfFilterFactory2.__init__(self)
        base: mupdf.PdfFilterFactory2 = self
        base.use_virtual_filter()
        self.probe: mupdf.PdfSanitizeFilterOptions2 = probe

    def filter(
        self,
        ctx: mupdf.fz_context,
        doc: mupdf.pdf_document,
        chain: mupdf.pdf_processor,
        struct_parents: int,
        transform: mupdf.fz_matrix,
        options: mupdf.pdf_filter_options,
    ) -> mupdf.pdf_processor:
        processor: mupdf.pdf_processor = mupdf.ll_pdf_new_sanitize_filter(
            doc, chain, struct_parents, transform, options, self.probe.internal()
        )
        return processor


def _probe(page: pymupdf.Page) -> list[list[tuple[float, float]]]:
    probe = _Probe(page)
    factory: mupdf.PdfFilterFactory2 = _Factory(probe)
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
    """A literal string whose ( is at data[start]: the decoded byte count and the position after the string."""
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
    """Total decoded byte count of the strings among the operands."""
    total, pos = 0, 0
    while pos < len(operands):
        char = operands[pos]
        if char == 0x28:
            count, pos = _literal_length(operands, pos)
            total += count
            continue
        if char == 0x3C and operands[pos : pos + 2] != b"<<":
            end = operands.find(b">", pos)
            if end < 0:
                break
            digits = len(re.sub(rb"\s", b"", operands[pos + 1 : end]))
            total += (digits + 1) // 2
            pos = end + 1
            continue
        pos += 1
    return total


def _num(value: float) -> str:
    """Numbers in PDF cannot use scientific notation."""
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _box_of(corners: list[tuple[float, float]]) -> pymupdf.Rect | None:
    """The rectangle formed by four points (possibly returning to the start) when it is axis-aligned."""
    if corners and corners[-1] == corners[0]:
        corners = corners[:-1]
    if len(corners) != 4:
        return None
    xs = sorted({round(x, 3) for x, _ in corners})
    ys = sorted({round(y, 3) for _, y in corners})
    if len(xs) != 2 or len(ys) != 2:
        return None
    # Adjacent points differ in one coordinate only: a box, not a crossing polyline
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        if (round(x0, 3) != round(x1, 3)) == (round(y0, 3) != round(y1, 3)):
            return None
    return pymupdf.Rect(xs[0], ys[0], xs[1], ys[1])


def _numbers(operands: bytes) -> list[float]:
    return [float(value) for value in _NUMBER_RE.findall(operands)]


@dataclass
class _Op:
    """An operator in the top-level stream: its position, its text object and the state before it runs."""

    name: bytes
    start: int
    group: int | None = None
    glyphs: int = 0
    line: pymupdf.Matrix | None = None  # text line matrix before it runs
    ctm: pymupdf.Matrix | None = None
    clips: list[pymupdf.Rect] | None = None  # page coordinates; None when a clip is not a rectangle
    bounds: pymupdf.Rect | None = None  # intersection of the bounding boxes of all clip levels, page coordinates


@dataclass
class _Mark:
    number: int
    at: int  # byte position in the original stream
    inside: bool  # inside a text object, which must be ended and begun again
    line: pymupdf.Matrix | None  # when inside, the text line matrix at that point
    blocks: list[int] = field(default_factory=list)


@dataclass
class _Cut:
    """State at a marker in the stream after erasing. When rewriting, MuPDF defers state such as cm until
    something is drawn, so the CTM at the marker must be computed on the rewritten stream."""

    end: int  # byte position after the marker
    ctm: pymupdf.Matrix
    clips: list[pymupdf.Rect] | None
    paints: list[pymupdf.Rect]  # opaque areas drawn after the marker, page coordinates


class ReadingOrder:
    """Reading order of one page: mark() before erasing, record() when writing each paragraph, assemble() after the
    page is written."""

    def __init__(self, page: pymupdf.Page, moved: Iterable[TextBlock], enabled: bool = True) -> None:
        self.page = page
        self.doc = page.parent
        self.marks: list[_Mark] = []
        self.streams: dict[int, list[int]] = {}
        self.rects: dict[int, list[pymupdf.Rect]] = {}
        self.base: list[int] = []
        if not enabled or page.rotation:
            return
        try:
            self._mark(list(moved))
        except Exception:  # noqa: BLE001 - a page that cannot be parsed keeps the translations at the end
            self.marks = []

    # ---- Before erasing: find positions, place markers ----

    def _fonts(self) -> dict[str, int | None]:
        """Page font resource name → bytes per character code; None for an unrecognized encoding."""
        widths: dict[str, int | None] = {}
        for _xref, _ext, kind, _base, name, encoding, *_rest in self.page.get_fonts(full=True):
            if kind == "Type0":
                widths[name] = 2 if encoding in ("Identity-H", "Identity-V") else None
            else:
                widths[name] = 1
        return widths

    def _xobjects(self) -> dict[str, tuple[pymupdf.Rect, bool]]:
        """XObject resource name at the page level → the area drawn in user space (the unit square for images) and
        whether it is a Form. Resources inside Forms are listed too and may reuse names of the page level; they
        are ignored."""
        out: dict[str, tuple[pymupdf.Rect, bool]] = {}
        for img in self.page.get_images(full=True):
            if img[9] == 0:
                out[img[7]] = (pymupdf.Rect(0, 0, 1, 1), False)
        for xref, name, invoker, bbox in self.page.get_xobjects():
            if invoker != 0:
                continue
            rect = pymupdf.Rect(bbox)
            kind, value = self.doc.xref_get_key(xref, "Matrix")
            if kind == "array":
                values = _numbers(value.encode())
                if len(values) == 6:
                    rect = rect * pymupdf.Matrix(*values)
            out[name] = (rect, True)
        return out

    def _walk(self, raw: bytes, strict: bool = True) -> list[_Op] | None:
        """Walk the top-level stream, recording the state before each operator and how many glyphs each show
        operator draws. With strict, an unrecognized font or operand returns None; otherwise glyphs are not
        counted and only the state is tracked."""
        fonts = self._fonts()
        xobjects = self._xobjects()
        to_page = self.page.transformation_matrix
        whole = self.page.rect
        ops: list[_Op] = []
        # Graphics state stack: CTM, clip, clip bounding box, TL, Tr
        ctm = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
        clips: list[pymupdf.Rect] | None = []
        bounds = pymupdf.Rect(whole)
        leading, render = 0.0, 0
        stack: list[tuple[pymupdf.Matrix, list[pymupdf.Rect] | None, pymupdf.Rect, float, int]] = []
        group: int | None = None
        groups = 0
        line = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
        width: int | None = 1
        # Path extent kept as coordinates: a Rect of one point is empty and PyMuPDF drops it from unions
        path: list[float] | None = None  # [x0, y0, x1, y1]
        path_rect: pymupdf.Rect | None = None  # the rectangle when the path is a single re
        corners: list[tuple[float, float]] = []  # points of a path of only m and l; four forming an axis-aligned box
        # count as a rectangle
        path_parts = 0
        clipping = False
        self.paints: list[tuple[int, pymupdf.Rect]] = []
        self.forms: list[tuple[int, pymupdf.Rect]] = []  # Do operators drawing Forms, and the area drawn
        self.unsplittable: set[int] = set()

        def paint(area: pymupdf.Rect) -> None:
            # Nothing is drawn outside the clip; empty or very thin results do not count
            area = area & bounds
            if not area.is_empty and min(area.width, area.height) >= _THIN:
                self.paints.append((len(ops) - 1, area))

        for name, start, end in _operators(raw):
            operands = raw[start : end - len(name)]
            op = _Op(
                name,
                start,
                group,
                line=line if group is not None else None,
                ctm=ctm,
                clips=None if clips is None else list(clips),
                bounds=bounds,
            )
            ops.append(op)
            if name == b"q":
                stack.append((ctm, clips, bounds, leading, render))
            elif name == b"Q":
                if stack:
                    ctm, clips, bounds, leading, render = stack.pop()
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
            # Paths and clipping
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
                    paint(pymupdf.Rect(path) * ctm * to_page)
                if clipping and path is not None:
                    bounds = bounds & (pymupdf.Rect(path) * ctm * to_page)
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
                xobject = xobjects.get(found[0].decode("latin1")) if found else None
                area = xobject[0] * ctm * to_page if xobject is not None else whole
                paint(area)
                if xobject is not None and xobject[1]:
                    self.forms.append((len(ops) - 1, area & bounds))
            elif name == b"EI":
                paint(pymupdf.Rect(0, 0, 1, 1) * ctm * to_page)
            elif name == b"sh":
                paint(whole)
        return ops

    def _mark(self, moved: list[TextBlock]) -> None:
        raw = self.page.read_contents()
        ops = self._walk(raw)
        if ops is None:
            return
        objects = _probe(self.page)
        # The glyph count of each text object must match MuPDF's for glyphs to map onto operators
        counted: dict[int, int] = {}
        for op in ops:
            if op.group is not None:
                counted[op.group] = counted.get(op.group, 0) + op.glyphs
        if len(objects) != len(counted) or any(counted.get(i, 0) != len(pts) for i, pts in enumerate(objects)):
            return
        # Line boxes bucketed by y, for looking up the line of each of the thousands of glyphs on a page
        rows: dict[int, list[tuple[pymupdf.Rect, int]]] = {}
        for block in moved:
            for r in block.line_rects or [block.rect]:
                area = r + (-1, -1, 1, 1)
                for row in range(int(area.y0 // _ROW), int(area.y1 // _ROW) + 1):
                    rows.setdefault(row, []).append((area, id(block)))

        def owner(x: float, y: float, bounds: pymupdf.Rect) -> int | None:
            for area, key in rows.get(int(y // _ROW), ()):
                if area.x0 <= x <= area.x1 and area.y0 <= y <= area.y1:
                    # A glyph whose origin is more than a line height outside the clip is invisible on the
                    # original page and not extracted, so it does not belong to this paragraph
                    pad = area.height
                    if bounds.x0 - pad <= x <= bounds.x1 + pad and bounds.y0 - pad <= y <= bounds.y1 + pad:
                        return key
                    return None
            return None

        # Glyphs in stream order: (operator index, owner)
        glyphs: list[tuple[int, int | None]] = []
        cursor = iter(objects)
        points: list[tuple[float, float]] = []
        taken = 0
        for index, op in enumerate(ops):
            if op.name == b"BT":
                points, taken = next(cursor), 0
            # _walk gives every operator the clip bounds in effect
            assert op.bounds is not None
            for x, y in points[taken : taken + op.glyphs]:
                glyphs.append((index, owner(x, y, op.bounds)))
            taken += op.glyphs

        def between(left: int, right: int) -> int | None:
            """Index of an operator between glyphs[left] and glyphs[right] where a marker can go."""
            a, b = glyphs[left][0], glyphs[right][0]
            if ops[a].group != ops[b].group:
                # Before the BT of the later text object
                return next(i for i in range(b, -1, -1) if ops[i].name == b"BT")
            if ops[a].group in self.unsplittable:
                return None
            for i in range(b, a, -1):
                if ops[i].name in _POSITION or ops[i].name in (b"'", b'"'):
                    return i
            return None

        # Break when something is drawn between two glyphs: inserted at the earlier glyph, the later part would
        # end up under what was drawn
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
            while (
                j + 1 < len(glyphs)
                and glyphs[j + 1][1] is not None
                and not paints_between(glyphs[j][0], glyphs[j + 1][0])
            ):
                j += 1
            # Every glyph from i to j has an owner (see the loops above)
            run = list(dict.fromkeys(key for _, key in glyphs[i : j + 1] if key is not None and key not in assigned))
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
        # A paragraph with no glyphs in the top-level stream has its text in a Form: the marker goes after the Do
        # of the last Form covering the paragraph's center
        in_stream = {key for _, key in glyphs}
        after: dict[int, list[int]] = {}
        for block in moved:
            if id(block) in in_stream:
                continue
            middle = (block.rect.tl + block.rect.br) / 2
            hosts = [index for index, area in self.forms if area.contains(middle)]
            if hosts:
                after.setdefault(hosts[-1] + 1, []).append(id(block))
        for at, keys in after.items():
            self._add(ops, at, len(raw), keys)
        if self.marks:
            self._write(raw)

    def _add(self, ops: list[_Op], at: int, end: int, blocks: list[int]) -> None:
        if at >= len(ops):
            self.marks.append(_Mark(len(self.marks), end, False, None, blocks))
        else:
            op = ops[at]
            inside = op.group is not None and op.name != b"BT"
            self.marks.append(_Mark(len(self.marks), op.start, inside, op.line, blocks))

    def _write(self, raw: bytes) -> None:
        pieces, last = [], 0
        for mark in sorted(self.marks, key=lambda m: m.at):
            pieces.append(raw[last : mark.at])
            if mark.inside:
                # A mark inside a text object takes the line matrix of an operator in that object, always set
                assert mark.line is not None
                m = mark.line
                line = " ".join(_num(v) for v in (m.a, m.b, m.c, m.d, m.e, m.f))
                pieces.append(f"\nET\n/DTr{mark.number} MP\nBT\n{line} Tm\n".encode())
            else:
                pieces.append(f"\n/DTr{mark.number} MP\n".encode())
            last = mark.at
        pieces.append(raw[last:])
        stream = self.doc.get_new_xref()
        self.doc.update_object(stream, "<<>>")
        self.doc.update_stream(stream, b"".join(pieces))
        self.doc.xref_set_key(self.page.xref, "Contents", f"{stream} 0 R")

    # ---- While writing: record the stream written for each paragraph ----

    def after_erase(self) -> None:
        """Called after erasing and before writing. Balances the stream first: insert_htmlbox adds q/Q on its first
        write, and that added stream belongs to no paragraph."""
        if not self.marks:
            return
        self.page.wrap_contents()
        self.base = self.page.get_contents()

    def snapshot(self) -> set[int]:
        return set(self.page.get_contents()) if self.marks else set()

    def record(self, block: TextBlock, before: set[int], rect: pymupdf.Rect) -> None:
        if not self.marks:
            return
        added = [xref for xref in self.page.get_contents() if xref not in before]
        self.streams.setdefault(id(block), []).extend(added)
        self.rects.setdefault(id(block), []).append(pymupdf.Rect(rect))

    # ---- After the page is written: insert at the markers ----

    def _extent(self, key: int) -> list[pymupdf.Rect]:
        """Extent of each glyph written for this paragraph, from the glyph outlines. The write box extends as far
        as writing may reach and is larger than the text, and taking page text by position would pick up other
        paragraphs, so the page content stream is temporarily replaced by this paragraph's stream. Falls back to
        the write box when nothing was written."""
        streams = self.streams.get(key)
        if not streams:
            return list(self.rects.get(key, []))
        _kind, contents = self.doc.xref_get_key(self.page.xref, "Contents")
        self.doc.xref_set_key(self.page.xref, "Contents", "[" + " ".join(f"{x} 0 R" for x in streams) + "]")
        try:
            text = text_dict(self.page, flags=pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_ACCURATE_BBOXES)
        finally:
            self.doc.xref_set_key(self.page.xref, "Contents", contents)
        found = [
            pymupdf.Rect(span["bbox"])
            for block in text["blocks"]
            if block.get("type") == 0
            for line in block["lines"]
            for span in line["spans"]
            if span["text"].strip()
        ]
        return found or list(self.rects.get(key, []))

    def _movable(self, key: int, cut: _Cut) -> bool:
        """Text written for this paragraph, inserted at cut, would be neither clipped nor covered by what is drawn
        later."""
        rects = self._extent(key)
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
            # The stream can be cut at a marker only outside text objects
            if found is None or op.group is not None:
                continue
            mark = by_number.get(int(found.group(1)))
            if mark is None:
                continue
            assert op.ctm is not None  # _walk gives every operator the CTM in effect
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
                refs.append(
                    self._stream(
                        f"q {inverse.a:.6f} {inverse.b:.6f} {inverse.c:.6f} {inverse.d:.6f} "
                        f"{inverse.e:.4f} {inverse.f:.4f} cm {_RESET} /{_GS_NAME} gs".encode()
                    )
                )
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
        xref: int = self.doc.get_new_xref()
        self.doc.update_object(xref, "<<>>")
        self.doc.update_stream(xref, b"\n" + data + b"\n")
        return xref

    def _ensure_state(self) -> None:
        """Add an ExtGState used for resetting to the page resources."""
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
