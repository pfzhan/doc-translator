"""Erase the original text to be reflowed: drawing operators in the content stream are removed by glyph
ownership rather than cut out by rectangles.

Rectangle erasing also removes other text touching the box edge (descenders of the line above, labels in a
figure), and a box that is too small misses some of its own text. Here MuPDF rewrites the content stream and is
asked, glyph by glyph, who owns it: a glyph whose origin lies on the baseline of a span being reflowed, within
its width, belongs to that span. Spans without a reliable font box, such as bitmap glyphs, carry the origin of
each glyph (origins) and are matched by point.

Lines and images belong to no span. Lines lying entirely within a line being reflowed (the underline of \\_,
inline fraction bars), underlines right below the line's baseline and one-pixel-high thin images are removed
with the text; backgrounds, table rules and figures stay.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import pymupdf
import pymupdf.mupdf as mupdf

from .pdf_types import PdfDict

# Allowed distance between an origin and the baseline (as a share of the font size). Glyphs of one span share
# exactly one baseline, superscripts and subscripts are separate spans, and another line is at least one font
# size away.
_BASELINE_TOLERANCE = 0.15
# Allowance (points) for origins left of the span's left edge. Simulated bold draws the same glyph again slightly
# offset.
_X_TOLERANCE = 0.5
# Precision for comparing origins; content stream coordinates and those reported by PyMuPDF differ only by
# floating-point error
_ORIGIN_DIGITS = 1
# Thin images: old Ghostscript draws \_ and inline fraction bars as one-pixel-high images
_RULE_IMAGE_HEIGHT = 1.5
# Maximum depth of an underline below the baseline (as a share of the font size), and how far (points) it may
# extend beyond the text on either side
_UNDERLINE_DEPTH = 0.45
_UNDERLINE_REACH = 1.0
# Maximum gap between words of one run of body text (as a share of the font size). Spaces widened by
# justification stay within it
_WORD_GAP = 0.8

_PAINTED_PATHS = {mupdf.FZ_CULL_PATH_FILL, mupdf.FZ_CULL_PATH_STROKE, mupdf.FZ_CULL_PATH_FILL_STROKE}

# Lexical tokens of a content stream. Strings may contain nested parentheses and are scanned separately
_TOKEN_RE = re.compile(rb"\s+|%[^\r\n]*|<<|>>|<[^>]*>|[\[\]{}]|/[^\s()<>\[\]{}/%]*|[^\s()<>\[\]{}/%]+|\(")
_NUMBER_RE = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)")
_INLINE_END_RE = re.compile(rb"\sEI(?=\s|$)")
# The stream may contain cs or CS operators: preceded and followed by whitespace, a delimiter or the stream
# boundary. Every cs or CS found by _operators matches this
_COLOR_SPACE_RE = re.compile(rb"(?<![^\s()<>\[\]{}/%])(?:cs|CS)(?![^\s()<>\[\]{}/%])")
# Color-setting operators: fill color, stroke color
_FILL_COLOR = {b"g", b"rg", b"k", b"sc", b"scn"}
_STROKE_COLOR = {b"G", b"RG", b"K", b"SC", b"SCN"}
_PAINT_OPS = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"S", b"s", b"Tj", b"TJ", b"'", b'"', b"Do", b"EI"}


@dataclass
class _Line:
    baseline: float
    size: float
    x0: float
    x1: float


@dataclass
class Erasure:
    """What to erase on a page. lines are the line boxes being reflowed: blocks without spans match glyphs by
    them, and lines and thin images are removed by them too."""

    spans: list[PdfDict] = field(default_factory=list)
    lines: list[pymupdf.Rect] = field(default_factory=list)
    spanless: list[pymupdf.Rect] = field(default_factory=list)
    # Tells whether a span is formula text. A horizontal line below formula text is not erased as an underline
    math: Callable[[PdfDict], bool] = lambda span: False

    def add_block(self, span_lines: Sequence[Sequence[PdfDict]] | None, line_rects: Iterable[pymupdf.Rect]) -> None:
        rects = [pymupdf.Rect(rect) for rect in line_rects]
        self.lines.extend(rects)
        if not span_lines:
            self.spanless.extend(rects)
            return
        for index, spans in enumerate(span_lines):
            shown = [span for span in spans if str(span.get("text") or "").strip()]
            if all("bbox" in span and "origin" in span for span in shown):
                self.spans.extend(shown)
            elif index < len(rects):
                self.spanless.append(rects[index])
        # Merged punctuation fragments carry only line boxes, no spans
        if len(rects) > len(span_lines):
            self.spanless.extend(rects[len(span_lines) :])


class _Owners:
    """chars holds the origins of the characters in each span on the page, looked up by span box.

    A span's own origin is only that of its first character. When the first character is a space, it may sit on
    the baseline of a preceding subscript, a little off from the rest of the line, and using it as the baseline
    would leave a whole paragraph unerased. Spans whose per-character origins are found are matched by point.
    """

    def __init__(self, erasure: Erasure, chars: dict[tuple[float, ...], list[tuple[float, float]]] | None = None):
        self.points: set[tuple[float, float]] = set()
        self.lines: list[_Line] = []
        self.boxes: list[pymupdf.Rect] = []
        chars = chars or {}
        lined: list[PdfDict] = []
        for span in erasure.spans:
            origins = span.get("origins")
            if origins is not None:
                self.points.update(_key(x, y) for x, y in origins)
                continue
            bbox = pymupdf.Rect(span["bbox"])
            direction = span.get("dir", (1, 0))
            if abs(direction[1]) > 0.01:
                self.boxes.append(bbox)
                continue
            size = float(span.get("size") or bbox.height)
            found = chars.get(_box_key(bbox))
            baseline = span["origin"][1]
            if found:
                self.points.update(_key(x, y) for x, y in found)
                baseline = statistics.median(y for _x, y in found)
            # Baseline matching is still needed: a glyph drawn again slightly offset for simulated bold has an
            # origin different from the one PyMuPDF reports
            self.lines.append(_Line(baseline, size, bbox.x0, bbox.x1))
            lined.append(span)
        self.boxes.extend(erasure.spanless)
        # Every glyph is compared with these boxes. Converting them first to the single precision MuPDF stores
        # rectangles in gives the same results as Rect.contains
        self._box_bounds = [single_precision(box) for box in self.boxes]
        self.runs = _prose_runs([line for line, span in zip(self.lines, lined) if not erasure.math(span)])

    def underlines(self, rect: pymupdf.Rect) -> bool:
        """A thin horizontal line below the baseline of a run of body words, within the descender depth and not
        extending horizontally beyond the run, is their underline (links are often drawn this way) and is erased
        with the text. The bottom of the font box is often a little above the underline, so line boxes miss it.
        A line below formula text is a fraction bar or radical line; with the denominator still in place,
        erasing it would break the fraction, so it does not count."""
        if rect.height > _RULE_IMAGE_HEIGHT:
            return False
        return any(
            run.x0 - _UNDERLINE_REACH <= rect.x0
            and rect.x1 <= run.x1 + _UNDERLINE_REACH
            and run.baseline <= rect.y0
            and rect.y1 <= run.baseline + run.size * _UNDERLINE_DEPTH
            for run in self.runs
        )

    def owns(self, x: float, y: float) -> bool:
        """x and y are page coordinates computed by MuPDF, already single-precision values."""
        if _key(x, y) in self.points:
            return True
        for line in self.lines:
            if abs(y - line.baseline) <= line.size * _BASELINE_TOLERANCE and line.x0 - _X_TOLERANCE <= x < line.x1:
                return True
        # Same as MuPDF's point-in-rectangle test: the left and top edges are inside, the right and bottom are not
        return any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in self._box_bounds)


def single_precision(box: pymupdf.Rect) -> tuple[float, float, float, float]:
    """The rectangle in the single precision MuPDF stores. Comparing single-precision points with it gives the same
    result as Rect.contains without converting each time."""
    rect = mupdf.FzRect(box.x0, box.y0, box.x1, box.y1)
    return rect.x0, rect.y0, rect.x1, rect.y1


def _prose_runs(lines: list[_Line]) -> list[_Line]:
    """Join adjacent body spans on the same baseline into runs: an underline also runs below the spaces between
    words. A formula in between (leaving a wide gap) breaks the run."""
    runs: list[_Line] = []
    for line in sorted(lines, key=lambda item: (round(item.baseline, 1), item.x0)):
        last = runs[-1] if runs else None
        if (
            last is not None
            and round(last.baseline, 1) == round(line.baseline, 1)
            and line.x0 - last.x1 <= max(last.size, line.size) * _WORD_GAP
        ):
            runs[-1] = _Line(last.baseline, max(last.size, line.size), last.x0, max(last.x1, line.x1))
        else:
            runs.append(line)
    return runs


def _key(x: float, y: float) -> tuple[float, float]:
    return round(x, _ORIGIN_DIGITS), round(y, _ORIGIN_DIGITS)


def _box_key(box: pymupdf.Rect) -> tuple[float, ...]:
    return tuple(round(value, 1) for value in (box.x0, box.y0, box.x1, box.y1))


_BLANKS = frozenset(" \t\n\u00a0\u2002\u2003\u2009\u3000")


def _char_origins(page: pymupdf.Page) -> dict[tuple[float, ...], list[tuple[float, float]]]:
    """Per-character origins of each span on the page (whitespace excluded), looked up by span box. Uses the same
    parameters as extraction."""
    data = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
    out: dict[tuple[float, ...], list[tuple[float, float]]] = {}
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                # Skip only real spaces. Control characters such as \r are glyphs without Unicode (the large norm
                # bars of CMEX) and must be erased too
                points = [
                    tuple(char["origin"])
                    for char in span.get("chars") or []
                    if str(char.get("c") or " ") not in _BLANKS
                ]
                if points:
                    out.setdefault(_box_key(pymupdf.Rect(span["bbox"])), []).extend(points)
    return out


def _page_point(x: float, y: float, ctm: mupdf.fz_matrix, to_page: mupdf.FzMatrix) -> tuple[float, float]:
    """Convert a content stream point to page coordinates. Calls the same MuPDF function as Point * Matrix with
    the same result, but it runs for every glyph, so the argument handling of the PyMuPDF wrappers is skipped.
    ctm is the low-level matrix given to the callback and is passed directly instead of being read entry by
    entry into a new one."""
    point = mupdf.ll_fz_transform_point_xy(x, y, ctm)
    point = mupdf.ll_fz_transform_point_xy(point.x, point.y, to_page.internal())
    return point.x, point.y


def _rect(r: mupdf.fz_rect) -> pymupdf.Rect:
    return pymupdf.Rect(r.x0, r.y0, r.x1, r.y1)


class _Sanitizer(mupdf.PdfSanitizeFilterOptions2):
    def __init__(self, page: pymupdf.Page, erasure: Erasure) -> None:
        # The SWIG bindings are unannotated; mypy accepts calls into them only through the binding's own class
        # (untyped_calls_exclude), not through this subclass
        mupdf.PdfSanitizeFilterOptions2.__init__(self)
        base: mupdf.PdfSanitizeFilterOptions2 = self
        # Every callback is explicitly enabled or disabled instead of relying on constructor defaults
        base.use_virtual_text_filter()
        base.use_virtual_culler()
        base.use_virtual_image_filter()
        base.use_virtual_after_text_object(False)
        self.to_page = page.transformation_matrix
        self.to_page_fz = mupdf.FzMatrix(*self.to_page)
        self.owners = _Owners(erasure, _char_origins(page))
        self.lines = erasure.lines
        self.removed = 0
        # MuPDF treats an exception in the callback as "keep", so it is recorded and raised again
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
            if self.owners.owns(*_page_point(trm.e, trm.f, ctm, self.to_page_fz)):
                self.removed += 1
                return 1
        except BaseException as exc:  # noqa: BLE001
            self.error = self.error or exc
        return 0

    def image_filter(
        self, ctx: mupdf.fz_context, ctm: mupdf.fz_matrix, name: str, image: mupdf.fz_image, scissor: mupdf.fz_rect
    ) -> mupdf.fz_image:
        # Images are always kept; thin images are removed by box in the culler. MuPDF releases the return value,
        # so a reference is added first
        kept: mupdf.fz_image = mupdf.ll_fz_keep_image(image)
        return kept

    def culler(self, ctx: mupdf.fz_context, bbox: mupdf.fz_rect, kind: int) -> int:
        try:
            if kind not in _PAINTED_PATHS and kind != mupdf.FZ_CULL_IMAGE:
                return 0
            rect = _rect(bbox) * self.to_page
            if kind == mupdf.FZ_CULL_IMAGE and rect.height > _RULE_IMAGE_HEIGHT:
                return 0
            if any(_covers(line, rect) for line in self.lines) or self.owners.underlines(rect):
                return 1
        except BaseException as exc:  # noqa: BLE001
            self.error = self.error or exc
        return 0


def _covers(line: pymupdf.Rect, rect: pymupdf.Rect) -> bool:
    return (
        line.x0 <= rect.x0
        and rect.x1 <= line.x1
        and line.y0 <= rect.y0
        and rect.y1 <= line.y1
        and not math.isinf(rect.width)
    )


class _Factory(mupdf.PdfFilterFactory2):
    def __init__(self, sanitizer: _Sanitizer) -> None:
        # Base-class calls for the same reason as in _Sanitizer
        mupdf.PdfFilterFactory2.__init__(self)
        base: mupdf.PdfFilterFactory2 = self
        base.use_virtual_filter()
        # Must not be named options: the base class has a field of that name, and assigning it would change the
        # pointer in the C structure
        self.sanitizer: mupdf.PdfSanitizeFilterOptions2 = sanitizer

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
            doc, chain, struct_parents, transform, options, self.sanitizer.internal()
        )
        return processor


def erase(page: pymupdf.Page, erasure: Erasure) -> int:
    """Remove the glyphs and lines of this page that are being reflowed, by ownership, and return the number of
    glyphs removed.

    The page's own content stream is rewritten; Form XObjects it uses are copied per page before being changed,
    so Forms shared with other pages are not affected.
    """
    if not erasure.spans and not erasure.lines:
        return 0
    doc = page.parent
    glyphs = type3_glyphs(page)
    streams = [*page.get_contents(), *(xref for xref, *_rest in page.get_xobjects())]
    bare = any(_paints_in_bare_space(doc.xref_stream(xref) or b"") for xref in streams)
    first_new = doc.xref_length()
    sanitizer = _Sanitizer(page, erasure)
    factory: mupdf.PdfFilterFactory2 = _Factory(sanitizer)
    options = mupdf.PdfFilterOptions()
    # The constructor does not guarantee zeroed fields, and a garbage value in no_update would silently discard
    # the filter result
    options.recurse = 1
    options.instance_forms = 1
    options.ascii = 0
    options.no_update = 0
    options.newlines = 0
    options.add_factory(factory.internal())
    mupdf.pdf_filter_page_contents(pymupdf._as_pdf_document(doc), pymupdf._as_pdf_page(page), options)
    restore_glyphs(doc, glyphs)
    if sanitizer.error is not None:
        raise sanitizer.error
    # The rewritten page stream and Forms are new objects. When the original stream draws without setting a color,
    # restoring colors would change its initial color, so nothing on the page is restored
    if not bare:
        for xref in range(first_new, doc.xref_length()):
            data = doc.xref_stream(xref) if doc.xref_is_stream(xref) else None
            fixed = _restore_dropped_colors(data) if data else data
            if fixed != data:
                doc.update_stream(xref, fixed)
    return sanitizer.removed


def _operators(data: bytes) -> Iterable[tuple[bytes, int, int]]:
    """Operators in a content stream, with the start of their operands and their own end."""
    pos, end, start = 0, len(data), -1
    while pos < end:
        match = _TOKEN_RE.match(data, pos)
        if match is None:
            pos += 1
            continue
        token = match.group()
        if token[:1].isspace() or token[:1] == b"%":
            pos = match.end()
            continue
        if start < 0:
            start = pos
        if token == b"(":
            depth = 0
            while pos < end:
                char = data[pos]
                if char == 0x5C:
                    pos += 2
                    continue
                pos += 1
                depth += (char == 0x28) - (char == 0x29)
                if depth == 0:
                    break
            continue
        pos = match.end()
        if token[:1] in b"<>[]{}/" or _NUMBER_RE.fullmatch(token) or token in (b"true", b"false", b"null"):
            continue
        yield token, start, pos
        start = -1
        if token == b"ID":
            # Inline image data is binary; skip to EI
            inline = _INLINE_END_RE.search(data, pos)
            pos = end if inline is None else inline.start() + 1


def _paints_in_bare_space(data: bytes) -> bool:
    """Draws after a color space change without setting a color, which by the specification uses the space's
    initial color."""
    # Most streams never change the color space and need no pass over every operator
    if _COLOR_SPACE_RE.search(data) is None:
        return False
    fill = stroke = False
    for op, _start, _end in _operators(data):
        if op == b"cs" or op == b"CS":
            fill, stroke = fill or op == b"cs", stroke or op == b"CS"
        elif op in _FILL_COLOR:
            fill = False
        elif op in _STROKE_COLOR:
            stroke = False
        elif op in _PAINT_OPS and (fill or stroke):
            return True
    return False


def _restore_dropped_colors(data: bytes) -> bytes:
    """When MuPDF rewrites a content stream, it omits a color after a color space change if the value equals the
    one before, but changing the color space resets the color to the initial color, so a white background is
    drawn as a black block. The previous color value is restored after every cs or CS not followed by a color."""
    if _COLOR_SPACE_RE.search(data) is None:
        return data
    ops = list(_operators(data))
    current: dict[bool, bytes | None] = {True: None, False: None}
    saved: list[dict[bool, bytes | None]] = []
    pieces: list[bytes] = []
    copied = 0
    for index, (op, start, op_end) in enumerate(ops):
        if op == b"q":
            saved.append(dict(current))
        elif op == b"Q" and saved:
            current = saved.pop()
        elif op in _FILL_COLOR or op in _STROKE_COLOR:
            current[op in _FILL_COLOR] = data[start : op_end - len(op)].strip()
        elif op == b"cs" or op == b"CS":
            fill = op == b"cs"
            follow = ops[index + 1][0] if index + 1 < len(ops) else b""
            values = current[fill]
            if values and follow not in ((b"sc", b"scn") if fill else (b"SC", b"SCN")):
                pieces += [data[copied:op_end], b" ", values, b" scn" if fill else b" SCN"]
                copied = op_end
    if not pieces:
        return data
    pieces.append(data[copied:])
    return b"".join(pieces)


def type3_glyphs(page: pymupdf.Page) -> dict[int, bytes]:
    """Glyph streams of the Type3 fonts used on this page. When filtering recursively, MuPDF also enters glyphs and
    asks for ownership in the glyph's own coordinates, removing parts of glyphs shared by the whole document: the
    same character would vanish from other pages and from the original page used for pasting formulas. They are
    written back unchanged after filtering."""
    doc = page.parent
    glyphs: dict[int, bytes] = {}
    for xref, _ext, kind, *_rest in page.get_fonts(full=True):
        if kind != "Type3" or xref <= 0:
            continue
        procs = doc.xref_get_key(xref, "CharProcs")
        if procs[0] == "xref":
            procs = ("dict", doc.xref_object(int(procs[1].split()[0])))
        for ref in re.findall(r"(\d+) 0 R", procs[1]):
            glyphs[int(ref)] = doc.xref_stream(int(ref))
    return glyphs


def restore_glyphs(doc: pymupdf.Document, glyphs: dict[int, bytes]) -> None:
    for xref, data in glyphs.items():
        if doc.xref_stream(xref) != data:
            doc.update_stream(xref, data)
