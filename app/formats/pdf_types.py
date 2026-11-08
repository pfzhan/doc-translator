"""Types shared by the PDF modules.

pymupdf returns its text, drawing and image dicts untyped. The TypedDicts below describe the keys this package reads
from them and the keys it adds. The package also builds dicts of these shapes itself (Type3 lines, spans cut by
character, test fixtures) holding only the keys their readers need, and readers that cannot rely on a key use .get(),
so the pymupdf shapes are declared total=False.
"""

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

import pymupdf

if TYPE_CHECKING:
    from .pdf_blocks import _FormulaImage

# A box as pymupdf writes it into its dicts: x0, y0, x1, y1
Box = tuple[float, float, float, float]

Point = tuple[float, float]


class Char(TypedDict, total=False):
    """A character of a "rawdict" span."""

    c: str
    origin: Point
    bbox: Box
    synthetic: bool


class Span(TypedDict, total=False):
    """A span of get_text("dict") / "rawdict", plus the keys this package adds to it."""

    text: str
    bbox: Box
    origin: Point
    size: float
    font: str
    flags: int
    char_flags: int
    color: int
    alpha: int
    bidi: int
    ascender: float
    descender: float
    # Only in "rawdict"
    chars: list[Char]
    # Added by this package
    # Ink box measured from the glyph outlines (see _attach_ink); ink_drawn marks one measured by drawing the glyph
    ink: Box
    ink_drawn: bool
    # Box of the span this piece was cut from (see _split_covered and _outside_formula)
    source: Box
    # The span is a formula: decided from the fonts of the whole text (see _mark_math_roman), or the characters lie
    # under a radical or an overlay (see _split_covered)
    math: bool
    # Characters boxed by the formula detection model (see _mark_detected_formulas)
    detected_formula: bool
    # Underlined text (see _mark_underlines)
    underline: bool
    # Origins of the glyphs of a span rebuilt from a Type3 font (see pdf_type3_lines._span)
    origins: list[Point]
    # Per-character boxes recorded before the page is erased (see _attach_chars)
    char_boxes: list[tuple[str, pymupdf.Rect]]


class Line(TypedDict, total=False):
    """A line of a text block of get_text("dict") / "rawdict"."""

    spans: list[Span]
    bbox: Box
    wmode: int
    dir: Point


class Block(TypedDict, total=False):
    """A block of get_text("dict") / "rawdict": text blocks (type 0) have lines, image blocks (type 1) the image
    keys."""

    type: int
    number: int
    flags: int
    bbox: Box
    lines: list[Line]
    # Added by this package: a block gathered from the margin notes of the page (see _detach_margin_notes)
    side_note: bool
    # Image blocks
    width: int
    height: int
    ext: str
    colorspace: int
    xres: int
    yres: int
    bpc: int
    size: int
    image: bytes
    mask: bytes | None
    transform: tuple[float, float, float, float, float, float]


class PageDict(TypedDict, total=False):
    """The result of get_text("dict") / "rawdict" for one page."""

    width: float
    height: float
    blocks: list[Block]


# One path command of a drawing: ("l", p1, p2), ("re", rect, orientation), ("qu", quad), ("c", p1, p2, p3, p4).
# The arity and the types after the tag depend on the tag, and the readers index the tuple after testing the tag,
# which a union of tuple types cannot narrow, hence Any
DrawingItem = tuple[Any, ...]


class Drawing(TypedDict, total=False):
    """A path of get_drawings(extended=True), or a rule this package writes in that format (see _image_rules).
    Clip and group entries carry only some of the keys."""

    type: str
    rect: pymupdf.Rect
    items: list[DrawingItem]
    fill: tuple[float, ...] | None
    color: tuple[float, ...] | None
    width: float | None
    closePath: bool
    even_odd: bool | None
    fill_opacity: float | None
    stroke_opacity: float | None
    dashes: str | None
    lineCap: tuple[int, ...]
    lineJoin: float
    layer: str
    seqno: int
    level: int
    scissor: pymupdf.Rect


class ImageInfo(TypedDict, total=False):
    """An image of get_image_info."""

    number: int
    bbox: Box
    width: int
    height: int
    colorspace: int
    xres: int
    yres: int
    bpc: int
    size: int
    transform: tuple[float, float, float, float, float, float]


class SpanRun(TypedDict):
    """Spans of a block collected into one inline formula while placeholders are made (see _placeholderize)."""

    spans: list[Span]
    bbox: pymupdf.Rect
    max_size: float
    # Line in the block, and the first and last span index on that line
    pos: tuple[int, int, int]
    # Pieces of the same formula wrapped onto the following lines (see _join_line_wrapped_formulas)
    extras: NotRequired[list["SpanRun"]]


# A formula record: what is cropped from the source page and where it goes in the translation. A formula wrapped over
# lines lists its pieces in parts, itself first; each piece is a record of its own.
# _formula_record and _part_record fill page through text; the rest is set by later steps or only for some formulas.
# Declared total=False all the same: the cropping and placement steps also take records holding only the keys they
# read (the tests build such records), and read the keys that may be missing with .get().
# Declared with the functional syntax because "raise" is a keyword
Formula = TypedDict(
    "Formula",
    {
        "page": int,
        # Crop area on the source page
        "bbox": pymupdf.Rect,
        "name": str,
        # Placement geometry: width, height, depth below the baseline, and how far a lone script sits above its line
        "w": float,
        "h": float,
        "d": float,
        "raise": float,
        "spans": list[Span],
        # Curves joined to the crop: radicals, fraction bars
        "curves": list[pymupdf.Rect],
        # Fallback when no image is available
        "text": str,
        # Offset from the start of its line on the source page
        "dx": float,
        # Index of the block in its translation unit
        "owner": int,
        "parts": list["Formula"],
        # Gap before this piece of a wrapped formula
        "lead": float,
        # Scale applied when placing, and its cap for a formula set on a line of its own
        "rs": float,
        "cap": float,
        # The cropped bitmap: a _FormulaImage in the pipeline, PNG bytes in tests
        "png": "_FormulaImage | bytes",
        "has_img": bool,
        # Index of the marker image that stands for the formula in the typeset text
        "mid": int,
    },
    total=False,
)


# A text line of a block: (text, line box, spans with text, all spans of the line)
LineItem = tuple[str, pymupdf.Rect, list[Span], list[Span]]


# pymupdf hands these dicts back untyped. Reading them through the functions below is the one place they are taken to
# have the shapes declared above


def text_dict(
    page: pymupdf.Page,
    option: Literal["dict", "rawdict"] = "dict",
    *,
    flags: int | None = None,
    clip: pymupdf.Rect | None = None,
) -> PageDict:
    """page.get_text(option, flags=flags, clip=clip); None leaves pymupdf's default."""
    return cast(PageDict, page.get_text(option, flags=flags, clip=clip))


def page_drawings(page: pymupdf.Page) -> list[Drawing]:
    """page.get_drawings(extended=True)."""
    return cast(list[Drawing], page.get_drawings(extended=True))


def image_infos(page: pymupdf.Page) -> list[ImageInfo]:
    """page.get_image_info()."""
    return cast(list[ImageInfo], page.get_image_info())
