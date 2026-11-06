"""Types shared by the PDF modules."""

from typing import Any

import pymupdf

# A dict produced by pymupdf (the blocks, lines, spans and chars of get_text("dict") / "rawdict", the paths of
# get_drawings, the items of get_texttrace) or built in the same shape by this package (formulas, runs). pymupdf
# does not type these, and their keys vary with the kind of item, so values are Any.
PdfDict = dict[str, Any]

# A text line of a block: (text, line box, spans with text, all spans of the line)
LineItem = tuple[str, pymupdf.Rect, list[PdfDict], list[PdfDict]]
