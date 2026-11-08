"""Helpers shared by the format tests."""

import asyncio
import zipfile
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pymupdf

from app.formats.pdf_types import Span
from app.runner import Runner
from app.translators import MockTranslator

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class MemCache:
    def __init__(self) -> None:
        self.d: dict[tuple[str, str], str] = {}

    def get_many(self, prefix: str, texts: list[str]) -> dict[str, str]:
        return {t: self.d[(prefix, t)] for t in texts if (prefix, t) in self.d}

    def put_many(self, prefix: str, pairs: dict[str, str]) -> None:
        for s, t in pairs.items():
            self.d[(prefix, s)] = t


def runner() -> Runner:
    return Runner(MockTranslator("zh-CN"), cache=MemCache())


# The coroutines of the format functions return different things; each test knows what it gets
def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


# --- Second batch: regression tests for format fixes (EPUB/MOBI/Markdown/PDF samples are all built inline) ---

CONTAINER_XML = (
    '<?xml version="1.0"?>'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def _write_epub(path: Path, manifest: str, spine: str, docs: dict[str, str]) -> None:
    """Build a minimal EPUB: mimetype first and uncompressed, OEBPS/content.opf plus the given documents."""
    opf = (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>Test Book</dc:title><dc:language>en</dc:language>"
        f"</metadata><manifest>{manifest}</manifest><spine>{spine}</spine></package>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONTAINER_XML)
        z.writestr("OEBPS/content.opf", opf)
        for name, data in docs.items():
            z.writestr(f"OEBPS/{name}", data)


def _pdf_with_text(text: str, fontname: str = "helv") -> "pymupdf.Document":
    doc = pymupdf.open()
    doc.new_page(width=400, height=300).insert_textbox(
        pymupdf.Rect(30, 30, 370, 250), text, fontsize=12, fontname=fontname
    )
    return doc


def _glyph(
    text: str,
    font: str,
    size: float,
    flags: int,
    bbox: tuple[float, float, float, float],
    ink: tuple[float, float, float, float],
    math: bool = False,
) -> Span:
    span: Span = {
        "text": text,
        "font": font,
        "size": size,
        "flags": flags,
        "bbox": bbox,
        "ink": ink,
        "origin": (bbox[0], ink[3]),
    }
    if math:
        span["math"] = True
    return span
