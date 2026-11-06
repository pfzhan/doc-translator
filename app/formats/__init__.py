import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from . import docx, html_page, subtitles
from .docx import translate_docx
from .epub import translate_epub
from .html_page import translate_html
from .markdown import translate_markdown
from .mobi import translate_mobi
from .pdf import translate_pdf
from .pdf_vision import FormulaModel
from .subtitles import translate_subtitles

if TYPE_CHECKING:
    from ..runner import Runner

SUPPORTED = {
    ".md",
    ".markdown",
    ".epub",
    ".mobi",
    ".azw3",
    ".azw",
    ".pdf",
    *docx.SUFFIXES,
    *html_page.SUFFIXES,
    *subtitles.SUFFIXES,
}
# Formats that support preview while translation is running
PREVIEW_FORMATS = set(SUPPORTED)
# Output file names are {name}.{bilingual|translated}.{extension}. The marker must sit right before the
# extension, and a substring match is not enough: uploading book.bilingual.md again gives
# book.bilingual.translated.md.
_VARIANT_MARKERS = frozenset({"bilingual", "translated"})


def output_variant(name: str) -> str | None:
    """Return the version marker right before the extension, or None when the file name has no such marker."""
    token = Path(name).stem.rsplit(".", 1)[-1]
    if token in _VARIANT_MARKERS:
        return token
    return None


def document_digest(src: Path) -> str:
    """Digest of the file content. Glossaries and cache entries are keyed by it, so the same file uploaded
    several times counts as one document."""
    return hashlib.sha1(src.read_bytes()).hexdigest()


async def translate_file(
    src: Path,
    out_dir: Path,
    runner: "Runner",
    bilingual: bool,
    target_lang: str,
    layout_model: str = "",
    plan_path: Path | None = None,
    formula_model: FormulaModel | None = None,
) -> list[Path]:
    """layout_model, formula_model and plan_path apply only to PDF: the path of the layout model file (empty
    string means not used), the formula detection model settings (None means not used), and the file that
    stores the layout analysis result (read back when a job is resumed)."""
    runner.set_document(document_digest(src))
    ext = src.suffix.lower()
    if ext in (".md", ".markdown"):
        return await translate_markdown(src, out_dir, runner, bilingual)
    if ext == ".epub":
        return await translate_epub(src, out_dir, runner, bilingual, target_lang)
    if ext in (".mobi", ".azw3", ".azw"):
        return await translate_mobi(src, out_dir, runner, bilingual, target_lang)
    if ext in docx.SUFFIXES:
        return await translate_docx(src, out_dir, runner, bilingual)
    if ext in html_page.SUFFIXES:
        return await translate_html(src, out_dir, runner, bilingual, target_lang)
    if ext in subtitles.SUFFIXES:
        return await translate_subtitles(src, out_dir, runner, bilingual)
    if ext == ".pdf":
        return await translate_pdf(src, out_dir, runner, bilingual, target_lang, layout_model, plan_path, formula_model)
    raise ValueError(f"不支持的格式: {ext}")
