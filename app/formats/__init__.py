from pathlib import Path

from .epub import translate_epub
from .markdown import translate_markdown
from .mobi import translate_mobi
from .pdf import translate_pdf

SUPPORTED = {".md", ".markdown", ".epub", ".mobi", ".azw3", ".azw", ".pdf"}


async def translate_file(src: Path, out_dir: Path, runner, bilingual: bool, target_lang: str) -> list[Path]:
    ext = src.suffix.lower()
    if ext in (".md", ".markdown"):
        return await translate_markdown(src, out_dir, runner, bilingual)
    if ext == ".epub":
        return await translate_epub(src, out_dir, runner, bilingual, target_lang)
    if ext in (".mobi", ".azw3", ".azw"):
        return await translate_mobi(src, out_dir, runner, bilingual, target_lang)
    if ext == ".pdf":
        return await translate_pdf(src, out_dir, runner, bilingual)
    raise ValueError(f"不支持的格式: {ext}")
