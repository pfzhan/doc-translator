from pathlib import Path

from .epub import translate_epub
from .markdown import translate_markdown
from .mobi import translate_mobi
from .pdf import translate_pdf

SUPPORTED = {".md", ".markdown", ".epub", ".mobi", ".azw3", ".azw", ".pdf"}
# 支持边翻边预览的格式
PREVIEW_FORMATS = {".md", ".markdown", ".epub", ".mobi", ".azw3", ".azw", ".pdf"}
# 译文文件名是 {原名}.{bilingual|translated}.{扩展名}。标记必须贴着扩展名，
# 不能用子串：再上传 book.bilingual.md 得到的是 book.bilingual.translated.md。
_VARIANT_MARKERS = frozenset({"bilingual", "translated"})


def output_variant(name: str) -> str | None:
    """返回紧挨扩展名的版本标记；文件名里没有这个标记时返回 None。"""
    token = Path(name).stem.rsplit(".", 1)[-1]
    if token in _VARIANT_MARKERS:
        return token
    return None


async def translate_file(src: Path, out_dir: Path, runner, bilingual: bool, target_lang: str) -> list[Path]:
    ext = src.suffix.lower()
    if ext in (".md", ".markdown"):
        return await translate_markdown(src, out_dir, runner, bilingual)
    if ext == ".epub":
        return await translate_epub(src, out_dir, runner, bilingual, target_lang)
    if ext in (".mobi", ".azw3", ".azw"):
        return await translate_mobi(src, out_dir, runner, bilingual, target_lang)
    if ext == ".pdf":
        return await translate_pdf(src, out_dir, runner, bilingual, target_lang)
    raise ValueError(f"不支持的格式: {ext}")
