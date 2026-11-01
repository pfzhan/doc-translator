"""Build the output files: side-by-side pages, the translated-only version cut from the
bilingual one, translated bookmarks, and font subsetting."""

import re
from collections.abc import Callable
from pathlib import Path

import pymupdf

from .pdf_blocks import _NUMBER_LABEL_RE, TextBlock, _pages_to_write
from .pdf_dedupe import merge_duplicates
from .pdf_flow import (
    strip_style_marks,
)
from .pdf_parts import plan_parts, render_in_parts, worker_count


def _render_by_parts(
    src_doc: pymupdf.Document,
    blocks: list[TextBlock],
    translations: list[str],
    continuations: dict[int, TextBlock],
    render,
    on_page: Callable[[int, int], None],
) -> pymupdf.Document:
    """分几段并行写回。跨页段落的宿主和续块分在同一段；有旋转页、页太少时整本一起写。"""
    weights = {pno: len(items) for pno, items in _pages_to_write(blocks, translations).items()}
    page_of = {id(b): b.page for b in blocks}
    links = [(page_of[host], tail.page) for host, tail in continuations.items() if host in page_of]
    parts = [list(weights)] if any(page.rotation for page in src_doc) else plan_parts(weights, links, worker_count())
    return render_in_parts(render, parts, on_page) or render(None, on_page)


def _render_side_by_side(src: pymupdf.Document, translated: pymupdf.Document) -> pymupdf.Document:
    out = pymupdf.open()
    for i in range(len(src)):
        w, h = src[i].rect.width, src[i].rect.height
        # 译文页可能为了整段跨页段落向下加长。并排页跟着加高，超出的几行才看得见。
        th = translated[i].rect.height
        page = out.new_page(width=w * 2, height=max(h, th))
        page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, i)
        page.show_pdf_page(pymupdf.Rect(w, 0, w * 2, th), translated, i)
    return out


def translated_from_bilingual(src: Path, bilingual: Path, preview: dict | None) -> bytes | None:
    """从双语版取出右半边的译文页，换进原书的副本，得到仅译文版，几秒就好。

    重新排版一本书要几分钟。原书副本里链接、命名目标、页码标签、书签和元数据都在，
    和重新排出来的一样，书签按 preview（任务的预览快照）换成译文（见 translate_bookmarks）。
    页面转过、页面框不从原点起、带裁剪框，译文页和原页大小不同，或原书有书签而没有预览快照时
    返回 None，交给重新排版。"""
    src_doc, both = pymupdf.open(src), pymupdf.open(bilingual)
    if len(src_doc) != len(both) or (preview is None and src_doc.get_toc()):
        return None
    forms = []
    for page, side in zip(src_doc, both):
        box = page.mediabox
        if (
            page.rotation
            or box.x0
            or box.y0
            or page.cropbox != box
            or abs(side.rect.width - 2 * box.width) > 0.01
            or abs(side.rect.height - box.height) > 0.01
        ):
            return None
        form = _translated_form(both, side, box)
        if form is None:
            return None
        forms.append(form)
    # 并排页只留译文那一半：页面框挪到右半边，内容只画译文页。资源和内容流可能被合并成几页共用，换成新的
    shown = both.get_new_xref()
    both.update_object(shown, "<<>>")
    both.update_stream(shown, b"q /T Do Q")
    for page, side, form in zip(src_doc, both, forms):
        both.xref_set_key(side.xref, "Resources", f"<</XObject<</T {form} 0 R>>>>")
        both.xref_set_key(side.xref, "Contents", f"{shown} 0 R")
        side.set_mediabox(pymupdf.Rect(page.rect.width, 0, 2 * page.rect.width, page.rect.height))
    blank = src_doc.get_new_xref()
    src_doc.update_object(blank, "<<>>")
    src_doc.update_stream(blank, b"")
    for i, page in enumerate(src_doc):
        src_doc.xref_set_key(page.xref, "Resources", "<<>>")
        src_doc.xref_set_key(page.xref, "Contents", f"{blank} 0 R")
        page.show_pdf_page(page.rect, both, i)
    translate_bookmarks(src_doc, preview)
    # 双语版已经合并、子集化过，这里只丢掉原书用不到的对象。garbage=3 再比一遍重复对象，一本书要一分钟
    return src_doc.tobytes(garbage=2, deflate=True)


def translate_bookmarks(doc: pymupdf.Document, preview: dict | None) -> None:
    """把仅译文版的书签标题换成正文里那个标题的译文。preview 是 Runner.preview(-1) 的结果。

    书签标题和某段原文去掉节号、不计大小写后相同，就用那段的译文。同一个标题有几种译法时
    （书前目录页和正文各译一次），用书签指向页或下一页上出现的那种：hyperref 的锚点常落在标题前一页的页底。
    对不上的、译文里有公式的保留原文。原书签不带节号时，译文开头的节号也去掉。"""
    toc = doc.get_toc()
    if not toc or not preview:
        return
    latest: dict[int, str | None] = {}
    for i, text, kept in preview.get("updates") or []:
        latest[i] = None if kept else text
    choices: dict[str, list[str]] = {}
    for i, segment in enumerate(preview.get("segments") or []):
        text = latest.get(i)
        if text is None:
            continue
        text = strip_style_marks(text).strip()
        if text and text != segment["s"].strip() and not re.search(r"[\x00-\x1f]|\{[a-z]\d+\}", text):
            choices.setdefault(_bookmark_key(segment["s"]), []).append(text)
    page_texts: dict[int, str] = {}

    def on_page(pno: int, text: str) -> bool:
        if not 1 <= pno <= len(doc):
            return False
        if pno not in page_texts:
            page_texts[pno] = re.sub(r"\s+", "", doc[pno - 1].get_text())
        return re.sub(r"\s+", "", text) in page_texts[pno]

    for xref, (_, title, pno) in zip(doc.get_outline_xrefs(), toc):
        found = list(dict.fromkeys(choices.get(_bookmark_key(title), ())))
        if len(found) > 1:
            found = [text for q in (pno, pno + 1) for text in found if on_page(q, text)]
        if not found:
            continue
        text = found[0]
        if not _NUMBER_LABEL_RE.match(title.strip()) and (number := _NUMBER_LABEL_RE.match(text)):
            text = text[number.end() :].strip()
        doc.xref_set_key(xref, "Title", pymupdf.get_pdf_str(text))


def _bookmark_key(text: str) -> str:
    text = strip_style_marks(text).strip().replace("’", "'")
    if number := _NUMBER_LABEL_RE.match(text):
        text = text[number.end() :]
    return " ".join(text.split()).casefold()


def _translated_form(doc: pymupdf.Document, page: pymupdf.Page, box: pymupdf.Rect) -> int | None:
    """并排页上画译文页的 Form XObject：show_pdf_page 平移到右半边、和原页一样大的那个。"""
    kind, value = doc.xref_get_key(page.xref, "Resources/XObject")
    if kind != "dict":
        return None
    found = []
    for xref in map(int, re.findall(r"(\d+) 0 R", value)):
        matrix = [float(v) for v in doc.xref_get_key(xref, "Matrix")[1].strip("[]").split()]
        bbox = [float(v) for v in doc.xref_get_key(xref, "BBox")[1].strip("[]").split()]
        if (
            len(matrix) == 6
            and len(bbox) == 4
            and matrix[:4] == [1, 0, 0, 1]
            and abs(matrix[4] - box.width) < 0.01
            and matrix[5] == 0
            and all(abs(a - b) < 0.01 for a, b in zip(bbox, box))
        ):
            found.append(xref)
    return found[0] if len(found) == 1 else None


def _subset_worker(data: bytes, queue):
    doc = pymupdf.open("pdf", data)
    doc.subset_fonts()
    queue.put(doc.tobytes(garbage=1, deflate=True))


def _subset_fonts_safe(data: bytes) -> bytes:
    """子集化放到子进程里跑（BabelDOC 同款）：PyMuPDF 子集化偶尔崩溃或卡死，
    60 秒没结果就放弃，用未子集化的版本——文件大一些，但能看。
    用 fork 上下文：spawn 会重新 import 主模块，在 stdin/REPL 里直接失败。
    不支持 fork 的平台（Windows）退回进程内直接子集化。"""
    import multiprocessing

    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        try:
            doc = pymupdf.open("pdf", data)
            doc.subset_fonts()
            return doc.tobytes(garbage=1, deflate=True)
        except Exception:  # noqa: BLE001
            return data
    queue = ctx.Queue()
    proc = ctx.Process(target=_subset_worker, args=(data, queue))
    proc.start()
    try:
        out = queue.get(timeout=60)
        proc.join(timeout=5)
        return out
    except Exception:  # noqa: BLE001 - 超时或子进程崩溃都回退
        proc.kill()
        return data


def _drop_mupdf_cache() -> None:
    """清空 MuPDF 的资源缓存。缓存里是前面几步加载的字体、图片，后面用不上，缓存满了以后
    后面每一步都明显变慢：ed3book 上合并重复对象慢两倍多，子集化慢三倍。写回的子进程从主进程
    fork 出来，也带着这份缓存。在写回、拼并排页、压缩之前各清一次。"""
    pymupdf.TOOLS.store_shrink(100)


def _compact(doc: pymupdf.Document) -> bytes:
    """合并内容相同的对象，再子集化字体，返回要写出的文件内容。

    insert_htmlbox 每次都嵌入整套字体，双语版里原书的图片、字体左右各一份。
    MuPDF 的 garbage=4 也能合并，但它两两比较所有对象，一本几百页的书要十几分钟。
    合并前先不压缩：重复的字体有几百 MB，压完大半又被合并掉。合并后再压，一篇 25 页的论文省 20 秒。
    """
    doc = pymupdf.open("pdf", doc.tobytes(garbage=2))
    merge_duplicates(doc)
    return _subset_fonts_safe(doc.tobytes(garbage=1, deflate=True))
