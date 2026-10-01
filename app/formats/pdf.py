"""PDF 翻译，保留原版式。

1. 用 PyMuPDF 提取每页的文本块（坐标、字号、颜色、粗体）。
2. 跳过公式、纯数字、旋转文字等不适合翻译的块。
3. 译文版：用 redaction 擦掉原文字（图片和矢量图形保留），在原位置用 HTML 盒子写入译文，字号放不下时自动缩小。
4. 双语版：原页面和译文页面左右并排放在同一页上，方便对照阅读。
"""
import asyncio
import html
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

MATH_FONT_RE = re.compile(r"CMMI|CMSY|CMEX|MSBM|Math|Symbol|STIX|Cambria Math", re.I)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
NO_TEXT_RE = re.compile(r"^[\W\d_]*$")


@dataclass
class TextBlock:
    page: int
    rect: pymupdf.Rect
    line_rects: list[pymupdf.Rect]
    text: str
    size: float
    color: str
    bold: bool


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and re.match(r"[a-z]", line):
            out = out[:-1] + line  # 行尾连字符断词
        elif CJK_RE.search(out[-1]) or CJK_RE.search(line[0]):
            out += line
        else:
            out += " " + line
    return out


def extract_blocks(doc: pymupdf.Document) -> list[TextBlock]:
    blocks = []
    for pno, page in enumerate(doc):
        data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
        for b in data["blocks"]:
            if b.get("type") != 0:
                continue
            lines, line_rects, spans = [], [], []
            for ln in b["lines"]:
                if abs(ln["dir"][1]) > 0.01:  # 旋转 / 竖排文字不处理
                    continue
                text = "".join(s["text"] for s in ln["spans"])
                if not text.strip():
                    continue
                lines.append(text)
                line_rects.append(pymupdf.Rect(ln["bbox"]))
                spans.extend(s for s in ln["spans"] if s["text"].strip())
            if not spans:
                continue
            text = _join_lines(lines)
            total = sum(len(s["text"]) for s in spans)
            math_chars = sum(len(s["text"]) for s in spans if MATH_FONT_RE.search(s["font"]))
            letters = sum(c.isalpha() for c in text)
            if NO_TEXT_RE.match(text) or math_chars > total * 0.3 or letters < 2 or letters < len(text) * 0.4:
                continue
            # 字号、颜色取占比最多的 span
            main = max(spans, key=lambda s: len(s["text"]))
            rect = pymupdf.Rect()
            for r in line_rects:
                rect |= r
            blocks.append(TextBlock(
                page=pno,
                rect=rect,
                line_rects=line_rects,
                text=text,
                size=round(main["size"], 1),
                color=f"#{main['color']:06x}",
                bold=bool(main["flags"] & 16) or "Bold" in main["font"],
            ))
    return blocks


def _render_translated(src_path: Path, blocks: list[TextBlock], translations: list[str]) -> pymupdf.Document:
    doc = pymupdf.open(src_path)
    by_page: dict[int, list[tuple[TextBlock, str]]] = {}
    for b, t in zip(blocks, translations):
        if t and t.strip() and t.strip() != b.text.strip():
            by_page.setdefault(b.page, []).append((b, t))

    for pno, items in by_page.items():
        page = doc[pno]
        for b, _ in items:
            for r in b.line_rects:
                page.add_redact_annot(r, fill=False)
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_NONE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        )
        for b, t in items:
            weight = "bold" if b.bold else "normal"
            body = html.escape(t).replace("\n", "<br>")
            css = (
                f"* {{font-family: sans-serif; font-size: {b.size}px; color: {b.color}; "
                f"font-weight: {weight}; line-height: 1.2; margin: 0; padding: 0;}}"
            )
            # 留一点余量，避免译文比原文长时被截断；放不下由 scale_low=0 自动缩小
            rect = pymupdf.Rect(b.rect.x0, b.rect.y0, b.rect.x1 + 2, b.rect.y1 + b.size * 0.3)
            page.insert_htmlbox(rect, f"<div>{body}</div>", css=css, scale_low=0)
    return doc


def _render_side_by_side(src: pymupdf.Document, translated: pymupdf.Document) -> pymupdf.Document:
    out = pymupdf.open()
    for i in range(len(src)):
        w, h = src[i].rect.width, src[i].rect.height
        page = out.new_page(width=w * 2, height=h)
        page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, i)
        page.show_pdf_page(pymupdf.Rect(w, 0, w * 2, h), translated, i)
    return out


async def translate_pdf(src: Path, out_dir: Path, runner, bilingual: bool) -> list[Path]:
    src_doc = pymupdf.open(src)
    if src_doc.needs_pass:
        raise ValueError("PDF 有密码保护，暂不支持")
    blocks = await asyncio.to_thread(extract_blocks, src_doc)
    if not blocks:
        raise ValueError("没有提取到文字，可能是扫描版 PDF（需要 OCR，暂不支持）")
    # PDF 元数据里的标题经常是 "Microsoft Word - xxx.docx" 之类，看起来像文件名就不用
    meta_title = ((src_doc.metadata or {}).get("title") or "").strip()
    if re.search(r"\.(docx?|pdf|tex|indd)$|^untitled$", meta_title, re.I):
        meta_title = ""
    runner.set_title(meta_title or src.stem)
    translations = await runner.translate_all([b.text for b in blocks])

    def build():
        translated = _render_translated(src, blocks, translations)
        # insert_htmlbox 每次都会嵌入完整的 CJK 字体（十几 MB），必须做子集化
        translated.subset_fonts()
        if bilingual:
            dst = out_dir / f"{src.stem}.bilingual.pdf"
            # 先落盘再读回，并排页面引用的是已经子集化的字体
            tmp = pymupdf.open("pdf", translated.tobytes(garbage=4, deflate=True))
            _render_side_by_side(src_doc, tmp).save(dst, garbage=4, deflate=True)
        else:
            dst = out_dir / f"{src.stem}.translated.pdf"
            translated.save(dst, garbage=4, deflate=True)
        return dst

    return [await asyncio.to_thread(build)]
