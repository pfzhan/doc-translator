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

from ..languages import RTL_LANGUAGES

MATH_FONT_RE = re.compile(r"CMMI|CMSY|CMEX|MSBM|Math|Symbol|STIX|Cambria Math", re.I)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
NO_TEXT_RE = re.compile(r"^[\W\d_]*$")
# 行首列表标记：符号弹点（可不带空格）、短横线类（须带空格，避免误伤连字符单词）、数字/字母编号
BULLET_RE = re.compile(r"^\s*(?:[•◦▪‣·○■□►»]\s*|[-–—*]\s+|\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+)")
# 新段落的开头：大写字母、数字、CJK、引号/括号（小写字母开头多半是自动换行的续行）
SEG_START_RE = re.compile(r"^[A-Z0-9À-Þ぀-ヿ㐀-鿿가-힯(\"'“‘\[（【]")


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
        elif out.endswith("-") and re.match(r"[a-zA-ZÀ-ÿ]", line):
            out = out[:-1] + line  # 行尾连字符断词（德语等断词后首字母可能大写）
        elif CJK_RE.search(out[-1]) or CJK_RE.search(line[0]):
            out += line
        else:
            out += " " + line
    return out


def _split_lines(items: list[tuple[str, "pymupdf.Rect", list[dict]]]) -> list[list[int]]:
    """把一个文本块里的行分成独立段落（返回每组的行下标）。

    幻灯片里多个列表项、甚至不同的图注标签常被 PyMuPDF 归进同一块，整段翻译会把
    项目符号变成行内文字、把不相干的内容拼在一起。三处拆段：
    - 行首带列表标记；
    - 纯符号行（“=”“→”等）单独成段，之后被过滤器跳过、原样保留；
    - 行首明显左跳，且上行没撑到块的右边距（并列短标签被并进同一块）；两行都撑满
      右边距是首行缩进的自动换行，不能拆。跳幅超过约 3 个字号、且不是两行都撑满时
      也拆，用来抓住上行恰好是块里最宽行的并列标签。
    - 上行明显没到右边距（硬换行而非自动换行）且下行像新句子开头；上行以逗号等
      连接标点结尾的不算硬换行。
    自动换行的续行（小写开头、上行撑满）仍并入上一段。
    """
    if not items:
        return []
    max_x1 = max(r.x1 for _, r, _ in items)
    groups, cur = [], [0]
    for i in range(1, len(items)):
        text, rect, _ = items[i]
        ptext, prect, pspans = items[cur[-1]]
        psize = max((s["size"] for s in pspans), default=10)
        stripped = text.lstrip()
        prev_symbol = bool(NO_TEXT_RE.match(ptext.strip()))  # 纯符号行独立成段，后面的内容不和它拼
        hard_break = (
            max_x1 - prect.x1 > 3 * psize
            and SEG_START_RE.match(stripped)
            and not prev_symbol
            and not ptext.rstrip().endswith((",", "，", "、", ";", "；", "(", "+", "*", "/", "=", "<", ">"))
        )
        jump = prect.x0 - rect.x0
        prev_short = max_x1 - prect.x1 > 3 * psize
        both_full = max_x1 - prect.x1 <= 3 * psize and max_x1 - rect.x1 <= 3 * psize
        # 1–2em 的左跳是首行缩进；超过约 3 个字号才像换了一列标签
        left_jump = jump > 8 and not both_full and (prev_short or jump > 3 * psize)
        if BULLET_RE.match(text) or NO_TEXT_RE.match(stripped) or prev_symbol or hard_break or left_jump:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return groups


def _preview_kind(block: TextBlock, median: float) -> str:
    """短块里，明显大于正文的、以及略大于正文的粗体，当作小标题。

    只看粗体不够：作者名和图注标签也常是粗体，但字号不超过正文。
    """
    if len(block.text) >= 100:
        return "p"
    if block.size >= median * 1.3 or (block.bold and block.size >= median * 1.15):
        return "h2"
    return "p"


def extract_blocks(doc: pymupdf.Document) -> list[TextBlock]:
    blocks = []
    for pno, page in enumerate(doc):
        data = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES)
        for b in data["blocks"]:
            if b.get("type") != 0:
                continue
            items: list[tuple[str, pymupdf.Rect, list[dict]]] = []
            for ln in b["lines"]:
                if abs(ln["dir"][1]) > 0.01:  # 旋转 / 竖排文字不处理
                    continue
                text = "".join(s["text"] for s in ln["spans"])
                if not text.strip():
                    continue
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if spans:
                    items.append((text, pymupdf.Rect(ln["bbox"]), spans))
            for group in _split_lines(items):
                texts = [items[i][0] for i in group]
                line_rects = [items[i][1] for i in group]
                spans = [s for i in group for s in items[i][2]]
                text = _join_lines(texts)
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


def _render_translated(src_path: Path, blocks: list[TextBlock], translations: list[str],
                       target_lang: str = "") -> pymupdf.Document:
    rtl = target_lang.split("-")[0] in RTL_LANGUAGES
    # CJK 字体的行框比拉丁高（约 1.31em vs 1.16em），且字形顶部会越出给定区域：
    # 按原字号写入会和下一行叠在一起，字号缩小并下移补偿
    cjk = target_lang.split("-")[0] in ("zh", "ja", "ko")
    # lang 让 MuPDF 为中日韩选对字形；dir=rtl 让阿拉伯语、希伯来语从右往左排
    div_attrs = f' lang="{html.escape(target_lang)}"' if target_lang else ""
    if rtl:
        div_attrs += ' dir="rtl" style="text-align: right"'
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
            size = b.size * 0.88 if cjk else b.size
            css = (
                f"* {{font-family: sans-serif; font-size: {size}px; color: {b.color}; "
                f"font-weight: {weight}; line-height: 1.2; margin: 0; padding: 0;}}"
            )
            # 留一点余量，避免译文比原文长时被截断；放不下由 scale_low=0 自动缩小
            y0 = b.rect.y0 + b.size * 0.1 if cjk else b.rect.y0
            rect = pymupdf.Rect(b.rect.x0, y0, b.rect.x1 + 2, b.rect.y1 + b.size * 0.3)
            page.insert_htmlbox(rect, f"<div{div_attrs}>{body}</div>", css=css, scale_low=0)
    return doc


def _render_side_by_side(src: pymupdf.Document, translated: pymupdf.Document) -> pymupdf.Document:
    out = pymupdf.open()
    for i in range(len(src)):
        w, h = src[i].rect.width, src[i].rect.height
        page = out.new_page(width=w * 2, height=h)
        page.show_pdf_page(pymupdf.Rect(0, 0, w, h), src, i)
        page.show_pdf_page(pymupdf.Rect(w, 0, w * 2, h), translated, i)
    return out


async def translate_pdf(src: Path, out_dir: Path, runner, bilingual: bool, target_lang: str = "") -> list[Path]:
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
    median = sorted(b.size for b in blocks)[len(blocks) // 2]
    kinds = [_preview_kind(b, median) for b in blocks]
    translations = await runner.translate_all([b.text for b in blocks], kinds=kinds, preview=True)

    def build():
        translated = _render_translated(src, blocks, translations, target_lang)
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
