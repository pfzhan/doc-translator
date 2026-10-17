"""Markdown 翻译：按块切分，跳过代码块 / front matter / HTML 块，保留列表、引用、表格结构。"""
import re
from dataclasses import dataclass, field
from pathlib import Path

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*?)(\s+#+\s*)?$")
LIST_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?)(.*)$")
QUOTE_RE = re.compile(r"^(\s*(?:>\s?)+)(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
HR_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
HTML_BLOCK_RE = re.compile(r"^\s*<(/?[a-zA-Z][\w-]*|!--)")
# 链接引用定义：[id]: https://...。脚注 [^1]: 有正文，不算
LINK_DEF_RE = re.compile(r"^\s{0,3}\[(?!\^)[^\]]+\]:\s*\S")
# GitHub 提示框标记：> [!NOTE]
ALERT_RE = re.compile(r"^\[![A-Za-z]+\]$")
# 只有链接 / 图片 / 符号，没有可翻译文字
NO_TEXT_RE = re.compile(r"^[\W\d_]*$")


def _translatable(text: str) -> bool:
    stripped = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # 去掉图片
    stripped = re.sub(r"`[^`]*`", "", stripped)  # 去掉行内代码
    stripped = re.sub(r"https?://\S+", "", stripped)
    return bool(stripped.strip()) and not NO_TEXT_RE.match(stripped.strip())


def _interrupts_paragraph(line: str) -> bool:
    """CommonMark：引用、非空的无序列表项、从 1 开始的有序列表项可以紧跟在段落后面，不用空行隔开。"""
    if QUOTE_RE.match(line):
        return True
    m = LIST_RE.match(line)
    if not m or not m.group(2).strip():
        return False
    number = re.match(r"\s*(\d+)", m.group(1))
    return number is None or int(number.group(1)) == 1


@dataclass
class Block:
    kind: str  # raw / heading / para / list / quote / table
    lines: list[str]
    # 每个可翻译片段：(行号, 前缀, 文本)
    segments: list[tuple[int, str, str]] = field(default_factory=list)


def split_blocks(text: str) -> list[Block]:
    lines = text.split("\n")
    blocks: list[Block] = []
    i = 0
    # YAML front matter
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() in ("---", "..."):
                blocks.append(Block("raw", lines[: j + 1]))
                i = j + 1
                break

    para: list[str] = []

    def flush():
        if para:
            blocks.append(Block("para", para.copy()))
            para.clear()

    while i < len(lines):
        line = lines[i]
        m = FENCE_RE.match(line)
        if m:
            flush()
            fence = m.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(fence):
                j += 1
            blocks.append(Block("raw", lines[i : j + 1]))
            i = j + 1
            continue
        if not line.strip():
            flush()
            blocks.append(Block("raw", [line]))
            i += 1
            continue
        if HR_RE.match(line) or line.startswith("    ") and not para:
            # 分隔线 / 缩进代码块
            flush()
            blocks.append(Block("raw", [line]))
            i += 1
            continue
        if HEADING_RE.match(line):
            flush()
            blocks.append(Block("heading", [line]))
            i += 1
            continue
        if HTML_BLOCK_RE.match(line) and not para:
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            blocks.append(Block("raw", lines[i:j]))
            i = j
            continue
        if LINK_DEF_RE.match(line) and not para:
            # 连续几行定义并成一段会让后面的定义失效，原样保留
            j = i
            while j < len(lines) and LINK_DEF_RE.match(lines[j]):
                j += 1
            blocks.append(Block("raw", lines[i:j]))
            i = j
            continue
        if "|" in line and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            flush()
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                j += 1
            blocks.append(Block("table", lines[i:j]))
            i = j
            continue
        if para and _interrupts_paragraph(line):
            flush()
        if LIST_RE.match(line) and not para:
            j = i
            while j < len(lines) and lines[j].strip() and not FENCE_RE.match(lines[j]):
                # 列表后紧跟的标题 / 引用 / HTML 块 / 分隔线是新的块级结构，不能并进最后一个列表项
                if j > i and not LIST_RE.match(lines[j]) and (
                    HEADING_RE.match(lines[j]) or QUOTE_RE.match(lines[j])
                    or HTML_BLOCK_RE.match(lines[j]) or HR_RE.match(lines[j])
                ):
                    break
                j += 1
            blocks.append(Block("list", lines[i:j]))
            i = j
            continue
        if QUOTE_RE.match(line) and not para:
            j = i
            while j < len(lines) and QUOTE_RE.match(lines[j]):
                j += 1
            blocks.append(Block("quote", lines[i:j]))
            i = j
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


def _split_cells(line: str) -> tuple[str, list[str], str]:
    s = line.strip()
    lead = "|" if s.startswith("|") else ""
    trail = "|" if s.endswith("|") and len(s) > 1 else ""
    inner = s[len(lead) : len(s) - len(trail)]
    return lead, [c.strip() for c in re.split(r"(?<!\\)\|", inner)], trail


def collect_segments(blocks: list[Block]) -> list[str]:
    """填充每个块的 segments，返回全部待翻译文本。"""
    texts = []
    for b in blocks:
        if b.kind == "heading":
            m = HEADING_RE.match(b.lines[0])
            b.segments = [(0, m.group(1), m.group(2))]
        elif b.kind == "para":
            b.segments = [(0, "", " ".join(l.strip() for l in b.lines))]
        elif b.kind == "list":
            b.segments = [(idx, pre, text) for idx, pre, text, _ in _list_items(b.lines)]
        elif b.kind == "quote":
            b.segments = [(idx, qpre + mpre, text) for kind, idx, qpre, mpre, text, _ in _quote_units(b.lines)
                          if kind != "raw"]
        elif b.kind == "table":
            for idx, line in enumerate(b.lines):
                if idx == 1:
                    continue
                _, cells, _ = _split_cells(line)
                for c in cells:
                    b.segments.append((idx, "", c))
        if b.kind in ("heading", "para", "quote"):
            # 没有可翻译文字的块（纯图片、纯链接等）当作 raw 原样输出
            b.segments = [s for s in b.segments if _translatable(s[2])]
        texts.extend(s[2] for s in b.segments if _translatable(s[2]))
    return texts


def segment_kinds(blocks: list[Block]) -> list[str]:
    """和 collect_segments 返回的文本一一对应的段落类型，给预览用。"""
    kinds = []
    for b in blocks:
        for _, pre, text in b.segments:
            if not _translatable(text):
                continue
            if b.kind == "heading":
                kinds.append(f"h{min(pre.count('#'), 6)}")
            else:
                kinds.append({"list": "li", "table": "td", "quote": "quote"}.get(b.kind, "p"))
    return kinds


def _list_items(lines: list[str]) -> list[tuple[int, str, str, list[str]]]:
    """把列表块拆成列表项：(起始行号, 标记前缀, 合并后的文本, 原始行)。懒续行并入上一项。"""
    items = []
    for idx, line in enumerate(lines):
        m = LIST_RE.match(line)
        if m or not items:
            pre, text = (m.group(1), m.group(2)) if m else ("", line.strip())
            items.append((idx, pre, text, [line]))
        else:
            start, pre, text, raw = items[-1]
            items[-1] = (start, pre, text + " " + line.strip(), raw + [line])
    return items


def _quote_units(lines: list[str]) -> list[tuple[str, int, str, str, str, list[str]]]:
    """把引用块拆成 (类型, 起始行号, 引用前缀, 列表标记, 合并后的文本, 原始行)。

    类型 raw 原样输出：空的引用行、提示框标记、引用里的代码块。para 和 item 要翻译：
    空引用行之间的每一段、引用里的每个列表项各自成段，引用层级变了也另起一段。
    """
    units: list[tuple[str, int, str, str, str, list[str]]] = []
    fence = ""
    for idx, line in enumerate(lines):
        m = QUOTE_RE.match(line)
        qpre, body = m.group(1), m.group(2)
        fm = FENCE_RE.match(body)
        if fence or fm:
            if fence and body.lstrip().startswith(fence):
                fence = ""
            elif not fence:
                fence = fm.group(1)
            units.append(("raw", idx, qpre, "", body, [line]))
            continue
        if not body.strip() or ALERT_RE.match(body.strip()):
            units.append(("raw", idx, qpre, "", body, [line]))
            continue
        item = LIST_RE.match(body)
        if item:
            units.append(("item", idx, qpre, item.group(1), item.group(2), [line]))
            continue
        last = units[-1] if units else None
        if last and last[0] != "raw" and last[2].count(">") == qpre.count(">"):
            kind, start, lpre, mpre, text, raw = last
            units[-1] = (kind, start, lpre, mpre, text + " " + body.strip(), raw + [line])
        else:
            units.append(("para", idx, qpre, "", body.strip(), [line]))
    return units


def render(blocks: list[Block], tr: dict[str, str], bilingual: bool) -> str:
    out: list[str] = []

    def t(s: str) -> str:
        return tr.get(s, s)

    for b in blocks:
        if b.kind == "raw" or not b.segments:
            out.extend(b.lines)
        elif b.kind == "heading":
            _, pre, text = b.segments[0]
            if bilingual:
                out.append(b.lines[0])
                out.append("")
            out.append(pre + t(text))
        elif b.kind == "para":
            text = b.segments[0][2]
            if bilingual:
                out.extend(b.lines)
                out.append("")
            out.append(t(text))
        elif b.kind == "quote":
            for kind, _, qpre, mpre, text, raw in _quote_units(b.lines):
                if kind == "raw" or not _translatable(text):
                    out.extend(raw)
                elif kind == "item":
                    if bilingual:
                        out.append(qpre + mpre + text + "  ")
                        out.append(qpre + " " * len(mpre) + t(text))
                    else:
                        out.append(qpre + mpre + t(text))
                else:
                    if bilingual:
                        out.extend(raw)
                        out.append(qpre.rstrip())
                    out.append(qpre + t(text))
        elif b.kind == "list":
            for _, pre, text, raw in _list_items(b.lines):
                if not _translatable(text):
                    out.extend(raw)
                elif bilingual:
                    # 原文 + 硬换行 + 译文，保持在同一个列表项里
                    out.append(pre + text + "  ")
                    out.append(" " * len(pre) + t(text))
                else:
                    out.append(pre + t(text))
        elif b.kind == "table":
            for idx, line in enumerate(b.lines):
                if idx == 1:
                    out.append(line)
                    continue
                lead, cells, trail = _split_cells(line)
                new_cells = []
                for c in cells:
                    if not _translatable(c):
                        new_cells.append(c)
                    elif bilingual:
                        new_cells.append(f"{c}<br>{t(c)}")
                    else:
                        new_cells.append(t(c))
                row = " | ".join(new_cells)
                out.append((lead + " " if lead else "") + row + (" " + trail if trail else ""))
    return "\n".join(out)


async def translate_markdown(src: Path, out_dir: Path, runner, bilingual: bool) -> list[Path]:
    text = src.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    blocks = split_blocks(text)
    texts = collect_segments(blocks)
    # 标题：第一个一级标题，没有就用文件名
    h1 = next((b.segments[0][2] for b in blocks if b.kind == "heading" and b.lines[0].lstrip().startswith("# ")), "")
    runner.set_title(h1 or src.stem)
    results = await runner.translate_all(texts, kinds=segment_kinds(blocks), preview=True)
    tr = dict(zip(texts, results))
    suffix = "bilingual" if bilingual else "translated"
    dst = out_dir / f"{src.stem}.{suffix}{src.suffix}"
    dst.write_text(render(blocks, tr, bilingual), encoding="utf-8")
    return [dst]
