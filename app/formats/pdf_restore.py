"""Put the model's translation back together: restore formula placeholders and section numbers,
split units that span two blocks, and update the preview."""

import asyncio
import re

from ..runner import clean_translation
from .pdf_blocks import (
    _CAPTION_NUM_RE,
    _MARKUP_RE,
    _SENTINEL_NUM_RE,
    _SRC_HEAD_NUM_RE,
    SPILL_TAIL,
    TextBlock,
    _ends_sentence,
    _join_lines,
)
from .pdf_flow import (
    COLOR_MARK_RE,
    STYLE_MARK_RE,
    balance_marks,
    strip_style_marks,
)
from .pdf_runs import (
    split_page_boundary,
    strip_boundary,
)


def _snap_cut(text: str, cut: int) -> int:
    cut = max(0, min(len(text), cut))
    for pattern in (_SENTINEL_NUM_RE, STYLE_MARK_RE):
        for m in pattern.finditer(text):
            if m.start() < cut < m.end():
                return m.start() if cut - m.start() <= m.end() - cut else m.end()
    return cut


def _split_translation(t: str, ratio: float) -> tuple[str, str]:
    """把跨页合并段的译文按比例切回两页：优先在比例附近的句读点断开，
    没有合适标点时按空格，再不行硬切。切点避开公式哨兵，且两侧样式标记配对。"""
    target = len(t) * ratio
    best = None
    for m in re.finditer(r"[。！？；，、：.!?;,:]", t):
        if best is None or abs(m.end() - target) < abs(best - target):
            best = m.end()
    if best is not None and abs(best - target) <= len(t) * 0.3:
        cut = _snap_cut(t, best)
        return balance_marks(t[:cut], t[cut:])
    # 拉丁文本按空格切
    spaces = [m.start() for m in re.finditer(r"\s", t)]
    if spaces:
        cut = _snap_cut(t, min(spaces, key=lambda s: abs(s - target)))
        return balance_marks(t[:cut], t[cut:])
    cut = _snap_cut(t, round(target))
    return balance_marks(t[:cut], t[cut:])


# 译文里游离的 {vN} 占位符（模型把占位符写进了别的段）
_STRAY_PLACEHOLDER_RE = re.compile(r"\{\s*v\s*\d+\s*\}")


def _restore_placeholders(translation: str, formulas: list[dict], lenient: bool = False) -> str | None:
    """译后恢复 {vN}（容忍模型在占位符里塞的空格）；有占位符丢失时返回 None（回退原文）。
    恢复出来的是图片哨兵（\\x01i\\x02N\\x01/i\\x02），渲染层换成对应公式的 <img>。
    译文里多出来的幻觉占位符直接删掉。

    lenient=True 时，丢的只是一两个字形的小公式（等号、单个变量、点阵字的公式洞）也收下译文：
    模型多半按上下文把它写成了文字，比整段退回原文好。
    """
    for n, f in enumerate(formulas, 1):
        pattern = re.compile(rf"\{{\s*v\s*{n}\s*\}}")
        if not pattern.search(translation):
            if lenient and _small_formula(f):
                continue
            return None
        if _ends_in_relation(f):
            # 式子图里带着等号（y_c =），后面的数字是正文字（1 if …）。模型不知道占位符里已有等号，
            # 又写一遍「为 1」，读出来是「y_c = 为 1」
            translation = re.sub(rf"(\{{\s*v\s*{n}\s*\}}\s*){_RELATION_WORD_RE}", r"\1", translation, count=1)
        translation = pattern.sub(lambda _: f"\x01i\x02{n}\x01/i\x02", translation, count=1)
    return re.sub(r"\{\s*v\s*\d+\s*\}", "", translation)


# 公式串里夹在占位符之间的东西：空白、样式标记、单个字母的变量和运算符（σ(Σ…) {b}W{/b} x_i + b）。
# 两个字母以上的是词（as、the），不算
_CHAIN_GLUE = r"(?:\s+|\{/?[biu]\}|\{/[cz]\}|\{c[0-9a-f]{6}\}|\{z\d+\}|(?<![A-Za-z0-9])[A-Za-z0-9](?![A-Za-z0-9])|[+\-−=·×*/,.()\[\]|]{1,3})"
_CHAIN_RE = re.compile(rf"\{{v\d+\}}(?:{_CHAIN_GLUE})*?\{{v\d+\}}(?:(?:{_CHAIN_GLUE})*?\{{v\d+\}})*")
_PLACEHOLDER_NUM_RE = re.compile(r"\{\s*v\s*(\d+)\s*\}")


def _keep_formula_chains(translation: str, source: str) -> str:
    """原文里一串挨着的式子（{v2} {b}W{/b}{v3}，一个公式被正文字号的变量切成几截），
    模型有时把几截的次序调了（{b}W{/b}{v3}{v2}），画出来前后颠倒。
    译文里这几截仍挨在一起、只是次序不同时，按原文的次序换回去。中间夹着中文的不动。"""
    glue = re.compile(rf"(?:{_CHAIN_GLUE})")
    for chain in _CHAIN_RE.finditer(source):
        order = [int(n) for n in _PLACEHOLDER_NUM_RE.findall(chain.group(0))]
        spots = {int(m.group(1)): m for m in _PLACEHOLDER_NUM_RE.finditer(translation) if int(m.group(1)) in order}
        if len(spots) != len(order):
            continue
        start = min(m.start() for m in spots.values())
        end = max(m.end() for m in spots.values())
        middle = translation[start:end]
        rest = _PLACEHOLDER_NUM_RE.sub("", middle)
        if glue.sub("", rest):
            continue  # 中间夹着别的字
        found = [int(n) for n in _PLACEHOLDER_NUM_RE.findall(middle)]
        if found == order:
            continue
        # 向两边扩到挨着的变量和标记，免得原文的 W 换回去以后又多出一个
        while True:
            before = re.search(rf"(?:{_CHAIN_GLUE})$", translation[:start])
            if before is None or not before.group(0).strip():
                break
            start = before.start()
        while True:
            after = glue.match(translation, end)
            if after is None or not after.group(0).strip() or after.group(0) in ".,":
                break
            end = after.end()
        translation = translation[:start] + chain.group(0) + translation[end:]
    return translation


# 跟在以关系符收尾的式子后面、数字前面的「为」「等于」「大于」
_RELATION_WORD_RE = r"(?:(?:大于|小于|不)?等于|约为|大于|小于|为|是)\s*(?=[-−+]?\d)"


def _ends_in_relation(formula: dict) -> bool:
    text = re.sub(r"\x01[^\x02]*\x02", "", str(formula.get("text") or "")).strip()
    return text.endswith(("=", "<", ">", "≤", "≥", "≠", "≈"))


def _small_formula(formula: dict) -> bool:
    return len((formula.get("text") or "").strip()) <= 2


async def _retry_lost_placeholders(
    runner: object,
    sent_texts: list[str],
    sent_formulas: list[list[dict]],
    raw: list[str],
) -> None:
    """译文丢了 {vN} 的单元单独再翻一次，补上占位符的结果替换 raw 并写进缓存。

    只排版（cache_only）时不请求翻译服务。
    """
    translator = getattr(runner, "translator", None)
    if translator is None or getattr(runner, "cache_only", False):
        return
    lost = [
        i
        for i, (t, formulas) in enumerate(zip(raw, sent_formulas))
        if formulas and _restore_placeholders(t, formulas) is None
    ]
    if not lost:
        return
    results = await asyncio.gather(
        *(translator.translate_batch([sent_texts[i]]) for i in lost),
        return_exceptions=True,
    )
    fixed: dict[str, str] = {}
    for i, result in zip(lost, results):
        if not isinstance(result, list) or not result or not (result[0] or "").strip():
            continue
        text = clean_translation(sent_texts[i], result[0], translator.target_lang)
        if _restore_placeholders(text, sent_formulas[i]) is not None:
            # 预览按 _done 显示，换成新译文，后面恢复占位符时才对得上
            done = getattr(runner, "_done", None)
            if isinstance(done, dict) and done.get(sent_texts[i]) == raw[i]:
                done[sent_texts[i]] = text
            raw[i] = text
            fixed[sent_texts[i]] = text
    cache = getattr(runner, "keyed_cache", None) or getattr(runner, "cache", None)
    if fixed and cache is not None:
        cache.put_many(translator.cache_key, fixed)


_CN_NUM = r"[0-9零〇一二三四五六七八九十百]+"
# 模型会把 2.1 写成 第二点一节、四点二、一.三.，把 3.1.3 写成 第三章第一节第三小节，把 8 写成 八,。
# 都从标题开头剥掉，改回原文编号。
_CN_DOT = r"[零〇一二三四五六七八九十]+"
_TRANS_HEAD_NUM_RE = re.compile(
    r"^(?:"
    r"第" + _CN_NUM + r"点" + _CN_NUM + r"节?"
    r"|第" + _CN_NUM + r"(?:章|节|小节)(?:\s*第" + _CN_NUM + r"(?:节|小节))*"
    r"|第\d+(?:\.\d+)+节?"
    r"|" + _CN_NUM + r"(?:点" + _CN_NUM + r")+"
    r"|第?" + _CN_NUM + r"[章节]"
    r"|" + _CN_DOT + r"(?:[.．]" + _CN_DOT + r")+\.?"
    r"|" + _CN_NUM + r"\s*[,，、]"
    r"|\d+(?:\.\d+)*\.?[章节]?"
    r"|" + _CN_DOT + r"(?=\s|\{/)"
    r")\s*"
)


def _take_owner(text: str, formulas: list[dict], owner: int) -> tuple[str, list[str]]:
    """留下属于 owner 的公式哨兵，其余原样抽出（保持相对顺序）。"""
    kept: list[str] = []
    foreign: list[str] = []
    pos = 0
    for m in _SENTINEL_NUM_RE.finditer(text):
        kept.append(text[pos : m.start()])
        n = int(m.group(1))
        if 1 <= n <= len(formulas) and formulas[n - 1].get("owner", 0) == owner:
            kept.append(m.group(0))
        elif 1 <= n <= len(formulas):
            foreign.append(m.group(0))
        pos = m.end()
    kept.append(text[pos:])
    return "".join(kept), foreign


def _localize_sentinels(text: str, formulas: list[dict], owner: int) -> tuple[str, list[dict]]:
    """哨兵改成本块公式列表的序号，渲染时按下标取图。"""
    local = [f for f in formulas if f.get("owner", 0) == owner]
    index_map: dict[int, int] = {}
    local_n = 1
    for i, rec in enumerate(formulas, 1):
        if rec.get("owner", 0) == owner:
            index_map[i] = local_n
            local_n += 1

    def repl(m: re.Match[str]) -> str:
        mapped = index_map.get(int(m.group(1)))
        if mapped is None:
            return ""
        return f"\x01i\x02{mapped}\x01/i\x02"

    return _SENTINEL_NUM_RE.sub(repl, text), local


def _join_boundary(text: str) -> str:
    """跨页分界收成一段。汉字两侧不加空格，否则「相当」和「不错」会分开。"""
    split = split_page_boundary(text)
    if split is None:
        return strip_boundary(text)
    return _join_lines([split[0], split[1]])


def _assign_restored_parts(
    restored: str,
    unit: list[TextBlock],
    formulas: list[dict],
    apart: bool = False,
) -> tuple[list[str], list[list[dict]]]:
    """跨页单元的译文整段留在起始块。续页只擦原文，不再按分界切回两页。

    模型丢掉 {| } 时同样整段留下。哨兵仍按单元内的序号，公式图跟这段走。
    apart 时两块中间隔着行间公式、例句（见 _seam_apart），按分界切回两块，公式哨兵跟着自己的块走。
    """
    if len(unit) != 2:
        return [strip_boundary(restored)], [formulas]
    split = split_page_boundary(restored) if apart else None
    if split is None:
        return [_join_boundary(restored), SPILL_TAIL], [formulas, []]
    left, to_right = _take_owner(split[0], formulas, 0)
    right, to_left = _take_owner(split[1], formulas, 1)
    left_text, left_formulas = _localize_sentinels(left + "".join(to_left), formulas, 0)
    right_text, right_formulas = _localize_sentinels("".join(to_right) + right, formulas, 1)
    return [left_text.strip(), right_text.strip()], [left_formulas, right_formulas]


def _with_label(label: str, translation: str) -> str:
    """把摘下的编号接回译文开头。"""
    return f"{label} {translation.lstrip()}" if label else translation


def _normalize_heading_number(src_text: str, translation: str) -> str:
    """标题译文开头已有章节号时改回原文编号（'第二章' → '2'）。

    没有章节号就不加前缀。译文与原文相同则原样返回，避免跳过块被重新排版。
    """
    if translation.strip() == src_text.strip():
        return translation
    m = _SRC_HEAD_NUM_RE.match(src_text.strip())
    if not m or len(src_text) > 60 or _ends_sentence(src_text):
        return translation
    # 编号可能包在字号、粗体标记里：{z125}二{/z} 算法
    lead = re.match(r"^\s*((?:\{[bizu]\d*\})*)", translation)
    body = translation[lead.end() :]
    if not _TRANS_HEAD_NUM_RE.match(body):
        return translation
    rest = _TRANS_HEAD_NUM_RE.sub("", body, count=1)
    close = re.match(r"^((?:\{/[bizu]\})*)\s*", rest)
    rest = rest[close.end() :]
    return f"{lead.group(1)}{m.group(1)}{close.group(1)} {rest}" if rest else translation


def _heading_number_for(kind: str, src_text: str, translation: str) -> str:
    if kind == "p":
        # 正文尺寸的节标题（'2.1 Model' 与正文同字号，kind 是 p）：带点节号
        # （2.1、3.2.1）几乎不会出现在正文句首；bare 整数开头可能是正文数量
        # （'2 samples'），不动。且译文以章节号开头才改写。
        if not (re.match(r"^\d+(?:\.\d+)+\s", src_text.strip()) and _TRANS_HEAD_NUM_RE.match(translation.strip())):
            return translation
    return _normalize_heading_number(src_text, translation)


def _list_number(src_text: str, translation: str) -> str:
    """列表编号 1. 被译成 1。或 1、，图号 Figure 1 被译成 图一，都改回原文的数字。"""
    caption = _CAPTION_NUM_RE.match(src_text)
    if caption:
        return _CN_CAPTION_RE.sub(rf"\g<1>\g<2> {caption.group(1)}", translation, count=1)
    m = re.match(r"\s*(\d{1,3})([.)])\s", src_text)
    if not m:
        return translation
    return re.sub(rf"^(\s*){m.group(1)}\s*[。．、]\s*", rf"\g<1>{m.group(1)}{m.group(2)} ", translation, count=1)


# 图注标签常带字号或粗体标记：{z110}图一:{/z}
_CN_CAPTION_RE = re.compile(r"^(\s*(?:\{/?[bizu]\d*\}\s*)*)([图表])\s*[一二三四五六七八九十百零〇两]+")


def _strip_boundary_echo(prev_src: str, cur_src: str, prev_t: str, cur_t: str) -> str:
    """去掉模型在段界复读的连接语。

    复读只发生在一句话被切成两块时，这一块原文接着上一块的句子，以小写字母开头。
    标题后面另起的一段，开头常常就是标题里的词，不是复读。
    相同原文会命中同一条缓存，译文必然相同，不能删。无标点的短重叠
    （「如下所示」）是常见短语，也不是断点复读。整段重叠不删。
    """
    if not prev_t or not cur_t or prev_src.strip() == cur_src.strip() or "\x01" in cur_t[:40]:
        return cur_t
    if not re.match(r"(?:\{[^{}\s]{1,10}\}|\s)*[a-z]", cur_src):
        return cur_t
    limit = min(40, len(prev_t), len(cur_t) - 1)
    for k in range(limit, 3, -1):
        overlap = cur_t[:k]
        if not prev_t.endswith(overlap) or overlap in prev_src:
            continue
        if k <= 8 and not re.search(r"[，。；：、,.;:）)\]]", overlap):
            continue
        return cur_t[k:]
    return cur_t


def _sync_unit_preview(
    runner: object,
    units: list[list[TextBlock]],
    sent_texts: list[str],
    per_block: dict[int, str],
) -> None:
    """标题改写和段界去重发生在预览写入之后，按段落序号把最终译文补回去。"""
    overrides = getattr(runner, "_index_overrides", None)
    done = getattr(runner, "_done", None)
    if not isinstance(overrides, dict) or not isinstance(done, dict):
        return
    for ui, (unit, sent_text) in enumerate(zip(units, sent_texts)):
        pieces = [part for b in unit if (part := per_block.get(id(b), b.text)) != SPILL_TAIL]
        shown = strip_style_marks(_join_lines(pieces))
        current = done.get(sent_text)
        if current is not None and shown != current:
            overrides[ui] = shown


def _sound_colors(translation: str, sent: str) -> str:
    """模型弄乱的颜色标记整段去掉，字按块的颜色写：开收对不上，或者用了原文里没有的颜色。
    少一个 {/c}，后半段都会染上色。"""
    marks = COLOR_MARK_RE.findall(translation)
    if not marks:
        return translation
    allowed = set(COLOR_MARK_RE.findall(sent))
    depth = 0
    for mark in marks:
        depth += -1 if mark == "{/c}" else 1
        if depth not in (0, 1) or mark not in allowed:
            return COLOR_MARK_RE.sub("", translation)
    return translation if depth == 0 else COLOR_MARK_RE.sub("", translation)


def _drop_stray_braces(translation: str, sent: str) -> str:
    """模型偶尔在标记外面多套一层括号（{{b}填充词{/b}、{{c0000ff}Fox Tree{/c}}），
    写出来就剩下孤零零的 { 和 }。原文标记以外没有花括号时，译文标记以外的花括号都去掉。"""
    if "{" not in translation and "}" not in translation:
        return translation
    if re.search(r"[{}]", _MARKUP_RE.sub("", sent)):
        return translation
    pieces = _MARKUP_RE.split(translation)
    marks = _MARKUP_RE.findall(translation)
    cleaned = [re.sub(r"[{}]", "", piece) for piece in pieces]
    return "".join(piece + mark for piece, mark in zip(cleaned, [*marks, ""]))
