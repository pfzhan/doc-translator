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
from .pdf_types import Formula


def _snap_cut(text: str, cut: int) -> int:
    cut = max(0, min(len(text), cut))
    for pattern in (_SENTINEL_NUM_RE, STYLE_MARK_RE):
        for m in pattern.finditer(text):
            if m.start() < cut < m.end():
                return m.start() if cut - m.start() <= m.end() - cut else m.end()
    return cut


def _split_translation(t: str, ratio: float) -> tuple[str, str]:
    """Split the translation of a paragraph joined across pages back into two parts by ratio: at the
    punctuation mark closest to the ratio if there is one, otherwise at a space, otherwise at the exact
    position. The cut avoids formula sentinels, and style marks on both sides are balanced."""
    target = len(t) * ratio
    best = None
    for m in re.finditer(r"[。！？；，、：.!?;,:]", t):
        if best is None or abs(m.end() - target) < abs(best - target):
            best = m.end()
    if best is not None and abs(best - target) <= len(t) * 0.3:
        cut = _snap_cut(t, best)
        return balance_marks(t[:cut], t[cut:])
    # Latin-script text is cut at a space
    spaces = [m.start() for m in re.finditer(r"\s", t)]
    if spaces:
        cut = _snap_cut(t, min(spaces, key=lambda s: abs(s - target)))
        return balance_marks(t[:cut], t[cut:])
    cut = _snap_cut(t, round(target))
    return balance_marks(t[:cut], t[cut:])


# A stray {vN} placeholder in a translation (the model wrote a placeholder from another segment)
_STRAY_PLACEHOLDER_RE = re.compile(r"\{\s*v\s*\d+\s*\}")


def _restore_placeholders(translation: str, formulas: list[Formula], lenient: bool = False) -> str | None:
    """Restore {vN} after translation (tolerating spaces the model put inside a placeholder). Returns None,
    falling back to the original, when a placeholder is missing. Each placeholder becomes an image sentinel
    (\\x01i\\x02N\\x01/i\\x02), which the renderer replaces with the <img> of the formula. Extra placeholders
    the model invented are deleted.

    With lenient=True, the translation is accepted even if it lost a small formula of one or two glyphs
    (an equals sign, a single variable, a formula hole in bitmap text): the model most likely wrote it out
    as text from the context, which is better than falling back to the original for the whole paragraph.
    """
    for n, f in enumerate(formulas, 1):
        pattern = re.compile(rf"\{{\s*v\s*{n}\s*\}}")
        if not pattern.search(translation):
            if lenient and _small_formula(f):
                continue
            return None
        if _ends_in_relation(f):
            # The formula image ends with an equals sign (y_c =) and the number after it is body text (1 if ...).
            # The model does not know the placeholder already holds the sign and writes "为 1" again, which
            # reads as "y_c = 为 1"
            translation = re.sub(rf"(\{{\s*v\s*{n}\s*\}}\s*){_RELATION_WORD_RE}", r"\1", translation, count=1)
        translation = pattern.sub(lambda _: f"\x01i\x02{n}\x01/i\x02", translation, count=1)
    return re.sub(r"\{\s*v\s*\d+\s*\}", "", translation)


# What may sit between placeholders in a chain of formulas: whitespace, style marks, single-letter variables
# and operators (σ(Σ...) {b}W{/b} x_i + b). Two or more letters make a word (as, the) and do not count
_CHAIN_GLUE = (
    r"(?:\s+|\{/?[biu]\}|\{/[cz]\}|\{c[0-9a-f]{6}\}|\{z\d+\}"
    r"|(?<![A-Za-z0-9])[A-Za-z0-9](?![A-Za-z0-9])|[+\-−=·×*/,.()\[\]|]{1,3})"
)
_CHAIN_RE = re.compile(rf"\{{v\d+\}}(?:{_CHAIN_GLUE})*?\{{v\d+\}}(?:(?:{_CHAIN_GLUE})*?\{{v\d+\}})*")
_PLACEHOLDER_NUM_RE = re.compile(r"\{\s*v\s*(\d+)\s*\}")


def _keep_formula_chains(translation: str, source: str) -> str:
    """A chain of adjacent formulas in the source ({v2} {b}W{/b}{v3}, one formula cut into pieces by a variable
    set at body size) sometimes comes back with the pieces reordered ({b}W{/b}{v3}{v2}), which draws them in
    the wrong order. When the pieces are still adjacent in the translation and only their order differs, the
    source order is restored. Chains with Chinese text between the pieces are left alone."""
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
            continue  # other text in between
        found = [int(n) for n in _PLACEHOLDER_NUM_RE.findall(middle)]
        if found == order:
            continue
        # Extend over adjacent variables and marks on both sides, so the source's W is not doubled after the swap
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


# "为", "等于", "大于" and similar words after a formula ending in a relation sign and before a number
_RELATION_WORD_RE = r"(?:(?:大于|小于|不)?等于|约为|大于|小于|为|是)\s*(?=[-−+]?\d)"


def _ends_in_relation(formula: Formula) -> bool:
    text = re.sub(r"\x01[^\x02]*\x02", "", str(formula.get("text") or "")).strip()
    return text.endswith(("=", "<", ">", "≤", "≥", "≠", "≈"))


def _small_formula(formula: Formula) -> bool:
    return len((formula.get("text") or "").strip()) <= 2


async def _retry_lost_placeholders(
    runner: object,
    sent_texts: list[str],
    sent_formulas: list[list[Formula]],
    raw: list[str],
) -> None:
    """Translate units whose translation lost a {vN} once more, one at a time. Results with all placeholders
    replace the entries in raw and go into the cache.

    No request is sent when only rendering from the cache (cache_only).
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
            # The preview shows _done; store the new translation there so placeholder restoration matches later
            done = getattr(runner, "_done", None)
            if isinstance(done, dict) and done.get(sent_texts[i]) == raw[i]:
                done[sent_texts[i]] = text
            raw[i] = text
            fixed[sent_texts[i]] = text
    cache = getattr(runner, "keyed_cache", None) or getattr(runner, "cache", None)
    if fixed and cache is not None:
        cache.put_many(translator.cache_key, fixed)


_CN_NUM = r"[0-9零〇一二三四五六七八九十百]+"
# The model writes 2.1 as "第二点一节", "四点二" or "一.三.", 3.1.3 as "第三章第一节第三小节", and 8 as "八,".
# All of these are stripped from the start of a heading and replaced with the source number.
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


def _take_owner(text: str, formulas: list[Formula], owner: int) -> tuple[str, list[str]]:
    """Keep the formula sentinels that belong to owner and take the others out unchanged, in their order."""
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


def _localize_sentinels(text: str, formulas: list[Formula], owner: int) -> tuple[str, list[Formula]]:
    """Renumber sentinels by their index in this block's formula list; rendering looks images up by index."""
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
    """Join the text across a page boundary into one paragraph. No space is added next to Chinese characters,
    which would separate "相当" from "不错"."""
    split = split_page_boundary(text)
    if split is None:
        return strip_boundary(text)
    return _join_lines([split[0], split[1]])


def _assign_restored_parts(
    restored: str,
    unit: list[TextBlock],
    formulas: list[Formula],
    apart: bool = False,
) -> tuple[list[str], list[list[Formula]]]:
    """The whole translation of a unit across two pages stays in the first block. The original text on the
    next page is only erased; the translation is no longer split back at the boundary.

    The same applies when the model dropped the {|} mark. Sentinels keep their numbers within the unit and
    the formula images go with the paragraph. With apart, the two blocks are separated by a display formula
    or an example (see _seam_apart); the text is split at the boundary and each formula sentinel goes with
    its own block.
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
    """Put the peeled number back at the start of the translation."""
    return f"{label} {translation.lstrip()}" if label else translation


def _normalize_heading_number(src_text: str, translation: str) -> str:
    """Replace a chapter or section number at the start of a heading translation with the source number
    ("第二章" -> "2").

    No number is added when the translation has none. A translation identical to the source is returned
    unchanged, so that skipped blocks are not typeset again.
    """
    if translation.strip() == src_text.strip():
        return translation
    m = _SRC_HEAD_NUM_RE.match(src_text.strip())
    if not m or len(src_text) > 60 or _ends_sentence(src_text):
        return translation
    # The number may be wrapped in size or bold marks: {z125}二{/z} 算法
    lead = re.match(r"^\s*((?:\{[bizu]\d*\})*)", translation)
    # Every pattern part is optional (star / \s*), so it matches any string
    assert lead is not None
    body = translation[lead.end() :]
    if not _TRANS_HEAD_NUM_RE.match(body):
        return translation
    rest = _TRANS_HEAD_NUM_RE.sub("", body, count=1)
    close = re.match(r"^((?:\{/[bizu]\})*)\s*", rest)
    assert close is not None  # all optional: always matches
    rest = rest[close.end() :]
    return f"{lead.group(1)}{m.group(1)}{close.group(1)} {rest}" if rest else translation


def _heading_number_for(kind: str, src_text: str, translation: str) -> str:
    if kind == "p":
        # Section headings at body size ('2.1 Model' has the body font size, so kind is p): dotted numbers
        # (2.1, 3.2.1) almost never start a body sentence, while a bare integer may be a quantity
        # ('2 samples') and is left alone. The translation is changed only if it starts with a section number.
        if not (re.match(r"^\d+(?:\.\d+)+\s", src_text.strip()) and _TRANS_HEAD_NUM_RE.match(translation.strip())):
            return translation
    return _normalize_heading_number(src_text, translation)


def _list_number(src_text: str, translation: str) -> str:
    """Restore the source number when a list number 1. became "1。" or "1、", or Figure 1 became "图一"."""
    caption = _CAPTION_NUM_RE.match(src_text)
    if caption:
        return _CN_CAPTION_RE.sub(rf"\g<1>\g<2> {caption.group(1)}", translation, count=1)
    m = re.match(r"\s*(\d{1,3})([.)])\s", src_text)
    if not m:
        return translation
    return re.sub(rf"^(\s*){m.group(1)}\s*[。．、]\s*", rf"\g<1>{m.group(1)}{m.group(2)} ", translation, count=1)


# Caption labels often carry size or bold marks: {z110}图一:{/z}
_CN_CAPTION_RE = re.compile(r"^(\s*(?:\{/?[bizu]\d*\}\s*)*)([图表])\s*[一二三四五六七八九十百零〇两]+")


def _strip_boundary_echo(prev_src: str, cur_src: str, prev_t: str, cur_t: str) -> str:
    """Remove connecting words the model repeated across a block boundary.

    Repetition only happens when one sentence was cut into two blocks: the source of this block continues
    the sentence of the previous one and starts with a lowercase letter. A paragraph after a heading often
    starts with words from the heading, which is not repetition. Identical source text hits the same cache
    entry and must give the same translation, so it is not removed. A short overlap without punctuation
    ("如下所示") is a common phrase rather than repetition at the break. An overlap covering the whole
    block is not removed.
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
    """Heading number fixes and boundary deduplication happen after the preview was filled, so write the final
    translation back into the preview by segment index."""
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
    """Remove all color marks from a translation where the model broke them, so the text takes the block's
    color: when opening and closing marks do not pair up, or a color is not in the source. One missing {/c}
    would color the rest of the paragraph."""
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
    """The model sometimes wraps a mark in an extra pair of braces ({{b}填充词{/b}, {{c0000ff}Fox Tree{/c}}),
    which leaves stray { and } in the output. When the source has no braces outside its marks, all braces
    outside the marks are removed from the translation."""
    if "{" not in translation and "}" not in translation:
        return translation
    if re.search(r"[{}]", _MARKUP_RE.sub("", sent)):
        return translation
    pieces = _MARKUP_RE.split(translation)
    marks = _MARKUP_RE.findall(translation)
    cleaned = [re.sub(r"[{}]", "", piece) for piece in pieces]
    return "".join(piece + mark for piece, mark in zip(cleaned, [*marks, ""]))
