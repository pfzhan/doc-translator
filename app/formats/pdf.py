"""PDF 翻译，保留原版式。

1. 用 PyMuPDF 提取每页的文本块（坐标、字号、颜色、粗体）。
2. 跳过整块公式、纯数字、旋转文字。行内公式是段落里的一个 run，译后按原样贴回。
3. 译文版：擦掉原文字（图片和矢量图形保留），在原位置写入译文。
   放不下时先在本栏内右扩（不盖住图和其他文字），再压行距，最后缩字号。
   有线表格按格翻译，写回时不越出格子。重复的页眉页脚和页码保留原文。
4. 双语版：原页面和译文页面左右并排放在同一页上，方便对照阅读。
"""

import asyncio
import contextlib
import functools
import hashlib
import os
import pickle
import re
import tempfile
import weakref
import zlib
from dataclasses import replace
from pathlib import Path

import pymupdf

from .. import background
from .pdf_blocks import _CAMEL_WORDS, SPILL_TAIL, _join_lines
from .pdf_crop import _crop_formulas
from .pdf_flow import (
    separate_after_formula,
    strip_style_marks,
)
from .pdf_output import _compact, _drop_mupdf_cache, _render_by_parts, _render_side_by_side, translate_bookmarks
from .pdf_restore import (
    _STRAY_PLACEHOLDER_RE,
    _assign_restored_parts,
    _drop_stray_braces,
    _heading_number_for,
    _keep_formula_chains,
    _list_number,
    _restore_placeholders,
    _retry_lost_placeholders,
    _sound_colors,
    _strip_boundary_echo,
    _sync_unit_preview,
    _with_label,
)
from .pdf_units import _PdfPlan, _plan_document
from .pdf_vision import FormulaModel, ModelBoxes
from .pdf_write import _render_translated

_CAMEL_RE = re.compile(r"[A-Za-z]*[a-z][A-Z][A-Za-z]*")


# PyMuPDF 不支持在几个线程里同时调 MuPDF。它的 find_tables 临时打开进程全局的
# small_glyph_heights，另一个任务这时提取出的字框就变矮；两边的开、关交错时开关还会一直
# 开着，之后进程里的每个任务都按矮字框分段，重启服务才恢复
_MUPDF_TURNS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


def mupdf_turn() -> asyncio.Lock:
    """用 MuPDF 前先 async with 拿到它，同一时刻只有一个任务在用。"""
    loop = asyncio.get_running_loop()
    turn = _MUPDF_TURNS.get(loop)
    if turn is None:
        turn = _MUPDF_TURNS[loop] = asyncio.Lock()
    return turn


@contextlib.asynccontextmanager
async def _job_turn(runner):
    """轮到这个任务用 MuPDF。别的任务在读写一本大书时要等几分钟，任务页上显示在等。"""
    turn = mupdf_turn()
    if turn.locked():
        runner.report_stage("等待其他任务")
    async with turn:
        yield


def _plan_pdf(
    src: Path,
    runner,
    layout_model: str,
    formula_model: FormulaModel | None = None,
    boxes: ModelBoxes | None = None,
) -> _PdfPlan:
    """送翻前的部分：读版面、定角色、并跨页段落，给每个单元打好占位符。
    断行处的连字符要不要留，看全文里有没有接起来的词（见 _join_lines），整个分析期间都按这一篇的词。
    boxes 是存模型框的地方，给了时两个模型的框读回或存下（见 ModelBoxes）。"""
    src_doc = pymupdf.open(src)
    if src_doc.needs_pass:
        raise ValueError("不支持有密码保护的 PDF")
    words = frozenset(word for page in src_doc for word in _CAMEL_RE.findall(page.get_text("text")))
    token = _CAMEL_WORDS.set(words)
    try:
        return _plan_document(src, src_doc, runner, layout_model, formula_model, boxes)
    finally:
        _CAMEL_WORDS.reset(token)


def _saved_plan(
    src: Path,
    runner,
    layout_model: str,
    path: Path | None,
    formula_model: FormulaModel | None = None,
) -> _PdfPlan:
    """暂停、中断后继续，或生成另一种版本时，原文件、两个模型和代码都没变，就读回上次的分析结果。
    一本几百页的书逐页读版面要一分多钟，读回只要一两秒。path 为 None 时不存。
    改了代码要重新分析时，模型框出的框存在旁边的 boxes.json 里（见 ModelBoxes），读回来只重跑规则。"""
    if path is None:
        return _plan_pdf(src, runner, layout_model, formula_model)
    key = _plan_key(src, layout_model, formula_model)
    plan = _load_plan(path, key, src, runner)
    if plan is not None:
        runner.set_title(plan.title)
        return plan
    plan = _plan_pdf(src, runner, layout_model, formula_model, ModelBoxes(path.with_name("boxes.json"), src))
    _save_plan(path, key, plan)
    return plan


def _plan_key(src: Path, layout_model: str, formula_model: FormulaModel | None = None) -> str:
    digest = hashlib.sha1(src.read_bytes())
    digest.update(_code_version().encode())
    # 两个模型分开记：只开了公式模型和只开了版面模型的分析结果不一样。公式模型换了输入尺寸也要重新分析
    models = [("layout", layout_model)]
    if formula_model is not None:
        models.append((f"formula@{formula_model.size}", formula_model.path))
    for tag, path in models:
        model = Path(path) if path else None
        if model is not None and model.is_file():
            stat = model.stat()
            digest.update(f"{tag}:{model.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


@functools.cache
def _code_version() -> str:
    """程序代码和 PyMuPDF 的版本。改了代码，旧的分析结果就作废。"""
    digest = hashlib.sha1(str(pymupdf.VersionBind).encode())
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _save_plan(path: Path, key: str, plan: _PdfPlan) -> None:
    """块按 id 记的两张表存成序号，读回后块是新对象。先写临时文件再改名，写到一半退出不留坏文件。
    键不压缩，读的时候先比键；分析结果用最快的一档压缩，几百页的书从 70 多 MB 压到十几 MB，多花不到一秒。"""
    blocks = plan.blocks
    skipped = [index for index, block in enumerate(blocks) if id(block) in plan.skip_ids]
    kinds = [plan.block_kind[id(block)] for block in blocks]
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            pickle.dump(key, out)
            body = pickle.dumps(
                (replace(plan, src_doc=None, skip_ids=set(), block_kind={}), skipped, kinds),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            out.write(zlib.compress(body, 1))
        tmp.replace(path)
    except Exception:  # noqa: BLE001 - 存不下只是下次要重新分析
        tmp.unlink(missing_ok=True)


def _load_plan(path: Path, key: str, src: Path, runner) -> _PdfPlan | None:
    try:
        with path.open("rb") as saved:
            if pickle.load(saved) != key:
                return None
            runner.report_stage("读取上次的版面分析结果")
            plan, skipped, kinds = pickle.loads(zlib.decompress(saved.read()))
    except Exception:  # noqa: BLE001 - 没有、读不出或格式旧了都重新分析
        return None
    plan.src_doc = pymupdf.open(src)
    plan.skip_ids = {id(plan.blocks[index]) for index in skipped}
    plan.block_kind = {id(block): kind for block, kind in zip(plan.blocks, kinds)}
    return plan


def saved_texts(path: Path) -> list[tuple[str, str]]:
    """上次分析结果里送翻的 (原文, 说明)，不管代码和模型变没变。清理任务时据此找出登记归属之前存下的译文。"""
    try:
        with path.open("rb") as saved:
            pickle.load(saved)
            plan, _skipped, _kinds = pickle.loads(zlib.decompress(saved.read()))
    except Exception:  # noqa: BLE001 - 没有、读不出或格式旧了就当没有
        return []
    return list(zip(plan.texts, plan.notes or [""] * len(plan.texts)))


async def translate_pdf(
    src: Path,
    out_dir: Path,
    runner,
    bilingual: bool,
    target_lang: str = "",
    layout_model: str = "",
    plan_path: Path | None = None,
    formula_model: FormulaModel | None = None,
) -> list[Path]:
    """layout_model 是版面模型文件的路径，空串表示不用；formula_model 是公式检测模型的设置，None 表示不用。
    都不用时只用规则。plan_path 是存分析结果的文件，见 _saved_plan。

    读版面、裁公式图、写回都用 MuPDF，一本书要算几分钟，放在线程里做，事件循环还能响应
    任务页的轮询。这几步轮到才做（见 mupdf_turn），等翻译接口时放开，别的任务可以接着
    做它的准备。任务被暂停时线程在下一页停下，停了才放开 MuPDF（见 background.run）。"""
    async with _job_turn(runner):
        plan = await background.run(_saved_plan, src, runner, layout_model, plan_path, formula_model)
    src_doc, blocks, geometries, toc = plan.src_doc, plan.blocks, plan.geometries, plan.toc
    flat_units, sent_texts, sent_formulas = plan.units, plan.texts, plan.formulas
    skip_ids, block_kind = plan.skip_ids, plan.block_kind
    raw = await runner.translate_all(sent_texts, kinds=plan.kinds, preview=True, skip=plan.skips, notes=plan.notes)
    await _retry_lost_placeholders(runner, sent_texts, sent_formulas, raw)

    async with _job_turn(runner):
        archive = await background.run(_crop_formulas, src_doc, flat_units, sent_formulas)

    per_block: dict[int, str] = {}
    block_formulas: dict[int, list] = {}
    fallbacks = 0
    kept_ids = set(skip_ids)
    labels = plan.labels or [""] * len(flat_units)
    aparts = plan.aparts or [False] * len(flat_units)
    # 按分界切回两块写的单元，写回时不再排满接续
    apart_hosts: set[int] = set()
    for unit, t, sent_text, formulas, label, apart in zip(flat_units, raw, sent_texts, sent_formulas, labels, aparts):
        if t == sent_text:
            # 模型原样返回（编号、代码、专名）。送翻的文字规整过（TeX 的 `u 合成 ù），
            # 和块里的原字不同，按译文写回会擦掉原字重排。留原字，和跳过的块一样
            kept_ids.update(id(b) for b in unit)
            for b in unit:
                per_block[id(b)] = b.text
                block_formulas[id(b)] = []
            continue
        t = _keep_formula_chains(_sound_colors(_drop_stray_braces(t, sent_text), sent_text), sent_text)
        restored = _restore_placeholders(t, formulas, lenient=True) if formulas else _STRAY_PLACEHOLDER_RE.sub("", t)
        if restored is not None:
            restored = separate_after_formula(restored)
        if restored is None:
            fallbacks += 1
            parts = [b.text for b in unit]  # 模型弄丢占位符：整单元回退原文，不产出坏文档
            formula_lists: list[list[dict]] = [[] for _ in unit]
            shown = _join_lines(parts)
        else:
            parts, formula_lists = _assign_restored_parts(restored, unit, formulas, apart)
            if len(parts) == 2 and parts[1] != SPILL_TAIL:
                apart_hosts.add(id(unit[0]))
            parts[0] = _with_label(label, parts[0])
            shown = strip_style_marks(_join_lines(parts) if id(unit[0]) in apart_hosts else parts[0])
        for b, part, flist in zip(unit, parts, formula_lists):
            if part == SPILL_TAIL:
                per_block[id(b)] = SPILL_TAIL
                block_formulas[id(b)] = []
                continue
            per_block[id(b)] = _list_number(b.text, _heading_number_for(block_kind[id(b)], b.text, part))
            block_formulas[id(b)] = flist
        # 预览里显示恢复后的译文（缓存仍按占位符版本存，重跑照样命中）
        if formulas and getattr(runner, "_done", None) and runner._done.get(sent_text) == t:
            runner._done[sent_text] = shown
    # 退回原文的单元算进失败段数，任务页才会提示，不会显示成全部翻完
    info = getattr(runner, "info", None)
    if fallbacks and isinstance(info, dict):
        info["failed"] = int(info.get("failed") or 0) + fallbacks
    translations = [per_block[id(b)] for b in blocks]
    for i in range(1, len(blocks)):
        if translations[i] == SPILL_TAIL or translations[i - 1] == SPILL_TAIL:
            continue
        translations[i] = _strip_boundary_echo(
            blocks[i - 1].text,
            blocks[i].text,
            translations[i - 1],
            translations[i],
        )
        per_block[id(blocks[i])] = translations[i]
    # 段界去重也可能改到跳过的块。写回只认和原文逐字相同的字符串，才会把原字留下。
    for b in blocks:
        if id(b) in kept_ids:
            per_block[id(b)] = b.text
            block_formulas[id(b)] = []
    translations = [per_block[id(b)] for b in blocks]
    _sync_unit_preview(runner, flat_units, sent_texts, per_block)
    continuations = {
        id(unit[0]): unit[1]
        for unit in flat_units
        if len(unit) == 2 and not all(id(b) in skip_ids for b in unit) and id(unit[0]) not in apart_hosts
    }

    def build():
        def render(pages, on_page):
            return _render_translated(
                src,
                blocks,
                translations,
                target_lang,
                block_formulas,
                archive,
                geometries=geometries,
                continuations=continuations,
                on_page=on_page,
                pages=pages,
            )

        # 一本书要写好几分钟，这时译文已经全部到齐，进度条停在满格，按页报写到哪里
        _drop_mupdf_cache()
        translated = _render_by_parts(
            src_doc,
            blocks,
            translations,
            continuations,
            render,
            lambda done, total: runner.report_stage("写回译文", done, total, "页"),
        )
        runner.report_stage("生成文件")
        _drop_mupdf_cache()
        if bilingual:
            dst = out_dir / f"{src.stem}.bilingual.pdf"
            # 先落盘再读回，并排页从读回的文档里取页面
            output = _render_side_by_side(src_doc, pymupdf.open("pdf", translated.tobytes()))
        else:
            dst = out_dir / f"{src.stem}.translated.pdf"
            output = translated
        if toc:
            output.set_toc(toc)  # 并排页和原页一一对应，书签页码不用映射
            if not bilingual and hasattr(runner, "preview"):
                translate_bookmarks(output, runner.preview(-1))
        _drop_mupdf_cache()
        dst.write_bytes(_compact(output))
        return dst

    async with _job_turn(runner):
        return [await background.run(build)]
