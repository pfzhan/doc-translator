"""PDF translation that keeps the original layout.

1. PyMuPDF extracts the text blocks of each page (position, font size, color, bold).
2. Whole-block formulas, bare numbers and rotated text are skipped. An inline formula is one run inside
   a paragraph and is pasted back unchanged after translation.
3. Translated-only version: the original text is erased (images and vector graphics stay) and the
   translation is written in its place. When it does not fit, the box first grows to the right within its
   column (without covering figures or other text), then the line height shrinks, then the font size.
   Ruled tables are translated cell by cell and the text stays inside each cell. Repeated running headers,
   footers and page numbers keep the original text.
4. Bilingual version: the original page and the translated page sit side by side on one page.

The stages live in separate modules: pdf_extract reads pages into blocks (with pdf_rows splitting lines and
pdf_paragraphs joining them), pdf_merge joins paragraphs across pages and columns, pdf_skip picks the blocks to
keep, pdf_units builds the translation units, pdf_restore puts the translations back together, pdf_crop cuts out
formula images, pdf_write and pdf_place write the translations, and pdf_output assembles the output files. This
module runs them in order.
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
from collections.abc import AsyncIterator, Callable, Collection
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

from .. import background
from ..runner import RunInfo
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
from .pdf_types import Formula
from .pdf_units import _PdfPlan, _plan_document
from .pdf_vision import FormulaModel, ModelBoxes
from .pdf_write import _render_translated

if TYPE_CHECKING:
    from ..runner import Runner

_CAMEL_RE = re.compile(r"[A-Za-z]*[a-z][A-Z][A-Za-z]*")


# PyMuPDF cannot call MuPDF from several threads at once. Its find_tables temporarily turns on the
# process-wide small_glyph_heights option, and character boxes extracted by another job at that moment come
# out shorter. When the jobs' on and off calls interleave, the option can stay on, and every later job in
# the process segments text with short boxes until the service restarts.
_MUPDF_TURNS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


def mupdf_turn() -> asyncio.Lock:
    """Acquire this with async with before using MuPDF, so that only one job uses it at a time."""
    loop = asyncio.get_running_loop()
    turn = _MUPDF_TURNS.get(loop)
    if turn is None:
        turn = _MUPDF_TURNS[loop] = asyncio.Lock()
    return turn


@contextlib.asynccontextmanager
async def _job_turn(runner: "Runner") -> AsyncIterator[None]:
    """Wait for this job's turn with MuPDF. While another job reads or writes a large book the wait can take
    minutes, and the job page shows that it is waiting."""
    turn = mupdf_turn()
    if turn.locked():
        runner.report_stage("等待其他任务")
    async with turn:
        yield


def _plan_pdf(
    src: Path,
    runner: "Runner",
    layout_model: str,
    formula_model: FormulaModel | None = None,
    boxes: ModelBoxes | None = None,
) -> _PdfPlan:
    """Everything before translation: read the layout, assign roles, join paragraphs across pages and put
    placeholders into each unit. Whether a hyphen at a line break is kept depends on whether the joined word
    appears elsewhere in the document (see _join_lines); the words of this document are used throughout the
    analysis. boxes is where the boxes of the two models are stored; when given, they are read back or
    saved (see ModelBoxes)."""
    src_doc = pymupdf.open(src)
    if src_doc.needs_pass:
        raise ValueError("不支持有密码保护的 PDF")
    words = frozenset(word for page in src_doc.pages() for word in _CAMEL_RE.findall(page.get_text("text")))
    token = _CAMEL_WORDS.set(words)
    try:
        return _plan_document(src, src_doc, runner, layout_model, formula_model, boxes)
    finally:
        _CAMEL_WORDS.reset(token)


def _saved_plan(
    src: Path,
    runner: "Runner",
    layout_model: str,
    path: Path | None,
    formula_model: FormulaModel | None = None,
) -> _PdfPlan:
    """Read back the previous analysis result when a job resumes after a pause or an interruption, or when
    the other version is generated, as long as the source file, both models and the code are unchanged.
    Reading the layout of a book of a few hundred pages takes over a minute; reading the result back takes a
    second or two. Nothing is stored when path is None. When the code changed and the analysis has to run
    again, the model boxes come from boxes.json next to it (see ModelBoxes) and only the rules run again."""
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
    # The two models are recorded separately: the result differs between only the formula model and only the
    # layout model being enabled. A different input size for the formula model also requires a new analysis.
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
    """Version of the program code and of PyMuPDF. Changing the code invalidates earlier analysis results."""
    digest = hashlib.sha1(str(pymupdf.VersionBind).encode())
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _save_plan(path: Path, key: str, plan: _PdfPlan) -> None:
    """The two tables keyed by block id are stored as indexes, since blocks read back are new objects. The
    file is written to a temporary name and renamed, so an exit halfway leaves no broken file. The key is
    stored uncompressed and compared first on reading. The result uses the fastest compression level, which
    takes a book of a few hundred pages from over 70 MB to a little over 10 MB in less than a second."""
    blocks = plan.blocks
    skipped = [index for index, block in enumerate(blocks) if id(block) in plan.skip_ids]
    kinds = [plan.block_kind[id(block)] for block in blocks]
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            pickle.dump(key, out)
            # A Document cannot be pickled: the stored copy has none and _load_plan opens the source again, so
            # src_doc is None only between here and there
            stored = replace(plan, src_doc=None, skip_ids=set(), block_kind={})  # type: ignore[arg-type]
            body = pickle.dumps(
                (stored, skipped, kinds),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            out.write(zlib.compress(body, 1))
        tmp.replace(path)
    except Exception:  # noqa: BLE001 - if it cannot be saved, the next run analyses again
        tmp.unlink(missing_ok=True)


def _load_plan(path: Path, key: str, src: Path, runner: "Runner") -> _PdfPlan | None:
    try:
        with path.open("rb") as saved:
            if pickle.load(saved) != key:
                return None
            runner.report_stage("读取上次的版面分析结果")
            plan: _PdfPlan
            skipped: list[int]
            kinds: list[str]
            plan, skipped, kinds = pickle.loads(zlib.decompress(saved.read()))
    except Exception:  # noqa: BLE001 - missing, unreadable or an old format: analyse again
        return None
    plan.src_doc = pymupdf.open(src)
    plan.skip_ids = {id(plan.blocks[index]) for index in skipped}
    plan.block_kind = {id(block): kind for block, kind in zip(plan.blocks, kinds)}
    return plan


def saved_texts(path: Path) -> list[tuple[str, str]]:
    """The (source text, note) pairs sent for translation in the previous analysis result, whether or not the
    code and models changed. Cleaning up a job uses them to find translations cached before cache entries
    recorded their owner."""
    try:
        with path.open("rb") as saved:
            pickle.load(saved)
            plan, _skipped, _kinds = pickle.loads(zlib.decompress(saved.read()))
    except Exception:  # noqa: BLE001 - missing, unreadable or an old format counts as none
        return []
    return list(zip(plan.texts, plan.notes or [""] * len(plan.texts)))


async def translate_pdf(
    src: Path,
    out_dir: Path,
    runner: "Runner",
    bilingual: bool,
    target_lang: str = "",
    layout_model: str = "",
    plan_path: Path | None = None,
    formula_model: FormulaModel | None = None,
) -> list[Path]:
    """layout_model is the path of the layout model file, or an empty string for none; formula_model holds
    the formula detection model settings, or None for none. Without either, only the rules are used.
    plan_path is the file that stores the analysis result, see _saved_plan.

    Reading the layout, cropping formula images and writing all use MuPDF and take minutes on a book. They
    run in a thread so that the event loop keeps answering the job page's polling. Each of these steps waits
    for its turn (see mupdf_turn) and releases it while waiting for the translation service, so another job
    can do its own preparation. When the job is paused, the thread stops at the next page and releases MuPDF
    only after it has stopped (see background.run)."""
    async with _job_turn(runner):
        plan: _PdfPlan = await background.run(_saved_plan, src, runner, layout_model, plan_path, formula_model)
    src_doc, blocks, geometries, toc = plan.src_doc, plan.blocks, plan.geometries, plan.toc
    flat_units, sent_texts, sent_formulas = plan.units, plan.texts, plan.formulas
    skip_ids, block_kind = plan.skip_ids, plan.block_kind
    raw = await runner.translate_all(sent_texts, kinds=plan.kinds, preview=True, skip=plan.skips, notes=plan.notes)
    await _retry_lost_placeholders(runner, sent_texts, sent_formulas, raw)

    async with _job_turn(runner):
        archive = await background.run(_crop_formulas, src_doc, flat_units, sent_formulas)

    per_block: dict[int, str] = {}
    block_formulas: dict[int, list[Formula]] = {}
    fallbacks = 0
    kept_ids = set(skip_ids)
    labels = plan.labels or [""] * len(flat_units)
    aparts = plan.aparts or [False] * len(flat_units)
    # Units split back into two blocks at their boundary; the tail is not filled to the full width when written
    apart_hosts: set[int] = set()
    for unit, t, sent_text, formulas, label, apart in zip(flat_units, raw, sent_texts, sent_formulas, labels, aparts):
        if t == sent_text:
            # The model returned the text unchanged (numbers, code, proper names). The text sent was normalized
            # (TeX's `u composed into ù) and differs from the characters in the block, so writing it back would
            # erase and reset the original characters. Keep the original, as for a skipped block
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
            parts = [b.text for b in unit]  # the model lost a placeholder: the whole unit falls back to the original
            formula_lists: list[list[Formula]] = [[] for _ in unit]
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
        # The preview shows the restored translation (the cache keeps the placeholder version, so reruns still hit)
        if formulas and getattr(runner, "_done", None) and runner._done.get(sent_text) == t:
            runner._done[sent_text] = shown
    # Units that fell back to the original count as failed segments, so the job page reports them instead of
    # showing everything as translated
    info = getattr(runner, "info", None)
    if fallbacks and isinstance(info, RunInfo):
        info.failed = (info.failed or 0) + fallbacks
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
    # Removing text repeated across a unit boundary may also change skipped blocks. Writing keeps the original
    # characters only for strings identical to the source text.
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

    def build() -> Path:
        def render(pages: Collection[int] | None, on_page: Callable[[int, int], None] | None) -> pymupdf.Document:
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

        # Writing a book takes several minutes after all translations have arrived and the progress bar is full,
        # so progress is reported by page
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
            # Serialize and reopen first; the side-by-side pages are taken from the reopened document
            output = _render_side_by_side(src_doc, pymupdf.open("pdf", translated.tobytes()))
        else:
            dst = out_dir / f"{src.stem}.translated.pdf"
            output = translated
        if toc:
            output.set_toc(toc)  # side-by-side pages map one to one to source pages, so bookmark pages need no mapping
            if not bilingual and hasattr(runner, "preview"):
                translate_bookmarks(output, runner.preview(-1))
        _drop_mupdf_cache()
        dst.write_bytes(_compact(output))
        return dst

    async with _job_turn(runner):
        return [await background.run(build)]
