"""分几个子进程写回译文，再拼成一份。

写回逐页排版，一本几百页的书要好几分钟，页和页之间几乎互不相干。唯一的牵连是跨页
段落：上一页写不下的半段留给下一页的续块（见 _render_translated 的 carried），
所以宿主和续块要分在同一段里。

子进程用 fork 起：写回要用的块、译文、公式图都在内存里，不用序列化。每个子进程从原书
打开一份，只写自己那段页，存成临时文件。拼的时候以第一段的结果为底稿，书签、链接都在
它上面；别的段写好的页整页贴进来，盖掉底稿里没写的那页。
"""
from __future__ import annotations

import multiprocessing
import os
import tempfile
import time
import traceback
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import pymupdf

# 每段至少这么多页。页少时起子进程、拼页的开销比省下的多
_MIN_PART_PAGES = 12
_MAX_WORKERS = 4
# 这么久没开始写新的一页就当子进程卡住了（fork 时别的线程拿着的锁在子进程里永远不放）
_STALL_SECONDS = 300

Render = Callable[[Collection[int] | None, Callable[[int, int], None] | None], pymupdf.Document]


def worker_count() -> int:
    """留一个核给服务本身。"""
    return max(1, min(_MAX_WORKERS, (os.cpu_count() or 2) - 1))


def plan_parts(
    weights: dict[int, int], links: Sequence[tuple[int, int]], parts: int,
) -> list[list[int]]:
    """把要写的页（按顺序，值是这页要写的块数）切成至多 parts 段，各段的块数相近。

    links 是 (宿主页, 续块页)。切点 q 前后不能有一对跨过去：宿主在 q 之前、续块在 q 或之后。
    """
    order = list(weights)
    if parts <= 1 or len(order) < _MIN_PART_PAGES * 2:
        return [order]
    parts = min(parts, len(order) // _MIN_PART_PAGES)
    blocked = {q for host, tail in links for q in order if host < q <= tail}
    target = sum(weights.values()) / parts
    out: list[list[int]] = [[]]
    filled = 0
    for pno in order:
        if (
            filled >= target and len(out) < parts and pno not in blocked
            and len(out[-1]) >= _MIN_PART_PAGES
        ):
            out.append([])
            filled = 0
        out[-1].append(pno)
        filled += weights[pno]
    return out


def render_in_parts(
    render: Render, parts: list[list[int]], on_page: Callable[[int, int], None] | None = None,
) -> pymupdf.Document | None:
    """各段在子进程里写，拼回一份。不能 fork、有子进程失败或卡住时返回 None，调用方自己整本写。
    旋转页整页贴回去会再转一次，有旋转页的文件调用方不要分段。"""
    if len(parts) < 2:
        return None
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        return None
    total = sum(len(part) for part in parts)
    started = ctx.Value("i", 0)
    with tempfile.TemporaryDirectory(prefix="pdf-parts-") as tmp:
        paths = [Path(tmp) / f"{index}.pdf" for index in range(len(parts))]
        procs = [
            ctx.Process(target=_render_part, args=(render, part, path, started))
            for part, path in zip(parts, paths)
        ]
        for proc in procs:
            proc.start()
        seen, moved = -1, time.monotonic()
        while any(proc.is_alive() for proc in procs):
            if started.value != seen:
                seen, moved = started.value, time.monotonic()
            elif time.monotonic() - moved > _STALL_SECONDS:
                for proc in procs:
                    proc.kill()
            if on_page is not None:
                on_page(min(seen, total), total)
            time.sleep(0.5)
        for proc in procs:
            proc.join()
        if any(proc.exitcode != 0 or not path.exists() for proc, path in zip(procs, paths)):
            return None
        base = pymupdf.open("pdf", paths[0].read_bytes())
        for part, path in zip(parts[1:], paths[1:]):
            written = pymupdf.open("pdf", path.read_bytes())
            for pno in part:
                _replace_page(base, pno, written)
    return base


def _render_part(render: Render, part: list[int], path: Path, started) -> None:
    """写完直接退出。fork 自服务的工作线程，multiprocessing 收尾时清理线程会出错，退出码成了 1。"""
    def count(_done: int, _total: int) -> None:
        with started.get_lock():
            started.value += 1

    try:
        doc = render(set(part), count)
        path.write_bytes(doc.tobytes())
    except BaseException:  # noqa: BLE001 - 子进程里的异常只能打印，父进程按退出码整本重写
        traceback.print_exc()
        os._exit(1)
    os._exit(0)


def _replace_page(base: pymupdf.Document, pno: int, written: pymupdf.Document) -> None:
    """底稿的第 pno 页换成别的子进程写好的那页。注释（链接）留在底稿上，和原书一样。"""
    page = base[pno]
    blank = base.get_new_xref()
    base.update_object(blank, "<<>>")
    base.update_stream(blank, b" ")
    page.set_contents(blank)
    base.xref_set_key(page.xref, "Resources", "<<>>")
    page.show_pdf_page(page.rect, written, pno)
