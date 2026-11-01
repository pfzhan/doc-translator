"""Write the translation in several subprocesses, then combine the results into one document.

Writing typesets page by page and takes several minutes for a book of a few hundred pages, and pages are almost
independent of each other. The only dependency is a paragraph carried over to the next page: the part a page
cannot fit is left to the tail block on the next page (see carried in pdf_write._render_translated), so a host
and its tail must be in the same part.

Subprocesses are started with fork: the blocks, translations and formula images that writing needs are in
memory and need no serialization. Each subprocess opens its own copy of the original book, writes only its own
pages and saves them to a temporary file. When combining, the result of the first part is the base document,
which carries the bookmarks and links; pages written by the other parts are pasted in as whole pages, replacing
the unwritten pages of the base.
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

# Minimum pages per part. With fewer pages, starting subprocesses and combining pages costs more than it saves
_MIN_PART_PAGES = 12
# Maximum number of subprocesses. For a 646-page book each writing subprocess uses an extra 0.6-0.9 GB; on 6
# cores, 5 subprocesses write 10-20 seconds faster than 4
_MAX_WORKERS = 6
# A subprocess that has not started a new page for this long is considered stuck (a lock held by another thread
# at fork time is never released in the child)
_STALL_SECONDS = 300

Render = Callable[[Collection[int] | None, Callable[[int, int], None] | None], pymupdf.Document]


def worker_count() -> int:
    """Leave one core for the service itself."""
    return max(1, min(_MAX_WORKERS, (os.cpu_count() or 2) - 1))


def plan_parts(
    weights: dict[int, int],
    links: Sequence[tuple[int, int]],
    parts: int,
) -> list[list[int]]:
    """Split the pages to write (in order; each value is the number of blocks to write on that page) into at
    most parts parts with similar block counts.

    links are (host page, tail page) pairs. No pair may straddle a cut point q, i.e. have its host before q and
    its tail at or after q.
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
        if filled >= target and len(out) < parts and pno not in blocked and len(out[-1]) >= _MIN_PART_PAGES:
            out.append([])
            filled = 0
        out[-1].append(pno)
        filled += weights[pno]
    return out


def render_in_parts(
    render: Render,
    parts: list[list[int]],
    on_page: Callable[[int, int], None] | None = None,
) -> pymupdf.Document | None:
    """Write each part in a subprocess and combine them into one document. Return None when fork is
    unavailable or a subprocess fails or gets stuck; the caller then writes the whole book itself.
    A rotated page pasted back as a whole page is rotated a second time, so callers must not split files that
    have rotated pages."""
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
            ctx.Process(target=_render_part, args=(render, part, path, started)) for part, path in zip(parts, paths)
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
    """Exit right after writing. The process is forked from a worker thread of the service, and the thread
    cleanup in multiprocessing shutdown would fail and make the exit code 1."""

    def count(_done: int, _total: int) -> None:
        with started.get_lock():
            started.value += 1

    try:
        doc = render(set(part), count)
        path.write_bytes(doc.tobytes())
    except BaseException:  # noqa: BLE001 - the child can only print it; the parent rewrites the book by exit code
        traceback.print_exc()
        os._exit(1)
    os._exit(0)


def _replace_page(base: pymupdf.Document, pno: int, written: pymupdf.Document) -> None:
    """Replace page pno of the base document with the page written by another subprocess. Annotations (links)
    stay on the base, as in the original book."""
    page = base[pno]
    blank = base.get_new_xref()
    base.update_object(blank, "<<>>")
    base.update_stream(blank, b" ")
    page.set_contents(blank)
    base.xref_set_key(page.xref, "Resources", "<<>>")
    page.show_pdf_page(page.rect, written, pno)
