"""Work that takes minutes, such as reading layout and writing output, runs in a thread (see run).

A thread started by asyncio.to_thread cannot be cancelled: when a job is paused, the coroutine stops but the thread
keeps computing to the end and occupies the CPU, while the enclosing async with mupdf_turn() has already been
released, so another job would use MuPDF at the same time as this thread. run tells the thread to stop when the
job is cancelled and raises only after it has actually stopped. Per-page and per-paragraph loops in the thread call
check_stop, and so does Runner.report_stage when reporting progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
from collections.abc import Callable
from typing import Any

_STOP: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("background_stop", default=None)


class Stopped(Exception):
    """Work in the thread stopped because its job was cancelled (paused, deleted, server shut down)."""


def check_stop() -> None:
    """Called per page and per paragraph in the thread: raises Stopped when its job has been cancelled. Does nothing
    outside a thread started by run."""
    event = _STOP.get()
    if event is not None and event.is_set():
        raise Stopped


async def run(fn: Callable[..., Any], *args: Any) -> Any:
    """Call fn(*args) in a thread. When the job is cancelled, the thread stops at its next check_stop, and the
    cancellation is raised after it has stopped."""
    event = threading.Event()
    token = _STOP.set(event)
    try:
        # Creating the task copies the current context, so check_stop in the thread sees this event
        work = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    finally:
        _STOP.reset(token)
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        event.set()
        # The thread may still be using MuPDF until it stops; only then is the enclosing mupdf_turn released
        with contextlib.suppress(Exception):
            await work
        raise
