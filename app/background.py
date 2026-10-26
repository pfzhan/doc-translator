"""读版面、写回这类要算几分钟的活放在线程里做（见 run）。

asyncio.to_thread 起的线程取消不了：任务被暂停时协程停了，线程照样算到最后，CPU 一直占着；
外面的 async with mupdf_turn() 却已经放开，另一个任务会和这个线程同时用 MuPDF。
run 在任务被取消时通知线程停下，等它真的停了才往外抛。线程里逐页、逐段的循环调用 check_stop，
报进度的 Runner.report_stage 也会调用。
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
    """线程里的活因为任务被取消（暂停、删除、关服务）停下了。"""


def check_stop() -> None:
    """在线程里逐页、逐段调用：所在的任务被取消时抛 Stopped。不在 run 起的线程里时什么也不做。"""
    event = _STOP.get()
    if event is not None and event.is_set():
        raise Stopped


async def run(fn: Callable[..., Any], *args: Any) -> Any:
    """在线程里调用 fn(*args)。任务被取消时让线程在下一次 check_stop 停下，等它停了再抛出取消。"""
    event = threading.Event()
    token = _STOP.set(event)
    try:
        # 任务创建时复制当前的上下文，线程里的 check_stop 看得到这个 event
        work = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    finally:
        _STOP.reset(token)
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        event.set()
        # 线程停下前可能还在用 MuPDF，等它停了，外面的 mupdf_turn 才放开
        with contextlib.suppress(Exception):
            await work
        raise
