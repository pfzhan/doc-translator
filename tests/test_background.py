import asyncio
import threading
import time

import pytest

from app import background
from app.formats import pdf
from app.runner import Cache, Runner


def test_run_returns_the_result_and_passes_errors_through():
    assert asyncio.run(background.run(lambda a, b: a + b, 1, 2)) == 3

    def broken():
        raise ValueError("bad page")

    with pytest.raises(ValueError):
        asyncio.run(background.run(broken))
    # 不在 run 起的线程里时 check_stop 什么也不做
    background.check_stop()


def test_a_cancelled_run_stops_the_thread_and_waits_for_it():
    """任务被取消时线程在下一次 check_stop 停下；取消等线程停了才抛出来"""
    steps = []
    finished = threading.Event()

    def work():
        try:
            for step in range(200):
                background.check_stop()
                steps.append(step)
                time.sleep(0.01)
        finally:
            time.sleep(0.05)
            finished.set()

    async def main():
        task = asyncio.ensure_future(background.run(work))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(main())
    assert len(steps) < 50


def test_pausing_a_pdf_during_analysis_stops_it_before_freeing_mupdf(monkeypatch, tmp_path):
    """读版面时暂停：分析在下一页停下（报进度时），MuPDF 等分析停了才让给别的任务"""
    pages = []
    finished = threading.Event()

    def analysis(src, runner, *args):
        try:
            for page in range(300):
                runner.report_stage("分析版面", page, 300, "页")
                pages.append(page)
                time.sleep(0.01)
        finally:
            time.sleep(0.05)
            finished.set()

    monkeypatch.setattr(pdf, "_saved_plan", analysis)
    runner = Runner(object(), cache=Cache(tmp_path / "c.sqlite3"))

    async def main():
        task = asyncio.ensure_future(pdf.translate_pdf(tmp_path / "a.pdf", tmp_path, runner, False))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert finished.is_set() and not pdf.mupdf_turn().locked()

    asyncio.run(main())
    assert 0 < len(pages) < 50
