import asyncio
import threading
import time
from pathlib import Path
from typing import Never

import pytest

from app import background
from app.formats import pdf
from app.runner import Cache, Runner
from app.translators import MockTranslator


def test_run_returns_the_result_and_passes_errors_through() -> None:
    assert asyncio.run(background.run(lambda a, b: a + b, 1, 2)) == 3

    def broken() -> Never:
        raise ValueError("bad page")

    with pytest.raises(ValueError):
        asyncio.run(background.run(broken))
    # Outside a thread started by run, check_stop does nothing
    background.check_stop()


def test_a_cancelled_run_stops_the_thread_and_waits_for_it() -> None:
    """When the job is cancelled the thread stops at its next check_stop, and the cancellation is raised only after
    the thread has stopped"""
    steps = []
    finished = threading.Event()

    def work() -> None:
        try:
            for step in range(200):
                background.check_stop()
                steps.append(step)
                time.sleep(0.01)
        finally:
            time.sleep(0.05)
            finished.set()

    async def main() -> None:
        task = asyncio.ensure_future(background.run(work))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(main())
    assert len(steps) < 50


def test_pausing_a_pdf_during_analysis_stops_it_before_freeing_mupdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pause while reading layout: analysis stops at the next page (when reporting progress), and MuPDF is handed to
    another job only after the analysis has stopped"""
    pages: list[int] = []
    finished = threading.Event()

    def analysis(src: Path, runner: Runner, *args: object) -> None:
        try:
            for page in range(300):
                runner.report_stage("分析版面", page, 300, "页")
                pages.append(page)
                time.sleep(0.01)
        finally:
            time.sleep(0.05)
            finished.set()

    monkeypatch.setattr(pdf, "_saved_plan", analysis)
    runner = Runner(MockTranslator("zh-CN"), cache=Cache(tmp_path / "c.sqlite3"))

    async def main() -> None:
        task = asyncio.ensure_future(pdf.translate_pdf(tmp_path / "a.pdf", tmp_path, runner, False))
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert finished.is_set() and not pdf.mupdf_turn().locked()

    asyncio.run(main())
    assert 0 < len(pages) < 50
