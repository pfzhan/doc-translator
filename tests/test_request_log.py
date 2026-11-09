"""The request log: every attempt an engine sends upstream is recorded, and a job keeps the log and a summary."""

import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from app import state
from app.history import Job, JobStore
from app.request_log import RequestLog, RequestReason, RequestRecord, RequestStats, read_records, summarize
from app.translators import Call, GoogleTranslator, MockTranslator, OpenAITranslator, _Attempt
from tests.helpers import run
from tests.test_history import submit, wait_done


async def _no_sleep(_seconds: float) -> None:
    return None


def _openai(handler: httpx.MockTransport, log: RequestLog) -> OpenAITranslator:
    tr = OpenAITranslator(
        "zh-CN", "en", api_key="k", base_url="https://api.example.com/v1", model="m", transport=handler
    )
    tr.request_log = log
    return tr


def _record(
    duration: float = 1.0,
    gate_wait: float = 0.0,
    attempt: int = 1,
    status: int | None = 200,
    error: str = "",
    reason: RequestReason = "normal",
) -> RequestRecord:
    return RequestRecord(
        started=0.0,
        gate_wait=gate_wait,
        duration=duration,
        attempt=attempt,
        status=status,
        error=error,
        reason=reason,
        items=1,
        chars=10,
    )


def test_every_attempt_is_recorded_with_its_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed attempt and its retry are two records; the successful one carries the reported tokens and the model
    the API says answered."""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502, json={"error": {"message": "bad gateway"}})
        return httpx.Response(
            200,
            json={
                "model": "m-2026-01-01",
                "choices": [{"message": {"content": "[[p0]]\n甲\n[[p1]]\n乙"}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            },
        )

    log = RequestLog(tmp_path / "requests.jsonl")
    tr = _openai(httpx.MockTransport(handler), log)
    assert run(tr.translate_batch(["First one.", "Second."])) == ["甲", "乙"]
    run(tr.aclose())
    log.flush()
    failed, ok = read_records(tmp_path / "requests.jsonl")
    assert (failed.attempt, failed.status, failed.error, failed.input_tokens) == (1, 502, "http", None)
    assert (ok.attempt, ok.status, ok.error) == (2, 200, "")
    assert (ok.input_tokens, ok.output_tokens, ok.model) == (120, 30, "m-2026-01-01")
    assert (ok.reason, ok.items, ok.chars) == ("normal", 2, len("First one.") + len("Second."))
    assert ok.duration >= 0 and ok.gate_wait == 0  # The gate was waited for once, before the first attempt


def test_records_say_why_a_request_was_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch the service keeps rejecting is split, and a paragraph the reply leaves out is translated on its own;
    the records tell those requests from the batch itself."""
    monkeypatch.setattr("app.translators.asyncio.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][-1]["content"]
        sent = sum(text in user for text in ("One.", "Two.", "Three."))
        if sent == 3:  # The batch of three is rejected
            return httpx.Response(400, json={"error": {"message": "upstream rejected"}})
        if sent == 2:  # The half of two: the reply leaves the second paragraph out
            return httpx.Response(200, json={"choices": [{"message": {"content": "[[p0]]\n译"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "译"}}]})

    log = RequestLog(tmp_path / "requests.jsonl")
    tr = _openai(httpx.MockTransport(handler), log)
    assert run(tr.translate_batch(["One.", "Two.", "Three."])) == ["译", "译", "译"]
    run(tr.aclose())
    records = log.summary()
    assert records is not None
    # The batch twice (one retry of the 400), two halves, one paragraph left out of the half of two
    assert records.reasons == {"normal": 2, "split": 2, "retranslate": 1}
    assert (records.count, records.retries, records.failed) == (5, 1, 2)


def test_google_attempts_are_recorded(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[["你好", "en"]])

    tr = GoogleTranslator("zh-CN", transport=httpx.MockTransport(handler))
    tr.request_log = RequestLog(tmp_path / "requests.jsonl")
    assert run(tr.translate_batch(["hello"])) == ["你好"]
    run(tr.aclose())
    tr.request_log.flush()
    [record] = read_records(tmp_path / "requests.jsonl")
    assert (record.status, record.items, record.chars, record.input_tokens) == (200, 1, 5, None)


def test_summary_of_the_records() -> None:
    records = [
        _record(duration=float(n), gate_wait=1.0 if n == 1 else 0.0, attempt=2 if n == 2 else 1) for n in range(1, 11)
    ]
    records.append(_record(duration=0.0, status=429, error="http", reason="split"))
    stats = summarize(records)
    assert stats == RequestStats(
        count=11,
        retries=1,
        failed=1,
        mean_latency=5.0,
        p90_latency=9.0,
        mean_gate_wait=round(1 / 11, 3),
        reasons={"normal": 10, "split": 1},
    )
    assert summarize([]) is None


def test_log_is_buffered_and_not_written_for_a_deleted_record(tmp_path: Path) -> None:
    path = tmp_path / "job" / "requests.jsonl"
    path.parent.mkdir()
    log = RequestLog(path)
    log.FLUSH_EVERY = 3
    log.add(_record())
    log.add(_record())
    assert not path.exists()  # Still buffered
    log.add(_record())
    assert len(read_records(path)) == 3
    log.add(_record(status=500, error="http"))
    with path.open("a", encoding="utf-8") as f:
        f.write("not json\n")  # A damaged line is skipped
    log.flush()
    assert [r.status for r in read_records(path)] == [200, 200, 200, 500]
    # The record was deleted while the job ran: its directory is not made again
    path.unlink()
    path.parent.rmdir()
    log.add(_record())
    log.flush()
    assert not path.parent.exists()


def test_old_and_new_records_load(tmp_path: Path) -> None:
    """job.json from before the request log has no summary; a saved summary comes back as RequestStats."""
    store = JobStore(tmp_path / "jobs")
    old = Job(id="old", filename="a.md", status="done")
    new = Job(id="new", filename="b.md", status="done", requests=RequestStats(count=3, reasons={"normal": 3}))
    for job in (old, new):
        store.dir(job.id).mkdir(parents=True)
        store.save(job)
    legacy = json.loads((store.dir("old") / "job.json").read_text(encoding="utf-8"))
    del legacy["requests"]
    (store.dir("old") / "job.json").write_text(json.dumps(legacy), encoding="utf-8")
    loaded = JobStore(tmp_path / "jobs")
    assert loaded.jobs["old"].requests is None
    assert loaded.jobs["new"].requests == RequestStats(count=3, reasons={"normal": 3})


def test_a_job_keeps_its_requests_until_its_cache_or_itself_is_gone(
    app_env: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run writes requests.jsonl in the record's directory and a summary in the record. Clearing the record's cache
    removes the log with the token counts; deleting the record removes it with the directory."""
    original = MockTranslator.translate_batch

    async def logged(self: MockTranslator, texts: list[str]) -> list[str]:
        call = Call.of(texts, "normal")
        self._log_request(call, _Attempt(number=1, started=0.0, gate_wait=0.5, duration=2.0), 200)
        return await original(self, texts)

    monkeypatch.setattr(MockTranslator, "translate_batch", logged)
    main = app_env
    with TestClient(main.app) as client:
        job = wait_done(client, submit(client)["id"])
        log = state.jobs.requests_path(job["id"])
        count = len(read_records(log))
        assert count >= 1
        assert job["requests"]["count"] == count
        assert job["requests"]["mean_latency"] == 2.0 and job["requests"]["mean_gate_wait"] == 0.5
        assert client.delete(f"/api/jobs/{job['id']}/cache").status_code == 200
        assert not log.exists()
        assert client.get(f"/api/jobs/{job['id']}").json()["requests"] is None

        again = wait_done(client, submit(client)["id"])
        log = state.jobs.requests_path(again["id"])
        assert log.exists()
        assert client.delete(f"/api/jobs/{again['id']}").status_code == 200
        assert not log.exists()
