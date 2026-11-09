"""A record of every request a translation engine sends upstream, kept per job to explain how a run went.

Each attempt, retries included, becomes one RequestRecord: how long it waited for the service's concurrency gate,
how long the HTTP call took, what came back (status, tokens, the model that answered) and why it was sent. A job
appends them to requests.jsonl in its directory (RequestLog) and keeps a small RequestStats summary in job.json, so a
run that was slower than another on the same service can be compared request by request.
"""

import json
import logging
import math
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# Why a request was sent: the batch itself ("normal"), the same request again after an empty or unusable reply
# ("repeat"), one paragraph translated on its own because the batch reply left it out or shifted it ("retranslate"),
# half of a batch the service rejected or cut off ("split"), or fixing the document glossary ("terms")
RequestReason = Literal["normal", "repeat", "retranslate", "split", "terms"]

# File name of the log in a job's directory
LOG_NAME = "requests.jsonl"


@dataclass
class RequestRecord:
    """One attempt of one request. Times are in seconds; started is the wall clock when the attempt began to wait
    for the gate."""

    started: float
    gate_wait: float
    duration: float
    attempt: int  # 1 for the first attempt of a request, 2 for its first retry, ...
    status: int | None  # HTTP status, None when no response came back
    # "" when a usable answer came back; otherwise "network", "timeout", "http" (an error status) or "format" (an
    # answer that was not JSON)
    error: str
    reason: RequestReason
    items: int  # paragraphs in the request
    chars: int  # characters of their source text
    model: str = ""  # the model the API says answered, which a relay may change
    input_tokens: int | None = None  # None when the API did not report usage
    output_tokens: int | None = None

    def to_json(self) -> str:
        data = asdict(self)
        for key in ("started", "gate_wait", "duration"):
            data[key] = round(data[key], 3)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> "RequestRecord | None":
        """Read one line of the log; None for a line that is damaged or from an incompatible version."""
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                return None
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        except (ValueError, TypeError):
            return None


@dataclass
class RequestStats:
    """Summary of a job's requests, kept in job.json and shown with the job."""

    count: int = 0  # attempts sent
    retries: int = 0  # attempts after the first of a request
    failed: int = 0  # attempts without a usable answer
    mean_latency: float = 0.0  # mean seconds of the HTTP calls
    p90_latency: float = 0.0
    mean_gate_wait: float = 0.0  # mean seconds waiting for the concurrency gate
    # Attempts by RequestReason; many "retranslate" or "split" requests explain a run slower than usual
    reasons: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> "RequestStats | None":
        """The summary saved in job.json; None when there is none or it cannot be read."""
        if not isinstance(data, dict):
            return None
        try:
            known = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})
        except TypeError:
            return None


def summarize(records: list[RequestRecord]) -> RequestStats | None:
    if not records:
        return None
    latencies = sorted(r.duration for r in records)
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    return RequestStats(
        count=len(records),
        retries=sum(1 for r in records if r.attempt > 1),
        failed=sum(1 for r in records if r.error),
        mean_latency=round(sum(latencies) / len(latencies), 3),
        p90_latency=round(latencies[max(0, math.ceil(len(latencies) * 0.9) - 1)], 3),
        mean_gate_wait=round(sum(r.gate_wait for r in records) / len(records), 3),
        reasons=reasons,
    )


def read_records(path: Path) -> list[RequestRecord]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [record for line in lines if line.strip() and (record := RequestRecord.from_json(line)) is not None]


class RequestLog:
    """Appends records to a job's requests.jsonl. Records are buffered and written FLUSH_EVERY at a time, so a run
    pays for one small write per many requests; the job writes the rest when the run ends (flush)."""

    FLUSH_EVERY = 50

    def __init__(self, path: Path) -> None:
        self.path = path
        self._buffer: list[RequestRecord] = []
        # The final flush runs in a worker thread while engines may still add records from the event loop
        self._lock = threading.Lock()

    def add(self, record: RequestRecord) -> None:
        with self._lock:
            self._buffer.append(record)
            full = len(self._buffer) >= self.FLUSH_EVERY
        if full:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            pending, self._buffer = self._buffer, []
            if not pending:
                return
            # The record has been deleted meanwhile: do not create its directory again
            if not self.path.parent.is_dir():
                return
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write("".join(record.to_json() + "\n" for record in pending))
            except OSError:
                log.warning("Could not write the request log %s", self.path, exc_info=True)

    def summary(self) -> RequestStats | None:
        """Summary of every request in the file: all runs of the job, since the file is cleared only with the
        job's cache."""
        self.flush()
        return summarize(read_records(self.path))
