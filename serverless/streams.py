# Xcelsior — SSE stream helpers for serverless jobs

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, Iterator

from serverless.repo import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_IN_PROGRESS,
    JOB_STATUS_QUEUED,
    ServerlessRepo,
)

SSE_HEARTBEAT_INTERVAL_SEC = 15.0
SSE_POLL_INTERVAL_SEC = 0.5

SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

TERMINAL_STATUSES = frozenset(
    {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_CANCELLED}
)
ACTIVE_STATUSES = frozenset({JOB_STATUS_QUEUED, JOB_STATUS_IN_PROGRESS})


def format_sse_event(
    data: dict[str, Any] | str,
    *,
    event: str = "message",
    seq_no: int | None = None,
) -> str:
    """Format one SSE frame (OpenAI-compatible data lines)."""
    if isinstance(data, dict) and seq_no is not None:
        data = {**data, "seq_no": seq_no}
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    lines = []
    if event:
        lines.append(f"event: {event}")
    for line in payload.splitlines() or [payload]:
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines) + "\n"


def format_ping_comment() -> str:
    return ": ping\n\n"


def format_done_chunk() -> str:
    return format_sse_event("[DONE]", event="message")


def format_error_chunk(code: str, message: str, *, retryable: bool = False) -> str:
    return format_sse_event(
        {
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            }
        },
        event="error",
    )


def _terminal_frames(job: dict) -> list[str]:
    status = str(job.get("status") or "")
    if status == JOB_STATUS_FAILED:
        return [
            format_error_chunk(
                "job_failed",
                str(job.get("error") or "Job failed"),
                retryable=False,
            )
        ]
    if status == JOB_STATUS_CANCELLED:
        return [
            format_error_chunk("job_cancelled", "Job cancelled", retryable=False)
        ]
    if status == JOB_STATUS_COMPLETED:
        return [format_done_chunk()]
    return []


def _yield_new_events(
    repo: ServerlessRepo,
    job_id: str,
    *,
    after_seq: int,
) -> tuple[list[str], int, bool]:
    """Return (frames, last_seq, saw_error_event)."""
    frames: list[str] = []
    last_seq = after_seq
    saw_error = False
    events = repo.list_stream_events(job_id, after_seq=after_seq)
    for ev in events:
        last_seq = int(ev["seq_no"])
        etype = str(ev.get("event_type") or "message")
        payload = ev.get("payload") or {}
        if etype == "error":
            saw_error = True
            msg = payload.get("message") if isinstance(payload, dict) else str(payload)
            code = payload.get("code", "stream_error") if isinstance(payload, dict) else "stream_error"
            frames.append(
                format_error_chunk(str(code), str(msg or "Stream error"), retryable=False)
            )
            continue
        frames.append(
            format_sse_event(payload, event=etype, seq_no=last_seq)
        )
    return frames, last_seq, saw_error


def iter_job_stream(
    repo: ServerlessRepo,
    job_id: str,
    *,
    after_seq: int = 0,
    heartbeat: bool = True,
    max_polls: int = 0,
) -> Iterator[str]:
    """
    Replay stored stream events, optionally poll until terminal.
    max_polls=0 → single snapshot (legacy). max_polls<0 → poll until terminal.
    """
    last_seq = after_seq
    polls = 0
    last_ping = time.time()
    poll_limit = max_polls if max_polls > 0 else 1

    while polls < poll_limit:
        frames, last_seq, saw_error = _yield_new_events(repo, job_id, after_seq=last_seq)
        for frame in frames:
            yield frame
        if saw_error:
            return

        job = repo.get_job(job_id)
        if not job:
            yield format_error_chunk("job_not_found", "Job not found")
            return

        status = str(job.get("status") or "")
        if status in TERMINAL_STATUSES:
            for frame in _terminal_frames(job):
                yield frame
            return

        if heartbeat:
            now = time.time()
            if now - last_ping >= SSE_HEARTBEAT_INTERVAL_SEC:
                yield format_ping_comment()
                last_ping = now
        elif status in ACTIVE_STATUSES:
            yield format_sse_event({"heartbeat": True}, event="ping")

        if max_polls == 0:
            return

        polls += 1
        if poll_limit > 1:
            time.sleep(SSE_POLL_INTERVAL_SEC)


async def async_live_job_stream(
    repo: ServerlessRepo,
    job_id: str,
    *,
    after_seq: int = 0,
    request: Any | None = None,
    on_disconnect: Callable[[], None] | None = None,
    poll_interval: float = SSE_POLL_INTERVAL_SEC,
    heartbeat_interval: float = SSE_HEARTBEAT_INTERVAL_SEC,
) -> AsyncIterator[str]:
    """Live SSE until terminal status; reconnect-safe via after_seq."""
    last_seq = after_seq
    last_ping = time.time()

    while True:
        if request is not None:
            try:
                if await request.is_disconnected():
                    if on_disconnect:
                        on_disconnect()
                    return
            except Exception:
                pass

        frames, last_seq, saw_error = _yield_new_events(repo, job_id, after_seq=last_seq)
        for frame in frames:
            yield frame
        if saw_error:
            return

        job = repo.get_job(job_id)
        if not job:
            yield format_error_chunk("job_not_found", "Job not found")
            return

        status = str(job.get("status") or "")
        if status in TERMINAL_STATUSES:
            for frame in _terminal_frames(job):
                yield frame
            return

        now = time.time()
        if now - last_ping >= heartbeat_interval:
            yield format_ping_comment()
            last_ping = now

        await asyncio.sleep(poll_interval)


async def async_job_stream(
    repo: ServerlessRepo,
    job_id: str,
    *,
    after_seq: int = 0,
    poll_interval: float = SSE_POLL_INTERVAL_SEC,
) -> AsyncIterator[str]:
    """Poll DB for new stream events until terminal job status."""
    async for frame in async_live_job_stream(
        repo,
        job_id,
        after_seq=after_seq,
        poll_interval=poll_interval,
    ):
        yield frame