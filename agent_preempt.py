"""Worker preemption queue — shared between scheduler and agent routes."""

from __future__ import annotations

import threading
from collections import defaultdict

_lock = threading.Lock()
_pending: dict[str, list[str]] = defaultdict(list)


def schedule_host_preemptions(host_id: str, job_ids: list[str]) -> None:
    """Queue job IDs for a host agent to stop on next GET /agent/preempt/{host_id}."""
    if not host_id or not job_ids:
        return
    with _lock:
        for job_id in job_ids:
            if job_id and job_id not in _pending[host_id]:
                _pending[host_id].append(job_id)


def pop_host_preemptions(host_id: str) -> list[str]:
    """Drain and return pending preemption job IDs for a host."""
    with _lock:
        return _pending.pop(host_id, [])


def peek_host_preemptions(host_id: str) -> list[str]:
    """Return a copy of pending preemptions without draining."""
    with _lock:
        return list(_pending.get(host_id, []))