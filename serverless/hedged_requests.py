"""Hedged requests: duplicate to a second worker after p95 latency; bill once."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

log = logging.getLogger("xcelsior.serverless.hedged_requests")

HEDGE_AFTER_MS = int(os.environ.get("XCELSIOR_HEDGE_AFTER_MS", "5000"))


def hedged_requests_enabled() -> bool:
    return os.environ.get("XCELSIOR_HEDGED_REQUESTS_ENABLED", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def hedge_p95_ms(endpoint: dict, metrics: dict | None = None) -> int:
    if metrics and int(metrics.get("ttft_p95_ms") or 0) > 0:
        return int(metrics["ttft_p95_ms"])
    return int(endpoint.get("hedge_after_ms") or HEDGE_AFTER_MS)


def should_hedge_job(job: dict, *, now: float | None = None, p95_ms: int) -> bool:
    if not hedged_requests_enabled():
        return False
    if str(job.get("status") or "") != "IN_PROGRESS":
        return False
    if job.get("hedge_worker_id"):
        return False
    started = float(job.get("started_at") or 0)
    if started <= 0:
        return False
    now = now if now is not None else time.time()
    return (now - started) * 1000.0 >= float(p95_ms)


def dispatch_hedge(
    repo: Any,
    dispatcher: Any,
    endpoint: dict,
    job: dict,
    *,
    p95_ms: int,
) -> dict | None:
    """Assign duplicate in-flight job to a second worker; primary keeps running."""
    if not should_hedge_job(job, p95_ms=p95_ms):
        return None
    endpoint_id = str(endpoint["endpoint_id"])
    primary = str(job.get("worker_id") or "")
    workers = repo.list_workers(endpoint_id)
    secondary = None
    for w in workers:
        wid = str(w.get("worker_id") or "")
        if wid == primary:
            continue
        if str(w.get("state") or "") in ("ready", "idle") and int(w.get("current_concurrency") or 0) == 0:
            secondary = w
            break
    if not secondary:
        return None
    wid = str(secondary["worker_id"])
    if hasattr(repo, "mark_job_hedged"):
        repo.mark_job_hedged(str(job["job_id"]), hedge_worker_id=wid)
    repo.increment_worker_concurrency(wid)
    log.info("Hedged job %s → secondary worker %s (primary %s)", job["job_id"], wid, primary)
    return {"job_id": job["job_id"], "hedge_worker_id": wid, "primary_worker_id": primary}


def resolve_hedge_winner(
    repo: Any,
    job_id: str,
    *,
    winner_worker_id: str,
) -> None:
    """Cancel loser worker concurrency; idempotent billing uses job_id + attempt."""
    if not hasattr(repo, "complete_hedge"):
        return
    repo.complete_hedge(job_id, winner_worker_id=winner_worker_id)