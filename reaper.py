"""Stuck-job reaper — thin periodic invoker over domain transitions.

Scans the jobs table for rows stuck in a non-terminal transition state
(queued / assigned / leased / starting) past their configured deadline.

Production rules (control-plane cutover):
  - Attempt-owned / fenced jobs (``active_attempt_id IS NOT NULL``) are
    never candidates — the lease-expiry / lifecycle controllers own that
    failure class end-to-end.
  - Legacy stuck jobs are failed only through
    :func:`control_plane.stuck_jobs.fail_stuck_legacy_job` →
    :func:`scheduler.update_job_status` (CAS via ``expected_status``,
    fence gate, durable outbox SSE). No raw unconstrained SQL status
    machine in this module.

Invocation:
  Registered in api.py / bg_worker (60 s cadence). The tick is a no-op
  when no stuck jobs exist.
"""

from __future__ import annotations

import logging
import os
import time

from prometheus_client import Counter

from control_plane.stuck_jobs import DEFAULT_TIMEOUTS, fail_stuck_legacy_job

log = logging.getLogger("xcelsior.reaper")

# ── Metrics ──────────────────────────────────────────────────────────
_reaper_killed = Counter(
    "xcelsior_reaper_jobs_killed_total",
    "Jobs killed by the stuck-job reaper",
    ["status", "reason"],
)

# ── Timeouts (seconds, env-configurable) ─────────────────────────────
_TIMEOUTS: dict[str, int] = {
    status: int(os.environ.get(f"REAPER_{status.upper()}_TIMEOUT_SEC", str(default)))
    for status, default in DEFAULT_TIMEOUTS.items()
}


def list_stuck_legacy_job_ids(
    *,
    status: str,
    cutoff: float,
    conn=None,
) -> list[str]:
    """Return job_ids stuck in *status* past *cutoff* that are not attempt-owned.

    Pure candidate selection — no mutations. Excludes fenced work so the
    lease controller remains the sole failure authority for that set.
    """
    from db import _get_pg_pool

    def _select(c) -> list[str]:
        cur = c.cursor()
        if status == "queued":
            cur.execute(
                "SELECT job_id FROM jobs WHERE status = %s AND submitted_at < %s "
                "AND active_attempt_id IS NULL "
                "AND COALESCE(payload->>'queue_reason', '') != 'gpu_busy'",
                (status, cutoff),
            )
        else:
            cur.execute(
                "SELECT job_id FROM jobs WHERE status = %s AND "
                "active_attempt_id IS NULL AND "
                "COALESCE((payload->>'updated_at')::float, submitted_at) < %s",
                (status, cutoff),
            )
        return [
            str(r[0] if not isinstance(r, dict) else r["job_id"]) for r in cur.fetchall()
        ]

    if conn is not None:
        return _select(conn)
    with _get_pg_pool().connection() as owned:
        return _select(owned)


def reaper_tick() -> int:
    """Scan once; fail every legacy stuck job past its deadline.

    Returns the number of jobs successfully failed this tick.
    """
    now = time.time()
    total_killed = 0

    for status, timeout_sec in _TIMEOUTS.items():
        cutoff = now - timeout_sec
        try:
            candidates = list_stuck_legacy_job_ids(status=status, cutoff=cutoff)
        except Exception as e:
            log.error("reaper_tick candidate scan failed status=%s: %s", status, e)
            continue

        for job_id in candidates:
            try:
                updated = fail_stuck_legacy_job(
                    job_id,
                    stuck_status=status,
                    timeout_sec=timeout_sec,
                )
            except Exception as e:
                log.error(
                    "reaper_tick fail path error job=%s status=%s: %s",
                    job_id,
                    status,
                    e,
                )
                continue

            if updated is None:
                # Race lost (status advanced) or fenced — not a kill.
                log.debug(
                    "Reaper skipped job=%s (status changed or fenced between SELECT and fail)",
                    job_id,
                )
                continue

            total_killed += 1
            log.warning(
                "Reaper killed job=%s stuck in status=%s for >%ds",
                job_id,
                status,
                timeout_sec,
            )
            _reaper_killed.labels(status=status, reason="timeout").inc()

            try:
                from routes.instances import emit_lifecycle_log

                job_row = dict(updated)
                job_row["failure_reason"] = updated.get("failure_reason") or (
                    f"stuck in '{status}' state for >{timeout_sec}s without progress "
                    "(scheduler or worker never advanced the job)"
                )
                emit_lifecycle_log(job_id, status, "failed", job_row)
            except Exception as e:
                log.debug("reaper lifecycle log failed for %s: %s", job_id, e)
                try:
                    from routes.instances import push_job_log

                    push_job_log(
                        job_id,
                        f"Failed: stuck in '{status}' state for >{timeout_sec}s without progress "
                        "(scheduler or worker never advanced the job)",
                        level="error",
                    )
                except Exception:
                    pass

    return total_killed
