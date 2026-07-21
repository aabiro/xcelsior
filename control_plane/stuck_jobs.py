"""Stuck non-terminal job failure domain (blueprint §12 / reaper cutover).

The periodic reaper is a thin invoker. Failure of *legacy* stuck jobs goes
through :func:`scheduler.update_job_status` so:

- attempt-owned / fenced jobs hit the fence gate and are refused
- payload + column status stay consistent
- durable outbox SSE intents fire on the same path as other writers

Attempt-owned rows are filtered at candidate selection (lease expiry owns
that failure class); the fence is defense-in-depth.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("xcelsior.control_plane.stuck_jobs")

# Default budgets mirror reaper env defaults (seconds).
DEFAULT_TIMEOUTS: dict[str, int] = {
    "queued": 7200,
    "assigned": 180,
    "leased": 1200,
    "starting": 1200,
}


def stuck_failure_reason(stuck_status: str, timeout_sec: int) -> str:
    return (
        f"stuck in '{stuck_status}' state for >{timeout_sec}s without progress "
        "(scheduler or worker never advanced the job)"
    )


def fail_stuck_legacy_job(
    job_id: str,
    *,
    stuck_status: str,
    timeout_sec: int,
) -> dict[str, Any] | None:
    """Fail one legacy stuck job via the guarded status transition path.

    Returns the updated job dict when the CAS write succeeds, or None when
    the row raced away / is attempt-owned / missing.
    """
    from scheduler import update_job_status

    reason = stuck_failure_reason(stuck_status, timeout_sec)
    try:
        updated = update_job_status(
            job_id,
            "failed",
            expected_status=stuck_status,
            failure_reason=reason,
            failure_code="stuck-no-progress",
        )
    except RuntimeError as e:
        # Fence gate: attempt-owned jobs must not be raw-failed here.
        msg = str(e)
        if "attempt-owned" in msg or "fenced" in msg:
            log.debug(
                "stuck-job fail skipped job=%s (fenced/attempt-owned): %s",
                job_id,
                e,
            )
            return None
        raise
    return updated
