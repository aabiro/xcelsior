"""Stuck-job reaper.

Periodically scans the jobs table for rows stuck in a non-terminal transition
state (queued / assigned / starting) past their configured deadline and fails
them. The reaper exists because the scheduler + worker_agent + SSH relay chain
has several points where a job can wedge silently:

  - queued:   user submitted, scheduler never matched (no suitable host online)
  - assigned: scheduler matched a host, worker agent never claimed (agent crash,
              network partition, host reboot between NOTIFY and poll)
  - starting: worker pulled image but container never reached running (bad image,
              OOM at start, cudnn mismatch, volume mount failure)

Without this, instances accumulate in limbo — users see "starting…" forever,
admin dashboards show inflated active counts, billing may tick incorrectly, and
scheduler re-matching can be blocked.

Design principles:
  - Compare-and-swap idempotency: UPDATE ... WHERE status = <expected> so a
    worker that simultaneously transitions the row to running/failed wins
    trivially (0 rows returned = back off silently).
  - Env-configurable timeouts: per-status budgets so ops can tune without code
    changes. Defaults chosen for production safety (long queued budget because
    users legitimately queue overnight; short assigned because that transition
    should complete in seconds; medium starting because image pulls can be slow).
  - First-class observability: every kill emits a structured log line AND
    increments `xcelsior_reaper_jobs_killed_total{status, reason}`. The Prom
    metric lets ops alert on reaper surges (e.g. "more than 10 reaps/hour"
    suggests scheduler or image infrastructure is degraded).
  - Transition logging: each killed job gets a job_log entry so users see
    "Failed: stuck in starting state for >1200s" instead of silent "failed".

Invocation:
  Registered in api.py's background task list (60 s cadence). The tick function
  is a no-op if no stuck jobs exist, so 60 s is cheap.
"""

from __future__ import annotations

import logging
import os
import time

from prometheus_client import Counter

log = logging.getLogger("xcelsior.reaper")

# ── Metrics ──────────────────────────────────────────────────────────
_reaper_killed = Counter(
    "xcelsior_reaper_jobs_killed_total",
    "Jobs killed by the stuck-job reaper",
    ["status", "reason"],
)

# ── Timeouts (seconds, env-configurable) ─────────────────────────────
# Defaults align with expected transition budgets:
#   - queued    2h   — users legitimately queue jobs overnight; 2h catches
#                       truly abandoned queues without false-positives.
#   - assigned  3m   — NOTIFY fires instantly; workers poll every 10s;
#                       3m is 18× normal budget.
#   - starting  20m  — image pull over slow links can take 5-15m for large
#                       CUDA base images; 20m covers worst-case + container init.
_TIMEOUTS: dict[str, int] = {
    "queued": int(os.environ.get("REAPER_QUEUED_TIMEOUT_SEC", "7200")),
    "assigned": int(os.environ.get("REAPER_ASSIGNED_TIMEOUT_SEC", "180")),
    "starting": int(os.environ.get("REAPER_STARTING_TIMEOUT_SEC", "1200")),
}


def reaper_tick() -> int:
    """Scan once, fail every stuck job past its deadline.

    Returns the number of jobs successfully killed this tick (useful for
    admin debug endpoints that want to trigger a reap and see the result).
    """
    from db import _get_pg_pool

    now = time.time()
    total_killed = 0

    for status, timeout_sec in _TIMEOUTS.items():
        cutoff = now - timeout_sec
        try:
            with _get_pg_pool().connection() as conn, conn.cursor() as cur:
                # Candidate selection — use different timestamp source per status:
                #   - queued: submitted_at is a real column, indexed as part of
                #     the jobs PK workflow; simplest signal of "how long queued".
                #   - assigned/starting: the row's last transition time lives in
                #     payload->>'updated_at' (set by update_job_status). Fall back
                #     to submitted_at for legacy rows without that field.
                if status == "queued":
                    cur.execute(
                        "SELECT job_id FROM jobs WHERE status = %s AND submitted_at < %s",
                        (status, cutoff),
                    )
                else:
                    cur.execute(
                        "SELECT job_id FROM jobs WHERE status = %s AND "
                        "COALESCE((payload->>'updated_at')::float, submitted_at) < %s",
                        (status, cutoff),
                    )
                candidates = [r[0] for r in cur.fetchall()]

                for job_id in candidates:
                    # Compare-and-swap: only transition if the row is STILL in
                    # `status`. Racing workers that already claimed/completed the
                    # job win trivially — we get 0 rows back and move on.
                    cur.execute(
                        "UPDATE jobs SET status = 'failed', "
                        "payload = jsonb_set("
                        "  jsonb_set("
                        "    jsonb_set(payload, '{failure_reason}', '\"stuck-no-progress\"'::jsonb), "
                        "    '{completed_at}', to_jsonb(EXTRACT(EPOCH FROM NOW()))), "
                        "  '{updated_at}', to_jsonb(EXTRACT(EPOCH FROM NOW()))) "
                        "WHERE job_id = %s AND status = %s "
                        "RETURNING job_id",
                        (job_id, status),
                    )
                    won = cur.fetchone() is not None
                    if not won:
                        log.debug(
                            "Reaper lost race for job=%s (status changed between SELECT and UPDATE)",
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

                    # Surface the reason to the user — avoids silent "failed"
                    # rows in the UI. Guarded in a try so log-push failure never
                    # blocks the reaper itself.
                    try:
                        from routes.instances import push_job_log

                        push_job_log(
                            job_id,
                            f"Failed: stuck in '{status}' state for >{timeout_sec}s without progress "
                            "(scheduler or worker never advanced the job)",
                            level="error",
                        )
                    except Exception as e:
                        log.debug("reaper log push failed for %s: %s", job_id, e)

        except Exception as e:
            log.error("reaper_tick error for status=%s: %s", status, e)

    return total_killed
