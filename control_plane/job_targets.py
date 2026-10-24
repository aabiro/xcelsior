"""Resolve job directives only from current native attempt and lease authority.

Historical observations and guessed legacy container names never authorize a
container mutation. Host administration has its separate explicit protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

JobTargetClass = Literal["unadmitted", "attempt_owned", "fenced_history"]


def attempt_container_name(job_id: str, attempt_id: str) -> str:
    """Match worker_agent._v2_container_name exactly."""
    return f"xcl-{job_id}-{str(attempt_id)[:8]}"


def is_fenced_history_job(
    *,
    active_attempt_id: object | None = None,
    has_fenced_history: bool = False,
) -> bool:
    """True when lifecycle authority is the fenced path (active or history).

    Same classification as billing start/restart and bg_worker stop
    redelivery: an active attempt *or* any ``job_attempts`` row means
    unfenced lifecycle dual-commands must not run against the job.
    """
    if active_attempt_id is not None:
        return True
    return bool(has_fenced_history)


@dataclass(frozen=True)
class JobCommandTarget:
    """Native target; a missing live attempt always refuses container work."""

    job_id: str
    host_id: str | None
    status: str
    class_: JobTargetClass
    active_attempt_id: str | None
    container_name: str
    has_live_attempt: bool


def _row_get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row).get(key)
    return row[index]


def classify_job_target(
    *,
    job_id: str,
    host_id: str | None,
    status: str,
    active_attempt_id: object | None,
    has_fenced_history: bool,
) -> JobCommandTarget:
    """Pure policy: classify and name the residual container target."""
    if active_attempt_id is not None:
        attempt_id = str(active_attempt_id)
        return JobCommandTarget(
            job_id=job_id,
            host_id=host_id or None,
            status=status or "",
            class_="attempt_owned",
            active_attempt_id=attempt_id,
            container_name=attempt_container_name(job_id, attempt_id),
            has_live_attempt=True,
        )
    if has_fenced_history:
        return JobCommandTarget(
            job_id=job_id,
            host_id=host_id or None,
            status=status or "",
            class_="fenced_history",
            active_attempt_id=None,
            container_name="",
            has_live_attempt=False,
        )
    return JobCommandTarget(
        job_id=job_id,
        host_id=host_id or None,
        status=status or "",
        class_="unadmitted",
        active_attempt_id=None,
        container_name="",
        has_live_attempt=False,
    )


def resolve_job_command_target(job_id: str) -> JobCommandTarget | None:
    """Look up a job and resolve residual-command container identity.

    Returns ``None`` when the job row is missing. Uses real Postgres
    (``active_attempt_id`` + ``EXISTS job_attempts``).
    """
    from db import _get_pg_pool

    pool = _get_pg_pool()
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT j.job_id,j.status,j.host_id,
                   CASE WHEN l.lease_id IS NOT NULL THEN a.attempt_id END AS active_attempt_id,
                   EXISTS(SELECT 1 FROM job_attempts history WHERE history.job_id=j.job_id)
                     AS has_fenced_history
              FROM jobs j
              LEFT JOIN job_attempts a ON a.attempt_id=j.active_attempt_id
               AND a.job_id=j.job_id AND a.host_id=j.host_id AND a.job_generation=j.generation
               AND a.spec_hash=j.spec_hash AND a.status='running'
              LEFT JOIN placement_leases l ON l.attempt_id=a.attempt_id AND l.job_id=j.job_id
               AND l.host_id=a.host_id AND l.fencing_token=a.fencing_token
               AND l.status='active' AND l.expires_at>clock_timestamp()
             WHERE j.job_id=%s
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return classify_job_target(
        job_id=str(_row_get(row, "job_id", 0)),
        host_id=_row_get(row, "host_id", 2),
        status=str(_row_get(row, "status", 1) or ""),
        active_attempt_id=_row_get(row, "active_attempt_id", 3),
        has_fenced_history=bool(_row_get(row, "has_fenced_history", 4)),
    )
