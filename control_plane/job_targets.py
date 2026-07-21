"""Residual agent-command targeting for legacy vs attempt-owned jobs.

Worker v2 starts attempt containers as ``xcl-{job_id}-{attempt_id[:8]}``
(``worker_agent._v2_container_name``). Residual ops that still drain on
the unfenced v1 path (reinject, volume hot mount/unmount, snapshot,
reset) must resolve that identity for attempt-owned work — never default
to ``xcl-{job_id}`` while an active attempt owns the host container.

Pure-legacy jobs keep the historical ``payload.container_name`` /
``xcl-{job_id}`` name. Fenced-history jobs without a live active attempt
must not receive authority-sensitive unfenced container commands that
guess the legacy name (fail closed at the call site).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

JobTargetClass = Literal["legacy", "attempt_owned", "fenced_history"]


def attempt_container_name(job_id: str, attempt_id: str) -> str:
    """Match worker_agent._v2_container_name exactly."""
    return f"xcl-{job_id}-{str(attempt_id)[:8]}"


def legacy_container_name(job_id: str, payload_name: str | None = None) -> str:
    name = (payload_name or "").strip()
    return name or f"xcl-{job_id}"


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
    """Resolved residual-command target for one job.

    ``allows_unfenced_container_command`` is True for pure legacy and for
    attempt-owned work (with the correct attempt container name). It is
    False for fenced-history without live authority — residual writers
    that would guess ``xcl-{job_id}`` must fail closed or skip instead.
    """

    job_id: str
    host_id: str | None
    status: str
    class_: JobTargetClass
    active_attempt_id: str | None
    container_name: str
    allows_unfenced_container_command: bool


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
    payload_container_name: str | None = None,
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
            allows_unfenced_container_command=True,
        )
    if has_fenced_history:
        return JobCommandTarget(
            job_id=job_id,
            host_id=host_id or None,
            status=status or "",
            class_="fenced_history",
            active_attempt_id=None,
            container_name=legacy_container_name(job_id, payload_container_name),
            allows_unfenced_container_command=False,
        )
    return JobCommandTarget(
        job_id=job_id,
        host_id=host_id or None,
        status=status or "",
        class_="legacy",
        active_attempt_id=None,
        container_name=legacy_container_name(job_id, payload_container_name),
        allows_unfenced_container_command=True,
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
            SELECT j.job_id, j.status, j.host_id, j.active_attempt_id,
                   j.payload->>'container_name' AS container_name,
                   EXISTS (
                       SELECT 1 FROM job_attempts a WHERE a.job_id = j.job_id
                   ) AS has_fenced_history
              FROM jobs j
             WHERE j.job_id = %s
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
        has_fenced_history=bool(_row_get(row, "has_fenced_history", 5)),
        payload_container_name=_row_get(row, "container_name", 4),
    )
