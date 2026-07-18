"""Desired-vs-observed reconciliation (blueprint §12, Phase 6).

Report-only mode: every divergence between what the control plane
*wants* on a host (active attempts) and what the agent *actually saw*
(latest observation) becomes a durable, deduplicated finding in
``reconciliation_findings``. Nothing is auto-remediated yet — the
blueprint's rollout rule is observe first, then enable actions per
finding type once the false-positive rate is understood. The existing
authority machinery (lease expiry sweep, worker fence-loss kill) remains
the acting layer; findings are the evidence that it is sufficient — or
not.

Finding types:

- ``attempt_container_missing`` (warning): an active attempt past its
  grace window has no corresponding workload in the latest observation.
  The container may have crashed without a report, or the agent lost it.
- ``stale_fence_container`` (error): a workload is running under a fence
  that is no longer the job's active attempt — the §11.5 kill should
  have fired; this is the backstop that notices when it didn't.
- ``unmanaged_workload`` (info): an ``xcl-*`` container whose job the
  control plane doesn't know at all.

Findings auto-resolve: when a later reconcile pass no longer reproduces
the condition, the open finding is marked resolved.
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass, field
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from control_plane.observations import latest_observation

log = logging.getLogger("xcelsior.control_plane.reconcile")

# An attempt this young may simply not have a container yet (image pull).
DEFAULT_MISSING_GRACE_SEC = 180.0

_ACTIVE_RUNTIME_ATTEMPT_STATUSES = ("lease_claimed", "starting", "running")
_OBSERVED_LIVE_STATES = ("preparing", "running", "paused")

# Stale-fence stop command lives long enough to survive a worker restart
# cycle; the worker GCs expired admin commands on drain.
_STOP_COMMAND_EXPIRY_SEC = 3600

# When the missing-container action expedites a lease's expiry, it stamps
# expires_at this far into the past so the very next expire_stale_leases
# sweep (its grace is much smaller) settles the attempt uniformly.
_EXPEDITE_PAST_SEC = 3600


class ActionPolicy(str, enum.Enum):
    """Per-finding-type remediation posture (blueprint Phase 6 rollout)."""

    REPORT_ONLY = "report_only"
    ENFORCE = "enforce"


# Only these finding types have a defined, safe remediation. Everything
# else is report-only regardless of env — an operator cannot enable an
# action that does not exist.
#
# - `stale_fence_container`: a container under a revoked fence should
#   never be running (§11.5); remediation is a durable stop_container
#   command by name, idempotent and harmless to the current authority.
# - `attempt_container_missing`: a runtime-active attempt whose container
#   vanished but whose worker may still be renewing the lease (a "zombie"
#   the lease-deadline sweep never catches). Remediation expedites the
#   lease's expiry so the tested lease controller revokes authority and
#   requeues — the retry mints a higher fence that fences any zombie.
_ENFORCEABLE: frozenset[str] = frozenset(
    {"stale_fence_container", "attempt_container_missing"}
)


def action_policy_for(finding_type: str) -> ActionPolicy:
    """Resolve the remediation policy for a finding type from the env.

    ``XCELSIOR_RECONCILE_ACTION_<FINDING_TYPE>=enforce`` opts a single
    enforceable finding type into action; the default (and the only
    option for non-enforceable types) is report-only. A malformed value
    fails safe to report-only.
    """
    if finding_type not in _ENFORCEABLE:
        return ActionPolicy.REPORT_ONLY
    raw = os.environ.get(
        f"XCELSIOR_RECONCILE_ACTION_{finding_type.upper()}", ""
    ).strip().lower()
    return ActionPolicy.ENFORCE if raw == "enforce" else ActionPolicy.REPORT_ONLY


def _get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


@dataclass(frozen=True)
class ReconcileReport:
    host_id: str
    findings_opened: list[str] = field(default_factory=list)
    findings_resolved: int = 0
    actions_taken: list[str] = field(default_factory=list)
    observation_id: str | None = None


def _open_finding(
    conn: Connection,
    *,
    resource_type: str,
    resource_id: str,
    finding_type: str,
    severity: str,
    summary: str,
    desired: dict[str, Any] | None,
    observed: dict[str, Any] | None,
) -> str | None:
    """Record a finding unless an identical one is already open.

    Returns the new finding_id when a row was created, else None (an
    identical finding is already open — dedupe, and the newly-created
    return signals the caller to run any enforced remediation exactly
    once per occurrence).
    """
    existing = conn.execute(
        """
        SELECT 1 FROM reconciliation_findings
         WHERE resource_type = %s AND resource_id = %s
           AND finding_type = %s AND resolved_at IS NULL
         LIMIT 1
        """,
        (resource_type, resource_id, finding_type),
    ).fetchone()
    if existing is not None:
        return None
    row = conn.execute(
        """
        INSERT INTO reconciliation_findings
            (resource_type, resource_id, finding_type, severity, summary,
             desired, observed, action_taken)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'report_only')
        RETURNING finding_id
        """,
        (
            resource_type, resource_id, finding_type, severity, summary,
            Jsonb(desired) if desired else None,
            Jsonb(observed) if observed else None,
        ),
    ).fetchone()
    log.warning("reconcile finding [%s] %s %s: %s",
                severity, finding_type, resource_id, summary)
    return str(_get(row, "finding_id", 0)) if row is not None else None


def _enqueue_stop_container(
    conn: Connection, *, host_id: str, container_name: str, reason: str
) -> dict[str, Any]:
    """Durable stop_container command by name (idempotent per container).

    Enqueued inside the reconcile transaction, so the stop intent is
    atomic with recording the action. A stale-fence container's fence is
    not current, so the fenced /agent/v2 path would reject it — this
    admin-level command (attempt_id NULL) is delivered by the worker's v1
    drain, which already knows stop_container.
    """
    row = conn.execute(
        """
        INSERT INTO agent_commands
            (host_id, command, args, status, created_by, expires_at,
             idempotency_key)
        VALUES (%s, 'stop_container', %s, 'pending', 'reconciler',
                EXTRACT(EPOCH FROM NOW()) + %s, %s)
        ON CONFLICT (host_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL
            DO NOTHING
        RETURNING id
        """,
        (
            host_id,
            Jsonb({"container_name": container_name, "reason": reason}),
            _STOP_COMMAND_EXPIRY_SEC,
            f"reconcile_stop:{host_id}:{container_name}",
        ),
    ).fetchone()
    enqueued = row is not None
    return {
        "action": "stop_container",
        "container_name": container_name,
        "enqueued": enqueued,
        "reason": reason,
    }


def _expedite_lease_expiry(
    conn: Connection, *, attempt_id: str
) -> dict[str, Any]:
    """Stamp this attempt's active lease as already-expired.

    The reconcile transaction does not settle the attempt itself — it
    hands the work to the lease controller by moving ``expires_at`` well
    into the past, so the next ``expire_stale_leases`` sweep (running in
    the scheduler service tick) performs the one tested terminal
    settlement: attempt → lost, allocations released once, start command
    cancelled, job requeued with a durable reason, higher fence on retry.
    Reusing that path avoids duplicating settlement SQL and keeps a
    single authority for attempt failure (§12 domain controllers).
    """
    result = conn.execute(
        """
        UPDATE placement_leases
           SET expires_at = clock_timestamp() - make_interval(secs => %s)
         WHERE attempt_id = %s
           AND status = 'active'
        """,
        (_EXPEDITE_PAST_SEC, attempt_id),
    )
    return {
        "action": "expire_lease",
        "attempt_id": attempt_id,
        "leases_expedited": result.rowcount,
    }


def _record_action(
    conn: Connection, finding_id: str, action_taken: str, result: dict[str, Any]
) -> None:
    conn.execute(
        """
        UPDATE reconciliation_findings
           SET action_taken = %s, action_result = %s
         WHERE finding_id = %s
        """,
        (action_taken, Jsonb(result), finding_id),
    )


def _resolve_findings_not_in(
    conn: Connection,
    *,
    host_id: str,
    still_open: set[tuple[str, str, str]],
) -> int:
    """Resolve open findings for this host's resources that no longer hold."""
    rows = conn.execute(
        """
        SELECT finding_id, resource_type, resource_id, finding_type
          FROM reconciliation_findings
         WHERE resolved_at IS NULL
           AND (
                (resource_type = 'host' AND resource_id = %(host)s)
             OR (resource_type = 'attempt' AND resource_id IN (
                     SELECT attempt_id::text FROM job_attempts
                      WHERE host_id = %(host)s))
           )
        """,
        {"host": host_id},
    ).fetchall()
    resolved = 0
    for row in rows:
        key = (
            str(_get(row, "resource_type", 1)),
            str(_get(row, "resource_id", 2)),
            str(_get(row, "finding_type", 3)),
        )
        if key in still_open:
            continue
        conn.execute(
            "UPDATE reconciliation_findings SET resolved_at = clock_timestamp() "
            "WHERE finding_id = %s",
            (_get(row, "finding_id", 0),),
        )
        resolved += 1
    return resolved


def reconcile_host(
    conn: Connection,
    host_id: str,
    *,
    missing_grace_sec: float = DEFAULT_MISSING_GRACE_SEC,
) -> ReconcileReport:
    """One report-only reconcile pass for a host, in the caller's txn."""
    observation = latest_observation(conn, host_id)
    if observation is None:
        return ReconcileReport(host_id=host_id)
    observed = observation["workloads"]
    observed_by_attempt = {
        str(w["attempt_id"]): w for w in observed if w.get("attempt_id")
    }

    # Desired: this host's runtime-active attempts (with their job's
    # current-authority view for fence comparison).
    desired_rows = conn.execute(
        """
        SELECT a.attempt_id, a.job_id, a.status, a.fencing_token,
               GREATEST(
                   COALESCE(EXTRACT(EPOCH FROM (%(obs_at)s::timestamptz
                                                - a.lease_claimed_at)), 0),
                   0
               ) AS age_at_observation
          FROM job_attempts a
          JOIN jobs j ON j.job_id = a.job_id AND j.active_attempt_id = a.attempt_id
         WHERE a.host_id = %(host)s
           AND a.status = ANY(%(statuses)s)
        """,
        {
            "host": host_id,
            "obs_at": observation["received_at"],
            "statuses": list(_ACTIVE_RUNTIME_ATTEMPT_STATUSES),
        },
    ).fetchall()

    still_open: set[tuple[str, str, str]] = set()
    opened: list[str] = []
    actions: list[str] = []

    for row in desired_rows:
        attempt_id = str(_get(row, "attempt_id", 0))
        job_id = str(_get(row, "job_id", 1))
        status = str(_get(row, "status", 2))
        fence = int(_get(row, "fencing_token", 3))
        age = float(_get(row, "age_at_observation", 4) or 0)
        seen = observed_by_attempt.get(attempt_id)
        if seen is None or seen.get("state") not in _OBSERVED_LIVE_STATES:
            if age < missing_grace_sec:
                continue  # young attempt: container may still be starting
            key = ("attempt", attempt_id, "attempt_container_missing")
            still_open.add(key)
            finding_id = _open_finding(
                conn,
                resource_type="attempt", resource_id=attempt_id,
                finding_type="attempt_container_missing", severity="warning",
                summary=(
                    f"attempt {attempt_id[:8]} for job {job_id} is {status} "
                    f"but the latest observation from {host_id} has no live "
                    f"container for it"
                ),
                desired={"job_id": job_id, "status": status, "fence": fence},
                observed={"observation_id": observation["observation_id"],
                          "workload": seen},
            )
            if finding_id is not None:
                opened.append("attempt_container_missing")
                # Enforced remediation: hand the zombie attempt to the
                # lease controller by expediting its lease expiry.
                if (
                    action_policy_for("attempt_container_missing")
                    is ActionPolicy.ENFORCE
                ):
                    result = _expedite_lease_expiry(conn, attempt_id=attempt_id)
                    _record_action(conn, finding_id, "expire_lease", result)
                    actions.append("attempt_container_missing")

    # Observed → desired: stale fences and unknown workloads.
    current_fence_by_job = {
        str(_get(r, "job_id", 1)): int(_get(r, "fencing_token", 3))
        for r in desired_rows
    }
    for w in observed:
        if w.get("state") not in _OBSERVED_LIVE_STATES:
            continue
        job_id = w.get("job_id")
        fence = w.get("fencing_token")
        if job_id and fence is not None:
            current = current_fence_by_job.get(str(job_id))
            if current is not None and int(fence) != current:
                key = ("host", host_id, "stale_fence_container")
                still_open.add(key)
                finding_id = _open_finding(
                    conn,
                    resource_type="host", resource_id=host_id,
                    finding_type="stale_fence_container", severity="error",
                    summary=(
                        f"container {w.get('container_name')} runs job {job_id} "
                        f"under fence {fence}, but the current authority is "
                        f"{current} — worker fence-loss kill did not fire"
                    ),
                    desired={"job_id": job_id, "fence": current},
                    observed=w,
                )
                if finding_id is not None:
                    opened.append("stale_fence_container")
                    # First enforced remediation (§11.5 backstop): stop the
                    # stale container, once per occurrence, if enabled.
                    container_name = w.get("container_name")
                    if (
                        container_name
                        and action_policy_for("stale_fence_container")
                        is ActionPolicy.ENFORCE
                    ):
                        result = _enqueue_stop_container(
                            conn,
                            host_id=host_id,
                            container_name=str(container_name),
                            reason=(
                                f"stale fence {fence} (current authority "
                                f"{current}) for job {job_id}"
                            ),
                        )
                        _record_action(
                            conn, finding_id, "stop_container", result
                        )
                        actions.append("stale_fence_container")
            continue
        if job_id is None:
            continue  # not attributable at all; ingest normalized elsewhere
        exists = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = %s", (str(job_id),)
        ).fetchone()
        if exists is None:
            key = ("host", host_id, "unmanaged_workload")
            still_open.add(key)
            finding_id = _open_finding(
                conn,
                resource_type="host", resource_id=host_id,
                finding_type="unmanaged_workload", severity="info",
                summary=(
                    f"container {w.get('container_name')} claims job {job_id}, "
                    f"which the control plane does not know"
                ),
                desired=None, observed=w,
            )
            if finding_id is not None:
                opened.append("unmanaged_workload")

    resolved = _resolve_findings_not_in(conn, host_id=host_id, still_open=still_open)
    return ReconcileReport(
        host_id=host_id,
        findings_opened=opened,
        findings_resolved=resolved,
        actions_taken=actions,
        observation_id=observation["observation_id"],
    )


# ── Queue processing ─────────────────────────────────────────────────

_QUEUE_CLAIM_TTL_SEC = 60
_QUEUE_RETRY_BACKOFF_SEC = 30.0


def process_due(
    conn: Connection,
    *,
    worker_id: str,
    limit: int = 20,
    missing_grace_sec: float = DEFAULT_MISSING_GRACE_SEC,
) -> dict[str, int]:
    """Claim and reconcile due queue entries (§12.3), settling each.

    Success deletes the entry (the queue is PK-coalesced — a new
    observation re-enqueues); failure records the error and backs off.
    """
    rows = conn.execute(
        """
        WITH due AS (
            SELECT resource_type, resource_id
              FROM reconciliation_queue
             WHERE due_at <= clock_timestamp()
               AND (claim_expires_at IS NULL
                    OR claim_expires_at < clock_timestamp())
             ORDER BY priority DESC, due_at
             LIMIT %(limit)s
               FOR UPDATE SKIP LOCKED
        )
        UPDATE reconciliation_queue q
           SET claim_owner = %(owner)s,
               claim_expires_at = clock_timestamp()
                                  + make_interval(secs => %(ttl)s),
               attempt_count = q.attempt_count + 1,
               updated_at = clock_timestamp()
          FROM due
         WHERE q.resource_type = due.resource_type
           AND q.resource_id = due.resource_id
        RETURNING q.resource_type, q.resource_id
        """,
        {"limit": limit, "owner": worker_id, "ttl": _QUEUE_CLAIM_TTL_SEC},
    ).fetchall()

    stats = {"claimed": len(rows), "reconciled": 0, "failed": 0,
             "findings_opened": 0, "findings_resolved": 0, "actions_taken": 0}
    for row in rows:
        rtype = str(_get(row, "resource_type", 0))
        rid = str(_get(row, "resource_id", 1))
        try:
            if rtype == "host":
                report = reconcile_host(
                    conn, rid, missing_grace_sec=missing_grace_sec
                )
                stats["findings_opened"] += len(report.findings_opened)
                stats["findings_resolved"] += report.findings_resolved
                stats["actions_taken"] += len(report.actions_taken)
            else:
                log.info("reconcile: no controller for %s/%s yet", rtype, rid)
            conn.execute(
                "DELETE FROM reconciliation_queue "
                "WHERE resource_type = %s AND resource_id = %s",
                (rtype, rid),
            )
            stats["reconciled"] += 1
        except Exception as exc:  # contain per-entry: bounded backoff
            stats["failed"] += 1
            conn.execute(
                """
                UPDATE reconciliation_queue
                   SET claim_owner = NULL,
                       claim_expires_at = NULL,
                       last_error = left(%s, 500),
                       due_at = clock_timestamp()
                                + make_interval(secs => %s),
                       updated_at = clock_timestamp()
                 WHERE resource_type = %s AND resource_id = %s
                """,
                (repr(exc), _QUEUE_RETRY_BACKOFF_SEC, rtype, rid),
            )
            log.exception("reconcile %s/%s failed", rtype, rid)
    return stats
