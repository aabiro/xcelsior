"""Fenced lifecycle controller for attempt-owned (v2) jobs — P5.5.

Terminate and cancel of attempt-owned work must not use the legacy unfenced
kill/detach path. This module records stop/terminate/cancel intent on the
job row and enqueues a durable ``stop_attempt`` bound to the current
host/attempt/fence — **in one transaction**, so a refused enqueue never
leaves the job stuck in ``stopping`` without a command.

Volume detach stays deferred until observation/ACK settles the attempt
(or a later residual step); never pre-ACK.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from control_plane.commands import (
    CommandProtocolError,
    EnqueuedAttemptCommand,
    enqueue_current_attempt_command,
)
from control_plane.db import run_transaction

log = logging.getLogger("xcelsior.control_plane.lifecycle")

LifecycleIntent = Literal["stop", "cancel", "terminate", "restart", "resume"]


@dataclass(frozen=True)
class FencedLifecycleResult:
    ok: bool
    job_id: str
    intent: str
    status: str
    reason: str | None = None
    attempt_id: str | None = None
    command_id: str | None = None
    fencing_token: int | None = None
    host_id: str | None = None
    # True only when this call inserted a new agent_commands row (not an
    # idempotent collision on the same intent:attempt key). Billing anchors
    # and similar side effects must gate on this.
    command_created: bool = False


@dataclass(frozen=True)
class FreshAttemptResult:
    """Outcome of re-admitting a stopped fenced job for a new placement."""

    ok: bool
    job_id: str
    status: str
    reason: str | None = None
    intent: str = "resume"


def _mark_stopping_with_intent(
    conn: Any,
    *,
    job_id: str,
    intent: LifecycleIntent,
    now: float,
    reason_tag: str,
) -> None:
    """Project intermediate ``stopping`` and durable lifecycle_intent."""
    conn.execute(
        """
        UPDATE jobs
           SET status = 'stopping',
               payload = jsonb_set(
                   jsonb_set(
                       jsonb_set(
                           jsonb_set(
                               COALESCE(payload, '{}'::jsonb),
                               '{stopping_at}',
                               to_jsonb(%s::float),
                               true
                           ),
                           '{status}',
                           '"stopping"'::jsonb,
                           true
                       ),
                       '{stop_reason}',
                       %s::jsonb,
                       true
                   ),
                   '{lifecycle_intent}',
                   %s::jsonb,
                   true
               ),
               version = version + 1,
               updated_at = clock_timestamp()
         WHERE job_id = %s
        """,
        (now, json.dumps(reason_tag), json.dumps(intent), job_id),
    )


def request_fenced_stop_remove(
    *,
    job_id: str,
    intent: LifecycleIntent,
    created_by: str,
    container_name: str | None = None,
    reason_tag: str | None = None,
) -> FencedLifecycleResult:
    """Record terminate/cancel intent and enqueue fenced ``stop_attempt``.

    Intent projection and command insert commit together. If the job has no
    active fenced authority, the transaction rolls back and the job row is
    left unchanged (not ``stopping`` with zero commands).

    ``preserve=False`` so the worker stops and removes the attempt container.
    Idempotent on the same attempt via command idempotency key
    ``{intent}:{attempt_id}``.
    """
    # User stop (preserve=True) stays on BillingEngine.stop_instance.
    # resume is requeue-only (see request_fresh_attempt_resume).
    if intent not in ("cancel", "terminate", "restart"):
        return FencedLifecycleResult(
            ok=False,
            job_id=job_id,
            intent=str(intent),
            status="",
            reason="invalid_intent",
        )

    now = time.time()
    tag = reason_tag or {
        "terminate": "user_terminated",
        "cancel": "user_cancelled",
        "restart": "user_restart",
    }.get(intent, intent)

    def _txn(conn: Any) -> FencedLifecycleResult:
        job = conn.execute(
            """
            SELECT job_id, status, host_id, active_attempt_id,
                   payload->>'container_name' AS container_name,
                   payload->>'lifecycle_intent' AS lifecycle_intent
              FROM jobs
             WHERE job_id = %s
               AND status NOT IN (
                   'terminated', 'completed', 'failed', 'preempted', 'cancelled'
               )
             FOR UPDATE
            """,
            (job_id,),
        ).fetchone()
        if job is None:
            return FencedLifecycleResult(
                ok=False,
                job_id=job_id,
                intent=intent,
                status="",
                reason="already_terminal_or_not_found",
            )
        # Support both dict_row and tuple_row pool connections.
        if isinstance(job, dict):
            active_attempt_id = job.get("active_attempt_id")
            job_status = str(job.get("status") or "")
            cname_from_job = job.get("container_name")
        else:
            active_attempt_id = job[3]
            job_status = str(job[1] or "")
            cname_from_job = job[4]

        if not active_attempt_id:
            return FencedLifecycleResult(
                ok=False,
                job_id=job_id,
                intent=intent,
                status=job_status,
                reason="not_attempt_owned",
            )

        attempt_id = str(active_attempt_id)
        cname = container_name or cname_from_job or f"xcl-{job_id}"
        idemp_key = f"{intent}:{attempt_id}"

        prior = conn.execute(
            """
            SELECT command_id FROM agent_commands
             WHERE job_id = %s
               AND idempotency_key = %s
             LIMIT 1
            """,
            (job_id, idemp_key),
        ).fetchone()
        command_created = prior is None

        # Authority check + command insert first; failure raises and rolls
        # back any subsequent mark (and we mark after so refuse leaves job
        # untouched even if mark ran — still order enqueue before mark).
        enqueued: EnqueuedAttemptCommand = enqueue_current_attempt_command(
            conn,
            job_id=job_id,
            command="stop_attempt",
            args={
                "container_name": cname,
                "preserve": False,
                "intent": intent,
            },
            created_by=created_by,
            idempotency_key=idemp_key,
        )

        # Intermediate projection — never terminal before worker proof.
        _mark_stopping_with_intent(
            conn,
            job_id=job_id,
            intent=intent,
            now=now,
            reason_tag=tag,
        )

        return FencedLifecycleResult(
            ok=True,
            job_id=job_id,
            intent=intent,
            status="stopping",
            attempt_id=enqueued.attempt_id,
            command_id=enqueued.command_id,
            fencing_token=enqueued.fencing_token,
            host_id=enqueued.host_id,
            command_created=command_created,
        )

    try:
        result = run_transaction(_txn, what=f"fenced_{intent}")
    except CommandProtocolError as exc:
        log.warning(
            "fenced %s enqueue refused for %s: %s", intent, job_id, exc
        )
        # No durable write — job projection unchanged.
        return FencedLifecycleResult(
            ok=False,
            job_id=job_id,
            intent=intent,
            status="",
            reason="no_active_fenced_authority",
        )
    except Exception as exc:
        log.warning("fenced %s failed for %s: %s", intent, job_id, exc)
        return FencedLifecycleResult(
            ok=False,
            job_id=job_id,
            intent=intent,
            status="",
            reason="enqueue_failed",
        )

    if result.ok:
        log.info(
            "fenced %s queued job=%s attempt=%s fence=%s cmd=%s created=%s",
            intent,
            job_id,
            (result.attempt_id or "")[:8],
            result.fencing_token,
            (result.command_id or "")[:8],
            result.command_created,
        )
    return result


def request_fresh_attempt_resume(
    *,
    job_id: str,
    created_by: str,
    intent: LifecycleIntent = "resume",
) -> FreshAttemptResult:
    """Re-admit a stopped fenced-history job for a **new** placement attempt.

    Does not enqueue ``start_container`` / revive the old fence labels.
    Sets ``status=queued`` (trigger → phase=pending, desired_state=running),
    clears host assignment and schedule-claim residue, records
    ``lifecycle_intent``. Scheduler claim/reserve creates the next attempt.
    """
    if intent not in ("resume", "restart"):
        return FreshAttemptResult(
            ok=False, job_id=job_id, status="", reason="invalid_intent", intent=str(intent)
        )

    now = time.time()

    def _txn(conn: Any) -> FreshAttemptResult:
        job = conn.execute(
            """
            SELECT job_id, status, host_id, active_attempt_id,
                   payload->>'lifecycle_intent' AS lifecycle_intent,
                   EXISTS (
                       SELECT 1 FROM job_attempts a WHERE a.job_id = jobs.job_id
                   ) AS has_fenced_history
              FROM jobs
             WHERE job_id = %s
             FOR UPDATE
            """,
            (job_id,),
        ).fetchone()
        if job is None:
            return FreshAttemptResult(
                ok=False, job_id=job_id, status="", reason="not_found", intent=intent
            )
        if isinstance(job, dict):
            status = str(job.get("status") or "")
            active_attempt_id = job.get("active_attempt_id")
            has_fenced = bool(job.get("has_fenced_history"))
        else:
            status = str(job[1] or "")
            active_attempt_id = job[3]
            has_fenced = bool(job[5])

        # Idempotent: already waiting for placement.
        if status == "queued" and not active_attempt_id:
            return FreshAttemptResult(
                ok=True, job_id=job_id, status="queued", intent=intent
            )

        if status == "stopping" and not active_attempt_id:
            # Tear-down already in flight without authority — treat as resume.
            pass
        elif status not in ("stopped", "stopping"):
            return FreshAttemptResult(
                ok=False,
                job_id=job_id,
                status=status,
                reason="not_stopped",
                intent=intent,
            )

        if active_attempt_id:
            return FreshAttemptResult(
                ok=False,
                job_id=job_id,
                status=status,
                reason="still_attempt_owned",
                intent=intent,
            )
        if not has_fenced:
            return FreshAttemptResult(
                ok=False,
                job_id=job_id,
                status=status,
                reason="not_fenced_history",
                intent=intent,
            )

        conn.execute(
            """
            UPDATE jobs
               SET status = 'queued',
                   host_id = NULL,
                   active_attempt_id = NULL,
                   schedule_claim_owner = NULL,
                   schedule_claim_token = NULL,
                   schedule_claim_expires_at = NULL,
                   generation = generation + 1,
                   submitted_at = %s,
                   payload = (
                       jsonb_set(
                           jsonb_set(
                               jsonb_set(
                                   COALESCE(payload, '{}'::jsonb)
                                       - 'stopped_at'
                                       - 'stop_reason'
                                       - 'stopping_at',
                                   '{status}',
                                   '"queued"'::jsonb,
                                   true
                               ),
                               '{lifecycle_intent}',
                               %s::jsonb,
                               true
                           ),
                           '{resumed_at}',
                           to_jsonb(%s::float),
                           true
                       )
                   ),
                   version = version + 1,
                   updated_at = clock_timestamp()
             WHERE job_id = %s
            """,
            (now, json.dumps(intent), now, job_id),
        )
        log.info(
            "fresh-attempt resume job=%s intent=%s by=%s",
            job_id,
            intent,
            created_by,
        )
        return FreshAttemptResult(
            ok=True, job_id=job_id, status="queued", intent=intent
        )

    try:
        return run_transaction(_txn, what=f"fresh_attempt_{intent}")
    except Exception as exc:
        log.warning("fresh-attempt %s failed for %s: %s", intent, job_id, exc)
        return FreshAttemptResult(
            ok=False, job_id=job_id, status="", reason="resume_failed", intent=intent
        )


def settle_queued_cancel_without_attempt(
    *,
    job_id: str,
    now: float | None = None,
) -> FencedLifecycleResult:
    """Cancel a non-running job that has no active attempt (no host kill).

    Used when the instance is still queued / never assigned — no container
    exists, so unfenced kill is unnecessary and unsafe to invent.
    """
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    ts = now if now is not None else time.time()
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        job = conn.execute(
            """
            SELECT job_id, status, active_attempt_id, host_id
              FROM jobs
             WHERE job_id = %s
             FOR UPDATE
            """,
            (job_id,),
        ).fetchone()
        if not job:
            return FencedLifecycleResult(
                ok=False,
                job_id=job_id,
                intent="cancel",
                status="",
                reason="not_found",
            )
        if job.get("active_attempt_id"):
            return FencedLifecycleResult(
                ok=False,
                job_id=job_id,
                intent="cancel",
                status=str(job.get("status") or ""),
                reason="has_active_attempt",
            )
        if job.get("status") in (
            "terminated",
            "completed",
            "failed",
            "preempted",
            "cancelled",
        ):
            return FencedLifecycleResult(
                ok=False,
                job_id=job_id,
                intent="cancel",
                status=str(job["status"]),
                reason="already_terminal",
            )
        conn.execute(
            """
            UPDATE jobs
               SET status = 'cancelled',
                   payload = jsonb_set(
                       jsonb_set(
                           COALESCE(payload, '{}'::jsonb),
                           '{status}',
                           '"cancelled"'::jsonb,
                           true
                       ),
                       '{cancelled_at}',
                       to_jsonb(%s::float),
                       true
                   ),
                   version = version + 1,
                   updated_at = clock_timestamp()
             WHERE job_id = %s
            """,
            (ts, job_id),
        )
        conn.commit()
    return FencedLifecycleResult(
        ok=True,
        job_id=job_id,
        intent="cancel",
        status="cancelled",
    )


def terminal_job_status_for_stopped_report(
    lifecycle_intent: str | None,
) -> str:
    """Map worker ``stopped`` report → jobs.status using recorded intent."""
    if lifecycle_intent == "terminate":
        return "terminated"
    if lifecycle_intent == "cancel":
        return "cancelled"
    if lifecycle_intent == "restart":
        return "queued"
    return "stopped"
