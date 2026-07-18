"""Fenced lifecycle controller for attempt-owned (v2) jobs — P5.5.

Terminate and cancel of attempt-owned work must not use the legacy unfenced
kill/detach path. This module records stop/terminate/cancel intent on the
job row and enqueues a durable ``stop_attempt`` bound to the current
host/attempt/fence. Volume detach stays deferred until observation/ACK
settles the attempt (or a later residual step); never pre-ACK.
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

LifecycleIntent = Literal["stop", "cancel", "terminate"]

_VALID_INTENTS = frozenset({"stop", "cancel", "terminate"})


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

    ``preserve=False`` so the worker stops and removes the attempt container.
    Idempotent on the same attempt via command idempotency key
    ``{intent}:{attempt_id}``.
    """
    # User stop (preserve=True) stays on BillingEngine.stop_instance.
    if intent not in ("cancel", "terminate"):
        return FencedLifecycleResult(
            ok=False,
            job_id=job_id,
            intent=str(intent),
            status="",
            reason="invalid_intent",
        )

    from db import _get_pg_pool
    from psycopg.rows import dict_row

    now = time.time()
    tag = reason_tag or ("user_terminated" if intent == "terminate" else "user_cancelled")
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
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
        if not job:
            return FencedLifecycleResult(
                ok=False,
                job_id=job_id,
                intent=intent,
                status="",
                reason="already_terminal_or_not_found",
            )
        if not job.get("active_attempt_id"):
            return FencedLifecycleResult(
                ok=False,
                job_id=job_id,
                intent=intent,
                status=str(job.get("status") or ""),
                reason="not_attempt_owned",
            )

        cname = (
            container_name
            or job.get("container_name")
            or f"xcl-{job_id}"
        )
        # Intermediate projection — never mark terminal before worker proof.
        _mark_stopping_with_intent(
            conn,
            job_id=job_id,
            intent=intent,
            now=now,
            reason_tag=tag,
        )
        conn.commit()

    try:
        enqueued: EnqueuedAttemptCommand = run_transaction(
            lambda c: enqueue_current_attempt_command(
                c,
                job_id=job_id,
                command="stop_attempt",
                args={
                    "container_name": cname,
                    "preserve": False,
                    "intent": intent,
                },
                created_by=created_by,
                # Distinct from preserve-stop's default key stop_attempt:{attempt}.
                idempotency_key=f"{intent}:{job.get('active_attempt_id')}",
            ),
            what=f"enqueue_fenced_{intent}",
        )
    except CommandProtocolError as exc:
        log.warning(
            "fenced %s enqueue refused for %s: %s", intent, job_id, exc
        )
        return FencedLifecycleResult(
            ok=False,
            job_id=job_id,
            intent=intent,
            status="stopping",
            reason="no_active_fenced_authority",
        )
    except Exception as exc:
        log.warning("fenced %s enqueue failed for %s: %s", intent, job_id, exc)
        return FencedLifecycleResult(
            ok=False,
            job_id=job_id,
            intent=intent,
            status="stopping",
            reason="enqueue_failed",
        )

    log.info(
        "fenced %s queued job=%s attempt=%s fence=%s cmd=%s",
        intent,
        job_id,
        enqueued.attempt_id[:8],
        enqueued.fencing_token,
        enqueued.command_id[:8],
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
    return "stopped"
