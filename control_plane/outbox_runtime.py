"""Outbox dispatcher runtime + destination handlers (blueprint Phase 7).

The transactional writers (reservation, lease expiry, fenced attempt
status) already append their side-effect intents to ``outbox_events`` in
the same commit as the state change. This module makes those intents
actually *happen*:

- ``default``    → translate ``job.v1.*`` events into the dashboard's
  SSE vocabulary and publish them on the existing ``xcelsior_events``
  NOTIFY channel. SSE subscriber queues are process-local to each API
  replica, so the dispatcher cannot broadcast directly — every replica
  already runs ``db.start_pg_listen(broadcast_sse)`` (api.py lifespan),
  and the ssh gateway listens on the same channel, so publishing here
  reaches every connected client with zero new plumbing.
- ``agent_wake`` → no-op acknowledgement today (workers poll on
  ``POLL_INTERVAL``); a future push channel plugs in here to cut
  placement-to-start latency.

Delivery is at-least-once (claim/settle with redelivery on crash), so
handlers are idempotent: re-notifying an SSE event is harmless — clients
key on job state, not message count.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from psycopg import Connection

from control_plane.db import run_transaction
from control_plane.outbox import OutboxDispatcher, OutboxEvent

log = logging.getLogger("xcelsior.control_plane.outbox_runtime")

# The channel db.PgEventBus/start_pg_listen already speak; payload shape
# is {"type": <sse event>, "data": {...}, "ts": epoch}.
EVENTS_CHANNEL = "xcelsior_events"

_DISPATCH_INTERVAL_SEC = 3.0
_PRUNE_EVERY_CYCLES = 200
_PRUNE_RETENTION_DAYS = 7
_PRUNE_BATCH = 1000

# Attempt-status vocabulary → the legacy job-status strings the dashboard
# SSE stream has always spoken.
_ATTEMPT_TO_LEGACY = {
    "lease_claimed": "leased",
    "starting": "starting",
    "running": "running",
    "succeeded": "completed",
    "failed": "failed",
}


def sse_payload_for(event: OutboxEvent) -> dict | None:
    """Map one outbox event to its SSE projection, or None when it has no
    user-visible shape (unknown types publish silently — forward
    compatible with producers this build doesn't know yet)."""
    p = event.payload or {}
    if event.event_type == "job.v1.placement_reserved":
        return {
            "type": "job_status",
            "data": {
                "job_id": event.aggregate_id,
                "status": "assigned",
                "host_id": p.get("host_id"),
                "attempt_id": p.get("attempt_id"),
            },
        }
    if event.event_type == "job.v1.attempt_status_changed":
        legacy = _ATTEMPT_TO_LEGACY.get(str(p.get("status")))
        if legacy is None:
            return None
        return {
            "type": "job_status",
            "data": {
                "job_id": event.aggregate_id,
                "status": legacy,
                "host_id": p.get("host_id"),
            },
        }
    if event.event_type == "job.v1.lease_expired":
        return {
            "type": "job_status",
            "data": {
                "job_id": event.aggregate_id,
                "status": "queued",
                "reason": p.get("reason"),
            },
        }
    # Legacy update_job_status path (queued→assigned→running→…).
    if event.event_type == "job.v1.legacy_status_changed":
        status = p.get("status")
        if not status:
            return None
        data = {
            "job_id": event.aggregate_id,
            "status": status,
            "host_id": p.get("host_id"),
        }
        if p.get("previous_status") is not None:
            data["previous_status"] = p.get("previous_status")
        return {"type": "job_status", "data": data}
    # Host / job lifecycle residual producers (multi-replica durable SSE).
    if event.event_type == "host.v1.status_changed":
        data = dict(p)
        data["host_id"] = event.aggregate_id
        return {"type": "host_update", "data": data}
    if event.event_type == "host.v1.removed":
        return {
            "type": "host_removed",
            "data": {"host_id": event.aggregate_id, **dict(p)},
        }
    if event.event_type == "job.v1.submitted":
        data = dict(p)
        data["job_id"] = event.aggregate_id
        return {"type": "job_submitted", "data": data}
    if event.event_type == "job.v1.preempted":
        data = dict(p)
        data["job_id"] = event.aggregate_id
        return {"type": "job_preempted", "data": data}
    # Queue-block diagnostics + spot price refresh residual producers.
    if event.event_type == "job.v1.queue_blocked":
        return {
            "type": "job_error",
            "data": {
                "job_id": event.aggregate_id,
                "error": p.get("error"),
                "message": p.get("message"),
            },
        }
    if event.event_type == "pricing.v1.spot_prices_updated":
        # Match emit_event("spot_prices", prices): data *is* the price map.
        prices = p.get("prices") if isinstance(p.get("prices"), dict) else p
        if not isinstance(prices, dict):
            return None
        return {"type": "spot_prices", "data": prices}
    # Request-path instance lifecycle SSE (multi-replica durable fan-out).
    # Projections preserve the dashboard vocabulary historically spoken by
    # process-local ``broadcast_sse`` on stop/start/restart/terminate/cancel.
    if event.event_type == "job.v1.instance_stopped":
        data = dict(p)
        data["job_id"] = event.aggregate_id
        return {"type": "instance_stopped", "data": data}
    if event.event_type == "job.v1.instance_started":
        data = dict(p)
        data["job_id"] = event.aggregate_id
        return {"type": "instance_started", "data": data}
    if event.event_type == "job.v1.instance_restarted":
        data = dict(p)
        data["job_id"] = event.aggregate_id
        return {"type": "instance_restarted", "data": data}
    if event.event_type == "job.v1.instance_terminated":
        data = dict(p)
        data["job_id"] = event.aggregate_id
        return {"type": "instance_terminated", "data": data}
    if event.event_type == "job.v1.cancelled":
        data = dict(p)
        data["job_id"] = event.aggregate_id
        return {"type": "job_cancelled", "data": data}
    return None


def try_append_lifecycle_outbox(
    conn: Connection,
    *,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict | None = None,
    idempotency_key: str,
    savepoint: str = "lifecycle_sse_outbox",
) -> bool:
    """Append a durable SSE intent under SAVEPOINT; never poison the caller txn.

    Returns True when the outbox row is durable (insert or idempotent conflict
    already present). Returns False on missing schema / append failure so the
    caller can fall back to process-local ``emit_event`` only then.
    """
    from control_plane.outbox import append_event

    try:
        conn.execute(f"SAVEPOINT {savepoint}")
    except Exception as e:
        log.warning(
            "lifecycle outbox SAVEPOINT create failed %s/%s: %s",
            aggregate_type,
            aggregate_id,
            e,
        )
        return False
    try:
        append_event(
            conn,
            aggregate_type=aggregate_type,
            aggregate_id=str(aggregate_id),
            event_type=event_type,
            payload=payload or {},
            destination_class="default",
            idempotency_key=idempotency_key,
        )
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        return True
    except Exception as e:
        try:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        except Exception as rb_err:
            log.error(
                "lifecycle outbox SAVEPOINT rollback failed %s/%s: %s (original: %s)",
                aggregate_type,
                aggregate_id,
                rb_err,
                e,
            )
            raise e from rb_err
        log.warning(
            "lifecycle outbox append failed %s/%s type=%s: %s "
            "(SAVEPOINT restored; process-local emit fallback)",
            aggregate_type,
            aggregate_id,
            event_type,
            e,
        )
        return False


def enqueue_lifecycle_sse_outbox(
    *,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict | None = None,
    idempotency_key: str,
) -> bool:
    """Request-path durable SSE intent in a short transaction (no open conn).

    Used by API routes that historically called process-local
    ``broadcast_sse`` after a successful mutation. When this returns True
    the outbox row is committed and multi-replica delivery is owned by the
    dispatcher (LISTEN → ``broadcast_sse`` on every API replica). Callers
    must skip process-local fan-out when True and fall back only when False
    (schema missing, pool failure, append failure).
    """
    try:
        return bool(
            run_transaction(
                lambda conn: try_append_lifecycle_outbox(
                    conn,
                    aggregate_type=aggregate_type,
                    aggregate_id=str(aggregate_id),
                    event_type=event_type,
                    payload=payload or {},
                    idempotency_key=idempotency_key,
                ),
                what="request_path_sse_outbox",
            )
        )
    except Exception as e:
        log.warning(
            "request-path SSE outbox enqueue failed %s/%s type=%s: %s "
            "(process-local broadcast fallback)",
            aggregate_type,
            aggregate_id,
            event_type,
            e,
        )
        return False


def handle_default(event: OutboxEvent) -> None:
    """Publish the event's SSE projection on the shared NOTIFY channel."""
    message = sse_payload_for(event)
    if message is None:
        log.debug(
            "outbox %s (%s): no SSE projection", event.event_id, event.event_type
        )
        return
    message["ts"] = time.time()
    run_transaction(
        lambda conn: conn.execute(
            "SELECT pg_notify(%s, %s)", (EVENTS_CHANNEL, json.dumps(message))
        ),
        what="outbox_sse_notify",
    )


def handle_agent_wake(event: OutboxEvent) -> None:
    """Workers poll today; the intent is settled by durable state (the
    start command row). A push channel plugs in here later."""
    log.debug(
        "agent_wake %s for %s (poll-based delivery)",
        event.event_id,
        event.aggregate_id,
    )


def default_handlers() -> dict:
    return {"default": handle_default, "agent_wake": handle_agent_wake}


def prune_settled_events(
    conn: Connection,
    *,
    retention_days: int = _PRUNE_RETENTION_DAYS,
    limit: int = _PRUNE_BATCH,
) -> int:
    """Drop published (or stale dead-lettered) events past retention.

    Dead-lettered rows get double the retention so operators have time to
    notice them (they also surface via logs at delivery time).
    """
    result = conn.execute(
        """
        DELETE FROM outbox_events
         WHERE event_id IN (
             SELECT event_id FROM outbox_events
              WHERE (published_at IS NOT NULL
                     AND published_at < clock_timestamp()
                                        - make_interval(days => %(days)s))
                 OR (dead_lettered_at IS NOT NULL
                     AND dead_lettered_at < clock_timestamp()
                                            - make_interval(days => %(days)s * 2))
              LIMIT %(limit)s
         )
        """,
        {"days": retention_days, "limit": limit},
    )
    return result.rowcount


def outbox_schema_ready() -> bool:
    row = run_transaction(
        lambda conn: conn.execute("SELECT to_regclass('outbox_events')").fetchone(),
        what="outbox_schema_check",
    )
    if row is None:
        return False
    value = row[0] if not isinstance(row, dict) else next(iter(row.values()))
    return value is not None


def run_dispatcher_loop(
    *,
    dispatcher_id: str | None = None,
    stop: threading.Event | None = None,
    interval_sec: float = _DISPATCH_INTERVAL_SEC,
) -> None:
    """Blocking dispatch loop: claim → deliver → settle, plus retention.

    A full batch chains immediately (backlog drain); an idle cycle sleeps.
    Failures are contained per cycle — the loop must outlive transient DB
    or handler trouble.
    """
    dispatcher = OutboxDispatcher(
        dispatcher_id or f"outbox-{os.getpid()}", default_handlers()
    )
    stop = stop or threading.Event()
    cycles = 0
    log.info("outbox dispatcher %s starting", dispatcher.dispatcher_id)
    while not stop.is_set():
        try:
            stats = dispatcher.run_once()
            cycles += 1
            if cycles % _PRUNE_EVERY_CYCLES == 0:
                pruned = run_transaction(
                    lambda conn: prune_settled_events(conn), what="outbox_prune"
                )
                if pruned:
                    log.info("outbox retention: pruned %d settled events", pruned)
            if stats["claimed"] >= dispatcher.batch_size:
                continue  # backlog: drain without sleeping
        except Exception:
            log.exception("outbox dispatch cycle failed; continuing")
        stop.wait(interval_sec)


def start_outbox_dispatcher() -> threading.Thread | None:
    """Start the dispatcher thread (scheduler-worker container).

    Gated by ``XCELSIOR_OUTBOX_DISPATCHER`` (default on) and the presence
    of the outbox table; SKIP LOCKED claims make extra replicas safe.
    """
    enabled = (os.environ.get("XCELSIOR_OUTBOX_DISPATCHER") or "true").strip().lower()
    if enabled in ("0", "false", "no", "off"):
        return None
    try:
        if not outbox_schema_ready():
            log.warning("outbox_events missing — dispatcher not starting")
            return None
    except Exception as e:
        log.error("outbox schema check failed (%s) — dispatcher not starting", e)
        return None
    thread = threading.Thread(
        target=run_dispatcher_loop, name="outbox-dispatcher", daemon=True
    )
    thread.start()
    return thread
