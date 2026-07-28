"""Production runtime for Track B's two-stage projection delivery.

The Track A outbox dispatcher and this runtime have different jobs:

* the outbox dispatcher settles the original transactional side-effect intent
  (`default` SSE notification, worker wake-up, and similar low-latency effects);
* the projection runtime materializes and settles independent per-sink
  obligations from the same outbox row.

There is one outbox authority. This module does not create or republish events.
At the current Track B stage only ``audit_log`` is active. SSE remains on the
existing outbox/NOTIFY path, and the warehouse does not activate until B11
provisions its governed landing and BigQuery implementation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import socket
import uuid
from typing import Any

from psycopg.types.json import Jsonb

from analytics.contracts import FORBIDDEN_IN_EVENTS, register_all
from control_plane.db import stable_advisory_key
from control_plane.projection_delivery import (
    DeliverableEvent,
    PermanentDeliveryError,
    deliver_pending,
    ensure_sink,
    prepare_fanout,
    prune_delivered,
)

log = logging.getLogger("xcelsior.control_plane.projection_runtime")

AUDIT_SINK = "audit_log"
_AUDIT_BACKFILL_START = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
_FORBIDDEN_KEY_PARTS = (
    "authorization",
    "credential",
    "env",
    "environment",
    "init_script",
    "password",
    "private_key",
    "registry_password",
    "secret",
    "signed_url",
    "token",
)


def _positive_int(name: str, default: int, *, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


def projection_schema_ready(conn: Any) -> bool:
    row = conn.execute("""
        SELECT to_regclass('event_contracts'),
               to_regclass('projection_deliveries'),
               to_regclass('projection_checkpoints'),
               to_regclass('audit_events_v2')
        """).fetchone()
    return bool(row and all(value is not None for value in row))


def bootstrap_projection_runtime(conn: Any) -> bool:
    """Idempotently register contracts and the sinks valid at today's stage."""
    if not projection_schema_ready(conn):
        return False
    register_all(conn)
    # This explicit range is intentional. audit_events_v2 is the durable audit
    # projection and should absorb every still-hot outbox event after rollout.
    # Future sinks must choose their own explicit backfill range.
    ensure_sink(conn, AUDIT_SINK, backfill_from=_AUDIT_BACKFILL_START)
    return True


def prepare_projection_fanout_task() -> int:
    """Bounded backlog-draining scheduled task for stage-one fan-out."""
    from db import _get_pg_pool

    batch_size = _positive_int("XCELSIOR_PROJECTION_BATCH_SIZE", 200, maximum=2_000)
    max_batches = _positive_int("XCELSIOR_PROJECTION_MAX_BATCHES_PER_RUN", 10, maximum=100)
    prepared = 0
    pool = _get_pg_pool()
    for _ in range(max_batches):
        with pool.connection() as conn:
            if not bootstrap_projection_runtime(conn):
                conn.rollback()
                log.warning("projection schema is not ready; fan-out task skipped")
                return prepared
            count = prepare_fanout(conn, limit=batch_size)
            conn.commit()
        prepared += count
        if count < batch_size:
            break
    if prepared:
        log.info("projection fan-out prepared %d outbox event(s)", prepared)
    return prepared


def _redact_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        named_secret = any(
            key in value
            and any(
                part in str(value[key]).lower()
                for part in _FORBIDDEN_KEY_PARTS
            )
            for key in ("key", "name", "variable")
        )
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_KEY_PARTS) or (
                named_secret and lowered in {"content", "value"}
            ):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_payload(item)
        return redacted
    return value


def _header(event: DeliverableEvent, *names: str) -> str | None:
    for name in names:
        value = event.headers.get(name)
        if value is not None and str(value):
            return str(value)
    return None


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _audit_event_hash(
    event: DeliverableEvent,
    *,
    stream_sequence: int,
    classification: str,
    payload: dict[str, Any],
    prev_hash: str | None,
    created_at: dt.datetime,
) -> str:
    return _canonical_hash(
        {
            "event_id": event.event_id,
            "tenant_id": event.tenant_id
            or _header(event, "tenant_id", "workspace_id", "customer_id"),
            "stream_type": event.aggregate_type,
            "stream_id": event.aggregate_id,
            "stream_sequence": stream_sequence,
            "aggregate_version": event.aggregate_version,
            "event_type": event.event_type,
            "actor_id": _header(event, "actor_id", "principal_id"),
            "client_id": _header(event, "client_id"),
            "request_id": _header(event, "request_id"),
            "trace_id": event.trace_id or _header(event, "trace_id"),
            "classification": classification,
            "payload": payload,
            "prev_hash": prev_hash,
            "created_at": created_at.isoformat(),
        }
    )


def deliver_audit_event(event: DeliverableEvent) -> str:
    """Idempotently append one outbox event to the WORM audit stream."""
    if event.sink != AUDIT_SINK:
        raise PermanentDeliveryError(f"unsupported audit sink {event.sink!r}")
    if not event.aggregate_type or not event.aggregate_id or event.created_at is None:
        raise PermanentDeliveryError("outbox source lacks audit stream identity")

    from db import _get_pg_pool

    pool = _get_pg_pool()
    external_id = f"audit:{event.event_id}"
    with pool.connection() as conn:
        # Retry after "audit INSERT committed, delivery receipt did not" returns
        # the same stable external id without creating a second stream sequence.
        existing = conn.execute(
            "SELECT 1 FROM audit_events_v2 WHERE event_id = %s LIMIT 1",
            (event.event_id,),
        ).fetchone()
        if existing is not None:
            conn.rollback()
            return external_id

        contract = conn.execute(
            """
            SELECT classification
              FROM event_contracts
             WHERE event_type = %s
               AND version = %s
               AND active
            """,
            (event.event_type, event.event_version),
        ).fetchone()
        if contract is None:
            conn.rollback()
            raise PermanentDeliveryError(
                f"no active contract for {event.event_type} v{event.event_version}"
            )
        classification = str(contract[0])
        if classification in FORBIDDEN_IN_EVENTS:
            conn.rollback()
            raise PermanentDeliveryError(
                f"{event.event_type} is forbidden from durable event sinks"
            )

        stream_id = f"{event.aggregate_type}\x1f{event.aggregate_id}"
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (stable_advisory_key("audit_events_v2_stream", stream_id),),
        )
        # Recheck under the stream lock: another replica may have won before us.
        existing = conn.execute(
            "SELECT 1 FROM audit_events_v2 WHERE event_id = %s LIMIT 1",
            (event.event_id,),
        ).fetchone()
        if existing is not None:
            conn.rollback()
            return external_id
        head = conn.execute(
            """
            SELECT stream_sequence, event_hash
              FROM audit_events_v2
             WHERE stream_id = %s
             ORDER BY stream_sequence DESC, created_at DESC
             LIMIT 1
            """,
            (stream_id,),
        ).fetchone()
        stream_sequence = int(head[0]) + 1 if head else 1
        prev_hash = str(head[1]) if head and head[1] else None
        payload = _redact_payload(event.payload)
        if not isinstance(payload, dict):  # defensive; DeliverableEvent says dict
            payload = {}
        payload["_xcelsior_projection"] = {
            "outbox_created_at": event.created_at.isoformat(),
            "source_occurred_at": (
                event.occurred_at.isoformat()
                if event.occurred_at is not None
                else event.created_at.isoformat()
            ),
        }
        # Checkpoints seal append intervals. Backfilling a week-old outbox row
        # into its original occurrence interval would mutate a previously sealed
        # interval and make an honest checkpoint fail verification. Preserve
        # source time in the payload metadata and timestamp the immutable audit
        # row at projection append time.
        created_at = conn.execute("SELECT clock_timestamp()").fetchone()[0]
        event_hash = _audit_event_hash(
            event,
            stream_sequence=stream_sequence,
            classification=classification,
            payload=payload,
            prev_hash=prev_hash,
            created_at=created_at,
        )
        conn.execute(
            """
            INSERT INTO audit_events_v2 (
                event_id, tenant_id, stream_type, stream_id, stream_sequence,
                aggregate_version, event_type, actor_id, client_id, request_id,
                trace_id, classification, payload, prev_hash, event_hash,
                created_at
            ) VALUES (
                %(event_id)s, %(tenant_id)s, %(stream_type)s, %(stream_id)s,
                %(stream_sequence)s, %(aggregate_version)s, %(event_type)s,
                %(actor_id)s, %(client_id)s, %(request_id)s, %(trace_id)s,
                %(classification)s, %(payload)s, %(prev_hash)s, %(event_hash)s,
                %(created_at)s
            )
            ON CONFLICT (event_id, created_at) DO NOTHING
            """,
            {
                "event_id": event.event_id,
                "tenant_id": event.tenant_id
                or _header(event, "tenant_id", "workspace_id", "customer_id"),
                "stream_type": event.aggregate_type,
                "stream_id": stream_id,
                "stream_sequence": stream_sequence,
                "aggregate_version": event.aggregate_version,
                "event_type": event.event_type,
                "actor_id": _header(event, "actor_id", "principal_id"),
                "client_id": _header(event, "client_id"),
                "request_id": _header(event, "request_id"),
                "trace_id": event.trace_id or _header(event, "trace_id"),
                "classification": classification,
                "payload": Jsonb(payload),
                "prev_hash": prev_hash,
                "event_hash": event_hash,
                "created_at": created_at,
            },
        )
        conn.commit()
    return external_id


def deliver_audit_log_task() -> int:
    """Bounded stage-two audit delivery task with replica-safe claims."""
    from db import _get_pg_pool

    batch_size = _positive_int("XCELSIOR_PROJECTION_BATCH_SIZE", 200, maximum=2_000)
    max_batches = _positive_int("XCELSIOR_PROJECTION_MAX_BATCHES_PER_RUN", 10, maximum=100)
    lease_sec = _positive_int("XCELSIOR_PROJECTION_LEASE_SEC", 60, maximum=3_600)
    pool = _get_pg_pool()
    with pool.connection() as conn:
        if not bootstrap_projection_runtime(conn):
            conn.rollback()
            log.warning("projection schema is not ready; audit delivery task skipped")
            return 0
        conn.commit()

    delivered = 0
    # PID alone is not unique across containers (each replica is commonly PID
    # 1), so it cannot safely fence a stale claim from another replica.
    owner = (
        f"audit-log-{socket.gethostname()}-{os.getpid()}-"
        f"{uuid.uuid4().hex[:12]}"
    )
    for _ in range(max_batches):
        count = deliver_pending(
            pool,
            AUDIT_SINK,
            deliver_audit_event,
            limit=batch_size,
            owner=owner,
            lease_sec=lease_sec,
        )
        delivered += count
        if count < batch_size:
            break
    if delivered:
        log.info("audit projection delivered %d event(s)", delivered)
    return delivered


def projection_retention_task() -> int:
    """Apply the documented receipt-retention policy without hiding failures."""
    from db import _get_pg_pool

    days = _positive_int(
        "XCELSIOR_PROJECTION_DELIVERY_RETENTION_DAYS",
        30,
        maximum=3_650,
    )
    batch = _positive_int(
        "XCELSIOR_PROJECTION_RETENTION_BATCH_SIZE",
        1_000,
        maximum=100_000,
    )
    with _get_pg_pool().connection() as conn:
        if not projection_schema_ready(conn):
            conn.rollback()
            return 0
        pruned = prune_delivered(conn, retention_days=days, limit=batch)
        conn.commit()
    if pruned:
        log.info("projection retention pruned %d delivered receipt(s)", pruned)
    return pruned
