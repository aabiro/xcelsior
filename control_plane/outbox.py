"""Transactional outbox writer and dispatcher (blueprint §16.1, ADR-006).

Writers call :func:`append_event` inside the same transaction as the
state mutation that implies the side effect — commit makes both durable
or neither. Dispatcher replicas then claim batches with ``SKIP LOCKED``,
deliver outside any database transaction, and mark published. Delivery is
at-least-once by construction: a dispatcher crash between delivery and
``mark_published`` redelivers after the claim expires, so every consumer
effect must be idempotent (keyed by ``idempotency_key``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from psycopg import Connection
from psycopg.types.json import Jsonb

from control_plane.db import run_transaction

log = logging.getLogger("xcelsior.control_plane.outbox")

DEFAULT_CLAIM_TTL_SEC = 60
DEFAULT_RETRY_BASE_BACKOFF_SEC = 2.0
DEFAULT_RETRY_MAX_BACKOFF_SEC = 600.0


@dataclass(frozen=True)
class OutboxEvent:
    event_id: str
    aggregate_type: str
    aggregate_id: str
    event_type: str
    payload: dict[str, Any]
    headers: dict[str, Any]
    destination_class: str
    idempotency_key: str
    attempt_count: int


def _get(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


def append_event(
    conn: Connection,
    *,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, Any] | None = None,
    destination_class: str = "default",
    idempotency_key: str | None = None,
    aggregate_version: int = 0,
) -> str | None:
    """Append one side-effect intent in the caller's open transaction.

    Returns the new event id, or None when the (destination, idempotency
    key) pair already exists — the intent is already durable, appending
    again must not duplicate it.
    """
    key = idempotency_key or f"{event_type}:{aggregate_id}:{aggregate_version}"
    row = conn.execute(
        """
        INSERT INTO outbox_events
            (aggregate_type, aggregate_id, aggregate_version, event_type,
             payload, headers, destination_class, idempotency_key)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (destination_class, idempotency_key) DO NOTHING
        RETURNING event_id
        """,
        (
            aggregate_type, aggregate_id, aggregate_version, event_type,
            Jsonb(payload or {}), Jsonb(headers or {}), destination_class, key,
        ),
    ).fetchone()
    return str(_get(row, "event_id", 0)) if row is not None else None


def claim_batch(
    conn: Connection,
    *,
    dispatcher_id: str,
    destination_class: str | None = None,
    limit: int = 50,
    claim_ttl_sec: int = DEFAULT_CLAIM_TTL_SEC,
) -> list[OutboxEvent]:
    """Claim due, unpublished, non-dead-lettered events for delivery."""
    rows = conn.execute(
        """
        WITH due AS (
            SELECT event_id
              FROM outbox_events
             WHERE published_at IS NULL
               AND dead_lettered_at IS NULL
               AND available_at <= clock_timestamp()
               AND (claim_expires_at IS NULL
                    OR claim_expires_at < clock_timestamp())
               AND (%(dest)s::text IS NULL
                    OR destination_class = %(dest)s)
             ORDER BY available_at
             LIMIT %(limit)s
               FOR UPDATE SKIP LOCKED
        )
        UPDATE outbox_events o
           SET claim_owner = %(owner)s,
               claim_expires_at = clock_timestamp()
                                  + make_interval(secs => %(ttl)s),
               attempt_count = o.attempt_count + 1
          FROM due
         WHERE o.event_id = due.event_id
        RETURNING o.event_id, o.aggregate_type, o.aggregate_id, o.event_type,
                  o.payload, o.headers, o.destination_class,
                  o.idempotency_key, o.attempt_count
        """,
        {
            "dest": destination_class,
            "limit": limit,
            "owner": dispatcher_id,
            "ttl": claim_ttl_sec,
        },
    ).fetchall()
    return [
        OutboxEvent(
            event_id=str(_get(r, "event_id", 0)),
            aggregate_type=str(_get(r, "aggregate_type", 1)),
            aggregate_id=str(_get(r, "aggregate_id", 2)),
            event_type=str(_get(r, "event_type", 3)),
            payload=_get(r, "payload", 4) or {},
            headers=_get(r, "headers", 5) or {},
            destination_class=str(_get(r, "destination_class", 6)),
            idempotency_key=str(_get(r, "idempotency_key", 7)),
            attempt_count=int(_get(r, "attempt_count", 8)),
        )
        for r in rows
    ]


def mark_published(conn: Connection, event_ids: list[str]) -> int:
    if not event_ids:
        return 0
    result = conn.execute(
        """
        UPDATE outbox_events
           SET published_at = clock_timestamp(),
               claim_owner = NULL,
               claim_expires_at = NULL,
               last_error = NULL
         WHERE event_id = ANY(%s)
           AND published_at IS NULL
        """,
        (event_ids,),
    )
    return result.rowcount


def mark_failed(
    conn: Connection,
    event_id: str,
    error: str,
    *,
    base_backoff_sec: float = DEFAULT_RETRY_BASE_BACKOFF_SEC,
    max_backoff_sec: float = DEFAULT_RETRY_MAX_BACKOFF_SEC,
) -> str:
    """Record a delivery failure: bounded backoff, then dead-letter."""
    row = conn.execute(
        """
        UPDATE outbox_events
           SET last_error = left(%(error)s, 2000),
               claim_owner = NULL,
               claim_expires_at = NULL,
               dead_lettered_at = CASE
                   WHEN attempt_count >= max_attempts THEN clock_timestamp()
                   ELSE NULL
               END,
               available_at = clock_timestamp() + make_interval(
                   secs => LEAST(
                       %(max_backoff)s,
                       %(base_backoff)s * power(2, GREATEST(attempt_count - 1, 0))
                   )
               )
         WHERE event_id = %(event_id)s
           AND published_at IS NULL
        RETURNING dead_lettered_at IS NOT NULL AS dead
        """,
        {
            "error": error,
            "event_id": event_id,
            "base_backoff": base_backoff_sec,
            "max_backoff": max_backoff_sec,
        },
    ).fetchone()
    if row is None:
        return "already_published"
    return "dead_letter" if _get(row, "dead", 0) else "retry_scheduled"


class OutboxDispatcher:
    """One dispatcher replica: claim → deliver → settle, forever safe.

    Handlers are keyed by destination class and must be idempotent — a
    crash after delivery but before settlement redelivers the event.
    Handler exceptions never abort the batch; each event settles
    individually (§6.1 outbox dispatcher responsibilities).
    """

    def __init__(
        self,
        dispatcher_id: str,
        handlers: dict[str, Callable[[OutboxEvent], None]],
        *,
        batch_size: int = 50,
        claim_ttl_sec: int = DEFAULT_CLAIM_TTL_SEC,
    ):
        self.dispatcher_id = dispatcher_id
        self.handlers = dict(handlers)
        self.batch_size = batch_size
        self.claim_ttl_sec = claim_ttl_sec

    def run_once(self) -> dict[str, int]:
        """Claim and deliver one batch. Returns delivery stats."""
        stats = {"claimed": 0, "published": 0, "failed": 0, "unroutable": 0}
        events = run_transaction(
            lambda conn: claim_batch(
                conn,
                dispatcher_id=self.dispatcher_id,
                limit=self.batch_size,
                claim_ttl_sec=self.claim_ttl_sec,
            ),
            what="outbox_claim",
        )
        stats["claimed"] = len(events)
        delivered: list[str] = []
        for event in events:
            handler = self.handlers.get(event.destination_class)
            if handler is None:
                # No handler for this destination in this replica: leave
                # the claim to expire so a capable replica picks it up,
                # but record the miss — a permanently unroutable class is
                # an operations bug, not silent noise.
                stats["unroutable"] += 1
                log.warning(
                    "outbox %s: no handler for destination %s",
                    event.event_id,
                    event.destination_class,
                )
                continue
            try:
                handler(event)
            except Exception as exc:
                stats["failed"] += 1
                outcome = run_transaction(
                    lambda conn, _eid=event.event_id, _err=repr(exc): mark_failed(
                        conn, _eid, _err
                    ),
                    what="outbox_mark_failed",
                )
                log.warning(
                    "outbox %s delivery failed (%s): %s",
                    event.event_id,
                    outcome,
                    exc,
                )
            else:
                delivered.append(event.event_id)
        if delivered:
            stats["published"] = run_transaction(
                lambda conn: mark_published(conn, delivered),
                what="outbox_mark_published",
            )
        return stats
