"""Two-stage per-sink projection delivery (Track B B4.4, companion §12.1).

Extends the one outbox authority — it never becomes a second outbox. Fan-out is
two durable stages so a dispatcher crash converges on exactly one logical
delivery per (event, sink):

  Stage 1 — prepare: claim un-prepared `outbox_events` and, for each active sink,
  materialize a `projection_deliveries` row (idempotent by the PK) and stamp
  `fanout_prepared_at`. All in one short transaction. `fanout_prepared_at` means
  *obligations were materialized*, never that any sink succeeded.

  Stage 2 — deliver: claim pending delivery rows (SKIP LOCKED, leased), do the
  external I/O **outside** the transaction, then record success by the sink's
  stable external id. The `(sink, external_id)` unique index turns at-least-once
  I/O into exactly-once logical delivery.

A sink added later starts from an explicit `backfilled_from` bound, so it never
silently replays the whole history.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable

log = logging.getLogger("xcelsior")


class PermanentDeliveryError(RuntimeError):
    """A contract/privacy/source failure that retries cannot repair."""


@dataclass
class DeliverableEvent:
    event_id: str
    event_type: str
    payload: dict[str, Any]
    sink: str
    aggregate_type: str = ""
    aggregate_id: str = ""
    aggregate_version: int = 0
    headers: dict[str, Any] = field(default_factory=dict)
    event_version: int = 1
    tenant_id: str | None = None
    occurred_at: _dt.datetime | None = None
    classification: str | None = None
    payload_sha256: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    trace_id: str | None = None
    created_at: _dt.datetime | None = None


def ensure_sink(
    conn: Any,
    sink: str,
    *,
    backfill_from: _dt.datetime | None = None,
    active: bool = True,
) -> None:
    """Register (or reactivate) a sink. A late-added sink must pass an explicit
    `backfill_from`; without one it only sees events created after registration.

    The original implementation stored NULL for a new sink and interpreted NULL
    as "no lower bound". A newly enabled sink could therefore absorb an arbitrary
    unprepared backlog, contradicting the explicit-backfill contract. Persisting
    `clock_timestamp()` as the default boundary makes the safe behavior real.
    """
    conn.execute(
        """
        INSERT INTO projection_checkpoints (sink, active, backfilled_from, last_prepared_at)
        VALUES (%s, %s, COALESCE(%s, clock_timestamp()), clock_timestamp())
        ON CONFLICT (sink) DO UPDATE
           SET active = EXCLUDED.active,
               backfilled_from = CASE
                   WHEN %s::timestamptz IS NOT NULL THEN %s::timestamptz
                   ELSE COALESCE(
                       projection_checkpoints.backfilled_from,
                       clock_timestamp()
                   )
               END,
               updated_at = clock_timestamp()
        """,
        (sink, active, backfill_from, backfill_from, backfill_from),
    )


def active_sinks(conn: Any) -> list[tuple[str, _dt.datetime | None]]:
    rows = conn.execute(
        "SELECT sink, backfilled_from FROM projection_checkpoints WHERE active ORDER BY sink"
    ).fetchall()
    return [(str(r[0]), r[1]) for r in rows]


def prepare_fanout(conn: Any, *, limit: int = 200, only_event_ids: list[str] | None = None) -> int:
    """Stage 1: materialize per-sink delivery obligations for un-prepared events.

    Idempotent: the `projection_deliveries` PK collapses a re-prepared event, and
    only events at/after a sink's `backfilled_from` create a row for it. Returns
    the number of outbox events prepared. `only_event_ids` restricts the pass to
    named events (targeted prepare / repair).
    """
    sinks = active_sinks(conn)
    if not sinks:
        return 0
    if only_event_ids is None:
        events = conn.execute(
            """
            SELECT event_id, created_at
              FROM outbox_events
             WHERE fanout_prepared_at IS NULL
             ORDER BY created_at
               FOR UPDATE SKIP LOCKED
             LIMIT %s
            """,
            (limit,),
        ).fetchall()
    else:
        events = conn.execute(
            """
            SELECT event_id, created_at
              FROM outbox_events
             WHERE fanout_prepared_at IS NULL
               AND event_id = ANY(%s)
             ORDER BY created_at
               FOR UPDATE SKIP LOCKED
             LIMIT %s
            """,
            ([str(event_id) for event_id in only_event_ids], limit),
        ).fetchall()
    prepared = 0
    for event_id, created_at in events:
        for sink, backfilled_from in sinks:
            if (
                backfilled_from is not None
                and created_at is not None
                and created_at < backfilled_from
            ):
                continue  # before this sink existed — not its obligation
            conn.execute(
                """
                INSERT INTO projection_deliveries (event_id, sink)
                VALUES (%s, %s)
                ON CONFLICT (event_id, sink) DO NOTHING
                """,
                (str(event_id), sink),
            )
        conn.execute(
            "UPDATE outbox_events SET fanout_prepared_at = clock_timestamp(), "
            "fanout_attempts = fanout_attempts + 1 WHERE event_id = %s",
            (str(event_id),),
        )
        prepared += 1
    if prepared:
        conn.execute("""
            UPDATE projection_checkpoints
               SET last_prepared_at = clock_timestamp(),
                   updated_at = clock_timestamp()
             WHERE active
            """)
    return prepared


def backfill_sink(conn: Any, sink: str, *, frm: _dt.datetime, to: _dt.datetime) -> int:
    """Materialize delivery rows for a **late-added** sink over an explicit range.

    `prepare_fanout` only ever fans out *new* (un-prepared) outbox events, so a
    sink added after the fact receives nothing historical by default. Getting it
    older events is a deliberate act with a stated `[frm, to)` bound — never an
    accidental full-history replay. Idempotent. Returns rows created.
    """
    if frm >= to:
        raise ValueError("projection backfill requires frm < to")
    registered = conn.execute(
        "SELECT 1 FROM projection_checkpoints WHERE sink = %s", (sink,)
    ).fetchone()
    if registered is None:
        raise ValueError(f"projection sink {sink!r} is not registered")
    rows = conn.execute(
        """
        INSERT INTO projection_deliveries (event_id, sink)
        SELECT o.event_id, %s
          FROM outbox_events o
         WHERE o.created_at >= %s AND o.created_at < %s
        ON CONFLICT (event_id, sink) DO NOTHING
        RETURNING event_id
        """,
        (sink, frm, to),
    ).fetchall()
    return len(rows)


def _claim_batch(
    conn: Any,
    sink: str,
    *,
    limit: int,
    owner: str,
    lease_sec: int,
) -> list[DeliverableEvent]:
    rows = conn.execute(
        """
        WITH claimed AS (
            SELECT d.event_id
              FROM projection_deliveries d
             WHERE d.sink = %(sink)s
               AND d.status = 'pending'
               AND d.available_at <= clock_timestamp()
               AND (d.claim_owner IS NULL OR d.claim_expires_at < clock_timestamp())
             ORDER BY d.available_at
               FOR UPDATE SKIP LOCKED
             LIMIT %(limit)s
        )
        UPDATE projection_deliveries d
           SET claim_owner = %(owner)s,
               claim_expires_at = clock_timestamp() + make_interval(secs => %(lease)s)
          FROM claimed
         WHERE d.event_id = claimed.event_id AND d.sink = %(sink)s
        RETURNING d.event_id
        """,
        {"sink": sink, "limit": limit, "owner": owner, "lease": lease_sec},
    ).fetchall()
    if not rows:
        return []
    ids = [str(r[0]) for r in rows]
    evs = conn.execute(
        """
        SELECT event_id, event_type, payload, aggregate_type, aggregate_id,
               aggregate_version, headers, event_version, tenant_id, occurred_at,
               classification, payload_sha256, correlation_id, causation_id,
               trace_id, created_at
          FROM outbox_events
         WHERE event_id = ANY(%s)
        """,
        (ids,),
    ).fetchall()
    by_id = {str(e[0]): e for e in evs}
    missing = [eid for eid in ids if eid not in by_id]
    if missing:
        conn.execute(
            """
            UPDATE projection_deliveries
               SET status = 'dead_lettered',
                   attempt_count = attempt_count + 1,
                   last_error = 'source outbox event is missing',
                   claim_owner = NULL,
                   claim_expires_at = NULL
             WHERE event_id = ANY(%s)
               AND sink = %s
               AND status = 'pending'
               AND claim_owner = %s
            """,
            (missing, sink, owner),
        )
        log.error(
            "projection sink=%s dead-lettered %d orphaned obligation(s)",
            sink,
            len(missing),
        )
    out: list[DeliverableEvent] = []
    for eid in ids:
        e = by_id.get(eid)
        if e is not None:
            out.append(
                DeliverableEvent(
                    event_id=eid,
                    event_type=str(e[1]),
                    payload=e[2] if isinstance(e[2], dict) else {},
                    sink=sink,
                    aggregate_type=str(e[3]),
                    aggregate_id=str(e[4]),
                    aggregate_version=int(e[5] or 0),
                    headers=e[6] if isinstance(e[6], dict) else {},
                    event_version=int(e[7] or 1),
                    tenant_id=str(e[8]) if e[8] is not None else None,
                    occurred_at=e[9],
                    classification=str(e[10]) if e[10] is not None else None,
                    payload_sha256=str(e[11]) if e[11] is not None else None,
                    correlation_id=str(e[12]) if e[12] is not None else None,
                    causation_id=str(e[13]) if e[13] is not None else None,
                    trace_id=str(e[14]) if e[14] is not None else None,
                    created_at=e[15],
                )
            )
    return out


def record_delivery(
    conn: Any,
    event_id: str,
    sink: str,
    external_id: str,
    *,
    owner: str | None = None,
) -> bool:
    """Idempotent success record. The (sink, external_id) unique index means a
    replayed I/O with the same external id never double-delivers."""
    result = conn.execute(
        """
        UPDATE projection_deliveries
           SET status = 'delivered',
               delivered_at = clock_timestamp(),
               external_id = %s,
               claim_owner = NULL,
               claim_expires_at = NULL
         WHERE event_id = %s
           AND sink = %s
           AND status = 'pending'
           AND (%s::text IS NULL OR claim_owner = %s)
        """,
        (external_id, event_id, sink, owner, owner),
    )
    return result.rowcount == 1


def record_failure(
    conn: Any,
    event_id: str,
    sink: str,
    error: str,
    *,
    backoff_sec: float | None = None,
    permanent: bool = False,
    owner: str | None = None,
) -> str:
    """Settle one failure with bounded full-jitter backoff or dead-letter it."""
    row = conn.execute(
        """
        UPDATE projection_deliveries
           SET attempt_count = attempt_count + 1,
               last_error = left(%s, 500),
               available_at = clock_timestamp() + make_interval(
                   secs => CASE
                       WHEN %s::double precision IS NOT NULL
                           THEN GREATEST(%s::double precision, 0)
                       ELSE random() * LEAST(
                           600.0,
                           power(2.0, GREATEST(attempt_count, 0))
                       )
                   END
               ),
               claim_owner = NULL,
               claim_expires_at = NULL,
               status = CASE
                   WHEN %s OR attempt_count + 1 >= max_attempts
                       THEN 'dead_lettered'
                   ELSE 'pending'
               END
         WHERE event_id = %s
           AND sink = %s
           AND status = 'pending'
           AND (%s::text IS NULL OR claim_owner = %s)
        RETURNING status
        """,
        (
            error,
            backoff_sec,
            backoff_sec,
            permanent,
            event_id,
            sink,
            owner,
            owner,
        ),
    ).fetchone()
    if row is None:
        return "stale_claim"
    return str(row[0])


def deliver_pending(
    pool: Any,
    sink: str,
    deliver_fn: Callable[[DeliverableEvent], str],
    *,
    limit: int = 100,
    owner: str = "dispatcher",
    lease_sec: int = 60,
) -> int:
    """Stage 2: claim → external I/O outside the txn → record by external id.

    `deliver_fn(event)` performs the I/O and returns the sink's stable external
    id. Returns the number of deliveries recorded this pass.
    """
    with pool.connection() as conn:
        batch = _claim_batch(conn, sink, limit=limit, owner=owner, lease_sec=lease_sec)
        conn.commit()
    if not batch:
        return 0

    results: list[tuple[str, str | None, str | None, bool]] = []
    for ev in batch:
        try:
            external_id = deliver_fn(ev)  # external I/O — OUTSIDE any transaction
            if not external_id:
                raise PermanentDeliveryError("sink returned an empty external id")
            results.append((ev.event_id, str(external_id), None, False))
        except PermanentDeliveryError as exc:
            results.append((ev.event_id, None, str(exc), True))
        except Exception as exc:  # contained per-delivery
            results.append((ev.event_id, None, str(exc), False))

    delivered = 0
    with pool.connection() as conn:
        for event_id, external_id, err, permanent in results:
            conn.execute("SAVEPOINT projection_delivery_settle")
            try:
                if err is None and external_id is not None:
                    if record_delivery(
                        conn,
                        event_id,
                        sink,
                        external_id,
                        owner=owner,
                    ):
                        delivered += 1
                else:
                    outcome = record_failure(
                        conn,
                        event_id,
                        sink,
                        err or "unknown",
                        permanent=permanent,
                        owner=owner,
                    )
                    log.warning(
                        "projection delivery failed sink=%s event_id=%s outcome=%s error=%s",
                        sink,
                        event_id,
                        outcome,
                        err or "unknown",
                    )
                conn.execute("RELEASE SAVEPOINT projection_delivery_settle")
            except Exception as exc:
                conn.execute("ROLLBACK TO SAVEPOINT projection_delivery_settle")
                outcome = record_failure(
                    conn,
                    event_id,
                    sink,
                    f"settlement failed: {exc}",
                    permanent=True,
                    owner=owner,
                )
                conn.execute("RELEASE SAVEPOINT projection_delivery_settle")
                log.exception(
                    "projection settlement failed sink=%s event_id=%s outcome=%s",
                    sink,
                    event_id,
                    outcome,
                )
        conn.commit()
    return delivered


def retry_dead_letters(
    conn: Any,
    *,
    sink: str,
    event_id: str | None = None,
    limit: int = 100,
) -> int:
    """Explicit operator primitive: requeue named/bounded dead letters."""
    result = conn.execute(
        """
        WITH retryable AS (
            SELECT event_id
              FROM projection_deliveries
             WHERE sink = %(sink)s
               AND status = 'dead_lettered'
               AND (%(event_id)s::uuid IS NULL OR event_id = %(event_id)s::uuid)
             ORDER BY prepared_at
             LIMIT %(limit)s
               FOR UPDATE SKIP LOCKED
        )
        UPDATE projection_deliveries d
           SET status = 'pending',
               attempt_count = 0,
               available_at = clock_timestamp(),
               last_error = NULL,
               claim_owner = NULL,
               claim_expires_at = NULL
          FROM retryable
         WHERE d.event_id = retryable.event_id
           AND d.sink = %(sink)s
        """,
        {"sink": sink, "event_id": event_id, "limit": limit},
    )
    return result.rowcount


def prune_delivered(
    conn: Any,
    *,
    retention_days: int = 30,
    limit: int = 1000,
) -> int:
    """Prune only successful delivery receipts; dead-letter evidence is retained."""
    result = conn.execute(
        """
        DELETE FROM projection_deliveries
         WHERE (event_id, sink) IN (
             SELECT event_id, sink
               FROM projection_deliveries
              WHERE status = 'delivered'
                AND delivered_at < clock_timestamp()
                    - make_interval(days => %(days)s)
              ORDER BY delivered_at
              LIMIT %(limit)s
         )
        """,
        {"days": max(retention_days, 1), "limit": max(limit, 1)},
    )
    return result.rowcount


def health_snapshot(conn: Any) -> dict[str, Any]:
    """Database-derived projection health for APIs, metrics, and operators."""
    rows = conn.execute("""
        SELECT c.sink,
               c.active,
               c.last_prepared_at,
               count(d.*) FILTER (WHERE d.status = 'pending') AS pending,
               count(d.*) FILTER (WHERE d.status = 'delivered') AS delivered,
               count(d.*) FILTER (WHERE d.status = 'dead_lettered') AS dead_lettered,
               COALESCE(
                   max(EXTRACT(EPOCH FROM (clock_timestamp() - d.prepared_at)))
                       FILTER (WHERE d.status = 'pending'),
                   0
               ) AS oldest_pending_seconds,
               COALESCE(
                   percentile_cont(0.95) WITHIN GROUP (
                       ORDER BY EXTRACT(EPOCH FROM (d.delivered_at - d.prepared_at))
                   ) FILTER (WHERE d.status = 'delivered'),
                   0
               ) AS delivery_latency_p95_seconds
          FROM projection_checkpoints c
     LEFT JOIN projection_deliveries d ON d.sink = c.sink
      GROUP BY c.sink, c.active, c.last_prepared_at
      ORDER BY c.sink
        """).fetchall()
    sinks = [
        {
            "sink": str(row[0]),
            "active": bool(row[1]),
            "last_prepared_at": row[2],
            "pending": int(row[3] or 0),
            "delivered": int(row[4] or 0),
            "dead_lettered": int(row[5] or 0),
            "oldest_pending_seconds": float(row[6] or 0),
            "delivery_latency_p95_seconds": float(row[7] or 0),
        }
        for row in rows
    ]
    unprepared = conn.execute(
        "SELECT count(*) FROM outbox_events WHERE fanout_prepared_at IS NULL"
    ).fetchone()[0]
    orphaned = conn.execute("""
        SELECT count(*)
          FROM projection_deliveries d
     LEFT JOIN outbox_events o ON o.event_id = d.event_id
         WHERE o.event_id IS NULL
           AND d.status <> 'delivered'
        """).fetchone()[0]
    return {
        "unprepared": int(unprepared or 0),
        "orphaned": int(orphaned or 0),
        "sinks": sinks,
    }
