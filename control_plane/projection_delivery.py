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
from typing import Any, Callable

log = logging.getLogger("xcelsior")

DEFAULT_SINKS = ("warehouse", "sse", "webhook")


@dataclass
class DeliverableEvent:
    event_id: str
    event_type: str
    payload: dict[str, Any]
    sink: str


def ensure_sink(conn: Any, sink: str, *, backfill_from: _dt.datetime | None = None, active: bool = True) -> None:
    """Register (or reactivate) a sink. A late-added sink must pass an explicit
    `backfill_from`; without one it only ever sees events prepared after now."""
    conn.execute(
        """
        INSERT INTO projection_checkpoints (sink, active, backfilled_from, last_prepared_at)
        VALUES (%s, %s, %s, clock_timestamp())
        ON CONFLICT (sink) DO UPDATE
           SET active = EXCLUDED.active,
               backfilled_from = COALESCE(EXCLUDED.backfilled_from, projection_checkpoints.backfilled_from),
               updated_at = clock_timestamp()
        """,
        (sink, active, backfill_from),
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
    where = "fanout_prepared_at IS NULL"
    params: list[Any] = []
    if only_event_ids is not None:
        where += " AND event_id = ANY(%s)"
        params.append([str(e) for e in only_event_ids])
    params.append(limit)
    events = conn.execute(
        f"""
        SELECT event_id, created_at
          FROM outbox_events
         WHERE {where}
         ORDER BY created_at
           FOR UPDATE SKIP LOCKED
         LIMIT %s
        """,
        tuple(params),
    ).fetchall()
    prepared = 0
    for event_id, created_at in events:
        for sink, backfilled_from in sinks:
            if backfilled_from is not None and created_at is not None and created_at < backfilled_from:
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
    return prepared


def backfill_sink(conn: Any, sink: str, *, frm: _dt.datetime, to: _dt.datetime) -> int:
    """Materialize delivery rows for a **late-added** sink over an explicit range.

    `prepare_fanout` only ever fans out *new* (un-prepared) outbox events, so a
    sink added after the fact receives nothing historical by default. Getting it
    older events is a deliberate act with a stated `[frm, to)` bound — never an
    accidental full-history replay. Idempotent. Returns rows created.
    """
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


def _claim_batch(conn: Any, sink: str, *, limit: int, owner: str, lease_sec: int) -> list[DeliverableEvent]:
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
        "SELECT event_id, event_type, payload FROM outbox_events WHERE event_id = ANY(%s)",
        (ids,),
    ).fetchall()
    by_id = {str(e[0]): e for e in evs}
    out = []
    for eid in ids:
        e = by_id.get(eid)
        if e is not None:
            out.append(DeliverableEvent(eid, str(e[1]), e[2] if isinstance(e[2], dict) else {}, sink))
    return out


def record_delivery(conn: Any, event_id: str, sink: str, external_id: str) -> None:
    """Idempotent success record. The (sink, external_id) unique index means a
    replayed I/O with the same external id never double-delivers."""
    conn.execute(
        """
        UPDATE projection_deliveries
           SET status = 'delivered',
               delivered_at = clock_timestamp(),
               external_id = %s,
               claim_owner = NULL,
               claim_expires_at = NULL
         WHERE event_id = %s AND sink = %s AND status <> 'delivered'
        """,
        (external_id, event_id, sink),
    )


def record_failure(conn: Any, event_id: str, sink: str, error: str, *, backoff_sec: int = 30) -> None:
    conn.execute(
        """
        UPDATE projection_deliveries
           SET attempt_count = attempt_count + 1,
               last_error = left(%s, 500),
               available_at = clock_timestamp() + make_interval(secs => %s),
               claim_owner = NULL,
               claim_expires_at = NULL,
               status = CASE WHEN attempt_count + 1 >= max_attempts THEN 'dead_lettered' ELSE 'pending' END
         WHERE event_id = %s AND sink = %s AND status = 'pending'
        """,
        (error, backoff_sec, event_id, sink),
    )


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

    results: list[tuple[str, str | None, str | None]] = []
    for ev in batch:
        try:
            external_id = deliver_fn(ev)  # external I/O — OUTSIDE any transaction
            results.append((ev.event_id, str(external_id), None))
        except Exception as exc:  # contained per-delivery
            results.append((ev.event_id, None, str(exc)))

    delivered = 0
    with pool.connection() as conn:
        for event_id, external_id, err in results:
            if err is None and external_id is not None:
                record_delivery(conn, event_id, sink, external_id)
                delivered += 1
            else:
                record_failure(conn, event_id, sink, err or "unknown")
        conn.commit()
    return delivered
