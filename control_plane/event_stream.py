"""Durable, resumable event stream cursor (Track B B4.6, §16.3).

SSE clients must survive a replica dying: on reconnect with `Last-Event-ID` they
receive every transition they missed, exactly once, in order. That requires a
**durable, monotonically-ordered** source — the `outbox_events` projection —
behind the process-local `broadcast_sse` fan-out (which stays only as a latency
optimization).

The cursor is the outbox row's `(created_at, event_id)` — a total order (the
UUID tiebreaks equal timestamps). `resume_after` returns the events strictly
after a cursor, in that order, so a reconnecting client replays exactly the gap
and nothing it already saw.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    cursor: str
    event_id: str
    event_type: str
    payload: dict[str, Any]


def encode_cursor(created_at: Any, event_id: str) -> str:
    """`<epoch_micros>:<event_id>` — opaque, but a total order over the stream."""
    if hasattr(created_at, "timestamp"):
        micros = int(created_at.timestamp() * 1_000_000)
    else:
        micros = int(float(created_at) * 1_000_000)
    return f"{micros}:{event_id}"


def decode_cursor(cursor: str | None) -> tuple[_dt.datetime, str] | None:
    """Parse a Last-Event-ID back to `(created_at, event_id)`; None if absent/bad."""
    if not cursor:
        return None
    try:
        micros_str, event_id = cursor.split(":", 1)
        ts = _dt.datetime.fromtimestamp(int(micros_str) / 1_000_000, tz=_dt.timezone.utc)
        return ts, event_id
    except (ValueError, TypeError):
        return None


def resume_after(conn: Any, cursor: str | None, *, limit: int = 500) -> list[StreamEvent]:
    """Durable events strictly after `cursor`, in total order (the replay gap).

    With no cursor, returns the earliest `limit` events. The `(created_at,
    event_id)` row comparison gives a stable, exactly-once, in-order replay.
    """
    decoded = decode_cursor(cursor)
    if decoded is None:
        rows = conn.execute(
            """
            SELECT event_id, event_type, payload, created_at
              FROM outbox_events
             ORDER BY created_at, event_id
             LIMIT %s
            """,
            (limit,),
        ).fetchall()
    else:
        ts, event_id = decoded
        rows = conn.execute(
            """
            SELECT event_id, event_type, payload, created_at
              FROM outbox_events
             WHERE (created_at, event_id) > (%s, %s)
             ORDER BY created_at, event_id
             LIMIT %s
            """,
            (ts, event_id, limit),
        ).fetchall()
    out: list[StreamEvent] = []
    for event_id, event_type, payload, created_at in rows:
        out.append(
            StreamEvent(
                cursor=encode_cursor(created_at, str(event_id)),
                event_id=str(event_id),
                event_type=str(event_type),
                payload=payload if isinstance(payload, dict) else {},
            )
        )
    return out
