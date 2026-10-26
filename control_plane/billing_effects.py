"""Durable handoff from financial transactions to API-owned side effects.

The billing login never acquires identity or host-command privileges. API
outbox consumers materialize notices and revalidate fenced stop requests.
"""

import logging
import time
import uuid

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from control_plane.db_roles import current_service
from control_plane.outbox import OutboxEvent, append_event

log = logging.getLogger(__name__)


def lock_period(conn, kind, subject_id, *, wait=False):
    """Serialize billing cursors without UPDATE authority on control rows."""
    key = f"billing_period:{kind}:{subject_id}"
    if wait:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (key,))
        return True
    row = conn.execute(
        "SELECT pg_try_advisory_xact_lock(hashtextextended(%s,0)) AS acquired", (key,)
    ).fetchone()
    return bool(row["acquired"] if isinstance(row, dict) else row[0])


def serialize_native_period(conn, job_id, *, wait=True):
    """Native writers take this after the job lock, before recording times.

    Financial writers never lock the job. This shared advisory lock prevents
    a billing snapshot from racing an uncommitted terminal timestamp.
    """
    row = conn.execute(
        "SELECT spec->>'job_type' AS job_type FROM jobs WHERE job_id=%s", (job_id,)
    ).fetchone()
    if row is None:
        return True  # The lifecycle fence check reports the missing job.
    job_type = row["job_type"] if isinstance(row, dict) else row[0]
    return lock_period(
        conn, "serverless" if job_type == "serverless_worker" else "gpu", job_id, wait=wait
    )


def queue_wallet_notice(conn, customer_id, kind, title, body, data):
    if kind not in {"billing_low_balance", "billing_suspended"}:
        raise ValueError("unsupported wallet notice")
    event_id = append_event(
        conn,
        aggregate_type="wallet",
        aggregate_id=customer_id,
        event_type="billing.v1.wallet_notice_requested",
        payload={
            "customer_id": customer_id,
            "kind": kind,
            "title": title,
            "body": body,
            "data": data,
        },
        destination_class="billing_notice",
        idempotency_key="wallet_notice:" + uuid.uuid4().hex,
        tenant_id=customer_id,
    )
    assert event_id is not None
    return "notif-billing-" + event_id


def _cell(row, key, index):
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    return row[index]


def queue_attempt_settlement(conn, job_id, attempt_id):
    """Ask the API identity to settle a lease the scheduler role cannot close.

    The scheduler transaction has already fenced the attempt. Settlement
    closes that attempt's meter and serverless binding. The hold id is the
    one on the job at expiry. A placement that lands before the API consumer
    runs owns a different hold, and settlement must leave that hold in place.
    """
    held = conn.execute(
        "SELECT wallet_hold_id FROM jobs WHERE job_id=%s",
        (job_id,),
    ).fetchone()
    hold_id = _cell(held, "wallet_hold_id", 0)
    return append_event(
        conn,
        aggregate_type="job",
        aggregate_id=job_id,
        event_type="billing.v1.attempt_settlement_requested",
        payload={
            "attempt_id": str(attempt_id),
            "wallet_hold_id": None if hold_id is None else str(hold_id),
        },
        destination_class="attempt_settlement",
        idempotency_key="attempt_settlement:" + str(attempt_id),
    )


def queue_failed_cycle_stop(conn, job, result):
    if not job.get("active_attempt_id") or job.get("generation") is None:
        raise ValueError("billing stop requires the observed native authority")
    return append_event(
        conn,
        aggregate_type="job",
        aggregate_id=job["job_id"],
        event_type="billing.v1.failed_cycle_stop_requested",
        payload={
            "cycle_id": result["cycle_id"],
            "attempt_id": str(job["active_attempt_id"]),
            "generation": int(job["generation"]),
        },
        destination_class="billing_stop",
        idempotency_key="billing_stop:" + result["cycle_id"],
        tenant_id=job["owner"],
    )


def handle_wallet_notice(event: OutboxEvent):
    from db import _get_pg_pool, NotificationStore

    if current_service() != "api":
        raise RuntimeError("wallet notice consumer requires the API identity")
    if event.event_type != "billing.v1.wallet_notice_requested":
        raise ValueError("unexpected billing notice event")
    p = event.payload
    if p.get("kind") not in {"billing_low_balance", "billing_suspended"}:
        raise ValueError("unsupported wallet notice")
    notification_id = "notif-billing-" + event.event_id
    with _get_pg_pool().connection() as conn:
        inserted = conn.execute(
            """INSERT INTO notifications(id,user_email,type,title,body,data,read,created_at)
               VALUES(%s,%s,%s,%s,%s,%s,0,%s) ON CONFLICT(id) DO NOTHING RETURNING id""",
            (
                notification_id,
                p["customer_id"],
                p["kind"],
                p["title"],
                p["body"],
                Jsonb(p["data"]),
                time.time(),
            ),
        ).fetchone()
    if inserted:
        NotificationStore.deliver(notification_id)


def handle_failed_cycle_stop(event: OutboxEvent):
    from control_plane.lifecycle import request_billing_stop

    if current_service() != "api":
        raise RuntimeError("billing stop consumer requires the API identity")
    if event.event_type != "billing.v1.failed_cycle_stop_requested":
        raise ValueError("unexpected billing stop event")
    result = request_billing_stop(
        job_id=event.aggregate_id,
        cycle_id=event.payload["cycle_id"],
        expected_attempt_id=event.payload["attempt_id"],
        expected_generation=event.payload["generation"],
    )
    if not result.ok and result.reason == "enqueue_failed":
        raise RuntimeError("billing stop delivery remains pending")


def handle_attempt_settlement(event: OutboxEvent):
    from control_plane.attempt_billing import sync_attempt_billing
    from control_plane.serverless_workers import sync_serverless_attempt
    from db import _get_pg_pool

    if current_service() != "api":
        raise RuntimeError("attempt settlement consumer requires the API identity")
    if event.event_type != "billing.v1.attempt_settlement_requested":
        raise ValueError("unexpected attempt settlement event")
    attempt_id = str(event.payload["attempt_id"])
    hold_id = event.payload.get("wallet_hold_id")
    with _get_pg_pool().connection() as conn:
        # The lease transaction that fenced this attempt held the period lock;
        # its deferred settlement takes it too, so a concurrent billing sweep
        # never charges a slice of this attempt while its meter is closing.
        serialize_native_period(conn, event.aggregate_id)
        sync_attempt_billing(
            conn,
            attempt_id,
            released_hold_id=None if hold_id is None else str(hold_id),
            bind_release=True,
        )
        sync_serverless_attempt(conn, job_id=event.aggregate_id, attempt_id=attempt_id)


def publish_committed_effects(*, notice_ids=(), stop_event_ids=()):
    """API callers may project immediately; billing leaves delivery to the outbox.

    A failed immediate projection never consumes its durable retry obligation.
    No service-scope override or secondary API credential is used here.
    """
    if current_service() != "api":
        return
    from db import _get_pg_pool

    ids = [nid.removeprefix("notif-billing-") for nid in notice_ids]
    ids.extend(event_id for event_id in stop_event_ids if event_id)
    if not ids:
        return
    try:
        with _get_pg_pool().connection() as conn:
            conn.row_factory = dict_row
            rows = conn.execute(
                """SELECT event_id,aggregate_type,aggregate_id,event_type,payload,headers,
                          destination_class,idempotency_key,attempt_count,created_at
                     FROM outbox_events WHERE event_id=ANY(%s::uuid[])""",
                (ids,),
            ).fetchall()
        for row in rows:
            event = OutboxEvent(**{**row, "event_id": str(row["event_id"])})
            if event.destination_class == "billing_notice":
                handle_wallet_notice(event)
            elif event.destination_class == "billing_stop":
                handle_failed_cycle_stop(event)
    except Exception as exc:
        log.warning("committed billing effect awaits outbox retry (%s)", type(exc).__name__)
