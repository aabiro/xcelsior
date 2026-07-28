"""Stripe Billing Meters dual-write (observability only).

Wallet ledger remains the system of record for real-time prepaid debit.
Meter events are best-effort: failures never roll back a wallet charge.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

log = logging.getLogger("xcelsior.stripe_meters")

# Event names must match config/stripe_catalog.json / seed_stripe_products.py
EVENT_GPU_HOUR = "xcelsior_gpu_hour"
EVENT_SERVERLESS_SECOND = "xcelsior_serverless_worker_second"
EVENT_STORAGE_GB_MONTH = "xcelsior_storage_gb_month"


def meters_enabled() -> bool:
    return os.environ.get("XCELSIOR_STRIPE_METERS_ENABLED", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def _infer_event(description: str, job_id: str) -> tuple[str, float]:
    """Map a wallet charge to a meter event name + numeric value.

    Default: treat CAD amount as proxy GPU-hour units when duration unknown
    (value is analytical; wallet already charged the CAD amount).
    """
    desc = (description or "").lower()
    if "serverless" in desc or job_id.startswith("slvr") or "worker" in desc:
        # Prefer seconds if embedded like "(123s @"; else 1 unit.
        import re

        m = re.search(r"\((\d+)\s*s", description or "")
        seconds = float(m.group(1)) if m else 1.0
        return EVENT_SERVERLESS_SECOND, max(1.0, seconds)
    if "storage" in desc or "volume" in desc or "gb" in desc:
        return EVENT_STORAGE_GB_MONTH, 1.0
    # Hosted GPU: use hours approximated from description duration if present.
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hours)", desc)
    if m:
        return EVENT_GPU_HOUR, float(m.group(1))
    m = re.search(r"\((\d+)\s*s", description or "")
    if m:
        return EVENT_GPU_HOUR, max(1.0 / 3600.0, int(m.group(1)) / 3600.0)
    return EVENT_GPU_HOUR, 1.0


def enqueue_usage_from_charge(
    *,
    customer_id: str,
    amount_cad: float,
    job_id: str = "",
    description: str = "",
    tx_id: str = "",
) -> dict[str, Any]:
    if not meters_enabled():
        return {"enqueued": False, "reason": "disabled"}
    event_name, value = _infer_event(description, job_id)
    return enqueue_meter_event(
        customer_id=customer_id,
        event_name=event_name,
        value=value,
        idempotency_key=tx_id or f"charge-{customer_id}-{time.time()}",
        payload={
            "amount_cad": amount_cad,
            "job_id": job_id,
            "description": description[:240],
        },
    )


def enqueue_meter_event(
    *,
    customer_id: str,
    event_name: str,
    value: float,
    idempotency_key: str,
    payload: dict | None = None,
) -> dict[str, Any]:
    """Insert into stripe_meter_event_outbox (created by migration 079)."""
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    now = time.time()
    event_id = str(uuid.uuid4())
    pool = _get_pg_pool()
    try:
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """
                INSERT INTO stripe_meter_event_outbox
                    (event_id, customer_id, event_name, value, payload_json,
                     idempotency_key, status, attempts, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, 'pending', 0, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (
                    event_id,
                    customer_id,
                    event_name,
                    float(value),
                    json.dumps(payload or {}),
                    idempotency_key,
                    now,
                    now,
                ),
            )
            conn.commit()
        return {"enqueued": True, "event_id": event_id, "event_name": event_name}
    except Exception as exc:
        # Table may not exist yet — never break billing.
        log.debug("meter outbox enqueue failed: %s", exc)
        return {"enqueued": False, "reason": str(exc)[:200]}


def _stripe_customer_id(customer_id: str) -> str:
    try:
        from billing import get_billing_engine

        wallet = get_billing_engine().get_wallet(customer_id)
        return (wallet.get("stripe_customer_id") or "").strip()
    except Exception:
        return ""


def drain_meter_outbox(*, limit: int = 100) -> dict[str, int]:
    """Send pending meter events to Stripe. Free of wallet side effects."""
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    from stripe_connect import STRIPE_ENABLED, stripe

    if not meters_enabled() or not STRIPE_ENABLED or not stripe:
        return {"sent": 0, "failed": 0, "skipped": 0}

    pool = _get_pg_pool()
    sent = failed = skipped = 0
    try:
        with pool.connection() as conn:
            conn.row_factory = dict_row
            rows = conn.execute(
                """
                SELECT event_id, customer_id, event_name, value, idempotency_key, attempts
                  FROM stripe_meter_event_outbox
                 WHERE status = 'pending' AND attempts < 8
                 ORDER BY created_at ASC
                 LIMIT %s
                FOR UPDATE SKIP LOCKED
                """,
                (limit,),
            ).fetchall()
    except Exception as exc:
        log.debug("meter outbox drain skipped: %s", exc)
        return {"sent": 0, "failed": 0, "skipped": 0}

    for row in rows:
        event_id = row["event_id"]
        customer_id = row["customer_id"]
        stripe_cust = _stripe_customer_id(customer_id)
        if not stripe_cust:
            # Ensure customer exists without creating PaymentIntents.
            try:
                from billing import get_billing_engine

                stripe_cust = get_billing_engine().ensure_stripe_customer(customer_id)
            except Exception:
                stripe_cust = ""
        if not stripe_cust:
            skipped += 1
            _mark(event_id, status="pending", attempts=int(row["attempts"] or 0) + 1, err="no_stripe_customer")
            continue
        try:
            # stripe.billing.MeterEvent.create — no charge to customer.
            stripe.billing.MeterEvent.create(
                event_name=row["event_name"],
                payload={
                    "stripe_customer_id": stripe_cust,
                    "value": str(int(max(1, round(float(row["value"]))))),
                },
                identifier=row["idempotency_key"][:100],
            )
            _mark(event_id, status="sent", attempts=int(row["attempts"] or 0) + 1, err="")
            sent += 1
        except Exception as exc:
            failed += 1
            _mark(
                event_id,
                status="pending",
                attempts=int(row["attempts"] or 0) + 1,
                err=str(exc)[:240],
            )
            log.warning("meter event send failed %s: %s", event_id, exc)
    return {"sent": sent, "failed": failed, "skipped": skipped}


def _mark(event_id: str, *, status: str, attempts: int, err: str) -> None:
    from db import _get_pg_pool

    try:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                """
                UPDATE stripe_meter_event_outbox
                   SET status=%s, attempts=%s, last_error=%s, updated_at=%s,
                       sent_at = CASE WHEN %s = 'sent' THEN %s ELSE sent_at END
                 WHERE event_id=%s
                """,
                (status, attempts, err, time.time(), status, time.time(), event_id),
            )
            conn.commit()
    except Exception as exc:
        log.debug("meter outbox mark failed: %s", exc)
