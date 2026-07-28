"""Stripe Billing Meters dual-write (observability only).

Wallet ledger remains the system of record for real-time prepaid debit.
Meter events are best-effort: failures never roll back a wallet charge.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Callable

log = logging.getLogger("xcelsior.stripe_meters")

# Event names must match config/stripe_catalog.json / seed_stripe_products.py
EVENT_GPU_HOUR = "xcelsior_gpu_hour"
EVENT_SERVERLESS_SECOND = "xcelsior_serverless_worker_second"
EVENT_STORAGE_GB_MONTH = "xcelsior_storage_gb_month"

# Optional test hook: when set, enqueue/drain use this instead of Postgres.
_OUTBOX_BACKEND: Callable[..., Any] | None = None


def meters_enabled() -> bool:
    return os.environ.get("XCELSIOR_STRIPE_METERS_ENABLED", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def set_outbox_backend(backend: Callable[..., Any] | None) -> None:
    """Test-only injectable outbox (must not be used for production paths)."""
    global _OUTBOX_BACKEND
    _OUTBOX_BACKEND = backend


def _infer_event(description: str, job_id: str) -> tuple[str, float]:
    """Map a wallet charge to a meter event name + numeric value."""
    desc = (description or "").lower()
    jid = (job_id or "").lower()
    if "serverless" in desc or jid.startswith("slvr") or "worker" in desc:
        m = re.search(r"\((\d+)\s*s", description or "")
        seconds = float(m.group(1)) if m else 1.0
        return EVENT_SERVERLESS_SECOND, max(1.0, seconds)
    if "storage" in desc or "volume" in desc:
        return EVENT_STORAGE_GB_MONTH, 1.0
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
        idempotency_key=tx_id or f"charge-{customer_id}-{uuid.uuid4().hex[:12]}",
        payload={
            "amount_cad": amount_cad,
            "job_id": job_id,
            "description": (description or "")[:240],
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
    if _OUTBOX_BACKEND is not None:
        return _OUTBOX_BACKEND(
            "enqueue",
            customer_id=customer_id,
            event_name=event_name,
            value=float(value),
            idempotency_key=idempotency_key,
            payload=payload or {},
        )

    from db import _get_pg_pool
    from psycopg.rows import dict_row

    now = time.time()
    event_id = str(uuid.uuid4())
    try:
        pool = _get_pg_pool()
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
        return {"enqueued": True, "event_id": event_id, "event_name": event_name, "value": float(value)}
    except Exception as exc:
        # Table may not exist yet — never break billing.
        log.warning("meter outbox enqueue failed (non-fatal): %s", exc)
        return {"enqueued": False, "reason": str(exc)[:200]}


def _stripe_customer_id(customer_id: str) -> str:
    try:
        from billing import get_billing_engine

        wallet = get_billing_engine().get_wallet(customer_id)
        return (wallet.get("stripe_customer_id") or "").strip()
    except Exception:
        return ""


def drain_meter_outbox(*, limit: int = 100, stripe_mod=None) -> dict[str, int]:
    """Send pending meter events to Stripe. Never touches wallet balances."""
    if _OUTBOX_BACKEND is not None:
        result = _OUTBOX_BACKEND("drain", limit=limit, stripe_mod=stripe_mod)
        return result if isinstance(result, dict) else {"sent": 0, "failed": 0, "skipped": 0}

    from db import _get_pg_pool
    from psycopg.rows import dict_row
    from stripe_connect import STRIPE_ENABLED, stripe as default_stripe

    client = stripe_mod if stripe_mod is not None else default_stripe
    if not meters_enabled() or not STRIPE_ENABLED or not client:
        return {"sent": 0, "failed": 0, "skipped": 0}

    sent = failed = skipped = 0
    try:
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            # No FOR UPDATE across connection boundary — select then process.
            rows = conn.execute(
                """
                SELECT event_id, customer_id, event_name, value, idempotency_key, attempts
                  FROM stripe_meter_event_outbox
                 WHERE status = 'pending' AND attempts < 8
                 ORDER BY created_at ASC
                 LIMIT %s
                """,
                (limit,),
            ).fetchall()
            rows = [dict(r) for r in rows]
    except Exception as exc:
        log.warning("meter outbox drain skipped: %s", exc)
        return {"sent": 0, "failed": 0, "skipped": 0}

    for row in rows:
        event_id = row["event_id"]
        customer_id = row["customer_id"]
        stripe_cust = _stripe_customer_id(customer_id)
        if not stripe_cust:
            try:
                from billing import get_billing_engine

                stripe_cust = get_billing_engine().ensure_stripe_customer(customer_id) or ""
            except Exception:
                stripe_cust = ""
        if not stripe_cust:
            skipped += 1
            _mark(event_id, status="pending", attempts=int(row["attempts"] or 0) + 1, err="no_stripe_customer")
            continue
        try:
            value_int = max(1, int(round(float(row["value"]))))
            # MeterEvent payload keys depend on meter config; use customer + value.
            client.billing.MeterEvent.create(
                event_name=row["event_name"],
                payload={
                    "stripe_customer_id": stripe_cust,
                    "value": str(value_int),
                },
                identifier=str(row["idempotency_key"])[:100],
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

    now = time.time()
    try:
        with _get_pg_pool().connection() as conn:
            conn.execute(
                """
                UPDATE stripe_meter_event_outbox
                   SET status=%s, attempts=%s, last_error=%s, updated_at=%s,
                       sent_at = CASE WHEN %s = 'sent' THEN %s ELSE sent_at END
                 WHERE event_id=%s
                """,
                (status, attempts, err, now, status, now, event_id),
            )
            conn.commit()
    except Exception as exc:
        log.warning("meter outbox mark failed: %s", exc)
