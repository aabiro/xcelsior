"""Receipts for finished runs, read from the ledger that billed them.

One builder, used by the instance receipt route, the receipts list, the v1 API
and the MCP tool. It reads; it never bills, rounds a charge or writes.

Where each number comes from:

* **Seconds and amount owed** — the closed ``usage_meters`` row of each attempt.
  The meter is closed from the attempt's server-recorded start and end
  (``billing._close_meter_from_attempt``), and its ``total_cost_micros`` is the
  amount that attempt owes.
* **Paid** — the ``billing_cycles`` rows the billing loop charged for the job,
  and the ``wallet_transactions`` debits it made. A receipt is ``reconciled``
  only when owed, charged and debited agree to the micro.
* **Locked rate** — the rate the attempt was billed at: the service-tier quote's
  ``total_rate_micros`` when the meter carries one, otherwise the meter's
  captured rate less its spot discount.
* **Host verification at start** — what the placement read under the host lock
  (``placement_explanation.host_verification``, recorded since 2026-11-22), or
  for older attempts the last verification-history row before the start.

The whole-hours figure answers "what would this run cost if billed by the hour
at the same rate". Each attempt is rounded up on its own, because each attempt
is a separate run. It is labelled as a comparison; nothing is billed that way.

Runs without a service-tier quote are billed by the legacy formula, which rounds
the amount to four decimal places of a dollar before converting to micros. The
receipt reports what that formula billed and marks those attempts
``"rounding": "legacy_4dp"``. Changing the formula would change amounts already
billed and break the meter-equals-charges check provider settlement relies on,
so it stays; every new customer instance carries a quote and bills exactly.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from decimal import Decimal
from typing import Any

from money import cad_to_micros

RECEIPT_VERSION = 1
HOURLY_ROUNDED_LABEL = "if billed in whole hours at the same rate"
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "preempted", "cancelled", "terminated"})
MAX_LIST_LIMIT = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _epoch(value: Any) -> float | None:
    """Epoch seconds from a float column or a timestamptz value."""
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    return _num(value)


def _rate_micros(meter: dict) -> tuple[int | None, str]:
    """The hourly rate this attempt was billed at, and how its amount was rounded."""
    snapshot = meter.get("pricing_snapshot") or {}
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot or "{}")
    total = snapshot.get("total_rate_micros") if isinstance(snapshot, dict) else None
    if type(total) is int and total >= 0:
        return total, "exact"
    rate = _num(meter.get("base_rate_per_hour"))
    discount = _num(meter.get("spot_discount")) or 0.0
    if rate is None or rate < 0 or not 0 <= discount <= 1:
        return None, "legacy_4dp"
    return cad_to_micros(rate * (1 - discount)), "legacy_4dp"


def hourly_rounded_micros(rate_micros: int | None, billed_seconds: float) -> int | None:
    """``ceil(seconds / 3600) × rate`` — the comparison figure, never a charge."""
    if rate_micros is None:
        return None
    hours = math.ceil(Decimal(str(billed_seconds)) / Decimal(3600)) if billed_seconds > 0 else 0
    return int(hours) * rate_micros


def _verification_at_start(conn, attempt: dict) -> dict | None:
    explanation = attempt.get("placement_explanation") or {}
    recorded = explanation.get("host_verification") if isinstance(explanation, dict) else None
    if isinstance(recorded, dict) and recorded.get("state"):
        return {
            "state": recorded.get("state"),
            "freshness": recorded.get("freshness"),
            "verification_id": recorded.get("verification_id"),
            "checked_at": _num(recorded.get("last_check_at")),
            "source": "placement",
        }
    host_id = attempt.get("host_id")
    started = _epoch(attempt.get("started_at"))
    if not host_id or started is None:
        return None
    row = conn.execute(
        """SELECT state, verification_id, timestamp FROM verification_history
            WHERE host_id = %s AND timestamp <= %s
            ORDER BY timestamp DESC LIMIT 1""",
        (host_id, started),
    ).fetchone()
    if row is None:
        return {
            "state": "unverified",
            "freshness": None,
            "verification_id": None,
            "checked_at": None,
            "source": "history",
        }
    return {
        "state": row["state"],
        # History rows carry no schedule, so the stamp's age at the start is unknown.
        "freshness": None,
        "verification_id": row["verification_id"] or None,
        "checked_at": _num(row["timestamp"]),
        "source": "history",
    }


def build_receipt(conn, job: dict) -> dict:
    """The receipt for one job. The caller has already checked who may see it.

    ``conn`` must return dict rows. Attempts whose meter is still open (a run
    in progress) are left out of every total; ``final`` is true only once the
    job itself is terminal.
    """
    job_id = str(job["job_id"])
    attempts = conn.execute(
        """SELECT attempt_id::text AS attempt_id, attempt_number, status, host_id,
                  placement_explanation, started_at, ended_at
             FROM job_attempts WHERE job_id = %s ORDER BY attempt_number""",
        (job_id,),
    ).fetchall()
    meters = conn.execute(
        """SELECT meter_id, attempt_id, started_at, completed_at, duration_sec,
                  total_cost_micros, base_rate_per_hour, spot_discount, pricing_mode,
                  pricing_snapshot, gpu_model, host_id
             FROM usage_meters WHERE job_id = %s ORDER BY started_at""",
        (job_id,),
    ).fetchall()
    charged = conn.execute(
        """SELECT COALESCE(SUM(amount_micros), 0)::bigint AS micros
             FROM billing_cycles
            WHERE job_id = %s AND resource_type = 'gpu' AND status = 'charged'""",
        (job_id,),
    ).fetchone()
    wallet = conn.execute(
        """SELECT tx_type, COALESCE(SUM(amount_micros), 0)::bigint AS micros,
                  COUNT(*) AS n
             FROM wallet_transactions
            WHERE job_id = %s AND customer_id = %s AND tx_type IN ('charge', 'refund', 'credit')
            GROUP BY tx_type""",
        (job_id, job.get("owner") or ""),
    ).fetchall()
    by_tx = {row["tx_type"]: row for row in wallet}

    meter_by_attempt = {str(m["attempt_id"]): m for m in meters if m.get("attempt_id")}
    lines: list[dict] = []
    for attempt in attempts:
        meter = meter_by_attempt.pop(str(attempt["attempt_id"]), None)
        lines.append(_attempt_line(conn, attempt, meter))
    # Meters written before attempts carried their id still billed something.
    for meter in meter_by_attempt.values():
        lines.append(_attempt_line(conn, None, meter))
    for meter in meters:
        if not meter.get("attempt_id"):
            lines.append(_attempt_line(conn, None, meter))

    closed = [line for line in lines if line["closed"]]
    billed_seconds = round(sum(line["billed_seconds"] for line in closed), 2)
    total_micros = sum(line["total_micros"] for line in closed)
    rates = {line["rate_micros_per_hour"] for line in closed}
    rate = next(iter(rates)) if len(rates) == 1 else None
    rounded = [line["hourly_rounded_micros"] for line in closed]
    wallet_debit = -int(by_tx["charge"]["micros"]) if "charge" in by_tx else 0
    cycles_charged = int(charged["micros"]) if charged else 0
    credits = sum(int(by_tx[t]["micros"]) for t in ("refund", "credit") if t in by_tx)
    starts = [line["started_at"] for line in closed if line["started_at"] is not None]
    ends = [line["ended_at"] for line in closed if line["ended_at"] is not None]
    status = str(job.get("status") or "")
    first = closed[0] if closed else None
    return {
        "receipt_version": RECEIPT_VERSION,
        "job_id": job_id,
        "name": job.get("name") or None,
        "status": status,
        "final": status in TERMINAL_JOB_STATES and all(line["closed"] for line in lines),
        "currency": "CAD",
        "gpu_model": job.get("gpu_model") or (first or {}).get("gpu_model"),
        "gpu_count": int(job.get("num_gpus") or 1),
        "pricing_mode": job.get("pricing_mode") or (first or {}).get("pricing_mode"),
        "image": job.get("image") or None,
        # The quick-start template the launch form started from, and the
        # saved image (user_images) a launch named; either, both or neither.
        "launch_template_id": job.get("launch_template_id") or None,
        "saved_image_id": job.get("source_template_id") or None,
        "started_at": min(starts) if starts else None,
        "ended_at": max(ends) if ends else None,
        "billed_seconds": billed_seconds,
        "rate_micros_per_hour": rate,
        "total_micros": total_micros,
        "cycles_charged_micros": cycles_charged,
        "wallet_debit_micros": wallet_debit,
        "wallet_debit_count": int(by_tx["charge"]["n"]) if "charge" in by_tx else 0,
        "outstanding_micros": max(0, total_micros - wallet_debit),
        "credits_micros": credits,
        "reconciled": total_micros == cycles_charged == wallet_debit,
        "hourly_rounded_micros": (
            sum(r for r in rounded if r is not None) if None not in rounded else None
        ),
        "hourly_rounded_label": HOURLY_ROUNDED_LABEL,
        "rounding": (
            "exact"
            if all(line["rounding"] == "exact" for line in closed)
            else "legacy_4dp" if closed else "exact"
        ),
        "host_verification_at_start": (first or {}).get("host_verification_at_start"),
        "attempts": [_public_line(line) for line in lines],
    }


def _attempt_line(conn, attempt: dict | None, meter: dict | None) -> dict:
    started = _epoch((meter or {}).get("started_at")) or _epoch((attempt or {}).get("started_at"))
    completed = _num((meter or {}).get("completed_at"))
    closed = bool(meter) and completed is not None and completed > 0
    billed_seconds = (_num((meter or {}).get("duration_sec")) or 0.0) if closed else 0.0
    rate, rounding = _rate_micros(meter) if meter else (None, "exact")
    total = int((meter or {}).get("total_cost_micros") or 0) if closed else 0
    never_started = attempt is not None and meter is None
    return {
        "attempt_id": (attempt or {}).get("attempt_id") or (meter or {}).get("attempt_id"),
        "attempt_number": (attempt or {}).get("attempt_number"),
        "status": (attempt or {}).get("status"),
        "host_id": (attempt or {}).get("host_id") or (meter or {}).get("host_id"),
        "gpu_model": (meter or {}).get("gpu_model"),
        "pricing_mode": (meter or {}).get("pricing_mode"),
        "started_at": started if meter else None,
        "ended_at": completed if closed else None,
        # An attempt that never started has nothing open: it billed nothing.
        "closed": closed or never_started,
        "billed_seconds": billed_seconds,
        "rate_micros_per_hour": rate,
        "total_micros": total,
        "hourly_rounded_micros": (
            hourly_rounded_micros(rate, billed_seconds) if closed else 0 if never_started else None
        ),
        "rounding": rounding,
        "host_verification_at_start": (
            _verification_at_start(conn, attempt) if attempt and meter else None
        ),
    }


def _public_line(line: dict) -> dict:
    out = {k: v for k, v in line.items() if k not in ("closed", "gpu_model", "pricing_mode")}
    out["finished"] = line["closed"]
    return out


# ── Listing ─────────────────────────────────────────────────────────


def encode_cursor(submitted_at: float, job_id: str) -> str:
    raw = json.dumps([submitted_at, job_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[float, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        submitted_at, job_id = json.loads(base64.urlsafe_b64decode(padded.encode()))
        value = float(submitted_at)
        if not math.isfinite(value) or not isinstance(job_id, str):
            raise ValueError
        return value, job_id
    except (ValueError, TypeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ValueError("invalid cursor") from exc


def list_receipt_jobs(
    conn, owner_ids: set[str] | None, *, limit: int, cursor: str | None
) -> tuple[list[str], str | None]:
    """Job ids with a receipt, newest submission first.

    A job has a receipt once it is terminal or one of its attempts has a closed
    meter (a stopped instance has billed its finished runs). ``owner_ids=None``
    lists every owner (platform admins only; the caller decides).
    """
    limit = max(1, min(int(limit), MAX_LIST_LIMIT))
    where = [
        "(j.status = ANY(%s) OR EXISTS (SELECT 1 FROM usage_meters m "
        "WHERE m.job_id = j.job_id AND m.completed_at > 0))"
    ]
    params: list[Any] = [sorted(TERMINAL_JOB_STATES)]
    if owner_ids is not None:
        where.append("j.owner_id = ANY(%s)")
        params.append(sorted(owner_ids))
    if cursor:
        after_ts, after_id = decode_cursor(cursor)
        where.append("(COALESCE(j.submitted_at, 0), j.job_id) < (%s, %s)")
        params.extend([after_ts, after_id])
    rows = conn.execute(
        "SELECT j.job_id, COALESCE(j.submitted_at, 0) AS submitted_at FROM jobs j "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY COALESCE(j.submitted_at, 0) DESC, j.job_id DESC LIMIT %s",
        (*params, limit + 1),
    ).fetchall()
    page = rows[:limit]
    next_cursor = (
        encode_cursor(float(page[-1]["submitted_at"]), page[-1]["job_id"])
        if len(rows) > limit
        else None
    )
    return [row["job_id"] for row in page], next_cursor


def billed_compute_by_job(conn, job_ids: list[str]) -> dict[str, dict]:
    """What each job's finished runs were billed, from their closed meters.

    One query for a whole instance list. Jobs with no closed meter are absent:
    nothing was billed, or nothing is known, and neither is a number.
    """
    ids = sorted({str(j) for j in job_ids if j})
    if not ids:
        return {}
    rows = conn.execute(
        """SELECT job_id, COALESCE(SUM(total_cost_micros), 0)::bigint AS micros,
                  COALESCE(SUM(duration_sec), 0)::float8 AS seconds
             FROM usage_meters
            WHERE job_id = ANY(%s) AND completed_at > 0
            GROUP BY job_id""",
        (ids,),
    ).fetchall()
    return {
        row["job_id"]: {"micros": int(row["micros"]), "seconds": float(row["seconds"])}
        for row in rows
    }
