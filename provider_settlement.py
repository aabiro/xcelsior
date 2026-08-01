"""Exact, PostgreSQL-authoritative provider settlement primitives.

The public payout APIs identify a job and a rail; they never supply money.
Eligibility is reconstructed under a job row lock from:

* ``wallet_transactions.amount_micros`` — exact amount actually charged;
* ``usage_meters.total_cost_micros`` — terminal usage evidence;
* ``jobs`` / ``hosts`` — terminal state and normalized provider ownership; and
* ``provider_accounts`` — payout currency and tax province.

The resulting row in ``payout_splits`` is the single cross-rail authority for
the job.  External calls happen only after a durable leased claim is committed.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Iterable

from money import MICROS_PER_CAD, micros_to_cad

MICROS_PER_CENT = MICROS_PER_CAD // 100
_ALLOWED_RAILS = {"stripe", "paypal"}
_ELIGIBLE_JOB_STATUSES = {"completed", "terminated"}
_ELIGIBLE_JOB_PHASES = {"succeeded"}


class SettlementError(RuntimeError):
    """Base error carrying a stable API-safe reason."""

    def __init__(self, reason: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


class SettlementNotFound(SettlementError):
    def __init__(self, message: str = "Settlement source not found"):
        super().__init__("not_found", message, status_code=404)


class SettlementNotEligible(SettlementError):
    def __init__(self, reason: str, message: str):
        super().__init__(reason, message, status_code=409)


class SettlementConflict(SettlementError):
    def __init__(self, reason: str, message: str):
        super().__init__(reason, message, status_code=409)


@dataclass(frozen=True)
class ExactSettlementSplit:
    """Cent-payable rail split represented in integer micro-CAD."""

    source_total_micros: int
    total_micros: int
    provider_share_micros: int
    platform_share_micros: int
    gst_hst_micros: int
    rounding_adjustment_micros: int
    platform_cut_bps: int
    tax_rate_bps: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _half_up_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("settlement division requires non-negative numerator")
    return (numerator + denominator // 2) // denominator


def platform_cut_bps(raw: str | Decimal | None = None) -> int:
    """Resolve the configured commission to integer basis points."""

    value = raw if raw is not None else os.environ.get("XCELSIOR_PLATFORM_CUT", "0.15")
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("XCELSIOR_PLATFORM_CUT must be a decimal number") from exc
    if decimal < 0:
        raise ValueError("XCELSIOR_PLATFORM_CUT cannot be negative")
    bps = (decimal * Decimal(10_000) if decimal <= Decimal(1) else decimal * Decimal(100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    result = int(bps)
    if result < 0 or result > 10_000:
        raise ValueError("XCELSIOR_PLATFORM_CUT must be between 0 and 100 percent")
    return result


def tax_rate_bps_for_province(province: str) -> tuple[int, str]:
    """Return tax basis points and description using the billing tax table.

    ``get_tax_rate_for_province`` returns ``(rate, description)``.  Keeping the
    unpack here prevents the historic tuple-as-number payout failure.
    """

    from billing import get_tax_rate_for_province

    result = get_tax_rate_for_province(province)
    if isinstance(result, tuple):
        rate, description = result
    else:  # Compatibility with narrow test doubles; production returns a tuple.
        rate, description = result, ""
    bps = int((Decimal(str(rate)) * Decimal(10_000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if bps < 0 or bps > 10_000:
        raise ValueError("province tax rate is outside the supported range")
    return bps, str(description or "")


def split_source_micros(
    source_total_micros: int,
    *,
    cut_bps: int | None = None,
    tax_bps: int = 0,
) -> ExactSettlementSplit:
    """Create a deterministic, zero-residual settlement split.

    Stripe and PayPal settle CAD in cents.  The paid source stays preserved in
    micro-CAD, while the rail total is rounded half-up once to a cent.  The
    explicit adjustment makes that boundary auditable.  Platform commission is
    rounded once to cents and the provider receives the exact residual, so:

    ``provider_share_micros + platform_share_micros == total_micros``.
    """

    source = int(source_total_micros)
    if source <= 0:
        raise ValueError("source_total_micros must be positive")
    cut = platform_cut_bps() if cut_bps is None else int(cut_bps)
    tax = int(tax_bps)
    if not 0 <= cut <= 10_000:
        raise ValueError("cut_bps must be between 0 and 10000")
    if not 0 <= tax <= 10_000:
        raise ValueError("tax_bps must be between 0 and 10000")

    total_cents = _half_up_div(source, MICROS_PER_CENT)
    if total_cents <= 0:
        raise ValueError("source amount is below the minimum payable rail unit")
    platform_cents = _half_up_div(total_cents * cut, 10_000)
    provider_cents = total_cents - platform_cents
    tax_cents = _half_up_div(total_cents * tax, 10_000)

    total = total_cents * MICROS_PER_CENT
    platform = platform_cents * MICROS_PER_CENT
    provider = provider_cents * MICROS_PER_CENT
    return ExactSettlementSplit(
        source_total_micros=source,
        total_micros=total,
        provider_share_micros=provider,
        platform_share_micros=platform,
        gst_hst_micros=tax_cents * MICROS_PER_CENT,
        rounding_adjustment_micros=total - source,
        platform_cut_bps=cut,
        tax_rate_bps=tax,
    )


def settlement_key(job_id: str) -> str:
    return f"provider-job:{job_id}"


def rail_idempotency_key(job_id: str) -> str:
    return f"provider-settlement:{job_id}"


def _as_dict(row: Any) -> dict:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        raise TypeError("settlement queries require dict-like rows") from None


def _validate_existing(
    row: dict,
    *,
    provider_id: str,
    rail: str,
    expected_customer_id: str | None,
) -> dict:
    if str(row.get("provider_id") or "") != provider_id:
        raise SettlementConflict(
            "provider_mismatch",
            "The job is already bound to a different provider settlement",
        )
    if str(row.get("payment_rail") or "") != rail:
        raise SettlementConflict(
            "rail_already_selected",
            f"The job is already bound to the {row.get('payment_rail')} payout rail",
        )
    customer_id = str(row.get("customer_id") or "")
    if expected_customer_id and customer_id != expected_customer_id:
        raise SettlementNotFound("No eligible settlement exists for this customer")
    return row


def get_settlement(
    conn,
    *,
    job_id: str,
    provider_id: str | None = None,
    rail: str | None = None,
    expected_customer_id: str | None = None,
) -> dict | None:
    row = conn.execute(
        "SELECT * FROM payout_splits WHERE settlement_key = %s",
        (settlement_key(job_id),),
    ).fetchone()
    if row is None:
        return None
    result = _as_dict(row)
    if provider_id is not None and rail is not None:
        return _validate_existing(
            result,
            provider_id=provider_id,
            rail=rail,
            expected_customer_id=expected_customer_id,
        )
    return result


def prepare_settlement(
    conn,
    *,
    job_id: str,
    provider_id: str,
    rail: str,
    expected_customer_id: str | None = None,
    cut_bps: int | None = None,
) -> dict:
    """Create or return one exact cross-rail settlement under a job lock."""

    job_id = str(job_id or "").strip()
    provider_id = str(provider_id or "").strip()
    rail = str(rail or "").strip().lower()
    if not job_id or not provider_id:
        raise SettlementNotEligible("invalid_identity", "job_id and provider_id are required")
    if rail not in _ALLOWED_RAILS:
        raise SettlementNotEligible("unsupported_rail", "payment rail must be stripe or paypal")

    authority = conn.execute(
        """
        SELECT j.job_id, j.status AS job_status, j.phase AS job_phase,
               j.host_id, j.owner_id,
               COALESCE(NULLIF(j.owner_id, ''), NULLIF(j.payload->>'owner', '')) AS job_owner,
               h.provider_id AS host_provider_id,
               pa.default_currency, pa.province
          FROM jobs j
          JOIN hosts h ON h.host_id = j.host_id
          JOIN provider_accounts pa ON pa.provider_id = h.provider_id
         WHERE j.job_id = %s
         FOR UPDATE OF j
        """,
        (job_id,),
    ).fetchone()
    if authority is None:
        raise SettlementNotFound("Job, host, or provider settlement authority was not found")
    auth = _as_dict(authority)

    existing = get_settlement(
        conn,
        job_id=job_id,
        provider_id=provider_id,
        rail=rail,
        expected_customer_id=expected_customer_id,
    )
    if existing is not None:
        return existing

    authoritative_provider = str(auth.get("host_provider_id") or "")
    if authoritative_provider != provider_id:
        # Return not-found rather than revealing another provider's earning.
        raise SettlementNotFound("No eligible settlement exists for this provider")

    status = str(auth.get("job_status") or "").lower()
    phase = str(auth.get("job_phase") or "").lower()
    if status not in _ELIGIBLE_JOB_STATUSES and phase not in _ELIGIBLE_JOB_PHASES:
        raise SettlementNotEligible(
            "job_not_terminal",
            "Provider settlement is available only after authoritative terminal completion",
        )

    currency = str(auth.get("default_currency") or "").upper()
    if currency != "CAD":
        raise SettlementNotEligible(
            "unsupported_currency",
            "Provider settlement currently requires a PostgreSQL currency of CAD",
        )

    charge_rows = conn.execute(
        """
        SELECT customer_id, -SUM(amount_micros)::bigint AS charged_micros
          FROM wallet_transactions
         WHERE job_id = %s
           AND tx_type = 'charge'
           AND amount_micros < 0
         GROUP BY customer_id
         ORDER BY customer_id
        """,
        (job_id,),
    ).fetchall()
    charges = [_as_dict(row) for row in charge_rows]
    if len(charges) != 1:
        reason = "no_paid_charge" if not charges else "multiple_charge_owners"
        raise SettlementNotEligible(
            reason,
            "The job does not have one reconciled, exact wallet charge owner",
        )
    customer_id = str(charges[0].get("customer_id") or "")
    charged_micros = int(charges[0].get("charged_micros") or 0)
    if charged_micros <= 0:
        raise SettlementNotEligible("no_paid_charge", "The job has no paid wallet charge")

    job_owner = str(auth.get("job_owner") or "")
    if job_owner and job_owner != customer_id:
        raise SettlementNotEligible(
            "customer_mismatch",
            "Job ownership and the exact wallet charge owner do not reconcile",
        )
    if expected_customer_id and expected_customer_id != customer_id:
        raise SettlementNotFound("No eligible settlement exists for this customer")

    meter = _as_dict(
        conn.execute(
            """
            SELECT COALESCE(SUM(total_cost_micros), 0)::bigint AS metered_micros,
                   COUNT(*)::integer AS meter_count,
                   COUNT(DISTINCT host_id)::integer AS host_count,
                   MIN(host_id) AS meter_host_id,
                   COUNT(DISTINCT owner)::integer AS owner_count,
                   MIN(owner) AS meter_owner
              FROM usage_meters
             WHERE job_id = %s
            """,
            (job_id,),
        ).fetchone()
    )
    metered_micros = int(meter.get("metered_micros") or 0)
    if metered_micros <= 0:
        raise SettlementNotEligible(
            "no_terminal_meter",
            "The job has no positive exact terminal usage meter",
        )
    if int(meter.get("host_count") or 0) != 1 or str(meter.get("meter_host_id") or "") != str(
        auth.get("host_id") or ""
    ):
        raise SettlementNotEligible(
            "meter_host_mismatch",
            "Terminal meter host ownership is ambiguous",
        )
    if (
        int(meter.get("owner_count") or 0) != 1
        or str(meter.get("meter_owner") or "") != customer_id
    ):
        raise SettlementNotEligible(
            "meter_customer_mismatch",
            "Terminal meter and wallet customer do not reconcile",
        )
    if metered_micros != charged_micros:
        raise SettlementNotEligible(
            "billing_not_reconciled",
            "Exact terminal meter and paid wallet charge totals do not reconcile",
        )

    tax_bps, _description = tax_rate_bps_for_province(str(auth.get("province") or ""))
    exact = split_source_micros(
        charged_micros,
        cut_bps=cut_bps,
        tax_bps=tax_bps,
    )
    values = exact.to_dict()
    now = time.time()
    conn.execute(
        """
        INSERT INTO payout_splits (
            job_id, provider_id, customer_id, tenant_id, currency,
            source_total_micros, total_micros, provider_share_micros,
            platform_share_micros, gst_hst_micros, rounding_adjustment_micros,
            platform_cut_bps, tax_rate_bps,
            stripe_transfer_id, paypal_capture_id, paypal_order_id,
            payment_rail, settlement_status, settlement_error,
            settlement_key, rail_idempotency_key,
            attempt_count, created_at, updated_at, legacy_imported
        )
        VALUES (
            %(job_id)s, %(provider_id)s, %(customer_id)s,
            %(customer_id)s, %(currency)s,
            %(source_total_micros)s, %(total_micros)s,
            %(provider_share_micros)s, %(platform_share_micros)s,
            %(gst_hst_micros)s, %(rounding_adjustment_micros)s,
            %(platform_cut_bps)s, %(tax_rate_bps)s,
            '', '', '', %(payment_rail)s, 'queued', '',
            %(settlement_key)s, %(rail_idempotency_key)s,
            0, clock_timestamp(), clock_timestamp(), FALSE
        )
        ON CONFLICT (settlement_key) WHERE settlement_key IS NOT NULL DO NOTHING
        """,
        {
            "job_id": job_id,
            "provider_id": provider_id,
            "customer_id": customer_id,
            "currency": currency,
            "total_cad": micros_to_cad(values["total_micros"]),
            "provider_share_cad": micros_to_cad(values["provider_share_micros"]),
            "platform_share_cad": micros_to_cad(values["platform_share_micros"]),
            "gst_hst_cad": micros_to_cad(values["gst_hst_micros"]),
            **values,
            "payment_rail": rail,
            "settlement_key": settlement_key(job_id),
            "rail_idempotency_key": rail_idempotency_key(job_id),
            "created_at": now,
        },
    )
    created = get_settlement(
        conn,
        job_id=job_id,
        provider_id=provider_id,
        rail=rail,
        expected_customer_id=expected_customer_id,
    )
    if created is None:
        raise SettlementConflict(
            "settlement_create_race",
            "The provider settlement could not be read after creation",
        )
    return created


def claim_settlements(
    conn,
    *,
    rail: str,
    owner: str,
    limit: int = 100,
    lease_seconds: int = 300,
    job_id: str | None = None,
    allowed_statuses: Iterable[str] = ("pending", "queued", "failed"),
) -> list[dict]:
    """Atomically claim settlement rows with ``FOR UPDATE SKIP LOCKED``."""

    rail = str(rail).lower()
    if rail not in _ALLOWED_RAILS:
        raise ValueError("unsupported settlement rail")
    safe_limit = max(1, min(int(limit), 500))
    safe_lease = max(30, min(int(lease_seconds), 3600))
    statuses = [str(value) for value in allowed_statuses]
    if not statuses:
        return []
    token = str(uuid.uuid4())
    rows = conn.execute(
        """
        WITH candidates AS (
            SELECT id
              FROM payout_splits
             WHERE settlement_key IS NOT NULL
               AND payment_rail = %(rail)s
               AND (
                   settlement_status = ANY(%(statuses)s)
                   OR (
                       settlement_status = 'processing'
                       AND (
                           claim_expires_at IS NULL
                           OR claim_expires_at < clock_timestamp()
                       )
                   )
               )
               AND (%(job_id)s::text IS NULL OR job_id = %(job_id)s::text)
               AND (next_attempt_at IS NULL OR next_attempt_at <= clock_timestamp())
               AND (claim_expires_at IS NULL OR claim_expires_at < clock_timestamp())
               AND CASE
                       WHEN payment_rail = 'stripe'
                           THEN COALESCE(stripe_transfer_id, '') = ''
                       ELSE COALESCE(paypal_capture_id, '') = ''
                   END
             ORDER BY created_at ASC, id ASC
             FOR UPDATE SKIP LOCKED
             LIMIT %(limit)s
        )
        UPDATE payout_splits ps
           SET settlement_status = 'processing',
               claim_owner = %(owner)s,
               claim_token = %(token)s,
               claim_expires_at =
                   clock_timestamp() + make_interval(secs => %(lease_seconds)s),
               attempt_count = attempt_count + 1,
               updated_at = clock_timestamp()
          FROM candidates
         WHERE ps.id = candidates.id
        RETURNING ps.*
        """,
        {
            "rail": rail,
            "statuses": statuses,
            "job_id": job_id,
            "limit": safe_limit,
            "owner": str(owner),
            "token": token,
            "lease_seconds": safe_lease,
        },
    ).fetchall()
    return [_as_dict(row) for row in rows]


def mark_settlement_paid(
    conn,
    *,
    settlement_id: int,
    claim_token: str,
    stripe_transfer_id: str = "",
    paypal_capture_id: str = "",
) -> dict:
    """Fence completion on the durable claim token."""

    row = conn.execute(
        """
        UPDATE payout_splits
           SET settlement_status = 'paid',
               settlement_error = '',
               stripe_transfer_id = CASE
                   WHEN %(stripe_transfer_id)s <> '' THEN %(stripe_transfer_id)s
                   ELSE stripe_transfer_id
               END,
               paypal_capture_id = CASE
                   WHEN %(paypal_capture_id)s <> '' THEN %(paypal_capture_id)s
                   ELSE paypal_capture_id
               END,
               settled_at = clock_timestamp(),
               claim_owner = NULL,
               claim_token = NULL,
               claim_expires_at = NULL,
               next_attempt_at = NULL,
               updated_at = clock_timestamp()
         WHERE id = %(settlement_id)s
           AND claim_token = %(claim_token)s
           AND settlement_status = 'processing'
        RETURNING *
        """,
        {
            "settlement_id": int(settlement_id),
            "claim_token": claim_token,
            "stripe_transfer_id": stripe_transfer_id,
            "paypal_capture_id": paypal_capture_id,
        },
    ).fetchone()
    if row is None:
        raise SettlementConflict(
            "claim_lost",
            "Settlement claim expired or was replaced before completion",
        )
    return _as_dict(row)


def mark_awaiting_paypal_capture(
    conn,
    *,
    settlement_id: int,
    claim_token: str,
    paypal_order_id: str,
) -> dict:
    row = conn.execute(
        """
        UPDATE payout_splits
           SET settlement_status = 'awaiting_capture',
               settlement_error = '',
               paypal_order_id = %(paypal_order_id)s,
               claim_owner = NULL,
               claim_token = NULL,
               claim_expires_at = NULL,
               next_attempt_at = NULL,
               updated_at = clock_timestamp()
         WHERE id = %(settlement_id)s
           AND claim_token = %(claim_token)s
           AND settlement_status = 'processing'
        RETURNING *
        """,
        {
            "settlement_id": int(settlement_id),
            "claim_token": claim_token,
            "paypal_order_id": paypal_order_id,
        },
    ).fetchone()
    if row is None:
        raise SettlementConflict(
            "claim_lost",
            "Settlement claim expired or was replaced before PayPal order persistence",
        )
    return _as_dict(row)


def mark_settlement_retry(
    conn,
    *,
    settlement_id: int,
    claim_token: str,
    error: str,
    retry_status: str = "queued",
) -> bool:
    if retry_status not in {"queued", "failed", "awaiting_capture", "manual_review"}:
        raise ValueError("unsupported retry status")
    row = conn.execute(
        """
        UPDATE payout_splits
           SET settlement_status = %(retry_status)s,
               settlement_error = %(error)s,
               next_attempt_at =
                   clock_timestamp()
                   + make_interval(
                       secs => LEAST(3600, GREATEST(30, 30 * attempt_count))
                     ),
               claim_owner = NULL,
               claim_token = NULL,
               claim_expires_at = NULL,
               updated_at = clock_timestamp()
         WHERE id = %(settlement_id)s
           AND claim_token = %(claim_token)s
           AND settlement_status = 'processing'
        RETURNING id
        """,
        {
            "settlement_id": int(settlement_id),
            "claim_token": claim_token,
            "error": str(error)[:500],
            "retry_status": retry_status,
        },
    ).fetchone()
    return row is not None


def settlement_response(row: dict) -> dict:
    """Stable public response without worker lease or fencing credentials."""

    public_fields = (
        "job_id",
        "provider_id",
        "currency",
        "source_total_micros",
        "total_micros",
        "provider_share_micros",
        "platform_share_micros",
        "gst_hst_micros",
        "rounding_adjustment_micros",
        "platform_cut_bps",
        "tax_rate_bps",
        "payment_rail",
        "settlement_status",
        "settlement_error",
        "stripe_transfer_id",
        "paypal_order_id",
        "paypal_capture_id",
        "created_at",
        "updated_at",
        "settled_at",
    )
    result = {key: row.get(key) for key in public_fields if key in row}
    for micros_key, cad_key in (
        ("source_total_micros", "source_total_cad"),
        ("total_micros", "total_cad"),
        ("provider_share_micros", "provider_share_cad"),
        ("platform_share_micros", "platform_share_cad"),
        ("gst_hst_micros", "gst_hst_cad"),
    ):
        if result.get(micros_key) is not None:
            result[cad_key] = micros_to_cad(int(result[micros_key]))
    result["tax_rate"] = int(result.get("tax_rate_bps") or 0) / 10_000
    return result
