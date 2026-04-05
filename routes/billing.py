"""Routes: billing."""

import io
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from routes._deps import (
    XCELSIOR_ENV,
    _USE_PERSISTENT_AUTH,
    _get_current_user,
    _require_admin,
    _require_auth,
    _users_db,
    broadcast_sse,
    log,
    otel_span,
)
from scheduler import (
    bill_job,
    get_total_revenue,
    load_billing,
    log,
)
from db import UserStore
from stripe_connect import get_stripe_manager
from billing import get_billing_engine, get_tax_rate_for_province
from reputation import GPU_REFERENCE_PRICING_CAD, estimate_job_cost

router = APIRouter()

# Billing constants
_FREE_CREDIT_AMOUNT = 10.0  # CAD
RESERVED_PRICING_TIERS = {
    "1_month": {
        "commitment": "1 month",
        "discount_pct": 20,
        "description": "20% off on-demand rates for 1-month commitment",
        "min_hours_per_day": 4,
    },
    "3_month": {
        "commitment": "3 months",
        "discount_pct": 30,
        "description": "30% off on-demand rates for 3-month commitment",
        "min_hours_per_day": 4,
    },
    "1_year": {
        "commitment": "1 year",
        "discount_pct": 45,
        "description": "45% off on-demand rates for 1-year commitment",
        "min_hours_per_day": 0,
    },
}
try:
    import bitcoin as _btc_mod
except ImportError:
    _btc_mod = None  # type: ignore[assignment]

@router.post("/billing/bill/{job_id}", tags=["Billing"])
def api_bill_instance(job_id: str, request: Request):
    """Bill a specific completed job."""
    _require_auth(request)
    with otel_span("billing.bill_job", {"job.id": job_id}):
        record = bill_job(job_id)
        if not record:
            raise HTTPException(status_code=400, detail=f"Could not bill job {job_id}")
        return {"ok": True, "bill": record}

@router.post("/billing/bill-all", tags=["Billing"])
def api_bill_all(request: Request):
    """Bill all unbilled completed jobs."""
    _require_auth(request)
    bills = bill_all_completed()
    return {"billed": len(bills), "bills": bills}

@router.get("/billing", tags=["Billing"])
def api_billing():
    """Get all billing records and total revenue."""
    records = load_billing()
    return {
        "records": records,
        "total_revenue": get_total_revenue(),
    }

@router.get("/api/billing/wallet/{customer_id}", tags=["Billing"])
def api_get_wallet(customer_id: str):
    """Get credit wallet balance and status."""
    be = get_billing_engine()
    wallet = be.get_wallet(customer_id)
    return {"ok": True, "wallet": wallet}


# ── Model: DepositRequest ──

class DepositRequest(BaseModel):
    amount_cad: float
    description: str = "Credit deposit"


# ── Model: PaymentIntentRequest ──

class PaymentIntentRequest(BaseModel):
    customer_id: str
    amount_cad: float
    description: str = "Compute credits"

@router.post("/api/billing/payment-intent", tags=["Billing"])
def api_create_payment_intent(req: PaymentIntentRequest):
    """Create a Stripe PaymentIntent for depositing compute credits.

    Returns client_secret for front-end Stripe Elements confirmation.
    On payment_intent.succeeded webhook the wallet is credited automatically.
    """
    if req.amount_cad < 1 or req.amount_cad > 10000:
        raise HTTPException(400, "Amount must be between $1 and $10,000 CAD")
    mgr = get_stripe_manager()
    result = mgr.create_credit_deposit(req.customer_id, req.amount_cad, req.description)
    return {"ok": True, "intent": result}

@router.post("/api/billing/wallet/{customer_id}/deposit", tags=["Billing"])
def api_deposit(customer_id: str, req: DepositRequest):
    """Deposit credits into a customer wallet."""
    be = get_billing_engine()
    result = be.deposit(customer_id, req.amount_cad, req.description)
    return {"ok": True, **result}

@router.post("/api/billing/wallet/{customer_id}/reset-testing", tags=["Billing"])
def api_reset_wallet_testing_state(customer_id: str, request: Request):
    """Reset wallet balance and promo state for admin testing. Disabled in production."""
    if XCELSIOR_ENV in ("production", "prod"):
        raise HTTPException(403, "Wallet reset is disabled in production")
    _require_admin(request)
    be = get_billing_engine()
    result = be.reset_wallet_testing_state(customer_id)
    return {"ok": True, **result}

@router.post("/api/billing/free-credits/{customer_id}", tags=["Billing"])
def api_claim_free_credits(customer_id: str, request: Request):
    """Claim one-time $10 CAD signup bonus.

    Uses an idempotency key derived from the customer_id so the bonus
    can only be claimed once per customer.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Authentication required")
    # Resolve customer_id from full user profile (session may lack it)
    full_user = UserStore.get_user(user["email"]) if _USE_PERSISTENT_AUTH else _users_db.get(user["email"], {})
    uid = (full_user or {}).get("customer_id") or user.get("customer_id") or user.get("user_id") or ""
    if uid != customer_id:
        raise HTTPException(403, "You can only claim credits for your own account")

    idempotency_key = f"free-credits-{customer_id}"
    be = get_billing_engine()
    result = be.deposit(
        customer_id,
        _FREE_CREDIT_AMOUNT,
        "Welcome bonus — $10 free credits",
        idempotency_key=idempotency_key,
    )
    already_claimed = result.get("dedup", False)
    return {
        "ok": True,
        "amount_cad": _FREE_CREDIT_AMOUNT,
        "balance_cad": result["balance_cad"],
        "already_claimed": already_claimed,
    }

@router.get("/api/billing/free-credits/{customer_id}/status", tags=["Billing"])
def api_free_credits_status(customer_id: str, request: Request):
    """Check whether the customer has already claimed the free signup bonus."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Authentication required")
    # Resolve customer_id from full user profile (session may lack it)
    full_user = UserStore.get_user(user["email"]) if _USE_PERSISTENT_AUTH else _users_db.get(user["email"], {})
    uid = (full_user or {}).get("customer_id") or user.get("customer_id") or user.get("user_id") or ""
    if uid != customer_id:
        raise HTTPException(403, "Forbidden")
    be = get_billing_engine()
    be._ensure_wallet_table()
    with be._conn() as conn:
        row = conn.execute(
            "SELECT tx_id FROM wallet_transactions WHERE idempotency_key = %s",
            (f"free-credits-{customer_id}",),
        ).fetchone()
    return {"ok": True, "claimed": row is not None}

@router.get("/api/billing/wallet/{customer_id}/history", tags=["Billing"])
def api_wallet_history(customer_id: str, limit: int = 50):
    """Get transaction history for a wallet."""
    be = get_billing_engine()
    history = be.get_wallet_history(customer_id, limit)
    return {"ok": True, "customer_id": customer_id, "transactions": history}

@router.get("/api/billing/wallet/{customer_id}/depletion", tags=["Billing"])
def api_wallet_depletion(customer_id: str):
    """Get real-time balance depletion projection.

    Returns burn rate, seconds-to-zero, per-instance cost breakdown,
    and alert thresholds (T-30min, T-5min, T-0).
    """
    be = get_billing_engine()
    return {"ok": True, **be.time_to_zero(customer_id)}

@router.get("/api/billing/usage/{customer_id}", tags=["Billing"])
def api_usage_summary(customer_id: str, period_start: float = 0, period_end: float = 0):
    """Get usage summary for a customer."""
    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400  # Last 30 days
    be = get_billing_engine()
    summary = be.get_usage_summary(customer_id, period_start, period_end)
    return {"ok": True, **summary}

@router.get("/api/billing/invoice/{customer_id}", tags=["Billing"])
def api_generate_invoice(
    customer_id: str,
    customer_name: str = "",
    period_start: float = 0,
    period_end: float = 0,
    tax_rate: float = 0.13,
):
    """Generate an AI Compute Access Fund–aligned invoice."""
    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400
    be = get_billing_engine()
    invoice = be.generate_invoice(customer_id, customer_name, period_start, period_end, tax_rate)
    return {"ok": True, "invoice": invoice.to_dict()}

@router.get("/api/billing/export/caf/{customer_id}", tags=["Billing"])
def api_export_caf(
    customer_id: str, period_start: float = 0, period_end: float = 0, format: str = "json"
):
    """Export AI Compute Access Fund rebate documentation.

    From REPORT_FEATURE_2.md: /billing/export?format=caf
    Supports json and csv formats.
    """
    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400
    be = get_billing_engine()

    if format == "csv":
        csv_data = be.export_caf_csv(customer_id, period_start, period_end)
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=xcelsior-caf-{customer_id}.csv"},
        )

    report = be.export_caf_report(customer_id, period_start, period_end)
    return {"ok": True, **report}

@router.get("/api/billing/invoices/{customer_id}", tags=["Billing"])
def api_list_invoices(customer_id: str, limit: int = 12):
    """List past invoices for a customer (monthly summaries).

    Generates monthly invoice stubs for the last N months showing
    total spend, tax, job count, and top GPUs used.
    """
    be = get_billing_engine()
    now = time.time()
    invoices = []
    for i in range(limit):
        period_end = now - (i * 30 * 86400)
        period_start = period_end - 30 * 86400
        try:
            inv = be.generate_invoice(customer_id, "", period_start, period_end, 0.13)
            inv_dict = inv.to_dict()
            # Only include months with actual usage
            if inv_dict.get("total_compute_cad", 0) > 0 or inv_dict.get("line_items"):
                invoices.append(
                    {
                        "invoice_id": f"INV-{customer_id[:8]}-{i+1:03d}",
                        "period_start": period_start,
                        "period_end": period_end,
                        "total_cad": inv_dict.get(
                            "total_with_tax_cad", inv_dict.get("total_compute_cad", 0)
                        ),
                        "subtotal_cad": inv_dict.get("total_compute_cad", 0),
                        "tax_cad": inv_dict.get("tax_cad", 0),
                        "tax_rate": inv_dict.get("tax_rate", 0.13),
                        "line_items": len(inv_dict.get("line_items", [])),
                        "caf_eligible_cad": inv_dict.get("caf_eligible_cad", 0),
                        "status": "paid",
                    }
                )
        except Exception as e:
            log.debug("invoice formatting failed: %s", e)
    return {"ok": True, "invoices": invoices, "count": len(invoices)}

@router.get("/api/billing/invoice/{customer_id}/download", tags=["Billing"])
def api_download_invoice(
    customer_id: str,
    format: str = "csv",
    period_start: float = 0,
    period_end: float = 0,
    tax_rate: float = 0.13,
    customer_name: str = "",
):
    """Download an invoice as CSV or plain-text PDF-style document.

    Formats: csv (spreadsheet-ready), txt (printable receipt).
    """
    import io
    import csv as csv_mod
    from datetime import datetime

    if period_end == 0:
        period_end = time.time()
    if period_start == 0:
        period_start = period_end - 30 * 86400

    be = get_billing_engine()
    inv = be.generate_invoice(customer_id, customer_name, period_start, period_end, tax_rate)
    inv_dict = inv.to_dict()
    date_str = datetime.utcfromtimestamp(period_end).strftime("%Y-%m-%d")

    if format == "csv":
        output = io.StringIO()
        writer = csv_mod.writer(output)
        writer.writerow(["Xcelsior Invoice", f"INV-{customer_id[:8]}", date_str])
        writer.writerow([])
        writer.writerow(["Description", "GPU", "Duration (h)", "Rate (CAD/h)", "Amount (CAD)"])
        for item in inv_dict.get("line_items", []):
            writer.writerow(
                [
                    item.get("description", "Compute"),
                    item.get("gpu_model", "—"),
                    round(item.get("duration_hours", 0), 2),
                    round(item.get("rate_cad_per_hour", 0), 2),
                    round(item.get("amount_cad", 0), 2),
                ]
            )
        writer.writerow([])
        writer.writerow(["Subtotal", "", "", "", round(inv_dict.get("total_compute_cad", 0), 2)])
        writer.writerow(["Tax", "", "", "", round(inv_dict.get("tax_cad", 0), 2)])
        writer.writerow(
            ["Total (CAD)", "", "", "", round(inv_dict.get("total_with_tax_cad", 0), 2)]
        )
        writer.writerow(["CAF Eligible", "", "", "", round(inv_dict.get("caf_eligible_cad", 0), 2)])
        csv_data = output.getvalue()
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=xcelsior-invoice-{customer_id[:8]}-{date_str}.csv"
            },
        )

    # Plain-text receipt format
    lines = [
        "=" * 60,
        "XCELSIOR — GPU COMPUTE INVOICE",
        "=" * 60,
        f"Invoice ID:  INV-{customer_id[:8]}",
        f"Customer:    {customer_name or customer_id}",
        f"Date:        {date_str}",
        f"Period:      {datetime.utcfromtimestamp(period_start).strftime('%Y-%m-%d')} to {date_str}",
        "-" * 60,
        f"{'Description':<25} {'GPU':<12} {'Hours':>6} {'Rate':>8} {'Amount':>10}",
        "-" * 60,
    ]
    for item in inv_dict.get("line_items", []):
        lines.append(
            f"{item.get('description', 'Compute')[:25]:<25} "
            f"{item.get('gpu_model', '—')[:12]:<12} "
            f"{item.get('duration_hours', 0):>6.2f} "
            f"${item.get('rate_cad_per_hour', 0):>7.2f} "
            f"${item.get('amount_cad', 0):>9.2f}"
        )
    lines += [
        "-" * 60,
        f"{'Subtotal':<55} ${inv_dict.get('total_compute_cad', 0):>8.2f}",
        f"{'Tax (' + str(round(tax_rate*100, 1)) + '%)':<55} ${inv_dict.get('tax_cad', 0):>8.2f}",
        f"{'TOTAL (CAD)':<55} ${inv_dict.get('total_with_tax_cad', 0):>8.2f}",
        "",
        f"AI Compute Access Fund Eligible: ${inv_dict.get('caf_eligible_cad', 0):.2f} CAD",
        "=" * 60,
        "Xcelsior Inc. | xcelsior.ca | Built in Canada 🍁",
    ]
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=xcelsior-invoice-{customer_id[:8]}-{date_str}.txt"
        },
    )

@router.get("/api/billing/attestation", tags=["Billing"])
def api_provider_attestation():
    """Get Xcelsior supplier attestation bundle for Fund claims."""
    be = get_billing_engine()
    attestation = be.generate_attestation()
    return {"ok": True, "attestation": attestation.to_dict()}


# ── Model: RefundRequest ──

class RefundRequest(BaseModel):
    job_id: str
    exit_code: int
    failure_reason: str = ""

@router.post("/api/billing/refund", tags=["Billing"])
def api_process_refund(req: RefundRequest):
    """Process a refund for a failed job.

    From REPORT_FEATURE_1.md:
    - Hardware error → full refund
    - User OOM (exit 137) → zero refund
    """
    be = get_billing_engine()
    result = be.process_refund(req.job_id, req.exit_code, req.failure_reason)
    return {"ok": True, **result}


# ── Model: CryptoDepositRequest ──

class CryptoDepositRequest(BaseModel):
    customer_id: str
    amount_cad: float

@router.post("/api/billing/crypto/deposit", tags=["Billing"])
def api_crypto_deposit(req: CryptoDepositRequest):
    """Create a BTC deposit request. Returns address, amount, and QR data."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    service_status = _btc_mod.get_service_status()
    if not service_status.get("available", False):
        raise HTTPException(
            503,
            service_status.get("reason") or "Bitcoin service is currently unavailable",
        )
    if req.amount_cad < 1 or req.amount_cad > 10000:
        raise HTTPException(400, "Amount must be between $1 and $10,000 CAD")
    try:
        result = _btc_mod.create_deposit(req.customer_id, req.amount_cad)
        return {"ok": True, **result}
    except Exception as e:
        log.error("Crypto deposit error: %s", e)
        detail = _btc_mod.describe_service_error(e)
        raise HTTPException(503, detail)

@router.get("/api/billing/crypto/deposit/{deposit_id}", tags=["Billing"])
def api_crypto_deposit_status(deposit_id: str):
    """Poll deposit confirmation status."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    dep = _btc_mod.get_deposit(deposit_id)
    if not dep:
        raise HTTPException(404, "Deposit not found")
    return {"ok": True, **dep}

@router.get("/api/billing/crypto/rate", tags=["Billing"])
def api_crypto_rate():
    """Get current BTC/CAD exchange rate."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    try:
        rate = _btc_mod.get_btc_cad_rate()
        return {"ok": True, "btc_cad": rate, "currency": "CAD"}
    except Exception as e:
        raise HTTPException(502, f"Unable to fetch rate: {e}")

@router.post("/api/billing/crypto/refresh/{deposit_id}", tags=["Billing"])
def api_crypto_refresh(deposit_id: str):
    """Refresh an expired deposit with a new BTC/CAD rate."""
    if not _btc_mod or not _btc_mod.BTC_ENABLED:
        raise HTTPException(503, "Bitcoin deposits are not enabled")
    dep = _btc_mod.refresh_deposit(deposit_id)
    if not dep:
        raise HTTPException(404, "Deposit not found")
    return {"ok": True, **dep}

@router.get("/api/billing/crypto/enabled", tags=["Billing"])
def api_crypto_enabled():
    """Check if Bitcoin deposits are enabled."""
    if not _btc_mod:
        return {
            "ok": True,
            "enabled": False,
            "available": False,
            "reason": "Bitcoin deposits are not enabled",
        }
    return {"ok": True, **_btc_mod.get_service_status()}


# ── Model: EstimateRequest ──

class EstimateRequest(BaseModel):
    gpu_model: str = "RTX 4090"
    duration_hours: float = 1.0
    spot: bool = False
    sovereignty: bool = False
    is_canadian: bool = True

@router.post("/api/pricing/estimate", tags=["Billing"])
def api_estimate_cost(req: EstimateRequest):
    """Estimate job cost with AI Compute Access Fund rebate preview.

    From REPORT_FEATURE_2.md: --estimate-rebate / simulate=true
    """
    estimate = estimate_job_cost(
        req.gpu_model,
        req.duration_hours,
        spot=req.spot,
        sovereignty=req.sovereignty,
        is_canadian=req.is_canadian,
    )
    return {"ok": True, **estimate}

@router.get("/api/pricing/reference", tags=["Billing"])
def api_reference_pricing():
    """Get reference GPU pricing table in CAD."""
    return {"ok": True, "currency": "CAD", "pricing": GPU_REFERENCE_PRICING_CAD}


# ── Model: ReservedCommitmentRequest ──

class ReservedCommitmentRequest(BaseModel):
    customer_id: str
    gpu_model: str = "RTX 4090"
    commitment_type: str = "1_month"  # 1_month | 3_month | 1_year
    quantity: int = 1  # number of GPU slots reserved
    province: str = "ON"

@router.get("/api/pricing/reserved-plans", tags=["Billing"])
def api_reserved_plans():
    """List available reserved pricing tiers with discount percentages.

    Three commitment levels:
    - **1_month**: 20% discount, minimum 4 hrs/day usage
    - **3_month**: 30% discount, minimum 4 hrs/day usage
    - **1_year**: 45% discount, no minimum daily usage

    Compare with on-demand (`POST /job`) and spot/interruptible (`POST /spot/job`).
    """
    # Enrich each tier with sample pricing based on reference GPU pricing
    enriched = {}
    for tier_key, tier in RESERVED_PRICING_TIERS.items():
        samples = {}
        for gpu, ref in GPU_REFERENCE_PRICING_CAD.items():
            rate = (
                ref.get("base_rate_cad", ref.get("cad_per_hour", 0))
                if isinstance(ref, dict)
                else ref
            )
            samples[gpu] = round(rate * (1 - tier["discount_pct"] / 100), 4)
        enriched[tier_key] = {**tier, "sample_hourly_rates_cad": samples}
    return {"ok": True, "currency": "CAD", "reserved_tiers": enriched}

@router.post("/api/pricing/reserve", tags=["Billing"])
def api_reserve_commitment(req: ReservedCommitmentRequest):
    """Create a reserved pricing commitment for a customer.

    Reserved instances are 20-45% cheaper than on-demand, depending on
    commitment length. The customer pre-commits to a term and receives
    a guaranteed discount on all GPU hours consumed during that period.
    """
    tier = RESERVED_PRICING_TIERS.get(req.commitment_type)
    if not tier:
        raise HTTPException(
            400,
            f"Invalid commitment_type: {req.commitment_type}. "
            f"Valid: {list(RESERVED_PRICING_TIERS.keys())}",
        )

    # Calculate pricing
    ref_pricing = GPU_REFERENCE_PRICING_CAD.get(req.gpu_model, {})
    base_rate = (
        ref_pricing.get("base_rate_cad", ref_pricing.get("cad_per_hour", 0))
        if isinstance(ref_pricing, dict)
        else (ref_pricing if isinstance(ref_pricing, (int, float)) else 0)
    )
    if base_rate <= 0:
        raise HTTPException(400, f"Unknown GPU model: {req.gpu_model}")

    discounted_rate = round(base_rate * (1 - tier["discount_pct"] / 100), 4)
    tax_rate, tax_desc = get_tax_rate_for_province(req.province)

    commitment = {
        "commitment_id": str(uuid.uuid4()),
        "customer_id": req.customer_id,
        "commitment_type": req.commitment_type,
        "gpu_model": req.gpu_model,
        "quantity": req.quantity,
        "base_rate_cad": base_rate,
        "discounted_rate_cad": discounted_rate,
        "discount_pct": tier["discount_pct"],
        "province": req.province,
        "tax_rate": tax_rate,
        "tax_description": tax_desc,
        "commitment_description": tier["description"],
        "min_hours_per_day": tier["min_hours_per_day"],
        "created_at": time.time(),
        "status": "active",
    }

    # Charge upfront or set up recurring billing via Stripe
    billing = get_billing_engine()
    monthly_estimate = discounted_rate * req.quantity * 24 * 30
    commitment["monthly_estimate_cad"] = round(monthly_estimate, 2)
    commitment["monthly_estimate_with_tax_cad"] = round(monthly_estimate * (1 + tax_rate), 2)

    broadcast_sse(
        "reservation_created",
        {
            "commitment_id": commitment["commitment_id"],
            "customer_id": req.customer_id,
            "type": req.commitment_type,
        },
    )
    return {"ok": True, **commitment}

@router.get("/api/analytics/usage", tags=["Billing"])
def api_usage_analytics(
    request: Request,
    customer_id: str = "",
    provider_id: str = "",
    days: int = 30,
    offset_days: int = 0,
    group_by: str = "day",  # day | week | gpu_model | province
):
    """Usage analytics for both providers and submitters.

    Provides cost breakdowns, GPU utilization trends, and hardware health
    aggregates over time. Supports grouping by day, week, GPU model,
    or province for detailed reporting.

    Non-admin users are automatically scoped to their own data.

    Query params:
    - `customer_id` — filter to one customer (submitter view, admin only)
    - `provider_id` — filter to one provider (earnings view, admin only)
    - `days` — lookback window (default 30)
    - `group_by` — aggregation: `day`, `week`, `gpu_model`, `province`
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    # Non-admin users are scoped to their own data automatically
    is_admin = bool(user.get("is_admin"))
    if not is_admin:
        customer_id = user["email"]
        provider_id = ""  # provider filtering requires admin

    billing = get_billing_engine()
    now = time.time()
    safe_days = max(1, min(days, 3650))
    safe_offset_days = max(0, min(offset_days, 3650))
    window_end = now - (safe_offset_days * 86400)
    since = window_end - (safe_days * 86400)

    _GROUP_SQL = {
        "day": "to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS period",
        "week": "to_char(to_timestamp(started_at), 'IYYY-\"W\"IW') AS period",
        "gpu_model": "gpu_model AS period",
        "province": "province AS period",
    }
    if group_by not in _GROUP_SQL:
        group_by = "day"
    group_sql = _GROUP_SQL[group_by]

    where_clauses = ["started_at >= %s", "started_at < %s"]
    params: list = [since, window_end]
    if customer_id:
        where_clauses.append("owner = %s")
        params.append(customer_id)
    # Provider filter: match host_id (providers are hosts)
    if provider_id:
        where_clauses.append("host_id = %s")
        params.append(provider_id)

    where_sql = " AND ".join(where_clauses)

    try:
        with billing._conn() as conn:
            rows = conn.execute(
                f"SELECT {group_sql}, "
                "COUNT(*) AS job_count, "
                "ROUND(SUM(total_cost_cad), 2) AS total_cost_cad, "
                "ROUND(SUM(gpu_seconds), 0) AS total_gpu_seconds, "
                "ROUND(AVG(gpu_utilization_pct), 1) AS avg_gpu_util_pct, "
                "SUM(is_canadian_compute) AS canadian_jobs, "
                "COUNT(*) - SUM(is_canadian_compute) AS international_jobs "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY period ORDER BY period",
                params,
            ).fetchall()

            analytics = [
                {
                    "period": r["period"],
                    "job_count": r["job_count"],
                    "total_cost_cad": r["total_cost_cad"],
                    "total_gpu_hours": (
                        round(r["total_gpu_seconds"] / 3600, 2) if r["total_gpu_seconds"] else 0
                    ),
                    "avg_gpu_utilization_pct": r["avg_gpu_util_pct"],
                    "canadian_jobs": r["canadian_jobs"],
                    "international_jobs": r["international_jobs"],
                }
                for r in rows
            ]

            # Summary
            summary_row = conn.execute(
                "SELECT COUNT(*) AS total_jobs, "
                "ROUND(SUM(total_cost_cad), 2) AS total_spend, "
                "ROUND(SUM(gpu_seconds) / 3600.0, 2) AS total_gpu_hours, "
                "ROUND(AVG(gpu_utilization_pct), 1) AS avg_util "
                f"FROM usage_meters WHERE {where_sql}",
                params,
            ).fetchone()
    except Exception as e:
        return {"ok": False, "error": str(e), "analytics": [], "summary": {}}

    return {
        "ok": True,
        "days": safe_days,
        "offset_days": safe_offset_days,
        "group_by": group_by,
        "analytics": analytics,
        "summary": {
            "total_jobs": summary_row["total_jobs"] if summary_row else 0,
            "total_spend_cad": summary_row["total_spend"] if summary_row else 0,
            "total_gpu_hours": summary_row["total_gpu_hours"] if summary_row else 0,
            "avg_gpu_utilization_pct": summary_row["avg_util"] if summary_row else 0,
        },
    }


@router.get("/api/analytics/enhanced", tags=["Billing"])
def api_usage_analytics_enhanced(
    request: Request,
    customer_id: str = "",
    provider_id: str = "",
    days: int = 30,
    offset_days: int = 0,
):
    """Enhanced usage analytics with additional aggregations for dashboards."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    is_admin = bool(user.get("is_admin"))
    if not is_admin:
        customer_id = user["email"]
        provider_id = ""

    billing = get_billing_engine()
    now = time.time()
    safe_days = max(1, min(days, 3650))
    safe_offset_days = max(0, min(offset_days, 3650))
    window_end = now - (safe_offset_days * 86400)
    since = window_end - (safe_days * 86400)

    where_clauses = ["started_at >= %s", "started_at < %s"]
    params: list = [since, window_end]
    if customer_id:
        where_clauses.append("owner = %s")
        params.append(customer_id)
    if provider_id:
        where_clauses.append("host_id = %s")
        params.append(provider_id)

    where_sql = " AND ".join(where_clauses)

    try:
        with billing._conn() as conn:
            avg_cost_rows = conn.execute(
                "SELECT to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS date, "
                "ROUND(SUM(total_cost_cad) / NULLIF(SUM(gpu_seconds) / 3600.0, 0), 4) AS cost_per_hour "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY date ORDER BY date",
                params,
            ).fetchall()

            cumulative_rows = conn.execute(
                "SELECT date, running_total FROM ("
                "SELECT to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS date, "
                "ROUND(SUM(total_cost_cad), 2) AS daily_spend, "
                "ROUND(SUM(SUM(total_cost_cad)) OVER (ORDER BY to_char(to_timestamp(started_at), 'YYYY-MM-DD')), 2) AS running_total "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY date"
                ") s ORDER BY date",
                params,
            ).fetchall()

            histogram_rows = conn.execute(
                "SELECT bucket, COUNT(*) AS count FROM ("
                "SELECT CASE "
                "WHEN gpu_seconds < 60 THEN '< 1min' "
                "WHEN gpu_seconds < 300 THEN '1-5min' "
                "WHEN gpu_seconds < 1800 THEN '5-30min' "
                "WHEN gpu_seconds < 3600 THEN '30min-1h' "
                "ELSE '1h+' END AS bucket "
                f"FROM usage_meters WHERE {where_sql}"
                ") d GROUP BY bucket",
                params,
            ).fetchall()

            top_hosts_rows = conn.execute(
                "SELECT host_id, COUNT(*) AS job_count, ROUND(SUM(total_cost_cad), 2) AS total_cost "
                f"FROM usage_meters WHERE {where_sql} AND host_id IS NOT NULL "
                "GROUP BY host_id ORDER BY job_count DESC, total_cost DESC LIMIT 10",
                params,
            ).fetchall()

            daily_hours_rows = conn.execute(
                "SELECT to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS date, "
                "ROUND(SUM(gpu_seconds) / 3600.0, 3) AS hours "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY date ORDER BY date",
                params,
            ).fetchall()

            # Some deployments may not yet have a `status` column on usage_meters.
            # Keep endpoint resilient by degrading this one aggregation only.
            try:
                status_row = conn.execute(
                    "SELECT "
                    "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed, "
                    "SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed, "
                    "SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled "
                    f"FROM usage_meters WHERE {where_sql}",
                    params,
                ).fetchone()
            except Exception:
                status_row = {"completed": 0, "failed": 0, "cancelled": 0}
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "avg_cost_per_hour_trend": [],
            "cumulative_spend": [],
            "duration_histogram": [],
            "job_status_breakdown": {"completed": 0, "failed": 0, "cancelled": 0},
            "top_hosts_used": [],
            "daily_gpu_hours": [],
        }

    order = {"< 1min": 0, "1-5min": 1, "5-30min": 2, "30min-1h": 3, "1h+": 4}
    histogram_sorted = sorted(
        [{"bucket": r["bucket"], "count": int(r["count"] or 0)} for r in histogram_rows],
        key=lambda r: order.get(r["bucket"], 99),
    )

    return {
        "ok": True,
        "days": safe_days,
        "offset_days": safe_offset_days,
        "avg_cost_per_hour_trend": [
            {"date": r["date"], "cost_per_hour": float(r["cost_per_hour"] or 0)} for r in avg_cost_rows
        ],
        "cumulative_spend": [
            {"date": r["date"], "running_total": float(r["running_total"] or 0)} for r in cumulative_rows
        ],
        "duration_histogram": histogram_sorted,
        "job_status_breakdown": {
            "completed": int((status_row["completed"] if status_row else 0) or 0),
            "failed": int((status_row["failed"] if status_row else 0) or 0),
            "cancelled": int((status_row["cancelled"] if status_row else 0) or 0),
        },
        "top_hosts_used": [
            {
                "host_id": r["host_id"],
                "job_count": int(r["job_count"] or 0),
                "total_cost": float(r["total_cost"] or 0),
            }
            for r in top_hosts_rows
        ],
        "daily_gpu_hours": [
            {"date": r["date"], "hours": float(r["hours"] or 0)} for r in daily_hours_rows
        ],
    }


# ── Model: AutoTopupConfig ──

class AutoTopupConfig(BaseModel):
    enabled: bool = True
    amount_cad: float = 25.0
    threshold_cad: float = 5.0
    stripe_payment_method_id: str = ""

@router.post("/api/v2/billing/auto-topup", tags=["Billing"])
def api_billing_configure_topup(body: AutoTopupConfig, request: Request):
    """Configure wallet auto-top-up via Stripe."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    be = get_billing_engine()
    customer_id = user.get("customer_id", user.get("user_id", user.get("email", "")))
    be.configure_auto_topup(
        customer_id=customer_id,
        enabled=body.enabled,
        amount_cad=body.amount_cad,
        threshold_cad=body.threshold_cad,
        payment_method_id=body.stripe_payment_method_id,
    )
    return {"ok": True, "auto_topup": body.model_dump()}

@router.get("/api/v2/billing/auto-topup", tags=["Billing"])
def api_billing_get_topup(request: Request):
    """Get current auto-top-up configuration."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    be = get_billing_engine()
    customer_id = user.get("customer_id", user.get("user_id", user.get("email", "")))
    wallet = be.get_or_create_wallet(customer_id)
    return {
        "ok": True,
        "auto_topup": {
            "enabled": bool(wallet.get("auto_topup_enabled", False)),
            "amount_cad": wallet.get("auto_topup_amount", 0),
            "threshold_cad": wallet.get("auto_topup_threshold", 0),
        },
    }
