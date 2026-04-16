"""Routes: billing."""

import httpx
import io
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from routes._deps import (
    XCELSIOR_ENV,
    _USE_PERSISTENT_AUTH,
    _get_current_user,
    _merge_auth_user,
    _require_admin,
    _require_auth,
    _require_scope,
    _users_db,
    broadcast_sse,
    log,
    otel_span,
)
from scheduler import (
    bill_all_completed,
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

try:
    import lightning as _ln_mod
except ImportError:
    _ln_mod = None  # type: ignore[assignment]


def _analytics_customer_scope(user: dict) -> str:
    """Return the billing owner identifier for the authenticated user.

    Usage meters and wallet records are keyed by customer_id, not email.
    Falling back preserves older/test flows that may still use user_id/email.
    """
    return str(
        user.get("customer_id")
        or user.get("user_id")
        or user.get("email")
        or ""
    ).strip()

@router.post("/billing/bill/{job_id}", tags=["Billing"])
def api_bill_instance(job_id: str, request: Request):
    """Bill a specific completed job."""
    user = _require_auth(request)
    _require_scope(user, "billing:write")
    with otel_span("billing.bill_job", {"job.id": job_id}):
        record = bill_job(job_id)
        if not record:
            raise HTTPException(status_code=400, detail=f"Could not bill job {job_id}")
        return {"ok": True, "bill": record}

@router.post("/billing/bill-all", tags=["Billing"])
def api_bill_all(request: Request):
    """Bill all unbilled completed jobs."""
    user = _require_auth(request)
    _require_scope(user, "billing:write")
    try:
        bills = bill_all_completed()
    except Exception as exc:
        log.error("bill_all_completed failed: %s", exc)
        raise HTTPException(status_code=500, detail="Billing run failed") from exc
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

# ── PayPal ───────────────────────────────────────────────────────────────

_PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID", "")
_PAYPAL_CLIENT_SECRET = os.environ.get("PAYPAL_CLIENT_SECRET", "")
_PAYPAL_MODE = os.environ.get("PAYPAL_MODE", "sandbox")  # sandbox | live
_PAYPAL_BASE = (
    "https://api-m.paypal.com"
    if _PAYPAL_MODE == "live"
    else "https://api-m.sandbox.paypal.com"
)


def _paypal_access_token() -> str:
    """Get a PayPal OAuth2 access token (short-lived, not cached)."""
    resp = httpx.post(
        f"{_PAYPAL_BASE}/v1/oauth2/token",
        data={"grant_type": "client_credentials"},
        auth=(_PAYPAL_CLIENT_ID, _PAYPAL_CLIENT_SECRET),
        headers={"Accept": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


class PayPalCreateOrderRequest(BaseModel):
    customer_id: str
    amount_cad: float


class PayPalCaptureRequest(BaseModel):
    customer_id: str
    order_id: str


@router.get("/api/billing/paypal/enabled", tags=["Billing"])
def api_paypal_enabled():
    """Check whether PayPal is configured."""
    return {"enabled": bool(_PAYPAL_CLIENT_ID and _PAYPAL_CLIENT_SECRET)}


@router.post("/api/billing/paypal/create-order", tags=["Billing"])
def api_paypal_create_order(req: PayPalCreateOrderRequest, request: Request):
    """Create a PayPal order for depositing compute credits."""
    _require_auth(request)
    if not _PAYPAL_CLIENT_ID or not _PAYPAL_CLIENT_SECRET:
        raise HTTPException(503, "PayPal is not configured")
    if req.amount_cad < 5 or req.amount_cad > 10000:
        raise HTTPException(400, "Amount must be between $5 and $10,000 CAD")

    token = _paypal_access_token()
    order_resp = httpx.post(
        f"{_PAYPAL_BASE}/v2/checkout/orders",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "amount": {
                        "currency_code": "CAD",
                        "value": f"{req.amount_cad:.2f}",
                    },
                    "description": f"Xcelsior compute credits — {req.customer_id}",
                    "custom_id": req.customer_id,
                }
            ],
            "application_context": {
                "brand_name": "Xcelsior",
                "shipping_preference": "NO_SHIPPING",
            },
        },
        timeout=15,
    )
    if order_resp.status_code >= 400:
        log.error("PayPal create-order failed: %s", order_resp.text)
        raise HTTPException(502, "PayPal order creation failed")

    data = order_resp.json()
    return {"ok": True, "order_id": data["id"]}


@router.post("/api/billing/paypal/capture-order", tags=["Billing"])
def api_paypal_capture_order(req: PayPalCaptureRequest, request: Request):
    """Capture a PayPal order and credit the wallet."""
    user = _require_auth(request)
    if not _PAYPAL_CLIENT_ID or not _PAYPAL_CLIENT_SECRET:
        raise HTTPException(503, "PayPal is not configured")

    token = _paypal_access_token()
    cap_resp = httpx.post(
        f"{_PAYPAL_BASE}/v2/checkout/orders/{req.order_id}/capture",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    if cap_resp.status_code >= 400:
        log.error("PayPal capture failed: %s", cap_resp.text)
        raise HTTPException(502, "PayPal capture failed")

    data = cap_resp.json()
    if data.get("status") != "COMPLETED":
        raise HTTPException(400, f"PayPal order status: {data.get('status', 'unknown')}")

    # Extract captured amount
    capture = data["purchase_units"][0]["payments"]["captures"][0]
    amount_cad = float(capture["amount"]["value"])
    paypal_id = capture["id"]

    # Credit wallet
    be = get_billing_engine()
    result = be.deposit(
        req.customer_id,
        amount_cad,
        f"PayPal deposit ({paypal_id})",
        idempotency_key=f"paypal-{req.order_id}",
    )
    log.info("PayPal deposit: %s → $%.2f CAD (capture %s)", req.customer_id, amount_cad, paypal_id)
    return {"ok": True, "balance_cad": result["balance_cad"], "amount_cad": amount_cad}


@router.post("/api/billing/wallet/{customer_id}/deposit", tags=["Billing"])
def api_deposit(customer_id: str, req: DepositRequest):
    """Deposit credits into a customer wallet."""
    be = get_billing_engine()
    result = be.deposit(customer_id, req.amount_cad, req.description)
    return {"ok": True, **result}

@router.post("/api/billing/wallet/{customer_id}/reset-testing", tags=["Billing"])
def api_reset_wallet_testing_state(customer_id: str, request: Request):
    """Reset wallet balance and promo state for admin testing."""
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


# ── Lightning Network Deposits ────────────────────────────────────────


class LnDepositRequest(BaseModel):
    customer_id: str
    amount_cad: float


@router.get("/api/billing/lightning/enabled", tags=["Billing"])
def api_ln_enabled():
    """Check if Lightning deposits are enabled and node is reachable."""
    if not _ln_mod:
        return {"ok": True, "enabled": False, "available": False, "reason": "Lightning module not available"}
    return {"ok": True, **_ln_mod.get_service_status()}


@router.post("/api/billing/lightning/deposit", tags=["Billing"])
def api_ln_create_deposit(req: LnDepositRequest, request: Request):
    """Create a Lightning invoice for depositing CAD credits."""
    user = _require_auth(request)
    _require_scope(user, "billing:write")
    if not _ln_mod or not _ln_mod.LN_ENABLED:
        raise HTTPException(503, "Lightning deposits are not enabled")
    try:
        result = _ln_mod.create_deposit(req.customer_id, req.amount_cad)
        broadcast_sse("ln_deposit_created", {"deposit_id": result["deposit_id"], "customer_id": req.customer_id})
        return {"ok": True, **result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.get("/api/billing/lightning/deposit/{deposit_id}", tags=["Billing"])
def api_ln_check_deposit(deposit_id: str, request: Request):
    """Check the status of a Lightning deposit."""
    user = _require_auth(request)
    _require_scope(user, "billing:read")
    if not _ln_mod or not _ln_mod.LN_ENABLED:
        raise HTTPException(503, "Lightning deposits are not enabled")
    dep = _ln_mod.check_deposit(deposit_id)
    if not dep:
        raise HTTPException(404, "Deposit not found")
    return {"ok": True, **dep}


@router.get("/api/billing/lightning/rate", tags=["Billing"])
def api_ln_rate():
    """Get current BTC/CAD rate for Lightning deposits."""
    if not _ln_mod:
        raise HTTPException(503, "Lightning module not available")
    try:
        rate = _ln_mod.get_btc_cad_rate()
        return {"ok": True, "btc_cad": rate, "currency": "CAD"}
    except RuntimeError as e:
        raise HTTPException(503, str(e))


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


@router.get("/api/pricing/rates", tags=["Billing"])
def api_pricing_rates(
    gpu_model: str = "RTX 4090",
    tier: str = "standard",
    mode: str = "on_demand",
    priority: str = "normal",
    num_gpus: int = 1,
    province: str = "ON",
):
    """Compute effective GPU rate with all pricing variables.

    Returns a full breakdown: base rate, priority multiplier, sovereignty
    premium, multi-GPU discount, tax rate, and final per-hour total.
    Volume (storage) costs are calculated at runtime in billing — not here.
    """
    from db import pg_connection, GPU_PRIORITY_MULTIPLIERS

    # Validate priority
    priority = priority.lower()
    pri_mult = GPU_PRIORITY_MULTIPLIERS.get(priority, 1.0)

    with pg_connection() as conn:
        cur = conn.execute(
            """SELECT base_rate_cad, sovereignty_premium,
                      multi_gpu_discount_4, multi_gpu_discount_8, vram_gb
               FROM gpu_pricing
               WHERE gpu_model = %s AND tier = %s AND pricing_mode = %s
                     AND active = TRUE
               LIMIT 1""",
            (gpu_model, tier, mode),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"No pricing for {gpu_model}/{tier}/{mode}")

    base_rate = row[0]
    sovereignty_premium = row[1]
    mg4_disc = row[2]
    mg8_disc = row[3]
    vram_gb = row[4]

    # Multi-GPU discount
    if num_gpus >= 8:
        multi_gpu_discount = mg8_disc
    elif num_gpus >= 4:
        multi_gpu_discount = mg4_disc
    else:
        multi_gpu_discount = 0.0

    # Effective per-GPU rate
    effective_rate = base_rate * pri_mult * (1 + sovereignty_premium) * (1 - multi_gpu_discount)
    effective_rate = round(effective_rate, 4)

    # Per-hour total across all GPUs
    total_per_hour = round(effective_rate * num_gpus, 4)

    # Tax
    tax_rate, tax_desc = get_tax_rate_for_province(province)
    tax_amount = round(total_per_hour * tax_rate, 4)
    total_with_tax = round(total_per_hour + tax_amount, 4)

    return {
        "ok": True,
        "currency": "CAD",
        "gpu_model": gpu_model,
        "vram_gb": vram_gb,
        "tier": tier,
        "pricing_mode": mode,
        "priority": priority,
        "num_gpus": num_gpus,
        "base_rate_cad": base_rate,
        "priority_multiplier": pri_mult,
        "sovereignty_premium": sovereignty_premium,
        "multi_gpu_discount": multi_gpu_discount,
        "effective_rate_per_gpu": effective_rate,
        "total_per_hour": total_per_hour,
        "province": province,
        "tax_rate": tax_rate,
        "tax_description": tax_desc,
        "tax_amount": tax_amount,
        "total_with_tax": total_with_tax,
    }


@router.get("/api/pricing/models", tags=["Billing"])
def api_pricing_models():
    """List all available GPU models with on-demand standard pricing."""
    from db import pg_connection

    with pg_connection() as conn:
        cur = conn.execute(
            """SELECT DISTINCT gpu_model, vram_gb, base_rate_cad
               FROM gpu_pricing
               WHERE active = TRUE AND tier = 'standard' AND pricing_mode = 'on_demand'
               ORDER BY base_rate_cad ASC"""
        )
        rows = cur.fetchall()

    models = [{"gpu_model": r[0], "vram_gb": r[1], "base_rate_cad": r[2]} for r in rows]
    return {"ok": True, "models": models}


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
    user = _require_auth(request)
    _require_scope(user, "billing:read")

    # Non-admin users are scoped to their own data automatically
    is_admin = bool(user.get("is_admin"))
    if not is_admin:
        customer_id = _analytics_customer_scope(user)
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
    # Provider filter: use payout ownership instead of assuming provider_id == host_id.
    if provider_id:
        where_clauses.append(
            "job_id IN (SELECT job_id FROM payout_splits WHERE provider_id = %s)"
        )
        params.append(provider_id)

    where_sql = " AND ".join(where_clauses)

    try:
        with billing._conn() as conn:
            rows = conn.execute(
                f"SELECT {group_sql}, "
                "COUNT(*) AS job_count, "
                "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS total_cost_cad, "
                "ROUND(COALESCE(SUM(gpu_seconds), 0)::numeric, 0) AS total_gpu_seconds, "
                "ROUND(COALESCE(AVG(gpu_utilization_pct), 0)::numeric, 1) AS avg_gpu_util_pct, "
                "COALESCE(SUM(is_canadian_compute), 0) AS canadian_jobs, "
                "COUNT(*) - COALESCE(SUM(is_canadian_compute), 0) AS international_jobs "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY period ORDER BY period",
                params,
            ).fetchall()

            analytics = [
                {
                    "period": r["period"],
                    "job_count": r["job_count"],
                    "total_cost_cad": float(r["total_cost_cad"] or 0),
                    "total_gpu_hours": (
                        round(float(r["total_gpu_seconds"] or 0) / 3600, 2) if r["total_gpu_seconds"] else 0
                    ),
                    "avg_gpu_utilization_pct": float(r["avg_gpu_util_pct"] or 0),
                    "canadian_jobs": r["canadian_jobs"],
                    "international_jobs": r["international_jobs"],
                }
                for r in rows
            ]

            # Summary
            summary_row = conn.execute(
                "SELECT COUNT(*) AS total_jobs, "
                "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS total_spend, "
                "ROUND((COALESCE(SUM(gpu_seconds), 0) / 3600.0)::numeric, 2) AS total_gpu_hours, "
                "ROUND(COALESCE(AVG(gpu_utilization_pct), 0)::numeric, 1) AS avg_util "
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
            "total_spend_cad": float(summary_row["total_spend"] or 0) if summary_row else 0,
            "total_gpu_hours": float(summary_row["total_gpu_hours"] or 0) if summary_row else 0,
            "avg_gpu_utilization_pct": float(summary_row["avg_util"] or 0) if summary_row else 0,
        },
    }


@router.get("/api/analytics/enhanced", tags=["Billing"])
def api_analytics_enhanced(
    request: Request,
    days: int = 30,
):
    """Enhanced analytics endpoint for the rich analytics dashboard.

    Returns aggregated data including cost-per-hour trends, cumulative spend,
    duration histograms, job status breakdowns, wallet trends, top hosts,
    sovereignty stats, and hourly heatmap data.

    Non-admin users are automatically scoped to their own data.
    Providers get both provider and customer views.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    if user.get("email"):
        full_user = UserStore.get_user(user["email"]) if _USE_PERSISTENT_AUTH else _users_db.get(user["email"], {})
        user = _merge_auth_user(user, full_user)

    is_admin = bool(user.get("is_admin"))
    customer_id = _analytics_customer_scope(user)
    provider_id = user.get("provider_id", "")

    billing = get_billing_engine()
    now = time.time()
    safe_days = max(1, min(days, 3650))
    since = now - (safe_days * 86400)

    # Build WHERE clause based on role
    if is_admin:
        where_clauses = ["started_at >= %s", "started_at < %s"]
        params: list = [since, now]
    else:
        where_clauses = ["started_at >= %s", "started_at < %s", "owner = %s"]
        params = [since, now, customer_id]

    where_sql = " AND ".join(where_clauses)

    result: dict = {
        "ok": True,
        "days": safe_days,
        "role": "admin" if is_admin else ("provider" if provider_id else "customer"),
        "customer_id": customer_id,
        "provider_id": provider_id,
    }

    try:
        with billing._conn() as conn:
            # ── 1. Cost-per-hour trend (daily avg cost per GPU hour) ──
            cph_rows = conn.execute(
                "SELECT to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS date, "
                "CASE WHEN SUM(gpu_seconds) > 0 "
                "  THEN ROUND((SUM(total_cost_cad) / (SUM(gpu_seconds) / 3600.0))::numeric, 4) "
                "  ELSE 0 END AS cost_per_hour, "
                "ROUND((COALESCE(SUM(gpu_seconds), 0) / 3600.0)::numeric, 2) AS gpu_hours, "
                "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS spend "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY date ORDER BY date",
                params,
            ).fetchall()
            result["cost_per_hour_trend"] = [
                {"date": r["date"], "cost_per_hour": float(r["cost_per_hour"]),
                 "gpu_hours": float(r["gpu_hours"]), "spend": float(r["spend"])}
                for r in cph_rows
            ]

            # ── 2. Cumulative spend ──
            running = 0.0
            cumulative = []
            for r in cph_rows:
                running += float(r["spend"])
                cumulative.append({"date": r["date"], "total": round(running, 2)})
            result["cumulative_spend"] = cumulative

            # ── 3. Duration histogram (job duration buckets) ──
            dur_rows = conn.execute(
                "SELECT "
                "CASE "
                "  WHEN duration_sec < 60 THEN '< 1 min' "
                "  WHEN duration_sec < 300 THEN '1-5 min' "
                "  WHEN duration_sec < 1800 THEN '5-30 min' "
                "  WHEN duration_sec < 3600 THEN '30-60 min' "
                "  WHEN duration_sec < 14400 THEN '1-4 hr' "
                "  ELSE '4+ hr' "
                "END AS bucket, "
                "COUNT(*) AS count, "
                "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS total_cost "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY bucket ORDER BY MIN(duration_sec)",
                params,
            ).fetchall()
            result["duration_histogram"] = [
                {"bucket": r["bucket"], "count": int(r["count"]),
                 "total_cost": float(r["total_cost"])}
                for r in dur_rows
            ]

            # ── 4. GPU hours by day (dedicated chart) ──
            result["daily_gpu_hours"] = [
                {"date": r["date"], "hours": float(r["gpu_hours"])}
                for r in cph_rows
            ]

            # ── 5. Hourly heatmap (jobs by day-of-week + hour) ──
            heat_rows = conn.execute(
                "SELECT EXTRACT(DOW FROM to_timestamp(started_at)) AS dow, "
                "EXTRACT(HOUR FROM to_timestamp(started_at)) AS hour, "
                "COUNT(*) AS count "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY dow, hour ORDER BY dow, hour",
                params,
            ).fetchall()
            result["hourly_heatmap"] = [
                {"dow": int(r["dow"]), "hour": int(r["hour"]),
                 "count": int(r["count"])}
                for r in heat_rows
            ]

            # ── 6. Top hosts used (or top customers if admin) ──
            if is_admin:
                top_rows = conn.execute(
                    "SELECT owner AS entity, COUNT(*) AS job_count, "
                    "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS total_cost, "
                    "ROUND((COALESCE(SUM(gpu_seconds), 0) / 3600.0)::numeric, 2) AS gpu_hours "
                    f"FROM usage_meters WHERE {where_sql} "
                    "GROUP BY owner ORDER BY total_cost DESC LIMIT 10",
                    params,
                ).fetchall()
            else:
                top_rows = conn.execute(
                    "SELECT host_id AS entity, COUNT(*) AS job_count, "
                    "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS total_cost, "
                    "ROUND((COALESCE(SUM(gpu_seconds), 0) / 3600.0)::numeric, 2) AS gpu_hours "
                    f"FROM usage_meters WHERE {where_sql} "
                    "GROUP BY host_id ORDER BY job_count DESC LIMIT 10",
                    params,
                ).fetchall()
            result["top_entities"] = [
                {"entity": r["entity"], "job_count": int(r["job_count"]),
                 "total_cost": float(r["total_cost"]),
                 "gpu_hours": float(r["gpu_hours"])}
                for r in top_rows
            ]

            # ── 7. Sovereignty summary ──
            sov_row = conn.execute(
                "SELECT "
                "COUNT(*) AS total, "
                "SUM(CASE WHEN is_canadian_compute = 1 THEN 1 ELSE 0 END) AS canadian, "
                "ROUND(COALESCE(SUM(CASE WHEN is_canadian_compute = 1 THEN total_cost_cad ELSE 0 END), 0)::numeric, 2) AS ca_spend, "
                "ROUND(COALESCE(SUM(CASE WHEN is_canadian_compute = 0 THEN total_cost_cad ELSE 0 END), 0)::numeric, 2) AS intl_spend "
                f"FROM usage_meters WHERE {where_sql}",
                params,
            ).fetchone()
            total_jobs_sov = int(sov_row["total"]) if sov_row else 0
            result["sovereignty"] = {
                "total_jobs": total_jobs_sov,
                "canadian_jobs": int((sov_row["canadian"] or 0)) if sov_row else 0,
                "canadian_pct": round(int((sov_row["canadian"] or 0)) / total_jobs_sov * 100, 1) if total_jobs_sov else 0,
                "canadian_spend": float(sov_row["ca_spend"]) if sov_row else 0,
                "international_spend": float(sov_row["intl_spend"]) if sov_row else 0,
            }

            # ── 8. GPU model performance breakdown ──
            gpu_perf_rows = conn.execute(
                "SELECT gpu_model, COUNT(*) AS jobs, "
                "ROUND(COALESCE(AVG(gpu_utilization_pct), 0)::numeric, 1) AS avg_util, "
                "ROUND(COALESCE(AVG(duration_sec / 60.0), 0)::numeric, 1) AS avg_duration_min, "
                "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS total_cost, "
                "ROUND((COALESCE(SUM(gpu_seconds), 0) / 3600.0)::numeric, 2) AS gpu_hours, "
                "ROUND(COALESCE(AVG(CASE WHEN gpu_seconds > 0 "
                "  THEN total_cost_cad / (gpu_seconds / 3600.0) ELSE 0 END), 0)::numeric, 4) AS avg_cost_per_hour "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY gpu_model ORDER BY total_cost DESC LIMIT 12",
                params,
            ).fetchall()
            result["gpu_performance"] = [
                {
                    "gpu_model": r["gpu_model"],
                    "jobs": int(r["jobs"]),
                    "avg_util": float(r["avg_util"]),
                    "avg_duration_min": float(r["avg_duration_min"]),
                    "total_cost": float(r["total_cost"]),
                    "gpu_hours": float(r["gpu_hours"]),
                    "avg_cost_per_hour": float(r["avg_cost_per_hour"]),
                }
                for r in gpu_perf_rows
            ]

            # ── 9. Provider earnings (if user is a provider) ──
            if provider_id:
                prov_rows = conn.execute(
                    "SELECT to_char(to_timestamp(COALESCE(um.completed_at, ps.created_at)), 'YYYY-MM-DD') AS date, "
                    "COUNT(DISTINCT ps.job_id) AS jobs_served, "
                    "ROUND(COALESCE(SUM(ps.provider_share_cad), 0)::numeric, 2) AS total_revenue, "
                    "ROUND(COALESCE(AVG(COALESCE(um.gpu_utilization_pct, 0)), 0)::numeric, 1) AS avg_util "
                    "FROM payout_splits ps "
                    "LEFT JOIN usage_meters um ON um.job_id = ps.job_id "
                    "WHERE ps.provider_id = %s AND ps.created_at >= %s AND ps.created_at < %s "
                    "GROUP BY date ORDER BY date",
                    (provider_id, since, now),
                ).fetchall()
                result["provider_daily"] = [
                    {"date": r["date"], "jobs_served": int(r["jobs_served"]),
                     "total_revenue": float(r["total_revenue"]),
                     "avg_util": float(r["avg_util"])}
                    for r in prov_rows
                ]

                prov_summary = conn.execute(
                    "SELECT COUNT(DISTINCT ps.job_id) AS total_jobs_served, "
                    "ROUND(COALESCE(SUM(ps.provider_share_cad), 0)::numeric, 2) AS total_revenue, "
                    "ROUND((COALESCE(SUM(um.gpu_seconds), 0) / 3600.0)::numeric, 2) AS total_gpu_hours, "
                    "ROUND(COALESCE(AVG(COALESCE(um.gpu_utilization_pct, 0)), 0)::numeric, 1) AS avg_util "
                    "FROM payout_splits ps "
                    "LEFT JOIN usage_meters um ON um.job_id = ps.job_id "
                    "WHERE ps.provider_id = %s AND ps.created_at >= %s AND ps.created_at < %s",
                    (provider_id, since, now),
                ).fetchone()
                result["provider_summary"] = {
                    "total_jobs_served": int(prov_summary["total_jobs_served"]) if prov_summary else 0,
                    "total_revenue": float(prov_summary["total_revenue"]) if prov_summary else 0,
                    "total_gpu_hours": float(prov_summary["total_gpu_hours"]) if prov_summary else 0,
                    "avg_util": float(prov_summary["avg_util"]) if prov_summary else 0,
                }

            # ── 10. Wallet balance trend (from wallet_transactions) ──
            try:
                wallet_rows = conn.execute(
                    "SELECT to_char(to_timestamp(created_at), 'YYYY-MM-DD') AS date, "
                    "tx_type, "
                    "SUM(amount_cad) AS total_amount, "
                    "COUNT(*) AS tx_count "
                    "FROM wallet_transactions "
                    "WHERE customer_id = %s AND created_at >= %s "
                    "GROUP BY date, tx_type ORDER BY date",
                    [customer_id, since],
                ).fetchall()
                result["wallet_activity"] = [
                    {"date": r["date"], "tx_type": r["tx_type"],
                     "total_amount": float(r["total_amount"]),
                     "tx_count": int(r["tx_count"])}
                    for r in wallet_rows
                ]
            except Exception:
                result["wallet_activity"] = []

            # ── 11. Peak usage periods ──
            peak_rows = conn.execute(
                "SELECT to_char(to_timestamp(started_at), 'YYYY-MM-DD') AS date, "
                "COUNT(*) AS jobs, "
                "ROUND((COALESCE(SUM(gpu_seconds), 0) / 3600.0)::numeric, 2) AS gpu_hours, "
                "ROUND(COALESCE(SUM(total_cost_cad), 0)::numeric, 2) AS spend, "
                "ROUND(COALESCE(AVG(gpu_utilization_pct), 0)::numeric, 1) AS avg_util "
                f"FROM usage_meters WHERE {where_sql} "
                "GROUP BY date ORDER BY jobs DESC LIMIT 5",
                params,
            ).fetchall()
            result["peak_days"] = [
                {"date": r["date"], "jobs": int(r["jobs"]),
                 "gpu_hours": float(r["gpu_hours"]),
                 "spend": float(r["spend"]),
                 "avg_util": float(r["avg_util"])}
                for r in peak_rows
            ]

    except Exception as e:
        log.error("Enhanced analytics query failed: %s", e)
        return {"ok": False, "error": "An error occurred while loading analytics"}

    return result


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
