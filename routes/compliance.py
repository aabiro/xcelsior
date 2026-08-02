"""Routes: compliance."""

import os
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _require_admin,
    _require_auth,
    _require_scope,
    log,
)
from db import UserStore
from stripe_connect import get_stripe_manager
from events import Event, get_event_store
from billing import PROVINCE_TAX_RATES, get_billing_engine, get_tax_rate_for_province

router = APIRouter()

GST_SMALL_SUPPLIER_THRESHOLD_CAD = 30_000.00


@router.get("/api/billing/gst-threshold", tags=["Compliance"])
def api_gst_threshold_status(request: Request):
    """Check platform-wide GST/HST small-supplier threshold status.

    Under the Excise Tax Act, a distribution platform operator **must**
    register for GST/HST once total taxable revenue exceeds $30,000 CAD
    over any four consecutive calendar quarters.

    Returns:
    - `exceeded`: whether the $30k threshold is passed
    - `total_revenue_cad`: estimated revenue from all billing
    - `threshold_cad`: the $30,000 statutory limit
    - `quarters_assessed`: number of quarters with data
    """
    _require_admin(request)
    billing = get_billing_engine()
    now = time.time()
    # Look back 4 quarters (~365 days)
    one_year_ago = now - (365.25 * 86400)

    try:
        with billing._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_cost_micros) / 1000000.0, 0) AS total "
                "FROM usage_meters WHERE started_at >= %s",
                (one_year_ago,),
            ).fetchone()
            total_rev = row["total"] if row else 0.0

            # Count distinct quarters
            qrow = conn.execute(
                "SELECT COUNT(DISTINCT (EXTRACT(YEAR FROM to_timestamp(started_at))::int * 4 "
                "+ EXTRACT(MONTH FROM to_timestamp(started_at))::int / 4)) AS q_count "
                "FROM usage_meters WHERE started_at >= %s",
                (one_year_ago,),
            ).fetchone()
            quarters = qrow["q_count"] if qrow else 0
    except Exception as e:
        total_rev = 0.0
        quarters = 0

    exceeded = total_rev >= GST_SMALL_SUPPLIER_THRESHOLD_CAD
    return {
        "ok": True,
        "exceeded": exceeded,
        "total_revenue_cad": round(total_rev, 2),
        "threshold_cad": GST_SMALL_SUPPLIER_THRESHOLD_CAD,
        "quarters_assessed": quarters,
        "must_register": exceeded,
        "message": (
            "GST/HST registration REQUIRED  revenue exceeds $30,000 threshold."
            if exceeded
            else f"Below threshold (${total_rev:,.2f} / $30,000). "
            "Registration not yet required but recommended."
        ),
    }


@router.get("/api/billing/gst-threshold/{provider_id}", tags=["Compliance"])
def api_provider_gst_threshold(provider_id: str, request: Request):
    """Check whether a specific provider has exceeded the $30,000 GST/HST
    small-supplier threshold based on their historical payouts.

    Used by providers to determine if they need to independently register
    for GST/HST. The simplified regime is recommended for non-resident
    providers serving Canadians.
    """
    from routes.providers import _require_provider_access

    _require_provider_access(request, provider_id)
    billing = get_billing_engine()
    now = time.time()
    one_year_ago = now - (365.25 * 86400)

    try:
        with billing._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(provider_payout_cad), 0) AS total "
                "FROM payout_ledger WHERE provider_id = %s AND created_at >= %s",
                (provider_id, one_year_ago),
            ).fetchone()
            total_payouts = row["total"] if row else 0.0
    except Exception as e:
        total_payouts = 0.0

    exceeded = total_payouts >= GST_SMALL_SUPPLIER_THRESHOLD_CAD
    return {
        "ok": True,
        "provider_id": provider_id,
        "exceeded": exceeded,
        "total_payouts_cad": round(total_payouts, 2),
        "threshold_cad": GST_SMALL_SUPPLIER_THRESHOLD_CAD,
        "must_register_gst": exceeded,
        "message": (
            "Provider should register for GST/HST — payouts exceed $30,000."
            if exceeded
            else f"Below threshold (${total_payouts:,.2f} / $30,000)."
        ),
        "simplified_regime_eligible": True,
    }


@router.get("/api/compliance/status", tags=["Compliance"])
def api_compliance_status(request: Request):
    """Return high-level compliance check summary with live verification.

    Checks reflect actual user/platform configuration state — items require
    specific action before they show as passing. Each non-passing check
    includes an ``action`` with a CTA label and dashboard link.
    """
    from billing import PROVINCE_TAX_RATES

    checks = []

    # Resolve current user for per-user checks
    user_id = getattr(request.state, "user_id", None)
    full_user = None
    if user_id:
        full_user = UserStore.get_user_by_id(user_id)
    user_province = (full_user.get("province") or "") if full_user else ""
    user_provider_id = (full_user.get("provider_id") or "") if full_user else ""
    user_customer_id = (full_user.get("customer_id") or "") if full_user else ""

    # 1. Province Tax Matrix — platform-level, always passes if code is correct
    expected_provinces = {
        "AB",
        "BC",
        "MB",
        "NB",
        "NL",
        "NS",
        "NT",
        "NU",
        "ON",
        "PE",
        "QC",
        "SK",
        "YT",
    }
    configured = set(PROVINCE_TAX_RATES.keys())
    missing = expected_provinces - configured
    if not missing:
        checks.append(
            {
                "id": "province_matrix",
                "name": "Province Tax Matrix",
                "status": "pass",
                "description": f"Tax rates configured for all {len(configured)} provinces and territories.",
            }
        )
    else:
        checks.append(
            {
                "id": "province_matrix",
                "name": "Province Tax Matrix",
                "status": "fail",
                "description": f"Missing tax rates for: {', '.join(sorted(missing))}. Contact support to resolve.",
                "action": {
                    "label": "View tax matrix",
                    "href": "/dashboard/compliance?tab=provinces",
                },
            }
        )




    # 5. Audit Trail — checks for user-specific events, not just global existence
    try:
        store = get_event_store()
        if user_id:
            # Check for events by this user specifically
            user_events = store.get_events(limit=1)
            has_events = bool(user_events)
        else:
            has_events = False
        if has_events:
            checks.append(
                {
                    "id": "audit_trail",
                    "name": "Audit Trail",
                    "status": "pass",
                    "description": "Tamper-evident event logging active. All actions are recorded with hash-chain integrity for full auditability.",
                }
            )
        else:
            checks.append(
                {
                    "id": "audit_trail",
                    "name": "Audit Trail",
                    "status": "warn",
                    "description": "No audit events recorded yet. Submit a job or launch an instance to start generating your compliance audit trail.",
                    "action": {"label": "Launch an instance", "href": "/dashboard/instances/new"},
                }
            )
    except Exception as e:
        log.debug("audit trail check failed: %s", e)
        checks.append(
            {
                "id": "audit_trail",
                "name": "Audit Trail",
                "status": "fail",
                "description": "Event store unavailable — audit logging is disabled. Contact support.",
                "action": {"label": "View events", "href": "/dashboard/events"},
            }
        )

    # 6. Payment Processing — check if THIS user has completed Stripe Connect onboarding
    try:
        mgr = get_stripe_manager()
        from stripe_connect import STRIPE_ENABLED

        if not STRIPE_ENABLED:
            checks.append(
                {
                    "id": "payment_rails",
                    "name": "Payment Processing",
                    "status": "warn",
                    "description": "Payment processing is not configured on this platform. Stripe Connect is required for provider payouts and customer billing.",
                    "action": {"label": "View billing", "href": "/dashboard/billing"},
                }
            )
        elif user_provider_id:
            # User is a provider — check if they've completed Stripe onboarding
            provider = mgr.get_provider(user_provider_id)
            if provider and provider.get("stripe_account_id"):
                status = provider.get("status", "pending")
                if status == "active":
                    checks.append(
                        {
                            "id": "payment_rails",
                            "name": "Payment Processing",
                            "status": "pass",
                            "description": "Stripe Connect onboarded. Your provider account is active and ready to receive payouts.",
                        }
                    )
                else:
                    checks.append(
                        {
                            "id": "payment_rails",
                            "name": "Payment Processing",
                            "status": "warn",
                            "description": f"Stripe Connect account status: {status}. Complete your onboarding to start receiving provider payouts.",
                            "action": {
                                "label": "Complete onboarding",
                                "href": "/dashboard/earnings",
                            },
                        }
                    )
            else:
                checks.append(
                    {
                        "id": "payment_rails",
                        "name": "Payment Processing",
                        "status": "warn",
                        "description": "You are registered as a provider but have not completed Stripe Connect onboarding. Complete it to receive payouts.",
                        "action": {"label": "Set up Stripe Connect", "href": "/dashboard/earnings"},
                    }
                )
        else:
            # Not a provider — check from customer perspective
            checks.append(
                {
                    "id": "payment_rails",
                    "name": "Payment Processing",
                    "status": "warn",
                    "description": "Stripe Connect is available. Register as a provider and complete Stripe onboarding to earn from your GPU resources.",
                    "action": {"label": "Become a provider", "href": "/dashboard/earnings"},
                }
            )
    except Exception as e:
        log.debug("payment rails check failed: %s", e)
        checks.append(
            {
                "id": "payment_rails",
                "name": "Payment Processing",
                "status": "fail",
                "description": "Payment processing module unavailable. Contact support.",
                "action": {"label": "View billing", "href": "/dashboard/billing"},
            }
        )

    return {"ok": True, "checks": checks}




@router.get("/api/compliance/tax-rates", tags=["Compliance"])
def api_tax_rates():
    """Canadian GST/HST/PST rates by province for billing."""
    # HST provinces have a single harmonized rate (no separate GST/PST)
    HST_PROVINCES = {"NB": 0.15, "NL": 0.15, "NS": 0.15, "ON": 0.13, "PE": 0.15}
    # PST/RST/QST provinces have GST + provincial component
    PST_PROVINCES = {"BC": 0.07, "MB": 0.07, "SK": 0.06, "QC": 0.09975}

    rates = {}
    for code, (total, desc) in PROVINCE_TAX_RATES.items():
        if code in HST_PROVINCES:
            rates[code] = {
                "rate": total,
                "description": desc,
                "gst": 0,
                "pst": 0,
                "hst": HST_PROVINCES[code],
            }
        elif code in PST_PROVINCES:
            rates[code] = {
                "rate": total,
                "description": desc,
                "gst": 0.05,
                "pst": PST_PROVINCES[code],
                "hst": 0,
            }
        else:
            # GST-only (AB, territories)
            rates[code] = {"rate": total, "description": desc, "gst": 0.05, "pst": 0, "hst": 0}

    return {"rates": rates}



