"""Stripe Tax helpers for wallet top-ups (platform merchant of record).

Dashboard prerequisites (already configured for Xcelsior):
- Head office: London, ON
- Preset tax code: txcd_10000000 (electronically supplied services)
- Tax behavior: automatic / exclusive for CAD
- Active CA standard registration

PaymentIntents cannot use automatic_tax[enabled]=true. Use Tax Calculation
API + hooks.inputs.tax.calculation (simplified PaymentIntent integration).
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("xcelsior.stripe_tax")

# Match catalog / Dashboard preset for compute credits (digital services).
DEFAULT_TAX_CODE = "txcd_10000000"

# Canadian province codes for validation / defaults.
CA_PROVINCES = frozenset(
    {
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
)


def tax_enabled() -> bool:
    return os.environ.get("XCELSIOR_STRIPE_TAX_ENABLED", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def normalize_customer_address(address: dict | None) -> dict[str, str] | None:
    """Normalize account/billing location for Stripe Tax.

    Uses the same model as user profile / provider accounts: country + province
    (state). No deposit UI province picker — callers pass account fields.
    Returns None when we only have IP (caller should use ip_address instead).
    """
    raw = address or {}
    country = str(raw.get("country") or "").strip().upper()[:2]
    state = str(raw.get("state") or raw.get("province") or "").strip().upper()
    if not country and not state:
        return None
    if not country:
        country = "CA"
    if country == "CA" and state and len(state) > 2:
        # Allow full province names lightly if already 2-letter leave as-is.
        pass
    if country == "CA" and state and state not in CA_PROVINCES and len(state) == 2:
        # Unknown 2-letter code — still pass through for Stripe.
        pass
    out: dict[str, str] = {"country": country}
    if state:
        out["state"] = state
    for key in ("line1", "line2", "city", "postal_code"):
        val = str(raw.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def resolve_tax_customer_details(
    *,
    address: dict | None = None,
    ip_address: str = "",
) -> tuple[dict[str, Any], str]:
    """Build Stripe customer_details and a short source label.

    Priority:
      1. Explicit/account address (country + province/state) — preferred
      2. Client IP estimation via Stripe Tax
    """
    addr = normalize_customer_address(address)
    if addr and addr.get("country"):
        return (
            {"address": addr, "address_source": "billing"},
            "account_address",
        )
    ip = (ip_address or "").strip()
    # Strip port / proxy list — first hop
    if "," in ip:
        ip = ip.split(",")[0].strip()
    if ip and ip.lower() not in ("unknown", "127.0.0.1", "::1"):
        return {"ip_address": ip}, "ip_address"
    # Last resort: platform home market (London ON) so CA registration applies
    # rather than silent zero tax. Account profile should normally supply province.
    fallback = {
        "country": "CA",
        "state": os.environ.get("XCELSIOR_TAX_DEFAULT_PROVINCE", "ON").strip().upper() or "ON",
    }
    return {"address": fallback, "address_source": "billing"}, "platform_default"


def calculate_wallet_deposit_tax(
    *,
    amount_cents: int,
    address: dict | None = None,
    ip_address: str = "",
    currency: str = "cad",
    reference: str = "wallet_deposit",
    stripe_mod=None,
) -> dict[str, Any]:
    """Create a Stripe Tax Calculation for a pretax credit deposit.

    Tax location comes from the customer **account** (country/province), same
    idea as provider location — not a checkout province picker.
    """
    credit = max(0, int(amount_cents))
    base: dict[str, Any] = {
        "tax_calculation_id": "",
        "amount_total": credit,
        "tax_amount_cents": 0,
        "credit_amount_cents": credit,
        "currency": currency.lower(),
        "tax_enabled": False,
        "breakdown": [],
        "error": "",
        "location_source": "",
    }
    if not tax_enabled() or credit <= 0:
        return base

    from stripe_connect import STRIPE_ENABLED, stripe as default_stripe

    client = stripe_mod if stripe_mod is not None else default_stripe
    if not (STRIPE_ENABLED and client):
        base["error"] = "stripe_disabled"
        return base

    customer_details, location_source = resolve_tax_customer_details(
        address=address, ip_address=ip_address
    )
    try:
        calc = client.tax.Calculation.create(
            currency=currency.lower(),
            customer_details=customer_details,
            line_items=[
                {
                    "amount": credit,
                    "reference": reference[:500],
                    "tax_code": os.environ.get("XCELSIOR_STRIPE_TAX_CODE", DEFAULT_TAX_CODE),
                    # CAD defaults exclusive under Dashboard "Automatic" behavior.
                    "tax_behavior": "exclusive",
                }
            ],
        )
        tax_exclusive = int(getattr(calc, "tax_amount_exclusive", 0) or 0)
        amount_total = int(getattr(calc, "amount_total", credit) or credit)
        breakdown = []
        raw_bd = getattr(calc, "tax_breakdown", None) or []
        for b in raw_bd:
            rate = getattr(b, "tax_rate_details", None)
            breakdown.append(
                {
                    "amount": int(getattr(b, "amount", 0) or 0),
                    "percentage": float(getattr(rate, "percentage_decimal", 0) or 0)
                    if rate
                    else 0.0,
                    "tax_type": getattr(rate, "tax_type", "") if rate else "",
                    "state": getattr(rate, "state", "") if rate else "",
                    "country": getattr(rate, "country", "") if rate else "",
                    "taxability_reason": getattr(b, "taxability_reason", "") or "",
                }
            )
        return {
            "tax_calculation_id": calc.id,
            "amount_total": amount_total,
            "tax_amount_cents": tax_exclusive,
            "credit_amount_cents": credit,
            "currency": currency.lower(),
            "tax_enabled": True,
            "breakdown": breakdown,
            "error": "",
            "location_source": location_source,
            "customer_details": customer_details,
        }
    except Exception as exc:
        log.warning("Stripe Tax calculation failed (charging pretax only): %s", exc)
        base["error"] = str(exc)[:240]
        base["location_source"] = location_source
        return base
