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
from typing import Any, cast

log = logging.getLogger("xcelsior.stripe_tax")

# Match catalog / Dashboard preset for compute credits (digital services).
DEFAULT_TAX_CODE = "txcd_10000000"


class TaxCalculationUnavailable(RuntimeError):
    """Tax is enabled, but no valid calculation is available for a payment."""


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
    value = os.environ.get("XCELSIOR_STRIPE_TAX_ENABLED", "1").strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise TaxCalculationUnavailable("The tax configuration is invalid. Please contact support.")


def normalize_customer_address(address: dict | None) -> dict[str, str] | None:
    """Validate the full saved billing address without inventing geography."""
    if not address:
        return None
    from host_metadata import profile_address_values

    raw = address
    profile = {
        "country": raw.get("country"),
        "province": raw.get("state") or raw.get("province"),
        "address_line1": raw.get("line1"),
        "address_line2": raw.get("line2"),
        "city": raw.get("city"),
        "postal_code": raw.get("postal_code"),
    }
    cleaned, errors = profile_address_values(profile)
    if errors:
        raise ValueError("Complete your billing address before calculating tax.")
    return {
        "country": cleaned["country"],
        "state": cleaned["province"],
        "line1": cleaned["address_line1"],
        "line2": cleaned["address_line2"],
        "city": cleaned["city"],
        "postal_code": cleaned["postal_code"],
    }


def resolve_tax_customer_details(
    *,
    address: dict | None = None,
    ip_address: str = "",
) -> tuple[dict[str, Any], str]:
    """Tax uses the validated saved address; IP cannot fill a missing profile."""
    addr = normalize_customer_address(address)
    if addr is None:
        raise ValueError("Complete your billing address before calculating tax.")
    return {"address": addr, "address_source": "billing"}, "account_address"


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
    if type(amount_cents) is not int or amount_cents <= 0:
        raise ValueError("The credit amount must be a positive integer number of cents.")
    credit = amount_cents
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
    if not tax_enabled():
        return base

    from stripe_connect import STRIPE_ENABLED, stripe as default_stripe

    client = stripe_mod if stripe_mod is not None else default_stripe
    if not (STRIPE_ENABLED and client):
        raise TaxCalculationUnavailable("Tax could not be calculated. Please try again shortly.")

    customer_details, location_source = resolve_tax_customer_details(
        address=address, ip_address=ip_address
    )
    try:
        calc = client.tax.Calculation.create(
            currency=currency.lower(),
            # stripe 15.3 types this as CalculationCreateParamsCustomerDetails;
            # our resolver builds the same shape as a plain dict.
            customer_details=cast(Any, customer_details),
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
        tax_exclusive = calc.tax_amount_exclusive
        amount_total = calc.amount_total
        if (
            not isinstance(calc.id, str)
            or not calc.id.startswith("taxcalc_")
            or calc.currency != currency.lower()
            or type(tax_exclusive) is not int
            or tax_exclusive < 0
            or type(amount_total) is not int
            or amount_total != credit + tax_exclusive
        ):
            raise ValueError("Invalid tax calculation totals or currency")
        breakdown = []
        raw_bd = getattr(calc, "tax_breakdown", None) or []
        for b in raw_bd:
            rate = getattr(b, "tax_rate_details", None)
            breakdown.append(
                {
                    "amount": int(getattr(b, "amount", 0) or 0),
                    "percentage": (
                        float(getattr(rate, "percentage_decimal", 0) or 0) if rate else 0.0
                    ),
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
        # Provider errors can contain submitted addresses. Keep that data out
        # of logs and client responses; no PaymentIntent may use guessed tax.
        log.warning("Stripe Tax calculation failed (%s)", type(exc).__name__)
        raise TaxCalculationUnavailable(
            "Tax could not be calculated. Please try again shortly."
        ) from None
