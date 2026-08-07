"""The reserved-commitment discount schedule. One copy, read by both doors.

Two endpoints create a reserved commitment, and until this module existed they
disagreed about what one costs:

* ``POST /api/pricing/reserve`` (``routes/billing.RESERVED_PRICING_TIERS``) —
  1 month 20%, 3 months 30%, 1 year 45%. Writes ``reserved_commitments``,
  applies provincial tax, records ``min_hours_per_day``.
* ``POST /api/v2/marketplace/reservations`` (``marketplace.RESERVED_DISCOUNTS``) —
  1 month 20%, 3 months 30%, **6 months 40%**. Writes ``reservations``, prices
  off ``MIN(gpu_offers.ask_cents_per_hour)`` and falls back to a hardcoded 20¢.

They agreed at one and three months and diverged after, so a customer could be
quoted differently depending on which endpoint they reached.

A *third* number set existed inside the marketplace function itself: its
docstring read "1-month=10%, 3-month=20%, 6-month=30%, 12-month=40%" directly
beneath a dict saying 20/30/40. Prose next to code is not a source of truth, and
that is the argument for this module rather than for a comment asking the two to
be kept in step.

**The 6-month tier was promoted rather than dropped**, which is the
non-destructive of the two available choices: no caller loses a term that works
today, and the discount curve stays monotonic (20 → 30 → 40 → 45). Dropping it
instead is deleting the ``6`` entry below and nothing else — the decision lives
in one place now, which was the point.

Pricing is the *rate*; the money itself is integer micros everywhere it is
stored (see ``tests/test_money_representation.py``). ``discount_pct`` is a
percentage, not an amount, so it is an ordinary integer.
"""

from __future__ import annotations

from typing import Any

#: Keyed by commitment length in months — the unit both callers actually have.
#: `key` is the string form `routes/billing.py` exposes in its API, kept so the
#: public request shape does not change.
RESERVED_TIERS: dict[int, dict[str, Any]] = {
    1: {
        "key": "1_month",
        "commitment": "1 month",
        "discount_pct": 20,
        "term_days": 30,
        "min_hours_per_day": 4,
    },
    3: {
        "key": "3_month",
        "commitment": "3 months",
        "discount_pct": 30,
        "term_days": 90,
        "min_hours_per_day": 4,
    },
    6: {
        "key": "6_month",
        "commitment": "6 months",
        "discount_pct": 40,
        "term_days": 182,
        "min_hours_per_day": 4,
    },
    12: {
        "key": "1_year",
        "commitment": "1 year",
        "discount_pct": 45,
        "term_days": 365,
        "min_hours_per_day": 0,
    },
}

#: The same rows keyed by the string the billing API accepts.
RESERVED_TIERS_BY_KEY: dict[str, dict[str, Any]] = {
    tier["key"]: {**tier, "months": months} for months, tier in RESERVED_TIERS.items()
}


def tier_for_months(months: int) -> dict[str, Any] | None:
    return RESERVED_TIERS.get(int(months))


def tier_for_key(key: str) -> dict[str, Any] | None:
    return RESERVED_TIERS_BY_KEY.get(str(key))


def discount_fraction(months: int) -> float | None:
    """`marketplace` multiplies by a fraction; billing reads whole percent."""
    tier = tier_for_months(months)
    return None if tier is None else tier["discount_pct"] / 100.0
