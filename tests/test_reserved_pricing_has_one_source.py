"""Two endpoints create a reserved commitment. They must quote the same price.

`POST /api/pricing/reserve` and `POST /api/v2/marketplace/reservations` each
carried their own discount table. They agreed at one and three months and
diverged after — billing offered a year at 45%, marketplace six months at 40% —
so what a commitment cost depended on which endpoint the customer reached.

A third set of numbers lived in `marketplace.create_reservation`'s own
docstring: "1-month=10%, 3-month=20%, 6-month=30%, 12-month=40%", directly
beneath a dict that said 20/30/40. Prose beside code is not a source of truth.

Both now derive from `reserved_pricing.RESERVED_TIERS`. This asserts the
derivation rather than the numbers: pinning the percentages here would just
create a fourth copy to keep in step.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")


def test_both_tables_derive_from_the_one_schedule():
    """The same terms, at the same discounts, through either door."""
    from marketplace import RESERVED_DISCOUNTS
    from reserved_pricing import RESERVED_TIERS
    from routes.billing import RESERVED_PRICING_TIERS

    assert set(RESERVED_DISCOUNTS) == set(RESERVED_TIERS), (
        "the marketplace offers different commitment lengths from the schedule: "
        f"{sorted(RESERVED_DISCOUNTS)} vs {sorted(RESERVED_TIERS)}"
    )
    for months, tier in RESERVED_TIERS.items():
        assert RESERVED_DISCOUNTS[months] * 100 == tier["discount_pct"], (
            f"marketplace quotes {RESERVED_DISCOUNTS[months] * 100}% for "
            f"{months} months; the schedule says {tier['discount_pct']}%"
        )
        billing = RESERVED_PRICING_TIERS[tier["key"]]
        assert billing["discount_pct"] == tier["discount_pct"], (
            f"billing quotes {billing['discount_pct']}% for {tier['key']}; the "
            f"schedule says {tier['discount_pct']}%"
        )


def test_the_term_lengths_match_the_commitment_names():
    """A '6 months' tier billed for 30 days would be a silent overcharge."""
    from reserved_pricing import RESERVED_TIERS
    from routes.billing import _RESERVED_TERM_DAYS

    for months, tier in RESERVED_TIERS.items():
        days = _RESERVED_TERM_DAYS[tier["key"]]
        expected = months * 30.4
        assert abs(days - expected) <= 16, (
            f"{tier['key']} runs for {days} days, which is not {months} months"
        )


def test_the_discount_curve_rewards_longer_commitments():
    """Monotonic, or a customer is better off committing for less.

    Not a style preference: a schedule where six months discounts less than
    three is a pricing bug that reads as a rounding error.
    """
    from reserved_pricing import RESERVED_TIERS

    ordered = sorted(RESERVED_TIERS.items())
    percentages = [tier["discount_pct"] for _, tier in ordered]
    assert percentages == sorted(percentages), (
        f"the discount curve is not monotonic across terms: {ordered}"
    )


def test_neither_module_restates_the_percentages():
    """The failure was two tables, so a second literal table is the regression."""
    import inspect

    import marketplace
    import routes.billing as billing

    for module in (marketplace, billing):
        source = inspect.getsource(module)
        head = source[: source.index("class ") if "class " in source else 4000]
        assert "0.40" not in head or "reserved_pricing" in head, (
            f"{module.__name__} appears to hardcode a reserved discount again; "
            "the schedule lives in reserved_pricing.py"
        )
        assert "reserved_pricing" in source, (
            f"{module.__name__} no longer derives from reserved_pricing, so the "
            "two doors can quote different prices again"
        )
