"""GPU rates come only from the imported Stripe catalog (``gpu_pricing``).

Expected values are read from ``billing/stripe_catalog.json``, the snapshot the
session fixture imports exactly as deploy does, never restated here.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from pricing_reference import (
    PRICING_MODES,
    build_gpu_pricing_reference,
    build_reference_list,
    commitment_pricing_mode,
    estimate_job_cost,
    get_on_demand_rate,
    get_rate,
    get_reserved_rate,
    parse_gpu_name,
)
from service_tier_pricing import TierPricingUnavailable

_CATALOG = json.loads(
    (Path(__file__).resolve().parents[1] / "billing/stripe_catalog.json").read_text()
)["service_tier_catalog"]["prices"]
_CATALOG_MODELS = sorted({row["variant"]["gpu_model"] for row in _CATALOG})


def _catalog_price(model, tier="standard", mode="on_demand", *, vram_gb=None, form_factor=None):
    """(CAD rate, Stripe Price id) on the variant a quote picks."""
    rows = [
        row
        for row in _CATALOG
        if row["variant"]["gpu_model"] == model
        and row["tier"] == tier
        and row["mode"] == mode
        and (vram_gb is None or row["variant"]["vram_gb"] == vram_gb)
        and (form_factor is None or row["variant"]["form_factor"] == form_factor)
    ]
    row = min(
        rows,
        key=lambda r: (
            r["variant"]["vram_gb"],
            r["variant"]["high_frequency"],
            r["variant"]["form_factor"],
        ),
    )
    price = row["price"]
    return float(Decimal(price["unit_amount_decimal"]) / 100), price["id"]


def test_commitment_mode_mapping():
    assert commitment_pricing_mode("1_month") == "reserved_1mo"
    assert commitment_pricing_mode("3_month") == "reserved_3mo"
    assert commitment_pricing_mode("1_year") == "reserved_1yr"
    assert commitment_pricing_mode("6_month") is None


def test_parse_gpu_name_splits_model_vram_and_form_factor():
    assert parse_gpu_name("NVIDIA A100 80GB SXM") == ("A100", 80, "SXM")
    assert parse_gpu_name("A100-SXM4-80GB") == ("A100", 80, "SXM")
    assert parse_gpu_name("A100 40GB PCIe") == ("A100", 40, "PCIe")
    assert parse_gpu_name("NVIDIA GeForce RTX 3060") == ("RTX 3060", None, None)
    assert parse_gpu_name("rtx 4090") == ("RTX 4090", None, None)
    assert parse_gpu_name("RTX 4060 Ti 8GB") == ("RTX 4060 Ti", 8, None)
    assert parse_gpu_name("L40S") == ("L40S", None, None)
    assert parse_gpu_name("H100 NVL") == ("H100 NVL", None, None)


def test_reference_covers_every_catalog_model_at_its_stripe_prices():
    pricing = build_gpu_pricing_reference()
    for model in _CATALOG_MODELS:
        entry = pricing[model]
        on_demand, _ = _catalog_price(model)
        assert entry["base_rate_cad"] == pytest.approx(on_demand), model
        for mode in PRICING_MODES[1:]:
            assert entry[f"{mode}_cad"] == pytest.approx(_catalog_price(model, mode=mode)[0]), (
                model,
                mode,
            )
        assert entry["premium_rate_cad"] == pytest.approx(_catalog_price(model, "premium")[0])
        assert entry["dedicated_rate_cad"] == pytest.approx(_catalog_price(model, "dedicated")[0])
    rates = [entry["base_rate_cad"] for entry in pricing.values()]
    assert rates == sorted(rates), "reference is cheapest first"


def test_reference_uses_the_variant_a_quote_picks():
    pricing = build_gpu_pricing_reference()
    # Smallest VRAM, then standard frequency, then form factor (PCIe < SXM).
    assert (pricing["A100"]["vram_gb"], pricing["A100"]["form_factor"]) == (40, "PCIe")
    assert (pricing["H100"]["vram_gb"], pricing["H100"]["form_factor"]) == (80, "PCIe")
    assert pricing["RTX 4090"]["high_frequency"] is False


def test_reference_list_shape():
    pricing = build_gpu_pricing_reference()
    rows = {row["gpu_model"]: row for row in build_reference_list(pricing)}
    assert set(rows) == set(pricing)
    row = rows["RTX 4090"]
    assert row["on_demand_cad"] == pricing["RTX 4090"]["base_rate_cad"]
    assert row["vram_gb"] == pricing["RTX 4090"]["vram_gb"]
    for mode in PRICING_MODES[1:]:
        assert row[f"{mode}_cad"] == pricing["RTX 4090"][f"{mode}_cad"]


def test_rate_lookups_read_the_stripe_catalog():
    assert get_on_demand_rate("RTX 4090") == pytest.approx(_catalog_price("RTX 4090")[0])
    reserved = get_reserved_rate("RTX 4090", "3_month")
    assert reserved == pytest.approx(_catalog_price("RTX 4090", mode="reserved_3mo")[0])
    assert 0 < reserved < get_on_demand_rate("RTX 4090")
    assert get_reserved_rate("RTX 4090", "6_month") == 0.0  # no Stripe Price for that term


def test_get_rate_resolves_names_to_catalog_variants():
    expect = lambda *a, **k: pytest.approx(_catalog_price(*a, **k)[0])  # noqa: E731
    assert get_rate("NVIDIA GeForce RTX 3060") == expect("RTX 3060")
    assert get_rate("rtx 4090") == expect("RTX 4090")
    assert get_rate("A100 40GB") == expect("A100", vram_gb=40)
    assert get_rate("A100 80GB") == expect("A100", vram_gb=80)
    assert get_rate("NVIDIA A100 80GB SXM") == expect("A100", vram_gb=80, form_factor="SXM")
    assert get_rate("A100 80GB") < get_rate("NVIDIA A100 80GB SXM")
    assert get_rate("NVIDIA L40S") == expect("L40S")
    assert get_rate("L40") == expect("L40")
    assert get_rate("L40") != get_rate("L40S")
    assert get_rate("RTX 4090", "spot") == expect("RTX 4090", mode="spot")
    assert get_rate("RTX 4090", tier="premium") == expect("RTX 4090", "premium")
    # Orderings the old static table asserted, now held by the Stripe Prices.
    assert get_rate("A100 80GB") > get_rate("A100 40GB")
    assert get_rate("H200") > get_rate("H100")
    assert get_rate("RTX 4090", "spot") < get_rate("RTX 4090")


def test_unpriced_gpu_has_no_rate():
    # No stand-in rate: an unknown GPU used to be priced as an RTX 4090.
    assert get_rate("Tesla K80") == 0.0
    assert get_rate("") == 0.0
    assert get_rate("RTX 4090 SXM") == 0.0  # no such variant


def test_estimate_job_cost_is_the_launch_quote():
    rate, price_id = _catalog_price("RTX 4090")
    estimate = estimate_job_cost("RTX 4090", 2.5, num_gpus=2)
    assert estimate["stripe_price_id"] == price_id
    assert estimate["rate_per_gpu_cad"] == pytest.approx(rate)
    assert estimate["rate_cad_per_hour"] == pytest.approx(rate * 2)
    assert estimate["gross_cost_cad"] == pytest.approx(rate * 2 * 2.5)
    assert estimate["currency"] == "CAD"
    assert estimate["spot"] is False

    spot_rate, spot_id = _catalog_price("RTX 4090", mode="spot")
    spot = estimate_job_cost("RTX 4090", 1, pricing_mode="spot")
    assert (spot["stripe_price_id"], spot["spot"]) == (spot_id, True)
    assert spot["gross_cost_cad"] == pytest.approx(spot_rate)

    sxm_rate, sxm_id = _catalog_price("H100", form_factor="SXM")
    sxm = estimate_job_cost("H100 SXM", 1)
    assert (sxm["stripe_price_id"], sxm["form_factor"]) == (sxm_id, "SXM")
    assert sxm["gross_cost_cad"] == pytest.approx(sxm_rate)


def test_estimate_refuses_an_unpriced_gpu():
    with pytest.raises(TierPricingUnavailable):
        estimate_job_cost("Tesla K80", 1)
    with pytest.raises(TierPricingUnavailable):
        estimate_job_cost("RTX 4090 SXM", 1)
