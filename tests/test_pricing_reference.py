"""Tests for live gpu_pricing reference builder."""

from __future__ import annotations

from pricing_reference import (
    build_gpu_pricing_reference,
    build_reference_list,
    commitment_pricing_mode,
    get_on_demand_rate,
    get_reserved_rate,
)


def test_commitment_mode_mapping():
    assert commitment_pricing_mode("3_month") == "reserved_3mo"
    assert commitment_pricing_mode("1_year") == "reserved_1yr"


def test_build_gpu_pricing_reference_has_modes():
    pricing = build_gpu_pricing_reference()
    if not pricing:
        return  # empty DB in CI without gpu_pricing seed
    sample = next(iter(pricing.values()))
    assert "base_rate_cad" in sample
    assert "spot_cad" in sample
    assert "reserved_3mo_cad" in sample
    assert sample["reserved_3mo_cad"] > 0


def test_reference_list_shape():
    pricing = build_gpu_pricing_reference()
    if not pricing:
        return
    rows = build_reference_list(pricing)
    assert rows
    row = rows[0]
    assert "gpu_model" in row
    assert "reserved_3mo_cad" in row


def test_rate_lookups():
    pricing = build_gpu_pricing_reference()
    if not pricing:
        return
    model = next(iter(pricing))
    on_demand = get_on_demand_rate(model)
    reserved = get_reserved_rate(model, "3_month")
    assert on_demand > 0
    assert reserved > 0
    assert reserved < on_demand