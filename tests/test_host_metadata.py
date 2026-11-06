"""Host metadata normalization for dashboard + worker heartbeats."""

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from host_metadata import (
    CANONICAL_GPU_MODELS,
    default_vram_gb,
    enrich_host_for_api,
    infer_gpu_from_host_id,
    merge_host_update,
    normalize_host_region,
    normalize_gpu_model,
    normalize_region,
    platform_rate_cad,
)

# The Stripe catalog snapshot deploy imports; models and rates come from it.
_CATALOG = json.loads(
    (Path(__file__).resolve().parents[1] / "billing/stripe_catalog.json").read_text()
)["service_tier_catalog"]["prices"]


def _standard_rate(model: str, mode: str = "on_demand") -> float:
    rows = [
        r
        for r in _CATALOG
        if r["variant"]["gpu_model"] == model and r["tier"] == "standard" and r["mode"] == mode
    ]
    row = min(
        rows,
        key=lambda r: (
            r["variant"]["vram_gb"],
            r["variant"]["high_frequency"],
            r["variant"]["form_factor"],
        ),
    )
    return float(Decimal(row["price"]["unit_amount_decimal"]) / 100)


def test_canonical_models_are_the_stripe_catalog_models():
    assert CANONICAL_GPU_MODELS == {r["variant"]["gpu_model"] for r in _CATALOG}
    assert len(CANONICAL_GPU_MODELS) == 55


def test_default_vram_is_the_smallest_catalog_variant():
    for model in CANONICAL_GPU_MODELS:
        smallest = min(r["variant"]["vram_gb"] for r in _CATALOG if r["variant"]["gpu_model"] == model)
        assert default_vram_gb(model) == smallest, model
    assert default_vram_gb("A100") == 40
    assert default_vram_gb("NVIDIA GeForce RTX 3060") == 12
    assert default_vram_gb("not a gpu") == 0


def test_platform_rate_is_the_standard_stripe_price():
    assert platform_rate_cad("RTX 3060") == pytest.approx(_standard_rate("RTX 3060"))
    assert platform_rate_cad("NVIDIA GeForce RTX 4090") == pytest.approx(_standard_rate("RTX 4090"))
    assert platform_rate_cad("RTX 4090", pricing_mode="spot") == pytest.approx(
        _standard_rate("RTX 4090", "spot")
    )
    # No static fallback table: an unpriced GPU has no platform rate.
    assert platform_rate_cad("Tesla K80") == 0.0
    assert platform_rate_cad("") == 0.0


def test_normalize_gpu_model_legacy_dash_and_nvidia_prefix():
    assert normalize_gpu_model("RTX-3060") == "RTX 3060"
    assert normalize_gpu_model("NVIDIA GeForce RTX 3060") == "RTX 3060"
    assert normalize_gpu_model("RTX 4090") == "RTX 4090"


def test_normalize_gpu_model_ignores_spacing_and_case():
    assert normalize_gpu_model("RTX4090") == "RTX 4090"
    assert normalize_gpu_model("rtx_4090") == "RTX 4090"
    assert normalize_gpu_model("rtx 4060 ti") == "RTX 4060 Ti"


def test_merge_host_update_preserves_dashboard_metadata():
    existing = {
        "host_id": "gpu-3060-aaryn",
        "hostname": "Aaryn 3060 Box",
        "gpu_model": "RTX 3060",
        "total_vram_gb": 12,
        "cost_per_hour": 0.13,
        "owner": "user-1",
    }
    incoming = {
        "host_id": "gpu-3060-aaryn",
        "ip": "100.64.0.6",
        "gpu_model": "",
        "total_vram_gb": 0,
        "free_vram_gb": 11.5,
        "cost_per_hour": 0,
    }
    merged = merge_host_update(existing, incoming)
    assert merged["hostname"] == "Aaryn 3060 Box"
    assert merged["gpu_model"] == "RTX 3060"
    assert merged["total_vram_gb"] == 12
    assert merged["cost_per_hour"] == 0.13
    assert merged["owner"] == "user-1"


def test_infer_gpu_from_host_id():
    assert infer_gpu_from_host_id("gpu-3060-aaryn") == "RTX 3060"
    assert infer_gpu_from_host_id("gpu-2060-aaryn-local") == "RTX 2060"
    assert infer_gpu_from_host_id("my-a100-box") == "A100"


def test_enrich_host_for_api_fills_missing_fields():
    host = {
        "host_id": "gpu-3060-lab",
        "gpu_model": "RTX 3060",
        "total_vram_gb": 0,
        "cost_per_hour": 0,
    }
    out = enrich_host_for_api(host)
    assert out["hostname"] == "gpu 3060 lab"
    assert out["vram_gb"] == 12
    assert out["total_vram_gb"] == 12
    assert out["cost_per_hour"] == pytest.approx(_standard_rate("RTX 3060"))


def test_enrich_host_for_api_infers_gpu_from_host_id():
    host = {
        "host_id": "gpu-3060-lab",
        "gpu_model": "",
        "total_vram_gb": 0,
        "cost_per_hour": 0,
    }
    out = enrich_host_for_api(host)
    assert out["gpu_model"] == "RTX 3060"
    assert out["vram_gb"] == 12
    assert out["cost_per_hour"] > 0


def test_region_normalization_from_province_and_legacy_values():
    assert normalize_region("ON") == "ca-on"
    assert normalize_region("ca-ON") == "ca-on"
    assert normalize_region("", country="CA", province="QC") == "ca-qc"
    assert normalize_host_region({"country": "CA", "province": "BC"}) == "ca-bc"


def test_enrich_host_for_api_fills_region_from_province():
    out = enrich_host_for_api({"host_id": "h-region", "country": "CA", "province": "ON"})
    assert out["region"] == "ca-on"



@pytest.mark.parametrize("country,province,region", [
    ("ca", "on", "ca-on"), ("US", "NY", "us-ny"), ("GB", "ENG", "gb-eng"),
    ("DE", "BE", "de-be"), ("AU", "NSW", "au-nsw"),
])
def test_exact_location_uses_the_shared_iso_catalog(country, province, region):
    from host_metadata import require_location

    assert require_location(country, province)["region"] == region


@pytest.mark.parametrize("country,province,region", [
    ("", "", ""), ("XX", "", ""), ("CA", "", "ca-central"),
    ("CA", "", "ca-east"), ("CA", "Ontario", ""),
    ("CA", "ON", "ca-east"), ("US", "ON", ""),
])
def test_missing_ambiguous_or_conflicting_location_is_an_error(country, province, region):
    from host_metadata import require_location

    with pytest.raises(ValueError):
        require_location(country, province, region)


def test_no_region_is_invented_for_an_incomplete_record():
    assert normalize_region("") == ""
    assert normalize_host_region({}) == ""


def test_country_choices_are_complete_names_from_iso_data():
    import pycountry
    from host_metadata import location_catalog

    catalog = location_catalog("CA")
    assert {c["code"] for c in catalog["countries"]} == {c.alpha_2 for c in pycountry.countries}
    assert next(c for c in catalog["countries"] if c["code"] == "CA")["name"] == "Canada"
    assert next(s for s in catalog["subdivisions"] if s["code"] == "ON")["name"] == "Ontario"


def test_profile_completion_requires_every_required_account_field():
    from host_metadata import profile_missing_fields

    valid = {"name":"Account Owner", "email":"owner@example.test", "country":"CA", "province":"ON",
             "account_type":"individual", "onboarding_intent":"rent", "address_line1":"1 Test Street",
             "city":"London", "postal_code":"N6A 1A1"}
    assert profile_missing_fields(valid) == []
    for field in valid:
        assert profile_missing_fields({**valid, field:""})


def test_address_requirements_follow_each_country_not_a_canadian_default():
    from host_metadata import address_required_fields, profile_address_values

    assert "postal_code" in address_required_fields("CA")
    assert "postal_code" not in address_required_fields("HK")
    _, errors = profile_address_values({"country":"CA", "province":"ON", "address_line1":"1 Test Street", "city":"London", "postal_code":"V6B 1A1"})
    assert errors == ["postal_code"]  # A BC postal prefix cannot be saved as Ontario.


def test_address_metadata_covers_every_country_we_offer():
    import pycountry
    from host_metadata import address_required_fields

    for country in pycountry.countries:
        assert isinstance(address_required_fields(country.alpha_2), list)
