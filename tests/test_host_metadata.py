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
