"""Host metadata normalization for dashboard + worker heartbeats."""

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

from host_metadata import (
    enrich_host_for_api,
    infer_gpu_from_host_id,
    merge_host_update,
    normalize_host_region,
    normalize_gpu_model,
    normalize_region,
    platform_rate_cad,
)


def test_normalize_gpu_model_legacy_dash_and_nvidia_prefix():
    assert normalize_gpu_model("RTX-3060") == "RTX 3060"
    assert normalize_gpu_model("NVIDIA GeForce RTX 3060") == "RTX 3060"
    assert normalize_gpu_model("RTX 4090") == "RTX 4090"


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
    assert out["cost_per_hour"] > 0


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
