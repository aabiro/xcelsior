"""Normalize and enrich host records for API responses and worker heartbeats."""

from __future__ import annotations

import re
from typing import Any

from db import _GPU_PRICING_BASE

# Canonical GPU titles — must match frontend gpu-models.ts and db._GPU_PRICING_BASE.
CANONICAL_GPU_MODELS: frozenset[str] = frozenset({row[0] for row in _GPU_PRICING_BASE})

_DEFAULT_VRAM_BY_MODEL: dict[str, int] = {}
_DEFAULT_RATE_BY_MODEL: dict[str, float] = {}
for _model, _vram, _ff, _hf, _rate in _GPU_PRICING_BASE:
    if _model not in _DEFAULT_VRAM_BY_MODEL or _vram < _DEFAULT_VRAM_BY_MODEL[_model]:
        _DEFAULT_VRAM_BY_MODEL[_model] = int(_vram)
    if _model not in _DEFAULT_RATE_BY_MODEL:
        _DEFAULT_RATE_BY_MODEL[_model] = float(_rate)

_NVIDIA_PREFIX_RE = re.compile(r"^NVIDIA\s+(GeForce\s+)?", re.IGNORECASE)
_DEFAULT_REGION = "ca-east"
_CA_PROVINCES = frozenset(
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


def normalize_gpu_model(raw: str) -> str:
    """Map worker/dashboard aliases to canonical short GPU titles."""
    model = (raw or "").strip()
    if not model:
        return ""

    if model in CANONICAL_GPU_MODELS:
        return model

    model = _NVIDIA_PREFIX_RE.sub("", model).strip()

    # Legacy dashed IDs from an older dashboard validator (RTX-3060 → RTX 3060).
    if "-" in model and " " not in model:
        dashed_as_spaced = model.replace("-", " ")
        if dashed_as_spaced in CANONICAL_GPU_MODELS:
            return dashed_as_spaced

    # Case-insensitive match against catalogue.
    lower = model.lower()
    for canonical in CANONICAL_GPU_MODELS:
        if canonical.lower() == lower:
            return canonical

    return model


def default_vram_gb(gpu_model: str) -> float:
    model = normalize_gpu_model(gpu_model)
    return float(_DEFAULT_VRAM_BY_MODEL.get(model, 0))


def platform_rate_cad(gpu_model: str, *, pricing_mode: str = "on_demand") -> float:
    """Resolve the platform list rate for a GPU model."""
    model = normalize_gpu_model(gpu_model)
    if not model:
        return 0.0

    try:
        from pricing_reference import get_on_demand_rate

        if pricing_mode == "on_demand":
            rate = float(get_on_demand_rate(model))
            if rate > 0:
                return rate
    except Exception:
        pass

    try:
        from spot_pricing import get_platform_rate

        rate = get_platform_rate(model, pricing_mode=pricing_mode)
        if rate and rate > 0:
            return float(rate)
    except Exception:
        pass

    return float(_DEFAULT_RATE_BY_MODEL.get(model, 0.0))


def normalize_region(
    region: str | None = "",
    *,
    country: str | None = "",
    province: str | None = "",
    default: str = _DEFAULT_REGION,
) -> str:
    """Normalize stored host/offer regions for API selectors and placement."""
    value = str(region or "").strip().replace("_", "-")
    province_code = str(province or "").strip().upper()
    country_code = str(country or "").strip().upper()

    if value:
        upper = value.upper()
        if upper in _CA_PROVINCES:
            return f"ca-{upper.lower()}"
        if upper.startswith("CA-") and upper[3:] in _CA_PROVINCES:
            return f"ca-{upper[3:].lower()}"
        return value.lower()

    if country_code == "CA" and province_code in _CA_PROVINCES:
        return f"ca-{province_code.lower()}"
    if country_code in {"US", "EU", "AP"}:
        return country_code.lower()
    return default


def normalize_host_region(host: dict[str, Any]) -> str:
    """Return the canonical region for a host payload."""
    return normalize_region(
        str(host.get("region") or ""),
        country=str(host.get("country") or ""),
        province=str(host.get("province") or ""),
    )


def infer_gpu_from_host_id(host_id: str) -> str:
    """Best-effort GPU model from host_id tokens (e.g. gpu-3060-lab → RTX 3060)."""
    raw = (host_id or "").strip().lower()
    if not raw:
        return ""

    compact = re.sub(r"[\s\-_]+", "", raw)
    for canonical in sorted(CANONICAL_GPU_MODELS, key=len, reverse=True):
        key = canonical.lower().replace(" ", "")
        if key in compact:
            return canonical

    match = re.search(r"(?:rtx[\-_ ]?)?(\d{4})", raw)
    if match:
        candidate = f"RTX {match.group(1)}"
        if candidate in CANONICAL_GPU_MODELS:
            return candidate

    return ""


def _display_hostname(host: dict[str, Any]) -> str:
    explicit = str(host.get("hostname") or "").strip()
    if explicit:
        return explicit
    host_id = str(host.get("host_id") or "").strip()
    if not host_id:
        return ""
    # gpu-2060-aaryn-local → gpu 2060 aaryn local
    return host_id.replace("_", " ").replace("-", " ").strip()


def enrich_host_for_api(host: dict[str, Any]) -> dict[str, Any]:
    """Add dashboard-friendly aliases and fill missing catalogue fields."""
    out = dict(host)
    gpu = normalize_gpu_model(str(out.get("gpu_model") or ""))
    if not gpu:
        gpu = infer_gpu_from_host_id(str(out.get("host_id") or ""))
    if gpu:
        out["gpu_model"] = gpu

    total = float(out.get("total_vram_gb") or out.get("vram_gb") or 0)
    if total <= 0 and gpu:
        total = default_vram_gb(gpu)
    if total > 0:
        out["total_vram_gb"] = total
        out["vram_gb"] = total

    rate = float(out.get("cost_per_hour") or out.get("price_per_hour") or 0)
    if rate <= 0 and gpu:
        rate = platform_rate_cad(gpu)
    if rate > 0:
        out["cost_per_hour"] = rate
        out["price_per_hour"] = rate

    hostname = _display_hostname(out)
    if hostname:
        out["hostname"] = hostname

    out["region"] = normalize_host_region(out)

    return out


def merge_host_update(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge a worker heartbeat into a stored host without clobbering dashboard metadata."""
    merged = dict(incoming)
    ex = existing or {}

    for field in (
        "hostname",
        "notes",
        "owner",
        "provider_id",
        "user_id",
        "country",
        "province",
        "region",
        "corporation_name",
        "business_number",
        "gst_hst_number",
        "legal_name",
        "admitted",
        "admission_details",
        "recommended_runtime",
        "compute_score",
        "registered_at",
    ):
        if ex.get(field) and not merged.get(field):
            merged[field] = ex[field]

    incoming_gpu = normalize_gpu_model(str(merged.get("gpu_model") or ""))
    existing_gpu = normalize_gpu_model(str(ex.get("gpu_model") or ""))
    if incoming_gpu:
        merged["gpu_model"] = incoming_gpu
    elif existing_gpu:
        merged["gpu_model"] = existing_gpu
    else:
        inferred = infer_gpu_from_host_id(str(merged.get("host_id") or ex.get("host_id") or ""))
        if inferred:
            merged["gpu_model"] = inferred

    incoming_total = float(merged.get("total_vram_gb") or 0)
    existing_total = float(ex.get("total_vram_gb") or ex.get("vram_gb") or 0)
    if incoming_total > 0:
        merged["total_vram_gb"] = incoming_total
    elif existing_total > 0:
        merged["total_vram_gb"] = existing_total
    elif merged.get("gpu_model"):
        merged["total_vram_gb"] = default_vram_gb(str(merged["gpu_model"]))

    incoming_free = float(merged.get("free_vram_gb") or 0)
    if incoming_free <= 0 and merged.get("total_vram_gb"):
        merged["free_vram_gb"] = float(merged["total_vram_gb"])
    elif incoming_free > 0:
        merged["free_vram_gb"] = incoming_free
    elif ex.get("free_vram_gb"):
        merged["free_vram_gb"] = ex["free_vram_gb"]

    incoming_cost = float(merged.get("cost_per_hour") or 0)
    existing_cost = float(ex.get("cost_per_hour") or 0)
    platform = platform_rate_cad(str(merged.get("gpu_model") or ""))
    if incoming_cost > 0:
        merged["cost_per_hour"] = incoming_cost
    elif existing_cost > 0:
        merged["cost_per_hour"] = existing_cost
    elif platform > 0:
        merged["cost_per_hour"] = platform

    if not merged.get("hostname"):
        merged["hostname"] = _display_hostname(ex) or _display_hostname(merged)

    return merged
