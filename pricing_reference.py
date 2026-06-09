"""Live GPU pricing reference built from ``gpu_pricing`` (canonical variant per model)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger("xcelsior.pricing_reference")

_MODE_FIELDS = {
    "on_demand": "base_rate_cad",
    "spot": "spot_cad",
    "reserved_1mo": "reserved_1mo_cad",
    "reserved_3mo": "reserved_3mo_cad",
    "reserved_1yr": "reserved_1yr_cad",
}

_COMMITMENT_TO_MODE = {
    "1_month": "reserved_1mo",
    "3_month": "reserved_3mo",
    "1_year": "reserved_1yr",
}


def commitment_pricing_mode(commitment_type: str) -> str | None:
    return _COMMITMENT_TO_MODE.get(commitment_type)


def _pricing_row(row) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        return dict(row)
    return {
        "gpu_model": row[0],
        "vram_gb": row[1],
        "tier": row[2],
        "pricing_mode": row[3],
        "base_rate_cad": row[4],
    }


def _canonical_rates_by_model(rows: list) -> dict[str, dict[str, Any]]:
    """Pick lowest-VRAM variant per model; collect standard/premium/sovereign rates."""
    canonical_vram: dict[str, int] = {}
    by_model: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for raw in rows:
        row = _pricing_row(raw)
        model = str(row["gpu_model"])
        vram = int(row["vram_gb"] or 0)
        tier = str(row["tier"])
        mode = str(row["pricing_mode"])
        rate = float(row["base_rate_cad"] or 0)
        if model not in canonical_vram:
            canonical_vram[model] = vram
        if vram != canonical_vram[model]:
            continue
        by_model[model][tier][mode] = rate

    out: dict[str, dict[str, Any]] = {}
    for model, tiers in by_model.items():
        standard = tiers.get("standard", {})
        premium = tiers.get("premium", {})
        sovereign = tiers.get("sovereign", {})
        on_demand = standard.get("on_demand", 0.0)
        spot = standard.get("spot", 0.0)
        if on_demand <= 0 and not spot:
            continue
        entry: dict[str, Any] = {
            "vram_gb": canonical_vram[model],
            "base_rate_cad": round(on_demand, 4),
            "spot_cad": round(spot, 4),
            "reserved_1mo_cad": round(standard.get("reserved_1mo", 0.0), 4),
            "reserved_3mo_cad": round(standard.get("reserved_3mo", 0.0), 4),
            "reserved_1yr_cad": round(standard.get("reserved_1yr", 0.0), 4),
            "premium_rate_cad": round(premium.get("on_demand", on_demand * 1.3), 4),
            "subsidized_starter_cad": round(spot or on_demand * 0.4, 4),
            "min_rate_cad": round(spot or on_demand * 0.4, 4),
            "max_rate_cad": round(sovereign.get("on_demand", on_demand * 1.43), 4),
        }
        out[model] = entry
    return out


def build_gpu_pricing_reference() -> dict[str, dict[str, Any]]:
    """Return per-GPU reference pricing from live ``gpu_pricing`` rows."""
    from db import pg_connection

    try:
        with pg_connection() as conn:
            rows = conn.execute(
                """SELECT gpu_model, vram_gb, tier, pricing_mode, base_rate_cad
                   FROM gpu_pricing
                   WHERE active = TRUE
                   ORDER BY gpu_model, vram_gb ASC, tier, pricing_mode"""
            ).fetchall()
    except Exception as exc:
        log.warning("gpu_pricing reference query failed: %s", exc)
        return {}

    if not rows:
        return {}

    pricing = _canonical_rates_by_model(rows)
    return dict(sorted(pricing.items(), key=lambda kv: kv[1].get("base_rate_cad", 0)))


def build_reference_list(pricing: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Flat list shape consumed by ``frontend/src/lib/api.ts``."""
    data = pricing if pricing is not None else build_gpu_pricing_reference()
    out: list[dict[str, Any]] = []
    for model, rates in data.items():
        out.append(
            {
                "gpu_model": model,
                "vram_gb": rates.get("vram_gb"),
                "on_demand_cad": rates.get("base_rate_cad"),
                "spot_cad": rates.get("spot_cad"),
                "reserved_1mo_cad": rates.get("reserved_1mo_cad"),
                "reserved_3mo_cad": rates.get("reserved_3mo_cad"),
                "reserved_1yr_cad": rates.get("reserved_1yr_cad"),
            }
        )
    return out


def get_on_demand_rate(gpu_model: str, *, tier: str = "standard") -> float:
    """Look up on-demand rate for reserve/commitment flows."""
    from db import pg_connection

    try:
        with pg_connection() as conn:
            row = conn.execute(
                """SELECT base_rate_cad FROM gpu_pricing
                   WHERE gpu_model = %s AND tier = %s AND pricing_mode = 'on_demand'
                     AND active = TRUE
                   ORDER BY vram_gb ASC
                   LIMIT 1""",
                (gpu_model, tier),
            ).fetchone()
        if row:
            return float(row[0])
    except Exception as exc:
        log.debug("on_demand lookup failed for %s: %s", gpu_model, exc)
    return 0.0


def get_reserved_rate(gpu_model: str, commitment_type: str, *, tier: str = "standard") -> float:
    mode = commitment_pricing_mode(commitment_type)
    if not mode:
        return 0.0
    from db import pg_connection

    try:
        with pg_connection() as conn:
            row = conn.execute(
                """SELECT base_rate_cad FROM gpu_pricing
                   WHERE gpu_model = %s AND tier = %s AND pricing_mode = %s
                     AND active = TRUE
                   ORDER BY vram_gb ASC
                   LIMIT 1""",
                (gpu_model, tier, mode),
            ).fetchone()
        if row:
            return float(row[0])
    except Exception as exc:
        log.debug("reserved lookup failed for %s/%s: %s", gpu_model, commitment_type, exc)
    return 0.0