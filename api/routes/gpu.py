"""Routes: gpu."""

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    log,
)

router = APIRouter()


@router.get("/api/v2/gpu/available", tags=["GPU"])
def api_gpu_available(request: Request, region: str | None = None):
    """Free physical GPUs on admitted, ready hosts with fresh observations.

    Use the same device/allocation snapshot as placement. Provider offers are
    pricing metadata; they never create devices or override host geography.
    """
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "gpu:read")
    try:
        from db import _get_pg_pool
        from host_metadata import normalize_gpu_model, normalize_region, require_location
        from psycopg.rows import dict_row
        from control_plane.scheduler.config import SchedulerConfig
        from control_plane.scheduler.filters import FilterContext, host_admitted, host_ready
        from control_plane.scheduler.snapshot import read_fleet

        with _get_pg_pool().connection() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            fleet = read_fleet(
                conn,
                host_freshness_timeout_sec=SchedulerConfig.from_env().host_freshness_timeout_sec,
            )
            with conn.cursor(row_factory=dict_row) as cur:
                offers = cur.execute(
                    """SELECT host_id, gpu_model, MIN(ask_cents_per_hour) AS price_cents
                         FROM gpu_offers
                        WHERE available = true AND gpu_count_available > 0
                        GROUP BY host_id, gpu_model"""
                ).fetchall()

        prices = {
            (str(offer["host_id"]), normalize_gpu_model(offer["gpu_model"])):
            float(offer["price_cents"] or 0) / 100
            for offer in offers
        }
        ctx = FilterContext(stale_host_ids=fleet.stale_host_ids)
        requested_region = normalize_region(region, default="") if region else None
        gpu_map = {}
        for host in fleet.hosts:
            if host_admitted({}, host, ctx) or host_ready({}, host, ctx):
                continue
            try:
                location = require_location(
                    host.get("country", ""), host.get("province", ""), host.get("region", ""),
                )
            except ValueError:
                continue
            host_region = location["region"]
            if requested_region is not None and host_region != requested_region:
                continue
            province = str(host.get("province") or "").upper()
            for device in host["gpu_devices"]:
                if not device["free"]:
                    continue
                model = normalize_gpu_model(device["model"])
                vram_gb = device["total_vram_mb"] / 1024
                if not model or vram_gb <= 0:
                    continue
                key = (model, vram_gb, host_region, province)
                observed_price = prices.get(
                    (str(host["host_id"]), model), float(host.get("cost_per_hour") or 0)
                )
                gpu = gpu_map.setdefault(key, {
                    "gpu_model": model, "vram_gb": vram_gb,
                    "region": host_region, "province": province,
                    "count_available": 0, "price_per_hour_cad": 0,
                })
                gpu["count_available"] += 1
                current_price = gpu["price_per_hour_cad"]
                if observed_price > 0:
                    gpu["price_per_hour_cad"] = round(
                        min(current_price, observed_price) if current_price > 0 else observed_price, 2
                    )
        gpus = sorted(
            gpu_map.values(),
            key=lambda g: (g["gpu_model"], g["region"], g["vram_gb"], g["province"]),
        )
        # A launch charges the imported Stripe Price, so show that starting
        # price where the variant has one. Inventory without a sellable SKU
        # stays listed with its observed rate and no purchasable tiers.
        from service_tier_pricing import tier_catalog, TierPricingUnavailable
        from money import micros_to_cad

        for gpu in gpus:
            try:
                catalog = tier_catalog(gpu["gpu_model"], vram_gb=round(gpu["vram_gb"] or 0) or None)
            except (TierPricingUnavailable, ValueError):
                gpu["service_tiers_available"] = False
                continue
            gpu["service_tiers_available"] = True
            gpu["price_per_hour_cad"] = micros_to_cad(catalog["base_rate_micros"])
        source = "host_gpu_devices" if gpus else "none"
        return {"ok": True, "gpus": gpus, "source": source}
    except Exception as e:
        log.error("GPU availability query failed: %s", e)
        raise HTTPException(503, f"GPU availability service unavailable: {e}")
