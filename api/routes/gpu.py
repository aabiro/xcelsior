"""Routes: gpu."""

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    log,
)

router = APIRouter()


@router.get("/api/v2/gpu/available", tags=["GPU"])
def api_gpu_available(request: Request, region: str | None = None):
    """Free GPUs plus admitted inventory and online/allocated capacity counts.

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
        from control_plane.scheduler.filters import (
            FilterContext,
            host_admitted,
            host_ready,
            host_verified,
        )
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
            (str(offer["host_id"]), normalize_gpu_model(offer["gpu_model"])): float(
                offer["price_cents"] or 0
            )
            / 100
            for offer in offers
        }
        ctx = FilterContext(stale_host_ids=fleet.stale_host_ids)
        requested_region = normalize_region(region, default="") if region else None
        gpu_map = {}
        gpu_hosts: dict[tuple, set[str]] = {}
        inventory_map = {}
        capacity = {
            "hosts_online": 0,
            "hosts_ready": 0,
            "gpus_total": 0,
            "gpus_available": 0,
            "gpus_allocated": 0,
            "gpus_held": 0,
            "gpus_unavailable": 0,
        }
        for host in fleet.hosts:
            if host_admitted({}, host, ctx):
                continue
            try:
                location = require_location(
                    host.get("country", ""),
                    host.get("province", ""),
                    host.get("region", ""),
                )
            except ValueError:
                continue
            host_region = location["region"]
            if requested_region is not None and host_region != requested_region:
                continue
            # Ready means placement would take it: with verified-only placement
            # on, a host whose verification is not current counts as online
            # but its GPUs as unavailable, never as free.
            ready = host_ready({}, host, ctx) is None and host_verified({}, host, ctx) is None
            online = str(host["host_id"]) not in fleet.stale_host_ids and str(
                host.get("status") or ""
            ).lower() in ("active", "ready")
            capacity["hosts_online"] += int(online)
            capacity["hosts_ready"] += int(ready)
            province = str(host.get("province") or "").upper()
            for device in host["gpu_devices"]:
                model = normalize_gpu_model(device["model"])
                vram_gb = device["total_vram_mb"] / 1024
                if not model or vram_gb <= 0:
                    continue
                key = (model, vram_gb, host_region, province)
                inventory = inventory_map.setdefault(
                    key,
                    {
                        "gpu_model": model,
                        "vram_gb": vram_gb,
                        "region": host_region,
                        "province": province,
                        "count_total": 0,
                        "count_available": 0,
                        "count_allocated": 0,
                        "count_held": 0,
                        "count_unavailable": 0,
                    },
                )
                # These states partition physical devices. A preemption hold on
                # an allocated GPU must not count that same GPU twice.
                if device["allocated"]:
                    state = "allocated"
                elif device["held_by_plan"]:
                    state = "held"
                elif ready and device["free"]:
                    state = "available"
                else:
                    state = "unavailable"
                inventory["count_total"] += 1
                inventory[f"count_{state}"] += 1
                capacity["gpus_total"] += 1
                capacity[f"gpus_{state}"] += 1
                if state != "available":
                    continue
                observed_price = prices.get(
                    (str(host["host_id"]), model), float(host.get("cost_per_hour") or 0)
                )
                gpu = gpu_map.setdefault(
                    key,
                    {
                        "gpu_model": model,
                        "vram_gb": vram_gb,
                        "region": host_region,
                        "province": province,
                        "count_available": 0,
                        "price_per_hour_cad": 0,
                    },
                )
                gpu["count_available"] += 1
                gpu_hosts.setdefault(key, set()).add(str(host["host_id"]))
                current_price = gpu["price_per_hour_cad"]
                if observed_price > 0:
                    gpu["price_per_hour_cad"] = round(
                        min(current_price, observed_price) if current_price > 0 else observed_price,
                        2,
                    )
        gpus = sorted(
            gpu_map.values(),
            key=lambda g: (g["gpu_model"], g["region"], g["vram_gb"], g["province"]),
        )
        # A launch charges the imported Stripe Price, so show that starting
        # price where the variant has one: the same listed rate marketplace
        # search shows. Inventory without a sellable SKU stays listed with its
        # observed rate and no purchasable tiers.
        from service_tier_pricing import listed_rate_lookup
        from money import micros_to_cad

        listed_rate = listed_rate_lookup()
        from host_evidence import evidence_for_hosts, evidence_summary

        evidence = evidence_for_hosts(sorted(set().union(*gpu_hosts.values())))
        for key, gpu in gpu_map.items():
            micros = listed_rate(gpu["gpu_model"], round(gpu["vram_gb"] or 0) or None)
            gpu["service_tiers_available"] = micros is not None
            if micros is not None:
                gpu["price_per_hour_cad"] = micros_to_cad(micros)
            # A row aggregates hosts; the summary says how many of them are
            # verified and when one was last benchmarked. get_host_evidence
            # (or search_marketplace) gives each host's own.
            host_ids = sorted(gpu_hosts.get(key, ()))
            gpu["host_ids"] = host_ids
            gpu["evidence"] = evidence_summary([evidence.get(h) for h in host_ids])
        source = "host_gpu_devices" if gpus else "none"
        inventory = sorted(
            inventory_map.values(),
            key=lambda g: (g["gpu_model"], g["region"], g["vram_gb"], g["province"]),
        )
        return {
            "ok": True,
            "gpus": gpus,
            "source": source,
            "capacity": capacity,
            "inventory": inventory,
            # True while only currently verified hosts take jobs; their GPUs
            # alone are counted as available.
            "verified_only": ctx.require_verified,
            "observed_at": fleet.taken_at.isoformat(),
        }
    except Exception as e:
        log.error("GPU availability query failed: %s", e)
        raise HTTPException(503, f"GPU availability service unavailable: {e}")
