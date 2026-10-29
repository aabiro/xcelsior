"""GPU availability derived from admitted, observed physical device capacity."""

from fastapi import APIRouter, HTTPException, Request

from routes._deps import log

router = APIRouter()


@router.get("/api/v2/gpu/available", tags=["GPU"])
def api_gpu_available(request: Request):
    """List schedulable physical GPU variants and Stripe-backed starting prices."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "gpu:read")
    try:
        from db import pg_connection
        from control_plane.scheduler.snapshot import read_fleet
        from host_metadata import normalize_gpu_model
        from service_tier_pricing import tier_catalog, TierPricingUnavailable
        from money import micros_to_cad

        with pg_connection() as conn:
            fleet = read_fleet(conn)
        groups = {}
        for host in fleet.hosts:
            if (
                host.get("administrative_state") != "admitted"
                or host.get("availability_state") != "ready"
                or host.get("status") != "active"
                or host["host_id"] in fleet.stale_host_ids
            ):
                continue
            for device in host.get("gpu_devices", []):
                model = normalize_gpu_model(device["model"])
                memory = round(device["allocatable_vram_mb"] / 1024)
                key = (model, memory, host.get("region") or "", host.get("province") or "")
                if key not in groups:
                    try:
                        catalog = tier_catalog(model, vram_gb=memory)
                        rate = micros_to_cad(catalog["base_rate_micros"])
                    except TierPricingUnavailable:
                        # An observed device without a sellable Stripe SKU
                        # remains provider inventory, not a purchase option.
                        continue
                    groups[key] = {
                        "gpu_model": model,
                        "vram_gb": memory,
                        "region": key[2],
                        "province": key[3],
                        "count_available": 0,
                        "price_per_hour_cad": rate,
                    }
                groups[key]["count_available"] += int(device["free"])
        return {
            "ok": True,
            "gpus": [groups[key] for key in sorted(groups)],
            "source": "physical_inventory",
        }
    except Exception as exc:
        log.error("GPU availability query failed (%s)", type(exc).__name__)
        raise HTTPException(503, "GPU availability service unavailable") from exc
