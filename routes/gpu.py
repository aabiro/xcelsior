"""Routes: gpu."""

from fastapi import APIRouter, HTTPException

from routes._deps import (
    log,
)

router = APIRouter()

@router.get("/api/v2/gpu/available", tags=["GPU"])
def api_gpu_available(request: 'Request' = None):
    """List available GPU types with regions, VRAM, pricing, and counts.

    Used by both Serverless and Volumes to populate GPU/region pickers.
    Queries gpu_offers first, then hosts table. If neither has data,
    returns an empty list — no fake inventory.
    """
    from routes._deps import _require_scope, _get_current_user
    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "gpu:read")
    try:
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        gpus = []
        source = "gpu_offers"
        try:
            with pool.connection() as conn:
                conn.row_factory = dict_row
                rows = conn.execute(
                    """SELECT gpu_model, vram_gb, region, province,
                              COUNT(*) FILTER (WHERE available = true) AS count_available,
                              MIN(ask_cents_per_hour) AS min_price_cents
                       FROM gpu_offers
                       GROUP BY gpu_model, vram_gb, region, province
                       ORDER BY gpu_model, region""",
                ).fetchall()
            for r in rows:
                gpus.append({
                    "gpu_model": r["gpu_model"],
                    "vram_gb": r["vram_gb"],
                    "region": r["region"],
                    "province": r.get("province", ""),
                    "count_available": r["count_available"],
                    "price_per_hour_cad": round(r["min_price_cents"] / 100, 2) if r["min_price_cents"] else 0,
                })
        except Exception as e:
            log.debug("gpu_offers query failed (table may not exist): %s", e)
            gpus = []
        if not gpus:
            # Fallback: derive from registered hosts
            source = "hosts"
            with pool.connection() as conn:
                conn.row_factory = dict_row
                hosts = conn.execute(
                    """SELECT payload->>'gpu_model' AS gpu_model,
                              (payload->>'total_vram_gb')::float AS total_vram_gb,
                              payload->>'region' AS region,
                              payload->>'province' AS province,
                              COUNT(*) FILTER (WHERE status = 'active') AS count_available,
                              MIN((payload->>'cost_per_hour')::float) AS min_price
                       FROM hosts
                       WHERE COALESCE(payload->>'admitted', '') IN ('true', 'True', '1', 'yes')
                       GROUP BY payload->>'gpu_model', payload->>'total_vram_gb', payload->>'region', payload->>'province'
                       ORDER BY payload->>'gpu_model'""",
                ).fetchall()
            for h in hosts:
                gpus.append({
                    "gpu_model": h.get("gpu_model", "Unknown"),
                    "vram_gb": h.get("total_vram_gb", 0),
                    "region": h.get("region", "ca-east"),
                    "province": h.get("province", ""),
                    "count_available": h.get("count_available", 0),
                    "price_per_hour_cad": round(float(h.get("min_price", 0)), 2),
                })
        if not gpus:
            source = "none"
            log.warning("GPU availability: no GPUs found in gpu_offers or hosts tables")
        return {"ok": True, "gpus": gpus, "source": source}
    except Exception as e:
        log.error("GPU availability query failed: %s", e)
        raise HTTPException(503, f"GPU availability service unavailable: {e}")

