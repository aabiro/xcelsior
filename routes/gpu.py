"""Routes: gpu."""

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    log,
)

router = APIRouter()


@router.get("/api/v2/gpu/available", tags=["GPU"])
def api_gpu_available(request: Request):
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
        from host_metadata import normalize_gpu_model, normalize_region
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        gpu_map = {}
        offers_seen = False
        hosts_seen = False

        def add_gpu(
            *,
            gpu_model: str,
            vram_gb: float,
            region: str,
            province: str = "",
            count_available: int = 0,
            price_per_hour_cad: float = 0,
        ) -> None:
            model = normalize_gpu_model(gpu_model)
            if not model:
                return
            province_code = (province or "").upper()
            normalized_region = normalize_region(
                region,
                country="CA" if province_code else "",
                province=province_code,
            )
            key = (model, float(vram_gb or 0), normalized_region, province_code)
            price = round(float(price_per_hour_cad or 0), 2)
            existing = gpu_map.get(key)
            if existing:
                existing["count_available"] = max(
                    int(existing.get("count_available") or 0),
                    int(count_available or 0),
                )
                if price > 0:
                    existing_price = float(existing.get("price_per_hour_cad") or 0)
                    existing["price_per_hour_cad"] = (
                        min(existing_price, price) if existing_price > 0 else price
                    )
                return

            gpu_map[key] = {
                "gpu_model": model,
                "vram_gb": vram_gb,
                "region": normalized_region,
                "province": province_code,
                "count_available": int(count_available or 0),
                "price_per_hour_cad": price,
            }

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
                offers_seen = True
                add_gpu(
                    gpu_model=r.get("gpu_model", ""),
                    vram_gb=r.get("vram_gb", 0),
                    region=r.get("region", ""),
                    province=r.get("province", ""),
                    count_available=r.get("count_available", 0),
                    price_per_hour_cad=(
                        float(r["min_price_cents"]) / 100 if r.get("min_price_cents") else 0
                    ),
                )
        except Exception as e:
            log.debug("gpu_offers query failed (table may not exist): %s", e)
        try:
            # Always merge registered hosts. This fills gaps when gpu_offers is
            # partial or stale, while max-count de-duping avoids double counting
            # hosts already mirrored into offers.
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
                hosts_seen = True
                add_gpu(
                    gpu_model=h.get("gpu_model", ""),
                    vram_gb=h.get("total_vram_gb", 0),
                    region=h.get("region", ""),
                    province=h.get("province", ""),
                    count_available=h.get("count_available", 0),
                    price_per_hour_cad=h.get("min_price", 0),
                )
        except Exception as e:
            log.debug("hosts GPU availability query failed: %s", e)

        gpus = sorted(
            gpu_map.values(),
            key=lambda g: (str(g.get("gpu_model") or ""), str(g.get("region") or "")),
        )
        if offers_seen and hosts_seen:
            source = "gpu_offers+hosts"
        elif offers_seen:
            source = "gpu_offers"
        elif hosts_seen:
            source = "hosts"
        else:
            source = "none"
            log.warning("GPU availability: no GPUs found in gpu_offers or hosts tables")
        return {"ok": True, "gpus": gpus, "source": source}
    except Exception as e:
        log.error("GPU availability query failed: %s", e)
        raise HTTPException(503, f"GPU availability service unavailable: {e}")
