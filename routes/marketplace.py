"""Routes: marketplace."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _get_current_user,
)
from marketplace import get_marketplace_engine
from scheduler import get_marketplace, list_rig, marketplace_bill, marketplace_stats, unlist_rig

router = APIRouter()

@router.get("/marketplace/search", tags=["Marketplace"])
def api_marketplace_search(
    gpu_model: str | None = None,
    min_vram: float | None = None,
    max_price: float | None = None,
    province: str | None = None,
    country: str | None = None,
    min_reputation: int | None = None,
    sort_by: str = "price",
    limit: int = 50,
):
    """Search marketplace listings with filters and sorting."""
    listings = get_marketplace(active_only=True)

    if gpu_model:
        listings = [l for l in listings if gpu_model.lower() in (l.get("gpu_model", "")).lower()]
    if min_vram is not None:
        listings = [l for l in listings if (l.get("vram_gb", 0) or 0) >= min_vram]
    if max_price is not None:
        listings = [l for l in listings if (l.get("price_per_hour", 999) or 999) <= max_price]
    if province:
        listings = [l for l in listings if l.get("province", "").upper() == province.upper()]
    if country:
        listings = [l for l in listings if (l.get("country", "").upper()) == country.upper()]
    if min_reputation is not None:
        listings = [l for l in listings if (l.get("reputation_score", 0) or 0) >= min_reputation]

    # Sort
    sort_keys = {
        "price": lambda x: x.get("price_per_hour", 999),
        "vram": lambda x: -(x.get("vram_gb", 0) or 0),
        "reputation": lambda x: -(x.get("reputation_score", 0) or 0),
        "score": lambda x: -(x.get("compute_score", 0) or 0),
    }
    if sort_by in sort_keys:
        listings.sort(key=sort_keys[sort_by])

    return {
        "ok": True,
        "total": len(listings),
        "listings": listings[:limit],
        "filters_applied": {
            "gpu_model": gpu_model,
            "min_vram": min_vram,
            "max_price": max_price,
            "province": province,
            "country": country,
            "min_reputation": min_reputation,
            "sort_by": sort_by,
        },
    }


# ── Model: RigListing ──

class RigListing(BaseModel):
    host_id: str
    gpu_model: str
    vram_gb: float
    price_per_hour: float
    description: str = ""
    owner: str = "anonymous"

@router.post("/marketplace/list", tags=["Marketplace"])
def api_list_rig(rig: RigListing, request: Request):
    """List a rig on the marketplace."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    listing = list_rig(
        rig.host_id, rig.gpu_model, rig.vram_gb, rig.price_per_hour, rig.description, rig.owner
    )
    return {"ok": True, "listing": listing}

@router.delete("/marketplace/{host_id}", tags=["Marketplace"])
def api_unlist_rig(host_id: str, request: Request):
    """Remove a rig from the marketplace."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    if not unlist_rig(host_id):
        raise HTTPException(status_code=404, detail=f"Listing {host_id} not found")
    return {"ok": True, "unlisted": host_id}

@router.get("/marketplace", tags=["Marketplace"])
def api_get_marketplace(request: Request, active_only: bool = True):
    """Browse marketplace listings."""
    user = _get_current_user(request) if request else None
    if user:
        from routes._deps import _require_scope
        _require_scope(user, "marketplace:read")
    return {"listings": get_marketplace(active_only=active_only)}

@router.post("/marketplace/bill/{job_id}", tags=["Marketplace"])
def api_marketplace_bill(job_id: str, request: Request):
    """Bill a marketplace job — split between host and platform."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    result = marketplace_bill(job_id)
    if not result:
        raise HTTPException(status_code=400, detail=f"Could not bill marketplace job {job_id}")
    return {"ok": True, "bill": result}

@router.get("/marketplace/stats", tags=["Marketplace"])
def api_marketplace_stats(request: Request):
    """Marketplace aggregate stats."""
    user = _get_current_user(request) if request else None
    if user:
        from routes._deps import _require_scope
        _require_scope(user, "marketplace:read")
    return {"stats": marketplace_stats()}


# ── Model: GPUOfferCreate ──

class GPUOfferCreate(BaseModel):
    host_id: str
    gpu_model: str
    gpu_count_total: int = 1
    vram_gb: float = 0
    ask_cents_per_hour: int = 20
    region: str = "ca-east"
    spot_enabled: bool = True
    spot_min_cents: int = 10

@router.post("/api/v2/marketplace/offers", tags=["Marketplace v2"])
def api_marketplace_create_offer(body: GPUOfferCreate, request: Request):
    """Create or update a GPU offer on the marketplace."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    offer = me.upsert_offer(
        provider_id=user.get("user_id", user.get("email", "")),
        host_id=body.host_id,
        gpu_model=body.gpu_model,
        gpu_count_total=body.gpu_count_total,
        vram_gb=body.vram_gb,
        ask_cents_per_hour=body.ask_cents_per_hour,
        region=body.region,
        spot_enabled=body.spot_enabled,
        spot_min_cents=body.spot_min_cents,
    )
    return {"ok": True, "offer": offer}


# ── Model: MarketplaceSearchParams ──

class MarketplaceSearchParams(BaseModel):
    gpu_model: str = ""
    min_vram_gb: float = 0
    max_price_cents: int = 0
    region: str = ""
    canada_only: bool = False
    sort_by: str = "price"
    limit: int = 50

@router.post("/api/v2/marketplace/search", tags=["Marketplace v2"])
def api_marketplace_search(body: MarketplaceSearchParams):
    """Search available GPU offers with filters."""
    me = get_marketplace_engine()
    offers = me.search_offers(
        gpu_model=body.gpu_model or None,
        min_vram_gb=body.min_vram_gb or None,
        max_price_cents=body.max_price_cents or None,
        region=body.region or None,
        canada_only=body.canada_only,
        sort_by=body.sort_by,
        limit=body.limit,
    )
    return {"ok": True, "offers": offers, "count": len(offers)}

@router.get("/api/v2/marketplace/spot-prices", tags=["Marketplace v2"])
def api_marketplace_spot_prices():
    """Get current spot prices for all GPU models."""
    me = get_marketplace_engine()
    prices = me.get_current_spot_prices_list()
    return {"ok": True, "spot_prices": prices}

@router.get("/api/v2/marketplace/spot-prices/{gpu_model}/history", tags=["Marketplace v2"])
def api_marketplace_spot_history(gpu_model: str, hours: int = 24):
    """Get spot price history for a GPU model."""
    me = get_marketplace_engine()
    history = me.get_spot_price_history(gpu_model, hours=hours)
    return {"ok": True, "gpu_model": gpu_model, "history": history}

@router.get("/api/v2/marketplace/stats", tags=["Marketplace v2"])
def api_marketplace_stats_v2():
    """Get marketplace aggregate statistics."""
    me = get_marketplace_engine()
    stats = me.get_marketplace_stats()
    return {"ok": True, **stats}


# ── Model: ReservationCreate ──

class ReservationCreate(BaseModel):
    gpu_model: str
    gpu_count: int = 1
    period_months: int = 1

@router.post("/api/v2/marketplace/reservations", tags=["Marketplace v2"])
def api_marketplace_create_reservation(body: ReservationCreate, request: Request):
    """Create a reserved instance commitment."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    try:
        res = me.create_reservation(
            customer_id=user.get("user_id", user.get("email", "")),
            gpu_model=body.gpu_model,
            gpu_count=body.gpu_count,
            period_months=body.period_months,
        )
        return {"ok": True, "reservation": res}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.delete("/api/v2/marketplace/reservations/{reservation_id}", tags=["Marketplace v2"])
def api_marketplace_cancel_reservation(reservation_id: str, request: Request):
    """Cancel a reserved instance commitment early.

    Computes early termination fee: remaining_months * monthly_rate * 50%.
    """
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    result = me.cancel_reservation(
        reservation_id=reservation_id,
        customer_id=user.get("user_id", user.get("email", "")),
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return {"ok": True, **result}


# ── Model: AllocateGPURequest ──

class AllocateGPURequest(BaseModel):
    offer_id: str
    job_id: str
    gpu_count: int = 1
    spot: bool = False

@router.post("/api/v2/marketplace/allocate", tags=["Marketplace v2"])
def api_marketplace_allocate(body: AllocateGPURequest, request: Request):
    """Allocate GPUs from an offer for a job. Atomic — prevents double-sell."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    alloc = me.allocate_gpu(body.offer_id, body.job_id, body.gpu_count, spot=body.spot)
    if not alloc:
        raise HTTPException(409, "Offer not available or insufficient GPUs")
    return {"ok": True, "allocation": alloc}

@router.post("/api/v2/marketplace/release/{allocation_id}", tags=["Marketplace v2"])
def api_marketplace_release(allocation_id: str, request: Request):
    """Release a GPU allocation (job completed/failed)."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    from routes._deps import _require_scope
    _require_scope(user, "marketplace:write")
    me = get_marketplace_engine()
    me.release_allocation(allocation_id)
    return {"ok": True}
