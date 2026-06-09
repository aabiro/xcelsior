"""Routes: spot pricing (admin + public price feeds)."""

from fastapi import APIRouter, Request

from routes._deps import (
    _require_admin,
    broadcast_sse,
)
from spot_pricing import (
    effective_spot_rate_cad,
    get_current_spot_prices,
    get_current_spot_prices_list,
    suggested_spot_min_cents,
    update_all_spot_prices,
)
from scheduler import preemption_cycle
from spot.feature import spot_feature_status

router = APIRouter()


@router.get("/spot-prices", tags=["Spot Pricing"])
def api_spot_prices():
    """Get current unified spot prices for all GPU models."""
    prices_cad = get_current_spot_prices()
    spot_prices = get_current_spot_prices_list()
    return {"ok": True, "prices": prices_cad, "spot_prices": spot_prices}


@router.post("/spot-prices/update", tags=["Spot Pricing"])
def api_update_spot_prices(request: Request):
    """Trigger spot price recalculation."""
    _require_admin(request)
    prices = update_all_spot_prices()
    broadcast_sse("spot_prices_updated", {"prices": prices})
    return {"ok": True, "prices": prices}


@router.get("/api/pricing/spot-floor-suggestion", tags=["Spot Pricing"])
def api_spot_floor_suggestion(gpu_model: str, spot_min_cents: int | None = None):
    """Suggested provider spot floor and effective rate preview for a GPU model."""
    floor = spot_min_cents if spot_min_cents is not None else suggested_spot_min_cents(gpu_model)
    preview = effective_spot_rate_cad(gpu_model, floor)
    return {"ok": True, **preview}


@router.get("/api/pricing/spot-enabled", tags=["Spot Pricing"])
def api_spot_enabled():
    """Spot instance feature flag — global kill switch status for UI."""
    return {"ok": True, **spot_feature_status()}


@router.post("/spot/preemption-cycle", tags=["Spot Pricing"])
def api_preemption_cycle(request: Request, dry_run: bool = False):
    """Capacity preemption cycle for ops (set ``dry_run=true`` to preview only)."""
    _require_admin(request)
    prices, result = preemption_cycle(dry_run=dry_run)
    if dry_run:
        return {"ok": True, "dry_run": True, "candidates": result, "prices": prices}
    return {"ok": True, "preempted": result, "prices": prices}