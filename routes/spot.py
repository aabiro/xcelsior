"""Routes: spot pricing (admin + public price feeds)."""

from fastapi import APIRouter, Request

from routes._deps import (
    _require_admin,
    broadcast_sse,
)
from scheduler import (
    get_current_spot_prices,
    preemption_cycle,
    update_spot_prices,
)

router = APIRouter()


@router.get("/spot-prices", tags=["Spot Pricing"])
def api_spot_prices():
    """Get current spot prices for all GPU models."""
    return {"ok": True, "prices": get_current_spot_prices()}


@router.post("/spot-prices/update", tags=["Spot Pricing"])
def api_update_spot_prices(request: Request):
    """Trigger spot price recalculation."""
    _require_admin(request)
    prices = update_spot_prices()
    broadcast_sse("spot_prices_updated", {"prices": prices})
    return {"ok": True, "prices": prices}


@router.post("/spot/preemption-cycle", tags=["Spot Pricing"])
def api_preemption_cycle(request: Request):
    """Run a preemption cycle (capacity-based eviction in Phase 7)."""
    _require_admin(request)
    preempted = preemption_cycle()
    return {"ok": True, "preempted": preempted}