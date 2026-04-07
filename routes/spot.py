"""Routes: spot."""

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from routes._deps import (
    _require_admin,
    _require_auth,
    broadcast_sse,
    log,
)
from scheduler import (
    get_current_spot_prices,
    log,
    preemption_cycle,
    process_queue,
    update_spot_prices,
)

router = APIRouter()


# ── Model: SpotJobIn ──

class SpotJobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(gt=0)
    max_bid: float = Field(gt=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None
    image: str | None = None

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return v
        from security import validate_docker_image
        return validate_docker_image(v)

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

@router.post("/spot/instance", tags=["Spot Pricing"])
def api_submit_spot_instance(j: SpotJobIn, request: Request):
    """Submit a spot job — delegates to unified POST /instance."""
    from routes.instances import JobIn, api_submit_instance
    unified = JobIn(
        name=j.name,
        vram_needed_gb=j.vram_needed_gb,
        max_bid=j.max_bid,
        priority=j.priority,
        tier=j.tier,
        image=j.image,
    )
    return api_submit_instance(unified, request)

@router.post("/spot/preemption-cycle", tags=["Spot Pricing"])
def api_preemption_cycle(request: Request):
    """Run a preemption cycle — reclaim resources from underbidding spot jobs."""
    _require_admin(request)
    preempted = preemption_cycle()
    return {"ok": True, "preempted": preempted}

