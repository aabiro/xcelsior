"""Routes: spot."""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

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
    submit_spot_job,
    update_spot_prices,
)
from routes.instances import _refresh_job

router = APIRouter()


# ── Model: SpotJobIn ──

class SpotJobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(gt=0)
    max_bid: float = Field(gt=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None

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
    """Submit a spot job with a maximum bid price."""
    user = _require_auth(request)
    job = submit_spot_job(j.name, j.vram_needed_gb, j.max_bid, j.priority, tier=j.tier)
    job["submitted_by"] = user.get("email", "")
    job["customer_id"] = user.get("customer_id", user.get("user_id", ""))

    # Auto-process queue
    try:
        process_queue()
        job = _refresh_job(job["job_id"]) or job
    except Exception as e:
        log.debug("process_queue failed: %s", e)

    broadcast_sse(
        "spot_job_submitted",
        {
            "job_id": job["job_id"],
            "name": job["name"],
            "max_bid": j.max_bid,
        },
    )
    return {"ok": True, "instance": job}

@router.post("/spot/preemption-cycle", tags=["Spot Pricing"])
def api_preemption_cycle(request: Request):
    """Run a preemption cycle — reclaim resources from underbidding spot jobs."""
    _require_admin(request)
    preempted = preemption_cycle()
    return {"ok": True, "preempted": preempted}

