"""Routes: autoscale."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from routes._deps import (
    _require_admin,
)
from scheduler import (
    add_to_pool,
    autoscale_cycle,
    autoscale_down,
    autoscale_up,
    load_autoscale_pool,
    remove_from_pool,
)

router = APIRouter()


# ── Model: PoolHost ──

class PoolHost(BaseModel):
    host_id: str
    ip: str
    gpu_model: str
    vram_gb: float
    cost_per_hour: float = 0.20
    country: str = "CA"

@router.post("/autoscale/pool", tags=["Autoscale"])
def api_add_to_pool(h: PoolHost, request: Request):
    """Add a host to the autoscale pool."""
    from routes._deps import _require_scope
    _require_scope(_require_admin(request), "autoscale:write")
    entry = add_to_pool(h.host_id, h.ip, h.gpu_model, h.vram_gb, h.cost_per_hour, h.country)
    return {"ok": True, "pool_entry": entry}

@router.delete("/autoscale/pool/{host_id}", tags=["Autoscale"])
def api_remove_from_pool(host_id: str, request: Request):
    """Remove a host from the autoscale pool."""
    from routes._deps import _require_scope
    _require_scope(_require_admin(request), "autoscale:write")
    remove_from_pool(host_id)
    return {"ok": True, "removed": host_id}

@router.get("/autoscale/pool", tags=["Autoscale"])
def api_get_pool(request: Request):
    """List the autoscale pool."""
    from routes._deps import _require_scope
    _require_scope(_require_admin(request), "autoscale:read")
    return {"pool": load_autoscale_pool()}

@router.post("/autoscale/cycle", tags=["Autoscale"])
def api_autoscale_cycle(request: Request):
    """Run a full autoscale cycle: scale up, process queue, scale down."""
    from routes._deps import _require_scope
    _require_scope(_require_admin(request), "autoscale:write")
    provisioned, assigned, deprovisioned = autoscale_cycle()
    return {
        "provisioned": [{"host_id": h["host_id"], "gpu": h["gpu_model"]} for h in provisioned],
        "assigned": [
            {"job": j["name"], "job_id": j["job_id"], "host": h["host_id"]} for j, h in assigned
        ],
        "deprovisioned": deprovisioned,
    }

@router.post("/autoscale/up", tags=["Autoscale"])
def api_autoscale_up(request: Request):
    """Scale up: provision hosts for queued jobs."""
    from routes._deps import _require_scope
    _require_scope(_require_admin(request), "autoscale:write")
    provisioned = autoscale_up()
    return {"provisioned": [{"host_id": h["host_id"], "gpu": h["gpu_model"]} for h in provisioned]}

@router.post("/autoscale/down", tags=["Autoscale"])
def api_autoscale_down(request: Request):
    """Scale down: deprovision idle autoscaled hosts."""
    from routes._deps import _require_scope
    _require_scope(_require_admin(request), "autoscale:write")
    deprovisioned = autoscale_down()
    return {"deprovisioned": deprovisioned}

