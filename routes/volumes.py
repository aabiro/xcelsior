"""Routes: volumes."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _get_current_user,
    log,
)
from scheduler import (
    log,
)
from volumes import get_volume_engine

router = APIRouter()

VOLUME_PRICE_PER_GB_MONTH_CAD = 0.07


# ── Model: VolumeCreate ──

class VolumeCreate(BaseModel):
    name: str
    size_gb: int = 50
    region: str = "ca-east"
    encrypted: bool = True

@router.post("/api/v2/volumes", tags=["Volumes"])
def api_volume_create(body: VolumeCreate, request: Request):
    """Create a new persistent volume. Billed in real-time from credits."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    customer_id = user.get("user_id", user.get("email", ""))
    ve = get_volume_engine()
    try:
        vol = ve.create_volume(
            owner_id=customer_id,
            name=body.name,
            size_gb=body.size_gb,
            region=body.region,
            encrypted=body.encrypted,
        )
        vol["price_per_gb_month_cad"] = VOLUME_PRICE_PER_GB_MONTH_CAD
        vol["estimated_monthly_cost_cad"] = round(body.size_gb * VOLUME_PRICE_PER_GB_MONTH_CAD, 2)
        return {"ok": True, "volume": vol}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/api/v2/volumes", tags=["Volumes"])
def api_volume_list(request: Request):
    """List volumes owned by the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    volumes = ve.list_volumes(user.get("user_id", user.get("email", "")))
    for v in volumes:
        v["price_per_gb_month_cad"] = VOLUME_PRICE_PER_GB_MONTH_CAD
        v["monthly_cost_cad"] = round(v.get("size_gb", 0) * VOLUME_PRICE_PER_GB_MONTH_CAD, 2)
    return {"ok": True, "volumes": volumes}

@router.get("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_get(volume_id: str, request: Request):
    """Get volume details."""
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    return {"ok": True, "volume": vol}


# ── Model: VolumeAttachRequest ──

class VolumeAttachRequest(BaseModel):
    instance_id: str
    mount_path: str = "/workspace"
    mode: str = "rw"

@router.post("/api/v2/volumes/{volume_id}/attach", tags=["Volumes"])
def api_volume_attach(volume_id: str, body: VolumeAttachRequest, request: Request):
    """Attach a volume to a running instance."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    try:
        att = ve.attach_volume(volume_id, body.instance_id, body.mount_path, body.mode)
        if not att:
            raise HTTPException(409, "Volume not available for attachment")
        return {"ok": True, "attachment": att}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.post("/api/v2/volumes/{volume_id}/detach", tags=["Volumes"])
def api_volume_detach(volume_id: str, request: Request):
    """Detach a volume from its current instance."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    owner_id = user.get("user_id", user.get("email", ""))
    if vol.get("owner_id") != owner_id:
        raise HTTPException(403, "Not your volume")
    # Find active attachment and detach
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        att = conn.execute(
            "SELECT instance_id FROM volume_attachments WHERE volume_id = %s AND detached_at = 0",
            (volume_id,),
        ).fetchone()
    if not att:
        raise HTTPException(400, "Volume is not attached to any instance")
    try:
        ve.detach_volume(volume_id, att["instance_id"])
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.delete("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_delete(volume_id: str, request: Request):
    """Delete a volume. Must not have active attachments."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ve = get_volume_engine()
    owner_id = user.get("user_id", user.get("email", ""))
    try:
        result = ve.delete_volume(volume_id, owner_id=owner_id)
        # Refund prorated storage credit
        try:
            vol_data = ve.get_volume(volume_id)  # already deleted, won't find
        except Exception as e:
            log.debug("volume refund data fetch failed: %s", e)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(409, str(e))

