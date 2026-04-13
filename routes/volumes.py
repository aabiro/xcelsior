"""Routes: volumes."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Literal

from routes._deps import (
    _get_current_user,
    broadcast_sse,
    log,
)
from scheduler import (
    log,
)
from volumes import get_volume_engine, VOLUME_PRICE_PER_GB_MONTH_CAD

router = APIRouter()


# ── Model: VolumeCreate ──

class VolumeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    size_gb: int = Field(default=50, ge=1, le=2000)
    region: str = "ca-east"
    encrypted: bool = True

@router.post("/api/v2/volumes", tags=["Volumes"])
def api_volume_create(body: VolumeCreate, request: Request):
    """Create a new persistent volume. Billed in real-time from credits."""
    from routes._deps import _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
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
        broadcast_sse("volume.created", {"volume_id": vol["volume_id"], "name": vol["name"], "size_gb": vol["size_gb"]})
        return {"ok": True, "volume": vol}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/api/v2/volumes", tags=["Volumes"])
def api_volume_list(request: Request):
    """List volumes owned by the current user."""
    from routes._deps import _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    ve = get_volume_engine()
    volumes = ve.list_volumes(user.get("user_id", user.get("email", "")))
    for v in volumes:
        v["price_per_gb_month_cad"] = VOLUME_PRICE_PER_GB_MONTH_CAD
        v["monthly_cost_cad"] = round(v.get("size_gb", 0) * VOLUME_PRICE_PER_GB_MONTH_CAD, 2)
    return {"ok": True, "volumes": volumes}

@router.get("/api/v2/volumes/available", tags=["Volumes"])
def api_volumes_available(request: Request):
    """List volumes available for attachment (status=available) for the current user."""
    from routes._deps import _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    ve = get_volume_engine()
    volumes = ve.list_volumes(user.get("user_id", user.get("email", "")))
    available = [
        {
            "volume_id": v["volume_id"],
            "name": v.get("name", ""),
            "size_gb": v.get("size_gb", 0),
            "region": v.get("region", ""),
            "encrypted": v.get("encrypted", False),
        }
        for v in volumes
        if v.get("status") == "available"
    ]
    return {"ok": True, "volumes": available}

@router.get("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_get(volume_id: str, request: Request):
    """Get volume details."""
    from routes._deps import _require_scope, _get_current_user
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:read")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    owner_id = user.get("user_id", user.get("email", ""))
    if vol.get("owner_id") != owner_id:
        raise HTTPException(404, "Volume not found")
    return {"ok": True, "volume": vol}


# ── Model: VolumeAttachRequest ──

class VolumeAttachRequest(BaseModel):
    instance_id: str
    mount_path: str = Field(default="/workspace", pattern=r"^/(workspace|mnt/[a-zA-Z0-9._-]+|data)$")
    mode: Literal["rw", "ro"] = "rw"

@router.post("/api/v2/volumes/{volume_id}/attach", tags=["Volumes"])
def api_volume_attach(volume_id: str, body: VolumeAttachRequest, request: Request):
    """Attach a volume to a running instance."""
    from routes._deps import _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    owner_id = user.get("user_id", user.get("email", ""))
    ve = get_volume_engine()
    # Ownership check
    vol = ve.get_volume(volume_id)
    if not vol or vol.get("owner_id") != owner_id:
        raise HTTPException(404, "Volume not found")
    try:
        att = ve.attach_volume(volume_id, body.instance_id, body.mount_path, body.mode)
        if not att:
            raise HTTPException(409, "Volume not available for attachment")
        broadcast_sse("volume.attached", {"volume_id": volume_id, "instance_id": body.instance_id})
        return {"ok": True, "attachment": att}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.post("/api/v2/volumes/{volume_id}/detach", tags=["Volumes"])
def api_volume_detach(volume_id: str, request: Request):
    """Detach a volume from its current instance."""
    from routes._deps import _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    vol = ve.get_volume(volume_id)
    if not vol:
        raise HTTPException(404, "Volume not found")
    owner_id = user.get("user_id", user.get("email", ""))
    if vol.get("owner_id") != owner_id:
        raise HTTPException(403, "Not your volume")
    # Detach using atomic FOR UPDATE inside detach_volume
    if vol.get("status") != "attached":
        raise HTTPException(400, "Volume is not attached to any instance")
    try:
        ve.detach_volume(volume_id, instance_id=None)
        broadcast_sse("volume.detached", {"volume_id": volume_id})
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.delete("/api/v2/volumes/{volume_id}", tags=["Volumes"])
def api_volume_delete(volume_id: str, request: Request):
    """Delete a volume. Must not have active attachments."""
    from routes._deps import _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    owner_id = user.get("user_id", user.get("email", ""))
    try:
        result = ve.delete_volume(volume_id, owner_id=owner_id)
        broadcast_sse("volume.deleted", {"volume_id": volume_id})
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(409, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@router.post("/api/v2/volumes/{volume_id}/retry", tags=["Volumes"])
def api_volume_retry_provision(volume_id: str, request: Request):
    """Retry provisioning for a volume stuck in 'error' status."""
    from routes._deps import _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "volumes:write")
    ve = get_volume_engine()
    owner_id = user.get("user_id", user.get("email", ""))
    try:
        result = ve.retry_provision(volume_id, owner_id=owner_id)
        broadcast_sse("volume.retried", {"volume_id": volume_id})
        return {"ok": True, "volume": result}
    except PermissionError:
        raise HTTPException(403, "Not authorised to retry this volume")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))

