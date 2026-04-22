"""Routes: cloudburst."""

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    _get_current_user,
)
from cloudburst import get_burst_engine

router = APIRouter()


@router.get("/api/v2/burst/status", tags=["Cloud Burst"])
def api_burst_status(request: Request):
    """Get cloud burst auto-scaling status."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    cbe = get_burst_engine()
    status = cbe.get_burst_status()
    return {"ok": True, **status}
