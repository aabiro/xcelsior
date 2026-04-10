"""Routes: artifacts."""

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    log,
)
from artifacts import get_artifact_manager

router = APIRouter()


# ── Model: UploadRequest ──

class UploadRequest(BaseModel):
    job_id: str
    filename: str
    artifact_type: str = "job_output"
    residency_policy: str = "canada_only"

@router.post("/api/artifacts/upload", tags=["Artifacts"])
def api_request_upload(req: UploadRequest, request: Request):
    """Get a presigned upload URL for an artifact."""
    from artifacts import ArtifactType, ResidencyPolicy
    from routes._deps import _get_current_user, _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:write")
    try:
        atype = ArtifactType(req.artifact_type)
        rpolicy = ResidencyPolicy(req.residency_policy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    mgr = get_artifact_manager()
    result = mgr.request_upload(req.job_id, req.filename, atype, rpolicy)
    return {"ok": True, **result}


# ── Model: DownloadRequest ──

class DownloadRequest(BaseModel):
    job_id: str
    filename: str
    artifact_type: str = "job_output"

@router.post("/api/artifacts/download", tags=["Artifacts"])
def api_request_download(req: DownloadRequest, request: Request):
    """Get a presigned download URL for an artifact."""
    from artifacts import ArtifactType
    from routes._deps import _get_current_user, _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:write")
    try:
        atype = ArtifactType(req.artifact_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    mgr = get_artifact_manager()
    result = mgr.request_download(req.job_id, req.filename, atype)
    return {"ok": True, **result}

@router.get("/api/artifacts/{job_id}", tags=["Artifacts"])
def api_list_artifacts(job_id: str, request: Request):
    """List all artifacts for a job."""
    from routes._deps import _get_current_user, _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:read")
    mgr = get_artifact_manager()
    artifacts = mgr.get_job_artifacts(job_id)
    return {"ok": True, "job_id": job_id, "artifacts": artifacts}

@router.get("/api/artifacts/{job_id}/expiry", tags=["Artifacts"])
def api_artifact_expiry(job_id: str, request: Request):
    """Get expiry/cleanup dates for artifacts of a given job.

    Returns each artifact with its created_at and estimated expiry date
    based on the configured retention policy.
    """
    from routes._deps import _get_current_user, _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:read")
    try:
        am = get_artifact_manager()
        arts = am.get_job_artifacts(job_id)
    except Exception as e:
        arts = []

    # Default retention: 90 days for job_output, 180 for model_checkpoint, 30 for logs
    retention_days = {
        "job_output": 90,
        "model_checkpoint": 180,
        "dataset": 365,
        "log_bundle": 30,
    }
    result = []
    for a in arts:
        art_type = a.get("artifact_type", "job_output")
        created = a.get("created_at", time.time())
        ttl_days = retention_days.get(art_type, 90)
        expiry = created + ttl_days * 86400
        result.append(
            {
                "artifact_id": a.get("artifact_id", ""),
                "artifact_type": art_type,
                "created_at": created,
                "ttl_days": ttl_days,
                "expires_at": expiry,
                "days_remaining": max(0, int((expiry - time.time()) / 86400)),
            }
        )

    return {"ok": True, "job_id": job_id, "artifacts": result}

@router.get("/api/artifacts", tags=["Artifacts"])
def api_list_all_artifacts(request: Request):
    """List all artifacts (no job filter)."""
    from routes._deps import _get_current_user, _require_scope
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:read")
    mgr = get_artifact_manager()
    try:
        artifacts = []
        from artifacts import ArtifactType as AT
        for atype in AT:
            artifacts.extend(mgr.primary.list_objects(f"{atype.value}/"))
        return {"ok": True, "artifacts": artifacts}
    except Exception as e:
        return {"ok": True, "artifacts": []}

