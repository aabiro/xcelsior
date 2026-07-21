"""Routes: artifacts."""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    log,
)
from artifacts import get_artifact_manager, ArtifactType, ResidencyPolicy

router = APIRouter()


# ── Model: UploadRequest ──


class UploadRequest(BaseModel):
    job_id: str
    filename: str
    artifact_type: str = "job_output"
    residency_policy: str = "canada_only"


def _user_upload_slot(user: dict) -> str:
    """Key segment for standalone (non-job) uploads, scoped per user."""
    return f"user-{user.get('user_id', 'unknown')}"


def _resolve_artifact_job_id(user: dict, job_id: str) -> str:
    """Map an empty job_id to the caller's personal upload slot.

    Standalone uploads from the dashboard have no job; without this they
    failed job-access checks with a 404.
    """
    from routes.instances import _check_job_access

    job_id = (job_id or "").strip()
    if not job_id:
        return _user_upload_slot(user)
    if job_id.startswith("user-"):
        if job_id != _user_upload_slot(user):
            raise HTTPException(403, "Not authorized to access these artifacts")
        return job_id
    _check_job_access(user, job_id)
    return job_id


def _entry_from_object(obj: dict) -> dict:
    """Shape a raw storage listing into the ArtifactEntry the dashboard renders."""
    key = obj.get("key", "")
    parts = key.split("/")
    atype = parts[0] if parts else ""
    job_id = parts[1] if len(parts) >= 3 else ""
    return {
        "artifact_id": key,
        "key": key,
        "job_id": "" if job_id.startswith("user-") else job_id,
        "filename": parts[-1] if parts else key,
        "artifact_type": atype,
        "size_bytes": obj.get("size_bytes"),
        "created_at": obj.get("last_modified"),
    }


@router.post("/api/artifacts/upload", tags=["Artifacts"])
def api_request_upload(req: UploadRequest, request: Request):
    """Get a presigned upload URL for an artifact."""
    from routes._deps import _get_current_user, _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:write")

    job_id = _resolve_artifact_job_id(user, req.job_id)
    try:
        atype = ArtifactType(req.artifact_type)
        rpolicy = ResidencyPolicy(req.residency_policy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    mgr = get_artifact_manager()
    result = mgr.request_upload(
        atype,
        job_id,
        req.filename,
        residency=rpolicy,
        owner_user_id=user.get("user_id"),
    )
    return {"ok": True, **result}


# ── Model: FinalizeRequest ──


class FinalizeRequest(BaseModel):
    upload_session_id: str
    checksum: Optional[str] = None


@router.post("/api/artifacts/finalize", tags=["Artifacts"])
def api_finalize_upload(req: FinalizeRequest, request: Request):
    """Finalize an upload session, performing validation and transitioning status."""
    from routes._deps import _get_current_user, _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:write")

    mgr = get_artifact_manager()
    try:
        result = mgr.finalize_upload(req.upload_session_id, req.checksum)
        return {"ok": True, **result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.error("finalize_upload failed: %s", e)
        raise HTTPException(500, f"Failed to finalize upload: {e}")


# ── Model: DownloadRequest ──


class DownloadRequest(BaseModel):
    job_id: Optional[str] = None
    filename: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_type: str = "job_output"


@router.post("/api/artifacts/download", tags=["Artifacts"])
def api_request_download(req: DownloadRequest, request: Request):
    """Get a presigned download URL for an artifact."""
    from routes._deps import _get_current_user, _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:write")

    mgr = get_artifact_manager()

    if req.artifact_id:
        try:
            result = mgr.request_download_by_id(req.artifact_id)
            return {"ok": True, **result}
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to get download URL: {e}")

    if not req.job_id or not req.filename:
        raise HTTPException(400, "Must specify artifact_id, or job_id and filename")

    job_id = _resolve_artifact_job_id(user, req.job_id)
    try:
        atype = ArtifactType(req.artifact_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = mgr.download_url_for(atype, job_id, req.filename)
    return {"ok": True, **result}


@router.get("/api/artifacts", tags=["Artifacts"])
def api_list_all_artifacts(request: Request):
    """List artifacts visible to the caller: their jobs' outputs plus standalone uploads."""
    from routes._deps import (
        _get_current_user,
        _is_platform_admin,
        _require_scope,
        _user_team_id,
        _filter_jobs_for_user,
    )
    from artifacts import StorageUnavailable
    from control_plane.db import control_plane_transaction

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:read")

    mgr = get_artifact_manager()

    try:
        if not mgr._is_db_active():
            # Fallback for non-postgres or non-active DB environments
            objects: list[dict] = []
            if _is_platform_admin(user):
                for atype in ArtifactType:
                    objects.extend(mgr.primary.list_objects(f"{atype.value}/"))
            else:
                from scheduler import list_jobs
                jobs = _filter_jobs_for_user(list_jobs(), user)
                slots = {j.get("job_id", "") for j in jobs}
                slots.discard("")
                slots.add(_user_upload_slot(user))
                for slot in slots:
                    objects.extend(mgr.get_job_artifacts(slot))
            return {"ok": True, "artifacts": [_entry_from_object(o) for o in objects]}

        # Query storage.artifacts table directly for sub-millisecond, low-egress listing
        with control_plane_transaction() as conn:
            if _is_platform_admin(user):
                rows = conn.execute(
                    """SELECT artifact_id, object_key, logical_name, job_id, tenant_id, artifact_type, size_bytes, created_at
                       FROM storage.artifacts
                       WHERE state = 'available'
                       ORDER BY created_at DESC"""
                ).fetchall()
            else:
                active_team = _user_team_id(user) or _user_upload_slot(user)
                rows = conn.execute(
                    """SELECT artifact_id, object_key, logical_name, job_id, tenant_id, artifact_type, size_bytes, created_at
                       FROM storage.artifacts
                       WHERE tenant_id = %s AND state = 'available'
                       ORDER BY created_at DESC""",
                    (active_team,),
                ).fetchall()

        artifacts = []
        for r in rows:
            artifacts.append({
                "artifact_id": str(r[0]),
                "key": r[1],
                "filename": r[2],
                "job_id": r[3] or "",
                "tenant_id": r[4],
                "artifact_type": r[5],
                "size_bytes": r[6],
                "created_at": r[7].timestamp() if r[7] else time.time(),
            })

        return {"ok": True, "artifacts": artifacts}

    except StorageUnavailable as e:
        log.error("artifacts.list_all storage unavailable user=%s: %s",
                  user.get("user_id"), e)
        raise HTTPException(
            status_code=503,
            detail={"error": {"code": "storage_unavailable",
                              "message": "Artifact storage is unavailable"}},
        )
    except Exception as e:
        log.error("artifacts.list_all failed user=%s: %s", user.get("user_id"), e)
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "artifact_list_failed",
                              "message": "Failed to list artifacts"}},
        )


@router.get("/api/artifacts/{job_id}", tags=["Artifacts"])
def api_list_artifacts(job_id: str, request: Request):
    """List all artifacts for a job."""
    from routes._deps import _get_current_user, _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "artifacts:read")
    from routes.instances import _check_job_access

    _check_job_access(user, job_id)
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
    from routes.instances import _check_job_access

    _check_job_access(user, job_id)
    try:
        am = get_artifact_manager()
        arts = am.get_job_artifacts(job_id)
    except Exception:
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
        created = a.get("last_modified", time.time())
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
