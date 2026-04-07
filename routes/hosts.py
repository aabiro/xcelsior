"""Routes: hosts."""

import os
import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from routes._deps import (
    _require_admin,
    broadcast_sse,
    log,
)
from scheduler import (
    check_hosts,
    get_compute_score,
    list_hosts,
    list_jobs,
    log,
    register_host,
    remove_host,
    set_host_draining,
)
from db import UserStore
from verification import get_verification_engine
from security import admit_node
from reputation import VerificationType, get_reputation_engine

router = APIRouter()


def _interactive_host_jobs(host_id: str) -> list[dict]:
    """Interactive jobs that should block worker maintenance."""
    blocking_statuses = {"assigned", "leased", "running"}
    jobs = []
    for job in list_jobs():
        if job.get("host_id") != host_id:
            continue
        if not job.get("interactive", False):
            continue
        if job.get("status") not in blocking_statuses:
            continue
        jobs.append(job)
    return jobs


# ── Model: HostIn ──

class HostIn(BaseModel):
    host_id: str
    ip: str
    gpu_model: str
    total_vram_gb: float
    free_vram_gb: float
    cost_per_hour: float = 0.20
    country: str = "CA"  # ISO 3166-1 alpha-2
    province: str = ""  # CA province code (ON, QC, BC, etc.)
    # Optional: agent-reported versions for inline admission
    versions: dict | None = None  # {"runc": "1.2.4", "nvidia_ctk": "1.17.8", ...}
    # Canadian company fields (Report #1.B — Provider Onboarding)
    corporation_name: str = ""  # Legal corporation name
    business_number: str = ""  # CRA Business Number (BN), e.g. 123456789RC0001
    gst_hst_number: str = ""  # GST/HST registration number
    legal_name: str = ""  # Legal name of individual or company


# ── Model: JobIn ──

class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None
    num_gpus: int = Field(default=1, ge=1, le=64)
    host_id: str | None = None  # Direct host assignment (marketplace)
    gpu_model: str | None = None  # Hint for VRAM lookup
    nfs_server: str | None = None
    nfs_path: str | None = None
    nfs_mount_point: str | None = None
    image: str | None = None
    interactive: bool = True
    command: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)


# ── Model: StatusUpdate ──

class StatusUpdate(BaseModel):
    status: str
    host_id: str | None = None
    container_id: str | None = None
    container_name: str | None = None

@router.put("/host", tags=["Hosts"])
def api_register_host(h: HostIn):
    """Register or update a host with strict admission gating.

    Per REPORT_FEATURE_FINAL.md §62 and REPORT_FEATURE_2.md §37:
    - Hosts must pass node admission (version gating) before accepting work
    - Country/province recorded for jurisdiction-aware scheduling
    - Hosts register as 'pending' until agent completes benchmark + admission
    - If versions are provided inline, admission is checked immediately
    """
    from security import admit_node

    # Register the host with country/province metadata
    entry = register_host(
        h.host_id, h.ip, h.gpu_model, h.total_vram_gb, h.free_vram_gb, h.cost_per_hour
    )
    entry["country"] = h.country.upper()
    entry["province"] = h.province.upper() if h.province else ""
    # Persist Canadian company info if provided
    if h.corporation_name:
        entry["corporation_name"] = h.corporation_name
    if h.business_number:
        entry["business_number"] = h.business_number
    if h.gst_hst_number:
        entry["gst_hst_number"] = h.gst_hst_number
    if h.legal_name:
        entry["legal_name"] = h.legal_name

    # Inline admission check if versions provided
    if h.versions:
        admitted, details = admit_node(h.host_id, h.versions, h.gpu_model)
        entry["admitted"] = admitted
        entry["admission_details"] = details
        entry["recommended_runtime"] = details.get("recommended_runtime", "runc")
        if not admitted:
            # Host is registered but marked as not-admitted — won't receive work
            entry["status"] = "pending"
            log.warning(
                "HOST %s registered but NOT ADMITTED: %s",
                h.host_id,
                details.get("rejection_reasons", []),
            )
    else:
        # No versions provided — only set pending for NEW hosts.
        # Existing hosts preserve their admission status from /agent/versions.
        if not entry.get("admitted"):
            entry.setdefault("admitted", False)
            entry["status"] = "pending"

    # Persist the updated entry (country, province, admitted status)
    from scheduler import _atomic_mutation, _upsert_host_row, _migrate_hosts_if_needed

    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        _upsert_host_row(conn, entry)

    # Auto-compute score and auto-list on marketplace
    from scheduler import estimate_compute_score, register_compute_score, list_rig

    score = estimate_compute_score(h.gpu_model)
    register_compute_score(h.host_id, h.gpu_model, score)
    entry["compute_score"] = score

    list_rig(
        h.host_id,
        h.gpu_model,
        h.total_vram_gb,
        h.cost_per_hour,
        description=f"{h.gpu_model} ({h.total_vram_gb}GB) in {h.country.upper()}",
        owner=h.host_id,
    )

    # ── Auto-create verification + reputation records ────────────────────
    # Ensures the host appears on the trust page immediately (as "unverified")
    # and gets a baseline reputation score.  Hardware verification scoring
    # happens later when the agent sends a full benchmark report.
    try:
        ve = get_verification_engine()
        if not ve.store.get_verification(h.host_id):
            from verification import HostVerification, HostVerificationState

            ve.store.save_verification(
                HostVerification(
                    verification_id=str(uuid.uuid4())[:12],
                    host_id=h.host_id,
                    state=HostVerificationState.UNVERIFIED,
                )
            )
            log.info("VERIFY RECORD created for new host %s", h.host_id)
        # Bootstrap reputation — email verification is implicit for registered users
        re = get_reputation_engine()
        re.add_verification(h.host_id, VerificationType.EMAIL)
    except Exception as e:
        log.exception("Non-fatal: could not bootstrap verification/reputation for %s", h.host_id)

    broadcast_sse(
        "host_update",
        {
            "host_id": h.host_id,
            "gpu_model": h.gpu_model,
            "admitted": entry.get("admitted", False),
            "country": entry.get("country", ""),
        },
    )
    return {"ok": True, "host": entry}

@router.get("/host/{host_id}", tags=["Hosts"])
def api_get_host(host_id: str):
    """Get a single host by ID."""
    hosts = list_hosts(active_only=False)
    host = next((h for h in hosts if h["host_id"] == host_id), None)
    if not host:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    return {"ok": True, "host": host}

@router.get("/hosts", tags=["Hosts"])
def api_list_hosts(active_only: bool = True):
    """List all hosts."""
    return {"hosts": list_hosts(active_only=active_only)}


@router.get("/host/{host_id}/maintenance", tags=["Hosts"])
def api_host_maintenance(host_id: str, request: Request):
    """Return maintenance readiness for a host."""
    _require_admin(request)
    hosts = list_hosts(active_only=False)
    host = next((h for h in hosts if h["host_id"] == host_id), None)
    if not host:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")

    interactive_jobs = _interactive_host_jobs(host_id)
    summary = [
        {
            "job_id": job.get("job_id"),
            "name": job.get("name"),
            "status": job.get("status"),
            "owner": job.get("owner"),
        }
        for job in interactive_jobs
    ]
    safe_to_maintain = host.get("status") == "draining" and len(summary) == 0

    return {
        "ok": True,
        "host_id": host_id,
        "status": host.get("status"),
        "draining": host.get("status") == "draining",
        "admitted": host.get("admitted", False),
        "active_interactive_instances": len(summary),
        "interactive_instances": summary,
        "safe_to_maintain": safe_to_maintain,
    }


@router.post("/host/{host_id}/drain", tags=["Hosts"])
def api_drain_host(host_id: str, request: Request):
    """Stop new placements on a host without evicting active instances."""
    _require_admin(request)
    hosts = list_hosts(active_only=False)
    host = next((h for h in hosts if h["host_id"] == host_id), None)
    if not host:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    if host.get("status") == "dead":
        raise HTTPException(status_code=409, detail="Cannot drain a dead host")

    updated = set_host_draining(host_id, draining=True)
    broadcast_sse("host_update", {"host_id": host_id, "status": "draining"})
    return {
        "ok": True,
        "host": updated,
        "maintenance": api_host_maintenance(host_id, request),
    }


@router.post("/host/{host_id}/undrain", tags=["Hosts"])
def api_undrain_host(host_id: str, request: Request):
    """Restore a drained host to active or pending status."""
    _require_admin(request)
    hosts = list_hosts(active_only=False)
    host = next((h for h in hosts if h["host_id"] == host_id), None)
    if not host:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    if host.get("status") == "dead":
        raise HTTPException(status_code=409, detail="Cannot undrain a dead host")

    updated = set_host_draining(host_id, draining=False)
    broadcast_sse("host_update", {"host_id": host_id, "status": updated.get("status", "pending")})
    return {"ok": True, "host": updated}

@router.delete("/host/{host_id}", tags=["Hosts"])
def api_remove_host(host_id: str):
    """Remove a host."""
    hosts = list_hosts(active_only=False)
    if not any(h["host_id"] == host_id for h in hosts):
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    remove_host(host_id)
    broadcast_sse("host_removed", {"host_id": host_id})
    return {"ok": True, "removed": host_id}

@router.post("/hosts/check", tags=["Hosts"])
def api_check_hosts():
    """Ping all hosts and update status."""
    results = check_hosts()
    return {"results": results}

@router.get("/compute-score/{host_id}", tags=["Hosts"])
def api_get_compute_score(host_id: str):
    """Get the compute score (XCU) for a host."""
    score = get_compute_score(host_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"No compute score for host {host_id}")
    return {"ok": True, "host_id": host_id, "score": score}

@router.get("/compute-scores", tags=["Hosts"])
def api_list_compute_scores():
    """List compute scores for all hosts."""
    hosts = list_hosts(active_only=False)
    scores = {}
    for h in hosts:
        score = get_compute_score(h["host_id"])
        if score is not None:
            scores[h["host_id"]] = {
                "score": score,
                "gpu_model": h.get("gpu_model", "unknown"),
            }
    return {"ok": True, "scores": scores}
