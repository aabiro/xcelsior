"""Routes: jurisdiction."""

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from routes._deps import (
    _require_admin,
    broadcast_sse,
)
from scheduler import (
    list_hosts,
    list_hosts_filtered,
    list_jobs,
    process_queue_filtered,
    process_queue_sovereign,
    set_canada_only,
)
from jurisdiction import JurisdictionConstraint, TrustTier, generate_residency_trace

router = APIRouter()


# ── Model: CanadaToggle ──

class CanadaToggle(BaseModel):
    enabled: bool

@router.get("/canada", tags=["Jurisdiction"])
def api_canada_status(request: Request):
    """Check if Canada-only mode is active."""
    _require_admin(request)
    import scheduler

    return {"canada_only": scheduler.CANADA_ONLY}

@router.put("/canada", tags=["Jurisdiction"])
def api_set_canada(toggle: CanadaToggle, request: Request):
    """Toggle Canada-only mode."""
    _require_admin(request)
    set_canada_only(toggle.enabled)
    return {"ok": True, "canada_only": toggle.enabled}

@router.get("/hosts/ca", tags=["Jurisdiction"])
def api_list_canadian_hosts():
    """List only Canadian hosts."""
    return {"hosts": list_hosts_filtered(canada_only=True)}

@router.post("/queue/process/ca", tags=["Jurisdiction"])
def api_process_queue_ca():
    """Process queue with Canada-only hosts."""
    assigned = process_queue_filtered(canada_only=True)
    return {
        "canada_only": True,
        "assigned": [
            {
                "job": j["name"],
                "job_id": j["job_id"],
                "host": h["host_id"],
                "country": h.get("country", "?"),
            }
            for j, h in assigned
        ],
    }


# ── Model: JurisdictionFilterRequest ──

class JurisdictionFilterRequest(BaseModel):
    canada_only: bool = True
    province: str = None
    trust_tier: str = None

@router.post("/api/jurisdiction/hosts", tags=["Jurisdiction"])
def api_jurisdiction_hosts(req: JurisdictionFilterRequest):
    """Filter hosts by jurisdiction constraints."""
    hosts = list_hosts(active_only=True)
    constraint = JurisdictionConstraint(
        canada_only=req.canada_only,
        province=req.province,
        trust_tier=TrustTier(req.trust_tier) if req.trust_tier else None,
    )
    from jurisdiction import filter_hosts_by_jurisdiction

    filtered = filter_hosts_by_jurisdiction(hosts, constraint)
    return {"ok": True, "count": len(filtered), "hosts": filtered}

@router.get("/api/jurisdiction/residency-trace/{job_id}", tags=["Jurisdiction"])
def api_residency_trace(job_id: str):
    """Generate a residency trace for a job (compliance artifact)."""
    jobs = list_jobs()
    job = next((j for j in jobs if j["job_id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    host_data = {}
    if job.get("host_id"):
        hosts = list_hosts(active_only=False)
        host_data = next((h for h in hosts if h["host_id"] == job["host_id"]), {})

    # Build jurisdiction object from host data
    from jurisdiction import HostJurisdiction

    jurisdiction = None
    if host_data:
        jurisdiction = HostJurisdiction(
            country=host_data.get("country", "CA"),
            province=host_data.get("province", ""),
            city=host_data.get("city", ""),
        )
    trace = generate_residency_trace(
        job_id=job_id,
        host_id=job.get("host_id", ""),
        jurisdiction=jurisdiction,
        started_at=job.get("started_at", job.get("submitted_at", 0)) or 0,
        completed_at=job.get("completed_at") or time.time(),
    )
    return {"ok": True, "job_id": job_id, "trace": trace}

@router.get("/api/trust-tiers", tags=["Jurisdiction"])
def api_trust_tiers():
    """List available trust tiers and their requirements."""
    from jurisdiction import TRUST_TIER_REQUIREMENTS

    min_scores = {"community": 0, "residency": 25, "sovereignty": 50, "regulated": 75}
    req_labels = {
        "requires_canada": "Host physically located in Canada",
        "requires_verified": "Host identity verified",
        "requires_sovereignty_vetting": "Canadian-incorporated operator vetted",
        "requires_audit_trail": "Full audit trail enabled",
    }
    tiers = {}
    for t, v in TRUST_TIER_REQUIREMENTS.items():
        reqs = [label for key, label in req_labels.items() if v.get(key)]
        tiers[t.value] = {**v, "min_score": min_scores.get(t.value, 0), "requirements": reqs}
    return {"ok": True, "tiers": tiers}


# ── Model: SovereignQueueRequest ──

class SovereignQueueRequest(BaseModel):
    canada_only: bool = True
    province: str = None
    trust_tier: str = None

@router.post("/api/queue/process-sovereign", tags=["Jurisdiction"])
def api_process_queue_sovereign(req: SovereignQueueRequest):
    """Process queue with jurisdiction + verification + reputation awareness."""
    assigned = process_queue_sovereign(
        canada_only=req.canada_only,
        province=req.province,
        trust_tier=req.trust_tier,
    )
    results = []
    for job, host in assigned:
        results.append(
            {
                "job_id": job["job_id"],
                "job_name": job.get("name"),
                "host_id": host["host_id"],
                "gpu_model": host.get("gpu_model"),
                "country": host.get("country", ""),
            }
        )
        broadcast_sse(
            "job_assigned",
            {
                "job_id": job["job_id"],
                "host_id": host["host_id"],
            },
        )
    return {"ok": True, "assigned": len(results), "jobs": results}

