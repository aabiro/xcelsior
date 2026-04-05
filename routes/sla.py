"""Routes: sla."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from routes._deps import (
    _require_admin,
)
from scheduler import (
    list_hosts,
)
from sla import SLA_TARGETS, get_sla_engine

router = APIRouter()


# ── Model: SLAEnforceRequest ──

class SLAEnforceRequest(BaseModel):
    host_id: str
    month: str  # YYYY-MM
    tier: str = "community"
    monthly_spend_cad: float = 0.0

@router.post("/api/sla/enforce", tags=["SLA"])
def api_sla_enforce(req: SLAEnforceRequest, request: Request):
    """Run monthly SLA enforcement for a host.

    Calculates uptime percentage, downtime incidents, and credits owed
    based on the SLA tier. Credits follow the Google Cloud / Azure model:
    - 95–99% uptime → 10% credit
    - 90–95% uptime → 25% credit
    - <90% uptime   → 100% credit
    """
    _require_admin(request)
    engine = get_sla_engine()
    record = engine.enforce_monthly(
        req.host_id,
        req.tier,
        req.month,
        req.monthly_spend_cad,
    )
    return {
        "ok": True,
        "host_id": record.host_id,
        "month": record.month,
        "tier": record.tier,
        "uptime_pct": round(record.uptime_pct, 4),
        "downtime_seconds": record.downtime_seconds,
        "incidents": record.incidents,
        "credit_pct": record.credit_pct,
        "credit_cad": record.credit_cad,
    }

@router.get("/api/sla/hosts-summary", tags=["SLA"])
def api_sla_hosts_summary():
    """Get SLA status summary for all known hosts.

    Returns per-host cards with uptime %, violation count, and SLA tier.
    Used by dashboard UI-8.1 SLA Dashboard.
    """
    try:
        engine = get_sla_engine()
    except Exception as e:
        return {"ok": True, "hosts": [], "count": 0}
    import scheduler as _sched

    hosts = _sched.list_hosts(active_only=False)
    summaries = []
    for h in hosts:
        hid = h.get("host_id", "")
        if not hid:
            continue
        try:
            uptime = engine.get_host_uptime_pct(hid)
            violations = engine.get_violations(hid)
        except Exception as e:
            uptime = 0.0
            violations = []
        tier = h.get("sla_tier", "community")
        summaries.append(
            {
                "host_id": hid,
                "gpu_model": h.get("gpu_model", "Unknown"),
                "status": h.get("status", "unknown"),
                "sla_tier": tier,
                "uptime_30d_pct": round(uptime, 4),
                "violation_count": len(violations),
                "last_violation": violations[-1] if violations else None,
                "country": h.get("country", ""),
                "province": h.get("province", ""),
            }
        )
    return {"ok": True, "hosts": summaries, "count": len(summaries)}

@router.get("/api/sla/{host_id}", tags=["SLA"])
def api_sla_status(host_id: str, month: str = ""):
    """Get SLA record and rolling uptime for a host."""
    engine = get_sla_engine()
    uptime_30d = engine.get_host_uptime_pct(host_id)
    record = None
    if month:
        rec = engine.get_host_sla(host_id, month)
        record = (
            {
                "month": rec.month,
                "tier": rec.tier,
                "uptime_pct": round(rec.uptime_pct, 4),
                "downtime_seconds": rec.downtime_seconds,
                "incidents": rec.incidents,
                "credit_pct": rec.credit_pct,
                "credit_cad": rec.credit_cad,
            }
            if rec
            else None
        )
    return {
        "ok": True,
        "host_id": host_id,
        "uptime_30d_pct": round(uptime_30d, 4),
        "monthly_record": record,
    }

@router.get("/api/sla/violations/{host_id}", tags=["SLA"])
def api_sla_violations(host_id: str, since: float = 0):
    """Get SLA violation history for a host."""
    engine = get_sla_engine()
    violations = engine.get_violations(host_id, since)
    return {"ok": True, "host_id": host_id, "violations": violations, "count": len(violations)}

@router.get("/api/sla/downtimes", tags=["SLA"])
def api_sla_active_downtimes():
    """Get all currently-open downtime periods across all hosts."""
    engine = get_sla_engine()
    downtimes = engine.get_active_downtimes()
    return {"ok": True, "downtimes": downtimes, "count": len(downtimes)}

@router.get("/api/sla/targets", tags=["SLA"])
def api_sla_targets():
    """Get SLA target definitions for all tiers."""
    from dataclasses import asdict

    targets = {t.value: asdict(v) for t, v in SLA_TARGETS.items()}
    return {"ok": True, "targets": targets}

