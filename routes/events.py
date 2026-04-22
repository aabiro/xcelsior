"""Routes: events."""

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    _require_admin,
)
from events import get_event_store, get_state_machine

router = APIRouter()


@router.get("/api/events/{entity_type}/{entity_id}", tags=["Events"])
def api_get_events(entity_type: str, entity_id: str, request: Request, limit: int = 50):
    """Get event history for a job or host."""
    from routes._deps import _get_current_user, _require_scope

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "events:read")
    store = get_event_store()
    events = store.get_events(entity_type, entity_id, limit=limit)
    return {"ok": True, "entity_type": entity_type, "entity_id": entity_id, "events": events}


@router.get("/api/events/leases/{job_id}", tags=["Events"])
def api_get_lease(job_id: str, request: Request):
    """Get active lease for a job."""
    from routes._deps import _get_current_user, _require_scope

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "events:read")
    store = get_event_store()
    lease = store.get_lease(job_id)
    if not lease:
        raise HTTPException(status_code=404, detail=f"No active lease for job {job_id}")
    return {"ok": True, "lease": lease}


@router.get("/api/audit/verify-chain", tags=["Events"])
def api_verify_event_chain(request: Request):
    """Verify the tamper-evident hash chain on all events.

    Returns chain integrity status. If any event was modified after
    being written, the chain will report the break point.
    """
    _require_admin(request)
    store = get_event_store()
    result = store.verify_chain()
    return {"ok": True, "chain_integrity": result}


@router.get("/api/audit/instance/{job_id}", tags=["Events"])
def api_instance_audit_trail(job_id: str):
    """Full auditable trail for a job — every event with hash chain.

    This is the dispute-resolution artifact: every state change,
    lease renewal, billing event, ordered by time with tamper-evident hashes.
    """
    sm = get_state_machine()
    timeline = sm.get_job_timeline(job_id)
    if not timeline:
        raise HTTPException(404, f"No events for job {job_id}")
    return {"ok": True, "job_id": job_id, "events": timeline, "count": len(timeline)}


@router.get("/api/events", tags=["Events"])
def api_get_all_events(limit: int = 100):
    """Get recent events across all entities."""
    store = get_event_store()
    events = store.get_events(limit=limit)
    return {"ok": True, "events": [e if isinstance(e, dict) else e.__dict__ for e in events]}
