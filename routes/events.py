"""Routes: events."""

import base64
import json
import math
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from routes._deps import (
    _require_admin,
    _require_auth,
    _require_entity_event_access,
    _require_scope,
)
from events import get_event_store, get_state_machine

router = APIRouter()


@router.get("/api/events/leases/{job_id}", tags=["Events"])
def api_get_lease(job_id: str, request: Request):
    """Get active lease for a job."""
    user = _require_auth(request)
    _require_scope(user, "events:read")
    from routes.instances import _check_job_access

    _check_job_access(user, job_id)
    store = get_event_store()
    lease = store.get_active_lease(job_id)
    if not lease:
        raise HTTPException(status_code=404, detail=f"No active lease for job {job_id}")
    return {"ok": True, "lease": lease}


@router.get("/api/events/{entity_type}/{entity_id}", tags=["Events"])
def api_get_events(entity_type: str, entity_id: str, request: Request, limit: int = Query(50, ge=1, le=1000)):
    """Get event history for a job or host."""
    user = _require_auth(request)
    _require_scope(user, "events:read")
    _require_entity_event_access(user, entity_type, entity_id)
    store = get_event_store()
    events = store.get_events(entity_type, entity_id, limit=limit)
    return {"ok": True, "entity_type": entity_type, "entity_id": entity_id, "events": events}


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
def api_instance_audit_trail(job_id: str, request: Request):
    """Full auditable trail for a job — every event with hash chain.

    This is the dispute-resolution artifact: every state change,
    lease renewal, billing event, ordered by time with tamper-evident hashes.
    """
    user = _require_auth(request)
    _require_scope(user, "events:read")
    from routes.instances import _check_job_access

    _check_job_access(user, job_id)
    sm = get_state_machine()
    timeline = sm.get_job_timeline(job_id)
    if not timeline:
        raise HTTPException(404, f"No events for job {job_id}")
    return {"ok": True, "job_id": job_id, "events": timeline, "count": len(timeline)}


@router.get("/api/events", tags=["Events"])
def api_get_all_events(
    request: Request, limit: int = Query(25, ge=1, le=1000),
    before: str | None = Query(None, max_length=256),
    event_type: str | None = Query(None, max_length=128),
    severity: Literal["info", "warning", "error", "critical"] | None = None,
    include_verbose: bool = False,
):
    """Filtered event history, newest first, with an opaque next-page cursor (admin only)."""
    _require_admin(request)
    cursor = None
    if before:
        try:
            at, event_id = json.loads(base64.urlsafe_b64decode(before + "=" * (-len(before) % 4)))
            if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
                raise ValueError()
            if not isinstance(event_id, str) or not event_id:
                raise ValueError()
            cursor = (float(at), event_id)
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid events cursor") from None
    page = get_event_store().get_event_page(limit=limit, before=cursor, event_type=event_type,
                                            severity=severity, include_verbose=include_verbose)
    rows = page["events"]
    last = rows[-1] if rows else None
    next_cursor = None
    if page.pop("has_more") and last:
        raw = json.dumps([float(last["timestamp"]), str(last["event_id"])], separators=(",", ":"))
        next_cursor = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    return {"ok": True, **page, "next_cursor": next_cursor}
