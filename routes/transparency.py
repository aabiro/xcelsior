"""Routes: transparency."""

import time
import uuid
from contextlib import contextmanager

from fastapi import APIRouter, Request
from pydantic import BaseModel

from events import Event, get_event_store

router = APIRouter()


# ── Helper: _transparency_db ──


@contextmanager
def _transparency_db():
    """PostgreSQL connection for transparency tables."""
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise


# ── Model: LegalRequestRecord ──


class LegalRequestRecord(BaseModel):
    request_type: str = "subpoena"  # subpoena, warrant, mlat, production_order, informal
    jurisdiction: str = "CA"
    authority: str = ""
    scope: str = ""
    notes: str = ""


@router.post("/api/transparency/legal-request", tags=["Transparency"])
def api_record_legal_request(req: LegalRequestRecord, request: Request):
    """Record a legal request (subpoena, warrant, MLAT, etc.)."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "transparency:write")
    import uuid

    with _transparency_db() as conn:
        request_id = str(uuid.uuid4())[:12]
        conn.execute(
            """INSERT INTO legal_requests
               (request_id, received_at, request_type, jurisdiction, authority, scope, notes)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                request_id,
                time.time(),
                req.request_type,
                req.jurisdiction,
                req.authority,
                req.scope,
                req.notes,
            ),
        )

    # Also record as an auditable event in the hash chain
    store = get_event_store()
    store.append(
        Event(
            event_type="transparency.legal_request",
            entity_type="legal",
            entity_id=request_id,
            actor="admin",
            data={"request_type": req.request_type, "jurisdiction": req.jurisdiction},
        )
    )

    return {"ok": True, "request_id": request_id}


@router.post("/api/transparency/legal-request/{request_id}/respond", tags=["Transparency"])
def api_respond_legal_request(
    request_id: str,
    request: Request,
    complied: bool = False,
    challenged: bool = False,
    notes: str = "",
):
    """Record response to a legal request."""
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "transparency:write")
    with _transparency_db() as conn:
        conn.execute(
            """UPDATE legal_requests
               SET status = 'responded', responded_at = %s, complied = %s, challenged = %s, notes = %s
               WHERE request_id = %s""",
            (time.time(), int(complied), int(challenged), notes, request_id),
        )
    return {"ok": True, "request_id": request_id}


@router.get("/api/transparency/report", tags=["Transparency"])
def api_transparency_report(request: Request, months: int = 12):
    """Generate transparency report — CLOUD Act diligence artifact.

    Returns summary of all legal requests and data disclosures.
    Monthly JSON per REPORT_FEATURE_2.md Phase B §3.
    """
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "transparency:read")
    with _transparency_db() as conn:
        since = time.time() - (months * 30 * 86400)

        requests_rows = conn.execute(
            "SELECT * FROM legal_requests WHERE received_at >= %s ORDER BY received_at DESC",
            (since,),
        ).fetchall()

        disclosures_rows = conn.execute(
            "SELECT * FROM data_disclosures WHERE disclosed_at >= %s ORDER BY disclosed_at DESC",
            (since,),
        ).fetchall()

    requests_list = [dict(r) for r in requests_rows]
    disclosures_list = [dict(r) for r in disclosures_rows]

    # Summary statistics
    total = len(requests_list)
    complied = sum(1 for r in requests_list if r.get("complied"))
    challenged = sum(1 for r in requests_list if r.get("challenged"))
    by_type = {}
    for r in requests_list:
        t = r.get("request_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    by_jurisdiction = {}
    for r in requests_list:
        j = r.get("jurisdiction", "unknown")
        by_jurisdiction[j] = by_jurisdiction.get(j, 0) + 1

    return {
        "ok": True,
        "period_months": months,
        "generated_at": time.time(),
        "summary": {
            "requests_received": total,
            "complied": complied,
            "challenged": challenged,
            "pending": total - complied - challenged,
            "by_type": by_type,
            "by_jurisdiction": by_jurisdiction,
            "data_disclosures": len(disclosures_list),
        },
        "cloud_act_note": (
            "Xcelsior is a Canadian-controlled entity. All data resides in Canadian "
            "jurisdiction. Foreign legal process requires MLAT through Canadian courts. "
            "No US CLOUD Act compelled disclosure has been made."
        ),
        "requests": requests_list,
        "disclosures": disclosures_list,
    }
