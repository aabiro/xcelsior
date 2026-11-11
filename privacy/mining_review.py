"""The control plane's side of the resource-abuse response (CVP-01 brief §3).

A worker observes one container and walks it up a ladder (flag, throttle,
suspend, terminate). Every pass of an open episode is reported here; this
module keeps the record an administrator reviews, raises the alert, and
answers with what only the control plane knows:

* ``prior``: the account prior (signal S5), at most 0.05; a first flagged
  instance on an account without a settled payment.
* ``repeat_offender``: another instance of this account was suspended or
  terminated for the same reason before. With it, 0.90 confidence ends the
  instance without waiting for a human.
* ``decision``: ``clear`` or ``terminate`` from an administrator, which the
  worker applies on its next pass.

State lives in the ``state`` table next to the other security decisions:
``security.mining.job.<job_id>`` (the episode), ``security.mining.account.<owner>``
(the account's episodes), and ``security.review.<owner>.<job_id>`` (the review
queue entry, same shape as a confirmed violation's).
"""

from __future__ import annotations

import json
import logging
import time

from db import DatabaseOps, get_engine

log = logging.getLogger("xcelsior.mining")

RUNGS = ("flag", "throttle", "suspend", "terminate")
DECISIONS = ("clear", "terminate")
REASON = "mining_detected"
PRIOR = 0.05
EVIDENCE_MAX_BYTES = 16_384

JOB_NS = "security.mining.job."
ACCOUNT_NS = "security.mining.account."


def _job_ns(job_id: str) -> str:
    return f"{JOB_NS}{job_id}"


def _account_ns(owner: str) -> str:
    return f"{ACCOUNT_NS}{owner}"


def _trim_evidence(evidence: dict | None) -> dict:
    evidence = dict(evidence or {})
    while len(json.dumps(evidence, default=str)) > EVIDENCE_MAX_BYTES:
        signals = evidence.get("signals")
        if isinstance(signals, list) and signals:
            signals.pop()
            continue
        return {"truncated": True}
    return evidence


def _emit(job_id: str, owner: str, host_id: str, record: dict) -> None:
    """Append the security event and raise the operator alert; both best-effort."""
    try:
        from events import Event, EventType, get_event_store

        get_event_store().append(
            Event(
                event_type=EventType.SECURITY_MINING_DETECTED.value,
                entity_type="job",
                entity_id=job_id,
                actor=f"agent:{host_id}",
                data={
                    "owner": owner,
                    "rung": record.get("rung"),
                    "confidence": record.get("confidence"),
                    "container_name": record.get("container_name"),
                    "signals": [
                        s.get("detail") for s in (record.get("evidence") or {}).get("signals") or []
                    ],
                },
            )
        )
    except Exception as exc:  # the record is written either way
        log.warning("mining event not appended for %s: %s", job_id, exc)
    try:
        from scheduler import alert

        alert(
            f"MINING {str(record.get('rung') or '').upper()}: job {job_id}",
            f"host {host_id} container {record.get('container_name')} owner {owner} "
            f"confidence {float(record.get('confidence') or 0):.2f}",
        )
    except Exception as exc:
        log.warning("mining alert not sent for %s: %s", job_id, exc)


def record_episode(
    *,
    host_id: str,
    job_id: str,
    container_name: str,
    rung: str | None,
    confidence: float,
    evidence: dict | None,
) -> dict:
    """One pass of a worker's episode. Returns the reply the worker acts on.

    ``{"ok": False, "error": ...}`` when the job is unknown or is not this
    host's: a host may only speak for the containers it runs.
    """
    job_id = str(job_id or "").strip()
    if not job_id:
        return {"ok": False, "error": "job_id required"}
    now = time.time()
    with get_engine().transaction() as (conn, backend):
        job = DatabaseOps.get_job(conn, job_id, backend=backend)
        if not job:
            return {"ok": False, "error": "unknown job"}
        if str(job.get("host_id") or "") != str(host_id or ""):
            return {"ok": False, "error": "job is not on this host"}
        owner = str(job.get("owner") or job.get("owner_id") or "")
        ns = _job_ns(job_id)
        record = DatabaseOps.get_state(conn, ns, backend=backend) or {}
        account = (
            DatabaseOps.get_state(conn, _account_ns(owner), backend=backend) if owner else None
        ) or {"jobs": {}}
        first_sight = not record
        previous_rung = record.get("rung")
        closed = bool((evidence or {}).get("closed"))

        if first_sight:
            record = {"job_id": job_id, "owner": owner, "opened_at": now}
        record.update(
            {
                "host_id": host_id,
                "container_name": container_name,
                "rung": rung,
                "confidence": float(confidence or 0),
                "evidence": _trim_evidence(evidence),
                "suspended_at": (evidence or {}).get("suspended_at"),
                "resumed_at": (evidence or {}).get("resumed_at"),
                "updated_at": now,
                "closed": (evidence or {}).get("closed") if closed else None,
            }
        )
        # Signal S5: an unfunded account's first flagged instance.
        funded = str(job.get("egress") or "").lower() in ("funded", "open")
        others = {j: r for j, r in (account.get("jobs") or {}).items() if j != job_id}
        prior = PRIOR if (not funded and not others) else 0.0
        repeat = any(r.get("rung") in ("suspend", "terminate") for r in others.values()) or any(
            DatabaseOps.get_state(conn, f"security.job.{j}", backend=backend) for j in others
        )
        if owner:
            account.setdefault("jobs", {})[job_id] = {
                "rung": rung,
                "updated_at": now,
                "closed": record["closed"],
            }
            DatabaseOps.upsert_state(conn, _account_ns(owner), account, backend=backend)
            if rung and rung != previous_rung:
                DatabaseOps.upsert_state(
                    conn,
                    f"security.review.{owner}.{job_id}",
                    {
                        "job_id": job_id,
                        "reason": REASON,
                        "detected_at": record["opened_at"],
                        "account_id": owner,
                        "status": "needs_review",
                        "rung": rung,
                        "confidence": record["confidence"],
                    },
                    backend=backend,
                )
        DatabaseOps.upsert_state(conn, ns, record, backend=backend)
    if rung and rung != previous_rung:
        _emit(job_id, owner, host_id, record)
    return {
        "ok": True,
        "received": True,
        "prior": prior,
        "repeat_offender": bool(repeat),
        "decision": record.get("decision"),
    }


def decide(job_id: str, decision: str, *, actor: str, reason: str) -> dict:
    """An administrator's word on an open episode; the worker applies it within a minute."""
    decision = str(decision or "").strip().lower()
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    actor, reason = str(actor or "").strip(), str(reason or "").strip()
    if not actor or not reason:
        raise ValueError("A decision requires an actor and a reason")
    with get_engine().transaction() as (conn, backend):
        ns = _job_ns(job_id)
        record = DatabaseOps.get_state(conn, ns, backend=backend)
        if not record:
            raise KeyError(job_id)
        record.update(
            {
                "decision": decision,
                "decided_by": actor,
                "decided_at": time.time(),
                "decision_reason": reason[:500],
            }
        )
        DatabaseOps.upsert_state(conn, ns, record, backend=backend)
        owner = record.get("owner")
        if owner:
            review = DatabaseOps.get_state(
                conn, f"security.review.{owner}.{job_id}", backend=backend
            )
            if review:
                review["status"] = "cleared" if decision == "clear" else "confirmed"
                review["decided_by"] = actor
                DatabaseOps.upsert_state(
                    conn, f"security.review.{owner}.{job_id}", review, backend=backend
                )
    return record


def list_episodes(*, include_closed: bool = False) -> list[dict]:
    pattern = f"{JOB_NS}%"
    with get_engine().connection() as (conn, backend):
        if backend == "postgres":
            rows = conn.execute(
                "SELECT payload FROM state WHERE namespace LIKE %s", (pattern,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT payload FROM state WHERE namespace LIKE ?", (pattern,)
            ).fetchall()
    records = []
    for row in rows:
        payload = row["payload"] if isinstance(row, dict) else row[0]
        record = DatabaseOps.decode_payload(payload)
        if not isinstance(record, dict):
            continue
        if record.get("closed") and not include_closed:
            continue
        records.append(record)
    records.sort(key=lambda r: float(r.get("updated_at") or 0), reverse=True)
    return records
