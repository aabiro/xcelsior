"""Durable security decisions, independent of transient job failure/retry state.

Uses the existing transactional state store. Suspensions are administrator
decisions; a worker detection records a review flag, never an account ban.
"""
from __future__ import annotations

import time

from db import DatabaseOps, get_engine

MINING_REASON = "mining_detected"


def job_blocked(conn, job_id: str, *, backend="postgres") -> bool:
    return bool(DatabaseOps.get_state(conn, f"security.job.{job_id}", backend=backend))


def record_mining_violation(conn, job: dict, *, backend="postgres") -> None:
    """Caller holds the job row lock; commit together with its terminal state."""
    record = {"job_id": job["job_id"], "reason": MINING_REASON, "detected_at": time.time()}
    DatabaseOps.upsert_state(conn, f"security.job.{job['job_id']}", record, backend=backend)
    owner = job.get("owner") or job.get("owner_id")
    if owner:
        # One record per incident prevents concurrent reports overwriting flags.
        DatabaseOps.upsert_state(
            conn, f"security.review.{owner}.{job['job_id']}",
            dict(record, account_id=owner, status="needs_review"), backend=backend,
        )


def account_suspended(principal: dict) -> bool:
    identifiers = {
        str(principal.get(k) or "").strip()
        for k in ("user_id", "customer_id", "email", "workspace_customer_id", "created_by_email", "client_id", "sub")
    } - {""}
    if not identifiers:
        return False
    # Never cache an allow decision or fail open when the store is unavailable.
    with get_engine().connection() as (conn, backend):
        return any(
            (DatabaseOps.get_state(conn, f"security.suspended.{uid}", backend=backend) or {}).get("active")
            for uid in identifiers
        )


def set_account_suspension(conn, *, user_id: str, active: bool, actor: str, reason: str, backend="postgres") -> None:
    if not user_id or not actor or not reason:
        raise ValueError("Suspension requires a user, actor and reason")
    decision = {"active": active, "actor": actor, "reason": reason[:500], "updated_at": time.time()}
    DatabaseOps.upsert_state(conn, f"security.suspended.{user_id}", decision, backend=backend)
