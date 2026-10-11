"""Durable security decisions, independent of transient job failure/retry state.

Uses the existing transactional state store. Suspensions are administrator
decisions; a worker detection records a review flag, never an account ban.
"""

from __future__ import annotations

import time

from db import DatabaseOps, get_engine

MINING_REASON = "mining_detected"


def _resolve_backend(conn=None, backend: str | None = None) -> str:
    if backend:
        return backend
    if conn is not None:
        # By module name: importing sqlite3 here would put shared security state
        # outside the local-state boundary (tests/test_local_state_boundary.py).
        module = type(conn).__module__
        if module == "sqlite3":
            return "sqlite"
        if module.startswith("psycopg") or hasattr(conn, "pgconn"):
            return "postgres"
    try:
        engine = get_engine()
        if engine.backend == "postgres" or (
            engine.backend == "dual" and engine.read_from == "postgres"
        ):
            return "postgres"
        elif engine.backend == "sqlite" or (
            engine.backend == "dual" and engine.read_from == "sqlite"
        ):
            return "sqlite"
        return getattr(engine, "backend", "postgres")
    except Exception:
        return "postgres"


def job_blocked(conn, job_id: str, *, backend: str | None = None) -> bool:
    if not job_id:
        return False
    backend = _resolve_backend(conn, backend)
    return bool(DatabaseOps.get_state(conn, f"security.job.{job_id}", backend=backend))


def record_mining_violation(conn, job: dict, *, backend: str | None = None) -> None:
    """Caller holds the job row lock; commit together with its terminal state."""
    if not job or not isinstance(job, dict):
        return
    job_id = job.get("job_id")
    if not job_id:
        return
    backend = _resolve_backend(conn, backend)
    record = {"job_id": job_id, "reason": MINING_REASON, "detected_at": time.time()}
    DatabaseOps.upsert_state(conn, f"security.job.{job_id}", record, backend=backend)
    owner = job.get("owner") or job.get("owner_id")
    if owner:
        # One record per incident prevents concurrent reports overwriting flags.
        DatabaseOps.upsert_state(
            conn,
            f"security.review.{owner}.{job_id}",
            dict(record, account_id=owner, status="needs_review"),
            backend=backend,
        )


_IDENTIFIER_KEYS = (
    "user_id",
    "customer_id",
    "email",
    "workspace_customer_id",
    "created_by_email",
    "client_id",
    "sub",
    "owner",
    "owner_id",
    "account_id",
)


def _identifiers(principal: dict | str | None) -> list[str]:
    if not principal:
        return []
    if isinstance(principal, str):
        principal = {"user_id": principal}
    elif not isinstance(principal, dict):
        return []
    return sorted({str(principal.get(k) or "").strip() for k in _IDENTIFIER_KEYS} - {""})


def _any_active(prefix: str, identifiers: list[str]) -> bool:
    if not identifiers:
        return False
    # One round trip, however many identifiers. Never cache an allow decision or
    # fail open when the store is unavailable: errors propagate to the caller.
    with get_engine().connection() as (conn, backend):
        found = DatabaseOps.get_states(
            conn, [f"{prefix}.{uid}" for uid in identifiers], backend=backend
        )
    return any(bool((record or {}).get("active")) for record in found.values())


def account_suspended(principal: dict | str | None) -> bool:
    return _any_active("security.suspended", _identifiers(principal))


def egress_for_owner(principal: dict | str | None) -> str:
    """Rented containers get restricted egress unless an administrator opened it.

    Restricted means the isolated bridge plus the allowlisting proxy
    (``egress_isolation``). The job tier is not a trust signal: anyone can
    pick it, and signup credit funds it.
    """
    from egress_isolation import OPEN, RESTRICTED

    return OPEN if _any_active("security.egress_open", _identifiers(principal)) else RESTRICTED


def set_egress_open(
    conn, *, user_id: str, active: bool, actor: str, reason: str, backend: str | None = None
) -> None:
    user_id, actor, reason = (str(v or "").strip() for v in (user_id, actor, reason))
    if not user_id or not actor or not reason:
        raise ValueError("Opening egress requires a user, actor and reason")
    backend = _resolve_backend(conn, backend)
    decision = {"active": active, "actor": actor, "reason": reason[:500], "updated_at": time.time()}
    DatabaseOps.upsert_state(conn, f"security.egress_open.{user_id}", decision, backend=backend)


def set_account_suspension(
    conn=None, *, user_id: str, active: bool, actor: str, reason: str, backend: str | None = None
) -> None:
    user_id = str(user_id or "").strip()
    actor = str(actor or "").strip()
    reason = str(reason or "").strip()
    if not user_id or not actor or not reason:
        raise ValueError("Suspension requires a user, actor and reason")
    if conn is None:
        with get_engine().connection() as (c, b):
            return set_account_suspension(
                c, user_id=user_id, active=active, actor=actor, reason=reason, backend=b
            )
    backend = _resolve_backend(conn, backend)
    decision = {"active": active, "actor": actor, "reason": reason[:500], "updated_at": time.time()}
    DatabaseOps.upsert_state(conn, f"security.suspended.{user_id}", decision, backend=backend)


def suspend_account(
    conn=None, *, identifier: str, actor: str, reason: str, backend: str | None = None
) -> list[str]:
    """Suspend an account across all linked identifiers (user_id, customer_id, email)."""
    identifier = str(identifier or "").strip()
    if not identifier:
        return []
    if conn is None:
        with get_engine().connection() as (c, b):
            return suspend_account(c, identifier=identifier, actor=actor, reason=reason, backend=b)

    ids = {identifier}
    try:
        from db import UserStore

        user = (
            UserStore.get_user(identifier)
            or UserStore.get_user_by_id(identifier)
            or UserStore.get_user_by_customer_id(identifier)
            or {}
        )
    except Exception:
        user = {}
    if not user:
        try:
            from routes._deps import _users_db

            user = _users_db.get(identifier) or next(
                (
                    u
                    for u in _users_db.values()
                    if u.get("customer_id") == identifier or u.get("user_id") == identifier
                ),
                {},
            )
        except Exception:
            pass

    for k in ("user_id", "customer_id", "email"):
        val = str(user.get(k) or "").strip()
        if val:
            ids.add(val)
    backend = _resolve_backend(conn, backend)
    for uid in sorted(ids):
        set_account_suspension(
            conn, user_id=uid, active=True, actor=actor, reason=reason, backend=backend
        )
    return sorted(ids)


def unsuspend_account(
    conn=None, *, identifier: str, actor: str, reason: str, backend: str | None = None
) -> list[str]:
    """Unsuspend an account across all linked identifiers."""
    identifier = str(identifier or "").strip()
    if not identifier:
        return []
    if conn is None:
        with get_engine().connection() as (c, b):
            return unsuspend_account(
                c, identifier=identifier, actor=actor, reason=reason, backend=b
            )

    ids = {identifier}
    try:
        from db import UserStore

        user = (
            UserStore.get_user(identifier)
            or UserStore.get_user_by_id(identifier)
            or UserStore.get_user_by_customer_id(identifier)
            or {}
        )
    except Exception:
        user = {}
    if not user:
        try:
            from routes._deps import _users_db

            user = _users_db.get(identifier) or next(
                (
                    u
                    for u in _users_db.values()
                    if u.get("customer_id") == identifier or u.get("user_id") == identifier
                ),
                {},
            )
        except Exception:
            pass

    for k in ("user_id", "customer_id", "email"):
        val = str(user.get(k) or "").strip()
        if val:
            ids.add(val)
    backend = _resolve_backend(conn, backend)
    for uid in sorted(ids):
        set_account_suspension(
            conn, user_id=uid, active=False, actor=actor, reason=reason, backend=backend
        )
    return sorted(ids)
