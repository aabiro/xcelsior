"""Durable security decisions, independent of transient job failure/retry state.

Uses the existing transactional state store. Suspensions are administrator
decisions; a worker detection records a review flag, never an account ban.
A mining detection also revokes funded internet until an administrator clears
it. Payment never opens the host network: it only selects the funded bridge.
"""

from __future__ import annotations

import logging
import time

from db import DatabaseOps, get_engine

log = logging.getLogger(__name__)

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
        # Card-funded mining loses direct internet. The account is not banned.
        set_egress_revoked(
            conn,
            user_id=str(owner),
            active=True,
            actor="mining-detector",
            reason=MINING_REASON,
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
    """Pick the network for this account.

    A mining revocation wins. An administrator ``open`` decision is the only
    way off the isolated bridges. A settled card deposit selects the funded
    bridge: public internet, host and LAN still closed. Promotional credit
    does not. Anything else, including a lookup failure, stays on the proxy.
    """
    from egress_isolation import FUNDED, OPEN, RESTRICTED

    identifiers = _identifiers(principal)
    if _any_active("security.egress_revoked", identifiers):
        return RESTRICTED
    if _any_active("security.egress_open", identifiers):
        return OPEN
    if _card_paid(identifiers):
        return FUNDED
    return RESTRICTED


def _card_paid(identifiers: list[str]) -> bool:
    if _any_active("security.card_paid", identifiers):
        return True
    for user_id in identifiers:
        if _backfill_card_paid(user_id):
            return True
    return False


def _backfill_card_paid(user_id: str) -> bool:
    """A deposit recorded before this flag existed still counts as payment."""
    try:
        with get_engine().transaction() as (conn, backend):
            row = conn.execute(
                """SELECT 1 AS ok FROM payment_intents
                   WHERE customer_id = %s AND status = 'succeeded' AND amount_cents > 0
                   LIMIT 1""",
                (user_id,),
            ).fetchone()
            if not row:
                return False
            set_card_paid(
                conn,
                user_id=user_id,
                active=True,
                actor="billing",
                reason="settled card deposit",
                backend=backend,
            )
            return True
    except Exception:
        log.warning("card payment lookup failed for egress; leaving internet closed", exc_info=True)
        return False


def set_card_paid(
    conn, *, user_id: str, active: bool, actor: str, reason: str, backend: str | None = None
) -> None:
    _write_decision(conn, "security.card_paid", user_id, active, actor, reason, backend)


def set_egress_revoked(
    conn, *, user_id: str, active: bool, actor: str, reason: str, backend: str | None = None
) -> None:
    _write_decision(conn, "security.egress_revoked", user_id, active, actor, reason, backend)


def _write_decision(conn, prefix, user_id, active, actor, reason, backend) -> None:
    user_id, actor, reason = (str(v or "").strip() for v in (user_id, actor, reason))
    if not user_id or not actor or not reason:
        raise ValueError("A security decision requires a user, actor and reason")
    backend = _resolve_backend(conn, backend)
    decision = {"active": active, "actor": actor, "reason": reason[:500], "updated_at": time.time()}
    DatabaseOps.upsert_state(conn, f"{prefix}.{user_id}", decision, backend=backend)


def set_egress_open(
    conn, *, user_id: str, active: bool, actor: str, reason: str, backend: str | None = None
) -> None:
    _write_decision(conn, "security.egress_open", user_id, active, actor, reason, backend)


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
