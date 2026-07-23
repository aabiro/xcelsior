"""Persistence for the action-plan lifecycle (§9.5, migration 069).

Thin repository over the ``action_plans`` table. The state machine itself is
enforced by CHECK constraints in the database (migration 069); this module
performs the *transitions* — always as the single authority for a plan's
lifecycle (B0.2 rule 11), never letting a caller write a status directly.

Every function takes an open ``conn`` so the caller controls the transaction
boundary: preview persists in its own short transaction, while execute (B2.5)
transitions the plan *inside* the same transaction that creates the wallet
hold, job, idempotency row, and outbox events, so the plan is consumed
atomically with its effects.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb


def _row_to_dict(cur: Any) -> dict[str, Any] | None:
    row = cur.fetchone()
    if row is None:
        return None
    cols = [c.name for c in cur.description]
    return dict(zip(cols, row))


def create_quoted_plan(
    conn: Connection,
    *,
    action_type: str,
    principal_id: str,
    client_id: str | None,
    tenant_id: str,
    team_id: str | None,
    canonical_args: dict[str, Any],
    canonical_args_hash: str,
    spec_hash: str,
    quote_id: str,
    pricing_version: str,
    estimate_micros: int,
    currency: str,
    price_tolerance_bps: int,
    required_scopes: list[str],
    approval_mode: str,
    ttl_sec: int,
) -> dict[str, Any]:
    """Insert a fresh ``quoted`` plan and return it. No side effects beyond it."""
    plan_id = str(uuid.uuid4())
    cur = conn.execute(
        """
        INSERT INTO action_plans (
            plan_id, action_type, principal_id, client_id, tenant_id, team_id,
            canonical_args, canonical_args_hash, spec_hash, quote_id,
            pricing_version, estimate_micros, currency, price_tolerance_bps,
            required_scopes, approval_mode, status, expires_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, 'quoted',
            clock_timestamp() + make_interval(secs => %s)
        )
        RETURNING *
        """,
        (
            plan_id,
            action_type,
            principal_id,
            client_id,
            tenant_id,
            team_id,
            Jsonb(canonical_args),
            canonical_args_hash,
            spec_hash,
            quote_id,
            pricing_version,
            int(estimate_micros),
            currency,
            int(price_tolerance_bps),
            list(required_scopes),
            approval_mode,
            int(ttl_sec),
        ),
    )
    plan = _row_to_dict(cur)
    if plan is None:  # pragma: no cover - INSERT ... RETURNING always yields
        raise RuntimeError("INSERT ... RETURNING returned no row")
    return plan


def get_plan(conn: Connection, plan_id: str) -> dict[str, Any] | None:
    cur = conn.execute("SELECT * FROM action_plans WHERE plan_id = %s", (plan_id,))
    return _row_to_dict(cur)


def get_plan_for_update(conn: Connection, plan_id: str) -> dict[str, Any] | None:
    """Row-locked read for execute/approve — serializes concurrent callers."""
    cur = conn.execute(
        "SELECT * FROM action_plans WHERE plan_id = %s FOR UPDATE", (plan_id,)
    )
    return _row_to_dict(cur)


def is_expired(plan: dict[str, Any], *, now: Any = None) -> bool:
    """A plan past its expiry that never reached a terminal/approved state."""
    import datetime as _dt

    now = now or _dt.datetime.now(_dt.timezone.utc)
    expires_at = plan.get("expires_at")
    return bool(expires_at is not None and expires_at < now)


def mark_approved(
    conn: Connection,
    plan_id: str,
    *,
    approved_by: str,
    approval_method: str,
    approval_session_id: str | None = None,
) -> dict[str, Any]:
    """quoted/awaiting_approval → approved. Caller must hold the row lock."""
    cur = conn.execute(
        """
        UPDATE action_plans
           SET status = 'approved',
               approved_at = clock_timestamp(),
               approved_by = %s,
               approval_method = %s,
               approval_session_id = %s,
               version = version + 1
         WHERE plan_id = %s
        RETURNING *
        """,
        (approved_by, approval_method, approval_session_id, plan_id),
    )
    plan = _row_to_dict(cur)
    if plan is None:  # pragma: no cover
        raise RuntimeError("plan vanished during approval")
    return plan


def mark_revoked(conn: Connection, plan_id: str, *, reason: str) -> dict[str, Any]:
    """→ revoked. Caller must hold the row lock and have checked the source state."""
    cur = conn.execute(
        """
        UPDATE action_plans
           SET status = 'revoked',
               revoked_at = clock_timestamp(),
               revoked_reason = %s,
               version = version + 1
         WHERE plan_id = %s
        RETURNING *
        """,
        (reason, plan_id),
    )
    plan = _row_to_dict(cur)
    if plan is None:  # pragma: no cover
        raise RuntimeError("plan vanished during revoke")
    return plan


def mark_consumed(
    conn: Connection,
    plan_id: str,
    *,
    job_id: str,
    wallet_hold_id: str | None,
    idempotent_response: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    """approved → succeeded, atomically with its effects. Caller holds the lock.

    Sets every column the §9.5 ``succeeded`` CHECK requires — ``consumed_at``,
    ``job_id``, ``idempotent_response`` — plus the links to the wallet hold and
    the idempotency key it was executed under, so a replay returns this exact
    response instead of launching again.
    """
    cur = conn.execute(
        """
        UPDATE action_plans
           SET status = 'succeeded',
               consumed_at = clock_timestamp(),
               job_id = %s,
               wallet_hold_id = %s,
               idempotent_response = %s,
               idempotency_key = %s,
               resulting_resource_id = %s,
               version = version + 1
         WHERE plan_id = %s
        RETURNING *
        """,
        (
            job_id,
            wallet_hold_id,
            Jsonb(idempotent_response),
            idempotency_key,
            job_id,
            plan_id,
        ),
    )
    plan = _row_to_dict(cur)
    if plan is None:  # pragma: no cover
        raise RuntimeError("plan vanished during consume")
    return plan


def mark_expired(conn: Connection, plan_id: str) -> dict[str, Any]:
    """quoted/awaiting_approval/approved → expired. Caller holds the row lock."""
    cur = conn.execute(
        """
        UPDATE action_plans
           SET status = 'expired', version = version + 1
         WHERE plan_id = %s
        RETURNING *
        """,
        (plan_id,),
    )
    plan = _row_to_dict(cur)
    if plan is None:  # pragma: no cover
        raise RuntimeError("plan vanished during expiry")
    return plan


def load_client_policy(
    conn: Connection, *, client_id: str | None, tenant_id: str
) -> dict[str, Any] | None:
    """The standing spend policy for this (client, tenant), if any."""
    if not client_id:
        return None
    cur = conn.execute(
        "SELECT * FROM mcp_client_policies WHERE client_id = %s AND tenant_id = %s",
        (client_id, tenant_id),
    )
    return _row_to_dict(cur)
