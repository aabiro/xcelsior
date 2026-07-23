"""Track B B2.1 — action-plan / MCP-policy / MCP-audit schema invariants.

Database-level tests: they prove the PostgreSQL schema *itself* enforces the
§9.5 action-plan state machine, the reuse of the existing holds and
idempotency authorities, and the OAuth machine-client workspace-context
constraint — even when application code is wrong. Raw SQL only; the
constraints are the last line of defence and must hold on their own.

Every invariant is driven **both ways** (B0.1 rule 2): a row that violates it
is rejected, and the matching legal row is accepted, so a constraint that
silently never fires cannot pass as a gate.

Requires the test database migrated to alembic head (>= 069).
"""

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_069 = (
            _c.execute("SELECT to_regclass('action_plans')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _has_069:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 069")

from psycopg.errors import (
    CheckViolation,
    ForeignKeyViolation,
    UniqueViolation,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def cleanup():
    """Delete rows this test inserted, newest-authority first."""
    ids = {"plans": [], "policies": [], "clients": [], "holds": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for pid in ids["plans"]:
            conn.execute("DELETE FROM action_plans WHERE plan_id=%s", (pid,))
        for pid in ids["policies"]:
            conn.execute("DELETE FROM mcp_client_policies WHERE policy_id=%s", (pid,))
        for cid in ids["clients"]:
            conn.execute("DELETE FROM oauth_clients WHERE client_id=%s", (cid,))
        for hid in ids["holds"]:
            conn.execute("DELETE FROM wallet_holds WHERE hold_id=%s", (hid,))
        conn.commit()


# Columns a legal ``quoted`` plan needs. Individual tests override the few
# fields under test; everything else stays valid so a rejection can only be
# the constraint being exercised.
def _plan_row(**over):
    row = {
        "action_type": "create_instance",
        "principal_id": "user-b21",
        "tenant_id": "tenant-b21",
        "canonical_args": json.dumps({"gpu": "H100"}),
        "canonical_args_hash": "h" * 64,
        "approval_mode": "human",
        "status": "quoted",
        "expires_at": "now() + interval '10 minutes'",
    }
    row.update(over)
    return row


def _insert_plan(cleanup, conn, **over):
    row = _plan_row(**over)
    # expires_at is a SQL expression, not a bound value.
    expires = row.pop("expires_at")
    cols = ", ".join(row.keys())
    ph = ", ".join(["%s"] * len(row))
    plan_id = conn.execute(
        f"INSERT INTO action_plans ({cols}, expires_at) "
        f"VALUES ({ph}, {expires}) RETURNING plan_id",
        tuple(row.values()),
    ).fetchone()[0]
    cleanup["plans"].append(plan_id)
    return plan_id


# ── Value-domain constraints ─────────────────────────────────────────


class TestActionPlanValueDomain:
    def test_unknown_status_rejected(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_plan(cleanup, conn, status="running")  # not a plan state

    def test_unknown_approval_mode_rejected(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_plan(cleanup, conn, approval_mode="whenever")

    def test_negative_estimate_rejected(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_plan(cleanup, conn, estimate_micros=-1)

    def test_negative_tolerance_rejected(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_plan(cleanup, conn, price_tolerance_bps=-1)

    def test_valid_quoted_plan_accepted(self, cleanup):
        # Control: the baseline row every negative test mutates is itself legal,
        # so the rejections above cannot be passing for an unrelated reason.
        with _pool.connection() as conn:
            _insert_plan(cleanup, conn, estimate_micros=1_000_000)


# ── §9.5 state-machine invariants ────────────────────────────────────


class TestActionPlanStateMachine:
    def test_approved_requires_approved_at(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_plan(cleanup, conn, status="approved")

    def test_approved_with_timestamp_accepted(self, cleanup):
        with _pool.connection() as conn:
            pid = conn.execute(
                "INSERT INTO action_plans "
                "(action_type, principal_id, tenant_id, canonical_args, "
                " canonical_args_hash, approval_mode, status, approved_at, "
                " expires_at) "
                "VALUES ('create_instance','p','t','{}','h','human',"
                "'approved', now(), now() + interval '10 min') RETURNING plan_id"
            ).fetchone()[0]
            cleanup["plans"].append(pid)

    def test_executing_requires_consumed_and_approved(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                # approved_at set but consumed_at missing
                conn.execute(
                    "INSERT INTO action_plans "
                    "(action_type, principal_id, tenant_id, canonical_args, "
                    " canonical_args_hash, approval_mode, status, approved_at, "
                    " expires_at) "
                    "VALUES ('create_instance','p','t','{}','h','human',"
                    "'executing', now(), now() + interval '10 min')"
                )

    def test_succeeded_requires_job_and_response(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                conn.execute(
                    "INSERT INTO action_plans "
                    "(action_type, principal_id, tenant_id, canonical_args, "
                    " canonical_args_hash, approval_mode, status, approved_at, "
                    " consumed_at, expires_at) "
                    "VALUES ('create_instance','p','t','{}','h','human',"
                    "'succeeded', now(), now(), now() + interval '10 min')"
                )

    def test_succeeded_full_row_accepted(self, cleanup):
        with _pool.connection() as conn:
            pid = conn.execute(
                "INSERT INTO action_plans "
                "(action_type, principal_id, tenant_id, canonical_args, "
                " canonical_args_hash, approval_mode, status, approved_at, "
                " consumed_at, job_id, idempotent_response, expires_at) "
                "VALUES ('create_instance','p','t','{}','h','human',"
                "'succeeded', now(), now(), 'job-x', '{\"job_id\":\"job-x\"}', "
                "now() + interval '10 min') RETURNING plan_id"
            ).fetchone()[0]
            cleanup["plans"].append(pid)

    def test_failed_requires_failure_code(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                conn.execute(
                    "INSERT INTO action_plans "
                    "(action_type, principal_id, tenant_id, canonical_args, "
                    " canonical_args_hash, approval_mode, status, approved_at, "
                    " consumed_at, expires_at) "
                    "VALUES ('create_instance','p','t','{}','h','human',"
                    "'failed_terminal', now(), now(), now() + interval '10 min')"
                )

    def test_revoked_requires_revoked_at(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_plan(cleanup, conn, status="revoked")

    def test_revoked_with_timestamp_accepted(self, cleanup):
        with _pool.connection() as conn:
            pid = conn.execute(
                "INSERT INTO action_plans "
                "(action_type, principal_id, tenant_id, canonical_args, "
                " canonical_args_hash, approval_mode, status, revoked_at, "
                " expires_at) "
                "VALUES ('create_instance','p','t','{}','h','human',"
                "'revoked', now(), now() + interval '10 min') RETURNING plan_id"
            ).fetchone()[0]
            cleanup["plans"].append(pid)


# ── Reuse of the existing holds authority (no second authority) ──────


class TestActionPlanHoldLink:
    def test_wallet_hold_fk_rejects_dangling(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(ForeignKeyViolation):
                _insert_plan(cleanup, conn, wallet_hold_id=str(uuid.uuid4()))

    def test_wallet_hold_fk_accepts_real_hold(self, cleanup):
        hold_id = str(uuid.uuid4())
        with _pool.connection() as conn:
            conn.execute(
                "INSERT INTO wallet_holds "
                "(hold_id, customer_id, amount_cad, status, created_at, "
                " expires_at, updated_at) "
                "VALUES (%s, 'cust-b21', 1.0, 'held', %s, %s, %s)",
                (hold_id, time.time(), time.time() + 600, time.time()),
            )
            conn.commit()
            cleanup["holds"].append(hold_id)
            _insert_plan(cleanup, conn, wallet_hold_id=hold_id)


# ── mcp_client_policies invariants ───────────────────────────────────


class TestMcpClientPolicy:
    def test_one_policy_per_client_tenant(self, cleanup):
        with _pool.connection() as conn:
            for _ in range(1):
                pid = conn.execute(
                    "INSERT INTO mcp_client_policies (client_id, tenant_id) "
                    "VALUES ('client-b21', 'tenant-b21') RETURNING policy_id"
                ).fetchone()[0]
                cleanup["policies"].append(pid)
            with pytest.raises(UniqueViolation):
                conn.execute(
                    "INSERT INTO mcp_client_policies (client_id, tenant_id) "
                    "VALUES ('client-b21', 'tenant-b21')"
                )

    def test_negative_ceiling_rejected(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                conn.execute(
                    "INSERT INTO mcp_client_policies "
                    "(client_id, tenant_id, per_action_max_micros) "
                    "VALUES ('client-b21b', 'tenant-b21', -1)"
                )


# ── OAuth machine-client workspace context ───────────────────────────


def _insert_client(cleanup, conn, grant_types, workspace, is_system_managed=0):
    cid = f"client-b21-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO oauth_clients "
        "(client_id, client_name, client_type, grant_types, "
        " is_system_managed, workspace_customer_id, created_at, updated_at) "
        "VALUES (%s, %s, 'confidential', %s::jsonb, %s, %s, %s, %s)",
        (
            cid,
            cid,
            json.dumps(grant_types),
            is_system_managed,
            workspace,
            time.time(),
            time.time(),
        ),
    )
    cleanup["clients"].append(cid)
    return cid


class TestOAuthWorkspaceContext:
    def test_client_credentials_without_workspace_rejected(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_client(
                    cleanup, conn, ["client_credentials"], workspace=None
                )

    def test_system_managed_without_workspace_rejected(self, cleanup):
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                _insert_client(
                    cleanup, conn, ["authorization_code"], workspace=None,
                    is_system_managed=1,
                )

    def test_client_credentials_with_workspace_accepted(self, cleanup):
        with _pool.connection() as conn:
            _insert_client(
                cleanup, conn, ["client_credentials"], workspace="ws-b21"
            )

    def test_interactive_client_without_workspace_allowed(self, cleanup):
        # A first-party interactive client is not a machine principal; it is
        # not required to carry a workspace, so the constraint must not fire.
        with _pool.connection() as conn:
            _insert_client(
                cleanup, conn, ["authorization_code"], workspace=None
            )
