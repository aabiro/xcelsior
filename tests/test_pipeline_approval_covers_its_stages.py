"""Gate P4, clause by clause, through the routes rather than the executor.

`tests/test_pipeline_runs_in_order_and_halts.py` drives `run_pipeline` directly
and proves the ordering, halt, retry and ceiling semantics. This file is the
gate: it goes through `POST /api/v1/pipelines`, approval, and
`POST /api/v1/pipelines/{id}/execute`, because three of the four clauses are
about the *approval*, and an executor test cannot reach an approval it never
touches.

| Clause | Asserted here |
|---|---|
| One approval, three stages, one audit chain | one plan row, three stage rows sharing its `plan_id` |
| A mid-pipeline failure does not silently continue | executor tests; the route inherits it |
| The approved graph is server-bound; editing a stage invalidates it | **by attempting exactly that** |
| Spend is bounded by what was approved | the quote is the plan's `estimate_micros`, and execute passes it as the ceiling |

The third clause is the one that needed the route. Until now it was asserted on
the *hash* — that editing a stage changes it — which is a fact about a function,
not about whether anything refuses. The clause says "asserted by attempting
exactly that", so this alters an approved plan in the database and calls execute.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = (
            _c.execute("SELECT to_regclass('pipeline_stages')").fetchone()[0] is not None
            and _c.execute("SELECT to_regclass('action_plans')").fetchone()[0] is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 104")


GRAPH = {
    "name": "train-eval-serve",
    "stages": [
        {"name": "train", "action_type": "create_instance", "estimate_micros": 1000},
        {"name": "evaluate", "action_type": "create_instance", "estimate_micros": 500},
        {"name": "serve", "action_type": "create_serverless_endpoint", "estimate_micros": 250},
    ],
}


@pytest.fixture
def client(monkeypatch):
    """A caller holding every scope the graph needs."""
    from fastapi.testclient import TestClient

    import api as api_mod
    import routes.action_plans as ap
    from routes import _deps

    tag = uuid.uuid4().hex[:10]
    tenant = f"tenant-{tag}"
    principal = {
        "email": "demo@xcelsior.ca", "user_id": f"user-{tag}", "role": "user",
        "auth_type": "oauth_access_token", "session_type": "browser",
        "client_id": "xcelsior-web", "customer_id": tenant,
        "scopes": ["instances:write", "instances:read", "instances:operate", "inference:write"],
    }
    monkeypatch.setattr(_deps, "_require_auth", lambda request: dict(principal))
    monkeypatch.setattr(ap, "_require_auth", lambda request: dict(principal))
    monkeypatch.setattr(ap, "_effective_billing_customer_id", lambda user: tenant)
    monkeypatch.setattr(ap, "_canonical_owner_id", lambda user: principal["user_id"])
    monkeypatch.setattr(ap, "_user_team_id", lambda user: None)

    yield TestClient(api_mod.app), tenant

    with pg_transaction() as conn:
        conn.execute("DELETE FROM pipeline_stages WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM action_plans WHERE tenant_id = %s", (tenant,))


def _approve(plan_id: str) -> None:
    """Approve directly. The approval *path* is P1's gate, not P4's."""
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE action_plans SET status = 'approved', approved_at = clock_timestamp() "
            " WHERE plan_id = %s",
            (plan_id,),
        )


def test_one_approval_covers_three_stages(client):
    """Clause 1. One plan row, three stage rows, one shared id."""
    c, tenant = client
    created = c.post("/api/v1/pipelines", json=GRAPH)
    assert created.status_code == 200, created.text
    body = created.json()
    plan_id = body["plan_id"]

    assert len(body["stages"]) == 3
    assert body["approved_max_micros"] == 1750, "the ceiling is not the sum of the stages"

    with pg_transaction() as conn:
        plans = conn.execute(
            "SELECT count(*) FROM action_plans WHERE tenant_id = %s AND action_type = 'run_pipeline'",
            (tenant,),
        ).fetchone()[0]
    assert plans == 1, f"{plans} approvals for one pipeline; the clause asks for one"

    _approve(plan_id)
    c.post(f"/api/v1/pipelines/{plan_id}/execute")

    with pg_transaction() as conn:
        rows = conn.execute(
            "SELECT count(*) FROM pipeline_stages WHERE plan_id = %s", (plan_id,)
        ).fetchone()[0]
    assert rows == 3, "the three stages do not share the one plan's id"


def test_the_quote_is_stated_before_approval(client):
    """Clause 4's precondition: the ceiling is visible *before* anyone approves.

    A total disclosed after approval is a bill, not a budget.
    """
    c, _ = client
    body = c.post("/api/v1/pipelines", json=GRAPH).json()
    assert body["approval_state"] != "approved"
    assert body["approved_max_micros"] == 1750
    assert body["currency"] == "CAD"


def test_editing_a_stage_after_approval_invalidates_it(client):
    """Clause 3, asserted by attempting exactly that.

    This is why the route exists. Asserting that the hash changes proves a
    property of a function; it proves nothing about whether anything refuses.
    """
    c, _ = client
    plan_id = c.post("/api/v1/pipelines", json=GRAPH).json()["plan_id"]
    _approve(plan_id)

    # Alter the approved graph in place — the tampering the clause names.
    with pg_transaction() as conn:
        canonical = dict(
            conn.execute(
                "SELECT canonical_args FROM action_plans WHERE plan_id = %s", (plan_id,)
            ).fetchone()[0]
        )
        canonical["stages"][2]["action_type"] = "evict_host_workloads"
        conn.execute(
            "UPDATE action_plans SET canonical_args = %s WHERE plan_id = %s",
            (json.dumps(canonical), plan_id),
        )

    response = c.post(f"/api/v1/pipelines/{plan_id}/execute")
    assert response.status_code == 409, (
        f"an altered pipeline executed anyway ({response.status_code}) — the "
        "approval covered a different graph than the one that ran"
    )
    assert "argument_hash_mismatch" in response.text

    with pg_transaction() as conn:
        stages = conn.execute(
            "SELECT count(*) FROM pipeline_stages WHERE plan_id = %s", (plan_id,)
        ).fetchone()[0]
    assert stages == 0, "an altered pipeline materialised stages before being refused"


def test_an_unapproved_pipeline_does_not_run(client):
    """The approval is not decorative."""
    c, _ = client
    plan_id = c.post("/api/v1/pipelines", json=GRAPH).json()["plan_id"]
    response = c.post(f"/api/v1/pipelines/{plan_id}/execute")
    assert response.status_code == 409
    assert "approval_required" in response.text


def test_a_graph_needs_every_scope_its_stages_need(client, monkeypatch):
    """One approval must not silently widen authority.

    A caller holding only `instances:operate` can quote a train stage. Adding a
    serve stage means the same single approval would cover `inference:write`
    too — so the create route demands it up front rather than discovering the
    gap when stage 3 runs.
    """
    import routes.action_plans as ap
    from routes import _deps

    narrow = {
        "email": "demo@xcelsior.ca", "user_id": "u-narrow", "role": "user",
        "auth_type": "client_credentials", "grant_type": "client_credentials",
        "client_id": "narrow", "customer_id": "tenant-narrow",
        "scopes": ["instances:write", "instances:operate"],   # no inference:write
    }
    monkeypatch.setattr(_deps, "_require_auth", lambda request: dict(narrow))
    monkeypatch.setattr(ap, "_require_auth", lambda request: dict(narrow))
    monkeypatch.setattr(ap, "_effective_billing_customer_id", lambda user: "tenant-narrow")

    c, _ = client
    response = c.post("/api/v1/pipelines", json=GRAPH)
    assert response.status_code == 403, (
        f"a caller without inference:write quoted a pipeline containing a serve "
        f"stage ({response.status_code}) — one approval would have widened its "
        "own authority"
    )


def test_a_stage_naming_an_unknown_action_is_refused(client):
    """Finding out at stage 3 is finding out after two stages were paid for."""
    c, _ = client
    graph = json.loads(json.dumps(GRAPH))
    graph["stages"][1]["action_type"] = "make_coffee"
    response = c.post("/api/v1/pipelines", json=graph)
    assert response.status_code == 422
    assert "unknown_action_type" in response.text


def test_a_foreign_pipeline_is_not_found_rather_than_forbidden(client):
    """404, so plan ids cannot be enumerated."""
    c, _ = client
    plan_id = c.post("/api/v1/pipelines", json=GRAPH).json()["plan_id"]
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE action_plans SET tenant_id = 'someone-else' WHERE plan_id = %s",
            (plan_id,),
        )
    assert c.get(f"/api/v1/pipelines/{plan_id}").status_code == 404
    assert c.post(f"/api/v1/pipelines/{plan_id}/execute").status_code == 404


def test_the_executor_reports_unwired_stages_rather_than_faking_success(client):
    """B4 wires no stage bodies, and says so.

    A pipeline reporting three green stages having run nothing is the same
    failure as an unverified copy that looks like a backup — and this whole
    phase is about approvals meaning what they say.
    """
    c, _ = client
    plan_id = c.post("/api/v1/pipelines", json=GRAPH).json()["plan_id"]
    _approve(plan_id)
    body = c.post(f"/api/v1/pipelines/{plan_id}/execute").json()

    assert body["failed"] is True
    with pg_transaction() as conn:
        codes = [
            r[0] for r in conn.execute(
                "SELECT failure_code FROM pipeline_stages WHERE plan_id = %s ORDER BY stage_index",
                (plan_id,),
            ).fetchall()
        ]
    assert codes[0] == "stage_executor_not_wired"
