"""A person sets the envelope; a credential launches inside it without asking.

The envelope is the standing policy (`mcp_client_policies`) on one OAuth
client. It is what lets an unattended caller — `terraform apply` in CI — launch
at all: its plans approve themselves inside the envelope and wait for a person
outside it. Every rule below is one a human relies on when they hand a token to
a pipeline and walk away:

- only a person in the dashboard can set or raise an envelope;
- a launch inside it approves itself, one outside it waits for a person;
- the daily ceiling holds across launches, and holds at execute when two
  plans race for the last of it;
- a plan a person approved is not refused by the ceiling set for unattended
  use;
- a credential under an envelope cannot step around it by launching directly;
- `terraform plan`'s quote persists nothing.

Real credentials through the real auth path, as in
`tests/test_agents_launch_only_through_plans.py`.
"""

from __future__ import annotations

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

from api import app
from billing import BillingEngine
from control_plane.launch import spend_counters
from control_plane.launch.service import (
    PlanConflict,
    Principal,
    approve,
    execute,
    preview,
)
from db import UserStore, _get_pg_pool, auth_connection
from oauth_service import issue_agent_api_key

pytestmark = pytest.mark.needs_db

client = TestClient(app)

LAUNCH = {"name": "envelope", "gpu_model": "RTX 4090", "num_gpus": 1, "interactive": True}
SCOPES = ["instances:read", "instances:write", "instances:operate"]


@pytest.fixture(autouse=True)
def _memory_counters(monkeypatch):
    """Counters in process memory, so windows start empty and do not leak."""
    monkeypatch.setenv("MCP_RATE_LIMIT_BACKEND", "memory")
    spend_counters.reset_for_tests()
    yield
    spend_counters.reset_for_tests()


@pytest.fixture
def account():
    email = f"envelope-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Envelope"},
    )
    assert reg.status_code == 200, reg.text[:300]
    user = dict(UserStore.get_user(email))
    tenant = str(user.get("customer_id") or user["user_id"])
    client_id = f"envelope-{uuid.uuid4().hex[:8]}"
    with auth_connection() as conn:
        conn.execute(
            "INSERT INTO oauth_clients (client_id, client_name, client_type, created_at, "
            "updated_at, workspace_customer_id, created_by_user_id) "
            "VALUES (%s, 'Terraform', 'confidential', 0, 0, %s, %s)",
            (client_id, tenant, user["user_id"]),
        )
    key = issue_agent_api_key(user=user, client_id=client_id, scopes=SCOPES, name="terraform")
    made: dict = {
        "user": user,
        "tenant": tenant,
        "client_id": client_id,
        "browser": {"Authorization": f"Bearer {reg.json()['access_token']}"},
        "key": {"Authorization": f"Bearer {key['access_token']}"},
        "plans": [],
        "jobs": [],
    }
    yield made
    with _get_pg_pool().connection() as conn:
        for job_id in made["jobs"]:
            conn.execute("UPDATE jobs SET wallet_hold_id=NULL WHERE job_id=%s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM action_plans WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM mcp_client_policies WHERE tenant_id = %s", (tenant,))
        for table in ("wallet_holds", "wallet_transactions", "wallets"):
            conn.execute(f"DELETE FROM {table} WHERE customer_id = %s", (tenant,))
        conn.commit()


def _estimate(account) -> int:
    r = client.post("/api/v1/launch-plans/quote", json=LAUNCH, headers=account["key"])
    assert r.status_code == 200, r.text[:400]
    return int(r.json()["estimate"]["estimate_micros"])


def _set_envelope(account, **ceilings):
    return client.put(
        f"/api/v1/spend-envelopes/{account['client_id']}",
        json=ceilings,
        headers=account["browser"],
    )


def _plan_count(tenant: str) -> int:
    with _get_pg_pool().connection() as conn:
        return conn.execute(
            "SELECT count(*) FROM action_plans WHERE tenant_id = %s", (tenant,)
        ).fetchone()[0]


# ── Who may set it ───────────────────────────────────────────────────


def test_a_person_in_the_dashboard_sets_an_envelope(account):
    r = _set_envelope(account, daily_spend_max_micros=50_000_000, max_gpus_per_launch=2)
    assert r.status_code == 200, r.text[:400]
    envelope = r.json()["envelope"]
    assert envelope["daily_spend_max_micros"] == 50_000_000
    assert envelope["max_gpus_per_launch"] == 2
    assert envelope["auto_approve"] is True

    again = _set_envelope(account, daily_spend_max_micros=60_000_000)
    assert again.json()["envelope"]["version"] == envelope["version"] + 1
    # Replaced whole: a ceiling left out of the second call is gone.
    assert again.json()["envelope"]["max_gpus_per_launch"] is None


def test_the_credential_cannot_set_its_own_envelope(account):
    r = client.put(
        f"/api/v1/spend-envelopes/{account['client_id']}",
        json={"daily_spend_max_micros": 10**12},
        headers=account["key"],
    )
    assert r.status_code == 403, r.text[:300]
    assert r.json().get("code") == "human_required"


def test_an_envelope_must_bound_the_day(account):
    r = _set_envelope(account, per_action_max_micros=1_000_000)
    assert r.status_code == 422, r.text[:300]


def test_an_envelope_cannot_be_put_on_someone_elses_client(account):
    stranger = f"stranger-{uuid.uuid4().hex[:8]}"
    with auth_connection() as conn:
        conn.execute(
            "INSERT INTO oauth_clients (client_id, client_name, client_type, created_at, "
            "updated_at, workspace_customer_id) VALUES (%s, 'Not yours', 'confidential', 0, 0, %s)",
            (stranger, f"cust-{uuid.uuid4().hex[:8]}"),
        )
    r = client.put(
        f"/api/v1/spend-envelopes/{stranger}",
        json={"daily_spend_max_micros": 1_000_000},
        headers=account["browser"],
    )
    assert r.status_code == 404, r.text[:300]


# ── What it approves ─────────────────────────────────────────────────


def test_without_an_envelope_every_launch_waits_for_a_person(account):
    plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=account["key"]).json()
    assert plan["approval_mode"] == "human", plan
    r = client.post(f"/api/v1/launch-plans/{plan['plan_id']}/approve", headers=account["key"])
    assert r.status_code == 409 and r.json()["code"] == "human_approval_required"


def test_a_launch_inside_the_envelope_approves_itself(account):
    estimate = _estimate(account)
    assert _set_envelope(account, daily_spend_max_micros=estimate * 3).status_code == 200

    plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=account["key"]).json()
    assert plan["approval_mode"] == "standing_policy", plan
    assert plan["approval_url"] is None
    r = client.post(f"/api/v1/launch-plans/{plan['plan_id']}/approve", headers=account["key"])
    assert r.status_code == 200, r.text[:300]
    assert r.json()["plan"]["status"] == "approved"
    assert r.json()["plan"]["approval_method"] == "standing_policy"


def test_a_launch_outside_the_envelope_waits_for_a_person(account):
    estimate = _estimate(account)
    _set_envelope(account, daily_spend_max_micros=estimate * 10, per_action_max_micros=estimate - 1)
    plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=account["key"]).json()
    assert plan["approval_mode"] == "human", plan
    assert plan["approval_url"].endswith(plan["plan_id"])
    assert any("per-action" in reason for reason in plan["policy"]["reasons"])


def test_a_launch_that_would_cross_the_daily_ceiling_waits_for_a_person(account):
    estimate = _estimate(account)
    _set_envelope(account, daily_spend_max_micros=estimate * 2)
    # Two launches' worth already reserved today by this credential.
    spend_counters.reserve(
        plan_id=f"earlier-{uuid.uuid4().hex}",
        client_id=account["client_id"],
        tenant_id=account["tenant"],
        amount_micros=estimate * 2,
        hourly_limit_micros=None,
        daily_limit_micros=estimate * 2,
    )
    plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=account["key"]).json()
    assert plan["approval_mode"] == "human", plan
    assert any("daily ceiling" in reason for reason in plan["policy"]["reasons"])


def test_the_credential_reads_its_envelope_and_what_is_left(account):
    _set_envelope(account, daily_spend_max_micros=9_000_000)
    r = client.get("/api/v1/spend-envelope", headers=account["key"])
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["client_id"] == account["client_id"]
    assert body["envelope"]["daily_spend_max_micros"] == 9_000_000
    assert body["usage"] == {"available": True, "hourly_micros": 0, "daily_micros": 0}


def test_the_credential_can_remove_its_own_envelope(account):
    estimate = _estimate(account)
    _set_envelope(account, daily_spend_max_micros=estimate * 3)
    r = client.delete(f"/api/v1/spend-envelopes/{account['client_id']}", headers=account["key"])
    assert r.status_code == 200 and r.json()["removed"] is True
    plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=account["key"]).json()
    assert plan["approval_mode"] == "human", plan


# ── It cannot be stepped around ──────────────────────────────────────


def test_a_client_credentials_token_under_an_envelope_cannot_launch_directly(account, monkeypatch):
    """The token that could POST `/instance` freely is held to plans once enveloped."""
    import routes.instances as instances

    machine = {
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "client_id": account["client_id"],
        "user_id": f"client:{account['client_id']}",
        "email": "",
        "role": "machine",
        "scopes": SCOPES,
        "customer_id": account["tenant"],
        "tenant_id": account["tenant"],
        "team_id": None,
    }
    monkeypatch.setattr(instances, "_require_auth", lambda _request: machine)
    monkeypatch.setattr(instances, "_require_team_instance_write", lambda _user: None)

    _set_envelope(account, daily_spend_max_micros=10_000_000)
    r = client.post("/instance", json=LAUNCH)
    assert r.status_code == 409, r.text[:300]
    assert r.json()["code"] == "launch_plan_required"


def test_an_agent_key_cannot_create_a_serverless_endpoint_directly(account, monkeypatch):
    monkeypatch.setenv("XCELSIOR_SERVERLESS_ENABLED", "true")
    monkeypatch.delenv("XCELSIOR_SERVERLESS_ALLOWLIST", raising=False)
    key = issue_agent_api_key(
        user=account["user"],
        client_id=account["client_id"],
        scopes=["inference:read", "inference:write"],
        name="serverless",
    )["access_token"]
    r = client.post(
        "/api/v2/serverless/endpoints",
        json={"name": "direct", "model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert r.status_code == 409, r.text[:300]
    assert r.json()["code"] == "launch_plan_required"
    assert "/api/v1/serverless/endpoint-plans" in r.json()["detail"]


# ── terraform plan asks without leaving anything behind ──────────────


def test_a_quote_answers_like_a_preview_and_persists_nothing(account):
    estimate = _estimate(account)
    _set_envelope(account, daily_spend_max_micros=estimate * 3)
    before = _plan_count(account["tenant"])
    r = client.post("/api/v1/launch-plans/quote", json=LAUNCH, headers=account["key"])
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["plan_id"] is None and body["persisted"] is False
    assert body["approval_mode"] == "standing_policy"
    assert _plan_count(account["tenant"]) == before

    plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=account["key"]).json()
    assert plan["approval_mode"] == body["approval_mode"]
    assert plan["estimate"]["estimate_micros"] == body["estimate"]["estimate_micros"]
    assert plan["spec_hash"] == body["spec_hash"]


# ── Execute holds the line, and only for unattended plans ────────────


def _funded_principal(account) -> Principal:
    BillingEngine().deposit(account["tenant"], 5000.0, description="envelope test seed")
    return Principal(
        principal_id=account["tenant"],
        tenant_id=account["tenant"],
        client_id=account["client_id"],
    )


def test_two_plans_racing_for_the_last_of_the_day_cannot_both_run(account):
    principal = _funded_principal(account)
    estimate = _estimate(account)
    _set_envelope(account, daily_spend_max_micros=int(estimate * 1.5))
    first = preview(dict(LAUNCH), principal=principal)
    assert first["approval_mode"] == "standing_policy", first
    approve(first["plan_id"], principal=principal)
    # Another launch takes the window between this plan's approval and its run.
    spend_counters.reserve(
        plan_id=f"racer-{uuid.uuid4().hex}",
        client_id=account["client_id"],
        tenant_id=account["tenant"],
        amount_micros=estimate,
        hourly_limit_micros=None,
        daily_limit_micros=int(estimate * 1.5),
    )
    with pytest.raises(PlanConflict) as refused:
        execute(first["plan_id"], principal=principal)
    assert refused.value.code == "daily_spend_limit_exceeded"


def test_a_plan_a_person_approved_is_not_held_to_the_unattended_ceiling(account):
    principal = _funded_principal(account)
    _set_envelope(account, daily_spend_max_micros=1)
    plan = preview(dict(LAUNCH), principal=principal)
    assert plan["approval_mode"] == "human", plan
    approve(plan["plan_id"], principal=principal, is_human=True)
    result = execute(plan["plan_id"], principal=principal)
    account["jobs"].append(result["job"]["job_id"])
    assert result["ok"] is True, result
