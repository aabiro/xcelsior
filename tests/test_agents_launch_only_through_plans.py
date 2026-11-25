"""An agent launches a GPU only through a launch plan the user approved.

`run_training_job` and `schedule_under_budget` POSTed `/instance` on
confirm:true, so an agent could launch and bill a GPU the user never saw a plan
for, while both descriptions said approval applied. The MCP tools now prepare
and execute plans; this asserts the API refuses the shortcut too, so no tool,
present or future, can skip the approval.

Real credentials through the real auth path: an MCP-bound connector token and
an agent key are refused; a browser session and the wizard's device-flow token
(delegated, but a person at a terminal launching their own instance) are not.
"""

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

from api import app
from db import UserStore, auth_connection
from oauth_service import (
    MCP_RESOURCE_AUDIENCE,
    _issue_cached_access_token,
    issue_agent_api_key,
)

client = TestClient(app)

LAUNCH = {"name": "plan-gate", "gpu_model": "RTX 4090", "num_gpus": 1, "interactive": True}
SCOPES = ["instances:read", "instances:write"]


@pytest.fixture(scope="module")
def account():
    email = f"plan-gate-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Plan Gate"},
    )
    assert reg.status_code == 200, reg.text[:300]
    body = reg.json()
    connector = f"plan-gate-{uuid.uuid4().hex[:8]}"
    with auth_connection() as conn:
        conn.execute(
            "INSERT INTO oauth_clients (client_id, client_name, client_type, created_at, updated_at) "
            "VALUES (%s, 'Plan gate connector', 'public', 0, 0)",
            (connector,),
        )
    return {
        "user": dict(UserStore.get_user(email)),
        "browser": {"Authorization": f"Bearer {body['access_token']}"},
        "connector": connector,
    }


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _refused_for_a_plan(r) -> None:
    assert r.status_code == 409, r.text[:400]
    problem = r.json()
    assert problem.get("code") == "launch_plan_required", problem
    assert "/api/v1/launch-plans" in problem.get("detail", "")


def test_an_mcp_connector_token_is_sent_to_launch_plans(account):
    token = _issue_cached_access_token(
        account["user"],
        client_id=account["connector"],
        scopes=SCOPES,
        session_token=f"plan-gate-{uuid.uuid4().hex}",
        session_type="browser",
        audience=MCP_RESOURCE_AUDIENCE,
    )["access_token"]
    _refused_for_a_plan(client.post("/instance", json=LAUNCH, headers=_bearer(token)))


def test_an_agent_key_is_sent_to_launch_plans(account):
    key = issue_agent_api_key(
        user=account["user"],
        client_id=account["connector"],
        scopes=SCOPES,
        name="plan gate",
    )["access_token"]
    _refused_for_a_plan(client.post("/instance", json=LAUNCH, headers=_bearer(key)))


def test_the_same_connector_token_can_still_prepare_a_plan(account):
    token = _issue_cached_access_token(
        account["user"],
        client_id=account["connector"],
        scopes=SCOPES,
        session_token=f"plan-gate-{uuid.uuid4().hex}",
        session_type="browser",
        audience=MCP_RESOURCE_AUDIENCE,
    )["access_token"]
    r = client.post("/api/v1/launch-plans", json=LAUNCH, headers=_bearer(token))
    assert r.status_code == 200, r.text[:400]
    assert r.json().get("plan_id")


def _past_the_gate(r) -> None:
    # Whatever the launch itself answers (queued, unfunded, no capacity), it
    # is not the agent refusal.
    assert r.status_code < 500, r.text[:400]
    assert "launch_plan_required" not in r.text


def test_a_person_in_the_dashboard_launches_directly(account):
    _past_the_gate(client.post("/instance", json=LAUNCH, headers=account["browser"]))


def test_the_wizard_device_token_launches_directly(account):
    """A delegated token without the MCP audience is a person at a terminal."""
    token = _issue_cached_access_token(
        account["user"],
        client_id=account["connector"],
        scopes=SCOPES,
        session_token=f"plan-gate-{uuid.uuid4().hex}",
        session_type="device",
    )["access_token"]
    _past_the_gate(client.post("/instance", json=LAUNCH, headers=_bearer(token)))


# ── An agent cannot approve its own plan ──────────────────────────────
#
# The gate above is only as good as the approval. `_is_human` counted every
# credential but client_credentials as a person, so an agent key or an MCP
# token could prepare a plan, approve it itself and execute it.

#: Every scope approval asks for, so only the human check can refuse.
APPROVER_SCOPES = [*SCOPES, "instances:operate"]


def _agent_credentials(account) -> dict[str, str]:
    mcp = _issue_cached_access_token(
        account["user"],
        client_id=account["connector"],
        scopes=APPROVER_SCOPES,
        session_token=f"plan-gate-{uuid.uuid4().hex}",
        session_type="browser",
        audience=MCP_RESOURCE_AUDIENCE,
    )["access_token"]
    key = issue_agent_api_key(
        user=account["user"],
        client_id=account["connector"],
        scopes=APPROVER_SCOPES,
        name="self-approve",
    )["access_token"]
    return {"mcp connector token": mcp, "agent key": key}


def test_an_agent_cannot_approve_its_own_plan(account):
    for label, token in _agent_credentials(account).items():
        plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=_bearer(token))
        assert plan.status_code == 200, (label, plan.text[:300])
        assert plan.json().get("approval_mode") == "human", (label, plan.json())
        plan_id = plan.json()["plan_id"]

        approved = client.post(
            f"/api/v1/launch-plans/{plan_id}/approve",
            json={"confirmation": "approve"},
            headers=_bearer(token),
        )
        assert approved.status_code == 409, (label, approved.text[:300])
        assert approved.json().get("code") == "human_approval_required", (label, approved.json())

        ran = client.post(
            f"/api/v1/launch-plans/{plan_id}/execute",
            json={"confirm": True},
            headers=_bearer(token),
        )
        assert ran.status_code >= 400, (label, ran.text[:300])
        assert "launch_plan_required" not in ran.text


def test_the_person_in_the_dashboard_approves_an_agent_plan(account):
    token = _agent_credentials(account)["mcp connector token"]
    plan = client.post("/api/v1/launch-plans", json=LAUNCH, headers=_bearer(token)).json()
    approved = client.post(
        f"/api/v1/launch-plans/{plan['plan_id']}/approve",
        json={"confirmation": "approve"},
        headers=account["browser"],
    )
    assert approved.status_code == 200, approved.text[:300]
    assert approved.json()["plan"]["status"] == "approved"
