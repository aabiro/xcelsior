"""Phase 12 — serverless endpoint tenancy (owner/team scope, viewer 403, cross-account 404)."""

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app
from db import UserStore

client = TestClient(app)

_ENDPOINT_BODY = {
    "name": "tenancy-llama",
    "mode": "preset",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "gpu_type": "RTX 4090",
    "min_workers": 0,
    "max_workers": 1,
}


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _register(label: str) -> dict:
    email = f"sl-ten-{label}-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": label},
    )
    assert reg.status_code == 200, reg.text[:200]
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200, login.text[:200]
    body = login.json()
    user = reg.json().get("user") or body.get("user") or {}
    assert UserStore.get_user(email) is not None
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    deposit = client.post(
        f"/api/billing/wallet/{user['customer_id']}/deposit",
        json={"amount_cad": 25.0, "description": "serverless tenancy"},
        headers=headers,
    )
    assert deposit.status_code == 200, deposit.text[:200]
    return {"email": email, "headers": headers, "customer_id": user["customer_id"]}


@pytest.fixture(scope="module")
def tenancy_ctx():
    owner = _register("owner")
    outsider = _register("outsider")
    viewer = _register("viewer")

    team = client.post(
        "/api/teams",
        json={"name": "SL Tenancy", "plan": "free"},
        headers=owner["headers"],
    )
    assert team.status_code == 200, team.text[:200]
    team_id = team.json()["team_id"]
    billing_id = UserStore.get_team(team_id)["billing_customer_id"]

    add = client.post(
        f"/api/teams/{team_id}/members",
        json={"email": viewer["email"], "role": "viewer"},
        headers=owner["headers"],
    )
    assert add.status_code == 200, add.text[:200]

    created = client.post(
        "/api/v2/serverless/endpoints",
        headers=owner["headers"],
        json=_ENDPOINT_BODY,
    )
    assert created.status_code == 200, created.text[:300]
    ep_id = created.json()["endpoint"]["endpoint_id"]

    return {
        "owner": owner,
        "outsider": outsider,
        "viewer": viewer,
        "team_id": team_id,
        "billing_id": billing_id,
        "endpoint_id": ep_id,
    }


def test_cross_account_endpoint_get_returns_404(tenancy_ctx):
    r = client.get(
        f"/api/v2/serverless/endpoints/{tenancy_ctx['endpoint_id']}",
        headers=tenancy_ctx["outsider"]["headers"],
    )
    assert r.status_code == 404


def test_cross_account_jobs_list_returns_404(tenancy_ctx):
    r = client.get(
        f"/api/v2/serverless/endpoints/{tenancy_ctx['endpoint_id']}/jobs",
        headers=tenancy_ctx["outsider"]["headers"],
    )
    assert r.status_code == 404


def test_viewer_can_read_team_endpoint(tenancy_ctx):
    r = client.get(
        f"/api/v2/serverless/endpoints/{tenancy_ctx['endpoint_id']}",
        headers=tenancy_ctx["viewer"]["headers"],
    )
    assert r.status_code == 200, r.text[:200]


def test_viewer_cannot_delete_endpoint(tenancy_ctx):
    r = client.delete(
        f"/api/v2/serverless/endpoints/{tenancy_ctx['endpoint_id']}",
        headers=tenancy_ctx["viewer"]["headers"],
    )
    assert r.status_code == 403, r.text[:200]
    assert "viewer" in r.text.lower()


def test_viewer_cannot_create_api_key(tenancy_ctx):
    r = client.post(
        f"/api/v2/serverless/endpoints/{tenancy_ctx['endpoint_id']}/keys",
        headers=tenancy_ctx["viewer"]["headers"],
        json={"name": "blocked"},
    )
    assert r.status_code == 403, r.text[:200]
    assert "viewer" in r.text.lower()


def test_owner_can_create_and_revoke_key(tenancy_ctx):
    created = client.post(
        f"/api/v2/serverless/endpoints/{tenancy_ctx['endpoint_id']}/keys",
        headers=tenancy_ctx["owner"]["headers"],
        json={"name": "smoke-key"},
    )
    assert created.status_code == 200, created.text[:200]
    key_id = created.json()["key"]["key_id"]
    assert created.json()["key"].get("api_key")

    revoked = client.delete(
        f"/api/v2/serverless/endpoints/{tenancy_ctx['endpoint_id']}/keys/{key_id}",
        headers=tenancy_ctx["owner"]["headers"],
    )
    assert revoked.status_code == 200


def test_viewer_cannot_enqueue_job(tenancy_ctx):
    r = client.post(
        f"/v1/serverless/{tenancy_ctx['endpoint_id']}/run",
        headers=tenancy_ctx["viewer"]["headers"],
        json={"input": {"ping": True}},
    )
    assert r.status_code == 403, r.text[:200]
    assert "viewer" in r.text.lower()