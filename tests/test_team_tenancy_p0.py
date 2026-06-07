"""P0 team tenancy — shared billing wallet and job ownership."""

import os
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

from api import app
from db import UserStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    """Teams and billing use UserStore (Postgres); auth must match."""
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _register_and_login(label: str) -> dict:
    email = f"team-p0-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": f"Team P0 {label}"},
    )
    assert reg.status_code == 200, reg.text[:200]
    reg_body = reg.json()
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text[:200]
    body = login.json()
    user = reg_body.get("user") or body.get("user") or {}
    assert UserStore.get_user(email) is not None, (
        f"register did not persist {email}: reg={str(reg_body)[:200]}"
    )
    return {
        "email": email,
        "customer_id": user["customer_id"],
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


def _fund_wallet(customer_id: str, headers: dict) -> None:
    r = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0, "description": "Team P0 credits"},
        headers=headers,
    )
    assert r.status_code == 200, r.text[:200]


@pytest.fixture(scope="module")
def team_setup():
    admin = _register_and_login("admin")
    member = _register_and_login("member")

    team_resp = client.post(
        "/api/teams",
        json={"name": "P0 Team", "plan": "free"},
        headers=admin["headers"],
    )
    assert team_resp.status_code == 200, team_resp.text[:200]
    team_id = team_resp.json()["team_id"]
    team = UserStore.get_team(team_id)
    assert team is not None
    billing_customer_id = team["billing_customer_id"]
    assert billing_customer_id == admin["customer_id"]

    _fund_wallet(billing_customer_id, admin["headers"])

    assert UserStore.get_user(member["email"]) is not None

    add = client.post(
        f"/api/teams/{team_id}/members",
        json={"email": member["email"], "role": "member"},
        headers=admin["headers"],
    )
    assert add.status_code == 200, add.text[:200]
    add_body = add.json()
    assert not add_body.get("invited"), f"member not added directly: {add_body}"

    return {
        "admin": admin,
        "member": member,
        "team_id": team_id,
        "billing_customer_id": billing_customer_id,
    }


def test_team_create_sets_billing_customer_id(team_setup):
    assert team_setup["billing_customer_id"] == team_setup["admin"]["customer_id"]


def test_team_member_can_read_team_wallet(team_setup):
    r = client.get(
        f"/api/billing/wallet/{team_setup['billing_customer_id']}",
        headers=team_setup["member"]["headers"],
    )
    assert r.status_code == 200, r.text[:200]
    assert r.json()["wallet"]["balance_cad"] >= 50.0


def test_team_member_cannot_deposit_to_team_wallet(team_setup):
    r = client.post(
        f"/api/billing/wallet/{team_setup['billing_customer_id']}/deposit",
        json={"amount_cad": 10.0, "description": "member deposit"},
        headers=team_setup["member"]["headers"],
    )
    assert r.status_code == 403, r.text[:200]


def test_team_admin_can_deposit_to_team_wallet(team_setup):
    r = client.post(
        f"/api/billing/wallet/{team_setup['billing_customer_id']}/deposit",
        json={"amount_cad": 5.0, "description": "admin top-up"},
        headers=team_setup["admin"]["headers"],
    )
    assert r.status_code == 200, r.text[:200]


def test_team_member_job_uses_team_wallet(team_setup):
    r = client.post(
        "/instance",
        json={"name": "team-shared-job", "vram_needed_gb": 1},
        headers=team_setup["member"]["headers"],
    )
    assert r.status_code == 200, r.text[:200]
    job_id = r.json()["instance"]["job_id"]

    detail = client.get(f"/instance/{job_id}", headers=team_setup["admin"]["headers"])
    assert detail.status_code == 200, detail.text[:200]
    assert detail.json()["instance"]["owner"] == team_setup["billing_customer_id"]


def test_team_member_sees_peer_instances(team_setup):
    created = client.post(
        "/instance",
        json={"name": "peer-visible", "vram_needed_gb": 1},
        headers=team_setup["admin"]["headers"],
    )
    assert created.status_code == 200, created.text[:200]
    job_id = created.json()["instance"]["job_id"]

    listed = client.get("/instances", headers=team_setup["member"]["headers"])
    assert listed.status_code == 200, listed.text[:200]
    ids = {j["job_id"] for j in listed.json()["instances"]}
    assert job_id in ids


def test_non_member_cannot_read_team_wallet(team_setup):
    outsider = _register_and_login("outsider")
    r = client.get(
        f"/api/billing/wallet/{team_setup['billing_customer_id']}",
        headers=outsider["headers"],
    )
    assert r.status_code == 403, r.text[:200]