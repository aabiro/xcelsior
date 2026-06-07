"""Smoke coverage for routes/teams.py (UNTESTED_ENDPOINTS.md)."""

import os
import secrets
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app
from db import UserStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as _deps_mod
    import routes.auth as _auth_mod

    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_deps_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_auth_mod, "_USE_PERSISTENT_AUTH", True)


def _login(email: str, password: str = "StrongPass123!") -> dict:
    client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": "Team Cov"},
    )
    UserStore.update_user(email, {"email_verified": 1})
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture
def team_ctx(persistent_auth):
    leader_email = f"teamlead-{uuid.uuid4().hex[:10]}@xcelsior.ca".lower()
    leader_headers = _login(leader_email)
    cr = client.post(
        "/api/teams",
        json={"name": f"Cov Team {uuid.uuid4().hex[:6]}", "plan": "free"},
        headers=leader_headers,
    )
    assert cr.status_code == 200
    team_id = cr.json()["team_id"]
    return {"leader_email": leader_email, "leader_headers": leader_headers, "team_id": team_id}


def test_teams_invite_pending_user_missing(team_ctx):
    token = secrets.token_urlsafe(32)
    pending_email = f"pending-{uuid.uuid4().hex[:10]}@xcelsior.ca".lower()
    UserStore.create_team_invite(
        {
            "token": token,
            "team_id": team_ctx["team_id"],
            "email": pending_email,
            "role": "member",
            "invited_by": team_ctx["leader_email"],
            "created_at": time.time(),
            "expires_at": time.time() + 86400,
        }
    )
    r = client.get(f"/api/teams/invite/{token}")
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("pending") is True
    assert r.json().get("email") == pending_email.lower()


def test_teams_invite_accept_existing_user(team_ctx):
    invitee_email = f"invitee-{uuid.uuid4().hex[:10]}@xcelsior.ca".lower()
    invitee_headers = _login(invitee_email)
    token = secrets.token_urlsafe(32)
    UserStore.create_team_invite(
        {
            "token": token,
            "team_id": team_ctx["team_id"],
            "email": invitee_email,
            "role": "viewer",
            "invited_by": team_ctx["leader_email"],
            "created_at": time.time(),
            "expires_at": time.time() + 86400,
        }
    )
    r = client.get(f"/api/teams/invite/{token}")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("accepted") is True or body.get("pending") is not True

    r3 = client.get(
        f"/api/teams/{team_ctx['team_id']}",
        headers=team_ctx["leader_headers"],
    )
    assert r3.status_code == 200
    emails = {m["email"].lower() for m in r3.json().get("members", [])}
    assert invitee_email in emails


def test_teams_invite_accept_authenticated(team_ctx):
    invitee_email = f"accept-{uuid.uuid4().hex[:10]}@xcelsior.ca".lower()
    invitee_headers = _login(invitee_email)
    token = secrets.token_urlsafe(32)
    UserStore.create_team_invite(
        {
            "token": token,
            "team_id": team_ctx["team_id"],
            "email": invitee_email,
            "role": "member",
            "invited_by": team_ctx["leader_email"],
            "created_at": time.time(),
            "expires_at": time.time() + 86400,
        }
    )
    r = client.post(
        f"/api/teams/invite/{token}/accept",
        headers=invitee_headers,
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert r.json().get("accepted") is True


def test_teams_switch_active_team(team_ctx):
    second_email = f"member2-{uuid.uuid4().hex[:10]}@xcelsior.ca".lower()
    second_headers = _login(second_email)
    team_b = client.post(
        "/api/teams",
        json={"name": f"Second Team {uuid.uuid4().hex[:6]}", "plan": "free"},
        headers=second_headers,
    )
    assert team_b.status_code == 200
    team_b_id = team_b.json()["team_id"]

    me = client.get("/api/teams/me", headers=second_headers)
    assert me.status_code == 200
    assert me.json().get("active_team_id") == team_b_id

    switch = client.patch(
        "/api/teams/active",
        json={"team_id": team_ctx["team_id"]},
        headers=second_headers,
    )
    assert switch.status_code == 403

    leader_invite = client.post(
        f"/api/teams/{team_ctx['team_id']}/members",
        json={"email": second_email, "role": "member"},
        headers=team_ctx["leader_headers"],
    )
    assert leader_invite.status_code == 200

    switch_ok = client.patch(
        "/api/teams/active",
        json={"team_id": team_ctx["team_id"]},
        headers=second_headers,
    )
    assert switch_ok.status_code == 200
    assert switch_ok.json().get("active_team_id") == team_ctx["team_id"]

    personal = client.patch(
        "/api/teams/active",
        json={"team_id": None},
        headers=second_headers,
    )
    assert personal.status_code == 200
    assert personal.json().get("active_team_id") is None

    user_row = UserStore.get_user(second_email)
    assert user_row.get("team_id") in (None, "")


def test_teams_invite_accept_wrong_email_forbidden(team_ctx):
    token = secrets.token_urlsafe(32)
    UserStore.create_team_invite(
        {
            "token": token,
            "team_id": team_ctx["team_id"],
            "email": f"target-{uuid.uuid4().hex[:8]}@xcelsior.ca".lower(),
            "role": "member",
            "invited_by": team_ctx["leader_email"],
            "created_at": time.time(),
            "expires_at": time.time() + 86400,
        }
    )
    other_headers = _login(f"other-{uuid.uuid4().hex[:10]}@xcelsior.ca".lower())
    r = client.post(
        f"/api/teams/invite/{token}/accept",
        headers=other_headers,
    )
    assert r.status_code == 403