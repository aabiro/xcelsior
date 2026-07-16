"""Smoke coverage for routes/auth.py and routes/mfa.py (UNTESTED_ENDPOINTS.md)."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

OK_OR_HANDLED = {200, 302, 307, 400, 401, 403, 404, 410, 422, 503}


@pytest.fixture(scope="module")
def user_headers():
    email = f"authcov-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Auth Cov"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200
    return email, {"Authorization": f"Bearer {login.json()['access_token']}"}


# ── Auth routes (routes/auth.py) ────────────────────────────────────────


def test_auth_list_sessions(user_headers):
    _, headers = user_headers
    r = client.get("/api/auth/sessions", headers=headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True
    assert isinstance(r.json().get("sessions"), list)


def test_auth_revoke_session_unknown_prefix(user_headers):
    _, headers = user_headers
    r = client.delete("/api/auth/sessions/deadbeef", headers=headers)
    assert r.status_code == 404


def test_auth_logout(user_headers):
    _, headers = user_headers
    r = client.post("/api/auth/logout", headers=headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_auth_resend_verification(user_headers):
    email, _ = user_headers
    r = client.post("/api/auth/resend-verification", json={"email": email})
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_auth_oauth_callback_missing_params():
    r = client.get(
        "/api/auth/oauth/google/callback",
        follow_redirects=False,
    )
    assert r.status_code in (302, 307)
    assert "dashboard" in (r.headers.get("location") or "")


def test_auth_oauth_callback_unsupported_provider():
    r = client.get(
        "/api/auth/oauth/not-a-provider/callback",
        follow_redirects=False,
    )
    assert r.status_code == 400


def test_user_preferences_get_and_put(user_headers):
    _, headers = user_headers
    r = client.get("/api/users/me/preferences", headers=headers)
    assert r.status_code == 200
    assert r.json().get("ok") is True

    r2 = client.put(
        "/api/users/me/preferences",
        headers=headers,
        json={"notifications": True, "preferences": {"ai_panel_open": True}},
    )
    assert r2.status_code == 200
    assert r2.json().get("ok") is True


def test_oauth_device_authorize(user_headers):
    _, headers = user_headers
    created = client.post(
        "/api/oauth/clients",
        headers=headers,
        json={
            "client_name": "Device Cov",
            "client_type": "confidential",
            "redirect_uris": [],
            "grant_types": ["urn:ietf:params:oauth:grant-type:device_code"],
            "scopes": ["profile"],
        },
    )
    assert created.status_code == 200
    oauth_client = created.json()["client"]
    original_secret = oauth_client["client_secret"]
    assert oauth_client["client_secret_preview"] == f"{original_secret[:4]}...{original_secret[-4:]}"
    r = client.post(
        "/oauth/device/authorize",
        json={
            "client_id": oauth_client["client_id"],
            "client_secret": oauth_client["client_secret"],
            "scope": "profile",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("device_code") or body.get("user_code")


def test_oauth_rotate_secret(user_headers):
    _, headers = user_headers
    created = client.post(
        "/api/oauth/clients",
        headers=headers,
        json={
            "client_name": "Rotate Cov",
            "client_type": "confidential",
            "redirect_uris": [],
            "grant_types": ["client_credentials"],
            "scopes": ["profile"],
        },
    )
    assert created.status_code == 200
    oauth_client = created.json()["client"]
    r = client.post(
        f"/api/oauth/clients/{oauth_client['client_id']}/rotate-secret",
        headers=headers,
    )
    assert r.status_code == 200
    rotated = r.json()
    assert rotated.get("client_secret")
    assert rotated["client_secret_preview"] == (
        f"{rotated['client_secret'][:4]}...{rotated['client_secret'][-4:]}"
    )

    listed = client.get("/api/oauth/clients", headers=headers)
    assert listed.status_code == 200
    listed_client = next(
        item for item in listed.json()["clients"] if item["client_id"] == oauth_client["client_id"]
    )
    assert listed_client["client_secret_preview"] == rotated["client_secret_preview"]
    assert "client_secret" not in listed_client


# ── MFA routes (routes/mfa.py) — reachable, no unhandled 500 ────────────


_MFA_ROUTES = [
    ("DELETE", "/api/auth/mfa/all", None),
    ("DELETE", "/api/auth/mfa/sms", None),
    ("DELETE", "/api/auth/mfa/totp", None),
    ("POST", "/api/auth/mfa/backup-codes/regenerate", {}),
    ("POST", "/api/auth/mfa/passkey/authenticate-complete", {}),
    ("POST", "/api/auth/mfa/passkey/authenticate-options", {}),
    ("POST", "/api/auth/mfa/passkey/delete", {}),
    ("POST", "/api/auth/mfa/passkey/register-complete", {}),
    ("POST", "/api/auth/mfa/passkey/register-options", {}),
    ("POST", "/api/auth/mfa/sms/send", {}),
    ("POST", "/api/auth/mfa/sms/setup", {}),
    ("POST", "/api/auth/mfa/sms/verify", {}),
    ("POST", "/api/auth/mfa/totp/setup", {}),
    ("POST", "/api/auth/mfa/totp/verify", {}),
    ("POST", "/api/auth/mfa/verify", {}),
]


@pytest.mark.parametrize("method,path,body", _MFA_ROUTES)
def test_mfa_route_unauthenticated_not_500(method, path, body):
    if method == "DELETE":
        r = client.delete(path)
    else:
        r = client.post(path, json=body or {})
    assert r.status_code in OK_OR_HANDLED
    assert r.status_code != 500


@pytest.mark.parametrize("method,path,body", _MFA_ROUTES)
def test_mfa_route_authenticated_handled(method, path, body, user_headers):
    _, headers = user_headers
    if method == "DELETE":
        r = client.delete(path, headers=headers)
    else:
        r = client.post(path, json=body or {}, headers=headers)
    assert r.status_code in OK_OR_HANDLED
    assert r.status_code != 500
