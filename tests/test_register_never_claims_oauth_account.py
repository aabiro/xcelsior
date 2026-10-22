"""Registering with an OAuth-only account's address never claims that account.

`POST /api/auth/register` used to treat an existing account with no password
as an invitation to "link" one: it stored the caller's password, set
`email_verified=1` ("already verified via OAuth") and answered with session
tokens. The provider verified the *original* owner, not the caller, so anyone
who knew the address of a Google/GitHub user could sign in as them with a
password of their choosing.

Registering now answers 409 `oauth_only`, the same guidance the login door
gives, and changes nothing. A password is added through the emailed reset link,
which proves the requester holds the address.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

import api as _api_mod
import routes._deps as _deps_mod
import routes.auth as _auth_mod
from api import app
from db import UserStore


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    monkeypatch.setattr(_api_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_deps_mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(_auth_mod, "_USE_PERSISTENT_AUTH", True)
    _deps_mod._AUTH_RATE_BUCKETS.clear()


@pytest.fixture
def oauth_only_user():
    email = f"oauth-only-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    UserStore.create_user(
        {
            "email": email,
            "user_id": f"user-{uuid.uuid4().hex[:12]}",
            "name": "OAuth Owner",
            "customer_id": f"cust-{uuid.uuid4().hex[:8]}",
            "oauth_provider": "google",
        }
    )
    yield email
    UserStore.delete_user(email)


def test_register_does_not_set_a_password_or_sign_in(oauth_only_user):
    client = TestClient(app)
    resp = client.post(
        "/api/auth/register",
        json={"email": oauth_only_user, "password": "AttackerChosen123!", "name": "x"},
    )

    assert resp.status_code == 409
    body = resp.json()
    assert body["error"]["code"] == "oauth_only"
    assert "access_token" not in body and "user" not in body
    assert not any(c.lower().startswith("xcelsior") for c in resp.cookies.keys())

    stored = UserStore.get_user(oauth_only_user)
    assert not stored.get("password_hash")
    assert not stored.get("email_verified")


def test_the_attacker_password_does_not_log_in(oauth_only_user):
    client = TestClient(app)
    client.post(
        "/api/auth/register",
        json={"email": oauth_only_user, "password": "AttackerChosen123!", "name": "x"},
    )
    login = client.post(
        "/api/auth/login",
        json={"email": oauth_only_user, "password": "AttackerChosen123!"},
    )
    assert login.status_code == 403
    assert login.json()["error"]["code"] == "oauth_only"
