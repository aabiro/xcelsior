"""Admin account actions: suspend (ban), restore, and delete through the
durable deletion workflow, from the admin users menu."""

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def member():
    email = f"acct-action-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Account Action"},
    )
    assert reg.status_code in (200, 201), reg.text
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    return {"email": email, "headers": {"Authorization": f"Bearer {login.json()['access_token']}"}}


def _listed(email: str) -> dict:
    users = client.get("/api/admin/users", headers=_admin_headers()).json()["users"]
    return next(u for u in users if u["email"] == email)


def test_suspend_then_restore_shows_in_the_admin_list(member):
    r = client.post(f"/api/admin/users/{member['email']}/suspend", headers=_admin_headers())
    assert r.status_code == 200, r.text
    assert _listed(member["email"])["is_suspended"] is True

    r = client.post(f"/api/admin/users/{member['email']}/unsuspend", headers=_admin_headers())
    assert r.status_code == 200, r.text
    assert _listed(member["email"])["is_suspended"] is False


def test_delete_suspends_and_starts_one_deletion(member):
    r = client.post(f"/api/admin/users/{member['email']}/delete", headers=_admin_headers())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_suspended"] is True
    assert body["request_id"]
    assert _listed(member["email"])["is_suspended"] is True

    # A double click or retry joins the same request instead of starting another.
    again = client.post(f"/api/admin/users/{member['email']}/delete", headers=_admin_headers())
    assert again.status_code == 200, again.text
    assert again.json()["request_id"] == body["request_id"]
    assert again.json()["already_existed"] is True


def test_an_admin_account_cannot_be_deleted_from_the_menu(member):
    r = client.post(f"/api/admin/users/{member['email']}/toggle-admin", headers=_admin_headers())
    assert r.status_code == 200, r.text
    r = client.post(f"/api/admin/users/{member['email']}/delete", headers=_admin_headers())
    assert r.status_code == 409
    assert "Revoke admin" in r.json().get("detail", r.text)


def test_a_regular_user_cannot_delete_anyone(member):
    other = f"acct-target-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    r = client.post(f"/api/admin/users/{other}/delete", headers=member["headers"])
    assert r.status_code in (401, 403)


def test_deleting_an_unknown_account_is_404():
    r = client.post(
        f"/api/admin/users/nobody-{uuid.uuid4().hex[:8]}@xcelsior.ca/delete",
        headers=_admin_headers(),
    )
    assert r.status_code == 404
