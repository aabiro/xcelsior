"""A new address is untrusted until its delivered, single-use link is consumed."""

import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from api import app
from db import UserStore, auth_connection


@pytest.fixture
def account(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "routes.auth._send_team_email", lambda *args, **kwargs: sent.append((args, kwargs))
    )
    client = TestClient(app)
    email = f"email-review-{uuid.uuid4().hex}@example.test"
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Email Review"},
    )
    assert response.status_code == 200, response.text
    user = UserStore.get_user(email)
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    sent.clear()
    yield client, user, headers, sent
    with auth_connection() as conn:
        conn.execute("DELETE FROM users WHERE user_id = %s", (user["user_id"],))


def request_change(account):
    client, user, headers, sent = account
    address = f"new-{uuid.uuid4().hex}@example.test"
    response = client.post(
        "/api/auth/me/email-change", json={"new_email": address}, headers=headers
    )
    assert response.status_code == 200, response.text
    assert sent[-1][0][0] == address
    assert sent[-1][1]["require_delivery"] is True
    token = re.search(r"token=([^\s]+)", sent[-1][1]["cta_url"]).group(1)
    assert UserStore.get_user(user["email"])["pending_email"] == address
    assert UserStore.get_user(address) is None
    return address, token


def test_new_address_is_verified_once_and_existing_identity_survives(account):
    client, user, headers, _ = account
    address, token = request_change(account)
    result = client.post("/api/auth/me/email-change/confirm", json={"token": token})
    assert result.status_code == 200, result.text
    updated = UserStore.get_user(address)
    assert updated["user_id"] == user["user_id"]
    assert updated["email_verified"]
    assert not updated["email_change_token"]
    assert client.get("/api/auth/me", headers=headers).json()["user"]["email"] == address
    assert (
        client.post("/api/auth/me/email-change/confirm", json={"token": token}).status_code == 400
    )


def test_expired_confirmation_leaves_the_original_account_unchanged(account):
    client, user, _, _ = account
    _, token = request_change(account)
    UserStore.update_user(user["email"], {"email_change_expires": time.time() - 1})
    result = client.post("/api/auth/me/email-change/confirm", json={"token": token})
    assert result.status_code == 400
    assert "expired" in result.text
    assert UserStore.get_user(user["email"])


def test_a_new_request_invalidates_the_old_link(account):
    client, user, _, _ = account
    _, old_token = request_change(account)
    address, new_token = request_change(account)
    assert (
        client.post("/api/auth/me/email-change/confirm", json={"token": old_token}).status_code
        == 400
    )
    assert (
        client.post("/api/auth/me/email-change/confirm", json={"token": new_token}).json()["email"]
        == address
    )


def test_smtp_failure_is_not_reported_as_a_sent_verification(account, monkeypatch):
    client, user, headers, _ = account

    def unavailable(*args, **kwargs):
        raise OSError("SMTP unavailable")

    monkeypatch.setattr("routes.auth._send_team_email", unavailable)
    result = client.post(
        "/api/auth/me/email-change", json={"new_email": "new@example.test"}, headers=headers
    )
    assert result.status_code == 503
    updated = UserStore.get_user(user["email"])
    assert not updated["pending_email"]
    assert not updated["email_change_token"]


def test_two_simultaneous_confirmations_cannot_both_succeed(account):
    _, _, _, _ = account
    address, token = request_change(account)

    def confirm():
        try:
            return UserStore.confirm_email_change(token)[1]
        except ValueError:
            return "consumed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: confirm(), range(2)))
    assert sorted(results) == sorted([address, "consumed"])
