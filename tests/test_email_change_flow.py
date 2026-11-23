"""A verified email change, end to end: request, link, confirm.

Settings showed the account email read-only, so the owner believed the change
was never built (review, 2026-11-22). The routes existed but only the request
and a bogus-token confirm were ever exercised. This walks the whole flow the
way the page does, and holds the two things the page relies on:

* ``/api/auth/me`` reports an unconfirmed change, so a reload can say where
  the link went instead of offering the form again.
* The session the change was made from keeps working with the new address,
  and both addresses are told.
"""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
from core.db import UserStore
import routes.auth as auth_routes

client = TestClient(app)


@pytest.fixture
def sent(monkeypatch):
    mails: list[tuple[str, str, str]] = []

    def _capture(to_email, subject, body_text, cta_url=None, cta_label=""):
        mails.append((to_email, subject, body_text))

    monkeypatch.setattr(auth_routes, "_send_team_email", _capture)
    return mails


@pytest.fixture
def account():
    email = f"change-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Change Probe"},
    )
    token = r.json().get("access_token")
    assert token, f"registration did not return a token: {r.status_code} {r.text[:200]}"
    return email, {"Authorization": f"Bearer {token}"}


def _me(headers: dict) -> dict:
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200, r.text[:200]
    return r.json()["user"]


def test_an_email_change_completes_from_the_link(account, sent):
    old_email, headers = account
    new_email = f"moved-{uuid.uuid4().hex[:8]}@xcelsior.ca"

    r = client.post("/api/auth/me/email-change", json={"new_email": new_email}, headers=headers)
    assert r.status_code == 200, r.text[:200]
    assert r.json()["pending_email"] == new_email

    # Nothing changes until the link is used, and the page can see it is pending.
    me = _me(headers)
    assert me["email"] == old_email
    assert me["pending_email"] == new_email
    assert [to for to, _, _ in sent] == [new_email]

    token = UserStore.get_user(old_email)["email_change_token"]
    assert token and token in sent[0][2], "the link mailed to the new address carries the token"

    r = client.post("/api/auth/me/email-change/confirm", json={"token": token})
    assert r.status_code == 200, r.text[:200]
    assert r.json()["email"] == new_email

    # The same session, now under the new address, with nothing pending.
    me = _me(headers)
    assert me["email"] == new_email
    assert "pending_email" not in me
    assert UserStore.get_user(old_email) is None

    # The new address is confirmed; the old one is told it was replaced.
    recipients = [to for to, _, _ in sent[1:]]
    assert new_email in recipients
    assert old_email in recipients, "the address the account left must hear about the change"
    notice = next(body for to, _, body in sent[1:] if to == old_email)
    assert new_email in notice

    # The link is single-use.
    r = client.post("/api/auth/me/email-change/confirm", json={"token": token})
    assert r.status_code == 400


def test_an_expired_change_is_not_reported_as_pending(account, sent):
    old_email, headers = account
    new_email = f"late-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post("/api/auth/me/email-change", json={"new_email": new_email}, headers=headers)
    assert r.status_code == 200, r.text[:200]

    UserStore.update_user(old_email, {"email_change_expires": time.time() - 1})

    assert "pending_email" not in _me(headers)
    token = UserStore.get_user(old_email)["email_change_token"]
    r = client.post("/api/auth/me/email-change/confirm", json={"token": token})
    assert r.status_code == 400
    assert "expired" in r.text


def test_an_address_another_account_holds_is_refused(account, sent):
    _, headers = account
    other = f"taken-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": other, "password": "StrongPass123!", "name": "Other"},
    )
    sent.clear()  # the other account's own sign-up mail
    r = client.post("/api/auth/me/email-change", json={"new_email": other}, headers=headers)
    assert r.status_code == 409
    assert sent == [], "nothing is mailed for a refused change"
