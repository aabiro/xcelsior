"""A provider may read their own reputation detail, and no one else's.

`reputation:read` was added to the Quick Connect grant, and the argument for
that was **not** "reputation is harmless". It was specific: `/api/reputation/me`
and `/me/journey` resolve the subject from the caller's own credential,
`/api/trust-tiers` is one public ladder with no personal data, and the two
detail routes — `breakdown` and `history` — call
`_require_reputation_entity_access`, which admits the owner or a platform admin
and 403s everyone else.

That last clause is the whole load-bearing part, and when the grant was widened
**it had no test**. The check was read and believed. This suite has repeatedly
found that the seam you are relying on is the one nobody measured — the
admission filter that passed a raw payload, the WebSocket route that did not
exist, the three auth tests green against a nonexistent path. A grant justified
by a guard should not be the first thing to exercise it.

## The asymmetry is deliberate

`GET /api/reputation/{entity_id}` — the plain score — does **not** check
ownership, and that is correct rather than an oversight: a buyer choosing
between hosts has to be able to see a host's trust score, which is also why the
leaderboard exists. What is protected is the *derivation* — the event history and
the breakdown computed from it, which expose a provider's job-by-job record.

Score public, working public, history private. Asserted in both directions here
so that shape is a decision rather than a coincidence.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _account() -> tuple[dict[str, str], str]:
    """A registered user, their auth header, and their canonical owner id."""
    email = f"repacc-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Rep Access"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}, email


@pytest.fixture(scope="module")
def owner():
    return _account()


@pytest.fixture(scope="module")
def stranger():
    return _account()


# ── The protected derivation ──────────────────────────────────────────


@pytest.mark.parametrize("route", ["breakdown", "history"])
def test_a_stranger_cannot_read_another_entitys_detail(owner, stranger, route):
    _, owner_email = owner
    headers, _ = stranger
    r = client.get(f"/api/reputation/{owner_email}/{route}", headers=headers)
    assert r.status_code == 403, (
        f"a caller who is not the owner read {route} for {owner_email} and got "
        f"{r.status_code}. `reputation:read` is in the Quick Connect grant on "
        "the strength of this refusal — if it is gone, the grant must be "
        "reconsidered, not this test relaxed"
    )


@pytest.mark.parametrize("route", ["breakdown", "history"])
def test_the_owner_can_read_their_own(owner, route):
    """The other direction, so the guard cannot pass by refusing everyone."""
    headers, owner_email = owner
    r = client.get(f"/api/reputation/{owner_email}/{route}", headers=headers)
    assert r.status_code == 200, r.text


# ── The deliberately open half ────────────────────────────────────────


def test_the_plain_score_is_readable_by_anyone_authenticated(owner, stranger):
    """Not a hole. A buyer comparing hosts needs the score, which is why the
    leaderboard is open too. If this ever starts refusing, the change was
    deliberate and this test should be deleted along with the reasoning above —
    silently flipping it would break marketplace trust display."""
    _, owner_email = owner
    headers, _ = stranger
    r = client.get(f"/api/reputation/{owner_email}", headers=headers)
    assert r.status_code == 200, r.text


def test_the_detail_routes_are_not_simply_open(owner, stranger):
    """Calibration. If everything returned 200 the refusal test above could pass
    for the wrong reason — the parametrised cases assert a specific code, but
    this pins that the two halves genuinely differ for the same caller."""
    _, owner_email = owner
    headers, _ = stranger
    plain = client.get(f"/api/reputation/{owner_email}", headers=headers).status_code
    detail = client.get(f"/api/reputation/{owner_email}/breakdown", headers=headers).status_code
    assert (plain, detail) == (200, 403), (
        f"expected the score open and the breakdown closed, got {plain} and {detail}"
    )


def test_unauthenticated_callers_get_nothing(owner, monkeypatch):
    """Auth has to be switched back on to ask this.

    `tests/conftest.py` sets `routes._deps.AUTH_REQUIRED = False` for the whole
    suite, and `_require_auth` then returns an anonymous principal carrying
    `role: "admin", is_admin: True`. So an unauthenticated request here does not
    merely pass — it passes *as a platform admin*, which is also why it clears
    `_require_reputation_entity_access`. Asserting a 401 without flipping the
    flag would have been a test failing for an environmental reason and, in the
    other direction, any "unauthenticated access is refused" claim written
    without this is green against a configuration nobody runs. See
    `tests/test_compliance_surfaces_require_auth.py`, which exists for this.
    """
    import routes._deps as deps

    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    assert deps.AUTH_REQUIRED is True, "the flag was reverted; this proves nothing"

    _, owner_email = owner
    for path in (f"/api/reputation/{owner_email}", f"/api/reputation/{owner_email}/breakdown"):
        assert client.get(path).status_code in (401, 403), path
