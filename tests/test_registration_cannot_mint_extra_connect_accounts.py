"""One account, one provider identity — the caller does not get to choose it.

`POST /api/providers/register` pins `email` to the caller: *"You can only
register a provider for your own email."* It did **not** pin `provider_id`. That
field came whole from the request body, and `create_provider_account` looks for
an existing row by it — so a caller sending a fresh id each time created a **new
Stripe Connect account each time**, and the re-registration then ran

    UserStore.update_user(register_email, {"provider_id": ..., "role": "provider"})

which moved the link and orphaned the previous account: live at Stripe, belonging
to nobody, invisible to the user who caused it.

This is not hypothetical. The **1,389** test-mode Connect accounts purged on
2026-08-16 were fixtures registering as `prov-{uuid4}`, `idor-*`, `provcov-*`.
The fixtures were fixed at the time; the route was not, and those fixtures were
merely the callers that happened to do it. The dashboard was safe only by
convention — it sends `providerId || customerId`, deterministic per user — so
the rule lived in the frontend and the server had none. Every other caller, an
agent tool included, arrived without it.

The server now resolves what the UI already did: a non-admin gets their existing
`provider_id`, else their canonical owner id. Admins keep the explicit argument,
the same exemption the email check grants them, because they register on behalf
of someone else.

## Why this does not touch Stripe

`create_provider_account` is replaced with a recorder. The property under test is
which id the route *resolves and passes down*, and a test that proved it by
creating real Connect accounts would be committing the very act this constraint
exists to bound.

## What this replaces

An earlier version of this file documented the hazard instead of fixing it,
under "Named, not fixed". The fix was written then, and reverted because it broke
nine tests whose fixtures assumed the caller named the id — including a
cross-account isolation test that answered 404 where it asserted 403, because it
was probing a provider that had never been created. Those nine were the evidence
that the old contract had spread, not a reason to keep it: each now reads the
authoritative id out of the response.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


@pytest.fixture
def recorded_registrations(monkeypatch):
    """Replace the Stripe call with a recorder. Returns the list of ids seen."""
    import routes.providers as providers_routes

    seen: list[str] = []

    class _Recorder:
        def create_provider_account(self, *, provider_id, **_kwargs):
            seen.append(provider_id)
            return {
                "provider_id": provider_id,
                "stripe_account_id": f"acct_fake_{len(seen)}",
                "onboarding_url": "https://connect.stripe.test/setup",
                "status": "onboarding",
            }

        def get_provider(self, provider_id):
            return {"provider_id": provider_id, "status": "onboarding"}

    monkeypatch.setattr(providers_routes, "get_stripe_manager", lambda: _Recorder())
    return seen


def _account() -> tuple[dict[str, str], str]:
    email = f"regpin-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Reg Pin"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}, email


def _register(headers: dict[str, str], email: str, provider_id: str):
    return client.post(
        "/api/providers/register",
        json={
            "provider_id": provider_id,
            "email": email,
            "provider_type": "individual",
            "legal_name": "Reg Pin",
            "province": "ON",
        },
        headers=headers,
    )


# ── The constraint ────────────────────────────────────────────────────


def test_the_requested_id_is_ignored(recorded_registrations):
    headers, email = _account()
    invented = f"attacker-chosen-{uuid.uuid4().hex[:8]}"

    response = _register(headers, email, invented)
    assert response.status_code == 200, response.text

    assert recorded_registrations == [recorded_registrations[0]]
    assert recorded_registrations[0] != invented, (
        "the caller's chosen provider_id reached create_provider_account, so a "
        "caller can still mint a new Stripe Connect account per request by "
        "varying this field"
    )
    assert response.json()["provider_id"] != invented


def test_registering_twice_with_different_ids_resolves_to_one_account(recorded_registrations):
    """The property that bounds the blast radius.

    Two requests, two different invented ids, one resolved identity — so
    `create_provider_account` finds the existing row the second time and
    regenerates a link instead of creating an account.
    """
    headers, email = _account()

    first = _register(headers, email, f"first-{uuid.uuid4().hex[:8]}")
    second = _register(headers, email, f"second-{uuid.uuid4().hex[:8]}")
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    assert len(recorded_registrations) == 2, "both registrations should reach the manager"
    assert recorded_registrations[0] == recorded_registrations[1], (
        f"two registrations resolved to different ids "
        f"{recorded_registrations[0]!r} and {recorded_registrations[1]!r} — the "
        "second one creates a second Connect account and orphans the first"
    )
    assert first.json()["provider_id"] == second.json()["provider_id"]


def test_two_callers_never_resolve_to_the_same_provider(recorded_registrations):
    """It resolves to the *caller*, not merely to something stable.

    A constant would satisfy the test above and be far worse than the bug it
    replaced: two users would share one provider identity, and whoever
    registered second would inherit the first one's payout account.

    Asserted this way rather than by comparing against a stored `customer_id`
    because the resolved value comes from the caller's credential, and
    `UserStore.get_user` does not carry the same fields in this environment —
    a check written against the row passes or fails for reasons unrelated to
    the property.
    """
    headers_a, email_a = _account()
    headers_b, email_b = _account()

    assert _register(headers_a, email_a, "same-id-for-both").status_code == 200
    assert _register(headers_b, email_b, "same-id-for-both").status_code == 200

    resolved_a, resolved_b = recorded_registrations
    assert resolved_a != resolved_b, (
        f"two different accounts both resolved to {resolved_a!r}, so one "
        "caller's registration lands on another caller's provider account"
    )
    # And neither took the id they both asked for.
    assert "same-id-for-both" not in (resolved_a, resolved_b)


# ── The exemption ─────────────────────────────────────────────────────


def test_the_email_is_still_pinned(recorded_registrations):
    """The half that was always constrained. Asserted so a change that loosens
    the id rule cannot quietly take the email rule with it."""
    headers, _ = _account()
    other = f"someone-else-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    response = _register(headers, other, f"any-{uuid.uuid4().hex[:8]}")
    assert response.status_code == 403, (
        f"registering a provider under another party's email returned "
        f"{response.status_code}"
    )
    assert recorded_registrations == []


# ── The admin branch ──────────────────────────────────────────────────


def test_an_admin_may_still_name_a_provider_id(recorded_registrations, monkeypatch):
    """Admins register on another party's behalf — the same exemption the email
    check grants them, and the reason this is a constraint on callers rather
    than the removal of a field."""
    import routes.providers as providers_routes

    monkeypatch.setattr(providers_routes, "_is_platform_admin", lambda _user: True)
    headers, email = _account()
    chosen = f"admin-chosen-{uuid.uuid4().hex[:8]}"

    response = _register(headers, email, chosen)
    assert response.status_code == 200, response.text
    assert recorded_registrations == [chosen], (
        "an admin's explicit provider_id was overridden; they can no longer "
        "enrol a provider on someone else's behalf"
    )


def test_an_admin_who_names_nothing_resolves_to_their_own(recorded_registrations, monkeypatch):
    """`register_provider` sends no id at all.

    Without this fallback the admin branch took the empty string literally and
    answered "No provider identity for this account" — a confusing refusal for
    the one caller allowed to pass the field.
    """
    import routes.providers as providers_routes

    monkeypatch.setattr(providers_routes, "_is_platform_admin", lambda _user: True)
    headers, email = _account()

    response = _register(headers, email, "")
    assert response.status_code == 200, response.text
    assert recorded_registrations and recorded_registrations[0], (
        "an admin registering without naming an id got no provider identity"
    )
