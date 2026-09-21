"""The remaining singletons and small clusters nothing had entered.

`scripts/measure_route_execution.py` reports handlers the suite never reaches.
This file takes the ones that do not belong to a cluster big enough for a file
of their own — marketplace reservations, provider disconnects, host spot
controls, volume rename and promotion previews, compliance, notifications, team
roles, and the email/device verification routes.

Several of these can only be driven to their lookup and refusal paths here: a
Stripe Connect account session needs a configured processor, and a reservation
cancellation needs a reservation. Those still enter the handler and exercise
its own logic, which is the distinction this whole exercise is about — a 422
from FastAPI proves nothing about the code inside, a 404 the handler decided on
proves the lookup ran. Where a test can only reach a refusal, it says so.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth():
    email = f"scattered-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Scattered"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}", "_email": email}


def _h(auth: dict) -> dict:
    return {"Authorization": auth["Authorization"]}


# ── Marketplace v2 ────────────────────────────────────────────────────────


def test_creating_a_reservation(auth) -> None:
    r = client.post(
        "/api/v2/marketplace/reservations",
        json={"gpu_model": "RTX4090", "gpu_count": 1, "period_months": 1},
        headers=_h(auth),
    )
    assert r.status_code < 500, f"reservation faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (200, 400, 402, 404, 409), r.text[:200]


def test_cancelling_an_unknown_reservation(auth) -> None:
    r = client.delete(
        f"/api/v2/marketplace/reservations/{uuid.uuid4()}", headers=_h(auth)
    )
    assert r.status_code < 500, f"cancellation faulted: {r.status_code} {r.text[:300]}"


def test_allocating_from_an_unknown_offer(auth) -> None:
    """Allocation is the double-sell guard; an unknown offer must not fault."""
    r = client.post(
        "/api/v2/marketplace/allocate",
        json={
            "offer_id": f"offer-{uuid.uuid4().hex[:10]}",
            "job_id": f"job-{uuid.uuid4().hex[:10]}",
            "gpu_count": 1,
        },
        headers=_h(auth),
    )
    assert r.status_code < 500, f"allocate faulted: {r.status_code} {r.text[:300]}"


def test_marketplace_writes_require_authentication() -> None:
    r = client.post(
        "/api/v2/marketplace/reservations",
        json={"gpu_model": "RTX4090", "gpu_count": 1, "period_months": 1},
    )
    assert r.status_code in (401, 403), r.text[:200]


# ── Providers ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/api/providers/{pid}/account-session",
        "/api/providers/{pid}/stripe/disconnect",
        "/api/providers/{pid}/paypal/disconnect",
    ],
)
def test_provider_routes_do_not_fault_on_an_unknown_provider(path) -> None:
    pid = f"provider-{uuid.uuid4().hex[:10]}"
    r = client.post(path.format(pid=pid), headers=_admin_headers())
    assert r.status_code < 500, f"{path} faulted: {r.status_code} {r.text[:300]}"


@pytest.mark.parametrize(
    "path",
    [
        "/api/providers/{pid}/stripe/disconnect",
        "/api/providers/{pid}/paypal/disconnect",
    ],
)
def test_disconnecting_another_providers_payouts_is_refused(auth, path) -> None:
    """Unlinking a payout account is how a provider stops being paid.

    `_require_provider_access` is the only thing between this route and someone
    else's payouts, and nothing had ever called it from here.
    """
    pid = f"someone-else-{uuid.uuid4().hex[:10]}"
    r = client.post(path.format(pid=pid), headers=_h(auth))
    assert r.status_code in (403, 404), (
        f"a signed-in user got {r.status_code} unlinking another provider's payouts"
    )


# ── Host spot controls ────────────────────────────────────────────────────


def test_spot_preview_for_an_unknown_host_is_404() -> None:
    r = client.get(
        f"/api/hosts/no-such-host-{uuid.uuid4().hex[:8]}/spot-preview",
        headers=_admin_headers(),
    )
    assert r.status_code == 404, r.text[:200]


def test_spot_settings_for_an_unknown_host_is_404() -> None:
    r = client.patch(
        f"/api/hosts/no-such-host-{uuid.uuid4().hex[:8]}/spot-settings",
        json={"spot_enabled": True},
        headers=_admin_headers(),
    )
    assert r.status_code == 404, r.text[:200]


def test_spot_settings_bounds_are_enforced_by_the_model() -> None:
    """`spot_gpu_slots` is `ge=0, le=64`; out of range never reaches the host."""
    r = client.patch(
        f"/api/hosts/some-host-{uuid.uuid4().hex[:8]}/spot-settings",
        json={"spot_gpu_slots": 9999},
        headers=_admin_headers(),
    )
    assert r.status_code == 422, r.text[:300]


# ── Volumes ───────────────────────────────────────────────────────────────


def test_renaming_an_unknown_volume_is_404(auth) -> None:
    r = client.patch(
        f"/api/v2/volumes/{uuid.uuid4()}",
        json={"name": "renamed-by-test"},
        headers=_h(auth),
    )
    assert r.status_code < 500, f"rename faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404), r.text[:200]


def test_promotion_preview_for_an_unknown_volume(auth) -> None:
    r = client.get(
        f"/api/v2/volumes/{uuid.uuid4()}/promotions/preview",
        params={"job_id": f"job-{uuid.uuid4().hex[:10]}"},
        headers=_h(auth),
    )
    assert r.status_code < 500, f"preview faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404), r.text[:200]


def test_getting_an_unknown_promotion(auth) -> None:
    r = client.get(
        f"/api/v2/volumes/{uuid.uuid4()}/promotions/{uuid.uuid4()}", headers=_h(auth)
    )
    assert r.status_code < 500, f"promotion get faulted: {r.status_code} {r.text[:300]}"


def test_reopening_encrypted_volumes_is_admin_only(auth) -> None:
    """This route reopens every encrypted volume on the platform.

    It was gated with `_require_scope(user, "admin")` alone, which is not an
    admin check — `_require_scope` no-ops for interactive user sessions by
    design, because a browser session's OIDC scopes say nothing about API
    authority. An ordinary registered account got 200 from it.
    """
    r = client.post("/api/v2/admin/volumes/reopen-encrypted", json={}, headers=_h(auth))
    assert r.status_code in (401, 403), (
        f"a non-admin reached the encrypted-volume reopen route ({r.status_code})"
    )


# ── Compliance, notifications, teams ──────────────────────────────────────


def test_compliance_status_renders_its_checks() -> None:
    """Every non-passing check carries a CTA; a 500 here is a blank page."""
    r = client.get("/api/compliance/status", headers=_admin_headers())
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert "checks" in body, f"no checks in the compliance summary: {body}"


def test_listing_notifications(auth) -> None:
    """A new account already has a welcome notification, so this asserts shape."""
    r = client.get("/api/notifications", headers=_h(auth))
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["notifications"], list)
    assert isinstance(body["unread_count"], int)
    for item in body["notifications"]:
        assert "created_at" in item and "body" in item, f"malformed notification: {item}"


def test_listing_notifications_honours_the_unread_filter(auth) -> None:
    r = client.get("/api/notifications", params={"unread": True}, headers=_h(auth))
    assert r.status_code == 200, r.text[:300]


def test_notifications_require_authentication() -> None:
    r = client.get("/api/notifications")
    assert r.status_code in (401, 403), r.text[:200]


def test_updating_a_member_role_on_an_unknown_team(auth) -> None:
    r = client.patch(
        f"/api/teams/{uuid.uuid4()}/members/{auth['_email']}",
        json={"role": "member"},
        headers=_h(auth),
    )
    assert r.status_code < 500, f"role update faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404), r.text[:200]


def test_an_unknown_role_is_refused(auth) -> None:
    """The handler checks the value itself, before the team lookup."""
    r = client.patch(
        f"/api/teams/{uuid.uuid4()}/members/{auth['_email']}",
        json={"role": "superuser"},
        headers=_h(auth),
    )
    assert r.status_code in (400, 422), r.text[:300]


# ── Email and device verification ─────────────────────────────────────────


def test_verifying_an_email_with_a_bogus_token() -> None:
    r = client.post("/api/auth/verify-email", json={"token": f"bogus-{uuid.uuid4().hex}"})
    assert r.status_code == 400, r.text[:300]
    assert "token" in r.text.lower() or "invalid" in r.text.lower()


def test_verifying_a_device_with_a_bogus_code() -> None:
    r = client.post("/api/auth/verify", json={"user_code": "XXXX-XXXX"})
    assert r.status_code < 500, f"device verify faulted: {r.status_code} {r.text[:300]}"
