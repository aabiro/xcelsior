"""Operator scopes are refused on a LIVE server, at every path that writes them.

P0's gate is refusals proven with a real token against a running server, because
"a mock is what passed while production did not". Everything else asserting these
refusals runs against a `TestClient`; this file is the only thing that exercises
the deployed code.

Referenced by `.github/workflows/live-gates.yml`, manual dispatch only — it needs
a live tenant and a real credential, and a gate that silently no-ops is worse
than one that is visibly absent.

## Why the positive control is not optional

A server that answers 403 to everything passes a refusal-only test perfectly.
So each refusal here is paired with a **benign registration that must succeed**
on the same token, through the same path. Without it the suite would have
reported the fix live on 2026-08-04 when the 403s were coming from Cloudflare's
browser-integrity check (`error code: 1010`) and never reached the origin at all.

## Why the token must be a non-admin user

`_refuse_undelegatable_scopes` returns early for `_is_platform_admin(user)`, by
design: an admin *may* grant operator scopes. Run this with an admin token and
every probe "succeeds", the test reports the deployment vulnerable, and it will
have created a real `hosts:evict` client owned by that admin. The identity is
therefore asserted before any probe runs, rather than trusted.

Manual predecessor: `scripts/incident/` and the 2026-08-04 incident, where this
ran by hand against production and returned exit 0 at both write paths.
"""

from __future__ import annotations

import os
import uuid

import pytest

httpx = pytest.importorskip("httpx", reason="live gates install httpx explicitly")

BASE = (os.environ.get("XCELSIOR_LIVE_BASE_URL") or os.environ.get("XCELSIOR_STAGING_URL") or "").rstrip("/")
TOKEN = os.environ.get("XCELSIOR_NONADMIN_TOKEN", "")

#: Drain or evict any host on the platform. `control_plane_v1._require_host_operator`
#: authorises a machine principal on scope alone, so holding this *is* operator
#: authority — which is why a non-admin must never be able to mint it.
OPERATOR_SCOPE = "hosts:evict"
BENIGN_SCOPE = "instances:read"

#: Cloudflare answers `Python-urllib/*` and other default agents with a 403
#: carrying `error code: 1010`, before the origin sees the request. That is
#: indistinguishable from an authorization refusal at the status-code level, so
#: the edge has to admit us for any 403 below to mean what the test claims.
HEADERS = {
    "User-Agent": "xcelsior-live-gates/1.0 (+scope refusal gate)",
    "Accept": "application/json",
}

pytestmark = pytest.mark.skipif(
    not (BASE and TOKEN),
    reason="live gate: set XCELSIOR_LIVE_BASE_URL (or XCELSIOR_STAGING_URL) and XCELSIOR_NONADMIN_TOKEN",
)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(
        base_url=BASE,
        headers={**HEADERS, "Authorization": f"Bearer {TOKEN}"},
        timeout=30.0,
        follow_redirects=False,
    ) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def caller_is_a_non_admin(client):
    """Refuse to run at all on an admin token. A false pass here is expensive.

    Not a courtesy check: with an admin credential every probe below would be
    *allowed*, the suite would report the deployment broken, and it would leave a
    live operator-scoped client behind while doing so.
    """
    r = client.get("/api/auth/me")
    assert r.status_code == 200, (
        f"could not identify the caller (HTTP {r.status_code}). A gate that "
        f"cannot establish who it is proves nothing. Body: {r.text[:200]!r}"
    )
    user = r.json().get("user", r.json())
    assert user.get("is_admin") in (False, 0, None), (
        f"XCELSIOR_NONADMIN_TOKEN belongs to {user.get('email')!r}, which is a "
        "platform admin. Admins may grant operator scopes, so this token cannot "
        "test the refusal — it can only appear to fail it."
    )


@pytest.fixture
def benign_client_id(client):
    """A client the caller is genuinely allowed to create. Removed afterwards."""
    r = client.post(
        "/api/oauth/clients",
        json={
            "client_name": f"live-gate-benign-{uuid.uuid4().hex[:6]}",
            "grant_types": ["client_credentials"],
            "scopes": [BENIGN_SCOPE],
        },
    )
    assert r.status_code < 400, (
        f"the positive control failed (HTTP {r.status_code}): this token cannot "
        "register even a benign client, so a refusal below would prove nothing "
        f"about authorization. Body: {r.text[:300]!r}"
    )
    # The route answers {"ok": true, "client": {...}} — not a flat body.
    client_id = (r.json().get("client") or {}).get("client_id")
    assert client_id, f"registration returned no client_id: {r.text[:200]!r}"
    yield client_id
    client.delete(f"/api/oauth/clients/{client_id}")


def test_registration_refuses_an_operator_scope(client, benign_client_id):
    """`POST /api/oauth/clients` — the path the escalation was found on."""
    r = client.post(
        "/api/oauth/clients",
        json={
            "client_name": f"live-gate-probe-{uuid.uuid4().hex[:6]}",
            "grant_types": ["client_credentials"],
            "scopes": [OPERATOR_SCOPE],
        },
    )
    if r.status_code < 400:
        created = (r.json().get("client") or {}).get("client_id")
        if created:
            client.delete(f"/api/oauth/clients/{created}")
        pytest.fail(
            f"ACCEPTED an operator scope from a non-admin (HTTP {r.status_code}). "
            f"A client holding {OPERATOR_SCOPE!r} was created as {created!r} and "
            "this test attempted to remove it — verify it is gone. The "
            "deployment is vulnerable."
        )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text[:200]!r}"
    assert OPERATOR_SCOPE in r.text, (
        "refused, but without naming the scope — that reads as a generic denial "
        f"rather than the delegation check firing. Body: {r.text[:200]!r}"
    )


def test_update_refuses_an_operator_scope(client, benign_client_id):
    """`PATCH /api/oauth/clients/{id}` — the second writer, missed on the first pass.

    Registration and update both write `oauth_clients.scopes`. Guarding only the
    first left a two-request path to the same escalation: register benignly, then
    amend. This asserts the second door is shut on the same deployment.
    """
    r = client.patch(
        f"/api/oauth/clients/{benign_client_id}",
        json={"scopes": [BENIGN_SCOPE, OPERATOR_SCOPE]},
    )
    if r.status_code < 400:
        client.patch(f"/api/oauth/clients/{benign_client_id}", json={"scopes": [BENIGN_SCOPE]})
        pytest.fail(
            f"ACCEPTED an operator scope on update (HTTP {r.status_code}). The "
            f"client {benign_client_id!r} was amended to hold {OPERATOR_SCOPE!r} "
            "and this test attempted to revert it — verify. Registration being "
            "guarded is not sufficient; this is the second writer."
        )
    assert r.status_code == 403, f"expected 403, got {r.status_code}: {r.text[:200]!r}"


def test_the_benign_scope_still_works_after_both_refusals(client, benign_client_id):
    """The refusals are narrow, not a seized-up endpoint.

    Two 403s and a working registration in the same run is the shape that
    distinguishes "operator scopes are refused" from "client writes are broken".
    """
    r = client.patch(
        f"/api/oauth/clients/{benign_client_id}",
        json={"scopes": [BENIGN_SCOPE]},
    )
    assert r.status_code < 400, (
        f"a benign scope update failed (HTTP {r.status_code}) — the refusals "
        f"above may be an outage rather than a guard. Body: {r.text[:200]!r}"
    )
