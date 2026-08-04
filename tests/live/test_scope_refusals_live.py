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

#: Skipping lives on the fixture, not the module, so `assert_refusal_came_from
#: _the_origin` is still exercised in runs with no live credentials — including
#: ordinary CI. A guard against a false pass that only runs during a real
#: dispatch is a guard that is absent exactly when it is cheapest to break.
_NEEDS_LIVE = "live gate: set XCELSIOR_LIVE_BASE_URL (or XCELSIOR_STAGING_URL) and XCELSIOR_NONADMIN_TOKEN"


#: Markers that identify a 403 as the edge's, not the application's. Cloudflare's
#: browser-integrity block answers `error code: 1010` as plain text with a
#: `cf-ray` header, and nothing from the origin looks like that.
_EDGE_MARKERS = ("error code:", "cloudflare", "attention required", "<!doctype html")


def assert_refusal_came_from_the_origin(response) -> None:
    """A 403 only means "authorization refused" if the origin produced it.

    The status code alone cannot distinguish an authorization refusal from an
    edge rule that never reached the application — and on 2026-08-04 an edge rule
    is exactly what a first live run met. Where the response body names the scope
    the distinction is self-evident; where it does not (the update path answers
    with a bare detail message) it has to be asserted, or this test passes on a
    Cloudflare block and reports the guard working.
    """
    body = (response.text or "")[:500].lower()
    for marker in _EDGE_MARKERS:
        assert marker not in body, (
            f"the 403 came from the edge, not the application ({marker!r} in the "
            "body) — the request never reached the origin, so this proves nothing "
            f"about authorization. Body: {response.text[:200]!r}"
        )
    assert "cf-ray" not in {k.lower() for k in response.headers} or response.headers.get(
        "content-type", ""
    ).startswith("application/json"), (
        "the 403 carries edge headers and a non-JSON body — treat as an edge "
        f"block rather than a refusal. Headers: {dict(response.headers)}"
    )


@pytest.fixture(scope="module")
def client():
    if not (BASE and TOKEN):
        pytest.skip(_NEEDS_LIVE)
    with httpx.Client(
        base_url=BASE,
        headers={**HEADERS, "Authorization": f"Bearer {TOKEN}"},
        timeout=30.0,
        follow_redirects=False,
    ) as c:
        yield c


@pytest.fixture(scope="module")
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
def benign_client_id(client, caller_is_a_non_admin):
    """A client the caller is genuinely allowed to create. Removed afterwards.

    Depends on `caller_is_a_non_admin` rather than leaving it `autouse`: chained
    here it still runs before anything touches production (every live test below
    takes this fixture), while the edge-detector tests at the bottom — which need
    no credentials — are not dragged into a skip by it.
    """
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
    assert_refusal_came_from_the_origin(r)


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


# ── The edge detector, driven both ways ───────────────────────────────
# No credentials needed, so this runs everywhere.


def test_an_edge_block_is_not_accepted_as_a_refusal():
    """The exact response that met the first live run on 2026-08-04."""
    edge = httpx.Response(403, text="error code: 1010\n", headers={"cf-ray": "8f0-YYZ"})
    with pytest.raises(AssertionError, match="came from the edge"):
        assert_refusal_came_from_the_origin(edge)


def test_an_html_challenge_page_is_not_accepted_as_a_refusal():
    challenge = httpx.Response(
        403,
        text="<!DOCTYPE html><title>Attention Required! | Cloudflare</title>",
        headers={"content-type": "text/html"},
    )
    with pytest.raises(AssertionError):
        assert_refusal_came_from_the_origin(challenge)


def test_the_application_refusal_is_accepted():
    """The other direction: a real origin refusal must pass cleanly."""
    origin = httpx.Response(
        403,
        json={"detail": "These scopes may only be granted by a platform administrator: hosts:evict"},
    )
    assert_refusal_came_from_the_origin(origin)
