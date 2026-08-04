"""A non-admin may not put operator scopes on an OAuth client row.

`control_plane_v1._require_host_operator` authorizes a *machine* principal on
its scope alone — correctly, since a machine has no role to inspect. That is
sound only while no non-admin can obtain such a credential, which makes every
write to `oauth_clients.scopes` an authorization decision.

Two routes write that column and neither checked:

* `POST /api/oauth/clients` — scopes went straight from the request body to the
  client row.
* `PATCH /api/oauth/clients/{client_id}` — `scopes` is in the `allowed` set of
  both `OAuthStore.update_client` and `update_client_in_workspace`, so a client
  created with harmless scopes could be amended afterwards.

Guarding only the first leaves the second open, which is why both are asserted
here and why each assertion names its route.

**This covers a stopgap, not the design.** The durable fix moves the check onto
the store methods that persist the column so every caller inherits it; there are
five paths to three writers and one bypasses the service funnel. When that
lands, these tests should keep passing against the new mechanism — if they need
editing, the mechanism does not cover what the stopgap did.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

#: Every scope that confers platform-operator authority. Listed rather than
#: imported so this test keeps meaning if the source constant is renamed or
#: moved — a guard that imports the thing it guards fails open when that thing
#: disappears.
OPERATOR_SCOPES = [
    "control_plane:operate",
    "control_plane:read",
    "hosts:evict",
    "hosts:fleet",
    "hosts:operate",
    "transparency:read",
    "transparency:write",
]

BENIGN = ["instances:read"]


@pytest.fixture(scope="module")
def headers():
    email = f"opscope-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Op Scope"},
    )
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.parametrize("scope", OPERATOR_SCOPES)
def test_registration_refuses_an_operator_scope(headers, scope):
    """`POST /api/oauth/clients` — the direct route."""
    r = client.post(
        "/api/oauth/clients",
        json={
            "client_name": f"probe-{uuid.uuid4().hex[:6]}",
            "grant_types": ["client_credentials"],
            "scopes": [scope],
        },
        headers=headers,
    )
    assert r.status_code == 403, (
        f"registering a client with {scope!r} returned {r.status_code}, not 403 — "
        f"a non-admin can mint themselves platform-operator authority: {r.text[:200]}"
    )


@pytest.mark.parametrize("scope", OPERATOR_SCOPES)
def test_update_refuses_an_operator_scope(headers, scope):
    """`PATCH /api/oauth/clients/{client_id}` — the second writer.

    Guarding registration alone leaves this open: register with harmless
    scopes, then amend. One extra request, same outcome.
    """
    created = client.post(
        "/api/oauth/clients",
        json={
            "client_name": f"probe-{uuid.uuid4().hex[:6]}",
            "grant_types": ["client_credentials"],
            "scopes": list(BENIGN),
        },
        headers=headers,
    )
    assert created.status_code in (200, 201), created.text
    client_id = created.json()["client"]["client_id"]

    r = client.patch(
        f"/api/oauth/clients/{client_id}",
        json={"scopes": [*BENIGN, scope]},
        headers=headers,
    )
    assert r.status_code == 403, (
        f"amending a client to hold {scope!r} returned {r.status_code}, not 403 — "
        f"the registration guard is bypassed by one extra request: {r.text[:200]}"
    )


def test_a_benign_registration_still_succeeds(headers):
    """The calibration control.

    Without this, a refusal of *everything* would satisfy the tests above and
    look like a working guard.
    """
    r = client.post(
        "/api/oauth/clients",
        json={
            "client_name": f"benign-{uuid.uuid4().hex[:6]}",
            "grant_types": ["client_credentials"],
            "scopes": list(BENIGN),
        },
        headers=headers,
    )
    assert r.status_code in (200, 201), (
        f"a client asking only for {BENIGN} was refused ({r.status_code}); the "
        f"guard is too broad: {r.text[:200]}"
    )


def test_a_benign_update_still_succeeds(headers):
    """The same control for the update path."""
    created = client.post(
        "/api/oauth/clients",
        json={
            "client_name": f"benign-{uuid.uuid4().hex[:6]}",
            "grant_types": ["client_credentials"],
            "scopes": list(BENIGN),
        },
        headers=headers,
    )
    assert created.status_code in (200, 201), created.text
    r = client.patch(
        f"/api/oauth/clients/{created.json()['client']['client_id']}",
        json={"client_name": "renamed-by-owner"},
        headers=headers,
    )
    assert r.status_code in (200, 204), (
        f"renaming an owned client was refused ({r.status_code}): {r.text[:200]}"
    )
