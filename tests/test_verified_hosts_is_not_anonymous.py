"""The verification table was readable by anyone.

`GET /api/verified-hosts` returns, for every host with a verification record
(not just the passing ones), its `overall_score`, its `gpu_fingerprint`, and
its `deverify_reason` — the reason that provider's machine *failed*
verification. It had no authentication.

So anyone could enumerate which providers had been deverified and why. That is
commercially sensitive to the provider and was never the point of the endpoint,
which exists so the trust page can show verification state.

Authentication rather than admin, because that is who consumes it:
`/dashboard/trust` is an ordinary signed-in dashboard page, and the legacy
console reaches it through a `window.fetch` wrapper that attaches the bearer
token to every API call. Requiring admin would break the first; requiring
nothing was the bug.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

import routes._deps as deps
from api import app

client = TestClient(app)

LEAKY_FIELDS = ("deverify_reason", "gpu_fingerprint")


@pytest.fixture
def signed_in():
    email = f"trust-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Trust Probe"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


def test_an_anonymous_caller_is_refused(monkeypatch) -> None:
    """The test environment disables AUTH_REQUIRED, so it is pinned back on.

    Without this the endpoint answers 200 to everyone here whether or not the
    guard exists, and the test would pass against the bug.
    """
    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    r = client.get("/api/verified-hosts")
    assert r.status_code in (401, 403), (
        f"anonymous read returned {r.status_code}; every host's deverify_reason "
        "and gpu_fingerprint is public"
    )
    for field in LEAKY_FIELDS:
        assert field not in r.text, f"{field} leaked in the refusal body"


def test_a_signed_in_user_can_still_read_it(signed_in) -> None:
    """Admin would have been the wrong level — /dashboard/trust is not admin."""
    r = client.get("/api/verified-hosts", headers=signed_in)
    assert r.status_code == 200, (
        f"a signed-in user was refused ({r.status_code}); the trust page is an "
        f"ordinary dashboard page: {r.text[:200]}"
    )
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["hosts"], list)


def test_the_handler_can_authenticate_at_all() -> None:
    """Structural: the defect was a missing `Request`, not a wrong value."""
    import inspect

    from routes.verification import api_verified_hosts

    assert "request" in inspect.signature(api_verified_hosts).parameters, (
        "api_verified_hosts takes no Request, so it cannot authenticate anyone"
    )
    assert "_require_auth(" in inspect.getsource(api_verified_hosts)
