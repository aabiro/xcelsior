"""Submitting a benchmark report must prove which host you are.

`POST /agent/verify` took a `host_id` from the request body and had **no
`Request` parameter at all** — so there was nothing to authenticate with. Any
unauthenticated caller could report for any host.

That is not a read. `run_verification` writes the host's verification state,
and a passing result grants `HARDWARE_AUDIT` reputation, which feeds provider
scoring and earnings. So the endpoint allowed a provider to verify a machine
with a forged benchmark, and to push a competitor's host toward deverification
with a failing one.

Its user-facing twin in the same file, `api_verify_host`, has always required
`verification:write`. Every other agent-reported endpoint in `routes/agent.py`
calls `_require_agent_auth(request, host_id=...)`, binding the credential to
the host named in the body — which is the part that stops one provider
reporting as another. This route was the half without a gate.

`/agent/v2/verify` is an alias onto the same function, so it was open too.
"""

from __future__ import annotations

import inspect
import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)

PATHS = ("/agent/verify", "/agent/v2/verify")


def _payload(host_id: str) -> dict:
    return {
        "host_id": host_id,
        "report": {"gpu_model": "RTX 4090", "total_vram_gb": 24, "gpu_temp_celsius": 55},
    }


@pytest.mark.parametrize("path", PATHS)
def test_the_handler_can_authenticate_at_all(path) -> None:
    """The defect was structural: no `Request`, so no credential to inspect.

    A behavioural test alone cannot catch its return — the test environment
    accepts unauthenticated agent calls by design, so a handler that silently
    lost its guard would still answer 200 here. This asserts the parameter and
    the call exist.
    """
    from routes.verification import api_agent_verify

    params = inspect.signature(api_agent_verify).parameters
    assert "request" in params, (
        "api_agent_verify takes no Request, so it cannot authenticate anyone — "
        "which is how it came to accept a host_id from any caller"
    )

    source = inspect.getsource(api_agent_verify)
    assert "_require_agent_auth(" in source, (
        "the handler no longer calls _require_agent_auth"
    )
    assert "host_id=payload.host_id" in source, (
        "the guard is not bound to the host named in the body, so one provider "
        "can still report as another"
    )


@pytest.mark.parametrize("path", PATHS)
def test_a_report_for_an_unknown_host_does_not_fault(path) -> None:
    """The route still answers; the gate refuses rather than crashing."""
    r = client.post(path, json=_payload(f"no-such-host-{uuid.uuid4().hex[:8]}"))
    assert r.status_code < 500, f"{path} faulted: {r.status_code} {r.text[:300]}"


@pytest.mark.parametrize("path", PATHS)
def test_an_unauthenticated_report_is_refused_under_production_rules(path, monkeypatch) -> None:
    """Behavioural proof, taken under the rules that matter.

    `_require_agent_auth` deliberately accepts unauthenticated agent calls when
    `XCELSIOR_ENV=test`, which is what the suite normally runs under — so a
    test that posts anonymously here gets 200 whether or not the guard exists,
    and proves nothing. Pinning the env to production selects the branch that
    "NEVER bypasses" and makes the refusal observable.

    It also keeps this file from having side effects. Before the guard, an
    anonymous post with a failing report ran `run_verification` and logged
    `HOST DEVERIFIED` for a host id of the caller's choosing — which is the
    vulnerability, demonstrated by the test that found it.
    """
    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.setenv("XCELSIOR_ALLOW_UNAUTH_AGENT", "0")

    r = client.post(path, json=_payload(f"probe-{uuid.uuid4().hex[:8]}"))
    assert r.status_code in (401, 403), (
        f"{path} returned {r.status_code} to an anonymous caller under "
        "production rules — anyone could write verification state, and "
        "reputation, for any host"
    )
    assert "DEVERIFIED" not in r.text


def test_the_v2_alias_points_at_the_same_handler() -> None:
    """Otherwise closing one leaves the other open.

    The alias is registered at startup from a table; if it ever stops resolving
    to this function, the two paths can drift apart in exactly the way that
    made this bug reachable by two routes at once.
    """
    endpoints = {
        getattr(r, "path", None): getattr(r, "endpoint", None) for r in app.routes
    }
    assert endpoints.get("/agent/verify") is not None
    assert endpoints.get("/agent/v2/verify") is not None
    assert endpoints["/agent/verify"] is endpoints["/agent/v2/verify"], (
        "the v2 alias no longer shares a handler with /agent/verify; a guard "
        "added to one would not protect the other"
    )
