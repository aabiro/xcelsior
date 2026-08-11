"""A worker carrying the gateway secret passes `deny`; anything else does not.

`AgentIngressMiddleware` answers every worker path with 410 when
`XCELSIOR_AGENT_PUBLIC_INGRESS=deny`, unless the request proves it came through
the private gateway. That proof is a shared secret compared with
`hmac.compare_digest` — a bare `X-Xcelsior-Agent-Gateway: 1` is forgeable on
public ingress, which is the whole reason the secret exists.

The check is **hostname-independent**, so this restores a fleet without a client
CA, a server certificate, an nginx change or a DNS change — none of which exist
yet. It is a stopgap with a real leak radius: the secret is being used as a
bearer credential on public ingress rather than as gateway attestation, so
whoever holds it is at `allow`. That is why it is scoped to restoring
reachability and not treated as the cutover.

**No secret value appears in this file.** The tests generate their own throwaway
strings; nothing reads, prints or asserts against the production value.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from control_plane.identity import gateway_headers_authenticated  # noqa: E402

#: Local to this file, never the real one.
FAKE_SECRET = "test-only-gateway-secret-not-production"


@pytest.fixture
def secret_set(monkeypatch):
    monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", FAKE_SECRET)
    return FAKE_SECRET


# ── The server side ───────────────────────────────────────────────────


def test_the_correct_header_authenticates(secret_set):
    assert gateway_headers_authenticated({"x-xcelsior-gateway-auth": FAKE_SECRET})


def test_a_wrong_header_does_not(secret_set):
    assert not gateway_headers_authenticated({"x-xcelsior-gateway-auth": "wrong"})


def test_an_absent_header_does_not(secret_set):
    assert not gateway_headers_authenticated({})


def test_the_forgeable_marker_alone_does_not(secret_set):
    """`X-Xcelsior-Agent-Gateway: 1` is settable by anyone reaching the API."""
    assert not gateway_headers_authenticated({"x-xcelsior-agent-gateway": "1"})


def test_no_secret_configured_means_no_one_passes(monkeypatch):
    """Fail closed: an unset secret must not become "everyone authenticates"."""
    monkeypatch.delenv("XCELSIOR_AGENT_GATEWAY_SECRET", raising=False)
    assert not gateway_headers_authenticated({"x-xcelsior-gateway-auth": FAKE_SECRET})
    assert not gateway_headers_authenticated({"x-xcelsior-gateway-auth": ""})


# ── The worker side ───────────────────────────────────────────────────


def _worker_headers(monkeypatch, secret: str | None):
    import worker_agent

    if secret is None:
        monkeypatch.delenv("XCELSIOR_AGENT_GATEWAY_SECRET", raising=False)
    else:
        monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", secret)
    monkeypatch.setattr(worker_agent, "_host_token_value", lambda: "")
    monkeypatch.setattr(worker_agent, "_oauth_client_credentials_enabled", lambda: False)
    return worker_agent._api_headers()


def test_the_worker_sends_the_header_when_enrolled(monkeypatch):
    headers = _worker_headers(monkeypatch, FAKE_SECRET)
    assert headers.get("X-Xcelsior-Gateway-Auth") == FAKE_SECRET


def test_the_worker_omits_it_when_not_enrolled(monkeypatch):
    """A host with no secret sends no header and fails exactly as it does today.

    Not an empty string: an empty header is a presented-and-wrong credential,
    which is a different thing from not claiming one.
    """
    headers = _worker_headers(monkeypatch, None)
    assert "X-Xcelsior-Gateway-Auth" not in headers


def test_a_whitespace_only_secret_is_treated_as_absent(monkeypatch):
    headers = _worker_headers(monkeypatch, "   ")
    assert "X-Xcelsior-Gateway-Auth" not in headers


def test_what_the_worker_sends_is_what_the_server_accepts(monkeypatch):
    """The end-to-end pairing, which is the only thing that restores the fleet.

    Both sides read `XCELSIOR_AGENT_GATEWAY_SECRET` by that exact name — a
    translation layer between them would be one more place to disagree silently.
    """
    headers = _worker_headers(monkeypatch, FAKE_SECRET)
    lowered = {k.lower(): v for k, v in headers.items()}
    assert gateway_headers_authenticated(lowered)


# ── Through the middleware, which is what actually 410s ───────────────


@pytest.fixture
def denied(monkeypatch):
    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "deny")
    monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", FAKE_SECRET)
    from fastapi.testclient import TestClient

    import api as api_mod

    return TestClient(api_mod.app)


def test_a_worker_path_is_refused_without_the_header(denied):
    response = denied.post("/agent/ssh-status/does-not-matter", json={})
    assert response.status_code == 410
    assert "agent_ingress_retired" in response.text


def test_a_worker_path_with_a_wrong_header_is_refused(denied):
    response = denied.post(
        "/agent/ssh-status/does-not-matter",
        json={},
        headers={"X-Xcelsior-Gateway-Auth": "wrong"},
    )
    assert response.status_code == 410


def test_the_header_gets_past_the_gate(denied):
    """Past the *gate* — the route's own auth still applies beyond it.

    Anything other than 410 proves the ingress refusal was cleared, which is the
    only thing this header is responsible for.
    """
    response = denied.post(
        "/agent/ssh-status/does-not-matter",
        json={},
        headers={"X-Xcelsior-Gateway-Auth": FAKE_SECRET},
    )
    assert response.status_code != 410, (
        "the gateway secret did not clear the ingress gate, so no worker "
        "carrying it can reach the API"
    )
