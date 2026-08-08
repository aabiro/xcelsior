"""A narrowed credential must actually be refused, not merely be checked for.

`tests/test_serverless_writes_honour_scope.py` counts `_require_scope` calls in
the source. That is a structural claim: it proves a line exists. It cannot prove
the line *does* anything — a scope named wrong, checked against the wrong
principal, or swallowed by an `except` would leave the count unchanged and the
door open.

So this drives the real handlers with real principals and asserts the 403.

**Why a machine principal and not a browser session.** `_require_scope` is a
no-op for interactive sessions by design (`routes/_deps.py`) — it gates
`client_credentials` and `agent_api_key`. That is exactly the credential class
Quick Connect issues, so these assertions describe what a pasted connector token
can and cannot do. It also means these tests say nothing about a first-party
dashboard session, and they should not: scoping was never meant to bite there.
The consequence worth keeping in view is the converse — a third-party
authorization-code token is *also* exempt today, so route scoping does not
constrain it either. That is a separate, larger decision, and this file does not
pretend to cover it.

Quick Connect carries both `inference:read` and `inference:write`
(`MCP_QUICK_CONNECT_SCOPES`), so scoping these routes cannot reproduce the
`instances:connect` failure — a scope enforced on routes and missing from the
set the quickstart token holds. The last test here asserts that directly rather
than trusting the reasoning.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

READ_ROUTE = ("GET", "/api/v2/serverless/endpoints")
WRITE_ROUTE = ("DELETE", "/api/v2/serverless/endpoints/does-not-exist")


def _machine(scopes: list[str]) -> dict:
    """A client-credentials principal — the class `_require_scope` gates."""
    return {
        "email": "demo@xcelsior.ca",
        "user_id": "demo-user",
        "role": "user",
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "client_id": "client-test",
        "scopes": list(scopes),
    }


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import api as api_mod

    return TestClient(api_mod.app)


def _as(monkeypatch, principal: dict) -> None:
    from routes import _deps

    monkeypatch.setattr(_deps, "_get_current_user", lambda request: dict(principal))


def test_the_scope_helper_still_gates_machine_principals():
    """Calibration. If `_require_scope` stopped refusing, every test below would
    pass by never being reached."""
    from fastapi import HTTPException

    from routes._deps import _require_scope

    with pytest.raises(HTTPException) as excinfo:
        _require_scope(_machine(["gpu:read"]), "inference:read")
    assert excinfo.value.status_code == 403


def test_a_read_scope_alone_cannot_delete_an_endpoint(client, monkeypatch):
    """The property the whole re-scope exists for."""
    _as(monkeypatch, _machine(["inference:read"]))
    verb, path = WRITE_ROUTE
    response = client.request(verb, path)
    assert response.status_code == 403, (
        f"a credential holding only inference:read got {response.status_code} "
        "from a delete — scope reduction is not being enforced"
    )
    assert "scope" in response.text.lower()


def test_no_inference_scope_at_all_cannot_read(client, monkeypatch):
    _as(monkeypatch, _machine(["gpu:read", "billing:read"]))
    verb, path = READ_ROUTE
    response = client.request(verb, path)
    assert response.status_code == 403, (
        f"a credential with no inference scope got {response.status_code} from a "
        "serverless read"
    )


def test_the_right_scope_is_admitted(client, monkeypatch):
    """The other half. A guard that refuses everything is not enforcement.

    The endpoint does not exist, so anything other than 403 means the scope
    check passed and the handler proceeded — which is what is being asserted.
    """
    _as(monkeypatch, _machine(["inference:write", "inference:read"]))
    verb, path = WRITE_ROUTE
    response = client.request(verb, path)
    assert response.status_code != 403, (
        "a credential holding inference:write was refused by the scope check"
    )


def test_quick_connect_holds_what_these_routes_now_demand():
    """The pre-check that the `instances:connect` failure earned.

    That defect was a scope enforced on routes and absent from the set the
    quickstart token carries, discovered in production. Asserted here rather
    than reasoned about.
    """
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    held = set(MCP_QUICK_CONNECT_SCOPES)
    missing = {"inference:read", "inference:write"} - held
    assert not missing, (
        f"the serverless routes now require {sorted(missing)}, which a pasted "
        "Quick Connect token does not carry — this is the shape of the "
        "instances:connect 403"
    )
