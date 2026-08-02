"""Regression tests for API-layer scope enforcement.

`_require_scope` used to gate on ``grant_type == "client_credentials"`` and
return immediately otherwise. An agent-key principal has no ``grant_type`` field
at all (see ``validate_agent_api_key`` in ``oauth_service.py``), so scope
enforcement was skipped entirely for every agent key — including the Quick
Connect keys the MCP quickstarts tell users to paste, which are issued with a
deliberately narrowed scope set.

Each test here was confirmed to fail against the previous implementation.
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi import HTTPException

from routes._deps import _require_scope

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _agent_key(*scopes: str) -> dict:
    """An agent-key principal, shaped as `validate_agent_api_key` returns it."""
    return {"auth_type": "agent_api_key", "scopes": list(scopes)}


class TestAgentKeysAreScopeChecked:
    def test_agent_key_cannot_exceed_its_scopes(self):
        with pytest.raises(HTTPException) as exc:
            _require_scope(_agent_key("gpu:read"), "instances:write", "billing:write")
        assert exc.value.status_code == 403
        assert "Insufficient scope" in str(exc.value.detail)

    def test_agent_key_within_its_scopes_passes(self):
        _require_scope(_agent_key("instances:write", "billing:read"), "instances:write")

    def test_every_required_scope_must_be_present(self):
        """Not any-one-of: holding one of two required scopes is not enough."""
        with pytest.raises(HTTPException):
            _require_scope(_agent_key("instances:write"), "instances:write", "billing:write")


class TestNoWildcardGrant:
    """`api` was a wildcard that satisfied every check. It is gone entirely."""

    @pytest.mark.parametrize(
        "scope", ["instances:write", "billing:write", "hosts:evict", "control_plane:operate"]
    )
    def test_api_satisfies_nothing(self, scope):
        with pytest.raises(HTTPException, match="Insufficient scope"):
            _require_scope({"grant_type": "client_credentials", "scopes": ["api"]}, scope)

    def test_operator_scope_granted_explicitly_passes(self):
        _require_scope(
            {"grant_type": "client_credentials", "scopes": ["hosts:evict"]}, "hosts:evict"
        )

    def test_every_required_scope_must_be_held_explicitly(self):
        _require_scope(
            {"grant_type": "client_credentials", "scopes": ["instances:write", "hosts:operate"]},
            "instances:write",
            "hosts:operate",
        )


class TestNonScopedPrincipalsUnaffected:
    """Interactive and legacy sessions must keep working."""

    @pytest.mark.parametrize("principal", [{}, {"scopes": None}, {"scopes": []}])
    def test_principal_without_scopes_is_not_gated(self, principal):
        _require_scope(principal, "hosts:evict", "billing:write")

    def test_oidc_identity_scopes_are_not_api_authority(self):
        """A browser session carries OIDC scopes, not API scopes.

        Gating on "has a scopes list" rather than "is a machine credential"
        denied every interactive request for lacking API scopes it was never
        meant to hold — 122 test failures with
        `granted: email, offline_access, profile`.
        """
        browser = {
            "auth_type": "oauth_access_token",
            "grant_type": "authorization_code",
            "scopes": ["profile", "email", "offline_access"],
        }
        _require_scope(browser, "volumes:write", "hosts:evict")

    def test_master_token_is_gated_by_admin_checks_not_scopes(self):
        _require_scope({"auth_type": "master_token", "scopes": ["profile"]}, "hosts:evict")


def test_no_wildcard_scope_survives_in_either_layer():
    """Neither layer may reintroduce a value that means "everything"."""
    ts = (ROOT / "mcp" / "src" / "auth" / "scopes.ts").read_text(encoding="utf-8")
    assert '| "api"' not in ts, "`api` is back in the McpScope enum"
    assert 'granted.has("api")' not in ts, "`api` is back as a wildcard grant"

    # Sweep every route module, not just _deps. A third bypass was hiding in
    # routes/action_plans.py — `if "api" not in held` — and was only caught
    # because a worker test failed. A grep for the pattern finds it directly.
    for module in sorted((ROOT / "routes").glob("*.py")):
        body = module.read_text(encoding="utf-8")
        for pattern in ('"api" in granted', '"api" not in held', '"api" in held',
                        '"api" in scopes', '"api" in user'):
            assert pattern not in body, f"{module.name} reintroduces an `api` bypass: {pattern}"

    auth = (ROOT / "routes" / "auth.py").read_text(encoding="utf-8")
    assert 'default_factory=lambda: ["api"]' not in auth, (
        "client registration defaults to a wildcard scope again"
    )
