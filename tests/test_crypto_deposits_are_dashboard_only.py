"""Bitcoin and Lightning deposits are made in the dashboard, not by an agent.

The MCP crypto tools were withdrawn on 2026-11-24, but the deposit routes still
answered a connector or agent-key token holding `billing:topup`. So the
consent page and `get_permissions` told users that the permission let an agent
"create Bitcoin and Lightning deposits". That sentence was removed and the
routes now refuse agent credentials, so the two agree.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest
from fastapi import HTTPException

import routes._deps as _deps
from oauth_service import SCOPE_DESCRIPTIONS, accepted_mcp_audiences


def _user(**extra) -> dict:
    return {
        "email": "human@xcelsior.ca",
        "user_id": "human-1",
        "customer_id": "cust-1",
        "scopes": ["billing:topup"],
        **extra,
    }


@pytest.fixture
def as_caller(monkeypatch):
    def use(user: dict):
        monkeypatch.setattr(_deps, "_get_current_user", lambda request: user)
        monkeypatch.setattr(_deps, "_require_scope", lambda user, scope: None)
        monkeypatch.setattr(_deps, "_customer_ids_accessible_by_user", lambda user: {"cust-1"})
        monkeypatch.setattr(_deps, "_is_platform_admin", lambda user: False)

    return use


def _agents() -> list[dict]:
    audience = sorted(accepted_mcp_audiences())[0]
    return [
        _user(auth_type="oauth_access_token", audience=audience),
        _user(auth_type="agent_api_key"),
    ]


@pytest.mark.parametrize("agent", _agents(), ids=["mcp-connector-token", "agent-key"])
def test_an_agent_cannot_create_a_crypto_deposit(as_caller, agent):
    as_caller(agent)
    with pytest.raises(HTTPException) as refused:
        _deps._require_customer_access(None, "cust-1", funds_wallet=True)
    assert refused.value.status_code == 403
    assert refused.value.detail["code"] == "dashboard_only"


def test_the_dashboard_still_can(as_caller):
    as_caller(_user(auth_type="session"))
    try:
        _deps._require_customer_access(None, "cust-1", funds_wallet=True)
    except HTTPException as e:  # pragma: no cover - the failure being guarded
        assert e.detail.get("code") != "dashboard_only", e.detail


def test_the_topup_permission_no_longer_promises_crypto_deposits():
    text = SCOPE_DESCRIPTIONS["billing:topup"].lower()
    assert "bitcoin" not in text and "lightning" not in text
    assert "charge your saved card" in text
