"""An agent may add an MFA factor. It may never take one away.

`DELETE /api/auth/mfa/all` deletes every MFA method **and** every backup code.
Its entire authorization was: authenticated, and holds `mfa:write`. No password
re-entry, no step-up, no challenge on the factor being removed.

`mfa:write` is not in `OPERATOR_SCOPES`, so any non-admin could delegate it to a
third-party OAuth client, which could then strip the account's second factor
silently. And until the connector-scope cutover, `_require_scope` was a no-op for
`oauth_access_token` — so on this surface the consequence was not "a narrowed
token behaves like a full one", it was that **any** connector token could call it
whether or not `mfa:write` had ever been granted.

## Why a first-party session, specifically

No mainstream provider exposes "disable MFA" to a third-party OAuth app. GitHub's
API has no such endpoint; Google publishes no such scope; Microsoft Graph puts
authentication-method deletion behind the user's own session or an admin role.
Account-security settings are first-party everywhere, and that is the line
`_is_interactive_human` draws.

## The asymmetry, and why it points this way

Setup routes stay open to machine callers. Removal does not. Safety must never be
harder to reach than the risk it undoes — here the *safe* state is having MFA, so
friction belongs on removal alone. This is the same shape as widening auto-top-up
needing an approved plan while lowering it needs nothing.

## What this does not do

It is **not** step-up re-authentication. A dashboard session open for a week
still passes, where the industry norm is also a fresh password inside a short
window. That is a separate change. This closes the delegation hole — the half no
other platform leaves open.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

#: Every route that reduces the account's authentication strength.
REMOVAL_HANDLERS = (
    "api_mfa_totp_disable",
    "api_mfa_sms_disable",
    "api_mfa_passkey_delete",
    "api_mfa_disable_all",
    "api_mfa_regenerate_backup_codes",
)

#: Routes that *add* a factor. These must stay reachable by a machine caller —
#: gating them would make the platform less secure by making MFA harder to adopt.
SETUP_HANDLERS = (
    "api_mfa_totp_setup",
    "api_mfa_sms_setup",
)


def _calls_in(handler_name: str) -> set[str]:
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((repo / "routes" / "mfa.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == handler_name:
            return {
                getattr(c.func, "id", "") or getattr(c.func, "attr", "")
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
            }
    raise AssertionError(f"{handler_name} not found in routes/mfa.py")


def test_the_handlers_are_still_where_this_thinks_they_are():
    """Prove the reach — a rename makes every assertion below vacuous."""
    for name in REMOVAL_HANDLERS + SETUP_HANDLERS:
        assert _calls_in(name), f"{name} has no calls; the parser is broken"


@pytest.mark.parametrize("handler", REMOVAL_HANDLERS)
def test_every_removal_route_requires_a_human(handler: str):
    assert "_require_human_to_weaken_mfa" in _calls_in(handler), (
        f"{handler} reduces the account's authentication strength without "
        "requiring a first-party session — an API key or a connected agent "
        "could strip the second factor"
    )


@pytest.mark.parametrize("handler", SETUP_HANDLERS)
def test_setup_routes_are_not_gated(handler: str):
    """The other half. A blanket gate would be the wrong shape.

    If adding a factor needed a human too, this would be friction on the safe
    direction, and the asymmetry the guard exists to express would be lost.
    """
    assert "_require_human_to_weaken_mfa" not in _calls_in(handler), (
        f"{handler} adds a factor and must not require a human — gating setup "
        "makes MFA harder to adopt, which is the opposite of the intent"
    )


def test_the_guard_admits_a_dashboard_session():
    from routes.mfa import _require_human_to_weaken_mfa

    _require_human_to_weaken_mfa(
        {
            "auth_type": "oauth_access_token",
            "session_type": "browser",
            "client_id": "xcelsior-web",
        }
    )


@pytest.mark.parametrize(
    "principal,label",
    [
        ({"auth_type": "client_credentials", "grant_type": "client_credentials"}, "an OAuth client"),
        ({"auth_type": "agent_api_key"}, "an agent key"),
        (
            {
                "auth_type": "oauth_access_token",
                "session_type": "browser",
                "client_id": "third-party-agent",
            },
            "a third-party connector token",
        ),
    ],
)
def test_the_guard_refuses_every_machine_caller(principal: dict, label: str):
    from fastapi import HTTPException

    from routes.mfa import _require_human_to_weaken_mfa

    with pytest.raises(HTTPException) as excinfo:
        _require_human_to_weaken_mfa(principal)
    assert excinfo.value.status_code == 403, f"{label} was not refused"


def test_the_connector_token_case_is_the_one_the_weaker_predicate_would_admit():
    """Named separately because it is the whole reason for the new predicate.

    `routes.action_plans._is_human` asks only whether the grant was
    `client_credentials`, so it counts a third-party connector token as a
    person. Had this guard used it, the delegation hole would have survived the
    fix intended to close it.
    """
    from routes._deps import _is_interactive_human
    from routes.action_plans import _is_human

    connector = {
        "auth_type": "oauth_access_token",
        "session_type": "browser",
        "client_id": "third-party-agent",
    }
    assert _is_human(connector) is True
    assert _is_interactive_human(connector) is False
