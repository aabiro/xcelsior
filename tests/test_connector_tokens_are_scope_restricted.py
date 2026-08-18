"""A connector token's scopes are a restriction; a browser session's are an identity.

Both arrive as `auth_type == "oauth_access_token"`, and `_require_scope` treats
both as exempt. For the browser session that is correct and must stay correct: its
scopes are OIDC claims (`profile`, `email`, `offline_access`) that say nothing
about API authority, so gating on them would deny every dashboard request for
lacking scopes it was never meant to hold.

For a **third-party connector token** it is wrong. Those scopes are exactly what
the user ticked on a consent screen, and treating them as decorative means route
scoping buys nothing against the credential class most likely to be pointed at the
REST API by an agent.

## The discriminator was already here

`oauth_service._issue_access_token` decides token lifetime with
`session_type == "browser" and client_id == "xcelsior-web"`, and says in a comment
that third-party grants "keep the short access-token TTL **even though they also
carry session_type=browser**". So `session_type` alone cannot separate them and
`client_id` can. This file asserts that the same pair separates the two credential
classes for scope enforcement — one distinction, not two that can drift apart.

## Why the default is `shadow` and not `enforce`

Routes demand 36 distinct scopes. A Quick Connect token carries 14. Flipping
straight to `enforce` would refuse connector traffic wherever those sets differ —
the `instances:connect` failure (a scope enforced on routes and absent from the
token that must hold it), reproduced across production in one deploy.

`shadow` records what `enforce` would have refused, naming the client and the
missing scope, so the required set can be *read off real traffic* rather than
guessed. The cutover is then an owner's decision backed by evidence.

**A shadow that observes nothing is worse than no shadow**, because it looks like
progress. `_record_connector_scope_shadow` swallows every exception by design, so
a typo or a missing logger would silence it invisibly — this file therefore proves
it emits, rather than trusting that it does.
"""

from __future__ import annotations

import logging
import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")


def _browser_session(scopes: list[str] | None = None) -> dict:
    """A first-party dashboard session — the one that must never be gated."""
    return {
        "email": "human@xcelsior.ca",
        "user_id": "human-1",
        "auth_type": "oauth_access_token",
        "session_type": "browser",
        "client_id": "xcelsior-web",
        "scopes": scopes if scopes is not None else ["profile", "email", "offline_access"],
    }


def _connector_token(scopes: list[str]) -> dict:
    """A third-party authorization-code token — a delegation, not an identity."""
    return {
        "email": "human@xcelsior.ca",
        "user_id": "human-1",
        "auth_type": "oauth_access_token",
        "session_type": "browser",  # deliberately the same as the browser session
        "client_id": "some-third-party-connector",
        "scopes": list(scopes),
    }


# --------------------------------------------------------------------------
# The discriminator
# --------------------------------------------------------------------------


def test_a_browser_session_is_not_a_delegated_token():
    from routes._deps import _is_delegated_connector_token

    assert _is_delegated_connector_token(_browser_session()) is False


def test_a_third_party_token_is_delegated_even_though_it_says_browser():
    """The exact case the TTL comment warns about.

    If this ever returns False, the two credential classes have become
    indistinguishable again and every assertion below is vacuous.
    """
    from routes._deps import _is_delegated_connector_token

    assert _is_delegated_connector_token(_connector_token(["instances:read"])) is True


def test_a_machine_credential_is_not_reclassified_as_a_connector():
    """`client_credentials` and agent keys keep their existing path."""
    from routes._deps import _is_delegated_connector_token

    assert _is_delegated_connector_token({"auth_type": "client_credentials"}) is False
    assert _is_delegated_connector_token({"auth_type": "agent_api_key"}) is False


def test_the_discriminator_matches_the_one_that_decides_token_lifetime():
    """One distinction, asserted against its other user rather than duplicated.

    `oauth_service` computes `session_type == "browser" and client_id ==
    "xcelsior-web"` to choose the long browser TTL. If that string ever changes
    there, this fails rather than leaving two copies quietly disagreeing about
    who is first-party.
    """
    import inspect

    import oauth_service

    from routes._deps import _FIRST_PARTY_WEB_CLIENT_ID

    source = inspect.getsource(oauth_service)
    assert f'client_id == "{_FIRST_PARTY_WEB_CLIENT_ID}"' in source, (
        "the first-party client id in routes/_deps.py no longer matches the one "
        "oauth_service uses to decide token lifetime — the two have drifted"
    )


# --------------------------------------------------------------------------
# Mode behaviour
# --------------------------------------------------------------------------


def test_shadow_mode_changes_nothing(monkeypatch):
    """The default must be behaviourally identical to the feature being absent."""
    from fastapi import HTTPException

    from routes import _deps

    monkeypatch.setattr(_deps, "_CONNECTOR_SCOPE_MODE", "shadow")
    try:
        _deps._require_scope(_connector_token(["instances:read"]), "billing:write")
    except HTTPException as exc:  # pragma: no cover - the failure we are asserting against
        pytest.fail(f"shadow mode refused a request it must only observe: {exc.detail}")


def test_shadow_mode_actually_records_what_it_would_refuse(monkeypatch, caplog):
    """Calibration, and the reason this file exists.

    `_record_connector_scope_shadow` catches every exception so that observation
    can never break traffic. That safety makes silence indistinguishable from
    health: a missing logger or a renamed key would emit nothing and look fine.
    """
    from routes import _deps

    monkeypatch.setattr(_deps, "_CONNECTOR_SCOPE_MODE", "shadow")
    with caplog.at_level(logging.INFO, logger="xcelsior"):
        _deps._require_scope(_connector_token(["instances:read"]), "billing:write")

    records = [r.getMessage() for r in caplog.records if "connector_scope.shadow" in r.getMessage()]
    assert records, (
        "shadow mode recorded nothing for a token missing the required scope — "
        "the observation is silently failing, which is the one outcome worse "
        "than not having it"
    )
    assert "billing:write" in records[0], "the log does not name the missing scope"
    assert "some-third-party-connector" in records[0], "the log does not name the client"


def test_shadow_mode_is_quiet_when_the_token_holds_the_scope(monkeypatch, caplog):
    """The other half. A shadow that fires on everything measures nothing."""
    from routes import _deps

    monkeypatch.setattr(_deps, "_CONNECTOR_SCOPE_MODE", "shadow")
    with caplog.at_level(logging.INFO, logger="xcelsior"):
        _deps._require_scope(_connector_token(["billing:write"]), "billing:write")

    assert not [r for r in caplog.records if "connector_scope.shadow" in r.getMessage()], (
        "shadow mode logged a would-refuse for a token that holds the scope"
    )


def test_enforce_mode_refuses_a_connector_token_missing_the_scope(monkeypatch):
    """What the cutover buys, asserted now so the flip is a config change."""
    from fastapi import HTTPException

    from routes import _deps

    monkeypatch.setattr(_deps, "_CONNECTOR_SCOPE_MODE", "enforce")
    with pytest.raises(HTTPException) as excinfo:
        _deps._require_scope(_connector_token(["instances:read"]), "billing:write")
    assert excinfo.value.status_code == 403


def test_enforce_mode_admits_a_connector_token_that_holds_the_scope(monkeypatch):
    """A guard that refuses everything is not enforcement."""
    from routes import _deps

    monkeypatch.setattr(_deps, "_CONNECTOR_SCOPE_MODE", "enforce")
    _deps._require_scope(_connector_token(["billing:write", "instances:read"]), "billing:write")


@pytest.mark.parametrize("mode", ["shadow", "enforce", "off"])
def test_a_browser_session_is_never_gated_in_any_mode(mode, monkeypatch):
    """The regression that would take the dashboard down.

    A browser session carries OIDC identity scopes and no API scopes. If any
    mode ever treats it as a scoped credential, every logged-in page 403s.
    """
    from routes import _deps

    monkeypatch.setattr(_deps, "_CONNECTOR_SCOPE_MODE", mode)
    _deps._require_scope(_browser_session(), "billing:write")


def test_the_default_mode_is_enforce():
    """The cutover, asserted so that silently falling back to `shadow` fails.

    This file first asserted the opposite — that the default must *not* be
    `enforce`, on the grounds that flipping it would refuse connector traffic
    wherever the routes' 36 demanded scopes exceeded Quick Connect's 14. That
    reasoning rested on a bad measurement: the 36 came from a TypeScript type
    union rather than from tool requirements, and it read `anyOf` as `allOf`.

    Measured properly, exactly two published tools cannot be satisfied by a
    Quick Connect token, both needing `billing:write`, and both are already
    refused at the MCP layer. So enforcement removes no working capability — it
    closes the direct-REST path that route scoping never reached.

    A regression to `shadow` would be silent: every request would still succeed,
    and the enforcement would simply stop happening.
    """
    from routes import _deps

    assert _deps._CONNECTOR_SCOPE_MODE == "enforce", (
        f"the default connector scope mode is {_deps._CONNECTOR_SCOPE_MODE!r}, "
        "not 'enforce' — a third-party connector token's scopes are being "
        "treated as decorative again"
    )


def test_the_tools_quick_connect_cannot_satisfy_are_the_expected_ones():
    """The blast radius, pinned rather than described.

    If a third tool ever needs a scope Quick Connect withholds, enforcement
    starts refusing it and this fails — which is the intended way to find out,
    rather than from a user whose agent stopped working.
    """
    import json
    import pathlib
    import re

    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    repo = pathlib.Path(__file__).resolve().parent.parent
    src = (repo / "mcp" / "src" / "auth" / "scopes.ts").read_text(encoding="utf-8")
    published = {t["name"] for t in json.loads((repo / "mcp" / "tool-surface.json").read_text())["tools"]}
    held = set(MCP_QUICK_CONNECT_SCOPES)

    unsatisfiable = set()
    for match in re.finditer(r"(\w+)\s*:\s*\{\s*(allOf|anyOf)\s*:\s*\[([^\]]*)\]", src):
        tool, kind, body = match.group(1), match.group(2), match.group(3)
        scopes = set(re.findall(r'"([a-z]+:[a-z]+)"', body))
        if not scopes or tool not in published:
            continue
        satisfied = scopes <= held if kind == "allOf" else bool(scopes & held)
        if not satisfied:
            unsatisfiable.add(tool)

    # The two billing writes this cutover was measured against, plus the two SSH
    # key-management tools added 2026-08-15. That addition is the event this test
    # was written to surface, and the answer was to accept it rather than widen
    # the token: `GET /api/ssh/keys` says the split is deliberate — "Quick
    # Connect holds `ssh:write` and not `ssh:read`, so it registers and does not
    # enumerate" — because enumerating tells a connector which other machines and
    # people hold shell access. `delete_ssh_key` requires read as well as write
    # so it lands on the same side: revoking what you cannot list means acting on
    # an id from elsewhere, on a call that disconnects live sessions.
    #
    # **A connector agent can register a key and cannot list or revoke one.**
    # That is the cost, and it is written here so the next person meets it as a
    # decision rather than as a refusal in production.
    assert unsatisfiable == {
        "top_up_wallet",
        "configure_auto_topup",
        "list_ssh_keys",
        "delete_ssh_key",
    }, (
        f"the set of published tools a Quick Connect token cannot use is "
        f"{sorted(unsatisfiable)}. Adding to it widens what a connector agent "
        "cannot do — record why here, in the same commit."
    )
