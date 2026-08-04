"""`scopes_supported` must not advertise operator authority.

The MCP authorization spec makes `scopes_supported` load-bearing in a way that
is easy to miss. Its Scope Selection Strategy says a client that receives no
`scope` parameter in a `WWW-Authenticate` challenge should

    use all scopes defined in `scopes_supported` from the Protected Resource
    Metadata document

and the field itself "is intended to represent the **minimal** set of scopes
necessary for basic functionality". Publishing everything is named explicitly as
a common mistake under Scope Minimization.

So this list is not documentation — it is the default ask. Advertising
`hosts:evict`, `hosts:operate`, `hosts:fleet`, `control_plane:read` and
`control_plane:operate` meant an ordinary MCP client's first connection
presented the user with a consent screen requesting platform-operator authority
in order to, say, list their instances. Users who decline that dialog decline
the product; users who accept it hold a token whose compromise is
platform-wide.

The scopes remain enforced, remain in `SCOPE_DESCRIPTIONS` so a consent screen
can render them when an admin deliberately grants one, and remain refused to
non-admins by `oauth_delegation.assert_delegable`. They are simply not the
opening request.

**Both directions are asserted.** Advertising nothing would satisfy the rule
above while breaking discovery for every client, so the baseline scopes are
required to be present too.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

from oauth_delegation import OPERATOR_SCOPES  # noqa: E402

#: Scopes an ordinary client needs on first connection. Named individually so
#: removing one is visible in review rather than showing up as a count change.
BASELINE = {
    "profile",
    "email",
    "offline_access",
    "instances:read",
    "gpu:read",
    "marketplace:read",
}


def _advertised() -> list[str]:
    """`scopes_supported` as the metadata endpoint actually serves it.

    Read from the live response rather than from a constant, because the
    constant is not what a client sees — a middleware or a serialiser could
    change it between here and the wire.
    """
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)
    r = client.get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200, r.text
    return list(r.json().get("scopes_supported") or [])


def test_no_operator_scope_is_advertised_as_a_baseline():
    """The load-bearing rule."""
    advertised = set(_advertised())
    leaked = sorted(advertised & OPERATOR_SCOPES)
    assert not leaked, (
        f"{leaked} are advertised in scopes_supported. A client with no scope "
        "challenge requests everything listed there, so this asks every user to "
        "grant platform-operator authority on first connect. Operator scopes "
        "stay enforced and describable; they are not a baseline."
    )


def test_the_baseline_scopes_are_still_advertised():
    """The calibration control.

    An empty `scopes_supported` satisfies the rule above and breaks discovery
    for every client — which is how this gets 'fixed' in the wrong direction.
    """
    advertised = set(_advertised())
    missing = sorted(BASELINE - advertised)
    assert not missing, (
        f"{missing} are no longer advertised; a client following the spec's "
        "scope-selection fallback could not request them"
    )


def test_the_advertised_set_is_not_empty_or_everything():
    """A crude shape check, so neither extreme passes silently."""
    advertised = _advertised()
    assert 5 < len(advertised) < 40, (
        f"scopes_supported has {len(advertised)} entries, which is either a "
        "stub or the whole catalogue"
    )


def test_every_advertised_scope_has_a_description():
    """A scope a user is asked to grant must be renderable on a consent screen.

    An advertised scope with no description shows the user a bare identifier and
    asks them to approve it, which is not consent.
    """
    from oauth_service import SCOPE_DESCRIPTIONS

    #: OIDC identity scopes are defined by OpenID Connect, not by this server's
    #: capability vocabulary, and clients render them from their own catalogues.
    OIDC = {"openid", "profile", "email", "offline_access"}

    undescribed = sorted(set(_advertised()) - set(SCOPE_DESCRIPTIONS) - OIDC)
    assert not undescribed, (
        f"{undescribed} are advertised but have no entry in SCOPE_DESCRIPTIONS, "
        "so a consent screen can only show the raw string"
    )
