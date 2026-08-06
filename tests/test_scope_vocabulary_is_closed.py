"""A scope nobody can describe must not reach a consent screen.

`assert_delegable` guarded scope *writes* with a denylist: it refused
`OPERATOR_SCOPES` and let everything else through. So an authenticated
non-admin could create an OAuth client holding `totally:invented` — or, the
version that matters:

    "Full access to your account - this is safe and standard"

An invented scope grants nothing. Enforcement is membership (`scope in
granted`), so a string like that can never satisfy a check. **The damage is at
consent, not at enforcement.** `oauth_service.describe_scope` falls back to
rendering a scope *as itself* when it has no description:

    def describe_scope(scope: str) -> str:
        return SCOPE_DESCRIPTIONS.get(scope, scope)

So an unknown scope becomes attacker-chosen prose on a first-party
authorization page, presented as a permission the user is about to grant. The
victim need not be the attacker: OAuth is built so any user may authorize any
registered client, and the authorize link carries the client id.

**Dynamic registration was already closed against this.**
`oauth_registration.CONNECTOR_ALLOWED_SCOPES` is a fixed allowlist and
`POST /oauth/register` rejects anything outside it. The authenticated
client-creation path — `POST /api/oauth/clients`, through
`db.update_client_in_workspace` — was not, which is why one door being shut did
not close the room.

**Admins are checked too**, and that is deliberate rather than strict. Adding a
scope means giving it a description a user can consent to, which is a code
change. An admin typing `instances:writes` should be refused here rather than
create a phantom scope that nothing enforces and no screen can explain.

**The vocabulary is derived, not restated.** `known_scopes()` is the union of
`SCOPE_DESCRIPTIONS`, `OPERATOR_SCOPES` and `SYSTEM_ALLOWED_SCOPES` — a fourth
hand-kept list is how a check like this starts refusing something legitimate.
Verified against production before landing: every scope held by every existing
client is inside it, so no client breaks on its next edit.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

from oauth_delegation import (  # noqa: E402
    OPERATOR_SCOPES,
    ScopeDelegationError,
    assert_delegable,
    known_scopes,
)

USER = {"user_id": "u_probe", "is_admin": False, "role": "user"}
ADMIN = {"user_id": "u_admin", "is_admin": True, "role": "admin"}

#: The one that is not a typo.
PHISHING = "Full access to your account - this is safe and standard"


def test_an_invented_scope_is_refused():
    """The headline."""
    with pytest.raises(ScopeDelegationError) as excinfo:
        assert_delegable(["totally:invented"], actor=USER)
    assert "totally:invented" in str(excinfo.value)


def test_prose_dressed_as_a_scope_is_refused():
    """Why this is a consent defect rather than a tidiness one.

    Stored, this string is rendered verbatim on the authorization page as
    something the user is agreeing to grant.
    """
    with pytest.raises(ScopeDelegationError):
        assert_delegable([PHISHING], actor=USER)


def test_a_real_scope_is_still_accepted():
    """The calibration control.

    A guard that refuses everything satisfies every assertion above and breaks
    client creation entirely.
    """
    assert_delegable(["instances:read", "billing:read"], actor=USER)


def test_an_admin_typo_is_refused_too():
    """Admins are not exempt, on purpose.

    `instances:writes` is one character from a real scope. Accepted, it becomes
    a phantom: nothing enforces it, no screen can describe it, and the client
    silently holds nothing.
    """
    with pytest.raises(ScopeDelegationError) as excinfo:
        assert_delegable(["instances:writes"], actor=ADMIN)
    assert "instances:writes" in str(excinfo.value)


def test_the_operator_rule_still_applies():
    """The vocabulary check must not have replaced the authority check.

    `hosts:evict` is a real scope, so it passes the vocabulary. It must still be
    refused to a non-admin.
    """
    with pytest.raises(ScopeDelegationError) as excinfo:
        assert_delegable(["hosts:evict"], actor=USER)
    assert "administrator" in str(excinfo.value)
    # And an admin may still grant it.
    assert_delegable(["hosts:evict"], actor=ADMIN)


def test_the_vocabulary_is_derived_from_the_definitions():
    """A fourth hand-kept list is how this starts refusing real scopes.

    Every operator scope must be in the vocabulary — otherwise the authority
    check below could never be reached, because the vocabulary check would
    refuse first and report the wrong reason.
    """
    vocabulary = known_scopes()
    assert len(vocabulary) > 20, f"vocabulary collapsed to {len(vocabulary)}"
    missing = sorted(OPERATOR_SCOPES - vocabulary)
    assert not missing, (
        f"operator scopes absent from the vocabulary: {missing}. They would be "
        "refused as undefined rather than as requiring an administrator"
    )


def test_every_connector_scope_is_grantable():
    """Dynamic registration and this guard must agree.

    A scope `POST /oauth/register` accepts but `assert_delegable` refuses would
    let a client be created and then fail to be edited.
    """
    from oauth_registration import CONNECTOR_ALLOWED_SCOPES

    outside = sorted(set(CONNECTOR_ALLOWED_SCOPES) - known_scopes())
    assert not outside, (
        f"connector scopes the delegation guard would refuse: {outside}"
    )


def test_scopes_held_in_production_remain_grantable():
    """Checked against the live database before this landed.

    Every scope on every existing OAuth client, so none of them becomes
    un-editable. `api` is here because clients hold it; it grants nothing now
    that `_require_scope` has no wildcard, but refusing it would strand real
    clients for no gain.
    """
    observed = {
        "api", "billing:read", "email", "events:read", "gpu:read",
        "inference:read", "inference:write", "instances:operate",
        "instances:read", "instances:write", "marketplace:read",
        "mcp_actions:approve", "offline_access", "profile",
    }
    outside = sorted(observed - known_scopes())
    assert not outside, (
        f"scopes held by existing clients would now be refused: {outside}"
    )
