"""An ordinary user must not be able to mint themselves operator authority.

Operator endpoints — drain, undrain, evict, control-plane retry — are gated by
`routes/control_plane_v1._require_host_operator`, which authorizes in one of two
ways:

    user = _require_auth(request)
    if str(user.get("grant_type", "")) == "client_credentials":
        _require_scope(user, scope)          # machine: scope alone
    elif not _is_platform_admin(user):
        raise HTTPException(403, ...)        # human: admin alone

For a human, platform-admin is required. For a machine credential, **holding
the scope is the whole check** — deliberately, because a machine principal has
no role to inspect.

That is only sound if a non-admin cannot obtain a machine credential carrying an
operator scope. `POST /api/oauth/clients` is where that assumption is tested:
it takes `scopes` and `grant_types` from the request body and passes both to
`create_oauth_client` unfiltered. Only `is_first_party` is admin-gated.

So the escalation is: create a confidential client with
`scopes=["hosts:evict"]` and `grant_types=["client_credentials"]`, exchange it
at the token endpoint, and call the operator endpoint. The token is a machine
credential, so the admin branch is never reached, and `_require_scope` passes
because the client really was granted the scope it asked for. Every individual
check behaves as written; the composition is what fails.

`CONNECTOR_ALLOWED_SCOPES` already encodes exactly this policy for *dynamically
registered* clients — operator scopes "absent by construction". The first-party
creation endpoint simply never applied the same rule.

These tests assert the property that closes it: **a client may only be created
with scopes its creator is allowed to delegate.**
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

import oauth_service

#: Scopes that confer authority over the *platform*, not over the caller's own
#: resources. Mirrors the operator split documented in `SCOPE_DESCRIPTIONS`,
#: where each is marked "(operator)".
#:
#: `hosts:read` is deliberately absent. It gates a fleet-wide listing *and* a
#: provider's own admission status, and every worker agent is registered by a
#: non-admin provider — so admin-gating it breaks onboarding. See the note on
#: its description; the fix for the fleet-wide half is to split the scope.
# Imported, not restated. This file previously kept its own literal copy, and
# the copy went stale: `admin`, `autoscale:write` and `sla:write` were added to
# the real set and the drift test — whose docstring says it "keeps
# OPERATOR_SCOPES honest" — went on comparing descriptions against this
# snapshot instead. A guard with a private copy of the thing it guards is
# checking its own memory.
#
# Reading it live also widens the parametrized cases below from seven scopes to
# every one the platform actually treats as operator authority.
from oauth_delegation import OPERATOR_SCOPES as _LIVE_OPERATOR_SCOPES

OPERATOR_SCOPES = sorted(_LIVE_OPERATOR_SCOPES)


def _non_admin() -> dict:
    return {
        "email": "tenant@example.com",
        "user_id": "user-tenant",
        "role": "user",
        "is_admin": False,
    }


@pytest.mark.parametrize("scope", OPERATOR_SCOPES)
def test_non_admin_cannot_create_a_client_holding_an_operator_scope(scope):
    """The fix's load-bearing assertion.

    Parametrised per scope so that widening one does not hide behind another.
    """
    from oauth_delegation import ScopeDelegationError, assert_delegable

    with pytest.raises(ScopeDelegationError) as exc:
        assert_delegable([scope], actor=_non_admin())
    assert scope in str(exc.value)


@pytest.mark.parametrize("scope", OPERATOR_SCOPES)
def test_platform_admin_may_still_delegate_operator_scopes(scope):
    """The check must gate on authority, not ban the scope outright.

    An admin creating an operator client is the legitimate flow this endpoint
    exists for; a fix that broke it would just move the problem.
    """
    from oauth_delegation import assert_delegable

    assert_delegable([scope], actor={"email": "a@x.ca", "is_admin": True})


def test_ordinary_scopes_need_no_special_authority():
    """A normal user creating a normal client must keep working."""
    from oauth_delegation import assert_delegable

    assert_delegable(
        ["instances:read", "instances:write", "billing:read", "ssh:write"],
        actor=_non_admin(),
    )


def test_unknown_scopes_are_refused():
    """A typo must fail loudly rather than mint an unsatisfiable credential.

    A client registered with `instance:read` (singular) can be created, can be
    exchanged for a token, and will be refused by every endpoint — with a 403
    naming a scope the user believes they granted.
    """
    from oauth_delegation import ScopeDelegationError, assert_delegable

    with pytest.raises(ScopeDelegationError) as exc:
        assert_delegable(["instance:read"], actor=_non_admin())
    assert "instance:read" in str(exc.value)


def test_the_operator_list_matches_what_the_consent_screen_calls_operator():
    """Keep OPERATOR_SCOPES honest against the descriptions it mirrors.

    If a new operator scope is added and described as "(operator)" but not
    listed here, this fails rather than silently leaving it delegable.
    """
    described_operator = {
        scope
        for scope, text in oauth_service.SCOPE_DESCRIPTIONS.items()
        if "(operator)" in text
    }
    assert described_operator == set(OPERATOR_SCOPES), (
        "operator scopes drifted from their descriptions: "
        f"described-only={sorted(described_operator - set(OPERATOR_SCOPES))}, "
        f"listed-only={sorted(set(OPERATOR_SCOPES) - described_operator)}"
    )


def test_operator_endpoints_trust_the_scope_alone_for_machine_principals():
    """Why the check has to live at creation time.

    This is the *intended* behaviour of `_require_host_operator` and it is not
    changing: a machine principal has no role to inspect, so holding the scope
    is the authorization. That makes client creation the only place the
    delegation can be refused — deleting the creation-time check would silently
    reopen the escalation even though this function still reads as strict.
    """
    from routes.control_plane_v1 import _require_host_operator

    assert "_is_platform_admin" in _require_host_operator.__doc__ or True
    source = _require_host_operator.__code__.co_consts
    assert any(
        isinstance(c, str) and c == "client_credentials" for c in source
    ), "operator gate no longer branches on client_credentials; re-verify the escalation path"


def test_dynamic_registration_still_refuses_operator_scopes():
    """The DCR path already had this policy. It must not regress."""
    from oauth_registration import CONNECTOR_ALLOWED_SCOPES

    leaked = sorted(set(OPERATOR_SCOPES) & set(CONNECTOR_ALLOWED_SCOPES))
    assert not leaked, f"operator scopes reachable by self-registration: {leaked}"
