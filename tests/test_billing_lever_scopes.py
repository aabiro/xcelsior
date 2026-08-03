"""The billing levers require `billing:write`, not merely a session.

P0: *`setup-intent` and `portal-session` move from `_get_current_user` to
`_require_scope(billing:write)`.*

Both endpoints authorized with `_get_current_user(...)` and `if not user: raise
401` — authentication without authorization, the §0.1 defect applied to money.
Any credential that could log in could mint a Stripe SetupIntent or a Customer
Portal session, including a token deliberately narrowed to `billing:read` so an
agent could watch spend without touching payment methods.

What each one actually confers:

* **`setup-intent`** returns a `client_secret` for `confirmCardSetup`. The card
  it saves becomes selectable for auto-top-up, so it is the entry point to
  charging that card off-session later.
* **`portal-session`** returns a Stripe Customer Portal URL. From there a holder
  can change payment methods, view invoices, and cancel subscriptions.

Neither is readable-only, so `billing:read` must not reach either. P1 adds a
manual top-up on the same footing — it charges a real card, so it is
`billing:write` from the start rather than inheriting the looser guard.

The refusal below is keyed on the **adjacent** scope. Refusing a credential
holding `instances:read` proves little, because it shares no surface with
billing; refusing one holding `billing:read` is the case that matters, since
those two are the pair most likely to be collapsed by a later simplification.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest
from fastapi import HTTPException

import routes.billing as billing_routes
from routes._deps import _require_scope

#: The levers P0 names. Listed individually so removing one is visible in
#: review rather than showing up as a count change.
BILLING_LEVERS = ["api_billing_setup_intent", "api_billing_portal_session"]


@pytest.mark.parametrize("handler", BILLING_LEVERS)
def test_billing_levers_require_billing_write(handler):
    """Authentication is not authorization — the §0.1 defect, applied to money."""
    source = inspect.getsource(getattr(billing_routes, handler))
    assert '_require_scope(user, "billing:write")' in source, (
        f"{handler} does not require `billing:write`, so any principal that can "
        "authenticate can reach a Stripe payment surface regardless of what its "
        "credential was granted"
    )


@pytest.mark.parametrize("handler", BILLING_LEVERS)
def test_billing_levers_still_reject_the_unauthenticated(handler):
    """The scope is additional to authentication, never a replacement."""
    source = inspect.getsource(getattr(billing_routes, handler))
    assert "_require_auth" in source or "Not authenticated" in source, (
        f"{handler} no longer establishes the caller"
    )


def test_billing_read_does_not_satisfy_billing_write():
    """The adjacent refusal, and the property that makes a read token useful.

    An agent granted `billing:read` so it can watch spend must not be able to
    save a card or open the customer portal. If `billing:read` ever satisfies
    these, issuing a read-only billing credential means nothing.
    """
    reader = {
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "scopes": ["billing:read"],
    }
    with pytest.raises(HTTPException) as exc:
        _require_scope(reader, "billing:write")
    assert exc.value.status_code == 403
    assert "billing:write" in str(exc.value.detail), (
        "the refusal does not name the missing scope, so a caller cannot tell "
        "what to request"
    )


def test_billing_write_is_admitted():
    """The matching positive.

    A refusal alone would pass just as well if the endpoint were unreachable —
    which is the sealed-scope failure. Granting the scope must open the door.
    """
    writer = {
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "scopes": ["billing:read", "billing:write"],
    }
    _require_scope(writer, "billing:write")  # must not raise


def test_an_interactive_session_is_unaffected():
    """Browser sessions carry OIDC claims, not API scopes.

    `_require_scope` is a no-op for interactive principals. Without this, moving
    the levers behind a scope would lock every human out of their own billing
    settings — the failure that broke 122 tests when `_require_scope` was first
    tightened.
    """
    session = {"email": "human@example.com", "auth_type": "session"}
    _require_scope(session, "billing:write")  # must not raise


def test_billing_write_is_grantable():
    """A scope enforced but unissuable seals the endpoint instead of guarding it."""
    import oauth_service

    assert "billing:write" in oauth_service.SCOPE_DESCRIPTIONS


def test_billing_write_is_not_a_silent_connector_default():
    """A connector that asked for nothing must not arrive able to save a card."""
    from oauth_registration import CONNECTOR_DEFAULT_SCOPES

    assert "billing:write" not in CONNECTOR_DEFAULT_SCOPES
