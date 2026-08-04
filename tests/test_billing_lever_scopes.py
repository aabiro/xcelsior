"""The billing levers require a scope, not merely a session.

Six money-adjacent routes authorized with

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

and nothing else. Authentication without authorization: any credential that can
log in reaches them, regardless of what that credential was granted.

That matters because of what the levers are:

* `POST /api/v2/billing/auto-topup` — sets the threshold and amount that charge
  a saved card unattended.
* `POST /api/billing/portal-session` — mints a Stripe Customer Portal session,
  which is full billing management in a browser.
* `POST /api/billing/setup-intent` — mints a `client_secret` for saving a card.
* `DELETE /api/billing/payment-methods/{id}` — detaches a card, which disables
  auto-top-up and strands a running workload when the wallet empties.
* `GET /api/billing/payment-methods`, `GET /api/v2/billing/auto-topup` — read
  the funding configuration.

The credential that makes this sharp is the one this product tells users to
create. A Quick Connect token narrowed to `instances:read` — pasted into an
agent for read-only monitoring — reached all six. The narrowing was decorative
on the billing surface.

**`_require_scope` is a no-op for interactive sessions**, so adding it changes
nothing for a browser user and everything for a machine credential. That is the
whole point: the dashboard keeps working, and a token that asked for
`instances:read` stops being able to reconfigure unattended spending.

Split by verb, because reading which cards exist and detaching one are not the
same act:

* `billing:read` — the two reads.
* `billing:write` — the four that move money, mint credentials, or change what
  gets charged.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

import routes.billing as billing_routes  # noqa: E402

#: handler name -> the scope it must require. Listed individually so removing
#: one is visible in review rather than showing up as a count change.
LEVERS = {
    "api_billing_configure_topup": "billing:write",
    "api_billing_portal_session": "billing:write",
    "api_billing_setup_intent": "billing:write",
    "api_billing_detach_payment_method": "billing:write",
    "api_billing_list_payment_methods": "billing:read",
    "api_billing_get_topup": "billing:read",
}


@pytest.mark.parametrize("handler,scope", sorted(LEVERS.items()))
def test_each_billing_lever_requires_its_scope(handler, scope):
    """Authentication is not authorization, applied to money."""
    fn = getattr(billing_routes, handler, None)
    assert fn is not None, f"{handler} no longer exists — was it renamed?"

    source = inspect.getsource(fn)
    assert f'_require_scope(user, "{scope}")' in source, (
        f"{handler} does not require {scope!r}, so any principal that can "
        "authenticate reaches it regardless of what its credential was granted "
        "— including a Quick Connect token narrowed to instances:read"
    )


@pytest.mark.parametrize("handler", sorted(LEVERS))
def test_each_lever_still_refuses_an_anonymous_caller(handler):
    """Adding authorization must not weaken authentication.

    The obvious tidy-up is to replace

        user = _get_current_user(request)
        if not user:
            raise HTTPException(401, "Not authenticated")

    with `_require_auth(request)`. It is wrong here, and this test exists
    because I made that change and an existing test caught it.

    `_require_auth` returns a synthetic principal with `is_admin: True` when
    `AUTH_REQUIRED` is false, which it is in every relaxed environment. So the
    swap would take these six routes from *refusing* an anonymous caller to
    *admitting them as an administrator* on any development or test deployment
    — a strict weakening, made while fixing an authorization gap, and invisible
    in production where `AUTH_REQUIRED` is true.

    The defect these routes had was missing authorization. Changing how they
    authenticate was not part of it.
    """
    source = inspect.getsource(getattr(billing_routes, handler))
    assert "_get_current_user(request)" in source, (
        f"{handler} no longer resolves its principal with _get_current_user. If "
        "this became _require_auth, the route now admits anonymous callers as "
        "admin wherever AUTH_REQUIRED is false"
    )
    assert 'raise HTTPException(401, "Not authenticated")' in source, (
        f"{handler} lost its explicit anonymous refusal"
    )


def test_require_scope_is_a_no_op_for_interactive_sessions():
    """The reason this is safe to add, asserted rather than assumed.

    If `_require_scope` refused browser sessions, adding it to six routes would
    break the dashboard — and the fix would be reverted rather than corrected.
    An interactive session carries OIDC identity scopes, which say nothing about
    API authority.
    """
    from routes._deps import _require_scope

    session = {
        "email": "human@example.com",
        "user_id": "u1",
        "auth_type": "oauth_access_token",
        "scopes": ["openid", "profile", "email"],
    }
    _require_scope(session, "billing:write")  # must not raise


def test_a_narrowed_machine_credential_is_refused():
    """The calibration control: the check must actually be capable of refusing.

    Without this, `_require_scope` being a no-op for *everything* would satisfy
    every assertion above.
    """
    from fastapi import HTTPException

    from routes._deps import _require_scope

    narrowed = {
        "email": "agent@example.com",
        "user_id": "u1",
        "auth_type": "agent_api_key",
        "scopes": ["instances:read"],
    }
    with pytest.raises(HTTPException) as excinfo:
        _require_scope(narrowed, "billing:write")
    assert excinfo.value.status_code == 403
