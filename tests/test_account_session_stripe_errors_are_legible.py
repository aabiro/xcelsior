"""A Stripe refusal on the payouts panel must reach the provider as a reason.

`POST /api/providers/{id}/account-session` mints the client secret for the
embedded Stripe Connect payouts panel. The route caught only `RuntimeError`,
and `create_account_session` let every `stripe.StripeError` escape. A detached
or deleted connected account therefore came back as a bare 500 with no
`detail`, and the panel showed `Request failed (HTTP 500)`.

Now Stripe errors map to two outcomes:

* **409** when Stripe no longer recognizes the stored account
  (`PermissionError`, or `InvalidRequestError` with `account_invalid` /
  `resource_missing`). The provider can act on this: restart onboarding.
* **502** for everything else. The fault is upstream, not the caller's.

The errors are real `stripe` exceptions, faked only at `AccountSession.create`.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402
import stripe  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402
from stripe_connect import AccountSessionError, StripeConnectManager  # noqa: E402

client = TestClient(app)

PROVIDER = {"provider_id": "p-acs", "stripe_account_id": "acct_detached", "status": "active"}


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


def _create_session_raising(err: Exception):
    mgr = StripeConnectManager.__new__(StripeConnectManager)
    with (
        patch("stripe_connect.STRIPE_ENABLED", True),
        patch("stripe_connect.stripe", stripe),
        patch.object(StripeConnectManager, "get_provider", return_value=PROVIDER),
        patch.object(stripe.AccountSession, "create", side_effect=err),
    ):
        return mgr.create_account_session("p-acs")


@pytest.mark.parametrize(
    "err",
    [
        stripe.PermissionError(
            "The provided key does not have access to account 'acct_detached'",
            http_status=403,
            code="account_invalid",
        ),
        stripe.InvalidRequestError(
            "No such account: 'acct_detached'", "account", code="resource_missing", http_status=404
        ),
        stripe.InvalidRequestError(
            "Account is invalid", "account", code="account_invalid", http_status=400
        ),
    ],
)
def test_unreachable_account_asks_for_onboarding(err):
    with pytest.raises(AccountSessionError) as exc:
        _create_session_raising(err)
    assert exc.value.status_code == 409
    assert "onboarding" in str(exc.value)
    assert exc.value.__cause__ is err


@pytest.mark.parametrize(
    "err",
    [
        stripe.InvalidRequestError("Bad component", "components", code="parameter_unknown"),
        stripe.APIConnectionError("Network down"),
        stripe.RateLimitError("Too many requests", http_status=429),
        stripe.AuthenticationError("Invalid API key", http_status=401),
    ],
)
def test_other_stripe_errors_are_upstream_failures(err):
    with pytest.raises(AccountSessionError) as exc:
        _create_session_raising(err)
    assert exc.value.status_code == 502
    assert str(exc.value)


def test_stripe_error_is_logged(caplog):
    err = stripe.InvalidRequestError(
        "No such account: 'acct_detached'", "account", code="resource_missing", http_status=404
    )
    with caplog.at_level(logging.WARNING, logger="xcelsior.stripe"):
        with pytest.raises(AccountSessionError):
            _create_session_raising(err)
    record = next(r for r in caplog.records if "AccountSession create failed" in r.getMessage())
    assert "p-acs" in record.getMessage()
    assert "resource_missing" in record.getMessage()


@pytest.mark.parametrize("status_code", [409, 502])
def test_route_returns_the_mapped_status_with_a_detail(status_code):
    err = AccountSessionError("Restart Stripe onboarding.", status_code=status_code)
    with patch.object(StripeConnectManager, "create_account_session", side_effect=err):
        r = client.post("/api/providers/p-acs/account-session", headers=_admin_headers())
    assert r.status_code == status_code, r.text[:300]
    assert "Restart Stripe onboarding." in r.text
