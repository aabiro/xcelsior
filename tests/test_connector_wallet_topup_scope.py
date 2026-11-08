"""An OAuth connector may top up the user's wallet, and nothing else of billing.

The August design (`5c7b8d5d`) made `top_up_wallet` gated by consent already
given — the card was added in the dashboard, the user grants the capability on
a consent page that says what it does, Stripe enforces the rest — and it only
ever moves money *into* the user's own wallet. But the capability was
`billing:write`, which connectors were never allowed to hold, so no connector
could reach the tools that design was for.

Granting connectors `billing:write` would have handed them far more than
topping up. Of the fifteen routes that demand it, nine have nothing to do with
it: the Stripe billing portal ("full billing management in a browser"), the
SetupIntent that attaches a new card, detaching a card ("a running workload
hard-stops"), the browser PaymentIntent, four PayPal order routes, and the SCA
resume that hands back a payment-completing secret.

So topping up is its own scope, `billing:topup`:

* it covers the routes the four top-up tools call, and the Bitcoin refresh on
  the same rail;
* `billing:write` contains it (`IMPLIED_SCOPES`), so every credential that
  already tops up keeps doing so without being reissued;
* it runs one way: holding `billing:topup` satisfies nothing that asks for
  `billing:write`;
* a connector gets it only by leaving it ticked on the consent page, under its
  own "Payments" heading, in words that say a card will be charged;
* it is never in the connector default set and never on the Quick Connect key,
  which the dashboard issues without a consent page.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import re

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi import HTTPException  # noqa: E402

import routes.billing as billing_routes  # noqa: E402
from oauth_registration import CONNECTOR_ALLOWED_SCOPES, CONNECTOR_DEFAULT_SCOPES  # noqa: E402
from oauth_service import MCP_QUICK_CONNECT_SCOPES, MCP_RESOURCE_AUDIENCE, describe_scope  # noqa: E402
from routes import _deps  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Routes that only put money into the caller's own wallet.
FUNDING_HANDLERS = {
    "api_billing_manual_topup",
    "api_billing_configure_topup",
    "api_billing_auto_topup_plan",
    "api_billing_execute_auto_topup_plan",
    "api_crypto_deposit",
    "api_crypto_refresh",
    "api_ln_create_deposit",
}

#: The rest of `billing:write`, which a connector must not reach.
WRITE_ONLY_HANDLERS = {
    "api_billing_portal_session",
    "api_billing_setup_intent",
    "api_billing_detach_payment_method",
    "api_create_payment_intent",
    "api_paypal_create_order",
    "api_paypal_capture_order",
    "api_paypal_marketplace_create_order",
    "api_paypal_marketplace_capture_order",
}


def _connector_token(scopes: list[str]) -> dict:
    """A third-party authorization-code token, as `_deps` sees one."""
    return {
        "email": "human@xcelsior.ca",
        "user_id": "human-1",
        "auth_type": "oauth_access_token",
        "session_type": "browser",
        "client_id": "some-third-party-connector",
        "scopes": list(scopes),
    }


# --------------------------------------------------------------------------
# The implication
# --------------------------------------------------------------------------


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(_deps, "_CONNECTOR_SCOPE_MODE", "enforce")


def test_a_connector_holding_the_topup_scope_may_top_up(enforce):
    _deps._require_scope(_connector_token(["billing:topup"]), "billing:topup")


def test_billing_write_contains_the_topup_scope(enforce):
    """Nothing that already tops up has to be reissued."""
    _deps._require_scope(_connector_token(["billing:write"]), "billing:topup")


def test_the_topup_scope_does_not_contain_billing_write(enforce):
    with pytest.raises(HTTPException) as excinfo:
        _deps._require_scope(_connector_token(["billing:topup"]), "billing:write")
    assert excinfo.value.status_code == 403


def test_the_topup_scope_is_not_a_read_scope(enforce):
    with pytest.raises(HTTPException):
        _deps._require_scope(_connector_token(["billing:topup"]), "billing:read")


def test_the_refusal_names_what_the_token_holds_not_what_it_implies(enforce):
    with pytest.raises(HTTPException) as excinfo:
        _deps._require_scope(_connector_token(["billing:write"]), "instances:write")
    assert "granted: billing:write" in str(excinfo.value.detail)
    assert "billing:topup" not in str(excinfo.value.detail)


def test_the_mcp_server_applies_the_same_implication():
    """Two copies of one rule: a difference is a tool the API and the MCP layer
    disagree about, which shows up as a refusal nobody can explain."""
    src = (ROOT / "api" / "mcp" / "src" / "auth" / "scopes.ts").read_text(encoding="utf-8")
    block = re.search(r"export const IMPLIED_SCOPES[^=]*=\s*\{(.*?)\};", src, re.S)
    assert block, "IMPLIED_SCOPES is gone from api/mcp/src/auth/scopes.ts"
    ts = {
        broader: frozenset(re.findall(r'"([a-z_]+:[a-z_]+)"', body))
        for broader, body in re.findall(r'"([a-z_]+:[a-z_]+)"\s*:\s*\[([^\]]*)\]', block.group(1))
    }
    assert ts == _deps.IMPLIED_SCOPES


# --------------------------------------------------------------------------
# Which routes it reaches
# --------------------------------------------------------------------------


@pytest.mark.parametrize("handler", sorted(FUNDING_HANDLERS))
def test_each_funding_route_asks_for_the_topup_scope(handler):
    source = inspect.getsource(getattr(billing_routes, handler))
    assert (
        '_require_scope(user, "billing:topup")' in source or "funds_wallet=True" in source
    ), f"{handler} no longer accepts billing:topup, so a connector cannot use the tool that calls it"
    assert '"billing:write"' not in source and "billing_write=True" not in source, (
        f"{handler} still demands billing:write, which no connector can hold"
    )


@pytest.mark.parametrize("handler", sorted(WRITE_ONLY_HANDLERS))
def test_the_rest_of_billing_write_stays_out_of_reach(handler):
    source = inspect.getsource(getattr(billing_routes, handler))
    assert '_require_scope(user, "billing:write")' in source, (
        f"{handler} no longer requires billing:write — if it now accepts "
        "billing:topup, a third-party connector can reach it"
    )
    assert "billing:topup" not in source and "funds_wallet" not in source


def test_the_sca_resume_stays_behind_billing_write():
    """It hands back a secret that completes a charge; the browser is its path."""
    source = inspect.getsource(billing_routes.api_billing_resume_verification)
    assert "billing_write=True" in source
    assert "funds_wallet" not in source


# --------------------------------------------------------------------------
# Who can hold it
# --------------------------------------------------------------------------


def test_a_connector_may_be_granted_the_topup_scope_and_not_billing_write():
    assert "billing:topup" in CONNECTOR_ALLOWED_SCOPES
    assert "billing:write" not in CONNECTOR_ALLOWED_SCOPES


def test_only_a_connector_that_asks_gets_it():
    assert "billing:topup" not in CONNECTOR_DEFAULT_SCOPES


def test_the_quick_connect_key_cannot_top_up():
    """Issued without a consent page, so it carries neither (August, link 2)."""
    assert "billing:topup" not in MCP_QUICK_CONNECT_SCOPES
    assert "billing:write" not in MCP_QUICK_CONNECT_SCOPES


def test_the_description_says_a_card_will_be_charged():
    text = describe_scope("billing:topup")
    assert text != "billing:topup"
    assert "charge your saved card" in text.lower()


# --------------------------------------------------------------------------
# The consent page
# --------------------------------------------------------------------------


def _consent_page(scopes: list[str]) -> str:
    from routes.auth import _oauth_consent_page

    return _oauth_consent_page(
        client={"client_id": "claude-ai", "client_name": "Claude", "client_uri": "https://claude.ai"},
        user={"email": "human@xcelsior.ca", "user_id": "human-1"},
        scopes=scopes,
        resource=MCP_RESOURCE_AUDIENCE,
        consent_key="k",
    )


def test_consent_shows_topping_up_as_an_optional_payment():
    page = _consent_page(["instances:read", "billing:read", "billing:topup"])
    assert "Payments · Optional" in page
    assert "Charge your saved card" in page
    checkbox = re.search(r'<input type="checkbox" name="scope:billing:topup"[^>]*>', page)
    assert checkbox, "billing:topup has no checkbox on the consent page"
    assert "disabled" not in checkbox.group(0), "the user cannot untick topping up"
    assert 'type="hidden" name="scope:billing:topup"' not in page


def test_consent_never_files_topping_up_under_account_and_connection():
    page = _consent_page(["billing:topup"])
    assert "Account and connection" not in page
