"""Owning a wallet is not the same as being allowed to read it.

`_require_customer_access` guards nineteen routes in `routes/billing.py`. It
asked who is calling and whose records they are, and never asked whether the
*credential* was allowed near billing at all. Eleven of the nineteen had no
`_require_scope` anywhere in them:

    GET  /api/billing/wallet/{customer_id}              balance
    GET  /api/billing/wallet/{customer_id}/history      every transaction
    GET  /api/billing/wallet/{customer_id}/depletion    burn rate, runway
    GET  /api/billing/usage/{customer_id}               usage summary
    GET  /api/billing/invoice/{customer_id}             invoice
    GET  /api/billing/invoices/{customer_id}            invoice list
    GET  /api/billing/invoice/{customer_id}/download    invoice export
    GET  /api/billing/crypto/deposit/{deposit_id}       deposit status
    POST /api/billing/wallet/{customer_id}/deposit      direct credit
    POST /api/billing/crypto/refresh/{deposit_id}       re-rate a deposit
    POST /api/pricing/reserve                           term commitment

So an agent key narrowed to `instances:read` — one a user reduced *on purpose*,
which is what Quick Connect ships — could still read the account's full
financial history and commit it to a reserved pricing term. Scope reduction was
real and had no effect, which is the same defect the `api` wildcard was and the
same one `239643b` fixed on the six write levers. This is the read side of that
surface.

**Fixed in the helper, not in eleven handlers.** The routes that move or commit
money already pass `billing_write=True`, so the required scope follows that flag
instead of a list of route names someone has to remember to extend — a list to
keep in sync is precisely how the gap survived being fixed next door.

**Not a severity claim about `api_deposit`.** That route is independently
defended: `_allow_direct_wallet_deposit` refuses it with a 403 by default, plus
a rate limit and a team billing-write check. It is in the list because it lacked
the scope, not because it was reachable.

**Both directions, and the third case that matters.** A missing scope must be
refused, a present scope must be admitted, and an interactive browser session —
which carries OIDC identity scopes and no API scopes — must be unaffected, or
this fix logs every human out of their own billing page.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from routes._deps import _require_customer_access  # noqa: E402

#: The eleven that had no scope check, by handler name.
PREVIOUSLY_UNSCOPED = {
    "api_get_wallet",
    "api_deposit",
    "api_wallet_history",
    "api_wallet_depletion",
    "api_usage_summary",
    "api_generate_invoice",
    "api_list_invoices",
    "api_download_invoice",
    "api_crypto_deposit_status",
    "api_crypto_refresh",
    "api_reserve_commitment",
}


class _Request:
    """Just enough Request for the guard: it only reads `state` and headers."""

    def __init__(self, user):
        self.state = type("S", (), {"user": user})()
        self.headers = {}
        self.cookies = {}
        self.query_params = {}
        self.url = type("U", (), {"path": "/api/billing/wallet/cus_probe"})()
        self.client = None
        self.method = "GET"


def _machine(scopes):
    """An agent-key principal: a machine credential carrying explicit scopes."""
    return {
        "user_id": "u_probe",
        "customer_id": "cus_probe",
        "auth_type": "agent_api_key",
        "scopes": list(scopes),
        "is_admin": False,
    }


def _call(user, *, billing_write=False):
    import routes._deps as deps

    request = _Request(user)
    original = deps._get_current_user
    deps._get_current_user = lambda _r: user
    try:
        return _require_customer_access(request, "cus_probe", billing_write=billing_write)
    finally:
        deps._get_current_user = original


def test_the_guard_asks_for_a_scope_at_all():
    """The fix, asserted where it lives rather than in eleven handlers."""
    source = inspect.getsource(_require_customer_access)
    assert "_require_scope" in source, (
        "_require_customer_access no longer checks scope; every customer-billing "
        "route it guards is authenticated and ownership-checked but not authorized"
    )
    assert '"billing:write" if billing_write else "billing:read"' in source, (
        "the required scope no longer follows the billing_write flag, so the "
        "read routes and the money-moving routes ask for the same thing"
    )


def test_a_read_scoped_credential_cannot_read_billing():
    """The headline: narrowing a token to instances:read must mean something."""
    with pytest.raises(HTTPException) as excinfo:
        _call(_machine(["instances:read"]))
    assert excinfo.value.status_code == 403
    assert "billing:read" in str(excinfo.value.detail)


def test_a_billing_read_credential_is_admitted():
    """The inverse. A guard that refuses everyone is not a guard."""
    user = _call(_machine(["billing:read"]))
    assert user["customer_id"] == "cus_probe"


def test_reading_does_not_require_the_write_scope():
    """`billing:read` alone must reach the reads, or the fix breaks every reader."""
    assert _call(_machine(["billing:read", "instances:read"]))


def test_a_read_credential_cannot_commit_money():
    """`POST /api/pricing/reserve` commits the customer to a pricing term.

    It passes `billing_write=True`, so `billing:read` must not reach it.
    """
    with pytest.raises(HTTPException) as excinfo:
        _call(_machine(["billing:read"]), billing_write=True)
    assert excinfo.value.status_code == 403
    assert "billing:write" in str(excinfo.value.detail)


def test_a_write_credential_may_commit_money():
    assert _call(_machine(["billing:write"]), billing_write=True)


def test_an_admin_machine_credential_is_still_bound_by_its_scopes():
    """Scope is checked before the admin bypass, on purpose.

    Being an admin says *whose* records you may see. It does not widen what a
    token may do — otherwise narrowing an admin's credential would be
    decorative, and the most powerful tokens would be the only unnarrowable
    ones.
    """
    admin = _machine(["instances:read"])
    admin["is_admin"] = True
    with pytest.raises(HTTPException) as excinfo:
        _call(admin)
    assert excinfo.value.status_code == 403


def test_a_browser_session_is_unaffected():
    """The regression this fix could most easily cause.

    Interactive sessions carry OIDC identity scopes — `profile`, `email`,
    `offline_access` — which say nothing about API authority. If those counted
    as "scope restricted", every human would be locked out of their own billing
    page. `_require_scope` no-ops for anything that is not a machine credential.
    """
    session = {
        "user_id": "u_probe",
        "customer_id": "cus_probe",
        "auth_type": "session",
        "grant_type": "authorization_code",
        "scopes": ["profile", "email", "offline_access"],
        "is_admin": False,
    }
    assert _call(session)
    assert _call(session, billing_write=True)


def test_every_route_the_helper_guards_is_now_covered():
    """No handler was left behind, and the count is pinned.

    Asserted against the live route table rather than the list above, so a
    twelfth route added tomorrow is covered by construction instead of by
    someone remembering to edit a set.
    """
    import ast
    import pathlib

    source = pathlib.Path(
        inspect.getsourcefile(_require_customer_access)
    ).resolve().parent / "billing.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    guarded = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "_require_customer_access" in ast.unparse(node)
    }
    assert PREVIOUSLY_UNSCOPED <= guarded, (
        "a route that used to be unscoped no longer calls the guard at all: "
        f"{sorted(PREVIOUSLY_UNSCOPED - guarded)}"
    )
    assert len(guarded) >= 19, (
        f"only {len(guarded)} routes use the guard; it was 19. A route that "
        "stopped using it is a route that lost the scope check too."
    )
