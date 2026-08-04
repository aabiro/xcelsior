"""The Stripe Connect router denies by default.

`routes/stripe_connect_v2.py` mounts eleven endpoints and none of them reached
an auth call. Verified by request rather than by reading: `GET
/api/connect/accounts` returned 200 with no credential, and the POSTs failed on
schema validation (422) rather than authorization, so a well-formed body would
have proceeded. The module logs `Stripe Connect ENABLED (mode=live)`.

That is unauthenticated *write* to Stripe — creating connected accounts and
products — and unauthenticated enumeration of Connect tenants.

**Deny by default, then reopen with justification.** Leaving the surface open
while the authorization model is designed has the risk backwards: the design
work is unbounded in time, the exposure is not. A router-level dependency now
refuses everything, and each exemption is named with the reason it is safe.

Two exemptions, both verified in the source rather than assumed:

* **`POST /api/connect/webhooks`** — requires a `stripe-signature` header,
  rejects the request when it is absent, and `parse_event_notification` verifies
  it against the endpoint secret. The signature *is* the credential; Stripe does
  not send a bearer, so requiring one would break webhook delivery while adding
  nothing.
* **The three HTML pages** — `/connect/dashboard`, `/connect/storefront`,
  `/connect/success` return a static constant and render no account data. They
  fetch with the browser afterwards, which is where the credential belongs.

**Known consequence, stated rather than discovered later:** those pages call
`/api/connect/*` with plain `fetch()` and no `Authorization` header, so gating
the JSON endpoints breaks the prototype dashboard until it sends credentials.
That is the correct trade against leaving unauthenticated account creation
reachable, and it is why the pages stay public while the APIs do not.

The refusals below run under `enforced_auth`, because with the suite's relaxed
default an anonymous caller resolves to a synthetic admin and every assertion
here would pass without exercising anything.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

#: Every JSON endpoint on the router. Named individually so removing one from
#: the gate is visible in review rather than showing up as a count change.
GUARDED = [
    ("POST", "/api/connect/accounts", {"display_name": "probe"}),
    ("GET", "/api/connect/accounts", None),
    ("GET", "/api/connect/accounts/acct_probe/status", None),
    ("GET", "/api/connect/accounts/acct_probe/onboarding-link", None),
    ("POST", "/api/connect/products", {"account_id": "acct_probe", "name": "p", "unit_amount": 100}),
    ("GET", "/api/connect/products", None),
    ("POST", "/api/connect/checkout", {"account_id": "acct_probe", "price_id": "price_probe"}),
]

#: Reachable without a credential, each for a verified reason.
PUBLIC = [
    ("GET", "/connect/dashboard", "static HTML shell; fetches with credentials"),
    ("GET", "/connect/storefront", "static HTML shell"),
    ("GET", "/connect/success", "static HTML shell; post-checkout return"),
]


@pytest.mark.enforced_auth
@pytest.mark.parametrize("method,path,body", GUARDED)
def test_connect_json_endpoints_refuse_an_anonymous_caller(
    method, path, body, auth_enforced
):
    """The gate. A 422 here would mean the body was parsed before authorization."""
    response = client.request(method, path, json=body) if body else client.request(method, path)
    assert response.status_code in (401, 403), (
        f"{method} {path} answered {response.status_code} without a credential: "
        f"{response.text[:160]}"
    )


@pytest.mark.enforced_auth
@pytest.mark.parametrize("method,path,reason", PUBLIC)
def test_the_named_public_pages_stay_reachable(method, path, reason, auth_enforced):
    """Deny-by-default must not sweep up the pages it was never about."""
    response = client.request(method, path)
    assert response.status_code == 200, (
        f"{method} {path} ({reason}) became unreachable: {response.status_code}"
    )


@pytest.mark.enforced_auth
def test_the_webhook_is_reachable_but_refuses_an_unsigned_body(auth_enforced):
    """Stripe sends no bearer; the signature is the credential.

    Requiring a bearer would break delivery. The endpoint must still refuse a
    request carrying no signature — otherwise "exempt from auth" would mean
    "accepts anything".
    """
    response = client.post("/api/connect/webhooks", json={"probe": True})
    assert response.status_code != 401, (
        "the webhook now demands a bearer; Stripe will never send one"
    )
    assert response.status_code >= 400, (
        f"an unsigned webhook body was accepted: {response.status_code} "
        f"{response.text[:160]}"
    )


def test_no_exempt_path_carries_a_mutating_verb():
    """The exemption list is keyed by path, so it is method-blind.

    `_PUBLIC_CONNECT_PATHS` matches `request.url.path`, which means exempting a
    path exempts **every verb on it**. That was demonstrated rather than
    theorised: adding `/api/connect/products` to the list opened `list_products`
    *and* `create_product`, because a GET and a POST share the path.

    Today's four exemptions are safe — three are GET-only pages and the fourth
    is a POST that verifies a Stripe signature. But the mechanism means a future
    author intending to open a read would silently open the write beside it,
    which is the GET/POST asymmetry `tests/test_conditional_scope_guard.py`
    exists to catch, reintroduced through the exemption list.

    So the rule is asserted rather than the current state: an exempt path may
    not carry a mutating verb unless it authenticates by another means, and the
    only such case is named.
    """
    import ast
    import inspect

    import routes.stripe_connect_v2 as mod

    #: Exempt paths that *do* accept a mutating verb, each with the credential
    #: it uses instead of a bearer.
    NON_BEARER_AUTH = {
        "/api/connect/webhooks": "Stripe signature, verified by parse_event_notification",
    }

    tree = ast.parse(inspect.getsource(mod))
    exempt = {
        el.value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Set, ast.List, ast.Tuple))
        for el in node.elts
        if isinstance(el, ast.Constant)
        and isinstance(el.value, str)
        and el.value.startswith("/")
    }

    mutating: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
                continue
            verb = deco.func.attr.upper()
            if verb not in {"POST", "PUT", "PATCH", "DELETE"}:
                continue
            if not (deco.args and isinstance(deco.args[0], ast.Constant)):
                continue
            path = deco.args[0].value
            if path in exempt and path not in NON_BEARER_AUTH:
                mutating.setdefault(path, []).append(f"{verb} {node.name}")

    assert not mutating, (
        "these exempt paths accept a mutating verb with no credential at all — "
        "exempting a path exempts every method on it, so a read opened here "
        f"opens the write beside it: {mutating}"
    )


def test_the_exemption_list_is_declared_in_the_route_module():
    """Exemptions live with the router, not only in this test.

    A reviewer reading `stripe_connect_v2.py` must be able to see which paths
    bypass the gate and why, without finding this file first.
    """
    import inspect

    import routes.stripe_connect_v2 as mod

    source = inspect.getsource(mod)
    assert "_PUBLIC_CONNECT_PATHS" in source, (
        "the router does not declare its exemptions by name"
    )
    for _, path, _ in PUBLIC:
        assert path in source, f"{path} is not named in the exemption list"
