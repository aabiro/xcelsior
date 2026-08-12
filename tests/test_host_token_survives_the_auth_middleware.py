"""A per-host agent token must reach the route that verifies it.

`TokenAuthMiddleware` compares the presented bearer against the single shared
`XCELSIOR_API_TOKEN` and answers 401 to anything else. `/agent/*` and `/host`
are in `AGENT_RATE_LIMIT_EXEMPT_PREFIXES` — which is the **rate limiter**, not
auth — and in neither `PUBLIC_PATHS` nor `PUBLIC_PATH_PREFIXES`. So wherever
`AUTH_REQUIRED` is true, an `xat_` credential is rejected before
`_resolve_host_token_identity` ever runs.

## Why that is an outage and not an inconvenience

`XCELSIOR_AGENT_HOST_TOKENS=require` is described as field-wide rotation being
complete. It is the opposite: the route refuses the shared fleet bearer, the
middleware refuses the per-host token, and nothing is left that works. Turning
on the finished state of the rotation takes the fleet down.

## Why no existing test caught it

`tests/conftest.py` sets `AUTH_REQUIRED = False`, and the middleware's first
statement is `if not AUTH_REQUIRED: return await call_next(request)`. Every
host-token test in this suite passes through a middleware that is not running,
so `test_require_mode_refuses_the_shared_fleet_bearer` documents a configuration
that would be a fleet outage. These tests turn it back on — and on **`api`'s own
binding**, because `api.py` imports the flag by value at module load, so
patching `routes._deps` does nothing to it.

## This is the second instance, not a novel defect

The comment above `/api/connect/webhooks` in `PUBLIC_PATHS` records the first:
Stripe thin events "got 401 from this middleware before the handler ran — the
endpoint has been unreachable for as long as it has existed, which the missing
signing secret masked". Same middleware, same shape, different caller.

## The shape of the fix these tests require

The middleware must **recognise** a host-token-shaped credential on the worker
prefixes and pass it through *without granting a principal* — the route still
has to verify it against the host. A blanket exemption of `/agent/*` would make
middleware safety depend on every route under that prefix remembering to call
its own auth, which is the failure mode one layer down.
"""

from __future__ import annotations

import os
import pathlib
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

SHARED_BEARER = "test-shared-fleet-bearer-000000000000000"
HOST_TOKEN = "xat_" + "z" * 40


@pytest.fixture
def client(monkeypatch):
    """The real app, with the middleware actually running.

    `api.AUTH_REQUIRED` rather than `routes._deps.AUTH_REQUIRED`: the module
    imports the flag by value, so patching the source of truth leaves the
    middleware's copy untouched — which is exactly how a test can appear to
    exercise this and not.
    """
    import api

    monkeypatch.setattr(api, "AUTH_REQUIRED", True)
    monkeypatch.setenv("XCELSIOR_API_TOKEN", SHARED_BEARER)
    with TestClient(api.app, raise_server_exceptions=False) as test_client:
        yield test_client


def _is_middleware_rejection(response) -> bool:
    """True for the middleware's own generic 401, not the route's.

    The middleware answers a fixed envelope. A route that rejects the same
    token says *why* — "Host agent token rejected: …" — so the two are
    distinguishable, which is what makes red/green here mean something.
    """
    if response.status_code != 401:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    error = body.get("error") or {}
    return error.get("code") == "unauthorized" and error.get("message") == "Unauthorized"


# ── The middleware must not be a no-op in these tests ─────────────────


def test_the_middleware_is_actually_running(client):
    """Without this the file proves nothing — it is the defect that hid the bug.

    A bare request with no credential must be rejected *by the middleware*. If
    this passes through, `AUTH_REQUIRED` did not take effect and every
    assertion below is vacuous.
    """
    response = client.get("/instances")
    assert _is_middleware_rejection(response), (
        f"the middleware did not challenge an uncredentialed request "
        f"({response.status_code}); AUTH_REQUIRED is not in force and the rest "
        "of this file would pass without testing anything"
    )


def test_the_shared_bearer_still_works(client):
    """The fix must not widen anything. This is the credential that worked before."""
    response = client.get("/instances", headers={"Authorization": f"Bearer {SHARED_BEARER}"})
    assert not _is_middleware_rejection(response), (
        "the shared fleet bearer is now rejected by the middleware; the fix has "
        "broken the credential the fleet currently runs on"
    )


# ── The defect ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/agent/work/h-abc",
        "/agent/commands/h-abc",
        "/agent/telemetry",
        "/host",
    ],
)
def test_a_host_token_reaches_the_route_on_worker_paths(client, path: str):
    """The clause: `xat_` must get past the middleware so the route can verify it.

    The route is expected to reject this particular token — it is not
    registered — but it must be the *route* rejecting, with a reason, not the
    middleware refusing to let it through.
    """
    headers = {"Authorization": f"Bearer {HOST_TOKEN}"}
    # `TestClient.get` takes no `json=`; only the write verbs carry a body.
    if path == "/host":
        response = client.put(path, headers=headers, json={})
    elif path == "/agent/telemetry":
        response = client.post(path, headers=headers, json={})
    else:
        response = client.get(path, headers=headers)
    assert not _is_middleware_rejection(response), (
        f"{path}: a per-host agent token was rejected by TokenAuthMiddleware "
        "before the route could verify it. With XCELSIOR_AGENT_HOST_TOKENS="
        "require this is a total fleet outage — the route refuses the shared "
        "bearer and the middleware refuses the host token."
    )


def test_a_host_token_is_still_challenged_off_the_worker_paths(client):
    """The pass is scoped. An `xat_` credential is not a skeleton key.

    If it worked on `/instances` the middleware would have been widened rather
    than taught, and every product route would accept an unverified string
    beginning `xat_`.
    """
    response = client.get("/instances", headers={"Authorization": f"Bearer {HOST_TOKEN}"})
    assert _is_middleware_rejection(response), (
        "a host token was accepted on a product route; the middleware pass must "
        "be scoped to the worker protocol paths"
    )


def test_a_malformed_token_is_still_challenged_on_worker_paths(client):
    """Only a *host-token-shaped* credential passes, not any string.

    `looks_like_host_token` is a cheap shape check, and it is the whole of the
    middleware's decision — so a bearer that is not shaped like one must still
    be refused here rather than handed to the route.
    """
    response = client.get("/agent/work/h-abc", headers={"Authorization": "Bearer not-a-host-token"})
    assert _is_middleware_rejection(response), (
        "an arbitrary bearer reached the route on a worker path; the middleware "
        "is passing more than host-token-shaped credentials"
    )


# ── The prefix must not repeat the /hosts mistake ─────────────────────


def test_the_worker_prefix_does_not_swallow_the_product_host_routes():
    """`/hosts` is product API. `"/hosts".startswith("/host")` is true.

    That exact prefix bug shipped in `AgentIngressMiddleware` and returned 410
    to every dashboard user. A pass-through here is the opposite risk — it would
    let an unverified `xat_` string through on `GET /hosts`.
    """
    from api import TokenAuthMiddleware

    decide = TokenAuthMiddleware._may_carry_host_token
    for worker_path in ("/host", "/host/h-abc", "/agent/work/x"):
        assert decide(worker_path), f"{worker_path} is worker protocol"
    for product_path in ("/hosts", "/hosts/check", "/hostsanything"):
        assert not decide(product_path), (
            f"{product_path} is product API and must not accept an unverified host token"
        )


# ── Derived: no agent route may be challengeable ──────────────────────


def _routes_that_verify_host_tokens() -> list[str]:
    """Every route whose handler authenticates as an agent, found by behaviour.

    **Not by prefix.** Deriving worker routes from `/agent/` and `/host/` would
    ask the middleware's own rule to validate itself, and would be blind to the
    thing this guard exists for: a worker route added under a *new* prefix. A
    route that calls `_require_agent_auth` is one that expects to verify a host
    token, whatever it is called — so that call is the signal.
    """
    import ast

    from api import app

    auth_entry = "_require_agent_auth"
    handlers: set[str] = set()
    for module in sorted(pathlib.Path("routes").glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    name = getattr(inner.func, "attr", None) or getattr(inner.func, "id", None)
                    if name == auth_entry:
                        handlers.add(node.name)
                        break

    paths: set[str] = set()
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        path = getattr(route, "path", None)
        if endpoint is not None and isinstance(path, str):
            if getattr(endpoint, "__name__", "") in handlers:
                paths.add(path)
    return sorted(paths)


def test_the_agent_route_derivation_finds_something():
    """A guard over an empty set passes; this is what stops that reading green."""
    found = _routes_that_verify_host_tokens()
    assert len(found) >= 3, (
        f"only {len(found)} routes appear to verify host tokens: {found}. The "
        "derivation has stopped seeing them and the guard below is vacuous."
    )


def test_no_route_that_verifies_a_host_token_is_challenged_by_the_middleware():
    """Derived from behaviour on one side, from the middleware on the other.

    The failure this prevents is the one that just happened, in its next form: a
    worker route ships under a prefix the middleware does not know, the
    middleware 401s the host token before the route can verify it, and nothing
    notices because the suite runs with the middleware switched off.

    Identified by "this handler calls `_require_agent_auth`" rather than by
    path, so a new prefix is caught rather than assumed.
    """
    from api import TokenAuthMiddleware
    from routes._deps import PUBLIC_PATHS, PUBLIC_PATH_PREFIXES

    decide = TokenAuthMiddleware._may_carry_host_token
    challenged = [
        path
        for path in _routes_that_verify_host_tokens()
        if not decide(path)
        and path not in PUBLIC_PATHS
        and not path.startswith(PUBLIC_PATH_PREFIXES)
    ]
    assert not challenged, (
        f"these routes verify a per-host agent token but the middleware would "
        f"401 it first: {challenged}. Add the prefix to "
        "`TokenAuthMiddleware.HOST_TOKEN_PREFIXES` — a worker carrying a valid "
        "credential cannot reach them, and with XCELSIOR_AGENT_HOST_TOKENS="
        "require that is a fleet outage."
    )


# ── Cross-host refusal, through the whole stack ───────────────────────


@pytest.fixture
def registered_host_token():
    """A genuinely issued token for one host, so the refusal below is real."""
    try:
        from control_plane.agent_tokens import issue_token
        from control_plane.db import control_plane_transaction
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no control plane: {exc}")

    host_a = f"h-a-{uuid.uuid4().hex[:8]}"
    host_b = f"h-b-{uuid.uuid4().hex[:8]}"
    with control_plane_transaction() as conn:
        for host in (host_a, host_b):
            conn.execute(
                "INSERT INTO hosts (host_id, status, registered_at) "
                "VALUES (%s, 'active', %s) ON CONFLICT (host_id) DO NOTHING",
                (host, time.time()),
            )
        issued = issue_token(conn, host_a, issued_by="test", reason="cross-host guard")
    yield host_a, host_b, issued.secret
    with control_plane_transaction() as conn:
        conn.execute("DELETE FROM host_agent_tokens WHERE host_id = ANY(%s)", ([host_a, host_b],))
        conn.execute("DELETE FROM hosts WHERE host_id = ANY(%s)", ([host_a, host_b],))


def test_a_valid_token_for_one_host_is_refused_on_another(client, registered_host_token):
    """Passing the middleware is not being authorised.

    The middleware grants no principal — it only declines to answer. The route
    still binds the token to its host, and this asserts that binding *through
    the full stack* rather than by calling `verify_token` directly, because the
    middleware is precisely the layer that was hiding the route's behaviour.
    """
    host_a, host_b, secret = registered_host_token
    monkey = {"Authorization": f"Bearer {secret}"}

    own = client.get(f"/agent/commands/{host_a}", headers=monkey)
    assert not _is_middleware_rejection(own), (
        "a valid host token did not reach the route for its own host"
    )

    other = client.get(f"/agent/commands/{host_b}", headers=monkey)
    assert other.status_code == 403, (
        f"a token issued for {host_a} was not refused with 403 on {host_b} "
        f"(got {other.status_code}: {other.text[:200]}). Passing the middleware "
        "must not confer authority over another host."
    )
