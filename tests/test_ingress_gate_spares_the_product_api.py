"""`/hosts` is product API. The worker-ingress gate must not swallow it.

`AgentIngressMiddleware` answers `410 agent_ingress_retired` for worker
protocol paths when `XCELSIOR_AGENT_PUBLIC_INGRESS=deny`. Its prefix list read
`("/agent/", "/host")`, and `"/hosts".startswith("/host")` is true — so the
*product* endpoints `GET /hosts` and `POST /hosts/check` were answered with a
message telling the caller their host list had "moved to the private agent
gateway and enrol this host in the SPIRE trust domain".

Production runs this middleware in `deny`. That is what the dashboard's host
list has been returning.

It surfaced from a live gate reading "0 active host(s)" while a host was
demonstrably active, admitted and heartbeating — the gate could not list hosts
because listing them was 410ing, and the skip message named the fleet rather
than the gate. A wrong reason is worse than a failure: it sent the last hour
looking at the fleet.

## Why the trailing slash rather than an exclusion list

`/host/` matches the worker surface exactly — `PUT /host` through the
`path == prefix` clause, `/host/{id}` through the slash — and nothing else. An
exclusion list of product paths would need a new entry every time one is added,
and the entry that is forgotten is the one that 410s in production.
"""

from __future__ import annotations

import pytest

from api import AgentIngressMiddleware


@pytest.fixture
def is_worker_path():
    middleware = AgentIngressMiddleware.__new__(AgentIngressMiddleware)
    return middleware._is_worker_path


@pytest.mark.parametrize(
    "path",
    ["/host", "/host/h-abc", "/host/h-abc/drain", "/agent/work/h-abc", "/agent/telemetry"],
)
def test_the_worker_surface_is_still_gated(path: str, is_worker_path):
    """The gate must keep doing its job; a fix that opens it is not a fix."""
    assert is_worker_path(path), f"{path} is worker protocol and must stay gated"


@pytest.mark.parametrize("path", ["/hosts", "/hosts/check", "/hostsanything"])
def test_the_product_api_is_not_gated(path: str, is_worker_path):
    """The regression, named. These are reached by users, not by agents."""
    assert not is_worker_path(path), (
        f"{path} is product API and would be answered 410 "
        "'agent_ingress_retired' under deny — a user being told their host "
        "list moved to a private gateway"
    )


def test_the_prefix_carries_the_trailing_slash_that_makes_this_work():
    """Structural, because the behaviour above is one character away.

    Dropping the slash reintroduces the bug and every parametrised case above
    still reads plausibly to someone skimming.
    """
    assert (
        "/host/" in AgentIngressMiddleware.WORKER_PREFIXES
    ), "the worker prefix lost its trailing slash; /hosts is swallowed again"
    assert (
        "/host" not in AgentIngressMiddleware.WORKER_PREFIXES
    ), "a bare '/host' prefix matches '/hosts' by startswith"


# ── The product spellings of the per-host routes ─────────────────────
#
# Found 2026-11-11 (CVP-01): the console's host page and an administrator's
# ``DELETE /host/{id}`` were answered 410 in production, because those product
# routes were registered only under the worker prefix. They are served under
# ``/api/hosts/`` as well, which the gate never matches.

PRODUCT_HOST_ROUTES = {
    ("GET", "/api/hosts/{host_id}"),
    ("DELETE", "/api/hosts/{host_id}"),
    ("GET", "/api/hosts/{host_id}/maintenance"),
    ("POST", "/api/hosts/{host_id}/drain"),
    ("POST", "/api/hosts/{host_id}/undrain"),
}


@pytest.mark.parametrize("path", ["/api/hosts/h-abc", "/api/hosts/h-abc/drain"])
def test_the_product_host_routes_are_not_gated(path: str, is_worker_path):
    assert not is_worker_path(path)


def test_every_per_host_product_route_has_an_api_spelling():
    from api import app

    served = {
        (method, route.path)
        for route in app.routes
        for method in (getattr(route, "methods", None) or ())
    }
    missing = PRODUCT_HOST_ROUTES - served
    assert not missing, f"not served under /api/hosts: {sorted(missing)}"


def test_the_console_calls_the_ungated_spelling():
    """``fetchHost``/``deleteHost`` must not go back to ``/host/`` (410 in production)."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "api/frontend/src/lib/api.ts"
    ).read_text()
    for name in ("fetchHost", "deleteHost"):
        body = src[src.index(f"function {name}(") :][:400]
        assert "`/api/hosts/${" in body, f"{name} must call /api/hosts/, not /host/"
        assert "`/host/${" not in body


def test_delete_under_api_hosts_reaches_the_route_in_deny_mode(monkeypatch):
    """End to end: with the ingress in deny, the product spelling is answered by
    the route (404 for an unknown host), while the worker spelling is 410."""
    import os

    from fastapi.testclient import TestClient

    from api import app

    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "deny")
    os.environ.setdefault("XCELSIOR_ENV", "test")
    with TestClient(app) as client:
        assert client.delete("/host/no-such-host-xyz").status_code == 410
        assert client.delete("/api/hosts/no-such-host-xyz").status_code == 404
        assert client.get("/api/hosts/no-such-host-xyz").status_code == 404
