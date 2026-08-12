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
    assert "/host/" in AgentIngressMiddleware.WORKER_PREFIXES, (
        "the worker prefix lost its trailing slash; /hosts is swallowed again"
    )
    assert "/host" not in AgentIngressMiddleware.WORKER_PREFIXES, (
        "a bare '/host' prefix matches '/hosts' by startswith"
    )
