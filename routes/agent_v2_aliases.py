"""Mount the worker protocol under `/agent/v2/*`, without a second copy of it.

Blueprint §22.3 requires the agent gateway to *"allow only `/agent/v2/*` routes"*
and *"no frontend or general API proxying"*. `infra/envoy/agent-gateway.yaml` is
faithful to that: it proxies `/agent/v2/` and answers 404 to everything else.

The worker, however, still speaks mostly v1 — `PUT /host`, `/agent/work/{id}`,
`/agent/telemetry`, and a set of worker reports that were filed under
*product* prefixes (`/instances/{id}/http-ports/report`,
`/user-images/{id}/complete`, `/api/v1/image-sweeps/.../fingerprint`). Cutting
the gateway over to Envoy without this module 404s the heartbeat and the fleet
goes dark within a minute.

## Why aliases rather than new handlers

Nine of the twelve endpoints already authenticate with `_require_agent_auth`.
They are worker protocol; only their URL prefix says otherwise. So this is a
**relocation**, and writing v2 handler bodies would create a second
implementation of logic that is already correct — the drift this codebase keeps
paying for (two ranker paths, two tier ladders, two admission gates).

Each entry below mounts the **same function object** at a `/agent/v2/…` path.
There is one implementation; the alias cannot behave differently, because it is
not a different thing.

## What this deliberately does not do

* It does not change any endpoint's authentication. `PUT /host` keeps
  `_require_auth` + `hosts:write` + `_require_host_operator`; tightening it to
  agent-auth is a real behaviour change and belongs in its own commit, decided
  on its own evidence.
* It does not retire the v1 paths. Both are live during migration, which is what
  makes the cutover reversible — point `XCELSIOR_SCHEDULER_URL` back at the
  nginx gateway and v1 still answers.
* It does not alias anything that is *not* worker protocol. A product endpoint
  reachable through the agent gateway is exactly what §22.3 forbids.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: `(v1 path, method, v2 path)`. The v2 path is the one Envoy will proxy.
#:
#: Ordered as the worker uses them: lifecycle first, then reports.
ALIASES: tuple[tuple[str, str, str], ...] = (
    # Host lifecycle. `/host` registers the host and mints its marketplace
    # offer — without this the fleet cannot come up at all.
    ("/host", "PUT", "/agent/v2/hosts/heartbeat"),
    # Work and control loop.
    ("/agent/work/{host_id}", "GET", "/agent/v2/work/{host_id}"),
    ("/agent/telemetry", "POST", "/agent/v2/telemetry"),
    ("/agent/preempt/{host_id}", "GET", "/agent/v2/preempt/{host_id}"),
    # Reports the worker makes about itself and its jobs.
    ("/agent/logs/{job_id}", "POST", "/agent/v2/logs/{job_id}"),
    ("/agent/benchmark", "POST", "/agent/v2/benchmark"),
    ("/agent/degraded", "POST", "/agent/v2/degraded"),
    ("/agent/mining-alert", "POST", "/agent/v2/mining-alert"),
    # Worker protocol that was filed under product prefixes.
    (
        "/instances/{job_id}/http-ports/report",
        "POST",
        "/agent/v2/instances/{job_id}/http-ports/report",
    ),
    ("/user-images/{image_id}/complete", "POST", "/agent/v2/user-images/{image_id}/complete"),
    ("/agent/ssh-keys/{job_id}", "GET", "/agent/v2/ssh-keys/{job_id}"),
    ("/agent/ssh-status/{job_id}", "POST", "/agent/v2/ssh-status/{job_id}"),
    ("/agent/verify", "POST", "/agent/v2/verify"),
    ("/agent/versions", "POST", "/agent/v2/versions"),
    # Deregistration, on clean worker shutdown.
    ("/host/{host_id}", "DELETE", "/agent/v2/hosts/{host_id}"),
    # Reading and updating the instance the worker is executing.
    ("/instance/{job_id}", "PATCH", "/agent/v2/instances/{job_id}"),
    # Artifact promotion: the worker performs the copy and reports the outcome.
    (
        "/api/v1/promotions/{promotion_id}/manifest",
        "GET",
        "/agent/v2/promotions/{promotion_id}/manifest",
    ),
    (
        "/api/v1/promotions/{promotion_id}/files",
        "POST",
        "/agent/v2/promotions/{promotion_id}/files",
    ),
    (
        "/api/v1/promotions/{promotion_id}/result",
        "POST",
        "/agent/v2/promotions/{promotion_id}/result",
    ),
    # Sweep fingerprint — already lives in routes/agent.py behind
    # `_require_agent_auth`; only its URL said "product API".
    (
        "/api/v1/image-sweeps/{sweep_id}/members/{member_index}/fingerprint",
        "POST",
        "/agent/v2/image-sweeps/{sweep_id}/members/{member_index}/fingerprint",
    ),
    # Serverless worker lifecycle callbacks. `ready` and `exited` are the
    # suffixes `_serverless_callback` is actually invoked with — a grep for the
    # single-line form finds only `ready`, because the others are multi-line
    # calls. The guard resolves them from the call sites for that reason.
    (
        "/api/v2/serverless/workers/{worker_id}/ready",
        "POST",
        "/agent/v2/serverless/workers/{worker_id}/ready",
    ),
    (
        "/api/v2/serverless/workers/{worker_id}/exited",
        "POST",
        "/agent/v2/serverless/workers/{worker_id}/exited",
    ),
)

#: Worker calls whose v2 replacement already exists under a *different* name and
#: with different semantics — claim+ACK instead of poll+drain, fenced leases
#: instead of bare ones (blueprint §11, Phase 5).
#:
#: These must not be aliased. Mounting the v1 poll handler at a v2 path would
#: put the unfenced protocol back on the surface the fence was introduced to
#: protect. The worker migrates its call sites instead.
SUPERSEDED: dict[str, str] = {
    "/agent/commands/{host_id}": "/agent/v2/commands/claim",
    "/agent/lease/claim": "/agent/v2/leases/claim",
    "/agent/lease/release": "/agent/v2/leases/release",
    "/agent/lease/renew": "/agent/v2/leases/renew",
}


def mount_agent_v2_aliases(app) -> list[str]:
    """Register each v2 alias against the *same* endpoint function as v1.

    Returns the v2 paths mounted, so a caller (and a test) can assert what
    happened rather than trust that it did.

    A v1 path that is not found is a hard error, not a warning. Silently
    skipping it would mean the gateway 404s that endpoint after the Envoy
    cutover, and the failure would surface as a worker malfunction in
    production rather than as a startup problem here.
    """
    by_path_method: dict[tuple[str, str], object] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or ()
        if not path:
            continue
        for method in methods:
            by_path_method[(path, method)] = route

    mounted: list[str] = []
    missing: list[str] = []
    for v1_path, method, v2_path in ALIASES:
        route = by_path_method.get((v1_path, method))
        if route is None:
            missing.append(f"{method} {v1_path}")
            continue
        if (v2_path, method) in by_path_method:
            continue  # already mounted; re-running must not duplicate
        app.add_api_route(
            v2_path,
            route.endpoint,
            methods=[method],
            name=f"agent_v2_alias_{route.name}",
            include_in_schema=False,
            response_model=None,
            tags=["AgentV2"],
        )
        mounted.append(v2_path)

    if missing:
        raise RuntimeError(
            "agent v2 alias targets not found: "
            + ", ".join(missing)
            + ". The Envoy gateway serves /agent/v2/* only, so an unmounted "
            "alias becomes a 404 for the worker after cutover."
        )
    log.info("agent v2 aliases mounted: %d", len(mounted))
    return mounted
