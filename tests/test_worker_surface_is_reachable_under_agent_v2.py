"""Every path the worker calls must be reachable under `/agent/v2/`.

Blueprint §22.3 requires the agent gateway to *"allow only `/agent/v2/*` routes"*,
and `infra/envoy/agent-gateway.yaml` implements that literally: `/agent/v2/` is
proxied, everything else gets a 404 with *"agent gateway serves /agent/v2 only"*.

So the day Envoy replaces the interim nginx gateway, any worker call whose path
does not start with `/agent/v2/` stops working. The heartbeat is in that set —
`PUT /host` registers the host and mints its marketplace offer — so the failure
mode is the whole fleet going dark about a minute after cutover.

## Why this reads `worker_agent.py` rather than a list

A checklist in this file would be a second, hand-maintained copy of "what the
worker calls", and it would go stale in the direction of saying *less* — the
same shape as the tier ladders and the payout fields. The requirement is
derived from the worker's own `_api_url(...)` call sites, so adding a call to a
new endpoint fails this test until that endpoint is reachable under v2.

## What is exempt, and why it is not a loophole

`/oauth/token` is how the worker *obtains* the credential it presents. Under the
Envoy gateway the SVID is the identity, so this exchange is expected to fall
away rather than be relocated — but that is a design decision, not something to
alias quietly. It is listed here so the decision stays visible instead of being
absorbed into a passing test.
"""

from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKER = ROOT / "worker_agent.py"

#: Paths the worker calls that are deliberately not being relocated, with the
#: reason. Anything here is a decision someone made, not an oversight.
NOT_RELOCATED = {
    # The credential exchange itself, and the one worker path that stays public
    # on purpose.
    #
    # It is not gated by `AgentIngressMiddleware` (whose prefixes are `/agent/`
    # and `/host/`) and that is correct rather than an oversight: this endpoint
    # is authenticated by the client secret it is being handed, so it cannot sit
    # behind a credential the caller does not have yet. Standard OAuth shape.
    #
    # It *could* be relocated now — the worker holds a client certificate
    # provisioned out of band, so it can reach the gateway before it holds any
    # token. It is deliberately not, because SVID identity removes the exchange
    # rather than moving it: a `/agent/v2/oauth/token` would be built to be
    # deleted.
    #
    # The decision therefore has a trigger, so it cannot sit here unexamined:
    # when SPIRE lands and `XCELSIOR_SPIFFE_STRICT` goes back to `1`, delete
    # this endpoint from the worker's path. If SPIRE is abandoned instead,
    # relocate it like the other 22.
    "/oauth/token",
}


def _serverless_suffixes() -> set[str]:
    """The literal suffixes `_serverless_callback` is actually called with.

    That helper builds `.../workers/{worker_id}/{suffix}`, so the path is only
    knowable by reading its call sites. Resolving them keeps the coverage check
    exact — the alternative is matching the dynamic shape loosely, which would
    mark the whole serverless subtree "covered" on the strength of one alias.
    """
    src = WORKER.read_text(encoding="utf-8")
    return set(re.findall(r'_serverless_callback\(\s*[^,]+,\s*"([a-z/]+)"', src))


def _worker_paths() -> set[str]:
    """Every distinct API path `worker_agent.py` builds with `_api_url`."""
    src = WORKER.read_text(encoding="utf-8")
    found = set()
    for raw in re.findall(r'_api_url\(\s*f?"([^"]+)"', src):
        # Normalise f-string interpolations to a single placeholder so
        # `/agent/logs/{job_id}` and `/agent/logs/{other}` are one path.
        found.add(re.sub(r"\{[^}]*\}", "{}", raw).rstrip("/") or "/")
    # Expand the one dynamically-built family into the paths it can produce.
    dynamic = "/api/v2/serverless/workers/{}/{}"
    if dynamic in found:
        found.discard(dynamic)
        for suffix in _serverless_suffixes():
            found.add(f"/api/v2/serverless/workers/{{}}/{suffix}")
    return found


def _mounted_v2_paths() -> set[str]:
    from api import app

    return {
        re.sub(r"\{[^}]*\}", "{}", getattr(r, "path", "")).rstrip("/")
        for r in app.routes
        if getattr(r, "path", "").startswith("/agent/v2/")
    }


def test_the_scan_finds_worker_calls():
    """Calibration — an empty set would satisfy the real assertion."""
    paths = _worker_paths()
    assert len(paths) > 15, f"only {len(paths)} worker paths found; the pattern is wrong"
    assert "/host" in paths, "the heartbeat should be among them"


def test_every_worker_path_has_an_agent_v2_route():
    worker = _worker_paths()
    v2 = _mounted_v2_paths()

    from routes.agent_v2_aliases import ALIASES, SUPERSEDED

    # The mapping is *declared*, not derivable: `/host` becomes
    # `/agent/v2/hosts/heartbeat`, which no suffix rule would ever match. The
    # first version of this test tried to infer it and reported ten endpoints
    # as missing that were mounted and working.
    declared = {re.sub(r"\{[^}]*\}", "{}", v1).rstrip("/") for v1, _m, _v2 in ALIASES}
    superseded = {re.sub(r"\{[^}]*\}", "{}", k).rstrip("/") for k in SUPERSEDED}

    missing = []
    for path in sorted(worker):
        if path in NOT_RELOCATED or path.startswith("/agent/v2"):
            continue
        if path in superseded:
            continue  # a v2 replacement exists; the worker migrates its calls
        if path not in declared:
            missing.append(path)

    assert not missing, (
        "these paths are called by worker_agent.py and are NOT reachable under "
        "/agent/v2/, so the Envoy gateway will 404 them after cutover "
        f"(§22.3, 'serves /agent/v2 only'):\n  " + "\n  ".join(missing)
    )


def test_every_declared_alias_is_actually_mounted():
    """Declaring a mapping is not mounting it."""
    from routes.agent_v2_aliases import ALIASES

    v2 = _mounted_v2_paths()
    unmounted = [
        v2_path
        for _v1, _m, v2_path in ALIASES
        if re.sub(r"\{[^}]*\}", "{}", v2_path).rstrip("/") not in v2
    ]
    assert not unmounted, f"declared but not mounted: {unmounted}"


def test_superseded_paths_are_not_aliased():
    """The fenced protocol must not be bypassable through a v2 URL.

    `/agent/commands/{host}` is poll+drain and `/agent/lease/*` is unfenced.
    Mounting either at a `/agent/v2/` path would put the protocol the fence was
    introduced to replace back onto the surface the fence protects.
    """
    from routes.agent_v2_aliases import ALIASES, SUPERSEDED

    aliased = {v1 for v1, _m, _v2 in ALIASES}
    overlap = aliased & set(SUPERSEDED)
    assert not overlap, f"superseded paths must not be aliased: {sorted(overlap)}"


def test_the_exemptions_are_still_called():
    """An exemption for a call the worker no longer makes hides the next one."""
    stale = NOT_RELOCATED - _worker_paths()
    assert not stale, f"NOT_RELOCATED lists paths the worker no longer calls: {sorted(stale)}"


def test_each_alias_behaves_exactly_like_its_v1_path():
    """Same function, so same answer — asserted rather than assumed.

    Run with gateway headers present, because the relocated endpoints now live
    under `/agent/` and are therefore subject to `AgentIngressMiddleware`. That
    is the point: `/instances/{job}/http-ports/report`,
    `/user-images/{id}/complete`, the promotion callbacks and the sweep
    fingerprint were reachable on the public origin before this. Under
    `/agent/v2/` they are refused unless the request came through the gateway.

    Without the headers the pairs *should* differ, and the companion test below
    pins that so the protection cannot be lost silently.
    """
    import os

    os.environ["XCELSIOR_AGENT_GATEWAY_SECRET"] = "test-gateway-secret"
    os.environ["XCELSIOR_AGENT_PUBLIC_INGRESS"] = "deny"
    from fastapi.testclient import TestClient

    from api import app
    from routes.agent_v2_aliases import ALIASES

    client = TestClient(app)
    headers = {
        "X-Xcelsior-Agent-Gateway": "1",
        "X-Xcelsior-Gateway-Auth": "test-gateway-secret",
        "X-Worker-Host-Id": "probe",
    }

    def fill(path: str) -> str:
        for token, value in (
            ("{host_id}", "probe"),
            ("{job_id}", "probe"),
            ("{image_id}", "probe"),
            ("{promotion_id}", "probe"),
            ("{sweep_id}", "probe"),
            ("{member_index}", "0"),
            ("{worker_id}", "probe"),
        ):
            path = path.replace(token, value)
        return path

    differing = []
    for v1, method, v2 in ALIASES:
        r1 = client.request(method, fill(v1), json={}, headers=headers)
        r2 = client.request(method, fill(v2), json={}, headers=headers)
        if r1.status_code != r2.status_code:
            differing.append(f"{method} {v1}: v1={r1.status_code} v2={r2.status_code}")

    assert not differing, (
        "an alias answered differently from the path it aliases, which means it "
        "is not the same handler after all:\n  " + "\n  ".join(differing)
    )


def test_the_relocated_endpoints_are_now_behind_the_ingress_gate():
    """Relocation tightened them; prove it, so a later change cannot loosen it.

    These were reachable on the public origin under their product prefixes.
    Under `/agent/v2/` the ingress gate refuses them without gateway proof.
    """
    import os

    os.environ["XCELSIOR_AGENT_GATEWAY_SECRET"] = "test-gateway-secret"
    os.environ["XCELSIOR_AGENT_PUBLIC_INGRESS"] = "deny"
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)
    # One representative of each formerly-public family.
    for path, method in (
        ("/agent/v2/instances/probe/http-ports/report", "POST"),
        ("/agent/v2/user-images/probe/complete", "POST"),
        ("/agent/v2/promotions/probe/result", "POST"),
    ):
        r = client.request(method, path, json={})
        assert r.status_code == 410, (
            f"{method} {path} answered {r.status_code} without gateway proof; "
            "it should be refused by AgentIngressMiddleware"
        )
