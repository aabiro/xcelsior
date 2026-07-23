"""Track B B2.8 / §3.3 — v1 drain never evicts; evict is a separate action.

The safety property: draining a host stops *new* placements but leaves every
running workload running. Removing workloads is a distinct, distinctly-scoped,
distinctly-audited action. This gate proves the behaviour end to end and pins
the scope separation and the RFC 9457 error shape.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
from routes.problem import PROBLEM_MEDIA_TYPE

client = TestClient(app)

_OWNED = {"hosts": set(), "jobs": set()}


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """Recognize persistent admin/users and allow anonymous host registration —
    the same auth wiring test_api uses, without its destructive table wipes
    (this suite owns only its own rows, B0.2 rule 14)."""
    import api as api_mod
    import oauth_service as oauth_mod
    import routes._deps as deps_mod
    import routes.auth as auth_mod

    monkeypatch.setattr(api_mod, "AUTH_REQUIRED", False)
    monkeypatch.setattr(deps_mod, "AUTH_REQUIRED", False)
    for mod in (api_mod, deps_mod, auth_mod):
        if hasattr(mod, "_USE_PERSISTENT_AUTH"):
            monkeypatch.setattr(mod, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(oauth_mod, "AUTH_CACHE_BACKEND", "memory")
    yield
    import scheduler as sched

    with sched._atomic_mutation() as conn:
        for jid in _OWNED["jobs"]:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in _OWNED["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
    _OWNED["hosts"].clear()
    _OWNED["jobs"].clear()


GOOD_VERSIONS = {
    "runc": "1.2.4",
    "nvidia_ctk": "1.17.8",
    "nvidia_driver": "560.35.03",
    "docker": "27.1.1",
}


def _admin_headers(email: str) -> dict:
    """Register a platform admin and fund its wallet (so /instance passes the
    wallet preflight), returning auth headers."""
    from billing import BillingEngine
    from db import UserStore

    reg = client.post(
        "/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}
    ).json()
    UserStore.update_user(email, {"is_admin": 1})
    BillingEngine().deposit(reg["user"]["customer_id"], 1000.0, description="cpv1 seed")
    return {"Authorization": f"Bearer {reg['access_token']}"}


def _platform_headers() -> dict:
    import os

    return {"Authorization": f"Bearer {os.environ.get('XCELSIOR_API_TOKEN', 'test-token-not-for-production')}"}


def _register_host(host_id: str, ip: str = "10.9.0.1") -> None:
    r = client.put(
        "/host",
        json={
            "host_id": host_id,
            "ip": ip,
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "free_vram_gb": 24,
            "versions": GOOD_VERSIONS,
        },
    )
    assert r.status_code == 200, r.text
    _OWNED["hosts"].add(host_id)


def _running_job_on(host_id: str, headers: dict) -> str:
    """Submit, place, and drive a job to running on *host_id*; return job_id."""
    from scheduler import process_queue

    sub = client.post(
        "/instance", json={"name": f"drain-{uuid.uuid4().hex[:6]}", "vram_needed_gb": 8}, headers=headers
    )
    assert sub.status_code == 200, sub.text
    job_id = sub.json()["instance"]["job_id"]
    _OWNED["jobs"].add(job_id)
    process_queue()
    client.patch(
        f"/instance/{job_id}",
        json={"status": "running", "host_id": host_id},
        headers=_platform_headers(),
    )
    detail = client.get(f"/instance/{job_id}").json()["instance"]
    assert detail["status"] == "running", detail
    return job_id


# ── The core §3.3 property ──────────────────────────────────────────────


def test_v1_drain_leaves_running_workloads_running():
    admin = _admin_headers(f"drain-admin-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    host_id = f"cpv1-drain-{uuid.uuid4().hex[:6]}"
    _register_host(host_id)
    job_id = _running_job_on(host_id, admin)

    resp = client.post(f"/api/v1/hosts/{host_id}/drain", headers=admin, json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["evicted"] == []  # drain never evicts
    assert body["host"]["status"] == "draining"

    # The running workload is untouched.
    still = client.get(f"/instance/{job_id}").json()["instance"]
    assert still["status"] == "running"
    assert still["host_id"] == host_id


def test_v1_evictions_removes_running_workloads():
    admin = _admin_headers(f"evict-admin-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    host_id = f"cpv1-evict-{uuid.uuid4().hex[:6]}"
    _register_host(host_id)
    job_id = _running_job_on(host_id, admin)

    # §3.3 sequence: evicting a live host is refused — drain first.
    early = client.post(f"/api/v1/hosts/{host_id}/evictions", headers=admin, json={})
    assert early.status_code == 409
    assert early.json()["code"] == "host_not_draining"

    assert client.post(f"/api/v1/hosts/{host_id}/drain", headers=admin, json={}).status_code == 200
    resp = client.post(f"/api/v1/hosts/{host_id}/evictions", headers=admin, json={})
    assert resp.status_code == 200, resp.text
    assert job_id in resp.json()["evicted"]

    # Evicted workload is no longer running on the host (preempted → requeued).
    after = client.get(f"/instance/{job_id}").json()["instance"]
    assert after["status"] != "running"


# ── Separation of authority (§3.3 "separately authorized") ──────────────


def test_drain_and_evict_require_distinct_scopes():
    """Source-level pin: drain requires hosts:operate, evict requires hosts:evict."""
    import routes.control_plane_v1 as mod

    src = pathlib.Path(inspect.getfile(mod)).read_text()
    tree = ast.parse(src)

    def _scope_of(fn_name: str) -> set[str]:
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == fn_name
        )
        scopes = set()
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_require_host_operator"
            ):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) and ":" in a.value:
                        scopes.add(a.value)
        return scopes

    drain_scopes = _scope_of("api_v1_drain_host")
    evict_scopes = _scope_of("api_v1_evict_host_workloads")
    assert drain_scopes == {"hosts:operate"}
    assert evict_scopes == {"hosts:evict"}
    assert drain_scopes.isdisjoint(evict_scopes)  # a drainer cannot evict


# ── RFC 9457 + optimistic concurrency ───────────────────────────────────


def test_unknown_host_is_problem_json_404():
    admin = _admin_headers(f"nf-admin-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    resp = client.post(f"/api/v1/hosts/nope-{uuid.uuid4().hex}/drain", headers=admin, json={})
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert resp.json()["code"] == "host_not_found"


def test_stale_expected_version_is_refused():
    admin = _admin_headers(f"ver-admin-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    host_id = f"cpv1-ver-{uuid.uuid4().hex[:6]}"
    _register_host(host_id)
    resp = client.post(
        f"/api/v1/hosts/{host_id}/drain", headers=admin, json={"expected_version": 999999}
    )
    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert resp.json()["code"] == "version_conflict"


# ── v1 read endpoints (control-plane / timeline / findings) ─────────────


def _plain_user_headers(email: str) -> dict:
    reg = client.post(
        "/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}
    ).json()
    return {"Authorization": f"Bearer {reg['access_token']}"}


def test_instance_control_plane_and_timeline_are_tenant_scoped():
    owner = _admin_headers(f"cp-owner-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    host_id = f"cpv1-read-{uuid.uuid4().hex[:6]}"
    _register_host(host_id)
    job_id = _running_job_on(host_id, owner)

    cp = client.get(f"/api/v1/instances/{job_id}/control-plane", headers=owner)
    assert cp.status_code == 200, cp.text
    body = cp.json()
    assert body["job_id"] == job_id
    assert "status" in body and "phase" in body and "desired_state" in body
    assert isinstance(body["attempt_count"], int)

    tl = client.get(f"/api/v1/instances/{job_id}/timeline", headers=owner)
    assert tl.status_code == 200
    assert isinstance(tl.json()["attempts"], list)


def test_cross_tenant_instance_is_not_found_no_leak():
    owner = _admin_headers(f"cp-a-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    host_id = f"cpv1-leak-{uuid.uuid4().hex[:6]}"
    _register_host(host_id)
    job_id = _running_job_on(host_id, owner)

    intruder = _plain_user_headers(f"cp-b-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    for path in (f"/api/v1/instances/{job_id}/control-plane", f"/api/v1/instances/{job_id}/timeline"):
        resp = client.get(path, headers=intruder)
        assert resp.status_code == 404, (path, resp.text)
        assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
        assert resp.json()["code"] == "instance_not_found"  # not a 403 permission hint


def test_reconciliation_findings_requires_admin():
    plain = _plain_user_headers(f"rf-user-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    denied = client.get("/api/v1/control-plane/reconciliation-findings", headers=plain)
    assert denied.status_code == 403

    admin = _admin_headers(f"rf-admin-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    ok = client.get("/api/v1/control-plane/reconciliation-findings", headers=admin)
    assert ok.status_code == 200
    assert isinstance(ok.json()["findings"], list)
    bad = client.get("/api/v1/control-plane/reconciliation-findings?status=bogus", headers=admin)
    assert bad.status_code == 422
    assert bad.json()["code"] == "invalid_status"
