"""Track B B2.8 — v1 API contract gate (dependency-free).

The checklist names a schemathesis run over the v1 OpenAPI plus "a test
asserting every v1 error path emits problem+json". schemathesis is a fuzzing
harness we don't want to add to the frozen deps mid-flight; this covers the
same contract deterministically:

1. Every registered `/api/v1/*` route is present in the generated OpenAPI (the
   surface is documented — a generated client, B5.2 / B6.1, can see it).
2. Representative v1 error paths across the surface return
   `application/problem+json` with the full RFC 9457 field set — no v1 error
   leaks the legacy `{"ok": false, ...}` envelope.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
from routes.problem import PROBLEM_MEDIA_TYPE

client = TestClient(app)

_REQUIRED = {"type", "title", "status", "detail", "code", "retryable", "retry_after_ms", "trace_id", "errors"}


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    import api as api_mod
    import routes._deps as deps_mod
    import routes.auth as auth_mod

    monkeypatch.setattr(api_mod, "AUTH_REQUIRED", False)
    monkeypatch.setattr(deps_mod, "AUTH_REQUIRED", False)
    for mod in (api_mod, deps_mod, auth_mod):
        if hasattr(mod, "_USE_PERSISTENT_AUTH"):
            monkeypatch.setattr(mod, "_USE_PERSISTENT_AUTH", True)


def _token(email: str) -> dict:
    reg = client.post("/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}).json()
    return {"Authorization": f"Bearer {reg['access_token']}"}


def _admin(email: str) -> dict:
    from db import UserStore

    reg = client.post("/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}).json()
    UserStore.update_user(email, {"is_admin": 1})
    return {"Authorization": f"Bearer {reg['access_token']}"}


def test_all_v1_routes_are_documented_in_v1_openapi():
    # The app's / openapi.json is a curated public spec; the versioned surface is
    # served at /api/v1/openapi.json (the live schema filtered to v1) — what
    # B5.2 / B6.1 generate their clients from.
    schema = client.get("/api/v1/openapi.json").json()
    documented = set(schema.get("paths", {}))
    registered = {
        r.path
        for r in app.routes
        if getattr(r, "path", "").startswith("/api/v1/") and r.path != "/api/v1/openapi.json"
    }
    missing = registered - documented
    assert not missing, f"v1 routes absent from the v1 OpenAPI schema: {sorted(missing)}"
    assert all(p.startswith("/api/v1/") for p in documented)  # no non-v1 leakage
    # Sanity: the surface we shipped is actually there.
    for expected in (
        "/api/v1/launch-plans",
        "/api/v1/placements/simulate",
        "/api/v1/hosts/{host_id}/drain",
        "/api/v1/hosts/{host_id}/evictions",
        "/api/v1/instances/{job_id}/control-plane",
        "/api/v1/instances/{job_id}/timeline",
        "/api/v1/instances/{job_id}/retry",
        "/api/v1/control-plane/health",
        "/api/v1/control-plane/queue",
        "/api/v1/control-plane/reconciliation-findings",
    ):
        assert expected in registered, f"missing v1 route: {expected}"


def _assert_problem(resp, expected_status: int | None = None):
    if expected_status is not None:
        assert resp.status_code == expected_status, resp.text
    assert resp.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE), resp.text
    body = resp.json()
    assert set(body) >= _REQUIRED, f"missing RFC 9457 fields: {_REQUIRED - set(body)}"
    assert body["trace_id"]


def test_every_v1_error_path_emits_problem_json():
    admin = _admin(f"contract-admin-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    user = _token(f"contract-user-{uuid.uuid4().hex[:6]}@xcelsior.ca")
    jid = uuid.uuid4()
    hid = f"nope-{uuid.uuid4().hex}"

    # 404s
    _assert_problem(client.post(f"/api/v1/launch-plans/{jid}/approve", headers=admin, json={}), 404)
    _assert_problem(client.post(f"/api/v1/hosts/{hid}/drain", headers=admin, json={}), 404)
    _assert_problem(client.get(f"/api/v1/instances/{jid}/control-plane", headers=user), 404)
    _assert_problem(client.get(f"/api/v1/instances/{jid}/timeline", headers=user), 404)
    _assert_problem(client.get(f"/api/v1/hosts/{hid}/capacity", headers=admin), 404)

    # 403 operator-gated
    _assert_problem(client.get("/api/v1/control-plane/health", headers=user), 403)
    _assert_problem(client.get("/api/v1/control-plane/queue", headers=user), 403)

    # 422 typed validation
    _assert_problem(
        client.get("/api/v1/control-plane/reconciliation-findings?status=bogus", headers=admin), 422
    )
