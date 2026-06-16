"""Platform endpoints supporting the Xcelsior MCP server."""

import os
import uuid
from unittest.mock import patch

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _auth_headers() -> dict:
    email = f"mcpplat-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "MCP Platform"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200, login.text[:200]
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_oauth_metadata_includes_mcp_scopes():
    r = client.get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200
    scopes = r.json().get("scopes_supported", [])
    assert "gpu:read" in scopes
    assert "marketplace:read" in scopes
    assert "events:read" in scopes


def test_instance_telemetry_requires_auth(monkeypatch):
    import routes._deps as deps

    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    r = client.get("/api/instances/fake-job-id/telemetry")
    assert r.status_code == 401


def test_instance_telemetry_not_found():
    r = client.get(
        "/api/instances/nonexistent-job-xyz/telemetry",
        headers=_auth_headers(),
    )
    assert r.status_code == 404


def test_instance_telemetry_unassigned():
    email = f"mcpplat-tel-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "MCP Tel"},
    ).json()
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    owner = reg["user"]["customer_id"]

    job = {
        "job_id": "j1",
        "host_id": None,
        "name": "test",
        "owner": owner,
    }
    with patch("scheduler.get_job", return_value=job), patch(
        "routes.instances.get_job", return_value=job
    ):
        r = client.get("/api/instances/j1/telemetry", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["telemetry"] is None