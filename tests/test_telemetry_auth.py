"""Diagnostic telemetry ingest requires agent/host authentication.

Drives the *shipped* ``POST /agent/telemetry`` entry point
(``routes.agent.api_agent_telemetry``): unauthenticated callers are
rejected outside the documented non-production bypasses; authenticated
agent/host principals are accepted. Spoofing a host_id without auth must
not overwrite fleet telemetry.
"""

from __future__ import annotations

import os
import time
import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


def _env(env: str | None = None, allow: str | None = None) -> dict[str, str]:
    new = {
        k: v
        for k, v in os.environ.items()
        if k not in ("XCELSIOR_ENV", "XCELSIOR_ALLOW_UNAUTH_AGENT")
    }
    if env is not None:
        new["XCELSIOR_ENV"] = env
    if allow is not None:
        new["XCELSIOR_ALLOW_UNAUTH_AGENT"] = allow
    elif "XCELSIOR_ALLOW_UNAUTH_AGENT" in new:
        del new["XCELSIOR_ALLOW_UNAUTH_AGENT"]
    return new


@pytest.fixture
def client():
    # Import app after env is set by conftest (test).
    from api import app

    return TestClient(app)


class TestTelemetryEndpointAuth:
    def test_unauthenticated_production_rejected(self, client):
        """Real route: production + no bearer → 401, no cache write."""
        from routes import agent as agent_mod

        host_id = f"spoof-{uuid.uuid4().hex[:8]}"
        agent_mod._host_telemetry.pop(host_id, None)

        with (
            patch.dict(os.environ, _env(env="production", allow="1"), clear=True),
            patch("routes.agent._get_current_user", return_value=None),
        ):
            r = client.post(
                "/agent/telemetry",
                json={
                    "host_id": host_id,
                    "metrics": {"gpu_util": 99.0},
                    "timestamp": time.time(),
                },
            )
        assert r.status_code == 401, r.text
        assert host_id not in agent_mod._host_telemetry

    def test_unauthenticated_dev_without_bypass_rejected(self, client):
        from routes import agent as agent_mod

        host_id = f"spoof-{uuid.uuid4().hex[:8]}"
        agent_mod._host_telemetry.pop(host_id, None)

        with (
            patch.dict(os.environ, _env(env="dev"), clear=True),
            patch("routes.agent._get_current_user", return_value=None),
        ):
            r = client.post(
                "/agent/telemetry",
                json={
                    "host_id": host_id,
                    "metrics": {"gpu_util": 50.0},
                    "timestamp": time.time(),
                },
            )
        assert r.status_code == 401
        assert host_id not in agent_mod._host_telemetry

    def test_authenticated_agent_accepted(self, client):
        """Real route: agent principal may write telemetry for its host."""
        from routes import agent as agent_mod

        host_id = f"agent-h-{uuid.uuid4().hex[:8]}"
        agent_mod._host_telemetry.pop(host_id, None)
        agent_user = {
            "user_id": "worker-1",
            "email": "agent@xcelsior.ca",
            "is_admin": True,
            "role": "admin",
        }
        # Phase 10: production maps host_id to a registered+admitted identity.
        admitted_hosts = [{"host_id": host_id, "admitted": True, "owner": "worker-1"}]

        with (
            patch.dict(os.environ, _env(env="production"), clear=True),
            patch("routes.agent._get_current_user", return_value=agent_user),
            patch("routes.agent.list_hosts", return_value=admitted_hosts),
        ):
            r = client.post(
                "/agent/telemetry",
                json={
                    "host_id": host_id,
                    "metrics": {"gpu_util": 42.5, "temp_c": 61},
                    "timestamp": time.time(),
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok") is True
        assert host_id not in agent_mod._host_telemetry
        shared = agent_mod._read_telemetry_latest(host_id)
        assert shared[host_id]["metrics"]["gpu_util"] == 42.5

        # Cleanup process-local cache
        agent_mod._host_telemetry.pop(host_id, None)

    def test_require_agent_auth_wired_on_telemetry_handler(self):
        """Structural: shipped handler calls the agent auth gate with host_id."""
        import inspect
        from routes.agent import api_agent_telemetry

        src = inspect.getsource(api_agent_telemetry)
        assert "_require_agent_auth" in src
        assert "host_id=payload.host_id" in src or "host_id=" in src


class TestTelemetryAuthHelper:
    """Direct gate tests for the helper the telemetry route uses."""

    def test_production_unauth_raises_on_helper(self):
        from routes.agent import _require_agent_auth
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/agent/telemetry",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 1),
        }
        req = Request(scope)
        with (
            patch.dict(os.environ, _env(env="production"), clear=True),
            patch("routes.agent._get_current_user", return_value=None),
        ):
            with pytest.raises(HTTPException) as ei:
                _require_agent_auth(req, host_id="any-host")
            assert ei.value.status_code == 401


@pytest.fixture
def telemetry_owners(client):
    from routes import agent

    owners = []
    for index in range(2):
        email = f"telemetry-owner-{uuid.uuid4().hex}@example.test"
        result = client.post(
            "/api/auth/register",
            json={"email": email, "password": "Telemetry-Test-2026!", "name": "Telemetry Owner"},
        )
        assert result.status_code == 200, result.text
        login = client.post(
            "/api/auth/login", json={"email": email, "password": "Telemetry-Test-2026!"}
        )
        assert login.status_code == 200, login.text
        headers = {"Authorization": "Bearer " + login.json()["access_token"]}
        result = client.post(
            "/api/hosts/register",
            headers=headers,
            json={
                "hostname": f"Telemetry Fixture {index}",
                "gpu_model": "RTX 4090",
                "vram_gb": 24,
                "country": "CA",
                "province": "ON",
            },
        )
        assert result.status_code == 200, result.text
        host_id = result.json()["host"]["host_id"]
        record = {
            "timestamp": time.time(),
            "received_at": time.time(),
            "metrics": {"utilization": 37 + index, "temp": 56},
        }
        agent._persist_telemetry_latest(host_id, record)
        owners.append((host_id, headers))
    client.cookies.clear()
    yield owners
    for host_id, _ in owners:
        agent._host_telemetry.pop(host_id, None)


def test_host_owner_reads_only_their_own_telemetry(client, telemetry_owners):
    own, foreign = telemetry_owners
    result = client.get("/api/telemetry/all", headers=own[1])
    assert result.status_code == 200, result.text
    assert set(result.json()["hosts"]) == {own[0]}
    assert result.json()["hosts"][own[0]]["metrics"]["utilization"] == 37
    assert client.get("/agent/telemetry/" + own[0], headers=own[1]).status_code == 200
    assert client.get("/agent/telemetry/" + foreign[0], headers=own[1]).status_code == 404


def test_platform_admin_keeps_fleet_telemetry_access(client, telemetry_owners):
    headers = {"Authorization": "Bearer " + os.environ["XCELSIOR_API_TOKEN"]}
    result = client.get("/api/telemetry/all", headers=headers)
    assert result.status_code == 200, result.text
    assert {host_id for host_id, _ in telemetry_owners} <= set(result.json()["hosts"])


def test_telemetry_database_outage_is_not_reported_as_empty_success(
    client, telemetry_owners, monkeypatch
):
    from routes import agent

    own = telemetry_owners[0]

    def unavailable():
        raise RuntimeError("private storage error")

    monkeypatch.setattr(agent, "_get_pg_pool", unavailable)
    response = client.get("/api/telemetry/all", headers=own[1])
    assert response.status_code == 503, response.text
    assert "private storage error" not in response.text
    response = client.get("/agent/telemetry/" + own[0], headers=own[1])
    assert response.status_code == 503, response.text
    with pytest.raises(HTTPException) as error:
        agent._persist_telemetry_latest(own[0], {"metrics": {"utilization": 37}})
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    "metrics",
    [
        {"utilization": 101},
        {"utilization": -1},
        {"utilization": "37"},
        {"utilization": True},
        {"memory_used_gb": 25, "memory_total_gb": 24},
        {"active_jobs": 1.5},
        {"gpu_count": -1},
        {"nested": [{"value": float("nan")}]},
        {"power_draw_w": float("inf")},
    ],
)
def test_invalid_telemetry_is_rejected_before_any_durable_write(client, telemetry_owners, metrics):
    import json
    from routes.agent import _read_telemetry_latest

    host_id = telemetry_owners[0][0]
    response = client.post(
        "/agent/telemetry",
        content=json.dumps({"host_id": host_id, "metrics": metrics}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, response.text
    assert all("input" not in error for error in response.json()["error"]["details"])
    assert _read_telemetry_latest(host_id)[host_id]["metrics"]["utilization"] == 37
