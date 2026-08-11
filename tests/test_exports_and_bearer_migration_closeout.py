"""Phase 10 residual closeout: /exports off API, shared bearer migration, ingress."""

from __future__ import annotations

from pathlib import Path

import pytest

from control_plane.identity import (
    allow_shared_bearer_for_host,
    shared_bearer_migration_enabled,
)

ROOT = Path(__file__).resolve().parents[1]


def test_compose_api_has_no_exports_mount():
    compose = (ROOT / "docker-compose.yml").read_text()
    start = compose.index("\n  api:\n")
    mid = compose.index("\n  api-blue:\n", start)
    end = compose.index("\n  scheduler-worker:\n", mid)
    api_block = compose[start:mid]
    blue_block = compose[mid:end]
    assert "/exports:/exports" not in api_block
    assert "/exports:/exports" not in blue_block
    # Shared anchor must not reintroduce exports for API/bg-worker.
    anchor = compose.split("x-api-volumes:")[1].split("services:")[0]
    assert "/exports" not in anchor
    # Provisioner profile still owns exports.
    assert "volume-provisioner:" in compose
    prov = compose.split("volume-provisioner:")[1].split("\njaeger:")[0]
    assert "/exports:/exports" in prov
    assert "SYS_ADMIN" in prov


def test_scheduler_has_no_exports_mount():
    compose = (ROOT / "docker-compose.yml").read_text()
    # scheduler-worker block
    start = compose.index("\n  scheduler-worker:\n")
    end = compose.index("\n  bg-worker:\n", start)
    block = compose[start:end]
    assert "/exports:/exports" not in block


def test_volume_privilege_default_host_ssh(monkeypatch):
    import volumes as vol_mod

    monkeypatch.setenv("XCELSIOR_VOLUME_PRIVILEGE", "host_ssh")
    eng = vol_mod.VolumeEngine.__new__(vol_mod.VolumeEngine)
    assert eng._luks_force_ssh() is True


def test_shared_bearer_rejected_without_migration(monkeypatch):
    monkeypatch.delenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", raising=False)
    assert shared_bearer_migration_enabled() is False
    # Real master-token shape from routes._deps: user_id=api-admin, is_admin=True
    ok, reason = allow_shared_bearer_for_host(
        {
            "email": "api-token@xcelsior.ca",
            "user_id": "api-admin",
            "role": "admin",
            "is_admin": True,
        },
        host_id="h1",
        host_owner="someone-else",
    )
    assert ok is False
    assert reason == "shared_bearer_migration_disabled"
    # Alternate principal shape also used by websocket/API paths
    ok2, reason2 = allow_shared_bearer_for_host(
        {"user_id": "api-token", "role": "admin", "is_admin": True},
        host_id="h1",
        host_owner="someone-else",
    )
    assert ok2 is False
    assert reason2 == "shared_bearer_migration_disabled"


def test_shared_bearer_allowed_with_migration_flag(monkeypatch):
    monkeypatch.setenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", "1")
    monkeypatch.delenv("XCELSIOR_AGENT_SHARED_BEARER_HOSTS", raising=False)
    ok, reason = allow_shared_bearer_for_host(
        {
            "email": "api-token@xcelsior.ca",
            "user_id": "api-admin",
            "role": "admin",
            "is_admin": True,
        },
        host_id="h1",
        host_owner="provider-x",
    )
    assert ok is True
    assert reason == "shared_bearer_migration"


def test_human_admin_still_allowed_without_migration(monkeypatch):
    """Human DB admin (not shared fleet bearer) may still act on any host."""
    monkeypatch.delenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", raising=False)
    ok, reason = allow_shared_bearer_for_host(
        {"user_id": "user-42", "email": "ops@xcelsior.ca", "is_admin": True, "role": "admin"},
        host_id="h1",
        host_owner="provider-x",
    )
    assert ok is True
    assert reason == "admin"


def test_shared_bearer_respects_host_allowlist(monkeypatch):
    monkeypatch.setenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", "1")
    monkeypatch.setenv("XCELSIOR_AGENT_SHARED_BEARER_HOSTS", "h-ok,h-two")
    master = {
        "email": "api-token@xcelsior.ca",
        "user_id": "api-admin",
        "role": "admin",
        "is_admin": True,
    }
    ok, reason = allow_shared_bearer_for_host(
        master, host_id="h-bad", host_owner="other"
    )
    assert ok is False
    assert reason == "shared_bearer_host_not_allowlisted"
    ok2, _ = allow_shared_bearer_for_host(master, host_id="h-ok", host_owner="other")
    assert ok2 is True


def test_owner_always_allowed_without_migration(monkeypatch):
    monkeypatch.delenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", raising=False)
    ok, reason = allow_shared_bearer_for_host(
        {"user_id": "owner-1", "email": "o@x.ca"},
        host_id="h1",
        host_owner="owner-1",
    )
    assert ok is True
    assert reason == "owner"


def test_public_nginx_strips_gateway_headers_structural():
    conf = (ROOT / "nginx" / "xcelsior.conf").read_text()
    agent = conf.split("location /agent/")[1].split("location /api/")[0]
    assert 'proxy_set_header X-Xcelsior-Gateway-Auth ""' in agent
    assert 'proxy_set_header X-Xcelsior-Agent-Gateway ""' in agent
    agent_priv = (ROOT / "nginx" / "agent-xcelsior.conf").read_text()
    assert "X-Xcelsior-Gateway-Auth" in agent_priv
    assert "ssl_verify_client on" in agent_priv


def test_mcp_compose_wires_redis_rate_limit_fail_closed():
    """Production MCP service must not default to process-local unlimited Map."""
    compose = (ROOT / "docker-compose.yml").read_text()
    start = compose.index("\n  mcp:\n")
    # Next top-level service after mcp
    rest = compose[start + 1 :]
    end_rel = rest.find("\n  frontend:\n")
    assert end_rel > 0
    block = rest[:end_rel]
    assert "MCP_RATE_LIMIT_BACKEND" in block
    assert "MCP_REDIS_URL" in block
    assert "MCP_RATE_LIMIT_REQUIRE_REDIS" in block
    assert "MCP_RATE_LIMIT_FAIL_CLOSED" in block
    # Defaults must be production-shaped redis + fail-closed
    assert "redis" in block.lower()
    assert "REQUIRE_REDIS" in block


def test_agent_auth_production_shared_bearer_route(monkeypatch):
    """Drive shipped _require_agent_auth with REAL master-token principal shape.

    ``routes._deps._get_current_user`` maps XCELSIOR_API_TOKEN →
    ``user_id=api-admin``, ``is_admin=True``. The gate must still require
    migration — is_admin must not short-circuit shared fleet identity.
    """
    import routes.agent as agent_mod
    from fastapi import HTTPException

    monkeypatch.setenv("XCELSIOR_ENV", "production")
    monkeypatch.delenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", raising=False)
    monkeypatch.delenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", raising=False)
    # Exact shape from routes/_deps.py master-token branch (not a fake non-admin).
    real_master = {
        "email": "api-token@xcelsior.ca",
        "user_id": "api-admin",
        "role": "admin",
        "is_admin": True,
        "scopes": ["*"],
    }
    monkeypatch.setattr(agent_mod, "_get_current_user", lambda r: real_master)
    monkeypatch.setattr(
        agent_mod,
        "list_hosts",
        lambda active_only=False: [
            {"host_id": "host-z", "admitted": True, "owner": "provider-y"}
        ],
    )

    class _Req:
        headers = {}

    with pytest.raises(HTTPException) as ei:
        agent_mod._require_agent_auth(_Req(), host_id="host-z")
    assert ei.value.status_code == 403
    detail = str(ei.value.detail)
    assert (
        "SHARED_BEARER_MIGRATION" in detail
        or "migration" in detail.lower()
        or "Shared fleet" in detail
    ), detail

    # With migration enabled, same principal is admitted.
    monkeypatch.setenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", "1")
    user = agent_mod._require_agent_auth(_Req(), host_id="host-z")
    assert user.get("admitted_host_id") == "host-z"
    assert user.get("admission_reason") == "shared_bearer_migration"
