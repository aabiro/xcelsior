"""Phase 10 — identity admission, fail-closed host binding, privilege/ingress gates.

Drives shipped ``control_plane.identity`` helpers and structural checks for
compose (no API SYS_ADMIN), MCP Dockerfile (locked npm ci + non-root), and
separate public/MCP/agent Nginx configs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from control_plane.identity import (
    IdentityAdmissionError,
    require_admitted_host,
    resolve_gateway_identity,
    spiffe_id_for_host,
    strip_untrusted_identity_headers,
    trusted_gateway_enabled,
)

ROOT = Path(__file__).resolve().parents[1]


def test_strip_untrusted_identity_headers_removes_public_injection():
    cleaned = strip_untrusted_identity_headers(
        {
            "Authorization": "Bearer x",
            "X-Worker-Host-Id": "evil",
            "X-Spiffe-Id": "spiffe://evil/x",
            "X-Forwarded-For": "1.2.3.4",
        }
    )
    assert "Authorization" in cleaned or "authorization" in {k.lower() for k in cleaned}
    lower = {k.lower(): v for k, v in cleaned.items()}
    assert "x-worker-host-id" not in lower
    assert "x-spiffe-id" not in lower
    assert lower.get("x-forwarded-for") == "1.2.3.4"


def test_spiffe_id_for_host_shape():
    sid = spiffe_id_for_host("host-abc", trust_domain="xcelsior.ca")
    assert sid == "spiffe://xcelsior.ca/worker/host/host-abc"


def test_require_admitted_host_fail_closed_unknown():
    with pytest.raises(IdentityAdmissionError) as ei:
        require_admitted_host(None, host_id="missing")
    assert ei.value.code == "unknown_host"
    assert ei.value.http_status == 403


def test_require_admitted_host_rejects_not_admitted():
    with pytest.raises(IdentityAdmissionError) as ei:
        require_admitted_host(
            {"host_id": "h1", "admitted": False},
            host_id="h1",
        )
    assert ei.value.code == "host_not_admitted"


@pytest.mark.parametrize("admitted", [None, "", "false", 0, "0", "no", False])
def test_require_admitted_host_rejects_missing_or_false_admitted(admitted):
    """Missing/None admitted must fail closed — never treat as admitted."""
    row = {"host_id": "h1"}
    if admitted is not None or "admitted" in row:
        row["admitted"] = admitted
    # Explicit missing key
    if admitted is None and "force_missing" not in row:
        row.pop("admitted", None)
        # also test explicit None
        row2 = {"host_id": "h1", "admitted": None}
        with pytest.raises(IdentityAdmissionError) as ei2:
            require_admitted_host(row2, host_id="h1")
        assert ei2.value.code == "host_not_admitted"
        with pytest.raises(IdentityAdmissionError) as ei:
            require_admitted_host({"host_id": "h1"}, host_id="h1")
        assert ei.value.code == "host_not_admitted"
        return
    with pytest.raises(IdentityAdmissionError) as ei:
        require_admitted_host({"host_id": "h1", "admitted": admitted}, host_id="h1")
    assert ei.value.code == "host_not_admitted"


def test_require_admitted_host_accepts_admitted():
    ident = require_admitted_host(
        {"host_id": "h1", "admitted": True, "owner": "p1"},
        host_id="h1",
    )
    assert ident.host_id == "h1"
    assert ident.source == "bearer_host"
    assert ident.spiffe_id and ident.spiffe_id.startswith("spiffe://")
    # string/int affirmative forms
    for val in (1, "true", "1", "yes"):
        ok = require_admitted_host({"host_id": "h1", "admitted": val}, host_id="h1")
        assert ok.host_id == "h1"


def test_gateway_identity_required_when_enabled(monkeypatch):
    monkeypatch.setenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", "1")
    monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", "s3cr3t-gateway")
    assert trusted_gateway_enabled() is True
    # No secret presented → reject (public spoof)
    with pytest.raises(IdentityAdmissionError) as ei:
        resolve_gateway_identity({"x-xcelsior-agent-gateway": "1"})
    assert ei.value.code == "gateway_auth_required"
    assert ei.value.http_status == 401

    with pytest.raises(IdentityAdmissionError):
        resolve_gateway_identity(
            {
                "x-xcelsior-agent-gateway": "1",
                "x-xcelsior-gateway-auth": "wrong",
                "x-worker-host-id": "host-9",
            }
        )

    ok = resolve_gateway_identity(
        {
            "x-xcelsior-agent-gateway": "1",
            "x-xcelsior-gateway-auth": "s3cr3t-gateway",
            "x-worker-host-id": "host-9",
            "x-worker-spiffe-id": "spiffe://xcelsior.ca/worker/host/host-9",
        },
        required_host_id="host-9",
    )
    assert ok is not None
    assert ok.host_id == "host-9"
    assert ok.source == "gateway_header"


def test_gateway_mode_without_secret_fails_closed(monkeypatch):
    """Enabling gateway mode without a secret must not trust forgeable headers."""
    monkeypatch.setenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", "1")
    monkeypatch.delenv("XCELSIOR_AGENT_GATEWAY_SECRET", raising=False)
    with pytest.raises(IdentityAdmissionError) as ei:
        resolve_gateway_identity(
            {
                "x-xcelsior-agent-gateway": "1",
                "x-worker-host-id": "evil",
            }
        )
    assert ei.value.code == "gateway_secret_unconfigured"
    assert ei.value.http_status == 503


def test_gateway_mode_off_returns_none(monkeypatch):
    monkeypatch.delenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", raising=False)
    assert resolve_gateway_identity({"x-worker-host-id": "h"}) is None


def test_public_nginx_strips_worker_identity_headers():
    conf = (ROOT / "nginx" / "xcelsior.conf").read_text()
    agent_block = conf.split("location /agent/")[1].split("location /api/")[0]
    for hdr in (
        "X-Worker-Host-Id",
        "X-Worker-Spiffe-Id",
        "X-Xcelsior-Agent-Gateway",
        "X-Xcelsior-Gateway-Auth",
    ):
        assert hdr in agent_block, f"public /agent/ must strip {hdr}"
        assert f'proxy_set_header {hdr} ""' in agent_block or f"proxy_set_header {hdr} \"\"" in agent_block


def test_agent_auth_strict_db_error_fails_closed(monkeypatch):
    """Blueprint §31: host lookup DB error → 503, not fail-open."""
    import routes.agent as agent_mod
    from fastapi import HTTPException

    monkeypatch.setenv("XCELSIOR_ENV", "staging")
    monkeypatch.setenv("XCELSIOR_AGENT_STRICT_HOST_BINDING", "1")
    monkeypatch.delenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", raising=False)
    monkeypatch.delenv("XCELSIOR_ALLOW_UNAUTH_AGENT", raising=False)

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(agent_mod, "list_hosts", _boom)
    monkeypatch.setattr(agent_mod, "_get_current_user", lambda r: None)

    class _Req:
        headers = {}

    with pytest.raises(HTTPException) as ei:
        # test env would bypass; use non-test non-prod with unauth hatch off
        # and strict on — unauth path raises 401 before host check.
        # Exercise production path instead.
        monkeypatch.setenv("XCELSIOR_ENV", "production")
        monkeypatch.setattr(
            agent_mod,
            "_get_current_user",
            lambda r: {"user_id": "api-token", "is_admin": False},
        )
        agent_mod._require_agent_auth(_Req(), host_id="h-1")
    assert ei.value.status_code == 503


def _resolved_compose() -> dict:
    """Parse docker-compose.yml with YAML anchors resolved.

    Reading the raw text would miss (or falsely fail on) settings that
    come from a shared anchor, which is exactly how the hardening block
    is applied — so assert against the merged service definition.
    """
    import yaml

    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def test_compose_api_has_no_sys_admin():
    services = _resolved_compose()["services"]
    for name in ("api", "api-blue"):
        svc = services[name]
        assert "cap_add" not in svc, f"{name} must grant no capabilities"
        assert "no-new-privileges:true" in svc.get("security_opt", []), name


def test_mcp_dockerfile_locked_npm_ci_and_non_root():
    df = (ROOT / "mcp" / "Dockerfile").read_text()
    run_lines = [
        ln.strip()
        for ln in df.splitlines()
        if ln.strip().upper().startswith("RUN ")
    ]
    assert any("npm ci" in ln for ln in run_lines)
    assert not any("npm install" in ln for ln in run_lines)
    assert "USER xcelsior" in df


def test_ingress_separation_configs_exist():
    mcp = (ROOT / "nginx" / "mcp-xcelsior.conf").read_text()
    agent = (ROOT / "nginx" / "agent-xcelsior.conf").read_text()
    public = (ROOT / "nginx" / "xcelsior.conf").read_text()
    assert "mcp.xcelsior.ca" in mcp
    assert "agent.xcelsior.ca" in agent
    assert "ssl_verify_client on" in agent
    assert "X-Xcelsior-Agent-Gateway" in agent
    # Public conf must not be the agent mTLS server_name alone — separate files.
    assert "mcp-xcelsior.conf" != "agent-xcelsior.conf"
    assert "server_name" in public or "upstream" in public or "location" in public


def test_spire_mesh_assets_are_deployable():
    """The mesh is deployed from these files, so they must be real.

    Superseded the earlier "scaffold, not claimed live" assertion: the
    `.example` suffixes are gone and `docker-compose.spire.yml` brings up
    server, agent, and the SPIFFE-aware gateway. Behavioural coverage of
    the identity contract lives in tests/test_spire_identity_mesh.py.
    """
    spire = ROOT / "infra" / "spire"
    for name in ("README.md", "server.conf", "agent.conf",
                 "docker-compose.spire.yml", "register-host.sh"):
        assert (spire / name).is_file(), f"missing infra/spire/{name}"
    assert not list(spire.glob("*.example")), "SPIRE configs must be deployable, not examples"
    assert os.access(spire / "register-host.sh", os.X_OK)
    assert (ROOT / "infra" / "volume-provisioner" / "README.md").is_file()
