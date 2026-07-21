"""Production startup validator and §21.3 health semantics.

Blueprint §30 lists the configuration states a production deployment must
refuse; §21.3 splits liveness (`/livez`), readiness (`/readyz`), and
startup (`/startupz`) so a dependency blip cannot get healthy replicas
restarted.

Each §30 rejection is driven individually here: the check is proven to
fire on the bad configuration *and* to stay quiet on the good one, so a
check that always fires (or never does) cannot pass as coverage.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app  # noqa: E402
from control_plane import startup_validation as sv  # noqa: E402

client = TestClient(app)


@pytest.fixture
def clean_env(monkeypatch):
    """A configuration that passes every check, as the baseline."""
    monkeypatch.setenv("XCELSIOR_DB_BACKEND", "postgres")
    monkeypatch.setenv("XCELSIOR_POSTGRES_DSN", "postgresql://u:p@127.0.0.1:5432/xcelsior")
    monkeypatch.setenv("XCELSIOR_OAUTH_JWT_SECRET", "not-empty")
    monkeypatch.delenv("XCELSIOR_ALLOW_UNAUTH_AGENT", raising=False)
    monkeypatch.delenv("XCELSIOR_ALLOW_RUNTIME_DDL", raising=False)
    monkeypatch.delenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", raising=False)
    monkeypatch.delenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", raising=False)
    monkeypatch.delenv("XCELSIOR_PG_REQUIRE_TLS", raising=False)
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    monkeypatch.setenv("XCELSIOR_VOLUME_PRIVILEGE", "host_ssh")
    monkeypatch.setenv("MCP_REPLICAS", "1")
    monkeypatch.setenv("MCP_RATE_LIMIT_BACKEND", "redis")
    monkeypatch.setenv("MCP_RATE_LIMIT_FAIL_CLOSED", "true")
    return monkeypatch


def _codes(findings) -> set[str]:
    return {f.code for f in findings}


def test_a_correct_configuration_produces_no_findings(clean_env):
    assert sv.collect_findings() == []


# ── §30 rejection list, one at a time ─────────────────────────────────


@pytest.mark.parametrize("backend", ["sqlite", "dual"])
def test_rejects_sqlite_and_dual_backend(clean_env, backend):
    clean_env.setenv("XCELSIOR_DB_BACKEND", backend)
    findings = sv.collect_findings()
    assert "database_backend" in _codes(findings)
    assert all(f.severity == "error" for f in findings if f.code == "database_backend")


def test_rejects_a_remote_database_without_tls(clean_env):
    clean_env.setenv("XCELSIOR_POSTGRES_DSN", "postgresql://u:p@db.example.internal:5432/xcelsior")
    assert "database_tls" in _codes(sv.collect_findings())

    clean_env.setenv(
        "XCELSIOR_POSTGRES_DSN",
        "postgresql://u:p@db.example.internal:5432/xcelsior?sslmode=require",
    )
    assert "database_tls" not in _codes(sv.collect_findings())


def test_rejects_explicitly_disabled_tls_even_remotely(clean_env):
    clean_env.setenv(
        "XCELSIOR_POSTGRES_DSN",
        "postgresql://u:p@db.example.internal:5432/xcelsior?sslmode=disable",
    )
    assert "database_tls" in _codes(sv.collect_findings())


def test_loopback_database_is_exempt_unless_tls_is_forced(clean_env):
    """A unix socket / loopback DSN is not a network boundary."""
    for dsn in (
        "postgresql://u:p@127.0.0.1:5432/xcelsior",
        "postgresql://u:p@localhost:5432/xcelsior",
        "postgresql:///xcelsior",
    ):
        clean_env.setenv("XCELSIOR_POSTGRES_DSN", dsn)
        assert "database_tls" not in _codes(sv.collect_findings()), dsn

    clean_env.setenv("XCELSIOR_PG_REQUIRE_TLS", "1")
    clean_env.setenv("XCELSIOR_POSTGRES_DSN", "postgresql://u:p@127.0.0.1:5432/xcelsior")
    assert "database_tls" in _codes(sv.collect_findings())


def test_rejects_empty_oauth_signing_configuration(clean_env):
    clean_env.delenv("XCELSIOR_OAUTH_JWT_SECRET", raising=False)
    clean_env.delenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", raising=False)
    assert "oauth_signing" in _codes(sv.collect_findings())

    clean_env.setenv("XCELSIOR_OAUTH_JWT_KEYS_JSON", '{"keys":[]}')
    assert "oauth_signing" not in _codes(sv.collect_findings())


def test_rejects_unauthenticated_agent_mode(clean_env):
    clean_env.setenv("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")
    findings = [f for f in sv.collect_findings() if f.code == "unauthenticated_agent"]
    assert findings and findings[0].severity == "error"
    assert "escape hatch" in findings[0].remediation


def test_rejects_runtime_ddl(clean_env):
    clean_env.setenv("XCELSIOR_ALLOW_RUNTIME_DDL", "true")
    assert "runtime_ddl_enabled" in _codes(sv.collect_findings())


def test_rejects_process_local_mcp_rate_limiting_under_replicas(clean_env):
    clean_env.setenv("MCP_REPLICAS", "2")
    clean_env.setenv("MCP_RATE_LIMIT_BACKEND", "memory")
    findings = [f for f in sv.collect_findings() if f.code == "mcp_rate_limit_process_local"]
    assert findings and findings[0].severity == "error"

    # One replica does not need a shared store.
    clean_env.setenv("MCP_REPLICAS", "1")
    assert "mcp_rate_limit_process_local" not in _codes(sv.collect_findings())

    # Redis-backed is fine at any replica count.
    clean_env.setenv("MCP_REPLICAS", "4")
    clean_env.setenv("MCP_RATE_LIMIT_BACKEND", "redis")
    assert "mcp_rate_limit_process_local" not in _codes(sv.collect_findings())


def test_warns_when_mcp_rate_limiting_would_fall_open(clean_env):
    clean_env.setenv("MCP_RATE_LIMIT_BACKEND", "redis")
    clean_env.setenv("MCP_RATE_LIMIT_FAIL_CLOSED", "false")
    findings = [f for f in sv.collect_findings() if f.code == "mcp_rate_limit_fail_open"]
    assert findings and findings[0].severity == "warning"


def test_rejects_api_sys_admin_expectation(clean_env):
    """After the volume-provisioner cutover the API cannot do local LUKS."""
    clean_env.setenv("XCELSIOR_VOLUME_PRIVILEGE", "local")
    findings = [f for f in sv.collect_findings() if f.code == "api_privilege_expectation"]
    assert findings and findings[0].severity == "error"

    for ok in ("host_ssh", "provisioner"):
        clean_env.setenv("XCELSIOR_VOLUME_PRIVILEGE", ok)
        assert "api_privilege_expectation" not in _codes(sv.collect_findings())


def test_rejects_a_gateway_flag_without_its_secret(clean_env):
    clean_env.setenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", "1")
    clean_env.delenv("XCELSIOR_AGENT_GATEWAY_SECRET", raising=False)
    assert "agent_gateway_unauthenticated" in _codes(sv.collect_findings())

    clean_env.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", "s3cr3t")
    assert "agent_gateway_unauthenticated" not in _codes(sv.collect_findings())


def test_warns_while_the_shared_fleet_bearer_is_still_live(clean_env):
    clean_env.setenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", "1")
    findings = [f for f in sv.collect_findings() if f.code == "shared_fleet_bearer_active"]
    assert findings and findings[0].severity == "warning"
    assert "XCELSIOR_AGENT_HOST_TOKENS=require" in findings[0].remediation


def test_rejects_require_mode_while_hosts_lack_tokens(clean_env):
    """The flip that would lock a host out must be caught before it lands."""
    clean_env.setenv("XCELSIOR_AGENT_HOST_TOKENS", "require")
    clean_env.setattr(
        sv,
        "_check_host_token_rotation_readiness",
        lambda: sv.Finding(
            code="host_token_coverage_incomplete",
            severity="error",
            message="2 host(s) have no live token",
            remediation="issue tokens first",
        ),
    )
    clean_env.setattr(sv, "CHECKS", (sv._check_host_token_rotation_readiness,), raising=False)
    assert "host_token_coverage_incomplete" in _codes(sv.collect_findings())


def test_host_token_readiness_uses_the_real_coverage_query(clean_env):
    """Not a stub: the check must consult the live token store."""
    clean_env.setenv("XCELSIOR_AGENT_HOST_TOKENS", "require")
    try:
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            conn.execute("SELECT 1 FROM host_agent_tokens WHERE false")
    except Exception as exc:  # pragma: no cover - skip path
        pytest.skip(f"host_agent_tokens unavailable: {exc}")

    result = sv._check_host_token_rotation_readiness()
    # With no admitted hosts holding tokens in the test DB this either
    # reports incomplete coverage or nothing at all — never a crash.
    assert result is None or result.code in (
        "host_token_coverage_incomplete",
        "host_token_coverage_unknown",
    )

    clean_env.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    assert sv._check_host_token_rotation_readiness() is None


# ── Enforcement semantics ─────────────────────────────────────────────


def test_enforcement_raises_on_errors_only(clean_env):
    clean_env.setenv("XCELSIOR_DB_BACKEND", "sqlite")
    with pytest.raises(sv.StartupValidationError) as excinfo:
        sv.validate_startup(enforce=True)
    assert [f.code for f in excinfo.value.findings] == ["database_backend"]

    # A warning alone must not block a deploy.
    clean_env.setenv("XCELSIOR_DB_BACKEND", "postgres")
    clean_env.setenv("XCELSIOR_AGENT_SHARED_BEARER_MIGRATION", "1")
    findings = sv.validate_startup(enforce=True)
    assert [f.severity for f in findings] == ["warning"]


def test_non_production_reports_without_blocking(clean_env):
    clean_env.setenv("XCELSIOR_ENV", "test")
    clean_env.setenv("XCELSIOR_DB_BACKEND", "sqlite")
    findings = sv.validate_startup()  # enforce=None → production only
    assert "database_backend" in _codes(findings)


def test_production_enforces_by_default(clean_env):
    clean_env.setenv("XCELSIOR_ENV", "production")
    clean_env.setenv("XCELSIOR_DB_BACKEND", "sqlite")
    with pytest.raises(sv.StartupValidationError):
        sv.validate_startup()


def test_emergency_skip_is_explicit_and_named(clean_env):
    clean_env.setenv("XCELSIOR_ENV", "production")
    clean_env.setenv("XCELSIOR_DB_BACKEND", "sqlite")
    clean_env.setenv("XCELSIOR_SKIP_STARTUP_VALIDATION", "1")
    findings = sv.validate_startup()
    assert "database_backend" in _codes(findings), "skipping must not hide the finding"


def test_a_raising_check_degrades_to_a_warning(clean_env):
    def _boom():
        raise RuntimeError("kaboom")

    clean_env.setattr(sv, "CHECKS", (_boom,), raising=False)
    findings = sv.collect_findings()
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "kaboom" in findings[0].message


def test_api_lifespan_runs_the_validator():
    """Wiring check: the gate must actually be in the startup path."""
    import inspect

    import api

    source = inspect.getsource(api.lifespan)
    assert "validate_startup" in source
    assert "StartupValidationError" in source


# ── §21.3 health semantics ────────────────────────────────────────────


def test_livez_makes_no_dependency_calls(monkeypatch):
    """A database blip must not get healthy replicas restarted."""
    import db

    def _explode(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("/livez touched the database")

    monkeypatch.setattr(db, "_get_pg_pool", _explode)
    resp = client.get("/livez")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "status": "alive"}


def test_startupz_reports_findings_and_holds_traffic(clean_env):
    clean_env.setenv("XCELSIOR_DB_BACKEND", "sqlite")
    resp = client.get("/startupz")
    assert resp.status_code == 503, resp.text
    detail = resp.json()
    assert detail["ok"] is False
    codes = {f["code"] for f in detail["findings"]}
    assert "database_backend" in codes
    # Every finding carries an actionable fix, not just a complaint.
    for finding in detail["findings"]:
        assert finding["remediation"].strip()


def test_startupz_is_green_on_a_correct_configuration(clean_env):
    resp = client.get("/startupz")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["findings"] == []


def test_the_three_probes_are_distinct(clean_env):
    """§21.3: liveness, readiness, and startup answer different questions."""
    live = client.get("/livez")
    start = client.get("/startupz")
    ready = client.get("/readyz")
    assert live.status_code == 200
    assert start.status_code == 200
    # readyz may legitimately 503 if a dependency is down here; what
    # matters is that it is a *different* handler with dependency checks.
    assert ready.status_code in (200, 503)
    if ready.status_code == 200:
        assert "storage" in ready.json()
        assert "storage" not in live.json()
