"""Gate tests for B8.2: Service health semantics completion (§21.3).

Verifies per-service readiness transitions:
- scheduler readiness proves DB primitives, schema compatibility, queue access, and heartbeat.
- reconciler readiness proves DB connection, queue access, and heartbeat.
- MCP readiness proves API auth metadata/JWKS reachable, Redis reachable, and tool registry complete.
- worker readiness proves identity, API, GPU runtime, inventory, and capability probes.
- PID-string health checks are strictly absent.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from control_plane.readiness import (
    check_scheduler_readiness,
    check_reconciler_readiness,
    check_mcp_readiness,
    check_worker_readiness,
)

ROOT = Path(__file__).resolve().parent.parent


def test_scheduler_readiness_transitions_on_dependency_break_and_repair():
    """Scheduler readiness is false when DB/heartbeat fails and true when repaired."""
    # 1. Broken: DB connection error
    broken_conn = MagicMock()
    broken_conn.execute.side_effect = RuntimeError("Database connection refused")
    ok, details = check_scheduler_readiness(conn=broken_conn)
    assert ok is False
    assert details["db_connected"] is False

    # 2. Broken: Stale heartbeat
    stale_conn = MagicMock()
    stale_conn.execute.return_value.fetchone.side_effect = [
        (1,),        # SELECT 1
        (120.0,),    # Heartbeat age = 120s (stale, threshold 60s)
        (0,),        # count jobs
    ]
    with patch("control_plane.schema_compat.assert_schema_compatible"):
        ok, details = check_scheduler_readiness(conn=stale_conn, max_heartbeat_age_sec=60.0)
    assert ok is False
    assert details["db_connected"] is True
    assert details["heartbeat_fresh"] is False

    # 3. Repaired: All healthy
    healthy_conn = MagicMock()
    healthy_conn.execute.return_value.fetchone.side_effect = [
        (1,),        # SELECT 1
        (5.0,),      # Heartbeat age = 5s (fresh)
        (0,),        # count jobs
    ]
    with patch("control_plane.schema_compat.assert_schema_compatible"):
        ok, details = check_scheduler_readiness(conn=healthy_conn, max_heartbeat_age_sec=60.0)
    assert ok is True
    assert details["db_connected"] is True
    assert details["schema_compatible"] is True
    assert details["heartbeat_fresh"] is True
    assert details["queue_accessible"] is True


def test_reconciler_readiness_transitions_on_dependency_break_and_repair():
    """Reconciler readiness is false when queue/heartbeat fails and true when repaired."""
    # 1. Broken: Heartbeat missing/stale
    stale_conn = MagicMock()
    stale_conn.execute.return_value.fetchone.side_effect = [
        (1,),        # SELECT 1
        (999.0,),    # Heartbeat age = 999s (stale)
        (0,),        # count queue
    ]
    ok, details = check_reconciler_readiness(conn=stale_conn, max_heartbeat_age_sec=60.0)
    assert ok is False
    assert details["heartbeat_fresh"] is False

    # 2. Broken: Queue inaccessible
    queue_broken_conn = MagicMock()
    queue_broken_conn.execute.return_value.fetchone.side_effect = [
        (1,),        # SELECT 1
        (5.0,),      # Heartbeat fresh
        RuntimeError("relation reconciliation_queue does not exist"),
    ]
    ok, details = check_reconciler_readiness(conn=queue_broken_conn, max_heartbeat_age_sec=60.0)
    assert ok is False
    assert details["queue_accessible"] is False

    # 3. Repaired: All healthy
    healthy_conn = MagicMock()
    healthy_conn.execute.return_value.fetchone.side_effect = [
        (1,),        # SELECT 1
        (10.0,),     # Heartbeat fresh
        (0,),        # count queue
    ]
    ok, details = check_reconciler_readiness(conn=healthy_conn, max_heartbeat_age_sec=60.0)
    assert ok is True
    assert details["db_connected"] is True
    assert details["heartbeat_fresh"] is True
    assert details["queue_accessible"] is True


def test_mcp_readiness_transitions_on_dependency_break_and_repair():
    """MCP readiness is false when any check fails and true when all pass."""
    # 1. Broken: Redis down or JWKS failed
    class FakeBrokenResp:
        status_code = 503
        def json(self):
            return {
                "ok": False,
                "checks": {
                    "redis": True,
                    "authorization_server": True,
                    "jwks": False,
                    "tool_registry": True,
                },
            }

    with patch("httpx.Client.get", return_value=FakeBrokenResp()):
        ok, details = check_mcp_readiness("http://localhost:8770/readyz")
    assert ok is False
    assert details["checks"]["jwks"] is False

    # 2. Repaired: All pass
    class FakeHealthyResp:
        status_code = 200
        def json(self):
            return {
                "ok": True,
                "checks": {
                    "redis": True,
                    "authorization_server": True,
                    "jwks": True,
                    "tool_registry": True,
                },
            }

    with patch("httpx.Client.get", return_value=FakeHealthyResp()):
        ok, details = check_mcp_readiness("http://localhost:8770/readyz")
    assert ok is True
    assert details["checks"]["tool_registry"] is True


def test_worker_readiness_transitions_on_dependency_break_and_repair():
    """Worker readiness is false when identity/API/GPU fails and true when repaired."""
    # 1. Broken: Missing identity
    ok, details = check_worker_readiness(config={"API_URL": "http://api:8000"}, probe_gpu=False)
    assert ok is False
    assert details["identity_configured"] is False

    # 2. Broken: API unreachable
    class FakeApiDown:
        status_code = 502

    with patch("httpx.Client.get", return_value=FakeApiDown()):
        ok, details = check_worker_readiness(
            config={"HOST_ID": "h-1", "API_URL": "http://api:8000"},
            probe_gpu=False,
        )
    assert ok is False
    assert details["api_reachable"] is False

    # 3. Repaired: All dependencies healthy
    class FakeApiUp:
        status_code = 200

    with patch("httpx.Client.get", return_value=FakeApiUp()):
        ok, details = check_worker_readiness(
            config={"HOST_ID": "h-1", "API_URL": "http://api:8000"},
            probe_gpu=False,
        )
    assert ok is True
    assert details["identity_configured"] is True
    assert details["api_reachable"] is True
    assert details["gpu_runtime_available"] is True


def test_pid_string_health_checks_are_completely_removed():
    """B8.2: PID-string health checks (/proc/1/cmdline) must not appear in compose."""
    compose_path = ROOT / "docker-compose.yml"
    assert compose_path.exists()
    text = compose_path.read_text()
    assert "/proc/1/cmdline" not in text
    assert "grep -qa" not in text
