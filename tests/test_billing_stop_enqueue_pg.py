"""
Real PG-backed integration tests for billing.py stop/start/pause enqueue contract.

Augments the static-source guard in tests/test_billing_stop_enqueue.py. These
tests drive BillingService methods against real PostgreSQL rows and assert
that the agent command queue receives the right command type, args, and
created_by tag at every billing call site.

Skips automatically if no PG pool is available (CI without postgres service).
"""
import json
import os
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    # Smoke-test the connection so we skip cleanly on PG-down rather than
    # explode inside fixtures.
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def cleanup_ids():
    """Track inserted PK rows and delete them after the test."""
    ids = {
        "jobs": [],
        "hosts": [],
        "wallets": [],
        "agent_commands_hosts": [],
    }
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM billing_cycles WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["agent_commands_hosts"]:
            conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (hid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        for cid in ids["wallets"]:
            conn.execute("DELETE FROM wallets WHERE customer_id=%s", (cid,))
        conn.commit()


def _mkhost(cleanup, host_id):
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'online', %s, %s)
               ON CONFLICT (host_id) DO NOTHING""",
            (host_id, time.time(), json.dumps({"name": host_id})),
        )
        conn.commit()
    cleanup["hosts"].append(host_id)
    cleanup["agent_commands_hosts"].append(host_id)


def _mkjob(
    cleanup,
    *,
    owner="alice@test",
    host_id="h-test",
    status="running",
    container_name="c-test",
    extra_payload=None,
):
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    payload = {
        "owner": owner,
        "name": job_id,
        "container_name": container_name,
    }
    if extra_payload:
        payload.update(extra_payload)
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, payload)
               VALUES (%s, %s, 0, %s, %s, %s)""",
            (job_id, status, time.time(), host_id, json.dumps(payload)),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mkwallet(cleanup, customer_id, balance_cad=10.0):
    now = time.time()
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO wallets
               (customer_id, balance_cad, total_deposited_cad, total_spent_cad,
                total_refunded_cad, status, created_at, updated_at)
               VALUES (%s, %s, %s, 0, 0, 'active', %s, %s)
               ON CONFLICT (customer_id) DO UPDATE SET balance_cad=EXCLUDED.balance_cad""",
            (customer_id, balance_cad, balance_cad, now, now),
        )
        conn.commit()
    cleanup["wallets"].append(customer_id)


@pytest.fixture
def captured_enqueue(monkeypatch):
    """Patch routes.agent.enqueue_agent_command and record every call."""
    calls = []

    def _fake(host_id, command, args, *, created_by, expires_in=900):
        calls.append(
            {
                "host_id": host_id,
                "command": command,
                "args": dict(args) if args else {},
                "created_by": created_by,
                "expires_in": expires_in,
            }
        )
        return f"cmd-fake-{len(calls)}"

    import routes.agent as _agent_mod

    monkeypatch.setattr(_agent_mod, "enqueue_agent_command", _fake)
    return calls


# ── Tests ────────────────────────────────────────────────────────────


def test_stop_instance_enqueues_pause_container_with_billing_stop_tag(
    cleanup_ids, captured_enqueue
):
    """BillingService.stop_instance MUST use pause_container (NOT stop_container)
    so the state machine STOPPED → RESTARTING → RUNNING can resume the same
    container. Tag must be billing_stop so it can be distinguished from pause/resume."""
    _mkhost(cleanup_ids, "h-stop")
    job_id = _mkjob(
        cleanup_ids,
        owner="dan@test",
        host_id="h-stop",
        status="running",
        container_name="xcl-dan-1",
    )

    from billing import BillingEngine

    svc = BillingEngine()
    result = svc.stop_instance(job_id, "user_stopped")

    assert result.get("stopped") is True, f"expected stopped=True, got {result!r}"
    assert len(captured_enqueue) == 1
    call = captured_enqueue[0]
    assert call["command"] == "pause_container", (
        "STOP must use pause_container so RESTARTING→RUNNING can `docker start` "
        "the preserved container; stop_container would `docker rm -f` and break restart"
    )
    assert call["created_by"] == "billing_stop"
    assert call["args"]["job_id"] == job_id


def test_stop_instance_rejects_invalid_reason(cleanup_ids, captured_enqueue):
    """Invalid reasons must short-circuit BEFORE any DB write or enqueue."""
    from billing import BillingEngine

    svc = BillingEngine()
    result = svc.stop_instance("never-existed", "totally_made_up_reason")

    assert result.get("stopped") is False
    assert "invalid_reason" in result.get("reason", "")
    assert captured_enqueue == [], "must not enqueue on invalid reason"


def test_stop_skips_enqueue_when_job_not_running(cleanup_ids, captured_enqueue):
    """If job is already stopped/completed/never-running, stop must no-op
    cleanly without enqueueing a duplicate command."""
    _mkhost(cleanup_ids, "h-skip")
    job_id = _mkjob(
        cleanup_ids,
        owner="eve@test",
        host_id="h-skip",
        status="completed",  # NOT running
        container_name="xcl-eve-1",
    )

    from billing import BillingEngine

    svc = BillingEngine()
    result = svc.stop_instance(job_id, "user_stopped")

    assert result.get("stopped") is False
    assert result.get("reason") == "not_running"
    assert captured_enqueue == []


def test_billing_stop_args_shape_complete(cleanup_ids, captured_enqueue):
    """Contract guard: every billing-originated lifecycle command MUST carry
    container_name AND job_id in args. Missing either breaks the worker handler."""
    _mkhost(cleanup_ids, "h-shape")
    job_id = _mkjob(
        cleanup_ids,
        owner="frank@test",
        host_id="h-shape",
        status="running",
        container_name="xcl-frank-1",
    )

    from billing import BillingEngine

    svc = BillingEngine()
    svc.stop_instance(job_id, "user_stopped")

    assert len(captured_enqueue) == 1
    args = captured_enqueue[0]["args"]
    assert "container_name" in args and args["container_name"], (
        "container_name is required by worker drain handler"
    )
    assert "job_id" in args and args["job_id"], (
        "job_id is required for worker callback / metrics correlation"
    )
