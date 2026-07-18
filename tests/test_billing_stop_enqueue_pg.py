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


def _mk_attempt_history(job_id, host_id, *, active=False):
    attempt_id = str(uuid.uuid4())
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO job_attempts
                   (attempt_id, job_id, attempt_number, status, host_id,
                    fencing_token, ended_at)
               VALUES (%s, %s, 1, %s, %s,
                       nextval('placement_fencing_token_seq'),
                       CASE WHEN %s THEN NULL ELSE clock_timestamp() END)""",
            (
                attempt_id,
                job_id,
                "running" if active else "cancelled",
                host_id,
                active,
            ),
        )
        if active:
            conn.execute(
                "UPDATE jobs SET active_attempt_id=%s WHERE job_id=%s",
                (attempt_id, job_id),
            )
        conn.commit()
    return attempt_id


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
    assert (
        "container_name" in args and args["container_name"]
    ), "container_name is required by worker drain handler"
    assert (
        "job_id" in args and args["job_id"]
    ), "job_id is required for worker callback / metrics correlation"


def test_start_fenced_history_requeues_without_start_container(
    cleanup_ids, captured_enqueue
):
    """P5.5 resume: stopped fenced job → queued for new attempt, no start_container."""
    _mkwallet(cleanup_ids, "v2-start@test")
    _mkhost(cleanup_ids, "h-v2-stopped")
    job_id = _mkjob(
        cleanup_ids,
        owner="v2-start@test",
        host_id="h-v2-stopped",
        status="stopped",
        container_name="xcl-v2-stopped",
    )
    _mk_attempt_history(job_id, "h-v2-stopped")

    from billing import BillingEngine

    result = BillingEngine().start_instance(job_id)
    assert result.get("started") is True, result
    assert result.get("status") == "queued"
    assert result.get("fresh_attempt") is True
    assert captured_enqueue == []
    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT status, host_id, active_attempt_id,
                      payload->>'lifecycle_intent'
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
    assert row[0] == "queued"
    assert row[1] is None
    assert row[2] is None
    assert row[3] == "resume"


def test_restart_active_fenced_attempt_enqueues_stop_with_restart_intent(
    cleanup_ids, captured_enqueue
):
    """P5.5 restart (running): fenced stop_attempt intent=restart, no legacy enqueue."""
    _mkwallet(cleanup_ids, "v2-restart@test")
    _mkhost(cleanup_ids, "h-v2-running")
    job_id = _mkjob(
        cleanup_ids,
        owner="v2-restart@test",
        host_id="h-v2-running",
        status="running",
        container_name="xcl-v2-running",
    )
    attempt_id = _mk_attempt_history(job_id, "h-v2-running", active=True)
    fence = _mk_active_lease(job_id, "h-v2-running", attempt_id)

    from billing import BillingEngine

    result = BillingEngine().restart_instance(job_id)
    assert result.get("restarted") is True, result
    assert result.get("status") == "stopping"
    assert result.get("attempt_id") == attempt_id
    assert captured_enqueue == []
    with _pool.connection() as conn:
        job = conn.execute(
            """SELECT status, payload->>'lifecycle_intent'
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        cmd = conn.execute(
            """SELECT command, args, fencing_token
                 FROM agent_commands WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
    assert job[0] == "stopping"
    assert job[1] == "restart"
    assert cmd[0] == "stop_attempt"
    assert cmd[1].get("preserve") is False
    assert cmd[1].get("intent") == "restart"
    assert cmd[2] == fence


def test_stop_active_fenced_attempt_uses_v2_command_and_stays_stopping(
    cleanup_ids, captured_enqueue
):
    _mkhost(cleanup_ids, "h-v2-stop")
    job_id = _mkjob(
        cleanup_ids,
        owner="v2-stop@test",
        host_id="h-v2-stop",
        status="running",
        container_name="xcl-v2-stop",
    )
    attempt_id = _mk_attempt_history(job_id, "h-v2-stop", active=True)
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT fencing_token FROM job_attempts WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO placement_leases
                   (job_id, attempt_id, host_id, fencing_token, status,
                    claim_deadline, claimed_at, last_renewed_at, expires_at)
               VALUES (%s, %s, %s, %s, 'active',
                       clock_timestamp() + interval '10 minutes',
                       clock_timestamp(), clock_timestamp(),
                       clock_timestamp() + interval '10 minutes')""",
            (job_id, attempt_id, "h-v2-stop", fence),
        )
        conn.commit()

    from billing import BillingEngine

    result = BillingEngine().stop_instance(job_id, "user_stopped")
    assert result == {"stopped": True, "job_id": job_id, "status": "stopping"}
    assert captured_enqueue == []
    with _pool.connection() as conn:
        job_status = conn.execute(
            "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
        command = conn.execute(
            """SELECT command, attempt_id, fencing_token, status, args
                 FROM agent_commands
                WHERE job_id=%s AND command='stop_attempt'""",
            (job_id,),
        ).fetchone()
    assert job_status == "stopping"
    assert command[0] == "stop_attempt"
    assert str(command[1]) == attempt_id
    assert command[2] == fence
    assert command[3] == "pending"
    assert command[4]["job_id"] == job_id


def _mk_active_lease(job_id, host_id, attempt_id):
    with _pool.connection() as conn:
        fence = conn.execute(
            "SELECT fencing_token FROM job_attempts WHERE attempt_id=%s",
            (attempt_id,),
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO placement_leases
                   (job_id, attempt_id, host_id, fencing_token, status,
                    claim_deadline, claimed_at, last_renewed_at, expires_at)
               VALUES (%s, %s, %s, %s, 'active',
                       clock_timestamp() + interval '10 minutes',
                       clock_timestamp(), clock_timestamp(),
                       clock_timestamp() + interval '10 minutes')""",
            (job_id, attempt_id, host_id, fence),
        )
        conn.commit()
    return fence


def test_terminate_active_fenced_attempt_enqueues_stop_remove(
    cleanup_ids, captured_enqueue
):
    """P5.5: attempt-owned terminate → fenced stop_attempt (preserve=False)."""
    host_id = "h-v2-terminate"
    _mkhost(cleanup_ids, host_id)
    job_id = _mkjob(
        cleanup_ids,
        owner="v2-terminate@test",
        host_id=host_id,
        status="running",
        container_name="xcl-v2-terminate",
    )
    attempt_id = _mk_attempt_history(job_id, host_id, active=True)
    fence = _mk_active_lease(job_id, host_id, attempt_id)

    from billing import BillingEngine

    result = BillingEngine().terminate_instance(job_id)
    assert result.get("terminated") is True, result
    assert result.get("status") == "stopping"
    assert result.get("attempt_id") == attempt_id
    assert result.get("command_id")
    # Must not fall through to legacy unfenced agent enqueue.
    assert captured_enqueue == []

    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT status, payload->>'lifecycle_intent' AS intent,
                      active_attempt_id::text
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
        command = conn.execute(
            """SELECT command, attempt_id::text, fencing_token, status, args
                 FROM agent_commands
                WHERE job_id=%s AND command='stop_attempt'
                ORDER BY created_at DESC LIMIT 1""",
            (job_id,),
        ).fetchone()
        vol_detach_risk = conn.execute(
            "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()[0]

    assert row[0] == "stopping"
    assert row[1] == "terminate"
    assert row[2] == attempt_id  # authority not cleared pre-ACK
    assert command is not None
    assert command[0] == "stop_attempt"
    assert command[1] == attempt_id
    assert command[2] == fence
    assert command[3] == "pending"
    assert command[4].get("preserve") is False
    assert command[4].get("intent") == "terminate"
    # Intermediate only — not terminal before worker proof.
    assert vol_detach_risk == "stopping"


def test_terminate_idempotent_second_call_reuses_command(
    cleanup_ids, captured_enqueue
):
    host_id = "h-v2-term-idem"
    _mkhost(cleanup_ids, host_id)
    job_id = _mkjob(
        cleanup_ids,
        owner="v2-term-idem@test",
        host_id=host_id,
        status="running",
        container_name="xcl-v2-term-idem",
    )
    attempt_id = _mk_attempt_history(job_id, host_id, active=True)
    _mk_active_lease(job_id, host_id, attempt_id)

    from billing import BillingEngine

    first = BillingEngine().terminate_instance(job_id)
    second = BillingEngine().terminate_instance(job_id)
    assert first.get("terminated") and second.get("terminated")
    with _pool.connection() as conn:
        n_cmd = conn.execute(
            """SELECT COUNT(*) FROM agent_commands
                WHERE job_id=%s AND command='stop_attempt'""",
            (job_id,),
        ).fetchone()[0]
        n_bc = conn.execute(
            """SELECT COUNT(*) FROM billing_cycles
                WHERE job_id=%s AND status='terminated'
                  AND cycle_id LIKE 'BC-term-%%'""",
            (job_id,),
        ).fetchone()[0]
    # Same attempt idempotency key → one durable command row.
    assert n_cmd == 1
    # Second terminate must not insert another billing anchor.
    assert n_bc == 1


def test_terminate_enqueue_refuse_leaves_job_running(cleanup_ids, captured_enqueue):
    """No active lease → no command and job stays running (not stuck stopping)."""
    host_id = "h-v2-term-nolease"
    _mkhost(cleanup_ids, host_id)
    job_id = _mkjob(
        cleanup_ids,
        owner="v2-term-nolease@test",
        host_id=host_id,
        status="running",
        container_name="xcl-v2-term-nolease",
    )
    # Active attempt but intentionally no placement_lease.
    _mk_attempt_history(job_id, host_id, active=True)

    from billing import BillingEngine

    result = BillingEngine().terminate_instance(job_id)
    assert result.get("terminated") is False
    assert result.get("reason") == "no_active_fenced_authority"
    with _pool.connection() as conn:
        status = conn.execute(
            "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
        n_cmd = conn.execute(
            "SELECT COUNT(*) FROM agent_commands WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
        intent = conn.execute(
            "SELECT payload->>'lifecycle_intent' FROM jobs WHERE job_id=%s",
            (job_id,),
        ).fetchone()[0]
        n_bc = conn.execute(
            """SELECT COUNT(*) FROM billing_cycles
                WHERE job_id=%s AND cycle_id LIKE 'BC-term-%%'""",
            (job_id,),
        ).fetchone()[0]
    assert status == "running"
    assert n_cmd == 0
    assert intent is None
    assert n_bc == 0
    assert captured_enqueue == []
