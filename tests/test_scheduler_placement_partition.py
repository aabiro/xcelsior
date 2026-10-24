"""Exclusive transactional placement and host pins against real PostgreSQL."""

from __future__ import annotations

from control_plane.state import initial_desired_state, job_phase
from tests.native_jobs import admit_fixture_job

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _migrated = (
            _c.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='jobs' AND column_name='phase'"
            ).fetchone()
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database not migrated to >= 059")

from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.service import SchedulerService


@pytest.fixture
def fleet():
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute(
                "DELETE FROM gpu_device_allocations WHERE attempt_id IN "
                "(SELECT attempt_id FROM job_attempts WHERE job_id=%s)",
                (jid,),
            )
            conn.execute("DELETE FROM placement_leases WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM job_attempts WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM host_gpu_devices WHERE host_id=%s", (hid,))
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


@pytest.fixture
def reset_cp_config(monkeypatch):
    """Clear the cached SchedulerConfig so env changes take effect."""
    import scheduler as sched

    monkeypatch.setattr(sched, "_cp_config", None)
    monkeypatch.setattr(sched, "_cp_service", None)
    yield
    monkeypatch.setattr(sched, "_cp_config", None)
    monkeypatch.setattr(sched, "_cp_service", None)


def _mk_host(fleet, host_id, *, gpu_model, gpu_count=2, total_vram_gb=48.0):
    payload = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "free_vram_gb": total_vram_gb,
        "total_vram_gb": total_vram_gb,
        "cost_per_hour": 1.0,
        "admitted": True,
        "capabilities": {"egress_isolation": True},
        "last_seen": time.time(),
        "ip": "127.0.0.1",
        "status": "active",
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload, admission_state, administrative_state, availability_state, capabilities)
               VALUES (%s, 'active', %s, %s, 'admitted', 'admitted', 'ready', '{"egress_isolation":true}')""",
            (host_id, time.time(), json.dumps(payload)),
        )
        conn.commit()
    fleet["hosts"].append(host_id)
    return payload


def _mk_job(
    fleet,
    job_id,
    *,
    gpu_model,
    num_gpus=1,
    vram_needed_gb=8.0,
    priority=10,
    status="queued",
    scheduler=None,
):
    payload = {
        "job_id": job_id,
        "name": job_id,
        "gpu_model": gpu_model,
        "num_gpus": num_gpus,
        "vram_needed_gb": vram_needed_gb,
        "priority": priority,
        "status": status,
        "submitted_at": time.time(),
    }
    if scheduler:
        payload["scheduler"] = scheduler
    with _pool.connection() as conn:
        fixture_submitted_at = time.time()
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, phase, desired_state, effective_priority, queued_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CASE WHEN %s IN ('queued','preempted') THEN to_timestamp(%s) ELSE NULL END)""",
            (
                job_id,
                status,
                priority,
                fixture_submitted_at,
                json.dumps(payload),
                job_phase(status),
                initial_desired_state(status),
                priority,
                status,
                fixture_submitted_at,
            ),
        )
        admit_fixture_job(conn, job_id)
        conn.commit()
    fleet["jobs"].append(job_id)
    return payload


def _job_row(job_id):
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT job_id, status, host_id, active_attempt_id, phase " "FROM jobs WHERE job_id=%s",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return {
        "job_id": row[0],
        "status": row[1],
        "host_id": row[2],
        "active_attempt_id": row[3],
        "phase": row[4],
    }


def _cfg_canary(marker, hosts):
    return SchedulerConfig(
        mode=SchedulerMode.CANARY,
        replica_id=f"place-part-{marker}",
        canary_gpu_models=frozenset({f"rtx-{marker}".lower()}),
        canary_host_ids=frozenset(h.lower() for h in hosts),
        tick_max_placements=5,
    )


class TestLegacyQueueSkipsOwnedJobs:
    def test_process_queue_skips_canary_owned_job(self, fleet, reset_cp_config, monkeypatch):
        """Under canary, legacy process_queue must not assign owned jobs."""
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        owned_job = f"j-{marker}-owned"
        free_job = f"j-{marker}-free"
        free_model = f"RTX-FREE-{marker}"

        _mk_host(fleet, host_id, gpu_model=model)
        free_host = f"h-{marker}-free"
        _mk_host(fleet, free_host, gpu_model=free_model)
        _mk_job(fleet, owned_job, gpu_model=model)
        _mk_job(fleet, free_job, gpu_model=free_model)

        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "canary")
        monkeypatch.setenv("XCELSIOR_SCHEDULER_CANARY_GPU_MODELS", model.lower())
        monkeypatch.setenv("XCELSIOR_SCHEDULER_CANARY_HOSTS", host_id)

        import scheduler as sched

        # Ensure config reloads from env.
        monkeypatch.setattr(sched, "_cp_config", None)

        assert sched._control_plane_owns_job({"gpu_model": model, "job_id": owned_job})
        assert not sched._control_plane_owns_job({"gpu_model": free_model, "job_id": free_job})

        assigned = sched.process_queue()
        assigned_ids = {
            (a[0]["job_id"] if isinstance(a, tuple) else a.get("job_id")) for a in assigned
        }
        # owned must not appear; free may be assigned by legacy.
        assert owned_job not in assigned_ids
        assert _job_row(owned_job)["status"] == "queued"
        assert _job_row(owned_job)["active_attempt_id"] is None

    def test_process_queue_active_owns_everything(self, fleet, reset_cp_config, monkeypatch):
        """Under active, every job is owned — legacy process_queue is a no-op."""
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        job_id = f"j-{marker}-all"
        _mk_host(fleet, host_id, gpu_model=model)
        _mk_job(fleet, job_id, gpu_model=model)

        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
        import scheduler as sched

        monkeypatch.setattr(sched, "_cp_config", None)

        assigned = sched.process_queue()
        assert assigned == []
        assert _job_row(job_id)["status"] == "queued"

        # Transactional tick is the sole writer.
        report = SchedulerService(
            SchedulerConfig(
                mode=SchedulerMode.ACTIVE,
                replica_id=f"place-act-{marker}",
                tick_max_placements=5,
            )
        ).tick()
        assert any(p.job_id == job_id for p in report.placements), report
        row = _job_row(job_id)
        assert row["status"] == "assigned"
        assert row["active_attempt_id"] is not None

        with _pool.connection() as conn:
            lease = conn.execute(
                "SELECT status FROM placement_leases WHERE job_id=%s",
                (job_id,),
            ).fetchone()
            cmd = conn.execute(
                "SELECT command, status FROM agent_commands WHERE job_id=%s",
                (job_id,),
            ).fetchone()
        assert lease is not None
        assert cmd is not None

    def test_all_legacy_walkers_skip_owned(self, fleet, reset_cp_config, monkeypatch):
        """binpack / filtered / ranked walkers honor the same partition."""
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        job_id = f"j-{marker}-walk"
        _mk_host(fleet, host_id, gpu_model=model)
        _mk_job(fleet, job_id, gpu_model=model, scheduler="v2")

        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "canary")
        # Empty GPU models — ownership comes from payload scheduler=v2.
        monkeypatch.setenv("XCELSIOR_SCHEDULER_CANARY_GPU_MODELS", "")
        import scheduler as sched

        monkeypatch.setattr(sched, "_cp_config", None)

        for walker in (
            sched.process_queue,
            sched.process_queue_binpack,
            lambda: sched.process_queue_filtered(),
            lambda: sched.process_queue_ranked(),
        ):
            walker()
            assert _job_row(job_id)["status"] == "queued", walker
            assert _job_row(job_id)["active_attempt_id"] is None


@pytest.mark.parametrize("mode", ["active", "canary", "paused"])
def test_host_pin_is_persisted_before_any_assignment(fleet, reset_cp_config, monkeypatch, mode):
    """Only a later transactional tick may create the attempt and start command."""
    import scheduler as sched

    marker = uuid.uuid4().hex[:8]
    model = f"RTX-{marker}"
    pinned, other = f"h-{marker}-pin", f"h-{marker}-other"
    _mk_host(fleet, pinned, gpu_model=model)
    _mk_host(fleet, other, gpu_model=model)
    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", mode)
    monkeypatch.setattr(sched, "_cp_config", None)
    job = sched.submit_job("pinned", 8, gpu_model=model, preferred_host_id=pinned)
    jid = job["job_id"]
    fleet["jobs"].append(jid)
    before = _job_row(jid)
    assert before["status"] == "queued"
    assert before["host_id"] is None
    assert before["active_attempt_id"] is None
    with _pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT payload->>'preferred_host_id' FROM jobs WHERE job_id=%s", (jid,)
            ).fetchone()[0]
            == pinned
        )
        assert (
            conn.execute("SELECT count(*) FROM agent_commands WHERE job_id=%s", (jid,)).fetchone()[
                0
            ]
            == 0
        )

    report = SchedulerService(_cfg_canary(marker, [pinned, other])).tick()
    assert any(p.job_id == jid for p in report.placements)
    after = _job_row(jid)
    assert after["host_id"] == pinned
    assert after["active_attempt_id"] is not None
    with _pool.connection() as conn:
        commands = conn.execute(
            "SELECT host_id,command FROM agent_commands WHERE job_id=%s", (jid,)
        ).fetchall()
    assert commands == [(pinned, "start_attempt")]


def test_unavailable_pinned_host_does_not_fall_back(fleet, reset_cp_config):
    import scheduler as sched

    marker = uuid.uuid4().hex[:8]
    model, other = f"RTX-{marker}", f"h-{marker}-other"
    _mk_host(fleet, other, gpu_model=model)
    job = sched.submit_job("missing-pin", 8, gpu_model=model, preferred_host_id="absent-host")
    jid = job["job_id"]
    fleet["jobs"].append(jid)
    SchedulerService(_cfg_canary(marker, [other])).tick()
    assert _job_row(jid)["status"] == "queued"
    assert _job_row(jid)["active_attempt_id"] is None


@pytest.mark.parametrize("mode", ["paused", "canary"])
@pytest.mark.parametrize("pin_available", [True, False])
def test_non_native_queue_respects_host_pin_without_fallback(
    fleet, reset_cp_config, monkeypatch, mode, pin_available
):
    import scheduler as sched
    from psycopg.types.json import Jsonb

    marker = uuid.uuid4().hex[:10]
    model = f"pin-model-{marker}"
    pin, other, jid = f"pin-{marker}", f"other-{marker}", f"job-{marker}"
    _mk_host(fleet, other, gpu_model=model)
    if pin_available:
        _mk_host(fleet, pin, gpu_model=model)
    _mk_job(fleet, jid, gpu_model=model)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload||%s WHERE job_id=%s",
            (Jsonb({"preferred_host_id": pin}), jid),
        )
    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", mode)
    monkeypatch.setenv("XCELSIOR_SCHEDULER_CANARY_GPU_MODELS", "different-model")
    monkeypatch.setenv("XCELSIOR_SCHEDULER_CANARY_HOSTS", "different-host")
    sched.process_queue()
    row = _job_row(jid)
    assert row["host_id"] == (pin if pin_available else None)
    assert row["status"] == ("assigned" if pin_available else "queued")
