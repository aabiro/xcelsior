"""Exclusive placement partition + no SSH double-start for control-plane-owned jobs.

Drives the *shipped* legacy entry points (``process_queue``,
``process_queue_binpack``, ``process_queue_filtered``,
``process_queue_sovereign``, ``process_assigned``) against a real Postgres
with the transactional scheduler in canary/active mode. Proves:

1. Owned jobs are never assigned by legacy queue walkers.
2. Only the transactional tick creates attempt + lease + command for
   owned work.
3. ``process_assigned`` never SSH-starts attempt-owned jobs (and fails
   closed when ownership lookup fails).
"""

from __future__ import annotations

import json
import time
import uuid
from unittest.mock import patch

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _migrated = (
            _c.execute(
                "SELECT tgname FROM pg_trigger "
                "WHERE tgname='trg_jobs_control_plane_projection'"
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
            conn.execute(
                "DELETE FROM host_gpu_devices WHERE host_id=%s", (hid,)
            )
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
        "last_seen": time.time(),
        "ip": "127.0.0.1",
        "status": "active",
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, 'active', %s, %s)""",
            (host_id, time.time(), json.dumps(payload)),
        )
        conn.commit()
    fleet["hosts"].append(host_id)
    return payload


def _mk_job(fleet, job_id, *, gpu_model, num_gpus=1, vram_needed_gb=8.0,
            priority=10, status="queued", scheduler=None):
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
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
               VALUES (%s, %s, %s, %s, %s)""",
            (job_id, status, priority, time.time(), json.dumps(payload)),
        )
        conn.commit()
    fleet["jobs"].append(job_id)
    return payload


def _job_row(job_id):
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT job_id, status, host_id, active_attempt_id, phase "
            "FROM jobs WHERE job_id=%s",
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
    def test_process_queue_skips_canary_owned_job(
        self, fleet, reset_cp_config, monkeypatch
    ):
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
        monkeypatch.setenv(
            "XCELSIOR_SCHEDULER_CANARY_GPU_MODELS", model.lower()
        )
        monkeypatch.setenv("XCELSIOR_SCHEDULER_CANARY_HOSTS", host_id)

        import scheduler as sched

        # Ensure config reloads from env.
        monkeypatch.setattr(sched, "_cp_config", None)

        assert sched._control_plane_owns_job(
            {"gpu_model": model, "job_id": owned_job}
        )
        assert not sched._control_plane_owns_job(
            {"gpu_model": free_model, "job_id": free_job}
        )

        assigned = sched.process_queue()
        assigned_ids = {
            (a[0]["job_id"] if isinstance(a, tuple) else a.get("job_id"))
            for a in assigned
        }
        # owned must not appear; free may be assigned by legacy.
        assert owned_job not in assigned_ids
        assert _job_row(owned_job)["status"] == "queued"
        assert _job_row(owned_job)["active_attempt_id"] is None

    def test_process_queue_active_owns_everything(
        self, fleet, reset_cp_config, monkeypatch
    ):
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
        assert any(p.job_id == job_id for p in report.placements)
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

    def test_all_legacy_walkers_skip_owned(
        self, fleet, reset_cp_config, monkeypatch
    ):
        """binpack / filtered / sovereign walkers honor the same partition."""
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
            lambda: sched.process_queue_filtered(canada_only=False),
            lambda: sched.process_queue_sovereign(canada_only=False),
        ):
            walker()
            assert _job_row(job_id)["status"] == "queued", walker
            assert _job_row(job_id)["active_attempt_id"] is None


class TestDirectHostLaunchDefersOwned:
    """api_submit_instance host-pin path must not dual-write under canary/active."""

    def test_direct_launch_defers_owned_job_no_assign_no_ssh(
        self, fleet, reset_cp_config, monkeypatch
    ):
        """Shipped ``_direct_host_launch_or_defer`` refuses owned work.

        Under canary, a host-pinned job matching the partition must stay
        queued, never receive ``assigned`` via update_job_status, and never
        call ``run_job`` (SSH). preferred_host_id is recorded for the
        transactional path.
        """
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-pin"
        job_id = f"j-{marker}-pin"
        _mk_host(fleet, host_id, gpu_model=model)
        _mk_job(fleet, job_id, gpu_model=model)

        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "canary")
        monkeypatch.setenv(
            "XCELSIOR_SCHEDULER_CANARY_GPU_MODELS", model.lower()
        )
        import scheduler as sched

        monkeypatch.setattr(sched, "_cp_config", None)

        job = {
            "job_id": job_id,
            "gpu_model": model,
            "status": "queued",
            "name": job_id,
        }
        assert not sched.legacy_inline_placement_allowed(job)

        run_job_calls: list = []
        assign_calls: list = []

        def _spy_run_job(*a, **k):
            run_job_calls.append((a, k))
            return "should-not-start"

        def _spy_update(job_id_arg, status, host_id=None, **kw):
            assign_calls.append((job_id_arg, status, host_id))
            # Must not advance owned jobs to assigned on this path.
            return {"job_id": job_id_arg, "status": status, "host_id": host_id}

        monkeypatch.setattr(sched, "run_job", _spy_run_job)
        # Patch where the launch helper imports them from.
        import routes.instances as inst_mod

        monkeypatch.setattr(inst_mod, "run_job", _spy_run_job)
        monkeypatch.setattr(inst_mod, "update_job_status", _spy_update)

        host_payload = {
            "host_id": host_id,
            "gpu_model": model,
            "status": "active",
            "free_vram_gb": 48.0,
        }
        result = inst_mod._direct_host_launch_or_defer(
            job,
            target_host_id=host_id,
            target_host=host_payload,
            image="alpine:latest",
            pricing_mode="on_demand",
            num_gpus=1,
        )
        assert run_job_calls == []
        # No assigned transition for owned job on the direct path.
        assert not any(
            c[0] == job_id and c[1] == "assigned" for c in assign_calls
        )
        row = _job_row(job_id)
        assert row["status"] == "queued"
        assert row["active_attempt_id"] is None
        # preferred pin recorded on payload
        with _pool.connection() as conn:
            payload = conn.execute(
                "SELECT payload FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        p = payload[0] if not isinstance(payload, dict) else payload["payload"]
        if isinstance(p, str):
            p = json.loads(p)
        assert p.get("preferred_host_id") == host_id
        assert result is not None

    def test_direct_launch_still_runs_for_unowned_legacy_job(
        self, fleet, reset_cp_config, monkeypatch
    ):
        """Paused/shadow mode (or out-of-canary GPU) keeps legacy direct launch."""
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-LEGACY-{marker}"
        host_id = f"h-{marker}-leg"
        job_id = f"j-{marker}-leg"
        _mk_host(fleet, host_id, gpu_model=model)
        _mk_job(fleet, job_id, gpu_model=model)

        monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "canary")
        # Canary only owns a different model — this job is free for legacy.
        monkeypatch.setenv(
            "XCELSIOR_SCHEDULER_CANARY_GPU_MODELS", "rtx-canary-only"
        )
        import scheduler as sched

        monkeypatch.setattr(sched, "_cp_config", None)

        job = {
            "job_id": job_id,
            "gpu_model": model,
            "status": "queued",
            "name": job_id,
        }
        assert sched.legacy_inline_placement_allowed(job)

        run_job_calls: list = []
        import routes.instances as inst_mod

        def _fake_run(job_arg, host, docker_image=None):
            run_job_calls.append(job_arg.get("job_id"))
            return f"ctr-{job_arg.get('job_id')}"

        monkeypatch.setattr(inst_mod, "run_job", _fake_run)
        # list_hosts must see our host for the assign path.
        monkeypatch.setattr(
            inst_mod,
            "list_hosts",
            lambda active_only=True: [
                {
                    "host_id": host_id,
                    "gpu_model": model,
                    "status": "active",
                    "free_vram_gb": 48.0,
                    "total_vram_gb": 48.0,
                    "admitted": True,
                }
            ],
        )

        result = inst_mod._direct_host_launch_or_defer(
            job,
            target_host_id=host_id,
            target_host={
                "host_id": host_id,
                "gpu_model": model,
                "status": "active",
            },
            image="alpine:latest",
            pricing_mode="on_demand",
        )
        assert job_id in run_job_calls
        row = _job_row(job_id)
        assert row["status"] == "assigned"
        assert row["host_id"] == host_id
        assert result is not None

    def test_api_submit_instance_uses_defer_helper(self):
        """Structural: host-pin branch routes through the fenced helper."""
        import inspect
        import routes.instances as inst_mod

        src = inspect.getsource(inst_mod.api_submit_instance)
        assert "_direct_host_launch_or_defer" in src
        # Must not call run_job inline in the submit path anymore.
        assert "run_job(" not in src


class TestProcessAssignedNoDoubleStart:
    def test_skips_attempt_owned_and_never_calls_run_job(
        self, fleet, reset_cp_config, monkeypatch
    ):
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        job_id = f"j-{marker}-ssh"
        _mk_host(fleet, host_id, gpu_model=model)
        _mk_job(fleet, job_id, gpu_model=model)

        cfg = _cfg_canary(marker, [host_id])
        report = SchedulerService(cfg).tick()
        assert len(report.placements) == 1
        assert _job_row(job_id)["active_attempt_id"] is not None

        import scheduler as sched

        run_job_calls: list = []

        def _fake_run_job(job, host, docker_image=None):
            run_job_calls.append(job.get("job_id"))
            return "container-should-not-start"

        monkeypatch.setattr(sched, "run_job", _fake_run_job)
        # process_assigned walks every assigned job in the shared test DB;
        # only our attempt-owned job is under test (residue must not start it).
        started = sched.process_assigned()
        started_ids = {
            (s[0]["job_id"] if isinstance(s, tuple) else s.get("job_id"))
            for s in started
        }
        assert job_id not in started_ids
        assert job_id not in run_job_calls
        # Still assigned; container authority remains the durable command.
        assert _job_row(job_id)["status"] == "assigned"
        assert _job_row(job_id)["active_attempt_id"] is not None

    def test_fail_closed_when_ownership_lookup_fails(
        self, fleet, reset_cp_config, monkeypatch
    ):
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        # Legacy-style assigned job (no active attempt) — still must not
        # start if ownership lookup fails, because we cannot prove it is
        # not fenced.
        job_id = f"j-{marker}-failclosed"
        _mk_host(fleet, host_id, gpu_model=model)
        _mk_job(fleet, job_id, gpu_model=model, status="assigned")
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET host_id=%s, status='assigned' WHERE job_id=%s",
                (host_id, job_id),
            )
            conn.execute(
                """UPDATE jobs SET payload =
                       jsonb_set(payload, '{host_id}', to_jsonb(%s::text))
                    WHERE job_id=%s""",
                (host_id, job_id),
            )
            conn.commit()

        import scheduler as sched

        run_job_calls: list = []
        monkeypatch.setattr(sched, "_attempt_owned_job_ids", lambda: None)
        monkeypatch.setattr(
            sched,
            "run_job",
            lambda *a, **k: run_job_calls.append(1) or "c",
        )
        started = sched.process_assigned()
        assert started == []
        assert run_job_calls == []
