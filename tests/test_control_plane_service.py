"""Phase 4 — projection triggers, inventory sync, canary partition, and
the authoritative scheduler tick (claim → filter → score → reserve).

Runs against the real test PostgreSQL. Everything is scoped by unique
markers (gpu_model / host pool) so shared-DB residue can't interfere.
"""

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
                "SELECT tgname FROM pg_trigger "
                "WHERE tgname='trg_jobs_control_plane_projection'"
            ).fetchone()
            is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 059")

from control_plane.db import run_transaction
from control_plane.inventory import sync_host_gpu_inventory
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
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mk_host(fleet, host_id, *, gpu_model, gpu_count=2, total_vram_gb=48.0,
             admitted=True, status="active"):
    payload = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "free_vram_gb": total_vram_gb,
        "total_vram_gb": total_vram_gb,
        "cost_per_hour": 1.0,
        "admitted": admitted,
        "last_seen": time.time(),
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload)
               VALUES (%s, %s, %s, %s)""",
            (host_id, status, time.time(), json.dumps(payload)),
        )
        conn.commit()
    fleet["hosts"].append(host_id)
    return payload


def _mk_job(fleet, job_id, *, gpu_model, num_gpus=1, vram_needed_gb=8.0,
            priority=0, status="queued", scheduler=None):
    payload = {
        "job_id": job_id, "name": job_id, "gpu_model": gpu_model,
        "num_gpus": num_gpus, "vram_needed_gb": vram_needed_gb,
        "priority": priority, "status": status,
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
            """SELECT status, phase, desired_state, effective_priority,
                      queued_at, host_id, active_attempt_id, reason_code,
                      next_schedule_at
                 FROM jobs WHERE job_id=%s""",
            (job_id,),
        ).fetchone()
    if row is None or isinstance(row, dict):
        return row
    return {
        "status": row[0], "phase": row[1], "desired_state": row[2],
        "effective_priority": row[3], "queued_at": row[4], "host_id": row[5],
        "active_attempt_id": row[6], "reason_code": row[7],
        "next_schedule_at": row[8],
    }


def _host_row(host_id):
    with _pool.connection() as conn:
        row = conn.execute(
            """SELECT administrative_state, availability_state,
                      inventory_generation
                 FROM hosts WHERE host_id=%s""",
            (host_id,),
        ).fetchone()
    if row is None or isinstance(row, dict):
        return row
    return {
        "administrative_state": row[0], "availability_state": row[1],
        "inventory_generation": row[2],
    }


class TestProjectionTriggers:
    def test_insert_projects_all_columns(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id = f"j-{marker}-a"
        _mk_job(fleet, job_id, gpu_model=f"m-{marker}", priority=7)
        row = _job_row(job_id)
        assert row["phase"] == "pending"
        assert row["desired_state"] == "running"
        assert row["effective_priority"] == 7
        assert row["queued_at"] is not None

    def test_raw_status_update_reprojects(self, fleet):
        """The billing/reaper-style raw UPDATE keeps the projection true."""
        marker = uuid.uuid4().hex[:8]
        job_id = f"j-{marker}-b"
        _mk_job(fleet, job_id, gpu_model=f"m-{marker}")
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status='running' WHERE job_id=%s", (job_id,)
            )
            conn.execute(
                "UPDATE jobs SET status='stopping' WHERE job_id=%s", (job_id,)
            )
            conn.commit()
        row = _job_row(job_id)
        assert row["phase"] == "running"
        assert row["desired_state"] == "stopped"

    def test_host_admission_follows_payload(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-adm"
        _mk_host(fleet, host_id, gpu_model=f"m-{marker}", admitted=False)
        assert _host_row(host_id)["administrative_state"] == "pending"
        with _pool.connection() as conn:
            conn.execute(
                """UPDATE hosts
                      SET payload = jsonb_set(payload, '{admitted}', 'true')
                    WHERE host_id=%s""",
                (host_id,),
            )
            conn.commit()
        row = _host_row(host_id)
        assert row["administrative_state"] == "admitted"
        assert row["availability_state"] == "ready"

    def test_dead_host_not_ready(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-dead"
        _mk_host(fleet, host_id, gpu_model=f"m-{marker}", status="dead")
        assert _host_row(host_id)["availability_state"] == "not_ready"


class TestInventorySync:
    def test_sync_creates_and_reconciles_devices(self, fleet):
        marker = uuid.uuid4().hex[:8]
        host_id = f"h-{marker}-inv"
        payload = _mk_host(
            fleet, host_id, gpu_model=f"RTX-{marker}", gpu_count=2,
            total_vram_gb=48.0,
        )
        gen0 = _host_row(host_id)["inventory_generation"]

        result = run_transaction(
            lambda c: sync_host_gpu_inventory(c, payload), what="test_inv"
        )
        assert result.created == 2 and result.changed
        with _pool.connection() as conn:
            rows = conn.execute(
                """SELECT gpu_uuid, model, allocatable_vram_mb FROM host_gpu_devices
                    WHERE host_id=%s AND retired_at IS NULL ORDER BY gpu_uuid""",
                (host_id,),
            ).fetchall()
        assert len(rows) == 2
        vram = rows[0][2] if not isinstance(rows[0], dict) else rows[0]["allocatable_vram_mb"]
        assert vram == 24576  # 48 GB / 2 GPUs
        assert _host_row(host_id)["inventory_generation"] == gen0 + 1

        # Idempotent: second sync is a no-op, generation stays put.
        again = run_transaction(
            lambda c: sync_host_gpu_inventory(c, payload), what="test_inv"
        )
        assert not again.changed
        assert _host_row(host_id)["inventory_generation"] == gen0 + 1

        # Shrink: retire the surplus slot, bump the generation.
        payload["gpu_count"] = 1
        shrunk = run_transaction(
            lambda c: sync_host_gpu_inventory(c, payload), what="test_inv"
        )
        assert shrunk.retired == 1
        assert _host_row(host_id)["inventory_generation"] == gen0 + 2


class TestCanaryPartition:
    def test_owns_job_modes(self):
        base = dict(mode=SchedulerMode.CANARY, replica_id="t",
                    canary_gpu_models=frozenset({"rtx 4090"}))
        cfg = SchedulerConfig(**base)
        assert cfg.owns_job({"gpu_model": "RTX 4090"})
        assert cfg.owns_job({"gpu_model": "rtx 4090"})
        assert not cfg.owns_job({"gpu_model": "RTX 3060"})
        assert not cfg.owns_job({})
        assert cfg.owns_job({"scheduler": "v2"})  # explicit opt-in

        shadow = SchedulerConfig(mode=SchedulerMode.SHADOW, replica_id="t")
        assert not shadow.owns_job({"gpu_model": "RTX 4090"})
        active = SchedulerConfig(mode=SchedulerMode.ACTIVE, replica_id="t")
        assert active.owns_job({})  # active owns everything

    def test_host_scope(self):
        cfg = SchedulerConfig(
            mode=SchedulerMode.CANARY, replica_id="t",
            canary_host_ids=frozenset({"h-1"}),
        )
        assert cfg.host_in_scope("h-1")
        assert not cfg.host_in_scope("h-2")
        open_cfg = SchedulerConfig(mode=SchedulerMode.CANARY, replica_id="t")
        assert open_cfg.host_in_scope("anything")


class TestSchedulerServiceTick:
    def _cfg(self, marker, hosts, **overrides):
        defaults = dict(
            mode=SchedulerMode.CANARY,
            replica_id=f"svc-{marker}",
            canary_gpu_models=frozenset({f"rtx-{marker}".lower()}),
            canary_host_ids=frozenset(h.lower() for h in hosts),
            tick_max_placements=5,
        )
        defaults.update(overrides)
        return SchedulerConfig(**defaults)

    def test_tick_places_canary_job_end_to_end(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        job_id = f"j-{marker}-place"
        _mk_host(fleet, host_id, gpu_model=model, gpu_count=2)
        _mk_job(fleet, job_id, gpu_model=model, num_gpus=1, vram_needed_gb=8.0)

        report = SchedulerService(self._cfg(marker, [host_id])).tick()
        assert len(report.placements) == 1
        resv = report.placements[0]
        assert resv.job_id == job_id and resv.host_id == host_id

        row = _job_row(job_id)
        assert row["status"] == "assigned"
        assert row["phase"] == "scheduled"
        assert row["host_id"] == host_id
        assert row["active_attempt_id"] is not None

        with _pool.connection() as conn:
            attempt = conn.execute(
                "SELECT status, placement_explanation FROM job_attempts "
                "WHERE attempt_id=%s",
                (resv.attempt_id,),
            ).fetchone()
            lease = conn.execute(
                "SELECT status FROM placement_leases WHERE lease_id=%s",
                (resv.lease_id,),
            ).fetchone()
            command = conn.execute(
                "SELECT status, command FROM agent_commands WHERE command_id=%s",
                (resv.command_id,),
            ).fetchone()
            allocs = conn.execute(
                "SELECT count(*) FROM gpu_device_allocations "
                "WHERE attempt_id=%s AND status='active'",
                (resv.attempt_id,),
            ).fetchone()
        assert (attempt[0] if not isinstance(attempt, dict) else attempt["status"]) == "reserved"
        expl = attempt[1] if not isinstance(attempt, dict) else attempt["placement_explanation"]
        assert expl and expl.get("selected_host_id") == host_id
        assert (lease[0] if not isinstance(lease, dict) else lease["status"]) == "offered"
        assert (command[0] if not isinstance(command, dict) else command["status"]) == "pending"
        assert (allocs[0] if not isinstance(allocs, dict) else allocs["count"]) == 1

        # The legacy SSH starter must skip this attempt-owned job.
        import scheduler as legacy_scheduler

        owned = legacy_scheduler._attempt_owned_job_ids()
        assert job_id in owned

    def test_kill_switch_stops_claims(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        host_id = f"h-{marker}-1"
        job_id = f"j-{marker}-kill"
        _mk_host(fleet, host_id, gpu_model=model)
        _mk_job(fleet, job_id, gpu_model=model)

        cfg = self._cfg(marker, [host_id], claims_enabled=False)
        report = SchedulerService(cfg).tick()
        assert report.placements == [] and report.released == []
        assert _job_row(job_id)["status"] == "queued"

    def test_unplaceable_job_released_with_durable_reason(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"RTX-{marker}"
        job_id = f"j-{marker}-stuck"
        # No hosts in scope at all.
        _mk_job(fleet, job_id, gpu_model=model, num_gpus=4)

        cfg = self._cfg(marker, [f"h-{marker}-nonexistent"])
        report = SchedulerService(cfg).tick()
        assert report.placements == []
        assert (job_id, "no_eligible_host") in report.released or (
            job_id, "no_hosts"
        ) in report.released

        row = _job_row(job_id)
        assert row["status"] == "queued"
        assert row["reason_code"] in ("no_eligible_host", "no_hosts")
        assert row["next_schedule_at"] is not None  # bounded backoff

    def test_paused_mode_is_noop(self, fleet):
        marker = uuid.uuid4().hex[:8]
        _mk_job(fleet, f"j-{marker}-idle", gpu_model=f"RTX-{marker}")
        cfg = SchedulerConfig(mode=SchedulerMode.PAUSED, replica_id="t")
        report = SchedulerService(cfg).tick()
        assert report.placements == [] and report.mode == "paused"
        assert _job_row(f"j-{marker}-idle")["status"] == "queued"
