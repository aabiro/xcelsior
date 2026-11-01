"""Placement capacity comes from physical devices, and only hosts that can start work get it.

Regression coverage for four verified defects:

1. every host was treated as single-GPU (``max(1, payload.gpu_count)``, a
   field nothing in production writes) and model/VRAM came from the GPU0
   heartbeat, so multi-GPU hosts took one job and heterogeneous hosts lied;
2. reservation bound the first free devices by UUID order whatever their model;
3. a complete inventory was ignored entirely while any GPU on the host was busy,
   so a vanished GPU stayed allocatable;
4. freshness came from the ICMP-refreshed payload ``last_seen``, synthetic
   ``slot:N`` rows were reservable although the lease claim rejects them, and
   a host whose offers kept timing out unclaimed kept receiving offers.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from tests.native_jobs import admit_fixture_job, insert_schedulable_host

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None

from control_plane.db import run_transaction
from control_plane.physical_inventory import sync_physical_inventory
from control_plane.scheduler.claim import claim_next_job
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.reservation import (
    CapacityConflict,
    HostNotEligible,
    reserve_and_bind,
)
from control_plane.scheduler.service import SchedulerService
from control_plane.scheduler.snapshot import read_fleet


@pytest.fixture
def fleet():
    ids = {"jobs": [], "hosts": []}
    yield ids
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id=%s", (jid,))
            conn.execute("DELETE FROM agent_commands WHERE job_id=%s", (jid,))
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _marker():
    return uuid.uuid4().hex[:8]


def _job(fleet, model, **fields):
    job_id = f"j-cap-{uuid.uuid4().hex[:10]}"
    payload = {"name": job_id, "gpu_model": model, **fields}
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at, effective_priority)
               VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', now(), 0)""",
            (job_id, time.time(), json.dumps(payload)),
        )
        admit_fixture_job(conn, job_id)
        conn.commit()
    fleet["jobs"].append(job_id)
    return job_id


def _host(fleet, *, models, vram_mb=24576, payload=None, observed=True, uuids=None):
    """A host whose devices have the given models, in device-index (and UUID) order."""
    host_id = f"h-cap-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        insert_schedulable_host(conn, host_id, gpus=0, payload=payload or {}, observed=observed)
        for index, model in enumerate(models):
            gpu_uuid = (uuids or {}).get(index, f"GPU-{host_id}-{index}")
            conn.execute(
                """INSERT INTO host_gpu_devices
                       (host_id, gpu_uuid, device_index, model, total_vram_mb,
                        allocatable_vram_mb, health)
                   VALUES (%s, %s, %s, %s, %s, %s, 'healthy')""",
                (host_id, gpu_uuid, index, model, vram_mb, vram_mb),
            )
        conn.commit()
    fleet["hosts"].append(host_id)
    return host_id


def _tick(model, hosts, **overrides):
    cfg = SchedulerConfig(
        mode=SchedulerMode.CANARY,
        replica_id=f"cap-{_marker()}",
        canary_gpu_models=frozenset({model.lower()}),
        canary_host_ids=frozenset(h.lower() for h in hosts),
        **overrides,
    )
    return SchedulerService(cfg).tick()


def _reserve(job_id, model, host_id):
    def place(conn):
        claimed = claim_next_job(conn, replica_id="cap-test", scope_gpu_models=[model])
        assert claimed is not None and claimed.job_id == job_id
        return reserve_and_bind(
            conn,
            job_id=job_id,
            claim_token=claimed.claim_token,
            replica_id="cap-test",
            host_id=host_id,
        )

    return run_transaction(place)


def _fleet_host(host_id):
    with _pool.connection() as conn:
        with conn.transaction():
            fleet = read_fleet(conn)
    host = next(h for h in fleet.hosts if h["host_id"] == host_id)
    return host, fleet


def _device_models(attempt_id):
    with _pool.connection() as conn:
        return [
            r[0]
            for r in conn.execute(
                """SELECT d.model FROM gpu_device_allocations a
                   JOIN host_gpu_devices d USING(gpu_device_id)
                   WHERE a.attempt_id=%s ORDER BY d.gpu_uuid""",
                (attempt_id,),
            ).fetchall()
        ]


# ── 1. capacity comes from the device rows ──────────────────────────────


def test_a_multi_gpu_host_takes_one_job_per_physical_gpu(fleet):
    model = f"RTX 4090 multi-{_marker()}"
    host_id = _host(fleet, models=[model] * 3, payload={"gpu_model": model})
    jobs = {_job(fleet, model.lower()) for _ in range(4)}

    report = _tick(model, [host_id])
    placed = {r.job_id for r in report.placements}
    assert len(placed) == 3 and placed <= jobs
    assert {r.host_id for r in report.placements} == {host_id}
    released = dict(report.released)
    assert released == {(jobs - placed).pop(): "no_eligible_host"}


def test_a_multi_gpu_job_is_placed_on_a_host_with_enough_devices(fleet):
    model = f"RTX 4090 pair-{_marker()}"
    host_id = _host(fleet, models=[model] * 2, payload={"gpu_model": model})
    job_id = _job(fleet, model.lower(), num_gpus=2)

    report = _tick(model, [host_id])
    assert [(r.job_id, len(r.gpu_device_ids)) for r in report.placements] == [(job_id, 2)]


def test_model_and_vram_come_from_devices_not_the_gpu0_heartbeat(fleet):
    marker = _marker()
    small, big = f"RTX 3060 het-{marker}", f"RTX 4090 het-{marker}"
    # The heartbeat payload describes GPU0 (the 3060) only.
    host_id = _host(
        fleet,
        models=[small, big],
        payload={"gpu_model": small, "free_vram_gb": 12, "total_vram_gb": 12},
    )
    host, _ = _fleet_host(host_id)
    assert host["gpu_count"] == 2 and host["free_gpu_count"] == 2
    job_id = _job(fleet, big.lower(), vram_needed_gb=20)

    report = _tick(big, [host_id])
    assert [r.job_id for r in report.placements] == [job_id]
    assert _device_models(report.placements[0].attempt_id) == [big]


def test_the_spot_pool_is_sized_by_physical_devices(fleet):
    model = f"RTX 4090 spot-{_marker()}"
    # No payload gpu_count at all: the old pool was "unqualified" (or 1 GPU).
    host_id = _host(
        fleet,
        models=[model] * 2,
        payload={
            "gpu_model": model,
            "spot_enabled": True,
            "spot_gpu_slots": 2,
            "spot_min_cents": 0,
        },
    )
    spot = dict(pricing_mode="spot", spot=True, preemptible=True, spot_rate_cad=0.3)
    jobs = {_job(fleet, model.lower(), **spot) for _ in range(2)}

    report = _tick(model, [host_id])
    assert {r.job_id for r in report.placements} == jobs


def test_held_and_allocated_devices_are_not_free_in_the_snapshot(fleet):
    model = f"RTX 4090 held-{_marker()}"
    host_id = _host(fleet, models=[model] * 3, payload={"gpu_model": model})
    _reserve(_job(fleet, model.lower()), model.lower(), host_id)
    host, _ = _fleet_host(host_id)
    assert (host["gpu_count"], host["free_gpu_count"]) == (3, 2)


# ── 2. reservation binds the requested model ────────────────────────────


def test_reservation_binds_only_devices_of_the_requested_model(fleet):
    marker = _marker()
    small, big = f"RTX 3060 bind-{marker}", f"RTX 4090 bind-{marker}"
    # The 3060 sorts first by UUID; the old selection took it for a 4090 job.
    host_id = _host(
        fleet,
        models=[small, big],
        uuids={0: f"GPU-0000-{marker}", 1: f"GPU-ffff-{marker}"},
    )
    job_id = _job(fleet, big.lower())
    reservation = _reserve(job_id, big.lower(), host_id)
    assert _device_models(reservation.attempt_id) == [big]


def test_reservation_refuses_a_host_without_the_requested_model(fleet):
    marker = _marker()
    host_id = _host(fleet, models=[f"RTX 3060 none-{marker}"])
    job_id = _job(fleet, f"a100 none-{marker}")
    with pytest.raises(CapacityConflict):
        _reserve(job_id, f"a100 none-{marker}", host_id)


# ── 3. complete inventories on a busy host ──────────────────────────────


def _inventory(*items):
    return [
        dict(gpu_uuid=u, device_index=i, model=m, total_vram_mb=24576)
        for i, (u, m) in enumerate(items)
    ]


def _devices(host_id):
    with _pool.connection() as conn:
        return {
            r[0]: (r[1] is not None, r[2])
            for r in conn.execute(
                "SELECT gpu_uuid,retired_at,health FROM host_gpu_devices WHERE host_id=%s",
                (host_id,),
            ).fetchall()
        }


def test_a_vanished_idle_gpu_is_retired_even_while_another_gpu_is_busy(fleet):
    model = f"RTX 4090 inv-{_marker()}"
    host_id = _host(fleet, models=[model, model])
    busy_uuid, idle_uuid = f"GPU-{host_id}-0", f"GPU-{host_id}-1"
    reservation = _reserve(_job(fleet, model.lower()), model.lower(), host_id)
    assert reservation.gpu_device_ids  # GPU 0 is now allocated

    new_uuid = f"GPU-{host_id}-new"
    with _pool.connection() as conn:
        assert sync_physical_inventory(
            conn, host_id=host_id, inventory=_inventory((busy_uuid, model), (new_uuid, model))
        )
        conn.commit()
    devices = _devices(host_id)
    assert devices[idle_uuid] == (True, "healthy")  # retired: never offered again
    assert devices[busy_uuid] == (False, "healthy")
    assert devices[new_uuid][0] is False  # a new GPU is usable immediately
    host, _ = _fleet_host(host_id)
    assert (host["gpu_count"], host["free_gpu_count"]) == (2, 1)


def test_a_vanished_busy_gpu_is_quarantined_until_its_allocation_settles(fleet):
    model = f"RTX 4090 busy-{_marker()}"
    host_id = _host(fleet, models=[model, model])
    busy_uuid, idle_uuid = f"GPU-{host_id}-0", f"GPU-{host_id}-1"
    reservation = _reserve(_job(fleet, model.lower()), model.lower(), host_id)
    with _pool.connection() as conn:
        sync_physical_inventory(conn, host_id=host_id, inventory=_inventory((idle_uuid, model)))
        conn.commit()
    devices = _devices(host_id)
    # The attempt keeps its device row; the device can never be offered again.
    assert devices[busy_uuid] == (False, "unhealthy")
    host, _ = _fleet_host(host_id)
    assert (host["gpu_count"], host["free_gpu_count"]) == (1, 1)

    with _pool.connection() as conn:
        conn.execute(
            "UPDATE gpu_device_allocations SET status='released',released_at=clock_timestamp() "
            "WHERE attempt_id=%s",
            (reservation.attempt_id,),
        )
        sync_physical_inventory(conn, host_id=host_id, inventory=_inventory((idle_uuid, model)))
        conn.commit()
    assert _devices(host_id)[busy_uuid][0] is True

    with _pool.connection() as conn:
        sync_physical_inventory(
            conn, host_id=host_id, inventory=_inventory((busy_uuid, model), (idle_uuid, model))
        )
        conn.commit()
    assert _devices(host_id)[busy_uuid] == (False, "unknown")  # it came back


# ── 4. only hosts that can start work get offers ────────────────────────


def test_icmp_liveness_without_an_observation_is_not_fresh(fleet):
    model = f"RTX 4090 ping-{_marker()}"
    host_id = _host(
        fleet,
        models=[model],
        payload={"gpu_model": model, "last_seen": time.time()},
        observed=False,
    )
    _job(fleet, model.lower())
    _, snapshot = _fleet_host(host_id)
    assert host_id in snapshot.stale_host_ids
    report = _tick(model, [host_id])
    assert report.placements == []


def test_reservation_revalidates_observation_freshness_under_the_lock(fleet):
    model = f"RTX 4090 stale-{_marker()}"
    host_id = _host(fleet, models=[model])
    job_id = _job(fleet, model.lower())
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE hosts SET last_observed_at=clock_timestamp()-interval '1 hour' "
            "WHERE host_id=%s",
            (host_id,),
        )
        conn.commit()
    with pytest.raises(HostNotEligible, match="current observation"):
        _reserve(job_id, model.lower(), host_id)


def test_synthetic_slot_rows_are_never_reservable(fleet):
    model = f"RTX 4090 slot-{_marker()}"
    host_id = _host(fleet, models=[model], uuids={0: "slot:0"})
    job_id = _job(fleet, model.lower())
    host, _ = _fleet_host(host_id)
    assert host["gpu_devices"] == []
    with pytest.raises(CapacityConflict):
        _reserve(job_id, model.lower(), host_id)


def test_a_host_that_is_not_ready_is_filtered_before_reservation(fleet):
    model = f"RTX 4090 avail-{_marker()}"
    host_id = _host(fleet, models=[model], payload={"gpu_model": model})
    job_id = _job(fleet, model.lower())
    with _pool.connection() as conn:
        conn.execute("UPDATE hosts SET availability_state='not_ready' WHERE host_id=%s", (host_id,))
        conn.commit()
    report = _tick(model, [host_id])
    assert report.released == [(job_id, "no_eligible_host")]


def test_repeated_unclaimed_offers_back_a_host_off(fleet):
    model = f"RTX 4090 backoff-{_marker()}"
    host_id = _host(fleet, models=[model], payload={"gpu_model": model})
    history = _job(fleet, f"history-{_marker()}")
    with _pool.connection() as conn:
        for number in range(1, 4):
            conn.execute(
                """INSERT INTO job_attempts
                       (job_id, attempt_number, status, host_id, fencing_token,
                        failure_code, ended_at)
                   VALUES (%s, %s, 'failed', %s, nextval('placement_fencing_token_seq'),
                           'lease_claim_timeout', clock_timestamp())""",
                (history, number, host_id),
            )
        conn.commit()
    host, _ = _fleet_host(host_id)
    assert host["lease_claim_backoff"]["consecutive_claim_timeouts"] == 3
    job_id = _job(fleet, model.lower())
    assert _tick(model, [host_id]).released == [(job_id, "no_eligible_host")]

    # A successful claim afterwards ends the back-off.
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO job_attempts
                   (job_id, attempt_number, status, host_id, fencing_token,
                    lease_claimed_at, ended_at)
               VALUES (%s, 4, 'succeeded', %s, nextval('placement_fencing_token_seq'),
                       clock_timestamp(), clock_timestamp())""",
            (history, host_id),
        )
        conn.execute("UPDATE jobs SET next_schedule_at=NULL WHERE job_id=%s", (job_id,))
        conn.commit()
    host, _ = _fleet_host(host_id)
    assert "lease_claim_backoff" not in host
    assert [r.job_id for r in _tick(model, [host_id]).placements] == [job_id]
