"""Scheduler placement onto a Slurm cluster (blueprint 2.2).

A job that names hpc_cluster_id is placed on that cluster when the switch is
on, and stays queued when it is off. It is not placed on a host either way.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _exc:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_exc}")
    _pool = None

from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.service import SchedulerService
from control_plane.state import initial_desired_state, job_phase
from tests.native_jobs import admit_fixture_job


@pytest.fixture
def fleet():
    ids = {"jobs": [], "hosts": [], "clusters": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for job_id in ids["jobs"]:
            conn.execute("DELETE FROM outbox_events WHERE aggregate_id = %s", (job_id,))
            conn.execute("DELETE FROM agent_commands WHERE job_id = %s", (job_id,))
            conn.execute(
                "DELETE FROM gpu_device_allocations WHERE attempt_id IN "
                "(SELECT attempt_id FROM job_attempts WHERE job_id = %s)",
                (job_id,),
            )
            conn.execute("DELETE FROM placement_leases WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM slurm_job_mappings WHERE xcelsior_job_id = %s", (job_id,))
            conn.execute("DELETE FROM job_attempts WHERE job_id = %s", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        for host_id in ids["hosts"]:
            conn.execute("DELETE FROM host_gpu_devices WHERE host_id = %s", (host_id,))
            conn.execute("DELETE FROM hosts WHERE host_id = %s", (host_id,))
        for cluster_id in ids["clusters"]:
            conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cluster_id,))
        conn.commit()


def _host(fleet, host_id, gpu_model):
    payload = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "gpu_count": 1,
        "free_vram_gb": 24.0,
        "total_vram_gb": 24.0,
        "cost_per_hour": 1.0,
        "admitted": True,
        "capabilities": {"egress_isolation": True},
        "last_seen": time.time(),
        "ip": "127.0.0.1",
        "status": "active",
    }
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO hosts (
                host_id, status, registered_at, payload, admission_state,
                administrative_state, availability_state, capabilities, last_observed_at
            )
            VALUES (%s, 'active', %s, %s, 'admitted', 'admitted', 'ready',
                    '{"egress_isolation":true}', clock_timestamp())
            """,
            (host_id, time.time(), json.dumps(payload)),
        )
        conn.commit()
    fleet["hosts"].append(host_id)
    from tests._db_helpers import report_test_host_inventory

    report_test_host_inventory(host_id)
    return payload


def _job(fleet, job_id, *, gpu_model, cluster_id, preferred_host_id=None):
    payload = {
        "job_id": job_id,
        "name": job_id,
        "gpu_model": gpu_model,
        "num_gpus": 1,
        "vram_needed_gb": 8.0,
        "priority": 10,
        "status": "queued",
        "submitted_at": time.time(),
        "hpc_cluster_id": cluster_id,
    }
    if preferred_host_id:
        payload["preferred_host_id"] = preferred_host_id
    status = "queued"
    with _pool.connection() as conn:
        submitted = time.time()
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, priority, submitted_at, payload, phase, desired_state,
                effective_priority, queued_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s IN ('queued','preempted') THEN to_timestamp(%s) ELSE NULL END)
            """,
            (
                job_id,
                status,
                10,
                submitted,
                json.dumps(payload),
                job_phase(status),
                initial_desired_state(status),
                10,
                status,
                submitted,
            ),
        )
        admit_fixture_job(conn, job_id)
        conn.commit()
    fleet["jobs"].append(job_id)


def _cluster(fleet, cluster_id, *, enabled=True, research_only=False):
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO slurm_clusters (
                cluster_id, profile_key, display_name, enabled, research_only,
                marketplace, agent_access, poll_interval_seconds
            )
            VALUES (%s, 'generic', 'test', %s, %s, false, false, 60)
            """,
            (cluster_id, enabled, research_only),
        )
        conn.commit()
    fleet["clusters"].append(cluster_id)


def _tick(model, *, slurm_placement):
    return SchedulerService(
        SchedulerConfig(
            mode=SchedulerMode.CANARY,
            replica_id=f"slurm-place-{model}",
            canary_gpu_models=frozenset({model.lower()}),
            tick_max_placements=2,
            slurm_placement=slurm_placement,
        )
    ).tick()


def _job_row(job_id):
    with _pool.connection() as conn:
        row = conn.execute(
            """
            SELECT status, phase, host_id, active_attempt_id, reason_code
              FROM jobs WHERE job_id = %s
            """,
            (job_id,),
        ).fetchone()
    return {
        "status": row[0],
        "phase": row[1],
        "host_id": row[2],
        "active_attempt_id": row[3],
        "reason_code": row[4],
    }


def test_an_enabled_switch_places_the_job_on_the_cluster_and_not_on_a_host(fleet):
    marker = uuid.uuid4().hex[:8]
    model = f"SLURM-{marker}"
    job_id = f"job{marker}"
    host_id = f"host{marker}"
    cluster_id = f"c{marker}"
    _host(fleet, host_id, model)
    _cluster(fleet, cluster_id)
    _job(fleet, job_id, gpu_model=model, cluster_id=cluster_id)

    report = _tick(model, slurm_placement=True)

    assert [item.job_id for item in report.slurm_placements] == [job_id]
    assert report.placements == []
    saved = _job_row(job_id)
    assert saved["status"] == "assigned"
    assert saved["phase"] == "scheduled"
    assert saved["host_id"] is None
    assert saved["active_attempt_id"] is not None
    with _pool.connection() as conn:
        submission = conn.execute(
            """
            SELECT state, cluster_id FROM slurm_submissions WHERE xcelsior_job_id = %s
            """,
            (job_id,),
        ).fetchone()
        attempt = conn.execute(
            """
            SELECT host_id, status, placement_explanation->>'target'
              FROM job_attempts WHERE job_id = %s
            """,
            (job_id,),
        ).fetchone()
        allocations = conn.execute(
            "SELECT count(*) FROM gpu_device_allocations WHERE job_id = %s",
            (job_id,),
        ).fetchone()
        commands = conn.execute(
            "SELECT count(*) FROM agent_commands WHERE job_id = %s",
            (job_id,),
        ).fetchone()
        wakes = conn.execute(
            """
            SELECT destination_class FROM outbox_events
             WHERE aggregate_id = %s AND event_type = 'job.v1.placement_reserved'
            """,
            (job_id,),
        ).fetchall()
    assert submission == ("pending", cluster_id)
    assert attempt[0] is None
    assert attempt[1] == "reserved"
    assert attempt[2] == f"slurm:{cluster_id}"
    assert allocations == (0,)
    assert commands == (0,)
    assert wakes == [("default",)]


def test_a_disabled_switch_leaves_the_job_queued_and_off_every_host(fleet):
    marker = uuid.uuid4().hex[:8]
    model = f"SLURM-{marker}"
    job_id = f"job{marker}"
    cluster_id = f"c{marker}"
    _host(fleet, f"host{marker}", model)
    _cluster(fleet, cluster_id)
    _job(fleet, job_id, gpu_model=model, cluster_id=cluster_id)

    report = _tick(model, slurm_placement=False)

    assert report.slurm_placements == []
    assert report.placements == []
    assert (job_id, "slurm_placement_disabled") in report.released
    saved = _job_row(job_id)
    assert saved["status"] == "queued"
    assert saved["phase"] == "pending"
    assert saved["host_id"] is None
    assert saved["active_attempt_id"] is None
    assert saved["reason_code"] == "slurm_placement_disabled"
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM slurm_submissions WHERE xcelsior_job_id = %s",
            (job_id,),
        ).fetchone() == (0,)


def test_a_host_pin_beside_a_cluster_is_refused(fleet):
    marker = uuid.uuid4().hex[:8]
    model = f"SLURM-{marker}"
    job_id = f"job{marker}"
    host_id = f"host{marker}"
    cluster_id = f"c{marker}"
    _host(fleet, host_id, model)
    _cluster(fleet, cluster_id)
    _job(fleet, job_id, gpu_model=model, cluster_id=cluster_id, preferred_host_id=host_id)

    report = _tick(model, slurm_placement=True)

    assert (job_id, "slurm_and_host_pin") in report.released
    assert _job_row(job_id)["host_id"] is None
    assert _job_row(job_id)["active_attempt_id"] is None


def test_a_research_only_cluster_does_not_take_the_job(fleet):
    marker = uuid.uuid4().hex[:8]
    model = f"SLURM-{marker}"
    job_id = f"job{marker}"
    cluster_id = f"c{marker}"
    _cluster(fleet, cluster_id, research_only=True)
    _job(fleet, job_id, gpu_model=model, cluster_id=cluster_id)

    report = _tick(model, slurm_placement=True)

    assert (job_id, "slurm_cluster_research_only") in report.released
    assert _job_row(job_id)["status"] == "queued"
    with _pool.connection() as conn:
        assert conn.execute(
            "SELECT count(*) FROM slurm_submissions WHERE xcelsior_job_id = %s",
            (job_id,),
        ).fetchone() == (0,)
