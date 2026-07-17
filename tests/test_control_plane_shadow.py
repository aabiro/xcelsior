"""Shadow-mode scheduler (Phase 3) — decisions, explanations, comparison.

Runs the real ShadowRunner against the test database: persisted decision
rows, capacity charging across a cycle, the old-vs-new comparator's
classification matrix, retention pruning, and the mismatch-rate rollup.
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
            _c.execute("SELECT to_regclass('scheduler_shadow_decisions')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _migrated:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 058")

from control_plane.db import control_plane_transaction, run_transaction
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.shadow import (
    ShadowRunner,
    prune_old_decisions,
    simulate_cycle,
    summarize_comparisons,
)
from control_plane.scheduler.snapshot import SchedulerSnapshot, take_snapshot


@pytest.fixture
def fleet():
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute(
                "DELETE FROM scheduler_shadow_decisions WHERE job_id=%s", (jid,)
            )
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _config(**overrides) -> SchedulerConfig:
    defaults = dict(
        mode=SchedulerMode.SHADOW,
        replica_id="shadow-test",
        shadow_interval_sec=1,
        shadow_compare_grace_sec=0,
        shadow_retention_days=14,
        host_freshness_timeout_sec=300,
    )
    defaults.update(overrides)
    return SchedulerConfig(**defaults)


def _mk_host(
    fleet,
    host_id,
    *,
    gpu_model,
    gpu_count=2,
    free_vram_gb=24.0,
    total_vram_gb=24.0,
    cost_per_hour=1.0,
    admitted=True,
    last_seen=None,
    status="active",
):
    payload = {
        "host_id": host_id,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "free_vram_gb": free_vram_gb,
        "total_vram_gb": total_vram_gb,
        "cost_per_hour": cost_per_hour,
        "admitted": admitted,
        "last_seen": time.time() if last_seen is None else last_seen,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  administrative_state, availability_state)
               VALUES (%s, %s, %s, %s, %s, 'ready')""",
            (
                host_id,
                status,
                time.time(),
                json.dumps(payload),
                "admitted" if admitted else "pending",
            ),
        )
        conn.commit()
    fleet["hosts"].append(host_id)
    return payload


def _mk_job(
    fleet,
    job_id,
    *,
    gpu_model,
    priority=0,
    num_gpus=1,
    vram_needed_gb=8.0,
    status="queued",
):
    payload = {
        "job_id": job_id,
        "name": job_id,
        "gpu_model": gpu_model,
        "num_gpus": num_gpus,
        "vram_needed_gb": vram_needed_gb,
        "status": status,
        "priority": priority,
    }
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
               VALUES (%s, %s, %s, %s, %s)""",
            (job_id, status, priority, time.time(), json.dumps(payload)),
        )
        conn.commit()
    fleet["jobs"].append(job_id)
    return payload


def _decisions_for(job_ids):
    with _pool.connection() as conn:
        rows = conn.execute(
            """SELECT job_id, outcome, selected_host_id, queue_reason_code,
                      placement_score, explanation, compared_at, comparison,
                      legacy_host_id, snapshot_at
                 FROM scheduler_shadow_decisions
                WHERE job_id = ANY(%s)
                ORDER BY created_at""",
            (list(job_ids),),
        ).fetchall()
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            row = {
                "job_id": row[0], "outcome": row[1], "selected_host_id": row[2],
                "queue_reason_code": row[3], "placement_score": row[4],
                "explanation": row[5], "compared_at": row[6], "comparison": row[7],
                "legacy_host_id": row[8], "snapshot_at": row[9],
            }
        out.setdefault(str(row["job_id"]), []).append(row)
    return out


class TestSnapshot:
    def test_snapshot_projections_and_capacity(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"TESTGPU-{marker}"
        _mk_host(fleet, f"h-{marker}-a", gpu_model=model, gpu_count=2)
        # Stale host: last heartbeat far in the past.
        _mk_host(
            fleet, f"h-{marker}-stale", gpu_model=model,
            last_seen=time.time() - 3600,
        )
        _mk_job(fleet, f"j-{marker}-hi", gpu_model=model, priority=5)
        _mk_job(fleet, f"j-{marker}-lo", gpu_model=model, priority=1)
        # Active job consumes one of host-a's two GPUs.
        _mk_job(
            fleet, f"j-{marker}-run", gpu_model=model, status="running",
        )
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET host_id=%s WHERE job_id=%s",
                (f"h-{marker}-a", f"j-{marker}-run"),
            )
            conn.commit()

        with control_plane_transaction() as conn:
            snap = take_snapshot(conn, host_freshness_timeout_sec=300)

        mine = [j for j in snap.jobs if str(j.get("job_id", "")).endswith(("-hi", "-lo")) and marker in str(j.get("job_id"))]
        assert [j["job_id"] for j in mine] == [f"j-{marker}-hi", f"j-{marker}-lo"]

        hosts = {h["host_id"]: h for h in snap.hosts}
        host_a = hosts[f"h-{marker}-a"]
        assert host_a["administrative_state"] == "admitted"
        assert host_a["free_gpu_count"] == 1  # 2 GPUs - 1 running job
        assert f"h-{marker}-stale" in snap.stale_host_ids
        assert f"h-{marker}-a" not in snap.stale_host_ids


class TestSimulation:
    def test_capacity_charged_across_cycle(self):
        """Third job must not fit once two simulated placements land."""
        host = {
            "host_id": "h1", "status": "active", "administrative_state": "admitted",
            "gpu_model": "RTX 4090", "gpu_count": 2, "free_gpu_count": 2,
            "free_vram_gb": 24.0, "total_vram_gb": 24.0, "cost_per_hour": 1.0,
            "inventory_generation": 0,
        }
        jobs = [
            {"job_id": f"j{i}", "gpu_model": "RTX 4090", "num_gpus": 1,
             "vram_needed_gb": 8.0}
            for i in range(3)
        ]
        snap = SchedulerSnapshot(taken_at=None, jobs=jobs, hosts=[host])
        decisions = simulate_cycle(snap)
        assert [d.outcome for d in decisions] == ["place", "place", "queue"]
        assert decisions[0].selected_host_id == "h1"
        assert decisions[2].queue_reason_code == "no_eligible_host"
        # Non-decision still carries a full explanation.
        summary = decisions[2].explanation["rejection_summary"]
        assert summary["failed_constraints"].get("insufficient_gpus") == 1

    def test_no_hosts_reason(self):
        snap = SchedulerSnapshot(
            taken_at=None, jobs=[{"job_id": "j1"}], hosts=[]
        )
        (decision,) = simulate_cycle(snap)
        assert decision.outcome == "queue"
        assert decision.queue_reason_code == "no_hosts"
        assert decision.explanation["hosts_considered"] == 0


class TestShadowRunner:
    def test_decisions_persisted_with_explanations(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"TESTGPU-{marker}"
        _mk_host(fleet, f"h-{marker}-1", gpu_model=model, gpu_count=2)
        placeable = _mk_job(fleet, f"j-{marker}-fit", gpu_model=model, priority=99999)
        _mk_job(
            fleet, f"j-{marker}-big", gpu_model=model, priority=99998, num_gpus=8
        )
        del placeable

        report = ShadowRunner(_config()).run_once()
        assert report.jobs_considered >= 2

        rows = _decisions_for([f"j-{marker}-fit", f"j-{marker}-big"])
        (fit,) = rows[f"j-{marker}-fit"]
        assert fit["outcome"] == "place"
        assert fit["selected_host_id"] == f"h-{marker}-1"
        assert fit["placement_score"] is not None
        assert fit["explanation"]["selected_host_id"] == f"h-{marker}-1"
        assert fit["explanation"]["ranked"], "placed decision must show ranking"

        (big,) = rows[f"j-{marker}-big"]
        assert big["outcome"] == "queue"
        assert big["queue_reason_code"] == "no_eligible_host"
        constraints = big["explanation"]["rejection_summary"]["failed_constraints"]
        assert "insufficient_gpus" in constraints

    def test_comparator_classifies_outcomes(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"TESTGPU-{marker}"
        _mk_host(fleet, f"h-{marker}-1", gpu_model=model, gpu_count=1)
        _mk_host(fleet, f"h-{marker}-2", gpu_model=model, gpu_count=1)
        _mk_job(fleet, f"j-{marker}-agree", gpu_model=model, priority=99999)
        _mk_job(fleet, f"j-{marker}-differ", gpu_model=model, priority=99998)
        _mk_job(
            fleet, f"j-{marker}-stuck", gpu_model=model, priority=99997, num_gpus=8
        )
        _mk_job(fleet, f"j-{marker}-gone", gpu_model=model, priority=99996, num_gpus=8)

        runner = ShadowRunner(_config())
        runner.run_once()
        first = _decisions_for([f"j-{marker}-agree", f"j-{marker}-differ"])
        agree_host = first[f"j-{marker}-agree"][0]["selected_host_id"]
        differ_host = first[f"j-{marker}-differ"][0]["selected_host_id"]
        other_host = (
            f"h-{marker}-1" if differ_host != f"h-{marker}-1" else f"h-{marker}-2"
        )

        # Legacy "acts": agrees on one job, differs on another, leaves
        # 'stuck' queued, and 'gone' disappears entirely.
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status='assigned', host_id=%s WHERE job_id=%s",
                (agree_host, f"j-{marker}-agree"),
            )
            conn.execute(
                "UPDATE jobs SET status='assigned', host_id=%s WHERE job_id=%s",
                (other_host, f"j-{marker}-differ"),
            )
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (f"j-{marker}-gone",))
            conn.commit()

        report = runner.run_once()  # grace=0: settles cycle-1 decisions
        assert sum(report.comparisons.values()) >= 4

        rows = _decisions_for(
            [
                f"j-{marker}-agree", f"j-{marker}-differ",
                f"j-{marker}-stuck", f"j-{marker}-gone",
            ]
        )
        assert rows[f"j-{marker}-agree"][0]["comparison"] == "match_place"
        assert rows[f"j-{marker}-agree"][0]["legacy_host_id"] == agree_host
        assert rows[f"j-{marker}-differ"][0]["comparison"] == "host_mismatch"
        assert rows[f"j-{marker}-stuck"][0]["comparison"] == "match_queue"
        assert rows[f"j-{marker}-gone"][0]["comparison"] == "job_missing"

        stats = run_transaction(
            lambda conn: summarize_comparisons(conn, since_hours=1),
            what="test_summary",
        )
        assert stats["compared"] >= 4
        assert stats["by_class"].get("match_place", 0) >= 1
        assert stats["match_rate"] is not None

    def test_shadow_never_mutates_jobs_or_hosts(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"TESTGPU-{marker}"
        _mk_host(fleet, f"h-{marker}-1", gpu_model=model)
        _mk_job(fleet, f"j-{marker}-a", gpu_model=model)

        ShadowRunner(_config()).run_once()

        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status, host_id FROM jobs WHERE job_id=%s",
                (f"j-{marker}-a",),
            ).fetchone()
        status = row["status"] if isinstance(row, dict) else row[0]
        host_id = row["host_id"] if isinstance(row, dict) else row[1]
        assert status == "queued"
        assert host_id is None

    def test_grace_window_defers_comparison(self, fleet):
        marker = uuid.uuid4().hex[:8]
        model = f"TESTGPU-{marker}"
        _mk_host(fleet, f"h-{marker}-1", gpu_model=model)
        _mk_job(fleet, f"j-{marker}-a", gpu_model=model)

        runner = ShadowRunner(_config(shadow_compare_grace_sec=3600))
        runner.run_once()
        runner.run_once()
        rows = _decisions_for([f"j-{marker}-a"])
        assert all(r["compared_at"] is None for r in rows[f"j-{marker}-a"])

    def test_prune_respects_retention(self, fleet):
        marker = uuid.uuid4().hex[:8]
        job_id = f"j-{marker}-old"
        fleet["jobs"].append(job_id)
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO scheduler_shadow_decisions
                       (cycle_id, replica_id, job_id, snapshot_at,
                        policy_version, outcome, queue_reason_code,
                        explanation, created_at)
                   VALUES (gen_random_uuid(), 'shadow-test', %s,
                           clock_timestamp(), 'filters/v1', 'queue',
                           'no_hosts', '{}',
                           clock_timestamp() - interval '30 days')""",
                (job_id,),
            )
            conn.commit()
        deleted = run_transaction(
            lambda conn: prune_old_decisions(conn, retention_days=14),
            what="test_prune",
        )
        assert deleted >= 1
        assert _decisions_for([job_id]) == {}
