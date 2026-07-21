"""Host free-VRAM reconcile — sole free-capacity repair path.

Drives the shipped ``scheduler.reconcile_host_vram`` and pure helpers in
``control_plane.capacity`` against real Postgres host/job rows.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from control_plane.capacity import (
    expected_free_vram_gb,
    free_vram_correction,
    vram_used_by_host,
)

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None


@pytest.fixture
def cleanup():
    ids = {"jobs": [], "hosts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


class TestCapacityPureHelpers:
    def test_vram_used_counts_only_running_reservations(self):
        jobs = [
            {"status": "running", "host_id": "h1", "vram_reserved_gb": 8.0},
            {"status": "running", "host_id": "h1", "vram_reserved_gb": 4.0},
            {"status": "assigned", "host_id": "h1", "vram_reserved_gb": 0, "vram_needed_gb": 24},
            {"status": "running", "host_id": "h2", "vram_reserved_gb": 16.0},
        ]
        used = vram_used_by_host(jobs)
        assert used["h1"] == 12.0
        assert used["h2"] == 16.0

    def test_expected_free_clamped(self):
        assert expected_free_vram_gb(24.0, 8.0) == 16.0
        assert expected_free_vram_gb(24.0, 40.0) == 0.0
        assert expected_free_vram_gb(24.0, -1.0) == 24.0

    def test_correction_none_within_tolerance(self):
        assert free_vram_correction(16.0, 16.005) is None
        delta = free_vram_correction(10.0, 16.0)
        assert delta == 6.0


class TestReconcileHostVram:
    def _mk_host(self, cleanup, host_id, *, total=24.0, free=24.0):
        payload = {
            "host_id": host_id,
            "total_vram_gb": total,
            "free_vram_gb": free,
            "gpu_model": "RTX-TEST",
            "admitted": True,
        }
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO hosts (host_id, status, registered_at, payload)
                   VALUES (%s, 'active', %s, %s)""",
                (host_id, time.time(), json.dumps(payload)),
            )
            conn.commit()
        cleanup["hosts"].append(host_id)

    def _mk_running(self, cleanup, job_id, host_id, *, reserved: float):
        payload = {
            "job_id": job_id,
            "status": "running",
            "host_id": host_id,
            "vram_reserved_gb": reserved,
            "vram_needed_gb": reserved,
        }
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO jobs
                       (job_id, status, priority, submitted_at, host_id, payload)
                   VALUES (%s, 'running', 0, %s, %s, %s)""",
                (job_id, time.time(), host_id, json.dumps(payload)),
            )
            conn.commit()
        cleanup["jobs"].append(job_id)

    def _host_free(self, host_id: str) -> float:
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT payload FROM hosts WHERE host_id=%s", (host_id,)
            ).fetchone()
        payload = row[0] if not isinstance(row, dict) else row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return float(payload.get("free_vram_gb", 0) or 0)

    def test_corrects_drifted_free_vram(self, cleanup):
        from scheduler import reconcile_host_vram

        marker = uuid.uuid4().hex[:8]
        host_id = f"h-vram-{marker}"
        job_id = f"j-vram-{marker}"
        # Host claims 24 free but a running job reserved 8 → free should be 16.
        self._mk_host(cleanup, host_id, total=24.0, free=24.0)
        self._mk_running(cleanup, job_id, host_id, reserved=8.0)

        corrections = reconcile_host_vram()
        assert host_id in corrections
        assert abs(corrections[host_id] - (-8.0)) < 0.02 or abs(
            self._host_free(host_id) - 16.0
        ) < 0.02
        assert abs(self._host_free(host_id) - 16.0) < 0.02

    def test_noop_when_free_already_correct(self, cleanup):
        from scheduler import reconcile_host_vram

        marker = uuid.uuid4().hex[:8]
        host_id = f"h-vram-ok-{marker}"
        job_id = f"j-vram-ok-{marker}"
        self._mk_host(cleanup, host_id, total=24.0, free=16.0)
        self._mk_running(cleanup, job_id, host_id, reserved=8.0)

        corrections = reconcile_host_vram()
        # This host should not appear (or zero correction); free stays 16.
        assert host_id not in corrections or abs(corrections.get(host_id, 0)) < 0.02
        assert abs(self._host_free(host_id) - 16.0) < 0.02

    def test_reconcile_does_not_mutate_job_status(self, cleanup):
        from scheduler import reconcile_host_vram

        marker = uuid.uuid4().hex[:8]
        host_id = f"h-vram-j-{marker}"
        job_id = f"j-vram-j-{marker}"
        self._mk_host(cleanup, host_id, total=24.0, free=0.0)
        self._mk_running(cleanup, job_id, host_id, reserved=8.0)
        reconcile_host_vram()
        with _pool.connection() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        status = row[0] if not isinstance(row, dict) else row["status"]
        assert status == "running"
