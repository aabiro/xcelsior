"""Tests for scheduler queue-block diagnostics and renter notification pipeline.

Covers:
- _diagnose_queue_block reason codes for each failure mode
- _persist_queue_reason writing to job payload
- _notify_renter_queue_block throttle + notification creation
- process_queue_binpack emitting job_error SSE events for skipped jobs
- Edge cases: no job, empty hosts, partial admission, GPU model filtering
"""

import os
import time
import tempfile
import logging

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_diag_test_")
_tmpdir = _tmp_ctx.name
os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")

import scheduler

# Patch file paths to temp directory
scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

# Redirect file handler
for h in scheduler.log.handlers[:]:
    if isinstance(h, logging.FileHandler):
        scheduler.log.removeHandler(h)
        h.close()
_fh = logging.FileHandler(scheduler.LOG_FILE)
_fh.setLevel(logging.INFO)
scheduler.log.addHandler(_fh)


@pytest.fixture(autouse=True)
def clean_data():
    """Clean DB state and throttle caches before each test."""
    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM hosts")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM state")
    # Clear all throttle dicts
    scheduler._job_error_notified.clear()
    scheduler._renter_notified.clear()
    yield


def _admit_host(host_id):
    """Mark a host as admitted so it can receive work."""
    import json as _json
    with scheduler._atomic_mutation() as conn:
        row = conn.execute("SELECT payload FROM hosts WHERE host_id = %s", (host_id,)).fetchone()
        if row:
            data = row["payload"] if isinstance(row["payload"], dict) else _json.loads(row["payload"])
            data["admitted"] = True
            conn.execute("UPDATE hosts SET payload = %s WHERE host_id = %s", (_json.dumps(data), host_id))


# ── _diagnose_queue_block ────────────────────────────────────────────


class TestDiagnoseQueueBlock:
    """Unit tests for _diagnose_queue_block reason-code logic."""

    def test_none_job_returns_unknown(self):
        reason, detail = scheduler._diagnose_queue_block(None, [])
        assert reason == "unknown"

    def test_no_active_hosts(self):
        job = {"job_id": "j1", "vram_needed_gb": 8}
        reason, detail = scheduler._diagnose_queue_block(job, [])
        assert reason == "no_hosts_online"

    def test_no_active_hosts_all_dead(self):
        job = {"job_id": "j1", "vram_needed_gb": 8}
        hosts = [{"host_id": "h1", "status": "dead", "free_vram_gb": 24}]
        reason, _ = scheduler._diagnose_queue_block(job, hosts)
        assert reason == "no_hosts_online"

    def test_insufficient_vram(self):
        job = {"job_id": "j1", "vram_needed_gb": 48}
        hosts = [{"host_id": "h1", "status": "active", "free_vram_gb": 24, "admitted": True}]
        reason, detail = scheduler._diagnose_queue_block(job, hosts)
        assert reason == "insufficient_vram"
        assert "48" in detail

    def test_no_matching_gpu(self):
        job = {"job_id": "j1", "vram_needed_gb": 8, "gpu_model": "H100"}
        hosts = [{"host_id": "h1", "status": "active", "free_vram_gb": 24, "gpu_model": "RTX 4090", "admitted": True}]
        reason, detail = scheduler._diagnose_queue_block(job, hosts)
        assert reason == "no_matching_gpu"
        assert "H100" in detail

    def test_hosts_not_admitted(self):
        """Hosts match VRAM+GPU but aren't admitted."""
        job = {"job_id": "j1", "vram_needed_gb": 8}
        hosts = [{"host_id": "h1", "status": "active", "free_vram_gb": 24, "admitted": False}]
        reason, detail = scheduler._diagnose_queue_block(job, hosts)
        assert reason == "hosts_not_admitted"
        assert "provider has been notified" in detail.lower()

    def test_hosts_not_admitted_gpu_scoped(self):
        """Only the GPU-matching host is unadmitted; others are admitted but wrong GPU."""
        job = {"job_id": "j1", "vram_needed_gb": 8, "gpu_model": "A100"}
        hosts = [
            {"host_id": "h1", "status": "active", "free_vram_gb": 80, "gpu_model": "A100", "admitted": False},
            {"host_id": "h2", "status": "active", "free_vram_gb": 24, "gpu_model": "RTX 4090", "admitted": True},
        ]
        reason, _ = scheduler._diagnose_queue_block(job, hosts)
        assert reason == "hosts_not_admitted"

    def test_fallback_no_hosts_available(self):
        """Hosts match all criteria but allocate still fails (e.g. volume affinity)."""
        job = {"job_id": "j1", "vram_needed_gb": 8}
        hosts = [{"host_id": "h1", "status": "active", "free_vram_gb": 24, "admitted": True}]
        reason, _ = scheduler._diagnose_queue_block(job, hosts)
        assert reason == "no_hosts_available"

    def test_no_gpu_model_filter_skips_gpu_check(self):
        """Job with no gpu_model preference skips the GPU model filter."""
        job = {"job_id": "j1", "vram_needed_gb": 8}
        hosts = [{"host_id": "h1", "status": "active", "free_vram_gb": 24, "gpu_model": "A100", "admitted": True}]
        reason, _ = scheduler._diagnose_queue_block(job, hosts)
        # Should fall through to "no_hosts_available", not "no_matching_gpu"
        assert reason == "no_hosts_available"

    def test_empty_gpu_model_string_skips_gpu_check(self):
        job = {"job_id": "j1", "vram_needed_gb": 8, "gpu_model": "  "}
        hosts = [{"host_id": "h1", "status": "active", "free_vram_gb": 24, "admitted": True}]
        reason, _ = scheduler._diagnose_queue_block(job, hosts)
        assert reason != "no_matching_gpu"

    def test_zero_vram_passes_vram_check(self):
        """Interactive instances with vram_needed_gb=0 should pass VRAM check."""
        job = {"job_id": "j1", "vram_needed_gb": 0}
        hosts = [{"host_id": "h1", "status": "active", "free_vram_gb": 0, "admitted": True}]
        reason, _ = scheduler._diagnose_queue_block(job, hosts)
        assert reason != "insufficient_vram"


# ── _persist_queue_reason ─────────────────────────────────────────────


class TestPersistQueueReason:
    """Test that queue_reason is written to the job payload in the DB."""

    def test_persists_reason_on_job(self):
        job = scheduler.submit_job("test-persist", 8, priority=1)
        job_id = job["job_id"]
        scheduler._persist_queue_reason(job, "insufficient_vram", "Not enough VRAM")

        # Reload from DB
        jobs = scheduler.list_jobs()
        found = next(j for j in jobs if j["job_id"] == job_id)
        assert found["queue_reason"] == "insufficient_vram"
        assert found["queue_reason_detail"] == "Not enough VRAM"

    def test_updates_existing_reason(self):
        job = scheduler.submit_job("test-update", 8, priority=1)
        scheduler._persist_queue_reason(job, "no_hosts_online", "No hosts")
        scheduler._persist_queue_reason(job, "insufficient_vram", "Not enough VRAM")

        jobs = scheduler.list_jobs()
        found = next(j for j in jobs if j["job_id"] == job["job_id"])
        assert found["queue_reason"] == "insufficient_vram"


# ── _notify_renter_queue_block ──────────────────────────────────────


class TestNotifyRenterQueueBlock:
    """Test renter notification creation and throttle."""

    def _create_test_user(self, email="test@xcelsior.ca", user_id="test-user-1"):
        from db import UserStore
        try:
            UserStore.create_user({"email": email, "user_id": user_id, "name": "Test User", "role": "renter"})
        except Exception:
            pass  # already exists
        return user_id

    def test_creates_notification(self):
        user_id = self._create_test_user()
        job = {"job_id": "j-notify-1", "name": "test-notif", "owner": user_id}
        scheduler._renter_notified.clear()

        scheduler._notify_renter_queue_block(
            "j-notify-1", job, "no_hosts_online", "No hosts available", time.time()
        )

        from db import NotificationStore
        notifs = NotificationStore.list_for_user("test@xcelsior.ca", limit=10)
        queue_notifs = [n for n in notifs if n.get("type") == "job_queue_blocked"]
        assert len(queue_notifs) >= 1
        assert "test-notif" in queue_notifs[0]["title"]

    def test_throttle_within_15_minutes(self):
        user_id = self._create_test_user("throttle@test.ca", "throttle-user")
        job = {"job_id": "j-throttle", "name": "test", "owner": user_id}
        now = time.time()
        scheduler._renter_notified.clear()

        scheduler._notify_renter_queue_block("j-throttle", job, "r", "d", now)
        from db import NotificationStore
        count1 = len([n for n in NotificationStore.list_for_user("throttle@test.ca", limit=50) if n.get("type") == "job_queue_blocked"])

        # Second call within 15 min should be throttled
        scheduler._notify_renter_queue_block("j-throttle", job, "r", "d", now + 60)
        count2 = len([n for n in NotificationStore.list_for_user("throttle@test.ca", limit=50) if n.get("type") == "job_queue_blocked"])
        assert count2 == count1

    def test_throttle_expires_after_15_minutes(self):
        user_id = self._create_test_user("expire@test.ca", "expire-user")
        job = {"job_id": "j-expire", "name": "test", "owner": user_id}
        now = time.time()
        scheduler._renter_notified.clear()

        scheduler._notify_renter_queue_block("j-expire", job, "r", "d", now)
        from db import NotificationStore
        count1 = len([n for n in NotificationStore.list_for_user("expire@test.ca", limit=50) if n.get("type") == "job_queue_blocked"])

        # After 15+ min, should create another notification
        scheduler._notify_renter_queue_block("j-expire", job, "r", "d2", now + 901)
        count2 = len([n for n in NotificationStore.list_for_user("expire@test.ca", limit=50) if n.get("type") == "job_queue_blocked"])
        assert count2 == count1 + 1

    def test_no_notification_for_missing_owner(self):
        job = {"job_id": "j-no-owner", "name": "test"}
        scheduler._renter_notified.clear()
        # Should not raise
        scheduler._notify_renter_queue_block("j-no-owner", job, "r", "d", time.time())

    def test_no_notification_for_none_job(self):
        scheduler._renter_notified.clear()
        scheduler._notify_renter_queue_block("j-none", None, "unknown", "d", time.time())


# ── process_queue_binpack SSE emission ───────────────────────────────


class TestProcessQueueEmitsJobError:
    """Integration test: skipped jobs get job_error events with diagnostic info."""

    def test_emits_job_error_for_skipped_job(self):
        """Submit a job with no available hosts → process_queue → job_error emitted."""
        from unittest.mock import patch
        job = scheduler.submit_job("stuck-job", 8, priority=1)

        emitted = []
        original_emit = scheduler.emit_event

        def capture_emit(event_type, data):
            emitted.append({"type": event_type, "data": data})
            original_emit(event_type, data)

        with patch.object(scheduler, "emit_event", side_effect=capture_emit):
            scheduler.process_queue_binpack()

        errors = [e for e in emitted if e["type"] == "job_error"]
        assert len(errors) >= 1
        assert errors[0]["data"]["job_id"] == job["job_id"]
        assert errors[0]["data"]["error"] == "no_hosts_online"

    def test_persists_queue_reason_after_process_queue(self):
        """After process_queue, skipped jobs should have queue_reason in payload."""
        job = scheduler.submit_job("queue-reason-test", 16, priority=1)

        scheduler.process_queue_binpack()

        jobs = scheduler.list_jobs()
        found = next(j for j in jobs if j["job_id"] == job["job_id"])
        assert found.get("queue_reason") == "no_hosts_online"
        assert found.get("queue_reason_detail")

    def test_throttles_job_error_within_5_minutes(self):
        """job_error emission is throttled to once per 5 min per job."""
        from unittest.mock import patch
        scheduler.submit_job("throttle-test", 8, priority=1)

        emitted = []

        def capture_emit(event_type, data):
            if event_type == "job_error":
                emitted.append(data)

        with patch.object(scheduler, "emit_event", side_effect=capture_emit):
            scheduler.process_queue_binpack()
            count1 = len(emitted)

            # Second call within 5 min should not emit again
            scheduler.process_queue_binpack()
            count2 = len(emitted)
            assert count2 == count1

    def test_no_error_when_job_assigned(self):
        """If a job is successfully assigned, no job_error should be emitted."""
        from unittest.mock import patch
        scheduler.register_host("h-assign", "10.0.0.1", "RTX 4090", 24, 24)
        _admit_host("h-assign")
        scheduler.submit_job("assign-me", 8, priority=1)

        emitted = []

        def capture_emit(event_type, data):
            if event_type == "job_error":
                emitted.append(data)

        with patch.object(scheduler, "emit_event", side_effect=capture_emit):
            scheduler.process_queue_binpack()

        assert len(emitted) == 0
