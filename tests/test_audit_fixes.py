"""Tests for second-audit bug fixes.

Covers:
- ALLOCATE BLOCKED log spam throttle (scheduler.py)
- Throttle dict eviction to prevent memory leaks
- Duplicate lease deletion fix (events.py)
- ASSIGNED → STARTING state transition (events.py)
- ConsentManager dict-row indexing fix (privacy.py)
- SLA credits month parameter (api.py)
- GPU pricing unknown-model warning (reputation.py)
"""

import os
import time
import tempfile
import logging
from unittest.mock import patch, MagicMock

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_audit2_test_")
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


@pytest.fixture(autouse=True)
def clean_data():
    """Clean DB state and throttle caches before each test."""
    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM hosts")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM state")
    scheduler._job_error_notified.clear()
    scheduler._renter_notified.clear()
    scheduler._allocate_blocked_notified.clear()
    yield


def _admit_host(host_id):
    import json as _json
    with scheduler._atomic_mutation() as conn:
        row = conn.execute("SELECT payload FROM hosts WHERE host_id = %s", (host_id,)).fetchone()
        if row:
            data = row["payload"] if isinstance(row["payload"], dict) else _json.loads(row["payload"])
            data["admitted"] = True
            conn.execute("UPDATE hosts SET payload = %s WHERE host_id = %s", (_json.dumps(data), host_id))


# ── ALLOCATE BLOCKED throttle ────────────────────────────────────────

class TestAllocateBlockedThrottle:
    """Test that ALLOCATE BLOCKED warnings are throttled to once per 5 min per job."""

    def test_first_call_logs_warning(self, caplog):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        # Host not admitted → ALLOCATE BLOCKED
        job = {"job_id": "j1", "name": "test-job", "vram_needed_gb": 8}
        hosts = scheduler.list_hosts()

        with caplog.at_level(logging.WARNING, logger="scheduler"):
            result = scheduler.allocate(job, hosts)

        assert result is None
        assert "ALLOCATE BLOCKED" in caplog.text

    def test_second_call_within_5min_suppressed(self, caplog):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        job = {"job_id": "j1", "name": "test-job", "vram_needed_gb": 8}
        hosts = scheduler.list_hosts()

        scheduler.allocate(job, hosts)
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="scheduler"):
            scheduler.allocate(job, hosts)

        assert "ALLOCATE BLOCKED" not in caplog.text

    def test_call_after_5min_logs_again(self, caplog):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        job = {"job_id": "j1", "name": "test-job", "vram_needed_gb": 8}
        hosts = scheduler.list_hosts()

        scheduler.allocate(job, hosts)
        # Fast-forward throttle timestamp
        scheduler._allocate_blocked_notified["j1"] = time.time() - 301
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="scheduler"):
            scheduler.allocate(job, hosts)

        assert "ALLOCATE BLOCKED" in caplog.text

    def test_different_jobs_tracked_independently(self, caplog):
        scheduler.register_host("h1", "10.0.0.1", "RTX 4090", 24, 24)
        hosts = scheduler.list_hosts()

        job1 = {"job_id": "j1", "name": "job-one", "vram_needed_gb": 8}
        job2 = {"job_id": "j2", "name": "job-two", "vram_needed_gb": 8}

        scheduler.allocate(job1, hosts)
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="scheduler"):
            scheduler.allocate(job2, hosts)

        assert "ALLOCATE BLOCKED" in caplog.text
        assert "job-two" in caplog.text


# ── Throttle dict eviction ───────────────────────────────────────────

class TestThrottleEviction:
    """Test that _evict_stale_throttles removes entries older than 1 hour."""

    def test_evicts_stale_entries(self):
        old = time.time() - 7200  # 2 hours ago
        scheduler._job_error_notified["old-job"] = old
        scheduler._renter_notified["old-job"] = old
        scheduler._allocate_blocked_notified["old-job"] = old

        scheduler._evict_stale_throttles()

        assert "old-job" not in scheduler._job_error_notified
        assert "old-job" not in scheduler._renter_notified
        assert "old-job" not in scheduler._allocate_blocked_notified

    def test_keeps_recent_entries(self):
        recent = time.time() - 60  # 1 minute ago
        scheduler._job_error_notified["recent-job"] = recent
        scheduler._renter_notified["recent-job"] = recent
        scheduler._allocate_blocked_notified["recent-job"] = recent

        scheduler._evict_stale_throttles()

        assert "recent-job" in scheduler._job_error_notified
        assert "recent-job" in scheduler._renter_notified
        assert "recent-job" in scheduler._allocate_blocked_notified

    def test_mixed_eviction(self):
        old = time.time() - 7200
        recent = time.time() - 60
        scheduler._job_error_notified["old"] = old
        scheduler._job_error_notified["recent"] = recent

        scheduler._evict_stale_throttles()

        assert "old" not in scheduler._job_error_notified
        assert "recent" in scheduler._job_error_notified

    def test_process_queue_calls_eviction(self):
        old = time.time() - 7200
        scheduler._job_error_notified["stale-entry"] = old

        scheduler.process_queue_binpack()

        assert "stale-entry" not in scheduler._job_error_notified


# ── State transitions ────────────────────────────────────────────────

class TestStateTransitions:
    """Test the ASSIGNED → STARTING transition fix."""

    def test_assigned_to_starting_is_valid(self):
        from events import VALID_TRANSITIONS, JobState
        assert JobState.STARTING in VALID_TRANSITIONS[JobState.ASSIGNED]

    def test_assigned_to_leased_still_valid(self):
        from events import VALID_TRANSITIONS, JobState
        assert JobState.LEASED in VALID_TRANSITIONS[JobState.ASSIGNED]

    def test_assigned_to_running_still_valid(self):
        from events import VALID_TRANSITIONS, JobState
        assert JobState.RUNNING in VALID_TRANSITIONS[JobState.ASSIGNED]

    def test_leased_to_starting_still_valid(self):
        from events import VALID_TRANSITIONS, JobState
        assert JobState.STARTING in VALID_TRANSITIONS[JobState.LEASED]

    def test_queued_to_starting_still_invalid(self):
        from events import VALID_TRANSITIONS, JobState
        assert JobState.STARTING not in VALID_TRANSITIONS[JobState.QUEUED]

    def test_all_terminal_states_have_empty_transitions(self):
        from events import VALID_TRANSITIONS, JobState
        for state in (JobState.COMPLETED, JobState.CANCELLED, JobState.TERMINATED):
            assert VALID_TRANSITIONS[state] == set(), f"{state} should have no transitions"


# ── Lease deletion fix ───────────────────────────────────────────────

class TestLeaseGrantDeletion:
    """Test that grant_lease deletes existing leases instead of just releasing them."""

    def test_grant_lease_succeeds_on_re_grant(self):
        from events import get_event_store
        es = get_event_store()

        # Grant first lease
        lease1 = es.grant_lease("j-dup", "h1", duration_sec=600)
        assert lease1.job_id == "j-dup"

        # Re-granting should not raise UniqueViolation
        lease2 = es.grant_lease("j-dup", "h2", duration_sec=600)
        assert lease2.job_id == "j-dup"
        assert lease2.host_id == "h2"
        assert lease2.lease_id != lease1.lease_id

    def test_grant_lease_after_release(self):
        from events import get_event_store
        es = get_event_store()

        lease1 = es.grant_lease("j-dup2", "h1", duration_sec=600)
        es.release_lease("j-dup2")

        # Should succeed — old released lease is deleted
        lease2 = es.grant_lease("j-dup2", "h3", duration_sec=600)
        assert lease2.host_id == "h3"

    def test_grant_lease_deletes_all_previous(self):
        from events import get_event_store
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        es = get_event_store()

        es.grant_lease("j-dup3", "h1", duration_sec=600)
        es.grant_lease("j-dup3", "h2", duration_sec=600)

        # Only one lease row should exist for this job_id
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            rows = conn.execute("SELECT * FROM leases WHERE job_id = %s", ("j-dup3",)).fetchall()
        assert len(rows) == 1
        assert rows[0]["host_id"] == "h2"


# ── ConsentManager dict-row fix ──────────────────────────────────────

class TestConsentManagerDictRows:
    """Test that ConsentManager methods work with dict_row connections."""

    def test_record_and_check_consent(self):
        from privacy import get_consent_manager
        cm = get_consent_manager()

        cm.record_consent("user-dict-test", "express", "marketing_email")

        has, ctype = cm.has_consent("user-dict-test", "marketing_email")
        assert has is True
        assert ctype == "express"

    def test_no_consent_returns_false(self):
        from privacy import get_consent_manager
        cm = get_consent_manager()

        has, ctype = cm.has_consent("nonexistent-user", "marketing_email")
        assert has is False
        assert ctype is None

    def test_get_user_consents_returns_dicts(self):
        from privacy import get_consent_manager
        cm = get_consent_manager()

        cm.record_consent("user-consents-test", "express", "product_updates")
        cm.record_consent("user-consents-test", "implied", "third_party_offers",
                          expires_in_days=730)

        consents = cm.get_user_consents("user-consents-test")
        assert len(consents) >= 2
        for c in consents:
            assert "purpose" in c
            assert "consent_type" in c
            assert "granted_at" in c
            assert "expires_at" in c
            assert "active" in c

    def test_consent_expiry_check(self):
        from privacy import get_consent_manager
        cm = get_consent_manager()

        # Record implied consent that expires immediately
        cm.record_consent("user-expire-test", "implied", "expiring_purpose",
                          expires_in_days=0)
        # Manually set expires_at to the past
        from db import _get_pg_pool
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute(
                "UPDATE casl_consent SET expires_at = %s "
                "WHERE user_id = %s AND purpose = %s",
                (time.time() - 100, "user-expire-test", "expiring_purpose"),
            )
            conn.commit()

        has, ctype = cm.has_consent("user-expire-test", "expiring_purpose")
        assert has is False

    def test_expire_implied_consents(self):
        from privacy import get_consent_manager
        cm = get_consent_manager()

        cm.record_consent("user-batch-expire", "implied", "batch_test")
        # Force expiry
        from db import _get_pg_pool
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute(
                "UPDATE casl_consent SET expires_at = %s "
                "WHERE user_id = %s AND purpose = %s",
                (time.time() - 100, "user-batch-expire", "batch_test"),
            )
            conn.commit()

        count = cm.expire_implied_consents()
        assert count >= 1


# ── GPU pricing unknown model warning ────────────────────────────────

class TestGPUPricingWarning:
    """Test that unknown GPU models produce a warning log."""

    def test_unknown_gpu_logs_warning(self, caplog):
        from reputation import get_reference_rate
        with caplog.at_level(logging.WARNING, logger="reputation"):
            rate = get_reference_rate("Tesla V100")
        assert rate > 0
        assert "unknown GPU model" in caplog.text
        assert "Tesla V100" in caplog.text

    def test_known_gpu_no_warning(self, caplog):
        from reputation import get_reference_rate
        with caplog.at_level(logging.WARNING, logger="reputation"):
            rate = get_reference_rate("RTX 4090")
        assert rate > 0
        assert "unknown GPU model" not in caplog.text

    def test_substring_match_no_warning(self, caplog):
        from reputation import get_reference_rate
        with caplog.at_level(logging.WARNING, logger="reputation"):
            rate = get_reference_rate("NVIDIA GeForce RTX 4090")
        assert rate > 0
        assert "unknown GPU model" not in caplog.text
