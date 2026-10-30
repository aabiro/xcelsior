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
from datetime import datetime, timedelta, timezone
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
    from tests._db_helpers import admit_test_host

    admit_test_host(host_id)


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

    def test_throttle_pruning_does_not_require_a_queue_walker(self):
        old = time.time() - 7200
        scheduler._job_error_notified["stale-entry"] = old

        scheduler._evict_stale_throttles()

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
        cm.record_consent(
            "user-consents-test", "implied", "third_party_offers", expires_in_days=730
        )

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
        cm.record_consent("user-expire-test", "implied", "expiring_purpose", expires_in_days=0)
        # Manually set expires_at to the past
        from db import _get_pg_pool

        pool = _get_pg_pool()
        expired_at = datetime.now(timezone.utc) - timedelta(seconds=100)
        with pool.connection() as conn:
            updated = conn.execute(
                "UPDATE casl_consent SET expires_at = %s " "WHERE user_id = %s AND purpose = %s",
                (expired_at, "user-expire-test", "expiring_purpose"),
            ).rowcount
            conn.commit()
        # Premise probe: the direct UPDATE must have hit the recorded row and
        # be visible on the manager's own connection path before the expiry
        # check means anything (flaked once in a full-suite run — if this
        # trips, the bug is in write visibility, not the expiry logic).
        assert updated == 1, f"expected 1 casl_consent row updated, got {updated}"
        with cm._conn() as conn:
            row = conn.execute(
                "SELECT expires_at, active FROM casl_consent WHERE user_id = %s AND purpose = %s",
                ("user-expire-test", "expiring_purpose"),
            ).fetchone()
        assert row is not None
        assert abs((row["expires_at"] - expired_at).total_seconds()) < 1, row

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
                "UPDATE casl_consent SET expires_at = %s " "WHERE user_id = %s AND purpose = %s",
                (
                    datetime.now(timezone.utc) - timedelta(seconds=100),
                    "user-batch-expire",
                    "batch_test",
                ),
            )
            conn.commit()

        count = cm.expire_implied_consents()
        assert count >= 1


# ── GPU pricing unknown model warning ────────────────────────────────


class TestGPUPricingUnknownModel:
    """An unknown GPU has no rate; it was once priced as an RTX 4090 with a warning."""

    def test_unknown_gpu_has_no_rate(self):
        from pricing_reference import get_rate

        assert get_rate("Tesla K80") == 0.0

    def test_known_gpu_is_priced(self):
        from pricing_reference import get_rate

        assert get_rate("RTX 4090") > 0

    def test_driver_name_is_priced_as_its_model(self):
        from pricing_reference import get_rate

        assert get_rate("NVIDIA GeForce RTX 4090") == get_rate("RTX 4090") > 0
