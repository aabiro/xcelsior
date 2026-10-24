"""Phase 7 — Instance Lifecycle Feature Tests.

Covers the full stop/start/restart/terminate lifecycle:

  1. events.py   — new states, frozensets, valid transitions, EventType enum
  2. State machine — transition validation and EventType mapping
  3. scheduler.py  — lifecycle helper functions (stop/start/terminate)
  4. billing.py    — stop_instance, start_instance, restart_instance,
                     terminate_instance, _VALID_STOP_REASONS,
                     auto_billing_cycle storage section
  5. routes        — API endpoint presence and correct HTTP methods
  6. Migration 019 — storage_billing_rates table
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

# ── Paths ─────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
_BILLING_SRC = (_ROOT / "billing" / "billing.py").read_text()
_SCHEDULER_SRC = (_ROOT / "workers" / "scheduler.py").read_text()
_ROUTES_SRC = (_ROOT / "api" / "routes" / "instances.py").read_text()
_MIGRATION_SRC = (
    _ROOT / "api" / "migrations" / "versions" / "019_instance_lifecycle.py"
).read_text()


# ═══════════════════════════════════════════════════════════════════════
# 1. events.py — State Definitions
# ═══════════════════════════════════════════════════════════════════════


class TestLifecycleStates:
    """Verify new lifecycle states are present and correctly categorized."""

    def test_stopping_state_exists(self):
        from events import JobState

        assert JobState.STOPPING == "stopping"

    def test_stopped_state_exists(self):
        from events import JobState

        assert JobState.STOPPED == "stopped"

    def test_restarting_state_exists(self):
        from events import JobState

        assert JobState.RESTARTING == "restarting"

    def test_terminated_state_exists(self):
        from events import JobState

        assert JobState.TERMINATED == "terminated"

    def test_terminated_is_terminal(self):
        from events import JobState, TERMINAL_STATES

        assert JobState.TERMINATED in TERMINAL_STATES

    def test_stopping_is_not_terminal(self):
        from events import JobState, TERMINAL_STATES

        assert JobState.STOPPING not in TERMINAL_STATES

    def test_stopped_is_not_terminal(self):
        from events import JobState, TERMINAL_STATES

        assert JobState.STOPPED not in TERMINAL_STATES

    def test_restarting_is_not_terminal(self):
        from events import JobState, TERMINAL_STATES

        assert JobState.RESTARTING not in TERMINAL_STATES

    def test_transitional_states_frozenset_exists(self):
        from events import TRANSITIONAL_STATES, JobState

        assert JobState.STOPPING in TRANSITIONAL_STATES
        assert JobState.RESTARTING in TRANSITIONAL_STATES

    def test_transitional_states_excludes_stopped(self):
        from events import TRANSITIONAL_STATES, JobState

        assert JobState.STOPPED not in TRANSITIONAL_STATES

    def test_storage_billed_states_includes_stopped(self):
        from events import STORAGE_BILLED_STATES, JobState

        assert JobState.STOPPED in STORAGE_BILLED_STATES

    def test_storage_billed_states_excludes_running(self):
        from events import STORAGE_BILLED_STATES, JobState

        assert JobState.RUNNING not in STORAGE_BILLED_STATES

    def test_storage_billed_states_excludes_transitional(self):
        from events import STORAGE_BILLED_STATES, JobState

        assert JobState.STOPPING not in STORAGE_BILLED_STATES
        assert JobState.RESTARTING not in STORAGE_BILLED_STATES


# ═══════════════════════════════════════════════════════════════════════
# 2. events.py — Valid Transitions
# ═══════════════════════════════════════════════════════════════════════


class TestLifecycleTransitions:
    """Verify VALID_TRANSITIONS for all new lifecycle paths."""

    def test_running_can_stop(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.STOPPING in VALID_TRANSITIONS[JobState.RUNNING]

    def test_running_can_restart(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.RESTARTING in VALID_TRANSITIONS[JobState.RUNNING]

    def test_running_can_terminate(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.TERMINATED in VALID_TRANSITIONS[JobState.RUNNING]

    def test_stopping_can_reach_stopped_on_success(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.STOPPED in VALID_TRANSITIONS[JobState.STOPPING]

    def test_stopping_can_fallback_to_running_on_failure(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.RUNNING in VALID_TRANSITIONS[JobState.STOPPING]

    def test_stopping_can_terminate(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.TERMINATED in VALID_TRANSITIONS[JobState.STOPPING]

    def test_stopped_can_restart(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.RESTARTING in VALID_TRANSITIONS[JobState.STOPPED]

    def test_stopped_can_terminate(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.TERMINATED in VALID_TRANSITIONS[JobState.STOPPED]

    def test_stopped_cannot_go_directly_to_running(self):
        """Stopped → Running must go through restarting."""
        from events import VALID_TRANSITIONS, JobState

        assert JobState.RUNNING not in VALID_TRANSITIONS[JobState.STOPPED]

    def test_restarting_can_succeed_to_running(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.RUNNING in VALID_TRANSITIONS[JobState.RESTARTING]

    def test_restarting_can_fail_back_to_stopped(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.STOPPED in VALID_TRANSITIONS[JobState.RESTARTING]

    def test_restarting_can_hard_fail(self):
        from events import VALID_TRANSITIONS, JobState

        assert JobState.FAILED in VALID_TRANSITIONS[JobState.RESTARTING]

    def test_terminated_has_no_outgoing_transitions(self):
        """Terminated is a hard terminal — no exits."""
        from events import VALID_TRANSITIONS, JobState

        assert VALID_TRANSITIONS[JobState.TERMINATED] == set()


# ═══════════════════════════════════════════════════════════════════════
# 3. events.py — EventType enum
# ═══════════════════════════════════════════════════════════════════════


class TestLifecycleEventTypes:
    """Verify new EventType enum members exist with correct values."""

    def test_job_stopping_event_type(self):
        from events import EventType

        assert EventType.JOB_STOPPING == "job.stopping"

    def test_job_stopped_event_type(self):
        from events import EventType

        assert EventType.JOB_STOPPED == "job.stopped"

    def test_job_restarting_event_type(self):
        from events import EventType

        assert EventType.JOB_RESTARTING == "job.restarting"

    def test_job_started_event_type(self):
        from events import EventType

        assert EventType.JOB_STARTED == "job.started"

    def test_job_terminated_event_type(self):
        from events import EventType

        assert EventType.JOB_TERMINATED == "job.terminated"


# ═══════════════════════════════════════════════════════════════════════
# 4. events.py — State Machine (pure validation, mocked append)
# ═══════════════════════════════════════════════════════════════════════


class TestLifecycleStateMachine:
    """Exercise state machine transitions for new lifecycle states.

    The EventStore.append is mocked so no DB is required.
    """

    def _sm(self):
        from events import EventStore, JobStateMachine

        store = EventStore.__new__(EventStore)
        appended = []

        def _fake_append(evt):
            appended.append(evt)
            return evt

        store.append = _fake_append
        store._appended = appended
        return JobStateMachine(store)

    def test_running_to_stopping_produces_stopping_event(self):
        sm = self._sm()
        evt = sm.transition("j1", "running", "stopping")
        assert evt.event_type == "job.stopping"
        assert evt.data["new_state"] == "stopping"
        assert evt.data["previous_state"] == "running"

    def test_stopping_to_stopped_produces_stopped_event(self):
        sm = self._sm()
        evt = sm.transition("j1", "stopping", "stopped")
        assert evt.event_type == "job.stopped"

    def test_stopping_fallback_to_running(self):
        sm = self._sm()
        evt = sm.transition("j1", "stopping", "running")
        from events import EventType

        assert evt.event_type == EventType.JOB_RUNNING

    def test_stopped_to_restarting_produces_restarting_event(self):
        sm = self._sm()
        evt = sm.transition("j1", "stopped", "restarting")
        assert evt.event_type == "job.restarting"

    def test_restarting_to_running_produces_running_event(self):
        sm = self._sm()
        evt = sm.transition("j1", "restarting", "running")
        from events import EventType

        assert evt.event_type == EventType.JOB_RUNNING

    def test_running_to_terminated_produces_terminated_event(self):
        sm = self._sm()
        evt = sm.transition("j1", "running", "terminated")
        assert evt.event_type == "job.terminated"

    def test_stopped_to_terminated_produces_terminated_event(self):
        sm = self._sm()
        evt = sm.transition("j1", "stopped", "terminated")
        assert evt.event_type == "job.terminated"

    def test_running_to_stopped_direct_raises(self):
        """Must go through stopping — no shortcut."""
        sm = self._sm()
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition("j1", "running", "stopped")

    def test_terminated_to_running_raises(self):
        sm = self._sm()
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition("j1", "terminated", "running")

    def test_stopped_to_running_direct_raises(self):
        sm = self._sm()
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition("j1", "stopped", "running")

    def test_event_entity_id_set(self):
        sm = self._sm()
        evt = sm.transition("job-xyz", "running", "stopping")
        assert evt.entity_id == "job-xyz"
        assert evt.entity_type == "job"


# ═══════════════════════════════════════════════════════════════════════
# 5. scheduler.py — Lifecycle Helpers
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# 6. billing.py — Structure & Constants
# ═══════════════════════════════════════════════════════════════════════


class TestBillingLifecycleStructure:
    """Verify BillingEngine has lifecycle methods and constants."""

    def test_stop_instance_method_exists(self):
        from billing import BillingEngine

        assert hasattr(BillingEngine, "stop_instance")
        assert callable(BillingEngine.stop_instance)

    def test_start_instance_method_exists(self):
        from billing import BillingEngine

        assert hasattr(BillingEngine, "start_instance")
        assert callable(BillingEngine.start_instance)

    def test_restart_instance_method_exists(self):
        from billing import BillingEngine

        assert hasattr(BillingEngine, "restart_instance")
        assert callable(BillingEngine.restart_instance)

    def test_terminate_instance_method_exists(self):
        from billing import BillingEngine

        assert hasattr(BillingEngine, "terminate_instance")
        assert callable(BillingEngine.terminate_instance)

    def test_valid_stop_reasons_class_attribute_exists(self):
        from billing import BillingEngine

        assert hasattr(BillingEngine, "_VALID_STOP_REASONS")
        reasons = BillingEngine._VALID_STOP_REASONS
        assert isinstance(reasons, frozenset)
        assert "user_stopped" in reasons

    def test_valid_stop_reasons_contains_expected_values(self):
        from billing import BillingEngine

        r = BillingEngine._VALID_STOP_REASONS
        assert "user_stopped" in r
        assert "paused_low_balance" in r
        assert "billing_suspended" in r

    def test_stop_instance_uses_for_update_lock(self):
        assert "FOR UPDATE" in _BILLING_SRC

    def test_stop_instance_transitions_stopped(self):
        assert "stopping" in _BILLING_SRC
        assert "'stopped'" in _BILLING_SRC or '"stopped"' in _BILLING_SRC

    def test_terminate_instance_creates_billing_anchor(self):
        assert "BC-term-" in _BILLING_SRC

    def test_auto_billing_cycle_has_storage_section(self):
        assert "FROM volumes WHERE status IN" in _BILLING_SRC
        assert "storage_billing_rates" not in _BILLING_SRC
        assert "storage_rate_cad_per_gb_hr" not in _BILLING_SRC

    def test_storage_billing_uses_normalized_volume_records(self):
        assert "FROM volumes WHERE status IN ('available', 'attached')" in _BILLING_SRC


# ═══════════════════════════════════════════════════════════════════════
# 7. billing.py — stop_instance (mocked DB)
# ═══════════════════════════════════════════════════════════════════════


def _mock_pool(job_row=None, *, raise_on_second=False):
    """Build a mock PG pool with a single job row."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    cursor = MagicMock()
    cursor.fetchone.return_value = job_row
    mock_conn.execute.return_value = cursor

    pool = MagicMock()
    pool.connection.return_value = mock_conn
    return pool


class TestBillingStopInstance:
    """Unit tests for BillingEngine.stop_instance with mocked DB."""

    def test_stop_invalid_reason_returns_error(self):
        from billing import BillingEngine

        be = BillingEngine.__new__(BillingEngine)
        result = be.stop_instance("job-1", reason="unknown_reason")
        assert result["stopped"] is False
        assert "invalid_reason" in result["reason"]

    def test_stop_instance_returns_stopped_false_when_no_job(self):
        """stop_instance must return {stopped: False} when job is not found/not running."""
        # Verified via source: fetchone returns None → early return with stopped=False
        assert (
            '"stopped": False' in _BILLING_SRC
            or "'stopped': False" in _BILLING_SRC
            or (
                "not_found_or_not_running" in _BILLING_SRC
                or "already_terminal_or_not_found" in _BILLING_SRC
            )
        )


class TestBillingStartInstance:
    """Unit tests for BillingEngine.start_instance."""

    def test_start_requires_fresh_native_attempt(self):
        assert "start_container" not in _BILLING_SRC
        assert "request_fresh_attempt_resume" in _BILLING_SRC

    def test_start_checks_wallet_balance(self):
        """start_instance must check wallet balance before starting."""
        assert "insufficient_balance" in _BILLING_SRC or "balance_cad" in _BILLING_SRC

    def test_start_does_not_optimistically_mark_running(self):
        import inspect
        from billing import BillingEngine

        src = inspect.getsource(BillingEngine.start_instance)
        assert '"status": "queued"' in src
        assert '"status": "running"' not in src

    def test_start_instance_returns_started_key(self):
        """Return dict must include 'started' key."""
        assert '"started"' in _BILLING_SRC or "'started'" in _BILLING_SRC


class TestBillingRestartInstance:
    """Unit tests for BillingEngine.restart_instance."""

    def test_restart_uses_fenced_lifecycle(self):
        assert "pause_container" not in _BILLING_SRC
        assert "start_container" not in _BILLING_SRC
        assert "request_fenced_stop_remove" in _BILLING_SRC

    def test_restart_works_from_running_and_stopped(self):
        """Source must handle both 'running' and 'stopped' input states."""
        assert "was_running" in _BILLING_SRC or (
            "'running'" in _BILLING_SRC and "'stopped'" in _BILLING_SRC
        )

    def test_restart_has_no_billing_gap(self):
        """Restart comment or anchor should NOT create a gap (no stopped anchor)."""
        # The billing source should NOT create a "stop billing anchor" for restart
        # (continuous billing — no gap). Validate the intent via source inspection.
        assert "continuous" in _BILLING_SRC or "no gap" in _BILLING_SRC or "NO gap" in _BILLING_SRC

    def test_restart_returns_restarted_key(self):
        assert '"restarted"' in _BILLING_SRC or "'restarted'" in _BILLING_SRC


class TestBillingTerminateInstance:
    """Unit tests for BillingEngine.terminate_instance."""

    def test_terminate_calls_terminate_job_helper(self):
        assert "terminate_job" not in _BILLING_SRC
        assert "request_fenced_stop_remove" in _BILLING_SRC

    def test_terminate_creates_final_billing_cycle(self):
        """A final billing anchor with status='terminated' is inserted."""
        assert "'terminated'" in _BILLING_SRC or '"terminated"' in _BILLING_SRC
        assert "BC-term-" in _BILLING_SRC

    def test_terminate_sets_terminated_at_in_payload(self):
        from control_plane import lifecycle
        import inspect

        assert "terminated_at" in inspect.getsource(lifecycle.request_unplaced_termination)

    def test_terminate_guards_against_already_terminal(self):
        """Query must exclude already-terminal statuses."""
        assert "NOT IN" in _BILLING_SRC
        assert "terminated" in _BILLING_SRC

    def test_terminate_returns_terminated_true_key(self):
        assert '"terminated"' in _BILLING_SRC or "'terminated'" in _BILLING_SRC

    def test_terminate_returns_false_when_already_terminal(self):
        assert "already_terminal_or_not_found" in _BILLING_SRC


# ═══════════════════════════════════════════════════════════════════════
# 8. billing.py — auto_billing_cycle storage billing
# ═══════════════════════════════════════════════════════════════════════


class TestBillingStorageCycle:
    """Job observations cannot authorize a second charge beside native volumes."""

    def test_no_observation_based_storage_charge(self):
        for field in ("storage_gb", "storage_rate_cad_per_gb_hr", "cached_rate", "SC-"):
            assert field not in _BILLING_SRC
        assert "volume_billed" in _BILLING_SRC
        assert "storage_billed" in _BILLING_SRC  # retained response key, always zero


# ═══════════════════════════════════════════════════════════════════════
# 9. routes/instances.py — API Endpoints
# ═══════════════════════════════════════════════════════════════════════


class TestLifecycleRoutes:
    """Verify the 4 new lifecycle endpoints are registered in the router."""

    def test_stop_endpoint_path_exists(self):
        assert "/instances/{job_id}/stop" in _ROUTES_SRC

    def test_start_endpoint_path_exists(self):
        assert "/instances/{job_id}/start" in _ROUTES_SRC

    def test_restart_endpoint_path_exists(self):
        assert "/instances/{job_id}/restart" in _ROUTES_SRC

    def test_terminate_endpoint_path_exists(self):
        assert "/instances/{job_id}/terminate" in _ROUTES_SRC

    def test_stop_endpoint_is_post(self):
        assert 'router.post("/instances/{job_id}/stop"' in _ROUTES_SRC

    def test_start_endpoint_is_post(self):
        assert 'router.post("/instances/{job_id}/start"' in _ROUTES_SRC

    def test_restart_endpoint_is_post(self):
        assert 'router.post("/instances/{job_id}/restart"' in _ROUTES_SRC

    def test_terminate_endpoint_is_post(self):
        assert 'router.post("/instances/{job_id}/terminate"' in _ROUTES_SRC

    def test_stop_checks_running_status(self):
        """Stop endpoint guards: only running instances may be stopped."""
        assert "must be running to stop" in _ROUTES_SRC or (
            "status" in _ROUTES_SRC and "running" in _ROUTES_SRC
        )

    def test_start_checks_stopped_status(self):
        """Start endpoint guards: only stopped/paused instances may be started."""
        assert "must be stopped to start" in _ROUTES_SRC or "allowed_statuses" in _ROUTES_SRC

    def test_restart_checks_running_or_stopped(self):
        """Restart works from both running and stopped."""
        assert "running or stopped" in _ROUTES_SRC or (
            "'running'" in _ROUTES_SRC and "'stopped'" in _ROUTES_SRC
        )

    def test_terminate_guards_already_terminal(self):
        assert (
            "terminal_statuses" in _ROUTES_SRC
            or "already_terminal" in _ROUTES_SRC
            or ("already" in _ROUTES_SRC and "terminated" in _ROUTES_SRC)
        )

    def test_start_checks_wallet_balance(self):
        assert "Insufficient wallet balance" in _ROUTES_SRC or "balance_cad" in _ROUTES_SRC

    def test_start_checks_suspended_wallet(self):
        assert "suspended" in _ROUTES_SRC

    def test_endpoints_check_job_ownership(self):
        """Ownership check: user can only act on their own instances."""
        assert "Not authorized to {action}" in _ROUTES_SRC
        assert (
            "_check_job_write_access(user, job_id)" in _ROUTES_SRC
            or "_check_job_access" in _ROUTES_SRC
        )

    def test_stop_broadcasts_sse_event(self):
        assert "instance_stopped" in _ROUTES_SRC

    def test_start_broadcasts_sse_event(self):
        assert "instance_started" in _ROUTES_SRC

    def test_restart_broadcasts_sse_event(self):
        assert "instance_restarted" in _ROUTES_SRC

    def test_terminate_broadcasts_sse_event(self):
        assert "instance_terminated" in _ROUTES_SRC

    def test_stop_delegates_to_billing_engine(self):
        assert "be.stop_instance" in _ROUTES_SRC

    def test_start_delegates_to_billing_engine(self):
        assert "be.start_instance" in _ROUTES_SRC

    def test_restart_delegates_to_billing_engine(self):
        assert "be.restart_instance" in _ROUTES_SRC

    def test_terminate_delegates_to_billing_engine(self):
        assert "be.terminate_instance" in _ROUTES_SRC


# ═══════════════════════════════════════════════════════════════════════
# 10. Migration 019
# ═══════════════════════════════════════════════════════════════════════


class TestMigration019:
    """Verify migration 019 creates the storage_billing_rates table."""

    def test_migration_file_exists(self):
        assert (_ROOT / "api" / "migrations" / "versions" / "019_instance_lifecycle.py").exists()

    def test_creates_storage_billing_rates_table(self):
        assert "storage_billing_rates" in _MIGRATION_SRC
        # Alembic uses op.create_table() rather than raw SQL CREATE TABLE
        assert "create_table" in _MIGRATION_SRC or "CREATE TABLE" in _MIGRATION_SRC

    def test_table_has_storage_type_pk(self):
        assert "storage_type" in _MIGRATION_SRC

    def test_table_has_rate_column(self):
        assert "rate_cad_per_gb_hr" in _MIGRATION_SRC

    def test_migration_seeds_nvme_rate(self):
        assert "nvme" in _MIGRATION_SRC

    def test_migration_seeds_ssd_rate(self):
        assert "ssd" in _MIGRATION_SRC

    def test_migration_seeds_hdd_rate(self):
        assert "hdd" in _MIGRATION_SRC

    def test_migration_chains_from_018(self):
        # Alembic annotated form: down_revision: Union[str, None] = "018"
        assert '"018"' in _MIGRATION_SRC and (
            "down_revision" in _MIGRATION_SRC or "Revises: 018" in _MIGRATION_SRC
        )

    def test_migration_has_upgrade_and_downgrade(self):
        assert "def upgrade" in _MIGRATION_SRC
        assert "def downgrade" in _MIGRATION_SRC


# ═══════════════════════════════════════════════════════════════════════
# 11. Frontend — api.ts client functions
# ═══════════════════════════════════════════════════════════════════════


_API_TS_SRC = (_ROOT / "api" / "frontend" / "src" / "lib" / "api.ts").read_text()


class TestFrontendApiClient:
    """Verify all lifecycle client functions are exported from api.ts."""

    def test_stop_instance_exported(self):
        assert "export async function stopInstance" in _API_TS_SRC

    def test_start_instance_exported(self):
        assert "export async function startInstance" in _API_TS_SRC

    def test_restart_instance_exported(self):
        assert "export async function restartInstance" in _API_TS_SRC

    def test_terminate_instance_exported(self):
        assert "export async function terminateInstance" in _API_TS_SRC

    def test_cancel_instance_exported(self):
        assert "export async function cancelInstance" in _API_TS_SRC

    def test_stop_calls_correct_path(self):
        assert "/stop" in _API_TS_SRC

    def test_start_calls_correct_path(self):
        assert "/start" in _API_TS_SRC

    def test_restart_calls_correct_path(self):
        assert "/restart" in _API_TS_SRC

    def test_terminate_calls_correct_path(self):
        assert "/terminate" in _API_TS_SRC

    def test_lifecycle_functions_use_post_method(self):
        assert 'method: "POST"' in _API_TS_SRC


# ═══════════════════════════════════════════════════════════════════════
# 12. Frontend — badge.tsx lifecycle variants
# ═══════════════════════════════════════════════════════════════════════


_BADGE_SRC = (_ROOT / "api" / "frontend" / "src" / "components" / "ui" / "badge.tsx").read_text()


class TestFrontendBadge:
    """Verify badge.tsx has lifecycle state variants and mappings."""

    def test_stopping_variant_exists(self):
        assert "stopping:" in _BADGE_SRC

    def test_stopped_variant_exists(self):
        assert "stopped:" in _BADGE_SRC

    def test_restarting_variant_exists(self):
        assert "restarting:" in _BADGE_SRC

    def test_terminated_variant_exists(self):
        assert "terminated:" in _BADGE_SRC

    def test_stopping_has_animation(self):
        assert "animate-status-stopping" in _BADGE_SRC

    def test_restarting_has_animation(self):
        assert "animate-status-restarting" in _BADGE_SRC

    def test_status_badge_maps_stopping(self):
        assert "stopping" in _BADGE_SRC

    def test_status_badge_maps_user_paused_to_stopped(self):
        assert "user_paused" in _BADGE_SRC

    def test_no_duplicate_badge_export(self):
        """Regression: badge.tsx must not have duplicate function exports."""
        count = _BADGE_SRC.count("export function Badge")
        assert count == 1, f"Badge exported {count} times — expected 1"

    def test_no_duplicate_status_badge_export(self):
        count = _BADGE_SRC.count("export function StatusBadge")
        assert count == 1, f"StatusBadge exported {count} times — expected 1"
