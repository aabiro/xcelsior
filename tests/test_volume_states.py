"""State machine transition matrix tests for VolumeEngine.

Tests every valid and invalid transition combination (6×6=36)
against the _VALID_TRANSITIONS guard.
"""

import os
import pytest
from unittest.mock import MagicMock
from contextlib import contextmanager

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")


ALL_STATUSES = ["provisioning", "available", "attached", "deleting", "deleted", "error"]


def _make_engine_with_status(current_status):
    """Return a VolumeEngine with a mocked conn returning the given current status."""
    from volumes import VolumeEngine

    engine = VolumeEngine()
    fake_conn = MagicMock()

    def _execute(sql, params=None):
        result = MagicMock()
        sql_lower = sql.strip().lower()
        if "select status" in sql_lower and "for update" in sql_lower:
            result.fetchone.return_value = {"status": current_status}
        elif "update volumes set status" in sql_lower:
            result.rowcount = 1
        else:
            result.fetchone.return_value = None
        return result

    fake_conn.execute = _execute
    return engine, fake_conn


class TestTransitionMatrix:
    """Exhaustive state machine transition test: 36 combos."""

    # Expected valid transitions from the plan
    VALID = {
        "provisioning": {"available", "error"},
        "available":    {"attached", "deleting"},
        "attached":     {"available"},
        "deleting":     {"deleted", "available"},
        "error":        {"provisioning", "deleting"},
        # "deleted" → nothing (terminal)
    }

    @pytest.mark.parametrize("from_status", ALL_STATUSES)
    @pytest.mark.parametrize("to_status", ALL_STATUSES)
    def test_transition(self, from_status, to_status):
        """Test every from→to combo: valid ones succeed, invalid ones raise."""
        engine, fake_conn = _make_engine_with_status(from_status)
        allowed = self.VALID.get(from_status, set())

        if to_status in allowed:
            result = engine._transition_status(fake_conn, "vol-test", to_status)
            assert result == to_status
        else:
            with pytest.raises(ValueError, match="Invalid volume transition"):
                engine._transition_status(fake_conn, "vol-test", to_status)


class TestTransitionWithExplicitCurrent:
    """Test _transition_status with current= parameter (skips SELECT)."""

    def test_valid_with_current_param(self):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        fake_conn = MagicMock()
        fake_conn.execute.return_value = MagicMock(rowcount=1)

        result = engine._transition_status(fake_conn, "vol-x", "available", current="provisioning")
        assert result == "available"
        # Should NOT have done a SELECT since current was provided
        calls = [c[0][0] for c in fake_conn.execute.call_args_list]
        assert not any("SELECT" in c.upper() for c in calls)

    def test_invalid_with_current_param(self):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        fake_conn = MagicMock()

        with pytest.raises(ValueError, match="Invalid volume transition"):
            engine._transition_status(fake_conn, "vol-x", "deleted", current="provisioning")

    def test_volume_not_found(self):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        fake_conn = MagicMock()
        result = MagicMock()
        result.fetchone.return_value = None
        fake_conn.execute.return_value = result

        with pytest.raises(ValueError, match="not found"):
            engine._transition_status(fake_conn, "vol-ghost", "available")


class TestTerminalState:
    """'deleted' is a terminal state — no outgoing transitions."""

    @pytest.mark.parametrize("to_status", ALL_STATUSES)
    def test_deleted_rejects_all(self, to_status):
        engine, fake_conn = _make_engine_with_status("deleted")
        with pytest.raises(ValueError, match="terminal state"):
            engine._transition_status(fake_conn, "vol-dead", to_status)


class TestTransitionsDictMatchesCode:
    """Verify the _VALID_TRANSITIONS dict in volumes.py matches expectations."""

    def test_dict_has_all_non_terminal_states(self):
        from volumes import VolumeEngine
        transitions = VolumeEngine._VALID_TRANSITIONS
        # Every non-terminal status must have an entry
        for s in ["provisioning", "available", "attached", "deleting", "error"]:
            assert s in transitions, f"{s} missing from _VALID_TRANSITIONS"

    def test_deleted_not_in_dict(self):
        from volumes import VolumeEngine
        assert "deleted" not in VolumeEngine._VALID_TRANSITIONS

    def test_no_self_transitions(self):
        """No status should be able to transition to itself."""
        from volumes import VolumeEngine
        for status, targets in VolumeEngine._VALID_TRANSITIONS.items():
            assert status not in targets, f"Self-transition found: {status} → {status}"

    def test_error_allows_retry_and_delete(self):
        from volumes import VolumeEngine
        error_targets = VolumeEngine._VALID_TRANSITIONS["error"]
        assert "provisioning" in error_targets, "error must allow retry (→provisioning)"
        assert "deleting" in error_targets, "error must allow delete (→deleting)"

    def test_deleting_allows_rollback(self):
        from volumes import VolumeEngine
        del_targets = VolumeEngine._VALID_TRANSITIONS["deleting"]
        assert "available" in del_targets, "deleting must allow rollback (→available)"
        assert "deleted" in del_targets, "deleting must allow completion (→deleted)"
