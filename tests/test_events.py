"""Tests for Xcelsior event store — hash chain, state machine, leases."""

import json
import os
import tempfile
import time

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_events_test_")
_tmpdir = _tmp_ctx.name
os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from events import (
    Event,
    EventStore,
    EventType,
    JobState,
    JobStateMachine,
    Lease,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
)


def _store() -> EventStore:
    """Isolated event store per test group."""
    return EventStore(db_path=os.path.join(_tmpdir, f"events_{os.urandom(4).hex()}.db"))


# ── Event Append & Hash Chain ─────────────────────────────────────────


class TestEventStore:
    """Append-only event log with tamper-evident hash chaining."""

    def test_append_returns_event_with_hash(self):
        es = _store()
        evt = es.append(Event(
            event_type=EventType.JOB_SUBMITTED,
            entity_type="job",
            entity_id="j1",
            actor="test",
        ))
        assert evt.event_hash
        assert len(evt.event_hash) == 64  # SHA-256 hex

    def test_first_event_has_empty_prev_hash(self):
        es = _store()
        evt = es.append(Event(
            event_type=EventType.JOB_SUBMITTED,
            entity_type="job",
            entity_id="j1",
        ))
        assert evt.prev_hash == ""

    def test_second_event_chains_to_first(self):
        es = _store()
        e1 = es.append(Event(event_type=EventType.JOB_SUBMITTED, entity_type="job", entity_id="j1"))
        e2 = es.append(Event(event_type=EventType.JOB_ASSIGNED, entity_type="job", entity_id="j1"))
        assert e2.prev_hash == e1.event_hash

    def test_verify_chain_valid(self):
        es = _store()
        for i in range(5):
            es.append(Event(event_type=EventType.HOST_REGISTERED, entity_type="host", entity_id=f"h{i}"))
        result = es.verify_chain()
        assert result["valid"] is True
        assert result["events_checked"] == 5

    def test_verify_chain_empty_store(self):
        es = _store()
        result = es.verify_chain()
        assert result["valid"] is True
        assert result["events_checked"] == 0

    def test_get_events_filters_by_entity(self):
        es = _store()
        es.append(Event(event_type=EventType.JOB_SUBMITTED, entity_type="job", entity_id="j1"))
        es.append(Event(event_type=EventType.HOST_REGISTERED, entity_type="host", entity_id="h1"))
        es.append(Event(event_type=EventType.JOB_COMPLETED, entity_type="job", entity_id="j1"))

        job_events = es.get_events(entity_type="job", entity_id="j1")
        assert len(job_events) == 2
        assert all(e.entity_id == "j1" for e in job_events)

    def test_get_events_filters_by_type(self):
        es = _store()
        es.append(Event(event_type=EventType.JOB_SUBMITTED, entity_type="job", entity_id="j1"))
        es.append(Event(event_type=EventType.JOB_COMPLETED, entity_type="job", entity_id="j1"))
        evts = es.get_events(event_type=EventType.JOB_COMPLETED)
        assert len(evts) == 1

    def test_get_entity_history(self):
        es = _store()
        es.append(Event(event_type=EventType.HOST_REGISTERED, entity_type="host", entity_id="h1"))
        es.append(Event(event_type=EventType.HOST_VERIFIED, entity_type="host", entity_id="h1"))
        hist = es.get_entity_history("host", "h1")
        assert len(hist) == 2


# ── State Machine ────────────────────────────────────────────────────


class TestStateMachine:
    """Job state machine — valid and invalid transitions."""

    def test_valid_transition_queued_to_assigned(self):
        es = _store()
        sm = JobStateMachine(es)
        evt = sm.transition("j1", "queued", "assigned")
        assert evt.event_type == EventType.JOB_ASSIGNED

    def test_valid_transition_assigned_to_leased(self):
        es = _store()
        sm = JobStateMachine(es)
        evt = sm.transition("j1", "assigned", "leased")
        assert evt.data["new_state"] == "leased"

    def test_valid_transition_running_to_completed(self):
        es = _store()
        sm = JobStateMachine(es)
        evt = sm.transition("j1", "running", "completed")
        assert evt.event_type == EventType.JOB_COMPLETED

    def test_invalid_transition_raises(self):
        es = _store()
        sm = JobStateMachine(es)
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition("j1", "queued", "completed")

    def test_invalid_from_terminal_state(self):
        es = _store()
        sm = JobStateMachine(es)
        with pytest.raises(ValueError):
            sm.transition("j1", "completed", "running")

    def test_failed_can_retry_to_queued(self):
        es = _store()
        sm = JobStateMachine(es)
        evt = sm.transition("j1", "failed", "queued")
        assert evt.event_type == EventType.JOB_REQUEUED

    def test_preempted_can_requeue(self):
        es = _store()
        sm = JobStateMachine(es)
        evt = sm.transition("j1", "preempted", "queued")
        assert evt.data["new_state"] == "queued"

    def test_unknown_state_raises(self):
        es = _store()
        sm = JobStateMachine(es)
        with pytest.raises(ValueError, match="Unknown state"):
            sm.transition("j1", "queued", "exploded")

    def test_get_job_timeline(self):
        es = _store()
        sm = JobStateMachine(es)
        sm.transition("j-timeline", "queued", "assigned")
        sm.transition("j-timeline", "assigned", "leased")
        sm.transition("j-timeline", "leased", "running")
        sm.transition("j-timeline", "running", "completed")
        timeline = sm.get_job_timeline("j-timeline")
        assert len(timeline) == 4
        assert timeline[-1]["data"]["new_state"] == "completed"


# ── Terminal States ──────────────────────────────────────────────────


class TestTerminalStates:
    """Verify terminal state definitions."""

    def test_completed_is_terminal(self):
        assert JobState.COMPLETED in TERMINAL_STATES

    def test_failed_is_terminal(self):
        assert JobState.FAILED in TERMINAL_STATES

    def test_cancelled_is_terminal(self):
        assert JobState.CANCELLED in TERMINAL_STATES

    def test_running_is_not_terminal(self):
        assert JobState.RUNNING not in TERMINAL_STATES


# ── Leases ───────────────────────────────────────────────────────────


class TestLeases:
    """Lease grant, renewal, expiry."""

    def test_grant_lease(self):
        es = _store()
        lease = es.grant_lease("j-lease-1", "h1")
        assert lease.job_id == "j-lease-1"
        assert lease.host_id == "h1"
        assert lease.status == "active"
        assert lease.expires_at > time.time()

    def test_renew_lease(self):
        es = _store()
        lease = es.grant_lease("j-renew", "h1")
        old_expiry = lease.expires_at
        time.sleep(0.05)
        renewed = es.renew_lease("j-renew", "h1")
        assert renewed is not None
        assert renewed.expires_at >= old_expiry

    def test_get_active_lease(self):
        es = _store()
        es.grant_lease("j-active", "h1")
        active = es.get_active_lease("j-active")
        assert active is not None
        assert active.status == "active"

    def test_release_lease(self):
        es = _store()
        es.grant_lease("j-release", "h1")
        released = es.release_lease("j-release")
        assert released is True
        assert es.get_active_lease("j-release") is None

    def test_no_active_lease_returns_none(self):
        es = _store()
        assert es.get_active_lease("nonexistent") is None

    def test_lease_dataclass_is_expired(self):
        lease = Lease(expires_at=time.time() - 1000)
        assert lease.is_expired is True

    def test_lease_dataclass_renew(self):
        lease = Lease()
        old = lease.expires_at
        time.sleep(0.01)
        new_exp = lease.renew()
        assert new_exp > old
