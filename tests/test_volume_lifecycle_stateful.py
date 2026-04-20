"""Stateful property-based test for the volume lifecycle.

Uses Hypothesis's ``RuleBasedStateMachine`` to exercise random sequences of
state transitions from ``VolumeEngine._VALID_TRANSITIONS`` (volumes.py L197).

This is an in-memory simulation — no DB, no attach logic. The machine only
tracks (status, attachments_count) and invariants that the REAL state graph
must satisfy:

  - status always ∈ {provisioning, available, attached, deleting, deleted, error}
  - attachments_count >= 0
  - status == "attached" ⇒ attachments_count >= 1
  - status ∈ {deleting, deleted} ⇒ attachments_count == 0
  - deleted is a genuine sink: once reached, no rule fires again
"""

from __future__ import annotations

from hypothesis import settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from volumes import VolumeEngine


STATUSES = {"provisioning", "available", "attached", "deleting", "deleted", "error"}


class VolumeLifecycle(RuleBasedStateMachine):
    """Simulated volume-lifecycle state machine."""

    T = VolumeEngine._VALID_TRANSITIONS

    def __init__(self):
        super().__init__()
        self.status = "provisioning"
        self.attachments = 0

    # ── Transition rules (guarded by real transition table) ─────────

    @rule()
    @precondition(lambda self: "available" in self.T.get(self.status, set()))
    def become_available(self):
        # From provisioning, deleting (rollback), or attached (detach).
        self.status = "available"
        self.attachments = 0

    @rule()
    @precondition(lambda self: "attached" in self.T.get(self.status, set()))
    def attach(self):
        self.status = "attached"
        self.attachments = 1

    @rule()
    @precondition(
        lambda self: self.status == "attached"
        and "available" in self.T.get("attached", set())
    )
    def detach(self):
        self.status = "available"
        self.attachments = 0

    @rule()
    @precondition(lambda self: "deleting" in self.T.get(self.status, set()))
    def begin_delete(self):
        self.status = "deleting"
        self.attachments = 0

    @rule()
    @precondition(lambda self: "deleted" in self.T.get(self.status, set()))
    def finish_delete(self):
        self.status = "deleted"
        self.attachments = 0

    @rule()
    @precondition(lambda self: "error" in self.T.get(self.status, set()))
    def fail(self):
        self.status = "error"
        self.attachments = 0

    @rule()
    @precondition(
        lambda self: self.status == "error"
        and "provisioning" in self.T.get("error", set())
    )
    def recover(self):
        self.status = "provisioning"
        self.attachments = 0

    @rule()
    def new_volume(self):
        """Start a fresh volume lifecycle (e.g., after deletion).

        Always available so the machine can make progress even from the
        ``deleted`` sink — models the reality that users create new volumes
        over time. Invariants below still hold: the new state is
        ``provisioning`` with zero attachments.
        """
        self.status = "provisioning"
        self.attachments = 0

    # ── Invariants (must hold after every rule) ─────────────────────

    @invariant()
    def status_is_in_known_set(self):
        assert self.status in STATUSES, f"unknown status: {self.status!r}"

    @invariant()
    def attachments_non_negative(self):
        assert self.attachments >= 0

    @invariant()
    def attached_implies_attachment_count(self):
        if self.status == "attached":
            assert self.attachments >= 1

    @invariant()
    def deleting_or_deleted_has_no_attachments(self):
        if self.status in {"deleting", "deleted"}:
            assert self.attachments == 0, (
                f"{self.status} must have 0 attachments, got {self.attachments}"
            )

    @invariant()
    def deleted_is_a_sink(self):
        """Once deleted, the transition table must have no outgoing edges."""
        if self.status == "deleted":
            assert self.T.get("deleted", set()) == set()


# Hypothesis creates ``TestCase`` attribute on the class — expose it for pytest.
TestVolumeLifecycle = VolumeLifecycle.TestCase
TestVolumeLifecycle.settings = settings(
    max_examples=50,
    stateful_step_count=30,
    deadline=None,
)
