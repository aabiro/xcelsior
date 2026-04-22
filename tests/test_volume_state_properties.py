"""Property-based tests for the volume state machine.

Target: ``VolumeEngine._VALID_TRANSITIONS`` — the pure-data transition graph
declared at volumes.py L196-L204.

These tests exercise graph invariants only. No DB access, no mocks required.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from volumes import VolumeEngine

# Canonical list of statuses the state machine must recognise.
# Must match the keys + union of targets in ``_VALID_TRANSITIONS``.
STATUSES: list[str] = [
    "provisioning",
    "available",
    "attached",
    "deleting",
    "deleted",
    "error",
]


def _transitions() -> dict[str, set[str]]:
    """Return the live transition table (read once per test for clarity)."""
    return VolumeEngine._VALID_TRANSITIONS


def test_transition_table_exists_and_is_nonempty():
    t = _transitions()
    assert isinstance(t, dict)
    assert len(t) >= 5, "transition table shrank unexpectedly"


def test_deleted_is_terminal():
    """Once a volume is deleted, no transitions out are permitted."""
    # Either absent from the map, or present with an empty set — both count
    # as terminal. The map docstring declares ``deleted`` is terminal.
    assert _transitions().get("deleted", set()) == set()


def test_all_transition_targets_are_valid_statuses():
    """No typos/orphans: every declared target is a known status."""
    t = _transitions()
    for src, targets in t.items():
        assert src in STATUSES, f"unknown source status in map: {src!r}"
        for tgt in targets:
            assert tgt in STATUSES, f"unknown target status {tgt!r} in transitions from {src!r}"


def test_all_non_terminal_reachable_from_provisioning():
    """BFS from provisioning must reach every non-terminal status."""
    t = _transitions()
    visited = {"provisioning"}
    queue = ["provisioning"]
    while queue:
        s = queue.pop(0)
        for nxt in t.get(s, set()):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    # ``deleted`` is terminal — its presence in visited is required by graph
    # reachability (delete flow: provisioning→available→deleting→deleted),
    # so include it.
    for s in STATUSES:
        assert s in visited, f"status {s!r} unreachable from provisioning"


def test_available_attached_roundtrip():
    """Core user flow: available ↔ attached must work both directions."""
    t = _transitions()
    assert "attached" in t["available"], "available → attached missing"
    assert "available" in t["attached"], "attached → available missing"


def test_deleting_cannot_return_to_attached():
    """Data integrity: once a delete is started, re-attach is forbidden."""
    t = _transitions()
    assert "attached" not in t.get(
        "deleting", set()
    ), "deleting → attached would allow attach of half-deleted volume"


def test_error_state_is_recoverable():
    """``error`` must have at least one outgoing edge (no dead-lock)."""
    t = _transitions()
    next_from_error = t.get("error", set())
    assert next_from_error & {
        "provisioning",
        "deleting",
    }, f"error has no recovery path; outgoing={next_from_error!r}"


def test_no_self_loops():
    """A status should never transition to itself."""
    t = _transitions()
    for src, targets in t.items():
        assert src not in targets, f"self-loop detected: {src!r} → {src!r}"


def test_provisioning_cannot_skip_to_attached():
    """Invariant from the state diagram: must pass through ``available``."""
    t = _transitions()
    assert "attached" not in t.get(
        "provisioning", set()
    ), "provisioning → attached would skip NFS-export verification"


@given(status=st.sampled_from(STATUSES))
@settings(deadline=None, max_examples=100)
def test_transition_entries_are_sets_of_valid_statuses(status):
    """Every entry in the map (if present) is a set whose elements are valid."""
    t = _transitions()
    if status in t:
        entry = t[status]
        assert isinstance(
            entry, (set, frozenset)
        ), f"transitions from {status!r} must be a set, got {type(entry).__name__}"
        for tgt in entry:
            assert isinstance(tgt, str)
            assert tgt in STATUSES


@given(
    src=st.sampled_from(STATUSES),
    tgt=st.sampled_from(STATUSES),
)
@settings(deadline=None, max_examples=100)
def test_transitions_are_consistent_with_declared_map(src, tgt):
    """Property: (src, tgt) is in the map iff tgt ∈ transitions[src].

    This is a definitional tautology — its purpose is to catch the case where
    someone accidentally switches the ``_VALID_TRANSITIONS`` structure to a
    different type (e.g. list of tuples) that would silently break lookups.
    """
    t = _transitions()
    expected = tgt in t.get(src, set())
    # Re-query via ``in`` operator — both ways must agree.
    actual = tgt in t.get(src, frozenset())
    assert expected == actual
