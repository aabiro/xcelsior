"""Every append-only table is either erasable or knowingly exempt.

**This test pins a known-unresolved state. It is load-bearing — do not delete it
as a stale assertion about a bug.**

`privacy_sinks.verify_subject_absence` returns a verdict named *absence* over a
hand-enumerated list of tables. Three tables carry an append-only (WORM) trigger
and appear in no erasure path at all: the trigger rejects DELETE unconditionally
and partitioning prunes by time rather than by tenant, so no existing mechanism
reaches them for one subject. Whether that is right — pseudonymise at erasure
time, or retain under a documented basis — **has not been decided**, and this
records that rather than asserting either answer.

## Why nothing here is written down twice

Both sides are derived:

* the WORM set comes from `pg_trigger`, so a table that grows an append-only
  trigger tomorrow is included without anyone remembering;
* the reachable set comes from `verify_subject_absence`'s own source, so adding
  a table to that function is what clears it here.

The single literal is `ACKNOWLEDGED_UNRESOLVED` — the ratchet: the exceptions
someone has actually looked at. A **new** WORM table cannot join it silently,
which is the whole job of this file.
"""

from __future__ import annotations

import inspect
import os
import re

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _c.execute("SELECT 1")
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")

#: WORM tables whose erasure treatment is open, and known to be open.
#:
#: Not a list of tables to ignore — a list of decisions someone owes. Removing a
#: name means the decision was made: either the table became reachable from the
#: erasure path, or its retention basis was written down at
#: `verify_subject_absence`. Adding one means a new table arrived in the same
#: unresolved state and that should be a deliberate act, not a silent one.
ACKNOWLEDGED_UNRESOLVED = {
    "audit_events_v2",       # 072 — the audit stream
    "audit_checkpoints",     # 075 — signed Merkle manifests over it
    "placement_decisions",   # 105/106 — the placement trail, P5 clause 3
}


def _worm_tables(conn) -> set[str]:
    """Append-only parents, derived from the triggers themselves.

    Partitions are excluded: a row trigger on a partitioned parent propagates to
    every child, so counting children would list the same table four times and
    grow the set every month.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT c.relname
          FROM pg_trigger t
          JOIN pg_class c ON c.oid = t.tgrelid
          JOIN pg_proc  p ON p.oid = t.tgfoid
         WHERE NOT t.tgisinternal
           AND pg_get_functiondef(p.oid) ILIKE '%%append-only%%'
           AND NOT EXISTS (SELECT 1 FROM pg_inherits WHERE inhrelid = c.oid)
        """
    ).fetchall()
    return {str(r[0]) for r in rows}


def _tables_the_erasure_check_reaches() -> set[str]:
    """Tables named in `verify_subject_absence`, read from its own source."""
    import privacy_sinks

    source = inspect.getsource(privacy_sinks.verify_subject_absence)
    return set(re.findall(r"FROM\s+([a-z_][a-z0-9_]*)", source))


def test_the_derivation_finds_something_on_both_sides():
    """Calibration. Two empty sets would make every assertion below vacuous."""
    with pg_transaction() as conn:
        worm = _worm_tables(conn)
    reached = _tables_the_erasure_check_reaches()

    assert worm, (
        "no append-only triggers found — the derivation is broken, not the "
        "database. Check that the trigger functions still raise 'append-only'."
    )
    assert len(reached) > 5, (
        "verify_subject_absence appears to name almost no tables; the source "
        "regex has stopped matching and this file is now asserting nothing"
    )


def test_no_new_worm_table_joins_the_unresolved_set_silently():
    """The ratchet.

    Red here does **not** mean erasure is broken. It means a table now carries
    an append-only trigger and nobody has said what happens to it when a subject
    asks to be erased. Answer the question, then update
    `ACKNOWLEDGED_UNRESOLVED`.
    """
    with pg_transaction() as conn:
        worm = _worm_tables(conn)
    unreachable = worm - _tables_the_erasure_check_reaches()

    new = unreachable - ACKNOWLEDGED_UNRESOLVED
    assert not new, (
        f"append-only table(s) {sorted(new)} are unreachable from "
        "privacy_sinks.verify_subject_absence, and their erasure treatment has "
        "not been decided. This is a decision owed, not a bug to patch: either "
        "make the table reachable, or record its retention basis in that "
        "function's docstring — then add it to ACKNOWLEDGED_UNRESOLVED."
    )


def test_a_resolved_table_is_removed_from_the_list_rather_than_left():
    """The other direction, so the list cannot rot into a permanent excuse.

    If a table became reachable from the erasure path, it is no longer an open
    decision and must leave `ACKNOWLEDGED_UNRESOLVED` — otherwise the list stops
    describing anything and the next reader cannot tell which entries are live.
    """
    with pg_transaction() as conn:
        worm = _worm_tables(conn)
    reached = _tables_the_erasure_check_reaches()

    resolved = {t for t in ACKNOWLEDGED_UNRESOLVED if t in reached}
    assert not resolved, (
        f"{sorted(resolved)} are now reachable from verify_subject_absence, so "
        "their erasure treatment is decided. Remove them from "
        "ACKNOWLEDGED_UNRESOLVED."
    )

    gone = {t for t in ACKNOWLEDGED_UNRESOLVED if t not in worm}
    assert not gone, (
        f"{sorted(gone)} no longer carry an append-only trigger, so they are "
        "not WORM tables and do not belong in ACKNOWLEDGED_UNRESOLVED."
    )


def test_the_function_states_that_its_enumeration_is_partial():
    """The claim the code makes has to carry its own caveat.

    A verdict named *absence* over a hand-enumerated list is the one place in
    the erasure path that asserts something stronger than the mechanism
    establishes. That is fine while it is written down, and a defect the moment
    it is not — so the note is part of the contract, not commentary on it.
    """
    import privacy_sinks

    doc = (privacy_sinks.verify_subject_absence.__doc__ or "").lower()
    assert "append-only" in doc and "hand-enumerated" in doc, (
        "verify_subject_absence no longer records that its enumeration is "
        "partial and that the WORM tables sit outside it. The function returns "
        "a verdict named 'absence'; without that note it reads as complete."
    )
