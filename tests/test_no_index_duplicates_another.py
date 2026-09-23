"""No index may duplicate what another index already covers.

A single-column btree index is redundant when a second index has that column as
its leading column and carries the same predicate — Postgres uses the second for
every lookup the first could serve, and the duplicate is maintained on every
INSERT and UPDATE for nothing. Seventeen had accumulated this way, five of them
exact duplicates of a UNIQUE constraint's own index:

    uq_usage_meters_one_per_attempt  UNIQUE (attempt_id) WHERE attempt_id IS NOT NULL
    idx_usage_meters_attempt                (attempt_id) WHERE attempt_id IS NOT NULL

They accumulate the same way every time: a migration adds `(customer_id)`, a
later one adds `(customer_id, created_at DESC)` for a new sort, and nobody
looks back at the first. Migration `117` dropped them and this keeps the count
at zero, so the next composite index has to come with the removal of the prefix
it supersedes.

The rule is deliberately narrow — leading column *and* identical predicate. A
two-column index is not judged against a three-column one here, because whether
that pays depends on selectivity, and this should not be guessing.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from db import _get_pg_pool


def _indexes() -> list[tuple[str, str, list[str], str, bool]]:
    """(table, name, columns, predicate, is_unique) for every public btree index."""
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'"
        ).fetchall()

    out = []
    for table, name, definition in rows:
        if "USING btree" not in definition:
            continue
        predicate = definition.split(" WHERE ", 1)[1].strip() if " WHERE " in definition else ""
        body = definition[definition.index("(") :]
        body = (
            body[: body.index(") WHERE")] if " WHERE " in definition else body[: body.rindex(")")]
        )
        columns = [c.strip() for c in body.lstrip("(").split(",")]
        out.append((table, name, columns, predicate, "UNIQUE INDEX" in definition))
    return out


def test_no_single_column_index_is_covered_by_another():
    indexes = _indexes()
    if not indexes:
        pytest.skip("no PostgreSQL schema available")

    redundant: list[str] = []
    for table, name, columns, predicate, _unique in indexes:
        if len(columns) != 1:
            continue
        for other_table, other_name, other_columns, other_predicate, other_unique in indexes:
            if other_name == name or other_table != table:
                continue
            if other_columns[0] != columns[0] or other_predicate != predicate:
                continue
            # Covered by a longer index sharing the prefix, or by a unique index
            # on exactly the same column — a unique index is an ordinary one too.
            if len(other_columns) > 1 or other_unique:
                redundant.append(f"{table}.{name} ({columns[0]}) — covered by {other_name}")
                break

    assert not redundant, (
        "these indexes are maintained on every write but serve no lookup another "
        "index does not already serve; drop them in a migration:\n  "
        + "\n  ".join(sorted(redundant))
    )
