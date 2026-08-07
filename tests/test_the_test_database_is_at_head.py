"""The database the suite runs against must be at the repository's head.

`scripts/setup_pytest_db.sh` builds `xcelsior_pytest` from a **schema-only dump**
and then *stamps* `alembic_version` with whatever head was current at the time.
It does not run migrations, and `run-tests.sh` does not call it. So the test
database is provisioned once and drifts from that moment on — and because it is
stamped rather than migrated, `alembic upgrade head` against it is a no-op that
reports success.

The consequence is quiet and total: **a new migration is never exercised by the
suite.** Every test runs against the schema as it stood when someone last ran
the setup script. A migration that drops a column tests still read, adds a NOT
NULL constraint fixtures violate, or renames something the ORM maps would all
pass locally and fail on deploy.

Found on 2026-08-07 by `test_the_schema_holds_no_cad_column`, which asserted
zero `_cad` columns and failed — not because migration `100` was wrong, but
because the suite's database was still at `099` and nothing had noticed. The
main development database was at `100`; the one the tests read was not.

This guard fails loudly and says what to run, rather than migrating the
database out from under whoever is debugging. Silently mutating state during a
test run is how a suite becomes unreproducible.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

#: Both spellings appear in `migrations/versions/`: `revision = "037"` and the
#: annotated `revision: str = "037"`. A pattern that only matches the first
#: silently drops those files, so their parents look like heads — the first
#: draft of this file reported three (`036`, `059`, `100`) for exactly that
#: reason, and would have been a confusing failure for whoever hit it next.
_REVISION = re.compile(r"^revision\s*(?::\s*\w+\s*)?=\s*[\"']([^\"']+)[\"']", re.M)
_DOWN_REVISION = re.compile(r"^down_revision\s*(?::[^=]*)?=\s*[\"']([^\"']+)[\"']", re.M)

REPO = pathlib.Path(__file__).resolve().parent.parent
VERSIONS = REPO / "migrations" / "versions"


def repository_head() -> str:
    """The single head revision, read from the migration files themselves.

    Deliberately not `alembic heads`: that shells out and needs configuration,
    and `tests/test_migration_ledger.py` already proves the chain has exactly
    one head. This only needs to know which revision it is.
    """
    revisions: dict[str, str | None] = {}
    for path in VERSIONS.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        rev = _REVISION.search(text)
        down = _DOWN_REVISION.search(text)
        if rev:
            revisions[rev.group(1)] = down.group(1) if down else None
    assert revisions, "no migration revisions found"
    parents = {d for d in revisions.values() if d}
    heads = sorted(set(revisions) - parents)
    assert len(heads) == 1, f"expected exactly one head, found {heads}"
    return heads[0]


@pytest.mark.needs_db
def test_the_suite_runs_against_the_head_schema():
    """The whole point: what the tests read is what the migrations produce."""
    import psycopg

    dsn = os.environ.get("XCELSIOR_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    assert dsn, "no database DSN configured for the test session"

    with psycopg.connect(dsn, connect_timeout=10) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()

    assert row, "alembic_version is empty; the test database was never stamped"
    head = repository_head()
    assert row[0] == head, (
        f"the test database is at {row[0]} but the repository head is {head}, so "
        "every test in this suite is running against a schema that does not "
        "include the newer migration(s). Bring it up with:\n\n"
        "    XCELSIOR_POSTGRES_DSN=<the xcelsior_pytest DSN> \\\n"
        "      PYTHONPATH=$PWD .venv/bin/alembic upgrade head\n\n"
        "or rebuild it from a fresh dump with scripts/setup_pytest_db.sh. Note "
        "that the setup script *stamps* rather than migrates, which is why this "
        "drift is silent without this check."
    )


def test_the_head_reader_agrees_with_the_ledger():
    """Calibration: two independent derivations of the same fact.

    `tests/test_migration_ledger.py` declares the head as a literal that a human
    updates. This file derives it from the files. If they ever disagree, one of
    them is wrong and this says so — rather than both drifting together.
    """
    ledger = (REPO / "tests" / "test_migration_ledger.py").read_text(encoding="utf-8")
    declared = re.search(r"^EXPECTED_HEAD\s*=\s*[\"']([^\"']+)[\"']", ledger, re.M)
    assert declared, "test_migration_ledger.py no longer declares EXPECTED_HEAD"
    assert declared.group(1) == repository_head(), (
        f"the ledger declares head {declared.group(1)} but the migration files "
        f"chain to {repository_head()}"
    )
