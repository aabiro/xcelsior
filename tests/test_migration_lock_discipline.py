"""Migration lock discipline (`migrations/README.md` rule 5).

Rule 5 requires `lock_timeout` set and expand-contract shapes that can run
against live traffic. It was prose only, and on 2026-08-04 the first production
deploy of migrations `080`–`098` died on `095` with `deadlock detected` while
holding `ACCESS EXCLUSIVE` on fifteen tables in one transaction. This file is
the gate that prose was missing.

Three things are checked, and the contention ones are driven **both ways**
against real PostgreSQL: once proving the scenario genuinely deadlocks the old
single-transaction shape, once proving `migrations/lock_safe.py` survives it. A
gate that has never been observed to fail is not a gate, and here the failing
arm is the production incident reproduced in miniature.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import NullPool, create_engine, text
from sqlalchemy.exc import DBAPIError, OperationalError

from migrations.lock_safe import apply_in_own_transactions
from tests.test_from_empty_bootstrap import (
    _admin_dsn,
    _db_url,
    _drop_database,
    _try_create_database,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PY = PROJECT_ROOT / "migrations" / "env.py"

DEADLOCK = "40P01"
LOCK_TIMEOUT = "55P03"


# ── 1. The runner configuration `lock_safe` depends on ────────────────


def _runner_defects(source: str) -> list[str]:
    """Faults in an Alembic `env.py` that reintroduce the 095 failure."""
    defects = []
    if not re.search(r"transaction_per_migration\s*=\s*True", source):
        defects.append(
            "transaction_per_migration=True missing: one transaction would span "
            "the whole upgrade, holding every migration's locks until the last "
            "one commits, and migrations/lock_safe.py would block on its own run"
        )
    if not re.search(r"SET lock_timeout", source):
        defects.append(
            "no session lock_timeout: a blocked ALTER TABLE queues ahead of "
            "every later query on the table and stalls live traffic (README rule 5)"
        )
    return defects


def test_env_py_keeps_the_runner_shape_lock_safe_requires():
    assert _runner_defects(ENV_PY.read_text()) == []


def test_the_runner_check_fails_on_a_runner_missing_both():
    """The same predicate against the shape that produced the incident."""
    legacy = (
        "with engine.connect() as connection:\n"
        "    context.configure(connection=connection, target_metadata=None)\n"
        "    with context.begin_transaction():\n"
        "        context.run_migrations()\n"
    )
    defects = _runner_defects(legacy)
    assert len(defects) == 2
    assert any("transaction_per_migration" in d for d in defects)
    assert any("lock_timeout" in d for d in defects)


# ── 2. Contention, against real PostgreSQL ────────────────────────────


@pytest.fixture(scope="module")
def scratch_db():
    """A throwaway database with two tables, no schema and no chain needed."""
    admin = _admin_dsn()
    if not admin:
        pytest.skip("no PostgreSQL DSN in the environment")
    name = f"xcel_locktest_{uuid.uuid4().hex[:10]}"
    _try_create_database(admin, name)  # skips if privileges are missing
    url = _db_url(admin, name).replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE first_table (id int, amount_cad float)"))
            conn.execute(text("CREATE TABLE second_table (id int, amount_cad float)"))
            conn.execute(text("INSERT INTO first_table VALUES (1, 1.5)"))
            conn.execute(text("INSERT INTO second_table VALUES (1, 2.5)"))
            conn.commit()
        yield engine
    finally:
        engine.dispose()
        _drop_database(admin, name)


def _sqlstate(exc: BaseException) -> str | None:
    orig = getattr(exc, "orig", None)
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


class _ReverseOrderRequest:
    """One application transaction reading the two tables back to front.

    This is the other half of the production deadlock: the migration walks its
    table list in order while a request reads in whatever order its query
    needs, and the two end up each holding what the other wants. Driven by
    events rather than sleeps, because a race that only sometimes deadlocks is
    not a gate — it is a flake in whichever direction is convenient.

    Sequence: read `second_table` and hold it → wait until the migration holds
    `first_table` → reach for `first_table`.
    """

    def __init__(self, engine):
        self.engine = engine
        self.holds_second = threading.Event()
        self.may_proceed = threading.Event()
        self.sqlstate: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT count(*) FROM second_table"))
                self.holds_second.set()
                self.may_proceed.wait(timeout=30)
                conn.execute(text("SELECT count(*) FROM first_table"))
                conn.commit()
        except DBAPIError as exc:
            # A deadlock may pick the request as the victim rather than the
            # migration; either way the scenario worked.
            self.sqlstate = _sqlstate(exc)

    def __enter__(self):
        self._thread.start()
        assert self.holds_second.wait(timeout=15), "request never took its lock"
        return self

    def __exit__(self, *exc):
        self.may_proceed.set()
        self._thread.join(timeout=20)
        return False


@pytest.mark.needs_db
def test_one_transaction_across_both_tables_deadlocks(scratch_db):
    """The failing arm: the shape 095 had, against the traffic it met.

    Without this, the passing arm below proves only that a migration can
    succeed when nothing contends — which was never in doubt.
    """
    migration_state: str | None = None
    with _ReverseOrderRequest(scratch_db) as request, scratch_db.connect() as conn:
        conn.execute(text("SET lock_timeout = '15s'"))
        conn.execute(text("ALTER TABLE first_table ADD COLUMN IF NOT EXISTS c1 bigint"))
        # Held, not released: one transaction across both tables is the defect.
        request.may_proceed.set()
        time.sleep(0.5)  # let the request enqueue behind this lock
        try:
            conn.execute(text("ALTER TABLE second_table ADD COLUMN IF NOT EXISTS c2 bigint"))
        except OperationalError as exc:
            migration_state = _sqlstate(exc)
        conn.rollback()

    assert DEADLOCK in {migration_state, request.sqlstate}, (
        f"expected one side to be a deadlock victim; migration={migration_state}, "
        f"request={request.sqlstate}"
    )


@pytest.mark.needs_db
def test_per_table_transactions_survive_the_same_traffic(scratch_db):
    """The passing arm: same DDL, same traffic, one transaction per table.

    The claim is stronger than "no exception": *neither* side is killed. The
    migration commits `first_table` before reaching for `second_table`, so the
    request's second read is granted, it commits, and the migration's retry
    finds the lock free.
    """

    def add(table: str, column: str):
        def apply(conn):
            conn.execute(
                text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} bigint")
            )
            conn.execute(
                text(f"UPDATE {table} SET {column} = ROUND(amount_cad * 1000000)::bigint")
            )

        return apply

    with _ReverseOrderRequest(scratch_db) as request, scratch_db.connect() as bind:
        request.may_proceed.set()
        apply_in_own_transactions(
            [
                ("first_table.amount_micros", add("first_table", "amount_micros")),
                ("second_table.amount_micros", add("second_table", "amount_micros")),
            ],
            tables=["first_table", "second_table"],
            lock_timeout="1s",
            attempts=15,
            bind=bind,
        )

    assert request.sqlstate is None, f"the request was killed: {request.sqlstate}"
    with scratch_db.connect() as conn:
        for table in ("first_table", "second_table"):
            got = conn.execute(text(f"SELECT amount_micros FROM {table}")).scalar()
            assert got is not None, f"{table} was not backfilled"


@pytest.mark.needs_db
def test_a_shared_transaction_is_refused_rather_than_hanging(scratch_db):
    """The precondition, checked instead of documented.

    If the runner ever loses `transaction_per_migration`, `lock_safe`'s second
    connection would wait on a transaction that cannot commit until it returns.
    That must present as a named configuration error, not a mystery timeout.
    """
    with scratch_db.connect() as bind:
        bind.execute(text("LOCK TABLE first_table IN ACCESS EXCLUSIVE MODE"))
        with pytest.raises(RuntimeError, match="transaction_per_migration"):
            apply_in_own_transactions(
                [("noop", lambda conn: conn.execute(text("SELECT 1")))],
                tables=["first_table"],
                bind=bind,
            )
        bind.rollback()
