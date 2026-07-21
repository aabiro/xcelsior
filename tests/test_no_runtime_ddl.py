"""Production Postgres startup is seed-only (no runtime CREATE/ALTER).

Proves the shipped ``_ensure_pg_tables`` / ``_ensure_oauth_auth_tables``
paths no longer mutate schema on a head-migrated PostgreSQL, and that
residual tables formerly created only by ensure now live in Alembic
(migration 061).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIG_061 = PROJECT_ROOT / "migrations" / "versions" / "061_residual_runtime_ddl.py"


try:
    from db import _get_pg_pool, _ensure_pg_tables, _ensure_oauth_auth_tables
    from db import _pg_schema_is_migrated

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None


def test_migration_061_owns_residual_ensure_objects():
    """Static inventory: 061 must CREATE/ALTER every residual ensure object."""
    assert MIG_061.is_file(), "migration 061 missing"
    src = MIG_061.read_text(encoding="utf-8")
    for table in (
        "job_logs",
        "oauth_clients",
        "oauth_refresh_tokens",
        "team_invites",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in src, table
    for col in (
        "max_concurrent_instances",
        "pending_email",
        "token_cost_cad",
        "model_ref",
    ):
        assert col in src, col
    assert 'revision: str = "061"' in src or "revision = '061'" in src or 'revision = "061"' in src
    assert "a0985327493e" in src  # down_revision


def test_ensure_pg_tables_source_is_seed_only():
    """Shipped ensure must not contain CREATE/ALTER for control-plane tables."""
    src = inspect.getsource(_ensure_pg_tables)
    # Must seed reference pricing.
    assert "_seed_gpu_pricing" in src
    # Must refuse unmigrated DBs rather than invent schema.
    assert "_pg_schema_is_migrated" in src
    # No residual CREATE / ALTER TABLE in the body.
    assert "CREATE TABLE" not in src
    assert "ALTER TABLE" not in src


def test_ensure_oauth_source_is_noop():
    src = inspect.getsource(_ensure_oauth_auth_tables)
    assert "CREATE TABLE" not in src
    assert "ALTER TABLE" not in src


def test_startup_on_migrated_db_issues_no_schema_ddl(monkeypatch):
    """Instrument cursor.execute: ensure must not emit CREATE/ALTER.

    Spies via monkeypatch on the *class* method so pooled connections are
    not permanently mutated (mutating ``conn.cursor`` leaks across tests).
    """
    if _pool is None:
        pytest.skip("no pool")

    from psycopg import Cursor

    ddl_statements: list[str] = []
    real_execute = Cursor.execute

    def spy_execute(self, query, params=None, **kwargs):
        q = query if isinstance(query, str) else str(query)
        q_stripped = q.lstrip().upper()
        if q_stripped.startswith(("CREATE ", "ALTER ", "DROP ", "TRUNCATE ")):
            ddl_statements.append(q)
        return real_execute(self, query, params, **kwargs)

    monkeypatch.setattr(Cursor, "execute", spy_execute)

    with _pool.connection() as conn:
        assert _pg_schema_is_migrated(conn), "test DB must be Alembic-managed"
        _ensure_pg_tables(conn)
        _ensure_oauth_auth_tables(conn)
        conn.commit()

    assert ddl_statements == [], (
        f"runtime DDL on migrated DB (runtime-DDL violation): {ddl_statements!r}"
    )


def test_residual_tables_exist_via_alembic_not_ensure():
    """job_logs / oauth_* / team_invites must exist after migrations."""
    if _pool is None:
        pytest.skip("no pool")
    required = (
        "job_logs",
        "oauth_clients",
        "oauth_refresh_tokens",
        "team_invites",
        "gpu_pricing",
        "user_avatars",
    )
    with _pool.connection() as conn:
        for table in required:
            row = conn.execute(
                "SELECT to_regclass(%s)", (f"public.{table}",)
            ).fetchone()
            reg = row[0] if not isinstance(row, dict) else row["to_regclass"]
            assert reg is not None, f"missing table {table} (run alembic to 061)"

        for table, column in (
            ("users", "max_concurrent_instances"),
            ("users", "pending_email"),
            ("billing_cycles", "token_cost_cad"),
            ("billing_cycles", "model_ref"),
        ):
            row = conn.execute(
                """
                SELECT 1 FROM information_schema.columns
                 WHERE table_schema='public'
                   AND table_name=%s AND column_name=%s
                """,
                (table, column),
            ).fetchone()
            assert row is not None, f"missing {table}.{column} (migration 061)"


def test_pool_init_path_calls_seed_only_helper():
    """Static: pool construction still invokes ensure — which is seed-only."""
    import db as db_mod

    pool_src = inspect.getsource(db_mod._get_pg_pool)
    assert "_ensure_pg_tables" in pool_src
    # The helper itself is seed-only (covered above); together this is the
    # production startup contract: connect → ensure(seed) → ready.
