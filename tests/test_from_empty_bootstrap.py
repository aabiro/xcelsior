"""Deterministic from-empty PostgreSQL bootstrap (Alembic head).

Drives the *shipped* bootstrap path (`scripts/bootstrap_pg_from_empty.sh`
→ real `alembic upgrade head`) against a throwaway empty database. This is
the gap that previously forced CI to restore `ci-cache/pg_schema.sql`.

Requires a Postgres role that can CREATE DATABASE (CI's POSTGRES_USER is
superuser; local runners fall back to `sudo -u postgres createdb`).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_SCRIPT = PROJECT_ROOT / "scripts" / "bootstrap_pg_from_empty.sh"
MIG_056 = PROJECT_ROOT / "migrations" / "versions" / "056_durable_control_work.py"

# Tables that pure-alembic must create without runtime DDL or a dump restore.
REQUIRED_AT_HEAD = (
    "agent_commands",
    "jobs",
    "hosts",
    "outbox_events",
    "job_attempts",
    "alembic_version",
    # Residual tables formerly ensure-only (migration 061).
    "job_logs",
    "oauth_clients",
    "oauth_refresh_tokens",
    "team_invites",
)
EXPECTED_HEAD = "069"



def _admin_dsn() -> str | None:
    """DSN used to CREATE/DROP throwaway databases (connect to 'postgres')."""
    dsn = (
        os.environ.get("XCELSIOR_POSTGRES_DSN")
        or os.environ.get("XCELSIOR_PG_DSN")
        or os.environ.get("XCELSIOR_TEST_POSTGRES_DSN")
        or os.environ.get("DATABASE_URL")
    )
    if not dsn:
        return None
    # Strip SQLAlchemy driver suffix if present.
    dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    parsed = urlparse(dsn)
    if not parsed.hostname:
        return None
    return urlunparse(parsed._replace(path="/postgres"))


def _db_url(admin_dsn: str, dbname: str) -> str:
    parsed = urlparse(admin_dsn)
    return urlunparse(parsed._replace(path=f"/{dbname}"))


def _psql_env(admin_dsn: str) -> dict[str, str]:
    parsed = urlparse(admin_dsn)
    env = os.environ.copy()
    if parsed.password:
        env["PGPASSWORD"] = parsed.password
    return env


def _try_create_database(admin_dsn: str, dbname: str) -> bool:
    """Create an empty database. Returns False if privileges are missing."""
    parsed = urlparse(admin_dsn)
    user = parsed.username or "xcelsior"
    host = parsed.hostname or "localhost"
    port = str(parsed.port or 5432)
    env = _psql_env(admin_dsn)

    sql = f'CREATE DATABASE "{dbname}" OWNER "{user}"'
    r = subprocess.run(
        [
            "psql",
            "-h",
            host,
            "-p",
            port,
            "-U",
            user,
            "-d",
            "postgres",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            sql,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode == 0:
        return True

    # Local app roles often lack CREATEDB; peer-auth superuser fallback.
    if shutil.which("sudo") and shutil.which("createdb"):
        r2 = subprocess.run(
            ["sudo", "-u", "postgres", "createdb", "-O", user, dbname],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r2.returncode == 0:
            return True
        pytest.skip(
            "cannot CREATE DATABASE "
            f"(psql: {r.stderr.strip()!r}; sudo: {r2.stderr.strip()!r})"
        )
    pytest.skip(f"cannot CREATE DATABASE: {r.stderr.strip()!r}")


def _drop_database(admin_dsn: str, dbname: str) -> None:
    parsed = urlparse(admin_dsn)
    user = parsed.username or "xcelsior"
    host = parsed.hostname or "localhost"
    port = str(parsed.port or 5432)
    env = _psql_env(admin_dsn)
    # Terminate other sessions first so DROP is not blocked.
    term = (
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{dbname}' AND pid <> pg_backend_pid();"
    )
    subprocess.run(
        [
            "psql",
            "-h",
            host,
            "-p",
            port,
            "-U",
            user,
            "-d",
            "postgres",
            "-c",
            term,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    r = subprocess.run(
        [
            "psql",
            "-h",
            host,
            "-p",
            port,
            "-U",
            user,
            "-d",
            "postgres",
            "-c",
            f'DROP DATABASE IF EXISTS "{dbname}"',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0 and shutil.which("sudo"):
        subprocess.run(
            ["sudo", "-u", "postgres", "dropdb", "--if-exists", dbname],
            capture_output=True,
            text=True,
            timeout=60,
        )


def _run_bootstrap(db_dsn: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["XCELSIOR_POSTGRES_DSN"] = db_dsn
    env["XCELSIOR_PG_DSN"] = db_dsn
    env["DATABASE_URL"] = db_dsn
    # Schema authority only; seed ensure is separate from the dump gap.
    env["BOOTSTRAP_SKIP_ENSURE"] = "1"
    parsed = urlparse(db_dsn.replace("postgresql+psycopg://", "postgresql://", 1))
    if parsed.password:
        env["PGPASSWORD"] = parsed.password
    return subprocess.run(
        ["bash", str(BOOTSTRAP_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _alembic_current(db_dsn: str) -> str:
    env = os.environ.copy()
    env["XCELSIOR_POSTGRES_DSN"] = db_dsn
    env["XCELSIOR_PG_DSN"] = db_dsn
    env["DATABASE_URL"] = db_dsn
    r = subprocess.run(
        ["alembic", "current"],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, f"alembic current failed: {r.stderr}\n{r.stdout}"
    # e.g. "059 (head)"
    m = re.search(r"([0-9a-fA-F]+)\s*\(head\)", r.stdout)
    if not m:
        m = re.search(r"\b([0-9a-fA-F]+)\b", r.stdout)
    assert m, f"no revision in alembic current output: {r.stdout!r}"
    return m.group(1)


def _table_exists(db_dsn: str, table: str) -> bool:
    import psycopg

    with psycopg.connect(db_dsn) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (f"public.{table}",)).fetchone()
        return row is not None and row[0] is not None


def _column_exists(db_dsn: str, table: str, column: str) -> bool:
    import psycopg

    with psycopg.connect(db_dsn) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = %s
               AND column_name = %s
            """,
            (table, column),
        ).fetchone()
        return row is not None


@pytest.fixture
def empty_db():
    admin = _admin_dsn()
    if admin is None:
        pytest.skip("no Postgres DSN in environment")
    if not BOOTSTRAP_SCRIPT.is_file():
        pytest.fail(f"missing bootstrap script: {BOOTSTRAP_SCRIPT}")

    dbname = f"bootstrap_empty_{uuid.uuid4().hex[:10]}"
    if not _try_create_database(admin, dbname):
        pytest.skip("CREATE DATABASE unavailable")
    dsn = _db_url(admin, dbname)
    try:
        yield dsn
    finally:
        _drop_database(admin, dbname)


def test_migration_056_creates_agent_commands_base():
    """Static gate: 056 must CREATE the base table before ALTERing it."""
    src = MIG_056.read_text(encoding="utf-8")
    create_pos = src.find("CREATE TABLE IF NOT EXISTS agent_commands")
    alter_pos = src.find("ALTER TABLE agent_commands ADD COLUMN IF NOT EXISTS command_id")
    assert create_pos != -1, "056 must CREATE agent_commands for from-empty bootstrap"
    assert alter_pos != -1, "056 must still expand agent_commands"
    assert create_pos < alter_pos, "CREATE must precede ALTER in 056"


def test_from_empty_bootstrap_reaches_head(empty_db):
    """Empty DB → shipped bootstrap → Alembic head + required relations."""
    result = _run_bootstrap(empty_db)
    assert result.returncode == 0, (
        f"bootstrap failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "from-empty bootstrap: done" in result.stdout

    rev = _alembic_current(empty_db)
    assert rev == EXPECTED_HEAD, f"expected head {EXPECTED_HEAD}, got {rev!r}"

    for table in REQUIRED_AT_HEAD:
        assert _table_exists(empty_db, table), f"missing table after bootstrap: {table}"

    # 056 expand columns must exist (proves ALTER path ran, not just CREATE).
    for col in ("command_id", "claim_owner", "idempotency_key", "attempt_id"):
        assert _column_exists(empty_db, "agent_commands", col), (
            f"agent_commands.{col} missing after bootstrap"
        )
    # 061 residual ensure columns
    for table, col in (
        ("billing_cycles", "token_cost_cad"),
        ("users", "max_concurrent_instances"),
    ):
        assert _column_exists(empty_db, table, col), f"{table}.{col} missing"


def test_from_empty_bootstrap_is_deterministic(empty_db):
    """Two independent empty databases both reach the same head state.

    Uses the fixture DB plus a second throwaway DB created inside the test.
    """
    admin = _admin_dsn()
    assert admin is not None

    first = _run_bootstrap(empty_db)
    assert first.returncode == 0, first.stderr
    rev1 = _alembic_current(empty_db)

    dbname2 = f"bootstrap_empty_{uuid.uuid4().hex[:10]}"
    assert _try_create_database(admin, dbname2)
    dsn2 = _db_url(admin, dbname2)
    try:
        second = _run_bootstrap(dsn2)
        assert second.returncode == 0, second.stderr
        rev2 = _alembic_current(dsn2)
        assert rev1 == rev2 == EXPECTED_HEAD
        assert _table_exists(dsn2, "agent_commands")
        assert _column_exists(dsn2, "agent_commands", "command_id")
    finally:
        _drop_database(admin, dbname2)
