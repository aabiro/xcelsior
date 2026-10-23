"""The retired unfenced redelivery task is archived, never run or resurrected."""

import os

import psycopg
import pytest
from psycopg.errors import CheckViolation, InsufficientPrivilege
from psycopg.types.json import Jsonb

from control_plane import db_roles
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect
from tests.test_db_service_roles import provisioned_roles  # noqa: F401


@pytest.fixture
def before(empty_db):
    _alembic(empty_db, "upgrade", "135")
    with _connect(empty_db) as conn:
        conn.execute(
            """INSERT INTO scheduled_tasks(task_name,interval_seconds,payload)
            VALUES('reconcile_paused_stopped',60,%s),('unrelated',300,'{}')""",
            (Jsonb({"recorded": "preserve", "count": 3}),),
        )
        rows = conn.execute("SELECT task_name,to_jsonb(t) FROM scheduled_tasks t").fetchall()
    return empty_db, dict(rows)


def test_retirement_archives_exact_row_and_roundtrips(before):
    dsn, original = before
    _alembic(dsn, "upgrade", "136")
    with _connect(dsn) as conn:
        assert dict(conn.execute("SELECT task_name,to_jsonb(t) FROM scheduled_tasks t")) == {
            k: v for k, v in original.items() if k != "reconcile_paused_stopped"
        }
        archived = conn.execute(
            "SELECT record,record_sha256 FROM retired_control_task_archive"
        ).fetchone()
        assert archived[0] == original["reconcile_paused_stopped"]
        with pytest.raises(CheckViolation):
            with conn.transaction():
                conn.execute(
                    "INSERT INTO scheduled_tasks(task_name,interval_seconds) VALUES('reconcile_paused_stopped',60)"
                )
    _alembic(dsn, "downgrade", "135")
    with _connect(dsn) as conn:
        assert dict(conn.execute("SELECT task_name,to_jsonb(t) FROM scheduled_tasks t")) == original
    _alembic(dsn, "upgrade", "136")
    with _connect(dsn) as conn:
        assert (
            conn.execute("SELECT record,record_sha256 FROM retired_control_task_archive").fetchone()
            == archived
        )
        assert conn.execute("SELECT count(*) FROM jobs").fetchone() == (0,)
        assert conn.execute("SELECT count(*) FROM agent_commands").fetchone() == (0,)


def test_live_task_claim_refuses_without_partial_archive(before):
    dsn, _ = before
    with _connect(dsn) as conn:
        conn.execute(
            "UPDATE scheduled_tasks SET claim_owner='old-worker',claim_expires_at=clock_timestamp()+interval '1 hour' WHERE task_name='reconcile_paused_stopped'"
        )
    result = _alembic(dsn, "upgrade", "136", check=False)
    assert result.returncode != 0 and "quiesce" in result.stderr
    with _connect(dsn) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("135",)
        assert conn.execute("SELECT to_regclass('retired_control_task_archive')").fetchone() == (
            None,
        )
        conn.execute(
            "UPDATE scheduled_tasks SET claim_owner=NULL,claim_expires_at=NULL WHERE task_name='reconcile_paused_stopped'"
        )
    _alembic(dsn, "upgrade", "136")


def test_unknown_task_change_after_rollback_refuses_archive_overwrite(before):
    dsn, _ = before
    _alembic(dsn, "upgrade", "136")
    _alembic(dsn, "downgrade", "135")
    with _connect(dsn) as conn:
        conn.execute(
            "UPDATE scheduled_tasks SET interval_seconds=61 WHERE task_name='reconcile_paused_stopped'"
        )
    result = _alembic(dsn, "upgrade", "136", check=False)
    assert result.returncode != 0 and "archive mismatch" in result.stderr
    with _connect(dsn) as conn:
        assert conn.execute(
            "SELECT interval_seconds FROM scheduled_tasks WHERE task_name='reconcile_paused_stopped'"
        ).fetchone() == (61,)
        conn.execute(
            "UPDATE scheduled_tasks SET interval_seconds=60 WHERE task_name='reconcile_paused_stopped'"
        )
    _alembic(dsn, "upgrade", "136")


@pytest.mark.parametrize("service", db_roles.runtime_role_keys())
@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE retired_control_task_archive SET record=record",
        "DELETE FROM retired_control_task_archive",
        "INSERT INTO retired_control_task_archive(task_name,record,record_sha256) VALUES('reconcile_paused_stopped','{}','bad')",
    ],
)
def test_runtime_cannot_rewrite_retired_task(provisioned_roles, service, statement):
    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        conn.execute(
            psycopg.sql.SQL("SET ROLE {}").format(
                psycopg.sql.Identifier(db_roles.SERVICE_ROLES[service].name)
            )
        )
        with pytest.raises(InsufficientPrivilege):
            conn.execute(statement)
