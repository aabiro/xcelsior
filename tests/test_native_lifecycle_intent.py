"""Recorded migration evidence and observation writes cannot replace intent."""

import ast

import pytest
from psycopg.errors import CheckViolation
from psycopg.types.json import Jsonb

from control_plane.attempts import report_attempt_status
from tests.test_job_reconcile_observations import placed  # noqa: F401
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


def seed(conn, jid, value):
    conn.execute(
        """INSERT INTO jobs(job_id,status,priority,submitted_at,payload,phase,desired_state)
        VALUES(%s,'stopping',0,100,%s,'running','stopped')""",
        (jid, Jsonb(dict(lifecycle_intent=value))),
    )


def test_backfill_preserves_recorded_values_and_consumed_null_across_rollback(empty_db):
    _alembic(empty_db, "upgrade", "133")
    with _connect(empty_db) as conn:
        for value in ["stop", "cancel", "terminate", "restart", None]:
            seed(conn, str(value), value)
        before = conn.execute("SELECT job_id,payload FROM jobs ORDER BY job_id").fetchall()
    _alembic(empty_db, "upgrade", "134")
    with _connect(empty_db) as conn:
        assert conn.execute("SELECT job_id,payload FROM jobs ORDER BY job_id").fetchall() == before
        assert dict(conn.execute("SELECT job_id,lifecycle_intent FROM jobs").fetchall()) == {
            **{v: v for v in ["stop", "cancel", "terminate", "restart"]},
            "None": None,
        }
        conn.execute("UPDATE jobs SET lifecycle_intent=NULL WHERE job_id='restart'")
        conn.execute("UPDATE jobs SET lifecycle_intent='cancel' WHERE job_id='None'")
        for value in ["invented", "resume", ""]:
            with pytest.raises(CheckViolation):
                with conn.transaction():
                    conn.execute(
                        "UPDATE jobs SET lifecycle_intent=%s WHERE job_id='stop'", (value,)
                    )
    _alembic(empty_db, "downgrade", "133")
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT payload?'lifecycle_intent' FROM jobs WHERE job_id='restart'"
        ).fetchone() == (False,)
        assert conn.execute(
            "SELECT payload->>'lifecycle_intent' FROM jobs WHERE job_id='None'"
        ).fetchone() == ("cancel",)
    _alembic(empty_db, "upgrade", "134")
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT lifecycle_intent FROM jobs WHERE job_id='restart'"
        ).fetchone() == (None,)
        assert conn.execute("SELECT lifecycle_intent FROM jobs WHERE job_id='None'").fetchone() == (
            "cancel",
        )


@pytest.mark.parametrize("value", ["bogus", {}, 1, True])
def test_invalid_recorded_intent_refuses_without_partial_schema(empty_db, value):
    _alembic(empty_db, "upgrade", "133")
    with _connect(empty_db) as conn:
        seed(conn, "bad", value)
    result = _alembic(empty_db, "upgrade", "134", check=False)
    assert result.returncode != 0 and "unqualified recorded lifecycle intent" in result.stderr
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT count(*) FROM information_schema.columns WHERE table_name='jobs' AND column_name='lifecycle_intent'"
        ).fetchone() == (0,)
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("133",)
        conn.execute("UPDATE jobs SET payload=payload-'lifecycle_intent' WHERE job_id='bad'")
    _alembic(empty_db, "upgrade", "134")


def test_payload_cannot_forge_terminal_intent_or_wake_reconciliation(placed):
    with placed["pool"].connection() as conn:
        conn.execute(
            "DELETE FROM reconciliation_queue WHERE resource_type='job' AND resource_id=%s",
            (placed["job_id"],),
        )
        conn.execute(
            "UPDATE jobs SET payload=payload||%s WHERE job_id=%s",
            (Jsonb(dict(lifecycle_intent="terminate")), placed["job_id"]),
        )
        assert conn.execute(
            "SELECT lifecycle_intent FROM jobs WHERE job_id=%s", (placed["job_id"],)
        ).fetchone() == (None,)
        assert conn.execute(
            "SELECT count(*) FROM reconciliation_queue WHERE resource_type='job' AND resource_id=%s",
            (placed["job_id"],),
        ).fetchone() == (0,)
        report_attempt_status(
            conn,
            **{k: placed[k] for k in ["job_id", "attempt_id", "host_id", "fencing_token"]},
            status="stopped",
        )
        assert conn.execute(
            "SELECT status FROM jobs WHERE job_id=%s", (placed["job_id"],)
        ).fetchone() == ("stopped",)


def test_native_intent_change_wakes_queue_atomically(placed):
    with placed["pool"].connection() as conn:
        conn.execute(
            "DELETE FROM reconciliation_queue WHERE resource_type='job' AND resource_id=%s",
            (placed["job_id"],),
        )
        with pytest.raises(RuntimeError):
            with conn.transaction():
                conn.execute(
                    "UPDATE jobs SET lifecycle_intent='cancel' WHERE job_id=%s", (placed["job_id"],)
                )
                assert conn.execute(
                    "SELECT count(*) FROM reconciliation_queue WHERE resource_type='job' AND resource_id=%s",
                    (placed["job_id"],),
                ).fetchone() == (1,)
                raise RuntimeError("rollback")
        assert conn.execute(
            "SELECT lifecycle_intent FROM jobs WHERE job_id=%s", (placed["job_id"],)
        ).fetchone() == (None,)
        assert conn.execute(
            "SELECT count(*) FROM reconciliation_queue WHERE resource_type='job' AND resource_id=%s",
            (placed["job_id"],),
        ).fetchone() == (0,)


def test_runtime_sql_never_reads_intent_from_observation_json():
    from tests._source_tree import iter_source_files

    for path, rel in iter_source_files():
        if not rel.startswith(
            ("control_plane/", "api/routes/", "api/serverless/", "core/", "workers/", "billing/")
        ):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "->>'lifecycle_intent'" not in node.value, rel
