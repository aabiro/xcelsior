"""Real ordered migration with exact retained command evidence."""

from datetime import datetime, timezone

from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


def test_command_time_backfill_matches_the_original_start_authority(empty_db):
    _alembic(empty_db, "upgrade", "139")
    with _connect(empty_db) as conn:
        conn.execute(
            "INSERT INTO hosts(host_id,status,registered_at,payload) VALUES('h','dead',1,'{}')"
        )
        conn.execute(
            "INSERT INTO jobs(job_id,status,phase,desired_state,priority,submitted_at,payload) VALUES('j','failed','failed','running',0,1,'{}')"
        )
        attempt = conn.execute(
            "INSERT INTO job_attempts(job_id,attempt_number,status,host_id,fencing_token,spec_hash) VALUES('j',1,'failed','h',1,'proof') RETURNING attempt_id"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO agent_commands(host_id,command,args,created_at,expires_at,job_id,attempt_id,fencing_token,spec_hash) VALUES('h','start_attempt','{}',1700000000,1700000100,'j',%s,1,'proof')",
            (attempt,),
        )
        conn.execute(
            "INSERT INTO agent_commands(host_id,command,args,created_at,expires_at,job_id,attempt_id,fencing_token,spec_hash) VALUES('h','start_attempt','{}',1600000000,1600000100,'j',%s,2,'proof')",
            (attempt,),
        )
    _alembic(empty_db, "upgrade", "140")
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT command_created_at FROM job_attempts WHERE attempt_id=%s", (attempt,)
        ).fetchone() == (
            datetime.fromtimestamp(1700000000, tz=timezone.utc),
        )
        assert conn.execute(
            "SELECT count(*) FROM pg_trigger WHERE tgname='trg_ln_deposits_project_typed'"
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT count(*) FROM pg_trigger WHERE tgname='trg_ln_deposits_immutable'"
        ).fetchone() == (1,)
    _alembic(empty_db, "downgrade", "139")
    _alembic(empty_db, "upgrade", "140")
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT command_created_at FROM job_attempts WHERE attempt_id=%s", (attempt,)
        ).fetchone() == (
            datetime.fromtimestamp(1700000000, tz=timezone.utc),
        )
