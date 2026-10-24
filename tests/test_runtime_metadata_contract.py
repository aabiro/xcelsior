"""Real ordered migration with exact retained command evidence."""

from datetime import datetime, timezone
import pytest

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
        assert conn.execute(
            "SELECT count(*) FROM information_schema.columns WHERE table_schema='public' AND ((table_name='agent_commands' AND column_name='result') OR (table_name='host_observations' AND column_name='command_journal_watermark'))"
        ).fetchone() == (0,)
    _alembic(empty_db, "downgrade", "139")
    _alembic(empty_db, "upgrade", "140")
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT command_created_at FROM job_attempts WHERE attempt_id=%s", (attempt,)
        ).fetchone() == (
            datetime.fromtimestamp(1700000000, tz=timezone.utc),
        )


@pytest.mark.parametrize("table", ["agent_commands", "host_observations"])
def test_column_contraction_refuses_to_discard_recorded_data(empty_db, table):
    _alembic(empty_db, "upgrade", "139")
    with _connect(empty_db) as conn:
        if table == "agent_commands":
            conn.execute(
                "INSERT INTO agent_commands(host_id,command,args,result) VALUES('h','health_probe','{}','{}')"
            )
        else:
            conn.execute(
                "INSERT INTO hosts(host_id,status,registered_at,payload) VALUES('h','dead',1,'{}')"
            )
            conn.execute(
                "INSERT INTO host_observations(host_id,session_id,inventory_generation,command_journal_watermark) VALUES('h','session',1,3)"
            )
    result = _alembic(empty_db, "upgrade", "140", check=False)
    assert result.returncode != 0 and "contains evidence" in result.stderr
    with _connect(empty_db) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("139",)
        if table == "agent_commands":
            assert conn.execute(
                "SELECT result FROM agent_commands WHERE host_id='h'"
            ).fetchone() == ({},)
        else:
            assert conn.execute(
                "SELECT command_journal_watermark FROM host_observations WHERE host_id='h'"
            ).fetchone() == (3,)
