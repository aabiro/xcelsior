"""Real contraction, privilege-boundary and interrupted-unit migration checks."""

import pytest
from psycopg.errors import CheckViolation, NotNullViolation

from db import DatabaseOps
from control_plane.spec_qualification import audit_spec_qualification
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


def ddl(conn):
    functions = conn.execute("""SELECT proname,pg_get_functiondef(p.oid)
        FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
        WHERE n.nspname='public' AND proname IN
          ('control_plane_project_job','control_plane_project_host') ORDER BY proname""").fetchall()
    triggers = conn.execute("""SELECT tgname,pg_get_triggerdef(oid) FROM pg_trigger
        WHERE tgname IN ('trg_jobs_control_plane_projection','trg_hosts_control_plane_projection')
        ORDER BY tgname""").fetchall()
    return functions, triggers


def rows(conn):
    return [
        conn.execute(f"SELECT to_jsonb(r) FROM {table} r ORDER BY {pk}").fetchall()
        for table, pk in [("jobs", "job_id"), ("hosts", "host_id")]
    ]


@pytest.fixture
def before_contract(empty_db):
    _alembic(empty_db, "upgrade", "130")
    with _connect(empty_db) as conn:
        DatabaseOps.upsert_job(
            conn,
            dict(job_id="native", status="queued", priority=7, submitted_at=100),
            backend="postgres",
        )
        DatabaseOps.upsert_host(
            conn, dict(host_id="host", status="active", registered_at=100), backend="postgres"
        )
        original = rows(conn), ddl(conn)
    return empty_db, original


def test_removal_preserves_rows_and_native_fields_own_state(before_contract):
    dsn, (original, _) = before_contract
    _alembic(dsn, "upgrade", "131")
    with _connect(dsn) as conn:
        assert ddl(conn) == ([], [])
        assert rows(conn) == original
        conn.execute(
            "UPDATE jobs SET desired_state='stopped',effective_priority=1000,fair_share_finish=17 WHERE job_id='native'"
        )
        conn.execute(
            "UPDATE jobs SET status='running',priority=99,payload=payload||'{\"observation\":true}' WHERE job_id='native'"
        )
        assert conn.execute(
            "SELECT phase,desired_state,effective_priority,fair_share_finish,EXTRACT(EPOCH FROM queued_at) FROM jobs WHERE job_id='native'"
        ).fetchone() == ("pending", "stopped", 1000, 17, 100)
        for statement, error in [
            ("UPDATE jobs SET phase=NULL WHERE job_id='native'", NotNullViolation),
            ("UPDATE jobs SET phase='invented' WHERE job_id='native'", CheckViolation),
            ("UPDATE jobs SET desired_state='invented' WHERE job_id='native'", CheckViolation),
            ("UPDATE jobs SET queued_at=NULL WHERE job_id='native'", CheckViolation),
        ]:
            with pytest.raises(error):
                with conn.transaction():
                    conn.execute(statement)


def test_default_and_constraint_refuse_unadmitted_hosts(before_contract):
    dsn, _ = before_contract
    _alembic(dsn, "upgrade", "131")
    with _connect(dsn) as conn:
        conn.execute(
            "INSERT INTO hosts(host_id,status,registered_at,payload) VALUES('forged','active',1,'{\"admitted\":true}')"
        )
        assert conn.execute(
            "SELECT admission_state,administrative_state FROM hosts WHERE host_id='forged'"
        ).fetchone() == ("pending", "pending")
        for admin in ["admitted", "draining"]:
            with pytest.raises(CheckViolation):
                with conn.transaction():
                    conn.execute(
                        "UPDATE hosts SET administrative_state=%s WHERE host_id='forged'", (admin,)
                    )
        conn.execute(
            "UPDATE hosts SET admission_state='admitted',administrative_state='admitted' WHERE host_id='forged'"
        )
        with pytest.raises(CheckViolation):
            with conn.transaction():
                conn.execute("UPDATE hosts SET admission_state='revoked' WHERE host_id='forged'")
        conn.execute(
            "UPDATE hosts SET admission_state='revoked',administrative_state='disabled' WHERE host_id='forged'"
        )
        assert conn.execute(
            "SELECT convalidated FROM pg_constraint WHERE conname='ck_hosts_admission_authority'"
        ).fetchone() == (True,)


def test_aggregate_audit_reports_actual_trigger_retirement(before_contract):
    dsn, _ = before_contract
    with _connect(dsn) as conn:
        database, port = conn.execute(
            "SELECT current_database(),current_setting('port')::integer"
        ).fetchone()
        conn.rollback()
        before = audit_spec_qualification(conn, expected_database=database, expected_port=port)
        assert before["trigger_removal_requires_native_writer_gate"]
        assert not before["projection_triggers_retired"]
    _alembic(dsn, "upgrade", "131")
    with _connect(dsn) as conn:
        after = audit_spec_qualification(conn, expected_database=database, expected_port=port)
        assert after["projection_triggers_retired"]
        assert not after["trigger_removal_requires_native_writer_gate"]
        assert after["transitional_triggers"] == []


def test_up_down_up_restores_exact_prior_ddl_without_row_changes(before_contract):
    dsn, (original, original_ddl) = before_contract
    for _ in range(2):
        _alembic(dsn, "upgrade", "131")
        with _connect(dsn) as conn:
            assert rows(conn) == original and ddl(conn) == ([], [])
        _alembic(dsn, "downgrade", "130")
        with _connect(dsn) as conn:
            assert rows(conn) == original and ddl(conn) == original_ddl
            assert conn.execute(
                "SELECT column_default FROM information_schema.columns WHERE table_name='hosts' AND column_name='administrative_state'"
            ).fetchone() == ("'admitted'::text",)


def test_unqualified_host_authority_refuses_unit_without_repair(before_contract):
    dsn, _ = before_contract
    with _connect(dsn) as conn:
        conn.execute("ALTER TABLE hosts DISABLE TRIGGER trg_hosts_control_plane_projection")
        conn.execute("UPDATE hosts SET administrative_state='admitted' WHERE host_id='host'")
        conn.execute("ALTER TABLE hosts ENABLE TRIGGER trg_hosts_control_plane_projection")
        original, original_ddl = rows(conn), ddl(conn)
    result = _alembic(dsn, "upgrade", "131", check=False)
    assert result.returncode and "ck_hosts_admission_authority" in result.stderr
    with _connect(dsn) as conn:
        assert rows(conn) == original and ddl(conn) == original_ddl
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("130",)
        assert conn.execute(
            "SELECT column_default FROM information_schema.columns WHERE table_name='hosts' AND column_name='administrative_state'"
        ).fetchone() == ("'admitted'::text",)
        conn.execute("UPDATE hosts SET administrative_state='pending' WHERE host_id='host'")
    _alembic(dsn, "upgrade", "131")


def test_interrupted_second_unit_is_safe_to_resume(before_contract, monkeypatch):
    dsn, (original, _) = before_contract
    monkeypatch.setenv("XCELSIOR_MIGRATION_LOCK_ATTEMPTS", "1")
    monkeypatch.setenv("XCELSIOR_MIGRATION_LOCK_TIMEOUT", "250ms")
    with _connect(dsn) as blocker:
        blocker.execute("SELECT count(*) FROM jobs")
        result = _alembic(dsn, "upgrade", "131", check=False)
        assert result.returncode and "lock timeout" in result.stderr
        with _connect(dsn) as conn:
            assert rows(conn) == original
            assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("130",)
            functions, triggers = ddl(conn)
            assert [f[0] for f in functions] == ["control_plane_project_job"]
            assert [t[0] for t in triggers] == ["trg_jobs_control_plane_projection"]
            assert conn.execute(
                "SELECT column_default FROM information_schema.columns WHERE table_name='hosts' AND column_name='administrative_state'"
            ).fetchone() == ("'pending'::text",)
    _alembic(dsn, "upgrade", "131")
    with _connect(dsn) as conn:
        assert rows(conn) == original and ddl(conn) == ([], [])


def test_interrupted_downgrade_is_safe_to_resume(before_contract, monkeypatch):
    dsn, (original, original_ddl) = before_contract
    _alembic(dsn, "upgrade", "131")
    monkeypatch.setenv("XCELSIOR_MIGRATION_LOCK_ATTEMPTS", "1")
    monkeypatch.setenv("XCELSIOR_MIGRATION_LOCK_TIMEOUT", "250ms")
    with _connect(dsn) as blocker:
        blocker.execute("SELECT count(*) FROM hosts")
        result = _alembic(dsn, "downgrade", "130", check=False)
        assert result.returncode and "lock timeout" in result.stderr
        with _connect(dsn) as conn:
            assert rows(conn) == original
            assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("131",)
            functions, triggers = ddl(conn)
            assert [f[0] for f in functions] == ["control_plane_project_job"]
            assert [t[0] for t in triggers] == ["trg_jobs_control_plane_projection"]
    _alembic(dsn, "downgrade", "130")
    with _connect(dsn) as conn:
        assert rows(conn) == original and ddl(conn) == original_ddl
