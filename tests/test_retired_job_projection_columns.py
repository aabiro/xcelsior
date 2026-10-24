"""Contraction preserves original column evidence and every linked job row."""

import os

import psycopg
import pytest
from psycopg.errors import InsufficientPrivilege
from psycopg.types.json import Jsonb

from control_plane import db_roles
from db import DatabaseOps
from control_plane.launch.canonicalize import canonicalize
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect
from tests.test_db_service_roles import provisioned_roles  # noqa: F401

COLUMNS = [
    "placement_min_uptime_bps",
    "placement_min_tier",
    "placement_require_verified",
    "placement_max_premium_bps",
]


def column_count(conn):
    return conn.execute(
        "SELECT count(*) FROM information_schema.columns WHERE table_name='jobs' AND column_name=ANY(%s)",
        (COLUMNS,),
    ).fetchone()[0]


@pytest.fixture
def before(empty_db):
    _alembic(empty_db, "upgrade", "134")
    with _connect(empty_db) as conn:
        conn.execute(
            """INSERT INTO jobs(job_id,status,priority,submitted_at,payload,phase,desired_state,
            lifecycle_intent,tenant_id,placement_min_uptime_bps,placement_min_tier,
            placement_require_verified,placement_max_premium_bps)
            VALUES('recorded','stopped',0,100,%s,'stopped','stopped','cancel','tenant',870,'gold',TRUE,115)""",
            (Jsonb(dict(lifecycle_intent="restart", unrelated=dict(evidence="preserved"))),),
        )
        conn.execute(
            """INSERT INTO billing_cycles(cycle_id,job_id,customer_id,host_id,resource_type,
            period_start,period_end,duration_seconds,rate_per_hour,gpu_model,tier,tier_multiplier,
            amount_micros,status,created_at)
            VALUES('linked-cycle','recorded','owner','host','gpu',100,200,100,1,'RTX 4090','free',1,2500000,'billed',200)"""
        )
    return empty_db


def test_archive_drop_and_rollback_replay_preserve_history_and_latest_native_intent(before):
    _alembic(before, "upgrade", "135")
    with _connect(before) as conn:
        assert column_count(conn) == 0
        record, digest = conn.execute(
            "SELECT record,record_sha256 FROM job_retired_projection_archive WHERE job_id='recorded'"
        ).fetchone()
        assert record["columns"] == dict(zip(COLUMNS, [870, "gold", True, 115]))
        assert record["payload_lifecycle_present"] and record["payload_lifecycle"] == "restart"
        assert conn.execute(
            "SELECT payload,lifecycle_intent FROM jobs WHERE job_id='recorded'"
        ).fetchone() == ({"unrelated": {"evidence": "preserved"}}, "cancel")
        assert conn.execute(
            "SELECT amount_micros FROM billing_cycles WHERE job_id='recorded'"
        ).fetchone() == (2500000,)
        job = dict(
            job_id="new-native",
            status="queued",
            submitted_at=200,
            placement_preference=dict(min_uptime_pct=99.5, require_verified=True),
        )
        DatabaseOps.upsert_job(conn, job, backend="postgres", admission_spec=canonicalize(job))
        conn.execute("UPDATE jobs SET lifecycle_intent=NULL WHERE job_id='recorded'")
    _alembic(before, "downgrade", "134")
    with _connect(before) as conn:
        assert column_count(conn) == 4
        assert conn.execute(
            "SELECT " + ",".join(COLUMNS) + " FROM jobs WHERE job_id='recorded'"
        ).fetchone() == (870, "gold", True, 115)
        assert conn.execute(
            "SELECT placement_min_uptime_bps,placement_require_verified FROM jobs WHERE job_id='new-native'"
        ).fetchone() == (9950, True)
        assert conn.execute(
            "SELECT payload?'lifecycle_intent',lifecycle_intent FROM jobs WHERE job_id='recorded'"
        ).fetchone() == (False, None)
    _alembic(before, "upgrade", "135")
    with _connect(before) as conn:
        assert column_count(conn) == 0
        assert conn.execute(
            "SELECT record,record_sha256 FROM job_retired_projection_archive WHERE job_id='recorded'"
        ).fetchone() == (record, digest)
        assert conn.execute("SELECT sum(amount_micros) FROM billing_cycles").fetchone() == (
            2500000,
        )


def test_unknown_view_dependency_refuses_the_entire_contract(before):
    with _connect(before) as conn:
        conn.execute("CREATE VIEW unknown_consumer AS SELECT placement_min_tier FROM jobs")
    result = _alembic(before, "upgrade", "135", check=False)
    assert result.returncode != 0 and "depend" in result.stderr
    with _connect(before) as conn:
        assert column_count(conn) == 4
        assert conn.execute("SELECT to_regclass('job_retired_projection_archive')").fetchone() == (
            None,
        )
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("134",)
        conn.execute("DROP VIEW unknown_consumer")
    _alembic(before, "upgrade", "135")


def test_unreviewed_column_type_refuses_before_archiving(before):
    with _connect(before) as conn:
        conn.execute("ALTER TABLE jobs ALTER COLUMN placement_min_uptime_bps TYPE BIGINT")
    result = _alembic(before, "upgrade", "135", check=False)
    assert result.returncode != 0 and "unexpected placement column types" in result.stderr
    with _connect(before) as conn:
        assert column_count(conn) == 4
        assert conn.execute("SELECT to_regclass('job_retired_projection_archive')").fetchone() == (
            None,
        )


def test_archive_mismatch_refuses_then_explicit_correction_allows_retry(before):
    _alembic(before, "upgrade", "135")
    _alembic(before, "downgrade", "134")
    with _connect(before) as conn:
        conn.execute("UPDATE jobs SET placement_min_uptime_bps=871 WHERE job_id='recorded'")
    result = _alembic(before, "upgrade", "135", check=False)
    assert result.returncode != 0 and "archive mismatch" in result.stderr
    with _connect(before) as conn:
        assert column_count(conn) == 4
        conn.execute("UPDATE jobs SET placement_min_uptime_bps=870 WHERE job_id='recorded'")
    _alembic(before, "upgrade", "135")


@pytest.mark.parametrize("service", db_roles.runtime_role_keys())
@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE job_retired_projection_archive SET record=record",
        "DELETE FROM job_retired_projection_archive",
        "INSERT INTO job_retired_projection_archive(job_id,record,record_sha256) VALUES('forbidden','{}','bad')",
    ],
)
def test_every_runtime_role_cannot_rewrite_archive(provisioned_roles, service, statement):
    with psycopg.connect(os.environ["XCELSIOR_POSTGRES_DSN"]) as conn:
        conn.execute(
            psycopg.sql.SQL("SET ROLE {}").format(
                psycopg.sql.Identifier(db_roles.SERVICE_ROLES[service].name)
            )
        )
        with pytest.raises(InsufficientPrivilege):
            conn.execute(statement)
