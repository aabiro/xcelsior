"""Column contraction and narrow legacy bridge against real PostgreSQL."""

import pytest
from psycopg.errors import CheckViolation, NotNullViolation
from psycopg.types.json import Jsonb

from db import DatabaseOps
from control_plane.scheduler.claim import claim_next_job, release_claim
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


@pytest.fixture
def projected(empty_db):
    _alembic(empty_db, "upgrade", "129")
    with _connect(empty_db) as conn:
        for jid in ["canonical-intent", "claim-target"]:
            payload = dict(
                job_id=jid, name=jid, status="queued", priority=2, submitted_at=100, gpu_model=jid
            )
            conn.execute(
                "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) VALUES(%s,'queued',2,100,%s)",
                (jid, Jsonb(payload)),
            )
        conn.execute("ALTER TABLE jobs DISABLE TRIGGER trg_jobs_control_plane_projection")
        conn.execute("UPDATE jobs SET effective_priority=1000,fair_share_finish=9")
        conn.execute("UPDATE jobs SET desired_state='stopped' WHERE job_id='canonical-intent'")
        conn.execute("ALTER TABLE jobs ENABLE TRIGGER trg_jobs_control_plane_projection")
    return empty_db


def test_claims_and_metadata_writes_preserve_priority_and_native_intent(projected):
    _alembic(projected, "upgrade", "130")
    with _connect(projected) as conn:
        conn.execute(
            "UPDATE jobs SET payload=payload || '{\"observation\":true}',version=version+1"
        )
        job = DatabaseOps.get_job(conn, "canonical-intent", backend="postgres")
        DatabaseOps.upsert_job(conn, job, backend="postgres")
        assert conn.execute(
            "SELECT desired_state,effective_priority FROM jobs WHERE job_id='canonical-intent'"
        ).fetchone() == ("stopped", 1000)
        claimed = claim_next_job(
            conn, replica_id="contract-test", scope_gpu_models=["claim-target"]
        )
        assert claimed and claimed.effective_priority == 1000
        assert release_claim(
            conn, claimed.job_id, claimed.claim_token, reason_code="contract-proof"
        )
        assert conn.execute(
            "SELECT effective_priority,fair_share_finish,EXTRACT(EPOCH FROM queued_at) FROM jobs WHERE job_id='claim-target'"
        ).fetchone() == (1000, 9, 100)
        conn.execute("UPDATE jobs SET status='running' WHERE job_id='claim-target'")
        assert conn.execute(
            "SELECT phase,desired_state,effective_priority FROM jobs WHERE job_id='claim-target'"
        ).fetchone() == ("running", "running", 1000)
        conn.execute("UPDATE jobs SET status='queued',submitted_at=200 WHERE job_id='claim-target'")
        assert conn.execute(
            "SELECT phase,EXTRACT(EPOCH FROM queued_at),effective_priority FROM jobs WHERE job_id='claim-target'"
        ).fetchone() == ("pending", 200, 1000)
        conn.execute("UPDATE jobs SET priority=3 WHERE job_id='claim-target'")
        assert (
            conn.execute(
                "SELECT effective_priority FROM jobs WHERE job_id='claim-target'"
            ).fetchone()[0]
            == 3
        )


@pytest.mark.parametrize(
    "statement,error",
    [
        ("UPDATE jobs SET phase=NULL WHERE job_id='claim-target'", NotNullViolation),
        ("UPDATE jobs SET desired_state=NULL WHERE job_id='claim-target'", NotNullViolation),
        ("UPDATE jobs SET status='unknown-status' WHERE job_id='claim-target'", NotNullViolation),
        ("UPDATE jobs SET queued_at=NULL WHERE job_id='claim-target'", CheckViolation),
        (
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) VALUES('missing-time','queued',0,NULL,'{}')",
            NotNullViolation,
        ),
    ],
)
def test_invalid_core_projections_fail_at_the_database(projected, statement, error):
    _alembic(projected, "upgrade", "130")
    with _connect(projected) as conn:
        with pytest.raises(error):
            with conn.transaction():
                conn.execute(statement)
        assert conn.execute(
            "SELECT phase,desired_state FROM jobs WHERE job_id='claim-target'"
        ).fetchone() == ("pending", "running")


def test_missing_projection_recovers_only_recorded_status_and_submission(projected):
    with _connect(projected) as conn:
        conn.execute("ALTER TABLE jobs DISABLE TRIGGER trg_jobs_control_plane_projection")
        conn.execute(
            "UPDATE jobs SET phase=NULL,desired_state=NULL,queued_at=NULL WHERE job_id='claim-target'"
        )
        conn.execute("ALTER TABLE jobs ENABLE TRIGGER trg_jobs_control_plane_projection")
        before = conn.execute(
            "SELECT to_jsonb(j) FROM jobs j WHERE job_id='claim-target'"
        ).fetchone()[0]
    _alembic(projected, "upgrade", "130")
    with _connect(projected) as conn:
        after = conn.execute(
            "SELECT to_jsonb(j) FROM jobs j WHERE job_id='claim-target'"
        ).fetchone()[0]
        assert {
            k: v for k, v in before.items() if k not in {"phase", "desired_state", "queued_at"}
        } == {k: v for k, v in after.items() if k not in {"phase", "desired_state", "queued_at"}}
        assert conn.execute(
            "SELECT phase,desired_state,EXTRACT(EPOCH FROM queued_at) FROM jobs WHERE job_id='claim-target'"
        ).fetchone() == ("pending", "running", 100)
        assert dict(
            conn.execute(
                "SELECT column_name,is_nullable FROM information_schema.columns WHERE table_name='jobs' AND column_name IN ('phase','desired_state')"
            ).fetchall()
        ) == {"phase": "NO", "desired_state": "NO"}


def test_unqualified_projection_refuses_the_whole_table_unit(projected):
    with _connect(projected) as conn:
        conn.execute("ALTER TABLE jobs DISABLE TRIGGER trg_jobs_control_plane_projection")
        conn.execute(
            "UPDATE jobs SET status='unknown-history',phase=NULL,desired_state=NULL WHERE job_id='claim-target'"
        )
        conn.execute("ALTER TABLE jobs ENABLE TRIGGER trg_jobs_control_plane_projection")
        before = conn.execute("SELECT to_jsonb(j) FROM jobs j ORDER BY job_id").fetchall()
        trigger = conn.execute(
            "SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgname='trg_jobs_control_plane_projection'"
        ).fetchone()[0]
    result = _alembic(projected, "upgrade", "130", check=False)
    assert (
        result.returncode != 0
        and "requires recorded status and submission evidence" in result.stderr
    )
    with _connect(projected) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "129"
        assert conn.execute("SELECT to_jsonb(j) FROM jobs j ORDER BY job_id").fetchall() == before
        assert (
            conn.execute(
                "SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgname='trg_jobs_control_plane_projection'"
            ).fetchone()[0]
            == trigger
        )


def test_contract_roundtrip_retains_data_and_evidence(projected):
    for _ in range(2):
        _alembic(projected, "upgrade", "130")
        with _connect(projected) as conn:
            before = conn.execute("SELECT to_jsonb(j) FROM jobs j ORDER BY job_id").fetchall()
        _alembic(projected, "downgrade", "129")
        with _connect(projected) as conn:
            assert (
                conn.execute("SELECT to_jsonb(j) FROM jobs j ORDER BY job_id").fetchall() == before
            )
            assert (
                conn.execute(
                    "SELECT is_nullable FROM information_schema.columns WHERE table_name='jobs' AND column_name='phase'"
                ).fetchone()[0]
                == "YES"
            )
            assert conn.execute("SELECT to_regclass('job_spec_backfill_evidence')").fetchone()[0]
