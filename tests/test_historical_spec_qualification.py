"""Recorded execution recovery, provenance and refusal gates on real PostgreSQL."""

import json

import pytest
from psycopg.types.json import Jsonb

from control_plane.job_specs import canonical_spec_hash, make_execution_spec
from control_plane.spec_qualification import audit_spec_qualification
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


@pytest.fixture
def historical(empty_db):
    _alembic(empty_db, "upgrade", "128")
    return empty_db


def seed(dsn, *, job_id="recorded", tamper=None):
    spec = {"job_id": job_id, "name": "recorded-\u00e9", "image": "python:3.12", "num_gpus": 1}
    if tamper == "spec_job_id":
        spec["job_id"] = "other-job"
    digest = canonical_spec_hash(spec)
    hid = "host-" + job_id
    with _connect(dsn) as conn:
        conn.execute(
            "INSERT INTO hosts(host_id,status,registered_at,payload) VALUES(%s,'active',1,'{}'),('other-host','active',1,'{}') ON CONFLICT DO NOTHING",
            (hid,),
        )
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload,spec_hash) "
            "VALUES(%s,%s,5,1,%s,%s)",
            (
                job_id,
                "running" if tamper == "nonterminal" else "failed",
                Jsonb(
                    {
                        "job_id": job_id,
                        "image": "changed-current-payload",
                        "private": "never-report",
                    }
                ),
                None if tamper == "missing_hash" else digest,
            ),
        )
        aid = conn.execute(
            "INSERT INTO job_attempts(job_id,attempt_number,status,host_id,fencing_token,job_generation,spec_hash,policy_version) "
            "VALUES(%s,1,'failed',%s,7,%s,%s,'historical-test') RETURNING attempt_id",
            (
                job_id,
                hid,
                2 if tamper == "generation" else 1,
                "wrong" if tamper == "attempt_hash" else digest,
            ),
        ).fetchone()[0]
        args = {
            "job_id": "other-job" if tamper == "args_job_id" else job_id,
            "attempt_id": "other-attempt" if tamper == "args_attempt_id" else str(aid),
            "fencing_token": "7" if tamper == "args_fence" else 7,
            "spec_hash": "wrong" if tamper == "args_hash" else digest,
            "spec": {**spec, "image": "tampered"} if tamper == "spec_bytes" else spec,
        }
        cid = conn.execute(
            "INSERT INTO agent_commands(host_id,command,args,status,job_id,attempt_id,fencing_token,spec_hash,expires_at) "
            "VALUES(%s,'start_attempt',%s,'acknowledged',%s,%s,%s,%s,1000) RETURNING command_id",
            (
                "other-host" if tamper == "host" else hid,
                Jsonb(args),
                job_id,
                aid,
                8 if tamper == "command_fence" else 7,
                "wrong" if tamper == "command_hash" else digest,
            ),
        ).fetchone()[0]
        before = conn.execute(
            "SELECT to_jsonb(j) FROM jobs j WHERE job_id=%s", (job_id,)
        ).fetchone()[0]
    return spec, digest, str(cid), str(aid), before


def test_backfill_recovers_exact_recorded_spec_without_rehashing_payload(historical):
    spec, digest, command, attempt, before = seed(historical)
    with _connect(historical) as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) "
            "VALUES('unknown','stopped',0,1,'{\"image\":\"not-execution-evidence\"}')"
        )
        unknown = conn.execute("SELECT to_jsonb(j) FROM jobs j WHERE job_id='unknown'").fetchone()[
            0
        ]
    _alembic(historical, "upgrade", "129")
    with _connect(historical) as conn:
        recovered = conn.execute(
            "SELECT to_jsonb(j) FROM jobs j WHERE job_id='recorded'"
        ).fetchone()[0]
        assert recovered == {**before, "spec": spec}
        assert recovered["spec_hash"] == digest
        assert (
            conn.execute("SELECT to_jsonb(j) FROM jobs j WHERE job_id='unknown'").fetchone()[0]
            == unknown
        )
        evidence = conn.execute(
            "SELECT outcome,recorded_spec_hash,source_command_id,source_attempt_id,recorded_row_sha256 "
            "FROM job_spec_backfill_evidence WHERE job_id='recorded'"
        ).fetchone()
        assert evidence[:3] == ("recovered_execution", digest, command)
        assert str(evidence[3]) == attempt
        assert evidence[4] == canonical_spec_hash(before).removeprefix("sha256:")
        assert conn.execute(
            "SELECT outcome,reason_code FROM job_spec_backfill_evidence WHERE job_id='unknown'"
        ).fetchone() == ("unavailable", "no_recorded_execution_hash")
    for _ in range(2):
        _alembic(historical, "downgrade", "128")
        _alembic(historical, "upgrade", "129")
    with _connect(historical) as conn:
        assert conn.execute("SELECT count(*) FROM job_spec_backfill_evidence").fetchone()[0] == 2
        assert conn.execute(
            "SELECT spec,spec_hash FROM jobs WHERE job_id='recorded'"
        ).fetchone() == (spec, digest)


@pytest.mark.parametrize(
    "tamper",
    [
        "spec_job_id",
        "spec_bytes",
        "args_job_id",
        "args_attempt_id",
        "args_fence",
        "args_hash",
        "command_hash",
        "command_fence",
        "attempt_hash",
        "host",
        "generation",
        "nonterminal",
        "missing_hash",
    ],
)
def test_incomplete_or_mismatched_evidence_never_changes_job(historical, tamper):
    *_, before = seed(historical, tamper=tamper)
    _alembic(historical, "upgrade", "129")
    with _connect(historical) as conn:
        assert (
            conn.execute("SELECT to_jsonb(j) FROM jobs j WHERE job_id='recorded'").fetchone()[0]
            == before
        )
        assert (
            conn.execute("SELECT outcome FROM job_spec_backfill_evidence").fetchone()[0]
            == "unavailable"
        )


def test_locked_job_is_reported_and_retry_does_not_lose_history(historical):
    seed(historical)
    with _connect(historical) as locked:
        locked.execute("SELECT 1 FROM jobs WHERE job_id='recorded' FOR UPDATE").fetchone()
        result = _alembic(historical, "upgrade", "129", check=False)
        assert result.returncode != 0 and "locked rows" in result.stderr
    _alembic(historical, "upgrade", "129")
    with _connect(historical) as conn:
        assert (
            conn.execute("SELECT spec FROM jobs WHERE job_id='recorded'").fetchone()[0] is not None
        )


def test_batches_commit_independently_and_refuse_trigger_drift(historical):
    *_, before = seed(historical, job_id="z-last")
    with _connect(historical) as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) "
            "SELECT 'a-' || i::text,'stopped',0,1,'{}' FROM generate_series(1,500) AS i"
        )
        conn.execute("ALTER TABLE jobs DISABLE TRIGGER trg_jobs_control_plane_projection")
        conn.execute("UPDATE jobs SET effective_priority=99 WHERE job_id='z-last'")
        conn.execute("ALTER TABLE jobs ENABLE TRIGGER trg_jobs_control_plane_projection")
    failed = _alembic(historical, "upgrade", "129", check=False)
    assert failed.returncode != 0 and "changed unrelated job state" in failed.stderr
    with _connect(historical) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "128"
        assert conn.execute("SELECT count(*) FROM job_spec_backfill_evidence").fetchone()[0] == 500
        assert conn.execute(
            "SELECT spec,effective_priority FROM jobs WHERE job_id='z-last'"
        ).fetchone() == (None, 99)
        conn.execute("UPDATE jobs SET effective_priority=priority WHERE job_id='z-last'")
    _alembic(historical, "upgrade", "129")
    with _connect(historical) as conn:
        assert conn.execute("SELECT count(*) FROM job_spec_backfill_evidence").fetchone()[0] == 501


def test_native_admission_and_readonly_gap_audit_are_preserved(historical):
    spec = make_execution_spec({"name": "private-never-print", "image": "python:3.12"})
    with _connect(historical) as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload,spec,spec_hash) "
            "VALUES('native','queued',0,1,'{}',%s,%s)",
            (Jsonb(spec), canonical_spec_hash(spec)),
        )
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) VALUES('unknown','failed',0,1,'{}')"
        )
        database, port = conn.execute(
            "SELECT current_database(),current_setting('port')::integer"
        ).fetchone()
    _alembic(historical, "upgrade", "129")
    with _connect(historical) as conn:
        report = audit_spec_qualification(conn, expected_database=database, expected_port=port)
        assert report["specs"]["valid_admission"] == 1 and report["specs"]["missing"] == 1
        assert report["backfill_evidence"] == {"unavailable": 1}
        assert not report["all_job_specs_qualified"]
        assert "private-never-print" not in json.dumps(report)
        assert conn.execute("SHOW transaction_read_only").fetchone()[0] == "off"
        conn.rollback()
        with pytest.raises(ValueError, match="unexpected target"):
            audit_spec_qualification(conn, expected_database="wrong", expected_port=port)


def test_hash_alone_does_not_qualify_unproven_history(historical):
    spec = {"name": "old-snapshot", "image": "python:3.12"}
    with _connect(historical) as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload,spec,spec_hash) "
            "VALUES('unproven','failed',0,1,'{}',%s,%s)",
            (Jsonb(spec), canonical_spec_hash(spec)),
        )
        database, port = conn.execute(
            "SELECT current_database(),current_setting('port')::integer"
        ).fetchone()
    _alembic(historical, "upgrade", "129")
    with _connect(historical) as conn:
        report = audit_spec_qualification(conn, expected_database=database, expected_port=port)
        assert (
            report["specs"]["valid_recorded"] == report["specs"]["recorded_without_provenance"] == 1
        )
        assert not report["all_job_specs_qualified"]
