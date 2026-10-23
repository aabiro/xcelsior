"""Historical retention is explicit, lossless and never execution authority."""

import importlib.util
import json
from pathlib import Path

from importlib import import_module
import json

import pytest
from sqlalchemy import create_engine
from psycopg.errors import CheckViolation
from psycopg.types.json import Jsonb

from control_plane.job_specs import canonical_spec_hash, make_execution_spec
from control_plane.lifecycle import request_fresh_attempt_resume
from control_plane.spec_qualification import audit_spec_qualification
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_historical_spec_qualification import seed
from tests.test_migration_from_production_snapshot import _alembic, _connect


@pytest.fixture
def historical(empty_db):
    _alembic(empty_db, "upgrade", "128")
    seed(empty_db)
    with _connect(empty_db) as conn:
        conn.execute("""INSERT INTO jobs(job_id,status,priority,submitted_at,payload)
            VALUES('unknown','stopped',0,1,'{"image":"not-approved","private":"retained"}')""")
        spec = make_execution_spec({"name": "native", "image": "python:3.12"})
        conn.execute(
            """INSERT INTO jobs(job_id,status,priority,submitted_at,payload,spec,spec_hash)
            VALUES('native','queued',0,1,'{}',%s,%s)""",
            (Jsonb(spec), canonical_spec_hash(spec)),
        )
        conn.execute("""INSERT INTO billing_cycles(cycle_id,customer_id,job_id,
            period_start,period_end,duration_seconds,rate_per_hour,amount_micros,created_at)
            VALUES('history-cycle','customer','recorded',1,3601,3600,2.5,2500000,3601)""")
    _alembic(empty_db, "upgrade", "131")
    return empty_db


def snapshot(conn):
    return {
        table: conn.execute(
            f"SELECT to_jsonb(r) FROM {table} r ORDER BY to_jsonb(r)::text"
        ).fetchall()
        for table in (
            "jobs",
            "billing_cycles",
            "job_attempts",
            "agent_commands",
            "job_spec_backfill_evidence",
        )
    }


def test_disposition_preserves_all_history_money_and_native_admission(historical):
    with _connect(historical) as conn:
        before = snapshot(conn)
    _alembic(historical, "upgrade", "132")
    with _connect(historical) as conn:
        after = snapshot(conn)
        assert {k: v for k, v in after.items() if k != "jobs"} == {
            k: v for k, v in before.items() if k != "jobs"
        }
        prior = {row[0]["job_id"]: row[0] for row in before["jobs"]}
        current = {row[0]["job_id"]: row[0] for row in after["jobs"]}
        for jid, disposition in [
            ("recorded", "retained_recovered"),
            ("unknown", "retained_unavailable"),
            ("native", None),
        ]:
            assert current[jid] == {**prior[jid], "history_disposition": disposition}
        receipts = conn.execute("""SELECT job_id,disposition,identity_evidence,
            recorded_row_sha256,recorded_payload_sha256
            FROM job_history_dispositions ORDER BY job_id""").fetchall()
        assert len(receipts) == 2
        for jid, disposition, identity, row_hash, payload_hash in receipts:
            assert disposition == current[jid]["history_disposition"]
            assert identity == "unavailable"
            assert row_hash == canonical_spec_hash(
                {**prior[jid], "history_disposition": None}
            ).removeprefix("sha256:")
            assert payload_hash == canonical_spec_hash(prior[jid]["payload"]).removeprefix(
                "sha256:"
            )
        assert conn.execute("SELECT amount_micros FROM billing_cycles").fetchone() == (2_500_000,)
        for jid in ("recorded", "unknown"):
            with pytest.raises(CheckViolation), conn.transaction():
                conn.execute(
                    "UPDATE jobs SET phase='pending',queued_at=now() WHERE job_id=%s", (jid,)
                )
        database, port = conn.execute(
            "SELECT current_database(),current_setting('port')::int"
        ).fetchone()
        conn.commit()
        audit = audit_spec_qualification(conn, expected_database=database, expected_port=port)
        assert audit["historical_disposition"] == dict(
            recorded={"retained_recovered": 1, "retained_unavailable": 1},
            undisposed=0,
            terminal_exclusion_validated=True,
            complete=True,
        )
        assert audit["all_job_specs_qualified"] is False
        assert "not-approved" not in json.dumps(audit)
        assert "private" not in json.dumps(audit)


def test_downgrade_and_replay_keep_disposition_receipts_unchanged(historical):
    _alembic(historical, "upgrade", "132")
    with _connect(historical) as conn:
        before = snapshot(conn)
        receipts = conn.execute(
            "SELECT to_jsonb(e) FROM job_history_dispositions e ORDER BY job_id"
        ).fetchall()
    for _ in range(2):
        _alembic(historical, "downgrade", "131")
        with _connect(historical) as conn:
            with pytest.raises(CheckViolation), conn.transaction():
                conn.execute(
                    "UPDATE jobs SET phase='pending',queued_at=now() WHERE job_id='unknown'"
                )
        _alembic(historical, "upgrade", "132")
    with _connect(historical) as conn:
        assert snapshot(conn) == before
        assert (
            conn.execute(
                "SELECT to_jsonb(e) FROM job_history_dispositions e ORDER BY job_id"
            ).fetchall()
            == receipts
        )


def test_disposition_refuses_unqualified_live_legacy_jobs(historical):
    with _connect(historical) as conn:
        conn.execute(
            """INSERT INTO jobs(job_id,status,priority,submitted_at,payload,
            phase,desired_state,queued_at) VALUES('a-live','queued',0,1,'{}','pending','running',now())"""
        )
        # This is recorded unavailability, never permission to dispose live work.
        conn.execute(
            """INSERT INTO job_spec_backfill_evidence(job_id,outcome,reason_code,
            recorded_row_sha256,recorded_payload_sha256)
            VALUES('a-live','unavailable','nonterminal_requires_qualification',%s,%s)""",
            ("a" * 64, "b" * 64),
        )
        before = snapshot(conn)
    result = _alembic(historical, "upgrade", "132", check=False)
    assert result.returncode != 0 and "terminal, unowned jobs" in result.stderr
    with _connect(historical) as conn:
        # Expansion may commit; no historical state is silently repaired.
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("131",)
        assert conn.execute("SELECT count(*) FROM job_history_dispositions").fetchone() == (0,)
        after = snapshot(conn)
        for row in after["jobs"]:
            row[0].pop("history_disposition")
        assert after == before


def test_locked_history_refuses_and_retry_completes(historical):
    # Simulate the independently committed expansion before a later batch is
    # interrupted by row contention. Retry must not retake unnecessary DDL locks.
    migration = import_module("migrations.versions.132_dispose_historical_jobs")
    engine = create_engine(historical.replace("postgresql:", "postgresql+psycopg:", 1))
    try:
        with engine.begin() as expanded:
            migration._expand(expanded)
    finally:
        engine.dispose()
    with _connect(historical) as conn:
        conn.execute(
            "SELECT 1 FROM jobs WHERE job_id IN ('recorded','unknown') FOR UPDATE"
        ).fetchall()
        result = _alembic(historical, "upgrade", "132", check=False)
        assert result.returncode != 0 and "locked rows" in result.stderr
    _alembic(historical, "upgrade", "132")
    with _connect(historical) as conn:
        assert conn.execute("SELECT count(*) FROM job_history_dispositions").fetchone() == (2,)


def test_handoff_batches_keep_completed_history_protected_after_later_refusal(historical):
    with _connect(historical) as conn:
        conn.execute(
            """INSERT INTO jobs(job_id,status,priority,submitted_at,payload,phase,desired_state)
            SELECT 'a-batch-'||lpad(i::text,4,'0'),'failed',0,1,'{}','failed','running'
            FROM generate_series(1,500) AS i"""
        )
        conn.execute("""UPDATE jobs SET active_attempt_id=(SELECT attempt_id FROM job_attempts
            WHERE job_id='recorded') WHERE job_id='recorded'""")
    refused = _alembic(historical, "upgrade", "132", check=False)
    assert refused.returncode != 0 and "terminal, unowned jobs" in refused.stderr
    with _connect(historical) as conn:
        assert conn.execute("SELECT count(*) FROM job_history_dispositions").fetchone() == (500,)
        receipts = conn.execute(
            "SELECT to_jsonb(h) FROM job_history_dispositions h ORDER BY job_id"
        ).fetchall()
        assert conn.execute(
            "SELECT convalidated FROM pg_constraint WHERE conname='ck_jobs_history_disposition'"
        ).fetchone() == (False,)
        with pytest.raises(CheckViolation), conn.transaction():
            conn.execute(
                "UPDATE jobs SET phase='pending',queued_at=now() WHERE job_id='a-batch-0001'"
            )
        # Model a separate, explicit handoff after the refusal. Migration never
        # clears this pointer or interprets terminal status as permission to do so.
        conn.execute("UPDATE jobs SET active_attempt_id=NULL WHERE job_id='recorded'")
    _alembic(historical, "upgrade", "132")
    with _connect(historical) as conn:
        assert conn.execute("SELECT count(*) FROM job_history_dispositions").fetchone() == (502,)
        assert (
            conn.execute(
                "SELECT to_jsonb(h) FROM job_history_dispositions h WHERE job_id LIKE 'a-batch-%' ORDER BY job_id"
            ).fetchall()
            == receipts
        )
        assert conn.execute(
            "SELECT outcome,count(*) FROM job_spec_backfill_evidence GROUP BY outcome ORDER BY outcome"
        ).fetchall() == [("recovered_execution", 1), ("unavailable", 501)]


def test_historical_resume_returns_actionable_refusal(monkeypatch):
    class Connection:
        def execute(self, query, params):
            if "job_attempts" not in query:
                raise AssertionError("historical resume attempted a mutation")
            return self

        def fetchone(self):
            return (
                "historical",
                "stopped",
                None,
                None,
                None,
                True,
                "retained_recovered",
                None,
                None,
            )

    monkeypatch.setattr("abuse_policy.job_blocked", lambda *args: False)
    monkeypatch.setattr(
        "control_plane.lifecycle.run_transaction", lambda fn, **kwargs: fn(Connection())
    )
    result = request_fresh_attempt_resume(job_id="historical", created_by="customer")
    assert result.ok is False and result.reason == "historical_job_requires_new_admission"
