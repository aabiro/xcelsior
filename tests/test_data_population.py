"""Population audit and corrected writers against real disposable PostgreSQL."""

import json
import uuid

import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from db import DatabaseOps, _get_pg_pool
from control_plane.data_population import audit_population
from control_plane.operational_metrics import heartbeat_once
from control_plane.outbox import append_event, payload_digest
from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


def test_audit_counts_null_blank_json_and_zero_without_record_values():
    pool = _get_pg_pool()
    table = "population_" + uuid.uuid4().hex
    with pool.connection() as conn:
        conn.execute(
            sql.SQL("CREATE TABLE {} (n INTEGER,t TEXT,j JSONB)").format(sql.Identifier(table))
        )
        conn.execute(
            sql.SQL(
                "INSERT INTO {} VALUES (0,'  ','{{}}'),(NULL,NULL,NULL),(4,'never-print-this-record','{{\"private\":true}}')"
            ).format(sql.Identifier(table))
        )
    try:
        with pool.connection() as conn:
            database, port = conn.execute(
                "SELECT current_database(),current_setting('port')::integer"
            ).fetchone()
            conn.rollback()
            report = audit_population(conn, expected_database=database, expected_port=port)
            assert conn.execute("SHOW transaction_read_only").fetchone()[0] == "off"
        assert report["complete"]
        assert "never-print-this-record" not in json.dumps(report)
        entry = next(t for t in report["tables"] if t["table"] == table)
        assert entry["rows"] == 3
        columns = {c["name"]: c for c in entry["columns"]}
        assert columns["n"]["nulls"] == columns["n"]["zero"] == 1
        assert columns["t"]["nulls"] == columns["t"]["blank"] == 1
        assert columns["j"]["nulls"] == columns["j"]["empty_json"] == 1
        with pool.connection() as conn:
            with pytest.raises(ValueError, match="expected target"):
                audit_population(conn, expected_database="wrong-database", expected_port=port)
    finally:
        with pool.connection() as conn:
            conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))


def test_job_host_writers_populate_normalized_identity_and_security_capabilities():
    marker = uuid.uuid4().hex
    with _get_pg_pool().connection() as conn:
        DatabaseOps.upsert_job(
            conn, {"job_id": marker, "owner": "customer", "team_id": "team"}, backend="postgres"
        )
        row = conn.execute(
            "SELECT owner_id,tenant_id,team_id,updated_at FROM jobs WHERE job_id=%s", (marker,)
        ).fetchone()
        assert row[:3] == ("customer", "customer", "team") and row[3] is not None
        DatabaseOps.upsert_job(conn, {"job_id": marker}, backend="postgres")
        assert conn.execute(
            "SELECT owner_id,tenant_id,team_id FROM jobs WHERE job_id=%s", (marker,)
        ).fetchone() == ("customer", "customer", "team")
        DatabaseOps.upsert_host(
            conn,
            {
                "host_id": marker,
                "owner": "provider-user",
                "country": "CA",
                "region": "ca-east",
                "capabilities": {"egress_isolation": True},
            },
            backend="postgres",
        )
        DatabaseOps.upsert_host(conn, {"host_id": marker, "capabilities": None}, backend="postgres")
        assert conn.execute(
            "SELECT owner_id,tenant_id,country,region,capabilities FROM hosts WHERE host_id=%s",
            (marker,),
        ).fetchone() == (
            "provider-user",
            "provider-user",
            "CA",
            "ca-east",
            {"egress_isolation": True},
        )
        conn.rollback()


def test_outbox_writer_persists_verified_envelope_with_native_identity():
    marker = uuid.uuid4().hex
    with _get_pg_pool().connection() as conn:
        DatabaseOps.upsert_job(conn, {"job_id": marker, "owner": "customer"}, backend="postgres")
        payload = {"unicode": "caf\u00e9", "n": 1.5}
        eid = append_event(
            conn,
            aggregate_type="job",
            aggregate_id=marker,
            event_type="job.v1.submitted",
            payload=payload,
            headers={
                "trace_id": "trace",
                "correlation_id": "request",
                "causation_id": "cause",
                "actor_id": "actor",
            },
        )
        row = conn.execute(
            "SELECT tenant_id,occurred_at,classification,payload_sha256,trace_id,correlation_id,causation_id,payload,headers FROM outbox_events WHERE event_id=%s",
            (eid,),
        ).fetchone()
        assert row[0] == "customer" and row[1] is not None and row[2] == "internal"
        assert row[3] == payload_digest(row[7])
        assert row[4:7] == ("trace", "request", "cause") and row[8]["actor_id"] == "actor"
        assert (
            append_event(
                conn,
                aggregate_type="job",
                aggregate_id=marker,
                event_type="job.v1.submitted",
                payload={"changed": True},
            )
            is None
        )
        assert (
            conn.execute("SELECT payload FROM outbox_events WHERE event_id=%s", (eid,)).fetchone()[
                0
            ]
            == payload
        )
        conn.rollback()


def test_runtime_heartbeat_records_actual_database_revision(monkeypatch):
    marker = uuid.uuid4().hex
    monkeypatch.setenv("XCELSIOR_SERVICE_VERSION", "test-source-version")
    try:
        heartbeat_once("scheduler", replica_id=marker)
        with _get_pg_pool().connection() as conn:
            assert conn.execute(
                "SELECT schema_revision,service_version FROM service_heartbeats WHERE replica_id=%s",
                (marker,),
            ).fetchone() == ("128", "test-source-version")
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM service_heartbeats WHERE replica_id=%s", (marker,))


def test_recorded_identity_migration_is_resumable_and_never_guesses(empty_db):
    _alembic(empty_db, "upgrade", "127")
    with _connect(empty_db) as conn:
        for jid, payload in [
            ("recorded", {"owner": "customer", "team_id": "team"}),
            ("unknown", {}),
            ("invalid", {"owner": {"invalid": "object"}}),
        ]:
            conn.execute(
                "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) VALUES(%s,'stopped',0,1,%s)",
                (jid, Jsonb(payload)),
            )
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload,owner_id,tenant_id) VALUES('known','stopped',0,1,'{\"owner\":\"stale\"}','known-owner','known-tenant')"
        )
        conn.execute(
            'INSERT INTO hosts(host_id,status,registered_at,payload) VALUES(\'host\',\'active\',1,\'{"owner":"provider-user","capabilities":{"egress_isolation":true}}\')'
        )
        record = {
            "lease_id": "archive",
            "job_id": "recorded",
            "host_id": "host",
            "status": "released",
        }
        conn.execute(
            "INSERT INTO legacy_lease_archive(lease_id,tenant_id,tenant_attributed,record,record_sha256) VALUES('archive','system:unattributed-legacy-lease-archive',false,%s,encode(sha256(convert_to(%s::jsonb::text,'UTF8')),'hex'))",
            (Jsonb(record), Jsonb(record)),
        )
    _alembic(empty_db, "upgrade", "128")
    with _connect(empty_db) as conn:
        expected = [
            ("invalid", None, None, None),
            ("known", "known-owner", "known-tenant", None),
            ("recorded", "customer", "customer", "team"),
            ("unknown", None, None, None),
        ]
        assert (
            conn.execute(
                "SELECT job_id,owner_id,tenant_id,team_id FROM jobs ORDER BY job_id"
            ).fetchall()
            == expected
        )
        assert conn.execute(
            "SELECT tenant_id,tenant_attributed,record FROM legacy_lease_archive"
        ).fetchone() == ("customer", True, record)
        assert conn.execute("SELECT owner_id,tenant_id,capabilities FROM hosts").fetchone() == (
            "provider-user",
            "provider-user",
            {"egress_isolation": True},
        )
        assert (
            conn.execute("SELECT status FROM jobs WHERE job_id='recorded'").fetchone()[0]
            == "stopped"
        )
    _alembic(empty_db, "downgrade", "127")
    _alembic(empty_db, "upgrade", "128")
    with _connect(empty_db) as conn:
        assert (
            conn.execute(
                "SELECT job_id,owner_id,tenant_id,team_id FROM jobs ORDER BY job_id"
            ).fetchall()
            == expected
        )


def test_envelope_backfill_keeps_original_payload_and_historical_attribution(empty_db):
    _alembic(empty_db, "upgrade", "127")
    from analytics.contracts import register_all

    with _connect(empty_db) as conn:
        register_all(conn)
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload,tenant_id) VALUES('job','stopped',0,1,'{}','current-owner-is-not-historical-proof')"
        )
        eids = []
        for headers in ({"tenant_id": "recorded-tenant", "trace_id": "recorded-trace"}, {}):
            eids.append(
                conn.execute(
                    "INSERT INTO outbox_events(aggregate_type,aggregate_id,event_type,payload,headers,idempotency_key) VALUES('job','job','job.v1.submitted',%s,%s,%s) RETURNING event_id",
                    (Jsonb({"unicode": "caf\u00e9", "n": 1.5}), Jsonb(headers), uuid.uuid4().hex),
                ).fetchone()[0]
            )
        original = conn.execute(
            "SELECT event_id,payload,headers,created_at FROM outbox_events ORDER BY event_id"
        ).fetchall()
    _alembic(empty_db, "upgrade", "128")
    with _connect(empty_db) as conn:
        assert (
            conn.execute(
                "SELECT event_id,payload,headers,created_at FROM outbox_events ORDER BY event_id"
            ).fetchall()
            == original
        )
        for i, eid in enumerate(eids):
            tenant, trace, digest, occurred, created, classification, payload = conn.execute(
                "SELECT tenant_id,trace_id,payload_sha256,occurred_at,created_at,classification,payload FROM outbox_events WHERE event_id=%s",
                (eid,),
            ).fetchone()
            assert tenant == ("recorded-tenant" if i == 0 else None)
            assert trace == ("recorded-trace" if i == 0 else None)
            assert (
                digest == payload_digest(payload)
                and occurred == created
                and classification == "internal"
            )


def test_metadata_backfill_refuses_transitional_trigger_state_drift(empty_db):
    _alembic(empty_db, "upgrade", "127")
    with _connect(empty_db) as conn:
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) VALUES('guarded','stopped',0,1,'{\"owner\":\"recorded-owner\"}')"
        )
        conn.execute("ALTER TABLE jobs DISABLE TRIGGER trg_jobs_control_plane_projection")
        conn.execute("UPDATE jobs SET effective_priority=42 WHERE job_id='guarded'")
        conn.execute("ALTER TABLE jobs ENABLE TRIGGER trg_jobs_control_plane_projection")
    result = _alembic(empty_db, "upgrade", "128", check=False)
    assert result.returncode != 0
    assert "changed unrelated state" in result.stderr + result.stdout
    with _connect(empty_db) as conn:
        assert conn.execute(
            "SELECT owner_id,effective_priority FROM jobs WHERE job_id='guarded'"
        ).fetchone() == (None, 42)
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "127"
