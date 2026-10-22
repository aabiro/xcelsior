"""Lossless row/column retirement, including an explicit production export."""

import json
import os
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


@pytest.fixture
def populated(empty_db):
    _alembic(empty_db, "upgrade", "125")
    snapshot = os.environ.get("XCELSIOR_TEST_LEASE_SNAPSHOT")
    exported = json.loads(Path(snapshot).read_text()) if snapshot else None
    records = (
        exported["records"]
        if snapshot
        else [
            {
                "lease_id": "lease-closed",
                "job_id": "historical-job",
                "host_id": "old-host",
                "granted_at": 1.25,
                "expires_at": 301.25,
                "last_renewed": 1.25,
                "duration_sec": 300,
                "status": "expired",
            },
            {
                "lease_id": "lease-unattributed",
                "job_id": "unattributed-job",
                "host_id": "old-host",
                "granted_at": 1.25,
                "expires_at": 301.25,
                "last_renewed": 1.25,
                "duration_sec": 300,
                "status": "released",
            },
        ]
    )
    tenants = exported["job_tenants"] if exported else {"historical-job": "historical-tenant"}
    with _connect(empty_db) as conn:
        for job_id, tenant_id in tenants.items():
            conn.execute(
                "INSERT INTO jobs(job_id,status,tenant_id,priority,submitted_at,payload) VALUES(%s,'stopped',%s,0,1,'{}')",
                (job_id, tenant_id),
            )
        for row in records:
            conn.execute(
                "INSERT INTO leases SELECT (jsonb_populate_record(NULL::leases,%s)).*",
                (Jsonb(row),),
            )
        conn.execute(
            "INSERT INTO jobs(job_id,status,priority,submitted_at,payload) VALUES('protected-job','stopped',0,1,'{\"keep\":true}')"
        )
    return empty_db, records, tenants


def test_retirement_up_down_up_preserves_every_row_and_protected_job(populated):
    dsn, records, tenants = populated
    from control_plane.db_roles import domain_of

    _alembic(dsn, "upgrade", "126")
    assert domain_of("public.leases") == "control"
    assert domain_of("public.serverless_attempt_credentials") == "control"
    with _connect(dsn) as conn:
        protected = conn.execute(
            "SELECT to_jsonb(j) FROM jobs j WHERE job_id='protected-job'"
        ).fetchone()[0]
    for _ in range(2):
        _alembic(dsn, "upgrade", "127")
        with _connect(dsn) as conn:
            assert conn.execute("SELECT to_regclass('public.leases')").fetchone()[0] is None
            archived = [
                r[0]
                for r in conn.execute("SELECT record FROM legacy_lease_archive ORDER BY lease_id")
            ]
            assert archived == sorted(records, key=lambda r: r["lease_id"])
            attribution = conn.execute(
                "SELECT record->>'job_id',tenant_id,tenant_attributed FROM legacy_lease_archive"
            ).fetchall()
            for job_id, tenant_id, attributed in attribution:
                assert attributed == (job_id in tenants)
                assert tenant_id == tenants.get(job_id, "system:unattributed-legacy-lease-archive")
            assert (
                conn.execute(
                    "SELECT to_jsonb(j) FROM jobs j WHERE job_id='protected-job'"
                ).fetchone()[0]
                == protected
            )
        _alembic(dsn, "downgrade", "125")
        with _connect(dsn) as conn:
            restored = [
                r[0] for r in conn.execute("SELECT to_jsonb(l) FROM leases l ORDER BY lease_id")
            ]
            assert restored == sorted(records, key=lambda r: r["lease_id"])
    _alembic(dsn, "upgrade", "127")


@pytest.mark.parametrize("blocker", ["active", "unknown", "extra_column", "dependency"])
def test_retirement_refuses_unreviewed_state_without_partial_archive(populated, blocker):
    dsn, records, _ = populated
    _alembic(dsn, "upgrade", "126")
    with _connect(dsn) as conn:
        if blocker in ("active", "unknown"):
            conn.execute(
                "UPDATE leases SET status=%s WHERE lease_id=%s", (blocker, records[0]["lease_id"])
            )
        elif blocker == "extra_column":
            conn.execute("ALTER TABLE leases ADD COLUMN unreviewed TEXT")
        else:
            conn.execute("CREATE VIEW lease_dependency AS SELECT lease_id FROM leases")
    result = _alembic(dsn, "upgrade", "127", check=False)
    assert result.returncode != 0
    with _connect(dsn) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "126"
        assert conn.execute("SELECT count(*) FROM leases").fetchone()[0] == len(records)
        assert (
            conn.execute("SELECT to_regclass('public.legacy_lease_archive')").fetchone()[0] is None
        )
