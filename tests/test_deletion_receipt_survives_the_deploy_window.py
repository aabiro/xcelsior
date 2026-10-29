"""A 145 writer still stores a deletion receipt after 146 commits.

Deploys are blue-green: `alembic upgrade head` runs while the previous release
keeps serving, and that release inserts receipts without `tenant_id`. 146 makes
the column NOT NULL, so without a default those inserts fail, and the callback
swallows the error and hands Facebook a code whose receipt page never opens.
"""

import uuid

import pytest

from db import _get_pg_pool

try:
    _probe_pool = _get_pg_pool()
    with _probe_pool.connection() as _probe_conn:
        _probe_conn.execute("SELECT 1").fetchone()
except Exception as _probe_error:  # pragma: no cover - environment dependent
    pytestmark = pytest.mark.skip(f"no pg pool available: {_probe_error}")


def _insert_as_145(conn, code_hash, request_id):
    conn.execute(
        """INSERT INTO facebook_deletion_confirmations (code_hash, request_id, outcome)
           VALUES (%s, %s, 'deleted')""",
        (code_hash, request_id),
    )
    return conn.execute(
        "SELECT tenant_id FROM facebook_deletion_confirmations WHERE code_hash = %s",
        (code_hash,),
    ).fetchone()[0]


def test_receipt_without_a_request_takes_its_code_hash_as_tenant():
    code_hash = f"test-receipt-{uuid.uuid4().hex}"
    with _get_pg_pool().connection() as conn:
        try:
            assert _insert_as_145(conn, code_hash, None) == code_hash
        finally:
            conn.rollback()


def test_receipt_for_a_request_takes_the_request_tenant():
    code_hash = f"test-receipt-{uuid.uuid4().hex}"
    tenant = f"tenant-{uuid.uuid4().hex}"
    with _get_pg_pool().connection() as conn:
        try:
            request_id = conn.execute(
                """INSERT INTO privacy_deletion_requests
                       (subject_reference_hash, requested_by, idempotency_key,
                        status_token_hash, deadline_at, tenant_id)
                   VALUES (%s, 'test', %s, %s, clock_timestamp() + interval '30 days', %s)
                   RETURNING request_id""",
                (f"subject-{code_hash}", code_hash, code_hash, tenant),
            ).fetchone()[0]
            assert _insert_as_145(conn, code_hash, request_id) == tenant
        finally:
            conn.rollback()
