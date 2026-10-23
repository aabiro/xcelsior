"""137 up/down/up preserves prior rows and narrows the host result exception."""

import uuid

import psycopg
import pytest

from tests.test_from_empty_bootstrap import empty_db  # noqa: F401
from tests.test_migration_from_production_snapshot import _alembic, _connect


def test_native_host_result_constraint_roundtrip(empty_db):
    _alembic(empty_db, "upgrade", "136")
    pid = str(uuid.uuid4())
    with _connect(empty_db) as conn:
        conn.execute(
            """INSERT INTO action_plans(plan_id,action_type,principal_id,tenant_id,
            canonical_args,canonical_args_hash,approval_mode,status,expires_at)
            VALUES(%s,'evict_host_workloads','operator','tenant','{}','hash','human','quoted',now()+interval '1 hour')""",
            (pid,),
        )
        original = conn.execute(
            "SELECT to_jsonb(p) FROM action_plans p WHERE plan_id=%s", (pid,)
        ).fetchone()[0]
    for _ in range(2):
        _alembic(empty_db, "upgrade", "137")
        with _connect(empty_db) as conn:
            assert (
                conn.execute(
                    "SELECT to_jsonb(p) FROM action_plans p WHERE plan_id=%s", (pid,)
                ).fetchone()[0]
                == original
            )
            for action, resource in [("create_instance", "host"), ("evict_host_workloads", None)]:
                with pytest.raises(psycopg.errors.CheckViolation):
                    with conn.transaction():
                        conn.execute(
                            """UPDATE action_plans SET action_type=%s,status='succeeded',
                            approved_at=now(),consumed_at=now(),idempotent_response='{}',
                            resulting_resource_id=%s WHERE plan_id=%s""",
                            (action, resource, pid),
                        )
        _alembic(empty_db, "downgrade", "136")
        with _connect(empty_db) as conn:
            assert (
                conn.execute(
                    "SELECT to_jsonb(p) FROM action_plans p WHERE plan_id=%s", (pid,)
                ).fetchone()[0]
                == original
            )
    _alembic(empty_db, "upgrade", "137")
