"""Track B / Phase 8 schema catalog — storage catalog schema invariant tests.

These database-level tests verify that the PostgreSQL storage schema and tables
enforce the required catalog state machine, multi-provider replicas, and lifecycle
deletion queue invariants.
"""

import json
import time
import uuid
import pytest

try:
    from db import _get_pg_pool
    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_storage = _c.execute(
            "SELECT to_regclass('storage.artifacts')"
        ).fetchone()[0] is not None
except Exception as _e:
    pytestmark = pytest.mark.skip(f"no pg pool available: {_e}")
    _pool = None
else:
    if not _has_storage:
        pytestmark = pytest.mark.skip("test database not migrated to >= 064")

from psycopg.errors import (
    CheckViolation,
    ForeignKeyViolation,
    UniqueViolation,
)


@pytest.fixture
def cleanup_ids():
    """Track inserted rows and clean them up after testing."""
    ids = {"jobs": [], "hosts": [], "artifacts": []}
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for aid in ids["artifacts"]:
            conn.execute("DELETE FROM storage.artifacts WHERE artifact_id=%s", (aid,))
        for jid in ids["jobs"]:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (jid,))
        for hid in ids["hosts"]:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (hid,))
        conn.commit()


def _mkjob(cleanup):
    job_id = f"job-storage-{uuid.uuid4().hex[:10]}"
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload,
                                 phase, desired_state, queued_at, updated_at)
               VALUES (%s, 'queued', 0, %s, %s, 'pending', 'running', now(), now())""",
            (job_id, time.time(), json.dumps({"name": job_id})),
        )
        conn.commit()
    cleanup["jobs"].append(job_id)
    return job_id


def _mkattempt(conn, job_id, number=1):
    fence = conn.execute("SELECT nextval('placement_fencing_token_seq')").fetchone()[0]
    row = conn.execute(
        """INSERT INTO job_attempts (job_id, attempt_number, status, fencing_token)
           VALUES (%s, %s, 'reserved', %s) RETURNING attempt_id""",
        (job_id, number, fence),
    ).fetchone()
    return row[0]


def _mkartifact(cleanup, tenant_id="tenant-1", job_id=None, attempt_id=None,
                logical_name="model.safetensors", state="requested",
                provider="b2", bucket="weights-bucket", key="models/model.bin"):
    artifact_id = uuid.uuid4()
    with _pool.connection() as conn:
        conn.execute(
            """INSERT INTO storage.artifacts (
                   artifact_id, tenant_id, job_id, attempt_id, artifact_type,
                   logical_name, state, primary_provider, primary_bucket,
                   object_key, content_type, residency_region, retention_class
               ) VALUES (%s, %s, %s, %s, 'model_weights', %s, %s, %s, %s, %s,
                         'application/octet-stream', 'ca-central-1', 'standard')""",
            (artifact_id, tenant_id, job_id, attempt_id, logical_name, state, provider, bucket, key),
        )
        conn.commit()
    cleanup["artifacts"].append(artifact_id)
    return artifact_id


class TestStorageSchemaPresence:
    def test_storage_tables_exist(self):
        with _pool.connection() as conn:
            for table in (
                "storage.artifacts",
                "storage.artifact_upload_sessions",
                "storage.artifact_replicas",
                "storage.artifact_deletion_jobs",
            ):
                assert conn.execute(
                    "SELECT to_regclass(%s)", (table,)
                ).fetchone()[0] == table, f"missing table {table}"


class TestArtifactInvariants:
    def test_duplicate_primary_object_rejected(self, cleanup_ids):
        _mkartifact(cleanup_ids, key="same/key.bin")
        with pytest.raises(UniqueViolation):
            _mkartifact(cleanup_ids, key="same/key.bin")

    def test_invalid_state_rejected(self, cleanup_ids):
        with pytest.raises(CheckViolation):
            _mkartifact(cleanup_ids, state="invalid_state")

    def test_negative_size_bytes_rejected(self, cleanup_ids):
        art_id = _mkartifact(cleanup_ids)
        with _pool.connection() as conn:
            with pytest.raises(CheckViolation):
                conn.execute(
                    "UPDATE storage.artifacts SET size_bytes = -10 WHERE artifact_id = %s",
                    (art_id,),
                )

    def test_attempt_delete_sets_null(self, cleanup_ids):
        job_id = _mkjob(cleanup_ids)
        with _pool.connection() as conn:
            attempt_id = _mkattempt(conn, job_id)
            conn.commit()
        art_id = _mkartifact(cleanup_ids, job_id=job_id, attempt_id=attempt_id)
        with _pool.connection() as conn:
            conn.execute("DELETE FROM job_attempts WHERE attempt_id = %s", (attempt_id,))
            conn.commit()
            row = conn.execute(
                "SELECT attempt_id FROM storage.artifacts WHERE artifact_id = %s",
                (art_id,),
            ).fetchone()
            assert row[0] is None


class TestArtifactUploadSessions:
    def test_duplicate_session_idempotency_rejected(self, cleanup_ids):
        art_id = _mkartifact(cleanup_ids)
        sess_id1 = uuid.uuid4()
        sess_id2 = uuid.uuid4()
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO storage.artifact_upload_sessions (
                       upload_session_id, artifact_id, tenant_id, principal_id,
                       expires_at, idempotency_key
                   ) VALUES (%s, %s, 'tenant-1', 'user-1', now() + interval '1 hour', 'idem-1')""",
                (sess_id1, art_id),
            )
            with pytest.raises(UniqueViolation):
                conn.execute(
                    """INSERT INTO storage.artifact_upload_sessions (
                           upload_session_id, artifact_id, tenant_id, principal_id,
                           expires_at, idempotency_key
                       ) VALUES (%s, %s, 'tenant-1', 'user-1', now() + interval '1 hour', 'idem-1')""",
                    (sess_id2, art_id),
                )

    def test_cascade_delete_artifact_deletes_session(self, cleanup_ids):
        art_id = _mkartifact(cleanup_ids)
        sess_id = uuid.uuid4()
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO storage.artifact_upload_sessions (
                       upload_session_id, artifact_id, tenant_id, principal_id,
                       expires_at, idempotency_key
                   ) VALUES (%s, %s, 'tenant-1', 'user-1', now() + interval '1 hour', 'idem-2')""",
                (sess_id, art_id),
            )
            conn.execute("DELETE FROM storage.artifacts WHERE artifact_id = %s", (art_id,))
            conn.commit()
            row = conn.execute(
                "SELECT count(*) FROM storage.artifact_upload_sessions WHERE upload_session_id = %s",
                (sess_id,),
            ).fetchone()
            assert row[0] == 0


class TestArtifactReplicas:
    def test_primary_key_uniqueness(self, cleanup_ids):
        art_id = _mkartifact(cleanup_ids)
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO storage.artifact_replicas (
                       artifact_id, provider, bucket, object_key, state
                   ) VALUES (%s, 'r2', 'weights-r2', 'models/model.bin', 'active')""",
                (art_id,),
            )
            with pytest.raises(UniqueViolation):
                conn.execute(
                    """INSERT INTO storage.artifact_replicas (
                           artifact_id, provider, bucket, object_key, state
                       ) VALUES (%s, 'r2', 'weights-r2', 'models/model.bin', 'active')""",
                    (art_id,),
                )

    def test_cascade_delete(self, cleanup_ids):
        art_id = _mkartifact(cleanup_ids)
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO storage.artifact_replicas (
                       artifact_id, provider, bucket, object_key, state
                   ) VALUES (%s, 'r2', 'weights-r2', 'models/model.bin', 'active')""",
                (art_id,),
            )
            conn.execute("DELETE FROM storage.artifacts WHERE artifact_id = %s", (art_id,))
            conn.commit()
            row = conn.execute(
                "SELECT count(*) FROM storage.artifact_replicas WHERE artifact_id = %s",
                (art_id,),
            ).fetchone()
            assert row[0] == 0


class TestArtifactDeletionJobs:
    def test_one_active_deletion_job_per_artifact(self, cleanup_ids):
        art_id = _mkartifact(cleanup_ids)
        del_id1 = uuid.uuid4()
        del_id2 = uuid.uuid4()
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO storage.artifact_deletion_jobs (
                       deletion_id, artifact_id, reason, requested_by, state
                   ) VALUES (%s, %s, 'user request', 'user-1', 'requested')""",
                (del_id1, art_id),
            )
            with pytest.raises(UniqueViolation):
                conn.execute(
                    """INSERT INTO storage.artifact_deletion_jobs (
                           deletion_id, artifact_id, reason, requested_by, state
                       ) VALUES (%s, %s, 'policy request', 'system', 'delete_pending')""",
                    (del_id2, art_id),
                )

    def test_completed_deletion_job_does_not_block_new_active(self, cleanup_ids):
        art_id = _mkartifact(cleanup_ids)
        del_id1 = uuid.uuid4()
        del_id2 = uuid.uuid4()
        with _pool.connection() as conn:
            conn.execute(
                """INSERT INTO storage.artifact_deletion_jobs (
                       deletion_id, artifact_id, reason, requested_by, state
                   ) VALUES (%s, %s, 'user request', 'user-1', 'completed')""",
                (del_id1, art_id),
            )
            conn.execute(
                """INSERT INTO storage.artifact_deletion_jobs (
                       deletion_id, artifact_id, reason, requested_by, state
                   ) VALUES (%s, %s, 'policy request', 'system', 'requested')""",
                (del_id2, art_id),
            )
            conn.commit()
