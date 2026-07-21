import os
import time
import uuid
import pytest
from artifacts import get_artifact_manager, ArtifactType, ResidencyPolicy
from control_plane.db import control_plane_transaction

pytestmark = pytest.mark.anyio


@pytest.fixture
def test_setup():
    """Seed tenant and job in DB."""
    tenant_id = "tenant-jan-1"
    job_id = "job-jan-1"

    with control_plane_transaction() as conn:
        conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))

        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, tenant_id)
               VALUES (%s, 'submitted', 1, %s, '{}', %s)""",
            (job_id, time.time(), tenant_id),
        )

    yield {"tenant_id": tenant_id, "job_id": job_id}

    with control_plane_transaction() as conn:
        conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def test_janitor_cleans_expired_upload_session(test_setup):
    job_id = test_setup["job_id"]
    mgr = get_artifact_manager()

    # Request upload
    req = mgr.request_upload(
        artifact_type="job_output",
        job_id=job_id,
        filename="temp_data.bin",
    )
    art_id = req["artifact_id"]
    sess_id = req["upload_session_id"]

    # Manually backdate the session's expires_at to simulate expiration
    with control_plane_transaction() as conn:
        conn.execute(
            "UPDATE storage.artifact_upload_sessions SET expires_at = clock_timestamp() - INTERVAL '1 hour' WHERE upload_session_id = %s",
            (uuid.UUID(sess_id),),
        )

    # Trigger cleanup
    mgr.cleanup_expired()

    # Assert state is abandoned
    with control_plane_transaction() as conn:
        row = conn.execute(
            "SELECT state FROM storage.artifacts WHERE artifact_id = %s",
            (uuid.UUID(art_id),),
        ).fetchone()
        assert row is not None
        assert row[0] == "abandoned"

        sess = conn.execute(
            "SELECT completed_at FROM storage.artifact_upload_sessions WHERE upload_session_id = %s",
            (uuid.UUID(sess_id),),
        ).fetchone()
        assert sess is not None
        assert sess[0] is not None


def test_janitor_processes_artifact_deletion_job(test_setup):
    job_id = test_setup["job_id"]
    tenant_id = test_setup["tenant_id"]
    mgr = get_artifact_manager()

    # 1. Create available artifact and replica
    art_id = uuid.uuid4()
    sess_id = uuid.uuid4()
    del_id = uuid.uuid4()
    key = f"job_output/{job_id}/{art_id}_to_delete.bin"

    with control_plane_transaction() as conn:
        conn.execute(
            """INSERT INTO storage.artifacts (
                artifact_id, tenant_id, job_id, artifact_type, logical_name,
                state, primary_provider, primary_bucket, object_key, content_type,
                size_bytes, residency_region, retention_class, created_at, available_at
               ) VALUES (%s, %s, %s, 'job_output', 'to_delete.bin', 'available', 'local', 'local-bucket', %s, 'binary', 10, 'canada_only', 'standard', clock_timestamp(), clock_timestamp())""",
            (art_id, tenant_id, job_id, key),
        )
        conn.execute(
            """INSERT INTO storage.artifact_replicas (artifact_id, provider, bucket, object_key, state)
               VALUES (%s, 'local', 'local-bucket', %s, 'active')""",
            (art_id, key),
        )

    # Seed mock local file to delete
    local_path = os.path.join(mgr.primary._local_dir(), key)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(b"1234567890")

    assert os.path.exists(local_path)

    # 2. Insert deletion job
    with control_plane_transaction() as conn:
        conn.execute(
            """INSERT INTO storage.artifact_deletion_jobs (deletion_id, artifact_id, reason, requested_by, state, next_attempt_at)
               VALUES (%s, %s, 'user_request', 'admin', 'requested', clock_timestamp() - INTERVAL '1 minute')""",
            (del_id, art_id),
        )

    # 3. Trigger cleanup / janitor
    mgr.cleanup_expired()

    # Assert artifact state is deleted, replica state is deleted, and deletion job is completed
    with control_plane_transaction() as conn:
        row = conn.execute(
            "SELECT state, deleted_at FROM storage.artifacts WHERE artifact_id = %s",
            (art_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "deleted"
        assert row[1] is not None

        rep = conn.execute(
            "SELECT state FROM storage.artifact_replicas WHERE artifact_id = %s",
            (art_id,),
        ).fetchone()
        assert rep is not None
        assert rep[0] == "deleted"

        job = conn.execute(
            "SELECT state, completed_at, attempt_count FROM storage.artifact_deletion_jobs WHERE deletion_id = %s",
            (del_id,),
        ).fetchone()
        assert job is not None
        assert job[0] == "completed"
        assert job[1] is not None
        assert job[2] == 1

    # Verify physical file is gone
    assert not os.path.exists(local_path)
