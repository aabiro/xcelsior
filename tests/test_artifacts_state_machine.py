import os
import time
import uuid
import pytest
from artifacts import get_artifact_manager, ArtifactType, ResidencyPolicy
from control_plane.db import control_plane_transaction

# Ensure we run tests on postgres dev database cloned for tests
pytestmark = pytest.mark.anyio


@pytest.fixture
def test_setup():
    """Seed a tenant and job into the test database to satisfy foreign keys."""
    tenant_id = "tenant-sm-1"
    job_id = "job-sm-1"

    with control_plane_transaction() as conn:
        # Clean up any existing records
        conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))

        # Create dummy job
        conn.execute(
            """INSERT INTO jobs (job_id, status, priority, submitted_at, payload, tenant_id)
               VALUES (%s, 'submitted', 1, %s, '{}', %s)""",
            (job_id, time.time(), tenant_id),
        )

    yield {"tenant_id": tenant_id, "job_id": job_id}

    # Cleanup
    with control_plane_transaction() as conn:
        conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def test_artifact_upload_and_finalize_lifecycle(test_setup):
    job_id = test_setup["job_id"]
    mgr = get_artifact_manager()

    # 1. Request upload
    req = mgr.request_upload(
        artifact_type="job_output",
        job_id=job_id,
        filename="model.bin",
        content_type="application/octet-stream",
        residency=ResidencyPolicy.CANADA_ONLY,
    )

    assert "url" in req
    assert "artifact_id" in req
    assert "upload_session_id" in req

    art_id = req["artifact_id"]
    sess_id = req["upload_session_id"]

    # Verify initial database state (should be 'requested')
    with control_plane_transaction() as conn:
        row = conn.execute(
            "SELECT state, logical_name, primary_provider FROM storage.artifacts WHERE artifact_id = %s",
            (uuid.UUID(art_id),),
        ).fetchone()
        assert row is not None
        assert row[0] == "requested"
        assert row[1] == "model.bin"

        sess_row = conn.execute(
            "SELECT completed_at FROM storage.artifact_upload_sessions WHERE upload_session_id = %s",
            (uuid.UUID(sess_id),),
        ).fetchone()
        assert sess_row is not None
        assert sess_row[0] is None

    # Simulate S3 upload by creating a local file at the expected upload path (mock client uses local filesystem)
    # The file path is extracted from the local mock upload URL
    local_path = req["url"].replace("file://", "")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(b"model-binary-data")

    # 2. Finalize upload
    final = mgr.finalize_upload(sess_id)
    assert final["state"] == "available"
    assert final["size_bytes"] == 17  # len(b"model-binary-data")

    # Verify updated database state
    with control_plane_transaction() as conn:
        row = conn.execute(
            "SELECT state, size_bytes, available_at FROM storage.artifacts WHERE artifact_id = %s",
            (uuid.UUID(art_id),),
        ).fetchone()
        assert row is not None
        assert row[0] == "available"
        assert row[1] == 17
        assert row[2] is not None

        sess_row = conn.execute(
            "SELECT completed_at FROM storage.artifact_upload_sessions WHERE upload_session_id = %s",
            (uuid.UUID(sess_id),),
        ).fetchone()
        assert sess_row is not None
        assert sess_row[0] is not None

        replica = conn.execute(
            "SELECT state FROM storage.artifact_replicas WHERE artifact_id = %s",
            (uuid.UUID(art_id),),
        ).fetchone()
        assert replica is not None
        assert replica[0] == "active"

    # 3. List artifacts for job
    job_artifacts = mgr.get_job_artifacts(job_id)
    assert len(job_artifacts) == 1
    assert job_artifacts[0]["filename"] == "model.bin"
    assert job_artifacts[0]["size_bytes"] == 17

    # 4. Request download
    dl_url = mgr.download_url_for(
        artifact_type="job_output",
        job_id=job_id,
        filename="model.bin",
    )
    assert "url" in dl_url

    # 5. Request download by ID
    dl_by_id = mgr.request_download_by_id(art_id)
    assert "url" in dl_by_id
    assert dl_by_id["logical_name"] == "model.bin"
