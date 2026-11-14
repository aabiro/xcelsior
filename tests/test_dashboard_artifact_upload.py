"""Dashboard artifact upload: personal slot, bytes, finalize, and read-back."""

import uuid

import pytest
from fastapi.testclient import TestClient

from api import app
from artifacts import _safe_artifact_filename, get_artifact_manager


def test_filename_cannot_escape_the_storage_prefix():
    assert _safe_artifact_filename("../../etc/passwd") == "passwd"
    assert _safe_artifact_filename("..") == "artifact"
    assert _safe_artifact_filename("weights.bin") == "weights.bin"


def test_standalone_upload_is_stored_finalized_and_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("XCELSIOR_STORAGE_BACKEND", "local")
    monkeypatch.setenv("XCELSIOR_STORAGE_LOCAL_DIR", str(tmp_path))
    # The manager is a process singleton and may already have been built.
    import artifacts as artifacts_mod

    artifacts_mod._artifact_manager = None
    owner = f"owner-{uuid.uuid4().hex[:8]}"
    tenant = f"team-{uuid.uuid4().hex[:8]}"
    mgr = get_artifact_manager()
    try:
        req = mgr.request_upload(
            artifact_type="model_weights",
            job_id=f"user-{owner}",
            filename="weights.bin",
            tenant_id=tenant,
            owner_user_id=owner,
            retain_days=0,
            storage_policy="primary",
        )
        assert req["url"]
        assert req["upload_session_id"]

        with pytest.raises(PermissionError):
            mgr.write_session_bytes(
                req["upload_session_id"],
                b"not-yours",
                principal_id="someone-else",
            )

        written = mgr.write_session_bytes(
            req["upload_session_id"],
            b"weights-bytes",
            principal_id=owner,
            content_type="application/octet-stream",
        )
        assert written["size_bytes"] == len(b"weights-bytes")

        final = mgr.finalize_upload(req["upload_session_id"])
        assert final["state"] == "available"
        assert final["size_bytes"] == len(b"weights-bytes")

        data, _content_type, name = mgr.read_artifact_bytes(
            req["artifact_id"],
            authorize=lambda _owner: None,
        )
        assert data == b"weights-bytes"
        assert name == "weights.bin"
    finally:
        from control_plane.db import control_plane_transaction

        with control_plane_transaction() as conn:
            conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant,))
        artifacts_mod._artifact_manager = None


def test_dashboard_http_upload_is_listed_and_downloadable(tmp_path, monkeypatch):
    """The artifacts page: ticket, same-origin bytes, finalize, list, download."""
    monkeypatch.setenv("XCELSIOR_STORAGE_BACKEND", "local")
    monkeypatch.setenv("XCELSIOR_STORAGE_LOCAL_DIR", str(tmp_path))
    import artifacts as artifacts_mod

    artifacts_mod._artifact_manager = None
    email = f"artup-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    artifact_id = None
    try:
        with TestClient(app) as client:
            registered = client.post(
                "/api/auth/register",
                json={"email": email, "password": "StrongPass123!", "name": "Upload"},
            )
            assert registered.status_code in (200, 201), registered.text
            login = client.post(
                "/api/auth/login",
                json={"email": email, "password": "StrongPass123!"},
            )
            assert login.status_code == 200, login.text
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

            ticket = client.post(
                "/api/artifacts/upload",
                json={
                    "job_id": "",
                    "filename": "../../weights.bin",
                    "artifact_type": "model_weights",
                    "storage_policy": "primary",
                    "retain_days": 0,
                },
                headers=headers,
            )
            assert ticket.status_code == 200, ticket.text
            body = ticket.json()
            assert body["url"]
            assert body["upload_url"] == body["url"]
            assert body["filename"] == "weights.bin"
            assert body["max_upload_bytes"] > 0
            session_id = body["upload_session_id"]
            artifact_id = body["artifact_id"]

            # Login stores the session cookie on this client. A second client
            # has neither that cookie nor a bearer token.
            with TestClient(app) as anonymous:
                denied = anonymous.put(
                    f"/api/artifacts/sessions/{session_id}/content",
                    content=b"stolen",
                    headers={"Content-Type": "application/octet-stream"},
                )
            assert denied.status_code == 401

            stored = client.put(
                f"/api/artifacts/sessions/{session_id}/content",
                content=b"weights-bytes",
                headers={**headers, "Content-Type": "application/octet-stream"},
            )
            assert stored.status_code == 200, stored.text
            assert stored.json()["size_bytes"] == len(b"weights-bytes")

            final = client.post(
                "/api/artifacts/finalize",
                json={"upload_session_id": session_id},
                headers=headers,
            )
            assert final.status_code == 200, final.text
            assert final.json()["state"] == "available"

            listed = client.get("/api/artifacts", headers=headers)
            assert listed.status_code == 200, listed.text
            match = next(
                row for row in listed.json()["artifacts"] if row["artifact_id"] == artifact_id
            )
            assert match["filename"] == "weights.bin"
            assert match["artifact_type"] == "model_weights"
            assert match["job_id"] in ("", None)

            downloaded = client.get(f"/api/artifacts/content/{artifact_id}", headers=headers)
            assert downloaded.status_code == 200, downloaded.text
            assert downloaded.content == b"weights-bytes"
            assert "weights.bin" in downloaded.headers.get("content-disposition", "")
    finally:
        if artifact_id:
            from control_plane.db import control_plane_transaction

            with control_plane_transaction() as conn:
                conn.execute(
                    "DELETE FROM storage.artifacts WHERE artifact_id = %s",
                    (artifact_id,),
                )
        artifacts_mod._artifact_manager = None
