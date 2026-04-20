"""HTTP API route tests for volume endpoints.

Uses FastAPI TestClient with auth mocking.
Tests: validation, auth, lifecycle, retry endpoint, error handling.
"""

import os
import uuid
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Reload .env.test to pick up correct token even if another test module
# (e.g. test_instance_flow) clobbered XCELSIOR_API_TOKEN at module level.
_env_test = Path(__file__).resolve().parent.parent / ".env.test"
if _env_test.exists():
    load_dotenv(_env_test, override=True)

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")


TEST_USER = {
    "email": "vol-test@xcelsior.ca",
    "user_id": "vol-test-user",
    "role": "customer",
    "is_admin": False,
    "name": "Vol Tester",
    "scopes": ["volumes:read", "volumes:write"],
}


@pytest.fixture
def client():
    """FastAPI TestClient with volume routes."""
    from api import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers():
    """Bearer token that resolves to admin user."""
    token = os.environ.get("XCELSIOR_API_TOKEN", "test-token-not-for-production")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def cleanup_vids():
    """Collect and cleanup volume IDs after test."""
    vids = []
    yield vids
    if not vids:
        return
    try:
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            for vid in vids:
                conn.execute("DELETE FROM volume_attachments WHERE volume_id = %s", (vid,))
                conn.execute("DELETE FROM volumes WHERE volume_id = %s", (vid,))
            conn.commit()
    except Exception:
        pass


def _unique_name():
    return f"api-test-{uuid.uuid4().hex[:8]}"


# ── Auth tests ───────────────────────────────────────────────────────


class TestVolumeAuth:
    """Volume endpoints require authentication."""

    def test_list_unauthenticated(self, client):
        r = client.get("/api/v2/volumes")
        assert r.status_code in (401, 403)

    def test_create_unauthenticated(self, client):
        r = client.post("/api/v2/volumes", json={"name": "bad", "size_gb": 1})
        assert r.status_code in (401, 403)

    def test_get_unauthenticated(self, client):
        r = client.get("/api/v2/volumes/vol-doesnotexist")
        assert r.status_code in (401, 403)


# ── Validation tests ─────────────────────────────────────────────────


class TestVolumeValidation:
    """Input validation on create endpoint."""

    def test_name_too_short(self, client, auth_headers):
        r = client.post("/api/v2/volumes", json={"name": "", "size_gb": 1}, headers=auth_headers)
        assert r.status_code == 422

    def test_size_zero(self, client, auth_headers):
        r = client.post("/api/v2/volumes", json={"name": "valid", "size_gb": 0}, headers=auth_headers)
        assert r.status_code == 422

    def test_size_too_large(self, client, auth_headers):
        r = client.post("/api/v2/volumes", json={"name": "valid", "size_gb": 3000}, headers=auth_headers)
        assert r.status_code == 422

    def test_invalid_mount_path(self, client, auth_headers):
        r = client.post(
            "/api/v2/volumes/vol-x/attach",
            json={"instance_id": "j-1", "mount_path": "/etc/shadow"},
            headers=auth_headers,
        )
        assert r.status_code == 422


# ── CRUD lifecycle (uses admin token) ────────────────────────────────


class TestVolumeCRUD:
    """Create → List → Get → Delete lifecycle via HTTP."""

    def test_create_volume(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 5}, headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["volume"]["name"] == name
        assert body["volume"]["size_gb"] == 5
        assert body["volume"]["status"] == "available"
        cleanup_vids.append(body["volume"]["volume_id"])

    def test_list_volumes_includes_created(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        vid = r.json()["volume"]["volume_id"]
        cleanup_vids.append(vid)

        r2 = client.get("/api/v2/volumes", headers=auth_headers)
        assert r2.status_code == 200
        ids = [v["volume_id"] for v in r2.json()["volumes"]]
        assert vid in ids

    def test_get_volume(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        vid = r.json()["volume"]["volume_id"]
        cleanup_vids.append(vid)

        r2 = client.get(f"/api/v2/volumes/{vid}", headers=auth_headers)
        assert r2.status_code == 200
        assert r2.json()["volume"]["volume_id"] == vid

    def test_get_nonexistent_volume(self, client, auth_headers):
        r = client.get("/api/v2/volumes/vol-doesnotexist", headers=auth_headers)
        assert r.status_code == 404

    def test_delete_volume(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        vid = r.json()["volume"]["volume_id"]
        cleanup_vids.append(vid)

        r2 = client.delete(f"/api/v2/volumes/{vid}", headers=auth_headers)
        assert r2.status_code == 200

        # Should no longer be visible
        r3 = client.get(f"/api/v2/volumes/{vid}", headers=auth_headers)
        assert r3.status_code == 404

    def test_duplicate_name_rejected(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        cleanup_vids.append(r.json()["volume"]["volume_id"])

        r2 = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        assert r2.status_code == 400


# ── Retry endpoint ───────────────────────────────────────────────────


class TestRetryEndpoint:
    """POST /api/v2/volumes/{volume_id}/retry endpoint tests."""

    def test_retry_error_volume(self, client, auth_headers, cleanup_vids):
        """Retry a volume in 'error' state."""
        # Create a volume and force it to error state
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()

        vid = f"vol-{uuid.uuid4().hex[:12]}"
        cleanup_vids.append(vid)
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """INSERT INTO volumes (volume_id, owner_id, name, storage_type, size_gb,
                   region, province, encrypted, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (vid, "api-admin", "retry-api-test", "nfs", 1, "ca-east", "ON", False, "error", time.time()),
            )
            conn.commit()

        r = client.post(f"/api/v2/volumes/{vid}/retry", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json()["volume"]["status"] == "available"

    def test_retry_available_volume_rejected(self, client, auth_headers, cleanup_vids):
        """Cannot retry a volume that's not in 'error' state."""
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        vid = r.json()["volume"]["volume_id"]
        cleanup_vids.append(vid)

        r2 = client.post(f"/api/v2/volumes/{vid}/retry", headers=auth_headers)
        assert r2.status_code == 400

    def test_retry_nonexistent_volume(self, client, auth_headers):
        r = client.post("/api/v2/volumes/vol-doesnotexist/retry", headers=auth_headers)
        assert r.status_code in (400, 404)


# ── Available volumes endpoint ───────────────────────────────────────


class TestAvailableEndpoint:
    """GET /api/v2/volumes/available filters correctly."""

    def test_only_returns_available(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        vid = r.json()["volume"]["volume_id"]
        cleanup_vids.append(vid)

        r2 = client.get("/api/v2/volumes/available", headers=auth_headers)
        assert r2.status_code == 200
        for v in r2.json()["volumes"]:
            # Only available volumes should appear
            if "status" in v:
                assert v["status"] == "available"

    def test_includes_created_volume(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        vid = r.json()["volume"]["volume_id"]
        cleanup_vids.append(vid)

        r2 = client.get("/api/v2/volumes/available", headers=auth_headers)
        ids = [v["volume_id"] for v in r2.json()["volumes"]]
        assert vid in ids


# ── Detach endpoint edge cases ───────────────────────────────────────


class TestDetachEndpoint:
    """POST /api/v2/volumes/{volume_id}/detach edge cases."""

    def test_detach_unattached_volume(self, client, auth_headers, cleanup_vids):
        """Detaching a volume that's not attached returns 400."""
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        vid = r.json()["volume"]["volume_id"]
        cleanup_vids.append(vid)

        r2 = client.post(f"/api/v2/volumes/{vid}/detach", headers=auth_headers)
        assert r2.status_code == 400

    def test_detach_nonexistent(self, client, auth_headers):
        r = client.post("/api/v2/volumes/vol-doesnotexist/detach", headers=auth_headers)
        assert r.status_code == 404


# ── Response shape verification ──────────────────────────────────────


class TestResponseShape:
    """Verify API responses contain expected fields."""

    def test_create_includes_pricing(self, client, auth_headers, cleanup_vids):
        from volumes import VOLUME_PRICE_PER_GB_MONTH_CAD
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 10}, headers=auth_headers)
        vol = r.json()["volume"]
        cleanup_vids.append(vol["volume_id"])

        assert "price_per_gb_month_cad" in vol
        assert "estimated_monthly_cost_cad" in vol
        assert vol["estimated_monthly_cost_cad"] == round(10 * VOLUME_PRICE_PER_GB_MONTH_CAD, 2)

    def test_list_includes_pricing(self, client, auth_headers, cleanup_vids):
        name = _unique_name()
        r = client.post("/api/v2/volumes", json={"name": name, "size_gb": 1}, headers=auth_headers)
        cleanup_vids.append(r.json()["volume"]["volume_id"])

        r2 = client.get("/api/v2/volumes", headers=auth_headers)
        for v in r2.json()["volumes"]:
            assert "price_per_gb_month_cad" in v
            assert "monthly_cost_cad" in v
