"""Smoke coverage for volume snapshot routes (routes/volumes.py)."""

import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app, raise_server_exceptions=False)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def cleanup_vids():
    vids: list[str] = []
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
                conn.execute(
                    "DELETE FROM volume_snapshots WHERE volume_id = %s", (vid,)
                )
                conn.execute(
                    "DELETE FROM volume_attachments WHERE volume_id = %s", (vid,)
                )
                conn.execute("DELETE FROM volumes WHERE volume_id = %s", (vid,))
            conn.commit()
    except Exception:
        pass


@pytest.fixture
def volume_id(cleanup_vids):
    name = f"snapcov-{uuid.uuid4().hex[:8]}"
    r = client.post(
        "/api/v2/volumes",
        json={"name": name, "size_gb": 1},
        headers=_admin_headers(),
    )
    assert r.status_code == 200, r.text[:200]
    vid = r.json()["volume"]["volume_id"]
    cleanup_vids.append(vid)
    return vid


def test_volume_snapshot_create_list_delete(volume_id):
    headers = _admin_headers()
    r_create = client.post(
        f"/api/v2/volumes/{volume_id}/snapshots",
        headers=headers,
        json={"label": "cov-snap"},
    )
    assert r_create.status_code == 200, r_create.text[:200]
    snap_id = r_create.json()["snapshot"]["snapshot_id"]

    r_list = client.get(
        f"/api/v2/volumes/{volume_id}/snapshots",
        headers=headers,
    )
    assert r_list.status_code == 200
    ids = [s["snapshot_id"] for s in r_list.json()["snapshots"]]
    assert snap_id in ids

    r_restore = client.post(
        f"/api/v2/volumes/{volume_id}/snapshots/{snap_id}/restore",
        headers=headers,
    )
    assert r_restore.status_code == 200
    assert r_restore.json().get("ok") is True

    r_del = client.delete(
        f"/api/v2/volumes/{volume_id}/snapshots/{snap_id}",
        headers=headers,
    )
    assert r_del.status_code == 200
    assert r_del.json().get("ok") is True


def test_volume_snapshot_create_requires_auth(volume_id):
    r = client.post(
        f"/api/v2/volumes/{volume_id}/snapshots",
        json={"label": "no-auth"},
    )
    assert r.status_code in (401, 403)


def test_volume_snapshot_list_not_found():
    r = client.get(
        "/api/v2/volumes/vol-nonexistent/snapshots",
        headers=_admin_headers(),
    )
    assert r.status_code == 404


def test_volume_snapshot_delete_missing(volume_id):
    r = client.delete(
        f"/api/v2/volumes/{volume_id}/snapshots/snap-does-not-exist",
        headers=_admin_headers(),
    )
    assert r.status_code == 404