"""C7 — TestClient + real-PG integration tests for the user_images endpoints.

Covers the P3.1 / A1 / A2 / B contract:
    - POST /instances/{job_id}/snapshot — happy path + 400 (not running) + 409 (dup) + 503 (registry unset)
    - GET /user-images — lists caller's live images only (soft-delete excluded)
    - DELETE /user-images/{id} — soft-delete semantics (recreate allowed afterwards)
    - POST /user-images/{id}/complete — **URL regression guard** (must exist at
      this path, not /api/v2/... which 404s — A1 bug) + idempotent recall
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

# Must set BEFORE importing api — _build_image_ref / rate-limit read env at module import
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")
os.environ["XCELSIOR_REGISTRY_URL"] = "ghcr.io/xcelsior-test"
os.environ["XCELSIOR_SNAPSHOT_RATE_LIMIT"] = "0"

import api  # noqa: E402  (FastAPI app)
import routes.instances as rinst  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    # Override auth to a NON-admin user so 403 path is actually reachable.
    # (Default test-mode _require_auth returns is_admin=True which would
    # bypass every owner check.)
    from routes import _deps as _rdeps

    original = rinst._require_auth

    def _fake_auth(request):
        return {
            "email": "alice@test",
            "user_id": "alice",
            "customer_id": "",
            "role": "user",
            "is_admin": False,
        }

    rinst._require_auth = _fake_auth  # type: ignore[assignment]
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        rinst._require_auth = original  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _force_registry_healthy(monkeypatch):
    """Default: registry is healthy → snapshot enters 'pending', not
    'queued_registry_down'. Individual tests can override."""
    import registry_health

    monkeypatch.setattr(registry_health, "is_registry_healthy", lambda: True)


@pytest.fixture(autouse=True)
def _stub_agent_enqueue(monkeypatch):
    """Avoid booking real agent_commands rows — return a synthetic cmd id."""
    import routes.agent as ragent

    def _fake(host_id, command, args, *, created_by, ttl_sec=900):
        return f"cmd-{uuid.uuid4().hex[:8]}"

    monkeypatch.setattr(ragent, "enqueue_agent_command", _fake)


@pytest.fixture
def seeded_job():
    """Insert a running job + host owned by 'alice'; delete on teardown."""
    from db import _get_pg_pool

    pool = _get_pg_pool()
    job_id = f"job-int-{uuid.uuid4().hex[:8]}"
    host_id = f"host-int-{uuid.uuid4().hex[:8]}"
    image_ids: list[str] = []

    now = time.time()
    host_payload = {
        "host_id": host_id,
        "status": "active",
        "owner": "alice",
        "registered_at": now,
    }
    job_payload = {
        "job_id": job_id,
        "status": "running",
        "owner": "alice",
        "host": host_id,
        "host_id": host_id,
        "image": "python:3.11",
        "name": "integration-test",
        "submitted_at": now,
    }

    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload) "
            "VALUES (%s, %s, %s, %s::jsonb) "
            "ON CONFLICT (host_id) DO UPDATE SET payload=EXCLUDED.payload",
            (host_id, "active", now, json.dumps(host_payload)),
        )
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, payload) "
            "VALUES (%s, %s, 0, %s, %s, %s::jsonb)",
            (job_id, "running", now, host_id, json.dumps(job_payload)),
        )
        conn.commit()

    yield {"job_id": job_id, "host_id": host_id, "image_ids": image_ids}

    # Teardown: drop any images created during the test, then the job + host.
    with pool.connection() as conn:
        if image_ids:
            conn.execute(
                "DELETE FROM user_images WHERE image_id = ANY(%s)", (image_ids,)
            )
        conn.execute("DELETE FROM user_images WHERE source_job_id=%s", (job_id,))
        conn.execute("DELETE FROM agent_commands WHERE host_id=%s", (host_id,))
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# POST /instances/{job_id}/snapshot
# ---------------------------------------------------------------------------

def test_snapshot_happy_path(client, seeded_job):
    r = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "mydemo", "tag": "v1", "description": "integration happy"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "pending"
    assert body["image_id"].startswith("img-")
    assert body["image_ref"].startswith("ghcr.io/xcelsior-test/")
    assert body["image_ref"].endswith("/mydemo:v1")
    seeded_job["image_ids"].append(body["image_id"])

    # Row must exist in DB
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT status, owner_id, source_job_id FROM user_images WHERE image_id=%s",
            (body["image_id"],),
        ).fetchone()
    assert row is not None
    assert row[0] == "pending"
    assert row[1] == "alice"
    assert row[2] == seeded_job["job_id"]


def test_snapshot_400_when_job_not_running(client, seeded_job):
    # Flip the seeded job to 'stopped'
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='stopped', "
            "payload = jsonb_set(payload, '{status}', '\"stopped\"') "
            "WHERE job_id=%s",
            (seeded_job["job_id"],),
        )
        conn.commit()

    r = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "nope", "tag": "v1", "description": "should 400"},
    )
    assert r.status_code == 400
    assert "running" in r.text.lower()


def test_snapshot_403_when_owner_mismatch(client, seeded_job):
    # Seeded job.owner='alice', but flip to 'bob' so our auth user 'alice'
    # gets rejected (remember: fixture pins is_admin=False).
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload = jsonb_set(payload, '{owner}', '\"bob\"') "
            "WHERE job_id=%s",
            (seeded_job["job_id"],),
        )
        conn.commit()

    r = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "nope", "tag": "v1", "description": "should 403"},
    )
    assert r.status_code == 403


def test_snapshot_409_duplicate_then_soft_delete_allows_recreate(client, seeded_job):
    # First — create
    r1 = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "dupname", "tag": "v1", "description": "first"},
    )
    assert r1.status_code == 200, r1.text
    img1 = r1.json()["image_id"]
    seeded_job["image_ids"].append(img1)

    # Second with same name:tag — 409
    r2 = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "dupname", "tag": "v1", "description": "dup"},
    )
    assert r2.status_code == 409

    # Soft-delete img1
    rd = client.delete(f"/user-images/{img1}")
    assert rd.status_code == 200, rd.text

    # Third attempt — same name:tag — MUST now succeed (A2 partial unique)
    r3 = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "dupname", "tag": "v1", "description": "reborn"},
    )
    assert r3.status_code == 200, r3.text
    img3 = r3.json()["image_id"]
    assert img3 != img1
    seeded_job["image_ids"].append(img3)


def test_snapshot_503_when_registry_unset(client, seeded_job, monkeypatch):
    monkeypatch.delenv("XCELSIOR_REGISTRY_URL", raising=False)
    r = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "noreg", "tag": "v1", "description": "registry unset"},
    )
    assert r.status_code == 503
    assert "registry_not_configured" in r.text


# ---------------------------------------------------------------------------
# GET /user-images
# ---------------------------------------------------------------------------

def test_list_excludes_soft_deleted(client, seeded_job):
    r1 = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "listme", "tag": "v1", "description": "to list"},
    )
    assert r1.status_code == 200
    img_id = r1.json()["image_id"]
    seeded_job["image_ids"].append(img_id)

    lst = client.get("/user-images")
    assert lst.status_code == 200
    items = lst.json().get("images") or lst.json().get("items") or lst.json()
    ids = [i["image_id"] for i in items] if isinstance(items, list) else []
    assert img_id in ids

    # Soft-delete, then re-list — must disappear
    assert client.delete(f"/user-images/{img_id}").status_code == 200
    lst2 = client.get("/user-images")
    items2 = lst2.json().get("images") or lst2.json().get("items") or lst2.json()
    ids2 = [i["image_id"] for i in items2] if isinstance(items2, list) else []
    assert img_id not in ids2


# ---------------------------------------------------------------------------
# POST /user-images/{id}/complete — A1 URL regression guard
# ---------------------------------------------------------------------------

def test_complete_callback_url_is_mounted_at_expected_path(client, seeded_job):
    """A1 regression: the worker agent posts to
    ``/user-images/{id}/complete`` — NOT ``/api/v2/user-images/{id}/complete``.
    Calling the wrong URL silently 404s and snapshots hang in 'pending'
    forever. This test asserts the correct path is live (even for a
    non-existent id, the route must exist and return 404 — not a
    router-level 404 meaning 'no such endpoint', but a handler-level
    404 meaning 'image not found'; we use status_code + body shape).
    """
    # Bogus id → handler returns 404 'Image not found' (with auth
    # bypassed in test env, the code path reaches the SELECT + raise).
    r = client.post(
        "/user-images/img-does-not-exist-xyz/complete",
        json={"status": "ready", "size_bytes": 0, "error": ""},
    )
    assert r.status_code == 404
    assert "image not found" in r.text.lower()

    # And the WRONG path (the A1 bug we're guarding against) must 404
    # as an unknown route — proving /api/v2/... is NOT where it lives.
    wrong = client.post(
        "/api/v2/user-images/anything/complete",
        json={"status": "ready", "size_bytes": 0, "error": ""},
    )
    assert wrong.status_code == 404


def test_complete_flips_pending_to_ready_and_is_idempotent(client, seeded_job):
    r1 = client.post(
        f"/instances/{seeded_job['job_id']}/snapshot",
        json={"name": "done", "tag": "v1", "description": "complete flow"},
    )
    assert r1.status_code == 200, r1.text
    image_id = r1.json()["image_id"]
    seeded_job["image_ids"].append(image_id)

    cb1 = client.post(
        f"/user-images/{image_id}/complete",
        json={"status": "ready", "size_bytes": 42, "error": ""},
    )
    assert cb1.status_code == 200, cb1.text
    assert cb1.json()["ok"] is True

    # DB flipped
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT status, size_bytes FROM user_images WHERE image_id=%s",
            (image_id,),
        ).fetchone()
    assert row[0] == "ready"
    assert row[1] == 42

    # Idempotent recall — a second callback must NOT re-update; returns
    # the CURRENT status (still 'ready'), not an error.
    cb2 = client.post(
        f"/user-images/{image_id}/complete",
        json={"status": "failed", "size_bytes": 99, "error": "retried"},
    )
    assert cb2.status_code == 200
    assert cb2.json().get("status") == "ready"  # NOT 'failed'

    # DB MUST still be ready / 42
    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT status, size_bytes FROM user_images WHERE image_id=%s",
            (image_id,),
        ).fetchone()
    assert row[0] == "ready"
    assert row[1] == 42
