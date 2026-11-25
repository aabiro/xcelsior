"""A provider reads its own host's events; nobody else learns the host exists.

The guard compared the host id with the caller's owner ids (customer, user,
email, provider), so it matched no real host: every owner got 403 for their own
host, and only admins could read host events. The live rehearsal of every MCP
tool found it on 2026-11-24 (get_event_history entity_type=host).
"""

import json
import os
import time
import uuid

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_RATE_LIMIT_REQUESTS"] = "5000"
os.environ["XCELSIOR_AUTH_RATE_LIMIT_REQUESTS"] = "5000"

import pytest
from fastapi.testclient import TestClient

from api import app
from db import _get_pg_pool

client = TestClient(app)


def _user(label: str) -> dict:
    email = f"host-events-{label}-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": f"Host {label}"},
    )
    assert reg.status_code == 200, reg.text[:300]
    body = reg.json()
    return {
        "customer_id": body["user"]["customer_id"],
        "headers": {"Authorization": f"Bearer {body['access_token']}"},
    }


@pytest.fixture()
def owned_host():
    owner, stranger = _user("owner"), _user("stranger")
    host_id = f"host-events-{uuid.uuid4().hex[:12]}"
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload, owner_id, tenant_id) "
            "VALUES (%s, 'active', %s, %s, %s, %s)",
            (
                host_id,
                time.time(),
                json.dumps({"host_id": host_id, "gpu_model": "RTX 4090"}),
                owner["customer_id"],
                owner["customer_id"],
            ),
        )
        conn.commit()
    try:
        yield host_id, owner, stranger
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
            conn.commit()


def test_the_owner_reads_its_host_events(owned_host):
    host_id, owner, _ = owned_host
    r = client.get(f"/api/events/host/{host_id}", headers=owner["headers"])
    assert r.status_code == 200, r.text[:300]
    assert r.json()["entity_id"] == host_id


def test_another_account_is_told_the_host_is_not_found(owned_host):
    host_id, _, stranger = owned_host
    r = client.get(f"/api/events/host/{host_id}", headers=stranger["headers"])
    assert r.status_code == 404, r.text[:300]


def test_an_unknown_host_is_not_found(owned_host):
    _, owner, _ = owned_host
    r = client.get(f"/api/events/host/no-such-{uuid.uuid4().hex[:8]}", headers=owner["headers"])
    assert r.status_code == 404
