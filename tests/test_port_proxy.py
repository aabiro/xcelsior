"""P2.2 HTTP port proxy — integration tests.

Covers the three new endpoints introduced for subdomain routing:

    * ``POST /instances/{job_id}/http-ports/report`` — worker reports its
      final ``{container_port: host_port}`` allocation.
    * ``POST /instances/{job_id}/expose`` — user-facing; returns the
      public URL for a port that was already declared + mapped.
    * ``GET /internal/route/{slug}/{port}`` — nginx auth_request target.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")
os.environ.setdefault("XCELSIOR_PUBLIC_HOST", "xcelsior.test")

import api  # noqa: E402
import routes.instances as rinst  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Non-admin alice — exercises ownership checks that admin bypasses."""
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


@pytest.fixture
def seeded_job():
    """Running job owned by 'alice' with exposed_ports=[8888,8080].

    We do NOT pre-populate ``http_ports`` — individual tests either seed
    it directly (to test /expose happy path) or let the /http-ports/report
    endpoint install it (to test that path).
    """
    from db import _get_pg_pool

    pool = _get_pg_pool()
    job_id = f"abcd{uuid.uuid4().hex[:20]}"  # 24 chars, stable slug prefix
    host_id = f"host-pp-{uuid.uuid4().hex[:8]}"

    now = time.time()
    host_payload = {
        "host_id": host_id,
        "status": "active",
        "owner": "alice",
        "registered_at": now,
        "ip": "100.64.1.42",
    }
    job_payload = {
        "job_id": job_id,
        "status": "running",
        "owner": "alice",
        "host": host_id,
        "host_id": host_id,
        "image": "python:3.11",
        "name": "pp-test",
        "submitted_at": now,
        "exposed_ports": [8888, 8080],
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

    yield {"job_id": job_id, "host_id": host_id, "slug": job_id[:12]}

    with pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=%s", (job_id,))
        conn.execute("DELETE FROM hosts WHERE host_id=%s", (host_id,))
        conn.commit()


def _set_http_ports(job_id: str, ports: dict[str, int]) -> None:
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload = jsonb_set(payload, '{http_ports}', %s::jsonb, true) "
            "WHERE job_id=%s",
            (json.dumps(ports), job_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# POST /instances/{job_id}/http-ports/report
# ---------------------------------------------------------------------------

def test_http_ports_report_stores_mapping(client, seeded_job):
    r = client.post(
        f"/instances/{seeded_job['job_id']}/http-ports/report",
        json={
            "host_id": seeded_job["host_id"],
            "ports": {"8888": 55123, "8080": 55124},
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT payload->'http_ports' FROM jobs WHERE job_id=%s",
            (seeded_job["job_id"],),
        ).fetchone()
    http_ports = row[0]
    if isinstance(http_ports, str):
        http_ports = json.loads(http_ports)
    assert http_ports == {"8888": 55123, "8080": 55124}


def test_http_ports_report_404_for_wrong_host(client, seeded_job):
    r = client.post(
        f"/instances/{seeded_job['job_id']}/http-ports/report",
        json={"host_id": "other-host", "ports": {"8888": 55123}},
    )
    assert r.status_code == 404


def test_http_ports_report_drops_invalid_entries(client, seeded_job):
    r = client.post(
        f"/instances/{seeded_job['job_id']}/http-ports/report",
        json={
            "host_id": seeded_job["host_id"],
            "ports": {
                "22": 55100,        # reserved → dropped
                "8888": 54999,      # host port below 55000 → dropped
                "8080": 60001,      # host port above 59999 → dropped
                "9000": 55500,      # ok
                "not-a-port": 55501,  # non-numeric → dropped
            },
        },
    )
    assert r.status_code == 200

    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT payload->'http_ports' FROM jobs WHERE job_id=%s",
            (seeded_job["job_id"],),
        ).fetchone()
    http_ports = row[0]
    if isinstance(http_ports, str):
        http_ports = json.loads(http_ports)
    assert http_ports == {"9000": 55500}


# ---------------------------------------------------------------------------
# POST /instances/{job_id}/expose
# ---------------------------------------------------------------------------

def test_expose_happy_path(client, seeded_job):
    _set_http_ports(seeded_job["job_id"], {"8888": 55123})
    r = client.post(
        f"/instances/{seeded_job['job_id']}/expose",
        json={"container_port": 8888},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["url"] == f"https://{seeded_job['slug']}-8888.xcelsior.test"
    assert body["host_port"] == 55123
    assert body["host_ip"] == "100.64.1.42"


def test_expose_400_when_port_not_declared(client, seeded_job):
    _set_http_ports(seeded_job["job_id"], {"8888": 55123})
    r = client.post(
        f"/instances/{seeded_job['job_id']}/expose",
        json={"container_port": 9999},  # not in exposed_ports
    )
    assert r.status_code == 400
    assert "not declared" in r.text.lower()


def test_expose_409_when_mapping_not_yet_reported(client, seeded_job):
    # http_ports is empty → mapping isn't ready yet
    r = client.post(
        f"/instances/{seeded_job['job_id']}/expose",
        json={"container_port": 8888},
    )
    assert r.status_code == 409


def test_expose_400_when_not_running(client, seeded_job):
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='stopped', "
            "payload = jsonb_set(payload, '{status}', '\"stopped\"') "
            "WHERE job_id=%s",
            (seeded_job["job_id"],),
        )
        conn.commit()

    _set_http_ports(seeded_job["job_id"], {"8888": 55123})
    r = client.post(
        f"/instances/{seeded_job['job_id']}/expose",
        json={"container_port": 8888},
    )
    assert r.status_code == 400


def test_expose_403_when_owner_mismatch(client, seeded_job):
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload = jsonb_set(payload, '{owner}', '\"bob\"') "
            "WHERE job_id=%s",
            (seeded_job["job_id"],),
        )
        conn.commit()

    _set_http_ports(seeded_job["job_id"], {"8888": 55123})
    r = client.post(
        f"/instances/{seeded_job['job_id']}/expose",
        json={"container_port": 8888},
    )
    assert r.status_code == 403


# NOTE: cport=22 rejection is exercised via the _HttpPortsReport
# validator in test_http_ports_report_drops_invalid_entries. We skip a
# direct /expose 422 test because the app's request-validation handler
# has a pre-existing serializer bug on ValueError that is unrelated to
# this change.


# ---------------------------------------------------------------------------
# GET /internal/route/{slug}/{port}
# ---------------------------------------------------------------------------

def test_internal_route_returns_upstream_header(client, seeded_job):
    _set_http_ports(seeded_job["job_id"], {"8888": 55123})
    r = client.get(f"/internal/route/{seeded_job['slug']}/8888")
    assert r.status_code == 200, r.text
    assert r.headers.get("X-Upstream") == "100.64.1.42:55123"
    body = r.json()
    assert body["upstream"] == "100.64.1.42:55123"
    assert body["job_id"] == seeded_job["job_id"]


def test_internal_route_404_for_unknown_slug(client):
    r = client.get("/internal/route/deadbeefcafe/8888")
    assert r.status_code == 404


def test_internal_route_404_when_port_not_mapped(client, seeded_job):
    _set_http_ports(seeded_job["job_id"], {"8888": 55123})
    r = client.get(f"/internal/route/{seeded_job['slug']}/9999")
    assert r.status_code == 404


def test_internal_route_404_when_job_not_running(client, seeded_job):
    from db import _get_pg_pool

    _set_http_ports(seeded_job["job_id"], {"8888": 55123})
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET status='stopped' WHERE job_id=%s",
            (seeded_job["job_id"],),
        )
        conn.commit()

    r = client.get(f"/internal/route/{seeded_job['slug']}/8888")
    assert r.status_code == 404


def test_internal_route_rejects_malformed_slug(client):
    # uppercase not allowed — slug is DNS-safe lowercase hex only
    r = client.get("/internal/route/ABCDEF123456/8888")
    assert r.status_code == 404
