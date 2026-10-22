"""Retired v1 leases cannot mutate history; diagnostics use current fencing."""

import json
import os
import time
import uuid
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api import app
from db import _get_pg_pool
from tests.test_control_plane_leases import _claim_grant, _reserved
from tests.test_control_plane_leases import fleet as fleet

client = TestClient(app)


@pytest.fixture
def reservation(fleet):
    return _reserved(fleet)


@pytest.fixture
def old_lease(reservation):
    lease_id = str(uuid.uuid4())
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO leases(lease_id,job_id,host_id,granted_at,expires_at,"
            "last_renewed,duration_sec,status) VALUES(%s,%s,%s,%s,%s,%s,300,'active')",
            (
                lease_id,
                reservation.job_id,
                reservation.host_id,
                time.time(),
                time.time() + 300,
                time.time(),
            ),
        )
    yield lease_id
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM leases WHERE lease_id=%s", (lease_id,))


def _snapshot(job_id, lease_id):
    with _get_pg_pool().connection() as conn:
        return (
            conn.execute(
                "SELECT to_jsonb(l) FROM leases l WHERE lease_id=%s", (lease_id,)
            ).fetchone()[0],
            conn.execute("SELECT to_jsonb(j) FROM jobs j WHERE job_id=%s", (job_id,)).fetchone()[0],
            conn.execute("SELECT count(*) FROM events WHERE entity_id=%s", (job_id,)).fetchone()[0],
            conn.execute(
                "SELECT count(*) FROM outbox_events WHERE aggregate_id=%s", (job_id,)
            ).fetchone()[0],
        )


@pytest.mark.parametrize("mode", ["paused", "canary", "active"])
@pytest.mark.parametrize("operation", ["claim", "renew", "release"])
def test_retired_lease_requests_preserve_history_and_current_job(
    reservation, old_lease, monkeypatch, mode, operation
):
    import routes.agent as routes

    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", mode)
    notify = Mock(side_effect=AssertionError("retired lease notified a worker"))
    monkeypatch.setattr(routes, "broadcast_sse", notify)
    before = _snapshot(reservation.job_id, old_lease)
    response = client.post(
        f"/agent/lease/{operation}",
        json={"job_id": reservation.job_id, "host_id": reservation.host_id},
        headers={
            "Authorization": "Bearer "
            + (os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production")
        },
    )
    assert response.status_code == 410, response.text
    assert response.json()["error"]["code"] == "v2_protocol_required"
    assert _snapshot(reservation.job_id, old_lease) == before
    notify.assert_not_called()


@pytest.fixture
def caller(reservation, monkeypatch):
    import routes.control_plane_v1 as v1
    import routes.events as events

    user = {
        "customer_id": "lease-owner",
        "user_id": "lease-owner",
        "email": "lease-owner",
        "grant_type": "client_credentials",
        "scopes": ["instances:read", "events:read"],
    }
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE jobs SET payload=jsonb_set(payload,'{owner}',%s::jsonb) WHERE job_id=%s",
            (json.dumps(user["customer_id"]), reservation.job_id),
        )
    monkeypatch.setattr(v1, "_require_auth", lambda request: user)
    monkeypatch.setattr(events, "_require_auth", lambda request: user)
    return user


def _diagnostic_paths(job_id):
    return [f"/api/events/leases/{job_id}", f"/api/v1/instances/{job_id}/active-lease"]


def test_both_diagnostics_show_the_same_offered_and_claimed_fenced_lease(
    reservation, old_lease, caller
):
    from control_plane.lease_views import current_lease_view

    for expected in ("offered", "active"):
        if expected == "active":
            _claim_grant(reservation)
        with _get_pg_pool().connection() as conn:
            assert current_lease_view(conn, reservation.job_id)["status"] == expected
        leases = []
        for path in _diagnostic_paths(reservation.job_id):
            response = client.get(path)
            assert response.status_code == 200, response.text
            lease = response.json()["lease"]
            assert lease["lease_id"] == reservation.lease_id
            assert lease["attempt_id"] == reservation.attempt_id
            assert lease["status"] == expected
            assert "host_id" not in lease and "fencing_token" not in lease
            assert lease["host_alias"].startswith("host-")
            leases.append(lease)
        assert leases[0] == leases[1]


@pytest.mark.parametrize("mismatch", ["pointer", "host", "fence", "terminal"])
def test_diagnostics_refuse_stale_or_closed_attempt_bindings(
    reservation, old_lease, caller, mismatch
):
    statements = {
        "pointer": "UPDATE jobs SET active_attempt_id=NULL WHERE job_id=%s",
        "host": "UPDATE jobs SET host_id='another-host' WHERE job_id=%s",
        "fence": "UPDATE placement_leases SET fencing_token=fencing_token+1 WHERE job_id=%s",
        "terminal": "UPDATE job_attempts SET status='cancelled',ended_at=clock_timestamp() WHERE job_id=%s",
    }
    with _get_pg_pool().connection() as conn:
        conn.execute(statements[mismatch], (reservation.job_id,))
    assert client.get(_diagnostic_paths(reservation.job_id)[0]).status_code == 404
    response = client.get(_diagnostic_paths(reservation.job_id)[1])
    assert response.status_code == 200, response.text
    assert response.json()["lease"] is None


def test_diagnostics_preserve_tenant_and_scope_boundaries(reservation, caller):
    caller["scopes"] = ["profile"]
    for path in _diagnostic_paths(reservation.job_id):
        assert client.get(path).status_code == 403
    caller["scopes"] = ["instances:read", "events:read"]
    caller["customer_id"] = caller["user_id"] = caller["email"] = "other-tenant"
    assert client.get(_diagnostic_paths(reservation.job_id)[0]).status_code == 403
    assert client.get(_diagnostic_paths(reservation.job_id)[1]).status_code == 404


def test_overdue_current_lease_stays_visible_for_diagnosis(reservation, caller):
    from datetime import datetime

    _claim_grant(reservation)
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 minute' "
            "WHERE lease_id=%s",
            (reservation.lease_id,),
        )
    for path in _diagnostic_paths(reservation.job_id):
        response = client.get(path)
        assert response.status_code == 200, response.text
        lease = response.json()["lease"]
        assert lease["status"] == "active"
        assert datetime.fromisoformat(lease["expires_at"]).timestamp() < time.time()
