"""Verified-only placement: "hosts are verified before they take jobs" (Phase 0.6).

The gate is the `host_verified` hard filter, switched by
`XCELSIOR_PLACEMENT_REQUIRE_VERIFIED` (off unless set). A host is eligible only
while its verification reads `verified` at decision time — not overdue,
de-verified, unverified or mid-verification. The reservation re-reads it, and a
platform admin may override it per job for operator testing.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from verification import HostVerification, VerificationStore  # noqa: E402


def _host() -> str:
    return f"vgate-{uuid.uuid4().hex[:10]}"


def _cleanup(host_id: str) -> None:
    with VerificationStore()._conn() as conn:
        for table in ("host_verifications", "verification_history", "job_failure_log"):
            conn.execute(f"DELETE FROM {table} WHERE host_id = %s", (host_id,))


def _client():
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


@pytest.fixture(scope="module")
def user_headers():
    client = _client()
    email = f"vgate-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Gate Override"},
    )
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.parametrize(
    "freshness, eligible",
    [
        ("verified", True),
        ("overdue", False),
        ("deverified", False),
        ("unverified", False),
        ("verifying", False),
        (None, False),
    ],
)
def test_the_gate_refuses_hosts_that_are_not_currently_verified(freshness, eligible):
    from control_plane.scheduler.filters import FilterContext, host_verified

    host = {"host_id": "h", "verification_freshness": freshness}
    reason = host_verified({}, host, FilterContext(require_verified=True))
    assert (reason is None) is eligible
    if reason is not None:
        assert reason.code == "host_not_verified"
    # Off: nobody is refused on verification.
    assert host_verified({}, host, FilterContext(require_verified=False)) is None
    # The admin override places anyway.
    assert (
        host_verified(
            {"verification_gate_override": True}, host, FilterContext(require_verified=True)
        )
        is None
    )


def test_the_gate_reads_its_switch_when_a_context_is_built(monkeypatch):
    from control_plane.scheduler.filters import FilterContext, HARD_FILTERS, host_verified

    monkeypatch.setenv("XCELSIOR_PLACEMENT_REQUIRE_VERIFIED", "true")
    assert FilterContext().require_verified is True
    monkeypatch.setenv("XCELSIOR_PLACEMENT_REQUIRE_VERIFIED", "false")
    assert FilterContext().require_verified is False
    monkeypatch.delenv("XCELSIOR_PLACEMENT_REQUIRE_VERIFIED")
    assert FilterContext().require_verified is False
    assert host_verified in HARD_FILTERS


def test_read_fleet_carries_each_hosts_current_verification():
    from control_plane.db import control_plane_transaction
    from control_plane.scheduler.snapshot import read_fleet

    fresh, late, none = _host(), _host(), _host()
    now = time.time()
    store = VerificationStore()
    with control_plane_transaction() as conn:
        for h in (fresh, late, none):
            conn.execute(
                "INSERT INTO hosts (host_id, status, registered_at, payload) "
                "VALUES (%s, 'active', %s, '{}')",
                (h, now),
            )
    try:
        store.save_verification(
            HostVerification(
                host_id=fresh, state="verified", last_check_at=now, next_check_at=now + 3600
            )
        )
        store.save_verification(
            HostVerification(
                host_id=late,
                state="verified",
                last_check_at=now - 3 * 86400,
                next_check_at=now - 2 * 86400,
            )
        )
        with control_plane_transaction() as conn:
            fleet = read_fleet(conn)
        by_id = {h["host_id"]: h for h in fleet.hosts}
        assert by_id[fresh]["verification_freshness"] == "verified"
        assert by_id[late]["verification_freshness"] == "overdue"
        assert by_id[none]["verification_freshness"] == "unverified"
    finally:
        for h in (fresh, late, none):
            _cleanup(h)
        with control_plane_transaction() as conn:
            conn.execute("DELETE FROM hosts WHERE host_id = ANY(%s)", ([fresh, late, none],))


def test_only_an_admin_may_ask_for_the_override(user_headers):
    r = _client().post(
        "/instance",
        json={
            "name": "gate-override",
            "vram_needed_gb": 4,
            "image": "nvidia/cuda:12.4.1-base-ubuntu22.04",
            "verification_gate_override": True,
        },
        headers=user_headers,
    )
    assert r.status_code == 403, r.text
    assert "verification_gate_override is operator-only" in r.text, r.text


def test_the_override_is_an_execution_field_the_scheduler_reads():
    from control_plane.job_specs import EXECUTION_FIELDS

    assert "verification_gate_override" in EXECUTION_FIELDS


def test_the_switch_reaches_every_process_that_places():
    """No `env_file:` in compose: a variable reaches a container only if named."""
    from pathlib import Path

    import yaml

    compose = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text()
    )
    for service in ("api", "api-blue", "bg-worker", "scheduler-worker"):
        env = compose["services"][service]["environment"]
        assert "XCELSIOR_PLACEMENT_REQUIRE_VERIFIED" in env, service
