"""Track B B2.8 — POST /api/v1/placements/simulate.

Read-only placement feasibility: reuses the launch service's snapshot + Stage-C
filter, so it must create no plan, attempt, allocation, or lease — only report
whether the spec could be placed right now.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT to_regclass('action_plans')").fetchone()
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None

client = TestClient(app)


def _token(email: str) -> str:
    return client.post(
        "/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}
    ).json()["access_token"]


def _counts() -> dict[str, int]:
    with _pool.connection() as c:
        return {
            "plans": c.execute("SELECT count(*) FROM action_plans").fetchone()[0],
            "attempts": c.execute("SELECT count(*) FROM job_attempts").fetchone()[0],
            "leases": c.execute("SELECT count(*) FROM placement_leases").fetchone()[0],
        }


def test_simulate_is_readonly_and_returns_availability():
    token = _token("b28-sim@xcelsior.ca")
    before = _counts()
    r = client.post(
        "/api/v1/placements/simulate",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "sim", "interactive": False, "vram_needed_gb": 8},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["spec_hash"]
    assert "feasible" in body["availability"]
    assert "hosts_considered" in body["availability"]
    # Strictly read-only: nothing was persisted.
    assert _counts() == before
