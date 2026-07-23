"""Track B B2.3 — launch-plan preview creates a plan and nothing else.

The whole point of preview (§14.1) is that it is *informational*: it quotes
and persists an action plan, but it must not reserve a GPU, take a lease,
hold funds, or create a job. Those are execute's job (B2.5). This test pins
that boundary by counting the side-effect tables around a real HTTP preview.
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
except Exception as _e:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None

client = TestClient(app)

# Tables that preview must leave untouched.
_SIDE_EFFECT_TABLES = (
    "job_attempts",
    "gpu_device_allocations",
    "placement_leases",
    "wallet_holds",
    "jobs",
)


def _counts() -> dict[str, int]:
    out: dict[str, int] = {}
    with _pool.connection() as conn:
        for t in _SIDE_EFFECT_TABLES + ("action_plans",):
            out[t] = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    return out


def _register(email: str) -> str:
    reg = client.post(
        "/api/auth/register", json={"email": email, "password": "Str0ngPass!abc"}
    ).json()
    return reg["access_token"]


@pytest.fixture
def cleanup_plans():
    ids: list[str] = []
    yield ids
    if _pool is None:
        return
    with _pool.connection() as conn:
        for pid in ids:
            conn.execute("DELETE FROM action_plans WHERE plan_id = %s", (pid,))
        conn.commit()


def test_preview_creates_a_plan_and_no_side_effects(cleanup_plans):
    token = _register("b23-preview@xcelsior.ca")
    before = _counts()

    resp = client.post(
        "/api/v1/launch-plans",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "b23-preview",
            "num_gpus": 1,
            "image": "pytorch/pytorch:latest",
            "interactive": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    plan_id = body["plan_id"]
    assert plan_id
    cleanup_plans.append(plan_id)

    # Shape of the preview response (§14.1 step 8).
    assert body["status"] == "quoted"
    assert body["approval_mode"] in ("human", "standing_policy")
    assert "estimate" in body and "estimate_micros" in body["estimate"]
    assert "availability" in body
    assert body["expires_at"]

    after = _counts()
    # Exactly one new action plan …
    assert after["action_plans"] == before["action_plans"] + 1
    # … and zero of everything a launch would create.
    for t in _SIDE_EFFECT_TABLES:
        assert after[t] == before[t], f"preview created rows in {t}"


def test_preview_persists_canonical_spec_and_hash(cleanup_plans):
    token = _register("b23-canon@xcelsior.ca")
    resp = client.post(
        "/api/v1/launch-plans",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "b23-canon", "gpu_model": "H100", "num_gpus": 2, "interactive": False,
              "vram_needed_gb": 40},
    )
    assert resp.status_code == 200, resp.text
    plan_id = resp.json()["plan_id"]
    cleanup_plans.append(plan_id)

    # The stored canonical args and hash must match what the canonicalizer
    # produces for the same input — the binding the worker later verifies.
    from control_plane.launch import canonicalize, spec_hash

    expected = canonicalize(
        {"name": "b23-canon", "gpu_model": "H100", "num_gpus": 2,
         "interactive": False, "vram_needed_gb": 40}
    )
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT canonical_args, canonical_args_hash, spec_hash, estimate_micros "
            "FROM action_plans WHERE plan_id = %s",
            (plan_id,),
        ).fetchone()
    stored_args, stored_hash, stored_spec_hash, estimate = row
    assert stored_args == expected
    assert stored_hash == spec_hash(expected) == stored_spec_hash
    assert estimate is not None and int(estimate) >= 0


def test_invalid_spec_returns_422_and_no_plan(cleanup_plans):
    token = _register("b23-invalid@xcelsior.ca")
    before = _counts()
    resp = client.post(
        "/api/v1/launch-plans",
        headers={"Authorization": f"Bearer {token}"},
        # exposed_ports includes 22 (reserved) — rejected by JobIn before the
        # service is even reached, so still no plan is created.
        json={"name": "bad", "exposed_ports": [22]},
    )
    assert resp.status_code == 422
    after = _counts()
    assert after["action_plans"] == before["action_plans"]
