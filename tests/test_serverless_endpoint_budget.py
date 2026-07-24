"""Track B B3.2 — endpoint spend budget replaces per-request approval (§4.2/§15.3).

An inference call must never require a human approval click. Instead the
endpoint carries a server-enforced spend ceiling; when accrued cost reaches it,
the invocation is denied with a typed problem (402 `budget_exhausted`) — not a
silent success, not an unbounded bill, not an approval prompt.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from routes.serverless import _check_endpoint_budget, _enqueue_serverless_job

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _has = (
            _c.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name='serverless_endpoints' AND column_name='spend_limit_cad'"
            ).fetchone()
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("serverless_endpoints.spend_limit_cad missing — upgrade >= 071")

from serverless.repo import EndpointCreate, ServerlessRepo


# ── Unit: the budget gate ────────────────────────────────────────────


def test_over_budget_denies_with_typed_problem():
    ep = {"spend_limit_cad": 10.0, "total_cost_cad": 8.0, "unbilled_token_cost_cad": 3.0}
    with pytest.raises(HTTPException) as ei:
        _check_endpoint_budget(ep)
    assert ei.value.status_code == 402
    assert ei.value.detail["error"]["code"] == "budget_exhausted"
    assert ei.value.detail["error"]["accrued_cad"] == 11.0
    assert ei.value.detail["error"]["spend_limit_cad"] == 10.0


def test_within_budget_is_allowed():
    _check_endpoint_budget({"spend_limit_cad": 10.0, "total_cost_cad": 4.0, "unbilled_token_cost_cad": 1.0})


def test_null_limit_is_uncapped():
    # No endpoint cap set → the gate never fires (wallet/client budgets still apply).
    _check_endpoint_budget({"spend_limit_cad": None, "total_cost_cad": 999.0})


def test_exactly_at_limit_is_denied():
    with pytest.raises(HTTPException):
        _check_endpoint_budget({"spend_limit_cad": 5.0, "total_cost_cad": 5.0})


# ── Integration: the invocation path refuses before enqueuing ─────────


@pytest.fixture
def endpoint():
    repo = ServerlessRepo()
    ep = repo.create_endpoint(
        EndpointCreate(owner_id=f"own-{uuid.uuid4().hex[:8]}", name="b32", mode="preset", model_ref="m", min_workers=0)
    )
    yield repo, ep
    if _pool is not None:
        with _pool.connection() as conn:
            conn.execute("DELETE FROM serverless_jobs WHERE endpoint_id=%s", (ep["endpoint_id"],))
            conn.execute("DELETE FROM serverless_endpoints WHERE endpoint_id=%s", (ep["endpoint_id"],))
            conn.commit()


def test_enqueue_refused_when_endpoint_over_budget_no_job_created(endpoint):
    repo, ep = endpoint
    endpoint_id = ep["endpoint_id"]
    owner = str(ep["owner_id"])
    # Set a $1 cap and $2 accrued — exhausted.
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE serverless_endpoints SET spend_limit_cad = 1.0, total_cost_cad = 2.0 WHERE endpoint_id=%s",
            (endpoint_id,),
        )
        conn.commit()
    ep_over = repo.get_endpoint(endpoint_id)

    depth_before = repo.queue_depth(endpoint_id)
    with pytest.raises(HTTPException) as ei:
        _enqueue_serverless_job(
            endpoint_id=endpoint_id, owner_id=owner, payload={"input": "x"},
            ep=ep_over, idempotency_key=None, webhook_url=None,
        )
    assert ei.value.status_code == 402
    assert ei.value.detail["error"]["code"] == "budget_exhausted"
    # No job was enqueued — the refusal is *before* the queue write.
    assert repo.queue_depth(endpoint_id) == depth_before


def test_enqueue_proceeds_within_budget(endpoint):
    repo, ep = endpoint
    endpoint_id = ep["endpoint_id"]
    owner = str(ep["owner_id"])
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE serverless_endpoints SET spend_limit_cad = 100.0, total_cost_cad = 1.0 WHERE endpoint_id=%s",
            (endpoint_id,),
        )
        conn.commit()
    ep_ok = repo.get_endpoint(endpoint_id)

    job, _ = _enqueue_serverless_job(
        endpoint_id=endpoint_id, owner_id=owner, payload={"input": "x"},
        ep=ep_ok, idempotency_key=None, webhook_url=None,
    )
    assert job.get("job_id")
