"""Releasing a GPU allocation does something, and only to your own.

`POST /api/v2/marketplace/release/{allocation_id}` passed an **allocation id**
to `release_allocation(job_id)`, which queries `WHERE job_id = %s`. It matched
nothing, updated nothing, and returned `{"ok": true}` — the worst kind of bug,
because the caller is told the GPU was freed and it was not. The offer stays
unavailable and the row stays open forever.

## Why the one-line fix was the wrong fix

Correcting the lookup to `WHERE allocation_id = %s` makes the route work and
simultaneously makes it a **cross-tenant capability**: `gpu_allocations` had no
owner column, so any caller holding `marketplace:write` could release anyone's
allocation by id and free a GPU out from under a running job.

The no-op was the only thing preventing that. So migration 101 adds `owner_id`
first, and the lookup is corrected only because the column now makes it safe.
This file asserts both halves — that it works, and that it works only for you.

## Why a real database

The property is a `WHERE` clause. A fake that returns a canned row proves the
Python around the query and nothing about the query, and the query *is* the
authorization here.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _cols = {
            r[0]
            for r in _c.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'gpu_allocations'"
            ).fetchall()
        }
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no pg pool: {_e}")
    _pool = None
    _cols = set()
else:
    if "owner_id" not in _cols:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 101")


@pytest.fixture
def allocation():
    """One offer and one allocation owned by a tenant this test invents."""
    tag = uuid.uuid4().hex[:10]
    owner = f"owner-{tag}"
    offer_id = f"offer-{tag}"
    allocation_id = f"alloc-{tag}"
    now = time.time()
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO gpu_offers (offer_id, provider_id, host_id, gpu_model, vram_gb, "
            "gpu_count_total, gpu_count_available, ask_cents_per_hour, available, "
            "created_at, updated_at) "
            "VALUES (%s, %s, %s, 'RTX 4090', 24, 4, 2, 100, false, %s, %s)",
            (offer_id, f"prov-{tag}", f"host-{tag}", now, now),
        )
        conn.execute(
            "INSERT INTO gpu_allocations (allocation_id, offer_id, job_id, gpu_count, "
            "price_cents_per_hour, allocation_type, created_at, released_at, owner_id) "
            "VALUES (%s, %s, %s, 2, 100, 'on_demand', %s, 0, %s)",
            (allocation_id, offer_id, f"job-{tag}", now, owner),
        )
        conn.commit()
    yield {"allocation_id": allocation_id, "offer_id": offer_id, "owner": owner}
    with _pool.connection() as conn:
        conn.execute("DELETE FROM gpu_allocations WHERE allocation_id = %s", (allocation_id,))
        conn.execute("DELETE FROM gpu_offers WHERE offer_id = %s", (offer_id,))
        conn.commit()


def _released_at(allocation_id: str) -> float:
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT released_at FROM gpu_allocations WHERE allocation_id = %s",
            (allocation_id,),
        ).fetchone()
    return float(row[0] if not isinstance(row, dict) else row["released_at"])


def test_the_owner_can_release_and_it_actually_releases(allocation):
    """The half that was silently broken."""
    from marketplace import get_marketplace_engine

    assert _released_at(allocation["allocation_id"]) == 0
    released = get_marketplace_engine().release_allocation_by_id(
        allocation["allocation_id"], owner_id=allocation["owner"]
    )
    assert released is True
    assert _released_at(allocation["allocation_id"]) > 0, (
        "release reported success without writing released_at — the exact "
        "shape of the original bug"
    )


def test_the_gpu_goes_back_on_the_market(allocation):
    """Releasing is two writes that must agree. Only one of them is the row."""
    from marketplace import get_marketplace_engine

    get_marketplace_engine().release_allocation_by_id(
        allocation["allocation_id"], owner_id=allocation["owner"]
    )
    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT gpu_count_available, available FROM gpu_offers WHERE offer_id = %s",
            (allocation["offer_id"],),
        ).fetchone()
    available = row["gpu_count_available"] if isinstance(row, dict) else row[0]
    flag = row["available"] if isinstance(row, dict) else row[1]
    assert available == 4, "the offer's freed GPUs were not restored"
    assert flag is True


def test_another_tenant_cannot_release_it(allocation):
    """The reason migration 101 had to come before the lookup was corrected."""
    from marketplace import get_marketplace_engine

    released = get_marketplace_engine().release_allocation_by_id(
        allocation["allocation_id"], owner_id="some-other-tenant"
    )
    assert released is False
    assert _released_at(allocation["allocation_id"]) == 0, (
        "a different tenant released this allocation — correcting the lookup "
        "without the owner column would have shipped exactly this"
    )


def test_an_empty_owner_releases_nothing(allocation):
    """A caller with no resolvable tenant must not become a wildcard.

    `owner_id=""` against `WHERE owner_id = ''` would match nothing anyway; this
    asserts the guard rather than relying on that coincidence, because a future
    refactor to `WHERE owner_id = COALESCE(%s, owner_id)` would silently make it
    match everything.
    """
    from marketplace import get_marketplace_engine

    assert (
        get_marketplace_engine().release_allocation_by_id(
            allocation["allocation_id"], owner_id=""
        )
        is False
    )
    assert _released_at(allocation["allocation_id"]) == 0


def test_releasing_twice_reports_the_second_as_nothing_to_do(allocation):
    """Idempotent, and honest about which call did the work."""
    from marketplace import get_marketplace_engine

    me = get_marketplace_engine()
    assert me.release_allocation_by_id(
        allocation["allocation_id"], owner_id=allocation["owner"]
    ) is True
    assert me.release_allocation_by_id(
        allocation["allocation_id"], owner_id=allocation["owner"]
    ) is False


def test_an_unknown_allocation_is_not_a_success(allocation):
    from marketplace import get_marketplace_engine

    assert (
        get_marketplace_engine().release_allocation_by_id(
            "alloc-does-not-exist", owner_id=allocation["owner"]
        )
        is False
    )


def test_the_route_no_longer_calls_the_job_id_variant():
    """The defect stated as source, because the route is where it was visible.

    `release_allocation` takes a job id. If the route ever calls it again with
    an allocation id, the silent no-op is back and every assertion above still
    passes, because they exercise the engine rather than the handler.
    """
    import inspect

    import routes.marketplace as mp

    source = inspect.getsource(mp.api_marketplace_release)
    assert "release_allocation_by_id" in source
    assert "me.release_allocation(" not in source, (
        "the route is calling the job-id variant with an allocation id again"
    )
