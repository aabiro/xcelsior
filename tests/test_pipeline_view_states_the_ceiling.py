"""The live pipeline view can state spend against the ceiling that bounds it.

Gate P4's frontend clause: *"a pipeline view showing the graph, which stage is
live, and one approval covering all of it — with the total committed spend
stated **before** approval, not after."*

`POST /api/v1/pipelines` already returned `approved_max_micros` — the ceiling,
stated at quote time, before anyone approves anything. `GET
/api/v1/pipelines/{plan_id}` returned per-stage state and **not** the ceiling,
so a view rendering a live run had the spend and nothing to measure it against.

The obvious workaround is worse than the gap: fetch the ceiling from the plan
endpoint and the spend from the pipeline endpoint, and render one against the
other. Those are two reads at two moments, and "over budget" is the wrong thing
to be wrong about — a stale ceiling beside a fresh spend draws a bar past full
and tells the user a guarantee failed when it did not.

So both come from one call, computed inside one transaction.

## What this asserts, and what it deliberately does not

It asserts the two numbers arrive together and mean what the executor means by
them — `spent_micros` is `spent_so_far`, the same sum §3.3's pre-stage check
uses, so the view's arithmetic and the enforcement's arithmetic cannot diverge.

It does not assert the ceiling is *enforced*; that is
`tests/test_pipeline_runs_in_order_and_halts.py`, which causes the overrun
rather than reading the code. This file is about what the view can say.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = _c.execute("SELECT to_regclass('pipeline_stages')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 104")

from control_plane.pipelines import (  # noqa: E402
    canonical_graph,
    finish_stage,
    materialise_stages,
    spent_so_far,
)

THREE_STAGES = [
    {"name": "train", "action_type": "create_instance", "estimate_micros": 4_000_000},
    {"name": "evaluate", "action_type": "create_instance", "estimate_micros": 1_000_000},
    {"name": "serve", "action_type": "create_serverless_endpoint", "estimate_micros": 2_000_000},
]


@pytest.fixture
def pipeline():
    tag = uuid.uuid4().hex[:10]
    ids = {"plan_id": f"plan-{tag}", "tenant_id": f"tenant-{tag}"}
    yield ids
    with pg_transaction() as conn:
        conn.execute("DELETE FROM pipeline_stages WHERE plan_id = %s", (ids["plan_id"],))


def test_the_route_returns_the_ceiling_and_the_spend_together():
    """Read the route's source: both keys, from one handler, one transaction."""
    import inspect

    from routes import action_plans

    source = inspect.getsource(action_plans.api_get_pipeline)
    for key in ("approved_max_micros", "spent_micros", "currency"):
        assert f'"{key}"' in source, (
            f"GET /api/v1/pipelines/{{plan_id}} no longer returns {key}. A live "
            "view then has to source it from a second endpoint at a second "
            "moment, which is how a stale ceiling gets drawn against a fresh "
            "spend."
        )
    # Both inside the same `with` block — a spend read after the transaction
    # closed is a spend from a different instant than the ceiling beside it.
    #
    # Counted, not ordered. The first version of this asserted that
    # `spent_so_far` appeared before `return {` in the source, which stays true
    # when the spend is moved into a second transaction — it was a guard that
    # could not fail for the reason its message gave. Its own positive control
    # is what caught that.
    assert source.count("control_plane_transaction()") == 1, (
        "the handler opens more than one transaction, so the ceiling and the "
        "spend are read at two different instants. A stale ceiling beside a "
        "fresh spend draws the bar past full and tells the user a guarantee "
        f"failed when it did not. Opens: {source.count('control_plane_transaction()')}"
    )


def test_the_view_spend_is_the_same_sum_the_ceiling_check_uses(pipeline):
    """The view's number and the enforcement's number are one function.

    If the route summed `spent_micros` itself, the view could disagree with
    `would_exceed_ceiling` about how much has been spent — and the user would be
    reading a different budget than the one being enforced.
    """
    plan_id, tenant_id = pipeline["plan_id"], pipeline["tenant_id"]
    with pg_transaction() as conn:
        stages, _ = canonical_graph(THREE_STAGES)
        materialise_stages(conn, plan_id, tenant_id, stages)

    with pg_transaction() as conn:
        assert spent_so_far(conn, plan_id) == 0

    with pg_transaction() as conn:
        finish_stage(conn, plan_id, 0, state="succeeded", spent_micros=3_500_000)

    with pg_transaction() as conn:
        assert spent_so_far(conn, plan_id) == 3_500_000

    import inspect

    from routes import action_plans

    source = inspect.getsource(action_plans.api_get_pipeline)
    assert "spent_so_far(conn, plan_id)" in source, (
        "the route computes spend some other way than `spent_so_far`; the view "
        "and the pre-stage ceiling check must agree by construction, not by two "
        "queries that happen to match today"
    )


def test_spend_counts_a_failed_attempt_too(pipeline):
    """A view that showed only successful spend would understate the budget.

    §3.2's bounded retry exists because attempts cost money whether or not they
    work. `spent_so_far` sums every stage row, so the bar reflects what was
    actually spent rather than what was spent usefully.
    """
    plan_id, tenant_id = pipeline["plan_id"], pipeline["tenant_id"]
    with pg_transaction() as conn:
        stages, _ = canonical_graph(THREE_STAGES)
        materialise_stages(conn, plan_id, tenant_id, stages)
        finish_stage(conn, plan_id, 0, state="failed", spent_micros=900_000, failure_code="boom")

    with pg_transaction() as conn:
        assert spent_so_far(conn, plan_id) == 900_000
