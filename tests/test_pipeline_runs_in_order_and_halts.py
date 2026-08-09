"""Stages run in order, and a failure stops the ones after it.

B1 of `docs/pipeline-plan.md`, and the half of Gate P4 that says *"a
mid-pipeline failure does not silently continue; the declared failure semantics
are what happens."*

The failure is **caused**, not simulated by inspecting the executor: a handler
returns `ok: False` on stage 2 and the assertions are about what the database
holds afterwards. A test that read the code for the word `break` would pass
against an executor that broke out of the wrong loop.

## Why `skipped` rather than leaving them pending

A halted pipeline marks its remaining stages `skipped` with a reason. Left
`pending`, "why did stage 3 never run" is answerable only by inferring it from
silence — and a sweep looking for work to do would later pick them up and run
them, which is the opposite of halting.

## The concurrency case is not decoration

Two executors reaching one pipeline is ordinary: a retry, a sweep, two API
replicas. Without `FOR UPDATE SKIP LOCKED` both claim stage 1, and a training
stage runs twice and bills twice. That is asserted by racing two claims in
separate transactions rather than by trusting the SQL to say so.
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


THREE_STAGES = [
    {"name": "train", "action_type": "create_instance", "estimate_micros": 1000},
    {"name": "evaluate", "action_type": "create_instance", "estimate_micros": 500},
    {"name": "serve", "action_type": "create_serverless_endpoint", "estimate_micros": 250},
]


@pytest.fixture
def pipeline():
    tag = uuid.uuid4().hex[:10]
    ids = {"plan_id": f"plan-{tag}", "tenant_id": f"tenant-{tag}"}
    yield ids
    with pg_transaction() as conn:
        conn.execute("DELETE FROM pipeline_stages WHERE plan_id = %s", (ids["plan_id"],))


def _states(plan_id: str) -> list[tuple]:
    with pg_transaction() as conn:
        return [
            (r[0], r[1], r[2])
            for r in conn.execute(
                "SELECT stage_index, state, failure_code FROM pipeline_stages "
                " WHERE plan_id = %s ORDER BY stage_index",
                (plan_id,),
            ).fetchall()
        ]


def test_the_graph_hash_changes_when_a_stage_is_edited():
    """Gate P4: editing any stage after approval invalidates it.

    Asserted on the hash rather than end to end, because that hash is what
    `action_plans` compares — the binding is inherited, not reimplemented.
    """
    from control_plane.pipelines import canonical_graph

    _, before = canonical_graph(THREE_STAGES)
    edited = [dict(s) for s in THREE_STAGES]
    edited[1]["action_type"] = "evict_host_workloads"
    _, after = canonical_graph(edited)
    assert before != after, "an edited stage produced the same hash; the approval would survive it"


def test_the_hash_is_stable_across_key_order():
    """A dict that serialises differently on two machines must not self-invalidate."""
    from control_plane.pipelines import canonical_graph

    a = [{"name": "train", "action_type": "create_instance", "estimate_micros": 1}]
    b = [{"estimate_micros": 1, "action_type": "create_instance", "name": "train"}]
    assert canonical_graph(a)[1] == canonical_graph(b)[1]


def test_all_stages_succeed_in_index_order(pipeline):
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph(THREE_STAGES)
    seen: list[str] = []

    def handler(stage):
        seen.append(stage["name"])
        return {"ok": True, "result_ref": f"ref-{stage['stage_index']}"}

    result = run_pipeline(pipeline["plan_id"], pipeline["tenant_id"], stages, handler)

    assert seen == ["train", "evaluate", "serve"], f"stages ran out of order: {seen}"
    assert result["finished"] is True
    assert result["failed"] is False
    assert [s[1] for s in _states(pipeline["plan_id"])] == ["succeeded"] * 3


def test_a_failure_halts_and_the_rest_are_skipped(pipeline):
    """The clause. Caused, not simulated."""
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph(THREE_STAGES)
    seen: list[str] = []

    def handler(stage):
        seen.append(stage["name"])
        if stage["name"] == "evaluate":
            return {"ok": False, "failure_code": "eval_regressed"}
        return {"ok": True}

    result = run_pipeline(pipeline["plan_id"], pipeline["tenant_id"], stages, handler)

    assert seen == ["train", "evaluate"], (
        f"a stage ran after the halt: {seen} — the pipeline continued past a "
        "failure the graph said to stop on"
    )
    rows = _states(pipeline["plan_id"])
    assert [r[1] for r in rows] == ["succeeded", "failed", "skipped"]
    assert rows[1][2] == "eval_regressed"
    assert rows[2][2].startswith("halted_after_stage_"), (
        "the skipped stage carries no reason, so 'why did serve never run' is "
        "answerable only by inferring it from silence"
    )
    assert result["failed"] is True


def test_a_handler_that_raises_is_a_failed_stage_not_a_crash(pipeline):
    """An exception in someone's stage must not lose the pipeline's record."""
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph(THREE_STAGES)

    def handler(stage):
        if stage["name"] == "train":
            raise ValueError("boom")
        return {"ok": True}

    run_pipeline(pipeline["plan_id"], pipeline["tenant_id"], stages, handler)
    rows = _states(pipeline["plan_id"])
    assert rows[0][1] == "failed"
    assert "handler_error" in (rows[0][2] or "")
    assert [r[1] for r in rows[1:]] == ["skipped", "skipped"]


def test_two_executors_do_not_claim_the_same_stage(pipeline):
    """Without SKIP LOCKED, a training stage runs twice and bills twice."""
    from control_plane.pipelines import claim_next_stage, materialise_stages, canonical_graph

    stages, _ = canonical_graph(THREE_STAGES)
    with pg_transaction() as conn:
        materialise_stages(conn, pipeline["plan_id"], pipeline["tenant_id"], stages)

    # Two open transactions, both reaching for the next stage.
    with pg_transaction() as conn_a:
        first = claim_next_stage(conn_a, pipeline["plan_id"])
        with pg_transaction() as conn_b:
            second = claim_next_stage(conn_b, pipeline["plan_id"])

    assert first is not None
    assert second is None or second["stage_index"] != first["stage_index"], (
        "two executors claimed the same stage — it would run twice"
    )


def test_materialising_twice_does_not_duplicate_the_graph(pipeline):
    from control_plane.pipelines import canonical_graph, materialise_stages

    stages, _ = canonical_graph(THREE_STAGES)
    with pg_transaction() as conn:
        first = materialise_stages(conn, pipeline["plan_id"], pipeline["tenant_id"], stages)
    with pg_transaction() as conn:
        again = materialise_stages(conn, pipeline["plan_id"], pipeline["tenant_id"], stages)
    assert first == 3
    assert again == 0, "a second executor duplicated the graph"


@pytest.mark.parametrize("mode", ["continue", "retry"])
def test_an_unimplemented_failure_mode_is_refused_not_downgraded(mode):
    """B1 implements `halt` only, and says so at creation.

    Silently treating `continue` as `halt` would give the user a pipeline whose
    declared behaviour is not its actual behaviour — they would believe a
    failure is tolerated when it stops the run.
    """
    from control_plane.pipelines import PipelineError, canonical_graph

    stages = [dict(THREE_STAGES[0], on_failure=mode)]
    with pytest.raises(PipelineError) as excinfo:
        canonical_graph(stages)
    assert excinfo.value.code == "on_failure_not_implemented"


def test_an_empty_graph_is_refused():
    from control_plane.pipelines import PipelineError, canonical_graph

    with pytest.raises(PipelineError) as excinfo:
        canonical_graph([])
    assert excinfo.value.code == "empty_graph"


def test_a_stage_without_an_action_type_is_refused():
    from control_plane.pipelines import PipelineError, canonical_graph

    with pytest.raises(PipelineError):
        canonical_graph([{"name": "train"}])


# ── B2: the spend ceiling ───────────────────────────────────────────


def test_a_stage_that_would_exceed_the_ceiling_never_starts(pipeline):
    """Gate P4: "spend is bounded by what was approved."

    The stage is *skipped*, not failed — nothing went wrong with it, there was
    no room left for it — and crucially the handler is never called. A ceiling
    enforced after the fact is a report, not a bound.
    """
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph(THREE_STAGES)   # quotes: 1000, 500, 250
    ran: list[str] = []

    def handler(stage):
        ran.append(stage["name"])
        return {"ok": True, "spent_micros": stage["estimate_micros"]}

    # Room for train (1000) and evaluate (500) but not serve (250 more = 1750).
    result = run_pipeline(
        pipeline["plan_id"], pipeline["tenant_id"], stages, handler,
        approved_micros=1600,
    )

    assert ran == ["train", "evaluate"], (
        f"a stage ran past the approved ceiling: {ran}"
    )
    rows = _states(pipeline["plan_id"])
    assert [r[1] for r in rows] == ["succeeded", "succeeded", "skipped"]
    assert rows[2][2] == "budget_exceeded"
    assert result["finished"] is True


def test_an_overrunning_stage_eats_into_what_remains(pipeline):
    """Actual spend, not the original estimate.

    If the ceiling were compared against the *quotes*, a stage that came in at
    triple its estimate would leave the bound intact on paper while the pipeline
    overspent in fact. That is the mispricing case, and it is the one where a
    ceiling has to bite.
    """
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph(THREE_STAGES)
    ran: list[str] = []

    def handler(stage):
        ran.append(stage["name"])
        # train quotes 1000 and actually costs 1500.
        return {"ok": True, "spent_micros": 1500 if stage["name"] == "train" else 0}

    run_pipeline(
        pipeline["plan_id"], pipeline["tenant_id"], stages, handler,
        approved_micros=1600,
    )

    assert ran == ["train"], (
        f"the pipeline continued after a stage overran its quote: {ran} — the "
        "ceiling was compared against estimates rather than actual spend"
    )
    assert _states(pipeline["plan_id"])[1][2] == "budget_exceeded"


def test_no_ceiling_means_no_refusal(pipeline):
    """A pipeline created without a quote must not be silently capped at zero.

    Otherwise every such pipeline fails its first stage for a reason nobody can
    see, which is worse than having no bound.
    """
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph(THREE_STAGES)
    result = run_pipeline(
        pipeline["plan_id"], pipeline["tenant_id"], stages,
        lambda s: {"ok": True, "spent_micros": 10_000},
        approved_micros=0,
    )
    assert result["failed"] is False
    assert [r[1] for r in _states(pipeline["plan_id"])] == ["succeeded"] * 3


def test_the_ceiling_is_checked_before_the_handler_not_after(pipeline):
    """Stated as its own test because it is the whole difference.

    A ceiling checked after the handler would let the stage run, spend the
    money, and then report that it should not have.
    """
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph(THREE_STAGES)
    calls: list[str] = []

    run_pipeline(
        pipeline["plan_id"], pipeline["tenant_id"], stages,
        lambda s: (calls.append(s["name"]), {"ok": True})[1],
        approved_micros=1,   # not even the first stage fits
    )
    assert calls == [], "the handler ran for a stage the ceiling had already refused"
    assert [r[1] for r in _states(pipeline["plan_id"])] == ["skipped"] * 3
