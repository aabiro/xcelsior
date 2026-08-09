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


def test_every_declared_failure_mode_is_actually_implemented():
    """This test used to assert the opposite, and that was right at the time.

    Through B1–B2 only `halt` was executed, and a graph declaring `continue` or
    `retry` was **refused** rather than quietly downgraded — a pipeline whose
    declared behaviour is not its actual behaviour is worse than one that will
    not start. B3 implements them, so the assertion inverts: the schema and the
    executor must now agree.

    Kept rather than deleted because the failure it guards has simply moved. A
    fourth mode added to the CHECK constraint without an executor branch would
    be accepted at creation and then silently behave as something else, which
    is the original bug wearing a different name.
    """
    from control_plane.pipelines import DECLARED_ON_FAILURE, IMPLEMENTED_ON_FAILURE

    unimplemented = DECLARED_ON_FAILURE - IMPLEMENTED_ON_FAILURE
    assert not unimplemented, (
        f"the schema permits {sorted(unimplemented)} but the executor has no "
        "branch for it — a graph would be accepted and then behave as "
        "something the user did not declare"
    )


@pytest.mark.parametrize("mode", ["nonsense", "abort", "HALT"])
def test_a_mode_the_schema_does_not_define_is_refused(mode):
    """The vocabulary check outlives the implementation gap.

    `"HALT"` is here on purpose: a mode differing only in case would pass a
    sloppy check and then violate the database CHECK at insert time, turning a
    clear refusal into a crash partway through creating a pipeline.
    """
    from control_plane.pipelines import PipelineError, canonical_graph

    with pytest.raises(PipelineError) as excinfo:
        canonical_graph([dict(THREE_STAGES[0], on_failure=mode)])
    assert excinfo.value.code == "invalid_on_failure"


def test_surrounding_whitespace_is_tolerated_not_refused():
    """`" continue "` is a typo, not a different policy.

    Split from the case above after I put it in the refused list while the
    docstring said it was accepted — the note and the assertion contradicted
    each other in the same breath. `.strip()` runs before the membership check,
    which is the right behaviour: refusing on whitespace would reject a valid
    intent for a reason the user cannot see in their own input.
    """
    from control_plane.pipelines import canonical_graph

    stages, _ = canonical_graph([dict(THREE_STAGES[0], on_failure="  continue  ")])
    assert stages[0]["on_failure"] == "continue"


def test_an_omitted_failure_mode_defaults_to_the_safe_one():
    """Absent or empty means `halt`, not a refusal.

    My first version of the test above asserted that `""` was refused. It is
    not, and should not be: defaulting an unstated failure policy to the mode
    that *stops* is the safe direction, and it matches the behaviour of leaving
    the key out entirely. The code was right and the test was wrong.
    """
    from control_plane.pipelines import canonical_graph

    absent = dict(THREE_STAGES[0])
    absent.pop("on_failure", None)
    assert canonical_graph([absent])[0][0]["on_failure"] == "halt"
    assert canonical_graph([dict(THREE_STAGES[0], on_failure="")])[0][0]["on_failure"] == "halt"


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


# ── B3: continue and retry ──────────────────────────────────────────


def test_continue_proceeds_past_a_failed_stage(pipeline):
    """The declared semantics are what happens — the other direction.

    `halt` stopping is only half the clause. If `continue` also stopped, the
    graph's declaration would be decorative and the user's choice would mean
    nothing.
    """
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph([
        dict(THREE_STAGES[0]),
        dict(THREE_STAGES[1], on_failure="continue"),
        dict(THREE_STAGES[2]),
    ])
    ran: list[str] = []

    def handler(stage):
        ran.append(stage["name"])
        return {"ok": stage["name"] != "evaluate"}

    result = run_pipeline(pipeline["plan_id"], pipeline["tenant_id"], stages, handler)

    assert ran == ["train", "evaluate", "serve"], (
        f"`continue` did not continue: {ran}"
    )
    assert [r[1] for r in _states(pipeline["plan_id"])] == [
        "succeeded", "failed", "succeeded",
    ]
    assert result["failed"] is True, "a pipeline containing a failed stage reported clean"


def test_retry_reattempts_then_succeeds(pipeline):
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph([
        dict(THREE_STAGES[0], on_failure="retry", max_attempts=3),
        dict(THREE_STAGES[1]),
    ])
    attempts: list[str] = []

    def handler(stage):
        attempts.append(stage["name"])
        # train fails once, then works.
        if stage["name"] == "train" and attempts.count("train") == 1:
            return {"ok": False, "failure_code": "flaky"}
        return {"ok": True}

    result = run_pipeline(pipeline["plan_id"], pipeline["tenant_id"], stages, handler)

    assert attempts == ["train", "train", "evaluate"], f"retry did not re-attempt: {attempts}"
    assert [r[1] for r in _states(pipeline["plan_id"])] == ["succeeded", "succeeded"]
    assert result["failed"] is False


def test_an_exhausted_retry_halts_rather_than_continuing(pipeline):
    """The decision worth arguing with, so it is asserted rather than assumed.

    A stage that failed every attempt it was allowed did not work. Falling
    through to the next stage would make `retry` mean "try a few times and then
    ignore the result", which nobody would approve if it were written that way.
    """
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph([
        dict(THREE_STAGES[0], on_failure="retry", max_attempts=2),
        dict(THREE_STAGES[1]),
    ])
    attempts: list[str] = []

    def handler(stage):
        attempts.append(stage["name"])
        return {"ok": False, "failure_code": "always_broken"}

    run_pipeline(pipeline["plan_id"], pipeline["tenant_id"], stages, handler)

    assert attempts == ["train", "train"], f"wrong attempt count: {attempts}"
    rows = _states(pipeline["plan_id"])
    assert rows[0][1] == "failed"
    assert rows[1][1] == "skipped", "an exhausted retry let the pipeline continue"
    assert rows[1][2].startswith("retries_exhausted"), (
        "the reason does not distinguish an exhausted retry from an ordinary "
        "halt, so the audit chain cannot say which happened"
    )


def test_retry_never_exceeds_max_attempts(pipeline):
    """The CHECK constraint enforces this too; the executor must not rely on
    hitting it, because a constraint violation is a crash rather than a halt."""
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph([dict(THREE_STAGES[0], on_failure="retry", max_attempts=3)])
    attempts = []
    run_pipeline(
        pipeline["plan_id"], pipeline["tenant_id"], stages,
        lambda s: (attempts.append(1), {"ok": False})[1],
    )
    assert len(attempts) == 3, f"ran {len(attempts)} attempts against max_attempts=3"

    with pg_transaction() as conn:
        count = conn.execute(
            "SELECT attempt_count FROM pipeline_stages WHERE plan_id = %s AND stage_index = 0",
            (pipeline["plan_id"],),
        ).fetchone()[0]
    assert count == 3


def test_a_failed_attempt_still_counts_against_the_ceiling(pipeline):
    """Otherwise a retrying stage spends without limit inside a bounded pipeline.

    This is the hole `max_attempts` closes from one side and the banked spend
    closes from the other: bounded attempts alone would still allow three
    expensive failures inside a ceiling meant for one success.
    """
    from control_plane.pipelines import canonical_graph, run_pipeline

    stages, _ = canonical_graph([
        dict(THREE_STAGES[0], on_failure="retry", max_attempts=5, estimate_micros=100),
        dict(THREE_STAGES[1], estimate_micros=100),
    ])
    attempts = []

    def handler(stage):
        attempts.append(stage["name"])
        return {"ok": False, "spent_micros": 400, "failure_code": "expensive"}

    run_pipeline(
        pipeline["plan_id"], pipeline["tenant_id"], stages, handler,
        approved_micros=900,
    )

    # Each failed attempt banks 400; the ceiling stops it before five.
    assert len(attempts) < 5, (
        f"a retrying stage ran all {len(attempts)} attempts despite banking "
        "spend past the ceiling"
    )
