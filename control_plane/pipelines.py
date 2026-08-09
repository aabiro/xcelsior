"""Sequential pipeline execution. B1 of `docs/pipeline-plan.md`.

One approved `action_plans` row carries the graph; `pipeline_stages` records
what happened to each stage. This module is the executor between them: claim the
next stage, run it, record the outcome, and stop when the graph says to stop.

## Why the executor claims a stage rather than iterating

Two callers can reach the same pipeline — a retry, a sweep, two API replicas.
`claim_next_stage` selects `FOR UPDATE SKIP LOCKED` and flips `pending →
running` inside the transaction, so exactly one caller gets a given stage. An
executor that iterated a list would run stage 2 twice under concurrency and
charge for both.

## Why failure semantics live on the row

`on_failure` was fixed when the user approved the graph (§3.2). Reading it here
rather than accepting it from the caller is what makes "halt" mean halt: a
caller that could pass its own failure policy could turn a halt into a continue
at exactly the moment that matters.

## What is here, and what is not

All three failure modes execute (B3). The spend ceiling is enforced **before**
each stage rather than reconciled after it (B2), so a stage that would exceed
the approved total never starts.

Not here: the tool (B4).

Through B1–B2 this section said `continue` and `retry` were *refused* at
creation rather than downgraded to `halt`, on the grounds that a graph whose
declared behaviour is not its actual behaviour is worse than one that will not
start. That refusal has done its job and gone. What replaces it is a test
asserting `DECLARED_ON_FAILURE == IMPLEMENTED_ON_FAILURE`, so a fourth mode
added to the schema without an executor branch fails the build instead of
silently behaving as something else.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Iterable

from control_plane.db import control_plane_transaction

#: Failure semantics the executor can actually honour. This was `{"halt"}`
#: through B1–B2, and a graph declaring the others was refused rather than
#: quietly downgraded; B3 implements them, so the refusal falls away on its own.
#:
#: **An exhausted `retry` halts.** It does not fall through to `continue`. A
#: stage that failed every attempt it was allowed is a stage that did not work,
#: and letting the pipeline proceed past it would make `retry` mean "try a few
#: times and then ignore the result" — which nobody would approve if it were
#: written that way.
IMPLEMENTED_ON_FAILURE = frozenset({"halt", "continue", "retry"})

#: Every failure mode the schema permits. The gap between this and
#: `IMPLEMENTED_ON_FAILURE` is the honest statement of what is left to build.
DECLARED_ON_FAILURE = frozenset({"halt", "continue", "retry"})

TERMINAL_STAGE_STATES = frozenset({"succeeded", "failed", "skipped"})


class PipelineError(RuntimeError):
    """A pipeline could not be created or advanced."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical_graph(stages: Iterable[dict]) -> tuple[list[dict], str]:
    """Normalise a graph and hash it.

    The hash is what binds the approval: Gate P4 requires that editing any stage
    after approval invalidates it, and that falls out of `action_plans`'
    existing `canonical_args_hash` check rather than needing its own mechanism.

    Key order is normalised because a dict that serialises differently on two
    machines would invalidate its own approval for no reason.
    """
    normalised = []
    for index, stage in enumerate(stages):
        name = str(stage.get("name") or "").strip()
        action_type = str(stage.get("action_type") or "").strip()
        if not name or not action_type:
            raise PipelineError("invalid_stage", f"stage {index} needs a name and an action_type")
        on_failure = str(stage.get("on_failure") or "halt").strip()
        if on_failure not in DECLARED_ON_FAILURE:
            raise PipelineError(
                "invalid_on_failure",
                f"stage {index!r} declares on_failure={on_failure!r}; "
                f"expected one of {sorted(DECLARED_ON_FAILURE)}",
            )
        if on_failure not in IMPLEMENTED_ON_FAILURE:
            # Refused rather than downgraded. A graph whose declared behaviour
            # is not the behaviour it gets is worse than one that will not run:
            # the user believes a failure will be tolerated, and it will not.
            raise PipelineError(
                "on_failure_not_implemented",
                f"on_failure={on_failure!r} is declared but not yet executed "
                f"(B3). Only {sorted(IMPLEMENTED_ON_FAILURE)} runs today, and "
                "a pipeline will not start with semantics it cannot honour.",
            )
        max_attempts = int(stage.get("max_attempts") or 1)
        if max_attempts < 1:
            raise PipelineError("invalid_max_attempts", "max_attempts must be at least 1")
        normalised.append({
            "name": name,
            "action_type": action_type,
            "on_failure": on_failure,
            "max_attempts": max_attempts,
            "args": stage.get("args") or {},
            "estimate_micros": int(stage.get("estimate_micros") or 0),
        })
    if not normalised:
        raise PipelineError("empty_graph", "a pipeline needs at least one stage")

    blob = json.dumps(normalised, sort_keys=True, separators=(",", ":"), default=str)
    return normalised, hashlib.sha256(blob.encode()).hexdigest()


def materialise_stages(conn, plan_id: str, tenant_id: str, stages: list[dict]) -> int:
    """Write the stage rows for an approved plan. Idempotent.

    Called at execution rather than approval: an approved plan that is never run
    should leave no execution state, and `ON CONFLICT DO NOTHING` means a second
    executor arriving at the same moment does not duplicate the graph.
    """
    written = 0
    for index, stage in enumerate(stages):
        cur = conn.execute(
            """INSERT INTO pipeline_stages
                 (plan_id, stage_index, tenant_id, name, action_type,
                  state, on_failure, max_attempts, estimate_micros)
               VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s, %s)
               ON CONFLICT (plan_id, stage_index) DO NOTHING""",
            (
                plan_id, index, tenant_id, stage["name"], stage["action_type"],
                stage["on_failure"], stage["max_attempts"], stage["estimate_micros"],
            ),
        )
        written += cur.rowcount or 0
    return written


def claim_next_stage(conn, plan_id: str) -> dict | None:
    """The lowest-index pending stage, flipped to `running` under lock.

    `SKIP LOCKED` so two executors racing the same pipeline do not both take the
    same stage — the failure that would run a training job twice and bill for
    both. Returns None when there is nothing left to claim.
    """
    row = conn.execute(
        """SELECT stage_index, name, action_type, on_failure, max_attempts,
                  attempt_count, estimate_micros, tenant_id
             FROM pipeline_stages
            WHERE plan_id = %s AND state = 'pending'
            ORDER BY stage_index ASC
            LIMIT 1
              FOR UPDATE SKIP LOCKED""",
        (plan_id,),
    ).fetchone()
    if not row:
        return None
    stage = {
        "stage_index": row[0], "name": row[1], "action_type": row[2],
        "on_failure": row[3], "max_attempts": row[4], "attempt_count": row[5],
        "estimate_micros": row[6], "tenant_id": row[7],
    }
    conn.execute(
        """UPDATE pipeline_stages
              SET state = 'running',
                  attempt_count = attempt_count + 1,
                  started_at = COALESCE(started_at, clock_timestamp()),
                  updated_at = clock_timestamp()
            WHERE plan_id = %s AND stage_index = %s""",
        (plan_id, stage["stage_index"]),
    )
    return stage


def finish_stage(
    conn,
    plan_id: str,
    stage_index: int,
    *,
    state: str,
    result_ref: str | None = None,
    failure_code: str = "",
    spent_micros: int = 0,
) -> None:
    """Record a stage's outcome. `finished_at` is set by the same statement.

    The schema requires a finished stage to carry a finish time; setting it here
    rather than in a later update means there is no window in which a succeeded
    stage exists without one.
    """
    if state not in ("succeeded", "failed", "skipped"):
        raise PipelineError("invalid_state", f"{state!r} is not a terminal stage state")
    conn.execute(
        """UPDATE pipeline_stages
              SET state = %s,
                  result_ref = COALESCE(%s, result_ref),
                  failure_code = NULLIF(%s, ''),
                  spent_micros = spent_micros + %s,
                  finished_at = clock_timestamp(),
                  updated_at = clock_timestamp()
            WHERE plan_id = %s AND stage_index = %s""",
        (state, result_ref, failure_code, int(spent_micros), plan_id, stage_index),
    )


def requeue_stage_for_retry(
    conn,
    plan_id: str,
    stage_index: int,
    *,
    failure_code: str,
    spent_micros: int,
) -> None:
    """Return a failed stage to `pending` so it is claimed again.

    **The spend is banked, not discarded.** A failed attempt cost whatever it
    cost, and the ceiling keeps counting it — otherwise a retrying stage could
    spend without limit inside a bounded pipeline, which is the exact hole the
    `max_attempts` constraint exists to close from the other side.

    `finished_at` stays NULL: the stage has not finished, it is going round
    again. That also keeps it clear of the schema's rule that a succeeded or
    failed stage must carry a finish time.
    """
    conn.execute(
        """UPDATE pipeline_stages
              SET state = 'pending',
                  failure_code = NULLIF(%s, ''),
                  spent_micros = spent_micros + %s,
                  updated_at = clock_timestamp()
            WHERE plan_id = %s AND stage_index = %s""",
        (failure_code, int(spent_micros), plan_id, stage_index),
    )


def halt_remaining(conn, plan_id: str, after_index: int, reason: str) -> int:
    """Mark every later pending stage `skipped`. Returns how many.

    This is what makes `halt` visible rather than merely absent. A pipeline that
    stopped leaves its remaining stages as `skipped` with a reason, so "why did
    stage 3 never run" is answerable from the row instead of by inferring it
    from silence.
    """
    cur = conn.execute(
        """UPDATE pipeline_stages
              SET state = 'skipped',
                  failure_code = %s,
                  finished_at = clock_timestamp(),
                  updated_at = clock_timestamp()
            WHERE plan_id = %s AND stage_index > %s AND state = 'pending'""",
        (reason, plan_id, after_index),
    )
    return cur.rowcount or 0


def pipeline_state(conn, plan_id: str) -> dict:
    """A summary a caller can act on: counts by state, and what is next."""
    rows = conn.execute(
        "SELECT state, count(*) FROM pipeline_stages WHERE plan_id = %s GROUP BY state",
        (plan_id,),
    ).fetchall()
    counts = {r[0]: int(r[1]) for r in rows}
    total = sum(counts.values())
    done = sum(counts.get(s, 0) for s in TERMINAL_STAGE_STATES)
    return {
        "plan_id": plan_id,
        "counts": counts,
        "total": total,
        "finished": total > 0 and done == total,
        "failed": counts.get("failed", 0) > 0,
    }


def spent_so_far(conn, plan_id: str) -> int:
    """Micros this pipeline has actually spent across finished stages."""
    row = conn.execute(
        "SELECT COALESCE(SUM(spent_micros), 0) FROM pipeline_stages WHERE plan_id = %s",
        (plan_id,),
    ).fetchone()
    return int(row[0] or 0)


def would_exceed_ceiling(conn, plan_id: str, stage: dict, approved_micros: int) -> bool:
    """True when running `stage` could take the pipeline past what was approved.

    §3.3. The naive implementation sums the stages and compares once, which
    fails the way that matters: stage 2 overruns, stage 3 starts anyway, and the
    overspend is discovered at the end. The user's ceiling is a promise about
    what *can* be spent, and a promise checked only afterwards is a report.

    Compared against **actual** spend so far plus **this stage's quote**, not
    against the original estimates — a stage that came in over its estimate has
    to eat into what remains, or the ceiling means nothing the moment any single
    stage is mispriced.

    `approved_micros <= 0` means no ceiling was set, and nothing is refused.
    That is deliberate and narrow: a pipeline created without a quote is not
    silently capped at zero, which would make every such pipeline fail its first
    stage for a reason nobody could see.
    """
    if approved_micros <= 0:
        return False
    return spent_so_far(conn, plan_id) + int(stage.get("estimate_micros") or 0) > approved_micros


def run_pipeline(
    plan_id: str,
    tenant_id: str,
    stages: list[dict],
    handler: Callable[[dict], dict],
    *,
    approved_micros: int = 0,
) -> dict:
    """Run stages in order until one fails or all succeed.

    `handler` performs a stage and returns `{"ok": bool, "result_ref": str,
    "failure_code": str, "spent_micros": int}`. It is injected rather than
    imported so the executor's ordering and halt semantics can be tested by
    causing a failure, not by reading the code.

    Each stage is claimed, run, and recorded in **its own transaction**. Holding
    one transaction across a training stage would keep a database lock open for
    hours; more importantly, a crash mid-pipeline must leave the stages that
    finished recorded as finished.
    """
    with control_plane_transaction() as conn:
        materialise_stages(conn, plan_id, tenant_id, stages)

    while True:
        with control_plane_transaction() as conn:
            stage = claim_next_stage(conn, plan_id)
        if stage is None:
            break

        # The ceiling is checked here — after the stage is claimed, so the
        # refusal names which stage, but *before* the handler runs, so the stage
        # never starts. Marked `skipped` rather than `failed`: nothing went
        # wrong with it, there was no room left for it.
        with control_plane_transaction() as conn:
            over_budget = would_exceed_ceiling(conn, plan_id, stage, approved_micros)
            if over_budget:
                finish_stage(
                    conn, plan_id, stage["stage_index"],
                    state="skipped", failure_code="budget_exceeded",
                )
                halt_remaining(
                    conn, plan_id, stage["stage_index"], "budget_exceeded"
                )
        if over_budget:
            break

        try:
            outcome = handler(stage) or {}
        except Exception as exc:  # a handler that raises is a failed stage
            outcome = {"ok": False, "failure_code": f"handler_error:{type(exc).__name__}"}

        ok = bool(outcome.get("ok"))
        spent = int(outcome.get("spent_micros") or 0)
        failure_code = str(outcome.get("failure_code") or ("" if ok else "stage_failed"))

        # `claim_next_stage` already incremented, and returned the value from
        # *before* the increment — so this attempt is the (attempt_count + 1)th.
        # Getting that off by one would either burn an attempt or allow one past
        # the CHECK, which is why it is spelled out rather than inlined.
        attempts_used = int(stage["attempt_count"]) + 1
        attempts_left = int(stage["max_attempts"]) - attempts_used
        mode = stage["on_failure"]

        if not ok and mode == "retry" and attempts_left > 0:
            # Back to `pending` so the next loop claims it again. A failed
            # attempt still spent whatever it spent, so the ceiling keeps
            # counting it — otherwise a retrying stage could spend without limit
            # inside a bounded pipeline.
            with control_plane_transaction() as conn:
                requeue_stage_for_retry(
                    conn, plan_id, stage["stage_index"],
                    failure_code=failure_code, spent_micros=spent,
                )
            continue

        with control_plane_transaction() as conn:
            finish_stage(
                conn, plan_id, stage["stage_index"],
                state="succeeded" if ok else "failed",
                result_ref=outcome.get("result_ref"),
                failure_code=failure_code,
                spent_micros=spent,
            )
            if not ok and mode != "continue":
                # `halt`, and an exhausted `retry`. The latter does **not** fall
                # through to `continue`: a stage that failed every attempt it
                # was allowed is a stage that did not work, and proceeding past
                # it would make `retry` mean "try a few times, then ignore the
                # result" — which nobody would approve if it were written that
                # way.
                reason = (
                    f"halted_after_stage_{stage['stage_index']}"
                    if mode == "halt"
                    else f"retries_exhausted_at_stage_{stage['stage_index']}"
                )
                halt_remaining(conn, plan_id, stage["stage_index"], reason)
                break

    with control_plane_transaction() as conn:
        return pipeline_state(conn, plan_id)
