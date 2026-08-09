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

## What B1 deliberately does not do

No budget ceiling (B2), no `continue`/`retry` semantics (B3), no tool (B4).
`halt` is the only failure mode implemented, and the others are *rejected* at
creation rather than silently treated as halt — a graph whose declared
behaviour is not the behaviour it gets is worse than one that will not start.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Iterable

from control_plane.db import control_plane_transaction

#: Failure semantics the executor can actually honour today. `continue` and
#: `retry` are B3; until then a graph declaring them is refused at creation
#: rather than quietly downgraded to `halt`.
IMPLEMENTED_ON_FAILURE = frozenset({"halt"})

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


def run_pipeline(
    plan_id: str,
    tenant_id: str,
    stages: list[dict],
    handler: Callable[[dict], dict],
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

        try:
            outcome = handler(stage) or {}
        except Exception as exc:  # a handler that raises is a failed stage
            outcome = {"ok": False, "failure_code": f"handler_error:{type(exc).__name__}"}

        ok = bool(outcome.get("ok"))
        with control_plane_transaction() as conn:
            finish_stage(
                conn, plan_id, stage["stage_index"],
                state="succeeded" if ok else "failed",
                result_ref=outcome.get("result_ref"),
                failure_code=str(outcome.get("failure_code") or ("" if ok else "stage_failed")),
                spent_micros=int(outcome.get("spent_micros") or 0),
            )
            if not ok and stage["on_failure"] == "halt":
                halt_remaining(
                    conn, plan_id, stage["stage_index"],
                    f"halted_after_stage_{stage['stage_index']}",
                )
                break

    with control_plane_transaction() as conn:
        return pipeline_state(conn, plan_id)
