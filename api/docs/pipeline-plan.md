# One approval for a dependency graph

*Plan phase P4. `train → evaluate → serve` approved once, executed in order,
with failure semantics that are stated rather than discovered.*

## 0. Directive to the implementer

The six prohibitions in `docs/artifact-promotion-plan.md` §0 apply unchanged.
Three more belong to this phase specifically:

1. **Never execute a stage the approval did not cover.** Not "a stage very like
   it", not "the same stage with a corrected parameter". If the graph changed,
   the approval is void and a new one is required.
2. **Never let a pipeline outspend its own quote.** The number shown before
   approval is a ceiling, not an estimate. A stage that would exceed it stops
   the pipeline; it does not proceed and reconcile afterwards.
3. **Never continue past a failure the graph said to halt on.** "Halt" that
   continues is worse than no failure semantics at all, because the user
   believes the run stopped.

## 1. Why this is the phase that makes approval worth having

P1 through P3 each put one action behind one approval. That is correct and it
does not scale: a `train → evaluate → serve` journey is three approvals, so an
agent either interrupts a person three times or the person stops reading them.
Approval fatigue is not a UX complaint, it is a **security failure** — the third
prompt is the one that gets waved through, and it is the one that spends the
most.

One approval covering a stated graph, with a stated ceiling, is the shape that
makes the gate survive contact with a real workflow.

## 2. The shape, and what it reuses

`action_plans` already provides everything the *approval* needs:

| Requirement | Already exists |
|---|---|
| server-bound arguments | `canonical_args` + `canonical_args_hash` |
| "editing invalidates" | hash mismatch → `PlanConflict("argument_hash_mismatch")` |
| a spend ceiling | `estimate_micros`, `price_tolerance_bps` |
| human-only approval | `approval_mode`, and `_is_interactive_human` |
| replay safety | `mark_consumed` + `idempotent_response` |

So a pipeline is **one `action_plans` row** whose `canonical_args` carries the
graph. Gate P4's "editing any stage after approval invalidates it" then comes
for free from a mechanism already proven by the promotion work — and, more to
the point, one that is already *tested*.

What does not exist is per-stage execution state. That is a new table:

```
pipeline_stages(plan_id, stage_index, name, action_type, state,
                on_failure, attempt_count, result_ref, tenant_id, …)
```

`plan_id` is the approval. `stage_index` is the order. Everything else is what
happened.

## 3. Decisions, and what each costs

### 3.1 The graph is a list, not a DAG — for now

The plan says "dependency graph"; the reference journey is a chain. A general
DAG needs a topological executor, concurrency limits, and partial-failure
semantics per branch. A list needs an index.

**Ship the list.** `train → evaluate → serve` is the journey the gate names, and
a list is a DAG with one path. The cost is stated plainly: fan-out (`train →
{eval-a, eval-b} → serve`) is not expressible, and adding it later is a schema
change plus an executor rewrite, not a flag. That is the right trade only
because the alternative is a general executor with no user, and a general
executor nobody has run is not a capability.

### 3.2 Failure semantics are per stage, and declared before approval

`on_failure ∈ {halt, continue, retry}` on each stage, fixed at approval time
because it is part of the canonical args. A user approving a graph is approving
its failure behaviour too — deciding it afterwards, at the moment something
broke, is exactly when the decision is worst.

`retry` carries a bounded `max_attempts`. Unbounded retry inside an approved
spend ceiling is a way to spend the whole ceiling on a stage that cannot work.

### 3.3 The ceiling is enforced before each stage, not after the pipeline

The naive implementation sums the stages and compares once. That fails the way
that matters: stage 2 overruns, stage 3 starts anyway, and the overspend is
discovered at the end.

**Before each stage, the executor compares spend-so-far plus this stage's quote
against the approved total.** If it would exceed, the pipeline halts with
`budget_exceeded` and the stage never starts. The user's ceiling is a promise
about what can be spent, and a promise checked only afterwards is a report.

### 3.4 A stage's output is the next stage's input, by reference

`train` produces artifacts; `evaluate` needs them. Passing them by value would
put a manifest in the plan args and make the graph enormous; passing them by
*reference* — `result_ref`, the job id or promotion id a stage produced — keeps
the plan small and lets P3's promotion do the moving.

This is why P4 comes after P3 rather than beside it.

### 3.5 One audit chain

Gate P4 asks for "one audit chain". Every stage event carries the `plan_id`, so
the whole run is one filterable trail rather than three unrelated ones. That is
the difference between "what did this pipeline do" being a query and being an
investigation.

## 4. Gate P4, clause by clause

| Clause | How it is met | How it is proven |
|---|---|---|
| One approval, three stages, one audit chain, end to end | one `action_plans` row + `pipeline_stages` | run `train → evaluate → serve`, assert one approval and three stage rows sharing a `plan_id` |
| A mid-pipeline failure does not silently continue | `on_failure` per stage | fail stage 2 deliberately; assert `halt` stops and `continue` proceeds, each as declared |
| The approved graph is server-bound; editing a stage invalidates it | `canonical_args_hash` | mutate a stage after approval, assert execute refuses |
| Spend is bounded by what was approved | pre-stage ceiling check | quote a pipeline, inflate stage 3's cost, assert it halts rather than overspends |

## 5. What would make this the wrong design

- **If pipelines are usually one stage.** Then this is ceremony over a single
  approval and the primitive should not exist.
- **If stages need to fan out from the start.** The list would be rework rather
  than a foundation, and the DAG should be built first.
- **If the ceiling cannot be quoted per stage.** The whole spend guarantee rests
  on knowing each stage's cost *before* running it; if a stage's cost is only
  knowable by running it, the guarantee weakens to "we stop when we notice".

## 6. Sequence

* **B0 — the graph, stored and approved.** Table, canonical args carrying
  stages, and the hash binding. Nothing executes.
* **B1 — sequential execution with halt.** The narrowest useful executor: run in
  order, stop on failure.
* **B2 — the ceiling.** Pre-stage budget check, `budget_exceeded` halt.
* **B3 — `continue` and `retry`.** The other two failure semantics, each
  asserted by causing the failure rather than by reading the code.
* **B4 — the tool.** `run_pipeline`, async handle like
  `promote_artifact_to_volume`, and a description that says a pipeline is
  running rather than finished.

## 7. The §1 gates this phase owes

This is the first phase after the eval discipline lapsed. §1.5 asks for an eval
delta per phase, and P2 and P3 shipped without theirs because the only runner
was GitHub Actions and it could not execute.

That excuse is gone: `scripts/run_mcp_eval_locally.sh` runs it. **The baseline
must be captured before B4 adds a tool, and again after**, or this phase repeats
the failure the truth table exists to record.

Current surface is 48 tools; `eval-baseline.json` is at 46, so the pre-P4
capture is owed regardless.
