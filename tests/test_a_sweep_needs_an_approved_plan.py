"""A sweep is the largest single spend here, and it had the weakest gate.

`create_instance` will not launch **one** instance without an approved
`plan_id`: the tool prepares a plan, a human approves it, and only then does
anything run. `POST /api/v1/image-sweeps` launched up to sixty-four on a bare
call. Nothing was wrong with the code — the gate simply had not been built for
it — but exposing that shape as an MCP tool would have made the bulk path the
documented way to skip the approval that guards the single path.

## Why reusing `/api/v1/launch-plans` would have been worse than nothing

That endpoint quotes **one job**. A sweep's `count` would sit outside the
approved canonical args, so a plan approved for one instance would authorise
sixty-four, and the approval record would look correct afterwards. A gate that
can be satisfied by an approval for something smaller is worse than an absent
gate, because it produces evidence that consent was given.

So the plan is sweep-shaped: `count` is inside `canonical_args`, and
`canonical_args_hash` binds it. Approving means approving the number shown.

## What is asserted here

The structural facts, because the behavioural version needs a funded wallet, a
ready image with a digest, and an approval — and the parts of that which are
reachable in-process are already covered by `test_a_sweep_is_a_record.py`. What
these guards protect is the *shape*: that the creating route cannot launch, that
the executing route checks approval before it looks at anything else, and that
the count cannot be edited between approval and execution.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROUTES = pathlib.Path(__file__).resolve().parent.parent / "routes" / "instances.py"


def _function(name: str) -> ast.FunctionDef:
    """One parse, one lookup. Re-parsing per call breaks identity comparisons."""
    tree = ast.parse(ROUTES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from routes/instances.py; if it moved, repoint this")


def _source(name: str) -> str:
    return ast.get_source_segment(ROUTES.read_text(encoding="utf-8"), _function(name)) or ""


def _calls(name: str) -> set[str]:
    """Every function called in the body, by bare name."""
    found: set[str] = set()
    for node in ast.walk(_function(name)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                found.add(func.id)
            elif isinstance(func, ast.Attribute):
                found.add(func.attr)
    return found


# ── Calibration ───────────────────────────────────────────────────────


def test_both_routes_exist_and_parse():
    """Two empty bodies satisfy every 'does not call' assertion below."""
    assert len(_calls("api_create_image_sweep")) > 5, "the creating route parsed nearly empty"
    assert len(_calls("api_execute_image_sweep_plan")) > 5, "the executing route parsed nearly empty"


# ── The creating route must not be able to launch ─────────────────────


@pytest.mark.parametrize("forbidden", ["create_sweep", "submit_job", "_wallet_preflight"])
def test_creating_a_sweep_plan_launches_nothing(forbidden: str):
    """The whole point of the split. Quoting must not be able to spend.

    If any of these reappear here, the approval is decorative: the caller gets
    a plan id *and* their instances, and nobody had to approve anything.
    """
    assert forbidden not in _calls("api_create_image_sweep"), (
        f"api_create_image_sweep calls {forbidden}. Preparing a plan must not "
        "launch, fund, or record a sweep — otherwise the approval step is "
        "something a caller can ignore."
    )


def test_the_plan_records_the_member_count():
    """`count` inside the approved args is what stops a 1-plan buying 64.

    The canonical args are the body, and the body carries `count`. Asserted
    because the tempting shortcut — quoting a single member and multiplying at
    execution — puts the number outside what was approved.
    """
    body = _source("api_create_image_sweep")
    assert "body.model_dump" in body, (
        "the plan's canonical args are no longer the request body, so `count` "
        "may no longer be inside what gets approved"
    )
    assert "canonical_args_hash=canonical_hash" in body, (
        "the plan no longer binds a hash of its arguments; an approved sweep "
        "could be edited before execution"
    )
    assert "body.count" in body, "the estimate no longer scales with the member count"


def test_the_estimate_is_per_member_times_the_count():
    """A plan that under-quotes a bulk spend is consent for a smaller number."""
    body = _source("api_create_image_sweep")
    assert "estimate_launch_hold_cad" in body, "the estimate is no longer the real fund-gate hold"
    assert "* body.count" in body.replace("  ", " "), (
        "the estimate no longer multiplies by the member count"
    )


# ── The executing route must refuse an unapproved plan ────────────────


def test_execution_refuses_a_plan_that_is_not_approved():
    body = _source("api_execute_image_sweep_plan")
    assert 'plan["status"] != "approved"' in body, (
        "the execute route no longer checks that the plan was approved"
    )
    assert "approval_required" in body, "the refusal no longer names itself"


def test_execution_refuses_a_plan_belonging_to_another_tenant():
    """Not-found rather than forbidden: no existence oracle for plan ids."""
    body = _source("api_execute_image_sweep_plan")
    assert 'str(plan["tenant_id"]) != principal.tenant_id' in body, (
        "the execute route no longer checks plan ownership"
    )
    assert "PlanNotFound" in body, (
        "a plan owned by someone else must read as absent; `forbidden` confirms "
        "that a plan with that id exists"
    )


def test_execution_refuses_a_plan_whose_arguments_changed():
    """The check that makes the approved `count` mean anything."""
    body = _source("api_execute_image_sweep_plan")
    assert "argument_hash_mismatch" in body, (
        "an approved sweep plan could be edited between approval and execution "
        "— including its member count — and still execute"
    )


def test_execution_refuses_a_plan_for_a_different_action():
    """An approved plan is approval for *that* action, not for any spend."""
    body = _source("api_execute_image_sweep_plan")
    assert 'plan["action_type"] != "create_image_sweep"' in body, (
        "an approved plan for some other action could be spent on a sweep"
    )


def test_a_replayed_execution_returns_the_same_sweep():
    """One approval, one sweep. A retry must not launch a second set of members."""
    body = _source("api_execute_image_sweep_plan")
    assert body.count('"succeeded"') >= 2, (
        "the executed-plan short circuit is checked in fewer than two places; "
        "it is needed before launching *and* after, because two callers can "
        "pass the first check together"
    )
    assert "mark_consumed" in body, "execution is no longer recorded against the plan"
    assert "idempotent" in body, "a replay no longer identifies itself as one"
