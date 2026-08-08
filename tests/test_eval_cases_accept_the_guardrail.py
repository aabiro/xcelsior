"""An eval case must not penalise the model for obeying the tool surface.

`should_i_run_this`'s published description says:

    Use this instead of estimate_job_cost whenever you are about to launch — it
    answers whether the job *should* run, not merely what it costs.

So a model that calls it before a launch is doing what the connector told it to.
`approval-training-repo` expected `run_training_job` or `create_instance` and
nothing else, and marked the model down for calling the guardrail — while its
siblings `approval-launch` and `approval-serverless` already accepted theirs.
That is an inconsistency inside the eval, not a judgement about the model, and
it cost three of the fifteen approval trials in the captured baseline.

The rule this file enforces: **if a case's prompt asks for a spend, the
guardrail for that spend is an acceptable answer.** Not because it makes the
number better, but because the alternative is an eval that contradicts the
surface it grades — and the surface, not the eval, is what ships to users.

This is not a licence to accept any read. `approval-terminate` still expects
`terminate_instance`, because there is no affordability guardrail for
destruction — the "check first" affordance there is the tool's own
`confirm:false` preview, which is a different thing from a separate tool.
"""

from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("XCELSIOR_ENV", "test")

REPO = pathlib.Path(__file__).resolve().parent.parent
CASES = REPO / "mcp" / "evals" / "tool-selection.jsonl"

#: The guardrail that belongs with each kind of spend, by the tool the case
#: expects. Read from the tool surface's own intent, not invented here.
GUARDRAIL_FOR = {
    "create_instance": "should_i_run_this",
    "run_training_job": "should_i_run_this",
    "schedule_under_budget": "should_i_run_this",
    "create_serverless_endpoint": "should_i_run_pel_job",
    "run_serverless_job": "should_i_run_pel_job",
}


def _cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_case_file_parses_and_has_approval_cases():
    """Prove the reach — an empty selection passes everything below."""
    approval = [c for c in _cases() if c.get("category") == "approval"]
    assert len(approval) >= 4, f"only {len(approval)} approval cases found"


def test_every_spend_case_accepts_its_guardrail():
    """The rule, applied to whichever cases exist rather than to a fixed list."""
    offenders = []
    for case in _cases():
        expected = set(case.get("expect_any_of") or [])
        needed = {GUARDRAIL_FOR[t] for t in expected if t in GUARDRAIL_FOR}
        if needed and not (needed & expected):
            offenders.append((case["id"], sorted(expected), sorted(needed)))
    assert not offenders, (
        "these cases expect a spending tool but reject its guardrail, so a model "
        "following the guardrail's own description is scored as wrong: "
        f"{offenders}"
    )


def test_the_guardrails_are_real_tools():
    """A guardrail named here but absent from the surface guards nothing."""
    surface = json.loads((REPO / "mcp" / "tool-surface.json").read_text(encoding="utf-8"))
    names = {tool["name"] for tool in surface["tools"]}
    missing = sorted(set(GUARDRAIL_FOR.values()) - names)
    assert not missing, f"guardrail tools not published on the surface: {missing}"


def test_destructive_cases_still_demand_the_destructive_tool():
    """The rule above must not become 'any read counts'.

    There is no affordability guardrail for destroying an instance; the caution
    step is `confirm:false` on the tool itself. A case that started accepting
    `get_instance` for a terminate request would stop measuring anything.
    """
    by_id = {c["id"]: c for c in _cases()}
    for case_id in ("approval-terminate", "followup-cancel-after-cost"):
        case = by_id.get(case_id)
        if case is None:
            continue
        expected = set(case.get("expect_any_of") or [])
        assert not (expected & {"get_instance", "list_instances", "get_instance_logs"}), (
            f"{case_id} now accepts a plain read, so it no longer checks that the "
            "model commits to the action the user asked for"
        )
