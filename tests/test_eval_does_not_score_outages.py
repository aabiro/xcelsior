"""An unreachable API is not a wrong answer.

`scripts/mcp_tool_eval.py` graded each case in a `try/except` that appended
`False` on any exception. So an expired token, a rate limit, or an exhausted
credit balance did not stop the run — it silently became "the model chose
badly", and the run finished with a plausible number.

That is not hypothetical. A 5-sample capture on 2026-08-08 lost its Anthropic
balance partway through and reported:

    expected_tool_accuracy 0.54    abstention 0.0    unsafe_write_rate 1.0

Read literally, that says the connector started grabbing tools on every case
where it should have stayed silent — a serious regression. What actually
happened is that 66 calls never reached the API. The tells were structural:
`direct` 40/40 and `indirect` 35/35 perfect with everything after them zero, in
file order, at *lower* total cost than a smaller run that completed.

Had that been written to `eval-baseline.json`, it would have become the number
every later phase compares against — a fabricated regression, permanently.

The harness now aborts on an API error and writes no artifact. A partial
baseline is worse than no baseline, because a number gets recorded and trusted.
"""

from __future__ import annotations

import ast
import os
import pathlib

os.environ.setdefault("XCELSIOR_ENV", "test")

REPO = pathlib.Path(__file__).resolve().parent.parent
EVAL = REPO / "scripts" / "mcp_tool_eval.py"


def _grading_loop() -> ast.AST:
    """The `main` function, where cases are graded."""
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("mcp_tool_eval.py has no main()")


def test_the_grader_is_still_where_this_thinks_it_is():
    """Prove the reach — a rename makes everything below vacuous."""
    main = _grading_loop()
    calls = {
        getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    }
    assert "selected_tools" in calls, "main() no longer calls selected_tools"
    assert "grade" in calls, "main() no longer grades anything"


def test_an_api_error_is_not_recorded_as_a_failed_case():
    """The defect: `except Exception: results.append(False)`.

    Asserted structurally rather than by string search, because the fix is about
    *what the handler does* — a handler that logs and continues reads very
    differently from one that stops.
    """
    main = _grading_loop()
    offenders = []
    for node in ast.walk(main):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Does this handler stop the run?
        stops = any(
            isinstance(inner, (ast.Raise,))
            or (
                isinstance(inner, ast.Expr)
                and isinstance(inner.value, ast.Call)
                and (getattr(inner.value.func, "id", "") in {"SystemExit", "exit"})
            )
            for inner in ast.walk(node)
        )
        # Does it instead record a result?
        records = any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "attr", "") == "append"
            and any(
                isinstance(a, ast.Constant) and a.value is False for a in inner.args
            )
            for inner in ast.walk(node)
        )
        if records and not stops:
            offenders.append(getattr(node, "lineno", "?"))
    assert not offenders, (
        "an exception handler in the grading loop records False without "
        f"stopping (line {offenders}). An unreachable API would be scored as "
        "the model choosing badly, and the run would finish with a number."
    )


def test_the_abort_says_no_baseline_was_written():
    """The operator has to know the artifact is stale, not fresh.

    Silence here is the dangerous case: a run that dies leaves the *previous*
    `eval-baseline.json` on disk, which looks exactly like a fresh result. That
    already caused one near-miss with an expired token.
    """
    source = EVAL.read_text(encoding="utf-8")
    assert "No baseline written" in source, (
        "the abort no longer tells the operator that no artifact was produced, "
        "so a stale eval-baseline.json will be read as the run's result"
    )


def test_the_artifact_records_when_it_was_captured():
    """`captured_at` is how a stale artifact is caught."""
    source = EVAL.read_text(encoding="utf-8")
    assert '"captured_at"' in source, (
        "the baseline no longer records captured_at, which is the only way to "
        "tell a fresh artifact from one left behind by a run that died"
    )
