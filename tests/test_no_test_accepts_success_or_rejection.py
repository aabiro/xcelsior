"""No test may accept a 2xx *and* a 422 from the same call.

A 422 means the request body never satisfied the route's model. Depending on
where the check lives that is either FastAPI refusing before the handler runs,
or the handler refusing a precondition before it does anything — but either way
the behaviour the test is named for did not happen. Listing 422 beside 200
therefore makes the assertion true no matter which occurred, and the test
reports coverage of a path it never took.

This is not hypothetical. Six tests written in this suite passed against a body
the model had rejected, and were only caught by re-measuring which handlers the
suite actually enters. `test_deciding_admission` asserted
`in (200, 409, 422)` and had never once made a successful admission decision —
`expected_version` is required and it was not sending one, so every run took the
`admission_precondition_failed` branch.

Asserting 422 on its own is fine and is often the point: a test that a malformed
body is refused should say exactly that. What this forbids is the pair, where
success and refusal are both accepted and the test can no longer fail.
"""

from __future__ import annotations

import ast

from tests._source_tree import iter_source_files

SUCCESS = {200, 201, 202, 204}


def _accepted_codes(node: ast.Compare) -> set[int]:
    """The integer literals in `x.status_code in (...)`."""
    if not (len(node.ops) == 1 and isinstance(node.ops[0], ast.In)):
        return set()
    left = node.left
    if not (isinstance(left, ast.Attribute) and left.attr == "status_code"):
        return set()
    right = node.comparators[0]
    if not isinstance(right, (ast.Tuple, ast.List, ast.Set)):
        return set()
    return {e.value for e in right.elts if isinstance(e, ast.Constant) and isinstance(e.value, int)}


def test_no_assertion_accepts_both_success_and_an_unprocessable_body():
    offenders: list[str] = []
    for path, rel in iter_source_files(include_tests=True, include_prefixes=("tests/",)):
        if (
            not rel.startswith("tests/")
            or path.name == "test_no_test_accepts_success_or_rejection.py"
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            codes = _accepted_codes(node)
            if 422 in codes and codes & SUCCESS:
                offenders.append(
                    f"{rel}:{node.lineno}: accepts {sorted(codes)} — a 422 means the "
                    "call did not do the thing this test is named for"
                )

    assert not offenders, (
        "these assertions pass whether the request succeeded or was refused "
        "unparsed:\n  " + "\n  ".join(sorted(offenders))
    )
