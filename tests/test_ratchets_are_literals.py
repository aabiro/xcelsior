"""A ratchet ceiling must be a literal, never derived from what it bounds.

`tests/test_open_endpoint_registry.py` carried this:

    MAX_UNCLASSIFIED = len(NEEDS_REVIEW)
    ...
    assert len(NEEDS_REVIEW) <= MAX_UNCLASSIFIED

That can never fail. It compares a number to itself, so for its entire life
`test_the_unclassified_count_does_not_grow` asserted nothing — and "the ratchet
held" was never evidence. It is the same defect as
`assert r.status_code in (401, 403, 200)`: an assertion satisfied equally by the
condition it guards against and the condition it wants, sitting inside the guard
written to close that very class.

A ratchet only means something when the ceiling is a decision someone recorded.
Lowering it is the unit of progress; a computed ceiling silently follows the
value up and calls that compliance.

This scans for the shape rather than the instance, because the instance is
already fixed and the shape is what recurs.
"""

from __future__ import annotations

import ast
import pathlib

TESTS = pathlib.Path(__file__).resolve().parent

#: Names that bound something and therefore must be literals.
_RATCHET_PREFIXES = ("MAX_", "BUDGET", "LIMIT_", "CEILING_", "EXPECTED_")

#: Calls that mean the ceiling is derived from the thing it bounds.
_DERIVING_CALLS = {"len", "sum", "count", "max", "min"}


def _module_level_assignments() -> list[tuple[str, str, ast.AST]]:
    """(file, name, value-node) for every module-level constant in tests/."""
    found: list[tuple[str, str, ast.AST]] = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and node.value is not None:
                    found.append((path.name, target.id, node.value))
    return found


def _is_deriving_call(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else ""
            )
            if name in _DERIVING_CALLS:
                return True
    return False


def test_no_ratchet_ceiling_is_computed_from_what_it_bounds():
    """The load-bearing rule."""
    offenders = [
        f"{file}: {name}"
        for file, name, value in _module_level_assignments()
        if name.startswith(_RATCHET_PREFIXES) and _is_deriving_call(value)
    ]
    assert not offenders, (
        "these ratchet ceilings are computed rather than recorded, so the "
        "assertion compares a number to itself and can never fail — the shape "
        f"of `MAX_UNCLASSIFIED = len(NEEDS_REVIEW)`:\n  " + "\n  ".join(offenders)
    )


def test_the_scanner_detects_the_original_defect():
    """Prove the reach rather than trusting the silence.

    A scanner that matches nothing reports clean, and clean is exactly what a
    broken scanner looks like.
    """
    planted = ast.parse("MAX_UNCLASSIFIED = len(NEEDS_REVIEW)").body[0]
    assert isinstance(planted, ast.Assign)
    assert _is_deriving_call(planted.value), "the scanner no longer sees len()"
    assert "MAX_UNCLASSIFIED".startswith(_RATCHET_PREFIXES)


def test_the_scanner_accepts_a_recorded_literal():
    """`MAX_X = 3` is the correct shape and must not be flagged."""
    literal = ast.parse("MAX_SUITE_RELAXATIONS = 3").body[0]
    assert isinstance(literal, ast.Assign)
    assert not _is_deriving_call(literal.value)


def test_the_scanner_does_not_flag_an_unrelated_computed_constant():
    """Only names that bound something are ratchets.

    A helper computing a set with `len()` is ordinary code; flagging it would
    push authors to rename around the guard rather than satisfy it.
    """
    other = ast.parse("SOME_TOTAL = len([1, 2, 3])").body[0]
    assert isinstance(other, ast.Assign)
    assert not "SOME_TOTAL".startswith(_RATCHET_PREFIXES)


def test_the_known_ratchets_are_all_literals():
    """Named explicitly, so one being deleted is visible in review.

    These are the ceilings the suite's honesty rests on. Each was checked by
    hand when this file was written; only `MAX_UNCLASSIFIED` was computed.
    """
    expected = {
        ("test_companion_schema_discipline.py", "MAX_LEGACY_FLOAT_CAD_COLUMNS"),
        ("test_enforcement_ratchet.py", "MAX_SUITE_RELAXATIONS"),
        # GT0 bounds inventory rows lacking a *classification*.
        ("test_gt0_classification_ratchet.py", "MAX_UNCLASSIFIED"),
        # The registry bounds open endpoints lacking a *justification*. It was
        # also called MAX_UNCLASSIFIED until the collision was noticed: two
        # ceilings sharing a name while bounding unrelated sets is how a number
        # gets quoted against the wrong denominator.
        ("test_open_endpoint_registry.py", "MAX_UNJUSTIFIED_OPEN_ENDPOINTS"),
        ("test_tool_scope_registry_completeness.py", "EXPECTED_TOOL_TOTAL"),
    }
    seen = {
        (file, name)
        for file, name, _ in _module_level_assignments()
        if name.startswith(_RATCHET_PREFIXES)
    }
    missing = sorted(expected - seen)
    assert not missing, (
        f"ratchet constants that vanished — was a guard deleted? {missing}"
    )
