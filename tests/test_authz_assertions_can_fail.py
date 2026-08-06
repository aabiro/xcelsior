"""An authorization assertion must not accept success as a pass.

`tests/test_health_endpoints_coverage.py::test_ssh_keygen_requires_auth` read:

    assert r.status_code in (401, 403, 200)

The endpoint being wide open and the endpoint being locked both satisfied it, so
the test could not fail. It sat in the coverage file for `/ssh/keygen` — one of
the four endpoints §0.1 names as authenticated but not authorized — and passed
throughout the period the defect existed.

The sharp class is narrow and worth stating precisely: a tuple containing a
success code **and** 401 or 403. Such an assertion claims something about
authorization while admitting the outcome that would prove it absent.

**Not swept up here:** tuples mixing success with 404, 503, 400 or 422. Those
are usually a legitimate tolerance — an endpoint that may be disabled, a feature
flag off in this environment, a resource that may not exist yet. There are 19 of
those, they are a weaker pattern, and treating them as the same defect would
inflate a real finding into a false one. If they should be tightened, that is a
separate piece of work with a separate argument.

The rule enforced: **zero tuples mix success with an authorization refusal.**

**This scanner reads the syntax tree, not the text**, and the first version did
not. Recovered from the closed branch it flagged three files, and all three hits
were *prose*: docstrings quoting `assert r.status_code in (401, 403, 200)` while
explaining why it is wrong — including the sibling file recovered alongside it.
A guard against tests that cannot fail, failing on the documentation of the
thing it forbids.

That is the tenth time in this suite a text-scanning guard has flagged its own
explanation. The others were fixed by stripping comments; this one parses
`ast.Assert` nodes and reads the literal codes out of the comparison, so prose
is invisible to it by construction rather than by filtering.
"""

from __future__ import annotations

import ast
import pathlib

TESTS = pathlib.Path(__file__).resolve().parent

SUCCESS_CODES = {200, 201, 202, 204}
AUTHZ_REFUSALS = {401, 403}


def _status_code_membership(node: ast.expr) -> list[int] | None:
    """The literal codes in a `<...>.status_code in (...)` comparison, if any."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], ast.In):
        return None
    left = node.left
    if not (isinstance(left, ast.Attribute) and left.attr == "status_code"):
        return None
    container = node.comparators[0]
    if not isinstance(container, (ast.Tuple, ast.List, ast.Set)):
        return None
    codes = [
        el.value
        for el in container.elts
        if isinstance(el, ast.Constant) and isinstance(el.value, int)
    ]
    return codes or None


def _mixed_authz_assertions() -> list[tuple[str, int, list[int], str]]:
    found = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - a broken test file fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            codes = _status_code_membership(node.test)
            if not codes:
                continue
            distinct = set(codes)
            if distinct & SUCCESS_CODES and distinct & AUTHZ_REFUSALS:
                found.append(
                    (path.name, node.lineno, sorted(distinct), ast.unparse(node.test)[:100])
                )
    return found


def test_no_authorization_assertion_accepts_success():
    """The load-bearing rule."""
    offenders = _mixed_authz_assertions()
    assert not offenders, (
        "these assertions accept both a success and an authorization refusal, "
        "so they pass whether the endpoint is guarded or wide open:\n"
        + "\n".join(f"  {f}:{n} {codes}\n      {line}" for f, n, codes, line in offenders)
    )


def test_the_scanner_detects_a_planted_assertion():
    """Prove the reach rather than trusting the silence.

    A scanner that matches nothing reports clean, and clean is what a broken
    scanner looks like — the vocabulary guard reported zero while four blog
    posts were full of what it was hunting.
    """
    planted = ast.parse("assert r.status_code in (401, 403, 200)").body[0]
    assert isinstance(planted, ast.Assert)
    codes = _status_code_membership(planted.test)
    assert codes, "the scanner no longer recognises the original defect"
    distinct = set(codes)
    assert distinct & SUCCESS_CODES and distinct & AUTHZ_REFUSALS


def test_the_scanner_ignores_the_same_text_in_prose():
    """The calibration control, and the reason this file parses rather than greps.

    Recovered from the closed branch, this scanner read raw lines and flagged
    three docstrings that quote the defect while explaining it. A guard that
    cannot distinguish a violation from its own description will be silenced by
    whoever it inconveniences, and silencing it costs the real check too.
    """
    module = ast.parse('"""Bad: assert r.status_code in (401, 403, 200)."""\nx = 1\n')
    offenders = [
        n for n in ast.walk(module)
        if isinstance(n, ast.Assert) and _status_code_membership(n.test)
    ]
    assert not offenders, "prose describing the defect was read as the defect"


def test_the_scanner_does_not_flag_a_legitimate_tolerance():
    """`(200, 404)` for an endpoint that may be disabled is not this defect."""
    node = ast.parse("assert r.status_code in (200, 404)").body[0]
    codes = set(_status_code_membership(node.test) or [])
    assert codes, "the scanner stopped recognising a status_code membership test"
    assert not (codes & AUTHZ_REFUSALS)


def test_the_scanner_does_not_flag_a_pure_refusal_tuple():
    """`(401, 403)` is correct — either refusal code is acceptable."""
    node = ast.parse("assert r.status_code in (401, 403)").body[0]
    codes = set(_status_code_membership(node.test) or [])
    assert codes, "the scanner stopped recognising a status_code membership test"
    assert not (codes & SUCCESS_CODES)
