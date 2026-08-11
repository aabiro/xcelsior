"""Every path a live gate posts to must be a route this app serves.

## The defect this exists to prevent

Live gates are written as refusals: "this call is *not* a 200". A path that
does not exist is also not a 200, so a gate aimed at a phantom route passes
forever while asserting nothing. Two were found this way, both by running the
gates rather than by reading them:

| gate | aimed at | actual route |
|---|---|---|
| P1's replay-safety assertion | `/api/billing/topup` | never existed |
| P3's promotion refusal | `/api/v1/promotions` | `/api/v2/volumes/{id}/promotions` |

Both reported a refusal, both for eight commits, and neither could ever fail.

The live gates carry a runtime `_assert_routed` probe now, but that only fires
when a fleet and a credential are present — which, for most of this project's
history, they have not been. This test needs neither. It resolves the paths
against the app's own router at commit time, so a phantom path is caught by
the ordinary suite instead of surviving until someone finally runs the gates.

## Why both sides are derived

There is no list of expected paths here to fall out of date. The paths come
from walking the live tests' syntax trees, and the routes come from
`app.routes`. Adding a gate or renaming a route is picked up with no edit to
this file — the failure mode of a hand-maintained duplicate list is exactly
what it is guarding against.

## Why the AST rather than a grep

An earlier version of a similar guard matched a word inside a docstring and
reported a passing check that had examined nothing. String literals are read
from the parsed tree, and docstrings are excluded explicitly, so a path
mentioned in prose is prose and a path passed to a request is a path.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

LIVE_DIR = pathlib.Path(__file__).resolve().parent / "live"

#: What counts as "a path this app should serve". Deliberately narrow: these
#: are the prefixes the gates actually call. A literal that is not one of these
#: (a hostname, a fragment, a message) is not a claim about routing.
PATH_PREFIXES = ("/api/", "/instances", "/agent/", "/host", "/healthz", "/v1/")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """`id()` of every Constant that is a docstring, so prose is not a path."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def _literal_paths(source: str) -> set[str]:
    """Every non-docstring string literal that looks like a route on this app."""
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in skip:
            continue
        value = node.value
        if not isinstance(value, str) or not value.startswith(PATH_PREFIXES):
            continue
        # Drop a query string; routing does not consider it.
        paths.add(value.split("?", 1)[0].rstrip("/") or value)
    return paths


def _route_matchers() -> list[re.Pattern[str]]:
    """One anchored regex per route template, with `{param}` as one segment."""
    from api import app

    matchers: list[re.Pattern[str]] = []
    for route in app.routes:
        template = getattr(route, "path", None)
        if not isinstance(template, str):
            continue
        # `re.escape` escapes the braces as well, so the substitution has to
        # match the escaped form `\{name\}` rather than the literal `{name}`.
        pattern = re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(template))
        matchers.append(re.compile("^" + pattern + "/?$"))
    return matchers


def _live_test_files() -> list[pathlib.Path]:
    return sorted(LIVE_DIR.glob("test_*_live.py"))


def test_the_live_gates_exist_at_all():
    """A guard over an empty set passes; this is what stops that reading green."""
    files = _live_test_files()
    assert files, f"no live gates found under {LIVE_DIR}"
    assert any(_literal_paths(f.read_text()) for f in files), (
        "no live gate names any API path; either they stopped calling the API "
        "or this guard stopped being able to see the calls"
    )


@pytest.mark.parametrize("gate", _live_test_files(), ids=lambda p: p.stem.removesuffix("_live"))
def test_every_path_a_live_gate_calls_is_a_route_this_app_serves(gate: pathlib.Path):
    matchers = _route_matchers()
    assert matchers, "the app exposes no routes; the comparison would be vacuous"

    unresolved = sorted(
        path
        for path in _literal_paths(gate.read_text())
        if not any(m.match(path) for m in matchers)
    )
    assert not unresolved, (
        f"{gate.name} calls {len(unresolved)} path(s) this app does not serve: "
        f"{unresolved}. A refusal asserted against a path that does not exist "
        "passes whatever the server does. Fix the path, not the assertion."
    )
