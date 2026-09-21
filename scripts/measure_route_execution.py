#!/usr/bin/env python3
"""Which route handlers does the test suite actually *enter*?

`UNTESTED_ENDPOINTS.md` scores a route as covered when its path prefix or its
handler function name appears anywhere under `tests/`. That is a cheap proxy and
a useful worklist, but it measures **mention**, not **execution** — and the two
diverge badly:

    ledger:   0 of 529 routes untested
    measured: 78 of 525 handlers never entered by the suite

Both numbers are true about different things. A route can be named by a test
that only ever receives a 422, because FastAPI rejected the request body before
the handler ran: such a test proves the route is mounted and that its model
validates, and nothing whatsoever about the code inside. Five tests in
`tests/test_untested_endpoints_coverage.py` were exactly that until this script
found them.

It is also what let two handlers sharing one name mark each other covered —
`GET /marketplace/search` was reported tested for as long as the ledger existed
because the v2 POST handler had the same function name.

Usage:

    python -m pytest tests/ --ignore=tests/test_e2e_live.py -q \\
        --cov=routes --cov-report=json:coverage.json
    python scripts/measure_route_execution.py

Exits 0 always: this reports, it does not gate. Gating on it would mean every
contributor runs a 17-minute instrumented suite, and the number is only
meaningful for a whole-suite run — a subset produces a scary figure that means
nothing.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ROUTES = REPO / "routes"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _routes_of(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str, str]]:
    """The (METHOD, path) pairs a handler is decorated with."""
    found = []
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call) or not dec.args:
            continue
        target = dec.func
        method = getattr(target, "attr", "")
        if method not in HTTP_METHODS:
            continue
        first = dec.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append((method.upper(), first.value))
    return found


def _body_lines(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    """Every line of the handler body, excluding its docstring.

    The docstring is excluded because it is executed at *import* time as part
    of building the function object, so counting it would mark every handler in
    an imported module as entered.
    """
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    lines: set[int] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            lineno = getattr(node, "lineno", None)
            if lineno is not None:
                lines.add(lineno)
    return lines


def main() -> int:
    report = REPO / "coverage.json"
    if not report.exists():
        print(
            "coverage.json not found — run the suite with "
            "`--cov=routes --cov-report=json:coverage.json` first (see the "
            "module docstring).",
            file=sys.stderr,
        )
        return 0

    files = json.loads(report.read_text(encoding="utf-8"))["files"]
    never: list[str] = []
    entered = 0
    total = 0

    for path in sorted(ROUTES.glob("*.py")):
        if path.name.startswith("_"):
            continue
        rel = str(path.relative_to(REPO))
        entry = files.get(rel) or files.get(f"./{rel}")
        if entry is None:
            print(f"warning: no coverage data for {rel}", file=sys.stderr)
            continue
        executed = set(entry["executed_lines"])
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorated = _routes_of(node)
            if not decorated:
                continue
            lines = _body_lines(node)
            if not lines:
                continue
            total += 1
            if lines & executed:
                entered += 1
            else:
                for method, route in decorated:
                    never.append(f"{method:6} {route}  ({path.name}:{node.lineno} {node.name})")

    print(f"handlers entered by the suite: {entered}/{total}")
    print(f"handlers never entered:        {len(never)}\n")
    for line in sorted(never):
        print(" ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
