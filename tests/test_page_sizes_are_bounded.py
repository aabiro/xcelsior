"""A caller must not be able to choose how much of the database to read.

Fifteen handlers took `limit: int = <n>` straight off the query string with no
ceiling, so `?limit=100000000` was a valid request. Most of these read tables
that grow without bound — `wallet_transactions`, `job_attempts`,
`serverless_token_ledger` — and the value goes into `LIMIT %s` unchanged, so one
authenticated request could ask the database for every row it has and the API to
serialise all of them.

`GET /api/billing/wallet/{id}/history?limit=100000000` returned 200. It was fast
only because the wallet under test was empty.

The bound is declared rather than clamped in the body: `Query(n, ge=1, le=1000)`
puts it in the OpenAPI schema, so generated clients and the docs carry it, and an
out-of-range value is refused with a 422 by the same validation that handles
every other malformed parameter — instead of being silently rewritten to
something the caller did not ask for.

`ge=1` matters too. A negative `LIMIT` is a Postgres error, so it was a 500
rather than a 400.

This checks the shape for any future handler: a page-size parameter on a route
must carry an upper bound, either declaratively or by being clamped in the body.
"""

from __future__ import annotations

import ast

from tests._source_tree import iter_source_files

HTTP_DECORATORS = {"get", "post", "put", "delete", "patch"}
PAGE_SIZE_NAMES = {"limit", "count", "size", "per_page"}


def _is_route(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(node, ast.Attribute) and node.attr in HTTP_DECORATORS:
            return True
    return False


def _declares_ceiling(default: ast.expr) -> bool:
    """`Query(n, le=...)` / `Query(n, lt=...)` — the bound is in the schema."""
    return isinstance(default, ast.Call) and any(kw.arg in ("le", "lt") for kw in default.keywords)


def _clamped_in_body(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """`min(limit, N)` or an explicit comparison — bounded, if less legibly."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("min", "max")
            and any(isinstance(a, ast.Name) and a.id == name for a in node.args)
        ):
            return True
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(isinstance(s, ast.Name) and s.id == name for s in sides):
                return True
    return False


def test_no_route_lets_the_caller_choose_the_page_size():
    offenders: list[str] = []
    for path, rel in iter_source_files(include_prefixes=("routes/",)):
        if not rel.startswith("routes/"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not _is_route(fn):
                continue
            args = fn.args
            if not args.defaults:
                continue
            for arg, default in zip(args.args[-len(args.defaults) :], args.defaults):
                if arg.arg not in PAGE_SIZE_NAMES:
                    continue
                if _declares_ceiling(default) or _clamped_in_body(fn, arg.arg):
                    continue
                offenders.append(f"{rel}:{fn.lineno} {fn.name}({arg.arg}) has no upper bound")

    assert not offenders, (
        "these handlers let the caller choose how many rows to read; give the "
        "parameter a ceiling with Query(n, ge=1, le=N):\n  " + "\n  ".join(sorted(offenders))
    )
