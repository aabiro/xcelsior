"""A route on an `/admin/` path must actually check for an admin.

`_require_scope(user, "admin")` looks like an admin check and is not one. Its
own docstring says it is a **no-op for interactive user sessions**, and for
good reason: a browser session authenticated through `authorization_code`
carries OIDC identity scopes (`profile`, `email`, `offline_access`) that say
nothing about API authority, so treating their absence as a denial would reject
every browser request. It constrains machine credentials. It does not decide
who is an admin.

`POST /api/v2/admin/volumes/reopen-encrypted` was gated with that alone. Its
docstring said "Admin-only"; its path said `/admin/`; an ordinary registered
account got 200, on a route that reopens the LUKS device of every encrypted
volume on the platform. It survived because nothing had ever called it — it was
one of the 78 handlers `scripts/measure_route_execution.py` found unentered,
while `UNTESTED_ENDPOINTS.md` scored it covered.

The check has to be structural. A behavioural test can only cover the routes
someone thought to write a test for, and the whole failure here was that nobody
had.
"""

from __future__ import annotations

import ast

from tests._source_tree import iter_source_files, read_source

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}

#: Anything that genuinely establishes platform-admin identity.
ADMIN_GATES = {"_require_admin", "_is_platform_admin", "_require_platform_admin"}

#: Looks like a gate, is not one on an interactive session.
NOT_AN_ADMIN_GATE = "_require_scope"


def _route_paths(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    paths = []
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call) or not dec.args:
            continue
        if getattr(dec.func, "attr", "") not in HTTP_METHODS:
            continue
        first = dec.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            paths.append(first.value)
    return paths


def _called_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "id", None) or getattr(target, "attr", None)
            if name:
                names.add(name)
    return names


def test_every_admin_path_route_checks_for_an_admin() -> None:
    offenders: list[str] = []
    for path, rel in sorted(iter_source_files(), key=lambda pair: pair[1]):
        if not rel.startswith("routes/"):
            continue
        tree = ast.parse(read_source(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            routes = [p for p in _route_paths(node) if "/admin/" in p or p.endswith("/admin")]
            if not routes:
                continue
            called = _called_names(node)
            if called & ADMIN_GATES:
                continue
            how = (
                f"only {NOT_AN_ADMIN_GATE}"
                if NOT_AN_ADMIN_GATE in called
                else "no authorization call at all"
            )
            offenders.append(f"{rel}:{node.lineno} {node.name} — {how}\n      {', '.join(routes)}")

    assert not offenders, (
        "these serve an /admin/ path without establishing that the caller is an "
        "admin. `_require_scope` is not a substitute — it no-ops for interactive "
        "sessions, so it constrains machine credentials and admits every signed-in "
        "browser user:\n    " + "\n    ".join(offenders)
    )
