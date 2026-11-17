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


#: Gates that refuse by themselves when called, and the predicate that only
#: answers the question.
_REQUIRING_GATES = {"_require_admin", "_require_platform_admin"}
_ADMIN_PREDICATES = {"_is_platform_admin"}


def _call_name(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return getattr(node.func, "id", None) or getattr(node.func, "attr", None)


def _refuses_non_admins(fn: ast.FunctionDef | ast.AsyncFunctionDef, requiring: set[str]) -> bool:
    """Whether every call of `fn` by a non-admin ends in a refusal.

    Read from the statements at the top of its body only, where nothing makes
    them conditional: a requiring gate called there, or an
    `if not _is_platform_admin(...) [or ...]: raise`. A disjunct `not admin`
    means not being an admin is enough to be refused. `if not (admin or
    owner): raise` lets an owner through, so it is not an admin gate.
    """
    for stmt in fn.body:
        if isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return)):
            if _call_name(stmt.value) in requiring:
                return True
        if isinstance(stmt, ast.If) and any(isinstance(s, ast.Raise) for s in stmt.body):
            test = stmt.test
            disjuncts = (
                test.values
                if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)
                else [test]
            )
            for d in disjuncts:
                if (
                    isinstance(d, ast.UnaryOp)
                    and isinstance(d.op, ast.Not)
                    and _call_name(d.operand) in _ADMIN_PREDICATES
                ):
                    return True
    return False


def _local_admin_gates(tree: ast.Module) -> set[str]:
    """Helpers in this module that refuse anyone who is not an admin.

    A module may wrap the check, as `routes/wise.py` does with
    `_require_wise_admin` (a platform admin, and a person at a first-party
    session). Calling an admin check is not enough to count: many helpers call
    `_is_platform_admin` to let an admin past an owner check, which admits
    owners too, and `global_access = _is_platform_admin(user)` decides what to
    show, not who may enter. See `_refuses_non_admins`. A helper that calls one
    that counts, unconditionally, counts too. Route handlers are not helpers.
    """
    helpers = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _route_paths(node)
    }
    gates: set[str] = set()
    grew = True
    while grew:
        grew = False
        for name, fn in helpers.items():
            if name not in gates and _refuses_non_admins(fn, _REQUIRING_GATES | gates):
                gates.add(name)
                grew = True
    return gates


def test_a_local_wrapper_counts_only_when_it_refuses_every_non_admin() -> None:
    tree = ast.parse(
        "def _require_ops_admin(request):\n"
        "    user = _require_auth(request)\n"
        "    if not _is_platform_admin(user) or not _person_present(user):\n"
        "        raise HTTPException(403, 'no')\n"
        "    return user\n"
        "def _via_wrapper(request):\n"
        "    return _require_ops_admin(request)\n"
        "def _via_require_admin(request):\n"
        "    _require_admin(request)\n"
        "def _sees_everything(user):\n"
        "    return _is_platform_admin(user)\n"
        "def _admin_or_owner(user, owner):\n"
        "    if not (_is_platform_admin(user) or user['id'] == owner):\n"
        "        raise HTTPException(403, 'no')\n"
        "def _admin_only_for_machines(user):\n"
        "    if user.get('machine'):\n"
        "        if not _is_platform_admin(user):\n"
        "            raise HTTPException(403, 'no')\n"
        "def _wrapper_in_a_branch(request, flag):\n"
        "    if flag:\n"
        "        _require_ops_admin(request)\n"
    )
    assert _local_admin_gates(tree) == {"_require_ops_admin", "_via_wrapper", "_via_require_admin"}


def test_every_admin_path_route_checks_for_an_admin() -> None:
    offenders: list[str] = []
    for path, rel in sorted(iter_source_files(), key=lambda pair: pair[1]):
        if not rel.startswith("api/routes/"):
            continue
        tree = ast.parse(read_source(path))
        gates = ADMIN_GATES | _local_admin_gates(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            routes = [p for p in _route_paths(node) if "/admin/" in p or p.endswith("/admin")]
            if not routes:
                continue
            called = _called_names(node)
            if called & gates:
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
