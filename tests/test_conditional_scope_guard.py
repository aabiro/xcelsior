"""A scope check that only runs when you are already authenticated is not a check.

Twelve route handlers authorize like this:

    user = _get_current_user(request) if request else None
    if user:
        _require_scope(user, "privacy:write")

`_get_current_user` returns `None` for an anonymous caller, so the `if` is
skipped and the handler proceeds. The scope is enforced *only* against callers
who presented a credential — the opposite of the intent. Presenting no
credential is strictly more powerful than presenting a valid one.

**This is not always a bug.** On a public read it is the correct shape: the
resource is meant to be world-readable, and the scope check exists to keep a
*narrowed* credential from over-reading. `GET /marketplace`,
`GET /api/v2/gpu/available` and the published transparency report all rely on
that, and locking them down would break anonymous browsing.

On a write it is a hole. `POST /api/privacy/purge-expired` destroyed expired
retention records for anyone who could reach the URL;
`POST /api/transparency/legal-request` let anyone insert a warrant that was
never served. No credential, no scope, no audit trail.

So the rule this file enforces is the distinction, not the pattern:

    authentication is unconditional; scope refinement may be conditional.

Conditional scope on its own is not the defect — several handlers legitimately
call `_require_auth` and *then* narrow by scope only for machine principals, or
grant an admin a bypass. `control_plane_v1.api_v1_reconciliation_findings` and
`instances._require_worker_status_update` both do exactly that, and flagging
them would push authors toward suppressing the guard instead of satisfying it.

The defect is conditional scope reached from `_get_current_user`, which returns
`None`, with nothing raising in between. Those handlers are listed by name as
public-read exemptions — deliberately world-readable, using the conditional
check only to constrain a *narrowed* credential — and anything else that
matches is a bug.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTES = ROOT / "routes"

#: Read-only handlers that intentionally serve anonymous callers and use the
#: conditional check to constrain *credentialed* ones. Each is a GET.
PUBLIC_READ_EXEMPTIONS = {
    ("gpu.py", "api_gpu_available"),
    ("marketplace.py", "api_get_marketplace"),
    ("privacy.py", "api_retention_policies"),
    ("privacy.py", "api_retention_summary"),
    ("privacy.py", "api_get_privacy_config"),
    ("privacy.py", "api_get_consents"),
    ("transparency.py", "api_transparency_report"),
}

_MUTATING = {"post", "put", "patch", "delete"}


def _http_methods(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The HTTP verbs a handler is registered for, from its @router decorators."""
    methods: set[str] = set()
    for deco in node.decorator_list:
        call = deco if isinstance(deco, ast.Call) else None
        func = call.func if call else deco
        if isinstance(func, ast.Attribute) and func.attr.lower() in {
            "get", "post", "put", "patch", "delete", "head", "options",
        }:
            methods.add(func.attr.lower())
    return methods


class _ScopeCallFinder(ast.NodeVisitor):
    """Records whether each `_require_scope` call sits under a conditional.

    String-matching `"if user:"` would miss `if user is not None:`,
    `if user and ...:`, or a renamed variable — and a guard that can be evaded
    by rephrasing is the kind that reports clean while the defect is present.
    Structure is what matters: a scope check reached only on some paths is
    conditional however the condition is spelled.

    An `if` whose body always raises is the *correct* shape and is not counted.
    `if not user: raise HTTPException(401)` followed by an unconditional
    `_require_scope` is exactly what these handlers should do, and flagging it
    would push authors toward suppressing the guard rather than satisfying it.
    """

    #: Helpers that establish the caller and raise when there is none.
    ESTABLISHES_CALLER = frozenset({
        "_require_auth",
        "_require_admin",
        "_require_user_grant",
        "_require_provider_or_admin",
        "_resolve_principal",
        "_require_agent_auth",
        "_require_host_operator",
    })

    def __init__(self) -> None:
        self.unconditional = 0
        self.conditional = 0
        self.establishes_caller = False
        self._user_names: set[str] = set()
        self._depth = 0

    def visit_Assign(self, node: ast.Assign) -> None:
        # Remember what `_get_current_user(...)` was bound to, so a later
        # `if not <that name>: raise` can be recognised as the auth guard
        # rather than as an unrelated validation raise.
        call = node.value
        if isinstance(call, ast.IfExp):
            call = call.body
        if isinstance(call, ast.Call):
            func = call.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else ""
            )
            if name == "_get_current_user":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._user_names.add(target.id)
        self.generic_visit(node)

    def _is_absent_user_guard(self, test: ast.expr) -> bool:
        """`not user` / `user is None`, for a name bound from _get_current_user."""
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = test.operand
            return isinstance(inner, ast.Name) and inner.id in self._user_names
        if isinstance(test, ast.Compare) and isinstance(test.ops[0], ast.Is):
            left, right = test.left, test.comparators[0]
            return (
                isinstance(left, ast.Name)
                and left.id in self._user_names
                and isinstance(right, ast.Constant)
                and right.value is None
            )
        return False

    def visit_If(self, node: ast.If) -> None:
        if node.body and all(isinstance(s, ast.Raise) for s in node.body):
            # `if not user: raise 401` at the top level establishes the caller.
            # Narrow on purpose: an unrelated `if not job_id: raise 400` must
            # not be mistaken for authentication.
            if self._depth == 0 and self._is_absent_user_guard(node.test):
                self.establishes_caller = True
            for child in node.orelse:
                self.visit(child)
            return
        self._depth += 1
        for child in node.body:
            self.visit(child)
        for child in node.orelse:
            self.visit(child)
        self._depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else ""
        )
        if name == "_require_scope":
            if self._depth:
                self.conditional += 1
            else:
                self.unconditional += 1
        elif name in self.ESTABLISHES_CALLER and not self._depth:
            self.establishes_caller = True
        self.generic_visit(node)


def conditional_scope_handlers() -> list[tuple[str, str, set[str]]]:
    """(file, handler, methods) for handlers whose scope check is conditional."""
    found: list[tuple[str, str, set[str]]] = []
    for path in sorted(ROUTES.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            finder = _ScopeCallFinder()
            for stmt in node.body:
                finder.visit(stmt)
            if finder.conditional and not finder.establishes_caller:
                found.append((path.name, node.name, _http_methods(node)))
    return found



def test_no_mutating_handler_makes_its_scope_check_conditional():
    """The load-bearing rule: writes must know who is calling."""
    offenders = [
        (f, n, sorted(m))
        for f, n, m in conditional_scope_handlers()
        if m & _MUTATING
    ]
    assert not offenders, (
        "These handlers mutate state but only check scopes when the caller "
        "already authenticated, so an anonymous request skips authorization "
        "entirely. Call `_require_auth` (or a stronger guard) unconditionally "
        "before `_require_scope`:\n"
        + "\n".join(f"  {f}::{n} [{', '.join(m)}]" for f, n, m in offenders)
    )


def test_read_only_exemptions_are_still_read_only():
    """An exempted handler must not quietly gain a mutating verb.

    The exemption was granted because the endpoint is a public read. If a POST
    is added to it later, the exemption silently becomes a hole.
    """
    by_name = {(f, n): m for f, n, m in conditional_scope_handlers()}
    violations = [
        (f, n, sorted(by_name[(f, n)]))
        for (f, n) in PUBLIC_READ_EXEMPTIONS
        if (f, n) in by_name and by_name[(f, n)] & _MUTATING
    ]
    assert not violations, (
        f"exempted public-read handlers now accept writes: {violations}"
    )


def test_exemption_list_has_no_dead_entries():
    """A stale exemption is a permission nobody reviewed.

    If a handler is fixed or deleted, its exemption must go with it — otherwise
    the list slowly becomes a blanket the next author reads as approval.
    """
    live = {(f, n) for f, n, _ in conditional_scope_handlers()}
    dead = sorted(PUBLIC_READ_EXEMPTIONS - live)
    assert not dead, (
        f"exemptions for handlers that no longer use the pattern: {dead}"
    )


def test_every_conditional_handler_is_either_exempt_or_absent():
    """No third category: the set is exactly the reviewed public reads."""
    unreviewed = sorted(
        (f, n)
        for f, n, m in conditional_scope_handlers()
        if (f, n) not in PUBLIC_READ_EXEMPTIONS
    )
    assert not unreviewed, (
        "conditional scope checks not covered by a reviewed exemption: "
        f"{unreviewed}"
    )
