"""Every route an MCP tool calls must admit the credential the tool carries.

Twice on 2026-08-06 a capability was promised to a credential that could not
reach it, and both times every layer was correct in isolation:

* `instances:connect` was enforced on the connection routes and absent from
  `MCP_QUICK_CONNECT_SCOPES`, so the product's own quickstart token got
  `403 Insufficient scope` at the terminal door;
* `register_ssh_key` shipped correctly scoped to `ssh:write` against a route
  guarded by `_require_user_grant`, which rejects `client_credentials` **before
  any scope is read** — so the tool answered 403 no matter what its token held.

Each was found by hand, after shipping, by reading one route. `tests/
test_agent_can_register_its_own_key.py` closes the second for the SSH routes
specifically. Nothing closed it for the rest of the surface, and "read every
handler carefully" is not a mechanism — it is the thing that already failed.

So this joins the two sides automatically: the paths the TypeScript tools call,
against the guards the handlers those paths resolve to actually invoke. It is the
general form of a defect that has now occurred twice, and it runs in seconds.

The join is only as good as its reach, which is why an unmatched tool path is a
**failure** rather than a skip. A tool calling a path the inventory does not know
is either a typo in the tool or a route the generator missed, and both are worth
stopping for.

**Read from the live route table, not from the generated inventory**, and that
is the second version of this file. The first joined against
`docs/generated/endpoint-inventory.md`, which was quicker and quietly useless:
its *Auth dependency* column reads `none found — verify by hand` for **16 of the
34** paths the tools call, because the generator only sees module-level imports.
Every one of those rows passed the check by having no data to fail it — a guard
that reports green across half its surface because it cannot see it. Importing
the app costs a few seconds and removes the blind spot entirely.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "mcp" / "src" / "tools"

#: Helpers that refuse a machine credential outright, before any scope check.
#: `_require_user_grant`'s own docstring: "Rejects client_credentials (machine)
#: tokens outright". It is the correct guard for MFA, password and account
#: deletion — and wrong for anything an agent is meant to do.
HUMAN_ONLY = {"_require_user_grant"}

#: `client.get("/x")`, `client.post<T>(`/x/${y}`, …)` — verb, then first argument.
CALL = re.compile(
    r"client\.(get|post|patch|delete)\s*(?:<[^>]*>)?\s*\(\s*(`[^`]*`|\"[^\"]*\")",
    re.S,
)
#: `${anything}` including one level of nested braces, e.g. `${f({a: 1})}`.
INTERPOLATION = re.compile(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _normalise(path: str) -> str:
    """Reduce a path to method-agnostic shape: every parameter becomes `{}`."""
    path = INTERPOLATION.sub("{}", path)
    path = re.sub(r"\{[^{}]*\}", "{}", path)
    return path.rstrip("/") or "/"


def tool_calls() -> set[tuple[str, str]]:
    """(METHOD, normalised path) for every API call the tools make."""
    found: set[tuple[str, str]] = set()
    for source in sorted(TOOLS_DIR.glob("*.ts")):
        text = source.read_text(encoding="utf-8")
        for verb, raw in CALL.findall(text):
            literal = raw[1:-1]
            if not literal.startswith("/"):
                continue  # a variable, not a literal path
            found.add((verb.upper(), _normalise(literal)))
    return found


def app_routes() -> dict[tuple[str, str], object]:
    """(METHOD, normalised path) -> the handler FastAPI will actually call."""
    import api

    table: dict[tuple[str, str], object] = {}
    for route in api.app.routes:
        endpoint = getattr(route, "endpoint", None)
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if endpoint is None or not methods or not path:
            continue
        for method in methods:
            table[(method.upper(), _normalise(path))] = endpoint
    return table


def _guards_used_by(endpoint: object) -> set[str]:
    """Every function *called* in the handler body.

    AST over calls rather than a substring scan, for the reason
    `tests/test_agent_can_register_its_own_key.py` records: `api_add_ssh_key`'s
    docstring names `_require_user_grant` at length while explaining why it is
    the wrong guard there. A text search would read that explanation as the
    defect it describes.
    """
    import ast
    import inspect
    import textwrap

    try:
        source = textwrap.dedent(inspect.getsource(endpoint))
    except (OSError, TypeError):
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        getattr(call.func, "id", "") or getattr(call.func, "attr", "")
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
    }


def test_the_extractor_finds_calls_at_all():
    """Prove the reach. A regex that matches nothing passes every assertion."""
    calls = tool_calls()
    assert len(calls) > 15, f"only {len(calls)} tool call sites found; the extractor is broken"
    assert ("POST", "/api/ssh/keys") in calls, "register_ssh_key's call site is not being seen"
    assert ("POST", "/api/terminal/ticket") in calls, "open_instance_access's call site is not seen"


def test_the_route_table_loads():
    routes = app_routes()
    assert len(routes) > 400, f"only {len(routes)} routes on the app"


def test_every_tool_path_is_served_by_a_real_route():
    """A tool calling a path nothing serves is a 404 waiting for a user."""
    missing = sorted(tool_calls() - set(app_routes()))
    assert not missing, (
        "these MCP tools call paths the application does not serve — either a "
        f"typo in the tool or a route that moved: {missing}"
    )


def test_no_tool_calls_a_route_that_refuses_machine_credentials():
    """The defect, in its general form.

    A tool's token is `client_credentials` or an agent API key. A route guarded
    by a human-only helper answers 403 before it ever looks at the scope the
    tool was so carefully given.
    """
    routes = app_routes()
    offenders = {
        call: sorted(HUMAN_ONLY & _guards_used_by(routes[call]))
        for call in sorted(tool_calls())
        if call in routes and (HUMAN_ONLY & _guards_used_by(routes[call]))
    }
    assert not offenders, (
        "these routes are called by an MCP tool but reject machine credentials "
        "before any scope check, so the tool answers 403 whatever its token "
        f"holds: {offenders}"
    )


def test_the_check_sees_a_guard_when_one_is_there():
    """Calibration for the AST reader itself.

    If `_guards_used_by` silently returned an empty set — an import failure, a
    handler it cannot fetch source for — every assertion above would pass while
    reading nothing. So point it at a route known to carry the human-only guard
    and require that it finds it.
    """
    routes = app_routes()
    known = ("DELETE", "/api/auth/me")
    assert known in routes, "the account-deletion route moved; pick another known guard site"
    assert HUMAN_ONLY & _guards_used_by(routes[known]), (
        "the AST reader found no human-only guard on account deletion, where one "
        "certainly is — so it is not reading handlers and this file guards nothing"
    )


@pytest.mark.parametrize(
    "helper",
    sorted(HUMAN_ONLY),
)
def test_the_human_only_helpers_still_refuse_machine_credentials(helper: str):
    """Calibration: the premise of the test above, asserted rather than assumed.

    If `_require_user_grant` were ever relaxed to admit machine credentials,
    this file would keep passing while guarding nothing — and the *real* risk
    would have moved to the MFA and password routes it protects.
    """
    source = (ROOT / "routes" / "_deps.py").read_text(encoding="utf-8")
    body = source.split(f"def {helper}(", 1)[1].split("\ndef ", 1)[0]
    assert "client_credentials" in body, (
        f"{helper} no longer mentions client_credentials; if it stopped "
        "rejecting machine tokens, this whole file is now vacuous"
    )
    assert "403" in body, f"{helper} no longer refuses anything"
