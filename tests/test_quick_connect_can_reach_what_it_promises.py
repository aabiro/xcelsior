"""A pasted Quick Connect token must reach the routes the product promises it.

`406c0a1` put `instances:connect` in front of the stream ticket, auto-launch
discovery and port exposure; `9c9bb5a` added the terminal ticket. Its commit
message said:

    Nothing breaks: Quick Connect already carries `instances:connect` — it is
    listed under *MCP quick connect* — so the token this product tells people to
    paste keeps working.

That was wrong, and it was wrong in production for a day. A live token minted
from `/api/mcp/quick-connect` answered:

    403 Insufficient scope — required: instances:connect;
    granted: billing:read, events:read, gpu:read, inference:read,
             inference:write, instances:operate, instances:read,
             instances:write, marketplace:read

**The mistake was reading the wrong list.** `SYSTEM_ALLOWED_SCOPES` in
`oauth_delegation.py` contains `instances:connect` beneath a comment reading
`# MCP quick connect.` — but that list is what the *system principal may grant*,
not what a Quick Connect token *holds*. The held set is
`MCP_QUICK_CONNECT_SCOPES` in `oauth_service.py`, and it did not have it.

So this file asserts the relationship that actually matters: **every scope the
connection routes enforce is a scope the quickstart credential carries.** Read
from the routes rather than restated, so a scope added to a connection route
tomorrow fails here instead of in someone's terminal.

The second test covers the *other* way these drift: the backend list and the
frontend's `MCP_SCOPES` are maintained separately, with a comment on each asking
for them to be kept in sync. They still diverged. A comment is not a mechanism.
"""

from __future__ import annotations

import ast
import os
import pathlib
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The routes whose whole purpose is "connect to your instance". Named rather
#: than pattern-matched: this is the set the consent text describes, and a
#: reviewer should see it change.
CONNECTION_HANDLERS = [
    ("routes/instances.py", "api_instance_stream_ticket"),
    ("routes/instances.py", "api_instances_auto_launch_get"),
    ("routes/instances.py", "api_instances_expose"),
    ("routes/terminal.py", "api_terminal_ticket"),
]


def _scopes_required_by(module_path: str, func_name: str) -> set[str]:
    tree = ast.parse((ROOT / module_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func_name:
            continue
        found = set()
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id", "") or getattr(call.func, "attr", "")
            if name != "_require_scope":
                continue
            for arg in call.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
        return found
    raise AssertionError(f"{func_name} not found in {module_path}")


def test_the_connection_routes_are_still_where_this_thinks_they_are():
    """Prove the reach.

    If a handler is renamed the lookup raises, which is the point — silently
    finding nothing would make every assertion below vacuous.
    """
    total = set()
    for module_path, func in CONNECTION_HANDLERS:
        total |= _scopes_required_by(module_path, func)
    assert "instances:connect" in total, (
        "none of the named connection handlers requires instances:connect any "
        "more; either the guard was removed or this list is stale"
    )


def test_quick_connect_carries_every_scope_the_connection_routes_demand():
    """The regression, asserted against the routes rather than a copy of them."""
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    required = set()
    for module_path, func in CONNECTION_HANDLERS:
        required |= _scopes_required_by(module_path, func)

    missing = sorted(required - set(MCP_QUICK_CONNECT_SCOPES))
    assert not missing, (
        f"the pasted Quick Connect token cannot reach its own connection "
        f"routes: {missing}. This is what a user sees as "
        f"'403 Insufficient scope' the first time they try to open a terminal."
    )


def test_the_frontend_list_matches_the_backend_one():
    """Both files carry a comment asking for this, and they drifted anyway.

    `McpAgentSetup.tsx` sends its own `MCP_SCOPES` when creating the client, so
    a divergence means the dashboard provisions a different credential from the
    one the API documents.
    """
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    tsx = (
        ROOT / "frontend" / "src" / "components" / "settings" / "McpAgentSetup.tsx"
    ).read_text(encoding="utf-8")
    block = tsx.split("const MCP_SCOPES = [", 1)[1].split("]", 1)[0]
    frontend = set(re.findall(r'"([a-z_]+:[a-z_]+)"', block))

    backend = set(MCP_QUICK_CONNECT_SCOPES)
    assert frontend == backend, (
        "the dashboard provisions a different scope set from the one the API "
        f"defines: only in frontend={sorted(frontend - backend)}, "
        f"only in backend={sorted(backend - frontend)}"
    )


#: The MCP server's own tests assert what the connector credential can reach,
#: against a literal list of its scopes. That is a third copy of a set already
#: written twice, in a language that cannot import either of the others.
TS_SCOPE_TEST = "mcp/tests/unit/scope-enforcement.test.ts"


def _ts_quick_connect_literals() -> list[set[str]]:
    text = (ROOT / TS_SCOPE_TEST).read_text(encoding="utf-8")
    blocks = re.findall(r"const quickConnect = \[(.*?)\]", text, flags=re.S)
    return [set(re.findall(r'"([a-z_]+:[a-z_]+)"', block)) for block in blocks]


def test_the_typescript_test_still_declares_its_scope_set_inline():
    """Prove the reach — a rename would make the pin below vacuous."""
    literals = _ts_quick_connect_literals()
    assert literals, (
        f"no `const quickConnect = [...]` found in {TS_SCOPE_TEST}; either it "
        "was renamed or it now derives the set, in which case delete the pin "
        "below rather than leaving it passing on nothing"
    )
    assert all(literals), "a quickConnect literal parsed as empty"


def test_the_typescript_copy_matches_the_backend_list():
    """The third copy, pinned for the same reason as the second.

    This one had already drifted before anything pinned it: `instances:connect`
    went into `MCP_QUICK_CONNECT_SCOPES` after a live token was refused by the
    terminal, and this file's literals kept the pre-fix set — so the MCP suite
    was asserting reachability for a credential no user holds.
    """
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    backend = set(MCP_QUICK_CONNECT_SCOPES)
    for index, literal in enumerate(_ts_quick_connect_literals()):
        assert literal == backend, (
            f"{TS_SCOPE_TEST} quickConnect literal #{index + 1} is not the "
            f"scope set Quick Connect issues: only in TS="
            f"{sorted(literal - backend)}, only in backend="
            f"{sorted(backend - literal)}"
        )


def test_every_quick_connect_scope_is_one_the_platform_defines():
    """A typo here mints a credential holding a scope nothing enforces."""
    from oauth_delegation import known_scopes
    from oauth_service import MCP_QUICK_CONNECT_SCOPES

    unknown = sorted(set(MCP_QUICK_CONNECT_SCOPES) - known_scopes())
    assert not unknown, f"Quick Connect grants undefined scopes: {unknown}"
