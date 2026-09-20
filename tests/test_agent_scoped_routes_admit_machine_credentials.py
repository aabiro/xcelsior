"""No route may demand a scope an agent holds and then refuse its credential.

This is the general form of `tests/test_agent_can_register_its_own_key.py`,
which asserts the same join for the SSH-key routes only. That file records why:

    That is the second time in one day the same shape got through: a capability
    promised to a credential that could not reach it.

Both instances were a *handler* choosing `_require_user_grant`, which rejects
`client_credentials` before any scope is consulted, on a route whose scope a
Quick Connect token carries. Each layer was correct alone — the tool was
scoped right, the route was guarded right — so every existing test passed while
the capability was unreachable. An integration test that happened to
authenticate as a session user would also have passed.

The ssh-only version left the rest of the surface uncovered, and the session
note that shipped it said so: *"nothing asserts it for the rest of the surface.
That is the most likely place for a third instance."* This closes that.

**The scan currently finds nothing, and that is the point of the third test
below.** A detector that reports zero while being incapable of reporting
anything is the failure mode this whole file exists to prevent, so the scanner
is run against a synthetic handler that *does* have the defect and must find it.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oauth_service import MCP_QUICK_CONNECT_SCOPES  # noqa: E402

#: Scopes a pasted Quick Connect token actually carries. Read from the source of
#: truth rather than listed here: a scope added there without this updating
#: would silently narrow the scan.
AGENT_SCOPES = frozenset(MCP_QUICK_CONNECT_SCOPES)

#: Helpers that refuse `client_credentials` outright, before any scope is read.
#: `_require_user_grant` says so in its own docstring: "Rejects
#: client_credentials (machine) tokens outright".
HUMAN_ONLY_HELPERS = frozenset({"_require_user_grant"})


def _called_names(node: ast.AST) -> set[str]:
    """Every function *called* in this handler, by name.

    Calls only, never text: the docstring on `api_add_ssh_key` explains at
    length why `_require_user_grant` is wrong for it, and a text scan would read
    that explanation as the defect it describes.
    """
    return {
        getattr(call.func, "id", "") or getattr(call.func, "attr", "")
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


def _scopes_demanded(node: ast.AST) -> set[str]:
    """Scope literals passed to `_require_scope` inside this handler."""
    found: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        if (getattr(call.func, "id", "") or getattr(call.func, "attr", "")) != "_require_scope":
            continue
        for arg in call.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.add(arg.value)
    return found


def _unreachable_handlers(source: str, where: str) -> list[str]:
    """Handlers that demand an agent scope *and* refuse machine credentials."""
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        human = _called_names(node) & HUMAN_ONLY_HELPERS
        scopes = _scopes_demanded(node) & AGENT_SCOPES
        if human and scopes:
            out.append(
                f"{where}::{node.name} demands {sorted(scopes)} but calls "
                f"{sorted(human)}, which rejects the credential that carries it"
            )
    return out


def _route_sources() -> list[tuple[str, str]]:
    return [
        (p.name, p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "routes").glob("*.py"))
    ]


# ── the scan ─────────────────────────────────────────────────────────

def test_no_route_promises_a_capability_its_credential_cannot_reach() -> None:
    offenders: list[str] = []
    for name, src in _route_sources():
        offenders.extend(_unreachable_handlers(src, name))
    assert not offenders, (
        "these handlers are reachable by scope and refused by helper — the tool "
        "is authorised and the backend answers 403:\n  " + "\n  ".join(offenders)
    )


# ── the scan must be capable of finding something ────────────────────

def test_the_inputs_are_not_empty() -> None:
    """Either set going empty turns the scan above into a guard that cannot fail."""
    assert AGENT_SCOPES, "no agent scopes; the scan matches nothing"
    assert HUMAN_ONLY_HELPERS, "no human-only helpers; the scan matches nothing"
    assert "ssh:write" in AGENT_SCOPES, (
        "ssh:write is not in the Quick Connect scopes — either the credential "
        "changed or this test is reading the wrong source of truth"
    )


def test_the_scanner_detects_the_defect_it_exists_for() -> None:
    """The original bug, verbatim in shape, must be found.

    Without this, "zero offenders" is indistinguishable from a scanner that
    cannot see. Verified against the real thing too: re-introducing
    `_require_user_grant(request)` into `api_add_ssh_key` makes the scan above
    report exactly that handler.
    """
    synthetic = (
        "def api_add_ssh_key(body, request):\n"
        '    """A handler shaped like the original defect."""\n'
        "    user = _require_user_grant(request)\n"
        '    _require_scope(user, "ssh:write")\n'
        "    return {}\n"
    )
    found = _unreachable_handlers(synthetic, "synthetic.py")
    assert len(found) == 1, f"the scanner missed the defect it exists for: {found}"
    assert "api_add_ssh_key" in found[0]
    assert "ssh:write" in found[0]


def test_a_machine_safe_handler_is_not_flagged() -> None:
    """And it must not flag the correct shape, or it would be noise."""
    ok = (
        "def api_list_ssh_keys(request):\n"
        "    user = _get_current_user(request)\n"
        '    _require_scope(user, "ssh:write")\n'
        "    return []\n"
    )
    assert _unreachable_handlers(ok, "ok.py") == []
