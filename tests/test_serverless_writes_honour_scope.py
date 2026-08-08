"""Scope reduction must mean something on the serverless surface.

A user can mint an OAuth client with a deliberately narrow scope set — that is
the entire point of the consent screen listing scopes. On `/api/v2/serverless/*`
that promise is currently not kept: **34 of 35 routes call no `_require_scope`
at all.** They check ownership (`_user_can_access_serverless_endpoint`) and team
role (`_require_team_instance_write`), and stop there.

So a token narrowed to `inference:read` — or holding no inference scope
whatsoever — can still delete an endpoint, cancel a running job, or create a new
endpoint that bills, provided the caller owns it. It is not a cross-tenant hole.
It is a *scope-reduction* hole: a narrowed credential behaves identically to a
full one, which is the same class of defect as the `api` wildcard that once made
every MCP tool contract decorative.

`scripts/audit_route_auth.py` does not catch this, and that is by design rather
than oversight: it accepts `_require_auth` or `_get_current_user` as a guard,
because it answers "is this route authenticated and access-checked?" This file
asks the different question — "does this route honour the scopes the credential
was issued with?"

**This is a ratchet, not a fix.** Re-scoping 34 routes is a change with real
blast radius and belongs in a reviewed commit of its own; what this prevents is
the number growing while nobody is looking. Lower it as routes are scoped. It may
never rise: a new serverless write arrives scoped, or the commit that adds it
scopes it.

The plan's Gate P0 says "every access and billing endpoint refuses a token
missing its new scope". Serverless was not in that list, and this records the
gap rather than leaving it to be rediscovered.
"""

from __future__ import annotations

import ast
import os
import pathlib

os.environ.setdefault("XCELSIOR_ENV", "test")

REPO = pathlib.Path(__file__).resolve().parent.parent
SERVERLESS = REPO / "routes" / "serverless.py"

MUTATING = {"post", "patch", "delete", "put"}

#: Lower as routes are scoped. It may never rise.
MAX_UNSCOPED_SERVERLESS_ROUTES = 34


def _routes() -> list[tuple[str, str, bool]]:
    """(verb, path, has_scope_check) for every /api/v2/serverless route."""
    tree = ast.parse(SERVERLESS.read_text(encoding="utf-8"))
    found: list[tuple[str, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorated: list[tuple[str, str]] = []
        for d in node.decorator_list:
            if not isinstance(d, ast.Call) or not d.args:
                continue
            attr = getattr(d.func, "attr", "")
            if not isinstance(d.args[0], ast.Constant):
                continue
            path = d.args[0].value
            if isinstance(path, str) and path.startswith("/api/v2/serverless"):
                decorated.append((attr.upper(), path))
        if not decorated:
            continue
        # Calls in the body — AST rather than substring, so a docstring naming
        # `_require_scope` while explaining its absence is not counted as one.
        calls = {
            getattr(c.func, "id", "") or getattr(c.func, "attr", "")
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        }
        has_scope = "_require_scope" in calls
        for verb, path in decorated:
            found.append((verb, path, has_scope))
    return found


def test_the_scan_still_finds_the_serverless_surface():
    """Prove the reach. An empty scan passes every assertion below."""
    routes = _routes()
    assert len(routes) >= 30, f"only {len(routes)} serverless routes found; the scan is broken"


def test_the_scan_can_tell_a_scoped_route_from_an_unscoped_one():
    """Calibration: at least one route *is* scoped, so a `False` means something.

    If every route came back unscoped because the detector never matches, the
    ratchet below would pass by measuring nothing.
    """
    routes = _routes()
    assert any(has_scope for _, _, has_scope in routes), (
        "no serverless route registers as scoped — the detector is not finding "
        "_require_scope calls at all, so the count is meaningless"
    )


def test_unscoped_serverless_routes_do_not_grow():
    """The ratchet."""
    unscoped = [(v, p) for v, p, has in _routes() if not has]
    assert len(unscoped) <= MAX_UNSCOPED_SERVERLESS_ROUTES, (
        f"{len(unscoped)} serverless routes enforce no scope, up from "
        f"{MAX_UNSCOPED_SERVERLESS_ROUTES}. A route that never reads the "
        "caller's scopes makes a deliberately narrowed credential behave "
        f"exactly like a full one. New: {sorted(set(unscoped))[:5]}"
    )


def test_the_destructive_ones_are_named_so_they_are_not_forgotten():
    """The rows that matter most, listed rather than buried in a count.

    Deleting an endpoint and cancelling a running job are the money-stopping
    operations. If either ever gains a scope check this test fails, which is the
    intended way to find out that the ratchet can be lowered.
    """
    scoped = {(v, p) for v, p, has in _routes() if has}
    for verb, path in (
        ("DELETE", "/api/v2/serverless/endpoints/{endpoint_id}"),
        ("POST", "/api/v2/serverless/endpoints/{endpoint_id}/jobs/{job_id}/cancel"),
    ):
        assert (verb, path) not in scoped, (
            f"{verb} {path} now enforces a scope — good. Remove it from this "
            "list and lower MAX_UNSCOPED_SERVERLESS_ROUTES in the same commit."
        )
