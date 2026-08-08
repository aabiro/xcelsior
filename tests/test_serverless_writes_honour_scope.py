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
#:
#: **The floor is 7, not 0.** Seven of these are `/workers/*` fleet callbacks
#: behind `_require_worker_callback`, whose only callers are
#: `serverless/worker_sdk/client.py` and `worker_agent.py`. They authenticate a
#: machine, carry no user credential, and must never take a user scope — GT0
#: classifies them `internal`. Recording that here rather than in a commit
#: message, because a bare `34` implies 34 routes of work and seven of them
#: would be a mistake to do.
MAX_UNSCOPED_SERVERLESS_ROUTES = 34

#: The `/v1/serverless/*` family, counted as distinct operations rather than
#: decorators — each carries an `{endpoint_slug}` twin pointing at the same
#: handler.
#:
#: These are not unguarded. They use `_resolve_serverless_endpoint_auth`, and
#: that helper is **asymmetric**: the endpoint-API-key branch checks
#: `key_has_scope(key_row, "inference:write")` and raises 403 without it, while
#: the OAuth/user branch checks ownership and team role and reads no scope at
#: all. The narrower credential is enforced; the broader one is not.
MAX_UNSCOPED_V1_OPERATIONS = 8


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


def _v1_operations() -> set[tuple[str, str]]:
    """Distinct `/v1/serverless` operations, slug twins collapsed.

    `POST /v1/serverless/{id}/run` and
    `POST /v1/serverless/{id}/{endpoint_slug}/run` are two decorators on one
    handler. Counting decorators would double every number here and make the
    ratchet drift for a cosmetic reason.
    """
    tree = ast.parse(SERVERLESS.read_text(encoding="utf-8"))
    ops: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = {
            getattr(c.func, "id", "") or getattr(c.func, "attr", "")
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        }
        if "_require_scope" in calls:
            continue
        for d in node.decorator_list:
            if not isinstance(d, ast.Call) or not d.args:
                continue
            if not isinstance(d.args[0], ast.Constant):
                continue
            path = d.args[0].value
            if isinstance(path, str) and path.startswith("/v1/serverless"):
                ops.add((getattr(d.func, "attr", "").upper(), node.name))
    return ops


def test_the_v1_family_is_visible_to_this_measurement():
    """Prove the reach. The first version of this file could not see `/v1` at all."""
    assert _v1_operations(), "no /v1/serverless operations found; the scan is broken"


def test_unscoped_v1_operations_do_not_grow():
    """The `/v1` half of the same gap, which the `/api/v2` count never covered."""
    ops = _v1_operations()
    assert len(ops) <= MAX_UNSCOPED_V1_OPERATIONS, (
        f"{len(ops)} /v1/serverless operations read no scope, up from "
        f"{MAX_UNSCOPED_V1_OPERATIONS}. This family includes `run` — the route "
        f"`run_serverless_job` calls. New: {sorted(ops)[:5]}"
    )


def test_the_auth_resolver_is_still_asymmetric():
    """The inversion, asserted so that fixing it is noticed.

    `_resolve_serverless_endpoint_auth` checks `inference:write` on an endpoint
    API key and checks no scope at all on a user credential. That is backwards:
    the endpoint key is the *narrower* credential, deliberately minted for one
    endpoint, and it is the one held to a scope.

    When the OAuth branch gains a scope check this test fails — which is the
    intended way to discover that `MAX_UNSCOPED_V1_OPERATIONS` can be lowered.
    """
    import inspect
    import textwrap

    from routes import _deps

    source = textwrap.dedent(inspect.getsource(_deps._resolve_serverless_endpoint_auth))
    tree = ast.parse(source)

    key_branch_checks_scope = "key_has_scope" in source
    # The user branch is whatever follows the *last* `_get_current_user(request)`.
    #
    # `rsplit`, not `split`, and the difference is not cosmetic: the helper calls
    # `_get_current_user` twice, so splitting on the first occurrence swallows
    # the whole endpoint-key branch — including its `key_has_scope` call — into
    # what this then treats as the user branch. The first draft did exactly that
    # and reported the asymmetry resolved when it was not, which is the failure
    # mode this whole file exists to prevent in other people's code.
    user_branch = source.rsplit("_get_current_user(request)", 1)[-1]
    user_branch_checks_scope = "_require_scope" in user_branch or "key_has_scope" in user_branch

    assert key_branch_checks_scope, (
        "the endpoint-key branch no longer checks inference:write — if that was "
        "removed rather than the user branch gaining a check, the surface got "
        "weaker, not more consistent"
    )
    assert not user_branch_checks_scope, (
        "the OAuth/user branch now checks a scope — the asymmetry is resolved. "
        "Lower MAX_UNSCOPED_V1_OPERATIONS and delete this test in the same commit."
    )
    assert isinstance(tree, ast.Module)
