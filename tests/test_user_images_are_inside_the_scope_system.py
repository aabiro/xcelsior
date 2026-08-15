"""Every mutating instance route must answer to a scope, an admin, or a host.

Scope reduction is only real if the routes enforce it. `_require_scope` is a
no-op for browser sessions and checks **every machine credential** — so a route
that never calls it is a route where a deliberately narrowed token has its
narrowing ignored.

## What this found

`DELETE /user-images/{image_id}` and `PATCH /user-images/{image_id}` check
authentication and ownership and **no scope at all**. A Quick Connect key issued
with `instances:read` — the reduced set our own quickstarts tell people to
paste — could delete or rename any image its owner had. Ownership is not a
substitute: the owner's *token* was deliberately restricted, and that
restriction was the thing being ignored.

It surfaced because the sweep needs images, so the image surface was about to
get MCP tools. A route nobody could reach from an agent had a latent hole; a
route with a tool in front of it has a live one.

## Three legitimate ways to not have a scope

Enumerated rather than pattern-matched, because "it looked internal" is how a
route ends up unguarded:

* **admin** — `_require_admin`, or an explicit `is_admin` refusal. Queue and
  failover control.
* **host** — `_require_agent_auth`. The worker callbacks
  (`auto-launch/report`, `http-ports/report`, `user-images/{id}/complete`) are
  authenticated by a per-host token and carry no user scopes by design; this is
  the same set `TokenAuthMiddleware` must let past.
* **scope** — everything else.

## Why this is scoped to one file, and what that leaves unchecked

The same scan across all of `routes/` reports 88 mutating routes in none of
those three categories, and **that number is not a finding**. It is dominated by
things where a user scope is the wrong question: pre-authentication login and
OAuth flows, signature-verified webhooks, worker callbacks, and the public
inference endpoints authenticated by an endpoint key. Auditing them is real work
and each needs its own judgement.

Asserting the whole directory now would mean writing an 88-line exemption list
that nobody has checked, which reads as an audit and is not one. This file
covers the surface that was actually examined. **`routes/` beyond
`instances.py` is unaudited, not clean** — and saying so is the point of this
paragraph.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROUTES = pathlib.Path(__file__).resolve().parent.parent / "routes" / "instances.py"
SOURCE = ROUTES.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
FUNCS = {
    n.name: n
    for n in ast.walk(TREE)
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
}

#: An admin gate is a **refusal**, not a mention. The first version of this
#: list held the bare string `is_admin`, and `api_delete_user_image` computes
#: `is_admin` to decide whether to *bypass* an ownership check — so the route
#: this file exists to catch was classified as admin-gated and passed. That is
#: the match-a-mention defect, in the guard written to find it.
ADMIN_MARKERS = ("_require_admin(", '"Admin only"')
HOST_MARKERS = ("_require_agent_auth", "_require_host_token")


def _called_names(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            found.add(func.id)
        elif isinstance(func, ast.Attribute):
            found.add(func.attr)
    return found


def _guarded(name: str, depth: int = 0, seen: frozenset[str] = frozenset()) -> str | None:
    """Which of the three guards protects this handler, following helpers.

    Following helpers is the whole reason this is an AST walk rather than a
    grep. A first version looked for `_require_scope` in the handler body alone
    and reported seventeen unguarded routes including `terminate` — which is
    guarded, by `_authorize_instance_mutation`. Eight of those seventeen were
    my own false positives.
    """
    if name in seen or depth > 3 or name not in FUNCS:
        return None
    body = ast.get_source_segment(SOURCE, FUNCS[name]) or ""
    if "_require_scope" in body:
        return "scope"
    if any(marker in body for marker in HOST_MARKERS):
        return "host"
    if any(marker in body for marker in ADMIN_MARKERS):
        return "admin"
    for helper in _called_names(FUNCS[name]):
        if helper in FUNCS:
            found = _guarded(helper, depth + 1, seen | {name})
            if found:
                return found
    return None


def _mutating_routes() -> list[tuple[str, str]]:
    """`(method path, handler)` for every POST/PUT/PATCH/DELETE route."""
    out: list[tuple[str, str]] = []
    for func in FUNCS.values():
        for dec in func.decorator_list:
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                continue
            if dec.func.attr not in ("post", "put", "patch", "delete"):
                continue
            if dec.args and isinstance(dec.args[0], ast.Constant):
                out.append((f"{dec.func.attr.upper()} {dec.args[0].value}", func.name))
    return out


# ── Calibration ───────────────────────────────────────────────────────


def test_the_parse_finds_routes_and_guards():
    """An empty parse passes every assertion below."""
    routes = _mutating_routes()
    assert len(routes) > 15, f"only {len(routes)} mutating routes parsed"
    guarded = [r for r in routes if _guarded(r[1])]
    assert len(guarded) > 10, "guard detection found almost nothing; the walk is broken"


def test_helper_enforcement_is_followed_not_just_the_handler_body():
    """The specific mistake this file was nearly built on.

    `api_terminate_instance` contains no `_require_scope`; it calls
    `_authorize_instance_mutation`, which does. A guard that cannot see that
    reports a false hole on the most destructive route in the file.
    """
    assert "_require_scope" not in (
        ast.get_source_segment(SOURCE, FUNCS["api_terminate_instance"]) or ""
    ), "terminate now enforces inline; this test's premise needs rewriting"
    assert _guarded("api_terminate_instance") == "scope", (
        "helper-based enforcement is no longer detected, so every route that "
        "guards through _authorize_instance_mutation will read as unguarded"
    )


# ── The rule ──────────────────────────────────────────────────────────


def test_every_mutating_instance_route_is_guarded():
    """Scoped, admin-gated, or host-authenticated. No fourth category."""
    unguarded = sorted(path for path, handler in _mutating_routes() if not _guarded(handler))
    assert not unguarded, (
        "these mutating routes enforce no scope, no admin check and no host "
        f"token, so a deliberately narrowed machine credential is unrestricted "
        f"on them: {unguarded}"
    )


@pytest.mark.parametrize(
    "path",
    ["DELETE /user-images/{image_id}", "PATCH /user-images/{image_id}"],
)
def test_the_image_mutations_require_a_write_scope_specifically(path: str):
    """Named individually because these are the two that were wrong.

    A generic rule can be satisfied by the *wrong* guard — marking them admin
    would pass the test above while locking every customer out, and marking
    them host-authenticated would open them to any worker.
    """
    handler = dict((p, h) for p, h in _mutating_routes())[path]
    assert _guarded(handler) == "scope", f"{path} is not guarded by a scope"
    body = ast.get_source_segment(SOURCE, FUNCS[handler]) or ""
    assert "instances:write" in body, (
        f"{path} mutates a tenant's image and must require instances:write; a "
        "read-scoped key deleting an image is the defect this asserts against"
    )


def test_listing_images_requires_a_read_scope():
    """The read half. Lower stakes, same principle: a token that holds no
    instance scope at all should not enumerate a tenant's images."""
    body = ast.get_source_segment(SOURCE, FUNCS["api_list_user_images"]) or ""
    assert "_require_scope" in body and "instances:read" in body, (
        "GET /user-images enforces no scope, so scope reduction does not reach it"
    )
