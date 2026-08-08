"""A mutation on the serverless surface needs write access, not read access.

`POST /api/v2/serverless/endpoints/{id}/jobs/{job_id}/cancel` called
`_require_auth` and `_get_endpoint_for_user` and stopped — read-level access —
while every sibling mutation on that surface calls
`_require_serverless_endpoint_write`, and the `/v1` twin cancels the *same job*
through the *same* `_svc().cancel_inflight_job(...)` behind
`_resolve_serverless_endpoint_auth(write=True)`.

Two routes, one operation, two different answers to "may this caller do it".

The consequence is specific: `_require_serverless_endpoint_write` is
`_require_serverless_endpoint_access` plus `_require_team_instance_write`, and
that second half is what refuses a **team viewer** — the role whose entire
definition is that it cannot modify things. On this one route a viewer could
cancel other people's inference jobs.

Latent while the route was dashboard-only, and reachable by an agent the moment
`cancel_serverless_job` shipped as a tool pointing at it rather than at the v1
twin. That is the lesson worth keeping: **publishing a tool changes who can
reach a route**, so a weaker sibling stops being a curiosity and becomes the
door people use.

This is a different defect from the scope-reduction gap in
`tests/test_serverless_writes_honour_scope.py`. That one is about narrowed
credentials behaving like full ones; this one is a missing write check that no
amount of scoping would have covered — a viewer holding every scope should still
be refused.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import textwrap

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

REPO = pathlib.Path(__file__).resolve().parent.parent
SERVERLESS = REPO / "routes" / "serverless.py"

#: Handlers that change something a user owns. Named rather than inferred from
#: the HTTP verb: `POST .../test/run` mutates, and a future GET that mutates
#: would be a different bug worth failing on separately.
MUTATING_HANDLERS = (
    "api_serverless_dashboard_cancel_job",
    "api_serverless_delete_endpoint",
    "api_serverless_patch_endpoint",
    "api_serverless_warm_endpoint",
)

#: Either helper answers "may this caller write here". The `/v1` family uses the
#: second because it also accepts endpoint keys.
WRITE_GUARDS = {"_require_serverless_endpoint_write", "_resolve_serverless_endpoint_auth"}


def _calls_in(handler_name: str) -> set[str]:
    tree = ast.parse(SERVERLESS.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == handler_name:
            return {
                getattr(c.func, "id", "") or getattr(c.func, "attr", "")
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
            }
    raise AssertionError(f"{handler_name} not found in routes/serverless.py")


def test_the_handlers_are_still_where_this_thinks_they_are():
    """Prove the reach — a rename would make the assertions below vacuous."""
    for name in MUTATING_HANDLERS:
        assert _calls_in(name), f"{name} has no calls; the parser is broken"


@pytest.mark.parametrize("handler", MUTATING_HANDLERS)
def test_every_mutation_requires_write_access(handler: str):
    """The defect, and its three siblings that always had the check."""
    calls = _calls_in(handler)
    assert WRITE_GUARDS & calls, (
        f"{handler} mutates but never establishes write access — "
        "`_get_endpoint_for_user` is a read check, so a team viewer would be "
        "admitted. Every other mutation on this surface calls "
        "`_require_serverless_endpoint_write`."
    )


def test_the_two_cancel_routes_agree():
    """One operation must not have two different authorisation answers.

    Both call `_svc().cancel_inflight_job(...)`. If the v1 twin were ever
    relaxed instead of the v2 one being tightened, this still fails — the
    assertion is that they agree, not that either is written a particular way.
    """
    v2 = _calls_in("api_serverless_dashboard_cancel_job")
    v1 = _calls_in("api_serverless_cancel_job")
    assert bool(WRITE_GUARDS & v2) == bool(WRITE_GUARDS & v1), (
        "the two cancel routes disagree about whether cancelling needs write "
        f"access — v2 guards: {sorted(WRITE_GUARDS & v2)}, "
        f"v1 guards: {sorted(WRITE_GUARDS & v1)}"
    )


def test_the_write_guard_still_refuses_a_team_viewer():
    """Calibration for the premise.

    If `_require_serverless_endpoint_write` stopped consulting
    `_require_team_instance_write`, every assertion above would keep passing
    while guarding nothing — and the viewer hole would have moved rather than
    closed.
    """
    from routes import _deps

    source = textwrap.dedent(inspect.getsource(_deps._require_serverless_endpoint_write))
    assert "_require_team_instance_write" in source, (
        "the serverless write guard no longer defers to the team-role check, so "
        "it no longer refuses a viewer and this file is asserting nothing"
    )
