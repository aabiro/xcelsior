"""The ingress gate must not answer 410 for faults that are not ingress.

`AgentIngressMiddleware` refuses worker paths with `410 agent_ingress_retired`
when the request did not come through the private gateway. The check was written
as:

    try:
        from control_plane.identity import gateway_headers_authenticated
        if gateway_headers_authenticated(headers):
            await self.app(scope, receive, send)   # ← inside the try
            return
    except Exception:      # "fail closed on import trouble"
        pass
    → 410

The `except` was meant to cover an `ImportError` on the line above it. Because
`await self.app(...)` sat inside the same block, **every** exception raised
downstream — a 403 from `_require_host_operator`, a database error, a bug in a
route — was swallowed and answered `410 agent_ingress_retired`.

That is worse than a wrong status code, because the message is actively
misleading: *"The worker protocol has moved to the private agent gateway. Point
XCELSIOR_SCHEDULER_URL at https://agent.xcelsior.ca and enrol this host in the
SPIRE trust domain."*

It was found on a correctly-configured gateway. mTLS verified, the CN mapped,
the shared secret matched, `/agent/*` returning 200 — and `PUT /host` answering
"you need to migrate", which is advice to rebuild infrastructure that already
worked. The tell was that an **incomplete** body returned 422 while a
**complete** one returned 410: validation is handled inside the app and raises
nothing, so only a request that reached the route could trip the swallow.

## What these assert

That the gate still fails closed for its real purpose, and that a downstream
error surfaces as itself.
"""

from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = ROOT / "api.py"


def _middleware_call_source() -> str:
    src = API.read_text(encoding="utf-8")
    start = src.index("class AgentIngressMiddleware")
    body = src[start : start + 6000]
    call = body.index("async def __call__")
    return body[call : body.index("app.add_middleware", call)]


def test_the_downstream_call_is_not_inside_the_try():
    """The regression, stated structurally rather than by behaviour."""
    src = _middleware_call_source()
    try_at = src.index("try:")
    except_at = src.index("except Exception", try_at)
    guarded = src[try_at:except_at]
    assert "await self.app(" not in guarded, (
        "`await self.app(...)` is inside the `try` again, so every downstream "
        "exception is swallowed and answered `410 agent_ingress_retired` — "
        "telling an operator to migrate a gateway that is already working."
    )


def test_the_guard_still_covers_the_import_it_was_written_for():
    src = _middleware_call_source()
    try_at = src.index("try:")
    except_at = src.index("except Exception", try_at)
    guarded = src[try_at:except_at]
    assert "from control_plane.identity import" in guarded, (
        "the import is no longer guarded; an ImportError would now propagate "
        "instead of failing closed"
    )


def test_it_still_fails_closed_when_the_identity_check_cannot_run():
    """`authenticated = False` on error, never a pass-through."""
    src = _middleware_call_source()
    assert re.search(r"except Exception:.*\n\s+authenticated = False", src), (
        "the except branch must set `authenticated = False`. Anything else "
        "risks an unauthenticated request reaching worker endpoints when the "
        "identity module fails to import."
    )


def test_the_410_is_still_reachable_for_genuine_public_ingress():
    """The gate's actual job must survive the fix."""
    src = _middleware_call_source()
    assert "status_code=410" in src
    assert "agent_ingress_retired" in src
