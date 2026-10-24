"""Trusted request provenance for new transactional events.

Identity comes from accepted authentication, never request headers. Context
ends with its ASGI request and is absent for independent background tasks.
"""

from contextvars import ContextVar
from uuid import uuid4

_current: ContextVar[dict | None] = ContextVar("xcelsior_event_request", default=None)


class EventRequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        context = {"request_id": str(uuid4())}
        token = _current.set(context)
        try:
            return await self.app(scope, receive, send)
        finally:
            context.clear()
            _current.reset(token)


def bind_authenticated_principal(principal: dict) -> None:
    context = _current.get()
    if context is None:
        return
    # ASGI and sync dependencies copy ContextVar bindings across threadpool
    # boundaries. Mutate this request's private value so the accepted identity
    # also reaches an async endpoint after its sync dependency returns.
    actor = principal.get("user_id") or principal.get("sub") or principal.get("email")
    client = principal.get("client_id") or principal.get("oauth_client_id")
    if isinstance(actor, str) and actor:
        context["actor_id"] = actor
    if isinstance(client, str) and client:
        context["client_id"] = client


def event_context() -> dict:
    from structured_logging import _get_active_trace_context

    context = dict(_current.get() or {})
    trace, _ = _get_active_trace_context()
    if trace:
        context["trace_id"] = trace
    return context
