"""RFC 9457 (``application/problem+json``) errors for the versioned API (§18.5).

Every ``/api/v1`` error is a problem document with a stable machine ``code``, a
``retryable`` signal, a ``retry_after_ms`` hint, a ``trace_id`` for end-to-end
correlation, and an optional ``errors`` array for field-level problems. This is
deliberately **scoped to the typed control-plane error** (``LaunchPlanError``
and anything that raises :class:`ProblemException`) so it never changes the
legacy API's ``{"ok": false, "error": {...}}`` shape and carries zero blast
radius for existing clients.

A generated MCP / SDK client (B5.2, B6.1) decodes this one shape for every v1
failure instead of guessing per endpoint.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

PROBLEM_MEDIA_TYPE = "application/problem+json"

_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

# Codes whose failure is transient — a client may retry with backoff. Kept as a
# small explicit set so "retryable" is a deliberate contract, not a guess.
_RETRYABLE_CODES = {"rate_limited", "capacity_unavailable", "service_unavailable"}


def current_trace_id() -> str:
    """The active W3C trace id if a span is in scope, else a fresh id.

    Never raises — observability being unconfigured must not break error
    handling.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is not None and getattr(ctx, "trace_id", 0):
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return uuid.uuid4().hex


def problem_response(
    *,
    status: int,
    code: str,
    detail: str,
    title: str | None = None,
    retryable: bool | None = None,
    retry_after_ms: int | None = None,
    errors: list[dict[str, Any]] | None = None,
    trace_id: str | None = None,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build an ``application/problem+json`` response (RFC 9457)."""
    body: dict[str, Any] = {
        "type": f"https://xcelsior.ca/problems/{code}",
        "title": title or _TITLES.get(status, "Error"),
        "status": status,
        "detail": detail,
        "code": code,
        "retryable": bool(code in _RETRYABLE_CODES if retryable is None else retryable),
        "retry_after_ms": retry_after_ms,
        "trace_id": trace_id or current_trace_id(),
        "errors": errors,
    }
    # RFC 9457 permits extension members alongside the standard ones (e.g. a
    # replacement plan for a quote_changed conflict).
    if extra:
        for k, v in extra.items():
            body.setdefault(k, v)
    hdrs = dict(headers or {})
    if retry_after_ms is not None:
        hdrs.setdefault("Retry-After", str(max(0, retry_after_ms // 1000)))
    return JSONResponse(
        status_code=status, content=body, media_type=PROBLEM_MEDIA_TYPE, headers=hdrs
    )


class ProblemException(Exception):
    """Raise from a v1 handler to emit an ``application/problem+json`` error."""

    def __init__(
        self,
        *,
        status: int,
        code: str,
        detail: str,
        title: str | None = None,
        retryable: bool | None = None,
        retry_after_ms: int | None = None,
        errors: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.title = title
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
        self.errors = errors
        self.extra = extra

    def to_response(self) -> JSONResponse:
        return problem_response(
            status=self.status,
            code=self.code,
            detail=self.detail,
            title=self.title,
            retryable=self.retryable,
            retry_after_ms=self.retry_after_ms,
            errors=self.errors,
            extra=self.extra,
        )


def register_problem_handlers(app) -> None:
    """Install problem+json handlers for the typed v1 errors on ``app``."""
    from control_plane.launch.service import LaunchPlanError

    @app.exception_handler(LaunchPlanError)
    async def _launch_plan_error(_: Request, exc: LaunchPlanError):  # noqa: ANN202
        return problem_response(
            status=exc.status,
            code=exc.code,
            detail=exc.detail,
            retryable=getattr(exc, "retryable", False),
        )

    @app.exception_handler(ProblemException)
    async def _problem_exception(_: Request, exc: ProblemException):  # noqa: ANN202
        return exc.to_response()
