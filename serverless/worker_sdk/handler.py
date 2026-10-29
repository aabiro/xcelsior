"""Handler registration for queue-mode workers."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterator
from typing import Any

HandlerFn = Callable[..., Any]

_registry: HandlerFn | None = None


class CancelToken:
    """Cooperative cancellation flag passed to handlers."""

    def __init__(self, *, cancelled: bool = False):
        self._event = threading.Event()
        if cancelled:
            self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def check(self) -> None:
        if self.is_cancelled:
            raise asyncio.CancelledError("job cancelled")


def handler(fn: HandlerFn) -> HandlerFn:
    """Register the job handler for queue-mode workers."""
    global _registry
    _registry = fn
    return fn


def get_registered_handler() -> HandlerFn | None:
    return _registry


def _extract_usage(result: Any) -> tuple[Any, dict[str, int]]:
    usage: dict[str, int] = {}
    if isinstance(result, dict) and "usage" in result:
        raw = result.get("usage") or {}
        if isinstance(raw, dict):
            usage = {
                "input_tokens": int(raw.get("input_tokens") or raw.get("prompt_tokens") or 0),
                "output_tokens": int(raw.get("output_tokens") or raw.get("completion_tokens") or 0),
            }
        body = {k: v for k, v in result.items() if k != "usage"}
        return body, usage
    return result, usage


def normalize_handler_result(result: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, int]]:
    """
    Normalize handler return values:
    - dict → output (optional ``usage`` key stripped)
    - str/int/float → {"result": value}
    - generator/async generator → stream events + final output
    """
    events: list[dict[str, Any]] = []
    if result is None:
        return {}, events, {}
    body, usage = _extract_usage(result)
    if isinstance(body, dict):
        return body, events, usage
    if isinstance(body, (str, int, float, bool)):
        return {"result": body}, events, usage
    if inspect.isasyncgen(body):
        return _collect_async_gen(body, events)
    if inspect.isgenerator(body):
        return _collect_sync_gen(body, events)
    return {"result": body}, events, usage


async def invoke_handler(
    fn: HandlerFn,
    job: dict[str, Any],
    *,
    cancel: CancelToken | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, int]]:
    payload = job.get("payload") or job.get("input") or {}
    if not isinstance(payload, dict):
        payload = {"input": payload}
    job_ctx = {**job, "input": payload}
    token = cancel or CancelToken()

    kwargs: dict[str, Any] = {}
    sig = inspect.signature(fn)
    if "cancel" in sig.parameters:
        kwargs["cancel"] = token

    if inspect.iscoroutinefunction(fn):
        result = await fn(job_ctx, **kwargs) if kwargs else await fn(job_ctx)
        out, events, usage = normalize_handler_result(result)
        return out, events, usage

    result = fn(job_ctx, **kwargs) if kwargs else fn(job_ctx)
    if inspect.isawaitable(result):
        result = await result
        out, events, usage = normalize_handler_result(result)
        return out, events, usage
    if inspect.isasyncgen(result):
        out, events = await _collect_async_gen(result, [], cancel=token)
        _, usage = _extract_usage(out)
        return out, events, usage
    out, events, usage = normalize_handler_result(result)
    return out, events, usage


def _collect_sync_gen(
    gen: Generator[Any, None, None],
    events: list[dict[str, Any]],
    *,
    cancel: CancelToken | None = None,
) -> tuple[dict, list, dict[str, int]]:
    output: dict[str, Any] = {}
    usage: dict[str, int] = {}
    for item in gen:
        if cancel and cancel.is_cancelled:
            break
        if isinstance(item, dict) and item.get("type") in ("progress", "log", "output"):
            events.append(item)
        else:
            body, usage = _extract_usage(item)
            output = body if isinstance(body, dict) else {"result": body}
    return output, events, usage


async def _collect_async_gen(
    gen: AsyncGenerator[Any, None] | AsyncIterator[Any],
    events: list[dict[str, Any]],
    *,
    cancel: CancelToken | None = None,
) -> tuple[dict, list]:
    output: dict[str, Any] = {}
    async for item in gen:
        if cancel and cancel.is_cancelled:
            break
        if isinstance(item, dict) and item.get("type") in ("progress", "log", "output"):
            events.append(item)
        else:
            body, _usage = _extract_usage(item)
            output = body if isinstance(body, dict) else {"result": body}
    return output, events