"""Handler registration for queue-mode workers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterator
from typing import Any

HandlerFn = Callable[[dict[str, Any]], Any]

_registry: HandlerFn | None = None


def handler(fn: HandlerFn) -> HandlerFn:
    """Register the job handler for queue-mode workers."""
    global _registry
    _registry = fn
    return fn


def get_registered_handler() -> HandlerFn | None:
    return _registry


def normalize_handler_result(result: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Normalize handler return values:
    - dict → output
    - str/int/float → {"result": value}
    - generator/async generator → stream events + final output
    """
    events: list[dict[str, Any]] = []
    if result is None:
        return {}, events
    if isinstance(result, dict):
        return result, events
    if isinstance(result, (str, int, float, bool)):
        return {"result": result}, events
    if inspect.isasyncgen(result):
        return _collect_async_gen(result, events)
    if inspect.isgenerator(result):
        return _collect_sync_gen(result, events)
    return {"result": result}, events


async def invoke_handler(fn: HandlerFn, job: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    payload = job.get("payload") or job.get("input") or {}
    if not isinstance(payload, dict):
        payload = {"input": payload}
    job_ctx = {**job, "input": payload}

    if inspect.iscoroutinefunction(fn):
        result = await fn(job_ctx)
        return normalize_handler_result(result)

    result = fn(job_ctx)
    if inspect.isawaitable(result):
        result = await result
        return normalize_handler_result(result)
    if inspect.isasyncgen(result):
        return await _collect_async_gen(result, [])
    return normalize_handler_result(result)


def _collect_sync_gen(gen: Generator[Any, None, None], events: list[dict[str, Any]]) -> tuple[dict, list]:
    output: dict[str, Any] = {}
    for item in gen:
        if isinstance(item, dict) and item.get("type") in ("progress", "log", "output"):
            events.append(item)
        else:
            output = item if isinstance(item, dict) else {"result": item}
    return output, events


async def _collect_async_gen(
    gen: AsyncGenerator[Any, None] | AsyncIterator[Any],
    events: list[dict[str, Any]],
) -> tuple[dict, list]:
    output: dict[str, Any] = {}
    async for item in gen:
        if isinstance(item, dict) and item.get("type") in ("progress", "log", "output"):
            events.append(item)
        else:
            output = item if isinstance(item, dict) else {"result": item}
    return output, events