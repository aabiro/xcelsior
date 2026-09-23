"""An `async def` route handler must not run database work inline.

FastAPI runs a `def` handler in a threadpool but an `async def` handler on the
event loop itself. So synchronous database work inside an `async def` blocks
every other request that worker process is serving, for as long as the queries
take — it is not slow for the caller, it is slow for everyone.

`POST /agent/logs/{job_id}` was the worst case. It has to be `async` (it awaits
`request.body()` to size-check and gunzip the payload), and it opened a
connection and ran one INSERT per log line, up to 500, inline. Every agent in
the fleet uploads logs continuously, so the loop was being held closed in a
steady stream of multi-hundred-INSERT batches.

The fix is to move the queries to a worker thread with `asyncio.to_thread`, and
it has a boundary worth stating: for the log endpoint, only the *writes* moved.
`push_job_log` calls `broadcast_sse`, which does `put_nowait` on `asyncio.Queue`
objects, and those are not thread-safe. Offloading that half as well would have
traded a latency bug for a correctness one, so it stays on the loop.

This checks the shape rather than either endpoint: a blocking call sitting
directly in an async handler's body fails; the same call inside a nested
function is fine, because that is what gets handed to `to_thread`.
"""

from __future__ import annotations

import ast

from tests._source_tree import iter_source_files

#: Entry points to synchronous database work.
BLOCKING_CALLS = {"_get_pg_pool", "pg_transaction", "_atomic_mutation"}

HTTP_DECORATORS = {"get", "post", "put", "delete", "patch", "websocket"}


def _is_route(fn: ast.AsyncFunctionDef) -> bool:
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(node, ast.Attribute) and node.attr in HTTP_DECORATORS:
            return True
    return False


def _inline_blocking_calls(fn: ast.AsyncFunctionDef) -> list[str]:
    """Blocking calls in the handler's own body, not inside a nested function.

    A nested `def` is how the offloaded work is written — it is defined here and
    handed to `asyncio.to_thread` — so calls inside one are exactly the calls
    that are *not* on the loop.
    """
    nested: set[int] = set()
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for inner in ast.walk(node):
                nested.add(id(inner))

    found = []
    for node in ast.walk(fn):
        if id(node) in nested or not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name in BLOCKING_CALLS:
            found.append(f"{name}() at line {node.lineno}")
    return found


def test_no_async_route_handler_opens_a_connection_on_the_loop():
    offenders: list[str] = []
    for path, rel in iter_source_files(include_prefixes=("routes/",)):
        if not rel.startswith("routes/"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]:
            if not _is_route(fn):
                continue
            blocking = _inline_blocking_calls(fn)
            if blocking:
                offenders.append(f"{rel}: async {fn.name} — {', '.join(blocking)}")

    assert not offenders, (
        "these async handlers run database work on the event loop, stalling every "
        "other request on the worker process; move the queries into a nested "
        "function and `await asyncio.to_thread(...)` it:\n  " + "\n  ".join(sorted(offenders))
    )
