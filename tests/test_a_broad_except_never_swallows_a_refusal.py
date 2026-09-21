"""`except Exception` must not eat the `HTTPException` raised beside it.

`HTTPException` is an ordinary `Exception`. A `try` block that raises one for
policy reasons — a failed signature, a payload over the limit — and is followed
by `except Exception:` catches its own refusal, and the handler carries on to
whatever it does when nothing went wrong.

Both Facebook callbacks shipped this. The signature check raised
`HTTPException(400, ...)` two lines above an `except Exception` that logged it
as a parse failure and fell through to `{"status": "success"}`, so a forged
`signed_request` was accepted in production with the app secret correctly
configured. Nothing observable distinguished that from working: the route
returned 200 either way, which is what it is supposed to do for a *valid*
request.

The codebase already gets this right in the two other places where the shape
occurs (`routes/agent.py`'s gzip bound and `routes/health.py`'s identity gate),
both by putting `except HTTPException: raise` first. That is the fix this test
asks for, and it is why the test starts out passing.
"""

from __future__ import annotations

import ast

from tests._source_tree import iter_source_files, read_source

BROAD = {"Exception", "BaseException"}


def _caught_names(handler: ast.ExceptHandler) -> set[str] | None:
    """Exception names a handler catches; None means a bare `except:`."""
    if handler.type is None:
        return None
    if isinstance(handler.type, ast.Name):
        return {handler.type.id}
    if isinstance(handler.type, ast.Tuple):
        return {e.id for e in handler.type.elts if isinstance(e, ast.Name)}
    return set()


def _reraises(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            if node.exc is None:
                return True
            if isinstance(node.exc, ast.Name) and handler.name and node.exc.id == handler.name:
                return True
            # `raise HTTPException(...)` in the handler also preserves a refusal.
            func = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if (getattr(func, "id", None) or getattr(func, "attr", None)) == "HTTPException":
                return True
    return False


def _http_exception_lines(body: list[ast.stmt]) -> list[int]:
    lines = []
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Raise) and node.exc is not None:
                func = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if (getattr(func, "id", None) or getattr(func, "attr", None)) == "HTTPException":
                    lines.append(node.lineno)
    return lines


def test_no_handler_catches_the_refusal_it_just_raised() -> None:
    offenders: list[str] = []
    for path, rel in sorted(iter_source_files(), key=lambda pair: pair[1]):
        try:
            tree = ast.parse(read_source(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            raised = _http_exception_lines(node.body)
            if not raised:
                continue
            # An earlier handler that catches HTTPException and re-raises
            # protects every handler after it — the established fix.
            protected = False
            for handler in node.handlers:
                caught = _caught_names(handler)
                if caught is not None and "HTTPException" in caught and _reraises(handler):
                    protected = True
                    break
                broad = caught is None or bool(caught & BROAD)
                if broad and not _reraises(handler):
                    offenders.append(
                        f"{rel}:{handler.lineno}: except {sorted(caught) if caught else 'bare'} "
                        f"swallows the HTTPException raised at line(s) {raised}"
                    )
                    break
            if protected:
                continue

    assert not offenders, (
        "a broad except catches the HTTPException raised in its own try block, so "
        "the refusal never reaches the caller and the handler continues as though "
        "the check passed. Put `except HTTPException: raise` first, or narrow the "
        "except to the errors the try block can actually produce:\n  "
        + "\n  ".join(offenders)
    )
