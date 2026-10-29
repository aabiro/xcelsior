"""The inventory's guard list must not fall behind the guards.

`docs/generated/endpoint-inventory.md` answers "which guard runs on this
endpoint" — it is the artifact GT0 classifies from. The generator recovers the
answer by looking for calls to names in `_AUTH_CALLS`, a list maintained by
hand.

A hand-kept list of this kind has one failure mode and it is silent: a guard
that is not on it is invisible, so a route that *is* protected reads as
`none found`, which the document's own note defines as "exactly that". **192 of
557 rows said that**, and the two that mattered were genuinely unguarded routes
sitting unnoticed among 190 false ones. A reviewer either believes the column
and clears a guarded route, or learns not to trust it — and the second is how a
real gap gets waved through.

So the list is checked against the code rather than remembered. A function in
`routes/` that refuses with 401 or 403 *itself* is a guard by definition; if it
is not in `_AUTH_CALLS`, the generator cannot see it and this fails, naming it.

Route handlers are excluded: they raise 401/403 inline all the time, and they
are the thing being described, not a guard that describes them.
"""

from __future__ import annotations

import ast

from tests._source_tree import iter_source_files, read_source

GENERATOR = "scripts/generate_endpoint_inventory.py"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
REFUSAL_CODES = {401, 403}


def _known_names() -> set[str]:
    """The literals inside the generator's `_AUTH_CALLS` tuple."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / GENERATOR).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_AUTH_CALLS" for t in node.targets
        ):
            return {
                e.value
                for e in ast.walk(node.value)
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    raise AssertionError(f"no _AUTH_CALLS in {GENERATOR}")


def _is_route_handler(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(d, ast.Call) and getattr(d.func, "attr", "") in HTTP_METHODS
        for d in fn.decorator_list
    )


def _refuses_directly(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Raises HTTPException(401|403) in its own body."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        target = node.exc.func
        if (getattr(target, "id", None) or getattr(target, "attr", None)) != "HTTPException":
            continue
        for arg in list(node.exc.args) + [kw.value for kw in node.exc.keywords]:
            if isinstance(arg, ast.Constant) and arg.value in REFUSAL_CODES:
                return True
    return False


def test_the_generator_still_has_a_guard_list() -> None:
    """A rule about an empty set passes for the wrong reason."""
    assert len(_known_names()) > 5, "_AUTH_CALLS is suspiciously small or gone"


def test_every_guard_in_routes_is_known_to_the_inventory() -> None:
    known = _known_names()
    missing: list[str] = []

    for path, rel in sorted(iter_source_files(), key=lambda pair: pair[1]):
        if not rel.startswith("routes/"):
            continue
        tree = ast.parse(read_source(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_route_handler(node) or node.name in known:
                continue
            if _refuses_directly(node):
                missing.append(f"{rel}:{node.lineno} {node.name}")

    assert not missing, (
        "these refuse with 401/403 themselves, so they are guards, and "
        f"`_AUTH_CALLS` in {GENERATOR} does not know them. Every route relying "
        "on one is reported `none found` — which that document defines as "
        "meaning the endpoint has no guard at all:\n  " + "\n  ".join(missing)
    )
