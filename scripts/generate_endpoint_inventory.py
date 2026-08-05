#!/usr/bin/env python3
"""Emit every API endpoint with method, path, auth dependency, and docstring.

This is step 1 of GT0 in `docs/mcp-tool-surface-plan.md`: the tool-surface
enumeration is an *audit* deliverable, not a guess from route names. The plan is
explicit — "I am not going to invent tool names from endpoint paths I haven't
read" — so the classification columns are emitted deliberately blank. A human
(or a research pass) fills `class` with one of:

    covered    a tool already reaches it
    gap        a journey needs it and no tool reaches it
    internal   never exposed, with a reason in `notes`
    redundant  folded into another tool, named in `notes`

Zero rows may remain unclassified when GT0 closes.

The point of the file is condensation: 41 route modules and ~14k lines become
one table a reviewer can read in a sitting, which is what makes the surface
argument checkable by someone who did not write it.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_PATH = ROOT / "docs" / "generated" / "endpoint-inventory.md"
#: GT0's judgement, hand-maintained. Separate from the generated table so that
#: regenerating the inventory does not erase the audit.
CLASSIFICATION_PATH = ROOT / "docs" / "endpoint-classification.json"
HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
# Machinery, not product surface. Listing these buries the endpoints that matter.
SKIP_PREFIXES = ("/static", "/docs", "/redoc", "/openapi.json")


# This codebase authenticates *inside* handler bodies (`_require_auth(request)`),
# not through FastAPI `Depends`. Reading only `route.dependant` reports "no
# dependency" for all 528 operations, which a reader would reasonably — and
# dangerously — take to mean "unauthenticated". So the guard is recovered from
# the handler source as well, and the two sources are labelled differently.
_AUTH_CALLS = (
    "_require_scope",
    "_require_auth",
    "_require_admin",
    "_require_agent_auth",
    "_require_provider_access",
    "_require_provider_or_admin",
    "_require_host_operator",
    "_require_worker_callback",
    "_require_worker_status_update",
    "_get_current_user",
    "validate_key",
)


def _auth_dependencies(route, endpoint) -> str:
    """The guards that run for this route, from both places they can live.

    Reported rather than interpreted: "which guard runs here" is exactly the
    question a reviewer asks, and paraphrasing it into `public`/`authed` would
    throw away the distinction between guards that differ in what they enforce.
    `_get_current_user` in particular is *not* a guard — it resolves an optional
    principal — so it is reported under its own name and never merged with the
    `_require_*` family.
    """
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    def walk(dependant, depth: int = 0) -> None:
        if depth > 4:
            return
        for sub in getattr(dependant, "dependencies", []) or []:
            add(getattr(getattr(sub, "call", None), "__name__", None))
            walk(sub, depth + 1)

    walk(getattr(route, "dependant", None) or object())

    try:
        source = inspect.getsource(endpoint)
    except (OSError, TypeError):
        source = ""
    for call in _AUTH_CALLS:
        if f"{call}(" in source:
            add(call)

    return ", ".join(names) or "none found — verify by hand"


def _summary(endpoint) -> str:
    doc = inspect.getdoc(endpoint) or ""
    first = doc.strip().split("\n", 1)[0].strip()
    return first.replace("|", "\\|") or "—"


def _module(endpoint) -> str:
    mod = getattr(endpoint, "__module__", "") or ""
    return mod.rsplit(".", 1)[-1] if mod else "—"


def collect(app) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None)
        endpoint = getattr(route, "endpoint", None)
        if not path or not methods or endpoint is None:
            continue
        if any(path.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        for method in sorted(m for m in methods if m in HTTP_METHODS):
            rows.append(
                {
                    "method": method,
                    "path": path,
                    "module": _module(endpoint),
                    "auth": _auth_dependencies(route, endpoint),
                    "summary": _summary(endpoint),
                }
            )
    rows.sort(key=lambda r: (r["module"], r["path"], r["method"]))
    return rows


VALID_CLASSES = {"covered", "gap", "internal", "redundant"}


def load_classification() -> dict[str, dict[str, str]]:
    """GT0's labels, keyed ``"METHOD /path"``.

    Kept beside this generator rather than inside its output because the output
    is regenerated: 158 rows of audit were filled into the table by hand once,
    and the next regeneration would have erased every one of them. Storing the
    judgement separately is what makes the audit survive its own tooling.

    A stale key — a label for an endpoint that no longer exists — is reported by
    `tests/test_gt0_classification_ratchet.py` rather than ignored here, because
    a route being deleted is exactly when its classification should be revisited.
    """
    if not CLASSIFICATION_PATH.exists():
        return {}
    data = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))
    for key, entry in data.items():
        label = entry.get("class", "")
        if label not in VALID_CLASSES:
            raise SystemExit(f"{key}: {label!r} is not one of {sorted(VALID_CLASSES)}")
        if len(entry.get("notes", "")) < 8:
            raise SystemExit(f"{key}: labelled {label!r} with no reason")
    return data


def render(rows: list[dict[str, str]]) -> str:
    by_module: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_module.setdefault(row["module"], []).append(row)

    classification = load_classification()
    classified = sum(
        1 for row in rows if f"{row['method']} {row['path']}" in classification
    )

    out: list[str] = [
        "# Endpoint inventory (generated)",
        "",
        "Generated by `scripts/generate_endpoint_inventory.py`. **Do not edit by "
        "hand** — regenerate it. GT0 step 1 of "
        "[mcp-tool-surface-plan.md](../mcp-tool-surface-plan.md).",
        "",
        f"**{len(rows)} operations across {len(by_module)} modules.**",
        "",
        "**Auth column:** this codebase authenticates inside handler bodies rather "
        "than via FastAPI `Depends`, so the guard is recovered from the handler "
        "source. `_get_current_user` resolves an *optional* principal and is not "
        "a guard. `none found` means exactly that — verify by hand before "
        "concluding an endpoint is public.",
        "",
        f"**{classified} of {len(rows)} operations classified.** `class` is "
        "`covered` / `gap` / `internal` / `redundant`, each with a reason, and "
        "GT0 closes only when zero rows are unclassified. Edit "
        "[endpoint-classification.json](../endpoint-classification.json), not "
        "this table — this file is regenerated and would discard the labels.",
        "",
    ]
    for module in sorted(by_module):
        module_rows = by_module[module]
        out.append(f"## `routes/{module}.py` — {len(module_rows)} operations")
        out.append("")
        out.append("| Method | Path | Auth dependency | Summary | class | notes |")
        out.append("|---|---|---|---|---|---|")
        for row in module_rows:
            label = classification.get(f"{row['method']} {row['path']}", {})
            out.append(
                f"| {row['method']} | `{row['path']}` | {row['auth']} "
                f"| {row['summary']} | {label.get('class', '')} "
                f"| {label.get('notes', '')} |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    os.environ.setdefault("XCELSIOR_API_TOKEN", "inventory")
    from api import app

    rows = collect(app)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    modules = len({row["module"] for row in rows})
    print(f"Wrote {args.output} — {len(rows)} operations across {modules} modules")


if __name__ == "__main__":
    main()
