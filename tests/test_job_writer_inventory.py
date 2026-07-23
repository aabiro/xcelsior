"""Track B B2.7 — inventory of every writer of a new ``jobs`` row.

`submit_job` (scheduler) is the single job-creation authority (B0.2 rule 11):
it is the only function that inserts a new ``jobs`` row (`_upsert_job_row` →
`DatabaseOps.upsert_job`). Everything that wants desired GPU state calls it.

This gate enumerates every caller and asserts each is either the unified launch
service or an **explicitly-listed, justified exception**. A new call site that
is not classified fails CI, forcing a conscious decision: route it through the
launch service, or record why it legitimately does not. It also pins the
low-level row insert to the persistence authority so no module grows a second
one.

The gate is static (AST) — no database — so it runs anywhere and cannot be
made to pass by luck of ordering.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# Every module permitted to call `submit_job`, with why. The launch service is
# THE authority; the rest are justified exceptions whose semantics differ from
# an interactive-instance launch (or are non-network operator tools).
_JOB_WRITER_ALLOWLIST: dict[str, str] = {
    "control_plane/launch/service.py": "the unified launch service — the sanctioned authority (§14)",
    "routes/instances.py": "REST /instance adapter — canonical-spec-aligned with the service (B2.6); full delegation is the B2.7 consolidation residual",
    "serverless/service.py": "serverless worker provisioning — distinct lifecycle; B3.1 binds these to fenced attempts",
    "inference.py": "PEL inference job submission — request/response inference, not an interactive instance",
    "routes/inference.py": "inference API endpoints — request/response inference, not an interactive instance",
    "ai_assistant.py": "AI assistant _tool_launch_job — a thin convenience wrapper over the REST launch surface",
    "cli.py": "local operator/dev CLI — not a network surface",
}

# The low-level ``jobs`` row insert must live only in the persistence authority.
_ROW_INSERT_ALLOWED = {"scheduler.py", "db.py"}


def _iter_py_files():
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", ".venv/", "venv/", "node_modules/", "mcp/")):
            continue
        yield path, rel


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
    )


def _calls_submit_job(tree: ast.Module) -> bool:
    """True if the module calls submit_job outside any __main__ guard."""
    # Strip __main__ demo blocks — those are not production writers.
    body = [n for n in tree.body if not _is_main_guard(n)]
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                if name == "submit_job":
                    return True
    return False


def _defines_submit_job(tree: ast.Module) -> bool:
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "submit_job"
        for n in ast.walk(tree)
    )


def test_every_job_writer_is_classified():
    """The set of modules that call submit_job == the justified allowlist."""
    callers = set()
    for path, rel in _iter_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        if _defines_submit_job(tree):
            continue  # the authority module itself defines it
        if _calls_submit_job(tree):
            callers.add(rel)

    allow = set(_JOB_WRITER_ALLOWLIST)
    unclassified = callers - allow
    assert not unclassified, (
        "new writer(s) of a jobs row are not routed through the launch service "
        "and not listed as a justified exception (B2.7): "
        f"{sorted(unclassified)} — route through control_plane.launch.service or "
        "add an explicit justification to _JOB_WRITER_ALLOWLIST"
    )
    # Keep the allowlist honest: a listed module that no longer writes jobs must
    # be removed, so the list can never rot into a rubber stamp.
    stale = allow - callers
    assert not stale, f"allowlist lists non-writers (remove them): {sorted(stale)}"


def _inserts_job_row(tree: ast.Module) -> bool:
    """AST check: an actual `upsert_job(...)` call or an `INSERT INTO jobs` SQL
    string literal — not a docstring or comment that merely names them."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "upsert_job":
                return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # SQL is executed, never a docstring — but skip the module/function
            # docstring position defensively by requiring the INSERT verb form.
            if "INSERT INTO jobs" in node.value and "(" in node.value:
                return True
    return False


def test_row_insert_stays_in_persistence_authority():
    """`INSERT INTO jobs` / DatabaseOps.upsert_job live only in the authority."""
    offenders = []
    for path, rel in _iter_py_files():
        if path.name in _ROW_INSERT_ALLOWED:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        if _inserts_job_row(tree):
            offenders.append(rel)
    assert not offenders, (
        "raw jobs-row insert outside the persistence authority "
        f"(scheduler.py / db.py): {offenders}"
    )
