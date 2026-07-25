"""Track B B4.7 — inventory of every event/SSE emitter call site (§16.2/DA§12.2).

Track A left a set of process-local `broadcast_sse` / `emit_event` emitters that
should either route through the durable outbox (`try_append_lifecycle_outbox`)
or be explicitly documented as a legitimate latency-only, UI-facing broadcast.
This is the structural gate: it discovers every non-test emitter call site (by
file, via AST) and asserts each is classified. A **new, unclassified** emitter
fails CI — the drift can never re-open silently.

Classification is coarse (per file) and one of:
  * ``durable``       — the durable/persistence path itself, or the file also
                        writes the outbox so the SSE is a latency optimization
                        behind persistence;
  * ``process_local`` — a UI-facing, latency-only broadcast with no durable
                        requirement (the transition it mirrors is persisted
                        elsewhere). These are the residuals B4.7 routes through
                        the outbox surface-by-surface.
  * ``primitive``     — defines the emitter/audit primitive itself.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
_EMITTERS = {"broadcast_sse", "emit_event"}

# Every non-test file that CALLS an emitter, with its classification + why.
_EMITTER_FILES: dict[str, str] = {
    # ── durable / persistence path ────────────────────────────────────
    "scheduler.py": "durable — job lifecycle routes through try_append_lifecycle_outbox; SSE mirrors it",
    "routes/instances.py": "durable — instance transitions are persisted via the scheduler outbox; these SSE calls mirror for the UI",
    # ── emitter primitive ─────────────────────────────────────────────
    "api.py": "primitive — app-level SSE broadcast wiring",
    # ── process-local, UI-facing (B4.7 residuals to route through outbox) ──
    "routes/hosts.py": "process_local — host UI SSE (hosts dual-emit residual)",
    "routes/volumes.py": "process_local — volume UI SSE (volumes residual)",
    "routes/teams.py": "process_local — team UI SSE (teams residual)",
    "routes/billing.py": "process_local — wallet/billing UI SSE (billing wallet residual)",
    "routes/agent.py": "process_local — agent telemetry/command UI SSE (agent telemetry residual)",
    "routes/admin.py": "process_local — admin dashboard UI SSE",
    "routes/auth.py": "process_local — auth/session UI SSE",
    "routes/providers.py": "process_local — provider UI SSE",
    "routes/verification.py": "process_local — host verification UI SSE",
    "routes/inference.py": "process_local — inference UI SSE",
    "routes/serverless.py": "process_local — serverless UI SSE",
    "serverless/service.py": "process_local — serverless worker UI SSE",
    "routes/spot.py": "process_local — spot pricing UI SSE",
    "routes/jurisdiction.py": "process_local — jurisdiction queue UI SSE",
}


def _emitter_files() -> set[str]:
    found: set[str] = set()
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", ".venv/", "venv/", "node_modules/", "mcp/")):
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                if name in _EMITTERS:
                    found.add(rel)
                    break
    return found


def test_every_emitter_call_site_is_classified():
    found = _emitter_files()
    allow = set(_EMITTER_FILES)
    unclassified = found - allow
    assert not unclassified, (
        "new event/SSE emitter file(s) are not classified (B4.7): "
        f"{sorted(unclassified)} — route through the durable outbox "
        "(try_append_lifecycle_outbox) or add an explicit classification to "
        "_EMITTER_FILES with a justification"
    )
    # Keep the inventory honest: a listed file that no longer emits must be
    # removed, so the list can never rot into a rubber stamp.
    stale = allow - found
    assert not stale, f"inventory lists non-emitters (remove them): {sorted(stale)}"
