"""Unified launch service (Track B B2, blueprint §14).

Every launch surface — MCP, dashboard, REST, training, serverless — turns
its request into the *same* canonical spec, quotes it against the *same*
pricing snapshot, checks it against the *same* spend policy, and (once
approved) executes it exactly once through the *same* action plan. No
surface performs its own wallet/image/volume checks and then calls
``submit_job`` independently (§14).

Module layout mirrors the blueprint:

- ``canonicalize`` — deterministic, versioned request → canonical spec, and
  the spec hash the scheduler and worker already verify.
- ``validation`` — side-effect-free structural validation of a canonical
  spec.
- ``quoting`` — versioned pricing snapshot → estimate, in micro-CAD.
- ``spend_policy`` — a canonical spec + quote against an
  ``mcp_client_policies`` row: allowed, and auto-approvable?
- ``action_plans`` / ``service`` — the persisted plan lifecycle (B2.3+).
"""

from control_plane.launch.canonicalize import (
    CANON_SPEC_VERSION,
    canonicalize,
    spec_hash,
)

__all__ = ["CANON_SPEC_VERSION", "canonicalize", "spec_hash"]
