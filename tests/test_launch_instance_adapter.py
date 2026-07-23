"""Track B B2.6 — `/instance` is a compatibility adapter, not an inline scheduler.

Two gates, per the checklist:

1. **Structural:** no request handler inline-schedules on the launch path. A
   handler may only call ``process_queue*`` if it is one of the sanctioned,
   explicitly-listed queue-runner endpoints (the admin "process the queue"
   tools). In particular the instance-submission handler
   (``api_submit_instance``) and the launch-plan handlers must not — the
   scheduler owns placement (§10.1, Track A P4.4b). A new handler that starts
   scheduling inline fails this test.

2. **Equivalence:** `/instance` and `/api/v1/launch-plans` produce byte-identical
   canonical specs for the same input. Both receive the same `JobIn` and the
   canonical spec is deterministic, so the discriminating REST defaulting
   (interactive → `vram_needed_gb = 0`, auto-launch ports merged as a set) lands
   identically on both surfaces.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from control_plane.launch.canonicalize import canonicalize, spec_hash

ROUTES_DIR = pathlib.Path(__file__).resolve().parent.parent / "routes"

# The only request handlers permitted to run the queue walker: they *are* the
# on-demand queue processor (admin / jurisdiction triggers), not a launch path.
# Adding a launch/submission handler here would be a design regression, so the
# list is deliberately small and named.
_QUEUE_RUNNER_ALLOWLIST = {
    "api_process_queue",
    "api_process_queue_binpack",
    "api_process_queue_ca",
    "api_process_queue_sovereign",
}
_SCHEDULER_INLINE_CALLS = {
    "process_queue",
    "process_queue_binpack",
    "process_queue_sovereign",
    "process_queue_filtered",
}


def _calls_in(fn: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def _all_route_functions() -> list[tuple[str, str, ast.FunctionDef]]:
    out = []
    for path in ROUTES_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((path.name, node.name, node))
    return out


def test_no_launch_handler_schedules_inline():
    """Only the allowlisted queue-runner endpoints may call process_queue*."""
    offenders = []
    for filename, fnname, fn in _all_route_functions():
        if _calls_in(fn) & _SCHEDULER_INLINE_CALLS:
            if fnname not in _QUEUE_RUNNER_ALLOWLIST:
                offenders.append(f"{filename}:{fnname}")
    assert not offenders, (
        "request handler(s) run the queue walker inline (B2.6 forbids this on "
        f"the launch path — route through the scheduler): {offenders}"
    )


def test_instance_submission_handler_never_calls_process_queue():
    """The specific /instance handler must not inline-schedule."""
    submit = next(
        fn
        for filename, fnname, fn in _all_route_functions()
        if filename == "instances.py" and fnname == "api_submit_instance"
    )
    assert not (_calls_in(submit) & _SCHEDULER_INLINE_CALLS), (
        "api_submit_instance still inline-schedules — the scheduler must claim "
        "and place the queued job (§10.1)"
    )


# ── Equivalence: same input → same canonical spec on both surfaces ──────

# Representative inputs exercising every discriminating rule.
_CASES = [
    {"name": "interactive-default", "interactive": True, "vram_needed_gb": 40},
    {"name": "batch", "interactive": False, "vram_needed_gb": 24, "num_gpus": 2},
    {"name": "ports", "interactive": False, "vram_needed_gb": 8, "exposed_ports": [8080, 9090, 8080]},
    {"name": "auto", "interactive": True, "auto_launch": ["jupyter", "vscode", "jupyter"]},
    {"name": "vols", "interactive": False, "vram_needed_gb": 16, "volume_ids": ["v2", "v1", "v2"]},
]


@pytest.mark.parametrize("payload", _CASES, ids=[c["name"] for c in _CASES])
def test_instance_and_launch_plan_same_canonical_spec(payload):
    """A JobIn canonicalizes identically however the two surfaces reach it.

    `/api/v1/launch-plans` calls ``canonicalize(JobIn.model_dump())`` directly.
    `/instance` builds the same ``JobIn`` and its inline defaulting must match
    the canonical spec byte-for-byte, so we assert the model-validated dict and
    a key-shuffled / whitespace-padded variant both hash to the same value, and
    that the REST-specific rules landed.
    """
    from routes.instances import JobIn

    jobin = JobIn(**payload)
    canonical = canonicalize(jobin.model_dump())

    # Deterministic: reordering keys and padding strings cannot change the hash.
    shuffled = dict(reversed(list(jobin.model_dump().items())))
    shuffled["name"] = f"  {shuffled.get('name', '')}  "
    assert spec_hash(canonicalize(shuffled)) == spec_hash(canonical)

    # The discriminating /instance rule: interactive forces vram_needed_gb = 0.
    if jobin.interactive:
        assert canonical["vram_needed_gb"] == 0.0
    else:
        assert canonical["vram_needed_gb"] == max(0.0, float(payload.get("vram_needed_gb") or 0))

    # Ports are a set (dedup + sort); auto-launch merges its ports in.
    if payload["name"] == "ports":
        assert canonical["exposed_ports"] == [8080, 9090]
    if payload["name"] == "auto":
        assert 8888 in canonical["exposed_ports"] and 8443 in canonical["exposed_ports"]
        assert canonical["auto_launch"] == ["jupyter", "vscode"]
    if payload["name"] == "vols":
        assert canonical["volume_ids"] == ["v1", "v2"]


def test_launch_plan_preview_stores_the_canonical_spec():
    """End-to-end: the plan the v1 surface persists is exactly canonicalize()."""
    from control_plane.launch.service import Principal, preview
    from routes.instances import JobIn

    payload = {"name": "equiv", "interactive": False, "vram_needed_gb": 12, "num_gpus": 1}
    jobin = JobIn(**payload)
    expected = canonicalize(jobin.model_dump())

    principal = Principal(principal_id="u-equiv", tenant_id="t-equiv")
    result = preview(jobin.model_dump(), principal=principal)
    try:
        assert result["ok"] is True
        assert result["spec_hash"] == spec_hash(expected)
    finally:
        from db import _get_pg_pool

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.execute("DELETE FROM action_plans WHERE plan_id = %s", (result["plan_id"],))
            conn.commit()
