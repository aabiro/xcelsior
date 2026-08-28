"""The reason an instance is queued was readable by agents and not by people.

`control_plane/scheduler/explain.py` builds a bounded explanation, the scheduler
persists it on the attempt, and
`GET /api/v1/instances/{job_id}/placement-explanation` serves it tenant-scoped.
`explain_instance_placement` — a published MCP tool — has read it since it
shipped. The browser never called the route.

So an agent could tell someone *why* their instance was waiting, and the
dashboard showed them a spinner and a timeline that said "queued". B6.5 asks for
exactly the missing half: *"plain-language current reason ('Queued because no
healthy H100 with 80 GB is available in Ontario')"*.

This is the fourth surface in a week where the agent saw more than the person —
after the host-key fingerprint, the provider payout state, and the trust ladder.

## The redaction line, which is the part worth guarding

The stored payload also carries a per-host `rejections` map: `host_not_ready` /
"host status is drained", `host_observation_stale` / "heartbeat is stale", keyed
by host id — for hosts this job was **not** placed on. That is other tenants'
fleet state. §20.3 says customers see redacted infrastructure detail, so the
customer component renders `rejection_summary` (counts per constraint code) and
never the per-host map.

`aggregate_reason` is asserted to keep counts only, because the component's
safety rests on the aggregate never gaining a host id or a message.
"""

from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

from control_plane.scheduler.explain import build_explanation  # noqa: E402
from control_plane.scheduler.filters import FilterReason, aggregate_reason  # noqa: E402
from tests._source_tree import strip_ts_comments  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPONENT = ROOT / "frontend/src/components/instances/placement-explanation.tsx"
PAGE = ROOT / "frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx"
API_TS = ROOT / "frontend/src/lib/api.ts"
MCP_TOOL = ROOT / "mcp/src/tools/diagnostics.ts"


def _explanation() -> dict:
    return build_explanation(
        job={"gpu_model": "H100", "num_gpus": 1, "vram_needed_gb": 80, "region": "Ontario"},
        host_count=3,
        rejections={
            "host-secret-1": [FilterReason("host_not_ready", "host status is drained", {})],
            "host-secret-2": [
                FilterReason("gpu_model_mismatch", "requires H100, host has A10", {})
            ],
        },
        ranked=[],
        selected_host_id=None,
    )


def test_the_aggregate_carries_counts_and_no_host_identity():
    """The customer surface's whole safety argument rests on this."""
    summary = aggregate_reason(
        {
            "host-secret-1": [FilterReason("host_not_ready", "host status is drained", {})],
            "host-secret-2": [FilterReason("host_not_ready", "host status is drained", {})],
        }
    )
    flat = repr(summary)
    assert "host-secret" not in flat, f"aggregate_reason leaks host identity: {summary}"
    assert "drained" not in flat, f"aggregate_reason leaks host state: {summary}"
    assert summary["failed_constraints"] == {"host_not_ready": 2}


def test_the_full_payload_does_carry_host_detail_which_is_why_the_ui_filters():
    """Calibration: prove there is something to redact, or the guard is vacuous."""
    payload = _explanation()
    flat = repr(payload)
    assert "host-secret-1" in flat, (
        "the stored explanation no longer carries per-host detail. If that is "
        "deliberate the UI's filtering is now belt-and-braces — but this test "
        "was written because it did, and a vacuous guard is worse than none."
    )
    assert "drained" in flat


def test_the_customer_component_reads_the_aggregate_and_not_the_per_host_map():
    source = strip_ts_comments(COMPONENT.read_text(encoding="utf-8"))
    assert "rejection_summary" in source, "the component no longer reads the safe aggregate"
    assert not re.search(r"payload\.rejections\b", source), (
        "the component reads `payload.rejections` — the per-host map naming "
        "hosts this job was not placed on, with their administrative state. "
        "§20.3: customers see redacted infrastructure detail."
    )


def test_the_browser_actually_calls_the_route():
    """Typed is not called; called is not rendered. Check all three."""
    assert "placement-explanation" in API_TS.read_text(encoding="utf-8"), (
        "frontend/src/lib/api.ts no longer declares the placement-explanation call"
    )
    page = strip_ts_comments(PAGE.read_text(encoding="utf-8"))
    assert "fetchPlacementExplanation(" in page, (
        "the instance page does not fetch the explanation, so the component "
        "renders its empty state forever — the defect P5's placement control "
        "shipped with"
    )
    assert "<PlacementExplanation" in page, "the component is fetched but never rendered"
    assert "explained={placement.explained}" in page, (
        "the component is rendered without `explained`, so an attempt that "
        "recorded nothing is indistinguishable from one with no reason"
    )


def test_the_timeline_and_lease_routes_reach_the_browser_too():
    """B6.5's other half, found by the same sweep.

    `/api/v1/instances/{id}/timeline` and `/active-lease` were read only by
    `get_instance_timeline` and `get_active_lease` — both MCP tools — while the
    browser called neither. The dashboard's five status pills were the whole
    story, so a first attempt that failed before the successful one was
    invisible to the person and legible to their agent.
    """
    api = API_TS.read_text(encoding="utf-8")
    for route in ("/timeline", "/active-lease"):
        assert route in api, f"frontend/src/lib/api.ts no longer declares {route}"

    page = strip_ts_comments(PAGE.read_text(encoding="utf-8"))
    assert "fetchInstanceTimeline(" in page and "fetchActiveLease(" in page, (
        "the instance page does not fetch the attempts or the lease"
    )
    assert "<AttemptTimeline" in page, "fetched but never rendered"


def test_the_attempt_type_does_not_carry_the_explanation_blob():
    """The timeline route includes it; typing it invites a row to render it.

    `_attempts_for_job` selects `placement_explanation` per attempt, and that
    payload holds the per-host rejections map — the same fleet state
    `PlacementExplanation` refuses to show. Leaving it off `InstanceAttempt`
    means a timeline row cannot reach it without a cast, which is a deliberate
    act rather than an accident.
    """
    api = strip_ts_comments(API_TS.read_text(encoding="utf-8"))
    match = re.search(r"export interface InstanceAttempt \{(.*?)\n\}", api, re.DOTALL)
    assert match, "InstanceAttempt is gone; re-point this guard"
    assert "placement_explanation" not in match.group(1), (
        "`InstanceAttempt` now types `placement_explanation`, which carries the "
        "per-host rejections map. The customer surface renders the redacted "
        "aggregate from the dedicated endpoint instead."
    )


def test_the_lease_route_aliases_the_host_rather_than_naming_it():
    """The component renders `host_alias` as-is, so the route must redact it."""
    source = (ROOT / "routes/control_plane_v1.py").read_text(encoding="utf-8")
    # The whole function, not up to the first `return` — the handler returns
    # early for "no lease", and matching that stops before the aliasing code.
    # The first draft did exactly that and failed against a correct route.
    match = re.search(
        r"def api_v1_instance_active_lease\(.*?(?=\n@router\.|\ndef )", source, re.DOTALL
    )
    assert match, "the active-lease handler moved; re-point this guard"
    body = match.group(0)
    assert "host_alias" in body, "the lease no longer carries an alias"
    assert 'lease.pop("host_id")' in body, (
        "the raw `host_id` is no longer popped from the lease payload, so the "
        "component — which renders what it is given — would print it"
    )


def test_the_agent_and_the_browser_read_the_same_route():
    """Parity: one route, so the two surfaces cannot describe different attempts."""
    tool = MCP_TOOL.read_text(encoding="utf-8")
    match = re.search(r"explain_instance_placement.*?placement-explanation", tool, re.DOTALL)
    assert match, (
        "`explain_instance_placement` no longer calls the placement-explanation "
        "route. If it moved, the browser must move with it."
    )


def test_a_queued_payload_has_what_the_sentence_needs():
    """The UI composes its sentence from `request`; assert the fields survive."""
    payload = _explanation()
    assert payload["request"]["gpu_model"] == "H100"
    assert payload["request"]["vram_needed_gb"] == 80
    assert payload["request"]["region"] == "Ontario"
    # No host chosen means the UI must find a queue reason rather than guess.
    assert "selected_host_id" not in payload
    assert payload["queue_reason_code"] == "no_eligible_host"
