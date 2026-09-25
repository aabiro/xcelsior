"""§25.4 — SLO definitions, error budget computation, and invariant breach detection.

Every SLO defined here has a matching Prometheus alert rule in
`infra/observability/prometheus/alert-rules.yml`. The definitions are
kept in Python so gate tests can verify:
  1. Every SLO here has a matching alert rule.
  2. Every hard invariant has a zero-tolerance counter.
  3. Burn-rate windows are consistent.
"""
from dataclasses import dataclass, field
from enum import Enum

class SloKind(Enum):
    HARD_INVARIANT = "hard_invariant"  # must stay zero, pages immediately
    LATENCY = "latency"               # p95/p99 target
    AVAILABILITY = "availability"      # success rate target
    COMPLETENESS = "completeness"      # 100% coverage target
    FRESHNESS = "freshness"            # max staleness

@dataclass(frozen=True)
class SloDefinition:
    name: str
    kind: SloKind
    description: str
    target: float          # 0.0 for invariants, 0.9999 for availability, etc.
    metric: str            # Prometheus metric name
    alert_name: str        # Matching Prometheus alert rule name
    burn_rate_windows: list[tuple[int, int]] = field(default_factory=list)  # (short_min, long_min)
    blueprint_ref: str = ""

HARD_INVARIANTS: list[SloDefinition] = [
    SloDefinition(
        name="exclusive_allocation_collision",
        kind=SloKind.HARD_INVARIANT,
        description="No two active allocations may claim the same exclusive GPU resource",
        target=0.0,
        metric="xcelsior_scheduler_allocation_constraint_violations_total",
        alert_name="XcelsiorInvariantExclusiveAllocationCollision",
        burn_rate_windows=[(5, 60), (30, 360)],  # 5m/1h fast, 30m/6h slow
        blueprint_ref="§25.4 invariant 1",
    ),
    SloDefinition(
        name="unvetted_worker_start",
        kind=SloKind.HARD_INVARIANT,
        description="No start accepted without a valid current attempt, lease, and fence",
        target=0.0,
        metric="xcelsior_worker_unvetted_starts_total",
        alert_name="XcelsiorInvariantUnvettedWorkerStart",
        burn_rate_windows=[(5, 60), (30, 360)],
        blueprint_ref="§25.4 invariant 2",
    ),
    SloDefinition(
        name="stale_fence_mutation",
        kind=SloKind.HARD_INVARIANT,
        description="No mutation, route, secret, storage write, or billing acceptance behind a stale fence",
        target=0.0,
        metric="xcelsior_stale_fence_acceptances_total",
        alert_name="XcelsiorInvariantStaleFenceMutation",
        burn_rate_windows=[(5, 60), (30, 360)],
        blueprint_ref="§25.4 invariant 3",
    ),
    SloDefinition(
        name="premature_strict_reassignment",
        kind=SloKind.HARD_INVARIANT,
        description="No strict workload reassigned before definitive fencing",
        target=0.0,
        metric="xcelsior_premature_strict_reassignments_total",
        alert_name="XcelsiorInvariantPrematureStrictReassignment",
        burn_rate_windows=[(5, 60), (30, 360)],
        blueprint_ref="§25.4 invariant 4",
    ),
]

LATENCY_SLOS: list[SloDefinition] = [
    SloDefinition(
        name="placement_latency_p95",
        kind=SloKind.LATENCY,
        description="Placement latency p95 <= 2s",
        target=2.0,
        metric="xcelsior_scheduler_placement_duration_seconds",
        alert_name="XcelsiorSloPlacementLatency",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
    SloDefinition(
        name="assignment_to_claim_p95",
        kind=SloKind.LATENCY,
        description="Assignment to claim p95 <= two poll intervals",
        target=60.0,
        metric="xcelsior_scheduler_claim_latency_seconds",
        alert_name="XcelsiorSloAssignmentToClaimLatency",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
    SloDefinition(
        name="convergence_99",
        kind=SloKind.LATENCY,
        description="Convergence 99% <= 60s",
        target=60.0,
        metric="xcelsior_reconciler_convergence_duration_seconds",
        alert_name="XcelsiorSloConvergenceLatency",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
    SloDefinition(
        name="command_ack_p95",
        kind=SloKind.LATENCY,
        description="Command ACK p95 <= 15s",
        target=15.0,
        metric="xcelsior_worker_command_latency_seconds",
        alert_name="XcelsiorSloCommandAckLatency",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
]

AVAILABILITY_SLOS: list[SloDefinition] = [
    SloDefinition(
        name="mcp_preview_availability",
        kind=SloKind.AVAILABILITY,
        description="MCP preview availability >= 99.95%",
        target=0.9995,
        metric="xcelsior_mcp_funnel_transitions_total",
        alert_name="XcelsiorSloMcpPreviewAvailability",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
    SloDefinition(
        name="approved_launch_success",
        kind=SloKind.AVAILABILITY,
        description="Approved-launch success >= 99.9%",
        target=0.999,
        metric="xcelsior_mcp_launch_outcomes_total",
        alert_name="XcelsiorSloApprovedLaunchSuccess",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
]

COMPLETENESS_SLOS: list[SloDefinition] = [
    SloDefinition(
        name="queue_reason_completeness",
        kind=SloKind.COMPLETENESS,
        description="Queue entries with a current reason 100%",
        target=1.0,
        metric="xcelsior_queue_reason_completeness",
        alert_name="XcelsiorSloQueueReasonCompleteness",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
    SloDefinition(
        name="billing_meter_consistency",
        kind=SloKind.COMPLETENESS,
        description="Billing meter consistency 100%",
        target=1.0,
        metric="xcelsior_billing_meter_mismatches_total",
        alert_name="XcelsiorSloBillingMeterConsistency",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
]

FRESHNESS_SLOS: list[SloDefinition] = [
    SloDefinition(
        name="stale_host_removal",
        kind=SloKind.FRESHNESS,
        description="Stale host removed within freshness + 5s",
        target=5.0, # max staleness delta
        metric="xcelsior_stale_host_observations",
        alert_name="XcelsiorSloStaleHostRemoval",
        burn_rate_windows=[(5, 60), (30, 360)],
    ),
]

ALL_SLOS: list[SloDefinition] = HARD_INVARIANTS + LATENCY_SLOS + AVAILABILITY_SLOS + COMPLETENESS_SLOS + FRESHNESS_SLOS
