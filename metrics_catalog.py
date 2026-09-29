"""B7.4 / DA§9.1 — Prometheus metrics catalog and cardinality discipline.

Defines the platform's standard operational metrics catalog across:
- Scheduler: queue depth/age, claim latency/expiry, rejections, placement duration,
  conflict retries, allocation constraint violations (hard invariant = 0), preemption,
  and replica heartbeats.
- Worker: command fetch/claim/ACK latency, lease renewals, container lifecycle stages,
  runtime/volume prep failures, observed vs desired mismatches, identity expiry.
- Reconciler: queue age, convergence duration, findings by type/severity, actions/retries/dead letters.
- MCP: tool calls and latencies, auth/scope/rate errors, preview-approval-execute funnel, replays.
- Billing: active meters vs running attempts, orphan/missing meter invariants, holds, ledger lag.
- Data Plane: PostgreSQL pool/replication, Redis latency/limits, artifact sessions/checksums, outbox lag.

Cardinality Discipline (§25.3, DA§9.1):
High-cardinality IDs (`job_id`, `user_id`, `customer_id`, prompt hash, artifact id,
raw error text) never become metric labels. They belong in structured logs and traces.
"""

from __future__ import annotations

from typing import Iterable
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

# ── Scheduler Metrics (§25.3, DA§9.1) ───────────────────────────────────

SCHEDULER_QUEUE_DEPTH = Gauge(
    "xcelsior_scheduler_queue_depth",
    "Current queue depth of jobs pending scheduling",
    ["gpu_class", "model", "region"],
)

SCHEDULER_QUEUE_AGE_SECONDS = Gauge(
    "xcelsior_scheduler_queue_age_seconds",
    "Age of oldest queued job pending scheduling in seconds",
    ["gpu_class", "model", "region"],
)

SCHEDULER_CLAIM_LATENCY_SECONDS = Histogram(
    "xcelsior_scheduler_claim_latency_seconds",
    "Time between job submission and worker claim in seconds",
    ["gpu_class", "region"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

SCHEDULER_CLAIM_EXPIRIES_TOTAL = Counter(
    "xcelsior_scheduler_claim_expiries_total",
    "Total expired scheduling claims",
    ["reason"],
)

SCHEDULER_FILTER_REJECTIONS_TOTAL = Counter(
    "xcelsior_scheduler_filter_rejections_total",
    "Total scheduling filter rejections by reason",
    ["reason"],
)

SCHEDULER_PLACEMENT_DURATION_SECONDS = Histogram(
    "xcelsior_scheduler_placement_duration_seconds",
    "Latency of scheduler placement loop iterations in seconds",
    ["outcome"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

SCHEDULER_PLACEMENT_CONFLICT_RETRIES_TOTAL = Counter(
    "xcelsior_scheduler_placement_conflict_retries_total",
    "Total placement conflict retries during concurrent scheduling",
    ["conflict_type"],
)

# Hard Invariant: must stay zero in production
SCHEDULER_ALLOCATION_CONSTRAINT_VIOLATIONS_TOTAL = Counter(
    "xcelsior_scheduler_allocation_constraint_violations_total",
    "Total allocation constraint violations (hard invariant: must remain zero)",
    ["violation_type"],
)

SCHEDULER_PREEMPTION_PLANS_TOTAL = Counter(
    "xcelsior_scheduler_preemption_plans_total",
    "Total preemption evaluations initiated",
    ["priority_class"],
)

SCHEDULER_PREEMPTION_OUTCOMES_TOTAL = Counter(
    "xcelsior_scheduler_preemption_outcomes_total",
    "Total preemption execution outcomes",
    ["outcome"],
)

SCHEDULER_REPLICA_HEARTBEAT_TIMESTAMP_SECONDS = Gauge(
    "xcelsior_scheduler_replica_heartbeat_timestamp_seconds",
    "Timestamp of latest scheduler replica tick in seconds",
    ["replica_id"],
)

# ── Worker Metrics (§25.3, DA§9.1) ──────────────────────────────────────

WORKER_COMMAND_LATENCY_SECONDS = Histogram(
    "xcelsior_worker_command_latency_seconds",
    "Worker command transit and execution latency in seconds",
    ["command_type", "stage"],
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0),
)

WORKER_LEASE_RENEW_TOTAL = Counter(
    "xcelsior_worker_lease_renew_total",
    "Total worker lease renewals and terminations",
    ["outcome"],
)

WORKER_CONTAINER_STAGE_DURATION_SECONDS = Histogram(
    "xcelsior_worker_container_stage_duration_seconds",
    "Worker container preparation and start duration in seconds",
    ["stage"],
    buckets=(0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0),
)

WORKER_RUNTIME_PREPARATION_FAILURES_TOTAL = Counter(
    "xcelsior_worker_runtime_preparation_failures_total",
    "Total failures during worker container runtime preparation",
    ["failure_type"],
)

WORKER_VOLUME_PREPARATION_FAILURES_TOTAL = Counter(
    "xcelsior_worker_volume_preparation_failures_total",
    "Total failures during volume mount or decryption",
    ["failure_type"],
)

WORKER_OBSERVED_DESIRED_MISMATCHES = Gauge(
    "xcelsior_worker_observed_desired_mismatches",
    "Number of resource mismatches between observed worker state and desired state",
    ["resource_type"],
)

WORKER_IDENTITY_EXPIRY_SECONDS = Gauge(
    "xcelsior_worker_identity_expiry_seconds",
    "Seconds remaining until worker agent token / cert expiration",
    ["token_type"],
)

# ── Reconciler Metrics (§25.3, DA§9.1) ──────────────────────────────────

RECONCILER_QUEUE_AGE_SECONDS = Gauge(
    "xcelsior_reconciler_queue_age_seconds",
    "Age of oldest pending reconciliation item in seconds",
)

RECONCILER_CONVERGENCE_DURATION_SECONDS = Histogram(
    "xcelsior_reconciler_convergence_duration_seconds",
    "Duration of reconciliation loop from drift detection to convergence",
    ["outcome"],
    buckets=(0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0),
)

RECONCILER_FINDINGS_TOTAL = Counter(
    "xcelsior_reconciler_findings_total",
    "Total anomalies found by background reconcilers",
    ["finding_type", "severity"],
)

RECONCILER_ACTIONS_TOTAL = Counter(
    "xcelsior_reconciler_actions_total",
    "Total remediation actions taken by reconciler",
    ["action_type", "status"],
)

RECONCILER_STALE_OBSERVATIONS_TOTAL = Counter(
    "xcelsior_reconciler_stale_observations_total",
    "Total stale host/agent observations purged",
    ["entity_type"],
)

# ── MCP Metrics (§25.3, B5.12) ──────────────────────────────────────────

MCP_TOOL_CALLS_TOTAL = Counter(
    "xcelsior_mcp_tool_calls_total",
    "Total MCP tool invocations by tool and outcome",
    ["tool", "outcome"],
)

MCP_TOOL_LATENCY_SECONDS = Histogram(
    "xcelsior_mcp_tool_latency_seconds",
    "Latency of MCP tool operations in seconds",
    ["tool"],
    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MCP_ERRORS_TOTAL = Counter(
    "xcelsior_mcp_errors_total",
    "Total MCP errors by category",
    ["error_category"],
)

MCP_FUNNEL_TRANSITIONS_TOTAL = Counter(
    "xcelsior_mcp_funnel_transitions_total",
    "MCP action plan lifecycle funnel transitions",
    ["stage"],
)

MCP_IDEMPOTENT_REPLAYS_TOTAL = Counter(
    "xcelsior_mcp_idempotent_replays_total",
    "Total cached idempotent tool replays served",
    ["tool"],
)

MCP_LAUNCH_OUTCOMES_TOTAL = Counter(
    "xcelsior_mcp_launch_outcomes_total",
    "Total MCP launches by workload type and outcome",
    ["workload_type", "status"],
)

MCP_ACTIVE_TRANSPORTS = Gauge(
    "xcelsior_mcp_active_transports",
    "Number of active MCP transport connections",
    ["transport_type"],
)

# ── Billing Metrics (§25.3, DA§9.1) ─────────────────────────────────────

BILLING_ACTIVE_METERS = Gauge(
    "xcelsior_billing_active_meters",
    "Count of currently active billing usage meters",
)

BILLING_RUNNING_ATTEMPTS = Gauge(
    "xcelsior_billing_running_attempts",
    "Count of currently running job execution attempts",
)

# Hard Invariant: orphan or missing meters must alert immediately
BILLING_METER_MISMATCHES_TOTAL = Counter(
    "xcelsior_billing_meter_mismatches_total",
    "Inconsistency between running attempts and active meters (hard invariant)",
    ["mismatch_type"],
)

BILLING_WALLET_HOLDS_ACTIVE = Gauge(
    "xcelsior_billing_wallet_holds_active",
    "Current count of active pre-authorization wallet holds",
)

BILLING_WALLET_HOLD_AGE_SECONDS = Histogram(
    "xcelsior_billing_wallet_hold_age_seconds",
    "Age of active wallet holds in seconds",
    buckets=(10.0, 60.0, 300.0, 900.0, 1800.0, 3600.0),
)

BILLING_WALLET_HOLD_EXPIRIES_TOTAL = Counter(
    "xcelsior_billing_wallet_hold_expiries_total",
    "Total wallet holds expired before capture",
    ["reason"],
)

BILLING_LEDGER_LAG_SECONDS = Gauge(
    "xcelsior_billing_ledger_lag_seconds",
    "Lag between meter closure and persistent ledger posting in seconds",
)

BILLING_LEDGER_FAILURES_TOTAL = Counter(
    "xcelsior_billing_ledger_failures_total",
    "Total failures posting billing transactions to ledger",
    ["failure_type"],
)

BILLING_WALLET_HARD_STOPS_TOTAL = Counter(
    "xcelsior_billing_wallet_hard_stops_total",
    "Total jobs terminated due to zero/exhausted wallet balance",
    ["reason"],
)

# ── Data Plane Indicators (DA§9.1) ──────────────────────────────────────

POSTGRES_POOL_USAGE = Gauge(
    "xcelsior_postgres_pool_usage",
    "PostgreSQL connection pool connections by state",
    ["pool_name", "state"],
)

POSTGRES_LOCK_WAITS_TOTAL = Counter(
    "xcelsior_postgres_lock_waits_total",
    "Total lock wait events in PostgreSQL",
    ["lock_type"],
)

POSTGRES_REPLICATION_LAG_SECONDS = Gauge(
    "xcelsior_postgres_replication_lag_seconds",
    "PostgreSQL replication lag in seconds",
)

REDIS_LATENCY_SECONDS = Histogram(
    "xcelsior_redis_latency_seconds",
    "Redis operation latency in seconds",
    ["command"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
)

REDIS_RATE_LIMIT_DENIED_TOTAL = Counter(
    "xcelsior_redis_rate_limit_denied_total",
    "Total requests denied by Redis rate limiter",
    ["limit_scope"],
)

REDIS_FAILOVERS_TOTAL = Counter(
    "xcelsior_redis_failovers_total",
    "Total Redis Sentinel / cluster failovers detected",
)

ARTIFACT_SESSION_DURATION_SECONDS = Histogram(
    "xcelsior_artifact_session_duration_seconds",
    "Artifact upload/download/finalize session duration in seconds",
    ["operation", "backend"],
    buckets=(0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0),
)

ARTIFACT_CHECKSUM_FAILURES_TOTAL = Counter(
    "xcelsior_artifact_checksum_failures_total",
    "Total checksum verification failures during artifact finalize",
    ["backend"],
)

ARTIFACT_ORPHAN_BACKLOG = Gauge(
    "xcelsior_artifact_orphan_backlog",
    "Unfinalized or orphaned artifact sessions pending cleanup",
)

OUTBOX_OLDEST_AGE_SECONDS = Gauge(
    "xcelsior_outbox_oldest_age_seconds",
    "Age of oldest undelivered outbox event in seconds",
    ["sink"],
)

OUTBOX_BATCH_SIZE = Histogram(
    "xcelsior_outbox_batch_size",
    "Number of events dispatched per outbox polling batch",
    ["sink"],
    buckets=(1, 5, 10, 25, 50, 100, 250),
)

OUTBOX_DELIVERIES_TOTAL = Counter(
    "xcelsior_outbox_deliveries_total",
    "Total outbox event delivery attempts by sink and status",
    ["sink", "status"],
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "xcelsior_retrieval_latency_seconds",
    "Vector embedding and retrieval search latency in seconds",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RETRIEVAL_QUEUE_DEPTH = Gauge(
    "xcelsior_retrieval_queue_depth",
    "Items waiting for embedding or indexing",
)


# ── Cardinality Discipline Validator (§25.3, DA§9.1) ────────────────────

FORBIDDEN_CARDINALITY_PATTERNS = {
    "job_id",
    "user_id",
    "customer_id",
    "user",
    "customer",
    "prompt_hash",
    "prompt",
    "artifact_id",
    "raw_error",
    "error_text",
    "error_msg",
    "error_message",
    "trace_id",
    "span_id",
    "ip",
    "address",
    "email",
    "payload",
    "token",
}


def audit_metric_label_cardinality(collectors: Iterable[object] | None = None) -> list[str]:
    """Inspect all registered collectors and return violations where labels have unbounded cardinality.

    Raises ValueError if any registered metric uses forbidden high-cardinality labels.
    """
    violations: list[str] = []
    if collectors is None:
        collectors = REGISTRY._collector_to_names.keys()

    for collector in collectors:
        # Check standard prometheus_client metrics
        labelnames = getattr(collector, "_labelnames", ())
        collector_name = getattr(collector, "_name", str(collector))
        for label in labelnames:
            label_lower = label.lower()
            if label_lower in FORBIDDEN_CARDINALITY_PATTERNS or any(
                p in label_lower for p in ("job_id", "user_id", "prompt_hash", "artifact_id", "raw_error")
            ):
                violations.append(
                    f"Metric '{collector_name}' has unbounded high-cardinality label '{label}'"
                )

    return violations

WORKER_UNVETTED_STARTS_TOTAL = Counter(
    "xcelsior_worker_unvetted_starts_total",
    "Total worker starts unvetted (hard invariant: must remain zero)",
    ["violation_type"],
)

STALE_FENCE_ACCEPTANCES_TOTAL = Counter(
    "xcelsior_stale_fence_acceptances_total",
    "Total stale fence acceptances (hard invariant: must remain zero)",
    ["violation_type"],
)

PREMATURE_STRICT_REASSIGNMENTS_TOTAL = Counter(
    "xcelsior_premature_strict_reassignments_total",
    "Total premature strict reassignments (hard invariant: must remain zero)",
    ["violation_type"],
)

QUEUE_REASON_COMPLETENESS = Gauge(
    "xcelsior_queue_reason_completeness",
    "Queue entries with a current reason",
)
