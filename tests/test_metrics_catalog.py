"""B7.4 Gate — Metrics catalog and cardinality discipline.

Gate B7.4: *"A metric-registry test failing on a label whose cardinality class is unbounded."*

Verifies:
1. Every collector registered in `prometheus_client.REGISTRY` uses only bounded
   cardinality label names (gpu_class, model, region, reason, outcome, etc.).
   Forbidden labels (`job_id`, `user_id`, `prompt_hash`, `artifact_id`, `raw_error`, etc.)
   must never appear on any metric.
2. The negative canary fails: creating a collector with an unbounded label (`job_id`,
   `user_id`, `raw_error`) is detected by `audit_metric_label_cardinality`.
3. All catalog metrics specified by blueprint §25.3 and DA§9.1 exist and are registered.
4. Hard invariants (`allocation_constraint_violations_total`, `billing_meter_mismatches_total`)
   exist and start at 0.
5. The `/metrics/prometheus` HTTP endpoint exports the declared catalog metrics.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter, REGISTRY

from api import app
import metrics_catalog
from metrics_catalog import audit_metric_label_cardinality

client = TestClient(app)

REQUIRED_CATALOG_METRICS = [
    # Scheduler (§25.3, DA§9.1)
    "xcelsior_scheduler_queue_depth",
    "xcelsior_scheduler_queue_age_seconds",
    "xcelsior_scheduler_claim_latency_seconds",
    "xcelsior_scheduler_claim_expiries_total",
    "xcelsior_scheduler_filter_rejections_total",
    "xcelsior_scheduler_placement_duration_seconds",
    "xcelsior_scheduler_placement_conflict_retries_total",
    "xcelsior_scheduler_allocation_constraint_violations_total",
    "xcelsior_scheduler_preemption_plans_total",
    "xcelsior_scheduler_preemption_outcomes_total",
    "xcelsior_scheduler_replica_heartbeat_timestamp_seconds",
    # Worker (§25.3, DA§9.1)
    "xcelsior_worker_command_latency_seconds",
    "xcelsior_worker_lease_renew_total",
    "xcelsior_worker_container_stage_duration_seconds",
    "xcelsior_worker_runtime_preparation_failures_total",
    "xcelsior_worker_volume_preparation_failures_total",
    "xcelsior_worker_observed_desired_mismatches",
    "xcelsior_worker_identity_expiry_seconds",
    # Reconciler (§25.3, DA§9.1)
    "xcelsior_reconciler_queue_age_seconds",
    "xcelsior_reconciler_convergence_duration_seconds",
    "xcelsior_reconciler_findings_total",
    "xcelsior_reconciler_actions_total",
    "xcelsior_reconciler_stale_observations_total",
    # MCP (§25.3, B5.12)
    "xcelsior_mcp_tool_calls_total",
    "xcelsior_mcp_tool_latency_seconds",
    "xcelsior_mcp_errors_total",
    "xcelsior_mcp_funnel_transitions_total",
    "xcelsior_mcp_idempotent_replays_total",
    "xcelsior_mcp_launch_outcomes_total",
    "xcelsior_mcp_active_transports",
    # Billing (§25.3, DA§9.1)
    "xcelsior_billing_active_meters",
    "xcelsior_billing_running_attempts",
    "xcelsior_billing_meter_mismatches_total",
    "xcelsior_billing_wallet_holds_active",
    "xcelsior_billing_wallet_hold_age_seconds",
    "xcelsior_billing_wallet_hold_expiries_total",
    "xcelsior_billing_ledger_lag_seconds",
    "xcelsior_billing_ledger_failures_total",
    "xcelsior_billing_wallet_hard_stops_total",
    # Data plane (DA§9.1)
    "xcelsior_postgres_pool_usage",
    "xcelsior_postgres_lock_waits_total",
    "xcelsior_postgres_replication_lag_seconds",
    "xcelsior_redis_latency_seconds",
    "xcelsior_redis_rate_limit_denied_total",
    "xcelsior_redis_failovers_total",
    "xcelsior_artifact_session_duration_seconds",
    "xcelsior_artifact_checksum_failures_total",
    "xcelsior_artifact_orphan_backlog",
    "xcelsior_outbox_oldest_age_seconds",
    "xcelsior_outbox_batch_size",
    "xcelsior_outbox_deliveries_total",
    "xcelsior_retrieval_latency_seconds",
    "xcelsior_retrieval_queue_depth",
]


def test_registry_contains_zero_unbounded_cardinality_labels():
    """All registered prometheus collectors must pass the cardinality audit."""
    violations = audit_metric_label_cardinality(REGISTRY._collector_to_names.keys())
    assert not violations, (
        f"Unbounded high-cardinality label violations found in metric registry:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize(
    "bad_label",
    ["job_id", "user_id", "prompt_hash", "artifact_id", "raw_error", "customer_id"],
)
def test_cardinality_audit_fails_on_unbounded_labels(bad_label):
    """Canary test: prove the audit check detects forbidden labels."""
    canary_reg = CollectorRegistry()
    _ = Counter(
        "canary_metric_with_bad_label",
        "Canary metric testing negative audit detection",
        [bad_label],
        registry=canary_reg,
    )
    violations = audit_metric_label_cardinality(canary_reg._collector_to_names.keys())
    assert len(violations) >= 1
    assert any(bad_label in v for v in violations)


@pytest.mark.parametrize("metric_name", REQUIRED_CATALOG_METRICS)
def test_required_catalog_metric_is_registered(metric_name):
    """Every metric required by §25.3 / DA§9.1 must be present in the registry."""
    registered_names = set(REGISTRY._names_to_collectors.keys())
    assert metric_name in registered_names, f"Metric {metric_name} is not registered in REGISTRY"


def test_allocation_and_billing_invariants_are_zero():
    """Hard invariant metrics must be instantiated and begin at zero."""
    sched_samples = [
        s.value
        for m in metrics_catalog.SCHEDULER_ALLOCATION_CONSTRAINT_VIOLATIONS_TOTAL.collect()
        for s in m.samples
        if not s.name.endswith("_created")
    ]
    assert sum(sched_samples) == 0

    billing_samples = [
        s.value
        for m in metrics_catalog.BILLING_METER_MISMATCHES_TOTAL.collect()
        for s in m.samples
        if not s.name.endswith("_created")
    ]
    assert sum(billing_samples) == 0


def test_metrics_prometheus_exposition_includes_catalog():
    """Verify /metrics/prometheus exposes the registered catalog metrics."""
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200
    text = response.text

    assert "xcelsior_scheduler_queue_depth" in text
    assert "xcelsior_scheduler_allocation_constraint_violations_total" in text
    assert "xcelsior_billing_meter_mismatches_total" in text
    assert "xcelsior_worker_command_latency_seconds" in text
    assert "xcelsior_mcp_tool_calls_total" in text
