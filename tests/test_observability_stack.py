"""Structural and signal-contract tests for the private observability plane."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from control_plane import operational_metrics

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "infra" / "observability"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _compose() -> dict:
    return _yaml(ROOT / "docker-compose.yml")


def _prometheus_rules() -> list[dict]:
    documents = (
        _yaml(OBS / "prometheus" / "recording-rules.yml"),
        _yaml(OBS / "prometheus" / "alert-rules.yml"),
    )
    return [
        rule for document in documents for group in document["groups"] for rule in group["rules"]
    ]


OBSERVABILITY_SERVICES = {
    "otel-collector",
    "prometheus",
    "alertmanager",
    "grafana",
    "loki",
    "tempo",
    "postgres-exporter",
    "node-exporter",
    "jaeger",
}


def test_compose_contains_bounded_hardened_observability_services():
    compose = _compose()
    services = compose["services"]
    assert OBSERVABILITY_SERVICES <= services.keys()
    assert compose["networks"]["observability"]["driver"] == "bridge"

    for name in OBSERVABILITY_SERVICES:
        service = services[name]
        assert service["restart"] == "unless-stopped", name
        assert service["healthcheck"]["test"], name
        assert service["read_only"] is True, name
        assert service["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in service["security_opt"], name
        assert service["mem_limit"], name
        assert float(service["cpus"]) > 0, name
        assert int(service["pids_limit"]) > 0, name
        assert "observability" in service["networks"], name


def test_all_observability_host_ports_are_loopback_only():
    services = _compose()["services"]
    published = {
        name: port for name in OBSERVABILITY_SERVICES for port in services[name].get("ports", [])
    }
    assert published
    assert all(str(port).startswith("127.0.0.1:") for port in published.values())
    assert services["otel-collector"]["ports"] == [
        "127.0.0.1:4317:4317",
        "127.0.0.1:4318:4318",
    ]
    assert services["prometheus"]["ports"] == ["127.0.0.1:9091:9090"]
    assert services["grafana"]["ports"] == ["127.0.0.1:3001:3000"]
    assert services["jaeger"]["ports"] == ["127.0.0.1:16686:16686"]


def test_stateful_observability_services_have_named_volumes_and_retention():
    compose = _compose()
    volumes = compose["volumes"]
    required = {
        "otel-collector-data",
        "prometheus-data",
        "alertmanager-data",
        "grafana-data",
        "loki-data",
        "tempo-data",
        "jaeger-data",
    }
    assert required <= volumes.keys()

    services_text = (ROOT / "docker-compose.yml").read_text()
    assert "--storage.tsdb.retention.time=30d" in services_text
    assert "--storage.tsdb.retention.size=20GB" in services_text
    assert "--data.retention=120h" in services_text
    assert "retention_period: 336h" in (OBS / "loki" / "loki.yaml").read_text()
    assert "block_retention: 168h" in (OBS / "tempo" / "tempo.yaml").read_text()


def test_node_exporter_reads_backup_metrics_from_exact_private_textfile_mount():
    service = _compose()["services"]["node-exporter"]
    expected = (
        "/var/lib/node_exporter/textfile_collector:/var/lib/node_exporter/textfile_collector:ro"
    )
    assert expected in service["volumes"]
    assert (
        "--collector.textfile.directory=/var/lib/node_exporter/textfile_collector"
        in service["command"]
    )
    assert not service.get("ports")


def test_prometheus_scrapes_private_application_and_exporter_targets():
    config = _yaml(OBS / "prometheus" / "prometheus.yml")
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["xcelsior-api"]["metrics_path"] == "/metrics/prometheus"
    api_targets = [
        target for static in jobs["xcelsior-api"]["static_configs"] for target in static["targets"]
    ]
    assert api_targets == [
        "host.docker.internal:9500",
        "host.docker.internal:9501",
    ]
    assert jobs["xcelsior-mcp"]["metrics_path"] == "/metrics"
    for required in (
        "otel-collector",
        "postgres",
        "node",
        "alertmanager",
        "loki",
        "tempo",
    ):
        assert required in jobs


def test_public_nginx_edge_blocks_metrics_instead_of_proxying_them():
    config = (ROOT / "nginx" / "xcelsior.conf").read_text()
    assert "location = /metrics" in config
    assert "location ^~ /metrics/" in config
    assert config.count("return 404;") >= 2
    public_api_pattern = next(
        line for line in config.splitlines() if line.strip().startswith("location ~ ^/(openapi")
    )
    assert "metrics" not in public_api_pattern


def test_otel_collector_has_persistent_bounded_queues_and_all_signal_pipelines():
    config = _yaml(OBS / "otel-collector" / "config.yaml")
    assert {"health_check", "file_storage"} <= set(config["extensions"])
    assert {"traces", "metrics", "logs"} == set(config["service"]["pipelines"])
    assert "filelog/docker" in config["service"]["pipelines"]["logs"]["receivers"]
    assert config["exporters"]["otlphttp/loki"]["endpoint"] == "http://loki:3100/otlp"
    for exporter in ("otlp/tempo", "otlp/jaeger", "otlphttp/loki"):
        queue = config["exporters"][exporter]["sending_queue"]
        assert queue["enabled"] is True
        assert queue["storage"] == "file_storage"
        assert queue["queue_size"] > 0
    limiter = config["processors"]["memory_limiter"]
    assert limiter["limit_mib"] <= 384


def test_loki_and_tempo_configs_are_single_node_persistent_and_bounded():
    loki = _yaml(OBS / "loki" / "loki.yaml")
    assert loki["auth_enabled"] is False
    assert loki["limits_config"]["allow_structured_metadata"] is True
    assert loki["limits_config"]["retention_period"] == "336h"
    assert loki["compactor"]["retention_enabled"] is True
    assert loki["analytics"]["reporting_enabled"] is False

    tempo = _yaml(OBS / "tempo" / "tempo.yaml")
    assert tempo["storage"]["trace"]["backend"] == "local"
    assert tempo["compactor"]["compaction"]["block_retention"] == "168h"
    assert (
        tempo["metrics_generator"]["storage"]["remote_write"][0]["url"]
        == "http://prometheus:9090/api/v1/write"
    )
    assert tempo["usage_report"]["reporting_enabled"] is False


def test_alert_rules_cover_required_failure_modes_and_are_actionable():
    alerts = {rule["alert"]: rule for rule in _prometheus_rules() if "alert" in rule}
    required = {
        "XcelsiorApiUnavailable",
        "XcelsiorWorkerUnavailable",
        "XcelsiorQueueBacklogHigh",
        "XcelsiorQueueOldestJobStale",
        "XcelsiorBillingMetersMissing",
        "XcelsiorBillingMetersOpenAfterTerminal",
        "XcelsiorStalePlacementLease",
        "XcelsiorStaleFenceFinding",
        "XcelsiorHostObservationsStale",
        "XcelsiorPostgresExporterUnavailable",
        "XcelsiorPostgresConnectionsHigh",
        "XcelsiorBackupMetricMissing",
        "XcelsiorBackupStale",
        "XcelsiorBackupFailedSinceLastSuccess",
        "XcelsiorOutboxBacklog",
        "XcelsiorOutboxDeadLetters",
        "XcelsiorProjectionBacklog",
        "XcelsiorProjectionDeadLetters",
        "XcelsiorControlPlaneMetricsMissing",
        "XcelsiorControlPlaneMetricsStale",
    }
    assert required <= alerts.keys()
    for name, rule in alerts.items():
        assert rule.get("for"), name
        assert rule["labels"]["severity"] in {"warning", "critical"}, name
        assert rule["labels"]["owner"], name
        assert rule["annotations"]["summary"], name
        assert rule["annotations"]["description"], name
        assert rule["annotations"]["runbook_url"].startswith("https://"), name


def test_backup_and_restore_alerts_use_exact_textfile_metric_contract():
    rules = (OBS / "prometheus" / "alert-rules.yml").read_text()
    for metric in (
        "xcelsior_backup_last_success_timestamp_seconds",
        "xcelsior_backup_last_failure_timestamp_seconds",
        "xcelsior_restore_last_success_timestamp_seconds",
        "xcelsior_restore_last_failure_timestamp_seconds",
    ):
        assert metric in rules
    assert (
        "xcelsior_backup_last_failure_timestamp_seconds > "
        "xcelsior_backup_last_success_timestamp_seconds"
    ) in rules


def test_recording_rules_reduce_replica_duplicates_and_expose_freshness():
    records = {
        rule["record"]: str(rule["expr"]) for rule in _prometheus_rules() if "record" in rule
    }
    required = {
        "xcelsior:api_available",
        "xcelsior:queue_depth",
        "xcelsior:queue_oldest_age_seconds",
        "xcelsior:billing_meter_invariant_violations",
        "xcelsior:outbox_backlog",
        "xcelsior:projection_pending_deliveries",
        "xcelsior:projection_dead_letters",
        "xcelsior:postgres_connections_utilization",
        "xcelsior:control_plane_metrics_age_seconds",
    }
    assert required <= records.keys()
    assert "max(" in records["xcelsior:queue_depth"]
    assert "time()" in records["xcelsior:control_plane_metrics_age_seconds"]


def test_runtime_loops_publish_durable_worker_freshness_signals():
    scheduler = (ROOT / "scheduler.py").read_text()
    reconciler = (ROOT / "control_plane" / "scheduler" / "service.py").read_text()
    outbox = (ROOT / "control_plane" / "outbox_runtime.py").read_text()
    maintenance = (ROOT / "bg_worker.py").read_text()

    assert 'ServiceHeartbeat("scheduler")' in scheduler
    assert '"reconciler",' in reconciler
    assert 'ServiceHeartbeat(\n        "outbox"' in outbox
    assert 'register_task("maintenance_heartbeat", lambda: None, 15)' in maintenance


def test_default_alertmanager_config_commits_no_receiver_secret():
    default = _yaml(OBS / "alertmanager" / "alertmanager.yml")
    assert default["route"]["receiver"] == "xcelsior-default"
    assert default["receivers"]
    text = (OBS / "alertmanager" / "alertmanager.yml").read_text().lower()
    assert "webhook_configs" not in text
    assert "api_url" not in text
    assert "password:" not in text

    example = (OBS / "alertmanager" / "alertmanager.routing.example.yml").read_text()
    assert "webhook_configs" in example
    assert "example.invalid" in example
    compose_mount = _compose()["services"]["alertmanager"]["volumes"][0]
    assert "XCELSIOR_ALERTMANAGER_CONFIG_PATH" in compose_mount


def test_grafana_provisions_correlated_datasources_and_operations_dashboard():
    sources = _yaml(OBS / "grafana" / "provisioning" / "datasources" / "datasources.yml")[
        "datasources"
    ]
    by_uid = {source["uid"]: source for source in sources}
    assert set(by_uid) == {"prometheus", "loki", "tempo"}
    assert by_uid["tempo"]["jsonData"]["tracesToLogsV2"]["datasourceUid"] == "loki"
    assert by_uid["tempo"]["jsonData"]["tracesToMetrics"]["datasourceUid"] == "prometheus"

    dashboard = json.loads(
        (OBS / "grafana" / "dashboards" / "xcelsior-operations.json").read_text()
    )
    assert dashboard["uid"] == "xcelsior-operations"
    assert dashboard["refresh"] == "30s"
    expressions = {
        target["expr"] for panel in dashboard["panels"] for target in panel.get("targets", [])
    }
    assert "xcelsior:queue_depth" in expressions
    assert "xcelsior:outbox_backlog" in expressions
    assert "xcelsior:postgres_connections_utilization" in expressions
    assert '{service_namespace="xcelsior"}' in expressions


class _Result:
    def __init__(self, *, one=None, all_rows=None):
        self._one = one
        self._all = all_rows or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all


class _OperationalConnection:
    def execute(self, query, params=()):
        if "WITH\nqueue_state" in query:
            return _Result(
                one=(
                    4,
                    91.5,
                    2,
                    1,
                    1,
                    0,
                    3,
                    1,
                    402.0,
                    7,
                    52.0,
                    2,
                    1,
                    3,
                    8,
                    63.0,
                    1,
                    2,
                    1,
                    5.0,
                )
            )
        if "FROM service_heartbeats" in query:
            return _Result(
                all_rows=[
                    ("scheduler", 1, 4.0),
                    ("maintenance", 2, 3.0),
                ]
            )
        raise AssertionError(query)


def test_operational_snapshot_renders_real_zeroes_and_explicit_freshness(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SCHEDULER_MODE", "active")
    monkeypatch.setenv("XCELSIOR_OUTBOX_DISPATCHER", "true")
    snapshot = operational_metrics.collect_operational_snapshot(
        _OperationalConnection(),
        observation_stale_seconds=300,
        heartbeat_fresh_seconds=60,
    )
    assert snapshot["queue_depth"] == 4
    assert snapshot["billing_missing_meters"] == 2
    assert snapshot["services"]["scheduler"]["fresh_replicas"] == 1
    assert snapshot["services"]["outbox"]["fresh_replicas"] == 0
    assert snapshot["services"]["maintenance"]["fresh_replicas"] == 1
    assert snapshot["services"]["maintenance"]["latest_age_seconds"] == 5.0
    assert snapshot["expected_services"]["reconciler"] == 1

    rendered = "\n".join(operational_metrics.render_operational_metrics(snapshot))
    assert "xcelsior_control_plane_metrics_available 1" in rendered
    assert "xcelsior_queue_oldest_age_seconds 91.5" in rendered
    assert "xcelsior_outbox_dead_letters 1" in rendered
    assert 'xcelsior_service_heartbeat_fresh_replicas{service="outbox"} 0' in rendered
    assert 'xcelsior_service_expected{service="reconciler"} 1' in rendered

    unavailable = "\n".join(operational_metrics.render_operational_metrics_unavailable())
    assert "xcelsior_control_plane_metrics_available 0" in unavailable
    assert "xcelsior_control_plane_metrics_last_success_timestamp_seconds 0.000" not in unavailable


def test_heartbeat_upsert_is_bounded_and_uses_durable_table(monkeypatch):
    captured = {}

    class _Connection:
        def execute(self, query, params):
            captured["query"] = query
            captured["params"] = params

    def _run_transaction(callback, *, what):
        captured["what"] = what
        callback(_Connection())

    monkeypatch.setattr(operational_metrics, "run_transaction", _run_transaction)
    operational_metrics.heartbeat_once(
        "scheduler",
        replica_id="scheduler-green",
        details={"mode": "active"},
    )
    assert captured["what"] == "scheduler_heartbeat"
    assert "INSERT INTO service_heartbeats" in captured["query"]
    assert "ON CONFLICT (service, replica_id) DO UPDATE" in captured["query"]
    assert captured["params"][0:2] == ("scheduler", "scheduler-green")

    with pytest.raises(ValueError):
        operational_metrics.heartbeat_once("unbounded-user-label")
