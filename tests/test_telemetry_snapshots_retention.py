"""Gate tests for B7.6 — telemetry_snapshots retention and size alerts (DA§9.4)."""

import inspect
from pathlib import Path
from unittest.mock import MagicMock
import yaml

import bg_worker
from control_plane.telemetry_retention import (
    prune_telemetry_snapshots,
    telemetry_retention_task,
    DEFAULT_RETENTION_DAYS,
)

OBS_ROOT = Path(__file__).resolve().parent.parent / "infra" / "observability"


def test_prune_telemetry_snapshots_deletes_old_rows_in_batches():
    """Verify prune_telemetry_snapshots deletes rows older than retention_days."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.side_effect = [
        [(1,), (2,)],
        [],
    ]
    mock_conn.execute.return_value = mock_cur

    deleted = prune_telemetry_snapshots(mock_conn, retention_days=14, batch_limit=2)
    assert deleted == 2
    assert mock_conn.execute.call_count == 2
    args, kwargs = mock_conn.execute.call_args_list[0]
    assert "DELETE FROM telemetry_snapshots" in args[0]
    assert args[1] == (14, 2)


def test_telemetry_retention_task_wires_to_db(monkeypatch):
    """Verify telemetry_retention_task executes and returns result."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = []
    mock_conn.execute.return_value = mock_cur

    class MockPoolContext:
        def __enter__(self):
            return mock_conn

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("control_plane.telemetry_retention.pg_connection", lambda: MockPoolContext())

    result = telemetry_retention_task()
    assert result["ok"] is True
    assert result["pruned_rows"] == 0
    assert result["retention_days"] == DEFAULT_RETENTION_DAYS
    mock_conn.commit.assert_called_once()


def test_telemetry_retention_scheduled_task_registered():
    """Verify telemetry_retention is registered as a durable task in bg_worker."""
    source = inspect.getsource(bg_worker.main)
    assert 'register_task("telemetry_retention"' in source


def test_telemetry_snapshots_size_alert_exists_and_is_actionable():
    """Verify Prometheus alert rule exists for telemetry_snapshots table size."""
    rules_path = OBS_ROOT / "prometheus" / "alert-rules.yml"
    assert rules_path.exists()

    with rules_path.open() as f:
        data = yaml.safe_load(f)

    alerts = {
        rule["alert"]: rule
        for group in data.get("groups", [])
        for rule in group.get("rules", [])
        if "alert" in rule
    }

    assert "XcelsiorTelemetrySnapshotsSizeHigh" in alerts
    rule = alerts["XcelsiorTelemetrySnapshotsSizeHigh"]
    assert "telemetry_snapshots" in rule["expr"]
    assert rule["labels"]["severity"] in ("warning", "critical")
    assert rule["labels"]["owner"] == "platform"
    assert rule["annotations"]["summary"]
    assert rule["annotations"]["description"]
    assert rule["annotations"]["runbook_url"].startswith("https://")
