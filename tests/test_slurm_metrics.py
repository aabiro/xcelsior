"""Slurm reconciler metrics (control_plane/slurm_metrics.py) and their alerts.

The metrics are read from what the reconciler recorded, so these tests write
that state directly and check the numbers. The alert rules are checked two
ways: every Slurm metric a rule names must be one the exporter emits (a
renamed metric would otherwise leave an alert that can never fire), and,
where Docker has the Prometheus image, promtool runs the outage fixture.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from control_plane.slurm_metrics import collect, render
from db import pg_connection, pg_transaction

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "api/infra/observability/prometheus/native-control-plane-rules.yml"
PROMETHEUS_IMAGE = "prom/prometheus:v3.5.0"


@pytest.fixture
def cluster():
    cid = f"sm-{uuid.uuid4().hex[:8]}"
    with pg_transaction() as conn:
        conn.execute(
            "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, "
            "poll_interval_seconds, controller_ok_at, failure_counts, facts) VALUES "
            "(%s, 'generic', 'x', 30, clock_timestamp() - interval '200 seconds', "
            '\'{"poll:transport_error": 4, "accounting:sacct_failed": 1}\', '
            '\'{"orphans": ["7", "8"]}\')',
            (cid,),
        )
        for n, extra in (
            (1, "last_observed_at = clock_timestamp()"),
            (2, "last_observed_at = clock_timestamp() - interval '10 minutes'"),
            (
                3,
                "last_observed_at = clock_timestamp(), "
                "left_controller_at = clock_timestamp() - interval '20 minutes'",
            ),
            (
                4,
                "terminal_at = clock_timestamp(), "
                "left_controller_at = clock_timestamp() - interval '2 hours'",
            ),  # settled: not counted
        ):
            conn.execute(
                "INSERT INTO slurm_job_mappings (xcelsior_job_id, slurm_job_id, created_at, "
                "cluster_id, observed_state, submitted_at) VALUES (%s, %s, 0, %s, 'running', "
                "clock_timestamp())",
                (f"{cid}-{n}", str(n), cid),
            )
            conn.execute(
                f"UPDATE slurm_job_mappings SET {extra} WHERE xcelsior_job_id = %s",
                (f"{cid}-{n}",),
            )
    yield cid
    with pg_transaction() as conn:
        conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
        conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cid,))


def _mine(cid):
    with pg_connection() as conn:
        [c] = [c for c in collect(conn) if c["cluster"] == cid]
    return c


def test_the_numbers_come_from_recorded_state(cluster):
    c = _mine(cluster)
    assert c["poll_interval_seconds"] == 30
    assert 200 <= c["reconcile_lag_seconds"] < 260
    assert c["open_jobs"] == 3 and c["stale_jobs"] == 1
    assert c["awaiting_accounting_jobs"] == 1
    assert 1200 <= c["awaiting_accounting_oldest_seconds"] < 1260
    assert c["orphan_jobs"] == 2
    assert c["failures"] == {"poll:transport_error": 4, "accounting:sacct_failed": 1}


def test_rendered_with_bounded_labels(cluster):
    text = "\n".join(render([_mine(cluster)]))
    assert "xcelsior_slurm_metrics_available 1" in text
    assert f'xcelsior_slurm_open_jobs{{cluster="{cluster}"}} 3' in text
    assert (
        f'xcelsior_slurm_failures_total{{cluster="{cluster}",stage="poll",'
        f'error_code="transport_error"}} 4'
    ) in text
    assert "# TYPE xcelsior_slurm_failures_total counter" in text


def test_a_disabled_cluster_is_not_exported(cluster):
    with pg_transaction() as conn:
        conn.execute("UPDATE slurm_clusters SET enabled = FALSE WHERE cluster_id = %s", (cluster,))
    with pg_connection() as conn:
        assert cluster not in {c["cluster"] for c in collect(conn)}


def test_unreadable_state_is_unavailable_not_zero():
    assert render(None) == [
        "# HELP xcelsior_slurm_metrics_available 1 when the Slurm reconciler's state was read",
        "# TYPE xcelsior_slurm_metrics_available gauge",
        "xcelsior_slurm_metrics_available 0",
    ]


def test_every_metric_an_alert_reads_is_exported(cluster):
    exported = set(
        re.findall(r"^# TYPE (xcelsior_slurm_\w+)", "\n".join(render([_mine(cluster)])), re.M)
    )
    used = set(re.findall(r"xcelsior_slurm_\w+", RULES.read_text()))
    assert used and used <= exported, f"rules read metrics nothing exports: {used - exported}"


def test_the_api_exports_them():
    from fastapi.testclient import TestClient

    from api import app

    body = TestClient(app).get("/metrics/prometheus").text
    assert "xcelsior_slurm_metrics_available 1" in body


def _have_promtool() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", PROMETHEUS_IMAGE], capture_output=True, timeout=20
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _have_promtool(), reason=f"needs docker with {PROMETHEUS_IMAGE}")
def test_the_alerts_fire_in_the_outage_fixture():
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{ROOT}:/w:ro",
            "-w",
            "/w/tests/fixtures",
            "--entrypoint",
            "promtool",
            PROMETHEUS_IMAGE,
            "test",
            "rules",
            "native_slurm_alerts.yml",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
