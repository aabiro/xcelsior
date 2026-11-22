"""A renter sees their own GPUs' telemetry, live and as history, and nothing else.

The worker's report carries host-wide fields (other tenants' job count, every
mounted volume id) and, from agent 2.3.0, one entry per GPU. The renter route
returns only the entries for the GPUs the run was given, only the fields in
``gpu_telemetry.RENTER_GPU_FIELDS``. Before this, the route read an in-process
dict that is never filled on PostgreSQL, so it returned 404 on every real
deployment.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from gpu_telemetry import RENTER_GPU_FIELDS, validate_gpu_samples  # noqa: E402
from tests.test_run_receipts import _client, _register  # noqa: E402


def _gpu(index: int, uuid: str = "", **over) -> dict:
    entry = {
        "index": index,
        "gpu_uuid": uuid,
        "utilization": 50.0,
        "memory_util": 40.0,
        "memory_used_gb": 8.0,
        "memory_total_gb": 24.0,
        "temp": 61.0,
        "power_draw_w": 210.0,
        "power_limit_w": 450.0,
    }
    entry.update(over)
    return entry


@pytest.mark.parametrize(
    "gpus",
    [
        "not a list",
        [{"gpu_uuid": "x"}],
        [_gpu(0), _gpu(0)],
        [_gpu(0, utilization=101.0)],
        [_gpu(0, memory_used_gb=30.0)],
        [_gpu(0, temp=float("nan"))],
        [_gpu(i) for i in range(65)],
    ],
)
def test_a_malformed_gpu_list_is_refused(gpus):
    with pytest.raises(ValueError):
        validate_gpu_samples(gpus)


def test_the_worker_sends_exactly_the_fields_a_renter_may_see():
    """The worker's entry and the server's allowlist cannot drift apart."""
    import worker_agent

    sample = worker_agent._gpu_sample(
        {
            "index": 1,
            "uuid": "GPU-smi",
            "utilization": 12,
            "temperature_c": 55,
            "memory_used_gb": 1.5,
            "memory_total_gb": 12.0,
            "serial": "secret-serial",
        },
        0,
    )
    assert set(sample) == {"index", "gpu_uuid", *RENTER_GPU_FIELDS}
    assert sample["index"] == 1 and sample["gpu_uuid"] == "GPU-smi" and sample["temp"] == 55.0
    validate_gpu_samples([sample])


def test_the_frontend_type_carries_every_field_the_route_returns():
    """The agent/human parity guard: a field the route sends but the type drops never renders."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "api/frontend/src/lib/api.ts").read_text()
    body = re.search(r"export interface InstanceGpuSample \{(.*?)\n\}", source, re.S)
    assert body, "InstanceGpuSample is missing from api.ts"
    fields = set(re.findall(r"^\s+(\w+)\??:", body.group(1), re.M))
    assert fields == {"index", "gpu_model", *RENTER_GPU_FIELDS, "received_at", "stale"}


@pytest.fixture
def shared_host_run():
    """A running instance holding GPU 0 of a host that also has a GPU 1."""
    from db import _get_pg_pool
    from tests.fixtures.native_runtime import running_instance

    client = _client()
    headers, customer_id = _register(client, "telemetry")
    with running_instance(owner=customer_id) as native:
        host_id = native["host_id"]
        with _get_pg_pool().connection() as conn:
            conn.execute(
                "UPDATE host_gpu_devices SET device_index = 0 WHERE host_id = %s", (host_id,)
            )
            conn.execute(
                """INSERT INTO host_gpu_devices(host_id, gpu_uuid, device_index, model,
                   total_vram_mb, allocatable_vram_mb, health)
                   VALUES (%s, %s, 1, 'RTX 4090', 24576, 24576, 'healthy')""",
                (host_id, f"GPU-{host_id}-other"),
            )
        try:
            yield client, headers, native
        finally:
            with _get_pg_pool().connection() as conn:
                conn.execute("DELETE FROM telemetry_latest WHERE host_id = %s", (host_id,))
                conn.execute("DELETE FROM telemetry_samples WHERE host_id = %s", (host_id,))


def _report(client, host_id: str, gpus: list[dict]):
    metrics = {
        "utilization": 50.0,
        "gpu_count": len(gpus),
        "active_jobs": 7,
        "volume_health": [{"volume_id": "vol-other-tenant", "mounted": True}],
        "gpus": gpus,
    }
    with patch("routes.agent._require_agent_auth", return_value=None):
        return client.post(
            "/agent/telemetry",
            json={"host_id": host_id, "timestamp": time.time(), "metrics": metrics},
        )


@pytest.mark.needs_db
def test_a_renter_sees_only_their_gpu_and_no_host_wide_fields(shared_host_run):
    client, headers, native = shared_host_run
    host_id, job_id = native["host_id"], native["job_id"]

    before = client.get(f"/api/instances/{job_id}/telemetry", headers=headers)
    assert before.status_code == 200, before.text
    assert before.json()["telemetry"] is None
    assert before.json()["reason"] == "no_readings"

    # GPU 0 arrives without a UUID (the nvidia-smi fallback): matched by index.
    r = _report(
        client,
        host_id,
        [_gpu(0, utilization=33.3), _gpu(1, f"GPU-{host_id}-other", utilization=77.7)],
    )
    assert r.status_code == 200, r.text

    body = client.get(f"/api/instances/{job_id}/telemetry", headers=headers).json()
    gpus = body["telemetry"]["gpus"]
    assert [g["index"] for g in gpus] == [0]
    assert gpus[0]["utilization"] == 33.3 and gpus[0]["stale"] is False
    assert set(gpus[0]) == {"index", "gpu_model", *RENTER_GPU_FIELDS, "received_at", "stale"}
    text = json.dumps(body)
    for leaked in ("77.7", "vol-other-tenant", "active_jobs", "volume_health", "-other"):
        assert leaked not in text

    history = client.get(f"/api/instances/{job_id}/telemetry/history", headers=headers).json()
    series = history["history"]["gpus"]
    assert [s["index"] for s in series] == [0]
    assert [p["utilization"] for p in series[0]["points"]] == [33.3]
    assert "77.7" not in json.dumps(history)


@pytest.mark.needs_db
def test_history_keeps_one_point_per_gpu_every_30_seconds(shared_host_run):
    client, headers, native = shared_host_run
    host_id, job_id = native["host_id"], native["job_id"]
    for value in (10.0, 20.0, 30.0):
        assert _report(client, host_id, [_gpu(0, utilization=value)]).status_code == 200
    latest = client.get(f"/api/instances/{job_id}/telemetry", headers=headers).json()
    assert latest["telemetry"]["gpus"][0]["utilization"] == 30.0
    points = client.get(f"/api/instances/{job_id}/telemetry/history", headers=headers).json()[
        "history"
    ]["gpus"][0]["points"]
    assert [p["utilization"] for p in points] == [10.0]


@pytest.mark.needs_db
def test_another_account_gets_not_found(shared_host_run):
    client, _, native = shared_host_run
    other, _ = _register(client, "telemetry-other")
    for path in ("telemetry", "telemetry/history"):
        r = client.get(f"/api/instances/{native['job_id']}/{path}", headers=other)
        assert r.status_code == 404, r.text


@pytest.mark.needs_db
def test_a_bad_gpu_list_is_dropped_and_the_report_still_counts(shared_host_run):
    """The list feeds only charts; it never costs the host its liveness report."""
    from db import _get_pg_pool

    client, headers, native = shared_host_run
    host_id = native["host_id"]
    r = _report(client, host_id, [_gpu(0, utilization=140.0)])
    assert r.status_code == 200, r.text
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            "SELECT gpu_uuid FROM telemetry_latest WHERE host_id = %s", (host_id,)
        ).fetchall()
    assert [row[0] for row in rows] == [""]
    body = client.get(f"/api/instances/{native['job_id']}/telemetry", headers=headers).json()
    assert body["telemetry"] is None and body["reason"] == "no_readings"
