"""Per-GPU telemetry: what each GPU reported, kept as history, shown to its renter.

The worker's telemetry report carries host-wide fields (NIC and disk IO, host
CPU, ``active_jobs``, ``volume_health`` with every mounted volume id) and, since
agent 2.3.0, a ``gpus`` list: one entry per device, keyed by ``index`` and
``gpu_uuid``. Only that list reaches a renter, and only the entries for the
GPUs their run was given, filtered to :data:`RENTER_GPU_FIELDS`. A renter on a
shared host never sees another tenant's GPU, the host's job count or anyone's
volume ids.

Storage:

* ``telemetry_latest`` keeps the newest sample per ``(host_id, gpu_uuid)``
  (the host-wide row is ``gpu_uuid = ''``, as before).
* ``telemetry_samples`` (057, monthly partitions kept by
  ``telemetry_partition_maintenance``) keeps history, at most one row per GPU
  every :data:`HISTORY_INTERVAL_SECONDS`.

Which GPUs a run had is recorded exactly by ``gpu_device_allocations`` joined
to ``host_gpu_devices.gpu_uuid``; a run's history is the samples for those
GPUs between ``allocated_at`` and ``released_at``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("xcelsior.gpu_telemetry")

#: The fields a renter sees for each of their GPUs. The worker sends these
#: names (``worker_agent._gpu_sample``); the route and the frontend type read
#: this list.
RENTER_GPU_FIELDS = (
    "utilization",
    "memory_util",
    "memory_used_gb",
    "memory_total_gb",
    "temp",
    "power_draw_w",
    "power_limit_w",
)
#: Fields kept in each history point (the chart's series).
HISTORY_FIELDS = ("utilization", "memory_used_gb", "temp", "power_draw_w")
HISTORY_INTERVAL_SECONDS = 30
#: A sample older than this reads as stale.
STALE_AFTER_SECONDS = 30
MAX_GPUS_PER_REPORT = 64
MAX_HISTORY_POINTS = 2880  # 24 h at one point per 30 s

_RANGES: dict[str, tuple[float, float | None]] = {
    "utilization": (0, 100),
    "memory_util": (0, 100),
    "memory_used_gb": (0, None),
    "memory_total_gb": (0, None),
    "temp": (-273.15, None),
    "power_draw_w": (0, None),
    "power_limit_w": (0, None),
}


def validate_gpu_samples(gpus: Any) -> list[dict]:
    """The ``gpus`` list of a telemetry report, checked. Raises ValueError."""
    if not isinstance(gpus, list):
        raise ValueError("Telemetry gpus must be a list")
    if len(gpus) > MAX_GPUS_PER_REPORT:
        raise ValueError("Too many GPUs in one telemetry report")
    seen: set[int] = set()
    for entry in gpus:
        if not isinstance(entry, dict):
            raise ValueError("Each telemetry gpu must be an object")
        index = entry.get("index")
        if type(index) is not int or not 0 <= index < MAX_GPUS_PER_REPORT or index in seen:
            raise ValueError("Each telemetry gpu needs a distinct index")
        seen.add(index)
        uuid = entry.get("gpu_uuid", "")
        if not isinstance(uuid, str) or len(uuid) > 128:
            raise ValueError("Invalid telemetry gpu_uuid")
        for key, (minimum, maximum) in _RANGES.items():
            value = entry.get(key)
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid telemetry measurement: gpus[].{key}")
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f"Invalid telemetry measurement: gpus[].{key}")
        used, total = entry.get("memory_used_gb"), entry.get("memory_total_gb")
        if used is not None and total is not None and used > total:
            raise ValueError("GPU memory used cannot exceed total GPU memory")
    return gpus


def renter_view(sample: dict) -> dict:
    """One GPU's sample with only the fields a renter may see."""
    return {key: sample.get(key) for key in RENTER_GPU_FIELDS}


def record_gpu_samples(conn, host_id: str, gpus: list[dict]) -> int:
    """Keep each GPU's newest sample, and history every 30 s. Returns rows added.

    A sample without a ``gpu_uuid`` (the nvidia-smi fallback on an old
    ``nvml_telemetry.py``) is matched to the host's inventory by device index;
    one that matches nothing is dropped, never stored under a guess.
    """
    from psycopg.types.json import Jsonb

    if not gpus:
        return 0
    by_index = {
        int(row[0]): str(row[1])
        for row in conn.execute(
            """SELECT device_index, gpu_uuid FROM host_gpu_devices
                WHERE host_id = %s AND device_index IS NOT NULL
                  AND parent_gpu_device_id IS NULL""",
            (host_id,),
        ).fetchall()
    }
    added = 0
    for entry in gpus:
        uuid = str(entry.get("gpu_uuid") or "") or by_index.get(int(entry["index"]), "")
        if not uuid:
            continue
        sample = {"index": int(entry["index"]), **renter_view(entry)}
        conn.execute(
            """INSERT INTO telemetry_latest (host_id, gpu_uuid, sample, received_at)
               VALUES (%s, %s, %s, clock_timestamp())
               ON CONFLICT (host_id, gpu_uuid) DO UPDATE
                  SET sample = EXCLUDED.sample, received_at = clock_timestamp()""",
            (host_id, uuid, Jsonb(sample)),
        )
        row = conn.execute(
            """INSERT INTO telemetry_samples (host_id, gpu_uuid, sample)
               SELECT %s, %s, %s
                WHERE NOT EXISTS (
                      SELECT 1 FROM telemetry_samples
                       WHERE host_id = %s AND gpu_uuid = %s
                         AND received_at > clock_timestamp() - make_interval(secs => %s))
               RETURNING id""",
            (host_id, uuid, Jsonb(sample), host_id, uuid, HISTORY_INTERVAL_SECONDS),
        ).fetchone()
        added += int(row is not None)
    return added


def run_gpus(conn, job_id: str) -> list[dict]:
    """The GPUs of this job's newest attempt that held any, with their time span."""
    rows = conn.execute(
        """WITH latest AS (
               SELECT attempt_id FROM gpu_device_allocations
                WHERE job_id = %s
                ORDER BY allocated_at DESC LIMIT 1)
           SELECT a.host_id, d.gpu_uuid, d.device_index, d.model,
                  EXTRACT(EPOCH FROM a.allocated_at) AS allocated_at,
                  EXTRACT(EPOCH FROM a.released_at) AS released_at
             FROM gpu_device_allocations a
             JOIN host_gpu_devices d ON d.gpu_device_id = a.gpu_device_id
            WHERE a.job_id = %s AND a.attempt_id IN (SELECT attempt_id FROM latest)
            ORDER BY d.device_index NULLS LAST, d.gpu_uuid""",
        (job_id, job_id),
    ).fetchall()
    return [
        {
            "host_id": str(r[0]),
            "gpu_uuid": str(r[1]),
            "index": r[2],
            "gpu_model": str(r[3] or ""),
            "allocated_at": float(r[4]),
            "released_at": float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]


def latest_for_run(conn, job_id: str) -> dict | None:
    """Each of the run's GPUs with its newest sample; None when it had no GPUs."""
    gpus = run_gpus(conn, job_id)
    if not gpus:
        return None
    out = []
    for gpu in gpus:
        row = conn.execute(
            """SELECT sample, EXTRACT(EPOCH FROM received_at),
                      EXTRACT(EPOCH FROM clock_timestamp())
                 FROM telemetry_latest WHERE host_id = %s AND gpu_uuid = %s""",
            (gpu["host_id"], gpu["gpu_uuid"]),
        ).fetchone()
        sample = row[0] if row else None
        received = float(row[1]) if row else None
        now = float(row[2]) if row else None
        out.append(
            {
                "index": gpu["index"],
                "gpu_model": gpu["gpu_model"],
                **(renter_view(sample) if isinstance(sample, dict) else {}),
                "received_at": received,
                "stale": (received is None or now is None or now - received > STALE_AFTER_SECONDS),
            }
        )
    reporting = [g for g in out if g["received_at"] is not None]
    return {
        "gpus": out,
        "stale": not reporting or any(g["stale"] for g in out),
        "reporting": bool(reporting),
    }


def history_for_run(conn, job_id: str, since: float | None = None, limit: int = 720) -> dict | None:
    """The run's per-GPU series, oldest first, at most ``limit`` points per GPU."""
    gpus = run_gpus(conn, job_id)
    if not gpus:
        return None
    limit = max(1, min(int(limit), MAX_HISTORY_POINTS))
    series = []
    for gpu in gpus:
        start = max(gpu["allocated_at"], float(since or 0))
        rows = conn.execute(
            """SELECT EXTRACT(EPOCH FROM received_at), sample FROM telemetry_samples
                WHERE host_id = %s AND gpu_uuid = %s
                  AND received_at >= to_timestamp(%s)
                  AND (%s::double precision IS NULL OR received_at <= to_timestamp(%s))
                ORDER BY received_at DESC LIMIT %s""",
            (
                gpu["host_id"],
                gpu["gpu_uuid"],
                start,
                gpu["released_at"],
                gpu["released_at"],
                limit,
            ),
        ).fetchall()
        points = [
            {"t": float(t), **{key: (s or {}).get(key) for key in HISTORY_FIELDS}}
            for t, s in reversed(rows)
        ]
        series.append({"index": gpu["index"], "gpu_model": gpu["gpu_model"], "points": points})
    return {
        "interval_seconds": HISTORY_INTERVAL_SECONDS,
        "fields": list(HISTORY_FIELDS),
        "gpus": series,
    }
