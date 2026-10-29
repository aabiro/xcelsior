"""Physical GPU inventory sync (blueprint §10.9 transitional bootstrap).

The reservation transaction allocates *concrete devices* from
``host_gpu_devices`` — but until the Track B worker observation pipeline
reports real per-GPU identities (NVML UUID, PCI bus), the only inventory
truth is the host payload's declared ``gpu_count`` / ``gpu_model`` /
``total_vram_gb``. This module projects that declaration into device
rows so the new scheduler can bind placements at device granularity.

Synthesized device identity is ``slot:{index}`` — stable per host+slot
across syncs, so allocations survive re-syncs. When real UUIDs arrive,
the observation pipeline replaces these rows (retire synthetic, insert
real) and ``inventory_generation`` invalidates in-flight scores.

All changes happen in the caller's transaction. Rows are locked in
``(host_id, gpu_uuid)`` order — the same canonical order the reservation
path uses — so sync and reservation can never deadlock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from psycopg import Connection

log = logging.getLogger("xcelsior.control_plane.inventory")

_SYNTHETIC_PREFIX = "slot:"


def _row(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return cast("dict[str, Any]", row)[key]
    return row[index]


@dataclass(frozen=True)
class InventorySyncResult:
    host_id: str
    created: int = 0
    retired: int = 0
    updated: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.created or self.retired or self.updated)


def _declared_inventory(host: dict[str, Any]) -> tuple[int, str, int]:
    """(gpu_count, model, per_gpu_vram_mb) from the host payload."""
    try:
        count = max(1, int(host.get("gpu_count", 1) or 1))
    except (TypeError, ValueError):
        count = 1
    model = str(host.get("gpu_model") or "").strip()
    try:
        total_vram_gb = float(host.get("total_vram_gb", 0) or 0)
    except (TypeError, ValueError):
        total_vram_gb = 0.0
    per_gpu_vram_mb = int(total_vram_gb * 1024 / count) if total_vram_gb > 0 else 0
    return count, model, per_gpu_vram_mb


def sync_host_gpu_inventory(conn: Connection, host: dict[str, Any]) -> InventorySyncResult:
    """Reconcile synthetic device rows with the host's declared inventory.

    Creates missing slots, retires surplus ones (never rows holding an
    active allocation — those are flagged and left for the reconciler),
    updates model/VRAM drift, and bumps ``hosts.inventory_generation``
    when anything changed so stale scores fail revalidation (§10.5).

    Device rows reported by the real observation pipeline (non-synthetic
    ``gpu_uuid``) make this a no-op for the host: real inventory always
    wins over declaration.
    """
    host_id = str(host.get("host_id") or "").strip()
    if not host_id:
        raise ValueError("host dict has no host_id")
    count, model, per_gpu_vram_mb = _declared_inventory(host)

    rows = conn.execute(
        """
        SELECT gpu_device_id, gpu_uuid, model, allocatable_vram_mb, retired_at
          FROM host_gpu_devices
         WHERE host_id = %s
         ORDER BY gpu_uuid
           FOR UPDATE
        """,
        (host_id,),
    ).fetchall()

    live_real = [
        r for r in rows
        if not str(_row(r, "gpu_uuid", 1)).startswith(_SYNTHETIC_PREFIX)
        and _row(r, "retired_at", 4) is None
    ]
    if live_real:
        return InventorySyncResult(host_id=host_id)

    by_uuid = {str(_row(r, "gpu_uuid", 1)): r for r in rows}
    wanted = {f"{_SYNTHETIC_PREFIX}{i}": i for i in range(count)}
    created = retired = updated = 0

    for gpu_uuid, index in wanted.items():
        existing = by_uuid.get(gpu_uuid)
        if existing is None:
            conn.execute(
                """
                INSERT INTO host_gpu_devices
                    (host_id, gpu_uuid, device_index, model,
                     total_vram_mb, allocatable_vram_mb, health)
                VALUES (%s, %s, %s, %s, %s, %s, 'unknown')
                """,
                (host_id, gpu_uuid, index, model, per_gpu_vram_mb, per_gpu_vram_mb),
            )
            created += 1
            continue
        if _row(existing, "retired_at", 4) is not None:
            conn.execute(
                """
                UPDATE host_gpu_devices
                   SET retired_at = NULL, model = %s,
                       total_vram_mb = %s, allocatable_vram_mb = %s
                 WHERE gpu_device_id = %s
                """,
                (model, per_gpu_vram_mb, per_gpu_vram_mb,
                 _row(existing, "gpu_device_id", 0)),
            )
            updated += 1
        elif (
            str(_row(existing, "model", 2)) != model
            or int(_row(existing, "allocatable_vram_mb", 3)) != per_gpu_vram_mb
        ):
            conn.execute(
                """
                UPDATE host_gpu_devices
                   SET model = %s, total_vram_mb = %s, allocatable_vram_mb = %s
                 WHERE gpu_device_id = %s
                """,
                (model, per_gpu_vram_mb, per_gpu_vram_mb,
                 _row(existing, "gpu_device_id", 0)),
            )
            updated += 1

    for gpu_uuid, row in by_uuid.items():
        if gpu_uuid in wanted or _row(row, "retired_at", 4) is not None:
            continue
        if not gpu_uuid.startswith(_SYNTHETIC_PREFIX):
            continue
        device_id = _row(row, "gpu_device_id", 0)
        active = conn.execute(
            """
            SELECT 1 FROM gpu_device_allocations
             WHERE gpu_device_id = %s AND status = 'active'
             LIMIT 1
            """,
            (device_id,),
        ).fetchone()
        if active is not None:
            # Shrinking under an active allocation is a reconciler case
            # (§12) — never yank a device out from under a placement.
            log.warning(
                "inventory shrink deferred: host=%s device=%s has an active "
                "allocation", host_id, gpu_uuid,
            )
            continue
        conn.execute(
            "UPDATE host_gpu_devices SET retired_at = clock_timestamp() "
            "WHERE gpu_device_id = %s",
            (device_id,),
        )
        retired += 1

    result = InventorySyncResult(
        host_id=host_id, created=created, retired=retired, updated=updated
    )
    if result.changed:
        conn.execute(
            """
            UPDATE hosts
               SET inventory_generation = COALESCE(inventory_generation, 0) + 1
             WHERE host_id = %s
            """,
            (host_id,),
        )
        log.info(
            "inventory sync host=%s: +%d created, %d retired, %d updated",
            host_id, created, retired, updated,
        )
    return result
