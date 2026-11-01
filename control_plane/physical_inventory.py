"""Promote complete authenticated physical inventories without losing allocations.

A complete report changes device membership on a busy host too: only devices
that hold an allocation are deferred (marked unhealthy when missing). Old
synthetic rows remain as retired historical identities, while fresh
observations supply real GPU UUIDs. An unavailable probe is distinct from an
explicit complete empty inventory.
"""

from __future__ import annotations

import re
from typing import Any


def validate_inventory(inventory: list[dict[str, Any]]) -> None:
    uuids, indices = set(), set()
    for item in inventory:
        uuid = item.get("gpu_uuid")
        index, memory = item.get("device_index"), item.get("total_vram_mb")
        if (
            not isinstance(uuid, str)
            or not re.fullmatch(r"GPU-[A-Za-z0-9-]{1,100}", uuid)
            or type(index) is not int
            or not 0 <= index < 64
            or type(memory) is not int
            or not 0 < memory < 2**31
            or not isinstance(item.get("model"), str)
            or not str(item["model"]).strip()
            or len(item["model"]) > 256
            or (item.get("pci_bus_id") is not None and not isinstance(item["pci_bus_id"], str))
            or uuid in uuids
            or index in indices
        ):
            raise ValueError("physical GPU inventory is incomplete or ambiguous")
        uuids.add(uuid)
        indices.add(index)


def sync_physical_inventory(conn, *, host_id: str, inventory: list[dict]) -> bool:
    """Caller holds the host row; use the reservation's host/device lock order.

    A complete inventory is authoritative for every device it can safely
    change. Devices that still hold an active allocation keep their row (the
    attempt's authority points at it) but a device missing from the report is
    marked ``unhealthy`` so it is never offered again; it is retired by the
    first complete report after its allocation is released. Every other
    device missing from the report is retired immediately, and new devices
    are added, whether or not some other GPU on the host is busy.
    """
    validate_inventory(inventory)
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cursor:
        rows = cursor.execute(
            "SELECT gpu_device_id,gpu_uuid,device_index,pci_bus_id,model,total_vram_mb,retired_at,"
            "health FROM host_gpu_devices WHERE host_id=%s ORDER BY gpu_uuid FOR UPDATE",
            (host_id,),
        ).fetchall()
        busy = {
            str(row["gpu_device_id"])
            for row in cursor.execute(
                "SELECT DISTINCT gpu_device_id FROM gpu_device_allocations "
                "WHERE host_id=%s AND released_at IS NULL",
                (host_id,),
            ).fetchall()
        }
    by_uuid = {row["gpu_uuid"]: row for row in rows}
    wanted = {item["gpu_uuid"] for item in inventory}
    missing = [row for row in rows if row["retired_at"] is None and row["gpu_uuid"] not in wanted]
    retire = [row["gpu_device_id"] for row in missing if str(row["gpu_device_id"]) not in busy]
    # A busy device that vanished is deferred, never yanked from its attempt.
    quarantine = [
        row["gpu_device_id"]
        for row in missing
        if str(row["gpu_device_id"]) in busy and row["health"] != "unhealthy"
    ]
    changed = bool(retire or quarantine)
    for item in inventory:
        old = by_uuid.get(item["gpu_uuid"])
        if (
            old is None
            or old["retired_at"] is not None
            or old["health"] == "unhealthy"
            or any(
                old[key] != item.get(key)
                for key in ("device_index", "pci_bus_id", "model", "total_vram_mb")
            )
        ):
            changed = True
    if changed:
        conn.execute(
            "UPDATE hosts SET inventory_generation=inventory_generation+1 WHERE host_id=%s",
            (host_id,),
        )
    if retire:
        conn.execute(
            "UPDATE host_gpu_devices SET retired_at=clock_timestamp() "
            "WHERE gpu_device_id=ANY(%s) AND retired_at IS NULL",
            (retire,),
        )
    if quarantine:
        conn.execute(
            "UPDATE host_gpu_devices SET health='unhealthy' WHERE gpu_device_id=ANY(%s)",
            (quarantine,),
        )
    for item in sorted(inventory, key=lambda record: record["gpu_uuid"]):
        # Reappearing in a complete report clears the missing-device mark;
        # nothing else writes device health.
        conn.execute(
            """INSERT INTO host_gpu_devices
                (host_id,gpu_uuid,device_index,pci_bus_id,model,total_vram_mb,allocatable_vram_mb,
                 health,last_observed_at,inventory_generation)
            SELECT %s,%s,%s,%s,%s,%s,%s,'unknown',clock_timestamp(),inventory_generation
            FROM hosts WHERE host_id=%s
            ON CONFLICT(host_id,gpu_uuid) DO UPDATE SET
                device_index=EXCLUDED.device_index,pci_bus_id=EXCLUDED.pci_bus_id,
                model=EXCLUDED.model,total_vram_mb=EXCLUDED.total_vram_mb,
                allocatable_vram_mb=EXCLUDED.allocatable_vram_mb,
                last_observed_at=EXCLUDED.last_observed_at,retired_at=NULL,
                health=CASE WHEN host_gpu_devices.health='unhealthy' THEN 'unknown'
                            ELSE host_gpu_devices.health END,
                inventory_generation=EXCLUDED.inventory_generation""",
            (
                host_id,
                item["gpu_uuid"],
                item["device_index"],
                item.get("pci_bus_id"),
                item["model"],
                item["total_vram_mb"],
                item["total_vram_mb"],
                host_id,
            ),
        )
    return changed
