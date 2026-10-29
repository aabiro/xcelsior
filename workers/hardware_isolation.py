"""Fail-closed checks for purchased GPU or physical-host exclusivity.

Scheduling locks and exact GPU UUID binding exclude managed tenants. Providers
must qualify the host as exclusively managed by this agent. Busy or unreadable
hardware is refused without terminating unrelated workloads.
"""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
from pathlib import Path

GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]+")


def collect_inventory() -> list[dict]:
    """Physical identities and VRAM reported by the NVIDIA driver, never slots."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5 or not GPU_UUID.fullmatch(fields[1]):
            raise ValueError("Physical GPU inventory could not be verified")
        index, total, free = int(fields[0]), int(fields[3]), int(fields[4])
        if index < 0 or total <= 0 or not 0 <= free <= total:
            raise ValueError("Physical GPU inventory has invalid capacity")
        devices.append(
            {
                "device_index": index,
                "gpu_uuid": fields[1],
                "model": fields[2],
                "total_vram_mb": total,
                "free_vram_mb": free,
            }
        )
    if not devices or len({d["gpu_uuid"] for d in devices}) != len(devices):
        raise ValueError("Physical GPU inventory is empty or duplicated")
    return devices


def qualified_capabilities() -> dict[str, bool]:
    managed = os.environ.get("XCELSIOR_MANAGED_HARDWARE_ONLY", "").lower() == "true"
    full_host = os.environ.get("XCELSIOR_DEDICATED_HOST_ENABLED", "").lower() == "true"
    return {
        "gpu_exclusivity": managed,
        "dedicated_host": managed and full_host,
        "gpu_uuid_binding": True,
    }


def acquire_scope(
    gpu_uuids: list[str], *, physical_host: bool = False, lock_dir: str = "/run/xcelsior"
) -> dict:
    capabilities = qualified_capabilities()
    if not capabilities["dedicated_host" if physical_host else "gpu_exclusivity"]:
        raise ValueError("This host has not qualified the requested hardware isolation")
    if (
        not gpu_uuids
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(not GPU_UUID.fullmatch(uuid) for uuid in gpu_uuids)
    ):
        raise ValueError("Isolation requires the lease's exact physical GPU UUIDs")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    all_gpus = {line.strip() for line in inventory.stdout.splitlines() if line.strip()}
    selected = set(gpu_uuids)
    if not selected.issubset(all_gpus) or (physical_host and selected != all_gpus):
        raise ValueError("Dedicated must own every physical GPU reported by the host")
    state = {"fds": []}
    try:
        for uuid in sorted(selected):
            fd = os.open(Path(lock_dir) / f"exclusive-{uuid}.lock", os.O_WRONLY | os.O_CREAT, 0o660)
            state["fds"].append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        for line in processes.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or not GPU_UUID.fullmatch(fields[0]) or not fields[1].isdigit():
                raise ValueError("GPU occupancy could not be verified")
            if fields[0] in selected:
                raise ValueError("An allocated GPU is occupied by another process")
        return state
    except BaseException:
        release_scope(state)
        raise


def release_scope(state: dict) -> None:
    for fd in state.pop("fds", []):
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def restore_scope(gpu_uuids: list[str], *, lock_dir: str = "/run/xcelsior") -> dict:
    """Reacquire locks for an already running, journaled native attempt.

    The fenced lease and Docker labels authorize adoption; checking for an
    idle GPU here would reject the survivor's own processes.
    """
    if (
        not gpu_uuids
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(not GPU_UUID.fullmatch(value) for value in gpu_uuids)
    ):
        raise ValueError("Restoring isolation requires its journaled physical UUIDs")
    state = {"fds": []}
    try:
        for value in sorted(gpu_uuids):
            fd = os.open(
                Path(lock_dir) / f"exclusive-{value}.lock", os.O_WRONLY | os.O_CREAT, 0o660
            )
            state["fds"].append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return state
    except BaseException:
        release_scope(state)
        raise
