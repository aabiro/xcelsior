"""Boot-time fitness checks — fatal exit on failure in production workers."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class FitnessConfig:
    require_cuda: bool = True
    min_disk_gb: float = 5.0
    min_free_mem_gb: float = 2.0
    disk_mounts: list[str] = field(default_factory=lambda: ["/"])
    require_psutil: bool = False


def run_fitness_checks(config: FitnessConfig | None = None) -> list[str]:
    """
    Return a list of failure reasons. Empty list means all checks passed.
    """
    cfg = config or FitnessConfig()
    failures: list[str] = []

    if cfg.require_cuda and not _cuda_available():
        failures.append("CUDA GPU not available")

    mounts = cfg.disk_mounts or ["/"]
    for mount in mounts:
        try:
            disk = shutil.disk_usage(mount)
        except OSError:
            failures.append(f"Disk mount unavailable: {mount}")
            continue
        free_disk_gb = disk.free / (1024**3)
        if free_disk_gb < cfg.min_disk_gb:
            failures.append(f"Insufficient disk space on {mount} ({free_disk_gb:.1f} GB free)")

    try:
        import psutil  # type: ignore[import-untyped]

        free_mem_gb = psutil.virtual_memory().available / (1024**3)
        if free_mem_gb < cfg.min_free_mem_gb:
            failures.append(f"Insufficient memory ({free_mem_gb:.1f} GB available)")
    except ImportError:
        if cfg.require_psutil:
            failures.append("psutil is required for memory fitness checks but is not installed")

    weights_path = os.environ.get("XCELSIOR_WEIGHTS_READY_PATH", "")
    if weights_path and not os.path.exists(weights_path):
        failures.append(f"Weights not found at {weights_path}")

    cache_mount = os.environ.get("XCELSIOR_MODEL_CACHE_MOUNT", "")
    if cache_mount and cache_mount not in mounts:
        try:
            disk = shutil.disk_usage(cache_mount)
            free_disk_gb = disk.free / (1024**3)
            if free_disk_gb < cfg.min_disk_gb:
                failures.append(
                    f"Insufficient model-cache disk on {cache_mount} ({free_disk_gb:.1f} GB free)"
                )
        except OSError:
            failures.append(f"Model cache mount unavailable: {cache_mount}")

    return failures


def _cuda_available() -> bool:
    if os.environ.get("XCELSIOR_SKIP_CUDA_CHECK", "").lower() in ("1", "true", "yes"):
        return True
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0 and "GPU" in (proc.stdout or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False