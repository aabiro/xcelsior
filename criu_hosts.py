"""GPU/Docker checkpoint capability probing and resumable-job helpers.

Competitors (Modal GPU Memory Snapshots, RunPod, cedana) expose checkpoint/restore
for spot preemption and faster cold starts. xcelsior uses:

- ``docker-criu``: Docker ``checkpoint create`` (CPU + process state; GPU via driver ≥570)
- ``gpu-criu``: same stack + NVIDIA driver gate for CUDA context restore (CRIUgpu path)

Hosts report ``checkpoint_class`` on heartbeat; preempted jobs carry ``resume_from``
metadata so the scheduler can resume instead of cold-restart.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("xcelsior.criu_hosts")

CHECKPOINT_CLASS_NONE = ""
CHECKPOINT_CLASS_DOCKER_CRIU = "docker-criu"
CHECKPOINT_CLASS_GPU_CRIU = "gpu-criu"

MIN_NVIDIA_DRIVER_MAJOR = 570
# Validated reference host: ASUS RTX 2060 + driver 580.x (CRIUgpu gate passes; criu binary host ops).
REFERENCE_CHECKPOINT_GPU = "RTX 2060"
REFERENCE_CHECKPOINT_DRIVER = "580"
_CHECKPOINT_DIR = Path(os.environ.get("XCELSIOR_CHECKPOINT_DIR", "checkpoints"))


def cuda_driver_requirements() -> dict[str, Any]:
    """Pinned CUDA/driver requirements for CRIUgpu rollout (§10 row 31)."""
    return {
        "min_nvidia_driver_major": MIN_NVIDIA_DRIVER_MAJOR,
        "reference_gpu": REFERENCE_CHECKPOINT_GPU,
        "reference_driver_prefix": REFERENCE_CHECKPOINT_DRIVER,
        "docker_experimental_required": True,
        "criu_required": True,
        "checkpoint_classes": [CHECKPOINT_CLASS_DOCKER_CRIU, CHECKPOINT_CLASS_GPU_CRIU],
        "notes": (
            f"NVIDIA driver >={MIN_NVIDIA_DRIVER_MAJOR} enables gpu-criu CUDA context restore; "
            "Docker experimental=true required for checkpoint create."
        ),
    }


def _run(cmd: list[str] | str, *, timeout: float = 15.0) -> tuple[int, str, str]:
    try:
        if isinstance(cmd, str):
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return 127, "", str(exc)


def _parse_nvidia_driver_version(raw: str) -> float | None:
    m = re.search(r"(\d{3})\.(\d+)", raw or "")
    if not m:
        return None
    try:
        return float(f"{m.group(1)}.{m.group(2)}")
    except ValueError:
        return None


def probe_checkpoint_stack(*, force_class: str | None = None) -> dict[str, Any]:
    """Probe local Docker/CRIU/NVIDIA stack for checkpoint capability."""
    if force_class:
        fc = force_class.strip().lower()
        if fc in (CHECKPOINT_CLASS_DOCKER_CRIU, CHECKPOINT_CLASS_GPU_CRIU):
            return {
                "criu_available": True,
                "criu_version": "forced",
                "docker_experimental": True,
                "nvidia_driver": "forced",
                "nvidia_driver_version": 999.0,
                "checkpoint_class": fc,
                "probe_source": "env_force",
            }

    forced = os.environ.get("XCELSIOR_CHECKPOINT_CLASS", "").strip().lower()
    if forced in (CHECKPOINT_CLASS_DOCKER_CRIU, CHECKPOINT_CLASS_GPU_CRIU):
        return probe_checkpoint_stack(force_class=forced)

    criu_rc, criu_out, _ = _run(["criu", "--version"])
    criu_available = criu_rc == 0
    criu_version = criu_out.splitlines()[0] if criu_out else ""

    docker_rc, docker_out, docker_err = _run(["docker", "info", "--format", "{{.Experimental}}"])
    docker_experimental = docker_rc == 0 and docker_out.lower() == "true"
    if not docker_experimental:
        # Docker 29+ removed {{.Experimental}} template field; parse human output.
        text_rc, text_out, _ = _run(["docker", "info"])
        if text_rc == 0 and "experimental: true" in text_out.lower():
            docker_experimental = True

    nvidia_rc, nvidia_out, _ = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    driver_raw = nvidia_out.splitlines()[0].strip() if nvidia_rc == 0 and nvidia_out else ""
    driver_ver = _parse_nvidia_driver_version(driver_raw)

    checkpoint_class = CHECKPOINT_CLASS_NONE
    if criu_available and docker_experimental:
        checkpoint_class = CHECKPOINT_CLASS_DOCKER_CRIU
        if driver_ver is not None and driver_ver >= MIN_NVIDIA_DRIVER_MAJOR:
            checkpoint_class = CHECKPOINT_CLASS_GPU_CRIU

    return {
        "criu_available": criu_available,
        "criu_version": criu_version,
        "docker_experimental": docker_experimental,
        "nvidia_driver": driver_raw,
        "nvidia_driver_version": driver_ver,
        "checkpoint_class": checkpoint_class,
        "probe_source": "local",
        "min_nvidia_driver": MIN_NVIDIA_DRIVER_MAJOR,
    }


def host_supports_checkpoint(host: dict[str, Any] | None) -> bool:
    """True when a host row advertises docker-criu or gpu-criu."""
    if not host:
        return False
    cls = str(host.get("checkpoint_class") or "").strip().lower()
    return cls in (CHECKPOINT_CLASS_DOCKER_CRIU, CHECKPOINT_CLASS_GPU_CRIU)


def host_checkpoint_class(host: dict[str, Any] | None) -> str:
    return str((host or {}).get("checkpoint_class") or "").strip().lower()


def job_has_checkpoint(job: dict[str, Any] | None) -> bool:
    if not job:
        return False
    meta = job.get("resume_from")
    return isinstance(meta, dict) and bool(meta.get("success") or meta.get("checkpoint_name"))


def job_is_resumable(job: dict[str, Any] | None) -> bool:
    """A job can resume from checkpoint when metadata is present and not expired."""
    if not job_has_checkpoint(job):
        return False
    # job_has_checkpoint() already rejects None; restate it for the reader
    # (and the type checker) rather than relying on that coupling.
    if job is None:
        return False
    meta = job.get("resume_from") or {}
    if meta.get("success") is False:
        return False
    created = float(meta.get("created_at") or 0)
    if created and (time.time() - created) > float(
        os.environ.get("XCELSIOR_CHECKPOINT_TTL_SEC", "86400")
    ):
        return False
    return True


def enrich_job_resumable(job: dict[str, Any]) -> dict[str, Any]:
    """Add ``resumable`` and ``checkpoint_class`` fields for API consumers."""
    job["resumable"] = job_is_resumable(job)
    raw_meta = job.get("resume_from")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    job["checkpoint_class"] = str(meta.get("checkpoint_class") or job.get("checkpoint_class") or "")
    return job


def docker_checkpoint_local(
    container_name: str,
    job_id: str,
    *,
    checkpoint_class: str = CHECKPOINT_CLASS_DOCKER_CRIU,
    leave_running: bool = False,
) -> dict[str, Any] | None:
    """Create a local Docker CRIU checkpoint (worker-agent path)."""
    container_name = (container_name or "").strip()
    job_id = (job_id or "").strip()
    if not container_name or not job_id:
        return None

    checkpoint_name = f"ckpt-{job_id}-{int(time.time())}"
    remote_dir = Path(f"/tmp/xcelsior-checkpoints/{checkpoint_name}")
    local_dir = _CHECKPOINT_DIR / checkpoint_name
    local_dir.mkdir(parents=True, exist_ok=True)

    leave = "false" if not leave_running else "true"
    cmd = (
        f"docker checkpoint create "
        f"--checkpoint-dir={shlex.quote(str(remote_dir))} "
        f"--leave-running={leave} "
        f"{shlex.quote(container_name)} {shlex.quote(checkpoint_name)}"
    )
    rc, stdout, stderr = _run(cmd, timeout=120.0)
    meta = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(remote_dir),
        "local_path": str(local_dir),
        "job_id": job_id,
        "container": container_name,
        "created_at": time.time(),
        "success": rc == 0,
        "checkpoint_class": checkpoint_class,
        "stderr": (stderr or stdout or "")[:500],
    }
    try:
        (local_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError as exc:
        log.warning("checkpoint meta write failed job=%s: %s", job_id, exc)
    if rc != 0:
        log.error("docker checkpoint failed job=%s container=%s: %s", job_id, container_name, stderr)
        return None
    log.info("CHECKPOINT OK job=%s container=%s name=%s class=%s", job_id, container_name, checkpoint_name, checkpoint_class)
    return meta


def merge_checkpoint_capabilities(
    host_entry: dict[str, Any],
    probe: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge probe results into a host heartbeat/register payload."""
    probe = probe or {}
    cls = str(probe.get("checkpoint_class") or "").strip()
    if cls:
        host_entry["checkpoint_class"] = cls
    caps = dict(host_entry.get("capabilities") or {})
    caps["checkpoint"] = {
        "class": cls,
        "criu_available": bool(probe.get("criu_available")),
        "docker_experimental": bool(probe.get("docker_experimental")),
        "nvidia_driver": probe.get("nvidia_driver"),
        "nvidia_driver_version": probe.get("nvidia_driver_version"),
        "criu_version": probe.get("criu_version"),
        "probed_at": time.time(),
    }
    host_entry["capabilities"] = caps
    if probe.get("nvidia_driver"):
        host_entry["cuda_driver_version"] = probe.get("nvidia_driver")
    return host_entry