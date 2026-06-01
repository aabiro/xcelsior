"""Local provider acceptance checks for Xcelsior worker hosts."""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass

from security import admit_node, get_local_versions, recommend_runtime

REQUIRED_VERSION_KEYS = ("runc", "docker", "nvidia_driver", "nvidia_toolkit")
DEFAULT_DOCKER_PROBE_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"


@dataclass
class AcceptanceCheck:
    name: str
    ok: bool
    detail: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return asdict(self)


def _format_missing(keys: list[str]) -> str:
    if not keys:
        return "all required versions detected"
    return "missing: " + ", ".join(keys)


def _gpu_summary(gpus: list[dict]) -> list[dict]:
    summary = []
    for gpu in gpus:
        summary.append(
            {
                "index": gpu.get("index", 0),
                "gpu_model": gpu.get("gpu_model", ""),
                "memory_total_gb": gpu.get("memory_total_gb", 0),
                "memory_free_gb": gpu.get("memory_free_gb", 0),
                "compute_capability": gpu.get("compute_capability", ""),
                "driver_version": gpu.get("driver_version", ""),
                "temperature_c": gpu.get("temperature_c", 0),
            }
        )
    return summary


def _run_docker_probe(image: str, timeout: int = 60) -> AcceptanceCheck:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        image,
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return AcceptanceCheck("docker_gpu_probe", False, "docker command not found")
    except subprocess.TimeoutExpired:
        return AcceptanceCheck("docker_gpu_probe", False, f"timed out after {timeout}s")

    output = (result.stdout or result.stderr or "").strip()
    if result.returncode == 0:
        return AcceptanceCheck("docker_gpu_probe", True, output or "GPU visible inside container")
    return AcceptanceCheck(
        "docker_gpu_probe",
        False,
        output or f"docker exited with status {result.returncode}",
    )


def probe_local_host(
    host_id: str = "local",
    expected_gpu_model: str = "",
    min_vram_gb: float = 0,
    docker_probe: bool = False,
    docker_image: str = DEFAULT_DOCKER_PROBE_IMAGE,
) -> dict:
    """Run local checks that indicate whether a host is ready for worker admission."""
    checks: list[AcceptanceCheck] = []
    versions = get_local_versions()
    missing_versions = [key for key in REQUIRED_VERSION_KEYS if not versions.get(key)]
    checks.append(
        AcceptanceCheck(
            "versions_present",
            not missing_versions,
            _format_missing(missing_versions),
        )
    )

    admitted, admission_details = admit_node(host_id, versions, expected_gpu_model or None)
    checks.append(
        AcceptanceCheck(
            "version_gate",
            admitted,
            (
                "passes security version gate"
                if admitted
                else "; ".join(admission_details.get("rejection_reasons", []))
            ),
        )
    )

    try:
        from nvml_telemetry import collect_all_gpus, nvml_init

        nvml_init()
        gpus = collect_all_gpus()
    except Exception as exc:
        gpus = []
        checks.append(AcceptanceCheck("gpu_telemetry", False, f"telemetry failed: {exc}"))
    else:
        checks.append(
            AcceptanceCheck(
                "gpu_telemetry",
                bool(gpus),
                f"{len(gpus)} GPU(s) detected" if gpus else "no NVIDIA GPUs detected",
            )
        )

    if expected_gpu_model:
        expected = expected_gpu_model.lower()
        matched = any(expected in (gpu.get("gpu_model") or "").lower() for gpu in gpus)
        checks.append(
            AcceptanceCheck(
                "expected_gpu",
                matched,
                (
                    f"found {expected_gpu_model}"
                    if matched
                    else f"{expected_gpu_model} not found in detected GPUs"
                ),
            )
        )

    if min_vram_gb:
        max_vram = max((float(gpu.get("memory_total_gb") or 0) for gpu in gpus), default=0.0)
        checks.append(
            AcceptanceCheck(
                "minimum_vram",
                max_vram >= min_vram_gb,
                f"max detected VRAM {max_vram:g} GB; required {min_vram_gb:g} GB",
            )
        )

    gpu_for_runtime = expected_gpu_model
    if not gpu_for_runtime and gpus:
        gpu_for_runtime = gpus[0].get("gpu_model", "")
    runtime, runtime_reason = recommend_runtime(gpu_for_runtime or "unknown")
    checks.append(
        AcceptanceCheck("runtime_recommendation", True, f"{runtime}: {runtime_reason}", "info")
    )

    if docker_probe:
        checks.append(_run_docker_probe(docker_image))

    ready = all(check.ok for check in checks if check.severity == "error")
    return {
        "host_id": host_id,
        "ready": ready,
        "checked_at": time.time(),
        "versions": versions,
        "admission": admission_details,
        "recommended_runtime": runtime,
        "runtime_reason": runtime_reason,
        "gpus": _gpu_summary(gpus),
        "checks": [check.to_dict() for check in checks],
    }
