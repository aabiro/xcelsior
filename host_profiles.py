"""Reusable provider hardware profiles for Xcelsior operators."""

from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HostProfile:
    profile_id: str
    label: str
    gpu_model: str
    total_vram_gb: float
    usable_vram_gb: float
    compute_capability: str
    architecture: str
    default_rate_cad_per_hour: float
    min_rate_cad_per_hour: float
    max_rate_cad_per_hour: float
    recommended_runtime: str
    description: str
    recommended_workloads: tuple[str, ...]
    avoid_workloads: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["recommended_workloads"] = list(self.recommended_workloads)
        data["avoid_workloads"] = list(self.avoid_workloads)
        return data


HOST_PROFILES: dict[str, HostProfile] = {
    "rtx3060-local": HostProfile(
        profile_id="rtx3060-local",
        label="RTX 3060 12GB Local Dev Tier",
        gpu_model="RTX 3060",
        total_vram_gb=12.0,
        usable_vram_gb=11.0,
        compute_capability="8.6",
        architecture="Ampere sm86",
        default_rate_cad_per_hour=0.16,
        min_rate_cad_per_hour=0.08,
        max_rate_cad_per_hour=0.35,
        recommended_runtime="runsc",
        description=(
            "Affordable consumer GPU profile for dev/staging inference, telemetry burn-in, "
            "and scheduler dogfooding on 12 GB VRAM hosts."
        ),
        recommended_workloads=(
            "CUDA smoke tests and provider acceptance",
            "small PyTorch inference jobs",
            "7B/14B quantized LLM inference",
            "SD 1.5 or conservative SDXL single-batch experiments",
            "worker telemetry, billing, and lifecycle burn-in",
        ),
        avoid_workloads=(
            "large 24 GB+ diffusion pipelines",
            "multi-ControlNet SDXL production runs",
            "70B model serving or fine-tuning",
            "multi-GPU or MIG workloads",
        ),
    )
}

PROFILE_ALIASES = {
    "3060": "rtx3060-local",
    "rtx3060": "rtx3060-local",
    "rtx-3060": "rtx3060-local",
    "local-dev": "rtx3060-local",
}


def list_host_profiles() -> list[dict]:
    """Return all supported host profiles as serializable dictionaries."""
    return [profile.to_dict() for profile in HOST_PROFILES.values()]


def normalize_profile_id(profile_id: str) -> str:
    """Resolve aliases to canonical profile IDs."""
    key = (profile_id or "").strip().lower()
    return PROFILE_ALIASES.get(key, key)


def get_host_profile(profile_id: str) -> dict:
    """Return a host profile by ID or alias."""
    canonical = normalize_profile_id(profile_id)
    profile = HOST_PROFILES.get(canonical)
    if not profile:
        available = ", ".join(sorted(HOST_PROFILES))
        raise KeyError(f"Unknown host profile {profile_id!r}. Available profiles: {available}")
    return profile.to_dict()


def profile_for_gpu_model(gpu_model: str) -> dict | None:
    """Best-effort profile lookup from an observed GPU model string."""
    model = (gpu_model or "").lower()
    for profile in HOST_PROFILES.values():
        if profile.gpu_model.lower() in model:
            return profile.to_dict()
    return None


def _quote_cmd(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def build_host_add_command(
    profile_id: str,
    host_id: str,
    ip: str,
    country: str = "CA",
    province: str = "ON",
    free_vram_gb: float | None = None,
) -> str:
    """Build the CLI command that registers a host from a profile."""
    profile = get_host_profile(profile_id)
    free_vram = profile["usable_vram_gb"] if free_vram_gb is None else free_vram_gb
    return _quote_cmd(
        [
            "xcelsior",
            "host-add",
            "--id",
            host_id,
            "--ip",
            ip,
            "--gpu",
            profile["gpu_model"],
            "--vram",
            profile["total_vram_gb"],
            "--free-vram",
            free_vram,
            "--rate",
            profile["default_rate_cad_per_hour"],
            "--country",
            country,
            "--province",
            province,
        ]
    )


def build_market_list_command(
    profile_id: str,
    host_id: str,
    owner: str = "internal",
    description: str | None = None,
) -> str:
    """Build the CLI command that lists a profiled host in the legacy marketplace."""
    profile = get_host_profile(profile_id)
    desc = description or profile["description"]
    return _quote_cmd(
        [
            "xcelsior",
            "market-list",
            host_id,
            "--gpu",
            profile["gpu_model"],
            "--vram",
            profile["total_vram_gb"],
            "--price",
            profile["default_rate_cad_per_hour"],
            "--owner",
            owner,
            "--desc",
            desc,
        ]
    )


def validate_job_for_profile(
    profile_id: str,
    vram_needed_gb: float,
    gpu_model: str = "",
) -> dict:
    """Explain whether a job shape fits a provider profile."""
    profile = get_host_profile(profile_id)
    reasons: list[str] = []
    if gpu_model and profile["gpu_model"].lower() not in gpu_model.lower():
        reasons.append(f"requested GPU {gpu_model!r} does not match {profile['gpu_model']}")
    if vram_needed_gb > profile["usable_vram_gb"]:
        reasons.append(
            f"requested VRAM {vram_needed_gb:g} GB exceeds usable profile VRAM "
            f"{profile['usable_vram_gb']:g} GB"
        )
    return {
        "profile_id": profile["profile_id"],
        "fits": not reasons,
        "reasons": reasons,
        "usable_vram_gb": profile["usable_vram_gb"],
        "default_rate_cad_per_hour": profile["default_rate_cad_per_hour"],
    }
