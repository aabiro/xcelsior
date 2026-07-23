"""Deterministic, versioned canonicalization of a launch request (§14).

The output of :func:`canonicalize` is *the* canonical spec: the same input,
however its keys are ordered or its strings are padded, produces the same
dict, and an omitted optional field is indistinguishable from one passed at
its default. That determinism is what makes :func:`spec_hash` a stable
binding — the scheduler stamps it on the attempt and the worker recomputes
it before it starts a container (Track A P5.5). The hash function itself is
reused from the scheduler (:func:`control_plane.scheduler.reservation.
canonical_spec_hash`) so there is exactly one control-plane implementation;
the worker keeps an independent copy that a drift guard pins to this one.

Why a separate canonical spec and not just "the job payload": the payload
carries volatile, per-submission fields (``job_id``, ``submitted_at``,
``owner``) that must *not* enter the hash, or two identical launches would
never share a spec hash and the binding would bind nothing logical. The
canonical spec is only the fields that define *what* is being run.
"""

from __future__ import annotations

from typing import Any

from control_plane.scheduler.reservation import canonical_spec_hash

# Bump when the canonical field set or normalization changes in a way that
# alters the hash. Stored on the action plan so a spec quoted under one
# version is never executed as though it were another.
CANON_SPEC_VERSION = "v1"

# Auto-launch services and the container ports they require (mirrors the
# JobIn model so /instance and /api/v1/launch-plans agree — B2.6).
_AUTO_LAUNCH_PORTS = {"jupyter": 8888, "vscode": 8443}
_ALLOWED_AUTO_LAUNCH = ("jupyter", "vscode")


def _s(value: Any) -> str:
    """Normalize to a stripped string; None and non-strings become ''."""
    if value is None:
        return ""
    return str(value).strip()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_region(raw: str) -> str:
    if not raw:
        return ""
    try:
        from host_metadata import normalize_region

        return normalize_region(raw, default="")
    except Exception:
        return raw


def _normalize_pricing_mode(raw: str) -> str:
    mode = (raw or "on_demand").strip().lower()
    return mode if mode in ("on_demand", "spot", "reserved") else "on_demand"


def _sorted_unique_ports(value: Any) -> list[int]:
    """Ports are a *set*: order does not matter, so sort for determinism."""
    if not value:
        return []
    out: set[int] = set()
    for p in value:
        n = _int(p, 0)
        if 1 <= n <= 65535:
            out.add(n)
    return sorted(out)


def _sorted_unique_strs(value: Any) -> list[str]:
    if not value:
        return []
    return sorted({_s(v) for v in value if _s(v)})


def _auto_launch(value: Any) -> list[str]:
    """Preserve request order for auto-launch (a sequence), deduped/normalized."""
    if not value:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        s = _s(item).lower()
        if s in _ALLOWED_AUTO_LAUNCH and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def canonicalize(request: dict[str, Any]) -> dict[str, Any]:
    """Turn a raw launch request into the canonical spec dict.

    ``request`` uses the same field names as the REST ``JobIn`` model. Every
    optional field is defaulted here so that omitting it and passing its
    default hash identically. Collections that are logically *sets*
    (``exposed_ports``, ``volume_ids``) are sorted; the auto-launch list is a
    sequence and keeps request order (deduped). Auto-launch ports are merged
    in exactly as the JobIn model does, so the two surfaces agree.
    """
    auto_launch = _auto_launch(request.get("auto_launch"))

    ports = set(_sorted_unique_ports(request.get("exposed_ports")))
    for svc in auto_launch:
        p = _AUTO_LAUNCH_PORTS.get(svc)
        if p is not None:
            ports.add(p)
    exposed_ports = sorted(ports)

    pricing_mode = _normalize_pricing_mode(_s(request.get("pricing_mode")))

    # Interactive instances get exclusive GPU access (one per host), so a
    # fractional VRAM reservation is meaningless — the REST /instance path
    # forces vram_needed=0 for them, and canonicalization must apply the same
    # defaulting or the two surfaces would produce different specs (B2.6).
    interactive = bool(request.get("interactive", True))
    vram_needed = 0.0 if interactive else max(0.0, float(request.get("vram_needed_gb") or 0))

    spec: dict[str, Any] = {
        "spec_version": CANON_SPEC_VERSION,
        "name": _s(request.get("name")),
        "vram_needed_gb": vram_needed,
        "num_gpus": max(1, _int(request.get("num_gpus"), 1)),
        "gpu_model": _s(request.get("gpu_model")),
        "region": _normalize_region(_s(request.get("region"))),
        "image": _s(request.get("image")),
        "interactive": interactive,
        "command": _s(request.get("command")),
        "ssh_port": _int(request.get("ssh_port"), 22),
        "pricing_mode": pricing_mode,
        "tier": _s(request.get("tier")),
        "nfs_server": _s(request.get("nfs_server")),
        "nfs_path": _s(request.get("nfs_path")),
        "nfs_mount_point": _s(request.get("nfs_mount_point")),
        "volume_ids": _sorted_unique_strs(request.get("volume_ids")),
        "encrypted_workspace": bool(request.get("encrypted_workspace", False)),
        "init_script": _s(request.get("init_script")),
        "git_repo": _s(request.get("git_repo")),
        "auto_launch": auto_launch,
        "exposed_ports": exposed_ports,
        "template_image_id": _s(request.get("template_image_id")),
    }
    return spec


def spec_hash(spec: dict[str, Any]) -> str:
    """The canonical spec hash — reused from the scheduler (one authority)."""
    return canonical_spec_hash(spec)
