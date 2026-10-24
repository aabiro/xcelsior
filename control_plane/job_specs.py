"""Immutable admission snapshots, distinct from mutable job observations.

The launch hash binds the logical workload approved by a customer. The
execution hash additionally binds server-selected execution constraints. Both
are retained: an egress decision or serverless identity must never be lost by
reducing the execution command to the customer-facing launch schema.

This flat execution format remains consumable by the fenced worker protocol.
Per-attempt authority and volatile status are deliberately supplied separately.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

EXECUTION_SPEC_VERSION = "v1"

# Explicit contract: adding an observation to payload never turns it into an
# execution input. Include the worker's storage/runtime inputs as well as the
# scheduler's placement and billing constraints.
EXECUTION_FIELDS = frozenset("""name vram_needed_gb num_gpus gpu_model region image docker_image
    interactive command ssh_port pricing_mode tier nfs_server nfs_path
    nfs_mount_point volume_ids encrypted_workspace init_script git_repo
    auto_launch exposed_ports source_template_id owner owner_id tenant_id team_id
    egress exclusive_gpu job_type serverless_worker_id serverless_endpoint_id
    environment volume_mounts http_port health_check_path volumes require_nfs
    encrypted_volumes require_image_signature image_signature_key required_runtime
    preferred_host_id placement_preference max_price_per_hour spot_rate_cad
    spot preemptible scheduler serverless_billing_quote""".split())


def canonical_spec_hash(spec: dict[str, Any]) -> str:
    """Wire-compatible hash, independently recomputed by the worker."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def make_execution_spec(
    job: dict[str, Any],
    *,
    launch_spec: dict[str, Any] | None = None,
    admission_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from control_plane.launch.canonicalize import canonicalize

    if launch_spec is None:
        launch_spec = canonicalize(
            {
                **job,
                "interactive": bool(job.get("interactive", False)),
                "template_image_id": job.get("source_template_id") or "",
            }
        )
    if canonicalize(launch_spec) != launch_spec:
        raise ValueError("admission requires a supported canonical launch spec")
    spec = {key: value for key, value in job.items() if key in EXECUTION_FIELDS}
    if spec.get("placement_preference"):
        from control_plane.scheduler.native_preference import normalize_preference

        spec["placement_preference"] = normalize_preference(spec["placement_preference"])
    spec.update(
        execution_spec_version=EXECUTION_SPEC_VERSION,
        launch_spec=launch_spec,
        launch_spec_hash=canonical_spec_hash(launch_spec),
    )
    if admission_request is not None:
        if admission_request.get("launch_spec") != launch_spec:
            raise ValueError("admission request does not match its approved launch spec")
        spec["admission_request"] = admission_request
    # Reject non-JSON/nonfinite execution inputs, and detach caller-owned
    # nested dictionaries so an environment update cannot mutate admission.
    return json.loads(json.dumps(spec, allow_nan=False))


def validate_execution_spec(
    spec: Any, expected_hash: str | None, *, require_admission: bool = False
) -> dict[str, Any]:
    """Refuse corrupt stored authority instead of falling back to payload."""
    if (
        not isinstance(spec, dict)
        or not expected_hash
        or canonical_spec_hash(spec) != expected_hash
    ):
        raise ValueError("stored execution spec failed its hash check")
    if require_admission and spec.get("execution_spec_version") != EXECUTION_SPEC_VERSION:
        raise ValueError("execution requires a native admission snapshot")
    if "execution_spec_version" in spec:
        if spec["execution_spec_version"] != EXECUTION_SPEC_VERSION:
            raise ValueError("unsupported execution spec version")
        launch = spec.get("launch_spec")
        if not isinstance(launch, dict) or canonical_spec_hash(launch) != spec.get(
            "launch_spec_hash"
        ):
            raise ValueError("stored launch spec failed its hash check")
        request = spec.get("admission_request")
        if request is not None and (
            not isinstance(request, dict) or request.get("launch_spec") != launch
        ):
            raise ValueError("stored admission request disagrees with its launch spec")
    return spec
