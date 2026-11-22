"""Whose fault a failed job was, and what a host-fault failure does to the host.

Verification promises that repeated job failures de-verify a host. That only
means something if the failures counted are the host's: a renter's script that
exits 1, an image tag that does not exist, or a job that filled its own disk
quota says nothing about the machine, and counting them would let any renter
de-verify any host by launching three broken jobs on it.

So a failure is classified, and only `host` counts:

* **host** — the machine or its agent could not provide what the job needed:
  a volume or NFS mount that failed on the host, a missing container runtime,
  a GPU it could not hand over, registry credentials or disk the host lacks, a
  container that vanished, a state journal it cannot write.
* **customer** — the workload itself: a non-zero exit, a missing or private
  image, a failed signature check, a quota the job exceeded, mining.
* **unknown** — anything else, including control-plane conditions (a lease the
  host could not confirm, a job stuck before it started) and agent exceptions.
  Unknown is not counted: a control-plane outage fails jobs on every host at
  once, and counting it would de-verify the whole fleet for the platform's own
  fault.

Lost leases (a host that stopped reporting mid-job) are not counted here either,
for the same correlated-outage reason; they lower reputation and SLA instead.

The agent sends an explicit `host_*` failure code for host-side failures. Older
agents send `container_failed` with one of the worker's own fixed messages, so
those exact messages are recognised too.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("xcelsior")

HOST = "host"
CUSTOMER = "customer"
UNKNOWN = "unknown"

#: Failure codes the agent sends for failures on the host's side.
HOST_FAULT_CODES = frozenset(
    {
        "host_volume_mount_failed",
        "host_nfs_mount_failed",
        "host_workspace_failed",
        "host_runtime_unavailable",
        "host_gpu_unavailable",
        "host_registry_credentials",
        "host_disk_full",
        "container_missing",
        "journal_unavailable",
    }
)

#: Failure codes that describe the workload, not the host.
CUSTOMER_FAULT_CODES = frozenset(
    {
        "container_failed",
        "container_exited",
        "mining_detected",
        "disk_quota_exceeded",
        "image_not_found",
        "image_access_denied",
        "image_signature_failed",
    }
)

#: Messages the worker (before explicit codes) used for host-side failures,
#: matched as prefixes of the reported error. Only the worker's own fixed
#: strings, never free text from a container.
_HOST_FAULT_MESSAGES = (
    "required NFS mount failed",
    "required encrypted volume attach failed",
    "required volume mount failed",
    "encrypted workspace provisioning failed",
    "required gVisor runtime is unavailable",
    "exclusive_gpu acquire failed",
    "Image pull failed: this host could not read its own registry credentials",
    "Image pull failed: the host is out of disk space",
)


def _error_text(detail: Any) -> str:
    if isinstance(detail, dict):
        for key in ("error", "error_message", "failure_reason"):
            value = detail.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    return detail if isinstance(detail, str) else ""


def classify_failure(failure_code: Any, detail: Any = None) -> str:
    """`host`, `customer` or `unknown` for one failed attempt."""
    code = str(failure_code or "").strip()
    if code in HOST_FAULT_CODES:
        return HOST
    error = _error_text(detail)
    if error and any(error.startswith(prefix) for prefix in _HOST_FAULT_MESSAGES):
        return HOST
    if code in CUSTOMER_FAULT_CODES:
        return CUSTOMER
    return UNKNOWN


def record_failed_attempt(host_id: str, job_id: str, failure_code: Any, detail: Any = None) -> bool:
    """Count a failed attempt against its host when it was the host's fault.

    Returns True when this failure de-verified the host. Never raises: a
    failure to record must not turn a worker's status report into an error.
    """
    if not host_id or not job_id:
        return False
    fault = classify_failure(failure_code, detail)
    if fault != HOST:
        return False
    try:
        from verification import get_verification_engine

        reason = str(failure_code or "") or "host_fault"
        error = _error_text(detail)
        if error:
            reason = f"{reason}: {error[:200]}"
        return get_verification_engine().check_failure_threshold(host_id, job_id, reason)
    except Exception:
        log.exception("could not record host-fault failure host=%s job=%s", host_id, job_id)
        return False
