# Xcelsior Host Verification & Deverification Lifecycle
#
# Host states: unverified → verifying → verified → deverified → verifying (loop)
#
# Seven checks, every one of which must PASS for a host to be verified
# (`VERIFICATION_CHECKS` is the list and `verification_check_count()` the count
# that public copy reads):
#   1. gpu_identity      — measured GPU model and VRAM against the host's
#                          listing (two-sided, within 5%), and the GPU UUIDs /
#                          serials against the baseline recorded at the host's
#                          first full pass
#   2. cuda_readiness    — CUDA, driver and compute capability present
#   3. pcie_bandwidth    — measured host<->device copy rate
#   4. thermal_stability — peak temperature under the benchmark's load and,
#                          from xcelsior-bench, how much of the sustained-load
#                          window the GPU spent thermal- or power-fault throttled
#   5. network_quality   — loss, jitter and throughput to the scheduler
#   6. allocatable_vram  — the largest single allocation that succeeds, against
#                          the listing
#   7. security_posture  — runc, NVIDIA Container Toolkit, driver and Docker
#                          versions against their CVE minimums
#
# A check reads `pass`, `fail` or `unknown`. **Unknown is not a pass**: a
# temperature of 0 or none, a missing version, a missing GPU UUID, a missing
# measurement — each makes the check `unknown`, and an unknown check blocks
# `verified` exactly as a failure does. The one reading that is recorded as
# unavailable without blocking is the board serial: consumer cards do not expose
# one through NVML, so for them the GPU UUID is the identity, and the record says
# `serial: unavailable` rather than claiming a serial was checked.
#
# Deverification triggers:
#   - a failed or unknown check at (re-)verification
#   - GPU identity differs from the recorded baseline (holds until an admin
#     clears it)
#   - hardware fingerprint drift: driver, CUDA, model or VRAM changed since the
#     last check (re-verifies on the next passing check)
#   - repeated host-fault job failures inside the window (holds for
#     `job_failure_hold_sec`)
#   - admin rejection (holds until an admin clears it)
#
# Freshness: a verified stamp is current until `next_check_at` plus
# `reverify_grace_sec`. Past that it reads `overdue` everywhere it is shown, and
# the placement gate treats an overdue host as not verified.

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from host_metadata import normalize_gpu_model
from xcelsior_bench import (
    ALLOCATABLE_FAULT_RATIO,
    MIN_SUSTAINED_SECONDS,
    THROTTLE_FAULT_SHARE,
    sustained_too_short,
)

log = logging.getLogger("xcelsior")


# ── Host Verification States ─────────────────────────────────────────


class HostVerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    DEVERIFIED = "deverified"


#: What a verified stamp reads once it is past `next_check_at` plus the grace.
#: Not a stored state: it is computed at read time by `verification_freshness`.
OVERDUE = "overdue"

# Check outcomes. `passed` on a result is True only for CHECK_PASS.
CHECK_PASS = "pass"
CHECK_FAIL = "fail"
CHECK_UNKNOWN = "unknown"

#: Why a de-verified host stays de-verified when a later benchmark passes.
HOLD_IDENTITY_MISMATCH = "identity_mismatch"
HOLD_ADMIN_REJECTED = "admin_rejected"
HOLD_JOB_FAILURES = "job_failures"

#: Prefix on fingerprints that include the GPU UUIDs. A stored fingerprint
#: without it was computed before the agent sent any identity, so comparing the
#: two would report drift on every host the first time it upgrades.
FINGERPRINT_VERSION = "v2:"


# ── Benchmark / Verification Thresholds ──────────────────────────────

VERIFICATION_THRESHOLDS = {
    "min_pcie_bandwidth_gbps": 8.0,  # Minimum PCIe bandwidth in GB/s
    "max_gpu_temp_celsius": 90,  # Max peak GPU temp during the benchmark load
    # Share of xcelsior-bench's sustained window spent thermal- or power-fault
    # throttled. The benchmark's own verdict uses the same number.
    "max_throttle_share_pct": THROTTLE_FAULT_SHARE * 100,
    "max_vram_deviation_pct": 5.0,  # |measured - advertised| / advertised
    "min_allocatable_vram_pct": ALLOCATABLE_FAULT_RATIO * 100,  # Largest allocation vs listing
    "min_cuda_compute_capability": 7.0,  # Minimum CUDA compute capability
    "max_network_loss_pct": 2.0,  # Max acceptable packet loss
    "max_network_jitter_ms": 50.0,  # Max acceptable jitter
    "min_network_throughput_mbps": 100.0,  # Min network throughput
    "job_failure_window_sec": 3600,  # Window for counting host-fault failures
    "job_failure_threshold": 3,  # Host-fault failures in window → deverify
    "job_failure_hold_sec": 86400,  # How long a failure deverify holds
    "reverify_interval_sec": 86400,  # Re-verify every 24 hours
    "reverify_grace_sec": 21600,  # Overdue 6 h after next_check_at
    "deverify_cooldown_sec": 1800,  # 30 min cooldown before re-verify
}

#: Component versions the security check requires the agent to report.
REQUIRED_VERSION_KEYS = ("runc", "nvidia_toolkit", "nvidia_driver", "docker")


@dataclass
class VerificationResult:
    """Result of a host verification check.

    `status` is the outcome (`pass`, `fail`, `unknown`); `passed` is kept for
    readers that predate it and is True only when `status` is `pass`.
    """

    check_name: str
    passed: bool
    expected: str = ""
    actual: str = ""
    details: str = ""
    timestamp: float = field(default_factory=time.time)
    status: str = ""

    def __post_init__(self):
        if not self.status:
            self.status = CHECK_PASS if self.passed else CHECK_FAIL
        self.passed = self.status == CHECK_PASS

    def to_dict(self) -> dict:
        return asdict(self)


def _unknown(check_name: str, expected: str, actual: str, details: str = "") -> VerificationResult:
    return VerificationResult(
        check_name=check_name,
        passed=False,
        expected=expected,
        actual=actual,
        details=details,
        status=CHECK_UNKNOWN,
    )


def check_status(check: dict) -> str:
    """The outcome of a stored check, reading records written before `status`."""
    status = check.get("status")
    if status in (CHECK_PASS, CHECK_FAIL, CHECK_UNKNOWN):
        return status
    return CHECK_PASS if check.get("passed") else CHECK_FAIL


@dataclass
class HostVerification:
    """Complete verification record for a host."""

    verification_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    host_id: str = ""
    state: str = HostVerificationState.UNVERIFIED
    checks: list = field(default_factory=list)
    verified_at: Optional[float] = None
    deverified_at: Optional[float] = None
    deverify_reason: str = ""
    last_check_at: Optional[float] = None
    next_check_at: Optional[float] = None
    failure_count: int = 0
    gpu_fingerprint: str = ""  # SHA256 of GPU identity for drift detection
    overall_score: float = 0.0  # 0-100 verification score
    #: GPU UUIDs and serials accepted at the first fully passing verification.
    identity_baseline: dict = field(default_factory=dict)
    #: Non-empty while a de-verified host must stay de-verified.
    hold_reason: str = ""
    #: When the hold lapses; None with a hold_reason means until an admin clears it.
    hold_until: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def hold_active(self, now: Optional[float] = None) -> bool:
        if not self.hold_reason:
            return False
        if self.hold_until is None:
            return True
        return (now if now is not None else time.time()) < float(self.hold_until)


# ── Freshness ────────────────────────────────────────────────────────


def verification_freshness(
    state: Any,
    last_check_at: Any = None,
    next_check_at: Any = None,
    *,
    now: Optional[float] = None,
) -> str:
    """`verified`, `overdue`, or the stored state — the one rule for a stamp's age.

    A stored `verified` means "passed when last checked". It is current until
    `next_check_at` plus `reverify_grace_sec`; after that it is `overdue`, which
    the placement gate, the trust page, the host page and the MCP tools all read
    from here. A verified row with no schedule falls back to `last_check_at` plus
    the interval, and with neither it is overdue: an unstamped host is neither
    trusted nor ignored.
    """
    s = str(getattr(state, "value", state) or "").strip().lower()
    if s != HostVerificationState.VERIFIED.value:
        return s or HostVerificationState.UNVERIFIED.value
    due: Optional[float] = None
    try:
        if next_check_at is not None:
            due = float(next_check_at)
        elif last_check_at is not None:
            due = float(last_check_at) + float(VERIFICATION_THRESHOLDS["reverify_interval_sec"])
    except (TypeError, ValueError):
        due = None
    if due is None:
        return OVERDUE
    current = now if now is not None else time.time()
    grace = float(VERIFICATION_THRESHOLDS["reverify_grace_sec"])
    return HostVerificationState.VERIFIED.value if current <= due + grace else OVERDUE


def overdue_after(last_check_at: Any = None, next_check_at: Any = None) -> Optional[float]:
    """The moment a verified stamp becomes overdue, or None when it has no schedule."""
    try:
        if next_check_at is not None:
            due = float(next_check_at)
        elif last_check_at is not None:
            due = float(last_check_at) + float(VERIFICATION_THRESHOLDS["reverify_interval_sec"])
        else:
            return None
    except (TypeError, ValueError):
        return None
    return due + float(VERIFICATION_THRESHOLDS["reverify_grace_sec"])


# ── GPU identity ──────────────────────────────────────────────────────


def reported_identity(report: dict) -> dict:
    """The GPU identity a report carries: sorted UUIDs and serials, and their status.

    The agent sends `gpu_identities` (one entry per GPU). A report from an agent
    that predates it may carry a top-level `gpu_uuid` / `serial` for GPU 0, or
    nothing. A serial equal to a UUID is not a serial: an older NVML helper fell
    back to the UUID when the board serial was unavailable.
    """
    entries = report.get("gpu_identities")
    if not isinstance(entries, list) or not entries:
        entries = [{"gpu_uuid": report.get("gpu_uuid"), "serial": report.get("serial")}]
    uuids: list[str] = []
    serials: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        gpu_uuid = str(entry.get("gpu_uuid") or entry.get("uuid") or "").strip()
        serial = str(entry.get("serial") or "").strip()
        if serial.upper() in ("[N/A]", "N/A", "NONE", "0") or serial.startswith("GPU-"):
            serial = ""
        if gpu_uuid:
            uuids.append(gpu_uuid)
        if serial:
            serials.append(serial)
    uuids.sort()
    serials.sort()
    if not serials:
        serial_status = "unavailable"
    elif len(serials) == len(uuids):
        serial_status = "ok"
    else:
        serial_status = "partial"
    return {"gpu_uuids": uuids, "serials": serials, "serial_status": serial_status}


def identity_mismatch(baseline: dict, identity: dict) -> str:
    """Why `identity` differs from `baseline`, or "" when it matches.

    UUIDs always compare. Serials compare only when both sides have them: a
    card that exposes no serial cannot be held to one, and a serial appearing
    where none was readable is not evidence of a different card when the UUIDs
    agree.
    """
    base_uuids = list(baseline.get("gpu_uuids") or [])
    if not base_uuids:
        return ""
    now_uuids = list(identity.get("gpu_uuids") or [])
    if sorted(base_uuids) != sorted(now_uuids):
        return f"GPU UUIDs changed: baseline {sorted(base_uuids)}, reported {sorted(now_uuids)}"
    base_serials = sorted(baseline.get("serials") or [])
    now_serials = sorted(identity.get("serials") or [])
    if base_serials and now_serials and base_serials != now_serials:
        return f"GPU serials changed: baseline {base_serials}, reported {now_serials}"
    return ""


# ── Verification Checks ──────────────────────────────────────────────
# Each check takes a host_report dict (the agent's measurements, plus the
# `advertised_*` fields the engine reads from the host's listing) and returns a
# VerificationResult.


def _num(report: dict, key: str) -> Optional[float]:
    raw = report.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _canonical_model(model: str) -> str:
    return normalize_gpu_model(model).lower().replace(" ", "")


def check_gpu_identity(report: dict) -> VerificationResult:
    """Measured model and VRAM against the listing; the GPU UUID must be present.

    "Advertised" is what the host's listing says (`advertised_gpu_model`,
    `advertised_vram_gb`), filled in by `VerificationEngine` from the hosts table
    and the host's available offers. It is never the agent's own reading: the
    old check compared the agent's NVML name with PyTorch on the same machine,
    which proves nothing about what a renter was sold.

    VRAM is two-sided: over-reporting is as wrong as under-reporting.

    The UUID baseline comparison happens in the engine, which has the stored
    record; here a missing UUID makes the check `unknown`.
    """
    advertised_model = str(report.get("advertised_gpu_model") or "")
    measured_model = str(report.get("gpu_model") or "")
    advertised_vram = _num(report, "advertised_vram_gb")
    measured_vram = _num(report, "total_vram_gb")
    identity = reported_identity(report)
    max_dev = float(VERIFICATION_THRESHOLDS["max_vram_deviation_pct"])

    expected = (
        f"{advertised_model or '?'} / {advertised_vram if advertised_vram else '?'}GB (listing)"
    )
    actual = f"{measured_model or '?'} / {measured_vram if measured_vram else '?'}GB"
    serial_detail = (
        f"serial: {identity['serial_status']}"
        if identity["serial_status"] != "ok"
        else f"serial: {', '.join(s[:16] for s in identity['serials'])}"
    )
    uuid_detail = (
        f"UUID: {', '.join(identity['gpu_uuids'])}" if identity["gpu_uuids"] else "UUID: missing"
    )

    if not advertised_model or not advertised_vram:
        return _unknown(
            "gpu_identity",
            expected,
            actual,
            f"no listing to compare against; {uuid_detail}; {serial_detail}",
        )
    if not measured_model or not measured_vram:
        return _unknown(
            "gpu_identity", expected, actual, f"measurement missing; {uuid_detail}; {serial_detail}"
        )

    model_match = _canonical_model(advertised_model) == _canonical_model(measured_model)
    deviation = abs(measured_vram - advertised_vram) / advertised_vram * 100
    vram_ok = deviation <= max_dev
    problems = []
    if not model_match:
        problems.append(f"model {measured_model!r} is not the listed {advertised_model!r}")
    if not vram_ok:
        problems.append(f"VRAM differs from the listing by {deviation:.1f}% (max {max_dev:g}%)")
    for offer in report.get("advertised_offers") or []:
        offer_model = str(offer.get("gpu_model") or "")
        offer_vram = _num(offer, "vram_gb")
        if offer_model and _canonical_model(offer_model) != _canonical_model(measured_model):
            problems.append(f"offer {offer.get('offer_id')} lists {offer_model!r}")
        if offer_vram and abs(measured_vram - offer_vram) / offer_vram * 100 > max_dev:
            problems.append(f"offer {offer.get('offer_id')} lists {offer_vram:g}GB")
    count = _num(report, "advertised_gpu_count")
    if count and report.get("gpu_identities") and len(identity["gpu_uuids"]) != int(count):
        problems.append(f"{len(identity['gpu_uuids'])} GPUs reported, {int(count)} listed")

    details = f"VRAM deviation {deviation:.1f}%; {uuid_detail}; {serial_detail}"
    if problems:
        return VerificationResult(
            check_name="gpu_identity",
            passed=False,
            expected=expected,
            actual=actual,
            details="; ".join(problems) + f" ({details})",
            status=CHECK_FAIL,
        )
    if not identity["gpu_uuids"]:
        return _unknown(
            "gpu_identity",
            expected,
            actual,
            f"model and VRAM match the listing but the agent sent no GPU UUID; {details}",
        )
    return VerificationResult(
        check_name="gpu_identity",
        passed=True,
        expected=expected,
        actual=actual,
        details=details,
    )


def check_cuda_readiness(report: dict) -> VerificationResult:
    """Verify CUDA and driver are functional."""
    cuda_version = str(report.get("cuda_version") or "")
    driver_version = str(report.get("driver_version") or "")
    compute_capability = _num(report, "compute_capability")
    min_cc = VERIFICATION_THRESHOLDS["min_cuda_compute_capability"]
    expected = f"CUDA present, compute capability >= {min_cc}"
    actual = f"CUDA {cuda_version or '?'}, driver {driver_version or '?'}, CC {compute_capability}"

    if not cuda_version or not driver_version or compute_capability is None:
        return _unknown("cuda_readiness", expected, actual, "CUDA, driver or CC not reported")
    return VerificationResult(
        check_name="cuda_readiness",
        passed=compute_capability >= min_cc,
        expected=expected,
        actual=actual,
    )


def check_pcie_bandwidth(report: dict) -> VerificationResult:
    """Verify PCIe bandwidth meets minimum threshold."""
    bandwidth = _num(report, "pcie_bandwidth_gbps")
    threshold = VERIFICATION_THRESHOLDS["min_pcie_bandwidth_gbps"]
    if not bandwidth:
        return _unknown(
            "pcie_bandwidth",
            f">= {threshold} GB/s",
            "not measured",
            str(report.get("pcie_error") or "no PCIe measurement in the report"),
        )
    return VerificationResult(
        check_name="pcie_bandwidth",
        passed=bandwidth >= threshold,
        expected=f">= {threshold} GB/s",
        actual=f"{bandwidth:.2f} GB/s",
    )


def sustained_load(report: dict) -> dict | None:
    """The xcelsior-bench sustained-load summary in a report, if the agent ran it."""
    bench = report.get("bench")
    sustained = bench.get("sustained") if isinstance(bench, dict) else None
    return sustained if isinstance(sustained, dict) else None


def check_thermal_stability(report: dict) -> VerificationResult:
    """Peak GPU temperature under load, and throttling in the sustained window.

    A reading of 0 or none is a sensor that did not answer, not a cool GPU. A
    report from xcelsior-bench also carries the sustained-load phase: more than
    ``max_throttle_share_pct`` of it thermal- or power-fault throttled fails,
    and a sustained phase with no throttle samples is unknown. A report from an
    agent older than the benchmark has no sustained phase and is judged on its
    peak temperature alone.
    """
    gpu_temp = _num(report, "gpu_temp_celsius")
    max_temp = VERIFICATION_THRESHOLDS["max_gpu_temp_celsius"]
    if not gpu_temp or gpu_temp <= 0 or report.get("gpu_temp_samples") == 0:
        return _unknown(
            "thermal_stability",
            f"<= {max_temp}°C",
            "no reading",
            str(report.get("thermal_error") or "temperature sensor returned no reading"),
        )
    sustained = sustained_load(report)
    if sustained is None:
        return VerificationResult(
            check_name="thermal_stability",
            passed=gpu_temp <= max_temp,
            expected=f"<= {max_temp}°C",
            actual=f"{gpu_temp}°C",
        )
    limit = float(VERIFICATION_THRESHOLDS["max_throttle_share_pct"])
    expected = f"<= {max_temp}°C, throttled <= {limit:g}% of the sustained window"
    thermal = sustained.get("thermal_throttle_share")
    power = sustained.get("power_fault_share")
    if thermal is None:
        return _unknown(
            "thermal_stability",
            expected,
            f"{gpu_temp}°C peak, no throttle samples",
            "the sustained-load phase took no throttle readings",
        )
    power = float(power or 0)
    passed = gpu_temp <= max_temp and thermal * 100 <= limit and power * 100 <= limit
    actual = (
        f"{gpu_temp}°C peak; thermal throttling {thermal * 100:.1f}%, "
        f"power faults {power * 100:.1f}% of {sustained.get('duration_s', '?')}s"
    )
    # A fault seen in a short run is still a fault; a clean short run proves nothing.
    if passed and sustained_too_short(sustained):
        return _unknown(
            "thermal_stability",
            f"{expected}, over at least {MIN_SUSTAINED_SECONDS}s",
            actual,
            f"the sustained-load phase ran {sustained.get('duration_s', '?')}s with "
            f"{sustained.get('samples', 0)} samples; it needs {MIN_SUSTAINED_SECONDS}s",
        )
    return VerificationResult(
        check_name="thermal_stability",
        passed=passed,
        expected=expected,
        actual=actual,
    )


def check_network_quality(report: dict) -> VerificationResult:
    """Verify network meets quality thresholds."""
    loss = _num(report, "packet_loss_pct")
    jitter = _num(report, "jitter_ms")
    throughput = _num(report, "throughput_mbps")
    expected = (
        f"loss <= {VERIFICATION_THRESHOLDS['max_network_loss_pct']}%, "
        f"jitter <= {VERIFICATION_THRESHOLDS['max_network_jitter_ms']}ms, "
        f"throughput >= {VERIFICATION_THRESHOLDS['min_network_throughput_mbps']} Mbps"
    )
    if loss is None or jitter is None or throughput is None:
        return _unknown("network_quality", expected, "not measured", "network benchmark missing")

    loss_ok = loss <= VERIFICATION_THRESHOLDS["max_network_loss_pct"]
    jitter_ok = jitter <= VERIFICATION_THRESHOLDS["max_network_jitter_ms"]
    throughput_ok = throughput >= VERIFICATION_THRESHOLDS["min_network_throughput_mbps"]

    return VerificationResult(
        check_name="network_quality",
        passed=loss_ok and jitter_ok and throughput_ok,
        expected=expected,
        actual=f"loss={loss}%, jitter={jitter}ms, throughput={throughput} Mbps",
    )


def check_allocatable_vram(report: dict) -> VerificationResult:
    """The largest single allocation that succeeds, against the listed VRAM.

    Replaces the old memory_fragmentation check, which read PyTorch allocator
    fields the agent never sent and so was skipped — and counted as a pass — on
    every host. This one measures what a renter's job can actually take.
    """
    allocatable = _num(report, "allocatable_vram_gb")
    advertised = _num(report, "advertised_vram_gb")
    floor_pct = float(VERIFICATION_THRESHOLDS["min_allocatable_vram_pct"])
    expected = f">= {floor_pct:g}% of the listed VRAM"
    if not allocatable or not advertised:
        return _unknown(
            "allocatable_vram",
            expected,
            f"{allocatable or '?'}GB allocatable / {advertised or '?'}GB listed",
            str(report.get("allocatable_vram_error") or "no allocation probe or no listing"),
        )
    pct = allocatable / advertised * 100
    return VerificationResult(
        check_name="allocatable_vram",
        passed=pct >= floor_pct,
        expected=expected,
        actual=f"{allocatable:.2f}GB of {advertised:g}GB ({pct:.1f}%)",
    )


def check_security_posture(report: dict) -> VerificationResult:
    """Component versions against their CVE minimums. A missing version is unknown."""
    from security import check_node_versions

    versions = report.get("versions") or {}
    if not isinstance(versions, dict):
        versions = {}
    admitted, reasons = check_node_versions(versions)
    missing = [k for k in REQUIRED_VERSION_KEYS if not str(versions.get(k) or "").strip()]
    expected = "runc, NVIDIA Container Toolkit, driver and Docker at or above their minimums"
    if not admitted:
        return VerificationResult(
            check_name="security_posture",
            passed=False,
            expected=expected,
            actual="; ".join(reasons),
            details=(f"not reported: {', '.join(missing)}" if missing else ""),
        )
    if missing:
        return _unknown(
            "security_posture",
            expected,
            f"not reported: {', '.join(missing)}",
            "a version the host did not report cannot be checked against its minimum",
        )
    return VerificationResult(
        check_name="security_posture",
        passed=True,
        expected=expected,
        actual=", ".join(f"{k} {versions[k]}" for k in REQUIRED_VERSION_KEYS),
    )


#: Every check, in order. Its length is the count public copy quotes
#: ("7 checks"), read through `verification_check_count()` and release facts.
VERIFICATION_CHECKS = [
    check_gpu_identity,
    check_cuda_readiness,
    check_pcie_bandwidth,
    check_thermal_stability,
    check_network_quality,
    check_allocatable_vram,
    check_security_posture,
]


def verification_check_count() -> int:
    return len(VERIFICATION_CHECKS)


# ── Verification Store ────────────────────────────────────────────────


def _jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class VerificationStore:
    """Persistent store for host verification state and history."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path  # Legacy compat

    @contextmanager
    def _conn(self):
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _from_row(row: Any) -> HostVerification:
        return HostVerification(
            verification_id=row["verification_id"],
            host_id=row["host_id"],
            state=row["state"],
            verified_at=row["verified_at"],
            deverified_at=row["deverified_at"],
            deverify_reason=row["deverify_reason"],
            last_check_at=row["last_check_at"],
            next_check_at=row["next_check_at"],
            failure_count=row["failure_count"],
            gpu_fingerprint=row["gpu_fingerprint"],
            overall_score=row["overall_score"],
            checks=_jsonish(row["checks"], []),
            identity_baseline=_jsonish(row.get("identity_baseline"), {}),
            hold_reason=row.get("hold_reason") or "",
            hold_until=row.get("hold_until"),
        )

    def get_verification(self, host_id: str) -> Optional[HostVerification]:
        """Get current verification state for a host."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM host_verifications WHERE host_id = %s",
                (host_id,),
            ).fetchone()
            if not row:
                return None
            return self._from_row(row)

    def save_verification(self, v: HostVerification):
        """Upsert verification state."""
        from psycopg.types.json import Jsonb

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO host_verifications
                   (host_id, verification_id, state, verified_at,
                    deverified_at, deverify_reason, last_check_at,
                    next_check_at, failure_count, gpu_fingerprint,
                    overall_score, checks, updated_at,
                    identity_baseline, hold_reason, hold_until)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s)
                   ON CONFLICT(host_id) DO UPDATE SET
                    verification_id=excluded.verification_id,
                    state=excluded.state,
                    verified_at=excluded.verified_at,
                    deverified_at=excluded.deverified_at,
                    deverify_reason=excluded.deverify_reason,
                    last_check_at=excluded.last_check_at,
                    next_check_at=excluded.next_check_at,
                    failure_count=excluded.failure_count,
                    gpu_fingerprint=excluded.gpu_fingerprint,
                    overall_score=excluded.overall_score,
                    checks=excluded.checks,
                    updated_at=excluded.updated_at,
                    identity_baseline=excluded.identity_baseline,
                    hold_reason=excluded.hold_reason,
                    hold_until=excluded.hold_until""",
                (
                    v.host_id,
                    v.verification_id,
                    getattr(v.state, "value", v.state),
                    v.verified_at,
                    v.deverified_at,
                    v.deverify_reason,
                    v.last_check_at,
                    v.next_check_at,
                    v.failure_count,
                    v.gpu_fingerprint,
                    v.overall_score,
                    Jsonb(v.checks),
                    time.time(),
                    Jsonb(v.identity_baseline or {}),
                    v.hold_reason or "",
                    v.hold_until,
                ),
            )
            # History entry
            conn.execute(
                """INSERT INTO verification_history
                   (host_id, verification_id, state, checks, score, reason, timestamp)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    v.host_id,
                    v.verification_id,
                    getattr(v.state, "value", v.state),
                    Jsonb(v.checks),
                    v.overall_score,
                    v.deverify_reason,
                    time.time(),
                ),
            )

    def record_benchmark(self, host_id: str, verification_id: str, report: dict) -> bool:
        """Keep an xcelsior-bench run as history in `benchmarks` (167).

        Evidence, not state: a failure to write it is logged and never fails
        the verification it came with. The verdict is judged here, against the
        VRAM the listing advertises (the agent does not know it), with the same
        ``xcelsior_bench.verdict`` a renter's own run uses.
        """
        bench = report.get("bench")
        if not isinstance(bench, dict) or not bench.get("bench_version"):
            return False
        from psycopg.types.json import Jsonb
        from xcelsior_bench import verdict as bench_verdict

        judged = bench_verdict(
            bench.get("sustained") if isinstance(bench.get("sustained"), dict) else None,
            _num(bench, "allocatable_vram_gb"),
            _num(report, "advertised_vram_gb"),
        )
        bench = {**bench, "verdict": judged, "advertised_vram_gb": report.get("advertised_vram_gb")}

        tflops = _num(bench, "fp16_tflops")
        identities = report.get("gpu_identities") or []
        gpu_uuid = str(
            report.get("gpu_uuid")
            or bench.get("gpu_uuid")
            or (
                identities[0].get("gpu_uuid")
                if identities and isinstance(identities[0], dict)
                else ""
            )
            or ""
        )
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO benchmarks
                       (host_id, gpu_model, score, tflops, memory_bandwidth_gbps,
                        benchmark_type, details, run_at, gpu_uuid, bench_version,
                        verification_id, verdict)
                       VALUES (%s, %s, %s, %s, %s, 'xcelsior-bench', %s, %s, %s, %s, %s, %s)""",
                    (
                        host_id,
                        str(bench.get("gpu_model") or report.get("gpu_model") or ""),
                        (tflops or 0) / 10,
                        tflops,
                        _num(bench, "memory_bandwidth_gbps"),
                        Jsonb(bench),
                        float(bench.get("finished_at") or time.time()),
                        gpu_uuid,
                        str(bench.get("bench_version")),
                        verification_id,
                        judged["result"],
                    ),
                )
            return True
        except Exception as exc:
            log.warning("benchmark history write failed for %s: %s", host_id, exc)
            return False

    def get_advertised_hardware(self, host_id: str) -> Optional[dict]:
        """What the host's listing says it is: the hosts row and its available offers.

        This is the "claimed" side of the identity check. None when the host has
        no record — there is then nothing a renter was sold to compare against.
        """
        with self._conn() as conn:
            host = conn.execute(
                """SELECT payload->>'gpu_model' AS gpu_model,
                          payload->>'total_vram_gb' AS total_vram_gb,
                          payload->>'num_gpus' AS num_gpus
                     FROM hosts WHERE host_id = %s""",
                (host_id,),
            ).fetchone()
            if not host:
                return None
            offers = conn.execute(
                """SELECT offer_id, gpu_model, vram_gb, gpu_count_total
                     FROM gpu_offers WHERE host_id = %s AND available = TRUE
                    ORDER BY offer_id""",
                (host_id,),
            ).fetchall()

        def _f(raw: Any) -> Optional[float]:
            try:
                return float(raw) if raw not in (None, "") else None
            except (TypeError, ValueError):
                return None

        return {
            "gpu_model": host["gpu_model"] or "",
            "vram_gb": _f(host["total_vram_gb"]),
            "gpu_count": _f(host["num_gpus"]),
            "offers": [
                {
                    "offer_id": o["offer_id"],
                    "gpu_model": o["gpu_model"],
                    "vram_gb": _f(o["vram_gb"]),
                    "gpu_count_total": o["gpu_count_total"],
                }
                for o in offers
            ],
        }

    def record_job_failure(self, host_id: str, job_id: str, reason: str = ""):
        """Record a host-fault job failure for deverification tracking."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO job_failure_log (host_id, job_id, failed_at, reason) VALUES (%s, %s, %s, %s)",
                (host_id, job_id, time.time(), reason),
            )

    def get_recent_failures(
        self, host_id: str, window_sec: Optional[int] = None, now: Optional[float] = None
    ) -> int:
        """Count recent failures for a host within the window."""
        window = window_sec or VERIFICATION_THRESHOLDS["job_failure_window_sec"]
        cutoff = (now if now is not None else time.time()) - window
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT job_id) as cnt FROM job_failure_log "
                "WHERE host_id = %s AND failed_at >= %s",
                (host_id, cutoff),
            ).fetchone()
            return row["cnt"] if row else 0

    def list_verified_hosts(self) -> list[str]:
        """Get all host_ids with verified status."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT host_id FROM host_verifications WHERE state = 'verified'",
            ).fetchall()
            return [r["host_id"] for r in rows]

    def list_hosts_needing_reverification(self, now: Optional[float] = None) -> list[str]:
        """Hosts the sweep should ask to verify again.

        * **verified / verifying** — due when `next_check_at` has passed, and a
          missing schedule is due, not exempt: a host with no stamp would
          otherwise be refused by the placement gate forever and never asked.
        * **deverified** — due once its cooldown has passed (`next_check_at` is
          set to the end of the cooldown when it is de-verified), so a host
          that failed a check can recover by passing one. Before this a
          de-verified host was never asked again.
        * **unverified** — asked only while its host row is active: a
          registration whose agent never produced a report should be asked,
          an abandoned one should not be asked every hour forever.

        A host under an active hold is never due: a swapped card or an admin
        rejection must not be undone by a benchmark that happens to pass.
        """
        current = now if now is not None else time.time()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT hv.host_id FROM host_verifications hv
                    WHERE (next_check_at IS NULL OR next_check_at <= %(now)s)
                      AND NOT (COALESCE(hold_reason, '') <> ''
                               AND (hold_until IS NULL OR hold_until > %(now)s))
                      AND (state IN ('verified', 'verifying', 'deverified')
                           OR (state = 'unverified'
                               AND EXISTS (SELECT 1 FROM hosts h
                                            WHERE h.host_id = hv.host_id
                                              AND h.status = 'active')))
                    ORDER BY next_check_at NULLS FIRST, hv.host_id""",
                {"now": current},
            ).fetchall()
            return [r["host_id"] for r in rows]

    def get_verification_summary(self) -> dict:
        """Dashboard summary of verification states."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) as cnt FROM host_verifications GROUP BY state",
            ).fetchall()
            summary = {r["state"]: r["cnt"] for r in rows}
            summary["total"] = sum(summary.values())
            return summary


# ── Verification Engine ──────────────────────────────────────────────


def _fingerprint(report: dict, identity: dict) -> str:
    """Hardware fingerprint for drift detection: model, VRAM, driver, CUDA, UUIDs."""
    raw = (
        f"{report.get('gpu_model', '')}:{report.get('total_vram_gb', '')}:"
        f"{report.get('driver_version', '')}:{report.get('cuda_version', '')}:"
        f"{','.join(identity['gpu_uuids'])}:{','.join(identity['serials'])}"
    )
    return FINGERPRINT_VERSION + hashlib.sha256(raw.encode()).hexdigest()[:16]


class VerificationEngine:
    """Runs verification checks and manages host verification lifecycle.

    - Hosts begin unverified
    - Seven checks, all passing, promote to verified
    - A failed or unknown check, identity or fingerprint drift, repeated
      host-fault failures or an admin rejection deverify
    - Deverified hosts re-qualify after the cooldown, unless held
    """

    def __init__(self, store: Optional[VerificationStore] = None):
        self.store = store or VerificationStore()

    def _with_listing(self, host_id: str, report: dict) -> dict:
        """The report with `advertised_*` taken from the listing, never the agent.

        An agent could otherwise send its own `advertised_*` and certify itself.
        Its `claimed_*` fields are kept for the record and compared with nothing.
        """
        merged = {k: v for k, v in report.items() if not str(k).startswith("advertised_")}
        try:
            listing = self.store.get_advertised_hardware(host_id)
        except Exception as exc:
            log.warning("verification: listing for %s unreadable: %s", host_id, exc)
            listing = None
        if listing:
            merged["advertised_gpu_model"] = listing.get("gpu_model") or ""
            merged["advertised_vram_gb"] = listing.get("vram_gb")
            merged["advertised_gpu_count"] = listing.get("gpu_count")
            merged["advertised_offers"] = listing.get("offers") or []
        return merged

    def run_verification(self, host_id: str, report: dict) -> HostVerification:
        """Run all verification checks against a host telemetry report.

        Args:
            host_id: The host to verify
            report: The agent's measurements, including:
                - gpu_model, total_vram_gb, allocatable_vram_gb
                - gpu_identities: [{gpu_uuid, serial, pci_bus_id, ...}]
                - cuda_version, driver_version, compute_capability
                - pcie_bandwidth_gbps, gpu_temp_celsius
                - packet_loss_pct, jitter_ms, throughput_mbps
                - versions: {runc, nvidia_toolkit, nvidia_driver, docker}

        Returns:
            Updated HostVerification with check results
        """
        existing = self.store.get_verification(host_id)
        now = time.time()

        # Check cooldown for deverified hosts
        if existing and existing.state == HostVerificationState.DEVERIFIED:
            if existing.deverified_at:
                cooldown_end = (
                    existing.deverified_at + VERIFICATION_THRESHOLDS["deverify_cooldown_sec"]
                )
                if now < cooldown_end:
                    log.info(
                        "VERIFY COOLDOWN host=%s (%.0fs remaining)", host_id, cooldown_end - now
                    )
                    return existing

        report = self._with_listing(host_id, report)
        identity = reported_identity(report)

        results = []
        for check_fn in VERIFICATION_CHECKS:
            try:
                result = check_fn(report)
            except Exception as e:
                result = _unknown(
                    check_fn.__name__.removeprefix("check_"),
                    "",
                    "",
                    f"Check error: {e}",
                )
            results.append(result.to_dict())

        # The identity baseline is compared on every verification.
        baseline = dict(existing.identity_baseline) if existing else {}
        mismatch = identity_mismatch(baseline, identity) if baseline else ""
        if mismatch:
            for r in results:
                if r["check_name"] == "gpu_identity":
                    r.update(
                        passed=False,
                        status=CHECK_FAIL,
                        details=f"{mismatch}; {r.get('details', '')}".rstrip("; "),
                    )

        passed_count = sum(1 for r in results if check_status(r) == CHECK_PASS)
        total_count = len(results)
        score = (passed_count / total_count * 100) if total_count > 0 else 0
        all_passed = passed_count == total_count

        fingerprint = _fingerprint(report, identity)
        fingerprint_drift = bool(
            existing
            and existing.gpu_fingerprint.startswith(FINGERPRINT_VERSION)
            and existing.gpu_fingerprint != fingerprint
        )
        if fingerprint_drift and existing is not None:
            log.warning(
                "GPU DRIFT host=%s old=%s new=%s", host_id, existing.gpu_fingerprint, fingerprint
            )

        hold_reason = existing.hold_reason if existing and existing.hold_active(now) else ""
        hold_until = existing.hold_until if hold_reason and existing else None
        if mismatch:
            hold_reason, hold_until = HOLD_IDENTITY_MISMATCH, None

        if all_passed and not fingerprint_drift and not hold_reason:
            new_state = HostVerificationState.VERIFIED
        else:
            new_state = HostVerificationState.DEVERIFIED

        v = HostVerification(
            verification_id=existing.verification_id if existing else str(uuid.uuid4())[:12],
            host_id=host_id,
            state=new_state,
            checks=results,
            last_check_at=now,
            failure_count=existing.failure_count if existing else 0,
            gpu_fingerprint=fingerprint,
            overall_score=score,
            identity_baseline=baseline,
            hold_reason=hold_reason,
            hold_until=hold_until,
        )

        if new_state == HostVerificationState.VERIFIED:
            v.verified_at = now
            v.deverified_at = None
            v.deverify_reason = ""
            v.next_check_at = now + VERIFICATION_THRESHOLDS["reverify_interval_sec"]
            if not baseline.get("gpu_uuids"):
                v.identity_baseline = {
                    **identity,
                    "pci_bus_ids": sorted(
                        str(e.get("pci_bus_id") or "")
                        for e in (report.get("gpu_identities") or [])
                        if isinstance(e, dict) and e.get("pci_bus_id")
                    ),
                    "recorded_at": now,
                    "verification_id": v.verification_id,
                }
                log.info(
                    "VERIFY BASELINE host=%s uuids=%s serial=%s",
                    host_id,
                    identity["gpu_uuids"],
                    identity["serial_status"],
                )
            log.info("HOST VERIFIED host=%s score=%.0f fingerprint=%s", host_id, score, fingerprint)
        else:
            v.verified_at = existing.verified_at if existing else None
            v.deverified_at = now
            failed = [r["check_name"] for r in results if check_status(r) == CHECK_FAIL]
            unknown = [r["check_name"] for r in results if check_status(r) == CHECK_UNKNOWN]
            parts = []
            if failed:
                parts.append(f"Failed checks: {', '.join(failed)}")
            if unknown:
                parts.append(f"Unknown (not passed): {', '.join(unknown)}")
            if mismatch:
                parts.append(f"GPU identity changed ({mismatch}); held for admin review")
            elif hold_reason:
                parts.append(f"Held: {hold_reason}")
            if fingerprint_drift:
                parts.append("GPU fingerprint drift")
            v.deverify_reason = "; ".join(parts)
            # A held host is not swept; anyone else is asked again after the
            # cooldown, so a host that failed a check can recover by passing one.
            v.next_check_at = (
                hold_until
                if hold_reason
                else now + VERIFICATION_THRESHOLDS["deverify_cooldown_sec"]
            )
            log.warning(
                "HOST DEVERIFIED host=%s reason=%s score=%.0f", host_id, v.deverify_reason, score
            )

        self.store.save_verification(v)
        self.store.record_benchmark(host_id, v.verification_id, report)
        return v

    def check_failure_threshold(
        self, host_id: str, job_id: str, reason: str = "", now: Optional[float] = None
    ) -> bool:
        """Record a host-fault job failure and deverify at the threshold.

        Only host-fault failures belong here — `failure_fault.classify_failure`
        decides, and its callers pass nothing else. Distinct jobs are counted,
        so one job reported failed twice is one failure.

        Returns True if the host was deverified by this call.
        """
        current = now if now is not None else time.time()
        self.store.record_job_failure(host_id, job_id, reason)
        recent = self.store.get_recent_failures(host_id, now=current)
        threshold = VERIFICATION_THRESHOLDS["job_failure_threshold"]

        if recent >= threshold:
            existing = self.store.get_verification(host_id)
            if existing and existing.state in (
                HostVerificationState.VERIFIED,
                HostVerificationState.VERIFIED.value,
            ):
                hold = float(VERIFICATION_THRESHOLDS["job_failure_hold_sec"])
                existing.state = HostVerificationState.DEVERIFIED
                existing.deverified_at = current
                existing.deverify_reason = (
                    f"Exceeded failure threshold: {recent} host-fault job failures in "
                    f"{VERIFICATION_THRESHOLDS['job_failure_window_sec']}s window"
                )
                existing.failure_count = recent
                existing.hold_reason = HOLD_JOB_FAILURES
                existing.hold_until = current + hold
                existing.next_check_at = current + hold
                self.store.save_verification(existing)
                log.warning("HOST AUTO-DEVERIFIED host=%s failures=%d", host_id, recent)
                return True
        return False

    def get_verified_hosts(self) -> list[str]:
        """Get all currently verified host_ids."""
        return self.store.list_verified_hosts()

    def get_hosts_needing_reverification(self) -> list[str]:
        """Get hosts due for periodic re-verification."""
        return self.store.list_hosts_needing_reverification()

    def get_summary(self) -> dict:
        """Dashboard summary of verification states."""
        return self.store.get_verification_summary()


# ── Singleton ─────────────────────────────────────────────────────────

_verification_store: Optional[VerificationStore] = None
_verification_engine: Optional[VerificationEngine] = None


def get_verification_store() -> VerificationStore:
    global _verification_store
    if _verification_store is None:
        _verification_store = VerificationStore()
    return _verification_store


def get_verification_engine() -> VerificationEngine:
    global _verification_engine
    if _verification_engine is None:
        _verification_engine = VerificationEngine(get_verification_store())
    return _verification_engine
