# Xcelsior Host Verification & Deverification Lifecycle
# Implements REPORT_FEATURE_FINAL.md: Automated verification + continuous
# monitoring + reversible status, modeled after Vast.ai's proven lifecycle.
#
# Host states: unverified → verifying → verified → deverified → verifying (loop)
#
# Verification checks:
#   1. GPU identity & VRAM (nvidia-smi)
#   2. CUDA + driver readiness
#   3. PCIe bandwidth sanity
#   4. Sustained thermal/power stability
#   5. Network throughput + loss/jitter
#   6. Container runtime + security posture
#
# Deverification triggers:
#   - Driver/GPU changes detected
#   - VRAM mismatch > threshold
#   - Repeated job failures (3+ in window)
#   - Network degradation beyond SLA
#   - Thermal throttling above threshold
#   - Failed periodic re-verification

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Host Verification States ─────────────────────────────────────────

class HostVerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    DEVERIFIED = "deverified"


# ── Benchmark / Verification Thresholds ──────────────────────────────

VERIFICATION_THRESHOLDS = {
    "min_pcie_bandwidth_gbps": 8.0,       # Minimum PCIe bandwidth in GB/s
    "max_gpu_temp_celsius": 90,           # Max sustained GPU temp
    "min_vram_match_pct": 95.0,           # VRAM must be >= 95% of claimed
    "min_cuda_compute_capability": 7.0,   # Minimum CUDA compute capability
    "max_network_loss_pct": 2.0,          # Max acceptable packet loss
    "max_network_jitter_ms": 50.0,        # Max acceptable jitter
    "min_network_throughput_mbps": 100.0,  # Min network throughput
    "job_failure_window_sec": 3600,       # Window for counting failures
    "job_failure_threshold": 3,           # Failures in window → deverify
    "reverify_interval_sec": 86400,       # Re-verify every 24 hours
    "deverify_cooldown_sec": 1800,        # 30 min cooldown before re-verify
}


@dataclass
class VerificationResult:
    """Result of a host verification check."""
    check_name: str
    passed: bool
    expected: str = ""
    actual: str = ""
    details: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


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
    gpu_fingerprint: str = ""           # SHA256 of GPU identity for drift detection
    overall_score: float = 0.0          # 0-100 verification score

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [c if isinstance(c, dict) else c for c in self.checks]
        return d


# ── Verification Checks ──────────────────────────────────────────────
# Each check takes a host_report dict (from agent telemetry) and returns
# a VerificationResult.

def check_gpu_identity(report: dict) -> VerificationResult:
    """Verify GPU identity, VRAM, and serial number match claimed values.

    Per REPORT_FEATURE_1.md §Automated Hardware Attestation:
    "GPU Model/Serial — NVML/PCI-ID check — Prevents spoofing lower-tier cards"
    "VRAM Capacity — nvmlDeviceGetMemoryInfo — Ensures model fit for large LLMs"
    """
    claimed_model = report.get("claimed_gpu_model", "")
    actual_model = report.get("gpu_model", "")
    claimed_vram = float(report.get("claimed_vram_gb", 0))
    actual_vram = float(report.get("total_vram_gb", 0))

    model_match = (
        claimed_model.lower().replace(" ", "") ==
        actual_model.lower().replace(" ", "")
    ) if claimed_model and actual_model else False

    vram_pct = (actual_vram / claimed_vram * 100) if claimed_vram > 0 else 0
    vram_ok = vram_pct >= VERIFICATION_THRESHOLDS["min_vram_match_pct"]

    # GPU serial / UUID check (REPORT_FEATURE_1.md §PCI-ID anti-spoofing)
    serial = report.get("serial", "")
    claimed_serial = report.get("claimed_serial", "")
    serial_ok = True
    serial_detail = ""
    if claimed_serial and serial:
        serial_ok = claimed_serial == serial
        serial_detail = f", Serial match: {serial_ok}"
    elif serial:
        serial_detail = f", Serial: {serial[:16]}..."

    # PCI bus ID check (additional anti-spoofing signal)
    pci_info = report.get("pci_info", {})
    pci_detail = ""
    if pci_info.get("bus_id"):
        pci_detail = f", PCI: {pci_info['bus_id']}"

    passed = model_match and vram_ok and serial_ok
    return VerificationResult(
        check_name="gpu_identity",
        passed=passed,
        expected=f"{claimed_model} / {claimed_vram}GB",
        actual=f"{actual_model} / {actual_vram}GB",
        details=f"Model match: {model_match}, VRAM: {vram_pct:.1f}%{serial_detail}{pci_detail}",
    )


def check_cuda_readiness(report: dict) -> VerificationResult:
    """Verify CUDA and driver are functional."""
    cuda_version = report.get("cuda_version", "")
    driver_version = report.get("driver_version", "")
    compute_capability = float(report.get("compute_capability", 0))

    min_cc = VERIFICATION_THRESHOLDS["min_cuda_compute_capability"]
    passed = bool(cuda_version) and bool(driver_version) and compute_capability >= min_cc

    return VerificationResult(
        check_name="cuda_readiness",
        passed=passed,
        expected=f"CUDA present, compute capability >= {min_cc}",
        actual=f"CUDA {cuda_version}, driver {driver_version}, CC {compute_capability}",
    )


def check_pcie_bandwidth(report: dict) -> VerificationResult:
    """Verify PCIe bandwidth meets minimum threshold."""
    bandwidth = float(report.get("pcie_bandwidth_gbps", 0))
    threshold = VERIFICATION_THRESHOLDS["min_pcie_bandwidth_gbps"]

    return VerificationResult(
        check_name="pcie_bandwidth",
        passed=bandwidth >= threshold,
        expected=f">= {threshold} GB/s",
        actual=f"{bandwidth:.2f} GB/s",
    )


def check_thermal_stability(report: dict) -> VerificationResult:
    """Verify GPU temps are within acceptable range."""
    gpu_temp = float(report.get("gpu_temp_celsius", 0))
    max_temp = VERIFICATION_THRESHOLDS["max_gpu_temp_celsius"]

    return VerificationResult(
        check_name="thermal_stability",
        passed=gpu_temp <= max_temp,
        expected=f"<= {max_temp}°C",
        actual=f"{gpu_temp}°C",
    )


def check_network_quality(report: dict) -> VerificationResult:
    """Verify network meets quality thresholds."""
    loss = float(report.get("packet_loss_pct", 100))
    jitter = float(report.get("jitter_ms", 999))
    throughput = float(report.get("throughput_mbps", 0))

    loss_ok = loss <= VERIFICATION_THRESHOLDS["max_network_loss_pct"]
    jitter_ok = jitter <= VERIFICATION_THRESHOLDS["max_network_jitter_ms"]
    throughput_ok = throughput >= VERIFICATION_THRESHOLDS["min_network_throughput_mbps"]

    passed = loss_ok and jitter_ok and throughput_ok
    return VerificationResult(
        check_name="network_quality",
        passed=passed,
        expected=(
            f"loss <= {VERIFICATION_THRESHOLDS['max_network_loss_pct']}%, "
            f"jitter <= {VERIFICATION_THRESHOLDS['max_network_jitter_ms']}ms, "
            f"throughput >= {VERIFICATION_THRESHOLDS['min_network_throughput_mbps']} Mbps"
        ),
        actual=f"loss={loss}%, jitter={jitter}ms, throughput={throughput} Mbps",
    )


def check_memory_fragmentation(report: dict) -> VerificationResult:
    """Check VRAM fragmentation from torch.cuda allocated vs reserved.

    Per REPORT_FEATURE_1.md: "track torch.cuda.memory_allocated() and
    torch.cuda.memory_reserved() because PyTorch's allocator can cause
    fragmentation where reserved memory remains locked."

    High fragmentation (>50%) indicates the host's PyTorch allocator
    is holding large amounts of unreachable memory, which can cause
    OOM failures despite apparently sufficient free VRAM.
    """
    allocated = report.get("torch_memory_allocated_bytes", 0)
    reserved = report.get("torch_memory_reserved_bytes", 0)
    fragmentation_pct = report.get("memory_fragmentation_pct", 0.0)

    # If no torch data available, pass by default
    if reserved == 0 and allocated == 0:
        return VerificationResult(
            check_name="memory_fragmentation",
            passed=True,
            expected="fragmentation < 50%",
            actual="no torch memory data (check skipped)",
        )

    max_frag = 50.0  # >50% fragmentation is a concern
    passed = fragmentation_pct < max_frag

    allocated_mb = round(allocated / (1024**2), 1)
    reserved_mb = round(reserved / (1024**2), 1)

    return VerificationResult(
        check_name="memory_fragmentation",
        passed=passed,
        expected=f"fragmentation < {max_frag}%",
        actual=f"{fragmentation_pct:.1f}% (allocated={allocated_mb}MB, reserved={reserved_mb}MB)",
    )


def check_security_posture(report: dict) -> VerificationResult:
    """Verify container runtime and security meet minimum requirements."""
    from security import check_node_versions
    versions = report.get("versions", {})
    admitted, reasons = check_node_versions(versions)

    return VerificationResult(
        check_name="security_posture",
        passed=admitted,
        expected="All component versions meet minimums",
        actual="; ".join(reasons) if reasons else "All checks passed",
    )


# All verification checks in order (7 checks per REPORT_FEATURE_1.md)
VERIFICATION_CHECKS = [
    check_gpu_identity,          # Model/serial/VRAM via NVML
    check_cuda_readiness,        # CUDA + driver + compute capability
    check_pcie_bandwidth,        # PCIe throughput sanity
    check_thermal_stability,     # GPU temp within limits
    check_network_quality,       # Loss, jitter, throughput
    check_memory_fragmentation,  # PyTorch allocated vs reserved
    check_security_posture,      # Container runtime + security
]


# ── Verification Store ────────────────────────────────────────────────

class VerificationStore:
    """Persistent store for host verification state and history."""

    def __init__(self, db_path: Optional[str] = None):
        import os
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "xcelsior_events.db"
        )
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS host_verifications (
                    host_id TEXT PRIMARY KEY,
                    verification_id TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'unverified',
                    verified_at REAL,
                    deverified_at REAL,
                    deverify_reason TEXT DEFAULT '',
                    last_check_at REAL,
                    next_check_at REAL,
                    failure_count INTEGER DEFAULT 0,
                    gpu_fingerprint TEXT DEFAULT '',
                    overall_score REAL DEFAULT 0.0,
                    checks TEXT DEFAULT '[]',
                    updated_at REAL
                );

                CREATE TABLE IF NOT EXISTS verification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    verification_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    checks TEXT DEFAULT '[]',
                    score REAL DEFAULT 0.0,
                    reason TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vh_host
                    ON verification_history(host_id);

                CREATE TABLE IF NOT EXISTS job_failure_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    failed_at REAL NOT NULL,
                    reason TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_jfl_host
                    ON job_failure_log(host_id);
                CREATE INDEX IF NOT EXISTS idx_jfl_time
                    ON job_failure_log(failed_at);
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_verification(self, host_id: str) -> Optional[HostVerification]:
        """Get current verification state for a host."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM host_verifications WHERE host_id = ?",
                (host_id,),
            ).fetchone()
            if not row:
                return None
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
                checks=json.loads(row["checks"]),
            )

    def save_verification(self, v: HostVerification):
        """Upsert verification state."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO host_verifications
                   (host_id, verification_id, state, verified_at,
                    deverified_at, deverify_reason, last_check_at,
                    next_check_at, failure_count, gpu_fingerprint,
                    overall_score, checks, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    updated_at=excluded.updated_at""",
                (
                    v.host_id, v.verification_id, v.state, v.verified_at,
                    v.deverified_at, v.deverify_reason, v.last_check_at,
                    v.next_check_at, v.failure_count, v.gpu_fingerprint,
                    v.overall_score, json.dumps(v.checks), time.time(),
                ),
            )
            # History entry
            conn.execute(
                """INSERT INTO verification_history
                   (host_id, verification_id, state, checks, score, reason, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    v.host_id, v.verification_id, v.state,
                    json.dumps(v.checks), v.overall_score,
                    v.deverify_reason, time.time(),
                ),
            )

    def record_job_failure(self, host_id: str, job_id: str, reason: str = ""):
        """Record a job failure for deverification tracking."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO job_failure_log (host_id, job_id, failed_at, reason) VALUES (?, ?, ?, ?)",
                (host_id, job_id, time.time(), reason),
            )

    def get_recent_failures(self, host_id: str,
                            window_sec: Optional[int] = None) -> int:
        """Count recent failures for a host within the window."""
        window = window_sec or VERIFICATION_THRESHOLDS["job_failure_window_sec"]
        cutoff = time.time() - window
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM job_failure_log WHERE host_id = ? AND failed_at >= ?",
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

    def list_hosts_needing_reverification(self) -> list[str]:
        """Get hosts that are due for periodic re-verification."""
        now = time.time()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT host_id FROM host_verifications
                   WHERE state = 'verified' AND next_check_at IS NOT NULL AND next_check_at <= ?""",
                (now,),
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

class VerificationEngine:
    """Runs verification checks and manages host verification lifecycle.

    Modeled after Vast.ai's lifecycle:
    - Hosts begin unverified
    - Automated checks promote to verified
    - Continuous monitoring can deverify
    - Deverified hosts can re-qualify after cooldown
    """

    def __init__(self, store: Optional[VerificationStore] = None):
        self.store = store or VerificationStore()

    def run_verification(self, host_id: str, report: dict) -> HostVerification:
        """Run all verification checks against a host telemetry report.

        Args:
            host_id: The host to verify
            report: Dict of telemetry data from the agent, including:
                - gpu_model, total_vram_gb, claimed_gpu_model, claimed_vram_gb
                - cuda_version, driver_version, compute_capability
                - pcie_bandwidth_gbps
                - gpu_temp_celsius
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
                cooldown_end = existing.deverified_at + VERIFICATION_THRESHOLDS["deverify_cooldown_sec"]
                if now < cooldown_end:
                    log.info("VERIFY COOLDOWN host=%s (%.0fs remaining)",
                             host_id, cooldown_end - now)
                    return existing

        # Run all checks
        results = []
        for check_fn in VERIFICATION_CHECKS:
            try:
                result = check_fn(report)
                results.append(result.to_dict())
            except Exception as e:
                results.append(VerificationResult(
                    check_name=check_fn.__name__,
                    passed=False,
                    details=f"Check error: {e}",
                ).to_dict())

        # Calculate overall score
        passed_count = sum(1 for r in results if r.get("passed"))
        total_count = len(results)
        score = (passed_count / total_count * 100) if total_count > 0 else 0

        # Compute GPU fingerprint for drift detection (includes serial for anti-spoofing)
        import hashlib
        fingerprint = hashlib.sha256(
            f"{report.get('gpu_model', '')}:{report.get('total_vram_gb', '')}:"
            f"{report.get('driver_version', '')}:{report.get('cuda_version', '')}:"
            f"{report.get('serial', '')}".encode()
        ).hexdigest()[:16]

        # Determine new state
        all_passed = all(r.get("passed") for r in results)

        # Check for GPU fingerprint drift (deverification trigger)
        fingerprint_drift = False
        if existing and existing.gpu_fingerprint and existing.gpu_fingerprint != fingerprint:
            fingerprint_drift = True
            log.warning("GPU DRIFT host=%s old=%s new=%s",
                        host_id, existing.gpu_fingerprint, fingerprint)

        if all_passed and not fingerprint_drift:
            new_state = HostVerificationState.VERIFIED
        else:
            new_state = HostVerificationState.DEVERIFIED

        # Build verification record
        v = HostVerification(
            verification_id=existing.verification_id if existing else str(uuid.uuid4())[:12],
            host_id=host_id,
            state=new_state,
            checks=results,
            last_check_at=now,
            next_check_at=now + VERIFICATION_THRESHOLDS["reverify_interval_sec"],
            failure_count=existing.failure_count if existing else 0,
            gpu_fingerprint=fingerprint,
            overall_score=score,
        )

        if new_state == HostVerificationState.VERIFIED:
            v.verified_at = now
            v.deverified_at = None
            v.deverify_reason = ""
            log.info("HOST VERIFIED host=%s score=%.0f fingerprint=%s",
                     host_id, score, fingerprint)
        else:
            v.deverified_at = now
            failed_checks = [r["check_name"] for r in results if not r.get("passed")]
            reason = f"Failed checks: {', '.join(failed_checks)}"
            if fingerprint_drift:
                reason += " + GPU fingerprint drift"
            v.deverify_reason = reason
            log.warning("HOST DEVERIFIED host=%s reason=%s score=%.0f",
                        host_id, reason, score)

        self.store.save_verification(v)
        return v

    def check_failure_threshold(self, host_id: str, job_id: str,
                                reason: str = "") -> bool:
        """Record a job failure and check if host should be deverified.

        Returns True if host was deverified due to exceeding failure threshold.
        """
        self.store.record_job_failure(host_id, job_id, reason)
        recent = self.store.get_recent_failures(host_id)
        threshold = VERIFICATION_THRESHOLDS["job_failure_threshold"]

        if recent >= threshold:
            existing = self.store.get_verification(host_id)
            if existing and existing.state == HostVerificationState.VERIFIED:
                existing.state = HostVerificationState.DEVERIFIED
                existing.deverified_at = time.time()
                existing.deverify_reason = (
                    f"Exceeded failure threshold: {recent} failures in "
                    f"{VERIFICATION_THRESHOLDS['job_failure_window_sec']}s window"
                )
                existing.failure_count = recent
                self.store.save_verification(existing)
                log.warning("HOST AUTO-DEVERIFIED host=%s failures=%d",
                            host_id, recent)
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
