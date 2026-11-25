"""Tests for Xcelsior host verification lifecycle — checks, scoring, de/verification."""

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from verification import (
    HostVerification,
    HostVerificationState,
    VerificationEngine,
    VerificationResult,
    VerificationStore,
    VERIFICATION_THRESHOLDS,
    check_cuda_readiness,
    check_gpu_identity,
    check_network_quality,
    check_pcie_bandwidth,
    check_thermal_stability,
)

LISTING = {"gpu_model": "NVIDIA RTX 4090", "vram_gb": 24, "gpu_count": 1, "offers": []}


class _ListedStore(VerificationStore):
    """The PostgreSQL store, with the host's listing supplied by the test."""

    def get_advertised_hardware(self, host_id: str):
        return LISTING


_HOSTS: list[str] = []


def _store() -> VerificationStore:
    return _ListedStore()


def _hid(prefix: str) -> str:
    host_id = f"{prefix}-{uuid.uuid4().hex[:8]}"
    _HOSTS.append(host_id)
    return host_id


@pytest.fixture(autouse=True, scope="module")
def _cleanup_hosts():
    yield
    store = VerificationStore()
    with store._conn() as conn:
        for table in ("host_verifications", "verification_history", "job_failure_log"):
            conn.execute(f"DELETE FROM {table} WHERE host_id = ANY(%s)", (_HOSTS,))


def _good_report() -> dict:
    """A host report that passes all checks.

    `advertised_*` stands in for the listing at the check level; the engine
    replaces it with the listing it reads, whatever the agent sends.
    """
    return {
        "advertised_gpu_model": "NVIDIA RTX 4090",
        "advertised_vram_gb": 24,
        "gpu_model": "NVIDIA RTX 4090",
        "total_vram_gb": 24,
        "allocatable_vram_gb": 23.2,
        "cuda_version": "12.4",
        "driver_version": "550.54",
        "compute_capability": 8.9,
        "pcie_bandwidth_gbps": 15.0,
        "gpu_temp_celsius": 65,
        "gpu_temp_samples": 30,
        "packet_loss_pct": 0.1,
        "jitter_ms": 5,
        "throughput_mbps": 500,
        "versions": {
            "runc": "1.4.0",
            "nvidia_toolkit": "1.20.2",
            "nvidia_driver": "580.178.04",
            "docker": "29.1.3",
        },
        "gpu_identities": [{"index": 0, "gpu_uuid": "GPU-1111-2222", "serial": "1650123456789"}],
    }


# ── Individual Checks ────────────────────────────────────────────────


class TestGPUIdentity:
    """GPU identity — the measured card against the listing, VRAM within 5% either way."""

    def test_matching_gpu_passes(self):
        r = check_gpu_identity(_good_report())
        assert r.passed is True

    def test_mismatched_model_fails(self):
        report = _good_report()
        report["gpu_model"] = "NVIDIA RTX 3090"
        r = check_gpu_identity(report)
        assert r.passed is False

    def test_canonical_claim_matches_raw_cuda_name(self):
        # What a real RTX 3060 host reports: the agent's canonical NVML name as the
        # claim, and the benchmark's raw torch device name as the actual model.
        report = _good_report()
        report["advertised_gpu_model"] = "RTX 3060"
        report["gpu_model"] = "NVIDIA GeForce RTX 3060"
        report["advertised_vram_gb"] = 12.0
        report["total_vram_gb"] = 11.6
        r = check_gpu_identity(report)
        assert r.passed is True

    def test_canonical_claim_still_rejects_a_different_card(self):
        report = _good_report()
        report["advertised_gpu_model"] = "RTX 4090"
        report["gpu_model"] = "NVIDIA GeForce RTX 3060"
        r = check_gpu_identity(report)
        assert r.passed is False

    def test_vram_below_95pct_fails(self):
        report = _good_report()
        report["total_vram_gb"] = 20  # 83% of 24
        r = check_gpu_identity(report)
        assert r.passed is False

    def test_vram_above_105pct_fails(self):
        report = _good_report()
        report["total_vram_gb"] = 48
        r = check_gpu_identity(report)
        assert r.passed is False

    def test_empty_model_fails(self):
        report = _good_report()
        report["advertised_gpu_model"] = ""
        report["gpu_model"] = ""
        r = check_gpu_identity(report)
        assert r.passed is False


class TestCUDAReadiness:
    """CUDA + driver + compute capability."""

    def test_good_cuda_passes(self):
        r = check_cuda_readiness(_good_report())
        assert r.passed is True

    def test_low_compute_capability_fails(self):
        report = _good_report()
        report["compute_capability"] = 5.0
        r = check_cuda_readiness(report)
        assert r.passed is False

    def test_missing_cuda_is_not_a_pass(self):
        report = _good_report()
        report["cuda_version"] = ""
        r = check_cuda_readiness(report)
        assert r.passed is False
        assert r.status == "unknown"


class TestPCIeBandwidth:
    """PCIe bandwidth minimum threshold."""

    def test_above_threshold_passes(self):
        r = check_pcie_bandwidth(_good_report())
        assert r.passed is True

    def test_below_threshold_fails(self):
        # A PCIe 3.0 x2 riser link measures about 1.6 GB/s.
        report = _good_report()
        report["pcie_bandwidth_gbps"] = 1.6
        r = check_pcie_bandwidth(report)
        assert r.passed is False

    def test_exactly_the_floor_passes(self):
        report = _good_report()
        report["pcie_bandwidth_gbps"] = VERIFICATION_THRESHOLDS["min_pcie_bandwidth_gbps"]
        r = check_pcie_bandwidth(report)
        assert r.passed is True


class TestThermalStability:
    """GPU temperature within acceptable range."""

    def test_within_limit_passes(self):
        r = check_thermal_stability(_good_report())
        assert r.passed is True

    def test_overheating_fails(self):
        report = _good_report()
        report["gpu_temp_celsius"] = 95
        r = check_thermal_stability(report)
        assert r.passed is False


class TestNetworkQuality:
    """Network loss, jitter, throughput."""

    def test_good_network_passes(self):
        r = check_network_quality(_good_report())
        assert r.passed is True

    def test_high_packet_loss_fails(self):
        report = _good_report()
        report["packet_loss_pct"] = 10
        r = check_network_quality(report)
        assert r.passed is False

    def test_high_jitter_fails(self):
        report = _good_report()
        report["jitter_ms"] = 200
        r = check_network_quality(report)
        assert r.passed is False

    def test_low_throughput_fails(self):
        report = _good_report()
        report["throughput_mbps"] = 10
        r = check_network_quality(report)
        assert r.passed is False


# ── Verification Store ────────────────────────────────────────────────


class TestVerificationStore:
    """Persistence — save, get, job failure tracking."""

    def test_save_and_get(self):
        store = _store()
        host_id = _hid("h-store")
        hv = HostVerification(host_id=host_id, state=HostVerificationState.VERIFIED)
        store.save_verification(hv)
        got = store.get_verification(host_id)
        assert got is not None
        assert got.state == HostVerificationState.VERIFIED

    def test_get_missing_returns_none(self):
        store = _store()
        assert store.get_verification("nonexistent") is None

    def test_list_verified_hosts(self):
        store = _store()
        v1, v2 = _hid("h-v1"), _hid("h-v2")
        store.save_verification(HostVerification(host_id=v1, state=HostVerificationState.VERIFIED))
        store.save_verification(
            HostVerification(host_id=v2, state=HostVerificationState.DEVERIFIED)
        )
        verified = store.list_verified_hosts()
        assert v1 in verified
        assert v2 not in verified

    def test_record_job_failure(self):
        store = _store()
        host_id = _hid("h-fail")
        store.save_verification(
            HostVerification(host_id=host_id, state=HostVerificationState.VERIFIED)
        )
        store.record_job_failure(host_id, "j-fail-1")
        assert store.get_recent_failures(host_id) == 1


# ── Verification Engine ──────────────────────────────────────────────


class TestVerificationEngine:
    """Full verification engine — orchestrates checks."""

    def test_run_verification_all_pass(self):
        store = _store()
        engine = VerificationEngine(store=store)
        result = engine.run_verification(_hid("h-eng"), _good_report())
        assert result.state == HostVerificationState.VERIFIED, result.deverify_reason
        assert result.overall_score == 100

    def test_run_verification_failing_check(self):
        store = _store()
        engine = VerificationEngine(store=store)
        report = _good_report()
        report["gpu_model"] = "FAKE GPU"
        result = engine.run_verification(_hid("h-eng-fail"), report)
        assert result.state == HostVerificationState.DEVERIFIED
        assert result.overall_score < 100

    def test_threshold_constants(self):
        assert VERIFICATION_THRESHOLDS["min_pcie_bandwidth_gbps"] > 0
        assert VERIFICATION_THRESHOLDS["job_failure_threshold"] >= 1


# ── VerificationResult dataclass ──────────────────────────────────────


class TestVerificationResult:
    """VerificationResult serialization."""

    def test_to_dict(self):
        r = VerificationResult(check_name="test", passed=True, expected="yes", actual="yes")
        d = r.to_dict()
        assert d["check_name"] == "test"
        assert d["passed"] is True
        assert d["status"] == "pass"

    def test_status_wins_over_passed(self):
        r = VerificationResult(
            check_name="t", passed=True, expected="", actual="", status="unknown"
        )
        assert r.passed is False
