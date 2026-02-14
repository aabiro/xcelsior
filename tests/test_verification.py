"""Tests for Xcelsior host verification lifecycle — checks, scoring, de/verification."""

import os
import tempfile
import time

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_verif_test_")
_tmpdir = _tmp_ctx.name
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


def _store() -> VerificationStore:
    return VerificationStore(
        db_path=os.path.join(_tmpdir, f"verif_{os.urandom(4).hex()}.db")
    )


def _good_report() -> dict:
    """A host report that passes all checks."""
    return {
        "claimed_gpu_model": "NVIDIA RTX 4090",
        "gpu_model": "NVIDIA RTX 4090",
        "claimed_vram_gb": 24,
        "total_vram_gb": 24,
        "cuda_version": "12.4",
        "driver_version": "550.54",
        "compute_capability": 8.9,
        "pcie_bandwidth_gbps": 15.0,
        "gpu_temp_celsius": 65,
        "packet_loss_pct": 0.1,
        "jitter_ms": 5,
        "throughput_mbps": 500,
    }


# ── Individual Checks ────────────────────────────────────────────────


class TestGPUIdentity:
    """GPU identity — model match and VRAM ≥95%."""

    def test_matching_gpu_passes(self):
        r = check_gpu_identity(_good_report())
        assert r.passed is True

    def test_mismatched_model_fails(self):
        report = _good_report()
        report["gpu_model"] = "NVIDIA RTX 3090"
        r = check_gpu_identity(report)
        assert r.passed is False

    def test_vram_below_95pct_fails(self):
        report = _good_report()
        report["total_vram_gb"] = 20  # 83% of 24
        r = check_gpu_identity(report)
        assert r.passed is False

    def test_empty_model_fails(self):
        report = _good_report()
        report["claimed_gpu_model"] = ""
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

    def test_missing_cuda_fails(self):
        report = _good_report()
        report["cuda_version"] = ""
        r = check_cuda_readiness(report)
        assert r.passed is False


class TestPCIeBandwidth:
    """PCIe bandwidth minimum threshold."""

    def test_above_threshold_passes(self):
        r = check_pcie_bandwidth(_good_report())
        assert r.passed is True

    def test_below_threshold_fails(self):
        report = _good_report()
        report["pcie_bandwidth_gbps"] = 2.0
        r = check_pcie_bandwidth(report)
        assert r.passed is False


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
        hv = HostVerification(host_id="h-store-1", state=HostVerificationState.VERIFIED)
        store.save_verification(hv)
        got = store.get_verification("h-store-1")
        assert got is not None
        assert got.state == HostVerificationState.VERIFIED

    def test_get_missing_returns_none(self):
        store = _store()
        assert store.get_verification("nonexistent") is None

    def test_list_verified_hosts(self):
        store = _store()
        store.save_verification(HostVerification(host_id="h-v1", state=HostVerificationState.VERIFIED))
        store.save_verification(HostVerification(host_id="h-v2", state=HostVerificationState.DEVERIFIED))
        verified = store.list_verified_hosts()
        assert "h-v1" in verified
        assert "h-v2" not in verified

    def test_record_job_failure(self):
        store = _store()
        store.save_verification(HostVerification(host_id="h-fail", state=HostVerificationState.VERIFIED))
        store.record_job_failure("h-fail", "j-fail-1")
        # Verify failure was recorded (get_recent_failures counts them)
        count = store.get_recent_failures("h-fail")
        assert count >= 1


# ── Verification Engine ──────────────────────────────────────────────


class TestVerificationEngine:
    """Full verification engine — orchestrates checks."""

    def test_run_verification_all_pass(self):
        store = _store()
        engine = VerificationEngine(store=store)
        result = engine.run_verification("h-eng-1", _good_report())
        assert result.state == HostVerificationState.VERIFIED
        assert result.overall_score > 80

    def test_run_verification_failing_check(self):
        store = _store()
        engine = VerificationEngine(store=store)
        report = _good_report()
        report["gpu_model"] = "FAKE GPU"
        result = engine.run_verification("h-eng-fail", report)
        # Should still produce a result but with lower score or deverified
        assert result is not None
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
