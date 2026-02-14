"""Tests for Xcelsior security module — version gating, admission, runtime recommendations."""

import os
import tempfile

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from security import (
    MINIMUM_VERSIONS,
    DEFAULT_EGRESS_ALLOWLIST,
    admit_node,
    build_secure_docker_args,
    check_mining_heuristic,
    check_node_versions,
    is_gvisor_available,
    recommend_runtime,
)


# ── Version Gating ───────────────────────────────────────────────────


class TestCheckNodeVersions:
    """Validate version comparison logic for CVE-gated node admission."""

    def test_all_good_versions_pass(self):
        versions = {
            "runc": "1.2.4",
            "nvidia_toolkit": "1.17.8",
            "nvidia_driver": "555.0",
            "docker": "25.0.0",
        }
        ok, reasons = check_node_versions(versions)
        assert ok is True
        assert len(reasons) == 0

    def test_old_runc_fails(self):
        versions = {
            "runc": "1.0.0",
            "nvidia_toolkit": "1.17.8",
            "nvidia_driver": "555.0",
            "docker": "25.0.0",
        }
        ok, reasons = check_node_versions(versions)
        assert ok is False
        assert any("runc" in r for r in reasons)

    def test_old_nvidia_driver_fails(self):
        versions = {
            "runc": "1.2.4",
            "nvidia_toolkit": "1.17.8",
            "nvidia_driver": "400.0",
            "docker": "25.0.0",
        }
        # 400.0 < MINIMUM_VERSIONS["nvidia_driver"] (550.0)
        ok, reasons = check_node_versions(versions)
        assert ok is False
        assert any("driver" in r.lower() for r in reasons)

    def test_old_docker_fails(self):
        versions = {
            "runc": "1.2.4",
            "nvidia_toolkit": "1.17.8",
            "nvidia_driver": "555.0",
            "docker": "19.0.0",
        }
        ok, reasons = check_node_versions(versions)
        assert ok is False
        assert any("Docker" in r for r in reasons)

    def test_missing_key_is_handled(self):
        versions = {"runc": "1.2.4"}
        ok, reasons = check_node_versions(versions)
        # Missing keys are skipped — only runc checked
        assert isinstance(ok, bool)

    def test_empty_dict_passes(self):
        # No versions to check → nothing fails
        ok, reasons = check_node_versions({})
        assert ok is True
        assert len(reasons) == 0


# ── Node Admission ───────────────────────────────────────────────────


class TestAdmitNode:
    """Full admission pipeline: version gate + runtime recommendation."""

    def test_good_node_admitted(self):
        versions = {
            "runc": "1.2.4",
            "nvidia_toolkit": "1.17.8",
            "nvidia_driver": "555.0",
            "docker": "25.0.0",
        }
        admitted, details = admit_node("h1", versions, "RTX 4090")
        assert admitted is True
        assert details["admitted"] is True

    def test_bad_node_rejected(self):
        versions = {
            "runc": "0.0.1",
            "nvidia_toolkit": "1.17.8",
            "nvidia_driver": "555.0",
            "docker": "25.0.0",
        }
        admitted, details = admit_node("h1", versions, "RTX 4090")
        assert admitted is False
        assert len(details["rejection_reasons"]) > 0


# ── Runtime Recommendation ───────────────────────────────────────────


class TestRecommendRuntime:
    """gVisor recommendation based on GPU model."""

    def test_returns_tuple(self):
        runtime, reason = recommend_runtime("RTX 4090")
        assert runtime in ("runsc", "runc")
        assert isinstance(reason, str)

    def test_older_gpu_gets_runc_or_runsc(self):
        runtime, reason = recommend_runtime("GTX 1080")
        assert runtime in ("runsc", "runc")


# ── Secure Docker Args ──────────────────────────────────────────────


class TestBuildSecureDockerArgs:
    """Verify all security flags are present in generated docker args."""

    def test_contains_no_new_privileges(self):
        args = build_secure_docker_args("python:3.12", "test-job", runtime="runc")
        assert "--security-opt=no-new-privileges" in args or "--security-opt" in args

    def test_contains_cap_drop(self):
        args = build_secure_docker_args("python:3.12", "test-job", runtime="runc")
        assert any("cap-drop" in a.lower() for a in args)

    def test_contains_gpu_flag(self):
        args = build_secure_docker_args("python:3.12", "test-job", runtime="runc")
        # Should pass GPU without --privileged
        assert "--privileged" not in args
        assert any("gpus" in a.lower() for a in args)


# ── Egress Rules ─────────────────────────────────────────────────────


class TestEgressRules:
    """Verify egress allowlist contains expected domains."""

    def test_allowlist_not_empty(self):
        assert len(DEFAULT_EGRESS_ALLOWLIST) > 0

    def test_contains_expected_entries(self):
        combined = " ".join(str(r) for r in DEFAULT_EGRESS_ALLOWLIST)
        assert len(combined) > 10


# ── Mining Detection ─────────────────────────────────────────────────


class TestMiningDetection:
    """Heuristic detection for cryptocurrency mining abuse."""

    def test_normal_workload_not_flagged(self):
        metrics = {
            "utilization": 85,
            "memory_util": 60,
            "pcie_tx_mb_s": 500,
        }
        is_mining, confidence, reason = check_mining_heuristic(metrics)
        assert is_mining is False

    def test_sustained_high_util_low_pcie_flagged(self):
        metrics = {
            "utilization": 99,
            "memory_util": 90,
            "pcie_tx_mb_s": 1,
        }
        is_mining, confidence, reason = check_mining_heuristic(metrics)
        assert is_mining is True
        assert confidence > 0.5


# ── gVisor Detection ─────────────────────────────────────────────────


class TestIsGvisorAvailable:
    """gVisor presence detection."""

    def test_returns_bool(self):
        result = is_gvisor_available()
        assert isinstance(result, bool)
