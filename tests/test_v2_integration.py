"""Integration tests for new v2 features: security, bin-packing, privacy, event snapshots."""

import os
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")


class TestSecurityModule:
    """Test security hardening features."""

    def test_version_gating_old_runc(self):
        """Old runc version should be rejected."""
        from security import check_node_versions

        admitted, reasons = check_node_versions({"runc": "1.0.0"})
        assert admitted is False
        assert any("CVE-2024-21626" in r for r in reasons)

    def test_version_gating_safe_runc(self):
        """Patched runc should be admitted."""
        from security import check_node_versions

        admitted, reasons = check_node_versions({"runc": "1.1.14"})
        assert admitted is True
        assert len(reasons) == 0

    def test_mig_detection_structure(self):
        """MIG detection should return expected dict structure."""
        from security import detect_mig_capability

        result = detect_mig_capability()
        assert "mig_capable" in result
        assert "mig_enabled" in result
        assert "mig_partitions" in result

    def test_select_runtime_consumer_gpu(self):
        """Consumer GPUs should get gVisor runtime."""
        from security import select_runtime_for_host

        runtime, reason = select_runtime_for_host("RTX 4090")
        assert runtime == "runsc"

    def test_select_runtime_mig_enabled(self):
        """MIG-enabled hosts should get runc (gVisor incompatible)."""
        from security import select_runtime_for_host

        runtime, reason = select_runtime_for_host("A100", mig_info={"mig_enabled": True})
        assert runtime == "runc"
        assert "MIG" in reason

    def test_secrets_encryption_roundtrip(self):
        """Encrypt → decrypt should return original value."""
        from security import encrypt_secret, decrypt_secret

        original = "sk-test-secret-key-12345"
        encrypted = encrypt_secret(original)
        assert encrypted != original
        decrypted = decrypt_secret(encrypted)
        assert decrypted == original

    def test_secrets_different_values_different_ciphertext(self):
        """Different plaintext → different ciphertext."""
        from security import encrypt_secret

        e1 = encrypt_secret("secret1")
        e2 = encrypt_secret("secret2")
        assert e1 != e2

    def test_mining_detection_high_util_low_pcie(self):
        """Mining signature: high GPU util + low PCIe TX."""
        from security import check_mining_heuristic

        is_mining, confidence, reason = check_mining_heuristic(
            {
                "utilization": 98,
                "memory_util": 95,
                "pcie_tx_mb_s": 1,
            }
        )
        assert is_mining is True
        assert confidence > 0.5

    def test_mining_detection_normal_usage(self):
        """Normal GPU usage should not trigger mining flag."""
        from security import check_mining_heuristic

        is_mining, confidence, reason = check_mining_heuristic(
            {
                "utilization": 60,
                "memory_util": 40,
                "pcie_tx_mb_s": 100,
            }
        )
        assert is_mining is False

    def test_container_bandwidth_commands(self):
        """Bandwidth limit commands should include tc qdisc."""
        from security import build_bandwidth_limit_commands

        cmds = build_bandwidth_limit_commands("test-container", mbps=50)
        assert any("tc qdisc" in c for c in cmds)

    def test_audit_container_has_method(self):
        from security import audit_container_security

        assert callable(audit_container_security)


class TestBinPackScheduler:
    """Test best-fit bin packing allocator."""

    def test_binpack_prefers_tighter_fit(self):
        """Bin packer should prefer host with less waste."""
        from scheduler import allocate_binpack

        hosts = [
            {
                "host_id": "big",
                "free_vram_gb": 80,
                "total_vram_gb": 80,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "A100",
                "cost_per_hour": 2.0,
            },
            {
                "host_id": "small",
                "free_vram_gb": 16,
                "total_vram_gb": 16,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "RTX 4060",
                "cost_per_hour": 0.3,
            },
        ]
        job = {"name": "test-job", "vram_needed_gb": 14, "num_gpus": 1}
        best = allocate_binpack(job, hosts)
        # Should pick "small" — tighter fit (14/16 vs 14/80)
        assert best["host_id"] == "small"

    def test_binpack_multi_gpu_requires_single_host(self):
        """Multi-GPU jobs require all GPUs on one host."""
        from scheduler import allocate_binpack

        hosts = [
            {
                "host_id": "single",
                "free_vram_gb": 48,
                "total_vram_gb": 48,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "A100",
            },
            {
                "host_id": "multi",
                "free_vram_gb": 160,
                "total_vram_gb": 160,
                "gpu_count": 4,
                "admitted": True,
                "gpu_model": "A100",
            },
        ]
        job = {"name": "gang-job", "vram_needed_gb": 40, "num_gpus": 4}
        best = allocate_binpack(job, hosts)
        assert best is not None
        assert best["host_id"] == "multi"

    def test_binpack_no_fit_returns_none(self):
        """If no host fits, return None."""
        from scheduler import allocate_binpack

        hosts = [
            {
                "host_id": "tiny",
                "free_vram_gb": 4,
                "total_vram_gb": 8,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "RTX 3060",
            },
        ]
        job = {"name": "huge-job", "vram_needed_gb": 80, "num_gpus": 1}
        assert allocate_binpack(job, hosts) is None

    def test_binpack_skips_draining_hosts(self):
        """Draining hosts should not receive new placements."""
        from scheduler import allocate_binpack

        hosts = [
            {
                "host_id": "draining",
                "free_vram_gb": 80,
                "total_vram_gb": 80,
                "gpu_count": 1,
                "admitted": True,
                "status": "draining",
                "gpu_model": "A100",
                "cost_per_hour": 1.0,
            },
            {
                "host_id": "active",
                "free_vram_gb": 40,
                "total_vram_gb": 40,
                "gpu_count": 1,
                "admitted": True,
                "status": "active",
                "gpu_model": "RTX 4090",
                "cost_per_hour": 0.8,
            },
        ]
        job = {"name": "test-job", "vram_needed_gb": 20, "num_gpus": 1}
        best = allocate_binpack(job, hosts)
        assert best is not None
        assert best["host_id"] == "active"

    def test_binpack_locality_bonus(self):
        """Volume-attached host should be preferred."""
        from scheduler import allocate_binpack

        hosts = [
            {
                "host_id": "remote",
                "free_vram_gb": 16,
                "total_vram_gb": 16,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "RTX 4090",
                "cost_per_hour": 0.5,
                "province": "BC",
                "country": "CA",
            },
            {
                "host_id": "local",
                "free_vram_gb": 16,
                "total_vram_gb": 16,
                "gpu_count": 1,
                "admitted": True,
                "gpu_model": "RTX 4090",
                "cost_per_hour": 0.5,
                "province": "ON",
                "country": "CA",
            },
        ]
        job = {"name": "data-job", "vram_needed_gb": 14, "num_gpus": 1}
        best = allocate_binpack(job, hosts, user_province="ON", volume_host_ids={"local"})
        # Should prefer "local" due to same province + volume attachment
        assert best["host_id"] == "local"


class TestPrivacyFeatures:
    """Test privacy module additions: crypto shredding, CASL."""

    def test_crypto_shredder_class_exists(self):
        from privacy import CryptoShredder, get_crypto_shredder

        assert CryptoShredder is not None

    def test_consent_manager_class_exists(self):
        from privacy import ConsentManager, get_consent_manager

        assert ConsentManager is not None

    def test_right_to_erasure_function_exists(self):
        from privacy import execute_right_to_erasure

        assert callable(execute_right_to_erasure)

    def test_retention_policies_complete(self):
        """All data categories should have retention policies."""
        from privacy import RETENTION_POLICIES, DataCategory

        for cat in DataCategory:
            assert cat in RETENTION_POLICIES, f"Missing retention policy for {cat}"

    def test_redact_pii_emails(self):
        """Email addresses should be redacted."""
        from privacy import redact_pii

        result = redact_pii("Contact john@example.com for details")
        assert "john@example.com" not in result
        assert "[REDACTED" in result or "***" in result


class TestEventSnapshots:
    """Test event snapshot manager."""

    def test_snapshot_manager_exists(self):
        from events import get_snapshot_manager

        sm = get_snapshot_manager()
        assert sm is not None
        assert hasattr(sm, "create_snapshot")
        assert hasattr(sm, "get_latest_snapshot")
        assert hasattr(sm, "snapshot_all_jobs")
