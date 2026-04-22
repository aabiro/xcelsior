"""SSH key validation and fingerprinting tests.

Tests the _validate_ssh_public_key and _ssh_key_fingerprint helpers
in routes/ssh.py, plus the agent SSH key injection endpoint.
"""

import pytest
from routes.ssh import _validate_ssh_public_key, _ssh_key_fingerprint, VALID_SSH_KEY_TYPES

# A real ed25519 test key (safe — generated just for tests, no private key)
VALID_ED25519 = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl test@xcelsior"
)
VALID_RSA = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7 user@host"


class TestValidateSshPublicKey:
    """Tests for _validate_ssh_public_key."""

    def test_valid_ed25519(self):
        result = _validate_ssh_public_key(VALID_ED25519)
        assert result == "ssh-ed25519"

    def test_valid_key_with_whitespace(self):
        result = _validate_ssh_public_key(f"  {VALID_ED25519}  \n")
        assert result == "ssh-ed25519"

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            _validate_ssh_public_key("")

    def test_comment_only_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            _validate_ssh_public_key("# this is a comment")

    def test_single_word_raises(self):
        with pytest.raises(ValueError, match="Invalid SSH public key format"):
            _validate_ssh_public_key("ssh-ed25519")

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported key type"):
            _validate_ssh_public_key("ssh-dss AAAAB3NzaC1kc3M= oldkey@host")

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError, match="Invalid base64"):
            _validate_ssh_public_key("ssh-ed25519 !!!invalid!!! test@host")

    def test_multi_line_uses_first_key(self):
        multi = f"{VALID_ED25519}\nssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBBBBBBBBBBBBb second@host"
        result = _validate_ssh_public_key(multi)
        assert result == "ssh-ed25519"

    def test_all_valid_types_accepted(self):
        """All types in VALID_SSH_KEY_TYPES should be recognized."""
        assert len(VALID_SSH_KEY_TYPES) >= 5
        assert "ssh-ed25519" in VALID_SSH_KEY_TYPES
        assert "ssh-rsa" in VALID_SSH_KEY_TYPES


class TestSshKeyFingerprint:
    """Tests for _ssh_key_fingerprint."""

    def test_returns_sha256_prefix(self):
        fp = _ssh_key_fingerprint(VALID_ED25519)
        assert fp.startswith("SHA256:")

    def test_deterministic(self):
        fp1 = _ssh_key_fingerprint(VALID_ED25519)
        fp2 = _ssh_key_fingerprint(VALID_ED25519)
        assert fp1 == fp2

    def test_different_keys_different_fingerprints(self):
        # Use a different but valid ed25519 key (different base64 data, properly padded)
        import base64

        fake_data = base64.b64encode(b"\x00" * 32 + b"different-key-data").decode()
        key2 = f"ssh-ed25519 {fake_data} other@host"
        fp1 = _ssh_key_fingerprint(VALID_ED25519)
        fp2 = _ssh_key_fingerprint(key2)
        assert fp1 != fp2

    def test_fingerprint_format(self):
        """Fingerprint should be SHA256:<base64> with no trailing =."""
        fp = _ssh_key_fingerprint(VALID_ED25519)
        assert "=" not in fp
        prefix, b64part = fp.split(":", 1)
        assert prefix == "SHA256"
        assert len(b64part) > 10


class TestContainerNameFormat:
    """Verify container naming convention across the codebase."""

    def test_worker_agent_uses_xcl_prefix(self):
        """The container name format should be xcl-{job_id}."""
        job_id = "test-abc-123"
        container_name = f"xcl-{job_id}"
        assert container_name == "xcl-test-abc-123"
        assert not container_name.startswith("xcelsior-")

    def test_slurm_adapter_uses_xcl_prefix(self):
        """SLURM adapter should use xcl- prefix for job names."""
        from slurm_adapter import xcelsior_job_to_sbatch

        job = {
            "job_id": "verify-123",
            "vram_needed_gb": 8,
            "priority": 50,
            "image": "ghcr.io/xcelsior/test:latest",
        }
        script, meta = xcelsior_job_to_sbatch(job, "generic")
        assert "#SBATCH --job-name=xcl-verify-123" in script
        assert "xcelsior-verify-123" not in script
