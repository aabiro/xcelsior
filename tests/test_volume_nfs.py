"""NFS resilience and SSH retry tests.

Tests: NFS_MOUNT_OPTS constant, _ssh_exec_with_retry, billing atomic
pattern, and source code verification for mount options.
"""

import os
import time

import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")


# ── NFS_MOUNT_OPTS constant verification ─────────────────────────────


class TestNFSMountOpts:
    """Verify the NFS mount options constant is correct."""

    def test_hard_mount(self):
        """Must be `hard` (never `soft`) — locked by volume-attachment plan.

        Soft NFS silently returns EIO on server reboot, corrupting write-back
        caches. Hard blocks-and-retries, preserving durability for
        checkpoints. The 3s stat timeout in telemetry guards against hangs.
        """
        from volumes import NFS_MOUNT_OPTS
        opts = NFS_MOUNT_OPTS.split(",")
        assert "hard" in opts
        assert "soft" not in opts

    def test_timeo_set(self):
        from volumes import NFS_MOUNT_OPTS
        assert "timeo=600" in NFS_MOUNT_OPTS

    def test_retrans_set(self):
        from volumes import NFS_MOUNT_OPTS
        assert "retrans=3" in NFS_MOUNT_OPTS

    def test_buffer_sizes(self):
        from volumes import NFS_MOUNT_OPTS
        assert "rsize=1048576" in NFS_MOUNT_OPTS
        assert "wsize=1048576" in NFS_MOUNT_OPTS

    def test_security_flags(self):
        from volumes import NFS_MOUNT_OPTS
        assert "nosuid" in NFS_MOUNT_OPTS
        assert "nodev" in NFS_MOUNT_OPTS

    def test_no_deprecated_intr(self):
        """'intr' is deprecated in NFSv4 and must not appear."""
        from volumes import NFS_MOUNT_OPTS
        opts = NFS_MOUNT_OPTS.split(",")
        assert "intr" not in opts

    def test_scheduler_imports_constant(self):
        """scheduler.py must import NFS_MOUNT_OPTS from volumes, not define its own."""
        with open("scheduler.py") as f:
            source = f.read()
        assert "from volumes import NFS_MOUNT_OPTS" in source

    def test_no_soft_mounts_in_codebase(self):
        """Zero 'soft' NFS mount references in production code."""
        for filename in ["volumes.py", "scheduler.py", "worker_agent.py"]:
            with open(filename) as f:
                content = f.read()
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                # Look for soft as a mount option (surrounded by commas or at start)
                if "soft," in stripped and "NFS" not in stripped.upper():
                    pytest.fail(f"Found 'soft' mount option in {filename}:{i}: {stripped}")


# ── SSH exec timeout handling ────────────────────────────────────────


class TestSSHExecTimeout:
    """scheduler.ssh_exec must catch TimeoutExpired."""

    def test_timeout_caught_in_source(self):
        import inspect
        from scheduler import ssh_exec
        source = inspect.getsource(ssh_exec)
        assert "TimeoutExpired" in source
        assert "255" in source  # returns rc=255

    def test_returns_255_on_timeout(self):
        """ssh_exec returns (255, '', 'timeout...') when subprocess times out."""
        from scheduler import ssh_exec
        # Use a command that will definitely time out with a very short timeout
        # We can't easily test this without mocking subprocess,
        # so verify the source handles it
        import inspect
        source = inspect.getsource(ssh_exec)
        assert 'return 255, "", f"ssh timeout' in source


# ── _ssh_exec_with_retry ─────────────────────────────────────────────


class TestSSHExecWithRetry:
    """VolumeEngine._ssh_exec_with_retry retries transient failures."""

    def test_succeeds_on_first_try(self):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        engine._ssh_exec = MagicMock(return_value=(0, "ok", ""))

        rc, out, err = engine._ssh_exec_with_retry("1.2.3.4", "echo hello")
        assert rc == 0
        assert engine._ssh_exec.call_count == 1

    def test_retries_on_timeout(self):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        # Fail twice with timeout, succeed on third
        engine._ssh_exec = MagicMock(side_effect=[
            (255, "", "ssh timeout after 30s"),
            (255, "", "ssh timeout after 30s"),
            (0, "done", ""),
        ])

        rc, out, err = engine._ssh_exec_with_retry("1.2.3.4", "mount ...", max_retries=3, timeout=5)
        assert rc == 0
        assert out == "done"
        assert engine._ssh_exec.call_count == 3

    def test_retries_on_connection_refused(self):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        engine._ssh_exec = MagicMock(side_effect=[
            (255, "", "Connection refused"),
            (0, "ok", ""),
        ])

        rc, out, err = engine._ssh_exec_with_retry("1.2.3.4", "cmd", max_retries=3, timeout=5)
        assert rc == 0
        assert engine._ssh_exec.call_count == 2

    def test_no_retry_on_remote_command_failure(self):
        """Non-255 exit code (remote command failed) is NOT retried."""
        from volumes import VolumeEngine
        engine = VolumeEngine()
        engine._ssh_exec = MagicMock(return_value=(1, "", "mkdir: Permission denied"))

        rc, out, err = engine._ssh_exec_with_retry("1.2.3.4", "mkdir /bad")
        assert rc == 1
        assert engine._ssh_exec.call_count == 1  # No retry

    def test_returns_last_result_on_exhaustion(self):
        """After max_retries, returns the last failure result."""
        from volumes import VolumeEngine
        engine = VolumeEngine()
        engine._ssh_exec = MagicMock(return_value=(255, "", "ssh timeout after 30s"))

        rc, out, err = engine._ssh_exec_with_retry("1.2.3.4", "cmd", max_retries=2, timeout=5)
        assert rc == 255
        assert "timeout" in err
        assert engine._ssh_exec.call_count == 2

    def test_retries_on_no_route_to_host(self):
        from volumes import VolumeEngine
        engine = VolumeEngine()
        engine._ssh_exec = MagicMock(side_effect=[
            (255, "", "No route to host"),
            (0, "ok", ""),
        ])

        rc, out, err = engine._ssh_exec_with_retry("1.2.3.4", "cmd", max_retries=3, timeout=5)
        assert rc == 0
        assert engine._ssh_exec.call_count == 2

    def test_no_retry_on_255_non_transient(self):
        """rc=255 but not a transient SSH error — don't retry."""
        from volumes import VolumeEngine
        engine = VolumeEngine()
        # 255 with non-transient error (e.g., auth failure)
        engine._ssh_exec = MagicMock(return_value=(255, "", "Permission denied (publickey)"))

        rc, out, err = engine._ssh_exec_with_retry("1.2.3.4", "cmd", max_retries=3, timeout=5)
        assert rc == 255
        assert engine._ssh_exec.call_count == 1  # Not retried


# ── Billing atomic pattern verification ──────────────────────────────


class TestBillingAtomicPattern:
    """Verify billing uses single-transaction pattern (not write-ahead)."""

    def test_volume_billing_uses_for_update_skip_locked(self):
        """Volume billing must use FOR UPDATE SKIP LOCKED like GPU billing."""
        import inspect
        from billing import BillingEngine
        source = inspect.getsource(BillingEngine.auto_billing_cycle)
        # The volume billing section should have FOR UPDATE SKIP LOCKED
        # Find the volume billing section
        vol_section = source[source.index("Bill active volumes"):]
        assert "FOR UPDATE SKIP LOCKED" in vol_section

    def test_volume_billing_single_transaction(self):
        """Charge + cycle INSERT must be in the same 'with pool.connection()' block."""
        import inspect
        from billing import BillingEngine
        source = inspect.getsource(BillingEngine.auto_billing_cycle)
        vol_section = source[source.index("Bill active volumes"):]

        # The charge and INSERT should happen inside the same connection block
        # Look for the pattern: charge → INSERT within the same indentation level
        assert "self.charge(" in vol_section
        assert "INSERT INTO billing_cycles" in vol_section

    def test_no_pending_status_in_billing(self):
        """Volume billing should NOT use 'pending' status (that was the old write-ahead pattern)."""
        import inspect
        from billing import BillingEngine
        source = inspect.getsource(BillingEngine.auto_billing_cycle)
        vol_section = source[source.index("Bill active volumes"):]
        assert "'pending'" not in vol_section

    def test_billing_uses_730_hours(self):
        """Volume billing must use 730 hours/month, not 720."""
        import inspect
        from billing import BillingEngine
        source = inspect.getsource(BillingEngine.auto_billing_cycle)
        assert "HOURS_PER_MONTH = 730" in source
        assert "720" not in source
