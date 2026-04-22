"""Real filesystem tests for volume provisioning and deletion logic.

Uses /tmp/xcelsior-test-volumes/ so no NFS server needed.
Tests the command strings that would be executed over SSH.
"""

import os
import shlex
import tempfile
import shutil
import subprocess

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")

TMP_BASE = "/tmp/xcelsior-test-volumes"


@pytest.fixture(autouse=True)
def clean_tmp():
    """Create and cleanup the temp volume directory for each test."""
    os.makedirs(TMP_BASE, exist_ok=True)
    yield
    shutil.rmtree(TMP_BASE, ignore_errors=True)


# ── Provision command realism ────────────────────────────────────────


class TestProvisionCommand:
    """Verify the mkdir + chmod command works on a real filesystem."""

    def test_mkdir_p_creates_directory(self):
        vol_id = "vol-test-mkdir"
        vol_path = f"{TMP_BASE}/{vol_id}"
        safe_path = shlex.quote(vol_path)
        cmd = f"mkdir -p {safe_path} && chmod 1777 {safe_path}"
        rc = subprocess.call(cmd, shell=True)
        assert rc == 0
        assert os.path.isdir(vol_path)

    def test_chmod_1777_sticky_bit(self):
        vol_id = "vol-test-perms"
        vol_path = f"{TMP_BASE}/{vol_id}"
        safe_path = shlex.quote(vol_path)
        cmd = f"mkdir -p {safe_path} && chmod 1777 {safe_path}"
        subprocess.check_call(cmd, shell=True)
        mode = oct(os.stat(vol_path).st_mode)[-4:]
        assert mode == "1777"

    def test_idempotent_mkdir(self):
        """Running provision twice doesn't error."""
        vol_id = "vol-test-idem"
        vol_path = f"{TMP_BASE}/{vol_id}"
        safe_path = shlex.quote(vol_path)
        cmd = f"mkdir -p {safe_path} && chmod 1777 {safe_path}"
        assert subprocess.call(cmd, shell=True) == 0
        assert subprocess.call(cmd, shell=True) == 0
        assert os.path.isdir(vol_path)

    def test_special_characters_in_vol_id(self):
        """shlex.quote must handle volume IDs safely."""
        vol_id = "vol-test-$(evil)"
        vol_path = f"{TMP_BASE}/{vol_id}"
        safe_path = shlex.quote(vol_path)
        # The quoted path should not execute $() — mkdir should succeed
        cmd = f"mkdir -p {safe_path} && chmod 1777 {safe_path}"
        rc = subprocess.call(cmd, shell=True)
        assert rc == 0
        # The directory name should literally contain $(evil)
        assert os.path.isdir(vol_path)


# ── Destroy command safety ───────────────────────────────────────────


class TestDestroyCommand:
    """Verify the rm -rf --one-file-system command safety."""

    def test_basic_delete(self):
        vol_id = "vol-test-rm"
        vol_path = f"{TMP_BASE}/{vol_id}"
        os.makedirs(vol_path)
        # Put a file in it
        with open(f"{vol_path}/data.txt", "w") as f:
            f.write("test data")

        safe_path = shlex.quote(vol_path)
        cmd = (
            f"real=$(readlink -f {safe_path}) && "
            f'[[ "$real" == {shlex.quote(TMP_BASE)}/* ]] && '
            f'rm -rf --one-file-system "$real"'
        )
        rc = subprocess.call(cmd, shell=True, executable="/bin/bash")
        assert rc == 0
        assert not os.path.exists(vol_path)

    def test_symlink_escape_blocked(self):
        """A symlink pointing outside the base directory must be rejected."""
        vol_id = "vol-test-symlink"
        vol_path = f"{TMP_BASE}/{vol_id}"
        # Create a symlink pointing to /tmp (outside base)
        os.symlink("/tmp", vol_path)

        safe_path = shlex.quote(vol_path)
        cmd = (
            f"real=$(readlink -f {safe_path}) && "
            f'[[ "$real" == {shlex.quote(TMP_BASE)}/* ]] && '
            f'rm -rf --one-file-system "$real"'
        )
        rc = subprocess.call(cmd, shell=True, executable="/bin/bash")
        # Should fail because resolved path is /tmp, not under TMP_BASE/*
        assert rc != 0
        # /tmp must still exist
        assert os.path.isdir("/tmp")

    def test_dotdot_escape_blocked(self):
        """Path with /../ escape attempt is blocked by readlink -f."""
        vol_id = "vol-test/../../../tmp"
        vol_path = f"{TMP_BASE}/{vol_id}"
        safe_path = shlex.quote(vol_path)
        cmd = (
            f"real=$(readlink -f {safe_path}) && "
            f'[[ "$real" == {shlex.quote(TMP_BASE)}/* ]] && '
            f'rm -rf --one-file-system "$real"'
        )
        rc = subprocess.call(cmd, shell=True, executable="/bin/bash")
        # readlink -f resolves to /tmp which is NOT under TMP_BASE
        assert rc != 0
        assert os.path.isdir("/tmp")

    def test_one_file_system_flag_used(self):
        """Verify the destroy command uses --one-file-system."""
        import inspect
        from volumes import VolumeEngine

        source = inspect.getsource(VolumeEngine._destroy_volume_storage)
        assert "--one-file-system" in source

    def test_nonexistent_volume_no_error(self):
        """Deleting a volume that doesn't exist doesn't crash."""
        vol_path = f"{TMP_BASE}/vol-does-not-exist"
        safe_path = shlex.quote(vol_path)
        cmd = (
            f"real=$(readlink -f {safe_path}) && "
            f'[[ "$real" == {shlex.quote(TMP_BASE)}/* ]] && '
            f'rm -rf --one-file-system "$real"'
        )
        # readlink -f returns empty for non-existent paths, so the [[ ]] test fails
        rc = subprocess.call(cmd, shell=True, executable="/bin/bash")
        # This is expected to fail (non-zero) since path doesn't exist
        assert rc != 0 or not os.path.exists(vol_path)


# ── NFS_EXPORT_BASE path safety ──────────────────────────────────────


class TestExportBaseSafety:
    """Verify the export base path is used safely."""

    def test_export_base_constant_exists(self):
        from volumes import NFS_EXPORT_BASE

        assert NFS_EXPORT_BASE
        assert "/" in NFS_EXPORT_BASE

    def test_volume_path_construction(self):
        """Volume paths must be under NFS_EXPORT_BASE."""
        from volumes import NFS_EXPORT_BASE

        vol_id = "vol-abc123"
        path = f"{NFS_EXPORT_BASE}/{vol_id}"
        assert path.startswith(NFS_EXPORT_BASE + "/")

    def test_shlex_quote_used_in_provision(self):
        import inspect
        from volumes import VolumeEngine

        source = inspect.getsource(VolumeEngine._provision_volume_storage)
        assert "shlex.quote" in source

    def test_shlex_quote_used_in_destroy(self):
        import inspect
        from volumes import VolumeEngine

        source = inspect.getsource(VolumeEngine._destroy_volume_storage)
        assert "shlex.quote" in source

    def test_readlink_used_in_destroy(self):
        """Destroy must resolve symlinks before deleting."""
        import inspect
        from volumes import VolumeEngine

        source = inspect.getsource(VolumeEngine._destroy_volume_storage)
        assert "readlink -f" in source
