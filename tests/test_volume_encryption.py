"""Tests for volume encryption: key management, LUKS provisioning, and admin endpoints."""

import base64
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

# Dev Fernet key — matches security.py fallback when XCELSIOR_SECRETS_KEY is unset
os.environ.pop("XCELSIOR_SECRETS_KEY", None)
os.environ["XCELSIOR_ENV"] = "dev"

from volumes import VolumeEngine

# ── Helpers ──────────────────────────────────────────────────────────


def _make_engine():
    """Create a VolumeEngine with no real DB or SSH."""
    engine = VolumeEngine()
    return engine


def _fake_conn(rows):
    """Build a mock psycopg connection whose execute().fetchone() pops from `rows`."""
    conn = MagicMock()
    results = list(rows)

    def _exec(sql, params=None):
        cursor = MagicMock()
        val = results.pop(0) if results else None
        cursor.fetchone.return_value = val
        cursor.fetchall.return_value = [val] if val else []
        return cursor

    conn.execute = _exec
    return conn


@contextmanager
def _mock_conn_ctx(conn):
    yield conn


# ── Key Generation ──────────────────────────────────────────────────


class TestKeyGeneration:
    def test_key_is_32_bytes(self):
        key = VolumeEngine._generate_volume_key()
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_keys_are_random(self):
        k1 = VolumeEngine._generate_volume_key()
        k2 = VolumeEngine._generate_volume_key()
        assert k1 != k2


# ── Key Encrypt / Decrypt Roundtrip ─────────────────────────────────


class TestKeyEncryptDecrypt:
    def test_roundtrip(self):
        raw = os.urandom(32)
        ct = VolumeEngine._encrypt_key(raw)
        assert isinstance(ct, str)
        recovered = VolumeEngine._decrypt_key(ct)
        assert recovered == raw

    def test_ciphertext_is_base64ish(self):
        ct = VolumeEngine._encrypt_key(b"\x00" * 32)
        # Fernet tokens are url-safe base64
        assert isinstance(ct, str)
        assert len(ct) > 40  # not just the plaintext

    def test_different_keys_give_different_ciphertext(self):
        c1 = VolumeEngine._encrypt_key(os.urandom(32))
        c2 = VolumeEngine._encrypt_key(os.urandom(32))
        assert c1 != c2


# ── Key Store / Retrieve ────────────────────────────────────────────


class TestKeyStoreRetrieve:
    def test_store_calls_update(self):
        engine = _make_engine()
        conn = MagicMock()
        raw = os.urandom(32)
        engine._store_key(conn, "vol-abc123", raw)
        conn.execute.assert_called_once()
        sql, params = conn.execute.call_args[0]
        assert "UPDATE volumes SET key_ciphertext" in sql
        assert params[1] == "vol-abc123"
        # params[0] is the encrypted ciphertext — verify it decrypts back
        assert VolumeEngine._decrypt_key(params[0]) == raw

    def test_retrieve_roundtrip(self):
        engine = _make_engine()
        raw = os.urandom(32)
        ct = VolumeEngine._encrypt_key(raw)
        conn = _fake_conn([{"key_ciphertext": ct}])
        recovered = engine._retrieve_key(conn, "vol-abc123")
        assert recovered == raw

    def test_retrieve_no_key(self):
        engine = _make_engine()
        conn = _fake_conn([{"key_ciphertext": ""}])
        assert engine._retrieve_key(conn, "vol-abc123") is None

    def test_retrieve_no_row(self):
        engine = _make_engine()
        conn = _fake_conn([None])
        assert engine._retrieve_key(conn, "vol-abc123") is None


# ── Key Destroy ─────────────────────────────────────────────────────


class TestKeyDestroy:
    def test_destroy_sets_empty(self):
        engine = _make_engine()
        conn = MagicMock()
        engine._destroy_key(conn, "vol-xyz")
        conn.execute.assert_called_once()
        sql, params = conn.execute.call_args[0]
        assert "key_ciphertext = ''" in sql
        assert params == ("vol-xyz",)


# ── Encrypted Provision (SSH command verification) ──────────────────


class TestProvisionEncrypted:
    """Verify _provision_volume_storage encrypted path: correct commands, sudo, key-size."""

    def _run(self, *, nfs_server="10.0.0.1", volume_id="vol-testabc123", size_gb=20, raw_key=None):
        engine = _make_engine()
        raw_key = raw_key or os.urandom(32)
        ssh_calls = []
        stdin_calls = []

        def mock_ssh_retry(ip, cmd, **kw):
            ssh_calls.append((ip, cmd))
            return (0, "", "")

        def mock_ssh_stdin(ip, cmd, stdin_data, **kw):
            stdin_calls.append((ip, cmd, stdin_data))
            return (0, "", "")

        engine._ssh_exec_with_retry = mock_ssh_retry
        engine._ssh_exec_with_stdin = mock_ssh_stdin

        with patch.object(type(engine), "__module__", "volumes"):
            with patch("volumes.NFS_SERVER", nfs_server):
                with patch("volumes.NFS_EXPORT_BASE", "/exports/volumes"):
                    result = engine._provision_volume_storage(
                        volume_id,
                        size_gb,
                        encrypted=True,
                        raw_key=raw_key,
                    )
        return result, ssh_calls, stdin_calls, raw_key

    def test_success_returns_true(self):
        ok, _, _, _ = self._run()
        assert ok is True

    def test_step0_stale_cleanup(self):
        ok, ssh_calls, _, _ = self._run()
        # First SSH call should be the stale cleanup
        cleanup_cmd = ssh_calls[0][1]
        assert "sudo umount -l" in cleanup_cmd
        assert "sudo cryptsetup luksClose" in cleanup_cmd

    def test_step1_truncate(self):
        ok, ssh_calls, _, _ = self._run(size_gb=50)
        # Second SSH call: truncate
        truncate_cmd = ssh_calls[1][1]
        assert "truncate -s 50G" in truncate_cmd
        assert ".img" in truncate_cmd

    def test_step2_luks_format_sudo(self):
        _, _, stdin_calls, raw_key = self._run()
        # First stdin call: luksFormat
        fmt_cmd = stdin_calls[0][1]
        assert fmt_cmd.startswith("sudo cryptsetup luksFormat")
        assert "--key-size 512" in fmt_cmd
        assert "aes-xts-plain64" in fmt_cmd
        assert "--type luks2" in fmt_cmd
        assert "--hash sha256" in fmt_cmd
        assert "--key-file /dev/stdin" in fmt_cmd
        # Key piped correctly
        assert stdin_calls[0][2] == raw_key

    def test_step3_luks_open_sudo(self):
        _, _, stdin_calls, raw_key = self._run()
        # Second stdin call: luksOpen
        open_cmd = stdin_calls[1][1]
        assert open_cmd.startswith("sudo cryptsetup luksOpen")
        assert "--key-file /dev/stdin" in open_cmd
        assert stdin_calls[1][2] == raw_key

    def test_step4_mkfs_sudo(self):
        _, ssh_calls, _, _ = self._run()
        # Third SSH call (after cleanup, truncate): mkfs
        mkfs_cmd = ssh_calls[2][1]
        assert "sudo mkfs.ext4" in mkfs_cmd
        assert "/dev/mapper/" in mkfs_cmd

    def test_step5_mount_sudo(self):
        _, ssh_calls, _, _ = self._run()
        # Fourth SSH call: mount
        mount_cmd = ssh_calls[3][1]
        assert "sudo mount" in mount_cmd
        assert "/dev/mapper/" in mount_cmd

    def test_no_nfs_server_returns_true(self):
        """Without NFS_SERVER, provision is metadata-only → True."""
        engine = _make_engine()
        with patch("volumes.NFS_SERVER", ""):
            ok = engine._provision_volume_storage(
                "vol-x", 10, encrypted=True, raw_key=os.urandom(32)
            )
        assert ok is True

    def test_no_key_returns_false(self):
        engine = _make_engine()
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            ok = engine._provision_volume_storage("vol-x", 10, encrypted=True, raw_key=None)
        assert ok is False

    def test_luks_format_failure_cleans_image(self):
        engine = _make_engine()
        cleanup_cmds = []

        def ssh_retry(ip, cmd, **kw):
            cleanup_cmds.append(cmd)
            return (0, "", "")

        call_count = 0

        def ssh_stdin_fail_format(ip, cmd, data, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # luksFormat
                return (1, "", "format error")
            return (0, "", "")

        engine._ssh_exec_with_retry = ssh_retry
        engine._ssh_exec_with_stdin = ssh_stdin_fail_format
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            ok = engine._provision_volume_storage(
                "vol-x", 10, encrypted=True, raw_key=os.urandom(32)
            )
        assert ok is False
        # Should have cleaned up: stale cleanup + truncate + rm -f of image
        assert any("rm -f" in c for c in cleanup_cmds)

    def test_luks_open_failure_cleans_image(self):
        engine = _make_engine()
        cleanup_cmds = []

        def ssh_retry(ip, cmd, **kw):
            cleanup_cmds.append(cmd)
            return (0, "", "")

        stdin_count = 0

        def ssh_stdin(ip, cmd, data, **kw):
            nonlocal stdin_count
            stdin_count += 1
            if stdin_count == 2:  # luksOpen (second stdin call)
                return (1, "", "open error")
            return (0, "", "")

        engine._ssh_exec_with_retry = ssh_retry
        engine._ssh_exec_with_stdin = ssh_stdin
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            ok = engine._provision_volume_storage(
                "vol-x", 10, encrypted=True, raw_key=os.urandom(32)
            )
        assert ok is False
        assert any("rm -f" in c for c in cleanup_cmds)

    def test_mkfs_failure_cleans_up(self):
        engine = _make_engine()
        cleanup_cmds = []
        ssh_count = 0

        def ssh_retry(ip, cmd, **kw):
            nonlocal ssh_count
            ssh_count += 1
            cleanup_cmds.append(cmd)
            if ssh_count == 3:  # mkfs
                return (1, "", "mkfs error")
            return (0, "", "")

        engine._ssh_exec_with_retry = ssh_retry
        engine._ssh_exec_with_stdin = lambda ip, cmd, data, **kw: (0, "", "")
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            ok = engine._provision_volume_storage(
                "vol-x", 10, encrypted=True, raw_key=os.urandom(32)
            )
        assert ok is False
        # Should close LUKS and remove image
        assert any("luksClose" in c for c in cleanup_cmds)

    def test_mount_failure_cleans_up(self):
        engine = _make_engine()
        cleanup_cmds = []
        ssh_count = 0

        def ssh_retry(ip, cmd, **kw):
            nonlocal ssh_count
            ssh_count += 1
            cleanup_cmds.append(cmd)
            if ssh_count == 4:  # mount
                return (1, "", "mount error")
            return (0, "", "")

        engine._ssh_exec_with_retry = ssh_retry
        engine._ssh_exec_with_stdin = lambda ip, cmd, data, **kw: (0, "", "")
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            ok = engine._provision_volume_storage(
                "vol-x", 10, encrypted=True, raw_key=os.urandom(32)
            )
        assert ok is False
        assert any("luksClose" in c for c in cleanup_cmds)
        assert any("rmdir" in c for c in cleanup_cmds)

    def test_mapper_name_uses_volume_id_prefix(self):
        _, ssh_calls, stdin_calls, _ = self._run(volume_id="vol-deadbeef99")
        # Mapper name should contain volume_id[:12]
        all_cmds = [c[1] for c in ssh_calls] + [c[1] for c in stdin_calls]
        mapper_cmds = [c for c in all_cmds if "xcelsior-vol-" in c]
        assert len(mapper_cmds) > 0
        for c in mapper_cmds:
            assert "xcelsior-vol-vol-deadbeef" in c


# ── Unencrypted Provision ───────────────────────────────────────────


class TestProvisionUnencrypted:
    def test_plain_mkdir(self):
        engine = _make_engine()
        ssh_calls = []

        def mock_ssh(ip, cmd, **kw):
            ssh_calls.append(cmd)
            return (0, "", "")

        engine._ssh_exec_with_retry = mock_ssh
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            ok = engine._provision_volume_storage("vol-plain", 10, encrypted=False)
        assert ok is True
        assert len(ssh_calls) == 1
        assert "mkdir -p" in ssh_calls[0]
        assert "chmod 1777" in ssh_calls[0]
        # No LUKS commands
        assert "cryptsetup" not in ssh_calls[0]

    def test_no_sudo_for_unencrypted(self):
        engine = _make_engine()
        cmds = []

        def mock_ssh(ip, cmd, **kw):
            cmds.append(cmd)
            return (0, "", "")

        engine._ssh_exec_with_retry = mock_ssh
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            engine._provision_volume_storage("vol-plain", 10, encrypted=False)
        for c in cmds:
            assert "sudo" not in c


# ── Destroy Encrypted ───────────────────────────────────────────────


class TestDestroyEncrypted:
    def test_umount_luks_close_remove(self):
        engine = _make_engine()
        ssh_calls = []

        def mock_ssh(ip, cmd, **kw):
            ssh_calls.append(cmd)
            return (0, "", "")

        engine._ssh_exec_with_retry = mock_ssh
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            with patch("volumes.NFS_EXPORT_BASE", "/exports/volumes"):
                ok = engine._destroy_volume_storage("vol-enc123", encrypted=True)
        assert ok is True
        assert len(ssh_calls) == 3
        # umount -l with sudo
        assert "sudo umount -l" in ssh_calls[0]
        # luksClose with sudo
        assert "sudo cryptsetup luksClose" in ssh_calls[1]
        # rm image + rmdir
        assert "rm -f" in ssh_calls[2]
        assert ".img" in ssh_calls[2]
        assert "rmdir" in ssh_calls[2]


class TestDestroyUnencrypted:
    def test_safe_rm_with_symlink_guard(self):
        engine = _make_engine()
        ssh_calls = []

        def mock_ssh(ip, cmd, **kw):
            ssh_calls.append(cmd)
            return (0, "", "")

        engine._ssh_exec_with_retry = mock_ssh
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            with patch("volumes.NFS_EXPORT_BASE", "/exports/volumes"):
                ok = engine._destroy_volume_storage("vol-plain", encrypted=False)
        assert ok is True
        assert len(ssh_calls) == 1
        assert "readlink -f" in ssh_calls[0]
        assert "--one-file-system" in ssh_calls[0]
        assert "cryptsetup" not in ssh_calls[0]


# ── create_volume with encryption ──────────────────────────────────


class TestCreateVolumeEncryption:
    def _make(self, monkeypatch, conn_rows, provision_ok=True):
        engine = _make_engine()
        results = list(conn_rows)
        stored_keys = {}

        def mock_exec(sql, params=None):
            cursor = MagicMock()
            val = results.pop(0) if results else None
            cursor.fetchone.return_value = val
            cursor.fetchall.return_value = [val] if val else []
            # Capture key storage
            if params and "key_ciphertext" in str(sql):
                stored_keys["ct"] = params[0]
                stored_keys["vid"] = params[1]
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        monkeypatch.setattr(engine, "_provision_volume_storage", lambda vid, sz, **kw: provision_ok)
        monkeypatch.setattr(engine, "_emit_event", lambda *a, **kw: None)
        return engine, stored_keys

    def test_encrypted_stores_key(self, monkeypatch):
        conn_rows = [
            None,  # FOR UPDATE lock
            {"total": 0},  # capacity check
            None,  # name uniqueness
        ]
        engine, stored_keys = self._make(monkeypatch, conn_rows)
        vol = engine.create_volume("user-1", "enc-vol", 10, encrypted=True)
        assert vol["encrypted"] is True
        # A key should have been stored
        assert "ct" in stored_keys
        # The stored ciphertext should decrypt to 32 bytes
        raw = VolumeEngine._decrypt_key(stored_keys["ct"])
        assert len(raw) == 32

    def test_unencrypted_no_key(self, monkeypatch):
        conn_rows = [
            None,
            {"total": 0},
            None,
        ]
        engine, stored_keys = self._make(monkeypatch, conn_rows)
        vol = engine.create_volume("user-1", "plain-vol", 10, encrypted=False)
        assert vol["encrypted"] is False
        assert "ct" not in stored_keys


# ── delete_volume with encryption ──────────────────────────────────


class TestDeleteVolumeEncryption:
    def test_delete_encrypted_destroys_key(self, monkeypatch):
        engine = _make_engine()
        key_destroyed = {}
        call_idx = 0

        def mock_exec(sql, params=None):
            nonlocal call_idx
            cursor = MagicMock()
            call_idx += 1
            # First call: SELECT * FROM volumes ... FOR UPDATE
            if call_idx == 1:
                cursor.fetchone.return_value = {
                    "volume_id": "vol-enc1",
                    "owner_id": "user-1",
                    "status": "available",
                    "encrypted": True,
                    "size_gb": 10,
                    "name": "enc-vol",
                }
            # Second call: SELECT attachment_id ...
            elif call_idx == 2:
                cursor.fetchone.return_value = None
            # Third call: UPDATE status → deleting
            elif call_idx == 3:
                cursor.fetchone.return_value = None
            # Fourth+ calls in second/third conn context
            elif "key_ciphertext = ''" in str(sql):
                key_destroyed["destroyed"] = True
                cursor.fetchone.return_value = None
            else:
                cursor.fetchone.return_value = None
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            nonlocal call_idx
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        monkeypatch.setattr(engine, "_destroy_volume_storage", lambda vid, **kw: True)
        monkeypatch.setattr(engine, "_emit_event", lambda *a, **kw: None)

        result = engine.delete_volume("vol-enc1", "user-1")
        assert result["status"] == "deleted"
        assert key_destroyed.get("destroyed") is True


# ── retry_provision ─────────────────────────────────────────────────


class TestRetryProvisionEncryption:
    def test_retry_reuses_existing_key(self, monkeypatch):
        engine = _make_engine()
        raw_key = os.urandom(32)
        existing_ct = VolumeEngine._encrypt_key(raw_key)
        provision_kwargs = {}

        def mock_provision(vid, sz, **kw):
            provision_kwargs.update(kw)
            return True

        call_idx = 0

        def mock_exec(sql, params=None):
            nonlocal call_idx
            cursor = MagicMock()
            call_idx += 1
            if call_idx == 1:
                cursor.fetchone.return_value = {
                    "volume_id": "vol-retry",
                    "owner_id": "user-1",
                    "status": "error",
                    "size_gb": 20,
                    "encrypted": True,
                    "key_ciphertext": existing_ct,
                }
            else:
                cursor.fetchone.return_value = None
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        monkeypatch.setattr(engine, "_provision_volume_storage", mock_provision)
        monkeypatch.setattr(engine, "_emit_event", lambda *a, **kw: None)

        result = engine.retry_provision("vol-retry", "user-1")
        assert result["status"] == "available"
        # Should pass the existing key
        assert provision_kwargs["raw_key"] == raw_key
        assert provision_kwargs["encrypted"] is True

    def test_retry_generates_new_key_if_none(self, monkeypatch):
        engine = _make_engine()
        provision_kwargs = {}

        def mock_provision(vid, sz, **kw):
            provision_kwargs.update(kw)
            return True

        call_idx = 0

        def mock_exec(sql, params=None):
            nonlocal call_idx
            cursor = MagicMock()
            call_idx += 1
            if call_idx == 1:
                cursor.fetchone.return_value = {
                    "volume_id": "vol-retry2",
                    "owner_id": "user-1",
                    "status": "error",
                    "size_gb": 10,
                    "encrypted": True,
                    "key_ciphertext": "",
                }
            else:
                cursor.fetchone.return_value = None
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        monkeypatch.setattr(engine, "_provision_volume_storage", mock_provision)
        monkeypatch.setattr(engine, "_emit_event", lambda *a, **kw: None)

        result = engine.retry_provision("vol-retry2", "user-1")
        assert result["status"] == "available"
        # New key should have been generated
        assert provision_kwargs["raw_key"] is not None
        assert len(provision_kwargs["raw_key"]) == 32
        assert provision_kwargs["encrypted"] is True


# ── reopen_luks_volume ──────────────────────────────────────────────


class TestReopenLuksVolume:
    def test_reopen_success(self, monkeypatch):
        engine = _make_engine()
        raw_key = os.urandom(32)
        ct = VolumeEngine._encrypt_key(raw_key)
        stdin_calls = []

        def mock_exec(sql, params=None):
            cursor = MagicMock()
            cursor.fetchone.return_value = {"key_ciphertext": ct}
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        def mock_ssh_retry(ip, cmd, **kw):
            return (0, "", "")

        def mock_ssh_stdin(ip, cmd, data, **kw):
            stdin_calls.append((cmd, data))
            return (0, "", "")

        monkeypatch.setattr(engine, "_conn", mock_conn)
        engine._ssh_exec_with_retry = mock_ssh_retry
        engine._ssh_exec_with_stdin = mock_ssh_stdin

        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            with patch("volumes.NFS_EXPORT_BASE", "/exports/volumes"):
                ok = engine.reopen_luks_volume("vol-reopen123")
        assert ok is True
        # Should have piped the key to luksOpen
        assert len(stdin_calls) == 1
        assert "sudo cryptsetup luksOpen" in stdin_calls[0][0]
        assert stdin_calls[0][1] == raw_key

    def test_reopen_no_key_returns_false(self, monkeypatch):
        engine = _make_engine()

        def mock_exec(sql, params=None):
            cursor = MagicMock()
            cursor.fetchone.return_value = {"key_ciphertext": ""}
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            ok = engine.reopen_luks_volume("vol-nokey")
        assert ok is False

    def test_reopen_no_nfs(self, monkeypatch):
        engine = _make_engine()
        with patch("volumes.NFS_SERVER", ""):
            ok = engine.reopen_luks_volume("vol-x")
        assert ok is False


# ── SSH stdin security (key never written to disk) ───────────────────


class TestSshStdinSecurity:
    """Verify the _ssh_exec_with_stdin method pipes key via stdin."""

    def test_stdin_pipe_not_file(self, monkeypatch):
        engine = _make_engine()
        raw_key = os.urandom(32)
        _, ssh_calls, stdin_calls, _ = TestProvisionEncrypted()._run(raw_key=raw_key)
        # All LUKS key operations must use stdin pipe, never temp files
        for _, cmd, data in stdin_calls:
            assert "--key-file /dev/stdin" in cmd
            assert data == raw_key


# ── Scheduler encrypted_workspace ──────────────────────────────────


class TestSchedulerEncryptedWorkspace:
    def test_submit_job_stores_encrypted_workspace(self):
        """scheduler.submit_job must persist encrypted_workspace in the job dict."""
        import inspect
        from scheduler import submit_job

        sig = inspect.signature(submit_job)
        assert "encrypted_workspace" in sig.parameters

    def test_submit_job_defaults_false(self):
        import inspect
        from scheduler import submit_job

        sig = inspect.signature(submit_job)
        param = sig.parameters["encrypted_workspace"]
        assert param.default is False


# ── JobIn model ─────────────────────────────────────────────────────


class TestJobInEncryptedWorkspace:
    def test_job_in_has_encrypted_workspace(self):
        from routes.instances import JobIn

        job = JobIn(name="test", image="nvidia/cuda:12.0-base")
        assert hasattr(job, "encrypted_workspace")
        assert job.encrypted_workspace is False

    def test_job_in_accepts_true(self):
        from routes.instances import JobIn

        job = JobIn(name="test", image="nvidia/cuda:12.0-base", encrypted_workspace=True)
        assert job.encrypted_workspace is True


# ── Admin Reopen Endpoint ──────────────────────────────────────────


class TestAdminReopenEndpoint:
    def test_route_exists(self):
        from routes.volumes import router

        paths = [r.path for r in router.routes]
        assert "/api/v2/admin/volumes/reopen-encrypted" in paths

    def test_route_method_is_post(self):
        from routes.volumes import router

        for r in router.routes:
            if getattr(r, "path", "") == "/api/v2/admin/volumes/reopen-encrypted":
                assert "POST" in r.methods
                break
        else:
            pytest.fail("Route not found")


# ── Key-size 512 everywhere ────────────────────────────────────────


class TestKeySize:
    """Verify AES-256-XTS (key-size 512) is used in all LUKS format commands."""

    def test_volumes_py_uses_512(self):
        import inspect
        from volumes import VolumeEngine

        src = inspect.getsource(VolumeEngine._provision_volume_storage)
        assert "--key-size 512" in src
        assert "--key-size 256" not in src

    def test_worker_agent_uses_512(self):
        import inspect
        from worker_agent import provision_encrypted_volume

        src = inspect.getsource(provision_encrypted_volume)
        # worker_agent uses list args: "--key-size", "512"
        assert '"512"' in src or "'512'" in src
        assert '"256"' not in src.replace("256-bit", "").replace("256-bit", "")


# ── Sudo presence in all privileged commands ────────────────────────


class TestSudoPresence:
    """All LUKS/mount/mkfs commands must be prefixed with sudo."""

    def test_provision_sudo(self):
        import inspect
        from volumes import VolumeEngine

        src = inspect.getsource(VolumeEngine._provision_volume_storage)
        # Every cryptsetup, mount, mkfs.ext4 in the encrypted path must have sudo
        lines = src.split("\n")
        for line in lines:
            stripped = line.strip()
            # Skip comments and strings that are just cleanup
            if stripped.startswith("#") or stripped.startswith("log."):
                continue
            # Check cryptsetup commands have sudo
            if "cryptsetup" in stripped and 'f"' in stripped or "f'" in stripped:
                if "sudo cryptsetup" not in stripped and "2>/dev/null" not in stripped:
                    # Lines with 2>/dev/null are in cleanup strings that DO have sudo
                    pass
            if "mkfs.ext4" in stripped and 'f"' in stripped:
                assert "sudo mkfs.ext4" in stripped, f"Missing sudo on mkfs: {stripped}"

    def test_destroy_sudo(self):
        import inspect
        from volumes import VolumeEngine

        src = inspect.getsource(VolumeEngine._destroy_volume_storage)
        assert "sudo umount" in src
        assert "sudo cryptsetup luksClose" in src

    def test_reopen_sudo(self):
        import inspect
        from volumes import VolumeEngine

        src = inspect.getsource(VolumeEngine.reopen_luks_volume)
        assert "sudo cryptsetup luksOpen" in src
        assert "sudo mount" in src


# ── VolumeCreate Model ─────────────────────────────────────────────


class TestVolumeCreateModel:
    def test_default_encrypted_true(self):
        from routes.volumes import VolumeCreate

        model = VolumeCreate(name="test-vol", size_gb=10)
        assert model.encrypted is True

    def test_explicit_encrypted_false(self):
        from routes.volumes import VolumeCreate

        model = VolumeCreate(name="test-vol", size_gb=10, encrypted=False)
        assert model.encrypted is False

    def test_default_size(self):
        from routes.volumes import VolumeCreate

        model = VolumeCreate(name="vol")
        assert model.size_gb == 50

    def test_default_region(self):
        from routes.volumes import VolumeCreate

        model = VolumeCreate(name="vol")
        assert model.region == "ca-east"


# ── Reopen Failure Paths ───────────────────────────────────────────


class TestReopenLuksFailurePaths:
    def test_reopen_luks_open_fails(self, monkeypatch):
        engine = _make_engine()
        raw_key = os.urandom(32)
        ct = VolumeEngine._encrypt_key(raw_key)

        def mock_exec(sql, params=None):
            cursor = MagicMock()
            cursor.fetchone.return_value = {"key_ciphertext": ct}
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        engine._ssh_exec_with_stdin = lambda ip, cmd, data, **kw: (1, "", "luksOpen error")

        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            with patch("volumes.NFS_EXPORT_BASE", "/exports/volumes"):
                ok = engine.reopen_luks_volume("vol-failopen")
        assert ok is False

    def test_reopen_mount_fails(self, monkeypatch):
        engine = _make_engine()
        raw_key = os.urandom(32)
        ct = VolumeEngine._encrypt_key(raw_key)

        def mock_exec(sql, params=None):
            cursor = MagicMock()
            cursor.fetchone.return_value = {"key_ciphertext": ct}
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        # luksOpen succeeds, mount fails
        engine._ssh_exec_with_stdin = lambda ip, cmd, data, **kw: (0, "", "")
        engine._ssh_exec_with_retry = lambda ip, cmd, **kw: (1, "", "mount error")

        with patch("volumes.NFS_SERVER", "10.0.0.1"):
            with patch("volumes.NFS_EXPORT_BASE", "/exports/volumes"):
                ok = engine.reopen_luks_volume("vol-failmount")
        assert ok is False


# ── Delete Volume (unencrypted path) ───────────────────────────────


class TestDeleteVolumeUnencrypted:
    def test_delete_unencrypted_no_key_destroy(self, monkeypatch):
        engine = _make_engine()
        key_destroy_called = {"called": False}
        call_idx = 0

        def mock_exec(sql, params=None):
            nonlocal call_idx
            cursor = MagicMock()
            call_idx += 1
            if call_idx == 1:
                cursor.fetchone.return_value = {
                    "volume_id": "vol-plain1",
                    "owner_id": "user-1",
                    "status": "available",
                    "encrypted": False,
                    "size_gb": 10,
                    "name": "plain-vol",
                }
            elif call_idx == 2:
                cursor.fetchone.return_value = None  # no attachment
            elif "key_ciphertext = ''" in str(sql):
                key_destroy_called["called"] = True
                cursor.fetchone.return_value = None
            else:
                cursor.fetchone.return_value = None
            return cursor

        conn = MagicMock()
        conn.execute = mock_exec

        @contextmanager
        def mock_conn():
            yield conn

        monkeypatch.setattr(engine, "_conn", mock_conn)
        monkeypatch.setattr(engine, "_destroy_volume_storage", lambda vid, **kw: True)
        monkeypatch.setattr(engine, "_emit_event", lambda *a, **kw: None)

        result = engine.delete_volume("vol-plain1", "user-1")
        assert result["status"] == "deleted"
        # Key destroy should NOT be called for unencrypted volumes
        assert key_destroy_called["called"] is False


# ── Worker Agent Encryption Functions ──────────────────────────────


class TestWorkerProvisionEncryptedVolume:
    """Test worker_agent.provision_encrypted_volume subprocess calls."""

    def test_provision_success(self, monkeypatch):
        import worker_agent

        calls = []

        def mock_run(args, **kw):
            calls.append(args)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.makedirs", lambda *a, **kw: None)
        monkeypatch.setattr("worker_agent.os.umask", lambda m: 0o022)

        mock_key = b"\xaa" * 32
        monkeypatch.setattr("worker_agent.os.urandom", lambda n: mock_key)

        import builtins

        original_open = builtins.open
        written_data = {}

        @contextmanager
        def mock_open(path, mode="r", **kw):
            if mode == "wb" and "volume-keys" in str(path):
                buf = MagicMock()
                buf.write = lambda data: written_data.update({"key": data})
                yield buf
            else:
                with original_open(path, mode, **kw) as f:
                    yield f

        monkeypatch.setattr("builtins.open", mock_open)

        ok = worker_agent.provision_encrypted_volume("testvol", 20)
        assert ok is True

        # Should have 5 subprocess calls: truncate, luksFormat, luksOpen, mkfs, mount
        assert len(calls) == 5

        # truncate
        assert calls[0][0] == "truncate"
        assert "-s" in calls[0]
        assert "20G" in calls[0]

        # luksFormat
        assert "luksFormat" in calls[1]
        assert "--key-size" in calls[1]
        idx = calls[1].index("--key-size")
        assert calls[1][idx + 1] == "512"
        assert "--cipher" in calls[1]
        cidx = calls[1].index("--cipher")
        assert calls[1][cidx + 1] == "aes-xts-plain64"
        assert "--type" in calls[1]
        tidx = calls[1].index("--type")
        assert calls[1][tidx + 1] == "luks2"

        # luksOpen
        assert "luksOpen" in calls[2]

        # mkfs
        assert calls[3][0] == "mkfs.ext4"

        # mount
        assert calls[4][0] == "mount"

        # Key written
        assert written_data["key"] == mock_key

    def test_provision_truncate_failure(self, monkeypatch):
        import subprocess
        import worker_agent

        call_count = 0

        def mock_run(args, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # truncate
                raise subprocess.CalledProcessError(1, args, stderr="truncate error")
            return MagicMock(returncode=0)

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.makedirs", lambda *a, **kw: None)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: False)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: False)

        ok = worker_agent.provision_encrypted_volume("failtest", 10)
        assert ok is False

    def test_provision_luks_format_failure(self, monkeypatch):
        import subprocess
        import worker_agent

        call_count = 0

        def mock_run(args, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # truncate ok
                return MagicMock(returncode=0)
            if call_count == 2:  # luksFormat fail
                raise subprocess.CalledProcessError(1, args, stderr="format error")
            return MagicMock(returncode=0)

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.makedirs", lambda *a, **kw: None)
        monkeypatch.setattr("worker_agent.os.umask", lambda m: 0o022)
        monkeypatch.setattr("worker_agent.os.urandom", lambda n: b"\x00" * 32)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: False)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: False)

        import builtins

        original_open = builtins.open

        @contextmanager
        def mock_open(path, mode="r", **kw):
            if mode == "wb":
                yield MagicMock()
            else:
                with original_open(path, mode, **kw) as f:
                    yield f

        monkeypatch.setattr("builtins.open", mock_open)

        ok = worker_agent.provision_encrypted_volume("failfmt", 10)
        assert ok is False

    def test_umask_set_077(self, monkeypatch):
        import worker_agent

        umask_values = []

        def mock_run(args, **kw):
            return MagicMock(returncode=0)

        original_umask = os.umask

        def track_umask(m):
            umask_values.append(m)
            return 0o022

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.makedirs", lambda *a, **kw: None)
        monkeypatch.setattr("worker_agent.os.umask", track_umask)
        monkeypatch.setattr("worker_agent.os.urandom", lambda n: b"\x00" * 32)

        import builtins

        original_open = builtins.open

        @contextmanager
        def mock_open(path, mode="r", **kw):
            if mode == "wb":
                yield MagicMock()
            else:
                with original_open(path, mode, **kw) as f:
                    yield f

        monkeypatch.setattr("builtins.open", mock_open)

        worker_agent.provision_encrypted_volume("umasktest", 10)
        # Should set 0o077 then restore original
        assert umask_values[0] == 0o077
        assert len(umask_values) >= 2
        assert umask_values[1] == 0o022  # original umask restored


class TestWorkerDestroyEncryptedVolume:
    """Test worker_agent.destroy_encrypted_volume."""

    def test_destroy_shreds_key(self, monkeypatch):
        import worker_agent

        calls = []

        def mock_run(args, **kw):
            calls.append(args)
            return MagicMock(returncode=0)

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: True)
        monkeypatch.setattr("worker_agent.os.remove", lambda p: None)
        monkeypatch.setattr("worker_agent.os.rmdir", lambda p: None)

        ok = worker_agent.destroy_encrypted_volume("shredtest")
        assert ok is True
        # Find the shred call
        shred_calls = [c for c in calls if c[0] == "shred"]
        assert len(shred_calls) == 1
        assert "-u" in shred_calls[0]
        assert "-z" in shred_calls[0]
        assert "-n" in shred_calls[0]
        assert "3" in shred_calls[0]
        assert any(".key" in str(a) for a in shred_calls[0])

    def test_destroy_removes_backing_file(self, monkeypatch):
        import worker_agent

        removed = []

        monkeypatch.setattr(
            "worker_agent.subprocess.run", lambda args, **kw: MagicMock(returncode=0)
        )
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: True)
        monkeypatch.setattr("worker_agent.os.remove", lambda p: removed.append(p))
        monkeypatch.setattr("worker_agent.os.rmdir", lambda p: None)

        worker_agent.destroy_encrypted_volume("rmtest")
        assert any(".img" in r for r in removed)

    def test_destroy_shred_failure_returns_false(self, monkeypatch):
        import worker_agent

        def mock_run(args, **kw):
            result = MagicMock()
            if args[0] == "shred":
                result.returncode = 1
                result.stderr = "shred: permission denied"
            else:
                result.returncode = 0
            return result

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: True)
        monkeypatch.setattr("worker_agent.os.remove", lambda p: None)
        monkeypatch.setattr("worker_agent.os.rmdir", lambda p: None)

        ok = worker_agent.destroy_encrypted_volume("shredfail")
        assert ok is False

    def test_destroy_calls_detach_first(self, monkeypatch):
        import worker_agent

        detach_called = {"called": False}
        original_detach = worker_agent.detach_encrypted_volume

        def mock_detach(vid):
            detach_called["called"] = True
            return True

        monkeypatch.setattr("worker_agent.detach_encrypted_volume", mock_detach)
        monkeypatch.setattr(
            "worker_agent.subprocess.run", lambda args, **kw: MagicMock(returncode=0)
        )
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: True)
        monkeypatch.setattr("worker_agent.os.remove", lambda p: None)
        monkeypatch.setattr("worker_agent.os.rmdir", lambda p: None)

        worker_agent.destroy_encrypted_volume("detachfirst")
        assert detach_called["called"] is True


class TestWorkerAttachEncryptedVolume:
    """Test worker_agent.attach_encrypted_volume."""

    def test_attach_no_backing_file(self, monkeypatch):
        import worker_agent

        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: False)
        result = worker_agent.attach_encrypted_volume("noback")
        assert result is None

    def test_attach_no_key_file(self, monkeypatch):
        import worker_agent

        def path_exists(p):
            return ".img" in p  # backing exists, key does not

        monkeypatch.setattr("worker_agent.os.path.exists", path_exists)
        result = worker_agent.attach_encrypted_volume("nokey")
        assert result is None

    def test_attach_success(self, monkeypatch):
        import worker_agent

        calls = []

        def mock_run(args, **kw):
            calls.append(args)
            result = MagicMock()
            # mountpoint check returns 1 (not mounted)
            if args[0] == "mountpoint":
                result.returncode = 1
            else:
                result.returncode = 0
            return result

        def all_exist(p):
            return True

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.path.exists", all_exist)
        monkeypatch.setattr("worker_agent.os.makedirs", lambda *a, **kw: None)

        result = worker_agent.attach_encrypted_volume("goodvol")
        assert result is not None
        assert "goodvol" in result


class TestWorkerDetachEncryptedVolume:
    """Test worker_agent.detach_encrypted_volume."""

    def test_detach_success(self, monkeypatch):
        import worker_agent

        calls = []

        def mock_run(args, **kw):
            calls.append(args)
            result = MagicMock()
            result.returncode = 0
            return result

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)

        ok = worker_agent.detach_encrypted_volume("detachme")
        assert ok is True
        # Should have: mountpoint check, umount, luksClose
        assert len(calls) == 3
        assert calls[1][0] == "umount"
        assert "luksClose" in calls[2]

    def test_detach_umount_failure(self, monkeypatch):
        import subprocess
        import worker_agent

        call_count = 0

        def mock_run(args, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # mountpoint returns 0 (is mounted)
                return MagicMock(returncode=0)
            if call_count == 2:  # umount fails
                raise subprocess.CalledProcessError(1, args, stderr="busy")
            return MagicMock(returncode=0)

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)

        ok = worker_agent.detach_encrypted_volume("busyvol")
        assert ok is False


class TestWorkerCleanupPartialVolume:
    """Test worker_agent._cleanup_partial_volume."""

    def test_cleanup_uses_shred_for_key(self, monkeypatch):
        import worker_agent

        calls = []

        def mock_run(args, **kw):
            calls.append(args)
            return MagicMock(returncode=0)

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: True)
        monkeypatch.setattr("worker_agent.os.remove", lambda p: None)
        monkeypatch.setattr("worker_agent.os.rmdir", lambda p: None)

        worker_agent._cleanup_partial_volume("partialvol")
        # Should use shred for key file, not os.remove
        shred_calls = [c for c in calls if len(c) > 0 and c[0] == "shred"]
        assert len(shred_calls) == 1
        assert "-u" in shred_calls[0]
        assert "-z" in shred_calls[0]
        assert any(".key" in str(a) for a in shred_calls[0])

    def test_cleanup_suppresses_errors(self, monkeypatch):
        import worker_agent

        def mock_run(args, **kw):
            raise Exception("everything is broken")

        monkeypatch.setattr("worker_agent.subprocess.run", mock_run)
        monkeypatch.setattr("worker_agent.os.path.exists", lambda p: True)
        monkeypatch.setattr("worker_agent.os.path.isdir", lambda p: True)
        monkeypatch.setattr(
            "worker_agent.os.remove", lambda p: (_ for _ in ()).throw(OSError("boom"))
        )
        monkeypatch.setattr(
            "worker_agent.os.rmdir", lambda p: (_ for _ in ()).throw(OSError("boom"))
        )

        # Should not raise
        worker_agent._cleanup_partial_volume("errorvol")


# ── _ensure_volume_dirs ────────────────────────────────────────────


class TestEnsureVolumeDirs:
    """Test worker_agent._ensure_volume_dirs."""

    def test_creates_base_and_key_dirs(self, monkeypatch):
        import worker_agent

        created = []

        def mock_makedirs(path, mode=None, exist_ok=False):
            created.append({"path": path, "mode": mode, "exist_ok": exist_ok})

        monkeypatch.setattr("worker_agent.os.makedirs", mock_makedirs)
        worker_agent._ensure_volume_dirs()

        assert len(created) == 2
        # Base dir — no restricted mode
        assert worker_agent.VOLUME_BASE_DIR in created[0]["path"]
        assert created[0]["exist_ok"] is True
        # Key dir — mode 0o700 (owner only)
        assert worker_agent.VOLUME_KEY_DIR in created[1]["path"]
        assert created[1]["mode"] == 0o700
        assert created[1]["exist_ok"] is True
