"""The root helper behind the unprivileged worker (worker_privileged, issue #65).

Commands are captured rather than run, so these tests need no root. The same code ran
for real on 2026-10-11 on the ASUS: LUKS provision/attach/detach/destroy, and attach
into a running container via open_tree/move_mount (rw and ro).
"""

import os
import stat
from types import SimpleNamespace

import pytest

import worker_privileged as wp


@pytest.fixture
def conf(tmp_path):
    return {
        **wp.load_config(str(tmp_path / "missing.conf")),
        "VOLUME_DIR": str(tmp_path / "volumes"),
        "VOLUME_KEY_DIR": str(tmp_path / "keys"),
    }


@pytest.fixture
def commands(monkeypatch):
    """Capture subprocess.run; per-command return codes via `commands.rc[name] = n`."""
    rec = SimpleNamespace(calls=[], rc={})

    def run(cmd, **kwargs):
        rec.calls.append(cmd)
        return SimpleNamespace(returncode=rec.rc.get(cmd[0], 0), stdout="", stderr=b"boom")

    monkeypatch.setattr(wp.subprocess, "run", run)
    return rec


def _req(op, **args):
    return {"v": wp.PROTOCOL, "op": op, "args": args}


def _names(calls):
    return [c[0] if c[0] != "cryptsetup" else f"cryptsetup {c[1]}" for c in calls]


# ── protocol ───────────────────────────────────────────────────────────


def test_protocol_mismatch_is_refused(conf):
    reply = wp.handle({"v": wp.PROTOCOL + 1, "op": "umount", "args": {}}, conf)
    assert reply["refused"] and "install.sh --update" in reply["error"]


def test_unknown_op_and_bad_args_are_refused(conf):
    assert wp.handle(_req("rm_rf"), conf)["refused"]
    assert wp.handle({"v": wp.PROTOCOL, "op": "umount", "args": ["x"]}, conf)["refused"]


def test_worker_without_helper_gets_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(wp, "SOCKET_PATH", str(tmp_path / "absent.sock"))
    monkeypatch.setattr(wp.os, "geteuid", lambda: 1000)
    reply = wp.call("umount", path="/mnt/xcelsior-nfs")
    assert not reply["ok"] and "install.sh --update" in reply["error"]


def test_pipe_caller_is_allowed_only_for_root(monkeypatch):
    r, w = os.pipe()
    try:
        monkeypatch.setattr(wp.os, "geteuid", lambda: 0)
        assert wp._peer_allowed(r)
        monkeypatch.setattr(wp.os, "geteuid", lambda: 1000)
        assert not wp._peer_allowed(r)
    finally:
        os.close(r)
        os.close(w)


# ── LUKS ───────────────────────────────────────────────────────────────


def test_provision_formats_luks2_aes_xts_512_and_mounts(conf, commands):
    reply = wp.handle(_req("luks_provision", volume_id="vol-abc123", size_gb=10), conf)
    assert reply["ok"], reply
    assert _names(commands.calls) == [
        "truncate",
        "cryptsetup luksFormat",
        "cryptsetup luksOpen",
        "mkfs.ext4",
        "mount",
    ]
    fmt = commands.calls[1]
    for flag, value in (
        ("--type", "luks2"),
        ("--cipher", "aes-xts-plain64"),
        ("--key-size", "512"),
        ("--hash", "sha256"),
    ):
        assert fmt[fmt.index(flag) + 1] == value
    assert commands.calls[0][:3] == ["truncate", "-s", "10G"]
    assert commands.calls[3][-1] == "/dev/mapper/xcelsior-vol-vol-abc123"
    assert reply["mount_point"] == os.path.join(conf["VOLUME_DIR"], "vol-abc123")


def test_provision_key_is_32_random_bytes_owner_only(conf, commands):
    wp.handle(_req("luks_provision", volume_id="keyvol", size_gb=1), conf)
    key = os.path.join(conf["VOLUME_KEY_DIR"], "keyvol.key")
    assert os.path.getsize(key) == 32
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(conf["VOLUME_KEY_DIR"]).st_mode) == 0o700


def test_provision_refuses_an_existing_volume_and_keeps_its_key(conf, commands):
    os.makedirs(conf["VOLUME_KEY_DIR"])
    key = os.path.join(conf["VOLUME_KEY_DIR"], "taken.key")
    open(key, "wb").write(b"k" * 32)
    reply = wp.handle(_req("luks_provision", volume_id="taken", size_gb=1), conf)
    assert reply["refused"] and os.path.exists(key) and commands.calls == []


def test_provision_failure_cleans_up_and_shreds_the_key(conf, commands):
    commands.rc["cryptsetup"] = 1
    reply = wp.handle(_req("luks_provision", volume_id="failfmt", size_gb=1), conf)
    assert not reply["ok"] and "cryptsetup" in reply["error"]
    assert [
        "shred",
        "-u",
        "-z",
        "-n",
        "3",
        os.path.join(conf["VOLUME_KEY_DIR"], "failfmt.key"),
    ] in commands.calls


@pytest.mark.parametrize("bad", ["../../etc", "a/b", "", "x" * 65, 7])
def test_volume_ids_are_validated(conf, commands, bad):
    assert wp.handle(_req("luks_provision", volume_id=bad, size_gb=1), conf)["refused"]
    assert commands.calls == []


@pytest.mark.parametrize("size", [0, -1, 16385, "10", 1.5])
def test_volume_size_is_validated(conf, commands, size):
    assert wp.handle(_req("luks_provision", volume_id="v1", size_gb=size), conf)["refused"]


def test_attach_needs_backing_file_and_key(conf, commands):
    assert wp.handle(_req("luks_attach", volume_id="nokey"), conf)["refused"]
    os.makedirs(conf["VOLUME_DIR"])
    os.makedirs(conf["VOLUME_KEY_DIR"])
    open(os.path.join(conf["VOLUME_DIR"], "v1.img"), "wb").close()
    open(os.path.join(conf["VOLUME_KEY_DIR"], "v1.key"), "wb").close()
    commands.rc["mountpoint"] = 1  # not mounted yet
    reply = wp.handle(_req("luks_attach", volume_id="v1"), conf)
    assert reply["ok"] and _names(commands.calls) == ["cryptsetup luksOpen", "mountpoint", "mount"]


def test_destroy_detaches_then_shreds_key_then_removes_backing(conf, commands, monkeypatch):
    os.makedirs(conf["VOLUME_DIR"])
    os.makedirs(conf["VOLUME_KEY_DIR"])
    backing = os.path.join(conf["VOLUME_DIR"], "v1.img")
    open(backing, "wb").close()
    open(os.path.join(conf["VOLUME_KEY_DIR"], "v1.key"), "wb").close()
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: True)
    real_exists = os.path.exists
    monkeypatch.setattr(
        wp.os.path, "exists", lambda p: p.startswith("/dev/mapper/") or real_exists(p)
    )
    assert wp.handle(_req("luks_destroy", volume_id="v1"), conf)["ok"]
    assert _names(commands.calls) == ["umount", "cryptsetup luksClose", "shred"]
    assert not real_exists(backing)


def test_destroy_keeps_the_backing_file_when_shred_fails(conf, commands):
    os.makedirs(conf["VOLUME_DIR"])
    os.makedirs(conf["VOLUME_KEY_DIR"])
    backing = os.path.join(conf["VOLUME_DIR"], "v1.img")
    open(backing, "wb").close()
    open(os.path.join(conf["VOLUME_KEY_DIR"], "v1.key"), "wb").close()
    commands.rc["shred"] = 1
    assert not wp.handle(_req("luks_destroy", volume_id="v1"), conf)["ok"]
    assert os.path.exists(backing)


def test_cleanup_is_best_effort(conf, monkeypatch):
    def explode(*a, **k):
        raise OSError("device busy")

    monkeypatch.setattr(wp.subprocess, "run", explode)
    assert wp.handle(_req("luks_cleanup", volume_id="v1"), conf)["ok"]


# ── NFS ─────────────────────────────────────────────────────────────────


@pytest.fixture
def no_mkdir(monkeypatch):
    monkeypatch.setattr(wp.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: False)


def test_volume_mount_uses_fixed_mount_point_and_safe_options(conf, commands, no_mkdir):
    reply = wp.handle(
        _req("nfs_mount", server="100.64.0.1", export="/exports/volumes/v1", volume_id="v1"), conf
    )
    assert reply["ok"] and reply["mount_point"] == "/mnt/xcelsior-volumes/v1"
    cmd = commands.calls[0]
    assert cmd[:3] == ["mount", "-t", "nfs"] and cmd[-2:] == [
        "100.64.0.1:/exports/volumes/v1",
        "/mnt/xcelsior-volumes/v1",
    ]
    opts = cmd[4].split(",")
    assert {"hard", "nosuid", "nodev"} <= set(opts) and "soft" not in opts


def test_options_from_config_gain_nosuid_nodev_and_select_nfs4(conf, commands, no_mkdir):
    conf["NFS_MOUNT_OPTS"] = "tcp,nfsvers=4.0,port=12049"
    wp.handle(_req("nfs_mount", server="100.64.0.1", export="/exports/models"), conf)
    cmd = commands.calls[0]
    assert cmd[2] == "nfs4" and cmd[-1] == "/mnt/xcelsior-nfs"
    assert cmd[4].split(",")[-2:] == ["nosuid", "nodev"]


@pytest.mark.parametrize(
    "server,export,why",
    [
        ("8.8.8.8", "/exports/v1", "not on the mesh"),
        ("nfs.evil.example", "/exports/v1", "not listed"),
        ("100.64.0.1", "/etc", "outside export roots"),
        ("100.64.0.1", "relative/path", "not absolute"),
        ("100.64.0.1;id", "/exports/v1", "shell metacharacters"),
    ],
)
def test_nfs_requests_are_validated(conf, commands, no_mkdir, server, export, why):
    assert wp.handle(_req("nfs_mount", server=server, export=export), conf)["refused"], why
    assert commands.calls == []


def test_listed_nfs_server_hostname_is_allowed(conf, commands, no_mkdir):
    conf["NFS_SERVERS"] = "nfs.xcelsior.internal"
    assert wp.handle(_req("nfs_mount", server="nfs.xcelsior.internal", export="/exports/m"), conf)[
        "ok"
    ]


def test_malformed_option_config_is_refused(conf, commands, no_mkdir):
    conf["NFS_MOUNT_OPTS"] = "hard,exec;id"
    assert wp.handle(_req("nfs_mount", server="100.64.0.1", export="/exports/m"), conf)["refused"]


def test_umount_only_xcelsior_mounts(conf, commands, monkeypatch):
    monkeypatch.setattr(wp, "_is_mountpoint", lambda p: True)
    assert wp.handle(_req("umount", path="/mnt/xcelsior-volumes/v1", lazy=True), conf)["ok"]
    assert commands.calls == [["umount", "-l", "/mnt/xcelsior-volumes/v1"]]
    for path in ("/", "/home", "/mnt/xcelsior-volumes", "/mnt/xcelsior-volumes/../../etc"):
        assert wp.handle(_req("umount", path=path), conf)["refused"], path


# ── attach into a running job ──────────────────────────────────────────


def _inspect_as(monkeypatch, stdout, rc=0):
    monkeypatch.setattr(
        wp.subprocess,
        "run",
        lambda cmd, **k: SimpleNamespace(returncode=rc, stdout=stdout, stderr=""),
    )


@pytest.mark.parametrize("stdout", ["true 4242 ", "false 0 true", "true 4242 false"])
def test_attach_only_into_running_managed_containers(conf, monkeypatch, stdout):
    _inspect_as(monkeypatch, stdout)
    reply = wp.handle(
        _req(
            "container_attach", container="xcl-job1", source="/mnt/xcelsior-volumes/v1", dst="/data"
        ),
        conf,
    )
    assert reply["refused"]


@pytest.mark.parametrize(
    "dst", ["/", "/etc/cron.d", "/proc/1", "/data/../etc", "relative", "/usr/bin/x", "/run/secrets"]
)
def test_attach_destination_is_validated(conf, monkeypatch, dst):
    _inspect_as(monkeypatch, "true 4242 true")
    reply = wp.handle(
        _req("container_attach", container="xcl-job1", source="/mnt/xcelsior-volumes/v1", dst=dst),
        conf,
    )
    assert reply["refused"], dst


def test_attach_source_must_be_a_real_xcelsior_mount(conf, monkeypatch, tmp_path):
    _inspect_as(monkeypatch, "true 4242 true")
    os.makedirs(conf["VOLUME_DIR"])
    os.symlink("/etc", os.path.join(conf["VOLUME_DIR"], "sneaky"))
    for source in ("/etc", "/home/aaryn", os.path.join(conf["VOLUME_DIR"], "sneaky")):
        reply = wp.handle(
            _req("container_attach", container="xcl-job1", source=source, dst="/data"), conf
        )
        assert reply["refused"], source


def test_attach_mode_is_rw_or_ro(conf, monkeypatch):
    _inspect_as(monkeypatch, "true 4242 true")
    reply = wp.handle(
        _req(
            "container_attach",
            container="xcl-job1",
            source="/mnt/xcelsior-nfs",
            dst="/data",
            mode="exec",
        ),
        conf,
    )
    assert reply["refused"]


# ── exclusive GPU ──────────────────────────────────────────────────────


def test_pause_stops_ollama_and_terms_tier3(conf, monkeypatch):
    sent = []
    monkeypatch.setattr(wp, "_pids", lambda match: [11, 12] if match is wp._is_ollama else [21])
    monkeypatch.setattr(wp.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    reply = wp.handle(_req("gpu_tenants", action="pause"), conf)
    assert reply == {"ok": True, "stopped": [11, 12]}
    assert sent == [(11, 19), (12, 19), (21, 15)]


def test_resume_only_continues_ollama_pids(conf, monkeypatch):
    sent = []
    monkeypatch.setattr(wp, "_is_ollama", lambda pid: pid == 11)
    monkeypatch.setattr(wp.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert wp.handle(_req("gpu_tenants", action="resume", pids=[11, 1]), conf)["ok"]
    assert sent == [(11, 18)]
    assert wp.handle(_req("gpu_tenants", action="resume", pids=["1"]), conf)["refused"]
    assert wp.handle(_req("gpu_tenants", action="kill"), conf)["refused"]
