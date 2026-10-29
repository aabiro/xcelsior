"""A promoted standby must be able to read what the promotion gave it.

The ASUS standby crash-looped 42,000+ times from 2026-10-19 on "Error loading
config": `/etc/headscale/config.yaml` was `root:root 0600`, and headscale.service
runs as `headscale`. `copy_file` wrote every promoted file that way — database,
keys, config and ACL — so any promotion produced a control plane that could not
start, discovered only after DNS was already in question.

These pin the ownership a promotion installs, and that `promote` refuses before
restarting anything when the service user still cannot read its files.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api" / "scripts"))

import headscale_failover as hf  # noqa: E402


@pytest.fixture(autouse=True)
def _not_root(monkeypatch):
    """Promotion chowns to the `headscale` user only as root; that user exists
    only on the Headscale hosts. A suite run as root elsewhere (a container)
    would try the chown, so these run as the unprivileged operator they model.
    The root path is covered with its own patched euid and chown."""
    import os

    monkeypatch.setattr(os, "geteuid", lambda: 1000)

PRIMARY_CONFIG = {
    "server_url": "https://hs.xcelsior.ca",
    "listen_addr": "127.0.0.1:8443",
    "derp": {
        "server": {"enabled": True, "region_id": 999, "ipv4": "46.225.20.97"},
        "urls": ["https://controlplane.tailscale.com/derpmap/default"],
    },
    "unix_socket_permission": "0770",
}


def _replica(tmp_path: Path) -> tuple[hf.Settings, hf.ReplicaStatus]:
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "db.sqlite").write_bytes(b"new database")
    (replica / "noise_private.key").write_text("privkey:deadbeef\n")
    (replica / "derp_server_private.key").write_text("privkey:derp\n")
    (replica / "acl.json").write_text("{}\n")
    (replica / "cloudflare.token").write_text("cf-token\n")
    (replica / "config.yaml").write_text(yaml.safe_dump(PRIMARY_CONFIG))
    settings = hf.Settings(
        replica_dir=replica,
        live_data_dir=tmp_path / "var-lib-headscale",
        live_config_dir=tmp_path / "etc-headscale",
        login_server="https://hs.xcelsior.ca",
    )
    status = hf.ReplicaStatus(
        sqlite_path=replica / "db.sqlite",
        noise_key_path=replica / "noise_private.key",
        user_count=1,
        node_count=6,
        replicated_at="2026-10-27T06:45:00+00:00",
        promotable=True,
        reason="",
    )
    return settings, status


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_promoted_files_get_the_modes_the_service_can_use(tmp_path: Path) -> None:
    settings, status = _replica(tmp_path)
    live, conf = settings.live_data_dir, settings.live_config_dir
    live.mkdir()
    # A WAL left by the database being replaced must not be replayed into the new one.
    (live / "db.sqlite-wal").write_bytes(b"stale wal")
    (live / "db.sqlite-shm").write_bytes(b"stale shm")

    hf.install_replica_into_live(settings, status)

    assert (live / "db.sqlite").read_bytes() == b"new database"
    assert not (live / "db.sqlite-wal").exists() and not (live / "db.sqlite-shm").exists()
    for state_file in ("db.sqlite", "noise_private.key", "derp_server_private.key"):
        assert _mode(live / state_file) == 0o600, state_file
    for config_file in ("config.yaml", "acl.json", "cloudflare.token"):
        assert _mode(conf / config_file) == 0o640, config_file
    assert (conf / "cloudflare.token").read_text() == "cf-token\n"


def test_copy_file_gives_the_file_to_the_service_user_when_root(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    chowned: list[tuple[str, str | None, str | None]] = []
    monkeypatch.setattr(hf.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        hf.shutil,
        "chown",
        lambda p, user=None, group=None: chowned.append((Path(p).name, user, group)),
    )
    src = tmp_path / "config.yaml"
    src.write_text("x: 1\n")

    hf.copy_file(src, tmp_path / "out" / "config.yaml", mode=0o640, owner="root", group="headscale")

    assert chowned == [("config.yaml.tmp", "root", "headscale")]
    assert _mode(tmp_path / "out" / "config.yaml") == 0o640


def test_the_primarys_derp_relay_is_not_advertised_from_the_standby(tmp_path: Path) -> None:
    """Region 999 names the primary's address; served from here it points at the dead host."""
    settings, status = _replica(tmp_path)

    hf.install_replica_into_live(settings, status)

    installed = yaml.safe_load((settings.live_config_dir / "config.yaml").read_text())
    assert installed["derp"]["server"]["enabled"] is False
    assert installed["derp"]["urls"] == PRIMARY_CONFIG["derp"]["urls"]
    assert installed["server_url"] == "https://hs.xcelsior.ca"
    # Rewritten as YAML, the socket mode must stay a string, not become octal 504.
    assert installed["unix_socket_permission"] == "0770"
    assert _mode(settings.live_config_dir / "config.yaml") == 0o640


def test_promotion_refuses_when_headscale_cannot_read_its_config(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    settings, status = _replica(tmp_path)
    commands: list[list[str]] = []
    moved: list[str] = []
    monkeypatch.setattr(hf, "set_dns_a", lambda *a, **k: moved.append("moved"))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, _m: None)

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        commands.append(argv)
        if argv[:2] == ["runuser", "-u"] and argv[-1].endswith("config.yaml"):
            return subprocess.CompletedProcess(argv, 1, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    with pytest.raises(hf.FailoverError) as excinfo:
        hf.promote(settings, replica=status, public_ip="207.219.125.161", runner=runner)

    assert "cannot read" in str(excinfo.value) and "config.yaml" in str(excinfo.value)
    assert not [c for c in commands if c[:2] == ["systemctl", "restart"]], "restarted anyway"
    assert not moved, "DNS moved to a standby that cannot load its config"
