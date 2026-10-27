"""The standby must run the primary's headscale build, and say so when it does not.

On 2026-10-25 the primary moved to a `main` build (commit 2d220b2e) and its
database was migrated to a schema v0.28 cannot open. The ASUS standby stayed on
v0.28.0. Replication kept succeeding, the replica stayed "promotable", and no
promotion could ever have started. Nothing reported it.

Replication now records the primary's build; promotion refuses a mismatch before
touching the live files; the watchdog pages once when they drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api" / "scripts"))

import headscale_failover as hf  # noqa: E402

PRIMARY_VERSION = (
    "headscale version v0.0.0-20261024134339-2d220b2eb8be\n"
    "commit: 2d220b2eb8be8d31ed603000d0b0be02c2cd94d0\n"
    "build time: 2026-10-24T13:43:39Z\n"
    "built with: go1.27.3 linux/amd64\n"
)
STANDBY_V028 = (
    "headscale version v0.28.0\n"
    "commit: 97fa117c48a58bee3941e5e8b4393873501afe14\n"
    "build time: 2026-02-04T20:26:22Z\n"
)


def test_a_build_is_named_by_its_commit() -> None:
    assert hf.headscale_identity(PRIMARY_VERSION) == "2d220b2eb8be8d31ed603000d0b0be02c2cd94d0"
    assert hf.headscale_identity("headscale version v0.29.6\n") == "v0.29.6"
    assert hf.headscale_identity("") == ""


def _make_db(path: Path) -> None:
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY)")
    con.execute("INSERT INTO users DEFAULT VALUES")
    con.executemany("INSERT INTO nodes DEFAULT VALUES", [()] * 6)
    con.commit()
    con.close()


def _primary(calls: list[list[str]], *, version: str = PRIMARY_VERSION):
    """A fake primary: answers the probe and `headscale version`, serves files over scp."""

    def run(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(argv)
        joined = " ".join(argv)
        if "db.sqlite && echo present" in joined:
            return subprocess.CompletedProcess(argv, 0, "present\n", "")
        if argv[-1] == "headscale version":
            return subprocess.CompletedProcess(argv, 0, version, "")
        if argv[0] == "scp":
            dest = Path(argv[-1])
            if argv[-2].endswith(".sqlite"):
                _make_db(dest)
            else:
                dest.write_bytes(b"material")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


def test_replication_records_the_primarys_build_and_its_dns_token(tmp_path: Path) -> None:
    settings = hf.Settings(ssh_key=str(tmp_path / "key"), replica_dir=tmp_path / "replica")
    calls: list[list[str]] = []

    status = hf.replicate(settings, runner=_primary(calls))

    meta = json.loads((settings.replica_dir / "replica-meta.json").read_text())
    assert meta["primary_headscale"] == "2d220b2eb8be8d31ed603000d0b0be02c2cd94d0"
    assert status.primary_headscale == meta["primary_headscale"]
    assert any(a[0] == "scp" and a[-2].endswith(":/etc/headscale/cloudflare.token") for a in calls)
    assert (settings.replica_dir / "cloudflare.token").read_bytes() == b"material"


def _status(tmp_path: Path, primary: str) -> tuple[hf.Settings, hf.ReplicaStatus]:
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "noise_private.key").write_text("privkey:deadbeef\n")
    settings = hf.Settings(
        replica_dir=replica,
        live_data_dir=tmp_path / "live",
        live_config_dir=tmp_path / "conf",
        cloudflare_zone_id="zone",
        cloudflare_token="tok",
    )
    status = hf.ReplicaStatus(
        sqlite_path=replica / "db.sqlite",
        noise_key_path=replica / "noise_private.key",
        user_count=1,
        node_count=6,
        replicated_at="2026-10-27T06:45:00+00:00",
        promotable=True,
        reason="",
        primary_headscale=primary,
    )
    return settings, status


def _standby(version: str, commands: list[list[str]]):
    def run(argv, **kwargs):  # noqa: ANN001, ANN003
        commands.append(argv)
        if argv == [hf.HEADSCALE_BIN, "version"]:
            return subprocess.CompletedProcess(argv, 0, version, "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


def test_promotion_refuses_when_the_standby_binary_differs_from_the_primary(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    settings, status = _status(tmp_path, primary=hf.headscale_identity(PRIMARY_VERSION))
    installed: list[str] = []
    moved: list[str] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(hf, "install_replica_into_live", lambda *a, **k: installed.append("x"))
    monkeypatch.setattr(hf, "set_dns_a", lambda *a, **k: moved.append("x"))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, _m: None)

    with pytest.raises(hf.FailoverError) as excinfo:
        hf.promote(
            settings,
            replica=status,
            public_ip="207.219.125.161",
            runner=_standby(STANDBY_V028, commands),
        )

    assert "97fa117c48a5" in str(excinfo.value) and "2d220b2eb8be" in str(excinfo.value)
    assert not installed, "live files were replaced before the build check"
    assert not [c for c in commands if c[0] == "systemctl"], f"touched the service: {commands}"
    assert not moved


def test_promotion_proceeds_on_the_primarys_build(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    settings, status = _status(tmp_path, primary=hf.headscale_identity(PRIMARY_VERSION))
    moved: list[str] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(hf, "install_replica_into_live", lambda *a, **k: None)
    monkeypatch.setattr(hf, "set_dns_a", lambda _s, ip, **k: moved.append(ip))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, _m: None)
    monkeypatch.setattr(hf, "standby_serves_dns_name", lambda *a, **k: (True, ""))

    hf.promote(
        settings,
        replica=status,
        public_ip="207.219.125.161",
        runner=_standby(PRIMARY_VERSION, commands),
    )

    assert moved == ["207.219.125.161"]
    systemctl = [c[1] for c in commands if c[0] == "systemctl"]
    # Stopped before the swap, enabled so a reboot keeps the control plane, then started.
    assert systemctl == ["stop", "enable", "restart"], systemctl


def test_the_watchdog_pages_once_when_the_builds_drift_and_once_when_they_match(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    settings = hf.Settings(ssh_key=str(tmp_path / "key"), replica_dir=tmp_path / "replica")
    sent: list[str] = []
    monkeypatch.setattr(hf, "telegram_send", lambda _s, m: sent.append(m))
    healthy = hf.Health(dns_ok=True, primary_ip_ok=True)
    standby_build = {"out": STANDBY_V028}

    def runner(argv, **kwargs):  # noqa: ANN001, ANN003
        if argv == [hf.HEADSCALE_BIN, "version"]:
            return subprocess.CompletedProcess(argv, 0, standby_build["out"], "")
        return _primary([])(argv, **kwargs)

    state = hf.FailoverState()
    for _ in range(4):
        _, state = hf.watchdog_tick(
            settings, health=healthy, state=state, execute=True, runner=runner
        )
    drift = [m for m in sent if "cannot be promoted" in m]
    assert len(drift) == 1, f"paged {len(drift)}x for one drift: {sent}"
    assert state.version_alert_active

    standby_build["out"] = PRIMARY_VERSION
    _, state = hf.watchdog_tick(settings, health=healthy, state=state, execute=True, runner=runner)
    assert any("matches the primary again" in m for m in sent)
    assert not state.version_alert_active


def test_failback_stops_and_disables_the_standby(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """Left running, the standby is a second control plane renewing certificates on the same zone."""
    settings, status = _status(tmp_path, primary="")
    (settings.replica_dir / "db.sqlite").write_bytes(b"")
    commands: list[list[str]] = []
    monkeypatch.setattr(hf, "set_dns_a", lambda *a, **k: None)
    monkeypatch.setattr(hf, "telegram_send", lambda _s, _m: None)
    monkeypatch.setattr(hf, "sqlite_counts", lambda _p: (1, 6))
    monkeypatch.setattr(hf, "serves_dns_name", lambda _s, _a: (True, ""))
    monkeypatch.setattr(hf, "primary_summary", lambda *a, **k: hf.PrimarySummary(missing=True))

    hf.failback(
        settings,
        replica=status,
        state=hf.FailoverState(role="promoted", original_a_record="46.225.20.97"),
        health=hf.Health(dns_ok=False, primary_ip_ok=True),
        runner=_standby("", commands),
    )

    assert ["systemctl", "disable", "--now", "headscale"] in commands
