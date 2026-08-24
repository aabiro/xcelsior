"""Headscale failover must preserve the tailnet identity across a VPS outage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.headscale_failover import (
    FailoverError,
    FailoverState,
    Health,
    ReplicaStatus,
    Settings,
    decide,
    failback_ready,
    inspect_replica,
    read_dotenv_values,
    settings_from_env,
    sqlite_counts,
    watchdog_tick,
    write_replica_meta,
)


def _sqlite_with(path: Path, *, users: int, nodes: int) -> None:
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE nodes (id INTEGER PRIMARY KEY, hostname TEXT);
        """)
    for i in range(users):
        con.execute("INSERT INTO users(name) VALUES (?)", (f"user-{i}",))
    for i in range(nodes):
        con.execute("INSERT INTO nodes(hostname) VALUES (?)", (f"node-{i}",))
    con.commit()
    con.close()


def _replica(
    tmp_path: Path, *, users: int, nodes: int, noise: bool = True
) -> tuple[Path, ReplicaStatus]:
    replica_dir = tmp_path / "replica"
    replica_dir.mkdir()
    _sqlite_with(replica_dir / "db.sqlite", users=users, nodes=nodes)
    if noise:
        (replica_dir / "noise_private.key").write_bytes(b"noise-key")
    write_replica_meta(replica_dir, users, nodes, source="test")
    return replica_dir, inspect_replica(replica_dir)


def test_dotenv_is_data_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "CLOUDFLARE_ZONE_ID=zone-1",
                "UNRELATED=$(touch %s)" % marker,
                "export XCELSIOR_HEADSCALE_HOST=203.0.113.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    values = read_dotenv_values(env)
    assert values["CLOUDFLARE_ZONE_ID"] == "zone-1"
    assert values["XCELSIOR_HEADSCALE_HOST"] == "203.0.113.9"
    assert not marker.exists()


def test_settings_prefer_environ_over_dotenv(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("XCELSIOR_HEADSCALE_HOST=10.0.0.1\nCLOUDFLARE_ZONE_ID=from-file\n")
    settings = settings_from_env(
        {
            "XCELSIOR_HEADSCALE_HOST": "10.0.0.8",
            "XCELSIOR_HEADSCALE_SSH_KEY": "/home/aaryn/.ssh/id_ed25519",
            "XCELSIOR_ENV_FILE": str(env),
        },
        dotenv_path=env,
    )
    assert settings.primary_host == "10.0.0.8"
    assert settings.cloudflare_zone_id == "from-file"
    assert settings.ssh_key == "/home/aaryn/.ssh/id_ed25519"


def test_empty_replica_is_not_promotable(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=0, nodes=0)
    assert replica.promotable is False
    assert "empty" in replica.reason
    assert "100.64.0.0/10" in replica.reason


def test_replica_without_noise_key_is_not_promotable(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=3, noise=False)
    assert replica.promotable is False
    assert "noise_private.key" in replica.reason


def test_populated_replica_is_promotable(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=5)
    assert replica.promotable is True
    assert replica.user_count == 1
    assert replica.node_count == 5


def test_sqlite_counts_old_machines_table(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE users (id INTEGER PRIMARY KEY); CREATE TABLE machines (id INTEGER PRIMARY KEY);"
        "INSERT INTO users DEFAULT VALUES; INSERT INTO machines DEFAULT VALUES;"
        "INSERT INTO machines DEFAULT VALUES;"
    )
    con.commit()
    con.close()
    users, nodes = sqlite_counts(path)
    assert users == 1
    assert nodes == 2


def test_watchdog_waits_until_failure_threshold(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=2)
    health = Health(dns_ok=False, primary_ip_ok=False)
    state = FailoverState(role="standby", consecutive_failures=1)
    decision = decide(health=health, replica=replica, state=state, failure_threshold=3)
    assert decision.action == "wait"


def test_watchdog_promotes_after_threshold_with_replica(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=2)
    health = Health(dns_ok=False, primary_ip_ok=False)
    state = FailoverState(role="standby", consecutive_failures=3)
    decision = decide(health=health, replica=replica, state=state, failure_threshold=3)
    assert decision.action == "promote"
    assert decision.alert is True


def test_watchdog_refuses_to_promote_empty_replica(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=0, nodes=0)
    health = Health(dns_ok=False, primary_ip_ok=False)
    state = FailoverState(role="standby", consecutive_failures=5)
    decision = decide(health=health, replica=replica, state=state, failure_threshold=3)
    assert decision.action == "refuse_promote"
    assert decision.alert is True


def test_healthy_primary_schedules_replicate_not_promote(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=2)
    health = Health(dns_ok=True, primary_ip_ok=True)
    state = FailoverState(role="standby", consecutive_failures=9)
    decision = decide(health=health, replica=replica, state=state, failure_threshold=3)
    assert decision.action == "replicate"


def test_promoted_standby_does_not_auto_failback(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=2)
    health = Health(dns_ok=True, primary_ip_ok=True)
    state = FailoverState(role="promoted")
    decision = decide(health=health, replica=replica, state=state, failure_threshold=3)
    assert decision.action == "alert_primary_back"


def test_failback_refuses_when_vps_still_down(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=2)
    decision = failback_ready(
        health=Health(dns_ok=True, primary_ip_ok=False, primary_ip_error="timed out"),
        replica=replica,
        state=FailoverState(role="promoted"),
    )
    assert decision.action == "refuse_failback"
    assert "DNS must not flip yet" in decision.reason


def test_failback_refuses_empty_replica_even_if_vps_is_up(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=0, nodes=0)
    decision = failback_ready(
        health=Health(dns_ok=False, primary_ip_ok=True),
        replica=replica,
        state=FailoverState(role="promoted"),
    )
    assert decision.action == "refuse_failback"
    assert "wipe" in decision.reason


def test_failback_ready_when_vps_up_and_replica_has_identity(tmp_path: Path) -> None:
    _, replica = _replica(tmp_path, users=1, nodes=4)
    decision = failback_ready(
        health=Health(dns_ok=True, primary_ip_ok=True),
        replica=replica,
        state=FailoverState(role="promoted"),
    )
    assert decision.action == "failback"
    assert "before" not in decision.reason.lower() or "copy" in decision.reason.lower()


def test_watchdog_tick_counts_failures_without_executing(tmp_path: Path) -> None:
    replica_dir, replica = _replica(tmp_path, users=1, nodes=2)
    settings = Settings(replica_dir=replica_dir, failure_threshold=3)
    health = Health(dns_ok=False, primary_ip_ok=False)
    state = FailoverState(role="standby", consecutive_failures=0)
    decision, new_state = watchdog_tick(
        settings, health=health, replica=replica, state=state, execute=False
    )
    assert decision.action == "wait"
    assert new_state.consecutive_failures == 1
    saved = json.loads((replica_dir / "failover-state.json").read_text())
    assert saved["consecutive_failures"] == 1


def test_promote_function_refuses_empty_replica(tmp_path: Path) -> None:
    from scripts.headscale_failover import promote

    replica_dir, replica = _replica(tmp_path, users=0, nodes=0)
    settings = Settings(replica_dir=replica_dir, live_data_dir=tmp_path / "live")
    with pytest.raises(FailoverError, match="empty"):
        promote(settings, replica=replica, public_ip="203.0.113.10")


def test_promote_rewrites_localhost_server_url(tmp_path: Path) -> None:
    from scripts.headscale_failover import install_replica_into_live

    replica_dir, replica = _replica(tmp_path, users=1, nodes=2)
    (replica_dir / "config.yaml").write_text(
        "server_url: http://127.0.0.1:8080\nlisten_addr: 127.0.0.1:8080\n"
    )
    live = tmp_path / "live"
    config_dir = tmp_path / "cfg"
    settings = Settings(
        replica_dir=replica_dir,
        live_data_dir=live,
        live_config_dir=config_dir,
        login_server="https://hs.xcelsior.ca",
    )
    install_replica_into_live(settings, replica)
    text = (config_dir / "config.yaml").read_text()
    assert "server_url: https://hs.xcelsior.ca" in text
    assert "127.0.0.1:8080" in text  # listen_addr stays local
    assert (live / "db.sqlite").is_file()
    assert (live / "noise_private.key").read_bytes() == b"noise-key"
