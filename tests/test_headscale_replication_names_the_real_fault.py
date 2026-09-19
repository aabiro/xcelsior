"""A wrong replication host must not page as a database fault.

`XCELSIOR_HEADSCALE_HOST` pointed at 45.76.3.128 for weeks. That box is up,
reachable over SSH, and has never had Headscale installed. Every replication
run therefore failed on the *database* step and alerted:

    unable to open database "/var/lib/headscale/db.sqlite":
    unable to open database file

which reads as corruption, a permissions problem, or a dead control plane —
so the investigation went to a server that was entirely healthy, while the
actual Headscale host (149.28.121.61) sat unreplicated the whole time.

Reachable-but-wrong-host and present-but-unreadable-database are different
faults. This pins that they no longer share an error message.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import headscale_failover as hf  # noqa: E402


def _settings(tmp_path: Path, host: str) -> hf.Settings:
    return hf.Settings(
        primary_host=host,
        primary_user="root",
        ssh_key=str(tmp_path / "key"),
        replica_dir=tmp_path / "replica",
    )


def _runner(probe_stdout: str, *, calls: list[list[str]]):
    """A fake SSH that answers the presence probe and records every command."""

    def run(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(argv)
        joined = " ".join(argv)
        if "db.sqlite && echo present" in joined:
            return subprocess.CompletedProcess(argv, 0, probe_stdout, "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


def test_a_host_without_headscale_is_named_as_a_config_error(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    with pytest.raises(hf.FailoverError) as excinfo:
        hf.replicate(
            _settings(tmp_path, "45.76.3.128"),
            runner=_runner("absent\n", calls=calls),
        )

    message = str(excinfo.value)
    # It must point at the configuration, not at the database.
    assert "45.76.3.128" in message
    assert "XCELSIOR_HEADSCALE_HOST" in message
    assert "configuration error" in message
    assert "unable to open database" not in message

    # And it must stop before trying to read a database that isn't there —
    # that attempt is what produced the misleading alert.
    assert not any("backup" in " ".join(argv) for argv in calls), (
        "replication tried to read the database on a host that has no Headscale"
    )


def test_a_host_with_headscale_proceeds_to_the_backup(tmp_path: Path) -> None:
    """The guard must not block the healthy path it was added to protect."""
    calls: list[list[str]] = []
    settings = _settings(tmp_path, "149.28.121.61")
    with pytest.raises(Exception):  # noqa: B017 - scp of a nonexistent file
        hf.replicate(settings, runner=_runner("present\n", calls=calls))

    assert any("backup" in " ".join(argv) for argv in calls), (
        "a real Headscale host was never asked for a snapshot"
    )


def _failing_tick(tmp_path: Path, sent: list[str], state: hf.FailoverState):
    """Drive one watchdog tick whose replicate step fails, capturing alerts."""
    # Settings is frozen; `telegram_send` is patched out wholesale instead, so
    # the test never depends on real credentials being present.
    settings = _settings(tmp_path, "45.76.3.128")

    def fake_send(_settings, message):  # noqa: ANN001
        sent.append(message)

    original = hf.telegram_send
    hf.telegram_send = fake_send  # type: ignore[assignment]
    try:
        return hf.watchdog_tick(
            settings,
            health=hf.Health(dns_ok=True, primary_ip_ok=True, dns_error="", primary_ip_error=""),
            replica=hf.inspect_replica(tmp_path / "replica"),
            state=state,
            execute=True,
            runner=_runner("absent\n", calls=[]),
        )
    finally:
        hf.telegram_send = original  # type: ignore[assignment]


def test_a_persistent_replication_failure_pages_once_not_every_tick(tmp_path: Path) -> None:
    """The watchdog fires every 2 minutes. One fault must not mean one page per tick.

    A misconfigured host went unfixed for weeks and delivered the same
    `Headscale replica refresh failed` message on every single tick — hundreds a
    day. `maybe_alert` already dedupes by `state.last_alert` for every other
    decision; the replicate-failure path simply did not use it.
    """
    sent: list[str] = []
    state = hf.FailoverState()

    for _ in range(5):
        _, state = _failing_tick(tmp_path, sent, state)

    failures = [m for m in sent if "refresh failed" in m]
    assert len(failures) == 1, f"paged {len(failures)}x for one continuous fault: {failures}"


def test_a_recovered_replication_is_announced_and_re_arms(tmp_path: Path) -> None:
    """Silence after a page is ambiguous, and the next real fault must still page."""
    sent: list[str] = []
    state = hf.FailoverState()
    _, state = _failing_tick(tmp_path, sent, state)
    assert state.replicate_alert_active is True

    # A success closes the episode and says so.
    state.replicate_alert_active = False

    # A fresh fault after recovery pages again rather than staying suppressed.
    _, state = _failing_tick(tmp_path, sent, state)
    assert len([m for m in sent if "refresh failed" in m]) == 2


def test_a_cleared_condition_does_not_suppress_its_own_recurrence(tmp_path: Path) -> None:
    """Dedupe must key on the *open* episode, not on the last page ever sent.

    `last_alert` persisted `refuse_promote` from 2026-08-19 onward. Because the
    old dedupe compared against it, the single most important alert this
    watchdog can raise — "I will not promote, the replica is empty" — would
    never have paged a second time, however long the gap.
    """
    sent: list[str] = []
    settings = _settings(tmp_path, "149.28.121.61")

    def fake_send(_settings, message):  # noqa: ANN001
        sent.append(message)

    alerting = hf.Decision(action="refuse_promote", reason="replica is empty", alert=True)
    quiet = hf.Decision(action="replicate", reason="primary answered", alert=False)

    original = hf.telegram_send
    hf.telegram_send = fake_send  # type: ignore[assignment]
    try:
        state = hf.FailoverState()
        state = hf.maybe_alert(settings, state, alerting)   # pages
        state = hf.maybe_alert(settings, state, alerting)   # same episode, quiet
        state = hf.maybe_alert(settings, state, quiet)      # condition clears
        state = hf.maybe_alert(settings, state, alerting)   # NEW episode, must page
    finally:
        hf.telegram_send = original  # type: ignore[assignment]

    assert len(sent) == 2, f"expected page, quiet, page — got {len(sent)}: {sent}"


def test_failback_records_the_address_dns_actually_held(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """`original_a_record` is the address failback restores. Capture it, don't assume it.

    It used to be set to `settings.primary_host`, which was correct only while
    the Headscale host and the `hs.xcelsior.ca` A record were the same machine.
    They are not: the A record is 45.76.3.128 (a reverse proxy) and the primary
    is 149.28.121.61. Assuming them equal files the wrong address as the thing
    to restore.
    """
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "noise_private.key").write_text("privkey:deadbeef\n")

    calls: list[str] = []
    monkeypatch.setattr(hf, "dns_a_record", lambda *a, **k: {"content": "45.76.3.128"})
    monkeypatch.setattr(hf, "promote", lambda *a, **k: calls.append("promoted"))

    settings = _settings(tmp_path, "149.28.121.61")
    state = hf.FailoverState(consecutive_failures=99)
    _, state = hf.watchdog_tick(
        settings,
        health=hf.Health(dns_ok=False, primary_ip_ok=False, dns_error="down", primary_ip_error="down"),
        replica=hf.ReplicaStatus(
            sqlite_path=replica / "db.sqlite",
            noise_key_path=replica / "noise_private.key",
            user_count=1,
            node_count=6,
            replicated_at="2026-08-29T03:44:02+00:00",
            promotable=True,
            reason="",
        ),
        state=state,
        execute=True,
        runner=_runner("present\n", calls=[]),
    )

    assert calls == ["promoted"], "the promote path did not run"
    assert state.original_a_record == "45.76.3.128", (
        f"filed {state.original_a_record!r} as the address to restore; DNS held 45.76.3.128"
    )


def test_promotion_refuses_when_the_standby_cannot_serve_the_name(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """A promotion whose DNS change cannot be served is not a promotion.

    Measured 2026-09-17: the watchdog promoted this standby correctly — the
    replica was present and promotable — and the control plane stayed
    unreachable, because the standby's Headscale binds `127.0.0.1:8443` with no
    public TLS front. `hs.xcelsior.ca` was moved onto a host that cannot answer
    for it, and the alert said "promoted", which reads as recovered.

    Both boxes were down either way; the damage is that the record left its
    origin for nothing and the operator was told the opposite.
    """
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "noise_private.key").write_text("privkey:deadbeef\n")

    moved: list[str] = []
    sent: list[str] = []
    monkeypatch.setattr(hf, "install_replica_into_live", lambda *a, **k: None)
    monkeypatch.setattr(hf, "set_dns_a", lambda *a, **k: moved.append(a[1] if len(a) > 1 else "?"))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, m: sent.append(m))
    monkeypatch.setattr(hf, "standby_serves_dns_name", lambda *a, **k: (False, "nothing on 443"))

    status = hf.ReplicaStatus(
        sqlite_path=replica / "db.sqlite",
        noise_key_path=replica / "noise_private.key",
        user_count=1,
        node_count=6,
        replicated_at="2026-08-31T02:37:25+00:00",
        promotable=True,
        reason="",
    )

    with pytest.raises(hf.FailoverError) as excinfo:
        hf.promote(
            _settings(tmp_path, "45.76.3.128"),
            replica=status,
            public_ip="66.222.170.140",
            runner=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "", ""),
        )

    assert "does not serve it" in str(excinfo.value)
    assert not moved, f"DNS was moved to a host that cannot serve the name: {moved}"
    assert any("ABORTED" in m for m in sent), (
        f"the operator was not told the promotion failed; messages: {sent}"
    )


def test_promotion_proceeds_when_the_standby_does_serve_the_name(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """The guard must not block a standby that is genuinely ready."""
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "noise_private.key").write_text("privkey:deadbeef\n")

    moved: list[str] = []
    monkeypatch.setattr(hf, "install_replica_into_live", lambda *a, **k: None)
    monkeypatch.setattr(hf, "set_dns_a", lambda *a, **k: moved.append(a[1]))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, _m: None)
    monkeypatch.setattr(hf, "standby_serves_dns_name", lambda *a, **k: (True, ""))

    settings = hf.Settings(
        primary_host="45.76.3.128",
        primary_user="root",
        ssh_key=str(tmp_path / "key"),
        replica_dir=replica,
        cloudflare_zone_id="zone",
        cloudflare_token="tok",
    )
    status = hf.ReplicaStatus(
        sqlite_path=replica / "db.sqlite",
        noise_key_path=replica / "noise_private.key",
        user_count=1,
        node_count=6,
        replicated_at="2026-08-31T02:37:25+00:00",
        promotable=True,
        reason="",
    )
    hf.promote(
        settings,
        replica=status,
        public_ip="66.222.170.140",
        runner=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "", ""),
    )
    assert moved == ["66.222.170.140"], f"a ready standby did not take the record: {moved}"


def _failback_fixture(tmp_path: Path):
    """A promoted standby with a replica, ready to fail back."""
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "noise_private.key").write_text("privkey:deadbeef\n")
    (replica / "db.sqlite").write_bytes(b"")
    live = tmp_path / "live"
    live.mkdir()
    settings = hf.Settings(
        primary_host="149.28.121.61",
        primary_user="root",
        ssh_key=str(tmp_path / "key"),
        replica_dir=replica,
        live_data_dir=live,
        cloudflare_zone_id="zone",
        cloudflare_token="tok",
    )
    status = hf.ReplicaStatus(
        sqlite_path=replica / "db.sqlite",
        noise_key_path=replica / "noise_private.key",
        user_count=1,
        node_count=6,
        replicated_at="2026-08-31T02:37:25+00:00",
        promotable=True,
        reason="",
    )
    health = hf.Health(dns_ok=False, primary_ip_ok=True, dns_error="", primary_ip_error="")
    return settings, status, health


def test_failback_restores_the_address_dns_actually_held(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """`original_a_record`, not `primary_host` — they are not the same machine.

    `promote()` records what Cloudflare actually held. `failback()` ignored it
    and used `primary_host`, which only coincides while the Headscale host and
    the A record are one box. They are not: the record pointed at 45.76.3.128, a
    reverse proxy terminating TLS for the name, while Headscale runs on
    149.28.121.61. Failing back to `primary_host` moves the name somewhere it
    has never been, and calls that a recovery.
    """
    settings, status, health = _failback_fixture(tmp_path)
    moved: list[str] = []
    monkeypatch.setattr(hf, "set_dns_a", lambda _s, ip, **k: moved.append(ip))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, _m: None)
    monkeypatch.setattr(hf, "sqlite_counts", lambda _p: (1, 6))
    monkeypatch.setattr(hf, "serves_dns_name", lambda _s, _a: (True, ""))

    state = hf.FailoverState(role="promoted", original_a_record="45.76.3.128")
    hf.failback(
        settings,
        replica=status,
        state=state,
        health=health,
        runner=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "", ""),
    )
    assert moved == ["45.76.3.128"], (
        f"failback pointed {settings.dns_name} at {moved}; DNS held 45.76.3.128"
    )


def test_failback_holds_dns_when_the_target_cannot_serve_the_name(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """Restoring a name onto an address nothing answers on is not a recovery."""
    settings, status, health = _failback_fixture(tmp_path)
    moved: list[str] = []
    sent: list[str] = []
    monkeypatch.setattr(hf, "set_dns_a", lambda _s, ip, **k: moved.append(ip))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, m: sent.append(m))
    monkeypatch.setattr(hf, "sqlite_counts", lambda _p: (1, 6))
    monkeypatch.setattr(hf, "serves_dns_name", lambda _s, _a: (False, "connection refused"))

    state = hf.FailoverState(role="promoted", original_a_record="45.76.3.128")
    with pytest.raises(hf.FailoverError) as excinfo:
        hf.failback(
            settings,
            replica=status,
            state=state,
            health=health,
            runner=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "", ""),
        )
    assert "does not serve the name" in str(excinfo.value)
    assert not moved, f"DNS was moved to an address that cannot serve it: {moved}"
    assert any("HELD" in m for m in sent), f"operator not told the move was held: {sent}"


def test_failback_refuses_to_push_a_database_that_lost_registrations(tmp_path: Path) -> None:
    """The pushed database is the standby's LIVE one, not the replica.

    So "the replica is non-empty" says nothing about what is about to overwrite
    the VPS. Measured 2026-09-19: live held 2 nodes while the replica held 6 —
    the same two plus vps-linuxuser .1, localhost .2, tower-server .4 and
    aarynfans-prod .5. Not a divergent rebuild: identical Noise keys, and the two
    live nodes carry the same machine keys and addresses as their replica
    records, so live is a strict subset.

    `failback` would have pushed that subset over the VPS *and* overwritten the
    replica with it first — destroying the only copy of the other four. The
    documented recovery command, run exactly when intended.
    """
    decision = hf.failback_ready(
        health=hf.Health(dns_ok=False, primary_ip_ok=True, dns_error="", primary_ip_error=""),
        replica=hf.ReplicaStatus(
            sqlite_path=tmp_path / "db.sqlite",
            noise_key_path=tmp_path / "noise_private.key",
            user_count=1,
            node_count=6,
            replicated_at="2026-08-31T02:37:25+00:00",
            promotable=True,
            reason="",
        ),
        state=hf.FailoverState(role="promoted"),
        live_nodes=2,
    )
    assert decision.action == "refuse_failback", (
        f"failback would proceed with 2 live nodes against a 6-node replica: {decision}"
    )
    assert "destroying the only copy" in decision.reason
    assert decision.alert is True, "an operator is not told the recovery was refused"


def test_failback_proceeds_when_nothing_would_be_lost(tmp_path: Path) -> None:
    """The guard must not block a legitimate recovery."""
    for live, repl in ((6, 6), (7, 6)):
        decision = hf.failback_ready(
            health=hf.Health(dns_ok=False, primary_ip_ok=True, dns_error="", primary_ip_error=""),
            replica=hf.ReplicaStatus(
                sqlite_path=tmp_path / "db.sqlite",
                noise_key_path=tmp_path / "noise_private.key",
                user_count=1,
                node_count=repl,
                replicated_at="2026-08-31T02:37:25+00:00",
                promotable=True,
                reason="",
            ),
            state=hf.FailoverState(role="promoted"),
            live_nodes=live,
        )
        assert decision.action == "failback", f"blocked a safe failback ({live} vs {repl})"


def test_force_is_required_to_accept_the_loss(tmp_path: Path) -> None:
    """Losing registrations must be an explicit choice, never a default."""
    kwargs = dict(
        health=hf.Health(dns_ok=False, primary_ip_ok=True, dns_error="", primary_ip_error=""),
        replica=hf.ReplicaStatus(
            sqlite_path=tmp_path / "db.sqlite",
            noise_key_path=tmp_path / "noise_private.key",
            user_count=1,
            node_count=6,
            replicated_at="2026-08-31T02:37:25+00:00",
            promotable=True,
            reason="",
        ),
        state=hf.FailoverState(role="promoted"),
        live_nodes=2,
    )
    assert hf.failback_ready(**kwargs).action == "refuse_failback"
    assert hf.failback_ready(**kwargs, force=True).action == "failback"


def test_failback_never_writes_over_the_replica(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """The backup must survive the push, so a failed push can be retried."""
    replica_dir = tmp_path / "headscale"
    replica_dir.mkdir()
    (replica_dir / "db.sqlite").write_bytes(b"REPLICA-6-NODES")
    (replica_dir / "noise_private.key").write_text("privkey:replica\n")
    live = tmp_path / "live"
    live.mkdir()
    (live / "db.sqlite").write_bytes(b"LIVE-2-NODES")
    (live / "noise_private.key").write_text("privkey:replica\n")

    settings = hf.Settings(
        primary_host="45.76.3.128",
        primary_user="root",
        ssh_key=str(tmp_path / "key"),
        replica_dir=replica_dir,
        live_data_dir=live,
        cloudflare_zone_id="zone",
        cloudflare_token="tok",
    )
    status = hf.ReplicaStatus(
        sqlite_path=replica_dir / "db.sqlite",
        noise_key_path=replica_dir / "noise_private.key",
        user_count=1,
        node_count=6,
        replicated_at="2026-08-31T02:37:25+00:00",
        promotable=True,
        reason="",
    )
    pushed: list[str] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        if argv and argv[0] == "scp":
            pushed.append(argv[-2])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(hf, "sqlite_counts", lambda p: (1, 2 if b"LIVE" in p.read_bytes() else 6))
    monkeypatch.setattr(hf, "set_dns_a", lambda *a, **k: None)
    monkeypatch.setattr(hf, "telegram_send", lambda _s, _m: None)
    monkeypatch.setattr(hf, "serves_dns_name", lambda _s, _a: (True, ""))

    hf.failback(
        settings,
        replica=status,
        state=hf.FailoverState(role="promoted", original_a_record="45.76.3.128"),
        health=hf.Health(dns_ok=False, primary_ip_ok=True, dns_error="", primary_ip_error=""),
        runner=fake_run,
        force=True,   # the point here is the file handling, not the gate
    )

    assert (replica_dir / "db.sqlite").read_bytes() == b"REPLICA-6-NODES", (
        "failback overwrote the replica; a failed push would leave no backup and "
        "the registrations only the replica held are gone"
    )
    assert pushed and "headscale-failback-" in pushed[0], (
        f"failback pushed {pushed} rather than a timestamped snapshot"
    )
