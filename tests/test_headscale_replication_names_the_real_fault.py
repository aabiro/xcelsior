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
