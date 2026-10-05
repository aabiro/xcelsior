"""Failback must not overwrite a VPS that already holds every registration.

`failback()` was written for a primary that *lost* its data, so it always pushed
the standby's database over the VPS. The outage of 2026-09-17 was the other
kind. The Vultr account went dark; the VPS's disk was never touched. Measured
when it returned on 2026-10-05:

    VPS       1 user, 6 nodes, latest update 2026-08-28 21:20:40.915744254
    replica   1 user, 6 nodes, latest update 2026-08-28 21:20:40.915744254
    standby   2 nodes (asus-pc, the MacBook) — a strict subset by machine_key
    Noise     identical digest on the VPS and the standby

The VPS was the most complete copy in existence. Every push failback could make
would only have subtracted: the standby's two nodes (refused by the node-count
gate, but then the operator's only way forward was `--force`, which deletes
four registrations), or the replica, identical at best and older at worst.

So failback now reads the VPS's database in place and compares by `machine_key`
— not row id, since the standby re-registers under fresh ids (asus-pc is 10 on
the VPS and 1 on the standby) — before deciding whether to push at all.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import headscale_failover as hf  # noqa: E402

VPS_KEYS = frozenset(f"mkey:{n}" for n in ("vps", "mac", "tower", "localhost", "prod", "asus"))
STANDBY_KEYS = frozenset({"mkey:mac", "mkey:asus"})
NOISE = "a" * 64


def _primary(keys=VPS_KEYS, noise=NOISE, **kw) -> hf.PrimarySummary:  # noqa: ANN001, ANN003
    return hf.PrimarySummary(machine_keys=frozenset(keys), users=1, noise_sha256=noise, **kw)


# ── The verdict, as a pure function ───────────────────────────────────────


def test_a_vps_holding_everything_is_left_alone() -> None:
    """The 2026-10-05 situation, exactly."""
    verdict = hf.push_verdict(_primary(), STANDBY_KEYS, NOISE)
    assert verdict.action == "skip", verdict.reason


def test_an_identical_database_is_not_pushed_either() -> None:
    """Pushing an identical copy is a restart of Headscale for nothing."""
    assert hf.push_verdict(_primary(), VPS_KEYS, NOISE).action == "skip"


def test_a_vps_that_lost_its_data_is_restored() -> None:
    """The case failback was written for must still work."""
    assert hf.push_verdict(hf.PrimarySummary(missing=True), STANDBY_KEYS, NOISE).action == "push"
    assert hf.push_verdict(_primary(keys=()), STANDBY_KEYS, NOISE).action == "push"


def test_registrations_only_the_standby_has_are_pushed() -> None:
    """A node that joined during the outage exists only on the standby."""
    verdict = hf.push_verdict(_primary(keys=STANDBY_KEYS), VPS_KEYS, NOISE)
    assert verdict.action == "push", verdict.reason


def test_diverged_databases_are_refused() -> None:
    """Each side has something the other lacks, so either push loses data."""
    verdict = hf.push_verdict(
        _primary(keys={"mkey:vps", "mkey:only-on-vps"}), {"mkey:vps", "mkey:only-here"}, NOISE
    )
    assert verdict.action == "refuse"
    assert "diverged" in verdict.reason


def test_a_vps_that_cannot_be_read_is_not_overwritten_blind() -> None:
    """Unverifiable is not the same as empty.

    The old behaviour pushed unconditionally, so an SSH hiccup while reading
    and a genuinely empty database were indistinguishable — and both ended
    with the VPS overwritten.
    """
    verdict = hf.push_verdict(hf.PrimarySummary(error="ssh: timed out"), STANDBY_KEYS, NOISE)
    assert verdict.action == "refuse"
    assert "--force" in verdict.reason, "the refusal must say how to override it"


def test_different_noise_keys_are_two_tailnets_and_refused() -> None:
    verdict = hf.push_verdict(_primary(noise="b" * 64), STANDBY_KEYS, NOISE)
    assert verdict.action == "refuse"
    assert "Noise" in verdict.reason


def test_force_still_pushes() -> None:
    """The documented override keeps working, including over a refusal."""
    assert hf.push_verdict(hf.PrimarySummary(error="x"), STANDBY_KEYS, NOISE, force=True).action == "push"


# ── failback() end to end, with the VPS faked at the SSH boundary ─────────


def _settings(tmp_path: Path) -> hf.Settings:
    replica = tmp_path / "replica"
    replica.mkdir()
    (replica / "noise_private.key").write_text("privkey\n")
    (replica / "db.sqlite").write_bytes(b"")
    live = tmp_path / "live"
    live.mkdir()
    return hf.Settings(
        primary_host="149.28.121.61",
        primary_user="root",
        ssh_key=str(tmp_path / "key"),
        replica_dir=replica,
        live_data_dir=live,
        cloudflare_zone_id="zone",
        cloudflare_token="tok",
    )


def _write_nodes(path: Path, keys) -> None:  # noqa: ANN001
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE nodes (machine_key TEXT, deleted_at TEXT)")
        conn.execute("CREATE TABLE users (id INTEGER)")
        conn.execute("INSERT INTO users VALUES (1)")
        conn.executemany("INSERT INTO nodes VALUES (?, NULL)", [(k,) for k in sorted(keys)])


def test_failback_onto_an_authoritative_vps_touches_nothing_but_dns(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """No `systemctl stop`, no scp — DNS restored and the standby demoted."""
    settings = _settings(tmp_path)
    _write_nodes(settings.live_data_dir / "db.sqlite", STANDBY_KEYS)
    (settings.live_data_dir / "noise_private.key").write_text("privkey\n")
    noise = hf.local_summary(
        settings.live_data_dir / "db.sqlite", settings.live_data_dir / "noise_private.key"
    )[1]

    commands: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        commands.append(list(argv))
        out = ""
        if argv and argv[0] == "ssh" and "python3 -c" in argv[-1]:
            out = json.dumps(
                {"missing": False, "machine_keys": sorted(VPS_KEYS), "users": 1, "noise_sha256": noise}
            )
        return subprocess.CompletedProcess(argv, 0, out, "")

    moved: list[str] = []
    sent: list[str] = []
    monkeypatch.setattr(hf, "set_dns_a", lambda _s, ip, **k: moved.append(ip))
    monkeypatch.setattr(hf, "telegram_send", lambda _s, m: sent.append(m))
    monkeypatch.setattr(hf, "serves_dns_name", lambda _s, _a: (True, ""))
    monkeypatch.setattr(hf, "sqlite_counts", lambda _p: (1, 2))

    state = hf.FailoverState(role="promoted", original_a_record="45.76.3.128")
    hf.failback(
        settings,
        replica=hf.ReplicaStatus(
            sqlite_path=settings.replica_dir / "db.sqlite",
            noise_key_path=settings.replica_dir / "noise_private.key",
            user_count=1,
            node_count=6,
            replicated_at="2026-08-31T02:37:25+00:00",
            promotable=True,
            reason="",
        ),
        state=state,
        health=hf.Health(dns_ok=False, primary_ip_ok=True, dns_error="", primary_ip_error=""),
        runner=fake_run,
    )

    touched = [c for c in commands if c[0] == "scp" or any("systemctl" in a for a in c)]
    assert not touched, f"failback modified a VPS that already held everything: {touched}"
    assert moved == ["45.76.3.128"], f"DNS not restored to the address it held: {moved}"
    assert state.role == "standby", "the standby was left promoted"
    assert any("Nothing pushed" in m for m in sent), f"the operator was not told why: {sent}"
    assert not list(settings.replica_dir.parent.glob("headscale-failback-*")), (
        "a push snapshot was taken for a push that did not happen"
    )


def test_an_unreadable_vps_holds_failback_before_anything_changes(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    settings = _settings(tmp_path)
    _write_nodes(settings.live_data_dir / "db.sqlite", STANDBY_KEYS)
    commands: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")  # no summary came back

    moved: list[str] = []
    monkeypatch.setattr(hf, "set_dns_a", lambda _s, ip, **k: moved.append(ip))
    monkeypatch.setattr(hf, "sqlite_counts", lambda _p: (1, 2))

    with pytest.raises(hf.FailoverError, match="could not read the VPS"):
        hf.failback(
            settings,
            replica=hf.ReplicaStatus(
                sqlite_path=settings.replica_dir / "db.sqlite",
                noise_key_path=settings.replica_dir / "noise_private.key",
                user_count=1, node_count=6, replicated_at="", promotable=True, reason="",
            ),
            state=hf.FailoverState(role="promoted", original_a_record="45.76.3.128"),
            health=hf.Health(dns_ok=False, primary_ip_ok=True, dns_error="", primary_ip_error=""),
            runner=fake_run,
        )
    assert not [c for c in commands if c[0] == "scp" or any("systemctl" in a for a in c)]
    assert not moved


# ── The script that runs on the VPS ───────────────────────────────────────


def test_the_remote_summary_script_survives_its_shell_quoting() -> None:
    """It travels as `python3 -c '...'`; one single quote and the shell eats it."""
    assert "'" not in hf._PRIMARY_SUMMARY_PY


def test_the_remote_summary_script_runs_and_reports_only_public_data(tmp_path: Path) -> None:
    """Execute the real script against a real database, through a real shell.

    Asserting on the string would prove nothing about whether it runs. This
    also pins what crosses the wire: machine keys (public) and a digest of the
    Noise key — never the key itself.
    """
    db = tmp_path / "db.sqlite"
    noise = tmp_path / "noise_private.key"
    _write_nodes(db, VPS_KEYS)
    noise.write_text("privkey:the-actual-secret\n")
    script = hf._PRIMARY_SUMMARY_PY.replace(
        "/var/lib/headscale/db.sqlite", str(db)
    ).replace("/var/lib/headscale/noise_private.key", str(noise))

    result = subprocess.run(
        ["sh", "-c", f"{sys.executable} -c '{script}'"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    payload = json.loads(result.stdout)
    assert frozenset(payload["machine_keys"]) == VPS_KEYS
    assert payload["users"] == 1 and payload["missing"] is False
    assert len(payload["noise_sha256"]) == 64
    assert "the-actual-secret" not in result.stdout, "the Noise private key left the VPS"


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0),
    reason="root reads through mode 000",
)
def test_status_without_root_says_so_instead_of_tracing_back(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # noqa: ANN001
    """`status` is the command someone runs by hand during an incident.

    It answered a non-root caller with a pathlib PermissionError traceback,
    which reads as the tool being broken rather than as "run this as root".
    """
    locked = tmp_path / "replica"
    locked.mkdir()
    locked.chmod(0)
    monkeypatch.setenv("XCELSIOR_HEADSCALE_REPLICA_DIR", str(locked))
    monkeypatch.setenv("XCELSIOR_ENV_FILE", str(tmp_path / "absent.env"))
    try:
        code = hf.main(["status"])
    finally:
        locked.chmod(0o700)
    err = capsys.readouterr().err
    assert code == 2
    assert "not readable" in err and "sudo" in err, err
    assert "Traceback" not in err


def test_the_runbook_recovery_command_can_find_its_ssh_key(tmp_path: Path) -> None:
    """`sudo headscale-failover failback`, as the runbook says, must authenticate.

    `.env` carries `XCELSIOR_SSH_KEY=$HOME/.ssh/xcelsior`. The tool reads `.env`
    itself and did not expand it, so ssh got the literal `$HOME/...` and the
    documented recovery could not reach the VPS — measured on the installed
    binary on 2026-10-05. Under `sudo` the fix also has to resolve against the
    invoking user's home, because `$HOME` is `/root` there.
    """
    import getpass
    import pwd

    user = getpass.getuser()
    if user == "root":
        pytest.skip("needs a non-root invoking user to distinguish the homes")
    env_file = tmp_path / ".env"
    env_file.write_text("XCELSIOR_SSH_KEY=$HOME/.ssh/xcelsior\n")
    settings = hf.settings_from_env(
        {"SUDO_USER": user, "HOME": "/root"}, dotenv_path=env_file
    )
    assert settings.ssh_key == f"{pwd.getpwnam(user).pw_dir}/.ssh/xcelsior", settings.ssh_key
    assert "$" not in settings.ssh_key


def test_a_tilde_key_path_resolves_too(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("XCELSIOR_HEADSCALE_SSH_KEY=~/.ssh/k\n")
    settings = hf.settings_from_env({"HOME": "/home/someone"}, dotenv_path=env_file)
    assert settings.ssh_key == "/home/someone/.ssh/k"
