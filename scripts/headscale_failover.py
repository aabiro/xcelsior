#!/usr/bin/env python3
"""Headscale control-plane failover for the Xcelsior tailnet.

Headscale on the VPS is the only coordination server the mesh has. When that
box dies, clients that reboot cannot fetch the Noise control key and fall into
NoState — there is no Tailscale. This tool keeps a copy of the sqlite database
and Noise private key off-box, promotes a standby that serves the same
https://hs.xcelsior.ca identity (same node IPs), and copies that state back
before DNS is pointed at the VPS again.

An empty local Headscale must never be promoted. That would mint a new tailnet
and new 100.64.0.0/10 assignments; when the VPS returned the networks would
not match.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlencode


def project_root() -> Path:
    env = os.environ.get("XCELSIOR_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    checkout = here.parent.parent
    if (checkout / "infra" / "headscale").is_dir():
        return checkout
    return Path("/mnt/storage/projects/xcelsior")


PROJECT = project_root()
DEFAULT_ENV_FILE = PROJECT / ".env"

# Headscale runs on the API VPS (`pixelenhance-labs`), not on 45.76.3.128 —
# that address is `aarynfans`, which has never had Headscale installed. The old
# default made every replication attempt fail against a healthy fleet and page
# about it. `headscale nodes list` on this host shows the original tailnet:
# .1 vps-linuxuser, .3 the Mac, .6 asus-pc.
DEFAULT_PRIMARY_HOST = "149.28.121.61"
DEFAULT_DNS_NAME = "hs.xcelsior.ca"
DEFAULT_LOGIN_SERVER = "https://hs.xcelsior.ca"
DEFAULT_REPLICA_DIR = Path("/var/backups/headscale")
DEFAULT_LIVE_DATA_DIR = Path("/var/lib/headscale")
DEFAULT_LIVE_CONFIG_DIR = Path("/etc/headscale")
DEFAULT_FAILURE_THRESHOLD = 3

# Headscale 0.26+ stores registered machines in `nodes`. Older installs used
# `machines`. Count whichever exists; a replica with zero of both is empty.
_NODE_TABLES = ("nodes", "machines")


class FailoverError(RuntimeError):
    """Operator-facing failure that should not traceback as a bug."""


class PrimaryUnreachable(FailoverError):
    """The Headscale VPS did not answer. Not a replica-corruption error."""


@dataclass(frozen=True)
class Settings:
    primary_host: str = DEFAULT_PRIMARY_HOST
    primary_user: str = "root"
    ssh_key: str = str(Path.home() / ".ssh/id_ed25519")
    login_server: str = DEFAULT_LOGIN_SERVER
    dns_name: str = DEFAULT_DNS_NAME
    replica_dir: Path = DEFAULT_REPLICA_DIR
    live_data_dir: Path = DEFAULT_LIVE_DATA_DIR
    live_config_dir: Path = DEFAULT_LIVE_CONFIG_DIR
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    health_timeout_sec: float = 5.0
    cloudflare_zone_id: str = ""
    cloudflare_token: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""
    standby_public_ip: str = ""
    standby_lan_ip: str = ""
    env_file: Path = DEFAULT_ENV_FILE


@dataclass(frozen=True)
class ReplicaStatus:
    sqlite_path: Path
    noise_key_path: Path
    user_count: int
    node_count: int
    replicated_at: str | None
    promotable: bool
    reason: str


@dataclass(frozen=True)
class Health:
    dns_ok: bool
    primary_ip_ok: bool
    dns_error: str = ""
    primary_ip_error: str = ""


@dataclass
class FailoverState:
    role: str = "standby"  # standby | promoted
    consecutive_failures: int = 0
    consecutive_primary_ip_ok: int = 0
    last_action: str = "none"
    last_alert: str = ""
    last_alert_at: str = ""
    # Which alerting condition is *currently* open, per channel. These are the
    # dedupe keys; `last_alert` above is the historical record and must not be
    # reused for dedupe (it persists after the condition clears, which silently
    # suppresses the next occurrence — `refuse_promote` sat in it from
    # 2026-08-19 onward, so a genuine refusal to promote would never have paged).
    #
    # The two channels reset on different events — a watchdog decision clears
    # when the decision stops alerting, a replication failure clears when
    # replication succeeds — so they cannot share one key: the watchdog's reset
    # runs every tick and would wipe an open replication episode, restoring the
    # every-two-minutes alert storm this was written to stop.
    alert_episode: str = ""
    replicate_alert_active: bool = False
    last_replicate_ok_at: str = ""
    promoted_at: str = ""
    original_a_record: str = DEFAULT_PRIMARY_HOST


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    alert: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_dotenv_values(path: Path) -> dict[str, str]:
    """Parse KEY=VAL lines without executing the file."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _env_or_dotenv(environ: Mapping[str, str], dotenv: Mapping[str, str], key: str) -> str:
    return (environ.get(key) or dotenv.get(key) or "").strip()


def default_ssh_key(environ: Mapping[str, str] | None = None) -> str:
    environ = environ or os.environ
    sudo_user = (environ.get("SUDO_USER") or "").strip()
    candidates = []
    if sudo_user and sudo_user != "root":
        candidates.append(Path("/home") / sudo_user / ".ssh" / "id_ed25519")
    candidates.append(Path.home() / ".ssh" / "id_ed25519")
    candidates.append(Path("/home/aaryn/.ssh/id_ed25519"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def settings_from_env(
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> Settings:
    environ = environ or os.environ
    dotenv_path = dotenv_path or Path(environ.get("XCELSIOR_ENV_FILE") or DEFAULT_ENV_FILE)
    dotenv = read_dotenv_values(dotenv_path)
    replica = environ.get("XCELSIOR_HEADSCALE_REPLICA_DIR") or str(DEFAULT_REPLICA_DIR)
    live = environ.get("XCELSIOR_HEADSCALE_LIVE_DIR") or str(DEFAULT_LIVE_DATA_DIR)
    config = environ.get("XCELSIOR_HEADSCALE_CONFIG_DIR") or str(DEFAULT_LIVE_CONFIG_DIR)
    threshold_raw = environ.get("XCELSIOR_HEADSCALE_FAILURE_THRESHOLD") or str(
        DEFAULT_FAILURE_THRESHOLD
    )
    timeout_raw = environ.get("XCELSIOR_HEADSCALE_HEALTH_TIMEOUT") or "5"
    return Settings(
        primary_host=_env_or_dotenv(environ, dotenv, "XCELSIOR_HEADSCALE_HOST")
        or DEFAULT_PRIMARY_HOST,
        primary_user=_env_or_dotenv(environ, dotenv, "XCELSIOR_HEADSCALE_USER") or "root",
        ssh_key=_env_or_dotenv(environ, dotenv, "XCELSIOR_HEADSCALE_SSH_KEY")
        or _env_or_dotenv(environ, dotenv, "XCELSIOR_SSH_KEY")
        or default_ssh_key(environ),
        login_server=_env_or_dotenv(environ, dotenv, "XCELSIOR_HEADSCALE_URL")
        or DEFAULT_LOGIN_SERVER,
        dns_name=_env_or_dotenv(environ, dotenv, "XCELSIOR_HEADSCALE_DNS_NAME") or DEFAULT_DNS_NAME,
        replica_dir=Path(replica),
        live_data_dir=Path(live),
        live_config_dir=Path(config),
        failure_threshold=max(1, int(threshold_raw)),
        health_timeout_sec=float(timeout_raw),
        cloudflare_zone_id=_env_or_dotenv(environ, dotenv, "CLOUDFLARE_ZONE_ID"),
        cloudflare_token=_env_or_dotenv(environ, dotenv, "CLOUDFLARE_API_TOKEN"),
        telegram_token=_env_or_dotenv(environ, dotenv, "XCELSIOR_TG_TOKEN"),
        telegram_chat_id=_env_or_dotenv(environ, dotenv, "XCELSIOR_TG_CHAT_ID"),
        standby_public_ip=_env_or_dotenv(environ, dotenv, "XCELSIOR_HEADSCALE_STANDBY_IP"),
        standby_lan_ip=_env_or_dotenv(environ, dotenv, "XCELSIOR_HEADSCALE_STANDBY_LAN_IP"),
        env_file=dotenv_path,
    )


def sqlite_counts(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        return 0, 0
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        users = 0
        if "users" in tables:
            users = int(con.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        nodes = 0
        for table in _NODE_TABLES:
            if table in tables:
                nodes += int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return users, nodes
    except sqlite3.Error:
        return 0, 0
    finally:
        con.close()


def inspect_replica(replica_dir: Path) -> ReplicaStatus:
    sqlite_path = replica_dir / "db.sqlite"
    noise_key_path = replica_dir / "noise_private.key"
    meta_path = replica_dir / "replica-meta.json"
    users, nodes = sqlite_counts(sqlite_path)
    replicated_at = None
    if meta_path.is_file():
        try:
            replicated_at = json.loads(meta_path.read_text(encoding="utf-8")).get("replicated_at")
        except (OSError, json.JSONDecodeError):
            replicated_at = None
    has_noise = noise_key_path.is_file() and noise_key_path.stat().st_size > 0
    if not sqlite_path.is_file():
        reason = "no replicated sqlite database"
        promotable = False
    elif users == 0 or nodes == 0:
        reason = (
            f"replica is empty (users={users}, nodes={nodes}); promoting it would "
            "mint a new tailnet and new 100.64.0.0/10 addresses"
        )
        promotable = False
    elif not has_noise:
        reason = "replica sqlite has nodes but noise_private.key is missing"
        promotable = False
    else:
        reason = f"replica has {users} user(s) and {node_label(nodes)}"
        promotable = True
    return ReplicaStatus(
        sqlite_path=sqlite_path,
        noise_key_path=noise_key_path,
        user_count=users,
        node_count=nodes,
        replicated_at=replicated_at,
        promotable=promotable,
        reason=reason,
    )


def node_label(count: int) -> str:
    return f"{count} node" if count == 1 else f"{count} nodes"


def load_state(path: Path) -> FailoverState:
    if not path.is_file():
        return FailoverState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return FailoverState()
    allowed = {f.name for f in FailoverState.__dataclass_fields__.values()}
    return FailoverState(**{k: v for k, v in raw.items() if k in allowed})


def save_state(path: Path, state: FailoverState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def decide(
    *,
    health: Health,
    replica: ReplicaStatus,
    state: FailoverState,
    failure_threshold: int,
) -> Decision:
    """Watchdog policy. Failback is never automatic — that is a split-brain risk."""
    if state.role == "promoted":
        if health.primary_ip_ok:
            return Decision(
                "alert_primary_back",
                "standby is serving hs.xcelsior.ca and the original VPS is reachable "
                "again; run failback to restore the VPS as control plane without "
                "changing node IPs",
                alert=True,
            )
        return Decision("stay_promoted", "standby is active and the original VPS is still down")

    if health.dns_ok:
        return Decision(
            "replicate",
            "primary Headscale answered on hs.xcelsior.ca; refresh the off-box replica",
        )

    failures = state.consecutive_failures
    if failures < failure_threshold:
        return Decision(
            "wait",
            f"hs.xcelsior.ca is down ({failures}/{failure_threshold} failures); not promoting yet",
        )

    if not replica.promotable:
        return Decision("refuse_promote", replica.reason, alert=True)

    return Decision(
        "promote",
        "hs.xcelsior.ca has been down past the threshold and a non-empty replica "
        "with a Noise key is available; promoting preserves existing 100.64.0.0/10 "
        "assignments",
        alert=True,
    )


def failback_ready(*, health: Health, replica: ReplicaStatus, state: FailoverState) -> Decision:
    if state.role != "promoted":
        return Decision("none", "standby is not promoted; nothing to fail back")
    if not health.primary_ip_ok:
        return Decision(
            "refuse_failback",
            f"original VPS {health.primary_ip_error or 'is unreachable'}; "
            "copying the database there would fail and DNS must not flip yet",
            alert=True,
        )
    if not replica.promotable:
        return Decision(
            "refuse_failback",
            "local replica is empty or missing the Noise key; failing back would "
            "wipe the tailnet identity on the VPS",
            alert=True,
        )
    return Decision(
        "failback",
        "VPS is reachable and the replica still has users/nodes; copy sqlite+Noise "
        "to the VPS, start Headscale there, then point DNS back",
    )


def probe_http(url: str, timeout: float) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(256).decode("utf-8", "replace")
            if response.status == 200 and ("pass" in body.lower() or "ok" in body.lower()):
                return True, ""
            return False, f"status={response.status} body={body[:80]!r}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def collect_health(settings: Settings) -> Health:
    dns_ok, dns_error = probe_http(
        f"{settings.login_server.rstrip('/')}/health", settings.health_timeout_sec
    )
    # Probe the VPS by IP so a promoted standby (which owns the DNS name) is
    # not mistaken for the original primary.
    ip_ok, ip_error = probe_http(
        f"https://{settings.primary_host}/health", settings.health_timeout_sec
    )
    if not ip_ok:
        # Headscale may only listen on loopback behind nginx; TCP 443 is enough
        # to know the box is back.
        ip_ok, ip_error_tcp = probe_tcp(settings.primary_host, 443, settings.health_timeout_sec)
        if not ip_ok:
            ip_error = ip_error or ip_error_tcp
        else:
            ip_error = ""
    return Health(
        dns_ok=dns_ok, primary_ip_ok=ip_ok, dns_error=dns_error, primary_ip_error=ip_error
    )


def probe_tcp(host: str, port: int, timeout: float) -> tuple[bool, str]:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except OSError as exc:
        return False, str(exc)


def replica_paths(replica_dir: Path) -> dict[str, Path]:
    return {
        "sqlite": replica_dir / "db.sqlite",
        "noise": replica_dir / "noise_private.key",
        "derp": replica_dir / "derp_server_private.key",
        "config": replica_dir / "config.yaml",
        "acl": replica_dir / "acl.json",
        "meta": replica_dir / "replica-meta.json",
        "state": replica_dir / "failover-state.json",
    }


def write_replica_meta(replica_dir: Path, users: int, nodes: int, source: str) -> None:
    replica_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "replicated_at": utc_now(),
        "user_count": users,
        "node_count": nodes,
        "source": source,
    }
    (replica_dir / "replica-meta.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def run_cmd(
    argv: list[str],
    *,
    timeout: int = 60,
    check: bool = True,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> subprocess.CompletedProcess:
    run = runner or subprocess.run
    try:
        completed = run(argv, check=False, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise PrimaryUnreachable(f"timed out: {' '.join(argv)}") from exc
    if check and completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        message = f"command failed ({completed.returncode}): {' '.join(argv)}\n{stderr}"
        lowered = stderr.lower()
        if any(
            token in lowered
            for token in (
                "timed out",
                "timeout",
                "connection refused",
                "network is unreachable",
                "no route to host",
                "could not resolve",
            )
        ):
            raise PrimaryUnreachable(message)
        raise FailoverError(message)
    return completed


def replicate(
    settings: Settings, *, runner: Callable[..., subprocess.CompletedProcess] | None = None
) -> ReplicaStatus:
    """Pull a consistent sqlite snapshot plus Noise key from the primary VPS."""
    settings.replica_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(settings.replica_dir, 0o700)
    remote = f"{settings.primary_user}@{settings.primary_host}"
    ssh = [
        "ssh",
        "-i",
        settings.ssh_key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=accept-new",
        remote,
    ]
    # Confirm this host actually runs Headscale before blaming its database.
    #
    # The replication target was pointed at 45.76.3.128 for weeks. That box is
    # healthy and reachable, and has never had Headscale on it — so every run
    # failed on the *database* path and paged `unable to open database file`,
    # which reads as corruption or permissions on the control plane. It sent the
    # investigation to a server that was fine. A missing database and a wrong
    # host are different faults and must not share an error message.
    probe = run_cmd(
        ssh + ["test -f /var/lib/headscale/db.sqlite && echo present || echo absent"],
        timeout=30,
        runner=runner,
        check=False,
    )
    if "absent" in (probe.stdout or ""):
        raise FailoverError(
            f"{settings.primary_host} has no /var/lib/headscale/db.sqlite — "
            "it is reachable but is not the Headscale control plane. Check "
            "XCELSIOR_HEADSCALE_HOST in /etc/default/headscale-failover; this is "
            "a configuration error, not a database fault."
        )

    # A consistent snapshot even while Headscale has the DB open, via SQLite's
    # backup API.
    #
    # Python's stdlib rather than the `sqlite3` CLI: that binary is not
    # installed on the Headscale host, and the failure it produced —
    # `unable to open database file` — reads as a missing or unreadable
    # *database* rather than a missing *tool*, which is what sent this chasing
    # permissions and paths on a healthy server. `python3` is present on every
    # host in this fleet and `Connection.backup()` has the same guarantee.
    remote_backup = (
        "python3 -c "
        "'import sqlite3,sys; "
        'src=sqlite3.connect("file:/var/lib/headscale/db.sqlite?mode=ro",uri=True,timeout=5); '
        'dst=sqlite3.connect("/tmp/headscale-replica.sqlite"); '
        "src.backup(dst); dst.close(); src.close()'"
    )
    run_cmd(ssh + [remote_backup], timeout=90, runner=runner)
    scp_base = [
        "scp",
        "-i",
        settings.ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    dest_sqlite = settings.replica_dir / "db.sqlite"
    run_cmd(
        scp_base + [f"{remote}:/tmp/headscale-replica.sqlite", str(dest_sqlite)],
        timeout=90,
        runner=runner,
    )
    for remote_path, local_name, required in (
        ("/var/lib/headscale/noise_private.key", "noise_private.key", True),
        ("/var/lib/headscale/derp_server_private.key", "derp_server_private.key", False),
        ("/etc/headscale/config.yaml", "config.yaml", False),
        ("/etc/headscale/acl.json", "acl.json", False),
    ):
        dest = settings.replica_dir / local_name
        try:
            run_cmd(scp_base + [f"{remote}:{remote_path}", str(dest)], timeout=60, runner=runner)
        except FailoverError:
            if required:
                raise
    users, nodes = sqlite_counts(dest_sqlite)
    if users == 0 or nodes == 0:
        raise FailoverError(
            f"replicated sqlite is empty (users={users}, nodes={nodes}); "
            "refusing to keep it as a failover replica"
        )
    write_replica_meta(settings.replica_dir, users, nodes, source=settings.primary_host)
    run_cmd(ssh + ["rm -f /tmp/headscale-replica.sqlite"], timeout=30, runner=runner, check=False)
    return inspect_replica(settings.replica_dir)


def cloudflare_headers(settings: Settings) -> dict[str, str]:
    if not settings.cloudflare_token:
        raise FailoverError("CLOUDFLARE_API_TOKEN is not set")
    return {
        "Authorization": f"Bearer {settings.cloudflare_token}",
        "Content-Type": "application/json",
    }


def cf_request(
    settings: Settings,
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    opener: Callable[..., object] | None = None,
) -> dict:
    if not settings.cloudflare_zone_id:
        raise FailoverError("CLOUDFLARE_ZONE_ID is not set")
    url = f"https://api.cloudflare.com/client/v4/zones/{settings.cloudflare_zone_id}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method, headers=cloudflare_headers(settings)
    )
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=20) as response:  # type: ignore[arg-type]
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise FailoverError(f"Cloudflare API {method} {path} failed: {exc.code} {detail}") from exc
    if not body.get("success"):
        raise FailoverError(f"Cloudflare API {method} {path} unsuccessful: {body.get('errors')}")
    return body


def dns_a_record(settings: Settings, *, opener: Callable[..., object] | None = None) -> dict:
    query = urlencode({"name": settings.dns_name, "type": "A"})
    body = cf_request(settings, "GET", f"/dns_records?{query}", opener=opener)
    records = body.get("result") or []
    if not records:
        raise FailoverError(f"no A record for {settings.dns_name}")
    return records[0]


def set_dns_a(
    settings: Settings,
    ipv4: str,
    *,
    opener: Callable[..., object] | None = None,
) -> dict:
    record = dns_a_record(settings, opener=opener)
    record_id = record["id"]
    payload = {
        "type": "A",
        "name": settings.dns_name,
        "content": ipv4,
        "ttl": 60,
        "proxied": False,
    }
    return cf_request(settings, "PUT", f"/dns_records/{record_id}", payload, opener=opener)


def detect_public_ip(timeout: float = 8.0) -> str:
    with urllib.request.urlopen("https://ipv4.icanhazip.com", timeout=timeout) as response:
        ip = response.read().decode("utf-8").strip()
    if not ip:
        raise FailoverError("could not detect standby public IPv4")
    return ip


def telegram_send(settings: Settings, text: str) -> None:
    if not settings.telegram_token or not settings.telegram_chat_id:
        return
    payload = urlencode(
        {
            "chat_id": settings.telegram_chat_id,
            "text": text[:3500],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage",
        data=payload,
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=10).read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o600)
    tmp.replace(dest)


def rewrite_server_url(config_path: Path, login_server: str) -> None:
    """Clients already have --login-server https://hs.xcelsior.ca. The promoted
    process must advertise that URL, not localhost."""
    if not config_path.is_file():
        return
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    rewritten = False
    out: list[str] = []
    for line in lines:
        if line.lstrip().startswith("server_url:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}server_url: {login_server}\n")
            rewritten = True
        else:
            out.append(line)
    if not rewritten:
        out.insert(0, f"server_url: {login_server}\n")
    config_path.write_text("".join(out), encoding="utf-8")


def install_replica_into_live(settings: Settings, replica: ReplicaStatus) -> None:
    live = settings.live_data_dir
    live.mkdir(parents=True, exist_ok=True)
    copy_file(replica.sqlite_path, live / "db.sqlite")
    copy_file(replica.noise_key_path, live / "noise_private.key")
    derp = settings.replica_dir / "derp_server_private.key"
    if derp.is_file():
        copy_file(derp, live / "derp_server_private.key")
    acl = settings.replica_dir / "acl.json"
    if acl.is_file():
        copy_file(acl, settings.live_config_dir / "acl.json")
    config = settings.replica_dir / "config.yaml"
    dest_config = settings.live_config_dir / "config.yaml"
    if config.is_file():
        copy_file(config, dest_config)
    elif not dest_config.is_file():
        template = PROJECT / "infra" / "headscale" / "config.yaml"
        if template.is_file():
            copy_file(template, dest_config)
    rewrite_server_url(dest_config, settings.login_server)


def maybe_alert(settings: Settings, state: FailoverState, decision: Decision) -> FailoverState:
    if not decision.alert:
        # Condition cleared: the next occurrence is a new episode and pages.
        state.alert_episode = ""
        return state
    if state.alert_episode == decision.action:
        return state
    telegram_send(
        settings,
        f"Headscale failover: {decision.action}\n{decision.reason}",
    )
    state.alert_episode = decision.action
    state.last_alert = decision.action
    state.last_alert_at = utc_now()
    return state


def watchdog_tick(
    settings: Settings,
    *,
    health: Health | None = None,
    replica: ReplicaStatus | None = None,
    state: FailoverState | None = None,
    execute: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> tuple[Decision, FailoverState]:
    paths = replica_paths(settings.replica_dir)
    state = state or load_state(paths["state"])
    health = health or collect_health(settings)
    replica = replica or inspect_replica(settings.replica_dir)

    if health.dns_ok and state.role != "promoted":
        state.consecutive_failures = 0
    elif state.role != "promoted":
        state.consecutive_failures += 1

    if health.primary_ip_ok:
        state.consecutive_primary_ip_ok += 1
    else:
        state.consecutive_primary_ip_ok = 0

    decision = decide(
        health=health,
        replica=replica,
        state=state,
        failure_threshold=settings.failure_threshold,
    )
    state.last_action = decision.action
    state = maybe_alert(settings, state, decision)

    if execute:
        if decision.action == "replicate":
            try:
                replicate(settings, runner=runner)
                state.last_replicate_ok_at = utc_now()
                if state.replicate_alert_active:
                    # Say so once, then go quiet. A page that heals silently
                    # leaves you believing it is still broken.
                    telegram_send(settings, "Headscale replica refresh recovered")
                    state.replicate_alert_active = False
                    state.last_alert_at = utc_now()
            except PrimaryUnreachable:
                state.last_action = "replicate_skipped_unreachable"
            except FailoverError as exc:
                state.last_action = "replicate_failed"
                # Alert once per episode, the way `maybe_alert` does for every
                # other decision. This call site used to send unconditionally,
                # and the watchdog timer runs every two minutes — so a single
                # persistent fault (a misconfigured host, for weeks) delivered
                # the identical message ~700 times a day until someone noticed.
                if not state.replicate_alert_active:
                    telegram_send(settings, f"Headscale replica refresh failed: {exc}")
                    state.replicate_alert_active = True
                    state.last_alert = "replicate_failed"
                    state.last_alert_at = utc_now()
        elif decision.action == "promote":
            # Record what the A record *actually* held before we take the name,
            # rather than assuming it equals the primary host.
            #
            # Those two coincided until 2026-08-28 and the assumption was
            # invisible. They no longer do: `hs.xcelsior.ca` points at
            # 45.76.3.128, which reverse-proxies to the real Headscale on
            # 149.28.121.61 (= primary_host). Assuming them equal would file the
            # wrong address as the thing to restore, and this value's only reader
            # is a human running failback under pressure.
            #
            # Best-effort: an emergency promotion must not be blocked because
            # Cloudflare was unreachable for a bookkeeping read.
            try:
                state.original_a_record = dns_a_record(settings)["content"]
            except (FailoverError, KeyError, OSError):
                state.original_a_record = settings.primary_host

            promote(settings, replica=replica, state=state, runner=runner)
            state.role = "promoted"
            state.promoted_at = utc_now()

    save_state(paths["state"], state)
    return decision, state


def promote(
    settings: Settings,
    *,
    replica: ReplicaStatus | None = None,
    state: FailoverState | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    opener: Callable[..., object] | None = None,
    public_ip: str | None = None,
) -> None:
    replica = replica or inspect_replica(settings.replica_dir)
    if not replica.promotable:
        raise FailoverError(f"refusing to promote: {replica.reason}")
    install_replica_into_live(settings, replica)
    ip = public_ip or settings.standby_public_ip or detect_public_ip()
    if settings.cloudflare_token and settings.cloudflare_zone_id:
        set_dns_a(settings, ip, opener=opener)
    run_cmd(["systemctl", "restart", "headscale"], timeout=30, runner=runner)
    telegram_send(
        settings,
        f"Headscale standby promoted on {ip}. DNS {settings.dns_name} now points here. "
        f"Replica: {replica.user_count} users, {node_label(replica.node_count)}. "
        "Node IPs are unchanged. Run failback when the VPS is healthy.",
    )


def failback(
    settings: Settings,
    *,
    replica: ReplicaStatus | None = None,
    state: FailoverState | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    opener: Callable[..., object] | None = None,
    health: Health | None = None,
) -> None:
    paths = replica_paths(settings.replica_dir)
    state = state or load_state(paths["state"])
    replica = replica or inspect_replica(settings.replica_dir)
    health = health or collect_health(settings)
    decision = failback_ready(health=health, replica=replica, state=state)
    if decision.action != "failback":
        raise FailoverError(decision.reason)

    # Snapshot whatever the standby served during the outage, then push that
    # identity to the VPS *before* DNS moves. Flipping DNS first would send
    # clients at a stale (or empty) VPS database.
    live_sqlite = settings.live_data_dir / "db.sqlite"
    if live_sqlite.is_file():
        copy_file(live_sqlite, replica.sqlite_path)
    live_noise = settings.live_data_dir / "noise_private.key"
    if live_noise.is_file():
        copy_file(live_noise, replica.noise_key_path)
    users, nodes = sqlite_counts(replica.sqlite_path)
    write_replica_meta(settings.replica_dir, users, nodes, source="standby-failback")

    remote = f"{settings.primary_user}@{settings.primary_host}"
    scp_base = [
        "scp",
        "-i",
        settings.ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    ssh = [
        "ssh",
        "-i",
        settings.ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        remote,
    ]
    run_cmd(ssh + ["systemctl stop headscale"], timeout=30, runner=runner)
    run_cmd(
        scp_base + [str(replica.sqlite_path), f"{remote}:/var/lib/headscale/db.sqlite"],
        timeout=90,
        runner=runner,
    )
    run_cmd(
        scp_base + [str(replica.noise_key_path), f"{remote}:/var/lib/headscale/noise_private.key"],
        timeout=60,
        runner=runner,
    )
    run_cmd(ssh + ["systemctl start headscale"], timeout=30, runner=runner)
    set_dns_a(settings, settings.primary_host, opener=opener)
    state.role = "standby"
    state.promoted_at = ""
    state.consecutive_failures = 0
    state.last_action = "failback"
    save_state(paths["state"], state)
    telegram_send(
        settings,
        f"Headscale failed back to {settings.primary_host}. "
        f"Pushed {users} users / {node_label(nodes)}. DNS restored. "
        "Node IPs unchanged.",
    )


def status_payload(settings: Settings) -> dict:
    paths = replica_paths(settings.replica_dir)
    replica = inspect_replica(settings.replica_dir)
    state = load_state(paths["state"])
    health = collect_health(settings)
    decision = decide(
        health=health,
        replica=replica,
        state=state,
        failure_threshold=settings.failure_threshold,
    )
    return {
        "dns_name": settings.dns_name,
        "primary_host": settings.primary_host,
        "health": asdict(health),
        "replica": {
            "promotable": replica.promotable,
            "reason": replica.reason,
            "user_count": replica.user_count,
            "node_count": replica.node_count,
            "replicated_at": replica.replicated_at,
            "dir": str(settings.replica_dir),
        },
        "state": asdict(state),
        "decision": {"action": decision.action, "reason": decision.reason},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("health", "status", "replicate", "decide", "watchdog", "promote", "failback"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="watchdog only: perform replicate/promote instead of printing the decision",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print machine-readable output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_env()
    try:
        if args.command in {"health", "status", "decide"}:
            payload = status_payload(settings)
            if args.as_json or args.command != "decide":
                json.dump(payload, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                decision = payload["decision"]
                print(f"{decision['action']}: {decision['reason']}")
            return 0
        if args.command == "replicate":
            try:
                replica = replicate(settings)
            except PrimaryUnreachable as exc:
                print(f"primary unreachable; skip replica refresh: {exc}", file=sys.stderr)
                return 0
            print(f"replica ok: {replica.reason} (at {replica.replicated_at})")
            return 0
        if args.command == "watchdog":
            decision, state = watchdog_tick(settings, execute=args.execute)
            print(f"{decision.action}: {decision.reason}")
            print(f"role={state.role} failures={state.consecutive_failures}")
            return 0
        if args.command == "promote":
            promote(settings)
            print("promoted")
            return 0
        if args.command == "failback":
            failback(settings)
            print("failed back")
            return 0
    except FailoverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
