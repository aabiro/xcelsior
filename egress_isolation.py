"""Tenant containers reach the internet only through an allowlisting proxy.

Every marketplace container joins a dedicated Docker bridge unless the control
plane marked its job ``egress: "open"``, which is an administrator decision per
account (``abuse_policy.egress_for_owner``). On that bridge the host:

* drops every connection a container starts towards another machine (FORWARD,
  via DOCKER-USER) but passes replies to connections made *to* it, so renter
  SSH and exposed ports keep working;
* drops everything a container sends to the host itself except the proxy port
  on the bridge gateway (INPUT), so host services, the LAN and the mesh are out
  of reach; IPv6 from the bridge is dropped outright;
* runs Squid on the gateway, allowing HTTP(S) only to package and model
  registries (``ALLOWED_DOMAINS``) and never to private addresses.

DNS points at the gateway, where INPUT drops it: names resolve only inside the
proxy, so DNS cannot carry a tunnel either.

``python3 egress_isolation.py provision`` (root, run by xcelsior-egress.service
and its timer) creates or repairs all of this, verifies it and writes a status
file. The worker reads that file: it advertises ``egress_isolation`` to the
scheduler only while the status is fresh and passing, and refuses to start a
restricted container otherwise. A host that cannot isolate gets no
restricted work; nothing falls back to open egress.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RESTRICTED = "restricted"
OPEN = "open"

NETWORK = "xcelsior-tenant"
BRIDGE = "xcl-tenant0"
SUBNET = "172.30.242.0/24"
GATEWAY = "172.30.242.1"
PROXY_PORT = 3128
OUT_CHAIN = "XCL-TENANT-OUT"
IN_CHAIN = "XCL-TENANT-IN"

PROXY_UNIT = "xcelsior-egress-proxy.service"
CONFIG_DIR = Path(os.environ.get("XCELSIOR_EGRESS_CONFIG_DIR", "/etc/xcelsior/egress"))
STATUS_FILE = Path(os.environ.get("XCELSIOR_EGRESS_STATUS", "/run/xcelsior-egress/status.json"))
# xcelsior-egress.timer re-verifies every 2 minutes.
STATUS_MAX_AGE_S = 300
# Only root writes the status; anything else could claim isolation it lacks.
_STATUS_OWNER = 0

# Registries the SSH setup (apt/apk) and ordinary ML workflows need. A leading
# dot covers the domain and its subdomains; Squid rejects overlapping entries.
ALLOWED_DOMAINS = (
    "pypi.org",
    "files.pythonhosted.org",
    "pypi.nvidia.com",
    "download.pytorch.org",
    "conda.anaconda.org",
    ".huggingface.co",
    ".hf.co",
    ".github.com",
    ".githubusercontent.com",
    "registry.npmjs.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "ports.ubuntu.com",
    "deb.debian.org",
    "security.debian.org",
    "dl-cdn.alpinelinux.org",
    "developer.download.nvidia.com",
)

SQUID_CONF = f"""\
# Written by egress_isolation.py; edits are overwritten.
http_port {GATEWAY}:{PROXY_PORT}
pid_filename /run/xcelsior-egress-proxy/squid.pid
cache deny all
cache_mem 8 MB
cache_log stdio:/dev/stderr
access_log none
cache_store_log none
logfile_rotate 0
shutdown_lifetime 1 seconds
acl CONNECT method CONNECT
acl web_ports port 80 443
acl tls_port port 443
acl tenants src {SUBNET}
acl approved dstdomain -n "{CONFIG_DIR}/domains.txt"
acl private_dst dst 0.0.0.0/8 10.0.0.0/8 100.64.0.0/10 127.0.0.0/8 169.254.0.0/16 172.16.0.0/12 192.168.0.0/16 224.0.0.0/4 240.0.0.0/4 ::1/128 fc00::/7 fe80::/10
http_access deny !tenants
http_access deny !web_ports
http_access deny CONNECT !tls_port
http_access deny !approved
http_access deny private_dst
http_access allow tenants approved
http_access deny all
forwarded_for delete
via off
"""

IPT = ("iptables", "-w", "5")
IP6T = ("ip6tables", "-w", "5")


class EgressUnavailable(RuntimeError):
    """This host cannot start a restricted container right now."""


def job_egress(job: dict | None) -> str:
    """Only an explicit, server-set ``open`` escapes the restricted network."""
    return OPEN if str((job or {}).get("egress") or "").strip().lower() == OPEN else RESTRICTED


def chain_rules() -> dict[str, list[list[str]]]:
    return {
        OUT_CHAIN: [
            ["-m", "conntrack", "--ctdir", "REPLY", "-j", "RETURN"],
            ["-m", "limit", "--limit", "10/min", "-j", "LOG", "--log-prefix", "XCL-TENANT-DROP "],
            ["-j", "DROP"],
        ],
        IN_CHAIN: [
            ["-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
            ["-d", GATEWAY, "-p", "tcp", "--dport", str(PROXY_PORT), "-j", "ACCEPT"],
            ["-j", "DROP"],
        ],
    }


# (command, chain, rule): each must be the first rule of its chain.
HOOKS = (
    (IPT, "DOCKER-USER", ["-i", BRIDGE, "-j", OUT_CHAIN]),
    (IPT, "INPUT", ["-i", BRIDGE, "-j", IN_CHAIN]),
    (IP6T, "FORWARD", ["-i", BRIDGE, "-j", "DROP"]),
    (IP6T, "INPUT", ["-i", BRIDGE, "-j", "DROP"]),
)


def _run(args, *, check=True, input_text=None):
    return subprocess.run(
        list(args), capture_output=True, text=True, timeout=30, check=check, input=input_text
    )


def _network() -> dict | None:
    r = _run(["docker", "network", "inspect", NETWORK], check=False)
    if r.returncode:
        return None
    return json.loads(r.stdout)[0]


def network_problems(net: dict | None) -> list[str]:
    if net is None:
        return [f"docker network {NETWORK} is missing"]
    problems = []
    options = net.get("Options") or {}
    if net.get("Driver") != "bridge":
        problems.append(f"driver is {net.get('Driver')}, not bridge")
    if net.get("Internal"):
        problems.append("network is internal, so published SSH ports would not work")
    if net.get("EnableIPv6"):
        problems.append("IPv6 is enabled")
    if options.get("com.docker.network.bridge.name") != BRIDGE:
        problems.append(f"bridge is not named {BRIDGE}")
    if options.get("com.docker.network.bridge.enable_icc") != "false":
        problems.append("containers can reach each other (ICC is on)")
    ipam = [
        (c.get("Subnet"), c.get("Gateway")) for c in (net.get("IPAM") or {}).get("Config") or []
    ]
    if ipam != [(SUBNET, GATEWAY)]:
        problems.append(f"addressing is {ipam}, expected {[(SUBNET, GATEWAY)]}")
    return problems


def _rules(cmd, chain) -> list[list[str]]:
    return [
        ln.split()[2:]
        for ln in _run([*cmd, "-S", chain]).stdout.splitlines()
        if ln.startswith("-A ")
    ]


def _first_rule(cmd, chain) -> list[str] | None:
    rules = _rules(cmd, chain)
    return rules[0] if rules else None


def _hook_problems() -> list[str]:
    return [
        f"{'ip6tables' if cmd is IP6T else 'iptables'} {chain} does not start with {' '.join(rule)}"
        for cmd, chain, rule in HOOKS
        if _first_rule(cmd, chain) != rule
    ]


def _chain_problems() -> list[str]:
    problems = []
    for chain, rules in chain_rules().items():
        listed = [
            ln
            for ln in _run([*IPT, "-S", chain], check=False).stdout.splitlines()
            if ln.startswith("-A ")
        ]
        if len(listed) != len(rules):
            problems.append(f"{chain} has {len(listed)} rules, expected {len(rules)}")
            continue
        for rule in rules:
            if _run([*IPT, "-C", chain, *rule], check=False).returncode:
                problems.append(f"{chain} is missing {' '.join(rule)}")
    return problems


def _config_problems() -> list[str]:
    problems = []
    for path in (CONFIG_DIR, CONFIG_DIR / "squid.conf", CONFIG_DIR / "domains.txt"):
        try:
            st = path.stat()
        except OSError:
            problems.append(f"{path} is missing")
            continue
        if st.st_uid != 0 or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            problems.append(f"{path} must be owned by root and writable only by root")
    return problems


def _proxy_problems() -> list[str]:
    # Squid answers 403 for a name outside the allowlist without resolving it,
    # so this proves the proxy is up and enforcing even with no internet.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": f"http://{GATEWAY}:{PROXY_PORT}"})
    )
    try:
        opener.open("http://egress-check.invalid/", timeout=5)
    except urllib.error.HTTPError as exc:
        return [] if exc.code == 403 else [f"proxy answered {exc.code} for a blocked name"]
    except OSError as exc:
        return [f"proxy unreachable: {exc}"]
    return ["proxy let a blocked name through"]


def verify() -> dict:
    net = None
    try:
        net = _network()
        problems = (
            network_problems(net)
            + _chain_problems()
            + _hook_problems()
            + _config_problems()
            + _proxy_problems()
        )
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        problems = [f"check failed: {exc}"]
    return {
        "ok": not problems,
        "problems": problems,
        "checked_at": time.time(),
        "network_id": (net or {}).get("Id", ""),
    }


def _write_root_file(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.chown(tmp, 0, 0)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def write_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    _write_root_file(STATUS_FILE, json.dumps(status))


def provision() -> dict:
    """Create or repair the network, firewall and proxy config; verify; record."""
    if os.geteuid() != 0:
        raise SystemExit("egress_isolation provision must run as root")

    CONFIG_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
    os.chown(CONFIG_DIR, 0, 0)
    os.chmod(CONFIG_DIR, 0o755)
    changed = False
    for name, text in (
        ("domains.txt", "\n".join(ALLOWED_DOMAINS) + "\n"),
        ("squid.conf", SQUID_CONF),
    ):
        path = CONFIG_DIR / name
        try:
            changed |= path.read_text() != text
        except OSError:
            changed = True
        _write_root_file(path, text)
    if changed:
        try:
            _run(["systemctl", "try-reload-or-restart", PROXY_UNIT], check=False)
        except OSError:
            pass  # no systemd: whoever runs the proxy reloads it

    net = _network()
    if net is None:
        _run(
            [
                "docker",
                "network",
                "create",
                "--driver",
                "bridge",
                "--subnet",
                SUBNET,
                "--gateway",
                GATEWAY,
                "-o",
                f"com.docker.network.bridge.name={BRIDGE}",
                "-o",
                "com.docker.network.bridge.enable_icc=false",
                "--label",
                "ca.xcelsior.role=tenant-egress",
                NETWORK,
            ]
        )
    elif network_problems(net):
        # Never rebuild a network that may hold tenant containers.
        status = verify()
        write_status(status)
        return status

    # Both chains are replaced in one iptables-restore transaction, so there is
    # no moment where a chain exists but is empty. --noflush leaves every other
    # chain alone.
    lines = ["*filter"] + [f":{chain} - [0:0]" for chain in chain_rules()]
    for chain, rules in chain_rules().items():
        lines += [" ".join(["-A", chain, *(_quote(a) for a in rule)]) for rule in rules]
    _run(["iptables-restore", "-w", "5", "--noflush"], input_text="\n".join(lines + ["COMMIT", ""]))

    for cmd, chain, rule in HOOKS:
        if _first_rule(cmd, chain) == rule:
            continue
        # Insert first, then remove displaced copies, so the hook is never absent.
        _run([*cmd, "-I", chain, "1", *rule])
        stale = [
            n for n, listed in enumerate(_rules(cmd, chain), start=1) if listed == rule and n > 1
        ]
        for number in reversed(stale):
            _run([*cmd, "-D", chain, str(number)])

    status = verify()
    write_status(status)
    return status


def _quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg


def read_status(now: float | None = None) -> dict:
    """The last provision result, trusted only if root wrote it recently."""
    try:
        st = STATUS_FILE.stat()
        data = json.loads(STATUS_FILE.read_text())
    except (OSError, ValueError):
        return {"ok": False, "problems": ["egress isolation has not been verified on this host"]}
    if st.st_uid != _STATUS_OWNER or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return {"ok": False, "problems": [f"{STATUS_FILE} is not a root-only file"]}
    age = (now if now is not None else time.time()) - float(data.get("checked_at") or 0)
    if not 0 <= age <= STATUS_MAX_AGE_S:
        return {"ok": False, "problems": [f"egress isolation was last verified {int(age)}s ago"]}
    return {**data, "ok": data.get("ok") is True}


def isolation_ready() -> bool:
    return read_status()["ok"]


def launch_args(job: dict, environment: dict | None) -> tuple[list[str], dict]:
    """``docker run`` arguments and environment for this job's egress policy."""
    env = dict(environment or {})
    if job_egress(job) == OPEN:
        return [], env
    status = read_status()
    if not status["ok"]:
        raise EgressUnavailable("; ".join(status.get("problems") or ["isolation check failed"]))
    proxy = f"http://{GATEWAY}:{PROXY_PORT}"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env[key] = proxy
    env["no_proxy"] = env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    return ["--network", NETWORK, "--dns", GATEWAY], env


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else ""
    if action == "provision":
        status = provision()
    elif action == "verify":
        status = verify()
    else:
        print("usage: egress_isolation.py provision|verify", file=sys.stderr)
        return 2
    print(json.dumps(status, indent=2))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
