"""Verify the administrator-provisioned free-tier boundary before Docker starts.

Containers use an internal bridge and a host proxy with a domain allowlist.
The host INPUT chain blocks access to every other host service; inter-container
traffic and raw sockets are disabled. Missing controls are launch failures.
"""
import json
import os
from pathlib import Path
import stat
import subprocess
import urllib.request

NETWORK = "xcelsior-free"
BRIDGE = "xcl-free0"
CHAIN = "XCL-FREE-IN"
SUBNET = "172.30.242.0/24"
GATEWAY = "172.30.242.1"
PORT = 3128


def firewall_rules():
    return [
        ["INPUT", "-i", BRIDGE, "-j", CHAIN],
        [CHAIN, "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
        [CHAIN, "-s", SUBNET, "-d", GATEWAY, "-p", "tcp", "--dport", str(PORT), "-j", "ACCEPT"],
        [CHAIN, "-j", "DROP"],
    ]


def privileged_iptables():
    return (["sudo", "-n"] if os.geteuid() else []) + ["/usr/sbin/iptables", "-w", "5"]


def validate_free_tier_network():
    result = subprocess.run(["docker", "network", "inspect", NETWORK], capture_output=True, text=True, timeout=10, check=True)
    network = json.loads(result.stdout)[0]
    options = network.get("Options") or {}
    if not (
        network.get("Internal") is True
        and network.get("EnableIPv6") is False
        and network.get("Driver") == "bridge"
        and options.get("com.docker.network.bridge.name") == BRIDGE
        and options.get("com.docker.network.bridge.enable_icc") == "false"
        and network.get("IPAM", {}).get("Config") == [{"Subnet": SUBNET, "Gateway": GATEWAY}]
    ):
        raise RuntimeError("Free-tier network does not satisfy the isolation policy")
    for rule in firewall_rules():
        subprocess.run(privileged_iptables() + ["-C"] + rule, capture_output=True, timeout=10, check=True)
    # Configuration must be root-controlled, never writable by a renter/worker.
    config = Path(os.environ.get("XCELSIOR_FREE_TIER_SQUID_CONFIG", "/etc/xcelsior/egress/squid.conf"))
    st = config.stat()
    if st.st_uid != 0 or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("Free-tier proxy configuration must be root-owned and not writable by others")
    # An explicit HTTP response proves the configured proxy is serving. This
    # URL must be rejected by its ACL, even when no upstream network is available.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": f"http://{GATEWAY}:{PORT}"}))
    try:
        opener.open("http://egress-check.invalid/", timeout=5)
    except urllib.error.HTTPError as exc:
        if exc.code != 403:
            raise RuntimeError("Free-tier proxy failed its deny check") from exc
    else:
        raise RuntimeError("Free-tier proxy accepted a non-allowlisted host")
    return {"http_proxy": f"http://{GATEWAY}:{PORT}", "https_proxy": f"http://{GATEWAY}:{PORT}"}


def prepare_free_tier(job, environment):
    if str(job.get("tier") or "").lower() != "free":
        return [], environment
    proxy = validate_free_tier_network()
    env = dict(environment)
    env.update(proxy)
    env.update({key.upper(): value for key, value in proxy.items()})
    env.update({"NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1"})
    return ["--network", NETWORK, "--dns", "127.0.0.1"], env
