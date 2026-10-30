"""Rented containers get restricted egress unless an administrator opened it.

The firewall and proxy behaviour itself was exercised end to end on a
disposable Docker 29 host (allowlisted registries reachable through the proxy;
direct connections, pool ports, DNS, host services, LAN, IPv6 and other tenants
blocked; published ports still reaching the container). These tests pin the
decisions around it: who is restricted, when a host may take restricted work,
and what the worker passes to ``docker run``.
"""

import copy
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

import abuse_policy
import egress_isolation as egress
from control_plane.scheduler import filters
from db import DatabaseOps, get_engine
from tests._db_helpers import PRICED_GPU, fund_test_owner

FIXTURE = Path(__file__).parent / "fixtures" / "docker29_network_inspect_xcelsior_tenant.json"


@pytest.fixture
def real_network():
    return json.loads(FIXTURE.read_text())[0]


@pytest.fixture
def status_file(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    monkeypatch.setattr(egress, "STATUS_FILE", path)
    # The worker trusts only a file written by root; tests run as a normal user.
    monkeypatch.setattr(egress, "_STATUS_OWNER", os.getuid())

    def write(ok=True, age=0.0, problems=(), mode=0o644):
        path.write_text(
            json.dumps({"ok": ok, "problems": list(problems), "checked_at": time.time() - age})
        )
        path.chmod(mode)
        return path

    return write


# ── Who is restricted ────────────────────────────────────────────────


@pytest.mark.parametrize("value", [None, "", "restricted", "OPEN?", "yes", "free", 1])
def test_anything_but_explicit_open_is_restricted(value):
    assert egress.job_egress({"egress": value}) == egress.RESTRICTED
    assert egress.job_egress({}) == egress.RESTRICTED
    assert egress.job_egress(None) == egress.RESTRICTED


def test_open_is_only_what_the_server_wrote():
    assert egress.job_egress({"egress": " Open "}) == egress.OPEN


def test_funded_is_explicit():
    assert egress.job_egress({"egress": " Funded "}) == egress.FUNDED


# ── What the worker passes to docker run ─────────────────────────────


def test_restricted_job_joins_the_bridge_and_uses_the_proxy(status_file):
    status_file(ok=True)
    args, env = egress.launch_args(
        {"egress": "restricted"}, {"HF_TOKEN": "kept", "HTTPS_PROXY": "http://renter-choice:8080"}
    )
    assert args == ["--network", egress.NETWORK, "--dns", egress.GATEWAY]
    proxy = f"http://{egress.GATEWAY}:{egress.PROXY_PORT}"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        assert env[key] == proxy
    assert env["HF_TOKEN"] == "kept"


def test_open_job_gets_no_network_arguments_even_without_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(egress, "STATUS_FILE", tmp_path / "missing.json")
    assert egress.launch_args({"egress": "open"}, {"A": "1"}) == ([], {"A": "1"})


@pytest.mark.parametrize(
    "state",
    [
        {"ok": False, "problems": ["proxy unreachable"]},
        {"ok": True, "age": egress.STATUS_MAX_AGE_S + 1},
        {"ok": True, "mode": 0o666},
    ],
    ids=["failing", "stale", "writable-by-others"],
)
def test_restricted_job_refused_when_isolation_is_not_verified(status_file, state):
    status_file(**state)
    with pytest.raises(egress.EgressUnavailable):
        egress.launch_args({"egress": "restricted"}, {})
    assert egress.isolation_ready() is False


def test_funded_job_gets_direct_internet_and_no_proxy(status_file):
    status_file(ok=True)
    args, env = egress.launch_args(
        {"egress": "funded"},
        {"HF_TOKEN": "kept", "https_proxy": "http://renter-choice:8080"},
    )
    assert args == [
        "--network",
        egress.FUNDED_NETWORK,
        "--dns",
        egress.PUBLIC_DNS[0],
        "--dns",
        egress.PUBLIC_DNS[1],
    ]
    assert "http_proxy" not in env
    assert "https_proxy" not in env
    assert env["HF_TOKEN"] == "kept"


def test_funded_job_refused_when_isolation_is_not_verified(status_file):
    status_file(ok=False, problems=["funded bridge missing"])
    with pytest.raises(egress.EgressUnavailable):
        egress.launch_args({"egress": "funded"}, {})


def test_missing_status_means_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(egress, "STATUS_FILE", tmp_path / "never-written.json")
    assert egress.isolation_ready() is False
    with pytest.raises(egress.EgressUnavailable):
        egress.launch_args({}, {})


def test_status_from_a_future_clock_is_not_trusted(status_file):
    status_file(ok=True, age=-3600)
    assert egress.isolation_ready() is False


# ── The isolation itself ─────────────────────────────────────────────


def test_provisioned_network_passes(real_network):
    assert egress.network_problems(real_network) == []


def test_internal_network_is_rejected_because_ssh_would_break(real_network):
    internal = copy.deepcopy(real_network)
    internal["Internal"] = True
    assert any("SSH" in p for p in egress.network_problems(internal))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda n: n["Options"].update({"com.docker.network.bridge.enable_icc": "true"}),
        lambda n: n.update({"EnableIPv6": True}),
        lambda n: n["IPAM"].update(
            {"Config": [{"Subnet": "172.30.0.0/16", "Gateway": "172.30.0.1"}]}
        ),
        lambda n: n["Options"].update({"com.docker.network.bridge.name": "br-other"}),
    ],
    ids=["icc", "ipv6", "subnet", "bridge-name"],
)
def test_weakened_network_is_rejected(real_network, mutate):
    changed = copy.deepcopy(real_network)
    mutate(changed)
    assert egress.network_problems(changed)


def test_missing_network_is_a_problem():
    assert egress.network_problems(None)


def test_forwarded_traffic_drops_everything_but_replies():
    out = egress.chain_rules()[egress.OUT_CHAIN]
    assert out[0] == ["-m", "conntrack", "--ctdir", "REPLY", "-j", "RETURN"]
    assert out[-1] == ["-j", "DROP"]
    assert all("ACCEPT" not in rule for rule in out)


def test_host_is_reachable_only_on_the_proxy_port():
    rules = egress.chain_rules()[egress.IN_CHAIN]
    accepts = [r for r in rules if r[-1] == "ACCEPT" and "--ctstate" not in r]
    assert accepts == [
        ["-d", egress.GATEWAY, "-p", "tcp", "--dport", str(egress.PROXY_PORT), "-j", "ACCEPT"]
    ]
    assert rules[-1] == ["-j", "DROP"]


def test_funded_internet_reaches_public_addresses_only():
    rules = egress.chain_rules()[egress.FUNDED_OUT_CHAIN]
    assert rules[0] == ["-m", "conntrack", "--ctdir", "REPLY", "-j", "RETURN"]
    assert rules[-1] == ["-j", "ACCEPT"]
    dropped = {rule[1] for rule in rules if rule[:1] == ["-d"] and rule[-2:] == ["-j", "DROP"]}
    for cidr in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "100.64.0.0/10",
    ):
        assert cidr in dropped
    assert all("--dport" not in rule for rule in rules)


def test_funded_bridge_cannot_reach_host_services():
    rules = egress.chain_rules()[egress.FUNDED_IN_CHAIN]
    assert rules == [
        ["-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
        ["-j", "DROP"],
    ]


def test_hooks_dispatch_only_our_bridges():
    rules = egress.chain_rules()
    assert egress.HOOKS[0][2] == ["-j", egress.DISPATCH_FWD]
    assert egress.HOOKS[1][2] == ["-j", egress.DISPATCH_IN]
    assert rules[egress.DISPATCH_FWD] == [
        ["-i", egress.BRIDGE, "-j", egress.OUT_CHAIN],
        ["-i", egress.FUNDED_BRIDGE, "-j", egress.FUNDED_OUT_CHAIN],
    ]
    assert rules[egress.DISPATCH_IN] == [
        ["-i", egress.BRIDGE, "-j", egress.IN_CHAIN],
        ["-i", egress.FUNDED_BRIDGE, "-j", egress.FUNDED_IN_CHAIN],
    ]


def test_proxy_allows_only_web_ports_to_approved_names():
    conf = egress.SQUID_CONF
    assert f"http_port {egress.GATEWAY}:{egress.PROXY_PORT}" in conf
    order = [line for line in conf.splitlines() if line.startswith("http_access")]
    assert order[-1] == "http_access deny all"
    assert order.index("http_access deny !approved") < order.index(
        "http_access allow tenants approved"
    )
    assert order.index("http_access deny private_dst") < order.index(
        "http_access allow tenants approved"
    )


def _blocked_name_proxy(delay_s: float):
    """A listener that answers like squid for a blocked name, after delay_s."""
    import http.server
    import socket
    import threading

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    class Squid(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(403)
            self.end_headers()

        def log_message(self, *args):
            pass

    servers = []

    def serve():
        time.sleep(delay_s)
        server = http.server.HTTPServer(("127.0.0.1", port), Squid)
        servers.append(server)
        server.serve_forever()

    threading.Thread(target=serve, daemon=True).start()
    return port, servers


def test_verify_waits_for_a_proxy_that_is_still_starting(monkeypatch):
    # At boot the proxy unit may start a moment after the check; that is not
    # a failed isolation.
    port, servers = _blocked_name_proxy(delay_s=1.5)
    monkeypatch.setattr(egress, "GATEWAY", "127.0.0.1")
    monkeypatch.setattr(egress, "PROXY_PORT", port)
    monkeypatch.setattr(egress, "PROXY_WAIT_S", 10.0)
    started = time.monotonic()
    try:
        assert egress._proxy_problems() == []
        assert time.monotonic() - started >= 1.0
    finally:
        for server in servers:
            server.shutdown()


def test_a_proxy_that_never_listens_is_reported(monkeypatch):
    port, _ = _blocked_name_proxy(delay_s=3600)
    monkeypatch.setattr(egress, "GATEWAY", "127.0.0.1")
    monkeypatch.setattr(egress, "PROXY_PORT", port)
    monkeypatch.setattr(egress, "PROXY_WAIT_S", 1.5)
    started = time.monotonic()
    problems = egress._proxy_problems()
    assert problems and problems[0].startswith("proxy unreachable")
    assert time.monotonic() - started >= 1.5


def test_squid_logs_to_the_journal_not_a_reopened_stderr():
    # Under systemd stderr is a socket; squid cannot open /dev/stderr there.
    assert "cache_log /dev/null" in egress.SQUID_CONF
    assert "/dev/stderr" not in egress.SQUID_CONF


def test_allowlist_has_no_overlaps_squid_would_reject():
    names = egress.ALLOWED_DOMAINS
    assert len(set(names)) == len(names)
    for wildcard in (n for n in names if n.startswith(".")):
        assert not any(n != wildcard and (n == wildcard[1:] or n.endswith(wildcard)) for n in names)


@pytest.mark.skipif(not shutil.which("squid"), reason="squid is not installed")
def test_squid_accepts_the_rendered_config(tmp_path):
    domains = tmp_path / "domains.txt"
    domains.write_text("\n".join(egress.ALLOWED_DOMAINS) + "\n")
    conf = (
        egress.SQUID_CONF.replace(str(egress.CONFIG_DIR / "domains.txt"), str(domains))
        .replace("/run/xcelsior-egress-proxy/squid.pid", str(tmp_path / "squid.pid"))
        .replace(
            f"http_port {egress.GATEWAY}:{egress.PROXY_PORT}", "http_port 127.0.0.1:39128"
        )  # parse only, nothing binds
    )
    path = tmp_path / "squid.conf"
    path.write_text(conf)
    result = subprocess.run(
        ["squid", "-k", "parse", "-f", str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "FATAL" not in result.stderr


# ── Placement ────────────────────────────────────────────────────────


def _host(**over):
    host = {
        "host_id": "h1",
        "status": "active",
        "administrative_state": "admitted",
        "free_vram_gb": 12.0,
        "gpu_count": 1,
        "free_gpu_count": 1,
        "capabilities": {"egress_isolation": True},
    }
    host.update(over)
    return host


def test_scheduler_sends_restricted_work_only_to_isolated_hosts():
    import scheduler

    isolated, plain = _host(), _host(host_id="h2", capabilities={"checkpoint": {"class": ""}})
    restricted = {"egress": "restricted"}
    assert scheduler.host_can_run_egress(isolated, restricted)
    assert not scheduler.host_can_run_egress(plain, restricted)
    assert not scheduler.host_can_run_egress(
        _host(capabilities={"egress_isolation": "true"}), restricted
    )
    assert not scheduler.host_can_run_egress(_host(capabilities=None), {})
    assert scheduler.host_can_run_egress(plain, {"egress": "open"})
    assert scheduler.host_can_run_egress(isolated, {"egress": "funded"})
    assert not scheduler.host_can_run_egress(plain, {"egress": "funded"})


def test_control_plane_filter_explains_the_rejection():
    reason = filters.egress_isolation(
        {"egress": "restricted"}, _host(capabilities={}), filters.FilterContext()
    )
    assert reason.code == "egress_isolation_unavailable"
    assert filters.egress_isolation({}, _host(), filters.FilterContext()) is None
    assert (
        filters.egress_isolation(
            {"egress": "open"}, _host(capabilities={}), filters.FilterContext()
        )
        is None
    )


# ── Account policy ───────────────────────────────────────────────────


@pytest.fixture
def account():
    uid = "egress-test-" + uuid.uuid4().hex
    yield uid
    with get_engine().transaction() as (conn, _backend):
        conn.execute("DELETE FROM state WHERE namespace LIKE %s", (f"security.%.{uid}",))


def test_accounts_are_restricted_until_an_administrator_opens_them(account):
    assert abuse_policy.egress_for_owner({"owner": account}) == egress.RESTRICTED
    with get_engine().transaction() as (conn, backend):
        abuse_policy.set_egress_open(
            conn,
            user_id=account,
            active=True,
            actor="admin-test",
            reason="verified lab",
            backend=backend,
        )
    assert abuse_policy.egress_for_owner({"owner": account}) == egress.OPEN
    with get_engine().transaction() as (conn, backend):
        abuse_policy.set_egress_open(
            conn,
            user_id=account,
            active=False,
            actor="admin-test",
            reason="revoked",
            backend=backend,
        )
    assert abuse_policy.egress_for_owner({"owner": account}) == egress.RESTRICTED


def test_a_settled_card_deposit_opens_public_internet(account):
    with get_engine().transaction() as (conn, backend):
        abuse_policy.set_card_paid(
            conn,
            user_id=account,
            active=True,
            actor="billing",
            reason="settled card deposit",
            backend=backend,
        )
    assert abuse_policy.egress_for_owner({"owner": account}) == egress.FUNDED


def test_mining_revocation_closes_internet_even_after_payment(account):
    with get_engine().transaction() as (conn, backend):
        abuse_policy.set_card_paid(
            conn,
            user_id=account,
            active=True,
            actor="billing",
            reason="settled card deposit",
            backend=backend,
        )
        abuse_policy.record_mining_violation(
            conn, {"job_id": "mine-1", "owner": account}, backend=backend
        )
    assert abuse_policy.egress_for_owner({"owner": account}) == egress.RESTRICTED


def test_opening_egress_requires_actor_and_reason():
    with pytest.raises(ValueError):
        abuse_policy.set_egress_open(None, user_id="u", active=True, actor="", reason="x")


def test_submitted_job_carries_the_account_decision(account):
    import scheduler

    job = scheduler.submit_job(
        "egress-probe", 0, gpu_model=PRICED_GPU, owner=fund_test_owner(account)
    )
    try:
        assert job["egress"] == egress.RESTRICTED
    finally:
        with scheduler._atomic_mutation() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job["job_id"],))


def test_suspension_check_is_one_query(account, monkeypatch):
    with get_engine().transaction() as (conn, backend):
        abuse_policy.set_account_suspension(
            conn, user_id=account, active=True, actor="admin-test", reason="test", backend=backend
        )
    calls = []
    real = DatabaseOps.get_states
    monkeypatch.setattr(
        DatabaseOps, "get_state", lambda *a, **k: pytest.fail("per-identifier lookup")
    )
    monkeypatch.setattr(
        DatabaseOps, "get_states", staticmethod(lambda *a, **k: calls.append(1) or real(*a, **k))
    )
    principal = {
        "user_id": "someone-else",
        "email": "x@example.invalid",
        "customer_id": account,
        "sub": "s",
    }
    assert abuse_policy.account_suspended(principal) is True
    assert calls == [1]
