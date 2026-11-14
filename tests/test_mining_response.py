"""CVP-01 §3: the resource-abuse response acts on one container, by evidence, one rung at a time."""

import json
from types import SimpleNamespace

import pytest

import mining_response as mr

GIB = 1024**3
POOL_PORTS = (3333, 4444)


def t(minutes: float) -> float:
    return 1_000_000.0 + minutes * 60


def gpu_sample(**over):
    g = {
        "utilization": 99,
        "memory_util": 60,
        "pcie_tx_mb_s": 0.4,
        "pcie_rx_mb_s": 0.3,
        "power_draw_w": 196,
        "power_limit_w": 200,
        "memory_used_bytes": 6 * GIB,
    }
    g.update(over)
    return g


# ── scoring ───────────────────────────────────────────────────────────


def test_independent_families_combine_and_a_family_counts_once():
    assert mr.combine(0.6, 0.75) == pytest.approx(0.9)
    s1a = mr.Signal("S1", 0.6, "", 0)
    s1b = mr.Signal("S1", 0.7, "", 0)
    assert mr.score([s1a, s1b]) == 0.7
    assert mr.score([s1b, mr.Signal("S3", 0.75, "", 0)]) == pytest.approx(0.925)


def test_prior_adds_at_most_five_points_and_never_acts_alone():
    assert mr.score([], prior=0.05) == 0.0
    assert mr.score([mr.Signal("S4", 0.5, "", 0)], prior=0.5) == 0.55


def test_gpu_only_evidence_is_flagged_and_never_acted_on():
    """T8b, 2026-11-13: ordinary training and inference matched the GPU profile on
    every sample. Whatever its score, that signal alone changes nothing for a tenant."""
    s1 = mr.Signal("S1", mr.S1_LONG, "", 0)
    ep = mr.Episode("j", "c", confidence=mr.score([s1], 0.05))
    ep.signals = [s1]
    ep.above_throttle_since = t(0)
    assert ep.confidence == 0.75
    assert mr.decide(ep, t(11)) == mr.FLAG
    assert mr.decide(ep, t(120)) == mr.FLAG


# ── the ladder ────────────────────────────────────────────────────────


def test_flag_then_throttle_after_ten_minutes_above_seventy():
    ep = mr.Episode("j", "c", confidence=0.72, above_throttle_since=t(0))
    ep.signals = [mr.Signal("S1", mr.S1_SHORT, "", 0), mr.Signal("S4", 0.5, "", 0)]
    assert mr.decide(ep, t(1)) == mr.FLAG
    assert mr.decide(ep, t(10)) == mr.THROTTLE


def test_suspend_needs_ninety_or_seventy_with_network_for_half_an_hour():
    ep = mr.Episode("j", "c", confidence=0.9)
    assert mr.decide(ep, t(0)) == mr.SUSPEND
    ep = mr.Episode("j", "c", confidence=0.75, above_throttle_since=t(0))
    ep.signals = [mr.Signal("S3", mr.S3_POOL, "", 0)]
    assert mr.decide(ep, t(29)) == mr.THROTTLE
    assert mr.decide(ep, t(30)) == mr.SUSPEND


def test_terminate_only_with_a_prior_incident_or_a_decision():
    ep = mr.Episode("j", "c", confidence=0.97)
    assert mr.decide(ep, t(0)) == mr.SUSPEND
    ep.repeat_offender = True
    assert mr.decide(ep, t(0)) == mr.TERMINATE
    ep = mr.Episode("j", "c", confidence=0.5, decision="terminate")
    assert mr.decide(ep, t(0)) == mr.TERMINATE


def test_rungs_never_go_down_on_their_own():
    ep = mr.Episode("j", "c", confidence=0.2, rung=mr.THROTTLE)
    assert mr.decide(ep, t(0)) == mr.THROTTLE


# ── S1 ────────────────────────────────────────────────────────────────


def test_gpu_profile_needs_fifteen_matching_minutes_and_a_flat_footprint():
    samples = [(t(m), gpu_sample()) for m in range(16)]
    sig = mr.gpu_profile_signal(samples, t(15))
    assert sig and sig.family == "S1" and sig.strength == mr.S1_SHORT
    assert mr.gpu_profile_signal(samples[:8], t(8)) is None
    drifting = [(t(m), gpu_sample(memory_used_bytes=(6 + m % 2) * GIB)) for m in range(16)]
    assert mr.gpu_profile_signal(drifting, t(15)) is None


@pytest.mark.parametrize(
    "over",
    [
        {"pcie_rx_mb_s": 40},  # single-GPU training: host-to-device traffic
        {"power_draw_w": 120},  # sampling and benchmarks breathe
        {"memory_util": 5},
        {"utilization": 93},
    ],
)
def test_gpu_profile_rejects_the_known_training_shapes(over):
    samples = [(t(m), gpu_sample(**over)) for m in range(16)]
    assert mr.gpu_profile_signal(samples, t(15)) is None


def test_an_hour_of_profile_is_worth_more():
    samples = [(t(m), gpu_sample()) for m in range(61)]
    sig = mr.gpu_profile_signal(samples, t(60))
    assert sig and sig.strength == mr.S1_LONG


# ── S2 ────────────────────────────────────────────────────────────────

TOP = "PID RSS COMMAND CMD\n17 204800 python3 python3 train.py --epochs 3\n"


def test_process_identity_defaults_to_the_pool_url_shape_only(tmp_path):
    sig = mr.load_signatures(str(tmp_path / "absent.json"))
    assert sig["names"] == [] and list(sig["arg_patterns"]) == list(mr.DEFAULT_ARG_PATTERNS)
    rows = mr.parse_docker_top(TOP)
    assert rows == [
        {"pid": 17, "rss": 204800 * 1024, "comm": "python3", "args": "python3 train.py --epochs 3"}
    ]
    assert mr.process_signal(rows, sig, t(0)) is None
    pooled = rows + [{"pid": 18, "rss": 0, "comm": "x", "args": "x -o stratum+tcp://h:3333"}]
    hit = mr.process_signal(pooled, sig, t(0))
    assert hit and hit.strength == mr.S2_ARGS


def test_operator_catalogue_adds_names_and_patterns_and_ignores_bad_entries(tmp_path):
    path = tmp_path / "sig.json"
    path.write_text(json.dumps({"names": ["Loud", 3, ""], "arg_patterns": ["--algo", "("]}))
    sig = mr.load_signatures(str(path))
    assert sig["names"] == ["loud"]
    assert "--algo" in sig["arg_patterns"] and "(" not in sig["arg_patterns"]
    rows = [{"pid": 1, "rss": 0, "comm": "/usr/bin/loud", "args": "loud"}]
    hit = mr.process_signal(rows, sig, t(0))
    assert hit and hit.strength == mr.S2_NAME


# ── S3 ────────────────────────────────────────────────────────────────


def test_funded_connection_counts_once_it_has_lived_ten_minutes():
    tracker = mr.ConnectionTracker()
    snap = {"connections": [{"dst": "203.0.113.9", "dport": 3333}]}
    assert mr.network_signal(tracker, snap, t(0), "funded", POOL_PORTS) is None
    assert mr.network_signal(tracker, snap, t(9), "funded", POOL_PORTS) is None
    sig = mr.network_signal(tracker, snap, t(10), "funded", POOL_PORTS)
    assert sig and sig.strength == mr.S3_POOL and "203.0.113.9:3333" in sig.detail


def test_a_connection_that_went_away_starts_over():
    tracker = mr.ConnectionTracker()
    snap = {"connections": [{"dst": "203.0.113.9", "dport": 4444}]}
    mr.network_signal(tracker, snap, t(0), "funded", POOL_PORTS)
    mr.network_signal(tracker, {"connections": []}, t(1), "funded", POOL_PORTS)
    mr.network_signal(tracker, {"connections": []}, t(4), "funded", POOL_PORTS)
    assert mr.network_signal(tracker, snap, t(12), "funded", POOL_PORTS) is None


def test_non_pool_ports_and_short_lived_connections_say_nothing():
    tracker = mr.ConnectionTracker()
    snap = {"connections": [{"dst": "203.0.113.9", "dport": 443}]}
    for m in range(0, 40, 1):
        assert mr.network_signal(tracker, snap, t(m), "funded", POOL_PORTS) is None


def test_restricted_attempts_only_flag():
    tracker = mr.ConnectionTracker()
    dropped = [{"dst": "203.0.113.9", "dport": 3333}] * 3
    sig = mr.network_signal(tracker, {"dropped": dropped}, t(0), "restricted", POOL_PORTS)
    assert sig and sig.strength == mr.S3_ATTEMPTED < mr.THROTTLE_AT
    assert (
        mr.network_signal(tracker, {"dropped": dropped[:2]}, t(0), "restricted", POOL_PORTS) is None
    )


# ── S4 ────────────────────────────────────────────────────────────────


def test_cpu_signal_needs_every_core_one_process_and_thirty_minutes():
    rows = [
        {"pid": 5, "rss": 3 * GIB, "comm": "a", "args": "a"},
        {"pid": 6, "rss": 50 * 2**20, "comm": "b", "args": "b"},
    ]
    stats = mr.parse_docker_stats("745.3%|7")
    assert stats == {"cpu_pct": 745.3, "pids": 7}
    sig, since = mr.cpu_signal(stats, rows, 8.0, None, t(0))
    assert sig is None and since == t(0)
    sig, since = mr.cpu_signal(stats, rows, 8.0, since, t(29))
    assert sig is None
    sig, since = mr.cpu_signal(stats, rows, 8.0, since, t(30))
    assert sig and sig.strength == mr.S4
    # Many balanced processes (a data loader) or idle cores reset the clock.
    sig, since = mr.cpu_signal(mr.parse_docker_stats("300%|7"), rows, 8.0, since, t(31))
    assert sig is None and since is None
    spread = [{"pid": i, "rss": GIB, "comm": "w", "args": "w"} for i in range(4)]
    assert mr.cpu_signal(stats, spread, 8.0, None, t(0)) == (None, None)


# ── the responder, with every effect captured ─────────────────────────


class Host:
    """Fake docker and root helper: records what was asked, answers from a script."""

    def __init__(self):
        self.calls = []
        self.privd_calls = []
        self.top = TOP
        self.stats = "12.0%|3"
        self.cpus = 8.0
        self.paused = False
        self.exists = True
        self.snapshot = {"ok": True, "connections": [], "dropped": []}
        self.reply = {}
        self.notes = []
        self.terminated = []

    def docker(self, args, timeout):
        self.calls.append(args)
        if args[0] == "inspect":
            if not self.exists:
                return 1, ""
            return (
                0,
                f"cid|{int(self.cpus * 1e9)}|{'true' if self.paused else 'false'}|xcelsior-funded=172.30.243.7 ",
            )
        if args[0] == "top":
            return 0, self.top
        if args[0] == "stats":
            return 0, self.stats
        if args[0] == "pause":
            self.paused = True
        if args[0] == "unpause":
            self.paused = False
        return 0, ""

    def privd(self, op, **kw):
        self.privd_calls.append((op, kw))
        if op == "tenant_net_snapshot":
            return self.snapshot
        if op == "gpu_power_limit":
            return {"ok": True, "previous_w": 200, "limit_w": 100}
        return {"ok": True}

    def report(self, ep, evidence):
        return dict(self.reply)

    def notify(self, job_id, text):
        self.notes.append(text)

    def terminate(self, job_id, cname):
        self.terminated.append((job_id, cname))


@pytest.fixture
def host(tmp_path):
    return Host()


def responder(host, tmp_path, clock):
    return mr.Responder(
        docker=host.docker,
        privd=host.privd,
        report=host.report,
        notify=host.notify,
        terminate=host.terminate,
        pool_ports=POOL_PORTS,
        state_path=str(tmp_path / "state.json"),
        signatures={"names": [], "arg_patterns": list(mr.DEFAULT_ARG_PATTERNS)},
        clock=clock,
    )


def observe(r, **over):
    kw = {"gpu_keys": ["GPU-1"], "shared_gpu": False, "gpu_indices": {"GPU-1": 0}}
    kw.update(over)
    return r.observe("job-1", "xcl-1", **kw)


def test_a_quiet_container_opens_no_episode(host, tmp_path):
    r = responder(host, tmp_path, lambda: t(0))
    assert observe(r) is None
    assert r.episodes == {}


def test_gpu_profile_alone_is_flagged_and_changes_nothing_for_the_tenant(host, tmp_path):
    now = SimpleNamespace(v=t(0))
    r = responder(host, tmp_path, lambda: now.v)
    for m in range(0, 130):
        now.v = t(m)
        r.record_gpu_samples([{"index": 0, "gpu_uuid": "GPU-1", **gpu_sample()}], now.v)
        ep = observe(r)
        if m == 15:
            assert ep and ep.rung == mr.FLAG
    assert ep.rung == mr.FLAG, "two hours of the GPU profile alone must not throttle"
    assert not host.paused and host.terminated == []
    assert not [c for c in host.calls if c[0] == "update"]
    assert not [op for op, _ in host.privd_calls if op in ("gpu_power_limit", "tenant_rate_limit")]
    assert ep.throttle == {}
    assert host.notes == [], "a flag is for the operator; the tenant is told nothing"


CATALOGUED = TOP + "18 1024 minerd minerd --threads 8\n"


def catalogued(host, tmp_path, clock):
    """A responder whose operator catalogue names one process (S2 by name, 0.80)."""
    r = responder(host, tmp_path, clock)
    r.signatures = {"names": ["minerd"], "arg_patterns": list(mr.DEFAULT_ARG_PATTERNS)}
    host.top = CATALOGUED
    return r


def test_a_catalogued_process_throttles_with_every_limit(host, tmp_path):
    now = SimpleNamespace(v=t(0))
    r = catalogued(host, tmp_path, lambda: now.v)
    for m in range(0, 15):
        now.v = t(m)
        ep = observe(r)
    assert ep.rung == mr.THROTTLE
    assert not host.paused and host.terminated == []
    assert ["update", "--cpus=4.0", "xcl-1"] in host.calls
    assert ("gpu_power_limit", {"index": 0, "action": "min"}) in host.privd_calls
    assert (
        "tenant_rate_limit",
        {"ip": "172.30.243.7", "mbps": mr.THROTTLE_NET_MBPS, "action": "apply"},
    ) in host.privd_calls
    assert ep.throttle == {"cpus": 8.0, "gpu_power": {"0": 200}, "net": "172.30.243.7"}
    assert any("throttled" in n for n in host.notes)


def test_shared_gpu_skips_the_power_rung(host, tmp_path):
    now = SimpleNamespace(v=t(0))
    r = catalogued(host, tmp_path, lambda: now.v)
    for m in range(0, 15):
        now.v = t(m)
        observe(r, shared_gpu=True)
    assert r.episodes["xcl-1"].rung == mr.THROTTLE
    assert all(op != "gpu_power_limit" for op, _ in host.privd_calls)
    assert ["update", "--cpus=4.0", "xcl-1"] in host.calls


def test_pool_url_in_a_process_suspends_at_once_and_keeps_state(host, tmp_path):
    host.top = TOP + "18 1024 x x -o stratum+tcp://h:3333\n"
    r = responder(host, tmp_path, lambda: t(0))
    ep = observe(r)
    assert ep.rung == mr.SUSPEND and host.paused
    assert ep.throttle.get("cpus") == 8.0  # the throttle rung was passed through
    assert host.terminated == []
    assert any("suspended" in n for n in host.notes)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["episodes"][0]["rung"] == mr.SUSPEND


def test_a_second_incident_on_the_account_terminates(host, tmp_path):
    host.top = TOP + "18 1024 x x -o stratum+tcp://h:3333\n"
    host.reply = {"repeat_offender": True}
    r = responder(host, tmp_path, lambda: t(0))
    observe(r)
    assert host.terminated == [("job-1", "xcl-1")]
    assert r.episodes == {}


def test_administrator_terminate_decision_ends_it(host, tmp_path):
    host.top = TOP + "18 1024 x x -o stratum+tcp://h:3333\n"
    r = responder(host, tmp_path, lambda: t(0))
    observe(r)
    host.reply = {"decision": "terminate"}
    observe(r)
    assert host.terminated == [("job-1", "xcl-1")]


def test_administrator_clear_lifts_everything_and_grants_a_day_of_grace(host, tmp_path):
    host.top = TOP + "18 1024 x x -o stratum+tcp://h:3333\n"
    now = SimpleNamespace(v=t(0))
    r = responder(host, tmp_path, lambda: now.v)
    observe(r)
    host.reply = {"decision": "clear"}
    now.v = t(1)
    assert observe(r) is not None
    assert not host.paused
    assert ["update", "--cpus=8.0", "xcl-1"] in host.calls
    assert ("gpu_power_limit", {"index": 0, "action": "restore", "watts": 200}) in host.privd_calls
    assert ("tenant_rate_limit", {"ip": "172.30.243.7", "action": "clear"}) in host.privd_calls
    assert r.episodes == {}
    host.reply = {}
    now.v = t(2)
    assert observe(r) is None  # still matching, but cleared for 24 h
    now.v = t(2) + mr.CLEARED_GRACE_S
    assert observe(r) is not None


def test_suspension_resumes_after_a_day_without_a_decision(host, tmp_path):
    host.top = TOP + "18 1024 x x -o stratum+tcp://h:3333\n"
    now = SimpleNamespace(v=t(0))
    r = responder(host, tmp_path, lambda: now.v)
    observe(r)
    now.v = t(0) + mr.AUTO_RESUME_S
    ep = observe(r)
    assert not host.paused and ep.rung == mr.THROTTLE
    assert ep.throttle  # still throttled
    assert any("resumed" in n for n in host.notes)
    # The same evidence does not suspend it again; a decision still can.
    now.v += 600
    assert observe(r).rung == mr.THROTTLE and not host.paused
    host.reply = {"decision": "terminate"}
    now.v += 60
    observe(r)
    assert host.terminated == [("job-1", "xcl-1")]


def test_signals_that_go_away_close_the_episode_and_undo_the_throttle(host, tmp_path):
    now = SimpleNamespace(v=t(0))
    r = catalogued(host, tmp_path, lambda: now.v)
    for m in range(0, 15):
        now.v = t(m)
        observe(r)
    assert r.episodes["xcl-1"].rung == mr.THROTTLE
    host.top = TOP
    for m in range(15, 47):
        now.v = t(m)
        observe(r)
    assert r.episodes == {}
    assert ["update", "--cpus=8.0", "xcl-1"] in host.calls
    assert ("tenant_rate_limit", {"ip": "172.30.243.7", "action": "clear"}) in host.privd_calls


def test_a_container_that_is_gone_is_forgotten(host, tmp_path):
    host.top = TOP + "18 1024 x x -o stratum+tcp://h:3333\n"
    r = responder(host, tmp_path, lambda: t(0))
    observe(r)
    host.exists = False
    assert observe(r) is None
    assert r.episodes == {}


def test_state_survives_a_restart_for_containers_that_still_exist(host, tmp_path):
    host.top = TOP + "18 1024 x x -o stratum+tcp://h:3333\n"
    r = responder(host, tmp_path, lambda: t(0))
    observe(r)
    again = responder(host, tmp_path, lambda: t(5))
    assert again.episodes["xcl-1"].rung == mr.SUSPEND
    again.restore_after_restart(live=set())
    assert again.episodes == {}


def test_the_control_plane_prior_is_capped(host, tmp_path):
    host.reply = {"prior": 0.9}
    now = SimpleNamespace(v=t(0))
    r = responder(host, tmp_path, lambda: now.v)
    for m in range(0, 16):
        now.v = t(m)
        r.record_gpu_samples([{"gpu_uuid": "GPU-1", **gpu_sample()}], now.v)
        ep = observe(r)
    assert ep.prior == mr.S5_MAX and ep.confidence == 0.65


# ── T8b: what real GPUs reported, 2026-11-13 ──────────────────────────
#
# RTX 2060 and RTX 3060, sampled through nvml_telemetry while idle and under
# ordinary PyTorch work. Neither card reports a power limit, and the PCIe
# counters read 0 on every sample, busy or not.

MEASURED = {
    "training, data already on the GPU": {"utilization": 100.0, "memory_util": 98.0},
    "inference, fixed weights": {"utilization": 99.0, "memory_util": 100.0},
}


@pytest.mark.parametrize("name", sorted(MEASURED))
def test_ordinary_work_on_a_card_with_dead_pcie_counters_is_not_a_profile_match(name):
    sample = gpu_sample(
        pcie_tx_mb_s=0, pcie_rx_mb_s=0, power_draw_w=85.0, power_limit_w=None, **MEASURED[name]
    )
    assert mr._s1_sample_matches(sample), "the per-sample rule alone does match; that was the bug"
    series = [(t(m), sample) for m in range(0, 70)]
    assert mr.gpu_profile_signal(series, t(69)) is None


def test_one_reading_proves_the_counter_works():
    quiet = gpu_sample(pcie_tx_mb_s=0, pcie_rx_mb_s=0)
    series = [(t(-5), gpu_sample(pcie_tx_mb_s=0.2, pcie_rx_mb_s=0))] + [
        (t(m), quiet) for m in range(0, 16)
    ]
    assert mr.gpu_profile_signal(series, t(15)) is not None
