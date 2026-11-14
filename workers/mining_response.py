"""Resource abuse on a rented GPU: detection signals and a per-container response.

CVP-01 brief §3. The unit of action is one container, named by the labels
Docker already carries; nothing here touches another tenant, the host's own
processes, or the account's other instances. A response rung is never taken
on the GPU profile alone, because that is the one signal with a real
false-positive risk (single-GPU training, diffusion sampling, long
benchmarks all look like it for minutes at a time).

Signals (each returns a strength in [0, 1] or nothing):

* S1 GPU profile over a window: SM busy, memory controller busy, power at
  the limit, PCIe quiet in *both* directions, VRAM footprint flat. 15 minutes
  of it is 0.60, an hour is 0.70. Never enough to suspend by itself.
* S2 process identity from ``docker top`` on the host (no exec into the
  tenant): a pool URL shape in a process's arguments is 0.95; a process name
  from the operator's signature file is 0.80. The signature file
  (``/etc/xcelsior/mining-signatures.json``) is read by the worker, so the
  catalogue can be updated without a signed release.
* S3 network, funded bridge only: one connection from the container that
  stays open for ten minutes to a conventional pool port is 0.75. On the
  restricted bridge nothing reaches a pool, so repeated dropped attempts to
  a pool port are logged as "attempted" at 0.40: a flag, never an action.
* S4 CPU: every granted vCPU pegged for thirty minutes by one process with a
  large resident set is 0.50.
* S5 account prior: the control plane may add up to 0.05 to a flagged
  episode. It never acts alone.

Independent families combine as ``1 - prod(1 - s)``. Thresholds and rungs
are in :func:`decide`; the actions are in :class:`Responder`.

Every host-side effect goes through a seam (``docker``, ``privd``) so the
decisions are tested without a GPU, and so the root helper, not the worker,
touches the GPU power limit and the firewall.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("xcelsior.mining")

# ── Rungs and thresholds (brief §3.2, §3.3) ───────────────────────────

FLAG = "flag"
THROTTLE = "throttle"
SUSPEND = "suspend"
TERMINATE = "terminate"
RUNGS = (FLAG, THROTTLE, SUSPEND, TERMINATE)

FLAG_AT = 0.40
THROTTLE_AT = 0.70
THROTTLE_SUSTAIN_S = 10 * 60
SUSPEND_AT = 0.90
SUSPEND_WITH_NETWORK_AT = 0.70
SUSPEND_WITH_NETWORK_SUSTAIN_S = 30 * 60
TERMINATE_AT = 0.90
#: A suspended container resumes by itself when nobody decides.
AUTO_RESUME_S = 24 * 3600
#: An episode whose signals have gone for this long closes and undoes its throttle.
CLEAR_AFTER_QUIET_S = 30 * 60
#: After an administrator clears a container, its signals are ignored for this long.
CLEARED_GRACE_S = 24 * 3600

# Signal strengths (brief §3.2)
S1_WINDOW_S = 15 * 60
S1_LONG_WINDOW_S = 60 * 60
S1_SHORT = 0.60
S1_LONG = 0.70
S2_ARGS = 0.95
S2_NAME = 0.80
S3_POOL = 0.75
S3_ATTEMPTED = 0.40
S3_LONG_LIVED_S = 10 * 60
S4_SUSTAIN_S = 30 * 60
S4 = 0.50
S5_MAX = 0.05

# S1 thresholds; the environment can retune them on one host for T8.
S1_UTIL = float(os.environ.get("XCELSIOR_MINING_GPU_UTIL", "95"))
S1_MEM_UTIL = float(os.environ.get("XCELSIOR_MINING_MEM_UTIL", "40"))
S1_PCIE_MB = float(os.environ.get("XCELSIOR_MINING_PCIE_MB", "5"))
S1_POWER_FRACTION = float(os.environ.get("XCELSIOR_MINING_POWER_FRACTION", "0.95"))
S1_VRAM_DRIFT = float(os.environ.get("XCELSIOR_MINING_VRAM_DRIFT", "0.02"))
#: Samples arrive once a check interval; a window needs most of its samples.
S1_MIN_SAMPLES = int(os.environ.get("XCELSIOR_MINING_S1_MIN_SAMPLES", "12"))

S4_CPU_FRACTION = 0.90
S4_RSS_BYTES = 1024**3
S4_DOMINANT_RSS = 0.80

#: Throttle rung: what the offending container keeps.
THROTTLE_CPU_FRACTION = 0.5
THROTTLE_NET_MBPS = int(os.environ.get("XCELSIOR_MINING_THROTTLE_MBPS", "5"))

SIGNATURE_FILE = os.environ.get(
    "XCELSIOR_MINING_SIGNATURES", "/etc/xcelsior/mining-signatures.json"
)
#: The one built-in process signature: the URL scheme mining pools speak.
#: Names and further argument shapes come from the operator's file.
DEFAULT_ARG_PATTERNS = (r"\bstratum2?\+(tcp|ssl|tls)://",)

FAILURE_CODE = "mining_detected"


def _state_dir() -> str:
    from_systemd = os.environ.get("STATE_DIRECTORY", "").split(":")[0]
    return os.environ.get("XCELSIOR_STATE_DIR") or from_systemd or "/var/lib/xcelsior"


STATE_PATH = os.environ.get(
    "XCELSIOR_MINING_STATE_PATH", os.path.join(_state_dir(), "mining_response.json")
)


def combine(*strengths: float) -> float:
    """Independent evidence: ``1 - prod(1 - s)``."""
    miss = 1.0
    for s in strengths:
        miss *= 1.0 - max(0.0, min(1.0, float(s)))
    return round(1.0 - miss, 4)


# ── Episode state ─────────────────────────────────────────────────────


@dataclass
class Signal:
    family: str  # S1..S5
    strength: float
    detail: str
    since: float


@dataclass
class Episode:
    """One container under observation. Rungs go up; only a decision or the clock brings them down."""

    job_id: str
    container_name: str
    container_id: str = ""
    opened_at: float = 0.0
    rung: str | None = None
    rung_since: float = 0.0
    confidence: float = 0.0
    prior: float = 0.0
    repeat_offender: bool = False
    decision: str | None = None  # "clear" | "terminate", from an administrator
    signals: list[Signal] = field(default_factory=list)
    above_throttle_since: float | None = None
    network_since: float | None = None
    quiet_since: float | None = None
    suspended_at: float | None = None
    #: Set when the 24 h clock resumed a suspension nobody reviewed. The same
    #: evidence then holds the container at throttle; only a decision ends it.
    resumed_at: float | None = None
    throttle: dict = field(default_factory=dict)  # what was changed, to undo it
    last_report: float = 0.0

    def has_evidence_beyond_gpu_profile(self) -> bool:
        return any(sig.family != "S1" for sig in self.signals)

    def has_network(self) -> bool:
        return any(s.family == "S3" and s.strength >= S3_POOL for s in self.signals)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = [asdict(s) for s in self.signals]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        signals = [Signal(**s) for s in d.pop("signals", []) or []]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        ep = cls(**known)
        ep.signals = signals
        return ep


def score(signals: list[Signal], prior: float = 0.0) -> float:
    """Strongest signal per family, families combined, prior on top (never alone)."""
    families: dict[str, float] = {}
    for s in signals:
        families[s.family] = max(families.get(s.family, 0.0), s.strength)
    if not families:
        return 0.0
    c = combine(*families.values())
    return round(min(1.0, c + max(0.0, min(S5_MAX, prior))), 4)


def decide(ep: Episode, now: float) -> str | None:
    """The rung the evidence supports right now; never lower than the current one.

    Terminate needs 0.90 and either a second incident on the account or an
    administrator's confirmation. Suspend needs 0.90, or 0.70 held for half
    an hour with a network signal. Throttle needs 0.70 held for ten minutes
    and at least one signal that is not the GPU profile.
    GPU-only evidence tops out at 0.70 (+0.05 prior): it is flagged for the
    operator and changes nothing for the tenant.
    """
    c = ep.confidence
    held = lambda since, secs: since is not None and now - since >= secs  # noqa: E731
    if ep.decision == "terminate":
        proposed = TERMINATE
    elif c >= TERMINATE_AT and ep.repeat_offender:
        proposed = TERMINATE
    elif c >= SUSPEND_AT:
        proposed = SUSPEND
    elif (
        c >= SUSPEND_WITH_NETWORK_AT
        and ep.has_network()
        and held(ep.above_throttle_since, SUSPEND_WITH_NETWORK_SUSTAIN_S)
    ):
        proposed = SUSPEND
    elif (
        c >= THROTTLE_AT
        and held(ep.above_throttle_since, THROTTLE_SUSTAIN_S)
        and ep.has_evidence_beyond_gpu_profile()
    ):
        # The GPU profile alone is reported, never acted on: it is the one
        # signal an honest workload produces (T8b). Taking CPU, bandwidth and
        # GPU power from a tenant needs a process, a connection or a CPU signal
        # as well.
        proposed = THROTTLE
    elif c >= FLAG_AT:
        proposed = FLAG
    else:
        proposed = None
    if proposed == SUSPEND and ep.resumed_at is not None:
        proposed = THROTTLE
    current = RUNGS.index(ep.rung) if ep.rung in RUNGS else -1
    if proposed is None:
        return ep.rung
    return RUNGS[max(current, RUNGS.index(proposed))]


# ── S1: GPU profile ───────────────────────────────────────────────────


def _s1_sample_matches(g: dict) -> bool:
    util = float(g.get("utilization") or 0)
    mem_util = float(g.get("memory_util") or 0)
    tx = float(g.get("pcie_tx_mb_s") or 0)
    rx = float(g.get("pcie_rx_mb_s") or 0)
    draw = float(g.get("power_draw_w") or 0)
    limit = float(g.get("power_limit_w") or 0)
    power_ok = limit <= 0 or draw >= S1_POWER_FRACTION * limit
    return (
        util > S1_UTIL
        and mem_util >= S1_MEM_UTIL
        and tx < S1_PCIE_MB
        and rx < S1_PCIE_MB
        and power_ok
    )


def _pcie_reading(g: dict) -> float:
    return float(g.get("pcie_tx_mb_s") or 0) + float(g.get("pcie_rx_mb_s") or 0)


def gpu_profile_signal(samples: list[tuple[float, dict]], now: float) -> Signal | None:
    """S1 over the samples (timestamp, telemetry) of one GPU.

    Every sample of the last 15 minutes must match and the VRAM footprint must
    stay within ±2 %; a window with too few samples says nothing.
    """
    # The profile is "busy, and the bus is quiet". On a device whose PCIe
    # counters never read anything (measured 2026-11-13: both fleet GPUs report
    # 0 MB/s idle and under load) the second half is always true, and what is
    # left matches ordinary training and inference on every sample. A counter
    # that has never moved is not evidence of a quiet bus, so without one
    # non-zero reading in the history this signal says nothing.
    if not any(_pcie_reading(g) > 0 for _, g in samples):
        return None
    window = [(t, g) for t, g in samples if now - t <= S1_LONG_WINDOW_S]
    recent = [(t, g) for t, g in window if now - t <= S1_WINDOW_S]
    if len(recent) < S1_MIN_SAMPLES or not all(_s1_sample_matches(g) for _, g in recent):
        return None
    vram = [float(g.get("memory_used_bytes") or 0) for _, g in recent]
    top = max(vram)
    if top <= 0 or (top - min(vram)) / top > S1_VRAM_DRIFT:
        return None
    first_match = recent[0][0]
    strength = S1_SHORT
    # The long window counts only if it is matching throughout.
    if len(window) >= 4 * S1_MIN_SAMPLES and all(_s1_sample_matches(g) for _, g in window):
        strength = S1_LONG
        first_match = window[0][0]
    g = recent[-1][1]
    return Signal(
        "S1",
        strength,
        f"util={g.get('utilization')}% mem={g.get('memory_util')}% "
        f"pcie tx/rx={g.get('pcie_tx_mb_s')}/{g.get('pcie_rx_mb_s')} MB/s "
        f"power={g.get('power_draw_w')}/{g.get('power_limit_w')} W "
        f"for {int((now - first_match) // 60)} min",
        first_match,
    )


# ── S2: process identity ──────────────────────────────────────────────


def load_signatures(path: str = SIGNATURE_FILE) -> dict:
    """Operator-maintained catalogue: ``{"names": [...], "arg_patterns": [...]}``."""
    sig = {"names": [], "arg_patterns": list(DEFAULT_ARG_PATTERNS)}
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return sig
    if isinstance(data, dict):
        names = data.get("names") or []
        pats = data.get("arg_patterns") or []
        sig["names"] = [str(n).lower() for n in names if isinstance(n, str) and n.strip()]
        for p in pats:
            if not isinstance(p, str):
                continue
            try:
                re.compile(p)
            except re.error:
                continue
            sig["arg_patterns"].append(p)
    return sig


def parse_docker_top(output: str) -> list[dict]:
    """Rows of ``docker top <c> -eo pid,rss,comm,args`` as dicts."""
    rows = []
    lines = [ln for ln in (output or "").splitlines() if ln.strip()]
    for ln in lines[1:]:
        parts = ln.split(None, 3)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        rss_kb = int(parts[1]) if parts[1].isdigit() else 0
        rows.append(
            {
                "pid": int(parts[0]),
                "rss": rss_kb * 1024,
                "comm": parts[2],
                "args": parts[3] if len(parts) > 3 else parts[2],
            }
        )
    return rows


def process_signal(rows: list[dict], signatures: dict, now: float) -> Signal | None:
    """S2: a pool URL shape in a process's arguments, or a catalogued name."""
    best: Signal | None = None
    for row in rows:
        args = str(row.get("args") or "")
        comm = str(row.get("comm") or "").lower()
        for pat in signatures.get("arg_patterns") or ():
            if re.search(pat, args, re.IGNORECASE):
                return Signal(
                    "S2", S2_ARGS, f"pid {row.get('pid')} ({comm}): pool URL in arguments", now
                )
        base = os.path.basename(comm)
        if base in (signatures.get("names") or ()) and best is None:
            best = Signal("S2", S2_NAME, f"pid {row.get('pid')}: process name '{base}'", now)
    return best


# ── S3: network ───────────────────────────────────────────────────────


class ConnectionTracker:
    """When each (dst, port) tuple of one container was first and last seen."""

    def __init__(self) -> None:
        self.first_seen: dict[tuple[str, int], float] = {}
        self.last_seen: dict[tuple[str, int], float] = {}

    def observe(self, entries: list[dict], now: float) -> None:
        seen = set()
        for e in entries:
            key = (str(e.get("dst") or ""), int(e.get("dport") or 0))
            if not key[0] or not key[1]:
                continue
            seen.add(key)
            self.first_seen.setdefault(key, now)
            self.last_seen[key] = now
        # A tuple absent for two check intervals is gone; its next sighting is new.
        for key in list(self.first_seen):
            if key not in seen and now - self.last_seen.get(key, now) > 150:
                self.first_seen.pop(key, None)
                self.last_seen.pop(key, None)

    def long_lived(self, now: float, ports: tuple[int, ...]) -> list[tuple[str, int, float]]:
        return [
            (dst, port, now - since)
            for (dst, port), since in self.first_seen.items()
            if port in ports and now - since >= S3_LONG_LIVED_S
        ]


def network_signal(
    tracker: ConnectionTracker,
    snapshot: dict,
    now: float,
    egress: str,
    pool_ports: tuple[int, ...],
) -> Signal | None:
    """S3 from a root-helper snapshot: ``{"connections": [...], "dropped": [...]}``."""
    tracker.observe(snapshot.get("connections") or [], now)
    if egress == "funded":
        lived = tracker.long_lived(now, pool_ports)
        if lived:
            dst, port, age = max(lived, key=lambda x: x[2])
            return Signal(
                "S3",
                S3_POOL,
                f"connection to {dst}:{port} open for {int(age // 60)} min",
                now - age,
            )
        return None
    attempts = [d for d in snapshot.get("dropped") or [] if int(d.get("dport") or 0) in pool_ports]
    if len(attempts) >= 3:
        d = attempts[-1]
        return Signal(
            "S3",
            S3_ATTEMPTED,
            f"{len(attempts)} dropped attempts to pool ports, last {d.get('dst')}:{d.get('dport')}",
            now,
        )
    return None


# ── S4: CPU ───────────────────────────────────────────────────────────


def parse_docker_stats(line: str) -> dict:
    """``{{.CPUPerc}}|{{.PIDs}}`` → ``{"cpu_pct": float, "pids": int}``."""
    parts = (line or "").strip().split("|")
    try:
        cpu = float(parts[0].strip().rstrip("%") or 0)
    except (ValueError, IndexError):
        cpu = 0.0
    try:
        pids = int(parts[1].strip()) if len(parts) > 1 else 0
    except ValueError:
        pids = 0
    return {"cpu_pct": cpu, "pids": pids}


def cpu_signal(
    stats: dict, rows: list[dict], granted_cpus: float, pegged_since: float | None, now: float
) -> tuple[Signal | None, float | None]:
    """S4: all granted vCPUs busy, one process holding most of the memory.

    Returns the signal (after thirty minutes) and the updated ``pegged_since``.
    """
    pegged = granted_cpus > 0 and stats.get("cpu_pct", 0.0) >= S4_CPU_FRACTION * granted_cpus * 100
    total_rss = sum(int(r.get("rss") or 0) for r in rows) or 0
    top = max(rows, key=lambda r: int(r.get("rss") or 0), default=None)
    dominant = (
        top is not None
        and int(top.get("rss") or 0) >= S4_RSS_BYTES
        and int(top.get("rss") or 0) >= S4_DOMINANT_RSS * total_rss
    )
    if not (pegged and dominant):
        return None, None
    since = pegged_since if pegged_since is not None else now
    if now - since < S4_SUSTAIN_S:
        return None, since
    assert top is not None
    return (
        Signal(
            "S4",
            S4,
            f"{stats.get('cpu_pct')}% of {granted_cpus} vCPUs for {int((now - since) // 60)} min, "
            f"pid {top.get('pid')} holds {int(top.get('rss') or 0) // (1024**2)} MiB",
            since,
        ),
        since,
    )


# ── The responder ─────────────────────────────────────────────────────

Runner = Callable[..., "object"]


class Responder:
    """Per-container observation and the four rungs, with every effect behind a seam.

    ``docker(args, timeout)`` runs the docker CLI and returns
    ``(returncode, stdout)``; ``privd(op, **args)`` calls the root helper;
    ``report(episode, evidence)`` tells the control plane and returns its reply
    (``prior``, ``repeat_offender``, ``decision``); ``notify(job_id, text)``
    writes to the instance log; ``terminate(job_id, container_name)`` is the
    existing policy kill.
    """

    def __init__(
        self,
        *,
        docker: Callable[[list[str], int], tuple[int, str]],
        privd: Callable[..., dict],
        report: Callable[[Episode, dict], dict],
        notify: Callable[[str, str], None],
        terminate: Callable[[str, str], None],
        pool_ports: tuple[int, ...],
        state_path: str = STATE_PATH,
        signatures: dict | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.docker = docker
        self.privd = privd
        self.report = report
        self.notify = notify
        self.terminate = terminate
        self.pool_ports = tuple(pool_ports)
        self.state_path = state_path
        self.signatures = signatures if signatures is not None else load_signatures()
        self.clock = clock
        self.episodes: dict[str, Episode] = {}
        self.cleared_until: dict[str, float] = {}
        self.gpu_samples: dict[str, deque] = {}  # gpu uuid or index -> deque[(t, sample)]
        self.trackers: dict[str, ConnectionTracker] = {}
        self.pegged_since: dict[str, float | None] = {}
        self._load()

    # ── persistence ──

    def _load(self) -> None:
        try:
            data = json.loads(Path(self.state_path).read_text())
        except (OSError, ValueError):
            return
        for d in data.get("episodes") or []:
            try:
                ep = Episode.from_dict(dict(d))
            except TypeError:
                continue
            self.episodes[ep.container_name] = ep
        self.cleared_until = {
            str(k): float(v) for k, v in (data.get("cleared_until") or {}).items()
        }

    def _save(self) -> None:
        payload = {
            "episodes": [ep.to_dict() for ep in self.episodes.values()],
            "cleared_until": self.cleared_until,
        }
        try:
            path = Path(self.state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("mining response state not saved: %s", exc)

    # ── observation helpers (host I/O through the docker seam) ──

    def container_facts(self, cname: str) -> dict | None:
        """Id, network, address and CPU grant of one container; None if it is gone."""
        rc, out = self.docker(
            [
                "inspect",
                "-f",
                "{{.Id}}|{{.HostConfig.NanoCpus}}|{{.State.Paused}}|"
                "{{range $k, $v := .NetworkSettings.Networks}}{{$k}}={{$v.IPAddress}} {{end}}",
                cname,
            ],
            10,
        )
        if rc != 0:
            return None
        parts = (out or "").strip().split("|")
        if len(parts) < 4:
            return None
        nets = {}
        for item in parts[3].split():
            if "=" in item:
                k, v = item.split("=", 1)
                nets[k] = v
        try:
            nano = int(parts[1] or 0)
        except ValueError:
            nano = 0
        egress = "open"
        ip = ""
        if "xcelsior-funded" in nets:
            egress, ip = "funded", nets["xcelsior-funded"]
        elif "xcelsior-tenant" in nets:
            egress, ip = "restricted", nets["xcelsior-tenant"]
        return {
            "id": parts[0],
            "cpus": nano / 1e9 if nano else 0.0,
            "paused": parts[2].strip() == "true",
            "egress": egress,
            "ip": ip,
        }

    def processes(self, cname: str) -> list[dict]:
        rc, out = self.docker(["top", cname, "-eo", "pid,rss,comm,args"], 10)
        return parse_docker_top(out) if rc == 0 else []

    def stats(self, cname: str) -> dict:
        rc, out = self.docker(
            ["stats", "--no-stream", "--format", "{{.CPUPerc}}|{{.PIDs}}", cname], 15
        )
        return parse_docker_stats(out) if rc == 0 else {"cpu_pct": 0.0, "pids": 0}

    def record_gpu_samples(self, telemetry: list[dict], now: float) -> None:
        keep = S1_LONG_WINDOW_S + 120
        for g in telemetry or []:
            key = str(g.get("gpu_uuid") or g.get("uuid") or g.get("index"))
            q = self.gpu_samples.setdefault(key, deque())
            q.append((now, g))
            while q and now - q[0][0] > keep:
                q.popleft()

    # ── one pass over one container ──

    def observe(
        self,
        job_id: str,
        cname: str,
        *,
        gpu_keys: list[str],
        shared_gpu: bool,
        gpu_indices: dict[str, int],
    ) -> Episode | None:
        """Collect the signals for one container and move it along the ladder.

        ``gpu_keys`` are the GPUs (uuid or index as recorded by
        :meth:`record_gpu_samples`) attributed to this container this pass;
        ``gpu_indices`` maps them to NVML indices for the power rung;
        ``shared_gpu`` is true when the provider's own GPU service shares the
        device, in which case the GPU power rung is skipped.
        """
        now = self.clock()
        until = self.cleared_until.get(cname)
        if until and now < until:
            return None
        facts = self.container_facts(cname)
        if facts is None:
            self.forget(cname)
            return None

        signals: list[Signal] = []
        for key in gpu_keys:
            s1 = gpu_profile_signal(list(self.gpu_samples.get(key, ())), now)
            if s1:
                signals.append(s1)
                break
        rows = self.processes(cname)
        s2 = process_signal(rows, self.signatures, now)
        if s2:
            signals.append(s2)
        if facts["egress"] in ("funded", "restricted") and facts["ip"]:
            snap = self.privd("tenant_net_snapshot", ip=facts["ip"])
            if snap.get("ok"):
                tracker = self.trackers.setdefault(cname, ConnectionTracker())
                s3 = network_signal(tracker, snap, now, facts["egress"], self.pool_ports)
                if s3:
                    signals.append(s3)
        s4, since = cpu_signal(
            self.stats(cname), rows, facts["cpus"], self.pegged_since.get(cname), now
        )
        self.pegged_since[cname] = since
        if s4:
            signals.append(s4)

        ep = self.episodes.get(cname)
        confidence = score(signals, ep.prior if ep else 0.0)
        if ep is None:
            if confidence < FLAG_AT:
                return None
            ep = Episode(job_id, cname, facts["id"], opened_at=now)
            self.episodes[cname] = ep
        ep.signals = signals
        ep.confidence = confidence
        if confidence >= THROTTLE_AT:
            ep.above_throttle_since = ep.above_throttle_since or now
        else:
            ep.above_throttle_since = None
        ep.quiet_since = None if confidence >= FLAG_AT else (ep.quiet_since or now)

        self._advance(ep, now, facts, gpu_indices, shared_gpu)
        self._save()
        return ep

    def _advance(
        self, ep: Episode, now: float, facts: dict, gpu_indices: dict[str, int], shared_gpu: bool
    ) -> None:
        # The control plane hears every pass of an open episode; its reply
        # carries the account prior, prior incidents and any decision.
        reply = self.report(ep, self.evidence(ep)) or {}
        ep.last_report = now
        if isinstance(reply.get("prior"), (int, float)):
            ep.prior = max(0.0, min(S5_MAX, float(reply["prior"])))
            ep.confidence = score(ep.signals, ep.prior)
        ep.repeat_offender = bool(reply.get("repeat_offender")) or ep.repeat_offender
        decision = reply.get("decision")
        if decision in ("clear", "terminate"):
            ep.decision = decision

        if ep.decision == "clear":
            self._clear(ep, now, "cleared by an administrator")
            self.cleared_until[ep.container_name] = now + CLEARED_GRACE_S
            return
        if ep.rung == SUSPEND and ep.suspended_at and now - ep.suspended_at >= AUTO_RESUME_S:
            if ep.decision != "terminate":
                self._resume(ep, now, "no decision within 24 h; resumed, still throttled")
        if ep.rung != SUSPEND and ep.quiet_since and now - ep.quiet_since >= CLEAR_AFTER_QUIET_S:
            self._clear(ep, now, "signals gone for 30 min")
            return

        target = decide(ep, now)
        if target is None or target == ep.rung:
            return
        # Climb one rung at a time so every rung leaves its own record.
        start = RUNGS.index(ep.rung) + 1 if ep.rung in RUNGS else 0
        for rung in RUNGS[start : RUNGS.index(target) + 1]:
            if rung == FLAG:
                self._flag(ep, now)
            elif rung == THROTTLE:
                self._throttle(ep, now, facts, gpu_indices, shared_gpu)
            elif rung == SUSPEND:
                self._suspend(ep, now)
            elif rung == TERMINATE:
                self._terminate(ep, now)
                return

    # ── rungs ──

    def evidence(self, ep: Episode) -> dict:
        return {
            "container_id": ep.container_id,
            "confidence": ep.confidence,
            "rung": ep.rung,
            "signals": [asdict(s) for s in ep.signals],
            "throttle": ep.throttle,
            "suspended_at": ep.suspended_at,
            "resumed_at": ep.resumed_at,
        }

    def _set_rung(self, ep: Episode, rung: str, now: float) -> None:
        ep.rung = rung
        ep.rung_since = now
        log.warning(
            "mining response: %s job=%s container=%s confidence=%.2f %s",
            rung,
            ep.job_id,
            ep.container_name,
            ep.confidence,
            "; ".join(s.detail for s in ep.signals),
        )

    def _flag(self, ep: Episode, now: float) -> None:
        self._set_rung(ep, FLAG, now)
        self.report(ep, self.evidence(ep))

    def _throttle(
        self, ep: Episode, now: float, facts: dict, gpu_indices: dict[str, int], shared_gpu: bool
    ) -> None:
        changed: dict = dict(ep.throttle)
        cpus = float(facts.get("cpus") or 0)
        if cpus > 0 and "cpus" not in changed:
            half = max(0.5, round(cpus * THROTTLE_CPU_FRACTION, 2))
            rc, _ = self.docker(["update", f"--cpus={half}", ep.container_name], 15)
            if rc == 0:
                changed["cpus"] = cpus
        if not shared_gpu and "gpu_power" not in changed:
            originals: dict[str, int] = {}
            for key, index in gpu_indices.items():
                reply = self.privd("gpu_power_limit", index=int(index), action="min")
                if reply.get("ok") and reply.get("previous_w"):
                    originals[str(index)] = int(reply["previous_w"])
            if originals:
                changed["gpu_power"] = originals
        if facts.get("egress") == "funded" and facts.get("ip") and "net" not in changed:
            reply = self.privd(
                "tenant_rate_limit", ip=facts["ip"], mbps=THROTTLE_NET_MBPS, action="apply"
            )
            if reply.get("ok"):
                changed["net"] = facts["ip"]
        ep.throttle = changed
        self._set_rung(ep, THROTTLE, now)
        self.notify(
            ep.job_id,
            "[xcelsior] This instance is throttled: its activity matches resource abuse "
            "(cryptocurrency mining). CPU, GPU power and upload are limited until an "
            "administrator reviews it. Contact support if this is a mistake.",
        )
        self.report(ep, self.evidence(ep))

    def _undo_throttle(self, ep: Episode) -> None:
        t = ep.throttle
        if t.get("cpus"):
            self.docker(["update", f"--cpus={t['cpus']}", ep.container_name], 15)
        for index, watts in (t.get("gpu_power") or {}).items():
            self.privd("gpu_power_limit", index=int(index), action="restore", watts=int(watts))
        if t.get("net"):
            self.privd("tenant_rate_limit", ip=t["net"], action="clear")
        ep.throttle = {}

    def _suspend(self, ep: Episode, now: float) -> None:
        rc, _ = self.docker(["pause", ep.container_name], 15)
        if rc != 0:
            log.error("mining response: pause failed for %s", ep.container_name)
            return
        ep.suspended_at = now
        self._set_rung(ep, SUSPEND, now)
        self.notify(
            ep.job_id,
            "[xcelsior] This instance is suspended pending review: its activity matches "
            "resource abuse (cryptocurrency mining). State is kept; billing is paused. "
            "It resumes in 24 hours if no decision is made. Contact support to appeal.",
        )
        self.report(ep, self.evidence(ep))

    def _resume(self, ep: Episode, now: float, why: str) -> None:
        rc, _ = self.docker(["unpause", ep.container_name], 15)
        if rc != 0:
            return
        ep.suspended_at = None
        ep.resumed_at = now
        ep.decision = None
        self._set_rung(ep, THROTTLE, now)
        self.notify(ep.job_id, f"[xcelsior] Instance resumed: {why}.")
        self.report(ep, self.evidence(ep))

    def _terminate(self, ep: Episode, now: float) -> None:
        self._set_rung(ep, TERMINATE, now)
        self.report(ep, self.evidence(ep))
        self.terminate(ep.job_id, ep.container_name)
        self.forget(ep.container_name)

    def _clear(self, ep: Episode, now: float, why: str) -> None:
        if ep.suspended_at:
            self.docker(["unpause", ep.container_name], 15)
            ep.suspended_at = None
        self._undo_throttle(ep)
        ep.rung = None
        ep.decision = None
        ep.signals = []
        ep.confidence = 0.0
        log.info("mining response: episode closed for %s (%s)", ep.container_name, why)
        self.notify(ep.job_id, f"[xcelsior] Instance limits lifted: {why}.")
        self.report(ep, {**self.evidence(ep), "closed": why})
        self.forget(ep.container_name)

    def forget(self, cname: str) -> None:
        self.episodes.pop(cname, None)
        self.trackers.pop(cname, None)
        self.pegged_since.pop(cname, None)
        self._save()

    def restore_after_restart(self, live: set[str]) -> None:
        """A worker restart keeps every episode whose container still exists."""
        for cname in list(self.episodes):
            if cname not in live:
                self.episodes.pop(cname, None)
        self._save()
