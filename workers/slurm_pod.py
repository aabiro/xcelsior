"""Pod listing spec backed by a live Slurm cluster — reference implementation.

A *pod* is a fabric-connected group of nodes rented as one unit. A pod
listing is only worth a premium if its specs are true, so every spec field
here carries where its value came from:

  observed  measured on the running cluster — from the scheduler (sinfo
            GRES/CPU/memory) or from probe jobs the adapter submitted to
            each node (NVML via nvidia-smi, NIC link speed via sysfs);
  declared  stated by the facility partner and *not* measurable from a job
            (switch speed, switch count, facility name).

``verify_pod_spec`` cross-checks the two: a declared value may never exceed
what was observed (an 800G claim on a 1G link fails), and the scheduler's
GPU count must equal what the probe jobs found. Only a spec whose checks
all pass is ``verified``.

This module is pure data + parsing. The probes are submitted by the caller
(the validation kit's POD-1 case) through slurm_adapter, so the evidence
for each observed field is a Slurm job id anyone can look up in sacct.

Not wired into the marketplace yet; hpc/IMPLEMENTATION_PLAN.md phase 4
covers the pod_listings table, API and MCP tools built on this shape.
"""

from __future__ import annotations

import re
import time

POD_SPEC_VERSION = "xcelsior.pod/v0"

# sinfo %G: "gpu:nvidia_geforce_rtx_3060:1(S:0)", "gpu:4", "gpu:h100:8(S:0-1)",
# several comma-separated, or "(null)".
_GRES_RE = re.compile(r"gpu(?::(?P<type>[A-Za-z0-9_.-]+))?:(?P<count>\d+)")
_NVLINK_RE = re.compile(r"Link\s+\d+:\s*([\d.]+)\s*GB/s", re.I)

OBSERVED = "observed"
DECLARED = "declared"


def field(value, source: str, method: str, evidence: str | None = None) -> dict:
    """One spec value plus its provenance."""
    return {"value": value, "source": source, "method": method, "evidence": evidence}


# ── Parsers ──────────────────────────────────────────────────────────


def parse_gres(gres: str) -> list[dict]:
    """sinfo %G → [{"type": str|None, "count": int}] for GPUs only."""
    out = []
    for item in re.split(r",(?![^(]*\))", gres or ""):
        m = _GRES_RE.match(item.strip())
        if m:
            out.append({"type": m.group("type"), "count": int(m.group("count"))})
    return out


def parse_gpu_probe(stdout: str) -> dict:
    """Parse the GPU probe job's output.

    Expected stdout (see GPU_PROBE_COMMAND):
        <index>, <name>, <memory MiB>, <uuid>      one line per GPU
        ---nvlink
        <nvidia-smi nvlink --status output>
    """
    gpus = []
    nvlink_links = 0
    section = "gpus"
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "---nvlink":
            section = "nvlink"
            continue
        if section == "gpus":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "model": parts[1],
                        "memory_mib": (
                            int(float(parts[2])) if parts[2].replace(".", "").isdigit() else 0
                        ),
                        "uuid": parts[3] if len(parts) > 3 else None,
                    }
                )
        elif _NVLINK_RE.search(line):
            nvlink_links += 1
    return {"gpus": gpus, "nvlink_active_links": nvlink_links}


def parse_nic_probe(stdout: str) -> dict:
    """Parse ``iface=<name>`` / ``speed=<Mb/s|unknown>`` lines."""
    vals = {}
    for line in (stdout or "").splitlines():
        if "=" in line:
            k, v = line.strip().split("=", 1)
            vals[k] = v
    speed = vals.get("speed", "")
    mbps = int(speed) if speed.lstrip("-").isdigit() and int(speed) > 0 else None
    return {"interface": vals.get("iface") or None, "speed_mbps": mbps}


# Probe payloads the validation kit submits through slurm_adapter.
GPU_PROBE_COMMAND = (
    "nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader,nounits; "
    "echo ---nvlink; nvidia-smi nvlink --status 2>/dev/null || true"
)
NIC_PROBE_COMMAND = (
    "dev=$(ip -o route get 1.1.1.1 2>/dev/null | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p'); "
    'echo "iface=${dev}"; '
    'echo "speed=$(cat /sys/class/net/${dev}/speed 2>/dev/null || echo unknown)"'
)


# ── Builder ──────────────────────────────────────────────────────────


def build_pod_spec(
    *,
    cluster_id: str,
    partition: str,
    inventory: dict,
    gpu_probes: dict,
    nic_probes: dict,
    declared: dict,
    scheduler_version: str | None = None,
) -> dict:
    """Assemble a pod spec from scheduler inventory, probe results and declarations.

    Args:
        inventory: slurm_adapter.get_cluster_inventory() result.
        gpu_probes: {node: {"probe": parse_gpu_probe(...), "slurm_job_id": "..."}}
        nic_probes: {node: {"probe": parse_nic_probe(...), "slurm_job_id": "..."}}
        declared: facility statements, e.g.
            {"listing_title": "...", "facility": "...",
             "interconnect": {"type": "ethernet", "overlay": "wireguard",
                              "nic_speed_gbps": 1, "switch_speed_gbps": None,
                              "switch_count": 0}}
    """
    nodes_in = inventory.get("nodes") or []
    sinfo_ev = f"sinfo -N -p {partition}"
    nodes = []
    sched_gpu_total = 0
    probe_gpu_total = 0
    gpu_models: set[str] = set()
    gpu_mem: set[int] = set()
    nvlink_domain = 0
    nic_speeds = []
    gpu_probe_jobs = []
    nic_probe_jobs = []

    for n in nodes_in:
        name = n["name"]
        gres = parse_gres(n.get("gres", ""))
        sched_gpus = sum(g["count"] for g in gres)
        sched_gpu_total += sched_gpus
        gp = gpu_probes.get(name) or {}
        probe = gp.get("probe") or {"gpus": [], "nvlink_active_links": 0}
        probe_gpu_total += len(probe["gpus"])
        if gp.get("slurm_job_id"):
            gpu_probe_jobs.append(str(gp["slurm_job_id"]))
        for g in probe["gpus"]:
            gpu_models.add(g["model"])
            gpu_mem.add(g["memory_mib"])
        if probe["gpus"]:
            # Approximation: any active NVLink on the node joins all of its
            # GPUs into one domain; none → each GPU is its own domain of 1.
            # Multi-node NVLink (NVL72-class) needs NVSwitch/IMEX data and
            # is out of scope for v0.
            domain = len(probe["gpus"]) if probe["nvlink_active_links"] > 0 else 1
            nvlink_domain = max(nvlink_domain, domain)
        np_ = nic_probes.get(name) or {}
        nic = np_.get("probe") or {"interface": None, "speed_mbps": None}
        if nic["speed_mbps"]:
            nic_speeds.append(nic["speed_mbps"])
        if np_.get("slurm_job_id"):
            nic_probe_jobs.append(str(np_["slurm_job_id"]))
        nodes.append(
            {
                "name": name,
                "state": n.get("state"),
                "cpus": field(n.get("cpus"), OBSERVED, "slurm:sinfo %c", sinfo_ev),
                "memory_mb": field(n.get("memory_mb"), OBSERVED, "slurm:sinfo %m", sinfo_ev),
                "gres": n.get("gres"),
                "gpus_scheduler": field(sched_gpus, OBSERVED, "slurm:sinfo %G", sinfo_ev),
                "gpus_probe": field(
                    probe["gpus"],
                    OBSERVED,
                    "nvml:nvidia-smi in allocated job",
                    _job_ev(gp.get("slurm_job_id")),
                ),
                "nic": field(
                    {"interface": nic["interface"], "speed_gbps": _gbps(nic["speed_mbps"])},
                    OBSERVED,
                    "sysfs:/sys/class/net/<dev>/speed in a job on the node",
                    _job_ev(np_.get("slurm_job_id")),
                ),
            }
        )

    ic = (declared or {}).get("interconnect") or {}
    gpu_ev = f"slurm jobs {', '.join(gpu_probe_jobs)}" if gpu_probe_jobs else None
    nic_ev = f"slurm jobs {', '.join(nic_probe_jobs)}" if nic_probe_jobs else None
    gpu_model = sorted(gpu_models)[0] if len(gpu_models) == 1 else (sorted(gpu_models) or None)
    spec = {
        "spec_version": POD_SPEC_VERSION,
        "pod_id": f"pod-{cluster_id}-{partition}",
        "listing_title": (declared or {}).get("listing_title"),
        "facility": field((declared or {}).get("facility"), DECLARED, "facility statement"),
        "cluster": {
            "cluster_id": cluster_id,
            "scheduler": "slurm",
            "scheduler_version": scheduler_version,
            "partition": partition,
        },
        "node_count": field(len(nodes), OBSERVED, "slurm:sinfo -N", sinfo_ev),
        "gpu": {
            "count": field(sched_gpu_total, OBSERVED, "slurm:GRES total", sinfo_ev),
            "count_probe": field(probe_gpu_total, OBSERVED, "nvml:probe jobs", gpu_ev),
            "model": field(gpu_model, OBSERVED, "nvml:probe jobs", gpu_ev),
            "memory_gb": field(
                round(min(gpu_mem) / 1024, 1) if gpu_mem else None,
                OBSERVED,
                "nvml:probe jobs",
                gpu_ev,
            ),
            "nvlink_domain_size": field(
                nvlink_domain or None, OBSERVED, "nvml:nvidia-smi nvlink --status", gpu_ev
            ),
        },
        "interconnect": {
            "type": field(ic.get("type"), DECLARED, "facility statement"),
            "overlay": field(ic.get("overlay"), DECLARED, "facility statement"),
            "nic_speed_gbps": field(
                _gbps(min(nic_speeds)) if nic_speeds else None,
                OBSERVED,
                "sysfs link speed, minimum across nodes",
                nic_ev,
            ),
            "nic_speed_gbps_declared": field(
                ic.get("nic_speed_gbps"), DECLARED, "facility statement"
            ),
            "switch_speed_gbps": field(ic.get("switch_speed_gbps"), DECLARED, "facility statement"),
            "switch_count": field(ic.get("switch_count"), DECLARED, "facility statement"),
        },
        "nodes": nodes,
    }
    spec["verification"] = verify_pod_spec(spec)
    return spec


def _gbps(mbps):
    return round(mbps / 1000, 3) if mbps else None


def _job_ev(job_id):
    return f"slurm job {job_id}" if job_id else None


# ── Verification ─────────────────────────────────────────────────────

_HEALTHY_STATES = {"idle", "mixed", "allocated", "completing"}


def verify_pod_spec(spec: dict) -> dict:
    """Cross-check observed against declared values; returns the verdict block."""
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    nodes = spec.get("nodes") or []
    check("has_nodes", len(nodes) > 0, f"{len(nodes)} node(s) in partition")
    bad = [
        n["name"]
        for n in nodes
        if str(n.get("state", "")).rstrip("*~#").lower() not in _HEALTHY_STATES
    ]
    check("nodes_schedulable", not bad, "all nodes up" if not bad else f"not schedulable: {bad}")

    sched = spec["gpu"]["count"]["value"]
    probed = spec["gpu"]["count_probe"]["value"]
    check(
        "gpu_count_matches_probes",
        sched == probed,
        f"scheduler GRES={sched}, NVML probes={probed}",
    )
    for n in nodes:
        s = n["gpus_scheduler"]["value"]
        p = len(n["gpus_probe"]["value"])
        if s != p:
            check(f"gpu_count_{n['name']}", False, f"{n['name']}: GRES={s}, NVML={p}")
        if s and not n["gpus_probe"]["evidence"]:
            check(f"gpu_probe_ran_{n['name']}", False, f"{n['name']}: no GPU probe job")

    model = spec["gpu"]["model"]["value"]
    check(
        "gpu_model_homogeneous",
        sched == 0 or isinstance(model, str),
        f"model={model!r}",
    )

    ic = spec["interconnect"]
    observed = ic["nic_speed_gbps"]["value"]
    declared = ic["nic_speed_gbps_declared"]["value"]
    if declared is None:
        check("nic_speed_declared", False, "facility did not declare a NIC speed")
    elif observed is None:
        check(
            "nic_speed_observed", False, "no node reported a link speed; cannot verify declaration"
        )
    else:
        check(
            "nic_speed_declared_not_above_observed",
            float(declared) <= float(observed) + 1e-9,
            f"declared {declared} Gb/s vs observed min {observed} Gb/s",
        )
    for n in nodes:
        if n["nic"]["value"]["speed_gbps"] is None:
            check(f"nic_observed_{n['name']}", False, f"{n['name']}: link speed not observed")

    passed = all(c["ok"] for c in checks)
    return {
        "status": "verified" if passed else "failed",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
    }


# ── Shape validation ─────────────────────────────────────────────────

_REQUIRED_FIELDS = (
    ("node_count",),
    ("gpu", "count"),
    ("gpu", "model"),
    ("gpu", "nvlink_domain_size"),
    ("interconnect", "type"),
    ("interconnect", "nic_speed_gbps"),
    ("interconnect", "switch_speed_gbps"),
)


def validate_pod_spec(spec: dict) -> list[str]:
    """Structural check of a spec document; returns a list of problems."""
    errors = []
    if spec.get("spec_version") != POD_SPEC_VERSION:
        errors.append(f"spec_version must be {POD_SPEC_VERSION}")
    for key in ("pod_id", "cluster", "nodes", "verification"):
        if key not in spec:
            errors.append(f"missing {key}")
    for path in _REQUIRED_FIELDS:
        cur = spec
        for p in path:
            cur = cur.get(p) if isinstance(cur, dict) else None
        name = ".".join(path)
        if not isinstance(cur, dict) or not {"value", "source", "method"} <= set(cur):
            errors.append(f"{name} must be a provenance field")
        elif cur["source"] not in (OBSERVED, DECLARED):
            errors.append(f"{name}.source must be observed|declared")
    return errors
