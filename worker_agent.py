#!/usr/bin/env python3
# Xcelsior Worker Agent v2.1.0
# Pull-based GPU worker agent for distributed GPU scheduling.
#
# Architecture change from v1.0.0 (push-based):
#   - Agents POLL the scheduler for work instead of scheduler SSHing in
#   - Works behind CGNAT / firewalls (no inbound ports needed)
#   - Integrates Tailscale/Headscale for optional mesh networking
#   - Container lifecycle management with defense-in-depth security
#   - Mining detection heuristics
#   - Compute score benchmarking (XCU)
#   - Graceful preemption handling

# Auto-load .env file
from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os
import re
import hashlib
import secrets
import signal
import subprocess
import shlex
import sys
import threading
import time
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Install with: pip install requests")
    sys.exit(1)

try:  # P2.4 — bandwidth / disk IO metrics. Optional: if psutil is missing, we
    # silently omit the fields rather than failing the agent.
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

from security import (
    admit_node,
    build_secure_docker_args,
    build_egress_iptables_rules,
    check_mining_heuristic,
    get_gpu_telemetry,
    get_local_versions,
    install_gvisor,
    recommend_runtime,
)

# ── NVML Telemetry (REPORT_FEATURE_1.md §Lightweight Monitoring) ─────
# Use pynvml bindings when available; security.py's get_gpu_telemetry()
# already delegates to nvml_telemetry when NVML is initialized.

try:
    from nvml_telemetry import (
        nvml_init,
        nvml_shutdown,
        is_nvml_available,
        collect_all_gpus,
        get_gpu_info_nvml,
        build_verification_report,
    )

    _nvml_imported = True
except ImportError:
    _nvml_imported = False

    def nvml_init():
        return False

    def nvml_shutdown():
        pass

    def is_nvml_available():
        return False

    def collect_all_gpus():
        return []

    def get_gpu_info_nvml():
        return None

    def build_verification_report(gpu_index=0):
        return None


# ── Configuration ─────────────────────────────────────────────────────

# Required
HOST_ID = os.environ.get("XCELSIOR_HOST_ID")
SCHEDULER_URL = os.environ.get("XCELSIOR_SCHEDULER_URL")

# Auth
API_TOKEN = os.environ.get("XCELSIOR_API_TOKEN", "")
OAUTH_CLIENT_ID = os.environ.get("XCELSIOR_OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("XCELSIOR_OAUTH_CLIENT_SECRET", "")
OAUTH_SCOPE = os.environ.get("XCELSIOR_OAUTH_SCOPE", "api").strip()
OAUTH_TOKEN_URL = os.environ.get("XCELSIOR_OAUTH_TOKEN_URL", "")
OAUTH_TOKEN_TIMEOUT_SEC = int(os.environ.get("XCELSIOR_OAUTH_TOKEN_TIMEOUT_SEC", "10"))
OAUTH_TOKEN_REFRESH_SKEW_SEC = int(os.environ.get("XCELSIOR_OAUTH_TOKEN_REFRESH_SKEW_SEC", "60"))

# Optional tuning
COST_PER_HOUR = float(os.environ.get("XCELSIOR_COST_PER_HOUR", "0.50"))
HEARTBEAT_INTERVAL = int(os.environ.get("XCELSIOR_HEARTBEAT_INTERVAL", "10"))
POLL_INTERVAL = int(os.environ.get("XCELSIOR_POLL_INTERVAL", "5"))
MINING_CHECK_INTERVAL = int(os.environ.get("XCELSIOR_MINING_CHECK_INTERVAL", "60"))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("XCELSIOR_MAX_FAILURES", "30"))
CONTAINER_START_TIMEOUT = int(os.environ.get("XCELSIOR_CONTAINER_START_TIMEOUT", "180"))

# Tailscale / Headscale
TAILSCALE_ENABLED = os.environ.get("XCELSIOR_TAILSCALE_ENABLED", "").lower() in ("1", "true", "yes")
TAILSCALE_AUTHKEY = os.environ.get("XCELSIOR_TAILSCALE_AUTHKEY", "")
HEADSCALE_URL = os.environ.get("XCELSIOR_HEADSCALE_URL", "")

# gVisor preference
PREFER_GVISOR = os.environ.get("XCELSIOR_PREFER_GVISOR", "true").lower() in ("1", "true", "yes")

VERSION = "2.1.0"


def _self_sha256() -> str:
    """Return the sha256 of this worker_agent.py file (best-effort).

    Used for:
      - Reporting agent_sha256 in /host heartbeats so the control plane
        knows which bytes are actually running on this host.
      - Deciding whether an ``upgrade_agent`` directive is a no-op.
    Returns empty string on any error (the file must exist; this is
    almost always successful).
    """
    try:
        import hashlib

        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return ""


# Reserved env-var keys the platform always injects into every container.
# User-supplied `environment` values for these keys are always overridden.
PLATFORM_ENV_KEYS = (
    "XCELSIOR_JOB_ID",
    "XCELSIOR_HOST_ID",
    "XCELSIOR_OWNER",
    "XCELSIOR_API_URL",
    "XCELSIOR_GPU_MODEL",
    "XCELSIOR_GPU_VRAM_GB",
    "XCELSIOR_INSTANCE_NAME",
    "XCELSIOR_PUBLIC_SSH_HOST",
    "XCELSIOR_PUBLIC_SSH_PORT",
    "XCELSIOR_TOOLS_PATH",
    "XCELSIOR_EXPOSED_PORTS",
    "XCELSIOR_AUTO_LAUNCH",
)


def _compute_public_ssh_port(job_id: str) -> int:
    """Deterministic public SSH port for a job (mirrors SSH gateway mapping)."""
    try:
        return 10000 + (int(str(job_id)[:4], 16) % 55000)
    except (ValueError, TypeError):
        return 0


def build_platform_env(job: dict, gpu_info: dict | None = None) -> dict:
    """Return the reserved platform env-var dict for ``job``.

    These keys are injected *after* user-supplied env so users cannot override
    them. Always returns strings (docker env requires string values).
    """
    job = job or {}
    job_id = str(job.get("job_id", "") or "")
    gi = gpu_info or {}
    api_url = os.environ.get("XCELSIOR_API_URL", "https://xcelsior.ca")
    # P2.1: surface tools path and user-requested exposed ports / auto-launch
    # services so containers can discover them without a second API round-trip.
    exposed_ports = job.get("exposed_ports") or []
    if isinstance(exposed_ports, (list, tuple)):
        exposed_ports_s = ",".join(str(int(p)) for p in exposed_ports if str(p).strip())
    else:
        exposed_ports_s = ""
    auto_launch = job.get("auto_launch") or []
    if isinstance(auto_launch, (list, tuple)):
        auto_launch_s = ",".join(
            str(s).strip().lower() for s in auto_launch if str(s).strip()
        )
    else:
        auto_launch_s = ""
    return {
        "XCELSIOR_JOB_ID": job_id,
        "XCELSIOR_HOST_ID": str(HOST_ID or ""),
        "XCELSIOR_OWNER": str(job.get("owner", "") or ""),
        "XCELSIOR_API_URL": str(api_url),
        "XCELSIOR_GPU_MODEL": str(gi.get("gpu_model", "") or ""),
        "XCELSIOR_GPU_VRAM_GB": str(gi.get("total_vram_gb", "") or ""),
        "XCELSIOR_INSTANCE_NAME": str(job.get("name", "") or job_id),
        "XCELSIOR_PUBLIC_SSH_HOST": "connect.xcelsior.ca",
        "XCELSIOR_PUBLIC_SSH_PORT": str(_compute_public_ssh_port(job_id)),
        "XCELSIOR_TOOLS_PATH": "/opt/xcelsior/bin",
        "XCELSIOR_EXPOSED_PORTS": exposed_ports_s,
        "XCELSIOR_AUTO_LAUNCH": auto_launch_s,
    }


# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("xcelsior-worker")

# ── Metrics ──────────────────────────────────────────────────────────
from prometheus_client import Counter as _PromCounter

_motd_reinjection_total = _PromCounter(
    "xcelsior_worker_motd_reinjection_total",
    "Shell/MOTD re-injection attempts by outcome",
    ["result"],
)

# P3/C7 — unknown/rejected commands at the drain-side allowlist gate.
_agent_commands_rejected_total = _PromCounter(
    "xcelsior_worker_agent_commands_rejected_total",
    "Agent commands rejected by worker-side allowlist (defence-in-depth)",
    ["command"],
)

# ── Shutdown Coordination ────────────────────────────────────────────

_shutdown = threading.Event()
_active_containers = {}  # job_id -> container_name
_adopted_containers = set()  # job_ids of containers adopted from scheduler (don't stop on shutdown)
_active_lock = threading.Lock()
_oauth_lock = threading.Lock()
_oauth_access_token = ""
_oauth_access_token_expires_at = 0.0


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT — graceful shutdown."""
    sig_name = signal.Signals(signum).name
    log.info("Received %s — initiating graceful shutdown", sig_name)
    _shutdown.set()


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── GPU Queries ──────────────────────────────────────────────────────


# Explicit 1:1 map from raw ``nvidia-smi --query-gpu=name`` output to the
# canonical short title used everywhere else in the platform (matches
# db.py _GPU_PRICING_BASE and frontend/src/lib/gpu-models.ts). This map
# is the ONLY place raw vendor names are translated — no fuzzy matching,
# no regex normalizer. Unknown names cause the host to quarantine.
_NVIDIA_SMI_NAME_MAP: dict[str, str] = {
    # ── NVIDIA Data Center ──
    "NVIDIA H200": "H200",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H100 PCIe": "H100",
    "NVIDIA H100": "H100",
    "NVIDIA H100 NVL": "H100 NVL",
    "NVIDIA A100-SXM4-80GB": "A100",
    "NVIDIA A100-SXM4-40GB": "A100",
    "NVIDIA A100-PCIE-40GB": "A100",
    "NVIDIA A100 80GB PCIe": "A100",
    "NVIDIA A40": "A40",
    "NVIDIA A30": "A30",
    "NVIDIA A10": "A10",
    "NVIDIA A16": "A16",
    "NVIDIA L40S": "L40S",
    "NVIDIA L40": "L40",
    "NVIDIA L4": "L4",
    "Tesla T4": "T4",
    "Tesla V100-SXM2-32GB": "V100",
    "Tesla V100-PCIE-32GB": "V100",
    "Tesla V100-SXM2-16GB": "V100",
    "Tesla V100-PCIE-16GB": "V100",
    # ── NVIDIA RTX 50 Series ──
    "NVIDIA GeForce RTX 5090": "RTX 5090",
    "NVIDIA GeForce RTX 5080": "RTX 5080",
    "NVIDIA GeForce RTX 5070 Ti": "RTX 5070 Ti",
    "NVIDIA GeForce RTX 5070": "RTX 5070",
    "NVIDIA GeForce RTX 5060 Ti": "RTX 5060 Ti",
    "NVIDIA GeForce RTX 5060": "RTX 5060",
    # ── NVIDIA RTX 40 Series ──
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA GeForce RTX 4080 SUPER": "RTX 4080 Super",
    "NVIDIA GeForce RTX 4080": "RTX 4080",
    "NVIDIA GeForce RTX 4070 Ti SUPER": "RTX 4070 Ti Super",
    "NVIDIA GeForce RTX 4070 Ti": "RTX 4070 Ti",
    "NVIDIA GeForce RTX 4070 SUPER": "RTX 4070 Super",
    "NVIDIA GeForce RTX 4070": "RTX 4070",
    "NVIDIA GeForce RTX 4060 Ti": "RTX 4060 Ti",
    "NVIDIA GeForce RTX 4060": "RTX 4060",
    # ── NVIDIA RTX 30 Series ──
    "NVIDIA GeForce RTX 3090 Ti": "RTX 3090 Ti",
    "NVIDIA GeForce RTX 3090": "RTX 3090",
    "NVIDIA GeForce RTX 3080 Ti": "RTX 3080 Ti",
    "NVIDIA GeForce RTX 3080": "RTX 3080",
    "NVIDIA GeForce RTX 3070 Ti": "RTX 3070 Ti",
    "NVIDIA GeForce RTX 3070": "RTX 3070",
    "NVIDIA GeForce RTX 3060 Ti": "RTX 3060 Ti",
    "NVIDIA GeForce RTX 3060": "RTX 3060",
    # ── NVIDIA RTX 20 Series ──
    "NVIDIA GeForce RTX 2080 Ti": "RTX 2080 Ti",
    "NVIDIA GeForce RTX 2080 SUPER": "RTX 2080 Super",
    "NVIDIA GeForce RTX 2080": "RTX 2080",
    "NVIDIA GeForce RTX 2070 SUPER": "RTX 2070 Super",
    "NVIDIA GeForce RTX 2070": "RTX 2070",
    "NVIDIA GeForce RTX 2060 SUPER": "RTX 2060 Super",
    "NVIDIA GeForce RTX 2060": "RTX 2060",
    # ── NVIDIA Workstation ──
    "NVIDIA RTX 6000 Ada Generation": "RTX 6000 Ada",
    "NVIDIA RTX 5000 Ada Generation": "RTX 5000 Ada",
    "NVIDIA RTX 4000 Ada Generation": "RTX 4000 Ada",
    "NVIDIA RTX A6000": "RTX A6000",
    "NVIDIA RTX A5000": "RTX A5000",
    "NVIDIA RTX A4000": "RTX A4000",
}


def _canonicalize_gpu_model(raw_name: str) -> str:
    """Translate a raw nvidia-smi GPU name to its canonical short title.

    Returns the canonical short title (e.g. ``RTX 4090``) if known, or
    the raw name as a last-resort fallback so the scheduler can still
    see the host. Unknown names should be added to
    ``_NVIDIA_SMI_NAME_MAP`` and the corresponding pricing row added to
    ``db.py _GPU_PRICING_BASE``.
    """
    name = (raw_name or "").strip()
    if not name:
        return ""
    mapped = _NVIDIA_SMI_NAME_MAP.get(name)
    if mapped:
        return mapped
    log.warning(
        "Unknown nvidia-smi GPU name %r — add to _NVIDIA_SMI_NAME_MAP",
        name,
    )
    return name


def get_gpu_info():
    """Query GPU name, total VRAM, free VRAM.

    Prefers NVML (pynvml) when available per REPORT_FEATURE_1.md:
    "avoid calling the nvidia-smi binary... use Python bindings to
    interact with libnvidia-ml.so directly."

    Falls back to nvidia-smi subprocess if NVML is not initialized.
    Returns dict with gpu_model, total_vram_gb, free_vram_gb.
    Raises RuntimeError on failure.
    """
    # Try NVML first
    if is_nvml_available():
        info = get_gpu_info_nvml()
        if info:
            info["gpu_model"] = _canonicalize_gpu_model(info.get("gpu_model", ""))
            return info

    # Fallback: nvidia-smi subprocess
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            raise RuntimeError("No GPU found in nvidia-smi output")

        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi format: {lines[0]}")

        return {
            "gpu_model": _canonicalize_gpu_model(parts[0]),
            "total_vram_gb": round(float(parts[1]) / 1024, 2),
            "free_vram_gb": round(float(parts[2]) / 1024, 2),
        }
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed: {e.stderr}")
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Failed to parse nvidia-smi output: {e}")


def get_host_ip():
    """Best-effort detection of the primary host IP."""
    # Allow explicit override via env var (useful when Tailscale and Headscale
    # assign different IPs and the scheduler sees a different one)
    override = os.environ.get("XCELSIOR_HOST_IP")
    if override:
        return override

    # If Headscale mesh is active, prefer the mesh IP
    if TAILSCALE_ENABLED:
        try:
            r = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # hostname -I
    try:
        r = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            ips = r.stdout.strip().split()
            if ips:
                return ips[0]
    except Exception as e:
        log.debug("hostname -I failed: %s", e)

    # ip route fallback
    try:
        r = subprocess.run(["ip", "route", "get", "1"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            match = re.search(r"src (\S+)", r.stdout)
            if match:
                return match.group(1)
    except Exception as e:
        log.debug("ip route fallback failed: %s", e)

    return "unknown"


# ── Compute Score Benchmark ──────────────────────────────────────────


def run_compute_benchmark():
    """Run comprehensive benchmark suite for all 6 verification checks.

    Collects: TFLOPS (FP16 matmul), CUDA/driver versions, compute capability,
    PCIe bandwidth, sustained thermal readings, and GPU identity data.
    Returns a full report dict that verification.py can consume, or None.
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import time, json, subprocess, re
report = {}

try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no_cuda"}))
        raise SystemExit

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    # ── GPU Identity ──
    report["gpu_model"] = props.name
    report["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
    report["compute_capability"] = f"{props.major}.{props.minor}"

    # ── CUDA / Driver Versions ──
    report["cuda_version"] = torch.version.cuda or ""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        report["driver_version"] = smi.stdout.strip().split("\\n")[0].strip() if smi.returncode == 0 else ""
    except Exception as e:
        log.debug("nvidia-smi driver query failed: %s", e)
        report["driver_version"] = ""

    # ── FP16 Matmul Benchmark (TFLOPS) ──
    a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()
    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    flops_per_iter = 2 * 4096**3
    report["tflops"] = round((flops_per_iter * iters) / elapsed / 1e12, 2)
    report["elapsed_s"] = round(elapsed, 3)
    report["iters"] = iters

    # ── PCIe Bandwidth (Host→Device→Host round-trip) ──
    try:
        size_mb = 256
        data_h = torch.randn(size_mb * 1024 * 256, dtype=torch.float32)  # 256 MB
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        data_d = data_h.to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = data_d.to("cpu")
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        h2d_gbps = round(size_mb / 1024 / (t1 - t0), 2)
        d2h_gbps = round(size_mb / 1024 / (t2 - t1), 2)
        report["pcie_bandwidth_gbps"] = round((h2d_gbps + d2h_gbps) / 2, 2)
        report["pcie_h2d_gbps"] = h2d_gbps
        report["pcie_d2h_gbps"] = d2h_gbps
        del data_h, data_d
    except Exception as e:
        log.debug("PCIe bandwidth test failed: %s", e)
        report["pcie_bandwidth_gbps"] = 0
        report["pcie_error"] = str(e)

    # ── Thermal Stability (sustained load for 15s, sample temp) ──
    try:
        temps = []
        stress_start = time.perf_counter()
        while time.perf_counter() - stress_start < 15:
            torch.mm(a, b)
            torch.cuda.synchronize()
            try:
                tsmi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu",
                     "--format=csv,noheader,nounits", "--id=0"],
                    capture_output=True, text=True, timeout=5,
                )
                if tsmi.returncode == 0:
                    temps.append(float(tsmi.stdout.strip()))
            except Exception as e:
                log.debug("nvidia-smi temp query failed: %s", e)
            time.sleep(0.5)
        if temps:
            report["gpu_temp_celsius"] = max(temps)
            report["gpu_temp_avg_celsius"] = round(sum(temps) / len(temps), 1)
            report["gpu_temp_samples"] = len(temps)
        else:
            report["gpu_temp_celsius"] = 0
    except Exception as e:
        log.debug("thermal stability test failed: %s", e)
        report["gpu_temp_celsius"] = 0
        report["thermal_error"] = str(e)

    # Clean up GPU memory
    del a, b
    torch.cuda.empty_cache()

    print(json.dumps(report))

except ImportError:
    print(json.dumps({"error": "no_torch"}))
""",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # Longer timeout for thermal test
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if "error" in data:
                log.warning("Benchmark skipped: %s", data["error"])
                return None
            return data
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        log.warning("Benchmark failed: %s", e)
    return None


def run_network_benchmark(scheduler_url: str = None):
    """Benchmark network quality to the scheduler (throughput, jitter, loss).

    Returns dict with throughput_mbps, jitter_ms, packet_loss_pct.
    """
    target = scheduler_url or SCHEDULER_URL
    if not target:
        return {}

    from urllib.parse import urlparse

    parsed = urlparse(target)
    host = parsed.hostname or "127.0.0.1"

    report = {}

    # ── Latency + Jitter + Loss via ping ──
    try:
        r = subprocess.run(
            ["ping", "-c", "20", "-i", "0.2", "-W", "2", host],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0:
            # Parse packet loss
            loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", r.stdout)
            if loss_match:
                report["packet_loss_pct"] = float(loss_match.group(1))

            # Parse rtt min/avg/max/mdev
            rtt_match = re.search(
                r"rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)",
                r.stdout,
            )
            if rtt_match:
                report["latency_min_ms"] = float(rtt_match.group(1))
                report["latency_avg_ms"] = float(rtt_match.group(2))
                report["latency_max_ms"] = float(rtt_match.group(3))
                report["jitter_ms"] = float(rtt_match.group(4))  # mdev ≈ jitter
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # ── Throughput estimate via HTTP download (scheduler /metrics endpoint) ──
    try:
        import requests as req

        # Download a known endpoint multiple times to estimate throughput
        sizes = []
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            resp = req.get(f"{target}/metrics", timeout=5, headers=_api_headers())
            t1 = time.perf_counter()
            sizes.append(len(resp.content))
            times.append(t1 - t0)
        total_bytes = sum(sizes)
        total_time = sum(times)
        if total_time > 0:
            report["throughput_mbps"] = round(total_bytes * 8 / total_time / 1_000_000, 2)
    except Exception as e:
        log.debug("HTTP throughput test failed: %s", e)

    # Defaults for missing values
    report.setdefault("packet_loss_pct", 0)
    report.setdefault("jitter_ms", 0)
    report.setdefault("throughput_mbps", 0)

    return report


# ── Tailscale / Headscale Integration ────────────────────────────────


def setup_tailscale():
    """Join the Tailscale/Headscale mesh network if configured."""
    if not TAILSCALE_ENABLED:
        return False

    try:
        # Check if already running
        r = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            status = json.loads(r.stdout)
            if status.get("BackendState") == "Running":
                log.info(
                    "Headscale mesh already connected (IP: %s)",
                    status.get("TailscaleIPs", ["?"])[0],
                )
                return True

        # Bring up tailscale
        up_cmd = ["tailscale", "up", "--hostname", HOST_ID or "xcelsior-worker"]
        if TAILSCALE_AUTHKEY:
            up_cmd.extend(["--authkey", TAILSCALE_AUTHKEY])
        if HEADSCALE_URL:
            up_cmd.extend(["--login-server", HEADSCALE_URL])

        r = subprocess.run(up_cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            log.info("Headscale mesh connected successfully")
            return True
        else:
            log.warning("Headscale mesh up failed: %s", r.stderr.strip())
            return False

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("Headscale mesh setup failed: %s", e)
        return False


# ── Scheduler Communication (Pull-Based) ────────────────────────────


def _oauth_client_credentials_enabled():
    """Return True when the worker is configured for OAuth client_credentials."""
    return bool(OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET)


def _clear_oauth_access_token_cache():
    """Reset the cached OAuth access token."""
    global _oauth_access_token, _oauth_access_token_expires_at
    with _oauth_lock:
        _oauth_access_token = ""
        _oauth_access_token_expires_at = 0.0


def _oauth_token_endpoint():
    """Resolve the OAuth token endpoint for client_credentials exchange."""
    if OAUTH_TOKEN_URL.strip():
        return OAUTH_TOKEN_URL.strip()
    return _api_url("/oauth/token")


def _oauth_access_token_is_fresh(now=None):
    """Return True when the cached OAuth access token is still usable."""
    now = time.time() if now is None else now
    refresh_skew = max(5, OAUTH_TOKEN_REFRESH_SKEW_SEC)
    return bool(_oauth_access_token) and (_oauth_access_token_expires_at - now) > refresh_skew


def _request_oauth_access_token():
    """Exchange OAuth client credentials for a short-lived bearer token."""
    data = {
        "grant_type": "client_credentials",
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
    }
    if OAUTH_SCOPE:
        data["scope"] = OAUTH_SCOPE

    resp = requests.post(
        _oauth_token_endpoint(),
        data=data,
        timeout=OAUTH_TOKEN_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    payload = resp.json()
    access_token = str(payload.get("access_token", "")).strip()
    token_type = str(payload.get("token_type", "Bearer")).strip().lower()
    try:
        expires_in = int(payload.get("expires_in", 0) or 0)
    except (TypeError, ValueError):
        expires_in = 0
    if token_type != "bearer":
        raise RuntimeError("OAuth token endpoint returned a non-bearer token")
    if not access_token or expires_in <= 0:
        raise RuntimeError("OAuth token response missing access_token or expires_in")
    return access_token, time.time() + expires_in


def _get_oauth_access_token(force_refresh=False):
    """Fetch and cache an OAuth access token for worker API calls."""
    global _oauth_access_token, _oauth_access_token_expires_at
    if not _oauth_client_credentials_enabled():
        return ""

    now = time.time()
    if not force_refresh and _oauth_access_token_is_fresh(now):
        return _oauth_access_token

    with _oauth_lock:
        now = time.time()
        if not force_refresh and _oauth_access_token_is_fresh(now):
            return _oauth_access_token
        try:
            access_token, expires_at = _request_oauth_access_token()
        except Exception as exc:
            _oauth_access_token = ""
            _oauth_access_token_expires_at = 0.0
            log.warning("OAuth client_credentials token request failed: %s", exc)
            return ""
        _oauth_access_token = access_token
        _oauth_access_token_expires_at = expires_at
        return access_token


def _api_headers():
    """Build standard API headers."""
    headers = {"Content-Type": "application/json"}
    access_token = _get_oauth_access_token() if _oauth_client_credentials_enabled() else ""
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    elif API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def _api_url(path):
    """Build absolute API URL."""
    return f"{SCHEDULER_URL.rstrip('/')}{path}"


def heartbeat(gpu_info, host_ip, compute_score=None):
    """Send heartbeat / register with scheduler.

    PUT /host — same as v1 but with optional compute_score.
    Returns True on success.
    """
    data = {
        "host_id": HOST_ID,
        "ip": host_ip,
        "gpu_model": gpu_info["gpu_model"],
        "total_vram_gb": gpu_info["total_vram_gb"],
        "free_vram_gb": gpu_info["free_vram_gb"],
        "cost_per_hour": COST_PER_HOUR,
        # P1.2: report our own version + sha so the control plane knows
        # which bytes are running and can schedule rolling self-updates.
        "agent_version": VERSION,
        "agent_sha256": _self_sha256(),
    }

    try:
        resp = requests.put(
            _api_url("/host"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error("Heartbeat failed: %s", e)
        return False


def report_versions(versions):
    """Report node component versions for admission control.

    POST /agent/versions
    """
    data = {"host_id": HOST_ID, "versions": versions}
    try:
        resp = requests.post(
            _api_url("/agent/versions"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("admitted", False), result.get("details", {})
        return False, {"error": f"HTTP {resp.status_code}"}
    except requests.RequestException as e:
        log.warning("Version report failed: %s", e)
        return False, {"error": str(e)}


def poll_for_work():
    """Poll the scheduler for work assigned to this host.

    GET /agent/work/{host_id}
    Returns list of job dicts, or empty list.
    """
    try:
        resp = requests.get(
            _api_url(f"/agent/work/{HOST_ID}"),
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("instances", [])
        elif resp.status_code == 204:
            return []
        else:
            log.debug("Poll returned %d", resp.status_code)
            return []
    except requests.RequestException as e:
        log.debug("Poll failed: %s", e)
        return []


def check_preemption():
    """Check if any jobs on this host are being preempted.

    GET /agent/preempt/{host_id}
    Returns list of job_ids to preempt.
    """
    try:
        resp = requests.get(
            _api_url(f"/agent/preempt/{HOST_ID}"),
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("preempt_jobs", [])
        return []
    except requests.RequestException:
        return []


def _handle_upgrade_agent(args: dict, cmd_id: str = "", by: str = "?") -> bool:
    """Handle an ``upgrade_agent`` directive from the control plane (P1.2).

    args: {
        "url": "https://xcelsior.ca/static/worker_agent.py",
        "sha256": "<hex>",          # required — we refuse to install unverified bytes
        "min_version": "2.1.0",      # optional — skip if our VERSION >= this
    }

    Flow:
      1. If ``min_version`` <= VERSION, log + skip (we're already at/above target).
      2. Download ``url`` to ``~/.xcelsior/worker_agent.py.new``.
      3. Verify sha256 matches ``args["sha256"]``. Abort on mismatch.
      4. Copy current file to ``~/.xcelsior/worker_agent.py.bak``.
      5. ``os.replace`` new → current path (atomic on same filesystem).
      6. Log + ``sys.exit(0)`` — systemd ``Restart=always`` respawns with the
         new bytes. If the new bytes fail to start, the old VERSION heartbeat
         will cease and the control plane will see the upgrade stalled (the
         rolling-upgrade driver can then roll back from .bak).

    Returns True on successful install (won't return if process exits).
    """
    import hashlib

    url = (args or {}).get("url") or ""
    expected_sha = ((args or {}).get("sha256") or "").lower().strip()
    min_version = (args or {}).get("min_version") or ""

    if not url or not expected_sha or len(expected_sha) != 64:
        log.warning("upgrade_agent cmd=%s bad args: %r", cmd_id, args)
        return False

    # Refuse plaintext URLs — integrity is enforced by sha256 but we also want
    # transport confidentiality + to avoid DNS/HTTP MITM feeding us a valid
    # (attacker-signed) blob alongside a poisoned sha. If your API_URL is
    # http://localhost for local dev, set XCELSIOR_ALLOW_INSECURE_UPGRADE=1.
    if not url.startswith("https://") and os.environ.get(
        "XCELSIOR_ALLOW_INSECURE_UPGRADE"
    ) != "1":
        log.error("upgrade_agent refusing non-https url: %s", url)
        return False

    # Sanity check hex before we do any I/O.
    if any(c not in "0123456789abcdef" for c in expected_sha):
        log.warning("upgrade_agent cmd=%s non-hex sha256", cmd_id)
        return False

    # Version gating — compare as tuples of ints where possible.
    def _vtuple(v: str) -> tuple:
        try:
            return tuple(int(p) for p in v.split(".") if p.isdigit())
        except Exception:
            return ()

    if min_version and _vtuple(VERSION) >= _vtuple(min_version) and _vtuple(min_version):
        # Already at or beyond target and args also asked for the same sha as us: skip.
        if expected_sha == _self_sha256():
            log.info("upgrade_agent cmd=%s already at %s (sha matches) — skipped", cmd_id, VERSION)
            return True

    self_path = Path(__file__).resolve()
    new_path = self_path.with_suffix(".py.new")
    bak_path = self_path.with_suffix(".py.bak")

    try:
        log.info(
            "upgrade_agent cmd=%s by=%s downloading %s (expect sha=%s)",
            cmd_id,
            by,
            url,
            expected_sha[:12],
        )
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code != 200:
            log.warning("upgrade_agent download failed: status=%s", resp.status_code)
            return False
        # Cap download at 10 MB — worker_agent.py is ~160 KB today; anything
        # an order of magnitude larger is a misconfiguration or attack.
        MAX_BYTES = 10 * 1024 * 1024
        hasher = hashlib.sha256()
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            buf.extend(chunk)
            hasher.update(chunk)
            if len(buf) > MAX_BYTES:
                log.error("upgrade_agent refusing oversized body (> %d bytes)", MAX_BYTES)
                return False
        data = bytes(buf)
        actual_sha = hasher.hexdigest()
        if actual_sha != expected_sha:
            log.error(
                "upgrade_agent sha mismatch: expected=%s got=%s — aborting",
                expected_sha,
                actual_sha,
            )
            return False
        new_path.write_bytes(data)
        try:
            os.chmod(new_path, 0o755)
        except OSError:
            pass

        # Backup current before replacing. Best effort — if the copy fails we
        # still proceed (the old bytes exist in the systemd journal + git).
        try:
            bak_path.write_bytes(self_path.read_bytes())
        except OSError as e:
            log.warning("upgrade_agent backup failed (continuing): %s", e)

        os.replace(new_path, self_path)
        log.warning(
            "upgrade_agent cmd=%s installed new agent sha=%s — exiting for systemd restart",
            cmd_id,
            actual_sha[:12],
        )
        # Flush handlers and exit cleanly. systemd Restart=always respawns us.
        logging.shutdown()
        os._exit(0)
    except Exception as e:
        log.exception("upgrade_agent cmd=%s failed: %s", cmd_id, e)
        # Clean up partial file
        try:
            if new_path.exists():
                new_path.unlink()
        except OSError:
            pass
        return False


#: P3/C7 — drain-side allowlist. Defence-in-depth against an API-side
#: bypass: the worker only executes commands on this list. Must stay in
#: sync with routes/agent.py::_AGENT_COMMAND_ALLOWED.
_AGENT_COMMAND_ALLOWED = frozenset({
    "reinject_shell",
    "upgrade_agent",
    "rollback_agent",     # P1.2 — auto-rollback driver restores .bak
    "stop_container",
    "pause_container",
    "start_container",
    "snapshot_container",
})


def drain_agent_commands() -> int:
    """Drain admin control-plane commands from the API and dispatch them.

    GET /agent/commands/{host_id} returns up to 50 pending commands and
    atomically deletes them. Each command dict has id/command/args/created_by.
    Unknown commands are hard-refused (P3/C7) — no forward-compat dispatch:
    if a newer API ships a command this agent doesn't recognise, fail
    loudly rather than silently drop, so operators upgrade agents before
    relying on new behaviour.

    Returns the number of commands successfully dispatched.
    """
    try:
        resp = requests.get(
            _api_url(f"/agent/commands/{HOST_ID}"),
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code != 200:
            return 0
        commands = resp.json().get("commands", []) or []
    except requests.RequestException as e:
        log.debug("drain_agent_commands: request failed: %s", e)
        return 0

    dispatched = 0
    for cmd in commands:
        name = cmd.get("command")
        args = cmd.get("args") or {}
        cmd_id = cmd.get("id")
        by = cmd.get("created_by") or "?"
        # P3/C7 — enforce allowlist before entering any branch.
        if name not in _AGENT_COMMAND_ALLOWED:
            log.warning(
                "drain_agent_commands refused unknown cmd=%s name=%r by=%s",
                cmd_id, name, by,
            )
            _agent_commands_rejected_total.labels(command=str(name)[:32]).inc()
            continue
        try:
            if name == "reinject_shell":
                job_id = args.get("job_id") or ""
                container_name = args.get("container_name") or (f"xcl-{job_id}" if job_id else "")
                if not job_id or not container_name:
                    log.warning("reinject_shell cmd=%s missing args: %r", cmd_id, args)
                    _motd_reinjection_total.labels(result="bad_args").inc()
                    continue
                # Verify container still belongs to us and is running
                inspect = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if inspect.returncode != 0 or inspect.stdout.strip() != "true":
                    log.info(
                        "reinject_shell cmd=%s container=%s not running — skipped",
                        cmd_id,
                        container_name,
                    )
                    _motd_reinjection_total.labels(result="skipped").inc()
                    continue
                log.info(
                    "Executing admin reinject_shell cmd=%s job=%s by=%s", cmd_id, job_id[:8], by
                )
                _inject_ssh_keys(job_id, container_name)
                _motd_reinjection_total.labels(result="success").inc()
                dispatched += 1
            elif name == "upgrade_agent":
                log.info("Executing admin upgrade_agent cmd=%s by=%s", cmd_id, by)
                if _handle_upgrade_agent(args, cmd_id=str(cmd_id or ""), by=str(by)):
                    dispatched += 1
            elif name == "rollback_agent":
                # P1.2 — auto-rollback driver: swap the previously-written
                # ``.bak`` back into place and exit so systemd respawns the
                # old bytes. No network I/O, no sha verification (we trust
                # our own backup). Idempotent: if .bak is absent (e.g. first
                # upgrade never happened or file has been GC'd) we log and
                # skip so the drain loop doesn't keep retrying.
                try:
                    from pathlib import Path as _P
                    _self = _P(__file__).resolve()
                    _bak = _self.with_suffix(".py.bak")
                    if not _bak.exists():
                        log.warning("rollback_agent cmd=%s no .bak file at %s — skipping", cmd_id, _bak)
                        continue
                    # Keep the current bytes around one level deeper so a
                    # subsequent re-upgrade has a recovery path (rotates
                    # .bak → .bak.old → dropped).
                    _old = _self.with_suffix(".py.bak.old")
                    try:
                        _old.write_bytes(_self.read_bytes())
                    except OSError:
                        pass
                    import os as _os
                    _os.replace(_bak, _self)
                    log.warning("rollback_agent cmd=%s restored .bak — exiting for restart", cmd_id)
                    logging.shutdown()
                    _os._exit(0)
                except Exception as e:
                    log.exception("rollback_agent cmd=%s failed: %s", cmd_id, e)
            elif name == "stop_container":
                # P3.2 — billing/admin initiated graceful stop + remove.
                # Replaces the legacy VPS→host `ssh_exec docker kill` path;
                # works over CGNAT because the agent pulls the directive.
                _job_id = str(args.get("job_id") or "")
                cname = str(args.get("container_name") or (f"xcl-{_job_id}" if _job_id else ""))
                if not cname:
                    log.warning("stop_container cmd=%s missing container_name", cmd_id)
                    continue
                log.info("Executing stop_container cmd=%s container=%s by=%s", cmd_id, cname, by)
                try:
                    subprocess.run(
                        ["docker", "stop", "-t", "30", cname],
                        capture_output=True, text=True, timeout=45,
                    )
                    subprocess.run(
                        ["docker", "rm", "-f", cname],
                        capture_output=True, text=True, timeout=15,
                    )
                    _active_containers.pop(_job_id, None)
                    dispatched += 1
                except subprocess.TimeoutExpired:
                    log.warning("stop_container cmd=%s container=%s timed out", cmd_id, cname)
            elif name == "pause_container":
                # P3/A3 — billing-initiated PAUSE (state-preserving).
                # Unlike stop_container, does NOT `docker rm`, so the container
                # can later be restarted cheaply with `docker start`. Used for
                # wallet-low-balance pauses where we expect the user to resume.
                _job_id = str(args.get("job_id") or "")
                cname = str(args.get("container_name") or (f"xcl-{_job_id}" if _job_id else ""))
                if not cname:
                    log.warning("pause_container cmd=%s missing container_name", cmd_id)
                    continue
                log.info("Executing pause_container cmd=%s container=%s by=%s", cmd_id, cname, by)
                try:
                    r = subprocess.run(
                        ["docker", "stop", "-t", "30", cname],
                        capture_output=True, text=True, timeout=45,
                    )
                    if r.returncode == 0:
                        dispatched += 1
                    else:
                        log.warning(
                            "pause_container cmd=%s rc=%s stderr=%s",
                            cmd_id, r.returncode, (r.stderr or "")[:200],
                        )
                except subprocess.TimeoutExpired:
                    log.warning("pause_container cmd=%s container=%s timed out", cmd_id, cname)
            elif name == "start_container":
                # P3.2 — resume a previously-stopped container. The image +
                # volumes are preserved so `docker start` is a cheap restart.
                cname = str(args.get("container_name") or "")
                if not cname:
                    log.warning("start_container cmd=%s missing container_name", cmd_id)
                    continue
                start_job_id = str(args.get("job_id") or "")
                log.info("Executing start_container cmd=%s container=%s by=%s", cmd_id, cname, by)
                try:
                    r = subprocess.run(
                        ["docker", "start", cname],
                        capture_output=True, text=True, timeout=30,
                    )
                    if r.returncode == 0:
                        dispatched += 1
                    else:
                        stderr_trim = (r.stderr or "").strip()[:400]
                        log.warning(
                            "start_container cmd=%s rc=%s stderr=%s",
                            cmd_id, r.returncode, stderr_trim,
                        )
                        # P3/B8 — previous code left the job stuck at status
                        # 'running' (set optimistically by the start path) even
                        # though the container never started. Revert to
                        # 'stopped' with an error message so the UI unsticks
                        # and the user sees a real reason. (Pre-phase-2 this
                        # used 'user_paused'; pause/resume were collapsed into
                        # stop/start.)
                        if start_job_id:
                            try:
                                report_job_status(
                                    start_job_id,
                                    "stopped",
                                    error_message=f"start failed: {stderr_trim}" or
                                                  "start failed: docker start exited non-zero",
                                )
                            except Exception as cb_err:
                                log.warning(
                                    "start_container failure-callback cmd=%s "
                                    "job=%s failed: %s",
                                    cmd_id, start_job_id, cb_err,
                                )
                except subprocess.TimeoutExpired:
                    log.warning("start_container cmd=%s container=%s timed out", cmd_id, cname)
                    if start_job_id:
                        try:
                            report_job_status(
                                start_job_id,
                                "stopped",
                                error_message="start failed: docker start timed out",
                            )
                        except Exception as cb_err:
                            log.warning(
                                "start_container timeout-callback cmd=%s "
                                "job=%s failed: %s",
                                cmd_id, start_job_id, cb_err,
                            )
            elif name == "reset_container":
                # P2 — restart the container with a fresh /workspace scratch
                # dir. Named/bind-mount volumes are untouched; only the
                # container's own ephemeral /workspace gets wiped so the
                # user gets a clean slate without losing mounted data.
                cname = str(args.get("container_name") or "")
                if not cname:
                    log.warning("reset_container cmd=%s missing container_name", cmd_id)
                else:
                    try:
                        # `docker exec <cname> sh -c 'rm -rf /workspace/* /workspace/.[!.]*'`
                        # wipes /workspace in-place, then restart the container
                        # so tmp/log state resets as well.
                        subprocess.run(
                            [
                                "docker",
                                "exec",
                                cname,
                                "sh",
                                "-c",
                                "rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null || true",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        r = subprocess.run(
                            ["docker", "restart", cname],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if r.returncode == 0:
                            dispatched += 1
                        else:
                            log.warning(
                                "reset_container cmd=%s rc=%s stderr=%s",
                                cmd_id,
                                r.returncode,
                                (r.stderr or "").strip()[:400],
                            )
                    except subprocess.TimeoutExpired:
                        log.warning(
                            "reset_container cmd=%s container=%s timed out", cmd_id, cname
                        )
            elif name == "snapshot_container":
                # P3.1 — `docker commit` the running container to a user
                # image tag. Pushed to a registry when XCELSIOR_REGISTRY_URL
                # is set; otherwise the tag remains local to this host.
                image_id = str(args.get("image_id") or "")
                cname = str(args.get("container_name") or "")
                image_ref = str(args.get("image_ref") or "")
                if not (image_id and cname and image_ref):
                    log.warning("snapshot_container cmd=%s missing required args", cmd_id)
                    continue
                log.info(
                    "Executing snapshot_container cmd=%s container=%s ref=%s by=%s",
                    cmd_id, cname, image_ref, by,
                )
                status = "failed"
                size_bytes = 0
                err_msg = ""
                committed_locally = False  # P3/B6 — track so we can rmi on push fail
                try:
                    # Step 1: docker commit (local tag created).
                    commit = subprocess.run(
                        ["docker", "commit", cname, image_ref],
                        capture_output=True, text=True, timeout=900,
                    )
                    if commit.returncode != 0:
                        # P3/B7 — distinguish commit vs push failures so the
                        # UI can surface actionable errors.
                        err_msg = f"commit failed: {(commit.stderr or '').strip()[:480]}"
                    else:
                        committed_locally = True
                        # Best-effort size inspection.
                        try:
                            insp = subprocess.run(
                                ["docker", "image", "inspect", "-f", "{{.Size}}", image_ref],
                                capture_output=True, text=True, timeout=10,
                            )
                            if insp.returncode == 0:
                                size_bytes = int((insp.stdout or "0").strip() or 0)
                        except (subprocess.TimeoutExpired, ValueError):
                            pass
                        registry_url = os.environ.get("XCELSIOR_REGISTRY_URL", "").strip()
                        if not registry_url:
                            # Phase E/E2 — no fallback. API-side now refuses
                            # new snapshots when registry is unset, but a
                            # stale queued command could still arrive here.
                            # Clean up the local commit + report failure.
                            err_msg = "registry_not_configured"
                            try:
                                subprocess.run(
                                    ["docker", "rmi", image_ref],
                                    capture_output=True, text=True, timeout=30,
                                )
                                committed_locally = False
                            except subprocess.TimeoutExpired:
                                pass
                        elif image_ref.startswith(registry_url.rstrip("/")):
                            # Step 2: docker push.
                            push = subprocess.run(
                                ["docker", "push", image_ref],
                                capture_output=True, text=True, timeout=900,
                            )
                            if push.returncode != 0:
                                err_msg = f"push failed: {(push.stderr or '').strip()[:480]}"
                                # P3/B6 — clean up local tag so this host's
                                # disk doesn't fill with orphaned layers from
                                # retried-and-failed pushes.
                                try:
                                    subprocess.run(
                                        ["docker", "rmi", image_ref],
                                        capture_output=True, text=True, timeout=30,
                                    )
                                    committed_locally = False
                                except subprocess.TimeoutExpired:
                                    log.warning(
                                        "snapshot_container rmi-after-push-fail "
                                        "cmd=%s ref=%s timed out", cmd_id, image_ref,
                                    )
                            else:
                                status = "ready"
                        else:
                            # Phase E/E2 — image_ref doesn't match registry
                            # prefix. Shouldn't happen (API builds the ref
                            # from the same env), but fail loudly rather
                            # than silently keeping an unpushable tag.
                            err_msg = (
                                f"image_ref {image_ref!r} does not match "
                                f"registry {registry_url!r}"
                            )
                            try:
                                subprocess.run(
                                    ["docker", "rmi", image_ref],
                                    capture_output=True, text=True, timeout=30,
                                )
                                committed_locally = False
                            except subprocess.TimeoutExpired:
                                pass
                        dispatched += 1
                except subprocess.TimeoutExpired:
                    err_msg = "commit/push timeout"
                    log.warning("snapshot_container cmd=%s container=%s timed out", cmd_id, cname)
                # Report back to API so the user_images row flips out of
                # 'pending'. Best-effort — a lost callback is recoverable
                # by a reconcile sweep that inspects docker images directly.
                try:
                    resp = requests.post(
                        _api_url(f"/user-images/{image_id}/complete"),
                        headers=_api_headers(),
                        json={"status": status, "size_bytes": size_bytes, "error": err_msg},
                        timeout=10,
                    )
                    if resp.status_code >= 400:
                        log.warning(
                            "snapshot_container callback cmd=%s got HTTP %s: %s",
                            cmd_id, resp.status_code, (resp.text or "")[:200],
                        )
                except requests.RequestException as e:
                    log.warning("snapshot_container callback failed cmd=%s: %s", cmd_id, e)
            # P3/C7 — no `else` branch: unknown commands are rejected at
            # the top of the loop via _AGENT_COMMAND_ALLOWED before we
            # reach any if/elif block.
        except Exception as e:
            log.warning("agent command cmd=%s name=%s failed: %s", cmd_id, name, e)
            if name == "reinject_shell":
                _motd_reinjection_total.labels(result="failure").inc()
    return dispatched


def report_job_status(
    job_id,
    status,
    host_id=None,
    container_id=None,
    container_name=None,
    ssh_port=None,
    interactive=None,
    error_message=None,
):
    """Update job status on the scheduler.

    PATCH /instance/{job_id}
    """
    data = {"status": status}
    if host_id:
        data["host_id"] = host_id
    if container_id:
        data["container_id"] = container_id
    if container_name:
        data["container_name"] = container_name
    if ssh_port is not None:
        data["ssh_port"] = ssh_port
    if interactive is not None:
        data["interactive"] = interactive
    if error_message:
        data["error_message"] = error_message[:500]
    try:
        resp = requests.patch(
            _api_url(f"/instance/{job_id}"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException as e:
        log.error("Status update failed for job %s: %s", job_id, e)
        return False


def _push_log_lines(job_id, lines):
    """Push log lines directly to the API (used during pull phase before LogForwarder starts)."""
    body = json.dumps({"lines": lines}).encode()
    try:
        requests.post(
            _api_url(f"/agent/logs/{job_id}"),
            data=body,
            headers=_api_headers(),
            timeout=5,
        )
    except requests.RequestException:
        pass  # best-effort — don't block job execution


def _report_http_ports(job_id, port_map):
    """Report the final {container_port: host_port} mapping to the API.

    Used by the P2.2 subdomain router to resolve
    ``{slug}-{cport}.xcelsior.ca`` without having to re-derive the
    deterministic-with-linear-probe allocation. Best-effort — a failed
    call leaves the public URL unresolved; users can simply retry
    ``POST /instances/{id}/expose`` once the next report lands.
    """
    if not port_map:
        return False
    try:
        resp = requests.post(
            _api_url(f"/instances/{job_id}/http-ports/report"),
            json={"host_id": HOST_ID, "ports": {str(k): int(v) for k, v in port_map.items()}},
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            log.info("Reported http_ports for job %s: %s", job_id, port_map)
            return True
        log.warning(
            "http_ports report for job %s returned %d: %s",
            job_id,
            resp.status_code,
            (resp.text or "")[:200],
        )
    except requests.RequestException as e:
        log.error("http_ports report failed for job %s: %s", job_id, e)
    return False


# ── Lease Protocol ────────────────────────────────────────────────────
# Per REPORT_FEATURE_FINAL.md: clean "lease/claim" protocol instead of
# conflating "assigned" with "running."
#   assign → lease claim → lease renewal → completion/release


def claim_lease(job_id):
    """Claim a lease for a job (assigned → leased).

    POST /agent/lease/claim
    Returns lease details or None on failure.
    """
    data = {"host_id": HOST_ID, "job_id": job_id}
    try:
        resp = requests.post(
            _api_url("/agent/lease/claim"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            log.info(
                "Lease claimed: job=%s lease=%s expires=%.0f",
                job_id,
                result.get("lease_id"),
                result.get("expires_at", 0),
            )
            return result
        else:
            log.warning("Lease claim failed for job %s: HTTP %d", job_id, resp.status_code)
            return None
    except requests.RequestException as e:
        log.error("Lease claim error for job %s: %s", job_id, e)
        return None


def renew_lease(job_id):
    """Renew a lease on a job.

    POST /agent/lease/renew
    Returns lease details or None if expired.
    """
    data = {"host_id": HOST_ID, "job_id": job_id}
    try:
        resp = requests.post(
            _api_url("/agent/lease/renew"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            log.warning("Lease renewal failed for job %s: HTTP %d", job_id, resp.status_code)
            return None
    except requests.RequestException as e:
        log.debug("Lease renewal error for job %s: %s", job_id, e)
        return None


def release_lease(job_id, reason="completed"):
    """Release a lease (job done/failed/preempted).

    POST /agent/lease/release
    """
    data = {"job_id": job_id, "reason": reason}
    try:
        requests.post(
            _api_url("/agent/lease/release"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
    except requests.RequestException:
        pass  # Best-effort


def report_mining_alert(gpu_index, confidence, reason):
    """Alert the scheduler about a mining detection.

    POST /agent/mining-alert
    """
    data = {
        "host_id": HOST_ID,
        "gpu_index": gpu_index,
        "confidence": confidence,
        "reason": reason,
        "timestamp": time.time(),
    }
    try:
        requests.post(
            _api_url("/agent/mining-alert"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
    except requests.RequestException:
        pass  # Best-effort


def report_benchmark(score_data, gpu_model):
    """Report compute benchmark score to scheduler.

    POST /agent/benchmark
    """
    data = {
        "host_id": HOST_ID,
        "gpu_model": gpu_model,
        "score": score_data.get("tflops", 0) / 10,  # XCU = TFLOPS / 10
        "tflops": score_data.get("tflops", 0),
        "details": score_data,
    }
    try:
        requests.post(
            _api_url("/agent/benchmark"),
            json=data,
            headers=_api_headers(),
            timeout=10,
        )
    except requests.RequestException:
        pass


def report_verification(full_report):
    """Submit full verification report to scheduler.

    POST /agent/verify — feeds verification.py's run_verification().
    """
    payload = {
        "host_id": HOST_ID,
        "report": full_report,
    }
    try:
        r = requests.post(
            _api_url("/agent/verify"),
            json=payload,
            headers=_api_headers(),
            timeout=15,
        )
        if r.ok:
            result = r.json()
            log.info(
                "Verification result: state=%s score=%.0f",
                result.get("state", "?"),
                result.get("score", 0),
            )
        else:
            log.warning("Verification submission failed: %s", r.status_code)
    except requests.RequestException as e:
        log.warning("Verification report failed: %s", e)


def report_telemetry(metrics):
    """Push periodic GPU telemetry to scheduler.

    POST /agent/telemetry — stores metrics for monitoring and SLA.
    """
    payload = {
        "host_id": HOST_ID,
        "timestamp": time.time(),
        "metrics": metrics,
    }
    try:
        requests.post(
            _api_url("/agent/telemetry"),
            json=payload,
            headers=_api_headers(),
            timeout=5,
        )
    except requests.RequestException:
        pass  # Best-effort; don't log every failure


# ── Container Image Cache Manager ─────────────────────────────────────
# Per REPORT_XCELSIOR_TECHNICAL_FINAL.md: pull-based agent enables
# "local image caching" (REPORT_EXCELSIOR_TECHNICAL_2.md §3.4).
# LRU tracking, cache eviction, and idle-time pre-pulling.

IMAGE_CACHE_MAX_GB = float(os.environ.get("XCELSIOR_IMAGE_CACHE_MAX_GB", "200"))
IMAGE_CACHE_EVICT_LOW_GB = float(os.environ.get("XCELSIOR_IMAGE_CACHE_EVICT_LOW_GB", "150"))

_image_cache_lock = threading.Lock()
# {image_tag: {"last_used": timestamp, "size_mb": float, "pull_count": int}}
_image_cache_index: dict[str, dict] = {}


def _get_local_images():
    """List locally cached Docker images with sizes."""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {}

        images = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            tag, size_str = parts
            # Parse size string (e.g., "2.5GB", "850MB")
            size_mb = _parse_docker_size(size_str)
            images[tag] = size_mb
        return images
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}


def _parse_docker_size(size_str):
    """Parse Docker size string to MB."""
    size_str = size_str.strip().upper()
    try:
        if "GB" in size_str:
            return float(size_str.replace("GB", "")) * 1024
        elif "MB" in size_str:
            return float(size_str.replace("MB", ""))
        elif "KB" in size_str:
            return float(size_str.replace("KB", "")) / 1024
        elif "B" in size_str:
            return float(size_str.replace("B", "")) / (1024 * 1024)
        return 0
    except (ValueError, TypeError):
        return 0


def _total_cache_size_mb():
    """Get total size of cached images in MB."""
    with _image_cache_lock:
        return sum(entry.get("size_mb", 0) for entry in _image_cache_index.values())


def cache_track_pull(image_tag, size_mb=0):
    """Record that an image was pulled/used. Update LRU tracking."""
    with _image_cache_lock:
        if image_tag in _image_cache_index:
            _image_cache_index[image_tag]["last_used"] = time.time()
            _image_cache_index[image_tag]["pull_count"] += 1
            if size_mb > 0:
                _image_cache_index[image_tag]["size_mb"] = size_mb
        else:
            _image_cache_index[image_tag] = {
                "last_used": time.time(),
                "size_mb": size_mb,
                "pull_count": 1,
            }


def cache_evict_lru(exclude_images: set | None = None):
    """Evict least-recently-used images when cache exceeds limit.

    Evicts until cache is below IMAGE_CACHE_EVICT_LOW_GB.
    Never evicts currently-running container images or images in *exclude_images*.
    """
    exclude_images = exclude_images or set()
    total_mb = _total_cache_size_mb()
    limit_mb = IMAGE_CACHE_MAX_GB * 1024

    if total_mb <= limit_mb:
        return 0

    target_mb = IMAGE_CACHE_EVICT_LOW_GB * 1024
    evicted = 0

    # Get currently running images (do not evict)
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Image}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        running_images = set(result.stdout.strip().split("\n")) if result.returncode == 0 else set()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        running_images = set()

    # Sort by last_used ascending (oldest first)
    with _image_cache_lock:
        sorted_images = sorted(
            _image_cache_index.items(),
            key=lambda x: x[1].get("last_used", 0),
        )

    for image_tag, entry in sorted_images:
        if total_mb <= target_mb:
            break
        if image_tag in running_images or image_tag in exclude_images:
            continue

        # Remove image
        try:
            rm = subprocess.run(
                ["docker", "rmi", image_tag],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if rm.returncode == 0:
                size = entry.get("size_mb", 0)
                total_mb -= size
                evicted += 1
                with _image_cache_lock:
                    _image_cache_index.pop(image_tag, None)
                log.info("Cache evicted: %s (%.0f MB freed)", image_tag, size)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return evicted


def cache_init():
    """Initialize cache index from locally available Docker images.

    Prunes dangling (<none>:<none>) images first so the cache only
    tracks real, reusable tagged images.
    """
    # Remove dangling images — they waste disk and inflate cache size
    try:
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    images = _get_local_images()
    with _image_cache_lock:
        for tag, size_mb in images.items():
            # Skip any residual untagged entries
            if "<none>" in tag:
                continue
            if tag not in _image_cache_index:
                _image_cache_index[tag] = {
                    "last_used": time.time(),
                    "size_mb": size_mb,
                    "pull_count": 0,
                }


def cache_prepull_popular(popular_images, max_concurrent=3):
    """Pre-pull popular images during idle time.

    Called from the main loop when the agent has no active work.
    Only pulls images not already cached, respecting cache limits.
    Pulls up to max_concurrent images in parallel.

    Args:
        popular_images: List of image tags ordered by popularity.
        max_concurrent: Max simultaneous docker pulls.
    """
    import concurrent.futures

    to_pull = []
    for image_tag in popular_images:
        with _image_cache_lock:
            if image_tag in _image_cache_index:
                continue  # Already cached

        total_mb = _total_cache_size_mb()
        if total_mb >= IMAGE_CACHE_MAX_GB * 1024:
            log.debug("Cache full — skipping pre-pull of %s", image_tag)
            break
        to_pull.append(image_tag)

    if not to_pull:
        return

    def _pull_one(img_tag):
        if _shutdown.is_set():
            return False
        log.info("Pre-pulling: %s", img_tag)
        try:
            pull = subprocess.Popen(
                ["docker", "pull", img_tag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # Poll with shutdown check so we can abort quickly on SIGTERM
            while pull.poll() is None:
                if _shutdown.is_set():
                    pull.terminate()
                    pull.wait(timeout=5)
                    return False
                time.sleep(1)
            if pull.returncode == 0:
                images = _get_local_images()
                size_mb = images.get(img_tag, 0)
                cache_track_pull(img_tag, size_mb)
                log.info("Pre-pulled %s (%.0f MB)", img_tag, size_mb)
                return True
            else:
                stderr = pull.stdout.read() if pull.stdout else ""
                log.warning(
                    "Pre-pull failed %s: %s", img_tag, stderr[:200] if stderr else "unknown"
                )
        except subprocess.TimeoutExpired:
            log.warning("Pre-pull timed out: %s (>15 min)", img_tag)
        except FileNotFoundError:
            log.error("docker not found — cannot pre-pull")
        return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        list(pool.map(_pull_one, to_pull))  # list() forces consumption, surfaces exceptions


def fetch_popular_images():
    """Ask the scheduler for popular/frequently-used images.

    GET /agent/popular-images
    Returns list of image tags.
    """
    try:
        resp = requests.get(
            _api_url("/agent/popular-images"),
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("images", [])
        return []
    except requests.RequestException:
        return []


# ── NFS Support ──────────────────────────────────────────────────────


def _mount_nfs(server, path, mount_point):
    """Mount an NFS share for shared model/data storage.

    Args:
        server: NFS server hostname or IP
        path: NFS export path (e.g., /exports/models)
        mount_point: Local mount point (e.g., /mnt/xcelsior-nfs)

    Returns:
        True if mounted successfully, False otherwise.
    """
    try:
        os.makedirs(mount_point, exist_ok=True)

        # Check if already mounted
        r = subprocess.run(
            ["mountpoint", "-q", mount_point],
            capture_output=True,
            timeout=5,
        )
        if r.returncode == 0:
            return True  # Already mounted

        # Mount NFS
        # Use `hard` mount + long timeo/retrans so I/O blocks-and-retries on
        # NFS server reboot instead of silently returning errors (which would
        # cause data corruption on write-back caches). Do not change to `soft`
        # without an explicit durability review.
        mount_cmd = [
            "mount",
            "-t",
            "nfs",
            "-o",
            os.environ.get(
                "XCELSIOR_NFS_MOUNT_OPTS",
                "hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp",
            ),
            f"{server}:{path}",
            mount_point,
        ]
        r = subprocess.run(mount_cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            log.warning("NFS mount failed: %s", r.stderr.strip())
            return False

        return True
    except Exception as e:
        log.warning("NFS mount error: %s", e)
        return False


def _unmount_nfs(mount_point):
    """Unmount an NFS share (best-effort, lazy unmount)."""
    try:
        subprocess.run(
            ["umount", "-l", mount_point],
            capture_output=True,
            timeout=10,
        )
        log.info("NFS unmounted: %s", mount_point)
    except Exception as e:
        log.debug("NFS unmount failed (non-fatal): %s", e)


# ── NVMe Model Cache (Cold Start Optimization) ──────────────────────
# Per Phase 3.4: pre-pull model weights to local NVMe cache on first deploy.
# Tiered cache hierarchy: VRAM → NVMe SSD → Network Storage (NFS)

NVME_CACHE_DIR = os.environ.get("XCELSIOR_NVME_CACHE_DIR", "/mnt/nvme/model-cache")
NVME_CACHE_MAX_GB = int(os.environ.get("XCELSIOR_NVME_CACHE_MAX_GB", "200"))

_nvme_cache_lock = threading.Lock()


def nvme_cache_init():
    """Initialize local NVMe model cache directory."""
    os.makedirs(NVME_CACHE_DIR, exist_ok=True)
    log.info("NVMe model cache initialized at %s (max %d GB)", NVME_CACHE_DIR, NVME_CACHE_MAX_GB)


def nvme_cache_path(model_id: str, revision: str = "main") -> str:
    """Return the local NVMe cache path for a model."""
    safe_name = model_id.replace("/", "--")
    return os.path.join(NVME_CACHE_DIR, f"{safe_name}_{revision}")


def nvme_cache_has(model_id: str, revision: str = "main") -> bool:
    """Check if a model is already cached on local NVMe."""
    path = nvme_cache_path(model_id, revision)
    return os.path.isdir(path) and any(os.scandir(path))


def nvme_cache_size_gb() -> float:
    """Total size of the NVMe model cache in GB."""
    total = 0
    if not os.path.isdir(NVME_CACHE_DIR):
        return 0.0
    for entry in os.scandir(NVME_CACHE_DIR):
        if entry.is_dir():
            for f in os.scandir(entry.path):
                if f.is_file():
                    total += f.stat().st_size
    return total / (1024**3)


def nvme_cache_evict_lru():
    """Evict least-recently-used models from NVMe cache if over limit."""
    import shutil

    if not os.path.isdir(NVME_CACHE_DIR):
        return

    entries = []
    for entry in os.scandir(NVME_CACHE_DIR):
        if entry.is_dir():
            entries.append((entry.path, entry.stat().st_mtime))

    # Sort oldest first
    entries.sort(key=lambda e: e[1])

    while nvme_cache_size_gb() > NVME_CACHE_MAX_GB and entries:
        oldest_path, _ = entries.pop(0)
        log.info("NVMe cache evicting: %s", os.path.basename(oldest_path))
        shutil.rmtree(oldest_path, ignore_errors=True)


def nvme_prepull_model(model_id: str, revision: str = "main", source_nfs: str = "") -> bool:
    """Pre-pull model weights to local NVMe cache for fast cold starts.

    Per Phase 3.4: tiered cache — VRAM → NVMe → Network Storage.
    Downloads model from NFS shared storage or HuggingFace Hub to local NVMe.

    Args:
        model_id: Model identifier (e.g. "meta-llama/Llama-3-70B")
        revision: Model revision/branch
        source_nfs: Optional NFS path to copy from (faster than download)

    Returns:
        True if model is now available on local NVMe.
    """
    if nvme_cache_has(model_id, revision):
        log.debug("Model %s@%s already in NVMe cache", model_id, revision)
        return True

    with _nvme_cache_lock:
        # Double-check under lock
        if nvme_cache_has(model_id, revision):
            return True

        # Evict if cache is full
        nvme_cache_evict_lru()

        dest = nvme_cache_path(model_id, revision)
        os.makedirs(dest, exist_ok=True)

        # Strategy 1: Copy from NFS shared storage (fastest)
        if source_nfs and os.path.isdir(source_nfs):
            try:
                import shutil

                log.info("NVMe pre-pull: copying %s from NFS %s", model_id, source_nfs)
                for item in os.listdir(source_nfs):
                    src = os.path.join(source_nfs, item)
                    dst = os.path.join(dest, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                log.info("NVMe pre-pull complete: %s (from NFS)", model_id)
                return True
            except Exception as e:
                log.warning("NFS copy failed for %s: %s, trying HuggingFace", model_id, e)

        # Strategy 2: Download from HuggingFace Hub
        try:
            dl_cmd = [
                "huggingface-cli",
                "download",
                model_id,
                "--revision",
                revision,
                "--local-dir",
                dest,
                "--quiet",
            ]
            result = subprocess.run(
                dl_cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max for large models
            )
            if result.returncode == 0:
                log.info("NVMe pre-pull complete: %s (from HuggingFace)", model_id)
                return True
            else:
                log.warning("HuggingFace download failed for %s: %s", model_id, result.stderr[:200])
                return False
        except FileNotFoundError:
            log.warning("huggingface-cli not found — cannot pre-pull %s", model_id)
            return False
        except subprocess.TimeoutExpired:
            log.warning("NVMe pre-pull timed out for %s", model_id)
            return False


# ── LUKS Encrypted Volume Provisioning ───────────────────────────────
# Per Phase 10.2: encrypted at rest via LUKS with per-volume key.
# Provides cryptographic erasure on volume delete (destroy LUKS key).

VOLUME_BASE_DIR = os.environ.get("XCELSIOR_VOLUME_DIR", "/mnt/xcelsior/volumes")
VOLUME_KEY_DIR = os.environ.get("XCELSIOR_VOLUME_KEY_DIR", "/etc/xcelsior/volume-keys")


def _ensure_volume_dirs():
    """Create base directories for volumes and keys (if running as root)."""
    os.makedirs(VOLUME_BASE_DIR, exist_ok=True)
    os.makedirs(VOLUME_KEY_DIR, mode=0o700, exist_ok=True)


def provision_encrypted_volume(volume_id: str, size_gb: int) -> bool:
    """Create a LUKS-encrypted volume with a per-volume key.

    Steps:
      1. Create a sparse file as the backing store
      2. Generate a random 256-bit LUKS key
      3. Format with LUKS2 using the key
      4. Open the LUKS device
      5. Create ext4 filesystem
      6. Mount to /mnt/xcelsior/volumes/{volume_id}

    Returns True on success, False on failure.
    """
    _ensure_volume_dirs()

    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)

    try:
        # 1. Create sparse backing file
        subprocess.run(
            ["truncate", "-s", f"{size_gb}G", backing_file],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # 2. Generate per-volume random key (256-bit)
        key_bytes = os.urandom(32)
        old_umask = os.umask(0o077)
        try:
            with open(key_file, "wb") as f:
                f.write(key_bytes)
        finally:
            os.umask(old_umask)

        # 3. LUKS format the backing file
        subprocess.run(
            [
                "cryptsetup",
                "luksFormat",
                "--batch-mode",
                "--type",
                "luks2",
                "--key-file",
                key_file,
                "--cipher",
                "aes-xts-plain64",
                "--key-size",
                "512",
                "--hash",
                "sha256",
                backing_file,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # 4. Open LUKS device
        subprocess.run(
            [
                "cryptsetup",
                "luksOpen",
                "--key-file",
                key_file,
                backing_file,
                mapper_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # 5. Create ext4 filesystem
        dm_path = f"/dev/mapper/{mapper_name}"
        subprocess.run(
            ["mkfs.ext4", "-q", "-L", f"vol-{volume_id[:8]}", dm_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # 6. Mount
        os.makedirs(mount_point, exist_ok=True)
        subprocess.run(
            ["mount", dm_path, mount_point],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        log.info("Volume %s provisioned: %dGB LUKS2+ext4 at %s", volume_id, size_gb, mount_point)
        return True

    except subprocess.CalledProcessError as e:
        log.error("Volume provisioning failed for %s: %s (stderr: %s)", volume_id, e, e.stderr)
        # Cleanup partial state
        _cleanup_partial_volume(volume_id)
        return False
    except Exception as e:
        log.error("Volume provisioning error for %s: %s", volume_id, e)
        _cleanup_partial_volume(volume_id)
        return False


def attach_encrypted_volume(volume_id: str) -> str | None:
    """Open and mount an existing LUKS volume. Returns mount path or None."""
    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)
    dm_path = f"/dev/mapper/{mapper_name}"

    if not os.path.exists(backing_file) or not os.path.exists(key_file):
        log.error("Volume %s: backing file or key not found", volume_id)
        return None

    try:
        # Check if already open
        if not os.path.exists(dm_path):
            subprocess.run(
                ["cryptsetup", "luksOpen", "--key-file", key_file, backing_file, mapper_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        # Mount if not already mounted
        os.makedirs(mount_point, exist_ok=True)
        r = subprocess.run(["mountpoint", "-q", mount_point], capture_output=True, timeout=5)
        if r.returncode != 0:
            subprocess.run(
                ["mount", dm_path, mount_point],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        log.info("Volume %s attached at %s", volume_id, mount_point)
        return mount_point

    except subprocess.CalledProcessError as e:
        log.error("Volume attach failed for %s: %s", volume_id, e.stderr)
        return None


def detach_encrypted_volume(volume_id: str) -> bool:
    """Unmount and close a LUKS volume."""
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)
    dm_path = f"/dev/mapper/{mapper_name}"

    try:
        # Unmount
        r = subprocess.run(["mountpoint", "-q", mount_point], capture_output=True, timeout=5)
        if r.returncode == 0:
            subprocess.run(
                ["umount", mount_point],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        # Close LUKS device
        if os.path.exists(dm_path):
            subprocess.run(
                ["cryptsetup", "luksClose", mapper_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        log.info("Volume %s detached", volume_id)
        return True

    except subprocess.CalledProcessError as e:
        log.error("Volume detach failed for %s: %s", volume_id, e.stderr)
        return False


def destroy_encrypted_volume(volume_id: str) -> bool:
    """Cryptographic erasure: destroy LUKS key then remove backing file.

    Once the key is shredded, the data is irrecoverable regardless
    of whether the backing file still exists.
    """
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)

    # Detach first (best-effort)
    detach_encrypted_volume(volume_id)

    try:
        # Destroy key — cryptographic erasure (overwrite with random then delete)
        if os.path.exists(key_file):
            r = subprocess.run(
                ["shred", "-u", "-z", "-n", "3", key_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.returncode != 0:
                log.critical(
                    "SECURITY: shred failed for volume %s key — key may remain on disk: %s",
                    volume_id,
                    r.stderr,
                )
                return False
            log.info("Volume %s: LUKS key destroyed (cryptographic erasure)", volume_id)

        # Remove backing file
        if os.path.exists(backing_file):
            os.remove(backing_file)

        # Remove mount point directory
        if os.path.isdir(mount_point):
            os.rmdir(mount_point)

        log.info("Volume %s destroyed", volume_id)
        return True

    except Exception as e:
        log.error("Volume destroy failed for %s: %s", volume_id, e)
        return False


def _cleanup_partial_volume(volume_id: str):
    """Best-effort cleanup after a failed provisioning attempt."""
    mapper_name = f"xcelsior-vol-{volume_id[:12]}"
    dm_path = f"/dev/mapper/{mapper_name}"
    key_file = os.path.join(VOLUME_KEY_DIR, f"{volume_id}.key")
    backing_file = os.path.join(VOLUME_BASE_DIR, f"{volume_id}.img")
    mount_point = os.path.join(VOLUME_BASE_DIR, volume_id)

    with suppress(Exception):
        subprocess.run(["umount", mount_point], capture_output=True, timeout=10)
    with suppress(Exception):
        if os.path.exists(dm_path):
            subprocess.run(
                ["cryptsetup", "luksClose", mapper_name], capture_output=True, timeout=10
            )
    with suppress(Exception):
        if os.path.exists(key_file):
            subprocess.run(
                ["shred", "-u", "-z", "-n", "3", key_file],
                capture_output=True,
                timeout=30,
            )
    with suppress(Exception):
        if os.path.exists(backing_file):
            os.remove(backing_file)
    with suppress(Exception):
        if os.path.isdir(mount_point):
            os.rmdir(mount_point)


# ── Phase 7 / §3: exclusive_gpu lifecycle ────────────────────────────
#
# A tier-1 (exclusive_gpu=true) tenant gets the entire 3060 to itself.
# Acquire flow:
#   1) flock /run/xcelsior/gpu.lock              (mutex against other tenants)
#   2) POST /admin/drain  → tower-serverless     (rejects new inference)
#   3) `ollama stop`                             (graceful unload)
#   4) SIGSTOP ollama runner pids                (free VRAM driver state)
#   5) pkill -TERM 'tier3-'                      (background jobs checkpoint)
# Release flow reverses it (SIGCONT, /admin/resume, flock release).
#
# All steps are best-effort: a missing ollama binary or a non-running
# tower-serverless does not abort the job.
GPU_LOCK_PATH = "/run/xcelsior/gpu.lock"
TOWER_SERVERLESS_URL = os.environ.get(
    "TOWER_SERVERLESS_URL", "http://127.0.0.1:8001"
)
EXCLUSIVE_GPU_LOCK_TIMEOUT_S = 60


def _acquire_exclusive_gpu(job_id: str) -> dict:
    """Acquire exclusive access to the local GPU. See module-level docstring.
    Returns a state dict with keys: ok, lock_fd, ollama_pids, error.
    """
    import fcntl
    state: dict = {"ok": False, "lock_fd": None, "ollama_pids": []}

    # 1. flock (blocking with timeout)
    try:
        fd = os.open(GPU_LOCK_PATH, os.O_WRONLY | os.O_CREAT, 0o664)
    except OSError as e:
        state["error"] = f"open(gpu.lock): {e}"
        return state
    deadline = time.time() + EXCLUSIVE_GPU_LOCK_TIMEOUT_S
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() >= deadline:
                with suppress(Exception):
                    os.close(fd)
                state["error"] = (
                    f"flock timeout ({EXCLUSIVE_GPU_LOCK_TIMEOUT_S}s) — "
                    "another tier-1 tenant holds the GPU"
                )
                return state
            time.sleep(0.5)
    state["lock_fd"] = fd
    log.info("exclusive_gpu[%s]: flock acquired", job_id)

    # 2. drain serverless
    try:
        r = requests.post(f"{TOWER_SERVERLESS_URL}/admin/drain", timeout=5)
        log.info(
            "exclusive_gpu[%s]: drain rc=%s body=%s",
            job_id, r.status_code, r.text[:120],
        )
    except Exception as e:
        log.warning("exclusive_gpu[%s]: drain skipped (%s)", job_id, e)

    # 3. ollama stop (graceful unload)
    try:
        subprocess.run(
            ["ollama", "stop"], timeout=10, capture_output=True
        )
        log.info("exclusive_gpu[%s]: ollama stop sent", job_id)
    except FileNotFoundError:
        log.info("exclusive_gpu[%s]: ollama not installed — skipping", job_id)
    except Exception as e:
        log.warning("exclusive_gpu[%s]: ollama stop failed (%s)", job_id, e)

    # 4. SIGSTOP ollama runner processes
    try:
        out = subprocess.run(
            ["pgrep", "-f", "ollama"],
            capture_output=True, text=True, timeout=5,
        )
        for raw in out.stdout.split():
            if not raw.isdigit():
                continue
            pid = int(raw)
            with suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGSTOP)
                state["ollama_pids"].append(pid)
        if state["ollama_pids"]:
            log.info(
                "exclusive_gpu[%s]: SIGSTOP'd ollama pids=%s",
                job_id, state["ollama_pids"],
            )
    except FileNotFoundError:
        pass  # pgrep missing
    except Exception as e:
        log.warning("exclusive_gpu[%s]: ollama SIGSTOP failed (%s)", job_id, e)

    # 5. SIGTERM tier3- background jobs
    try:
        subprocess.run(
            ["pkill", "-TERM", "-f", "tier3-"],
            timeout=5, capture_output=True,
        )
        log.info("exclusive_gpu[%s]: tier3- jobs SIGTERM'd", job_id)
    except Exception as e:
        log.debug("exclusive_gpu[%s]: pkill tier3 (%s)", job_id, e)

    state["ok"] = True
    return state


def _release_exclusive_gpu(job_id: str, state: dict) -> None:
    """Reverse of _acquire_exclusive_gpu. Idempotent and best-effort."""
    import fcntl
    if not state:
        return

    # 1. SIGCONT ollama runners
    for pid in state.get("ollama_pids", []):
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGCONT)

    # 2. resume serverless
    try:
        requests.post(f"{TOWER_SERVERLESS_URL}/admin/resume", timeout=5)
    except Exception as e:
        log.warning("exclusive_gpu[%s]: resume failed (%s)", job_id, e)

    # 3. release flock
    fd = state.get("lock_fd")
    if fd is not None:
        with suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with suppress(Exception):
            os.close(fd)

    log.info("exclusive_gpu[%s]: released", job_id)


# ── Container Lifecycle ──────────────────────────────────────────────


def run_job(job):
    """Execute a job in a hardened Docker container.

    Lifecycle:
      1. Mount NFS volume (if configured)
      1b. Attach encrypted volumes (if specified)
      2. Pull/verify image
      3. Build secure Docker args (from security.py)
      4. Start container
      5. Apply egress rules
      6. Monitor until completion
      7. Report final status
      8. Unmount NFS (if mounted)
      8b. Detach encrypted volumes
    """
    job_id = job.get("job_id", "unknown")
    job_name = job.get("name", "unnamed")
    image = job.get("image", job.get("docker_image", ""))
    command = job.get("command")
    env_vars = job.get("environment", {}) or {}
    # P1.4: inject platform env AFTER user env so the reserved XCELSIOR_* keys
    # always win (user containers cannot spoof job id / owner / api url).
    try:
        _gpu_info_for_env = get_gpu_info()
    except Exception:
        _gpu_info_for_env = {}
    env_vars = {**env_vars, **build_platform_env(job, _gpu_info_for_env)}
    volumes = job.get("volumes", [])

    # NFS configuration — from job dict or env vars
    nfs_server = job.get("nfs_server") or os.environ.get("XCELSIOR_NFS_SERVER", "")
    nfs_path = job.get("nfs_path") or os.environ.get("XCELSIOR_NFS_PATH", "")
    nfs_mount_point = job.get("nfs_mount_point") or os.environ.get(
        "XCELSIOR_NFS_MOUNT", "/mnt/xcelsior-nfs"
    )
    nfs_mounted = False
    log_fwd = None
    is_interactive = bool(job.get("interactive", False))

    if not image:
        log.error("Job %s has no image specified — skipping", job_id)
        report_job_status(job_id, "failed")
        return

    container_name = f"xcl-{job_id}"

    # ── Requeue guard: stop old container if this job was already running ──
    with _active_lock:
        if job_id in _active_containers:
            log.info("Job %s already has active container — stopping for requeue", job_id)
            del _active_containers[job_id]
    _kill_container(container_name)  # No-op if container doesn't exist
    _remove_container(container_name)  # Force remove stale container to prevent name conflict
    time.sleep(2)  # Let old monitor thread notice removal and exit

    log.info("Starting job %s (%s) — image=%s", job_id, job_name, image)

    # ── Lease Protocol: claim lease before starting work ──
    # Job arrives as "assigned"; agent claims lease → "leased" → then "running"
    lease_info = claim_lease(job_id)
    lease_interval = (lease_info or {}).get("duration_sec", 300) // 2  # Renew at half-life

    # Report starting status (leased → starting); will become "running" after
    # the container is actually created (image pull + docker run may take minutes).
    report_job_status(job_id, "starting", host_id=HOST_ID)

    with _active_lock:
        _active_containers[job_id] = container_name

    # Start lease renewal thread (only when we have an active lease)
    _lease_stop = threading.Event()

    def _lease_renewal_loop():
        consecutive_failures = 0
        while not _lease_stop.is_set() and not _shutdown.is_set():
            _lease_stop.wait(lease_interval)
            if _lease_stop.is_set() or _shutdown.is_set():
                break
            result = renew_lease(job_id)
            if not result:
                consecutive_failures += 1
                log.warning(
                    "Lease renewal failed for job %s — attempt %d",
                    job_id,
                    consecutive_failures,
                )
                if consecutive_failures >= 3:
                    # Lease may have expired after a VPS restart; try to reclaim.
                    new_lease = claim_lease(job_id)
                    if new_lease:
                        consecutive_failures = 0
                        log.info("Lease reclaimed for running job %s", job_id)
                    else:
                        log.error(
                            "Lease reclaim failed for job %s — giving up renewal loop",
                            job_id,
                        )
                        break
            else:
                consecutive_failures = 0

    if lease_info:
        lease_thread = threading.Thread(
            target=_lease_renewal_loop,
            name=f"lease-{job_id[:8]}",
            daemon=True,
        )
        lease_thread.start()
    else:
        log.warning(
            "Job %s: no active lease — skipping renewal loop (lease claim failed)",
            job_id,
        )

    encrypted_vol_ids = []
    managed_vol_mounts = []  # /mnt/xcelsior-volumes/{vid} paths for cleanup
    current_stage = "preparing instance"

    # ── Phase 7 / §3: acquire exclusive GPU before any container/IO setup ──
    exclusive_gpu = bool(job.get("exclusive_gpu"))
    _excl_state: dict = {}
    if exclusive_gpu:
        log.info(
            "Job %s requested exclusive_gpu — acquiring tier-1 GPU lock",
            job_id,
        )
        _excl_state = _acquire_exclusive_gpu(job_id)
        if not _excl_state.get("ok"):
            err = _excl_state.get("error", "unknown")
            log.error(
                "Job %s: exclusive_gpu acquire failed: %s", job_id, err
            )
            report_job_status(
                job_id, "failed",
                error_message=f"exclusive_gpu acquire failed: {err}",
            )
            release_lease(job_id, "failed")
            return

    try:
        # 0. Mount NFS volume (if configured — skip for interactive instances
        #    which don't need shared storage and the 30s timeout blocks startup)
        if nfs_server and nfs_path and not is_interactive:
            nfs_mounted = _mount_nfs(nfs_server, nfs_path, nfs_mount_point)
            if nfs_mounted:
                log.info("NFS mounted: %s:%s → %s", nfs_server, nfs_path, nfs_mount_point)
                volumes = list(volumes) + [f"{nfs_mount_point}:/data/nfs:rw"]
            else:
                log.warning("NFS mount failed — continuing without shared storage")

        # 0b. Attach encrypted volumes (if specified)
        for vol in job.get("encrypted_volumes", []):
            vol_id = vol.get("volume_id")
            vol_mount = vol.get("mount_path", f"/data/vol-{vol_id[:8]}")
            vol_mode = vol.get("mode", "rw")
            if vol_id:
                mount_path = attach_encrypted_volume(vol_id)
                if mount_path:
                    volumes = list(volumes) + [f"{mount_path}:{vol_mount}:{vol_mode}"]
                    encrypted_vol_ids.append(vol_id)
                    log.info("Encrypted volume %s attached at %s", vol_id, vol_mount)
                else:
                    log.warning("Encrypted volume %s attach failed — skipping", vol_id)

        # 0c. Mount managed volumes (volume_ids from job payload)
        nfs_vol_server = os.environ.get("XCELSIOR_NFS_SERVER", "")
        nfs_vol_export_base = os.environ.get("XCELSIOR_NFS_EXPORT_BASE", "/exports/volumes")
        vol_mount_paths = job.get("volume_mounts", {})  # vid → container path from API
        _vol_idx = 0
        for vid in job.get("volume_ids", []):
            vol_host_mount = f"/mnt/xcelsior-volumes/{vid}"
            vol_nfs_path = f"{nfs_vol_export_base}/{vid}"
            # Determine container mount path — from payload or auto-assign
            container_path = vol_mount_paths.get(vid)
            if not container_path:
                container_path = "/workspace" if _vol_idx == 0 else f"/workspace/vol-{_vol_idx}"
            if nfs_vol_server:
                mounted = _mount_nfs(nfs_vol_server, vol_nfs_path, vol_host_mount)
                if mounted:
                    volumes = list(volumes) + [f"{vol_host_mount}:{container_path}:rw"]
                    managed_vol_mounts.append(vol_host_mount)
                    log.info(
                        "Managed volume %s mounted: %s → %s", vid, vol_host_mount, container_path
                    )
                else:
                    log.warning("Managed volume %s mount failed — skipping", vid)
            else:
                log.warning("Managed volume %s: NFS server not configured — skipping", vid)
            _vol_idx += 1

        # 0d. Encrypted workspace (ephemeral LUKS volume on GPU host)
        encrypted_ws_vid = None
        if job.get("encrypted_workspace"):
            encrypted_ws_vid = f"ws-{job_id}"
            ws_size_gb = 20  # Default ephemeral workspace size
            if provision_encrypted_volume(encrypted_ws_vid, ws_size_gb):
                ws_mount = os.path.join(VOLUME_BASE_DIR, encrypted_ws_vid)
                # Ensure container user can write
                subprocess.run(["chmod", "1777", ws_mount], capture_output=True, timeout=10)
                volumes = list(volumes) + [f"{ws_mount}:/workspace:rw"]
                log.info(
                    "Encrypted workspace provisioned for job %s: %dGB LUKS at %s",
                    job_id,
                    ws_size_gb,
                    ws_mount,
                )
                _push_log_lines(
                    job_id,
                    [
                        {
                            "message": "Encrypted workspace ready (LUKS2+AES-256)",
                            "level": "info",
                            "timestamp": time.time(),
                        }
                    ],
                )
            else:
                log.error("Encrypted workspace provisioning failed for job %s — aborting", job_id)
                report_job_status(job_id, "failed")
                return

        # 1. Pull image (with cache tracking + LRU eviction)
        current_stage = "pulling image"

        # Check if image already cached — skip eviction + pull if so
        with _image_cache_lock:
            image_already_cached = image in _image_cache_index

        if image_already_cached:
            log.info("Image %s already cached — skipping pull", image)
            _push_log_lines(
                job_id,
                [
                    {
                        "message": f"Image {image} (cached ✓)",
                        "level": "info",
                        "timestamp": time.time(),
                    }
                ],
            )
            cache_track_pull(image)
        else:
            cache_evict_lru(exclude_images={image})  # Evict before pulling — protect job's image
            log.info("Pulling image %s...", image)
            _push_log_lines(
                job_id,
                [{"message": f"Pulling image {image}…", "level": "info", "timestamp": time.time()}],
            )

            pull_proc = subprocess.Popen(
                ["docker", "pull", image],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            pull_last_send = time.time()
            pull_lines = []
            for raw_line in pull_proc.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                # Send pull progress to API every 3 seconds (avoid flooding)
                now = time.time()
                if now - pull_last_send >= 3.0:
                    pull_lines.append({"message": line, "level": "info", "timestamp": now})
                    _push_log_lines(job_id, pull_lines)
                    pull_lines = []
                    pull_last_send = now
                else:
                    pull_lines.append({"message": line, "level": "info", "timestamp": now})
            # Flush remaining
            if pull_lines:
                _push_log_lines(job_id, pull_lines)

            pull_rc = pull_proc.wait(timeout=1800)
            if pull_rc != 0:
                log.error("Image pull failed for %s (exit %d)", image, pull_rc)
                _push_log_lines(
                    job_id,
                    [
                        {
                            "message": f"Image pull failed (exit {pull_rc})",
                            "level": "error",
                            "timestamp": time.time(),
                        }
                    ],
                )
                report_job_status(job_id, "failed")
                return

            # Track in LRU cache index
            local_images = _get_local_images()
            cache_track_pull(image, local_images.get(image, 0))

        # 2. Determine runtime
        gpu_info = get_gpu_info()
        runtime_name = "runc"
        is_interactive = bool(job.get("interactive", False))
        if PREFER_GVISOR and not is_interactive:
            runtime_name, reason = recommend_runtime(gpu_info["gpu_model"])
            log.info("Runtime: %s (%s)", runtime_name, reason)
        elif is_interactive:
            log.info("Runtime: runc (interactive GPU — gVisor nvproxy incompatible)")

        # Multi-GPU: get requested GPU count from job
        num_gpus = job.get("num_gpus", 1) or 1
        if num_gpus > 1:
            log.info("Multi-GPU job: requesting %d GPUs", num_gpus)

        # 3. Build secure Docker args
        # Container-side SSH port is always 22 (OpenSSH default). The
        # `ssh_port` we report back to the control plane is the PUBLIC
        # host port users connect to — if we read it back here from the
        # job payload on a requeue we'd end up with `-p 14096:14096`
        # and sshd (listening on 22) would refuse the gateway's relay,
        # producing "Connection closed by <vps>" on the client. See
        # incident 2026-04-23. Keep this value hardcoded.
        ssh_port = 22

        # Interactive mode: override entrypoint to keep container alive
        # and expose SSH port for user access
        extra_docker_args = []
        effective_command = command
        if is_interactive:
            log.info("INTERACTIVE MODE for job %s — container will stay running", job_id)
            # --- Interactive terminal UI (stdout banner + status notes) ---
            # This whole section is LOCKED at TERMINAL_UI_VERSION = "v1" per
            # user approval on 2026-04-22. Do not edit the echo lines, the
            # final 'Terminal ready …' summary lines, or the 'Tip:'/'SSH
            # daemon …'/'OpenSSH server …' notes without bumping the version
            # marker AND updating tests/test_terminal_ui_v1.py in lockstep.
            TERMINAL_UI_VERSION = "v1"  # noqa: F841 — referenced by tests via regex
            # Override entrypoint with an init script that prints startup info
            # (forwarded to UI via LogForwarder) then keeps the container alive.
            init_script = (
                "echo '[xcelsior] Initialising interactive instance…';"
                "echo '[xcelsior] GPU:' $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected');"
                "echo '[xcelsior] CUDA:' $(nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+' || echo 'N/A');"
                "echo '[xcelsior] Python:' $(python3 --version 2>/dev/null || python --version 2>/dev/null || echo 'N/A');"
                "echo '[xcelsior] PyTorch:' $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo 'N/A');"
                "echo '[xcelsior] Setting up SSH…';"
                "exec tail -f /dev/null"
            )
            extra_docker_args.extend(["--entrypoint", "sh"])
            effective_command = ["-c", init_script]
            # Expose SSH port (map to a unique host port based on job_id hash)
            host_port = 10000 + (int(job_id[:4], 16) % 55000)
            extra_docker_args.extend(["-p", f"{host_port}:{ssh_port}"])
            # P2.1: bind-mount host-side tools dir (rsync, git, etc) read-only
            # so interactive instances have these on $PATH without paying an
            # apt-install tax at boot. We skip the mount when the dir doesn't
            # exist on the host so unprovisioned dev machines still work.
            host_tools_dir = os.environ.get(
                "XCELSIOR_TOOLS_DIR", "/var/lib/xcelsior/tools"
            )
            try:
                if os.path.isdir(host_tools_dir):
                    extra_docker_args.extend(
                        ["-v", f"{host_tools_dir}:/opt/xcelsior/bin:ro"]
                    )
                    log.info("Mounted host tools: %s -> /opt/xcelsior/bin (ro)", host_tools_dir)
            except Exception as e:
                log.debug("Tools mount skipped: %s", e)
            # P2.1: publish user-requested HTTP ports. We allocate each host
            # port deterministically in the 55000-59999 range so the public
            # reverse proxy (P2.2) can resolve {job}-{port}.xcelsior.ca without
            # a lookup. Port 22 is reserved — validated at the API boundary.
            _user_ports = job.get("exposed_ports") or []
            # P2.2: capture the final {container_port: host_port} mapping so
            # we can report it back after docker run succeeds. The API uses
            # this to serve `/instances/{id}/expose` and the nginx internal
            # route lookup — without this, collisions from the linear probe
            # are invisible to the control plane.
            port_map: dict[str, int] = {}
            if isinstance(_user_ports, (list, tuple)):
                _seen: set[int] = set()
                _hports_used: set[int] = set()
                try:
                    _base = int(job_id[:4], 16)
                except (ValueError, TypeError):
                    _base = 0
                for _p in _user_ports:
                    try:
                        cport = int(_p)
                    except (TypeError, ValueError):
                        continue
                    # Port 22 is reserved for our SSH gateway path — silently
                    # skip if a user re-lists it in exposed_ports rather than
                    # erroring (RunPod / Vast both treat 22 as implicit SSH).
                    if cport < 1 or cport > 65535 or cport == 22 or cport in _seen:
                        continue
                    _seen.add(cport)
                    # Linear-probe to avoid host-port collisions when two
                    # different cports hash to the same hport (rare but real
                    # — without this, the second -p silently shadows the
                    # first and the user sees a confusing "port already
                    # bound" error from dockerd).
                    hport = 55000 + (_base + cport) % 5000
                    for _attempt in range(5000):
                        if hport not in _hports_used:
                            break
                        hport = 55000 + (hport - 55000 + 1) % 5000
                    else:
                        log.warning("No free host port for cport=%d; skipping", cport)
                        continue
                    _hports_used.add(hport)
                    extra_docker_args.extend(["-p", f"{hport}:{cport}"])
                    port_map[str(cport)] = hport
                    log.info("Expose port: host:%d -> container:%d", hport, cport)
            # Interactive containers need writable filesystem for SSH
            # Remove --read-only and add writable tmpfs for home/ssh
            extra_docker_args.extend(
                [
                    "--tmpfs",
                    "/home:rw,size=1g",
                    "--tmpfs",
                    "/run:rw,size=64m",
                    "--tmpfs",
                    "/var/log:rw,size=256m",
                ]
            )
            log.info("SSH port mapping: host:%d → container:%d", host_port, ssh_port)

        docker_args = build_secure_docker_args(
            image=image,
            container_name=container_name,
            gpu=True,
            num_gpus=num_gpus,
            runtime=runtime_name,
            environment=env_vars,
            volumes=volumes,
            labels={
                "xcelsior.managed": "true",
                "xcelsior.job_id": job_id,
                "xcelsior.container_name": container_name,
            },
            command=effective_command,
            extra_args=extra_docker_args if extra_docker_args else None,
            interactive=is_interactive,
        )

        # 4. Start container — remove stale container with same name (e.g. requeue)
        current_stage = "starting container"
        _remove_container(container_name)
        log.info("Starting container %s", container_name)
        _push_log_lines(
            job_id,
            [
                {
                    "message": "Pull complete — starting container…",
                    "level": "info",
                    "timestamp": time.time(),
                }
            ],
        )
        start = subprocess.run(
            docker_args,
            capture_output=True,
            text=True,
            timeout=CONTAINER_START_TIMEOUT,
        )
        if start.returncode != 0:
            log.error("Container start failed: %s", start.stderr.strip())
            _push_log_lines(
                job_id,
                [
                    {
                        "message": f"Container start failed: {start.stderr.strip()}",
                        "level": "error",
                        "timestamp": time.time(),
                    }
                ],
            )
            report_job_status(job_id, "failed")
            return

        container_id = start.stdout.strip()[:12]
        log.info("Container started: %s", container_id)
        _push_log_lines(
            job_id,
            [
                {
                    "message": f"Container started ({container_id})",
                    "level": "info",
                    "timestamp": time.time(),
                }
            ],
        )

        # 4a. Register container metadata with scheduler (status still "starting")
        #     so the web terminal backend can resolve container_name before we
        #     flip to "running". Keeping the UI badge on "starting" until SSH
        #     injection actually finishes means "Running" is a truthful signal.
        report_job_status(
            job_id,
            "starting",
            host_id=HOST_ID,
            container_id=container_id,
            container_name=container_name,
        )

        # 4b. Inject user's SSH keys into the container (best-effort, 45s base
        #     budget, extended by 45s when openssh-server must be installed).
        #     _inject_ssh_keys is guaranteed to emit a final summary line via
        #     both the log stream AND the container's PID-1 stdout, so the user
        #     never sees "Setting up SSH…" hang indefinitely.
        _inject_ssh_keys(job_id, container_name, interactive=is_interactive)

        # 4c. SSH is either ready or has emitted a reason — flip to "running".
        report_job_status(
            job_id,
            "running",
            host_id=HOST_ID,
            container_id=container_id,
            container_name=container_name,
        )

        # P2.2: report the final HTTP port mapping to the API so
        # `/instances/{id}/expose` and the nginx subdomain router can
        # resolve public URLs. Best-effort — a failed report just means
        # the public URL won't resolve yet; the next heartbeat cycle
        # will retry via _report_http_ports_with_retry.
        if is_interactive and port_map:
            _report_http_ports(job_id, port_map)

        # For interactive jobs, report SSH connection info
        if is_interactive:
            report_job_status(job_id, "running", ssh_port=host_port, interactive=True)
            log.info("Interactive instance %s ready — SSH port %d", job_id, host_port)
            _push_log_lines(
                job_id,
                [
                    {
                        "message": f"Interactive instance ready — SSH: root@{get_host_ip() or 'host'}:{host_port}",
                        "level": "info",
                        "timestamp": time.time(),
                    },
                ],
            )

        # P2.1: user-supplied provisioning hooks. These run AFTER SSH is ready
        # so the banner timing is not affected, each capped at 15 s via the
        # `timeout` coreutil so a stuck clone or hanging init can't strand
        # the instance.
        _run_provisioning_hooks(job_id, container_name, job)

        # P2.3: Jupyter / VSCode (code-server) auto-launch. Runs after
        # provisioning hooks so user init scripts can seed a venv /
        # conda env before Jupyter installs into the default python.
        # Never fatal — a failed auto-launch must not kill the instance.
        try:
            _run_auto_launch(job_id, container_name, job)
        except Exception as e:
            log.debug("auto_launch error (non-fatal): %s", e)

        # 5. Apply egress rules (best-effort)
        try:
            egress_rules = build_egress_iptables_rules(container_name)
            for rule in egress_rules:
                subprocess.run(
                    rule.split(),
                    capture_output=True,
                    timeout=5,
                )
        except Exception as e:
            log.debug("Egress rules failed (non-fatal): %s", e)

        # 6. Start log forwarding (container → API → SSE → frontend)
        log_fwd = LogForwarder(job_id, container_name)
        log_fwd.start()

        # 7. Monitor container until completion
        current_stage = "monitoring container"
        if is_interactive:
            _monitor_interactive(job_id, container_name, log_forwarder=log_fwd)
        else:
            _monitor_container(job_id, container_name, log_forwarder=log_fwd)

    except subprocess.TimeoutExpired:
        log.error("Job %s timed out while %s", job_id, current_stage)
        _push_log_lines(
            job_id,
            [
                {
                    "message": f"Timed out while {current_stage}",
                    "level": "error",
                    "timestamp": time.time(),
                }
            ],
        )
        _kill_container(container_name)
        report_job_status(job_id, "failed")
        release_lease(job_id, "failed")
    except Exception as e:
        log.error("Job %s failed: %s", job_id, e, exc_info=True)
        _push_log_lines(
            job_id,
            [
                {
                    "message": f"Instance failed: {e}",
                    "level": "error",
                    "timestamp": time.time(),
                }
            ],
        )
        _kill_container(container_name)
        report_job_status(job_id, "failed", error_message=str(e))
        release_lease(job_id, "failed")
    finally:
        # Stop log forwarding (final flush)
        if log_fwd:
            try:
                log_fwd.stop()
            except Exception:
                pass

        # Stop lease renewal thread
        _lease_stop.set()
        with _active_lock:
            _active_containers.pop(job_id, None)

        # Release exclusive GPU lock (Phase 7 / §3) — must run last so any
        # checkpoint/cleanup work above completes while we still own the GPU.
        if exclusive_gpu and _excl_state.get("ok"):
            _release_exclusive_gpu(job_id, _excl_state)

        # Unmount NFS if we mounted it
        if nfs_mounted:
            _unmount_nfs(nfs_mount_point)

        # Unmount managed volumes
        for vol_mount in managed_vol_mounts:
            try:
                _unmount_nfs(vol_mount)
            except Exception as e:
                log.warning("Failed to unmount managed volume %s: %s", vol_mount, e)

        # Detach encrypted volumes
        for vol_id in encrypted_vol_ids:
            detach_encrypted_volume(vol_id)

        # Destroy ephemeral encrypted workspace (cryptographic erasure)
        if encrypted_ws_vid:
            destroy_encrypted_volume(encrypted_ws_vid)
            log.info("Encrypted workspace destroyed for job %s (cryptographic erasure)", job_id)


# ── Log Forwarding ───────────────────────────────────────────────────
# Streams container stdout/stderr to the API via HTTP POST batches.
# Disk-backed buffer survives transient API downtime.
# Gzip compression on batches > 1KB, exponential backoff retry.

import gzip as _gzip
import tempfile as _tempfile

_LOG_BATCH_SIZE = 50  # flush every N lines
_LOG_FLUSH_INTERVAL = 1.0  # or every N seconds
_LOG_BACKOFF_MAX = 60  # max retry delay (seconds)
_LOG_SPOOL_MAX = 50_000  # max lines in disk spool before dropping oldest


class LogForwarder:
    """Forward container stdout/stderr to the API server.

    Runs two threads:
    - _tail_thread: reads `docker logs --follow --timestamps`
    - _flush_thread: batches lines and POSTs them to /agent/logs/{job_id}
    """

    def __init__(self, job_id: str, container_name: str):
        self.job_id = job_id
        self.container_name = container_name
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._spool_path = os.path.join(_tempfile.gettempdir(), f"xcelsior-logs-{job_id}.jsonl")
        self._tail_proc = None
        self._tail_thread = None
        self._flush_thread = None
        self._backoff = 1.0

    def start(self):
        """Start tailing container logs and flushing to the API."""
        self._tail_thread = threading.Thread(
            target=self._tail_loop, daemon=True, name=f"log-tail-{self.job_id[:8]}"
        )
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name=f"log-flush-{self.job_id[:8]}"
        )
        self._tail_thread.start()
        self._flush_thread.start()
        log.debug("LogForwarder started for %s", self.job_id)

    def stop(self):
        """Stop tailing and do a final flush."""
        self._stop.set()
        if self._tail_proc:
            with suppress(Exception):
                self._tail_proc.kill()
        # Wait for threads to finish processing
        if self._tail_thread:
            self._tail_thread.join(timeout=5)
        if self._flush_thread:
            self._flush_thread.join(timeout=5)
        # Final flush of remaining lines after threads have stopped
        self._flush_batch()
        # Clean up spool file
        with suppress(FileNotFoundError):
            os.remove(self._spool_path)
        log.debug("LogForwarder stopped for %s", self.job_id)

    def _tail_loop(self):
        """Run `docker logs --follow --timestamps` and buffer lines."""
        try:
            self._tail_proc = subprocess.Popen(
                ["docker", "logs", "--follow", "--timestamps", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for raw_line in self._tail_proc.stdout:
                if self._stop.is_set():
                    break
                line = raw_line.rstrip("\n")
                if not line:
                    continue

                # Docker --timestamps format: "2026-04-05T12:00:00.123456789Z <message>"
                ts = time.time()
                msg = line
                if len(line) > 31 and line[0].isdigit() and "T" in line[:30]:
                    # Try to parse the Docker timestamp
                    space = line.find(" ")
                    if space > 0:
                        try:
                            dt = datetime.fromisoformat(line[:space].rstrip("Z"))
                            ts = dt.timestamp()
                            msg = line[space + 1 :]
                        except (ValueError, IndexError):
                            pass

                level = "stderr" if "ERROR" in msg[:50] or "FATAL" in msg[:50] else "info"

                entry = {"message": msg, "level": level, "timestamp": ts}
                with self._lock:
                    self._buffer.append(entry)

                # Flush if batch is full
                if len(self._buffer) >= _LOG_BATCH_SIZE:
                    self._flush_batch()

        except Exception as e:
            if not self._stop.is_set():
                log.debug("Log tail ended for %s: %s", self.job_id, e)
        finally:
            if self._tail_proc:
                with suppress(Exception):
                    self._tail_proc.kill()

    def _flush_loop(self):
        """Periodically flush buffered log lines to the API."""
        while not self._stop.is_set():
            self._stop.wait(_LOG_FLUSH_INTERVAL)
            if self._buffer:
                self._flush_batch()

    def _flush_batch(self):
        """Send buffered lines to the API. Retry with exponential backoff."""
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:_LOG_BATCH_SIZE]
            self._buffer = self._buffer[_LOG_BATCH_SIZE:]

        # Spool to disk (WAL-style) so we don't lose lines if API is down
        try:
            spool_size = (
                os.path.getsize(self._spool_path) if os.path.exists(self._spool_path) else 0
            )
        except OSError:
            spool_size = 0
        if spool_size < 10_000_000:  # 10MB cap
            try:
                with open(self._spool_path, "a") as f:
                    for entry in batch:
                        f.write(json.dumps(entry) + "\n")
            except OSError:
                pass  # disk full — best-effort

        body = json.dumps({"lines": batch}).encode()

        # Gzip compress if > 1KB
        headers = {**_api_headers()}
        if len(body) > 1024:
            body = _gzip.compress(body)
            headers["Content-Encoding"] = "gzip"

        try:
            resp = requests.post(
                _api_url(f"/agent/logs/{self.job_id}"),
                data=body,
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                self._backoff = 1.0  # reset on success
                # Drain from disk spool what we successfully sent
                self._drain_spool(len(batch))
            else:
                log.debug("Log POST returned %d for %s", resp.status_code, self.job_id)
                self._backoff_wait()
        except requests.RequestException:
            self._backoff_wait()

    def _backoff_wait(self):
        """Wait with exponential backoff, capped at _LOG_BACKOFF_MAX."""
        self._stop.wait(self._backoff)
        self._backoff = min(self._backoff * 2, _LOG_BACKOFF_MAX)

    def _drain_spool(self, count: int):
        """Remove successfully sent lines from disk spool."""
        try:
            with open(self._spool_path, "r") as f:
                remaining = f.readlines()[count:]
            if remaining:
                with open(self._spool_path, "w") as f:
                    f.writelines(remaining[-_LOG_SPOOL_MAX:])
            else:
                os.remove(self._spool_path)
        except (FileNotFoundError, OSError):
            pass


def _monitor_container(job_id, container_name, log_forwarder=None):
    """Monitor a running container until it exits or is preempted."""
    check_interval = 5  # seconds

    while not _shutdown.is_set():
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            status = result.stdout.strip()

            if status == "exited":
                # Check exit code
                exit_result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                exit_code = int(exit_result.stdout.strip()) if exit_result.stdout.strip() else -1

                if exit_code == 0:
                    log.info("Job %s completed successfully", job_id)
                    report_job_status(job_id, "completed")
                    release_lease(job_id, "completed")
                else:
                    log.warning("Job %s exited with code %d", job_id, exit_code)
                    # Collect logs for debugging
                    try:
                        logs = subprocess.run(
                            ["docker", "logs", "--tail", "50", container_name],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if logs.stdout:
                            log.info("Container output (last 50 lines):\n%s", logs.stdout[-2000:])
                    except Exception as e:
                        log.debug("docker logs retrieval failed: %s", e)
                    report_job_status(job_id, "failed")
                    release_lease(job_id, "failed")

                # Cleanup container
                _remove_container(container_name)
                return

            elif status not in ("running", "created"):
                log.warning("Container %s in unexpected state: %s", container_name, status)
                report_job_status(job_id, "failed")
                release_lease(job_id, "failed")
                _remove_container(container_name)
                return

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            log.debug("Container inspect failed — may have been removed")
            report_job_status(job_id, "failed")
            return

        # Sleep in small increments to respond to shutdown/requeue quickly
        for _ in range(check_interval):
            if _shutdown.is_set():
                return
            with _active_lock:
                if job_id not in _active_containers:
                    return  # Requeued or preempted — let new run_job take over
            time.sleep(1)


def _monitor_interactive(job_id, container_name, log_forwarder=None):
    """Monitor an interactive container — stays running until cancelled or shutdown.

    Unlike batch containers, interactive containers run indefinitely.
    The container is only stopped when:
    - User cancels via API (preemption signal)
    - Worker shuts down gracefully
    - Container crashes unexpectedly
    """
    check_interval = 10  # Check less frequently — container is meant to stay alive

    log.info(
        "Monitoring interactive container %s (job %s) — will run until stopped",
        container_name,
        job_id,
    )

    while not _shutdown.is_set():
        # Check if this job has been preempted/cancelled
        with _active_lock:
            if job_id not in _active_containers:
                log.info("Interactive job %s removed from active containers — stopping", job_id)
                _kill_container(container_name)
                report_job_status(job_id, "cancelled")
                release_lease(job_id, "cancelled")
                return

        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            status = result.stdout.strip()

            if status == "exited":
                # Interactive container has stopped. This happens in three
                # scenarios, all of which we must handle correctly:
                #   1. `docker stop` via the agent command queue (billing
                #      stop/pause, admin kill) — the DB has already been
                #      flipped to stopped/paused/cancelled/terminated; don't
                #      overwrite it with "failed".
                #   2. User cleanly exited their shell (exit_code 0) or the
                #      process received SIGTERM/SIGINT/SIGKILL from a stop
                #      (exit codes 130/137/143). Report "stopped", not
                #      "failed" — the instance is resumable via restart.
                #   3. Non-zero, non-signal exit (e.g. OOM, segfault in the
                #      user's entrypoint). This is the only genuine "failed"
                #      case.
                exit_result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                try:
                    exit_code = int(exit_result.stdout.strip())
                except (TypeError, ValueError):
                    exit_code = -1

                # Clean-exit codes: 0 (explicit exit), 130 (SIGINT/Ctrl-C),
                # 137 (SIGKILL from `docker stop` timeout), 143 (SIGTERM
                # graceful). Anything else is a real crash.
                _CLEAN_EXIT_CODES = {0, 130, 137, 143}
                reported_status = (
                    "stopped" if exit_code in _CLEAN_EXIT_CODES else "failed"
                )
                log.info(
                    "Interactive container %s exited (code %d) — reporting %s",
                    container_name,
                    exit_code,
                    reported_status,
                )
                report_job_status(job_id, reported_status)
                release_lease(job_id, reported_status)
                _remove_container(container_name)
                return

            elif status not in ("running", "created", "restarting"):
                # Unexpected Docker state (paused, dead, removing). Treat
                # "paused" as a legitimate user-initiated pause (reported
                # elsewhere) and exit the monitor without clobbering status;
                # anything else is a hard failure.
                if status == "paused":
                    log.info(
                        "Interactive container %s is paused — monitor exiting", container_name
                    )
                    return
                log.warning(
                    "Interactive container %s in unexpected state: %s", container_name, status
                )
                report_job_status(job_id, "failed")
                release_lease(job_id, "failed")
                _remove_container(container_name)
                return

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            # `docker inspect` failed — the container was almost certainly
            # removed by the `stop_container` agent handler in another
            # drain cycle. That handler already flipped the DB status, so
            # reporting "failed" here would clobber the correct state.
            log.debug(
                "Interactive container %s inspect failed — assuming removed by stop_container",
                container_name,
            )
            return

        # Sleep in small increments to respond to shutdown/cancel/requeue quickly
        for _ in range(check_interval):
            if _shutdown.is_set():
                log.info("Shutdown signal — stopping interactive container %s", container_name)
                _kill_container(container_name)
                report_job_status(job_id, "cancelled")
                release_lease(job_id, "cancelled")
                return
            with _active_lock:
                if job_id not in _active_containers:
                    break  # Requeued — exit sleep, handle at top of loop
            time.sleep(1)


def _run_provisioning_hooks(job_id: str, container_name: str, job: dict):
    """P2.1 — apply optional init_script and git_repo, both capped at 15 s.

    Runs via ``docker exec`` against an already-running container. Each hook
    fails open so a broken hook can never wedge the instance: a timeout just
    surfaces as a warning in the user's log stream and execution proceeds. The
    interactive terminal UI v1 banner has already finished by the time we
    reach here, so these hooks do not affect the "SSH ready" perceived boot
    time.
    """
    if not job.get("interactive"):
        return
    init_script = (job.get("init_script") or "").strip()
    git_repo = (job.get("git_repo") or "").strip()
    if not init_script and not git_repo:
        return

    def _exec_capped(tag: str, shell_cmd: str, seconds: int = 15) -> None:
        # `timeout --kill-after=2 15 bash -c …` — hard-kill after grace so a
        # runaway process doesn't hold the agent thread. We still honour the
        # agent's outer timeout as a belt-and-braces measure.
        started = time.time()
        cmd = [
            "docker",
            "exec",
            container_name,
            "sh",
            "-c",
            f"timeout --kill-after=2 {seconds} bash -c {shlex.quote(shell_cmd)}"
            f" 2>&1 | tail -c 4096",
        ]
        _push_log_lines(
            job_id,
            [{"message": f"[xcelsior] {tag} starting (≤{seconds}s)…", "level": "info", "timestamp": started}],
        )
        try:
            # Outer timeout is slightly larger to let `timeout` emit its own
            # exit code first, giving us a cleaner message.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 5)
            out = (proc.stdout or "").strip()
            if proc.returncode == 124:  # coreutils timeout signal
                _push_log_lines(
                    job_id,
                    [
                        {
                            "message": f"[xcelsior] {tag} exceeded {seconds}s — continuing without waiting",
                            "level": "warn",
                            "timestamp": time.time(),
                        }
                    ],
                )
                return
            if proc.returncode != 0:
                _push_log_lines(
                    job_id,
                    [
                        {
                            "message": f"[xcelsior] {tag} failed (rc={proc.returncode}): {out[-512:]}",
                            "level": "warn",
                            "timestamp": time.time(),
                        }
                    ],
                )
                return
            elapsed = int((time.time() - started) * 1000)
            _push_log_lines(
                job_id,
                [
                    {
                        "message": f"[xcelsior] {tag} ok ({elapsed} ms)",
                        "level": "info",
                        "timestamp": time.time(),
                    }
                ],
            )
        except subprocess.TimeoutExpired:
            _push_log_lines(
                job_id,
                [
                    {
                        "message": f"[xcelsior] {tag} hung — abandoned",
                        "level": "warn",
                        "timestamp": time.time(),
                    }
                ],
            )
        except Exception as e:
            log.debug("provisioning %s error: %s", tag, e)

    # Git clone first — users' init scripts often want the repo present.
    if git_repo:
        # git_repo has been validated at the API boundary to be a plain
        # https:// URL with no credentials; we still wrap in quotes.
        safe_repo = shlex.quote(git_repo)
        clone_cmd = (
            "mkdir -p /workspace && cd /workspace && "
            # Idempotency: a relaunch with the same persistent volume will
            # already have /workspace/repo. Remove it so the user always
            # gets a fresh clone of the requested URL (matches `git clone`
            # semantics on a clean dir).
            "rm -rf /workspace/repo && "
            "if command -v git >/dev/null 2>&1; then "
            f"git clone --depth 1 {safe_repo} repo 2>&1 | tail -n 10; "
            "else echo 'git not installed in image'; fi"
        )
        _exec_capped("git_repo", clone_cmd, seconds=15)

    if init_script:
        # Caller's script runs under bash -lc so profile is loaded. We write
        # it to a file inside the container (via stdin) rather than inlining
        # to avoid shell-escaping pitfalls with multiline scripts.
        # docker exec -i streams stdin → temp file → bash -l.
        try:
            started = time.time()
            _push_log_lines(
                job_id,
                [{"message": "[xcelsior] init_script starting (≤15s)…", "level": "info", "timestamp": started}],
            )
            write_cmd = [
                "docker",
                "exec",
                "-i",
                container_name,
                "sh",
                "-c",
                "cat > /tmp/xcelsior_init.sh && chmod +x /tmp/xcelsior_init.sh && "
                "timeout --kill-after=2 15 bash -l /tmp/xcelsior_init.sh 2>&1 | tail -c 4096",
            ]
            proc = subprocess.run(
                write_cmd,
                input=init_script,
                capture_output=True,
                text=True,
                timeout=20,
            )
            out = (proc.stdout or "").strip()
            if proc.returncode == 124:
                _push_log_lines(
                    job_id,
                    [
                        {
                            "message": "[xcelsior] init_script exceeded 15s — continuing without waiting",
                            "level": "warn",
                            "timestamp": time.time(),
                        }
                    ],
                )
            elif proc.returncode != 0:
                _push_log_lines(
                    job_id,
                    [
                        {
                            "message": f"[xcelsior] init_script failed (rc={proc.returncode}): {out[-512:]}",
                            "level": "warn",
                            "timestamp": time.time(),
                        }
                    ],
                )
            else:
                elapsed = int((time.time() - started) * 1000)
                _push_log_lines(
                    job_id,
                    [
                        {
                            "message": f"[xcelsior] init_script ok ({elapsed} ms)",
                            "level": "info",
                            "timestamp": time.time(),
                        }
                    ],
                )
        except subprocess.TimeoutExpired:
            _push_log_lines(
                job_id,
                [
                    {
                        "message": "[xcelsior] init_script hung — abandoned",
                        "level": "warn",
                        "timestamp": time.time(),
                    }
                ],
            )
        except Exception as e:
            log.debug("provisioning init_script error: %s", e)


def _auto_launch_token(job_id: str) -> str:
    """Derive a per-instance secret for Jupyter / code-server auth.

    Combines the job_id with a host-side secret so leaked job_ids alone
    can't be used to log in. Falls back to a job-id-only hash when no
    host secret is configured (still opaque to the outside world; just
    predictable across hosts that share a job_id which never happens
    in practice).
    """
    host_secret = os.environ.get("HOST_SECRET") or os.environ.get(
        "XCELSIOR_HOST_SECRET", ""
    )
    return hashlib.sha256(f"{job_id}:{host_secret}".encode()).hexdigest()[:32]


def _run_auto_launch(job_id: str, container_name: str, job: dict) -> None:
    """P2.3 — bring up Jupyter and/or code-server inside the container.

    Mirrors the safety posture of ``_run_provisioning_hooks``: each
    launcher has a 15 s wall-clock cap, never blocks the agent thread,
    and surfaces one user-visible log line per service (the pip /
    curl install can take most of the budget on a cold image so a
    longer grace would just waste the user's perception of boot time).

    Security:
      * Token = sha256(job_id + HOST_SECRET)[:32] — 128 bits.
      * Token is never echoed to stdout; we print a prefix hash only
        (`token_sha=xxxxxxxx`) and the user retrieves it via the API.
      * ``--allow-root`` is acceptable because the container is
        ephemeral and already isolated by gVisor/Kata (see security.py).
    """
    if not job.get("interactive"):
        return
    services = job.get("auto_launch") or []
    if isinstance(services, str):
        services = [s.strip().lower() for s in services.split(",") if s.strip()]
    else:
        services = [str(s).strip().lower() for s in services if str(s).strip()]
    if not services:
        return

    token = _auto_launch_token(job_id)
    token_prefix = hashlib.sha256(token.encode()).hexdigest()[:8]

    def _exec_bg(tag: str, shell_cmd: str, seconds: int = 15) -> None:
        started = time.time()
        _push_log_lines(
            job_id,
            [{
                "message": f"[xcelsior] auto-launch {tag} starting…",
                "level": "info",
                "timestamp": started,
            }],
        )
        # ``docker exec -d`` detaches the exec process so the agent
        # returns immediately; we still wrap the inner command with
        # ``nohup`` + ``&`` so the process survives the exec session.
        cmd = [
            "docker", "exec", "-d", container_name,
            "bash", "-lc", shell_cmd,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()[-256:]
                _push_log_lines(
                    job_id,
                    [{
                        "message": f"[xcelsior] auto-launch {tag} rc={proc.returncode}: {err}",
                        "level": "warn",
                        "timestamp": time.time(),
                    }],
                )
                return
            elapsed_ms = int((time.time() - started) * 1000)
            _push_log_lines(
                job_id,
                [{
                    "message": f"[xcelsior] auto-launch {tag} ok ({elapsed_ms} ms) token_sha={token_prefix}",
                    "level": "info",
                    "timestamp": time.time(),
                }],
            )
        except subprocess.TimeoutExpired:
            _push_log_lines(
                job_id,
                [{
                    "message": f"[xcelsior] auto-launch {tag} exceeded {seconds}s — abandoned",
                    "level": "warn",
                    "timestamp": time.time(),
                }],
            )
        except Exception as e:
            log.debug("auto_launch %s error: %s", tag, e)

    ports: dict[str, int] = {}
    for svc in services:
        if svc == "jupyter":
            # JupyterLab on :8888. We install to a user-writable path
            # if pip is available and lab is not already present; if
            # both fail we try ``jupyter`` (classic notebook) as a
            # last resort. Token is passed via env so it doesn't show
            # up in ``ps``.
            shell = (
                f"export JUPY_TOKEN={shlex.quote(token)}; "
                "mkdir -p /workspace && cd /workspace && "
                "( command -v jupyter >/dev/null 2>&1 || "
                "  pip install --quiet --no-cache-dir jupyterlab 2>/dev/null || "
                "  pip3 install --quiet --no-cache-dir jupyterlab 2>/dev/null || true ); "
                "nohup jupyter lab --allow-root --no-browser --ip=0.0.0.0 "
                "--port=8888 --ServerApp.token=\"$JUPY_TOKEN\" "
                "--ServerApp.password='' "
                ">/tmp/xcelsior-jupyter.log 2>&1 &"
            )
            _exec_bg("jupyter", shell, seconds=30)
            ports["jupyter"] = 8888
        elif svc == "vscode":
            # code-server on :8443. Install via the official script if
            # not already present (fast if present; ~15 s on first boot).
            shell = (
                f"export PASSWORD={shlex.quote(token)}; "
                "( command -v code-server >/dev/null 2>&1 || "
                "  (curl -fsSL https://code-server.dev/install.sh | sh -s -- --quiet 2>/dev/null) || true ); "
                "nohup code-server --bind-addr 0.0.0.0:8443 --auth password "
                ">/tmp/xcelsior-code-server.log 2>&1 &"
            )
            _exec_bg("vscode", shell, seconds=45)
            ports["vscode"] = 8443
        else:
            _push_log_lines(
                job_id,
                [{
                    "message": f"[xcelsior] auto-launch {svc} not supported — skipping",
                    "level": "warn",
                    "timestamp": time.time(),
                }],
            )

    # Stash the port mapping in job.payload.http_ports so P2.2's
    # subdomain proxy can route `<job>-8888.xcelsior.ca` → host:port.
    # Best-effort: failing to record the mapping just means the user
    # must SSH-tunnel manually. Report back via the API so the UI can
    # display the URL.
    if ports:
        try:
            api_url = os.environ.get("XCELSIOR_API_URL", "").rstrip("/")
            if api_url:
                host_id = os.environ.get("HOST_ID") or os.environ.get("XCELSIOR_HOST_ID", "")
                payload = {
                    "host_id": host_id,
                    "ports": ports,
                    "token_sha": token_prefix,
                    "token": token,
                }
                try:
                    requests.post(
                        f"{api_url}/instances/{job_id}/auto-launch/report",
                        json=payload,
                        timeout=5,
                        headers={"X-Host-Secret": os.environ.get("HOST_SECRET", "")},
                    )
                except Exception as e:
                    log.debug("auto_launch report failed: %s", e)
        except Exception as e:
            log.debug("auto_launch report outer error: %s", e)


def _inject_ssh_keys(job_id: str, container_name: str, interactive: bool = False):
    """Fetch the job owner's SSH public keys from the API and inject into the container.

    Sets up /root/.ssh/authorized_keys and starts sshd if available.
    Best-effort: failures are logged but don't block the job. For interactive
    jobs, pushes user-visible log lines at each phase so the UI log stream
    doesn't silently end at "Setting up SSH…".

    Production guarantees (Phase 1.4 hardening):
      * Hard wall-clock budget of 45s — no single subprocess can exceed
        min(its-default, remaining-budget).
      * The `finally:` block ALWAYS emits a single summary line via _note()
        AND mirrors it to the container's PID-1 stdout (/proc/1/fd/1) so
        an already-attached web terminal sees the final state inline.
      * Exceptions and timeouts are caught and translated into a
        user-readable summary — nothing ever escapes this function.
    """

    def _note(msg: str, level: str = "info"):
        """Push a user-visible log line (only for interactive jobs)."""
        if interactive:
            try:
                _push_log_lines(
                    job_id, [{"message": msg, "level": level, "timestamp": time.time()}]
                )
            except Exception:
                pass

    def _mirror_to_container(msg: str):
        """Best-effort: echo a line to the container's PID-1 stdout so it
        shows up in docker logs AND any web terminal that's already attached
        via ``docker exec``. Wrapped in its own try/except with a 3s timeout
        so this can never block the finally block."""
        if not interactive:
            return
        try:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    f"printf '%s\\n' {_shell_quote(msg)} > /proc/1/fd/1 2>/dev/null || true",
                ],
                capture_output=True,
                timeout=3,
                check=False,
            )
        except Exception:
            pass

    start = time.monotonic()
    # Base budget 45s; extended by 45s if we have to install sshd on-demand.
    deadline = start + 45.0

    def _extend_budget(seconds: float) -> None:
        nonlocal deadline
        deadline += seconds

    def _remaining(default: float) -> float:
        """Clamp a subprocess timeout to the remaining wall-clock budget.
        Never returns less than 1s (so very-near-deadline calls still get
        a real chance, and TimeoutExpired will cleanly bubble to finally)."""
        return max(1.0, min(default, deadline - time.monotonic()))

    # Tracking flags for final summary composition.
    keys: list[str] = []
    sshd_present: bool = False
    sshd_started: bool = False
    root_password: str = ""
    final_msg: str = "[xcelsior] SSH setup complete"
    final_level: str = "info"

    try:
        # --- Fetch authorized keys from API (10s budget clamped to remaining) ---
        try:
            resp = requests.get(
                _api_url(f"/agent/ssh-keys/{job_id}"),
                headers=_api_headers(),
                timeout=_remaining(10),
            )
            if resp.status_code == 200:
                keys = resp.json().get("keys", []) or []
            else:
                log.debug("SSH keys endpoint returned %d for job %s", resp.status_code, job_id)
        except Exception as e:
            log.warning("SSH key fetch failed for job %s: %s", job_id, e)

        if not keys:
            _note(
                "Tip: add an SSH public key at Settings → SSH Keys to enable direct SSH (root@host:port) into this instance. The web terminal works without a key.",
                level="info",
            )
            log.info(
                "No SSH keys for job %s — skipping authorized_keys setup; sshd will still start for future key injection",
                job_id,
            )

        # --- Prepare /root/.ssh (always — sshd needs it even with no keys) ---
        subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", "/root/.ssh"],
            capture_output=True,
            timeout=_remaining(5),
        )
        subprocess.run(
            ["docker", "exec", container_name, "chmod", "700", "/root/.ssh"],
            capture_output=True,
            timeout=_remaining(5),
        )

        if keys:
            # Write authorized_keys
            authorized_keys = "\n".join(keys) + "\n"
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    f"echo {_shell_quote(authorized_keys)} > /root/.ssh/authorized_keys && "
                    "chmod 600 /root/.ssh/authorized_keys",
                ],
                capture_output=True,
                timeout=_remaining(5),
            )
            log.info(
                "Injected %d SSH key(s) into container %s for job %s",
                len(keys),
                container_name,
                job_id,
            )
            _note(f"Injected {len(keys)} SSH key(s)")

        # --- Detect sshd, install if missing ---
        sshd_check = subprocess.run(
            ["docker", "exec", container_name, "which", "sshd"],
            capture_output=True,
            timeout=_remaining(5),
        )
        sshd_present = sshd_check.returncode == 0

        if not sshd_present:
            # Install openssh-server on demand so SSH keys actually work on
            # images that ship without it (pytorch, cuda, etc). Detect package
            # manager; install quietly; re-check for sshd. Extends the budget
            # by 45s only when we actually have to install.
            _extend_budget(45.0)
            _note("Installing OpenSSH server in container (one-time setup)…")
            install_script = (
                "set -e; "
                "if command -v apt-get >/dev/null 2>&1; then "
                "  export DEBIAN_FRONTEND=noninteractive; "
                "  apt-get update -qq >/dev/null 2>&1 && "
                "  apt-get install -y -qq --no-install-recommends openssh-server >/dev/null 2>&1; "
                "elif command -v dnf >/dev/null 2>&1; then "
                "  dnf install -y -q openssh-server >/dev/null 2>&1; "
                "elif command -v yum >/dev/null 2>&1; then "
                "  yum install -y -q openssh-server >/dev/null 2>&1; "
                "elif command -v apk >/dev/null 2>&1; then "
                "  apk add --quiet --no-cache openssh-server >/dev/null 2>&1; "
                "elif command -v microdnf >/dev/null 2>&1; then "
                "  microdnf install -y -q openssh-server >/dev/null 2>&1; "
                "else "
                "  echo 'no-package-manager' >&2; exit 2; "
                "fi; "
                "command -v sshd >/dev/null || "
                "  [ -x /usr/sbin/sshd ] || "
                "  { echo 'sshd still missing after install' >&2; exit 3; }"
            )
            install = subprocess.run(
                ["docker", "exec", container_name, "sh", "-c", install_script],
                capture_output=True,
                timeout=_remaining(30),
            )
            sshd_present = install.returncode == 0
            if sshd_present:
                log.info("Installed openssh-server in container %s", container_name)
                _note("OpenSSH server installed")
            else:
                err = (
                    install.stderr.decode(errors="replace").strip()
                    if isinstance(install.stderr, bytes)
                    else (install.stderr or "").strip()
                )[:200]
                log.warning("sshd install failed in %s: %s", container_name, err)
                _note(
                    "Could not install OpenSSH server in this image — web terminal still works. "
                    "Add SSH keys at Settings → SSH Keys; direct SSH will be enabled on next launch with a compatible image.",
                    level="warning",
                )

        if sshd_present:
            # Generate host keys if missing
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    "ls /etc/ssh/ssh_host_*_key 2>/dev/null || ssh-keygen -A",
                ],
                capture_output=True,
                timeout=_remaining(10),
            )
            # Ensure PermitRootLogin is enabled and configure keepalive so SSH
            # sessions don't get cut by NAT/firewall idle timeouts.
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    # Allow root login via either SSH key or password. We generate
                    # a strong random password per instance (see below) so users
                    # can still connect even if they haven't uploaded a public key.
                    "sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config 2>/dev/null || true; "
                    "sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config 2>/dev/null || true; "
                    "sed -i 's/^#*KbdInteractiveAuthentication.*/KbdInteractiveAuthentication no/' /etc/ssh/sshd_config 2>/dev/null || true; "
                    "sed -i 's/^#*ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config 2>/dev/null || true; "
                    "sed -i '/^ClientAliveInterval/d;/^ClientAliveCountMax/d;/^TCPKeepAlive/d' /etc/ssh/sshd_config 2>/dev/null || true; "
                    "printf '\\nPermitRootLogin yes\\nPasswordAuthentication yes\\nKbdInteractiveAuthentication no\\nClientAliveInterval 30\\nClientAliveCountMax 6\\nTCPKeepAlive yes\\n' >> /etc/ssh/sshd_config",
                ],
                capture_output=True,
                timeout=_remaining(5),
            )

            # Generate a strong random root password per instance so users can
            # connect even without an uploaded SSH key. This matches the UX of
            # Lambda / RunPod / Paperspace — the dashboard surfaces it in the
            # connection panel, and it's unique-per-job so compromise is scoped.
            # 20 chars, URL/terminal-safe alphabet (no ambiguous 0/O/1/l/I).
            _pw_alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
            root_password = "".join(
                secrets.choice(_pw_alphabet) for _ in range(20)
            )
            try:
                pw_set = subprocess.run(
                    [
                        "docker", "exec", "-i", container_name,
                        "chpasswd",
                    ],
                    input=f"root:{root_password}\n".encode(),
                    capture_output=True,
                    timeout=_remaining(5),
                )
                if pw_set.returncode != 0:
                    err = pw_set.stderr.decode(errors="replace").strip()
                    log.warning("chpasswd failed in %s: %s", container_name, err)
                    root_password = ""  # don't surface a password that wasn't set
            except Exception as e:
                log.warning("chpasswd exception in %s: %s", container_name, e)
                root_password = ""

            # Set up MOTD + custom shell prompt for the interactive instance.
            short = job_id[:8]
            motd = (
                "\n"
                "  \033[36m╔═══════════════════════════════════════════════════════╗\033[0m\n"
                "  \033[36m║\033[0m  \033[1;35m✦ Xcelsior\033[0m \033[2m— GPU compute, on demand\033[0m              \033[36m║\033[0m\n"
                "  \033[36m╚═══════════════════════════════════════════════════════╝\033[0m\n"
                "\n"
                f"  \033[2mInstance\033[0m  \033[1m{short}\033[0m\n"
                f"  \033[2mContainer\033[0m \033[1m{container_name}\033[0m\n"
                "  \033[2mGPU      \033[0m $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'detached')\n"
                "  \033[2mDocs     \033[0m \033[34mhttps://xcelsior.ca/docs\033[0m\n"
                "\n"
                "  \033[2mTip: run \033[0m\033[33mnvidia-smi\033[0m\033[2m to see GPU status, \033[0m\033[33mexit\033[0m\033[2m to disconnect.\033[0m\n"
                "\n"
            )
            # Shell rc snippet — idempotent. Sets PS1, prints Last login + MOTD once per session.
            # Uses a sentinel env var so it doesn't print twice if both /etc/profile.d and
            # /root/.bashrc end up sourcing it.
            profile = (
                "# Xcelsior interactive shell setup (managed — safe to re-run)\n"
                f"export XCELSIOR_INSTANCE='{short}'\n"
                "# Coloured prompt: user@xcelsior-<id>:cwd $\n"
                "PS1='\\[\\e[1;36m\\]\\u\\[\\e[0m\\]@\\[\\e[1;35m\\]xcelsior-'\"$XCELSIOR_INSTANCE\"'\\[\\e[0m\\]:\\[\\e[1;33m\\]\\w\\[\\e[0m\\]\\$ '\n"
                "export PS1\n"
                'if [ -z "$XCELSIOR_MOTD_SHOWN" ] && [ -t 1 ] && [ -r /etc/motd.xcelsior ]; then\n'
                "  export XCELSIOR_MOTD_SHOWN=1\n"
                "  printf 'Last login: %s\\n' \"$(date '+%a %b %e %H:%M:%S %Y')\" 2>/dev/null || true\n"
                '  eval "echo \\"$(cat /etc/motd.xcelsior)\\""\n'
                "fi\n"
            )
            # Loader injected into every login-shell startup file. Sourcing /etc/profile.d/xcelsior.sh
            # directly is safe because the file itself is idempotent (sentinel env guard).
            loader = "[ -r /etc/profile.d/xcelsior.sh ] && . /etc/profile.d/xcelsior.sh"
            # `grep -qF` matches literal string; `|| echo >>` makes the append idempotent.
            append_loader = (
                f"for rc in /etc/profile /etc/bash.bashrc /root/.bashrc /root/.profile /root/.bash_profile; do "
                f'  touch "$rc" 2>/dev/null; '
                f"  grep -qF 'profile.d/xcelsior.sh' \"$rc\" 2>/dev/null || "
                f'  echo {_shell_quote(loader)} >> "$rc"; '
                f"done"
            )
            # Some minimal images (Alpine, nvidia/cuda:*-base, distroless-ish) ship /etc/profile
            # WITHOUT the `for f in /etc/profile.d/*.sh` loop, so profile.d entries are dead.
            # Detect & inject the loop if missing.
            ensure_profile_d_loop = (
                "grep -qF '/etc/profile.d' /etc/profile 2>/dev/null || "
                "printf '\\n# Xcelsior: source /etc/profile.d/*.sh\\n"
                'for f in /etc/profile.d/*.sh; do [ -r "$f" ] && . "$f"; done\\n\' '
                ">> /etc/profile"
            )
            # Force root's login shell to bash if available (default on ubuntu is /bin/bash already,
            # but `nvidia/cuda:*-base` on debian-slim defaults root to /bin/sh which doesn't do PS1).
            set_root_shell = (
                "if command -v bash >/dev/null 2>&1; then "
                '  (command -v usermod >/dev/null 2>&1 && usermod -s "$(command -v bash)" root) '
                '  || (command -v chsh    >/dev/null 2>&1 && chsh -s "$(command -v bash)" root) '
                "  || sed -i 's|^root:\\(.*\\):[^:]*$|root:\\1:'\"$(command -v bash)\"'|' /etc/passwd 2>/dev/null "
                "  || true; "
                "fi"
            )
            subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    f"echo {_shell_quote(motd)} > /etc/motd.xcelsior && "
                    f"echo {_shell_quote(profile)} > /etc/profile.d/xcelsior.sh && "
                    "chmod 644 /etc/motd.xcelsior /etc/profile.d/xcelsior.sh && "
                    + ensure_profile_d_loop
                    + " ; "
                    + append_loader
                    + " ; "
                    + set_root_shell
                    + " ; "
                    # Silence default debian/ubuntu motd noise so ours is the only banner shown
                    "rm -f /etc/update-motd.d/* 2>/dev/null; " ": > /etc/motd 2>/dev/null || true",
                ],
                capture_output=True,
                timeout=_remaining(15),
            )

            # Start sshd. Ensure privsep dir exists in the SAME exec so there's
            # no window for /run/sshd to disappear (seen on Ubuntu 24.04 pytorch
            # images where openssh-server postinst does not create /run/sshd —
            # it relies on systemd-tmpfiles which never runs in a container).
            sshd_start = subprocess.run(
                [
                    "docker", "exec", container_name, "sh", "-c",
                    "mkdir -p /run/sshd /var/run/sshd && "
                    "chmod 0755 /run/sshd /var/run/sshd && "
                    "/usr/sbin/sshd",
                ],
                capture_output=True,
                timeout=_remaining(8),
            )
            sshd_started = sshd_start.returncode == 0
            if sshd_started:
                log.info("sshd started in container %s", container_name)
                _note(
                    "SSH daemon ready — connections accepted"
                    if keys
                    else "SSH daemon started — password available on dashboard"
                )
            else:
                err = (
                    sshd_start.stderr.decode(errors="replace").strip()
                    if isinstance(sshd_start.stderr, bytes)
                    else (sshd_start.stderr or "").strip()
                )
                log.warning("sshd start failed in %s: %s", container_name, err)
                _note(f"SSH daemon failed to start: {err[:200]}", level="warning")
        else:
            log.info("sshd not found in image — web terminal only (container %s)", container_name)

        # --- Compose final summary based on observed state ---
        # When sshd isn't in the image, SSH keys do nothing — don't mention them.
        if sshd_started and keys:
            final_msg = f"[xcelsior] Terminal ready — SSH enabled ({len(keys)} key(s) + password)"
        elif sshd_started and root_password:
            final_msg = "[xcelsior] Terminal ready — SSH enabled (password on dashboard; add a key at Settings → SSH Keys for passwordless login)"
        elif sshd_started:
            final_msg = "[xcelsior] Terminal ready — SSH daemon up but no credentials set"
            final_level = "warning"
        elif sshd_present and not sshd_started:
            final_msg = "[xcelsior] Terminal ready — web terminal only (sshd failed to start)"
            final_level = "warning"
        else:
            final_msg = "[xcelsior] Terminal ready — web terminal only (image has no sshd)"

    except subprocess.TimeoutExpired:
        final_msg = "[xcelsior] SSH setup timed out — web terminal still works"
        final_level = "warning"
        log.warning("SSH inject exceeded wall-clock budget for job %s", job_id)
    except Exception as e:
        final_msg = f"[xcelsior] SSH setup failed: {e} — web terminal still works"
        final_level = "warning"
        log.warning("SSH key injection failed for job %s: %s", job_id, e)
    finally:
        elapsed = time.monotonic() - start
        summary = f"{final_msg} ({elapsed:.1f}s)"
        log.info("SSH inject for job %s completed in %.1fs: %s", job_id, elapsed, final_msg)
        # Guarantee: the UI always sees exactly one final summary line, via both
        # the log stream and the container's PID-1 stdout (for already-attached WS).
        _note(summary, level=final_level)
        _mirror_to_container(summary)
        # Report final SSH state back to the control plane so the dashboard
        # can surface a customer-facing notice when something went wrong
        # (no sshd installable, daemon failed to start, key fetch failed,
        # etc). Best-effort — never blocks or raises.
        try:
            # "ok" means the user can connect via at least one auth method
            # (key OR password). Without sshd_started neither works.
            ssh_ok = bool(sshd_started) and (len(keys) > 0 or bool(root_password))
            requests.post(
                _api_url(f"/agent/ssh-status/{job_id}"),
                headers=_api_headers(),
                json={
                    "ok": ssh_ok,
                    "sshd_present": bool(sshd_present),
                    "sshd_started": bool(sshd_started),
                    "key_count": len(keys),
                    "root_password": root_password if sshd_started else "",
                    "summary": final_msg,
                    "level": final_level,
                    "elapsed_sec": round(elapsed, 2),
                    "ts": time.time(),
                },
                timeout=5,
            )
        except Exception as e:
            log.debug("ssh-status report failed for job %s: %s", job_id, e)


def _shell_quote(s: str) -> str:
    """Shell-escape a string for use in sh -c."""
    return "'" + s.replace("'", "'\\''") + "'"


def _kill_container(container_name):
    """Force-kill and remove a container."""
    with suppress(Exception):
        subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            timeout=10,
        )
    _remove_container(container_name)


def _remove_container(container_name):
    """Remove a container (force). Falls back to removing by ID if name fails."""
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0 and "No such container" not in result.stderr:
            log.warning(
                "docker rm -f %s failed: %s — trying by ID", container_name, result.stderr.strip()
            )
            # Fallback: find container ID by name and remove by ID
            ps = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name=^/{container_name}$", "-q"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for cid in ps.stdout.strip().splitlines():
                if cid:
                    subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10)
                    log.info("Removed container %s by ID %s", container_name, cid)
    except Exception as e:
        log.warning("_remove_container(%s) error: %s", container_name, e)


def handle_preemptions(preempt_job_ids):
    """Handle preemption requests — gracefully stop containers."""
    for job_id in preempt_job_ids:
        with _active_lock:
            container_name = _active_containers.get(job_id)

        if not container_name:
            log.debug("Preemption for %s but no active container found", job_id)
            continue

        log.warning("PREEMPTING job %s (container %s)", job_id, container_name)

        # Send SIGTERM first (10s grace period), then SIGKILL
        try:
            subprocess.run(
                ["docker", "stop", "-t", "10", container_name],
                capture_output=True,
                timeout=20,
            )
        except Exception as e:
            log.warning("docker stop failed during preemption: %s", e)
            _kill_container(container_name)

        report_job_status(job_id, "preempted")

        with _active_lock:
            _active_containers.pop(job_id, None)

        log.info("Job %s preempted and cleaned up", job_id)


# ── Telemetry Push Thread ─────────────────────────────────────────────
# Per REPORT_FEATURE_2.md §3: "Enhance agent to report GPU metrics every 5s."
# Reuses security.py's get_gpu_telemetry() which delegates to NVML when available.
# Per REPORT_FEATURE_1.md: collect metrics at 10-15 second intervals.

TELEMETRY_INTERVAL = int(os.environ.get("XCELSIOR_TELEMETRY_INTERVAL", "10"))


def _sample_io_delta(state: dict) -> dict:
    """Return host-level net/disk throughput in Mbps / MB/s since the previous
    sample. ``state`` is a mutable dict carrying the previous counters between
    calls; first call returns zeros and primes the state.

    P2.4 — surfaces bandwidth + disk IO on the host dashboard. We sum across
    interfaces/disks (renter cares about aggregate throughput, not per-NIC).
    Falls back to zeros if psutil is unavailable or the counters are missing,
    so the telemetry pipeline never breaks.
    """
    if psutil is None:
        return {"net_rx_mbps": 0.0, "net_tx_mbps": 0.0, "disk_read_mb_s": 0.0, "disk_write_mb_s": 0.0}
    now = time.monotonic()
    out = {"net_rx_mbps": 0.0, "net_tx_mbps": 0.0, "disk_read_mb_s": 0.0, "disk_write_mb_s": 0.0}
    try:
        net = psutil.net_io_counters(pernic=False)
        disk = psutil.disk_io_counters(perdisk=False)
    except Exception:
        return out
    prev_t = state.get("t")
    prev_net = state.get("net")
    prev_disk = state.get("disk")
    state["t"] = now
    state["net"] = net
    state["disk"] = disk
    if prev_t is None or prev_net is None or prev_disk is None:
        return out
    dt = max(now - prev_t, 0.001)
    # Net counters are cumulative bytes. Mbps = megabits per second.
    try:
        rx_bytes = max(net.bytes_recv - prev_net.bytes_recv, 0)
        tx_bytes = max(net.bytes_sent - prev_net.bytes_sent, 0)
        out["net_rx_mbps"] = round((rx_bytes * 8) / dt / 1_000_000, 2)
        out["net_tx_mbps"] = round((tx_bytes * 8) / dt / 1_000_000, 2)
    except Exception:
        pass
    if disk is not None and prev_disk is not None:
        try:
            r_bytes = max(disk.read_bytes - prev_disk.read_bytes, 0)
            w_bytes = max(disk.write_bytes - prev_disk.write_bytes, 0)
            out["disk_read_mb_s"] = round(r_bytes / dt / 1_000_000, 2)
            out["disk_write_mb_s"] = round(w_bytes / dt / 1_000_000, 2)
        except Exception:
            pass
    return out


def telemetry_loop():
    """Background thread: push GPU metrics to scheduler every TELEMETRY_INTERVAL seconds.

    Per REPORT_FEATURE_1.md §Lightweight Monitoring with NVML:
    Collects GPU utilization, memory utilization (allocated + reserved for
    fragmentation detection), thermal data (current + rolling average),
    power consumption, and PCIe bandwidth at 10-15 second intervals.
    NFS health probe runs every ~60 seconds (every 6th iteration).
    """
    _nfs_probe_counter = 0
    _NFS_PROBE_EVERY = max(1, 60 // max(TELEMETRY_INTERVAL, 1))  # ~60s
    # P2.4 — prime last-sample state for bandwidth / disk IO delta math.
    _io_state: dict = {"t": None, "net": None, "disk": None}
    while not _shutdown.is_set():
        try:
            telemetry = get_gpu_telemetry()
            if telemetry:
                # Build compact metrics payload from first GPU (multi-GPU: aggregate)
                gpu = telemetry[0]
                metrics = {
                    "utilization": gpu.get("utilization", 0),
                    "memory_util": gpu.get("memory_util", 0),
                    "temp": gpu.get("temperature_c", 0),
                    "temp_avg": gpu.get("temperature_avg_c", gpu.get("temperature_c", 0)),
                    "power_draw_w": gpu.get("power_draw_w", 0),
                    "power_limit_w": gpu.get("power_limit_w", 0),
                    "pcie_gen": gpu.get("pcie_gen", ""),
                    "pcie_width": gpu.get("pcie_width", ""),
                    "pcie_tx_mb_s": gpu.get("pcie_tx_mb_s", 0),
                    "pcie_rx_mb_s": gpu.get("pcie_rx_mb_s", 0),
                    "gpu_count": len(telemetry),
                    # Memory fragmentation detection (REPORT_FEATURE_1.md)
                    "memory_total_gb": gpu.get("memory_total_gb", 0),
                    "memory_used_gb": gpu.get("memory_used_gb", 0),
                    "memory_free_gb": gpu.get("memory_free_gb", 0),
                    "torch_memory_allocated_bytes": gpu.get("torch_memory_allocated_bytes", 0),
                    "torch_memory_reserved_bytes": gpu.get("torch_memory_reserved_bytes", 0),
                    "memory_fragmentation_pct": gpu.get("memory_fragmentation_pct", 0.0),
                }

                # ECC memory errors
                metrics["memory_errors"] = gpu.get("memory_errors", 0)

                # P2.4 — host bandwidth + disk IO deltas (aggregate across NICs/disks)
                metrics.update(_sample_io_delta(_io_state))

                # Active container count
                with _active_lock:
                    metrics["active_jobs"] = len(_active_containers)

                # Volume mount health — only probe every ~60s to avoid overhead
                _nfs_probe_counter += 1
                if _nfs_probe_counter >= _NFS_PROBE_EVERY:
                    _nfs_probe_counter = 0
                    try:
                        vol_health = []
                        nfs_healthy = True
                        if os.path.isdir(VOLUME_BASE_DIR):
                            for entry in os.listdir(VOLUME_BASE_DIR):
                                if entry.endswith(".img"):
                                    continue
                                mp = os.path.join(VOLUME_BASE_DIR, entry)
                                # Skip symlinks to prevent information disclosure
                                if os.path.islink(mp):
                                    continue
                                if not os.path.isdir(mp):
                                    continue
                                try:
                                    # Use a subprocess with timeout to avoid hanging Python
                                    # if the NFS mount is stale. With hard mounts, stat will
                                    # block indefinitely on an unreachable server, so the 3s
                                    # timeout is the critical safety guard here.
                                    stat_result = subprocess.run(
                                        ["stat", "-f", "--format=%b %f %S", mp],
                                        capture_output=True,
                                        text=True,
                                        timeout=3,
                                    )
                                    if stat_result.returncode == 0:
                                        parts = stat_result.stdout.strip().split()
                                        blocks, bfree, bsize = (
                                            int(parts[0]),
                                            int(parts[1]),
                                            int(parts[2]),
                                        )
                                        total = (blocks * bsize) / (1024**3)
                                        used = ((blocks - bfree) * bsize) / (1024**3)
                                        vol_health.append(
                                            {
                                                "volume_id": entry,
                                                "mounted": True,
                                                "total_gb": round(total, 2),
                                                "used_gb": round(used, 2),
                                            }
                                        )
                                    else:
                                        vol_health.append(
                                            {
                                                "volume_id": entry,
                                                "mounted": False,
                                                "total_gb": 0,
                                                "used_gb": 0,
                                            }
                                        )
                                        nfs_healthy = False
                                except (subprocess.TimeoutExpired, OSError):
                                    vol_health.append(
                                        {
                                            "volume_id": entry,
                                            "mounted": False,
                                            "total_gb": 0,
                                            "used_gb": 0,
                                        }
                                    )
                                    nfs_healthy = False
                        if vol_health:
                            metrics["volume_health"] = vol_health
                        metrics["nfs_healthy"] = nfs_healthy
                    except Exception:
                        pass  # Non-critical; don't let volume scan break telemetry

                report_telemetry(metrics)
        except Exception as e:
            log.debug("Telemetry loop error: %s", e)

        # Interruptible sleep
        for _ in range(TELEMETRY_INTERVAL):
            if _shutdown.is_set():
                return
            time.sleep(1)


# ── Mining Detection Thread ──────────────────────────────────────────


def mining_detection_loop():
    """Background thread: periodically check for mining signatures."""
    consecutive_mining_detections = 0

    while not _shutdown.is_set():
        # Wait for the check interval (interruptible)
        for _ in range(MINING_CHECK_INTERVAL):
            if _shutdown.is_set():
                return
            time.sleep(1)

        telemetry = get_gpu_telemetry()
        for gpu in telemetry:
            is_mining, confidence, reason = check_mining_heuristic(gpu)
            if is_mining:
                consecutive_mining_detections += 1
                log.warning(
                    "MINING DETECTED GPU=%d confidence=%.0f%% (%s) [%d consecutive]",
                    gpu["index"],
                    confidence * 100,
                    reason,
                    consecutive_mining_detections,
                )
                report_mining_alert(gpu["index"], confidence, reason)

                # After sustained detection, consider killing the container
                if consecutive_mining_detections >= 5:
                    log.error("Sustained mining detected — killing suspicious containers")
                    with _active_lock:
                        for job_id, cname in list(_active_containers.items()):
                            _kill_container(cname)
                            report_job_status(job_id, "failed")
                        _active_containers.clear()
                    consecutive_mining_detections = 0
            else:
                consecutive_mining_detections = max(0, consecutive_mining_detections - 1)


# ── Heartbeat Thread ─────────────────────────────────────────────────


def heartbeat_loop(host_ip, compute_score=None):
    """Background thread: periodic heartbeat/registration updates."""
    consecutive_failures = 0

    while not _shutdown.is_set():
        try:
            gpu_info = get_gpu_info()
            success = heartbeat(gpu_info, host_ip, compute_score)

            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    log.error(
                        "Too many heartbeat failures (%d) — shutting down", consecutive_failures
                    )
                    _shutdown.set()
                    return
        except RuntimeError as e:
            log.error("Heartbeat GPU query failed: %s", e)
            consecutive_failures += 1

        # Interruptible sleep
        for _ in range(HEARTBEAT_INTERVAL):
            if _shutdown.is_set():
                return
            time.sleep(1)


# ── Graceful Shutdown ────────────────────────────────────────────────


def graceful_shutdown():
    """Stop agent-started containers and deregister from scheduler.

    Adopted containers (started by the scheduler via SSH) are left running
    so they survive agent restarts.
    """
    log.info("Graceful shutdown — stopping %d active containers", len(_active_containers))

    with _active_lock:
        for job_id, container_name in list(_active_containers.items()):
            if job_id in _adopted_containers:
                log.info("Leaving adopted container %s (job %s) running", container_name, job_id)
                continue
            log.info("Stopping container %s (job %s)", container_name, job_id)
            try:
                subprocess.run(
                    ["docker", "stop", "-t", "15", container_name],
                    capture_output=True,
                    timeout=25,
                )
            except Exception as e:
                log.warning("docker stop failed during shutdown: %s", e)
                _kill_container(container_name)

            # Mark job as queued so it can be rescheduled
            report_job_status(job_id, "queued")
        _active_containers.clear()

    # Clean up orphaned volume mounts after containers are stopped
    try:
        cleanup_orphaned_volume_mounts()
    except Exception as e:
        log.warning("Volume cleanup during shutdown failed: %s", e)

    # Deregister from scheduler
    try:
        requests.delete(
            _api_url(f"/host/{HOST_ID}"),
            headers=_api_headers(),
            timeout=10,
        )
        log.info("Deregistered from scheduler")
    except requests.RequestException:
        log.warning("Failed to deregister from scheduler")


# ── Main Loop ─────────────────────────────────────────────────────────


def validate_config():
    """Validate required configuration."""
    errors = []
    if not HOST_ID:
        errors.append("XCELSIOR_HOST_ID is not set")
    if not SCHEDULER_URL:
        errors.append("XCELSIOR_SCHEDULER_URL is not set")
    if errors:
        log.error("Configuration errors:")
        for e in errors:
            log.error("  - %s", e)
        sys.exit(1)


def print_startup_banner(gpu_info, host_ip, admitted, runtime):
    """Print startup banner with configuration info."""
    auth_mode = "none"
    if _oauth_client_credentials_enabled():
        auth_mode = f"oauth-client ({OAUTH_CLIENT_ID})"
    elif API_TOKEN:
        auth_mode = "api-token"
    log.info("=" * 64)
    log.info("  Xcelsior Worker Agent v%s (pull-based)", VERSION)
    log.info("=" * 64)
    log.info("  Host ID:        %s", HOST_ID)
    log.info("  Scheduler URL:  %s", SCHEDULER_URL)
    log.info("  Host IP:        %s", host_ip)
    log.info(
        "  GPU:            %s (%.1f GB VRAM)", gpu_info["gpu_model"], gpu_info["total_vram_gb"]
    )
    log.info("  Cost/hour:      $%.2f", COST_PER_HOUR)
    log.info("  Poll interval:  %ds", POLL_INTERVAL)
    log.info("  Heartbeat:      %ds", HEARTBEAT_INTERVAL)
    log.info("  Runtime:        %s", runtime)
    log.info("  Auth:           %s", auth_mode)
    log.info("  Admitted:       %s", "YES" if admitted else "NO (limited to heartbeats)")
    log.info("  Headscale:      %s", "enabled" if TAILSCALE_ENABLED else "disabled")
    log.info("=" * 64)


def adopt_running_containers():
    """Discover containers started by the scheduler (via SSH) and adopt them.

    The scheduler starts containers named 'xcl-{job_id}' via SSH.
    If the worker agent wasn't running at the time, it needs to discover
    and adopt these containers to provide log forwarding and monitoring.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=xcl-", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return

        containers = [name.strip() for name in result.stdout.strip().split("\n") if name.strip()]
        if not containers:
            return

        for container_name in containers:
            # Extract job_id from container name (xcl-{job_id})
            if not container_name.startswith("xcl-"):
                continue
            job_id = container_name[4:]  # Remove "xcl-" prefix

            with _active_lock:
                if job_id in _active_containers:
                    continue  # Already tracking this container
                _active_containers[job_id] = container_name
                _adopted_containers.add(job_id)

            log.info("Adopting running container %s (job %s)", container_name, job_id)

            # Reclaim the lease for this running job. The server now accepts
            # claim requests for jobs in leased/starting/running state if the
            # host matches, so adopted containers can keep their lease fresh
            # across worker restarts.
            try:
                lease_info = claim_lease(job_id)
            except Exception as e:
                log.debug("Lease reclaim on adoption failed for %s: %s", job_id, e)
                lease_info = None

            # Start log forwarding
            log_fwd = LogForwarder(job_id, container_name)
            log_fwd.start()

            # Start lease renewal loop if we have a live lease
            if lease_info:
                lease_interval = int(lease_info.get("duration_sec", 300)) // 2
                lease_stop = threading.Event()

                def _adopt_renew(jid=job_id, interval=lease_interval, stop=lease_stop):
                    fails = 0
                    while not stop.is_set() and not _shutdown.is_set():
                        stop.wait(interval)
                        if stop.is_set() or _shutdown.is_set():
                            return
                        with _active_lock:
                            if jid not in _active_containers:
                                return  # Container gone
                        res = renew_lease(jid)
                        if not res:
                            fails += 1
                            log.warning(
                                "Adopted-lease renewal failed for job %s — attempt %d",
                                jid,
                                fails,
                            )
                            if fails >= 3:
                                new = claim_lease(jid)
                                if new:
                                    fails = 0
                                    log.info("Adopted-lease reclaimed for job %s", jid)
                                else:
                                    log.error(
                                        "Adopted-lease reclaim failed for job %s — stopping",
                                        jid,
                                    )
                                    return
                        else:
                            fails = 0

                renew_thread = threading.Thread(
                    target=_adopt_renew,
                    name=f"adopt-lease-{job_id[:8]}",
                    daemon=True,
                )
                renew_thread.start()
            else:
                log.warning(
                    "Adopted container %s has no active lease — running unmonitored",
                    job_id,
                )

            # Start monitoring in a separate thread
            monitor_thread = threading.Thread(
                target=_monitor_interactive,
                args=(job_id, container_name),
                kwargs={"log_forwarder": log_fwd},
                name=f"adopt-{job_id[:8]}",
                daemon=True,
            )
            monitor_thread.start()

        log.info("Adopted %d running container(s)", len(containers))
    except Exception as e:
        log.warning("Container adoption failed: %s", e)


def reinject_shells_for_running():
    """Re-run `_inject_ssh_keys` for every running container on this host.

    Rationale: MOTD / PS1 / sshd-config logic in `_inject_ssh_keys` evolves
    over time. Existing containers started with an older version don't get
    the new shell setup, so SSH sessions land in a bare prompt. On every
    worker_agent boot, re-apply the injection (idempotent) to all containers
    currently in state=running. Each call exec's into the container to write
    files; safe to run repeatedly.

    Guardrails:
      - Gated by env `WORKER_AGENT_AUTO_REINJECT` (default "true"); set to
        anything else to disable (e.g. during debugging).
      - Rate-limited to 1 container per 2s so a host with many containers
        coming back online doesn't saturate the docker daemon.
      - Skips containers whose actual docker state is not `running` (caller
        list might include briefly stopped/exiting containers).
      - Each attempt is logged with container name, outcome, and duration.
    """
    if os.environ.get("WORKER_AGENT_AUTO_REINJECT", "true").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        log.info("Shell re-injection disabled via WORKER_AGENT_AUTO_REINJECT")
        return

    try:
        with _active_lock:
            candidates = list(_active_containers.items())  # [(job_id, container_name), ...]
    except Exception:
        candidates = []

    if not candidates:
        return

    log.info("Re-injecting shell setup for %d adopted container(s)", len(candidates))
    stats = {"success": 0, "failure": 0, "skipped": 0}

    for idx, (job_id, container_name) in enumerate(candidates):
        if _shutdown.is_set():
            break

        # Rate-limit between containers (but not before the first)
        if idx > 0:
            time.sleep(2.0)

        t0 = time.monotonic()
        try:
            # Verify container is actually running in docker's view
            inspect = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if inspect.returncode != 0 or inspect.stdout.strip() != "true":
                stats["skipped"] += 1
                log.debug("Skipping %s — not in running state", container_name)
                _motd_reinjection_total.labels(result="skipped").inc()
                continue

            _inject_ssh_keys(job_id, container_name)
            dur = time.monotonic() - t0
            stats["success"] += 1
            _motd_reinjection_total.labels(result="success").inc()
            log.info(
                "Re-injected shell for container=%s job=%s duration=%.2fs",
                container_name,
                job_id[:8],
                dur,
            )
        except Exception as e:
            dur = time.monotonic() - t0
            stats["failure"] += 1
            _motd_reinjection_total.labels(result="failure").inc()
            log.warning(
                "Re-inject failed container=%s job=%s duration=%.2fs err=%s",
                container_name,
                job_id[:8],
                dur,
                e,
            )

    log.info(
        "Shell re-injection complete: %d success, %d failure, %d skipped",
        stats["success"],
        stats["failure"],
        stats["skipped"],
    )


def cleanup_orphaned_volume_mounts():
    """Clean up NFS mounts for managed volumes that have no running container.

    On agent restart (or periodically), there may be stale mounts under
    /mnt/xcelsior-volumes/ whose containers have exited. This function
    inspects each running container's bind mounts to determine which volumes
    are actually in use, and unmounts the rest.
    """
    vol_mount_base = "/mnt/xcelsior-volumes"
    if not os.path.isdir(vol_mount_base):
        return

    # Build set of volume mount paths actually used by running containers
    active_mounts: set[str] = set()
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=xcl-", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for cid in result.stdout.strip().split("\n"):
                cid = cid.strip()
                if not cid:
                    continue
                # Inspect container to get bind mount sources
                insp = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        '{{range .Mounts}}{{if eq .Type "bind"}}{{.Source}}{{" "}}{{end}}{{end}}',
                        cid,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if insp.returncode == 0:
                    for src in insp.stdout.strip().split():
                        src = src.strip()
                        if src.startswith(vol_mount_base):
                            active_mounts.add(src)
    except Exception:
        return  # Can't determine running containers — don't clean up blindly

    # Get mounted volume dirs
    try:
        mounted_vols = os.listdir(vol_mount_base)
    except OSError:
        return

    cleaned = 0
    for vol_dir in mounted_vols:
        mount_path = os.path.join(vol_mount_base, vol_dir)
        if mount_path in active_mounts:
            continue  # Volume is in use by a running container

        # Check if it's actually a mountpoint
        try:
            check = subprocess.run(
                ["mountpoint", "-q", mount_path],
                capture_output=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            log.warning(
                "mountpoint check timed out for %s — skipping (possible stale NFS)", mount_path
            )
            continue
        if check.returncode != 0:
            continue  # Not mounted, skip

        _unmount_nfs(mount_path)
        cleaned += 1
        log.info("Cleaned up orphaned volume mount: %s", mount_path)

    if cleaned:
        log.info("Cleaned up %d orphaned volume mount(s)", cleaned)


def main():
    """Main entry point — pull-based worker agent."""
    validate_config()

    # ── Step 0: Initialize NVML (REPORT_FEATURE_1.md §Lightweight Monitoring) ──
    nvml_ok = nvml_init()
    if nvml_ok:
        log.info("NVML telemetry active — using pynvml bindings (no nvidia-smi overhead)")
    else:
        log.info("NVML not available — falling back to nvidia-smi subprocess")

    # ── Step 1: Tailscale / Headscale setup ──
    if TAILSCALE_ENABLED:
        setup_tailscale()

    # ── Step 2: Detect GPU ──
    try:
        gpu_info = get_gpu_info()
    except RuntimeError as e:
        log.error("GPU detection failed: %s", e)
        log.error("Make sure nvidia-smi is installed and NVIDIA drivers are loaded.")
        nvml_shutdown()
        sys.exit(1)

    host_ip = get_host_ip()

    # ── Step 2b: gVisor auto-install ──
    if PREFER_GVISOR:
        from security import is_gvisor_available

        if not is_gvisor_available():
            log.info("gVisor preferred but not installed — attempting auto-install...")
            success, msg = install_gvisor(enable_nvproxy=True)
            if success:
                log.info("gVisor auto-install: %s", msg)
            else:
                log.warning("gVisor auto-install failed: %s — falling back to runc", msg)
        else:
            log.info("gVisor (runsc) already available")

    # ── Step 3: Node admission control ──
    versions = get_local_versions()
    admitted, admission_details = admit_node(HOST_ID, versions, gpu_info["gpu_model"])
    runtime_name = admission_details.get("recommended_runtime", "runc")

    # Report versions to scheduler (best-effort)
    report_versions(versions)

    if not admitted:
        log.warning(
            "Node NOT ADMITTED — reasons: %s",
            "; ".join(admission_details.get("rejection_reasons", ["unknown"])),
        )
        log.warning("Agent will continue heartbeats but will NOT accept work.")
        log.warning("Upgrade vulnerable components to become eligible.")

    # ── Step 4: Comprehensive benchmark ──
    compute_score = None
    log.info("Running comprehensive benchmark (GPU + network, ~60 seconds)...")
    bench = run_compute_benchmark()
    if bench:
        compute_score = bench.get("tflops", 0) / 10  # XCU
        log.info("GPU Benchmark: %.1f TFLOPS = %.1f XCU", bench["tflops"], compute_score)
        log.info(
            "  CUDA %s | Driver %s | CC %s",
            bench.get("cuda_version", "?"),
            bench.get("driver_version", "?"),
            bench.get("compute_capability", "?"),
        )
        log.info(
            "  PCIe: %.2f GB/s | Peak Temp: %s°C",
            bench.get("pcie_bandwidth_gbps", 0),
            bench.get("gpu_temp_celsius", "?"),
        )
        report_benchmark(bench, gpu_info["gpu_model"])

        # Network benchmark (ping + throughput to scheduler)
        log.info("Running network quality benchmark...")
        net_bench = run_network_benchmark()
        if net_bench:
            log.info(
                "  Latency: %.1fms avg | Jitter: %.1fms | Loss: %.1f%% | Throughput: %.1f Mbps",
                net_bench.get("latency_avg_ms", 0),
                net_bench.get("jitter_ms", 0),
                net_bench.get("packet_loss_pct", 0),
                net_bench.get("throughput_mbps", 0),
            )
            bench.update(net_bench)

        # Build full verification report and submit
        # Merge benchmark data with security versions for complete check
        full_report = {**bench}
        full_report["claimed_gpu_model"] = gpu_info["gpu_model"]
        full_report["claimed_vram_gb"] = gpu_info["total_vram_gb"]
        full_report["versions"] = versions
        report_verification(full_report)
    else:
        log.info("Benchmark skipped (PyTorch/CUDA not available)")

    # ── Step 5: Initial heartbeat ──
    heartbeat(gpu_info, host_ip, compute_score)

    # ── Banner ──
    print_startup_banner(gpu_info, host_ip, admitted, runtime_name)

    # ── Step 6: Initialize image cache ──
    cache_init()
    log.info(
        "Image cache initialized (%d images, %.0f MB)",
        len(_image_cache_index),
        _total_cache_size_mb(),
    )

    # ── Step 6b: Startup image warmer ──
    # Like RunPod/Lambda — pre-pull ALL catalog images on boot so instances
    # start in seconds, not minutes. Runs on a background thread so it doesn't
    # block the main loop.
    def _startup_image_warmer():
        """Pre-pull entire IMAGE_TEMPLATES catalog on startup."""
        try:
            from security import IMAGE_TEMPLATES

            catalog_images = [t["image"] for t in IMAGE_TEMPLATES]
            # Also fetch popular images from API (covers user-custom images)
            api_popular = fetch_popular_images()
            # Merge: catalog first (guaranteed), then API popular (de-duped)
            seen = set(catalog_images)
            for img in api_popular:
                if img not in seen:
                    catalog_images.append(img)
                    seen.add(img)
            # Filter to only those not already cached locally
            to_pull = []
            for img in catalog_images:
                with _image_cache_lock:
                    if img in _image_cache_index:
                        continue
                to_pull.append(img)
            if not to_pull:
                log.info("Image warmer: all %d catalog images already cached", len(catalog_images))
                return
            log.info(
                "Image warmer: pre-pulling %d/%d images in background...",
                len(to_pull),
                len(catalog_images),
            )
            cache_prepull_popular(to_pull, max_concurrent=2)
            log.info("Image warmer: done — all catalog images cached")
        except Exception as e:
            log.warning("Image warmer error: %s", e)

    warmer_thread = threading.Thread(
        target=_startup_image_warmer,
        name="image-warmer",
        daemon=True,
    )
    warmer_thread.start()

    # ── Step 7: Adopt containers started by scheduler (before agent was running) ──
    adopt_running_containers()

    # ── Step 7a: Re-apply shell setup (MOTD/PS1/sshd keepalive) to adopted containers ──
    # The injection logic evolves; existing containers started under older agent
    # versions otherwise stay on a bare `#` prompt indefinitely. Idempotent, gated
    # by WORKER_AGENT_AUTO_REINJECT, rate-limited to 1 container per 2s.
    try:
        reinject_shells_for_running()
    except Exception as e:
        log.warning("reinject_shells_for_running failed: %s", e)

    # ── Step 7b: Clean up orphaned volume mounts ──
    cleanup_orphaned_volume_mounts()

    # ── Step 8: Start background threads ──
    threads = []

    # Heartbeat thread
    hb_thread = threading.Thread(
        target=heartbeat_loop,
        args=(host_ip, compute_score),
        name="heartbeat",
        daemon=True,
    )
    hb_thread.start()
    threads.append(hb_thread)

    # Telemetry push thread (GPU metrics every 5s)
    telem_thread = threading.Thread(
        target=telemetry_loop,
        name="telemetry",
        daemon=True,
    )
    telem_thread.start()
    threads.append(telem_thread)
    log.info("Telemetry push started (every %ds)", TELEMETRY_INTERVAL)

    # Mining detection thread
    mining_thread = threading.Thread(
        target=mining_detection_loop,
        name="mining-detection",
        daemon=True,
    )
    mining_thread.start()
    threads.append(mining_thread)

    # Periodic volume orphan cleanup thread (every 30 min)
    def _volume_orphan_cleanup_loop():
        """Periodically clean up NFS mounts for containers that have exited."""
        while not _shutdown.is_set():
            for _ in range(1800):  # 30 minutes, interruptible
                if _shutdown.is_set():
                    return
                time.sleep(1)
            try:
                cleanup_orphaned_volume_mounts()
            except Exception as e:
                log.debug("Periodic volume cleanup error: %s", e)

    vol_cleanup_thread = threading.Thread(
        target=_volume_orphan_cleanup_loop,
        name="volume-orphan-cleanup",
        daemon=True,
    )
    vol_cleanup_thread.start()
    threads.append(vol_cleanup_thread)

    # ── Step 8: Main polling loop ──
    log.info("Entering main polling loop...")
    consecutive_poll_failures = 0
    _last_prepull_time = 0
    PREPULL_INTERVAL = 60  # Pre-pull popular images every 60s when idle

    while not _shutdown.is_set():
        try:
            # Check for preemptions
            preempt_jobs = check_preemption()
            if preempt_jobs:
                handle_preemptions(preempt_jobs)

            # Drain admin control-plane commands (reinject_shell, etc.)
            drain_agent_commands()

            # Poll for new work (only if admitted)
            if admitted:
                jobs = poll_for_work()
                for job in jobs:
                    job_id = job.get("job_id", "?")
                    log.info("Received job: %s (%s)", job_id, job.get("name", "?"))

                    # Run job in a separate thread so we can continue polling
                    job_thread = threading.Thread(
                        target=run_job,
                        args=(job,),
                        name=f"job-{job_id[:8]}",
                        daemon=True,
                    )
                    job_thread.start()
                    threads.append(job_thread)

                # Idle-time pre-pulling: if no active containers and enough
                # time has passed, pre-pull popular images for faster cold starts
                if not jobs and len(_active_containers) == 0:
                    now = time.time()
                    if now - _last_prepull_time >= PREPULL_INTERVAL:
                        _last_prepull_time = now
                        popular = fetch_popular_images()
                        if popular:
                            log.info("Idle — pre-pulling %d popular images", len(popular))
                            cache_prepull_popular(popular)

            consecutive_poll_failures = 0

        except Exception as e:
            consecutive_poll_failures += 1
            log.error("Poll loop error: %s", e, exc_info=True)

            if consecutive_poll_failures > MAX_CONSECUTIVE_FAILURES:
                log.error("Too many poll failures — shutting down")
                _shutdown.set()
                break

            # Exponential backoff on repeated failures
            backoff = min(2**consecutive_poll_failures, 300)
            log.warning("Backing off %ds before next poll", backoff)
            for _ in range(backoff):
                if _shutdown.is_set():
                    break
                time.sleep(1)
            continue

        # Normal poll interval (interruptible)
        for _ in range(POLL_INTERVAL):
            if _shutdown.is_set():
                break
            time.sleep(1)

    # ── Shutdown ──
    graceful_shutdown()
    nvml_shutdown()
    log.info("Worker agent stopped.")


if __name__ == "__main__":
    main()
