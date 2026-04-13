#!/usr/bin/env python3
# Xcelsior Worker Agent v2.0.0
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
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from datetime import datetime

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Install with: pip install requests")
    sys.exit(1)

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
OAUTH_TOKEN_REFRESH_SKEW_SEC = int(
    os.environ.get("XCELSIOR_OAUTH_TOKEN_REFRESH_SKEW_SEC", "60")
)

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

VERSION = "2.0.0"

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("xcelsior-worker")

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
            "gpu_model": parts[0],
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

    # If Tailscale is active, prefer the Tailscale IP
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
                    "Tailscale already connected (IP: %s)", status.get("TailscaleIPs", ["?"])[0]
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
            log.info("Tailscale connected successfully")
            return True
        else:
            log.warning("Tailscale up failed: %s", r.stderr.strip())
            return False

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning("Tailscale setup failed: %s", e)
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


def report_job_status(job_id, status, host_id=None, container_id=None, container_name=None,
                      ssh_port=None, interactive=None):
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

IMAGE_CACHE_MAX_GB = float(os.environ.get("XCELSIOR_IMAGE_CACHE_MAX_GB", "50"))
IMAGE_CACHE_EVICT_LOW_GB = float(os.environ.get("XCELSIOR_IMAGE_CACHE_EVICT_LOW_GB", "40"))

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
    """Initialize cache index from locally available Docker images."""
    images = _get_local_images()
    with _image_cache_lock:
        for tag, size_mb in images.items():
            if tag not in _image_cache_index:
                _image_cache_index[tag] = {
                    "last_used": time.time(),
                    "size_mb": size_mb,
                    "pull_count": 0,
                }


def cache_prepull_popular(popular_images):
    """Pre-pull popular images during idle time.

    Called from the main loop when the agent has no active work.
    Only pulls images not already cached, respecting cache limits.

    Args:
        popular_images: List of image tags ordered by popularity.
    """
    for image_tag in popular_images:
        with _image_cache_lock:
            if image_tag in _image_cache_index:
                continue  # Already cached

        total_mb = _total_cache_size_mb()
        if total_mb >= IMAGE_CACHE_MAX_GB * 1024:
            log.debug("Cache full — skipping pre-pull of %s", image_tag)
            break

        log.info("Pre-pulling popular image: %s", image_tag)
        try:
            pull = subprocess.run(
                ["docker", "pull", image_tag],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if pull.returncode == 0:
                # Get size of pulled image
                images = _get_local_images()
                size_mb = images.get(image_tag, 0)
                cache_track_pull(image_tag, size_mb)
                log.info("Pre-pulled %s (%.0f MB)", image_tag, size_mb)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass


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
        mount_cmd = [
            "mount",
            "-t",
            "nfs",
            "-o",
            os.environ.get("XCELSIOR_NFS_MOUNT_OPTS",
                           "soft,timeo=150,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp"),
            f"{server}:{path}",
            mount_point,
        ]
        r = subprocess.run(mount_cmd, capture_output=True, text=True, timeout=30)
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
    return total / (1024 ** 3)


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
                "huggingface-cli", "download",
                model_id,
                "--revision", revision,
                "--local-dir", dest,
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
            check=True, capture_output=True, text=True, timeout=30,
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
                "cryptsetup", "luksFormat",
                "--batch-mode",
                "--type", "luks2",
                "--key-file", key_file,
                "--cipher", "aes-xts-plain64",
                "--key-size", "512",
                "--hash", "sha256",
                backing_file,
            ],
            check=True, capture_output=True, text=True, timeout=60,
        )

        # 4. Open LUKS device
        subprocess.run(
            [
                "cryptsetup", "luksOpen",
                "--key-file", key_file,
                backing_file, mapper_name,
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )

        # 5. Create ext4 filesystem
        dm_path = f"/dev/mapper/{mapper_name}"
        subprocess.run(
            ["mkfs.ext4", "-q", "-L", f"vol-{volume_id[:8]}", dm_path],
            check=True, capture_output=True, text=True, timeout=120,
        )

        # 6. Mount
        os.makedirs(mount_point, exist_ok=True)
        subprocess.run(
            ["mount", dm_path, mount_point],
            check=True, capture_output=True, text=True, timeout=30,
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
                check=True, capture_output=True, text=True, timeout=30,
            )

        # Mount if not already mounted
        os.makedirs(mount_point, exist_ok=True)
        r = subprocess.run(["mountpoint", "-q", mount_point], capture_output=True, timeout=5)
        if r.returncode != 0:
            subprocess.run(
                ["mount", dm_path, mount_point],
                check=True, capture_output=True, text=True, timeout=30,
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
                check=True, capture_output=True, text=True, timeout=30,
            )

        # Close LUKS device
        if os.path.exists(dm_path):
            subprocess.run(
                ["cryptsetup", "luksClose", mapper_name],
                check=True, capture_output=True, text=True, timeout=30,
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
            subprocess.run(
                ["shred", "-u", "-z", "-n", "3", key_file],
                capture_output=True, text=True, timeout=30,
            )
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
            subprocess.run(["cryptsetup", "luksClose", mapper_name], capture_output=True, timeout=10)
    with suppress(Exception):
        if os.path.exists(key_file):
            os.remove(key_file)
    with suppress(Exception):
        if os.path.exists(backing_file):
            os.remove(backing_file)
    with suppress(Exception):
        if os.path.isdir(mount_point):
            os.rmdir(mount_point)


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
    env_vars = job.get("environment", {})
    volumes = job.get("volumes", [])

    # NFS configuration — from job dict or env vars
    nfs_server = job.get("nfs_server") or os.environ.get("XCELSIOR_NFS_SERVER", "")
    nfs_path = job.get("nfs_path") or os.environ.get("XCELSIOR_NFS_PATH", "")
    nfs_mount_point = job.get("nfs_mount_point") or os.environ.get(
        "XCELSIOR_NFS_MOUNT", "/mnt/xcelsior-nfs"
    )
    nfs_mounted = False
    log_fwd = None

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

    # Start lease renewal thread
    _lease_stop = threading.Event()

    def _lease_renewal_loop():
        while not _lease_stop.is_set() and not _shutdown.is_set():
            _lease_stop.wait(lease_interval)
            if _lease_stop.is_set() or _shutdown.is_set():
                break
            result = renew_lease(job_id)
            if not result:
                log.warning("Lease renewal failed for job %s — lease may have expired", job_id)

    lease_thread = threading.Thread(
        target=_lease_renewal_loop,
        name=f"lease-{job_id[:8]}",
        daemon=True,
    )
    lease_thread.start()

    encrypted_vol_ids = []
    managed_vol_mounts = []  # /mnt/xcelsior-volumes/{vid} paths for cleanup
    current_stage = "preparing instance"
    try:
        # 0. Mount NFS volume (if configured)
        if nfs_server and nfs_path:
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
                    log.info("Managed volume %s mounted: %s → %s", vid, vol_host_mount, container_path)
                else:
                    log.warning("Managed volume %s mount failed — skipping", vid)
            else:
                log.warning("Managed volume %s: NFS server not configured — skipping", vid)
            _vol_idx += 1

        # 1. Pull image (with cache tracking + LRU eviction)
        current_stage = "pulling image"
        cache_evict_lru(exclude_images={image})  # Evict before pulling — protect job's image
        log.info("Pulling image %s...", image)
        _push_log_lines(job_id, [{"message": f"Pulling image {image}…", "level": "info", "timestamp": time.time()}])

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
            _push_log_lines(job_id, [{"message": f"Image pull failed (exit {pull_rc})", "level": "error", "timestamp": time.time()}])
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
        ssh_port = int(job.get("ssh_port", 22) or 22)

        # Interactive mode: override entrypoint to keep container alive
        # and expose SSH port for user access
        extra_docker_args = []
        effective_command = command
        if is_interactive:
            log.info("INTERACTIVE MODE for job %s — container will stay running", job_id)
            # Override entrypoint to keep container alive indefinitely
            extra_docker_args.extend(["--entrypoint", "sleep"])
            effective_command = ["infinity"]
            # Expose SSH port (map to a unique host port based on job_id hash)
            host_port = 10000 + (int(job_id[:4], 16) % 55000)
            extra_docker_args.extend(["-p", f"{host_port}:{ssh_port}"])
            # Interactive containers need writable filesystem for SSH
            # Remove --read-only and add writable tmpfs for home/ssh
            extra_docker_args.extend([
                "--tmpfs", "/home:rw,size=1g",
                "--tmpfs", "/run:rw,size=64m",
                "--tmpfs", "/var/log:rw,size=256m",
            ])
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
        _push_log_lines(job_id, [{"message": "Pull complete — starting container…", "level": "info", "timestamp": time.time()}])
        start = subprocess.run(
            docker_args,
            capture_output=True,
            text=True,
            timeout=CONTAINER_START_TIMEOUT,
        )
        if start.returncode != 0:
            log.error("Container start failed: %s", start.stderr.strip())
            _push_log_lines(job_id, [{"message": f"Container start failed: {start.stderr.strip()}", "level": "error", "timestamp": time.time()}])
            report_job_status(job_id, "failed")
            return

        container_id = start.stdout.strip()[:12]
        log.info("Container started: %s", container_id)
        _push_log_lines(job_id, [{"message": f"Container started ({container_id})", "level": "info", "timestamp": time.time()}])

        # Report container info to scheduler
        report_job_status(job_id, "running", host_id=HOST_ID,
                          container_id=container_id, container_name=container_name)

        # For interactive jobs, report SSH connection info
        if is_interactive:
            report_job_status(job_id, "running", ssh_port=host_port, interactive=True)
            log.info("Interactive instance %s ready — SSH port %d", job_id, host_port)

        # 4b. Inject user's SSH keys into the container (best-effort)
        _inject_ssh_keys(job_id, container_name)

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
        _push_log_lines(job_id, [{
            "message": f"Timed out while {current_stage}",
            "level": "error",
            "timestamp": time.time(),
        }])
        _kill_container(container_name)
        report_job_status(job_id, "failed")
        release_lease(job_id, "failed")
    except Exception as e:
        log.error("Job %s failed: %s", job_id, e, exc_info=True)
        _kill_container(container_name)
        report_job_status(job_id, "failed")
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


# ── Log Forwarding ───────────────────────────────────────────────────
# Streams container stdout/stderr to the API via HTTP POST batches.
# Disk-backed buffer survives transient API downtime.
# Gzip compression on batches > 1KB, exponential backoff retry.

import gzip as _gzip
import tempfile as _tempfile

_LOG_BATCH_SIZE = 50        # flush every N lines
_LOG_FLUSH_INTERVAL = 1.0   # or every N seconds
_LOG_BACKOFF_MAX = 60       # max retry delay (seconds)
_LOG_SPOOL_MAX = 50_000     # max lines in disk spool before dropping oldest


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
        self._spool_path = os.path.join(
            _tempfile.gettempdir(), f"xcelsior-logs-{job_id}.jsonl"
        )
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
                            msg = line[space + 1:]
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
            spool_size = os.path.getsize(self._spool_path) if os.path.exists(self._spool_path) else 0
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

    log.info("Monitoring interactive container %s (job %s) — will run until stopped", container_name, job_id)

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
                # Interactive container exited unexpectedly
                exit_result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                exit_code = int(exit_result.stdout.strip()) if exit_result.stdout.strip() else -1
                log.warning("Interactive container %s exited unexpectedly (code %d)", container_name, exit_code)
                report_job_status(job_id, "failed")
                release_lease(job_id, "failed")
                _remove_container(container_name)
                return

            elif status not in ("running", "created", "restarting"):
                log.warning("Interactive container %s in unexpected state: %s", container_name, status)
                report_job_status(job_id, "failed")
                release_lease(job_id, "failed")
                _remove_container(container_name)
                return

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            log.debug("Interactive container inspect failed — may have been removed")
            report_job_status(job_id, "failed")
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


def _inject_ssh_keys(job_id: str, container_name: str):
    """Fetch the job owner's SSH public keys from the API and inject into the container.

    Sets up /root/.ssh/authorized_keys and starts sshd if available.
    Best-effort: failures are logged but don't block the job.
    """
    try:
        resp = requests.get(
            _api_url(f"/agent/ssh-keys/{job_id}"),
            headers=_api_headers(),
            timeout=10,
        )
        if resp.status_code != 200:
            log.debug("SSH keys endpoint returned %d for job %s", resp.status_code, job_id)
            return
        keys = resp.json().get("keys", [])
        if not keys:
            log.debug("No SSH keys found for job %s owner", job_id)
            return

        # Create .ssh directory
        subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", "/root/.ssh"],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["docker", "exec", container_name, "chmod", "700", "/root/.ssh"],
            capture_output=True, timeout=5,
        )

        # Write authorized_keys
        authorized_keys = "\n".join(keys) + "\n"
        subprocess.run(
            ["docker", "exec", container_name, "sh", "-c",
             f"echo {_shell_quote(authorized_keys)} > /root/.ssh/authorized_keys && "
             "chmod 600 /root/.ssh/authorized_keys"],
            capture_output=True, timeout=5,
        )

        log.info("Injected %d SSH key(s) into container %s for job %s", len(keys), container_name, job_id)

        # Try to start sshd if available (best-effort)
        sshd_check = subprocess.run(
            ["docker", "exec", container_name, "which", "sshd"],
            capture_output=True, timeout=5,
        )
        if sshd_check.returncode == 0:
            # Generate host keys if missing
            subprocess.run(
                ["docker", "exec", container_name, "sh", "-c",
                 "ls /etc/ssh/ssh_host_*_key 2>/dev/null || ssh-keygen -A"],
                capture_output=True, timeout=10,
            )
            # Ensure PermitRootLogin is enabled
            subprocess.run(
                ["docker", "exec", container_name, "sh", "-c",
                 "sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config 2>/dev/null || true"],
                capture_output=True, timeout=5,
            )
            # Start sshd
            sshd_start = subprocess.run(
                ["docker", "exec", container_name, "/usr/sbin/sshd"],
                capture_output=True, timeout=5,
            )
            if sshd_start.returncode == 0:
                log.info("sshd started in container %s", container_name)
            else:
                log.debug("sshd start failed: %s", sshd_start.stderr.strip())
        else:
            log.debug("sshd not found in container %s — SSH keys injected but no daemon", container_name)

    except Exception as e:
        log.warning("SSH key injection failed for job %s: %s", job_id, e)


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
            log.warning("docker rm -f %s failed: %s — trying by ID", container_name, result.stderr.strip())
            # Fallback: find container ID by name and remove by ID
            ps = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name=^/{container_name}$", "-q"],
                capture_output=True, text=True, timeout=10,
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
                                    # if the NFS mount is stale (soft mount returns EIO quickly,
                                    # but stat may still block briefly on slow NFS).
                                    stat_result = subprocess.run(
                                        ["stat", "-f", "--format=%b %f %S", mp],
                                        capture_output=True, text=True, timeout=3,
                                    )
                                    if stat_result.returncode == 0:
                                        parts = stat_result.stdout.strip().split()
                                        blocks, bfree, bsize = int(parts[0]), int(parts[1]), int(parts[2])
                                        total = (blocks * bsize) / (1024 ** 3)
                                        used = ((blocks - bfree) * bsize) / (1024 ** 3)
                                        vol_health.append({
                                            "volume_id": entry,
                                            "mounted": True,
                                            "total_gb": round(total, 2),
                                            "used_gb": round(used, 2),
                                        })
                                    else:
                                        vol_health.append({
                                            "volume_id": entry,
                                            "mounted": False,
                                            "total_gb": 0,
                                            "used_gb": 0,
                                        })
                                        nfs_healthy = False
                                except (subprocess.TimeoutExpired, OSError):
                                    vol_health.append({
                                        "volume_id": entry,
                                        "mounted": False,
                                        "total_gb": 0,
                                        "used_gb": 0,
                                    })
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
    log.info("  Tailscale:      %s", "enabled" if TAILSCALE_ENABLED else "disabled")
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

            # Start log forwarding
            log_fwd = LogForwarder(job_id, container_name)
            log_fwd.start()

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
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for cid in result.stdout.strip().split("\n"):
                cid = cid.strip()
                if not cid:
                    continue
                # Inspect container to get bind mount sources
                insp = subprocess.run(
                    ["docker", "inspect", "--format",
                     '{{range .Mounts}}{{if eq .Type "bind"}}{{.Source}}{{" "}}{{end}}{{end}}',
                     cid],
                    capture_output=True, text=True, timeout=5,
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
                capture_output=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            log.warning("mountpoint check timed out for %s — skipping (possible stale NFS)", mount_path)
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

    # ── Step 7: Adopt containers started by scheduler (before agent was running) ──
    adopt_running_containers()

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
    PREPULL_INTERVAL = 600  # Pre-pull popular images every 10 min when idle

    while not _shutdown.is_set():
        try:
            # Check for preemptions
            preempt_jobs = check_preemption()
            if preempt_jobs:
                handle_preemptions(preempt_jobs)

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
                            cache_prepull_popular(popular[:5])  # Max 5 per cycle

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
