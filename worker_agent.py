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

    def nvml_init(): return False
    def nvml_shutdown(): pass
    def is_nvml_available(): return False
    def collect_all_gpus(): return []
    def get_gpu_info_nvml(): return None
    def build_verification_report(gpu_index=0): return None

# ── Configuration ─────────────────────────────────────────────────────

# Required
HOST_ID = os.environ.get("XCELSIOR_HOST_ID")
SCHEDULER_URL = os.environ.get("XCELSIOR_SCHEDULER_URL")

# Auth
API_TOKEN = os.environ.get("XCELSIOR_API_TOKEN", "")

# Optional tuning
COST_PER_HOUR = float(os.environ.get("XCELSIOR_COST_PER_HOUR", "0.50"))
HEARTBEAT_INTERVAL = int(os.environ.get("XCELSIOR_HEARTBEAT_INTERVAL", "10"))
POLL_INTERVAL = int(os.environ.get("XCELSIOR_POLL_INTERVAL", "5"))
MINING_CHECK_INTERVAL = int(os.environ.get("XCELSIOR_MINING_CHECK_INTERVAL", "60"))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("XCELSIOR_MAX_FAILURES", "30"))

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
_active_lock = threading.Lock()


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
            capture_output=True, text=True, check=True, timeout=10,
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
    # If Tailscale is active, prefer the Tailscale IP
    if TAILSCALE_ENABLED:
        try:
            r = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True, text=True, timeout=5,
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
    except Exception:
        pass

    # ip route fallback
    try:
        r = subprocess.run(["ip", "route", "get", "1"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            match = re.search(r"src (\S+)", r.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass

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
                sys.executable, "-c",
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
    report["total_vram_gb"] = round(props.total_mem / (1024**3), 2)
    report["compute_capability"] = f"{props.major}.{props.minor}"

    # ── CUDA / Driver Versions ──
    report["cuda_version"] = torch.version.cuda or ""
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        report["driver_version"] = smi.stdout.strip().split("\\n")[0].strip() if smi.returncode == 0 else ""
    except Exception:
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
            except Exception:
                pass
            time.sleep(0.5)
        if temps:
            report["gpu_temp_celsius"] = max(temps)
            report["gpu_temp_avg_celsius"] = round(sum(temps) / len(temps), 1)
            report["gpu_temp_samples"] = len(temps)
        else:
            report["gpu_temp_celsius"] = 0
    except Exception as e:
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
            capture_output=True, text=True, timeout=300,  # Longer timeout for thermal test
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
            capture_output=True, text=True, timeout=30,
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
    except Exception:
        pass

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
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            status = json.loads(r.stdout)
            if status.get("BackendState") == "Running":
                log.info("Tailscale already connected (IP: %s)", status.get("TailscaleIPs", ["?"])[0])
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

def _api_headers():
    """Build standard API headers."""
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
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
            _api_url("/host"), json=data, headers=_api_headers(), timeout=10,
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
            _api_url("/agent/versions"), json=data, headers=_api_headers(), timeout=10,
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
            headers=_api_headers(), timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("jobs", [])
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
            headers=_api_headers(), timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("preempt_jobs", [])
        return []
    except requests.RequestException:
        return []


def report_job_status(job_id, status, host_id=None):
    """Update job status on the scheduler.

    PATCH /job/{job_id}
    """
    data = {"status": status}
    if host_id:
        data["host_id"] = host_id
    try:
        resp = requests.patch(
            _api_url(f"/job/{job_id}"),
            json=data, headers=_api_headers(), timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException as e:
        log.error("Status update failed for job %s: %s", job_id, e)
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
            json=data, headers=_api_headers(), timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            log.info("Lease claimed: job=%s lease=%s expires=%.0f",
                     job_id, result.get("lease_id"), result.get("expires_at", 0))
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
            json=data, headers=_api_headers(), timeout=10,
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
            json=data, headers=_api_headers(), timeout=10,
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
            json=data, headers=_api_headers(), timeout=10,
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
            json=data, headers=_api_headers(), timeout=10,
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
            json=payload, headers=_api_headers(), timeout=15,
        )
        if r.ok:
            result = r.json()
            log.info("Verification result: state=%s score=%.0f",
                     result.get("state", "?"), result.get("score", 0))
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
            json=payload, headers=_api_headers(), timeout=5,
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
            capture_output=True, text=True, timeout=15,
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


def cache_evict_lru():
    """Evict least-recently-used images when cache exceeds limit.

    Evicts until cache is below IMAGE_CACHE_EVICT_LOW_GB.
    Never evicts currently-running container images.
    """
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
            capture_output=True, text=True, timeout=10,
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
        if image_tag in running_images:
            continue

        # Remove image
        try:
            rm = subprocess.run(
                ["docker", "rmi", image_tag],
                capture_output=True, text=True, timeout=30,
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
                capture_output=True, text=True, timeout=600,
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
            headers=_api_headers(), timeout=10,
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
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            return True  # Already mounted

        # Mount NFS
        mount_cmd = [
            "mount", "-t", "nfs",
            "-o", "noatime,nodiratime,rsize=65536,wsize=65536,hard,intr,timeo=600",
            f"{server}:{path}", mount_point,
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
            capture_output=True, timeout=10,
        )
        log.info("NFS unmounted: %s", mount_point)
    except Exception as e:
        log.debug("NFS unmount failed (non-fatal): %s", e)


# ── Container Lifecycle ──────────────────────────────────────────────

def run_job(job):
    """Execute a job in a hardened Docker container.

    Lifecycle:
      1. Mount NFS volume (if configured)
      2. Pull/verify image
      3. Build secure Docker args (from security.py)
      4. Start container
      5. Apply egress rules
      6. Monitor until completion
      7. Report final status
      8. Unmount NFS (if mounted)
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
    nfs_mount_point = job.get("nfs_mount_point") or os.environ.get("XCELSIOR_NFS_MOUNT", "/mnt/xcelsior-nfs")
    nfs_mounted = False

    if not image:
        log.error("Job %s has no image specified — skipping", job_id)
        report_job_status(job_id, "failed")
        return

    container_name = f"xcelsior-{job_id[:12]}"
    log.info("Starting job %s (%s) — image=%s", job_id, job_name, image)

    # ── Lease Protocol: claim lease before starting work ──
    # Job arrives as "assigned"; agent claims lease → "leased" → then "running"
    lease_info = claim_lease(job_id)
    lease_interval = (lease_info or {}).get("duration_sec", 300) // 2  # Renew at half-life

    # Report running status (leased → running)
    report_job_status(job_id, "running", host_id=HOST_ID)

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
        target=_lease_renewal_loop, name=f"lease-{job_id[:8]}", daemon=True,
    )
    lease_thread.start()

    try:
        # 0. Mount NFS volume (if configured)
        if nfs_server and nfs_path:
            nfs_mounted = _mount_nfs(nfs_server, nfs_path, nfs_mount_point)
            if nfs_mounted:
                log.info("NFS mounted: %s:%s → %s", nfs_server, nfs_path, nfs_mount_point)
                volumes = list(volumes) + [f"{nfs_mount_point}:/data/nfs:rw"]
            else:
                log.warning("NFS mount failed — continuing without shared storage")

        # 1. Pull image (with cache tracking + LRU eviction)
        cache_evict_lru()  # Evict before pulling if needed
        log.info("Pulling image %s...", image)
        pull = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True, timeout=600,
        )
        if pull.returncode != 0:
            log.error("Image pull failed: %s", pull.stderr.strip())
            report_job_status(job_id, "failed")
            return

        # Track in LRU cache index
        local_images = _get_local_images()
        cache_track_pull(image, local_images.get(image, 0))

        # 2. Determine runtime
        gpu_info = get_gpu_info()
        runtime_name = "runc"
        if PREFER_GVISOR:
            runtime_name, reason = recommend_runtime(gpu_info["gpu_model"])
            log.info("Runtime: %s (%s)", runtime_name, reason)

        # Multi-GPU: get requested GPU count from job
        num_gpus = job.get("num_gpus", 1) or 1
        if num_gpus > 1:
            log.info("Multi-GPU job: requesting %d GPUs", num_gpus)

        # 3. Build secure Docker args
        docker_args = build_secure_docker_args(
            image=image,
            container_name=container_name,
            gpu=True,
            num_gpus=num_gpus,
            runtime=runtime_name,
            environment=env_vars,
            volumes=volumes,
            command=command,
        )

        # 4. Start container
        log.info("Starting container %s", container_name)
        start = subprocess.run(
            docker_args, capture_output=True, text=True, timeout=60,
        )
        if start.returncode != 0:
            log.error("Container start failed: %s", start.stderr.strip())
            report_job_status(job_id, "failed")
            return

        container_id = start.stdout.strip()[:12]
        log.info("Container started: %s", container_id)

        # 5. Apply egress rules (best-effort)
        try:
            egress_rules = build_egress_iptables_rules(container_name)
            for rule in egress_rules:
                subprocess.run(
                    rule.split(), capture_output=True, timeout=5,
                )
        except Exception as e:
            log.debug("Egress rules failed (non-fatal): %s", e)

        # 6. Monitor container until completion
        _monitor_container(job_id, container_name)

    except subprocess.TimeoutExpired:
        log.error("Job %s timed out during execution", job_id)
        _kill_container(container_name)
        report_job_status(job_id, "failed")
        release_lease(job_id, "failed")
    except Exception as e:
        log.error("Job %s failed: %s", job_id, e, exc_info=True)
        _kill_container(container_name)
        report_job_status(job_id, "failed")
        release_lease(job_id, "failed")
    finally:
        # Stop lease renewal thread
        _lease_stop.set()
        with _active_lock:
            _active_containers.pop(job_id, None)

        # Unmount NFS if we mounted it
        if nfs_mounted:
            _unmount_nfs(nfs_mount_point)


def _monitor_container(job_id, container_name):
    """Monitor a running container until it exits or is preempted."""
    check_interval = 5  # seconds

    while not _shutdown.is_set():
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True, text=True, timeout=10,
            )
            status = result.stdout.strip()

            if status == "exited":
                # Check exit code
                exit_result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_name],
                    capture_output=True, text=True, timeout=10,
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
                            capture_output=True, text=True, timeout=10,
                        )
                        if logs.stdout:
                            log.info("Container output (last 50 lines):\n%s", logs.stdout[-2000:])
                    except Exception:
                        pass
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

        # Sleep in small increments to respond to shutdown quickly
        for _ in range(check_interval):
            if _shutdown.is_set():
                return
            time.sleep(1)


def _kill_container(container_name):
    """Force-kill and remove a container."""
    with suppress(Exception):
        subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True, timeout=10,
        )
    _remove_container(container_name)


def _remove_container(container_name):
    """Remove a container (force)."""
    with suppress(Exception):
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=10,
        )


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
                capture_output=True, timeout=20,
            )
        except (subprocess.TimeoutExpired, Exception):
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
    """
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
                    gpu["index"], confidence * 100, reason, consecutive_mining_detections,
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
                    log.error("Too many heartbeat failures (%d) — shutting down", consecutive_failures)
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
    """Stop all active containers and deregister from scheduler."""
    log.info("Graceful shutdown — stopping %d active containers", len(_active_containers))

    with _active_lock:
        for job_id, container_name in list(_active_containers.items()):
            log.info("Stopping container %s (job %s)", container_name, job_id)
            try:
                subprocess.run(
                    ["docker", "stop", "-t", "15", container_name],
                    capture_output=True, timeout=25,
                )
            except (subprocess.TimeoutExpired, Exception):
                _kill_container(container_name)

            # Mark job as queued so it can be rescheduled
            report_job_status(job_id, "queued")
        _active_containers.clear()

    # Deregister from scheduler
    try:
        requests.delete(
            _api_url(f"/host/{HOST_ID}"),
            headers=_api_headers(), timeout=10,
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
    log.info("=" * 64)
    log.info("  Xcelsior Worker Agent v%s (pull-based)", VERSION)
    log.info("=" * 64)
    log.info("  Host ID:        %s", HOST_ID)
    log.info("  Scheduler URL:  %s", SCHEDULER_URL)
    log.info("  Host IP:        %s", host_ip)
    log.info("  GPU:            %s (%.1f GB VRAM)", gpu_info["gpu_model"], gpu_info["total_vram_gb"])
    log.info("  Cost/hour:      $%.2f", COST_PER_HOUR)
    log.info("  Poll interval:  %ds", POLL_INTERVAL)
    log.info("  Heartbeat:      %ds", HEARTBEAT_INTERVAL)
    log.info("  Runtime:        %s", runtime)
    log.info("  Admitted:       %s", "YES" if admitted else "NO (limited to heartbeats)")
    log.info("  Tailscale:      %s", "enabled" if TAILSCALE_ENABLED else "disabled")
    log.info("=" * 64)


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
        log.info("  CUDA %s | Driver %s | CC %s",
                 bench.get("cuda_version", "?"),
                 bench.get("driver_version", "?"),
                 bench.get("compute_capability", "?"))
        log.info("  PCIe: %.2f GB/s | Peak Temp: %s°C",
                 bench.get("pcie_bandwidth_gbps", 0),
                 bench.get("gpu_temp_celsius", "?"))
        report_benchmark(bench, gpu_info["gpu_model"])

        # Network benchmark (ping + throughput to scheduler)
        log.info("Running network quality benchmark...")
        net_bench = run_network_benchmark()
        if net_bench:
            log.info("  Latency: %.1fms avg | Jitter: %.1fms | Loss: %.1f%% | Throughput: %.1f Mbps",
                     net_bench.get("latency_avg_ms", 0),
                     net_bench.get("jitter_ms", 0),
                     net_bench.get("packet_loss_pct", 0),
                     net_bench.get("throughput_mbps", 0))
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
    log.info("Image cache initialized (%d images, %.0f MB)",
             len(_image_cache_index), _total_cache_size_mb())

    # ── Step 7: Start background threads ──
    threads = []

    # Heartbeat thread
    hb_thread = threading.Thread(
        target=heartbeat_loop, args=(host_ip, compute_score),
        name="heartbeat", daemon=True,
    )
    hb_thread.start()
    threads.append(hb_thread)

    # Telemetry push thread (GPU metrics every 5s)
    telem_thread = threading.Thread(
        target=telemetry_loop,
        name="telemetry", daemon=True,
    )
    telem_thread.start()
    threads.append(telem_thread)
    log.info("Telemetry push started (every %ds)", TELEMETRY_INTERVAL)

    # Mining detection thread
    mining_thread = threading.Thread(
        target=mining_detection_loop,
        name="mining-detection", daemon=True,
    )
    mining_thread.start()
    threads.append(mining_thread)

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
                        target=run_job, args=(job,),
                        name=f"job-{job_id[:8]}", daemon=True,
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
            backoff = min(2 ** consecutive_poll_failures, 300)
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
