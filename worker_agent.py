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
    recommend_runtime,
)

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
    """Query nvidia-smi for GPU name, total VRAM, free VRAM.

    Returns dict with gpu_model, total_vram_gb, free_vram_gb.
    Raises RuntimeError on failure.
    """
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
    """Run a quick FP16 matmul benchmark to measure actual TFLOPS.

    Uses PyTorch if available; returns score dict or None.
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-c",
                """
import time, json
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no_cuda"}))
        raise SystemExit
    device = torch.device("cuda:0")
    # Warm up
    a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()
    # Benchmark
    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    flops_per_iter = 2 * 4096**3  # 2*N^3 for matmul
    tflops = (flops_per_iter * iters) / elapsed / 1e12
    print(json.dumps({"tflops": round(tflops, 2), "elapsed_s": round(elapsed, 3), "iters": iters}))
except ImportError:
    print(json.dumps({"error": "no_torch"}))
""",
            ],
            capture_output=True, text=True, timeout=120,
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


# ── Container Lifecycle ──────────────────────────────────────────────

def run_job(job):
    """Execute a job in a hardened Docker container.

    Lifecycle:
      1. Pull/verify image
      2. Build secure Docker args (from security.py)
      3. Start container
      4. Apply egress rules
      5. Monitor until completion
      6. Report final status
    """
    job_id = job.get("job_id", "unknown")
    job_name = job.get("name", "unnamed")
    image = job.get("image", job.get("docker_image", ""))
    command = job.get("command")
    env_vars = job.get("environment", {})
    volumes = job.get("volumes", [])

    if not image:
        log.error("Job %s has no image specified — skipping", job_id)
        report_job_status(job_id, "failed")
        return

    container_name = f"xcelsior-{job_id[:12]}"
    log.info("Starting job %s (%s) — image=%s", job_id, job_name, image)

    # Report running status
    report_job_status(job_id, "running", host_id=HOST_ID)

    with _active_lock:
        _active_containers[job_id] = container_name

    try:
        # 1. Pull image
        log.info("Pulling image %s...", image)
        pull = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True, timeout=600,
        )
        if pull.returncode != 0:
            log.error("Image pull failed: %s", pull.stderr.strip())
            report_job_status(job_id, "failed")
            return

        # 2. Determine runtime
        gpu_info = get_gpu_info()
        runtime_name = "runc"
        if PREFER_GVISOR:
            runtime_name, reason = recommend_runtime(gpu_info["gpu_model"])
            log.info("Runtime: %s (%s)", runtime_name, reason)

        # 3. Build secure Docker args
        docker_args = build_secure_docker_args(
            image=image,
            container_name=container_name,
            gpu=True,
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
    except Exception as e:
        log.error("Job %s failed: %s", job_id, e, exc_info=True)
        _kill_container(container_name)
        report_job_status(job_id, "failed")
    finally:
        with _active_lock:
            _active_containers.pop(job_id, None)


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

                # Cleanup container
                _remove_container(container_name)
                return

            elif status not in ("running", "created"):
                log.warning("Container %s in unexpected state: %s", container_name, status)
                report_job_status(job_id, "failed")
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

    # ── Step 1: Tailscale / Headscale setup ──
    if TAILSCALE_ENABLED:
        setup_tailscale()

    # ── Step 2: Detect GPU ──
    try:
        gpu_info = get_gpu_info()
    except RuntimeError as e:
        log.error("GPU detection failed: %s", e)
        log.error("Make sure nvidia-smi is installed and NVIDIA drivers are loaded.")
        sys.exit(1)

    host_ip = get_host_ip()

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

    # ── Step 4: Compute benchmark ──
    compute_score = None
    log.info("Running compute benchmark (this may take ~30 seconds)...")
    bench = run_compute_benchmark()
    if bench:
        compute_score = bench.get("tflops", 0) / 10  # XCU
        log.info("Benchmark: %.1f TFLOPS = %.1f XCU", bench["tflops"], compute_score)
        report_benchmark(bench, gpu_info["gpu_model"])
    else:
        log.info("Benchmark skipped (PyTorch/CUDA not available)")

    # ── Step 5: Initial heartbeat ──
    heartbeat(gpu_info, host_ip, compute_score)

    # ── Banner ──
    print_startup_banner(gpu_info, host_ip, admitted, runtime_name)

    # ── Step 6: Start background threads ──
    threads = []

    # Heartbeat thread
    hb_thread = threading.Thread(
        target=heartbeat_loop, args=(host_ip, compute_score),
        name="heartbeat", daemon=True,
    )
    hb_thread.start()
    threads.append(hb_thread)

    # Mining detection thread
    mining_thread = threading.Thread(
        target=mining_detection_loop,
        name="mining-detection", daemon=True,
    )
    mining_thread.start()
    threads.append(mining_thread)

    # ── Step 7: Main polling loop ──
    log.info("Entering main polling loop...")
    consecutive_poll_failures = 0

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
    log.info("Worker agent stopped.")


if __name__ == "__main__":
    main()
