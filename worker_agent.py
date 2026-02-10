#!/usr/bin/env python3
# Xcelsior Worker Agent v1.0.0
# GPU status monitoring and reporting agent for distributed GPU hosts.
# Runs on each GPU host and reports VRAM status to the scheduler API.

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Install with: pip install requests")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────

# Required environment variables
HOST_ID = os.environ.get("XCELSIOR_HOST_ID")
SCHEDULER_URL = os.environ.get("XCELSIOR_SCHEDULER_URL")
API_TOKEN = os.environ.get("XCELSIOR_API_TOKEN")

# Optional environment variables
COST_PER_HOUR = float(os.environ.get("XCELSIOR_COST_PER_HOUR", "0.50"))
REPORT_INTERVAL = int(os.environ.get("XCELSIOR_REPORT_INTERVAL", "5"))

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("xcelsior-worker")


# ── GPU Status ────────────────────────────────────────────────────────

def get_gpu_info():
    """
    Query nvidia-smi for GPU information.
    Returns dict with gpu_model, total_vram_gb, and free_vram_gb.
    Raises RuntimeError if nvidia-smi fails.
    """
    try:
        # Query for GPU name and memory info
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
        
        # Parse output (first GPU only for now)
        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            raise RuntimeError("No GPU found in nvidia-smi output")
        
        # Parse first GPU
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi output format: {lines[0]}")
        
        gpu_model = parts[0]
        total_vram_mb = float(parts[1])
        free_vram_mb = float(parts[2])
        
        # Convert to GB
        total_vram_gb = round(total_vram_mb / 1024, 2)
        free_vram_gb = round(free_vram_mb / 1024, 2)
        
        return {
            "gpu_model": gpu_model,
            "total_vram_gb": total_vram_gb,
            "free_vram_gb": free_vram_gb,
        }
    
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi command timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed: {e.stderr}")
    except (ValueError, IndexError) as e:
        raise RuntimeError(f"Failed to parse nvidia-smi output: {e}")


def get_host_ip():
    """
    Get the primary IP address of this host.
    Returns a best-effort IP or "unknown" if detection fails.
    """
    try:
        # Try to get IP from hostname -I
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            if ips:
                return ips[0]
    except Exception:
        pass
    
    # Fallback: try to get from ip route
    try:
        result = subprocess.run(
            ["ip", "route", "get", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            match = re.search(r"src (\S+)", result.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass
    
    return "unknown"


# ── Scheduler Communication ───────────────────────────────────────────

def register_or_update_host(gpu_info, host_ip):
    """
    Register or update this host with the scheduler.
    Sends a PUT request to /host endpoint with current GPU status.
    Returns True on success, False on failure.
    """
    url = f"{SCHEDULER_URL.rstrip('/')}/host"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    data = {
        "host_id": HOST_ID,
        "ip": host_ip,
        "gpu_model": gpu_info["gpu_model"],
        "total_vram_gb": gpu_info["total_vram_gb"],
        "free_vram_gb": gpu_info["free_vram_gb"],
        "cost_per_hour": COST_PER_HOUR,
    }
    
    try:
        response = requests.put(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to update scheduler: {e}")
        return False


# ── Retry Logic ───────────────────────────────────────────────────────

def exponential_backoff(attempt, max_delay=300):
    """
    Calculate exponential backoff delay.
    attempt: retry attempt number (0-indexed)
    max_delay: maximum delay in seconds
    Returns: delay in seconds
    """
    delay = min(2 ** attempt, max_delay)
    return delay


# ── Main Loop ─────────────────────────────────────────────────────────

def validate_config():
    """
    Validate required configuration.
    Exits with error if required environment variables are missing.
    """
    errors = []
    
    if not HOST_ID:
        errors.append("XCELSIOR_HOST_ID is not set")
    if not SCHEDULER_URL:
        errors.append("XCELSIOR_SCHEDULER_URL is not set")
    if not API_TOKEN:
        errors.append("XCELSIOR_API_TOKEN is not set")
    
    if errors:
        log.error("Configuration error:")
        for error in errors:
            log.error(f"  - {error}")
        log.error("\nPlease set the required environment variables and try again.")
        sys.exit(1)


def print_startup_banner():
    """Print startup banner with configuration info."""
    log.info("=" * 60)
    log.info("Xcelsior Worker Agent v1.0.0")
    log.info("=" * 60)
    log.info(f"Host ID:          {HOST_ID}")
    log.info(f"Scheduler URL:    {SCHEDULER_URL}")
    log.info(f"Cost per hour:    ${COST_PER_HOUR:.2f}")
    log.info(f"Report interval:  {REPORT_INTERVAL}s")
    log.info("=" * 60)


def main():
    """Main worker agent loop."""
    # Validate configuration
    validate_config()
    
    # Print startup banner
    print_startup_banner()
    
    # Get host IP (once at startup)
    host_ip = get_host_ip()
    log.info(f"Detected host IP: {host_ip}")
    
    # Initial GPU check
    try:
        gpu_info = get_gpu_info()
        log.info(f"Detected GPU: {gpu_info['gpu_model']} "
                f"({gpu_info['total_vram_gb']} GB total, "
                f"{gpu_info['free_vram_gb']} GB free)")
    except RuntimeError as e:
        log.error(f"Failed to query GPU: {e}")
        log.error("Make sure nvidia-smi is installed and NVIDIA drivers are working.")
        sys.exit(1)
    
    # Main monitoring loop
    consecutive_failures = 0
    last_success_time = datetime.now()
    
    log.info("Starting monitoring loop...")
    
    while True:
        try:
            # Get current GPU status
            gpu_info = get_gpu_info()
            
            # Report to scheduler
            success = register_or_update_host(gpu_info, host_ip)
            
            if success:
                log.info(f"✓ Updated scheduler - Free VRAM: {gpu_info['free_vram_gb']} GB")
                consecutive_failures = 0
                last_success_time = datetime.now()
            else:
                consecutive_failures += 1
                log.warning(f"Failed to update scheduler (attempt {consecutive_failures})")
                
                # If we've had many failures, use exponential backoff
                if consecutive_failures > 3:
                    backoff_delay = exponential_backoff(consecutive_failures - 3)
                    log.warning(f"Multiple failures. Waiting {backoff_delay}s before retry...")
                    time.sleep(backoff_delay)
                    continue
            
            # Normal sleep interval
            time.sleep(REPORT_INTERVAL)
        
        except RuntimeError as e:
            # GPU query failed
            log.error(f"GPU query failed: {e}")
            consecutive_failures += 1
            
            if consecutive_failures > 10:
                log.error("Too many consecutive GPU query failures. Check nvidia-smi.")
                sys.exit(1)
            
            time.sleep(REPORT_INTERVAL)
        
        except KeyboardInterrupt:
            log.info("\nShutting down worker agent...")
            break
        
        except Exception as e:
            # Unexpected error
            log.error(f"Unexpected error: {e}", exc_info=True)
            consecutive_failures += 1
            
            if consecutive_failures > 10:
                log.error("Too many consecutive failures. Exiting.")
                sys.exit(1)
            
            time.sleep(REPORT_INTERVAL)


if __name__ == "__main__":
    main()
