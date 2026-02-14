# Xcelsior Security Module
# Defense-in-depth for untrusted consumer GPU hosts.
#
# Implements REPORT_EXCELSIOR_TECHNICAL_FINAL.md security layers:
#   Layer 1: Version gating / node admission control
#   Layer 2: Least privilege container configs
#   Layer 3: Network controls + mining resistance
#   Layer 4: (Optional) Sandboxed runtimes (gVisor/Kata)

import json
import logging
import os
import re
import subprocess
import time

log = logging.getLogger("xcelsior")

# ── Layer 1: Version Gating / Node Admission Control ─────────────────
# Refuse to run workloads on nodes with known-vulnerable components.
#
# CVE-2024-21626: runc container escape — patched in runc 1.1.12
# CVE-2025-23266: NVIDIA Container Toolkit escape — patched in 1.17.8

# Minimum safe versions (update as new CVEs are published)
MINIMUM_VERSIONS = {
    "runc": "1.1.12",         # CVE-2024-21626 fix
    "nvidia_toolkit": "1.17.8",  # CVE-2025-23266 fix
    "nvidia_driver": "550.0",    # Minimum recommended driver
    "docker": "24.0.0",          # Modern Docker with security defaults
}


def parse_version(version_str):
    """Parse a version string into a comparable tuple of integers."""
    if not version_str:
        return (0,)
    # Strip leading 'v' and any suffixes like '-rc1'
    clean = re.sub(r"^v", "", str(version_str))
    clean = re.split(r"[-+~]", clean)[0]
    try:
        return tuple(int(x) for x in clean.split("."))
    except (ValueError, AttributeError):
        return (0,)


def version_gte(current, minimum):
    """Check if current version >= minimum version."""
    return parse_version(current) >= parse_version(minimum)


def check_node_versions(versions):
    """Validate node component versions against minimum requirements.

    Args:
        versions: dict with keys like 'runc', 'nvidia_toolkit', 'nvidia_driver', 'docker'

    Returns:
        (admitted: bool, reasons: list[str])
    """
    reasons = []

    # runc version check (CVE-2024-21626)
    runc_ver = versions.get("runc")
    if runc_ver and not version_gte(runc_ver, MINIMUM_VERSIONS["runc"]):
        reasons.append(
            f"runc {runc_ver} is vulnerable to CVE-2024-21626. "
            f"Minimum: {MINIMUM_VERSIONS['runc']}"
        )

    # NVIDIA Container Toolkit (CVE-2025-23266)
    toolkit_ver = versions.get("nvidia_toolkit")
    if toolkit_ver and not version_gte(toolkit_ver, MINIMUM_VERSIONS["nvidia_toolkit"]):
        reasons.append(
            f"NVIDIA Container Toolkit {toolkit_ver} is vulnerable to CVE-2025-23266. "
            f"Minimum: {MINIMUM_VERSIONS['nvidia_toolkit']}"
        )

    # NVIDIA driver
    driver_ver = versions.get("nvidia_driver")
    if driver_ver and not version_gte(driver_ver, MINIMUM_VERSIONS["nvidia_driver"]):
        reasons.append(
            f"NVIDIA driver {driver_ver} is below minimum {MINIMUM_VERSIONS['nvidia_driver']}"
        )

    # Docker version
    docker_ver = versions.get("docker")
    if docker_ver and not version_gte(docker_ver, MINIMUM_VERSIONS["docker"]):
        reasons.append(
            f"Docker {docker_ver} is below minimum {MINIMUM_VERSIONS['docker']}"
        )

    admitted = len(reasons) == 0
    return admitted, reasons


def get_local_versions():
    """Detect component versions on the local host.

    Returns dict of version strings.
    """
    versions = {}

    # runc version
    try:
        r = subprocess.run(
            ["runc", "--version"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            match = re.search(r"runc version (\S+)", r.stdout)
            if match:
                versions["runc"] = match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Docker version
    try:
        r = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            versions["docker"] = r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # NVIDIA driver version
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            versions["nvidia_driver"] = r.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # NVIDIA Container Toolkit version
    try:
        r = subprocess.run(
            ["nvidia-container-toolkit", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            match = re.search(r"(\d+\.\d+\.\d+)", r.stdout + r.stderr)
            if match:
                versions["nvidia_toolkit"] = match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Fallback: check nvidia-container-runtime
        try:
            r = subprocess.run(
                ["nvidia-container-runtime", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                match = re.search(r"(\d+\.\d+\.\d+)", r.stdout)
                if match:
                    versions["nvidia_toolkit"] = match.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return versions


# ── Layer 2: Least Privilege Container Configuration ──────────────────
# Enforce rootless, no-privileged, no-new-privileges, cap-drop ALL

# Default egress allowlist for containers (Layer 3)
DEFAULT_EGRESS_ALLOWLIST = [
    # Package registries
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    # Model/data registries
    "huggingface.co",
    "*.huggingface.co",
    "github.com",
    "*.githubusercontent.com",
    # Container registries
    "registry-1.docker.io",
    "auth.docker.io",
    "production.cloudflare.docker.com",
    "ghcr.io",
    # Cloud storage
    "*.s3.amazonaws.com",
    "s3.amazonaws.com",
    "storage.googleapis.com",
    # DNS
    "dns.google",
]


def build_secure_docker_args(
    image,
    container_name,
    gpu=True,
    num_gpus=0,
    runtime="runsc",
    egress_allowlist=None,
    extra_args=None,
    environment=None,
    volumes=None,
    command=None,
):
    """Build Docker run arguments with defense-in-depth security.

    Implements:
    - Non-root execution (USER 1000:1000)
    - No privileged mode
    - No new privileges (prevent setuid/setgid escalation)
    - Drop all capabilities, add back only what's needed
    - Read-only root filesystem
    - Tmpfs for /tmp
    - Memory limits
    - Optional gVisor runtime (runsc)
    - Optional GPU access (--gpus, not --privileged)
    - Multi-GPU support (specific device selection)
    """
    args = ["docker", "run", "-d"]

    # Container identity
    args.extend(["--name", container_name])

    # Security: no privileged, no new privileges
    args.append("--security-opt=no-new-privileges")

    # Security: drop all capabilities, add back minimums
    args.extend(["--cap-drop", "ALL"])

    # Security: read-only root filesystem + tmpfs for writable dirs
    args.append("--read-only")
    args.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=1g"])
    args.extend(["--tmpfs", "/var/tmp:rw,noexec,nosuid,size=512m"])

    # Security: non-root user
    args.extend(["--user", "1000:1000"])

    # GPU access (via --gpus, NOT --privileged)
    if gpu:
        if num_gpus > 1:
            # Multi-GPU: specify exact device indices
            devices = ",".join(str(i) for i in range(num_gpus))
            args.extend(["--gpus", f'"device={devices}"'])
        else:
            args.extend(["--gpus", "all"])

    # Sandboxed runtime (gVisor)
    if runtime and runtime != "runc":
        args.extend(["--runtime", runtime])

    # Resource limits
    args.extend(["--memory", "32g"])
    args.extend(["--memory-swap", "32g"])
    args.extend(["--pids-limit", "4096"])

    # Environment variables
    if environment:
        for key, val in environment.items():
            args.extend(["-e", f"{key}={val}"])

    # Volume mounts (read-only by default)
    if volumes:
        for vol in volumes:
            if ":" in vol and not vol.endswith(":ro") and not vol.endswith(":rw"):
                vol += ":ro"
            args.extend(["-v", vol])

    # Extra args (e.g., network settings)
    if extra_args:
        args.extend(extra_args)

    # Image
    args.append(image)

    # Command
    if command:
        if isinstance(command, list):
            args.extend(command)
        else:
            args.append(command)

    return args


def build_egress_iptables_rules(container_id, allowlist=None, strict=True):
    """Generate iptables rules for default-deny egress with allowlist.

    Per REPORT_XCELSIOR_TECHNICAL_FINAL.md Step 8:
    "Default-deny outbound networking with allowlists for common
    package/model registries as needed."

    This creates network isolation to prevent:
    - Cryptomining pool connections (Stratum protocol)
    - Data exfiltration
    - C2 communication

    Args:
        container_id: Docker container name or ID.
        allowlist: List of allowed domains (uses DEFAULT_EGRESS_ALLOWLIST if None).
        strict: If True (default), set OUTPUT policy to DROP (default-deny).
                Set to False for mining-port-block-only mode.

    Returns list of iptables command strings.
    """
    allowlist = allowlist or DEFAULT_EGRESS_ALLOWLIST
    rules = []

    # Allow DNS (needed for domain resolution)
    rules.append(
        f"docker exec {container_id} "
        "iptables -A OUTPUT -p udp --dport 53 -j ACCEPT"
    )
    rules.append(
        f"docker exec {container_id} "
        "iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT"
    )

    # Allow established connections
    rules.append(
        f"docker exec {container_id} "
        "iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT"
    )

    # Allow loopback
    rules.append(
        f"docker exec {container_id} "
        "iptables -A OUTPUT -o lo -j ACCEPT"
    )

    # Allow HTTPS/HTTP to allowlisted domains
    # Note: iptables works on IPs; in practice use Docker network policy
    # or DNS-based proxy. These rules permit ports 80/443 broadly —
    # combine with mining pool port blocks for defense-in-depth.
    rules.append(
        f"docker exec {container_id} "
        "iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT"
    )
    rules.append(
        f"docker exec {container_id} "
        "iptables -A OUTPUT -p tcp --dport 80 -j ACCEPT"
    )

    # Block common mining pool ports (Stratum protocol)
    for port in [3333, 4444, 5555, 7777, 8888, 9999, 14444, 14433]:
        rules.append(
            f"docker exec {container_id} "
            f"iptables -A OUTPUT -p tcp --dport {port} -j DROP"
        )

    # Default-deny: drop all other outbound traffic
    # This is the core of "default-deny egress with allowlists"
    if strict:
        rules.append(
            f"docker exec {container_id} "
            "iptables -P OUTPUT DROP"
        )

    return rules


# ── Layer 3: Cryptomining Detection ──────────────────────────────────
# Mining heuristic: GPU_UTIL > 95% AND PCIE_TX < threshold for > 5 min

MINING_GPU_UTIL_THRESHOLD = float(os.environ.get("XCELSIOR_MINING_GPU_UTIL", "95"))
MINING_PCIE_TX_THRESHOLD = float(os.environ.get("XCELSIOR_MINING_PCIE_TX_MB", "5"))
MINING_DURATION_THRESHOLD = int(os.environ.get("XCELSIOR_MINING_DURATION_SEC", "300"))


def check_mining_heuristic(gpu_stats):
    """Check if GPU utilization pattern matches cryptomining signature.

    Mining signature: High compute (>95%), high memory, low PCIe TX (<5MB/s).
    Legitimate training: Bursty compute, high PCIe bus usage for data shuffling.

    Args:
        gpu_stats: dict with 'utilization', 'memory_util', 'pcie_tx_mb_s'

    Returns:
        (is_mining: bool, confidence: float, reason: str)
    """
    util = gpu_stats.get("utilization", 0)
    mem_util = gpu_stats.get("memory_util", 0)
    pcie_tx = gpu_stats.get("pcie_tx_mb_s", 999)

    if util > MINING_GPU_UTIL_THRESHOLD and pcie_tx < MINING_PCIE_TX_THRESHOLD:
        confidence = min(1.0, (util - 90) / 10 * (1 - pcie_tx / MINING_PCIE_TX_THRESHOLD))
        return True, confidence, (
            f"GPU util={util}% mem={mem_util}% pcie_tx={pcie_tx}MB/s — "
            f"matches mining signature (high compute, low PCIe)"
        )

    return False, 0.0, "Normal GPU usage pattern"


def get_gpu_telemetry():
    """Query GPU telemetry used in mining detection.

    Uses NVML (pynvml) when available per REPORT_FEATURE_1.md:
    "avoid calling the nvidia-smi binary... use Python bindings to
    interact with libnvidia-ml.so directly."

    Falls back to nvidia-smi subprocess if NVML is not initialized.
    Returns list of dicts with utilization stats per GPU.
    """
    try:
        from nvml_telemetry import collect_all_gpus, is_nvml_available
        if is_nvml_available():
            return collect_all_gpus()
    except ImportError:
        pass

    # Legacy fallback: nvidia-smi subprocess
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "pcie.link.gen.current,pcie.link.width.current,"
                "power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "utilization": float(parts[1]),
                    "memory_util": float(parts[2]),
                    "pcie_gen": parts[3],
                    "pcie_width": parts[4],
                    "power_draw_w": float(parts[5]) if parts[5] != "[N/A]" else 0,
                    "temperature_c": float(parts[6]),
                    "pcie_tx_mb_s": 0,
                })
        return gpus

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []


# ── Layer 4: Runtime Selection ────────────────────────────────────────

# gVisor (runsc) with nvproxy for GPU passthrough
# Supported GPU families: T4, A100, A10G, L4, H100 (official)
# Consumer GPUs (RTX 3090/4090) on same microarchitectures "likely work"

GVISOR_SUPPORTED_GPUS = {
    "T4", "A100", "A10G", "L4", "H100",
    # Consumer cards on supported microarchitectures (unofficial but likely to work)
    "RTX 3060", "RTX 3070", "RTX 3080", "RTX 3090",
    "RTX 4060", "RTX 4070", "RTX 4080", "RTX 4090",
    "RTX 2060", "RTX 2070", "RTX 2080",
}

GVISOR_RELEASE_URL = "https://storage.googleapis.com/gvisor/releases/release/latest/x86_64"


def install_gvisor(enable_nvproxy=True):
    """Auto-install gVisor (runsc) runtime with optional nvproxy for GPU passthrough.

    Steps:
    1. Download runsc + containerd-shim-runsc-v1 binaries
    2. Install to /usr/local/bin
    3. Configure Docker daemon to register runsc runtime
    4. Restart Docker to pick up new runtime

    Returns (success: bool, message: str).
    """
    if is_gvisor_available():
        return True, "gVisor already installed"

    log.info("Installing gVisor (runsc) runtime...")

    # Require root for installation
    if os.geteuid() != 0:
        return False, "gVisor install requires root privileges (run with sudo)"

    try:
        # 1. Download runsc binary
        log.info("Downloading runsc binary...")
        dl_runsc = subprocess.run(
            ["wget", "-q", f"{GVISOR_RELEASE_URL}/runsc", "-O", "/usr/local/bin/runsc"],
            capture_output=True, text=True, timeout=120,
        )
        if dl_runsc.returncode != 0:
            return False, f"Failed to download runsc: {dl_runsc.stderr.strip()}"

        # 2. Download containerd-shim-runsc-v1
        log.info("Downloading containerd-shim-runsc-v1...")
        dl_shim = subprocess.run(
            ["wget", "-q", f"{GVISOR_RELEASE_URL}/containerd-shim-runsc-v1",
             "-O", "/usr/local/bin/containerd-shim-runsc-v1"],
            capture_output=True, text=True, timeout=120,
        )
        if dl_shim.returncode != 0:
            return False, f"Failed to download containerd-shim: {dl_shim.stderr.strip()}"

        # 3. Make executable
        os.chmod("/usr/local/bin/runsc", 0o755)
        os.chmod("/usr/local/bin/containerd-shim-runsc-v1", 0o755)

        # 4. Configure Docker daemon with runsc runtime
        daemon_config_path = "/etc/docker/daemon.json"
        daemon_config = {}
        if os.path.exists(daemon_config_path):
            with open(daemon_config_path) as f:
                daemon_config = json.load(f)

        runtimes = daemon_config.get("runtimes", {})
        runsc_opts = ["--network=sandbox", "--debug-log=/var/log/runsc/"]

        if enable_nvproxy:
            runsc_opts.append("--nvproxy")
            log.info("Enabling nvproxy for GPU passthrough")

        runtimes["runsc"] = {
            "path": "/usr/local/bin/runsc",
            "runtimeArgs": runsc_opts,
        }
        daemon_config["runtimes"] = runtimes

        os.makedirs(os.path.dirname(daemon_config_path), exist_ok=True)
        with open(daemon_config_path, "w") as f:
            json.dump(daemon_config, f, indent=2)

        # 5. Create log directory for runsc
        os.makedirs("/var/log/runsc", exist_ok=True)

        # 6. Restart Docker to pick up new runtime
        log.info("Restarting Docker to register runsc runtime...")
        restart = subprocess.run(
            ["systemctl", "restart", "docker"],
            capture_output=True, text=True, timeout=30,
        )
        if restart.returncode != 0:
            log.warning("Docker restart failed: %s — runtime may not be active",
                        restart.stderr.strip())

        # 7. Verify installation
        time.sleep(2)
        if is_gvisor_available():
            log.info("gVisor (runsc) installed and verified successfully")
            return True, "gVisor installed successfully with nvproxy" if enable_nvproxy else "gVisor installed successfully"
        else:
            return False, "gVisor binaries installed but runsc not responding"

    except subprocess.TimeoutExpired:
        return False, "gVisor installation timed out"
    except Exception as e:
        return False, f"gVisor installation failed: {e}"


def is_gvisor_available():
    """Check if gVisor (runsc) runtime is installed and available."""
    try:
        r = subprocess.run(
            ["runsc", "--version"], capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def recommend_runtime(gpu_model):
    """Recommend the best available container runtime for a GPU.

    Priority:
    1. gVisor (runsc) with nvproxy if available and GPU supported
    2. Standard runc with hardening

    Returns (runtime_name, reason).
    """
    if is_gvisor_available():
        # Check if GPU is in supported list
        for supported in GVISOR_SUPPORTED_GPUS:
            if supported.lower() in gpu_model.lower():
                return "runsc", f"gVisor with nvproxy (GPU {gpu_model} supported)"

        return "runsc", f"gVisor (GPU {gpu_model} not officially supported, may work)"

    return "runc", "Standard runc (gVisor not available — apply hardening)"


# ── Admission Controller ─────────────────────────────────────────────


def admit_node(host_id, versions, gpu_model=None):
    """Full admission control check for a node.

    Combines version gating, runtime recommendation, and security posture.

    Returns:
        (admitted: bool, details: dict)
    """
    admitted, reasons = check_node_versions(versions)

    runtime, runtime_reason = recommend_runtime(gpu_model or "unknown")

    details = {
        "host_id": host_id,
        "admitted": admitted,
        "versions": versions,
        "rejection_reasons": reasons,
        "recommended_runtime": runtime,
        "runtime_reason": runtime_reason,
        "checked_at": time.time(),
    }

    if admitted:
        log.info(
            "NODE ADMITTED host=%s runtime=%s versions=%s",
            host_id, runtime, json.dumps(versions),
        )
    else:
        log.warning(
            "NODE REJECTED host=%s reasons=%s",
            host_id, "; ".join(reasons),
        )

    return admitted, details
