# Xcelsior NVML Telemetry — Direct GPU monitoring via pynvml
#
# Per REPORT_FEATURE_1.md §"Lightweight Monitoring with NVML":
#   "The platform must adopt the NVIDIA Management Library (NVML) as its
#    primary telemetry source... avoid calling the nvidia-smi binary to
#    reduce overhead, instead using Python bindings to interact with
#    libnvidia-ml.so directly."
#
# Collects at configurable intervals (default 10s per report's 10–15s):
#   - GPU utilization (SM active %)
#   - Memory utilization (allocated + reserved for fragmentation detection)
#   - Thermal data (current + rolling average)
#   - Power consumption (wattage)
#   - PCIe bandwidth (actual throughput, not just link info)
#   - GPU serial number (for anti-spoofing PCI-ID check)
#   - ECC memory errors (volatile uncorrectable)

import logging
import os
import time
from collections import deque
from typing import Optional

log = logging.getLogger("xcelsior")

# ── NVML availability flag ────────────────────────────────────────────
# pynvml is optional — falls back to nvidia-smi subprocess on hosts
# without it (e.g., dev machines, CI).

_nvml_available = False
_nvml_initialized = False

try:
    import pynvml  # Provided by nvidia-ml-py
    _nvml_available = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    log.debug("nvidia-ml-py not installed — GPU telemetry will use nvidia-smi fallback")


# ── Thermal rolling average ──────────────────────────────────────────

_thermal_history: dict[int, deque] = {}  # gpu_index → recent temp readings
THERMAL_HISTORY_SIZE = int(os.environ.get("XCELSIOR_THERMAL_HISTORY_SIZE", "60"))


def _get_thermal_avg(gpu_index: int, current_temp: float) -> float:
    """Maintain a rolling average of GPU temperature readings."""
    if gpu_index not in _thermal_history:
        _thermal_history[gpu_index] = deque(maxlen=THERMAL_HISTORY_SIZE)
    _thermal_history[gpu_index].append(current_temp)
    return round(sum(_thermal_history[gpu_index]) / len(_thermal_history[gpu_index]), 1)


# ── NVML Lifecycle ────────────────────────────────────────────────────

def nvml_init() -> bool:
    """Initialize NVML. Call once at startup. Safe to call multiple times."""
    global _nvml_initialized
    if not _nvml_available:
        return False
    if _nvml_initialized:
        return True
    try:
        pynvml.nvmlInit()
        _nvml_initialized = True
        driver = pynvml.nvmlSystemGetDriverVersion()
        log.info("NVML initialized — driver %s", driver)
        return True
    except Exception as e:
        log.warning("NVML init failed: %s — falling back to nvidia-smi", e)
        return False


def nvml_shutdown():
    """Shutdown NVML. Call at program exit."""
    global _nvml_initialized
    if _nvml_initialized and _nvml_available:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        _nvml_initialized = False


def is_nvml_available() -> bool:
    """Check if NVML is initialized and ready for queries."""
    return _nvml_initialized


# ── Core Telemetry Collection ─────────────────────────────────────────

def get_device_count() -> int:
    """Return the number of NVIDIA GPUs visible to NVML."""
    if not _nvml_initialized:
        return 0
    try:
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        return 0


def get_gpu_serial(handle) -> str:
    """Get GPU serial number for PCI-ID anti-spoofing check.

    Per REPORT_FEATURE_1.md: "GPU Model/Serial — NVML/PCI-ID check —
    Prevents spoofing lower-tier cards."
    """
    try:
        return pynvml.nvmlDeviceGetSerial(handle)
    except Exception:
        # Some consumer cards don't expose serial via NVML
        try:
            return pynvml.nvmlDeviceGetUUID(handle)
        except Exception:
            return ""


def get_gpu_pci_info(handle) -> dict:
    """Get PCI bus info for hardware identity verification."""
    try:
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        return {
            "bus_id": pci.busId.decode() if isinstance(pci.busId, bytes) else str(pci.busId),
            "device_id": hex(pci.pciDeviceId) if hasattr(pci, "pciDeviceId") else "",
            "subsystem_id": hex(pci.pciSubSystemId) if hasattr(pci, "pciSubSystemId") else "",
        }
    except Exception:
        return {"bus_id": "", "device_id": "", "subsystem_id": ""}


def collect_gpu_telemetry(gpu_index: int = 0) -> Optional[dict]:
    """Collect comprehensive telemetry for a single GPU via NVML.

    Returns dict with all metrics prescribed by REPORT_FEATURE_1.md:
    - GPU utilization (SM active %)
    - Memory: total, used, free (bytes) + utilization %
    - Memory fragmentation: allocated vs reserved (via torch if available)
    - Thermal: current temp + rolling average
    - Power: current draw (watts) + limit
    - PCIe: throughput TX/RX (bytes/s), link gen, link width
    - Identity: model name, serial/UUID, PCI bus ID
    - ECC: volatile uncorrectable errors
    """
    if not _nvml_initialized:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception as e:
        log.debug("NVML: cannot get handle for GPU %d: %s", gpu_index, e)
        return None

    result = {"index": gpu_index, "timestamp": time.time()}

    # ── GPU Identity (REPORT_FEATURE_1 §Hardware ID) ──────────────────
    try:
        result["gpu_model"] = pynvml.nvmlDeviceGetName(handle)
        if isinstance(result["gpu_model"], bytes):
            result["gpu_model"] = result["gpu_model"].decode()
    except Exception:
        result["gpu_model"] = ""

    result["serial"] = get_gpu_serial(handle)
    result["pci_info"] = get_gpu_pci_info(handle)

    # ── GPU Utilization (REPORT_FEATURE_1 §SM active %) ───────────────
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        result["utilization"] = util.gpu      # SM active %
        result["memory_util"] = util.memory   # Memory controller %
    except Exception:
        result["utilization"] = 0
        result["memory_util"] = 0

    # ── Memory Info (REPORT_FEATURE_1 §nvmlDeviceGetMemoryInfo) ───────
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result["memory_total_bytes"] = mem.total
        result["memory_used_bytes"] = mem.used
        result["memory_free_bytes"] = mem.free
        result["memory_total_gb"] = round(mem.total / (1024**3), 2)
        result["memory_used_gb"] = round(mem.used / (1024**3), 2)
        result["memory_free_gb"] = round(mem.free / (1024**3), 2)
    except Exception:
        result["memory_total_bytes"] = 0
        result["memory_used_bytes"] = 0
        result["memory_free_bytes"] = 0
        result["memory_total_gb"] = 0
        result["memory_used_gb"] = 0
        result["memory_free_gb"] = 0

    # ── Memory Fragmentation (allocated vs reserved via torch) ────────
    # Per REPORT_FEATURE_1: track torch.cuda.memory_allocated() and
    # torch.cuda.memory_reserved() for fragmentation detection
    try:
        import torch
        if torch.cuda.is_available():
            result["torch_memory_allocated_bytes"] = torch.cuda.memory_allocated(gpu_index)
            result["torch_memory_reserved_bytes"] = torch.cuda.memory_reserved(gpu_index)
            reserved = result["torch_memory_reserved_bytes"]
            allocated = result["torch_memory_allocated_bytes"]
            if reserved > 0:
                result["memory_fragmentation_pct"] = round(
                    (1.0 - allocated / reserved) * 100, 1
                )
            else:
                result["memory_fragmentation_pct"] = 0.0
        else:
            result["torch_memory_allocated_bytes"] = 0
            result["torch_memory_reserved_bytes"] = 0
            result["memory_fragmentation_pct"] = 0.0
    except ImportError:
        result["torch_memory_allocated_bytes"] = 0
        result["torch_memory_reserved_bytes"] = 0
        result["memory_fragmentation_pct"] = 0.0

    # ── Thermal (REPORT_FEATURE_1 §current + average temperatures) ────
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        result["temperature_c"] = temp
        result["temperature_avg_c"] = _get_thermal_avg(gpu_index, temp)
    except Exception:
        result["temperature_c"] = 0
        result["temperature_avg_c"] = 0.0

    # ── Power (REPORT_FEATURE_1 §wattage draw) ───────────────────────
    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
        result["power_draw_w"] = round(power_mw / 1000.0, 1)
    except Exception:
        result["power_draw_w"] = 0

    try:
        limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
        result["power_limit_w"] = round(limit_mw / 1000.0, 1)
    except Exception:
        result["power_limit_w"] = 0

    # ── PCIe (REPORT_FEATURE_1 §bandwidth bottleneck detection) ───────
    try:
        gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
        width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        result["pcie_gen"] = gen
        result["pcie_width"] = width
    except Exception:
        result["pcie_gen"] = 0
        result["pcie_width"] = 0

    # PCIe throughput (actual bytes/s, not just link info)
    try:
        tx_bytes = pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
        )
        rx_bytes = pynvml.nvmlDeviceGetPcieThroughput(
            handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
        )
        result["pcie_tx_mb_s"] = round(tx_bytes / 1024, 2)   # KB/s → MB/s
        result["pcie_rx_mb_s"] = round(rx_bytes / 1024, 2)
    except Exception:
        result["pcie_tx_mb_s"] = 0
        result["pcie_rx_mb_s"] = 0

    # ── ECC Memory Errors ─────────────────────────────────────────────
    try:
        ecc_single = pynvml.nvmlDeviceGetTotalEccErrors(
            handle,
            pynvml.NVML_SINGLE_BIT_ECC,
            pynvml.NVML_VOLATILE_ECC,
        )
        ecc_double = pynvml.nvmlDeviceGetTotalEccErrors(
            handle,
            pynvml.NVML_DOUBLE_BIT_ECC,
            pynvml.NVML_VOLATILE_ECC,
        )
        result["ecc_errors_single"] = ecc_single
        result["ecc_errors_double"] = ecc_double
        result["memory_errors"] = ecc_single + ecc_double
    except Exception:
        result["ecc_errors_single"] = 0
        result["ecc_errors_double"] = 0
        result["memory_errors"] = 0

    # ── Compute Capability ────────────────────────────────────────────
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        result["compute_capability"] = f"{major}.{minor}"
    except Exception:
        result["compute_capability"] = ""

    # ── Driver & CUDA Version ─────────────────────────────────────────
    try:
        result["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(result["driver_version"], bytes):
            result["driver_version"] = result["driver_version"].decode()
    except Exception:
        result["driver_version"] = ""

    try:
        result["cuda_version"] = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        # Convert int version to string (e.g., 12040 → "12.4")
        if isinstance(result["cuda_version"], int):
            major = result["cuda_version"] // 1000
            minor = (result["cuda_version"] % 1000) // 10
            result["cuda_version"] = f"{major}.{minor}"
    except Exception:
        result["cuda_version"] = ""

    return result


def collect_all_gpus() -> list[dict]:
    """Collect telemetry for all GPUs on the system.

    Returns list of telemetry dicts, one per GPU.
    Falls back to nvidia-smi subprocess if NVML is not available.
    """
    if not _nvml_initialized:
        return _fallback_nvidia_smi()

    count = get_device_count()
    if count == 0:
        return []

    results = []
    for i in range(count):
        data = collect_gpu_telemetry(i)
        if data:
            results.append(data)
    return results


# ── nvidia-smi Fallback ──────────────────────────────────────────────
# Used when pynvml is not installed (dev machines, CI, etc.)

def _fallback_nvidia_smi() -> list[dict]:
    """Query nvidia-smi as subprocess fallback.

    This is the legacy path — per REPORT_FEATURE_1.md, NVML bindings
    are preferred to reduce overhead.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,"
                "pcie.link.gen.current,pcie.link.width.current,"
                "power.draw,temperature.gpu,name,serial,uuid,"
                "memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 13:
                total_mb = float(parts[10]) if parts[10] not in ("[N/A]", "") else 0
                used_mb = float(parts[11]) if parts[11] not in ("[N/A]", "") else 0
                free_mb = float(parts[12]) if parts[12] not in ("[N/A]", "") else 0
                temp = float(parts[6]) if parts[6] not in ("[N/A]", "") else 0
                gpus.append({
                    "index": int(parts[0]),
                    "utilization": float(parts[1]) if parts[1] not in ("[N/A]", "") else 0,
                    "memory_util": float(parts[2]) if parts[2] not in ("[N/A]", "") else 0,
                    "pcie_gen": parts[3],
                    "pcie_width": parts[4],
                    "power_draw_w": float(parts[5]) if parts[5] not in ("[N/A]", "") else 0,
                    "temperature_c": temp,
                    "temperature_avg_c": _get_thermal_avg(int(parts[0]), temp),
                    "gpu_model": parts[7],
                    "serial": parts[8] if parts[8] not in ("[N/A]", "") else "",
                    "uuid": parts[9],
                    "memory_total_gb": round(total_mb / 1024, 2),
                    "memory_used_gb": round(used_mb / 1024, 2),
                    "memory_free_gb": round(free_mb / 1024, 2),
                    "memory_total_bytes": int(total_mb * 1024 * 1024),
                    "memory_used_bytes": int(used_mb * 1024 * 1024),
                    "memory_free_bytes": int(free_mb * 1024 * 1024),
                    "pcie_tx_mb_s": 0,
                    "pcie_rx_mb_s": 0,
                    "memory_errors": 0,
                    "compute_capability": "",
                    "driver_version": "",
                    "cuda_version": "",
                    "torch_memory_allocated_bytes": 0,
                    "torch_memory_reserved_bytes": 0,
                    "memory_fragmentation_pct": 0.0,
                    "pci_info": {"bus_id": "", "device_id": "", "subsystem_id": ""},
                    "timestamp": time.time(),
                })
        return gpus

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []


# ── Convenience: GPU Info for Host Registration ──────────────────────

def get_gpu_info_nvml() -> Optional[dict]:
    """Get GPU model, total VRAM, free VRAM via NVML for host registration.

    Drop-in replacement for worker_agent.get_gpu_info() but uses NVML
    instead of nvidia-smi subprocess.
    """
    if not _nvml_initialized:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_model": name,
            "total_vram_gb": round(mem.total / (1024**3), 2),
            "free_vram_gb": round(mem.free / (1024**3), 2),
            "serial": get_gpu_serial(handle),
            "pci_info": get_gpu_pci_info(handle),
        }
    except Exception as e:
        log.debug("NVML get_gpu_info failed: %s", e)
        return None


# ── Convenience: Verification Report ─────────────────────────────────

def build_verification_report(gpu_index: int = 0) -> Optional[dict]:
    """Build a complete verification report for verification.py to consume.

    This replaces the torch-based run_compute_benchmark() approach for
    the identity/VRAM/thermal/PCIe checks. The compute benchmark
    (TFLOPS matmul) still requires torch and runs separately.
    """
    data = collect_gpu_telemetry(gpu_index)
    if not data:
        return None

    return {
        # Identity
        "gpu_model": data.get("gpu_model", ""),
        "serial": data.get("serial", ""),
        "pci_info": data.get("pci_info", {}),

        # VRAM (nvmlDeviceGetMemoryInfo per report)
        "total_vram_gb": data.get("memory_total_gb", 0),
        "free_vram_gb": data.get("memory_free_gb", 0),

        # Thermal
        "gpu_temp_celsius": data.get("temperature_c", 0),
        "gpu_temp_avg_celsius": data.get("temperature_avg_c", 0),

        # PCIe
        "pcie_bandwidth_gbps": 0,  # Requires active transfer benchmark
        "pcie_gen": data.get("pcie_gen", 0),
        "pcie_width": data.get("pcie_width", 0),
        "pcie_tx_mb_s": data.get("pcie_tx_mb_s", 0),
        "pcie_rx_mb_s": data.get("pcie_rx_mb_s", 0),

        # CUDA / Driver
        "compute_capability": data.get("compute_capability", ""),
        "cuda_version": data.get("cuda_version", ""),
        "driver_version": data.get("driver_version", ""),

        # Memory fragmentation
        "torch_memory_allocated_bytes": data.get("torch_memory_allocated_bytes", 0),
        "torch_memory_reserved_bytes": data.get("torch_memory_reserved_bytes", 0),
        "memory_fragmentation_pct": data.get("memory_fragmentation_pct", 0),

        # ECC
        "memory_errors": data.get("memory_errors", 0),
    }
