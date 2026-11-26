#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xcelsior Compute Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# This file alone is under Apache-2.0; the rest of the repository is under the
# Business Source License 1.1 (see LICENSE). Anyone may run, audit, change and
# redistribute the benchmark, so anyone can check a listing's numbers.
"""xcelsior-bench: one reproducible GPU benchmark for hosts and renters.

The same file runs in three places, so a renter's numbers and a host's
published numbers come from the same code:

- the worker agent, at admission and at every re-verification, only while the
  host is idle (it reads this file and runs it with ``python -c``, in the agent's
  venv or in the benchmark image);
- a renter, inside their own instance: ``python3 xcelsior_bench.py``;
- the tests, which import the pure functions below.

It prints one JSON document (see ``SCHEMA``). Pass/fail is only for clear
faults, everything else is evidence:

- thermal or power-fault throttling for more than ``THROTTLE_FAULT_SHARE`` of
  the sustained window;
- allocatable VRAM below ``ALLOCATABLE_FAULT_RATIO`` of the VRAM CUDA can
  address (only when ``--advertised-vram-gb`` is given; see
  :func:`allocatable_baseline_gb`).

Two VRAM figures are reported. ``board_vram_gb`` is the card's memory as
``nvidia-smi`` reports ``memory.total``, the same reading a host's listing is
made from. ``total_vram_gb`` is what CUDA can address: the board's memory less
the few hundred MiB the driver reserves for itself.

Throttle reasons are NVML's ``clocksThrottleReasons`` bits. Running at the
software power cap (0x4) is how a GPU under full load normally behaves, so it is
reported (``power_cap_share``) but is not a fault. The fault reasons are the
thermal slowdowns (0x20, 0x40) and the hardware slowdowns (0x8, 0x80).

Needs PyTorch with CUDA. NVML comes from ``pynvml`` when installed, otherwise
from ``nvidia-smi``. With neither, the sustained phase has no samples and the
verdict is ``unknown``, never ``pass``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

BENCH_VERSION = "1.1.0"
SCHEMA = "xcelsior-bench/1"

DEFAULT_SUSTAINED_SECONDS = 300
#: The shortest sustained phase that can pass. A longer run is fine; a shorter
#: one is still reported, but a few seconds under load cannot show that a GPU
#: does not throttle, so its verdict is ``unknown`` unless it already shows a
#: fault. ``XCELSIOR_BENCH_SUSTAINED_SEC`` cannot be used to skip the check.
MIN_SUSTAINED_SECONDS = 300
#: Share of those seconds that must have produced a sample.
MIN_SUSTAINED_SAMPLE_RATIO = 0.9
THROTTLE_FAULT_SHARE = 0.10
ALLOCATABLE_FAULT_RATIO = 0.90
#: How far below the listing CUDA's total may sit and still be the driver's
#: reservation, which every card has and no host can give a renter: 407 MiB of
#: a 6 GB RTX 2060 (6.6 %), about 3 % of a 12 GB RTX 3060.
DRIVER_RESERVE_MAX_RATIO = 0.10

#: NVML clocksThrottleReasons bits.
THROTTLE_BITS = {
    "gpu_idle": 0x1,
    "applications_clocks": 0x2,
    "sw_power_cap": 0x4,
    "hw_slowdown": 0x8,
    "sync_boost": 0x10,
    "sw_thermal": 0x20,
    "hw_thermal": 0x40,
    "hw_power_brake": 0x80,
    "display_clocks": 0x100,
}
THERMAL_FAULT_MASK = THROTTLE_BITS["sw_thermal"] | THROTTLE_BITS["hw_thermal"]
POWER_FAULT_MASK = THROTTLE_BITS["hw_slowdown"] | THROTTLE_BITS["hw_power_brake"]


# ── Pure functions (imported by the tests and the server) ──────────────────


def throttle_reasons(bits: int) -> list[str]:
    """The names of the reasons set in an NVML throttle bitmask."""
    return [name for name, bit in THROTTLE_BITS.items() if bits & bit]


def summarize_sustained(samples: list[dict], duration_s: float) -> dict:
    """Per-second samples → the sustained-load summary the verdict reads.

    Each sample has ``sm_clock_mhz``, ``temp_c``, ``power_w`` and
    ``throttle_bits`` (any may be None when the source did not answer).
    """
    summary: dict = {
        "duration_s": round(duration_s, 1),
        "samples": len(samples),
        "sm_clock_p50_mhz": None,
        "sm_clock_min_mhz": None,
        "sm_clock_max_mhz": None,
        "temp_peak_c": None,
        "temp_p50_c": None,
        "power_avg_w": None,
        "power_peak_w": None,
        "thermal_throttle_share": None,
        "power_fault_share": None,
        "power_cap_share": None,
        "throttle_share_by_reason": {},
    }
    clocks = [s["sm_clock_mhz"] for s in samples if s.get("sm_clock_mhz")]
    temps = [s["temp_c"] for s in samples if s.get("temp_c")]
    powers = [s["power_w"] for s in samples if s.get("power_w")]
    bits = [int(s["throttle_bits"]) for s in samples if s.get("throttle_bits") is not None]
    if clocks:
        summary["sm_clock_p50_mhz"] = round(statistics.median(clocks))
        summary["sm_clock_min_mhz"] = round(min(clocks))
        summary["sm_clock_max_mhz"] = round(max(clocks))
    if temps:
        summary["temp_peak_c"] = max(temps)
        summary["temp_p50_c"] = round(statistics.median(temps), 1)
    if powers:
        summary["power_avg_w"] = round(sum(powers) / len(powers), 1)
        summary["power_peak_w"] = round(max(powers), 1)
    if bits:
        n = len(bits)
        summary["thermal_throttle_share"] = round(
            sum(1 for b in bits if b & THERMAL_FAULT_MASK) / n, 4
        )
        summary["power_fault_share"] = round(sum(1 for b in bits if b & POWER_FAULT_MASK) / n, 4)
        summary["power_cap_share"] = round(
            sum(1 for b in bits if b & THROTTLE_BITS["sw_power_cap"]) / n, 4
        )
        summary["throttle_share_by_reason"] = {
            name: round(sum(1 for b in bits if b & bit) / n, 4)
            for name, bit in THROTTLE_BITS.items()
            if any(b & bit for b in bits)
        }
    return summary


def sustained_too_short(sustained: dict) -> bool:
    """True when the sustained phase is too short, or too sparse, to pass."""
    duration = sustained.get("duration_s") or 0
    samples = sustained.get("samples") or 0
    return duration < MIN_SUSTAINED_SECONDS or samples < (
        MIN_SUSTAINED_SECONDS * MIN_SUSTAINED_SAMPLE_RATIO
    )


def allocatable_baseline_gb(
    advertised_vram_gb: float | None, cuda_visible_vram_gb: float | None
) -> float | None:
    """The VRAM the allocation probe is held to.

    The listing is the board's memory; CUDA addresses that less the driver's
    reservation. Held to the listing, a small card failed for the reservation
    alone (a 6 GB card can give a job about 5.3 GB, 88 %), while the same
    overhead on a 12 GB card passed. Held to what CUDA addresses, it passes,
    and memory another process holds still fails, because CUDA's total does
    not shrink for it. A CUDA total more than ``DRIVER_RESERVE_MAX_RATIO``
    below the listing is not a reservation, so the listing is used instead.
    """
    if not advertised_vram_gb:
        return None
    floor = (1 - DRIVER_RESERVE_MAX_RATIO) * advertised_vram_gb
    if cuda_visible_vram_gb and floor <= cuda_visible_vram_gb < advertised_vram_gb:
        return cuda_visible_vram_gb
    return advertised_vram_gb


def verdict(
    sustained: dict | None,
    allocatable_vram_gb: float | None,
    advertised_vram_gb: float | None,
    cuda_visible_vram_gb: float | None = None,
) -> dict:
    """``pass`` / ``fail`` / ``unknown``, and the faults behind a fail.

    ``unknown`` when the sustained phase took no throttle samples, was shorter
    than ``MIN_SUSTAINED_SECONDS``, or the allocation probe did not run: a
    missing measurement is never a pass. A fault found in a short run still
    fails.
    """
    baseline = allocatable_baseline_gb(advertised_vram_gb, cuda_visible_vram_gb)
    faults: list[str] = []
    unknown: list[str] = []
    if not sustained or sustained.get("thermal_throttle_share") is None:
        unknown.append("sustained_load")
    else:
        if sustained["thermal_throttle_share"] > THROTTLE_FAULT_SHARE:
            faults.append("thermal_throttling")
        if (sustained.get("power_fault_share") or 0) > THROTTLE_FAULT_SHARE:
            faults.append("power_throttling")
        if sustained_too_short(sustained):
            unknown.append("sustained_load_too_short")
    if baseline:
        if not allocatable_vram_gb:
            unknown.append("allocatable_vram")
        elif allocatable_vram_gb < ALLOCATABLE_FAULT_RATIO * baseline:
            faults.append("allocatable_vram_below_listing")
    if faults:
        result = "fail"
    elif unknown:
        result = "unknown"
    else:
        result = "pass"
    return {
        "result": result,
        "faults": faults,
        "unknown": unknown,
        "throttle_fault_share": THROTTLE_FAULT_SHARE,
        "allocatable_fault_ratio": ALLOCATABLE_FAULT_RATIO,
        "allocatable_baseline_gb": baseline,
    }


# ── NVML sampling ──────────────────────────────────────────────────────────


class _Sampler:
    """One GPU's clock, temperature, power and throttle bits, via NVML or nvidia-smi."""

    def __init__(self, index: int):
        self.index = index
        self.source = "none"
        self._handle = None
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.source = "nvml"
        except Exception:
            self._nvml = None
            if self._smi(["clocks.sm"]) is not None:
                self.source = "nvidia-smi"

    def _smi(self, fields: list[str]) -> list[str] | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.index}",
                    f"--query-gpu={','.join(fields)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]

    def sample(self) -> dict:
        if self._nvml is not None and self._handle is not None:
            n, h = self._nvml, self._handle
            return {
                "sm_clock_mhz": _try(lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)),
                "temp_c": _try(lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                "power_w": _try(lambda: n.nvmlDeviceGetPowerUsage(h) / 1000.0),
                "throttle_bits": _try(lambda: n.nvmlDeviceGetCurrentClocksThrottleReasons(h)),
            }
        if self.source == "nvidia-smi":
            # Newer drivers call it clocks_event_reasons; older ones throttle_reasons.
            for field in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
                row = self._smi(["clocks.sm", "temperature.gpu", "power.draw", field])
                if row and len(row) == 4:
                    return {
                        "sm_clock_mhz": _float(row[0]),
                        "temp_c": _float(row[1]),
                        "power_w": _float(row[2]),
                        "throttle_bits": _int(row[3]),
                    }
        return {"sm_clock_mhz": None, "temp_c": None, "power_w": None, "throttle_bits": None}

    def identity(self) -> dict:
        row = self._smi(["uuid", "serial", "driver_version", "name", "memory.total"]) or []
        board_mib = _float(row[4]) if len(row) > 4 else None
        nvml, handle = self._nvml, self._handle
        if board_mib is None and nvml is not None and handle is not None:
            total = _try(lambda: nvml.nvmlDeviceGetMemoryInfo(handle).total)
            board_mib = total / 1024**2 if total else None
        return {
            "gpu_uuid": row[0] if len(row) > 0 else "",
            "serial": row[1] if len(row) > 1 and row[1] not in ("[N/A]", "N/A") else "",
            "driver_version": row[2] if len(row) > 2 else "",
            "nvidia_smi_name": row[3] if len(row) > 3 else "",
            # MiB / 1024, as the agent computes a listing's VRAM from the same field.
            "board_vram_gb": round(board_mib / 1024, 2) if board_mib else None,
        }


def _try(fn):
    try:
        return fn()
    except Exception:
        return None


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: str) -> int | None:
    try:
        return int(value, 16) if str(value).lower().startswith("0x") else int(value)
    except (TypeError, ValueError):
        return None


# ── Measurements (need torch + CUDA) ───────────────────────────────────────


def _allocatable_vram(torch, device) -> dict:
    """The largest single allocation that succeeds, before anything else is held."""
    torch.cuda.empty_cache()
    free_b, _total_b = torch.cuda.mem_get_info(device)
    step = 64 * 1024 * 1024
    size = free_b - (free_b % step)
    best = 0
    while size >= step:
        try:
            block = torch.empty(size, dtype=torch.uint8, device=device)
            del block
            torch.cuda.empty_cache()
            best = size
            break
        except RuntimeError:
            size -= step
    return {
        "allocatable_vram_gb": round(best / 1024**3, 2),
        "free_vram_gb_at_probe": round(free_b / 1024**3, 2),
    }


def _matmul_tflops(torch, device, dtype, n: int = 4096, iters: int = 50) -> float:
    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return round(2 * n**3 * iters / elapsed / 1e12, 2)


def _memory_bandwidth_gbps(torch, device, size_mb: int = 512, iters: int = 20) -> float:
    """Device-to-device copy: each copy reads and writes the buffer once."""
    src = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    dst.copy_(src)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del src, dst
    return round(2 * size_mb / 1024 * iters / elapsed, 1)


def _pcie(torch, device, size_mb: int = 256) -> dict:
    """Pinned host memory, one warm-up, then the median of five copies each way."""
    host = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8).pin_memory()
    dev = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    dev.copy_(host)
    host.copy_(dev)
    torch.cuda.synchronize()
    h2d, d2h = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        host.copy_(dev, non_blocking=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        h2d.append(size_mb / 1024 / (t1 - t0))
        d2h.append(size_mb / 1024 / (t2 - t1))
    del host, dev
    h2d_gbps = round(statistics.median(h2d), 2)
    d2h_gbps = round(statistics.median(d2h), 2)
    return {
        "pcie_h2d_gbps": h2d_gbps,
        "pcie_d2h_gbps": d2h_gbps,
        "pcie_bandwidth_gbps": round((h2d_gbps + d2h_gbps) / 2, 2),
    }


def _sustained(torch, device, sampler: _Sampler, seconds: int) -> tuple[dict, list[dict]]:
    """Full matmul load for ``seconds``, sampling the GPU once a second."""
    a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    samples: list[dict] = []
    start = time.perf_counter()
    next_sample = start + 1.0
    while True:
        now = time.perf_counter()
        if now - start >= seconds:
            break
        torch.mm(a, b)
        if now >= next_sample:
            torch.cuda.synchronize()
            sample = sampler.sample()
            sample["t"] = round(now - start, 1)
            samples.append(sample)
            next_sample += 1.0
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    del a, b
    torch.cuda.empty_cache()
    return summarize_sustained(samples, duration), samples


def run(
    sustained_seconds: int = DEFAULT_SUSTAINED_SECONDS,
    advertised_vram_gb: float | None = None,
    device_index: int = 0,
    keep_samples: bool = False,
) -> dict:
    """Run every measurement and return the report (``SCHEMA``)."""
    started_at = time.time()
    report: dict = {
        "schema": SCHEMA,
        "bench_version": BENCH_VERSION,
        "started_at": started_at,
        "sustained_seconds_requested": sustained_seconds,
        "advertised_vram_gb": advertised_vram_gb,
    }
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return {**report, "error": "no_torch"}
    if not torch.cuda.is_available():
        return {**report, "error": "no_cuda"}

    device = torch.device(f"cuda:{device_index}")
    props = torch.cuda.get_device_properties(device)
    sampler = _Sampler(device_index)
    report.update(
        {
            "gpu_model": props.name,
            "total_vram_gb": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_version": torch.version.cuda or "",
            "torch_version": torch.__version__,
            "telemetry_source": sampler.source,
            **sampler.identity(),
        }
    )
    errors: dict = {}

    def measure(name, fn):
        try:
            return fn()
        except Exception as exc:
            errors[name] = str(exc)[:200]
            return None

    report.update(measure("allocatable_vram", lambda: _allocatable_vram(torch, device)) or {})
    report["fp16_tflops"] = measure(
        "fp16_tflops", lambda: _matmul_tflops(torch, device, torch.float16)
    )
    bf16_ok = bool(_try(torch.cuda.is_bf16_supported))
    report["bf16_tflops"] = (
        measure("bf16_tflops", lambda: _matmul_tflops(torch, device, torch.bfloat16))
        if bf16_ok
        else None
    )
    report["memory_bandwidth_gbps"] = measure(
        "memory_bandwidth", lambda: _memory_bandwidth_gbps(torch, device)
    )
    report.update(measure("pcie", lambda: _pcie(torch, device)) or {})
    result = measure("sustained", lambda: _sustained(torch, device, sampler, sustained_seconds))
    sustained, samples = result if result else (None, [])
    report["sustained"] = sustained
    if keep_samples:
        report["sustained_samples"] = samples
    report["verdict"] = verdict(
        sustained,
        report.get("allocatable_vram_gb"),
        advertised_vram_gb,
        report.get("total_vram_gb"),
    )
    report["errors"] = errors
    report["finished_at"] = time.time()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xcelsior-bench", description="Reproducible GPU benchmark (prints JSON)."
    )
    parser.add_argument(
        "--sustained-seconds",
        type=int,
        default=int(os.environ.get("XCELSIOR_BENCH_SUSTAINED_SEC", DEFAULT_SUSTAINED_SECONDS)),
        help=f"length of the sustained-load phase (default {DEFAULT_SUSTAINED_SECONDS})",
    )
    parser.add_argument(
        "--advertised-vram-gb",
        type=float,
        default=_float(os.environ.get("XCELSIOR_BENCH_ADVERTISED_VRAM_GB", "")),
        help="the listing's VRAM, to check allocatable VRAM against it",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--samples", action="store_true", help="include every per-second sample in the output"
    )
    args = parser.parse_args(argv)
    report = run(
        sustained_seconds=max(0, args.sustained_seconds),
        advertised_vram_gb=args.advertised_vram_gb,
        device_index=args.device,
        keep_samples=args.samples,
    )
    print(json.dumps(report))
    return 0 if "error" not in report else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if sys.argv[0] != "-c" else []))
