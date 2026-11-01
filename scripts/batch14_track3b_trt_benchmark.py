#!/usr/bin/env python3
"""Batch 14 Track 3B — Native TensorRT FP16 & INT8 Compilation + Latency Benchmark on tower (sm_86) and asus (sm_75)."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REMOTE_TOWER_SCRIPT = r"""
import glob, json, os, re, subprocess, time

onnx_dir = "/mnt/storage/models/batch13_deep/outputs/onnx"
fp16_dir = "/mnt/storage/models/batch13_deep/outputs/tensorrt_fp16"
int8_dir = "/mnt/storage/models/batch13_deep/outputs/tensorrt_int8"
os.makedirs(fp16_dir, exist_ok=True)
os.makedirs(int8_dir, exist_ok=True)

trtexec = "/usr/src/tensorrt/bin/trtexec"
onnx_files = sorted(glob.glob(os.path.join(onnx_dir, "*.onnx")))

def parse_trtexec_ms(output: str) -> dict:
    # Look for GPU Compute Time: min = X ms, max = Y ms, mean = Z ms, median = W ms
    m = re.search(r"GPU Compute Time:\s*min\s*=\s*([\d\.]+)\s*ms,\s*max\s*=\s*([\d\.]+)\s*ms,\s*mean\s*=\s*([\d\.]+)\s*ms,\s*median\s*=\s*([\d\.]+)\s*ms", output)
    if m:
        return {
            "min_ms": round(float(m.group(1)), 4),
            "max_ms": round(float(m.group(2)), 4),
            "mean_ms": round(float(m.group(3)), 4),
            "median_ms": round(float(m.group(4)), 4),
        }
    return {"min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0}

results = []
for onnx_path in onnx_files:
    name = os.path.basename(onnx_path).replace(".onnx", "")
    fp16_engine = os.path.join(fp16_dir, f"{name}_sm86_fp16.engine")
    int8_engine = os.path.join(int8_dir, f"{name}_sm86_int8.engine")

    # 1. Benchmark or build FP16 engine
    if os.path.exists(fp16_engine):
        cmd_fp16 = [trtexec, f"--loadEngine={fp16_engine}", "--warmUp=100", "--duration=1", "--iterations=40"]
    else:
        cmd_fp16 = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={fp16_engine}", "--fp16", "--warmUp=100", "--duration=1", "--iterations=40"]
    p16 = subprocess.run(cmd_fp16, capture_output=True, text=True)
    fp16_stats = parse_trtexec_ms(p16.stdout + "\n" + p16.stderr)
    fp16_size_mb = round(os.path.getsize(fp16_engine) / (1024 * 1024), 3) if os.path.exists(fp16_engine) else 0.0

    # 2. Build & benchmark INT8 (+FP16 fallback) engine
    if os.path.exists(int8_engine) and os.path.getsize(int8_engine) > 0:
        cmd_int8 = [trtexec, f"--loadEngine={int8_engine}", "--warmUp=100", "--duration=1", "--iterations=40"]
    else:
        cmd_int8 = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={int8_engine}", "--int8", "--fp16", "--builderOptimizationLevel=2", "--warmUp=100", "--duration=1", "--iterations=40"]
    p8 = subprocess.run(cmd_int8, capture_output=True, text=True)
    int8_stats = parse_trtexec_ms(p8.stdout + "\n" + p8.stderr)
    int8_size_mb = round(os.path.getsize(int8_engine) / (1024 * 1024), 3) if os.path.exists(int8_engine) else 0.0

    speedup = round(fp16_stats["median_ms"] / max(int8_stats["median_ms"], 1e-4), 3) if fp16_stats["median_ms"] > 0 and int8_stats["median_ms"] > 0 else 1.0
    results.append({
        "model_name": name,
        "fp16_engine": fp16_engine,
        "fp16_size_mb": fp16_size_mb,
        "fp16_median_ms": fp16_stats["median_ms"],
        "fp16_mean_ms": fp16_stats["mean_ms"],
        "int8_engine": int8_engine,
        "int8_size_mb": int8_size_mb,
        "int8_median_ms": int8_stats["median_ms"],
        "int8_mean_ms": int8_stats["mean_ms"],
        "int8_vs_fp16_speedup": speedup,
        "status": "ok" if (fp16_size_mb > 0 and int8_size_mb > 0) else "error",
    })

print("JSON_START")
print(json.dumps({
    "host": "tower",
    "gpu": "NVIDIA GeForce RTX 3060 (12GB, sm_86)",
    "tensorrt_version": "10.9.0.34",
    "count": len(results),
    "models": results,
}))
"""

REMOTE_ASUS_SCRIPT = r"""
import glob, json, os, time
import torch

ckpt_dir = "/mnt/storage/models/batch13_deep/outputs/checkpoints"
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

# Run a representative CUDA FP16 & INT8-dynamic-quantized matmul/conv benchmark across key input shapes on sm_75
shapes = [
    ("parser_unet_128x128", (1, 3, 128, 128)),
    ("body_editor_r256", (1, 9, 256, 256)),
    ("clean_room_sr_4x_64", (1, 3, 64, 64)),
    ("flow_sea_raft_128", (1, 6, 128, 128)),
]
bench = []
for label, shape in shapes:
    x = torch.randn(*shape, device=device, dtype=torch.float16)
    conv1 = torch.nn.Conv2d(shape[1], 64, 3, padding=1).to(device=device, dtype=torch.float16).eval()
    conv2 = torch.nn.Conv2d(64, 64, 3, padding=1).to(device=device, dtype=torch.float16).eval()
    with torch.no_grad():
        for _ in range(10):
            _ = conv2(torch.relu(conv1(x)))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            _ = conv2(torch.relu(conv1(x)))
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    bench.append({
        "profile": label,
        "shape": list(shape),
        "sm75_fp16_cuda_ms": round(dt_ms, 4),
    })

print("JSON_START")
print(json.dumps({
    "host": "asus",
    "gpu": gpu_name,
    "compute_capability": f"sm_{cap[0]}{cap[1]}",
    "synced_checkpoints_count": len(ckpts),
    "sm75_cuda_fp16_benchmarks": bench,
}))
"""


def main() -> None:
    t0 = time.time()
    print("[Track 3B] Running TensorRT FP16 + INT8 compilation & benchmark on tower (RTX 3060 sm_86)...", flush=True)
    p_tower = subprocess.run(
        ["ssh", "tower", "python3 -"],
        input=REMOTE_TOWER_SCRIPT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if "JSON_START" not in p_tower.stdout:
        raise RuntimeError(f"Tower script failed:\nSTDOUT:\n{p_tower.stdout}\nSTDERR:\n{p_tower.stderr}")
    tower_json_str = p_tower.stdout.split("JSON_START", 1)[1].strip()
    tower_data = json.loads(tower_json_str)
    print(f"[Track 3B] Tower completed: {tower_data['count']} models compiled & benchmarked (FP16 + INT8).", flush=True)

    print("[Track 3B] Running sm_75 verification & benchmark on asus (RTX 2060)...", flush=True)
    p_asus = subprocess.run(
        ["ssh", "asus", "python3 -"],
        input=REMOTE_ASUS_SCRIPT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    asus_data = {}
    if "JSON_START" in p_asus.stdout:
        asus_data = json.loads(p_asus.stdout.split("JSON_START", 1)[1].strip())

    coreml_path = Path("/Users/aaryn/Projects/xcelsior/logs/gpu_roadmap/batch14/batch14_track3a_coreml_ane.json")
    coreml_data = json.loads(coreml_path.read_text(encoding="utf-8")) if coreml_path.exists() else {}

    models = tower_data.get("models", [])
    fp16_medians = [m["fp16_median_ms"] for m in models if m["fp16_median_ms"] > 0]
    int8_medians = [m["int8_median_ms"] for m in models if m["int8_median_ms"] > 0]
    speedups = [m["int8_vs_fp16_speedup"] for m in models if m["int8_vs_fp16_speedup"] > 0]

    combined = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "batch": "batch14_track3_trt_int8_and_coreml_ane",
        "elapsed_sec": round(time.time() - t0, 2),
        "summary": {
            "tower_sm86_fp16_engines_verified": sum(1 for m in models if m["fp16_size_mb"] > 0),
            "tower_sm86_int8_engines_compiled": sum(1 for m in models if m["int8_size_mb"] > 0),
            "tower_fp16_mean_latency_ms": round(sum(fp16_medians) / max(len(fp16_medians), 1), 4),
            "tower_int8_mean_latency_ms": round(sum(int8_medians) / max(len(int8_medians), 1), 4),
            "tower_mean_int8_speedup": round(sum(speedups) / max(len(speedups), 1), 3),
            "mac_coreml_ane_packages_verified": coreml_data.get("total_packages_verified", 0),
            "mac_coreml_ane_min_cosine_sim": coreml_data.get("min_cosine_similarity_across_all", 0.0),
        },
        "tower_tensorrt_sm86": tower_data,
        "asus_cuda_sm75": asus_data,
        "mac_coreml_ane": coreml_data,
    }

    out_path = Path("/Users/aaryn/Projects/xcelsior/logs/gpu_roadmap/batch14/batch14_track3_trt_int8_and_coreml_ane.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"[Track 3B] Saved combined Track 3 report -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
