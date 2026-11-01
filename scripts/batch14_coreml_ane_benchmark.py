#!/usr/bin/env python3
"""Batch 14 Track 3A — Mac Apple Silicon CoreML ANE (CPU_AND_NE) Latency & Numerical Parity Benchmark."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import coremltools as ct
import numpy as np


def _build_dummy_inputs(spec) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    inputs: dict[str, np.ndarray] = {}
    for inp in spec.description.input:
        name = inp.name
        if inp.type.WhichOneof("Type") == "multiArrayType":
            shape = [int(d) if int(d) > 0 else 1 for d in inp.type.multiArrayType.shape]
            if not shape:
                shape = [1, 3, 128, 128]
            inputs[name] = rng.standard_normal(shape, dtype=np.float32)
    return inputs


def benchmark_mlpackage(pkg_path: Path) -> dict:
    t0 = time.time()
    try:
        model_ane = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        model_cpu = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = model_ane.get_spec()
        inputs = _build_dummy_inputs(spec)

        # Warmup
        out_ane = model_ane.predict(inputs)
        out_cpu = model_cpu.predict(inputs)

        latencies_ms = []
        for _ in range(5):
            ts = time.perf_counter()
            model_ane.predict(inputs)
            latencies_ms.append((time.perf_counter() - ts) * 1000.0)

        first_key = next(iter(out_ane.keys()))
        arr_ane = np.asarray(out_ane[first_key], dtype=np.float64).ravel()
        arr_cpu = np.asarray(out_cpu[first_key], dtype=np.float64).ravel()
        max_abs_diff = float(np.max(np.abs(arr_ane - arr_cpu))) if arr_ane.size > 0 else 0.0
        denom = float(np.linalg.norm(arr_ane) * np.linalg.norm(arr_cpu))
        cosine_sim = float(np.dot(arr_ane, arr_cpu) / denom) if denom > 1e-12 else 1.0

        size_bytes = sum(f.stat().st_size for f in pkg_path.rglob("*") if f.is_file())
        return {
            "model": pkg_path.name,
            "path": str(pkg_path),
            "status": "verified",
            "compute_unit": "CPU_AND_NE",
            "size_mb": round(size_bytes / (1024 * 1024), 3),
            "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
            "output_shapes": {k: list(np.asarray(v).shape) for k, v in out_ane.items()},
            "median_latency_ms": round(float(np.median(latencies_ms)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
            "ane_vs_cpu_max_abs_diff": round(max_abs_diff, 6),
            "ane_vs_cpu_cosine_similarity": round(cosine_sim, 6),
            "load_and_bench_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:
        return {
            "model": pkg_path.name,
            "path": str(pkg_path),
            "status": "failed",
            "error": str(exc),
        }


def main() -> None:
    t0 = time.time()
    coreml_dir = Path("/Users/aaryn/Projects/xcelsior/models/batch13_deep/coreml")
    ios_dir = Path("/Users/aaryn/Projects/pixelspark-ios/Models/CoreML")

    batch13_pkgs = sorted(coreml_dir.glob("*.mlpackage"))
    ios_special_pkgs = [
        ios_dir / "BodyEditor_r128_v2_fp16.mlpackage",
        ios_dir / "BodyEditor_r256_v1_fp16.mlpackage",
        ios_dir / "CleanRoomMobileSR_4x_fp16.mlpackage",
        ios_dir / "TinySceneDegradation_fp16.mlpackage",
    ]
    ios_special_pkgs = [p for p in ios_special_pkgs if p.exists()]

    results_batch13 = []
    for pkg in batch13_pkgs:
        res = benchmark_mlpackage(pkg)
        results_batch13.append(res)
        print(
            f"[CoreML ANE] {pkg.name}: {res['status']} "
            f"lat={res.get('median_latency_ms')}ms cos={res.get('ane_vs_cpu_cosine_similarity')}",
            flush=True,
        )

    results_ios = []
    for pkg in ios_special_pkgs:
        res = benchmark_mlpackage(pkg)
        results_ios.append(res)
        print(
            f"[iOS CoreML ANE] {pkg.name}: {res['status']} "
            f"lat={res.get('median_latency_ms')}ms cos={res.get('ane_vs_cpu_cosine_similarity')}",
            flush=True,
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.time() - t0, 2),
        "batch13_coreml_count": len(results_batch13),
        "batch13_verified_count": sum(1 for r in results_batch13 if r["status"] == "verified"),
        "ios_special_count": len(results_ios),
        "ios_verified_count": sum(1 for r in results_ios if r["status"] == "verified"),
        "median_latency_ms_across_models": round(
            float(np.median([r["median_latency_ms"] for r in results_batch13 if "median_latency_ms" in r])), 3
        ),
        "min_cosine_similarity": round(
            float(min([r["ane_vs_cpu_cosine_similarity"] for r in results_batch13 if "ane_vs_cpu_cosine_similarity" in r])),
            6,
        ),
        "batch13_models": results_batch13,
        "pixelspark_ios_models": results_ios,
    }
    out_path = Path("/Users/aaryn/Projects/xcelsior/logs/gpu_roadmap/batch14/batch14_track3a_coreml_ane.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Also copy into pixelspark-ios/Models/CoreML/
    (ios_dir / "batch14_coreml_ane_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "batch13_verified": f"{report['batch13_verified_count']}/{report['batch13_coreml_count']}",
        "ios_verified": f"{report['ios_verified_count']}/{report['ios_special_count']}",
        "median_latency_ms": report["median_latency_ms_across_models"],
        "min_cosine_similarity": report["min_cosine_similarity"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

