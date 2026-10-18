#!/usr/bin/env python3
"""Ensure preset LLM weights exist under staging/2026H2/hf/ (ops — not runtime).

Qwen weights are pre-staged on ASUS (/mnt/storage/models/staging/2026H2/hf/Qwen_*)
and/or B2 per pxl-registry; this script only fetches models that are still missing.

Usage:
  python scripts/fetch_preset_models.py --check            # report status (no downloads)
  python scripts/fetch_preset_models.py                    # fetch any missing doc models
  python scripts/fetch_preset_models.py BAAI/bge-m3        # fetch one model if absent
  python scripts/fetch_preset_models.py --from-b2 Qwen/Qwen3-8B  # pull B2 shards from registry yaml

Credentials: B2_MODEL_SYNC_BUCKET, B2_MODEL_SYNC_KEY_ID, B2_MODEL_SYNC_KEY
(same as pxl-registry / ai-data-factory — source .env.model-sync before running).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serverless.b2_model_sync import default_b2_bucket, download_b2_object, ensure_b2_authorized
from serverless.registry_models import (
    REQUIRED_PRESET_MODELS,
    _load_registry_doc,
    canonical_model_dir,
    find_local_model,
    hf_to_staging_dirname,
    model_roots,
    model_usable,
)


def _download_target_dir(hf_ref: str, *, min_free_gb: float = 8.0) -> Path:
    """Pick the first model store root with enough free space for a download."""
    import shutil

    for root in model_roots():
        try:
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free < int(min_free_gb * (1024**3)):
                continue
            return root / "staging" / "2026H2" / "hf" / hf_to_staging_dirname(hf_ref)
        except OSError:
            continue
    raise RuntimeError(
        f"No writable model store root with ≥{min_free_gb}GB free "
        f"(checked: {', '.join(str(r) for r in model_roots())})"
    )


def _iter_b2_keys(doc: dict) -> list[tuple[str, str]]:
    """(bucket, key) pairs from registry variants."""
    out: list[tuple[str, str]] = []
    bucket = default_b2_bucket()
    for variant in doc.get("variants") or []:
        if not isinstance(variant, dict):
            continue
        storage = variant.get("storage") or {}
        if not isinstance(storage, dict):
            continue
        backend = str(storage.get("backend") or "").lower()
        key = str(storage.get("key") or variant.get("file") or "").strip()
        if not key:
            continue
        if backend in ("b2", "local"):
            out.append((str(storage.get("bucket") or bucket), key))
    return out


def _download_hf(hf_ref: str, target_dir: Path) -> bool:
    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf_bin:
        print(f"ERROR: hf CLI not found — cannot download {hf_ref}", file=sys.stderr)
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {hf_ref} → {target_dir}")
    try:
        subprocess.run(
            [hf_bin, "download", hf_ref, "--local-dir", str(target_dir)],
            check=True,
            timeout=7200,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"ERROR: HF download failed for {hf_ref}: {e}", file=sys.stderr)
        return False
    return model_usable(target_dir)


def _fetch_from_b2(hf_ref: str, *, target_dir: Path | None = None) -> bool:
    _, doc = _load_registry_doc(hf_ref)
    keys = _iter_b2_keys(doc)
    if not keys:
        return False
    target = target_dir or canonical_model_dir(hf_ref)
    target.mkdir(parents=True, exist_ok=True)
    ok = 0
    for bucket, key in keys:
        dest = target / Path(key).name
        if download_b2_object(bucket, key, dest):
            ok += 1
            print(f"  b2: {key}")
    # Sidecars share prefix
    prefix = str(Path(keys[0][1]).parent)
    bucket = keys[0][0]
    for sidecar in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ):
        dest = target / sidecar
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        key = f"{prefix}/{sidecar}"
        if download_b2_object(bucket, key, dest):
            ok += 1
    return model_usable(target)


def fetch_one(hf_ref: str, *, prefer_b2: bool = False) -> dict:
    existing = find_local_model(hf_ref)
    if existing:
        return {
            "hf_ref": hf_ref,
            "status": "present",
            "path": str(existing),
        }

    try:
        target = _download_target_dir(hf_ref)
    except RuntimeError as e:
        return {"hf_ref": hf_ref, "status": "failed", "target": str(canonical_model_dir(hf_ref)), "error": str(e)}
    result = {"hf_ref": hf_ref, "status": "missing", "target": str(target)}

    # B2 first when model-sync creds are available (Qwen shards live on B2).
    try_b2 = prefer_b2 or ensure_b2_authorized()
    if try_b2 and _fetch_from_b2(hf_ref, target_dir=target):
        result["status"] = "fetched_b2"
        result["path"] = str(target)
        return result

    if _download_hf(hf_ref, target):
        result["status"] = "fetched_hf"
        result["path"] = str(target)
        return result

    if not try_b2 and _fetch_from_b2(hf_ref, target_dir=target):
        result["status"] = "fetched_b2"
        result["path"] = str(target)
        return result

    result["status"] = "failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch preset models to disk")
    parser.add_argument("models", nargs="*", help="HF ids (default: all required presets)")
    parser.add_argument("--check", action="store_true", help="Only report status, do not download")
    parser.add_argument("--from-b2", action="store_true", help="Try B2 registry shards before HuggingFace")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    models = list(args.models) if args.models else list(REQUIRED_PRESET_MODELS)
    report: list[dict] = []

    for hf_ref in models:
        if args.check:
            local = find_local_model(hf_ref)
            row = {
                "hf_ref": hf_ref,
                "status": "present" if local else "missing",
                "path": str(local) if local else str(canonical_model_dir(hf_ref)),
            }
        else:
            row = fetch_one(hf_ref, prefer_b2=args.from_b2)
        report.append(row)
        if not args.json:
            print(f"{row['status']:12} {hf_ref}  {row.get('path', '')}")

    if args.json:
        print(json.dumps(report, indent=2))

    missing = [r for r in report if r["status"] in ("missing", "failed")]
    if missing and not args.check:
        print(f"\n{len(missing)} model(s) still missing — see above.", file=sys.stderr)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())