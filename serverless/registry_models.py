# Xcelsior — Resolve preset model refs to on-disk paths via pxl-registry layout.
# Qwen and other presets are pre-staged under /mnt/storage/models/staging/2026H2/hf/
# (local disk and/or B2 per registry variant keys). Runtime never downloads; ops uses
# scripts/fetch_preset_models.py only when a model is missing.

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("xcelsior.serverless.registry_models")

_REGISTRY_ROOT = Path(
    os.environ.get("XCELSIOR_PXL_REGISTRY_ROOT", "/mnt/storage/projects/pxl-registry")
)

# HF hub id → pxl-registry model yaml stem (without .yaml).
_HF_TO_REGISTRY_ID: dict[str, str] = {
    "qwen/qwen3-8b": "store_llm_qwen_qwen3_8b",
    "baai/bge-m3": "store_llm_baai_bge_m3",
    "baai/bge-reranker-v2-m3": "store_llm_baai_bge_reranker_v2_m3",
    "meta-llama/llama-3.1-8b-instruct": "store_llm_meta_llama_llama_3_1_8b_instruct",
    "meta-llama/llama-3.3-70b-instruct": "store_llm_meta_llama_llama_3_3_70b_instruct",
}

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")

# Preset model SKUs — paths resolved from pxl-registry / local model store.
DOC_PRESET_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-8B",
    "BAAI/bge-m3",  # embeddings preset (UPGRADE_PLAN §xcelsior)
    "BAAI/bge-reranker-v2-m3",  # rerank preset (RAG companion)
)
# Support weights required at worker start (not user-selectable presets).
DOC_SUPPORT_MODELS: tuple[str, ...] = (
    "RedHatAI/Qwen3-8B-speculator.eagle3",
)
REQUIRED_PRESET_MODELS: tuple[str, ...] = DOC_PRESET_MODELS + DOC_SUPPORT_MODELS


def model_roots() -> list[Path]:
    raw = os.environ.get(
        "XCELSIOR_MODEL_STORE_ROOTS",
        "/mnt/storage/models,/mnt/storage",
    )
    return [Path(p) for p in raw.split(",") if p.strip()]


@dataclass(frozen=True)
class ResolvedPresetModel:
    """Launch-time model resolution for vLLM presets."""

    hf_ref: str
    launch_ref: str
    registry_id: str | None = None
    source: str = "hf"  # local | hf
    local_path: str | None = None


def _normalize_hf(hf_ref: str) -> str:
    return (hf_ref or "").strip()


def hf_to_staging_dirname(hf_ref: str) -> str:
    """Canonical staging folder name: Qwen/Qwen3-8B → Qwen_Qwen3-8B."""
    return hf_ref.replace("/", "_")


def canonical_model_dir(hf_ref: str) -> Path:
    """Expected on-disk path for a preset HF id under the primary model store root."""
    roots = model_roots()
    base = roots[0] if roots else Path("/mnt/storage/models")
    return base / "staging" / "2026H2" / "hf" / hf_to_staging_dirname(hf_ref)


def _registry_yaml_path(registry_id: str) -> Path | None:
    models_dir = _REGISTRY_ROOT / "registry" / "models"
    if not models_dir.is_dir():
        return None
    direct = models_dir / f"{registry_id}.yaml"
    if direct.is_file():
        return direct
    for path in models_dir.glob("*.yaml"):
        if path.stem == registry_id:
            return path
    return None


def _load_registry_doc(hf_ref: str) -> tuple[str | None, dict[str, Any]]:
    rid = _HF_TO_REGISTRY_ID.get(hf_ref.lower())
    if not rid:
        safe = _SAFE.sub("_", hf_ref.replace("/", "_").lower())
        candidate = _registry_yaml_path(f"store_llm_{safe}")
        if candidate:
            rid = candidate.stem
    if not rid:
        return None, {}
    yaml_path = _registry_yaml_path(rid)
    if not yaml_path:
        return rid, {}
    try:
        with yaml_path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        return rid, doc if isinstance(doc, dict) else {}
    except Exception as e:
        log.debug("registry yaml read failed for %s: %s", rid, e)
        return rid, {}


def model_usable(model_dir: Path) -> bool:
    """True when a vLLM/HF layout looks loadable."""
    if not model_dir.is_dir():
        return False
    if (model_dir / "config.json").is_file():
        if any(model_dir.glob("*.safetensors")):
            return True
        if (model_dir / "model.safetensors").is_file():
            return True
        if (model_dir / "pytorch_model.bin").is_file():
            return True
    return any(model_dir.glob("*.safetensors"))


def _variant_local_dir(rel_file: str) -> Path | None:
    """Map a registry variant file key to the first existing directory on disk."""
    rel = rel_file.strip().lstrip("/")
    if not rel:
        return None
    parent = Path(rel).parent
    for root in model_roots():
        for candidate in (root / parent, root / "models" / parent):
            if candidate.is_dir() and model_usable(candidate):
                return candidate
    return None


def _resolve_local_from_registry(doc: dict[str, Any]) -> Path | None:
    variants = doc.get("variants") or []
    if not isinstance(variants, list):
        return None
    dirs: list[Path] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        storage = variant.get("storage") or {}
        key = ""
        if isinstance(storage, dict):
            key = str(storage.get("key") or variant.get("file") or "")
        else:
            key = str(variant.get("file") or "")
        found = _variant_local_dir(key)
        if found and found not in dirs:
            dirs.append(found)
    for d in dirs:
        if model_usable(d):
            return d
    return None


def find_local_model(hf_ref: str) -> Path | None:
    """Locate weights on disk: registry variant paths, then canonical staging dir."""
    ref = _normalize_hf(hf_ref)
    _, doc = _load_registry_doc(ref)
    if doc:
        local = _resolve_local_from_registry(doc)
        if local:
            return local

    override = os.environ.get(f"XCELSIOR_MODEL_PATH_{ref.replace('/', '__')}")
    if override:
        p = Path(override)
        if model_usable(p):
            return p

    canonical = canonical_model_dir(ref)
    if model_usable(canonical):
        return canonical

    for root in model_roots():
        alt = root / "staging" / "2026H2" / "hf" / hf_to_staging_dirname(ref)
        if model_usable(alt):
            return alt
    return None


def resolve_preset_model(hf_ref: str) -> ResolvedPresetModel:
    """Resolve HF model id to a vLLM --model path (local dir or HF hub id)."""
    ref = _normalize_hf(hf_ref)
    registry_id, _doc = _load_registry_doc(ref)
    local = find_local_model(ref)
    if local:
        return ResolvedPresetModel(
            hf_ref=ref,
            launch_ref=str(local),
            registry_id=registry_id,
            source="local",
            local_path=str(local),
        )
    return ResolvedPresetModel(
        hf_ref=ref,
        launch_ref=ref,
        registry_id=registry_id,
        source="hf",
    )


def ensure_preset_model_local(hf_ref: str, *, allow_b2: bool = True) -> ResolvedPresetModel:
    """Resolve preset model; does not download — weights must be on disk already."""
    _ = allow_b2  # legacy kwarg; downloads are ops-only (scripts/fetch_preset_models.py)
    return resolve_preset_model(hf_ref)


def launch_ref_for_preset(hf_ref: str) -> str:
    """Convenience: vLLM --model value for a preset HF id."""
    return resolve_preset_model(hf_ref).launch_ref