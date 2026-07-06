"""EAGLE-3 speculative decoding validation gate.

Speculative vLLM flags are appended only when rolling acceptance rate ≥
threshold AND measured throughput beats the no-spec baseline on real traffic
samples. Default: speculative decoding off until validated.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

MIN_ACCEPTANCE_RATE = float(os.environ.get("XCELSIOR_EAGLE3_MIN_ACCEPTANCE", "0.75"))
MIN_SAMPLES = int(os.environ.get("XCELSIOR_EAGLE3_MIN_SAMPLES", "5"))
MIN_THROUGHPUT_GAIN = float(os.environ.get("XCELSIOR_EAGLE3_MIN_THROUGHPUT_GAIN", "1.05"))
MAX_SAMPLES = int(os.environ.get("XCELSIOR_EAGLE3_MAX_SAMPLES", "200"))


def _store_path() -> Path:
    raw = os.environ.get("XCELSIOR_EAGLE3_VALIDATION_PATH", "").strip()
    if raw:
        return Path(raw)
    data = os.environ.get("XCELSIOR_DATA_DIR", "/var/lib/xcelsior").strip()
    return Path(data) / "eagle3_validation.json"


def _load_store() -> dict[str, Any]:
    path = _store_path()
    if not path.is_file():
        return {"models": {}, "updated_at": 0.0}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("models"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"models": {}, "updated_at": 0.0}


def _save_store(data: dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = time.time()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def extract_speculative_metrics(usage: dict[str, Any] | None) -> dict[str, float] | None:
    """Parse vLLM/OpenAI usage extensions for speculative acceptance."""
    if not usage or not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    accepted = usage.get("accepted_tokens") or details.get("accepted_tokens")
    rejected = usage.get("rejected_tokens") or details.get("rejected_tokens")
    draft = usage.get("draft_tokens") or details.get("draft_tokens")
    if accepted is None:
        accepted = usage.get("speculative_accepted_tokens")
    if rejected is None:
        rejected = usage.get("speculative_rejected_tokens")
    if draft is None:
        draft = usage.get("speculative_draft_tokens")
    try:
        acc = float(accepted) if accepted is not None else None
        rej = float(rejected) if rejected is not None else 0.0
        drf = float(draft) if draft is not None else None
    except (TypeError, ValueError):
        return None
    if acc is None:
        return None
    total = acc + max(0.0, rej)
    if total <= 0 and drf is not None and drf > 0:
        total = drf
    if total <= 0:
        return None
    rate = acc / total
    out: dict[str, float] = {
        "acceptance_rate": round(rate, 4),
        "accepted_tokens": acc,
        "rejected_tokens": rej,
    }
    if drf is not None:
        out["draft_tokens"] = drf
    return out


def record_speculative_sample(
    model_ref: str,
    *,
    acceptance_rate: float,
    tokens_per_sec: float | None = None,
    baseline_tokens_per_sec: float | None = None,
    source: str = "traffic",
) -> dict[str, Any]:
    """Append one real-traffic speculative observation for ``model_ref``."""
    model_ref = (model_ref or "").strip()
    if not model_ref:
        return {"recorded": False, "reason": "no_model_ref"}
    store = _load_store()
    models = store.setdefault("models", {})
    entry = models.setdefault(
        model_ref,
        {"samples": [], "baseline_tokens_per_sec": None, "validated": False},
    )
    if baseline_tokens_per_sec is not None and baseline_tokens_per_sec > 0:
        entry["baseline_tokens_per_sec"] = float(baseline_tokens_per_sec)
    sample = {
        "ts": time.time(),
        "acceptance_rate": round(float(acceptance_rate), 4),
        "tokens_per_sec": round(float(tokens_per_sec), 2) if tokens_per_sec else None,
        "source": source,
    }
    samples: list[dict[str, Any]] = entry.setdefault("samples", [])
    samples.append(sample)
    if len(samples) > MAX_SAMPLES:
        entry["samples"] = samples[-MAX_SAMPLES:]
    status = validation_status(model_ref, store=store)
    entry["validated"] = bool(status["validated"])
    _save_store(store)
    return {"recorded": True, "validation": status}


def record_speculative_from_usage(
    model_ref: str,
    usage: dict[str, Any] | None,
    *,
    tokens_per_sec: float | None = None,
    baseline_tokens_per_sec: float | None = None,
    source: str = "proxy",
) -> dict[str, Any] | None:
    """Record when upstream usage carries speculative token details."""
    metrics = extract_speculative_metrics(usage)
    if not metrics:
        return None
    return record_speculative_sample(
        model_ref,
        acceptance_rate=metrics["acceptance_rate"],
        tokens_per_sec=tokens_per_sec,
        baseline_tokens_per_sec=baseline_tokens_per_sec,
        source=source,
    )


def validation_status(model_ref: str, *, store: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rolling validation state for a base model."""
    model_ref = (model_ref or "").strip()
    store = store or _load_store()
    entry = (store.get("models") or {}).get(model_ref) or {}
    samples = [s for s in (entry.get("samples") or []) if isinstance(s, dict)]
    rates = [float(s["acceptance_rate"]) for s in samples if s.get("acceptance_rate") is not None]
    tps = [float(s["tokens_per_sec"]) for s in samples if s.get("tokens_per_sec")]
    baseline = entry.get("baseline_tokens_per_sec")
    mean_rate = sum(rates) / len(rates) if rates else 0.0
    mean_tps = sum(tps) / len(tps) if tps else 0.0
    throughput_ok = True
    if baseline and float(baseline) > 0 and tps:
        throughput_ok = mean_tps >= float(baseline) * MIN_THROUGHPUT_GAIN
    elif tps:
        throughput_ok = mean_tps > 0
    validated = (
        len(rates) >= MIN_SAMPLES
        and mean_rate >= MIN_ACCEPTANCE_RATE
        and throughput_ok
    )
    return {
        "model_ref": model_ref,
        "sample_count": len(rates),
        "mean_acceptance_rate": round(mean_rate, 4),
        "mean_tokens_per_sec": round(mean_tps, 2),
        "baseline_tokens_per_sec": baseline,
        "throughput_improved": throughput_ok,
        "min_acceptance_rate": MIN_ACCEPTANCE_RATE,
        "min_samples": MIN_SAMPLES,
        "validated": validated,
    }


def eagle3_enabled_for_model(model_ref: str) -> bool:
    """True when speculative decoding may be enabled for this base model.

    Compatible chat bases ship EAGLE-3 after validation passes (default on).
    Set ``XCELSIOR_VLLM_EAGLE3=0`` to opt out; ``XCELSIOR_VLLM_EAGLE3_FORCE=1`` skips the gate.
    """
    if os.environ.get("XCELSIOR_VLLM_EAGLE3_FORCE", "").lower() in ("1", "true", "yes"):
        return draft_model_for_speculative(model_ref) is not None
    if os.environ.get("XCELSIOR_VLLM_EAGLE3", "1").lower() in ("0", "false", "no"):
        return False
    if not draft_model_for_speculative(model_ref):
        return False
    return bool(validation_status(model_ref).get("validated"))


def draft_model_for_speculative(model_ref: str) -> str | None:
    """Resolve the EAGLE draft model id for a chat base, if any."""
    m = (model_ref or "").lower()
    if "qwen3" in m or "qwen2.5" in m or "qwen2" in m:
        return os.environ.get(
            "XCELSIOR_EAGLE3_DRAFT_QWEN",
            "RedHatAI/Qwen3-8B-speculator.eagle3",
        )
    if "llama-3" in m or "llama3" in m:
        return os.environ.get(
            "XCELSIOR_EAGLE3_DRAFT_LLAMA",
            "RedHatAI/Llama-3.1-8B-Instruct-EAGLE3",
        )
    return None


def speculative_startup_flags(model_ref: str) -> list[str]:
    """vLLM CLI flag tokens to append when speculative decoding is validated."""
    if not eagle3_enabled_for_model(model_ref):
        return []
    draft = draft_model_for_speculative(model_ref)
    if not draft:
        return []
    from serverless.registry_models import resolve_preset_model

    draft_launch = resolve_preset_model(draft).launch_ref
    return ["--speculative-algorithm", "EAGLE3", "--speculative-model", draft_launch]


def reset_validation_store() -> None:
    """Test helper — clear persisted validation samples."""
    path = _store_path()
    if path.is_file():
        path.unlink()