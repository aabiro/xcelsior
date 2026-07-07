"""Published SLO targets and GA status for token SKU endpoints."""

from __future__ import annotations

import os
from typing import Any

KV_CACHE_HIT_TARGET = float(os.environ.get("XCELSIOR_KV_CACHE_HIT_TARGET", "0.30"))
TTFT_P95_TARGET_MS = int(os.environ.get("XCELSIOR_TTFT_P95_TARGET_MS", "3000"))
TOKENS_PER_SEC_TARGET = float(os.environ.get("XCELSIOR_TOKENS_PER_SEC_TARGET", "50"))


def token_endpoint_is_ga(ep: dict) -> bool:
    """Preset chat/embed/rerank endpoints with published token pricing are GA."""
    if str(ep.get("mode") or "") != "preset" or not ep.get("model_ref"):
        return False
    from serverless.openai_proxy import model_task

    return model_task(str(ep.get("model_ref") or "")) in ("chat", "embed", "rerank")


def published_slo_targets() -> dict[str, Any]:
    return {
        "kv_cache_hit_rate_target": KV_CACHE_HIT_TARGET,
        "ttft_p95_ms_target": TTFT_P95_TARGET_MS,
        "tokens_per_sec_target": TOKENS_PER_SEC_TARGET,
    }


def enrich_pricing_with_ga(ep: dict, pricing: dict[str, Any]) -> dict[str, Any]:
    if not pricing.get("token_billing"):
        return pricing
    out = dict(pricing)
    out["ga_status"] = "ga"
    out["slo_targets"] = published_slo_targets()
    return out


def enrich_usage_with_slo(metrics: dict[str, Any]) -> dict[str, Any]:
    targets = published_slo_targets()
    kv = float(metrics.get("kv_cache_hit_rate") or 0)
    ttft = float(metrics.get("ttft_p95_ms") or 0)
    tps = float(metrics.get("tokens_per_sec") or 0)
    return {
        **metrics,
        "slo_targets": targets,
        "slo_status": {
            "kv_cache_hit_met": kv >= targets["kv_cache_hit_rate_target"],
            "ttft_p95_met": ttft <= targets["ttft_p95_ms_target"] if ttft > 0 else None,
            "tokens_per_sec_met": tps >= targets["tokens_per_sec_target"] if tps > 0 else None,
        },
        "cold_start_p50_sec": metrics.get("cold_start_p50_sec", 0),
        "cold_start_p95_sec": metrics.get("cold_start_p95_sec", 0),
    }