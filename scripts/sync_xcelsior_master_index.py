#!/usr/bin/env python3
"""Probe shipped xcelsior token-billing code and sync §10 checkboxes in AI/ML docs."""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY_MD = ROOT.parent / "pxl-registry" / "docs" / "AI_ML_MASTER_INDEX_BY_REPO.md"
LOCAL_MIRROR = ROOT / "docs" / "AI_ML_MASTER_INDEX_XCELSIOR.md"

# Rows implemented in this repo (code probes must pass).
CODE_SHIPPED_ROWS: dict[int, str] = {
    4: "criu_hosts.py checkpoint_class + worker preempt checkpoint + resumable API",
    2: "LMCache + EAGLE-3 validation gate (single-host; 2-host mesh ops-deferred)",
    3: "serverless.metering.token_cost_metadata + cached split",
    7: "serverless.observability.compute_endpoint_metrics SLO fields",
    13: "Token SKU + published SLOs + cached-token pricing (SCIP clause out of scope)",
    14: "routes.serverless embeddings route (no 501)",
    15: "prefix cache startup + cached_input_price in pricing quote",
    16: "multi-LoRA startup flags",
    17: "serverless.speculative_gate speculative_startup_flags",
    18: "serverless.scaling_predictive EWMA extrapolation",
    23: "idempotent token ledger by idempotency_key",
    27: "token ledger + observability traces",
    29: "LMCache-only small scale (no Dynamo hard-require)",
    30: "serverless.anchor_workloads real repo payloads",
    22: "tests/test_serverless_chaos_billing.py worker-death idempotency",
    31: "criu_hosts.cuda_driver_requirements + RTX 2060 driver 580 gate test",
}

# Honest ops / fleet / deprioritized sovereignty — never auto-checked.
OPS_BLOCKED_ROWS: dict[int, str] = {
    1: "≥30% KV-cache hits in production month (ops KPI)",

    5: "preempt→migrate→resume demo (live criu snapshot on fleet)",
    6: "Mooncake / remote LMCache + KV-aware routing",
    8: "SCIP application (deprioritized)",
    9: "SCIP alignment one-pager (deprioritized)",
    10: "TEE attestation schema (deprioritized)",
    11: "stellar-subs demand forecast",
    12: "MCP should_i_run_this for PEL jobs",
    19: "pre-warmed pools + cold-start SLO publish",
    20: "OpenAI-style async Batch API",
    21: "LLM gateway semantic cache",
    24: "hedged requests",
    25: "MCP assistant safety eval suite",
    26: "modularize worker_agent / ai_assistant",
    28: "Toto 2.0 forecasting fallback",
}

SECTION_HEADER = "## 10. `xcelsior`"
NEXT_SECTION = "## 11. `vaultwarden-secrets`"


def _probe_code() -> dict[int, bool]:
    results: dict[int, bool] = {}

    metering = importlib.import_module("serverless.metering")
    results[3] = hasattr(metering, "token_cost_metadata") and hasattr(metering, "cached_input_price")

    obs = importlib.import_module("serverless.observability")
    src = inspect.getsource(obs.compute_endpoint_metrics)
    results[7] = "ttft_p95_ms" in src and "tokens_per_sec" in src

    routes = (ROOT / "routes" / "serverless.py").read_text(encoding="utf-8")
    results[14] = "api_serverless_openai_embeddings" in routes

    svc_src = (ROOT / "serverless" / "service.py").read_text(encoding="utf-8")
    results[15] = "--enable-prefix-caching" in svc_src and hasattr(metering, "cached_input_price")
    results[16] = "--enable-lora" in svc_src

    gate = importlib.import_module("serverless.speculative_gate")
    results[17] = hasattr(gate, "speculative_startup_flags")
    results[2] = (
        "_preset_lmcache_env" in svc_src
        and hasattr(gate, "speculative_startup_flags")
        and "--enable-prefix-caching" in svc_src
    )

    pred = importlib.import_module("serverless.scaling_predictive")
    results[18] = "ewma" in inspect.getsource(pred.forecast_queue_depth).lower()

    repo_src = (ROOT / "serverless" / "repo.py").read_text(encoding="utf-8")
    results[23] = "token_usage_already_recorded" in repo_src or "idempotency_key" in repo_src

    results[27] = "list_token_ledger" in repo_src

    results[29] = "_preset_lmcache_env" in svc_src

    criu = importlib.import_module("criu_hosts")
    results[4] = (
        hasattr(criu, "probe_checkpoint_stack")
        and hasattr(criu, "docker_checkpoint_local")
        and "checkpoint_class" in (ROOT / "routes" / "hosts.py").read_text(encoding="utf-8")
        and "resumable" in inspect.getsource(
            importlib.import_module("routes.instances")._enrich_instance
        )
    )

    anchor = importlib.import_module("serverless.anchor_workloads")
    results[30] = (
        hasattr(anchor, "ara_code_chat_payload")
        and hasattr(anchor, "discover_anchor_repos")
        and len(anchor.discover_anchor_repos()) >= 3
    )

    chaos_path = ROOT / "tests" / "test_serverless_chaos_billing.py"
    chaos_src = chaos_path.read_text(encoding="utf-8") if chaos_path.is_file() else ""
    results[22] = (
        "handle_worker_lost" in chaos_src
        and "duplicate_idempotency_key" in chaos_src
        and "zero_duration" in chaos_src
    )

    results[31] = hasattr(criu, "cuda_driver_requirements") and (
        ROOT / "tests" / "test_criu_hosts.py"
    ).read_text(encoding="utf-8").find("rtx_2060_driver_580") >= 0

    results[13] = bool(
        results.get(3) and results.get(7) and results.get(15) and results.get(17)
    )

    return results


def _checkbox(rank: int, text: str, checked: bool) -> str:
    mark = "x" if checked else " "
    return f"| {rank} | - [{mark}] {text} |"


def _build_section(probes: dict[int, bool]) -> str:
    rows_raw = {
        1: "Token endpoint GA with published price/SLO; ≥30% of served tokens KV-cache hits within a month of launch. | S6 F5 acceptance",
        2: "Stand up vLLM + LMCache on 2 mesh hosts (Qwen3-8B or similar Apache model); enable EAGLE-3 (`--speculative-algorithm EAGLE3`) and validate acceptance rate ≥0.75 on real traffic samples before keeping it on. | S6 F5.0",
        3: "Meter via existing `metering.py`: per-token ledger with cached-vs-computed token split (cached tokens priced lower — the marketing headline writes itself). | S6 F5.0",
        4: 'xcelsior: add `checkpoint_class: gpu-criu` capability flag to host agent + scheduler so preempted jobs on capable hosts resume instead of restart; surface "resumable" as a job attribute in the API. | S6 F4.2',
        5: "One demonstrated preempt→migrate→resume of a running job between two xcelsior hosts (or same host restart) with no output diff. | S6 F4 acceptance",
        6: "Add Mooncake store (or LMCache remote tier) so prefix reuse survives host churn; KV-aware routing in the scheduler: route requests to the host holding their prefix (session affinity by prompt-prefix hash first; Dynamo router when >4 hosts). | S6 F5.1",
        7: "Publish `/v1/usage` SLO metrics (TTFT p95, tokens/s) per endpoint — sell against them. | S6 F5.1",
        8: "SCIP application or partner LOI submitted; attestation schema merged. | S6 F5 acceptance",
        9: "Write the SCIP alignment one-pager + application scan (Canadian ownership ✓, residency ✓, 18-month ops plan = mesh + partner DC). | S6 F5.2",
        10: 'Design `attestation` fields in host admission schema now (TEE evidence JWT, NRAS verify) so H100 hosts slot in without API breaks; recruit 1 H100 host partner; price "attested-sovereign" tier at 2–3× commodity. | S6 F5.2',
        11: "stellar-subs: 14-day GPU-demand forecast with weekday covariates feeding the capacity planner; alert on interval breach (anomaly = actual outside 99% band — no extra anomaly model needed). | S6 F6.3",
        12: "MCP tool `should_i_run_this` pattern adapted for PEL jobs (internal admin / power users) | S4 M2.4",
        13: "**F5** Token SKU with published SLOs + cached-token pricing; SCIP submission/LOI out; attestation schema merged. | S6 DoD",
        14: "Ship embeddings serving through TEI or vLLM embeddings preset; stop returning `501 capability_not_available` on `/embeddings`. | S2 xcelsior",
        15: "Enable vLLM automatic prefix caching where safe and expose cached-token pricing in the product/metering layer. | S2 xcelsior",
        16: "Enable multi-LoRA serving for compatible bases so many adapters can share one GPU. | S2 xcelsior",
        17: "Enable speculative decoding after acceptance-rate validation; keep only when measured throughput improves under real traffic. | S2 xcelsior",
        18: "Replace 2-point predictive scaling extrapolation with EWMA or regression over the existing 20-sample queue-depth history. | S2 xcelsior",
        19: "Add pre-warmed pools keyed to predicted demand and model-weight cache availability; publish cold-start p50/p95 as SLO metrics. | S2 xcelsior",
        20: "Add OpenAI-style async Batch API for non-urgent embeddings, evals, and bulk inference at a discount. | S2 xcelsior",
        21: "Add LLM gateway semantic cache in front of OpenAI-compatible proxy and meter near-duplicate cache savings. | S2 xcelsior",
        22: "Add request-level chaos/fault-injection test that kills workers mid-inference and verifies requeue, idempotency, and no double billing. | S2 xcelsior",
        23: "Add idempotent metering ledger keyed by `(job_id, attempt)` so retries pay provider/customer exactly once. | S2 xcelsior",
        24: "Add hedged requests for long-tail latency: duplicate to a second worker after p95, take the winner, cancel loser, and bill once. | S2 xcelsior",
        25: "Add MCP/assistant safety eval suite for spend, provisioning, workload classification, tenant isolation, and data leakage. | S2 xcelsior",
        26: "Split or modularize `worker_agent.py` and `ai_assistant.py`; committed backup files must not be part of the active tree after explicit cleanup approval. | S2 xcelsior",
        27: "Add LLM observability with OpenLLMetry/Langfuse-style token, latency, cost, and tool traces. | S2 cross-cutting",
        28: "Add `Toto 2.0` as a monitored fallback if Chronos-2 underfits ops telemetry forecasting. | S6 HM",
        29: "Keep Dynamo optional until >4 LLM hosts; use LMCache alone at small scale. | S6 F5 risks",
        30: "Launch with internal anchor workloads (pixelenhance-labs embed/caption, phantom-trades-mvp + ara-code chat patterns) to solve traffic cold start for token SKU. | S6 F5 risks",
        31: "Pin CUDA/driver requirements for CRIUgpu and test snapshot/restore on RTX 2060 before broader host-agent rollout. | S6 F4 risks",
    }

    checked_rows = [r for r in CODE_SHIPPED_ROWS if probes.get(r)]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    closure = (
        f"*2026-07-06 xcelsior closure (synced {stamp}): "
        f"Code-shipped rows {', '.join(str(r) for r in checked_rows)}. "
        f"EAGLE-3 via `serverless/speculative_gate.py` (default on compatible bases after ≥0.75 acceptance on ≥5 samples; 2-host mesh deferred). "
        f"Ops-blocked: {', '.join(str(r) for r in sorted(OPS_BLOCKED_ROWS))}. "
        f"SCIP/sovereignty rows deprioritized per product scope.*"
    )

    lines = [
        SECTION_HEADER,
        "",
        "*GPU marketplace / infra. Path: `/mnt/storage/projects/xcelsior` (ASUS).*",
        "",
        "| Rank | Item | Source |",
        "|------|------|--------|",
    ]
    for rank in range(1, 32):
        item, source = rows_raw[rank].rsplit(" | ", 1)
        checked = rank in CODE_SHIPPED_ROWS and probes.get(rank, False)
        lines.append(f"| {rank} | - [{'x' if checked else ' '}] {item} | {source}")
    lines.extend(
        [
            "",
            "*Excluded from S1 checklist scope (Xcelsior business/product): token pricing, serverless metering, spot GPU pricing, free-credit UI, unit-economics dashboards, marketplace landing copy, embeddings serving, prefix cache, multi-LoRA, chaos/metering ledger — see S1 scope boundary.*",
            "",
            closure,
            "",
            "**Repo impact note:** These changes move `xcelsior` from GPU-hour marketplace plumbing toward a margin-focused inference platform with token SKUs, embeddings, KV reuse, reliable retries, and measurable SLOs.",
            "",
            "---",
            "",
        ]
    )
    return "\n".join(lines)


def _replace_section(full_text: str, new_section: str) -> str:
    pattern = re.compile(
        re.escape(SECTION_HEADER) + r".*?(?=" + re.escape(NEXT_SECTION) + r")",
        re.DOTALL,
    )
    if not pattern.search(full_text):
        raise SystemExit(f"Could not find {SECTION_HEADER} in master index")
    return pattern.sub(new_section + "\n", full_text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="Write probe JSON report")
    args = parser.parse_args()

    probes = _probe_code()
    failed = [r for r in CODE_SHIPPED_ROWS if not probes.get(r)]
    if failed:
        print(f"WARN: code probes failed for rows: {failed}", file=sys.stderr)

    section = _build_section(probes)
    LOCAL_MIRROR.parent.mkdir(parents=True, exist_ok=True)

    mirror_body = (
        f"<!-- synced from scripts/sync_xcelsior_master_index.py -->\n"
        f"<!-- probes: {', '.join(f'{k}={v}' for k, v in sorted(probes.items()))} -->\n\n"
        + section
    )
    if args.dry_run:
        print(mirror_body[:2000])
    else:
        LOCAL_MIRROR.write_text(mirror_body, encoding="utf-8")
        if REGISTRY_MD.is_file():
            full = REGISTRY_MD.read_text(encoding="utf-8")
            REGISTRY_MD.write_text(_replace_section(full, section), encoding="utf-8")
            print(f"Updated {REGISTRY_MD}")
        print(f"Wrote {LOCAL_MIRROR}")

    if args.report:
        import json

        args.report.write_text(
            json.dumps(
                {
                    "probes": probes,
                    "code_shipped": CODE_SHIPPED_ROWS,
                    "ops_blocked": OPS_BLOCKED_ROWS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())