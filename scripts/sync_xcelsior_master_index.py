#!/usr/bin/env python3
"""Sync §10 xcelsior checkboxes from row-evidence.json (evidence-driven, not grep probes)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY_MD = ROOT.parent / "pxl-registry" / "docs" / "AI_ML_MASTER_INDEX_BY_REPO.md"
LOCAL_MIRROR = ROOT / "docs" / "AI_ML_MASTER_INDEX_XCELSIOR.md"
PXL_MIRROR_IN_XCELSIOR = ROOT / "docs" / "pxl_registry_AI_ML_MASTER_INDEX_BY_REPO.md"
DEFAULT_EVIDENCE = Path(
    __import__("os").environ.get(
        "XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer"
    )
) / "row-evidence.json"

SECTION_HEADER = "## 10. `xcelsior`"
NEXT_SECTION = "## 11. `vaultwarden-secrets`"


def _load_evidence(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing row evidence: {path} — run scripts/capture_row_evidence.py first")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("rows"), dict):
        raise SystemExit(f"Invalid row-evidence.json at {path}")
    return data


def _checked_rows(evidence: dict) -> dict[int, bool]:
    rows = evidence.get("rows") or {}
    out: dict[int, bool] = {}
    for rank in range(1, 32):
        row = rows.get(str(rank)) or {}
        out[rank] = row.get("status") == "verified"
    return out


def _build_section(checked: dict[int, bool], evidence: dict) -> str:
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

    verified = sorted(r for r, ok in checked.items() if ok)
    blocked = sorted(r for r, ok in checked.items() if not ok)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ev_rows = evidence.get("rows") or {}
    blocked_notes = "; ".join(
        f"row {r}: {(ev_rows.get(str(r)) or {}).get('reason', '')}" for r in blocked[:8]
    )
    if len(blocked) > 8:
        blocked_notes += f"; …+{len(blocked) - 8} more"

    policy = evidence.get("closure_policy") or "evidence_driven"
    eng_rows = evidence.get("engineering_shipped_rows") or []
    eng_note = ""
    if eng_rows:
        tiers = {
            r: (ev_rows.get(str(r)) or {}).get("evidence_tier", "")
            for r in eng_rows
        }
        eng_note = (
            f" Engineering shipped (unchecked): "
            + "; ".join(f"row {r}={tiers.get(r, '')}" for r in eng_rows)
            + "."
        )
    closure = (
        f"*xcelsior closure ({policy}, synced {stamp}): "
        f"[x] rows: {', '.join(str(r) for r in verified) or '(none)'}. "
        f"[ ] rows: {', '.join(str(r) for r in blocked) or '(none)'}. "
        f"{blocked_notes}. "
        f"Source: row-evidence.json. Billing gates unchanged.{eng_note}*"
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
        mark = "x" if checked.get(rank) else " "
        lines.append(f"| {rank} | - [{mark}] {item} | {source}")
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
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="Write sync report JSON")
    args = parser.parse_args()

    evidence = _load_evidence(args.evidence)
    checked = _checked_rows(evidence)
    section = _build_section(checked, evidence)
    policy = evidence.get("closure_policy") or "evidence_driven"
    LOCAL_MIRROR.parent.mkdir(parents=True, exist_ok=True)

    mirror_body = (
        f"<!-- synced from scripts/sync_xcelsior_master_index.py ({policy}) -->\n"
        f"<!-- evidence: {args.evidence} -->\n\n"
        + section
    )
    if args.dry_run:
        print(mirror_body[:2500])
    else:
        LOCAL_MIRROR.write_text(mirror_body, encoding="utf-8")
        if REGISTRY_MD.is_file():
            full = REGISTRY_MD.read_text(encoding="utf-8")
            REGISTRY_MD.write_text(_replace_section(full, section), encoding="utf-8")
            PXL_MIRROR_IN_XCELSIOR.write_text(
                _replace_section(
                    PXL_MIRROR_IN_XCELSIOR.read_text(encoding="utf-8")
                    if PXL_MIRROR_IN_XCELSIOR.is_file()
                    else full,
                    section,
                ),
                encoding="utf-8",
            )
            print(f"Updated {REGISTRY_MD}")
        print(f"Wrote {LOCAL_MIRROR}")

    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "checked": checked,
                    "verified": evidence.get("verified_rows"),
                    "blocked": evidence.get("blocked_rows"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    verified_count = sum(1 for v in checked.values() if v)
    print(f"§10: {verified_count}/31 verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())