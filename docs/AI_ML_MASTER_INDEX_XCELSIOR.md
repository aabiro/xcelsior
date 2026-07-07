<!-- synced from scripts/sync_xcelsior_master_index.py (engineering_partial) -->
<!-- evidence: /tmp/grok-goal-6f86c7cfe9c2/implementer/row-evidence.json -->

## 10. `xcelsior`

*GPU marketplace / infra. Path: `/mnt/storage/projects/xcelsior` (ASUS).*

| Rank | Item | Source |
|------|------|--------|
| 1 | - [ ] Token endpoint GA with published price/SLO; ≥30% of served tokens KV-cache hits within a month of launch. | S6 F5 acceptance
| 2 | - [ ] Stand up vLLM + LMCache on 2 mesh hosts (Qwen3-8B or similar Apache model); enable EAGLE-3 (`--speculative-algorithm EAGLE3`) and validate acceptance rate ≥0.75 on real traffic samples before keeping it on. | S6 F5.0
| 3 | - [x] Meter via existing `metering.py`: per-token ledger with cached-vs-computed token split (cached tokens priced lower — the marketing headline writes itself). | S6 F5.0
| 4 | - [x] xcelsior: add `checkpoint_class: gpu-criu` capability flag to host agent + scheduler so preempted jobs on capable hosts resume instead of restart; surface "resumable" as a job attribute in the API. | S6 F4.2
| 5 | - [ ] One demonstrated preempt→migrate→resume of a running job between two xcelsior hosts (or same host restart) with no output diff. | S6 F4 acceptance
| 6 | - [x] Add Mooncake store (or LMCache remote tier) so prefix reuse survives host churn; KV-aware routing in the scheduler: route requests to the host holding their prefix (session affinity by prompt-prefix hash first; Dynamo router when >4 hosts). | S6 F5.1
| 7 | - [x] Publish `/v1/usage` SLO metrics (TTFT p95, tokens/s) per endpoint — sell against them. | S6 F5.1
| 8 | - [x] SCIP application or partner LOI submitted; attestation schema merged. | S6 F5 acceptance
| 9 | - [x] Write the SCIP alignment one-pager + application scan (Canadian ownership ✓, residency ✓, 18-month ops plan = mesh + partner DC). | S6 F5.2
| 10 | - [x] Design `attestation` fields in host admission schema now (TEE evidence JWT, NRAS verify) so H100 hosts slot in without API breaks; recruit 1 H100 host partner; price "attested-sovereign" tier at 2–3× commodity. | S6 F5.2
| 11 | - [x] stellar-subs: 14-day GPU-demand forecast with weekday covariates feeding the capacity planner; alert on interval breach (anomaly = actual outside 99% band — no extra anomaly model needed). | S6 F6.3
| 12 | - [x] MCP tool `should_i_run_this` pattern adapted for PEL jobs (internal admin / power users) | S4 M2.4
| 13 | - [x] **F5** Token SKU with published SLOs + cached-token pricing; SCIP submission/LOI out; attestation schema merged. | S6 DoD
| 14 | - [x] Ship embeddings serving through TEI or vLLM embeddings preset; stop returning `501 capability_not_available` on `/embeddings`. | S2 xcelsior
| 15 | - [x] Enable vLLM automatic prefix caching where safe and expose cached-token pricing in the product/metering layer. | S2 xcelsior
| 16 | - [x] Enable multi-LoRA serving for compatible bases so many adapters can share one GPU. | S2 xcelsior
| 17 | - [x] Enable speculative decoding after acceptance-rate validation; keep only when measured throughput improves under real traffic. | S2 xcelsior
| 18 | - [x] Replace 2-point predictive scaling extrapolation with EWMA or regression over the existing 20-sample queue-depth history. | S2 xcelsior
| 19 | - [x] Add pre-warmed pools keyed to predicted demand and model-weight cache availability; publish cold-start p50/p95 as SLO metrics. | S2 xcelsior
| 20 | - [x] Add OpenAI-style async Batch API for non-urgent embeddings, evals, and bulk inference at a discount. | S2 xcelsior
| 21 | - [x] Add LLM gateway semantic cache in front of OpenAI-compatible proxy and meter near-duplicate cache savings. | S2 xcelsior
| 22 | - [x] Add request-level chaos/fault-injection test that kills workers mid-inference and verifies requeue, idempotency, and no double billing. | S2 xcelsior
| 23 | - [x] Add idempotent metering ledger keyed by `(job_id, attempt)` so retries pay provider/customer exactly once. | S2 xcelsior
| 24 | - [x] Add hedged requests for long-tail latency: duplicate to a second worker after p95, take the winner, cancel loser, and bill once. | S2 xcelsior
| 25 | - [x] Add MCP/assistant safety eval suite for spend, provisioning, workload classification, tenant isolation, and data leakage. | S2 xcelsior
| 26 | - [x] Split or modularize `worker_agent.py` and `ai_assistant.py`; committed backup files must not be part of the active tree after explicit cleanup approval. | S2 xcelsior
| 27 | - [x] Add LLM observability with OpenLLMetry/Langfuse-style token, latency, cost, and tool traces. | S2 cross-cutting
| 28 | - [x] Add `Toto 2.0` as a monitored fallback if Chronos-2 underfits ops telemetry forecasting. | S6 HM
| 29 | - [x] Keep Dynamo optional until >4 LLM hosts; use LMCache alone at small scale. | S6 F5 risks
| 30 | - [ ] Launch with internal anchor workloads (pixelenhance-labs embed/caption, phantom-trades-mvp + ara-code chat patterns) to solve traffic cold start for token SKU. | S6 F5 risks
| 31 | - [x] Pin CUDA/driver requirements for CRIUgpu and test snapshot/restore on RTX 2060 before broader host-agent rollout. | S6 F4 risks

*Excluded from S1 checklist scope (Xcelsior business/product): token pricing, serverless metering, spot GPU pricing, free-credit UI, unit-economics dashboards, marketplace landing copy, embeddings serving, prefix cache, multi-LoRA, chaos/metering ledger — see S1 scope boundary.*

*xcelsior closure (engineering_partial, synced 2026-07-06T15:24:15Z): [x] rows: 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31. [ ] rows: 1, 2, 5, 30. row 1: engineering_partial: openai_proxy_served_traffic — proxy→live vLLM KPI 83.33% in harness; 1-month production soak ≥30% pending; row 2: engineering_partial: live_eagle_validated_gated_off — Qwen3-4B-AWQ+EAGLE accept ~0.53 on live HTTP; kept off (<0.75); Qwen3-8B 2-host mesh pending (RTX 2060 6GB); row 5: engineering_partial: live_criu_process_demo — same-host process CRIU ok; cross-host docker serverless migrate pending; row 30: engineering_partial: anchor_workloads_proxy_ledger — 2 local + 2 Mac-SSH builders with ledger rows; sustained production anchor traffic pending. Source: row-evidence.json. Billing gates unchanged. Engineering shipped (unchecked): row 1=openai_proxy_served_traffic; row 2=live_eagle_validated_gated_off; row 5=live_criu_process_demo; row 30=anchor_workloads_proxy_ledger.*

**Repo impact note:** These changes move `xcelsior` from GPU-hour marketplace plumbing toward a margin-focused inference platform with token SKUs, embeddings, KV reuse, reliable retries, and measurable SLOs.

---
